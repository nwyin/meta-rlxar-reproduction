"""Run directories: the manifest, resume checks, the dry-run estimate and the operational summary."""

from __future__ import annotations

import json
import math
import statistics
import subprocess
from pathlib import Path

from xar.data import load_examples, task_data
from xar.openrouter import OpenRouter, endpoint_for, highest_prices, token_count
from xar.util import (
    ROOT,
    SCHEMA_VERSION,
    RunError,
    canonical,
    digest,
    file_hash,
    now,
    read_json,
    write_json,
)


def software_hashes():
    paths = (
        list(ROOT.glob("*.py"))
        + list((ROOT / "xar").glob("*.py"))
        + list((ROOT / "prompts").glob("*.md"))
        + [ROOT / "uv.lock", ROOT / "configs/tokenizers.json"]
    )
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def operational_summary(output):
    output = Path(output)
    attempts = {}

    def collect(value):
        if isinstance(value, dict):
            if value.get("status") in {"valid", "invalid"} and isinstance(value.get("response"), dict):
                response = value["response"]
                if response.get("request_key"):
                    attempts[response["request_key"]] = value["status"]
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    for folder in ("rubrics", "scores", "feedback"):
        for path in (output / folder).rglob("*.json"):
            collect(read_json(path))
    role_rates = {}
    for key, status in attempts.items():
        role = read_json(output / "requests" / key / "request.json")["role"]
        counts = role_rates.setdefault(role, {"structured_attempts": 0, "invalid_attempts": 0})
        counts["structured_attempts"] += 1
        counts["invalid_attempts"] += status == "invalid"
    for counts in role_rates.values():
        counts["invalid_fraction"] = counts["invalid_attempts"] / counts["structured_attempts"]
    statuses = [read_json(receipt)["status"] for receipt in (output / "requests").glob("*/attempt_*.json")]
    summary = {
        "format_validation": role_rates,
        "transport_attempts": len(statuses),
        "unresolved_transport": statuses.count("uncertain"),
    }
    write_json(output / "operational_summary.json", summary)
    return summary


BUDGET_ARGS = {"budget_usd", "total_budget_usd"}


def resolved_manifest(args, roles, experiment, extra=None):
    arguments = {k: v for k, v in vars(args).items() if k not in {"dry_run", "resume", "output_dir"}}
    endpoints = {}
    for role, cfg in roles.items():
        endpoint, catalog = endpoint_for(cfg)
        endpoints[role] = {"endpoint": endpoint, "canonical_slug": catalog["canonical_slug"]}
    manifest = {
        "experiment": experiment,
        "arguments": arguments,
        "roles": roles,
        "endpoints": endpoints,
        "dataset_hash": file_hash(args.dataset),
        "splits_hash": file_hash(args.splits),
        "dataset": str(Path(args.dataset).resolve()),
        "splits": str(Path(args.splits).resolve()),
        "software_hashes": software_hashes(),
        "schema_version": SCHEMA_VERSION,
        "extra": extra or {},
    }
    # Budgets may change on resume; everything else must match the saved run.
    hashed = {**manifest, "arguments": {k: v for k, v in arguments.items() if k not in BUDGET_ARGS}}
    manifest["substantive_hash"] = digest(hashed)
    return manifest


def initialize_run(args, roles, experiment, extra=None):
    manifest = resolved_manifest(args, roles, experiment, extra)
    out = Path(args.output_dir)
    path = out / "manifest.json"
    review_path = Path(args.dataset).parent / "human_review.json"
    if not review_path.exists():
        raise RunError("PLAN.md requires human extraction/suitability review; fill data/human_review.json")
    review = read_json(review_path)
    if review.get("dataset_hash") != file_hash(args.dataset):
        raise RunError("Human review does not match the frozen dataset hash")
    used_examples = load_examples(args.dataset, args.splits)
    used_papers = {
        e["paper_id"]
        for e in used_examples
        if (e["split"] in ("train", "validation") if args.split == "research" else e["split"] == args.split)
    }
    if any(review.get("papers", {}).get(p, {}).get("decision") != "approved" for p in used_papers):
        raise RunError("Selected papers still need human review; see data/review.md")
    # Check for an existing run before any network call or write.
    saved = read_json(path) if path.exists() else None
    if saved is not None:
        if not args.resume:
            raise RunError("Run exists; use --resume or a new output directory")
        if saved["substantive_hash"] != manifest["substantive_hash"]:
            raise RunError("Resume substantive configuration changed; create a new run")
    api = OpenRouter(out, roles, args.seed, args.budget_usd, args.total_budget_usd, args.budget_ledger)
    api.preflight()
    if saved is not None:
        if (
            saved["arguments"].get("budget_usd") != args.budget_usd
            or saved["arguments"].get("total_budget_usd") != args.total_budget_usd
        ):
            history = (
                read_json(out / "budget_continuations.json")
                if (out / "budget_continuations.json").exists()
                else []
            )
            change = {"budget_usd": args.budget_usd, "total_budget_usd": args.total_budget_usd}
            if not history or any(history[-1].get(k) != v for k, v in change.items()):
                history.append({"at": now(), **change})
                write_json(out / "budget_continuations.json", history)
    else:
        manifest["created_at"] = now()
        try:
            manifest["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        except subprocess.CalledProcessError:
            manifest["git_commit"] = None
        write_json(path, manifest, write_once=True)
    return api


# Rough numbers for the dry-run estimate. Live requests are priced exactly by max_request_cost.
BYTES_PER_TOKEN = 3.5  # typical for English prose
PROMPT_OVERHEAD_TOKENS = 1500  # instructions and JSON around the paper text
TYPICAL_OUTPUT_TOKENS = 5000
WORST_CASE_EXTRA_TOKENS = 12000  # added to the paper's byte count, which already over-counts tokens
CONTEXT_HEADROOM = 1.25
CONTEXT_EXTRA_TOKENS = 16000  # room for the rubric or meta prompt sent with the paper
RETRY_RESERVE = 0.25


def estimate(args, roles, examples, counts):
    """Print a rough cost estimate and check that every example fits each role's context window."""
    contexts = [len(e["context"].encode()) for e in examples]
    estimates = {}
    for role, count in counts.items():
        endpoint, _ = endpoint_for(roles[role])
        if count:
            for e in examples:
                task_tokens = token_count(canonical(task_data(e)), roles[role]["model"])
                tokens = math.ceil(CONTEXT_HEADROOM * task_tokens) + CONTEXT_EXTRA_TOKENS
                if tokens + roles[role]["max_tokens"] > endpoint["context_length"]:
                    raise RunError(f"{role} context does not fit {e['example_id']}")
        price = highest_prices(endpoint)
        typical_tokens = statistics.mean(contexts) / BYTES_PER_TOKEN + PROMPT_OVERHEAD_TOKENS
        output_tokens = min(roles[role]["max_tokens"], TYPICAL_OUTPUT_TOKENS)
        estimates[role] = {
            "requests": count,
            "typical_uncached_usd": count
            * (typical_tokens * price["prompt"] + output_tokens * price["completion"]),
            "per_request_conservative_usd": (max(contexts) + WORST_CASE_EXTRA_TOKENS) * price["prompt"]
            + roles[role]["max_tokens"] * price["completion"],
        }
    typical_total = sum(e["typical_uncached_usd"] for e in estimates.values())
    result = {
        "examples": len(examples),
        "roles": roles,
        "counts_and_costs": estimates,
        "estimated_usd_with_retry_reserve": typical_total * (1 + RETRY_RESERVE),
        "budget_usd": args.budget_usd,
        "total_budget_usd": args.total_budget_usd,
    }
    print(json.dumps(result, indent=2))
    return result
