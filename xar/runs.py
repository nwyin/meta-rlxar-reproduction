"""Run directories: the manifest, resume checks, the dry-run estimate and the operational summary."""

from __future__ import annotations

import json
import math
import statistics
import subprocess
from pathlib import Path

from xar.data import load_examples, run_examples, task_data
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
    """SHA-256 of every file that can change what a run does: code, prompts, lockfile, tokenizer list."""
    paths = (
        list(ROOT.glob("*.py"))
        + list((ROOT / "xar").glob("*.py"))
        + list((ROOT / "prompts").glob("*.md"))
        + [ROOT / "uv.lock", ROOT / "configs/tokenizers.json"]
    )
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def operational_summary(output):
    """Count invalid structured replies per role and all API sends; save operational_summary.json."""
    output = Path(output)
    records = []  # (role, structured-output record)
    for path in (output / "rubrics").rglob("rubric.json"):
        records.append(("rubric", read_json(path)))
    for path in (output / "scores").rglob("*.json"):
        if path.name in ("human.json", "model.json"):
            records.append(("judge", read_json(path)))
    for path in (output / "feedback").glob("*/proposal.json"):
        # Each optimizer attempt is itself a structured-output record with its own attempts.
        for proposal_attempt in read_json(path)["attempts"]:
            records.append(("optimizer", proposal_attempt))

    format_validation = {}
    for role, record in records:
        counts = format_validation.setdefault(role, {"structured_attempts": 0, "invalid_attempts": 0})
        for attempt in record["attempts"]:
            counts["structured_attempts"] += 1
            if attempt["status"] == "invalid":
                counts["invalid_attempts"] += 1
    for counts in format_validation.values():
        counts["invalid_fraction"] = counts["invalid_attempts"] / counts["structured_attempts"]

    send_statuses = [read_json(path)["status"] for path in (output / "requests").glob("*/attempt_*.json")]
    summary = {
        "format_validation": format_validation,
        "transport_attempts": len(send_statuses),
        "unresolved_transport": send_statuses.count("uncertain"),
    }
    write_json(output / "operational_summary.json", summary)
    return summary


# Budgets may change when a run is resumed; every other argument must match the saved run.
BUDGET_ARGS = {"budget_usd", "total_budget_usd"}
# Manifest keys added after substantive_hash is computed.
UNHASHED_KEYS = {"substantive_hash", "created_at", "git_commit"}


def _hashed_view(manifest):
    """The part of a manifest that substantive_hash covers: all but the budgets, created_at and git_commit."""
    view = {key: value for key, value in manifest.items() if key not in UNHASHED_KEYS}
    view["arguments"] = {k: v for k, v in manifest["arguments"].items() if k not in BUDGET_ARGS}
    return view


def _changed_settings(saved, manifest):
    """Names of the hashed manifest fields that differ, e.g. ["arguments.iterations", "roles"]."""
    old, new = _hashed_view(saved), _hashed_view(manifest)
    changed = []
    for key in sorted(old.keys() | new.keys()):
        if old.get(key) == new.get(key):
            continue
        if key in ("arguments", "software_hashes"):
            old_items, new_items = old.get(key) or {}, new.get(key) or {}
            names = sorted(old_items.keys() | new_items.keys())
            changed += [f"{key}.{name}" for name in names if old_items.get(name) != new_items.get(name)]
        else:
            changed.append(key)
    return changed


def resolved_manifest(args, roles, experiment, extra=None):
    """The run's manifest: arguments, role settings, pinned endpoints and hashes of data and code."""
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
    manifest["substantive_hash"] = digest(_hashed_view(manifest))
    return manifest


def _check_human_review(args):
    """Every paper the run uses must be approved in human_review.json next to the dataset."""
    review_path = Path(args.dataset).parent / "human_review.json"
    if not review_path.exists():
        raise RunError(f"Missing {review_path}; record a review decision for each paper before running")
    review = read_json(review_path)
    if review.get("dataset_hash") != file_hash(args.dataset):
        raise RunError(f"{review_path} was written for a different {args.dataset}; review the data again")
    papers = {e["paper_id"] for e in run_examples(load_examples(args.dataset, args.splits), args.split)}
    decisions = review.get("papers", {})
    pending = sorted(p for p in papers if decisions.get(p, {}).get("decision") != "approved")
    if pending:
        raise RunError(f"Papers {pending} are not approved in {review_path}")


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return None


def _record_budget_change(output, args):
    """Append the new budget limits to budget_continuations.json unless they are already the last entry."""
    path = output / "budget_continuations.json"
    history = read_json(path) if path.exists() else []
    limits = {"budget_usd": args.budget_usd, "total_budget_usd": args.total_budget_usd}
    last = {key: history[-1].get(key) for key in limits} if history else None
    if last != limits:
        history.append({"at": now(), **limits})
        write_json(path, history)


def initialize_run(args, roles, experiment, extra=None):
    """Check the data review and any saved run, run the live preflight, and return the API client.

    A new run gets its manifest written here. A resumed run must match the saved manifest except
    for the budgets; a budget change is logged to budget_continuations.json.
    """
    _check_human_review(args)
    manifest = resolved_manifest(args, roles, experiment, extra)
    output = Path(args.output_dir)
    manifest_path = output / "manifest.json"
    saved = read_json(manifest_path) if manifest_path.exists() else None
    # Check for an existing run before any network call or write.
    if saved is not None:
        if not args.resume:
            raise RunError(f"Run exists in {output}; pass --resume to continue it or use a new --output-dir")
        if saved["substantive_hash"] != manifest["substantive_hash"]:
            changed = ", ".join(_changed_settings(saved, manifest)) or "the settings hash"
            raise RunError(
                f"Settings differ from the saved run in {changed}; use a new --output-dir or revert"
            )
    api = OpenRouter(output, roles, args.seed, args.budget_usd, args.total_budget_usd, args.budget_ledger)
    api.preflight()
    if saved is None:
        manifest["created_at"] = now()
        manifest["git_commit"] = _git_commit()
        write_json(manifest_path, manifest, write_once=True)
    elif any(saved["arguments"].get(key) != getattr(args, key) for key in BUDGET_ARGS):
        _record_budget_change(output, args)
    return api


# Rough numbers for the dry-run estimate. Live requests are priced exactly by max_request_cost.
BYTES_PER_TOKEN = 3.5  # typical for English prose
PROMPT_OVERHEAD_TOKENS = 1500  # instructions and JSON around the paper text
TYPICAL_OUTPUT_TOKENS = 5000
WORST_CASE_EXTRA_TOKENS = 12000  # added to the paper's byte count, which already over-counts tokens
CONTEXT_HEADROOM = 1.25
CONTEXT_EXTRA_TOKENS = 16000  # room for the rubric or meta prompt sent with the paper
RETRY_RESERVE = 0.25


def _check_context_fits(role, cfg, endpoint, examples):
    for e in examples:
        task_tokens = token_count(canonical(task_data(e)), cfg["model"])
        prompt_tokens = math.ceil(CONTEXT_HEADROOM * task_tokens) + CONTEXT_EXTRA_TOKENS
        if prompt_tokens + cfg["max_tokens"] > endpoint["context_length"]:
            raise RunError(
                f"{role}: {e['example_id']} may need {prompt_tokens} prompt + {cfg['max_tokens']} output "
                f"tokens, more than the {endpoint['context_length']}-token context of {cfg['provider']}"
            )


def estimate(args, roles, examples, counts):
    """Print a rough cost estimate and check that every example fits each role's context window.

    counts maps each role to its number of requests. Returns the printed estimate.
    """
    context_bytes = [len(e["context"].encode()) for e in examples]
    typical_prompt_tokens = statistics.mean(context_bytes) / BYTES_PER_TOKEN + PROMPT_OVERHEAD_TOKENS
    worst_prompt_tokens = max(context_bytes) + WORST_CASE_EXTRA_TOKENS
    per_role = {}
    for role, count in counts.items():
        cfg = roles[role]
        endpoint, _ = endpoint_for(cfg)
        if count:
            _check_context_fits(role, cfg, endpoint, examples)
        price = highest_prices(endpoint)
        typical_output_tokens = min(cfg["max_tokens"], TYPICAL_OUTPUT_TOKENS)
        typical_request = (
            typical_prompt_tokens * price["prompt"] + typical_output_tokens * price["completion"]
        )
        worst_request = worst_prompt_tokens * price["prompt"] + cfg["max_tokens"] * price["completion"]
        per_role[role] = {
            "requests": count,
            "typical_usd": count * typical_request,
            "worst_case_usd_per_request": worst_request,
        }
    typical_total = sum(role_estimate["typical_usd"] for role_estimate in per_role.values())
    result = {
        "examples": len(examples),
        "roles": roles,
        "per_role": per_role,
        "estimated_usd_with_retry_reserve": typical_total * (1 + RETRY_RESERVE),
        "budget_usd": args.budget_usd,
        "total_budget_usd": args.total_budget_usd,
    }
    print(json.dumps(result, indent=2))
    return result
