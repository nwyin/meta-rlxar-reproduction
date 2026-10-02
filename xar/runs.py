"""Run directories: the manifest, resume checks, the dry-run cost estimate and operational_summary.json."""

from __future__ import annotations

import json
import math
import statistics
import subprocess
from dataclasses import asdict
from pathlib import Path

from xar.data import task_data
from xar.openrouter import OpenRouter, endpoint_for, highest_prices, token_count
from xar.util import (
    FAILING_FEEDBACK,
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
    """Count invalid structured replies per role and all API sends, and save operational_summary.json."""
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
        # "uncertain": the request was sent but no response was recorded.
        "unresolved_transport": send_statuses.count("uncertain"),
    }
    write_json(output / "operational_summary.json", summary)
    return summary


# This may change on --resume, because it does not change the results; every other argument
# must match the saved run.
RESUMABLE_ARGS = {"concurrency"}
# Manifest keys added after substantive_hash is computed.
UNHASHED_KEYS = {"substantive_hash", "created_at", "git_commit"}


def _hashed_view(manifest):
    """The part of a manifest that substantive_hash covers: all but RESUMABLE_ARGS and UNHASHED_KEYS."""
    view = {key: value for key, value in manifest.items() if key not in UNHASHED_KEYS}
    view["arguments"] = {k: v for k, v in manifest["arguments"].items() if k not in RESUMABLE_ARGS}
    return view


def _changed_settings(saved, manifest):
    """The hashed manifest fields that differ, e.g. ["arguments.iterations (saved 7, now 6)", "roles"]."""
    old, new = _hashed_view(saved), _hashed_view(manifest)
    changed = []
    for key in sorted(old.keys() | new.keys()):
        if old.get(key) == new.get(key):
            continue
        if key not in ("arguments", "software_hashes"):
            changed.append(key)
            continue
        old_items, new_items = old.get(key) or {}, new.get(key) or {}
        for name in sorted(old_items.keys() | new_items.keys()):
            before, after = old_items.get(name), new_items.get(name)
            if before == after:
                continue
            # Code hashes are long and say nothing more than the file name.
            values = f" (saved {before!r}, now {after!r})" if key == "arguments" else ""
            changed.append(f"{key}.{name}{values}")
    return changed


def resolved_manifest(settings, roles, experiment, extra=None):
    """The run's manifest: settings, role settings, pinned endpoints and hashes of data and code."""
    arguments = {k: v for k, v in asdict(settings).items() if k not in {"dry_run", "resume", "output_dir"}}
    endpoints = {}
    for role, cfg in roles.items():
        endpoint, catalog = endpoint_for(cfg)
        endpoints[role] = {"endpoint": endpoint, "canonical_slug": catalog["canonical_slug"]}
    manifest = {
        "experiment": experiment,
        "arguments": arguments,
        "roles": roles,
        "endpoints": endpoints,
        "dataset_hash": file_hash(settings.dataset),
        "splits_hash": file_hash(settings.splits),
        "dataset": str(Path(settings.dataset).resolve()),
        "splits": str(Path(settings.splits).resolve()),
        "software_hashes": software_hashes(),
        "schema_version": SCHEMA_VERSION,
        "extra": extra or {},
    }
    manifest["substantive_hash"] = digest(_hashed_view(manifest))
    return manifest


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        return None


def initialize_run(settings, roles, experiment, extra=None):
    """Check any saved run, run the live preflight, and return the API client.

    A new run gets its manifest written here. A resumed run must match the saved manifest except
    for RESUMABLE_ARGS.
    """
    manifest = resolved_manifest(settings, roles, experiment, extra)
    output = Path(settings.output_dir)
    manifest_path = output / "manifest.json"
    saved = read_json(manifest_path) if manifest_path.exists() else None
    # Where a new run goes: the run name in configs/experiments.yaml under --runs-root.
    elsewhere = "change the run name in configs/experiments.yaml or pass a different --runs-root"
    # Check for an existing run before any network call or write.
    if saved is not None:
        if not settings.resume:
            raise RunError(f"Run exists in {output}; pass --resume to continue it, or {elsewhere}")
        if saved["substantive_hash"] != manifest["substantive_hash"]:
            changed = ", ".join(_changed_settings(saved, manifest)) or "the settings hash"
            raise RunError(
                f"Settings differ from the saved run in {output}: {changed}. "
                f"Revert them to resume, or {elsewhere} to start a new run"
            )
    api = OpenRouter(output, roles, settings.seed)
    api.preflight()
    if saved is None:
        manifest["created_at"] = now()
        manifest["git_commit"] = _git_commit()
        write_json(manifest_path, manifest, write_once=True)
    return api


# Rough numbers for the dry-run estimate. Live requests are priced exactly by max_request_cost.
BYTES_PER_TOKEN = 3.5  # typical for English prose
PROMPT_OVERHEAD_TOKENS = 1500  # instructions and JSON around the paper text
TYPICAL_OUTPUT_TOKENS = 5000
WORST_CASE_EXTRA_TOKENS = 12000  # added to the paper's byte count, which already over-counts tokens
CONTEXT_HEADROOM = 1.25
CONTEXT_EXTRA_TOKENS = 16000  # room for the rubric or meta prompt sent with the paper
FEEDBACK_EXAMPLE_EXTRA_TOKENS = 3000  # a 1,000-word rubric and two grades, per failure
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


def _check_feedback_fits(settings, cfg, endpoint, examples):
    """The optimizer's largest request under FAILING_FEEDBACK: the failure_examples longest
    sections, each with the author's and the model's version (taken as equally long), a rubric and
    two grades. The legacy policy, with a paper per failure, was sized when the corpus was built."""
    if settings.feedback_policy != FAILING_FEEDBACK:
        return
    lengths = sorted((token_count(e["reference"], cfg["model"]) for e in examples), reverse=True)
    failures = min(settings.failure_examples, len(lengths))
    section_tokens = math.ceil(CONTEXT_HEADROOM * 2 * sum(lengths[:failures]))
    prompt_tokens = section_tokens + failures * FEEDBACK_EXAMPLE_EXTRA_TOKENS + CONTEXT_EXTRA_TOKENS
    if prompt_tokens + cfg["max_tokens"] > endpoint["context_length"]:
        raise RunError(
            f"optimizer: {failures} failures may need {prompt_tokens} prompt + {cfg['max_tokens']} output "
            f"tokens, more than the {endpoint['context_length']}-token context of {cfg['provider']}; "
            "lower failure_examples"
        )


def estimate(settings, roles, examples, counts):
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
        if role == "optimizer" and count:
            _check_feedback_fits(settings, cfg, endpoint, examples)
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
    }
    print(json.dumps(result, indent=2))
    return result
