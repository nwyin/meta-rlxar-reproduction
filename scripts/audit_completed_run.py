"""Re-audit the completed Muse/Kimi research run from its saved requests and outputs.

Makes no API calls and writes only the summary numbers to --output. The checks re-derive saved
artifacts with the current code (build_feedback, task_data, audit_proposal, contamination,
check_request_settings, max_request_cost), so a refactor that changes their output fails here.
"""

import argparse
import collections
import hashlib
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run import load_design
from xar.audit import audit_saved_output, audit_xar_run, check_request_settings
from xar.data import contamination, load_examples, run_examples, task_data
from xar.openrouter import ledger_key, max_request_cost
from xar.pipeline import (
    GRADE_SCHEMA,
    PROPOSAL_ATTEMPTS,
    PROPOSAL_SCHEMA,
    RUBRIC_SCHEMA,
    WRITER_ATTEMPTS,
    audit_proposal,
    build_feedback,
    length_revision_note,
    length_window,
)
from xar.stats import paired_improvement, summarize
from xar.util import ROOT, RunError, canonical, digest, now, read_json, words, write_json

DESIGN = load_design()
PILOT_RUN = ROOT / "runs" / DESIGN["pilot_run"]
PILOT_GATE = ROOT / "runs/meta_blog_pilot_gate.json"
# The writer returns plain text, so its requests carry no response_format.
SCHEMAS = {"writer": None, "rubric": RUBRIC_SCHEMA, "judge": GRADE_SCHEMA, "optimizer": PROPOSAL_SCHEMA}
# Row filters for the sensitivity table.
SUBSETS = {
    "all": lambda row: True,
    "length_compliant": lambda row: row["length_compliant"],
    "unflagged": lambda row: not row["contamination_flagged"],
}


def check(ok, message):
    """Like assert, but not stripped by python -O."""
    if not ok:
        raise RunError(message)


def check_software(manifest):
    """Check the manifest's code hashes against the files at the run's git commit.

    The working tree has changed since the run, so the files are read with git show.
    """
    commit = manifest["git_commit"]
    check(commit, "Run manifest has no git commit; cannot verify the code that produced it")
    for name, expected in manifest["software_hashes"].items():
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"])
        check(
            hashlib.sha256(blob).hexdigest() == expected,
            f"{name} at {commit} does not match the run manifest",
        )


def check_human_review(manifest, examples):
    review = read_json(Path(manifest["dataset"]).parent / "human_review.json")
    check(review["dataset_hash"] == manifest["dataset_hash"], "human_review.json is for a different dataset")
    for example in examples:
        decision = review["papers"][example["paper_id"]]["decision"]
        check(decision == "approved", f"{example['paper_id']}: human review decision is {decision!r}")


def check_requests(source, manifest, ledger):
    """Check every saved request and response against the manifest and the budget ledger.

    Returns per-role request counts and latencies, the total cost, the time the last response
    arrived, and (role, identity, attempt_0.json contents) for each request, for the timing checks.
    """
    counts = collections.Counter()
    latencies = collections.defaultdict(list)
    total_cost = 0
    finished = []
    first_sends = []
    for path in source.glob("requests/*/request.json"):
        request = read_json(path)
        role, payload = request["role"], request["payload"]
        cfg, endpoint = manifest["roles"][role], manifest["endpoints"][role]

        # The request key is the hash of what was sent, and names the folder.
        check(request["key"] == path.parent.name, f"{path}: key {request['key']} differs from its folder")
        hashed = {
            "payload": payload,
            "identity": request["identity"],
            "schema_version": manifest["schema_version"],
        }
        check(request["key"] == digest(hashed), f"{path}: key does not match the request contents")

        # Routing, decoding, schema and system prompt match the role.
        prompt_name = "rubric_wrapper" if role == "rubric" else role
        check_request_settings(
            payload, cfg, SCHEMAS[role], manifest["software_hashes"][f"prompts/{prompt_name}.md"], where=path
        )

        # Each send succeeded, came from the right model and provider, and was reserved and
        # settled in the ledger in order, at a cost within its reservation.
        max_cost = max_request_cost(payload, endpoint["endpoint"])
        for attempt_path in path.parent.glob("attempt_*.json"):
            sent = read_json(attempt_path)
            check(sent["status"] == "success", f"{attempt_path}: status is {sent['status']}")
            response = sent["response"]
            check(
                response["model"] in (cfg["model"], endpoint["canonical_slug"]),
                f"{attempt_path}: answered by model {response['model']}",
            )
            check(
                response["provider"] in (cfg["provider"], endpoint["endpoint"]["provider_name"]),
                f"{attempt_path}: answered by provider {response['provider']}",
            )
            finish = response["choices"][0]["finish_reason"]
            check(finish == "stop", f"{attempt_path}: finish reason is {finish}")
            attempt_number = attempt_path.stem.removeprefix("attempt_")
            key = ledger_key(source, request["key"], attempt_number)
            entry = ledger[key]
            cost = response["usage"]["cost"]
            check(entry["state"] == "complete", f"{key}: ledger entry is {entry['state']}")
            check(
                math.isclose(entry["charge"], cost), f"{key}: ledger charge {entry['charge']} != cost {cost}"
            )
            check(
                math.isclose(entry["upper"], max_cost),
                f"{key}: ledger reserved {entry['upper']}, but the maximum request cost is {max_cost}",
            )
            check(cost <= max_cost, f"{key}: cost {cost} exceeds the maximum request cost {max_cost}")
            check(
                entry["created_at"] <= sent["sent_at"] <= sent["timestamp"] <= entry["settled_at"],
                f"{key}: reserve, send, receive and settle times are out of order",
            )
            counts[role] += 1
            latencies[role].append(sent["duration_seconds"])
            total_cost += cost
            finished.append(sent["timestamp"])
        first_sends.append((role, canonical(request["identity"]), read_json(path.parent / "attempt_0.json")))
    return counts, latencies, total_cost, max(finished), first_sends


def check_writer(candidates, examples_by_id):
    """Re-derive each writer candidate from its raw responses; return when each response arrived."""
    finished = []
    for eid, candidate in candidates.items():
        example = examples_by_id[eid]
        attempts = candidate["attempts"]
        check(1 <= len(attempts) <= WRITER_ATTEMPTS, f"{eid}: {len(attempts)} writer attempts")
        check(
            candidate["accepted_attempt"] == len(attempts) - 1, f"{eid}: accepted attempt is not the last one"
        )
        check(
            not any(attempt["length_compliant"] for attempt in attempts[:-1]),
            f"{eid}: writer retried after a section that met the length target",
        )
        check(
            candidate["length_compliant"] or len(attempts) == WRITER_ATTEMPTS,
            f"{eid}: writer stopped before meeting the length target or using every attempt",
        )
        low, high = length_window(example["target_words"])
        for index, attempt in enumerate(attempts):
            where = f"{eid} attempt {index}"
            sent = read_json(attempt["response"]["raw_response"])
            text = sent["response"]["choices"][0]["message"]["content"].strip()
            check(text == attempt["text"], f"{where}: saved text differs from the response")
            check(digest(text) == attempt["text_hash"], f"{where}: text hash differs")
            check(words(text) == attempt["words"], f"{where}: word count differs")
            check(
                attempt["length_compliant"] == (low <= words(text) <= high),
                f"{where}: length_compliant flag is wrong",
            )
            # A retry also sends the previous section and the length revision note.
            request = read_json(Path(attempt["response"]["raw_response"]).parent / "request.json")
            expected_input = task_data(example)
            if index:
                expected_input.update(
                    previous_section=attempts[index - 1]["text"],
                    length_revision=length_revision_note(example["target_words"]),
                )
            check(
                canonical(expected_input) == request["payload"]["messages"][1]["content"],
                f"{where}: writer input differs from the task data",
            )
            finished.append(sent["timestamp"])
        check(candidate["text"] == attempts[-1]["text"], f"{eid}: candidate text is not the last attempt")
        check(candidate["complete"], f"{eid}: candidate is incomplete")
        check(
            candidate["contamination"] == contamination(candidate["text"], example["reference"]),
            f"{eid}: contamination check differs",
        )
    return finished


def check_proposals(source, run, train, candidates):
    """Rebuild each iteration's optimizer feedback and proposal audit; return one summary per proposal."""
    run_args = run["manifest"]["arguments"]
    max_words = run_args["max_meta_prompt_words"]
    initial = (source / "prompts/iter_00.md").read_text()
    proposals = []
    for iteration in range(1, run_args["iterations"] + 1):
        where = f"iteration {iteration}"
        previous = (source / f"prompts/iter_{iteration - 1:02d}.md").read_text()
        rows = read_json(source / f"scores/main/{iteration - 1}/train_rows.json")
        expected_feedback = build_feedback(train, candidates, rows, previous, run_args["failure_examples"])
        directory = source / f"feedback/iter_{iteration:02d}"
        feedback = read_json(directory / "training.json")
        proposal = read_json(directory / "proposal.json")
        check(canonical(expected_feedback) == canonical(feedback), f"{where}: feedback differs")
        check(proposal["parent_prompt_hash"] == digest(previous), f"{where}: parent prompt differs")
        check(proposal["feedback_hash"] == digest(feedback), f"{where}: feedback hash differs")
        check(proposal["update_consumed"], f"{where}: update not consumed")

        attempts = proposal["attempts"]
        check(1 <= len(attempts) <= PROPOSAL_ATTEMPTS, f"{where}: {len(attempts)} attempts")
        expected_input = canonical({"feedback": feedback, "max_meta_prompt_words": max_words})
        for attempt in attempts:
            payload = audit_saved_output(attempt, PROPOSAL_SCHEMA, where=where)
            check(payload["messages"][1]["content"] == expected_input, f"{where}: optimizer input differs")
            check(
                attempt["audit"] == audit_proposal(attempt["value"]["prompt"], train, initial, max_words),
                f"{where}: proposal audit differs",
            )

        # The last attempt's audit decides whether the prompt changes.
        final = attempts[-1]
        check(
            proposal["accepted"] == final["audit"]["accepted"],
            f"{where}: accepted flag differs from the last audit",
        )
        expected_prompt = final["value"]["prompt"] if proposal["accepted"] else previous
        check(
            proposal["prompt"] == expected_prompt, f"{where}: saved prompt is not the one the audit implies"
        )
        check(
            proposal["prompt_hash"] == run["freeze"]["prompt_hashes"][iteration],
            f"{where}: prompt hash differs from freeze.json",
        )
        proposals.append(
            {"iteration": iteration, "accepted": proposal["accepted"], "words": words(proposal["prompt"])}
        )
    return proposals


def check_order(run, writer_finished, first_sends):
    """Check that the phases ran in order: writer, training, freeze.json, then validation."""
    frozen_at = run["freeze"]["frozen_at"]
    first_rubric_sent = min(sent["sent_at"] for role, _, sent in first_sends if role == "rubric")
    scoring = [(identity, sent) for role, identity, sent in first_sends if role != "writer"]
    training_finished = [sent["timestamp"] for identity, sent in scoring if "/validation/" not in identity]
    validation_sent = [sent["sent_at"] for identity, sent in scoring if "/validation/" in identity]
    check(max(writer_finished) <= first_rubric_sent, "A rubric request was sent before the writer finished")
    check(max(training_finished) <= frozen_at, "A training request finished after freeze.json was written")
    check(frozen_at <= min(validation_sent), "A validation request was sent before freeze.json was written")


def sensitivity_table(run):
    """Initial vs selected checkpoint scores on each split, for all rows and two filtered subsets."""
    selected = run["freeze"]["selected"]
    table = {}
    for split in ("train", "validation"):
        for name, keep in SUBSETS.items():
            initial_rows = [row for row in run["rows"][(0, split)] if keep(row)]
            selected_rows = [row for row in run["rows"][(selected, split)] if keep(row)]
            table[f"{split}_{name}"] = {
                "examples": len(selected_rows),
                "initial": summarize(initial_rows),
                "selected": summarize(selected_rows),
                "improvement": paired_improvement(initial_rows, selected_rows),
            }
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default=ROOT / "runs" / DESIGN["research_run"])
    parser.add_argument("--output", default=ROOT / "reports/completion_audit.json")
    args = parser.parse_args()
    source, output = Path(args.source_run).resolve(), Path(args.output).resolve()
    # Saved runs store file paths relative to the repo root, so work from there.
    os.chdir(ROOT)

    # Rubrics, grades, checkpoint table and selection (see audit_xar_run).
    run = audit_xar_run(source)
    manifest = run["manifest"]
    run_args = manifest["arguments"]
    check_software(manifest)

    # The run's examples, their human review, and one writer candidate for each.
    examples = run_examples(load_examples(manifest["dataset"], manifest["splits"]), run_args["split"])
    train = [example for example in examples if example["split"] == "train"]
    examples_by_id = {example["example_id"]: example for example in examples}
    check_human_review(manifest, examples)
    candidates = run["candidates"]
    check(
        set(candidates) == set(examples_by_id),
        "Writer candidates do not match the train and validation examples",
    )

    ledger = read_json(run_args["budget_ledger"])["entries"]
    counts, latencies, total_cost, last_finished, first_sends = check_requests(source, manifest, ledger)
    writer_finished = check_writer(candidates, examples_by_id)
    proposals = check_proposals(source, run, train, candidates)
    check_order(run, writer_finished, first_sends)

    # costs.json and the shared ledger agree with the responses.
    costs = read_json(source / "costs.json")
    check(
        math.isclose(costs["actual_complete_usd"], total_cost),
        f"costs.json total {costs['actual_complete_usd']} != sum of responses {total_cost}",
    )
    check(costs["unresolved"] == 0, f"costs.json has {costs['unresolved']} unresolved requests")
    request_total = sum(counts.values())
    check(
        costs["requests"] == request_total,
        f"costs.json counts {costs['requests']} requests; found {request_total}",
    )
    shared_total = sum(entry["charge"] for entry in ledger.values())
    check(
        shared_total <= run_args["total_budget_usd"],
        f"Ledger total {shared_total} is over the ${run_args['total_budget_usd']} budget",
    )

    # The pilot run passes the same audit, and the gate file names it.
    pilot = audit_xar_run(PILOT_RUN)
    gate = read_json(PILOT_GATE)
    check(
        gate["pilot_substantive_hash"] == pilot["manifest"]["substantive_hash"],
        "Pilot gate file does not match the pilot run",
    )

    wallclock = datetime.fromisoformat(last_finished) - datetime.fromisoformat(manifest["created_at"])
    result = {
        "at": now(),
        "source_run": str(source),
        "proposals": proposals,
        "request_counts": dict(counts),
        "research_cost_usd": total_cost,
        "pilot_cost_usd": read_json(PILOT_RUN / "costs.json")["actual_complete_usd"],
        "shared_charged_or_reserved_usd": shared_total,
        "writer_length_compliant": sum(c["length_compliant"] for c in candidates.values()),
        "writer_contamination_flagged": sum(c["contamination"]["flagged"] for c in candidates.values()),
        "latency_seconds": {
            role: {"median": statistics.median(values), "total": sum(values)}
            for role, values in latencies.items()
        },
        "research_wallclock_seconds": wallclock.total_seconds(),
        "sensitivity": sensitivity_table(run),
    }
    write_json(output, result)
    print(
        "Completion audit passed; research USD:",
        round(total_cost, 2),
        "shared charged/reserved USD:",
        round(shared_total, 2),
    )


if __name__ == "__main__":
    main()
