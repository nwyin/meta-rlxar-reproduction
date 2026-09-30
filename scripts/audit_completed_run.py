"""Re-audit the completed Muse/Kimi research run from its saved requests and outputs.

Makes no API calls and writes only the summary numbers to --output. The checks re-derive saved
artifacts with the current code (build_feedback, task_data, audit_proposal, contamination,
audit_request_contract, request_upper), so a refactor that changes their output fails here.
"""

import argparse
import collections
import hashlib
import math
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exp_xar
import shared as s

WRITER_ATTEMPTS = 3  # shared.writer_candidates tries each section at most three times


def check(ok, message):
    """Like assert, but not stripped by python -O."""
    if not ok:
        raise s.ContractError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default=s.ROOT / "runs/meta-blog-seed0")
    parser.add_argument("--output", default=s.ROOT / "reports/completion_audit.json")
    args = parser.parse_args()
    root = Path(args.source_run)
    runs = s.ROOT / "runs"
    run = s.audit_xar_run(root)
    manifest = run["manifest"]
    run_args = manifest["arguments"]
    # The code has changed since the run; check the recorded hashes against the commit that produced it.
    commit = manifest["git_commit"]
    check(commit, "Run manifest has no git commit; cannot verify the code that produced it")
    for name, expected in manifest["software_hashes"].items():
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=s.ROOT)
        check(
            hashlib.sha256(blob).hexdigest() == expected,
            f"{name} at {commit} does not match the run manifest",
        )
    dataset = s.ROOT / manifest["dataset"]
    all_examples = s.load_examples(dataset, s.ROOT / manifest["splits"])
    examples = [e for e in all_examples if e["split"] in ("train", "validation")]
    train = [e for e in examples if e["split"] == "train"]
    lookup = {e["example_id"]: e for e in examples}
    review = s.read_json(dataset.parent / "human_review.json")
    check(review["dataset_hash"] == manifest["dataset_hash"], "human_review.json is for a different dataset")
    for e in examples:
        decision = review["papers"][e["paper_id"]]["decision"]
        check(decision == "approved", f"{e['paper_id']}: human review decision is {decision!r}")
    candidates = run["candidates"]
    check(set(candidates) == set(lookup), "Writer candidates do not match the train and validation examples")
    ledger = s.read_json(s.ROOT / run_args["budget_ledger"])["entries"]
    counts, latencies, ended = collections.Counter(), collections.defaultdict(list), {}
    schemas = {"rubric": s.RUBRIC_SCHEMA, "judge": s.GRADE_SCHEMA, "optimizer": s.PROPOSAL_SCHEMA}
    total_cost = 0
    for path in root.glob("requests/*/request.json"):
        request = s.read_json(path)
        role, payload = request["role"], request["payload"]
        cfg, endpoint = manifest["roles"][role], manifest["endpoints"][role]
        check(request["key"] == path.parent.name, f"{path}: key {request['key']} differs from its folder")
        hashed = {
            "payload": payload,
            "identity": request["identity"],
            "schema_version": manifest["schema_version"],
        }
        check(request["key"] == s.digest(hashed), f"{path}: key does not match the request contents")
        if role == "writer":
            for key in ("model", "temperature", "reasoning", "max_tokens"):
                check(
                    payload[key] == cfg[key],
                    f"{path}: writer {key} is {payload[key]!r}, expected {cfg[key]!r}",
                )
            routing = {
                "only": [cfg["provider"]],
                "order": [cfg["provider"]],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
            check(payload["provider"] == routing, f"{path}: writer provider routing differs")
            check(payload["plugins"] == [], f"{path}: writer request has plugins")
            check(payload["transforms"] == [], f"{path}: writer request has transforms")
            check(payload["stream"] is False, f"{path}: writer request is streamed")
            check("response_format" not in payload, f"{path}: writer request has a response_format")
            check(
                len(payload["messages"]) == 2,
                f"{path}: writer request has {len(payload['messages'])} messages",
            )
            check(
                s.digest(payload["messages"][0]["content"])
                == manifest["software_hashes"]["prompts/writer.md"],
                f"{path}: writer system prompt differs from prompts/writer.md",
            )
        else:
            wrapper = {"rubric": "rubric_wrapper", "judge": "judge", "optimizer": "optimizer"}[role]
            s.audit_request_contract(
                payload, cfg, endpoint, schemas[role], manifest["software_hashes"][f"prompts/{wrapper}.md"]
            )
        bound = s.request_upper(payload, endpoint["endpoint"])
        for receipt_path in path.parent.glob("attempt_*.json"):
            receipt = s.read_json(receipt_path)
            check(receipt["status"] == "success", f"{receipt_path}: status is {receipt['status']}")
            raw = receipt["response"]
            check(
                raw["model"] in (cfg["model"], endpoint["canonical_slug"]),
                f"{receipt_path}: answered by model {raw['model']}",
            )
            check(
                raw["provider"] in (cfg["provider"], endpoint["endpoint"]["provider_name"]),
                f"{receipt_path}: answered by provider {raw['provider']}",
            )
            finish = raw["choices"][0]["finish_reason"]
            check(finish == "stop", f"{receipt_path}: finish reason is {finish}")
            index = receipt_path.stem.removeprefix("attempt_")
            key = str(root.resolve()) + "/" + request["key"] + "/" + index
            entry = ledger[key]
            cost = raw["usage"]["cost"]
            check(entry["state"] == "complete", f"{key}: ledger entry is {entry['state']}")
            check(
                math.isclose(entry["charge"], cost), f"{key}: ledger charge {entry['charge']} != cost {cost}"
            )
            check(
                math.isclose(entry["upper"], bound),
                f"{key}: ledger reservation {entry['upper']} != bound {bound}",
            )
            check(cost <= bound, f"{key}: cost {cost} is more than its bound {bound}")
            check(
                entry["created_at"] <= receipt["sent_at"] <= receipt["timestamp"] <= entry["settled_at"],
                f"{key}: reserve, send, receive and settle times are out of order",
            )
            counts[role] += 1
            latencies[role].append(receipt["duration_seconds"])
            total_cost += cost
            ended[str(receipt_path)] = receipt["timestamp"]
    writer_times, training_times, validation_times = [], [], []
    for eid, candidate in candidates.items():
        e = lookup[eid]
        attempts = candidate["attempts"]
        check(1 <= len(attempts) <= WRITER_ATTEMPTS, f"{eid}: {len(attempts)} writer attempts")
        check(
            candidate["accepted_attempt"] == len(attempts) - 1, f"{eid}: accepted attempt is not the last one"
        )
        check(
            not any(a["length_compliant"] for a in attempts[:-1]),
            f"{eid}: writer retried after a section that met the length target",
        )
        check(
            candidate["length_compliant"] or len(attempts) == WRITER_ATTEMPTS,
            f"{eid}: writer stopped before meeting the length target or using every attempt",
        )
        for i, attempt in enumerate(attempts):
            receipt = s.read_json(attempt["response"]["raw_response"])
            raw_text = receipt["response"]["choices"][0]["message"]["content"].strip()
            check(raw_text == attempt["text"], f"{eid} attempt {i}: saved text differs from the response")
            check(s.digest(raw_text) == attempt["text_hash"], f"{eid} attempt {i}: text hash differs")
            check(s.words(raw_text) == attempt["words"], f"{eid} attempt {i}: word count differs")
            check(
                attempt["length_compliant"]
                == (0.85 * e["target_words"] <= s.words(raw_text) <= 1.15 * e["target_words"]),
                f"{eid} attempt {i}: length_compliant flag is wrong",
            )
            q = s.read_json(Path(attempt["response"]["raw_response"]).parent / "request.json")
            expected = s.task_data(e)
            if i:
                expected.update(
                    previous_section=attempts[i - 1]["text"],
                    length_revision=f"Revise only to fit {math.ceil(0.85 * e['target_words'])}–{math.floor(1.15 * e['target_words'])} words. Preserve claims.",
                )
            check(
                s.canonical(expected) == q["payload"]["messages"][1]["content"],
                f"{eid} attempt {i}: writer input differs from the task data",
            )
            writer_times.append(receipt["timestamp"])
        check(candidate["text"] == attempts[-1]["text"], f"{eid}: candidate text is not the last attempt")
        check(candidate["complete"], f"{eid}: candidate is incomplete")
        check(
            candidate["contamination"] == s.contamination(candidate["text"], e["reference"]),
            f"{eid}: contamination check differs",
        )
    initial = (root / "prompts/iter_00.md").read_text()
    max_words = run_args["max_meta_prompt_words"]
    proposals = []
    for iteration in range(1, run_args["iterations"] + 1):
        previous = (root / f"prompts/iter_{iteration - 1:02d}.md").read_text()
        rows = s.read_json(root / f"scores/main/{iteration - 1}/train_rows.json")
        expected = exp_xar.build_feedback(train, candidates, rows, previous, run_args["failure_examples"])
        directory = root / f"feedback/iter_{iteration:02d}"
        feedback, proposal = (
            s.read_json(directory / "training.json"),
            s.read_json(directory / "proposal.json"),
        )
        check(s.canonical(expected) == s.canonical(feedback), f"iteration {iteration}: feedback differs")
        check(
            proposal["parent_prompt_hash"] == s.digest(previous),
            f"iteration {iteration}: parent prompt differs",
        )
        check(
            proposal["feedback_hash"] == s.digest(feedback), f"iteration {iteration}: feedback hash differs"
        )
        check(proposal["update_consumed"], f"iteration {iteration}: update not consumed")
        check(
            1 <= len(proposal["attempts"]) <= 2,
            f"iteration {iteration}: {len(proposal['attempts'])} attempts",
        )
        for attempt in proposal["attempts"]:
            payload = s.audit_saved_output(attempt, s.PROPOSAL_SCHEMA)
            check(
                payload["messages"][1]["content"]
                == s.canonical({"feedback": feedback, "max_meta_prompt_words": max_words}),
                f"iteration {iteration}: optimizer input differs",
            )
            check(
                attempt["audit"] == s.audit_proposal(attempt["value"]["prompt"], train, initial, max_words),
                f"iteration {iteration}: proposal audit differs",
            )
        check(
            proposal["accepted"] == proposal["attempts"][-1]["audit"]["accepted"],
            f"iteration {iteration}: accepted flag differs from the last audit",
        )
        check(
            proposal["prompt"]
            == (proposal["attempts"][-1]["value"]["prompt"] if proposal["accepted"] else previous),
            f"iteration {iteration}: saved prompt is not the one the audit implies",
        )
        check(
            proposal["prompt_hash"] == run["freeze"]["prompt_hashes"][iteration],
            f"iteration {iteration}: prompt hash differs from freeze.json",
        )
        proposals.append(
            {"iteration": iteration, "accepted": proposal["accepted"], "words": s.words(proposal["prompt"])}
        )
    for path in root.glob("requests/*/request.json"):
        q = s.read_json(path)
        if q["role"] == "writer":
            continue
        data = s.read_json(path.parent / "attempt_0.json")
        identity = s.canonical(q["identity"])
        if "/validation/" in identity:
            validation_times.append(data["sent_at"])
        else:
            training_times.append(data["timestamp"])
    first_rubric = min(
        s.read_json(p.parent / "attempt_0.json")["sent_at"]
        for p in root.glob("requests/*/request.json")
        if s.read_json(p)["role"] == "rubric"
    )
    frozen_at = run["freeze"]["frozen_at"]
    check(max(writer_times) <= first_rubric, "A rubric request was sent before the writer finished")
    check(max(training_times) <= frozen_at, "A training request finished after freeze.json was written")
    check(frozen_at <= min(validation_times), "A validation request was sent before freeze.json was written")
    costs = s.read_json(root / "costs.json")
    check(
        math.isclose(costs["actual_complete_usd"], total_cost),
        f"costs.json total {costs['actual_complete_usd']} != sum of responses {total_cost}",
    )
    check(costs["unresolved"] == 0, f"costs.json has {costs['unresolved']} unresolved requests")
    check(
        costs["requests"] == sum(counts.values()),
        f"costs.json counts {costs['requests']} requests; found {sum(counts.values())}",
    )
    shared_total = sum(e["charge"] for e in ledger.values())
    check(
        shared_total <= run_args["total_budget_usd"],
        f"Ledger total {shared_total} is over the ${run_args['total_budget_usd']} budget",
    )
    pilot = s.audit_xar_run(runs / "pilot-meta-blog-attested")
    gate = s.read_json(runs / "meta_blog_pilot_gate.json")
    check(
        gate["pilot_substantive_hash"] == pilot["manifest"]["substantive_hash"],
        "Pilot gate file does not match the pilot run",
    )
    selected = run["freeze"]["selected"]
    sensitivity = {}
    for split in ("train", "validation"):
        a, b = run["rows"][(0, split)], run["rows"][(selected, split)]
        for label, keep in [
            ("all", lambda r: True),
            ("length_compliant", lambda r: r["length_compliant"]),
            ("unflagged", lambda r: not r["contamination_flagged"]),
        ]:
            aa, bb = [r for r in a if keep(r)], [r for r in b if keep(r)]
            sensitivity[f"{split}_{label}"] = {
                "examples": len(bb),
                "initial": s.summarize(aa),
                "selected": s.summarize(bb),
                "improvement": s.paired_improvement(aa, bb),
            }
    wallclock = datetime.fromisoformat(max(ended.values())) - datetime.fromisoformat(manifest["created_at"])
    result = {
        "at": s.now(),
        "source_run": str(root),
        "proposals": proposals,
        "request_counts": dict(counts),
        "research_cost_usd": total_cost,
        "pilot_cost_usd": s.read_json(runs / "pilot-meta-blog-attested/costs.json")["actual_complete_usd"],
        "shared_charged_or_reserved_usd": shared_total,
        "writer_length_compliant": sum(c["length_compliant"] for c in candidates.values()),
        "writer_contamination_flagged": sum(c["contamination"]["flagged"] for c in candidates.values()),
        "latency_seconds": {
            r: {"median": statistics.median(v), "total": sum(v)} for r, v in latencies.items()
        },
        "research_wallclock_seconds": wallclock.total_seconds(),
        "sensitivity": sensitivity,
    }
    s.write_json(args.output, result)
    print(
        "Completion audit passed; research USD:",
        round(total_cost, 2),
        "shared charged/reserved USD:",
        round(shared_total, 2),
    )


if __name__ == "__main__":
    main()
