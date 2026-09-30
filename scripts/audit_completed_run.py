"""Re-audit the completed Muse/Kimi research run from its saved requests and outputs; makes no API calls."""

import argparse
import collections
import hashlib
import math
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import exp_xar
import shared as s


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="runs/meta-blog-seed0")
    parser.add_argument("--output", default="reports/completion_audit.json")
    args = parser.parse_args()
    root = Path(args.source_run)
    run = s.audit_xar_run(root)
    manifest = run["manifest"]
    # The code has changed since the run; check the recorded hashes against the commit that produced it.
    commit = manifest["git_commit"]
    assert commit, "Run manifest has no git commit; cannot verify the code that produced it"
    for name, expected in manifest["software_hashes"].items():
        blob = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=s.ROOT)
        assert hashlib.sha256(blob).hexdigest() == expected, (
            f"{name} at {commit} does not match the run manifest"
        )
    all_examples = s.load_examples(manifest["dataset"], manifest["splits"])
    examples = [e for e in all_examples if e["split"] in ("train", "validation")]
    train = [e for e in examples if e["split"] == "train"]
    lookup = {e["example_id"]: e for e in examples}
    review = s.read_json(Path(manifest["dataset"]).parent / "human_review.json")
    assert review["dataset_hash"] == manifest["dataset_hash"]
    assert all(review["papers"][e["paper_id"]]["decision"] == "approved" for e in examples)
    candidates = run["candidates"]
    assert set(candidates) == set(lookup) and len(candidates) == 52
    ledger = s.read_json(manifest["arguments"]["budget_ledger"])["entries"]
    counts, latencies, sent, ended = collections.Counter(), collections.defaultdict(list), {}, {}
    schemas = {"rubric": s.RUBRIC_SCHEMA, "judge": s.GRADE_SCHEMA, "optimizer": s.PROPOSAL_SCHEMA}
    total_cost = 0
    for path in root.glob("requests/*/request.json"):
        request = s.read_json(path)
        role, payload = request["role"], request["payload"]
        cfg, endpoint = manifest["roles"][role], manifest["endpoints"][role]
        assert role in ("writer", "rubric", "judge", "optimizer")
        assert (
            request["key"]
            == path.parent.name
            == s.digest(
                {
                    "payload": payload,
                    "identity": request["identity"],
                    "schema_version": manifest["schema_version"],
                }
            )
        )
        if role == "writer":
            for key in ("model", "temperature", "reasoning", "max_tokens"):
                assert payload[key] == cfg[key]
            assert payload["provider"] == {
                "only": [cfg["provider"]],
                "order": [cfg["provider"]],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
            assert payload["plugins"] == payload["transforms"] == [] and payload["stream"] is False
            assert "response_format" not in payload
            assert len(payload["messages"]) == 2
            assert (
                s.digest(payload["messages"][0]["content"])
                == manifest["software_hashes"]["prompts/writer.md"]
            )
        else:
            wrapper = {"rubric": "rubric_wrapper", "judge": "judge", "optimizer": "optimizer"}[role]
            s.audit_request_contract(
                payload, cfg, endpoint, schemas[role], manifest["software_hashes"][f"prompts/{wrapper}.md"]
            )
        bound = s.request_upper(payload, endpoint["endpoint"])
        for receipt_path in path.parent.glob("attempt_*.json"):
            receipt = s.read_json(receipt_path)
            assert receipt["status"] == "success"
            raw = receipt["response"]
            assert raw["model"] in (cfg["model"], endpoint["canonical_slug"])
            assert raw["provider"] in (cfg["provider"], endpoint["endpoint"]["provider_name"])
            assert raw["choices"][0]["finish_reason"] == "stop"
            index = receipt_path.stem.removeprefix("attempt_")
            key = str(root.resolve()) + "/" + request["key"] + "/" + index
            entry = ledger[key]
            cost = raw["usage"]["cost"]
            assert entry["state"] == "complete" and math.isclose(entry["charge"], cost)
            assert math.isclose(entry["upper"], bound) and cost <= bound
            assert entry["created_at"] <= receipt["sent_at"] <= receipt["timestamp"] <= entry["settled_at"]
            counts[role] += 1
            latencies[role].append(receipt["duration_seconds"])
            total_cost += cost
            sent[str(receipt_path)] = receipt["sent_at"]
            ended[str(receipt_path)] = receipt["timestamp"]
    writer_times, training_times, validation_times = [], [], []
    for eid, candidate in candidates.items():
        e = lookup[eid]
        attempts = candidate["attempts"]
        assert 1 <= len(attempts) <= 3 and candidate["accepted_attempt"] == len(attempts) - 1
        assert not any(a["length_compliant"] for a in attempts[:-1])
        assert candidate["length_compliant"] or len(attempts) == 3
        for i, attempt in enumerate(attempts):
            receipt = s.read_json(attempt["response"]["raw_response"])
            raw_text = receipt["response"]["choices"][0]["message"]["content"].strip()
            assert raw_text == attempt["text"] and s.digest(raw_text) == attempt["text_hash"]
            assert s.words(raw_text) == attempt["words"]
            assert attempt["length_compliant"] == (
                0.85 * e["target_words"] <= s.words(raw_text) <= 1.15 * e["target_words"]
            )
            q = s.read_json(Path(attempt["response"]["raw_response"]).parent / "request.json")
            expected = s.task_data(e)
            if i:
                expected.update(
                    previous_section=attempts[i - 1]["text"],
                    length_revision=f"Revise only to fit {math.ceil(0.85 * e['target_words'])}–{math.floor(1.15 * e['target_words'])} words. Preserve claims.",
                )
            assert s.canonical(expected) == q["payload"]["messages"][1]["content"]
            writer_times.append(receipt["timestamp"])
        assert candidate["text"] == attempts[-1]["text"] and candidate["complete"]
        assert candidate["contamination"] == s.contamination(candidate["text"], e["reference"])
    initial = (root / "prompts/iter_00.md").read_text()
    proposals = []
    for iteration in range(1, 8):
        previous = (root / f"prompts/iter_{iteration - 1:02d}.md").read_text()
        rows = s.read_json(root / f"scores/main/{iteration - 1}/train_rows.json")
        expected = exp_xar.build_feedback(train, candidates, rows, previous, 4)
        directory = root / f"feedback/iter_{iteration:02d}"
        feedback, proposal = (
            s.read_json(directory / "training.json"),
            s.read_json(directory / "proposal.json"),
        )
        assert s.canonical(expected) == s.canonical(feedback)
        assert proposal["parent_prompt_hash"] == s.digest(previous)
        assert proposal["feedback_hash"] == s.digest(feedback) and proposal["update_consumed"]
        assert 1 <= len(proposal["attempts"]) <= 2
        for attempt in proposal["attempts"]:
            payload = s.audit_saved_output(attempt, s.PROPOSAL_SCHEMA)
            assert payload["messages"][1]["content"] == s.canonical(
                {"feedback": feedback, "max_meta_prompt_words": 800}
            )
            assert attempt["audit"] == s.audit_proposal(attempt["value"]["prompt"], train, initial, 800)
        assert proposal["accepted"] == proposal["attempts"][-1]["audit"]["accepted"]
        assert proposal["prompt"] == (
            proposal["attempts"][-1]["value"]["prompt"] if proposal["accepted"] else previous
        )
        assert proposal["prompt_hash"] == run["freeze"]["prompt_hashes"][iteration]
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
    assert max(writer_times) <= first_rubric
    assert max(training_times) <= run["freeze"]["frozen_at"] <= min(validation_times)
    costs = s.read_json(root / "costs.json")
    assert math.isclose(costs["actual_complete_usd"], total_cost) and costs["unresolved"] == 0
    assert costs["requests"] == sum(counts.values()) == 1389
    shared_total = sum(e["charge"] for e in ledger.values())
    assert shared_total <= 100
    pilot = s.audit_xar_run("runs/pilot-meta-blog-attested")
    gate = s.read_json("runs/meta_blog_pilot_gate.json")
    assert (
        gate["pilot_substantive_hash"] == pilot["manifest"]["substantive_hash"]
        and gate["gap_sign_used_for_gate"] is False
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
    result = {
        "at": s.now(),
        "source_run": str(root),
        "scientific_execution_complete": True,
        "frozen_software_verified": True,
        "writer_raw_inputs_outputs_and_repairs_verified": True,
        "all_requests_routing_schema_identity_price_reservations_verified": True,
        "seven_optimizer_proposals_feedback_and_static_audits_verified": proposals,
        "selection_and_validation_order_verified": True,
        "human_review_verified": True,
        "pilot_raw_verified": True,
        "request_counts": dict(counts),
        "research_cost_usd": total_cost,
        "pilot_cost_usd": s.read_json("runs/pilot-meta-blog-attested/costs.json")["actual_complete_usd"],
        "shared_charged_or_reserved_usd": shared_total,
        "research_unresolved_requests": 0,
        "writer_length_compliant": sum(c["length_compliant"] for c in candidates.values()),
        "writer_contamination_flagged": sum(c["contamination"]["flagged"] for c in candidates.values()),
        "latency_seconds": {
            r: {"median": statistics.median(v), "total": sum(v)} for r, v in latencies.items()
        },
        "research_wallclock_seconds": (
            s.dt.datetime.fromisoformat(max(ended.values()))
            - s.dt.datetime.fromisoformat(manifest["created_at"])
        ).total_seconds(),
        "sensitivity": sensitivity,
        "version_control_final_artifacts": "pending_commit",
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
