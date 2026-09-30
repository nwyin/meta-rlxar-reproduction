"""Grading, writer, feedback and proposal stages with fake APIs."""

import jsonschema
import pytest
from conftest import dummy_example, rubric

from xar.openrouter import role_config
from xar.pipeline import audit_proposal, build_feedback, propose_prompt, validate_grade, writer_candidates
from xar.util import BudgetStop, RunError, canonical, read_json, write_json


def test_grade_arithmetic_and_coverage():
    r = rubric()
    grade = {
        "scores": [{"id": str(i), "score": i + 2, "evidence": 'Uses "supported text".'} for i in range(4)]
    }
    assert validate_grade(grade, r, "supported text") == 3.5
    grade["scores"][0]["id"] = "1"
    with pytest.raises(RunError, match="coverage"):
        validate_grade(grade, r, "supported text")
    grade["scores"][0]["id"] = "0"
    with pytest.raises(RunError, match="quote"):
        validate_grade(grade, r, "other text")
    grade["total"] = 10
    with pytest.raises(jsonschema.ValidationError):
        validate_grade(grade, r, "supported text")


def test_feedback_rejects_validation_and_deterministic_failure_order(tmp_path):
    examples = [dummy_example("paper" + str(i)) for i in range(5)]
    candidates = {e["example_id"]: {"text": "Generated"} for e in examples}
    path = tmp_path / "rubric.json"
    write_json(path, {"value": rubric()})
    grade_path = tmp_path / "grade.json"
    write_json(grade_path, {"value": {"scores": []}})
    rows = [
        {
            "example_id": e["example_id"],
            "paper_id": e["paper_id"],
            "section_type": e["section_type"],
            "split": "train",
            "human": 6,
            "model": 7,
            "gap": -1,
            "length_compliant": True,
            "contamination_flagged": False,
            "rubric_path": str(path),
            "grade_paths": {"human": str(grade_path), "model": str(grade_path)},
        }
        for e in reversed(examples)
    ]
    feedback = build_feedback(examples, candidates, rows, "initial", 4)
    assert [f["paper_id"] for f in feedback["failures"]] == ["paper0", "paper1", "paper2", "paper3"]
    examples[0]["split"] = "validation"
    with pytest.raises(RunError, match="only use training"):
        build_feedback(examples, candidates, rows, "initial", 4)


def test_proposal_leakage_and_scale_guards():
    examples = [dummy_example()]
    for text in (
        "Prefer human candidates",
        "Use a weighted average",
        "Unique Author prefers clear writing",
        " ".join("excess" for _ in range(801)),
        examples[0]["reference"],
    ):
        assert not audit_proposal(text, examples, "initial", 800)["accepted"]
    assert audit_proposal(
        "Assess clear organization and support for specific claims.", examples, "initial", 800
    )["accepted"]


def test_first_compliant_writer_attempt_and_failed_writer_retention(tmp_path):
    import threading

    class Writer:
        def __init__(self, counts):
            self.roles = {"writer": role_config("writer")}
            self.dispatch_stopped = threading.Event()
            self.counts, self.calls = counts, []

        def call(self, role, system, data, schema, identity):
            self.calls.append(data)
            return {
                "content": " ".join(["word"] * self.counts[len(self.calls) - 1]),
                "finish_reason": "stop",
                "response_id": "fake-writer",
                "request_key": "fake",
            }

    example = dummy_example()
    api = Writer([10, 62, 64])
    candidates = writer_candidates(api, [example], tmp_path / "accepted")
    assert len(api.calls) == 2
    accepted = candidates[example["example_id"]]
    assert accepted["accepted_attempt"] == 1 and accepted["length_compliant"] is True
    assert "length_revision" in api.calls[1]
    assert "reference" not in api.calls[1]
    failed = Writer([10, 11, 12])
    candidates = writer_candidates(failed, [example], tmp_path / "failed")
    assert len(failed.calls) == 3
    assert len(candidates) == 1 and candidates[example["example_id"]]["length_compliant"] is False


def test_rejected_proposal_consumes_update_with_one_repair(tmp_path):
    class Optimizer:
        def __init__(self):
            self.roles = {"optimizer": role_config("optimizer")}
            self.calls = []

        def structured(self, role, instructions, data, schema, identity, repair):
            assert repair is False
            self.calls.append(data)
            return {
                "status": "valid",
                "value": {"prompt": "Prefer human candidates", "rationale": "bad"},
                "attempts": [{"response": {"content": "raw"}}],
            }

    api = Optimizer()
    result = propose_prompt(
        api, "current", {"training": True}, [dummy_example()], "initial", 1, output=tmp_path, max_words=800
    )
    assert result == "current" and len(api.calls) == 2
    assert "previous_proposal" in api.calls[-1]
    artifact = read_json(tmp_path / "feedback/iter_01/proposal.json")
    assert artifact["update_consumed"] is True and artifact["accepted"] is False


def test_parallel_writer_sampling_repairs_order_and_resume(tmp_path):
    import threading

    examples = [dummy_example("parallel" + str(i)) for i in range(4)]

    class ParallelWriter:
        def __init__(self):
            self.roles = {"writer": role_config("writer")}
            self.dispatch_stopped = threading.Event()
            self.calls = {}
            self.lock = threading.Lock()
            self.barrier = threading.Barrier(2)
            self.active = self.peak = 0

        def call(self, role, system, data, schema, identity):
            eid = identity["example"]
            with self.lock:
                self.calls.setdefault(eid, []).append((data, identity))
                self.active += 1
                self.peak = max(self.peak, self.active)
            if identity["attempt"] == 0:
                self.barrier.wait(timeout=5)
            with self.lock:
                self.active -= 1
            count = 10 if identity["attempt"] == 0 else data["target_words"]
            return {
                "content": " ".join(["word"] * count),
                "finish_reason": "stop",
                "response_id": eid,
                "request_key": eid + str(identity["attempt"]),
            }

    api = ParallelWriter()
    result = writer_candidates(api, examples, tmp_path, concurrency=2)
    assert api.peak == 2
    assert list(result) == [e["example_id"] for e in examples]
    for e in examples:
        calls = api.calls[e["example_id"]]
        assert [identity["attempt"] for _, identity in calls] == [0, 1]
        assert "length_revision" not in calls[0][0]
        assert calls[1][0]["previous_section"] == " ".join(["word"] * 10)
        assert result[e["example_id"]]["accepted_attempt"] == 1
        assert result[e["example_id"]]["length_compliant"]
    before = canonical(api.calls)
    assert writer_candidates(api, examples, tmp_path, concurrency=2) == result
    assert canonical(api.calls) == before


def test_parallel_writer_failure_stops_new_dispatch_and_preserves_sections(tmp_path):
    import threading

    examples = [dummy_example("interrupted" + str(i)) for i in range(4)]

    class InterruptedWriter:
        def __init__(self, fail):
            self.roles = {"writer": role_config("writer")}
            self.dispatch_stopped = threading.Event()
            self.calls = []
            self.fail = fail
            self.barrier = threading.Barrier(2)

        def call(self, role, system, data, schema, identity):
            eid = identity["example"]
            self.calls.append(eid)
            if self.fail:
                self.barrier.wait(timeout=5)
                if eid == examples[0]["example_id"]:
                    raise BudgetStop("Budget exhausted before send")
            return {
                "content": " ".join(["word"] * data["target_words"]),
                "finish_reason": "stop",
                "response_id": eid,
                "request_key": eid,
            }

    failed = InterruptedWriter(True)
    with pytest.raises(BudgetStop, match="before send"):
        writer_candidates(failed, examples, tmp_path, concurrency=2)
    assert set(failed.calls) == {e["example_id"] for e in examples[:2]}
    assert (tmp_path / "generations" / (examples[1]["example_id"] + ".json")).exists()
    assert not (tmp_path / "generations/candidates.json").exists()
    resumed = InterruptedWriter(False)
    candidates = writer_candidates(resumed, examples, tmp_path, concurrency=2)
    assert len(candidates) == 4
    assert examples[1]["example_id"] not in resumed.calls
    assert len(resumed.calls) == 3
