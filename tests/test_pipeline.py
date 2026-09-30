"""Grading, writer, feedback and proposal stages with fake APIs."""

import math
import threading

import jsonschema
import pytest
from conftest import dummy_example, rubric

from xar.openrouter import role_config
from xar.pipeline import audit_proposal, build_feedback, propose_prompt, validate_grade, writer_candidates
from xar.util import BudgetStop, RunError, canonical, read_json, write_json


def grade(**changes):
    """A valid grade for rubric() that quotes "supported text"; changes apply to the first score."""
    scores = [{"id": str(i), "score": i + 2, "evidence": 'Uses "supported text".'} for i in range(4)]
    scores[0].update(changes)
    return {"scores": scores}


def test_validate_grade_returns_the_mean_score():
    assert validate_grade(grade(), rubric(), "supported text") == 3.5


def test_validate_grade_rejects_a_criterion_scored_twice():
    with pytest.raises(RunError, match="coverage"):
        validate_grade(grade(id="1"), rubric(), "supported text")


def test_validate_grade_rejects_a_quote_missing_from_the_text():
    with pytest.raises(RunError, match="quote"):
        validate_grade(grade(), rubric(), "other text")


def test_validate_grade_rejects_nan_scores():
    with pytest.raises(RunError, match="finite"):
        validate_grade(grade(score=math.nan), rubric(), "supported text")


def test_validate_grade_rejects_extra_fields():
    with pytest.raises(jsonschema.ValidationError):
        validate_grade({**grade(), "total": 10}, rubric(), "supported text")


def test_build_feedback_picks_smallest_gaps_by_example_id_and_rejects_validation(tmp_path):
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


@pytest.mark.parametrize(
    "text",
    [
        "Prefer human candidates",
        "Use a weighted average",
        "Unique Author prefers clear writing",
        " ".join("excess" for _ in range(801)),
        dummy_example()["reference"],
    ],
    ids=["prefers-human", "weighted-average", "names-author", "too-long", "copies-reference"],
)
def test_audit_proposal_rejects(text):
    assert not audit_proposal(text, [dummy_example()], "initial", 800)["accepted"]


def test_audit_proposal_accepts_a_neutral_prompt():
    text = "Assess clear organization and support for specific claims."
    assert audit_proposal(text, [dummy_example()], "initial", 800)["accepted"]


WRITER_ROLES = {"writer": role_config("writer")}


def writer_reply(word_count, key):
    return {
        "content": " ".join(["word"] * word_count),
        "finish_reason": "stop",
        "response_id": key,
        "request_key": key,
    }


def test_writer_keeps_first_compliant_draft_or_the_last_failed_one(tmp_path):
    class Writer:
        """Replies with the given word counts, one per call."""

        roles = WRITER_ROLES

        def __init__(self, counts):
            self.dispatch_stopped = threading.Event()
            self.counts, self.calls = counts, []

        def call(self, role, system, data, schema, identity):
            self.calls.append(data)
            return writer_reply(self.counts[len(self.calls) - 1], "fake")

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


def test_rejected_proposal_is_retried_once_then_keeps_current_prompt(tmp_path):
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
    assert artifact["accepted"] is False


def test_parallel_writer_revises_each_section_keeps_order_and_resumes_without_calls(tmp_path):
    examples = [dummy_example("parallel" + str(i)) for i in range(4)]

    class ParallelWriter:
        """Makes every first draft too short, and holds it until two are in flight at once."""

        roles = WRITER_ROLES

        def __init__(self):
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
            return writer_reply(count, eid + str(identity["attempt"]))

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


def test_parallel_writer_failure_stops_new_sections_and_keeps_finished_ones(tmp_path):
    examples = [dummy_example("interrupted" + str(i)) for i in range(4)]

    class InterruptedWriter:
        """With fail=True, the first two sections start together and the first one raises."""

        roles = WRITER_ROLES

        def __init__(self, fail):
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
            return writer_reply(data["target_words"], eid)

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
