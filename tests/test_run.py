"""Offline tests for the two-script runner: experiment boundaries and recorded API failures."""

import json
import math
import threading
from collections import Counter
from dataclasses import replace
from pathlib import Path

import httpx
import jsonschema
import pytest
from conftest import PROVIDER_NAMES, call_kind, dummy_example, install_fake, rubric, write_dataset

import run
import run as openrouter
from run import (
    BACKOFF_SECONDS,
    FAILING_FEEDBACK,
    MAX_SENDS,
    MODEL_CATALOG,
    PRICING_HEADROOM,
    ROLES,
    SECTIONS,
    SNAPSHOTS,
    OpenRouter,
    RunError,
    UncertainSend,
    audit_proposal,
    bootstrap,
    build_feedback,
    canonical,
    check_model_allowed,
    endpoint_for,
    endpoints_filename,
    highest_prices,
    length_window,
    load_examples,
    paired_improvement,
    propose_prompt,
    read_json,
    role_config,
    run_examples,
    run_xar,
    summarize,
    task_data,
    validate_grade,
    write_json,
    writer_candidates,
)

WRITER_PROVIDER = PROVIDER_NAMES[role_config("writer")["provider"]]
MAX_WORDS = 800


def mock_api(tmp_path, role, handler):
    """An OpenRouter client for one role whose HTTP requests go to handler."""
    return OpenRouter(
        tmp_path / "run",
        {role: role_config(role)},
        seed=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_check_model_allowed_rejects_excluded_families_and_routers():
    for model in ("anthropic/claude-test", "google/gemini-test", "openrouter/auto", "qwen/qwen3.5-9b:free"):
        with pytest.raises(RunError):
            check_model_allowed(model)


def test_a_send_with_no_response_is_recorded_and_sent_again(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(openrouter.time, "sleep", waits.append)
    sends = []

    def handler(request):
        sends.append(request)
        if len(sends) == 1:
            raise httpx.ReadTimeout("timed out")
        if len(sends) == 2:
            raise httpx.ReadError("[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record mac")
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "provider": WRITER_PROVIDER,
                "usage": {"cost": 0.25},
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            },
        )

    api = mock_api(tmp_path, "writer", handler)
    result = api.call("writer", "Instructions", {"paper": "first"}, None, "first")
    assert result["content"] == "ok" and len(sends) == 3
    assert waits == [BACKOFF_SECONDS[0], BACKOFF_SECONDS[1]]
    attempts = [read_json(f) for f in sorted(tmp_path.glob("run/requests/*/attempt_*.json"))]
    assert [a["status"] for a in attempts] == ["uncertain", "uncertain", "success"]
    assert attempts[0]["error"].startswith("ReadTimeout") and attempts[1]["error"].startswith("ReadError")
    # The two sends without a response may have been billed, up to their cost bound each.
    costs = api.costs()
    assert costs["actual_complete_usd"] == 0.25 and costs["requests"] == 1 and costs["unresolved"] == 2
    assert costs["unresolved_upper_usd"] == pytest.approx(2 * attempts[0]["upper_usd"])


def test_costs_sum_the_billed_sends_and_reject_a_missing_cost(tmp_path):
    replies = iter([{"cost": 0.25}, {"cost": 0.5}, {}])

    def handler(request):
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "provider": WRITER_PROVIDER,
                "usage": next(replies),
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            },
        )

    api = mock_api(tmp_path, "writer", handler)
    api.call("writer", "Instructions", {"paper": "first"}, None, "first")
    api.call("writer", "Instructions", {"paper": "second"}, None, "second")
    assert api.costs() == {
        "actual_complete_usd": 0.75,
        "requests": 2,
        "unresolved": 0,
        "unresolved_upper_usd": 0.0,
    }
    with pytest.raises(UncertainSend, match="usage.cost"):
        api.call("writer", "Instructions", {"paper": "third"}, None, "third")


def test_invalid_reply_is_repaired_once_with_the_error(tmp_path):
    systems = []

    def handler(request):
        payload = json.loads(request.content)
        systems.append(payload["messages"][0]["content"])
        content = "not json" if len(systems) == 1 else json.dumps({"answer": "ok"})
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "provider": PROVIDER_NAMES[role_config("judge")["provider"]],
                "usage": {"cost": 0.0001},
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            },
        )

    api = mock_api(tmp_path, "judge", handler)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    result = api.structured("judge", "Instructions", {"paper": "text"}, schema, "task")
    assert result["status"] == "valid" and result["value"] == {"answer": "ok"}
    assert [a["status"] for a in result["attempts"]] == ["invalid", "valid"]
    assert "FORMAT REPAIR" in systems[1] and "Expecting value" in systems[1]


@pytest.mark.parametrize(
    "factor,accepted", [(0.5, True), (PRICING_HEADROOM - 0.01, True), (PRICING_HEADROOM + 0.05, False)]
)
def test_preflight_accepts_price_changes_up_to_the_headroom(tmp_path, factor, accepted):
    cfg = role_config("judge")
    endpoint, _ = endpoint_for(cfg)

    def handler(request):
        if request.url.path.endswith("/endpoints"):
            data = read_json(SNAPSHOTS / endpoints_filename(cfg["model"]))
            for e in data["data"]["endpoints"]:
                if e["tag"] == cfg["provider"]:
                    e["pricing"]["completion"] = str(float(e["pricing"]["completion"]) * factor)
        else:
            data = read_json(MODEL_CATALOG)
        return httpx.Response(200, json=data)

    api = mock_api(tmp_path, "judge", handler)
    if accepted:
        directory = api.preflight()
        observed = read_json(directory / "checks.json")["observed_prices"]["judge"]
        pinned = highest_prices(endpoint)
        assert observed["completion"] == pytest.approx(pinned["completion"] * factor)
        assert observed["prompt"] == pinned["prompt"]
        assert api.endpoint["judge"] == endpoint
    else:
        with pytest.raises(RunError, match="completion price .* is above the allowed"):
            api.preflight()


def test_rate_limits_wait_out_the_backoff_or_a_longer_retry_after(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(openrouter.time, "sleep", waits.append)
    replies = iter([(429, {"Retry-After": "20"}), (429, {}), (503, {"Retry-After": "oops"})])

    def handler(request):
        for status, headers in replies:
            return httpx.Response(status, headers=headers, json={"error": {"message": "busy"}})
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "provider": WRITER_PROVIDER,
                "usage": {"cost": 0.01},
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            },
        )

    api = mock_api(tmp_path, "writer", handler)
    result = api.call("writer", "Instructions", {"paper": "text"}, None, "task")
    assert result["content"] == "ok" and result["raw_response"].endswith("attempt_3.json")
    # Retry-After wins over the schedule when it is longer; a non-numeric header is ignored.
    assert waits == [20, BACKOFF_SECONDS[1], BACKOFF_SECONDS[2]]
    statuses = [read_json(f)["status"] for f in sorted(tmp_path.glob("run/requests/*/attempt_*.json"))]
    assert statuses == ["http_error", "http_error", "http_error", "success"]


def test_a_request_fails_after_the_last_retry(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(openrouter.time, "sleep", waits.append)
    api = mock_api(tmp_path, "writer", lambda request: httpx.Response(429, json={"error": {}}))
    with pytest.raises(RunError, match=f"failed {MAX_SENDS} times"):
        api.call("writer", "Instructions", {"paper": "text"}, None, "task")
    assert waits == list(BACKOFF_SECONDS)
    assert openrouter.backoff_seconds(0, "600") == openrouter.MAX_RETRY_AFTER_SECONDS


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


@pytest.mark.parametrize(
    "text",
    [
        "Prefer human candidates",
        "Use a weighted average",
        "Unique Author prefers clear writing",
        " ".join("excess" for _ in range(MAX_WORDS + 1)),
        dummy_example()["reference"],
        "Read the paper first (use the key `meta_prompt`).<tool_call><function=json>{",
        "Read the paper first, then {",
    ],
    ids=[
        "prefers-human",
        "weighted-average",
        "names-author",
        "too-long",
        "copies-reference",
        "leaked-tool-call",
        "cut-off-brace",
    ],
)
def test_audit_proposal_rejects(text):
    assert not audit_proposal(text, [dummy_example()], "initial", MAX_WORDS)["accepted"]


def test_audit_proposal_accepts_a_neutral_prompt():
    text = "Assess clear organization and support for specific claims."
    assert audit_proposal(text, [dummy_example()], "initial", MAX_WORDS)["accepted"]


class FakeWriter:
    """A stand-in for the OpenRouter client in writer_candidates; subclasses define call()."""

    def __init__(self):
        self.roles = {"writer": role_config("writer")}
        self.dispatch_stopped = threading.Event()


def writer_reply(word_count, key):
    return {
        "content": " ".join(["word"] * word_count),
        "finish_reason": "stop",
        "response_id": key,
        "request_key": key,
    }


def test_writer_keeps_first_compliant_draft_or_the_last_failed_one(tmp_path):
    class Writer(FakeWriter):
        """Replies with the given word counts, one per call."""

        def __init__(self, counts):
            super().__init__()
            self.counts, self.calls = counts, []

        def call(self, role, system, data, schema, identity):
            self.calls.append(data)
            return writer_reply(self.counts[len(self.calls) - 1], "fake")

    example = dummy_example()
    low, _ = length_window(example["target_words"])
    assert 10 < low <= example["target_words"]
    api = Writer([10, example["target_words"]])  # too short, then on target
    candidates = writer_candidates(api, [example], tmp_path / "accepted")
    assert len(api.calls) == 2
    accepted = candidates[example["example_id"]]
    assert accepted["accepted_attempt"] == 1 and accepted["length_compliant"] is True
    assert "length_revision" in api.calls[1]
    assert "reference" not in api.calls[1]
    failed = Writer([10, 11, 12])  # every draft too short
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
        api,
        "current",
        {"training": True},
        [dummy_example()],
        "initial",
        1,
        output=tmp_path,
        max_words=MAX_WORDS,
    )
    assert result == "current" and len(api.calls) == 2
    assert "previous_proposal" in api.calls[-1]
    artifact = read_json(tmp_path / "feedback/iter_01/proposal.json")
    assert artifact["accepted"] is False


def test_parallel_writer_revises_each_section_and_keeps_order(tmp_path):
    examples = [dummy_example("parallel" + str(i)) for i in range(4)]

    class ParallelWriter(FakeWriter):
        """Makes every first draft too short, and holds it until two are in flight at once."""

        def __init__(self):
            super().__init__()
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


def test_parallel_writer_failure_stops_new_sections_and_keeps_finished_ones(tmp_path):
    examples = [dummy_example("interrupted" + str(i)) for i in range(4)]

    class InterruptedWriter(FakeWriter):
        """With fail=True, the first two sections start together and the first one raises."""

        def __init__(self, fail):
            super().__init__()
            self.calls = []
            self.fail = fail
            self.barrier = threading.Barrier(2)

        def call(self, role, system, data, schema, identity):
            eid = identity["example"]
            self.calls.append(eid)
            if self.fail:
                self.barrier.wait(timeout=5)
                if eid == examples[0]["example_id"]:
                    raise UncertainSend("Timed out before send")
            return writer_reply(data["target_words"], eid)

    failed = InterruptedWriter(True)
    with pytest.raises(UncertainSend, match="before send"):
        writer_candidates(failed, examples, tmp_path, concurrency=2)
    assert set(failed.calls) == {e["example_id"] for e in examples[:2]}
    assert (tmp_path / "generations" / (examples[1]["example_id"] + ".json")).exists()
    assert not (tmp_path / "generations/candidates.json").exists()


def test_bootstrap_resamples_whole_papers():
    rows = [{"paper_id": "a", "gap": 1} for _ in range(4)] + [{"paper_id": "b", "gap": 3} for _ in range(4)]
    interval = bootstrap(rows)
    assert interval["paper_clusters"] == 2
    assert interval["low"] == 1 and interval["high"] == 3
    unequal = bootstrap([{"paper_id": "a", "gap": 0}] + [{"paper_id": "b", "gap": 10}] * 4)
    assert unequal["estimate"] == 8 and unequal["bootstrap_median"] == 8


def row(example, paper, section, human, model):
    gap = None if human is None else human - model
    return {
        "example_id": example,
        "paper_id": paper,
        "section_type": section,
        "human": human,
        "model": model,
        "gap": gap,
        "length_compliant": True,
        "contamination_flagged": False,
    }


def test_summarize_averages_only_sections_with_both_grades():
    rows = [
        row("a1", "a", "abstract", 8, 6),
        row("a2", "a", "conclusion", 5, 6),
        row("b1", "b", "abstract", None, 7),
    ]
    summary = summarize(rows)
    assert summary["examples"] == 3 and summary["paired_coverage"] == 2 and not summary["complete"]
    assert summary["gap"] == 0.5 and summary["human"] == 6.5
    assert summary["sections"]["abstract"] == {"coverage": 1, "gap": 2}
    assert summary["sections"]["introduction"] == {"coverage": 0, "gap": None}
    assert summary["paper_interval"]["paper_clusters"] == 1


def test_paired_improvement_skips_sections_missing_a_gap():
    initial = [row("a1", "a", "abstract", 5, 7), row("a2", "a", "conclusion", None, 7)]
    selected = [row("a1", "a", "abstract", 6, 6), row("a2", "a", "conclusion", 6, 6)]
    improvement = paired_improvement(initial, selected)
    assert improvement["paired_coverage"] == 1 and improvement["mean"] == 2


def test_task_data_sends_only_visible_fields():
    e = dummy_example()
    assert set(task_data(e)) == {"visible_paper", "section_type", "target_words"}
    assert e["reference"] not in canonical(task_data(e))


def test_load_examples_rejects_a_paper_in_two_splits(tmp_path):
    example = dummy_example()
    path = tmp_path / "examples.jsonl"
    path.write_text(canonical(example) + "\n")
    splits = tmp_path / "splits.json"
    write_json(splits, {"papers": {"train": ["paper1"], "validation": ["paper1"]}})
    with pytest.raises(RunError, match="overlap"):
        load_examples(path, splits)


def test_run_examples_splits_pilot_papers_into_train_and_validation():
    examples = [
        dummy_example(paper, section, "pilot") for paper in ("pilotB", "pilotA") for section in SECTIONS
    ]
    examples.append(dummy_example("paper9", "abstract", "train"))
    labelled = run_examples(examples, "pilot")
    assert {e["paper_id"]: e["split"] for e in labelled} == {"pilotA": "train", "pilotB": "validation"}
    with pytest.raises(RunError, match="needs 2 papers"):
        run_examples(examples[:4], "pilot")


def test_run_examples_pilot_uses_the_first_training_papers_when_no_pilot_split_exists():
    examples = [
        dummy_example(paper, section, "train")
        for paper in ("paperC", "paperA", "paperB")
        for section in SECTIONS
    ]
    labelled = run_examples(examples, "pilot")
    # Papers C and A come first in the dataset; A sorts first, so it trains and C validates.
    assert {e["paper_id"]: e["split"] for e in labelled} == {"paperA": "train", "paperC": "validation"}
    with pytest.raises(RunError, match="needs 2 papers"):
        run_examples(examples[: len(SECTIONS)], "pilot")


def test_run_examples_rejects_an_unknown_split():
    with pytest.raises(RunError, match="Unknown split 'confirmation'"):
        run_examples([dummy_example()], "confirmation")


@pytest.mark.parametrize("seed", [None, 0, 1, 2])
def test_seed_override_preserves_experiment_settings(monkeypatch, design, seed):
    design["seed"] = 7
    monkeypatch.setattr(run, "load_design", lambda: design)
    arguments = ["--dry-run", "--output", "runs/repeat"]
    if seed is not None:
        arguments.extend(["--seed", str(seed)])
    settings = run.settings_for(run.parse_args(arguments))
    assert settings.seed == (7 if seed is None else seed)
    assert settings.output_dir == "runs/repeat"
    assert settings.iterations == design["iterations"]
    assert settings.dry_run


def test_unsupported_feedback_policy_stops(monkeypatch, design):
    monkeypatch.setattr(
        run, "load_design", lambda: {**design, "feedback_policy": "smallest_gap_then_example_id"}
    )
    with pytest.raises(RunError, match="feedback_policy"):
        run.settings_for(run.parse_args(["--dry-run"]))


def test_feedback_preserves_active_selection_payload_and_training_boundary(tmp_path):
    examples = [dummy_example("paper" + str(index)) for index in range(5)]
    gaps = {"paper0": 2, "paper1": -1, "paper2": -1, "paper3": 0, "paper4": -2}
    candidates = {example["example_id"]: {"text": "Generated"} for example in examples}
    path, grade_path = tmp_path / "rubric.json", tmp_path / "grade.json"
    write_json(path, {"value": rubric()})
    write_json(grade_path, {"value": {"scores": []}})
    rows = [
        {
            "example_id": example["example_id"],
            "paper_id": example["paper_id"],
            "section_type": example["section_type"],
            "split": "train",
            "human": 6,
            "model": 6 - gaps[example["paper_id"]],
            "gap": gaps[example["paper_id"]],
            "length_compliant": True,
            "contamination_flagged": False,
            "rubric_path": str(path),
            "grade_paths": {"human": str(grade_path), "model": str(grade_path)},
        }
        for example in reversed(examples)
    ]
    feedback = build_feedback(examples, candidates, rows, "initial", 3)
    assert [failure["example_id"] for failure in feedback["failures"]] == [
        examples[index]["example_id"] for index in (4, 1, 2)
    ]
    assert feedback["selection"] == FAILING_FEEDBACK
    assert not {"visible_paper", "paper_id"} & feedback["failures"][0].keys()
    assert [item["gap"] for item in feedback["all_training_gaps"]] == [-2, -1, -1, 0]
    assert feedback["summary"] == summarize(rows)
    assert {"paper_interval", "compliant_sensitivity", "unflagged_sensitivity"} <= feedback["summary"].keys()
    assert build_feedback(examples, candidates, rows, "initial", 10)["failures"][-1]["gap"] == 0
    rows[0]["gap"] = None
    with pytest.raises(RunError, match="every training section graded"):
        build_feedback(examples, candidates, rows, "initial", 3)
    rows[0]["gap"] = -2
    examples[0]["split"] = "validation"
    with pytest.raises(RunError, match="only use training"):
        build_feedback(examples, candidates, rows, "initial", 3)


def configured_run(tmp_path, *, pilot=False, seed=2):
    dataset, splits = write_dataset(
        tmp_path,
        {"train": ["train0", "train1"], "validation": ["validation0"], "confirmation": ["confirmation0"]},
    )
    arguments = [
        "--output",
        str(tmp_path / "fresh-run"),
        "--dataset",
        str(dataset),
        "--splits",
        str(splits),
        "--seed",
        str(seed),
        "--concurrency",
        "2",
    ]
    return run.settings_for(run.parse_args(arguments + (["--pilot"] if pilot else [])))


@pytest.mark.parametrize("pilot", [False, True])
def test_fresh_run_completes_with_fixed_candidates_and_blind_grades(tmp_path, monkeypatch, pilot):
    settings = replace(configured_run(tmp_path, pilot=pilot), iterations=2)
    fake = install_fake(monkeypatch)
    original_evaluate = run.evaluate_checkpoint
    sequence = []

    def checked_evaluate(api, examples, *args):
        split = examples[0]["split"]
        sequence.append(split)
        if split == "validation":
            assert (Path(settings.output_dir) / "freeze.json").exists()
        return original_evaluate(api, examples, *args)

    monkeypatch.setattr(run, "evaluate_checkpoint", checked_evaluate)
    result = run_xar(settings)
    out = Path(settings.output_dir)
    freeze = read_json(out / "freeze.json")
    assert freeze["selected"] == 0
    assert len(freeze["prompt_hashes"]) == settings.iterations + 1
    assert sequence == ["train"] * 3 + ["validation"] * 3
    assert read_json(out / "status.json")["state"] == "complete"
    assert result == read_json(out / "summary.json")
    assert result["paired_improvement"]["interval"]["seed"] == settings.seed
    assert (out / "results.md").exists() and (out / "checkpoints.csv").exists()
    sections = len(SECTIONS) * (2 if pilot else 3)
    assert Counter(call_kind(payload) for payload in fake.payloads) == {
        "writer": sections,
        "rubric": sections * 3,
        "grade": 2 * sections * 3,
        "optimizer": 2,
    }
    assert {payload["model"] for payload in fake.payloads} == {role_config(role)["model"] for role in ROLES}
    assert "confirmation0" not in canonical(fake.payloads)
    for payload in fake.payloads:
        if call_kind(payload) in {"grade", "rubric", "writer"}:
            data = json.loads(payload["messages"][1]["content"])
            assert not {"reference", "split", "origin", "example_id", "paper_id"} & data.keys()
    before = {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}
    with pytest.raises(RunError, match="Output already exists"):
        run_xar(settings)
    assert before == {path.relative_to(out): path.read_bytes() for path in out.rglob("*") if path.is_file()}


def test_dry_run_has_no_requests_or_writes(tmp_path, monkeypatch):
    settings = replace(configured_run(tmp_path), dry_run=True)

    def forbidden(*args, **kwargs):
        pytest.fail("A dry run must not construct a network client")

    monkeypatch.setattr(run, "OpenRouter", forbidden)
    estimate = run_xar(settings)
    assert estimate["estimated_usd_with_retry_reserve"] > 0
    assert not Path(settings.output_dir).exists()


def test_existing_output_is_rejected_before_client_creation(tmp_path, monkeypatch):
    settings = configured_run(tmp_path)
    Path(settings.output_dir).mkdir()

    def forbidden(*args, **kwargs):
        pytest.fail("An existing output must be rejected before client creation")

    monkeypatch.setattr(run, "OpenRouter", forbidden)
    with pytest.raises(RunError, match="Output already exists"):
        run_xar(settings)
    assert not list(Path(settings.output_dir).iterdir())


def test_failed_run_records_error_costs_and_raw_response(tmp_path, monkeypatch):
    settings = configured_run(tmp_path)
    fake = install_fake(monkeypatch)
    original = fake.handle

    def fail_chat(request):
        return original(request) if request.method == "GET" else httpx.Response(200, json={"choices": []})

    # install_fake captured a bound method; replace the client transport for this scenario.
    real = OpenRouter
    monkeypatch.setattr(
        run,
        "OpenRouter",
        lambda *args, **kwargs: real(
            *args, **kwargs, client=httpx.Client(transport=httpx.MockTransport(fail_chat))
        ),
    )
    with pytest.raises(RunError, match="unusable HTTP 200"):
        run_xar(settings)
    out = Path(settings.output_dir)
    assert read_json(out / "status.json")["state"] == "failed"
    assert "RunError" in read_json(out / "status.json")["error"]
    assert read_json(out / "costs.json")["unresolved"] >= 1
    attempts = [read_json(path) for path in out.glob("requests/*/attempt_*.json")]
    assert all(item["response"] == {"choices": []} for item in attempts)


def test_missing_validation_grades_produce_an_incomplete_report(tmp_path, monkeypatch):
    settings = replace(configured_run(tmp_path), iterations=0)
    fake = install_fake(monkeypatch)
    original = fake.handle

    def missing_grades(request):
        response = original(request)
        if request.method == "POST" and (Path(settings.output_dir) / "freeze.json").exists():
            value = response.json()
            value["choices"][0]["message"]["content"] = "not valid JSON"
            return httpx.Response(200, json=value)
        return response

    monkeypatch.setattr(
        run,
        "OpenRouter",
        lambda *args, **kwargs: OpenRouter(
            *args, **kwargs, client=httpx.Client(transport=httpx.MockTransport(missing_grades))
        ),
    )
    result = run_xar(settings)
    out = Path(settings.output_dir)
    assert read_json(out / "status.json")["state"] == "incomplete"
    assert result["paired_improvement"] == {"paired_coverage": 0, "mean": None, "interval": None}
    assert result["summaries"][0]["validation"]["paper_interval"] is None
    assert "missing" in (out / "results.md").read_text()


@pytest.mark.parametrize("usage", [None, {}, {"cost": -1}, {"cost": "0.2"}, []])
def test_missing_or_invalid_billing_is_recorded_as_uncertain(tmp_path, usage):
    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": role_config("writer")["model"],
                "usage": usage,
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            },
        )

    api = mock_api(tmp_path, "writer", handler)
    with pytest.raises(UncertainSend, match="usage.cost"):
        api.call("writer", "Instructions", {}, None, "bad-billing")
    assert api.costs()["unresolved"] == 1
    assert api.costs()["unresolved_upper_usd"] > 0


@pytest.mark.parametrize("field, value", [("model", "different/model"), ("provider", "Different Provider")])
def test_unexpected_serving_model_or_provider_stops_dispatch(tmp_path, field, value):
    def handler(request):
        response = {
            "model": role_config("writer")["model"],
            "provider": WRITER_PROVIDER,
            "usage": {"cost": 0.1},
            "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
        }
        return httpx.Response(200, json={**response, field: value})

    api = mock_api(tmp_path, "writer", handler)
    with pytest.raises(RunError, match="Unexpected"):
        api.call("writer", "Instructions", {}, None, "wrong-serving")
    assert api.dispatch_stopped.is_set()
    assert api.costs()["actual_complete_usd"] == 0.1


def test_large_feedback_is_preserved_without_a_byte_context_rejection(tmp_path):
    supplied = "word " * 100_000

    def handler(request):
        payload = json.loads(request.content)
        assert json.loads(payload["messages"][1]["content"])["feedback"] == supplied
        assert payload["transforms"] == []
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "usage": {"cost": 0.1},
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            },
        )

    api = mock_api(tmp_path, "optimizer", handler)
    assert len(supplied.encode()) > api.endpoint["optimizer"]["context_length"]
    assert api.call("optimizer", "Instructions", {"feedback": supplied}, None, "large")["content"] == "ok"
