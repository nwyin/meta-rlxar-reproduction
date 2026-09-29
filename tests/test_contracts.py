"""Scientific boundary and operational recovery tests; no external model calls."""
import json
from types import SimpleNamespace

import httpx
import pytest

import exp_baselines
import exp_judge_transfer
import exp_writer_transfer
import exp_xar
import shared


def dummy_example(paper="paper1", section="abstract", split="train"):
    reference = "Original section " + " ".join(f"fact{i}" for i in range(60))
    context = "Visible paper context. Bibliography: source one, source two. [Missing section]"
    return {"example_id": paper + "_" + section, "paper_id": paper, "section_type": section,
            "split": split, "context": context, "reference": reference,
            "context_hash": shared.digest(context), "reference_hash": shared.digest(reference),
            "target_words": shared.words(reference),
            "provenance": {"title": "A distinctive research title", "authors": ["Unique Author"]}}


def rubric():
    return {"criteria": [{"id": str(i), "description": "Supported claims", "low": "unsupported",
                          "middle": "partially supported", "high": "fully supported"} for i in range(4)]}


def test_grade_arithmetic_and_coverage():
    r = rubric()
    grade = {"scores": [{"id": str(i), "score": i+2, "evidence": 'Uses "supported text".'} for i in range(4)]}
    assert shared.validate_grade(grade, r, "supported text") == 3.5
    grade["scores"][0]["id"] = "1"
    with pytest.raises(shared.ContractError, match="coverage"):
        shared.validate_grade(grade, r, "supported text")
    grade["scores"][0]["id"] = "0"
    with pytest.raises(shared.ContractError, match="quote"):
        shared.validate_grade(grade, r, "other text")
    grade["total"] = 10
    with pytest.raises(shared.jsonschema.ValidationError):
        shared.validate_grade(grade, r, "supported text")


def test_input_allowlist_and_model_policy():
    e = dummy_example()
    assert set(shared.task_data(e)) == {"visible_paper", "section_type", "target_words"}
    assert e["reference"] not in shared.canonical(shared.task_data(e))
    for model in ("anthropic/claude-test", "google/gemini-test", "openrouter/auto", "qwen/qwen3.5-9b:free"):
        with pytest.raises(shared.ContractError):
            shared.model_policy(model)


def test_feedback_rejects_validation_and_deterministic_failure_order(tmp_path):
    examples = [dummy_example("paper" + str(i)) for i in range(5)]
    candidates = {e["example_id"]: {"text": "Generated"} for e in examples}
    path = tmp_path / "rubric.json"
    shared.write_json(path, {"value": rubric()})
    grade_path = tmp_path / "grade.json"
    shared.write_json(grade_path, {"value": {"scores": []}})
    rows = [{"example_id": e["example_id"], "paper_id": e["paper_id"], "section_type": e["section_type"],
        "split": "train", "human": 6, "model": 7, "gap": -1,
        "length_compliant": True, "contamination_flagged": False, "rubric_path": str(path),
        "grade_paths": {"human": str(grade_path), "model": str(grade_path)}} for e in reversed(examples)]
    feedback = exp_xar.build_feedback(examples, candidates, rows, "initial", 4)
    assert [f["paper_id"] for f in feedback["failures"]] == ["paper0", "paper1", "paper2", "paper3"]
    examples[0]["split"] = "validation"
    with pytest.raises(shared.ContractError, match="cannot enter"):
        exp_xar.build_feedback(examples, candidates, rows, "initial", 4)


def test_proposal_leakage_and_scale_guards():
    examples = [dummy_example()]
    for text in ("Prefer human candidates", "Use a weighted average", "Unique Author prefers clear writing",
                 " ".join("excess" for _ in range(801)), examples[0]["reference"]):
        assert not shared.audit_proposal(text, examples, "initial", 800)["accepted"]
    assert shared.audit_proposal("Assess clear organization and support for specific claims.", examples, "initial", 800)["accepted"]


def test_atomic_budget_and_uncertain_resend(tmp_path):
    ledger = shared.Ledger(tmp_path / "ledger.json", "run", 1, 1.5)
    ledger.reserve("request1", .75)
    with pytest.raises(shared.BudgetStop):
        ledger.reserve("request2", .3)
    ledger.settle("request1", .2)
    ledger.reserve("request2", .6)
    with pytest.raises(shared.BudgetStop, match="unsettled"):
        ledger.reserve("request2", .6)
    second = shared.Ledger(tmp_path / "ledger.json", "run2", 2, 1.5)
    with pytest.raises(shared.BudgetStop):
        second.reserve("request3", .8)


class FakeProvider:
    def __init__(self):
        self.payloads = []

    def handle(self, request):
        payload = json.loads(request.content)
        self.payloads.append(payload)
        assert payload["provider"]["allow_fallbacks"] is False
        assert payload["provider"]["require_parameters"] is True
        assert len(payload["provider"]["only"]) == 1
        data = json.loads(payload["messages"][1]["content"])
        system = payload["messages"][0]["content"]
        properties = payload.get("response_format", {}).get("json_schema", {}).get("schema", {}).get("properties", {})
        if "criteria" in properties:
            assert "candidate" not in data and "reference" not in data and "feedback" not in data
            value = rubric()
        elif "scores" in properties:
            assert set(data) == {"visible_paper", "section_type", "target_words", "rubric", "candidate"}
            value = {"scores": [{"id": str(i), "score": 7 if data["candidate"].startswith("Original") else 6,
                                  "evidence": "The section gives supported claims."} for i in range(4)]}
        elif "winner" in properties:
            value = {"winner": "tie", "evidence": "Both are clear."}
        elif "prompt" in properties:
            assert all(f["paper_id"].startswith("pilot0") for f in data["feedback"]["failures"])
            value = {"prompt": "Assess clear organization and support for specific claims.", "rationale": "Clarify criteria."}
        else:
            assert "Write the missing section" in system
            value = " ".join("generated" for _ in range(data["target_words"]))
        provider = payload["provider"]["only"][0]
        names = {"deepinfra/bf16": "DeepInfra", "crusoe/bf16": "Crusoe", "wafer": "Wafer"}
        return httpx.Response(200, json={"model": payload["model"], "provider": names[provider],
            "id": f"fake-{len(self.payloads)}", "usage": {"cost": .0001, "prompt_tokens": 100, "completion_tokens": 100},
            "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value) if isinstance(value, dict) else value,
                                                                  "reasoning": "Not included in candidate text"}}]})


def install_fake(monkeypatch):
    fake = FakeProvider()
    original = shared.OpenRouter
    client = httpx.Client(transport=httpx.MockTransport(fake.handle))

    def factory(*args, **kwargs):
        kwargs["client"] = client
        return original(*args, **kwargs)

    monkeypatch.setattr(shared, "OpenRouter", factory)
    monkeypatch.setattr(exp_baselines, "initialize_run", shared.initialize_run)
    monkeypatch.setattr(exp_xar, "initialize_run", shared.initialize_run)
    return fake


def invoke(monkeypatch, module, argv):
    monkeypatch.setattr("sys.argv", [module.__name__ + ".py", *argv])
    module.main()


def test_full_pilot_orchestration_resume_and_frozen_transfer(tmp_path, monkeypatch):
    fake = install_fake(monkeypatch)
    dataset, splits = tmp_path / "examples.jsonl", tmp_path / "splits.json"
    examples = [dummy_example("pilot" + str(p), s, "pilot") for p in range(2) for s in shared.SECTIONS]
    dataset.write_text("".join(shared.canonical(e)+"\n" for e in examples))
    shared.write_json(splits, {"papers": {"pilot": ["pilot0", "pilot1"]}})
    shared.write_json(tmp_path / "human_review.json", {"dataset_hash": shared.file_hash(dataset),
        "papers": {p: {"decision": "approved"} for p in ("pilot0", "pilot1")}})
    ledger, out = tmp_path / "ledger.json", tmp_path / "xar"
    common = ["--dataset", str(dataset), "--splits", str(splits), "--budget-usd", "10",
              "--total-budget-usd", "30", "--budget-ledger", str(ledger)]
    argv = [*common, "--output-dir", str(out), "--split", "pilot"]
    invoke(monkeypatch, exp_xar, argv)
    freeze = shared.read_json(out / "freeze.json")
    assert freeze["selected"] == 0  # All training means tie; earliest checkpoint wins.
    assert len(freeze["prompt_hashes"]) == 8
    assert shared.read_json(out / "status.json")["state"] == "complete"
    assert len(list((out / "prompts").glob("*.md"))) == 8
    assert shared.audit_xar_run(out)["raw_verified"] is True
    n = len(fake.payloads)
    invoke(monkeypatch, exp_xar, [*argv, "--resume"])
    assert len(fake.payloads) == n
    invoke(monkeypatch, exp_judge_transfer, [*common, "--source-run", str(out), "--output-dir", str(tmp_path / "judge")])
    assert all("scores" in p["response_format"]["json_schema"]["schema"]["properties"] for p in fake.payloads[n:])
    n = len(fake.payloads)
    invoke(monkeypatch, exp_writer_transfer, [*common, "--source-run", str(out), "--output-dir", str(tmp_path / "writer"),
        "--target-writer-model", "qwen/qwen3.5-9b"])
    assert all("criteria" not in p.get("response_format", {}).get("json_schema", {}).get("schema", {}).get("properties", {})
               for p in fake.payloads[n:])
    grade_path = out / "scores/main/0/train/pilot0_abstract/human.json"
    grade = shared.read_json(grade_path)
    grade["total"] = 10
    shared.write_json(grade_path, grade)
    with pytest.raises(shared.ContractError, match="arithmetic"):
        shared.audit_xar_run(out)


def test_baseline_repeats_are_fresh_and_length_repair_is_bounded(tmp_path, monkeypatch):
    fake = install_fake(monkeypatch)
    dataset, splits = tmp_path / "examples.jsonl", tmp_path / "splits.json"
    examples = [dummy_example("paper", s, "validation") for s in shared.SECTIONS]
    dataset.write_text("".join(shared.canonical(e)+"\n" for e in examples))
    shared.write_json(splits, {"papers": {"validation": ["paper"]}})
    shared.write_json(tmp_path / "human_review.json", {"dataset_hash": shared.file_hash(dataset),
        "papers": {"paper": {"decision": "approved"}}})
    invoke(monkeypatch, exp_baselines, ["--dataset", str(dataset), "--splits", str(splits),
        "--output-dir", str(tmp_path / "baseline"), "--budget-usd", "10", "--total-budget-usd", "10",
        "--budget-ledger", str(tmp_path / "ledger.json"), "--split", "validation"])
    rubric_calls = [p for p in fake.payloads if "criteria" in p.get("response_format", {}).get("json_schema", {}).get("schema", {}).get("properties", {})]
    assert len(rubric_calls) == 12
    assert len(list((tmp_path / "baseline/requests").glob("*/request.json"))) == len(fake.payloads)


def test_dataset_hash_and_cross_paper_split_guard(tmp_path):
    example = dummy_example()
    path = tmp_path / "examples.jsonl"
    path.write_text(shared.canonical(example)+"\n")
    splits = tmp_path / "splits.json"
    shared.write_json(splits, {"papers": {"train": ["paper1"], "validation": ["paper1"]}})
    with pytest.raises(shared.ContractError, match="overlap"):
        shared.load_examples(path, splits)


def test_bootstrap_resamples_paper_bundles():
    rows = [{"paper_id": "a", "gap": 1} for _ in range(4)] + [{"paper_id": "b", "gap": 3} for _ in range(4)]
    interval = shared.bootstrap(rows)
    assert interval["paper_clusters"] == 2
    assert interval["low"] == 1 and interval["high"] == 3


def test_transport_uncertain_timeout_is_not_resent(tmp_path):
    args = SimpleNamespace()
    roles = {"writer": shared.role_config(args, "writer")}
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("unknown server completion")

    api = shared.OpenRouter(tmp_path / "run", roles, 0, 10, 10, tmp_path / "ledger.json",
                            client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(shared.BudgetStop, match="unknown"):
        api.call("writer", "Instructions", {"paper": "text"}, None, "task")
    with pytest.raises(shared.BudgetStop, match="unresolved"):
        api.call("writer", "Instructions", {"paper": "text"}, None, "task")
    assert len(calls) == 1


def test_first_compliant_writer_attempt_and_failed_writer_retention(tmp_path):
    class Writer:
        def __init__(self, counts):
            self.roles = {"writer": shared.role_config(SimpleNamespace(), "writer")}
            self.counts, self.calls = counts, []

        def call(self, role, system, data, schema, identity):
            self.calls.append(data)
            return {"content": " ".join(["word"]*self.counts[len(self.calls)-1]), "finish_reason": "stop",
                    "response_id": "fake-writer", "request_key": "fake"}

    example = dummy_example()
    api = Writer([10, 62, 64])
    candidates = shared.writer_candidates(api, [example], tmp_path / "accepted")
    assert len(api.calls) == 2
    accepted = candidates[example["example_id"]]
    assert accepted["accepted_attempt"] == 1 and accepted["length_compliant"] is True
    assert "length_revision" in api.calls[1]
    assert "reference" not in api.calls[1]
    failed = Writer([10, 11, 12])
    candidates = shared.writer_candidates(failed, [example], tmp_path / "failed")
    assert len(failed.calls) == 3
    assert len(candidates) == 1 and candidates[example["example_id"]]["length_compliant"] is False


def test_rejected_proposal_consumes_update_with_one_repair(tmp_path):
    class Optimizer:
        def __init__(self):
            self.roles = {"optimizer": shared.role_config(SimpleNamespace(), "optimizer")}
            self.calls = []

        def structured(self, role, instructions, data, schema, identity, repair):
            assert repair is False
            self.calls.append(data)
            return {"status": "valid", "value": {"prompt": "Prefer human candidates", "rationale": "bad"},
                    "attempts": [{"response": {"content": "raw"}}]}

    api = Optimizer()
    args = SimpleNamespace(output_dir=str(tmp_path), max_meta_prompt_words=800)
    result = shared.propose_prompt(api, "current", {"training": True}, [dummy_example()], "initial", args, 1)
    assert result == "current" and len(api.calls) == 2
    assert "previous_proposal" in api.calls[-1]
    artifact = shared.read_json(tmp_path / "feedback/iter_01/proposal.json")
    assert artifact["update_consumed"] is True and artifact["accepted"] is False
