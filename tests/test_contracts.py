"""Scientific boundary and operational recovery tests; no external model calls."""

import json
from types import SimpleNamespace

import httpx
import pytest

import exp_xar
import shared


def dummy_example(paper="paper1", section="abstract", split="train"):
    reference = "Original section " + " ".join(f"fact{i}" for i in range(60))
    context = "Visible paper context. Bibliography: source one, source two. [Missing section]"
    return {
        "example_id": paper + "_" + section,
        "paper_id": paper,
        "section_type": section,
        "split": split,
        "context": context,
        "reference": reference,
        "context_hash": shared.digest(context),
        "reference_hash": shared.digest(reference),
        "target_words": shared.words(reference),
        "provenance": {"title": "A distinctive research title", "authors": ["Unique Author"]},
    }


def rubric():
    return {
        "criteria": [
            {
                "id": str(i),
                "description": "Supported claims",
                "low": "unsupported",
                "middle": "partially supported",
                "high": "fully supported",
            }
            for i in range(4)
        ]
    }


def test_grade_arithmetic_and_coverage():
    r = rubric()
    grade = {
        "scores": [{"id": str(i), "score": i + 2, "evidence": 'Uses "supported text".'} for i in range(4)]
    }
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
    feedback = exp_xar.build_feedback(examples, candidates, rows, "initial", 4)
    assert [f["paper_id"] for f in feedback["failures"]] == ["paper0", "paper1", "paper2", "paper3"]
    examples[0]["split"] = "validation"
    with pytest.raises(shared.ContractError, match="cannot enter"):
        exp_xar.build_feedback(examples, candidates, rows, "initial", 4)


def test_proposal_leakage_and_scale_guards():
    examples = [dummy_example()]
    for text in (
        "Prefer human candidates",
        "Use a weighted average",
        "Unique Author prefers clear writing",
        " ".join("excess" for _ in range(801)),
        examples[0]["reference"],
    ):
        assert not shared.audit_proposal(text, examples, "initial", 800)["accepted"]
    assert shared.audit_proposal(
        "Assess clear organization and support for specific claims.", examples, "initial", 800
    )["accepted"]


def test_atomic_budget_and_uncertain_resend(tmp_path):
    ledger = shared.Ledger(tmp_path / "ledger.json", "run", 1, 1.5)
    ledger.reserve("request1", 0.75)
    with pytest.raises(shared.BudgetStop):
        ledger.reserve("request2", 0.3)
    ledger.settle("request1", 0.2)
    ledger.reserve("request2", 0.6)
    with pytest.raises(shared.BudgetStop, match="unsettled"):
        ledger.reserve("request2", 0.6)
    second = shared.Ledger(tmp_path / "ledger.json", "run2", 2, 1.5)
    with pytest.raises(shared.BudgetStop):
        second.reserve("request3", 0.8)


def test_overspend_is_saved_before_budget_stop(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = shared.Ledger(path, "run", 10, 10)
    ledger.reserve("request", 1.0)
    with pytest.raises(shared.BudgetStop, match="reservation"):
        ledger.settle("request", 2.0)
    entry = shared.read_json(path)["entries"]["request"]
    assert entry["state"] == "pricing_bound_violation"
    assert entry["charge"] == 2.0
    # Reading the summary does not rewrite the ledger.
    before = path.stat().st_ino  # a save replaces the file with a new one
    assert ledger.summary()["charged_or_reserved_usd"] == 2.0
    assert path.stat().st_ino == before


class FakeProvider:
    def __init__(self):
        self.payloads = []

    def handle(self, request):
        if request.method == "GET":
            if request.url.path.endswith("/endpoints"):
                model = request.url.path.split("/models/")[-1].removesuffix("/endpoints")
                return httpx.Response(
                    200,
                    json=shared.read_json(
                        shared.ROOT / "configs/snapshots" / (model.replace("/", "_") + "-endpoints.json")
                    ),
                )
            return httpx.Response(
                200,
                json=shared.read_json(shared.ROOT / "configs/snapshots/openrouter-models-2026-09-29.json"),
            )
        payload = json.loads(request.content)
        self.payloads.append(payload)
        assert payload["provider"]["allow_fallbacks"] is False
        assert payload["provider"]["require_parameters"] is True
        assert len(payload["provider"]["only"]) == 1
        data = json.loads(payload["messages"][1]["content"])
        system = payload["messages"][0]["content"]
        properties = (
            payload.get("response_format", {}).get("json_schema", {}).get("schema", {}).get("properties", {})
        )
        if "criteria" in properties:
            assert "candidate" not in data and "reference" not in data and "feedback" not in data
            value = rubric()
        elif "scores" in properties:
            assert set(data) == {"visible_paper", "section_type", "target_words", "rubric", "candidate"}
            value = {
                "scores": [
                    {
                        "id": str(i),
                        "score": 7 if data["candidate"].startswith("Original") else 6,
                        "evidence": "The section gives supported claims.",
                    }
                    for i in range(4)
                ]
            }
        elif "prompt" in properties:
            assert all(f["paper_id"].startswith("pilot0") for f in data["feedback"]["failures"])
            value = {
                "prompt": "Assess clear organization and support for specific claims.",
                "rationale": "Clarify criteria.",
            }
        else:
            assert "Write the missing section" in system
            value = " ".join("generated" for _ in range(data["target_words"]))
        provider = payload["provider"]["only"][0]
        names = {"siliconflow/fp8": "SiliconFlow", "meta": "Meta"}
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "provider": names[provider],
                "id": f"fake-{len(self.payloads)}",
                "usage": {"cost": 0.0001, "prompt_tokens": 100, "completion_tokens": 100},
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(value) if isinstance(value, dict) else value,
                            "reasoning": "Not included in candidate text",
                        },
                    }
                ],
            },
        )


def install_fake(monkeypatch):
    fake = FakeProvider()
    original = shared.OpenRouter
    client = httpx.Client(transport=httpx.MockTransport(fake.handle))

    def factory(*args, **kwargs):
        kwargs["client"] = client
        return original(*args, **kwargs)

    monkeypatch.setattr(shared, "OpenRouter", factory)
    monkeypatch.setattr(exp_xar, "initialize_run", shared.initialize_run)
    return fake


def invoke(monkeypatch, module, argv):
    monkeypatch.setattr("sys.argv", [module.__name__ + ".py", *argv])
    module.main()


def test_full_pilot_orchestration_and_resume(tmp_path, monkeypatch):
    fake = install_fake(monkeypatch)
    dataset, splits = tmp_path / "examples.jsonl", tmp_path / "splits.json"
    examples = [dummy_example("pilot" + str(p), s, "pilot") for p in range(2) for s in shared.SECTIONS]
    dataset.write_text("".join(shared.canonical(e) + "\n" for e in examples))
    shared.write_json(splits, {"papers": {"pilot": ["pilot0", "pilot1"]}})
    shared.write_json(
        tmp_path / "human_review.json",
        {
            "dataset_hash": shared.file_hash(dataset),
            "papers": {p: {"decision": "approved"} for p in ("pilot0", "pilot1")},
        },
    )
    ledger, out = tmp_path / "ledger.json", tmp_path / "xar"
    common = [
        "--dataset",
        str(dataset),
        "--splits",
        str(splits),
        "--budget-usd",
        "10",
        "--total-budget-usd",
        "30",
        "--budget-ledger",
        str(ledger),
    ]
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
    # Forgetting --resume fails before preflight writes anything into the run.
    preflights = sorted((out / "preflight").iterdir())
    with pytest.raises(shared.ContractError, match="Run exists"):
        invoke(monkeypatch, exp_xar, argv)
    assert sorted((out / "preflight").iterdir()) == preflights
    response_path = next((out / "scores/main/0/train").glob("*/human.json"))
    response = shared.read_json(response_path)["attempts"][-1]["response"]
    request_path = shared.Path(response["raw_response"]).parent / "request.json"
    request = shared.read_json(request_path)
    original = request["payload"]["temperature"]
    request["payload"]["temperature"] = 0.9
    shared.write_json(request_path, request)
    with pytest.raises(shared.ContractError, match="decoding contract"):
        shared.audit_xar_run(out)
    request["payload"]["temperature"] = original
    shared.write_json(request_path, request)
    grade_path = out / "scores/main/0/train/pilot0_abstract/human.json"
    grade = shared.read_json(grade_path)
    grade["total"] = 10
    shared.write_json(grade_path, grade)
    with pytest.raises(shared.ContractError, match="arithmetic"):
        shared.audit_xar_run(out)


def test_dataset_hash_and_cross_paper_split_guard(tmp_path):
    example = dummy_example()
    path = tmp_path / "examples.jsonl"
    path.write_text(shared.canonical(example) + "\n")
    splits = tmp_path / "splits.json"
    shared.write_json(splits, {"papers": {"train": ["paper1"], "validation": ["paper1"]}})
    with pytest.raises(shared.ContractError, match="overlap"):
        shared.load_examples(path, splits)


def test_bootstrap_resamples_paper_bundles():
    rows = [{"paper_id": "a", "gap": 1} for _ in range(4)] + [{"paper_id": "b", "gap": 3} for _ in range(4)]
    interval = shared.bootstrap(rows)
    assert interval["paper_clusters"] == 2
    assert interval["low"] == 1 and interval["high"] == 3
    unequal = shared.bootstrap([{"paper_id": "a", "gap": 0}] + [{"paper_id": "b", "gap": 10}] * 4)
    assert unequal["estimate"] == 8 and unequal["bootstrap_median"] == 8


def test_transport_uncertain_timeout_is_not_resent(tmp_path):
    roles = {"writer": shared.role_config("writer")}
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("unknown server completion")

    api = shared.OpenRouter(
        tmp_path / "run",
        roles,
        0,
        10,
        10,
        tmp_path / "ledger.json",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(shared.BudgetStop, match="unknown"):
        api.call("writer", "Instructions", {"paper": "text"}, None, "task")
    with pytest.raises(shared.BudgetStop, match="unresolved"):
        api.call("writer", "Instructions", {"paper": "text"}, None, "task")
    assert len(calls) == 1


def test_first_compliant_writer_attempt_and_failed_writer_retention(tmp_path):
    class Writer:
        def __init__(self, counts):
            self.roles = {"writer": shared.role_config("writer")}
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
            self.roles = {"optimizer": shared.role_config("optimizer")}
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
    result = shared.propose_prompt(
        api, "current", {"training": True}, [dummy_example()], "initial", 1, tmp_path, 800
    )
    assert result == "current" and len(api.calls) == 2
    assert "previous_proposal" in api.calls[-1]
    artifact = shared.read_json(tmp_path / "feedback/iter_01/proposal.json")
    assert artifact["update_consumed"] is True and artifact["accepted"] is False


def test_primary_matrix_rejects_different_judge_and_protocol():
    design = shared.yaml.safe_load((shared.ROOT / "configs/experiments.yaml").read_text())
    manifest = {
        "roles": {r: shared.role_config(r) for r in ("writer", "rubric", "optimizer", "judge")},
        "arguments": {k: design[k] for k in ("iterations", "max_meta_prompt_words", "failure_examples")},
        "extra": {"initial_meta_prompt_hash": shared.digest(shared.prompt("rubric_initial"))},
    }
    shared.validate_primary_manifest(manifest, design)
    changed = json.loads(json.dumps(manifest))
    changed["roles"]["judge"] = shared.role_config("judge", "moonshotai/kimi-k2.6")
    with pytest.raises(shared.ContractError, match="main judge fixed"):
        shared.validate_primary_manifest(changed, design)
    changed = json.loads(json.dumps(manifest))
    changed["arguments"]["iterations"] = 6
    with pytest.raises(shared.ContractError, match="iterations"):
        shared.validate_primary_manifest(changed, design)
    changed = json.loads(json.dumps(manifest))
    changed["roles"]["rubric"]["reasoning"] = {"enabled": False}
    with pytest.raises(shared.ContractError, match="decoding"):
        shared.validate_primary_manifest(changed, design)


def test_parallel_writer_sampling_repairs_order_and_resume(tmp_path):
    import threading

    examples = [dummy_example("parallel" + str(i)) for i in range(4)]

    class ParallelWriter:
        def __init__(self):
            self.roles = {"writer": shared.role_config("writer")}
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
    result = shared.writer_candidates(api, examples, tmp_path, concurrency=2)
    assert api.peak == 2
    assert list(result) == [e["example_id"] for e in examples]
    for e in examples:
        calls = api.calls[e["example_id"]]
        assert [identity["attempt"] for _, identity in calls] == [0, 1]
        assert "length_revision" not in calls[0][0]
        assert calls[1][0]["previous_section"] == " ".join(["word"] * 10)
        assert result[e["example_id"]]["accepted_attempt"] == 1
        assert result[e["example_id"]]["length_compliant"]
    before = shared.canonical(api.calls)
    assert shared.writer_candidates(api, examples, tmp_path, concurrency=2) == result
    assert shared.canonical(api.calls) == before


def test_parallel_writer_failure_stops_new_dispatch_and_preserves_sections(tmp_path):
    import threading

    examples = [dummy_example("interrupted" + str(i)) for i in range(4)]

    class InterruptedWriter:
        def __init__(self, fail):
            self.roles = {"writer": shared.role_config("writer")}
            self.calls = []
            self.fail = fail
            self.barrier = threading.Barrier(2)

        def call(self, role, system, data, schema, identity):
            eid = identity["example"]
            self.calls.append(eid)
            if self.fail:
                self.barrier.wait(timeout=5)
                if eid == examples[0]["example_id"]:
                    raise shared.BudgetStop("Budget exhausted before send")
            return {
                "content": " ".join(["word"] * data["target_words"]),
                "finish_reason": "stop",
                "response_id": eid,
                "request_key": eid,
            }

    failed = InterruptedWriter(True)
    with pytest.raises(shared.BudgetStop, match="before send"):
        shared.writer_candidates(failed, examples, tmp_path, concurrency=2)
    assert set(failed.calls) == {e["example_id"] for e in examples[:2]}
    assert (tmp_path / "generations" / (examples[1]["example_id"] + ".json")).exists()
    assert not (tmp_path / "generations/candidates.json").exists()
    resumed = InterruptedWriter(False)
    candidates = shared.writer_candidates(resumed, examples, tmp_path, concurrency=2)
    assert len(candidates) == 4
    assert examples[1]["example_id"] not in resumed.calls
    assert len(resumed.calls) == 3


def test_parallel_reservations_share_one_budget(tmp_path):
    import concurrent.futures

    path = tmp_path / "global_ledger.json"

    def reserve(i):
        ledger = shared.Ledger(path, str(i), 1, 1)
        try:
            ledger.reserve(str(i), 0.4)
            return True
        except shared.BudgetStop:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(8))) == 2
    entries = shared.read_json(path)["entries"]
    assert sum(e["charge"] for e in entries.values()) == 0.8


@pytest.mark.parametrize("factor,accepted", [(0.5, True), (1.1, True), (1.3, False)])
def test_preflight_pricing_changes_stay_within_frozen_bounds(tmp_path, factor, accepted):
    cfg = shared.role_config("judge")
    endpoint, _ = shared.endpoint_for(cfg)

    def handler(request):
        if request.url.path.endswith("/endpoints"):
            data = shared.read_json(
                shared.ROOT / "configs/snapshots" / (cfg["model"].replace("/", "_") + "-endpoints.json")
            )
            for e in data["data"]["endpoints"]:
                if e["tag"] == cfg["provider"]:
                    e["pricing"]["completion"] = str(float(e["pricing"]["completion"]) * factor)
        else:
            data = shared.read_json(shared.ROOT / "configs/snapshots/openrouter-models-2026-09-29.json")
        return httpx.Response(200, json=data)

    api = shared.OpenRouter(
        tmp_path / "run",
        {"judge": cfg},
        0,
        10,
        10,
        tmp_path / "ledger.json",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    if accepted:
        directory = api.preflight()
        bounds = shared.read_json(directory / "checks.json")["pricing_bounds"]["judge"]
        assert bounds["lower_prices_within_bound"] is (factor < 1)
        assert bounds["frozen_upper"] == shared.pricing_bound(endpoint)
        assert api.endpoints["judge"][0] == endpoint
    else:
        with pytest.raises(shared.ContractError, match="exceeds frozen upper"):
            api.preflight()
    assert not (tmp_path / "ledger.json").exists()


def test_unknown_send_halts_dispatch_for_other_tasks(tmp_path):
    cfg = shared.role_config("writer")
    sends = []

    def handler(request):
        sends.append(request)
        raise httpx.ReadTimeout("Completion outcome unknown")

    api = shared.OpenRouter(
        tmp_path / "run",
        {"writer": cfg},
        0,
        10,
        10,
        tmp_path / "ledger.json",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(shared.BudgetStop, match="unknown"):
        api.call("writer", "Instructions", {"paper": "first"}, None, "first")
    with pytest.raises(shared.ContractError, match="Dispatch halted"):
        api.call("writer", "Instructions", {"paper": "second"}, None, "second")
    assert len(sends) == 1
    assert len(shared.read_json(tmp_path / "ledger.json")["entries"]) == 1


def test_blog_model_contract_rejects_every_role_substitution():
    design = shared.yaml.safe_load((shared.ROOT / "configs/experiments.yaml").read_text())
    manifest = {
        "roles": {r: shared.role_config(r) for r in ("writer", "rubric", "optimizer", "judge")},
        "arguments": {k: design[k] for k in ("iterations", "max_meta_prompt_words", "failure_examples")},
        "extra": {"initial_meta_prompt_hash": shared.digest(shared.prompt("rubric_initial"))},
    }
    shared.validate_primary_manifest(manifest, design)
    for role in ("writer", "rubric", "optimizer", "judge"):
        # Swap in the other model: Muse for the optimizer, Kimi everywhere else.
        other = "meta/muse-spark-1.1" if role == "optimizer" else "moonshotai/kimi-k2.6"
        changed = json.loads(json.dumps(manifest))
        changed["roles"][role] = shared.role_config(role, other)
        with pytest.raises(shared.ContractError):
            shared.validate_primary_manifest(changed, design)


def test_blog_driver_excludes_sweep_and_old_writer_cache():
    import run_matrix

    design = shared.yaml.safe_load((shared.ROOT / "configs/experiments.yaml").read_text())
    args = SimpleNamespace(
        runs_root="runs", concurrency=4, budget_usd=100, total_budget_usd=100, dry_run=True, resume=False
    )
    for phase, split, iterations in [("pilot", "pilot", "1"), ("reproduction", "research", "7")]:
        command = run_matrix.build_command(args, design, phase)
        assert command[command.index("--split") + 1] == split
        assert command[command.index("--iterations") + 1] == iterations
        assert command[command.index("--seed") + 1] == "0"
        for role in ("writer", "rubric", "judge"):
            assert command[command.index(f"--{role}-model") + 1] == "meta/muse-spark-1.1"
        assert command[command.index("--optimizer-model") + 1] == "moonshotai/kimi-k2.6"


def test_blog_report_does_not_count_historical_runs(tmp_path):
    historical = tmp_path / "runs/pilot-schema-GG2"
    shared.write_json(historical / "manifest.json", {"experiment": "xar", "arguments": {"split": "pilot"}})
    output = tmp_path / "report"
    shared.render_report(tmp_path / "runs", output)
    audit = shared.read_json(output / "audit.json")
    assert audit["expected_trajectories"] == 1
    assert audit["raw_verified_trajectories"] == 0 and audit["state"] == "not_started"
    assert shared.read_json(output / "blog_comparison.json")["reproduction"] is None
    assert not (output / "gap_curves.png").exists()


def test_blog_research_report_rebuilds_all_52_sections_and_rejects_tampering(tmp_path, monkeypatch):
    fake = install_fake(monkeypatch)
    design = shared.yaml.safe_load((shared.ROOT / "configs/experiments.yaml").read_text())
    groups = {
        "train": [f"pilot0train{i}" for i in range(8)],
        "validation": [f"validation{i}" for i in range(5)],
    }
    examples = [
        dummy_example(p, section, split)
        for split, papers in groups.items()
        for p in papers
        for section in shared.SECTIONS
    ]
    dataset, splits = tmp_path / "examples.jsonl", tmp_path / "splits.json"
    dataset.write_text("".join(shared.canonical(e) + "\n" for e in examples))
    shared.write_json(splits, {"papers": groups})
    shared.write_json(
        tmp_path / "human_review.json",
        {
            "dataset_hash": shared.file_hash(dataset),
            "papers": {e["paper_id"]: {"decision": "approved"} for e in examples},
        },
    )
    runs = tmp_path / "runs"
    source = runs / design["research_run"]
    invoke(
        monkeypatch,
        exp_xar,
        [
            "--dataset",
            str(dataset),
            "--splits",
            str(splits),
            "--output-dir",
            str(source),
            "--budget-ledger",
            str(runs / "ledger.json"),
            "--budget-usd",
            "100",
            "--total-budget-usd",
            "100",
            "--concurrency",
            "4",
        ],
    )
    assert len(fake.payloads) == 1307
    assert {p["model"] for p in fake.payloads} == {"meta/muse-spark-1.1", "moonshotai/kimi-k2.6"}
    output = tmp_path / "report"
    shared.render_report(runs, output)
    assert shared.read_json(output / "audit.json")["raw_verified_trajectories"] == 1
    comparison = shared.read_json(output / "blog_comparison.json")["reproduction"]
    assert comparison["selected_iteration"] == 0
    assert comparison["initial_gap"] == 1 and comparison["selected_gap"] == 1
    assert comparison["descriptive_reversal"] is False
    assert comparison["validation_peak_used_for_selection"] is False
    assert comparison["paired_improvement"]["interval"]["paper_clusters"] == 5
    assert (output / "gap_curves.svg").exists()
    grade_path = next((source / "scores/main/0/validation").glob("*/human.json"))
    grade = shared.read_json(grade_path)
    grade["total"] = 10
    shared.write_json(grade_path, grade)
    shared.render_report(runs, output)
    audit = shared.read_json(output / "audit.json")
    assert audit["raw_verified_trajectories"] == 0 and audit["complete"] is False
    assert shared.read_json(output / "blog_comparison.json")["reproduction"] is None
    assert not (output / "gap_curves.svg").exists()
