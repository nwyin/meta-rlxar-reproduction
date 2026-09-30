"""Full pilot and research runs against a fake provider, including tamper detection."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from conftest import dummy_example, install_fake, invoke

import exp_xar
from xar.audit import audit_xar_run
from xar.data import SECTIONS
from xar.report import render_report
from xar.util import ROOT, RunError, canonical, file_hash, read_json, write_json


def test_full_pilot_orchestration_and_resume(tmp_path, monkeypatch):
    fake = install_fake(monkeypatch)
    dataset, splits = tmp_path / "examples.jsonl", tmp_path / "splits.json"
    examples = [dummy_example("pilot" + str(p), s, "pilot") for p in range(2) for s in SECTIONS]
    dataset.write_text("".join(canonical(e) + "\n" for e in examples))
    write_json(splits, {"papers": {"pilot": ["pilot0", "pilot1"]}})
    write_json(
        tmp_path / "human_review.json",
        {
            "dataset_hash": file_hash(dataset),
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
    freeze = read_json(out / "freeze.json")
    assert freeze["selected"] == 0  # All training means tie; earliest checkpoint wins.
    assert len(freeze["prompt_hashes"]) == 8
    assert read_json(out / "status.json")["state"] == "complete"
    assert len(list((out / "prompts").glob("*.md"))) == 8
    audit_xar_run(out)  # raises if the saved run does not check out
    n = len(fake.payloads)
    invoke(monkeypatch, exp_xar, [*argv, "--resume"])
    assert len(fake.payloads) == n
    # Forgetting --resume fails before preflight writes anything into the run.
    preflights = sorted((out / "preflight").iterdir())
    with pytest.raises(RunError, match="Run exists"):
        invoke(monkeypatch, exp_xar, argv)
    assert sorted((out / "preflight").iterdir()) == preflights
    response_path = next((out / "scores/main/0/train").glob("*/human.json"))
    response = read_json(response_path)["attempts"][-1]["response"]
    request_path = Path(response["raw_response"]).parent / "request.json"
    request = read_json(request_path)
    original = request["payload"]["temperature"]
    request["payload"]["temperature"] = 0.9
    write_json(request_path, request)
    with pytest.raises(RunError, match="decoding contract"):
        audit_xar_run(out)
    request["payload"]["temperature"] = original
    write_json(request_path, request)
    grade_path = out / "scores/main/0/train/pilot0_abstract/human.json"
    grade = read_json(grade_path)
    grade["total"] = 10
    write_json(grade_path, grade)
    with pytest.raises(RunError, match="arithmetic"):
        audit_xar_run(out)


def test_blog_driver_excludes_sweep_and_old_writer_cache():
    import run_matrix

    design = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
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


def test_blog_research_report_rebuilds_all_52_sections_and_rejects_tampering(tmp_path, monkeypatch):
    fake = install_fake(monkeypatch)
    design = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    groups = {
        "train": [f"pilot0train{i}" for i in range(8)],
        "validation": [f"validation{i}" for i in range(5)],
    }
    examples = [
        dummy_example(p, section, split)
        for split, papers in groups.items()
        for p in papers
        for section in SECTIONS
    ]
    dataset, splits = tmp_path / "examples.jsonl", tmp_path / "splits.json"
    dataset.write_text("".join(canonical(e) + "\n" for e in examples))
    write_json(splits, {"papers": groups})
    write_json(
        tmp_path / "human_review.json",
        {
            "dataset_hash": file_hash(dataset),
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
    render_report(runs, output)
    assert read_json(output / "audit.json")["raw_verified_trajectories"] == 1
    comparison = read_json(output / "blog_comparison.json")["reproduction"]
    assert comparison["selected_iteration"] == 0
    assert comparison["initial_gap"] == 1 and comparison["selected_gap"] == 1
    assert comparison["descriptive_reversal"] is False
    assert comparison["validation_peak_used_for_selection"] is False
    assert comparison["paired_improvement"]["interval"]["paper_clusters"] == 5
    assert (output / "gap_curves.svg").exists()
    grade_path = next((source / "scores/main/0/validation").glob("*/human.json"))
    grade = read_json(grade_path)
    grade["total"] = 10
    write_json(grade_path, grade)
    render_report(runs, output)
    audit = read_json(output / "audit.json")
    assert audit["raw_verified_trajectories"] == 0 and audit["complete"] is False
    assert read_json(output / "blog_comparison.json")["reproduction"] is None
    assert not (output / "gap_curves.svg").exists()
