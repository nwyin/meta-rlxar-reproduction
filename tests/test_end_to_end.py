"""Full pilot and research runs against a fake provider, including tamper detection."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import call_kind, install_fake, write_dataset

import run
from xar.audit import audit_xar_run
from xar.data import SECTIONS
from xar.pipeline import run_xar
from xar.report import render_report
from xar.util import ROLES, RunError, read_json, write_json


def run_phase(tmp_path, design, phase, papers_by_split, *flags):
    """Run one phase as `run.py <phase>` would, but on a test dataset under tmp_path."""
    dataset, splits = write_dataset(tmp_path, papers_by_split)
    options = run.parse_args(
        [
            phase,
            "--runs-root",
            str(tmp_path / "runs"),
            "--budget-usd",
            "100",
            "--total-budget-usd",
            "100",
            *flags,
        ]
    )
    settings = replace(run.settings_for(phase, design, options), dataset=str(dataset), splits=str(splits))
    run_xar(settings)
    return settings


@pytest.fixture
def pilot(tmp_path, monkeypatch, design):
    """A completed pilot run (design pilot_iterations updates), and the fake that served it."""
    fake = install_fake(monkeypatch)
    settings = run_phase(tmp_path, design, "pilot", {"pilot": ["pilot0", "pilot1"]})
    return fake, settings


def test_pilot_run_completes_and_resume_makes_no_calls(pilot, design):
    fake, settings = pilot
    out = Path(settings.output_dir)
    freeze = read_json(out / "freeze.json")
    assert freeze["selected"] == 0  # every checkpoint ties on training, and the earliest wins
    assert len(freeze["prompt_hashes"]) == design["pilot_iterations"] + 1
    assert len(list((out / "prompts").glob("*.md"))) == design["pilot_iterations"] + 1
    assert read_json(out / "status.json")["state"] == "complete"
    audit_xar_run(out)  # raises if the saved run does not check out
    sent = len(fake.payloads)
    run_xar(replace(settings, resume=True))
    assert len(fake.payloads) == sent
    # Without --resume the run stops before preflight writes anything into it.
    preflights = sorted((out / "preflight").iterdir())
    with pytest.raises(RunError, match="Run exists"):
        run_xar(settings)
    assert sorted((out / "preflight").iterdir()) == preflights


def test_audit_rejects_changed_request_temperature(pilot):
    out = Path(pilot[1].output_dir)
    grade = read_json(next((out / "scores/main/0/train").glob("*/human.json")))
    request_path = Path(grade["attempts"][-1]["response"]["raw_response"]).parent / "request.json"
    request = read_json(request_path)
    request["payload"]["temperature"] = 0.9
    write_json(request_path, request)
    with pytest.raises(RunError, match="decoding contract"):
        audit_xar_run(out)


def test_audit_rejects_wrong_grade_total(pilot):
    out = Path(pilot[1].output_dir)
    grade_path = out / "scores/main/0/train/pilot0_abstract/human.json"
    grade = read_json(grade_path)
    grade["total"] = 10
    write_json(grade_path, grade)
    with pytest.raises(RunError, match="arithmetic"):
        audit_xar_run(out)


def test_check_pilot_records_the_audited_pilot(pilot, design):
    runs_root = Path(pilot[1].output_dir).parent
    run.check_pilot(runs_root, design)
    manifest = read_json(Path(pilot[1].output_dir) / "manifest.json")
    gate = read_json(runs_root / "meta_blog_pilot_gate.json")
    assert gate == {"pilot_substantive_hash": manifest["substantive_hash"]}


def test_check_pilot_rejects_a_pilot_that_differs_from_the_design(pilot, design):
    out = Path(pilot[1].output_dir)
    manifest = read_json(out / "manifest.json")
    manifest["arguments"]["failure_examples"] += 1
    write_json(out / "manifest.json", manifest)
    with pytest.raises(RunError, match="failure_examples"):
        run.check_pilot(out.parent, design)
    assert not (out.parent / "meta_blog_pilot_gate.json").exists()


@pytest.mark.parametrize(
    "phase, split, iterations_key",
    [("pilot", "pilot", "pilot_iterations"), ("reproduce", "research", "iterations")],
)
def test_settings_follow_the_design(design, phase, split, iterations_key):
    options = run.parse_args([phase, "--concurrency", "3", "--dry-run"])
    settings = run.settings_for(phase, design, options)
    assert settings.split == split
    assert settings.iterations == design[iterations_key]
    assert settings.seed == design["seed"]
    for role in ROLES:
        assert getattr(settings, f"{role}_model") == design[role]
    assert settings.concurrency == 3 and settings.dry_run
    assert run.settings_for(phase, design, run.parse_args([phase])).concurrency == design["concurrency"]


def test_report_from_research_run_and_after_tampering(tmp_path, monkeypatch, design):
    fake = install_fake(monkeypatch)
    # The audit expects the real research size: 8 training and 5 validation papers.
    groups = {
        "train": [f"train{i}" for i in range(8)],
        "validation": [f"validation{i}" for i in range(5)],
    }
    settings = run_phase(tmp_path, design, "reproduce", groups, "--concurrency", "4")
    source, runs = Path(settings.output_dir), tmp_path / "runs"
    sections = len(SECTIONS) * sum(len(papers) for papers in groups.values())
    checkpoints = design["iterations"] + 1
    assert Counter(call_kind(p) for p in fake.payloads) == {
        "writer": sections,
        "rubric": sections * checkpoints,
        "grade": 2 * sections * checkpoints,
        "optimizer": design["iterations"],
    }
    assert {p["model"] for p in fake.payloads} == {"meta/muse-spark-1.1", "moonshotai/kimi-k2.6"}
    output = tmp_path / "report"
    render_report(runs, output)
    assert read_json(output / "audit.json")["state"] == "passed"
    comparison = read_json(output / "blog_comparison.json")["reproduction"]
    assert comparison["selected_iteration"] == 0
    assert comparison["initial_gap"] == 1 and comparison["selected_gap"] == 1
    assert comparison["reversed"] is False
    assert comparison["paired_improvement"]["interval"]["paper_clusters"] == 5
    assert (output / "gap_curves.svg").exists()
    grade_path = next((source / "scores/main/0/validation").glob("*/human.json"))
    grade = read_json(grade_path)
    grade["total"] = 10
    write_json(grade_path, grade)
    render_report(runs, output)
    audit = read_json(output / "audit.json")
    assert audit["state"] == "failed" and "arithmetic" in audit["error"]
    assert read_json(output / "blog_comparison.json")["reproduction"] is None
    assert not (output / "gap_curves.svg").exists()
