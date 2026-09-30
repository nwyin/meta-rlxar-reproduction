"""Full pilot and research runs against a fake provider, including tamper detection."""

import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import call_kind, install_fake, write_dataset

import run
from xar.audit import RESEARCH_SPLIT_SIZES, audit_xar_run
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
    """A completed pilot run with its settings and output directory, and the fake that served it."""
    fake = install_fake(monkeypatch)
    settings = run_phase(tmp_path, design, "pilot", {"pilot": ["pilot0", "pilot1"]})
    return SimpleNamespace(fake=fake, settings=settings, out=Path(settings.output_dir))


def test_pilot_run_completes_and_resume_makes_no_calls(pilot, design):
    fake, settings, out = pilot.fake, pilot.settings, pilot.out
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


def test_resume_allows_new_concurrency_and_logs_every_budget_change(pilot):
    settings, out = pilot.settings, pilot.out
    for budget in (20, settings.budget_usd, settings.budget_usd):
        run_xar(replace(settings, resume=True, concurrency=settings.concurrency + 1, budget_usd=budget))
    history = read_json(out / "budget_continuations.json")
    assert [entry["budget_usd"] for entry in history] == [20, settings.budget_usd]


def test_resume_rejects_changed_iterations_and_names_the_values(pilot):
    settings = pilot.settings
    changed = replace(settings, resume=True, iterations=settings.iterations + 1)
    expected = f"arguments.iterations (saved {settings.iterations}, now {settings.iterations + 1})"
    with pytest.raises(RunError, match=re.escape(expected)):
        run_xar(changed)


def test_audit_rejects_changed_request_temperature(pilot):
    out = pilot.out
    grade = read_json(next((out / "scores/main/0/train").glob("*/human.json")))
    request_path = Path(grade["attempts"][-1]["response"]["raw_response"]).parent / "request.json"
    request = read_json(request_path)
    request["payload"]["temperature"] = 0.9
    write_json(request_path, request)
    with pytest.raises(RunError, match="temperature is 0.9"):
        audit_xar_run(out)


def test_audit_rejects_wrong_grade_total(pilot):
    out = pilot.out
    grade_path = out / "scores/main/0/train/pilot0_abstract/human.json"
    grade = read_json(grade_path)
    grade["total"] = 10
    write_json(grade_path, grade)
    with pytest.raises(RunError, match="arithmetic"):
        audit_xar_run(out)


def test_check_pilot_records_the_audited_pilot(pilot, design):
    runs_root = pilot.out.parent
    run.check_pilot(runs_root, design)
    manifest = read_json(pilot.out / "manifest.json")
    gate = read_json(runs_root / "meta_blog_pilot_gate.json")
    assert gate == {"pilot_substantive_hash": manifest["substantive_hash"]}


def test_check_pilot_rejects_a_pilot_that_differs_from_the_design(pilot, design):
    out = pilot.out
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
    default = run.settings_for(phase, design, run.parse_args([phase, "--dry-run"]))
    assert default.concurrency == design["concurrency"]


@pytest.mark.parametrize("phase", ["pilot", "reproduce", "all"])
def test_paid_runs_require_both_budgets(phase, capsys):
    with pytest.raises(SystemExit):
        run.parse_args([phase, "--budget-usd", "10"])
    assert "--total-budget-usd are required unless --dry-run" in capsys.readouterr().err


def test_report_from_research_run_and_after_tampering(tmp_path, monkeypatch, design):
    fake = install_fake(monkeypatch)
    groups = {
        split: [f"{split}{i}" for i in range(count // len(SECTIONS))]
        for split, count in RESEARCH_SPLIT_SIZES.items()
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
    assert comparison["paired_improvement"]["interval"]["paper_clusters"] == len(groups["validation"])
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
