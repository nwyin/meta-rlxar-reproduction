"""Full pilot and research runs against a fake provider, including tamper detection."""

import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
    options = run.parse_args([phase, "--runs-root", str(tmp_path / "runs"), *flags])
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


def test_resume_allows_new_concurrency(pilot):
    settings = pilot.settings
    run_xar(replace(settings, resume=True, concurrency=settings.concurrency + 1))
    assert read_json(pilot.out / "manifest.json")["arguments"]["concurrency"] == settings.concurrency


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


def test_audit_rejects_a_writer_text_that_differs_from_its_response(pilot):
    out = pilot.out
    candidates = read_json(out / "generations/candidates.json")["candidates"]
    raw_path = next(iter(candidates.values()))["attempts"][0]["response"]["raw_response"]
    sent = read_json(raw_path)
    sent["response"]["choices"][0]["message"]["content"] += " An added sentence."
    write_json(raw_path, sent)
    with pytest.raises(RunError, match="saved text differs from the response"):
        audit_xar_run(out)


def test_audit_rejects_a_cost_file_that_differs_from_the_responses(pilot):
    out = pilot.out
    costs = read_json(out / "costs.json")
    costs["actual_complete_usd"] += 1
    write_json(out / "costs.json", costs)
    with pytest.raises(RunError, match="costs.json total differs"):
        audit_xar_run(out)


def test_audit_rejects_optimizer_feedback_that_does_not_follow_from_the_scores(pilot):
    out = pilot.out
    path = out / "feedback/iter_01/training.json"
    feedback = read_json(path)
    feedback["unexpected_field"] = "added after the fact"
    write_json(path, feedback)
    with pytest.raises(RunError, match="iteration 1: feedback differs"):
        audit_xar_run(out)


def test_check_pilot_accepts_an_audited_pilot_that_matches_the_design(pilot, design):
    run.check_pilot(pilot.out.parent, design)


def test_check_pilot_rejects_a_pilot_that_differs_from_the_design(pilot, design):
    out = pilot.out
    manifest = read_json(out / "manifest.json")
    manifest["arguments"]["failure_examples"] += 1
    write_json(out / "manifest.json", manifest)
    with pytest.raises(RunError, match="failure_examples"):
        run.check_pilot(out.parent, design)


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


def test_report_from_research_run_and_after_tampering(tmp_path, monkeypatch, design):
    fake = install_fake(monkeypatch)
    groups = {
        split: [f"{split}{i}" for i in range(count)] for split, count in {"train": 8, "validation": 5}.items()
    }
    settings = run_phase(tmp_path, design, "reproduce", groups, "--concurrency", "4")
    source, runs = Path(settings.output_dir), tmp_path / "runs"
    sections = len(SECTIONS) * sum(len(papers) for papers in groups.values())
    validation_sections = len(SECTIONS) * len(groups["validation"])
    checkpoints = design["iterations"] + 1
    # Every checkpoint ties, so P0 is selected and the cross judge grades one checkpoint.
    assert Counter(call_kind(p) for p in fake.payloads) == {
        "writer": sections,
        "rubric": sections * checkpoints,
        "grade": 2 * sections * checkpoints + 2 * validation_sections,
        "optimizer": design["iterations"],
    }
    assert {p["model"] for p in fake.payloads} == {
        design[role] for role in ("writer", "optimizer", "judge", "cross_judge")
    }
    cross = read_json(source / "cross_check.json")["checkpoints"]
    assert [entry["iteration"] for entry in cross] == [0]
    assert cross[0]["judge"]["gap"] == cross[0]["cross_judge"]["gap"] == 1
    output = tmp_path / "report"
    render_report(runs, output)
    assert read_json(output / "audit.json")["state"] == "passed"
    comparison = read_json(output / "blog_comparison.json")["reproduction"]
    assert comparison["selected_iteration"] == 0
    assert comparison["initial_gap"] == 1 and comparison["selected_gap"] == 1
    assert comparison["reversed"] is False
    assert comparison["paired_improvement"]["interval"]["paper_clusters"] == len(groups["validation"])
    assert (output / "gap_curves.svg").exists()
    assert "Cross-judge check" in (output / "results.md").read_text()
    cross_grade_path = next((source / "scores/cross/0/validation").glob("*/human.json"))
    cross_grade = read_json(cross_grade_path)
    cross_grade["total"] = 10
    write_json(cross_grade_path, cross_grade)
    with pytest.raises(RunError, match="scores/cross/0/validation.*arithmetic"):
        audit_xar_run(source)
    cross_grade["total"] = 7
    write_json(cross_grade_path, cross_grade)
    audit_xar_run(source)
    grade_path = next((source / "scores/main/0/validation").glob("*/human.json"))
    grade = read_json(grade_path)
    grade["total"] = 10
    write_json(grade_path, grade)
    render_report(runs, output)
    audit = read_json(output / "audit.json")
    assert audit["state"] == "failed" and "arithmetic" in audit["error"]
    assert read_json(output / "blog_comparison.json")["reproduction"] is None
    assert not (output / "gap_curves.svg").exists()
