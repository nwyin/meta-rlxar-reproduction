"""Full pilot and research runs against a fake provider, including report generation."""

import re
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import call_kind, install_fake, write_dataset

import run
from xar import openrouter
from xar.data import SECTIONS
from xar.pipeline import run_xar
from xar.report import load_run, render_report
from xar.util import ROLES, RunError, read_json


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


def test_a_lost_response_is_sent_again_and_counted_as_unresolved(tmp_path, monkeypatch, design):
    monkeypatch.setattr(openrouter.time, "sleep", lambda seconds: None)
    fake = install_fake(monkeypatch)
    fake.drop_responses = 2
    settings = run_phase(tmp_path, design, "pilot", {"pilot": ["pilot0", "pilot1"]})
    out = Path(settings.output_dir)
    costs = read_json(out / "costs.json")
    assert costs["unresolved"] == 2 and costs["unresolved_upper_usd"] > 0
    assert costs["requests"] == len(fake.payloads) - 2
    assert read_json(out / "operational_summary.json")["unresolved_transport"] == 2


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


def test_report_from_saved_research_run(tmp_path, monkeypatch, design):
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
    assert not (output / "audit.json").exists()
    comparison = read_json(output / "blog_comparison.json")["reproduction"]
    assert comparison["selected_iteration"] == 0
    assert comparison["initial_gap"] == 1 and comparison["selected_gap"] == 1
    assert comparison["reversed"] is False
    assert comparison["paired_improvement"]["interval"]["paper_clusters"] == len(groups["validation"])
    assert (output / "gap_curves.svg").exists()
    assert "Cross-judge check" in (output / "results.md").read_text()
    # Reporting uses saved rows even when individual grade records are unavailable.
    grade_path = next((source / "scores/main/0/validation").glob("*/human.json"))
    grade_path.unlink()
    render_report(runs, output)
    assert read_json(output / "blog_comparison.json")["reproduction"] == comparison
    assert load_run(source)["cross"] == cross
