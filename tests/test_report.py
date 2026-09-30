"""Report rendering without a completed research run, and the generated result sentences."""

from xar.report import checkpoint_table, render_report, sign_sentence, training_trend
from xar.util import read_json, write_json


def test_report_ignores_other_runs_until_research_starts(tmp_path):
    historical = tmp_path / "runs/pilot-schema-GG2"
    write_json(historical / "manifest.json", {"experiment": "xar", "arguments": {"split": "pilot"}})
    output = tmp_path / "report"
    render_report(tmp_path / "runs", output)
    audit = read_json(output / "audit.json")
    assert audit["state"] == "not_started" and audit["error"] is None
    assert read_json(output / "blog_comparison.json")["reproduction"] is None
    assert "has not started" in (output / "results.md").read_text()
    assert not (output / "gap_curves.png").exists()


def checkpoints(train_gaps, val_gaps):
    return [
        {"iteration": i, "train_gap": t, "val_gap": v, "val_human": 8.0, "val_model": 8.0 - v}
        for i, (t, v) in enumerate(zip(train_gaps, val_gaps))
    ]


def test_result_sentences_state_sign_and_training_direction():
    table = checkpoints([-0.8, -0.25, -0.4, -0.65], [-1.04, -0.16, -0.4, -0.37])
    summary = {"initial_gap": -1.04, "selected_gap": -0.16, "selected_iteration": 1, "reversed": False}
    assert sign_sentence(table, summary).startswith("The sign did not flip.")
    assert "negative at every checkpoint" in sign_sentence(table, summary)
    trend = training_trend(table, 1)
    assert "from -0.25 at P1 to -0.65 at P3." in trend and "moved away from zero" in trend
    rows = checkpoint_table(table, 1).splitlines()
    assert len(rows) == 2 + len(table)
    assert rows[3] == "| P1 (selected) | -0.25 | -0.16 | 8.00 | 8.16 |"
    flipped = {**summary, "selected_gap": 0.5, "reversed": True}
    assert sign_sentence(table, flipped).startswith("The sign flipped.")
