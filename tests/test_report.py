"""Report rendering without a completed research run."""

from xar.report import render_report
from xar.util import read_json, write_json


def test_blog_report_does_not_count_historical_runs(tmp_path):
    historical = tmp_path / "runs/pilot-schema-GG2"
    write_json(historical / "manifest.json", {"experiment": "xar", "arguments": {"split": "pilot"}})
    output = tmp_path / "report"
    render_report(tmp_path / "runs", output)
    audit = read_json(output / "audit.json")
    assert audit["expected_trajectories"] == 1
    assert audit["raw_verified_trajectories"] == 0 and audit["state"] == "not_started"
    assert read_json(output / "blog_comparison.json")["reproduction"] is None
    assert not (output / "gap_curves.png").exists()
