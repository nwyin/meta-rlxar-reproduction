"""Report rendering without a completed research run."""

from xar.report import render_report
from xar.util import read_json, write_json


def test_report_ignores_other_runs_until_research_starts(tmp_path):
    historical = tmp_path / "runs/pilot-schema-GG2"
    write_json(historical / "manifest.json", {"experiment": "xar", "arguments": {"split": "pilot"}})
    output = tmp_path / "report"
    render_report(tmp_path / "runs", output)
    audit = read_json(output / "audit.json")
    assert audit["state"] == "not_started" and audit["error"] is None
    assert read_json(output / "blog_comparison.json")["reproduction"] is None
    assert not (output / "gap_curves.png").exists()
