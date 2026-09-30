"""Checkpoint summaries and the paired whole-paper bootstrap."""

from xar.stats import bootstrap, paired_improvement, summarize


def test_bootstrap_resamples_whole_papers():
    rows = [{"paper_id": "a", "gap": 1} for _ in range(4)] + [{"paper_id": "b", "gap": 3} for _ in range(4)]
    interval = bootstrap(rows)
    assert interval["paper_clusters"] == 2
    assert interval["low"] == 1 and interval["high"] == 3
    unequal = bootstrap([{"paper_id": "a", "gap": 0}] + [{"paper_id": "b", "gap": 10}] * 4)
    assert unequal["estimate"] == 8 and unequal["bootstrap_median"] == 8


def row(example, paper, section, human, model):
    gap = None if human is None else human - model
    return {
        "example_id": example,
        "paper_id": paper,
        "section_type": section,
        "human": human,
        "model": model,
        "gap": gap,
        "length_compliant": True,
        "contamination_flagged": False,
    }


def test_summarize_averages_only_sections_with_both_grades():
    rows = [
        row("a1", "a", "abstract", 8, 6),
        row("a2", "a", "conclusion", 5, 6),
        row("b1", "b", "abstract", None, 7),
    ]
    summary = summarize(rows)
    assert summary["examples"] == 3 and summary["paired_coverage"] == 2 and not summary["complete"]
    assert summary["gap"] == 0.5 and summary["human"] == 6.5
    assert summary["sections"]["abstract"] == {"coverage": 1, "gap": 2}
    assert summary["sections"]["introduction"] == {"coverage": 0, "gap": None}
    assert summary["paper_interval"]["paper_clusters"] == 1


def test_paired_improvement_skips_sections_missing_a_gap():
    initial = [row("a1", "a", "abstract", 5, 7), row("a2", "a", "conclusion", None, 7)]
    selected = [row("a1", "a", "abstract", 6, 6), row("a2", "a", "conclusion", 6, 6)]
    improvement = paired_improvement(initial, selected)
    assert improvement["paired_coverage"] == 1 and improvement["mean"] == 2
