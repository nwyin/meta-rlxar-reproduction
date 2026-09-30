"""Checkpoint summaries and paired whole-paper bootstrap intervals."""

from __future__ import annotations

import random
import statistics

from xar.data import SECTIONS

BOOTSTRAP_REPLICATES = 2000
INTERVAL = (0.025, 0.975)  # percentiles of the 95% interval


def bootstrap(rows, seed=0):
    """Percentile bootstrap of the mean gap that resamples whole papers.

    Sections from one paper are not independent, so each replicate draws papers with
    replacement and takes all of a drawn paper's gaps. Rows without a gap are ignored;
    returns None if no row has one."""
    gaps_by_paper = {}
    for row in rows:
        if row.get("gap") is not None:
            gaps_by_paper.setdefault(row["paper_id"], []).append(row["gap"])
    if not gaps_by_paper:
        return None
    rng = random.Random(seed)
    papers = sorted(gaps_by_paper)
    means = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = rng.choices(papers, k=len(papers))
        means.append(statistics.mean(gap for paper in sample for gap in gaps_by_paper[paper]))
    means.sort()
    low, high = INTERVAL
    return {
        "method": "paired_whole_paper_percentile_bootstrap",
        "paper_clusters": len(papers),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "estimate": statistics.mean(gap for gaps in gaps_by_paper.values() for gap in gaps),
        "bootstrap_median": means[BOOTSTRAP_REPLICATES // 2],
        "low": means[int(low * BOOTSTRAP_REPLICATES)],
        "high": means[int(high * BOOTSTRAP_REPLICATES)],
    }


def section_summary(graded, section):
    gaps = [row["gap"] for row in graded if row["section_type"] == section]
    return {"coverage": len(gaps), "gap": statistics.mean(gaps) if gaps else None}


def summarize(rows, seed=0):
    """Summarize one checkpoint's rows: mean human, model and gap scores over the sections with
    both grades, per-section gaps, and paper-bootstrap intervals for all sections, the
    length-compliant ones and the ones not flagged for copying the reference."""
    graded = [row for row in rows if row["gap"] is not None]

    def mean(key):
        return statistics.mean(row[key] for row in graded) if graded else None

    return {
        "examples": len(rows),
        "paired_coverage": len(graded),
        "complete": len(graded) == len(rows),
        "human": mean("human"),
        "model": mean("model"),
        "gap": mean("gap"),
        "positive_gap_fraction": sum(row["gap"] > 0 for row in graded) / len(graded) if graded else None,
        "ties": sum(row["gap"] == 0 for row in graded),
        "paper_interval": bootstrap(graded, seed),
        "length_compliant": sum(row["length_compliant"] for row in rows),
        "contamination_flagged": sum(row["contamination_flagged"] for row in rows),
        "sections": {section: section_summary(graded, section) for section in SECTIONS},
        "compliant_sensitivity": bootstrap([row for row in graded if row["length_compliant"]], seed),
        "unflagged_sensitivity": bootstrap([row for row in graded if not row["contamination_flagged"]], seed),
    }


def paired_improvement(initial, selected, seed=0):
    """Per-section change in gap from the initial to the selected checkpoint, with a
    paper-bootstrap interval. Sections missing a gap at either checkpoint are left out."""
    initial_by_id = {row["example_id"]: row for row in initial}
    changes = []
    for row in selected:
        if row["gap"] is None:
            continue
        before = initial_by_id[row["example_id"]]["gap"]
        if before is not None:
            changes.append({"paper_id": row["paper_id"], "gap": row["gap"] - before})
    return {
        "paired_coverage": len(changes),
        "mean": statistics.mean(change["gap"] for change in changes) if changes else None,
        "interval": bootstrap(changes, seed),
    }
