"""Checkpoint summaries and paired whole-paper bootstrap intervals."""

from __future__ import annotations

import random
import statistics

from xar.data import SECTIONS

BOOTSTRAP_REPLICATES = 2000


def bootstrap(rows, seed=0):
    bundles = {}
    for r in rows:
        if r.get("gap") is not None:
            bundles.setdefault(r["paper_id"], []).append(r["gap"])
    if not bundles:
        return None
    rng = random.Random(seed)
    papers = sorted(bundles)
    samples = sorted(
        statistics.mean(value for p in rng.choices(papers, k=len(papers)) for value in bundles[p])
        for _ in range(BOOTSTRAP_REPLICATES)
    )
    return {
        "method": "paired_whole_paper_percentile_bootstrap",
        "paper_clusters": len(papers),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "estimate": statistics.mean(value for values in bundles.values() for value in values),
        "bootstrap_median": samples[BOOTSTRAP_REPLICATES // 2],
        "low": samples[int(0.025 * BOOTSTRAP_REPLICATES)],
        "high": samples[int(0.975 * BOOTSTRAP_REPLICATES)],
    }


def summarize(rows, seed=0):
    valid = [r for r in rows if r["gap"] is not None]

    def mean(key):
        return statistics.mean(r[key] for r in valid) if valid else None

    return {
        "examples": len(rows),
        "paired_coverage": len(valid),
        "complete": len(valid) == len(rows),
        "human": mean("human"),
        "model": mean("model"),
        "gap": mean("gap"),
        "positive_gap_fraction": sum(r["gap"] > 0 for r in valid) / len(valid) if valid else None,
        "ties": sum(r["gap"] == 0 for r in valid),
        "paper_interval": bootstrap(valid, seed),
        "length_compliant": sum(r["length_compliant"] for r in rows),
        "contamination_flagged": sum(r["contamination_flagged"] for r in rows),
        "sections": {
            s: {
                "coverage": sum(r["section_type"] == s for r in valid),
                "gap": statistics.mean(r["gap"] for r in valid if r["section_type"] == s)
                if any(r["section_type"] == s for r in valid)
                else None,
            }
            for s in SECTIONS
        },
        "compliant_sensitivity": bootstrap([r for r in valid if r["length_compliant"]], seed),
        "unflagged_sensitivity": bootstrap([r for r in valid if not r["contamination_flagged"]], seed),
    }


def paired_improvement(initial, selected, seed=0):
    base = {r["example_id"]: r for r in initial}
    rows = [
        {"paper_id": r["paper_id"], "gap": r["gap"] - base[r["example_id"]]["gap"]}
        for r in selected
        if r["gap"] is not None and base[r["example_id"]]["gap"] is not None
    ]
    return {
        "paired_coverage": len(rows),
        "mean": statistics.mean(r["gap"] for r in rows) if rows else None,
        "interval": bootstrap(rows, seed),
    }
