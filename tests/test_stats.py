"""Paired whole-paper bootstrap."""

from xar.stats import bootstrap


def test_bootstrap_resamples_paper_bundles():
    rows = [{"paper_id": "a", "gap": 1} for _ in range(4)] + [{"paper_id": "b", "gap": 3} for _ in range(4)]
    interval = bootstrap(rows)
    assert interval["paper_clusters"] == 2
    assert interval["low"] == 1 and interval["high"] == 3
    unequal = bootstrap([{"paper_id": "a", "gap": 0}] + [{"paper_id": "b", "gap": 10}] * 4)
    assert unequal["estimate"] == 8 and unequal["bootstrap_median"] == 8
