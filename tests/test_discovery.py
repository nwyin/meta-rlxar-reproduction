"""Candidate filtering: which sampled works reach a field's shortlist."""

import httpx
import pytest

from xar import discovery
from xar.discovery import (
    LAST_REVISION,
    MAX_AUTHORS,
    SHORTLIST_PER_STRATUM,
    arxiv_id,
    build_shortlists,
    get_with_retries,
    published_version,
)


def work(paper_id="1901.00001", published=True, title="A result"):
    locations = [{"landing_page_url": f"http://arxiv.org/abs/{paper_id}", "source": {"type": "repository"}}]
    if published:
        locations.append(
            {
                "landing_page_url": "https://doi.org/10.1/x",
                "version": "publishedVersion",
                "source": {"type": "journal", "display_name": "Physical Review X"},
            }
        )
    return {
        "id": "https://openalex.org/W" + paper_id,
        "doi": "https://doi.org/10.1/x",
        "title": title,
        "publication_date": "2020-01-01",
        "cited_by_count": 200,
        "locations": locations,
    }


def record(category="cs.CL", first="2019-01-02", revised="2019-06-01", authors=("A", "B")):
    return {
        "primary_category": category,
        "first_posted": first,
        "last_revised": revised,
        "authors": list(authors),
        "journal_ref": None,
    }


def test_arxiv_id_reads_abs_and_doi_links():
    assert arxiv_id(work("1901.00001")) == "1901.00001"
    doi_only = {"locations": [{"landing_page_url": "https://doi.org/10.48550/arxiv.2006.10256"}]}
    assert arxiv_id(doi_only) == "2006.10256"
    assert arxiv_id({"locations": [{"landing_page_url": "https://example.org"}]}) is None


def test_published_version_needs_a_journal_or_conference_location():
    assert published_version(work())["venue"] == "Physical Review X"
    assert published_version(work(published=False)) is None


def test_build_shortlists_applies_each_filter_and_counts_removals():
    works = [
        work("1901.00001"),
        work("1901.00002", published=False),
        work("1901.00003"),
        work("1901.00004"),
        work("1901.00005"),
        work("1901.00006"),
        work("1901.00007"),
        work("1901.00008"),
    ]
    records = {
        "1901.00001": record("cs.CL"),
        "1901.00002": record("cs.CL"),
        "1901.00003": record("cs.CL", first="2015-12-31"),
        "1901.00004": record("cs.CL", revised="2023-03-01"),
        "1901.00005": record("cs.CL", authors=[str(i) for i in range(MAX_AUTHORS + 1)]),
        "1901.00006": record("q-alg.XX"),
        "1901.00007": record("math.NT"),
    }
    shortlists, removed = build_shortlists(works, records)
    kept = {s: [c["paper_id"] for c in v["candidates"]] for s, v in shortlists.items() if v["candidates"]}
    assert kept == {"cs_eess": ["1901.00001"], "math": ["1901.00007"]}
    assert sum(removed.values()) == 6
    assert removed["no published journal or conference version"] == 1
    assert removed["not returned by the arXiv API"] == 1
    assert removed["revised on arXiv after 2022-11-29"] == 1
    assert LAST_REVISION == "2022-11-29"


def test_build_shortlists_caps_and_shuffles_each_field():
    works = [work(f"1901.{i:05d}") for i in range(SHORTLIST_PER_STRATUM + 20)]
    records = {f"1901.{i:05d}": record("hep-th") for i in range(SHORTLIST_PER_STRATUM + 20)}
    first, _ = build_shortlists(works, records)
    second, _ = build_shortlists(list(reversed(works)), records)
    ids = [c["paper_id"] for c in first["hep_gr_nucl"]["candidates"]]
    assert len(ids) == SHORTLIST_PER_STRATUM
    assert first["hep_gr_nucl"]["eligible_in_sample"] == SHORTLIST_PER_STRATUM + 20
    assert sorted(ids) != ids  # shuffled, not in ID order
    assert len(second["hep_gr_nucl"]["candidates"]) == SHORTLIST_PER_STRATUM
    assert discovery.STRATA["hep_gr_nucl"] <= SHORTLIST_PER_STRATUM


class FakeClient:
    def __init__(self, statuses):
        self.statuses = list(statuses)

    def get(self, url, params):
        return httpx.Response(self.statuses.pop(0), request=httpx.Request("GET", url))


def test_get_with_retries_waits_out_rate_limits(monkeypatch):
    waits = []
    monkeypatch.setattr(discovery.time, "sleep", waits.append)
    client = FakeClient([429, 503, 200])
    assert get_with_retries(client, "https://example.org", {}).status_code == 200
    assert waits == [30, 60]


def test_get_with_retries_stops_on_a_client_error_and_after_six_tries(monkeypatch):
    monkeypatch.setattr(discovery.time, "sleep", lambda seconds: None)
    with pytest.raises(httpx.HTTPStatusError):
        get_with_retries(FakeClient([404]), "https://example.org", {})
    with pytest.raises(httpx.HTTPStatusError):
        get_with_retries(FakeClient([429] * 6), "https://example.org", {})
