"""Candidate papers for the corpus: a seeded OpenAlex sample, checked against arXiv, saved as a shortlist."""

from __future__ import annotations

import random
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter

import httpx

from xar.util import ROOT, RunError, normalize, now, read_json, write_json

OPENALEX_WORKS = "https://api.openalex.org/works"
ARXIV_QUERY = "https://export.arxiv.org/api/query"
ARXIV_SOURCE_ID = "S4306400194"  # OpenAlex's ID for arXiv itself
ATOM = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}

# The filters are in the policy file's text too. 100 to 1,500 citations keeps papers with outside
# evidence of quality and leaves out the most memorized ones.
MIN_CITATIONS, MAX_CITATIONS = 100, 1500
FIRST_POSTED, LAST_POSTED = "2016-01-01", "2021-12-31"
# ChatGPT launched on 2022-11-30. A paper whose last arXiv revision and published version both come
# earlier cannot contain text from a chat model.
LAST_REVISION, LAST_PUBLISHED = "2022-11-29", "2022-11-29"
SAMPLE_SIZE = 10_000
SAMPLE_SEED = 20261001
SHORTLIST_PER_STRATUM = 60
MAX_AUTHORS = 20  # larger collaborations write by committee
REQUEST_PAUSE_SECONDS = 5  # arXiv's API asks for 3 s between calls and rate-limits close to that

# Papers are grouped by arXiv's primary category, so each field is represented in the corpus.
# Each value is the number of papers wanted from that group.
STRATA = {
    "cs_eess": 9,
    "math": 8,
    "stat_econ_fin": 3,
    "q_bio": 5,
    "astro_ph": 6,
    "cond_mat": 7,
    "hep_gr_nucl": 7,
    "quant_ph": 5,
    "physics_other": 6,
}
STRATUM_OF_ARCHIVE = {
    "cs": "cs_eess",
    "eess": "cs_eess",
    "math": "math",
    "math-ph": "math",
    "stat": "stat_econ_fin",
    "econ": "stat_econ_fin",
    "q-fin": "stat_econ_fin",
    "q-bio": "q_bio",
    "astro-ph": "astro_ph",
    "cond-mat": "cond_mat",
    "hep-th": "hep_gr_nucl",
    "hep-ph": "hep_gr_nucl",
    "hep-ex": "hep_gr_nucl",
    "hep-lat": "hep_gr_nucl",
    "gr-qc": "hep_gr_nucl",
    "nucl-th": "hep_gr_nucl",
    "nucl-ex": "hep_gr_nucl",
    "quant-ph": "quant_ph",
    "physics": "physics_other",
    "nlin": "physics_other",
}


def arxiv_id(work):
    """The arXiv ID of an OpenAlex work, or None. Only post-2007 IDs (YYMM.NNNNN) are recognized."""
    for location in work["locations"]:
        url = (location.get("landing_page_url") or "").lower()
        match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})|arxiv\.(\d{4}\.\d{4,5})", url)
        if match:
            return match.group(1) or match.group(2)
    return None


def published_version(work):
    """The first location that is the published journal or conference version, or None."""
    for location in work["locations"]:
        source = location.get("source") or {}
        if source.get("type") in ("journal", "conference") and location.get("version") == "publishedVersion":
            return {"venue": source["display_name"], "url": location.get("landing_page_url")}
    return None


def sample_works(client):
    """A seeded random sample of OpenAlex works that have an arXiv copy, in 200-work pages."""
    filters = ",".join(
        (
            f"locations.source.id:{ARXIV_SOURCE_ID}",
            f"cited_by_count:>{MIN_CITATIONS - 1}",
            f"cited_by_count:<{MAX_CITATIONS + 1}",
            f"from_publication_date:{FIRST_POSTED}",
            f"to_publication_date:{LAST_PUBLISHED}",
            "type:article",
            "is_retracted:false",
        )
    )
    fields = "id,doi,title,publication_date,cited_by_count,locations,authorships"
    works = []
    for page in range(1, SAMPLE_SIZE // 200 + 1):
        response = get_with_retries(
            client,
            OPENALEX_WORKS,
            {
                "filter": filters,
                "sample": SAMPLE_SIZE,
                "seed": SAMPLE_SEED,
                "per-page": 200,
                "page": page,
                "select": fields,
            },
        )
        works += response.json()["results"]
        print(f"OpenAlex sample: {len(works)} works", flush=True)
    return works


def get_with_retries(client, url, params):
    """GET that waits and retries on rate limits (429) and server errors, up to 6 attempts."""
    for attempt in range(6):
        response = client.get(url, params=params)
        if response.status_code != 429 and response.status_code < 500:
            response.raise_for_status()
            return response
        wait = 30 * 2**attempt
        print(f"HTTP {response.status_code} from {url}; waiting {wait} s", flush=True)
        time.sleep(wait)
    response.raise_for_status()


def arxiv_records(client, ids):
    """Category, dates, authors and journal reference from the arXiv API, in batches of 100.

    Fetched batches are cached in data/raw/arxiv_metadata.json, so an interrupted run resumes.
    """
    cache_path = ROOT / "data/raw/arxiv_metadata.json"
    records = read_json(cache_path) if cache_path.exists() else {}
    ids = [paper_id for paper_id in ids if paper_id not in records]
    for start in range(0, len(ids), 100):
        batch = ids[start : start + 100]
        response = get_with_retries(client, ARXIV_QUERY, {"id_list": ",".join(batch), "max_results": 100})
        for entry in ET.fromstring(response.text).findall("a:entry", ATOM):
            paper_id = re.sub(r"v\d+$", "", entry.find("a:id", ATOM).text.split("/abs/")[-1])
            journal_ref = entry.find("x:journal_ref", ATOM)
            records[paper_id] = {
                "primary_category": entry.find("x:primary_category", ATOM).get("term"),
                "first_posted": entry.find("a:published", ATOM).text[:10],
                "last_revised": entry.find("a:updated", ATOM).text[:10],
                "authors": [
                    normalize(author.find("a:name", ATOM).text) for author in entry.findall("a:author", ATOM)
                ],
                "journal_ref": normalize(journal_ref.text) if journal_ref is not None else None,
            }
        write_json(cache_path, records)
        print(f"arXiv metadata: {min(start + 100, len(ids))}/{len(ids)} new", flush=True)
        time.sleep(REQUEST_PAUSE_SECONDS)
    return records


def build_shortlists(works, records):
    """Filter the sample, group it by field and shuffle each group with SAMPLE_SEED.

    Returns the groups' candidates (at most SHORTLIST_PER_STRATUM each) and a count of how many
    works each filter removed, so the pool sizes are on record.
    """
    removed, pools = Counter(), {stratum: [] for stratum in STRATA}
    seen = set()
    for work in works:
        paper_id = arxiv_id(work)
        published = published_version(work)
        record = records.get(paper_id)
        if paper_id is None or paper_id in seen:
            removed["no arXiv ID, or a repeat"] += 1
        elif published is None:
            removed["no published journal or conference version"] += 1
        elif record is None:
            removed["not returned by the arXiv API"] += 1
        elif not FIRST_POSTED <= record["first_posted"] <= LAST_POSTED:
            removed["first posted outside 2016 to 2021"] += 1
        elif record["last_revised"] > LAST_REVISION:
            removed["revised on arXiv after 2022-11-29"] += 1
        elif len(record["authors"]) > MAX_AUTHORS:
            removed[f"more than {MAX_AUTHORS} authors"] += 1
        elif record["primary_category"].split(".")[0] not in STRATUM_OF_ARCHIVE:
            removed["arXiv archive not in a stratum"] += 1
        else:
            seen.add(paper_id)
            stratum = STRATUM_OF_ARCHIVE[record["primary_category"].split(".")[0]]
            pools[stratum].append(
                {
                    "paper_id": paper_id,
                    "title": normalize(work["title"]),
                    "stratum": stratum,
                    "openalex_id": work["id"],
                    "doi": work["doi"],
                    "citations": work["cited_by_count"],
                    "published_venue": published["venue"],
                    "published_url": published["url"],
                    "published_date": work["publication_date"],
                    **record,
                }
            )
    shortlists = {}
    for stratum, pool in pools.items():
        random.Random(f"{SAMPLE_SEED}-{stratum}").shuffle(pool)
        shortlists[stratum] = {"eligible_in_sample": len(pool), "candidates": pool[:SHORTLIST_PER_STRATUM]}
    return shortlists, dict(removed)


def discover_candidates():
    """Write data/discovery.json from OpenAlex and arXiv. Refuses to replace a different file."""
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        works = sample_works(client)
        ids = sorted({paper_id for work in works if (paper_id := arxiv_id(work))})
        records = arxiv_records(client, ids)
    shortlists, removed = build_shortlists(works, records)
    short = {s: len(v["candidates"]) for s, v in shortlists.items() if len(v["candidates"]) < STRATA[s] * 3}
    if short:
        raise RunError(f"Too few candidates (want 3 times the quota) in strata: {short}")
    write_json(
        ROOT / "data/discovery.json",
        {
            "retrieved_at": now(),
            "sample_seed": SAMPLE_SEED,
            "sample_size": SAMPLE_SIZE,
            "sampled_works": len(works),
            "removed_by_filter": removed,
            "strata": shortlists,
        },
        write_once=True,
    )
    for stratum, value in shortlists.items():
        print(f"{stratum}: {value['eligible_in_sample']} eligible, {len(value['candidates'])} shortlisted")


def load_discovery():
    """The saved shortlists: a dict of stratum to a list of candidates in frozen order."""
    path = ROOT / "data/discovery.json"
    if not path.exists():
        raise RunError(f"Missing {path}; run `uv run python run.py discover-data` first")
    return {stratum: value["candidates"] for stratum, value in read_json(path)["strata"].items()}
