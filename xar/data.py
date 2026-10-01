"""The paper dataset: building it from arXiv HTML, loading and checking it, and what models see of it."""

from __future__ import annotations

import json
import math
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

from xar.discovery import SAMPLE_SEED, STRATA, load_discovery
from xar.openrouter import token_count
from xar.util import (
    ROOT,
    RunError,
    canonical,
    digest,
    file_hash,
    normalize,
    read_json,
    words,
    write_json,
)

SECTIONS = ("abstract", "introduction", "related_work", "conclusion")
# Many papers, mathematics and physics ones especially, have no related-work section or no
# conclusion. Every paper gives an abstract and an introduction, and each other section it has.
REQUIRED_SECTIONS = ("abstract", "introduction")

# contamination() measures overlap in 8-word n-grams and flags a candidate that copies a run of
# 30 or more consecutive words from the author's section.
NGRAM = 8
VERBATIM_FLAG_WORDS = 30


def load_examples(dataset, splits):
    """Read the examples from `dataset` (JSON lines) and check them against `splits`.

    Checks that no paper is in two splits, each example's split and content hashes match, the
    withheld section is absent from the visible paper, and every paper has each of its sections
    once (by example ID), and always the abstract and the introduction.
    """
    papers_by_split = read_json(splits)["papers"]
    split_counts = Counter(paper for papers in papers_by_split.values() for paper in papers)
    repeated = sorted(paper for paper, count in split_counts.items() if count > 1)
    if repeated:
        raise RunError(f"Paper splits overlap in {splits}: {', '.join(repeated)} listed more than once")
    examples = [json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()]
    id_counts = Counter(e["example_id"] for e in examples)
    duplicates = sorted(eid for eid, count in id_counts.items() if count > 1)
    if duplicates:
        raise RunError(f"{dataset} repeats example IDs: {', '.join(duplicates)}")
    sections_by_paper = {}
    for e in examples:
        eid = e["example_id"]
        listed_in = [split for split, papers in papers_by_split.items() if e["paper_id"] in papers]
        if listed_in != [e["split"]]:
            raise RunError(
                f"{eid} is labelled {e['split']}, but {splits} lists paper {e['paper_id']} "
                f"under {', '.join(listed_in) or 'no split'}"
            )
        for field in ("context", "reference"):
            if e[field + "_hash"] != digest(e[field]):
                raise RunError(f"{eid}: {field} text does not match its stored {field}_hash")
        if e["target_words"] != words(e["reference"]):
            raise RunError(
                f"{eid}: target_words is {e['target_words']}, but the reference has "
                f"{words(e['reference'])} words"
            )
        if not e["target_words"]:
            raise RunError(f"{eid}: the reference is empty")
        if normalize(e["reference"]) in normalize(e["context"]):
            raise RunError(f"{eid}: the withheld reference still appears in the visible paper")
        sections_by_paper.setdefault(e["paper_id"], []).append(e["section_type"])
    for paper, sections in sections_by_paper.items():
        if not set(REQUIRED_SECTIONS) <= set(sections):
            raise RunError(
                f"Paper {paper} has sections {sorted(sections)}; expected {list(REQUIRED_SECTIONS)}"
            )
    return examples


def task_data(example):
    """The only example fields that go into writer, rubric and judge prompts.

    Everything else (the author's section, split labels, IDs, earlier feedback) stays out.
    """
    return {
        "visible_paper": example["context"],
        "section_type": example["section_type"],
        "target_words": example["target_words"],
    }


def contamination(candidate, reference):
    """Measure how much of the author's withheld section a candidate reproduces word for word.

    Reports the longest run of consecutive shared words and the fraction of the candidate's
    8-grams that also occur in the reference, and flags runs of VERBATIM_FLAG_WORDS or more.
    """
    candidate_words = normalize(candidate).split()
    reference_words = normalize(reference).split()
    reference_positions = {}
    for position, word in enumerate(reference_words):
        reference_positions.setdefault(word, []).append(position)
    # Longest common run by dynamic programming: run_ending_at[j] is the length of the shared run
    # that ends at the current candidate word and at reference word j.
    longest, run_ending_at = 0, {}
    for word in candidate_words:
        run_ending_at = {j: run_ending_at.get(j - 1, 0) + 1 for j in reference_positions.get(word, [])}
        longest = max(longest, max(run_ending_at.values(), default=0))
    reference_ngrams = {
        tuple(reference_words[i : i + NGRAM]) for i in range(len(reference_words) - NGRAM + 1)
    }
    candidate_ngrams = [
        tuple(candidate_words[i : i + NGRAM]) for i in range(len(candidate_words) - NGRAM + 1)
    ]
    shared = sum(ngram in reference_ngrams for ngram in candidate_ngrams)
    return {
        "longest_verbatim_run_words": longest,
        "eightgram_overlap_fraction": shared / max(1, len(candidate_ngrams)),
        "flagged": longest >= VERBATIM_FLAG_WORDS,
        # Always False: flagged candidates are reported, not dropped. The key stays because the
        # completed-run audit compares this whole dict with the saved one.
        "exclusion": False,
    }


# The pilot is a smoke test on the first training papers, not a split of its own.
PILOT_PAPERS = 2


def run_examples(examples, split):
    """The examples a run uses, labelled train or validation.

    A research run uses the train and validation papers as saved. A pilot run uses the papers
    labelled pilot in an older dataset, or else the first PILOT_PAPERS training papers; the first
    (sorted by ID) becomes training and the rest become validation.
    """
    if split not in ("pilot", "research"):
        raise RunError(f"Unknown split {split!r}; expected 'pilot' or 'research'")
    if split == "pilot":
        papers = list(dict.fromkeys(e["paper_id"] for e in examples if e["split"] == "pilot"))
        papers = papers or list(dict.fromkeys(e["paper_id"] for e in examples if e["split"] == "train"))
        papers = sorted(papers[:PILOT_PAPERS])
        pilot = [e for e in examples if e["paper_id"] in papers]
        if len(papers) < PILOT_PAPERS:
            raise RunError(
                f"The pilot needs {PILOT_PAPERS} papers (one to train on, one to validate), "
                f"but the dataset has {len(papers)}"
            )
        return [{**e, "split": "train" if e["paper_id"] == papers[0] else "validation"} for e in pilot]
    chosen = [e for e in examples if e["split"] in ("train", "validation")]
    if not chosen:
        raise RunError("The dataset has no train or validation examples")
    return chosen


HEADING_TAG = re.compile(r"^h[1-6]$")

# Top-level section headings (numbering removed, case-folded) that mark each target section.
TARGET_HEADINGS = {
    "introduction": r"introduction",
    "related_work": r"related works?|background and related works?|related work and background"
    r"|prior work|previous work",
    "conclusion": r"(concluding remarks|conclusions?|summary)"
    r"( and (conclusions?|discussion|outlook|future work|future directions|perspectives?))?"
    r"|discussion and conclusions?|conclusions? and discussion",
}
# Arabic or Roman section numbers ("2.", "IV", "A.") in front of a heading, and a closing period.
SECTION_NUMBER = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[ivx]+\.?|[a-z]\.)\s+|[.:]\s*$", re.IGNORECASE)
# An acknowledgement the markup does not set apart, such as a bold run-in label inside a paragraph.
ACKNOWLEDGEMENTS = re.compile(r"\backnowledg(?:e)?ments?\b", re.IGNORECASE)
MIN_BIBLIOGRAPHY_ITEMS = 10
MIN_SECTION_WORDS = 60

# A paper is usable only if the optimizer's largest request fits Kimi K2.6's context window. That
# request holds FAILURE_EXAMPLES failures, each with the visible paper, the author's withheld
# section and the model's version of it (assumed to be the same length as the author's). This is
# a frozen selection rule: the current feedback policy sends no paper to the optimizer, and
# runs.estimate sizes that request. Changing these constants would change the dataset.
OPTIMIZER_CONTEXT = 262_144
OPTIMIZER_MAX_OUTPUT = 16_384
PROMPT_OVERHEAD = 16_000  # instructions, rubrics and grades
SAFETY_MARGIN = 1.25
FAILURE_EXAMPLES = 4  # failure_examples in configs/experiments.yaml when the corpus was built
# Papers are sized with the larger of these two token counts. Qwen is left over from the earlier
# multi-model study; it stays so that the set of eligible papers cannot change.
SIZING_MODELS = ("qwen/qwen3.5-9b", "moonshotai/kimi-k2.6")


# Figures, tables, algorithm listings and acknowledgements inside a section are not its prose, so
# they are left out of the withheld reference (the visible context keeps them elsewhere).
NON_PROSE = ".ltx_figure, .ltx_table, .ltx_float, .ltx_listing, .ltx_tabular, .ltx_acknowledgements"


# Elements that start a new line of text. Everything else is inline, so "<b>T</b>HE" reads "THE".
BLOCK_TAGS = "p, div, li, ul, ol, dl, dt, dd, section, article, header, h1, h2, h3, h4, h5, h6, br, tr, td, th, table, figure, figcaption, blockquote, pre"


def html_text(node, strip_heading=False, strip_non_prose=False):
    """Plain, normalized text of an HTML element, with each formula replaced by its TeX source.

    Works on a copy, so `node` is left unchanged. Scripts, navigation, footers and LaTeXML error
    markers are dropped, and with strip_heading so is the element's first heading and with
    strip_non_prose so are the NON_PROSE elements.
    """
    fragment = BeautifulSoup(str(node), "html.parser")
    if strip_heading:
        heading = fragment.find(HEADING_TAG)
        if heading:
            heading.decompose()
    for math_node in fragment.find_all("math"):
        tex = math_node.find("annotation", attrs={"encoding": "application/x-tex"})
        if tex:
            math_node.replace_with(tex.get_text())
        else:
            math_node.replace_with(math_node.get("alttext", math_node.get_text(" ")))
    unwanted = "script, style, nav, footer, .ltx_ERROR" + (", " + NON_PROSE if strip_non_prose else "")
    for element in fragment.select(unwanted):
        element.decompose()
    for element in fragment.select(BLOCK_TAGS):
        element.insert_before(" ")
        element.insert_after(" ")
    return normalize(fragment.get_text())


def number_citations(document):
    """Give every citation in a parsed page one numeric style, in place.

    Each in-text citation becomes its bibliography numbers, such as "[3, 7]", and each reference
    label becomes "[n]". Journal styles differ ("Smith et al. 2010", "(1)", "e.g.,)"), and LaTeXML
    renders some of them badly; a model that writes clean prose would otherwise stand out from the
    authors on formatting alone. Only formatting changes, never the wording. A citation that links
    to no reference is left as it is.
    """
    numbers = {item["id"]: n for n, item in enumerate(document.select(".ltx_bibitem[id]"), start=1)}
    for item_id, n in numbers.items():
        tag = document.find(id=item_id).select_one(".ltx_tag_bibitem")
        if tag:
            tag.string = f"[{n}]"
    for cite in document.select(".ltx_cite"):
        if cite.find_parent(class_="ltx_cite"):
            continue
        cited = sorted(
            {numbers[a["href"][1:]] for a in cite.select("a[href^='#']") if a["href"][1:] in numbers}
        )
        if cited:
            cite.replace_with("[" + ", ".join(map(str, cited)) + "]")


def largest_token_count(text):
    return max(token_count(text, model) for model in SIZING_MODELS)


def extract_paper(html, metadata):
    """Split one LaTeXML page (ar5iv or arXiv HTML) into examples, one per section in SECTIONS it has.

    Each example's reference is the text of one section, and its context is the rest of the paper
    with that section replaced by a placeholder. Raises RunError saying why a paper is unusable.
    """
    document = BeautifulSoup(html, "html.parser").select_one(".ltx_document")
    if document is None:
        raise RunError("Page has no LaTeXML document (.ltx_document)")
    number_citations(document)
    abstracts = document.select(".ltx_abstract")
    if len(abstracts) != 1:
        raise RunError(f"Page has {len(abstracts)} abstracts; expected 1")
    targets = {"abstract": abstracts[0]}
    for section in document.select(".ltx_section"):
        if section.find_parent(class_="ltx_section"):
            continue
        heading_tag = section.find(HEADING_TAG)
        if not heading_tag:
            continue
        heading = SECTION_NUMBER.sub("", heading_tag.get_text(" ", strip=True)).strip().casefold()
        for kind, pattern in TARGET_HEADINGS.items():
            if re.fullmatch(pattern, heading):
                if kind in targets:
                    raise RunError(f"Found a second top-level {kind} section, headed {heading!r}")
                targets[kind] = section
    missing = [kind for kind in REQUIRED_SECTIONS if kind not in targets]
    if missing:
        raise RunError(f"Top-level sections not found: {', '.join(missing)}")
    bibliography_items = len(document.select(".ltx_bibitem"))
    if bibliography_items < MIN_BIBLIOGRAPHY_ITEMS:
        raise RunError(
            f"Bibliography has {bibliography_items} entries; at least {MIN_BIBLIOGRAPHY_ITEMS} required"
        )
    latexml_errors = len(document.select(".ltx_ERROR"))
    replacement_chars = document.get_text().count("\ufffd")
    if latexml_errors or replacement_chars:
        raise RunError(
            f"Page has {latexml_errors} LaTeXML error node(s) and {replacement_chars} U+FFFD character(s)"
        )
    examples = []
    for kind, target in targets.items():
        reference = html_text(target, strip_heading=True, strip_non_prose=True)
        if ACKNOWLEDGEMENTS.search(reference):
            raise RunError(f"{kind} contains acknowledgements the markup does not set apart")
        if words(reference) < MIN_SECTION_WORDS:
            raise RunError(f"{kind} has {words(reference)} words; at least {MIN_SECTION_WORDS} required")
        # find(id=None) would match the first element without an id, so require one.
        if not target.get("id"):
            raise RunError(f"{kind} section has no HTML id")
        page = BeautifulSoup(str(document), "html.parser")
        removed = page.find(id=target.get("id"))
        if removed is None:
            raise RunError(f"{kind} section id {target.get('id')!r} not found in the copied page")
        removed.replace_with(f"[Missing {kind.replace('_', ' ')} section]")
        context = html_text(page)
        if reference in context:
            raise RunError(
                f"{kind} text also appears elsewhere in the paper, so removing it does not hide it"
            )
        failure_tokens = largest_token_count(context) + 2 * largest_token_count(reference)
        worst_case_prompt = FAILURE_EXAMPLES * failure_tokens
        needed = math.ceil(SAFETY_MARGIN * (worst_case_prompt + PROMPT_OVERHEAD)) + OPTIMIZER_MAX_OUTPUT
        if needed > OPTIMIZER_CONTEXT:
            raise RunError(
                f"Paper too long: with its {kind} withheld, the largest optimizer request would "
                f"need {needed} tokens, more than the {OPTIMIZER_CONTEXT}-token context"
            )
        examples.append(
            {
                "example_id": metadata["paper_id"] + "_" + kind,
                "paper_id": metadata["paper_id"],
                "section_type": kind,
                "context": context,
                "reference": reference,
                "context_hash": digest(context),
                "reference_hash": digest(reference),
                "target_words": words(reference),
                "provenance": metadata,
                # Constant apart from the count, since examples are only written once every check
                # passed. The fields stay because they are part of the hashed dataset.
                "extraction_checks": {
                    "unique_target": True,
                    "reference_removed": True,
                    "bibliography_items": bibliography_items,
                    "ocr_debris": False,
                },
            }
        )
    return examples


PAPERS_NEEDED = sum(STRATA.values())
# Papers in field order, each field in its frozen shortlist order, are dealt to the splits in these
# numbers, so that every split holds papers from most fields.
SPLIT_SIZES = {"train": 35, "validation": 16, "confirmation": 5}
# Papers about rubric optimization itself are left out, since they could describe the method.
OFF_LIMITS_TITLE = re.compile(r"rubric|XAR|unslopp", re.IGNORECASE)
# Surveys, tutorials, software and data releases are structured differently from research papers.
NOT_RESEARCH_TITLE = re.compile(
    r"\b(survey|review|tutorial|overview|roadmap|perspective|primer|lecture notes|software|package"
    r"|toolkit|library|data release|catalogue?|dataset|benchmark)\b",
    re.IGNORECASE,
)
SOURCE = "ar5iv HTML of arXiv papers first posted 2016 to 2021 with a published journal or conference version"
SOURCE_URL = "https://ar5iv.labs.arxiv.org/html/"
DOWNLOAD_PAUSE_SECONDS = 0.5
MANIFEST_EXAMPLE_FIELDS = (
    "example_id",
    "paper_id",
    "section_type",
    "split",
    "context_hash",
    "reference_hash",
    "target_words",
)


def paper_metadata(candidate):
    """Provenance for one candidate from data/discovery.json. These fields are part of the hashed dataset."""
    paper_id = candidate["paper_id"]
    return {
        **candidate,
        "source_url": SOURCE_URL + paper_id,
        "abstract_url": "https://arxiv.org/abs/" + paper_id,
        "year": int(candidate["first_posted"][:4]),
        "venue": candidate["published_venue"],
        "extraction_version": "ar5iv-v1",
        "retrieved_at": time.strftime("%Y-%m-%d"),
        "redistribution": "not verified; text stays local",
    }


def assign_splits(papers):
    """Split names for `papers`, in order: SPLIT_SIZES' counts, spread evenly along the list."""
    slots = sorted(
        ((index + 0.5) / size, split) for split, size in SPLIT_SIZES.items() for index in range(size)
    )
    if len(papers) != len(slots):
        raise RunError(f"{len(papers)} papers cannot fill the {len(slots)} slots in SPLIT_SIZES")
    return [split for _, split in slots]


def prepare_data():
    """Build data/examples.jsonl and its split and source manifests from data/discovery.json.

    Walks each field's shortlist in frozen order (downloading any HTML missing from data/raw) and
    keeps the first eligible papers up to the field's quota, listing the rejected ones in
    the log. Refuses to replace an existing examples.jsonl, splits.json or source_manifest.json
    with different content.
    """
    shortlists = load_discovery()
    raw_dir = ROOT / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    accepted, exclusions, seen_authors, seen_titles = [], [], set(), set()
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for stratum, quota in STRATA.items():
            kept = 0
            for candidate in shortlists[stratum]:
                if kept == quota:
                    break
                metadata = paper_metadata(candidate)
                paper_id, title, authors = metadata["paper_id"], metadata["title"], metadata["authors"]
                try:
                    shared_authors = sorted(set(authors) & seen_authors)
                    if shared_authors:
                        raise RunError(
                            f"Shares authors with an earlier selected paper: {', '.join(shared_authors)}"
                        )
                    if title.casefold() in seen_titles:
                        raise RunError("Same title as an earlier selected paper")
                    if OFF_LIMITS_TITLE.search(title):
                        raise RunError("Title is about rubric optimization, the method under study")
                    if NOT_RESEARCH_TITLE.search(title):
                        raise RunError("Title suggests a survey, tutorial, software or data release")
                    path = raw_dir / (paper_id + ".html")
                    if not path.exists():
                        response = client.get(metadata["source_url"])
                        response.raise_for_status()
                        if "/html/" not in str(response.url):
                            raise RunError("ar5iv has no HTML for this paper (it redirected to the abstract)")
                        path.write_text(response.text)
                        time.sleep(DOWNLOAD_PAUSE_SECONDS)
                    metadata["html_hash"] = file_hash(path)
                    extracted = extract_paper(path.read_text(), metadata)
                    accepted.append({"metadata": metadata, "examples": extracted})
                    kept += 1
                    seen_authors.update(authors)
                    seen_titles.add(title.casefold())
                    print(
                        f"Eligible {len(accepted)}/{PAPERS_NEEDED}: {stratum} {paper_id} {title}", flush=True
                    )
                except (RunError, httpx.HTTPError) as e:
                    exclusions.append({"paper_id": paper_id, "stratum": stratum, "reason": str(e)})
                    print(f"Excluded {paper_id}: {e}", flush=True)
    if len(accepted) != PAPERS_NEEDED:
        counts = Counter(p["metadata"]["stratum"] for p in accepted)
        raise RunError(
            f"Only {len(accepted)} of {PAPERS_NEEDED} papers are eligible ({dict(counts)}); see "
            "the Excluded lines above and enlarge SHORTLIST_PER_STRATUM in xar/discovery.py"
        )
    groups = {split: [] for split in SPLIT_SIZES}
    for paper, split in zip(accepted, assign_splits(accepted)):
        groups[split].append(paper)
    splits = {
        "seed": SAMPLE_SEED,
        "papers": {split: [p["metadata"]["paper_id"] for p in papers] for split, papers in groups.items()},
    }
    records = []
    for split, papers in groups.items():
        for paper in papers:
            for example in paper["examples"]:
                example["split"] = split
                records.append(example)
    dataset = "".join(canonical(example) + "\n" for example in records)
    path = ROOT / "data/examples.jsonl"
    if path.exists() and path.read_text() != dataset:
        raise RunError(
            f"The rebuilt dataset differs from the existing {path}, so it was not replaced. Check for "
            "changes to data/raw, data/discovery.json or the extraction code. To build a new dataset, "
            "move examples.jsonl, splits.json and source_manifest.json out of data/ first."
        )
    path.write_text(dataset)
    write_json(ROOT / "data/splits.json", splits, write_once=True)
    source_manifest = {
        "source": SOURCE,
        "discovery_hash": file_hash(ROOT / "data/discovery.json"),
        "dataset_hash": file_hash(path),
        "papers": [p["metadata"] for p in accepted],
        "examples": [{field: e[field] for field in MANIFEST_EXAMPLE_FIELDS} for e in records],
    }
    write_json(ROOT / "data/source_manifest.json", source_manifest, write_once=True)
    load_examples(path, ROOT / "data/splits.json")
    counts = Counter(example["split"] for example in records)
    print(f"Wrote {len(records)} examples: " + ", ".join(f"{n} {split}" for split, n in counts.items()))


def prepare_tokenizers():
    """Download the tokenizers pinned in configs/tokenizers.json and check their file checksums."""
    manifest = read_json(ROOT / "configs/tokenizers.json")
    for name, cfg in manifest.items():
        subprocess.run(
            [
                "hf",
                "download",
                cfg["repo"],
                *cfg["checksums"],
                "--revision",
                cfg["revision"],
                "--local-dir",
                str(ROOT / "data/tokenizers" / name),
            ],
            check=True,
        )
        for filename, checksum in cfg["checksums"].items():
            if file_hash(ROOT / "data/tokenizers" / name / filename) != checksum:
                raise RunError(
                    f"Downloaded data/tokenizers/{name}/{filename} does not match its checksum in "
                    "configs/tokenizers.json; check the pinned revision"
                )
    print("Tokenizer checksums match configs/tokenizers.json")
