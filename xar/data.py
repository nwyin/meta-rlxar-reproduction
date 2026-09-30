"""The paper dataset: building it from arXiv HTML, loading and checking it, and what models see of it."""

from __future__ import annotations

import json
import math
import random
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

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

# contamination() measures overlap in 8-word n-grams and flags a candidate that copies a run of
# 30 or more consecutive words from the author's section.
NGRAM = 8
VERBATIM_FLAG_WORDS = 30


def load_examples(dataset, splits):
    """Read the examples from `dataset` (JSON lines) and check them against `splits`.

    Checks that no paper is in two splits, each example's split and content hashes match, the
    withheld section is absent from the visible paper, and every paper has each section once.
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
        if sorted(sections) != sorted(SECTIONS):
            raise RunError(
                f"Paper {paper} has sections {sorted(sections)}; expected one each of {list(SECTIONS)}"
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


def run_examples(examples, split):
    """The examples a run uses, labelled train or validation.

    A research run uses the train and validation papers as saved. A pilot run uses the pilot
    papers: the first (sorted by ID) becomes training and the rest become validation.
    """
    if split == "pilot":
        pilot = [e for e in examples if e["split"] == "pilot"]
        papers = sorted({e["paper_id"] for e in pilot})
        if len(papers) < 2:
            raise RunError(
                f"The pilot needs at least 2 papers (one to train on, one to validate), "
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
    "related_work": r"related works?",
    "conclusion": r"conclusions?( and future work)?",
}
MIN_BIBLIOGRAPHY_ITEMS = 10
MIN_SECTION_WORDS = 60

# A paper is only usable if the optimizer's largest possible request fits Kimi K2.6's context. That
# request holds FAILURE_EXAMPLES failures, each with the visible paper plus the author's and the
# model's version of the withheld section (the model's assumed as long as the author's).
OPTIMIZER_CONTEXT = 262_144
OPTIMIZER_MAX_OUTPUT = 16_384
PROMPT_OVERHEAD = 16_000  # instructions, rubrics and grades
SAFETY_MARGIN = 1.25
FAILURE_EXAMPLES = 4  # failure_examples in configs/experiments.yaml
# Papers are sized with the larger of these two token counts. Qwen is left over from the earlier
# multi-model study; it stays so that the set of eligible papers cannot change.
SIZING_MODELS = ("qwen/qwen3.5-9b", "moonshotai/kimi-k2.6")


def html_text(node, strip_heading=False):
    """Plain, normalized text of an HTML element, with each formula replaced by its TeX source.

    Works on a copy, so `node` is left unchanged. Scripts, navigation, footers and LaTeXML error
    markers are dropped, and with strip_heading so is the element's first heading.
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
    for element in fragment.select("script, style, nav, footer, .ltx_ERROR"):
        element.decompose()
    return normalize(fragment.get_text(" ", strip=True))


def max_tokens(text):
    return max(token_count(text, model) for model in SIZING_MODELS)


def extract_paper(html, metadata):
    """Split one arXiv LaTeXML page into four examples, one per section in SECTIONS.

    Each example's reference is the text of one section, and its context is the rest of the paper
    with that section replaced by a placeholder. Raises RunError saying why a paper is unusable.
    """
    document = BeautifulSoup(html, "html.parser").select_one(".ltx_document")
    if document is None:
        raise RunError("Page has no LaTeXML document (.ltx_document)")
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
        heading = re.sub(r"^\s*[\d.]+\s*", "", heading_tag.get_text(" ", strip=True)).casefold()
        for kind, pattern in TARGET_HEADINGS.items():
            if re.fullmatch(pattern, heading):
                if kind in targets:
                    raise RunError(f"More than one top-level section is headed like {kind}")
                targets[kind] = section
    missing = [kind for kind in SECTIONS if kind not in targets]
    if missing:
        raise RunError(f"Top-level sections not found: {', '.join(missing)}")
    bibliography_items = len(document.select(".ltx_bibitem"))
    if bibliography_items < MIN_BIBLIOGRAPHY_ITEMS:
        raise RunError(
            f"Bibliography has {bibliography_items} entries; at least {MIN_BIBLIOGRAPHY_ITEMS} required"
        )
    if document.select(".ltx_ERROR") or "�" in document.get_text():
        raise RunError("Page contains LaTeXML errors or U+FFFD replacement characters")
    examples = []
    for kind, target in targets.items():
        reference = html_text(target, strip_heading=True)
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
        worst_case_prompt = FAILURE_EXAMPLES * (max_tokens(context) + 2 * max_tokens(reference))
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


PAPERS_NEEDED = 20
# Accepted papers in discovery order: the first 2 are pilots, the last 5 are held back for
# confirmation, and the middle 13 are shuffled with SPLIT_SEED into 8 train and 5 validation papers.
PILOT_PAPERS = slice(0, 2)
RESEARCH_PAPERS = slice(2, 15)
CONFIRMATION_PAPERS = slice(15, 20)
TRAIN_PAPERS = 8
SPLIT_SEED = 20260929
# Papers about rubric optimization itself are left out, since they could describe the method.
OFF_LIMITS_TITLE = re.compile(r"rubric|XAR|unslopp", re.IGNORECASE)
DOWNLOAD_PAUSE_SECONDS = 0.25
ATOM = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
MANIFEST_EXAMPLE_FIELDS = (
    "example_id",
    "paper_id",
    "section_type",
    "split",
    "context_hash",
    "reference_hash",
    "target_words",
)


def discovery_metadata(entry):
    """Provenance for one arXiv Atom entry. These fields are part of the hashed dataset."""
    paper_id = entry.find("a:id", ATOM).text.split("/abs/")[-1]
    journal_ref = entry.find("x:journal_ref", ATOM)
    return {
        "paper_id": paper_id,
        "title": normalize(entry.find("a:title", ATOM).text),
        "authors": [
            normalize(author.find("a:name", ATOM).text) for author in entry.findall("a:author", ATOM)
        ],
        "source_url": "https://arxiv.org/html/" + paper_id,
        "abstract_url": "https://arxiv.org/abs/" + paper_id,
        "year": int(entry.find("a:published", ATOM).text[:4]),
        "venue": journal_ref.text if journal_ref is not None else "arXiv preprint; peer review not verified",
        "extraction_version": "latexhtml-v1",
        "retrieved_at": "2026-09-29",
        "redistribution": "not verified; text stays local",
    }


def prepare_data():
    """Build data/examples.jsonl and its split and source manifests from data/discovery.xml.

    Takes the first PAPERS_NEEDED eligible papers in discovery order (downloading any HTML missing
    from data/raw), lists the rejected ones in data/exclusions.json, and refuses to replace
    existing dataset files with different content.
    """
    policy = read_json(ROOT / "data/acquisition_policy.json")
    entries = ET.parse(ROOT / "data/discovery.xml").getroot().findall("a:entry", ATOM)
    raw_dir = ROOT / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    accepted, exclusions, seen_authors, seen_titles = [], [], set(), set()
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for entry in entries:
            metadata = discovery_metadata(entry)
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
                path = raw_dir / (paper_id + ".html")
                if not path.exists():
                    response = client.get(metadata["source_url"])
                    response.raise_for_status()
                    path.write_text(response.text)
                    time.sleep(DOWNLOAD_PAUSE_SECONDS)
                metadata["html_hash"] = file_hash(path)
                extracted = extract_paper(path.read_text(), metadata)
                # The examples' provenance is this same dict, so this lands in every example.
                # "pending" is stale (data/human_review.json records the approval), but it is part
                # of the hashed dataset, so it stays.
                metadata["inspection"] = {"agent_extraction_check": "passed", "human_review": "pending"}
                accepted.append({"metadata": metadata, "examples": extracted})
                seen_authors.update(authors)
                seen_titles.add(title.casefold())
                print(f"Eligible {len(accepted)}/{PAPERS_NEEDED}: {paper_id} {title}", flush=True)
                if len(accepted) == PAPERS_NEEDED:
                    break
            except (RunError, httpx.HTTPError) as e:
                exclusions.append({"paper_id": paper_id, "reason": str(e), "before_grading": True})
                print(f"Excluded {paper_id}: {e}", flush=True)
    write_json(ROOT / "data/exclusions.json", exclusions)
    if len(accepted) != PAPERS_NEEDED:
        raise RunError(
            f"Only {len(accepted)} of {PAPERS_NEEDED} papers are eligible; see data/exclusions.json "
            "and add more entries to data/discovery.xml"
        )
    research = accepted[RESEARCH_PAPERS]
    random.Random(SPLIT_SEED).shuffle(research)
    groups = {
        "pilot": accepted[PILOT_PAPERS],
        "train": research[:TRAIN_PAPERS],
        "validation": research[TRAIN_PAPERS:],
        "confirmation": accepted[CONFIRMATION_PAPERS],
    }
    splits = {
        "seed": SPLIT_SEED,
        "policy_hash": file_hash(ROOT / "data/acquisition_policy.json"),
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
            f"The rebuilt dataset differs from the existing {path} (arXiv pages may have changed); "
            "not replacing it. Move the old file aside to rebuild."
        )
    path.write_text(dataset)
    write_json(ROOT / "data/splits.json", splits, write_once=True)
    source_manifest = {
        "source": policy["source"],
        "source_deviation": policy["source_deviation"],
        "policy_hash": digest(policy),
        "discovery_hash": file_hash(ROOT / "data/discovery.xml"),
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
                *cfg["files"],
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
                    f"data/tokenizers/{name}/{filename} does not match its checksum in configs/tokenizers.json"
                )
    print("Tokenizer checksums match configs/tokenizers.json")
