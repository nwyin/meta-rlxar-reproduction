"""The frozen dataset: loading and checking examples, model inputs, and paper extraction."""

from __future__ import annotations

import json
import math
import random
import re
import subprocess
import time
import xml.etree.ElementTree as ET
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


def load_examples(dataset, splits):
    split_data = read_json(splits)
    groups = split_data["papers"]
    flat = [p for papers in groups.values() for p in papers]
    if len(flat) != len(set(flat)):
        raise RunError("Paper splits overlap")
    examples = [json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()]
    ids = [e["example_id"] for e in examples]
    if len(ids) != len(set(ids)):
        raise RunError("Duplicate example IDs")
    counts = {}
    for e in examples:
        matching = [s for s, papers in groups.items() if e["paper_id"] in papers]
        if matching != [e["split"]]:
            raise RunError("Paper/example split mismatch")
        if e["context_hash"] != digest(e["context"]) or e["reference_hash"] != digest(e["reference"]):
            raise RunError("Example content hash mismatch")
        if e["target_words"] != words(e["reference"]) or not e["target_words"]:
            raise RunError("Target word count mismatch")
        if normalize(e["reference"]) in normalize(e["context"]):
            raise RunError("Withheld reference remains in visible context")
        counts.setdefault(e["paper_id"], []).append(e["section_type"])
    for p, sections in counts.items():
        if sorted(sections) != sorted(SECTIONS):
            raise RunError(f"Missing/duplicate target sections: {p}")
    return examples


def task_data(example):
    # This allowlist prevents expert text, split labels, IDs, or prior feedback entering G/J.
    return {
        "visible_paper": example["context"],
        "section_type": example["section_type"],
        "target_words": example["target_words"],
    }


def contamination(candidate, reference):
    a, b = normalize(candidate).split(), normalize(reference).split()
    positions = {}
    for i, word in enumerate(b):
        positions.setdefault(word, []).append(i)
    longest, prior = 0, {}
    for word in a:
        current = {j: prior.get(j - 1, 0) + 1 for j in positions.get(word, [])}
        longest = max(longest, max(current.values(), default=0))
        prior = current
    reference_ngrams = {tuple(b[i : i + 8]) for i in range(max(0, len(b) - 7))}
    overlap = sum(tuple(a[i : i + 8]) in reference_ngrams for i in range(max(0, len(a) - 7)))
    return {
        "longest_verbatim_run_words": longest,
        "eightgram_overlap_fraction": overlap / max(1, len(a) - 7),
        "flagged": longest >= 30,
        "exclusion": False,
    }


def selected_examples(args):
    examples = load_examples(args.dataset, args.splits)
    chosen = (
        [e for e in examples if e["split"] in ("train", "validation")]
        if args.split == "research"
        else [e for e in examples if e["split"] == args.split]
    )
    if not chosen:
        raise RunError("Requested split has no examples")
    return chosen


def html_text(node, strip_heading=False):
    clone = BeautifulSoup(str(node), "html.parser")
    if strip_heading:
        head = clone.find(re.compile(r"^h[1-6]$"))
        if head:
            head.decompose()
    for math_node in clone.find_all("math"):
        annotation = math_node.find("annotation", attrs={"encoding": "application/x-tex"})
        math_node.replace_with(
            annotation.get_text() if annotation else math_node.get("alttext", math_node.get_text(" "))
        )
    for n in clone.select("script, style, nav, footer, .ltx_ERROR"):
        n.decompose()
    return normalize(clone.get_text(" ", strip=True))


def extract_paper(html, metadata):
    soup = BeautifulSoup(html, "html.parser")
    document = soup.select_one(".ltx_document")
    if document is None:
        raise RunError("No LaTeXML document")
    abstracts = document.select(".ltx_abstract")
    if len(abstracts) != 1:
        raise RunError("Missing or duplicated abstract")
    targets = {"abstract": abstracts[0]}
    patterns = {
        "introduction": r"^introduction$",
        "related_work": r"^(related work|related works)$",
        "conclusion": r"^(conclusion|conclusions|conclusion and future work|conclusions and future work)$",
    }
    for section in document.select(".ltx_section"):
        if section.find_parent(class_="ltx_section"):
            continue
        head = section.find(re.compile(r"^h[1-6]$"))
        heading = re.sub(r"^\s*[\d.]+\s*", "", head.get_text(" ", strip=True)).casefold() if head else ""
        for kind, pattern in patterns.items():
            if re.fullmatch(pattern, heading):
                if kind in targets:
                    raise RunError("Ambiguous target boundary")
                targets[kind] = section
    if set(targets) != set(SECTIONS):
        raise RunError("Required top-level sections missing")
    if len(document.select(".ltx_bibitem")) < 10:
        raise RunError("Incomplete or short bibliography")
    if document.select(".ltx_ERROR") or "�" in document.get_text():
        raise RunError("Extraction debris")
    examples = []
    for kind, target in targets.items():
        reference = html_text(target, strip_heading=True)
        if words(reference) < 60:
            raise RunError("Target section under 60 words")
        # find(id=None) would match the first element without an id, so require one.
        if not target.get("id"):
            raise RunError(f"{kind} section has no HTML id")
        context_document = BeautifulSoup(str(document), "html.parser")
        remove = context_document.find(id=target.get("id"))
        if remove is None:
            raise RunError("Target lacks unique HTML ID")
        remove.replace_with(f"[Missing {kind.replace('_', ' ')} section]")
        context = html_text(context_document)
        if reference in context:
            raise RunError("Withheld text remains duplicated in context")
        # Largest optimizer payload uses four full failure papers, candidate pairs, rubrics, evidence.
        count = max(token_count(context, m) for m in ("qwen/qwen3.5-9b", "moonshotai/kimi-k2.6"))
        reference_count = max(token_count(reference, m) for m in ("qwen/qwen3.5-9b", "moonshotai/kimi-k2.6"))
        if math.ceil(1.25 * (4 * count + 8 * reference_count + 16000)) + 16384 > 262144:
            raise RunError("Conservative four-failure optimizer context bound exceeded")
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
                "extraction_checks": {
                    "unique_target": True,
                    "reference_removed": True,
                    "bibliography_items": len(document.select(".ltx_bibitem")),
                    "ocr_debris": False,
                },
            }
        )
    return examples


def prepare_data():
    policy = read_json(ROOT / "data/acquisition_policy.json")
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    entries = ET.parse(ROOT / "data/discovery.xml").getroot().findall("a:entry", ns)
    raw_dir = ROOT / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    accepted, exclusions, seen_authors, seen_titles = [], [], set(), set()
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for entry in entries:
            paper_id = entry.find("a:id", ns).text.split("/abs/")[-1]
            title = normalize(entry.find("a:title", ns).text)
            authors = [normalize(a.find("a:name", ns).text) for a in entry.findall("a:author", ns)]
            ref = entry.find("x:journal_ref", ns)
            metadata = {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "source_url": "https://arxiv.org/html/" + paper_id,
                "abstract_url": "https://arxiv.org/abs/" + paper_id,
                "year": int(entry.find("a:published", ns).text[:4]),
                "venue": ref.text if ref is not None else "arXiv preprint; peer review not verified",
                "extraction_version": "latexhtml-v1",
                "retrieved_at": "2026-09-29",
                "redistribution": "not verified; text stays local",
            }
            try:
                if set(authors) & seen_authors or title.casefold() in seen_titles:
                    raise RunError("Author or manuscript overlap with prior selected paper")
                if re.search(r"rubric|XAR|unslopp", title, re.IGNORECASE):
                    raise RunError("Subject overlaps rubric optimization")
                path = raw_dir / (paper_id + ".html")
                if not path.exists():
                    response = client.get(metadata["source_url"])
                    response.raise_for_status()
                    path.write_text(response.text)
                    time.sleep(0.25)
                metadata["html_hash"] = file_hash(path)
                extracted = extract_paper(path.read_text(), metadata)
                metadata["inspection"] = {"agent_extraction_check": "passed", "human_review": "pending"}
                accepted.append({"metadata": metadata, "examples": extracted})
                seen_authors.update(authors)
                seen_titles.add(title.casefold())
                print(f"Eligible {len(accepted)}/20: {paper_id} {title}", flush=True)
                if len(accepted) == 20:
                    break
            except (RunError, httpx.HTTPError) as e:
                exclusions.append({"paper_id": paper_id, "reason": str(e), "before_grading": True})
                print(f"Excluded {paper_id}: {e}", flush=True)
    write_json(ROOT / "data/exclusions.json", exclusions)
    if len(accepted) != 20:
        raise RunError(
            f"Only {len(accepted)} of 20 papers are eligible; see data/exclusions.json "
            "and add more entries to data/discovery.xml"
        )
    research = accepted[2:15]
    random.Random(20260929).shuffle(research)
    groups = {
        "pilot": accepted[:2],
        "train": research[:8],
        "validation": research[8:],
        "confirmation": accepted[15:],
    }
    splits = {
        "seed": 20260929,
        "policy_hash": file_hash(ROOT / "data/acquisition_policy.json"),
        "papers": {s: [p["metadata"]["paper_id"] for p in papers] for s, papers in groups.items()},
    }
    records = []
    for split, papers in groups.items():
        for paper in papers:
            for e in paper["examples"]:
                e["split"] = split
                records.append(e)
    dataset = "".join(canonical(e) + "\n" for e in records)
    path = ROOT / "data/examples.jsonl"
    if path.exists() and path.read_text() != dataset:
        raise RunError("Frozen dataset differs; never silently replace splits")
    path.write_text(dataset)
    write_json(ROOT / "data/splits.json", splits, immutable=True)
    write_json(
        ROOT / "data/source_manifest.json",
        {
            "source": policy["source"],
            "source_deviation": policy["source_deviation"],
            "policy_hash": digest(policy),
            "discovery_hash": file_hash(ROOT / "data/discovery.xml"),
            "dataset_hash": file_hash(path),
            "papers": [p["metadata"] for p in accepted],
            "examples": [
                {
                    k: e[k]
                    for k in (
                        "example_id",
                        "paper_id",
                        "section_type",
                        "split",
                        "context_hash",
                        "reference_hash",
                        "target_words",
                    )
                }
                for e in records
            ],
        },
        immutable=True,
    )
    load_examples(path, ROOT / "data/splits.json")
    print("Frozen 80 examples: 8 pilot, 32 train, 20 validation, 20 confirmation")


def prepare_tokenizers():
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
                raise RunError(f"Pinned tokenizer checksum differs: {name}/{filename}")
    print("Pinned official tokenizer checksums verified")
