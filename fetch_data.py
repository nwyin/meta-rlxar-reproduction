"""Restore the frozen paper corpus and verify it against its saved hashes.

Run `uv run python fetch_data.py` to download missing HTML and build data/examples.jsonl.
Run with --check to validate the local corpus without downloading or writing anything.
Paper selection and provenance live in data/source_manifest.json and data/splits.json.
"""

from __future__ import annotations; import argparse, hashlib, json, os, re, time; from collections import Counter; from pathlib import Path; import httpx; from bs4 import BeautifulSoup  # noqa: I001  # fmt: skip

ROOT = Path(__file__).resolve().parent
REQUIRED_SECTIONS = ("abstract", "introduction")


class RunError(RuntimeError):
    """The local or downloaded corpus does not match its frozen manifests."""


def digest(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def words(text):
    return len(text.split())


def normalize(text):
    return " ".join(text.split())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_bytes(path, content):
    """Publish a complete file only after its contents have passed their checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


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
        math_node.replace_with(tex.get_text() if tex else math_node.get("alttext", math_node.get_text(" ")))
    unwanted = "script, style, nav, footer, .ltx_ERROR" + (", " + NON_PROSE if strip_non_prose else "")
    for element in fragment.select(unwanted):
        element.decompose()
    for element in fragment.select(BLOCK_TAGS):
        element.insert_before(" ")
        element.insert_after(" ")
    return normalize(fragment.get_text())


def extract_paper(html, metadata):
    """Split one LaTeXML page into examples, one per supported section it has.

    Each example's reference is the text of one section, and its context is the rest of the paper
    with that section replaced by a placeholder. Raises RunError saying why a paper is unusable.
    """
    document = BeautifulSoup(html, "html.parser").select_one(".ltx_document")
    if document is None:
        raise RunError("Page has no LaTeXML document (.ltx_document)")
    # Normalize citation formatting so it cannot signal who wrote a section. Preserve wording
    # and leave citations with no bibliography link unchanged.
    items = document.select(".ltx_bibitem[id]")
    numbers = {item["id"]: n for n, item in enumerate(items, start=1)}
    for item in items:
        if tag := item.select_one(".ltx_tag_bibitem"):
            tag.string = f"[{numbers[item['id']]}]"
    for cite in document.select(".ltx_cite"):
        if cite.find_parent(class_="ltx_cite"):
            continue
        cited = sorted({numbers[a["href"][1:]] for a in cite.select("a[href^='#']") if a["href"][1:] in numbers})
        if cited:
            cite.replace_with("[" + ", ".join(map(str, cited)) + "]")
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
        raise RunError(f"Bibliography has {bibliography_items} entries; at least {MIN_BIBLIOGRAPHY_ITEMS} required")
    latexml_errors = len(document.select(".ltx_ERROR"))
    replacement_chars = document.get_text().count("\ufffd")
    if latexml_errors or replacement_chars:
        raise RunError(f"Page has {latexml_errors} LaTeXML error node(s) and {replacement_chars} U+FFFD character(s)")
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
            raise RunError(f"{kind} text also appears elsewhere in the paper, so removing it does not hide it")
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


def validate_dataset(text, manifest, splits):
    """Check the exact dataset plus paper separation and withheld reference integrity."""
    if digest(text) != manifest["dataset_hash"]:
        raise RunError("examples.jsonl differs from the frozen dataset hash; the file was left unchanged")
    examples = [json.loads(line) for line in text.splitlines() if line.strip()]
    if len(examples) != len(manifest["examples"]):
        raise RunError("The dataset and source manifest have different example counts")
    paper_splits = {}
    for split, papers in splits["papers"].items():
        for paper in papers:
            if paper in paper_splits:
                raise RunError(f"Paper {paper} appears more than once in splits.json")
            paper_splits[paper] = split
    metadata = {paper["paper_id"]: paper for paper in manifest["papers"]}
    if len(metadata) != len(manifest["papers"]) or set(metadata) != set(paper_splits):
        raise RunError("source_manifest.json and splits.json must list the same papers exactly once")
    ids, sections = set(), {}
    for example, expected in zip(examples, manifest["examples"]):
        eid, paper = example["example_id"], example["paper_id"]
        if eid in ids or any(example.get(field) != value for field, value in expected.items()):
            raise RunError(f"{eid}: duplicate example or fields differ from source_manifest.json")
        ids.add(eid)
        if paper_splits.get(paper) != example["split"]:
            raise RunError(f"{eid}: its split differs from splits.json")
        if example["provenance"] != metadata.get(paper):
            raise RunError(f"{eid}: its provenance differs from source_manifest.json")
        for field in ("context", "reference"):
            if digest(example[field]) != example[field + "_hash"]:
                raise RunError(f"{eid}: {field} text differs from its saved hash")
        if not example["target_words"] or words(example["reference"]) != example["target_words"]:
            raise RunError(f"{eid}: reference length differs from target_words or is empty")
        if normalize(example["reference"]) in normalize(example["context"]):
            raise RunError(f"{eid}: the withheld reference still appears in the visible paper")
        sections.setdefault(paper, []).append(example["section_type"])
    if set(sections) != set(metadata):
        raise RunError("Some manifest papers have no examples")
    for paper, kinds in sections.items():
        if len(kinds) != len(set(kinds)) or not set(REQUIRED_SECTIONS) <= set(kinds):
            raise RunError(f"{paper}: requires one abstract, one introduction, and no repeated sections")
    return examples


def check_html(path, metadata):
    if not path.exists():
        raise RunError(f"Missing {path}; run `uv run python fetch_data.py` to download it")
    html = path.read_bytes().decode("utf-8")
    if digest(html) != metadata["html_hash"]:
        raise RunError(f"{path} differs from its frozen HTML hash; the file was left unchanged")
    return html


def restore_corpus(data_dir=ROOT / "data", *, check=False):
    """Restore only the selected papers, using their original saved provenance and order."""
    data_dir = Path(data_dir)
    manifest = read_json(data_dir / "source_manifest.json")
    splits = read_json(data_dir / "splits.json")
    dataset_path = data_dir / "examples.jsonl"
    if dataset_path.exists():
        validate_dataset(dataset_path.read_bytes().decode("utf-8"), manifest, splits)
    elif check:
        raise RunError(f"Missing {dataset_path}; run `uv run python fetch_data.py` to build it")
    if check:
        for paper in manifest["papers"]:
            check_html(data_dir / "raw" / (paper["paper_id"] + ".html"), paper)
        print(f"Verified {len(manifest['papers'])} papers and {len(manifest['examples'])} examples")
        return

    extracted = {}
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for paper in manifest["papers"]:
            path = data_dir / "raw" / (paper["paper_id"] + ".html")
            if not path.exists():
                response = client.get(paper["source_url"])
                response.raise_for_status()
                # The original fetch saved decoded HTML as UTF-8, rather than response bytes.
                if digest(response.text) != paper["html_hash"]:
                    raise RunError(
                        f"Downloaded HTML for {paper['paper_id']} differs from its frozen hash; "
                        "the source may have changed, so the page was not saved"
                    )
                write_bytes(path, response.text.encode("utf-8"))
                print(f"Downloaded {paper['paper_id']}", flush=True)
                time.sleep(0.5)
            for example in extract_paper(check_html(path, paper), paper):
                extracted[example["example_id"]] = example
    records = []
    for expected in manifest["examples"]:
        eid = expected["example_id"]
        if eid not in extracted:
            raise RunError(f"{eid}: extraction no longer produces a saved section")
        example = extracted.pop(eid)
        example["split"] = expected["split"]
        records.append(example)
    if extracted:
        raise RunError("Extraction produced sections absent from the frozen source manifest")
    text = "".join(json.dumps(example, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for example in records)
    validate_dataset(text, manifest, splits)
    if not dataset_path.exists():
        write_bytes(dataset_path, text.encode("utf-8"))
    counts = Counter(example["split"] for example in records)
    print(
        f"Verified {len(manifest['papers'])} papers and {len(records)} examples: "
        + ", ".join(f"{number} {split}" for split, number in counts.items())
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify local files without downloads or writes")
    options = parser.parse_args()
    try:
        restore_corpus(check=options.check)
    except (RunError, httpx.HTTPError, OSError, ValueError) as error:
        parser.exit(2, f"STOP: {error}\n")
