"""Build, freeze and restore the Project Gutenberg fiction corpus.

The blog's story-continuation data: public-domain novels by prize-winning authors, biased toward each
author's lesser-read works, with the story so far as context and the author's verbatim continuation
as the reference. Passages a temperature-0 writer reproduces are dropped. Steps, in order:

  discover  Read the Gutenberg catalog, choose lesser-read novels, download them from a Gutenberg
            mirror and propose cut points. Writes data/fiction/discovery.json. Makes no paid calls.
  probe     Paid. Ask the writer at temperature 0 to continue each proposed passage and measure its
            verbatim overlap with the author's continuation. Writes data/fiction/probe.json; the raw
            requests go to the --output run directory.
  freeze    Drop memorized passages, keep EXAMPLES_PER_BOOK per book and fill the splits. Writes
            source_manifest.json, splits.json and examples.jsonl.

With no step, the script restores data/fiction/examples.jsonl from the frozen manifest, downloading
missing books; --check verifies local files without downloads or writes.
"""

from __future__ import annotations; import argparse, bz2, csv, datetime as dt, hashlib, io, json, os, random, re, tarfile, time; from collections import Counter; from pathlib import Path; import httpx, yaml; import run  # noqa: I001  # fmt: skip

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "fiction"
CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
RDF_URL = "https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2"
# Gutenberg asks robots to use a mirror and wait 2 seconds between requests (policy/robot_access.html).
MIRROR = "https://gutenberg.pglaf.org"
REQUEST_DELAY_SECONDS = 2
USER_AGENT = "Independent XAR reproduction (research; https://facebookresearch.github.io/RAM/blogs/unslop/)"
SEED = 20261005
KIND = "continuation"

# The blog uses 52 examples for meta-optimization, 20 of them held out, and 60 for final testing:
# 4 passages from each of 8 training, 5 validation and 15 confirmation books. Spare books replace
# books that lose too many passages to the probe.
EXAMPLES_PER_BOOK = 4
SPLIT_BOOKS = {"train": 8, "validation": 5, "confirmation": 15}
SPARE_BOOKS = {"train": 1, "validation": 1, "confirmation": 0}
BOOKS_PER_AUTHOR = 3
CUTS_PER_BOOK = 6
MIN_BOOK_WORDS = 30_000
MIN_CONTEXT_WORDS = 2_000
CONTINUATION_WORDS = (300, 800)
MIN_SCENE_PARAGRAPHS = 3  # prose paragraphs between the last heading or scene break and a cut
# The blog's probe: drop a passage when the temperature-0 continuation reproduces more than 5% of
# the reference's 13-grams or shares a verbatim run of 20 or more words with it.
PROBE_NGRAM = 13
PROBE_MAX_NGRAM_FRACTION = 0.05
PROBE_MAX_RUN_WORDS = 20


class RunError(RuntimeError):
    """The corpus cannot be built or does not match its frozen files."""


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    write_bytes(Path(path), (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def write_bytes(path, content):
    """Publish a complete file only after its contents have passed their checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def words(text):
    return len(text.split())


class Gutenberg:
    """Polite HTTP: one request at a time, REQUEST_DELAY_SECONDS apart."""

    def __init__(self):
        self.client = httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": USER_AGENT})
        self.last = 0.0

    def get(self, url, missing_ok=False):
        time.sleep(max(0.0, self.last + REQUEST_DELAY_SECONDS - time.monotonic()))
        try:
            response = self.client.get(url)
        finally:
            self.last = time.monotonic()
        if missing_ok and response.status_code == 404:
            return None
        response.raise_for_status()
        return response

    def book_text(self, book_id):
        """The book's UTF-8 master file from the mirror, or the generated plain text if it has none."""
        directory = "/".join(book_id[:-1]) or "0"
        for url in (f"{MIRROR}/{directory}/{book_id}/{book_id}-0.txt", f"{MIRROR}/cache/epub/{book_id}/pg{book_id}.txt"):
            response = self.get(url, missing_ok=True)
            if response is not None:
                return url, response.content
        raise RunError(f"Book {book_id} has no plain text on {MIRROR}")


# Turn a Gutenberg plain-text file into the story's paragraphs.
START = re.compile(r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG E(?:BOOK|TEXT).*$", re.IGNORECASE | re.MULTILINE)
END = re.compile(r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG E(?:BOOK|TEXT).*$", re.IGNORECASE | re.MULTILINE)
# Editorial insertions: illustrations, footnotes and sidenotes in brackets, and footnote markers.
BRACKETED = re.compile(r"\[(?:Illustration|Footnote|Sidenote|Transcriber|Note)\b[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]", re.IGNORECASE | re.DOTALL)
FOOTNOTE_MARK = re.compile(r"\[(?:\d+|[A-Z])\]")
ITALICS = re.compile(r"(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)", re.DOTALL)
DASH = re.compile(r"-{2,}")
PRODUCTION_NOTE = re.compile(
    r"^(?:produced by|e-?text prepared by|transcriber'?s? notes?|this e-?book was produced|updated editions|end of (?:the )?project gutenberg)", re.IGNORECASE
)
HEADING = re.compile(
    r"^(?:(?:chapter|book|part|volume|epilogue|prologue|interlude)\b.{0,80}|[IVXLC]+\.?|\d+\.?|[IVXLC]+\.\s.{0,80}|the end\.?)$",
    re.IGNORECASE,
)
SCENE_BREAK = re.compile(r"^[\s*·.•\-—_=#~+]+$")
# A heading that opens the story, as opposed to a preface, introduction or contents entry.
STORY_HEADING = re.compile(r"^(?:chapter|book|part)\s+(?:i|1|one|first)\b|^(?:i|1)\.?(?:\s|$)", re.IGNORECASE)


def paragraph_type(paragraph, indented):
    """prose, heading, break (a scene break) or block (indented verse, letters, tables)."""
    if SCENE_BREAK.match(paragraph):
        return "break"
    if words(paragraph) <= 12 and (HEADING.match(paragraph) or (paragraph.isupper() and not paragraph.startswith('"'))):
        return "heading"
    return "block" if indented else "prose"


def book_paragraphs(raw):
    """The story's paragraphs as [type, text] pairs, from its first chapter to its end.

    Strips the Gutenberg header and licence, editorial insertions and production notes, turns
    _italics_ into plain words and runs of hyphens into an em dash, and unwraps each paragraph onto
    one line in the shared run.normalize_text form."""
    text = raw.decode("utf-8-sig").replace("\r\n", "\n")
    start, end = START.search(text), END.search(text)
    if not start or not end:
        raise RunError("No Project Gutenberg start or end marker")
    text = text[start.end() : end.start()]
    text = FOOTNOTE_MARK.sub("", BRACKETED.sub("", text))
    text = DASH.sub("—", ITALICS.sub(r"\1", text))
    paragraphs = []
    for chunk in re.split(r"\n[ \t]*\n", text):
        lines = [line for line in chunk.split("\n") if line.strip()]
        if not lines:
            continue
        paragraph = run.normalize_text(" ".join(lines), False)
        if PRODUCTION_NOTE.match(paragraph):
            continue
        indented = all(re.match(r"^\s{2,}\S", line) for line in lines)
        paragraphs.append([paragraph_type(paragraph, indented), paragraph])
    # The story starts at its first chapter heading that is followed by prose, not by more contents entries.
    for i, (kind, paragraph) in enumerate(paragraphs):
        if kind == "heading" and STORY_HEADING.match(paragraph):
            following = paragraphs[i + 1 : i + 4]
            if any(k == "prose" and words(p) >= 40 for k, p in following):
                return paragraphs[i:]
    raise RunError("No opening chapter heading followed by prose")


def propose_cuts(paragraphs, rng):
    """Up to CUTS_PER_BOOK passages, one from each equal stretch of the story after MIN_CONTEXT_WORDS.

    A cut falls between paragraphs, at least MIN_SCENE_PARAGRAPHS prose paragraphs after the last
    heading or scene break, so the story stops mid-scene. The continuation is whole prose paragraphs
    totalling CONTINUATION_WORDS, with no heading, scene break or indented block. Returns
    {"start", "end"} paragraph indices: the context is paragraphs[:start] and the reference
    paragraphs[start:end]."""
    counts = [words(p) for _, p in paragraphs]
    before = [0]
    for count in counts:
        before.append(before[-1] + count)
    valid = []
    for start in range(MIN_SCENE_PARAGRAPHS, len(paragraphs)):
        if before[start] < MIN_CONTEXT_WORDS or any(paragraphs[i][0] != "prose" for i in range(start - MIN_SCENE_PARAGRAPHS, start)):
            continue
        target = rng.randint(*CONTINUATION_WORDS)
        end, total = start, 0
        while end < len(paragraphs) and paragraphs[end][0] == "prose" and total < target:
            total += counts[end]
            end += 1
        if CONTINUATION_WORDS[0] <= total <= CONTINUATION_WORDS[1] and total >= target:
            valid.append({"start": start, "end": end})
    story_words = before[-1] - MIN_CONTEXT_WORDS
    cuts = []
    for stretch in range(CUTS_PER_BOOK):
        low = MIN_CONTEXT_WORDS + story_words * stretch / CUTS_PER_BOOK
        high = MIN_CONTEXT_WORDS + story_words * (stretch + 1) / CUTS_PER_BOOK
        options = [cut for cut in valid if low <= before[cut["start"]] < high and (not cuts or cut["start"] >= cuts[-1]["end"])]
        if options:
            cuts.append(rng.choice(options))
    return cuts


def passage(paragraphs, cut):
    """The context and reference texts for one cut, each in the shared paragraph form."""
    context = run.normalize_text("\n".join(p for _, p in paragraphs[: cut["start"]]), True)
    reference = run.normalize_text("\n".join(p for _, p in paragraphs[cut["start"] : cut["end"]]), True)
    return context, reference


# Discovery: the catalog, download counts and the chosen books.


def catalog_books(gutenberg, authors):
    """Each listed author's English novels in the catalog, keyed by author.

    A novel has Fiction in its subjects and no Short stories, Drama, Poetry or Juvenile subject, is
    not one volume of several or a collection of stories, tales, essays or sketches, is not in
    authors.json's excluded_books, and has no author other than the listed one (illustrators aside)."""
    path = DATA / "raw" / "pg_catalog.csv"
    if not path.exists():
        write_bytes(path, gutenberg.get(CATALOG_URL).content)
    books = {author["catalog_name"]: [] for author in authors}
    excluded = {book["book_id"] for book in read_json(DATA / "authors.json")["excluded_books"]}
    for row in csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))):
        names = [name.strip() for name in row["Authors"].split(";") if name.strip()]
        principal = [name for name in names if "[" not in name]
        subjects = row["Subjects"].casefold()
        title = " ".join(row["Title"].split())
        if (
            row["Type"] == "Text"
            and row["Language"] == "en"
            and row["Text#"] not in excluded
            and len(principal) == 1
            and principal[0] in books
            and all(name in principal or name.endswith("[Illustrator]") for name in names)
            and "fiction" in subjects
            and not any(word in subjects for word in ("short stories", "drama", "poetry", "juvenile"))
            and not re.search(r"\bvol(?:ume|\.)?\s*(?:\d|[ivx]+\b)|\b(?:stories|tales|essays|sketches)\b", title, re.IGNORECASE)
        ):
            books[principal[0]].append({"book_id": row["Text#"], "title": title, "author": principal[0], "subjects": row["Subjects"]})
    return books, {"url": CATALOG_URL, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def catalog_records(gutenberg, book_ids):
    """Download counts and rights statements from the Gutenberg RDF catalog for book_ids."""
    path = DATA / "raw" / "rdf-files.tar.bz2"
    if not path.exists():
        print("Downloading the Gutenberg RDF catalog (about 130 MB)", flush=True)
        write_bytes(path, gutenberg.get(RDF_URL).content)
    wanted = {f"cache/epub/{book_id}/pg{book_id}.rdf": book_id for book_id in book_ids}
    records = {}
    with tarfile.open(fileobj=bz2.open(path), mode="r|") as archive:
        for member in archive:
            book_id = wanted.get(member.name)
            if book_id:
                rdf = archive.extractfile(member).read().decode("utf-8")
                downloads = re.search(r"<pgterms:downloads[^>]*>(\d+)</pgterms:downloads>", rdf)
                rights = re.search(r"<dcterms:rights>([^<]*)</dcterms:rights>", rdf)
                records[book_id] = {"downloads": int(downloads.group(1)) if downloads else 0, "rights": rights.group(1) if rights else ""}
    return records, {"url": RDF_URL, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def lesser_read(novels):
    """The novels downloaded less than the author's median novel; one title counts once."""
    titles = {}
    for novel in sorted(novels, key=lambda n: -n["downloads"]):
        titles.setdefault(novel["title"].split(":")[0].casefold(), novel)
    unique = list(titles.values())
    if len(unique) < 2:
        return []
    median = sorted(n["downloads"] for n in unique)[len(unique) // 2]
    return sorted((n for n in unique if n["downloads"] < median), key=lambda n: n["book_id"])


def raw_book(gutenberg, book_id):
    path = DATA / "raw" / f"{book_id}.txt"
    if path.exists():
        return read_json(path.with_suffix(".json"))["url"], path.read_bytes()
    url, content = gutenberg.book_text(book_id)
    write_bytes(path, content)
    write_json(path.with_suffix(".json"), {"url": url, "retrieved_at": dt.datetime.now(dt.UTC).isoformat()})
    print(f"Downloaded {book_id} from {url}", flush=True)
    return url, content


def discover():
    """Choose books and cut points and write discovery.json.

    Authors are shuffled with SEED and assigned whole to the first split that still needs books, so
    no author appears in two splits. Each author contributes up to BOOKS_PER_AUTHOR lesser-read
    novels, and no more than the split still needs, that are public domain in the USA, long enough
    and have CUTS_PER_BOOK passages."""
    authors = read_json(DATA / "authors.json")["authors"]
    gutenberg = Gutenberg()
    novels, catalog = catalog_books(gutenberg, authors)
    records, rdf = catalog_records(gutenberg, [n["book_id"] for books in novels.values() for n in books])
    rng = random.Random(SEED)
    order = [author["catalog_name"] for author in authors]
    rng.shuffle(order)
    credentials = {author["catalog_name"]: author["credential"] for author in authors}
    needed = {split: SPLIT_BOOKS[split] + SPARE_BOOKS[split] for split in SPLIT_BOOKS}
    chosen, rejected, considered = [], [], {}
    for author in order:
        split = next((s for s in needed if needed[s] > 0), None)
        if split is None:
            break
        for novel in novels[author]:
            novel.update(records.get(novel["book_id"], {"downloads": 0, "rights": ""}))
        candidates = [n for n in lesser_read(novels[author]) if n["rights"] == "Public domain in the USA."]
        considered[author] = [{k: n[k] for k in ("book_id", "title", "downloads", "rights")} for n in novels[author]]
        rng.shuffle(candidates)
        accepted = 0
        for novel in candidates:
            if accepted == min(BOOKS_PER_AUTHOR, needed[split]):
                break
            url, content = raw_book(gutenberg, novel["book_id"])
            try:
                paragraphs = book_paragraphs(content)
                story_words = sum(words(p) for _, p in paragraphs)
                if story_words < MIN_BOOK_WORDS:
                    raise RunError(f"Story has {story_words} words; at least {MIN_BOOK_WORDS} required")
                cuts = propose_cuts(paragraphs, random.Random(f"{SEED}:{novel['book_id']}"))
                if len(cuts) < CUTS_PER_BOOK:
                    raise RunError(f"Only {len(cuts)} valid cut points")
            except RunError as error:
                rejected.append({"book_id": novel["book_id"], "title": novel["title"], "reason": str(error)})
                continue
            accepted += 1
            book = {
                "book_id": novel["book_id"],
                "title": novel["title"],
                "author": author,
                "credential": credentials[author],
                "split": split,
                "downloads": novel["downloads"],
                "rights": novel["rights"],
                "subjects": novel["subjects"],
                "source_url": url,
                "text_hash": hashlib.sha256(content).hexdigest(),
                "story_words": story_words,
                "cuts": [],
            }
            for index, cut in enumerate(cuts):
                context, reference = passage(paragraphs, cut)
                book["cuts"].append(
                    {
                        "example_id": f"pg{novel['book_id']}_{index}",
                        **cut,
                        "context_words": words(context),
                        "target_words": words(reference),
                        "context_hash": run.digest(context),
                        "reference_hash": run.digest(reference),
                    }
                )
            chosen.append(book)
        books_added = sum(book["author"] == author for book in chosen)
        needed[split] -= books_added
    if any(count > 0 for count in needed.values()):
        raise RunError(f"Not enough eligible books; still needed: {needed}")
    write_json(
        DATA / "discovery.json",
        {
            "retrieved_at": dt.datetime.now(dt.UTC).isoformat(),
            "seed": SEED,
            "catalog": catalog,
            "rdf_catalog": rdf,
            "author_order": order,
            "considered": considered,
            "rejected": rejected,
            "books": chosen,
        },
    )
    counts = Counter(book["split"] for book in chosen)
    print(f"Chose {len(chosen)} books ({dict(counts)}) with {sum(len(b['cuts']) for b in chosen)} candidate passages")


# Probe: does the writer reproduce the author's continuation at temperature 0?


def tokens(text):
    return re.findall(r"\w+", text.casefold())


def overlap(output, reference):
    """The fraction of the reference's PROBE_NGRAM-grams the output contains, and the longest run of
    words the two share, both over casefolded words with punctuation removed."""
    out, ref = tokens(output), tokens(reference)
    ref_grams = {tuple(ref[i : i + PROBE_NGRAM]) for i in range(len(ref) - PROBE_NGRAM + 1)}
    out_grams = {tuple(out[i : i + PROBE_NGRAM]) for i in range(len(out) - PROBE_NGRAM + 1)}
    longest, previous = 0, [0] * (len(ref) + 1)
    for word in out:
        current = [0] * (len(ref) + 1)
        for j, other in enumerate(ref, start=1):
            if word == other:
                current[j] = previous[j - 1] + 1
                longest = max(longest, current[j])
        previous = current
    return {"ngram_fraction": len(ref_grams & out_grams) / len(ref_grams) if ref_grams else 0.0, "longest_run_words": longest}


def display_name(catalog_name):
    """ "Stribling, T. S. (Thomas Sigismund), 1881-1965" -> "T. S. Stribling"."""
    last, first = catalog_name.split(", ")[:2]
    return re.sub(r"\s*\(.*\)", "", first) + " " + last


def examples_from(book, cuts, paragraphs, split):
    examples = []
    provenance = {k: book[k] for k in ("book_id", "title", "credential", "source_url", "downloads", "rights", "subjects")}
    # audit_proposal looks for these names in optimizer proposals.
    provenance["authors"] = [display_name(book["author"]), book["author"].split(",")[0]]
    for cut in cuts:
        context, reference = passage(paragraphs, cut)
        if run.digest(context) != cut["context_hash"] or run.digest(reference) != cut["reference_hash"]:
            raise RunError(f"{cut['example_id']}: passage differs from its saved hash")
        examples.append(
            {
                "example_id": cut["example_id"],
                "source_id": "pg" + book["book_id"],
                "kind": KIND,
                "split": split,
                "context": context,
                "reference": reference,
                "context_hash": cut["context_hash"],
                "reference_hash": cut["reference_hash"],
                "target_words": words(reference),
                "provenance": provenance,
                "cut": {"start": cut["start"], "end": cut["end"]},
            }
        )
    return examples


def checked_book(gutenberg, book, check=False):
    """The book's paragraphs, from the local copy or the mirror, after checking the saved text hash."""
    path = DATA / "raw" / f"{book['book_id']}.txt"
    if path.exists():
        content = path.read_bytes()
    elif check:
        raise RunError(f"Missing {path}; run `uv run python fetch_fiction.py` to download it")
    else:
        response = gutenberg.get(book["source_url"])
        content = response.content
    if hashlib.sha256(content).hexdigest() != book["text_hash"]:
        raise RunError(f"Book {book['book_id']} differs from its saved text hash; the source may have changed")
    if not path.exists():
        write_bytes(path, content)
    return book_paragraphs(content)


def probe(output, config_path):
    """Continue every proposed passage at temperature 0 with the writer's model, prompt and provider."""
    discovery = read_json(DATA / "discovery.json")
    config = yaml.safe_load(Path(config_path).read_text())
    config["roles"]["writer"]["temperature"] = 0
    prompt = run.read_prompt(config, "writer")
    api = run.OpenRouter(output, config)
    Path(output).mkdir(parents=True, exist_ok=False)
    examples = []
    for book in discovery["books"]:
        examples += examples_from(book, book["cuts"], checked_book(None, book, check=True), book["split"])

    def continue_passage(example):
        response = api.call("writer", prompt, run.task_data(example), None, {"probe": example["example_id"]})
        text = run.normalize_text(response["content"], True)
        scores = overlap(text, example["reference"])
        return {
            "example_id": example["example_id"],
            **scores,
            "memorized": scores["ngram_fraction"] > PROBE_MAX_NGRAM_FRACTION or scores["longest_run_words"] >= PROBE_MAX_RUN_WORDS,
            "words": words(text),
            "finish_reason": response["finish_reason"],
            "request_key": response["request_key"],
        }

    try:
        results = run.bounded_map(continue_passage, examples, config["concurrency"], api.dispatch_stopped)
    finally:
        api.client.close()
    write_json(
        DATA / "probe.json",
        {
            "probed_at": dt.datetime.now(dt.UTC).isoformat(),
            "discovery_hash": run.digest(read_json(DATA / "discovery.json")),
            "writer": config["roles"]["writer"],
            "writer_prompt_hash": run.digest(prompt),
            "requests": str(output),
            "rule": f"memorized if more than {PROBE_MAX_NGRAM_FRACTION:.0%} of the reference's {PROBE_NGRAM}-grams "
            f"or a run of {PROBE_MAX_RUN_WORDS} or more words appear in the temperature-0 continuation",
            "results": results,
        },
    )
    print(f"Probed {len(results)} passages; {sum(r['memorized'] for r in results)} memorized")


# Freeze and restore.


def freeze():
    """Keep the first EXAMPLES_PER_BOOK unmemorized passages of each book, and the first
    SPLIT_BOOKS[split] books of each split that have that many, taking authors in turn (each
    author's 1st book, then each author's 2nd) so that dropped spares come from the most
    represented authors."""
    discovery = read_json(DATA / "discovery.json")
    probe_record = read_json(DATA / "probe.json")
    if probe_record["discovery_hash"] != run.digest(discovery):
        raise RunError("probe.json was made from a different discovery.json; run the probe again")
    memorized = {r["example_id"]: r["memorized"] for r in probe_record["results"]}
    books, manifest_examples, records = [], [], []
    for split, count in SPLIT_BOOKS.items():
        kept = 0
        in_split = [b for b in discovery["books"] if b["split"] == split]
        authors = list(dict.fromkeys(b["author"] for b in in_split))
        rank = {b["book_id"]: [x for x in in_split if x["author"] == b["author"]].index(b) for b in in_split}
        for book in sorted(in_split, key=lambda b: (rank[b["book_id"]], authors.index(b["author"]))):
            if kept == count:
                break
            cuts = [cut for cut in book["cuts"] if not memorized[cut["example_id"]]][:EXAMPLES_PER_BOOK]
            if len(cuts) < EXAMPLES_PER_BOOK:
                continue
            kept += 1
            books.append({**{k: v for k, v in book.items() if k != "cuts"}, "cuts": cuts})
            records += examples_from(book, cuts, checked_book(None, book, check=True), split)
        if kept < count:
            raise RunError(f"Only {kept} {split} books keep {EXAMPLES_PER_BOOK} passages; {count} required")
    for example in records:
        manifest_examples.append({k: example[k] for k in ("example_id", "source_id", "kind", "split", "context_hash", "reference_hash", "target_words")})
    text = jsonl(records)
    manifest = {
        "source": "Project Gutenberg novels by Nobel and Pulitzer prize-winning authors; see authors.json",
        "discovery_hash": run.digest(discovery),
        "probe_hash": run.digest(probe_record),
        "dataset_hash": run.digest(text),
        "books": books,
        "examples": manifest_examples,
    }
    splits = {"seed": SEED, "books": {split: ["pg" + b["book_id"] for b in books if b["split"] == split] for split in SPLIT_BOOKS}}
    validate_dataset(text, manifest, splits)
    write_json(DATA / "source_manifest.json", manifest)
    write_json(DATA / "splits.json", splits)
    write_bytes(DATA / "examples.jsonl", text.encode("utf-8"))
    print(f"Froze {len(books)} books and {len(records)} examples: " + ", ".join(f"{n} {s}" for s, n in Counter(e["split"] for e in records).items()))


def jsonl(records):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for r in records)


def validate_dataset(text, manifest, splits):
    """Check the exact dataset, book separation, and that no reference shows in its context."""
    if run.digest(text) != manifest["dataset_hash"]:
        raise RunError("examples.jsonl differs from the frozen dataset hash")
    examples = [json.loads(line) for line in text.splitlines() if line.strip()]
    book_splits = {book: split for split, books in splits["books"].items() for book in books}
    if len(book_splits) != sum(len(b) for b in splits["books"].values()):
        raise RunError("A book appears in more than one split")
    authors = {}
    for book in manifest["books"]:
        if authors.setdefault(book["author"], book["split"]) != book["split"]:
            raise RunError(f"{book['author']} appears in more than one split")
    if [e["example_id"] for e in examples] != [e["example_id"] for e in manifest["examples"]]:
        raise RunError("The dataset and source manifest list different examples")
    for example, expected in zip(examples, manifest["examples"]):
        eid = example["example_id"]
        if any(example.get(field) != value for field, value in expected.items()):
            raise RunError(f"{eid}: fields differ from source_manifest.json")
        if book_splits.get(example["source_id"]) != example["split"]:
            raise RunError(f"{eid}: its split differs from splits.json")
        for field in ("context", "reference"):
            if run.digest(example[field]) != example[field + "_hash"]:
                raise RunError(f"{eid}: {field} differs from its saved hash")
        if words(example["reference"]) != example["target_words"] or example["reference"] in example["context"]:
            raise RunError(f"{eid}: reference length is wrong or the reference appears in its context")
    return examples


def restore(check=False):
    if not (DATA / "source_manifest.json").exists():
        raise RunError("No frozen fiction corpus; run the discover, probe and freeze steps first")
    manifest = read_json(DATA / "source_manifest.json")
    splits = read_json(DATA / "splits.json")
    dataset = DATA / "examples.jsonl"
    if dataset.exists():
        validate_dataset(dataset.read_text(encoding="utf-8"), manifest, splits)
    elif check:
        raise RunError(f"Missing {dataset}; run `uv run python fetch_fiction.py` to build it")
    gutenberg = None if check else Gutenberg()
    records = []
    for book in manifest["books"]:
        records += examples_from(book, book["cuts"], checked_book(gutenberg, book, check), book["split"])
    text = jsonl(records)
    validate_dataset(text, manifest, splits)
    if not check and not dataset.exists():
        write_bytes(dataset, text.encode("utf-8"))
    print(f"Verified {len(manifest['books'])} books and {len(records)} examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", nargs="?", choices=("discover", "probe", "freeze"), help="build step; omit to restore the frozen corpus")
    parser.add_argument("--check", action="store_true", help="verify local files without downloads or writes")
    parser.add_argument("--output", help="probe: a new run directory for the raw requests")
    parser.add_argument("--config", default=str(ROOT / "configs/fiction.yaml"), help="probe: config with the prompts and transport settings")
    options = parser.parse_args()
    try:
        if options.step == "discover":
            discover()
        elif options.step == "probe":
            if not options.output:
                parser.error("probe needs --output")
            probe(options.output, options.config)
        elif options.step == "freeze":
            freeze()
        else:
            restore(check=options.check)
    except (RunError, run.RunError, httpx.HTTPError, OSError, ValueError) as error:
        parser.exit(2, f"STOP: {error}\n")
