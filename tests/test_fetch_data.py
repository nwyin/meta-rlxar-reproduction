"""Frozen corpus restoration, without live downloads or model dependencies."""

import hashlib
import json
import shutil
from pathlib import Path

import httpx
import pytest
from bs4 import BeautifulSoup

import fetch_data as data


def paper_html():
    def prose(prefix):
        return " ".join(f"{prefix}{i}" for i in range(80))

    references = "".join(
        f'<li class="ltx_bibitem" id="b{i}"><span class="ltx_tag_bibitem">Author {i}</span></li>'
        for i in range(10)
    )
    return (
        '<article class="ltx_document"><div class="ltx_abstract" id="abs">'
        f'<h6>Abstract</h6><p>{prose("abstract")}</p></div>'
        f'<section class="ltx_section" id="intro"><h2>IV. INTRODUCTION</h2><p>{prose("intro")}</p>'
        '<figure class="ltx_figure">Figure caption</figure>'
        '<div class="ltx_acknowledgements">We thank our colleagues</div></section>'
        f'<section class="ltx_section" id="end"><h2>Summary and Outlook</h2><p>{prose("end")}</p>'
        f'</section><ul>{references}</ul></article>'
    )


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def frozen_corpus(tmp_path):
    html = paper_html()
    metadata = {
        "paper_id": "1901.00001",
        "source_url": "https://example.org/html/1901.00001",
        "retrieved_at": "2026-10-01",
        "html_hash": data.digest(html),
    }
    examples = data.extract_paper(html, metadata)
    for example in examples:
        example["split"] = "train"
    text = "".join(data.canonical(example) + "\n" for example in examples)
    fields = (
        "example_id", "paper_id", "section_type", "split", "context_hash", "reference_hash", "target_words"
    )
    manifest = {
        "papers": [metadata],
        "examples": [{field: example[field] for field in fields} for example in examples],
        "dataset_hash": data.digest(text),
    }
    splits = {"papers": {"train": [metadata["paper_id"]], "validation": [], "confirmation": []}}
    write_json(tmp_path / "source_manifest.json", manifest)
    write_json(tmp_path / "splits.json", splits)
    return tmp_path, html, text, manifest, splits


def mock_http(monkeypatch, handler=None):
    requests = []
    original = httpx.Client

    def respond(request):
        requests.append(request)
        if handler is None:
            raise AssertionError("This operation must use only local files")
        return handler(request)

    monkeypatch.setattr(
        data.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(respond), **kwargs)
    )
    monkeypatch.setattr(data.time, "sleep", lambda seconds: None)
    return requests


def cache_html(directory, html):
    path = directory / "raw" / "1901.00001.html"
    path.parent.mkdir()
    path.write_text(html, encoding="utf-8")
    return path


def test_restore_downloads_only_missing_html_and_reuses_original_provenance(frozen_corpus, monkeypatch):
    directory, html, expected, manifest, _ = frozen_corpus
    saved_manifest = (directory / "source_manifest.json").read_bytes()
    saved_splits = (directory / "splits.json").read_bytes()
    requests = mock_http(monkeypatch, lambda request: httpx.Response(200, text=html))
    data.restore_corpus(directory)
    assert len(requests) == 1
    assert str(requests[0].url) == manifest["papers"][0]["source_url"]
    assert (directory / "examples.jsonl").read_text() == expected
    assert (directory / "raw" / "1901.00001.html").read_text() == html
    assert json.loads(expected.splitlines()[0])["provenance"]["retrieved_at"] == "2026-10-01"
    data.restore_corpus(directory)
    assert len(requests) == 1
    assert (directory / "source_manifest.json").read_bytes() == saved_manifest
    assert (directory / "splits.json").read_bytes() == saved_splits


def test_changed_download_is_rejected_before_it_is_cached(frozen_corpus, monkeypatch):
    directory, html, *_ = frozen_corpus
    mock_http(monkeypatch, lambda request: httpx.Response(200, text=html + "changed"))
    with pytest.raises(data.RunError, match="Downloaded HTML.*differs"):
        data.restore_corpus(directory)
    assert not (directory / "raw" / "1901.00001.html").exists()
    assert not (directory / "examples.jsonl").exists()


def test_changed_cached_html_is_left_untouched(frozen_corpus, monkeypatch):
    directory, html, *_ = frozen_corpus
    path = cache_html(directory, html + "changed")
    mock_http(monkeypatch)
    with pytest.raises(data.RunError, match="frozen HTML hash"):
        data.restore_corpus(directory)
    assert path.read_text() == html + "changed"
    assert not (directory / "examples.jsonl").exists()


def test_different_existing_dataset_is_left_untouched(frozen_corpus, monkeypatch):
    directory, _, expected, *_ = frozen_corpus
    path = directory / "examples.jsonl"
    path.write_text(expected + "changed")
    mock_http(monkeypatch)
    with pytest.raises(data.RunError, match="frozen dataset hash"):
        data.restore_corpus(directory)
    assert path.read_text() == expected + "changed"


def test_existing_dataset_hash_checks_original_bytes(frozen_corpus, monkeypatch):
    directory, _, expected, *_ = frozen_corpus
    path = directory / "examples.jsonl"
    content = expected.replace("\n", "\r\n").encode("utf-8")
    path.write_bytes(content)
    mock_http(monkeypatch)
    with pytest.raises(data.RunError, match="frozen dataset hash"):
        data.restore_corpus(directory)
    assert path.read_bytes() == content


def test_check_requires_local_files_and_never_writes(frozen_corpus, monkeypatch):
    directory, html, expected, *_ = frozen_corpus
    mock_http(monkeypatch)
    with pytest.raises(data.RunError, match="Missing.*examples.jsonl"):
        data.restore_corpus(directory, check=True)
    (directory / "examples.jsonl").write_text(expected)
    with pytest.raises(data.RunError, match="Missing.*1901.00001.html"):
        data.restore_corpus(directory, check=True)
    cache_html(directory, html)
    before = {path: path.stat().st_mtime_ns for path in directory.rglob("*") if path.is_file()}
    data.restore_corpus(directory, check=True)
    assert {path: path.stat().st_mtime_ns for path in before} == before


def test_split_overlap_fails_even_when_dataset_hash_matches(frozen_corpus):
    _, _, text, manifest, splits = frozen_corpus
    splits["papers"]["validation"] = list(splits["papers"]["train"])
    with pytest.raises(data.RunError, match="appears more than once"):
        data.validate_dataset(text, manifest, splits)


def test_leaked_reference_fails_even_with_consistent_hashes(frozen_corpus):
    _, _, text, manifest, splits = frozen_corpus
    examples = [json.loads(line) for line in text.splitlines()]
    example = examples[0]
    example["context"] += " " + example["reference"]
    example["context_hash"] = data.digest(example["context"])
    manifest["examples"][0]["context_hash"] = example["context_hash"]
    text = "".join(data.canonical(example) + "\n" for example in examples)
    manifest["dataset_hash"] = data.digest(text)
    with pytest.raises(data.RunError, match="withheld reference still appears"):
        data.validate_dataset(text, manifest, splits)


def test_extraction_withholds_sections_and_excludes_nonprose():
    examples = data.extract_paper(paper_html(), {"paper_id": "paper"})
    assert [example["section_type"] for example in examples] == ["abstract", "introduction", "conclusion"]
    for example in examples:
        assert example["target_words"] == 80
        assert example["reference"] not in example["context"]
        assert "Figure caption" not in example["reference"]
        assert "colleagues" not in example["reference"]
    assert "Figure caption" in examples[0]["context"]
    assert examples[0]["reference"] in examples[1]["context"]


def test_citations_and_formula_text_keep_the_original_normalization():
    document = BeautifulSoup(
        '<div><p><b>T</b>HE formula <math><annotation encoding="application/x-tex">x^2</annotation>'
        '</math> <cite class="ltx_cite"><a href="#b2">Lee</a>, <a href="#b1">Kim</a></cite>.</p>'
        '<p>Next paragraph.</p><ul><li class="ltx_bibitem" id="b1">'
        '<span class="ltx_tag_bibitem">Kim</span></li><li class="ltx_bibitem" id="b2">'
        '<span class="ltx_tag_bibitem">Lee</span></li></ul></div>',
        "html.parser",
    )
    data.number_citations(document)
    assert data.html_text(document) == "THE formula x^2 [1, 2]. Next paragraph. [1] [2]"


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ('id="abs"', "", "abstract section has no HTML id"),
        ("INTRODUCTION", "METHODS", "Top-level sections not found: introduction"),
        ("end79", "end79 Acknowledgements We thank the funders", "contains acknowledgements"),
    ],
)
def test_extraction_rejects_unusable_section_markup(old, new, message):
    with pytest.raises(data.RunError, match=message):
        data.extract_paper(paper_html().replace(old, new), {"paper_id": "paper"})


def test_frozen_corpus_rebuild_matches_saved_hash_offline(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[1] / "data"
    manifest = data.read_json(source / "source_manifest.json")
    raw = [source / "raw" / (paper["paper_id"] + ".html") for paper in manifest["papers"]]
    if not all(path.exists() for path in raw):
        pytest.skip("The local paper cache is unavailable")
    for filename in ("source_manifest.json", "splits.json"):
        shutil.copyfile(source / filename, tmp_path / filename)
    (tmp_path / "raw").mkdir()
    for path in raw:
        (tmp_path / "raw" / path.name).symlink_to(path)
    mock_http(monkeypatch)
    data.restore_corpus(tmp_path)
    rebuilt = (tmp_path / "examples.jsonl").read_bytes()
    assert hashlib.sha256(rebuilt).hexdigest() == manifest["dataset_hash"]
    if (source / "examples.jsonl").exists():
        assert rebuilt == (source / "examples.jsonl").read_bytes()
