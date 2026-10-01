"""Dataset loading, paper extraction and the fields models see."""

import pytest
from bs4 import BeautifulSoup
from conftest import dummy_example

from xar import data
from xar.data import (
    SECTIONS,
    SPLIT_SIZES,
    assign_splits,
    extract_paper,
    html_text,
    load_examples,
    number_citations,
    run_examples,
    task_data,
)
from xar.util import RunError, canonical, write_json


def test_task_data_sends_only_visible_fields():
    e = dummy_example()
    assert set(task_data(e)) == {"visible_paper", "section_type", "target_words"}
    assert e["reference"] not in canonical(task_data(e))


def test_load_examples_rejects_a_paper_in_two_splits(tmp_path):
    example = dummy_example()
    path = tmp_path / "examples.jsonl"
    path.write_text(canonical(example) + "\n")
    splits = tmp_path / "splits.json"
    write_json(splits, {"papers": {"train": ["paper1"], "validation": ["paper1"]}})
    with pytest.raises(RunError, match="overlap"):
        load_examples(path, splits)


def test_run_examples_splits_pilot_papers_into_train_and_validation():
    examples = [
        dummy_example(paper, section, "pilot") for paper in ("pilotB", "pilotA") for section in SECTIONS
    ]
    examples.append(dummy_example("paper9", "abstract", "train"))
    labelled = run_examples(examples, "pilot")
    assert {e["paper_id"]: e["split"] for e in labelled} == {"pilotA": "train", "pilotB": "validation"}
    with pytest.raises(RunError, match="needs 2 papers"):
        run_examples(examples[:4], "pilot")


def test_run_examples_pilot_uses_the_first_training_papers_when_no_pilot_split_exists():
    examples = [
        dummy_example(paper, section, "train")
        for paper in ("paperC", "paperA", "paperB")
        for section in SECTIONS
    ]
    labelled = run_examples(examples, "pilot")
    # Papers C and A come first in the dataset; A sorts first, so it trains and C validates.
    assert {e["paper_id"]: e["split"] for e in labelled} == {"paperA": "train", "paperC": "validation"}
    with pytest.raises(RunError, match="needs 2 papers"):
        run_examples(examples[: len(SECTIONS)], "pilot")


def test_run_examples_rejects_an_unknown_split():
    with pytest.raises(RunError, match="Unknown split 'confirmation'"):
        run_examples([dummy_example()], "confirmation")


def latexml_page(abstract_id, titles=("Introduction", "Related Work", "Conclusions"), numbers=None):
    """A minimal LaTeXML page with an abstract, one section per title and a 10-item bibliography."""

    def body(name):
        return " ".join(f"{name}{i}" for i in range(80))

    abstract_attrs = f' id="{abstract_id}"' if abstract_id else ""
    sections = "".join(
        f'<section class="ltx_section" id="S{i}"><h2>{(numbers or str(i))} {title}</h2>'
        f"<p>{body(f's{i}w')}</p></section>"
        for i, title in enumerate(titles, start=1)
    )
    bibliography = "".join(f'<li class="ltx_bibitem">Ref {i}</li>' for i in range(10))
    return (
        f'<article class="ltx_document"><div class="ltx_abstract"{abstract_attrs}>'
        f"<h6>Abstract</h6><p>{body('abs')}</p></div>{sections}<ul>{bibliography}</ul></article>"
    )


def test_extract_paper_requires_an_id_on_each_withheld_section():
    # Without an id, find(id=None) would remove the wrong element and leave the section visible.
    with pytest.raises(RunError, match="abstract section has no HTML id"):
        extract_paper(latexml_page(abstract_id=None), {"paper_id": "p1"})


def test_extract_paper_withholds_each_section_from_its_context(monkeypatch):
    # Count words instead of tokens, so the test needs no downloaded tokenizer.
    monkeypatch.setattr(data, "largest_token_count", lambda text: len(text.split()))
    examples = extract_paper(latexml_page(abstract_id="abs"), {"paper_id": "p1"})
    assert [e["section_type"] for e in examples] == list(SECTIONS)
    for e in examples:
        assert e["example_id"] == "p1_" + e["section_type"]
        assert e["target_words"] == 80
        assert e["reference"] not in e["context"]
        assert f"[Missing {e['section_type'].replace('_', ' ')} section]" in e["context"]
    abstract, introduction = examples[0], examples[1]
    assert abstract["reference"].startswith("abs0 abs1")
    assert abstract["reference"] in introduction["context"]


def test_extract_paper_accepts_a_paper_without_related_work(monkeypatch):
    monkeypatch.setattr(data, "largest_token_count", lambda text: len(text.split()))
    page = latexml_page("abs", titles=("Introduction", "Summary and Outlook"))
    examples = extract_paper(page, {"paper_id": "p1"})
    assert [e["section_type"] for e in examples] == ["abstract", "introduction", "conclusion"]


def test_extract_paper_needs_an_introduction(monkeypatch):
    monkeypatch.setattr(data, "largest_token_count", lambda text: len(text.split()))
    with pytest.raises(RunError, match="introduction"):
        extract_paper(latexml_page("abs", titles=("Methods", "Conclusions")), {"paper_id": "p1"})
    examples = extract_paper(latexml_page("abs", titles=("Introduction", "Methods")), {"paper_id": "p1"})
    assert [e["section_type"] for e in examples] == ["abstract", "introduction"]


def test_extract_paper_reads_roman_numbered_headings(monkeypatch):
    monkeypatch.setattr(data, "largest_token_count", lambda text: len(text.split()))
    page = latexml_page("abs", titles=("INTRODUCTION", "Conclusion"), numbers="IV.")
    examples = extract_paper(page, {"paper_id": "p1"})
    assert [e["section_type"] for e in examples] == ["abstract", "introduction", "conclusion"]


def test_load_examples_accepts_missing_optional_sections_only(tmp_path):
    def build(sections):
        examples = [dummy_example("paper1", section, "train") for section in sections]
        path = tmp_path / "examples.jsonl"
        path.write_text("".join(canonical(e) + "\n" for e in examples))
        splits = tmp_path / "splits.json"
        write_json(splits, {"papers": {"train": ["paper1"]}})
        return path, splits

    assert len(load_examples(*build(["abstract", "introduction", "conclusion"]))) == 3
    with pytest.raises(RunError, match="paper1 has sections"):
        load_examples(*build(["abstract", "conclusion", "related_work"]))
    assert len(load_examples(*build(["abstract", "introduction"]))) == 2


def test_assign_splits_fills_each_split_and_mixes_neighbours():
    papers = list(range(sum(SPLIT_SIZES.values())))
    splits = assign_splits(papers)
    assert {split: splits.count(split) for split in SPLIT_SIZES} == SPLIT_SIZES
    # Any 7 papers in a row, a typical field's quota, hold more than one split.
    assert all(len(set(splits[i : i + 7])) > 1 for i in range(len(splits) - 6))
    with pytest.raises(RunError, match="cannot fill"):
        assign_splits(papers[:-1])


def test_extract_paper_leaves_figures_tables_and_acknowledgements_out_of_the_reference(monkeypatch):
    monkeypatch.setattr(data, "largest_token_count", lambda text: len(text.split()))
    page = latexml_page("abs").replace(
        "</section><section",
        '<figure class="ltx_figure"><figcaption>Figure 1: caption words</figcaption></figure>'
        '<div class="ltx_acknowledgements">We thank the funders</div></section><section',
        1,
    )
    introduction = extract_paper(page, {"paper_id": "p1"})[1]
    assert "caption words" not in introduction["reference"]
    assert "funders" not in introduction["reference"]
    assert introduction["target_words"] == 80


def test_number_citations_uses_one_style_for_citations_and_reference_labels():
    page = BeautifulSoup(
        '<div><p>See <cite class="ltx_cite">(<a href="#b2">Lee 2001</a>; <a href="#b1">Kim 2002</a>, e.g.,)'
        '</cite> and <cite class="ltx_cite"><a href="#nowhere">X</a></cite>.</p>'
        '<ul><li class="ltx_bibitem" id="b1"><span class="ltx_tag_bibitem">Kim (2002)</span> text</li>'
        '<li class="ltx_bibitem" id="b2"><span class="ltx_tag_bibitem">(2)</span> text</li></ul></div>',
        "html.parser",
    )
    number_citations(page)
    text = page.get_text(" ", strip=True)
    assert text.startswith("See [1, 2] and X")
    assert "[1] text" in text and "[2] text" in text and "Kim" not in text


def test_extract_paper_rejects_an_unmarked_acknowledgement(monkeypatch):
    monkeypatch.setattr(data, "largest_token_count", lambda text: len(text.split()))
    page = latexml_page("abs").replace("s3w79", "s3w79 Acknowledgements We thank the funders")
    with pytest.raises(RunError, match="conclusion contains acknowledgements"):
        extract_paper(page, {"paper_id": "p1"})


def test_html_text_joins_inline_elements_and_separates_blocks():
    node = BeautifulSoup(
        "<div><p><b>T</b>HE use of MoS<sub>2</sub> and <i>this</i>, (see <a>Fig. 1</a>).</p>"
        "<p>Next paragraph.</p><ul><li>one</li><li>two</li></ul></div>",
        "html.parser",
    )
    assert html_text(node) == "THE use of MoS2 and this, (see Fig. 1). Next paragraph. one two"
