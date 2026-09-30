"""Dataset loading, paper extraction and the fields models see."""

import pytest
from conftest import dummy_example

from xar.data import SECTIONS, extract_paper, load_examples, run_examples, task_data
from xar.openrouter import model_policy
from xar.util import RunError, canonical, write_json


def test_task_data_sends_only_visible_fields():
    e = dummy_example()
    assert set(task_data(e)) == {"visible_paper", "section_type", "target_words"}
    assert e["reference"] not in canonical(task_data(e))


def test_model_policy_rejects_excluded_families_and_routers():
    for model in ("anthropic/claude-test", "google/gemini-test", "openrouter/auto", "qwen/qwen3.5-9b:free"):
        with pytest.raises(RunError):
            model_policy(model)


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
    with pytest.raises(RunError, match="at least 2 papers"):
        run_examples(examples[:4], "pilot")


def latexml_page(abstract_id):
    """A minimal LaTeXML page with the four target sections and a 10-item bibliography."""

    def body(name):
        return " ".join(f"{name}{i}" for i in range(80))

    abstract_attrs = f' id="{abstract_id}"' if abstract_id else ""
    sections = "".join(
        f'<section class="ltx_section" id="S{i}"><h2>{i} {title}</h2><p>{body(f"s{i}w")}</p></section>'
        for i, title in enumerate(("Introduction", "Related Work", "Conclusions"), start=1)
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
