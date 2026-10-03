"""Blind sampling, durable judgments, and delayed source disclosure in the local viewer."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "blind_review", Path(__file__).resolve().parents[1] / "tools/data-viewer/review.py"
)
review = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(review)


@pytest.fixture
def corpus(tmp_path):
    dataset = tmp_path / "data/examples.jsonl"
    dataset.parent.mkdir()
    run = tmp_path / "runs" / review.RUN
    (run / "generations").mkdir(parents=True)
    examples = []
    training = []
    for kind in ("abstract", "introduction", "conclusion", "related_work"):
        for index in range(3):
            paper = f"{kind}-{index}"
            training.append(paper)
            example = {
                "example_id": paper, "paper_id": paper, "section_type": kind, "split": "train",
                "context_hash": "context-hash", "context": "The rest of the paper.",
                "reference": f"Reference {paper}", "target_words": 100,
                "provenance": {"title": f"Paper {paper}", "source_url": f"https://example.org/{paper}"},
            }
            examples.append(example)
            (run / "generations" / f"{paper}.json").write_text(json.dumps({
                "context_hash": "context-hash", "complete": True, "text": f"Draft {paper}",
            }))
    examples.append({**examples[0], "paper_id": "holdout", "example_id": "holdout", "split": "validation"})
    dataset.write_text("\n".join(json.dumps(e) for e in examples))
    splits = dataset.with_name("splits.json")
    splits.write_text(json.dumps({"papers": {"train": training, "validation": ["holdout"]}}))
    (run / "manifest.json").write_text(json.dumps({
        "dataset_hash": review.file_hash(dataset), "splits_hash": review.file_hash(splits),
    }))
    return tmp_path


def answer(pair_id, **overrides):
    return {"pair_id": pair_id, "choice": "A", "confidence": "medium", "notes": "Clearer.", **overrides}


def test_fixed_training_sample_is_balanced_and_blind(corpus):
    view = review.get_session(corpus)
    saved = review.load(corpus)
    assert len(saved["pairs"]) == 4
    assert len({p["section_type"] for p in saved["pairs"]}) == 4
    assert len({p["paper_id"] for p in saved["pairs"]}) == 4
    assert all(p["paper_id"] != "holdout" for p in saved["pairs"])
    assert sorted(p["author_side"] for p in saved["pairs"]) == ["A", "A", "B", "B"]
    for pair in view["pairs"]:
        assert set(pair) == {"id", "section_type", "field", "target_words", "context", "A", "B"}
    assert not view["complete"]
    assert review.get_session(corpus) == view
    assert review.create(corpus)["pairs"] == saved["pairs"]


def test_judgments_persist_lock_and_reveal_only_after_all_four(corpus):
    view = review.get_session(corpus)
    for index, pair in enumerate(view["pairs"]):
        result = review.record(corpus, answer(pair["id"]))
        assert result["complete"] == (index == 3)
        assert ("author_side" in result["pairs"][0]) == (index == 3)
        assert review.record(corpus, answer(pair["id"])) == result
    assert review.get_session(corpus) == result
    with pytest.raises(ValueError, match="locked"):
        review.record(corpus, answer("pair-1", choice="B"))
    assert review.get_session(corpus)["answers"]["pair-1"]["choice"] == "A"


def test_extraction_review_cannot_reveal_early_or_change_preference(corpus):
    payload = {"pair_id": "pair-1", "status": "problem", "notes": "Extra caption."}
    with pytest.raises(ValueError, match="Finish"):
        review.record(corpus, payload, extraction=True)
    for pair in review.get_session(corpus)["pairs"]:
        review.record(corpus, answer(pair["id"], choice="unsure", familiar=True))
    result = review.record(corpus, payload, extraction=True)
    assert result["extraction"]["pair-1"]["notes"] == "Extra caption."
    assert result["answers"]["pair-1"]["choice"] == "unsure"
    assert result["answers"]["pair-1"]["familiar"] is True


@pytest.mark.parametrize("payload", [
    [], answer("missing"), answer("pair-1", choice="human"), answer("pair-1", confidence=None),
    answer("pair-1", notes="x" * 10001), answer("pair-1", familiar="false"),
])
def test_invalid_answers_do_not_change_saved_judgments(corpus, payload):
    with pytest.raises((ValueError, TypeError)):
        review.record(corpus, payload)
    assert not review.get_session(corpus)["answers"]


def test_changed_dataset_stops_sampling(corpus):
    path = corpus / "data/examples.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="does not match"):
        review.get_session(corpus)
    assert not (corpus / "reports" / f"{review.SESSION}.json").exists()
