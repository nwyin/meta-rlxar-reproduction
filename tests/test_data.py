"""Dataset loading and the model-input allowlist."""

import pytest
from conftest import dummy_example

from xar.data import load_examples, task_data
from xar.openrouter import model_policy
from xar.util import RunError, canonical, write_json


def test_input_allowlist_and_model_policy():
    e = dummy_example()
    assert set(task_data(e)) == {"visible_paper", "section_type", "target_words"}
    assert e["reference"] not in canonical(task_data(e))
    for model in ("anthropic/claude-test", "google/gemini-test", "openrouter/auto", "qwen/qwen3.5-9b:free"):
        with pytest.raises(RunError):
            model_policy(model)


def test_dataset_hash_and_cross_paper_split_guard(tmp_path):
    example = dummy_example()
    path = tmp_path / "examples.jsonl"
    path.write_text(canonical(example) + "\n")
    splits = tmp_path / "splits.json"
    write_json(splits, {"papers": {"train": ["paper1"], "validation": ["paper1"]}})
    with pytest.raises(RunError, match="overlap"):
        load_examples(path, splits)
