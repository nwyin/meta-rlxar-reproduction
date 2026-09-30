"""Checks that a run manifest matches the preregistered design."""

import json

import pytest
import yaml

from xar.audit import validate_primary_manifest
from xar.openrouter import role_config
from xar.util import ROOT, RunError, digest, prompt


def test_primary_matrix_rejects_different_judge_and_protocol():
    design = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    manifest = {
        "roles": {r: role_config(r) for r in ("writer", "rubric", "optimizer", "judge")},
        "arguments": {k: design[k] for k in ("iterations", "max_meta_prompt_words", "failure_examples")},
        "extra": {"initial_meta_prompt_hash": digest(prompt("rubric_initial"))},
    }
    validate_primary_manifest(manifest, design)
    changed = json.loads(json.dumps(manifest))
    changed["roles"]["judge"] = role_config("judge", "moonshotai/kimi-k2.6")
    with pytest.raises(RunError, match="main judge fixed"):
        validate_primary_manifest(changed, design)
    changed = json.loads(json.dumps(manifest))
    changed["arguments"]["iterations"] = 6
    with pytest.raises(RunError, match="iterations"):
        validate_primary_manifest(changed, design)
    changed = json.loads(json.dumps(manifest))
    changed["roles"]["rubric"]["reasoning"] = {"enabled": False}
    with pytest.raises(RunError, match="decoding"):
        validate_primary_manifest(changed, design)


def test_blog_model_contract_rejects_every_role_substitution():
    design = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    manifest = {
        "roles": {r: role_config(r) for r in ("writer", "rubric", "optimizer", "judge")},
        "arguments": {k: design[k] for k in ("iterations", "max_meta_prompt_words", "failure_examples")},
        "extra": {"initial_meta_prompt_hash": digest(prompt("rubric_initial"))},
    }
    validate_primary_manifest(manifest, design)
    for role in ("writer", "rubric", "optimizer", "judge"):
        # Swap in the other model: Muse for the optimizer, Kimi everywhere else.
        other = "meta/muse-spark-1.1" if role == "optimizer" else "moonshotai/kimi-k2.6"
        changed = json.loads(json.dumps(manifest))
        changed["roles"][role] = role_config(role, other)
        with pytest.raises(RunError):
            validate_primary_manifest(changed, design)
