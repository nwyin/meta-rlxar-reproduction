"""Checks that a run manifest matches configs/experiments.yaml and configs/models.yaml."""

import json

import pytest

from xar.audit import check_manifest_matches_design
from xar.openrouter import role_config
from xar.util import ROLES, RunError, digest, prompt


@pytest.fixture
def manifest(design):
    """A manifest that matches the design."""
    manifest = {
        "roles": {r: role_config(r) for r in ROLES},
        "arguments": {k: design[k] for k in ("iterations", "max_meta_prompt_words", "failure_examples")},
        "extra": {"initial_meta_prompt_hash": digest(prompt("rubric_initial"))},
    }
    check_manifest_matches_design(manifest, design)
    return manifest


def test_manifest_check_rejects_changed_iterations_or_role_settings(manifest, design):
    changed = json.loads(json.dumps(manifest))
    changed["arguments"]["iterations"] = 6
    with pytest.raises(RunError, match="iterations=6"):
        check_manifest_matches_design(changed, design)
    changed = json.loads(json.dumps(manifest))
    changed["roles"]["rubric"]["reasoning"] = {"enabled": False}
    with pytest.raises(RunError, match="rubric settings differ from configs/models.yaml in reasoning"):
        check_manifest_matches_design(changed, design)


def test_manifest_check_rejects_every_role_substitution(manifest, design):
    for role in ROLES:
        # Swap in the other model: Muse for the optimizer, Kimi everywhere else.
        other = "meta/muse-spark-1.1" if role == "optimizer" else "moonshotai/kimi-k2.6"
        changed = json.loads(json.dumps(manifest))
        changed["roles"][role] = role_config(role, other)
        with pytest.raises(RunError, match=f"{role} settings differ from configs/models.yaml in .*model"):
            check_manifest_matches_design(changed, design)
