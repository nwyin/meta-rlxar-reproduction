"""Re-check a saved run against its raw responses, frozen settings and preregistered design."""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from xar.data import load_examples, run_examples, task_data
from xar.openrouter import FORMAT_REPAIR, json_schema_format, role_config, routing_fields
from xar.pipeline import (
    GRADE_SCHEMA,
    RUBRIC_SCHEMA,
    checkpoint_row,
    select_checkpoint,
    validate_grade,
    validate_rubric,
)
from xar.stats import summarize
from xar.util import ROLES, RunError, canonical, digest, file_hash, prompt, read_json


def audit_request_contract(payload, cfg, schema, prompt_file):
    """Check the actual sent request against the frozen role, scoring wrapper, and schema."""
    if any(payload.get(key) != value for key, value in routing_fields(cfg).items()):
        raise RunError("Saved request differs from frozen role/routing/decoding contract")
    if payload.get("response_format") != json_schema_format(schema):
        raise RunError("Saved request differs from fixed output schema")
    messages = payload.get("messages", [])
    if len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]:
        raise RunError("Saved request includes conversation history or unexpected message roles")
    system = messages[0]["content"]
    base = system.split(FORMAT_REPAIR, 1)[0]
    if digest(base) != prompt_file:
        raise RunError("Saved request changed the frozen grading/rubric wrapper")


def audit_saved_output(record, schema, cfg=None, endpoint=None, prompt_file=None):
    if record["status"] != "valid":
        raise RunError("Missing structured output")
    last = record["attempts"][-1]
    if last["status"] != "valid":
        raise RunError("Artifact marked valid with invalid final attempt")
    response = last["response"]
    receipt = read_json(response["raw_response"])
    if receipt["status"] != "success":
        raise RunError("Artifact lacks successful raw response")
    choice = receipt["response"]["choices"][0]
    if choice["finish_reason"] != "stop":
        raise RunError("Artifact uses incomplete raw response")
    parsed = json.loads(choice["message"]["content"])
    jsonschema.validate(parsed, schema)
    if canonical(parsed) != canonical(record["value"]):
        raise RunError("Derived artifact differs from raw response")
    request = read_json(Path(response["raw_response"]).parent / "request.json")
    if cfg is not None:
        audit_request_contract(request["payload"], cfg, schema, prompt_file)
        raw = receipt["response"]
        if raw.get("model") not in {cfg["model"], endpoint["canonical_slug"]}:
            raise RunError("Raw response used a different model from the frozen role")
        provider = raw.get("provider")
        if provider and provider not in {endpoint["endpoint"]["provider_name"], cfg["provider"]}:
            raise RunError("Raw response used a different provider from the frozen role")
    return request["payload"]


def audit_xar_run(path):
    """Rebuild means from raw-linked criterion grades and prove full-scope checkpoint coverage."""
    path = Path(path)
    manifest, freeze = read_json(path / "manifest.json"), read_json(path / "freeze.json")
    if manifest["experiment"] != "xar":
        raise RunError("Expected XAR run")
    run_split = manifest["arguments"]["split"]
    pilot = run_split == "pilot"
    examples = run_examples(load_examples(manifest["dataset"], manifest["splits"]), run_split)
    counts = {s: sum(e["split"] == s for e in examples) for s in ("train", "validation")}
    if not pilot and counts != {"train": 32, "validation": 20}:
        raise RunError("Primary dataset must have 32 train and 20 validation examples")
    if (
        file_hash(manifest["dataset"]) != manifest["dataset_hash"]
        or file_hash(manifest["splits"]) != manifest["splits_hash"]
    ):
        raise RunError("Run data changed")
    candidates = read_json(path / "generations/candidates.json")["candidates"]
    all_rows, table = {}, []
    iterations = manifest["arguments"]["iterations"]
    if not pilot and iterations != 7:
        raise RunError("Primary trajectory lacks seven updates")
    for iteration in range(iterations + 1):
        checkpoint = (path / f"prompts/iter_{iteration:02d}.md").read_text()
        if digest(checkpoint) != freeze["prompt_hashes"][iteration]:
            raise RunError("Frozen checkpoint differs")
        if iteration:
            feedback = read_json(path / f"feedback/iter_{iteration:02d}/training.json")
            training_ids = {e["example_id"] for e in examples if e["split"] == "train"}
            if set(feedback["example_ids_used_for_aggregate"]) != training_ids:
                raise RunError("Optimizer aggregate contains non-training examples")
            if any(f["example_id"] not in training_ids for f in feedback["failures"]):
                raise RunError("Optimizer failures contain non-training examples")
        summaries = {}
        for split in ("train", "validation"):
            rows = []
            for e in [e for e in examples if e["split"] == split]:
                eid = e["example_id"]
                candidate = candidates[eid]
                if candidate["context_hash"] != e["context_hash"] or candidate["text_hash"] != digest(
                    candidate["text"]
                ):
                    raise RunError("Writer candidate integrity failed")
                rubric = read_json(path / f"rubrics/main/{iteration}/{split}/{eid}/rubric.json")
                if rubric["generator_configuration"] != manifest["roles"]["rubric"]:
                    raise RunError("Rubric artifact configuration differs from the frozen generator")
                rubric_payload = audit_saved_output(
                    rubric,
                    RUBRIC_SCHEMA,
                    manifest["roles"]["rubric"],
                    manifest["endpoints"]["rubric"],
                    manifest["software_hashes"]["prompts/rubric_wrapper.md"],
                )
                if split == "validation":
                    if not freeze.get("frozen_at"):
                        raise RunError("Freeze lacks timestamp evidence for held-out evaluation ordering")
                    for record in rubric["attempts"]:
                        receipt = read_json(record["response"]["raw_response"])
                        if not receipt.get("sent_at") or receipt["sent_at"] < freeze["frozen_at"]:
                            raise RunError("Validation rubric was dispatched before trajectory freeze")
                validate_rubric(rubric["value"])
                if json.loads(rubric_payload["messages"][1]["content"]) != {
                    **task_data(e),
                    "meta_prompt": checkpoint,
                }:
                    raise RunError("Rubric inputs contain an unexpected field or changed context")
                totals = {}
                for origin, text in (("human", e["reference"]), ("model", candidate["text"])):
                    grade = read_json(path / f"scores/main/{iteration}/{split}/{eid}/{origin}.json")
                    if grade["judge_configuration"] != manifest["roles"]["judge"]:
                        raise RunError("Grade artifact configuration differs from the frozen judge")
                    grade_payload = audit_saved_output(
                        grade,
                        GRADE_SCHEMA,
                        manifest["roles"]["judge"],
                        manifest["endpoints"]["judge"],
                        manifest["software_hashes"]["prompts/judge.md"],
                    )
                    if split == "validation":
                        for attempt in grade["attempts"]:
                            receipt = read_json(attempt["response"]["raw_response"])
                            if not receipt.get("sent_at") or receipt["sent_at"] < freeze["frozen_at"]:
                                raise RunError("Validation grade was dispatched before trajectory freeze")
                    expected = {**task_data(e), "rubric": rubric["value"], "candidate": text}
                    if json.loads(grade_payload["messages"][1]["content"]) != expected:
                        raise RunError("Anonymous grade payload violates input contract")
                    totals[origin] = validate_grade(
                        grade["value"], rubric["value"], text + " " + e["context"]
                    )
                    if grade["total"] != totals[origin] or grade["candidate_hash"] != digest(text):
                        raise RunError("Grade arithmetic or candidate hash differs")
                rows.append(
                    {
                        "example_id": eid,
                        "paper_id": e["paper_id"],
                        "section_type": e["section_type"],
                        "split": split,
                        "checkpoint": iteration,
                        **totals,
                        "gap": totals["human"] - totals["model"],
                        "length_compliant": candidate["length_compliant"],
                        "length_ratio": candidate["length_ratio"],
                        "contamination_flagged": candidate["contamination"]["flagged"],
                    }
                )
            all_rows[(iteration, split)] = rows
            summaries[split] = summarize(rows, manifest["arguments"]["seed"])
        table.append(
            checkpoint_row(iteration, summaries["train"], summaries["validation"], freeze["selected"])
        )
    selected = select_checkpoint([r["train_gap"] for r in table])
    if selected != freeze["selected"]:
        raise RunError("Selection rule differs")
    if freeze["training_gaps"] != [r["train_gap"] for r in table]:
        raise RunError("Training-selection ledger differs")
    return {
        "manifest": manifest,
        "freeze": freeze,
        "table": table,
        "rows": all_rows,
        "candidates": candidates,
    }


def validate_primary_manifest(manifest, design):
    if manifest["roles"]["judge"]["model"] != design["judge"]:
        raise RunError("Primary matrix must hold the preregistered main judge fixed")
    if design.get("scope") == "meta_blog_initial_empirical_investigation":
        for role in ROLES:
            if manifest["roles"][role]["model"] != design[role]:
                raise RunError(f"Blog reproduction requires the declared {role} model")
    for field in ("iterations", "max_meta_prompt_words", "failure_examples"):
        if manifest["arguments"].get(field) != design[field]:
            raise RunError(f"Primary matrix differs from preregistered {field}")
    if manifest["extra"].get("initial_meta_prompt_hash") != digest(prompt("rubric_initial")):
        raise RunError("Primary matrix differs from the frozen neutral starting prompt")
    for role, cfg in manifest["roles"].items():
        if cfg != role_config(role, cfg["model"]):
            raise RunError(f"Primary {role} provider/decoding differs from the frozen configuration")
