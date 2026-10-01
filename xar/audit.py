"""Re-check a saved run against its raw API responses, its manifest and configs/experiments.yaml."""

from __future__ import annotations

import json
import math
from pathlib import Path

import jsonschema

from xar.data import contamination, load_examples, run_examples, task_data
from xar.openrouter import (
    FORMAT_REPAIR,
    json_schema_format,
    ledger_key,
    max_request_cost,
    role_config,
    routing_fields,
)
from xar.pipeline import (
    GRADE_SCHEMA,
    PROPOSAL_ATTEMPTS,
    PROPOSAL_SCHEMA,
    RUBRIC_SCHEMA,
    WRITER_ATTEMPTS,
    audit_proposal,
    build_feedback,
    checkpoint_row,
    length_revision_note,
    length_window,
    select_checkpoint,
    validate_grade,
    validate_rubric,
)
from xar.stats import summarize
from xar.util import ROLES, RunError, canonical, digest, file_hash, prompt, read_json, words


def check_request_settings(payload, cfg, schema, prompt_hash, where="saved request"):
    """Check that a saved request payload used the role's settings, schema and system prompt.

    prompt_hash is the system prompt file's hash from the manifest. schema=None means a plain-text
    request, which must have no response_format and is never retried with a format repair.
    """
    differences = [
        f"{key} is {payload.get(key)!r}, expected {value!r}"
        for key, value in routing_fields(cfg).items()
        if payload.get(key) != value
    ]
    if differences:
        raise RunError(
            f"{where}: request settings differ from configs/models.yaml for {cfg['model']}: "
            + "; ".join(differences)
        )
    if payload.get("response_format") != json_schema_format(schema):
        expected = "no response_format" if schema is None else "the role's JSON schema"
        raise RunError(f"{where}: response_format differs from {expected}")
    messages = payload.get("messages", [])
    roles = [message.get("role") for message in messages]
    if roles != ["system", "user"]:
        raise RunError(f"{where}: expected one system and one user message, got {roles}")
    system = messages[0]["content"]
    if schema is not None:
        system = system.split(FORMAT_REPAIR, 1)[0]
    if digest(system) != prompt_hash:
        raise RunError(f"{where}: system prompt does not match the prompt file hash in the manifest")


def audit_saved_output(record, schema, cfg=None, endpoint=None, prompt_hash=None, where="saved output"):
    """Check a saved structured-output record against the raw API response it came from.

    With cfg, also check the request with check_request_settings and that the response came from
    the role's model and provider. Returns the request payload.
    """
    if record["status"] != "valid":
        raise RunError(f"{where}: structured output is {record['status']}, not valid")
    last = record["attempts"][-1]
    if last["status"] != "valid":
        raise RunError(f"{where}: marked valid, but its last attempt is {last['status']}")
    attempt_path = last["response"]["raw_response"]
    sent = read_json(attempt_path)
    if sent["status"] != "success":
        raise RunError(f"{attempt_path}: API call status is {sent['status']}, not success")
    response = sent["response"]
    choice = response["choices"][0]
    if choice["finish_reason"] != "stop":
        raise RunError(f"{attempt_path}: response finished with {choice['finish_reason']!r}, not 'stop'")
    parsed = json.loads(choice["message"]["content"])
    jsonschema.validate(parsed, schema)
    if canonical(parsed) != canonical(record["value"]):
        raise RunError(f"{where}: saved value differs from the raw response in {attempt_path}")
    request_path = Path(attempt_path).parent / "request.json"
    payload = read_json(request_path)["payload"]
    if cfg is not None:
        check_request_settings(payload, cfg, schema, prompt_hash, where=request_path)
        model = response.get("model")
        if model not in {cfg["model"], endpoint["canonical_slug"]}:
            raise RunError(f"{attempt_path}: answered by model {model}; expected {cfg['model']}")
        provider = response.get("provider")
        if provider and provider not in {endpoint["endpoint"]["provider_name"], cfg["provider"]}:
            raise RunError(f"{attempt_path}: answered by provider {provider}; expected {cfg['provider']}")
    return payload


def _check_sent_after_freeze(record, frozen_at, where):
    """Validation requests must be sent after freeze.json, so they cannot affect which checkpoint is selected."""
    for attempt in record["attempts"]:
        attempt_path = attempt["response"]["raw_response"]
        sent_at = read_json(attempt_path).get("sent_at")
        if not sent_at:
            raise RunError(f"{where}: {attempt_path} has no sent_at time")
        if sent_at < frozen_at:
            raise RunError(f"{where}: validation request sent at {sent_at}, before freeze.json ({frozen_at})")


def _check_feedback(run, iteration, training_ids):
    """The optimizer's feedback for an iteration may only use training examples."""
    path = run / f"feedback/iter_{iteration:02d}/training.json"
    feedback = read_json(path)
    used = set(feedback["example_ids_used_for_aggregate"])
    if used != training_ids:
        raise RunError(
            f"{path}: feedback must aggregate exactly the training examples; "
            f"extra {sorted(used - training_ids)}, missing {sorted(training_ids - used)}"
        )
    failure_ids = [failure["example_id"] for failure in feedback["failures"]]
    leaked = [eid for eid in failure_ids if eid not in training_ids]
    if leaked:
        raise RunError(f"{path}: failure examples {leaked} are not training examples")


def _audit_section(run, manifest, frozen_at, example, candidate, iteration, meta_prompt):
    """Check one section's rubric and its two grades at one checkpoint, and return its score row."""
    eid, split = example["example_id"], example["split"]
    roles, endpoints, hashes = manifest["roles"], manifest["endpoints"], manifest["software_hashes"]

    rubric_path = run / f"rubrics/main/{iteration}/{split}/{eid}/rubric.json"
    rubric = read_json(rubric_path)
    if rubric["generator_configuration"] != roles["rubric"]:
        raise RunError(f"{rubric_path}: generator_configuration differs from the manifest's rubric role")
    rubric_payload = audit_saved_output(
        rubric,
        RUBRIC_SCHEMA,
        roles["rubric"],
        endpoints["rubric"],
        hashes["prompts/rubric_wrapper.md"],
        where=rubric_path,
    )
    if split == "validation":
        _check_sent_after_freeze(rubric, frozen_at, rubric_path)
    validate_rubric(rubric["value"])
    expected_input = {**task_data(example), "meta_prompt": meta_prompt}
    if json.loads(rubric_payload["messages"][1]["content"]) != expected_input:
        raise RunError(
            f"{rubric_path}: request input is not the task data plus the checkpoint {iteration} prompt"
        )

    totals = {}
    for origin, text in (("human", example["reference"]), ("model", candidate["text"])):
        grade_path = run / f"scores/main/{iteration}/{split}/{eid}/{origin}.json"
        grade = read_json(grade_path)
        if grade["judge_configuration"] != roles["judge"]:
            raise RunError(f"{grade_path}: judge_configuration differs from the manifest's judge role")
        grade_payload = audit_saved_output(
            grade,
            GRADE_SCHEMA,
            roles["judge"],
            endpoints["judge"],
            hashes["prompts/judge.md"],
            where=grade_path,
        )
        if split == "validation":
            _check_sent_after_freeze(grade, frozen_at, grade_path)
        expected_input = {**task_data(example), "rubric": rubric["value"], "candidate": text}
        if json.loads(grade_payload["messages"][1]["content"]) != expected_input:
            raise RunError(f"{grade_path}: request input is not the task data, rubric and {origin} text")
        totals[origin] = validate_grade(grade["value"], rubric["value"], text + " " + example["context"])
        if grade["total"] != totals[origin]:
            raise RunError(
                f"{grade_path}: grade arithmetic mismatch (saved {grade['total']}, recomputed {totals[origin]})"
            )
        if grade["candidate_hash"] != digest(text):
            raise RunError(f"{grade_path}: candidate_hash does not match the {origin} text")

    return {
        "example_id": eid,
        "paper_id": example["paper_id"],
        "section_type": example["section_type"],
        "split": split,
        "checkpoint": iteration,
        **totals,
        "gap": totals["human"] - totals["model"],
        "length_compliant": candidate["length_compliant"],
        "length_ratio": candidate["length_ratio"],
        "contamination_flagged": candidate["contamination"]["flagged"],
    }


def _check(ok, message):
    """Like assert, but not stripped by python -O."""
    if not ok:
        raise RunError(message)


def _check_writer(manifest, candidates, examples_by_id):
    """Re-derive each writer candidate from its raw responses.

    The writer's input must be exactly the task data (never the author's section), plus the previous
    section and a length note on a retry. The text, hash, length flags, retry rules and copy check
    must all follow from the saved responses.
    """
    hashes, writer = manifest["software_hashes"], manifest["roles"]["writer"]
    _check(set(candidates) == set(examples_by_id), "Writer candidates do not match the run's examples")
    for eid, candidate in candidates.items():
        example, attempts = examples_by_id[eid], candidate["attempts"]
        _check(1 <= len(attempts) <= WRITER_ATTEMPTS, f"{eid}: {len(attempts)} writer attempts")
        _check(candidate["accepted_attempt"] == len(attempts) - 1, f"{eid}: accepted attempt is not the last")
        _check(
            not any(attempt["length_compliant"] for attempt in attempts[:-1]),
            f"{eid}: writer retried after a section that met the length target",
        )
        _check(
            candidate["length_compliant"] or len(attempts) == WRITER_ATTEMPTS,
            f"{eid}: writer stopped before meeting the length target or using every attempt",
        )
        low, high = length_window(example["target_words"])
        for index, attempt in enumerate(attempts):
            where = f"{eid} attempt {index}"
            raw_path = attempt["response"]["raw_response"]
            text = read_json(raw_path)["response"]["choices"][0]["message"]["content"].strip()
            _check(text == attempt["text"], f"{where}: saved text differs from the response")
            _check(digest(text) == attempt["text_hash"], f"{where}: text hash differs")
            _check(words(text) == attempt["words"], f"{where}: word count differs")
            _check(
                attempt["length_compliant"] == (low <= words(text) <= high),
                f"{where}: length_compliant flag is wrong",
            )
            payload = read_json(Path(raw_path).parent / "request.json")["payload"]
            check_request_settings(payload, writer, None, hashes["prompts/writer.md"], where=where)
            expected_input = task_data(example)
            if index:
                expected_input.update(
                    previous_section=attempts[index - 1]["text"],
                    length_revision=length_revision_note(example["target_words"]),
                )
            _check(
                canonical(expected_input) == payload["messages"][1]["content"],
                f"{where}: writer input differs from the task data",
            )
        _check(candidate["text"] == attempts[-1]["text"], f"{eid}: candidate text is not the last attempt")
        _check(candidate["complete"], f"{eid}: candidate is incomplete")
        _check(
            candidate["contamination"] == contamination(candidate["text"], example["reference"]),
            f"{eid}: contamination check differs",
        )


def _check_proposals(run, manifest, freeze, train, candidates):
    """Rebuild each update's optimizer feedback and proposal audit from the saved scores.

    The feedback must follow from training rows alone, each proposal must come from its raw
    response with the audit the current code gives it, and the accepted prompt must be the one the
    audit implies.
    """
    arguments = manifest["arguments"]
    max_words = arguments["max_meta_prompt_words"]
    initial = (run / "prompts/iter_00.md").read_text()
    for iteration in range(1, arguments["iterations"] + 1):
        where = f"iteration {iteration}"
        previous = (run / f"prompts/iter_{iteration - 1:02d}.md").read_text()
        rows = read_json(run / f"scores/main/{iteration - 1}/train_rows.json")
        directory = run / f"feedback/iter_{iteration:02d}"
        feedback, proposal = read_json(directory / "training.json"), read_json(directory / "proposal.json")
        expected_feedback = build_feedback(train, candidates, rows, previous, arguments["failure_examples"])
        _check(canonical(expected_feedback) == canonical(feedback), f"{where}: feedback differs")
        _check(proposal["parent_prompt_hash"] == digest(previous), f"{where}: parent prompt differs")
        _check(proposal["feedback_hash"] == digest(feedback), f"{where}: feedback hash differs")
        _check(proposal["update_consumed"], f"{where}: update not consumed")
        attempts = proposal["attempts"]
        _check(1 <= len(attempts) <= PROPOSAL_ATTEMPTS, f"{where}: {len(attempts)} attempts")
        expected_input = canonical({"feedback": feedback, "max_meta_prompt_words": max_words})
        for attempt in attempts:
            payload = audit_saved_output(
                attempt,
                PROPOSAL_SCHEMA,
                manifest["roles"]["optimizer"],
                manifest["endpoints"]["optimizer"],
                manifest["software_hashes"]["prompts/optimizer.md"],
                where=where,
            )
            _check(payload["messages"][1]["content"] == expected_input, f"{where}: optimizer input differs")
            _check(
                attempt["audit"] == audit_proposal(attempt["value"]["prompt"], train, initial, max_words),
                f"{where}: proposal audit differs",
            )
        final = attempts[-1]
        _check(proposal["accepted"] == final["audit"]["accepted"], f"{where}: accepted flag differs")
        expected_prompt = final["value"]["prompt"] if proposal["accepted"] else previous
        _check(
            proposal["prompt"] == expected_prompt, f"{where}: saved prompt is not the one the audit implies"
        )
        _check(
            proposal["prompt_hash"] == freeze["prompt_hashes"][iteration],
            f"{where}: prompt hash differs from freeze.json",
        )


def _check_costs(run, manifest):
    """Check every saved request against the budget ledger, and the run's cost file against both.

    A request's key is the hash of what was sent. Each send must have succeeded and been reserved
    and settled in order, at a cost within its reservation, and costs.json must equal the sum.
    """
    ledger = read_json(manifest["arguments"]["budget_ledger"])["entries"]
    total, requests = 0, 0
    for path in run.glob("requests/*/request.json"):
        request = read_json(path)
        role, payload = request["role"], request["payload"]
        hashed = {
            "payload": payload,
            "identity": request["identity"],
            "schema_version": manifest["schema_version"],
        }
        _check(
            request["key"] == path.parent.name == digest(hashed), f"{path}: key does not match its contents"
        )
        max_cost = max_request_cost(payload, manifest["endpoints"][role]["endpoint"])
        for attempt_path in path.parent.glob("attempt_*.json"):
            sent = read_json(attempt_path)
            _check(sent["status"] == "success", f"{attempt_path}: status is {sent['status']}")
            key = ledger_key(run, request["key"], attempt_path.stem.removeprefix("attempt_"))
            entry, cost = ledger[key], sent["response"]["usage"]["cost"]
            _check(entry["state"] == "complete", f"{key}: ledger entry is {entry['state']}")
            _check(
                math.isclose(entry["charge"], cost), f"{key}: ledger charge {entry['charge']} != cost {cost}"
            )
            _check(
                math.isclose(entry["upper"], max_cost), f"{key}: ledger reservation is not the maximum cost"
            )
            _check(cost <= max_cost, f"{key}: cost {cost} exceeds the maximum request cost {max_cost}")
            _check(
                entry["created_at"] <= sent["sent_at"] <= sent["timestamp"] <= entry["settled_at"],
                f"{key}: reserve, send, receive and settle times are out of order",
            )
            total += cost
            requests += 1
    costs = read_json(run / "costs.json")
    _check(math.isclose(costs["actual_complete_usd"], total), "costs.json total differs from the responses")
    _check(costs["unresolved"] == 0, f"costs.json has {costs['unresolved']} unresolved requests")
    _check(costs["requests"] == requests, f"costs.json counts {costs['requests']} requests; found {requests}")


def audit_xar_run(path):
    """Re-check a saved XAR run and rebuild its checkpoint table from the raw API responses.

    Every rubric and grade must match its raw response, its request must use the manifest's role
    settings and prompts, validation requests must postdate freeze.json, and the recomputed grade
    totals, training gaps and selected checkpoint must equal the saved ones. The writer candidates,
    optimizer feedback and proposals are re-derived, and request keys, costs and the budget ledger
    must agree. Raises RunError if not.
    """
    run = Path(path)
    manifest = read_json(run / "manifest.json")
    freeze = read_json(run / "freeze.json")
    if manifest["experiment"] != "xar":
        raise RunError(f"{run} is a {manifest['experiment']!r} run, not an XAR run")
    for name in ("dataset", "splits"):
        if file_hash(manifest[name]) != manifest[f"{name}_hash"]:
            raise RunError(f"{manifest[name]} changed after the run started")

    arguments = manifest["arguments"]
    iterations = arguments["iterations"]
    examples = run_examples(load_examples(manifest["dataset"], manifest["splits"]), arguments["split"])
    by_split = {split: [e for e in examples if e["split"] == split] for split in ("train", "validation")}
    if arguments["split"] != "pilot":
        empty = [split for split, group in by_split.items() if not group]
        if empty:
            raise RunError(f"Research run has no {' or '.join(empty)} sections")

    candidates = read_json(run / "generations/candidates.json")["candidates"]
    for example in examples:
        eid = example["example_id"]
        candidate = candidates[eid]
        if candidate["context_hash"] != example["context_hash"]:
            raise RunError(f"{eid}: writer candidate was generated from a different context")
        if candidate["text_hash"] != digest(candidate["text"]):
            raise RunError(f"{eid}: writer candidate text does not match its text_hash")

    _check_writer(manifest, candidates, {e["example_id"]: e for e in examples})
    frozen_at = freeze.get("frozen_at")
    if not frozen_at:
        raise RunError(f"{run / 'freeze.json'} has no frozen_at time")
    training_ids = {e["example_id"] for e in by_split["train"]}

    all_rows, table = {}, []
    for iteration in range(iterations + 1):
        prompt_path = run / f"prompts/iter_{iteration:02d}.md"
        meta_prompt = prompt_path.read_text()
        if digest(meta_prompt) != freeze["prompt_hashes"][iteration]:
            raise RunError(f"{prompt_path} differs from its hash in freeze.json")
        if iteration:
            _check_feedback(run, iteration, training_ids)
        summaries = {}
        for split, split_examples in by_split.items():
            rows = [
                _audit_section(
                    run, manifest, frozen_at, e, candidates[e["example_id"]], iteration, meta_prompt
                )
                for e in split_examples
            ]
            all_rows[(iteration, split)] = rows
            summaries[split] = summarize(rows, arguments["seed"])
        table.append(
            checkpoint_row(iteration, summaries["train"], summaries["validation"], freeze["selected"])
        )

    gaps = [row["train_gap"] for row in table]
    if freeze["training_gaps"] != gaps:
        raise RunError(
            f"freeze.json training_gaps {freeze['training_gaps']} differ from the recomputed {gaps}"
        )
    selected = select_checkpoint(gaps)
    if selected != freeze["selected"]:
        raise RunError(
            f"freeze.json selected checkpoint {freeze['selected']}, "
            f"but the highest training gap is at checkpoint {selected}"
        )
    _check_proposals(run, manifest, freeze, by_split["train"], candidates)
    _check_costs(run, manifest)
    return {
        "manifest": manifest,
        "freeze": freeze,
        "table": table,
        "rows": all_rows,
        "candidates": candidates,
    }


def check_manifest_matches_design(manifest, design):
    """Check that a run used the role settings in configs/models.yaml, and the iterations, word
    limit, failure count and starting prompt in configs/experiments.yaml.

    design comes from util.load_design; for a pilot the caller puts pilot_iterations in place of
    iterations.
    """
    for role in ROLES:
        actual = manifest["roles"][role]
        expected = role_config(role)
        changed = sorted(
            key for key in expected.keys() | actual.keys() if actual.get(key) != expected.get(key)
        )
        if changed:
            details = "; ".join(
                f"{key} (run used {actual.get(key)!r}, config says {expected.get(key)!r})" for key in changed
            )
            raise RunError(f"{role} settings differ from configs/models.yaml in {details}")
    for field in ("iterations", "max_meta_prompt_words", "failure_examples"):
        value = manifest["arguments"].get(field)
        if value != design[field]:
            raise RunError(f"Run used {field}={value}; configs/experiments.yaml expects {design[field]}")
    if manifest["extra"].get("initial_meta_prompt_hash") != digest(prompt("rubric_initial")):
        raise RunError("Run started from a different prompts/rubric_initial.md than the current file")
