"""The XAR stages: writer candidates, rubrics, grades, optimizer feedback and proposals, and run_xar."""

from __future__ import annotations

import math
import random
import re
import statistics
from pathlib import Path

import jsonschema

from xar.data import contamination, load_examples, run_examples, task_data
from xar.openrouter import role_config
from xar.runs import estimate, initialize_run, operational_summary
from xar.stats import summarize
from xar.util import (
    ROLES,
    SCHEMA_VERSION,
    InvalidOutput,
    RunError,
    bounded_map,
    digest,
    normalize,
    now,
    prompt,
    read_json,
    run_lock,
    words,
    write_json,
    write_table,
)


def object_schema(properties):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


TEXT = {"type": "string", "minLength": 1}
CRITERION_SCHEMA = object_schema({k: TEXT for k in ("id", "description", "low", "middle", "high")})
RUBRIC_SCHEMA = object_schema(
    {"criteria": {"type": "array", "minItems": 4, "maxItems": 8, "items": CRITERION_SCHEMA}}
)
GRADE_SCHEMA = object_schema(
    {
        "scores": {
            "type": "array",
            "minItems": 4,
            "maxItems": 8,
            "items": object_schema(
                {"id": TEXT, "score": {"type": "number", "minimum": 0, "maximum": 10}, "evidence": TEXT}
            ),
        }
    }
)
PROPOSAL_SCHEMA = object_schema({"prompt": TEXT, "rationale": TEXT})


def validate_rubric(rubric):
    jsonschema.validate(rubric, RUBRIC_SCHEMA)
    ids = [c["id"] for c in rubric["criteria"]]
    if len(ids) != len(set(ids)):
        raise InvalidOutput("Duplicate rubric criterion IDs")
    if words(" ".join(str(v) for c in rubric["criteria"] for v in c.values())) > 1000:
        raise InvalidOutput("Rubric exceeds 1000 words")


def validate_grade(grade, rubric, supplied_text):
    jsonschema.validate(grade, GRADE_SCHEMA)
    ids = [s["id"] for s in grade["scores"]]
    if len(ids) != len(set(ids)) or set(ids) != {c["id"] for c in rubric["criteria"]}:
        raise InvalidOutput("Grade criterion coverage mismatch")
    if any(not math.isfinite(s["score"]) or isinstance(s["score"], bool) for s in grade["scores"]):
        raise InvalidOutput("Grade score must be a finite number")
    for score in grade["scores"]:
        for quoted in re.findall(r'["“]([^"”]+)["”]', score["evidence"]):
            if normalize(quoted).casefold() not in normalize(supplied_text).casefold():
                raise InvalidOutput("Evidence quote does not occur in supplied text")
    return statistics.mean(s["score"] for s in grade["scores"])


def length_window(target_words):
    """The word counts, low to high, that count as meeting a section's length target."""
    return 0.85 * target_words, 1.15 * target_words


def length_revision_note(target_words):
    """The instruction sent with a writer retry after a section missed its length target."""
    low, high = length_window(target_words)
    return f"Revise only to fit {math.ceil(low)}–{math.floor(high)} words. Preserve claims."


def writer_candidates(api, examples, output, concurrency=1):
    config_hash = digest(
        {"writer": api.roles["writer"], "prompt": prompt("writer"), "schema_version": SCHEMA_VERSION}
    )
    stopped = api.dispatch_stopped

    def generate_one(e):
        path = Path(output) / "generations" / (e["example_id"] + ".json")
        if path.exists():
            record = read_json(path)
        else:
            attempts = []
            for attempt in range(3):
                data = task_data(e)
                if attempt:
                    data.update(
                        previous_section=attempts[-1]["text"],
                        length_revision=length_revision_note(e["target_words"]),
                    )
                if stopped.is_set():
                    raise RunError("Writer generation halted after another section failed")
                response = api.call(
                    "writer",
                    prompt("writer"),
                    data,
                    None,
                    {"writer": config_hash, "example": e["example_id"], "attempt": attempt},
                )
                text = response["content"].strip()
                count = words(text)
                low, high = length_window(e["target_words"])
                compliant = low <= count <= high
                attempts.append(
                    {
                        "attempt_index": attempt,
                        "text": text,
                        "text_hash": digest(text),
                        "words": count,
                        "length_compliant": compliant,
                        "response": response,
                    }
                )
                if compliant and response["finish_reason"] == "stop":
                    break
            accepted = attempts[-1]
            record = {
                "example_id": e["example_id"],
                "context_hash": e["context_hash"],
                "writer_configuration_hash": config_hash,
                "writer_configuration": api.roles["writer"],
                "attempts": attempts,
                "accepted_attempt": accepted["attempt_index"],
                "text": accepted["text"],
                "text_hash": accepted["text_hash"],
                "words": accepted["words"],
                "length_compliant": accepted["length_compliant"],
                "complete": accepted["response"]["finish_reason"] == "stop",
                "length_ratio": accepted["words"] / e["target_words"],
                "contamination": contamination(accepted["text"], e["reference"]),
            }
            write_json(path, record, write_once=True)
        if record["context_hash"] != e["context_hash"] or record["text_hash"] != digest(record["text"]):
            raise RunError("Cached writer context or content differs")
        if record["writer_configuration_hash"] != config_hash:
            raise RunError("Local cached writer settings differ")
        return record

    records = bounded_map(generate_one, examples, concurrency, stopped)
    # Completion order never changes the frozen dataset/candidate order.
    candidates = {record["example_id"]: record for record in records}
    artifact = {
        "writer_configuration_hash": config_hash,
        "writer_configuration": api.roles["writer"],
        "candidates": candidates,
    }
    write_json(Path(output) / "generations/candidates.json", artifact, write_once=True)
    return candidates


def generate_rubric(api, example, meta_prompt, identity, output):
    result = api.structured(
        "rubric",
        prompt("rubric_wrapper"),
        {**task_data(example), "meta_prompt": meta_prompt},
        RUBRIC_SCHEMA,
        identity,
        validate_rubric,
    )
    record = {
        "example_id": example["example_id"],
        "context_hash": example["context_hash"],
        "meta_prompt_hash": digest(meta_prompt),
        "generator_configuration": api.roles["rubric"],
        "identity": identity,
        **result,
    }
    record["rubric_hash"] = digest(result["value"]) if result["value"] else None
    write_json(output, record, write_once=True)
    return record


def grade_candidate(api, example, rubric_record, text, identity, output):
    if rubric_record["status"] != "valid":
        result = {"status": "missing", "value": None, "attempts": [], "reason": "invalid_rubric"}
    else:
        rubric = rubric_record["value"]
        result = api.structured(
            "judge",
            prompt("judge"),
            {**task_data(example), "rubric": rubric, "candidate": text},
            GRADE_SCHEMA,
            identity,
            lambda value: validate_grade(value, rubric, text + " " + example["context"]),
        )
    record = {
        "example_id": example["example_id"],
        "candidate_hash": digest(text),
        "rubric_hash": rubric_record["rubric_hash"],
        "judge_configuration": api.roles["judge"],
        "identity": identity,
        **result,
    }
    record["total"] = (
        statistics.mean(s["score"] for s in result["value"]["scores"]) if result["value"] else None
    )
    write_json(output, record, write_once=True)
    return record


def evaluate_checkpoint(api, examples, candidates, meta_prompt, checkpoint, output, concurrency):
    root = Path(output)

    def evaluate(e):
        eid = e["example_id"]
        # The "main/" prefix is part of saved paths and request identities.
        label = f"main/{checkpoint}/{e['split']}/{eid}"
        rubric = generate_rubric(
            api, e, meta_prompt, {"rubric": label}, root / "rubrics" / label / "rubric.json"
        )
        labels = ["human", "model"]
        random.Random(int(digest({"seed": api.seed, "label": label})[:16], 16)).shuffle(labels)
        grades = {}
        for slot, origin in enumerate(labels):
            text = e["reference"] if origin == "human" else candidates[eid]["text"]
            # Origin is in artifact paths only, never model inputs or seed identity.
            grades[origin] = grade_candidate(
                api,
                e,
                rubric,
                text,
                {"grade": label, "slot": slot},
                root / "scores" / label / (origin + ".json"),
            )
        human, model = grades["human"]["total"], grades["model"]["total"]
        return {
            "example_id": eid,
            "paper_id": e["paper_id"],
            "section_type": e["section_type"],
            "split": e["split"],
            "checkpoint": checkpoint,
            "human": human,
            "model": model,
            "gap": human - model if human is not None and model is not None else None,
            "length_ratio": candidates[eid]["length_ratio"],
            "length_compliant": candidates[eid]["length_compliant"],
            "contamination_flagged": candidates[eid]["contamination"]["flagged"],
            "rubric_path": str(root / "rubrics" / label / "rubric.json"),
            "grade_paths": {o: str(root / "scores" / label / (o + ".json")) for o in labels},
        }

    rows = bounded_map(evaluate, examples, concurrency, api.dispatch_stopped)
    write_json(root / "scores" / "main" / str(checkpoint) / f"{examples[0]['split']}_rows.json", rows)
    return rows


def audit_proposal(text, examples, initial_prompt, max_words):
    reasons, flags = [], []
    if words(text) > max_words:
        reasons.append("meta_prompt_word_bound")
    lower = text.casefold()
    patterns = {
        "origin_preference": r"(?:prefer|reward|favor|favour|boost)\s+(?:the\s+)?(?:human|expert|original)|"
        r"(?:penaliz|penalis|punish|downscore)\w*\s+(?:the\s+)?(?:model|ai|generated)",
        "origin_detection": r"(?:detect|guess|infer|identify|determine)\s+(?:the\s+)?(?:authorship|origin|provenance)|"
        r"(?:human.written|ai.generated|model.generated)\s+(?:tells|markers|signals)",
        "wrapper_override": r"(?:ignore|override|replace)\s+(?:the\s+)?(?:wrapper|system|grading|schema)|"
        r"(?:weighted\s+(?:mean|average)|unequal\s+weight|score\s+(?:from\s+)?0\s*(?:to|[-–])\s*100)",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, lower):
            reasons.append(name)
    for e in examples:
        metadata = e["provenance"]
        for identifier in [e["paper_id"], metadata["title"], *metadata.get("authors", [])]:
            if len(identifier) >= 5 and identifier.casefold() in lower:
                flags.append({"type": "training_identifier", "paper_id": e["paper_id"], "match": identifier})
        a = lower.split()
        for field in ("reference", "context"):
            b = e[field].casefold().split()
            spans = {tuple(b[i : i + 12]) for i in range(max(0, len(b) - 11))}
            for i in range(max(0, len(a) - 11)):
                span = " ".join(a[i : i + 12])
                if tuple(a[i : i + 12]) in spans and span not in initial_prompt.casefold():
                    flags.append({"type": "copied_training_span", "paper_id": e["paper_id"], "match": span})
                    break
    if flags:
        reasons.append("training_leakage")
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "flags": flags,
        "words": words(text),
        "policy": "fixed_static_scan_v1",
        "limitation": "Static scan cannot prove semantic absence of provenance heuristics",
    }


def propose_prompt(api, current, feedback, examples, initial, iteration, *, output, max_words):
    directory = Path(output) / "feedback" / f"iter_{iteration:02d}"
    write_json(directory / "training.json", feedback, write_once=True)
    data = {"feedback": feedback, "max_meta_prompt_words": max_words}
    attempts = []
    for attempt in range(2):
        instruction = prompt("optimizer")
        if attempt:
            rejected = attempts[-1]
            instruction += "\nBOUNDED REPAIR: Fix these proposal violations: " + ", ".join(
                rejected["audit"]["reasons"]
            )
            # Show the rejected proposal, or the raw reply if it was not valid JSON.
            previous = rejected["value"] or rejected["attempts"][-1]["response"]["content"]
            data = {**data, "previous_proposal": previous}
        result = api.structured(
            "optimizer",
            instruction,
            data,
            PROPOSAL_SCHEMA,
            {"proposal": iteration, "bounded_attempt": attempt},
            repair=False,
        )
        audit = (
            audit_proposal(result["value"]["prompt"], examples, initial, max_words)
            if result["value"]
            else {"accepted": False, "reasons": ["invalid_format"], "flags": []}
        )
        attempts.append({"attempt": attempt, "audit": audit, **result})
        if audit["accepted"]:
            break
    accepted = attempts[-1]["audit"]["accepted"]
    proposal = attempts[-1]["value"]["prompt"] if accepted else current
    record = {
        "iteration": iteration,
        "parent_prompt_hash": digest(current),
        "feedback_hash": digest(feedback),
        "optimizer_configuration": api.roles["optimizer"],
        "attempts": attempts,
        "accepted": accepted,
        "update_consumed": True,
        "prompt": proposal,
        "prompt_hash": digest(proposal),
    }
    write_json(directory / "proposal.json", record, write_once=True)
    return proposal


def build_feedback(examples, candidates, rows, current_prompt, failure_count):
    if any(e["split"] != "train" for e in examples) or any(r["split"] != "train" for r in rows):
        raise RunError("Validation/pilot examples cannot enter optimizer feedback")
    if any(r["gap"] is None for r in rows):
        raise RunError("Incomplete training feedback; cannot optimize or select a partial checkpoint")
    lookup = {e["example_id"]: e for e in examples}
    failures = sorted(rows, key=lambda r: (r["gap"], r["example_id"]))[:failure_count]
    selected = []
    for row in failures:
        e = lookup[row["example_id"]]
        selected.append(
            {
                "example_id": e["example_id"],
                "paper_id": e["paper_id"],
                "section_type": e["section_type"],
                "target_words": e["target_words"],
                "visible_paper": e["context"],
                "human_candidate": e["reference"],
                "model_candidate": candidates[e["example_id"]]["text"],
                "rubric": read_json(row["rubric_path"])["value"],
                "grades": {o: read_json(p)["value"] for o, p in row["grade_paths"].items()},
                "gap": row["gap"],
            }
        )
    return {
        "current_prompt": current_prompt,
        "summary": summarize(rows),
        "failures": selected,
        "example_ids_used_for_aggregate": sorted(lookup),
        "selection": "smallest_gap_then_example_id",
    }


def select_checkpoint(gaps):
    """The checkpoint with the largest training gap; the earliest one wins a tie."""
    return gaps.index(max(gaps))


def checkpoint_row(iteration, train, validation, selected):
    """One row of the checkpoint table, from the training and validation summaries."""
    return {
        "iteration": iteration,
        "train_human": train["human"],
        "train_model": train["model"],
        "train_gap": train["gap"],
        "val_human": validation["human"],
        "val_model": validation["model"],
        "val_gap": validation["gap"],
        "selected_by_train": iteration == selected,
    }


def run_xar(args):
    """Run one trajectory: writer candidates, training updates, freeze, then validation."""
    initial = prompt("rubric_initial")
    if words(initial) > args.max_meta_prompt_words:
        raise RunError("Initial prompt exceeds bound")
    examples = run_examples(load_examples(args.dataset, args.splits), args.split)
    train, validation = (
        [e for e in examples if e["split"] == "train"],
        [e for e in examples if e["split"] == "validation"],
    )
    roles = {role: role_config(role, getattr(args, f"{role}_model")) for role in ROLES}
    if args.dry_run:
        estimate(
            args,
            roles,
            examples,
            {
                "writer": len(examples),
                "rubric": len(examples) * (args.iterations + 1),
                "judge": 2 * len(examples) * (args.iterations + 1),
                "optimizer": args.iterations,
            },
        )
        return
    with run_lock(args.output_dir):
        out = Path(args.output_dir)
        api = initialize_run(args, roles, "xar", {"initial_meta_prompt_hash": digest(initial)})
        candidates = writer_candidates(api, examples, out, args.concurrency)
        checkpoints, training_rows = [initial], []
        for iteration in range(args.iterations + 1):
            if iteration:
                feedback = build_feedback(
                    train, candidates, training_rows[-1], checkpoints[-1], args.failure_examples
                )
                proposal = propose_prompt(
                    api,
                    checkpoints[-1],
                    feedback,
                    train,
                    initial,
                    iteration,
                    output=out,
                    max_words=args.max_meta_prompt_words,
                )
                checkpoints.append(proposal)
            path = out / "prompts" / f"iter_{iteration:02d}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_text() != checkpoints[-1]:
                raise RunError("Frozen checkpoint changed")
            path.write_text(checkpoints[-1])
            rows = evaluate_checkpoint(
                api, train, candidates, checkpoints[-1], iteration, out, args.concurrency
            )
            training_rows.append(rows)
            summary = summarize(rows, args.seed)
            print(
                f"Checkpoint {iteration}: train gap {summary['gap']}, coverage {summary['paired_coverage']}/{len(train)}",
                flush=True,
            )
            if not summary["complete"]:
                write_json(out / "status.json", {"state": "incomplete"})
                raise RunError("Training checkpoint incomplete; selection and validation remain unopened")
        gaps = [statistics.mean(r["gap"] for r in rows) for rows in training_rows]
        selected = select_checkpoint(gaps)
        # Written before any validation request, so validation results cannot affect the choice.
        freeze = {
            "selected": selected,
            "training_gaps": gaps,
            "prompt_hashes": [digest(p) for p in checkpoints],
        }
        freeze["frozen_at"] = (
            read_json(out / "freeze.json")["frozen_at"] if (out / "freeze.json").exists() else now()
        )
        write_json(out / "freeze.json", freeze, write_once=True)
        validation_rows, table = [], []
        for iteration, meta_prompt in enumerate(checkpoints):
            rows = evaluate_checkpoint(
                api, validation, candidates, meta_prompt, iteration, out, args.concurrency
            )
            validation_rows.append(rows)
            val, tr = summarize(rows, args.seed), summarize(training_rows[iteration], args.seed)
            table.append(
                {**checkpoint_row(iteration, tr, val, selected), "val_coverage": val["paired_coverage"]}
            )
        complete = all(summarize(r)["complete"] for r in validation_rows)
        write_table(out / "scores/checkpoints.csv", table)
        operational_summary(out)
        write_json(out / "costs.json", api.ledger.summary())
        write_json(out / "status.json", {"state": "complete" if complete else "incomplete"})
