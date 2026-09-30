"""The XAR method: write candidate sections, generate rubrics, grade, and optimize the meta prompt.

run_xar runs one trajectory. It optimizes the rubric meta prompt on the training papers, freezes
the checkpoint with the best training gap, and only then scores every checkpoint on validation.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from dataclasses import dataclass
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
    Skipped,
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

# Output schemas. Their canonical JSON is part of every request key, so the saved runs only
# re-audit while these produce exactly the same schemas.
TEXT = {"type": "string", "minLength": 1}
SCORE = {"type": "number", "minimum": 0, "maximum": 10}


def object_schema(properties):
    """A JSON object with exactly these properties, all required."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def criteria_list(item):
    """A rubric has 4 to 8 criteria, and a grade has one score per criterion."""
    return {"type": "array", "minItems": 4, "maxItems": 8, "items": item}


CRITERION_SCHEMA = object_schema({"id": TEXT, "description": TEXT, "low": TEXT, "middle": TEXT, "high": TEXT})
CRITERION_SCORE_SCHEMA = object_schema({"id": TEXT, "score": SCORE, "evidence": TEXT})
RUBRIC_SCHEMA = object_schema({"criteria": criteria_list(CRITERION_SCHEMA)})
GRADE_SCHEMA = object_schema({"scores": criteria_list(CRITERION_SCORE_SCHEMA)})
PROPOSAL_SCHEMA = object_schema({"prompt": TEXT, "rationale": TEXT})

MAX_RUBRIC_WORDS = 1000
QUOTED_TEXT = re.compile(r'["“]([^"”]+)["”]')  # text inside straight or curly double quotes

WRITER_ATTEMPTS = 3  # the first draft plus up to two length revisions
PROPOSAL_ATTEMPTS = 2  # the first proposal plus one repair if the static audit rejects it


def validate_rubric(rubric):
    """Check a generated rubric: schema, unique criterion IDs and total length.

    Raises with a short message; structured() sends that message back in its format repair."""
    jsonschema.validate(rubric, RUBRIC_SCHEMA)
    ids = [c["id"] for c in rubric["criteria"]]
    if len(ids) != len(set(ids)):
        raise InvalidOutput("Duplicate rubric criterion IDs")
    total = sum(words(text) for criterion in rubric["criteria"] for text in criterion.values())
    if total > MAX_RUBRIC_WORDS:
        raise InvalidOutput("Rubric exceeds 1000 words")


def validate_grade(grade, rubric, supplied_text):
    """Check a judge's grade against its rubric and return the mean criterion score.

    Every criterion must be scored exactly once, and anything the judge quotes as evidence
    must occur in supplied_text (the graded section plus the visible paper)."""
    jsonschema.validate(grade, GRADE_SCHEMA)
    ids = [s["id"] for s in grade["scores"]]
    if len(ids) != len(set(ids)) or set(ids) != {c["id"] for c in rubric["criteria"]}:
        raise InvalidOutput("Grade criterion coverage mismatch")
    # json.loads accepts NaN, and NaN passes the schema's minimum and maximum.
    if not all(math.isfinite(s["score"]) for s in grade["scores"]):
        raise InvalidOutput("Grade score must be a finite number")
    for score in grade["scores"]:
        for quoted in QUOTED_TEXT.findall(score["evidence"]):
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
    """Write one model section per example, `concurrency` examples at a time.

    Each section gets up to WRITER_ATTEMPTS drafts: a draft that is cut off or outside the length
    window goes back to the writer with a revision note, and the last draft is kept either way.
    Each section is saved as it finishes, so a resumed run reuses it instead of calling the
    writer again. Returns the records keyed by example ID."""
    config_hash = digest(
        {"writer": api.roles["writer"], "prompt": prompt("writer"), "schema_version": SCHEMA_VERSION}
    )
    stopped = api.dispatch_stopped

    def write_section(example):
        eid = example["example_id"]
        path = Path(output) / "generations" / (eid + ".json")
        if path.exists():
            record = read_json(path)
        else:
            record = new_section(example, path)
        if record["context_hash"] != example["context_hash"] or record["text_hash"] != digest(record["text"]):
            raise RunError(
                f"{path} does not match example {eid} (its context or text hash differs); "
                "the dataset or the saved file changed, so start a new run directory"
            )
        if record["writer_configuration_hash"] != config_hash:
            raise RunError(
                f"{path} was written with different writer settings or prompt; start a new run directory"
            )
        return record

    def new_section(example, path):
        eid = example["example_id"]
        low, high = length_window(example["target_words"])
        attempts = []
        for attempt in range(WRITER_ATTEMPTS):
            data = task_data(example)
            if attempt:
                data.update(
                    previous_section=attempts[-1]["text"],
                    length_revision=length_revision_note(example["target_words"]),
                )
            if stopped.is_set():
                raise Skipped(f"Stopped writing {eid} because another section failed")
            identity = {"writer": config_hash, "example": eid, "attempt": attempt}
            response = api.call("writer", prompt("writer"), data, None, identity)
            text = response["content"].strip()
            count = words(text)
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
            "example_id": eid,
            "context_hash": example["context_hash"],
            "writer_configuration_hash": config_hash,
            "writer_configuration": api.roles["writer"],
            "attempts": attempts,
            "accepted_attempt": accepted["attempt_index"],
            "text": accepted["text"],
            "text_hash": accepted["text_hash"],
            "words": accepted["words"],
            "length_compliant": accepted["length_compliant"],
            "complete": accepted["response"]["finish_reason"] == "stop",
            "length_ratio": accepted["words"] / example["target_words"],
            "contamination": contamination(accepted["text"], example["reference"]),
        }
        write_json(path, record, write_once=True)
        return record

    records = bounded_map(write_section, examples, concurrency, stopped)
    # bounded_map returns results in input order, so candidates.json keeps the dataset order.
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
    if result["value"]:
        record["total"] = statistics.mean(s["score"] for s in result["value"]["scores"])
    else:
        record["total"] = None
    write_json(output, record, write_once=True)
    return record


def evaluate_checkpoint(api, examples, candidates, meta_prompt, checkpoint, output, concurrency):
    """Score one meta prompt on examples: generate a rubric per example, then grade both sections.

    The judge grades the human and the model section separately, in a seeded random order, and
    never learns which is which. Returns one row per example with the human-minus-model gap
    (None if the rubric or a grade is missing), and saves the rows next to the grades."""
    root = Path(output)

    def evaluate(example):
        eid = example["example_id"]
        candidate = candidates[eid]
        # The label is part of the saved paths and request identities, "main/" prefix included.
        label = f"main/{checkpoint}/{example['split']}/{eid}"
        rubric_path = root / "rubrics" / label / "rubric.json"
        rubric = generate_rubric(api, example, meta_prompt, {"rubric": label}, rubric_path)
        origins = ["human", "model"]
        order_seed = int(digest({"seed": api.seed, "label": label})[:16], 16)
        random.Random(order_seed).shuffle(origins)
        grade_paths = {origin: root / "scores" / label / (origin + ".json") for origin in origins}
        totals = {}
        for slot, origin in enumerate(origins):
            text = example["reference"] if origin == "human" else candidate["text"]
            # The origin appears only in the file path, never in the request or its identity.
            identity = {"grade": label, "slot": slot}
            grade = grade_candidate(api, example, rubric, text, identity, grade_paths[origin])
            totals[origin] = grade["total"]
        human, model = totals["human"], totals["model"]
        return {
            "example_id": eid,
            "paper_id": example["paper_id"],
            "section_type": example["section_type"],
            "split": example["split"],
            "checkpoint": checkpoint,
            "human": human,
            "model": model,
            "gap": human - model if human is not None and model is not None else None,
            "length_ratio": candidate["length_ratio"],
            "length_compliant": candidate["length_compliant"],
            "contamination_flagged": candidate["contamination"]["flagged"],
            "rubric_path": str(rubric_path),
            "grade_paths": {origin: str(path) for origin, path in grade_paths.items()},
        }

    rows = bounded_map(evaluate, examples, concurrency, api.dispatch_stopped)
    write_json(root / "scores" / "main" / str(checkpoint) / f"{examples[0]['split']}_rows.json", rows)
    return rows


# A proposal matching any of these is rejected. The optimizer may only change what the rubric
# looks for, not reward a section's origin or change how the fixed wrapper turns grades into scores.
PROPOSAL_RED_FLAGS = {
    # Tells the rubric to favour the human section or to penalize the model's.
    "origin_preference": r"(?:prefer|reward|favor|favour|boost)\s+(?:the\s+)?(?:human|expert|original)|"
    r"(?:penaliz|penalis|punish|downscore)\w*\s+(?:the\s+)?(?:model|ai|generated)",
    # Tells the rubric to work out who wrote the section.
    "origin_detection": r"(?:detect|guess|infer|identify|determine)\s+(?:the\s+)?(?:authorship|origin|provenance)|"
    r"(?:human.written|ai.generated|model.generated)\s+(?:tells|markers|signals)",
    # Overrides the wrapper, or weights or rescales the 0-10 criterion scores.
    "wrapper_override": r"(?:ignore|override|replace)\s+(?:the\s+)?(?:wrapper|system|grading|schema)|"
    r"(?:weighted\s+(?:mean|average)|unequal\s+weight|score\s+(?:from\s+)?0\s*(?:to|[-–])\s*100)",
}
SPAN = 12  # a run of this many words copied from a training paper counts as leakage
MIN_IDENTIFIER_LENGTH = 5  # shorter paper IDs, titles or author names match by chance


def audit_proposal(text, examples, initial_prompt, max_words):
    """Statically check an optimizer proposal before it can become the next meta prompt.

    Rejects a prompt that is longer than max_words, matches PROPOSAL_RED_FLAGS, names a training
    paper or its authors, or copies SPAN consecutive words from a training paper (unless the
    initial prompt already contains them). The completed-run audit recomputes this result and
    compares it with the saved one, so its output must not change."""
    word_count = words(text)
    reasons, flags = [], []
    if word_count > max_words:
        reasons.append("meta_prompt_word_bound")
    lower = text.casefold()
    for name, pattern in PROPOSAL_RED_FLAGS.items():
        if re.search(pattern, lower):
            reasons.append(name)
    proposal_words = lower.split()
    initial_lower = initial_prompt.casefold()
    for example in examples:
        paper = example["paper_id"]
        metadata = example["provenance"]
        for identifier in [paper, metadata["title"], *metadata.get("authors", [])]:
            if len(identifier) >= MIN_IDENTIFIER_LENGTH and identifier.casefold() in lower:
                flags.append({"type": "training_identifier", "paper_id": paper, "match": identifier})
        for field in ("reference", "context"):
            source_words = example[field].casefold().split()
            source_spans = {tuple(source_words[i : i + SPAN]) for i in range(len(source_words) - SPAN + 1)}
            for i in range(len(proposal_words) - SPAN + 1):
                span = proposal_words[i : i + SPAN]
                span_text = " ".join(span)
                if tuple(span) in source_spans and span_text not in initial_lower:
                    flags.append({"type": "copied_training_span", "paper_id": paper, "match": span_text})
                    break
    if flags:
        reasons.append("training_leakage")
    return {
        "accepted": not reasons,
        "reasons": sorted(reasons),
        "flags": flags,
        "words": word_count,
        "policy": "fixed_static_scan_v1",
        "limitation": "Static scan cannot prove semantic absence of provenance heuristics",
    }


def propose_prompt(api, current, feedback, examples, initial, iteration, *, output, max_words):
    """Ask the optimizer for the next meta prompt, given the training feedback on the current one.

    A proposal that is malformed or fails audit_proposal is sent back once with the reasons. If
    the repair fails too, the update is used up and the current prompt carries over unchanged.
    Saves the feedback and proposal under feedback/iter_XX/ and returns the next prompt."""
    directory = Path(output) / "feedback" / f"iter_{iteration:02d}"
    write_json(directory / "training.json", feedback, write_once=True)
    data = {"feedback": feedback, "max_meta_prompt_words": max_words}
    attempts = []
    for attempt in range(PROPOSAL_ATTEMPTS):
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
        if result["value"]:
            audit = audit_proposal(result["value"]["prompt"], examples, initial, max_words)
        else:
            audit = {"accepted": False, "reasons": ["invalid_format"], "flags": []}
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
    """The optimizer's input: the current prompt, its training summary, and the failure_count
    training sections with the smallest human-minus-model gap, with their rubrics and grades.

    Only training examples may appear. The completed-run audit rebuilds this and compares it
    with the saved feedback, so its output must not change."""
    not_training = sorted({x["example_id"] for x in [*examples, *rows] if x["split"] != "train"})
    if not_training:
        raise RunError(
            f"Optimizer feedback may only use training sections, but got {', '.join(not_training)}"
        )
    ungraded = [r["example_id"] for r in rows if r["gap"] is None]
    if ungraded:
        raise RunError(
            f"Training sections {', '.join(ungraded)} have no gap (a rubric or grade is missing); "
            "the optimizer needs every training section graded"
        )
    lookup = {e["example_id"]: e for e in examples}
    failures = sorted(rows, key=lambda r: (r["gap"], r["example_id"]))[:failure_count]
    selected = []
    for row in failures:
        example = lookup[row["example_id"]]
        selected.append(
            {
                "example_id": example["example_id"],
                "paper_id": example["paper_id"],
                "section_type": example["section_type"],
                "target_words": example["target_words"],
                "visible_paper": example["context"],
                "human_candidate": example["reference"],
                "model_candidate": candidates[example["example_id"]]["text"],
                "rubric": read_json(row["rubric_path"])["value"],
                "grades": {origin: read_json(path)["value"] for origin, path in row["grade_paths"].items()},
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


@dataclass
class RunSettings:
    """Everything run_xar needs for one trajectory; run.py builds it from configs/experiments.yaml.

    The manifest saves every field except output_dir, dry_run and resume under "arguments", and
    all of those except the two budgets go into its substantive hash, which --resume checks."""

    output_dir: str
    split: str  # "pilot" or "research"
    seed: int
    iterations: int
    max_meta_prompt_words: int
    failure_examples: int
    writer_model: str
    rubric_model: str
    optimizer_model: str
    judge_model: str
    concurrency: int
    budget_ledger: str
    budget_usd: float | None = None
    total_budget_usd: float | None = None
    dataset: str = "data/examples.jsonl"
    splits: str = "data/splits.json"
    dry_run: bool = False
    resume: bool = False


def planned_requests(examples, iterations):
    """Request counts for the dry-run estimate, not counting format repairs."""
    checkpoints = iterations + 1
    return {
        "writer": len(examples),
        "rubric": len(examples) * checkpoints,
        "judge": 2 * len(examples) * checkpoints,  # one grade each for the human and model section
        "optimizer": iterations,
    }


def optimize_on_training(api, settings, train, candidates, initial):
    """Score the initial prompt on the training sections, then make settings.iterations updates.

    Each update is proposed from the previous checkpoint's training feedback and then scored on
    training too. Returns the prompts (index = update number) and their training summaries."""
    out = Path(settings.output_dir)
    prompts = [initial]
    summaries = []
    rows = None  # the previous checkpoint's training rows
    for iteration in range(settings.iterations + 1):
        if iteration:
            feedback = build_feedback(train, candidates, rows, prompts[-1], settings.failure_examples)
            proposal = propose_prompt(
                api,
                prompts[-1],
                feedback,
                train,
                initial,
                iteration,
                output=out,
                max_words=settings.max_meta_prompt_words,
            )
            prompts.append(proposal)
        path = out / "prompts" / f"iter_{iteration:02d}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(prompts[-1])
        rows = evaluate_checkpoint(api, train, candidates, prompts[-1], iteration, out, settings.concurrency)
        summary = summarize(rows, settings.seed)
        summaries.append(summary)
        print(
            f"Checkpoint {iteration}: train gap {summary['gap']}, "
            f"coverage {summary['paired_coverage']}/{len(train)}",
            flush=True,
        )
        if not summary["complete"]:
            write_json(out / "status.json", {"state": "incomplete"})
            raise RunError(
                f"Checkpoint {iteration}: only {summary['paired_coverage']}/{len(train)} training "
                "sections have a valid rubric and both grades, so no checkpoint can be selected. "
                f"The failed responses are saved under {out}; --resume would reuse them, "
                "so start a new run directory."
            )
    return prompts, summaries


def freeze_selection(output, prompts, train_summaries):
    """Select the checkpoint with the best training gap and record it in freeze.json.

    Called before any validation request: the audit checks that every validation request was
    sent after frozen_at, which shows that validation scores played no part in the choice."""
    gaps = [summary["gap"] for summary in train_summaries]
    selected = select_checkpoint(gaps)
    path = Path(output) / "freeze.json"
    # A resumed run keeps the original timestamp, so the write-once check sees the same record.
    frozen_at = read_json(path)["frozen_at"] if path.exists() else now()
    freeze = {
        "selected": selected,
        "training_gaps": gaps,
        "prompt_hashes": [digest(p) for p in prompts],
        "frozen_at": frozen_at,
    }
    write_json(path, freeze, write_once=True)
    print(f"Selected checkpoint {selected} on training; scoring every checkpoint on validation", flush=True)
    return selected


def score_on_validation(api, settings, validation, candidates, prompts):
    """Score every checkpoint's prompt on the validation sections; one summary per checkpoint."""
    summaries = []
    for iteration, meta_prompt in enumerate(prompts):
        rows = evaluate_checkpoint(
            api, validation, candidates, meta_prompt, iteration, settings.output_dir, settings.concurrency
        )
        summaries.append(summarize(rows, settings.seed))
    return summaries


def run_xar(settings):
    """Run one trajectory: write the candidate sections, optimize the meta prompt on training,
    freeze the selected checkpoint, then score every checkpoint on validation.

    With settings.dry_run, only prints the cost estimate."""
    initial = prompt("rubric_initial")
    if words(initial) > settings.max_meta_prompt_words:
        raise RunError(
            f"prompts/rubric_initial.md has {words(initial)} words; "
            f"max_meta_prompt_words is {settings.max_meta_prompt_words}"
        )
    examples = run_examples(load_examples(settings.dataset, settings.splits), settings.split)
    train = [e for e in examples if e["split"] == "train"]
    validation = [e for e in examples if e["split"] == "validation"]
    roles = {role: role_config(role, getattr(settings, f"{role}_model")) for role in ROLES}
    if settings.dry_run:
        estimate(settings, roles, examples, planned_requests(examples, settings.iterations))
        return
    out = Path(settings.output_dir)
    with run_lock(out):
        api = initialize_run(settings, roles, "xar", {"initial_meta_prompt_hash": digest(initial)})
        candidates = writer_candidates(api, examples, out, settings.concurrency)
        prompts, train_summaries = optimize_on_training(api, settings, train, candidates, initial)
        selected = freeze_selection(out, prompts, train_summaries)
        validation_summaries = score_on_validation(api, settings, validation, candidates, prompts)
        table = []
        for iteration, (train_summary, validation_summary) in enumerate(
            zip(train_summaries, validation_summaries, strict=True)
        ):
            row = checkpoint_row(iteration, train_summary, validation_summary, selected)
            table.append({**row, "val_coverage": validation_summary["paired_coverage"]})
        write_table(out / "scores/checkpoints.csv", table)
        operational_summary(out)
        write_json(out / "costs.json", api.ledger.summary())
        complete = all(summary["complete"] for summary in validation_summaries)
        write_json(out / "status.json", {"state": "complete" if complete else "incomplete"})
