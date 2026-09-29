"""One independent W x G x O trajectory, training selection, then held-out curves."""
import statistics
from pathlib import Path

from shared import (
    ContractError,
    common_parser,
    digest,
    estimate,
    evaluate_checkpoint,
    initialize_run,
    main_guard,
    now,
    operational_summary,
    paired_improvement,
    parse_args,
    propose_prompt,
    read_json,
    role_config,
    run_lock,
    selected_examples,
    summarize,
    words,
    write_json,
    write_table,
    writer_candidates,
)


def build_feedback(examples, candidates, rows, current_prompt, failure_count):
    if any(e["split"] != "train" for e in examples) or any(r["split"] != "train" for r in rows):
        raise ContractError("Validation/pilot examples cannot enter optimizer feedback")
    if any(r["gap"] is None for r in rows):
        raise ContractError("Incomplete training feedback; cannot optimize or select a partial checkpoint")
    lookup = {e["example_id"]: e for e in examples}
    failures = sorted(rows, key=lambda r: (r["gap"], r["example_id"]))[:failure_count]
    selected = []
    for row in failures:
        e = lookup[row["example_id"]]
        selected.append({"example_id": e["example_id"], "paper_id": e["paper_id"],
                        "section_type": e["section_type"], "target_words": e["target_words"],
                        "visible_paper": e["context"], "human_candidate": e["reference"],
                        "model_candidate": candidates[e["example_id"]]["text"],
                        "rubric": read_json(row["rubric_path"])["value"],
                        "grades": {o: read_json(p)["value"] for o, p in row["grade_paths"].items()},
                        "gap": row["gap"]})
    return {"current_prompt": current_prompt, "summary": summarize(rows), "failures": selected,
            "example_ids_used_for_aggregate": sorted(lookup), "selection": "smallest_gap_then_example_id"}



def main():
    parser = common_parser(__doc__, ["writer", "rubric", "optimizer", "judge"])
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--initial-meta-prompt", default="prompts/rubric_initial.md")
    parser.add_argument("--max-meta-prompt-words", type=int, default=800)
    parser.add_argument("--failure-examples", type=int, default=4)
    parser.add_argument("--writer-generations")
    args = parse_args(parser)
    if not 0 <= args.iterations <= 7 or args.failure_examples < 1:
        parser.error("0–7 iterations and a positive failure count required")
    initial = Path(args.initial_meta_prompt).read_text()
    if words(initial) > args.max_meta_prompt_words:
        raise ContractError("Initial prompt exceeds bound")
    examples = selected_examples(args)
    if args.split == "pilot":
        papers = sorted({e["paper_id"] for e in examples})
        if len(papers) < 2:
            raise ContractError("Pilot needs two separate papers")
        examples = [{**e, "split": "train" if e["paper_id"] == papers[0] else "validation"} for e in examples]
    elif args.split != "research":
        raise ContractError("XAR requires research train/validation or separate pilots")
    train, validation = [e for e in examples if e["split"] == "train"], [e for e in examples if e["split"] == "validation"]
    roles = {r: role_config(args, r) for r in ("writer", "rubric", "optimizer", "judge")}
    if args.dry_run:
        estimate(args, roles, examples, {"writer": 0 if args.writer_generations else len(examples),
            "rubric": len(examples)*(args.iterations+1), "judge": 2*len(examples)*(args.iterations+1),
            "optimizer": args.iterations})
        return
    with run_lock(args.output_dir):
        out = Path(args.output_dir)
        api = initialize_run(args, roles, "xar", {"initial_meta_prompt_hash": digest(initial)})
        candidates = writer_candidates(api, examples, out, args.writer_generations)
        checkpoints, training_rows = [initial], []
        for iteration in range(args.iterations+1):
            if iteration:
                feedback = build_feedback(train, candidates, training_rows[-1], checkpoints[-1], args.failure_examples)
                checkpoints.append(propose_prompt(api, checkpoints[-1], feedback, train, initial, args, iteration))
            path = out / "prompts" / f"iter_{iteration:02d}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_text() != checkpoints[-1]:
                raise ContractError("Frozen checkpoint changed")
            path.write_text(checkpoints[-1])
            rows = evaluate_checkpoint(api, train, candidates, checkpoints[-1], iteration, out, args.concurrency)
            training_rows.append(rows)
            summary = summarize(rows, args.seed)
            write_json(out / f"scores/main/{iteration}/train_summary.json", summary)
            print(f"Checkpoint {iteration}: train gap {summary['gap']}, coverage {summary['paired_coverage']}/{len(train)}", flush=True)
            if not summary["complete"]:
                write_json(out / "status.json", {"state": "incomplete", "reason": "missing_training_grades", "checkpoint": iteration})
                raise ContractError("Training checkpoint incomplete; selection and validation remain unopened")
        gaps = [statistics.mean(r["gap"] for r in rows) for rows in training_rows]
        selected = max(range(len(gaps)), key=lambda i: (gaps[i], -i))
        # This immutable freeze MUST exist before the first held-out request is constructed.
        freeze = {"selected": selected, "terminal": args.iterations, "selection": "highest_train_gap_then_earliest",
                  "training_gaps": gaps, "prompt_hashes": [digest(p) for p in checkpoints],
                  "validation_used_for_selection": False}
        freeze["frozen_at"] = read_json(out / "freeze.json")["frozen_at"] if (out / "freeze.json").exists() else now()
        write_json(out / "freeze.json", freeze, immutable=True)
        validation_rows, table = [], []
        for iteration, meta_prompt in enumerate(checkpoints):
            rows = evaluate_checkpoint(api, validation, candidates, meta_prompt, iteration, out, args.concurrency)
            validation_rows.append(rows)
            val, tr = summarize(rows, args.seed), summarize(training_rows[iteration], args.seed)
            write_json(out / f"scores/main/{iteration}/validation_summary.json", val)
            table.append({"iteration": iteration, "train_human": tr["human"], "train_model": tr["model"],
                          "train_gap": tr["gap"], "val_human": val["human"], "val_model": val["model"],
                          "val_gap": val["gap"], "val_coverage": val["paired_coverage"], "selected_by_train": iteration == selected})
        baseline_gap, final_gap = table[0]["val_gap"], table[selected]["val_gap"]
        complete = all(summarize(r)["complete"] for r in validation_rows)
        write_json(out / "results.json", {"complete": complete, "selected": selected,
            "initial": summarize(validation_rows[0], args.seed), "selected_summary": summarize(validation_rows[selected], args.seed),
            "terminal": summarize(validation_rows[-1], args.seed),
            "improvement": paired_improvement(validation_rows[0], validation_rows[selected], args.seed),
            "descriptive_reversal": complete and baseline_gap < 0 < final_gap,
            "baseline_already_favors_humans": baseline_gap is not None and baseline_gap > 0})
        write_table(out / "scores/checkpoints.csv", table)
        operational_summary(out)
        write_json(out / "costs.json", api.ledger.summary())
        write_json(out / "status.json", {"state": "complete" if complete else "incomplete", "paid": True})


if __name__ == "__main__":
    main_guard(main)
