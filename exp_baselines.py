"""Writer caching, order-balanced pairwise judging, and fresh P0 controls."""
import statistics
from pathlib import Path

from shared import (
    PAIRWISE_SCHEMA,
    bounded_map,
    common_parser,
    estimate,
    evaluate_checkpoint,
    initialize_run,
    main_guard,
    operational_summary,
    parse_args,
    prompt,
    role_config,
    run_lock,
    selected_examples,
    summarize,
    task_data,
    write_json,
    write_table,
    writer_candidates,
)


def main():
    parser = common_parser(__doc__, ["writer", "rubric", "judge"])
    parser.add_argument("--baseline-methods", nargs="+", choices=["pairwise", "vanilla"],
                        default=["pairwise", "vanilla"])
    parser.add_argument("--baseline-repeats", type=int, default=3)
    parser.add_argument("--baseline-evaluation-split", choices=["validation", "all"], default="validation")
    parser.add_argument("--writer-generations")
    args = parse_args(parser)
    if args.baseline_repeats < 1:
        parser.error("--baseline-repeats must be positive")
    examples = selected_examples(args)
    roles = {r: role_config(args, r) for r in ("writer", "rubric", "judge")}
    n = len(examples)
    baseline_examples = [e for e in examples if e["split"] == "validation"] if (
        args.baseline_evaluation_split == "validation" and args.split == "research") else examples
    vanilla_n = len(baseline_examples)
    counts = {"writer": 0 if args.writer_generations else n,
              "rubric": vanilla_n*args.baseline_repeats if "vanilla" in args.baseline_methods else 0,
              "judge": 2*vanilla_n*(args.baseline_repeats if "vanilla" in args.baseline_methods else 0)
                       + 2*n*int("pairwise" in args.baseline_methods)}
    if args.dry_run:
        estimate(args, roles, examples, counts)
        return
    with run_lock(args.output_dir):
        api = initialize_run(args, roles, "baselines")
        candidates = writer_candidates(api, examples, args.output_dir, args.writer_generations, concurrency=args.concurrency)
        tables, pairwise = [], []
        if "pairwise" in args.baseline_methods:
            def evaluate_pairwise(e):
                results, preferences = [], []
                for order in range(2):
                    labels = ["human", "model"] if order == 0 else ["model", "human"]
                    texts = {"human": e["reference"], "model": candidates[e["example_id"]]["text"]}
                    result = api.structured("judge", prompt("pairwise"),
                        {**task_data(e), "A": texts[labels[0]], "B": texts[labels[1]]}, PAIRWISE_SCHEMA,
                        {"pairwise": e["example_id"], "order": order})
                    winner = result["value"]["winner"] if result["value"] else None
                    pref = .5 if winner == "tie" else float(labels[0 if winner == "A" else 1] == "human") if winner else None
                    preferences.append(pref)
                    results.append({"order": order, "origin_mapping": labels, "human_preference": pref, **result})
                record = {"example_id": e["example_id"], "paper_id": e["paper_id"], "split": e["split"],
                          "orders": results, "human_preference": statistics.mean(preferences) if None not in preferences else None,
                          "order_disagreement": preferences[0] != preferences[1] if None not in preferences else None}
                write_json(Path(args.output_dir) / "scores/pairwise" / (e["example_id"] + ".json"), record, immutable=True)
                return record
            pairwise = bounded_map(evaluate_pairwise, examples, args.concurrency, api.dispatch_stopped)
            write_json(Path(args.output_dir) / "scores/pairwise_summary.json", {
                "examples": n, "complete": all(r["human_preference"] is not None for r in pairwise),
                "coverage": sum(r["human_preference"] is not None for r in pairwise),
                "human_preference": statistics.mean(r["human_preference"] for r in pairwise if r["human_preference"] is not None)
                                     if any(r["human_preference"] is not None for r in pairwise) else None,
                "order_disagreements": sum(r["order_disagreement"] is True for r in pairwise)})
        if "vanilla" in args.baseline_methods:
            for replicate in range(args.baseline_repeats):
                # The replicate is part of every request identity: repeats cannot hit the same cache entry.
                for split in sorted({e["split"] for e in baseline_examples}):
                    rows = evaluate_checkpoint(api, [e for e in baseline_examples if e["split"] == split], candidates,
                        prompt("rubric_initial"), replicate, args.output_dir, args.concurrency, namespace="baseline")
                    summary = summarize(rows, args.seed)
                    write_json(Path(args.output_dir) / f"scores/baseline/{replicate}/{split}_summary.json", summary)
                    tables.append({"replicate": replicate, "split": split,
                                   **{k: summary[k] for k in ("examples", "paired_coverage", "complete", "human", "model", "gap")}})
        write_table(Path(args.output_dir) / "scores/baselines.csv", tables)
        operational_summary(args.output_dir)
        write_json(Path(args.output_dir) / "costs.json", api.ledger.summary())
        write_json(Path(args.output_dir) / "status.json", {"state": "complete" if all(r["complete"] for r in tables)
                   and all(r["human_preference"] is not None for r in pairwise) else "incomplete", "paid": True})


if __name__ == "__main__":
    main_guard(main)
