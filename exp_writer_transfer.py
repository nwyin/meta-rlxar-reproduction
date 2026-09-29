"""Use exact source rubrics with another writer's cached held-out sections."""
from pathlib import Path

from shared import (
    ContractError,
    common_parser,
    digest,
    estimate,
    evaluate_checkpoint,
    initialize_run,
    main_guard,
    operational_summary,
    paired_improvement,
    parse_args,
    role_config,
    run_lock,
    source_artifacts,
    summarize,
    write_json,
    writer_candidates,
)


def main():
    parser = common_parser(__doc__, ["writer", "judge"])
    parser.set_defaults(split="validation")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--checkpoints", nargs="+", choices=["initial", "selected", "terminal"], default=["initial", "selected"])
    parser.add_argument("--target-writer-model", required=True)
    parser.add_argument("--target-generations")
    args = parse_args(parser)
    source, freeze, examples, _, rubrics = source_artifacts(args)
    writer = role_config(args, "writer", model=args.target_writer_model)
    judge = source["roles"]["judge"]
    overridden = role_config(args, "judge") if any(getattr(args, "judge_" + key, None) is not None
        for key in ("model", "provider", "temperature", "reasoning_mode", "reasoning_effort", "max_output_tokens")) else judge
    if overridden != judge:
        raise ContractError("Cross-writer transfer must hold the source judge configuration fixed")
    roles = {"writer": writer, "judge": judge}
    if args.dry_run:
        estimate(args, roles, examples, {"writer": 0 if args.target_generations else len(examples),
                                       "judge": 2*len(examples)*len(args.checkpoints)})
        return
    with run_lock(args.output_dir):
        api = initialize_run(args, roles, "writer_transfer", {"source_hash": source["substantive_hash"], "freeze_hash": digest(freeze)})
        candidates = writer_candidates(api, examples, args.output_dir, args.target_generations, concurrency=args.concurrency)
        rows = {label: evaluate_checkpoint(api, examples, candidates, None, label, args.output_dir,
                args.concurrency, frozen_rubrics=rubrics[label], namespace="writer_transfer") for label in args.checkpoints}
        summaries = {label: summarize(values, args.seed) for label, values in rows.items()}
        write_json(Path(args.output_dir) / "results.json", {"summaries": summaries,
            "improvement": paired_improvement(rows["initial"], rows["selected"], args.seed)
                           if "initial" in rows and "selected" in rows else None, "rubric_regeneration": False})
        operational_summary(args.output_dir)
        write_json(Path(args.output_dir) / "costs.json", api.ledger.summary())
        write_json(Path(args.output_dir) / "status.json", {"state": "complete" if all(s["complete"] for s in summaries.values()) else "incomplete", "paid": True})


if __name__ == "__main__":
    main_guard(main)
