"""Apply a preregistered source configuration/P* to five reserved confirmation papers."""
from pathlib import Path

from shared import (
    ContractError,
    audit_xar_run,
    common_parser,
    digest,
    estimate,
    evaluate_checkpoint,
    file_hash,
    initialize_run,
    main_guard,
    operational_summary,
    paired_improvement,
    parse_args,
    read_json,
    run_lock,
    selected_examples,
    summarize,
    write_json,
    writer_candidates,
)


def main():
    parser = common_parser(__doc__, [])
    parser.set_defaults(split="confirmation")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--writer-generations")
    args = parse_args(parser)
    if args.split != "confirmation":
        raise ContractError("Confirmation must use the reserved confirmation split")
    source = Path(args.source_run)
    manifest, freeze = read_json(source / "manifest.json"), read_json(source / "freeze.json")
    if manifest["experiment"] != "xar" or manifest["arguments"]["split"] != "research":
        raise ContractError("Confirmation needs a completed research XAR source")
    # Configuration is committed before research results, avoiding favorable validation-cell selection.
    prereg = read_json("configs/confirmation.json")
    for role in ("writer", "rubric", "optimizer", "judge"):
        if manifest["roles"][role]["model"] != prereg[role]:
            raise ContractError("Source differs from preregistered confirmation configuration")
    if manifest["arguments"]["seed"] != prereg["source_seed"]:
        raise ContractError("Source trajectory differs from confirmation preregistration")
    for name in ("dataset", "splits"):
        given = getattr(args, name)
        default = "data/examples.jsonl" if name == "dataset" else "data/splits.json"
        path = manifest[name] if given == default else given
        if file_hash(path) != manifest[name + "_hash"]:
            raise ContractError(f"Confirmation {name} differs from the frozen source")
        setattr(args, name, path)
    examples = selected_examples(args)
    if len(examples) != 20 or len({e["paper_id"] for e in examples}) != 5:
        raise ContractError("Confirmation requires five distinct reserved papers / twenty examples")
    roles = {r: manifest["roles"][r] for r in ("writer", "rubric", "judge")}
    prompts = {label: (source / f"prompts/iter_{i:02d}.md").read_text()
               for label, i in (("initial", 0), ("selected", freeze["selected"]))}
    if args.dry_run:
        estimate(args, roles, examples, {"writer": 0 if args.writer_generations else 20, "rubric": 40, "judge": 80})
        return
    audit_xar_run(source)
    with run_lock(args.output_dir):
        api = initialize_run(args, roles, "confirmation", {"source_hash": manifest["substantive_hash"],
            "freeze_hash": digest(freeze), "confirmation_preregistration": prereg,
            "prompt_hashes": {label: digest(text) for label, text in prompts.items()}})
        write_json(Path(args.output_dir) / "confirmation_freeze.json", {"source": str(source.resolve()),
            "selected_checkpoint": freeze["selected"], "configuration": roles,
            "prompt_hashes": {label: digest(text) for label, text in prompts.items()},
            "frozen_before_confirmation_calls": True}, immutable=True)
        candidates = writer_candidates(api, examples, args.output_dir, args.writer_generations, concurrency=args.concurrency)
        rows = {label: evaluate_checkpoint(api, examples, candidates, text, label, args.output_dir,
                        args.concurrency, namespace="confirmation") for label, text in prompts.items()}
        summaries = {label: summarize(values, args.seed) for label, values in rows.items()}
        write_json(Path(args.output_dir) / "results.json", {"summaries": summaries,
            "improvement": paired_improvement(rows["initial"], rows["selected"], args.seed)})
        operational_summary(args.output_dir)
        write_json(Path(args.output_dir) / "costs.json", api.ledger.summary())
        write_json(Path(args.output_dir) / "status.json", {"state": "complete" if all(s["complete"] for s in summaries.values()) else "incomplete", "paid": True})


if __name__ == "__main__":
    main_guard(main)
