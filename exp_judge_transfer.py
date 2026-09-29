"""Apply alternate judges to exact saved P0/P* rubrics and anonymous candidates."""
import json
from pathlib import Path

from shared import (
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
)


def main():
    parser = common_parser(__doc__, ["judge"])
    parser.set_defaults(split="validation")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--checkpoints", nargs="+", choices=["initial", "selected", "terminal"], default=["initial", "selected"])
    parser.add_argument("--evaluation-judge-models", nargs="+", default=["z-ai/glm-5.2", "qwen/qwen3.5-9b"])
    parser.add_argument("--evaluation-provider-map", default="{}", help="JSON map keyed by model slug")
    args = parse_args(parser)
    source, freeze, examples, candidates, rubrics = source_artifacts(args)
    providers = json.loads(args.evaluation_provider_map)
    roles = {"evaluation_" + str(i): role_config(args, "judge", model=model, provider=providers.get(model))
             for i, model in enumerate(args.evaluation_judge_models)}
    if args.dry_run:
        estimate(args, roles, examples, {r: 2*len(examples)*len(args.checkpoints) for r in roles})
        return
    with run_lock(args.output_dir):
        api = initialize_run(args, roles, "judge_transfer", {"source_hash": source["substantive_hash"], "freeze_hash": digest(freeze)})
        report = {}
        for role in roles:
            rows = {}
            for label in args.checkpoints:
                rows[label] = evaluate_checkpoint(api, examples, candidates, None, label, args.output_dir,
                    args.concurrency, frozen_rubrics=rubrics[label], judge_role=role, namespace=role)
            report[roles[role]["model"]] = {"summaries": {label: summarize(values, args.seed) for label, values in rows.items()},
                "improvement": paired_improvement(rows["initial"], rows["selected"], args.seed)
                               if "initial" in rows and "selected" in rows else None,
                "rubric_regeneration": False, "optimization": False}
        write_json(Path(args.output_dir) / "results.json", report)
        operational_summary(args.output_dir)
        write_json(Path(args.output_dir) / "costs.json", api.ledger.summary())
        write_json(Path(args.output_dir) / "status.json", {"state": "complete" if all(s["complete"] for r in report.values()
                   for s in r["summaries"].values()) else "incomplete", "paid": True})


if __name__ == "__main__":
    main_guard(main)
