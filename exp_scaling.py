"""Later one-role-at-a-time scaling extension; invokes the same scientific entrypoints."""

import subprocess
import sys
from pathlib import Path

from shared import ContractError, common_parser, main_guard, parse_args, read_json, write_json


def main():
    parser = common_parser(__doc__, ["writer", "rubric", "optimizer", "judge"])
    parser.add_argument("--vary-role", choices=["writer", "rubric", "optimizer", "judge"], required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--source-run")
    parser.add_argument("--writer-generations", help="Frozen shared candidates for rubric/optimizer scaling")
    parser.add_argument(
        "--preregistration", help="Required extension-specific frozen JSON before paid execution"
    )
    args = parse_args(parser)
    intervention = "frozen_artifact_regrading" if args.vary_role == "judge" else "XAR_reoptimization"
    if not args.dry_run:
        if not args.preregistration:
            raise ContractError(
                "Intermediate scale is a later extension; an explicit preregistration and separate budgets are required"
            )
        prereg = read_json(args.preregistration)
        if (
            prereg.get("vary_role") != args.vary_role
            or prereg.get("models") != args.models
            or prereg.get("seeds") != args.seeds
        ):
            raise ContractError("Scaling invocation differs from its preregistration")
    else:
        prereg = {"pending": True}
    if not args.dry_run and args.vary_role in {"rubric", "optimizer"} and not args.writer_generations:
        raise ContractError(
            "Rubric/optimizer scaling requires frozen --writer-generations to hold the writer sample fixed"
        )
    commands = []
    for model in args.models:
        for seed in args.seeds:
            target = Path(args.output_dir) / f"{model.replace('/', '_')}-seed{seed}"
            command = [
                sys.executable,
                "exp_judge_transfer.py" if args.vary_role == "judge" else "exp_xar.py",
                "--output-dir",
                str(target),
                "--seed",
                str(seed),
                "--concurrency",
                str(args.concurrency),
            ]
            if args.vary_role == "judge":
                if not args.source_run:
                    raise ContractError("Frozen judge scaling requires --source-run")
                command += ["--source-run", args.source_run, "--evaluation-judge-models", model]
            else:
                command += ["--dataset", args.dataset, "--splits", args.splits, "--split", args.split]
                if args.vary_role in {"rubric", "optimizer"} and args.writer_generations:
                    command += ["--writer-generations", args.writer_generations]
                elif args.vary_role == "writer" and seed != args.seeds[0]:
                    first = Path(args.output_dir) / f"{model.replace('/', '_')}-seed{args.seeds[0]}"
                    command += ["--writer-generations", str(first / "generations/candidates.json")]
                for role in ("writer", "rubric", "optimizer", "judge"):
                    chosen = model if role == args.vary_role else getattr(args, role + "_model")
                    if chosen:
                        command += ["--" + role + "-model", chosen]
                    for field in (
                        "provider",
                        "temperature",
                        "reasoning_mode",
                        "reasoning_effort",
                        "max_output_tokens",
                    ):
                        value = getattr(args, role + "_" + field)
                        if value is not None:
                            command += ["--" + role + "-" + field.replace("_", "-"), str(value)]
            for flag, value in (
                ("--budget-usd", args.budget_usd),
                ("--total-budget-usd", args.total_budget_usd),
                ("--budget-ledger", args.budget_ledger),
            ):
                if value is not None:
                    command += [flag, str(value)]
            if args.dry_run:
                command += ["--dry-run"]
            if args.resume:
                command += ["--resume"]
            commands.append(command)
    if not args.dry_run:
        write_json(
            Path(args.output_dir) / "extension.json",
            {"intervention": intervention, "preregistration": prereg, "commands": commands},
            immutable=True,
        )
    print(f"Intervention: {intervention}")
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main_guard(main)
