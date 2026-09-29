"""Sequential batch driver; every scientific experiment remains independently resumable."""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from shared import ROOT, ContractError, audit_xar_run, main_guard, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pilot", "controls", "matrix", "transfers", "confirmation", "all"], required=True)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--total-budget-usd", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--concurrency", type=int, default=2)
    args = parser.parse_args()
    if not args.dry_run and (args.budget_usd is None or args.total_budget_usd is None):
        raise ContractError("Explicit per-run and total dollar ceilings required for a batch")
    design = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    models = {"weak": design["weak"], "strong": design["strong"]}
    root = Path(args.runs_root)
    common = ["--budget-ledger", str(root / "budget_ledger.json"), "--concurrency", str(args.concurrency)]
    for flag, value in (("--budget-usd", args.budget_usd), ("--total-budget-usd", args.total_budget_usd)):
        if value is not None:
            common += [flag, str(value)]
    if args.dry_run:
        common += ["--dry-run"]
    if args.resume:
        common += ["--resume"]

    def execute(script, output, extra):
        command = [sys.executable, str(ROOT / script), "--output-dir", str(output), *common, *extra]
        print("Executing:", " ".join(command), flush=True)
        subprocess.run(command, check=True, cwd=ROOT)

    def pilot_gate():
        if args.dry_run:
            return
        for gg in (2, 4):
            audit_xar_run(root / f"pilot-GG{gg}")
        write_json(root / "pilot_gate.json", {"operational_checks": "raw_verified", "research_allowed": True,
                                              "gap_sign_used_for_gate": False})

    phases = ["pilot", "controls", "matrix", "transfers", "confirmation"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if phase == "pilot":
            for gg, generator in ((2, "weak"), (4, "strong")):
                extra = ["--split", "pilot", "--writer-model", models["strong"], "--rubric-model", models[generator],
                         "--optimizer-model", models["strong"], "--judge-model", design["judge"], "--seed", "0"]
                if gg == 4:
                    extra += ["--writer-generations", str(root / "pilot-GG2/generations/candidates.json")]
                execute("exp_xar.py", root / f"pilot-GG{gg}", extra)
        elif phase == "controls":
            pilot_gate()
            for writer, writer_model in models.items():
                for generator, generator_model in models.items():
                    extra = ["--writer-model", writer_model, "--rubric-model", generator_model,
                             "--judge-model", design["judge"], "--baseline-repeats", "3", "--baseline-methods"]
                    extra += ["pairwise", "vanilla"] if generator == "weak" else ["vanilla"]
                    if generator == "strong":
                        extra += ["--writer-generations", str(root / f"baseline-{writer}-weak/generations/candidates.json")]
                    execute("exp_baselines.py", root / f"baseline-{writer}-{generator}", extra)
        elif phase == "matrix":
            pilot_gate()
            for writer, writer_model in models.items():
                artifact = root / f"baseline-{writer}-weak/generations/candidates.json"
                if not args.dry_run and not artifact.exists():
                    raise ContractError("Run controls first to freeze shared writer candidates")
                for gg, (generator, optimizer) in enumerate([(g, o) for g in models for o in models], 1):
                    for seed in design["seeds"]:
                        execute("exp_xar.py", root / f"xar-{writer}-GG{gg}-seed{seed}", [
                            "--writer-model", writer_model, "--rubric-model", models[generator],
                            "--optimizer-model", models[optimizer], "--judge-model", design["judge"],
                            "--writer-generations", str(artifact), "--seed", str(seed)])
        elif phase == "transfers":
            for writer in models:
                target = "weak" if writer == "strong" else "strong"
                for gg in range(1, 5):
                    for seed in design["seeds"]:
                        source = root / f"xar-{writer}-GG{gg}-seed{seed}"
                        execute("exp_judge_transfer.py", root / f"judges-{writer}-GG{gg}-seed{seed}", [
                            "--source-run", str(source), "--evaluation-judge-models", *design["evaluation_judges"], "--seed", str(seed)])
                        execute("exp_writer_transfer.py", root / f"writers-{writer}-GG{gg}-seed{seed}", [
                            "--source-run", str(source), "--target-writer-model", models[target], "--target-generations",
                            str(root / f"baseline-{target}-weak/generations/candidates.json"), "--seed", str(seed)])
        else:
            if not args.dry_run:
                for writer in models:
                    for gg in range(1, 5):
                        for seed in design["seeds"]:
                            status = read_json(root / f"xar-{writer}-GG{gg}-seed{seed}/status.json")
                            if status["state"] != "complete":
                                raise ContractError("Research matrix incomplete; confirmation remains reserved")
            execute("exp_confirmation.py", root / "confirmation", ["--source-run", str(root / "xar-strong-GG2-seed0")])
    if not args.dry_run:
        subprocess.run([sys.executable, str(ROOT / "shared.py"), "report", "--runs-root", str(root)], check=True)


if __name__ == "__main__":
    main_guard(main)
