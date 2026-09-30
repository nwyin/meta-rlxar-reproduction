"""Run the blog's single same-model reproduction after a separate capability pilot."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

from shared import (
    ROOT,
    ContractError,
    audit_xar_run,
    main_guard,
    render_report,
    validate_primary_manifest,
    write_json,
)


def build_command(args, design, phase):
    pilot = phase == "pilot"
    output = Path(args.runs_root) / design["pilot_run" if pilot else "research_run"]
    command = [
        sys.executable,
        str(ROOT / "exp_xar.py"),
        "--output-dir",
        str(output),
        "--budget-ledger",
        str(Path(args.runs_root) / "budget_ledger.json"),
        "--concurrency",
        str(args.concurrency or design["concurrency"]),
        "--split",
        "pilot" if pilot else "research",
        "--seed",
        str(design["seed"]),
        "--iterations",
        str(design["pilot_iterations"] if pilot else design["iterations"]),
        "--max-meta-prompt-words",
        str(design["max_meta_prompt_words"]),
        "--failure-examples",
        str(design["failure_examples"]),
    ]
    for role in ("writer", "rubric", "optimizer", "judge"):
        command += [f"--{role}-model", design[role]]
    for flag, value in (("--budget-usd", args.budget_usd), ("--total-budget-usd", args.total_budget_usd)):
        if value is not None:
            command += [flag, str(value)]
    if args.dry_run:
        command += ["--dry-run"]
    if args.resume:
        command += ["--resume"]
    return command


def pilot_gate(root, design):
    run = audit_xar_run(Path(root) / design["pilot_run"])
    manifest = run["manifest"]
    protocol = {**design, "iterations": design["pilot_iterations"]}
    validate_primary_manifest(manifest, protocol)
    if manifest["arguments"]["split"] != "pilot":
        raise ContractError("Separate pilot papers required before research")
    # Gap signs and proposal acceptance never select access to research.
    write_json(
        Path(root) / "meta_blog_pilot_gate.json",
        {
            "operational_checks": "raw_verified",
            "research_allowed": True,
            "gap_sign_used_for_gate": False,
            "pilot_substantive_hash": manifest["substantive_hash"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["pilot", "reproduction", "report", "all"], default="all")
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--total-budget-usd", type=float)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--concurrency", type=int)
    args = parser.parse_args()
    if args.concurrency is not None and not 1 <= args.concurrency <= 32:
        parser.error("Concurrency must be 1–32")
    if (
        args.phase != "report"
        and not args.dry_run
        and (args.budget_usd is None or args.total_budget_usd is None)
    ):
        raise ContractError("Explicit per-run and total dollar ceilings required for a batch")
    design = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    if design["scope"] != "meta_blog_initial_empirical_investigation":
        raise ContractError("Active driver requires the single declared blog reproduction")
    if args.phase == "report":
        render_report(args.runs_root, ROOT / "reports")
        return
    phases = ["pilot", "reproduction"] if args.phase == "all" else [args.phase]
    for phase in phases:
        if phase == "reproduction" and not args.dry_run:
            pilot_gate(args.runs_root, design)
        command = build_command(args, design, phase)
        print("Executing:", " ".join(command), flush=True)
        try:
            subprocess.run(command, check=True, cwd=ROOT)
        except subprocess.CalledProcessError as error:
            raise ContractError(
                f"{phase} stopped (exit {error.returncode}); inspect the saved run receipts"
            ) from None
    if "reproduction" in phases and not args.dry_run:
        render_report(args.runs_root, ROOT / "reports")


if __name__ == "__main__":
    main_guard(main)
