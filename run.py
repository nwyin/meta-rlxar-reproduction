"""Run the Meta blog XAR reproduction, and prepare, check and report on its data.

Run from the repository root. The experiment's settings come from configs/experiments.yaml.
"""

import argparse
from pathlib import Path

import yaml

from xar.audit import audit_xar_run, validate_primary_manifest
from xar.data import load_examples, prepare_data, prepare_tokenizers
from xar.pipeline import RunSettings, run_xar
from xar.report import render_report
from xar.util import ROLES, ROOT, RunError, main_guard, parse_concurrency, write_json

# phase -> (run directory key, data split, iterations key) in experiments.yaml
PHASES = {
    "pilot": ("pilot_run", "pilot", "pilot_iterations"),
    "reproduce": ("research_run", "research", "iterations"),
}


def load_design():
    return yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())


def settings_for(phase, design, options):
    """The run settings for a phase: the design's values plus the command-line options."""
    run_key, split, iterations_key = PHASES[phase]
    runs_root = Path(options.runs_root)
    return RunSettings(
        output_dir=str(runs_root / design[run_key]),
        split=split,
        seed=design["seed"],
        iterations=design[iterations_key],
        max_meta_prompt_words=design["max_meta_prompt_words"],
        failure_examples=design["failure_examples"],
        **{f"{role}_model": design[role] for role in ROLES},
        concurrency=options.concurrency or design["concurrency"],
        budget_ledger=str(runs_root / "budget_ledger.json"),
        budget_usd=options.budget_usd,
        total_budget_usd=options.total_budget_usd,
        dry_run=options.dry_run,
        resume=options.resume,
    )


def check_pilot(runs_root, design):
    """Audit the pilot run and check that it used the settings in experiments.yaml.

    reproduce calls this first. On success it saves the pilot's substantive hash to
    meta_blog_pilot_gate.json. The pilot's scores are never looked at."""
    run = audit_xar_run(Path(runs_root) / design["pilot_run"])
    manifest = run["manifest"]
    validate_primary_manifest(manifest, {**design, "iterations": design["pilot_iterations"]})
    split = manifest["arguments"]["split"]
    if split != "pilot":
        raise RunError(
            f"{design['pilot_run']} used the {split} split; the pilot must run on the pilot papers"
        )
    write_json(
        Path(runs_root) / "meta_blog_pilot_gate.json",
        {"pilot_substantive_hash": manifest["substantive_hash"]},
    )


def run_phases(options):
    design = load_design()
    phases = ["pilot", "reproduce"] if options.command == "all" else [options.command]
    for phase in phases:
        if phase == "reproduce" and not options.dry_run:
            check_pilot(options.runs_root, design)
        settings = settings_for(phase, design, options)
        action = "Estimating" if settings.dry_run else "Running"
        print(f"{action} the {phase} phase in {settings.output_dir}", flush=True)
        run_xar(settings)
    if "reproduce" in phases and not options.dry_run:
        render_report(options.runs_root, ROOT / "reports")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    run_options = argparse.ArgumentParser(add_help=False)
    run_options.add_argument("--budget-usd", type=float, help="spending limit for this run")
    run_options.add_argument(
        "--total-budget-usd", type=float, help="spending limit across all runs sharing the ledger"
    )
    run_options.add_argument(
        "--concurrency",
        type=parse_concurrency,
        help="parallel requests, 1-32 (default: experiments.yaml)",
    )
    run_options.add_argument(
        "--dry-run", action="store_true", help="print the cost estimate without sending requests"
    )
    run_options.add_argument(
        "--resume", action="store_true", help="continue an existing run whose settings are unchanged"
    )
    run_options.add_argument(
        "--runs-root", default="runs", help="directory for the runs and the shared budget ledger"
    )
    commands.add_parser(
        "pilot",
        parents=[run_options],
        help="a short run on the two pilot papers (pilot_iterations updates) to check the pipeline",
    )
    commands.add_parser(
        "reproduce",
        parents=[run_options],
        help="check the pilot, run the research trajectory, then write reports/",
    )
    commands.add_parser("all", parents=[run_options], help="pilot, then reproduce")
    report = commands.add_parser("report", help="audit the research run and write the report")
    report.add_argument("--runs-root", default="runs")
    report.add_argument("--output-dir", default="reports")
    audit = commands.add_parser("audit-run", help="audit one saved run")
    audit.add_argument("--source-run", required=True)
    commands.add_parser("validate-data", help="check the dataset and splits")
    commands.add_parser("prepare-data", help="rebuild the dataset from data/raw")
    commands.add_parser("prepare-tokenizers", help="download the pinned tokenizers")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    if options.command in PHASES or options.command == "all":
        run_phases(options)
    elif options.command == "report":
        render_report(options.runs_root, options.output_dir)
    elif options.command == "audit-run":
        audit_xar_run(options.source_run)
        print(f"{options.source_run}: audit passed")
    elif options.command == "validate-data":
        examples = load_examples(ROOT / "data/examples.jsonl", ROOT / "data/splits.json")
        print(f"Validated {len(examples)} examples")
    elif options.command == "prepare-data":
        prepare_data()
    elif options.command == "prepare-tokenizers":
        prepare_tokenizers()


if __name__ == "__main__":
    main_guard(main)
