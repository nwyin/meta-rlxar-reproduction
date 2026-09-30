"""Run the Meta blog XAR reproduction, and prepare, check and report on its data.

Run from the repository root. The experiment's settings come from configs/experiments.yaml, and the models and
their settings from configs/models.yaml.
"""

import argparse
from pathlib import Path

from xar.audit import audit_xar_run, validate_primary_manifest
from xar.data import load_examples, prepare_data, prepare_tokenizers
from xar.pipeline import RunSettings, run_xar
from xar.report import render_report
from xar.util import (
    MAX_CONCURRENCY,
    ROLES,
    ROOT,
    RunError,
    load_design,
    main_guard,
    parse_concurrency,
    write_json,
)

# phase -> (run directory key, data split, iterations key) in experiments.yaml
PHASES = {
    "pilot": ("pilot_run", "pilot", "pilot_iterations"),
    "reproduce": ("research_run", "research", "iterations"),
}


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

    reproduce calls this first. Only the pilot's settings and audit result are checked, not its
    scores. On success it saves the hash of the pilot's settings (the manifest's substantive_hash)
    to meta_blog_pilot_gate.json. Nothing in the pipeline reads that file; it is a record for
    scripts/audit_completed_run.py."""
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
        render_report(options.runs_root, options.output_dir)


EPILOG = """\
Setup: copy .env.example to .env and set OPENROUTER_API_KEY.
Order: prepare-tokenizers, prepare-data, validate-data, pilot, reproduce (or all), report."""


def parse_args(argv=None):
    design = load_design()
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    run_options = argparse.ArgumentParser(add_help=False)
    run_options.add_argument(
        "--budget-usd", type=float, help="USD limit for this run (required unless --dry-run)"
    )
    run_options.add_argument(
        "--total-budget-usd",
        type=float,
        help="USD limit summed over every run in <runs-root>/budget_ledger.json (required unless --dry-run)",
    )
    run_options.add_argument(
        "--concurrency",
        type=parse_concurrency,
        help=f"parallel requests, 1-{MAX_CONCURRENCY} (default: concurrency in experiments.yaml)",
    )
    run_options.add_argument(
        "--dry-run", action="store_true", help="print the cost estimate without sending requests"
    )
    run_options.add_argument(
        "--resume",
        action="store_true",
        help="continue an existing run; only budgets, concurrency and the ledger path may change",
    )
    run_options.add_argument(
        "--runs-root",
        default="runs",
        help="directory for the runs and the shared budget ledger (default: runs/)",
    )
    report_options = argparse.ArgumentParser(add_help=False)
    report_options.add_argument(
        "--output-dir",
        default="reports",
        help="where to write results.md, audit.json and the figures (default: reports/)",
    )
    commands.add_parser(
        "pilot",
        parents=[run_options],
        help="a short run on the two pilot papers to check the pipeline "
        f"(prompt updates: {design['pilot_iterations']})",
    )
    commands.add_parser(
        "reproduce",
        parents=[run_options, report_options],
        help="audit the completed pilot run (run `pilot` first), run the full experiment on the "
        f"train/validation papers with {design['iterations']} prompt updates, then write the report",
    )
    commands.add_parser("all", parents=[run_options, report_options], help="pilot, then reproduce")
    report = commands.add_parser(
        "report", parents=[report_options], help="audit the research run and write the report"
    )
    report.add_argument(
        "--runs-root", default="runs", help="directory holding the research run (default: runs/)"
    )
    audit = commands.add_parser(
        "audit-run",
        help="re-check a saved run's rubrics, grades and checkpoint table against its raw API responses "
        "(no API calls)",
    )
    audit.add_argument("run_dir", help="a run directory, e.g. runs/meta-blog-seed0")
    commands.add_parser("validate-data", help="check data/examples.jsonl against data/splits.json")
    commands.add_parser("prepare-data", help="rebuild data/examples.jsonl from data/raw")
    commands.add_parser(
        "prepare-tokenizers",
        help="download the tokenizers pinned in configs/tokenizers.json into data/tokenizers/",
    )
    options = parser.parse_args(argv)
    is_run = options.command in PHASES or options.command == "all"
    if is_run and not options.dry_run and None in (options.budget_usd, options.total_budget_usd):
        parser.error("--budget-usd and --total-budget-usd are required unless --dry-run")
    return options


def main(argv=None):
    options = parse_args(argv)
    if options.command in PHASES or options.command == "all":
        run_phases(options)
    elif options.command == "report":
        render_report(options.runs_root, options.output_dir)
    elif options.command == "audit-run":
        audit_xar_run(options.run_dir)
        print(f"{options.run_dir}: audit passed")
    elif options.command == "validate-data":
        examples = load_examples(ROOT / "data/examples.jsonl", ROOT / "data/splits.json")
        print(f"Validated {len(examples)} examples")
    elif options.command == "prepare-data":
        prepare_data()
    elif options.command == "prepare-tokenizers":
        prepare_tokenizers()


if __name__ == "__main__":
    main_guard(main)
