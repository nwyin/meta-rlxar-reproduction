"""Run the Meta blog XAR reproduction, and prepare, check and report on its data.

Run from the repository root. The experiment's settings come from configs/experiments.yaml, and the models and
their settings from configs/models.yaml.
"""

import argparse
from pathlib import Path

from xar.data import load_examples, prepare_data, prepare_tokenizers
from xar.discovery import discover_candidates
from xar.pipeline import RunSettings, run_xar
from xar.report import render_report
from xar.util import (
    MAX_CONCURRENCY,
    ROLES,
    ROOT,
    load_design,
    main_guard,
    parse_concurrency,
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
        feedback_policy=design["feedback_policy"],
        **{f"{role}_model": design[role] for role in ROLES},
        concurrency=options.concurrency or design["concurrency"],
        dry_run=options.dry_run,
        resume=options.resume,
    )


def run_phases(options):
    design = load_design()
    phases = ["pilot", "reproduce"] if options.command == "all" else [options.command]
    for phase in phases:
        settings = settings_for(phase, design, options)
        action = "Estimating" if settings.dry_run else "Running"
        print(f"{action} the {phase} phase in {settings.output_dir}", flush=True)
        run_xar(settings)
    if "reproduce" in phases and not options.dry_run:
        render_report(options.runs_root, options.output_dir)


EPILOG = """\
Setup: copy .env.example to .env and set OPENROUTER_API_KEY.
Order: prepare-tokenizers, discover-data, prepare-data, validate-data, pilot, reproduce (or all), report."""


def parse_args(argv=None):
    design = load_design()
    parser = argparse.ArgumentParser(
        description=__doc__, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    run_options = argparse.ArgumentParser(add_help=False)
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
        help="continue an existing run; only --concurrency may change",
    )
    run_options.add_argument("--runs-root", default="runs", help="directory for the runs (default: runs/)")
    report_options = argparse.ArgumentParser(add_help=False)
    report_options.add_argument(
        "--output-dir",
        default="reports",
        help="where to write results.md and the figures (default: reports/)",
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
        help="run the full experiment on the "
        f"train/validation papers with {design['iterations']} prompt updates, then write the report",
    )
    commands.add_parser("all", parents=[run_options, report_options], help="pilot, then reproduce")
    report = commands.add_parser(
        "report", parents=[report_options], help="write the report from the saved research run"
    )
    report.add_argument(
        "--runs-root", default="runs", help="directory holding the research run (default: runs/)"
    )
    commands.add_parser("validate-data", help="check data/examples.jsonl against data/splits.json")
    commands.add_parser(
        "discover-data", help="sample candidate papers from OpenAlex and arXiv into data/discovery.json"
    )
    commands.add_parser("prepare-data", help="rebuild data/examples.jsonl from data/discovery.json")
    commands.add_parser(
        "prepare-tokenizers",
        help="download the tokenizers pinned in configs/tokenizers.json into data/tokenizers/",
    )
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_args(argv)
    if options.command in PHASES or options.command == "all":
        run_phases(options)
    elif options.command == "report":
        render_report(options.runs_root, options.output_dir)
    elif options.command == "validate-data":
        examples = load_examples(ROOT / "data/examples.jsonl", ROOT / "data/splits.json")
        print(f"Validated {len(examples)} examples")
    elif options.command == "discover-data":
        discover_candidates()
    elif options.command == "prepare-data":
        prepare_data()
    elif options.command == "prepare-tokenizers":
        prepare_tokenizers()


if __name__ == "__main__":
    main_guard(main)
