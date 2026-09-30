"""Command line: python -m xar prepare-data | prepare-tokenizers | validate-data | audit-run."""

import argparse

from xar.audit import audit_xar_run
from xar.data import load_examples, prepare_data, prepare_tokenizers
from xar.util import ROOT, main_guard

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAR data preparation, validation and run audit")
    parser.add_argument(
        "command", choices=["prepare-data", "prepare-tokenizers", "validate-data", "audit-run"]
    )
    parser.add_argument("--source-run")
    args = parser.parse_args()
    if args.command == "prepare-data":
        main_guard(prepare_data)
    elif args.command == "prepare-tokenizers":
        main_guard(prepare_tokenizers)
    elif args.command == "validate-data":
        main_guard(
            lambda: print(
                f"Validated {len(load_examples(ROOT / 'data/examples.jsonl', ROOT / 'data/splits.json'))} examples"
            )
        )
    else:

        def audit_run():
            audit_xar_run(args.source_run)
            print(f"{args.source_run}: audit passed")

        main_guard(audit_run)
