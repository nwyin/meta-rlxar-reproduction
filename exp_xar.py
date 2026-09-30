"""One independent W x G x O trajectory, training selection, then held-out curves."""

import argparse

from xar.pipeline import run_xar
from xar.util import ROLES, main_guard, parse_concurrency


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/examples.jsonl")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget-usd", type=float)
    parser.add_argument("--total-budget-usd", type=float)
    parser.add_argument("--budget-ledger", default="runs/budget_ledger.json")
    parser.add_argument("--concurrency", type=parse_concurrency, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split", choices=["pilot", "research"], default="research")
    for role in ROLES:
        parser.add_argument(f"--{role}-model")
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--max-meta-prompt-words", type=int, default=800)
    parser.add_argument("--failure-examples", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.iterations <= 7 or args.failure_examples < 1:
        parser.error("0–7 iterations and a positive failure count required")
    return args


def main():
    run_xar(parse_args())


if __name__ == "__main__":
    main_guard(main)
