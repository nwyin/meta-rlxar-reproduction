"""Paths, errors, the experiment settings, hashing, JSON I/O and small concurrency helpers."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import json
import os
import threading
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
ROLES = ("writer", "rubric", "optimizer", "judge")


class RunError(RuntimeError):
    """The run cannot continue: bad settings, bad saved data, or an unexpected API result."""


class BudgetStop(RunError):
    """Continuing could exceed a budget or pay twice for the same request."""


class InvalidOutput(RunError):
    """A model reply failed validation; OpenRouter.structured() retries once with the error."""


class Skipped(RunError):
    """Work was not started because another task had already failed."""


def canonical(value):
    """Deterministic JSON (sorted keys, no whitespace) used for hashing and request bodies."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    """SHA-256 hex of a string, or of any other value's canonical JSON."""
    return hashlib.sha256((value if isinstance(value, str) else canonical(value)).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def words(text):
    """Count whitespace-separated words; headings and citation markers count as words too."""
    return len(text.split())


def normalize(text):
    return " ".join(text.split())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, write_once=False):
    """Write JSON atomically. With write_once, an existing file must already hold the same value."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_once and path.exists():
        if canonical(read_json(path)) != canonical(value):
            raise RunError(f"Refusing to overwrite {path}: it already exists with different content")
        return
    # A temp name per process and thread, so concurrent writers never share a temp file.
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def load_design():
    """The experiment settings in configs/experiments.yaml, plus each role's model.

    The models are set once, in configs/models.yaml under roles.<role>.model; they are copied in
    under the role's name (design["writer"] and so on) so that code can compare a run against
    one dict.
    """
    design = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    roles = yaml.safe_load((ROOT / "configs/models.yaml").read_text())["roles"]
    return {**design, **{role: roles[role]["model"] for role in ROLES}}


def prompt(name):
    return (ROOT / "prompts" / f"{name}.md").read_text()


def bounded_map(function, items, concurrency, stopped):
    """Run function over items on up to `concurrency` threads; return the results in input order.

    A failure sets `stopped`, and items that start after that raise Skipped instead of running.
    When every item has finished or been skipped, the first failure other than Skipped is raised.
    """

    def invoke(item):
        if stopped.is_set():
            raise Skipped("Skipped because another task failed")
        try:
            return function(item)
        except BaseException:
            stopped.set()
            raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(invoke, item) for item in items]
        try:
            finished = list(concurrent.futures.as_completed(futures))
        except BaseException:
            stopped.set()  # interrupted while waiting: items that have not started will skip themselves
            raise
    failures = [future.exception() for future in finished if future.exception() is not None]
    if failures:
        # Raise the error that caused the stop, not one of the Skipped errors that followed it.
        original = [error for error in failures if not isinstance(error, Skipped)]
        raise (original or failures)[0]
    return [future.result() for future in futures]


def write_table(path, rows):
    """Write rows (dicts with the same keys) as CSV; writes nothing when rows is empty."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


MAX_CONCURRENCY = 32


def parse_concurrency(value):
    """argparse type for --concurrency: an integer from 1 to MAX_CONCURRENCY."""
    number = int(value)
    if not 1 <= number <= MAX_CONCURRENCY:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_CONCURRENCY}")
    return number


@contextlib.contextmanager
def run_lock(output):
    """Hold an exclusive lock on the run directory so two processes never work on one run."""
    Path(output).mkdir(parents=True, exist_ok=True)
    with (Path(output) / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunError(f"Another process is already working on {output}") from None
        yield


def main_guard(main):
    """Run a command, turning an expected failure into a one-line STOP message and exit status 2.

    Other exceptions keep their traceback, since they point at a bug.
    """
    try:
        main()
    except (RunError, httpx.HTTPError) as e:
        print(f"STOP: {e}")
        raise SystemExit(2) from None
