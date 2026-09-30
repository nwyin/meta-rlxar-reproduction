"""Paths, errors, hashing, JSON I/O and small concurrency helpers shared by every module."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import csv
import datetime as dt
import fcntl
import hashlib
import itertools
import json
import os
import threading
from pathlib import Path

import httpx
import jsonschema

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
ROLES = ("writer", "rubric", "optimizer", "judge")


class RunError(RuntimeError):
    pass


class BudgetStop(RunError):
    pass


class InvalidOutput(RunError):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256((value if isinstance(value, str) else canonical(value)).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(dt.UTC).isoformat()


def words(text):
    """Whitespace words; internal headings/citation markers count; outer target heading excluded."""
    return len(text.split())


def normalize(text):
    return " ".join(text.split())


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value, write_once=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_once and path.exists():
        if canonical(read_json(path)) != canonical(value):
            raise RunError(f"Immutable artifact differs: {path}")
        return
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def prompt(name):
    return (ROOT / "prompts" / f"{name}.md").read_text()


def bounded_map(function, items, concurrency, stopped):
    """Keep at most concurrency tasks active; retain input order and stop refilling on errors."""

    def invoke(item):
        if stopped.is_set():
            raise RunError("Parallel work halted after another task failed")
        try:
            return function(item)
        except BaseException:
            stopped.set()
            raise

    remaining, results = iter(enumerate(items)), {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending = {pool.submit(invoke, item): i for i, item in itertools.islice(remaining, concurrency)}
        try:
            while pending:
                done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    results[pending.pop(future)] = future.result()
                if not stopped.is_set():
                    pending.update(
                        {pool.submit(invoke, item): i for i, item in itertools.islice(remaining, len(done))}
                    )
        except BaseException:
            stopped.set()
            for future in pending:
                future.cancel()
            raise
    return [results[i] for i in sorted(results)]


def write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_concurrency(value):
    """argparse type for --concurrency: an integer from 1 to 32."""
    number = int(value)
    if not 1 <= number <= 32:
        raise argparse.ArgumentTypeError("must be between 1 and 32")
    return number


@contextlib.contextmanager
def run_lock(output):
    Path(output).mkdir(parents=True, exist_ok=True)
    with (Path(output) / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunError("Another process holds this run") from None
        yield


def main_guard(main):
    try:
        main()
    except (RunError, FileNotFoundError, jsonschema.ValidationError, httpx.HTTPError) as e:
        print(f"STOP: {e}")
        raise SystemExit(2) from None
