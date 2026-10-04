"""Run a fresh Meta XAR reproduction with OpenRouter.

Prepare data with fetch_data.py, inspect --dry-run, then choose a new --output directory.
Settings and model roles come from configs/; prompts stay in prompts/. No inputs are truncated.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import statistics
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import jsonschema
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
ROLES = ("writer", "rubric", "optimizer", "judge")
SECTIONS = ("abstract", "introduction", "related_work", "conclusion")
REQUIRED_SECTIONS = ("abstract", "introduction")
FAILING_FEEDBACK = "failing_gap_without_paper"
MAX_CONCURRENCY = 32
SCHEMA_VERSION = 1


class RunError(RuntimeError):
    """The run cannot continue safely."""


class InvalidOutput(RunError):
    """A model reply failed validation."""


class UncertainSend(RunError):
    """A response arrived without a usable billed cost."""


class Skipped(RunError):
    """Another parallel task failed before this task started."""


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


def write_json(path, value):
    """Write a complete JSON record atomically, including from parallel tasks."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_design():
    return yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())


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


def parse_concurrency(value):
    """argparse type for --concurrency: an integer from 1 to MAX_CONCURRENCY."""
    number = int(value)
    if not 1 <= number <= MAX_CONCURRENCY:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_CONCURRENCY}")
    return number


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


# Read only the prepared JSON files; fetching and parsing papers live in fetch_data.py.
NGRAM = 8
VERBATIM_FLAG_WORDS = 30
PILOT_PAPERS = 2


def load_examples(dataset, splits):
    """Read the examples from `dataset` (JSON lines) and check them against `splits`.

    Checks that no paper is in two splits, each example's split and content hashes match, the
    withheld section is absent from the visible paper, and every paper has each of its sections
    once (by example ID), and always the abstract and the introduction.
    """
    papers_by_split = read_json(splits)["papers"]
    split_counts = Counter(paper for papers in papers_by_split.values() for paper in papers)
    repeated = sorted(paper for paper, count in split_counts.items() if count > 1)
    if repeated:
        raise RunError(f"Paper splits overlap in {splits}: {', '.join(repeated)} listed more than once")
    examples = [json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()]
    id_counts = Counter(e["example_id"] for e in examples)
    duplicates = sorted(eid for eid, count in id_counts.items() if count > 1)
    if duplicates:
        raise RunError(f"{dataset} repeats example IDs: {', '.join(duplicates)}")
    sections_by_paper = {}
    for e in examples:
        eid = e["example_id"]
        listed_in = [split for split, papers in papers_by_split.items() if e["paper_id"] in papers]
        if listed_in != [e["split"]]:
            raise RunError(
                f"{eid} is labelled {e['split']}, but {splits} lists paper {e['paper_id']} under {', '.join(listed_in) or 'no split'}"
            )
        for field in ("context", "reference"):
            if e[field + "_hash"] != digest(e[field]):
                raise RunError(f"{eid}: {field} text does not match its stored {field}_hash")
        if e["target_words"] != words(e["reference"]):
            raise RunError(f"{eid}: target_words is {e['target_words']}, but the reference has {words(e['reference'])} words")
        if not e["target_words"]:
            raise RunError(f"{eid}: the reference is empty")
        if normalize(e["reference"]) in normalize(e["context"]):
            raise RunError(f"{eid}: the withheld reference still appears in the visible paper")
        sections_by_paper.setdefault(e["paper_id"], []).append(e["section_type"])
    for paper, sections in sections_by_paper.items():
        if not set(REQUIRED_SECTIONS) <= set(sections):
            raise RunError(f"Paper {paper} has sections {sorted(sections)}; expected {list(REQUIRED_SECTIONS)}")
    return examples


def task_data(example):
    """The only example fields that go into writer, rubric and judge prompts.

    Everything else (the author's section, split labels, IDs, earlier feedback) stays out.
    """
    return {
        "visible_paper": example["context"],
        "section_type": example["section_type"],
        "target_words": example["target_words"],
    }


def contamination(candidate, reference):
    """Measure how much of the author's withheld section a candidate reproduces word for word.

    Reports the longest run of consecutive shared words and the fraction of the candidate's
    8-grams that also occur in the reference, and flags runs of VERBATIM_FLAG_WORDS or more.
    """
    candidate_words = normalize(candidate).split()
    reference_words = normalize(reference).split()
    reference_positions = {}
    for position, word in enumerate(reference_words):
        reference_positions.setdefault(word, []).append(position)
    # Longest common run by dynamic programming: run_ending_at[j] is the length of the shared run
    # that ends at the current candidate word and at reference word j.
    longest, run_ending_at = 0, {}
    for word in candidate_words:
        run_ending_at = {j: run_ending_at.get(j - 1, 0) + 1 for j in reference_positions.get(word, [])}
        longest = max(longest, max(run_ending_at.values(), default=0))
    reference_ngrams = {tuple(reference_words[i : i + NGRAM]) for i in range(len(reference_words) - NGRAM + 1)}
    candidate_ngrams = [tuple(candidate_words[i : i + NGRAM]) for i in range(len(candidate_words) - NGRAM + 1)]
    shared = sum(ngram in reference_ngrams for ngram in candidate_ngrams)
    return {
        "longest_verbatim_run_words": longest,
        "eightgram_overlap_fraction": shared / max(1, len(candidate_ngrams)),
        "flagged": longest >= VERBATIM_FLAG_WORDS,
        # Flagged candidates are reported, not dropped.
        "exclusion": False,
    }


def run_examples(examples, split):
    """The examples a run uses, labelled train or validation.

    A research run uses the train and validation papers as saved. A pilot run uses the papers
    labelled pilot in an older dataset, or else the first PILOT_PAPERS training papers; the first
    (sorted by ID) becomes training and the rest become validation.
    """
    if split not in ("pilot", "research"):
        raise RunError(f"Unknown split {split!r}; expected 'pilot' or 'research'")
    if split == "pilot":
        papers = list(dict.fromkeys(e["paper_id"] for e in examples if e["split"] == "pilot"))
        papers = papers or list(dict.fromkeys(e["paper_id"] for e in examples if e["split"] == "train"))
        papers = sorted(papers[:PILOT_PAPERS])
        pilot = [e for e in examples if e["paper_id"] in papers]
        if len(papers) < PILOT_PAPERS:
            raise RunError(f"The pilot needs {PILOT_PAPERS} papers (one to train on, one to validate), but the dataset has {len(papers)}")
        return [{**e, "split": "train" if e["paper_id"] == papers[0] else "validation"} for e in pilot]
    chosen = [e for e in examples if e["split"] in ("train", "validation")]
    if not chosen:
        raise RunError("The dataset has no train or validation examples")
    return chosen


# OpenRouter: fixed endpoints, bounded retries, raw sends and billing records.
API_BASE = "https://openrouter.ai/api/v1"
SNAPSHOTS = ROOT / "configs/snapshots"
MODEL_CATALOG = SNAPSHOTS / "openrouter-models-2026-09-29.json"
REQUIRED_PARAMETERS = {"temperature", "reasoning", "max_tokens", "response_format", "structured_outputs"}
PRICE_KEYS = ("prompt", "completion", "request")
PRICING_HEADROOM = 1.25
MAX_SENDS = 6
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
BACKOFF_SECONDS = (2, 5, 15, 30, 60)
MAX_RETRY_AFTER_SECONDS = 120
FORMAT_REPAIR = "\nFORMAT REPAIR: Return complete valid JSON matching the schema. "
MAX_REPAIR_ERROR_CHARS = 500


def role_config(role):
    config = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    selected = config["roles"][role]
    model = selected["model"]
    check_model_allowed(model)
    return {"model": model, **config["models"][model], "temperature": selected["temperature"]}


def check_model_allowed(model):
    """Reject anything but a specific OpenRouter model release from an allowed family."""
    if not isinstance(model, str) or "/" not in model:
        raise RunError(f"Model {model!r} is not an OpenRouter slug of the form vendor/model")
    family = model.split("/", 1)[0].lower()
    if family in {"anthropic", "google"} or any(name in model.lower() for name in ("claude", "gemini")):
        raise RunError(f"{model} is from an excluded model family (Anthropic or Google)")
    if ":" in model or model.endswith("/auto"):
        raise RunError(f"{model} is a routing alias or variant; name a specific model release instead")


def endpoints_filename(model):
    return model.replace("/", "_") + "-endpoints.json"


def find_endpoint(endpoints, provider, source):
    """The one endpoint in an OpenRouter endpoint listing whose tag is provider."""
    matches = [endpoint for endpoint in endpoints if endpoint["tag"] == provider]
    if len(matches) != 1:
        raise RunError(f"Expected one endpoint tagged {provider!r} in {source}, found {len(matches)}")
    return matches[0]


def check_endpoint_supports(cfg, endpoint, model_info):
    """Fail unless the endpoint accepts every request field and setting that cfg sends to it."""
    missing = REQUIRED_PARAMETERS - set(endpoint["supported_parameters"])
    if missing:
        raise RunError(
            f"{cfg['model']} endpoint {endpoint['tag']} does not support {sorted(missing)}; choose another provider in configs/models.yaml"
        )
    output_limit = endpoint.get("max_completion_tokens") or endpoint["context_length"]
    if cfg["max_tokens"] > output_limit:
        raise RunError(
            f"{cfg['model']} max_tokens is {cfg['max_tokens']}, but endpoint {endpoint['tag']} "
            f"allows at most {output_limit}; lower it in configs/models.yaml"
        )
    efforts = model_info.get("reasoning", {}).get("supported_efforts", [])
    effort = cfg["reasoning"].get("effort")
    if effort is not None and efforts and effort not in efforts:
        raise RunError(f"{cfg['model']} does not support reasoning effort {effort!r}; set one of {efforts} in configs/models.yaml")


def endpoint_for(cfg):
    """The saved endpoint and catalog entry for a role's model and provider, checked against cfg."""
    path = SNAPSHOTS / endpoints_filename(cfg["model"])
    endpoint = find_endpoint(read_json(path)["data"]["endpoints"], cfg["provider"], path.name)
    model_info = next((m for m in read_json(MODEL_CATALOG)["data"] if m["id"] == cfg["model"]), None)
    if model_info is None:
        raise RunError(f"{cfg['model']} is missing from {MODEL_CATALOG.name}; refresh configs/snapshots")
    check_endpoint_supports(cfg, endpoint, model_info)
    return endpoint, model_info


def highest_prices(endpoint):
    """Highest USD price per unit across the endpoint's base pricing and its conditional overrides."""
    base = endpoint["pricing"]
    tiers = [base, *base.get("overrides", [])]
    prices = {}
    for key in PRICE_KEYS:
        # An override that leaves out a price keeps the base price.
        prices[key] = max(float(tier.get(key, base.get(key, 0))) for tier in tiers)
    return prices


def allowed_prices(endpoint):
    """The highest prices the run accepts: the highest listed price plus PRICING_HEADROOM."""
    return {key: price * PRICING_HEADROOM for key, price in highest_prices(endpoint).items()}


def routing_fields(cfg):
    """Request fields that pin one role's model, provider and decoding settings."""
    return {
        "model": cfg["model"],
        "stream": False,
        "plugins": [],
        "transforms": [],
        "temperature": cfg["temperature"],
        "reasoning": cfg["reasoning"],
        "max_tokens": cfg["max_tokens"],
        "provider": {
            "only": [cfg["provider"]],
            "order": [cfg["provider"]],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }


def json_schema_format(schema):
    """The response_format asking for JSON that matches schema, or None when there is no schema."""
    if schema is None:
        return None
    return {"type": "json_schema", "json_schema": {"name": "xar_output", "strict": True, "schema": schema}}


def backoff_seconds(attempt, retry_after=None):
    """Seconds to wait after failed send `attempt`: the schedule, or a longer Retry-After, capped."""
    wait = BACKOFF_SECONDS[attempt]
    if retry_after is not None and retry_after.strip().isdigit():
        wait = max(wait, min(int(retry_after), MAX_RETRY_AFTER_SECONDS))
    return wait


def max_request_cost(payload, endpoint):
    """Conservative cost allowance using UTF-8 bytes as a token upper estimate.

    Tokenization differs by model. This is accounting, not a context rejection: the provider
    enforces its context window, with transforms disabled and the full input preserved.
    """
    input_tokens = math.ceil(1.25 * len(canonical(payload).encode())) + 1024
    price = allowed_prices(endpoint)
    return input_tokens * price["prompt"] + payload["max_tokens"] * price["completion"] + price["request"]


def billed_cost(response):
    usage = response.get("usage")
    cost = usage.get("cost") if isinstance(usage, dict) else None
    if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
        return cost
    return None


class OpenRouter:
    """Send each fresh-run request, keeping every attempt and any uncertain billed amount."""

    def __init__(self, output, roles, seed, client=None):
        load_dotenv(ROOT / ".env")
        self.key = os.getenv("OPENROUTER_API_KEY")
        if not self.key and client is None:
            raise RunError("OPENROUTER_API_KEY is not set; add it to .env")
        self.output, self.roles, self.seed = Path(output), roles, seed
        self.client = client or httpx.Client(timeout=httpx.Timeout(600, connect=30))
        resolved = {role: endpoint_for(cfg) for role, cfg in roles.items()}
        self.endpoint = {role: value[0] for role, value in resolved.items()}
        self.model_info = {role: value[1] for role, value in resolved.items()}
        self.dispatch_stopped = threading.Event()

    def costs(self):
        total, requests, unresolved, upper = 0.0, 0, 0, 0.0
        for path in self.output.glob("requests/*/attempt_*.json"):
            sent = read_json(path)
            cost = billed_cost(sent.get("response", {}))
            if cost is not None:
                total += cost
                requests += 1
            elif sent["status"] in {"uncertain", "uncertain_cost"}:
                unresolved += 1
                upper += sent["upper_usd"]
        return {
            "actual_complete_usd": total,
            "requests": requests,
            "unresolved": unresolved,
            "unresolved_upper_usd": upper,
        }

    def payload(self, role, system, data, schema):
        result = {
            **routing_fields(self.roles[role]),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": canonical(data)}],
        }
        if schema is not None:
            result["response_format"] = json_schema_format(schema)
        return result

    def _get(self, path):
        response = self.client.get(API_BASE + path)
        response.raise_for_status()
        return response.json()

    def preflight(self):
        """Check the live OpenRouter catalog against the saved snapshots before any paid call.

        Fails if a pinned model, provider, limit, price or quantization changed. Saves what it
        fetched, plus the observed prices, under output/preflight/<time> and returns that directory.
        """
        directory = self.output / "preflight" / now().replace(":", "-")
        catalog = self._get("/models")
        write_json(directory / "models.json", catalog)
        live_models = {m["id"]: m for m in catalog["data"]}
        live_endpoints = {}
        observed_prices = {}
        for role, cfg in self.roles.items():
            model = cfg["model"]
            pinned = self.endpoint[role]
            if model not in live_models:
                raise RunError(f"{role} model {model} is no longer in the OpenRouter catalog")
            live_slug = live_models[model]["canonical_slug"]
            pinned_slug = self.model_info[role]["canonical_slug"]
            if live_slug != pinned_slug:
                raise RunError(
                    f"{role} model {model} now resolves to {live_slug}, not the pinned {pinned_slug}; "
                    "refresh configs/snapshots and start a new run"
                )
            if model not in live_endpoints:
                live_endpoints[model] = self._get(f"/models/{model}/endpoints")
                write_json(directory / endpoints_filename(model), live_endpoints[model])
            current = find_endpoint(live_endpoints[model]["data"]["endpoints"], cfg["provider"], f"the live {model} listing")
            check_endpoint_supports(cfg, current, live_models[model])
            if current["context_length"] < pinned["context_length"]:
                raise RunError(
                    f"{role} endpoint {current['tag']} context shrank from {pinned['context_length']} "
                    f"to {current['context_length']} tokens; refresh configs/snapshots and start a new run"
                )
            prices = highest_prices(current)
            limits = allowed_prices(pinned)
            for key in PRICE_KEYS:
                if prices[key] > limits[key]:
                    raise RunError(
                        f"{role} endpoint {current['tag']} {key} price {prices[key]} is above the "
                        f"allowed {limits[key]} (snapshot price plus {PRICING_HEADROOM - 1:.0%}); "
                        "refresh configs/snapshots and start a new run"
                    )
            if current.get("quantization") != pinned.get("quantization"):
                raise RunError(
                    f"{role} endpoint {current['tag']} quantization changed from "
                    f"{pinned.get('quantization')} to {current.get('quantization')}; start a new run"
                )
            observed_prices[role] = prices
        write_json(
            directory / "checks.json",
            {"at": now(), "roles": self.roles, "observed_prices": observed_prices},
        )
        return directory

    def call(self, role, system, data, schema, identity):
        try:
            return self._call(role, system, data, schema, identity)
        except BaseException:
            self.dispatch_stopped.set()
            raise

    def _call(self, role, system, data, schema, identity):
        payload = self.payload(role, system, data, schema)
        key = digest({"payload": payload, "identity": identity, "schema_version": SCHEMA_VERSION})
        directory = self.output / "requests" / key
        directory.mkdir(parents=True, exist_ok=False)
        write_json(
            directory / "request.json",
            {
                "payload": payload,
                "identity": identity,
                "role": role,
                "key": key,
                "schema_version": SCHEMA_VERSION,
            },
        )
        upper = max_request_cost(payload, self.endpoint[role])
        headers = {"X-OpenRouter-Title": "Independent XAR reproduction"}
        if self.key:
            headers["Authorization"] = "Bearer " + self.key
        for attempt in range(MAX_SENDS):
            if self.dispatch_stopped.is_set():
                raise Skipped("Another request failed; no further requests will start")
            path = directory / f"attempt_{attempt}.json"
            record = {"status": "uncertain", "sent_at": now(), "upper_usd": upper}
            write_json(path, record)
            started, retry_after = time.monotonic(), None
            try:
                response = self.client.post(API_BASE + "/chat/completions", json=payload, headers=headers)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as error:
                record.update(status="connect_error", error=str(error))
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
            else:
                try:
                    raw = response.json()
                except ValueError:
                    raw = {"raw_body": response.text}
                if not isinstance(raw, dict):
                    raw = {"raw_body": raw}
                choices = raw.get("choices")
                valid_choice = (
                    isinstance(choices, list)
                    and bool(choices)
                    and isinstance(choices[0], dict)
                    and isinstance(choices[0].get("message"), dict)
                    and isinstance(choices[0]["message"].get("content", ""), (str, type(None)))
                )
                success = response.status_code == 200 and valid_choice and not raw.get("error")
                record.update(
                    response=raw,
                    http_status=response.status_code,
                    status="success" if success else "http_error",
                )
                if response.status_code == 200 and not success:
                    record["status"] = "uncertain_cost"
                record["duration_seconds"] = time.monotonic() - started
                write_json(path, record)
                if success:
                    return self._accept(raw, role, path)
                if response.status_code not in RETRYABLE_STATUS:
                    raise RunError(f"OpenRouter returned an unusable HTTP {response.status_code} reply; see {path}")
                retry_after = response.headers.get("Retry-After")
            record["duration_seconds"] = time.monotonic() - started
            write_json(path, record)
            if attempt < MAX_SENDS - 1:
                wait = backoff_seconds(attempt, retry_after)
                print(f"Retryable failure for {role}; waiting {wait} s (see {path})", flush=True)
                time.sleep(wait)
        raise RunError(f"{role} request failed {MAX_SENDS} times; see {directory}")

    def _accept(self, raw, role, attempt_file):
        usage = raw.get("usage") or {}
        if billed_cost(raw) is None:
            write_json(attempt_file, {**read_json(attempt_file), "status": "uncertain_cost"})
            raise UncertainSend(f"No valid usage.cost for {role}; see {attempt_file}")
        model = raw.get("model")
        if model not in {self.model_info[role]["id"], self.model_info[role]["canonical_slug"]}:
            raise RunError(f"Unexpected {role} model {model}; see {attempt_file}")
        provider, endpoint = raw.get("provider"), self.endpoint[role]
        if provider and provider not in {endpoint["provider_name"], endpoint["tag"]}:
            raise RunError(f"Unexpected {role} provider {provider}; see {attempt_file}")
        choice = raw["choices"][0]
        return {
            "content": choice["message"].get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "request_key": attempt_file.parent.name,
            "response_id": raw.get("id"),
            "usage": usage,
            "model": model,
            "provider": provider,
            "raw_response": str(attempt_file),
        }

    def structured(self, role, system, data, schema, identity, validator=None, repair=True):
        """Ask role for JSON matching schema and check it with validator.

        If the reply is incomplete, not valid JSON or fails a check, and repair is on, the request is
        sent once more with the error appended to the system prompt. Returns the status ("valid" or
        "missing"), the parsed value and every attempt.
        """
        attempts = []
        for format_attempt in range(2 if repair else 1):
            instructions = system
            if format_attempt:
                instructions += FORMAT_REPAIR
                instructions += "Keep the substantive task inputs unchanged. Previous validation error: " + attempts[-1]["error"]
            response = self.call(role, instructions, data, schema, {"task": identity, "format_attempt": format_attempt})
            try:
                if response["finish_reason"] != "stop":
                    raise InvalidOutput(f"Incomplete output: {response['finish_reason']}")
                value = json.loads(response["content"])
                jsonschema.validate(value, schema)
                if validator:
                    validator(value)
                attempts.append({"response": response, "status": "valid"})
                return {"status": "valid", "value": value, "attempts": attempts}
            except (json.JSONDecodeError, jsonschema.ValidationError, InvalidOutput) as e:
                attempts.append({"response": response, "status": "invalid", "error": str(e)[:MAX_REPAIR_ERROR_CHARS]})
        return {"status": "missing", "value": None, "attempts": attempts}


# Rubrics, anonymous grading, and training-only prompt optimization.
TEXT = {"type": "string", "minLength": 1}
SCORE = {"type": "number", "minimum": 0, "maximum": 10}


def object_schema(properties):
    """A JSON object with exactly these properties, all required."""
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def criteria_list(item):
    """A rubric has 4 to 8 criteria, and a grade has one score per criterion."""
    return {"type": "array", "minItems": 4, "maxItems": 8, "items": item}


CRITERION_SCHEMA = object_schema({"id": TEXT, "description": TEXT, "low": TEXT, "middle": TEXT, "high": TEXT})
CRITERION_SCORE_SCHEMA = object_schema({"id": TEXT, "score": SCORE, "evidence": TEXT})
RUBRIC_SCHEMA = object_schema({"criteria": criteria_list(CRITERION_SCHEMA)})
GRADE_SCHEMA = object_schema({"scores": criteria_list(CRITERION_SCORE_SCHEMA)})
PROPOSAL_SCHEMA = object_schema({"prompt": TEXT, "rationale": TEXT})
MAX_RUBRIC_WORDS = 1000
QUOTED_TEXT = re.compile(r'["“]([^"”]+)["”]')
WRITER_ATTEMPTS = 3
PROPOSAL_ATTEMPTS = 2


def validate_rubric(rubric):
    """Check a generated rubric: schema, unique criterion IDs and total length.

    Raises with a short message; structured() sends that message back in its format repair."""
    jsonschema.validate(rubric, RUBRIC_SCHEMA)
    ids = [c["id"] for c in rubric["criteria"]]
    if len(ids) != len(set(ids)):
        raise InvalidOutput("Duplicate rubric criterion IDs")
    total = sum(words(text) for criterion in rubric["criteria"] for text in criterion.values())
    if total > MAX_RUBRIC_WORDS:
        raise InvalidOutput("Rubric exceeds 1000 words")


def validate_grade(grade, rubric, supplied_text):
    """Check a judge's grade against its rubric and return the mean criterion score.

    Every criterion must be scored exactly once, and anything the judge quotes as evidence
    must occur in supplied_text (the graded section plus the visible paper)."""
    jsonschema.validate(grade, GRADE_SCHEMA)
    ids = [s["id"] for s in grade["scores"]]
    if len(ids) != len(set(ids)) or set(ids) != {c["id"] for c in rubric["criteria"]}:
        raise InvalidOutput("Grade criterion coverage mismatch")
    # json.loads accepts NaN, and NaN passes the schema's minimum and maximum.
    if not all(math.isfinite(s["score"]) for s in grade["scores"]):
        raise InvalidOutput("Grade score must be a finite number")
    for score in grade["scores"]:
        for quoted in QUOTED_TEXT.findall(score["evidence"]):
            if normalize(quoted).casefold() not in normalize(supplied_text).casefold():
                raise InvalidOutput("Evidence quote does not occur in supplied text")
    return grade_total(grade)


def grade_total(grade):
    """The mean criterion score of a grade."""
    return statistics.mean(s["score"] for s in grade["scores"])


def length_window(target_words):
    """The word counts, low to high, that count as meeting a section's length target."""
    return 0.85 * target_words, 1.15 * target_words


def length_revision_note(target_words):
    """The instruction sent with a writer retry after a section missed its length target."""
    low, high = length_window(target_words)
    return f"Revise only to fit {math.ceil(low)}–{math.floor(high)} words. Preserve claims."


def writer_candidates(api, examples, output, concurrency=1):
    """Write one model section per example, `concurrency` examples at a time.

    Each section gets up to WRITER_ATTEMPTS drafts: a draft that is cut off or outside the length
    window goes back to the writer with a revision note, and the last draft is kept either way.
    Each section is saved as it finishes. Returns the records keyed by example ID."""
    config_hash = digest({"writer": api.roles["writer"], "prompt": prompt("writer"), "schema_version": SCHEMA_VERSION})
    stopped = api.dispatch_stopped

    def write_section(example):
        eid = example["example_id"]
        path = Path(output) / "generations" / (eid + ".json")
        low, high = length_window(example["target_words"])
        attempts = []
        for attempt in range(WRITER_ATTEMPTS):
            data = task_data(example)
            if attempt:
                data.update(
                    previous_section=attempts[-1]["text"],
                    length_revision=length_revision_note(example["target_words"]),
                )
            if stopped.is_set():
                raise Skipped(f"Stopped writing {eid} because another section failed")
            identity = {"writer": config_hash, "example": eid, "attempt": attempt}
            response = api.call("writer", prompt("writer"), data, None, identity)
            text = response["content"].strip()
            count = words(text)
            compliant = low <= count <= high
            attempts.append(
                {
                    "attempt_index": attempt,
                    "text": text,
                    "text_hash": digest(text),
                    "words": count,
                    "length_compliant": compliant,
                    "response": response,
                }
            )
            if compliant and response["finish_reason"] == "stop":
                break
        accepted = attempts[-1]
        record = {
            "example_id": eid,
            "context_hash": example["context_hash"],
            "writer_configuration_hash": config_hash,
            "writer_configuration": api.roles["writer"],
            "attempts": attempts,
            "accepted_attempt": accepted["attempt_index"],
            "text": accepted["text"],
            "text_hash": accepted["text_hash"],
            "words": accepted["words"],
            "length_compliant": accepted["length_compliant"],
            "complete": accepted["response"]["finish_reason"] == "stop",
            "length_ratio": accepted["words"] / example["target_words"],
            "contamination": contamination(accepted["text"], example["reference"]),
        }
        write_json(path, record)
        return record

    records = bounded_map(write_section, examples, concurrency, stopped)
    # bounded_map returns results in input order, so candidates.json keeps the dataset order.
    candidates = {record["example_id"]: record for record in records}
    artifact = {
        "writer_configuration_hash": config_hash,
        "writer_configuration": api.roles["writer"],
        "candidates": candidates,
    }
    write_json(Path(output) / "generations/candidates.json", artifact)
    return candidates


def generate_rubric(api, example, meta_prompt, identity, output):
    result = api.structured(
        "rubric",
        prompt("rubric_wrapper"),
        {**task_data(example), "meta_prompt": meta_prompt},
        RUBRIC_SCHEMA,
        identity,
        validate_rubric,
    )
    record = {
        "example_id": example["example_id"],
        "context_hash": example["context_hash"],
        "meta_prompt_hash": digest(meta_prompt),
        "generator_configuration": api.roles["rubric"],
        "identity": identity,
        **result,
    }
    record["rubric_hash"] = digest(result["value"]) if result["value"] else None
    write_json(output, record)
    return record


def grade_candidate(api, example, rubric_record, text, identity, output):
    """Grade one anonymous section against a rubric."""
    if rubric_record["status"] != "valid":
        result = {"status": "missing", "value": None, "attempts": [], "reason": "invalid_rubric"}
    else:
        rubric = rubric_record["value"]
        result = api.structured(
            "judge",
            prompt("judge"),
            {**task_data(example), "rubric": rubric, "candidate": text},
            GRADE_SCHEMA,
            identity,
            lambda value: validate_grade(value, rubric, text + " " + example["context"]),
        )
    record = {
        "example_id": example["example_id"],
        "candidate_hash": digest(text),
        "rubric_hash": rubric_record["rubric_hash"],
        "judge_configuration": api.roles["judge"],
        "identity": identity,
        **result,
    }
    record["total"] = grade_total(result["value"]) if result["value"] else None
    write_json(output, record)
    return record


def evaluate_checkpoint(api, examples, candidates, meta_prompt, checkpoint, output, concurrency):
    """Score one meta prompt on examples: generate a rubric per example, then grade both sections.

    The judge grades the human and the model section separately, in a seeded random order, and
    never learns which is which. Returns one row per example with the human-minus-model gap
    (None if the rubric or a grade is missing), and saves the rows next to the grades."""
    root = Path(output)

    def evaluate(example):
        eid = example["example_id"]
        candidate = candidates[eid]
        # The label is part of the saved paths and request identities, "main/" prefix included.
        label = f"main/{checkpoint}/{example['split']}/{eid}"
        rubric_path = root / "rubrics" / label / "rubric.json"
        rubric = generate_rubric(api, example, meta_prompt, {"rubric": label}, rubric_path)
        origins = ["human", "model"]
        order_seed = int(digest({"seed": api.seed, "label": label})[:16], 16)
        random.Random(order_seed).shuffle(origins)
        grade_paths = {origin: root / "scores" / label / (origin + ".json") for origin in origins}
        totals = {}
        for slot, origin in enumerate(origins):
            text = example["reference"] if origin == "human" else candidate["text"]
            # The origin appears only in the file path, never in the request or its identity.
            identity = {"grade": label, "slot": slot}
            grade = grade_candidate(api, example, rubric, text, identity, grade_paths[origin])
            totals[origin] = grade["total"]
        human, model = totals["human"], totals["model"]
        return {
            "example_id": eid,
            "paper_id": example["paper_id"],
            "section_type": example["section_type"],
            "split": example["split"],
            "checkpoint": checkpoint,
            "human": human,
            "model": model,
            "gap": human - model if human is not None and model is not None else None,
            "length_ratio": candidate["length_ratio"],
            "length_compliant": candidate["length_compliant"],
            "contamination_flagged": candidate["contamination"]["flagged"],
            "rubric_path": str(rubric_path),
            "grade_paths": {origin: str(path) for origin, path in grade_paths.items()},
        }

    rows = bounded_map(evaluate, examples, concurrency, api.dispatch_stopped)
    write_json(root / "scores" / "main" / str(checkpoint) / f"{examples[0]['split']}_rows.json", rows)
    return rows


PROPOSAL_RED_FLAGS = {
    # Tells the rubric to favour the human section or to penalize the model's.
    "origin_preference": r"(?:prefer|reward|favor|favour|boost)\s+(?:the\s+)?(?:human|expert|original)|"
    r"(?:penaliz|penalis|punish|downscore)\w*\s+(?:the\s+)?(?:model|ai|generated)",
    # Tells the rubric to work out who wrote the section.
    "origin_detection": r"(?:detect|guess|infer|identify|determine)\s+(?:the\s+)?(?:authorship|origin|provenance)|"
    r"(?:human.written|ai.generated|model.generated)\s+(?:tells|markers|signals)",
    # Overrides the wrapper, or weights or rescales the 0-10 criterion scores.
    "wrapper_override": r"(?:ignore|override|replace)\s+(?:the\s+)?(?:wrapper|system|grading|schema)|"
    r"(?:weighted\s+(?:mean|average)|unequal\s+weight|score\s+(?:from\s+)?0\s*(?:to|[-–])\s*100)",
    # Model-side markup leaked into the prompt text, or a prompt cut off inside it. Seen from
    # MiMo-V2.6-Flash: a prompt ending "...<tool_call><function=json>{" inside valid JSON.
    "leaked_markup": r"<tool_call|<function=|</function|<\|im_|\{\s*$",
}
SPAN = 12  # a run of this many words copied from a training paper counts as leakage
MIN_IDENTIFIER_LENGTH = 5  # shorter paper IDs, titles or author names match by chance


def audit_proposal(text, examples, initial_prompt, max_words):
    """Statically check an optimizer proposal before it can become the next meta prompt.

    Rejects a prompt that is longer than max_words, matches PROPOSAL_RED_FLAGS (including leaked
    tool-call markup), names a training paper or its authors, or copies SPAN consecutive words
    from a training paper (unless the initial prompt already contains them)."""
    word_count = words(text)
    reasons, flags = [], []
    if word_count > max_words:
        reasons.append("meta_prompt_word_bound")
    lower = text.casefold()
    for name, pattern in PROPOSAL_RED_FLAGS.items():
        if re.search(pattern, lower):
            reasons.append(name)
    proposal_words = lower.split()
    initial_lower = initial_prompt.casefold()
    for example in examples:
        paper = example["paper_id"]
        metadata = example["provenance"]
        for identifier in [paper, metadata["title"], *metadata.get("authors", [])]:
            if len(identifier) >= MIN_IDENTIFIER_LENGTH and identifier.casefold() in lower:
                flags.append({"type": "training_identifier", "paper_id": paper, "match": identifier})
        for field in ("reference", "context"):
            source_words = example[field].casefold().split()
            source_spans = {tuple(source_words[i : i + SPAN]) for i in range(len(source_words) - SPAN + 1)}
            for i in range(len(proposal_words) - SPAN + 1):
                span = proposal_words[i : i + SPAN]
                span_text = " ".join(span)
                if tuple(span) in source_spans and span_text not in initial_lower:
                    flags.append({"type": "copied_training_span", "paper_id": paper, "match": span_text})
                    break
    if flags:
        reasons.append("training_leakage")
    return {
        "accepted": not reasons,
        "reasons": sorted(reasons),
        "flags": flags,
        "words": word_count,
        "policy": "fixed_static_scan_v1",
        "limitation": "Static scan cannot prove semantic absence of provenance heuristics",
    }


def propose_prompt(api, current, feedback, examples, initial, iteration, *, output, max_words):
    """Ask the optimizer for the next meta prompt, given the training feedback on the current one.

    A proposal that is malformed or fails audit_proposal is sent back once with the reasons. If
    the repair fails too, the update is used up and the current prompt carries over unchanged.
    Saves the feedback and proposal under feedback/iter_XX/ and returns the next prompt."""
    directory = Path(output) / "feedback" / f"iter_{iteration:02d}"
    write_json(directory / "training.json", feedback)
    data = {"feedback": feedback, "max_meta_prompt_words": max_words}
    attempts = []
    for attempt in range(PROPOSAL_ATTEMPTS):
        instruction = prompt("optimizer")
        if attempt:
            rejected = attempts[-1]
            instruction += "\nBOUNDED REPAIR: Fix these proposal violations: " + ", ".join(rejected["audit"]["reasons"])
            # Show the rejected proposal, or the raw reply if it was not valid JSON.
            previous = rejected["value"] or rejected["attempts"][-1]["response"]["content"]
            data = {**data, "previous_proposal": previous}
        result = api.structured(
            "optimizer",
            instruction,
            data,
            PROPOSAL_SCHEMA,
            {"proposal": iteration, "bounded_attempt": attempt},
            repair=False,
        )
        if result["value"]:
            audit = audit_proposal(result["value"]["prompt"], examples, initial, max_words)
        else:
            audit = {"accepted": False, "reasons": ["invalid_format"], "flags": []}
        attempts.append({"attempt": attempt, "audit": audit, **result})
        if audit["accepted"]:
            break
    accepted = attempts[-1]["audit"]["accepted"]
    proposal = attempts[-1]["value"]["prompt"] if accepted else current
    record = {
        "iteration": iteration,
        "parent_prompt_hash": digest(current),
        "feedback_hash": digest(feedback),
        "optimizer_configuration": api.roles["optimizer"],
        "attempts": attempts,
        "accepted": accepted,
        "update_consumed": True,
        "prompt": proposal,
        "prompt_hash": digest(proposal),
    }
    write_json(directory / "proposal.json", record)
    return proposal


def build_feedback(examples, candidates, rows, current_prompt, failure_count):
    """The active feedback policy: nonpositive training gaps, without full papers.

    Keep the recorded all_training_gaps field's existing meaning: all nonpositive gaps, before
    limiting detailed failure examples. Validation and confirmation never enter this payload.
    """
    if any(item["split"] != "train" for item in [*examples, *rows]):
        raise RunError("Optimizer feedback may only use training sections")
    if any(row["gap"] is None for row in rows):
        raise RunError("The optimizer needs every training section graded")
    lookup = {example["example_id"]: example for example in examples}
    by_gap = sorted((row for row in rows if row["gap"] <= 0), key=lambda row: (row["gap"], row["example_id"]))
    failures = []
    for row in by_gap[:failure_count]:
        example = lookup[row["example_id"]]
        failures.append(
            {
                "example_id": example["example_id"],
                "section_type": example["section_type"],
                "target_words": example["target_words"],
                "human_candidate": example["reference"],
                "model_candidate": candidates[example["example_id"]]["text"],
                "rubric": read_json(row["rubric_path"])["value"],
                "grades": {origin: read_json(path)["value"] for origin, path in row["grade_paths"].items()},
                "gap": row["gap"],
            }
        )
    return {
        "current_prompt": current_prompt,
        "summary": summarize(rows),
        "failures": failures,
        "example_ids_used_for_aggregate": sorted(lookup),
        "selection": FAILING_FEEDBACK,
        "all_training_gaps": [{"example_id": row["example_id"], "section_type": row["section_type"], "gap": row["gap"]} for row in by_gap],
    }


# Preserve whole-paper bootstrap statistics, including the fields sent to the optimizer.
BOOTSTRAP_REPLICATES = 2000
INTERVAL = (0.025, 0.975)


def bootstrap(rows, seed=0):
    """Percentile bootstrap of the mean gap that resamples whole papers.

    Sections from one paper are not independent, so each replicate draws papers with
    replacement and takes all of a drawn paper's gaps. Rows without a gap are ignored;
    returns None if no row has one."""
    gaps_by_paper = {}
    for row in rows:
        if row.get("gap") is not None:
            gaps_by_paper.setdefault(row["paper_id"], []).append(row["gap"])
    if not gaps_by_paper:
        return None
    rng = random.Random(seed)
    papers = sorted(gaps_by_paper)
    means = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = rng.choices(papers, k=len(papers))
        means.append(statistics.mean(gap for paper in sample for gap in gaps_by_paper[paper]))
    means.sort()
    low, high = INTERVAL
    return {
        "method": "paired_whole_paper_percentile_bootstrap",
        "paper_clusters": len(papers),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": seed,
        "estimate": statistics.mean(gap for gaps in gaps_by_paper.values() for gap in gaps),
        "bootstrap_median": means[BOOTSTRAP_REPLICATES // 2],
        "low": means[int(low * BOOTSTRAP_REPLICATES)],
        "high": means[int(high * BOOTSTRAP_REPLICATES)],
    }


def section_summary(graded, section):
    gaps = [row["gap"] for row in graded if row["section_type"] == section]
    return {"coverage": len(gaps), "gap": statistics.mean(gaps) if gaps else None}


def summarize(rows, seed=0):
    """Summarize one checkpoint's rows: mean human, model and gap scores over the sections with
    both grades, per-section gaps, and paper-bootstrap intervals for all sections, the
    length-compliant ones and the ones not flagged for copying the reference."""
    graded = [row for row in rows if row["gap"] is not None]

    def mean(key):
        return statistics.mean(row[key] for row in graded) if graded else None

    return {
        "examples": len(rows),
        "paired_coverage": len(graded),
        "complete": len(graded) == len(rows),
        "human": mean("human"),
        "model": mean("model"),
        "gap": mean("gap"),
        "positive_gap_fraction": sum(row["gap"] > 0 for row in graded) / len(graded) if graded else None,
        "ties": sum(row["gap"] == 0 for row in graded),
        "paper_interval": bootstrap(graded, seed),
        "length_compliant": sum(row["length_compliant"] for row in rows),
        "contamination_flagged": sum(row["contamination_flagged"] for row in rows),
        "sections": {section: section_summary(graded, section) for section in SECTIONS},
        "compliant_sensitivity": bootstrap([row for row in graded if row["length_compliant"]], seed),
        "unflagged_sensitivity": bootstrap([row for row in graded if not row["contamination_flagged"]], seed),
    }


def paired_improvement(initial, selected, seed=0):
    """Per-section change in gap from the initial to the selected checkpoint, with a
    paper-bootstrap interval. Sections missing a gap at either checkpoint are left out."""
    initial_by_id = {row["example_id"]: row for row in initial}
    changes = []
    for row in selected:
        if row["gap"] is None:
            continue
        before = initial_by_id[row["example_id"]]["gap"]
        if before is not None:
            changes.append({"paper_id": row["paper_id"], "gap": row["gap"] - before})
    return {
        "paired_coverage": len(changes),
        "mean": statistics.mean(change["gap"] for change in changes) if changes else None,
        "interval": bootstrap(changes, seed),
    }


@dataclass
class RunSettings:
    output_dir: str
    split: str
    seed: int
    iterations: int
    max_meta_prompt_words: int
    failure_examples: int
    concurrency: int
    dataset: str = "data/examples.jsonl"
    splits: str = "data/splits.json"
    dry_run: bool = False


def estimate(settings, roles, examples):
    """Offline rough cost estimate; retries and model tokenization make actual costs differ."""
    counts = {
        "writer": len(examples),
        "rubric": len(examples) * (settings.iterations + 1),
        "judge": 2 * len(examples) * (settings.iterations + 1),
        "optimizer": settings.iterations,
    }
    typical_tokens = statistics.mean(len(example["context"].encode()) for example in examples) / 3.5 + 1500
    per_role = {}
    for role, count in counts.items():
        endpoint, _ = endpoint_for(roles[role])
        prices = highest_prices(endpoint)
        per_role[role] = {
            "requests": count,
            "typical_usd": count
            * (typical_tokens * prices["prompt"] + min(roles[role]["max_tokens"], 5000) * prices["completion"] + prices["request"]),
        }
    result = {
        "examples": len(examples),
        "roles": roles,
        "per_role": per_role,
        "estimated_usd_with_retry_reserve": 1.25 * sum(value["typical_usd"] for value in per_role.values()),
        "note": "Rough estimate: 3.5 UTF-8 bytes per prompt token, 5,000 output tokens, 25% retry reserve. "
        "Optimizer feedback and actual tokenization vary. This is not a spending cap.",
    }
    print(json.dumps(result, indent=2))
    return result


def write_report(output, summaries, rows, selected, costs, seed):
    table = []
    for index, (train, validation) in enumerate(summaries):
        table.append(
            {
                "iteration": index,
                "train_human": train["human"],
                "train_model": train["model"],
                "train_gap": train["gap"],
                "val_human": validation["human"],
                "val_model": validation["model"],
                "val_gap": validation["gap"],
                "val_coverage": validation["paired_coverage"],
                "selected_by_train": index == selected,
            }
        )
    improvement = paired_improvement(rows[0], rows[selected], seed)
    result = {
        "selected_iteration": selected,
        "checkpoints": table,
        "summaries": [{"train": train, "validation": validation} for train, validation in summaries],
        "paired_improvement": improvement,
        "costs": costs,
    }
    write_table(output / "checkpoints.csv", table)
    write_json(output / "summary.json", result)
    lines = [
        "# Reproduction results",
        "",
        f"Training selected P{selected} before validation began.",
        "",
        "The gap is the author's score minus the model's score. Each criterion has equal weight.",
        "",
        "| Prompt | Training gap | Validation gap | Validation coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in table:
        gap = "missing" if row["val_gap"] is None else f"{row['val_gap']:+.3f}"
        lines.append(f"| P{row['iteration']} | {row['train_gap']:+.3f} | {gap} | {row['val_coverage']} |")
    interval = improvement["interval"]
    if interval:
        lines.extend(
            [
                "",
                (
                    f"The selected prompt changed the paired validation gap by {improvement['mean']:+.3f} "
                    f"(95% whole-paper bootstrap interval: {interval['low']:+.3f} to {interval['high']:+.3f})."
                ),
            ]
        )
    lines.extend(
        [
            "",
            (
                f"Recorded billed cost: ${costs['actual_complete_usd']:.4f}. "
                f"Sends with unresolved costs: {costs['unresolved']} "
                f"(estimated upper allowance: ${costs['unresolved_upper_usd']:.4f})."
            ),
            "",
            (
                "This is an independent reproduction with repository-specific papers and prompts. "
                "The confirmation split remains unused. Bootstrap intervals resample whole papers."
            ),
            "",
            "Full statistics and coverage are in summary.json; every request and response is saved.",
        ]
    )
    (output / "results.md").write_text("\n".join(lines) + "\n")
    return result


def run_xar(settings):
    initial = prompt("rubric_initial")
    if words(initial) > settings.max_meta_prompt_words:
        raise RunError("Initial meta prompt exceeds max_meta_prompt_words")
    examples = run_examples(load_examples(settings.dataset, settings.splits), settings.split)
    train = [example for example in examples if example["split"] == "train"]
    validation = [example for example in examples if example["split"] == "validation"]
    if not train or not validation:
        raise RunError("The run needs both training and validation papers")
    roles = {role: role_config(role) for role in ROLES}
    if settings.dry_run:
        return estimate(settings, roles, examples)
    output = Path(settings.output_dir)
    # Atomic directory creation also prevents two processes from starting in the same directory.
    if output.exists():
        raise RunError(f"Output already exists: {output}; choose a new --output directory")
    api = OpenRouter(output, roles, settings.seed)
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise RunError(f"Output already exists: {output}; choose a new --output directory") from None
    status = {"state": "failed"}
    try:
        files = [
            ROOT / "run.py",
            ROOT / "fetch_data.py",
            ROOT / "uv.lock",
            ROOT / "configs/experiments.yaml",
            ROOT / "configs/models.yaml",
            *sorted((ROOT / "prompts").glob("*.md")),
        ]
        write_json(
            output / "manifest.json",
            {
                "created_at": now(),
                "arguments": asdict(settings),
                "roles": roles,
                "dataset_hash": file_hash(settings.dataset),
                "splits_hash": file_hash(settings.splits),
                "software_hashes": {str(path.relative_to(ROOT)): file_hash(path) for path in files},
                "endpoints": {
                    role: {
                        "endpoint": api.endpoint[role],
                        "canonical_slug": api.model_info[role]["canonical_slug"],
                    }
                    for role in roles
                },
                "initial_meta_prompt_hash": digest(initial),
                "feedback_policy": FAILING_FEEDBACK,
                "context_estimation": "UTF-8 byte approximation for costs; providers enforce context, without truncation",
            },
        )
        api.preflight()
        candidates = writer_candidates(api, examples, output, settings.concurrency)
        prompts, training_summaries, training_rows = [initial], [], None
        for iteration in range(settings.iterations + 1):
            if iteration:
                feedback = build_feedback(train, candidates, training_rows, prompts[-1], settings.failure_examples)
                prompts.append(
                    propose_prompt(
                        api,
                        prompts[-1],
                        feedback,
                        train,
                        initial,
                        iteration,
                        output=output,
                        max_words=settings.max_meta_prompt_words,
                    )
                )
            (output / "prompts").mkdir(exist_ok=True)
            (output / "prompts" / f"iter_{iteration:02d}.md").write_text(prompts[-1])
            training_rows = evaluate_checkpoint(api, train, candidates, prompts[-1], iteration, output, settings.concurrency)
            summary = summarize(training_rows, settings.seed)
            training_summaries.append(summary)
            print(
                f"P{iteration}: training gap {summary['gap']}, coverage {summary['paired_coverage']}/{len(train)}",
                flush=True,
            )
            if not summary["complete"]:
                raise RunError("Training has missing grades; no checkpoint can be selected")
        gaps = [summary["gap"] for summary in training_summaries]
        selected = gaps.index(max(gaps))  # Earliest checkpoint wins a tie.
        write_json(
            output / "freeze.json",
            {
                "selected": selected,
                "training_gaps": gaps,
                "prompt_hashes": [digest(value) for value in prompts],
                "frozen_at": now(),
            },
        )
        print(f"Training selected P{selected}; scoring validation", flush=True)
        validation_rows = [
            evaluate_checkpoint(api, validation, candidates, value, index, output, settings.concurrency) for index, value in enumerate(prompts)
        ]
        validation_summaries = [summarize(rows, settings.seed) for rows in validation_rows]
        result = write_report(
            output,
            list(zip(training_summaries, validation_summaries, strict=True)),
            validation_rows,
            selected,
            api.costs(),
            settings.seed,
        )
        status = {"state": "complete" if all(summary["complete"] for summary in validation_summaries) else "incomplete"}
        return result
    except BaseException as error:
        status["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        write_json(output / "costs.json", api.costs())
        write_json(output / "status.json", status)
        api.client.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", help="smoke test on 2 training papers")
    parser.add_argument("--dry-run", action="store_true", help="estimate cost without network calls or writes")
    parser.add_argument("--output", help="new run directory (default: runs/<configured run name>)")
    parser.add_argument("--seed", type=int, help="grading-order and bootstrap seed, not a model decoding seed")
    parser.add_argument("--concurrency", type=parse_concurrency, help="parallel examples, 1–32")
    parser.add_argument("--dataset", default="data/examples.jsonl")
    parser.add_argument("--splits", default="data/splits.json")
    return parser.parse_args(argv)


def settings_for(options):
    design = load_design()
    if design["feedback_policy"] != FAILING_FEEDBACK:
        raise RunError(f"Only feedback_policy {FAILING_FEEDBACK!r} is supported")
    return RunSettings(
        output_dir=options.output or str(Path("runs") / design["pilot_run" if options.pilot else "research_run"]),
        split="pilot" if options.pilot else "research",
        seed=design["seed"] if options.seed is None else options.seed,
        iterations=design["pilot_iterations" if options.pilot else "iterations"],
        max_meta_prompt_words=design["max_meta_prompt_words"],
        failure_examples=design["failure_examples"],
        concurrency=options.concurrency or design["concurrency"],
        dataset=options.dataset,
        splits=options.splits,
        dry_run=options.dry_run,
    )


def main(argv=None):
    run_xar(settings_for(parse_args(argv)))


if __name__ == "__main__":
    try:
        main()
    except (RunError, httpx.HTTPError) as error:
        raise SystemExit(f"STOP: {error}") from None
