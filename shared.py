"""Shared contracts for the XAR experiments. Scientific stage order lives in entrypoints."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import csv
import datetime as dt
import fcntl
import functools
import hashlib
import itertools
import json
import math
import os
import random
import re
import statistics
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import jsonschema
import yaml
from bs4 import BeautifulSoup
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
API_BASE = "https://openrouter.ai/api/v1"
SECTIONS = ("abstract", "introduction", "related_work", "conclusion")
SCHEMA_VERSION = 1


class ContractError(RuntimeError):
    pass


class BudgetStop(ContractError):
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


def write_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if canonical(read_json(path)) != canonical(value):
            raise ContractError(f"Immutable artifact differs: {path}")
        return
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def prompt(name):
    return (ROOT / "prompts" / f"{name}.md").read_text()


def model_policy(model):
    if not isinstance(model, str) or "/" not in model:
        raise ContractError("Use an explicit OpenRouter model slug")
    family = model.split("/", 1)[0].lower()
    if family in {"anthropic", "google"} or any(s in model.lower() for s in ("claude", "gemini")):
        raise ContractError(f"Excluded model family: {model}")
    if ":" in model or model.endswith("/auto"):
        raise ContractError("Routing aliases and model variants are not pinned releases")


def object_schema(properties, required=None):
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties) if required is None else required,
        "additionalProperties": False,
    }


TEXT = {"type": "string", "minLength": 1}
CRITERION_SCHEMA = object_schema({k: TEXT for k in ("id", "description", "low", "middle", "high")})
RUBRIC_SCHEMA = object_schema(
    {"criteria": {"type": "array", "minItems": 4, "maxItems": 8, "items": CRITERION_SCHEMA}}
)
GRADE_SCHEMA = object_schema(
    {
        "scores": {
            "type": "array",
            "minItems": 4,
            "maxItems": 8,
            "items": object_schema(
                {"id": TEXT, "score": {"type": "number", "minimum": 0, "maximum": 10}, "evidence": TEXT}
            ),
        }
    }
)
PROPOSAL_SCHEMA = object_schema({"prompt": TEXT, "rationale": TEXT})


def validate_rubric(rubric):
    jsonschema.validate(rubric, RUBRIC_SCHEMA)
    ids = [c["id"] for c in rubric["criteria"]]
    if len(ids) != len(set(ids)):
        raise ContractError("Duplicate rubric criterion IDs")
    if words(" ".join(str(v) for c in rubric["criteria"] for v in c.values())) > 1000:
        raise ContractError("Rubric exceeds 1000 words")


def validate_grade(grade, rubric, supplied_text):
    jsonschema.validate(grade, GRADE_SCHEMA)
    ids = [s["id"] for s in grade["scores"]]
    if len(ids) != len(set(ids)) or set(ids) != {c["id"] for c in rubric["criteria"]}:
        raise ContractError("Grade criterion coverage mismatch")
    if any(not math.isfinite(s["score"]) or isinstance(s["score"], bool) for s in grade["scores"]):
        raise ContractError("Grade score must be a finite number")
    for score in grade["scores"]:
        for quoted in re.findall(r'["“]([^"”]+)["”]', score["evidence"]):
            if normalize(quoted).casefold() not in normalize(supplied_text).casefold():
                raise ContractError("Evidence quote does not occur in supplied text")
    return statistics.mean(s["score"] for s in grade["scores"])


def role_config(args, role, model=None, provider=None):
    cfg = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    defaults = cfg["roles"].get(role, cfg["roles"]["judge"])
    model = model or getattr(args, f"{role}_model", None) or defaults["model"]
    model_policy(model)
    saved = cfg["models"].get(model, {})
    provider = (
        provider
        or getattr(args, f"{role}_provider", None)
        or defaults.get("providers", {}).get(model)
        or saved.get("provider")
    )
    if not provider:
        raise ContractError(f"Explicit provider required for unclassified model {model}")
    reasoning = saved.get("reasoning", {"enabled": True})
    effort = getattr(args, f"{role}_reasoning_effort", None)
    mode = getattr(args, f"{role}_reasoning_mode", None)
    if effort:
        reasoning = {"effort": effort}
    if mode:
        reasoning = {"enabled": mode == "enabled"}
    temperature = getattr(args, f"{role}_temperature", None)
    max_tokens = getattr(args, f"{role}_max_output_tokens", None)
    return {
        "model": model,
        "provider": provider,
        "temperature": defaults["temperature"] if temperature is None else temperature,
        "reasoning": reasoning,
        "max_tokens": max_tokens or saved.get("max_tokens", 16384),
    }


def endpoint_for(cfg, snapshot=None):
    path = snapshot or ROOT / "configs/snapshots" / (cfg["model"].replace("/", "_") + "-endpoints.json")
    data = read_json(path)["data"]
    matches = [e for e in data["endpoints"] if e["tag"] == cfg["provider"]]
    if len(matches) != 1:
        raise ContractError(f"Provider must identify exactly one endpoint: {cfg['provider']}")
    endpoint = matches[0]
    for parameter in ("temperature", "reasoning", "max_tokens", "response_format", "structured_outputs"):
        if parameter not in endpoint["supported_parameters"]:
            raise ContractError(f"{cfg['provider']} lacks required {parameter}")
    if cfg["max_tokens"] > (endpoint.get("max_completion_tokens") or endpoint["context_length"]):
        raise ContractError("Output limit exceeds endpoint capability")
    models = read_json(ROOT / "configs/snapshots/openrouter-models-2026-09-29.json")["data"]
    catalog = next((m for m in models if m["id"] == cfg["model"]), None)
    if catalog is None:
        raise ContractError("Model needs a catalog snapshot before execution")
    efforts = catalog.get("reasoning", {}).get("supported_efforts", [])
    if "effort" in cfg["reasoning"] and efforts and cfg["reasoning"]["effort"] not in efforts:
        raise ContractError(f"Unsupported reasoning effort; catalog supports {efforts}")
    return endpoint, catalog


def pricing_upper(endpoint):
    base = endpoint["pricing"]
    return {
        k: max(float(x.get(k, base.get(k, 0))) for x in [base] + base.get("overrides", []))
        for k in ("prompt", "completion", "request")
    }


PRICING_HEADROOM = 1.25


def pricing_bound(endpoint):
    return {key: value * PRICING_HEADROOM for key, value in pricing_upper(endpoint).items()}


@functools.lru_cache(maxsize=4)
def tokenizer(name):
    cfg = read_json(ROOT / "configs/tokenizers.json")[name]
    filename = "tiktoken.model" if name == "kimi" else "tokenizer.json"
    if file_hash(ROOT / "data/tokenizers" / name / filename) != cfg["checksums"][filename]:
        raise ContractError(f"Tokenizer differs from pinned checksum: {name}")
    if name == "kimi":
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        # Pattern copied from the pinned official tokenization_kimi.py; no remote code execution.
        pattern = "|".join(  # noqa: FLY002 -- keep the official pattern's list form
            [
                r"[\p{Han}]+",
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
                r"\p{N}{1,3}",
                r" ?[^\s\p{L}\p{N}]+[\r\n]*",
                r"\s*[\r\n]+",
                r"\s+(?!\S)",
                r"\s+",
            ]
        )
        return tiktoken.Encoding(
            name="kimi-k2.6",
            pat_str=pattern,
            special_tokens={},
            mergeable_ranks=load_tiktoken_bpe(str(ROOT / "data/tokenizers/kimi/tiktoken.model")),
        )
    from tokenizers import Tokenizer

    value = Tokenizer.from_file(str(ROOT / f"data/tokenizers/{name}/tokenizer.json"))
    value.no_truncation()
    value.no_padding()
    return value


def token_count(text, model):
    name = (
        "qwen"
        if model.startswith("qwen/")
        else "kimi"
        if model.startswith("moonshotai/")
        else ("deepseek" if model.startswith("deepseek/") else "glm" if model.startswith("z-ai/") else None)
    )
    if name is None:
        return len(text.encode())
    if name == "kimi":
        return len(tokenizer(name).encode(text, disallowed_special=()))
    return len(tokenizer(name).encode(text, add_special_tokens=False).ids)


def request_upper(payload, endpoint):
    # Count the whole serialized payload, with 25% headroom plus chat-template overhead.
    # Byte bound remains the fallback for unclassified model tokenizers.
    tokens = math.ceil(1.25 * token_count(canonical(payload), payload["model"])) + 1024
    if tokens + payload["max_tokens"] > endpoint["context_length"]:
        raise ContractError("Conservative context bound exceeded; papers cannot be truncated")
    if endpoint.get("max_prompt_tokens") and tokens > endpoint["max_prompt_tokens"]:
        raise ContractError("Conservative prompt-token bound exceeded")
    price = pricing_bound(endpoint)
    return tokens * price["prompt"] + payload["max_tokens"] * price["completion"] + price["request"]


class Ledger:
    """Cross-process atomic reservations enforce both dollar ceilings, including uncertain sends."""

    def __init__(self, path, run_id, run_limit, total_limit):
        self.path, self.run_id = Path(path), run_id
        self.run_limit, self.total_limit = run_limit, total_limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread_lock = threading.Lock()

    @contextlib.contextmanager
    def transaction(self):
        with self.thread_lock, self.path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = read_json(self.path) if self.path.exists() else {"entries": {}}
            yield data
            write_json(self.path, data)
            fcntl.flock(lock, fcntl.LOCK_UN)

    def reserve(self, key, amount):
        with self.transaction() as data:
            if key in data["entries"]:
                raise BudgetStop(f"Request has an unsettled ledger entry: {key}; reconcile before resend")
            entries = list(data["entries"].values())
            total = sum(e["charge"] for e in entries)
            local = sum(e["charge"] for e in entries if e["run_id"] == self.run_id)
            if total + amount > self.total_limit or local + amount > self.run_limit:
                raise BudgetStop(
                    f"Budget ceiling reached before dispatch (reserved/charged ${local:.4f} run, "
                    f"${total:.4f} total; next bound ${amount:.4f})"
                )
            data["entries"][key] = {
                "run_id": self.run_id,
                "charge": amount,
                "upper": amount,
                "state": "reserved",
                "created_at": now(),
            }

    def settle(self, key, actual, state="complete"):
        with self.transaction() as data:
            entry = data["entries"][key]
            if actual is not None:
                if actual < 0 or not math.isfinite(actual):
                    raise ContractError("Invalid provider cost")
                entry["charge"] = actual
            entry.update(state=state, settled_at=now())
            if actual is not None and actual > entry["upper"]:
                entry["state"] = "pricing_bound_violation"
                raise BudgetStop("Actual charge exceeded reserved pricing bound; halt and refresh pricing")

    def summary(self):
        with self.transaction() as data:
            entries = [e for e in data["entries"].values() if e["run_id"] == self.run_id]
            return {
                "charged_or_reserved_usd": sum(e["charge"] for e in entries),
                "actual_complete_usd": sum(e["charge"] for e in entries if e["state"] == "complete"),
                "requests": len(entries),
                "unresolved": sum(e["state"] != "complete" for e in entries),
            }


class OpenRouter:
    def __init__(self, output, roles, seed, budget, total_budget, ledger_path, client=None):
        load_dotenv(ROOT / ".env")
        self.key = os.getenv("OPENROUTER_API_KEY")
        if not self.key and client is None:
            raise ContractError("OPENROUTER_API_KEY is not configured; no paid calls dispatched")
        if budget is None or total_budget is None or budget <= 0 or total_budget <= 0:
            raise ContractError("Positive per-run and total USD ceilings required")
        self.output, self.roles, self.seed = Path(output), roles, seed
        self.output.mkdir(parents=True, exist_ok=True)
        self.client = client or httpx.Client(timeout=httpx.Timeout(600, connect=30))
        self.endpoints = {r: endpoint_for(c) for r, c in roles.items()}
        self.ledger = Ledger(ledger_path, str(self.output.resolve()), budget, total_budget)
        self.dispatch_stopped = threading.Event()

    def payload(self, role, system, data, schema, identity):
        cfg = self.roles[role]
        payload = {
            "model": cfg["model"],
            "stream": False,
            "plugins": [],
            "transforms": [],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": canonical(data)}],
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
        if schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "xar_output", "strict": True, "schema": schema},
            }
        if "seed" in self.endpoints[role][0]["supported_parameters"]:
            payload["seed"] = int(digest({"seed": self.seed, "identity": identity})[:8], 16) % (2**31)
        return payload

    def preflight(self):
        """Fresh read-only catalog evidence before a batch; pinned settings never silently change."""
        stamp = now().replace(":", "-")
        directory = self.output / "preflight" / stamp
        response = self.client.get(API_BASE + "/models")
        response.raise_for_status()
        catalog = response.json()
        write_json(directory / "models.json", catalog)
        models = {m["id"]: m for m in catalog["data"]}
        endpoints = {}
        pricing_bounds = {}
        for role, cfg in self.roles.items():
            model = cfg["model"]
            if (
                model not in models
                or models[model]["canonical_slug"] != self.endpoints[role][1]["canonical_slug"]
            ):
                raise ContractError("Pinned catalog release changed; declare a new batch")
            if model not in endpoints:
                response = self.client.get(API_BASE + "/models/" + model + "/endpoints")
                response.raise_for_status()
                endpoints[model] = response.json()
                write_json(directory / (model.replace("/", "_") + "-endpoints.json"), endpoints[model])
            matching = [e for e in endpoints[model]["data"]["endpoints"] if e["tag"] == cfg["provider"]]
            if len(matching) != 1:
                raise ContractError("Pinned provider endpoint missing or ambiguous")
            current, prior = matching[0], self.endpoints[role][0]
            required = {"temperature", "reasoning", "max_tokens", "response_format", "structured_outputs"}
            if not required <= set(current["supported_parameters"]):
                raise ContractError("Pinned endpoint no longer supports required parameters")
            if (
                current["context_length"] < prior["context_length"]
                or (current.get("max_completion_tokens") or current["context_length"]) < cfg["max_tokens"]
            ):
                raise ContractError("Pinned endpoint limits changed; recheck whole payloads")
            current_price, frozen_upper = pricing_upper(current), pricing_bound(prior)
            if any(current_price[k] > frozen_upper[k] for k in frozen_upper):
                raise ContractError(
                    "Endpoint pricing exceeds frozen upper rates; refresh evidence for a new batch"
                )
            pricing_bounds[role] = {
                "observed": current_price,
                "frozen_upper": frozen_upper,
                "pricing_headroom_multiplier": PRICING_HEADROOM,
                "price_changed_within_bound": current_price != pricing_upper(prior),
                "lower_prices_within_bound": all(
                    current_price[k] <= pricing_upper(prior)[k] for k in current_price
                )
                and current_price != pricing_upper(prior),
            }
            if current.get("quantization") != prior.get("quantization"):
                raise ContractError("Pinned endpoint precision changed; declare a new batch")
            if "seed" in prior["supported_parameters"] and "seed" not in current["supported_parameters"]:
                raise ContractError("Pinned endpoint no longer supports the frozen seed parameter")
            efforts = models[model].get("reasoning", {}).get("supported_efforts", [])
            if efforts and cfg["reasoning"].get("effort", efforts[0]) not in efforts:
                raise ContractError("Pinned reasoning mapping changed")
        write_json(
            directory / "checks.json",
            {
                "at": now(),
                "roles": self.roles,
                "passed": True,
                "read_only": True,
                "pricing_bounds": pricing_bounds,
                "paid_capability_checks": "separate_operational_pilot",
            },
        )
        return directory

    def call(self, role, system, data, schema, identity):
        try:
            return self._call(role, system, data, schema, identity)
        except BaseException:
            self.dispatch_stopped.set()
            raise

    def _call(self, role, system, data, schema, identity):
        payload = self.payload(role, system, data, schema, identity)
        request_key = digest({"payload": payload, "identity": identity, "schema_version": SCHEMA_VERSION})
        directory = self.output / "requests" / request_key
        directory.mkdir(parents=True, exist_ok=True)
        cached = directory / "result.json"
        if cached.exists():
            return read_json(cached)
        write_json(
            directory / "request.json",
            {
                "payload": payload,
                "identity": identity,
                "key": request_key,
                "role": role,
                "schema_version": SCHEMA_VERSION,
            },
            immutable=True,
        )
        bound = request_upper(payload, self.endpoints[role][0])
        # One initial call plus three transport retries. Failed HTTP sends retain a reservation.
        for attempt in range(4):
            receipt = directory / f"attempt_{attempt}.json"
            ledger_key = str(self.output.resolve()) + "/" + request_key + f"/{attempt}"
            if receipt.exists():
                prior = read_json(receipt)
                if prior["status"] == "success":
                    result = self.extract(prior["response"], request_key, ledger_key, directory)
                    write_json(cached, result)
                    return result
                if prior["status"] == "uncertain":
                    raise BudgetStop(
                        f"Prior send unresolved: {receipt}; inspect provider history before retry"
                    )
                if not prior.get("retryable"):
                    raise ContractError(f"Prior nonretryable transport failure: {receipt}")
                continue
            if self.dispatch_stopped.is_set():
                raise ContractError("Dispatch halted after another request failed")
            self.ledger.reserve(ledger_key, bound)
            sent_at = now()
            write_json(receipt, {"status": "uncertain", "sent_at": sent_at, "upper_usd": bound})
            started = time.monotonic()
            try:
                response = self.client.post(
                    API_BASE + "/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": "Bearer " + (self.key or "test"),
                        "X-OpenRouter-Title": "Independent XAR reproduction",
                    },
                )
                try:
                    raw = response.json()
                except ValueError:
                    raw = {"raw_body": response.text}
                retryable = response.status_code in {408, 429, 500, 502, 503, 504}
                success = response.status_code == 200 and "choices" in raw and not raw.get("error")
                write_json(
                    receipt,
                    {
                        "status": "success" if success else "http_error",
                        "response": raw,
                        "http_status": response.status_code,
                        "retryable": retryable,
                        "timestamp": now(),
                        "sent_at": sent_at,
                        "duration_seconds": time.monotonic() - started,
                    },
                )
                if success:
                    result = self.extract(raw, request_key, ledger_key, directory)
                    write_json(cached, result)
                    return result
                self.ledger.settle(ledger_key, None, "http_error_reserved")
                if not retryable:
                    raise ContractError(f"OpenRouter HTTP {response.status_code}; see {receipt}")
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # Connection was never established; no billable completion was sent.
                self.ledger.settle(ledger_key, 0, "complete")
                write_json(receipt, {"status": "connect_error", "retryable": True, "timestamp": now()})
            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError):
                self.ledger.settle(ledger_key, None, "uncertain")
                raise BudgetStop(
                    f"Send outcome unknown; preserve reservation and reconcile {receipt}"
                ) from None
            if attempt < 3:
                time.sleep(min(2**attempt, 8))
        raise ContractError(f"Transport retry allowance exhausted: {directory}")

    def extract(self, raw, key, ledger_key, directory):
        usage = raw.get("usage", {})
        cost = usage.get("cost")
        self.ledger.settle(
            ledger_key,
            float(cost) if cost is not None else None,
            "complete" if cost is not None else "cost_unknown",
        )
        if cost is None:
            raise BudgetStop(f"Provider response lacks cost; reconcile usage before new calls: {directory}")
        choice = raw["choices"][0]
        returned = raw.get("model")
        expected = read_json(directory / "request.json")["payload"]["model"]
        catalog = next(v[1] for v in self.endpoints.values() if v[1]["id"] == expected)
        if returned not in {expected, catalog["canonical_slug"]}:
            raise ContractError(f"Returned model identity differs: {returned}")
        provider = raw.get("provider")
        request_role = read_json(directory / "request.json")["role"]
        endpoint = self.endpoints[request_role][0]
        if provider and provider not in {endpoint["provider_name"], endpoint["tag"]}:
            raise ContractError(f"Returned provider differs: {provider}")
        receipt = directory / ("attempt_" + ledger_key.rsplit("/", 1)[-1] + ".json")
        return {
            "content": choice["message"].get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "request_key": key,
            "response_id": raw.get("id"),
            "usage": usage,
            "model": returned,
            "provider": provider,
            "raw_response": str(receipt),
        }

    def structured(self, role, system, data, schema, identity, validator=None, repair=True):
        attempts = []
        for i in range(2 if repair else 1):
            instructions = system
            if i:
                instructions += "\nFORMAT REPAIR: Return complete valid JSON matching the schema. "
                instructions += (
                    "Keep the substantive task inputs unchanged. Previous validation error: "
                    + attempts[-1]["error"]
                )
            response = self.call(role, instructions, data, schema, {"task": identity, "format_attempt": i})
            try:
                if response["finish_reason"] != "stop":
                    raise ContractError(f"Incomplete output: {response['finish_reason']}")
                value = json.loads(response["content"])
                jsonschema.validate(value, schema)
                if validator:
                    validator(value)
                attempts.append({"response": response, "status": "valid"})
                return {"status": "valid", "value": value, "attempts": attempts}
            except (ValueError, jsonschema.ValidationError, ContractError) as e:
                attempts.append({"response": response, "status": "invalid", "error": str(e)[:500]})
        return {"status": "missing", "value": None, "attempts": attempts}


def load_examples(dataset, splits, selected_split=None):
    split_data = read_json(splits)
    groups = split_data["papers"]
    flat = [p for papers in groups.values() for p in papers]
    if len(flat) != len(set(flat)):
        raise ContractError("Paper splits overlap")
    examples = [json.loads(line) for line in Path(dataset).read_text().splitlines() if line.strip()]
    ids = [e["example_id"] for e in examples]
    if len(ids) != len(set(ids)):
        raise ContractError("Duplicate example IDs")
    counts = {}
    for e in examples:
        matching = [s for s, papers in groups.items() if e["paper_id"] in papers]
        if matching != [e["split"]]:
            raise ContractError("Paper/example split mismatch")
        if e["context_hash"] != digest(e["context"]) or e["reference_hash"] != digest(e["reference"]):
            raise ContractError("Example content hash mismatch")
        if e["target_words"] != words(e["reference"]) or not e["target_words"]:
            raise ContractError("Target word count mismatch")
        if normalize(e["reference"]) in normalize(e["context"]):
            raise ContractError("Withheld reference remains in visible context")
        counts.setdefault(e["paper_id"], []).append(e["section_type"])
    for p, sections in counts.items():
        if sorted(sections) != sorted(SECTIONS):
            raise ContractError(f"Missing/duplicate target sections: {p}")
    return [e for e in examples if selected_split is None or e["split"] == selected_split]


def task_data(example):
    # This allowlist prevents expert text, split labels, IDs, or prior feedback entering G/J.
    return {
        "visible_paper": example["context"],
        "section_type": example["section_type"],
        "target_words": example["target_words"],
    }


def contamination(candidate, reference):
    a, b = normalize(candidate).split(), normalize(reference).split()
    positions = {}
    for i, word in enumerate(b):
        positions.setdefault(word, []).append(i)
    longest, prior = 0, {}
    for word in a:
        current = {j: prior.get(j - 1, 0) + 1 for j in positions.get(word, [])}
        longest = max(longest, max(current.values(), default=0))
        prior = current
    reference_ngrams = {tuple(b[i : i + 8]) for i in range(max(0, len(b) - 7))}
    overlap = sum(tuple(a[i : i + 8]) in reference_ngrams for i in range(max(0, len(a) - 7)))
    return {
        "longest_verbatim_run_words": longest,
        "eightgram_overlap_fraction": overlap / max(1, len(a) - 7),
        "flagged": longest >= 30,
        "exclusion": False,
    }


def bounded_map(function, items, concurrency, stopped=None):
    """Keep at most concurrency tasks active; retain input order and stop refilling on errors."""
    if not 1 <= concurrency <= 32:
        raise ContractError("Concurrency must be 1–32")
    stopped = stopped if stopped is not None else threading.Event()

    def invoke(item):
        if stopped.is_set():
            raise ContractError("Parallel work halted after another task failed")
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


def writer_candidates(api, examples, output, existing=None, concurrency=1):
    if not 1 <= concurrency <= 32:
        raise ContractError("Writer concurrency must be 1–32")
    config_hash = digest(
        {"writer": api.roles["writer"], "prompt": prompt("writer"), "schema_version": SCHEMA_VERSION}
    )
    frozen = read_json(existing) if existing else None
    if frozen and frozen["writer_configuration_hash"] != config_hash:
        raise ContractError("Cached writer configuration differs")
    stopped = getattr(api, "dispatch_stopped", threading.Event())

    def generate_one(e):
        path = Path(output) / "generations" / (e["example_id"] + ".json")
        if frozen:
            record = frozen["candidates"].get(e["example_id"])
            if not record:
                raise ContractError("Cached writer candidate coverage incomplete")
        elif path.exists():
            record = read_json(path)
        else:
            attempts = []
            for attempt in range(3):
                data = task_data(e)
                if attempt:
                    data.update(
                        previous_section=attempts[-1]["text"],
                        length_revision=f"Revise only to fit {math.ceil(0.85 * e['target_words'])}–"
                        f"{math.floor(1.15 * e['target_words'])} words. Preserve claims.",
                    )
                if stopped.is_set():
                    raise ContractError("Writer generation halted after another section failed")
                response = api.call(
                    "writer",
                    prompt("writer"),
                    data,
                    None,
                    {"writer": config_hash, "example": e["example_id"], "attempt": attempt},
                )
                text = response["content"].strip()
                count = words(text)
                compliant = 0.85 * e["target_words"] <= count <= 1.15 * e["target_words"]
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
                "example_id": e["example_id"],
                "context_hash": e["context_hash"],
                "sampling_seed": getattr(api, "seed", 0),
                "writer_configuration_hash": config_hash,
                "writer_configuration": api.roles["writer"],
                "attempts": attempts,
                "accepted_attempt": accepted["attempt_index"],
                "text": accepted["text"],
                "text_hash": accepted["text_hash"],
                "words": accepted["words"],
                "length_compliant": accepted["length_compliant"],
                "complete": accepted["response"]["finish_reason"] == "stop",
                "length_ratio": accepted["words"] / e["target_words"],
                "contamination": contamination(accepted["text"], e["reference"]),
            }
            write_json(path, record, immutable=True)
        if record["context_hash"] != e["context_hash"] or record["text_hash"] != digest(record["text"]):
            raise ContractError("Cached writer context or content differs")
        if record["writer_configuration_hash"] != config_hash:
            raise ContractError("Local cached writer settings differ")
        return record

    records = bounded_map(generate_one, examples, concurrency, stopped)
    # Completion order never changes the frozen dataset/candidate order.
    candidates = {record["example_id"]: record for record in records}
    artifact = {
        "writer_configuration_hash": config_hash,
        "writer_configuration": api.roles["writer"],
        "candidates": candidates,
    }
    write_json(Path(output) / "generations/candidates.json", artifact, immutable=True)
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
    write_json(output, record, immutable=True)
    return record


def grade_candidate(api, example, rubric_record, text, identity, output, role="judge"):
    if rubric_record["status"] != "valid":
        result = {"status": "missing", "value": None, "attempts": [], "reason": "invalid_rubric"}
    else:
        rubric = rubric_record["value"]
        result = api.structured(
            role,
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
        "judge_configuration": api.roles[role],
        "identity": identity,
        **result,
    }
    record["total"] = (
        statistics.mean(s["score"] for s in result["value"]["scores"]) if result["value"] else None
    )
    write_json(output, record, immutable=True)
    return record


def evaluate_checkpoint(
    api,
    examples,
    candidates,
    meta_prompt,
    checkpoint,
    output,
    concurrency=2,
    frozen_rubrics=None,
    judge_role="judge",
    namespace="main",
):
    root = Path(output)

    def evaluate(e):
        eid = e["example_id"]
        label = f"{namespace}/{checkpoint}/{e['split']}/{eid}"
        if frozen_rubrics is not None:
            rubric = frozen_rubrics[eid]
            if rubric["context_hash"] != e["context_hash"]:
                raise ContractError("Frozen rubric context differs")
            write_json(root / "rubrics" / label / "rubric.json", rubric, immutable=True)
        else:
            rubric = generate_rubric(
                api, e, meta_prompt, {"rubric": label}, root / "rubrics" / label / "rubric.json"
            )
        labels = ["human", "model"]
        random.Random(int(digest({"seed": api.seed, "label": label})[:16], 16)).shuffle(labels)
        grades = {}
        for slot, origin in enumerate(labels):
            text = e["reference"] if origin == "human" else candidates[eid]["text"]
            # Origin is in artifact paths only, never model inputs or seed identity.
            grades[origin] = grade_candidate(
                api,
                e,
                rubric,
                text,
                {"grade": label, "slot": slot},
                root / "scores" / label / (origin + ".json"),
                judge_role,
            )
        human, model = grades["human"]["total"], grades["model"]["total"]
        return {
            "example_id": eid,
            "paper_id": e["paper_id"],
            "section_type": e["section_type"],
            "split": e["split"],
            "checkpoint": checkpoint,
            "human": human,
            "model": model,
            "gap": human - model if human is not None and model is not None else None,
            "length_ratio": candidates[eid]["length_ratio"],
            "length_compliant": candidates[eid]["length_compliant"],
            "contamination_flagged": candidates[eid]["contamination"]["flagged"],
            "rubric_path": str(root / "rubrics" / label / "rubric.json"),
            "grade_paths": {o: str(root / "scores" / label / (o + ".json")) for o in labels},
        }

    rows = bounded_map(evaluate, examples, concurrency, getattr(api, "dispatch_stopped", None))
    write_json(root / "scores" / namespace / str(checkpoint) / f"{examples[0]['split']}_rows.json", rows)
    return rows


def bootstrap(rows, seed=0, replicates=2000, field="gap"):
    bundles = {}
    for r in rows:
        if r.get(field) is not None:
            bundles.setdefault(r["paper_id"], []).append(r[field])
    if not bundles:
        return None
    rng = random.Random(seed)
    papers = sorted(bundles)
    samples = sorted(
        statistics.mean(value for p in rng.choices(papers, k=len(papers)) for value in bundles[p])
        for _ in range(replicates)
    )
    return {
        "method": "paired_whole_paper_percentile_bootstrap",
        "paper_clusters": len(papers),
        "replicates": replicates,
        "seed": seed,
        "estimate": statistics.mean(value for values in bundles.values() for value in values),
        "bootstrap_median": samples[replicates // 2],
        "low": samples[int(0.025 * replicates)],
        "high": samples[int(0.975 * replicates)],
    }


def summarize(rows, seed=0):
    valid = [r for r in rows if r["gap"] is not None]

    def mean(key):
        return statistics.mean(r[key] for r in valid) if valid else None

    return {
        "examples": len(rows),
        "paired_coverage": len(valid),
        "complete": len(valid) == len(rows),
        "human": mean("human"),
        "model": mean("model"),
        "gap": mean("gap"),
        "positive_gap_fraction": sum(r["gap"] > 0 for r in valid) / len(valid) if valid else None,
        "ties": sum(r["gap"] == 0 for r in valid),
        "paper_interval": bootstrap(valid, seed),
        "length_compliant": sum(r["length_compliant"] for r in rows),
        "contamination_flagged": sum(r["contamination_flagged"] for r in rows),
        "sections": {
            s: {
                "coverage": sum(r["section_type"] == s for r in valid),
                "gap": statistics.mean(r["gap"] for r in valid if r["section_type"] == s)
                if any(r["section_type"] == s for r in valid)
                else None,
            }
            for s in SECTIONS
        },
        "compliant_sensitivity": bootstrap([r for r in valid if r["length_compliant"]], seed),
        "unflagged_sensitivity": bootstrap([r for r in valid if not r["contamination_flagged"]], seed),
    }


def paired_improvement(initial, selected, seed=0):
    base = {r["example_id"]: r for r in initial}
    rows = [
        {"paper_id": r["paper_id"], "gap": r["gap"] - base[r["example_id"]]["gap"]}
        for r in selected
        if r["gap"] is not None and base[r["example_id"]]["gap"] is not None
    ]
    return {
        "paired_coverage": len(rows),
        "mean": statistics.mean(r["gap"] for r in rows) if rows else None,
        "interval": bootstrap(rows, seed),
    }


def write_table(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def common_parser(description, roles):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config")
    p.add_argument("--dataset", default="data/examples.jsonl")
    p.add_argument("--splits", default="data/splits.json")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budget-usd", type=float)
    p.add_argument("--total-budget-usd", type=float)
    p.add_argument("--budget-ledger", default="runs/budget_ledger.json")
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--split", choices=["train", "validation", "pilot", "confirmation", "research"], default="research"
    )
    for role in roles:
        p.add_argument(f"--{role}-model")
        p.add_argument(f"--{role}-provider")
        p.add_argument(f"--{role}-temperature", type=float)
        p.add_argument(f"--{role}-reasoning-mode", choices=["enabled", "disabled"])
        p.add_argument(f"--{role}-reasoning-effort")
        p.add_argument(f"--{role}-max-output-tokens", type=int)
    return p


def parse_args(parser, argv=None):
    import sys

    argv = sys.argv[1:] if argv is None else argv
    known, _ = parser.parse_known_args(argv)
    if known.config:
        config = yaml.safe_load(Path(known.config).read_text()) or {}
        allowed = {a.dest for a in parser._actions}
        invalid = set(config) - allowed
        if invalid:
            raise ContractError(f"Unknown config keys: {sorted(invalid)}")
        parser.set_defaults(**config)
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")
    if args.total_budget_usd is None and os.getenv("XAR_TOTAL_BUDGET_USD"):
        args.total_budget_usd = float(os.getenv("XAR_TOTAL_BUDGET_USD"))
    if args.concurrency < 1 or args.concurrency > 32:
        raise ContractError("Concurrency must be 1–32")
    return args


def selected_examples(args):
    examples = load_examples(args.dataset, args.splits)
    chosen = (
        [e for e in examples if e["split"] in ("train", "validation")]
        if args.split == "research"
        else [e for e in examples if e["split"] == args.split]
    )
    if not chosen:
        raise ContractError("Requested split has no examples")
    return chosen


def software_hashes():
    paths = (
        list(ROOT.glob("*.py"))
        + list((ROOT / "prompts").glob("*.md"))
        + [ROOT / "uv.lock", ROOT / "configs/tokenizers.json"]
    )
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def operational_summary(output):
    output = Path(output)
    attempts = {}

    def collect(value):
        if isinstance(value, dict):
            if value.get("status") in {"valid", "invalid"} and isinstance(value.get("response"), dict):
                response = value["response"]
                if response.get("request_key"):
                    attempts[response["request_key"]] = value["status"]
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    for folder in ("rubrics", "scores", "feedback"):
        for path in (output / folder).rglob("*.json"):
            collect(read_json(path))
    role_rates = {}
    inherited = 0
    for key, status in attempts.items():
        request_path = output / "requests" / key / "request.json"
        if not request_path.exists():
            inherited += 1
            continue
        role = read_json(request_path)["role"]
        counts = role_rates.setdefault(role, {"structured_attempts": 0, "invalid_attempts": 0})
        counts["structured_attempts"] += 1
        counts["invalid_attempts"] += status == "invalid"
    for counts in role_rates.values():
        counts["invalid_fraction"] = counts["invalid_attempts"] / counts["structured_attempts"]
    usage_rows = []
    for request in (output / "requests").glob("*/request.json"):
        metadata = read_json(request)
        for receipt in request.parent.glob("attempt_*.json"):
            data = read_json(receipt)
            usage = data.get("response", {}).get("usage", {})
            usage_rows.append(
                {
                    "role": metadata["role"],
                    "status": data["status"],
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "cost_usd": usage.get("cost"),
                    "latency_seconds": data.get("duration_seconds"),
                }
            )
    summary = {
        "format_validation": role_rates,
        "transport_attempts": len(usage_rows),
        "inherited_frozen_responses": inherited,
        "raw_usage_cost_usd": sum(r["cost_usd"] or 0 for r in usage_rows),
        "unresolved_transport": sum(r["status"] == "uncertain" for r in usage_rows),
    }
    write_json(output / "operational_summary.json", summary)
    write_table(output / "scores/request_usage.csv", usage_rows)
    return summary


def resolved_manifest(args, roles, experiment, extra=None):
    arguments = {
        k: v for k, v in vars(args).items() if k not in {"dry_run", "resume", "config", "output_dir"}
    }
    endpoints = {}
    for role, cfg in roles.items():
        e, m = endpoint_for(cfg)
        endpoints[role] = {
            "endpoint": e,
            "canonical_slug": m["canonical_slug"],
            "unsupported_seed": "seed" not in e["supported_parameters"],
        }
    manifest = {
        "experiment": experiment,
        "arguments": arguments,
        "roles": roles,
        "endpoints": endpoints,
        "dataset_hash": file_hash(args.dataset),
        "splits_hash": file_hash(args.splits),
        "dataset": str(Path(args.dataset).resolve()),
        "splits": str(Path(args.splits).resolve()),
        "software_hashes": software_hashes(),
        "schema_version": SCHEMA_VERSION,
        "extra": extra or {},
    }
    substantive = copy.deepcopy(manifest)
    for k in ("budget_usd", "total_budget_usd"):
        substantive["arguments"].pop(k, None)
    manifest["substantive_hash"] = digest(substantive)
    return manifest


def initialize_run(args, roles, experiment, extra=None):
    manifest = resolved_manifest(args, roles, experiment, extra)
    out = Path(args.output_dir)
    path = out / "manifest.json"
    review_path = Path(args.dataset).parent / "human_review.json"
    if not review_path.exists():
        raise ContractError(
            "PLAN.md requires human extraction/suitability review; fill data/human_review.json"
        )
    review = read_json(review_path)
    if review.get("dataset_hash") != file_hash(args.dataset):
        raise ContractError("Human review does not match the frozen dataset hash")
    used_examples = load_examples(args.dataset, args.splits)
    split = getattr(args, "split", "research")
    used_papers = {
        e["paper_id"]
        for e in used_examples
        if (e["split"] in ("train", "validation") if split == "research" else e["split"] == split)
    }
    if any(review.get("papers", {}).get(p, {}).get("decision") != "approved" for p in used_papers):
        raise ContractError("Selected papers still need human review; see data/review.md")
    manifest["human_review_hash"] = file_hash(review_path)
    api = OpenRouter(out, roles, args.seed, args.budget_usd, args.total_budget_usd, args.budget_ledger)
    api.preflight()
    if path.exists():
        if not args.resume:
            raise ContractError("Run exists; use --resume or a new output directory")
        saved = read_json(path)
        if saved["substantive_hash"] != manifest["substantive_hash"]:
            raise ContractError("Resume substantive configuration changed; create a new run")
        if (
            saved["arguments"].get("budget_usd") != args.budget_usd
            or saved["arguments"].get("total_budget_usd") != args.total_budget_usd
        ):
            history = (
                read_json(out / "budget_continuations.json")
                if (out / "budget_continuations.json").exists()
                else []
            )
            change = {"budget_usd": args.budget_usd, "total_budget_usd": args.total_budget_usd}
            if not history or any(history[-1].get(k) != v for k, v in change.items()):
                history.append({"at": now(), **change})
                write_json(out / "budget_continuations.json", history)
    else:
        manifest["created_at"] = now()
        try:
            manifest["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
        except subprocess.CalledProcessError:
            manifest["git_commit"] = None
        write_json(path, manifest, immutable=True)
    return api


@contextlib.contextmanager
def run_lock(output):
    Path(output).mkdir(parents=True, exist_ok=True)
    with (Path(output) / ".run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ContractError("Another process holds this run") from None
        yield


def estimate(args, roles, examples, counts):
    """Read-only planning; typical costs are estimates, not spending authorization."""
    contexts = [len(e["context"].encode()) for e in examples]
    estimates = {}
    for role, count in counts.items():
        endpoint, _ = endpoint_for(roles[role])
        if count:
            for e in examples:
                tokens = math.ceil(1.25 * token_count(canonical(task_data(e)), roles[role]["model"])) + 16000
                if tokens + roles[role]["max_tokens"] > endpoint["context_length"]:
                    raise ContractError(f"{role} context does not fit {e['example_id']}")
        price = pricing_upper(endpoint)
        typical_tokens = statistics.mean(contexts) / 3.5 + 1500
        output_tokens = min(roles[role]["max_tokens"], 5000)
        estimates[role] = {
            "requests": count,
            "typical_uncached_usd": count
            * (typical_tokens * price["prompt"] + output_tokens * price["completion"]),
            "per_request_conservative_usd": (max(contexts) + 12000) * price["prompt"]
            + roles[role]["max_tokens"] * price["completion"],
        }
    reuse = {}
    for field in ("writer_generations", "target_generations"):
        artifact = getattr(args, field, None)
        if artifact:
            path = Path(artifact)
            reuse[field] = {
                "path": artifact,
                "status": "available" if path.exists() else "preceding_phase_prerequisite_missing",
            }
            if path.exists() and "writer" in roles:
                saved = read_json(path)
                expected_hash = digest(
                    {"writer": roles["writer"], "prompt": prompt("writer"), "schema_version": SCHEMA_VERSION}
                )
                if saved["writer_configuration_hash"] != expected_hash:
                    raise ContractError("Planned cached writer artifact has incompatible settings")
                for e in examples:
                    candidate = saved["candidates"].get(e["example_id"])
                    if not candidate or candidate["context_hash"] != e["context_hash"]:
                        raise ContractError("Planned cached writer artifact lacks matching example coverage")
    result = {
        "dry_run": True,
        "examples": len(examples),
        "roles": roles,
        "counts_and_costs": estimates,
        "artifact_reuse": reuse,
        "retry_reserve_fraction": 0.25,
        "estimated_phase_usd_with_reserve": 1.25 * sum(e["typical_uncached_usd"] for e in estimates.values()),
        "pricing_source": "saved_endpoint_snapshots",
        "limitations": "Pilot token usage required; optimizer feedback may be larger.",
        "budget_usd": args.budget_usd,
        "total_budget_usd": args.total_budget_usd,
    }
    print(json.dumps(result, indent=2))
    return result


def audit_request_contract(payload, cfg, endpoint, schema, prompt_file):
    """Check the actual sent request against the frozen role, scoring wrapper, and schema."""
    expected = {
        "model": cfg["model"],
        "temperature": cfg["temperature"],
        "reasoning": cfg["reasoning"],
        "max_tokens": cfg["max_tokens"],
        "stream": False,
        "plugins": [],
        "transforms": [],
        "provider": {
            "only": [cfg["provider"]],
            "order": [cfg["provider"]],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ContractError("Saved request differs from frozen role/routing/decoding contract")
    if payload.get("response_format") != {
        "type": "json_schema",
        "json_schema": {"name": "xar_output", "strict": True, "schema": schema},
    }:
        raise ContractError("Saved request differs from fixed output schema")
    messages = payload.get("messages", [])
    if len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]:
        raise ContractError("Saved request includes conversation history or unexpected message roles")
    system = messages[0]["content"]
    base = system.split("\nFORMAT REPAIR: Return complete valid JSON matching the schema. ", 1)[0]
    if digest(base) != prompt_file:
        raise ContractError("Saved request changed the frozen grading/rubric wrapper")
    if "seed" in payload and "seed" not in endpoint["endpoint"]["supported_parameters"]:
        raise ContractError("Saved request used an unsupported seed")


def audit_saved_output(record, schema, cfg=None, endpoint=None, prompt_file=None):
    if record["status"] != "valid":
        raise ContractError("Missing structured output")
    last = record["attempts"][-1]
    if last["status"] != "valid":
        raise ContractError("Artifact marked valid with invalid final attempt")
    response = last["response"]
    receipt = read_json(response["raw_response"])
    if receipt["status"] != "success":
        raise ContractError("Artifact lacks successful raw response")
    choice = receipt["response"]["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ContractError("Artifact uses incomplete raw response")
    parsed = json.loads(choice["message"]["content"])
    jsonschema.validate(parsed, schema)
    if canonical(parsed) != canonical(record["value"]):
        raise ContractError("Derived artifact differs from raw response")
    request = read_json(Path(response["raw_response"]).parent / "request.json")
    if cfg is not None:
        audit_request_contract(request["payload"], cfg, endpoint, schema, prompt_file)
        raw = receipt["response"]
        if raw.get("model") not in {cfg["model"], endpoint["canonical_slug"]}:
            raise ContractError("Raw response used a different model from the frozen role")
        provider = raw.get("provider")
        if provider and provider not in {endpoint["endpoint"]["provider_name"], cfg["provider"]}:
            raise ContractError("Raw response used a different provider from the frozen role")
    return request["payload"]


def audit_xar_run(path):
    """Rebuild means from raw-linked criterion grades and prove full-scope checkpoint coverage."""
    path = Path(path)
    manifest, freeze = read_json(path / "manifest.json"), read_json(path / "freeze.json")
    if manifest["experiment"] != "xar":
        raise ContractError("Expected XAR run")
    examples = load_examples(manifest["dataset"], manifest["splits"])
    pilot = manifest["arguments"]["split"] == "pilot"
    if pilot:
        papers = sorted({e["paper_id"] for e in examples if e["split"] == "pilot"})
        examples = [
            {**e, "split": "train" if e["paper_id"] == papers[0] else "validation"}
            for e in examples
            if e["split"] == "pilot"
        ]
    else:
        examples = [e for e in examples if e["split"] in ("train", "validation")]
        if {s: sum(e["split"] == s for e in examples) for s in ("train", "validation")} != {
            "train": 32,
            "validation": 20,
        }:
            raise ContractError("Primary dataset must have 32 train and 20 validation examples")
    if (
        file_hash(manifest["dataset"]) != manifest["dataset_hash"]
        or file_hash(manifest["splits"]) != manifest["splits_hash"]
    ):
        raise ContractError("Run data changed")
    candidates = read_json(path / "generations/candidates.json")["candidates"]
    all_rows, table = {}, []
    iterations = manifest["arguments"]["iterations"]
    if not pilot and iterations != 7:
        raise ContractError("Primary trajectory lacks seven updates")
    for iteration in range(iterations + 1):
        checkpoint = (path / f"prompts/iter_{iteration:02d}.md").read_text()
        if digest(checkpoint) != freeze["prompt_hashes"][iteration]:
            raise ContractError("Frozen checkpoint differs")
        if iteration:
            feedback = read_json(path / f"feedback/iter_{iteration:02d}/training.json")
            training_ids = {e["example_id"] for e in examples if e["split"] == "train"}
            if set(feedback["example_ids_used_for_aggregate"]) != training_ids:
                raise ContractError("Optimizer aggregate contains non-training examples")
            if any(f["example_id"] not in training_ids for f in feedback["failures"]):
                raise ContractError("Optimizer failures contain non-training examples")
        summaries = {}
        for split in ("train", "validation"):
            rows = []
            for e in [e for e in examples if e["split"] == split]:
                eid = e["example_id"]
                candidate = candidates[eid]
                if candidate["context_hash"] != e["context_hash"] or candidate["text_hash"] != digest(
                    candidate["text"]
                ):
                    raise ContractError("Writer candidate integrity failed")
                rubric = read_json(path / f"rubrics/main/{iteration}/{split}/{eid}/rubric.json")
                if rubric["generator_configuration"] != manifest["roles"]["rubric"]:
                    raise ContractError("Rubric artifact configuration differs from the frozen generator")
                rubric_payload = audit_saved_output(
                    rubric,
                    RUBRIC_SCHEMA,
                    manifest["roles"]["rubric"],
                    manifest["endpoints"]["rubric"],
                    manifest["software_hashes"]["prompts/rubric_wrapper.md"],
                )
                if split == "validation":
                    if not freeze.get("frozen_at"):
                        raise ContractError(
                            "Freeze lacks timestamp evidence for held-out evaluation ordering"
                        )
                    for record in rubric["attempts"]:
                        receipt = read_json(record["response"]["raw_response"])
                        if not receipt.get("sent_at") or receipt["sent_at"] < freeze["frozen_at"]:
                            raise ContractError("Validation rubric was dispatched before trajectory freeze")
                validate_rubric(rubric["value"])
                if json.loads(rubric_payload["messages"][1]["content"]) != {
                    **task_data(e),
                    "meta_prompt": checkpoint,
                }:
                    raise ContractError("Rubric inputs contain an unexpected field or changed context")
                totals = {}
                for origin, text in (("human", e["reference"]), ("model", candidate["text"])):
                    grade = read_json(path / f"scores/main/{iteration}/{split}/{eid}/{origin}.json")
                    if grade["judge_configuration"] != manifest["roles"]["judge"]:
                        raise ContractError("Grade artifact configuration differs from the frozen judge")
                    grade_payload = audit_saved_output(
                        grade,
                        GRADE_SCHEMA,
                        manifest["roles"]["judge"],
                        manifest["endpoints"]["judge"],
                        manifest["software_hashes"]["prompts/judge.md"],
                    )
                    if split == "validation":
                        for attempt in grade["attempts"]:
                            receipt = read_json(attempt["response"]["raw_response"])
                            if not receipt.get("sent_at") or receipt["sent_at"] < freeze["frozen_at"]:
                                raise ContractError(
                                    "Validation grade was dispatched before trajectory freeze"
                                )
                    expected = {**task_data(e), "rubric": rubric["value"], "candidate": text}
                    if json.loads(grade_payload["messages"][1]["content"]) != expected:
                        raise ContractError("Anonymous grade payload violates input contract")
                    totals[origin] = validate_grade(
                        grade["value"], rubric["value"], text + " " + e["context"]
                    )
                    if grade["total"] != totals[origin] or grade["candidate_hash"] != digest(text):
                        raise ContractError("Grade arithmetic or candidate hash differs")
                rows.append(
                    {
                        "example_id": eid,
                        "paper_id": e["paper_id"],
                        "section_type": e["section_type"],
                        "split": split,
                        "checkpoint": iteration,
                        **totals,
                        "gap": totals["human"] - totals["model"],
                        "length_compliant": candidate["length_compliant"],
                        "length_ratio": candidate["length_ratio"],
                        "contamination_flagged": candidate["contamination"]["flagged"],
                    }
                )
            all_rows[(iteration, split)] = rows
            summaries[split] = summarize(rows, manifest["arguments"]["seed"])
        table.append(
            {
                "iteration": iteration,
                "train_human": summaries["train"]["human"],
                "train_model": summaries["train"]["model"],
                "train_gap": summaries["train"]["gap"],
                "val_human": summaries["validation"]["human"],
                "val_model": summaries["validation"]["model"],
                "val_gap": summaries["validation"]["gap"],
                "selected_by_train": iteration == freeze["selected"],
            }
        )
    selected = max(range(len(table)), key=lambda i: (table[i]["train_gap"], -i))
    if selected != freeze["selected"] or freeze["validation_used_for_selection"] is not False:
        raise ContractError("Selection rule differs")
    if freeze["training_gaps"] != [r["train_gap"] for r in table]:
        raise ContractError("Training-selection ledger differs")
    write_table(path / "scores/rebuilt_checkpoints.csv", table)
    return {
        "manifest": manifest,
        "freeze": freeze,
        "table": table,
        "rows": all_rows,
        "candidates": candidates,
        "raw_verified": True,
    }


def validate_primary_manifest(manifest, design):
    if manifest["roles"]["judge"]["model"] != design["judge"]:
        raise ContractError("Primary matrix must hold the preregistered main judge fixed")
    if design.get("scope") == "meta_blog_initial_empirical_investigation":
        for role in ("writer", "rubric", "optimizer", "judge"):
            if manifest["roles"][role]["model"] != design[role]:
                raise ContractError(f"Blog reproduction requires the declared {role} model")
    for field in ("iterations", "max_meta_prompt_words", "failure_examples"):
        if manifest["arguments"].get(field) != design[field]:
            raise ContractError(f"Primary matrix differs from preregistered {field}")
    if manifest["extra"].get("initial_meta_prompt_hash") != digest(prompt("rubric_initial")):
        raise ContractError("Primary matrix differs from the frozen neutral starting prompt")
    for role, cfg in manifest["roles"].items():
        if cfg != role_config(argparse.Namespace(), role, model=cfg["model"]):
            raise ContractError(f"Primary {role} provider/decoding differs from the frozen configuration")


def render_report(runs_root, output):
    """Report only the declared same-model trajectory; old pilots are historical evidence."""
    design = yaml.safe_load((ROOT / "configs/experiments.yaml").read_text())
    if design.get("scope") != "meta_blog_initial_empirical_investigation":
        raise ContractError("Reporting requires the active blog reproduction design")
    output, source = Path(output), Path(runs_root) / design["research_run"]
    output.mkdir(parents=True, exist_ok=True)
    audit = {
        "scope": design["scope"],
        "expected_trajectories": 1,
        "raw_verified_trajectories": 0,
        "complete": False,
        "source_run": str(source),
        "failures": [],
    }
    lines = [
        "# Meta blog same-model reproduction",
        "",
        "Muse Spark 1.1: writer, rubric generator, judge. Kimi K2.6: optimizer.",
        "",
        "One trajectory; 52 sections from 8 training and 5 validation papers; seven updates.",
        "",
        "The paper IDs, original prompts, and decoding settings were not published in the blog.",
        "This reconstruction uses frozen arXiv author sections and declared prompts/settings.",
        "A reversal measures optimized judging preferences, not objective writing quality.",
        "",
    ]
    comparison = {
        "source_url": design["source"],
        "blog_reported": design["reported_validation"],
        "reproduction": None,
        "exact_source_data_and_prompts_available": False,
    }
    if not (source / "manifest.json").exists():
        audit["state"] = "not_started"
        lines += ["Research has not started. Historical alternative-model pilots are excluded.", ""]
    else:
        manifest = read_json(source / "manifest.json")
        try:
            validate_primary_manifest(manifest, design)
            if (
                manifest["arguments"]["split"] != "research"
                or manifest["arguments"]["seed"] != design["seed"]
            ):
                raise ContractError("Declared research split and trajectory required")
            run = audit_xar_run(source)
        except (ContractError, FileNotFoundError, jsonschema.ValidationError, ValueError) as error:
            audit.update(state="incomplete", failures=[str(error)])
            lines += [
                "Research is incomplete; no verified comparison is available.",
                "",
                f"Audit: {error}",
                "",
            ]
        else:
            table, selected = run["table"], run["freeze"]["selected"]
            initial, final = table[0], table[selected]
            positives = [r["iteration"] for r in table if r["val_gap"] > 0]
            peak = max(table, key=lambda r: (r["val_gap"], -r["iteration"]))
            comparison["reproduction"] = {
                "initial_gap": initial["val_gap"],
                "selected_gap": final["val_gap"],
                "selected_iteration": selected,
                "terminal_gap": table[-1]["val_gap"],
                "first_positive_iteration": positives[0] if positives else None,
                "descriptive_validation_peak_gap": peak["val_gap"],
                "descriptive_validation_peak_iteration": peak["iteration"],
                "validation_peak_used_for_selection": False,
                "initial_human": initial["val_human"],
                "selected_human": final["val_human"],
                "initial_model": initial["val_model"],
                "selected_model": final["val_model"],
                "paired_improvement": paired_improvement(
                    run["rows"][(0, "validation")], run["rows"][(selected, "validation")], design["seed"]
                ),
                "descriptive_reversal": initial["val_gap"] < 0 < final["val_gap"],
            }
            audit.update(state="raw_verified", complete=True, raw_verified_trajectories=1)
            write_table(output / "checkpoints.csv", table)
            lines += [
                f"Raw-verified trajectory: 1/1. Training selected checkpoint {selected}.",
                "",
                f"Validation gap: {initial['val_gap']:.3f} → {final['val_gap']:.3f}.",
                "",
                "Blog reports −4.2 → +2.76; validation crossing at 4 and peak at 5.",
                "",
                "The entire checkpoint curve is reported; the validation maximum is descriptive only.",
                "",
            ]
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            xs = [r["iteration"] for r in table]
            axes[0].plot(xs, [r["train_gap"] for r in table], "--o", label="Train")
            axes[0].plot(xs, [r["val_gap"] for r in table], "-o", label="Validation")
            axes[0].axhline(0, color="black", linewidth=0.7)
            axes[0].set_title("Human minus model score")
            axes[1].plot(xs, [r["val_human"] for r in table], "-o", label="Author")
            axes[1].plot(xs, [r["val_model"] for r in table], "-o", label="Muse Spark 1.1")
            axes[1].set_title("Validation mean scores")
            for axis in axes:
                axis.set_xlabel("Prompt update")
                axis.legend()
            fig.tight_layout()
            fig.savefig(output / "gap_curves.png", dpi=160)
            fig.savefig(output / "gap_curves.svg")
            plt.close(fig)
    if not audit["complete"]:
        for name in ("checkpoints.csv", "gap_curves.png", "gap_curves.svg"):
            (output / name).unlink(missing_ok=True)
    write_json(output / "audit.json", audit)
    write_json(output / "blog_comparison.json", comparison)
    (output / "results.md").write_text("\n".join(lines))
    print(f"Report saved: {output}; blog trajectory verified {audit['raw_verified_trajectories']}/1")


def main_guard(main):
    try:
        main()
    except (ContractError, FileNotFoundError, jsonschema.ValidationError, httpx.HTTPError) as e:
        print(f"STOP: {e}")
        raise SystemExit(2) from None


def audit_proposal(text, examples, initial_prompt, max_words):
    reasons, flags = [], []
    if words(text) > max_words:
        reasons.append("meta_prompt_word_bound")
    lower = text.casefold()
    patterns = {
        "origin_preference": r"(?:prefer|reward|favor|favour|boost)\s+(?:the\s+)?(?:human|expert|original)|"
        r"(?:penaliz|penalis|punish|downscore)\w*\s+(?:the\s+)?(?:model|ai|generated)",
        "origin_detection": r"(?:detect|guess|infer|identify|determine)\s+(?:the\s+)?(?:authorship|origin|provenance)|"
        r"(?:human.written|ai.generated|model.generated)\s+(?:tells|markers|signals)",
        "wrapper_override": r"(?:ignore|override|replace)\s+(?:the\s+)?(?:wrapper|system|grading|schema)|"
        r"(?:weighted\s+(?:mean|average)|unequal\s+weight|score\s+(?:from\s+)?0\s*(?:to|[-–])\s*100)",
    }
    for name, pattern in patterns.items():
        if re.search(pattern, lower):
            reasons.append(name)
    for e in examples:
        metadata = e["provenance"]
        for identifier in [e["paper_id"], metadata["title"], *metadata.get("authors", [])]:
            if len(identifier) >= 5 and identifier.casefold() in lower:
                flags.append({"type": "training_identifier", "paper_id": e["paper_id"], "match": identifier})
        a = lower.split()
        for field in ("reference", "context"):
            b = e[field].casefold().split()
            spans = {tuple(b[i : i + 12]) for i in range(max(0, len(b) - 11))}
            for i in range(max(0, len(a) - 11)):
                span = " ".join(a[i : i + 12])
                if tuple(a[i : i + 12]) in spans and span not in initial_prompt.casefold():
                    flags.append({"type": "copied_training_span", "paper_id": e["paper_id"], "match": span})
                    break
    if flags:
        reasons.append("training_leakage")
    return {
        "accepted": not reasons,
        "reasons": sorted(set(reasons)),
        "flags": flags,
        "words": words(text),
        "policy": "fixed_static_scan_v1",
        "limitation": "Static scan cannot prove semantic absence of provenance heuristics",
    }


def propose_prompt(api, current, feedback, examples, initial, args, iteration):
    directory = Path(args.output_dir) / "feedback" / f"iter_{iteration:02d}"
    write_json(directory / "training.json", feedback, immutable=True)
    data = {"feedback": feedback, "max_meta_prompt_words": args.max_meta_prompt_words}
    attempts = []
    for attempt in range(2):
        instruction = prompt("optimizer")
        if attempt:
            instruction += "\nBOUNDED REPAIR: Fix these proposal violations: " + ", ".join(
                attempts[-1]["audit"]["reasons"]
            )
            data = {
                **data,
                "previous_proposal": attempts[-1]["value"]
                or attempts[-1]["attempts"][-1]["response"]["content"],
            }
        result = api.structured(
            "optimizer",
            instruction,
            data,
            PROPOSAL_SCHEMA,
            {"proposal": iteration, "bounded_attempt": attempt},
            repair=False,
        )
        audit = (
            audit_proposal(result["value"]["prompt"], examples, initial, args.max_meta_prompt_words)
            if result["value"]
            else {"accepted": False, "reasons": ["invalid_format"], "flags": []}
        )
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
    write_json(directory / "proposal.json", record, immutable=True)
    return proposal


def html_text(node, strip_heading=False):
    clone = BeautifulSoup(str(node), "html.parser")
    if strip_heading:
        head = clone.find(re.compile(r"^h[1-6]$"))
        if head:
            head.decompose()
    for math_node in clone.find_all("math"):
        annotation = math_node.find("annotation", attrs={"encoding": "application/x-tex"})
        math_node.replace_with(
            annotation.get_text() if annotation else math_node.get("alttext", math_node.get_text(" "))
        )
    for n in clone.select("script, style, nav, footer, .ltx_ERROR"):
        n.decompose()
    return normalize(clone.get_text(" ", strip=True))


def extract_paper(html, metadata):
    soup = BeautifulSoup(html, "html.parser")
    document = soup.select_one(".ltx_document")
    if document is None:
        raise ContractError("No LaTeXML document")
    abstracts = document.select(".ltx_abstract")
    if len(abstracts) != 1:
        raise ContractError("Missing or duplicated abstract")
    targets = {"abstract": abstracts[0]}
    patterns = {
        "introduction": r"^introduction$",
        "related_work": r"^(related work|related works)$",
        "conclusion": r"^(conclusion|conclusions|conclusion and future work|conclusions and future work)$",
    }
    for section in document.select(".ltx_section"):
        if section.find_parent(class_="ltx_section"):
            continue
        head = section.find(re.compile(r"^h[1-6]$"))
        heading = re.sub(r"^\s*[\d.]+\s*", "", head.get_text(" ", strip=True)).casefold() if head else ""
        for kind, pattern in patterns.items():
            if re.fullmatch(pattern, heading):
                if kind in targets:
                    raise ContractError("Ambiguous target boundary")
                targets[kind] = section
    if set(targets) != set(SECTIONS):
        raise ContractError("Required top-level sections missing")
    if len(document.select(".ltx_bibitem")) < 10:
        raise ContractError("Incomplete or short bibliography")
    if document.select(".ltx_ERROR") or "�" in document.get_text():
        raise ContractError("Extraction debris")
    examples = []
    for kind, target in targets.items():
        reference = html_text(target, strip_heading=True)
        if words(reference) < 60:
            raise ContractError("Target section under 60 words")
        context_document = BeautifulSoup(str(document), "html.parser")
        remove = context_document.find(id=target.get("id"))
        if remove is None:
            raise ContractError("Target lacks unique HTML ID")
        remove.replace_with(f"[Missing {kind.replace('_', ' ')} section]")
        context = html_text(context_document)
        if reference in context:
            raise ContractError("Withheld text remains duplicated in context")
        # Largest optimizer payload uses four full failure papers, candidate pairs, rubrics, evidence.
        count = max(token_count(context, m) for m in ("qwen/qwen3.5-9b", "moonshotai/kimi-k2.6"))
        reference_count = max(token_count(reference, m) for m in ("qwen/qwen3.5-9b", "moonshotai/kimi-k2.6"))
        if math.ceil(1.25 * (4 * count + 8 * reference_count + 16000)) + 16384 > 262144:
            raise ContractError("Conservative four-failure optimizer context bound exceeded")
        examples.append(
            {
                "example_id": metadata["paper_id"] + "_" + kind,
                "paper_id": metadata["paper_id"],
                "section_type": kind,
                "context": context,
                "reference": reference,
                "context_hash": digest(context),
                "reference_hash": digest(reference),
                "target_words": words(reference),
                "provenance": metadata,
                "extraction_checks": {
                    "unique_target": True,
                    "reference_removed": True,
                    "bibliography_items": len(document.select(".ltx_bibitem")),
                    "ocr_debris": False,
                },
            }
        )
    return examples


def prepare_data(args):
    policy = read_json(ROOT / "data/acquisition_policy.json")
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    entries = ET.parse(ROOT / "data/discovery.xml").getroot().findall("a:entry", ns)
    raw_dir = ROOT / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    accepted, exclusions, seen_authors, seen_titles = [], [], set(), set()
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for entry in entries:
            paper_id = entry.find("a:id", ns).text.split("/abs/")[-1]
            title = normalize(entry.find("a:title", ns).text)
            authors = [normalize(a.find("a:name", ns).text) for a in entry.findall("a:author", ns)]
            ref = entry.find("x:journal_ref", ns)
            metadata = {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "source_url": "https://arxiv.org/html/" + paper_id,
                "abstract_url": "https://arxiv.org/abs/" + paper_id,
                "year": int(entry.find("a:published", ns).text[:4]),
                "venue": ref.text if ref is not None else "arXiv preprint; peer review not verified",
                "extraction_version": "latexhtml-v1",
                "retrieved_at": "2026-09-29",
                "redistribution": "not verified; text stays local",
            }
            try:
                if set(authors) & seen_authors or title.casefold() in seen_titles:
                    raise ContractError("Author or manuscript overlap with prior selected paper")
                if re.search(r"rubric|XAR|unslopp", title, re.IGNORECASE):
                    raise ContractError("Subject overlaps rubric optimization")
                path = raw_dir / (paper_id + ".html")
                if not path.exists():
                    response = client.get(metadata["source_url"])
                    response.raise_for_status()
                    path.write_text(response.text)
                    time.sleep(0.25)
                metadata["html_hash"] = file_hash(path)
                extracted = extract_paper(path.read_text(), metadata)
                metadata["inspection"] = {"agent_extraction_check": "passed", "human_review": "pending"}
                accepted.append({"metadata": metadata, "examples": extracted})
                seen_authors.update(authors)
                seen_titles.add(title.casefold())
                print(f"Eligible {len(accepted)}/20: {paper_id} {title}", flush=True)
                if len(accepted) == 20:
                    break
            except (ContractError, httpx.HTTPError) as e:
                exclusions.append({"paper_id": paper_id, "reason": str(e), "before_grading": True})
                print(f"Excluded {paper_id}: {e}", flush=True)
    write_json(ROOT / "data/exclusions.json", exclusions)
    if len(accepted) != 20:
        raise ContractError(
            f"Only {len(accepted)} of 20 papers are eligible; see data/exclusions.json "
            "and add more entries to data/discovery.xml"
        )
    research = accepted[2:15]
    random.Random(20260929).shuffle(research)
    groups = {
        "pilot": accepted[:2],
        "train": research[:8],
        "validation": research[8:],
        "confirmation": accepted[15:],
    }
    splits = {
        "seed": 20260929,
        "policy_hash": file_hash(ROOT / "data/acquisition_policy.json"),
        "papers": {s: [p["metadata"]["paper_id"] for p in papers] for s, papers in groups.items()},
    }
    records = []
    for split, papers in groups.items():
        for paper in papers:
            for e in paper["examples"]:
                e["split"] = split
                records.append(e)
    dataset = "".join(canonical(e) + "\n" for e in records)
    path = ROOT / "data/examples.jsonl"
    if path.exists() and path.read_text() != dataset:
        raise ContractError("Frozen dataset differs; never silently replace splits")
    path.write_text(dataset)
    write_json(ROOT / "data/splits.json", splits, immutable=True)
    write_json(
        ROOT / "data/source_manifest.json",
        {
            "source": policy["source"],
            "source_deviation": policy["source_deviation"],
            "policy_hash": digest(policy),
            "discovery_hash": file_hash(ROOT / "data/discovery.xml"),
            "dataset_hash": file_hash(path),
            "papers": [p["metadata"] for p in accepted],
            "examples": [
                {
                    k: e[k]
                    for k in (
                        "example_id",
                        "paper_id",
                        "section_type",
                        "split",
                        "context_hash",
                        "reference_hash",
                        "target_words",
                    )
                }
                for e in records
            ],
        },
        immutable=True,
    )
    load_examples(path, ROOT / "data/splits.json")
    print("Frozen 80 examples: 8 pilot, 32 train, 20 validation, 20 confirmation")


def prepare_tokenizers():
    manifest = read_json(ROOT / "configs/tokenizers.json")
    for name, cfg in manifest.items():
        subprocess.run(
            [
                "hf",
                "download",
                cfg["repo"],
                *cfg["files"],
                "--revision",
                cfg["revision"],
                "--local-dir",
                str(ROOT / "data/tokenizers" / name),
            ],
            check=True,
        )
        for filename, checksum in cfg["checksums"].items():
            if file_hash(ROOT / "data/tokenizers" / name / filename) != checksum:
                raise ContractError(f"Pinned tokenizer checksum differs: {name}/{filename}")
    print("Pinned official tokenizer checksums verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAR input preparation and read-only preflight")
    parser.add_argument(
        "command", choices=["prepare-data", "prepare-tokenizers", "validate-data", "report", "audit-run"]
    )
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--source-run")
    args = parser.parse_args()
    if args.command == "prepare-data":
        main_guard(lambda: prepare_data(args))
    elif args.command == "prepare-tokenizers":
        main_guard(prepare_tokenizers)
    elif args.command == "validate-data":
        main_guard(
            lambda: print(
                f"Validated {len(load_examples(ROOT / 'data/examples.jsonl', ROOT / 'data/splits.json'))} examples"
            )
        )
    elif args.command == "report":
        main_guard(lambda: render_report(args.runs_root, args.output_dir))
    else:
        main_guard(lambda: print("Raw-verified:", audit_xar_run(args.source_run)["raw_verified"]))
