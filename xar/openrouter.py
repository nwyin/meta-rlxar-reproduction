"""Model settings, pinned endpoints, pricing, token counts, the budget ledger and the OpenRouter client."""

from __future__ import annotations

import contextlib
import fcntl
import functools
import json
import math
import os
import threading
import time
from pathlib import Path

import httpx
import jsonschema
import yaml
from dotenv import load_dotenv

from xar.util import (
    ROOT,
    SCHEMA_VERSION,
    BudgetStop,
    InvalidOutput,
    RunError,
    Skipped,
    canonical,
    digest,
    file_hash,
    now,
    read_json,
    write_json,
)

API_BASE = "https://openrouter.ai/api/v1"
SNAPSHOTS = ROOT / "configs/snapshots"
MODEL_CATALOG = SNAPSHOTS / "openrouter-models-2026-09-29.json"

# Every role relies on these request fields, so a pinned endpoint must accept all of them.
REQUIRED_PARAMETERS = {"temperature", "reasoning", "max_tokens", "response_format", "structured_outputs"}


def model_policy(model):
    """Reject anything but a specific OpenRouter model release from an allowed family."""
    if not isinstance(model, str) or "/" not in model:
        raise RunError(f"Model {model!r} is not an OpenRouter slug of the form vendor/model")
    family = model.split("/", 1)[0].lower()
    if family in {"anthropic", "google"} or any(name in model.lower() for name in ("claude", "gemini")):
        raise RunError(f"{model} is from an excluded model family (Anthropic or Google)")
    if ":" in model or model.endswith("/auto"):
        raise RunError(f"{model} is a routing alias or variant; name a specific model release instead")


def role_config(role, model=None):
    """Model, provider and decoding settings for one role, read from configs/models.yaml."""
    config = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    model = model or config["roles"][role]["model"]
    model_policy(model)
    if model not in config["models"]:
        raise RunError(f"{model} has no entry in configs/models.yaml")
    settings = config["models"][model]
    return {
        "model": model,
        "provider": settings["provider"],
        "temperature": config["roles"][role]["temperature"],
        "reasoning": settings["reasoning"],
        "max_tokens": settings["max_tokens"],
    }


def endpoints_filename(model):
    return model.replace("/", "_") + "-endpoints.json"


def find_endpoint(endpoints, provider, source):
    """The one endpoint in an OpenRouter endpoint listing whose tag is provider."""
    matches = [endpoint for endpoint in endpoints if endpoint["tag"] == provider]
    if len(matches) != 1:
        raise RunError(f"Expected one endpoint tagged {provider!r} in {source}, found {len(matches)}")
    return matches[0]


def check_endpoint_supports(cfg, endpoint, catalog):
    """Fail unless the endpoint accepts every request field and setting that cfg sends to it."""
    missing = REQUIRED_PARAMETERS - set(endpoint["supported_parameters"])
    if missing:
        raise RunError(
            f"{cfg['model']} endpoint {endpoint['tag']} does not support {sorted(missing)}; "
            "choose another provider in configs/models.yaml"
        )
    output_limit = endpoint.get("max_completion_tokens") or endpoint["context_length"]
    if cfg["max_tokens"] > output_limit:
        raise RunError(
            f"{cfg['model']} max_tokens is {cfg['max_tokens']}, but endpoint {endpoint['tag']} "
            f"allows at most {output_limit}; lower it in configs/models.yaml"
        )
    efforts = catalog.get("reasoning", {}).get("supported_efforts", [])
    effort = cfg["reasoning"].get("effort")
    if effort is not None and efforts and effort not in efforts:
        raise RunError(
            f"{cfg['model']} does not support reasoning effort {effort!r}; "
            f"set one of {efforts} in configs/models.yaml"
        )


def endpoint_for(cfg):
    """The saved endpoint and catalog entry for a role's model and provider, checked against cfg."""
    path = SNAPSHOTS / endpoints_filename(cfg["model"])
    endpoint = find_endpoint(read_json(path)["data"]["endpoints"], cfg["provider"], path.name)
    catalog = next((m for m in read_json(MODEL_CATALOG)["data"] if m["id"] == cfg["model"]), None)
    if catalog is None:
        raise RunError(f"{cfg['model']} is missing from {MODEL_CATALOG.name}; refresh configs/snapshots")
    check_endpoint_supports(cfg, endpoint, catalog)
    return endpoint, catalog


PRICE_KEYS = ("prompt", "completion", "request")
# Budget for price rises of up to 25% after the snapshot was taken.
PRICING_HEADROOM = 1.25


def highest_prices(endpoint):
    """Highest USD price per unit across the endpoint's base pricing and its conditional overrides."""
    base = endpoint["pricing"]
    tiers = [base, *base.get("overrides", [])]
    prices = {}
    for key in PRICE_KEYS:
        # An override that leaves out a price keeps the base price.
        prices[key] = max(float(tier.get(key, base.get(key, 0))) for tier in tiers)
    return prices


def budgeted_prices(endpoint):
    """The prices the budget assumes: the highest listed price plus PRICING_HEADROOM."""
    return {key: price * PRICING_HEADROOM for key, price in highest_prices(endpoint).items()}


# Local tokenizers in data/tokenizers/, keyed by the vendor part of the model slug.
TOKENIZER_BY_VENDOR = {"qwen": "qwen", "moonshotai": "kimi"}

# Split pattern copied from the official tokenization_kimi.py at the pinned revision, so counting
# Kimi tokens does not need to run the model repository's own code.
KIMI_PATTERN = "|".join(  # noqa: FLY002 -- keep the official pattern's list form
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


@functools.cache
def token_counter(name):
    """A function that counts tokens with the named local tokenizer, after checking its checksum."""
    filename = "tiktoken.model" if name == "kimi" else "tokenizer.json"
    path = ROOT / "data/tokenizers" / name / filename
    if file_hash(path) != read_json(ROOT / "configs/tokenizers.json")[name]["checksums"][filename]:
        raise RunError(
            f"{path} does not match its checksum in configs/tokenizers.json; "
            "rerun run.py prepare-tokenizers"
        )
    if name == "kimi":
        import tiktoken
        from tiktoken.load import load_tiktoken_bpe

        encoding = tiktoken.Encoding(
            name="kimi-k2.6",
            pat_str=KIMI_PATTERN,
            special_tokens={},
            mergeable_ranks=load_tiktoken_bpe(str(path)),
        )
        return lambda text: len(encoding.encode(text, disallowed_special=()))
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False).ids)


def token_count(text, model):
    name = TOKENIZER_BY_VENDOR.get(model.split("/")[0])
    if name is None:
        # Muse Spark has no public tokenizer. Its UTF-8 byte length over-estimates the token count,
        # so budget and context checks stay on the safe side.
        return len(text.encode())
    return token_counter(name)(text)


# The payload is counted locally, which can differ from the provider's count; add a safety margin.
TOKEN_HEADROOM = 1.25
CHAT_TEMPLATE_TOKENS = 1024  # allowance for the provider's chat template


def max_request_cost(payload, endpoint):
    """Worst-case USD cost of one request. Raises if the request might not fit the endpoint.

    TOKEN_HEADROOM and PRICING_HEADROOM compound on purpose: each covers a different estimate.
    """
    counted = token_count(canonical(payload), payload["model"])
    input_tokens = math.ceil(TOKEN_HEADROOM * counted) + CHAT_TEMPLATE_TOKENS
    output_tokens = payload["max_tokens"]
    if input_tokens + output_tokens > endpoint["context_length"]:
        raise RunError(
            f"{payload['model']} request may need {input_tokens + output_tokens} tokens, but endpoint "
            f"{endpoint['tag']} allows {endpoint['context_length']}; papers are never truncated"
        )
    if endpoint.get("max_prompt_tokens") and input_tokens > endpoint["max_prompt_tokens"]:
        raise RunError(
            f"{payload['model']} prompt may need {input_tokens} tokens, but endpoint {endpoint['tag']} "
            f"allows {endpoint['max_prompt_tokens']}"
        )
    price = budgeted_prices(endpoint)
    return input_tokens * price["prompt"] + output_tokens * price["completion"] + price["request"]


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


def ledger_key(output, request_key, attempt):
    """The budget ledger entry for one send of one request in the run at output."""
    return f"{Path(output).resolve()}/{request_key}/{attempt}"


# Added to the system prompt when a structured reply is retried; the audit strips it off again.
FORMAT_REPAIR = "\nFORMAT REPAIR: Return complete valid JSON matching the schema. "


class Ledger:
    """Budget ledger shared by all runs.

    Before each send, a request reserves its worst-case cost; the reservation is replaced by the
    billed cost when the response arrives. A send whose outcome is unknown keeps its reservation.
    A file lock makes updates safe across threads and processes.
    """

    def __init__(self, path, run_id, run_limit, total_limit):
        self.path = Path(path)
        self.run_id = run_id
        self.run_limit = run_limit
        self.total_limit = total_limit
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.thread_lock = threading.Lock()

    @contextlib.contextmanager
    def locked(self):
        """Load the ledger under an exclusive lock without saving it."""
        with self.thread_lock, self.path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)  # released when the lock file closes
            yield read_json(self.path) if self.path.exists() else {"entries": {}}

    @contextlib.contextmanager
    def transaction(self):
        """Load the ledger under the lock; save it only if the block finishes without raising."""
        with self.locked() as data:
            yield data
            write_json(self.path, data)

    def reserve(self, key, amount):
        """Reserve amount USD for the send key, or raise BudgetStop if either budget would be exceeded."""
        with self.transaction() as data:
            if key in data["entries"]:
                raise BudgetStop(
                    f"{key} already has an unsettled ledger entry; check the OpenRouter activity log "
                    "before sending it again"
                )
            entries = list(data["entries"].values())
            spent_all = sum(e["charge"] for e in entries)
            spent_run = sum(e["charge"] for e in entries if e["run_id"] == self.run_id)
            if spent_all + amount > self.total_limit or spent_run + amount > self.run_limit:
                raise BudgetStop(
                    f"Reserving ${amount:.4f} for the next request would exceed a budget: this run has "
                    f"${spent_run:.4f} of ${self.run_limit:.2f} charged or reserved, all runs "
                    f"${spent_all:.4f} of ${self.total_limit:.2f}"
                )
            data["entries"][key] = {
                "run_id": self.run_id,
                "charge": amount,
                "upper": amount,
                "state": "reserved",
                "created_at": now(),
            }

    def settle(self, key, cost, state="complete"):
        """Replace a reservation with the provider's actual cost (None keeps the reservation)."""
        with self.transaction() as data:
            entry = data["entries"][key]
            if cost is not None:
                if cost < 0 or not math.isfinite(cost):
                    raise RunError(f"Provider reported an invalid cost {cost!r} for {key}")
                entry["charge"] = cost
                if cost > entry["upper"]:
                    state = "pricing_bound_violation"
            entry.update(state=state, settled_at=now())
        # Raise only after the transaction has saved the overspend.
        if state == "pricing_bound_violation":
            raise BudgetStop(
                f"{key} cost ${cost:.4f}, more than its ${entry['upper']:.4f} reservation; "
                "refresh the endpoint pricing snapshots before sending more requests"
            )

    def summary(self):
        """This run's charges and request counts; reads the ledger without rewriting it."""
        with self.locked() as data:
            entries = [e for e in data["entries"].values() if e["run_id"] == self.run_id]
            return {
                "charged_or_reserved_usd": sum(e["charge"] for e in entries),
                "actual_complete_usd": sum(e["charge"] for e in entries if e["state"] == "complete"),
                "requests": len(entries),
                "unresolved": sum(e["state"] != "complete" for e in entries),
            }


READ_TIMEOUT_SECONDS = 600
CONNECT_TIMEOUT_SECONDS = 30
MAX_SENDS = 4  # one send plus three retries after connection errors or retryable HTTP statuses
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


class OpenRouter:
    """OpenRouter client that caches every request under output/requests and charges the ledger."""

    def __init__(self, output, roles, seed, budget, total_budget, ledger_path, client=None):
        load_dotenv(ROOT / ".env")
        self.key = os.getenv("OPENROUTER_API_KEY")
        if not self.key and client is None:
            raise RunError("OPENROUTER_API_KEY is not set; add it to .env")
        if budget is None or total_budget is None or budget <= 0 or total_budget <= 0:
            raise RunError(
                f"Budgets must be positive USD amounts; got {budget} for this run and {total_budget} in total"
            )
        self.output = Path(output)
        self.roles = roles
        self.seed = seed
        self.output.mkdir(parents=True, exist_ok=True)
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(READ_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
        )
        self.endpoint = {}
        self.catalog = {}
        for role, cfg in roles.items():
            self.endpoint[role], self.catalog[role] = endpoint_for(cfg)
        self.ledger = Ledger(ledger_path, str(self.output.resolve()), budget, total_budget)
        self.dispatch_stopped = threading.Event()

    def payload(self, role, system, data, schema):
        payload = {
            **routing_fields(self.roles[role]),
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": canonical(data)}],
        }
        response_format = json_schema_format(schema)
        if response_format:
            payload["response_format"] = response_format
        return payload

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
            pinned_slug = self.catalog[role]["canonical_slug"]
            if live_slug != pinned_slug:
                raise RunError(
                    f"{role} model {model} now resolves to {live_slug}, not the pinned {pinned_slug}; "
                    "refresh configs/snapshots and start a new run"
                )
            if model not in live_endpoints:
                live_endpoints[model] = self._get(f"/models/{model}/endpoints")
                write_json(directory / endpoints_filename(model), live_endpoints[model])
            current = find_endpoint(
                live_endpoints[model]["data"]["endpoints"], cfg["provider"], f"the live {model} listing"
            )
            check_endpoint_supports(cfg, current, live_models[model])
            if current["context_length"] < pinned["context_length"]:
                raise RunError(
                    f"{role} endpoint {current['tag']} context shrank from {pinned['context_length']} "
                    f"to {current['context_length']} tokens; refresh configs/snapshots and start a new run"
                )
            prices = highest_prices(current)
            limits = budgeted_prices(pinned)
            for key in PRICE_KEYS:
                if prices[key] > limits[key]:
                    raise RunError(
                        f"{role} endpoint {current['tag']} {key} price {prices[key]} is above the "
                        f"budgeted {limits[key]} (snapshot price plus {PRICING_HEADROOM - 1:.0%}); "
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
        """Send one request for role and return its result, reusing a saved result when there is one.

        Any failure stops this client from sending further requests (see dispatch_stopped).
        """
        try:
            return self._call(role, system, data, schema, identity)
        except BaseException:
            self.dispatch_stopped.set()
            raise

    def _call(self, role, system, data, schema, identity):
        """Send a request at most MAX_SENDS times, recording every send in an attempt file.

        The request is keyed by its payload and identity. Each send reserves its worst-case cost in
        the ledger first. On resume, saved attempt files decide what happens: a success is reused,
        a send with unknown outcome stops the run, and only retryable failures are sent again.
        """
        payload = self.payload(role, system, data, schema)
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
            write_once=True,
        )
        bound = max_request_cost(payload, self.endpoint[role])
        for attempt in range(MAX_SENDS):
            attempt_file = directory / f"attempt_{attempt}.json"
            key = ledger_key(self.output, request_key, attempt)
            if attempt_file.exists():
                prior = read_json(attempt_file)
                if prior["status"] == "success":
                    return self._accept(prior["response"], role, request_key, key, attempt_file)
                if prior["status"] == "uncertain":
                    raise BudgetStop(
                        f"Send outcome unresolved: {attempt_file} may have been billed but has no response; "
                        "check the OpenRouter activity log before retrying"
                    )
                if not prior.get("retryable"):
                    raise RunError(f"An earlier {role} send failed and cannot be retried; see {attempt_file}")
                continue
            if self.dispatch_stopped.is_set():
                raise Skipped(
                    f"Dispatch halted: another request failed, so {role} request {request_key} was not sent"
                )
            self.ledger.reserve(key, bound)
            sent_at = now()
            write_json(attempt_file, {"status": "uncertain", "sent_at": sent_at, "upper_usd": bound})
            headers = {"X-OpenRouter-Title": "Independent XAR reproduction"}
            if self.key:
                headers["Authorization"] = "Bearer " + self.key
            started = time.monotonic()
            try:
                response = self.client.post(API_BASE + "/chat/completions", json=payload, headers=headers)
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # The connection never opened, so nothing was sent or billed.
                self.ledger.settle(key, 0, "complete")
                write_json(attempt_file, {"status": "connect_error", "retryable": True, "timestamp": now()})
            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError):
                self.ledger.settle(key, None, "uncertain")
                raise BudgetStop(
                    f"The {role} request was sent but got no response, so whether it was billed is unknown; "
                    f"its reservation is kept. Check the OpenRouter activity log ({attempt_file})"
                ) from None
            else:
                try:
                    raw = response.json()
                except ValueError:
                    raw = {"raw_body": response.text}
                retryable = response.status_code in RETRYABLE_STATUS
                success = response.status_code == 200 and "choices" in raw and not raw.get("error")
                write_json(
                    attempt_file,
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
                    return self._accept(raw, role, request_key, key, attempt_file)
                # The provider may still bill a failed request, so the reservation stays.
                self.ledger.settle(key, None, "http_error_reserved")
                if not retryable:
                    raise RunError(
                        f"OpenRouter returned HTTP {response.status_code} for {role}; see {attempt_file}"
                    )
            if attempt < MAX_SENDS - 1:
                time.sleep(2**attempt)  # 1, 2 and 4 seconds
        raise RunError(f"{role} request failed {MAX_SENDS} times with retryable errors; see {directory}")

    def _accept(self, raw, role, request_key, ledger_key, attempt_file):
        """Settle the ledger for a successful response and save it as result.json.

        Fails if the response has no billed cost, or if a model or provider other than the pinned one
        answered.
        """
        usage = raw.get("usage", {})
        cost = usage.get("cost")
        if cost is None:
            self.ledger.settle(ledger_key, None, "cost_unknown")
            raise BudgetStop(
                f"The {role} response has no usage.cost, so its reservation is kept; "
                f"check the OpenRouter activity log for {attempt_file} before sending more requests"
            )
        self.ledger.settle(ledger_key, float(cost))
        catalog = self.catalog[role]
        model = raw.get("model")
        if model not in {catalog["id"], catalog["canonical_slug"]}:
            raise RunError(
                f"{role} response came from model {model}, not {catalog['id']}; see {attempt_file}"
            )
        endpoint = self.endpoint[role]
        provider = raw.get("provider")
        if provider and provider not in {endpoint["provider_name"], endpoint["tag"]}:
            raise RunError(
                f"{role} response came from provider {provider}, not {endpoint['provider_name']}; "
                f"see {attempt_file}"
            )
        choice = raw["choices"][0]
        result = {
            "content": choice["message"].get("content") or "",
            "finish_reason": choice.get("finish_reason"),
            "request_key": request_key,
            "response_id": raw.get("id"),
            "usage": usage,
            "model": model,
            "provider": provider,
            "raw_response": str(attempt_file),
        }
        write_json(attempt_file.parent / "result.json", result)
        return result

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
                instructions += (
                    "Keep the substantive task inputs unchanged. Previous validation error: "
                    + attempts[-1]["error"]
                )
            response = self.call(
                role, instructions, data, schema, {"task": identity, "format_attempt": format_attempt}
            )
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
                attempts.append({"response": response, "status": "invalid", "error": str(e)[:500]})
        return {"status": "missing", "value": None, "attempts": attempts}
