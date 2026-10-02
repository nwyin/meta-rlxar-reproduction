"""Model settings, pinned endpoints, pricing, token counts and the OpenRouter client."""

from __future__ import annotations

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
    InvalidOutput,
    RunError,
    Skipped,
    UncertainSend,
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

# Request fields the roles send (response_format only for structured roles); require them all so any
# role can use any pinned endpoint.
REQUIRED_PARAMETERS = {"temperature", "reasoning", "max_tokens", "response_format", "structured_outputs"}


def check_model_allowed(model):
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
    check_model_allowed(model)
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


def check_endpoint_supports(cfg, endpoint, model_info):
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
    efforts = model_info.get("reasoning", {}).get("supported_efforts", [])
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
    model_info = next((m for m in read_json(MODEL_CATALOG)["data"] if m["id"] == cfg["model"]), None)
    if model_info is None:
        raise RunError(f"{cfg['model']} is missing from {MODEL_CATALOG.name}; refresh configs/snapshots")
    check_endpoint_supports(cfg, endpoint, model_info)
    return endpoint, model_info


PRICE_KEYS = ("prompt", "completion", "request")
# Allow price rises of up to 25% after the snapshot was taken.
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


def allowed_prices(endpoint):
    """The highest prices the run accepts: the highest listed price plus PRICING_HEADROOM."""
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
            "rerun `uv run python run.py prepare-tokenizers`"
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
        # Models without a local tokenizer (currently Muse Spark) are counted by UTF-8 bytes, which
        # over-estimates tokens, so cost and context checks stay on the safe side.
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
    price = allowed_prices(endpoint)
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


# Added to the system prompt when a structured reply is retried.
FORMAT_REPAIR = "\nFORMAT REPAIR: Return complete valid JSON matching the schema. "
MAX_REPAIR_ERROR_CHARS = 500  # validation error length pasted into the repair prompt


READ_TIMEOUT_SECONDS = 600
CONNECT_TIMEOUT_SECONDS = 30
MAX_SENDS = 6  # one send plus 5 retries after connection errors or retryable HTTP statuses
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
BACKOFF_SECONDS = (2, 5, 15, 30, 60)  # before the 2nd to 6th send; a longer Retry-After header wins
MAX_RETRY_AFTER_SECONDS = 120


def backoff_seconds(attempt, retry_after=None):
    """Seconds to wait after failed send `attempt`: the schedule, or a longer Retry-After, capped."""
    wait = BACKOFF_SECONDS[attempt]
    if retry_after is not None and retry_after.strip().isdigit():
        wait = max(wait, min(int(retry_after), MAX_RETRY_AFTER_SECONDS))
    return wait


class OpenRouter:
    """OpenRouter client that caches every request, send and billed cost under output/requests.

    Spending is limited by the OpenRouter key's own limit, set on openrouter.ai, not here."""

    def __init__(self, output, roles, seed, client=None):
        load_dotenv(ROOT / ".env")
        self.key = os.getenv("OPENROUTER_API_KEY")
        if not self.key and client is None:
            raise RunError("OPENROUTER_API_KEY is not set; add it to .env")
        self.output = Path(output)
        self.roles = roles
        self.seed = seed
        self.output.mkdir(parents=True, exist_ok=True)
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(READ_TIMEOUT_SECONDS, connect=CONNECT_TIMEOUT_SECONDS)
        )
        self.endpoint = {}
        self.model_info = {}
        for role, cfg in roles.items():
            self.endpoint[role], self.model_info[role] = endpoint_for(cfg)
        self.dispatch_stopped = threading.Event()

    def costs(self):
        """This run's billed cost and request counts, from the saved sends under output/requests.

        Sends that got no response are counted as unresolved, with the most they could have cost."""
        total, requests, unresolved, upper = 0.0, 0, 0, 0.0
        for path in self.output.glob("requests/*/attempt_*.json"):
            sent = read_json(path)
            if sent["status"] == "success":
                total += sent["response"]["usage"]["cost"]
                requests += 1
            elif sent["status"] == "uncertain":
                unresolved += 1
                upper += sent["upper_usd"]
        return {
            "actual_complete_usd": total,
            "requests": requests,
            "unresolved": unresolved,
            "unresolved_upper_usd": upper,
        }

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
            pinned_slug = self.model_info[role]["canonical_slug"]
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

        The request is keyed by its payload and identity. Saved attempt files decide what happens,
        on resume too: a success is reused, and a connection error, a retryable HTTP status or a
        send that got no response is sent again after a wait. A send with no response may have been
        billed, so it stays on record as uncertain with its cost upper bound (costs() sums them).
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
        max_cost = max_request_cost(payload, self.endpoint[role])  # also checks the request fits
        for attempt in range(MAX_SENDS):
            attempt_file = directory / f"attempt_{attempt}.json"
            if attempt_file.exists():
                prior = read_json(attempt_file)
                if prior["status"] == "success":
                    return self._accept(prior["response"], role, attempt_file)
                if prior["status"] != "uncertain" and not prior.get("retryable"):
                    raise RunError(f"An earlier {role} send failed and cannot be retried; see {attempt_file}")
                continue
            if self.dispatch_stopped.is_set():
                raise Skipped(
                    f"Not sent: another request failed, so the {role} request {request_key} was skipped"
                )
            sent_at = now()
            write_json(attempt_file, {"status": "uncertain", "sent_at": sent_at, "upper_usd": max_cost})
            headers = {"X-OpenRouter-Title": "Independent XAR reproduction"}
            if self.key:
                headers["Authorization"] = "Bearer " + self.key
            started = time.monotonic()
            retry_after = None
            try:
                response = self.client.post(API_BASE + "/chat/completions", json=payload, headers=headers)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout):
                # The connection never opened, so nothing was sent or billed.
                write_json(attempt_file, {"status": "connect_error", "retryable": True, "timestamp": now()})
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as error:
                # Sent, but no response came back. OpenRouter may still have billed it, at most
                # upper_usd, so the send stays on record as uncertain and the request goes out again.
                write_json(
                    attempt_file,
                    {
                        "status": "uncertain",
                        "sent_at": sent_at,
                        "upper_usd": max_cost,
                        "error": f"{type(error).__name__}: {error}",
                        "timestamp": now(),
                    },
                )
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
                    return self._accept(raw, role, attempt_file)
                if not retryable:
                    raise RunError(
                        f"OpenRouter returned HTTP {response.status_code} for {role}; see {attempt_file}"
                    )
                retry_after = response.headers.get("Retry-After")
            if attempt < MAX_SENDS - 1:
                wait = backoff_seconds(attempt, retry_after)
                print(f"Retryable failure for {role} (see {attempt_file.name}); waiting {wait} s", flush=True)
                time.sleep(wait)
        raise RunError(f"{role} request failed {MAX_SENDS} times with retryable errors; see {directory}")

    def _accept(self, raw, role, attempt_file):
        """Check a successful response and save it as result.json.

        Fails if the response has no billed cost, or if a model or provider other than the pinned one
        answered.
        """
        usage = raw.get("usage", {})
        cost = usage.get("cost")
        if cost is None or cost < 0 or not math.isfinite(cost):
            raise UncertainSend(
                f"The {role} response reports no valid usage.cost ({cost!r}); check the OpenRouter "
                f"activity log for {attempt_file} before sending more requests"
            )
        model_info = self.model_info[role]
        model = raw.get("model")
        if model not in {model_info["id"], model_info["canonical_slug"]}:
            raise RunError(
                f"{role} response came from model {model}, not {model_info['id']}; see {attempt_file}"
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
            "request_key": attempt_file.parent.name,
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
                attempts.append(
                    {"response": response, "status": "invalid", "error": str(e)[:MAX_REPAIR_ERROR_CHARS]}
                )
        return {"status": "missing", "value": None, "attempts": attempts}
