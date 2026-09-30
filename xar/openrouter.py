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
    canonical,
    digest,
    file_hash,
    now,
    read_json,
    write_json,
)

API_BASE = "https://openrouter.ai/api/v1"


def model_policy(model):
    if not isinstance(model, str) or "/" not in model:
        raise RunError("Use an explicit OpenRouter model slug")
    family = model.split("/", 1)[0].lower()
    if family in {"anthropic", "google"} or any(s in model.lower() for s in ("claude", "gemini")):
        raise RunError(f"Excluded model family: {model}")
    if ":" in model or model.endswith("/auto"):
        raise RunError("Routing aliases and model variants are not pinned releases")


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


def endpoint_for(cfg):
    path = ROOT / "configs/snapshots" / (cfg["model"].replace("/", "_") + "-endpoints.json")
    data = read_json(path)["data"]
    matches = [e for e in data["endpoints"] if e["tag"] == cfg["provider"]]
    if len(matches) != 1:
        raise RunError(f"Provider must identify exactly one endpoint: {cfg['provider']}")
    endpoint = matches[0]
    for parameter in ("temperature", "reasoning", "max_tokens", "response_format", "structured_outputs"):
        if parameter not in endpoint["supported_parameters"]:
            raise RunError(f"{cfg['provider']} lacks required {parameter}")
    if cfg["max_tokens"] > (endpoint.get("max_completion_tokens") or endpoint["context_length"]):
        raise RunError("Output limit exceeds endpoint capability")
    models = read_json(ROOT / "configs/snapshots/openrouter-models-2026-09-29.json")["data"]
    catalog = next((m for m in models if m["id"] == cfg["model"]), None)
    if catalog is None:
        raise RunError("Model needs a catalog snapshot before execution")
    efforts = catalog.get("reasoning", {}).get("supported_efforts", [])
    if "effort" in cfg["reasoning"] and efforts and cfg["reasoning"]["effort"] not in efforts:
        raise RunError(f"Unsupported reasoning effort; catalog supports {efforts}")
    return endpoint, catalog


PRICING_HEADROOM = 1.25


def highest_prices(endpoint):
    base = endpoint["pricing"]
    return {
        k: max(float(x.get(k, base.get(k, 0))) for x in [base] + base.get("overrides", []))
        for k in ("prompt", "completion", "request")
    }


def budgeted_prices(endpoint):
    return {key: value * PRICING_HEADROOM for key, value in highest_prices(endpoint).items()}


@functools.lru_cache(maxsize=4)
def tokenizer(name):
    cfg = read_json(ROOT / "configs/tokenizers.json")[name]
    filename = "tiktoken.model" if name == "kimi" else "tokenizer.json"
    if file_hash(ROOT / "data/tokenizers" / name / filename) != cfg["checksums"][filename]:
        raise RunError(f"Tokenizer differs from pinned checksum: {name}")
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
    name = {"qwen": "qwen", "moonshotai": "kimi"}.get(model.split("/")[0])
    if name is None:
        # No public tokenizer (Muse Spark): the UTF-8 byte length over-estimates the token count.
        return len(text.encode())
    if name == "kimi":
        return len(tokenizer(name).encode(text, disallowed_special=()))
    return len(tokenizer(name).encode(text, add_special_tokens=False).ids)


TOKEN_HEADROOM = 1.25
CHAT_TEMPLATE_TOKENS = 1024


def max_request_cost(payload, endpoint):
    # Count the whole serialized payload, add headroom and room for the chat template.
    # TOKEN_HEADROOM and PRICING_HEADROOM compound on purpose: each covers a different estimate.
    counted = token_count(canonical(payload), payload["model"])
    tokens = math.ceil(TOKEN_HEADROOM * counted) + CHAT_TEMPLATE_TOKENS
    if tokens + payload["max_tokens"] > endpoint["context_length"]:
        raise RunError("Conservative context bound exceeded; papers cannot be truncated")
    if endpoint.get("max_prompt_tokens") and tokens > endpoint["max_prompt_tokens"]:
        raise RunError("Conservative prompt-token bound exceeded")
    price = budgeted_prices(endpoint)
    return tokens * price["prompt"] + payload["max_tokens"] * price["completion"] + price["request"]


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
    """Cross-process atomic reservations enforce both dollar ceilings, including uncertain sends."""

    def __init__(self, path, run_id, run_limit, total_limit):
        self.path, self.run_id = Path(path), run_id
        self.run_limit, self.total_limit = run_limit, total_limit
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
        with self.locked() as data:
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
            raise RunError("OPENROUTER_API_KEY is not configured; no paid calls dispatched")
        if budget is None or total_budget is None or budget <= 0 or total_budget <= 0:
            raise RunError("Positive per-run and total USD ceilings required")
        self.output, self.roles, self.seed = Path(output), roles, seed
        self.output.mkdir(parents=True, exist_ok=True)
        self.client = client or httpx.Client(timeout=httpx.Timeout(600, connect=30))
        self.endpoints = {r: endpoint_for(c) for r, c in roles.items()}
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
        observed_prices = {}
        for role, cfg in self.roles.items():
            model = cfg["model"]
            if (
                model not in models
                or models[model]["canonical_slug"] != self.endpoints[role][1]["canonical_slug"]
            ):
                raise RunError("Pinned catalog release changed; declare a new batch")
            if model not in endpoints:
                response = self.client.get(API_BASE + "/models/" + model + "/endpoints")
                response.raise_for_status()
                endpoints[model] = response.json()
                write_json(directory / (model.replace("/", "_") + "-endpoints.json"), endpoints[model])
            matching = [e for e in endpoints[model]["data"]["endpoints"] if e["tag"] == cfg["provider"]]
            if len(matching) != 1:
                raise RunError("Pinned provider endpoint missing or ambiguous")
            current, prior = matching[0], self.endpoints[role][0]
            required = {"temperature", "reasoning", "max_tokens", "response_format", "structured_outputs"}
            if not required <= set(current["supported_parameters"]):
                raise RunError("Pinned endpoint no longer supports required parameters")
            if (
                current["context_length"] < prior["context_length"]
                or (current.get("max_completion_tokens") or current["context_length"]) < cfg["max_tokens"]
            ):
                raise RunError("Pinned endpoint limits changed; recheck whole payloads")
            current_price, frozen_upper = highest_prices(current), budgeted_prices(prior)
            if any(current_price[k] > frozen_upper[k] for k in frozen_upper):
                raise RunError(
                    "Endpoint pricing exceeds frozen upper rates; refresh evidence for a new batch"
                )
            observed_prices[role] = current_price
            if current.get("quantization") != prior.get("quantization"):
                raise RunError("Pinned endpoint precision changed; declare a new batch")
            efforts = models[model].get("reasoning", {}).get("supported_efforts", [])
            if efforts and cfg["reasoning"].get("effort", efforts[0]) not in efforts:
                raise RunError("Pinned reasoning mapping changed")
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
        bound = max_request_cost(payload, self.endpoints[role][0])
        # One initial call plus three transport retries. Failed HTTP sends retain a reservation.
        for attempt in range(4):
            receipt = directory / f"attempt_{attempt}.json"
            key = ledger_key(self.output, request_key, attempt)
            if receipt.exists():
                prior = read_json(receipt)
                if prior["status"] == "success":
                    result = self.extract(prior["response"], request_key, key, directory)
                    write_json(cached, result)
                    return result
                if prior["status"] == "uncertain":
                    raise BudgetStop(
                        f"Prior send unresolved: {receipt}; inspect provider history before retry"
                    )
                if not prior.get("retryable"):
                    raise RunError(f"Prior nonretryable transport failure: {receipt}")
                continue
            if self.dispatch_stopped.is_set():
                raise RunError("Dispatch halted after another request failed")
            self.ledger.reserve(key, bound)
            sent_at = now()
            write_json(receipt, {"status": "uncertain", "sent_at": sent_at, "upper_usd": bound})
            headers = {"X-OpenRouter-Title": "Independent XAR reproduction"}
            if self.key:
                headers["Authorization"] = "Bearer " + self.key
            started = time.monotonic()
            try:
                response = self.client.post(
                    API_BASE + "/chat/completions",
                    json=payload,
                    headers=headers,
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
                    result = self.extract(raw, request_key, key, directory)
                    write_json(cached, result)
                    return result
                self.ledger.settle(key, None, "http_error_reserved")
                if not retryable:
                    raise RunError(f"OpenRouter HTTP {response.status_code}; see {receipt}")
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # Connection was never established; no billable completion was sent.
                self.ledger.settle(key, 0, "complete")
                write_json(receipt, {"status": "connect_error", "retryable": True, "timestamp": now()})
            except (httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError):
                self.ledger.settle(key, None, "uncertain")
                raise BudgetStop(
                    f"Send outcome unknown; preserve reservation and reconcile {receipt}"
                ) from None
            if attempt < 3:
                time.sleep(min(2**attempt, 8))
        raise RunError(f"Transport retry allowance exhausted: {directory}")

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
            raise RunError(f"Returned model identity differs: {returned}")
        provider = raw.get("provider")
        request_role = read_json(directory / "request.json")["role"]
        endpoint = self.endpoints[request_role][0]
        if provider and provider not in {endpoint["provider_name"], endpoint["tag"]}:
            raise RunError(f"Returned provider differs: {provider}")
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
                instructions += FORMAT_REPAIR
                instructions += (
                    "Keep the substantive task inputs unchanged. Previous validation error: "
                    + attempts[-1]["error"]
                )
            response = self.call(role, instructions, data, schema, {"task": identity, "format_attempt": i})
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
