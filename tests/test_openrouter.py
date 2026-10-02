"""Transport recovery, cost recording, format repair and preflight checks; no external model calls."""

import json

import httpx
import pytest
from conftest import PROVIDER_NAMES

from xar import openrouter
from xar.openrouter import (
    BACKOFF_SECONDS,
    MAX_SENDS,
    MODEL_CATALOG,
    PRICING_HEADROOM,
    SNAPSHOTS,
    OpenRouter,
    check_model_allowed,
    endpoint_for,
    endpoints_filename,
    highest_prices,
    role_config,
)
from xar.util import RunError, UncertainSend, read_json

WRITER_PROVIDER = PROVIDER_NAMES[role_config("writer")["provider"]]  # the name OpenRouter reports


def mock_api(tmp_path, role, handler):
    """An OpenRouter client for one role whose HTTP requests go to handler."""
    return OpenRouter(
        tmp_path / "run",
        {role: role_config(role)},
        seed=0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_check_model_allowed_rejects_excluded_families_and_routers():
    for model in ("anthropic/claude-test", "google/gemini-test", "openrouter/auto", "qwen/qwen3.5-9b:free"):
        with pytest.raises(RunError):
            check_model_allowed(model)


def test_a_send_with_no_response_is_recorded_and_sent_again(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(openrouter.time, "sleep", waits.append)
    sends = []

    def handler(request):
        sends.append(request)
        if len(sends) == 1:
            raise httpx.ReadTimeout("timed out")
        if len(sends) == 2:
            raise httpx.ReadError("[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record mac")
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "provider": WRITER_PROVIDER,
                "usage": {"cost": 0.25},
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            },
        )

    api = mock_api(tmp_path, "writer", handler)
    result = api.call("writer", "Instructions", {"paper": "first"}, None, "first")
    assert result["content"] == "ok" and len(sends) == 3
    assert waits == [BACKOFF_SECONDS[0], BACKOFF_SECONDS[1]]
    attempts = [read_json(f) for f in sorted(tmp_path.glob("run/requests/*/attempt_*.json"))]
    assert [a["status"] for a in attempts] == ["uncertain", "uncertain", "success"]
    assert attempts[0]["error"].startswith("ReadTimeout") and attempts[1]["error"].startswith("ReadError")
    # The two sends without a response may have been billed, up to their cost bound each.
    costs = api.costs()
    assert costs["actual_complete_usd"] == 0.25 and costs["requests"] == 1 and costs["unresolved"] == 2
    assert costs["unresolved_upper_usd"] == pytest.approx(2 * attempts[0]["upper_usd"])


def test_resume_sends_again_after_a_send_that_was_in_flight(tmp_path, monkeypatch):
    monkeypatch.setattr(openrouter.time, "sleep", lambda seconds: None)
    api = mock_api(tmp_path, "writer", lambda request: httpx.Response(429, json={"error": {}}))
    with pytest.raises(RunError, match=f"failed {MAX_SENDS} times"):
        api.call("writer", "Instructions", {"paper": "text"}, None, "task")
    directory = next(tmp_path.glob("run/requests/*"))
    # Make the record look like a process that died mid-send: one placeholder without a response.
    for attempt in sorted(directory.glob("attempt_*.json"))[1:]:
        attempt.unlink()
    (directory / "attempt_0.json").write_text(
        json.dumps({"status": "uncertain", "sent_at": "t", "upper_usd": 0.1})
    )

    def handler(request):
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "provider": WRITER_PROVIDER,
                "usage": {"cost": 0.05},
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            },
        )

    resumed = mock_api(tmp_path, "writer", handler)
    result = resumed.call("writer", "Instructions", {"paper": "text"}, None, "task")
    assert result["raw_response"].endswith("attempt_1.json")
    assert resumed.costs() == {
        "actual_complete_usd": 0.05,
        "requests": 1,
        "unresolved": 1,
        "unresolved_upper_usd": 0.1,
    }


def test_costs_sum_the_billed_sends_and_reject_a_missing_cost(tmp_path):
    replies = iter([{"cost": 0.25}, {"cost": 0.5}, {}])

    def handler(request):
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "provider": WRITER_PROVIDER,
                "usage": next(replies),
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            },
        )

    api = mock_api(tmp_path, "writer", handler)
    api.call("writer", "Instructions", {"paper": "first"}, None, "first")
    api.call("writer", "Instructions", {"paper": "second"}, None, "second")
    assert api.costs() == {
        "actual_complete_usd": 0.75,
        "requests": 2,
        "unresolved": 0,
        "unresolved_upper_usd": 0.0,
    }
    with pytest.raises(UncertainSend, match="usage.cost"):
        api.call("writer", "Instructions", {"paper": "third"}, None, "third")


def test_invalid_reply_is_repaired_once_with_the_error(tmp_path):
    systems = []

    def handler(request):
        payload = json.loads(request.content)
        systems.append(payload["messages"][0]["content"])
        content = "not json" if len(systems) == 1 else json.dumps({"answer": "ok"})
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "provider": PROVIDER_NAMES[role_config("judge")["provider"]],
                "usage": {"cost": 0.0001},
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            },
        )

    api = mock_api(tmp_path, "judge", handler)
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": False,
    }
    result = api.structured("judge", "Instructions", {"paper": "text"}, schema, "task")
    assert result["status"] == "valid" and result["value"] == {"answer": "ok"}
    assert [a["status"] for a in result["attempts"]] == ["invalid", "valid"]
    assert "FORMAT REPAIR" in systems[1] and "Expecting value" in systems[1]


@pytest.mark.parametrize(
    "factor,accepted", [(0.5, True), (PRICING_HEADROOM - 0.01, True), (PRICING_HEADROOM + 0.05, False)]
)
def test_preflight_accepts_price_changes_up_to_the_headroom(tmp_path, factor, accepted):
    cfg = role_config("judge")
    endpoint, _ = endpoint_for(cfg)

    def handler(request):
        if request.url.path.endswith("/endpoints"):
            data = read_json(SNAPSHOTS / endpoints_filename(cfg["model"]))
            for e in data["data"]["endpoints"]:
                if e["tag"] == cfg["provider"]:
                    e["pricing"]["completion"] = str(float(e["pricing"]["completion"]) * factor)
        else:
            data = read_json(MODEL_CATALOG)
        return httpx.Response(200, json=data)

    api = mock_api(tmp_path, "judge", handler)
    if accepted:
        directory = api.preflight()
        observed = read_json(directory / "checks.json")["observed_prices"]["judge"]
        pinned = highest_prices(endpoint)
        assert observed["completion"] == pytest.approx(pinned["completion"] * factor)
        assert observed["prompt"] == pinned["prompt"]
        assert api.endpoint["judge"] == endpoint
    else:
        with pytest.raises(RunError, match="completion price .* is above the allowed"):
            api.preflight()


def test_rate_limits_wait_out_the_backoff_or_a_longer_retry_after(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(openrouter.time, "sleep", waits.append)
    replies = iter([(429, {"Retry-After": "20"}), (429, {}), (503, {"Retry-After": "oops"})])

    def handler(request):
        for status, headers in replies:
            return httpx.Response(status, headers=headers, json={"error": {"message": "busy"}})
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "provider": WRITER_PROVIDER,
                "usage": {"cost": 0.01},
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            },
        )

    api = mock_api(tmp_path, "writer", handler)
    result = api.call("writer", "Instructions", {"paper": "text"}, None, "task")
    assert result["content"] == "ok" and result["raw_response"].endswith("attempt_3.json")
    # Retry-After wins over the schedule when it is longer; a non-numeric header is ignored.
    assert waits == [20, BACKOFF_SECONDS[1], BACKOFF_SECONDS[2]]
    statuses = [read_json(f)["status"] for f in sorted(tmp_path.glob("run/requests/*/attempt_*.json"))]
    assert statuses == ["http_error", "http_error", "http_error", "success"]


def test_a_request_fails_after_the_last_retry(tmp_path, monkeypatch):
    waits = []
    monkeypatch.setattr(openrouter.time, "sleep", waits.append)
    api = mock_api(tmp_path, "writer", lambda request: httpx.Response(429, json={"error": {}}))
    with pytest.raises(RunError, match=f"failed {MAX_SENDS} times"):
        api.call("writer", "Instructions", {"paper": "text"}, None, "task")
    assert waits == list(BACKOFF_SECONDS)
    assert openrouter.backoff_seconds(0, "600") == openrouter.MAX_RETRY_AFTER_SECONDS
