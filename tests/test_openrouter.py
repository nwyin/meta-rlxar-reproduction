"""Transport recovery, cost recording, format repair and preflight checks; no external model calls."""

import json

import httpx
import pytest

from xar.openrouter import (
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


def test_timeout_is_never_resent_and_blocks_later_calls(tmp_path):
    sends = []

    def handler(request):
        sends.append(request)
        raise httpx.ReadTimeout("timed out")

    api = mock_api(tmp_path, "writer", handler)
    with pytest.raises(UncertainSend, match="unknown"):
        api.call("writer", "Instructions", {"paper": "first"}, None, "first")
    with pytest.raises(UncertainSend, match="unresolved"):
        api.call("writer", "Instructions", {"paper": "first"}, None, "first")
    with pytest.raises(RunError, match="Not sent"):
        api.call("writer", "Instructions", {"paper": "second"}, None, "second")
    assert len(sends) == 1
    # The timed-out send stays on record as unresolved, and nothing was billed for certain.
    assert api.costs() == {"actual_complete_usd": 0.0, "requests": 0, "unresolved": 1}


def test_costs_sum_the_billed_sends_and_reject_a_missing_cost(tmp_path):
    replies = iter([{"cost": 0.25}, {"cost": 0.5}, {}])

    def handler(request):
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "provider": "Meta",
                "usage": next(replies),
                "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
            },
        )

    api = mock_api(tmp_path, "writer", handler)
    api.call("writer", "Instructions", {"paper": "first"}, None, "first")
    api.call("writer", "Instructions", {"paper": "second"}, None, "second")
    assert api.costs() == {"actual_complete_usd": 0.75, "requests": 2, "unresolved": 0}
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
                "provider": "Alibaba",  # the judge's endpoint
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
