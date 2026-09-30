"""Budget ledger, transport recovery, format repair and preflight checks; no external model calls."""

import concurrent.futures
import json

import httpx
import pytest

from xar.openrouter import Ledger, OpenRouter, endpoint_for, highest_prices, role_config
from xar.util import ROOT, BudgetStop, RunError, read_json


def mock_api(tmp_path, role, handler):
    """An OpenRouter client for one role whose HTTP requests go to handler."""
    return OpenRouter(
        tmp_path / "run",
        {role: role_config(role)},
        seed=0,
        budget=10,
        total_budget=10,
        ledger_path=tmp_path / "ledger.json",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_reservations_stop_at_run_and_total_budgets(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", "run", 1, 1.5)
    ledger.reserve("request1", 0.75)
    with pytest.raises(BudgetStop):
        ledger.reserve("request2", 0.3)
    ledger.settle("request1", 0.2)
    ledger.reserve("request2", 0.6)
    with pytest.raises(BudgetStop, match="unsettled"):
        ledger.reserve("request2", 0.6)
    second = Ledger(tmp_path / "ledger.json", "run2", 2, 1.5)
    with pytest.raises(BudgetStop):
        second.reserve("request3", 0.8)


def test_overspend_is_saved_before_budget_stop(tmp_path):
    path = tmp_path / "ledger.json"
    ledger = Ledger(path, "run", 10, 10)
    ledger.reserve("request", 1.0)
    with pytest.raises(BudgetStop, match="reservation"):
        ledger.settle("request", 2.0)
    entry = read_json(path)["entries"]["request"]
    assert entry["state"] == "pricing_bound_violation"
    assert entry["charge"] == 2.0
    # Reading the summary does not rewrite the ledger.
    before = path.stat().st_ino  # a save replaces the file with a new one
    assert ledger.summary()["charged_or_reserved_usd"] == 2.0
    assert path.stat().st_ino == before


def test_parallel_reservations_share_one_budget(tmp_path):
    path = tmp_path / "global_ledger.json"

    def reserve(i):
        ledger = Ledger(path, str(i), 1, 1)
        try:
            ledger.reserve(str(i), 0.4)
            return True
        except BudgetStop:
            return False

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(8))) == 2
    entries = read_json(path)["entries"]
    assert sum(e["charge"] for e in entries.values()) == 0.8


def test_timeout_is_never_resent_and_blocks_later_calls(tmp_path):
    sends = []

    def handler(request):
        sends.append(request)
        raise httpx.ReadTimeout("timed out")

    api = mock_api(tmp_path, "writer", handler)
    with pytest.raises(BudgetStop, match="unknown"):
        api.call("writer", "Instructions", {"paper": "first"}, None, "first")
    with pytest.raises(BudgetStop, match="unresolved"):
        api.call("writer", "Instructions", {"paper": "first"}, None, "first")
    with pytest.raises(RunError, match="Dispatch halted"):
        api.call("writer", "Instructions", {"paper": "second"}, None, "second")
    assert len(sends) == 1
    # The timed-out send keeps its reservation.
    assert len(read_json(tmp_path / "ledger.json")["entries"]) == 1


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
                "provider": "Meta",
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


@pytest.mark.parametrize("factor,accepted", [(0.5, True), (1.1, True), (1.3, False)])
def test_preflight_accepts_price_changes_up_to_the_headroom(tmp_path, factor, accepted):
    cfg = role_config("judge")
    endpoint, _ = endpoint_for(cfg)

    def handler(request):
        if request.url.path.endswith("/endpoints"):
            data = read_json(
                ROOT / "configs/snapshots" / (cfg["model"].replace("/", "_") + "-endpoints.json")
            )
            for e in data["data"]["endpoints"]:
                if e["tag"] == cfg["provider"]:
                    e["pricing"]["completion"] = str(float(e["pricing"]["completion"]) * factor)
        else:
            data = read_json(ROOT / "configs/snapshots/openrouter-models-2026-09-29.json")
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
        with pytest.raises(RunError, match="completion price .* is above the budgeted"):
            api.preflight()
    assert not (tmp_path / "ledger.json").exists()
