"""Structured copilot results must use the same plan and unknown-data gates as the UI."""
from copy import deepcopy

import pytest

from backend.ai.agent import AICopilot
from backend.api import v1
from tests.test_api_v1 import _call, _make_runtime


def _context():
    return {"date": "20261023", "market": {"date": "20261023", "state": "强势", "score": 75},
        "emotion": {"date": "20261023", "emotion_cycle": "高潮", "recommended_exposure": .8, "market_score": 94},
        "sectors": [], "risk": {}, "board": {"grade": "B", "limit_up_count": 20},
        "trade_plan": {"date": "20261023", "available": True, "can_attack": False,
            "total_exposure_cap": .3, "attack_note": "等待确认，当前仅观察", "plans": [
                {"ts_code": "000001.SZ", "name": "平安银行", "score": 74, "tier": "B", "state": "观察",
                 "why_not_buy": "尚未满足触发条件"}]},
        "data_meta": {"market": {"source": "tencent", "kind": "daily_snapshot"}, "trade_plan": {"status": "ok"}}}


def test_runtime_copilot_reads_main_plan_and_never_calls_legacy_scan(tmp_path, monkeypatch):
    runtime = _make_runtime(tmp_path)
    monkeypatch.setattr(v1, "_opportunities", lambda *args: pytest.fail("legacy opportunities must not drive the copilot"))
    _, expected = _call("GET", "/opportunities/plan?date=20240103", runtime)
    copilot = AICopilot(provider=runtime.provider, runtime=runtime)
    actual = copilot.ask("看看机会", "20240103")["data"]
    assert actual["plans"] == expected["data"]["plans"]
    assert actual["can_attack"] == expected["data"]["can_attack"]
    assert actual["basis_date"] == "20240103"
    assert actual["source"] == "机会页交易计划"
    briefing = copilot.daily_briefing("20240103")
    assert briefing["recommended_exposure"] == expected["data"]["total_exposure_cap"]


def test_market_tool_preserves_snapshot_score_but_uses_plan_permission(monkeypatch):
    ctx = _context()
    before = deepcopy(ctx)
    copilot = AICopilot()
    monkeypatch.setattr(copilot, "_ctx", lambda date: ctx)
    result = copilot.get_market_state("20261023").data
    assert result["score"] == 75 and result["can_attack"] is False
    assert result["source"] == "tencent" and result["data_kind"] == "行情快照"
    assert result["analysis_basis_date"] == "20261023"
    assert ctx == before  # Rendering metadata must never mutate a cached plan.
    briefing = copilot.daily_briefing("20261023")
    assert briefing["can_trade_today"] is False
    assert briefing["recommended_exposure"] == .3
    assert briefing["top_opportunities"][0]["suggested_action"] == "观察"
    assert briefing["top_opportunities"][0]["reason"] == "尚未满足触发条件"


@pytest.mark.parametrize("change", ["missing_plan", "old_plan", "unavailable", "historical_only", "no_data", "old_emotion"])
def test_unconfirmed_plan_returns_unknown_and_no_actionable_briefing(change, monkeypatch):
    ctx = _context()
    if change == "missing_plan":
        del ctx["trade_plan"]["can_attack"]
    elif change == "old_plan":
        ctx["trade_plan"]["date"] = "20261022"
    elif change == "unavailable":
        ctx["trade_plan"]["available"] = False
    elif change == "historical_only":
        ctx["trade_plan"]["historical_only"] = True
    elif change == "no_data":
        ctx["data_meta"]["trade_plan"]["status"] = "no_data"
    else:
        ctx["emotion"]["date"] = "20261022"
    copilot = AICopilot()
    monkeypatch.setattr(copilot, "_ctx", lambda date: ctx)
    assert copilot.get_opportunities("20261023").data["can_attack"] is None
    assert copilot.get_market_state("20261023").data["can_attack"] is None
    briefing = copilot.daily_briefing("20261023")
    assert briefing["can_trade_today"] is None
    assert briefing["recommended_exposure"] is None
    assert briefing["top_opportunities"] == []


def test_supported_audit_and_strategy_questions_route_to_correct_tools(monkeypatch):
    copilot = AICopilot()
    monkeypatch.setattr(copilot, "_ctx", lambda date: {"audit": {"findings": []}, "strategies": []})
    assert copilot.ask("审计交易", "20261023")["tool_calls"] == ["audit_trade"]
    assert copilot.ask("策略健康", "20261023")["tool_calls"] == ["get_strategy_health"]
