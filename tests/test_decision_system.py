"""Tests for risk decisions, strategy health, journal audit and profile."""

from __future__ import annotations

import tempfile
from pathlib import Path

from backend.journal.audit import TradeAuditor
from backend.journal.trades import TradeJournal, TradeRecord
from backend.profile.engine import ProfileEngine
from backend.risk.rules import DecisionRiskEngine, RiskContext
from backend.strategies.health import StrategyRegistry


def test_decision_risk_engine_outputs_action():
    decision = DecisionRiskEngine().decide(
        RiskContext(equity=100_000, cash=100_000, peak_equity=100_000),
        has_positions=False,
    )
    assert decision.action in {"BUY", "HOLD", "WAIT", "REDUCE", "EXIT"}
    assert 0 <= decision.score <= 100


def test_strategy_registry_enable_disable(tmp_path):
    registry = StrategyRegistry(Path(tmp_path))
    health = registry.update_performance(
        "strong_sector_breakout",
        "1.0",
        signals=3,
        trades=[0.02, -0.01, 0.03],
    )
    assert health.win_rate == 2 / 3
    registry.set_enabled("strong_sector_breakout", "1.0", False)
    assert registry.store.get("strong_sector_breakout", "1.0").enabled is False


def _trade(**overrides) -> TradeRecord:
    values = {
        "symbol": "000001",
        "ts_code": "000001.SZ",
        "action": "BUY",
        "price": 10.0,
        "quantity": 100,
        "position_before": 0,
        "position_after": 100,
        "strategy_id": "strong_sector_breakout",
        "strategy_version": "1.0",
        "signal_id": "sig_1",
        "market_state": "弱势",
        "emotion_state": "退潮",
        "sector_state": "",
        "stock_state": "",
        "system_recommendation": "AVOID",
        "user_reason": "追高",
        "execution_reason": "manual",
    }
    values.update(overrides)
    return TradeRecord.create(**values)


def test_journal_and_audit(tmp_path):
    journal = TradeJournal(Path(tmp_path))
    journal.record(_trade())
    trades = journal.list()
    assert len(trades) == 1

    audit = TradeAuditor().audit(trades[0])
    types = {f.error_type for f in audit.findings}
    assert "Emotional Error" in types
    assert "Execution Error" in types


def test_profile_computes_from_trades(tmp_path):
    journal = TradeJournal(Path(tmp_path))
    for pnl in [100, -50, 200, 30, -20, 80, 40, -10, 60, 25]:
        journal.record(
            _trade(
                action="SELL",
                market_state="中性",
                emotion_state="发酵",
                system_recommendation="BUY WATCH",
                result={"pnl": pnl, "return_pct": pnl / 1000, "holding_days": 3},
            )
        )

    engine = ProfileEngine(Path(tmp_path))
    profile = engine.load_or_create()
    refreshed = engine.refresh_stats_from_trades(profile)
    assert refreshed.stats.total_trades == 10
    assert refreshed.stats.sample_sufficient is True
    assert refreshed.stats.profit_factor > 0
