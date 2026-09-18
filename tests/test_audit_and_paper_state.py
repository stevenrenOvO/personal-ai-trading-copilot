"""Regression tests for the V1.0 wrap-up fixes."""

from __future__ import annotations

import pandas as pd

from backend.data.audit import KNOWN_DIFFERENCES, audit
from backend.data.history import MarketHistory
from backend.paper.engine import PaperTradingEngine
from backend.strategies.base import TradeSignal


def test_paper_positions_survive_a_restart(tmp_path):
    path = tmp_path / "paper" / "state.json"
    engine = PaperTradingEngine(initial_cash=100_000.0, state_path=path)
    engine.execute_signal(
        TradeSignal(ts_code="000001.SZ", trade_date="20240103", action="BUY",
                    price=10.0, quantity=1000)
    )
    assert path.exists()

    reloaded = PaperTradingEngine(initial_cash=100_000.0, state_path=path)
    assert reloaded.cash == engine.cash
    assert reloaded.positions["000001.SZ"].quantity == 1000
    assert len(reloaded.orders) == 1


def test_paper_sell_updates_cash_and_position(tmp_path):
    path = tmp_path / "state.json"
    engine = PaperTradingEngine(initial_cash=100_000.0, state_path=path)
    engine.execute_signal(
        TradeSignal(ts_code="000001.SZ", trade_date="20240103", action="BUY",
                    price=10.0, quantity=1000)
    )
    order = engine.execute_signal(
        TradeSignal(ts_code="000001.SZ", trade_date="20240104", action="SELL",
                    price=11.0, quantity=1000)
    )
    assert order.status == "FILLED"
    assert engine.positions["000001.SZ"].quantity == 0
    assert engine.cash > 100_000.0          # realised a gain on the sell
    reloaded = PaperTradingEngine(initial_cash=100_000.0, state_path=path)
    assert reloaded.cash == engine.cash


def test_audit_flags_no_bug_on_clean_data(tmp_path):
    db = tmp_path / "history" / "market_history.db"
    MarketHistory(db_path=db).ingest(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20240103"],
                "open": [10.0], "high": [11.0], "low": [9.9], "close": [10.5],
                "pre_close": [10.0], "change": [0.5], "pct_chg": [5.0],
                "vol": [1000.0], "amount": [10000.0],
            }
        )
    )
    report = audit(db_path=db)
    assert report.ok
    assert report.checks["ohlc_high_below_low"] == 0
    assert report.checks["duplicate_keys"] == 0


def test_audit_detects_our_own_bug(tmp_path):
    db = tmp_path / "history" / "market_history.db"
    MarketHistory(db_path=db).ingest(
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": ["20240103"],
                "open": [10.0], "high": [9.0], "low": [11.0], "close": [10.5],
                "pre_close": [10.0], "change": [0.5], "pct_chg": [5.0],
                "vol": [1000.0], "amount": [10000.0],
            }
        )
    )
    report = audit(db_path=db)
    assert report.ok is False
    assert any("ohlc_high_below_low" in b for b in report.bugs)


def test_known_difference_classes_are_labelled():
    classes = {klass for _item, klass, _note in KNOWN_DIFFERENCES}
    assert "数据源差异" in classes
    assert "统计口径差异" in classes
    assert "复权差异" in classes
    assert "时间差异" in classes
