"""Tests for backtesting engine."""

import pytest
import pandas as pd
from pathlib import Path

from backend.data.providers.fixture import FixtureProvider
from backend.strategies.ma_cross import MaCrossStrategy
from backend.backtest.engine import BacktestEngine
from backend.backtest.metrics import calculate_metrics


def test_backtest_engine_run(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()

    daily = pd.DataFrame({
        "trade_date": ["20240101", "20240102", "20240103", "20240104", "20240105"],
        "ts_code": ["000001.SZ"] * 5,
        "open": [10.0, 10.2, 10.4, 10.6, 10.4],
        "high": [10.5, 10.7, 10.9, 11.0, 10.8],
        "low": [9.8, 10.0, 10.2, 10.4, 10.2],
        "close": [10.0, 10.0, 10.5, 9.5, 9.5],
        "pre_close": [10.0, 10.0, 10.2, 10.4, 10.6],
        "change": [0.0, 0.2, 0.2, 0.2, -0.2],
        "pct_chg": [0.0, 2.0, 1.96, 1.92, -1.89],
        "vol": [10000, 12000, 15000, 14000, 13000],
        "amount": [100000, 122000, 156000, 148000, 135000],
    })
    daily.to_csv(fixture_dir / "daily.csv", index=False, encoding="utf-8")

    provider = FixtureProvider(base_dir=fixture_dir)
    strategy = MaCrossStrategy(fast_window=1, slow_window=2)
    engine = BacktestEngine(provider, strategy, initial_cash=10000.0, commission_rate=0.0, slippage=0.0)

    result = engine.run()

    assert len(result.orders) > 0
    assert result.final_total_value > 0
    assert len(result.equity_curve) == len(daily)

    metrics = calculate_metrics(result.equity_curve)
    assert "total_return" in metrics
    assert "max_drawdown" in metrics


def test_backtest_engine_no_signals(tmp_path):
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    daily = pd.DataFrame({
        "trade_date": ["20240101", "20240102"],
        "ts_code": ["000001.SZ"] * 2,
        "open": [10.0, 10.0],
        "high": [10.0, 10.0],
        "low": [10.0, 10.0],
        "close": [10.0, 10.0],
        "pre_close": [10.0, 10.0],
        "change": [0.0, 0.0],
        "pct_chg": [0.0, 0.0],
        "vol": [0, 0],
        "amount": [0, 0],
    })
    daily.to_csv(fixture_dir / "daily.csv", index=False, encoding="utf-8")

    provider = FixtureProvider(base_dir=fixture_dir)
    strategy = MaCrossStrategy(fast_window=5, slow_window=10)
    engine = BacktestEngine(provider, strategy, initial_cash=10000.0)
    result = engine.run()
    assert len(result.orders) == 0
    assert result.final_cash == 10000.0
    assert result.final_total_value == 10000.0


def test_metrics_calculation():
    equity = pd.DataFrame({
        "date": ["20240101", "20240102", "20240103"],
        "total": [10000.0, 11000.0, 9500.0],
    })
    metrics = calculate_metrics(equity, risk_free_rate=0.02)
    assert metrics["total_return"] == pytest.approx(-0.05, abs=1e-9)
    assert metrics["max_drawdown"] == pytest.approx(-0.1363636, abs=1e-6)
    assert metrics["total_days"] == 3

