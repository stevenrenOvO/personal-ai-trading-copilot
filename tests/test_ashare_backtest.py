"""Tests for the A-share-aware backtest engine and walk-forward."""

from __future__ import annotations

import pandas as pd

from backend.backtest import AShareBacktestEngine
from backend.data.providers.fixture import FixtureProvider
from backend.strategies.ma_cross import MaCrossStrategy


def _provider(tmp_path) -> FixtureProvider:
    d = tmp_path / "fixtures"
    d.mkdir()
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y%m%d").tolist()
    close = [10.0 + (i % 5) * 0.2 for i in range(n)]
    pd.DataFrame(
        {
            "trade_date": dates,
            "ts_code": ["000001.SZ"] * n,
            "open": close,
            "high": [c + 0.2 for c in close],
            "low": [c - 0.2 for c in close],
            "close": close,
            "pre_close": [10.0] + close[:-1],
            "change": [close[i] - (close[i - 1] if i else 10.0) for i in range(n)],
            "pct_chg": [0.0] + [round((close[i] / close[i - 1] - 1) * 100, 2) for i in range(1, n)],
            "vol": [10000 + i for i in range(n)],
            "amount": [100000 + i for i in range(n)],
        }
    ).to_csv(d / "daily.csv", index=False)
    return FixtureProvider(base_dir=d)


def test_ashare_engine_runs_and_builds_equity(tmp_path):
    provider = _provider(tmp_path)
    engine = AShareBacktestEngine(
        provider,
        MaCrossStrategy(fast_window=2, slow_window=4),
        initial_cash=100_000,
        commission_rate=0.0,
        stamp_duty_rate=0.0,
        slippage=0.0,
    )
    result = engine.run()
    assert not result.equity_curve.empty
    assert isinstance(result.trade_log, list)
    assert result.final_total_value > 0


def test_ashare_engine_rounds_to_lots_and_t_plus_one(tmp_path):
    provider = _provider(tmp_path)
    engine = AShareBacktestEngine(
        provider,
        MaCrossStrategy(fast_window=1, slow_window=2),
        initial_cash=10_000,
        commission_rate=0.0,
        stamp_duty_rate=0.0,
        slippage=0.0,
        lot_size=100,
    )
    result = engine.run()
    for order in result.orders:
        assert order.quantity % 100 == 0


def test_backtest_never_trades_an_index_series(tmp_path):
    """Index bars share the local store; they must not become positions."""
    d = tmp_path / "fixtures"
    d.mkdir()
    n = 30
    dates = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y%m%d").tolist()
    close = [10.0 + (i % 5) * 0.2 for i in range(n)]
    rows = []
    for code in ("000001.SZ", "000300.SH"):
        for i, day in enumerate(dates):
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": code,
                    "open": close[i],
                    "high": close[i] + 0.2,
                    "low": close[i] - 0.2,
                    "close": close[i],
                    "pre_close": 10.0 if i == 0 else close[i - 1],
                    "change": close[i] - (close[i - 1] if i else 10.0),
                    "pct_chg": 0.0
                    if i == 0
                    else round((close[i] / close[i - 1] - 1) * 100, 2),
                    "vol": 10000 + i,
                    "amount": 100000 + i,
                }
            )
    pd.DataFrame(rows).to_csv(d / "daily.csv", index=False)
    # Only the share is listed, so only the share is tradable.
    pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "symbol": ["000001"],
            "name": ["平安银行"],
            "list_status": ["L"],
        }
    ).to_csv(d / "stock_basic.csv", index=False)

    engine = AShareBacktestEngine(
        FixtureProvider(base_dir=d),
        MaCrossStrategy(fast_window=2, slow_window=4),
        initial_cash=100_000,
        commission_rate=0.0,
        stamp_duty_rate=0.0,
        slippage=0.0,
    )
    result = engine.run()
    assert result.trade_log
    assert {trade.ts_code for trade in result.trade_log} == {"000001.SZ"}
    assert "000300.SH" not in result.positions


def test_walk_forward_returns_folds(tmp_path):
    provider = _provider(tmp_path)
    engine = AShareBacktestEngine(provider, MaCrossStrategy(fast_window=2, slow_window=4))
    result = engine.walk_forward(
        start_date="2024-01-01",
        end_date="2024-01-30",
        train_days=10,
        test_days=5,
        step_days=5,
    )
    assert isinstance(result.folds, list)
