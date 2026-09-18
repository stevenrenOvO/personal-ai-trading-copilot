"""Regression tests for V1.0 audit fixes (strategy range, backtest rules)."""

from __future__ import annotations

import pandas as pd
import pytest

from backend.backtest.ashare_engine import AShareBacktestEngine
from backend.data.providers.fixture import FixtureProvider
from backend.emotion.engine import EmotionEngine
from backend.market.engine import MarketState
from backend.strategies.base import Strategy
from backend.strategies.strong_sector_breakout import StrongSectorBreakoutStrategy


def _write_daily(tmp_path, rows: list[dict]) -> FixtureProvider:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    pd.DataFrame(rows).to_csv(fixture_dir / "daily.csv", index=False, encoding="utf-8")
    return FixtureProvider(base_dir=fixture_dir)


class _BuyOnDateStrategy(Strategy):
    """Deterministic strategy: emits one BUY on a fixed date."""

    name = "buy_on_date"
    strategy_id = "buy_on_date"
    version = "1.0"

    def __init__(self, buy_date: str) -> None:
        self.buy_date = buy_date

    def generate_signals(self, daily: pd.DataFrame) -> list:
        signals = []
        for code in sorted(daily["ts_code"].astype(str).unique()):
            signals.append(
                self.make_signal(
                    ts_code=code,
                    trade_date=self.buy_date,
                    action="BUY",
                    price=10.0,
                    reason="test buy",
                )
            )
        return signals


def test_breakout_strategy_emits_signals_across_multiple_dates():
    """The primary strategy must produce a signal series, not just the last bar."""
    dates = pd.date_range("2024-01-01", periods=40, freq="D").strftime("%Y%m%d").tolist()
    rows = []
    for i, d in enumerate(dates):
        # Two breakout days far apart, each with a volume spike.
        breakout = i in (25, 35)
        close = 11.5 if breakout else 10.0
        vol = 300.0 if breakout else 100.0
        rows.append(
            {
                "trade_date": d,
                "ts_code": "000001.SZ",
                "open": 10.0,
                "high": close,
                "low": 9.5,
                "close": close,
                "pre_close": 10.0,
                "change": close - 10.0,
                "pct_chg": (close - 10.0) / 10.0 * 100,
                "vol": vol,
                "amount": 100000.0,
            }
        )
    daily = pd.DataFrame(rows)
    signals = StrongSectorBreakoutStrategy(lookback=5, min_volume_ratio=1.2).generate_signals(daily)
    signal_dates = sorted({s.trade_date for s in signals})
    assert len(signal_dates) >= 2, f"expected multiple signal dates, got {signal_dates}"
    assert dates[25] in signal_dates and dates[35] in signal_dates


def test_backtest_fills_next_open_without_lookahead(tmp_path):
    provider = _write_daily(
        tmp_path,
        [
            {"trade_date": "20240102", "ts_code": "000001.SZ", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0, "pre_close": 10.0, "change": 0.0, "pct_chg": 0.0, "vol": 1000.0, "amount": 10000.0},
            {"trade_date": "20240103", "ts_code": "000001.SZ", "open": 10.1, "high": 10.3, "low": 10.0, "close": 10.2, "pre_close": 10.0, "change": 0.2, "pct_chg": 2.0, "vol": 1000.0, "amount": 10200.0},
            {"trade_date": "20240104", "ts_code": "000001.SZ", "open": 10.4, "high": 10.6, "low": 10.3, "close": 10.5, "pre_close": 10.2, "change": 0.3, "pct_chg": 2.94, "vol": 1000.0, "amount": 10500.0},
        ],
    )
    engine = AShareBacktestEngine(
        provider,
        _BuyOnDateStrategy("20240102"),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_duty_rate=0.0,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
    )
    result = engine.run()
    assert result.orders, "expected an executed order"
    order = result.orders[0]
    assert order.trade_date == "20240103", "signal must fill on the NEXT trading day"
    assert order.price == pytest.approx(10.1), "fill price must be the next day open"


def test_backtest_position_sizing_respects_weight(tmp_path):
    provider = _write_daily(
        tmp_path,
        [
            {"trade_date": "20240102", "ts_code": "000001.SZ", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0, "pre_close": 10.0, "change": 0.0, "pct_chg": 0.0, "vol": 1000.0, "amount": 10000.0},
            {"trade_date": "20240103", "ts_code": "000001.SZ", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0, "pre_close": 10.0, "change": 0.0, "pct_chg": 0.0, "vol": 1000.0, "amount": 10000.0},
        ],
    )
    engine = AShareBacktestEngine(
        provider,
        _BuyOnDateStrategy("20240102"),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_duty_rate=0.0,
        max_position_weight=0.2,
        stop_loss_pct=0.0,
        take_profit_pct=0.0,
    )
    result = engine.run()
    assert result.orders
    order = result.orders[0]
    # 20% of 100k at 10 yuan -> at most 2000 shares (100-share lots)
    assert order.quantity <= 2000
    assert order.quantity >= 1900


def test_backtest_stop_loss_closes_trade(tmp_path):
    provider = _write_daily(
        tmp_path,
        [
            {"trade_date": "20240102", "ts_code": "000001.SZ", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0, "pre_close": 10.0, "change": 0.0, "pct_chg": 0.0, "vol": 1000.0, "amount": 10000.0},
            {"trade_date": "20240103", "ts_code": "000001.SZ", "open": 10.0, "high": 10.2, "low": 9.9, "close": 10.0, "pre_close": 10.0, "change": 0.0, "pct_chg": 0.0, "vol": 1000.0, "amount": 10000.0},
            {"trade_date": "20240104", "ts_code": "000001.SZ", "open": 9.0, "high": 9.2, "low": 8.8, "close": 9.0, "pre_close": 10.0, "change": -1.0, "pct_chg": -10.0, "vol": 1000.0, "amount": 9000.0},
        ],
    )
    engine = AShareBacktestEngine(
        provider,
        _BuyOnDateStrategy("20240102"),
        initial_cash=100000.0,
        commission_rate=0.0,
        stamp_duty_rate=0.0,
        stop_loss_pct=0.08,
        take_profit_pct=0.0,
    )
    result = engine.run()
    assert result.trade_log, "stop-loss should have closed the trade"
    assert result.trade_log[0].pnl < 0


def test_emotion_accepts_external_market_state():
    class _NoopProvider:
        name = "noop"

        def stock_basic(self, **kwargs):
            return pd.DataFrame()

        def daily(self, **kwargs):
            return pd.DataFrame()

        def stk_limit(self, **kwargs):
            return pd.DataFrame()

        def suspend_d(self, **kwargs):
            return pd.DataFrame()

        def trade_cal(self, **kwargs):
            return pd.DataFrame()

    market = MarketState(date="20240103", score=80.0, limit_up_count=30, limit_down_count=2, broken_count=1, breadth=0.7)
    state = EmotionEngine(_NoopProvider()).calculate("20240103", market=market)
    assert state.date == "20240103"
    assert state.market_score == 80.0
    # Step 3: the stage is decided by ladder structure, not by the market
    # score. With no structure available the engine must refuse to judge
    # instead of falling back to a score threshold.
    assert state.emotion_cycle == "数据不足"
    assert state.recommended_exposure == 0.0
    assert state.unavailable
