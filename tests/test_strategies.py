"""Tests for the strategy layer."""

import math

import pandas as pd
import pytest

from backend.strategies.base import StrategyError, prepare_daily
from backend.strategies.indicators import sma
from backend.strategies.ma_cross import MaCrossStrategy


def test_sma_requires_full_window():
    series = pd.Series([1.0, 2.0, 3.0])
    result = sma(series, window=2)

    assert math.isnan(result.iloc[0])
    assert result.iloc[1] == 1.5
    assert result.iloc[2] == 2.5


def test_prepare_daily_orders_and_normalizes():
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH", "000001.SZ"],
            "trade_date": ["20240102", "20240101", "20240101"],
            "close": ["10.5", "8.0", "10.0"],
        }
    )

    result = prepare_daily(daily)

    assert list(result["ts_code"]) == ["000001.SZ", "000001.SZ", "600000.SH"]
    assert list(result["trade_date"]) == ["20240101", "20240102", "20240101"]
    assert list(result["close"]) == [10.0, 10.5, 8.0]


def test_prepare_daily_rejects_missing_columns():
    daily = pd.DataFrame({"trade_date": ["20240101"], "close": [10.0]})
    with pytest.raises(StrategyError, match="ts_code"):
        prepare_daily(daily)


def test_ma_cross_generates_buy_and_sell():
    strategy = MaCrossStrategy(fast_window=1, slow_window=2)
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 5,
            "trade_date": [
                "20240101",
                "20240102",
                "20240103",
                "20240104",
                "20240105",
            ],
            "close": [10.0, 10.0, 10.4, 10.6, 10.4],
        }
    )

    signals = strategy.generate_signals(daily)

    assert [(s.action, s.trade_date, s.price) for s in signals] == [
        ("BUY", "20240103", 10.4),
        ("SELL", "20240105", 10.4),
    ]
    assert signals[0].ts_code == "000001.SZ"
    assert signals[1].ts_code == "000001.SZ"


def test_ma_cross_requires_fast_shorter_than_slow():
    with pytest.raises(ValueError, match="fast_window"):
        MaCrossStrategy(fast_window=5, slow_window=5)

