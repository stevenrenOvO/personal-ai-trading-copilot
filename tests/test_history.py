"""Tests for the full-market SQLite history store."""

from __future__ import annotations

import pandas as pd

from backend.data.history import MarketHistory


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "600000.SH"],
            "trade_date": ["20240102", "20240103", "20240102"],
            "open": [10.0, 10.2, 6.6],
            "high": [10.5, 10.4, 6.7],
            "low": [9.9, 10.1, 6.5],
            "close": [10.2, 10.3, 6.65],
            "pre_close": [10.0, 10.2, 6.6],
            "change": [0.2, 0.1, 0.05],
            "pct_chg": [2.0, 0.98, 0.76],
            "vol": [1000.0, 1200.0, 800.0],
            "amount": [10000.0, 12000.0, 5000.0],
        }
    )


def test_history_ingest_and_load_keeps_columns(tmp_path):
    history = MarketHistory(db_path=tmp_path / "h.db")
    assert history.ingest(_frame()) == 3

    loaded = history.load()
    assert list(loaded.columns) == [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    ]
    # Regression: ts_code and trade_date must not be swapped.
    assert set(loaded["ts_code"]) == {"000001.SZ", "600000.SH"}
    assert set(loaded["trade_date"]) == {"20240102", "20240103"}


def test_history_is_idempotent(tmp_path):
    history = MarketHistory(db_path=tmp_path / "h.db")
    history.ingest(_frame())
    history.ingest(_frame())  # same primary keys -> no duplicate rows
    assert history.coverage()["rows"] == 3


def test_history_filter_and_codes(tmp_path):
    history = MarketHistory(db_path=tmp_path / "h.db")
    history.ingest(_frame())
    assert history.codes_with_data() == {"000001.SZ", "600000.SH"}

    one = history.load(ts_codes=["000001.SZ"], start_date="20240103")
    assert len(one) == 1
    assert one.iloc[0]["trade_date"] == "20240103"

    window = history.load(start_date="20240102", end_date="20240102")
    assert len(window) == 2
