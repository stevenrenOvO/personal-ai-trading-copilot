"""Tests for the full-market SQLite history provider.

The provider exists so the engines stop describing a 300-stock CSI300 basket
as "the A-share market". These tests pin the parts that are easy to get wrong:
code/date integrity, derived limit prices, ST detection, the calendar derived
from observed bars, and the per-request fallback to the accepted CSV store.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.data.history import MarketHistory
from backend.data.providers.history import HistoryProvider


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "600000.SH", "600000.SH"],
            "trade_date": ["20240102", "20240103", "20240102", "20240103"],
            "open": [10.0, 10.2, 6.6, 6.7],
            "high": [10.5, 10.4, 6.8, 6.9],
            "low": [9.9, 10.1, 6.5, 6.6],
            "close": [10.2, 10.3, 6.65, 6.8],
            "pre_close": [10.0, 10.2, 6.6, 6.65],
            "change": [0.2, 0.1, 0.05, 0.15],
            "pct_chg": [2.0, 0.98, 0.76, 2.26],
            "vol": [1000.0, 1200.0, 800.0, 900.0],
            "amount": [None, None, None, None],
        }
    )


def _basic() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH", "000004.SZ"],
            "symbol": ["000001", "600000", "000004"],
            "name": ["平安银行", "浦发银行", "*ST国华"],
            "area": ["深圳", "上海", "深圳"],
            "industry": ["银行", "银行", "软件"],
            "market": ["主板", "主板", "主板"],
            "list_date": ["19910403", "19991110", "19910114"],
            "list_status": ["L", "L", "L"],
            "delist_date": ["", "", ""],
            "is_st": ["", "", ""],
        }
    )


def _provider(tmp_path: Path) -> HistoryProvider:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _basic().to_csv(tmp_path / "stock_basic.csv", index=False)
    db = tmp_path / "history" / "market_history.db"
    MarketHistory(db_path=db).ingest(_bars())
    return HistoryProvider(base_dir=tmp_path)


def test_daily_reads_full_market_store(tmp_path):
    provider = _provider(tmp_path)
    frame = provider.daily(start_date="20240103", end_date="20240103")
    assert sorted(frame["ts_code"]) == ["000001.SZ", "600000.SH"]
    assert set(frame["trade_date"]) == {"20240103"}
    # Regression: ts_code / trade_date must not be swapped on the way out.
    assert frame["ts_code"].str.endswith((".SZ", ".SH")).all()


def test_daily_filter_by_codes(tmp_path):
    provider = _provider(tmp_path)
    frame = provider.daily(ts_codes=["600000.SH"])
    assert frame["ts_code"].unique().tolist() == ["600000.SH"]
    assert len(frame) == 2


def test_amount_stays_missing_not_zero(tmp_path):
    """Tencent gap-fill days have no turnover; that must not become 0.0."""
    provider = _provider(tmp_path)
    frame = provider.daily(start_date="20240102", end_date="20240102")
    assert frame["amount"].isna().all()


def test_stk_limit_derived_from_pre_close(tmp_path):
    provider = _provider(tmp_path)
    limits = provider.stk_limit(trade_date="20240103")
    row = limits[limits["ts_code"] == "000001.SZ"].iloc[0]
    assert row["up_limit"] == 11.22  # 10.2 * 1.10, half-up rounded
    assert row["down_limit"] == 9.18  # 10.2 * 0.90


def test_stk_limit_flag_detects_st_from_name(tmp_path):
    """BaoStock's is_st column is empty; ST status lives in the name."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    _basic().to_csv(tmp_path / "stock_basic.csv", index=False)
    db = tmp_path / "history" / "market_history.db"
    MarketHistory(db_path=db).ingest(
        pd.DataFrame(
            {
                "ts_code": ["000004.SZ"],
                "trade_date": ["20240103"],
                "open": [5.0],
                "high": [5.25],
                "low": [4.95],
                "close": [5.25],
                "pre_close": [5.0],
                "change": [0.25],
                "pct_chg": [5.0],
                "vol": [100.0],
                "amount": [None],
            }
        )
    )
    provider = HistoryProvider(base_dir=tmp_path)
    limits = provider.stk_limit(trade_date="20240103")
    row = limits.iloc[0]
    assert row["up_limit"] == 5.25  # ST: 5% limit
    assert row["down_limit"] == 4.75


def test_trade_cal_derived_from_observed_bars(tmp_path):
    provider = _provider(tmp_path)
    cal = provider.trade_cal()
    assert set(cal["cal_date"]) == {"20240102", "20240103"}
    assert (cal["is_open"].astype(int) == 1).all()
    window = provider.trade_cal(start_date="20240103")
    assert window["cal_date"].tolist() == ["20240103"]


def test_latest_trade_date_and_coverage(tmp_path):
    provider = _provider(tmp_path)
    assert provider.latest_trade_date() == "20240103"
    coverage = provider.coverage()
    assert coverage["codes"] == 2
    assert coverage["end"] == "20240103"


def test_daily_basic_and_suspend_are_honestly_empty(tmp_path):
    provider = _provider(tmp_path)
    assert provider.daily_basic(trade_date="20240103").empty
    assert provider.suspend_d(trade_date="20240103").empty


def test_fallback_used_only_when_history_has_no_window(tmp_path):
    """Older dates the SQLite store never covered still resolve via the CSV."""

    class _Fallback:
        name = "fallback"
        calls = 0

        def daily(self, *, ts_codes=None, start_date=None, end_date=None):
            type(self).calls += 1
            return pd.DataFrame(
                {
                    "trade_date": ["20190102"],
                    "ts_code": ["000001.SZ"],
                    "open": [1.0],
                    "high": [1.0],
                    "low": [1.0],
                    "close": [1.0],
                    "pre_close": [1.0],
                    "change": [0.0],
                    "pct_chg": [0.0],
                    "vol": [1.0],
                    "amount": [1.0],
                }
            )

        def stk_limit(self, *, ts_codes=None, trade_date=None):
            return pd.DataFrame()

        def stock_basic(self, *, ts_codes=None, list_status=None):
            return _basic()

    tmp_path.mkdir(parents=True, exist_ok=True)
    _basic().to_csv(tmp_path / "stock_basic.csv", index=False)
    MarketHistory(db_path=tmp_path / "history" / "market_history.db").ingest(_bars())
    fallback = _Fallback()
    provider = HistoryProvider(base_dir=tmp_path, fallback=fallback)

    covered = provider.daily(start_date="20240102", end_date="20240103")
    assert sorted(covered["trade_date"].unique()) == ["20240102", "20240103"]
    assert _Fallback.calls == 0

    older = provider.daily(start_date="20190102", end_date="20190102")
    assert older["trade_date"].tolist() == ["20190102"]
    assert _Fallback.calls == 1


def test_provider_is_a_data_provider(tmp_path):
    from backend.data.providers.base import DataProvider

    provider = _provider(tmp_path)
    assert isinstance(provider, DataProvider)
    assert provider.name == "history"
    assert provider.data_timeliness == "historical"


def _wide_provider(tmp_path: Path, codes: int = 150) -> HistoryProvider:
    """A provider with enough codes to be eligible for window memoisation."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    ts_codes = [f"{600000 + i:06d}.SH" for i in range(codes)]
    symbols = [code.split(".")[0] for code in ts_codes]
    pd.DataFrame(
        {
            "ts_code": ts_codes,
            "symbol": symbols,
            "name": [f"股票{i}" for i in range(codes)],
            "area": ["" for _ in ts_codes],
            "industry": ["银行" for _ in ts_codes],
            "market": ["主板" for _ in ts_codes],
            "list_date": ["20100101" for _ in ts_codes],
            "list_status": ["L" for _ in ts_codes],
            "delist_date": ["" for _ in ts_codes],
            "is_st": ["" for _ in ts_codes],
        }
    ).to_csv(tmp_path / "stock_basic.csv", index=False)
    frames = []
    for day in ("20240102", "20240103"):
        frames.append(
            pd.DataFrame(
                {
                    "ts_code": ts_codes,
                    "trade_date": [day] * codes,
                    "open": [10.0] * codes,
                    "high": [10.5] * codes,
                    "low": [9.9] * codes,
                    "close": [10.2] * codes,
                    "pre_close": [10.0] * codes,
                    "change": [0.2] * codes,
                    "pct_chg": [2.0] * codes,
                    "vol": [1000.0] * codes,
                    "amount": [None] * codes,
                }
            )
        )
    MarketHistory(db_path=tmp_path / "history" / "market_history.db").ingest(
        pd.concat(frames, ignore_index=True)
    )
    return HistoryProvider(base_dir=tmp_path)


def test_full_market_window_is_memoised(tmp_path):
    """Repeated full-market reads must not re-query SQLite on every engine."""
    provider = _wide_provider(tmp_path)
    codes = provider.stock_basic(list_status="L")["ts_code"].tolist()

    first = provider.daily(ts_codes=codes, start_date="20240102", end_date="20240103")
    second = provider.daily(ts_codes=codes, start_date="20240102", end_date="20240103")
    assert first is second
    assert len(first) == 300

    provider.clear_cache()
    third = provider.daily(ts_codes=codes, start_date="20240102", end_date="20240103")
    assert third is not first
    assert len(third) == 300


def test_small_symbol_reads_are_not_cached(tmp_path):
    """Per-symbol lookups stay uncached so they can never serve stale bars."""
    provider = _wide_provider(tmp_path)
    first = provider.daily(ts_codes=["600000.SH"], start_date="20240102")
    second = provider.daily(ts_codes=["600000.SH"], start_date="20240102")
    assert first is not second
    assert first.equals(second)
