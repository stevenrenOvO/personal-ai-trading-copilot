"""Tests for free A-share data providers and shared utilities."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from backend.data.providers._a_share_utils import (
    compute_limit_prices,
    from_source_date,
    limit_rate,
    round_price,
    source_to_ts_code,
    to_source_date,
    ts_to_source_code,
)
from backend.data.providers.fallback import FallbackProvider
from backend.data.providers.storage import StorageProvider
from backend.market.snapshot import _sina_to_ts_code, rank_snapshot


# ---- pure utilities --------------------------------------------------


def test_code_conversion_roundtrip():
    assert ts_to_source_code("600000.SH") == "sh.600000"
    assert source_to_ts_code("sh.600000") == "600000.SH"
    assert source_to_ts_code("sz.000001") == "000001.SZ"


def test_date_conversion():
    assert to_source_date("20240102") == "2024-01-02"
    assert from_source_date("2024-01-02") == "20240102"


def test_limit_rate_by_board_and_st():
    assert limit_rate("600000.SH", False) == 0.10
    assert limit_rate("600000.SH", True) == 0.05
    assert limit_rate("300001.SZ", False) == 0.20
    assert limit_rate("688001.SH", False) == 0.20
    assert limit_rate("830001.BJ", False) == 0.30


def test_round_price_half_up():
    assert round_price(10.105) == 10.11
    assert round_price(10.104) == 10.10


def test_compute_limit_prices():
    raw = pd.DataFrame(
        {
            "trade_date": ["20240102", "20240102"],
            "ts_code": ["600000.SH", "300001.SZ"],
            "pre_close": [10.0, 20.0],
            "is_st": ["0", "0"],
        }
    )
    limits = compute_limit_prices(raw)
    rows = {r["ts_code"]: r for _, r in limits.iterrows()}
    assert rows["600000.SH"]["up_limit"] == 11.0
    assert rows["600000.SH"]["down_limit"] == 9.0
    assert rows["300001.SZ"]["up_limit"] == 24.0


def test_sina_code_conversion():
    assert _sina_to_ts_code("sh600000") == "600000.SH"
    assert _sina_to_ts_code("sz000001") == "000001.SZ"
    assert _sina_to_ts_code("bj920000") == "920000.BJ"


def test_rank_snapshot_sorts_and_filters():
    df = pd.DataFrame(
        {
            "ts_code": ["600000.SH", "000001.SZ", "300001.SZ"],
            "name": ["A", "B", "C"],
            "price": [10.0, 20.0, 30.0],
            "pct_chg": [1.0, 5.0, -2.0],
            "amount": [1e8, 5e8, 3e8],
        }
    )
    ranked = rank_snapshot(df, sort_by="pct_chg", top_n=2)
    assert list(ranked["ts_code"]) == ["000001.SZ", "600000.SH"]
    filtered = rank_snapshot(df, min_amount=4e8)
    assert len(filtered) == 1


# ---- fallback provider ----------------------------------------------


class _FakeProvider:
    name = "fake"

    def __init__(self, *, raise_on: set[str] | None = None, empty_on: set[str] | None = None):
        self.raise_on = raise_on or set()
        self.empty_on = empty_on or set()

    def stock_basic(self, *, ts_codes=None, list_status=None):
        if "stock_basic" in self.raise_on:
            raise RuntimeError("boom")
        if "stock_basic" in self.empty_on:
            return pd.DataFrame()
        return pd.DataFrame({"ts_code": ["000001.SZ"]})

    def daily(self, *, ts_codes=None, start_date=None, end_date=None):
        if "daily" in self.raise_on:
            raise RuntimeError("boom")
        if "daily" in self.empty_on:
            return pd.DataFrame()
        return pd.DataFrame({"trade_date": ["20240102"], "ts_code": ["000001.SZ"]})

    def daily_basic(self, *, ts_codes=None, trade_date=None):
        return pd.DataFrame()

    def stk_limit(self, *, ts_codes=None, trade_date=None):
        return pd.DataFrame()

    def trade_cal(self, *, exchange=None, start_date=None, end_date=None, is_open=None):
        return pd.DataFrame()

    def adj_factor(self, *, ts_codes=None, trade_date=None, start_date=None, end_date=None):
        return pd.DataFrame()

    def suspend_d(self, *, ts_codes=None, trade_date=None, start_date=None, end_date=None):
        return pd.DataFrame()


def test_fallback_provider_uses_second_provider():
    chain = FallbackProvider(
        [
            _FakeProvider(raise_on={"stock_basic"}),
            _FakeProvider(),
        ]
    )
    result = chain.stock_basic()
    assert not result.empty
    assert result.iloc[0]["ts_code"] == "000001.SZ"


def test_fallback_provider_skips_empty_frame():
    chain = FallbackProvider(
        [
            _FakeProvider(empty_on={"daily"}),
            _FakeProvider(),
        ]
    )
    result = chain.daily()
    assert not result.empty


def test_fallback_provider_raises_when_all_fail():
    chain = FallbackProvider(
        [_FakeProvider(raise_on={"stock_basic"}), _FakeProvider(raise_on={"stock_basic"})]
    )
    with pytest.raises(Exception):
        chain.stock_basic()


def test_storage_provider_metadata():
    assert StorageProvider.name == "storage"
    assert StorageProvider.data_timeliness == "historical"


# ---- lazy provider chain --------------------------------------------


def test_lazy_provider_defers_construction_until_used():
    """Choosing the fallback chain must not open a vendor connection."""
    from backend.data.providers.lazy import LazyProvider

    built: list[str] = []

    class _Real:
        name = "real"

        def __init__(self) -> None:
            built.append("constructed")

        def daily(self, *, ts_codes=None, start_date=None, end_date=None):
            return pd.DataFrame({"trade_date": ["20240102"], "ts_code": ["000001.SZ"]})

    lazy = LazyProvider(_Real, name="real")
    assert lazy.name == "real"
    assert lazy.is_connected is False
    assert built == []  # nothing touched the network yet

    frame = lazy.daily()
    assert lazy.is_connected is True
    assert built == ["constructed"]
    assert not frame.empty


def test_lazy_provider_inside_fallback_chain_stays_offline():
    from backend.data.providers.lazy import LazyProvider

    def _boom():
        raise AssertionError("the chain must not connect just by being built")

    class _Second:
        name = "second"

        def daily(self, *, ts_codes=None, start_date=None, end_date=None):
            return pd.DataFrame({"trade_date": ["20240102"], "ts_code": ["000001.SZ"]})

    chain = FallbackProvider([LazyProvider(_boom, name="boom"), _Second()])
    assert not chain.daily().empty


# ---- live providers (opt-in via env var) ----------------------------

live = pytest.mark.skipif(
    os.getenv("RUN_LIVE_DATA_TESTS") != "1",
    reason="set RUN_LIVE_DATA_TESTS=1 to run live free-provider tests",
)


@live
def test_baostock_live_smoke():
    from backend.data.providers.baostock import BaoStockProvider

    provider = BaoStockProvider()
    basic = provider.stock_basic(list_status="L")
    assert not basic.empty
    assert "ts_code" in basic.columns


@live
def test_akshare_live_smoke():
    from backend.data.providers.akshare import AkShareProvider

    provider = AkShareProvider()
    basic = provider.stock_basic()
    assert not basic.empty
