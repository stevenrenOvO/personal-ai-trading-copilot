"""Tests for data providers (using fixture provider)."""

import pytest
import pandas as pd
from pathlib import Path
from backend.data.providers.fixture import FixtureProvider
from backend.data.providers.base import DataProviderError, DataSourceNotConfiguredError
from backend.data.providers.tushare import TushareProvider
from backend.data.config import clear_settings_cache, get_settings


def test_fixture_provider_missing_dir():
    with pytest.raises(DataProviderError, match="Fixture directory not found"):
        FixtureProvider(base_dir=Path("/nonexistent"))


def test_fixture_provider_stock_basic(tmp_path):
    # Create a minimal fixture file
    df = pd.DataFrame({
        "ts_code": ["000001.SZ", "600000.SH"],
        "symbol": ["000001", "600000"],
        "name": ["平安银行", "浦发银行"],
        "list_status": ["L", "L"],
    })
    
    df.to_csv(tmp_path / "stock_basic.csv", index=False, encoding="utf-8")

    provider = FixtureProvider(base_dir=tmp_path)
    result = provider.stock_basic()
    assert len(result) == 2
    assert "ts_code" in result.columns


def test_fixture_provider_daily(tmp_path):
    df = pd.DataFrame({
        "trade_date": ["20240101", "20240102"],
        "ts_code": ["000001.SZ", "000001.SZ"],
        "open": [10.0, 10.5],
        "high": [11.0, 10.8],
        "low": [9.5, 10.2],
        "close": [10.5, 10.6],
        "pre_close": [10.0, 10.5],
        "change": [0.5, 0.1],
        "pct_chg": [5.0, 0.95],
        "vol": [10000, 12000],
        "amount": [100000, 130000],
    })
    
    df.to_csv(tmp_path / "daily.csv", index=False, encoding="utf-8")

    provider = FixtureProvider(base_dir=tmp_path)
    result = provider.daily(ts_codes=["000001.SZ"])
    assert len(result) == 2
    assert "trade_date" in result.columns

    result_filtered = provider.daily(ts_codes=["000001.SZ"], start_date="20240102")
    assert len(result_filtered) == 1


def test_tushare_provider_skips_without_token():
    clear_settings_cache()
    settings = get_settings()
    if settings.has_tushare_token:
        pytest.skip("TUSHARE_TOKEN is configured; skipping missing-token test")
    with pytest.raises(DataSourceNotConfiguredError):
        TushareProvider(token=None)

