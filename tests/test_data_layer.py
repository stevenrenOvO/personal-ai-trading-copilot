"""Tests for calendar utilities and the local ingestion pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.data.calendar import TradingCalendar
from backend.data.pipeline import DataPipeline
from backend.data.providers.fixture import FixtureProvider


def _fixture_provider(tmp_path: Path) -> FixtureProvider:
    fixture_dir = tmp_path / "fixtures"
    fixture_dir.mkdir()
    pd.DataFrame(
        {
            "trade_date": ["20240101", "20240102", "20240103"],
            "ts_code": ["000001.SZ"] * 3,
            "open": [10.0, 10.2, 10.4],
            "high": [10.5, 10.7, 10.9],
            "low": [9.8, 10.0, 10.2],
            "close": [10.0, 10.3, 10.6],
            "pre_close": [10.0, 10.0, 10.3],
            "change": [0.0, 0.3, 0.3],
            "pct_chg": [0.0, 3.0, 2.9],
            "vol": [1000, 1200, 1400],
            "amount": [10000, 12000, 14000],
        }
    ).to_csv(fixture_dir / "daily.csv", index=False)
    return FixtureProvider(base_dir=fixture_dir)


def test_calendar_falls_back_to_daily_dates(tmp_path):
    provider = _fixture_provider(tmp_path)
    cal = TradingCalendar(provider)
    assert cal.is_trading_day("20240102")
    assert cal.previous_trading_day("20240103") == "20240102"
    assert cal.next_trading_day("20240101") == "20240102"
    assert cal.trading_days("20240102") == ["20240102", "20240103"]


def test_pipeline_ingests_and_reloads_daily(tmp_path):
    provider = _fixture_provider(tmp_path)
    pipeline = DataPipeline(provider, root=tmp_path / "store")
    result = pipeline.ingest_daily()
    assert result.quality_ok
    assert result.rows == 3

    loaded = pipeline.load("daily")
    assert len(loaded) == 3
    assert "close" in loaded.columns
