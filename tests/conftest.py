"""Pytest configuration and shared fixtures."""

import os
from pathlib import Path

import pandas as pd
import pytest

from backend.data.config import clear_settings_cache, get_settings


@pytest.fixture(autouse=True)
def reset_settings():
    clear_settings_cache()
    yield
    clear_settings_cache()


@pytest.fixture(autouse=True)
def offline_market_services(tmp_path_factory):
    """Keep tests offline: no network calendar fetch, no shared quote cache."""
    from datetime import datetime

    from backend.market.quote_service import QuoteService, set_quote_service
    from backend.market.session import TradingSession, set_trading_session

    base = tmp_path_factory.mktemp("market-services")
    days = [
        "20231229", "20240101", "20240102", "20240103", "20240104", "20240105",
        "20260601", "20260908", "20260909", "20260916", "20260917", "20260918",
    ]
    session = TradingSession(
        db_path=base / "session.db",
        calendar_fetcher=lambda start, end: [(d, 1) for d in days],
        now_fn=lambda: datetime(2024, 1, 3, 10, 0),
    )
    set_trading_session(session)

    def empty_fetch():
        return pd.DataFrame(), {"source": "test", "fetched_at": "", "rows": 0}

    set_quote_service(QuoteService(db_path=base / "quotes.db", fetcher=empty_fetch))
    yield
    set_trading_session(None)
    set_quote_service(None)


@pytest.fixture
def sample_stock_basic() -> pd.DataFrame:
    return pd.DataFrame({
        "ts_code": ["000001.SZ", "600000.SH"],
        "symbol": ["000001", "600000"],
        "name": ["平安银行", "浦发银行"],
        "area": ["深圳", "上海"],
        "industry": ["银行", "银行"],
        "market": ["主板", "主板"],
        "list_date": ["19910403", "19991110"],
        "list_status": ["L", "L"],
        "is_st": ["N", "N"],
    })


@pytest.fixture
def sample_daily() -> pd.DataFrame:
    return pd.DataFrame({
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


@pytest.fixture
def sample_daily_basic() -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": ["20240101", "20240102"],
        "ts_code": ["000001.SZ", "000001.SZ"],
        "turnover_rate": [0.5, 0.6],
        "pe": [5.0, 5.5],
        "pb": [0.8, 0.9],
        "total_share": [1000000, 1000000],
        "float_share": [500000, 500000],
    })


@pytest.fixture
def sample_stk_limit() -> pd.DataFrame:
    return pd.DataFrame({
        "trade_date": ["20240101", "20240102"],
        "ts_code": ["000001.SZ", "000001.SZ"],
        "up_limit": [11.0, 11.66],
        "down_limit": [9.0, 9.45],
    })
