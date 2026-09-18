"""Provider abstraction for the A-share data foundation layer."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from backend.data.schemas import (
    ADJ_FACTOR,
    DAILY,
    DAILY_BASIC,
    STK_LIMIT,
    STOCK_BASIC,
    SUSPEND_D,
    TRADE_CAL,
    validate_dataframe,
)


class DataProviderError(RuntimeError):
    """Base error for data provider failures."""


class DataSourceNotConfiguredError(DataProviderError):
    """Raised when a provider is used without its required credentials."""


class DataProvider(ABC):
    """Common interface that upper-layer code depends on.

    New data sources only need to implement this class; strategy, signal, and
    decision modules must remain unaware of the underlying vendor.
    """

    name: str = "base"

    @abstractmethod
    def stock_basic(
        self,
        *,
        ts_codes: list[str] | None = None,
        list_status: str | None = None,
    ) -> pd.DataFrame:
        """Return normalized stock basic information."""

    @abstractmethod
    def daily(
        self,
        *,
        ts_codes: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Return normalized daily OHLCV bars."""

    @abstractmethod
    def daily_basic(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        """Return normalized daily basic metrics."""

    @abstractmethod
    def stk_limit(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        """Return normalized price-limit data."""

    def trade_cal(
        self,
        *,
        exchange: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        is_open: bool | None = None,
    ) -> pd.DataFrame:
        """Return normalized trading-calendar data.

        Providers without calendar support return an empty DataFrame rather than
        raising, so upper layers can fall back to deriving open days from daily
        bars when necessary.
        """
        return pd.DataFrame(columns=TRADE_CAL.all_columns)

    def adj_factor(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Return normalized adjusted-price factors."""
        return pd.DataFrame(columns=ADJ_FACTOR.all_columns)

    def suspend_d(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Return normalized suspension records."""
        return pd.DataFrame(columns=SUSPEND_D.all_columns)

    def latest_trade_date(self) -> str | None:
        """Return the latest available trading date, or None if unknown."""
        return None


def ensure_stock_basic(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, STOCK_BASIC.name)


def ensure_daily(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, DAILY.name)


def ensure_daily_basic(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, DAILY_BASIC.name)


def ensure_stk_limit(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, STK_LIMIT.name)


def ensure_trade_cal(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, TRADE_CAL.name)


def ensure_adj_factor(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, ADJ_FACTOR.name)


def ensure_suspend_d(df: pd.DataFrame) -> pd.DataFrame:
    return validate_dataframe(df, SUSPEND_D.name)
