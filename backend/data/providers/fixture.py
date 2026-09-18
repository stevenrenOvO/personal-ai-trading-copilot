"""Fixture data provider for testing and offline mode."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.data.config import get_settings
from backend.data.providers.base import (
    DataProvider,
    DataProviderError,
    ensure_adj_factor,
    ensure_daily,
    ensure_daily_basic,
    ensure_stk_limit,
    ensure_stock_basic,
    ensure_suspend_d,
    ensure_trade_cal,
)


class FixtureProvider(DataProvider):
    """Provider that reads pre-generated CSV files from a fixture directory."""

    name: str = "fixture"

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = (
            Path(base_dir).resolve()
            if base_dir is not None
            else get_settings().fixture_dir
        )
        if not self._base.is_dir():
            raise DataProviderError(f"Fixture directory not found: {self._base}")

    def _read_fixture(self, filename: str) -> pd.DataFrame:
        path = self._base / filename
        if not path.exists():
            raise DataProviderError(f"Fixture file not found: {path}")
        return pd.read_csv(path, dtype=str, keep_default_na=False)

    def stock_basic(
        self,
        *,
        ts_codes: list[str] | None = None,
        list_status: str | None = None,
    ) -> pd.DataFrame:
        df = self._read_fixture("stock_basic.csv")
        if ts_codes:
            df = df[df["ts_code"].isin(ts_codes)]
        if list_status:
            df = df[df["list_status"] == list_status]
        return ensure_stock_basic(df)

    def daily(
        self,
        *,
        ts_codes: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        df = self._read_fixture("daily.csv")
        if ts_codes:
            df = df[df["ts_code"].isin(ts_codes)]
        if start_date:
            df = df[df["trade_date"] >= start_date]
        if end_date:
            df = df[df["trade_date"] <= end_date]
        return ensure_daily(df)

    def daily_basic(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        df = self._read_fixture("daily_basic.csv")
        if ts_codes:
            df = df[df["ts_code"].isin(ts_codes)]
        if trade_date:
            df = df[df["trade_date"] == trade_date]
        return ensure_daily_basic(df)

    def stk_limit(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        df = self._read_fixture("stk_limit.csv")
        if ts_codes:
            df = df[df["ts_code"].isin(ts_codes)]
        if trade_date:
            df = df[df["trade_date"] == trade_date]
        return ensure_stk_limit(df)

    def trade_cal(
        self,
        *,
        exchange: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        is_open: bool | None = None,
    ) -> pd.DataFrame:
        if not (self._base / "trade_cal.csv").exists():
            return pd.DataFrame(columns=["exchange", "cal_date", "is_open", "pretrade_date"])
        df = self._read_fixture("trade_cal.csv")
        if exchange:
            df = df[df["exchange"] == exchange]
        if start_date:
            df = df[df["cal_date"] >= start_date]
        if end_date:
            df = df[df["cal_date"] <= end_date]
        if is_open is not None:
            df = df[df["is_open"].astype(int) == int(is_open)]
        return ensure_trade_cal(df)

    def adj_factor(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if not (self._base / "adj_factor.csv").exists():
            return pd.DataFrame(columns=["ts_code", "trade_date", "adj_factor"])
        df = self._read_fixture("adj_factor.csv")
        if ts_codes:
            df = df[df["ts_code"].isin(ts_codes)]
        if trade_date:
            df = df[df["trade_date"] == trade_date]
        if start_date:
            df = df[df["trade_date"] >= start_date]
        if end_date:
            df = df[df["trade_date"] <= end_date]
        return ensure_adj_factor(df)

    def suspend_d(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if not (self._base / "suspend_d.csv").exists():
            return pd.DataFrame(
                columns=["ts_code", "trade_date", "suspend_timing", "suspend_type"]
            )
        df = self._read_fixture("suspend_d.csv")
        if ts_codes:
            df = df[df["ts_code"].isin(ts_codes)]
        if trade_date:
            df = df[df["trade_date"] == trade_date]
        if start_date:
            df = df[df["trade_date"] >= start_date]
        if end_date:
            df = df[df["trade_date"] <= end_date]
        return ensure_suspend_d(df)

    def latest_trade_date(self) -> str | None:
        try:
            daily = self.daily()
        except Exception:
            return None
        if daily.empty or "trade_date" not in daily.columns:
            return None
        return str(daily["trade_date"].max())

