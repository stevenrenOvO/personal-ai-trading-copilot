"""Tushare data provider implementation."""

from __future__ import annotations

import pandas as pd
import tushare as ts

from backend.data.config import get_settings
from backend.data.providers.base import (
    DataProvider,
    DataProviderError,
    DataSourceNotConfiguredError,
    ensure_adj_factor,
    ensure_daily,
    ensure_daily_basic,
    ensure_stk_limit,
    ensure_stock_basic,
    ensure_suspend_d,
    ensure_trade_cal,
)
from backend.data.schemas import normalize_ts_codes


class TushareProvider(DataProvider):
    """Provider that fetches real data from Tushare."""

    name: str = "tushare"

    def __init__(self, token: str | None = None) -> None:
        self._token = token or get_settings().tushare_token
        if not self._token or self._token.strip() in ("", "your_token_here"):
            raise DataSourceNotConfiguredError(
                "Tushare token missing. Set TUSHARE_TOKEN in environment or .env"
            )
        ts.set_token(self._token)
        self._pro = ts.pro_api()

    def stock_basic(
        self,
        *,
        ts_codes: list[str] | None = None,
        list_status: str | None = None,
    ) -> pd.DataFrame:
        code_str = ",".join(ts_codes) if ts_codes else None
        df = self._pro.stock_basic(ts_code=code_str, list_status=list_status)
        return ensure_stock_basic(df)

    def daily(
        self,
        *,
        ts_codes: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if not ts_codes:
            raise DataProviderError("daily requires at least one ts_code")
        code_str = ",".join(normalize_ts_codes(ts_codes))
        df = self._pro.daily(
            ts_code=code_str,
            start_date=start_date,
            end_date=end_date,
        )
        return ensure_daily(df)

    def daily_basic(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        if not ts_codes:
            raise DataProviderError("daily_basic requires at least one ts_code")
        code_str = ",".join(normalize_ts_codes(ts_codes))
        df = self._pro.daily_basic(
            ts_code=code_str,
            trade_date=trade_date,
        )
        return ensure_daily_basic(df)

    def stk_limit(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        if not ts_codes:
            raise DataProviderError("stk_limit requires at least one ts_code")
        code_str = ",".join(normalize_ts_codes(ts_codes))
        df = self._pro.stk_limit(
            ts_code=code_str,
            trade_date=trade_date,
        )
        return ensure_stk_limit(df)

    def trade_cal(
        self,
        *,
        exchange: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        is_open: bool | None = None,
    ) -> pd.DataFrame:
        df = self._pro.trade_cal(
            exchange=exchange or "",
            start_date=start_date,
            end_date=end_date,
            is_open=is_open,
        )
        if df is None or df.empty:
            return ensure_trade_cal(pd.DataFrame())
        return ensure_trade_cal(df)

    def adj_factor(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        code_str = ",".join(normalize_ts_codes(ts_codes)) if ts_codes else None
        if trade_date:
            df = self._pro.adj_factor(ts_code=code_str, trade_date=trade_date)
        else:
            df = self._pro.adj_factor(
                ts_code=code_str,
                start_date=start_date,
                end_date=end_date,
            )
        if df is None or df.empty:
            return ensure_adj_factor(pd.DataFrame())
        return ensure_adj_factor(df)

    def suspend_d(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        code_str = ",".join(normalize_ts_codes(ts_codes)) if ts_codes else None
        if trade_date:
            df = self._pro.suspend_d(ts_code=code_str, trade_date=trade_date)
        else:
            df = self._pro.suspend_d(
                ts_code=code_str,
                start_date=start_date,
                end_date=end_date,
            )
        if df is None or df.empty:
            return ensure_suspend_d(pd.DataFrame())
        return ensure_suspend_d(df)
