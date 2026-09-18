"""AkShare free data provider (best-effort fallback).

AkShare covers many sources but relies on third-party endpoints (Eastmoney,
Sina, ...) that can change or rate-limit. This provider implements the
essential stock list and daily bars; unsupported tables return empty frames so
the fallback chain can continue.

Data is delayed/historical, never real-time.
"""

from __future__ import annotations

import pandas as pd

from backend.data.providers._a_share_utils import from_source_date, to_source_date
from backend.data.providers.base import (
    DataProvider,
    DataProviderError,
    DataSourceNotConfiguredError,
    ensure_daily,
    ensure_stock_basic,
)


def _exchange_suffix(symbol: str) -> str:
    if symbol.startswith(("60", "68", "90")):
        return "SH"
    if symbol.startswith(("00", "30", "20")):
        return "SZ"
    if symbol.startswith(("43", "83", "87", "88", "92")):
        return "BJ"
    return "SZ"


class AkShareProvider(DataProvider):
    """Provider backed by the free AkShare library."""

    name = "akshare"
    data_timeliness = "delayed"

    def __init__(self) -> None:
        try:
            import akshare as ak
        except Exception as exc:  # noqa: BLE001
            raise DataSourceNotConfiguredError(
                "akshare is not installed; run: pip install akshare"
            ) from exc
        self._ak = ak

    def stock_basic(
        self,
        *,
        ts_codes: list[str] | None = None,
        list_status: str | None = None,
    ) -> pd.DataFrame:
        raw = self._ak.stock_info_a_code_name()
        if raw is None or raw.empty:
            return ensure_stock_basic(pd.DataFrame())
        out = pd.DataFrame(
            {
                "ts_code": raw["code"].astype(str).map(lambda s: f"{s}.{_exchange_suffix(s)}"),
                "symbol": raw["code"].astype(str),
                "name": raw["name"].astype(str),
                "area": "",
                "industry": "",
                "market": "",
                "list_date": "",
                "list_status": "L",
                "is_st": "",
            }
        )
        if ts_codes:
            out = out[out["ts_code"].isin(set(ts_codes))]
        if list_status:
            out = out[out["list_status"] == list_status]
        return ensure_stock_basic(out)

    def daily(
        self,
        *,
        ts_codes: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if not ts_codes:
            raise DataProviderError("daily requires at least one ts_code")

        frames: list[pd.DataFrame] = []
        for ts_code in ts_codes:
            symbol = ts_code.split(".")[0]
            try:
                raw = self._ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=to_source_date(start_date) if start_date else "19900101",
                    end_date=to_source_date(end_date) if end_date else "20500101",
                    adjust="",
                )
            except Exception:
                continue
            if raw is None or raw.empty:
                continue
            frames.append(self._map_daily(raw, ts_code))

        if not frames:
            return ensure_daily(pd.DataFrame())
        return ensure_daily(pd.concat(frames, ignore_index=True))

    def daily_basic(self, *, ts_codes=None, trade_date=None):
        # AkShare daily hist does not expose PE/PB; leave empty for fallback.
        from backend.data.providers.base import ensure_daily_basic

        return ensure_daily_basic(pd.DataFrame())

    def stk_limit(self, *, ts_codes=None, trade_date=None):
        from backend.data.providers.base import ensure_stk_limit

        return ensure_stk_limit(pd.DataFrame())

    def trade_cal(self, *, exchange=None, start_date=None, end_date=None, is_open=None):
        from backend.data.providers.base import ensure_trade_cal

        raw = None
        for fn_name in ("tool_trade_date_hist_sina", "trade_date_hist_sina"):
            fn = getattr(self._ak, fn_name, None)
            if fn is None:
                continue
            try:
                raw = fn()
            except Exception:
                raw = None
            if raw is not None and not raw.empty:
                break
        if raw is None or raw.empty:
            return ensure_trade_cal(pd.DataFrame())

        date_col = next((c for c in ("trade_date", "日期") if c in raw.columns), None)
        if date_col is None:
            return ensure_trade_cal(pd.DataFrame())
        out = pd.DataFrame(
            {
                "exchange": "SSE",
                "cal_date": raw[date_col].astype(str).map(from_source_date),
                "is_open": 1,
            }
        )
        if start_date:
            out = out[out["cal_date"] >= start_date]
        if end_date:
            out = out[out["cal_date"] <= end_date]
        return ensure_trade_cal(out)

    def _map_daily(self, raw: pd.DataFrame, ts_code: str) -> pd.DataFrame:
        col = {
            "日期": "trade_date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "vol",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
            "涨跌额": "change",
        }
        renamed = raw.rename(columns=col)
        out = pd.DataFrame(
            {
                "trade_date": renamed["trade_date"].astype(str).map(from_source_date),
                "ts_code": ts_code,
                "open": pd.to_numeric(renamed["open"], errors="coerce"),
                "high": pd.to_numeric(renamed["high"], errors="coerce"),
                "low": pd.to_numeric(renamed["low"], errors="coerce"),
                "close": pd.to_numeric(renamed["close"], errors="coerce"),
                "pct_chg": pd.to_numeric(renamed["pct_chg"], errors="coerce"),
                "vol": pd.to_numeric(renamed["vol"], errors="coerce"),
                "amount": pd.to_numeric(renamed["amount"], errors="coerce"),
            }
        )
        out["change"] = pd.to_numeric(renamed["change"], errors="coerce")
        out["pre_close"] = out["close"] - out["change"]
        return out
