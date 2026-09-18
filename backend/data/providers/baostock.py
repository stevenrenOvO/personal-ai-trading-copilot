"""BaoStock free A-share data provider.

BaoStock provides stable, token-free historical daily bars, stock basics,
industry classification, trade calendar and adjustment factors. It is the
V1.1 default free provider for daily data.

Data is historical (after-market), never real-time. The provider marks this
explicitly via its ``data_timeliness`` attribute.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from backend.data.providers._a_share_utils import (
    compute_limit_prices,
    from_source_date,
    market_name,
    source_to_ts_code,
    to_source_date,
    ts_to_source_code,
)
from backend.data.providers.base import (
    DataProvider,
    DataProviderError,
    DataSourceNotConfiguredError,
    ensure_daily,
    ensure_daily_basic,
    ensure_adj_factor,
    ensure_stk_limit,
    ensure_stock_basic,
    ensure_suspend_d,
    ensure_trade_cal,
)
from backend.data.schemas import normalize_ts_codes


class BaoStockProvider(DataProvider):
    """Provider backed by the free BaoStock API."""

    name = "baostock"
    data_timeliness = "historical"

    def __init__(self) -> None:
        try:
            import baostock as bs
        except Exception as exc:  # noqa: BLE001
            raise DataSourceNotConfiguredError(
                "baostock is not installed; run: pip install baostock"
            ) from exc

        # BaoStock has no per-call timeout; without this a stalled request can
        # hang forever. A socket timeout turns a hang into a retryable error.
        import socket

        socket.setdefaulttimeout(25)
        self._bs = bs
        result = bs.login()
        if result.error_code != "0":
            raise DataProviderError(f"baostock login failed: {result.error_msg}")
        self._industry_map: dict[str, str] | None = None

    def reconnect(self) -> None:
        """Drop and re-establish the BaoStock session after a stall."""
        try:
            self._bs.logout()
        except Exception:  # noqa: BLE001
            pass
        result = self._bs.login()
        if result.error_code != "0":
            raise DataProviderError(f"baostock re-login failed: {result.error_msg}")

    # -- stock basic -----------------------------------------------------
    def stock_basic(
        self,
        *,
        ts_codes: list[str] | None = None,
        list_status: str | None = None,
    ) -> pd.DataFrame:
        rs = self._bs.query_stock_basic()
        rows = self._drain(rs)
        if not rows:
            return ensure_stock_basic(pd.DataFrame())

        df = pd.DataFrame(rows, columns=rs.fields)
        # Keep stocks only (type == "1"); indices/ETFs use other type codes.
        df = df[df["type"].astype(str) == "1"].copy()
        df["ts_code"] = df["code"].map(source_to_ts_code)
        df["symbol"] = df["code"].str.split(".").str[1]
        df["name"] = df["code_name"]
        df["market"] = df["ts_code"].map(market_name)
        df["list_date"] = df["ipoDate"].map(from_source_date)
        df["list_status"] = df["status"].map(lambda s: "L" if str(s) == "1" else "D")
        df["is_st"] = ""
        df["area"] = ""

        industry = self._industry()
        df["industry"] = df["ts_code"].map(lambda code: industry.get(code, ""))

        if ts_codes:
            codes = set(normalize_ts_codes(ts_codes))
            df = df[df["ts_code"].isin(codes)]
        if list_status:
            df = df[df["list_status"] == list_status]

        return ensure_stock_basic(
            df[
                [
                    "ts_code",
                    "symbol",
                    "name",
                    "area",
                    "industry",
                    "market",
                    "list_date",
                    "list_status",
                    "is_st",
                ]
            ]
        )

    # -- daily -----------------------------------------------------------
    def daily(
        self,
        *,
        ts_codes: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        raw = self.daily_raw(ts_codes=ts_codes, start_date=start_date, end_date=end_date)
        if raw.empty:
            return ensure_daily(pd.DataFrame())
        cols = [
            "trade_date",
            "ts_code",
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
        return ensure_daily(raw[cols])

    def daily_raw(
        self,
        *,
        ts_codes: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Return mapped daily bars plus ``is_st`` / ``tradestatus``."""
        if not ts_codes:
            raise DataProviderError("daily_raw requires at least one ts_code")

        codes = normalize_ts_codes(ts_codes)
        frames: list[pd.DataFrame] = []
        for ts_code in codes:
            rs = self._bs.query_history_k_data_plus(
                ts_to_source_code(ts_code),
                "date,code,open,high,low,close,preclose,volume,amount,pctChg,turn,tradestatus,isST,peTTM,pbMRQ",
                start_date=to_source_date(start_date) if start_date else "",
                end_date=to_source_date(end_date) if end_date else "",
                frequency="d",
                adjustflag="3",
            )
            rows = self._drain(rs)
            if rows:
                frames.append(pd.DataFrame(rows, columns=rs.fields))

        if not frames:
            return pd.DataFrame()
        raw = pd.concat(frames, ignore_index=True)
        return self._map_daily(raw)

    # -- daily basic -----------------------------------------------------
    def daily_basic(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        if not ts_codes:
            raise DataProviderError("daily_basic requires at least one ts_code")

        codes = normalize_ts_codes(ts_codes)
        frames: list[pd.DataFrame] = []
        for ts_code in codes:
            rs = self._bs.query_history_k_data_plus(
                ts_to_source_code(ts_code),
                "date,code,turn,peTTM,pbMRQ,isST",
                start_date=to_source_date(trade_date) if trade_date else "",
                end_date=to_source_date(trade_date) if trade_date else "",
                frequency="d",
                adjustflag="3",
            )
            rows = self._drain(rs)
            if rows:
                frames.append(pd.DataFrame(rows, columns=rs.fields))

        if not frames:
            return ensure_daily_basic(pd.DataFrame())
        raw = pd.concat(frames, ignore_index=True)
        out = pd.DataFrame(
            {
                "trade_date": raw["date"].map(from_source_date),
                "ts_code": raw["code"].map(source_to_ts_code),
                "turnover_rate": pd.to_numeric(raw["turn"], errors="coerce"),
                "pe_ttm": pd.to_numeric(raw["peTTM"], errors="coerce"),
                "pb": pd.to_numeric(raw["pbMRQ"], errors="coerce"),
            }
        )
        return ensure_daily_basic(out)

    # -- limit prices ----------------------------------------------------
    def stk_limit(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        if not ts_codes or not trade_date:
            raise DataProviderError("stk_limit requires ts_codes and trade_date")
        raw = self.daily_raw(
            ts_codes=ts_codes, start_date=trade_date, end_date=trade_date
        )
        if raw.empty:
            return ensure_stk_limit(pd.DataFrame())
        limits = compute_limit_prices(raw)
        return ensure_stk_limit(limits)

    # -- calendar --------------------------------------------------------
    def trade_cal(
        self,
        *,
        exchange: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        is_open: bool | None = None,
    ) -> pd.DataFrame:
        rs = self._bs.query_trade_dates(
            start_date=to_source_date(start_date) if start_date else "",
            end_date=to_source_date(end_date) if end_date else "",
        )
        rows = self._drain(rs)
        if not rows:
            return ensure_trade_cal(pd.DataFrame())
        df = pd.DataFrame(rows, columns=rs.fields)
        out = pd.DataFrame(
            {
                "exchange": "SSE",
                "cal_date": df["calendar_date"].map(from_source_date),
                "is_open": pd.to_numeric(df["is_trading_day"], errors="coerce").astype("int64"),
            }
        )
        if exchange:
            out = out[out["exchange"] == exchange]
        if start_date:
            out = out[out["cal_date"] >= start_date]
        if end_date:
            out = out[out["cal_date"] <= end_date]
        if is_open is not None:
            out = out[out["is_open"] == int(is_open)]
        return ensure_trade_cal(out)

    # -- adjustment factor ----------------------------------------------
    def adj_factor(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if not ts_codes:
            raise DataProviderError("adj_factor requires at least one ts_code")
        codes = normalize_ts_codes(ts_codes)
        frames: list[pd.DataFrame] = []
        for ts_code in codes:
            rs = self._bs.query_adjust_factor(
                code=ts_to_source_code(ts_code),
                start_date=to_source_date(start_date) if start_date else "",
                end_date=to_source_date(end_date) if end_date else "",
            )
            rows = self._drain(rs)
            if rows:
                frames.append(pd.DataFrame(rows, columns=rs.fields))
        if not frames:
            return ensure_adj_factor(pd.DataFrame())
        raw = pd.concat(frames, ignore_index=True)
        out = pd.DataFrame(
            {
                "ts_code": raw["code"].map(source_to_ts_code),
                "trade_date": raw["dividOperateDate"].map(from_source_date),
                "adj_factor": pd.to_numeric(raw["adjustFactor"], errors="coerce"),
            }
        )
        if trade_date:
            out = out[out["trade_date"] == trade_date]
        return ensure_adj_factor(out)

    # -- suspension ------------------------------------------------------
    def suspend_d(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        if not ts_codes:
            raise DataProviderError("suspend_d requires at least one ts_code")
        codes = normalize_ts_codes(ts_codes)
        frames: list[pd.DataFrame] = []
        for ts_code in codes:
            rs = self._bs.query_history_k_data_plus(
                ts_to_source_code(ts_code),
                "date,code,tradestatus",
                start_date=to_source_date(start_date) if start_date else "",
                end_date=to_source_date(end_date) if end_date else "",
                frequency="d",
                adjustflag="3",
            )
            rows = self._drain(rs)
            if rows:
                frames.append(pd.DataFrame(rows, columns=rs.fields))
        if not frames:
            return ensure_suspend_d(pd.DataFrame())
        raw = pd.concat(frames, ignore_index=True)
        suspended = raw[raw["tradestatus"].astype(str) == "0"]
        out = pd.DataFrame(
            {
                "ts_code": suspended["code"].map(source_to_ts_code),
                "trade_date": suspended["date"].map(from_source_date),
                "suspend_timing": "",
                "suspend_type": "S",
            }
        )
        if trade_date:
            out = out[out["trade_date"] == trade_date]
        return ensure_suspend_d(out)

    # -- index constituents (extra, not part of base interface) ---------
    def index_constituents(self, index_code: str = "hs300") -> list[str]:
        method = {
            "hs300": self._bs.query_hs300_stocks,
            "zz500": self._bs.query_zz500_stocks,
            "sz50": self._bs.query_sz50_stocks,
        }.get(index_code)
        if method is None:
            return []
        rs = method()
        rows = self._drain(rs)
        codes: list[str] = []
        for row in rows:
            data = dict(zip(rs.fields, row))
            if "code" in data:
                codes.append(source_to_ts_code(data["code"]))
        return sorted(set(codes))

    def latest_trade_date(self) -> str | None:
        cal = self.trade_cal(is_open=True)
        if cal.empty or "cal_date" not in cal.columns:
            return None
        return str(cal["cal_date"].max())

    # -- helpers ---------------------------------------------------------
    def _industry(self) -> dict[str, str]:
        if self._industry_map is not None:
            return self._industry_map
        rs = self._bs.query_stock_industry()
        rows = self._drain(rs)
        mapping: dict[str, str] = {}
        for row in rows:
            data = dict(zip(rs.fields, row))
            code = source_to_ts_code(data["code"])
            industry = data.get("industry", "")
            if industry:
                mapping[code] = industry
        self._industry_map = mapping
        return mapping

    @staticmethod
    def _drain(rs) -> list[list[str]]:
        rows: list[list[str]] = []
        if rs is None:
            return rows
        while rs.next():
            rows.append(rs.get_row_data())
        return rows

    def _map_daily(self, raw: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "trade_date": raw["date"].map(from_source_date),
                "ts_code": raw["code"].map(source_to_ts_code),
                "open": pd.to_numeric(raw["open"], errors="coerce"),
                "high": pd.to_numeric(raw["high"], errors="coerce"),
                "low": pd.to_numeric(raw["low"], errors="coerce"),
                "close": pd.to_numeric(raw["close"], errors="coerce"),
                "pre_close": pd.to_numeric(raw["preclose"], errors="coerce"),
                "pct_chg": pd.to_numeric(raw["pctChg"], errors="coerce"),
                "vol": pd.to_numeric(raw["volume"], errors="coerce"),
                "amount": pd.to_numeric(raw["amount"], errors="coerce"),
                "is_st": raw["isST"].astype(str),
                "tradestatus": raw["tradestatus"].astype(str),
            }
        )
        out["change"] = out["close"] - out["pre_close"]
        return out
