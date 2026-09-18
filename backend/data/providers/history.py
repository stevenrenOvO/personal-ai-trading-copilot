"""Provider over the locally ingested full-market daily history (SQLite).

Why this exists
---------------
After the full-market history landed in ``data/history/market_history.db`` the
engines still read ``data/daily.csv`` -- the 300-stock CSI300 store accepted in
phase 1. The dashboard therefore reported a ~300-stock "A-share market":
breadth, limit-up counts and sector strength all described an index basket
instead of the market. This provider exposes the SQLite store through the same
``DataProvider`` contract so no engine has to know where the bars come from.

Honest gaps (surfaced to the UI, never fabricated here)
------------------------------------------------------
* ``amount`` exists only for BaoStock-ingested days. Tencent gap-fill days
  store NULL, so turnover totals are missing there instead of being guessed.
* ``daily_basic`` (turnover rate / valuation) is not ingested -> empty frame.
* ``suspend_d`` is not ingested -> empty frame; a halted stock simply has no
  bar for that day and is excluded from breadth by the absence of data.
* Price limits are *derived* from the stored ``pre_close`` plus the exchange
  limit rules, not read from a vendor field.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from backend.data.calendar import trading_calendar_df
from backend.data.config import get_settings
from backend.data.history import MarketHistory
from backend.data.providers._a_share_utils import (
    compute_limit_prices,
    effective_limit_rate,
    is_st_flag,
)
from backend.data.providers.base import (
    DataProvider,
    DataProviderError,
    ensure_daily,
    ensure_stk_limit,
    ensure_stock_basic,
    ensure_trade_cal,
)
from backend.data.schemas import DAILY_BASIC, SUSPEND_D

# The dashboard asks several engines for the same full-market window within one
# page load (market / emotion / sector / board / opportunity). Re-reading and
# re-normalising ~300k rows per engine made a refresh take ~20s, so completed
# windows are memoised. Turnover in this store only changes when a backfill
# runs, and every response is labelled "historical", so a short TTL is safe.
_WINDOW_CACHE_TTL_SECONDS = 180.0
_MAX_CACHED_WINDOWS = 3
_CACHE_MIN_CODES = 100


class HistoryProvider(DataProvider):
    """Read full-market daily bars from the local SQLite history store."""

    name = "history"
    data_timeliness = "historical"

    def __init__(
        self,
        base_dir: Path | str | None = None,
        *,
        fallback: Optional[DataProvider] = None,
        db_path: Path | None = None,
    ) -> None:
        root = Path(base_dir) if base_dir is not None else get_settings().db_path.parent
        self.root = root
        self.history = MarketHistory(db_path or (root / "history" / "market_history.db"))
        self.fallback = fallback
        self._basic: Optional[pd.DataFrame] = None
        self._st_map: Optional[dict[str, bool]] = None
        self._limit_cache: dict[str, pd.DataFrame] = {}
        self._cal_cache: Optional[pd.DataFrame] = None
        self._window_cache: "OrderedDict[Any, tuple[float, pd.DataFrame]]" = OrderedDict()
        self._lock = threading.Lock()

    # -- introspection -------------------------------------------------
    def coverage(self) -> dict[str, Any]:
        return self.history.coverage()

    def latest_trade_date(self) -> Optional[str]:
        covered = self.history.coverage()
        if covered.get("end"):
            return str(covered["end"])
        if self.fallback is not None:
            return self.fallback.latest_trade_date()
        return None

    # -- stock basic ---------------------------------------------------
    def _stock_basic_frame(self) -> pd.DataFrame:
        if self._basic is not None:
            return self._basic
        path = self.root / "stock_basic.csv"
        if not path.exists():
            if self.fallback is None:
                raise DataProviderError(f"stock_basic source not found: {path}")
            self._basic = self.fallback.stock_basic()
            return self._basic
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        self._basic = ensure_stock_basic(frame)
        return self._basic

    def stock_basic(
        self,
        *,
        ts_codes: list[str] | None = None,
        list_status: str | None = None,
    ) -> pd.DataFrame:
        frame = self._stock_basic_frame()
        if ts_codes:
            frame = frame[frame["ts_code"].isin(ts_codes)]
        if list_status:
            frame = frame[frame["list_status"] == list_status]
        return frame.reset_index(drop=True)

    def _st_flags(self) -> dict[str, bool]:
        """ST/*ST flag per code.

        BaoStock's ingested ``stock_basic.csv`` leaves the ``is_st`` column
        empty, but ST status is part of the stock name ("ST"/"*ST" prefix), so
        the flag is derived from the name instead of being assumed False.
        A stale name is corrected with the stock's own recent moves: the 5% cap
        only holds when the stock actually trades inside a 5% band.
        """
        if self._st_map is not None:
            return self._st_map
        flags: dict[str, bool] = {}
        try:
            basic = self._stock_basic_frame()
        except Exception:  # noqa: BLE001
            self._st_map = flags
            return flags
        has_column = "is_st" in basic.columns
        recent = self._recent_max_moves()
        for _, row in basic.iterrows():
            code = str(row["ts_code"])
            explicit = str(row.get("is_st") or "").strip().upper() if has_column else ""
            name = str(row.get("name") or "").upper()
            named_st = is_st_flag(explicit) or "ST" in name
            flags[code] = effective_limit_rate(code, named_st, recent.get(code)) == 0.05
        self._st_map = flags
        return flags

    def _recent_max_moves(self) -> dict[str, float]:
        """Largest absolute daily move per code in the local store."""
        try:
            with self.history._connect() as conn:  # noqa: SLF001 - same store
                rows = conn.execute(
                    "SELECT ts_code, MAX(ABS(pct_chg)) FROM daily "
                    "WHERE pct_chg IS NOT NULL AND ABS(pct_chg) < 25 "
                    "GROUP BY ts_code"
                ).fetchall()
        except Exception:  # noqa: BLE001
            return {}
        return {str(code): float(value) for code, value in rows if value is not None}

    # -- bars ----------------------------------------------------------
    def daily(
        self,
        *,
        ts_codes: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        # The returned frame is shared with the cache; callers must not mutate
        # it in place (every engine filters or merges into a new frame instead).
        key = self._window_key(ts_codes, start_date, end_date)
        cached = self._window_get(key)
        if cached is not None:
            return cached

        result: Optional[pd.DataFrame] = None
        if self.history.has_bars(start_date=start_date, end_date=end_date):
            frame = self.history.load(
                ts_codes=ts_codes, start_date=start_date, end_date=end_date
            )
            if not frame.empty:
                result = ensure_daily(frame)
        if result is None:
            if self.fallback is not None:
                result = self.fallback.daily(
                    ts_codes=ts_codes, start_date=start_date, end_date=end_date
                )
            else:
                result = ensure_daily(pd.DataFrame(columns=["trade_date", "ts_code"]))
        self._window_put(key, result)
        return result

    # -- window memoisation -------------------------------------------
    def _window_key(
        self,
        ts_codes: list[str] | None,
        start_date: str | None,
        end_date: str | None,
    ) -> Optional[tuple]:
        if not start_date or not end_date:
            return None
        if ts_codes is None:
            return (None, start_date, end_date)
        codes = [str(code) for code in ts_codes]
        if len(codes) < _CACHE_MIN_CODES:
            return None
        return (len(codes), hash(tuple(sorted(codes))), start_date, end_date)

    def _window_get(self, key: Optional[tuple]) -> Optional[pd.DataFrame]:
        if key is None:
            return None
        with self._lock:
            entry = self._window_cache.get(key)
            if entry is None:
                return None
            stored_at, frame = entry
            if time.monotonic() - stored_at > _WINDOW_CACHE_TTL_SECONDS:
                del self._window_cache[key]
                return None
            self._window_cache.move_to_end(key)
            return frame

    def _window_put(self, key: Optional[tuple], frame: pd.DataFrame) -> None:
        if key is None or frame.empty:
            return
        with self._lock:
            self._window_cache[key] = (time.monotonic(), frame)
            self._window_cache.move_to_end(key)
            while len(self._window_cache) > _MAX_CACHED_WINDOWS:
                self._window_cache.popitem(last=False)

    def clear_cache(self) -> None:
        """Drop memoised windows (used after a backfill writes new bars)."""
        with self._lock:
            self._window_cache.clear()
            self._limit_cache.clear()
            self._cal_cache = None
            self._basic = None
            self._st_map = None

    def daily_basic(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        # Not ingested for the full market; an empty frame is the honest answer.
        return pd.DataFrame(columns=list(DAILY_BASIC.all_columns))

    def suspend_d(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        # Not ingested; see module docstring.
        return pd.DataFrame(columns=list(SUSPEND_D.all_columns))

    def stk_limit(
        self,
        *,
        ts_codes: list[str] | None = None,
        trade_date: str | None = None,
    ) -> pd.DataFrame:
        if not trade_date:
            if self.fallback is not None:
                return self.fallback.stk_limit(ts_codes=ts_codes, trade_date=trade_date)
            return pd.DataFrame(columns=["trade_date", "ts_code", "up_limit", "down_limit"])
        frame = self._limits_for_day(trade_date)
        if frame.empty and self.fallback is not None:
            return self.fallback.stk_limit(ts_codes=ts_codes, trade_date=trade_date)
        if ts_codes:
            frame = frame[frame["ts_code"].isin(ts_codes)]
        return frame.reset_index(drop=True)

    def _limits_for_day(self, trade_date: str) -> pd.DataFrame:
        cached = self._limit_cache.get(trade_date)
        if cached is not None:
            return cached
        bars = self.history.load(start_date=trade_date, end_date=trade_date)
        if bars.empty:
            return pd.DataFrame(columns=["trade_date", "ts_code", "up_limit", "down_limit"])
        flags = self._st_flags()
        bars = bars.assign(is_st=[flags.get(str(code), False) for code in bars["ts_code"]])
        limits = compute_limit_prices(bars)
        result = (
            ensure_stk_limit(limits)
            if not limits.empty
            else pd.DataFrame(columns=["trade_date", "ts_code", "up_limit", "down_limit"])
        )
        # A 20-day ladder window needs 20 days of limits at once, so the cache
        # must comfortably exceed the lookback or it evicts itself mid-run.
        if len(self._limit_cache) > 64:
            self._limit_cache.clear()
        self._limit_cache[trade_date] = result
        return result

    # -- calendar ------------------------------------------------------
    def _calendar_frame(self) -> pd.DataFrame:
        if self._cal_cache is not None:
            return self._cal_cache
        dates = self.history.distinct_dates()
        extra: set[str] = set()
        cal_path = self.root / "trade_cal.csv"
        if cal_path.exists():
            cal = pd.read_csv(cal_path, dtype=str, keep_default_na=False)
            if {"cal_date", "is_open"}.issubset(cal.columns):
                open_rows = cal[cal["is_open"].astype(str).str.strip() == "1"]
                extra = {str(d) for d in open_rows["cal_date"]}
        merged = sorted(set(dates) | extra)
        self._cal_cache = ensure_trade_cal(trading_calendar_df(merged))
        return self._cal_cache

    def trade_cal(
        self,
        *,
        exchange: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        is_open: bool | None = None,
    ) -> pd.DataFrame:
        frame = self._calendar_frame()
        if frame.empty:
            return frame
        if exchange:
            frame = frame[frame["exchange"] == exchange]
        if start_date:
            frame = frame[frame["cal_date"] >= start_date]
        if end_date:
            frame = frame[frame["cal_date"] <= end_date]
        if is_open is not None:
            frame = frame[frame["is_open"].astype(int) == int(is_open)]
        return frame.reset_index(drop=True)
