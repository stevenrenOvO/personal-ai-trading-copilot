"""Forward-adjusted (前复权) daily bars for the user's watchlist and holdings.

The full-market history is unadjusted. Moving averages and MACD across an
ex-dividend day would then see a fake gap, so the band rules read adjusted
bars instead -- but only for the handful of stocks the user follows, which is
why this is a small per-stock cache rather than a market-wide table.

Freshness rules:

* the series is refetched in full (never patched) at most once per trading
  day per stock -- forward adjustment rescales all earlier bars after every
  corporate action;
* if the vendor has not published today's bar yet, today's unadjusted bar
  from the local store is appended *only* when its previous close matches the
  adjusted series' last close (no corporate action today); otherwise the
  series stops at the last adjusted day and says so;
* when fetching fails, the cached series is used and reported as stale; with
  no cache, raw bars are used and reported as unadjusted.
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from backend.data.sqlite import connect

BAR_COLUMNS = ["trade_date", "open", "high", "low", "close", "vol"]
_CONTINUITY_TOLERANCE = 0.005
_RETRY_AFTER_SECONDS = 300.0
_FETCH_TIMEOUT_SECONDS = 5.0
_FAILED: dict[tuple[str, str], float] = {}


class AdjustedBarStore:
    def __init__(self, root: Path, fetcher: Optional[Callable[[str], pd.DataFrame]] = None) -> None:
        self.path = Path(root) / "personal" / "adjusted.db"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fetcher is None:
            from backend.data.providers import tencent

            def fetcher(code: str) -> pd.DataFrame:
                return tencent.fetch_qfq_daily(code, timeout=_FETCH_TIMEOUT_SECONDS)
        self._fetch = fetcher
        self._notes: dict[str, list[str]] = {}
        with connect(self.path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS bars (ts_code TEXT, trade_date TEXT, open REAL, "
                         "high REAL, low REAL, close REAL, vol REAL, PRIMARY KEY (ts_code, trade_date))")
            conn.execute("CREATE TABLE IF NOT EXISTS series (ts_code TEXT PRIMARY KEY, fetched_for TEXT, "
                         "fetched_at TEXT, source TEXT, last_date TEXT)")

    # -- storage -------------------------------------------------------
    def _load(self, ts_code: str) -> tuple[pd.DataFrame, Optional[dict]]:
        with connect(self.path) as conn:
            frame = pd.read_sql_query(
                "SELECT trade_date, open, high, low, close, vol FROM bars WHERE ts_code = ? ORDER BY trade_date",
                conn, params=[ts_code])
            row = conn.execute("SELECT fetched_for, fetched_at, source, last_date FROM series WHERE ts_code = ?",
                               (ts_code,)).fetchone()
        meta = dict(zip(("fetched_for", "fetched_at", "source", "last_date"), row)) if row else None
        return frame, meta

    def _replace(self, ts_code: str, frame: pd.DataFrame, fetched_for: str) -> dict:
        frame = frame[BAR_COLUMNS].sort_values("trade_date").drop_duplicates("trade_date")
        meta = {"fetched_for": fetched_for, "fetched_at": datetime.now().isoformat(timespec="seconds"),
                "source": "tencent_qfq", "last_date": str(frame["trade_date"].iloc[-1])}
        with connect(self.path) as conn:
            conn.execute("DELETE FROM bars WHERE ts_code = ?", (ts_code,))
            conn.executemany("INSERT INTO bars VALUES (?, ?, ?, ?, ?, ?, ?)",
                             [(ts_code, *row) for row in frame.itertuples(index=False, name=None)])
            conn.execute("INSERT OR REPLACE INTO series VALUES (?, ?, ?, ?, ?)",
                         (ts_code, meta["fetched_for"], meta["fetched_at"], meta["source"], meta["last_date"]))
        return meta

    # -- fetching ------------------------------------------------------
    def _backing_off(self, ts_code: str) -> bool:
        failed_at = _FAILED.get((str(self.path), ts_code))
        return failed_at is not None and time.monotonic() - failed_at < _RETRY_AFTER_SECONDS

    def _call_fetch(self, ts_code: str) -> Any:
        try:
            return self._fetch(ts_code)
        except Exception as exc:  # noqa: BLE001 - network failure must not break the page
            return exc

    def _apply(self, ts_code: str, as_of: str, meta: Optional[dict], outcome: Any) -> list[str]:
        """Store a fetch outcome; return the warnings it implies."""
        if isinstance(outcome, Exception):
            _FAILED[(str(self.path), ts_code)] = time.monotonic()
            if meta is not None:
                return [f"复权数据刷新失败（{type(outcome).__name__}），沿用 {meta['fetched_at'][:10]} 的缓存"]
            return [f"复权数据获取失败（{type(outcome).__name__}）"]
        if outcome is not None and not outcome.empty:
            self._replace(ts_code, outcome, as_of)
            _FAILED.pop((str(self.path), ts_code), None)
            return []
        return ["复权数据源未返回数据，沿用上次缓存"] if meta is not None else []

    def prefetch(self, codes: list[str], as_of: str, *, max_workers: int = 6) -> None:
        """Download every stale series at once, then store them one by one.

        Ten stocks fetched in sequence with an unreachable vendor used to take
        10 x timeout; in parallel it takes about one timeout.
        """
        todo = []
        for code in dict.fromkeys(codes):
            _, meta = self._load(code)
            if (meta is None or meta["fetched_for"] < as_of) and not self._backing_off(code):
                todo.append((code, meta))
        if not todo:
            return
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(max_workers, len(todo))) as pool:
            outcomes = list(pool.map(lambda item: self._call_fetch(item[0]), todo))
        for (code, meta), outcome in zip(todo, outcomes):
            self._notes[code] = self._apply(code, as_of, meta, outcome)

    # -- public --------------------------------------------------------
    def bars(self, ts_code: str, as_of: str, history: Any = None) -> tuple[pd.DataFrame, dict]:
        """Adjusted bars up to ``as_of`` plus a basis record for the UI."""
        frame, meta = self._load(ts_code)
        warnings: list[str] = []
        if ts_code in self._notes:
            warnings.extend(self._notes.pop(ts_code))  # already attempted by prefetch()
        elif (meta is None or meta["fetched_for"] < as_of) and self._backing_off(ts_code):
            warnings.append("复权数据刚刚获取失败，稍后自动重试")
        elif meta is None or meta["fetched_for"] < as_of:
            warnings.extend(self._apply(ts_code, as_of, meta, self._call_fetch(ts_code)))
        frame, meta = self._load(ts_code)
        basis = {"source": meta["source"] if meta else None, "adjusted": meta is not None,
                 "fetched_at": meta["fetched_at"] if meta else None,
                 "adjusted_last_date": meta["last_date"] if meta else None, "appended_raw_dates": []}

        if meta is None:
            raw = _raw_bars(history, ts_code, as_of)
            if raw.empty:
                basis["warnings"] = warnings + ["没有可用的日线"]
                return pd.DataFrame(columns=BAR_COLUMNS), basis
            basis.update(source="local_unadjusted", adjusted=False)
            basis["warnings"] = warnings + ["使用未复权日线：除权日附近的均线和 MACD 可能失真"]
            return raw, basis

        frame = frame[frame["trade_date"] <= as_of].reset_index(drop=True)
        last = str(frame["trade_date"].iloc[-1]) if not frame.empty else None
        if last and last < as_of:
            raw = _raw_bars(history, ts_code, as_of, after=last, with_pre_close=True)
            prev_close = float(frame["close"].iloc[-1])
            appended = []
            for row in raw.itertuples(index=False):
                pre = row.pre_close
                if not pre or abs(float(pre) / prev_close - 1) > _CONTINUITY_TOLERANCE:
                    warnings.append(f"{row.trade_date} 前收盘与复权序列不连续（可能除权），"
                                    f"状态按 {appended[-1].trade_date if appended else last} 计算，待复权数据更新")
                    break
                appended.append(row)
                prev_close = float(row.close)
            if appended:
                frame = pd.concat([frame, pd.DataFrame(appended)[BAR_COLUMNS]], ignore_index=True)
                basis["appended_raw_dates"] = [str(row.trade_date) for row in appended]
        basis["last_date"] = str(frame["trade_date"].iloc[-1]) if not frame.empty else None
        basis["warnings"] = warnings
        return frame, basis


def _raw_bars(history: Any, ts_code: str, as_of: str, *, after: Optional[str] = None,
              with_pre_close: bool = False) -> pd.DataFrame:
    if history is None:
        return pd.DataFrame(columns=BAR_COLUMNS)
    try:
        raw = history.load(ts_codes=[ts_code], end_date=as_of)
    except (sqlite3.Error, OSError):
        return pd.DataFrame(columns=BAR_COLUMNS)
    if raw.empty:
        return pd.DataFrame(columns=BAR_COLUMNS)
    raw = raw.sort_values("trade_date")
    raw["trade_date"] = raw["trade_date"].astype(str)
    if after:
        raw = raw[raw["trade_date"] > after]
    columns = BAR_COLUMNS + (["pre_close"] if with_pre_close else [])
    return raw[columns].reset_index(drop=True)
