"""Trading session service: what is the current trading day, and what phase is it.

The dashboard previously used "the last date in the local history" as "today",
which is wrong: on a later date that answers with an old day and, worse, makes
the live snapshot get labelled with the wrong date. This service resolves the
real trading day from the exchange calendar and reports the session phase.
"""

from __future__ import annotations

import sqlite3
import time as _time
from datetime import date as _date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Optional

from backend.data.config import get_settings


def _default_calendar_fetcher(start: str, end: str) -> list[tuple[str, int]]:
    from backend.data.providers.baostock import BaoStockProvider

    provider = BaoStockProvider()
    frame = provider.trade_cal(
        start_date=start.replace("-", ""), end_date=end.replace("-", "")
    )
    if frame.empty:
        return []
    return [
        (str(row["cal_date"]), int(row["is_open"]))
        for _, row in frame.iterrows()
    ]


class TradingSession:
    """Resolve the current trading day and the intraday phase, with a cache."""

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        calendar_fetcher: Callable[[str, str], list[tuple[str, int]]] | None = None,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        base = get_settings().db_path.parent / "cache"
        self.db_path = Path(db_path) if db_path is not None else base / "session.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fetch = calendar_fetcher or _default_calendar_fetcher
        self._now = now_fn or (lambda: datetime.now())
        self._blocked_until = 0.0
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS calendar (cal_date TEXT PRIMARY KEY, is_open INTEGER)"
            )
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")

    # -- calendar ------------------------------------------------------
    def _cached_month(self) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key='month'").fetchone()
        return row[0] if row else None

    def ensure_calendar(self, today: Optional[_date] = None) -> bool:
        """Make sure the exchange calendar covers the current month."""
        today = today or self._now().date()
        month_key = f"{today.year:04d}-{today.month:02d}"
        if self._cached_month() == month_key:
            return True
        if _time.time() < self._blocked_until:
            return False  # recent failure: don't hammer the vendor
        start = (today - timedelta(days=15)).isoformat()
        end = (today + timedelta(days=45)).isoformat()
        try:
            rows = self._fetch(start, end)
        except Exception:  # noqa: BLE001
            self._blocked_until = _time.time() + 300
            return False
        if not rows:
            self._blocked_until = _time.time() + 300
            return False
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO calendar (cal_date, is_open) VALUES (?, ?)", rows
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('month', ?)", (month_key,)
            )
        return True

    def current_trading_day(self) -> Optional[str]:
        """Latest open exchange day that is not in the future."""
        now = self._now()
        self.ensure_calendar(now.date())
        today_key = now.strftime("%Y%m%d")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(cal_date) FROM calendar WHERE is_open = 1 AND cal_date <= ?",
                (today_key,),
            ).fetchone()
        return row[0] if row and row[0] else None

    def is_trading_day(self, date_key: str) -> Optional[bool]:
        self.ensure_calendar()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT is_open FROM calendar WHERE cal_date = ?", (date_key,)
            ).fetchone()
        return None if row is None else bool(row[0])

    # -- session phase -------------------------------------------------
    def phase(self) -> str:
        now = self._now()
        day = now.strftime("%Y%m%d")
        trading = self.is_trading_day(day)
        if trading is None:
            return "未知"
        if not trading:
            return "休市"
        t = now.time()
        if t < time(9, 15):
            return "盘前"
        if t < time(9, 25):
            return "集合竞价"
        if t < time(9, 30):
            return "开盘前"
        if t < time(11, 30):
            return "盘中"
        if t < time(13, 0):
            return "午间休市"
        if t < time(15, 0):
            return "盘中"
        if t < time(15, 30):
            return "收盘"
        return "盘后"

    def describe(self) -> dict[str, Any]:
        now = self._now()
        return {
            "now": now.isoformat(timespec="seconds"),
            "phase": self.phase(),
            "current_trading_day": self.current_trading_day(),
            "is_trading_day": self.is_trading_day(now.strftime("%Y%m%d")),
        }


_SESSION: Optional[TradingSession] = None


def get_trading_session() -> TradingSession:
    global _SESSION
    if _SESSION is None:
        _SESSION = TradingSession()
    return _SESSION


def set_trading_session(session: Optional[TradingSession]) -> None:
    """Override the process-wide session (used by tests)."""
    global _SESSION
    _SESSION = session
