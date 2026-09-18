"""Intraday quote service: two-level cache with a background refresher.

Rules this service guarantees:
  * requests never wait for the vendor - they always read the cache
  * every response carries freshness metadata (as_of / age_seconds / is_stale)
  * a failed refresh keeps the previous data but marks it stale + records the error
  * missing data is never invented, and stale data is never presented as fresh
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

from backend.data.config import get_settings
from backend.market.snapshot import fetch_full_market_snapshot

SNAPSHOT_COLUMNS = [
    "ts_code",
    "name",
    "price",
    "change",
    "pct_chg",
    "pre_close",
    "open",
    "high",
    "low",
    "volume",
    "amount",
]

_NUMERIC = {"price", "change", "pct_chg", "pre_close", "open", "high", "low", "volume", "amount"}


@dataclass
class QuoteMeta:
    source: str = "sina"
    kind: str = "daily_snapshot"
    as_of: Optional[str] = None
    quote_time: Optional[str] = None
    age_seconds: Optional[float] = None
    is_stale: bool = True
    status: str = "warming"  # warming | ok | stale | error
    error: Optional[str] = None
    rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class QuoteService:
    """Cache-first access to the delayed full-market snapshot."""

    def __init__(
        self,
        *,
        db_path: Path | None = None,
        ttl_seconds: float = 30.0,
        refresh_interval: float = 5.0,
        fetcher: Callable[[], tuple[pd.DataFrame, dict]] | None = None,
    ) -> None:
        base = get_settings().db_path.parent / "cache"
        self.db_path = Path(db_path) if db_path is not None else base / "quotes.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.refresh_interval = refresh_interval
        self._fetch = fetcher or fetch_full_market_snapshot

        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fail_count = 0

        self._frame: Optional[pd.DataFrame] = None
        self._meta = QuoteMeta()
        self._init_schema()
        self._load_persisted()

    # -- schema / persistence -----------------------------------------
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS snapshot (
                    {', '.join(
                        (('ts_code TEXT PRIMARY KEY') if c == 'ts_code' else
                         ('name TEXT') if c == 'name' else f'{c} REAL')
                        for c in SNAPSHOT_COLUMNS
                    )}
                )
                """
            )
            conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")

    def _load_persisted(self) -> None:
        with self._connect() as conn:
            try:
                frame = pd.read_sql_query("SELECT * FROM snapshot", conn)
                meta_rows = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            except Exception:  # noqa: BLE001
                return
        if frame.empty:
            return
        meta = QuoteMeta(
            source=meta_rows.get("source", "sina"),
            kind="daily_snapshot",
            as_of=meta_rows.get("as_of"),
            quote_time=meta_rows.get("quote_time"),
            rows=int(meta_rows.get("rows", len(frame)) or len(frame)),
        )
        with self._lock:
            self._frame = frame
            self._meta = meta
            self._apply_age()

    def _persist(self, frame: pd.DataFrame, meta: QuoteMeta) -> None:
        rows = [
            tuple(None if pd.isna(v) else v for v in row)
            for row in frame[SNAPSHOT_COLUMNS].itertuples(index=False, name=None)
        ]
        with self._connect() as conn:
            conn.execute("DELETE FROM snapshot")
            conn.executemany(
                f"INSERT OR REPLACE INTO snapshot ({', '.join(SNAPSHOT_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in SNAPSHOT_COLUMNS)})",
                rows,
            )
            conn.execute("DELETE FROM meta")
            conn.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                [
                    ("source", meta.source),
                    ("as_of", meta.as_of or ""),
                    ("quote_time", meta.quote_time or ""),
                    ("rows", str(meta.rows)),
                ],
            )

    # -- freshness -----------------------------------------------------
    def _apply_age(self) -> None:
        if not self._meta.as_of:
            self._meta.age_seconds = None
            self._meta.is_stale = True
            if self._meta.status not in ("error",):
                self._meta.status = "warming"
            return
        try:
            fetched = datetime.fromisoformat(self._meta.as_of)
            if fetched.tzinfo is None:
                # Older cached rows stored a naive (local) timestamp.
                fetched = fetched.astimezone()
            age = (datetime.now(timezone.utc) - fetched).total_seconds()
        except Exception:  # noqa: BLE001
            age = None
        self._meta.age_seconds = round(age, 1) if age is not None else None
        # After the close, today's snapshot *is* the day's final value: calling
        # it "stale" for the rest of the evening is a false alarm, and it would
        # keep a pointless 30s refresh loop running all night.
        if age is not None and self._closed_for_today():
            self._meta.is_stale = False
            if self._meta.status != "error":
                self._meta.status = "final"
            return
        self._meta.is_stale = age is None or age > self.ttl_seconds
        if self._meta.status != "error":
            self._meta.status = "stale" if self._meta.is_stale else "ok"

    def _closed_for_today(self) -> bool:
        """True when the session is over for the day the snapshot belongs to."""
        try:
            from backend.market.session import get_trading_session

            session = get_trading_session()
            if session.phase() not in ("收盘", "盘后"):
                return False
            day = session.current_trading_day()
        except Exception:  # noqa: BLE001
            return False
        quote_day = (self._meta.quote_time or "")[:8]
        return bool(day) and quote_day == day

    # -- public API ----------------------------------------------------
    def get(self) -> tuple[pd.DataFrame, QuoteMeta]:
        """Return cached data immediately; schedule a refresh when stale."""
        with self._lock:
            self._apply_age()
            frame = self._frame
            meta = QuoteMeta(**self._meta.to_dict())
            stale = self._meta.is_stale
        if stale:
            self._wake.set()
        if frame is None:
            frame = pd.DataFrame(columns=SNAPSHOT_COLUMNS)
        return frame.copy(), meta

    def refresh(self) -> QuoteMeta:
        """Fetch synchronously (used by the background thread, tests and CLI)."""
        try:
            frame, raw = self._fetch()
        except Exception as exc:  # noqa: BLE001
            return self._record_error(str(exc))
        if frame is None or frame.empty:
            return self._record_error("empty snapshot from source")
        meta = QuoteMeta(
            source=str(raw.get("source", "sina")),
            kind="daily_snapshot",
            as_of=str(raw.get("fetched_at") or datetime.now(timezone.utc).isoformat(timespec="seconds")),
            quote_time=str(raw.get("quote_time", "") or ""),
            rows=int(len(frame)),
        )
        with self._lock:
            self._frame = frame
            self._meta = meta
            self._meta.error = None
            self._fail_count = 0
            # Let the age decide the status: never report "ok" for data whose
            # age already exceeds the TTL.
            self._apply_age()
            snapshot = QuoteMeta(**self._meta.to_dict())
        try:
            self._persist(frame, snapshot)
        except Exception:  # noqa: BLE001
            pass
        return snapshot

    def _record_error(self, message: str) -> QuoteMeta:
        with self._lock:
            self._fail_count += 1
            self._meta.error = message
            self._apply_age()
            # A failed refresh means the cached data is no longer confirmed
            # fresh, so it is reported as stale as well as errored.
            self._meta.status = "error"
            self._meta.is_stale = True
            snapshot = QuoteMeta(**self._meta.to_dict())
        return snapshot

    # -- background refresher -----------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="quote-refresher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                self._apply_age()
                need = self._meta.is_stale
            wait_for = self.refresh_interval
            if need:
                meta = self.refresh()
                if meta.status == "error":
                    # Hammering a failing vendor makes a block worse, so back
                    # off progressively until the source recovers.
                    wait_for = min(300.0, 15.0 * (2 ** min(max(self._fail_count - 1, 0), 4)))
            self._wake.wait(wait_for)
            self._wake.clear()


_SERVICE: Optional[QuoteService] = None
_SERVICE_LOCK = threading.Lock()


def get_quote_service() -> QuoteService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = QuoteService()
        return _SERVICE


def set_quote_service(service: Optional[QuoteService]) -> None:
    """Override the process-wide quote service (used by tests)."""
    global _SERVICE
    with _SERVICE_LOCK:
        _SERVICE = service
