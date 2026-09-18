"""Full-market daily history store (SQLite).

Kept separate from the existing CSV store so the already-accepted pages are
not affected. The primary key makes ingestion idempotent (dedup) and lets a
download resume by skipping stocks that already have data.

Besides stocks, the store holds a few index series (000300.SH, 000905.SH,
399006.SZ, ...) that the market page needs for 指数表现. Consumers that want
tradable instruments must filter against ``stock_basic``; the backtest engine
does this explicitly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd

from backend.data.config import get_settings

_DAILY_COLUMNS = [
    "ts_code",
    "trade_date",
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


class MarketHistory:
    """SQLite-backed full-market daily history."""

    def __init__(self, db_path: Path | None = None) -> None:
        base = get_settings().db_path.parent / "history"
        self.db_path = Path(db_path) if db_path is not None else base / "market_history.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS daily (
                    ts_code TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, pre_close REAL,
                    change REAL, pct_chg REAL, vol REAL, amount REAL,
                    PRIMARY KEY (ts_code, trade_date)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON daily(trade_date)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS ingest_state (key TEXT PRIMARY KEY, value TEXT)"
            )

    # -- write ---------------------------------------------------------
    def ingest(self, df: pd.DataFrame) -> int:
        """Upsert daily rows. Returns the number of rows written."""
        if df is None or df.empty:
            return 0
        frame = df.copy()
        for column in _DAILY_COLUMNS:
            if column not in frame.columns:
                frame[column] = None
        frame = frame[_DAILY_COLUMNS]
        rows = [tuple(row) for row in frame.itertuples(index=False, name=None)]
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO daily
                (ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    # -- read ----------------------------------------------------------
    def codes_with_data(self, min_rows: int = 1) -> set[str]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT ts_code FROM daily GROUP BY ts_code HAVING COUNT(*) >= ?",
                (min_rows,),
            )
            return {row[0] for row in cursor.fetchall()}

    def codes_with_date(self, date_key: str) -> set[str]:
        """Codes that already have a bar for a specific trading day."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT ts_code FROM daily WHERE trade_date = ?", (date_key,)
            )
            return {row[0] for row in cursor.fetchall()}

    def has_bars(
        self, *, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> bool:
        """Cheap "is this window covered at all" probe.

        Used to decide whether the SQLite store can answer a request before
        falling back to another store; a full ``load`` would be wasteful.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if start_date:
            clauses.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("trade_date <= ?")
            params.append(end_date)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM daily" + where + " LIMIT 1", params).fetchone()
        return row is not None

    def distinct_dates(self) -> list[str]:
        """Every trading date present in the store, ascending."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT trade_date FROM daily ORDER BY trade_date"
            ).fetchall()
        return [str(row[0]) for row in rows]

    def rows_on(self, date_key: str) -> int:
        """Number of bars stored for a single trading day."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM daily WHERE trade_date = ?", (date_key,)
            ).fetchone()
        return int(row[0]) if row else 0

    def load(
        self,
        *,
        ts_codes: Optional[Iterable[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        clauses: list[str] = []
        params: list[Any] = []
        if start_date:
            clauses.append("trade_date >= ?")
            params.append(start_date)
        if end_date:
            clauses.append("trade_date <= ?")
            params.append(end_date)

        code_list = list(dict.fromkeys(str(c) for c in (ts_codes or [])))
        frames: list[pd.DataFrame] = []
        with self._connect() as conn:
            if not code_list:
                where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
                sql = "SELECT * FROM daily" + where + " ORDER BY trade_date, ts_code"
                frames.append(pd.read_sql_query(sql, conn, params=params))
            else:
                for i in range(0, len(code_list), 900):
                    batch = code_list[i : i + 900]
                    placeholders = ",".join("?" for _ in batch)
                    batch_clauses = clauses + [f"ts_code IN ({placeholders})"]
                    sql = (
                        "SELECT * FROM daily WHERE "
                        + " AND ".join(batch_clauses)
                        + " ORDER BY trade_date, ts_code"
                    )
                    frames.append(pd.read_sql_query(sql, conn, params=params + batch))
        if not frames:
            return pd.DataFrame(columns=_DAILY_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def coverage(self) -> dict[str, Any]:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM daily").fetchone()[0]
            codes = conn.execute("SELECT COUNT(DISTINCT ts_code) FROM daily").fetchone()[0]
            dates = conn.execute("SELECT COUNT(DISTINCT trade_date) FROM daily").fetchone()[0]
            span = conn.execute("SELECT MIN(trade_date), MAX(trade_date) FROM daily").fetchone()
        return {
            "rows": total,
            "codes": codes,
            "dates": dates,
            "start": span[0],
            "end": span[1],
            "db_path": str(self.db_path),
            "db_mb": round(self.db_path.stat().st_size / 1e6, 2) if self.db_path.exists() else 0.0,
        }

    # -- state ---------------------------------------------------------
    def get_state(self, key: str) -> Optional[str]:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM ingest_state WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_state(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO ingest_state (key, value) VALUES (?, ?)", (key, value)
            )
