"""After-close auto-ingest of the current trading day's daily bars.

Why this exists: the intraday snapshot is live, but every structural engine
(market breadth, ladder, emotion, opportunities) reads *daily bars* from the
local store. Nothing used to write today's bars, so after each close the app
fell back to the previous day and the user saw "以下为最近交易日 ...".

This module closes that gap: once per trading day, after the close, it runs the
existing one-day backfill in the background (never blocking a request) and then
clears the derived caches so the next request sees the new bars. It is a no-op
when the provider is not the local history store (e.g. fixtures in tests), when
the session is still intraday (bars are not final yet), or when the day is
already present.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

MIN_ROWS_FOR_TODAY = 100          # a real session has thousands of bars
FINAL_PHASES = ("收盘", "盘后")

_state: dict[str, Any] = {"date": None, "running": False, "done": set()}
_row_cache: dict[str, tuple[float, int]] = {}
_ROW_CACHE_TTL = 240.0


def _rows_on(day: str) -> int:
    """Bars stored for one day (short-lived cache: this runs on every request)."""
    now = time.monotonic()
    cached = _row_cache.get(day)
    if cached is not None and now - cached[0] < _ROW_CACHE_TTL:
        return cached[1]
    try:
        from backend.data.history import MarketHistory

        rows = MarketHistory().rows_on(day)
    except Exception:  # noqa: BLE001
        rows = 0
    _row_cache[day] = (now, rows)
    return rows


def _backfill_day(day: str) -> dict:
    from backend.data.ingest_full import run_backfill

    return run_backfill(
        start_date=day, end_date=day, source="tencent", report_every=1000
    )


def _invalidate(runtime: Any) -> None:
    """Drop derived caches so the freshly ingested bars are picked up."""
    try:
        runtime.analysis_cache.clear()
        runtime.context_cache.clear()
    except Exception:  # noqa: BLE001
        pass
    try:
        from backend.api import v1

        v1._HISTORY_INFO = None          # noqa: SLF001 - module-level cache
        v1._HISTORY_DAY_CACHE.clear()    # noqa: SLF001
        v1._HISTORY_DATES = None         # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass
    _row_cache.clear()


def ensure_today_bars(
    runtime: Any,
    *,
    session: Any = None,
    rows_on: Optional[Callable[[str], int]] = None,
    backfill: Optional[Callable[[str], dict]] = None,
    run_sync: bool = False,
    max_gap_days: int = 10,
) -> bool:
    """Backfill every missing recent trading day (at most once per run).

    Self-healing: if the app was not opened for a few days, *all* the missing
    sessions are filled, not just the current one. Days are processed oldest
    first so a partial failure still leaves a contiguous history.
    """
    from backend.data.providers.history import HistoryProvider

    if not isinstance(getattr(runtime, "provider", None), HistoryProvider):
        return False
    try:
        if session is None:
            from backend.market.session import get_trading_session

            session = get_trading_session()
        if session.phase() not in FINAL_PHASES:
            return False                    # bars are not final during the session
        day = session.current_trading_day()
    except Exception:  # noqa: BLE001
        return False
    if not day or _state["running"]:
        return False

    counter = rows_on or _rows_on
    missing = [
        d
        for d in _recent_open_days(session, day, max_gap_days)
        if counter(d) < MIN_ROWS_FOR_TODAY
    ]
    if not missing:
        _state["done"].add(day)
        return False
    if all(d in _state["done"] for d in missing):
        return False                        # already attempted in this process

    runner = backfill or _backfill_day

    def job() -> None:
        _state["running"] = True
        try:
            for target in missing:
                runner(target)
                _state["done"].add(target)
        except Exception:  # noqa: BLE001 - a failed ingest must not break serving
            pass
        finally:
            _state["running"] = False
            _state["done"].add(day)
            _invalidate(runtime)

    if run_sync:
        job()
    else:
        _state["running"] = True
        threading.Thread(target=job, name="auto-ingest", daemon=True).start()
    return True


def _recent_open_days(session: Any, today: str, limit: int) -> list[str]:
    """Recent exchange open days up to ``today``, oldest first.

    Uses the trading-session calendar cache (BaoStock trade_cal); if it is not
    available we fall back to sampling the last ``limit`` calendar days.
    """
    try:
        with session._connect() as conn:  # noqa: SLF001 - same package family
            rows = conn.execute(
                "SELECT cal_date FROM calendar WHERE is_open = 1 AND cal_date <= ? "
                "ORDER BY cal_date DESC LIMIT ?",
                (today, limit),
            ).fetchall()
        days = [str(r[0]) for r in rows]
        if days:
            return sorted(days)
    except Exception:  # noqa: BLE001
        pass
    # No calendar: just try today (the vendor request will no-op if closed).
    return [today]


def start_background_loop(runtime_factory: Any, *, interval: float = 300.0) -> bool:
    """Poll in the background so the ingest does not depend on a page visit.

    Cheap by design: the gate re-reads a cached row count and returns
    immediately unless a session is missing.
    """
    try:
        first = runtime_factory()
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(getattr(first, "provider", None), HistoryProviderType()):
        return False

    def loop() -> None:
        while True:
            try:
                ensure_today_bars(runtime_factory())
            except Exception:  # noqa: BLE001
                pass
            time.sleep(interval)

    threading.Thread(target=loop, name="auto-ingest-loop", daemon=True).start()
    return True


def HistoryProviderType():  # noqa: N802 - tiny helper to avoid a module import cycle
    from backend.data.providers.history import HistoryProvider

    return HistoryProvider


def reset_state() -> None:
    """Test helper: forget which days were attempted."""
    _state["date"] = None
    _state["running"] = False
    _state["done"] = set()
    _row_cache.clear()
