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
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional

MIN_ROWS_FOR_TODAY = 100          # a real session has thousands of bars
FINAL_PHASES = ("收盘", "盘后")
INDEX_CODES = ("000001.SH", "000300.SH", "000905.SH", "399001.SZ", "399006.SZ")
_INCREMENTAL_INVALIDATE_SECONDS = 5.0
_TAIL_RETRY_SECONDS = 21_600

_state: dict[str, Any] = {"date": None, "running": False, "done": set()}
_job_lock = threading.Lock()
_state_lock = threading.RLock()
_retry_after: dict[tuple[str, str], float] = {}
_pending_memory: dict[str, dict] = {}
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


_snapshot_lock = threading.Lock()
_snapshot_done: dict[tuple[str, str], Optional[str]] = {}


def store_closing_snapshot(runtime: Any, day: str, quote_service: Any = None) -> Optional[dict]:
    """Write ``day``'s bars from the closing snapshot once per snapshot version.

    Returns the stored record, or ``None`` when there is nothing (new) to do.
    """
    history = getattr(getattr(runtime, "provider", None), "history", None)
    if history is None:
        return None
    try:
        if quote_service is None:
            from backend.market.quote_service import get_quote_service

            quote_service = get_quote_service()
        frame, meta = quote_service.get()
    except Exception:  # noqa: BLE001
        return None
    from backend.data.snapshot_bars import is_closing_snapshot, persist_closing_snapshot

    if not is_closing_snapshot(meta, day):
        return None
    key = (str(getattr(history, "db_path", id(history))), str(day))
    version = getattr(meta, "as_of", None)
    if key in _snapshot_done and _snapshot_done[key] == version:
        return None
    try:
        record = persist_closing_snapshot(history, frame, meta, day)
    except Exception:  # noqa: BLE001 - the per-stock backfill remains the fallback
        logging.getLogger(__name__).exception("Storing closing snapshot for %s failed", day)
        return None
    _snapshot_done[key] = version
    if record and record.get("inserted"):
        _row_cache.pop(day, None)
        _invalidate(runtime, day)
    return record


_SNAPSHOT_CHECK_SECONDS = 30.0
_snapshot_checked: dict[str, float] = {}


def _schedule_closing_snapshot(runtime: Any, day: str, quote_service: Any = None) -> None:
    now = time.monotonic()
    if now - _snapshot_checked.get(day, -_SNAPSHOT_CHECK_SECONDS) < _SNAPSHOT_CHECK_SECONDS:
        return
    if not _snapshot_lock.acquire(blocking=False):
        return  # already being written
    _snapshot_checked[day] = now

    def work() -> None:
        try:
            store_closing_snapshot(runtime, day, quote_service)
        finally:
            _snapshot_lock.release()

    try:
        threading.Thread(target=work, name="closing-snapshot", daemon=True).start()
    except Exception:  # noqa: BLE001
        _snapshot_lock.release()


def _watchlist_codes(runtime: Any) -> list[str]:
    try:
        from backend.personal.watchlist import WatchlistStore

        return [item["ts_code"] for item in WatchlistStore(runtime.root).list()
                if isinstance(item, dict) and item.get("ts_code")]
    except Exception:  # noqa: BLE001 - a broken watchlist must not stop ingest
        return []


def _invalidate(runtime: Any, earliest: Optional[str] = None) -> None:
    """Drop derived caches so the freshly ingested bars are picked up.

    With ``earliest`` only results that can see bars from that date onward are
    dropped. A long backfill of recent days must not keep wiping an analysis of
    an older day the user is looking at -- that analysis can take most of a
    minute and would otherwise never finish before the next wipe.
    """
    try:
        if earliest:
            runtime.analysis_cache.invalidate_from(earliest)
            runtime.context_cache.invalidate_from(earliest)
        else:
            runtime.analysis_cache.clear()
            runtime.context_cache.clear()
        if hasattr(runtime.provider, "clear_cache"):
            runtime.provider.clear_cache()
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
    max_gap_days: int = 60,
    quote_service: Any = None,
) -> bool:
    """Backfill every missing recent trading day (at most once per run).

    Latest confirmed gaps are filled first so current pages can become usable
    while older history is still downloading. A day with enough stock coverage
    remains explicitly partial when symbols are absent; those small tails get
    a longer retry delay and cannot hold up a newer incomplete session.
    """
    from backend.data.providers.history import HistoryProvider

    if not isinstance(getattr(runtime, "provider", None), HistoryProvider):
        return False
    try:
        if session is None:
            from backend.market.session import get_trading_session

            session = get_trading_session()
        phase = session.phase()
        if phase == "未知":
            return False
        day = session.current_trading_day()
    except Exception:  # noqa: BLE001
        return False
    if day and backfill is None and rows_on is None:
        # Cheapest path to today's bars: the closing snapshot, one request
        # for the whole market. Runs even while a long backfill is going.
        _schedule_closing_snapshot(runtime, day, quote_service)
    if not day or _state["running"]:
        return False

    threshold = MIN_ROWS_FOR_TODAY
    stock_codes: set[str] = set()
    try:
        basic = runtime.provider.stock_basic()
        stock_codes = set(basic["ts_code"].dropna().astype(str))
        threshold = max(threshold, int(len(stock_codes) * 0.90))
    except Exception:
        pass
    history = getattr(runtime.provider, "history", None)
    if rows_on is not None:
        counter = rows_on
    elif history is not None and stock_codes:
        # Indices must not make an incomplete stock universe pass the gate.
        counter = lambda target: len(stock_codes & history.codes_with_date(target))
    elif history is not None:
        counter = history.rows_on
    else:
        counter = _rows_on
    root_key = str(history.db_path.resolve()) if history is not None else str(id(runtime.provider))
    pending = dict(_pending_memory.get(root_key, {}))
    if history is not None:
        try:
            saved = json.loads(history.get_state("auto_ingest_pending") or "{}")
            if isinstance(saved, dict):
                pending = {key: item for key, item in saved.items() if isinstance(item, dict)}
        except (ValueError, OSError):
            pass
    recent = _recent_open_days(session, day, max_gap_days)
    if phase not in FINAL_PHASES and phase != "休市":
        recent = [d for d in recent if d < day]  # Only confirmed prior sessions.
    def retry_due(target):
        key = (root_key, target)
        if key in _retry_after:
            return time.monotonic() >= _retry_after[key]
        retry_at = pending.get(target, {}).get("next_retry_at")
        if retry_at:
            try:
                return time.time() >= datetime.fromisoformat(retry_at).timestamp()
            except (TypeError, ValueError):
                pass
        return True

    counts = {target: counter(target) for target in recent}
    critical = [target for target in reversed(recent)
                if counts[target] < threshold and retry_due(target)]
    index_gaps: list[str] = []
    if backfill is None and history is not None and recent:
        index_rows = history.load(ts_codes=INDEX_CODES, start_date=min(recent), end_date=max(recent))
        present = {target: set() for target in recent}
        if not index_rows.empty:
            for target, frame in index_rows.groupby("trade_date"):
                present[str(target)] = set(frame["ts_code"].astype(str))
        index_gaps = [target for target in reversed(recent)
                      if len(present[target]) < len(INDEX_CODES) and retry_due(target)]
    tails = [target for target in reversed(recent) if target in pending and retry_due(target)]
    # Work on the newest incomplete universe before repairing small existing
    # holes. An index-only pass can repair the market cards without refetching
    # thousands of stocks whose suspensions/new listings legitimately lack bars.
    mode = "stocks" if critical else "indices" if index_gaps else "tail"
    missing = critical or index_gaps or tails
    with _state_lock:
        _state["threshold_rows"] = threshold
        _state["priority_dates"] = list(missing)
    if not missing:
        return False
    if not _job_lock.acquire(blocking=False):
        return False
    if backfill is not None:
        runner = backfill

    def job() -> None:
        _state["running"] = True
        if _state.get("root") != root_key:
            _state["done"] = set()
        _state["root"] = root_key
        last_invalidated = time.monotonic()
        last_written = 0
        current_days = list(missing)
        job_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with _state_lock:
            _state["progress"] = {"phase": "queued", "priority": mode,
                "dates": list(missing), "processed": 0, "total": 0,
                "downloaded": 0, "failed": 0, "stored_rows": 0,
                "started_at": job_started_at, "updated_at": job_started_at}
        def on_progress(payload):
            nonlocal last_invalidated, last_written
            progress = {**payload, "priority": mode, "dates": list(missing)}
            with _state_lock:
                _state["progress"] = progress
            now = time.monotonic()
            written = int(payload.get("stored_rows", 0))
            if written > last_written and (now - last_invalidated >= _INCREMENTAL_INVALIDATE_SECONDS
                                           or payload.get("phase") == "completed"):
                _invalidate(runtime, min(current_days) if current_days else None)
                last_invalidated = now
                last_written = written
        def schedule_retry(target, details):
            attempts = int(pending.get(target, {}).get("attempts", 0)) + 1
            delay = min(21_600, 300 * (2 ** min(attempts - 1, 7)))
            if counts.get(target, 0) >= threshold:
                delay = max(delay, _TAIL_RETRY_SECONDS)
            _retry_after[(root_key, target)] = time.monotonic() + delay
            pending[target] = {**details, "attempts": attempts,
                "next_retry_at": datetime.fromtimestamp(time.time() + delay, timezone.utc).isoformat(timespec="seconds")}
        def finish_target(target, result):
            counts[target] = counter(target)
            if not result.get("failed", 0) and counts[target] >= threshold:
                _state["done"].add(target)
                _retry_after.pop((root_key, target), None)
                pending.pop(target, None)
            else:
                schedule_retry(target, {"failed": result.get("failed", 0),
                    "failed_codes": result.get("failed_codes", []),
                    "rows": counts[target], "threshold_rows": threshold,
                    "coverage_sufficient": counts[target] >= threshold})
            _pending_memory[root_key] = dict(pending)
            if history is not None:
                history.set_state("auto_ingest_pending", json.dumps(pending))
        job_error = None
        try:
            batch_results = {}
            if backfill is None:
                from backend.data.ingest_full import run_backfill
                # A short window fits the vendor's bounded daily response.
                # Download each stock once for several missing exchange days.
                targets = set(INDEX_CODES) if mode == "indices" else stock_codes | set(INDEX_CODES)
                codes = ",".join(sorted(targets))
                batches = []
                for day in missing:
                    if (not batches or len(batches[-1]) >= 10 or
                        abs((datetime.strptime(day, "%Y%m%d") - datetime.strptime(batches[-1][0], "%Y%m%d")).days) > 20):
                        batches.append([])
                    batches[-1].append(day)
                # The user's own watchlist is tiny and drives daily decisions:
                # repair it right after the indices, before the other ~5,000.
                priority = list(INDEX_CODES) + [code for code in _watchlist_codes(runtime)
                                                if code in targets and code not in INDEX_CODES]
                for days in batches:
                    days = sorted(days)
                    current_days = days
                    last_written = 0
                    try:
                        result = run_backfill(start_date=days[0], end_date=days[-1], source="tencent",
                            db_path=runtime.provider.history.db_path, report_every=1000,
                            codes=codes, required_dates=days, progress_callback=on_progress,
                            priority_codes=priority)
                        for day in days:
                            if mode == "indices":
                                absent = sorted((stock_codes | set(INDEX_CODES)) - history.codes_with_date(day))
                                batch_results[day] = {"failed": len(absent), "failed_codes": absent[:20]}
                            else:
                                batch_results[day] = result.get("failures_by_date", {}).get(day,
                                    {"failed": result.get("failed", 0), "failed_codes": result.get("failed_codes", [])})
                    except Exception as exc:
                        for day in days:
                            batch_results[day] = {"failed": 1, "error": type(exc).__name__}
                    for target in days:
                        finish_target(target, batch_results[target])
            else:
                for target in missing:
                    try:
                        finish_target(target, runner(target) or {})
                    except Exception as exc:
                        schedule_retry(target, {"error": type(exc).__name__})
                        _pending_memory[root_key] = dict(pending)
                        if history is not None:
                            history.set_state("auto_ingest_pending", json.dumps(pending))
                        logging.getLogger(__name__).warning("Ingest failed for %s; retry scheduled", target)
        except Exception as exc:  # noqa: BLE001 - a failed ingest must not break serving
            job_error = type(exc).__name__
            logging.getLogger(__name__).exception("Automatic history backfill interrupted")
        finally:
            with _state_lock:
                _state["running"] = False
                _state["progress"] = {**_state.get("progress", {}),
                    "phase": "error" if job_error else "retry_pending" if any(target in pending for target in missing) else "completed",
                    "error": job_error,
                    "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            _invalidate(runtime)
            _job_lock.release()

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


def status(runtime):
    pending = {}
    history = getattr(runtime.provider, "history", None)
    if history is not None:
        pending = json.loads(history.get_state("auto_ingest_pending") or "{}")
        if not isinstance(pending, dict):
            raise ValueError("补数状态格式损坏")
    root_key = str(history.db_path.resolve()) if history is not None else str(id(runtime.provider))
    with _state_lock:
        same_root = _state.get("root") == root_key
        progress = dict(_state.get("progress", {})) if same_root else {}
        return {"running": bool(_state["running"]) and same_root, "pending": pending,
                "completed_this_process": sorted(_state["done"]) if same_root else [],
                "progress": progress,
                "threshold_rows": _state.get("threshold_rows") if same_root else None,
                "priority_dates": list(_state.get("priority_dates", [])) if same_root else []}


def reset_state() -> None:
    """Test helper: forget which days were attempted."""
    with _state_lock:
        _state.clear()
        _state.update({"date": None, "running": False, "done": set()})
    _row_cache.clear()
    _retry_after.clear()
    _pending_memory.clear()
