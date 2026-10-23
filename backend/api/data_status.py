"""Shared availability of full-market daily data and date-aligned quotes."""
from __future__ import annotations

import time
import math


def snapshot_available(runtime, frame, meta, date: str | None) -> bool:
    quote_date = getattr(meta, "quote_date", None) or str(meta.quote_time or "")[:8]
    if frame.empty or not date or quote_date != date:
        return False
    if runtime is None or runtime.provider.name == "fixture":
        return True
    if "price" not in frame or "ts_code" not in frame:
        return False
    def valid(value):
        try:
            return math.isfinite(float(value)) and float(value) > 0
        except (ValueError, TypeError):
            return False
    local = hasattr(runtime.provider, "history") or runtime.provider.name == "storage"
    threshold = max(1000, int(len(runtime.provider.stock_basic()) * .9)) if local else 1000
    return frame.loc[frame["price"].map(valid), "ts_code"].nunique() >= threshold


def history_dates(runtime) -> list[dict]:
    """Count valid stock bars, excluding indexes and incomplete price records."""
    def compute():
        if not hasattr(runtime.provider, "history") and runtime.provider.name not in ("storage", "fixture"):
            return []  # A date picker must not trigger an unbounded network daily query.
        basic = runtime.provider.stock_basic()
        codes = sorted(set(basic["ts_code"].dropna().astype(str))) if not basic.empty else []
        count = len(codes)
        threshold = max(100, int(count * .9))
        history = getattr(runtime.provider, "history", None)
        if history is None:
            frame = runtime.provider.daily()
            if frame.empty:
                return []
            grouped = frame[frame["ts_code"].isin(codes)].groupby("trade_date")["ts_code"].nunique()
            return [{"date": str(day), "row_count": int(rows), "valid_rows": int(rows),
                     "coverage_ratio": min(1., int(rows) / count) if count else 0,
                     "complete": runtime.provider.name == "fixture" or (count >= 1000 and rows >= threshold)}
                    for day, rows in grouped.items()]
        totals = {}
        # Keep each IN clause within common SQLite parameter limits.
        with history._connect() as conn:
            for offset in range(0, count, 800):
                part = codes[offset:offset + 800]
                rows = conn.execute(
                    "SELECT trade_date,COUNT(*),SUM(CASE WHEN open>0 AND high>0 AND low>0 AND close>0 "
                    "AND high>=open AND high>=close AND low<=open AND low<=close AND high>=low "
                    "AND high<1e9 THEN 1 ELSE 0 END) FROM daily WHERE ts_code IN (" +
                    ",".join("?" for _ in part) + ") GROUP BY trade_date", part).fetchall()
                for day, total, valid in rows:
                    item = totals.setdefault(str(day), [0, 0])
                    item[0] += total
                    item[1] += valid or 0
        return [{"date": day, "row_count": rows, "valid_rows": valid,
                 "coverage_ratio": min(1., valid / count) if count else 0,
                 "complete": count >= 1000 and valid >= threshold}
                for day, (rows, valid) in sorted(totals.items())]
    history = getattr(runtime.provider, "history", None)
    if history is not None:
        stat = history.db_path.stat()
        revision = (stat.st_mtime_ns, stat.st_size)
    else:
        revision = int(time.monotonic() // 5)
    # Reuse the expensive market-wide scan while the database is unchanged;
    # a committed ingest changes the revision immediately.
    return runtime.analysis_cache.get_or_compute(("date_availability", revision), compute)


def date_status(runtime, date: str) -> dict:
    return next((item for item in history_dates(runtime) if item["date"] == date),
                {"date": date, "row_count": 0, "valid_rows": 0, "coverage_ratio": 0, "complete": False})


def latest_complete(runtime, before: str | None = None) -> str | None:
    return max((item["date"] for item in history_dates(runtime)
                if item["complete"] and (before is None or item["date"] <= before)), default=None)


def display_dates(runtime) -> dict:
    from backend.market.session import get_trading_session
    from backend.market.quote_service import get_quote_service
    session = get_trading_session()
    info = session.describe_cached()
    if info.get("is_trading_day") is None:
        session.ensure_calendar()
        info = session.describe_cached()
    frame, meta = get_quote_service().get()
    items = history_dates(runtime)
    expected = info.get("expected_daily_date")
    if runtime.provider.name == "fixture":
        expected = info.get("current_trading_day")
    upper = expected or str(info.get("now", ""))[:10].replace("-", "")
    complete = max((item["date"] for item in items if item["complete"]
                    and (not upper or item["date"] <= upper)), default=None)
    snapshot = meta.quote_date if snapshot_available(runtime, frame, meta, meta.quote_date) else None
    if snapshot and info.get("now") and snapshot > str(info["now"])[:10].replace("-", ""):
        snapshot = None
    default = complete or snapshot or info.get("current_trading_day")
    return {"date": default, "current_trading_day": info.get("current_trading_day"),
            "latest_complete_date": complete, "snapshot_date": snapshot,
            "phase": info.get("phase"), "now": info.get("now"),
            "display_reason": (f"默认展示 {complete} 覆盖充分的日线，当前交易日为 {info.get('current_trading_day') or '待确认'}"
                               if complete else "日线结构仍在补齐，可查看最近行情快照"),
            "history_dates": [{**item, "has_snapshot": item["date"] == snapshot} for item in items]}
