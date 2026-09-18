"""Consecutive limit-up ("连板") heights from normalized daily bars.

Shared by the market page (最高连板) and the board page (首板/二板/三板/最高板)
so the two can never disagree about the same day.

The previous implementation walked every code with a Python loop over sorted
bars, which cost ~7s for a full-market day. Here only the codes that actually
closed at their limit price are touched, so the work is proportional to the
number of limit-ups (~50-100) times the lookback window.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd


def _limit_up_codes(
    bars: pd.DataFrame,
    limits: Optional[pd.DataFrame],
) -> set[str]:
    """Codes whose close is at (or above) that day's up limit."""
    if bars is None or bars.empty:
        return set()
    if limits is None or limits.empty:
        # No vendor limit prices: fall back to the plain 10% rule, which is the
        # documented approximation already used elsewhere in the engines.
        pct = pd.to_numeric(bars.get("pct_chg"), errors="coerce")
        if pct is None:
            return set()
        return set(bars.loc[pct >= 9.5, "ts_code"].astype(str))

    merged = bars.merge(
        limits[["ts_code", "trade_date", "up_limit"]],
        on=["ts_code", "trade_date"],
        how="left",
    )
    close = pd.to_numeric(merged["close"], errors="coerce")
    up_limit = pd.to_numeric(merged["up_limit"], errors="coerce")
    hit = close.notna() & up_limit.notna() & (close >= up_limit)
    return set(merged.loc[hit, "ts_code"].astype(str))


def consecutive_limit_up_heights(
    daily: pd.DataFrame,
    date: str,
    limits_for_day: Callable[[str], Optional[pd.DataFrame]],
) -> dict[str, int]:
    """Height (consecutive limit-up days ending on ``date``) per code.

    Only codes limit-up on ``date`` appear in the result; a code that broke its
    run is simply absent.
    """
    if daily is None or daily.empty:
        return {}

    frame = daily.copy()
    frame["trade_date"] = frame["trade_date"].astype(str)
    days = sorted(day for day in frame["trade_date"].unique() if day <= date)
    if not days:
        return {}

    by_day = {str(day): group for day, group in frame.groupby("trade_date")}
    limit_up_by_day: dict[str, set[str]] = {}
    for day in days:
        bars = by_day.get(day)
        if bars is None:
            continue
        try:
            limits = limits_for_day(day)
        except Exception:  # noqa: BLE001 - missing limits must not abort the scan
            limits = None
        limit_up_by_day[day] = _limit_up_codes(bars, limits)
    return heights_from_sets(limit_up_by_day, days).get(date, {})


def heights_from_sets(
    limit_up_by_day: dict[str, set[str]],
    days: list[str],
) -> dict[str, dict[str, int]]:
    """Consecutive-limit-up heights for EVERY day, from pre-computed sets.

    Returns ``{day: {code: height}}``. A code that is not limit-up on a day
    simply drops out, so a suspension (no bar, hence no limit-up) correctly
    breaks the run instead of being skipped over.
    """
    heights_by_day: dict[str, dict[str, int]] = {}
    running: dict[str, int] = {}
    for day in days:
        running = {
            code: running.get(code, 0) + 1 for code in limit_up_by_day.get(day, set())
        }
        heights_by_day[day] = dict(running)
    return heights_by_day


def max_height(heights: dict[str, int]) -> Optional[int]:
    """Highest board in the ladder, or None when nothing is limit-up."""
    return max(heights.values()) if heights else None
