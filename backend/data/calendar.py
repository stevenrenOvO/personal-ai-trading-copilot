"""Trading-calendar utilities with an offline fallback.

The calendar is derived from ``trade_cal`` when the provider supplies it, and
from observed ``daily`` trading dates otherwise. This keeps every downstream
engine able to answer "what was the previous open day" even with the minimal
fixture provider.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from backend.data.providers.base import DataProvider

# How many trading days a consecutive-limit-up ("连板") calculation looks back.
# Two days was enough to see "yesterday vs today" and silently capped every
# height at 2, so a 5-board stock was reported as a 2-board one. The engines
# share this constant so the market page and the board page cannot disagree.
CONSECUTIVE_LIMIT_LOOKBACK_DAYS = 20


class TradingCalendar:
    """Resolve trading-day queries against a provider."""

    def __init__(self, provider: DataProvider) -> None:
        self.provider = provider

    @lru_cache(maxsize=1)
    def _open_days(self) -> tuple[str, ...]:
        cal = self.provider.trade_cal()
        if not cal.empty and {"cal_date", "is_open"}.issubset(cal.columns):
            open_days = cal.loc[cal["is_open"].astype(int) == 1, "cal_date"]
            days = sorted(str(d) for d in open_days.unique())
            if days:
                return tuple(days)

        # Fallback: unique dates observed in daily bars.
        daily = self.provider.daily()
        if daily.empty or "trade_date" not in daily.columns:
            return ()
        return tuple(sorted(str(d) for d in daily["trade_date"].unique()))

    def is_trading_day(self, date: str) -> bool:
        return date in self._open_days()

    def previous_trading_day(self, date: str) -> str | None:
        days = self._open_days()
        prior = [d for d in days if d < date]
        return prior[-1] if prior else None

    def next_trading_day(self, date: str) -> str | None:
        days = self._open_days()
        following = [d for d in days if d > date]
        return following[0] if following else None

    def trading_days(self, start_date: str | None = None, end_date: str | None = None) -> list[str]:
        days = list(self._open_days())
        if start_date:
            days = [d for d in days if d >= start_date]
        if end_date:
            days = [d for d in days if d <= end_date]
        return days

    def recent_trading_days(self, date: str, count: int) -> list[str]:
        days = [d for d in self._open_days() if d <= date]
        return days[-count:] if days else []


def trading_calendar_df(dates: list[str]) -> pd.DataFrame:
    """Build a minimal normalized trade_cal DataFrame from open dates."""
    return pd.DataFrame(
        {"exchange": ["SSE"] * len(dates), "cal_date": dates, "is_open": [1] * len(dates)}
    )
