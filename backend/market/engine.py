"""Market state engine for A-share breadth, limits and strength.

The engine is intentionally provider-agnostic. It computes a full
``MarketState`` from normalized daily bars and limit prices, with an explicit
factor list so every score can be explained to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from backend.data.calendar import (
    CONSECUTIVE_LIMIT_LOOKBACK_DAYS,
    TradingCalendar,
)
from backend.market.limit_ladder import consecutive_limit_up_heights, max_height
from backend.data.numeric import safe_sum
from backend.data.providers.base import DataProvider


@dataclass(frozen=True)
class MarketFactor:
    name: str
    value: float
    weight: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class MarketState:
    date: str
    index_code: Optional[str] = None
    index_change: Optional[float] = None
    index_pct: Optional[float] = None
    advance_count: Optional[int] = None
    decline_count: Optional[int] = None
    flat_count: Optional[int] = None
    total_count: Optional[int] = None
    limit_up_count: Optional[int] = None
    limit_down_count: Optional[int] = None
    broken_count: Optional[int] = None
    limit_up_success_rate: Optional[float] = None
    max_consecutive_up: Optional[int] = None
    yesterday_limit_up_performance: Optional[float] = None
    total_amount: Optional[float] = None
    breadth: Optional[float] = None
    score: float = 50.0
    state: str = "中性"
    confidence: float = 0.0
    risk_level: str = "medium"
    recommended_exposure: float = 0.5
    factors: tuple[MarketFactor, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "index_code": self.index_code,
            "index_change": self.index_change,
            "index_pct": self.index_pct,
            "advance_count": self.advance_count,
            "decline_count": self.decline_count,
            "flat_count": self.flat_count,
            "total_count": self.total_count,
            "limit_up_count": self.limit_up_count,
            "limit_down_count": self.limit_down_count,
            "broken_count": self.broken_count,
            "limit_up_success_rate": self.limit_up_success_rate,
            "max_consecutive_up": self.max_consecutive_up,
            "yesterday_limit_up_performance": self.yesterday_limit_up_performance,
            "total_amount": self.total_amount,
            "breadth": self.breadth,
            "score": self.score,
            "state": self.state,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "recommended_exposure": self.recommended_exposure,
            "factors": [
                {
                    "name": f.name,
                    "value": f.value,
                    "weight": f.weight,
                    "description": f.description,
                }
                for f in self.factors
            ],
        }


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def classify_market(score: float) -> tuple[str, str, float]:
    """Map a 0-100 score to a state label, risk level and exposure."""
    if score >= 80:
        return "亢奋", "high", 1.0
    if score >= 65:
        return "强势", "medium", 0.8
    if score >= 55:
        return "偏强", "medium", 0.6
    if score >= 45:
        return "中性", "medium", 0.5
    if score >= 30:
        return "偏弱", "high", 0.3
    if score >= 15:
        return "弱势", "high", 0.15
    return "冰点", "high", 0.0


class MarketEngine:
    def __init__(self, provider: DataProvider, index_codes: Optional[list[str]] = None) -> None:
        self.provider = provider
        self.calendar = TradingCalendar(provider)
        self.index_codes = index_codes or ["000300.SH", "000905.SH", "399006.SZ"]

    def calculate(self, date: str) -> MarketState:
        basic = self._stock_basic()
        if basic.empty:
            return MarketState(date=date, state="数据不足", confidence=0.0, risk_level="high")

        all_codes = sorted(basic["ts_code"].astype(str).unique().tolist())
        # Consecutive-limit-up height and yesterday's limit-up carry need more
        # than a single prior day, so fetch a real recent window.
        recent = self.calendar.recent_trading_days(
            date, CONSECUTIVE_LIMIT_LOOKBACK_DAYS
        )
        lookback_start = recent[0] if recent else date
        daily = self.provider.daily(
            ts_codes=all_codes, start_date=lookback_start, end_date=date
        )
        if daily.empty:
            return MarketState(date=date, state="数据不足", confidence=0.0, risk_level="high")

        limit_df = self.provider.stk_limit(ts_codes=all_codes, trade_date=date)
        suspended = self._suspended_codes(all_codes, date)
        active = daily[
            (~daily["ts_code"].isin(suspended))
            if suspended
            else pd.Series(True, index=daily.index)
        ]

        today = active[active["trade_date"] == date].copy()
        if today.empty:
            return MarketState(date=date, state="数据不足", confidence=0.0, risk_level="high")

        advance = int((today["pct_chg"] > 0).sum())
        decline = int((today["pct_chg"] < 0).sum())
        flat = int((today["pct_chg"] == 0).sum())
        total = len(today)

        limit_up, limit_down, broken, success_rate = self._limit_metrics(today, limit_df)
        max_consecutive = self._max_consecutive_up(daily, date)
        yesterday_perf = self._yesterday_limit_up_performance(daily, all_codes, date)
        # Turnover stays None when the day was not collected from a source that
        # publishes it, instead of collapsing to a fake "0 元".
        total_amount = safe_sum(today["amount"]) if "amount" in today.columns else None
        breadth = advance / total if total else None

        index_pct, index_change = self._index_performance(date)

        score, factors = self._score(
            breadth=breadth,
            limit_up=limit_up,
            limit_down=limit_down,
            broken=broken,
            success_rate=success_rate,
            yesterday_perf=yesterday_perf,
            max_consecutive=max_consecutive,
            index_pct=index_pct,
            total=total,
        )
        state, risk_level, exposure = classify_market(score)
        confidence = self._confidence(total, limit_df, basic)

        return MarketState(
            date=date,
            index_change=index_change,
            index_pct=index_pct,
            advance_count=advance,
            decline_count=decline,
            flat_count=flat,
            total_count=total,
            limit_up_count=limit_up,
            limit_down_count=limit_down,
            broken_count=broken,
            limit_up_success_rate=success_rate,
            max_consecutive_up=max_consecutive,
            yesterday_limit_up_performance=yesterday_perf,
            total_amount=total_amount,
            breadth=breadth,
            score=score,
            state=state,
            confidence=confidence,
            risk_level=risk_level,
            recommended_exposure=exposure,
            factors=tuple(factors),
        )

    def _stock_basic(self) -> pd.DataFrame:
        try:
            basic = self.provider.stock_basic(list_status="L")
        except Exception:
            basic = self.provider.stock_basic()
        return basic

    def _suspended_codes(self, all_codes: list[str], date: str) -> set[str]:
        try:
            suspended = self.provider.suspend_d(ts_codes=all_codes, trade_date=date)
            if not suspended.empty and "ts_code" in suspended.columns:
                return set(suspended["ts_code"].astype(str).unique())
        except Exception:
            pass
        return set()

    def _limit_metrics(
        self, today: pd.DataFrame, limit_df: pd.DataFrame
    ) -> tuple[int, int, int, Optional[float]]:
        limit_up = 0
        limit_down = 0
        broken = 0
        success_rate: Optional[float] = None

        merged = today.copy()
        if not limit_df.empty:
            merged = today.merge(
                limit_df[["ts_code", "trade_date", "up_limit", "down_limit"]],
                on=["ts_code", "trade_date"],
                how="left",
            )
        else:
            merged["up_limit"] = pd.NA
            merged["down_limit"] = pd.NA

        has_limits = merged["up_limit"].notna() & merged["down_limit"].notna()
        limit_up = int(((merged["close"] >= merged["up_limit"]) & has_limits).sum())
        limit_down = int(((merged["close"] <= merged["down_limit"]) & has_limits).sum())
        broken = int(
            ((merged["high"] >= merged["up_limit"])
             & (merged["close"] < merged["up_limit"])
             & has_limits).sum()
        )
        attempted = limit_up + broken
        success_rate = (limit_up / attempted) if attempted else None
        return limit_up, limit_down, broken, success_rate

    def _max_consecutive_up(self, daily: pd.DataFrame, date: str) -> Optional[int]:
        if daily.empty:
            return None
        codes = sorted(daily["ts_code"].astype(str).unique().tolist())
        if not codes:
            return None
        heights = consecutive_limit_up_heights(
            daily,
            date,
            lambda day: self.provider.stk_limit(ts_codes=codes, trade_date=day),
        )
        return max_height(heights)

    def _yesterday_limit_up_performance(
        self, daily: pd.DataFrame, all_codes: list[str], date: str
    ) -> Optional[float]:
        yesterday = self.calendar.previous_trading_day(date)
        if yesterday is None:
            return None

        limit_df = self.provider.stk_limit(ts_codes=all_codes, trade_date=yesterday)
        if limit_df.empty:
            return None

        y_bars = daily[daily["trade_date"] == yesterday].merge(
            limit_df[["ts_code", "trade_date", "up_limit"]],
            on=["ts_code", "trade_date"],
            how="left",
        )
        limit_up_codes = y_bars.loc[
            y_bars["close"] >= y_bars["up_limit"], "ts_code"
        ].tolist()
        if not limit_up_codes:
            return None

        today_bars = daily[
            (daily["trade_date"] == date) & (daily["ts_code"].isin(limit_up_codes))
        ]
        if today_bars.empty:
            return None
        return float(today_bars["pct_chg"].mean())

    def _limit_series(self, ts_codes: list[str], dates: list[str]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for day in dates:
            try:
                frame = self.provider.stk_limit(ts_codes=ts_codes, trade_date=day)
                if not frame.empty:
                    frames.append(frame)
            except Exception:
                continue
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _index_performance(self, date: str) -> tuple[Optional[float], Optional[float]]:
        try:
            bars = self.provider.daily(ts_codes=self.index_codes, start_date=date, end_date=date)
        except Exception:
            bars = pd.DataFrame()
        if bars.empty:
            return None, None
        pct = float(bars["pct_chg"].mean())
        change = float(bars["change"].mean())
        return pct, change

    def _score(
        self,
        *,
        breadth: Optional[float],
        limit_up: int,
        limit_down: int,
        broken: int,
        success_rate: Optional[float],
        yesterday_perf: Optional[float],
        max_consecutive: Optional[int],
        index_pct: Optional[float],
        total: int,
    ) -> tuple[float, list[MarketFactor]]:
        factors: list[MarketFactor] = []
        score = 50.0

        if breadth is not None:
            contribution = (breadth - 0.5) * 100.0
            score += contribution
            factors.append(
                MarketFactor(
                    "breadth",
                    round(breadth, 4),
                    1.0,
                    f"上涨家数占比 {breadth:.1%}",
                )
            )

        if total:
            limit_ratio = (limit_up - limit_down) / total
            contribution = limit_ratio * 50.0
            score += contribution
            factors.append(
                MarketFactor(
                    "limit_differential",
                    round(limit_ratio, 4),
                    1.0,
                    f"涨停 {limit_up} / 跌停 {limit_down}",
                )
            )

        if broken:
            score -= min(15.0, broken * 2.0)
            factors.append(
                MarketFactor("broken_boards", float(broken), 1.0, f"炸板 {broken} 家")
            )

        if success_rate is not None:
            contribution = (success_rate - 0.6) * 25.0
            score += contribution
            factors.append(
                MarketFactor(
                    "limit_up_success",
                    round(success_rate, 4),
                    1.0,
                    f"涨停成功率 {success_rate:.1%}",
                )
            )

        if yesterday_perf is not None:
            contribution = yesterday_perf * 2.0
            score += contribution
            factors.append(
                MarketFactor(
                    "yesterday_limit_up_performance",
                    round(yesterday_perf, 2),
                    1.0,
                    f"昨日涨停今日平均 {yesterday_perf:+.2f}%",
                )
            )

        if max_consecutive:
            contribution = min(10.0, max_consecutive * 2.0)
            score += contribution
            factors.append(
                MarketFactor(
                    "consecutive_height",
                    float(max_consecutive),
                    1.0,
                    f"最高连板 {max_consecutive} 板",
                )
            )

        if index_pct is not None:
            contribution = index_pct * 2.0
            score += contribution
            factors.append(
                MarketFactor(
                    "index_performance",
                    round(index_pct, 2),
                    1.0,
                    f"主要指数平均 {index_pct:+.2f}%",
                )
            )

        return round(clamp(score), 2), factors

    def _confidence(self, total: int, limit_df: pd.DataFrame, basic: pd.DataFrame) -> float:
        score = 0.0
        if total >= 1000:
            score += 0.5
        elif total >= 100:
            score += 0.35
        elif total >= 20:
            score += 0.2
        if not limit_df.empty:
            score += 0.3
        if not basic.empty and "industry" in basic.columns:
            score += 0.2
        return round(min(1.0, score), 2)
