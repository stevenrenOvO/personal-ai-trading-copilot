"""Board trading environment for A-share limit-up / ladder analysis.

Produces a ``BoardEnvironment`` with a A/B+/B/C/D grade plus a full explainable
factor list. The engine derives the board ladder from consecutive limit-up
days and never treats "limit-up" as an automatic buy signal: climax, high
differentiation and decline phases are explicitly risk-gated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from backend.data.calendar import (
    CONSECUTIVE_LIMIT_LOOKBACK_DAYS,
    TradingCalendar,
)
from backend.data.providers.base import DataProvider
from backend.emotion.engine import EmotionEngine, EmotionState
from backend.market.engine import MarketEngine, MarketState
from backend.market.limit_ladder import consecutive_limit_up_heights


@dataclass(frozen=True)
class BoardLadder:
    first_board: int = 0
    second_board: int = 0
    third_board: int = 0
    high_board: int = 0
    max_height: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_board": self.first_board,
            "second_board": self.second_board,
            "third_board": self.third_board,
            "high_board": self.high_board,
            "max_height": self.max_height,
        }


@dataclass(frozen=True)
class BoardEnvironment:
    date: str
    grade: str = "C"
    score: float = 50.0
    emotion_cycle: str = "中性"
    limit_up_count: int = 0
    broken_count: int = 0
    reseal_count: int = 0
    yesterday_limit_up_premium: Optional[float] = None
    leader: Optional[str] = None
    leader_name: Optional[str] = None
    leader_height: int = 0
    ladder: BoardLadder = BoardLadder()
    risk_level: str = "medium"
    recommended_exposure: float = 0.5
    factors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "grade": self.grade,
            "score": self.score,
            "emotion_cycle": self.emotion_cycle,
            "limit_up_count": self.limit_up_count,
            "broken_count": self.broken_count,
            "reseal_count": self.reseal_count,
            "yesterday_limit_up_premium": self.yesterday_limit_up_premium,
            "leader": self.leader,
            "leader_name": self.leader_name,
            "leader_height": self.leader_height,
            "ladder": self.ladder.to_dict(),
            "risk_level": self.risk_level,
            "recommended_exposure": self.recommended_exposure,
            "factors": list(self.factors),
        }


def grade_board(score: float) -> str:
    if score >= 85:
        return "A"
    if score >= 75:
        return "B+"
    if score >= 60:
        return "B"
    if score >= 45:
        return "C"
    return "D"


class BoardEngine:
    """Manage stock pools, filters and the board-trading environment."""

    def __init__(self, provider: DataProvider) -> None:
        self.provider = provider
        self.calendar = TradingCalendar(provider)
        self.market_engine = MarketEngine(provider)
        self.emotion_engine = EmotionEngine(provider)
        self._name_map: dict[str, str] | None = None

    def _names(self) -> dict[str, str]:
        if self._name_map is not None:
            return self._name_map
        try:
            basic = self.provider.stock_basic(list_status="L")
        except Exception:
            basic = self.provider.stock_basic()
        mapping: dict[str, str] = {}
        if not basic.empty and "ts_code" in basic.columns and "name" in basic.columns:
            for _, row in basic.iterrows():
                mapping[str(row["ts_code"])] = str(row.get("name", "") or "")
        self._name_map = mapping
        return mapping

    def get_index_constituents(self, index_code: str = "000300.SH") -> list[str]:
        """Return a conservative listed-stock universe.

        A true index-constituent feed can replace this later without changing
        the board API.
        """
        try:
            basic = self.provider.stock_basic(list_status="L")
        except Exception:
            basic = self.provider.stock_basic()
        if basic.empty:
            return []
        return sorted(basic["ts_code"].astype(str).unique().tolist())

    def filter_by_conditions(
        self,
        ts_codes: list[str],
        min_price: Optional[float] = None,
        max_pct_chg: Optional[float] = None,
        date: Optional[str] = None,
    ) -> list[str]:
        """Return codes that pass simple price and change filters on a date."""
        if not ts_codes:
            return []
        if date is None:
            return sorted(set(ts_codes))

        daily = self.provider.daily(
            ts_codes=list(dict.fromkeys(ts_codes)),
            start_date=date,
            end_date=date,
        )
        if daily.empty:
            return []
        if min_price is not None:
            daily = daily[daily["close"] >= float(min_price)]
        if max_pct_chg is not None:
            daily = daily[daily["pct_chg"] <= float(max_pct_chg)]
        return sorted(daily["ts_code"].astype(str).unique().tolist())

    def calculate(
        self,
        date: str,
        *,
        market: Optional[MarketState] = None,
        emotion: Optional[EmotionState] = None,
    ) -> BoardEnvironment:
        """Build the board environment.

        ``market``/``emotion`` can be supplied by the caller so a pipeline that
        already computed them does not pay for the same full-market pass again.
        """
        market = market or self.market_engine.calculate(date)
        emotion = emotion or self.emotion_engine.calculate(date, market=market)
        codes = self.get_index_constituents()
        if not codes:
            return self._empty_env(date, market, emotion)

        # A real lookback window: with only [previous_day, date] every height
        # was capped at 2, so 首板/二板/最高连板 were wrong for any longer run.
        window = self.calendar.recent_trading_days(date, CONSECUTIVE_LIMIT_LOOKBACK_DAYS)
        start = window[0] if window else date
        daily = self.provider.daily(ts_codes=codes, start_date=start, end_date=date)
        limit_df = self.provider.stk_limit(ts_codes=codes, trade_date=date)
        if daily.empty:
            return self._empty_env(date, market, emotion)

        today = daily[daily["trade_date"] == date].copy()
        if today.empty:
            return self._empty_env(date, market, emotion)

        heights = self._board_heights(daily, date)
        ladder = self._ladder_from_heights(heights)
        limit_up_codes = set(heights.keys())
        limit_up_count = len(limit_up_codes)

        merged = self._merge_limits(today, limit_df)
        broken_count = int(
            ((merged["high"] >= merged["up_limit"]) & (merged["close"] < merged["up_limit"])).sum()
        )
        reseal_count = int(
            (
                (merged["high"] >= merged["up_limit"])
                & (merged["low"] < merged["up_limit"])
                & (merged["close"] >= merged["up_limit"])
            ).sum()
        )
        yesterday_premium = market.yesterday_limit_up_performance

        leader: Optional[str] = None
        leader_name: Optional[str] = None
        leader_height = 0
        if heights:
            leader = max(heights, key=lambda code: (heights[code], code))
            leader_height = heights[leader]
            leader_name = self._names().get(leader, "")

        score, grade, risk, exposure, factors = self._score_environment(
            market=market,
            emotion=emotion,
            limit_up_count=limit_up_count,
            broken_count=broken_count,
            reseal_count=reseal_count,
            yesterday_premium=yesterday_premium,
            max_height=ladder.max_height,
            ladder=ladder,
        )

        return BoardEnvironment(
            date=date,
            grade=grade,
            score=score,
            emotion_cycle=emotion.emotion_cycle,
            limit_up_count=limit_up_count,
            broken_count=broken_count,
            reseal_count=reseal_count,
            yesterday_limit_up_premium=yesterday_premium,
            leader=leader,
            leader_name=leader_name,
            leader_height=leader_height,
            ladder=ladder,
            risk_level=risk,
            recommended_exposure=exposure,
            factors=tuple(factors),
        )

    def ladder(self, date: str) -> BoardLadder:
        codes = self.get_index_constituents()
        window = self.calendar.recent_trading_days(date, CONSECUTIVE_LIMIT_LOOKBACK_DAYS)
        start = window[0] if window else date
        daily = self.provider.daily(ts_codes=codes, start_date=start, end_date=date)
        heights = self._board_heights(daily, date)
        return self._ladder_from_heights(heights)

    def _empty_env(self, date: str, market: MarketState, emotion: EmotionState) -> BoardEnvironment:
        return BoardEnvironment(
            date=date,
            grade="D",
            score=0.0,
            emotion_cycle=emotion.emotion_cycle,
            risk_level="high",
            recommended_exposure=0.0,
            factors=("数据不足",),
        )

    def _board_heights(self, daily: pd.DataFrame, date: str) -> dict[str, int]:
        """Count consecutive limit-up days ending on ``date`` per stock."""
        codes = sorted(daily["ts_code"].astype(str).unique())
        return consecutive_limit_up_heights(
            daily,
            date,
            lambda day: self.provider.stk_limit(ts_codes=codes, trade_date=day),
        )

    def _ladder_from_heights(self, heights: dict[str, int]) -> BoardLadder:
        return BoardLadder(
            first_board=sum(1 for h in heights.values() if h == 1),
            second_board=sum(1 for h in heights.values() if h == 2),
            third_board=sum(1 for h in heights.values() if h == 3),
            high_board=sum(1 for h in heights.values() if h >= 4),
            max_height=max(heights.values()) if heights else 0,
        )

    def _merge_limits(self, today: pd.DataFrame, limit_df: pd.DataFrame) -> pd.DataFrame:
        if limit_df.empty:
            merged = today.copy()
            merged["up_limit"] = pd.NA
            merged["down_limit"] = pd.NA
            return merged
        return today.merge(
            limit_df[["ts_code", "trade_date", "up_limit", "down_limit"]],
            on=["ts_code", "trade_date"],
            how="left",
        )

    def _score_environment(
        self,
        *,
        market: MarketState,
        emotion: EmotionState,
        limit_up_count: int,
        broken_count: int,
        reseal_count: int,
        yesterday_premium: Optional[float],
        max_height: int,
        ladder: BoardLadder,
    ) -> tuple[float, str, str, float, list[str]]:
        score = market.score
        factors: list[str] = [f"市场强度分 {market.score:.0f}"]

        score += min(10.0, limit_up_count * 0.5)
        factors.append(f"涨停 {limit_up_count} 家")

        if broken_count:
            score -= min(15.0, broken_count * 3.0)
            factors.append(f"炸板 {broken_count} 家")
        if reseal_count:
            score += min(8.0, reseal_count * 2.0)
            factors.append(f"回封 {reseal_count} 家")
        if yesterday_premium is not None:
            score += max(-15.0, min(15.0, yesterday_premium * 2.0))
            factors.append(f"昨日涨停溢价 {yesterday_premium:+.2f}%")
        if max_height:
            score += min(8.0, max_height * 1.5)
            factors.append(f"最高 {max_height} 连板")

        # Risk gates: never reward a heated board blindly.
        if emotion.emotion_cycle in ("高潮", "分化", "退潮"):
            score -= 12.0
            factors.append(f"情绪周期 {emotion.emotion_cycle}，降低机会评分")
        if broken_count > limit_up_count:
            score -= 10.0
            factors.append("炸板数多于涨停数，打板环境差")
        if ladder.first_board == 0 and ladder.second_board == 0 and max_height < 2:
            score -= 8.0
            factors.append("连板梯队断档")

        score = max(0.0, min(100.0, score))
        grade = grade_board(score)
        if grade in ("A", "B+"):
            risk, exposure = "low", 0.8
        elif grade == "B":
            risk, exposure = "medium", 0.6
        elif grade == "C":
            risk, exposure = "high", 0.4
        else:
            risk, exposure = "high", 0.15
        return round(score, 2), grade, risk, exposure, factors
