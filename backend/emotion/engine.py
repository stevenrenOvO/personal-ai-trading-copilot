"""A-share short-term emotion cycle engine (Step 3).

The stage comes from the ordered rule table in :mod:`backend.emotion.cycle`,
which reads the real market structure produced by Step 2's ladder engine
(heights, ladder distribution, promotion rate, yesterday's limit-up premium,
broken ratio, high-board performance, ladder gaps) plus breadth and turnover.

It deliberately no longer derives the phase from a single composite score:
that made the stage impossible to explain and hid contradictory signals.
``emotion_score`` is still reported, but as a documented display index whose
per-component contributions are exposed in ``strength_parts``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from backend.data.providers.base import DataProvider
from backend.emotion.cycle import (
    CycleMeasures,
    CycleVerdict,
    classify_cycle,
)
from backend.market.engine import MarketEngine, MarketState

if TYPE_CHECKING:  # imported lazily to avoid a board <-> emotion import cycle
    from backend.board.ladder import LadderSnapshot


EMOTION_PHASES = ("冰点", "修复", "发酵", "高潮", "分化", "退潮")


@dataclass(frozen=True)
class EmotionState:
    date: str
    emotion_cycle: str = "中性"
    emotion_score: float = 50.0
    confidence: float = 0.0
    risk_level: str = "medium"
    recommended_exposure: float = 0.5
    factors: tuple[str, ...] = ()
    market_score: Optional[float] = None
    advance_decline_ratio: Optional[float] = None
    limit_up_count: Optional[int] = None
    limit_down_count: Optional[int] = None
    limit_up_ratio: Optional[float] = None
    max_continuous_up: Optional[int] = None
    yesterday_limit_up_avg_pct: Optional[float] = None
    # Step 3 additions
    matched_rule: str = ""
    evidence: tuple[dict[str, Any], ...] = ()
    invalidations: tuple[str, ...] = ()
    strength_parts: tuple[dict[str, Any], ...] = ()
    basis_date: Optional[str] = None
    notes: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()
    promotion_rate: Optional[float] = None
    broken_ratio: Optional[float] = None
    high_board_avg_pct: Optional[float] = None
    # Backwards-compatible aliases used by earlier callers.
    sentiment: Optional[str] = None
    score: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "emotion_cycle": self.emotion_cycle,
            "emotion_score": self.emotion_score,
            "confidence": self.confidence,
            "risk_level": self.risk_level,
            "recommended_exposure": self.recommended_exposure,
            "factors": list(self.factors),
            "market_score": self.market_score,
            "advance_decline_ratio": self.advance_decline_ratio,
            "limit_up_count": self.limit_up_count,
            "limit_down_count": self.limit_down_count,
            "limit_up_ratio": self.limit_up_ratio,
            "max_continuous_up": self.max_continuous_up,
            "yesterday_limit_up_avg_pct": self.yesterday_limit_up_avg_pct,
            "matched_rule": self.matched_rule,
            "evidence": [dict(e) for e in self.evidence],
            "invalidations": list(self.invalidations),
            "strength_parts": [dict(p) for p in self.strength_parts],
            "basis_date": self.basis_date or self.date,
            "notes": list(self.notes),
            "unavailable": list(self.unavailable),
            "promotion_rate": self.promotion_rate,
            "broken_ratio": self.broken_ratio,
            "high_board_avg_pct": self.high_board_avg_pct,
            "sentiment": self.sentiment,
            "score": self.score,
        }


def classify_emotion(market: MarketState) -> tuple[str, float, str, float]:
    """Classify the emotion phase from a market state.

    Returns (phase, recommended_exposure, risk_level, cycle_score).
    """
    score = market.score
    limit_up = market.limit_up_count or 0
    limit_down = market.limit_down_count or 0
    broken = market.broken_count or 0
    success = market.limit_up_success_rate
    yesterday = market.yesterday_limit_up_performance
    height = market.max_consecutive_up or 0

    cycle_score = score

    if score <= 25:
        phase = "冰点"
        exposure = 0.0
        risk = "high"
    elif score <= 42:
        phase = "修复"
        exposure = 0.25
        risk = "high"
    elif score <= 58:
        phase = "发酵"
        exposure = 0.6
        risk = "medium"
    else:
        # High-score regime: distinguish climax vs differentiation by
        # breadth, broken boards and yesterday's limit-up carry.
        if broken >= 5 or (yesterday is not None and yesterday < -2.0) or (
            success is not None and success < 0.55
        ):
            phase = "分化"
            exposure = 0.4
            risk = "high"
        elif score >= 75 and height >= 4:
            phase = "高潮"
            exposure = 0.5
            risk = "high"
        else:
            phase = "发酵"
            exposure = 0.6
            risk = "medium"

    # A falling day after strong sentiment is "退潮"; low score with many
    # limit-downs is also a decline phase.
    if limit_down > limit_up and score < 50:
        phase = "退潮"
        exposure = 0.2
        risk = "high"

    return phase, exposure, risk, cycle_score


def measures_from(date: str, market: MarketState, ladder: Optional[LadderSnapshot]) -> CycleMeasures:
    """Assemble the rule inputs from the market state and the Step-2 ladder."""
    notes: list[str] = []
    missing: list[str] = []

    if ladder is None or not ladder.available:
        detail = "；".join(ladder.notes) if ladder is not None else "未提供连板结构"
        return CycleMeasures(
            date=date,
            available=False,
            limit_up_count=market.limit_up_count,
            max_height=market.max_consecutive_up,
            advance_count=market.advance_count,
            decline_count=market.decline_count,
            traded_count=market.total_count,
            total_amount=market.total_amount,
            missing=("连板梯队结构",),
            notes=(detail or "该交易日无本地连板数据",),
        )

    today = next((d for d in ladder.recent if d.date == date), None)
    if today is None:
        return CycleMeasures(
            date=date,
            available=False,
            missing=("连板梯队结构",),
            notes=(f"{date} 不在连板窗口内",),
        )
    previous = None
    for day in ladder.recent:
        if day.date < date:
            previous = day

    if ladder.excluded_new_listing:
        notes.append(f"{ladder.excluded_new_listing} 只次新股未纳入涨停判定")
    if ladder.excluded_corporate_action:
        notes.append(f"{ladder.excluded_corporate_action} 只除权/送转个股未纳入涨停判定")

    if market.total_amount is None:
        missing.append("成交额（回补日数据源未提供）")

    return CycleMeasures(
        date=date,
        available=True,
        max_height=today.max_height,
        prev_max_height=previous.max_height if previous else None,
        limit_up_count=today.limit_up_count,
        prev_limit_up_count=previous.limit_up_count if previous else None,
        second_board=today.buckets.get("二板"),
        third_board=today.buckets.get("三板"),
        high_board=today.buckets.get("四板及以上"),
        broken_count=today.broken_count,
        broken_ratio=today.broken_ratio,
        limit_down_count=today.limit_down_count,
        promotion_rate=today.promotion_rate,
        prev_promotion_rate=previous.promotion_rate if previous else None,
        yesterday_avg_premium=today.avg_premium,
        yesterday_up_ratio=today.up_ratio,
        high_board_avg_pct=today.high_board_avg_pct,
        ladder_complete=today.is_complete,
        gap_count=today.gap_count,
        advance_count=today.advance_count or market.advance_count,
        decline_count=today.decline_count or market.decline_count,
        traded_count=today.traded_count or market.total_count,
        total_amount=today.total_amount or market.total_amount,
        prev_total_amount=previous.total_amount if previous else None,
        missing=tuple(missing),
        notes=tuple(notes),
    )


class EmotionEngine:
    def __init__(self, provider: DataProvider) -> None:
        self.provider = provider
        self.market_engine = MarketEngine(provider)

    def calculate(
        self,
        date: str,
        market: Optional[MarketState] = None,
        ladder: Optional[LadderSnapshot] = None,
    ) -> EmotionState:
        """Build the emotion state from market structure.

        ``market`` and ``ladder`` can be supplied by a caller that already
        computed them (the dashboard computes both for other panels), so the
        full-market passes are not repeated.
        """
        if market is None:
            market = self.market_engine.calculate(date)
        if ladder is None:
            from backend.board.ladder import LimitLadderEngine

            ladder = LimitLadderEngine(self.provider).snapshot(date)
        measures = measures_from(date, market, ladder)
        verdict = classify_cycle(measures)
        return self._to_state(date, market, measures, verdict)

    def _to_state(
        self,
        date: str,
        market: MarketState,
        measures: CycleMeasures,
        verdict: CycleVerdict,
    ) -> EmotionState:
        phase = verdict.stage
        legacy_sentiment = {
            "冰点": "Fear",
            "修复": "Neutral Bearish",
            "发酵": "Neutral Bullish",
            "高潮": "Greed",
            "分化": "Neutral",
            "退潮": "Neutral Bearish",
        }.get(phase, "Neutral")

        factors = [e.text() for e in verdict.evidence]
        if market.breadth is not None:
            factors.append(f"市场广度 {market.breadth:.1%}")
        limit_up_ratio = None
        total = measures.traded_count or market.total_count
        if total and measures.limit_up_count is not None:
            limit_up_ratio = measures.limit_up_count / total

        return EmotionState(
            date=date,
            emotion_cycle=phase,
            emotion_score=verdict.strength,
            confidence=market.confidence,
            risk_level=verdict.risk_level,
            recommended_exposure=verdict.recommended_exposure,
            factors=tuple(factors),
            market_score=market.score,
            advance_decline_ratio=market.breadth,
            limit_up_count=measures.limit_up_count,
            limit_down_count=measures.limit_down_count,
            limit_up_ratio=limit_up_ratio,
            max_continuous_up=measures.max_height,
            yesterday_limit_up_avg_pct=measures.yesterday_avg_premium,
            matched_rule=verdict.matched_rule,
            evidence=tuple(e.to_dict() for e in verdict.evidence),
            invalidations=verdict.invalidations,
            strength_parts=verdict.strength_parts,
            basis_date=date,
            notes=verdict.notes,
            unavailable=verdict.unavailable,
            promotion_rate=measures.promotion_rate,
            broken_ratio=measures.broken_ratio,
            high_board_avg_pct=measures.high_board_avg_pct,
            sentiment=legacy_sentiment,
            score=verdict.strength,
        )
