"""Opportunity engine: the decision core of V1.0.

Raw strategy signals pass through a six-stage filter chain:

    Market Filter -> Emotion Filter -> Sector Filter -> Strategy Signal ->
    Risk Filter -> Opportunity Ranking

The output is a small, explainable set of ``Opportunity`` objects (Top 3~5),
never a raw dump of hundreds of names.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from backend.board.engine import BoardEngine, BoardEnvironment
from backend.data.providers.base import DataProvider
from backend.emotion.engine import EmotionEngine, EmotionState
from backend.market.engine import MarketEngine, MarketState
from backend.risk.rules import RiskManager, RiskContext
from backend.sector.engine import SectorEngine, SectorScore
from backend.strategies.base import Strategy, TradeSignal


@dataclass(frozen=True)
class RiskSummary:
    score: float = 0.0
    level: str = "medium"
    violations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "level": self.level,
            "violations": list(self.violations),
        }


@dataclass(frozen=True)
class Opportunity:
    ts_code: str
    date: str
    action: str
    price: Optional[float]
    strength: float
    reason: str
    sector: Optional[str]
    sentiment_score: float
    sector_rank: Optional[int]
    symbol: str = ""
    name: str = ""
    strategy_id: str = ""
    strategy_version: str = ""
    signal_id: str = ""
    role: str = ""
    score: float = 0.0
    suggested_action: str = "WATCH"
    # True only when nothing (environment or stock specific) blocks a buy.
    buy_ready: bool = False
    market_reason: str = ""
    emotion_reason: str = ""
    sector_reason: str = ""
    stock_reason: str = ""
    volume_reason: str = ""
    breakout_reason: str = ""
    risk: RiskSummary = field(default_factory=RiskSummary)
    invalid_conditions: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    environment_blockers: tuple[str, ...] = ()
    stock_blockers: tuple[str, ...] = ()
    invalidation_conditions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts_code": self.ts_code,
            "date": self.date,
            "symbol": self.symbol,
            "name": self.name,
            "strategy_id": self.strategy_id,
            "strategy_version": self.strategy_version,
            "signal_id": self.signal_id,
            "role": self.role,
            "action": self.action,
            "price": self.price,
            "strength": self.strength,
            "score": self.score,
            "suggested_action": self.suggested_action,
            "buy_ready": self.buy_ready,
            "sector": self.sector,
            "sector_rank": self.sector_rank,
            "sentiment_score": self.sentiment_score,
            "market_reason": self.market_reason,
            "emotion_reason": self.emotion_reason,
            "sector_reason": self.sector_reason,
            "stock_reason": self.stock_reason,
            "volume_reason": self.volume_reason,
            "breakout_reason": self.breakout_reason,
            "risk": self.risk.to_dict(),
            "invalid_conditions": list(self.invalid_conditions),
            "blockers": list(self.blockers),
            "environment_blockers": list(self.environment_blockers),
            "stock_blockers": list(self.stock_blockers),
            "invalidation_conditions": list(self.invalidation_conditions),
            "reason": self.reason,
        }


class OpportunityEngine:
    """Filter one day's raw signals into contextualized opportunities."""

    def __init__(
        self,
        provider: DataProvider,
        strategy: Strategy,
        sector_engine: SectorEngine,
        market_engine: Optional[MarketEngine] = None,
        emotion_engine: Optional[EmotionEngine] = None,
        board_engine: Optional[BoardEngine] = None,
        risk_manager: Optional[RiskManager] = None,
        max_results: int = 5,
    ) -> None:
        self.provider = provider
        self.strategy = strategy
        self.sector_engine = sector_engine
        self.market_engine = market_engine or MarketEngine(provider)
        self.emotion_engine = emotion_engine or EmotionEngine(provider)
        self.board_engine = board_engine or BoardEngine(provider)
        self.risk_manager = risk_manager or RiskManager()
        self.max_results = max_results

    def scan(
        self,
        date: str,
        lookback_days: int = 60,
        *,
        market: Optional[MarketState] = None,
        emotion: Optional[EmotionState] = None,
        board: Optional[BoardEnvironment] = None,
        sectors: Optional[list[SectorScore]] = None,
    ) -> list[Opportunity]:
        """Scan one day.

        ``market``/``emotion``/``board``/``sectors`` can be supplied by a caller
        that has already computed them (the dashboard computes the same values
        for its other panels), so the full-market passes are not repeated.
        """
        basic = self._stock_basic()
        code_to_name = {
            str(row["ts_code"]): (str(row.get("symbol", "")), str(row.get("name", "")))
            for _, row in basic.iterrows()
        }
        codes = sorted(code_to_name.keys())
        if not codes:
            return []

        start_date = self._lookback_start(date, lookback_days)
        daily = self.provider.daily(ts_codes=codes, start_date=start_date, end_date=date)
        if daily.empty:
            return []
        today = daily[daily["trade_date"] == date]
        pct_map = {
            str(row["ts_code"]): float(row["pct_chg"])
            for _, row in today.iterrows()
            if "pct_chg" in today.columns
        }

        market = market or self.market_engine.calculate(date)
        # Hand the already-computed market/emotion to the downstream engines:
        # each one otherwise repeats the same full-market pass.
        emotion = emotion or self.emotion_engine.calculate(date, market=market)
        board = board or self.board_engine.calculate(
            date, market=market, emotion=emotion
        )
        sectors = (
            sectors
            if sectors is not None
            else self.sector_engine.ranked(date)
        )
        sector_by_name = {s.name: s for s in sectors}
        sector_by_code: dict[str, str] = {}
        for sector in sectors:
            for code in sector.stock_codes:
                sector_by_code[code] = sector.name

        signals = self._date_signals(daily, date, sectors, sector_by_code)
        if not signals:
            return []

        opportunities: list[Opportunity] = []
        for signal in signals:
            opp = self._evaluate_signal(
                signal=signal,
                date=date,
                code_to_name=code_to_name,
                market=market,
                emotion=emotion,
                board=board,
                sector_by_code=sector_by_code,
                sector_by_name=sector_by_name,
                ranked_sectors=sectors,
                pct_map=pct_map,
            )
            if opp is not None:
                opportunities.append(opp)

        opportunities.sort(key=lambda item: item.score, reverse=True)
        return opportunities[: self.max_results]

    def _date_signals(
        self,
        daily: pd.DataFrame,
        date: str,
        sectors: list[SectorScore],
        sector_by_code: dict[str, str],
    ) -> list[TradeSignal]:
        """Strategy signals for one day, with the sector ranking attached.

        ``StrongSectorBreakoutStrategy`` is documented as preferring stocks in
        strong sectors, but nothing ever called ``set_sector_context`` -- so the
        candidate list was chosen ignoring sectors and then rejected by the
        sector filter, leaving every row marked "回避". The ranking is attached
        here, on a per-scan copy so two concurrent scans for different dates
        cannot overwrite each other's context.
        """
        strategy = self.strategy
        setter = getattr(strategy, "set_sector_context", None)
        if setter is not None and sectors:
            ranks = {sector.name: sector.rank for sector in sectors}
            strategy = copy.copy(strategy)
            setter = strategy.set_sector_context
            setter(sector_by_code, ranks)
        try:
            return [
                signal
                for signal in strategy.generate_signals(daily)
                if signal.trade_date == date
            ]
        except Exception:
            return []

    def _evaluate_signal(
        self,
        *,
        signal: TradeSignal,
        date: str,
        code_to_name: dict[str, tuple[str, str]],
        market,
        emotion,
        board,
        sector_by_code: dict[str, str],
        sector_by_name: dict[str, SectorScore],
        ranked_sectors: list[SectorScore],
        pct_map: dict[str, float],
    ) -> Optional[Opportunity]:
        symbol, name = code_to_name.get(signal.ts_code, ("", ""))
        sector_name = sector_by_code.get(signal.ts_code)
        sector = sector_by_name.get(sector_name) if sector_name else None
        sector_rank = (ranked_sectors.index(sector) + 1) if sector in ranked_sectors else None
        pct_chg = pct_map.get(signal.ts_code, 0.0)
        role = self._classify_role(
            ts_code=signal.ts_code,
            sector=sector,
            sector_rank=sector_rank,
            pct_chg=pct_chg,
        )
        invalidation_conditions = self._invalidation_conditions(
            market_score=market.score,
            emotion_cycle=emotion.emotion_cycle,
            sector_rank=sector_rank,
            board_grade=board.grade,
        )

        # Six-stage filter chain, all producing reasons rather than silent drops.
        invalid: list[str] = []
        market_reason = f"市场状态 {market.state}（{market.score:.0f} 分）"
        emotion_reason = f"情绪周期 {emotion.emotion_cycle}（建议仓位 {emotion.recommended_exposure:.0%}）"
        sector_reason = (
            f"板块 {sector_name} 排名第 {sector_rank}"
            if sector_name and sector_rank
            else "无板块信息"
        )
        stock_reason = signal.reason or "策略信号"
        volume_reason = self._volume_reason(signal.ts_code, date)
        breakout_reason = signal.reason

        # The six-stage chain records *why* something is blocked, and separates
        # two very different situations:
        #   environment blockers - the market/emotion/board is not supportive,
        #                          but the stock itself may still be worth watching
        #   stock blockers       - this particular name/sector is not a candidate
        # Collapsing both into one list is what made every row on a weak day read
        # "回避", which told the user nothing about what to actually do.
        environment_blockers: list[str] = []
        stock_blockers: list[str] = []

        # Market filter
        if market.score < 30:
            environment_blockers.append(f"市场过弱({market.score:.0f}分)")
        # Emotion filter
        if emotion.emotion_cycle in ("冰点", "退潮"):
            environment_blockers.append(f"情绪处于{emotion.emotion_cycle}")
        # Sector filter (stock-specific: this name is in the wrong place)
        if sector is not None and sector.score < 45:
            stock_blockers.append(f"板块强度不足({sector.score:.0f}分)")
        if sector_rank is not None and sector_rank > 5:
            stock_blockers.append(f"板块排名第{sector_rank}，不在前5")
        # Board filter (environment)
        if board.grade in ("C", "D"):
            environment_blockers.append(f"涨停环境较差({board.grade}级)")
        # Strategy filter
        if not self.strategy.enabled:
            stock_blockers.append("策略已禁用")

        # Risk filter on a synthetic empty portfolio (no existing positions).
        risk_ctx = RiskContext(
            equity=1_000_000.0,
            cash=1_000_000.0,
            peak_equity=1_000_000.0,
            prices={signal.ts_code: float(signal.price or 0.0)},
        )
        risk_result = self.risk_manager.check_signal(signal, risk_ctx)
        risk_violations = tuple(v.message for v in risk_result.violations)
        stock_blockers.extend(risk_violations)

        risk_score = max(0.0, min(100.0, 100.0 - len(risk_violations) * 25.0))
        if len(risk_violations) >= 3:
            risk_level = "high"
        elif risk_violations:
            risk_level = "medium"
        else:
            risk_level = "low"

        # Composite score: signal, market, sector, emotion, board. This is a
        # *ranking* score -- environment problems are expressed through
        # ``buy_ready`` below instead of crushing the score, so the list still
        # orders candidates sensibly on a weak day.
        score = (
            signal.strength * 30
            + (market.score / 100.0) * 20
            + ((sector.score / 100.0) if sector else 0.5) * 20
            + (emotion.emotion_score / 100.0) * 15
            + (board.score / 100.0) * 15
        )
        score = max(0.0, min(100.0, score))
        if stock_blockers:
            score *= 0.7

        invalid = environment_blockers + stock_blockers
        buy_ready = not invalid and score >= 60

        suggested_action = self._suggested_action(
            signal=signal,
            score=score,
            environment_blockers=environment_blockers,
            stock_blockers=stock_blockers,
            risk_level=risk_level,
        )

        return Opportunity(
            ts_code=signal.ts_code,
            date=date,
            symbol=symbol,
            name=name,
            strategy_id=signal.strategy_id,
            strategy_version=signal.strategy_version,
            signal_id=signal.signal_id,
            role=role,
            action=signal.action,
            price=signal.price,
            strength=signal.strength,
            score=round(score, 2),
            suggested_action=suggested_action,
            sector=sector_name,
            sector_rank=sector_rank,
            sentiment_score=emotion.emotion_score,
            market_reason=market_reason,
            emotion_reason=emotion_reason,
            sector_reason=sector_reason,
            stock_reason=stock_reason,
            volume_reason=volume_reason,
            breakout_reason=breakout_reason,
            risk=RiskSummary(
                score=round(risk_score, 2),
                level=risk_level,
                violations=risk_violations,
            ),
            invalid_conditions=tuple(invalid),
            blockers=tuple(invalid),
            environment_blockers=tuple(environment_blockers),
            stock_blockers=tuple(stock_blockers),
            buy_ready=buy_ready,
            invalidation_conditions=invalidation_conditions,
            reason=signal.reason,
        )

    def _volume_reason(self, ts_code: str, date: str) -> str:
        try:
            bars = self.provider.daily(ts_codes=[ts_code], start_date=date, end_date=date)
            if bars.empty or "vol" not in bars.columns:
                return "无成交量数据"
            return f"当日成交量 {float(bars.iloc[0]['vol']):,.0f}"
        except Exception:
            return "无成交量数据"

    def _classify_role(
        self,
        *,
        ts_code: str,
        sector,
        sector_rank: Optional[int],
        pct_chg: float,
    ) -> str:
        """Classify a short-term role from sector rank and leader status."""
        if sector is None:
            return "普通"
        is_leader = sector.leader == ts_code
        if sector_rank is not None and sector_rank <= 3 and is_leader:
            return "空间龙"
        if sector_rank is not None and sector_rank <= 3:
            return "中军" if pct_chg >= 2.0 else "补涨龙"
        if sector_rank is not None and sector_rank <= 5 and pct_chg > 0:
            return "补涨龙"
        return "普通"

    def _invalidation_conditions(
        self,
        *,
        market_score: float,
        emotion_cycle: str,
        sector_rank: Optional[int],
        board_grade: str,
    ) -> tuple[str, ...]:
        """Conditions that would invalidate an opportunity."""
        conditions = [
            "收盘价跌破突破前高",
            "市场评分跌破 30",
            "情绪进入退潮或冰点",
            "涨停环境转弱至 C/D 级",
        ]
        if sector_rank is not None:
            conditions.insert(0, "板块排名跌出前 5")
        return tuple(conditions)

    def _suggested_action(
        self,
        *,
        signal: TradeSignal,
        score: float,
        environment_blockers: list[str],
        stock_blockers: list[str],
        risk_level: str,
    ) -> str:
        """What to do about this name *today*.

        "AVOID" is reserved for names that are wrong for this strategy (weak or
        low-ranked sector, risk violations). When only the environment is
        against us the honest answer is "wait", not "avoid" -- the user still
        needs to know what to watch and what would make it actionable.
        """
        if signal.action == "SELL":
            return "REDUCE"
        if stock_blockers and risk_level == "high":
            return "AVOID"
        if stock_blockers:
            return "AVOID"
        if environment_blockers:
            return "WAIT"
        if score >= 75:
            return "BUY WATCH"
        if score >= 60:
            return "WATCH"
        if score >= 45:
            return "WAIT"
        return "AVOID"

    def _stock_basic(self):
        try:
            basic = self.provider.stock_basic(list_status="L")
        except Exception:
            basic = self.provider.stock_basic()
        return basic

    def _lookback_start(self, date: str, lookback_days: int) -> str:
        try:
            return (
                datetime.strptime(date, "%Y%m%d") - timedelta(days=lookback_days)
            ).strftime("%Y%m%d")
        except ValueError:
            return date
