"""Reusable risk checks for trading signals and portfolio state.

The rules in this module are intentionally small and dependency-free. They
consume normalized signal/portfolio snapshots so they can be used both before
an order is submitted and as a post-trade safety review.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from backend.strategies.base import TradeSignal


@dataclass(frozen=True)
class PositionInfo:
    """A compact view of one position as seen by the risk layer."""

    ts_code: str
    quantity: float
    avg_price: float

    def market_value(self, price: float) -> float:
        return self.quantity * price


@dataclass
class RiskContext:
    """Portfolio snapshot required by risk rules."""

    equity: float
    cash: float
    peak_equity: float
    positions: dict[str, PositionInfo] = field(default_factory=dict)
    prices: dict[str, float] = field(default_factory=dict)

    @property
    def current_drawdown(self) -> float:
        if self.peak_equity <= 0:
            return 0.0
        return self.equity / self.peak_equity - 1.0


@dataclass(frozen=True)
class RiskViolation:
    """A single rule failure or warning."""

    rule: str
    message: str
    ts_code: str | None = None
    severity: str = "BLOCK"


@dataclass(frozen=True)
class RiskCheckResult:
    """Aggregated output of one or more risk rules."""

    violations: tuple[RiskViolation, ...] = ()

    @property
    def allowed(self) -> bool:
        return not any(v.severity == "BLOCK" for v in self.violations)


class RiskRule(ABC):
    """Base class for signal and portfolio risk rules."""

    name: str = "risk_rule"

    @abstractmethod
    def check(
        self,
        signal: TradeSignal | None,
        context: RiskContext,
    ) -> list[RiskViolation]:
        """Return violations for the supplied signal and portfolio context."""


class MaxPositionWeightRule(RiskRule):
    """Block additional buys when a single position is already overweight."""

    name = "max_position_weight"

    def __init__(self, max_weight: float = 0.25) -> None:
        if max_weight <= 0:
            raise ValueError("max_weight must be positive")
        self.max_weight = max_weight

    def check(
        self,
        signal: TradeSignal | None,
        context: RiskContext,
    ) -> list[RiskViolation]:
        if signal is None or signal.action != "BUY":
            return []

        position = context.positions.get(signal.ts_code)
        if position is None or position.quantity <= 0:
            return []

        price = context.prices.get(signal.ts_code)
        if price is None:
            price = signal.price
        if price is None:
            price = position.avg_price
        if price <= 0 or context.equity <= 0:
            return []

        weight = position.market_value(price) / context.equity
        if weight > self.max_weight:
            return [
                RiskViolation(
                    rule=self.name,
                    ts_code=signal.ts_code,
                    message=(
                        f"{signal.ts_code} position weight {weight:.2%} exceeds "
                        f"limit {self.max_weight:.2%}"
                    ),
                )
            ]
        return []


class MaxDrawdownRule(RiskRule):
    """Block new buys when portfolio drawdown breaches the configured limit."""

    name = "max_drawdown"

    def __init__(self, max_drawdown: float = 0.20) -> None:
        if max_drawdown <= 0:
            raise ValueError("max_drawdown must be positive")
        self.max_drawdown = max_drawdown

    def check(
        self,
        signal: TradeSignal | None,
        context: RiskContext,
    ) -> list[RiskViolation]:
        if signal is None or signal.action != "BUY":
            return []

        if context.current_drawdown <= -self.max_drawdown:
            return [
                RiskViolation(
                    rule=self.name,
                    message=(
                        f"portfolio drawdown {context.current_drawdown:.2%} "
                        f"exceeds limit -{self.max_drawdown:.2%}"
                    ),
                )
            ]
        return []


class StopLossRule(RiskRule):
    """Flag positions whose current loss exceeds the stop-loss threshold."""

    name = "stop_loss"

    def __init__(self, stop_loss_pct: float = 0.08) -> None:
        if stop_loss_pct <= 0:
            raise ValueError("stop_loss_pct must be positive")
        self.stop_loss_pct = stop_loss_pct

    def check(
        self,
        signal: TradeSignal | None,
        context: RiskContext,
    ) -> list[RiskViolation]:
        violations: list[RiskViolation] = []

        for ts_code, position in context.positions.items():
            if position.quantity <= 0:
                continue

            price = context.prices.get(ts_code)
            if price is None:
                price = signal.price if signal is not None and signal.ts_code == ts_code else None
            if price is None:
                price = position.avg_price
            if price <= 0 or position.avg_price <= 0:
                continue

            change = price / position.avg_price - 1.0
            if change <= -self.stop_loss_pct:
                violations.append(
                    RiskViolation(
                        rule=self.name,
                        ts_code=ts_code,
                        message=(
                            f"{ts_code} loss {change:.2%} breaches stop-loss "
                            f"threshold -{self.stop_loss_pct:.2%}"
                        ),
                    )
                )

        return violations


class RiskManager:
    """Apply a set of risk rules to signals and portfolio snapshots."""

    def __init__(self, rules: Iterable[RiskRule] | None = None) -> None:
        self.rules = list(
            rules
            if rules is not None
            else [
                MaxPositionWeightRule(),
                MaxDrawdownRule(),
                StopLossRule(),
            ]
        )

    def check_signal(
        self,
        signal: TradeSignal,
        context: RiskContext,
    ) -> RiskCheckResult:
        violations: list[RiskViolation] = []
        for rule in self.rules:
            violations.extend(rule.check(signal, context))
        return RiskCheckResult(tuple(violations))

    def check_portfolio(self, context: RiskContext) -> RiskCheckResult:
        violations: list[RiskViolation] = []
        for rule in self.rules:
            violations.extend(rule.check(None, context))
        return RiskCheckResult(tuple(violations))


@dataclass(frozen=True)
class RiskDecision:
    """A portfolio-level action decision with explanation."""

    action: str  # BUY | HOLD | WAIT | REDUCE | EXIT
    score: float
    level: str
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "score": self.score,
            "level": self.level,
            "reasons": list(self.reasons),
        }


class DecisionRiskEngine:
    """Combine rule violations and market/emotion context into one action."""

    ACTION_PRIORITY = ("EXIT", "REDUCE", "WAIT", "HOLD", "BUY")

    def __init__(self, risk_manager: Optional[RiskManager] = None) -> None:
        self.risk_manager = risk_manager or RiskManager()

    def decide(
        self,
        context: RiskContext,
        *,
        market: Any | None = None,
        emotion: Any | None = None,
        sector: Any | None = None,
        has_positions: bool = False,
        signal: TradeSignal | None = None,
    ) -> RiskDecision:
        reasons: list[str] = []

        result = (
            self.risk_manager.check_signal(signal, context)
            if signal is not None
            else self.risk_manager.check_portfolio(context)
        )
        block_reasons = [v.message for v in result.violations if v.severity == "BLOCK"]
        reasons.extend(block_reasons)

        score = max(0.0, 100.0 - len(block_reasons) * 30.0)

        if market is not None:
            mscore = float(getattr(market, "score", 50.0) or 50.0)
            if mscore < 30:
                reasons.append(f"市场过弱（{mscore:.0f} 分）")
                score -= 25.0
            elif mscore < 45:
                reasons.append(f"市场偏弱（{mscore:.0f} 分）")
                score -= 10.0

        if emotion is not None:
            cycle = getattr(emotion, "emotion_cycle", "中性")
            if cycle in ("冰点", "退潮"):
                reasons.append(f"情绪处于{cycle}")
                score -= 20.0
            elif cycle in ("分化", "高潮"):
                reasons.append(f"情绪处于{cycle}，追高风险上升")
                score -= 10.0

        if sector is not None:
            sscore = float(getattr(sector, "score", 50.0) or 50.0)
            if sscore < 40:
                reasons.append(f"板块强度不足（{sscore:.0f} 分）")
                score -= 15.0

        score = max(0.0, min(100.0, score))

        if block_reasons:
            action = "EXIT" if has_positions else "WAIT"
        elif score < 40:
            action = "REDUCE" if has_positions else "WAIT"
        elif score < 55:
            action = "HOLD" if has_positions else "WAIT"
        elif signal is not None and signal.action == "BUY":
            action = "BUY"
        else:
            action = "HOLD"

        level = "high" if score < 45 else ("low" if score >= 70 else "medium")
        return RiskDecision(
            action=action,
            score=round(score, 2),
            level=level,
            reasons=tuple(reasons),
        )
