"""Decision audit on real trade records.

Every audit finding is grounded in the recorded trade context (market state,
emotion state, system recommendation, position sizing) rather than an LLM
guessing. Findings are bucketed into Strategy / Execution / Risk / Emotional
errors with evidence and a concrete suggestion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from backend.journal.trades import TradeRecord


@dataclass(frozen=True)
class AuditFinding:
    error_type: str
    severity: str
    evidence: str
    suggestion: str

    def to_dict(self) -> dict[str, str]:
        return {
            "error_type": self.error_type,
            "severity": self.severity,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
        }


@dataclass
class AuditResult:
    trade_id: str
    findings: list[AuditFinding] = field(default_factory=list)
    overall: str = "OK"
    sample_sufficient: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "findings": [f.to_dict() for f in self.findings],
            "overall": self.overall,
            "sample_sufficient": self.sample_sufficient,
        }


class TradeAuditor:
    """Audit a trade record against its own logged context."""

    def audit(self, trade: TradeRecord) -> AuditResult:
        findings: list[AuditFinding] = []

        # Risk errors: position sizing and drawdown-driven entries.
        if trade.action == "BUY" and trade.quantity > 0 and trade.position_after > trade.position_before:
            if trade.position_after > 0.25 * max(trade.position_after, trade.quantity):
                findings.append(
                    AuditFinding(
                        error_type="Risk Error",
                        severity="high",
                        evidence=f"加仓后仓位 {trade.position_after:.0f} 股，权重可能过高",
                        suggestion="检查单票仓位上限，分批而非一次性加满",
                    )
                )

        # Execution errors: ignoring the system recommendation.
        rec = trade.system_recommendation.upper()
        action = trade.action.upper()
        if rec and action in ("BUY", "SELL"):
            if rec in ("AVOID", "EXIT", "REDUCE") and action == "BUY":
                findings.append(
                    AuditFinding(
                        error_type="Execution Error",
                        severity="high",
                        evidence=f"系统建议 {rec}，实际执行 {action}",
                        suggestion="遵守系统建议；若确实反向，需在 user_reason 中写明依据",
                    )
                )
            if rec == "BUY WATCH" and action == "SELL":
                findings.append(
                    AuditFinding(
                        error_type="Execution Error",
                        severity="medium",
                        evidence=f"系统建议观察/买入，实际执行卖出",
                        suggestion="卖出前确认是否触发止损或结构破坏",
                    )
                )

        # Emotional / market errors: buying into weak markets or declined phases.
        weak_market = any(token in trade.market_state for token in ("弱", "冰点", "数据不足"))
        weak_emotion = any(token in trade.emotion_state for token in ("冰点", "退潮", "分化", "高潮"))
        if action == "BUY" and weak_market:
            findings.append(
                AuditFinding(
                    error_type="Emotional Error",
                    severity="high",
                    evidence=f"市场状态为「{trade.market_state}」时买入",
                    suggestion="弱市优先等待修复或降低仓位",
                )
            )
        if action == "BUY" and weak_emotion:
            findings.append(
                AuditFinding(
                    error_type="Emotional Error",
                    severity="medium",
                    evidence=f"情绪周期为「{trade.emotion_state}」时买入",
                    suggestion="高潮/退潮/分化阶段减少追高",
                )
            )

        # Strategy errors: missing signal identity.
        if not trade.strategy_id or not trade.signal_id:
            findings.append(
                AuditFinding(
                    error_type="Strategy Error",
                    severity="low",
                    evidence="交易未关联 strategy_id / signal_id",
                    suggestion="所有交易应记录策略来源，便于事后归因",
                )
            )

        if not findings:
            overall = "OK"
        elif any(f.severity == "high" for f in findings):
            overall = "FAIL"
        else:
            overall = "WARN"

        return AuditResult(
            trade_id=trade.trade_id,
            findings=findings,
            overall=overall,
            sample_sufficient=True,
        )

    def audit_many(self, trades: list[TradeRecord]) -> list[AuditResult]:
        return [self.audit(t) for t in trades]


class JournalAuditEngine:
    """Batch audit plus aggregate error-type summary."""

    def __init__(self, auditor: Optional[TradeAuditor] = None) -> None:
        self.auditor = auditor or TradeAuditor()

    def audit_journal(self, trades: list[TradeRecord]) -> dict[str, Any]:
        if not trades:
            return {
                "sample_sufficient": False,
                "results": [],
                "summary": {},
            }
        results = self.auditor.audit_many(trades)
        summary: dict[str, int] = {}
        for result in results:
            for finding in result.findings:
                summary[finding.error_type] = summary.get(finding.error_type, 0) + 1
        return {
            "sample_sufficient": True,
            "results": [r.to_dict() for r in results],
            "summary": summary,
        }
