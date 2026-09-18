"""Data models for the decision journal and audit trail."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import uuid


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class DecisionRecord:
    decision_id: str
    created_at: str
    trade_date: str
    ts_code: str
    action: str
    signal_strength: float
    risk_allowed: bool
    risk_violations: list[str]
    decision: str
    reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        trade_date: str,
        ts_code: str,
        action: str,
        signal_strength: float,
        risk_allowed: bool,
        risk_violations: list[str],
        decision: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> "DecisionRecord":
        return cls(
            decision_id=new_id("decision"),
            created_at=utc_now_iso(),
            trade_date=trade_date,
            ts_code=ts_code,
            action=action,
            signal_strength=float(signal_strength),
            risk_allowed=bool(risk_allowed),
            risk_violations=list(risk_violations),
            decision=decision,
            reason=reason,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionRecord":
        return cls(**data)


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    created_at: str
    event_type: str
    actor: str
    target: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        event_type: str,
        actor: str,
        target: str,
        payload: dict[str, Any] | None = None,
    ) -> "AuditEvent":
        return cls(
            event_id=new_id("event"),
            created_at=utc_now_iso(),
            event_type=event_type,
            actor=actor,
            target=target,
            payload=dict(payload or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditEvent":
        return cls(**data)
