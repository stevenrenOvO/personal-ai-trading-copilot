"""Trade records and trade journal for real paper-trading history.

These are richer than ``DecisionRecord``: they capture positions before/after,
strategy identity, market/emotion/sector context, system recommendation and the
user's own reason. They are the source of truth for profile and audit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
import uuid


def trade_id() -> str:
    return f"trade_{uuid.uuid4().hex}"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TradeRecord:
    trade_id: str
    timestamp: str
    symbol: str
    ts_code: str
    action: str  # BUY | SELL | HOLD | WAIT | MISSED_SIGNAL
    price: Optional[float]
    quantity: float
    position_before: float
    position_after: float
    strategy_id: str
    strategy_version: str
    signal_id: str
    market_state: str
    emotion_state: str
    sector_state: str
    stock_state: str
    system_recommendation: str
    user_reason: str
    execution_reason: str
    result: Optional[dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        symbol: str,
        ts_code: str,
        action: str,
        price: Optional[float],
        quantity: float,
        position_before: float,
        position_after: float,
        strategy_id: str = "",
        strategy_version: str = "",
        signal_id: str = "",
        market_state: str = "",
        emotion_state: str = "",
        sector_state: str = "",
        stock_state: str = "",
        system_recommendation: str = "",
        user_reason: str = "",
        execution_reason: str = "",
        result: Optional[dict[str, Any]] = None,
    ) -> "TradeRecord":
        return cls(
            trade_id=trade_id(),
            timestamp=now_iso(),
            symbol=symbol,
            ts_code=ts_code,
            action=action,
            price=price,
            quantity=quantity,
            position_before=position_before,
            position_after=position_after,
            strategy_id=strategy_id,
            strategy_version=strategy_version,
            signal_id=signal_id,
            market_state=market_state,
            emotion_state=emotion_state,
            sector_state=sector_state,
            stock_state=stock_state,
            system_recommendation=system_recommendation,
            user_reason=user_reason,
            execution_reason=execution_reason,
            result=result or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TradeRecord":
        return cls(**data)


class TradeJournal:
    """JSONL persistence for executed trades."""

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "journal" / "trades.jsonl"

    def record(self, trade: TradeRecord) -> TradeRecord:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trade.to_dict(), ensure_ascii=False) + "\n")
        return trade

    def list(self) -> list[TradeRecord]:
        if not self.path.exists():
            return []
        records: list[TradeRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(TradeRecord.from_dict(json.loads(line)))
        return records
