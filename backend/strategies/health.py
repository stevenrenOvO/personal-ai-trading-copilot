"""Strategy health tracking.

Tracks daily signal counts, win rate, cumulative return, max drawdown,
consecutive failures, recent trade windows and anomalies. Production strategies
must never be auto-optimized; enabling/disabling is explicit and persisted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class StrategyHealth:
    strategy_id: str
    version: str
    status: str = "experimental"
    enabled: bool = True
    daily_signal_count: int = 0
    win_rate: float = 0.0
    cumulative_return: float = 0.0
    max_drawdown: float = 0.0
    consecutive_failures: int = 0
    recent_20_trades: list[float] = field(default_factory=list)
    recent_50_trades: list[float] = field(default_factory=list)
    anomalies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "version": self.version,
            "status": self.status,
            "enabled": self.enabled,
            "daily_signal_count": self.daily_signal_count,
            "win_rate": self.win_rate,
            "cumulative_return": self.cumulative_return,
            "max_drawdown": self.max_drawdown,
            "consecutive_failures": self.consecutive_failures,
            "recent_20_trades": self.recent_20_trades,
            "recent_50_trades": self.recent_50_trades,
            "anomalies": self.anomalies,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyHealth":
        return cls(
            strategy_id=data.get("strategy_id", "unknown"),
            version=data.get("version", "1.0"),
            status=data.get("status", "experimental"),
            enabled=data.get("enabled", True),
            daily_signal_count=data.get("daily_signal_count", 0),
            win_rate=data.get("win_rate", 0.0),
            cumulative_return=data.get("cumulative_return", 0.0),
            max_drawdown=data.get("max_drawdown", 0.0),
            consecutive_failures=data.get("consecutive_failures", 0),
            recent_20_trades=list(data.get("recent_20_trades", [])),
            recent_50_trades=list(data.get("recent_50_trades", [])),
            anomalies=list(data.get("anomalies", [])),
        )


class StrategyHealthStore:
    """JSONL-backed health records, keyed by strategy_id + version."""

    def __init__(self, root: Path) -> None:
        self.path = Path(root) / "strategies" / "health.jsonl"

    def _load(self) -> dict[tuple[str, str], StrategyHealth]:
        if not self.path.exists():
            return {}
        records: dict[tuple[str, str], StrategyHealth] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            import json

            data = json.loads(line)
            health = StrategyHealth.from_dict(data)
            records[(health.strategy_id, health.version)] = health
        return records

    def save(self, health: StrategyHealth) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        records = self._load()
        records[(health.strategy_id, health.version)] = health
        import json

        with self.path.open("w", encoding="utf-8") as handle:
            for item in records.values():
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    def get(self, strategy_id: str, version: str) -> Optional[StrategyHealth]:
        return self._load().get((strategy_id, version))

    def list(self) -> list[StrategyHealth]:
        return list(self._load().values())


class StrategyRegistry:
    """Registry of known strategies and their persisted health."""

    def __init__(self, root: Path) -> None:
        self.store = StrategyHealthStore(root)

    def register(self, strategy) -> StrategyHealth:
        health = self.store.get(strategy.strategy_id, strategy.version)
        if health is None:
            health = StrategyHealth(
                strategy_id=strategy.strategy_id,
                version=strategy.version,
                status=strategy.status,
                enabled=strategy.enabled,
            )
            self.store.save(health)
        else:
            health.status = strategy.status
            strategy.enabled = health.enabled
        return health

    def set_enabled(self, strategy_id: str, version: str, enabled: bool) -> StrategyHealth:
        health = self.store.get(strategy_id, version)
        if health is None:
            health = StrategyHealth(strategy_id=strategy_id, version=version)
        health.enabled = enabled
        self.store.save(health)
        return health

    def update_performance(
        self,
        strategy_id: str,
        version: str,
        *,
        signals: int = 0,
        trades: list[float] | None = None,
    ) -> StrategyHealth:
        health = self.store.get(strategy_id, version)
        if health is None:
            health = StrategyHealth(strategy_id=strategy_id, version=version)

        health.daily_signal_count = signals
        pnls = trades or []
        if pnls:
            wins = sum(1 for p in pnls if p > 0)
            health.win_rate = wins / len(pnls)
            health.cumulative_return = sum(pnls)
            health.max_drawdown = self._max_drawdown(pnls)
            health.consecutive_failures = self._consecutive_losses(pnls)
            health.recent_20_trades = pnls[-20:]
            health.recent_50_trades = pnls[-50:]
            health.anomalies = self._anomalies(pnls)
        self.store.save(health)
        return health

    def _max_drawdown(self, pnls: list[float]) -> float:
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for p in pnls:
            equity += p
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1)
        return max_dd

    def _consecutive_losses(self, pnls: list[float]) -> int:
        current = 0
        best = 0
        for p in pnls:
            if p < 0:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    def _anomalies(self, pnls: list[float]) -> list[str]:
        anomalies: list[str] = []
        if len(pnls) >= 5 and all(p < 0 for p in pnls[-5:]):
            anomalies.append("连续5笔亏损")
        if self._consecutive_losses(pnls) >= 4:
            anomalies.append("连续亏损次数过高")
        return anomalies
