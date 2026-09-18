"""Data models for user trading profile."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ProfileConfig:
    """User-configurable risk and style preferences."""

    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive
    max_position_weight: float = 0.20
    max_drawdown_threshold: float = 0.15
    stop_loss_pct: float = 0.08
    take_profit_pct: float = 0.20
    preferred_sectors: list[str] = field(default_factory=list)
    min_holding_days: int = 1
    max_holding_days: int = 30

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_tolerance": self.risk_tolerance,
            "max_position_weight": self.max_position_weight,
            "max_drawdown_threshold": self.max_drawdown_threshold,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "preferred_sectors": self.preferred_sectors[:],
            "min_holding_days": self.min_holding_days,
            "max_holding_days": self.max_holding_days,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileConfig":
        return cls(**data)


@dataclass
class ProfileStats:
    """Computed statistics from trading history."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_return: float = 0.0
    avg_holding_days: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    max_consecutive_losses: int = 0
    best_strategy: str = ""
    worst_strategy: str = ""
    best_market_phase: str = ""
    worst_market_phase: str = ""
    chase_frequency: float = 0.0
    early_exit_frequency: float = 0.0
    sample_sufficient: bool = False

    def update_from_trades(self, trades: list[dict[str, Any]]) -> None:
        """Recalculate stats from a list of trade records."""
        if not trades:
            self.sample_sufficient = False
            return
        self.total_trades = len(trades)
        self.winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
        self.losing_trades = sum(1 for t in trades if t.get("pnl", 0) < 0)
        self.win_rate = self.winning_trades / self.total_trades if self.total_trades else 0.0
        returns = [t.get("return_pct", 0.0) for t in trades]
        self.avg_return = sum(returns) / len(returns) if returns else 0.0
        holdings = [t.get("holding_days", 0) for t in trades]
        self.avg_holding_days = sum(holdings) / len(holdings) if holdings else 0.0
        pnls = [t.get("pnl", 0.0) for t in trades]
        self.total_pnl = sum(pnls)
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        self.profit_factor = gross_profit / gross_loss if gross_loss else 0.0
        self.max_consecutive_losses = self._max_consecutive_losses(pnls)
        self.max_drawdown = self._max_drawdown(pnls)
        self.sample_sufficient = len(trades) >= 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "avg_return": self.avg_return,
            "avg_holding_days": self.avg_holding_days,
            "max_drawdown": self.max_drawdown,
            "total_pnl": self.total_pnl,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "max_consecutive_losses": self.max_consecutive_losses,
            "best_strategy": self.best_strategy,
            "worst_strategy": self.worst_strategy,
            "best_market_phase": self.best_market_phase,
            "worst_market_phase": self.worst_market_phase,
            "chase_frequency": self.chase_frequency,
            "early_exit_frequency": self.early_exit_frequency,
            "sample_sufficient": self.sample_sufficient,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProfileStats":
        known = {
            "total_trades",
            "winning_trades",
            "losing_trades",
            "win_rate",
            "avg_return",
            "avg_holding_days",
            "max_drawdown",
            "total_pnl",
            "sharpe_ratio",
            "profit_factor",
            "max_consecutive_losses",
            "best_strategy",
            "worst_strategy",
            "best_market_phase",
            "worst_market_phase",
            "chase_frequency",
            "early_exit_frequency",
            "sample_sufficient",
        }
        return cls(**{k: v for k, v in data.items() if k in known})

    @staticmethod
    def _max_consecutive_losses(pnls: list[float]) -> int:
        current = 0
        best = 0
        for p in pnls:
            if p < 0:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    @staticmethod
    def _max_drawdown(pnls: list[float]) -> float:
        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        for p in pnls:
            equity += p
            peak = max(peak, equity)
            max_dd = min(max_dd, equity / peak - 1)
        return max_dd


@dataclass
class UserProfile:
    """Complete user profile combining config and computed stats."""

    user_id: str = "default"
    config: ProfileConfig = field(default_factory=ProfileConfig)
    stats: ProfileStats = field(default_factory=ProfileStats)
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "config": self.config.to_dict(),
            "stats": self.stats.to_dict(),
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserProfile":
        config = ProfileConfig.from_dict(data.get("config", {}))
        stats = ProfileStats.from_dict(data.get("stats", {}))
        return cls(
            user_id=data.get("user_id", "default"),
            config=config,
            stats=stats,
            last_updated=data.get("last_updated", ""),
        )
