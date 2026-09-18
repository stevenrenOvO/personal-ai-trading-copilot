"""Engine for computing and updating user profile statistics."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

from backend.journal.store import DecisionJournal
from backend.journal.trades import TradeJournal
from backend.profile.models import ProfileConfig, ProfileStats, UserProfile
from backend.profile.store import ProfileStore


class ProfileEngine:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.journal = DecisionJournal(root)
        self.store = ProfileStore(root)

    def load_or_create(self) -> UserProfile:
        existing = self.store.load()
        if existing:
            return existing
        return UserProfile()

    def refresh_stats(self, profile: UserProfile) -> UserProfile:
        """Recalculate statistics from the decision journal."""
        decisions = self.journal.list()
        # Extract trades from decisions (only approved BUY and SELL pairs, but for simplicity we treat each decision as a potential trade)
        # For now, we compute simple metrics from decisions.
        trades = []
        for rec in decisions:
            if rec.decision == "APPROVED":
                # Approximate trade record
                trades.append({
                    "pnl": 0.0,  # Not enough data to compute PnL from journal alone
                    "return_pct": 0.0,
                    "holding_days": 0,
                })
        stats = ProfileStats()
        stats.update_from_trades(trades)
        # For demonstration, we also incorporate some fake values if no trades
        if stats.total_trades == 0:
            stats.win_rate = 0.0
        profile.stats = stats
        profile.last_updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.store.save(profile)
        return profile

    def refresh_stats_from_trades(self, profile: UserProfile) -> UserProfile:
        """Recalculate statistics from real executed trades."""
        trades = TradeJournal(self.root).list()
        trade_dicts = []
        for trade in trades:
            result = trade.result or {}
            pnl = result.get("pnl", 0.0)
            trade_dicts.append(
                {
                    "pnl": pnl,
                    "return_pct": result.get("return_pct", 0.0),
                    "holding_days": result.get("holding_days", 0),
                    "strategy_id": trade.strategy_id,
                    "market_state": trade.market_state,
                }
            )

        stats = ProfileStats()
        stats.update_from_trades(trade_dicts)
        if stats.total_trades == 0:
            stats.win_rate = 0.0
        self._attach_breakdowns(stats, trade_dicts)
        profile.stats = stats
        profile.last_updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.store.save(profile)
        return profile

    def _attach_breakdowns(self, stats: ProfileStats, trades: list[dict]) -> None:
        """Populate best/worst strategy and market-phase fields."""
        strategy_pnl: dict[str, float] = {}
        phase_pnl: dict[str, float] = {}
        for t in trades:
            sid = t.get("strategy_id") or "unknown"
            strategy_pnl[sid] = strategy_pnl.get(sid, 0.0) + t.get("pnl", 0.0)
            phase = t.get("market_state") or "unknown"
            phase_pnl[phase] = phase_pnl.get(phase, 0.0) + t.get("pnl", 0.0)
        if strategy_pnl:
            stats.best_strategy = max(strategy_pnl, key=strategy_pnl.get)
            stats.worst_strategy = min(strategy_pnl, key=strategy_pnl.get)
        if phase_pnl:
            stats.best_market_phase = max(phase_pnl, key=phase_pnl.get)
            stats.worst_market_phase = min(phase_pnl, key=phase_pnl.get)

    def update_config(self, profile: UserProfile, config: ProfileConfig) -> UserProfile:
        profile.config = config
        profile.last_updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.store.save(profile)
        return profile
