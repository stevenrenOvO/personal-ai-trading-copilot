"""Formal V1.0 sector-relative breakout strategy."""

from __future__ import annotations

import pandas as pd

from backend.strategies.base import Strategy, prepare_daily


class StrongSectorBreakoutStrategy(Strategy):
    """Generate BUY signals when a stock breaks its prior high on volume.

    This is the primary V1.0 strategy. The optional sector map and ranking
    allow it to prefer stocks in strong sectors without hard-coding a
    classic moving-average crossover.
    """

    name = "strong_sector_breakout"
    strategy_id = "strong_sector_breakout"
    version = "1.0"
    status = "production"

    def __init__(
        self,
        lookback: int = 20,
        min_volume_ratio: float = 1.2,
        max_sector_rank: int | None = 3,
        max_signals: int = 5,
        sector_by_code: dict[str, str] | None = None,
        sector_rank: dict[str, int] | None = None,
    ) -> None:
        if lookback < 2:
            raise ValueError("lookback must be at least 2")
        if min_volume_ratio <= 0:
            raise ValueError("min_volume_ratio must be positive")
        if max_signals <= 0:
            raise ValueError("max_signals must be positive")

        self.lookback = lookback
        self.min_volume_ratio = min_volume_ratio
        self.max_sector_rank = max_sector_rank
        self.max_signals = max_signals
        self._sector_by_code = dict(sector_by_code or {})
        self._sector_rank = dict(sector_rank or {})

    def set_sector_context(
        self,
        sector_by_code: dict[str, str],
        sector_rank: dict[str, int],
    ) -> None:
        """Attach sector context computed by ``SectorEngine``."""

        self._sector_by_code = dict(sector_by_code)
        self._sector_rank = dict(sector_rank)

    def generate_signals(self, daily: pd.DataFrame) -> list[TradeSignal]:
        data = prepare_daily(daily)
        if data.empty:
            return []

        if "high" not in data.columns or "vol" not in data.columns:
            return []

        # Candidates are collected for every trade_date so the strategy works
        # both for a single-day scan and for a full historical backtest. Each
        # date only uses bars strictly before it, so there is no look-ahead.
        #
        # This is vectorised on purpose: a full-market scan is ~5 200 codes and
        # the previous per-row Python loop took ~13s per request.
        data = data.copy()
        for column in ("high", "close", "vol"):
            data[column] = pd.to_numeric(data[column], errors="coerce")

        prior_high = self._prior_window(data, "high", "max")
        avg_volume = self._prior_window(data, "vol", "mean")

        usable = (
            prior_high.notna()
            & avg_volume.notna()
            & (prior_high > 0)
            & (avg_volume > 0)
            & data["close"].notna()
            & data["vol"].notna()
        )
        if not usable.any():
            return []

        candidates = data.loc[usable, ["ts_code", "trade_date", "close"]].copy()
        candidates["breakout_ratio"] = (
            data.loc[usable, "close"] / prior_high.loc[usable]
        )
        candidates["volume_ratio"] = data.loc[usable, "vol"] / avg_volume.loc[usable]
        candidates = candidates[
            (candidates["breakout_ratio"] > 1.0)
            & (candidates["volume_ratio"] >= self.min_volume_ratio)
        ]
        if candidates.empty:
            return []

        ranks = candidates["ts_code"].map(
            lambda code: self._sector_rank.get(self._sector_by_code.get(code))
        )
        if self.max_sector_rank is not None:
            keep = ranks.isna() | (ranks <= self.max_sector_rank)
            candidates = candidates[keep]
            ranks = ranks[keep]
            if candidates.empty:
                return []
        candidates["_sector_rank"] = ranks.fillna(999).astype(int)

        signals: list[TradeSignal] = []
        for trade_date, group in candidates.groupby("trade_date", sort=True):
            ranked = group.sort_values(
                ["_sector_rank", "breakout_ratio", "volume_ratio"],
                ascending=[True, False, False],
                kind="mergesort",
            )
            for _, row in ranked.head(self.max_signals).iterrows():
                breakout_ratio = float(row["breakout_ratio"])
                volume_ratio = float(row["volume_ratio"])
                strength = min(
                    1.0,
                    (breakout_ratio - 1.0) + max(0.0, volume_ratio - 1.0) * 0.1,
                )
                signals.append(
                    self.make_signal(
                        ts_code=str(row["ts_code"]),
                        trade_date=str(trade_date),
                        action="BUY",
                        price=float(row["close"]),
                        strength=round(strength, 4),
                        reason=f"突破前高 {breakout_ratio:.3f}，量比 {volume_ratio:.2f}",
                    )
                )

        return signals

    def _prior_window(self, data: pd.DataFrame, column: str, how: str) -> pd.Series:
        """Rolling value over the ``lookback`` bars BEFORE each row, per code."""
        rolled = (
            data.groupby("ts_code", sort=False)[column]
            .rolling(self.lookback, min_periods=self.lookback)
            .agg(how)
        )
        # ``rolling`` returns a (ts_code, row-label) MultiIndex; put the values
        # back on the original row order, then shift one bar inside each code so
        # the current bar never contributes to its own breakout base.
        values = rolled.reset_index(level=0, drop=True).reindex(data.index)
        return values.groupby(data["ts_code"], sort=False).shift(1)
