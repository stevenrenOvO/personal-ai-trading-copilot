"""Simple moving-average crossover strategy."""

from __future__ import annotations

import pandas as pd

from backend.strategies.base import Strategy, prepare_daily
from backend.strategies.indicators import sma


class MaCrossStrategy(Strategy):
    """Generate BUY and SELL signals when fast SMA crosses slow SMA."""

    name = "ma_cross"
    strategy_id = "ma_cross"
    version = "1.0"
    status = "experimental"

    def __init__(self, fast_window: int = 5, slow_window: int = 20) -> None:
        if fast_window <= 0:
            raise ValueError("fast_window must be positive")
        if slow_window <= 0:
            raise ValueError("slow_window must be positive")
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window")

        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, daily: pd.DataFrame) -> list[TradeSignal]:
        data = prepare_daily(daily)
        signals: list[TradeSignal] = []

        for ts_code, group in data.groupby("ts_code", sort=False):
            bars = group.sort_values("trade_date").reset_index(drop=True)
            fast = sma(bars["close"], self.fast_window)
            slow = sma(bars["close"], self.slow_window)

            prev_fast = fast.shift(1)
            prev_slow = slow.shift(1)

            valid = (
                fast.notna()
                & slow.notna()
                & prev_fast.notna()
                & prev_slow.notna()
            )

            cross_up = valid & (prev_fast <= prev_slow) & (fast > slow)
            cross_down = valid & (prev_fast >= prev_slow) & (fast < slow)

            for idx in bars.index[cross_up]:
                row = bars.loc[idx]
                signals.append(
                    self.make_signal(
                        ts_code=str(ts_code),
                        trade_date=str(row["trade_date"]),
                        action="BUY",
                        price=float(row["close"]),
                        reason=(
                            f"fast SMA {self.fast_window} crossed above "
                            f"slow SMA {self.slow_window}"
                        ),
                    )
                )

            for idx in bars.index[cross_down]:
                row = bars.loc[idx]
                signals.append(
                    self.make_signal(
                        ts_code=str(ts_code),
                        trade_date=str(row["trade_date"]),
                        action="SELL",
                        price=float(row["close"]),
                        reason=(
                            f"fast SMA {self.fast_window} crossed below "
                            f"slow SMA {self.slow_window}"
                        ),
                    )
                )

        return signals
