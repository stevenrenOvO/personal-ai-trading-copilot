"""Performance metrics for backtest results."""

from __future__ import annotations

import math
import pandas as pd
from typing import Optional


def calculate_metrics(
    equity_curve: pd.DataFrame,
    risk_free_rate: float = 0.02,
    trade_log: list | None = None,
) -> dict:
    """Calculate common performance metrics from equity curve.

    When a ``trade_log`` of closed trades is supplied, additional trade-level
    statistics (win rate, payoff, profit factor, Calmar, consecutive losses)
    are computed from real fills rather than from the daily equity curve alone.
    """
    if equity_curve.empty or len(equity_curve) < 2:
        base = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "volatility": 0.0,
            "win_rate": None,
            "total_days": len(equity_curve),
            "trade_count": 0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "payoff_ratio": 0.0,
            "profit_factor": 0.0,
            "average_holding_period": 0.0,
            "max_consecutive_losses": 0,
        }
        return base

    # Ensure sorted by date
    df = equity_curve.sort_values("date").copy()
    df["total"] = df["total"].astype(float)

    # Total return
    initial = df["total"].iloc[0]
    final = df["total"].iloc[-1]
    total_return = (final / initial) - 1 if initial != 0 else 0.0

    # Annualized return (assuming daily data, ~252 trading days per year)
    n_days = len(df)
    annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0.0

    # Drawdown
    df["peak"] = df["total"].cummax()
    df["drawdown"] = (df["total"] - df["peak"]) / df["peak"]
    max_drawdown = df["drawdown"].min() if not df["drawdown"].empty else 0.0

    # Volatility (daily returns)
    df["daily_return"] = df["total"].pct_change()
    volatility = df["daily_return"].std() * math.sqrt(252) if len(df["daily_return"]) > 1 else 0.0

    # Sharpe ratio (annualized)
    excess_return = annualized_return - risk_free_rate
    sharpe_ratio = excess_return / volatility if volatility != 0 else 0.0

    # Win rate (daily positive returns)
    win_days = (df["daily_return"] > 0).sum()
    total_days = len(df["daily_return"]) - 1  # exclude first day
    win_rate = win_days / total_days if total_days > 0 else 0.0

    metrics = {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe_ratio,
        "calmar_ratio": annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0,
        "volatility": volatility,
        "win_rate": win_rate,
        "total_days": len(df),
        "trade_count": 0,
        "average_win": 0.0,
        "average_loss": 0.0,
        "payoff_ratio": 0.0,
        "profit_factor": 0.0,
        "average_holding_period": 0.0,
        "max_consecutive_losses": 0,
    }

    if trade_log:
        closed = [t for t in trade_log if getattr(t, "exit_price", None) is not None]
        pnls = [float(t.pnl) for t in closed if t.pnl is not None]
        if pnls:
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            avg_win = sum(wins) / len(wins) if wins else 0.0
            avg_loss = sum(losses) / len(losses) if losses else 0.0
            gross_profit = sum(wins)
            gross_loss = abs(sum(losses))

            max_consecutive = 0
            current = 0
            for p in pnls:
                if p < 0:
                    current += 1
                    max_consecutive = max(max_consecutive, current)
                else:
                    current = 0

            holdings = [
                float(t.holding_days) for t in closed if t.holding_days is not None
            ]
            metrics.update(
                {
                    "trade_count": len(closed),
                    "average_win": avg_win,
                    "average_loss": avg_loss,
                    "payoff_ratio": avg_win / abs(avg_loss) if avg_loss else 0.0,
                    "profit_factor": gross_profit / gross_loss if gross_loss else 0.0,
                    "average_holding_period": (
                        sum(holdings) / len(holdings) if holdings else 0.0
                    ),
                    "max_consecutive_losses": max_consecutive,
                }
            )

    return metrics
