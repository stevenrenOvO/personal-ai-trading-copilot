"""Performance metrics for backtest results."""

from __future__ import annotations

import math
import pandas as pd


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
    if equity_curve.empty:
        metrics = {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "volatility": 0.0,
            "win_rate": None,
            "positive_day_ratio": None,
            "total_days": len(equity_curve),
            "trade_count": 0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "payoff_ratio": None,
            "profit_factor": None,
            "average_holding_period": 0.0,
            "max_consecutive_losses": 0,
        }
    else:
        df = equity_curve.sort_values("date").copy()
        df["total"] = df["total"].astype(float)
        if not df["total"].map(math.isfinite).all() or (df["total"] <= 0).any():
            raise ValueError("equity must contain finite positive account values")
        if df.date.duplicated().any():
            raise ValueError("equity dates must be unique")
        if not math.isfinite(risk_free_rate) or risk_free_rate <= -1:
            raise ValueError("risk_free_rate must be finite and greater than -1")
        baseline = equity_curve.attrs.get("initial_equity")
        initial = float(baseline) if baseline is not None else df["total"].iloc[0]
        if not math.isfinite(initial) or initial <= 0:
            raise ValueError("initial equity must be finite and positive")
        total_return = float(df["total"].iloc[-1] / initial - 1)
        n_periods = len(df) if baseline is not None else len(df) - 1
        annualized_return = (1 + total_return) ** (252 / n_periods) - 1 if n_periods else 0.0
        peak = df["total"].cummax().clip(lower=initial)
        max_drawdown = float(((df["total"] - peak) / peak).min())
        daily_returns = df["total"].pct_change()
        if baseline is not None:
            daily_returns.iloc[0] = df["total"].iloc[0] / initial - 1
        daily_returns = daily_returns.dropna()
        daily_std = float(daily_returns.std()) if len(daily_returns) >= 2 else 0.0
        volatility = daily_std * math.sqrt(252)
        daily_risk_free = (1 + risk_free_rate) ** (1 / 252) - 1
        sharpe_ratio = float((daily_returns.mean() - daily_risk_free) / daily_std * math.sqrt(252)) if daily_std > 0 else 0.0
        positive_day_ratio = float((daily_returns > 0).mean()) if len(daily_returns) else None
        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0,
            "volatility": volatility,
            "win_rate": None,
            "positive_day_ratio": positive_day_ratio,
            "total_days": len(df),
            "trade_count": 0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "payoff_ratio": None,
            "profit_factor": None,
            "average_holding_period": 0.0,
            "max_consecutive_losses": 0,
        }

    if trade_log:
        closed = [t for t in trade_log if getattr(t, "exit_price", None) is not None
                  and t.pnl is not None and math.isfinite(float(t.pnl))]
        pnls = [float(t.pnl) for t in closed]
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
                    "win_rate": len(wins) / len(pnls),
                    "average_win": avg_win,
                    "average_loss": avg_loss,
                    "payoff_ratio": avg_win / abs(avg_loss) if avg_loss else None,
                    "profit_factor": gross_profit / gross_loss if gross_loss else None,
                    "average_holding_period": (
                        sum(holdings) / len(holdings) if holdings else 0.0
                    ),
                    "max_consecutive_losses": max_consecutive,
                }
            )

    return metrics
