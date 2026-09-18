"""Backtest engine: simulate trading based on strategy signals."""

from __future__ import annotations

import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass

from backend.data.providers.base import DataProvider
from backend.strategies.base import Strategy, TradeSignal
from backend.backtest.order import Order, OrderSide, Position


@dataclass
class BacktestResult:
    """Result of a backtest run."""
    equity_curve: pd.DataFrame
    positions: Dict[str, Position]
    orders: List[Order]
    final_cash: float
    final_total_value: float
    trade_log: list = None
    drawdown_curve: pd.DataFrame = None
    lookahead_issues: list = None

    def __post_init__(self):
        if self.trade_log is None:
            self.trade_log = []
        if self.lookahead_issues is None:
            self.lookahead_issues = []


class BacktestEngine:
    def __init__(
        self,
        provider: DataProvider,
        strategy: Strategy,
        initial_cash: float = 1_000_000.0,
        commission_rate: float = 0.0003,
        slippage: float = 0.0,
    ):
        self.provider = provider
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage = slippage

    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ts_codes: Optional[List[str]] = None,
    ) -> BacktestResult:
        daily = self.provider.daily(ts_codes=ts_codes, start_date=start_date, end_date=end_date)
        if daily.empty:
            raise ValueError("No daily data available")

        signals = self.strategy.generate_signals(daily)
        if not signals:
            # No signals, return initial cash
            return BacktestResult(
                equity_curve=pd.DataFrame({"date": [start_date or "19700101"], "total": [self.initial_cash]}),
                positions={},
                orders=[],
                final_cash=self.initial_cash,
                final_total_value=self.initial_cash,
            )

        # Sort dates
        all_dates = sorted(daily["trade_date"].unique())
        signal_map: Dict[str, List[TradeSignal]] = {}
        for sig in signals:
            signal_map.setdefault(sig.trade_date, []).append(sig)

        cash = self.initial_cash
        positions: Dict[str, Position] = {}
        orders: List[Order] = []
        equity_history: List[dict] = []

        # For quick price lookup, create a dict keyed by (date, ts_code)
        price_lookup = daily.set_index(["trade_date", "ts_code"])["close"].to_dict()

        for date in all_dates:
            # Mark to market
            total_value = cash
            for ts_code, pos in positions.items():
                if pos.quantity > 0:
                    price = price_lookup.get((date, ts_code))
                    if price is None:
                        continue
                    total_value += pos.quantity * price
            equity_history.append({"date": date, "total": total_value})

            # Process signals for this date
            for sig in signal_map.get(date, []):
                price = price_lookup.get((date, sig.ts_code))
                if price is None:
                    continue  # cannot execute without price

                exec_price = price * (1 + self.slippage if sig.action == "BUY" else 1 - self.slippage)

                if sig.action == "BUY":
                    # Determine max quantity affordable
                    max_q = cash / (exec_price * (1 + self.commission_rate))
                    if max_q <= 0:
                        continue
                    quantity = max_q
                    commission = quantity * exec_price * self.commission_rate
                    cost = quantity * exec_price + commission
                    if cost > cash:
                        # Adjust down
                        quantity = cash / (exec_price * (1 + self.commission_rate))
                        commission = quantity * exec_price * self.commission_rate
                        cost = quantity * exec_price + commission
                    if quantity <= 0:
                        continue

                    order = Order(
                        ts_code=sig.ts_code,
                        trade_date=date,
                        side=OrderSide.BUY,
                        price=exec_price,
                        quantity=quantity,
                        commission=commission,
                    )
                    orders.append(order)
                    cash -= cost
                    if sig.ts_code not in positions:
                        positions[sig.ts_code] = Position(ts_code=sig.ts_code, quantity=0.0, cost_basis=0.0)
                    positions[sig.ts_code].apply_order(order)

                elif sig.action == "SELL":
                    pos = positions.get(sig.ts_code)
                    if pos is None or pos.quantity <= 0:
                        continue
                    quantity = pos.quantity  # sell all
                    commission = quantity * exec_price * self.commission_rate
                    proceeds = quantity * exec_price - commission
                    order = Order(
                        ts_code=sig.ts_code,
                        trade_date=date,
                        side=OrderSide.SELL,
                        price=exec_price,
                        quantity=quantity,
                        commission=commission,
                    )
                    orders.append(order)
                    cash += proceeds
                    # Clear position
                    positions[sig.ts_code] = Position(ts_code=sig.ts_code, quantity=0.0, cost_basis=0.0)

        # Final total value
        final_total = cash
        last_date = all_dates[-1] if all_dates else start_date or "19700101"
        for ts_code, pos in positions.items():
            if pos.quantity > 0:
                price = price_lookup.get((last_date, ts_code))
                if price is not None:
                    final_total += pos.quantity * price

        equity_df = pd.DataFrame(equity_history)
        if equity_df.empty:
            equity_df = pd.DataFrame({"date": [last_date], "total": [self.initial_cash]})

        return BacktestResult(
            equity_curve=equity_df,
            positions=positions,
            orders=orders,
            final_cash=cash,
            final_total_value=final_total,
        )
