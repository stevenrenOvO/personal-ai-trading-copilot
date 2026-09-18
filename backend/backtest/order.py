"""Order and position data structures."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class Order:
    ts_code: str
    trade_date: str
    side: OrderSide
    price: float
    quantity: float  # number of shares / units
    commission: float = 0.0

    @property
    def cost(self) -> float:
        return self.quantity * self.price + self.commission


@dataclass
class Position:
    ts_code: str
    quantity: float
    cost_basis: float  # total cost including commissions

    def to_dict(self) -> dict[str, object]:
        return {
            "ts_code": self.ts_code,
            "quantity": self.quantity,
            "cost_basis": self.cost_basis,
            "avg_price": self.avg_price,
        }

    @property
    def avg_price(self) -> float:
        if self.quantity == 0:
            return 0.0
        return self.cost_basis / self.quantity

    def apply_order(self, order: Order) -> None:
        if order.side == OrderSide.BUY:
            self.quantity += order.quantity
            self.cost_basis += order.cost
        else:  # SELL
            # For simplicity, we close partial positions proportionally.
            # This assumes FIFO or average cost; we just reduce quantity and adjust cost basis.
            if self.quantity == 0:
                raise ValueError("Cannot sell from empty position")
            if order.quantity > self.quantity:
                raise ValueError("Insufficient quantity to sell")
            sell_proportion = order.quantity / self.quantity
            realized_cost = self.cost_basis * sell_proportion
            self.quantity -= order.quantity
            self.cost_basis -= realized_cost


@dataclass(frozen=True)
class Trade:
    """A closed or open trade record for backtest output."""

    ts_code: str
    entry_date: str
    exit_date: str | None
    side: str
    entry_price: float
    exit_price: float | None
    quantity: float
    pnl: float | None = None
    return_pct: float | None = None
    holding_days: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "ts_code": self.ts_code,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "pnl": self.pnl,
            "return_pct": self.return_pct,
            "holding_days": self.holding_days,
        }
