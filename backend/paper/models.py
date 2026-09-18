"""Data models for paper trading.

The models are kept small and serializable so the paper layer can be used from
the decision engine, FastAPI, and dashboard without depending on a live broker.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PaperPosition:
    """Synthetic position with average-cost accounting."""

    ts_code: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    buy_date: str = ""

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def apply_buy(
        self,
        *,
        price: float,
        quantity: float,
        commission: float = 0.0,
        trade_date: str = "",
    ) -> None:
        """Add a buy fill and update the average price."""

        old_cost = self.quantity * self.avg_price
        new_cost = quantity * price + commission
        self.quantity += quantity
        if quantity > 0:
            self.buy_date = trade_date
        if self.quantity > 0:
            self.avg_price = (old_cost + new_cost) / self.quantity

    def apply_sell(
        self,
        *,
        price: float,
        quantity: float,
        commission: float = 0.0,
    ) -> float:
        """Reduce position and return realized PnL for the fill."""

        if quantity <= 0:
            raise ValueError("sell quantity must be positive")
        if quantity > self.quantity:
            raise ValueError("insufficient paper position to sell")

        realized = quantity * (price - self.avg_price) - commission
        self.quantity -= quantity
        self.realized_pnl += realized
        if self.quantity <= 0:
            self.avg_price = 0.0
        return realized

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["buy_date"] = self.buy_date
        return data


@dataclass(frozen=True)
class PaperOrder:
    """Auditable record of a simulated order."""

    ts_code: str
    trade_date: str
    action: str
    price: float
    quantity: float
    commission: float = 0.0
    status: str = "FILLED"
    reason: str = ""

    @property
    def gross_value(self) -> float:
        return self.price * self.quantity

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PaperSnapshot:
    """Point-in-time account summary."""

    cash: float
    positions: dict[str, PaperPosition]
    equity: float
    realized_pnl: float
    orders: tuple[PaperOrder, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "positions": {
                code: position.to_dict() for code, position in self.positions.items()
            },
            "equity": self.equity,
            "realized_pnl": self.realized_pnl,
            "orders": [order.to_dict() for order in self.orders],
        }
