"""Offline paper trading engine.

The engine consumes normalized ``TradeSignal`` objects and executes them under
A-share-friendly defaults: 100-share lots, commission, and optional slippage.
It does not mutate the strategy or provider layers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend.paper.models import PaperOrder, PaperPosition, PaperSnapshot
from backend.strategies.base import TradeSignal


@dataclass
class PaperConfig:
    """Configuration shared by the paper broker."""

    initial_cash: float = 1_000_000.0
    lot_size: int = 100
    commission_rate: float = 0.0003
    slippage: float = 0.0


class PaperTradingEngine:
    """A dependency-free order simulator for strategy signals."""

    def __init__(
        self,
        initial_cash: float = 1_000_000.0,
        lot_size: int = 100,
        commission_rate: float = 0.0003,
        slippage: float = 0.0,
        t_plus_one: bool = False,
        journal: object | None = None,
        state_path: Optional[Path] = None,
    ) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if lot_size <= 0:
            raise ValueError("lot_size must be positive")
        if commission_rate < 0:
            raise ValueError("commission_rate cannot be negative")
        if slippage < 0:
            raise ValueError("slippage cannot be negative")

        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.lot_size = int(lot_size)
        self.commission_rate = float(commission_rate)
        self.slippage = float(slippage)
        self.t_plus_one = bool(t_plus_one)
        self.journal = journal
        # Paper positions must survive a restart: losing them made the review
        # page disagree with the decision journal after every service restart.
        self.state_path = Path(state_path) if state_path else None
        self.positions: dict[str, PaperPosition] = {}
        self.orders: list[PaperOrder] = []
        if self.state_path is not None:
            self.load()

    def reset(self) -> None:
        """Restore the account to its initial state."""

        self.cash = self.initial_cash
        self.positions.clear()
        self.orders.clear()
        self._persist()

    # -- persistence ---------------------------------------------------
    def _persist(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "positions": {
                code: {
                    "ts_code": p.ts_code,
                    "quantity": p.quantity,
                    "avg_price": p.avg_price,
                    "realized_pnl": p.realized_pnl,
                    "buy_date": getattr(p, "buy_date", None),
                }
                for code, p in self.positions.items()
            },
            "orders": [
                {
                    "ts_code": o.ts_code,
                    "trade_date": o.trade_date,
                    "action": o.action,
                    "price": o.price,
                    "quantity": o.quantity,
                    "commission": o.commission,
                    "status": o.status,
                    "reason": o.reason,
                }
                for o in self.orders[-500:]
            ],
        }
        try:
            self.state_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
            )
        except OSError:
            pass          # a read-only data dir must not break trading

    def load(self) -> bool:
        """Restore cash / positions / orders. Returns True when state was read."""
        if self.state_path is None or not self.state_path.exists():
            return False
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        self.cash = float(payload.get("cash", self.cash))
        self.initial_cash = float(payload.get("initial_cash", self.initial_cash))
        self.positions = {}
        for code, item in (payload.get("positions") or {}).items():
            position = PaperPosition(ts_code=str(item.get("ts_code") or code))
            position.quantity = float(item.get("quantity") or 0.0)
            position.avg_price = float(item.get("avg_price") or 0.0)
            position.realized_pnl = float(item.get("realized_pnl") or 0.0)
            if item.get("buy_date"):
                try:
                    position.buy_date = item["buy_date"]
                except Exception:  # noqa: BLE001
                    pass
            self.positions[code] = position
        self.orders = []
        for item in payload.get("orders") or []:
            try:
                self.orders.append(
                    PaperOrder(
                        ts_code=str(item.get("ts_code")),
                        trade_date=str(item.get("trade_date") or ""),
                        action=item.get("action", "BUY"),
                        price=float(item.get("price") or 0.0),
                        quantity=float(item.get("quantity") or 0.0),
                        commission=float(item.get("commission") or 0.0),
                        status=item.get("status", "FILLED"),
                        reason=item.get("reason", ""),
                    )
                )
            except Exception:  # noqa: BLE001
                continue
        return True

    def execute_signal(
        self,
        signal: TradeSignal,
        *,
        price: float | None = None,
    ) -> PaperOrder:
        """Execute one signal using the signal price or an explicit override."""

        if price is None:
            if signal.price is None:
                raise ValueError(f"no execution price available for {signal.ts_code}")
            price = float(signal.price)
        if price <= 0:
            raise ValueError("execution price must be positive")

        if signal.action == "BUY":
            order = self._buy(signal, price)
            self._record_if_attached(order, signal)
            return order
        if signal.action == "SELL":
            order = self._sell(signal, price)
            self._record_if_attached(order, signal)
            return order
        raise ValueError(f"unsupported signal action: {signal.action}")

    def execute_signals(
        self,
        signals: list[TradeSignal],
        *,
        price: float | None = None,
    ) -> list[PaperOrder]:
        """Execute a batch of signals sequentially and return every order."""

        return [self.execute_signal(signal, price=price) for signal in signals]

    def total_equity(self, prices: dict[str, float]) -> float:
        """Calculate mark-to-market equity using supplied prices."""

        market_value = 0.0
        for ts_code, position in self.positions.items():
            if position.quantity <= 0:
                continue
            current_price = prices.get(ts_code)
            if current_price is None or current_price <= 0:
                current_price = position.avg_price
            market_value += position.market_value(current_price)
        return self.cash + market_value

    def snapshot(self, prices: dict[str, float] | None = None) -> PaperSnapshot:
        """Return a serializable view of the current account state."""

        price_map = prices or {}
        realized_pnl = sum(
            position.realized_pnl for position in self.positions.values()
        )
        return PaperSnapshot(
            cash=self.cash,
            positions=dict(self.positions),
            equity=self.total_equity(price_map),
            realized_pnl=realized_pnl,
            orders=tuple(self.orders),
        )

    def _execution_price(self, action: str, base_price: float) -> float:
        if action == "BUY":
            return base_price * (1.0 + self.slippage)
        return base_price * (1.0 - self.slippage)

    def _round_down_to_lot(self, quantity: float) -> int:
        return int(quantity // self.lot_size) * self.lot_size

    def _buy(self, signal: TradeSignal, base_price: float) -> PaperOrder:
        price = self._execution_price(signal.action, base_price)
        if price <= 0:
            raise ValueError("computed buy price must be positive")

        per_share_cost = price * (1.0 + self.commission_rate)
        affordable = int(self.cash // per_share_cost)
        quantity = self._round_down_to_lot(affordable)
        # An explicit request (e.g. a manual simulated order) caps the size; it
        # must never be silently ignored and turned into a full-position buy.
        requested = signal.quantity
        if requested is not None and float(requested) > 0:
            quantity = min(quantity, self._round_down_to_lot(float(requested)))

        if quantity <= 0:
            return PaperOrder(
                ts_code=signal.ts_code,
                trade_date=signal.trade_date,
                action=signal.action,
                price=price,
                quantity=0,
                status="REJECTED",
                reason="insufficient cash for one lot",
            )

        gross = quantity * price
        commission = gross * self.commission_rate
        total_cost = gross + commission
        if total_cost > self.cash + 1e-9:
            quantity -= self.lot_size
            if quantity <= 0:
                return PaperOrder(
                    ts_code=signal.ts_code,
                    trade_date=signal.trade_date,
                    action=signal.action,
                    price=price,
                    quantity=0,
                    status="REJECTED",
                    reason="insufficient cash after commission",
                )
            gross = quantity * price
            commission = gross * self.commission_rate
            total_cost = gross + commission

        self.cash -= total_cost
        position = self.positions.setdefault(
            signal.ts_code, PaperPosition(ts_code=signal.ts_code)
        )
        position.apply_buy(
            price=price,
            quantity=float(quantity),
            commission=commission,
            trade_date=signal.trade_date,
        )

        order = PaperOrder(
            ts_code=signal.ts_code,
            trade_date=signal.trade_date,
            action=signal.action,
            price=price,
            quantity=float(quantity),
            commission=commission,
            status="FILLED",
            reason=signal.reason,
        )
        self.orders.append(order)
        self._persist()
        return order

    def _record_if_attached(self, order: PaperOrder, signal: TradeSignal) -> None:
        """Persist executed fills to the decision journal when configured."""
        if self.journal is None or order.status != "FILLED":
            return
        try:
            from backend.journal.trades import TradeRecord

            position = self.positions.get(signal.ts_code)
            qty_after = float(position.quantity) if position else 0.0
            if signal.action == "BUY":
                qty_before = max(0.0, qty_after - order.quantity)
            else:
                qty_before = max(0.0, qty_after + order.quantity)
            record = TradeRecord.create(
                symbol=signal.ts_code.split(".")[0],
                ts_code=signal.ts_code,
                action=signal.action,
                price=order.price,
                quantity=order.quantity,
                position_before=qty_before,
                position_after=qty_after,
                strategy_id=signal.strategy_id,
                strategy_version=signal.strategy_version,
                signal_id=signal.signal_id,
                market_state="",
                emotion_state="",
                sector_state="",
                stock_state="",
                system_recommendation=signal.action,
                user_reason=signal.reason,
                execution_reason="paper fill",
                result={"price": order.price, "quantity": order.quantity},
            )
            self.journal.record(record)
        except Exception:
            pass

    def _sell(self, signal: TradeSignal, base_price: float) -> PaperOrder:
        price = self._execution_price(signal.action, base_price)
        if price <= 0:
            raise ValueError("computed sell price must be positive")

        position = self.positions.get(signal.ts_code)
        if position is None or position.quantity <= 0:
            return PaperOrder(
                ts_code=signal.ts_code,
                trade_date=signal.trade_date,
                action=signal.action,
                price=price,
                quantity=0,
                status="REJECTED",
                reason="no position to sell",
            )

        if self.t_plus_one and position.buy_date == signal.trade_date:
            return PaperOrder(
                ts_code=signal.ts_code,
                trade_date=signal.trade_date,
                action=signal.action,
                price=price,
                quantity=0,
                status="REJECTED",
                reason="T+1: 当日买入不可卖出",
            )

        # 卖出数量：填了数量就按数量卖（支持部分止盈/减仓），未填数量才整仓卖出。
        # 清仓时允许带零股；部分卖出必须是 100 股整数倍，否则拒绝而不是悄悄改数量。
        held = float(position.quantity)
        requested = signal.quantity
        if requested is not None and float(requested) > 0:
            want = float(requested)
            whole_lots = int(want // self.lot_size) * self.lot_size
            if want >= held:
                quantity = held                     # 清仓（允许零股一次性卖出）
            elif whole_lots <= 0:
                return PaperOrder(
                    ts_code=signal.ts_code,
                    trade_date=signal.trade_date,
                    action=signal.action,
                    price=price,
                    quantity=0,
                    status="REJECTED",
                    reason=f"卖出数量不足 1 手（{want:g} 股），部分卖出需为 {self.lot_size} 股整数倍",
                )
            else:
                quantity = min(float(whole_lots), held)
        else:
            quantity = held                          # 未指定数量 = 全部卖出
        gross = quantity * price
        commission = gross * self.commission_rate
        proceeds = gross - commission
        position.apply_sell(
            price=price,
            quantity=quantity,
            commission=commission,
        )
        self.cash += proceeds

        order = PaperOrder(
            ts_code=signal.ts_code,
            trade_date=signal.trade_date,
            action=signal.action,
            price=price,
            quantity=quantity,
            commission=commission,
            status="FILLED",
            reason=signal.reason,
        )
        self.orders.append(order)
        self._persist()
        return order
