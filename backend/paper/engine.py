"""Offline paper trading engine.

The engine consumes normalized ``TradeSignal`` objects and executes them under
A-share-friendly defaults: 100-share lots, commission, and optional slippage.
It does not mutate the strategy or provider layers.
"""

from __future__ import annotations

import json
import math
import threading
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from backend.paper.models import PaperOrder, PaperPosition, PaperSnapshot
from backend.strategies.base import TradeSignal


class IdempotencyConflict(ValueError):
    """The caller reused a completed request's identity for a different order."""


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
        stamp_duty_rate: float = 0.0,
    ) -> None:
        if not math.isfinite(initial_cash) or initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if not math.isfinite(lot_size) or lot_size <= 0 or int(lot_size) != lot_size:
            raise ValueError("lot_size must be positive")
        if not math.isfinite(commission_rate) or commission_rate < 0:
            raise ValueError("commission_rate cannot be negative")
        if not math.isfinite(slippage) or slippage < 0 or slippage >= 1:
            raise ValueError("slippage cannot be negative")

        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.lot_size = int(lot_size)
        self.commission_rate = float(commission_rate)
        self.slippage = float(slippage)
        if not math.isfinite(stamp_duty_rate) or stamp_duty_rate < 0:
            raise ValueError("stamp_duty_rate must be finite and non-negative")
        self.stamp_duty_rate = stamp_duty_rate
        self.t_plus_one = bool(t_plus_one)
        self.journal = journal
        self._lock = threading.RLock()
        self.version = 0
        self.peak_equity = self.initial_cash
        self.trade_records: list[dict] = []
        self.requests: dict[str, dict] = {}
        self.journal_pending = False
        self._state_seen = False
        # Paper positions must survive a restart: losing them made the review
        # page disagree with the decision journal after every service restart.
        self.state_path = Path(state_path) if state_path else None
        self.positions: dict[str, PaperPosition] = {}
        self.orders: list[PaperOrder] = []
        if self.state_path is not None:
            self.load()

    def reset(self) -> None:
        """Restore the account to its initial state."""

        with self._transaction():
            before = self._checkpoint()
            try:
                self.cash = self.initial_cash
                self.positions.clear()
                self.orders.clear()
                self.requests.clear()
                self.peak_equity = self.initial_cash
                self.version += 1
                self._persist()
            except Exception:
                self._restore(before)
                raise

    def _checkpoint(self):
        return deepcopy((self.cash, self.positions, self.orders, self.version,
                         self.peak_equity, self.trade_records, self.requests))

    def _restore(self, checkpoint):
        (self.cash, self.positions, self.orders, self.version,
         self.peak_equity, self.trade_records, self.requests) = checkpoint

    @contextmanager
    def _transaction(self):
        """Refresh and mutate one account under a stable OS file lock."""
        with self._lock:
            if self.state_path is None:
                yield
                return
            from backend.data.atomic import file_lock
            with file_lock(self.state_path):
                self._load_state()
                yield

    def order_for_signal(self, signal_id: str) -> PaperOrder | None:
        """Read an existing fill after observing other account writers."""
        with self._transaction():
            return next((order for order in self.orders if order.signal_id == signal_id), None)

    def replay_order(self, signal_id: str, *, ts_code: str, action: str,
                     price: float, quantity: float, trade_date: str | None = None) -> PaperOrder | None:
        """Recover a completed request without ever executing a new order.

        An omitted date belongs to the original request, even after midnight.
        Looking up and comparing the durable request share the account lock.
        """
        if not signal_id:
            return None
        with self._transaction():
            previous = next((order for order in self.orders if order.signal_id == signal_id), None)
            if previous is None:
                return None
            request = {"ts_code": ts_code, "action": action,
                       "date": trade_date or previous.trade_date,
                       "price": float(price), "quantity": float(quantity)}
            self._check_request(previous, request)
            self._sync_journal()
            return deepcopy(previous)

    def _check_request(self, previous: PaperOrder, request: dict) -> None:
        saved = self.requests.get(previous.signal_id)
        if saved is not None:
            matches = saved == request
        else:
            # Older ledgers did not retain requested quantities. Only a request
            # matching the durable fill can be recovered with confidence.
            matches = (previous.ts_code, previous.action, previous.trade_date,
                       previous.price, previous.quantity) == (
                       request["ts_code"], request["action"], request["date"],
                       self._execution_price(request["action"], request["price"]),
                       request["quantity"])
        if not matches:
            raise IdempotencyConflict("相同 signal_id 不能用于不同请求")

    # -- persistence ---------------------------------------------------
    def _persist(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": self.cash,
            "initial_cash": self.initial_cash,
            "version": self.version,
            "peak_equity": self.peak_equity,
            "trade_records": self.trade_records,
            "requests": self.requests,
            "positions": {
                code: {
                    "ts_code": p.ts_code,
                    "quantity": p.quantity,
                    "avg_price": p.avg_price,
                    "realized_pnl": p.realized_pnl,
                    "buy_date": getattr(p, "buy_date", None),
                    "buy_lots": p.buy_lots,
                }
                for code, p in self.positions.items()
            },
            "orders": [o.to_dict() for o in self.orders],
        }
        from backend.data.atomic import atomic_write
        atomic_write(self.state_path, json.dumps(payload, ensure_ascii=False, allow_nan=False))
        self._state_seen = True

    def load(self) -> bool:
        """Restore cash / positions / orders. Returns True when state was read."""
        with self._lock:
            if self.state_path is None:
                return False
            from backend.data.atomic import file_lock
            with file_lock(self.state_path):
                loaded = self._load_state()
                self._sync_journal()
                return loaded

    @staticmethod
    def _number(value, *, minimum=None) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("account number must be numeric")
        number = float(value)
        if not math.isfinite(number) or (minimum is not None and number < minimum):
            raise ValueError("account number must be finite and within range")
        return number

    @staticmethod
    def _date(value) -> str:
        if not isinstance(value, str) or len(value) != 8 or not value.isascii() or not value.isdigit():
            raise ValueError("account date must be YYYYMMDD")
        datetime.strptime(value, "%Y%m%d")
        return value

    def _load_state(self) -> bool:
        if self.state_path is None:
            return False
        if not self.state_path.exists():
            if self._state_seen:
                raise RuntimeError("账户状态文件丢失，交易已停止；请恢复原文件")
            return False
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("account must be an object")
            cash = self._number(payload["cash"], minimum=0)
            initial_cash = self._number(payload["initial_cash"], minimum=0)
            peak_equity = self._number(payload.get("peak_equity", initial_cash), minimum=0)
            version = payload.get("version", 0)
            if initial_cash <= 0 or peak_equity < initial_cash or type(version) is not int or version < 0:
                raise ValueError("invalid account equity or version")
            raw_positions = payload["positions"]
            raw_orders = payload["orders"]
            trade_records = payload.get("trade_records", [])
            requests = payload.get("requests", {})
            if (not isinstance(raw_positions, dict) or not isinstance(raw_orders, list)
                    or not isinstance(trade_records, list) or not isinstance(requests, dict)):
                raise ValueError("invalid account collection")
            positions = {}
            for code, item in raw_positions.items():
                if not isinstance(code, str) or not code or not isinstance(item, dict):
                    raise ValueError("invalid position")
                if item.get("ts_code", code) != code:
                    raise ValueError("position key does not match its code")
                position = PaperPosition(ts_code=code,
                    quantity=self._number(item["quantity"], minimum=0),
                    avg_price=self._number(item["avg_price"], minimum=0),
                    realized_pnl=self._number(item.get("realized_pnl", 0)),
                    buy_date=item.get("buy_date") or "")
                if position.quantity > 0 and position.avg_price <= 0:
                    raise ValueError("held position must have a positive cost")
                if position.buy_date:
                    self._date(position.buy_date)
                lots = item.get("buy_lots")
                # V1.0 positions have only buy_date. An explicit invalid batch
                # list must not silently turn into a successful migration.
                if lots is None:
                    lots = [{"date": position.buy_date, "quantity": position.quantity}] if position.quantity else []
                if not isinstance(lots, list):
                    raise ValueError("invalid position batches")
                for lot in lots:
                    quantity = self._number(lot["quantity"], minimum=0)
                    if quantity <= 0:
                        raise ValueError("invalid batch quantity")
                    position.buy_lots.append({"date": self._date(lot["date"]), "quantity": quantity})
                if abs(sum(lot["quantity"] for lot in position.buy_lots) - position.quantity) > 1e-6:
                    raise ValueError("position batches do not reconcile")
                positions[code] = position
            orders = []
            signal_ids = set()
            for item in raw_orders:
                order = PaperOrder(**item)
                self._date(order.trade_date)
                if not isinstance(order.ts_code, str) or not order.ts_code or order.action not in ("BUY", "SELL"):
                    raise ValueError("invalid order identity")
                if order.status != "FILLED" or self._number(order.price, minimum=0) <= 0 or self._number(order.quantity, minimum=0) <= 0:
                    raise ValueError("invalid fill")
                self._number(order.commission, minimum=0)
                for value in (order.pnl, order.return_pct, order.holding_days):
                    if value is not None:
                        self._number(value)
                if not isinstance(order.signal_id, str) or (order.signal_id and order.signal_id in signal_ids):
                    raise ValueError("duplicate or invalid order id")
                if order.signal_id:
                    signal_ids.add(order.signal_id)
                orders.append(order)
            for signal_id, request in requests.items():
                if not isinstance(signal_id, str) or not signal_id or not isinstance(request, dict):
                    raise ValueError("invalid idempotency request")
                if request["action"] not in ("BUY", "SELL") or not isinstance(request["ts_code"], str):
                    raise ValueError("invalid idempotency identity")
                self._date(request["date"])
                if self._number(request["price"], minimum=0) <= 0:
                    raise ValueError("invalid request price")
                if request.get("quantity") is not None and self._number(request["quantity"], minimum=0) <= 0:
                    raise ValueError("invalid request quantity")
            from backend.journal.trades import TradeRecord
            trade_ids = set()
            for item in trade_records:
                record = TradeRecord.from_dict(item)
                if not isinstance(record.trade_id, str) or not record.trade_id or record.trade_id in trade_ids:
                    raise ValueError("invalid durable trade event id")
                if record.action not in ("BUY", "SELL") or self._number(record.quantity, minimum=0) <= 0:
                    raise ValueError("invalid durable fill event")
                if self._number(record.price, minimum=0) <= 0:
                    raise ValueError("invalid durable fill price")
                if record.trade_date:
                    self._date(record.trade_date)
                trade_ids.add(record.trade_id)
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            raise RuntimeError("账户状态无法恢复，请检查 state.json；交易已停止") from exc
        # Commit the decoded state only after every section validates. A bad
        # trailing order must never leave a partially replaced live account.
        self.cash, self.initial_cash = cash, initial_cash
        self.version, self.peak_equity = version, peak_equity
        self.positions, self.orders = positions, orders
        self.trade_records, self.requests = trade_records, requests
        self._state_seen = True
        return True

    def execute_signal(
        self,
        signal: TradeSignal,
        *,
        price: float | None = None,
    ) -> PaperOrder:
        """Execute one signal using the signal price or an explicit override."""

        if not isinstance(signal.ts_code, str) or not signal.ts_code.strip():
            raise ValueError("signal must have a non-empty stock code")
        if not isinstance(signal.signal_id, str):
            raise ValueError("signal_id must be a string")
        if price is None:
            if signal.price is None:
                raise ValueError(f"no execution price available for {signal.ts_code}")
            price = signal.price
        if isinstance(price, bool):
            raise ValueError("execution price must be numeric")
        price = float(price)
        if not math.isfinite(price) or price <= 0:
            raise ValueError("execution price must be positive")
        self._date(signal.trade_date)
        if signal.quantity is not None and (isinstance(signal.quantity, bool)
                or not math.isfinite(float(signal.quantity))
                or float(signal.quantity) <= 0):
            raise ValueError("quantity must be finite and positive")
        if signal.action not in ("BUY", "SELL"):
            raise ValueError(f"unsupported signal action: {signal.action}")
        with self._transaction():
            request = {"ts_code": signal.ts_code, "action": signal.action,
                       "date": signal.trade_date, "price": price,
                       "quantity": float(signal.quantity) if signal.quantity is not None else None}
            if signal.signal_id:
                previous = next((o for o in self.orders if o.signal_id == signal.signal_id), None)
                if previous:
                    self._check_request(previous, request)
                    self._sync_journal()
                    return previous
            if self.t_plus_one and self.orders and signal.trade_date < max(o.trade_date for o in self.orders):
                raise ValueError("交易日期不可倒退；历史重放请使用独立账户")
            before = self._checkpoint()
            try:
                order = self._buy(signal, price) if signal.action == "BUY" else self._sell(signal, price)
                order = replace(order, signal_id=signal.signal_id)
                if order.status == "FILLED":
                    self.orders[-1] = order
                    self._record_if_attached(order, signal)
                    if signal.signal_id:
                        self.requests[signal.signal_id] = request
                    self.version += 1
                    self._persist()
            except Exception:
                self._restore(before)
                raise
            # Once state.json is committed, a projection error must not roll
            # back memory while leaving the successful fill on disk.
            self._sync_journal()
            return order

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
        with self._transaction():
            return self._total_equity(prices)

    def _total_equity(self, prices: dict[str, float]) -> float:

        market_value = 0.0
        for ts_code, position in self.positions.items():
            if position.quantity <= 0:
                continue
            current_price = prices.get(ts_code)
            if current_price is None or not math.isfinite(current_price) or current_price <= 0:
                current_price = position.avg_price
            market_value += position.market_value(current_price)
        return self.cash + market_value

    def snapshot(self, prices: dict[str, float] | None = None) -> PaperSnapshot:
        """Return a serializable view of the current account state."""

        with self._transaction():
            equity = self._total_equity(prices or {})
            if equity > self.peak_equity:
                before = self.peak_equity
                self.peak_equity = equity
                try:
                    self._persist()
                except Exception:
                    self.peak_equity = before
                    raise
            self._sync_journal()
            return PaperSnapshot(cash=self.cash, positions=deepcopy(self.positions),
                equity=equity, realized_pnl=sum(p.realized_pnl for p in self.positions.values()),
                orders=tuple(self.orders), peak_equity=self.peak_equity,
                version=self.version, journal_pending=self.journal_pending)

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

        # The affordability comparison permits a tiny rounding tolerance;
        # never persist that tolerance as a negative cash balance.
        self.cash = max(0.0, self.cash - total_cost)
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
        return order

    def _record_if_attached(self, order: PaperOrder, signal: TradeSignal) -> None:
        """Persist executed fills to the decision journal when configured."""
        if order.status != "FILLED":
            return
        from backend.journal.trades import TradeRecord

        position = self.positions.get(signal.ts_code)
        qty_after = float(position.quantity) if position else 0.0
        qty_before = max(0.0, qty_after - order.quantity) if signal.action == "BUY" else qty_after + order.quantity
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
                trade_date=signal.trade_date,
                result={"price": order.price, "quantity": order.quantity,
                        "pnl": order.pnl, "return_pct": order.return_pct,
                        "holding_days": order.holding_days},
        )
        self.trade_records.append(record.to_dict())

    def sync_journal(self) -> None:
        """Replay durable fill events to the JSONL projection, without duplicates."""
        with self._transaction():
            self._sync_journal()

    def _sync_journal(self) -> None:
        if self.journal is None:
            return
        from backend.journal.trades import TradeRecord
        try:
            known = {r.trade_id: r for r in self.journal.list()}
            for payload in self.trade_records:
                record = TradeRecord.from_dict(payload)
                previous = known.get(record.trade_id)
                if previous is not None:
                    if not previous.same_payload_as(record):
                        raise ValueError("成交日志与账户中的已成交事件不一致")
                else:
                    self.journal.record(record)
                    known[record.trade_id] = record
            self.journal_pending = False
        except (OSError, ValueError, TypeError, KeyError):
            # Fill events are already committed inside state.json. Report that
            # the journal projection needs recovery rather than lose the fill.
            self.journal_pending = True

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

        held = float(position.quantity)
        available = sum(lot["quantity"] for lot in position.buy_lots
                        if lot["date"] < signal.trade_date) if self.t_plus_one else held
        if available <= 0:
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
                if want != whole_lots:
                    return PaperOrder(ts_code=signal.ts_code, trade_date=signal.trade_date,
                        action=signal.action, price=price, quantity=0, status="REJECTED",
                        reason="部分卖出数量必须为整手")
                quantity = min(float(whole_lots), held)
        else:
            quantity = available
        if quantity > available:
            return PaperOrder(ts_code=signal.ts_code, trade_date=signal.trade_date,
                action=signal.action, price=price, quantity=0, status="REJECTED",
                reason=f"T+1: 可卖数量为 {available:g} 股")
        cost = position.avg_price * quantity
        earliest = min((lot["date"] for lot in position.buy_lots), default=signal.trade_date)
        holding_days = (datetime.strptime(signal.trade_date, "%Y%m%d") -
                        datetime.strptime(earliest or signal.trade_date, "%Y%m%d")).days
        gross = quantity * price
        commission = gross * self.commission_rate
        tax = gross * self.stamp_duty_rate
        proceeds = gross - commission - tax
        realized = position.apply_sell(
            price=price,
            quantity=quantity,
            commission=commission + tax,
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
            pnl=realized,
            return_pct=realized / cost if cost else 0.0,
            holding_days=holding_days,
        )
        self.orders.append(order)
        return order
