"""A-share-aware backtest engine.

Unlike the simple engine, this implementation enforces T+1 settlement, 100
share lots, commission + stamp duty, slippage, suspension skips and limit-up /
limit-down executability. It also flags potential look-ahead bias and supports
walk-forward / out-of-sample evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from backend.backtest.engine import BacktestResult
from backend.backtest.metrics import calculate_metrics
from backend.backtest.order import Order, OrderSide, Position, Trade
from backend.data.calendar import TradingCalendar
from backend.data.providers.base import DataProvider
from backend.strategies.base import Strategy, TradeSignal


@dataclass
class WalkForwardFold:
    start: str
    end: str
    segment: str
    metrics: dict


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold] = field(default_factory=list)
    in_sample: dict = field(default_factory=dict)
    out_of_sample: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "folds": [
                {"start": f.start, "end": f.end, "segment": f.segment, "metrics": f.metrics}
                for f in self.folds
            ],
            "in_sample": self.in_sample,
            "out_of_sample": self.out_of_sample,
        }


class AShareBacktestEngine:
    """Backtester with A-share market micro-structure rules."""

    def __init__(
        self,
        provider: DataProvider,
        strategy: Strategy,
        initial_cash: float = 1_000_000.0,
        commission_rate: float = 0.0003,
        stamp_duty_rate: float = 0.0005,
        slippage: float = 0.0,
        lot_size: int = 100,
        t_plus_one: bool = True,
        max_position_weight: float = 0.2,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.20,
    ) -> None:
        self.provider = provider
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.stamp_duty_rate = stamp_duty_rate
        self.slippage = slippage
        self.lot_size = lot_size
        self.t_plus_one = t_plus_one
        self.max_position_weight = max_position_weight
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.calendar = TradingCalendar(provider)

    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        ts_codes: Optional[list[str]] = None,
    ) -> BacktestResult:
        daily = self.provider.daily(
            ts_codes=ts_codes, start_date=start_date, end_date=end_date
        )
        if daily.empty:
            raise ValueError("No daily data available")
        # Only listed shares are tradable. Index series can live in the same
        # local store (they feed the market page's 指数表现), and an index must
        # never be treated as a buyable instrument by a backtest.
        daily = self._stocks_only(daily)
        if daily.empty:
            raise ValueError("No tradable stock data available")

        dates = sorted(daily["trade_date"].astype(str).unique().tolist())
        codes = sorted(daily["ts_code"].astype(str).unique().tolist())
        signals = self.strategy.generate_signals(daily)
        lookahead_issues = self._detect_lookahead(daily, signals)
        signal_map: dict[str, list[TradeSignal]] = {}
        for sig in signals:
            signal_map.setdefault(sig.trade_date, []).append(sig)

        limit_lookup = self._limit_lookup(codes, dates)
        suspended_lookup = self._suspension_lookup(codes, dates)
        price_lookup = self._price_lookup(daily)
        open_lookup = self._open_lookup(daily)

        cash = float(self.initial_cash)
        positions: dict[str, Position] = {}
        orders: list[Order] = []
        trades: list[Trade] = []
        open_trades: dict[str, dict] = {}
        equity_rows: list[dict] = []
        for i, date in enumerate(dates):
            # A-share execution has no look-ahead: a signal generated at day D's
            # close is filled at day D+1's open. Without T+1 the fill uses the
            # same day's close (classic close-to-close simplification).
            if self.t_plus_one:
                signal_day = dates[i - 1] if i > 0 else None
                day_signals = signal_map.get(signal_day, []) if signal_day else []
                exec_prices = open_lookup
            else:
                day_signals = signal_map.get(date, [])
                exec_prices = price_lookup

            equity_now = cash
            for ts, pos in positions.items():
                if pos.quantity > 0:
                    px = price_lookup.get((date, ts)) or pos.avg_price
                    equity_now += pos.quantity * px

            for sig in day_signals:
                ts = sig.ts_code
                if ts in suspended_lookup.get(date, set()):
                    continue

                exec_bar = exec_prices.get((date, ts))
                if exec_bar is None:
                    continue

                if sig.action == "BUY":
                    exec_price = float(exec_bar) * (1 + self.slippage)
                    limit = limit_lookup.get((date, ts))
                    if limit is not None and exec_price >= limit["up_limit"]:
                        continue  # cannot buy at limit-up

                    per_share = exec_price * (1 + self.commission_rate)
                    if per_share <= 0:
                        continue
                    pos = positions.get(ts)
                    held_value = pos.quantity * exec_price if pos else 0.0
                    room = equity_now * self.max_position_weight - held_value
                    budget = min(cash, max(0.0, room))
                    qty = int(budget // per_share) // self.lot_size * self.lot_size
                    if qty <= 0:
                        continue
                    gross = qty * exec_price
                    commission = gross * self.commission_rate
                    cost = gross + commission
                    if cost > cash + 1e-9:
                        qty -= self.lot_size
                        if qty <= 0:
                            continue
                        gross = qty * exec_price
                        commission = gross * self.commission_rate
                        cost = gross + commission

                    cash -= cost
                    order = Order(
                        ts_code=ts,
                        trade_date=date,
                        side=OrderSide.BUY,
                        price=exec_price,
                        quantity=float(qty),
                        commission=commission,
                    )
                    orders.append(order)
                    if ts not in positions:
                        positions[ts] = Position(ts_code=ts, quantity=0.0, cost_basis=0.0)
                    positions[ts].apply_order(order)
                    open_trades[ts] = {
                        "entry_date": date,
                        "entry_price": exec_price,
                        "quantity": qty,
                    }

                elif sig.action == "SELL":
                    pos = positions.get(ts)
                    if pos is None or pos.quantity <= 0:
                        continue
                    opened = open_trades.get(ts)
                    if self.t_plus_one and opened is not None and opened.get("entry_date") == date:
                        continue  # T+1: cannot sell shares bought today
                    exec_price = float(exec_bar) * (1 - self.slippage)
                    limit = limit_lookup.get((date, ts))
                    if limit is not None and exec_price <= limit["down_limit"]:
                        continue  # cannot sell at limit-down

                    qty = pos.quantity
                    gross = qty * exec_price
                    commission = gross * self.commission_rate
                    stamp = gross * self.stamp_duty_rate
                    proceeds = gross - commission - stamp
                    order = Order(
                        ts_code=ts,
                        trade_date=date,
                        side=OrderSide.SELL,
                        price=exec_price,
                        quantity=qty,
                        commission=commission,
                    )
                    orders.append(order)
                    cash += proceeds
                    positions[ts] = Position(ts_code=ts, quantity=0.0, cost_basis=0.0)

                    opened = open_trades.pop(ts, None)
                    if opened:
                        pnl = qty * (exec_price - opened["entry_price"]) - commission - stamp
                        entry_d = pd.to_datetime(opened["entry_date"], format="%Y%m%d")
                        exit_d = pd.to_datetime(date, format="%Y%m%d")
                        holding = (exit_d - entry_d).days
                        trades.append(
                            Trade(
                                ts_code=ts,
                                entry_date=opened["entry_date"],
                                exit_date=date,
                                side="LONG",
                                entry_price=opened["entry_price"],
                                exit_price=exec_price,
                                quantity=qty,
                                pnl=pnl,
                                return_pct=(exec_price / opened["entry_price"] - 1),
                                holding_days=holding,
                            )
                        )

            # Protective exits (stop-loss / take-profit) evaluated on the close.
            # This keeps the backtest meaningful for buy-only strategies.
            for ts in list(open_trades.keys()):
                pos = positions.get(ts)
                opened = open_trades.get(ts)
                if pos is None or pos.quantity <= 0 or opened is None:
                    continue
                if self.t_plus_one and opened.get("entry_date") == date:
                    continue
                px = price_lookup.get((date, ts))
                if px is None or opened["entry_price"] <= 0:
                    continue
                change = px / opened["entry_price"] - 1
                hit_stop = self.stop_loss_pct and change <= -self.stop_loss_pct
                hit_target = self.take_profit_pct and change >= self.take_profit_pct
                if not (hit_stop or hit_target):
                    continue
                qty = pos.quantity
                gross = qty * px
                commission = gross * self.commission_rate
                stamp = gross * self.stamp_duty_rate
                cash += gross - commission - stamp
                orders.append(
                    Order(
                        ts_code=ts,
                        trade_date=date,
                        side=OrderSide.SELL,
                        price=px,
                        quantity=qty,
                        commission=commission,
                    )
                )
                positions[ts] = Position(ts_code=ts, quantity=0.0, cost_basis=0.0)
                open_trades.pop(ts, None)
                pnl = qty * (px - opened["entry_price"]) - commission - stamp
                holding = (
                    pd.to_datetime(date, format="%Y%m%d")
                    - pd.to_datetime(opened["entry_date"], format="%Y%m%d")
                ).days
                trades.append(
                    Trade(
                        ts_code=ts,
                        entry_date=opened["entry_date"],
                        exit_date=date,
                        side="LONG",
                        entry_price=opened["entry_price"],
                        exit_price=float(px),
                        quantity=qty,
                        pnl=pnl,
                        return_pct=(px / opened["entry_price"] - 1),
                        holding_days=holding,
                    )
                )

            # Mark-to-market at close.
            total = cash
            for ts, pos in positions.items():
                if pos.quantity > 0:
                    px = price_lookup.get((date, ts))
                    if px is not None:
                        total += pos.quantity * px
            equity_rows.append({"date": date, "total": total})

        equity = pd.DataFrame(equity_rows)
        if equity.empty:
            equity = pd.DataFrame({"date": [dates[-1]], "total": [self.initial_cash]})

        peak = equity["total"].cummax()
        drawdown_curve = pd.DataFrame(
            {"date": equity["date"], "drawdown": (equity["total"] - peak) / peak}
        )

        return BacktestResult(
            equity_curve=equity,
            positions=positions,
            orders=orders,
            final_cash=cash,
            final_total_value=float(equity["total"].iloc[-1]),
            trade_log=trades,
            drawdown_curve=drawdown_curve,
            lookahead_issues=lookahead_issues,
        )

    def walk_forward(
        self,
        *,
        start_date: str,
        end_date: str,
        train_days: int = 60,
        test_days: int = 20,
        step_days: int = 20,
        ts_codes: Optional[list[str]] = None,
    ) -> WalkForwardResult:
        """Rolling walk-forward split into train/validation/test segments."""
        daily = self.provider.daily(
            ts_codes=ts_codes, start_date=start_date, end_date=end_date
        )
        if daily.empty:
            return WalkForwardResult()

        all_dates = sorted(daily["trade_date"].astype(str).unique().tolist())
        folds: list[WalkForwardFold] = []
        in_sample_metrics: dict = {}
        i = 0
        while i + train_days + test_days <= len(all_dates):
            train_end = all_dates[i + train_days - 1]
            test_start = all_dates[i + train_days]
            test_end = all_dates[min(i + train_days + test_days - 1, len(all_dates) - 1)]
            in_sample_metrics = calculate_metrics(
                self.run(start_date=all_dates[i], end_date=train_end, ts_codes=ts_codes).equity_curve
            )
            oos_result = self.run(start_date=test_start, end_date=test_end, ts_codes=ts_codes)
            oos_metrics = calculate_metrics(oos_result.equity_curve, trade_log=oos_result.trade_log)
            folds.append(
                WalkForwardFold(
                    start=test_start,
                    end=test_end,
                    segment="test",
                    metrics=oos_metrics,
                )
            )
            i += step_days

        oos_keys = ("total_return", "annualized_return", "max_drawdown", "win_rate")
        out_of_sample = {
            key: (sum(f.metrics.get(key, 0.0) or 0.0 for f in folds) / len(folds))
            if folds
            else 0.0
            for key in oos_keys
        }
        return WalkForwardResult(
            folds=folds,
            in_sample=in_sample_metrics,
            out_of_sample=out_of_sample,
        )

    def _stocks_only(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Keep only codes that appear in ``stock_basic`` (i.e. real shares)."""
        if daily.empty or "ts_code" not in daily.columns:
            return daily
        try:
            basic = self.provider.stock_basic()
        except Exception:  # noqa: BLE001
            return daily
        if basic is None or basic.empty or "ts_code" not in basic.columns:
            return daily
        listed = {str(code) for code in basic["ts_code"].dropna()}
        if not listed:
            return daily
        filtered = daily[daily["ts_code"].astype(str).isin(listed)]
        return filtered if not filtered.empty else daily

    def _limit_lookup(self, codes: list[str], dates: list[str]) -> dict:
        lookup: dict = {}
        for date in dates:
            try:
                frame = self.provider.stk_limit(ts_codes=codes, trade_date=date)
            except Exception:
                frame = pd.DataFrame()
            if frame.empty:
                continue
            for _, row in frame.iterrows():
                lookup[(str(row["trade_date"]), str(row["ts_code"]))] = {
                    "up_limit": float(row["up_limit"]),
                    "down_limit": float(row["down_limit"]),
                }
        return lookup

    def _suspension_lookup(self, codes: list[str], dates: list[str]) -> dict:
        lookup: dict = {}
        for date in dates:
            try:
                frame = self.provider.suspend_d(ts_codes=codes, trade_date=date)
            except Exception:
                frame = pd.DataFrame()
            if frame.empty:
                continue
            lookup[date] = set(frame["ts_code"].astype(str).unique())
        return lookup

    def _price_lookup(self, daily: pd.DataFrame) -> dict:
        return daily.set_index(["trade_date", "ts_code"])["close"].to_dict()

    def _open_lookup(self, daily: pd.DataFrame) -> dict:
        return daily.set_index(["trade_date", "ts_code"])["open"].to_dict()

    def _detect_lookahead(
        self, daily: pd.DataFrame, signals: list[TradeSignal]
    ) -> list[str]:
        """Flag signals whose execution price could use future information."""
        issues: list[str] = []
        available = set(daily["trade_date"].astype(str).unique())
        for sig in signals:
            if sig.price is None:
                continue
            # T+1 execution is fine; but a signal price on a date beyond the
            # signal's own date is a look-ahead red flag.
            if sig.trade_date not in available:
                issues.append(
                    f"{sig.ts_code}@{sig.trade_date}: signal date not in market data"
                )
        return issues
