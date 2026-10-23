"""A-share-aware backtest engine.

Unlike the simple engine, this implementation enforces T+1 settlement, 100
share lots, commission + stamp duty, slippage, suspension skips and limit-up /
limit-down executability. It also flags potential look-ahead bias and supports
walk-forward / out-of-sample evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
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
    train_start: str = ""
    train_end: str = ""
    train_metrics: dict = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    folds: list[WalkForwardFold] = field(default_factory=list)
    in_sample: dict = field(default_factory=dict)
    out_of_sample: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "aggregation": "independent_fold_means; win_rate weighted by closed trades",
            "folds": [
                {"start": f.start, "end": f.end, "segment": f.segment, "metrics": f.metrics,
                 "train_start": f.train_start, "train_end": f.train_end, "train_metrics": f.train_metrics}
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
        for name, value in (("slippage", slippage), ("max_position_weight", max_position_weight),
                            ("stop_loss_pct", stop_loss_pct), ("take_profit_pct", take_profit_pct)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= slippage < 1:
            raise ValueError("slippage must be in [0, 1)")
        if not 0 < max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0, 1]")
        if not 0 <= stop_loss_pct < 1 or take_profit_pct < 0:
            raise ValueError("invalid protective exit percentage")
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
        *,
        warmup_start_date: Optional[str] = None,
    ) -> BacktestResult:
        if warmup_start_date and (not start_date or warmup_start_date > start_date):
            raise ValueError("warmup_start_date must be on or before start_date")
        daily = self.provider.daily(
            ts_codes=ts_codes, start_date=warmup_start_date or start_date, end_date=end_date
        )
        if daily.empty:
            raise ValueError("No daily data available")
        # Only listed shares are tradable. Index series can live in the same
        # local store (they feed the market page's 指数表现), and an index must
        # never be treated as a buyable instrument by a backtest.
        daily = self._stocks_only(daily)
        if daily.empty:
            raise ValueError("No tradable stock data available")
        daily = daily.copy()
        daily["trade_date"] = daily["trade_date"].astype(str)
        daily["ts_code"] = daily["ts_code"].astype(str)
        if daily.duplicated(["trade_date", "ts_code"]).any():
            raise ValueError("回测日线存在重复的股票/日期")
        for column in ("open", "high", "low", "close"):
            values = pd.to_numeric(daily[column], errors="coerce")
            if not values.map(lambda value: math.isfinite(value) and value > 0).all():
                raise ValueError(f"回测日线 {column} 存在无效价格")
            daily[column] = values
        if ((daily.high < daily[["open", "close"]].max(axis=1)) |
                (daily.low > daily[["open", "close"]].min(axis=1)) | (daily.high < daily.low)).any():
            raise ValueError("回测日线存在 OHLC 矛盾")

        all_dates = sorted(daily["trade_date"].astype(str).unique().tolist())
        dates = [day for day in all_dates if not start_date or day >= start_date]
        if not dates:
            raise ValueError("No daily data available in execution window")
        previous_day = dict(zip(all_dates[1:], all_dates[:-1]))
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
        # Index once; filtering the entire market for every fill was quadratic.
        bar_lookup = daily.set_index(["trade_date", "ts_code"])[["low", "vol"]].sort_index()

        from backend.paper.engine import PaperTradingEngine
        from backend.backtest.execution import rejection_reason

        broker = PaperTradingEngine(initial_cash=self.initial_cash, lot_size=self.lot_size,
            commission_rate=self.commission_rate, stamp_duty_rate=self.stamp_duty_rate,
            t_plus_one=self.t_plus_one)
        orders: list[Order] = []
        trades: list[Trade] = []
        equity_rows: list[dict] = []
        pending_exits: set[str] = set()
        last_prices: dict[str, float] = {}
        if warmup_start_date:
            before = daily[daily.trade_date < start_date].sort_values("trade_date")
            last_prices.update(before.groupby("ts_code").close.last().to_dict())
        for i, date in enumerate(dates):
            signal_day = previous_day.get(date) if self.t_plus_one else date
            day_signals = list(signal_map.get(signal_day, []))
            # A confirmed close-based protective exit is queued for the next open.
            day_signals = [TradeSignal(ts_code=ts, trade_date=date, action="SELL")
                           for ts in sorted(pending_exits)] + day_signals
            seen_sells = set()
            for sig in day_signals:
                ts = sig.ts_code
                if sig.action == "SELL" and ts in seen_sells:
                    continue
                if sig.action == "BUY" and ts in pending_exits:
                    continue
                source = open_lookup if self.t_plus_one or ts in pending_exits else price_lookup
                base = source.get((date, ts))
                if base is None:
                    continue
                price = float(base) * (1 + self.slippage if sig.action == "BUY" else 1 - self.slippage)
                bar = bar_lookup.loc[(date, ts)].to_dict()
                volume = float(bar.get("vol", float("nan")))
                if not math.isfinite(volume) or volume <= 0:
                    continue  # A price carried through a halt is not a fill.
                if rejection_reason(sig.action, price, limit_lookup.get((date, ts)),
                    suspended=ts in suspended_lookup.get(date, set()),
                    bar=bar):
                    continue
                position = broker.positions.get(ts)
                entry_price = position.avg_price if position else 0.0
                entry_date = min((lot["date"] for lot in position.buy_lots), default=date) if position else date
                if sig.action == "BUY":
                    # Size at the execution open using only previously known prices.
                    mark = dict(last_prices)
                    mark[ts] = float(base)
                    equity_now = broker.total_equity(mark)
                    held = position.quantity * price if position else 0.0
                    budget = min(broker.cash, max(0, equity_now * self.max_position_weight - held))
                    qty = int(budget // (price * (1 + self.commission_rate))) // self.lot_size * self.lot_size
                    if sig.quantity is not None:
                        qty = min(qty, int(sig.quantity) // self.lot_size * self.lot_size)
                    if qty <= 0:
                        continue
                else:
                    if position is None or position.quantity <= 0:
                        pending_exits.discard(ts)
                        continue
                    qty = sum(lot["quantity"] for lot in position.buy_lots if lot["date"] < date) if self.t_plus_one else position.quantity
                    if sig.quantity is not None:
                        qty = min(qty, float(sig.quantity))
                    if qty <= 0:
                        continue
                fill = broker.execute_signal(TradeSignal(ts_code=ts, trade_date=date,
                    action=sig.action, price=price, quantity=qty))
                if fill.status != "FILLED":
                    continue
                orders.append(Order(ts_code=ts, trade_date=date,
                    side=OrderSide.BUY if sig.action == "BUY" else OrderSide.SELL,
                    price=fill.price, quantity=fill.quantity, commission=fill.commission))
                if sig.action == "SELL":
                    seen_sells.add(ts)
                    if broker.positions[ts].quantity <= 0:
                        pending_exits.discard(ts)
                    trades.append(Trade(ts_code=ts, entry_date=entry_date, exit_date=date,
                        side="LONG", entry_price=entry_price, exit_price=price,
                        quantity=fill.quantity, pnl=fill.pnl, return_pct=fill.return_pct,
                        holding_days=fill.holding_days))
            # Observe close prices only after execution decisions have been made.
            for ts in codes:
                if (date, ts) in price_lookup:
                    last_prices[ts] = price_lookup[(date, ts)]
            for ts, position in broker.positions.items():
                if position.quantity <= 0:
                    continue
                px = price_lookup.get((date, ts))
                if px is None or position.avg_price <= 0:
                    continue
                change = px / position.avg_price - 1
                if ((self.stop_loss_pct and change <= -self.stop_loss_pct) or
                        (self.take_profit_pct and change >= self.take_profit_pct)):
                    pending_exits.add(ts)
            equity_rows.append({"date": date, "total": broker.total_equity(last_prices)})
        cash = broker.cash
        positions = {code: Position(ts_code=code, quantity=p.quantity,
                     cost_basis=p.avg_price * p.quantity) for code, p in broker.positions.items()}

        equity = pd.DataFrame(equity_rows)
        if equity.empty:
            equity = pd.DataFrame({"date": [dates[-1]], "total": [self.initial_cash]})
        # The first close can already include a fill and fees (e.g. a warmed
        # test fold). Metrics must compare it with the opening cash baseline.
        equity.attrs["initial_equity"] = float(self.initial_cash)

        peak = equity["total"].cummax().clip(lower=self.initial_cash)
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
        """Independent rolling train/test runs, not a stitched equity curve.

        Each fold starts in cash; its test indicators use only the preceding
        training bars as warmup, with no training positions carried forward.
        No parameters are fitted automatically. Returns are fold means; the
        win rate pools closed trades rather than assigning no-trade folds 0%.
        """
        for name, value in (("train_days", train_days), ("test_days", test_days), ("step_days", step_days)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        daily = self.provider.daily(
            ts_codes=ts_codes, start_date=start_date, end_date=end_date
        )
        if daily.empty:
            return WalkForwardResult()

        all_dates = sorted(daily["trade_date"].astype(str).unique().tolist())
        folds: list[WalkForwardFold] = []
        i = 0
        while i + train_days + test_days <= len(all_dates):
            train_end = all_dates[i + train_days - 1]
            test_start = all_dates[i + train_days]
            test_end = all_dates[min(i + train_days + test_days - 1, len(all_dates) - 1)]
            in_sample_result = self.run(start_date=all_dates[i], end_date=train_end, ts_codes=ts_codes)
            in_sample_metrics = calculate_metrics(in_sample_result.equity_curve, trade_log=in_sample_result.trade_log)
            oos_result = self.run(start_date=test_start, end_date=test_end, ts_codes=ts_codes,
                                  warmup_start_date=all_dates[i])
            oos_metrics = calculate_metrics(oos_result.equity_curve, trade_log=oos_result.trade_log)
            folds.append(
                WalkForwardFold(
                    start=test_start,
                    end=test_end,
                    segment="test",
                    metrics=oos_metrics,
                    train_start=all_dates[i], train_end=train_end, train_metrics=in_sample_metrics,
                )
            )
            i += step_days

        def aggregate(metrics):
            count = sum(m.get("trade_count", 0) for m in metrics)
            return {
                **{key: sum(m.get(key, 0.0) for m in metrics) / len(metrics) if metrics else 0.0
                   for key in ("total_return", "annualized_return", "max_drawdown")},
                "win_rate": sum((m.get("win_rate") or 0) * m.get("trade_count", 0)
                                for m in metrics) / count if count else None,
                "trade_count": count, "fold_count": len(metrics),
            }
        return WalkForwardResult(
            folds=folds,
            in_sample=aggregate([f.train_metrics for f in folds]),
            out_of_sample=aggregate([f.metrics for f in folds]),
        )

    def _stocks_only(self, daily: pd.DataFrame) -> pd.DataFrame:
        """Keep only codes that appear in ``stock_basic`` (i.e. real shares)."""
        if daily.empty or "ts_code" not in daily.columns:
            return daily
        try:
            basic = self.provider.stock_basic()
        except Exception as exc:
            if getattr(self.provider, "name", "") == "fixture":
                return daily  # Explicit offline fixtures may omit the basic table.
            raise ValueError("股票名单不可用，无法验证回测标的") from exc
        if basic is None or basic.empty or "ts_code" not in basic.columns:
            raise ValueError("股票名单为空，无法验证回测标的")
        listed = {str(code) for code in basic["ts_code"].dropna()}
        if not listed:
            raise ValueError("股票名单不含有效代码，无法验证回测标的")
        filtered = daily[daily["ts_code"].astype(str).isin(listed)]
        return filtered

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
