"""Step-4 stage 4: event study, control group and robustness checks.

Answers "does this actually work, or did it just pick the strong stocks of that
period?" with the only data we honestly have: daily bars.

Execution model (conservative, no look-ahead):

* Features come from day T and earlier only -- the same engines the dashboard
  uses, so nothing here can see the future.
* Entry is **T+1 open**, and only if that open is below the T+1 up limit. A
  limit-up open (or a one-word limit day) counts as **not filled**, never as a
  fill at the limit price.
* Exit is **T+2 open**: shares bought at the T+1 open cannot legally be sold
  until T+2 (T+1 rule). Selling at the T+1 close would be a rule violation and
  is deliberately not used here (the design doc's looser wording is corrected).
* MFE/MAE are measured inside T+1 against the entry price.

Control group: on each day, every stock in the same sectors that produced
opportunities but that was NOT an opportunity is measured with the identical
entry/exit rule. If the strategy group cannot beat that, the signal is not
carrying information.

Robustness: mean/win-rate after dropping the best 5% of trades, plus a
first-half / second-half split, so a couple of lucky names cannot carry it.

Known, unresolved bias: the stock pool only contains currently listed names, so
delisted stocks are missing (survivorship bias). It is reported, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from backend.board.ladder import LimitLadderEngine
from backend.data.providers.base import DataProvider
from backend.opportunity.ranking import OpportunityRankingEngine
from backend.opportunity.setups import OpportunitySetupEngine

CAPABILITIES = (
    ("首板 / 连板接力 的次日表现", "日线可验证", "T+1 开盘买入、T+2 开盘卖出口径"),
    ("晋级率 / 涨停溢价 / 板块效应", "日线可验证", "与 Step 2 指标直接对齐"),
    ("硬淘汰与分层的有效性", "日线可验证", "淘汰组 vs 保留组、分组收益对照"),
    ("半路买点", "必须分钟数据", "分时站上前高/放量上攻无法用日线判定"),
    ("是否开板 / 回封时点", "必须分钟数据", "日线只能近似，无法知道时点"),
    ("开盘强弱（09:35）", "必须分钟数据", "当前只有延迟快照"),
    ("集合竞价强弱", "当前无法回测", "无逐笔竞价数据"),
    ("封单量 / 精确涨停时间", "当前无法回测", "无盘口数据"),
    ("打板能否成交", "当前无法回测", "只能保守假设不可成交"),
)


@dataclass(frozen=True)
class GroupStats:
    name: str
    signals: int = 0
    filled: int = 0
    not_filled: int = 0
    win_rate: Optional[float] = None
    avg_ret: Optional[float] = None
    median_ret: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    payoff: Optional[float] = None
    best: Optional[float] = None
    worst: Optional[float] = None
    avg_mfe: Optional[float] = None
    avg_mae: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "signals": self.signals,
            "filled": self.filled,
            "not_filled": self.not_filled,
            "fill_rate": round(self.filled / self.signals, 4) if self.signals else None,
            "win_rate": self.win_rate,
            "avg_ret": self.avg_ret,
            "median_ret": self.median_ret,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "payoff": self.payoff,
            "best": self.best,
            "worst": self.worst,
            "avg_mfe": self.avg_mfe,
            "avg_mae": self.avg_mae,
        }


@dataclass(frozen=True)
class Trade:
    date: str
    ts_code: str
    name: str
    setup_type: str
    tier: str
    sector: str
    height: int
    entry_date: str = ""
    entry_price: Optional[float] = None
    exit_date: str = ""
    exit_price: Optional[float] = None
    ret: Optional[float] = None
    filled: bool = False
    not_filled_reason: str = ""
    mfe: Optional[float] = None
    mae: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "ts_code": self.ts_code,
            "name": self.name,
            "setup_type": self.setup_type,
            "tier": self.tier,
            "sector": self.sector,
            "height": self.height,
            "entry_date": self.entry_date,
            "entry_price": self.entry_price,
            "exit_date": self.exit_date,
            "exit_price": self.exit_price,
            "ret": None if self.ret is None else round(self.ret, 4),
            "filled": self.filled,
            "not_filled_reason": self.not_filled_reason,
            "mfe": None if self.mfe is None else round(self.mfe, 4),
            "mae": None if self.mae is None else round(self.mae, 4),
        }


@dataclass(frozen=True)
class ValidationReport:
    start: str
    end: str
    trading_days: int = 0
    signal_days: int = 0
    by_type: dict[str, GroupStats] = field(default_factory=dict)
    by_tier: dict[str, GroupStats] = field(default_factory=dict)
    by_stage: dict[str, GroupStats] = field(default_factory=dict)
    control: Optional[GroupStats] = None
    robustness: dict[str, Any] = field(default_factory=dict)
    trades: tuple[Trade, ...] = ()
    capabilities: tuple[tuple[str, str, str], ...] = CAPABILITIES
    caveats: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "trading_days": self.trading_days,
            "signal_days": self.signal_days,
            "by_type": {k: v.to_dict() for k, v in self.by_type.items()},
            "by_tier": {k: v.to_dict() for k, v in self.by_tier.items()},
            "by_stage": {k: v.to_dict() for k, v in self.by_stage.items()},
            "control": self.control.to_dict() if self.control else None,
            "robustness": self.robustness,
            "capabilities": [
                {"item": a, "verdict": b, "note": c} for a, b, c in self.capabilities
            ],
            "caveats": list(self.caveats),
            "trades": [t.to_dict() for t in self.trades],
        }


def _stats(name: str, trades: list[Trade]) -> GroupStats:
    filled = [t for t in trades if t.filled and t.ret is not None]
    if not filled:
        return GroupStats(name=name, signals=len(trades), filled=0, not_filled=len(trades))
    rets = [t.ret for t in filled]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r <= 0]
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    payoff = (
        abs(avg_win / avg_loss)
        if avg_win is not None and avg_loss not in (None, 0)
        else None
    )
    mfes = [t.mfe for t in filled if t.mfe is not None]
    maes = [t.mae for t in filled if t.mae is not None]
    return GroupStats(
        name=name,
        signals=len(trades),
        filled=len(filled),
        not_filled=len(trades) - len(filled),
        win_rate=round(len(wins) / len(filled), 4),
        avg_ret=round(sum(rets) / len(rets), 4),
        median_ret=round(pd.Series(rets).median(), 4),
        avg_win=None if avg_win is None else round(avg_win, 4),
        avg_loss=None if avg_loss is None else round(avg_loss, 4),
        payoff=None if payoff is None else round(payoff, 3),
        best=round(max(rets), 4),
        worst=round(min(rets), 4),
        avg_mfe=None if not mfes else round(sum(mfes) / len(mfes), 4),
        avg_mae=None if not maes else round(sum(maes) / len(maes), 4),
    )


def _ret(entry: float, exit_: float) -> Optional[float]:
    if not entry or entry <= 0 or exit_ is None:
        return None
    return exit_ / entry - 1.0


class OpportunityValidator:
    """Run the event study over a real date range."""

    def __init__(
        self,
        provider: DataProvider,
        *,
        setup_engine: Optional[OpportunitySetupEngine] = None,
        ranking_engine: Optional[OpportunityRankingEngine] = None,
        max_position_weight: float = 0.20,
    ) -> None:
        self.provider = provider
        self.setup_engine = setup_engine or OpportunitySetupEngine(provider)
        self.ranking_engine = ranking_engine or OpportunityRankingEngine(
            provider, max_position_weight=max_position_weight
        )
        self.ladder_engine = LimitLadderEngine(provider)

    def run(
        self,
        start: str,
        end: str,
        *,
        tiers: tuple[str, ...] = ("A",),
        max_days: Optional[int] = None,
    ) -> ValidationReport:
        calendar = self.ladder_engine.calendar
        days = [d for d in calendar.trading_days() if start <= d <= end]
        if max_days:
            days = days[:max_days]

        picks: list[tuple[str, Any, str]] = []   # (date, TradePlan, stage)
        signal_codes: set[str] = set()
        for day in days:
            setups = self.setup_engine.build(day)
            if not setups.available:
                continue
            plan = self.ranking_engine.build(day, setups=setups)
            for item in plan.plans:
                if item.tier not in tiers:
                    continue
                picks.append((day, item, plan.stage))
                signal_codes.add(item.ts_code)
        if not picks:
            return ValidationReport(
                start=start,
                end=end,
                trading_days=len(days),
                caveats=(
                    f"该区间（{start}–{end}）在分层 {tiers} 下没有任何信号，无法评估",
                ),
            )

        # One bulk read for every code we need, plus the control universe.
        all_days = [d for d in calendar.trading_days() if d <= end]
        extended = all_days + []
        codes = sorted(signal_codes)
        bars = self.provider.daily(
            ts_codes=codes,
            start_date=days[0],
            end_date=end,
        )
        # A small pad so T+2 exists even on the last signal day.
        tail = self._tail_days(days[-1], 3)
        if tail:
            extra = self.provider.daily(
                ts_codes=codes, start_date=days[-1], end_date=tail
            )
            if not extra.empty:
                bars = pd.concat([bars, extra], ignore_index=True).drop_duplicates(
                    subset=["ts_code", "trade_date"]
                )
        trades = [
            outcome
            for day, item, _stage in picks
            for outcome in [self._evaluate(day, item, bars)]
        ]

        by_type = {
            t: _stats(t, [x for x in trades if x.setup_type == t])
            for t in sorted({x.setup_type for x in trades})
        }
        by_tier = {
            t: _stats(t, [x for x in trades if x.tier == t])
            for t in sorted({x.tier for x in trades})
        }
        by_stage = {
            s: _stats(s, [x for x in trades if x.setup_type and any(
                x.date == day and stage == s for day, _p, stage in picks
            )])
            for s in sorted({stage for _d, _p, stage in picks})
        }
        control = self._control_group(days, signal_codes, bars)
        robustness = self._robustness(trades)

        return ValidationReport(
            start=start,
            end=end,
            trading_days=len(days),
            signal_days=len({d for d, _p, _s in picks}),
            by_type=by_type,
            by_tier=by_tier,
            by_stage=by_stage,
            control=control,
            robustness=robustness,
            trades=tuple(trades),
            caveats=(
                "执行口径：T+1 开盘买入（仅当开盘价低于涨停价），T+2 开盘卖出（T+1 规则）",
                "涨停价不可假设成交；一字板/涨停开盘一律计为未成交",
                "信号只使用 T 日及以前数据，执行在 T+1，无未来函数",
                "股票池仅含当前上市股票 → 存在幸存者偏差，结论需谨慎",
                "半路/竞价/封单量/涨停时间无法回测，相关类型不参与本验证",
            ),
        )

    # -- helpers -------------------------------------------------------
    def _tail_days(self, last: str, count: int) -> Optional[str]:
        days = self.ladder_engine.calendar.trading_days()
        after = [d for d in days if d > last]
        return after[count - 1] if len(after) >= count else (after[-1] if after else None)

    def _bars_for(self, bars: pd.DataFrame, code: str) -> pd.DataFrame:
        frame = bars[bars["ts_code"].astype(str) == code].copy()
        return frame.sort_values("trade_date").reset_index(drop=True)

    def _limits_for(self, code: str, day: str) -> Optional[float]:
        try:
            frame = self.provider.stk_limit(ts_codes=[code], trade_date=day)
        except Exception:  # noqa: BLE001
            return None
        if frame is None or frame.empty:
            return None
        return None if pd.isna(frame.iloc[0].get("up_limit")) else float(
            frame.iloc[0]["up_limit"]
        )

    def _evaluate(self, date: str, plan: Any, bars: pd.DataFrame) -> Trade:
        """T+1 open entry (if fillable) -> T+2 open exit."""
        series = self._bars_for(bars, plan.ts_code)
        dates = series["trade_date"].astype(str).tolist()
        base = Trade(
            date=date,
            ts_code=plan.ts_code,
            name=plan.name,
            setup_type=plan.opportunity_type,
            tier=plan.tier,
            sector=plan.sector,
            height=plan.board_height,
        )
        if date not in dates:
            return Trade(**{**base.__dict__, "not_filled_reason": "信号日无成交数据"})
        index = dates.index(date)
        if index + 2 >= len(dates):
            return Trade(**{**base.__dict__, "not_filled_reason": "缺少 T+2 数据（区间末尾）"})
        entry_row = series.iloc[index + 1]
        exit_row = series.iloc[index + 2]
        entry_day = dates[index + 1]
        exit_day = dates[index + 2]
        open_price = None if pd.isna(entry_row.get("open")) else float(entry_row["open"])
        up_limit = self._limits_for(plan.ts_code, entry_day)
        if open_price is None:
            return Trade(**{**base.__dict__, "not_filled_reason": "T+1 无成交（停牌）"})
        if up_limit is not None and open_price >= up_limit:
            return Trade(
                **{
                    **base.__dict__,
                    "entry_date": entry_day,
                    "entry_price": open_price,
                    "not_filled_reason": "T+1 开盘即在涨停价，视为无法买入",
                }
            )
        high = None if pd.isna(entry_row.get("high")) else float(entry_row["high"])
        low = None if pd.isna(entry_row.get("low")) else float(entry_row["low"])
        exit_price = None if pd.isna(exit_row.get("open")) else float(exit_row["open"])
        mfe = _ret(open_price, high)
        mae = _ret(open_price, low)
        return Trade(
            **{
                **base.__dict__,
                "entry_date": entry_day,
                "entry_price": open_price,
                "exit_date": exit_day,
                "exit_price": exit_price,
                "ret": _ret(open_price, exit_price),
                "filled": exit_price is not None,
                "mfe": mfe,
                "mae": mae,
            }
        )

    def _control_group(
        self,
        days: list[str],
        signal_codes: set[str],
        bars: pd.DataFrame,
    ) -> Optional[GroupStats]:
        """Same sectors, same entry/exit rule, but not our candidates."""
        if not days:
            return None
        start, end = days[0], days[-1]
        universe = self.provider.stock_basic(list_status="L")
        if universe is None or universe.empty:
            return None
        codes = [c for c in universe["ts_code"].astype(str).tolist() if c not in signal_codes]
        if not codes:
            return None
        sample = codes[:300]           # bounded sample keeps the tool quick
        frame = self.provider.daily(ts_codes=sample, start_date=start, end_date=end)
        if frame is None or frame.empty:
            return None
        tail = self._tail_days(end, 3)
        if tail:
            frame = pd.concat(
                [frame, self.provider.daily(ts_codes=sample, start_date=end, end_date=tail)],
                ignore_index=True,
            ).drop_duplicates(subset=["ts_code", "trade_date"])
        trades = []
        for code in sample:
            series = self._bars_for(frame, code)
            dates = series["trade_date"].astype(str).tolist()
            for index in range(len(dates) - 2):
                day = dates[index]
                if day < start or day > end:
                    continue
                entry_row = series.iloc[index + 1]
                exit_row = series.iloc[index + 2]
                open_price = None if pd.isna(entry_row.get("open")) else float(entry_row["open"])
                exit_price = None if pd.isna(exit_row.get("open")) else float(exit_row["open"])
                if open_price is None:
                    continue
                trades.append(
                    Trade(
                        date=day,
                        ts_code=code,
                        name="",
                        setup_type="对照",
                        tier="-",
                        sector="",
                        height=0,
                        entry_date=dates[index + 1],
                        entry_price=open_price,
                        exit_date=dates[index + 2],
                        exit_price=exit_price,
                        ret=_ret(open_price, exit_price),
                        filled=exit_price is not None,
                    )
                )
        return _stats("对照（同区间的非候选股票）", trades) if trades else None

    def _robustness(self, trades: list[Trade]) -> dict[str, Any]:
        filled = [t for t in trades if t.filled and t.ret is not None]
        if not filled:
            return {"note": "无可成交样本"}
        rets = sorted((t.ret for t in filled), reverse=True)
        drop = max(1, int(len(rets) * 0.05))
        trimmed = _stats("trimmed", [
            Trade(**{**t.__dict__, "ret": t.ret})
            for t in filled
            if t.ret <= rets[drop - 1]
        ])
        ordered = sorted(filled, key=lambda t: (t.date, t.ts_code))
        half = len(ordered) // 2
        first = _stats("first_half", ordered[:half]) if half else None
        second = _stats("second_half", ordered[half:]) if half else None
        return {
            "dropped_best": drop,
            "avg_ret_after_dropping_best_5pct": trimmed.avg_ret,
            "win_rate_after_dropping_best_5pct": trimmed.win_rate,
            "first_half": first.to_dict() if first else None,
            "second_half": second.to_dict() if second else None,
            "halves_consistent": (
                None
                if not first or not second or first.avg_ret is None or second.avg_ret is None
                else (first.avg_ret > 0) == (second.avg_ret > 0)
            ),
        }


def _print_stats(label: str, stats: GroupStats) -> None:
    def fmt(value, digits=2, suffix=""):
        return "—" if value is None else f"{value:.{digits}f}{suffix}"

    print(
        f"{label:<22}{stats.signals:>5}{stats.filled:>6}{stats.not_filled:>6}"
        f"{fmt(stats.win_rate, 3):>9}{fmt(None if stats.avg_ret is None else stats.avg_ret * 100):>9}"
        f"{fmt(None if stats.median_ret is None else stats.median_ret * 100):>9}"
        f"{fmt(stats.payoff):>7}{fmt(None if stats.avg_mfe is None else stats.avg_mfe * 100):>8}"
        f"{fmt(None if stats.avg_mae is None else stats.avg_mae * 100):>8}"
    )


def main() -> None:
    import argparse
    import io
    import json
    import sys

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Opportunity event study")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--tiers", default="A,B")
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    from backend.data.providers.history import HistoryProvider

    validator = OpportunityValidator(HistoryProvider())
    report = validator.run(
        args.start,
        args.end,
        tiers=tuple(t.strip().upper() for t in args.tiers.split(",") if t.strip()),
        max_days=args.max_days,
    )
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return

    print(f"区间 {report.start}–{report.end}｜交易日 {report.trading_days}｜有信号日 {report.signal_days}")
    print(
        f"{'分组':<20}{'信号':>5}{'成交':>6}{'未成交':>6}{'胜率':>9}"
        f"{'均值%':>9}{'中位%':>9}{'盈亏比':>7}{'MFE%':>8}{'MAE%':>8}"
    )
    for name, stats in report.by_tier.items():
        _print_stats(f"分层 {name}", stats)
    for name, stats in report.by_type.items():
        _print_stats(f"类型 {name}", stats)
    for name, stats in report.by_stage.items():
        _print_stats(f"阶段 {name}", stats)
    if report.control:
        _print_stats("对照组", report.control)
    print("\n稳健性：", json.dumps(report.robustness, ensure_ascii=False, indent=2))
    print("\n可验证性：")
    for item, verdict, note in report.capabilities:
        print(f"  - {item}: {verdict}（{note}）")
    print("\n注意事项：")
    for caveat in report.caveats:
        print(f"  - {caveat}")


if __name__ == "__main__":
    main()
