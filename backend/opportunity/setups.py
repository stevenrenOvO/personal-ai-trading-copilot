"""Step-4 stage 2: seven setup types + the entry state machine.

Every candidate from stage 1 is judged against each setup type **independently**
(its own entry / trigger / executable / invalidation conditions). A stock gets
exactly one primary type -- chosen by specificity -- so nothing is reported
twice, while every other type it also matches is listed in ``also_matches``.

The state machine distinguishes three things that are easy to conflate:

* 符合机会逻辑      the entry conditions of the type hold (it is this setup)
* 触发条件已满足    the type's trigger conditions hold
* 当前数据足以执行  the data needed to execute actually exists

Only when all three hold is the state ``可执行``. Today's data is daily bars
plus a delayed snapshot, so:

* 打板  = 收盘确认 / 次日开盘口径, never "filled at the limit today"
* 半路  = 观察 only (needs intraday minute bars -> 未接入)
* 弱转强 = needs more than "+5% today"; the previous board failure, sector
          strength, ladder position, structure and emotion stage are all checked
* 分歧转一致 = the divergence is defined with T-and-earlier data only; the
          confirmation is measured on T+1 (no future function)

Ordering/scoring is stage 3 and is deliberately absent here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from backend.board.ladder import LadderSnapshot
from backend.data.providers.base import DataProvider
from backend.opportunity.candidates import Candidate, CandidatePool, CandidatePoolEngine

OPPORTUNITY_TYPES = (
    "首板",
    "连板接力",
    "打板",
    "弱转强",
    "分歧转一致",
    "半路",
    "低吸",
)

STATES = ("观察", "触发中", "可执行", "失效")

# For choosing which applicable setup becomes the headline row: actionable
# first, then how specific the setup is. 失效 is never chosen.
_STATE_RANK = {"可执行": 3, "触发中": 2, "观察": 1, "失效": 0}

# Environment gate straight from the accepted Step-4 design.
STAGE_ALLOWED: dict[str, tuple[str, ...]] = {
    "冰点": (),
    "退潮": (),
    "修复": ("首板", "弱转强", "低吸"),
    "发酵": ("首板", "连板接力", "弱转强", "分歧转一致"),
    "高潮": ("首板", "连板接力"),
    "分化": ("首板", "低吸"),
    "数据不足": (),
}

# Most specific first: one stock, one primary type.
TYPE_PRIORITY = (
    "弱转强",
    "分歧转一致",
    "连板接力",
    "首板",
    "打板",
    "半路",
    "低吸",
)

DIVERGENCE_AMPLITUDE = 0.06     # (high/low - 1)
DIVERGENCE_VOLUME_RATIO = 2.0
SECTOR_PERSISTENCE_BROKEN_MAX = 0.50

_NEXT_DAY = "次日开盘口径"
_INTRADAY_UNAVAILABLE = "日内不可执行（缺分钟/盘口数据）"
_OBSERVE_ONLY = "仅观察"


@dataclass(frozen=True)
class Condition:
    name: str
    passed: Optional[bool]
    verifiable: bool
    detail: str = ""

    @property
    def data_status(self) -> str:
        return "已接入" if self.verifiable else "未接入"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "verifiable": self.verifiable,
            "data_status": self.data_status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Opportunity:
    ts_code: str
    symbol: str = ""
    name: str = ""
    opportunity_type: str = ""
    sector: str = ""
    role: str = "普通"
    board_height: int = 0
    board_form: str = ""
    state: str = "观察"
    state_reason: str = ""
    data_status: str = ""
    execution_mode: str = _OBSERVE_ONLY
    environment_allowed: bool = False
    required_stage: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    entry_conditions: tuple[Condition, ...] = ()
    trigger_conditions: tuple[Condition, ...] = ()
    executable_conditions: tuple[Condition, ...] = ()
    invalidation_conditions: tuple[Condition, ...] = ()
    also_matches: tuple[str, ...] = ()
    constraints: dict[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()
    # Carried forward from stage 1 / the ladder so stage 3 can score without
    # re-deriving the market: sector context, volume, ladder state.
    sector_rank: Optional[int] = None
    sector_limit_up_count: int = 0
    sector_relative_strength: Optional[float] = None
    sector_broken_ratio: Optional[float] = None
    volume_ratio: Optional[float] = None
    promotion_rate: Optional[float] = None
    ladder_complete: bool = False
    gap_count: int = 0
    price: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts_code": self.ts_code,
            "symbol": self.symbol,
            "name": self.name,
            "opportunity_type": self.opportunity_type,
            "sector": self.sector,
            "role": self.role,
            "board_height": self.board_height,
            "level": f"{self.board_height}板" if self.board_height >= 4 else {
                0: "未涨停",
                1: "首板",
                2: "二板",
                3: "三板",
            }.get(self.board_height, ""),
            "board_form": self.board_form,
            "state": self.state,
            "state_reason": self.state_reason,
            "data_status": self.data_status,
            "execution_mode": self.execution_mode,
            "environment_allowed": self.environment_allowed,
            "required_stage": list(self.required_stage),
            "reasons": list(self.reasons),
            "entry_conditions": [c.to_dict() for c in self.entry_conditions],
            "trigger_conditions": [c.to_dict() for c in self.trigger_conditions],
            "executable_conditions": [c.to_dict() for c in self.executable_conditions],
            "invalidation_conditions": [c.to_dict() for c in self.invalidation_conditions],
            "also_matches": list(self.also_matches),
            "constraints": dict(self.constraints),
            "warnings": list(self.warnings),
            "sector_rank": self.sector_rank,
            "volume_ratio": self.volume_ratio,
            "price": self.price,
        }


@dataclass(frozen=True)
class SetupResult:
    date: str
    available: bool = True
    stage: str = ""
    stage_rule: str = ""
    pool_size: int = 0
    opportunities: tuple[Opportunity, ...] = ()
    type_counts: dict[str, int] = field(default_factory=dict)
    state_counts: dict[str, int] = field(default_factory=dict)
    observe_only_types: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "available": self.available,
            "stage": self.stage,
            "stage_rule": self.stage_rule,
            "pool_size": self.pool_size,
            "type_counts": dict(self.type_counts),
            "state_counts": dict(self.state_counts),
            "observe_only_types": list(self.observe_only_types),
            "opportunities": [o.to_dict() for o in self.opportunities],
            "notes": list(self.notes),
            "unavailable": list(self.unavailable),
        }


class _Context:
    """Everything a setup rule is allowed to look at."""

    def __init__(
        self,
        *,
        date: str,
        stage: str,
        ladder: LadderSnapshot,
        bars: pd.DataFrame,
        limits: pd.DataFrame,
        sectors: dict[str, Any],
    ) -> None:
        self.date = date
        self.stage = stage
        self.ladder = ladder
        self.bars = bars
        self.limits = limits
        self.sectors = sectors
        self.promotion_rate = self._today_promotion()
        self.gap_count = len(ladder.gaps or ())

    def _today_promotion(self) -> Optional[float]:
        for day in self.ladder.recent:
            if day.date == self.date:
                return day.promotion_rate
        return None

    def bar(self, code: str, day: Optional[str] = None) -> Optional[pd.Series]:
        if self.bars is None or self.bars.empty:
            return None
        frame = self.bars[self.bars["ts_code"].astype(str) == code]
        if frame.empty:
            return None
        frame = frame.sort_values("trade_date")
        target = day or self.date
        match = frame[frame["trade_date"].astype(str) == target]
        return None if match.empty else match.iloc[0]

    def previous_bar(self, code: str) -> Optional[pd.Series]:
        if self.bars is None or self.bars.empty:
            return None
        frame = self.bars[self.bars["ts_code"].astype(str) == code].sort_values("trade_date")
        dates = frame["trade_date"].astype(str).tolist()
        if self.date not in dates:
            return None
        index = dates.index(self.date)
        return None if index == 0 else frame.iloc[index - 1]


@dataclass
class _TypeEval:
    type: str
    matched: bool
    reasons: list[str] = field(default_factory=list)
    entry: list[Condition] = field(default_factory=list)
    trigger: list[Condition] = field(default_factory=list)
    executable: list[Condition] = field(default_factory=list)
    invalidation: list[Condition] = field(default_factory=list)
    execution_mode: str = _OBSERVE_ONLY
    data_status: str = "日线已确认"


def _f(value: Any) -> Optional[float]:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class OpportunitySetupEngine:
    """Classify stage-1 candidates into setup types and run the state machine."""

    def __init__(
        self,
        provider: DataProvider,
        *,
        pool_engine: Optional[CandidatePoolEngine] = None,
    ) -> None:
        self.provider = provider
        self.pool_engine = pool_engine or CandidatePoolEngine(provider)

    # -- public --------------------------------------------------------
    def build(
        self,
        date: str,
        *,
        pool: Optional[CandidatePool] = None,
        ladder: Optional[LadderSnapshot] = None,
        stage: Optional[str] = None,
        stage_rule: str = "",
        sectors: Optional[dict[str, Any]] = None,
    ) -> SetupResult:
        if pool is None:
            pool = self.pool_engine.build(date, stage=stage, stage_rule=stage_rule)
        if not pool.available:
            return SetupResult(
                date=date,
                available=False,
                notes=tuple(pool.notes),
                unavailable=tuple(pool.unavailable) or ("无候选池，无法识别机会类型",),
            )
        ladder = ladder or self.pool_engine.ladder_engine.snapshot(date)
        stage = stage or pool.stage or "数据不足"
        if sectors is None:
            scores = self.pool_engine.sector_engine.ranked(date)
            sectors = {s.name: s for s in scores}
        bars, limits = self.pool_engine._market_data(date, ladder)
        ctx = _Context(
            date=date,
            stage=stage,
            ladder=ladder,
            bars=bars,
            limits=limits,
            sectors=sectors,
        )

        opportunities: list[Opportunity] = []
        type_counts = {t: 0 for t in OPPORTUNITY_TYPES}
        low_suck_used = 0
        for candidate in pool.candidates:
            evaluations = self._evaluate_all(candidate, ctx)
            matched = [e for e in evaluations.values() if e.matched]
            for evaluation in matched:
                type_counts[evaluation.type] += 1
            if not matched:
                continue
            # Pick the most *actionable* applicable setup, not merely the most
            # specific one: a gate-blocked 分歧转一致 must not hide the
            # 连板接力 that the current stage actually allows.
            resolved = [
                (self._resolve_for(e, ctx), e) for e in matched
            ]
            primary = max(
                resolved,
                key=lambda pair: (
                    _STATE_RANK[pair[0][0]],
                    -TYPE_PRIORITY.index(pair[1].type),
                ),
            )[1]
            also = tuple(sorted(e.type for e in matched if e.type != primary.type))
            if primary.type == "低吸":
                if low_suck_used >= 1:
                    continue  # design: 低吸 is auxiliary, at most one per day
                low_suck_used += 1
            opportunities.append(
                self._to_opportunity(candidate, primary, ctx, also=also)
            )

        state_counts = {s: 0 for s in STATES}
        for item in opportunities:
            state_counts[item.state] = state_counts.get(item.state, 0) + 1

        observe_only = tuple(
            t for t in OPPORTUNITY_TYPES if t in ("半路",)
        )
        notes = [
            "打板在当前数据下为收盘确认/次日开盘口径，不假设当日涨停价成交",
            "半路需要分钟数据，当前只输出观察，不输出可执行",
        ]
        unavailable = [
            "竞价过程/封单量/精确涨停时间：未接入",
            "候选股分钟数据：未接入（半路与日内打板因此不可执行）",
        ]
        return SetupResult(
            date=date,
            available=True,
            stage=stage,
            stage_rule=stage_rule or pool.stage_rule,
            pool_size=pool.pool_size,
            opportunities=tuple(opportunities),
            type_counts=type_counts,
            state_counts=state_counts,
            observe_only_types=observe_only,
            notes=tuple(notes),
            unavailable=tuple(unavailable),
        )

    # -- per-type evaluation -------------------------------------------
    def _evaluate_all(self, candidate: Candidate, ctx: _Context) -> dict[str, _TypeEval]:
        return {
            "首板": self._first_board(candidate, ctx),
            "连板接力": self._relay(candidate, ctx),
            "打板": self._hit_board(candidate, ctx),
            "弱转强": self._weak_to_strong(candidate, ctx),
            "分歧转一致": self._divergence(candidate, ctx),
            "半路": self._half_way(candidate, ctx),
            "低吸": self._dip_buy(candidate, ctx),
        }

    # -- helpers -------------------------------------------------------
    def _facts(self, candidate: Candidate, ctx: _Context) -> dict[str, Any]:
        bar = ctx.bar(candidate.ts_code)
        up = _f(candidate.up_limit)
        close = _f(candidate.close)
        low = _f(bar.get("low")) if bar is not None else None
        open_price = _f(bar.get("open")) if bar is not None else None
        high = _f(bar.get("high")) if bar is not None else None
        sealed = bool(up and close and close >= up)
        one_word = bool(
            up and open_price is not None and low is not None and open_price >= up and low >= up
        )
        opened_board = bool(up and low is not None and low < up)
        amplitude = None
        if high is not None and low is not None and low > 0:
            amplitude = high / low - 1.0
        sector = ctx.sectors.get(candidate.industry)
        # A sector counts as hot on the documented criterion (>=2 limit-ups or
        # positive relative strength), not on rank alone.
        sector_hot = bool(
            candidate.sector_limit_up_count >= 2
            or (candidate.sector_relative_strength is not None and candidate.sector_relative_strength > 0)
        )
        prev_bar = ctx.previous_bar(candidate.ts_code)
        return {
            "bar": bar,
            "prev_bar": prev_bar,
            "up": up,
            "close": close,
            "low": low,
            "high": high,
            "open": open_price,
            "sealed": sealed,
            "one_word": one_word,
            "opened_board": opened_board,
            "amplitude": amplitude,
            "sector": sector,
            "sector_hot": sector_hot,
            "prev_broken": candidate.ts_code in set(ctx.ladder.prev_broken_codes),
            "touched_prev_limit": candidate.ts_code in set(ctx.ladder.prev_broken_codes),
            "height": candidate.height,
        }

    @staticmethod
    def _stage_ok(stage: str, type_name: str) -> bool:
        return type_name in STAGE_ALLOWED.get(stage, ())

    def _resolve(
        self,
        evaluation: _TypeEval,
        ctx: _Context,
        *,
        executable_allowed: bool,
        block_reason: str = "",
    ) -> tuple[str, str]:
        """观察 / 触发中 / 可执行 / 失效, with the reason stated."""
        failed_invalidation = [c for c in evaluation.invalidation if c.passed is True]
        if failed_invalidation:
            return "失效", "失效条件成立：" + "；".join(c.name for c in failed_invalidation)
        if not self._stage_ok(ctx.stage, evaluation.type):
            return "观察", f"当前情绪阶段『{ctx.stage}』不允许该类型机会（环境闸门）"
        if not all(c.passed is True for c in evaluation.entry):
            pending = [c.name for c in evaluation.entry if c.passed is not True]
            return "观察", "进入条件未满足：" + "；".join(pending)
        pending_trigger = [c for c in evaluation.trigger if c.passed is not True]
        passed_trigger = [c for c in evaluation.trigger if c.passed is True]
        if pending_trigger:
            state = "触发中" if passed_trigger else "观察"
            return state, "等待触发：" + "；".join(c.name for c in pending_trigger)
        if not executable_allowed:
            return "观察", block_reason or "当前数据不足以执行"
        unmet = [c for c in evaluation.executable if c.passed is not True]
        if unmet:
            return "触发中", "可执行条件未满足：" + "；".join(c.name for c in unmet)
        return "可执行", "进入、触发与可执行条件全部成立"

    def _resolve_for(self, evaluation: _TypeEval, ctx: _Context) -> tuple[str, str]:
        """State machine entry point for one setup type (single source of truth)."""
        executable_allowed = evaluation.execution_mode == _NEXT_DAY
        block_reason = {
            _INTRADAY_UNAVAILABLE: "当前无盘口/分钟数据，无法验证日内成交，不输出可执行",
            _OBSERVE_ONLY: "该类型在当前数据下只能观察，不输出可执行",
        }.get(evaluation.execution_mode, "")
        return self._resolve(
            evaluation,
            ctx,
            executable_allowed=executable_allowed,
            block_reason=block_reason,
        )

    # -- the seven setups ----------------------------------------------
    def _first_board(self, c: Candidate, ctx: _Context) -> _TypeEval:
        f = self._facts(c, ctx)
        evaluation = _TypeEval(
            type="首板",
            matched=bool(f["height"] == 1 and f["sealed"]),
            execution_mode=_NEXT_DAY,
        )
        if not evaluation.matched:
            return evaluation
        evaluation.reasons = [
            f"首板结构（{c.board_form}），板块『{c.industry}』当日涨停 {c.sector_limit_up_count} 家",
            f"板块强度排名第 {c.sector_rank if c.sector_rank else '未入榜'}，"
            f"相对强度 {'—' if c.sector_relative_strength is None else f'{c.sector_relative_strength:+.2f}'}",
        ]
        evaluation.entry = [
            Condition("连板高度 = 1（首板）", True, True),
            Condition("收盘封板（收盘 ≥ 涨停价）", f["sealed"], True, f"收盘 {f['close']} / 涨停 {f['up']}"),
        ]
        evaluation.trigger = [
            Condition(
                "板块效应成立（板块涨停 ≥2 家 或 相对强度 >0）",
                f["sector_hot"],
                True,
                f"板块涨停 {c.sector_limit_up_count} 家",
            ),
            Condition(
                "个股量能放大",
                None if c.volume_ratio is None else c.volume_ratio >= 1.2,
                c.volume_ratio is not None,
                f"量比 {c.volume_ratio}" if c.volume_ratio is not None else "量能基线不足",
            ),
            Condition("收盘仍在涨停价（未炸板）", f["sealed"], True),
        ]
        evaluation.executable = [
            Condition("非一字板（次日仍有买入机会）", not f["one_word"], True, c.board_form),
            Condition("情绪阶段允许首板", self._stage_ok(ctx.stage, "首板"), True, ctx.stage),
        ]
        evaluation.invalidation = [
            Condition("跌破首板当日最低价", None, False, f"需 T+1 数据（首板最低 {f['low']}）"),
            Condition("次日板块无跟随（板块涨停 <2 家）", None, False, "需 T+1 数据"),
        ]
        return evaluation

    def _relay(self, c: Candidate, ctx: _Context) -> _TypeEval:
        f = self._facts(c, ctx)
        evaluation = _TypeEval(
            type="连板接力",
            matched=bool(2 <= f["height"] <= 4 and f["sealed"]),
            execution_mode=_NEXT_DAY,
        )
        if not evaluation.matched:
            return evaluation
        promotion = ctx.promotion_rate
        evaluation.reasons = [
            f"当前 {c.height} 板，角色『{c.structural_role}』",
            f"板块『{c.industry}』涨停 {c.sector_limit_up_count} 家",
        ]
        evaluation.entry = [
            Condition("连板高度 2–4 板（V1.0 上限）", 2 <= f["height"] <= 4, True, f"{c.height} 板"),
            Condition("收盘封板", f["sealed"], True, c.board_form),
        ]
        evaluation.trigger = [
            Condition(
                "板块效应成立或本股为龙头/中军",
                bool(f["sector_hot"] or c.structural_role in ("龙头候选", "中军候选")),
                True,
                c.structural_role,
            ),
            Condition("梯队未被断层孤立", ctx.gap_count <= 1, True, f"断层 {ctx.gap_count} 处"),
            Condition(
                "昨日涨停晋级率 ≥25%",
                None if promotion is None else promotion >= 0.25,
                promotion is not None,
                f"{promotion:.1%}" if promotion is not None else "未接入",
            ),
            Condition("高度风险可控（≤4 板）", True, True, f"{c.height} 板"),
        ]
        evaluation.executable = [
            Condition("情绪阶段为发酵/高潮（接力型）", ctx.stage in ("发酵", "高潮"), True, ctx.stage),
            Condition("非一字板", not f["one_word"], True, c.board_form),
        ]
        evaluation.invalidation = [
            Condition("今日炸板", False, True),
            Condition(
                "晋级率跌破 20%",
                None if promotion is None else promotion < 0.20,
                promotion is not None,
                f"{promotion:.1%}" if promotion is not None else "未接入",
            ),
        ]
        return evaluation

    def _hit_board(self, c: Candidate, ctx: _Context) -> _TypeEval:
        """打板: V1.0 can only confirm at the close / plan for the next open."""
        f = self._facts(c, ctx)
        evaluation = _TypeEval(
            type="打板",
            matched=bool(f["sealed"] and f["opened_board"]),
            execution_mode=_INTRADAY_UNAVAILABLE,
            data_status="日内成交无法验证（无盘口/分钟数据）",
        )
        if not evaluation.matched:
            return evaluation
        evaluation.reasons = [
            f"当日开过板后回封（{c.board_form}），说明盘中存在成交机会",
            "V1.0 无盘口数据，无法验证是否能在涨停价成交",
        ]
        evaluation.entry = [
            Condition("收盘封板", f["sealed"], True),
            Condition("盘中开过板（低点 < 涨停价）", f["opened_board"], True),
        ]
        evaluation.trigger = [
            Condition("板块效应成立", f["sector_hot"], True, f"板块涨停 {c.sector_limit_up_count} 家"),
            Condition("回封时间与封单量", None, False, "未接入（需盘口/分钟数据）"),
        ]
        evaluation.executable = [
            Condition("能否在涨停价成交", None, False, "未接入：不假设涨停价成交"),
        ]
        evaluation.invalidation = [
            Condition("尾盘炸板", not f["sealed"], True),
            Condition("一字板（不可能成交）", f["one_word"], True),
        ]
        return evaluation

    def _weak_to_strong(self, c: Candidate, ctx: _Context) -> _TypeEval:
        """Beyond '+5% today': verify the previous failure, strength, sector,
        ladder position, structure and the emotion stage."""
        f = self._facts(c, ctx)
        today_pct = c.pct_chg
        strong_today = today_pct is not None and today_pct >= 5.0
        evaluation = _TypeEval(
            type="弱转强",
            matched=bool(f["prev_broken"] and strong_today),
            execution_mode=_NEXT_DAY if f["sealed"] else _OBSERVE_ONLY,
        )
        if not evaluation.matched:
            return evaluation
        prev_low = _f(f["prev_bar"].get("low")) if f["prev_bar"] is not None else None
        structure_ok = bool(
            prev_low is None or (c.close is not None and c.close >= prev_low)
        )
        evaluation.reasons = [
            "昨日炸板/触板后今日转强（候选入口条件已满足，需继续验证）",
            f"今日 {today_pct:+.2f}%、{'封板' if f['sealed'] else '未封板'}",
            f"板块『{c.industry}』涨停 {c.sector_limit_up_count} 家",
        ]
        evaluation.entry = [
            Condition("昨日确实炸板/触板未封", f["prev_broken"], True),
            Condition("今日明显转强（涨幅 ≥5%）", strong_today, True, f"{today_pct:+.2f}%"),
        ]
        evaluation.trigger = [
            Condition("今日封板（转强结果确认）", f["sealed"], True, c.board_form),
            Condition("所属板块仍然强", f["sector_hot"], True, f"板块涨停 {c.sector_limit_up_count} 家"),
            Condition(
                "梯队位置合理（非被更高板压制）",
                c.structural_role in ("龙头候选", "中军候选", "补涨候选", "普通"),
                True,
                c.structural_role,
            ),
            Condition("今日结构未破坏（未跌破昨日最低）", structure_ok, True, f"昨日最低 {prev_low}"),
            Condition(
                "情绪周期允许弱转强（修复/发酵）",
                ctx.stage in ("修复", "发酵"),
                True,
                ctx.stage,
            ),
        ]
        evaluation.executable = [
            Condition("非一字板", not f["one_word"], True, c.board_form),
        ]
        evaluation.invalidation = [
            Condition("今日未封板", not f["sealed"], True),
            Condition("跌破昨日最低价", not structure_ok, True),
            Condition("板块转弱（涨停 <2 家且相对强度 ≤0）", not f["sector_hot"], True),
        ]
        return evaluation

    def _divergence(self, c: Candidate, ctx: _Context) -> _TypeEval:
        """T 日分歧（只用 T 及以前数据）→ 一致结果需 T+1 确认。"""
        f = self._facts(c, ctx)
        diverged = bool(
            f["height"] >= 2
            and f["sealed"]
            and (
                (f["amplitude"] is not None and f["amplitude"] >= DIVERGENCE_AMPLITUDE)
                or f["opened_board"]
                or (c.volume_ratio is not None and c.volume_ratio >= DIVERGENCE_VOLUME_RATIO)
            )
        )
        evaluation = _TypeEval(
            type="分歧转一致",
            matched=diverged,
            execution_mode=_OBSERVE_ONLY,
            data_status="需 T+1 数据确认一致性",
        )
        if not evaluation.matched:
            return evaluation
        evaluation.reasons = [
            f"T 日分歧结构：振幅 {'—' if f['amplitude'] is None else f'{f["amplitude"]:.1%}'}、"
            f"{'开过板' if f['opened_board'] else '未开板'}、量比 {c.volume_ratio}",
            "一致性结果只能由 T+1 确认，T 日不使用未来数据",
        ]
        evaluation.entry = [
            Condition("连板高度 ≥2", f["height"] >= 2, True, f"{c.height} 板"),
            Condition(
                "T 日出现分歧（大振幅/开板/放量）",
                diverged,
                True,
                f"振幅 {f['amplitude']}" if f["amplitude"] is not None else "未接入",
            ),
        ]
        evaluation.trigger = [
            Condition("T+1 一致性上板", None, False, "需 T+1 数据（不可用未来数据判断）"),
            Condition("板块仍强", f["sector_hot"], True, f"板块涨停 {c.sector_limit_up_count} 家"),
        ]
        evaluation.executable = [
            Condition("T+1 确认后执行", None, False, "T+1 数据未产生"),
        ]
        evaluation.invalidation = [
            Condition("T+1 炸板或大跌", None, False, "需 T+1 数据"),
        ]
        return evaluation

    def _half_way(self, c: Candidate, ctx: _Context) -> _TypeEval:
        f = self._facts(c, ctx)
        pct = c.pct_chg
        matched = bool(
            not f["sealed"]
            and pct is not None
            and pct >= 3.0
            and f["sector_hot"]
            and not f["prev_broken"]
        )
        evaluation = _TypeEval(
            type="半路",
            matched=matched,
            execution_mode=_INTRADAY_UNAVAILABLE,
            data_status="需分钟数据（未接入）",
        )
        if not evaluation.matched:
            return evaluation
        evaluation.reasons = [
            f"未封板但当日 {pct:+.2f}%，位于热点板块『{c.industry}』",
            "半路买点依赖分时结构，当前无分钟数据",
        ]
        evaluation.entry = [
            Condition("未封板但涨幅 ≥3%", True, True, f"{pct:+.2f}%"),
            Condition("属于热点板块", f["sector_hot"], True),
        ]
        evaluation.trigger = [
            Condition("站上前高/分时放量上攻", None, False, "未接入（需分钟数据）"),
            Condition("回调不破均价线", None, False, "未接入（需分钟数据）"),
        ]
        evaluation.executable = [
            Condition("分时确认", None, False, "未接入（需分钟数据）"),
        ]
        evaluation.invalidation = [
            Condition("回落跌破当日均价", None, False, "未接入"),
            Condition("板块转弱", not f["sector_hot"], True),
        ]
        return evaluation

    def _dip_buy(self, c: Candidate, ctx: _Context) -> _TypeEval:
        """低吸：辅助类型，仅在发酵/修复、龙头/中军、板块仍有持续性时允许。"""
        f = self._facts(c, ctx)
        pct = c.pct_chg
        leader = c.structural_role in ("龙头候选", "中军候选")
        sector_persistent = bool(
            c.sector_limit_up_count >= 2
            and (c.sector_broken_ratio is None or c.sector_broken_ratio < SECTOR_PERSISTENCE_BROKEN_MAX)
        )
        matched = bool(
            leader
            and not f["sealed"]
            and pct is not None
            and pct <= 3.0
            and f["sector_hot"]
            and ctx.stage in ("发酵", "修复")
        )
        evaluation = _TypeEval(
            type="低吸",
            matched=matched,
            execution_mode=_NEXT_DAY,
        )
        if not evaluation.matched:
            return evaluation
        prev_low = self._previous_limit_up_low(c, ctx)
        structure_ok = bool(prev_low is None or (c.close is not None and c.close >= prev_low))
        evaluation.reasons = [
            f"角色『{c.structural_role}』，阶段『{ctx.stage}』，属于辅助型低吸机会",
            f"当日 {pct:+.2f}%，未封板（回踩形态）",
        ]
        evaluation.entry = [
            Condition("角色为龙头/中军", leader, True, c.structural_role),
            Condition("情绪阶段为发酵/修复", ctx.stage in ("发酵", "修复"), True, ctx.stage),
            Condition("板块仍有持续性", sector_persistent, True, f"板块炸板率 {c.sector_broken_ratio}"),
        ]
        evaluation.trigger = [
            Condition("未封板且涨幅 ≤3%（回踩）", True, True, f"{pct:+.2f}%"),
            Condition("个股结构未破坏", structure_ok, True, f"上一涨停日最低 {prev_low}"),
        ]
        evaluation.executable = [
            Condition("未跌破结构位（可执行前提）", structure_ok, True),
        ]
        evaluation.invalidation = [
            Condition("跌破结构位", not structure_ok, True),
            Condition("板块掉出热度前列", not f["sector_hot"], True),
        ]
        return evaluation

    def _previous_limit_up_low(self, c: Candidate, ctx: _Context) -> Optional[float]:
        try:
            return self.pool_engine._previous_limit_up_low(ctx.bars, c.ts_code, ctx.date)
        except Exception:  # noqa: BLE001
            return None

    # -- output --------------------------------------------------------
    def _to_opportunity(
        self,
        candidate: Candidate,
        evaluation: _TypeEval,
        ctx: _Context,
        *,
        also: tuple[str, ...],
    ) -> Opportunity:
        state, reason = self._resolve_for(evaluation, ctx)
        warnings = list(candidate.warnings)
        if evaluation.execution_mode == _INTRADAY_UNAVAILABLE:
            warnings.append("日内成交不可验证：不得假设在涨停价成交")
        return Opportunity(
            ts_code=candidate.ts_code,
            symbol=candidate.symbol,
            name=candidate.name,
            opportunity_type=evaluation.type,
            sector=candidate.industry,
            role=candidate.structural_role,
            board_height=candidate.height,
            board_form=candidate.board_form,
            state=state,
            state_reason=reason,
            data_status=evaluation.data_status,
            execution_mode=evaluation.execution_mode,
            environment_allowed=self._stage_ok(ctx.stage, evaluation.type),
            required_stage=tuple(
                s for s, types in STAGE_ALLOWED.items() if evaluation.type in types
            ),
            reasons=tuple(evaluation.reasons),
            entry_conditions=tuple(evaluation.entry),
            trigger_conditions=tuple(evaluation.trigger),
            executable_conditions=tuple(evaluation.executable),
            invalidation_conditions=tuple(evaluation.invalidation),
            also_matches=also,
            constraints={
                "t_plus_one": True,
                "lot_size": 100,
                "limit_up_cannot_buy": True,
                "one_word_cannot_fill": True,
                "fill_assumption": (
                    "次日开盘买入（不假设涨停价成交）"
                    if evaluation.execution_mode == _NEXT_DAY
                    else "不可执行：缺少日内成交验证数据"
                ),
                # The T+1 open is unknowable at T, so it is an execution-time
                # rule, not an executable-condition we pretend to have checked.
                "execution_note": (
                    "执行时若次日开盘已在涨停价，则视为无法买入（不假设成交）"
                    if evaluation.execution_mode == _NEXT_DAY
                    else ""
                ),
            },
            warnings=tuple(warnings),
            sector_rank=candidate.sector_rank,
            sector_limit_up_count=candidate.sector_limit_up_count,
            sector_relative_strength=candidate.sector_relative_strength,
            sector_broken_ratio=candidate.sector_broken_ratio,
            volume_ratio=candidate.volume_ratio,
            promotion_rate=ctx.promotion_rate,
            ladder_complete=bool(ctx.gap_count == 0),
            gap_count=ctx.gap_count,
            price=candidate.close,
        )
