"""Step-4 stage 3: decomposable ranking, risk gating and position sizing.

Stage 2 can report 50+ "可执行" rows because a close-confirmed setup is
executable at the next open. That is not 50 buy signals -- this stage turns
them into a prioritised, capped, risk-checked plan.

The score is deliberately *not* a mystery: it is the sum of six published
blocks, each of which lists every item that contributed points.

    环境 20 + 板块 25 + 梯队 20 + 个股 20 + 买点 15 − 风险扣分 30

The score only orders candidates. It is not a probability and not a buy
instruction -- ``why_not_buy`` is attached to every row, including the A tier.

Environment first: 冰点/退潮 can never produce an A-tier offensive plan, no
matter how high a name scores. Missing data lowers confidence and adds risk
deduction; it is never treated as "passed" and never silently scored as zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
from typing import Any, Optional

from backend.data.providers.base import DataProvider
from backend.opportunity.setups import (
    STATES,
    STAGE_ALLOWED,
    Opportunity,
    OpportunitySetupEngine,
    SetupResult,
)

# ---- score blocks (fixed by the accepted design; do not tune per day) ----
STAGE_ENV_POINTS = {"发酵": 15, "高潮": 12, "修复": 10, "分化": 6, "退潮": 2, "冰点": 0}
TYPE_ALLOWED_POINTS = 5
BLOCK_MAX = {"环境": 20, "板块": 25, "梯队": 20, "个股": 20, "买点": 15}
RISK_MAX = 30

TIER_A_MIN_SCORE = 70

# "重点关注" is a limited slot count, not a分数线: on a strong day dozens of
# names clear every gate, and showing them all as A is the exact "52 可执行 =
# 52 推荐" failure this stage exists to prevent.
A_TIER_LIMIT = {"发酵": 3, "高潮": 3, "修复": 2, "分化": 0, "退潮": 0, "冰点": 0}

MAX_NEW_POSITIONS = {"发酵": 2, "高潮": 1, "修复": 1, "分化": 1, "退潮": 0, "冰点": 0}
STAGE_EXPOSURE_CAP = {
    "发酵": 0.60,
    "高潮": 0.50,
    "修复": 0.30,
    "分化": 0.30,
    "退潮": 0.10,
    "冰点": 0.10,
    "数据不足": 0.0,
}
TYPE_POSITION_FACTOR = {
    "首板": 1.0,
    "连板接力": 0.75,
    "弱转强": 0.8,
    "低吸": 0.6,
    "打板": 0.0,
    "半路": 0.0,
    "分歧转一致": 0.0,
}
ATTACK_STAGES = ("修复", "发酵", "高潮")

LOT_SIZE = 100


@dataclass(frozen=True)
class ScoreItem:
    name: str
    points: float
    max_points: float
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "points": round(self.points, 2),
            "max_points": self.max_points,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ScoreBlock:
    name: str
    points: float
    max_points: float
    items: tuple[ScoreItem, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "points": round(self.points, 2),
            "max_points": self.max_points,
            "items": [i.to_dict() for i in self.items],
        }


@dataclass(frozen=True)
class PositionPlan:
    suggested_weight: float = 0.0
    shares: int = 0
    lots: int = 0
    amount: float = 0.0
    affordable: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "suggested_weight": round(self.suggested_weight, 4),
            "shares": self.shares,
            "lots": self.lots,
            "amount": round(self.amount, 2),
            "affordable": self.affordable,
            "note": self.note,
        }


@dataclass(frozen=True)
class TradePlan:
    ts_code: str
    name: str = ""
    opportunity_type: str = ""
    tier: str = "C"
    rank: int = 0
    score: float = 0.0
    state: str = "观察"
    sector: str = ""
    role: str = "普通"
    board_height: int = 0
    blocks: tuple[ScoreBlock, ...] = ()
    risk_deduction: float = 0.0
    risk_items: tuple[ScoreItem, ...] = ()
    risk_level: str = "medium"
    reasons: tuple[str, ...] = ()
    trigger_conditions: tuple[dict[str, Any], ...] = ()
    invalidation_conditions: tuple[dict[str, Any], ...] = ()
    position: PositionPlan = PositionPlan()
    why_not_buy: str = ""
    data_freshness: dict[str, Any] = field(default_factory=dict)
    missing_data: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    price: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ts_code": self.ts_code,
            "name": self.name,
            "opportunity_type": self.opportunity_type,
            "tier": self.tier,
            "rank": self.rank,
            "score": round(self.score, 2),
            "score_breakdown": {
                "blocks": [b.to_dict() for b in self.blocks],
                "risk_deduction": round(self.risk_deduction, 2),
                "risk_items": [i.to_dict() for i in self.risk_items],
                "total": round(self.score, 2),
            },
            "state": self.state,
            "sector": self.sector,
            "role": self.role,
            "board_height": self.board_height,
            "risk_level": self.risk_level,
            "reasons": list(self.reasons),
            "trigger": list(self.trigger_conditions),
            "invalidation": list(self.invalidation_conditions),
            "position": self.position.to_dict(),
            "why_not_buy": self.why_not_buy,
            "data_freshness": dict(self.data_freshness),
            "missing_data": list(self.missing_data),
            "warnings": list(self.warnings),
            "price": self.price,
        }


@dataclass(frozen=True)
class RankedPlan:
    date: str
    available: bool = True
    stage: str = ""
    stage_rule: str = ""
    can_attack: bool = False
    attack_note: str = ""
    total_exposure_cap: float = 0.0
    available_exposure: float = 0.0
    current_exposure: float = 0.0
    max_new_positions: int = 0
    high_quality_count: int = 0
    tier_counts: dict[str, int] = field(default_factory=dict)
    plans: tuple[TradePlan, ...] = ()
    notes: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "available": self.available,
            "stage": self.stage,
            "stage_rule": self.stage_rule,
            "can_attack": self.can_attack,
            "attack_note": self.attack_note,
            "total_exposure_cap": round(self.total_exposure_cap, 4),
            "available_exposure": round(self.available_exposure, 4),
            "current_exposure": round(self.current_exposure, 4),
            "max_new_positions": self.max_new_positions,
            "high_quality_count": self.high_quality_count,
            "tier_counts": dict(self.tier_counts),
            "plans": [p.to_dict() for p in self.plans],
            "notes": list(self.notes),
            "unavailable": list(self.unavailable),
        }


class OpportunityRankingEngine:
    """Rank stage-2 opportunities, apply risk and size the positions."""

    def __init__(
        self,
        provider: DataProvider,
        *,
        setup_engine: Optional[OpportunitySetupEngine] = None,
        risk_manager: Any = None,
        max_position_weight: float = 0.20,
    ) -> None:
        self.provider = provider
        self.setup_engine = setup_engine or OpportunitySetupEngine(provider)
        self.max_position_weight = max_position_weight
        if risk_manager is None:
            try:
                from backend.risk.rules import RiskManager

                risk_manager = RiskManager()
            except Exception:  # noqa: BLE001
                risk_manager = None
        self.risk_manager = risk_manager

    # -- public --------------------------------------------------------
    def build(
        self,
        date: str,
        *,
        setups: Optional[SetupResult] = None,
        state: Optional[dict[str, Any]] = None,
    ) -> RankedPlan:
        setups = setups or self.setup_engine.build(date)
        if not setups.available:
            return RankedPlan(
                date=date,
                available=False,
                stage=setups.stage,
                notes=tuple(setups.notes),
                unavailable=tuple(setups.unavailable) or ("无机会识别结果，无法排序",),
            )
        stage = setups.stage or "数据不足"
        state = state or {}
        equity = float(state.get("equity") or 1_000_000.0)
        cash = float(state.get("cash") or equity)
        positions: dict[str, Any] = dict(state.get("positions") or {})
        prices: dict[str, float] = dict(state.get("prices") or {})

        current_exposure = 0.0
        if equity > 0:
            current_exposure = max(
                0.0, (equity - cash) / equity
            )
        cap = STAGE_EXPOSURE_CAP.get(stage, 0.0)
        available_exposure = max(0.0, cap - current_exposure)
        max_new = MAX_NEW_POSITIONS.get(stage, 0)
        if available_exposure <= 0:
            max_new = 0
        if self.max_position_weight:
            max_new = min(max_new, max(0, int(cap / max(self.max_position_weight, 1e-9))))

        scored = [self._score(o, stage, positions, prices) for o in setups.opportunities]
        scored.sort(key=lambda p: (-p.score, p.ts_code))
        scored = self._cap_tier_a(scored, stage)
        plans = self._assign_positions(
            scored,
            stage=stage,
            equity=equity,
            cash=cash,
            available_exposure=available_exposure,
            max_new=max_new,
        )
        ranked = tuple(
            TradePlan(**{**plan.__dict__, "rank": index})
            for index, plan in enumerate(plans, start=1)
        )
        tier_counts = {"A": 0, "B": 0, "C": 0}
        for plan in ranked:
            tier_counts[plan.tier] = tier_counts.get(plan.tier, 0) + 1

        can_attack = stage in ATTACK_STAGES and tier_counts["A"] > 0
        if not can_attack:
            if stage in ("冰点", "退潮"):
                note = f"情绪阶段『{stage}』禁止进攻型机会，建议等待"
            elif tier_counts["A"] == 0:
                note = "今日无高质量机会，建议等待"
            else:
                note = f"情绪阶段『{stage}』不建议主动进攻"
        else:
            note = f"情绪阶段『{stage}』可进攻，最多新开 {max_new} 只，总仓位上限 {cap:.0%}"

        notes = list(setups.notes)
        notes.append("排序分只代表优先级，不代表上涨概率，也不等于立即买入")
        if stage == "高潮":
            notes.append("高潮阶段：可进攻但需防范随后分化，仓位不放大")
        if current_exposure:
            notes.append(f"当前已有仓位约 {current_exposure:.1%}，可用空间 {available_exposure:.1%}")

        return RankedPlan(
            date=date,
            available=True,
            stage=stage,
            stage_rule=setups.stage_rule,
            can_attack=can_attack,
            attack_note=note,
            total_exposure_cap=cap,
            available_exposure=available_exposure,
            current_exposure=current_exposure,
            max_new_positions=max_new,
            high_quality_count=tier_counts["A"],
            tier_counts=tier_counts,
            plans=ranked,
            notes=tuple(notes),
            unavailable=tuple(setups.unavailable),
        )

    # -- scoring -------------------------------------------------------
    def _score(
        self,
        opportunity: Opportunity,
        stage: str,
        positions: dict[str, Any],
        prices: dict[str, float],
    ) -> TradePlan:
        allowed = opportunity.opportunity_type in STAGE_ALLOWED.get(stage, ())
        env_items = [
            ScoreItem("情绪阶段适配", STAGE_ENV_POINTS.get(stage, 0), 15, f"阶段『{stage}』"),
            ScoreItem(
                "类型被当前阶段允许",
                TYPE_ALLOWED_POINTS if allowed else 0,
                TYPE_ALLOWED_POINTS,
                f"{opportunity.opportunity_type} → {'允许' if allowed else '不允许'}",
            ),
        ]
        env = ScoreBlock("环境", sum(i.points for i in env_items), BLOCK_MAX["环境"], tuple(env_items))

        rank = self._sector_rank(opportunity)
        limit_ups = self._sector_limit_ups(opportunity)
        rel = self._sector_relative(opportunity)
        broken = self._sector_broken(opportunity)
        sector_items = [
            ScoreItem("板块热度排名", self._rank_points(rank), 10, f"排名 {rank or '未入榜'}"),
            ScoreItem("板块涨停家数", self._limit_up_points(limit_ups), 8, f"{limit_ups} 家"),
            ScoreItem(
                "板块相对强度",
                4 if (rel is not None and rel > 2) else (2 if (rel is not None and rel > 0) else 0),
                4,
                f"{'—' if rel is None else f'{rel:+.2f}'}",
            ),
            ScoreItem(
                "板块封板质量",
                3 if (broken is not None and broken < 0.30) else (1 if (broken is not None and broken < 0.50) else 0),
                3,
                f"板块炸板率 {'—' if broken is None else f'{broken:.0%}'}",
            ),
        ]
        sector_block = ScoreBlock(
            "板块", sum(i.points for i in sector_items), BLOCK_MAX["板块"], tuple(sector_items)
        )

        promotion = self._promotion(opportunity)
        height = opportunity.board_height
        ladder_items = [
            ScoreItem(
                "板数健康度",
                {0: 0, 1: 5, 2: 8, 3: 8, 4: 6}.get(height, 2),
                8,
                f"{height} 板",
            ),
            ScoreItem(
                "梯队完整度",
                5 if self._ladder_complete(opportunity) else 3,
                5,
                "完整" if self._ladder_complete(opportunity) else "有断层",
            ),
            ScoreItem(
                "昨日晋级率",
                7 if (promotion or 0) >= 0.35 else (5 if (promotion or 0) >= 0.25 else (2 if (promotion or 0) >= 0.15 else 0)),
                7,
                "—" if promotion is None else f"{promotion:.1%}",
            ),
        ]
        ladder_block = ScoreBlock(
            "梯队", sum(i.points for i in ladder_items), BLOCK_MAX["梯队"], tuple(ladder_items)
        )

        form = opportunity.board_form
        form_points = {"一字板/未开板": 8, "开盘封板": 7, "盘中触板回封": 5, "封板": 6}.get(form, 2)
        volume_ratio = self._volume_ratio(opportunity)
        role_points = {"龙头候选": 6, "中军候选": 5, "补涨候选": 3}.get(opportunity.role, 1)
        stock_items = [
            ScoreItem("封板形态", form_points, 8, form or "未接入"),
            ScoreItem(
                "量能放大",
                6 if (volume_ratio or 0) >= 2 else (4 if (volume_ratio or 0) >= 1.2 else 1),
                6,
                "—" if volume_ratio is None else f"{volume_ratio:.2f}x",
            ),
            ScoreItem("板块内角色", role_points, 6, opportunity.role),
        ]
        stock_block = ScoreBlock(
            "个股", sum(i.points for i in stock_items), BLOCK_MAX["个股"], tuple(stock_items)
        )

        state_points = {"可执行": 8, "触发中": 5, "观察": 2, "失效": 0}.get(opportunity.state, 0)
        mode_points = 4 if opportunity.execution_mode == "次日开盘口径" else 0
        one_word = opportunity.board_form == "一字板/未开板"
        entry_items = [
            ScoreItem("状态机位置", state_points, 8, opportunity.state),
            ScoreItem("执行口径", mode_points, 4, opportunity.execution_mode),
            ScoreItem("可成交性", 0 if one_word else 3, 3, "一字板不可买入" if one_word else "可买入"),
        ]
        entry_block = ScoreBlock(
            "买点", sum(i.points for i in entry_items), BLOCK_MAX["买点"], tuple(entry_items)
        )

        risk_items, risk_deduction = self._risk_items(opportunity, stage, one_word)
        total = (
            env.points + sector_block.points + ladder_block.points
            + stock_block.points + entry_block.points - risk_deduction
        )
        total = max(0.0, min(100.0, total))

        tier, why_not_buy, risk_level = self._tier(
            opportunity, stage, allowed, total, risk_items, risk_deduction
        )
        return TradePlan(
            ts_code=opportunity.ts_code,
            name=opportunity.name,
            opportunity_type=opportunity.opportunity_type,
            tier=tier,
            score=total,
            state=opportunity.state,
            sector=opportunity.sector,
            role=opportunity.role,
            board_height=opportunity.board_height,
            blocks=(env, sector_block, ladder_block, stock_block, entry_block),
            risk_deduction=risk_deduction,
            risk_items=tuple(risk_items),
            risk_level=risk_level,
            reasons=opportunity.reasons,
            trigger_conditions=tuple(
                c.to_dict() for c in opportunity.trigger_conditions
            ),
            invalidation_conditions=tuple(
                c.to_dict() for c in opportunity.invalidation_conditions
            ),
            why_not_buy=why_not_buy,
            data_freshness=dict(opportunity.constraints),
            missing_data=tuple(opportunity.warnings),
            warnings=tuple(opportunity.warnings),
            price=opportunity.price,
        )

    # -- scoring helpers ----------------------------------------------
    def _cap_tier_a(self, plans: list[TradePlan], stage: str) -> list[TradePlan]:
        """Keep at most ``A_TIER_LIMIT[stage]`` names in the A tier.

        Everything else that still passed every gate stays B with an explicit
        reason, so nothing is hidden and nothing is silently recommended.
        """
        limit = A_TIER_LIMIT.get(stage, 0)
        out: list[TradePlan] = []
        kept = 0
        for plan in plans:
            if plan.tier != "A":
                out.append(plan)
                continue
            if kept < limit:
                kept += 1
                out.append(plan)
                continue
            reason = (
                f"排序分 {plan.score:.1f} 未进入当日重点关注（{stage}阶段 A 级上限 {limit} 个），"
                "条件仍成立但优先级靠后"
            )
            out.append(
                TradePlan(
                    **{
                        **plan.__dict__,
                        "tier": "B",
                        "why_not_buy": (plan.why_not_buy + "；" + reason)
                        if plan.why_not_buy
                        else reason,
                    }
                )
            )
        return out

    @staticmethod
    def _sector_rank(opportunity: Opportunity) -> Optional[int]:
        return opportunity.sector_rank

    @staticmethod
    def _sector_limit_ups(opportunity: Opportunity) -> int:
        return int(opportunity.sector_limit_up_count or 0)

    @staticmethod
    def _sector_relative(opportunity: Opportunity) -> Optional[float]:
        return opportunity.sector_relative_strength

    @staticmethod
    def _sector_broken(opportunity: Opportunity) -> Optional[float]:
        return opportunity.sector_broken_ratio

    @staticmethod
    def _promotion(opportunity: Opportunity) -> Optional[float]:
        return opportunity.promotion_rate

    @staticmethod
    def _volume_ratio(opportunity: Opportunity) -> Optional[float]:
        return opportunity.volume_ratio

    @staticmethod
    def _ladder_complete(opportunity: Opportunity) -> bool:
        return bool(opportunity.ladder_complete)

    @staticmethod
    def _rank_points(rank: Optional[int]) -> float:
        if rank is None:
            return 2
        if rank <= 3:
            return 10
        if rank <= 10:
            return 6
        return 2

    @staticmethod
    def _limit_up_points(count: int) -> float:
        if count >= 5:
            return 8
        if count >= 2:
            return 5
        return 1

    def _risk_items(
        self, opportunity: Opportunity, stage: str, one_word: bool
    ) -> tuple[list[ScoreItem], float]:
        """Risk deduction. Missing data is a deduction, never a free pass."""
        items: list[ScoreItem] = []

        missing: list[str] = []
        for warning in opportunity.warnings:
            if "未接入" in warning or "未验证" in warning:
                missing.append(warning)
        for label in ("分钟数据", "竞价", "封单量", "涨停时间"):
            if any(label in w for w in missing):
                items.append(ScoreItem(f"数据缺失：{label}", -3, 3, "未接入"))
        if any("成交额" in w for w in missing):
            items.append(ScoreItem("流动性未验证（成交额缺失）", -3, 3, "未接入"))
        if opportunity.execution_mode == "日内不可执行（缺分钟/盘口数据）":
            items.append(ScoreItem("今日不可执行（缺日内数据）", -5, 5, opportunity.execution_mode))
        if opportunity.state != "可执行":
            items.append(ScoreItem(f"状态未达可执行（{opportunity.state}）", -4, 4, opportunity.state_reason[:60]))
        if stage in ("冰点", "退潮"):
            items.append(ScoreItem(f"情绪阶段风险（{stage}）", -8, 8, "禁止进攻型"))
        elif stage == "分化":
            items.append(ScoreItem("情绪阶段风险（分化）", -4, 4, "降低仓位、提高标准"))
        elif stage == "高潮":
            items.append(ScoreItem("情绪阶段风险（高潮后分化）", -2, 2, "不追高"))
        if opportunity.board_height >= 5:
            items.append(ScoreItem("高度风险（≥5 板）", -6, 6, f"{opportunity.board_height} 板"))
        elif opportunity.board_height == 4:
            items.append(ScoreItem("高度风险（4 板）", -3, 3, "4 板"))
        broken = opportunity.sector_broken_ratio
        if broken is not None and broken >= 0.50:
            items.append(ScoreItem("板块炸板率高", -4, 4, f"{broken:.0%}"))
        if one_word:
            items.append(ScoreItem("一字板不可买入", -3, 3, "无法验证在涨停价成交"))

        deduction = min(RISK_MAX, sum(abs(i.points) for i in items))
        return items, float(deduction)

    def _tier(
        self,
        opportunity: Opportunity,
        stage: str,
        allowed: bool,
        score: float,
        risk_items: list[ScoreItem],
        risk_deduction: float,
    ) -> tuple[str, str, str]:
        """Environment gate first: a blocked stage can never produce an A."""
        reasons: list[str] = []
        risk_level = "high" if risk_deduction >= 12 else ("medium" if risk_deduction >= 6 else "low")

        if not allowed:
            reasons.append(f"当前阶段『{stage}』不允许『{opportunity.opportunity_type}』类型机会")
            return "C", "；".join(reasons), risk_level
        if stage in ("冰点", "退潮"):
            reasons.append(f"情绪阶段『{stage}』禁止进攻型机会，仅观察")
            return "C", "；".join(reasons), "high"
        if opportunity.state == "失效":
            reasons.append("失效条件已成立")
            return "C", "；".join(reasons), risk_level

        # Critical gaps invalidate the *setup itself* (no way to verify the
        # entry), so they block A. Auxiliary gaps (turnover / auction) are
        # deducted and surfaced but do not by themselves cap the tier -- the
        # accepted design says turnover gaps are labelled, not used to淘汰.
        critical_missing = self._critical_missing(risk_items)
        auxiliary_missing = [i.name for i in risk_items if i.name.startswith("数据缺失") or "未验证" in i.name]
        if opportunity.state != "可执行":
            reasons.append(f"尚未进入可执行状态：{opportunity.state_reason}")
            return "B", "；".join(reasons), risk_level
        if score < TIER_A_MIN_SCORE:
            reasons.append(f"排序分 {score:.1f} 低于 A 级门槛 {TIER_A_MIN_SCORE}")
            return "B", "；".join(reasons), risk_level
        if critical_missing:
            reasons.append("关键数据缺失导致无法验证执行：" + "、".join(critical_missing))
            return "B", "；".join(reasons), risk_level
        if opportunity.execution_mode != "次日开盘口径":
            reasons.append("缺少日内成交验证数据，不能作为可执行机会")
            return "B", "；".join(reasons), risk_level
        if opportunity.board_form == "一字板/未开板":
            reasons.append("一字板：无法假设在涨停价买入")
            return "B", "；".join(reasons), risk_level
        if stage == "分化":
            # Requirement: a stage that restricts offense must never show a
            # "重点关注" row, however high the ranking score is.
            if opportunity.board_height >= 3:
                reasons.append("分化阶段且为高位（≥3 板），提高淘汰标准")
            reasons.append("分化阶段降低仓位、提高标准，不列重点关注")
            return "B", "；".join(reasons), "high"

        # A tier: still not "buy now" -- execution happens at the next open.
        if auxiliary_missing:
            reasons.append("辅助数据缺失（已扣分、不阻断）：" + "、".join(auxiliary_missing))
        reasons.append(
            "信号为收盘确认型：执行在次日开盘，若开盘已在涨停价则视为无法买入（不假设成交）"
        )
        return "A", "；".join(reasons), risk_level

    @staticmethod
    def _critical_missing(risk_items: list[ScoreItem]) -> list[str]:
        """Gaps that make the entry itself unverifiable (vs merely less precise)."""
        keywords = ("日内", "分钟", "盘口", "封单量", "涨停时间")
        out: list[str] = []
        for item in risk_items:
            if item.name.startswith("数据缺失") or "不可执行" in item.name:
                if any(key in item.name for key in keywords):
                    out.append(item.name)
        return out

    # -- positions -----------------------------------------------------
    def _assign_positions(
        self,
        plans: list[TradePlan],
        *,
        stage: str,
        equity: float,
        cash: float,
        available_exposure: float,
        max_new: int,
    ) -> list[TradePlan]:
        """Size the plan: stage cap x type factor x single-name cap, in lots."""
        out: list[TradePlan] = []
        taken = 0
        used_exposure = 0.0
        per_name_cap = min(self.max_position_weight, available_exposure) if available_exposure else 0.0
        if max_new:
            per_name_cap = min(per_name_cap, available_exposure / max_new)

        for plan in plans:
            weight = 0.0
            note = ""
            factor = TYPE_POSITION_FACTOR.get(plan.opportunity_type, 0.0)
            eligible = (
                plan.tier == "A"
                and factor > 0
                and taken < max_new
                and available_exposure > 0
                and plan.state == "可执行"
            )
            if eligible:
                weight = per_name_cap * factor
                # Never let a single plan push the book past the stage cap.
                weight = min(weight, max(0.0, available_exposure - used_exposure))
            price = self._price_for(plan)
            shares = 0
            if weight > 0 and price and price > 0:
                shares = int(floor(weight * equity / price / LOT_SIZE) * LOT_SIZE)
            affordable = True
            if weight > 0 and shares <= 0:
                affordable = False
                note = "无法按当前建议仓位建立最小仓位（资金不足以买入 1 手）"
            elif weight > 0:
                cost = shares * (price or 0.0)
                if cost > cash + 1e-9:
                    affordable = False
                    note = "现金不足以按建议仓位买入（需 ≤ 可用现金）"
                    shares = 0
            if shares > 0:
                taken += 1
                used_exposure += (shares * (price or 0.0)) / equity if equity else 0.0

            position = PositionPlan(
                suggested_weight=weight,
                shares=shares,
                lots=shares // LOT_SIZE,
                amount=shares * (price or 0.0),
                affordable=affordable,
                note=note,
            )
            why = plan.why_not_buy
            if plan.tier == "A" and shares <= 0 and not why:
                why = note or "未分配仓位"
            out.append(
                TradePlan(
                    **{
                        **plan.__dict__,
                        "position": position,
                        "why_not_buy": why or "排序分仅代表优先级；执行需等次日开盘确认",
                    }
                )
            )
        return out

    def _price_for(self, plan: TradePlan) -> Optional[float]:
        return None if plan.price is None else float(plan.price)
