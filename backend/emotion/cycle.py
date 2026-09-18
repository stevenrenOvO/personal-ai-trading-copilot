"""Rule-based A-share short-term emotion cycle.

The cycle is decided by an **ordered rule table over real market structure**,
not by a single composite score. Each stage carries the exact rule that fired
and the measured value behind every condition, so the UI can answer "why is
today this stage" with data.

Stages: 冰点 / 修复 / 发酵 / 高潮 / 分化 / 退潮

Inputs (all from Step 2's ladder engine + the market engine)
-----------------------------------------------------------
连板高度与变化, 连板家数与梯队分布, 昨日涨停溢价, 昨日涨停晋级率, 炸板率,
涨停/跌停家数, 高位板(>=3板)今日表现, 梯队完整度/断层, 涨跌家数, 成交额.

Thresholds are conventional A-share short-term levels (not fitted to any
particular day): 涨停家数 30/50/80, 炸板率 35%/40%, 晋级率 25%/30%/35%,
最高连板 2/3/5. They are declared once here so a change is reviewable.

This stage only judges **what the market is**, never what to buy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


PHASES = ("冰点", "修复", "发酵", "高潮", "分化", "退潮")

# Recommended total exposure cap per stage (documented policy, not fitted).
STAGE_EXPOSURE = {
    "冰点": 0.1,
    "修复": 0.3,
    "发酵": 0.6,
    "高潮": 0.5,
    "分化": 0.3,
    "退潮": 0.1,
    "数据不足": 0.0,
}

STAGE_RISK = {
    "冰点": "high",
    "修复": "medium",
    "发酵": "medium",
    "高潮": "high",
    "分化": "high",
    "退潮": "high",
    "数据不足": "high",
}

STAGE_INVALIDATION = {
    "冰点": (
        "涨停家数回升到 40 家以上且最高连板升到 3 板 → 转修复",
        "昨日涨停溢价继续为负且跌停增加 → 维持冰点",
    ),
    "修复": (
        "炸板率升到 40% 以上或溢价转负 → 回落冰点/退潮",
        "最高连板升到 4 板以上且晋级率≥30% → 转发酵",
    ),
    "发酵": (
        "溢价转负或炸板率≥40% → 转分化",
        "最高连板≥5 且涨停≥80 家且晋级率≥35% → 转高潮",
    ),
    "高潮": (
        "炸板率≥35% 或溢价转负或高位板转负 → 转分化",
        "最高连板下降且晋级率<25% → 转退潮",
    ),
    "分化": (
        "溢价与晋级率同步走高、炸板率回落到 35% 以下 → 转发酵",
        "涨家数占比<25% 且炸板率≥35% → 转退潮",
    ),
    "退潮": (
        "涨停家数止跌回升且溢价转正 → 转修复",
        "继续缩量杀跌、最高连板降到 2 板以下 → 转冰点",
    ),
    "数据不足": ("补齐本地历史后重新计算",),
}


@dataclass(frozen=True)
class CycleMeasures:
    """Everything the rules are allowed to look at."""

    date: str
    available: bool = True
    # 连板结构（Step 2）
    max_height: Optional[int] = None
    prev_max_height: Optional[int] = None
    limit_up_count: Optional[int] = None
    prev_limit_up_count: Optional[int] = None
    second_board: Optional[int] = None
    third_board: Optional[int] = None
    high_board: Optional[int] = None
    broken_count: Optional[int] = None
    broken_ratio: Optional[float] = None
    limit_down_count: Optional[int] = None
    promotion_rate: Optional[float] = None
    prev_promotion_rate: Optional[float] = None
    yesterday_avg_premium: Optional[float] = None
    yesterday_up_ratio: Optional[float] = None
    high_board_avg_pct: Optional[float] = None
    ladder_complete: Optional[bool] = None
    gap_count: Optional[int] = None
    # 市场环境
    advance_count: Optional[int] = None
    decline_count: Optional[int] = None
    traded_count: Optional[int] = None
    total_amount: Optional[float] = None
    prev_total_amount: Optional[float] = None
    missing: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def advance_ratio(self) -> Optional[float]:
        if self.advance_count is None or self.decline_count is None:
            return None
        total = self.advance_count + self.decline_count
        return round(self.advance_count / total, 4) if total else None

    @property
    def amount_change(self) -> Optional[float]:
        if not self.total_amount or not self.prev_total_amount:
            return None
        return round(self.total_amount / self.prev_total_amount - 1.0, 4)


@dataclass(frozen=True)
class Evidence:
    """One measured condition behind the verdict."""

    name: str
    value: Any
    unit: str = ""
    threshold: str = ""
    passed: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "threshold": self.threshold,
            "passed": self.passed,
        }

    def text(self) -> str:
        value = "—" if self.value is None else self.value
        suffix = f"{value}{self.unit}"
        return f"{self.name} {suffix}（判定 {self.threshold}）" if self.threshold else f"{self.name} {suffix}"


@dataclass(frozen=True)
class CycleVerdict:
    stage: str
    matched_rule: str
    strength: float
    risk_level: str
    recommended_exposure: float
    evidence: tuple[Evidence, ...] = ()
    invalidations: tuple[str, ...] = ()
    strength_parts: tuple[dict[str, Any], ...] = ()
    confidence: float = 0.0
    notes: tuple[str, ...] = ()
    unavailable: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "matched_rule": self.matched_rule,
            "strength": self.strength,
            "risk_level": self.risk_level,
            "recommended_exposure": self.recommended_exposure,
            "evidence": [e.to_dict() for e in self.evidence],
            "invalidations": list(self.invalidations),
            "strength_parts": list(self.strength_parts),
            "confidence": self.confidence,
            "notes": list(self.notes),
            "unavailable": list(self.unavailable),
        }


def _pct(value: Optional[float], digits: int = 1) -> Optional[str]:
    return None if value is None else f"{value * 100:.{digits}f}%"


def _num(value: Optional[float], digits: int = 2) -> Optional[str]:
    return None if value is None else f"{value:.{digits}f}"


def _strength_parts(m: CycleMeasures) -> tuple[tuple[dict[str, Any], ...], float]:
    """Transparent 0-100 intensity, with every contribution exposed.

    This is a *display* index for comparing days; it never decides the stage.
    """
    parts: list[dict[str, Any]] = []

    def add(name: str, raw: Optional[float], low: float, high: float, weight: float, unit: str = "") -> None:
        if raw is None:
            parts.append({"name": name, "value": None, "ratio": None, "weight": weight, "contribution": None, "unit": unit})
            return
        ratio = 0.0 if high == low else max(0.0, min(1.0, (raw - low) / (high - low)))
        parts.append(
            {
                "name": name,
                "value": round(raw, 4),
                "unit": unit,
                "ratio": round(ratio, 4),
                "weight": weight,
                "contribution": round(ratio * weight, 2),
            }
        )

    add("晋级率", m.promotion_rate, 0.0, 0.6, 30.0)
    add("昨日涨停溢价", m.yesterday_avg_premium, -3.0, 5.0, 20.0, "%")
    add("涨停家数", m.limit_up_count, 0.0, 100.0, 20.0)
    add("封板质量(1-炸板率)", None if m.broken_ratio is None else 1 - m.broken_ratio, 0.4, 0.9, 15.0)
    add("最高连板", m.max_height, 0.0, 7.0, 15.0)

    total = sum(p["contribution"] for p in parts if p["contribution"] is not None)
    return tuple(dict(p) for p in parts), round(max(0.0, min(100.0, total)), 2)


def classify_cycle(m: CycleMeasures) -> CycleVerdict:
    """Decide the stage with an ordered, documented rule table."""
    strength_parts, strength = _strength_parts(m)
    unavailable = list(m.missing)

    if not m.available or m.limit_up_count is None or m.max_height is None:
        return CycleVerdict(
            stage="数据不足",
            matched_rule="无有效连板结构数据（非交易日或本地历史未覆盖），不做阶段判断",
            strength=0.0,
            risk_level=STAGE_RISK["数据不足"],
            recommended_exposure=STAGE_EXPOSURE["数据不足"],
            evidence=tuple(Evidence(name=n, value="未接入") for n in unavailable),
            invalidations=STAGE_INVALIDATION["数据不足"],
            strength_parts=strength_parts,
            confidence=0.0,
            notes=m.notes,
            unavailable=tuple(unavailable),
        )

    lu = m.limit_up_count
    prev_lu = m.prev_limit_up_count if m.prev_limit_up_count is not None else lu
    height = m.max_height
    prev_height = m.prev_max_height if m.prev_max_height is not None else height
    broken = m.broken_ratio
    promotion = m.promotion_rate
    premium = m.yesterday_avg_premium
    high_board = m.high_board_avg_pct
    advance_ratio = m.advance_ratio
    lower_boards = sum(x or 0 for x in (m.second_board, m.third_board, m.high_board))
    gaps = m.gap_count

    def ev(name, value, unit="", threshold="", passed=None):
        return Evidence(name=name, value=value, unit=unit, threshold=threshold, passed=passed)

    def verdict(stage: str, rule: str, evidence: list[Evidence]) -> CycleVerdict:
        # 成交额环境 is reported as context but deliberately NOT used to pick a
        # stage: the Tencent gap-fill days carry no turnover, so a rule built on
        # it would silently stop working on those dates.
        evidence = list(evidence) + [
            Evidence(
                name="成交额较前一日",
                value=(
                    None
                    if m.amount_change is None
                    else f"{m.amount_change * 100:.1f}"
                ),
                unit="%" if m.amount_change is not None else "",
                threshold="仅作环境参考，不参与阶段判定",
            )
        ]
        confidence = 1.0 - 0.12 * len(unavailable)
        return CycleVerdict(
            stage=stage,
            matched_rule=rule,
            strength=strength,
            risk_level=STAGE_RISK[stage],
            recommended_exposure=STAGE_EXPOSURE[stage],
            evidence=tuple(evidence),
            invalidations=STAGE_INVALIDATION[stage],
            strength_parts=strength_parts,
            confidence=round(max(0.2, confidence), 2),
            notes=m.notes,
            unavailable=tuple(unavailable),
        )

    # R1/R2 冰点 ---------------------------------------------------------
    if lu <= 30 and height <= 2 and (
        (premium is not None and premium < 0) or (promotion is not None and promotion < 0.20)
    ):
        return verdict(
            "冰点",
            "涨停≤30家 且 最高连板≤2板 且 赚钱效应为负（溢价<0 或 晋级率<20%）",
            [
                ev("涨停家数", lu, "家", "≤30", True),
                ev("最高连板", height, "板", "≤2", True),
                ev("昨日涨停溢价", _num(premium), "%", "<0", premium is not None and premium < 0),
                ev("晋级率", _pct(promotion), "", "<20%", promotion is not None and promotion < 0.20),
            ],
        )
    if lu <= 20 and m.limit_down_count is not None and m.limit_down_count >= lu:
        return verdict(
            "冰点",
            "涨停≤20家 且 跌停≥涨停",
            [
                ev("涨停家数", lu, "家", "≤20", True),
                ev("跌停家数", m.limit_down_count, "家", "≥涨停", True),
            ],
        )

    # R3/R4 退潮 ---------------------------------------------------------
    if height < prev_height and broken is not None and broken >= 0.40 and (
        (premium is not None and premium <= 0) or (promotion is not None and promotion < 0.25)
    ):
        return verdict(
            "退潮",
            "最高连板下降 且 炸板率≥40% 且 赚钱效应转弱（溢价≤0 或 晋级率<25%）",
            [
                ev("最高连板", height, "板", f"低于前一日 {prev_height}", True),
                ev("炸板率", _pct(broken), "", "≥40%", True),
                ev("昨日涨停溢价", _num(premium), "%", "≤0", premium is not None and premium <= 0),
                ev("晋级率", _pct(promotion), "", "<25%", promotion is not None and promotion < 0.25),
            ],
        )
    if advance_ratio is not None and advance_ratio < 0.25 and broken is not None and broken >= 0.35 and (
        promotion is not None and promotion < 0.30
    ):
        return verdict(
            "退潮",
            "涨家数占比<25% 且 炸板率≥35% 且 晋级率<30%（广度与封板质量同步恶化）",
            [
                ev("上涨家数占比", _pct(advance_ratio), "", "<25%", True),
                ev("炸板率", _pct(broken), "", "≥35%", True),
                ev("晋级率", _pct(promotion), "", "<30%", True),
                ev("涨停家数", lu, "家", "—"),
            ],
        )

    # R5 高潮 -------------------------------------------------------------
    if (
        height >= 5
        and lu >= 80
        and (promotion or 0) >= 0.35
        and premium is not None
        and premium >= 2.0
        and lower_boards >= 8
    ):
        return verdict(
            "高潮",
            "最高连板≥5板 且 涨停≥80家 且 晋级率≥35% 且 溢价≥2% 且 二板以上家数≥8",
            [
                ev("最高连板", height, "板", "≥5", True),
                ev("涨停家数", lu, "家", "≥80", True),
                ev("晋级率", _pct(promotion), "", "≥35%", True),
                ev("昨日涨停溢价", _num(premium), "%", "≥2%", True),
                ev("二板及以上家数", lower_boards, "家", "≥8", True),
                ev("炸板率", _pct(broken), "", "越低越好"),
            ],
        )

    # R8 发酵：赚钱效应为正、梯队仍在，不需要所有指标同时完美 ------------
    if (
        height >= 3
        and (premium or 0) > 0
        and lu >= 50
        and (broken is None or broken < 0.40)
        # 晋级率过低说明昨日涨停基本没有延续，「发酵」名不副实
        and (promotion is None or promotion >= 0.15)
    ):
        return verdict(
            "发酵",
            "最高连板≥3板 且 涨停≥50家 且 溢价>0 且 炸板率<40%（赚钱效应为正、梯队仍在）",
            [
                ev("最高连板", height, "板", "≥3", True),
                ev("昨日涨停溢价", _num(premium), "%", ">0", True),
                ev("涨停家数", lu, "家", "≥50", True),
                ev("炸板率", _pct(broken), "", "<40%", broken is None or broken < 0.40),
                ev("晋级率", _pct(promotion), "", "≥15%", promotion is None or promotion >= 0.15),
                ev("二板及以上家数", lower_boards, "家", "—"),
            ],
        )

    # R6/R7 分化：赚钱效应还在，但要*同时*出现两项恶化，避免单个指标误杀 --
    worsening = [
        name
        for name, flag in (
            ("炸板率≥35%", broken is not None and broken >= 0.35),
            ("晋级率<20%", promotion is not None and promotion < 0.20),
            ("溢价<0", premium is not None and premium < 0),
            ("高位板今日为负", high_board is not None and high_board < 0),
        )
        if flag
    ]
    if lu >= 50 and len(worsening) >= 2:
        return verdict(
            "分化",
            "涨停家数≥50（赚钱效应尚在）但同时出现两项恶化：" + "、".join(worsening),
            [
                ev("涨停家数", lu, "家", "≥50", True),
                ev("炸板率", _pct(broken), "", "≥35%", broken is not None and broken >= 0.35),
                ev("晋级率", _pct(promotion), "", "<20%", promotion is not None and promotion < 0.20),
                ev("昨日涨停溢价", _num(premium), "%", "<0", premium is not None and premium < 0),
                ev("高位板(≥3板)今日均值", _num(high_board), "%", "<0", high_board is not None and high_board < 0),
            ],
        )
    if (
        gaps is not None
        and gaps >= 2
        and height >= 3
        and (promotion is None or promotion < 0.30)
    ):
        return verdict(
            "分化",
            "梯队断层≥2处 且 最高连板≥3 且 晋级率<30%（高度有余、承接不足）",
            [
                ev("梯队断层", gaps, "处", "≥2", True),
                ev("最高连板", height, "板", "≥3", True),
                ev("晋级率", _pct(promotion), "", "<30%", True),
                ev("涨停家数", lu, "家", "—"),
            ],
        )

    # R9/R10 修复 ---------------------------------------------------------
    if (prev_lu and lu >= prev_lu * 1.5) and height >= prev_height and (
        broken is None or broken < 0.40
    ):
        return verdict(
            "修复",
            "涨停家数较前一日增加≥50% 且 最高连板未下降 且 炸板率<40%（亏钱效应收敛）",
            [
                ev("涨停家数", lu, "家", f"较前一日 {prev_lu} 家增≥50%", True),
                ev("最高连板", height, "板", f"不低于前一日 {prev_height}", True),
                ev("炸板率", _pct(broken), "", "<40%", broken is None or broken < 0.40),
            ],
        )
    if (
        premium is not None
        and premium >= 0
        and m.prev_limit_up_count is not None
        and height <= 3
        and (broken is None or broken < 0.45)
    ):
        return verdict(
            "修复",
            "昨日涨停溢价转正 且 最高连板≤3板 且 炸板率<45%（低位回暖）",
            [
                ev("昨日涨停溢价", _num(premium), "%", "≥0", True),
                ev("最高连板", height, "板", "≤3", True),
                ev("炸板率", _pct(broken), "", "<45%", broken is None or broken < 0.45),
            ],
        )

    # R11 兜底：指标互相矛盾时按赚钱效应方向归类，并注明这是兜底规则 ------
    if (premium or 0) >= 0 and (promotion or 0) >= 0.25:
        return verdict(
            "发酵",
            "兜底规则：指标未命中明确规则，但赚钱效应为正（溢价≥0 且 晋级率≥25%）",
            [
                ev("昨日涨停溢价", _num(premium), "%", "≥0", True),
                ev("晋级率", _pct(promotion), "", "≥25%", True),
                ev("涨停家数", lu, "家", "—"),
                ev("炸板率", _pct(broken), "", "—"),
            ],
        )
    return verdict(
        "分化",
        "兜底规则：指标未命中明确规则且赚钱效应偏弱，按分化处理（需人工复核）",
        [
            ev("涨停家数", lu, "家", "—"),
            ev("昨日涨停溢价", _num(premium), "%", "—"),
            ev("晋级率", _pct(promotion), "", "—"),
            ev("炸板率", _pct(broken), "", "—"),
            ev("上涨家数占比", _pct(advance_ratio), "", "—"),
        ],
    )
