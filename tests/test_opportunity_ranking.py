"""Tests for Step-4 stage 3: ranking breakdown, risk gating and sizing."""

from __future__ import annotations

from backend.opportunity.ranking import (
    A_TIER_LIMIT,
    OpportunityRankingEngine,
)
from backend.opportunity.setups import Opportunity, SetupResult


class _NoProvider:
    """Ranking never touches the provider when setups are supplied."""


def _opportunity(
    code="000001.SZ",
    *,
    type_="首板",
    state="可执行",
    stage_allowed=True,
    height=1,
    role="龙头候选",
    sector_rank=1,
    limit_ups=6,
    relative=3.0,
    broken=0.1,
    volume=2.0,
    promotion=0.4,
    board_form="开盘封板",
    execution_mode="次日开盘口径",
    price=10.0,
    warnings=(),
):
    return Opportunity(
        ts_code=code,
        name=code[:6],
        opportunity_type=type_,
        sector="银行",
        role=role,
        board_height=height,
        board_form=board_form,
        state=state,
        state_reason="条件成立" if state == "可执行" else "条件未满足",
        data_status="日线已确认",
        execution_mode=execution_mode,
        environment_allowed=stage_allowed,
        reasons=("fixture",),
        sector_rank=sector_rank,
        sector_limit_up_count=limit_ups,
        sector_relative_strength=relative,
        sector_broken_ratio=broken,
        volume_ratio=volume,
        promotion_rate=promotion,
        ladder_complete=True,
        price=price,
        warnings=warnings,
        constraints={
            "t_plus_one": True,
            "lot_size": 100,
            "limit_up_cannot_buy": True,
            "one_word_cannot_fill": True,
            "fill_assumption": "次日开盘买入（不假设涨停价成交）",
        },
    )


def _plan(stage="发酵", opportunities=(), **state):
    setups = SetupResult(
        date="20240105",
        available=True,
        stage=stage,
        stage_rule="test",
        pool_size=len(opportunities),
        opportunities=tuple(opportunities),
        type_counts={},
        state_counts={},
    )
    engine = OpportunityRankingEngine(_NoProvider(), setup_engine=None, max_position_weight=0.2)
    return engine.build("20240105", setups=setups, state=state)


def test_score_is_fully_decomposable():
    plan = _plan(opportunities=[_opportunity()])
    item = plan.plans[0]
    blocks = item.to_dict()["score_breakdown"]["blocks"]
    assert [b["name"] for b in blocks] == ["环境", "板块", "梯队", "个股", "买点"]
    total_blocks = sum(b["points"] for b in blocks)
    assert abs(total_blocks - item.risk_deduction - item.score) < 1e-6
    for block in blocks:
        assert block["items"], "every block must list what contributed"
        for part in block["items"]:
            assert part["detail"]


def test_risk_deduction_only_subtracts_and_is_explained():
    clean = _plan(opportunities=[_opportunity()]).plans[0]
    noisy = _plan(
        opportunities=[
            _opportunity(
                code="000002.SZ",
                warnings=("成交额未接入（该交易日数据源未提供），流动性未验证",),
            )
        ]
    ).plans[0]
    assert noisy.risk_deduction > clean.risk_deduction
    assert all(i.points <= 0 for i in noisy.risk_items)
    assert any("流动性" in i.name for i in noisy.risk_items)


def test_environment_gate_beats_a_high_score():
    """A perfect score in a restricted stage must not be 重点关注."""
    plan = _plan(stage="退潮", opportunities=[_opportunity()])
    item = plan.plans[0]
    assert item.tier == "C"
    assert plan.can_attack is False
    assert plan.high_quality_count == 0
    assert "退潮" in item.why_not_buy


def test_ice_point_and_decline_never_produce_a_tier():
    for stage in ("冰点", "退潮"):
        plan = _plan(stage=stage, opportunities=[_opportunity()])
        assert plan.tier_counts["A"] == 0
        assert plan.max_new_positions == 0


def test_restricted_stage_differentiation_never_produces_a_tier():
    # 首板 is allowed in 分化 (the gate permits it) but must still not be A.
    plan = _plan(stage="分化", opportunities=[_opportunity(height=1, type_="首板")])
    assert plan.tier_counts["A"] == 0
    assert plan.plans[0].tier == "B"


def test_a_tier_is_a_limited_slot_count_not_a_threshold():
    many = [_opportunity(code=f"00000{i}.SZ") for i in range(1, 8)]
    plan = _plan(stage="发酵", opportunities=many)
    assert plan.tier_counts["A"] == A_TIER_LIMIT["发酵"]
    downgraded = [p for p in plan.plans if p.tier == "B"]
    assert any("未进入当日重点关注" in p.why_not_buy for p in downgraded)


def test_critical_missing_data_blocks_a_tier():
    plan = _plan(
        opportunities=[
            _opportunity(
                type_="打板",
                execution_mode="日内不可执行（缺分钟/盘口数据）",
                warnings=("日内成交不可验证：不得假设在涨停价成交",),
            )
        ]
    )
    assert plan.plans[0].tier != "A"
    why = plan.plans[0].why_not_buy
    assert "数据" in why or "不可执行" in why or "不允许" in why


def test_auxiliary_missing_data_deducts_but_does_not_block():
    plan = _plan(
        opportunities=[
            _opportunity(
                warnings=("成交额未接入（该交易日数据源未提供），流动性未验证",)
            )
        ]
    )
    item = plan.plans[0]
    assert item.tier == "A"
    assert "辅助数据缺失" in item.why_not_buy


def test_position_uses_lots_and_respects_caps():
    plan = _plan(stage="发酵", equity=1_000_000.0, cash=1_000_000.0,
                 opportunities=[_opportunity()])
    item = plan.plans[0]
    assert item.position.shares % 100 == 0
    assert item.position.suggested_weight <= 0.20 + 1e-9     # single-name cap
    assert item.position.suggested_weight <= plan.total_exposure_cap + 1e-9
    assert item.position.lots == item.position.shares // 100


def test_total_exposure_never_exceeds_stage_cap():
    many = [_opportunity(code=f"00000{i}.SZ") for i in range(1, 8)]
    plan = _plan(stage="修复", equity=1_000_000.0, cash=1_000_000.0, opportunities=many)
    total = sum(p.position.suggested_weight for p in plan.plans)
    assert total <= plan.total_exposure_cap + 1e-9
    assert len([p for p in plan.plans if p.position.shares > 0]) <= plan.max_new_positions


def test_insufficient_cash_reports_no_minimum_position():
    plan = _plan(
        stage="发酵",
        equity=1_000.0,      # too small for one lot at 10 yuan
        cash=1_000.0,
        opportunities=[_opportunity()],
    )
    item = plan.plans[0]
    assert item.position.shares == 0
    assert item.position.affordable is False
    assert "无法" in item.position.note or "现金不足" in item.position.note


def test_t_plus_one_limits_and_lot_size_are_declared():
    plan = _plan(opportunities=[_opportunity()])
    item = plan.plans[0]
    assert item.data_freshness["t_plus_one"] is True
    assert item.data_freshness["lot_size"] == 100
    assert item.data_freshness["limit_up_cannot_buy"] is True


def test_no_opportunity_returns_wait_message():
    plan = _plan(stage="发酵", opportunities=[])
    assert plan.plans == ()
    assert plan.high_quality_count == 0
    assert "无高质量机会" in plan.attack_note
    assert plan.can_attack is False


def test_why_not_buy_is_always_present():
    cases = [
        _plan(opportunities=[_opportunity()]),
        _plan(stage="退潮", opportunities=[_opportunity()]),
        _plan(stage="分化", opportunities=[_opportunity(height=3, type_="连板接力")]),
    ]
    for plan in cases:
        for item in plan.plans:
            assert item.why_not_buy
            assert len(item.why_not_buy) > 5


def test_unavailable_result_is_reported_not_invented():
    setups = SetupResult(date="20240105", available=False, notes=("no data",))
    engine = OpportunityRankingEngine(_NoProvider())
    plan = engine.build("20240105", setups=setups)
    assert plan.available is False
    assert plan.plans == ()
    assert plan.unavailable
