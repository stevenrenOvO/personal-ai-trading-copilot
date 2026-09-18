"""Tests for the Step-3 rule-based emotion cycle."""

from __future__ import annotations

from backend.emotion.cycle import (
    PHASES,
    STAGE_EXPOSURE,
    CycleMeasures,
    classify_cycle,
)


def _base(**overrides) -> CycleMeasures:
    values = dict(
        date="20240103",
        available=True,
        max_height=3,
        prev_max_height=3,
        limit_up_count=60,
        prev_limit_up_count=55,
        second_board=8,
        third_board=3,
        high_board=1,
        broken_count=10,
        broken_ratio=0.14,
        limit_down_count=3,
        promotion_rate=0.30,
        yesterday_avg_premium=1.5,
        high_board_avg_pct=2.0,
        ladder_complete=True,
        gap_count=0,
        advance_count=3000,
        decline_count=2000,
        traded_count=5000,
    )
    values.update(overrides)
    return CycleMeasures(**values)


def test_no_single_score_drives_the_stage():
    """A high intensity with broken structure must not silently read as strong."""
    strong_numbers = _base(
        limit_up_count=95,
        max_height=6,
        promotion_rate=0.40,
        yesterday_avg_premium=4.0,
        broken_ratio=0.10,
        advance_count=4200,
        decline_count=800,
    )
    # Same headline numbers, but the ladder underneath is falling apart.
    broken_ladder = _base(
        limit_up_count=95,
        max_height=6,
        promotion_rate=0.10,
        yesterday_avg_premium=-2.0,
        broken_ratio=0.45,
        high_board_avg_pct=-5.0,
        advance_count=1200,
        decline_count=3800,
    )
    assert classify_cycle(strong_numbers).stage == "高潮"
    assert classify_cycle(broken_ladder).stage in ("分化", "退潮")


def test_ice_point_requires_low_structure_and_negative_effect():
    verdict = classify_cycle(
        _base(limit_up_count=18, max_height=2, promotion_rate=0.10, yesterday_avg_premium=-1.8)
    )
    assert verdict.stage == "冰点"
    assert verdict.recommended_exposure == STAGE_EXPOSURE["冰点"]
    assert verdict.matched_rule


def test_collapse_of_breadth_is_a_decline_phase():
    verdict = classify_cycle(
        _base(
            limit_up_count=40,
            max_height=4,
            broken_ratio=0.42,
            promotion_rate=0.15,
            yesterday_avg_premium=0.2,
            advance_count=900,
            decline_count=4300,
        )
    )
    assert verdict.stage == "退潮"


def test_climax_needs_height_breadth_and_promotion_together():
    # Everything except the promotion rate: not a climax.
    without_promotion = _base(
        limit_up_count=95,
        max_height=6,
        promotion_rate=0.20,
        yesterday_avg_premium=4.0,
        second_board=10,
        third_board=4,
        high_board=3,
    )
    assert classify_cycle(without_promotion).stage != "高潮"

    climax = _base(
        limit_up_count=95,
        max_height=6,
        promotion_rate=0.40,
        yesterday_avg_premium=4.0,
        second_board=10,
        third_board=4,
        high_board=3,
    )
    verdict = classify_cycle(climax)
    assert verdict.stage == "高潮"
    # Climax is high risk and the exposure cap is deliberately not raised.
    assert verdict.risk_level == "high"
    assert verdict.recommended_exposure <= 0.5


def test_fermentation_requires_positive_money_effect():
    verdict = classify_cycle(
        _base(max_height=4, limit_up_count=70, promotion_rate=0.28, yesterday_avg_premium=2.0)
    )
    assert verdict.stage == "发酵"
    # Very low promotion means yesterday's limit-ups did not carry: not 发酵.
    weak = classify_cycle(
        _base(max_height=4, limit_up_count=70, promotion_rate=0.09, yesterday_avg_premium=0.2)
    )
    assert weak.stage != "发酵"


def test_contradictory_signals_fall_back_transparently():
    verdict = classify_cycle(
        _base(
            limit_up_count=35,
            max_height=3,
            promotion_rate=0.22,
            yesterday_avg_premium=-0.5,
            broken_ratio=0.30,
            gap_count=1,
        )
    )
    assert verdict.stage in PHASES
    assert "兜底" in verdict.matched_rule


def test_insufficient_data_never_forces_a_stage():
    verdict = classify_cycle(
        CycleMeasures(date="20240103", available=False, missing=("连板梯队结构",))
    )
    assert verdict.stage == "数据不足"
    assert verdict.recommended_exposure == 0.0
    assert verdict.confidence == 0.0
    assert verdict.unavailable


def test_verdict_is_explainable():
    verdict = classify_cycle(_base(limit_up_count=20, max_height=1, promotion_rate=0.05))
    assert verdict.evidence
    for item in verdict.evidence:
        assert item.name
        assert item.to_dict()["name"] == item.name
    assert verdict.invalidations           # what would change the stage
    assert verdict.strength_parts          # transparent intensity components
    assert any(p["weight"] for p in verdict.strength_parts)


def test_stage_exposure_caps_are_monotonic_sane():
    assert STAGE_EXPOSURE["冰点"] < STAGE_EXPOSURE["修复"] < STAGE_EXPOSURE["发酵"]
    assert STAGE_EXPOSURE["高潮"] <= STAGE_EXPOSURE["发酵"]
    assert STAGE_EXPOSURE["退潮"] <= STAGE_EXPOSURE["分化"]
