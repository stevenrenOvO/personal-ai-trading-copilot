"""Tests for the V1 market / emotion / sector / board engines."""

from __future__ import annotations

import pandas as pd

from backend.board.engine import BoardEngine
from backend.data.providers.fixture import FixtureProvider
from backend.emotion.engine import EMOTION_PHASES, EmotionEngine
from backend.market.engine import MarketEngine
from backend.sector.engine import SectorEngine


def test_market_state_has_score_and_factors():
    provider = FixtureProvider()
    state = MarketEngine(provider).calculate("20240103")
    assert 0 <= state.score <= 100
    assert state.state
    assert isinstance(state.factors, tuple)
    assert 0 <= state.confidence <= 1


def test_emotion_cycle_is_seven_phase():
    provider = FixtureProvider()
    state = EmotionEngine(provider).calculate("20240103")
    assert state.emotion_cycle in EMOTION_PHASES
    assert 0 <= state.emotion_score <= 100
    assert 0 <= state.recommended_exposure <= 1


def test_sector_ranked_orders_by_score():
    provider = FixtureProvider()
    scores = SectorEngine(provider).ranked("20240103")
    assert scores
    values = [s.score for s in scores]
    assert values == sorted(values, reverse=True)
    assert scores[0].rank == 1
    assert "score" in scores[0].to_dict()


def test_board_environment_has_grade_and_ladder():
    provider = FixtureProvider()
    env = BoardEngine(provider).calculate("20240103")
    assert env.grade in {"A", "B+", "B", "C", "D"}
    assert env.ladder.max_height >= 0


def _ladder_daily() -> pd.DataFrame:
    """Three codes: a 3-board run, a 1-board, and one that broke its run."""
    rows = []
    for day in ("20240101", "20240102", "20240103"):
        for code, limit_up in (
            ("000001.SZ", True),
            ("600000.SH", day == "20240103"),
            ("000002.SZ", day != "20240102"),  # broke the run on the middle day
        ):
            rows.append(
                {
                    "ts_code": code,
                    "trade_date": day,
                    "open": 10.0,
                    "high": 10.0,
                    "low": 10.0,
                    "close": 11.0 if limit_up else 10.0,
                    "pre_close": 10.0,
                    "change": 0.0,
                    "pct_chg": 10.0 if limit_up else 0.0,
                    "vol": 100.0,
                    "amount": 1000.0,
                }
            )
    return pd.DataFrame(rows)


def test_consecutive_limit_up_heights_counts_real_runs():
    from backend.market.limit_ladder import (
        consecutive_limit_up_heights,
        max_height,
    )

    heights = consecutive_limit_up_heights(
        _ladder_daily(), "20240103", lambda day: pd.DataFrame()
    )
    # 10% closes: 000001.SZ three in a row, 600000.SH only today,
    # 000002.SZ interrupted on the middle day so its run is 1.
    assert heights == {"000001.SZ": 3, "600000.SH": 1, "000002.SZ": 1}
    assert max_height(heights) == 3


def test_consecutive_limit_up_heights_uses_vendor_limits_when_present():
    from backend.market.limit_ladder import consecutive_limit_up_heights

    # 11.0 would pass the plain 10% fallback, but the vendor limit says the true
    # cap that day was 11.5 (a 20% board), so nothing closed limit-up.
    limits = pd.DataFrame(
        {
            "trade_date": ["20240103", "20240103"],
            "ts_code": ["000001.SZ", "600000.SH"],
            "up_limit": [11.5, 11.5],
        }
    )
    heights = consecutive_limit_up_heights(
        _ladder_daily(), "20240103", lambda day: limits if day == "20240103" else None
    )
    assert heights == {}
