"""Tests for emotion engine."""

import pytest

from backend.data.providers.fixture import FixtureProvider
from backend.emotion import EmotionEngine


@pytest.fixture
def fixture_provider():
    return FixtureProvider()


def test_emotion_calculation(fixture_provider):
    engine = EmotionEngine(fixture_provider)
    state = engine.calculate("20240103")
    assert state.date == "20240103"
    assert state.advance_decline_ratio is not None
    assert state.limit_up_count is not None
    assert state.sentiment in {
        "Greed",
        "Neutral Bullish",
        "Neutral",
        "Neutral Bearish",
        "Fear",
    }
    assert state.score is not None
    assert 0.0 <= state.score <= 100.0
