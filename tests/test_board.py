"""Tests for the board engine."""

from backend.board import BoardEngine
from backend.data.providers.fixture import FixtureProvider


def test_board_engine_builds_universe_and_filters():
    provider = FixtureProvider()
    engine = BoardEngine(provider)

    universe = engine.get_index_constituents()
    assert universe

    filtered = engine.filter_by_conditions(
        universe,
        min_price=10.0,
        date="20240103",
    )

    assert isinstance(filtered, list)
