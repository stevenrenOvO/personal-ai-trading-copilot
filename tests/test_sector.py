"""Tests for the sector engine."""

from backend.data.providers.fixture import FixtureProvider
from backend.sector import SectorEngine


def test_sector_engine_calculates_state():
    provider = FixtureProvider()
    engine = SectorEngine(provider)

    states = engine.calculate("20240103")

    assert states
    assert states[0].date == "20240103"
    assert states[0].name
    assert states[0].stock_codes
    assert states[0].advance_count + states[0].decline_count > 0
