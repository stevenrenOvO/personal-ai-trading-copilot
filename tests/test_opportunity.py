"""Tests for the opportunity engine."""

from backend.data.providers.fixture import FixtureProvider
from backend.opportunity import OpportunityEngine
from backend.sector import SectorEngine
from backend.strategies.ma_cross import MaCrossStrategy


def test_opportunity_scan_returns_list_without_crashing():
    provider = FixtureProvider()
    strategy = MaCrossStrategy(fast_window=1, slow_window=2)
    sector_engine = SectorEngine(provider)
    engine = OpportunityEngine(provider, strategy, sector_engine)

    opportunities = engine.scan("20240103")

    assert isinstance(opportunities, list)
