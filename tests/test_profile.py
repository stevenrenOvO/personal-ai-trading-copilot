"""Tests for user profile layer."""

import tempfile
from pathlib import Path
import pytest

from backend.profile.models import ProfileConfig, ProfileStats, UserProfile
from backend.profile.store import ProfileStore
from backend.profile.engine import ProfileEngine


def test_profile_config_roundtrip():
    config = ProfileConfig(
        risk_tolerance="aggressive",
        max_position_weight=0.3,
        max_drawdown_threshold=0.2,
        stop_loss_pct=0.05,
        take_profit_pct=0.3,
        preferred_sectors=["金融", "科技"],
        min_holding_days=1,
        max_holding_days=60,
    )
    d = config.to_dict()
    restored = ProfileConfig.from_dict(d)
    assert restored.risk_tolerance == "aggressive"
    assert restored.preferred_sectors == ["金融", "科技"]


def test_profile_stats_update_from_trades():
    stats = ProfileStats()
    trades = [
        {"pnl": 100.0, "return_pct": 0.05, "holding_days": 5},
        {"pnl": -50.0, "return_pct": -0.02, "holding_days": 3},
        {"pnl": 200.0, "return_pct": 0.08, "holding_days": 10},
    ]
    stats.update_from_trades(trades)
    assert stats.total_trades == 3
    assert stats.winning_trades == 2
    assert stats.win_rate == 2/3
    assert stats.avg_return == pytest.approx((0.05-0.02+0.08)/3)
    assert stats.avg_holding_days == pytest.approx((5+3+10)/3)
    assert stats.total_pnl == 250.0


def test_user_profile_roundtrip():
    config = ProfileConfig(risk_tolerance="moderate")
    stats = ProfileStats(total_trades=10, winning_trades=6)
    profile = UserProfile(
        user_id="test_user",
        config=config,
        stats=stats,
        last_updated="2026-09-09T00:00:00Z",
    )
    d = profile.to_dict()
    restored = UserProfile.from_dict(d)
    assert restored.user_id == "test_user"
    assert restored.config.risk_tolerance == "moderate"
    assert restored.stats.win_rate == 0.0  # not computed via update


def test_profile_store_save_load():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        store = ProfileStore(root)
        profile = UserProfile(user_id="test")
        store.save(profile)
        loaded = store.load()
        assert loaded is not None
        assert loaded.user_id == "test"
        # second save overwrites
        profile2 = UserProfile(user_id="test2")
        store.save(profile2)
        loaded2 = store.load()
        assert loaded2.user_id == "test2"


def test_profile_engine_load_or_create():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        engine = ProfileEngine(root)
        # first time returns default
        profile = engine.load_or_create()
        assert profile.user_id == "default"
        assert profile.config.risk_tolerance == "moderate"
        # save and reload
        profile.config.risk_tolerance = "aggressive"
        engine.store.save(profile)
        loaded = engine.load_or_create()
        assert loaded.config.risk_tolerance == "aggressive"


def test_profile_engine_refresh_stats():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        engine = ProfileEngine(root)
        # empty journal -> stats zero
        profile = engine.load_or_create()
        refreshed = engine.refresh_stats(profile)
        assert refreshed.stats.total_trades == 0
        assert refreshed.stats.win_rate == 0.0
        # now add some decisions to journal
        from backend.journal.store import DecisionJournal
        from backend.journal.models import DecisionRecord
        journal = DecisionJournal(root)
        rec1 = DecisionRecord.create(
            trade_date="20240103",
            ts_code="000001.SZ",
            action="BUY",
            signal_strength=0.8,
            risk_allowed=True, risk_violations=[],decision="APPROVED",
            reason="test1",
        )
        journal.record(rec1)
        rec2 = DecisionRecord.create(
            trade_date="20240104",
            ts_code="600000.SH",
            action="SELL",
            signal_strength=0.4,
            risk_allowed=True, risk_violations=[],decision="APPROVED",
            reason="test2",
        )
        journal.record(rec2)
        refreshed2 = engine.refresh_stats(profile)
        assert refreshed2.stats.total_trades == 2
        # win_rate defaults to 0.0 because pnl not computed
        assert refreshed2.stats.win_rate == 0.0


def test_profile_engine_update_config():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        engine = ProfileEngine(root)
        profile = engine.load_or_create()
        new_config = ProfileConfig(risk_tolerance="conservative", max_position_weight=0.1)
        updated = engine.update_config(profile, new_config)
        assert updated.config.risk_tolerance == "conservative"
        assert updated.config.max_position_weight == 0.1
        # reload and verify persistence
        loaded = engine.store.load()
        assert loaded is not None
        assert loaded.config.risk_tolerance == "conservative"

