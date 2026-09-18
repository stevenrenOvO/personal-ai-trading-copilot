"""Tests for the trading-session service."""

from __future__ import annotations

from datetime import datetime

from backend.market.session import TradingSession


def _calendar(start: str, end: str) -> list[tuple[str, int]]:
    # 2026-09-17/18 are open, 2026-09-19/20 are a weekend (closed).
    return [
        ("20260916", 1),
        ("20260917", 1),
        ("20260918", 1),
        ("20260919", 0),
        ("20260920", 0),
        ("20260921", 1),
    ]


def _session(tmp_path, now: datetime) -> TradingSession:
    return TradingSession(
        db_path=tmp_path / "s.db", calendar_fetcher=_calendar, now_fn=lambda: now
    )


def test_current_trading_day_uses_calendar(tmp_path):
    session = _session(tmp_path, datetime(2026, 9, 17, 10, 30))
    assert session.current_trading_day() == "20260917"


def test_current_trading_day_skips_future_and_holidays(tmp_path):
    session = _session(tmp_path, datetime(2026, 9, 20, 10, 0))
    # 09-20 is a closed day; the latest open day not in the future is 09-18.
    assert session.current_trading_day() == "20260918"


def test_phase_classification(tmp_path):
    cases = [
        (datetime(2026, 9, 17, 9, 20), "集合竞价"),
        (datetime(2026, 9, 17, 10, 0), "盘中"),
        (datetime(2026, 9, 17, 12, 0), "午间休市"),
        (datetime(2026, 9, 17, 14, 0), "盘中"),
        (datetime(2026, 9, 17, 15, 10), "收盘"),
        (datetime(2026, 9, 17, 20, 0), "盘后"),
        (datetime(2026, 9, 19, 10, 0), "休市"),
    ]
    for now, expected in cases:
        assert _session(tmp_path, now).phase() == expected, now


def test_calendar_is_cached(tmp_path):
    calls = {"n": 0}

    def counting(start: str, end: str):
        calls["n"] += 1
        return _calendar(start, end)

    session = TradingSession(
        db_path=tmp_path / "s.db",
        calendar_fetcher=counting,
        now_fn=lambda: datetime(2026, 9, 17, 10, 0),
    )
    session.current_trading_day()
    session.current_trading_day()
    assert calls["n"] == 1, "calendar should be fetched once per month"
