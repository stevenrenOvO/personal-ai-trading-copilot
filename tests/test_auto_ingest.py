"""Tests for the after-close auto-ingest of the current trading day's bars."""

from __future__ import annotations

from datetime import datetime

import pytest

from backend.data import auto_ingest
from backend.data.providers.history import HistoryProvider


class _Session:
    def __init__(self, phase: str, day: str = "20260918") -> None:
        self._phase = phase
        self._day = day

    def phase(self) -> str:
        return self._phase

    def current_trading_day(self):
        return self._day


class _Runtime:
    def __init__(self, provider) -> None:
        self.provider = provider
        self.calls: list[str] = []

        class _Cache:
            def clear(self_inner):  # noqa: N805
                self.calls.append("cache-cleared")

        self.analysis_cache = _Cache()
        self.context_cache = _Cache()


@pytest.fixture(autouse=True)
def _reset():
    auto_ingest.reset_state()
    yield
    auto_ingest.reset_state()


def test_skipped_when_provider_is_not_the_local_store():
    runtime = _Runtime(object())
    started = auto_ingest.ensure_today_bars(
        runtime, session=_Session("盘后"), rows_on=lambda d: 0, run_sync=True
    )
    assert started is False


def test_skipped_while_the_session_is_still_open():
    runtime = _Runtime(_HistoryProviderStub())
    started = auto_ingest.ensure_today_bars(
        runtime, session=_Session("盘中"), rows_on=lambda d: 0, run_sync=True
    )
    assert started is False


def test_runs_after_close_when_the_day_is_missing():
    runtime = _Runtime(_HistoryProviderStub())
    ran: list[str] = []
    started = auto_ingest.ensure_today_bars(
        runtime,
        session=_Session("盘后"),
        rows_on=lambda d: 0,
        backfill=lambda d: ran.append(d) or {"done": 1},
        run_sync=True,
    )
    assert started is True
    assert ran == ["20260918"]
    assert "cache-cleared" in runtime.calls      # derived caches dropped


def test_runs_only_once_per_day():
    runtime = _Runtime(_HistoryProviderStub())
    calls: list[str] = []
    args = dict(
        session=_Session("盘后"),
        rows_on=lambda d: 0,
        backfill=lambda d: calls.append(d) or {},
        run_sync=True,
    )
    assert auto_ingest.ensure_today_bars(runtime, **args) is True
    assert auto_ingest.ensure_today_bars(runtime, **args) is False
    assert calls == ["20260918"]


def test_skipped_when_the_day_is_already_stored():
    runtime = _Runtime(_HistoryProviderStub())
    calls: list[str] = []
    started = auto_ingest.ensure_today_bars(
        runtime,
        session=_Session("盘后"),
        rows_on=lambda d: 5000,
        backfill=lambda d: calls.append(d) or {},
        run_sync=True,
    )
    assert started is False
    assert calls == []


def test_intraday_phases_are_never_final():
    for phase in ("盘前", "集合竞价", "开盘前", "盘中", "午间休市"):
        assert phase not in auto_ingest.FINAL_PHASES
    assert set(auto_ingest.FINAL_PHASES) == {"收盘", "盘后"}


class _HistoryProviderStub(HistoryProvider):
    """Subclass without running __init__ (no DB needed for the gate logic)."""

    def __init__(self) -> None:  # noqa: D107
        pass


class _CalendarSession(_Session):
    """Session stub with a cached exchange calendar (in-memory sqlite)."""

    def __init__(self, days: list[str], phase: str = "盘后", day: str = "20260918"):
        super().__init__(phase, day)
        self._days = sorted(days)

    def _connect(self):
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE calendar (cal_date TEXT PRIMARY KEY, is_open INTEGER)")
        conn.executemany(
            "INSERT INTO calendar VALUES (?, 1)", [(d,) for d in self._days]
        )
        return conn


def test_all_missing_recent_sessions_are_filled_oldest_first():
    """After a few days away the app must heal every gap, not just today."""
    runtime = _Runtime(_HistoryProviderStub())
    stored = {"20260915"}                 # the store stopped here; older days exist
    stored |= {"20260911", "20260914"}    # ...and are present, so only these 3 are gaps
    calls: list[str] = []
    started = auto_ingest.ensure_today_bars(
        runtime,
        session=_CalendarSession(
            ["20260911", "20260914", "20260915", "20260916", "20260917", "20260918"]
        ),
        rows_on=lambda d: 5000 if d in stored else 0,
        backfill=lambda d: calls.append(d) or {},
        run_sync=True,
    )
    assert started is True
    assert calls == ["20260916", "20260917", "20260918"]   # oldest first


def test_gap_lookback_is_bounded():
    runtime = _Runtime(_HistoryProviderStub())
    calls: list[str] = []
    auto_ingest.ensure_today_bars(
        runtime,
        session=_CalendarSession([f"2026090{i}" for i in range(1, 9)] + ["20260918"]),
        rows_on=lambda d: 0,
        backfill=lambda d: calls.append(d) or {},
        run_sync=True,
        max_gap_days=3,
    )
    assert len(calls) == 3                # only the most recent 3 candidates
