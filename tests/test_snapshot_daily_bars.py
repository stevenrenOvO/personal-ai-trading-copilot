"""Closing snapshot -> daily bars, date-scoped cache invalidation, WAL.

Each test exposes a problem seen on 2026-11-17: the overview could not load
while a backfill ran (every progress tick wiped every cached analysis, and a
~50 s market-state computation never got to publish), and today's bars were
only obtainable through a ~5,000-request per-stock download even though the
closing snapshot already holds them.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.api.memo import TTLCache
from backend.data import auto_ingest
from backend.data.history import MarketHistory
from backend.data.snapshot_bars import (
    SOURCE_STATE_PREFIX, is_closing_snapshot, snapshot_to_bars,
)
from backend.market.quote_service import QuoteMeta

DAY = "20261117"


def _snapshot(**overrides):
    rows = [
        # traded normally
        dict(ts_code="000001.SZ", name="平安银行", price=12.25, change=-0.05, pct_chg=-0.4065,
             pre_close=12.30, open=12.25, high=12.32, low=12.11, volume=172_041_200.0, amount=2.1e9),
        dict(ts_code="600519.SH", name="贵州茅台", price=1330.75, change=None, pct_chg=None,
             pre_close=1355.90, open=1354.98, high=1354.98, low=1322.60, volume=3_880_500.0, amount=5.2e9),
        # suspended: Tencent repeats pre_close with zero volume
        dict(ts_code="600010.SH", name="停牌股", price=2.0, change=0.0, pct_chg=0.0,
             pre_close=2.0, open=0.0, high=0.0, low=0.0, volume=0.0, amount=0.0),
        # inconsistent: high below close
        dict(ts_code="000002.SZ", name="坏数据", price=10.0, change=0.1, pct_chg=1.0,
             pre_close=9.9, open=9.9, high=9.95, low=9.8, volume=1000.0, amount=1e4),
    ]
    frame = pd.DataFrame(rows)
    meta = QuoteMeta(source="tencent", quote_date=DAY, quote_time="15:20:15",
                     as_of="2026-11-17T07:20:15+00:00", status="final", is_stale=False, rows=len(rows))
    for key, value in overrides.items():
        setattr(meta, key, value)
    return frame, meta


class _Quotes:
    def __init__(self, frame, meta):
        self.frame, self.meta, self.calls = frame, meta, 0

    def get(self):
        self.calls += 1
        return self.frame.copy(), self.meta


def _runtime(tmp_path):
    history = MarketHistory(tmp_path / "history" / "market_history.db")
    return SimpleNamespace(
        root=tmp_path,
        provider=SimpleNamespace(history=history),
        analysis_cache=TTLCache(ttl_seconds=600),
        context_cache=TTLCache(ttl_seconds=600),
    )


@pytest.fixture(autouse=True)
def _reset_snapshot_memo():
    auto_ingest._snapshot_done.clear()
    auto_ingest._snapshot_checked.clear()
    yield
    auto_ingest._snapshot_done.clear()
    auto_ingest._snapshot_checked.clear()


# -- snapshot -> bars ----------------------------------------------------------

def test_only_a_final_after_close_snapshot_of_that_day_counts():
    _, meta = _snapshot()
    assert is_closing_snapshot(meta, DAY)
    assert not is_closing_snapshot(_snapshot(status="ok")[1], DAY)          # intraday
    assert not is_closing_snapshot(_snapshot(quote_time="14:59:58")[1], DAY)
    assert not is_closing_snapshot(_snapshot(quote_date="20261116")[1], DAY)  # yesterday's
    assert not is_closing_snapshot(_snapshot(quote_time="")[1], DAY)


def test_snapshot_rows_become_daily_bars_in_store_units():
    frame, _ = _snapshot()
    bars = snapshot_to_bars(frame, DAY).set_index("ts_code")
    assert set(bars.index) == {"000001.SZ", "600519.SH"}  # suspended and inconsistent dropped
    pa = bars.loc["000001.SZ"]
    assert (pa.trade_date, pa.open, pa.high, pa.low, pa.close) == (DAY, 12.25, 12.32, 12.11, 12.25)
    assert pa.vol == 172_041_200.0 and pa.amount == 2.1e9  # shares / yuan, as stored today
    mt = bars.loc["600519.SH"]
    assert mt.change == pytest.approx(1330.75 - 1355.90)
    assert mt.pct_chg == pytest.approx((1330.75 - 1355.90) / 1355.90 * 100)


def test_store_writes_whole_market_once_and_never_overwrites(tmp_path):
    runtime = _runtime(tmp_path)
    history = runtime.provider.history
    # A per-stock download already stored this bar; it must win.
    history.ingest(pd.DataFrame([dict(ts_code="000001.SZ", trade_date=DAY, open=1, high=1, low=1,
                                      close=1, pre_close=1, change=0, pct_chg=0, vol=1, amount=None)]))
    quotes = _Quotes(*_snapshot())
    record = auto_ingest.store_closing_snapshot(runtime, DAY, quotes)
    assert record["inserted"] == 1 and record["valid_bars"] == 2
    assert record["source"] == "tencent_snapshot"
    assert history.codes_with_date(DAY) == {"000001.SZ", "600519.SH"}
    kept = history.load(ts_codes=["000001.SZ"], start_date=DAY, end_date=DAY)
    assert kept.iloc[0]["close"] == 1
    assert '"source": "tencent_snapshot"' in history.get_state(SOURCE_STATE_PREFIX + DAY)
    # Same snapshot version again: no second write.
    assert auto_ingest.store_closing_snapshot(runtime, DAY, quotes) is None


def test_intraday_snapshot_is_never_stored(tmp_path):
    runtime = _runtime(tmp_path)
    quotes = _Quotes(*_snapshot(status="ok", quote_time="10:31:00"))
    assert auto_ingest.store_closing_snapshot(runtime, DAY, quotes) is None
    assert runtime.provider.history.rows_on(DAY) == 0


def test_storing_today_keeps_older_analysis_cached(tmp_path):
    runtime = _runtime(tmp_path)
    runtime.analysis_cache.put(("market", "20261028"), "old-day")
    runtime.analysis_cache.put(("market", DAY), "today-before")
    auto_ingest.store_closing_snapshot(runtime, DAY, _Quotes(*_snapshot()))
    assert runtime.analysis_cache.get(("market", "20261028")) == "old-day"
    assert runtime.analysis_cache.get(("market", DAY)) is None


def test_history_store_uses_wal(tmp_path):
    history = MarketHistory(tmp_path / "h.db")
    with sqlite3.connect(history.db_path) as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_watchlist_codes_feed_backfill_priority(tmp_path):
    from backend.personal import WatchlistStore

    WatchlistStore(tmp_path).add("600519.SH")
    assert auto_ingest._watchlist_codes(SimpleNamespace(root=tmp_path)) == ["600519.SH"]
    assert auto_ingest._watchlist_codes(SimpleNamespace()) == []


# -- date-scoped invalidation --------------------------------------------------

def test_invalidate_from_drops_only_dependent_keys():
    cache = TTLCache(ttl_seconds=600)
    cache.put(("market", "20261028"), 1)
    cache.put(("plan", "20261105", 3), 2)
    cache.put(("coverage",), 3)
    cache.invalidate_from("20261104")
    assert cache.get(("market", "20261028")) == 1
    assert cache.get(("plan", "20261105", 3)) is None
    assert cache.get(("coverage",)) is None


def test_slow_computation_of_older_day_survives_repeated_backfill_ticks():
    """The 2026-11-17 failure: a backfill tick every 5 s, a ~50 s computation."""
    cache = TTLCache(ttl_seconds=600)
    started, release = threading.Event(), threading.Event()
    calls = []

    def slow():
        calls.append(1)
        started.set()
        release.wait(5)
        return "state-1028"

    results = []
    first = threading.Thread(target=lambda: results.append(cache.get_or_compute(("market", "20261028"), slow)))
    first.start()
    assert started.wait(5)
    for _ in range(3):                       # backfill writes 11-04 .. 11-17
        cache.invalidate_from("20261104")
    second = threading.Thread(target=lambda: results.append(cache.get_or_compute(("market", "20261028"), slow)))
    second.start()
    time.sleep(0.05)
    release.set()
    first.join(5)
    second.join(5)
    assert results == ["state-1028", "state-1028"]
    assert len(calls) == 1                   # the second request joined, no recompute
    assert cache.get(("market", "20261028")) == "state-1028"


def test_computation_whose_inputs_changed_does_not_publish():
    cache = TTLCache(ttl_seconds=600)
    started, release = threading.Event(), threading.Event()

    def slow():
        started.set()
        release.wait(5)
        return "stale"

    worker = threading.Thread(target=lambda: cache.get_or_compute(("market", DAY), slow))
    worker.start()
    assert started.wait(5)
    cache.invalidate_from(DAY)
    # A new request must not join the stale computation.
    assert cache.get_or_compute(("market", DAY), lambda: "fresh") == "fresh"
    release.set()
    worker.join(5)
    assert cache.get(("market", DAY)) == "fresh"


def test_next_morning_writes_yesterdays_closing_snapshot(tmp_path):
    """Opened before the open: the cached snapshot is yesterday's close."""
    runtime = _runtime(tmp_path)

    class Session:
        def describe_cached(self):
            return {"phase": "盘前", "current_trading_day": "20261118", "expected_daily_date": DAY}

    days = auto_ingest.closing_snapshot_days(Session(), "20261118")
    assert days == ["20261118", DAY]
    auto_ingest._schedule_closing_snapshot(runtime, days, _Quotes(*_snapshot()))
    with auto_ingest._snapshot_lock:   # wait for the background writer
        pass
    assert runtime.provider.history.codes_with_date(DAY) == {"000001.SZ", "600519.SH"}
    assert runtime.provider.history.rows_on("20261118") == 0


def test_recheck_after_restart_keeps_the_original_source_record(tmp_path):
    """2026-11-18 pre-open: a restart re-checked yesterday's snapshot, inserted
    nothing and overwrote the record of the run that wrote 3,977 bars."""
    import json

    runtime = _runtime(tmp_path)
    history = runtime.provider.history
    first = auto_ingest.store_closing_snapshot(runtime, DAY, _Quotes(*_snapshot()))
    assert first["inserted"] == 2 and first["inserted_total"] == 2 and len(first["runs"]) == 1

    auto_ingest._snapshot_done.clear()  # process restart
    again = auto_ingest.store_closing_snapshot(
        runtime, DAY, _Quotes(*_snapshot(as_of="2026-11-17T16:05:00+00:00")))
    assert again["unchanged"] and again["inserted"] == 0
    saved = json.loads(history.get_state(SOURCE_STATE_PREFIX + DAY))
    assert saved["inserted"] == 2 and saved["as_of"] == "2026-11-17T07:20:15+00:00" and len(saved["runs"]) == 1


def test_later_snapshot_that_adds_bars_is_appended_as_a_run(tmp_path):
    import json

    runtime = _runtime(tmp_path)
    history = runtime.provider.history
    frame, meta = _snapshot()
    # Legacy (V1.2) record format already present for the day.
    history.set_state(SOURCE_STATE_PREFIX + DAY, json.dumps({
        "trade_date": DAY, "source": "tencent_snapshot", "quote_time": "15:01:00",
        "as_of": "2026-11-17T07:01:00+00:00", "snapshot_rows": 1, "valid_bars": 1, "inserted": 1}))
    auto_ingest.store_closing_snapshot(runtime, DAY, _Quotes(frame, meta))
    saved = json.loads(history.get_state(SOURCE_STATE_PREFIX + DAY))
    assert [run["inserted"] for run in saved["runs"]] == [1, 2] and saved["inserted_total"] == 3
