"""Gap backfill downloads in parallel but writes in order (1.2.2).

On 2026-11-20 a three-day gap took about an hour because ~5,200 stocks were
fetched one after another, each mostly waiting on the network.
"""

from __future__ import annotations

import threading
import time

import pandas as pd

from backend.data import ingest_full
from backend.data.history import MarketHistory

CODES = [f"{600000 + i}.SH" for i in range(12)]


def _bars(code, day="20261118"):
    return pd.DataFrame([{"ts_code": code, "trade_date": day, "open": 10, "high": 10.5, "low": 9.8,
                          "close": 10.2, "pre_close": 10, "vol": 1000}])


def test_downloads_overlap_and_everything_is_stored(tmp_path, monkeypatch):
    lock, active, peak = threading.Lock(), [0], [0]

    def fetch(code, start, end):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.1)
        with lock:
            active[0] -= 1
        return _bars(code)

    monkeypatch.setattr("backend.data.providers.tencent.fetch_daily", fetch)
    progress = []
    started = time.monotonic()
    result = ingest_full.run_backfill(start_date="20261118", end_date="20261118", codes=",".join(CODES),
                                      db_path=tmp_path / "h.db", workers=6,
                                      progress_callback=lambda p: progress.append((p["phase"], p["processed"])))
    elapsed = time.monotonic() - started
    assert result["done"] == len(CODES) and result["failed"] == 0
    # Overlap proves parallelism; the time bound only requires beating a sequential run.
    assert peak[0] > 1 and elapsed < 0.1 * len(CODES)
    assert MarketHistory(tmp_path / "h.db").codes_with_date("20261118") == set(CODES)
    assert [n for phase, n in progress if phase == "downloading" and n] == list(range(1, len(CODES) + 1))
    assert progress[-1] == ("completed", len(CODES))


def test_priority_stocks_are_written_first_even_if_others_finish_earlier(tmp_path, monkeypatch):
    priority = [CODES[-1], CODES[-2]]

    def fetch(code, start, end):
        time.sleep(0.15 if code in priority else 0.01)  # priority downloads are the slowest
        return _bars(code)

    written = []
    real_ingest = MarketHistory.ingest

    def ingest(self, frame):
        written.append(str(frame["ts_code"].iloc[0]))
        return real_ingest(self, frame)

    monkeypatch.setattr("backend.data.providers.tencent.fetch_daily", fetch)
    monkeypatch.setattr(MarketHistory, "ingest", ingest)
    ingest_full.run_backfill(start_date="20261118", end_date="20261118", codes=",".join(CODES),
                             db_path=tmp_path / "h.db", workers=6, priority_codes=priority)
    assert written[:2] == priority and sorted(written) == sorted(CODES)


def test_a_failing_stock_is_retried_reported_and_does_not_stop_the_rest(tmp_path, monkeypatch):
    attempts = []

    def fetch(code, start, end):
        if code == CODES[3]:
            attempts.append(code)
            raise ConnectionError("reset by peer")
        return _bars(code)

    monkeypatch.setattr("backend.data.providers.tencent.fetch_daily", fetch)
    monkeypatch.setattr(ingest_full.time, "sleep", lambda s: None)  # skip retry back-off
    result = ingest_full.run_backfill(start_date="20261118", end_date="20261118", codes=",".join(CODES),
                                      db_path=tmp_path / "h.db", workers=4, retries=1)
    assert result["failed"] == 1 and result["failed_codes"] == [CODES[3]]
    assert len(attempts) == 2  # retries=1 -> two attempts
    assert result["failures_by_date"]["20261118"]["failed"] == 1
    assert MarketHistory(tmp_path / "h.db").codes_with_date("20261118") == set(CODES) - {CODES[3]}


def test_bad_worker_setting_falls_back_and_store_errors_are_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("COPILOT_BACKFILL_WORKERS", "auto")
    monkeypatch.setattr("backend.data.providers.tencent.fetch_daily", lambda code, start, end: _bars(code))
    monkeypatch.setattr(ingest_full.time, "sleep", lambda s: None)
    real_ingest, calls = MarketHistory.ingest, {"n": 0}

    def flaky(self, frame):
        calls["n"] += 1
        if calls["n"] == 1:
            import sqlite3
            raise sqlite3.OperationalError("database is locked")
        return real_ingest(self, frame)

    monkeypatch.setattr(MarketHistory, "ingest", flaky)
    result = ingest_full.run_backfill(start_date="20261118", end_date="20261118", codes=",".join(CODES[:3]),
                                      db_path=tmp_path / "h.db")
    assert result["failed"] == 0 and result["done"] == 3


# -- per-request pre-check (was ~0.8 s on every API call) ---------------------------

def test_counts_by_date_is_one_query_excluding_indices(tmp_path):
    history = MarketHistory(tmp_path / "h.db")
    history.ingest(pd.concat([_bars(c, d) for c in CODES[:5] + ["000300.SH"] for d in ("20261118", "20261119")]))
    assert history.counts_by_date("20261118", "20261120", exclude=["000300.SH"]) == {"20261118": 5, "20261119": 5}
    assert history.counts_by_date("20261118", "20261118") == {"20261118": 6}


def test_production_precheck_uses_one_query_and_is_throttled(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from backend.data import auto_ingest
    from backend.data.providers.history import HistoryProvider
    from tests.test_auto_ingest import _CalendarSession

    auto_ingest.reset_state()
    codes = [f"{i:06d}.SZ" for i in range(1, 121)]
    pd.DataFrame({"ts_code": codes, "name": codes, "industry": "银行", "symbol": codes, "list_date": "19900101",
                  "list_status": "L", "is_st": "N"}).to_csv(tmp_path / "stock_basic.csv", index=False)
    provider = HistoryProvider(base_dir=tmp_path)
    days = ["20261117", "20261118", "20261119", "20261120"]
    provider.history.ingest(pd.concat([_bars(c, d) for c in codes + list(auto_ingest.INDEX_CODES) for d in days]))
    monkeypatch.setattr(ingest_full, "run_backfill", lambda **kw: (_ for _ in ()).throw(AssertionError("no download")))
    session = _CalendarSession(days, phase="盘后", day="20261120")
    runtime = SimpleNamespace(provider=provider, root=tmp_path, analysis_cache=None, context_cache=None)
    calls = {"grouped": 0}
    real = MarketHistory.counts_by_date

    def counting(self, *a, **kw):
        calls["grouped"] += 1
        return real(self, *a, **kw)

    def per_day(self, *a, **kw):
        raise AssertionError("per-day scan must not run on the request path")

    monkeypatch.setattr(MarketHistory, "counts_by_date", counting)
    monkeypatch.setattr(MarketHistory, "codes_with_date", per_day)
    monkeypatch.setattr(auto_ingest, "_schedule_closing_snapshot", lambda *a, **kw: None)
    for _ in range(5):
        assert auto_ingest.ensure_today_bars(runtime, session=session, max_gap_days=10) is False  # nothing missing
    assert calls["grouped"] == 1   # later calls within 30 s skip the scan
    auto_ingest.reset_state()


def test_date_availability_revision_sees_the_wal_file(tmp_path):
    from types import SimpleNamespace

    from backend.api.data_status import history_dates

    class Cache:
        def __init__(self):
            self.keys = []

        def get_or_compute(self, key, factory):
            self.keys.append(key)
            return []

    history = MarketHistory(tmp_path / "h.db")
    runtime = SimpleNamespace(provider=SimpleNamespace(history=history, name="history"), analysis_cache=Cache())
    history_dates(runtime)
    wal = history.db_path.with_name(history.db_path.name + "-wal")
    wal.write_bytes(b"x" * 10)  # simulate a write that has not reached the main file yet
    history_dates(runtime)
    assert runtime.analysis_cache.keys[0] != runtime.analysis_cache.keys[1]
