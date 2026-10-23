"""Recovery contracts for usable pages during an interrupted full-market fill."""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from backend.data import auto_ingest, ingest_full
from backend.data.history import MarketHistory
from backend.data.providers.history import HistoryProvider
from tests.test_auto_ingest import _CalendarSession, _Runtime, _Session


@pytest.fixture(autouse=True)
def reset_ingest_state():
    auto_ingest.reset_state()
    yield
    auto_ingest.reset_state()


def provider_at(path, codes=("000001.SZ",)):
    provider = HistoryProvider(base_dir=path)
    provider.stock_basic = lambda: pd.DataFrame({"ts_code": list(codes)})
    return provider


def test_progress_reports_committed_rows_before_the_entire_download_finishes(tmp_path, monkeypatch):
    path = tmp_path / "history.db"
    history = MarketHistory(path)
    reports = []

    def fetch(code, start, end):
        if code == "000002.SZ":
            return pd.DataFrame()
        return pd.DataFrame([{"ts_code": code, "trade_date": end, "close": 10}])

    def progress(payload):
        reports.append((dict(payload), history.rows_on("20261022")))

    monkeypatch.setattr("backend.data.providers.tencent.fetch_daily", fetch)
    result = ingest_full.run_backfill(start_date="20261022", end_date="20261022",
        codes="000001.SZ,000002.SZ", db_path=path, progress_callback=progress)

    assert reports[0][0]["processed"] == 0 and reports[0][0]["total"] == 2
    assert reports[1][0]["processed"] == 1 and reports[1][1] == 1
    assert reports[-1][0]["phase"] == "completed"
    assert reports[-1][0]["downloaded"] == reports[-1][0]["stored_rows"] == 1
    assert reports[-1][0]["failed"] == result["failed"] == 1
    assert reports[-1][0]["window_end"] == "20261022"
    assert datetime.fromisoformat(reports[-1][0]["updated_at"]).tzinfo is not None


def test_critical_market_indices_are_downloaded_before_stock_alphabetical_order(tmp_path, monkeypatch):
    calls = []

    def fetch(code, start, end):
        calls.append(code)
        return pd.DataFrame([{"ts_code": code, "trade_date": end, "close": 10}])

    monkeypatch.setattr("backend.data.providers.tencent.fetch_daily", fetch)
    ingest_full.run_backfill(start_date="20261022", end_date="20261022",
        codes="000001.SZ,399006.SZ,000300.SH", db_path=tmp_path / "history.db",
        priority_codes=["000300.SH", "399006.SZ"])
    assert calls == ["000300.SH", "399006.SZ", "000001.SZ"]


def test_latest_incomplete_day_precedes_high_coverage_missing_symbols(tmp_path):
    provider = provider_at(tmp_path, [f"{code:06}.SZ" for code in range(5000)])
    provider.history.set_state("auto_ingest_pending", json.dumps({"20261021": {"failed": 3}}))
    counts = {"20261021": 4997, "20261022": 2000}
    calls = []

    def fill(day):
        calls.append(day)
        counts[day] = 4997
        return {"failed": 3}

    assert auto_ingest.ensure_today_bars(_Runtime(provider),
        session=_CalendarSession(list(counts), day="20261022"), rows_on=counts.__getitem__,
        backfill=fill, run_sync=True)
    assert calls == ["20261022"]
    pending = json.loads(provider.history.get_state("auto_ingest_pending"))
    assert pending["20261022"]["coverage_sufficient"] is True
    retry = datetime.fromisoformat(pending["20261022"]["next_retry_at"])
    assert retry - datetime.now(timezone.utc) > timedelta(hours=5, minutes=59)
    assert "20261021" in pending


def test_recent_download_windows_are_processed_before_older_ones(tmp_path, monkeypatch):
    provider = provider_at(tmp_path)
    calls = []
    dates = [f"202610{day:02}" for day in range(1, 23)]

    def fill(**kwargs):
        calls.append(kwargs)
        # A completed window must be recorded before the older one starts,
        # so restarting a long-running job does not lose its gap report.
        if len(calls) > 1:
            pending = json.loads(provider.history.get_state("auto_ingest_pending"))
            assert "20261022" in pending
        return {"failures_by_date": {day: {"failed": 1} for day in kwargs["required_dates"]}}

    monkeypatch.setattr(ingest_full, "run_backfill", fill)
    assert auto_ingest.ensure_today_bars(_Runtime(provider),
        session=_CalendarSession(dates, day="20261022"), rows_on=lambda day: 0, run_sync=True)
    assert len(calls) == 3
    assert calls[0]["end_date"] == "20261022"
    assert calls[1]["end_date"] < calls[0]["start_date"]
    assert calls[0]["required_dates"] == sorted(calls[0]["required_dates"])
    assert set(calls[0]["codes"].split(",")) == {"000001.SZ", *auto_ingest.INDEX_CODES}


def test_incremental_writes_invalidate_visible_results_while_job_is_running(tmp_path, monkeypatch):
    provider = provider_at(tmp_path)
    runtime = _Runtime(provider)
    clock = [1000.0]
    monkeypatch.setattr(auto_ingest.time, "monotonic", lambda: clock[0])

    def fill(**kwargs):
        callback = kwargs["progress_callback"]
        callback({"processed": 0, "total": 20, "stored_rows": 0})
        assert runtime.calls == []
        clock[0] += 6
        callback({"processed": 5, "total": 20, "downloaded": 5, "stored_rows": 5})
        assert len(runtime.calls) == 2  # analysis and AI/context cache
        snapshot = auto_ingest.status(runtime)
        assert snapshot["running"] and snapshot["progress"]["processed"] == 5
        clock[0] += 6
        callback({"processed": 6, "total": 20, "stored_rows": 5})
        assert len(runtime.calls) == 2  # no new data, no needless recomputation
        callback({"processed": 7, "total": 20, "stored_rows": 7})
        assert len(runtime.calls) == 4
        return {"failed": 1}

    monkeypatch.setattr(ingest_full, "run_backfill", fill)
    assert auto_ingest.ensure_today_bars(runtime, session=_Session("盘后"),
        rows_on=lambda day: 0, run_sync=True)
    snapshot = auto_ingest.status(runtime)
    assert not snapshot["running"] and snapshot["progress"]["phase"] == "retry_pending"


def test_missing_indices_are_repaired_without_refetching_suspended_stock_tails(tmp_path, monkeypatch):
    provider = provider_at(tmp_path)
    provider.history.ingest(pd.DataFrame([
        {"ts_code": "000001.SZ", "trade_date": "20261022", "close": 10}]))
    calls = []

    def fill(**kwargs):
        calls.append(kwargs)
        provider.history.ingest(pd.DataFrame([
            {"ts_code": code, "trade_date": "20261022", "close": 1000}
            for code in auto_ingest.INDEX_CODES]))
        return {"failed": 0}

    monkeypatch.setattr(ingest_full, "run_backfill", fill)
    runtime = _Runtime(provider)
    assert auto_ingest.ensure_today_bars(runtime, session=_Session("盘后", "20261022"),
        rows_on=lambda day: 5000, run_sync=True)
    assert len(calls) == 1 and set(calls[0]["codes"].split(",")) == set(auto_ingest.INDEX_CODES)
    assert auto_ingest.status(runtime)["progress"]["priority"] == "indices"


def test_index_rows_cannot_make_the_stock_coverage_gate_pass(tmp_path, monkeypatch):
    codes = [f"{code:06}.SZ" for code in range(120)]
    provider = provider_at(tmp_path, codes)
    provider.history.ingest(pd.DataFrame([
        {"ts_code": code, "trade_date": "20261022", "close": 10}
        for code in codes[:107] + list(auto_ingest.INDEX_CODES)]))
    calls = []
    monkeypatch.setattr(ingest_full, "run_backfill", lambda **kwargs: calls.append(kwargs) or {"failed": 1})
    assert provider.history.rows_on("20261022") == 112
    runtime = _Runtime(provider)
    assert auto_ingest.ensure_today_bars(runtime, session=_Session("盘后", "20261022"), run_sync=True)
    assert calls and auto_ingest.status(runtime)["progress"]["priority"] == "stocks"
    assert auto_ingest.status(runtime)["threshold_rows"] == 108
