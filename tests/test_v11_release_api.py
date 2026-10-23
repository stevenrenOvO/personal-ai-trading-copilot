"""Cross-layer regressions for the V1.1 release completion."""
import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from backend.api import app
from backend.api.readiness import readiness
from backend.data import auto_ingest
from backend.data.providers.history import HistoryProvider
from backend.market.quote_service import QuoteService
from tests.test_api_v1 import _call, _make_runtime
from tests.test_auto_ingest import _Runtime, _Session
from tests.test_quote_service import _frame, _pin_session


def test_ready_does_not_initialize_runtime_or_call_vendor(monkeypatch):
    monkeypatch.setattr(app, "_RUNTIME", None)
    monkeypatch.setattr(app, "_RUNTIME_ERROR", False)
    monkeypatch.setattr(app, "_get_runtime", lambda: pytest.fail("readiness must not initialize synchronously"))
    replies = []
    payload = b"".join(app.application({"PATH_INFO": "/ready", "REQUEST_METHOD": "GET"}, lambda status, headers: replies.append(status)))
    assert replies == ["503 Service Unavailable"]
    assert json.loads(payload)["status"] == "starting"


def test_ready_reports_account_damage_without_hiding_homepage(tmp_path, monkeypatch):
    runtime = _make_runtime(tmp_path)
    monkeypatch.setattr(runtime.paper, "snapshot", lambda: (_ for _ in ()).throw(ValueError("broken")))
    status = readiness(runtime)
    assert status["status"] == "error" and status["checks"]["account"]["status"] == "error"
    monkeypatch.setattr(app, "_get_runtime", lambda: pytest.fail("static route requires no runtime"))
    replies = []
    payload = b"".join(app.application({"PATH_INFO": "/", "REQUEST_METHOD": "GET"}, lambda status, headers: replies.append(status)))
    assert replies == ["200 OK"] and b"<!DOCTYPE html>" in payload


def test_static_post_is_405_without_initializing(monkeypatch):
    monkeypatch.setattr(app, "_get_runtime", lambda: pytest.fail("unsupported method must not initialize"))
    replies = []
    app.application({"PATH_INFO": "/health", "REQUEST_METHOD": "POST"}, lambda status, headers: replies.append((status, headers)))
    assert replies[0][0] == "405 Method Not Allowed"
    assert ("Allow", "GET, HEAD") in replies[0][1]


def test_decision_request_retry_and_conflict_survive_runtime_restart(tmp_path):
    runtime = _make_runtime(tmp_path)
    body = {"request_id": "decision_retry_123", "ts_code": "000001.SZ", "action": "WAIT",
            "trade_date": "20240103", "user_reason": "等待确认"}
    first, data = _call("POST", "/trades", runtime, body)
    assert first == 201
    resumed = _make_runtime(tmp_path)
    second, duplicate = _call("POST", "/trades", resumed, body)
    assert second == 200 and duplicate["data"] == data["data"]
    conflict, _ = _call("POST", "/trades", runtime, {**body, "user_reason": "changed"})
    assert conflict == 409 and len(runtime.journal.list()) == 1


def test_backtest_history_survives_runtime_restart(tmp_path):
    runtime = _make_runtime(tmp_path)
    status, payload = _call("POST", "/backtest/runs", runtime,
        {"strategy": "ma_cross", "start_date": "20240101", "end_date": "20240105"})
    assert status == 201
    restored = _make_runtime(tmp_path)
    status, runs = _call("GET", "/backtest/runs", restored)
    assert status == 200 and runs["data"][0]["run_id"] == payload["data"]["run_id"]
    assert runs["data"][0]["metrics"] == payload["data"]["metrics"]


def test_concurrent_backtest_request_is_rejected_without_duplicate_work(tmp_path):
    import threading
    from backend.data.atomic import file_lock
    runtime = _make_runtime(tmp_path)
    entered, release = threading.Event(), threading.Event()
    def holder():
        with file_lock(tmp_path / "backtest" / "active"):
            entered.set()
            release.wait(5)
    worker = threading.Thread(target=holder)
    worker.start()
    try:
        assert entered.wait(2)
        status, payload = _call("POST", "/backtest/runs", runtime, {"strategy": "ma_cross"})
        assert status == 409 and "回测" in payload["detail"]
        assert not runtime.backtest_runs
    finally:
        release.set()
        worker.join(2)


def test_morning_quote_never_becomes_final_after_close(tmp_path):
    _pin_session(tmp_path, datetime(2024, 1, 3, 16, 0))
    service = QuoteService(db_path=tmp_path / "q.db", ttl_seconds=0, fetcher=lambda: (_frame(), {
        "source": "tencent", "quote_date": "20240103", "quote_time": "10:00:00",
        "fetched_at": datetime.now(timezone.utc).isoformat()}))
    meta = service.refresh()
    assert meta.status == "stale" and meta.is_stale


def test_failed_final_snapshot_stays_stale_on_later_read(tmp_path):
    _pin_session(tmp_path, datetime(2024, 1, 3, 16, 0))
    service = QuoteService(db_path=tmp_path / "q.db", fetcher=lambda: (_frame(), {
        "quote_date": "20240103", "quote_time": "15:00:00"}))
    assert service.refresh().status == "final"
    service._record_error("source unavailable")
    _, meta = service.get()
    assert meta.status == "error" and meta.is_stale


def test_quote_cache_read_does_not_refresh_exchange_calendar(tmp_path, monkeypatch):
    from backend.market.session import get_trading_session
    session = get_trading_session()
    service = QuoteService(db_path=tmp_path / "q.db", fetcher=lambda: (_frame(), {"quote_date": "20240103"}))
    service.refresh()
    monkeypatch.setattr(session, "ensure_calendar", lambda *args: pytest.fail("get is cache only"))
    assert not service.get()[0].empty


def test_page_reads_cannot_bypass_failed_quote_backoff(tmp_path, monkeypatch):
    service = QuoteService(db_path=tmp_path / "q.db")
    clock = [100.0]
    monkeypatch.setattr("backend.market.quote_service.time.monotonic", lambda: clock[0])
    service._record_error("blocked")
    service._wake.clear()
    for _ in range(20):
        service.get()
    assert not service._wake.is_set()
    clock[0] += 16
    service.get()
    assert service._wake.is_set()


def test_previous_day_snapshot_cannot_label_current_market_as_ready(tmp_path, monkeypatch):
    from backend.market.quote_service import QuoteMeta, get_quote_service
    runtime = _make_runtime(tmp_path)
    # This is a production cold-start check; explicit fixture mode has valid
    # local sample bars and should display those without a live quote feed.
    runtime.provider.name = "production-offline"
    monkeypatch.setattr(get_quote_service(), "get", lambda: (_frame(),
        QuoteMeta(quote_date="20240102", quote_time="15:00:00", status="ok", is_stale=False)))
    status, result = _call("GET", "/market/state?date=20240103", runtime)
    assert status == 200 and result["meta"]["status"] == "no_data"
    assert result["meta"]["is_stale"] and result["data"]["available"] is False
    assert result["data"]["score"] is None and result["data"]["recommended_exposure"] is None


@pytest.mark.parametrize("changes", [{"user_reason": {}}, {"price": True}, {"result": {"holding_days": -1}}])
def test_invalid_decision_fields_do_not_poison_journal(tmp_path, changes):
    runtime = _make_runtime(tmp_path)
    code, _ = _call("POST", "/trades", runtime, {"ts_code": "000001.SZ", "action": "WAIT", **changes})
    assert code == 400 and not runtime.journal.list()


def test_ingest_backoff_is_progressive_and_survives_restart(tmp_path, monkeypatch):
    auto_ingest.reset_state()
    provider = HistoryProvider(base_dir=tmp_path)
    runtime = _Runtime(provider)
    clock = [1000.0]
    monkeypatch.setattr(auto_ingest.time, "monotonic", lambda: clock[0])
    args = dict(session=_Session("盘后"), rows_on=lambda day: 0,
                backfill=lambda day: {"failed": 1}, run_sync=True)
    assert auto_ingest.ensure_today_bars(runtime, **args)
    first = json.loads(provider.history.get_state("auto_ingest_pending"))["20260918"]
    assert first["attempts"] == 1
    clock[0] += 301
    assert auto_ingest.ensure_today_bars(runtime, **args)
    second = json.loads(provider.history.get_state("auto_ingest_pending"))["20260918"]
    assert second["attempts"] == 2
    assert (datetime.fromisoformat(second["next_retry_at"]) - datetime.fromisoformat(first["next_retry_at"])).total_seconds() >= 299
    auto_ingest.reset_state()
    assert not auto_ingest.ensure_today_bars(runtime, **args)
    auto_ingest.reset_state()


def test_account_valuation_uses_valid_past_bar_when_latest_quote_invalid(tmp_path, monkeypatch):
    from backend.strategies.base import TradeSignal
    from backend.market.quote_service import QuoteMeta, get_quote_service
    runtime = _make_runtime(tmp_path)
    runtime.paper.execute_signal(TradeSignal("000001.SZ", "20240102", "BUY", price=10, quantity=100))
    monkeypatch.setattr(runtime.provider, "daily", lambda **kwargs: pd.DataFrame([
        {"ts_code": "000001.SZ", "trade_date": "20240102", "close": 12},
        {"ts_code": "000001.SZ", "trade_date": "20240103", "close": float("nan")},
        {"ts_code": "000001.SZ", "trade_date": "20240104", "close": 99}]))
    monkeypatch.setattr(get_quote_service(), "get", lambda: (pd.DataFrame([{"ts_code": "000001.SZ", "price": "bad"}]),
        QuoteMeta(quote_date="20240103", is_stale=False)))
    snapshot = runtime.account_snapshot("20240103")
    assert snapshot.valuation["prices"] == {"000001.SZ": 12}
    assert snapshot.valuation["price_dates"] == {"000001.SZ": "20240102"}
    assert snapshot.equity == pytest.approx(snapshot.cash + 1200)
