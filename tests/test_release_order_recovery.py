"""Release regressions for request recovery; all accounts are temporary."""

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from backend.api import v1
from tests.test_api_v1 import _call, _make_runtime


@pytest.fixture(autouse=True)
def no_background_ingest(monkeypatch):
    monkeypatch.setattr("backend.data.auto_ingest.ensure_today_bars", lambda *args: None)


def _request(**changes):
    return {"ts_code": "000001.SZ", "action": "BUY", "price": 10,
            "quantity": 100, "signal_id": "release-recovery-001", **changes}


@pytest.mark.parametrize("explicit_date", [False, True])
def test_confirmed_fill_recovery_survives_midnight_restart_and_calendar_outage(
        tmp_path, monkeypatch, explicit_date):
    runtime = _make_runtime(tmp_path)
    request = _request(**({"trade_date": "20240103"} if explicit_date else {}))
    status, first = _call("POST", "/paper/trade", runtime, request)
    assert status == 201 and first["data"]["order"]["status"] == "FILLED"
    restored = _make_runtime(tmp_path)
    monkeypatch.setattr(v1, "get_trading_session", lambda: pytest.fail("recovery requires no calendar"))
    monkeypatch.setattr(restored.provider, "stock_basic", lambda **kwargs: pytest.fail("recovery requires no listing"))
    monkeypatch.setattr(restored.paper, "execute_signal", lambda *args: pytest.fail("recovery must never execute"))
    status, retried = _call("POST", "/paper/trade", restored, request)
    assert status == 200 and retried["data"]["order"] == first["data"]["order"]
    assert len(restored.paper.snapshot().orders) == len(restored.journal.list()) == 1
    assert retried["data"]["snapshot"]["cash"] == first["data"]["snapshot"]["cash"]


@pytest.mark.parametrize("changes", [{"trade_date": "20240104"}, {"price": 10.1},
    {"quantity": 200}, {"action": "SELL"}, {"ts_code": "600000.SH"}])
def test_confirmed_identity_cannot_be_used_for_a_different_order(tmp_path, monkeypatch, changes):
    runtime = _make_runtime(tmp_path)
    assert _call("POST", "/paper/trade", runtime, _request())[0] == 201
    cash = runtime.paper.snapshot().cash
    monkeypatch.setattr(v1, "get_trading_session", lambda: pytest.fail("conflict requires no calendar"))
    status, result = _call("POST", "/paper/trade", runtime, _request(**changes))
    assert status == 409 and "signal_id" in result["detail"]
    assert runtime.paper.snapshot().cash == cash
    assert len(runtime.journal.list()) == 1


def test_recovery_does_not_enable_new_historical_orders(tmp_path):
    runtime = _make_runtime(tmp_path)
    status, _ = _call("POST", "/paper/trade", runtime, _request(trade_date="20240102"))
    assert status == 400 and not runtime.paper.snapshot().orders


def test_concurrent_duplicate_orders_across_runtimes_fill_once(tmp_path):
    runtimes = [_make_runtime(tmp_path) for _ in range(6)]
    barrier = threading.Barrier(len(runtimes))
    def submit(runtime):
        barrier.wait(timeout=5)
        return _call("POST", "/paper/trade", runtime, _request())
    with ThreadPoolExecutor(max_workers=len(runtimes)) as pool:
        replies = list(pool.map(submit, runtimes))
    assert all(status in (200, 201) for status, _ in replies)
    assert len({response["data"]["order"]["signal_id"] for _, response in replies}) == 1
    snapshot = runtimes[0].paper.snapshot()
    assert len(snapshot.orders) == len(runtimes[0].journal.list()) == 1
    assert snapshot.positions["000001.SZ"].quantity == 100


def test_concurrent_conflicting_orders_are_reported_as_conflict(tmp_path):
    runtimes = [_make_runtime(tmp_path) for _ in range(2)]
    barrier = threading.Barrier(2)
    def submit(item):
        runtime, quantity = item
        barrier.wait(timeout=5)
        return _call("POST", "/paper/trade", runtime, _request(quantity=quantity))[0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(submit, zip(runtimes, [100, 200])))
    assert sorted(statuses) == [201, 409]
    assert len(runtimes[0].paper.snapshot().orders) == 1


@pytest.mark.parametrize("signal_id", ["", "   "])
def test_blank_explicit_request_identity_does_not_disable_idempotency(tmp_path, signal_id):
    runtime = _make_runtime(tmp_path)
    assert _call("POST", "/paper/trade", runtime, _request(signal_id=signal_id))[0] == 400
    assert not runtime.paper.snapshot().orders


def test_concurrent_decision_retries_create_one_durable_record(tmp_path):
    runtimes = [_make_runtime(tmp_path) for _ in range(4)]
    request = {"ts_code": "000001.SZ", "action": "WAIT", "request_id": "release-decision-retry"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(lambda runtime: _call("POST", "/trades", runtime, request), runtimes))
    assert sorted(status for status, _ in responses) == [200, 200, 200, 201]
    assert len(runtimes[0].journal.list()) == 1


@pytest.mark.parametrize("changes", [{"price": 10 ** 500}, {"quantity": 10 ** 500},
    {"result": {"pnl": 10 ** 500}}, {"trade_date": []}, {"trade_date": False},
    {"trade_date": 0}])
def test_malformed_numeric_and_date_fields_return_client_error(tmp_path, changes):
    runtime = _make_runtime(tmp_path)
    status, _ = _call("POST", "/trades", runtime,
        {"ts_code": "000001.SZ", "action": "WAIT", **changes})
    assert status == 400 and not runtime.journal.list()


def test_blank_decision_price_is_normalized_for_audit_and_profile(tmp_path):
    runtime = _make_runtime(tmp_path)
    status, response = _call("POST", "/trades", runtime,
        {"ts_code": "000001.SZ", "action": "WAIT", "price": "", "trade_date": None})
    assert status == 201 and response["data"]["price"] is None
    assert response["data"]["trade_date"] == ""
    assert _call("POST", f"/trades/{response['data']['trade_id']}/audit", runtime)[0] == 200
