"""Durable account transactions against stale readers and real worker processes."""
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path
import threading

import pytest

from backend.data.atomic import file_lock
from backend.journal.trades import TradeJournal, TradeRecord
from backend.paper.engine import PaperTradingEngine
from backend.strategies.base import TradeSignal


def _signal(identity="", *, code="000001.SZ"):
    return TradeSignal(ts_code=code, trade_date="20240103", action="BUY",
                       price=10.0, quantity=100, signal_id=identity)


def _record(identity, quantity=100):
    return replace(TradeRecord.create(symbol="000001", ts_code="000001.SZ",
        action="BUY", price=10, quantity=quantity, position_before=0,
        position_after=quantity), trade_id=identity)


def _account_worker(root, index, ready, start, results):
    try:
        root = Path(root)
        engine = PaperTradingEngine(initial_cash=100000, commission_rate=0,
            state_path=root / "state.json", journal=TradeJournal(root))
        ready.put(index)
        if not start.wait(15):
            raise TimeoutError("worker start")
        engine.execute_signal(_signal("shared"))
        engine.execute_signal(_signal(f"worker-{index}"))
        results.put((index, "ok"))
    except Exception as exc:
        results.put((index, repr(exc)))


def _journal_worker(root, index, ready, start, results):
    try:
        journal = TradeJournal(Path(root))
        ready.put(index)
        if not start.wait(15):
            raise TimeoutError("worker start")
        journal.record(replace(_record("shared"), timestamp=f"2024-01-03T10:00:0{index}Z"))
        journal.record(_record(f"worker-{index}"))
        results.put((index, "ok"))
    except Exception as exc:
        results.put((index, repr(exc)))


def _run_workers(worker, root, count=3):
    # spawn is used on every OS: every worker owns an independently loaded
    # ledger and no inherited Python RLock can hide an OS locking defect.
    context = multiprocessing.get_context("spawn")
    ready, results, start = context.Queue(), context.Queue(), context.Event()
    processes = [context.Process(target=worker,
        args=(str(root), index, ready, start, results)) for index in range(count)]
    try:
        for process in processes:
            process.start()
        assert sorted(ready.get(timeout=20) for _ in processes) == list(range(count))
        start.set()
        outcomes = [results.get(timeout=20) for _ in processes]
        assert sorted(outcomes) == [(index, "ok") for index in range(count)]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        start.set()
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        ready.close()
        results.close()


def test_multiple_account_processes_keep_all_fills_and_deduplicate(tmp_path):
    _run_workers(_account_worker, tmp_path)
    engine = PaperTradingEngine(state_path=tmp_path / "state.json", journal=TradeJournal(tmp_path))
    snapshot = engine.snapshot()
    assert snapshot.cash == 96000
    assert snapshot.positions["000001.SZ"].quantity == 400
    assert len(snapshot.orders) == len(engine.requests) == len(engine.trade_records) == 4
    assert snapshot.version == 4
    assert len(TradeJournal(tmp_path).list()) == 4


def test_multiple_journal_processes_keep_each_event_once(tmp_path):
    _run_workers(_journal_worker, tmp_path)
    records = TradeJournal(tmp_path).list()
    assert {record.trade_id for record in records} == {"shared", "worker-0", "worker-1", "worker-2"}
    assert len(records) == 4


def test_stale_instance_reloads_before_peak_equity_write(tmp_path):
    path = tmp_path / "state.json"
    first = PaperTradingEngine(state_path=path, commission_rate=0)
    second = PaperTradingEngine(state_path=path, commission_rate=0)
    first.execute_signal(_signal("first"))
    snapshot = second.snapshot({"000001.SZ": 20})
    assert snapshot.positions["000001.SZ"].quantity == 100
    assert snapshot.peak_equity == 1001000
    second.execute_signal(_signal("second"))
    assert first.order_for_signal("second") == second.orders[-1]
    assert first.snapshot().positions["000001.SZ"].quantity == 200
    assert len(PaperTradingEngine(state_path=path).orders) == 2


@pytest.mark.parametrize("payload", [[], {},
    {"cash": True, "initial_cash": 10000, "positions": {}, "orders": []},
    {"cash": 10000, "initial_cash": 10000, "positions": [], "orders": []},
    {"cash": 10000, "initial_cash": 10000, "positions": {}, "orders": [], "version": -1},
    {"cash": 10000, "initial_cash": 10000, "positions": {}, "orders": [], "trade_records": [{}]},
    {"cash": 10000, "initial_cash": 10000, "positions": {"000001.SZ": {
        "quantity": 100, "avg_price": 10, "buy_date": "20240102", "buy_lots": []}}, "orders": []},
    {"cash": 10000, "initial_cash": 10000, "positions": {"000001.SZ": {
        "ts_code": "600000.SH", "quantity": 100, "avg_price": 10, "buy_date": "20240102"}}, "orders": []},
])
def test_invalid_account_shape_fails_closed_without_rewriting(tmp_path, payload):
    path = tmp_path / "state.json"
    original = json.dumps(payload)
    path.write_text(original, encoding="utf-8")
    with pytest.raises(RuntimeError, match="恢复"):
        PaperTradingEngine(state_path=path)
    assert path.read_text(encoding="utf-8") == original


def test_failed_reload_keeps_the_previous_complete_memory_state(tmp_path):
    path = tmp_path / "state.json"
    engine = PaperTradingEngine(state_path=path)
    engine.execute_signal(_signal("first"))
    before = path.read_text(encoding="utf-8")
    cash = engine.cash
    invalid = json.loads(before)
    invalid["cash"] = 1
    invalid["orders"][0]["quantity"] = -100
    path.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(RuntimeError, match="恢复"):
        engine.execute_signal(_signal("second"))
    assert engine.cash == cash and len(engine.orders) == 1
    path.write_text(before, encoding="utf-8")
    assert engine.snapshot().cash == cash


def test_removed_account_file_is_not_silently_recreated(tmp_path):
    path = tmp_path / "state.json"
    engine = PaperTradingEngine(state_path=path)
    engine.execute_signal(_signal("first"))
    path.unlink()
    with pytest.raises(RuntimeError, match="丢失"):
        engine.execute_signal(_signal("second"))
    assert not path.exists()
    assert len(engine.orders) == 1


def test_malformed_journal_cannot_undo_a_committed_fill(tmp_path):
    journal = TradeJournal(tmp_path)
    journal.path.parent.mkdir(parents=True)
    journal.path.write_text('{"trade_id": "incomplete"}\n', encoding="utf-8")
    engine = PaperTradingEngine(state_path=tmp_path / "state.json", journal=journal)
    assert engine.execute_signal(_signal("first")).status == "FILLED"
    assert engine.journal_pending
    restored = PaperTradingEngine(state_path=tmp_path / "state.json")
    assert len(restored.orders) == len(engine.orders) == 1
    assert restored.cash == engine.cash
    journal.path.write_text("", encoding="utf-8")
    engine.sync_journal()
    assert not engine.journal_pending and len(journal.list()) == 1


def test_journal_retries_ignore_only_timestamp_and_reject_conflicting_payload(tmp_path):
    journal = TradeJournal(tmp_path)
    first = _record("request-1")
    assert journal.record(first) == first
    assert journal.record(replace(first, timestamp="later")) == first
    with pytest.raises(ValueError, match="trade_id"):
        journal.record(replace(first, quantity=200))
    assert journal.list() == [first]


def _hold_lock(path, ready):
    with file_lock(Path(path)):
        ready.set()
        multiprocessing.get_context("spawn").Event().wait(30)


def test_worker_exit_releases_file_lock_without_deleting_sidecar(tmp_path):
    context = multiprocessing.get_context("spawn")
    path = tmp_path / "state.json"
    ready = context.Event()
    process = context.Process(target=_hold_lock, args=(str(path), ready))
    try:
        process.start()
        assert ready.wait(20)
        process.terminate()
        process.join(timeout=10)
        with file_lock(path, timeout=1):
            assert path.with_name("state.json.lock").exists()
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)


def test_nested_file_lock_reuses_the_owning_threads_os_lock(tmp_path):
    path = tmp_path / "state.json"
    with file_lock(path):
        with file_lock(path, timeout=0):
            assert path.with_name("state.json.lock").exists()
    # Both the OS handle and the local RLock must be released by the outer
    # context even when a nested caller returns early.
    with file_lock(path, timeout=0):
        pass


def test_file_lock_timeout_also_applies_to_another_thread(tmp_path):
    path = tmp_path / "state.json"
    ready, finish = threading.Event(), threading.Event()

    def holder():
        with file_lock(path):
            ready.set()
            finish.wait(5)

    worker = threading.Thread(target=holder)
    worker.start()
    try:
        assert ready.wait(2)
        with pytest.raises(TimeoutError):
            with file_lock(path, timeout=0.05):
                pytest.fail("another thread already owns the account lock")
    finally:
        finish.set()
        worker.join(timeout=2)
    assert not worker.is_alive()
    with file_lock(path, timeout=0):
        pass


def test_conflicting_existing_journal_event_remains_pending(tmp_path):
    journal = TradeJournal(tmp_path)
    path = tmp_path / "state.json"
    engine = PaperTradingEngine(state_path=path)
    engine.execute_signal(_signal("filled"))
    durable = TradeRecord.from_dict(engine.trade_records[0])
    journal.record(replace(durable, quantity=999))
    before = path.read_text(encoding="utf-8")
    engine.journal = journal
    engine.sync_journal()
    assert engine.journal_pending
    assert path.read_text(encoding="utf-8") == before
    assert journal.list()[0].quantity == 999
    # Repair the projection from the durable source, then recovery converges.
    journal.path.write_text("", encoding="utf-8")
    engine.sync_journal()
    assert not engine.journal_pending
    assert journal.list() == [durable]


@pytest.mark.parametrize("changes", [{"ts_code": ""}, {"ts_code": None},
    {"signal_id": None}, {"quantity": True}, {"price": True}])
def test_invalid_signal_identity_cannot_create_unloadable_account(tmp_path, changes):
    path = tmp_path / "state.json"
    engine = PaperTradingEngine(state_path=path)
    with pytest.raises(ValueError):
        engine.execute_signal(replace(_signal("bad"), **changes))
    assert not path.exists() and not engine.orders


def test_numeric_quantity_is_persisted_in_normalized_request(tmp_path):
    path = tmp_path / "state.json"
    engine = PaperTradingEngine(state_path=path)
    signal = replace(_signal("normalized"), quantity="100")
    first = engine.execute_signal(signal)
    restored = PaperTradingEngine(state_path=path)
    assert restored.execute_signal(replace(signal, quantity=100)) == first
    assert len(restored.orders) == 1
