"""Backtest history survives restarts, concurrent writers and corrupt files."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import multiprocessing
from pathlib import Path

import pytest

from backend.api.backtest_store import BacktestRunStore


def _run(index=0):
    return {
        "run_id": f"run_{index}",
        "created_at": (datetime(2026, 10, 23, tzinfo=timezone.utc) + timedelta(seconds=index)).isoformat(),
        "strategy_id": "ma_cross", "strategy_version": "1.0",
        "metrics": {"total_return": 0.05, "win_rate": None},
        "trade_log": [{"ts_code": "000001.SZ", "pnl": 50.0}],
        "equity_curve": [{"date": "20261023", "total": 100050}],
        "drawdown_curve": [{"date": "20261023", "drawdown": -0.01}],
        "lookahead_issues": [],
    }


def test_complete_run_survives_restart_and_caller_mutation(tmp_path):
    store = BacktestRunStore(tmp_path)
    original = _run()
    saved = store.save(original)
    original["metrics"]["total_return"] = 999
    saved["trade_log"].clear()
    restored = BacktestRunStore(tmp_path)
    assert restored.get("run_0") == _run()
    assert restored.list() == [_run()]
    assert restored.get("missing") is None
    assert restored.save(_run()) == _run()


def test_list_orders_actual_timestamps_and_keeps_all_results(tmp_path):
    store = BacktestRunStore(tmp_path)
    for index in reversed(range(60)):
        store.save(_run(index))
    assert [run["run_id"] for run in store.list()] == [f"run_{index}" for index in reversed(range(10, 60))]
    assert len(store.list(limit=100)) == 50
    assert len(store.list(limit=3)) == 3
    assert store.list(limit=0) == []
    assert store.get("run_0") == _run(0)
    assert len(list(store.path.glob("*.json"))) == 60
    newer = _run(60)
    newer["created_at"] = "2026-10-23T08:02:00+08:00"
    store.save(newer)
    assert store.list()[0] == newer


@pytest.mark.parametrize("content", ['{', '[]', '{}', '{"run_id":"other","created_at":"2026-10-23T00:00:00Z"}',
    '{"run_id":"run_0","created_at":"bad"}',
    '{"run_id":"run_0","created_at":"2026-10-23T00:00:00Z","metrics":{"win_rate":NaN}}',
    '{"run_id":"run_0","created_at":"2026-10-23T00:00:00Z","metrics":{"win_rate":1e999}}'])
def test_bad_record_is_reported_and_never_replaced(tmp_path, content):
    store = BacktestRunStore(tmp_path)
    store.path.mkdir(parents=True)
    path = store.path / "run_0.json"
    path.write_text(content, encoding="utf-8")
    for operation in (lambda: store.get("run_0"), lambda: store.list(), lambda: store.save(_run())):
        with pytest.raises(RuntimeError, match="run_0.json"):
            operation()
        assert path.read_text(encoding="utf-8") == content


def test_conflicting_id_cannot_change_saved_results(tmp_path):
    store = BacktestRunStore(tmp_path)
    original = _run()
    store.save(original)
    conflicting = _run()
    conflicting["metrics"]["total_return"] = -1
    with pytest.raises(ValueError, match="run_id"):
        store.save(conflicting)
    assert store.get("run_0") == original


@pytest.mark.parametrize("run_id", ["../outside", "a/b", "a\\b", "", None, "a.json"])
def test_run_identity_cannot_escape_the_store(tmp_path, run_id):
    store = BacktestRunStore(tmp_path)
    run = _run()
    run["run_id"] = run_id
    with pytest.raises(ValueError):
        store.save(run)
    with pytest.raises(ValueError):
        store.get(run_id)
    assert not store.path.exists()


def _worker_save(root, index):
    store = BacktestRunStore(Path(root))
    store.save(_run())
    for offset in range(5):
        store.save(_run(index * 5 + offset + 1))


def test_independent_processes_keep_every_run_and_replay_once(tmp_path):
    context = multiprocessing.get_context("spawn")
    workers = [context.Process(target=_worker_save, args=(str(tmp_path), index)) for index in range(3)]
    try:
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=20)
            assert worker.exitcode == 0
        store = BacktestRunStore(tmp_path)
        assert len(store.list()) == len(list(store.path.glob("*.json"))) == 16
        assert {run["run_id"] for run in store.list()} == {f"run_{index}" for index in range(16)}
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)


def test_concurrent_conflicting_saves_choose_one_immutable_result(tmp_path):
    def save(index):
        run = _run()
        run["metrics"]["total_return"] = index
        try:
            BacktestRunStore(tmp_path).save(run)
            return index
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(save, range(8)))
    successes = [result for result in outcomes if result is not None]
    assert len(successes) == 1
    assert BacktestRunStore(tmp_path).get("run_0")["metrics"]["total_return"] == successes[0]
