"""V1.2 personal watchlist: at most 10 stocks, validated codes, durable JSON."""

from __future__ import annotations

import json

import pytest

from backend.personal import WATCHLIST_LIMIT, WatchlistStore
from tests.test_api_v1 import _call, _make_runtime


def test_limit_is_ten():
    assert WATCHLIST_LIMIT == 10


def test_empty_when_no_file(tmp_path):
    store = WatchlistStore(tmp_path)
    assert store.list() == []
    assert not store.path.exists()


def test_add_normalizes_code_and_persists(tmp_path):
    store = WatchlistStore(tmp_path)
    entry = store.add(" 600519.sh ", name="贵州茅台", note="周线观察")
    assert entry["ts_code"] == "600519.SH"
    assert entry["name"] == "贵州茅台" and entry["note"] == "周线观察"
    assert entry["added_at"]
    assert [i["ts_code"] for i in WatchlistStore(tmp_path).list()] == ["600519.SH"]
    data = json.loads(store.path.read_text(encoding="utf-8"))
    assert list(data) == ["items", "closed"]


def test_duplicate_rejected(tmp_path):
    store = WatchlistStore(tmp_path)
    store.add("000001.SZ")
    with pytest.raises(ValueError, match="已在观察池中"):
        store.add("000001.sz")
    assert len(store.list()) == 1


def test_eleventh_stock_rejected_and_file_unchanged(tmp_path):
    store = WatchlistStore(tmp_path)
    for n in range(WATCHLIST_LIMIT):
        store.add(f"60000{n}.SH")
    before = store.path.read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="最多 10 只"):
        store.add("000001.SZ")
    assert store.path.read_text(encoding="utf-8") == before
    assert len(store.list()) == WATCHLIST_LIMIT


def test_remove_frees_a_slot(tmp_path):
    store = WatchlistStore(tmp_path)
    for n in range(WATCHLIST_LIMIT):
        store.add(f"60000{n}.SH")
    assert store.remove("600000.sh") is True
    assert store.remove("600000.SH") is False
    store.add("000001.SZ")
    assert len(store.list()) == WATCHLIST_LIMIT


@pytest.mark.parametrize("bad", ["", "600519", "600519.HK", "ABC123.SZ", None, 600519, "1234567.SH"])
def test_invalid_codes_rejected(tmp_path, bad):
    store = WatchlistStore(tmp_path)
    with pytest.raises(ValueError, match="股票代码"):
        store.add(bad)
    assert store.list() == []


def test_text_length_limits(tmp_path):
    store = WatchlistStore(tmp_path)
    with pytest.raises(ValueError, match="name"):
        store.add("000001.SZ", name="x" * 41)
    with pytest.raises(ValueError, match="note"):
        store.add("000001.SZ", note="x" * 501)


def test_corrupt_file_is_reported_not_overwritten(tmp_path):
    store = WatchlistStore(tmp_path)
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.path.write_text('{"items": "oops"}', encoding="utf-8")
    with pytest.raises(ValueError, match="格式错误"):
        store.add("000001.SZ")
    assert store.path.read_text(encoding="utf-8") == '{"items": "oops"}'


def test_watchlist_api(tmp_path):
    runtime = _make_runtime(tmp_path)
    code, data = _call("GET", "/personal/watchlist", runtime)
    assert code == 200 and data["data"] == {"items": [], "count": 0}
    assert data["meta"]["limit"] == WATCHLIST_LIMIT

    code, data = _call("POST", "/personal/watchlist", runtime, {"ts_code": "000001.sz", "name": "平安银行"})
    assert code == 200 and data["data"]["ts_code"] == "000001.SZ"

    code, data = _call("POST", "/personal/watchlist", runtime, {"ts_code": "000001.SZ"})
    assert code == 400 and "已在观察池中" in data["detail"]

    code, data = _call("GET", "/personal/watchlist", runtime)
    assert data["data"]["count"] == 1

    code, data = _call("POST", "/personal/watchlist/remove", runtime, {"ts_code": "000001.SZ"})
    assert code == 200 and data["data"]["removed"] == "000001.SZ"

    code, data = _call("POST", "/personal/watchlist/remove", runtime, {"ts_code": "000001.SZ"})
    assert code == 404
