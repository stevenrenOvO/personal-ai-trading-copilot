"""Tests for the short-TTL memo used by the API layer."""

from __future__ import annotations

import time

import pytest

from backend.api.memo import TTLCache


def test_get_returns_none_for_missing_key():
    cache = TTLCache(ttl_seconds=60.0)
    assert cache.get("nope") is None


def test_put_then_get_roundtrip():
    cache = TTLCache(ttl_seconds=60.0)
    cache.put("a", [1, 2, 3])
    assert cache.get("a") == [1, 2, 3]


def test_values_expire_after_ttl():
    cache = TTLCache(ttl_seconds=0.05)
    cache.put("a", "value")
    assert cache.get("a") == "value"
    time.sleep(0.08)
    assert cache.get("a") is None


def test_get_or_compute_only_runs_the_factory_once():
    cache = TTLCache(ttl_seconds=60.0)
    calls: list[int] = []

    def factory() -> int:
        calls.append(1)
        return 42

    assert cache.get_or_compute("k", factory) == 42
    assert cache.get_or_compute("k", factory) == 42
    assert len(calls) == 1


def test_cache_evicts_oldest_beyond_maxsize():
    cache = TTLCache(ttl_seconds=60.0, maxsize=2)
    cache.put("a", 1)
    time.sleep(0.01)
    cache.put("b", 2)
    time.sleep(0.01)
    cache.put("c", 3)
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_clear_drops_everything():
    cache = TTLCache(ttl_seconds=60.0)
    cache.put("a", 1)
    cache.clear()
    assert cache.get("a") is None


def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        TTLCache(ttl_seconds=-1)
    with pytest.raises(ValueError):
        TTLCache(ttl_seconds=1.0, maxsize=0)


def test_runtime_gets_its_own_caches(tmp_path):
    """Caches must live on the runtime, so tests cannot leak into each other."""
    from backend.api.runtime import Runtime

    first = Runtime.create(root=tmp_path / "a")
    second = Runtime.create(root=tmp_path / "b")
    first.analysis_cache.put(("market", "20240103"), "cached")
    assert second.analysis_cache.get(("market", "20240103")) is None
