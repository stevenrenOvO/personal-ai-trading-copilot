"""Small thread-safe TTL memo used to keep the dashboard responsive.

The engines are pure functions of (provider, date) but each one costs one to
four seconds on a full-market day: the dashboard asked for five of them on
every tab switch, and the AI panel asked for the same five again. A short TTL
cache makes the first load pay the cost and every later refresh nearly free.

It is safe here because every response is labelled with its own freshness
metadata and the underlying store only changes when a backfill runs; the TTL
is the explicit bound on how long a stale value could survive.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")


class TTLCache:
    """Keyed values with a time-to-live and a bounded number of entries."""

    def __init__(self, ttl_seconds: float, maxsize: int = 64) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds cannot be negative")
        if maxsize <= 0:
            raise ValueError("maxsize must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self.maxsize = int(maxsize)
        self._entries: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._flights: dict[Any, Future] = {}
        self._generation = 0

    def get(self, key: Any) -> Optional[Any]:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, value = entry
            if time.monotonic() - stored_at > self.ttl_seconds:
                del self._entries[key]
                return None
            return value

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            self._entries[key] = (time.monotonic(), value)
            while len(self._entries) > self.maxsize:
                oldest = min(self._entries, key=lambda k: self._entries[k][0])
                del self._entries[oldest]

    def get_or_compute(self, key: Any, factory: Callable[[], T]) -> T:
        """Share one computation per key; invalidation cannot revive old entries."""
        cached = self.get(key)
        if cached is not None:
            return cached
        with self._lock:
            generation = self._generation
            flight_key = (generation, key)
            future = self._flights.get(flight_key)
            owner = future is None
            if owner:
                future = Future()
                self._flights[flight_key] = future
        if not owner:
            return future.result()
        try:
            # Recheck after becoming the owner: another finished computation
            # may have published between the initial get and this lock.
            value = self.get(key)
            if value is None:
                value = factory()
            with self._lock:
                if generation == self._generation:
                    self._entries[key] = (time.monotonic(), value)
                    while len(self._entries) > self.maxsize:
                        oldest = min(self._entries, key=lambda k: self._entries[k][0])
                        del self._entries[oldest]
            future.set_result(value)
            return value
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._flights.pop(flight_key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._generation += 1
