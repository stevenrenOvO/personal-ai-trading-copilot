"""Tests for the intraday quote cache service (Step 1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from backend.market.quote_service import QuoteService


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH"],
            "name": ["平安银行", "浦发银行"],
            "price": [10.0, 6.6],
            "change": [0.1, -0.05],
            "pct_chg": [1.0, -0.75],
            "pre_close": [9.9, 6.65],
            "open": [9.95, 6.62],
            "high": [10.1, 6.68],
            "low": [9.9, 6.55],
            "volume": [1000.0, 2000.0],
            "amount": [10000.0, 13000.0],
        }
    )


def _ok_fetcher(frame: pd.DataFrame):
    def fetch():
        return frame, {
            "source": "sina",
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "quote_time": "15:00:00",
            "rows": len(frame),
        }

    return fetch


def test_cold_start_does_not_block(tmp_path):
    service = QuoteService(db_path=tmp_path / "q.db", fetcher=_ok_fetcher(_frame()))
    df, meta = service.get()  # no fetch triggered synchronously
    assert df.empty
    assert meta.status == "warming"
    assert meta.is_stale is True
    assert meta.rows == 0


def test_refresh_then_fresh_read(tmp_path):
    service = QuoteService(db_path=tmp_path / "q.db", ttl_seconds=60, fetcher=_ok_fetcher(_frame()))
    meta = service.refresh()
    assert meta.status == "ok"
    assert meta.rows == 2
    assert meta.is_stale is False

    df, read_meta = service.get()
    assert len(df) == 2
    assert read_meta.status == "ok"
    assert read_meta.as_of is not None


def test_stale_marking(tmp_path):
    service = QuoteService(db_path=tmp_path / "q.db", ttl_seconds=0, fetcher=_ok_fetcher(_frame()))
    service.refresh()
    df, meta = service.get()
    assert meta.is_stale is True
    assert meta.status == "stale"
    assert len(df) == 2  # cached data is still served, just marked stale


def _fetcher_with_quote_time(frame: pd.DataFrame, quote_time: str):
    def fetch():
        return frame, {
            "source": "tencent",
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "quote_time": quote_time,
            "rows": len(frame),
        }

    return fetch


def _pin_session(tmp_path, when: datetime) -> None:
    from backend.market.session import TradingSession, set_trading_session

    set_trading_session(
        TradingSession(
            db_path=tmp_path / "session.db",
            calendar_fetcher=lambda start, end: [("20240102", 1), ("20240103", 1)],
            now_fn=lambda: when,
        )
    )


def test_snapshot_is_final_after_close(tmp_path):
    """After the close today's snapshot is the settled value, not "stale"."""
    _pin_session(tmp_path, datetime(2024, 1, 3, 16, 0))
    service = QuoteService(
        db_path=tmp_path / "q.db",
        ttl_seconds=0,
        fetcher=_fetcher_with_quote_time(_frame(), "20240103150000"),
    )
    service.refresh()
    df, meta = service.get()
    assert meta.status == "final"
    assert meta.is_stale is False
    assert len(df) == 2


def test_previous_day_snapshot_is_still_stale_after_close(tmp_path):
    _pin_session(tmp_path, datetime(2024, 1, 3, 16, 0))
    service = QuoteService(
        db_path=tmp_path / "q.db",
        ttl_seconds=0,
        fetcher=_fetcher_with_quote_time(_frame(), "20240102150000"),
    )
    service.refresh()
    _df, meta = service.get()
    assert meta.status == "stale"
    assert meta.is_stale is True


def test_failure_keeps_old_data_and_reports_error(tmp_path):
    service = QuoteService(db_path=tmp_path / "q.db", ttl_seconds=60, fetcher=_ok_fetcher(_frame()))
    service.refresh()

    def boom():
        raise RuntimeError("vendor unavailable")

    service._fetch = boom
    meta = service.refresh()
    assert meta.status == "error"
    assert "vendor unavailable" in (meta.error or "")
    assert meta.is_stale is True

    df, read_meta = service.get()
    assert len(df) == 2  # previous data retained, clearly marked
    assert read_meta.status == "error"


def test_persistence_survives_restart(tmp_path):
    db = tmp_path / "q.db"
    first = QuoteService(db_path=db, ttl_seconds=60, fetcher=_ok_fetcher(_frame()))
    first.refresh()

    # New instance, no in-memory state, must load from the persisted cache.
    second = QuoteService(db_path=db, ttl_seconds=60, fetcher=_ok_fetcher(_frame()))
    df, meta = second.get()
    assert len(df) == 2
    assert meta.rows == 2
    assert meta.as_of is not None


def test_age_seconds_is_positive_and_flags_stale(tmp_path):
    def old_fetch():
        return _frame(), {
            "source": "sina",
            "fetched_at": (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat(
                timespec="seconds"
            ),
            "quote_time": "",
            "rows": 2,
        }

    service = QuoteService(db_path=tmp_path / "q.db", ttl_seconds=30, fetcher=old_fetch)
    meta = service.refresh()
    assert meta.age_seconds is not None
    assert meta.age_seconds > 100, "age must be positive for a UTC timestamp in the past"
    assert meta.is_stale is True
    assert meta.status == "stale"


def test_failure_counter_drives_backoff_and_resets(tmp_path):
    service = QuoteService(db_path=tmp_path / "q.db", fetcher=_ok_fetcher(_frame()))
    service.refresh()
    assert service._fail_count == 0

    def boom():
        raise RuntimeError("blocked")

    service._fetch = boom
    service.refresh()
    service.refresh()
    assert service._fail_count == 2, "repeated failures must increase the backoff"

    service._fetch = _ok_fetcher(_frame())
    service.refresh()
    assert service._fail_count == 0, "a successful refresh resets the backoff"
