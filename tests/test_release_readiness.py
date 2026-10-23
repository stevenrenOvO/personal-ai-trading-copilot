"""Release status must distinguish retry work from unavailable analysis data."""
from types import SimpleNamespace

import pandas as pd
import pytest

from backend.api.readiness import readiness
from backend.data.history import MarketHistory


@pytest.fixture
def ready_runtime(tmp_path, monkeypatch):
    from backend.api import data_status
    from backend.data import auto_ingest
    from backend.market import quote_service, session

    history = MarketHistory(tmp_path / "history.db")
    history.ingest(pd.DataFrame([
        {"ts_code": code, "trade_date": "20261023", "open": 10, "high": 11, "low": 9, "close": 10}
        for code in ("000300.SH", "000905.SH", "399006.SZ")
    ]))
    runtime = SimpleNamespace(
        provider=SimpleNamespace(name="local_history", history=history),
        paper=SimpleNamespace(snapshot=lambda: SimpleNamespace(journal_pending=False)),
    )
    info = {"is_trading_day": True, "phase": "盘后", "current_trading_day": "20261023", "expected_daily_date": "20261023"}
    monkeypatch.setattr(session, "get_trading_session", lambda: SimpleNamespace(describe_cached=lambda: info))
    meta = SimpleNamespace(quote_date="20261023", quote_time="15:10:00", is_stale=False, source="tencent", rows=5217, status="final")
    monkeypatch.setattr(quote_service, "get_quote_service", lambda: SimpleNamespace(get=lambda: (pd.DataFrame(), meta)))
    dates = [{"date": "20261023", "valid_rows": 5207, "coverage_ratio": .998, "complete": True}]
    monkeypatch.setattr(data_status, "history_dates", lambda _: dates)
    ingest = {"running": True, "pending": {"20261023": {"failed": 10}}, "progress": {}}
    monkeypatch.setattr(auto_ingest, "status", lambda _: ingest)
    return runtime, dates, ingest


def test_sufficient_coverage_retries_do_not_mark_service_degraded(ready_runtime):
    runtime, _, _ = ready_runtime
    result = readiness(runtime)
    assert result["status"] == "ready"
    assert result["checks"]["ingest"]["minor_missing_dates"] == ["20261023"]
    assert result["checks"]["ingest"]["critical_gap_dates"] == []
    assert "99.8%" in result["checks"]["history"]["message"]
    assert "少量缺失" in result["checks"]["ingest"]["message"]
    assert result["data_mode"] == "production"


def test_critical_date_gap_still_marks_service_degraded(ready_runtime):
    runtime, dates, ingest = ready_runtime
    dates.append({"date": "20261022", "valid_rows": 100, "coverage_ratio": .02, "complete": False})
    ingest["pending"]["20261022"] = {"failed": 5000}
    result = readiness(runtime)
    assert result["status"] == "degraded"
    assert result["checks"]["ingest"]["critical_gap_dates"] == ["20261022"]
    assert result["checks"]["ingest"]["minor_missing_dates"] == ["20261023"]


def test_future_complete_day_does_not_hide_missing_expected_day(ready_runtime):
    runtime, dates, _ = ready_runtime
    dates[:] = [{"date": "20261026", "valid_rows": 5217, "coverage_ratio": 1, "complete": True}]
    result = readiness(runtime)
    assert result["checks"]["history"]["status"] == "warning"
    assert result["checks"]["history"]["latest_complete_date"] is None


def test_old_or_invalid_index_bars_do_not_count_as_today(ready_runtime):
    runtime, _, _ = ready_runtime
    with runtime.provider.history._connect() as conn:
        conn.execute("UPDATE daily SET trade_date='20261022' WHERE ts_code='000905.SH'")
        conn.execute("UPDATE daily SET close=0 WHERE ts_code='399006.SZ'")
    check = readiness(runtime)["checks"]["indexes"]
    assert check["count"] == 1 and check["expected_count"] == 3
    assert check["latest_dates"]["000905.SH"] == "20261022"
    assert check["status"] == "warning"
    assert "旧日期不计入当日指数" in check["message"]
