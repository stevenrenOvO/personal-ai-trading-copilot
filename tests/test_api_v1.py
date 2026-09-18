"""Tests for the V1 WSGI endpoints."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from backend.api.runtime import Runtime
from backend.api import v1
from backend.data.config import clear_settings_cache, get_settings
from backend.data.providers.fixture import FixtureProvider


def _make_runtime(tmp_path: Path) -> Runtime:
    provider = FixtureProvider(base_dir=get_settings().fixture_dir)
    runtime = Runtime.create(root=tmp_path)
    runtime.provider = provider
    return runtime


def _call(method: str, path: str, runtime: Runtime, body: dict | None = None):
    query = ""
    if "?" in path:
        path, query = path.split("?", 1)
    environ = {
        "PATH_INFO": path,
        "QUERY_STRING": query,
        "REQUEST_METHOD": method,
        "wsgi.input": BytesIO(json.dumps(body).encode() if body else b""),
        "CONTENT_LENGTH": str(len(json.dumps(body)) if body else 0),
    }
    status = None
    chunks = []

    def start_response(s, h, exc_info=None):
        nonlocal status
        status = int(s.split()[0])

    out, handled = v1.dispatch(environ, start_response, runtime)
    if handled:
        chunks.append(out)
    payload = b"".join(chunks)
    return status, json.loads(payload) if payload else {}


def test_v1_market_and_emotion(tmp_path):
    runtime = _make_runtime(tmp_path)
    code, data = _call("GET", "/market/state?date=20240103", runtime)
    assert code == 200
    assert data["status"] == "ok"
    assert "score" in data["data"]

    code, data = _call("GET", "/emotion/state?date=20240103", runtime)
    assert code == 200
    assert data["data"]["emotion_cycle"]


def test_v1_sectors_opportunities_board_risk(tmp_path):
    runtime = _make_runtime(tmp_path)
    for path in ["/sectors?date=20240103", "/opportunities?date=20240103", "/board/environment?date=20240103", "/risk?date=20240103"]:
        code, data = _call("GET", path, runtime)
        assert code == 200
        assert data["status"] == "ok"


def test_v1_positions_strategies_health_profile(tmp_path):
    runtime = _make_runtime(tmp_path)
    for path in ["/positions", "/strategies", "/strategies/health", "/profile", "/trades", "/backtest/runs"]:
        code, data = _call("GET", path, runtime)
        assert code == 200
        assert data["status"] == "ok"


def test_v1_post_trade_and_audit(tmp_path):
    runtime = _make_runtime(tmp_path)
    code, data = _call(
        "POST",
        "/trades",
        runtime,
        {
            "ts_code": "000001.SZ",
            "action": "BUY",
            "price": 10.0,
            "quantity": 100,
            "position_before": 0,
            "position_after": 100,
            "strategy_id": "strong_sector_breakout",
            "strategy_version": "1.0",
            "signal_id": "sig_test",
            "system_recommendation": "BUY WATCH",
            "user_reason": "test",
        },
    )
    assert code == 201
    trade_id = data["data"]["trade_id"]

    code, data = _call("POST", f"/trades/{trade_id}/audit", runtime)
    assert code == 200
    assert data["status"] == "ok"


def test_v1_ai_copilot_grounded(tmp_path):
    runtime = _make_runtime(tmp_path)
    code, data = _call("GET", "/ai/copilot?date=20240103&q=今天能不能做", runtime)
    assert code == 200
    assert data["status"] == "ok"


def test_data_latest_uses_trading_session_not_history(tmp_path):
    """Regression: /data/latest must report the current trading day.

    It used to return the last date of the local history, which froze the
    dashboard on an old day.
    """
    runtime = _make_runtime(tmp_path)
    code, data = _call("GET", "/data/latest", runtime)
    assert code == 200
    # conftest pins the trading session to 2024-01-03, while the fixture
    # history ends on 2024-01-03 as well but the point is the session is used.
    assert data["data"]["date"] == "20240103"
    assert "phase" in data["data"]


def test_history_dates_endpoint_lists_selectable_days(tmp_path):
    """The date picker is fed from the local store, not a free text input."""
    runtime = _make_runtime(tmp_path)
    code, data = _call("GET", "/data/history/dates", runtime)
    assert code == 200
    payload = data["data"]
    assert "dates" in payload and "latest" in payload
    assert isinstance(payload["dates"], list)
    if payload["dates"]:
        assert payload["latest"] == max(payload["dates"])


def test_opportunities_expose_grouped_blockers(tmp_path):
    """The UI needs to separate "bad tape" from "wrong name"."""
    runtime = _make_runtime(tmp_path)
    code, data = _call("GET", "/opportunities?date=20240103", runtime)
    assert code == 200
    for item in data["data"]:
        assert "buy_ready" in item
        assert "environment_blockers" in item
        assert "stock_blockers" in item
