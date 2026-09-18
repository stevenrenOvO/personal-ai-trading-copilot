"""Tests for the WSGI API layer."""

from __future__ import annotations

import json
from wsgiref.util import setup_testing_defaults

import pytest
from backend.api.app import application


def _call_app(environ: dict) -> tuple[int, list, bytes]:
    """Call the WSGI app and return status code, headers, and body."""
    setup_testing_defaults(environ)
    status_code = None
    headers = None
    body_chunks = []

    def start_response(status, response_headers, exc_info=None):
        nonlocal status_code, headers
        status_code = int(status.split()[0])
        headers = response_headers
        def write(data):
            body_chunks.append(data)
        return write

    response = application(environ, start_response)
    for chunk in response:
        body_chunks.append(chunk)

    return status_code, headers, b"".join(body_chunks)


def test_health():
    environ = {"PATH_INFO": "/health", "REQUEST_METHOD": "GET"}
    code, headers, body = _call_app(environ)
    assert code == 200
    data = json.loads(body.decode("utf-8"))
    assert data["status"] == "ok"


def test_root_health():
    environ = {"PATH_INFO": "/", "REQUEST_METHOD": "GET"}
    code, headers, body = _call_app(environ)
    assert code == 200
    content_type = dict(headers).get("Content-Type", "")
    assert content_type.startswith("text/html")
    page = body.decode("utf-8-sig")
    assert "<!DOCTYPE html>" in page


def test_not_found():
    environ = {"PATH_INFO": "/unknown", "REQUEST_METHOD": "GET"}
    code, headers, body = _call_app(environ)
    assert code == 404
    data = json.loads(body.decode("utf-8"))
    assert data["status"] == "error"

# ---- V1.0 收尾：旧 /api/* fixture 路由已下线（契约） -------------------

LEGACY_FIXTURE_ROUTES = (
    "/api/backtest",
    "/api/paper/snapshot",
    "/api/profile",
    "/api/opportunity/scan",
)


@pytest.mark.parametrize("path", LEGACY_FIXTURE_ROUTES)
def test_legacy_fixture_routes_are_gone(path):
    """These early-skeleton endpoints served tests/fixtures as if real data."""
    code, _headers, body = _call_app({"PATH_INFO": path, "REQUEST_METHOD": "GET"})
    assert code == 404
    assert b"fixture" not in body.lower()


def test_legacy_routes_are_not_listed_anywhere():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    app_source = (root / "backend" / "api" / "app.py").read_text(encoding="utf-8")
    for path in LEGACY_FIXTURE_ROUTES:
        assert path not in app_source
    # 前端也不得引用这些旧接口
    page = (root / "backend" / "dashboard" / "static" / "index.html").read_text(encoding="utf-8")
    for path in LEGACY_FIXTURE_ROUTES:
        assert path not in page


def test_official_v1_router_still_served():
    code, _headers, body = _call_app({"PATH_INFO": "/health", "REQUEST_METHOD": "GET"})
    assert code == 200
    assert json.loads(body.decode("utf-8"))["status"] == "ok"
