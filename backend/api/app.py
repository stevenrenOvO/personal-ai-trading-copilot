"""Lightweight WSGI entry point for V1.0 (standard library only).

It serves the dashboard page, the health check and the V1 API router
(``backend.api.v1``, which holds every official endpoint). The early-skeleton
``/api/*`` fixture routes were removed at V1.0 wrap-up: they served sample data
from ``tests/fixtures`` and had no place in a system that promises real data.
"""

from __future__ import annotations

import json
import re
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, make_server
from wsgiref.util import setup_testing_defaults

from backend.api import v1
from backend.api.runtime import Runtime


# The runtime is initialized lazily and shared by every request.
_RUNTIME = None


def _get_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = Runtime.create()
    return _RUNTIME


def _json_response(status: str, payload: dict):
    return (
        status,
        [("Content-Type", "application/json; charset=utf-8")],
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


def _handle_health(environ):
    return _json_response("200 OK", {"status": "ok"})


def _handle_dashboard(environ):
    """Serve the dashboard static page."""
    try:
        from pathlib import Path
        base = Path(__file__).parent.parent / "dashboard" / "static" / "index.html"
        with open(base, 'rb') as f:
            content = f.read()
        return "200 OK", [("Content-Type", "text/html; charset=utf-8")], content
    except Exception as exc:
        return _json_response("500 Internal Server Error", {"status": "error", "detail": str(exc)})
_ROUTES = [
    (re.compile(r"^/$"), _handle_dashboard),
    (re.compile(r"^/health$"), _handle_health),
    (re.compile(r"^/api/health$"), _handle_health),
]


def application(environ, start_response):
    """WSGI application entry point."""
    setup_testing_defaults(environ)
    path = environ.get("PATH_INFO", "")
    method = environ.get("REQUEST_METHOD", "GET")

    try:
        body, handled = v1.dispatch(environ, start_response, _get_runtime())
        if handled:
            return [body]
    except Exception as exc:
        start_response(
            "500 Internal Server Error",
            [("Content-Type", "application/json; charset=utf-8")],
        )
        return [
            json.dumps(
                {"status": "error", "detail": str(exc)}, ensure_ascii=False
            ).encode("utf-8")
        ]

    for pattern, handler in _ROUTES:
        if pattern.match(path):
            # Only GET for most endpoints, but POST for profile
            if method not in ("GET", "POST"):
                start_response("405 Method Not Allowed", [("Content-Type", "application/json; charset=utf-8")])
                return [json.dumps({"status": "error", "detail": "Method not allowed"}).encode("utf-8")]
            status, headers, body = handler(environ)
            start_response(status, headers)
            return [body]
    start_response("404 Not Found", [("Content-Type", "application/json; charset=utf-8")])
    return [json.dumps({"status": "error", "detail": "Not found"}).encode("utf-8")]


def serve(host="127.0.0.1", port=8000):
    """Run the API server (for development)."""
    class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    def _start_quote_service():
        try:
            from backend.market.quote_service import get_quote_service

            get_quote_service().start()
        except Exception:
            pass

    def _start_auto_ingest():
        """收盘后自动补齐缺失交易日的日线（不依赖是否有人打开页面）。"""
        try:
            from backend.data.auto_ingest import start_background_loop

            start_background_loop(_get_runtime, interval=300.0)
        except Exception:
            pass

    import threading

    threading.Thread(target=_start_quote_service, daemon=True).start()
    threading.Thread(target=_start_auto_ingest, daemon=True).start()
    with make_server(host, port, application, server_class=ThreadingWSGIServer) as httpd:
        print(f"Serving on http://{host}:{port}")
        httpd.serve_forever()


def _main() -> None:
    """CLI entry: ``python -m backend.api.app [--host 0.0.0.0] [--port 8000]``.

    ``--host 0.0.0.0`` is what a phone on the same Wi-Fi needs; the default
    stays loopback-only so nothing is exposed by accident.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Personal AI Trading Copilot API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve(host=args.host, port=args.port)


if __name__ == "__main__":
    _main()
