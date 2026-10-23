"""Lightweight WSGI entry point for V1.1 (standard library only).

It serves the dashboard page, the health check and the V1 API router
(``backend.api.v1``, which holds every official endpoint). The early-skeleton
``/api/*`` fixture routes were removed at V1.0 wrap-up: they served sample data
from ``tests/fixtures`` and had no place in a system that promises real data.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from socketserver import ThreadingMixIn
from wsgiref.simple_server import WSGIServer, make_server
from wsgiref.util import setup_testing_defaults

from backend.api import v1
from backend.api.runtime import Runtime
from backend.api.readiness import version


# The runtime is initialized lazily and shared by every request.
_RUNTIME = None
_RUNTIME_LOCK = threading.Lock()
_RUNTIME_ERROR = False


def _get_runtime():
    global _RUNTIME, _RUNTIME_ERROR
    if _RUNTIME is None:
        with _RUNTIME_LOCK:
            if _RUNTIME is None:
                try:
                    _RUNTIME = Runtime.create()
                    _RUNTIME_ERROR = False
                except Exception:
                    _RUNTIME_ERROR = True
                    logging.getLogger(__name__).exception("Runtime initialization failed")
                    raise
    return _RUNTIME


def _json_response(status: str, payload: dict):
    return (
        status,
        [("Content-Type", "application/json; charset=utf-8")],
        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )


def _handle_health(environ):
    from backend.api.readiness import version
    from backend.data.config import get_settings
    return _json_response("200 OK", {"status": "ok", "version": version(),
        "mode": "demo" if get_settings().demo_mode else "production"})


def _handle_ready(environ):
    from backend.api.readiness import readiness
    result = readiness(_RUNTIME, _RUNTIME_ERROR)
    return _json_response("503 Service Unavailable" if result["status"] in ("starting", "error") else "200 OK", result)


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
    (re.compile(r"^/ready$"), _handle_ready),
]


def application(environ, start_response):
    """WSGI application entry point."""
    setup_testing_defaults(environ)
    path = environ.get("PATH_INFO", "")
    method = environ.get("REQUEST_METHOD", "GET")

    # A static page and liveness probe do not depend on vendors or account state.
    for pattern, handler in _ROUTES:
        if pattern.match(path):
            if method not in ("GET", "HEAD"):
                status, headers, body = _json_response("405 Method Not Allowed", {"status": "error", "detail": "Method not allowed"})
                headers.append(("Allow", "GET, HEAD"))
            else:
                status, headers, body = handler(environ)
            start_response(status, headers)
            return [b"" if method == "HEAD" else body]

    try:
        body, handled = v1.dispatch(environ, start_response, _get_runtime())
        if handled:
            return [body]
    except Exception as exc:
        logging.getLogger(__name__).exception("Request failed: %s", path)
        start_response(
            "500 Internal Server Error",
            [("Content-Type", "application/json; charset=utf-8")],
        )
        return [
            json.dumps(
                {"status": "error", "detail": "服务暂时无法完成请求，请查看运行状态或稍后重试"}, ensure_ascii=False
            ).encode("utf-8")
        ]

    start_response("404 Not Found", [("Content-Type", "application/json; charset=utf-8")])
    return [json.dumps({"status": "error", "detail": "Not found"}).encode("utf-8")]


def serve(host="127.0.0.1", port=8000):
    """Run the API server (for development)."""
    class ThreadingWSGIServer(ThreadingMixIn, WSGIServer):
        daemon_threads = True

    from backend.data.config import get_settings
    demo = get_settings().demo_mode
    if demo:
        from backend.api.demo import configure_demo_services
        configure_demo_services()

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

    with make_server(host, port, application, server_class=ThreadingWSGIServer) as httpd:
        if not demo:
            threading.Thread(target=_start_quote_service, daemon=True).start()
        threading.Thread(target=_start_auto_ingest, daemon=True).start()
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
