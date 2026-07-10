#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the CoinBacktest interactive OHLCV analyzer.

Start from the repository root:
    python analyze_tool/server.py --host 127.0.0.1 --port 8765

Then open:
    http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_tool.data_service import (  # noqa: E402
    config_payload,
    dataframe_to_candles,
    load_dataframe,
    parse_request,
)
from analyze_tool.plugins import build_default_registry  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
PLUGIN_REGISTRY = build_default_registry()


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(raw)


def error_response(handler: BaseHTTPRequestHandler, message: str, status: int = 400, *, debug: str | None = None) -> None:
    payload: dict[str, Any] = {"ok": False, "error": message}
    if debug:
        payload["debug"] = debug
    json_response(handler, payload, status=status)


def parse_query(path: str) -> tuple[str, dict[str, Any]]:
    parsed = urlparse(path)
    params = {k: v[-1] for k, v in parse_qs(parsed.query, keep_blank_values=True).items()}
    return parsed.path, params


class AnalyzeToolHandler(BaseHTTPRequestHandler):
    server_version = "CoinBacktestAnalyzeTool/0.1"

    def do_GET(self) -> None:  # noqa: N802
        path, params = parse_query(self.path)
        try:
            if path in {"/", "/index.html"}:
                self._serve_static("index.html")
                return
            if path.startswith("/static/"):
                self._serve_static(path.removeprefix("/static/"))
                return
            if path == "/api/health":
                json_response(self, {"ok": True, "project_root": str(PROJECT_ROOT)})
                return
            if path == "/api/config":
                json_response(self, {"ok": True, **config_payload()})
                return
            if path == "/api/plugins":
                json_response(self, {"ok": True, "plugins": PLUGIN_REGISTRY.list_plugins()})
                return
            if path == "/api/candles":
                req = parse_request(params)
                df, meta = load_dataframe(req)
                payload = dataframe_to_candles(df, meta)
                payload["ok"] = True
                json_response(self, payload)
                return
            error_response(self, f"not found: {path}", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover - shown to user in browser
            error_response(self, str(exc), status=HTTPStatus.BAD_REQUEST, debug=traceback.format_exc())

    def do_POST(self) -> None:  # noqa: N802
        path, _params = parse_query(self.path)
        try:
            if path == "/api/plugin-markers":
                payload = self._read_json_body()
                req = parse_request(payload.get("data", {}))
                plugin_id = str(payload.get("plugin_id", ""))
                plugin_params = payload.get("params") or {}
                df, meta = load_dataframe(req)
                plugin = PLUGIN_REGISTRY.get(plugin_id)
                result = plugin.run(df, plugin_params)
                out = result.as_dict()
                out["ok"] = True
                out["meta"] = meta
                json_response(self, out)
                return
            error_response(self, f"not found: {path}", status=HTTPStatus.NOT_FOUND)
        except Exception as exc:  # pragma: no cover - shown to user in browser
            error_response(self, str(exc), status=HTTPStatus.BAD_REQUEST, debug=traceback.format_exc())

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep console clean while preserving useful request lines.
        sys.stderr.write("[analyze_tool] " + fmt % args + "\n")

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def _serve_static(self, relative: str) -> None:
        rel = relative.replace("\\", "/").lstrip("/")
        if ".." in Path(rel).parts:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        target = STATIC_DIR / rel
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        raw = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CoinBacktest interactive OHLCV analyzer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), AnalyzeToolHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"[analyze_tool] serving CoinBacktest analyzer: {url}")
    print("[analyze_tool] press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[analyze_tool] stopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
