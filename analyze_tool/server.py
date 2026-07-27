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
import gzip
import json
import math
import mimetypes
import sys
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from collections.abc import Mapping
from datetime import date, datetime
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
from analyze_tool.plugin_api import PluginRunContext  # noqa: E402
from analyze_tool.plugins import build_default_registry  # noqa: E402
from analyze_tool.plugins.swing_extreme_move import SwingExtremeMovePlugin  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
PLUGIN_REGISTRY = build_default_registry()


def _ensure_swing_extreme_registered() -> None:
    """Register Swing Extreme even when plugins/__init__.py is an older file.

    Several analyze_tool patches historically replaced ``plugins/__init__.py``.
    Registering this plugin here makes the server robust to an old static
    registry and prevents the UI/backend mismatch seen on Windows installs.
    """

    registered = {str(item["id"]) for item in PLUGIN_REGISTRY.list_plugins()}
    if SwingExtremeMovePlugin.plugin_id not in registered:
        PLUGIN_REGISTRY.register(SwingExtremeMovePlugin())


_ensure_swing_extreme_registered()


def registered_plugin_ids() -> list[str]:
    return [str(item["id"]) for item in PLUGIN_REGISTRY.list_plugins()]


def _json_safe(value: Any) -> Any:
    """Recursively convert plugin/data payloads to strict JSON values.

    Python's default ``json.dumps`` emits ``NaN``/``Infinity`` tokens, but those
    are not valid JSON and browser ``JSON.parse`` rejects them.  Trade-bar
    features naturally contain missing warmup values, so sanitize centrally
    instead of requiring every plugin to clean every nested ``fields`` dict.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if type(value).__name__ in {"NAType", "NaTType"}:
        return None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        try:
            converted_list = to_list()
        except (TypeError, ValueError):
            converted_list = value
        if converted_list is not value:
            return _json_safe(converted_list)

    # NumPy/pandas scalar types expose ``item``.  This also converts np.float64
    # NaN/Inf, np.int64 and np.bool_ without importing heavy modules here.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            converted = item()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return _json_safe(converted)

    # pandas.NA/NaT and similar scalar sentinels are not JSON serializable.
    try:
        missing = value != value
    except Exception:
        missing = False
    if isinstance(missing, bool) and missing:
        return None

    return value


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    safe_payload = _json_safe(payload)
    raw = json.dumps(
        safe_payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    accepts_gzip = "gzip" in str(handler.headers.get("Accept-Encoding", "")).lower()
    compressed = accepts_gzip and len(raw) >= 64 * 1024
    body = gzip.compress(raw, compresslevel=3) if compressed else raw
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    if compressed:
        handler.send_header("Content-Encoding", "gzip")
        handler.send_header("Vary", "Accept-Encoding")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


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
    server_version = "CoinBacktestAnalyzeTool/0.1.1"

    def do_GET(self) -> None:  # noqa: N802
        path, params = parse_query(self.path)
        try:
            if path in {"/", "/index.html"}:
                self._serve_static("index.html")
                return
            if path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                return
            if path.startswith("/static/"):
                self._serve_static(path.removeprefix("/static/"))
                return
            if path == "/api/health":
                json_response(self, {"ok": True, "project_root": str(PROJECT_ROOT), "plugins": registered_plugin_ids()})
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
                try:
                    plugin = PLUGIN_REGISTRY.get(plugin_id)
                except KeyError as exc:
                    available = ", ".join(registered_plugin_ids()) or "<none>"
                    raise KeyError(f"unknown plugin: {plugin_id}; registered: {available}") from exc
                context = PluginRunContext(
                    display_df=df,
                    visible_df=df,
                    analysis_frames={},
                    request={
                        "data_type": req.data_type,
                        "timeframe": req.timeframe,
                        "range_pct": req.range_pct,
                        "start": req.start,
                        "end": req.end,
                    },
                    meta=meta,
                )
                contextual_run = getattr(plugin, "run_with_context", None)
                if callable(contextual_run):
                    result = contextual_run(context, plugin_params)
                else:
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
    print(f"[analyze_tool] registered plugins: {', '.join(registered_plugin_ids())}")
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
