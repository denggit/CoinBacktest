#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import traceback
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from human_replay_lab.data_service import DEFAULT_SYMBOL, ReplayDataService, SUPPORTED_TIMEFRAMES  # noqa: E402
from human_replay_lab.store import ReplayStore  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_STORE = PROJECT_ROOT / "data" / "human_replay_lab" / "replay.sqlite3"
EPISODE_ROUTE = re.compile(r"^/api/episodes/([a-f0-9]{12})(?:/(snapshot|snapshots|step|events|trade|close|export))?$")


class ReplayApplication:
    def __init__(self, data: ReplayDataService, store: ReplayStore) -> None:
        self.data = data
        self.store = store

    def create_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        symbol = str(payload.get("symbol") or DEFAULT_SYMBOL).strip().upper()
        mode = str(payload.get("mode") or "random")
        if mode == "random":
            cursor = self.data.random_cursor(
                symbol,
                str(payload.get("random_start") or "2023-01-01"),
                str(payload.get("random_end") or "2026-06-30"),
            )
        else:
            raw = payload.get("start_date") or payload.get("start_time")
            if not raw:
                raise ValueError("start_date is required in specific mode")
            # Specific mode also starts at 07:30 ET. Users choose a day, not a
            # hindsight-friendly intraday cursor.
            cursor = self.data.cursor_for_date(symbol, raw)
        self.data.prepare_episode(symbol, cursor, ["30m", "15m", "2m", "1m"], 700)
        episode = self.store.create_episode(symbol, cursor)
        self.store.add_event(
            episode.id,
            "EPISODE_START",
            episode.cursor_time,
            payload={"mode": mode, "timezone": "America/New_York", "start_et": "07:30", "weekdays_only": True},
        )
        return asdict(self.store.get_episode(episode.id))

    def snapshot(self, episode_id: str, timeframe: str, limit: int) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        window = self.data.candles(episode.symbol, timeframe, episode.cursor_time, limit)
        return {
            "episode": asdict(episode),
            "clock": self.data.clock_info(episode.cursor_time),
            "timeframe": timeframe,
            "source": window.source,
            "bars": window.bars,
            "events": self.store.list_events(episode_id),
        }

    @staticmethod
    def _clean_timeframes(timeframes: list[str]) -> list[str]:
        cleaned: list[str] = []
        for timeframe in timeframes:
            tf = str(timeframe).strip()
            if tf and tf not in cleaned:
                if tf not in SUPPORTED_TIMEFRAMES:
                    raise ValueError(f"unsupported timeframe: {tf}")
                cleaned.append(tf)
        if not cleaned:
            cleaned = ["30m", "15m", "2m", "1m"]
        if len(cleaned) > 8:
            raise ValueError("at most 8 chart timeframes are allowed")
        return cleaned

    def snapshots(self, episode_id: str, timeframes: list[str], limit: int) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        cleaned = self._clean_timeframes(timeframes)
        self.data.prepare_episode(episode.symbol, episode.cursor_time, cleaned, limit)
        windows = {tf: self.data.candles(episode.symbol, tf, episode.cursor_time, limit) for tf in cleaned}
        return {
            "episode": asdict(episode),
            "clock": self.data.clock_info(episode.cursor_time),
            "charts": {tf: {"timeframe": tf, "source": w.source, "bars": w.bars} for tf, w in windows.items()},
            "events": self.store.list_events(episode_id),
        }

    def step(self, episode_id: str, minutes: int, timeframes: list[str] | None = None) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        cleaned = self._clean_timeframes(timeframes or ["30m", "15m", "2m", "1m"])
        minutes = max(1, min(int(minutes), 240))
        old_cursor = pd.Timestamp(episode.cursor_time)
        cursor = old_cursor
        advanced = 0
        for _ in range(minutes):
            next_cursor = cursor + pd.Timedelta(minutes=1)
            if not self.data.can_step_to(episode.symbol, next_cursor):
                break
            cursor = next_cursor
            advanced += 1
        if advanced:
            episode = self.store.update_cursor(episode_id, cursor)
        updates = self.data.incremental_bars(episode.symbol, cleaned, old_cursor, cursor) if advanced else {tf: [] for tf in cleaned}
        return {
            "episode": asdict(episode),
            "clock": self.data.clock_info(episode.cursor_time),
            "updates": updates,
            "advanced_minutes": advanced,
            "at_data_end": advanced < minutes,
        }

    def add_event(self, episode_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        event_type = str(payload.get("event_type") or "").upper()
        allowed = {
            "LIQUIDITY", "BIAS", "TARGET", "WATCH", "WAIT", "SKIP", "INVALIDATE",
            "NOTE", "SL", "TP", "MOVE_SL", "PARTIAL", "CLOSE", "MARKER",
        }
        if event_type not in allowed:
            raise ValueError(f"unsupported event_type: {event_type}")
        price = payload.get("price")
        price = None if price is None or price == "" else float(price)
        timeframe = payload.get("timeframe")
        return self.store.add_event(
            episode_id,
            event_type,
            episode.cursor_time,
            timeframe=str(timeframe) if timeframe else None,
            price=price,
            payload=payload.get("payload") or {},
        )

    def trade(self, episode_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        side = str(payload.get("side") or "").upper()
        if side not in {"LONG", "SHORT", "CLOSE"}:
            raise ValueError("side must be LONG, SHORT or CLOSE")
        fill = self.data.execution_open(episode.symbol, episode.cursor_time)
        event = self.store.add_event(
            episode_id,
            side,
            episode.cursor_time,
            timeframe=str(payload.get("timeframe") or "1m"),
            price=fill,
            payload={"fill_model": "cursor_1m_open", "order_type": "market"},
        )
        return {"event": event, "fill_price": fill}


APP: ReplayApplication | None = None


def _json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(raw)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(raw)


def _error(handler: BaseHTTPRequestHandler, exc: Exception, status: int = 400) -> None:
    _json_response(handler, {"ok": False, "error": str(exc)}, status)


def _query(path: str) -> tuple[str, dict[str, str]]:
    parsed = urlparse(path)
    return parsed.path, {k: v[-1] for k, v in parse_qs(parsed.query).items()}


class ReplayHandler(BaseHTTPRequestHandler):
    server_version = "CoinBacktestHumanReplayLab/1.2"

    def do_GET(self) -> None:  # noqa: N802
        assert APP is not None
        path, params = _query(self.path)
        try:
            if path in {"/", "/index.html"}:
                self._static("index.html")
                return
            if path.startswith("/static/"):
                self._static(path.removeprefix("/static/"))
                return
            if path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT); self.end_headers(); return
            if path == "/api/health":
                _json_response(self, {"ok": True, "db_path": str(APP.data.db_path), "symbol": DEFAULT_SYMBOL, "timeframes": list(SUPPORTED_TIMEFRAMES)})
                return
            match = EPISODE_ROUTE.match(path)
            if match:
                episode_id, action = match.groups()
                if action == "snapshot":
                    _json_response(self, {"ok": True, **APP.snapshot(episode_id, params.get("timeframe", "15m"), int(params.get("limit", "320")))})
                    return
                if action == "snapshots":
                    tfs = [x.strip() for x in params.get("timeframes", "30m,15m,2m,1m").split(",") if x.strip()]
                    _json_response(self, {"ok": True, **APP.snapshots(episode_id, tfs, int(params.get("limit", "700")))})
                    return
                if action == "events":
                    _json_response(self, {"ok": True, "events": APP.store.list_events(episode_id)}); return
                if action == "export":
                    _json_response(self, {"ok": True, **APP.store.export_episode(episode_id)}); return
                if action is None:
                    episode = APP.store.get_episode(episode_id)
                    _json_response(self, {"ok": True, "episode": asdict(episode), "clock": APP.data.clock_info(episode.cursor_time)}); return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            sys.stderr.write(traceback.format_exc()); _error(self, exc)

    def do_POST(self) -> None:  # noqa: N802
        assert APP is not None
        path, _params = _query(self.path)
        try:
            payload = self._read_json()
            if path == "/api/episodes":
                episode = APP.create_episode(payload)
                _json_response(self, {"ok": True, "episode": episode, "clock": APP.data.clock_info(episode["cursor_time"])})
                return
            match = EPISODE_ROUTE.match(path)
            if not match:
                self.send_error(HTTPStatus.NOT_FOUND); return
            episode_id, action = match.groups()
            if action == "step":
                tfs = payload.get("timeframes") or ["30m", "15m", "2m", "1m"]
                _json_response(self, {"ok": True, **APP.step(episode_id, int(payload.get("minutes", 1)), list(tfs))}); return
            if action == "events":
                _json_response(self, {"ok": True, "event": APP.add_event(episode_id, payload)}); return
            if action == "trade":
                _json_response(self, {"ok": True, **APP.trade(episode_id, payload)}); return
            if action == "close":
                episode = APP.store.get_episode(episode_id)
                APP.store.add_event(episode_id, "CLOSE", episode.cursor_time, payload={"reason": "episode_end"})
                _json_response(self, {"ok": True, "episode": asdict(APP.store.close_episode(episode_id))}); return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            sys.stderr.write(traceback.format_exc()); _error(self, exc)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        return {} if length <= 0 else json.loads(self.rfile.read(length).decode("utf-8"))

    def _static(self, rel: str) -> None:
        rel = rel.replace("\\", "/")
        if ".." in Path(rel).parts:
            self.send_error(HTTPStatus.BAD_REQUEST); return
        target = STATIC_DIR / rel
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND); return
        raw = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CoinBacktest Human Trader Replay Lab - SOXL")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"), help="CoinBacktest data directory containing alpaca_stock_history.db")
    return parser


def main() -> int:
    global APP
    args = build_parser().parse_args()
    APP = ReplayApplication(ReplayDataService(args.data_dir), ReplayStore(args.store))
    httpd = ThreadingHTTPServer((args.host, args.port), ReplayHandler)
    print(f"[human_replay_lab] http://{args.host}:{args.port}")
    print(f"[human_replay_lab] SOXL source: {APP.data.db_path}")
    print("[human_replay_lab] replay: weekdays/trading-days only, 07:30 ET -> 16:00 ET")
    print(f"[human_replay_lab] labels: {args.store}")
    print("[human_replay_lab] Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[human_replay_lab] stopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
