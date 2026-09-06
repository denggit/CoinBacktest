#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import traceback
import uuid
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
DEFAULT_LIMIT_FEE_RATE = 0.0002
DEFAULT_MARKET_FEE_RATE = 0.0005
DEFAULT_MARKET_SLIPPAGE_RATE = 0.0002
DEFAULT_RISK_PCT = 1.0
DEFAULT_ACCOUNT_SIZE = 10_000.0
EPISODE_ROUTE = re.compile(r"^/api/episodes/([a-f0-9]{12})(?:/(snapshot|snapshots|history|step|rewind|events|delete-annotation|annotation-line|trade|update-order|cancel-order|close|export))?$")


class ReplayApplication:
    def __init__(self, data: ReplayDataService, store: ReplayStore) -> None:
        self.data = data
        self.store = store

    def _ui_event(self, event: dict[str, Any]) -> dict[str, Any]:
        item = dict(event)
        if item.get("event_time"):
            item["event_time_bjt"] = self.data.beijing_display(item["event_time"])
        payload = dict(item.get("payload") or {})
        for key in ("anchor_time", "from_cursor", "to_cursor", "placed_at", "trigger_bar_time", "entry_time", "exit_time", "metric_start_time"):
            if payload.get(key):
                try:
                    payload[f"{key}_bjt"] = self.data.beijing_display(payload[key])
                except Exception:
                    pass
        entry_context = payload.get("entry_context")
        if isinstance(entry_context, dict):
            entry_context = dict(entry_context)
            if entry_context.get("anchor_time"):
                try:
                    entry_context["anchor_time_bjt"] = self.data.beijing_display(entry_context["anchor_time"])
                except Exception:
                    pass
            payload["entry_context"] = entry_context
        item["payload"] = payload
        return item

    def _ui_events(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self._ui_event(event) for event in events]

    def create_episode(self, payload: dict[str, Any]) -> dict[str, Any]:
        symbol = str(payload.get("symbol") or DEFAULT_SYMBOL).strip().upper()
        mode = str(payload.get("mode") or "random")
        is_24x7 = self.data.is_24x7_symbol(symbol)
        previous_episode_id: str | None = None
        if mode == "random":
            cursor = self.data.random_cursor(
                symbol,
                str(payload.get("random_start") or "2026-05-20"),
                str(payload.get("random_end") or "2026-08-15"),
            )
        elif mode == "sequential":
            previous_episode_id = str(payload.get("previous_episode_id") or "").strip() or None
            if previous_episode_id:
                previous = self.store.get_episode(previous_episode_id)
                if previous.status != "closed":
                    raise ValueError("previous sequential Episode must be closed before continuing")
                if previous.symbol != symbol:
                    raise ValueError(f"previous Episode symbol mismatch: {previous.symbol} != {symbol}")
                cursor = self.data.next_sequential_cursor(symbol, previous.cursor_time)
            else:
                raw = payload.get("start_time") if is_24x7 else (payload.get("start_date") or payload.get("start_time"))
                if not raw:
                    raise ValueError(f"start_time is required for {symbol} sequential replay" if is_24x7 else "start_date is required in sequential mode")
                cursor = self.data.sequential_start_cursor(symbol, raw)
        else:
            raw = payload.get("start_time") if is_24x7 else (payload.get("start_date") or payload.get("start_time"))
            if not raw:
                raise ValueError(f"start_time is required for {symbol} 24/7 replay" if is_24x7 else "start_date is required in specific mode")
            cursor = self.data.cursor_for_start(symbol, raw) if is_24x7 else self.data.cursor_for_date(symbol, raw)
        self.data.prepare_episode(symbol, cursor, ["30m", "15m", "2m", "1m"], 700)
        episode = self.store.create_episode(symbol, cursor)
        self.store.add_event(
            episode.id,
            "EPISODE_START",
            episode.cursor_time,
            payload={
                "mode": mode,
                "session_timezone": "24/7" if is_24x7 else "America/New_York",
                "display_timezone": "Asia/Shanghai",
                "start_et": None if is_24x7 else "07:30",
                "episode_weekdays_only": not is_24x7,
                "chart_context": "all_available_okx_bars",
                "symbol": symbol,
                "session_profile": self.data.session_profile(symbol),
                "auto_close_on_bracket_exit": False,
                "continue_after_bracket_exit": True,
                "sequential_mode": mode == "sequential",
                "previous_episode_id": previous_episode_id,
                "sequence_policy": "next_available_1m_after_previous_close" if (mode == "sequential" and is_24x7) else ("next_available_weekday_0730_et" if mode == "sequential" else None),
            },
        )
        created = self.store.get_episode(episode.id)
        return asdict(created)

    def snapshot(self, episode_id: str, timeframe: str, limit: int) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        if episode.status == "active":
            self._sync_trade_lifecycle(episode_id, pd.Timestamp(episode.cursor_time))
        window = self.data.candles(episode.symbol, timeframe, episode.cursor_time, limit)
        return {
            "episode": asdict(episode),
            "clock": self.data.clock_info(episode.cursor_time, episode.symbol),
            "timeframe": timeframe,
            "source": window.source,
            "bars": window.bars,
            "events": self._ui_events(self.store.list_events(episode_id)),
            "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
            "active_trades": self.store.active_trades(episode_id),
            "trade_summary": self.store.trade_summary(episode_id),
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

    def _execution_price_for(self, episode: Any) -> float | None:
        """Return the causal cursor 1m Open exposed to market-order previews."""
        try:
            return float(self.data.execution_open(episode.symbol, episode.cursor_time))
        except (KeyError, ValueError):
            return None

    def snapshots(self, episode_id: str, timeframes: list[str], limit: int) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        if episode.status == "active":
            self._sync_trade_lifecycle(episode_id, pd.Timestamp(episode.cursor_time))
        cleaned = self._clean_timeframes(timeframes)
        self.data.prepare_episode(episode.symbol, episode.cursor_time, cleaned, limit)
        windows = {tf: self.data.candles(episode.symbol, tf, episode.cursor_time, limit) for tf in cleaned}
        return {
            "episode": asdict(episode),
            "clock": self.data.clock_info(episode.cursor_time, episode.symbol),
            "execution_price": self._execution_price_for(episode),
            "charts": {tf: {"timeframe": tf, "source": w.source, "bars": w.bars} for tf, w in windows.items()},
            "events": self._ui_events(self.store.list_events(episode_id)),
            "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
            "active_trades": self.store.active_trades(episode_id),
            "trade_summary": self.store.trade_summary(episode_id),
        }

    def history(self, episode_id: str, timeframe: str, before: str, limit: int) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        if not before:
            raise ValueError("before is required")
        boundary = min(pd.Timestamp(before), pd.Timestamp(episode.cursor_time))
        return self.data.historical_candles(episode.symbol, timeframe, boundary, limit)

    def _next_lifecycle_cursor(
        self,
        episode_id: str,
        cursor: pd.Timestamp,
        target: pd.Timestamp,
    ) -> pd.Timestamp | None:
        """Vectorized look-ahead used only to locate the earliest hidden lifecycle event.

        The simulator is allowed to inspect the cached OHLC chunk internally, but
        the visible cursor is never advanced beyond the earliest fill / SL / TP.
        This preserves exactly the same causal information boundary as the old
        minute-by-minute engine while avoiding dozens of repeated SQLite/state
        reconstructions for +15m/+30m/+60m and fast autoplay.
        """
        if target <= cursor:
            return None
        episode = self.store.get_episode(episode_id)
        active_orders = self.store.active_limit_orders(episode_id)
        active_trades = self.store.active_trades(episode_id)
        # The common chart-only path never descends into the cached 1m frame.
        # One-minute ordering is activated only while an entry/SL/TP level could
        # occur inside the requested higher-timeframe step.
        if not active_orders and not active_trades:
            return None
        bars = self.data.closed_1m_frame(episode.symbol, cursor, target)
        if bars.empty:
            return None

        candidates: list[pd.Timestamp] = []

        # Resting limits: each order is evaluated as a vector mask over the one
        # cached chunk; only the first touched bar is relevant.
        for order in active_orders:
            payload = order.get("payload") or {}
            side = str(payload.get("side") or "").upper()
            if side not in {"LONG", "SHORT"} or order.get("price") is None:
                continue
            start = max(cursor, pd.Timestamp(order["event_time"]))
            rows = bars[bars.index >= start]
            if rows.empty:
                continue
            price = float(order["price"])
            if side == "LONG":
                entry_mask = (rows["open"].astype(float) <= price) | (rows["low"].astype(float) <= price)
            else:
                entry_mask = (rows["open"].astype(float) >= price) | (rows["high"].astype(float) >= price)
            take = self._optional_price(payload, "take_profit")
            if take is None:
                mask = entry_mask
            elif side == "LONG":
                mask = entry_mask | (rows["open"].astype(float) >= take) | (rows["high"].astype(float) >= take)
            else:
                mask = entry_mask | (rows["open"].astype(float) <= take) | (rows["low"].astype(float) <= take)
            hits = rows.index[mask.to_numpy()]
            if len(hits):
                candidates.append(pd.Timestamp(hits[0]) + pd.Timedelta(minutes=1))

        # Open trades: find the first bar touching either current bracket.  The
        # existing close routine still resolves simultaneous SL+TP conservatively.
        for trade in active_trades:
            payload = trade.get("payload") or {}
            side = str(payload.get("side") or "").upper()
            if side not in {"LONG", "SHORT"}:
                continue
            entry_time = pd.Timestamp(payload.get("entry_time") or trade["event_time"])
            start = max(cursor, entry_time)
            rows = bars[bars.index >= start]
            if rows.empty:
                continue
            stop, take = self._current_bracket(episode_id, trade)
            if stop is None and take is None:
                continue
            if side == "LONG":
                stop_mask = False if stop is None else (rows["low"].astype(float) <= float(stop))
                take_mask = False if take is None else (rows["high"].astype(float) >= float(take))
            else:
                stop_mask = False if stop is None else (rows["high"].astype(float) >= float(stop))
                take_mask = False if take is None else (rows["low"].astype(float) <= float(take))
            if isinstance(stop_mask, bool):
                mask = take_mask
            elif isinstance(take_mask, bool):
                mask = stop_mask
            else:
                mask = stop_mask | take_mask
            if isinstance(mask, bool):
                continue
            hits = rows.index[mask.to_numpy()]
            if len(hits):
                candidates.append(pd.Timestamp(hits[0]) + pd.Timedelta(minutes=1))

        return min(candidates) if candidates else None

    def step(self, episode_id: str, minutes: int, timeframes: list[str] | None = None, *, pause_on_event: bool = False) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        if episode.status != "active":
            raise ValueError("episode is not active")
        cleaned = self._clean_timeframes(timeframes or ["30m", "15m", "2m", "1m"])
        minutes = max(1, min(int(minutes), 1440))
        old_cursor = pd.Timestamp(episode.cursor_time)
        target = self.data.fast_forward_target(episode.symbol, old_cursor, minutes)
        cursor = old_cursor
        trade_events: list[dict[str, Any]] = []
        trade_closed = False
        scan_passes = 0

        # +1m stays on the minimal direct path so normal autoplay does not pay
        # for an extra discovery scan. Larger jumps use an event-driven,
        # vectorized scan: inspect one cached OHLC chunk, jump to the earliest
        # lifecycle boundary, persist it, then scan the remainder. Runtime is
        # proportional to fills/exits rather than requested minutes.
        if minutes == 1 and target > cursor:
            scan_passes = 1
            minute_events: list[dict[str, Any]] = []
            minute_events.extend(self._process_limit_fills(episode_id, cursor, target))
            minute_events.extend(self._process_trade_exits(episode_id, cursor, target))
            trade_events.extend(minute_events)
            cursor = target
            trade_closed = any(
                event.get("event_type") == "TRADE_CLOSED"
                and str((event.get("payload") or {}).get("exit_reason") or "")
                in {"TAKE_PROFIT", "STOP_LOSS", "AMBIGUOUS_BOTH_HIT"}
                for event in minute_events
            )
        else:
            while cursor < target:
                scan_passes += 1
                boundary = self._next_lifecycle_cursor(episode_id, cursor, target)
                if boundary is None or boundary > target:
                    cursor = target
                    break

                minute_events = []
                minute_events.extend(self._process_limit_fills(episode_id, cursor, boundary))
                minute_events.extend(self._process_trade_exits(episode_id, cursor, boundary))
                trade_events.extend(minute_events)
                cursor = boundary

                trade_closed = trade_closed or any(
                    event.get("event_type") == "TRADE_CLOSED"
                    and str((event.get("payload") or {}).get("exit_reason") or "")
                    in {"TAKE_PROFIT", "STOP_LOSS", "AMBIGUOUS_BOTH_HIT"}
                    for event in minute_events
                )
                if pause_on_event and minute_events:
                    break
        advanced = max(0, int((cursor - old_cursor) / pd.Timedelta(minutes=1)))
        if advanced:
            episode = self.store.update_cursor(episode_id, cursor)
        updates = (
            self.data.incremental_bars(episode.symbol, cleaned, old_cursor, cursor)
            if advanced
            else {tf: [] for tf in cleaned}
        )
        return {
            "episode": asdict(episode),
            "clock": self.data.clock_info(episode.cursor_time, episode.symbol),
            "execution_price": self._execution_price_for(episode),
            "updates": updates,
            "trade_events": self._ui_events(trade_events),
            "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
            "active_trades": self.store.active_trades(episode_id),
            "trade_summary": self.store.trade_summary(episode_id),
            "advanced_minutes": advanced,
            "at_data_end": cursor >= target and target < old_cursor + pd.Timedelta(minutes=minutes),
            "paused_on_event": pause_on_event and bool(trade_events),
            "auto_finalized": False,
            "trade_closed": trade_closed,
            "episode_continues_after_trade": episode.status == "active",
            "step_engine": "direct_1m" if minutes == 1 else "vectorized_event_driven",
            "requested_bar_minutes": minutes,
            "lifecycle_resolution": "cached_1m_sequence" if scan_passes > 1 or trade_events else "timeframe_jump",
            "lifecycle_scan_passes": scan_passes,
        }

    def rewind(self, episode_id: str, minutes: int, timeframes: list[str] | None = None) -> dict[str, Any]:
        """Move the replay cursor backward and archive the abandoned future branch.

        Rewind is intentionally branch-aware: any labels, orders, fills or notes
        strictly after the new cursor are marked inactive rather than deleted.
        They remain available in the JSON export under ``discarded_events``.
        """
        episode = self.store.get_episode(episode_id)
        if episode.status != "active":
            raise ValueError("episode is not active")
        cleaned = self._clean_timeframes(timeframes or ["30m", "15m", "2m", "1m"])
        minutes = max(1, min(int(minutes), 1440))
        old_cursor = pd.Timestamp(episode.cursor_time)
        start_cursor = pd.Timestamp(episode.start_time)
        target = max(start_cursor, old_cursor - pd.Timedelta(minutes=minutes))
        rewound = int((old_cursor - target) / pd.Timedelta(minutes=1))
        if rewound <= 0:
            return {
                "episode": asdict(episode),
                "clock": self.data.clock_info(episode.cursor_time, episode.symbol),
                "execution_price": self._execution_price_for(episode),
                "events": self._ui_events(self.store.list_events(episode_id)),
                "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
                "active_trades": self.store.active_trades(episode_id),
                "trade_summary": self.store.trade_summary(episode_id),
                "rewound_minutes": 0,
                "discarded_event_count": 0,
                "at_episode_start": True,
            }
        discarded = self.store.deactivate_events_after(episode_id, target)
        episode = self.store.update_cursor(episode_id, target)
        rewind_event = self.store.add_event(
            episode_id,
            "REWIND",
            episode.cursor_time,
            payload={
                "from_cursor": old_cursor.strftime("%Y-%m-%d %H:%M:%S"),
                "to_cursor": target.strftime("%Y-%m-%d %H:%M:%S"),
                "rewound_minutes": rewound,
                "discarded_event_ids": [int(event["id"]) for event in discarded],
                "branch_policy": "archive_future_events",
            },
        )
        self.data.prepare_episode(episode.symbol, target, cleaned, 700)
        windows = {tf: self.data.candles(episode.symbol, tf, target, 700) for tf in cleaned}
        return {
            "episode": asdict(episode),
            "clock": self.data.clock_info(episode.cursor_time, episode.symbol),
            "execution_price": self._execution_price_for(episode),
            "charts": {tf: {"timeframe": tf, "source": w.source, "bars": w.bars} for tf, w in windows.items()},
            "events": self._ui_events(self.store.list_events(episode_id)),
            "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
            "active_trades": self.store.active_trades(episode_id),
            "trade_summary": self.store.trade_summary(episode_id),
            "rewind_event": self._ui_event(rewind_event),
            "rewound_minutes": rewound,
            "discarded_event_count": len(discarded),
            "at_episode_start": target <= start_cursor,
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
        event = self.store.add_event(
            episode_id,
            event_type,
            episode.cursor_time,
            timeframe=str(timeframe) if timeframe else None,
            price=price,
            payload=payload.get("payload") or {},
        )
        return self._ui_event(event)

    def delete_annotation(self, episode_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Archive a mistaken chart annotation and keep a correction audit event."""
        episode = self.store.get_episode(episode_id)
        event_id = int(payload.get("event_id") or 0)
        if event_id <= 0:
            raise ValueError("event_id is required")
        target = self.store.get_event(event_id)
        if target.get("episode_id") != episode_id:
            raise ValueError("annotation does not belong to this episode")
        allowed = {"LIQUIDITY", "TARGET", "MARKER"}
        if target.get("event_type") not in allowed:
            raise ValueError("only Liquidity / Target / shared marker annotations can be deleted here")
        archived = self.store.deactivate_event(episode_id, event_id)
        target_payload = archived.get("payload") or {}
        correction = self.store.add_event(
            episode_id,
            "ANNOTATION_DELETE",
            episode.cursor_time,
            timeframe=archived.get("timeframe"),
            price=None,
            payload={
                "target_event_id": int(event_id),
                "target_event_type": archived.get("event_type"),
                "target_kind": target_payload.get("kind"),
                "target_label": target_payload.get("label"),
                "target_price": archived.get("price"),
                "reason": "manual_correction",
            },
        )
        return {
            "deleted_event": self._ui_event(archived),
            "correction_event": self._ui_event(correction),
            "events": self._ui_events(self.store.list_events(episode_id)),
            "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
        }

    def set_annotation_line_visibility(self, episode_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Hide/show a chart line without deleting the original decision event.

        This is intentionally append-only: the original LIQUIDITY/TARGET/MARKER event
        remains active in the Decision Timeline and training dataset.  A separate
        visibility event only controls chart presentation.
        """
        episode = self.store.get_episode(episode_id)
        if episode.status != "active":
            raise ValueError("annotation line visibility can only be changed for an active Episode")
        event_id = int(payload.get("event_id") or 0)
        if event_id <= 0:
            raise ValueError("event_id is required")
        target = self.store.get_event(event_id)
        if target.get("episode_id") != episode_id:
            raise ValueError("annotation does not belong to this episode")
        allowed = {"LIQUIDITY", "TARGET", "MARKER"}
        if target.get("event_type") not in allowed:
            raise ValueError("only Liquidity / Target / shared marker chart lines can be hidden here")
        visible = bool(payload.get("visible", False))
        target_payload = target.get("payload") or {}
        visibility_event = self.store.add_event(
            episode_id,
            "ANNOTATION_LINE_VISIBILITY",
            episode.cursor_time,
            timeframe=target.get("timeframe"),
            price=None,
            payload={
                "target_event_id": event_id,
                "target_event_type": target.get("event_type"),
                "target_kind": target_payload.get("kind"),
                "target_label": target_payload.get("label"),
                "target_price": target.get("price"),
                "visible": visible,
                "reason": "manual_chart_cleanup",
            },
        )
        return {
            "visibility_event": self._ui_event(visibility_event),
            "events": self._ui_events(self.store.list_events(episode_id)),
            "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
        }

    @staticmethod
    def _optional_price(payload: dict[str, Any], key: str) -> float | None:
        raw = payload.get(key)
        if raw is None or raw == "":
            return None
        value = float(raw)
        if not value > 0:
            raise ValueError(f"{key} must be > 0 when provided")
        return value

    @staticmethod
    def _price_level_changed(previous: float | None, current: float | None) -> bool:
        """Ignore browser float round-trips that do not change an order level."""
        if previous is None or current is None:
            return previous is not current
        tolerance = max(1e-9, abs(float(previous)) * 1e-10, abs(float(current)) * 1e-10)
        return abs(float(previous) - float(current)) > tolerance

    @staticmethod
    def _validate_bracket(side: str, entry_price: float, stop_loss: float | None, take_profit: float | None) -> None:
        if stop_loss is not None:
            if side == "LONG" and stop_loss >= entry_price:
                raise ValueError("LONG stop_loss must be below entry price")
            if side == "SHORT" and stop_loss <= entry_price:
                raise ValueError("SHORT stop_loss must be above entry price")
        if take_profit is not None:
            if side == "LONG" and take_profit <= entry_price:
                raise ValueError("LONG take_profit must be above entry price")
            if side == "SHORT" and take_profit >= entry_price:
                raise ValueError("SHORT take_profit must be below entry price")

    def _bracket_events(
        self,
        episode_id: str,
        *,
        side: str,
        timeframe: str,
        event_time: Any,
        entry_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        order_id: str | None = None,
        trade_id: str | None = None,
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        common = {
            "attached_to_entry": True,
            "side": side,
            "entry_price": float(entry_price),
            "order_id": order_id,
            "trade_id": trade_id,
        }
        if stop_loss is not None:
            events.append(self.store.add_event(
                episode_id, "SL", event_time, timeframe=timeframe, price=stop_loss,
                payload={**common, "role": "attached_stop_loss"},
            ))
        if take_profit is not None:
            events.append(self.store.add_event(
                episode_id, "TP", event_time, timeframe=timeframe, price=take_profit,
                payload={**common, "role": "attached_take_profit"},
            ))
        return events

    @staticmethod
    def _entry_risk(side: str, entry_price: float, initial_stop: float | None) -> float | None:
        if initial_stop is None:
            return None
        risk = (entry_price - initial_stop) if side == "LONG" else (initial_stop - entry_price)
        return float(risk) if risk > 0 else None

    @staticmethod
    def _bounded_rate(payload: dict[str, Any], key: str, default: float) -> float:
        raw = payload.get(key)
        value = default if raw in (None, "") else float(raw)
        if value < 0 or value > 0.05:
            raise ValueError(f"{key} must be between 0 and 0.05")
        return float(value)

    @staticmethod
    def _positive_float(payload: dict[str, Any], key: str, default: float) -> float:
        raw = payload.get(key)
        value = default if raw in (None, "") else float(raw)
        if value <= 0:
            raise ValueError(f"{key} must be > 0")
        return float(value)

    @staticmethod
    def _market_execution_price(side: str, raw_price: float, slippage_rate: float, *, phase: str) -> float:
        """Apply adverse slippage for a market entry or exit."""
        if phase == "entry":
            direction = 1.0 if side == "LONG" else -1.0
        else:
            direction = -1.0 if side == "LONG" else 1.0
        return float(raw_price) * (1.0 + direction * float(slippage_rate))

    def _trade_config(
        self,
        payload: dict[str, Any],
        *,
        side: str,
        entry_price: float,
        stop_loss: float | None,
        order_type: str,
    ) -> dict[str, Any]:
        account_size = self._positive_float(payload, "account_size", DEFAULT_ACCOUNT_SIZE)
        risk_pct = self._positive_float(payload, "risk_pct", DEFAULT_RISK_PCT)
        if risk_pct > 100:
            raise ValueError("risk_pct must be <= 100")
        limit_fee_rate = self._bounded_rate(payload, "limit_fee_rate", DEFAULT_LIMIT_FEE_RATE)
        market_fee_rate = self._bounded_rate(payload, "market_fee_rate", DEFAULT_MARKET_FEE_RATE)
        market_slippage_rate = self._bounded_rate(
            payload, "market_slippage_rate", DEFAULT_MARKET_SLIPPAGE_RATE
        )
        entry_fee_rate = limit_fee_rate if order_type == "limit" else market_fee_rate
        planned_risk_amount = account_size * risk_pct / 100.0
        quantity_raw = payload.get("quantity")
        quantity = None if quantity_raw in (None, "") else float(quantity_raw)
        stop_execution = None
        risk_per_unit = None
        if stop_loss is not None:
            stop_execution = self._market_execution_price(
                side, float(stop_loss), market_slippage_rate, phase="exit"
            )
            # 1R is only the Entry -> raw SL price distance. Fees and adverse
            # stop execution are account costs on top of that risk budget.
            risk_per_unit = abs(float(entry_price) - float(stop_loss))
            if risk_per_unit > 0:
                quantity = planned_risk_amount / risk_per_unit
        if quantity is None:
            quantity = 1.0
        if quantity <= 0:
            raise ValueError("quantity must be > 0")
        entry_fee = float(entry_price) * quantity * entry_fee_rate
        planned_stop_slippage = (
            abs(float(stop_execution) - float(stop_loss)) * quantity
            if stop_execution is not None and stop_loss is not None
            else 0.0
        )
        planned_stop_fee = (
            float(stop_execution) * quantity * market_fee_rate
            if stop_execution is not None
            else 0.0
        )
        planned_stop_net_loss = (
            planned_risk_amount + planned_stop_slippage + entry_fee + planned_stop_fee
            if stop_loss is not None
            else None
        )
        return {
            "account_size": account_size,
            "risk_pct": risk_pct,
            "planned_risk_amount": planned_risk_amount,
            "quantity": quantity,
            "risk_per_unit": risk_per_unit,
            "planned_stop_execution": stop_execution,
            "limit_fee_rate": limit_fee_rate,
            "market_fee_rate": market_fee_rate,
            "market_slippage_rate": market_slippage_rate,
            "entry_fee_rate": entry_fee_rate,
            "entry_fee": entry_fee,
            "planned_stop_slippage": planned_stop_slippage,
            "planned_stop_fee": planned_stop_fee,
            "planned_stop_net_loss": planned_stop_net_loss,
        }

    def _trade_open_events(
        self,
        episode_id: str,
        entry_event: dict[str, Any],
        *,
        trade_id: str,
        side: str,
        timeframe: str,
        entry_price: float,
        order_type: str,
        stop_loss: float | None,
        take_profit: float | None,
        order_id: str | None,
        fill_model: str,
        entry_context: dict[str, Any] | None = None,
        trade_config: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        common = {
            "trade_id": trade_id,
            "side": side,
            "entry_event_id": int(entry_event["id"]),
            "entry_price": float(entry_price),
            "entry_time": entry_event["event_time"],
            "order_type": order_type,
            "order_id": order_id,
            "initial_stop_loss": stop_loss,
            "initial_take_profit": take_profit,
            "entry_context": entry_context or {},
            "fill_model": fill_model,
            "entry_bar_policy": "include_from_open" if order_type == "market" else "exclude_intrabar_limit_fill_bar",
            **(trade_config or {}),
        }
        filled = self.store.add_event(
            episode_id, "ORDER_FILLED", entry_event["event_time"], timeframe=timeframe, price=entry_price,
            payload=common,
        )
        opened = self.store.add_event(
            episode_id, "TRADE_OPEN", entry_event["event_time"], timeframe=timeframe, price=entry_price,
            payload=common,
        )
        return [filled, opened]

    def _current_bracket(self, episode_id: str, trade: dict[str, Any]) -> tuple[float | None, float | None]:
        payload = trade.get("payload") or {}
        trade_id = str(payload.get("trade_id") or trade.get("trade_id") or "")
        order_id = str(payload.get("order_id") or "")
        entry_time = pd.Timestamp(trade["event_time"])
        stop = payload.get("initial_stop_loss")
        take = payload.get("initial_take_profit")
        stop = None if stop in (None, "") else float(stop)
        take = None if take in (None, "") else float(take)
        for event in self.store.list_events(episode_id):
            if pd.Timestamp(event["event_time"]) < entry_time:
                continue
            ep = event.get("payload") or {}
            linked = (
                (trade_id and str(ep.get("trade_id") or "") == trade_id)
                or (order_id and str(ep.get("order_id") or "") == order_id)
                or (not ep.get("trade_id") and not ep.get("order_id"))
            )
            if not linked:
                continue
            if event["event_type"] in {"SL", "MOVE_SL"} and event.get("price") is not None:
                stop = float(event["price"])
            elif event["event_type"] == "TP" and event.get("price") is not None:
                take = float(event["price"])
        return stop, take

    @staticmethod
    def _bracket_exit_price(side: str, kind: str, level: float, bar: dict[str, Any]) -> float:
        open_ = float(bar["open"])
        if side == "LONG" and kind == "SL":
            return open_ if open_ <= level else level
        if side == "LONG" and kind == "TP":
            return open_ if open_ >= level else level
        if side == "SHORT" and kind == "SL":
            return open_ if open_ >= level else level
        if side == "SHORT" and kind == "TP":
            return open_ if open_ <= level else level
        return level

    @staticmethod
    def _trade_returns(side: str, entry: float, exit_price: float) -> float:
        return (exit_price - entry) / entry if side == "LONG" else (entry - exit_price) / entry

    def _path_metrics(
        self,
        side: str,
        entry_price: float,
        prior_bars: list[dict[str, Any]],
        exit_price: float,
    ) -> tuple[float, float]:
        highs = [float(bar["high"]) for bar in prior_bars]
        lows = [float(bar["low"]) for bar in prior_bars]
        highs.append(float(exit_price)); lows.append(float(exit_price))
        if side == "LONG":
            mfe = max(0.0, (max(highs) - entry_price) / entry_price)
            mae = max(0.0, (entry_price - min(lows)) / entry_price)
        else:
            mfe = max(0.0, (entry_price - min(lows)) / entry_price)
            mae = max(0.0, (max(highs) - entry_price) / entry_price)
        return mfe, mae

    def _close_trade(
        self,
        episode_id: str,
        trade: dict[str, Any],
        *,
        exit_time: str,
        exit_price: float,
        exit_reason: str,
        trigger_bar: dict[str, Any] | None,
        prior_bars: list[dict[str, Any]],
        resolution: str | None = None,
    ) -> list[dict[str, Any]]:
        payload = trade.get("payload") or {}
        side = str(payload.get("side") or "").upper()
        trade_id = str(payload.get("trade_id") or trade.get("trade_id") or "")
        entry_price = float(payload.get("entry_price") or trade.get("price"))
        entry_time = str(payload.get("entry_time") or trade.get("event_time"))
        initial_stop_raw = payload.get("initial_stop_loss")
        initial_stop = None if initial_stop_raw in (None, "") else float(initial_stop_raw)
        market_slippage_rate = float(payload.get("market_slippage_rate") or DEFAULT_MARKET_SLIPPAGE_RATE)
        limit_fee_rate = float(payload.get("limit_fee_rate") or DEFAULT_LIMIT_FEE_RATE)
        market_fee_rate = float(payload.get("market_fee_rate") or DEFAULT_MARKET_FEE_RATE)
        exit_order_type = "limit" if exit_reason == "TAKE_PROFIT" else "market"
        raw_exit_price = float(exit_price)
        executed_exit_price = (
            raw_exit_price
            if exit_order_type == "limit"
            else self._market_execution_price(side, raw_exit_price, market_slippage_rate, phase="exit")
        )
        gross = self._trade_returns(side, entry_price, executed_exit_price)
        risk = self._entry_risk(side, entry_price, initial_stop)
        pnl_per_share = (executed_exit_price - entry_price) if side == "LONG" else (entry_price - executed_exit_price)
        quantity = float(payload.get("quantity") or 1.0)
        raw_pnl_per_share = (raw_exit_price - entry_price) if side == "LONG" else (entry_price - raw_exit_price)
        gross_pnl = pnl_per_share * quantity
        raw_gross_pnl = raw_pnl_per_share * quantity
        entry_fee_rate = float(payload.get("entry_fee_rate") or (limit_fee_rate if payload.get("order_type") == "limit" else market_fee_rate))
        entry_fee = float(payload.get("entry_fee") or (entry_price * quantity * entry_fee_rate))
        exit_fee_rate = limit_fee_rate if exit_order_type == "limit" else market_fee_rate
        exit_fee = executed_exit_price * quantity * exit_fee_rate
        net_pnl = gross_pnl - entry_fee - exit_fee
        slippage_cost = max(0.0, raw_gross_pnl - gross_pnl)
        total_costs = entry_fee + exit_fee + slippage_cost
        account_size = float(payload.get("account_size") or DEFAULT_ACCOUNT_SIZE)
        net = net_pnl / account_size if account_size > 0 else 0.0
        planned_risk_amount = payload.get("planned_risk_amount")
        r_multiple = (
            net_pnl / float(planned_risk_amount)
            if planned_risk_amount not in (None, 0, "")
            else ((pnl_per_share / risk) if risk else None)
        )
        risk_overrun_amount = (
            max(0.0, -net_pnl - float(planned_risk_amount))
            if planned_risk_amount not in (None, 0, "") and net_pnl < 0
            else 0.0
        )
        mfe, mae = self._path_metrics(side, entry_price, prior_bars, executed_exit_price)
        hold_minutes = max(0.0, (pd.Timestamp(exit_time) - pd.Timestamp(entry_time)) / pd.Timedelta(minutes=1))
        common = {
            "trade_id": trade_id,
            "side": side,
            "entry_event_id": payload.get("entry_event_id") or trade.get("entry_event_id"),
            "entry_price": entry_price,
            "entry_time": entry_time,
            "exit_price": executed_exit_price,
            "raw_exit_price": raw_exit_price,
            "exit_time": exit_time,
            "exit_reason": exit_reason,
            "resolution": resolution,
            "gross_return_pct": gross * 100.0,
            "net_return_pct": net * 100.0,
            "account_size": account_size,
            "quantity": quantity,
            "planned_risk_amount": planned_risk_amount,
            "risk_per_unit": payload.get("risk_per_unit"),
            "planned_stop_execution": payload.get("planned_stop_execution"),
            "planned_stop_net_loss": payload.get("planned_stop_net_loss"),
            "gross_pnl": gross_pnl,
            "raw_gross_pnl": raw_gross_pnl,
            "net_pnl": net_pnl,
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "total_fees": entry_fee + exit_fee,
            "slippage_cost": slippage_cost,
            "total_costs": total_costs,
            "risk_overrun_amount": risk_overrun_amount,
            "entry_fee_rate": entry_fee_rate,
            "exit_fee_rate": exit_fee_rate,
            "exit_order_type": exit_order_type,
            "market_slippage_rate": market_slippage_rate,
            "r_multiple": r_multiple,
            "mfe_pct": mfe * 100.0,
            "mae_pct": mae * 100.0,
            "holding_minutes": hold_minutes,
            "path_bars_used": len(prior_bars),
            "entry_bar_policy": payload.get("entry_bar_policy") or ("exclude_intrabar_limit_fill_bar" if payload.get("order_type") == "limit" else "include_from_open"),
            "exit_bar_excluded_from_mfe_mae": True,
            "trigger_bar_time": None if trigger_bar is None else trigger_bar.get("time"),
            "trigger_bar": trigger_bar,
        }
        emitted: list[dict[str, Any]] = []
        hit_type = {
            "TAKE_PROFIT": "TAKE_PROFIT_HIT",
            "STOP_LOSS": "STOP_LOSS_HIT",
            "AMBIGUOUS_BOTH_HIT": "TRADE_EXIT_AMBIGUOUS",
        }.get(exit_reason, "TRADE_EXIT")
        emitted.append(self.store.add_event(
            episode_id, hit_type, exit_time, timeframe=trade.get("timeframe") or "1m", price=executed_exit_price, payload=common,
        ))
        emitted.append(self.store.add_event(
            episode_id, "TRADE_CLOSED", exit_time, timeframe=trade.get("timeframe") or "1m", price=executed_exit_price, payload=common,
        ))
        return emitted

    def _process_trade_exits(
        self,
        episode_id: str,
        old_cursor: pd.Timestamp | None,
        new_cursor: pd.Timestamp,
    ) -> list[dict[str, Any]]:
        episode = self.store.get_episode(episode_id)
        emitted: list[dict[str, Any]] = []
        for trade in self.store.active_trades(episode_id):
            payload = trade.get("payload") or {}
            side = str(payload.get("side") or "").upper()
            if side not in {"LONG", "SHORT"}:
                continue
            entry_time = pd.Timestamp(payload.get("entry_time") or trade["event_time"])
            scan_start = entry_time if old_cursor is None else max(pd.Timestamp(old_cursor), entry_time)
            # Limit fills occur intrabar; the fill event is timestamped at the
            # trigger bar close. Starting from entry_time naturally excludes the
            # unknowable pre/post-fill path inside that trigger bar.
            if new_cursor <= scan_start:
                continue
            stop, take = self._current_bracket(episode_id, trade)
            if stop is None and take is None:
                continue
            bars = self.data.closed_1m_bars(episode.symbol, scan_start, new_cursor)
            prior: list[dict[str, Any]] = []
            for bar in bars:
                high = float(bar["high"]); low = float(bar["low"])
                stop_hit = False if stop is None else (low <= stop if side == "LONG" else high >= stop)
                take_hit = False if take is None else (high >= take if side == "LONG" else low <= take)
                if not stop_hit and not take_hit:
                    prior.append(bar)
                    continue
                exit_time = str(bar["available_time"])
                if stop_hit and take_hit:
                    # 1m OHLC cannot reveal which boundary came first. Never
                    # award an optimistic TP; close conservatively at the stop
                    # and keep the ambiguity explicit for later audits.
                    exit_price = self._bracket_exit_price(side, "SL", float(stop), bar)
                    emitted.extend(self._close_trade(
                        episode_id, trade, exit_time=exit_time, exit_price=exit_price,
                        exit_reason="AMBIGUOUS_BOTH_HIT", trigger_bar=bar, prior_bars=prior,
                        resolution="conservative_stop_assumption",
                    ))
                elif stop_hit:
                    exit_price = self._bracket_exit_price(side, "SL", float(stop), bar)
                    emitted.extend(self._close_trade(
                        episode_id, trade, exit_time=exit_time, exit_price=exit_price,
                        exit_reason="STOP_LOSS", trigger_bar=bar, prior_bars=prior,
                    ))
                else:
                    exit_price = self._bracket_exit_price(side, "TP", float(take), bar)
                    emitted.extend(self._close_trade(
                        episode_id, trade, exit_time=exit_time, exit_price=exit_price,
                        exit_reason="TAKE_PROFIT", trigger_bar=bar, prior_bars=prior,
                    ))
                break
        return emitted

    def _sync_trade_lifecycle(self, episode_id: str, cursor: pd.Timestamp) -> list[dict[str, Any]]:
        """Catch up outcomes for legacy/current episodes through ``cursor``."""
        return self._process_trade_exits(episode_id, None, pd.Timestamp(cursor))

    def _limit_fill_event(
        self,
        episode_id: str,
        order_event: dict[str, Any],
        fill: dict[str, Any],
        fill_time: str,
        trade_id: str,
    ) -> dict[str, Any]:
        payload = order_event.get("payload") or {}
        side = str(payload.get("side") or "").upper()
        return self.store.add_event(
            episode_id,
            side,
            fill_time,
            timeframe=order_event.get("timeframe") or "1m",
            price=float(fill["fill_price"]),
            payload={
                "order_type": "limit",
                "trade_id": trade_id,
                "order_id": payload.get("order_id"),
                "limit_price": float(order_event["price"]),
                "stop_loss": payload.get("stop_loss"),
                "take_profit": payload.get("take_profit"),
                "fill_model": "resting_limit_1m_causal_touch",
                "fill_reason": fill.get("fill_reason"),
                "placed_at": order_event.get("event_time"),
                "trigger_bar_time": fill.get("trigger_bar_time"),
                "trigger_bar": fill.get("trigger_bar"),
                "sequence_resolution": fill.get("sequence_resolution"),
                "entry_context": payload.get("entry_context") or {},
                "account_size": payload.get("account_size"),
                "risk_pct": payload.get("risk_pct"),
                "quantity": payload.get("quantity"),
                "limit_fee_rate": payload.get("limit_fee_rate"),
                "market_fee_rate": payload.get("market_fee_rate"),
                "market_slippage_rate": payload.get("market_slippage_rate"),
            },
        )

    def _limit_auto_cancel_event(
        self,
        episode_id: str,
        order_event: dict[str, Any],
        outcome: dict[str, Any],
        cancel_time: str,
    ) -> dict[str, Any]:
        payload = order_event.get("payload") or {}
        return self.store.add_event(
            episode_id,
            "LIMIT_CANCEL",
            cancel_time,
            timeframe=order_event.get("timeframe") or "1m",
            price=float(outcome["trigger_price"]),
            payload={
                "order_id": payload.get("order_id"),
                "side": payload.get("side"),
                "limit_price": float(order_event["price"]),
                "take_profit": payload.get("take_profit"),
                "stop_loss": payload.get("stop_loss"),
                "reason": "take_profit_before_entry",
                "cancel_source": "replay_auto",
                "result": "MISSED_TRADE",
                "placed_at": order_event.get("event_time"),
                "original_order_event_id": order_event.get("id"),
                "trigger_bar_time": outcome.get("trigger_bar_time"),
                "trigger_bar": outcome.get("trigger_bar"),
                "sequence_resolution": outcome.get("sequence_resolution"),
            },
        )

    def _process_limit_fills(
        self, episode_id: str, old_cursor: pd.Timestamp, new_cursor: pd.Timestamp
    ) -> list[dict[str, Any]]:
        episode = self.store.get_episode(episode_id)
        emitted: list[dict[str, Any]] = []
        for order in self.store.active_limit_orders(episode_id):
            payload = order.get("payload") or {}
            placed_at = pd.Timestamp(order["event_time"])
            start = max(old_cursor, placed_at)
            result = self.data.limit_order_lifecycle(
                episode.symbol,
                str(payload.get("side") or ""),
                float(order["price"]),
                self._optional_price(payload, "take_profit"),
                start,
                new_cursor,
            )
            if result is None:
                continue
            trigger_start = pd.Timestamp(result["trigger_bar_time"])
            fill_time = (trigger_start + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
            if result.get("outcome") == "cancel":
                emitted.append(self._limit_auto_cancel_event(episode_id, order, result, fill_time))
                continue
            trade_id = uuid.uuid4().hex[:12]
            fill_event = self._limit_fill_event(episode_id, order, result, fill_time, trade_id)
            emitted.append(fill_event)
            emitted.extend(self._trade_open_events(
                episode_id, fill_event, trade_id=trade_id,
                side=str(payload.get("side") or "").upper(),
                timeframe=order.get("timeframe") or "1m",
                entry_price=float(fill_event["price"]), order_type="limit",
                stop_loss=self._optional_price(payload, "stop_loss"),
                take_profit=self._optional_price(payload, "take_profit"),
                order_id=str(payload.get("order_id") or "") or None,
                fill_model=str((fill_event.get("payload") or {}).get("fill_model") or "resting_limit_1m_causal_touch"),
                entry_context=payload.get("entry_context") or {},
                trade_config=self._trade_config(
                    payload,
                    side=str(payload.get("side") or "").upper(),
                    entry_price=float(fill_event["price"]),
                    stop_loss=self._optional_price(payload, "stop_loss"),
                    order_type="limit",
                ),
            ))
            emitted.extend(self._bracket_events(
                episode_id,
                side=str(payload.get("side") or "").upper(),
                timeframe=order.get("timeframe") or "1m",
                event_time=fill_time,
                entry_price=float(fill_event["price"]),
                stop_loss=self._optional_price(payload, "stop_loss"),
                take_profit=self._optional_price(payload, "take_profit"),
                order_id=str(payload.get("order_id") or "") or None,
                trade_id=trade_id,
            ))
        return emitted

    def trade(self, episode_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        side = str(payload.get("side") or "").upper()
        if side == "CLOSE":
            raise ValueError("manual close is disabled; Replay exits only at TP or SL")
        if side not in {"LONG", "SHORT"}:
            raise ValueError("side must be LONG or SHORT")
        if episode.status != "active":
            raise ValueError("episode is not active")
        order_type = str(payload.get("order_type") or "market").lower().strip()
        timeframe = str(payload.get("timeframe") or "1m")
        stop_loss = self._optional_price(payload, "stop_loss")
        take_profit = self._optional_price(payload, "take_profit")

        if order_type == "market":
            raw_fill = self.data.execution_open(episode.symbol, episode.cursor_time)
            market_slippage_rate = self._bounded_rate(
                payload, "market_slippage_rate", DEFAULT_MARKET_SLIPPAGE_RATE
            )
            fill = self._market_execution_price(side, raw_fill, market_slippage_rate, phase="entry")
            self._validate_bracket(side, fill, stop_loss, take_profit)
            trade_config = self._trade_config(
                payload, side=side, entry_price=fill, stop_loss=stop_loss, order_type="market"
            )
            trade_id = uuid.uuid4().hex[:12]
            event = self.store.add_event(
                episode_id, side, episode.cursor_time, timeframe=timeframe, price=fill,
                payload={
                    "fill_model": "cursor_1m_open", "order_type": "market", "trade_id": trade_id,
                    "stop_loss": stop_loss, "take_profit": take_profit, "raw_fill_price": raw_fill,
                    **trade_config,
                },
            )
            events = [event, *self._trade_open_events(
                episode_id, event, trade_id=trade_id, side=side, timeframe=timeframe,
                entry_price=fill, order_type="market", stop_loss=stop_loss, take_profit=take_profit,
                order_id=None, fill_model="cursor_1m_open_with_adverse_slippage", entry_context=payload.get("entry_context") or {},
                trade_config=trade_config,
            ), *self._bracket_events(
                episode_id, side=side, timeframe=timeframe, event_time=episode.cursor_time,
                entry_price=fill, stop_loss=stop_loss, take_profit=take_profit, trade_id=trade_id,
            )]
            return {
                "event": self._ui_event(event), "events": self._ui_events(events), "fill_price": fill, "status": "filled",
                "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
                "active_trades": self.store.active_trades(episode_id),
                "trade_summary": self.store.trade_summary(episode_id),
            }

        if order_type != "limit":
            raise ValueError("order_type must be market or limit")
        limit_price = float(payload.get("limit_price"))
        if not limit_price > 0:
            raise ValueError("limit_price must be > 0")
        self._validate_bracket(side, limit_price, stop_loss, take_profit)
        order_id = uuid.uuid4().hex[:12]
        entry_context = payload.get("entry_context") or {}
        order_config = self._trade_config(
            payload, side=side, entry_price=limit_price, stop_loss=stop_loss, order_type="limit"
        )
        order_event = self.store.add_event(
            episode_id, "LIMIT_ORDER", episode.cursor_time, timeframe=timeframe, price=limit_price,
            payload={
                "order_id": order_id, "side": side, "order_type": "limit", "status": "pending",
                "stop_loss": stop_loss, "take_profit": take_profit, "entry_context": entry_context,
                **order_config,
            },
        )

        # A crossing/marketable limit is executable immediately at the known
        # cursor 1m open. Normal FVG retracement orders remain pending.
        current_open = self.data.execution_open(episode.symbol, episode.cursor_time)
        marketable = (side == "LONG" and limit_price >= current_open) or (side == "SHORT" and limit_price <= current_open)
        if marketable:
            immediate = {
                "fill_price": current_open, "fill_reason": "marketable_limit_at_cursor_open",
                "trigger_bar_time": episode.cursor_time, "trigger_bar": None,
            }
            trade_id = uuid.uuid4().hex[:12]
            fill_event = self._limit_fill_event(episode_id, order_event, immediate, episode.cursor_time, trade_id)
            trade_config = self._trade_config(
                payload, side=side, entry_price=current_open, stop_loss=stop_loss, order_type="limit"
            )
            events = [order_event, fill_event, *self._trade_open_events(
                episode_id, fill_event, trade_id=trade_id, side=side, timeframe=timeframe,
                entry_price=current_open, order_type="limit", stop_loss=stop_loss, take_profit=take_profit,
                order_id=order_id, fill_model="marketable_limit_at_cursor_open", entry_context=entry_context,
                trade_config=trade_config,
            ), *self._bracket_events(
                episode_id, side=side, timeframe=timeframe, event_time=episode.cursor_time,
                entry_price=current_open, stop_loss=stop_loss, take_profit=take_profit, order_id=order_id, trade_id=trade_id,
            )]
            return {
                "event": self._ui_event(order_event), "events": self._ui_events(events), "fill_price": current_open, "status": "filled",
                "order_id": order_id, "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
                "active_trades": self.store.active_trades(episode_id),
                "trade_summary": self.store.trade_summary(episode_id),
            }
        return {
            "event": self._ui_event(order_event), "events": self._ui_events([order_event]), "fill_price": None, "status": "pending",
            "order_id": order_id, "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
            "active_trades": self.store.active_trades(episode_id),
            "trade_summary": self.store.trade_summary(episode_id),
        }


    def cancel_limit_order(self, episode_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        active = self.store.active_limit_orders(episode_id)
        requested = str(payload.get("order_id") or "").strip()
        if requested:
            matches = [event for event in active if str((event.get("payload") or {}).get("order_id")) == requested]
        else:
            matches = active[-1:]
        if not matches:
            raise ValueError("no active limit order to cancel")
        order = matches[-1]
        order_payload = order.get("payload") or {}
        event = self.store.add_event(
            episode_id,
            "LIMIT_CANCEL",
            episode.cursor_time,
            timeframe=order.get("timeframe"),
            price=order.get("price"),
            payload={
                "order_id": order_payload.get("order_id"),
                "side": order_payload.get("side"),
                "limit_price": order.get("price"),
            },
        )
        return {
            "event": self._ui_event(event),
            "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
            "active_trades": self.store.active_trades(episode_id),
            "trade_summary": self.store.trade_summary(episode_id),
        }

    def update_order(self, episode_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Append an edit to a resting order or an open trade bracket."""
        episode = self.store.get_episode(episode_id)
        if episode.status != "active":
            raise ValueError("episode is not active")
        order_id = str(payload.get("order_id") or "").strip()
        trade_id = str(payload.get("trade_id") or "").strip()
        stop_loss = self._optional_price(payload, "stop_loss")
        take_profit = self._optional_price(payload, "take_profit")

        if order_id:
            active_orders = [
                item for item in self.store.active_limit_orders(episode_id)
                if str((item.get("payload") or {}).get("order_id") or "") == order_id
            ]
            if active_orders:
                current = active_orders[-1]
                current_payload = dict(current.get("payload") or {})
                side = str(current_payload.get("side") or "").upper()
                limit_price = float(payload.get("limit_price", current.get("price")))
                self._validate_bracket(side, limit_price, stop_loss, take_profit)
                merged = {
                    **current_payload,
                    "order_id": order_id,
                    "side": side,
                    "order_type": "limit",
                    "status": "pending",
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "modified_from_event_id": current.get("id"),
                }
                for key in (
                    "account_size", "risk_pct", "limit_fee_rate", "market_fee_rate",
                    "market_slippage_rate", "entry_context",
                ):
                    if key in payload:
                        merged[key] = payload[key]
                merged.update(self._trade_config(
                    merged,
                    side=side,
                    entry_price=limit_price,
                    stop_loss=stop_loss,
                    order_type="limit",
                ))
                event = self.store.add_event(
                    episode_id,
                    "LIMIT_MODIFY",
                    episode.cursor_time,
                    timeframe=current.get("timeframe") or payload.get("timeframe") or "1m",
                    price=limit_price,
                    payload=merged,
                )
                return {
                    "events": self._ui_events([event]),
                    "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
                    "active_trades": self.store.active_trades(episode_id),
                    "trade_summary": self.store.trade_summary(episode_id),
                }

        active_trades = self.store.active_trades(episode_id)
        matches = [
            item for item in active_trades
            if (trade_id and str(item.get("trade_id") or "") == trade_id)
            or (order_id and str((item.get("payload") or {}).get("order_id") or "") == order_id)
        ]
        if not matches:
            raise ValueError("active order or trade not found")
        trade = matches[-1]
        trade_payload = trade.get("payload") or {}
        side = str(trade_payload.get("side") or "").upper()
        entry_price = float(trade_payload.get("entry_price") or trade.get("price"))
        self._validate_bracket(side, entry_price, stop_loss, take_profit)
        linked_trade_id = str(trade.get("trade_id") or trade_id)
        common = {
            "trade_id": linked_trade_id,
            "order_id": trade_payload.get("order_id"),
            "side": side,
            "source": "position_drag",
        }
        current_stop, current_take = self._current_bracket(episode_id, trade)
        events: list[dict[str, Any]] = []
        if stop_loss is not None and self._price_level_changed(current_stop, stop_loss):
            events.append(self.store.add_event(
                episode_id, "SL", episode.cursor_time,
                timeframe=trade.get("timeframe") or payload.get("timeframe") or "1m",
                price=stop_loss,
                payload={**common, "kind": "stop_loss", "previous_price": current_stop},
            ))
        if take_profit is not None and self._price_level_changed(current_take, take_profit):
            events.append(self.store.add_event(
                episode_id, "TP", episode.cursor_time,
                timeframe=trade.get("timeframe") or payload.get("timeframe") or "1m",
                price=take_profit,
                payload={**common, "kind": "take_profit", "previous_price": current_take},
            ))
        return {
            "events": self._ui_events(events),
            "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
            "active_trades": self.store.active_trades(episode_id),
            "trade_summary": self.store.trade_summary(episode_id),
        }

    def _finalize_episode(self, episode_id: str, *, reason: str, finalized_by: str) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        if episode.status != "active":
            return {"episode": episode, "summary": self.store.trade_summary(episode_id), "finalization_events": []}
        finalization_events: list[dict[str, Any]] = []
        for order in self.store.unresolved_limit_orders(episode_id):
            payload = order.get("payload") or {}
            finalization_events.append(self.store.add_event(
                episode_id, "LIMIT_EXPIRED", episode.cursor_time,
                timeframe=order.get("timeframe"), price=order.get("price"),
                payload={
                    "order_id": payload.get("order_id"), "side": payload.get("side"),
                    "limit_price": order.get("price"), "reason": reason,
                    "original_order_event_id": order.get("id"),
                },
            ))
        summary = self.store.trade_summary(episode_id)
        finalization_events.append(self.store.add_event(
            episode_id, "EPISODE_SUMMARY", episode.cursor_time,
            payload={**summary, "autosave": "every_event_immediate", "finalized_by": finalized_by, "finalize_reason": reason},
        ))
        finalization_events.append(self.store.add_event(episode_id, "CLOSE", episode.cursor_time, payload={"reason": reason}))
        closed = self.store.close_episode(episode_id)
        return {"episode": closed, "summary": self.store.trade_summary(episode_id), "finalization_events": finalization_events}

    def close_episode(self, episode_id: str) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        if episode.status != "active":
            raise ValueError("episode is not active")
        if self.store.active_trades(episode_id):
            raise ValueError("当前持仓必须由 Replay 触发 TP 或 SL 后才能结束训练")
        self._sync_trade_lifecycle(episode_id, pd.Timestamp(episode.cursor_time))
        finalized = self._finalize_episode(episode_id, reason="episode_end", finalized_by="end_episode")
        return {
            "episode": asdict(finalized["episode"]),
            "summary": finalized["summary"],
            "events": self._ui_events(self.store.list_events(episode_id)),
        }


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
    server_version = "CoinBacktestHumanReplayLab/1.9"

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
                symbols = APP.data.available_symbols()
                requested = str(params.get("symbol") or DEFAULT_SYMBOL).upper().strip()
                selected = requested if requested in symbols else (symbols[0] if symbols else requested)
                coverage = APP.data.coverage(selected) if symbols else None
                _json_response(self, {
                    "ok": True,
                    "db_path": str(APP.data.db_path),
                    "symbol": selected,
                    "symbols": symbols,
                    "timeframes": list(SUPPORTED_TIMEFRAMES),
                    "coverage": coverage,
                })
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
                if action == "history":
                    _json_response(self, {
                        "ok": True,
                        **APP.history(
                            episode_id,
                            params.get("timeframe", "30m"),
                            params.get("before", ""),
                            int(params.get("limit", "900")),
                        ),
                    })
                    return
                if action == "events":
                    _json_response(self, {"ok": True, "events": APP._ui_events(APP.store.list_events(episode_id))}); return
                if action == "export":
                    _json_response(self, {"ok": True, **APP.store.export_episode(episode_id)}); return
                if action is None:
                    episode = APP.store.get_episode(episode_id)
                    _json_response(self, {"ok": True, "episode": asdict(episode), "clock": APP.data.clock_info(episode.cursor_time, episode.symbol)}); return
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
                _json_response(self, {"ok": True, "episode": episode, "clock": APP.data.clock_info(episode["cursor_time"], episode["symbol"])})
                return
            match = EPISODE_ROUTE.match(path)
            if not match:
                self.send_error(HTTPStatus.NOT_FOUND); return
            episode_id, action = match.groups()
            if action == "step":
                tfs = payload.get("timeframes") or ["30m", "15m", "2m", "1m"]
                _json_response(self, {"ok": True, **APP.step(episode_id, int(payload.get("minutes", 1)), list(tfs), pause_on_event=payload.get("pause_on_event") is True)}); return
            if action == "rewind":
                tfs = payload.get("timeframes") or ["30m", "15m", "2m", "1m"]
                _json_response(self, {"ok": True, **APP.rewind(episode_id, int(payload.get("minutes", 1)), list(tfs))}); return
            if action == "events":
                _json_response(self, {"ok": True, "event": APP.add_event(episode_id, payload)}); return
            if action == "delete-annotation":
                _json_response(self, {"ok": True, **APP.delete_annotation(episode_id, payload)}); return
            if action == "annotation-line":
                _json_response(self, {"ok": True, **APP.set_annotation_line_visibility(episode_id, payload)}); return
            if action == "trade":
                _json_response(self, {"ok": True, **APP.trade(episode_id, payload)}); return
            if action == "update-order":
                _json_response(self, {"ok": True, **APP.update_order(episode_id, payload)}); return
            if action == "cancel-order":
                _json_response(self, {"ok": True, **APP.cancel_limit_order(episode_id, payload)}); return
            if action == "close":
                _json_response(self, {"ok": True, **APP.close_episode(episode_id)}); return
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
    parser = argparse.ArgumentParser(description="CoinBacktest Human Trader Replay Lab V1.11 - sequential replay + SOXL session + ETH/XAU 24/7")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8775)
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"), help="CoinBacktest data directory containing crypto_history.db")
    return parser


def main() -> int:
    global APP
    args = build_parser().parse_args()
    APP = ReplayApplication(ReplayDataService(args.data_dir), ReplayStore(args.store))
    httpd = ThreadingHTTPServer((args.host, args.port), ReplayHandler)
    print(f"[human_replay_lab] http://{args.host}:{args.port}")
    symbols = APP.data.available_symbols()
    if not symbols:
        raise RuntimeError(f"No local OKX *_1m tables found in {APP.data.db_path}")
    print(f"[human_replay_lab] OKX local 1m @ {APP.data.db_path}")
    print(f"[human_replay_lab] symbols: {', '.join(symbols)}")
    for symbol in symbols[:6]:
        coverage = APP.data.coverage(symbol)
        print(
            f"[human_replay_lab] {symbol}: {coverage['available_start_et']} -> "
            f"{coverage['available_end_et']} | rows={coverage['rows_1m']:,}"
        )
    print("[human_replay_lab] profiles: SOXL=weekday 07:30-16:00 ET; ETH/XAU=24/7 continuous Replay; each trade exits at TP/SL")
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
