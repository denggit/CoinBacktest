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
DEFAULT_ROUND_TRIP_FEE_RATE = 0.0011
EPISODE_ROUTE = re.compile(r"^/api/episodes/([a-f0-9]{12})(?:/(snapshot|snapshots|step|rewind|events|delete-annotation|annotation-line|trade|cancel-order|close|export))?$")


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
                "auto_close_on_bracket_exit": is_24x7,
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
            "charts": {tf: {"timeframe": tf, "source": w.source, "bars": w.bars} for tf, w in windows.items()},
            "events": self._ui_events(self.store.list_events(episode_id)),
            "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
            "active_trades": self.store.active_trades(episode_id),
            "trade_summary": self.store.trade_summary(episode_id),
        }

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
        bars = self.data.closed_1m_frame(episode.symbol, cursor, target)
        if bars.empty:
            return None

        candidates: list[pd.Timestamp] = []

        # Resting limits: each order is evaluated as a vector mask over the one
        # cached chunk; only the first touched bar is relevant.
        for order in self.store.active_limit_orders(episode_id):
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
                mask = (rows["open"].astype(float) <= price) | (rows["low"].astype(float) <= price)
            else:
                mask = (rows["open"].astype(float) >= price) | (rows["high"].astype(float) >= price)
            hits = rows.index[mask.to_numpy()]
            if len(hits):
                candidates.append(pd.Timestamp(hits[0]) + pd.Timedelta(minutes=1))

        # Open trades: find the first bar touching either current bracket.  The
        # existing close routine still resolves simultaneous SL+TP conservatively.
        for trade in self.store.active_trades(episode_id):
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

    def step(self, episode_id: str, minutes: int, timeframes: list[str] | None = None) -> dict[str, Any]:
        episode = self.store.get_episode(episode_id)
        if episode.status != "active":
            raise ValueError("episode is not active")
        cleaned = self._clean_timeframes(timeframes or ["30m", "15m", "2m", "1m"])
        minutes = max(1, min(int(minutes), 240))
        old_cursor = pd.Timestamp(episode.cursor_time)
        target = self.data.fast_forward_target(episode.symbol, old_cursor, minutes)
        cursor = old_cursor
        trade_events: list[dict[str, Any]] = []
        bracket_closed = False
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
            bracket_closed = any(
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

                bracket_closed = any(
                    event.get("event_type") == "TRADE_CLOSED"
                    and str((event.get("payload") or {}).get("exit_reason") or "")
                    in {"TAKE_PROFIT", "STOP_LOSS", "AMBIGUOUS_BOTH_HIT"}
                    for event in minute_events
                )
                if bracket_closed and self.data.auto_close_on_bracket_exit(episode.symbol):
                    break

        advanced = max(0, int((cursor - old_cursor) / pd.Timedelta(minutes=1)))
        if advanced:
            episode = self.store.update_cursor(episode_id, cursor)
        updates = (
            self.data.incremental_bars(episode.symbol, cleaned, old_cursor, cursor)
            if advanced
            else {tf: [] for tf in cleaned}
        )
        auto_finalized = False
        if bracket_closed and self.data.auto_close_on_bracket_exit(episode.symbol) and not self.store.active_trades(episode_id):
            finalized = self._finalize_episode(episode_id, reason="bracket_exit_auto", finalized_by="tp_sl_auto")
            episode = finalized["episode"]
            trade_events.extend(finalized["finalization_events"])
            auto_finalized = True
        return {
            "episode": asdict(episode),
            "clock": self.data.clock_info(episode.cursor_time, episode.symbol),
            "updates": updates,
            "trade_events": self._ui_events(trade_events),
            "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
            "active_trades": self.store.active_trades(episode_id),
            "trade_summary": self.store.trade_summary(episode_id),
            "advanced_minutes": advanced,
            "at_data_end": advanced < minutes and not auto_finalized,
            "auto_finalized": auto_finalized,
            "step_engine": "direct_1m" if minutes == 1 else "vectorized_event_driven",
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
        minutes = max(1, min(int(minutes), 240))
        old_cursor = pd.Timestamp(episode.cursor_time)
        start_cursor = pd.Timestamp(episode.start_time)
        target = max(start_cursor, old_cursor - pd.Timedelta(minutes=minutes))
        rewound = int((old_cursor - target) / pd.Timedelta(minutes=1))
        if rewound <= 0:
            return {
                "episode": asdict(episode),
                "clock": self.data.clock_info(episode.cursor_time, episode.symbol),
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
            "fee_round_trip_rate": DEFAULT_ROUND_TRIP_FEE_RATE,
            "entry_bar_policy": "include_from_open" if order_type == "market" else "exclude_intrabar_limit_fill_bar",
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
        gross = self._trade_returns(side, entry_price, float(exit_price))
        net = gross - DEFAULT_ROUND_TRIP_FEE_RATE
        risk = self._entry_risk(side, entry_price, initial_stop)
        pnl_per_share = (exit_price - entry_price) if side == "LONG" else (entry_price - exit_price)
        r_multiple = (pnl_per_share / risk) if risk else None
        mfe, mae = self._path_metrics(side, entry_price, prior_bars, float(exit_price))
        hold_minutes = max(0.0, (pd.Timestamp(exit_time) - pd.Timestamp(entry_time)) / pd.Timedelta(minutes=1))
        common = {
            "trade_id": trade_id,
            "side": side,
            "entry_event_id": payload.get("entry_event_id") or trade.get("entry_event_id"),
            "entry_price": entry_price,
            "entry_time": entry_time,
            "exit_price": float(exit_price),
            "exit_time": exit_time,
            "exit_reason": exit_reason,
            "resolution": resolution,
            "gross_return_pct": gross * 100.0,
            "fee_round_trip_rate": DEFAULT_ROUND_TRIP_FEE_RATE,
            "net_return_pct": net * 100.0,
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
            "MANUAL_CLOSE": "MANUAL_EXIT",
        }.get(exit_reason, "TRADE_EXIT")
        emitted.append(self.store.add_event(
            episode_id, hit_type, exit_time, timeframe=trade.get("timeframe") or "1m", price=float(exit_price), payload=common,
        ))
        emitted.append(self.store.add_event(
            episode_id, "TRADE_CLOSED", exit_time, timeframe=trade.get("timeframe") or "1m", price=float(exit_price), payload=common,
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
                "entry_context": payload.get("entry_context") or {},
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
            result = self.data.limit_order_fill(
                episode.symbol,
                str(payload.get("side") or ""),
                float(order["price"]),
                start,
                new_cursor,
            )
            if result is None:
                continue
            trigger_start = pd.Timestamp(result["trigger_bar_time"])
            fill_time = (trigger_start + pd.Timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
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
        if side not in {"LONG", "SHORT", "CLOSE"}:
            raise ValueError("side must be LONG, SHORT or CLOSE")
        order_type = str(payload.get("order_type") or "market").lower().strip()
        timeframe = str(payload.get("timeframe") or "1m")
        stop_loss = self._optional_price(payload, "stop_loss") if side != "CLOSE" else None
        take_profit = self._optional_price(payload, "take_profit") if side != "CLOSE" else None

        if side == "CLOSE":
            active = self.store.active_trades(episode_id)
            if not active:
                raise ValueError("当前没有可平的持仓")
            trade = active[-1]
            fill = self.data.execution_open(episode.symbol, episode.cursor_time)
            event = self.store.add_event(
                episode_id, side, episode.cursor_time, timeframe=timeframe, price=fill,
                payload={"fill_model": "cursor_1m_open", "order_type": "market", "trade_id": trade.get("trade_id")},
            )
            trade_payload = trade.get("payload") or {}
            path_start = pd.Timestamp(trade_payload.get("entry_time") or trade["event_time"])
            prior = self.data.closed_1m_bars(episode.symbol, path_start, pd.Timestamp(episode.cursor_time))
            lifecycle = self._close_trade(
                episode_id, trade, exit_time=episode.cursor_time, exit_price=fill,
                exit_reason="MANUAL_CLOSE", trigger_bar=None, prior_bars=prior,
            )
            events = [event, *lifecycle]
            return {
                "event": self._ui_event(event), "events": self._ui_events(events), "fill_price": fill, "status": "filled",
                "active_limit_orders": self._ui_events(self.store.active_limit_orders(episode_id)),
                "active_trades": self.store.active_trades(episode_id),
                "trade_summary": self.store.trade_summary(episode_id),
            }

        if order_type == "market":
            fill = self.data.execution_open(episode.symbol, episode.cursor_time)
            self._validate_bracket(side, fill, stop_loss, take_profit)
            trade_id = uuid.uuid4().hex[:12]
            event = self.store.add_event(
                episode_id, side, episode.cursor_time, timeframe=timeframe, price=fill,
                payload={
                    "fill_model": "cursor_1m_open", "order_type": "market", "trade_id": trade_id,
                    "stop_loss": stop_loss, "take_profit": take_profit,
                },
            )
            events = [event, *self._trade_open_events(
                episode_id, event, trade_id=trade_id, side=side, timeframe=timeframe,
                entry_price=fill, order_type="market", stop_loss=stop_loss, take_profit=take_profit,
                order_id=None, fill_model="cursor_1m_open", entry_context={},
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
        order_event = self.store.add_event(
            episode_id, "LIMIT_ORDER", episode.cursor_time, timeframe=timeframe, price=limit_price,
            payload={
                "order_id": order_id, "side": side, "order_type": "limit", "status": "pending",
                "stop_loss": stop_loss, "take_profit": take_profit, "entry_context": entry_context,
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
            events = [order_event, fill_event, *self._trade_open_events(
                episode_id, fill_event, trade_id=trade_id, side=side, timeframe=timeframe,
                entry_price=current_open, order_type="market", stop_loss=stop_loss, take_profit=take_profit,
                order_id=order_id, fill_model="marketable_limit_at_cursor_open", entry_context=entry_context,
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
                _json_response(self, {"ok": True, **APP.step(episode_id, int(payload.get("minutes", 1)), list(tfs))}); return
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
    print("[human_replay_lab] profiles: SOXL=weekday 07:30-16:00 ET; ETH/XAU=24/7 until TP/SL or manual end")
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
