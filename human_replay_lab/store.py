from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _ts(value: Any) -> str:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True)
class Episode:
    id: str
    symbol: str
    cursor_time: str
    start_time: str
    status: str
    created_at: str
    updated_at: str


class ReplayStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            # Keep the bootstrap schema compatible with Replay Lab V1.0-V1.5.
            # A pre-V1.6 events table does not have is_active yet, so an index
            # referencing that column must only be created after migration.
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS episodes (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    cursor_time TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    episode_id TEXT NOT NULL,
                    event_time TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timeframe TEXT,
                    price REAL,
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    FOREIGN KEY(episode_id) REFERENCES episodes(id)
                );
                CREATE INDEX IF NOT EXISTS idx_events_episode_id ON events(episode_id, id);
                """
            )

            columns = {row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()}
            if "is_active" not in columns:
                conn.execute("ALTER TABLE events ADD COLUMN is_active INTEGER NOT NULL DEFAULT 1")

            # Create V1.6 indexes only after every supported legacy schema has
            # been migrated. Existing rows receive DEFAULT 1 and stay active.
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_episode_active_time "
                "ON events(episode_id, is_active, event_time, id)"
            )

    def create_episode(self, symbol: str, start_time: Any) -> Episode:
        episode_id = uuid.uuid4().hex[:12]
        now = _now_iso()
        start = _ts(start_time)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO episodes(id,symbol,start_time,cursor_time,status,created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (episode_id, symbol, start, start, "active", now, now),
            )
        return self.get_episode(episode_id)

    def get_episode(self, episode_id: str) -> Episode:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM episodes WHERE id=?", (episode_id,)).fetchone()
        if row is None:
            raise KeyError(f"episode not found: {episode_id}")
        return Episode(**dict(row))

    def update_cursor(self, episode_id: str, cursor_time: Any) -> Episode:
        cursor = _ts(cursor_time)
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE episodes SET cursor_time=?, updated_at=? WHERE id=? AND status='active'",
                (cursor, _now_iso(), episode_id),
            )
            if cur.rowcount != 1:
                raise ValueError("episode is missing or not active")
        return self.get_episode(episode_id)

    def close_episode(self, episode_id: str) -> Episode:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE episodes SET status='closed', updated_at=? WHERE id=?",
                (_now_iso(), episode_id),
            )
            if cur.rowcount != 1:
                raise KeyError(f"episode not found: {episode_id}")
        return self.get_episode(episode_id)

    def add_event(
        self,
        episode_id: str,
        event_type: str,
        event_time: Any,
        *,
        timeframe: str | None = None,
        price: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_type = str(event_type).strip().upper()
        if not event_type:
            raise ValueError("event_type is required")
        payload_json = json.dumps(payload or {}, ensure_ascii=False, separators=(",", ":"))
        created_at = _now_iso()
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO events(episode_id,event_time,event_type,timeframe,price,payload_json,created_at)
                   VALUES(?,?,?,?,?,?,?)""",
                (episode_id, _ts(event_time), event_type, timeframe, price, payload_json, created_at),
            )
            event_id = int(cursor.lastrowid)
        return self.get_event(event_id)

    def get_event(self, event_id: int) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM events WHERE id=?", (int(event_id),)).fetchone()
        if row is None:
            raise KeyError(f"event not found: {event_id}")
        return self._decode_event(row)

    def list_events(self, episode_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE episode_id=? AND is_active=1 ORDER BY id",
                (episode_id,),
            ).fetchall()
        return [self._decode_event(row) for row in rows]

    def list_discarded_events(self, episode_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE episode_id=? AND is_active=0 ORDER BY id",
                (episode_id,),
            ).fetchall()
        return [self._decode_event(row) for row in rows]

    def deactivate_events_after(self, episode_id: str, cursor_time: Any) -> list[dict[str, Any]]:
        """Archive the abandoned future branch after a replay rewind."""
        cutoff = _ts(cursor_time)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events WHERE episode_id=? AND is_active=1 AND event_time>? ORDER BY id",
                (episode_id, cutoff),
            ).fetchall()
            ids = [int(row["id"]) for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"UPDATE events SET is_active=0 WHERE id IN ({placeholders})",
                    ids,
                )
        return [self._decode_event(row) for row in rows]

    def deactivate_event(self, episode_id: str, event_id: int) -> dict[str, Any]:
        """Archive one active event without physically deleting the audit trail."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM events WHERE id=? AND episode_id=?",
                (int(event_id), episode_id),
            ).fetchone()
            if row is None:
                raise KeyError(f"event not found in episode: {event_id}")
            if int(row["is_active"]) != 1:
                raise ValueError("event is already inactive")
            conn.execute(
                "UPDATE events SET is_active=0 WHERE id=? AND episode_id=? AND is_active=1",
                (int(event_id), episode_id),
            )
        return self._decode_event(row)

    @staticmethod
    def _decode_event(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item


    def unresolved_limit_orders(self, episode_id: str) -> list[dict[str, Any]]:
        """Reconstruct limit orders that have no active fill/cancel/expiry event.

        This is intentionally independent of episode status so a closed legacy
        episode can still report how many orders were left unfilled at finalization.
        """
        active: dict[str, dict[str, Any]] = {}
        for event in self.list_events(episode_id):
            payload = event.get("payload") or {}
            if event["event_type"] == "LIMIT_ORDER":
                order_id = str(payload.get("order_id") or "")
                if order_id:
                    active[order_id] = event
                continue
            if event["event_type"] in {"LIMIT_CANCEL", "LIMIT_EXPIRED"}:
                order_id = str(payload.get("order_id") or "")
                active.pop(order_id, None)
                continue
            if event["event_type"] in {"LONG", "SHORT"} and payload.get("order_type") == "limit":
                order_id = str(payload.get("order_id") or "")
                active.pop(order_id, None)
        return list(active.values())

    def active_limit_orders(self, episode_id: str) -> list[dict[str, Any]]:
        """Return currently resting limit orders for an active replay episode.

        Once an Episode is closed, unmatched orders are no longer *pending*; they
        are unfinished/unfilled historical intents. New V1.7.1 Episodes emit an
        explicit LIMIT_EXPIRED event at finalization. Legacy closed Episodes are
        interpreted the same way without rewriting their audit trail.
        """
        if self.get_episode(episode_id).status != "active":
            return []
        return self.unresolved_limit_orders(episode_id)


    def active_trades(self, episode_id: str) -> list[dict[str, Any]]:
        """Reconstruct active trades from the append-only event stream.

        V1.7 emits explicit TRADE_OPEN / TRADE_CLOSED events. Older V1.6
        episodes are still supported: their latest unmatched LONG/SHORT fill is
        exposed as a synthetic active trade so lifecycle/outcome recording can
        catch up without deleting the user's existing annotations.
        """
        events = self.list_events(episode_id)
        active: dict[str, dict[str, Any]] = {}
        represented_entry_ids: set[int] = set()
        closed_entry_ids: set[int] = set()

        for event in events:
            payload = event.get("payload") or {}
            if event["event_type"] == "TRADE_OPEN":
                trade_id = str(payload.get("trade_id") or "")
                if not trade_id:
                    continue
                entry_event_id = int(payload.get("entry_event_id") or 0)
                if entry_event_id:
                    represented_entry_ids.add(entry_event_id)
                active[trade_id] = {
                    "trade_id": trade_id,
                    "event_id": int(event["id"]),
                    "entry_event_id": entry_event_id or None,
                    "event_time": event["event_time"],
                    "timeframe": event.get("timeframe"),
                    "price": event.get("price"),
                    "payload": payload,
                    "legacy": False,
                }
            elif event["event_type"] == "TRADE_CLOSED":
                trade_id = str(payload.get("trade_id") or "")
                if trade_id:
                    active.pop(trade_id, None)
                entry_event_id = int(payload.get("entry_event_id") or 0)
                if entry_event_id:
                    closed_entry_ids.add(entry_event_id)

        # Backward compatibility for V1.6.x episodes that only stored LONG/SHORT
        # fill events plus SL/TP. We intentionally model one unmatched legacy
        # position at a time because the original UI itself exposed one Position.
        legacy_current: dict[str, Any] | None = None
        for event in events:
            et = event["event_type"]
            if et in {"LONG", "SHORT"}:
                entry_id = int(event["id"] )
                if entry_id in represented_entry_ids or entry_id in closed_entry_ids:
                    continue
                payload = event.get("payload") or {}
                trade_id = f"legacy-{entry_id}"
                legacy_current = {
                    "trade_id": trade_id,
                    "event_id": entry_id,
                    "entry_event_id": entry_id,
                    "event_time": event["event_time"],
                    "timeframe": event.get("timeframe"),
                    "price": event.get("price"),
                    "payload": {
                        "trade_id": trade_id,
                        "side": et,
                        "entry_event_id": entry_id,
                        "entry_price": event.get("price"),
                        "order_type": payload.get("order_type") or "market",
                        "order_id": payload.get("order_id"),
                        "initial_stop_loss": payload.get("stop_loss"),
                        "initial_take_profit": payload.get("take_profit"),
                        "entry_context": payload.get("entry_context") or {},
                        "legacy_migrated": True,
                    },
                    "legacy": True,
                }
            elif et == "CLOSE" and (event.get("payload") or {}).get("reason") != "episode_end":
                legacy_current = None
            elif et == "TRADE_CLOSED" and legacy_current is not None:
                payload = event.get("payload") or {}
                if int(payload.get("entry_event_id") or 0) == int(legacy_current["entry_event_id"]):
                    legacy_current = None
        if legacy_current is not None:
            active[legacy_current["trade_id"]] = legacy_current
        return list(active.values())

    def trade_summary(self, episode_id: str) -> dict[str, Any]:
        closed = [event for event in self.list_events(episode_id) if event["event_type"] == "TRADE_CLOSED"]
        wins = losses = breakeven = ambiguous = 0
        total_net_pct = 0.0
        total_r = 0.0
        r_count = 0
        for event in closed:
            payload = event.get("payload") or {}
            reason = str(payload.get("exit_reason") or "")
            if reason == "AMBIGUOUS_BOTH_HIT":
                ambiguous += 1
            net = payload.get("net_return_pct")
            if net is not None:
                net = float(net)
                total_net_pct += net
                if net > 1e-12:
                    wins += 1
                elif net < -1e-12:
                    losses += 1
                else:
                    breakeven += 1
            r = payload.get("r_multiple")
            if r is not None:
                total_r += float(r)
                r_count += 1
        active = self.active_trades(episode_id)
        episode = self.get_episode(episode_id)
        unresolved_orders = self.unresolved_limit_orders(episode_id)
        expired_orders = [event for event in self.list_events(episode_id) if event["event_type"] == "LIMIT_EXPIRED"]
        pending_orders = len(unresolved_orders) if episode.status == "active" else 0
        # V1.7.1 finalization writes LIMIT_EXPIRED explicitly. Legacy V1.7.0
        # closed Episodes may still have unmatched LIMIT_ORDER events instead.
        unfilled_orders = len(expired_orders) + (len(unresolved_orders) if episode.status == "closed" else 0)
        return {
            "closed_trades": len(closed),
            "wins": wins,
            "losses": losses,
            "breakeven": breakeven,
            "ambiguous": ambiguous,
            "active_trades": len(active),
            "pending_orders": pending_orders,
            "unfilled_orders": unfilled_orders,
            "total_net_return_pct": total_net_pct,
            "average_r": (total_r / r_count) if r_count else None,
            "latest_closed_trade": closed[-1] if closed else None,
        }

    def export_episode(self, episode_id: str) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "episode": asdict(self.get_episode(episode_id)),
            "events": self.list_events(episode_id),
            "discarded_events": self.list_discarded_events(episode_id),
            "trade_summary": self.trade_summary(episode_id),
        }
