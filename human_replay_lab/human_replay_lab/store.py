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
                    FOREIGN KEY(episode_id) REFERENCES episodes(id)
                );
                CREATE INDEX IF NOT EXISTS idx_events_episode_id ON events(episode_id, id);
                """
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
            rows = conn.execute("SELECT * FROM events WHERE episode_id=? ORDER BY id", (episode_id,)).fetchall()
        return [self._decode_event(row) for row in rows]

    @staticmethod
    def _decode_event(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def export_episode(self, episode_id: str) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "episode": asdict(self.get_episode(episode_id)),
            "events": self.list_events(episode_id),
        }
