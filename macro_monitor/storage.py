from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from .models import Observation


def _parse_utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class MacroStore:
    def __init__(self, path: Path, retention_days: int = 365) -> None:
        self.path = Path(path)
        self.retention_days = retention_days
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=10.0)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA busy_timeout=10000")
        self._create_schema()
        self.prune_if_due()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                source TEXT NOT NULL,
                metric TEXT NOT NULL,
                meeting_date TEXT,
                target_range TEXT,
                value REAL,
                status TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_observations_metric_time
                ON observations(metric, timestamp_utc DESC);
            CREATE INDEX IF NOT EXISTS idx_observations_fedwatch_window
                ON observations(metric, meeting_date, timestamp_utc DESC);
            CREATE TABLE IF NOT EXISTS alert_cooldowns (
                alert_key TEXT PRIMARY KEY,
                severity INTEGER NOT NULL,
                sent_at_utc TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        self.connection.commit()

    def insert_observations(self, rows: Iterable[Observation]) -> int:
        payload = [
            (row.timestamp_utc, row.source, row.metric, row.meeting_date, row.target_range, row.value, row.status)
            for row in rows
        ]
        if not payload:
            return 0
        with self.connection:
            self.connection.executemany(
                """
                INSERT INTO observations
                    (timestamp_utc, source, metric, meeting_date, target_range, value, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                payload,
            )
        return len(payload)

    def window_value(
        self,
        metric: str,
        now: str | datetime,
        window_minutes: int,
        *,
        meeting_date: str | None = None,
        target_range: str | None = None,
        tolerance_seconds: int = 120,
    ) -> Observation | None:
        now_dt = _parse_utc(now)
        target = now_dt - timedelta(minutes=window_minutes)
        lower = target - timedelta(seconds=tolerance_seconds)
        upper = target + timedelta(seconds=tolerance_seconds)
        clauses = ["metric = ?", "status = 'ok'", "value IS NOT NULL", "timestamp_utc BETWEEN ? AND ?"]
        params: list[object] = [metric, _iso(lower), _iso(upper)]
        if meeting_date is not None:
            clauses.append("meeting_date = ?")
            params.append(meeting_date)
        if target_range is not None:
            clauses.append("target_range = ?")
            params.append(target_range)
        params.append(_iso(target))
        row = self.connection.execute(
            f"""
            SELECT timestamp_utc, source, metric, meeting_date, target_range, value, status
            FROM observations
            WHERE {' AND '.join(clauses)}
            ORDER BY ABS((julianday(timestamp_utc) - julianday(?)) * 86400.0), timestamp_utc DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
        return Observation(**dict(row)) if row else None

    def fedwatch_observations_at(self, meeting_date: str, timestamp_utc: str) -> list[Observation]:
        rows = self.connection.execute(
            """
            SELECT timestamp_utc, source, metric, meeting_date, target_range, value, status
            FROM observations
            WHERE meeting_date=? AND timestamp_utc=? AND status='ok' AND value IS NOT NULL
              AND metric IN (
                  'fedwatch_cut_probability',
                  'fedwatch_hold_probability',
                  'fedwatch_hike_probability',
                  'fedwatch_expected_rate',
                  'fedwatch_target_probability'
              )
            ORDER BY id ASC
            """,
            (meeting_date, timestamp_utc),
        ).fetchall()
        return [Observation(**dict(row)) for row in rows]

    def cooldown_allows(self, alert_key: str, severity: int, now: str | datetime, cooldown_seconds: int) -> bool:
        now_dt = _parse_utc(now)
        row = self.connection.execute(
            "SELECT severity, sent_at_utc FROM alert_cooldowns WHERE alert_key = ?",
            (alert_key,),
        ).fetchone()
        if row:
            last_dt = _parse_utc(row["sent_at_utc"])
            if now_dt - last_dt < timedelta(seconds=cooldown_seconds) and severity <= int(row["severity"]):
                return False
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO alert_cooldowns(alert_key, severity, sent_at_utc) VALUES (?, ?, ?)
                ON CONFLICT(alert_key) DO UPDATE SET severity=excluded.severity, sent_at_utc=excluded.sent_at_utc
                """,
                (alert_key, severity, _iso(now_dt)),
            )
        return True

    def count_observations(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM observations").fetchone()[0])

    def prune_if_due(self, now: datetime | None = None) -> int:
        if self.retention_days <= 0:
            return 0
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        row = self.connection.execute("SELECT value FROM metadata WHERE key='last_prune_date'").fetchone()
        if row and row["value"] == now.date().isoformat():
            return 0
        cutoff = _iso(now - timedelta(days=self.retention_days))
        with self.connection:
            cursor = self.connection.execute("DELETE FROM observations WHERE timestamp_utc < ?", (cutoff,))
            self.connection.execute(
                """
                INSERT INTO metadata(key, value) VALUES ('last_prune_date', ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
                """,
                (now.date().isoformat(),),
            )
        return max(0, int(cursor.rowcount))

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "MacroStore":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
