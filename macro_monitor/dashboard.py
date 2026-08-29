from __future__ import annotations

import json
import logging
import mimetypes
import sqlite3
import threading
import time
from math import ceil
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qs, urlsplit

from .alerts import classify_macro
from .config import Thresholds


STATIC_DIR = Path(__file__).resolve().parent / "static"
METRIC_DEFINITIONS = {
    "fedwatch_cut_probability": {"label": "FedWatch Cut", "unit": "%", "windows": (15, 60), "factor": 1.0},
    "fedwatch_hold_probability": {"label": "FedWatch Hold", "unit": "%", "windows": (15, 60), "factor": 1.0},
    "fedwatch_hike_probability": {"label": "FedWatch Hike", "unit": "%", "windows": (15, 60), "factor": 1.0},
    "us2y_yield": {"label": "US 2Y", "unit": "%", "windows": (5, 15, 60), "factor": 100.0},
    "us10y_yield": {"label": "US 10Y", "unit": "%", "windows": (5, 15, 60), "factor": 100.0},
    "us10y_2y_spread": {"label": "10Y–2Y", "unit": "bp", "windows": (5, 15, 60), "factor": 100.0},
    "dxy_index": {"label": "DXY", "unit": "", "windows": (5, 15, 60), "change_kind": "percent"},
}

HISTORY_METRICS = {
    "fedwatch_bias": {"label": "FedWatch Policy Bias", "unit": "pct", "decimals": 1, "change_kind": "absolute"},
    "us2y_yield": {"label": "US 2Y Yield", "unit": "%", "decimals": 3, "change_kind": "bp"},
    "us10y_yield": {"label": "US 10Y Yield", "unit": "%", "decimals": 3, "change_kind": "bp"},
    "dxy_index": {"label": "US Dollar Index", "unit": "DXY", "decimals": 3, "change_kind": "percent"},
}
HISTORY_RANGES_HOURS = {"4h": 4, "24h": 24, "7d": 24 * 7, "30d": 24 * 30}
MAX_HISTORY_POINTS = 1800


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class DashboardDataService:
    """Build immutable dashboard snapshots from the monitor's SQLite database."""

    def __init__(
        self,
        db_path: Path,
        thresholds: Thresholds | None = None,
        *,
        now_provider: Callable[[], datetime] = _utc_now,
        series_minutes: int = 240,
    ) -> None:
        self.db_path = Path(db_path)
        self.thresholds = thresholds or Thresholds()
        self.now_provider = now_provider
        self.series_minutes = series_minutes

    def snapshot(self) -> dict[str, object]:
        now = self.now_provider().astimezone(timezone.utc)
        if not self.db_path.exists():
            return self._waiting_snapshot(now, "等待监控后端创建数据库")
        try:
            connection = self._connect()
        except sqlite3.Error as exc:
            return self._waiting_snapshot(now, f"数据库暂不可读：{exc}")
        try:
            if not self._has_observation_table(connection):
                return self._waiting_snapshot(now, "等待监控后端初始化数据表")
            metrics = {
                metric: self._metric_snapshot(connection, metric, definition)
                for metric, definition in METRIC_DEFINITIONS.items()
            }
            probabilities, meeting = self._probabilities(connection, metrics["fedwatch_cut_probability"])
            fedwatch = self._fedwatch_context(metrics, probabilities)
            series = {
                "fedwatch": self._series(connection, "fedwatch_cut_probability", now),
                "fedwatch_bias": self._fedwatch_bias_series(connection, now),
                "us2y": self._series(connection, "us2y_yield", now),
                "us10y": self._series(connection, "us10y_yield", now),
                "spread": self._series(connection, "us10y_2y_spread", now, factor=100.0),
                "dxy": self._series(connection, "dxy_index", now),
            }
            sources = self._source_health(metrics, now)
            regime = self._regime(metrics, fedwatch, sources)
            automatic_regime = self._automatic_regime(metrics, fedwatch)
            guidance = self._guidance(regime, metrics, fedwatch, probabilities, sources)
            revision_row = connection.execute("SELECT COALESCE(MAX(id), 0), COUNT(*) FROM observations").fetchone()
            revision = f"{revision_row[0]}:{revision_row[1]}"
            last_timestamp = max(
                (item["timestamp_utc"] for item in metrics.values() if item.get("timestamp_utc")),
                default=None,
            )
            return {
                "generated_at_utc": _iso(now),
                "revision": revision,
                "database": {"ready": True, "path": str(self.db_path)},
                "connection": {
                    "state": "live" if all(item["state"] != "stale" for item in sources) else "degraded",
                    "last_observation_utc": last_timestamp,
                },
                "meeting": meeting,
                "metrics": metrics,
                "fedwatch": fedwatch,
                "probabilities": probabilities,
                "series": series,
                "regime": regime,
                "automatic_regime": automatic_regime,
                "guidance": guidance,
                "sources": sources,
                "thresholds": asdict(self.thresholds),
            }
        except sqlite3.Error as exc:
            return self._waiting_snapshot(now, f"读取数据时发生暂时错误：{exc}")
        finally:
            connection.close()

    def history(self, metric: str, range_key: str) -> dict[str, object]:
        if metric not in HISTORY_METRICS:
            raise ValueError(f"Unsupported history metric: {metric}")
        if range_key not in HISTORY_RANGES_HOURS:
            raise ValueError(f"Unsupported history range: {range_key}")
        now = self.now_provider().astimezone(timezone.utc)
        definition = HISTORY_METRICS[metric]
        hours = HISTORY_RANGES_HOURS[range_key]
        cutoff = _iso(now - timedelta(hours=hours))
        bucket_seconds = max(1, ceil(hours * 3600 / MAX_HISTORY_POINTS))
        if not self.db_path.exists():
            return self._empty_history(metric, range_key, now, "等待监控后端创建数据库")
        try:
            connection = self._connect()
        except sqlite3.Error as exc:
            return self._empty_history(metric, range_key, now, f"数据库暂不可读：{exc}")
        try:
            if not self._has_observation_table(connection):
                return self._empty_history(metric, range_key, now, "等待监控后端初始化数据表")
            if metric == "fedwatch_bias":
                filtered_sql = """
                    SELECT cut.id, cut.timestamp_utc, cut.source, cut.meeting_date,
                           cut.value - hike.value AS value
                    FROM observations AS cut
                    JOIN observations AS hike
                      ON hike.timestamp_utc=cut.timestamp_utc
                     AND hike.meeting_date=cut.meeting_date
                     AND hike.metric='fedwatch_hike_probability'
                     AND hike.status='ok'
                     AND hike.value IS NOT NULL
                    WHERE cut.metric='fedwatch_cut_probability'
                      AND cut.status='ok'
                      AND cut.value IS NOT NULL
                      AND cut.timestamp_utc>=?
                """
                filter_params: tuple[object, ...] = (cutoff,)
            else:
                filtered_sql = """
                    SELECT id, timestamp_utc, source, meeting_date, value
                    FROM observations
                    WHERE metric=? AND status='ok' AND value IS NOT NULL
                      AND timestamp_utc>=?
                """
                filter_params = (metric, cutoff)

            rows = connection.execute(
                f"""
                WITH filtered AS ({filtered_sql}),
                ranked AS (
                    SELECT id, timestamp_utc, source, meeting_date, value,
                           ROW_NUMBER() OVER (
                               PARTITION BY CAST(unixepoch(timestamp_utc) / ? AS INTEGER)
                               ORDER BY timestamp_utc DESC, id DESC
                           ) AS row_number
                    FROM filtered
                )
                SELECT timestamp_utc, source, meeting_date, value
                FROM ranked
                WHERE row_number=1
                ORDER BY timestamp_utc ASC
                """,
                (*filter_params, bucket_seconds),
            ).fetchall()
            if len(rows) > MAX_HISTORY_POINTS:
                last_index = len(rows) - 1
                rows = [
                    rows[round(index * last_index / (MAX_HISTORY_POINTS - 1))]
                    for index in range(MAX_HISTORY_POINTS)
                ]
            stats = connection.execute(
                f"""
                WITH filtered AS ({filtered_sql})
                SELECT COUNT(*) AS raw_count,
                       MIN(value) AS minimum,
                       MAX(value) AS maximum,
                       (SELECT value FROM filtered ORDER BY timestamp_utc ASC, id ASC LIMIT 1) AS first_value,
                       (SELECT value FROM filtered ORDER BY timestamp_utc DESC, id DESC LIMIT 1) AS last_value
                FROM filtered
                """,
                filter_params,
            ).fetchone()
            first_value = float(stats["first_value"]) if stats and stats["first_value"] is not None else None
            last_value = float(stats["last_value"]) if stats and stats["last_value"] is not None else None
            change: float | None = None
            change_unit = str(definition["unit"])
            if first_value is not None and last_value is not None:
                if definition["change_kind"] == "bp":
                    change = (last_value - first_value) * 100.0
                    change_unit = "bp"
                elif definition["change_kind"] == "percent":
                    change = ((last_value / first_value) - 1.0) * 100.0 if first_value else None
                    change_unit = "%"
                else:
                    change = last_value - first_value
            points = [
                {"timestamp_utc": row["timestamp_utc"], "value": float(row["value"])}
                for row in rows
            ]
            latest = rows[-1] if rows else None
            return {
                "ready": bool(points),
                "detail": None if points else "所选范围暂无历史数据",
                "metric": metric,
                "label": definition["label"],
                "unit": definition["unit"],
                "decimals": definition["decimals"],
                "range": range_key,
                "range_hours": hours,
                "generated_at_utc": _iso(now),
                "source": latest["source"] if latest else None,
                "meeting_date": latest["meeting_date"] if latest else None,
                "raw_count": int(stats["raw_count"]) if stats else 0,
                "returned_count": len(points),
                "bucket_seconds": bucket_seconds,
                "first_timestamp_utc": points[0]["timestamp_utc"] if points else None,
                "last_timestamp_utc": points[-1]["timestamp_utc"] if points else None,
                "current": last_value,
                "minimum": float(stats["minimum"]) if stats and stats["minimum"] is not None else None,
                "maximum": float(stats["maximum"]) if stats and stats["maximum"] is not None else None,
                "period_change": change,
                "change_unit": change_unit,
                "points": points,
            }
        except sqlite3.Error as exc:
            return self._empty_history(metric, range_key, now, f"读取历史时发生暂时错误：{exc}")
        finally:
            connection.close()

    def _empty_history(self, metric: str, range_key: str, now: datetime, detail: str) -> dict[str, object]:
        definition = HISTORY_METRICS[metric]
        return {
            "ready": False,
            "detail": detail,
            "metric": metric,
            "label": definition["label"],
            "unit": definition["unit"],
            "decimals": definition["decimals"],
            "range": range_key,
            "range_hours": HISTORY_RANGES_HOURS[range_key],
            "generated_at_utc": _iso(now),
            "source": None,
            "meeting_date": None,
            "raw_count": 0,
            "returned_count": 0,
            "bucket_seconds": None,
            "first_timestamp_utc": None,
            "last_timestamp_utc": None,
            "current": None,
            "minimum": None,
            "maximum": None,
            "period_change": None,
            "change_unit": {"bp": "bp", "percent": "%"}.get(str(definition["change_kind"]), definition["unit"]),
            "points": [],
        }

    def _connect(self) -> sqlite3.Connection:
        uri = self.db_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=3.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=3000")
        return connection

    @staticmethod
    def _has_observation_table(connection: sqlite3.Connection) -> bool:
        return connection.execute(
            "SELECT 1 FROM sqlite_schema WHERE type='table' AND name='observations'"
        ).fetchone() is not None

    def _metric_snapshot(self, connection: sqlite3.Connection, metric: str, definition: dict[str, object]) -> dict[str, object]:
        row = connection.execute(
            """
            SELECT timestamp_utc, source, metric, meeting_date, target_range, value, status
            FROM observations
            WHERE metric=? AND status='ok' AND value IS NOT NULL
            ORDER BY timestamp_utc DESC, id DESC
            LIMIT 1
            """,
            (metric,),
        ).fetchone()
        if row is None:
            return {
                "label": definition["label"],
                "unit": definition["unit"],
                "value": None,
                "timestamp_utc": None,
                "source": None,
                "meeting_date": None,
                "changes": {},
            }
        current_raw = float(row["value"])
        changes: dict[str, float | None] = {}
        for minutes in definition["windows"]:
            previous = self._window_value(
                connection,
                metric,
                row["timestamp_utc"],
                int(minutes),
                meeting_date=row["meeting_date"] if metric.startswith("fedwatch_") else None,
                tolerance_seconds=180 if metric.startswith("fedwatch_") else 90,
            )
            if previous is None:
                changes[f"{minutes}m"] = None
            elif definition.get("change_kind") == "percent":
                changes[f"{minutes}m"] = ((current_raw / previous) - 1.0) * 100.0 if previous else None
            else:
                changes[f"{minutes}m"] = (current_raw - previous) * float(definition["factor"])
        value = current_raw
        if metric == "us10y_2y_spread":
            value *= 100.0
        return {
            "label": definition["label"],
            "unit": definition["unit"],
            "value": value,
            "timestamp_utc": row["timestamp_utc"],
            "source": row["source"],
            "meeting_date": row["meeting_date"],
            "changes": changes,
        }

    @staticmethod
    def _window_value(
        connection: sqlite3.Connection,
        metric: str,
        current_timestamp: str,
        window_minutes: int,
        *,
        meeting_date: str | None,
        tolerance_seconds: int,
    ) -> float | None:
        current = _parse_utc(current_timestamp)
        target = current - timedelta(minutes=window_minutes)
        lower = _iso(target - timedelta(seconds=tolerance_seconds))
        upper = _iso(target + timedelta(seconds=tolerance_seconds))
        clauses = ["metric=?", "status='ok'", "value IS NOT NULL", "timestamp_utc BETWEEN ? AND ?"]
        params: list[object] = [metric, lower, upper]
        if meeting_date:
            clauses.append("meeting_date=?")
            params.append(meeting_date)
        params.append(_iso(target))
        row = connection.execute(
            f"""
            SELECT value
            FROM observations
            WHERE {' AND '.join(clauses)}
            ORDER BY ABS((julianday(timestamp_utc) - julianday(?)) * 86400.0), timestamp_utc DESC
            LIMIT 1
            """,
            params,
        ).fetchone()
        return float(row["value"]) if row else None

    @staticmethod
    def _probabilities(
        connection: sqlite3.Connection,
        fedwatch_metric: dict[str, object],
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        meeting_date = fedwatch_metric.get("meeting_date")
        if not meeting_date:
            return [], {"date": None, "most_likely_range": None, "most_likely_probability": None}
        stamp_row = connection.execute(
            """
            SELECT MAX(timestamp_utc) AS timestamp_utc
            FROM observations
            WHERE metric='fedwatch_target_probability' AND status='ok' AND meeting_date=?
            """,
            (meeting_date,),
        ).fetchone()
        if not stamp_row or not stamp_row["timestamp_utc"]:
            return [], {"date": meeting_date, "most_likely_range": None, "most_likely_probability": None}
        rows = connection.execute(
            """
            SELECT target_range, value
            FROM observations
            WHERE metric='fedwatch_target_probability' AND status='ok'
              AND meeting_date=? AND timestamp_utc=?
            ORDER BY CAST(substr(target_range, 1, instr(target_range, '-') - 1) AS REAL)
            """,
            (meeting_date, stamp_row["timestamp_utc"]),
        ).fetchall()
        probabilities = [
            {"target_range": row["target_range"], "probability": float(row["value"])}
            for row in rows
        ]
        likely = max(probabilities, key=lambda item: item["probability"], default=None)
        return probabilities, {
            "date": meeting_date,
            "most_likely_range": likely["target_range"] if likely else None,
            "most_likely_probability": likely["probability"] if likely else None,
        }

    def _series(
        self,
        connection: sqlite3.Connection,
        metric: str,
        now: datetime,
        *,
        factor: float = 1.0,
    ) -> list[dict[str, object]]:
        cutoff = _iso(now - timedelta(minutes=self.series_minutes))
        rows = connection.execute(
            """
            SELECT timestamp_utc, value
            FROM observations
            WHERE metric=? AND status='ok' AND value IS NOT NULL AND timestamp_utc>=?
            ORDER BY timestamp_utc ASC, id ASC
            LIMIT 2000
            """,
            (metric, cutoff),
        ).fetchall()
        return [{"timestamp_utc": row["timestamp_utc"], "value": float(row["value"]) * factor} for row in rows]

    def _fedwatch_bias_series(self, connection: sqlite3.Connection, now: datetime) -> list[dict[str, object]]:
        cutoff = _iso(now - timedelta(minutes=self.series_minutes))
        rows = connection.execute(
            """
            SELECT cut.timestamp_utc, cut.value - hike.value AS value
            FROM observations AS cut
            JOIN observations AS hike
              ON hike.timestamp_utc=cut.timestamp_utc
             AND hike.meeting_date=cut.meeting_date
             AND hike.metric='fedwatch_hike_probability'
             AND hike.status='ok'
            WHERE cut.metric='fedwatch_cut_probability'
              AND cut.status='ok'
              AND cut.timestamp_utc>=?
            ORDER BY cut.timestamp_utc ASC, cut.id ASC
            LIMIT 1000
            """,
            (cutoff,),
        ).fetchall()
        return [{"timestamp_utc": row["timestamp_utc"], "value": float(row["value"])} for row in rows]

    @staticmethod
    def _target_midpoint(target_range: str | None) -> float | None:
        if not target_range or "-" not in target_range:
            return None
        try:
            low, high = (float(part.strip()) for part in target_range.split("-", 1))
        except ValueError:
            return None
        return (low + high) / 2.0

    def _fedwatch_context(
        self,
        metrics: dict[str, dict[str, object]],
        probabilities: list[dict[str, object]],
    ) -> dict[str, object]:
        cut = metrics["fedwatch_cut_probability"].get("value")
        hold = metrics["fedwatch_hold_probability"].get("value")
        hike = metrics["fedwatch_hike_probability"].get("value")
        values = {"cut": cut, "hold": hold, "hike": hike}
        available = {key: float(value) for key, value in values.items() if value is not None}
        dominant = max(available, key=available.get) if available else None
        bias = float(cut) - float(hike) if cut is not None and hike is not None else None
        if bias is None:
            skew = "waiting"
            skew_label = "WAITING"
        elif bias >= 5.0:
            skew = "dovish"
            skew_label = "DOVISH SKEW"
        elif bias <= -5.0:
            skew = "hawkish"
            skew_label = "HAWKISH SKEW"
        else:
            skew = "balanced"
            skew_label = "BALANCED SKEW"

        bias_changes: dict[str, float | None] = {}
        cut_changes = metrics["fedwatch_cut_probability"].get("changes", {})
        hike_changes = metrics["fedwatch_hike_probability"].get("changes", {})
        for window in ("15m", "60m"):
            cut_change = cut_changes.get(window)
            hike_change = hike_changes.get(window)
            bias_changes[window] = (
                float(cut_change) - float(hike_change)
                if cut_change is not None and hike_change is not None
                else None
            )

        current_index: int | None = None
        if probabilities and cut is not None and hold is not None and hike is not None:
            best_error: float | None = None
            for index, item in enumerate(probabilities):
                lower = sum(float(row["probability"]) for row in probabilities[:index])
                current = float(item["probability"])
                upper = sum(float(row["probability"]) for row in probabilities[index + 1:])
                error = abs(lower - float(cut)) + abs(current - float(hold)) + abs(upper - float(hike))
                if best_error is None or error < best_error:
                    best_error = error
                    current_index = index
            if best_error is not None and best_error > 3.0:
                current_index = None
        current_target_range = probabilities[current_index]["target_range"] if current_index is not None else None
        current_midpoint = self._target_midpoint(str(current_target_range)) if current_target_range else None
        expected_move_bp: float | None = None
        if current_midpoint is not None:
            expected_move_bp = sum(
                float(item["probability"]) / 100.0
                * ((self._target_midpoint(str(item["target_range"])) or current_midpoint) - current_midpoint)
                * 100.0
                for item in probabilities
            )
        return {
            "cut_probability": cut,
            "hold_probability": hold,
            "hike_probability": hike,
            "dominant": dominant,
            "dominant_label": f"{dominant.upper()} DOMINANT" if dominant else "WAITING FOR DATA",
            "policy_bias": bias,
            "skew": skew,
            "skew_label": skew_label,
            "bias_changes": bias_changes,
            "current_target_range": current_target_range,
            "expected_move_bp": expected_move_bp,
        }

    @staticmethod
    def _source_health(metrics: dict[str, dict[str, object]], now: datetime) -> list[dict[str, object]]:
        definitions = (
            ("FedWatch", "fedwatch_cut_probability", 180),
            ("US 2Y", "us2y_yield", 75),
            ("US 10Y", "us10y_yield", 75),
            ("DXY", "dxy_index", 75),
        )
        result: list[dict[str, object]] = []
        for label, metric, stale_after in definitions:
            item = metrics[metric]
            timestamp = item.get("timestamp_utc")
            age = max(0.0, (now - _parse_utc(timestamp)).total_seconds()) if timestamp else None
            state = "online" if age is not None and age <= stale_after else ("stale" if timestamp else "waiting")
            source = item.get("source")
            primary_sources = {
                "fedwatch_cut_probability": "cme_fedwatch_en",
                "us2y_yield": "investing_us2y",
                "us10y_yield": "investing_us10y",
                "dxy_index": "cnbc_dxy",
            }
            fallback = bool(source and source != primary_sources[metric])
            result.append(
                {
                    "label": label,
                    "metric": metric,
                    "source": source,
                    "timestamp_utc": timestamp,
                    "age_seconds": round(age, 1) if age is not None else None,
                    "state": state,
                    "fallback": fallback,
                }
            )
        return result

    def _regime(
        self,
        metrics: dict[str, dict[str, object]],
        fedwatch: dict[str, object],
        sources: list[dict[str, object]],
    ) -> dict[str, object]:
        if any(item["state"] == "stale" for item in sources):
            return {
                "code": "data_stale",
                "label": "DATA DELAY",
                "headline": "数据更新延迟",
                "detail": "至少一个核心来源超过正常更新间隔，暂缓方向性解读。",
                "severity": 1,
                "window": None,
                "action": "先确认数据源恢复",
            }
        fed_changes = fedwatch["bias_changes"]
        us2_changes = metrics["us2y_yield"]["changes"]
        candidates: list[tuple[int, int, str, int]] = []
        for priority, minutes in enumerate((15, 60)):
            classification = classify_macro(
                fedwatch_change_pct=fed_changes.get(f"{minutes}m"),
                fedwatch_threshold_pct=getattr(self.thresholds, f"fedwatch_{minutes}m_pct"),
                us2y_change_bp=us2_changes.get(f"{minutes}m"),
                us2y_threshold_bp=getattr(self.thresholds, f"us2y_{minutes}m_bp"),
            )
            if classification:
                severity = 3 if classification.startswith("STRONG") else (1 if classification.startswith("MIXED") else 2)
                candidates.append((severity, -priority, classification, minutes))
        five_minute = classify_macro(
            fedwatch_change_pct=None,
            fedwatch_threshold_pct=self.thresholds.fedwatch_15m_pct,
            us2y_change_bp=us2_changes.get("5m"),
            us2y_threshold_bp=self.thresholds.us2y_5m_bp,
        )
        if five_minute:
            candidates.append((2, -2, five_minute, 5))
        if not candidates:
            has_window = any(value is not None for value in (*fed_changes.values(), *us2_changes.values()))
            return {
                "code": "stable" if has_window else "warming_up",
                "label": "STABLE" if has_window else "BUILDING BASELINE",
                "headline": "重新定价暂未越过阈值" if has_window else "正在建立时间窗口基准",
                "detail": "当前变化仍在默认阈值内。" if has_window else "采集保持运行后，5m、15m、60m 信号将依次可用。",
                "severity": 0,
                "window": None,
                "action": "保持事件监控" if has_window else "等待窗口成熟",
            }
        severity, _, classification, minutes = max(candidates)
        code = (
            "mixed" if classification.startswith("MIXED")
            else "dovish" if "DOVISH" in classification
            else "hawkish"
        )
        strong = classification.startswith("STRONG")
        labels = {
            "dovish": "STRONG DOVISH" if strong else "DOVISH REPRICING",
            "hawkish": "STRONG HAWKISH" if strong else "HAWKISH REPRICING",
            "mixed": "MIXED SIGNALS",
        }
        headlines = {
            "dovish": "市场正在重定价更宽松的利率路径",
            "hawkish": "市场正在重定价更紧的利率路径",
            "mixed": "FedWatch 与短端利率信号存在分歧",
        }
        actions = {
            "dovish": "提高鸽派事件监控级别" if strong else "等待第二来源确认",
            "hawkish": "提高鹰派事件监控级别" if strong else "等待第二来源确认",
            "mixed": "降低当前方向解读置信度",
        }
        return {
            "code": code,
            "label": labels[code],
            "headline": headlines[code],
                "detail": f"{minutes} 分钟窗口已越过配置阈值。" + (" FedWatch Policy Bias 与 US2Y 同向确认。" if strong else " 当前仅有单侧或分歧信号。"),
            "severity": severity,
            "window": f"{minutes}m",
            "action": actions[code],
        }

    @staticmethod
    def _direction_for_change(change: float | None, threshold: float, *, positive: str = "hawkish") -> str:
        if change is None or abs(float(change)) < threshold:
            return "neutral"
        if float(change) > 0:
            return positive
        return "dovish" if positive == "hawkish" else "hawkish"

    def _automatic_regime(
        self,
        metrics: dict[str, dict[str, object]],
        fedwatch: dict[str, object],
    ) -> dict[str, object]:
        def change(metric: str, window: str) -> float | None:
            value = metrics[metric].get("changes", {}).get(window)
            return float(value) if value is not None else None

        window_scores: list[tuple[int, int, int]] = []
        for preference, minutes in enumerate((15, 60)):
            window = f"{minutes}m"
            checks = (
                (fedwatch.get("bias_changes", {}).get(window), getattr(self.thresholds, f"fedwatch_{minutes}m_pct")),
                (change("us2y_yield", window), getattr(self.thresholds, f"us2y_{minutes}m_bp")),
                (change("us10y_yield", window), getattr(self.thresholds, f"us10y_{minutes}m_bp")),
                (change("us10y_2y_spread", window), getattr(self.thresholds, f"curve_{minutes}m_bp")),
                (change("dxy_index", window), getattr(self.thresholds, f"dxy_{minutes}m_pct")),
            )
            actionable = sum(value is not None and abs(float(value)) >= limit for value, limit in checks)
            available = sum(value is not None for value, _ in checks)
            window_scores.append((actionable, available, -preference))
        selected_index = max(range(len(window_scores)), key=window_scores.__getitem__)
        minutes = (15, 60)[selected_index]
        window = f"{minutes}m"

        bias = fedwatch.get("policy_bias")
        if bias is None:
            fed_state = "pending"
        elif float(bias) >= 5.0:
            fed_state = "dovish"
        elif float(bias) <= -5.0:
            fed_state = "hawkish"
        else:
            fed_state = "neutral"

        us2_change = change("us2y_yield", window)
        us10_change = change("us10y_yield", window)
        curve_change = change("us10y_2y_spread", window)
        dxy_change = change("dxy_index", window)
        us2_state = self._direction_for_change(
            us2_change,
            getattr(self.thresholds, f"us2y_{minutes}m_bp"),
        )
        us10_state = self._direction_for_change(
            us10_change,
            getattr(self.thresholds, f"us10y_{minutes}m_bp"),
        )
        dxy_state = self._direction_for_change(
            dxy_change,
            getattr(self.thresholds, f"dxy_{minutes}m_pct"),
        )
        if dxy_state == "neutral" or metrics["dxy_index"].get("value") is None:
            dxy_state = "pending"

        direction_votes = [state for state in (fed_state, us2_state) if state in {"hawkish", "dovish"}]
        if len(set(direction_votes)) > 1:
            direction = "mixed"
        elif direction_votes:
            direction = direction_votes[0]
        else:
            direction = "stable"

        curve_limit = getattr(self.thresholds, f"curve_{minutes}m_bp")
        if curve_change is None or abs(curve_change) < curve_limit:
            curve_shape = "unchanged"
            curve_state = "neutral" if curve_change is not None else "pending"
        elif curve_change < 0:
            curve_shape = "flattening"
            curve_state = "hawkish" if direction == "hawkish" else ("mixed" if direction != "stable" else "neutral")
        else:
            curve_shape = "steepening"
            curve_state = "dovish" if direction == "dovish" else ("mixed" if direction != "stable" else "neutral")

        if direction == "mixed":
            label = "MIXED SIGNALS"
        elif direction == "stable":
            label = "STABLE"
        elif curve_shape in {"flattening", "steepening"}:
            label = f"{direction.upper()} {curve_shape.upper()}"
        else:
            label = f"{direction.upper()} BIAS"

        def signal(key: str, label_text: str, state: str, value: float | None, delta: float | None, unit: str) -> dict[str, object]:
            return {
                "key": key,
                "label": label_text,
                "state": state,
                "value": value,
                "change": delta,
                "unit": unit,
            }

        signals = [
            signal("fedwatch", "FedWatch", fed_state, float(bias) if bias is not None else None, fedwatch.get("bias_changes", {}).get(window), "pct"),
            signal("us2y", "2Y", us2_state, metrics["us2y_yield"].get("value"), us2_change, "bp"),
            signal("us10y", "10Y", us10_state, metrics["us10y_yield"].get("value"), us10_change, "bp"),
            signal("curve", "Curve", curve_state, metrics["us10y_2y_spread"].get("value"), curve_change, "bp"),
            signal("dxy", "DXY", dxy_state, metrics["dxy_index"].get("value"), dxy_change, "%"),
        ]
        return {
            "label": label,
            "direction": direction,
            "curve_shape": curve_shape,
            "window": window,
            "signals": signals,
        }

    @staticmethod
    def _guidance(
        regime: dict[str, object],
        metrics: dict[str, dict[str, object]],
        fedwatch: dict[str, object],
        probabilities: list[dict[str, object]],
        sources: list[dict[str, object]],
    ) -> list[dict[str, str]]:
        items: list[dict[str, str]] = [
            {
                "title": str(regime["action"]),
                "detail": str(regime["detail"]),
                "tone": str(regime["code"]),
            }
        ]
        bias = fedwatch.get("policy_bias")
        expected = fedwatch.get("expected_move_bp")
        if bias is not None:
            expected_text = f"，隐含期望变动 {float(expected):+.1f} bp" if expected is not None else ""
            items.append(
                {
                    "title": f"{fedwatch['dominant_label']} · {fedwatch['skew_label']}",
                    "detail": f"Policy Bias（Cut − Hike）为 {float(bias):+.1f} pct{expected_text}。",
                    "tone": str(fedwatch["skew"]),
                }
            )
        spread = metrics["us10y_2y_spread"].get("value")
        if spread is not None:
            curve_text = (
                f"10Y–2Y 为 {spread:+.1f} bp；观察事件后曲线是否继续陡峭化。"
                if spread >= 0
                else f"10Y–2Y 仍倒挂 {abs(spread):.1f} bp；观察倒挂是否快速收窄。"
            )
            items.append({"title": "核对收益率曲线", "detail": curve_text, "tone": "curve"})
        if probabilities:
            likely = max(probabilities, key=lambda item: item["probability"])
            items.append(
                {
                    "title": "盯住概率主区间",
                    "detail": f"当前最高概率为 {likely['target_range']}%，概率 {likely['probability']:.1f}%。",
                    "tone": "fedwatch",
                }
            )
        fallback_sources = [item["label"] for item in sources if item["fallback"]]
        if fallback_sources:
            items.append(
                {
                    "title": "当前使用公开备用源",
                    "detail": f"{', '.join(fallback_sources)} 正在通过已验证 fallback 更新；建议同时关注来源状态。",
                    "tone": "source",
                }
            )
        return items[:4]

    def _waiting_snapshot(self, now: datetime, detail: str) -> dict[str, object]:
        metrics = {
            metric: {
                "label": definition["label"],
                "unit": definition["unit"],
                "value": None,
                "timestamp_utc": None,
                "source": None,
                "meeting_date": None,
                "changes": {f"{minutes}m": None for minutes in definition["windows"]},
            }
            for metric, definition in METRIC_DEFINITIONS.items()
        }
        return {
            "generated_at_utc": _iso(now),
            "revision": "waiting",
            "database": {"ready": False, "path": str(self.db_path)},
            "connection": {"state": "waiting", "last_observation_utc": None},
            "meeting": {"date": None, "most_likely_range": None, "most_likely_probability": None},
            "metrics": metrics,
            "probabilities": [],
            "series": {"fedwatch": [], "fedwatch_bias": [], "us2y": [], "us10y": [], "spread": [], "dxy": []},
            "fedwatch": {
                "cut_probability": None,
                "hold_probability": None,
                "hike_probability": None,
                "dominant": None,
                "dominant_label": "WAITING FOR DATA",
                "policy_bias": None,
                "skew": "waiting",
                "skew_label": "WAITING",
                "bias_changes": {"15m": None, "60m": None},
                "current_target_range": None,
                "expected_move_bp": None,
            },
            "regime": {
                "code": "waiting",
                "label": "WAITING FOR DATA",
                "headline": "等待监控后端数据",
                "detail": detail,
                "severity": 0,
                "window": None,
                "action": "启动或保持后端采集",
            },
            "automatic_regime": {
                "label": "WAITING FOR DATA",
                "direction": "waiting",
                "curve_shape": "pending",
                "window": "15m",
                "signals": [
                    {"key": key, "label": label, "state": "pending", "value": None, "change": None, "unit": unit}
                    for key, label, unit in (
                        ("fedwatch", "FedWatch", "pct"),
                        ("us2y", "2Y", "bp"),
                        ("us10y", "10Y", "bp"),
                        ("curve", "Curve", "bp"),
                        ("dxy", "DXY", "%"),
                    )
                ],
            },
            "guidance": [{"title": "后端与前端完全独立", "detail": detail, "tone": "waiting"}],
            "sources": [],
            "thresholds": asdict(self.thresholds),
        }


class _DashboardHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address: tuple[str, int], handler: type[BaseHTTPRequestHandler], service: DashboardDataService, logger: logging.Logger, verbose: bool) -> None:
        self.service = service
        self.logger = logger
        self.verbose = verbose
        super().__init__(server_address, handler)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: _DashboardHTTPServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP method name
        parsed_url = urlsplit(self.path)
        path = parsed_url.path
        if path == "/api/snapshot":
            self._send_json(self.server.service.snapshot())
        elif path == "/api/history":
            query = parse_qs(parsed_url.query)
            metric = query.get("metric", ["fedwatch_bias"])[0]
            range_key = query.get("range", ["24h"])[0]
            try:
                self._send_json(self.server.service.history(metric, range_key))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        elif path == "/events":
            self._serve_events()
        elif path == "/healthz":
            self._send_json({"status": "ok"})
        elif path in {"/", "/index.html"}:
            self._send_static(STATIC_DIR / "dashboard.html")
        elif path in {"/history", "/history.html"}:
            self._send_static(STATIC_DIR / "history.html")
        elif path == "/assets/dashboard.css":
            self._send_static(STATIC_DIR / "dashboard.css")
        elif path == "/assets/dashboard.js":
            self._send_static(STATIC_DIR / "dashboard.js")
        elif path == "/assets/history.css":
            self._send_static(STATIC_DIR / "history.css")
        elif path == "/assets/history.js":
            self._send_static(STATIC_DIR / "history.js")
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _base_headers(self, content_type: str, content_length: int | None = None) -> None:
        self.send_header("Content-Type", content_type)
        if content_length is not None:
            self.send_header("Content-Length", str(content_length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; font-src 'self'",
        )

    def _send_json(self, payload: dict[str, object], *, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self._base_headers("application/json; charset=utf-8", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, path: Path) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self._base_headers(f"{content_type}; charset=utf-8" if content_type.startswith("text/") or "javascript" in content_type else content_type, len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self._base_headers("text/event-stream; charset=utf-8")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        previous_revision: str | None = None
        last_heartbeat = 0.0
        try:
            while True:
                payload = self.server.service.snapshot()
                revision = str(payload["revision"])
                if revision != previous_revision:
                    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                    event = f"event: snapshot\nid: {revision}\ndata: {encoded}\n\n".encode("utf-8")
                    self.wfile.write(event)
                    self.wfile.flush()
                    previous_revision = revision
                now = time.monotonic()
                if now - last_heartbeat >= 15.0:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    last_heartbeat = now
                time.sleep(1.0)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

    def log_message(self, format: str, *args: object) -> None:
        if self.server.verbose:
            self.server.logger.debug("[dashboard] %s", format % args)


class MacroDashboardServer:
    def __init__(
        self,
        service: DashboardDataService,
        logger: logging.Logger,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        verbose: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.logger = logger
        self.httpd = _DashboardHTTPServer((host, port), DashboardRequestHandler, service, logger, verbose)
        self.port = int(self.httpd.server_address[1])
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        display_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{display_host}:{self.port}"

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="macro-dashboard", daemon=True)
        self._thread.start()
        self.logger.info("[dashboard] live=%s", self.url)

    def wait(self) -> None:
        while self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
