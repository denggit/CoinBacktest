"""Local-only data coverage inspection for the portfolio research domain.

This module never downloads data and never mutates project databases. It reads
SQLite metadata with aggregate SQL, so coverage checks stay fast even when the
underlying dataset contains millions of rows.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .config import (
    DEFAULT_RESEARCH_END,
    DEFAULT_RESEARCH_START,
    DEFAULT_SYMBOL,
    DEFAULT_WARMUP_START,
    PROJECT_ROOT,
)


_TIMESTAMP_CANDIDATES: tuple[str, ...] = (
    "timestamp",
    "ts",
    "start_ts",
    "end_ts",
    "created_at",
)


@dataclass(frozen=True)
class DatasetRequirement:
    module: str
    dataset: str
    database: str
    table_pattern: str
    required_for: str
    minimum_start: pd.Timestamp | None
    minimum_end: pd.Timestamp | None
    optional: bool = False
    end_tolerance: pd.Timedelta = pd.Timedelta(minutes=1)


@dataclass(frozen=True)
class CoverageRecord:
    module: str
    dataset: str
    required_for: str
    database: str
    table: str | None
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    status: str
    reason: str
    optional: bool

    def to_dict(self) -> dict[str, object]:
        out = asdict(self)
        out["start"] = None if self.start is None else self.start.isoformat(sep=" ")
        out["end"] = None if self.end is None else self.end.isoformat(sep=" ")
        return out


def default_requirements(symbol: str = DEFAULT_SYMBOL) -> tuple[DatasetRequirement, ...]:
    safe = symbol.replace("-", "_")
    return (
        DatasetRequirement(
            module="baseline",
            dataset="ohlcv_1m",
            database="crypto_history.db",
            table_pattern=rf"^{re.escape(safe)}_1m$",
            required_for="warmup_and_full_history_baseline",
            minimum_start=DEFAULT_WARMUP_START,
            minimum_end=DEFAULT_RESEARCH_END,
            optional=True,
        ),
        DatasetRequirement(
            module="order_flow",
            dataset="trade_bars_1m",
            database="okx_trade_bars.db",
            table_pattern=rf"^{re.escape(safe)}.*1m.*$",
            required_for="full_history_order_flow",
            minimum_start=DEFAULT_WARMUP_START,
            minimum_end=DEFAULT_RESEARCH_END,
        ),
        DatasetRequirement(
            module="volatility",
            dataset="range_bars",
            database="okx_range_bars.db",
            table_pattern=rf"^{re.escape(safe)}_range_bars_.*$",
            required_for="full_history_range_state",
            minimum_start=DEFAULT_WARMUP_START,
            minimum_end=DEFAULT_RESEARCH_END,
        ),
        DatasetRequirement(
            module="positioning",
            dataset="open_interest",
            database="okx_derivatives.db",
            table_pattern=r"^open_interest$",
            required_for="positioning_window_only",
            minimum_start=None,
            minimum_end=None,
            optional=True,
        ),
        DatasetRequirement(
            module="positioning",
            dataset="funding_rate",
            database="okx_derivatives.db",
            table_pattern=r"^funding_rate$",
            required_for="positioning_window_only",
            minimum_start=None,
            minimum_end=None,
            optional=True,
        ),
        DatasetRequirement(
            module="positioning",
            dataset="mark_price",
            database="okx_derivatives.db",
            table_pattern=r"^mark_price$",
            required_for="positioning_window_only",
            minimum_start=None,
            minimum_end=None,
            optional=True,
        ),
        DatasetRequirement(
            module="positioning",
            dataset="liquidation",
            database="okx_derivatives.db",
            table_pattern=r"^liquidation$",
            required_for="liquidation_conditioning",
            minimum_start=None,
            minimum_end=None,
            optional=True,
        ),
    )


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _table_names(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in conn.execute(f"PRAGMA table_info({_quote_identifier(table)})")]


def _timestamp_column(columns: Sequence[str]) -> str | None:
    for candidate in _TIMESTAMP_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def _parse_timestamp(value: object) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            numeric = float(value)
            unit = "ms" if abs(numeric) > 10_000_000_000 else "s"
            return pd.to_datetime(numeric, unit=unit, errors="coerce")
        ts = pd.to_datetime(value, errors="coerce")
        return None if pd.isna(ts) else pd.Timestamp(ts)
    except (TypeError, ValueError, OverflowError):
        return None


def _status(
    *,
    rows: int,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    requirement: DatasetRequirement,
) -> tuple[str, str]:
    if rows <= 0:
        return ("MISSING_OPTIONAL" if requirement.optional else "BLOCKED", "no rows")
    failures: list[str] = []
    if requirement.minimum_start is not None and (start is None or start > requirement.minimum_start):
        failures.append(f"starts after {requirement.minimum_start}")
    if requirement.minimum_end is not None and (
        end is None or end + requirement.end_tolerance < requirement.minimum_end
    ):
        failures.append(f"ends before {requirement.minimum_end} (tolerance={requirement.end_tolerance})")
    if failures:
        return ("PARTIAL_OPTIONAL" if requirement.optional else "PARTIAL", "; ".join(failures))
    if requirement.minimum_start is None or requirement.minimum_end is None:
        return "WINDOW_ONLY", "available; use only overlapping local coverage"
    return "READY", "required coverage present"


def inspect_requirement(data_dir: Path, requirement: DatasetRequirement) -> list[CoverageRecord]:
    db_path = data_dir / requirement.database
    if not db_path.exists():
        return [
            CoverageRecord(
                module=requirement.module,
                dataset=requirement.dataset,
                required_for=requirement.required_for,
                database=str(db_path),
                table=None,
                rows=0,
                start=None,
                end=None,
                status="MISSING_OPTIONAL" if requirement.optional else "BLOCKED",
                reason="database not found",
                optional=requirement.optional,
            )
        ]

    pattern = re.compile(requirement.table_pattern, flags=re.IGNORECASE)
    records: list[CoverageRecord] = []
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        matches = [table for table in _table_names(conn) if pattern.search(table)]
        if not matches:
            return [
                CoverageRecord(
                    module=requirement.module,
                    dataset=requirement.dataset,
                    required_for=requirement.required_for,
                    database=str(db_path),
                    table=None,
                    rows=0,
                    start=None,
                    end=None,
                    status="MISSING_OPTIONAL" if requirement.optional else "BLOCKED",
                    reason="matching table not found",
                    optional=requirement.optional,
                )
            ]
        for table in matches:
            columns = _columns(conn, table)
            ts_col = _timestamp_column(columns)
            quoted_table = _quote_identifier(table)
            if ts_col is None:
                rows = int(conn.execute(f"SELECT COUNT(*) FROM {quoted_table}").fetchone()[0])
                start = end = None
            else:
                quoted_ts = _quote_identifier(ts_col)
                row = conn.execute(
                    f"SELECT COUNT(*), MIN({quoted_ts}), MAX({quoted_ts}) FROM {quoted_table}"
                ).fetchone()
                rows = int(row[0] or 0)
                start = _parse_timestamp(row[1])
                end = _parse_timestamp(row[2])
            status, reason = _status(rows=rows, start=start, end=end, requirement=requirement)
            records.append(
                CoverageRecord(
                    module=requirement.module,
                    dataset=requirement.dataset,
                    required_for=requirement.required_for,
                    database=str(db_path),
                    table=table,
                    rows=rows,
                    start=start,
                    end=end,
                    status=status,
                    reason=reason,
                    optional=requirement.optional,
                )
            )
    return records


def audit_local_coverage(
    *,
    data_dir: Path | None = None,
    requirements: Iterable[DatasetRequirement] | None = None,
) -> list[CoverageRecord]:
    root = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data"
    include_default_books = requirements is None
    selected = tuple(requirements) if requirements is not None else default_requirements()
    records: list[CoverageRecord] = []
    for requirement in selected:
        records.extend(inspect_requirement(root, requirement))
    if include_default_books:
        records.append(inspect_local_books(root, symbol=DEFAULT_SYMBOL))
    return records


def coverage_frame(records: Iterable[CoverageRecord]) -> pd.DataFrame:
    frame = pd.DataFrame([record.to_dict() for record in records])
    if frame.empty:
        return pd.DataFrame(
            columns=(
                "module",
                "dataset",
                "required_for",
                "database",
                "table",
                "rows",
                "start",
                "end",
                "status",
                "reason",
                "optional",
            )
        )
    return frame.sort_values(["module", "dataset", "table"], na_position="last").reset_index(drop=True)


def inspect_local_books(data_dir: Path, *, symbol: str) -> CoverageRecord:
    """Inspect local raw books archives without parsing their large contents.

    Historical books are stored by :class:`src.data_feed.okx_books_loader.OKXBooksLoader`
    under ``data/okx/raw/books/<symbol>`` rather than in SQLite.  File names carry
    the archive day, so a bounded directory scan is sufficient for coverage gating.
    """

    raw_dir = data_dir / "okx" / "raw" / "books" / symbol
    if not raw_dir.exists():
        return CoverageRecord(
            module="liquidity", dataset="historical_books", required_for="books_window_only",
            database=str(raw_dir), table=None, rows=0, start=None, end=None,
            status="MISSING_OPTIONAL", reason="raw books directory not found", optional=True,
        )
    date_re = re.compile(r"(20\d{2}-\d{2}-\d{2})")
    dates: list[pd.Timestamp] = []
    file_count = 0
    for path in raw_dir.rglob("*"):
        if not path.is_file() or path.stat().st_size <= 0:
            continue
        match = date_re.search(path.name)
        if match is None:
            continue
        ts = pd.to_datetime(match.group(1), errors="coerce")
        if pd.isna(ts):
            continue
        file_count += 1
        dates.append(pd.Timestamp(ts))
    if not dates:
        return CoverageRecord(
            module="liquidity", dataset="historical_books", required_for="books_window_only",
            database=str(raw_dir), table=None, rows=0, start=None, end=None,
            status="MISSING_OPTIONAL", reason="no dated raw books archives found", optional=True,
        )
    return CoverageRecord(
        module="liquidity", dataset="historical_books", required_for="books_window_only",
        database=str(raw_dir), table="raw_archives", rows=file_count, start=min(dates), end=max(dates),
        status="WINDOW_ONLY", reason="local raw books archives; use only actual overlap window", optional=True,
    )


def overall_gate(records: Iterable[CoverageRecord]) -> str:
    """Return the mandatory gate using any-ready semantics per logical dataset.

    Multiple cache tables can coexist (for example the preferred tzplus8 table and
    an older incomplete UTC table).  A stale alternative must not invalidate a ready
    canonical table.
    """

    groups: dict[tuple[str, str, str], list[str]] = {}
    for record in records:
        if record.optional:
            continue
        key = (record.module, record.dataset, record.required_for)
        groups.setdefault(key, []).append(record.status)
    resolved: list[str] = []
    for statuses in groups.values():
        if "READY" in statuses:
            resolved.append("READY")
        elif "PARTIAL" in statuses:
            resolved.append("PARTIAL")
        else:
            resolved.append("BLOCKED")
    if "BLOCKED" in resolved:
        return "BLOCKED"
    if "PARTIAL" in resolved:
        return "PARTIAL"
    return "READY"
