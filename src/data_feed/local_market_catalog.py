"""Read-only inventory of locally cached market-data series.

The catalog is intentionally metadata-only: it discovers SQLite tables and
dated archive files, then measures coverage inside a caller-supplied time
window.  It never creates schemas, downloads data, or reads rows outside the
physical ``end_exclusive`` boundary.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


DEFAULT_MARKET_DATABASES: tuple[str, ...] = (
    "crypto_history.db",
    "okx_trade_bars.db",
    "okx_range_bars.db",
    "okx_range_footprints.db",
    "okx_derivatives.db",
    "binance_futures_metrics.db",
)

DEFAULT_ARCHIVE_LANES: tuple[tuple[str, str], ...] = (
    ("okx_raw_trades", "okx/raw/trades"),
    ("okx_raw_books", "okx/raw/books"),
    ("okx_liquidity_primitives", "okx/derived/liquidity_primitives"),
    ("okx_liquidity_map", "okx/derived/liquidity_map"),
    ("binance_futures_metrics_raw", "binance/raw/futures_metrics"),
)

_TIMESTAMP_CANDIDATES: tuple[str, ...] = (
    "timestamp",
    "end_ts",
    "start_ts",
    "ts",
    "source_timestamp_utc",
)
_SERIES_DIMENSIONS: tuple[str, ...] = (
    "symbol",
    "timeframe",
    "period",
    "range_pct",
    "price_step",
)
_ADMIN_TABLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|_)coverage$", re.IGNORECASE),
    re.compile(r"(?:^|_)state$", re.IGNORECASE),
)
_DATE_PATTERN = re.compile(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)")


@dataclass(frozen=True)
class SQLiteSeriesCoverage:
    database: str
    table: str
    series_key: str
    timestamp_column: str
    dimensions: str
    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    window_start: pd.Timestamp
    end_exclusive: pd.Timestamp
    columns: str

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("start", "end", "window_start", "end_exclusive"):
            value = out[key]
            out[key] = None if value is None else pd.Timestamp(value).isoformat(sep=" ")
        return out


@dataclass(frozen=True)
class SQLiteTableMetadata:
    database: str
    table: str
    rows: int
    columns: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArchiveSeriesCoverage:
    lane: str
    series_key: str
    root: str
    files: int
    dated_days: int
    start: date | None
    end: date | None
    expected_days: int
    missing_days: int
    coverage_ratio: float
    window_start: date
    end_exclusive: date

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("start", "end", "window_start", "end_exclusive"):
            value = out[key]
            out[key] = None if value is None else value.isoformat()
        return out


@dataclass(frozen=True)
class LocalMarketCatalog:
    sqlite_series: tuple[SQLiteSeriesCoverage, ...]
    sqlite_metadata: tuple[SQLiteTableMetadata, ...]
    archive_series: tuple[ArchiveSeriesCoverage, ...]

    def sqlite_frame(self) -> pd.DataFrame:
        return pd.DataFrame([row.to_dict() for row in self.sqlite_series])

    def metadata_frame(self) -> pd.DataFrame:
        return pd.DataFrame([row.to_dict() for row in self.sqlite_metadata])

    def archive_frame(self) -> pd.DataFrame:
        return pd.DataFrame([row.to_dict() for row in self.archive_series])


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _read_only_connection(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=30.0)


def _table_names(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        )
    ]


def _column_info(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    quoted = _quote_identifier(table)
    return [(str(row[1]), str(row[2] or "")) for row in conn.execute(f"PRAGMA table_info({quoted})")]


def _timestamp_column(columns: Sequence[str]) -> str | None:
    lower = {column.lower(): column for column in columns}
    for candidate in _TIMESTAMP_CANDIDATES:
        if candidate in lower:
            return lower[candidate]
    return None


def _is_admin_table(table: str) -> bool:
    return any(pattern.search(table) for pattern in _ADMIN_TABLE_PATTERNS)


def _timestamp_parameters(
    timestamp_column: str,
    declared_type: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[Any, Any]:
    numeric_type = any(
        token in declared_type.upper() for token in ("INT", "REAL", "NUM", "DEC")
    )
    if numeric_type:
        lower_name = timestamp_column.lower()
        if lower_name.endswith("_ns"):
            unit = "ns"
        elif lower_name.endswith("_us"):
            unit = "us"
        elif lower_name.endswith("_ms"):
            unit = "ms"
        else:
            unit = "s"
        return int(start.value // pd.Timedelta(1, unit=unit).value), int(
            end_exclusive.value // pd.Timedelta(1, unit=unit).value
        )
    # Project caches use lexicographically ordered ISO timestamps, generally
    # without a fractional suffix.  A ``.000000`` upper bound would wrongly
    # admit the shorter exact-boundary string (for example 2025-07-01 00:00:00).
    return start.strftime("%Y-%m-%d %H:%M:%S"), end_exclusive.strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _parse_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, (int, float)):
            numeric = abs(float(value))
            unit = "ns" if numeric >= 10**17 else "us" if numeric >= 10**14 else "ms" if numeric >= 10**11 else "s"
            parsed = pd.to_datetime(value, unit=unit, errors="coerce")
        else:
            parsed = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError, OverflowError):
        return None
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _dimension_clause(dimensions: Sequence[str]) -> str:
    return ", ".join(_quote_identifier(value) for value in dimensions)


def _dimension_text(dimensions: Sequence[str], values: Sequence[Any]) -> str:
    return ";".join(f"{name}={value}" for name, value in zip(dimensions, values, strict=True))


def _series_key(table: str, dimensions: Sequence[str], values: Sequence[Any]) -> str:
    suffix = _dimension_text(dimensions, values)
    return table if not suffix else f"{table}|{suffix}"


def _daily_coverage_summary(
    conn: sqlite3.Connection,
    *,
    market_table: str,
    all_tables: Sequence[str],
    window_start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None] | None:
    """Use loader-maintained daily coverage for very large derived stores."""

    candidates = (
        "trade_bar_coverage",
        "range_bar_coverage",
        "range_footprint_coverage",
    )
    for coverage_table in candidates:
        if coverage_table not in all_tables:
            continue
        columns = {name.lower(): name for name, _ in _column_info(conn, coverage_table)}
        required = {"table_name", "utc_day", "rows"}
        if not required.issubset(columns):
            continue
        quoted_coverage = _quote_identifier(coverage_table)
        table_col = _quote_identifier(columns["table_name"])
        day_col = _quote_identifier(columns["utc_day"])
        rows_col = _quote_identifier(columns["rows"])
        row = conn.execute(
            f"SELECT COALESCE(SUM(day_rows), 0), MIN(day), MAX(day) FROM ("
            f"SELECT {day_col} AS day, MAX({rows_col}) AS day_rows "
            f"FROM {quoted_coverage} WHERE {table_col} = ? AND {day_col} >= ? AND {day_col} < ? "
            f"GROUP BY {day_col})",
            (market_table, window_start.date().isoformat(), end_exclusive.date().isoformat()),
        ).fetchone()
        if row and int(row[0] or 0) > 0:
            return int(row[0]), _parse_timestamp(row[1]), _parse_timestamp(row[2])
    return None


def _indexed_boundary(
    conn: sqlite3.Connection,
    *,
    table: str,
    timestamp_column: str,
    lower_bound: Any,
    upper_bound: Any,
    descending: bool,
) -> Any:
    quoted_table = _quote_identifier(table)
    quoted_ts = _quote_identifier(timestamp_column)
    direction = "DESC" if descending else "ASC"
    row = conn.execute(
        f"SELECT {quoted_ts} FROM {quoted_table} "
        f"WHERE {quoted_ts} >= ? AND {quoted_ts} < ? "
        f"ORDER BY {quoted_ts} {direction} LIMIT 1",
        (lower_bound, upper_bound),
    ).fetchone()
    return None if row is None else row[0]


def discover_sqlite_market_series(
    data_dir: str | Path,
    *,
    window_start: Any,
    end_exclusive: Any,
    databases: Iterable[str] = DEFAULT_MARKET_DATABASES,
) -> tuple[tuple[SQLiteSeriesCoverage, ...], tuple[SQLiteTableMetadata, ...]]:
    """Discover all timestamped market series in known local market databases."""

    root = Path(data_dir)
    start_ts = pd.Timestamp(window_start)
    end_ts = pd.Timestamp(end_exclusive)
    if end_ts <= start_ts:
        raise ValueError("end_exclusive must be later than window_start")

    series: list[SQLiteSeriesCoverage] = []
    metadata: list[SQLiteTableMetadata] = []
    for database in databases:
        path = root / database
        if not path.exists():
            metadata.append(SQLiteTableMetadata(database, "", 0, "", "database_not_found"))
            continue
        with _read_only_connection(path) as conn:
            all_tables = _table_names(conn)
            for table in all_tables:
                info = _column_info(conn, table)
                columns = [name for name, _ in info]
                declared_types = {name: declared for name, declared in info}
                columns_text = ",".join(columns)
                timestamp_column = _timestamp_column(columns)
                quoted_table = _quote_identifier(table)
                if timestamp_column is None or _is_admin_table(table):
                    reason = "administrative_table" if _is_admin_table(table) else "no_market_timestamp_column"
                    # Some auxiliary tables (for example per-price footprint
                    # buckets) can be much larger than their parent bar table.
                    # Their exact row count is not needed for source readiness.
                    metadata.append(SQLiteTableMetadata(database, table, -1, columns_text, reason))
                    continue

                lower = {column.lower(): column for column in columns}
                dimensions = [lower[name] for name in _SERIES_DIMENSIONS if name in lower]
                lower_bound, upper_bound = _timestamp_parameters(
                    timestamp_column,
                    declared_types.get(timestamp_column, ""),
                    start_ts,
                    end_ts,
                )
                quoted_ts = _quote_identifier(timestamp_column)
                coverage_summary = _daily_coverage_summary(
                    conn,
                    market_table=table,
                    all_tables=all_tables,
                    window_start=start_ts,
                    end_exclusive=end_ts,
                )
                if coverage_summary is not None:
                    count, _, _ = coverage_summary
                    first = _indexed_boundary(
                        conn,
                        table=table,
                        timestamp_column=timestamp_column,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        descending=False,
                    )
                    last = _indexed_boundary(
                        conn,
                        table=table,
                        timestamp_column=timestamp_column,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        descending=True,
                    )
                    series.append(
                        SQLiteSeriesCoverage(
                            database=database,
                            table=table,
                            series_key=table,
                            timestamp_column=timestamp_column,
                            dimensions="",
                            rows=count,
                            start=_parse_timestamp(first),
                            end=_parse_timestamp(last),
                            window_start=start_ts,
                            end_exclusive=end_ts,
                            columns=columns_text,
                        )
                    )
                    continue
                if not dimensions and database != "crypto_history.db":
                    first = _indexed_boundary(
                        conn,
                        table=table,
                        timestamp_column=timestamp_column,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        descending=False,
                    )
                    if first is None:
                        continue
                    last = _indexed_boundary(
                        conn,
                        table=table,
                        timestamp_column=timestamp_column,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        descending=True,
                    )
                    series.append(
                        SQLiteSeriesCoverage(
                            database=database,
                            table=table,
                            series_key=table,
                            timestamp_column=timestamp_column,
                            dimensions="",
                            rows=-1,
                            start=_parse_timestamp(first),
                            end=_parse_timestamp(last),
                            window_start=start_ts,
                            end_exclusive=end_ts,
                            columns=columns_text,
                        )
                    )
                    continue
                group_select = _dimension_clause(dimensions)
                prefix = f"{group_select}, " if group_select else ""
                group_by = f" GROUP BY {group_select}" if group_select else ""
                query = (
                    f"SELECT {prefix}COUNT(*), MIN({quoted_ts}), MAX({quoted_ts}) "
                    f"FROM {quoted_table} WHERE {quoted_ts} >= ? AND {quoted_ts} < ?{group_by}"
                )
                query_rows = conn.execute(query, (lower_bound, upper_bound)).fetchall()
                emitted = False
                for row in query_rows:
                    dim_values = row[: len(dimensions)]
                    count, first, last = row[len(dimensions) :]
                    if int(count or 0) <= 0:
                        continue
                    emitted = True
                    series.append(
                        SQLiteSeriesCoverage(
                            database=database,
                            table=table,
                            series_key=_series_key(table, dimensions, dim_values),
                            timestamp_column=timestamp_column,
                            dimensions=_dimension_text(dimensions, dim_values),
                            rows=int(count),
                            start=_parse_timestamp(first),
                            end=_parse_timestamp(last),
                            window_start=start_ts,
                            end_exclusive=end_ts,
                            columns=columns_text,
                        )
                    )
                if not emitted:
                    metadata.append(
                        SQLiteTableMetadata(
                            database, table, 0, columns_text, "no_rows_in_window"
                        )
                    )
    return tuple(series), tuple(metadata)


def _date_from_path(path: Path) -> date | None:
    for text in (path.name, *reversed(path.parts)):
        match = _DATE_PATTERN.search(text)
        if match is None:
            continue
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            continue
    return None


def discover_archive_market_series(
    data_dir: str | Path,
    *,
    window_start: Any,
    end_exclusive: Any,
    archive_lanes: Iterable[tuple[str, str]] = DEFAULT_ARCHIVE_LANES,
) -> tuple[ArchiveSeriesCoverage, ...]:
    """Inventory dated files without opening archive contents."""

    root = Path(data_dir)
    start_day = pd.Timestamp(window_start).date()
    end_day = pd.Timestamp(end_exclusive).date()
    if end_day <= start_day:
        raise ValueError("end_exclusive must be later than window_start")
    expected_days = (end_day - start_day).days
    out: list[ArchiveSeriesCoverage] = []
    for lane, relative in archive_lanes:
        lane_root = root / relative
        if not lane_root.exists():
            continue
        series_dirs = sorted(path for path in lane_root.iterdir() if path.is_dir())
        if not series_dirs:
            series_dirs = [lane_root]
        for series_dir in series_dirs:
            files = 0
            days: set[date] = set()
            for path in series_dir.rglob("*"):
                if not path.is_file() or path.stat().st_size <= 0:
                    continue
                day = _date_from_path(path.relative_to(series_dir))
                if day is None or not (start_day <= day < end_day):
                    continue
                files += 1
                days.add(day)
            series_key = series_dir.name if series_dir != lane_root else lane_root.name
            dated_days = len(days)
            out.append(
                ArchiveSeriesCoverage(
                    lane=lane,
                    series_key=series_key,
                    root=str(series_dir),
                    files=files,
                    dated_days=dated_days,
                    start=min(days) if days else None,
                    end=max(days) if days else None,
                    expected_days=expected_days,
                    missing_days=max(0, expected_days - dated_days),
                    coverage_ratio=dated_days / expected_days if expected_days else 0.0,
                    window_start=start_day,
                    end_exclusive=end_day,
                )
            )
    return tuple(out)


def catalog_local_market_data(
    data_dir: str | Path,
    *,
    window_start: Any,
    end_exclusive: Any,
) -> LocalMarketCatalog:
    """Return one immutable, read-only catalog for all supported local stores."""

    sqlite_series, sqlite_metadata = discover_sqlite_market_series(
        data_dir, window_start=window_start, end_exclusive=end_exclusive
    )
    archives = discover_archive_market_series(
        data_dir, window_start=window_start, end_exclusive=end_exclusive
    )
    return LocalMarketCatalog(sqlite_series, sqlite_metadata, archives)
