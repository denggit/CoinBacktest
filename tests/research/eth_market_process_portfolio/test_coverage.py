from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from research.eth_market_process_portfolio.common.coverage import (
    DatasetRequirement,
    audit_local_coverage,
    coverage_frame,
    overall_gate,
    inspect_local_books,
)


def _make_db(path: Path, table: str, rows: list[tuple[str, float]]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute(f'CREATE TABLE "{table}" (timestamp TEXT NOT NULL, value REAL NOT NULL)')
        conn.executemany(f'INSERT INTO "{table}" VALUES (?, ?)', rows)


def test_ready_requirement_uses_aggregate_coverage(tmp_path: Path) -> None:
    _make_db(
        tmp_path / "sample.db",
        "ETH_USDT_SWAP_1m",
        [("2022-01-01 00:00:00", 1.0), ("2026-06-30 23:59:59", 2.0)],
    )
    requirement = DatasetRequirement(
        module="baseline",
        dataset="ohlcv",
        database="sample.db",
        table_pattern=r"^ETH_USDT_SWAP_1m$",
        required_for="test",
        minimum_start=pd.Timestamp("2022-01-01"),
        minimum_end=pd.Timestamp("2026-06-30 23:59:59"),
    )

    records = audit_local_coverage(data_dir=tmp_path, requirements=[requirement])

    assert len(records) == 1
    assert records[0].status == "READY"
    assert records[0].rows == 2
    assert overall_gate(records) == "READY"


def test_missing_mandatory_database_blocks(tmp_path: Path) -> None:
    requirement = DatasetRequirement(
        module="order_flow",
        dataset="trade_bars",
        database="missing.db",
        table_pattern=r".*",
        required_for="test",
        minimum_start=pd.Timestamp("2022-01-01"),
        minimum_end=pd.Timestamp("2026-06-30"),
    )

    records = audit_local_coverage(data_dir=tmp_path, requirements=[requirement])

    assert records[0].status == "BLOCKED"
    assert overall_gate(records) == "BLOCKED"


def test_optional_short_history_is_window_only(tmp_path: Path) -> None:
    _make_db(
        tmp_path / "short.db",
        "open_interest",
        [("2026-06-01 00:00:00", 1.0), ("2026-06-30 00:00:00", 2.0)],
    )
    requirement = DatasetRequirement(
        module="positioning",
        dataset="open_interest",
        database="short.db",
        table_pattern=r"^open_interest$",
        required_for="window",
        minimum_start=None,
        minimum_end=None,
        optional=True,
    )

    records = audit_local_coverage(data_dir=tmp_path, requirements=[requirement])
    frame = coverage_frame(records)

    assert records[0].status == "WINDOW_ONLY"
    assert overall_gate(records) == "READY"
    assert frame.loc[0, "dataset"] == "open_interest"


def test_minute_bar_end_tolerance_avoids_false_partial(tmp_path: Path) -> None:
    _make_db(
        tmp_path / "minute.db",
        "ETH_USDT_SWAP_trade_bars_1m",
        [("2021-12-31 00:00:00", 1.0), ("2026-06-30 23:59:00", 2.0)],
    )
    requirement = DatasetRequirement(
        module="order_flow",
        dataset="trade_bars",
        database="minute.db",
        table_pattern=r".*trade_bars_1m$",
        required_for="test",
        minimum_start=pd.Timestamp("2022-01-01"),
        minimum_end=pd.Timestamp("2026-06-30 23:59:59"),
    )

    records = audit_local_coverage(data_dir=tmp_path, requirements=[requirement])

    matching = [record for record in records if record.dataset == "trade_bars"]
    assert matching[0].status == "READY"


def test_ready_alternative_table_wins_over_partial_legacy_table(tmp_path: Path) -> None:
    with sqlite3.connect(tmp_path / "multi.db") as conn:
        conn.execute('CREATE TABLE "ETH_USDT_SWAP_trade_bars_1m_preferred" (timestamp TEXT, value REAL)')
        conn.execute('CREATE TABLE "ETH_USDT_SWAP_trade_bars_1m_legacy" (timestamp TEXT, value REAL)')
        conn.executemany(
            'INSERT INTO "ETH_USDT_SWAP_trade_bars_1m_preferred" VALUES (?, ?)',
            [("2022-01-01 00:00:00", 1.0), ("2026-06-30 23:59:00", 2.0)],
        )
        conn.executemany(
            'INSERT INTO "ETH_USDT_SWAP_trade_bars_1m_legacy" VALUES (?, ?)',
            [("2023-01-01 00:00:00", 1.0), ("2026-06-15 00:00:00", 2.0)],
        )
    requirement = DatasetRequirement(
        module="order_flow",
        dataset="trade_bars",
        database="multi.db",
        table_pattern=r".*trade_bars_1m.*",
        required_for="test",
        minimum_start=pd.Timestamp("2022-01-01"),
        minimum_end=pd.Timestamp("2026-06-30 23:59:59"),
    )

    records = audit_local_coverage(data_dir=tmp_path, requirements=[requirement])

    assert {record.status for record in records if record.dataset == "trade_bars"} == {"READY", "PARTIAL"}
    assert overall_gate(records) == "READY"


def test_raw_books_directory_is_reported_as_window_only(tmp_path: Path) -> None:
    raw = tmp_path / "okx" / "raw" / "books" / "ETH-USDT-SWAP"
    raw.mkdir(parents=True)
    (raw / "ETH-USDT-SWAP-L2orderbook-400lv-2025-10-01.tar.gz").write_bytes(b"x")
    (raw / "ETH-USDT-SWAP-L2orderbook-400lv-2026-06-30.tar.gz").write_bytes(b"x")

    books = inspect_local_books(tmp_path, symbol="ETH-USDT-SWAP")

    assert books.status == "WINDOW_ONLY"
    assert books.start == pd.Timestamp("2025-10-01")
    assert books.end == pd.Timestamp("2026-06-30")
