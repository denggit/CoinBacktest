from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from src.data_feed.local_market_catalog import (
    ArchiveSeriesCoverage,
    LocalMarketCatalog,
    SQLiteSeriesCoverage,
    catalog_local_market_data,
    discover_archive_market_series,
    discover_sqlite_market_series,
)
from src.research_common.ict_mss2.source_readiness import (
    SourceReadinessConfig,
    assert_pre_embargo_catalog,
    build_mechanism_readiness_gate,
    r27_gate_decision,
)


def _write_price_db(path: Path, *, include_spot: bool = False) -> None:
    with sqlite3.connect(path) as conn:
        instruments = ["ETH_USDT_SWAP_1m", "BTC_USDT_SWAP_1m"]
        if include_spot:
            instruments.append("ETH_USDT_1m")
        for table in instruments:
            conn.execute(
                f'CREATE TABLE "{table}" ('
                "timestamp TEXT PRIMARY KEY, open REAL, high REAL, low REAL, close REAL, volume REAL)"
            )
            conn.executemany(
                f'INSERT INTO "{table}" VALUES (?, 1, 1, 1, 1, 1)',
                [
                    ("2022-01-01 00:00:00",),
                    ("2025-06-30 23:59:00",),
                    ("2025-07-01 00:00:00",),
                    ("2025-08-01 00:00:00",),
                ],
            )


def test_sqlite_catalog_physically_excludes_embargo_and_holdout(tmp_path: Path) -> None:
    _write_price_db(tmp_path / "crypto_history.db")

    series, metadata = discover_sqlite_market_series(
        tmp_path,
        window_start="2022-01-01",
        end_exclusive="2025-07-01",
        databases=["crypto_history.db"],
    )

    assert not metadata
    assert {row.table for row in series} == {"ETH_USDT_SWAP_1m", "BTC_USDT_SWAP_1m"}
    assert all(row.rows == 2 for row in series)
    assert all(row.end == pd.Timestamp("2025-06-30 23:59:00") for row in series)
    catalog = LocalMarketCatalog(series, (), ())
    checks = assert_pre_embargo_catalog(catalog)
    assert checks["passed"].all()


def test_archive_catalog_keeps_actual_post_only_series_as_empty_pre_embargo(tmp_path: Path) -> None:
    series_dir = tmp_path / "okx" / "raw" / "books" / "ETH-USDT-SWAP"
    series_dir.mkdir(parents=True)
    (series_dir / "ETH-USDT-SWAP_2025-07-02.csv").write_text("sealed", encoding="utf-8")

    rows = discover_archive_market_series(
        tmp_path,
        window_start="2025-06-29",
        end_exclusive="2025-07-01",
        archive_lanes=[("okx_raw_books", "okx/raw/books")],
    )

    assert len(rows) == 1
    assert rows[0].series_key == "ETH-USDT-SWAP"
    assert rows[0].files == 0
    assert rows[0].dated_days == 0
    assert rows[0].missing_days == 2
    assert rows[0].start is None


def _series(table: str) -> SQLiteSeriesCoverage:
    return SQLiteSeriesCoverage(
        database="crypto_history.db",
        table=table,
        series_key=table,
        timestamp_column="timestamp",
        dimensions="",
        rows=1_838_880,
        start=pd.Timestamp("2022-01-01"),
        end=pd.Timestamp("2025-06-30 23:59:00"),
        window_start=pd.Timestamp("2022-01-01"),
        end_exclusive=pd.Timestamp("2025-07-01"),
        columns="timestamp,open,high,low,close,volume",
    )


def _empty_archive(lane: str) -> ArchiveSeriesCoverage:
    return ArchiveSeriesCoverage(
        lane=lane,
        series_key="ETH-USDT-SWAP",
        root=lane,
        files=0,
        dated_days=0,
        start=None,
        end=None,
        expected_days=1277,
        missing_days=1277,
        coverage_ratio=0.0,
        window_start=pd.Timestamp("2022-01-01").date(),
        end_exclusive=pd.Timestamp("2025-07-01").date(),
    )


def test_mechanism_gate_requires_both_novelty_and_complete_source() -> None:
    base = (
        _series("ETH_USDT_SWAP_1m"),
        _series("BTC_USDT_SWAP_1m"),
    )
    archives = (
        _empty_archive("okx_raw_books"),
        _empty_archive("okx_liquidity_primitives"),
        _empty_archive("okx_liquidity_map"),
    )
    catalog = LocalMarketCatalog(base, (), archives)
    gate = build_mechanism_readiness_gate(catalog, config=SourceReadinessConfig())

    assert r27_gate_decision(gate) == "UNASSIGNED_NO_ELIGIBLE_MECHANISM"
    spot = gate.loc[gate["hypothesis"].eq("eth_spot_led_swap_convergence")].iloc[0]
    assert spot["source_gate"] == "MISSING"
    assert spot["mechanism_novelty"] == "NOVEL"
    btc = gate.loc[gate["hypothesis"].eq("btc_led_eth_repricing")].iloc[0]
    assert btc["source_gate"] == "READY"
    assert btc["mechanism_novelty"] == "FROZEN"

    with_spot = LocalMarketCatalog(
        base + (_series("ETH_USDT_1m"),), (), archives
    )
    eligible = build_mechanism_readiness_gate(with_spot)
    assert r27_gate_decision(eligible) == "R27_PRECOMMITMENT_ALLOWED"


def test_full_catalog_uses_supported_database_and_archive_discovery(tmp_path: Path) -> None:
    _write_price_db(tmp_path / "crypto_history.db")
    catalog = catalog_local_market_data(
        tmp_path, window_start="2022-01-01", end_exclusive="2025-07-01"
    )
    assert len(catalog.sqlite_series) == 2
    assert any(row.database == "okx_trade_bars.db" for row in catalog.sqlite_metadata)
