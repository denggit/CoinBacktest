from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "research"
    / "liquidity"
    / "panic_selloff_rejection_recovery_long"
    / "01_data_schema_and_coverage_audit.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_raw_zip(data_dir: Path) -> None:
    raw_dir = data_dir / "okx" / "raw" / "trades" / "ETH-USDT-SWAP"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_text = (
        "ts,trade_id,px,sz,side\n"
        "1672531200000,1,1200.0,2.0,sell\n"
        "1672531205000,2,1201.0,1.0,buy\n"
    )
    target = raw_dir / "ETH-USDT-SWAP-trades-2023-01-01.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ETH-USDT-SWAP-trades-2023-01-01.csv", csv_text)


def _write_trade_bars(data_dir: Path) -> None:
    loader = OKXTradeBarLoader(
        symbol="ETH-USDT-SWAP",
        timeframe="1m",
        data_dir=data_dir,
    )
    index = pd.DatetimeIndex(
        ["2023-01-01 08:00:00", "2023-01-01 08:01:00"],
        name="timestamp",
    )
    rows = pd.DataFrame(0.0, index=index, columns=loader.BASE_COLUMNS)
    rows.loc[:, ["open", "high", "low", "close"]] = [1200.0, 1202.0, 1198.0, 1201.0]
    rows["volume"] = 3.0
    rows["trades_count"] = 2.0
    rows["sell_volume"] = 2.0
    rows["buy_volume"] = 1.0
    rows["notional"] = 360.0
    rows["sell_notional"] = 240.0
    rows["buy_notional"] = 120.0
    rows["delta_volume"] = -1.0
    rows["delta_notional"] = -120.0
    rows["taker_buy_ratio"] = 1.0 / 3.0
    rows["avg_trade_size"] = 1.5
    rows["vwap"] = 1200.0
    rows["max_trade_notional"] = 240.0
    rows["max_trade_size"] = 2.0
    loader.save_local_data(rows)
    loader._mark_coverage(pd.Timestamp("2023-01-01").date(), rows=len(rows))


def _write_range_data(data_dir: Path) -> None:
    range_loader = OKXRangeBarLoader(
        symbol="ETH-USDT-SWAP",
        range_pct=0.002,
        data_dir=data_dir,
    )
    bar = {column: 0.0 for column in range_loader.BASE_COLUMNS}
    bar.update(
        {
            "bar_id": 20230101000001,
            "start_ts": "2023-01-01 08:00:00.000",
            "end_ts": "2023-01-01 08:00:05.000",
            "duration_seconds": 5.0,
            "open": 1200.0,
            "high": 1201.0,
            "low": 1197.0,
            "close": 1197.5,
            "range_pct": (1201.0 - 1197.0) / 1200.0,
            "range_size": 4.0,
            "direction": -1.0,
            "volume": 3.0,
            "notional": 360.0,
            "trades_count": 2.0,
            "buy_volume": 1.0,
            "sell_volume": 2.0,
            "buy_notional": 120.0,
            "sell_notional": 240.0,
            "delta_volume": -1.0,
            "delta_notional": -120.0,
            "taker_buy_ratio": 1.0 / 3.0,
            "vwap": 1200.0,
            "max_trade_notional": 240.0,
        }
    )
    range_loader._upsert_bars(pd.DataFrame([bar], columns=range_loader.BASE_COLUMNS))
    range_loader._mark_coverage(pd.Timestamp("2023-01-01").date(), rows=1)

    footprint_loader = OKXRangeFootprintLoader(
        symbol="ETH-USDT-SWAP",
        range_pct=0.002,
        price_step=1.0,
        data_dir=data_dir,
    )
    levels = []
    for price_bucket, sell_notional in ((1197.0, 200.0), (1198.0, 40.0)):
        level = {column: 0.0 for column in footprint_loader.BASE_COLUMNS}
        level.update(
            {
                "bar_id": bar["bar_id"],
                "start_ts": bar["start_ts"],
                "end_ts": bar["end_ts"],
                "price_bucket": price_bucket,
                "volume": 1.5,
                "notional": sell_notional,
                "trades_count": 1.0,
                "sell_volume": 1.5,
                "sell_notional": sell_notional,
                "sell_trades_count": 1.0,
                "delta_volume": -1.5,
                "delta_notional": -sell_notional,
                "max_trade_notional": sell_notional,
            }
        )
        levels.append(level)
    footprint_loader._upsert_footprints(
        pd.DataFrame(levels, columns=footprint_loader.BASE_COLUMNS)
    )
    footprint_loader._mark_coverage(
        pd.Timestamp("2023-01-01").date(),
        rows=len(levels),
        bars=1,
    )


def _write_ordinary_ohlcv(data_dir: Path) -> None:
    path = data_dir / "crypto_history.db"
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE ETH_USDT_SWAP_15m (
                timestamp TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL
            )
            """
        )
        conn.execute(
            "INSERT INTO ETH_USDT_SWAP_15m VALUES (?, ?, ?, ?, ?, ?)",
            ("2023-01-01 08:00:00", 1200.0, 1202.0, 1198.0, 1201.0, 100.0),
        )
        conn.commit()


def test_read_only_data_contract_audit_with_synthetic_cache(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "report"
    data_dir.mkdir(parents=True)
    _write_raw_zip(data_dir)
    _write_trade_bars(data_dir)
    _write_range_data(data_dir)
    _write_ordinary_ohlcv(data_dir)

    db_paths = sorted(data_dir.glob("*.db"))
    hashes_before = {path.name: _sha256(path) for path in db_paths}
    mtimes_before = {path.name: path.stat().st_mtime_ns for path in db_paths}

    command = [
        sys.executable,
        str(SCRIPT),
        "--symbol",
        "ETH-USDT-SWAP",
        "--start-date",
        "2023-01-01",
        "--end-date",
        "2023-01-01",
        "--data-dir",
        str(data_dir),
        "--out-dir",
        str(out_dir),
        "--trade-timeframes",
        "1m",
        "--range-pcts",
        "0.002",
        "--price-steps",
        "1.0",
        "--raw-schema-samples",
        "1",
        "--sample-rows",
        "1",
        "--alignment-sample-per-year",
        "10",
        "--no-review-pack",
        "--fail-on-critical",
    ]
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    required = {
        "event_counts.csv",
        "summary.csv",
        "yearly.csv",
        "monthly.csv",
        "events_or_trades.csv",
        "signal_audit.csv",
        "causal_audit.csv",
        "parameter_results.csv",
        "bad_trade_signatures.csv",
        "run_meta.json",
        "research_brief.md",
        "data_schema.csv",
        "raw_trade_files.csv",
        "timezone_audit.csv",
        "coverage_metadata.csv",
        "range_footprint_alignment.csv",
        "audit_findings.csv",
    }
    assert required.issubset({path.name for path in out_dir.iterdir()})

    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["stage"] == "data_contract"
    assert meta["status"] == "data_contract_ready_for_02_design"
    assert meta["critical_finding_count"] == 0

    summary = pd.read_csv(out_dir / "summary.csv")
    status_by_source = dict(zip(summary["source_id"], summary["status"], strict=True))
    assert status_by_source["raw_trades"] == "ok"
    assert status_by_source["trade_bar_1m"] == "ok"
    assert status_by_source["range_bar_r0020"] == "ok"
    assert status_by_source["range_footprint_r0020_step1"] == "ok"
    assert status_by_source["fixed_time_footprint"] == "unavailable_by_design"

    alignment = pd.read_csv(out_dir / "range_footprint_alignment.csv")
    assert len(alignment) == 1
    assert not alignment["missing_footprint_flag"].astype(bool).any()
    assert not alignment["start_ts_mismatch_flag"].astype(bool).any()
    assert not alignment["end_ts_mismatch_flag"].astype(bool).any()

    causal = pd.read_csv(out_dir / "causal_audit.csv")
    source_ids = set(causal["source_id"])
    assert "raw_trades" in source_ids
    assert "trade_bar_1m" in source_ids
    assert "range_bar_r0020" in source_ids
    assert "range_footprint_r0020_step1" in source_ids
    assert not causal["lookahead_flag"].astype(bool).any()

    assert hashes_before == {path.name: _sha256(path) for path in db_paths}
    assert mtimes_before == {path.name: path.stat().st_mtime_ns for path in db_paths}
