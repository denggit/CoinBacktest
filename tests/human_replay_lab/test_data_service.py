from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from human_replay_lab.data_service import ReplayDataService


def seed_soxl(
    data_dir: Path,
    start_et: str = "2026-06-02 04:00:00",
    end_et: str = "2026-06-02 16:01:00",
) -> None:
    """Seed the OKX loader table using its normal UTC+8 naive storage convention."""
    data_dir.mkdir(parents=True, exist_ok=True)
    idx_et = pd.date_range(start_et, end_et, freq="1min")
    idx_source = (
        idx_et.tz_localize("America/New_York")
        .tz_convert("Asia/Shanghai")
        .tz_localize(None)
    )
    n = len(idx_et)
    df = pd.DataFrame({
        "open": [100 + i * .01 for i in range(n)],
        "high": [100.2 + i * .01 for i in range(n)],
        "low": [99.8 + i * .01 for i in range(n)],
        "close": [100.1 + i * .01 for i in range(n)],
        "volume": [1000.0] * n,
    }, index=idx_source)
    df.index.name = "timestamp"
    with sqlite3.connect(data_dir / "crypto_history.db") as conn:
        df.to_sql("SOXL_USDT_SWAP_1m", conn, if_exists="replace", index=True)


def test_specific_day_always_starts_0730_and_weekends_rejected(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    cursor = service.cursor_for_date("SOXL-USDT-SWAP", "2026-06-02")
    assert str(cursor) == "2026-06-02 07:30:00"
    with pytest.raises(ValueError, match="weekend"):
        service.cursor_for_date("SOXL-USDT-SWAP", "2026-06-06")


def test_okx_source_timestamps_are_converted_to_new_york_wall_time(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    coverage = service.coverage()
    assert coverage["source"].startswith("OKX")
    assert coverage["available_start_et"] == "2026-06-02 04:00:00"
    assert coverage["available_end_et"] == "2026-06-02 16:01:00"
    assert coverage["first_episode_date"] == "2026-06-02"
    assert coverage["last_episode_date"] == "2026-06-02"


def test_high_timeframe_bar_hidden_until_available_time(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    before = service.candles("SOXL-USDT-SWAP", "15m", "2026-06-02 10:14:00", 100)
    assert before.bars[-1]["time"] == "2026-06-02 09:45:00"
    at_close = service.candles("SOXL-USDT-SWAP", "15m", "2026-06-02 10:15:00", 100)
    assert at_close.bars[-1]["time"] == "2026-06-02 10:00:00"
    assert at_close.bars[-1]["available_time"] == "2026-06-02 10:15:00"


def test_two_minute_incremental_updates_are_causal(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    updates = service.incremental_bars(
        "SOXL-USDT-SWAP",
        ["1m", "2m", "15m"],
        "2026-06-02 07:30:00",
        "2026-06-02 07:32:00",
    )
    assert [b["time"] for b in updates["1m"]] == ["2026-06-02 07:30:00", "2026-06-02 07:31:00"]
    assert [b["time"] for b in updates["2m"]] == ["2026-06-02 07:30:00"]
    assert updates["15m"] == []


def test_execution_fill_uses_cursor_1m_open(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    price = service.execution_open("SOXL-USDT-SWAP", "2026-06-02 07:30:00")
    # 04:00 -> 07:30 is 210 minutes.
    assert price == pytest.approx(102.10)


def test_repeated_playback_uses_memory_cache_instead_of_reloading_sqlite(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)

    calls = 0
    original = service.loader.load_local_data

    def counted_load():
        nonlocal calls
        calls += 1
        return original()

    service.loader.load_local_data = counted_load  # type: ignore[method-assign]
    service.prepare_episode("SOXL-USDT-SWAP", "2026-06-02 07:30:00", ["30m", "15m", "2m", "1m"], 700)
    baseline = calls
    for minute in range(31, 41):
        cursor = f"2026-06-02 07:{minute:02d}:00"
        service.incremental_bars(
            "SOXL-USDT-SWAP",
            ["30m", "15m", "2m", "1m"],
            "2026-06-02 07:30:00",
            cursor,
        )
    assert baseline == 1
    assert calls == baseline
