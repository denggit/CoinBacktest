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


def test_high_timeframe_forming_bar_is_visible_without_future_leak(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    before = service.candles("SOXL-USDT-SWAP", "15m", "2026-06-02 10:14:00", 100)
    forming = before.bars[-1]
    assert forming["time"] == "2026-06-02 10:00:00"
    assert forming["is_partial"] is True
    assert forming["is_closed"] is False
    assert forming["observed_through"] == "2026-06-02 10:14:00"
    assert forming["child_bars"] == 14  # 10:00..10:13 are closed.
    assert forming["includes_live_open"] is True
    assert forming["close"] == pytest.approx(103.74)  # 10:14 OPEN is visible; its future H/L/C are not.

    at_close = service.candles("SOXL-USDT-SWAP", "15m", "2026-06-02 10:15:00", 100)
    assert at_close.bars[-2]["time"] == "2026-06-02 10:00:00"
    assert at_close.bars[-2]["available_time"] == "2026-06-02 10:15:00"
    assert at_close.bars[-2]["is_partial"] is False
    assert at_close.bars[-2]["is_closed"] is True
    assert at_close.bars[-2]["close"] == pytest.approx(103.84)
    assert at_close.bars[-1]["time"] == "2026-06-02 10:15:00"
    assert at_close.bars[-1]["is_partial"] is True
    assert at_close.bars[-1]["child_bars"] == 0
    assert at_close.bars[-1]["includes_live_open"] is True


def test_incremental_updates_include_mutating_forming_htf_bar(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    one_minute = service.incremental_bars(
        "SOXL-USDT-SWAP",
        ["1m", "2m", "15m", "30m"],
        "2026-06-02 07:30:00",
        "2026-06-02 07:31:00",
    )
    assert [b["time"] for b in one_minute["1m"]] == ["2026-06-02 07:30:00"]
    assert one_minute["2m"][-1]["time"] == "2026-06-02 07:30:00"
    assert one_minute["2m"][-1]["is_partial"] is True
    assert one_minute["15m"][-1]["time"] == "2026-06-02 07:30:00"
    assert one_minute["15m"][-1]["is_partial"] is True
    assert one_minute["30m"][-1]["is_partial"] is True

    two_minutes = service.incremental_bars(
        "SOXL-USDT-SWAP",
        ["1m", "2m", "15m"],
        "2026-06-02 07:31:00",
        "2026-06-02 07:32:00",
    )
    assert [b["time"] for b in two_minutes["1m"]] == ["2026-06-02 07:31:00"]
    assert two_minutes["2m"][0]["time"] == "2026-06-02 07:30:00"
    assert two_minutes["2m"][0]["is_closed"] is True
    assert two_minutes["2m"][-1]["time"] == "2026-06-02 07:32:00"
    assert two_minutes["2m"][-1]["is_partial"] is True
    assert two_minutes["2m"][-1]["child_bars"] == 0
    assert two_minutes["15m"][-1]["time"] == "2026-06-02 07:30:00"
    assert two_minutes["15m"][-1]["is_partial"] is True
    assert two_minutes["15m"][-1]["child_bars"] == 2


def test_execution_fill_uses_cursor_1m_open(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    price = service.execution_open("SOXL-USDT-SWAP", "2026-06-02 07:30:00")
    # 04:00 -> 07:30 is 210 minutes.
    assert price == pytest.approx(102.10)


def test_repeated_playback_uses_episode_range_cache_instead_of_reloading_sqlite(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)

    calls = 0
    loader = service._loader("SOXL-USDT-SWAP")
    original = loader.load_local_data_range

    def counted_load(start_time=None, end_time=None):
        nonlocal calls
        calls += 1
        return original(start_time, end_time)

    loader.load_local_data_range = counted_load  # type: ignore[method-assign]
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
    assert baseline >= 1
    assert calls == baseline


def test_chart_context_keeps_off_hours_and_weekend_bars(tmp_path) -> None:
    # OKX context must be full-source context. Weekends are forbidden only as
    # Episode start dates; they are not removed from what the trader can see.
    seed_soxl(
        tmp_path,
        start_et="2026-06-05 15:00:00",  # Friday
        end_et="2026-06-08 07:31:00",    # Monday
    )
    service = ReplayDataService(tmp_path)
    window = service.candles("SOXL-USDT-SWAP", "15m", "2026-06-08 07:30:00", 700)
    times = {bar["time"] for bar in window.bars}
    assert "2026-06-06 12:00:00" in times  # Saturday context remains visible.
    assert "2026-06-07 23:45:00" in times  # Sunday overnight context remains visible.
    assert "2026-06-08 02:00:00" in times  # Weekday off-hours remain visible.
    coverage = service.coverage()
    assert coverage["chart_context_filters_session"] is False
    # Multi-symbol V1.8 no longer scans the entire 1m table just to compute
    # decorative row counts; the chart-window assertions above verify the
    # important behavior directly.
    assert coverage["weekend_rows_1m"] is None
    assert coverage["weekday_off_hours_rows_1m"] is None


def test_ui_time_fields_use_beijing_and_handle_dst(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    clock = service.clock_info("2026-06-02 07:30:00")
    assert clock["beijing_plain"] == "2026-06-02 19:30:00"
    assert clock["episode_start_bjt"] == "19:30"
    assert clock["market_open_bjt"] == "21:30"
    window = service.candles("SOXL-USDT-SWAP", "15m", "2026-06-02 10:14:00", 100)
    forming = window.bars[-1]
    assert forming["time_bjt"] == "2026-06-02 22:00:00"
    assert forming["observed_through_bjt"] == "2026-06-02 22:14:00"


def test_beijing_display_winter_offset_is_automatic(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    clock = service.clock_info("2026-12-02 07:30:00")
    assert clock["beijing_plain"] == "2026-12-02 20:30:00"
    assert clock["market_open_bjt"] == "22:30"
