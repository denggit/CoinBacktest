from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from human_replay_lab.data_service import ReplayDataService


def seed_soxl(data_dir: Path, start_et: str = "2025-01-02 04:00:00", end_et: str = "2025-01-02 16:01:00") -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    idx_et = pd.date_range(start_et, end_et, freq="1min")
    idx_utc = idx_et.tz_localize("America/New_York").tz_convert("UTC")
    n = len(idx_et)
    df = pd.DataFrame({
        "timestamp_utc": idx_utc.astype(str),
        "open": [100 + i * .01 for i in range(n)],
        "high": [100.2 + i * .01 for i in range(n)],
        "low": [99.8 + i * .01 for i in range(n)],
        "close": [100.1 + i * .01 for i in range(n)],
        "volume": [1000.0] * n,
        "trade_count": [10] * n,
        "vwap": [100.05 + i * .01 for i in range(n)],
    })
    with sqlite3.connect(data_dir / "alpaca_stock_history.db") as conn:
        df.to_sql("ALPACA_SOXL_1Min_sip_split", conn, if_exists="replace", index=False)


def test_specific_day_always_starts_0730_and_weekends_rejected(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    cursor = service.cursor_for_date("SOXL", "2025-01-02")
    assert str(cursor) == "2025-01-02 07:30:00"
    with pytest.raises(ValueError, match="weekend"):
        service.cursor_for_date("SOXL", "2025-01-04")


def test_high_timeframe_bar_hidden_until_available_time(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    before = service.candles("SOXL", "15m", "2025-01-02 10:14:00", 100)
    assert before.bars[-1]["time"] == "2025-01-02 09:45:00"
    at_close = service.candles("SOXL", "15m", "2025-01-02 10:15:00", 100)
    assert at_close.bars[-1]["time"] == "2025-01-02 10:00:00"
    assert at_close.bars[-1]["available_time"] == "2025-01-02 10:15:00"


def test_two_minute_incremental_updates_are_causal(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    updates = service.incremental_bars("SOXL", ["1m", "2m", "15m"], "2025-01-02 07:30:00", "2025-01-02 07:32:00")
    assert [b["time"] for b in updates["1m"]] == ["2025-01-02 07:30:00", "2025-01-02 07:31:00"]
    assert [b["time"] for b in updates["2m"]] == ["2025-01-02 07:30:00"]
    assert updates["15m"] == []


def test_execution_fill_uses_cursor_1m_open(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    price = service.execution_open("SOXL", "2025-01-02 07:30:00")
    # 04:00 -> 07:30 is 210 minutes.
    assert price == pytest.approx(102.10)


def test_repeated_playback_slices_memory_cache_instead_of_requerying(tmp_path) -> None:
    seed_soxl(tmp_path)

    class CountingService(ReplayDataService):
        def __init__(self, data_dir):
            self.load_calls = 0
            super().__init__(data_dir)
        def _load_1m(self, start_ny, end_ny):
            self.load_calls += 1
            return super()._load_1m(start_ny, end_ny)

    service = CountingService(tmp_path)
    service.prepare_episode("SOXL", "2025-01-02 07:30:00", ["30m", "15m", "2m", "1m"], 700)
    baseline = service.load_calls
    for minute in range(31, 41):
        cursor = f"2025-01-02 07:{minute:02d}:00"
        service.incremental_bars("SOXL", ["30m", "15m", "2m", "1m"], "2025-01-02 07:30:00", cursor)
    assert service.load_calls == baseline
