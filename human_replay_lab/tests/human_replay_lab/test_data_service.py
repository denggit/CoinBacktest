from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from human_replay_lab.data_service import ReplayDataService
from human_replay_lab.server import ReplayApplication
from human_replay_lab.store import ReplayStore


def seed_soxl(data_dir: Path, start_et: str = "2025-01-02 04:00:00", end_et: str = "2025-01-02 16:01:00") -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    idx_et = pd.date_range(start_et, end_et, freq="1min")
    idx_source = idx_et.tz_localize("America/New_York").tz_convert("Asia/Shanghai").tz_localize(None)
    n = len(idx_et)
    df = pd.DataFrame({
        "timestamp": idx_source.astype(str),
        "open": [100 + i * .01 for i in range(n)],
        "high": [100.2 + i * .01 for i in range(n)],
        "low": [99.8 + i * .01 for i in range(n)],
        "close": [100.1 + i * .01 for i in range(n)],
        "volume": [1000.0] * n,
        "trade_count": [10] * n,
        "vwap": [100.05 + i * .01 for i in range(n)],
    })
    with sqlite3.connect(data_dir / "crypto_history.db") as conn:
        df.to_sql("SOXL_USDT_SWAP_1m", conn, if_exists="replace", index=False)


def test_specific_day_always_starts_0730_and_weekends_rejected(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    cursor = service.cursor_for_date("SOXL-USDT-SWAP", "2025-01-02")
    assert str(cursor) == "2025-01-02 07:30:00"
    with pytest.raises(ValueError, match="weekend"):
        service.cursor_for_date("SOXL-USDT-SWAP", "2025-01-04")


def test_high_timeframe_bar_is_causally_partial_until_available_time(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    before = service.candles("SOXL-USDT-SWAP", "15m", "2025-01-02 10:14:00", 100)
    assert before.bars[-1]["time"] == "2025-01-02 10:00:00"
    assert before.bars[-1]["is_partial"] is True
    assert before.bars[-1]["observed_through"] == "2025-01-02 10:14:00"
    at_close = service.candles("SOXL-USDT-SWAP", "15m", "2025-01-02 10:15:00", 100)
    closed = [bar for bar in at_close.bars if bar["time"] == "2025-01-02 10:00:00"][0]
    assert closed["is_partial"] is False
    assert closed["available_time"] == "2025-01-02 10:15:00"
    assert at_close.bars[-1]["time"] == "2025-01-02 10:15:00"
    assert at_close.bars[-1]["is_partial"] is True


def test_two_minute_incremental_updates_are_causal(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    updates = service.incremental_bars("SOXL-USDT-SWAP", ["1m", "2m", "15m"], "2025-01-02 07:30:00", "2025-01-02 07:32:00")
    assert [b["time"] for b in updates["1m"]] == ["2025-01-02 07:30:00", "2025-01-02 07:31:00"]
    assert [(b["time"], b["is_partial"]) for b in updates["2m"]] == [
        ("2025-01-02 07:30:00", False),
        ("2025-01-02 07:32:00", True),
    ]
    assert [(b["time"], b["is_partial"]) for b in updates["15m"]] == [("2025-01-02 07:30:00", True)]


def test_execution_fill_uses_cursor_1m_open(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    price = service.execution_open("SOXL-USDT-SWAP", "2025-01-02 07:30:00")
    # 04:00 -> 07:30 is 210 minutes.
    assert price == pytest.approx(102.10)


def test_repeated_playback_slices_memory_cache_instead_of_requerying(tmp_path) -> None:
    seed_soxl(tmp_path)

    class CountingService(ReplayDataService):
        def __init__(self, data_dir):
            self.load_calls = 0
            super().__init__(data_dir)
        def _load_1m(self, symbol, start_ny, end_ny):
            self.load_calls += 1
            return super()._load_1m(symbol, start_ny, end_ny)

    service = CountingService(tmp_path)
    service.prepare_episode("SOXL-USDT-SWAP", "2025-01-02 07:30:00", ["30m", "15m", "2m", "1m"], 700)
    baseline = service.load_calls
    for minute in range(31, 41):
        cursor = f"2025-01-02 07:{minute:02d}:00"
        service.incremental_bars("SOXL-USDT-SWAP", ["30m", "15m", "2m", "1m"], "2025-01-02 07:30:00", cursor)
    assert service.load_calls == baseline


def test_historical_candles_page_strictly_precedes_cursor_and_can_repeat(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    first = service.historical_candles("SOXL-USDT-SWAP", "5m", "2025-01-02 12:00:00", 30)
    assert len(first["bars"]) == 30
    assert all(bar["is_partial"] is False for bar in first["bars"])
    assert first["bars"][-1]["available_time"] <= "2025-01-02 12:00:00"
    second = service.historical_candles("SOXL-USDT-SWAP", "5m", first["bars"][0]["time"], 30)
    assert second["bars"]
    assert second["bars"][-1]["time"] < first["bars"][0]["time"]
    assert not ({bar["time"] for bar in first["bars"]} & {bar["time"] for bar in second["bars"]})


def test_pending_limit_lifecycle_cancels_only_when_tp_is_confirmed_first(tmp_path) -> None:
    seed_soxl(tmp_path)
    service = ReplayDataService(tmp_path)
    cancelled = service.limit_order_lifecycle(
        "SOXL-USDT-SWAP", "LONG", 101.5, 102.25,
        "2025-01-02 07:30:00", "2025-01-02 07:31:00",
    )
    assert cancelled["outcome"] == "cancel"
    assert cancelled["cancel_reason"] == "take_profit_before_entry"
    short_cancelled = service.limit_order_lifecycle(
        "SOXL-USDT-SWAP", "SHORT", 102.5, 101.95,
        "2025-01-02 07:30:00", "2025-01-02 07:31:00",
    )
    assert short_cancelled["outcome"] == "cancel"
    assert short_cancelled["cancel_reason"] == "take_profit_before_entry"
    ambiguous = service.limit_order_lifecycle(
        "SOXL-USDT-SWAP", "LONG", 102.0, 102.25,
        "2025-01-02 07:30:00", "2025-01-02 07:31:00",
    )
    assert ambiguous["outcome"] == "fill"
    assert ambiguous["sequence_resolution"] == "same_1m_entry_tp_ambiguous_entry_priority"


def test_replay_step_records_tp_before_entry_auto_cancel(tmp_path) -> None:
    seed_soxl(tmp_path)
    store = ReplayStore(tmp_path / "replay.sqlite3")
    app = ReplayApplication(ReplayDataService(tmp_path), store)
    episode = app.create_episode({"symbol": "SOXL-USDT-SWAP", "mode": "specific", "start_date": "2025-01-02"})
    placed = app.trade(episode["id"], {
        "side": "LONG", "timeframe": "1m", "order_type": "limit",
        "limit_price": 101.5, "stop_loss": 101.0, "take_profit": 102.25,
        "account_size": 10_000, "risk_pct": 1,
    })
    assert placed["status"] == "pending"
    result = app.step(episode["id"], 30, ["30m"])
    cancelled = [event for event in result["trade_events"] if event["event_type"] == "LIMIT_CANCEL"]
    assert len(cancelled) == 1
    assert cancelled[0]["payload"]["reason"] == "take_profit_before_entry"
    assert cancelled[0]["payload"]["result"] == "MISSED_TRADE"
    assert result["active_limit_orders"] == []
    assert result["active_trades"] == []
    assert result["trade_summary"]["invalidated_orders"] == 1
    assert result["advanced_minutes"] == 30
    assert result["requested_bar_minutes"] == 30
    assert result["lifecycle_resolution"] == "cached_1m_sequence"
