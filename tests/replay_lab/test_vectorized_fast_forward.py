from __future__ import annotations

import numpy as np
import pandas as pd

from human_replay_lab.data_service import ReplayDataService
from human_replay_lab.server import ReplayApplication
from human_replay_lab.store import ReplayStore
from src.data_feed.okx_loader import OKXDataLoader


def _seed_eth(tmp_path) -> None:
    idx = pd.date_range("2026-06-21 11:30:00", "2026-06-21 14:30:00", freq="1min")
    n = len(idx)
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.20)
    low = np.full(n, 99.80)
    close = np.full(n, 100.0)
    # Beijing 12:10 / NY 00:10 fills the resting long limit at 99.50.
    fill_i = int(np.where(idx == pd.Timestamp("2026-06-21 12:10:00"))[0][0])
    low[fill_i] = 99.40
    # The fill is known at NY 00:11. TP is hit on the later 00:20 bar,
    # available to the replay cursor at 00:21.
    tp_i = int(np.where(idx == pd.Timestamp("2026-06-21 12:20:00"))[0][0])
    high[tp_i] = 101.20
    frame = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)
    frame.index.name = "timestamp"
    OKXDataLoader(symbol="ETH-USDT-SWAP", timeframe="1m", db_dir=str(tmp_path)).save_local_data(frame)


def _app(tmp_path) -> tuple[ReplayApplication, str]:
    _seed_eth(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))
    ep = app.create_episode({"symbol": "ETH-USDT-SWAP", "mode": "specific", "start_time": "2026-06-21T12:00"})
    return app, ep["id"]


def test_60m_without_lifecycle_event_is_one_vectorized_scan(tmp_path) -> None:
    app, episode_id = _app(tmp_path)
    stepped = app.step(episode_id, 60, ["30m", "15m", "5m", "2m", "1m", "4H"])
    assert stepped["advanced_minutes"] == 60
    assert stepped["step_engine"] == "vectorized_event_driven"
    assert stepped["lifecycle_scan_passes"] == 1
    assert stepped["trade_events"] == []


def test_60m_limit_fill_then_tp_records_exit_and_continues_to_step_end(tmp_path) -> None:
    app, episode_id = _app(tmp_path)
    pending = app.trade(episode_id, {
        "side": "LONG", "timeframe": "1m", "order_type": "limit",
        "limit_price": 99.50, "stop_loss": 98.50, "take_profit": 101.0,
    })
    assert pending["status"] == "pending"

    stepped = app.step(episode_id, 60, ["30m", "15m", "5m", "2m", "1m", "4H"])
    assert stepped["auto_finalized"] is False
    assert stepped["trade_closed"] is True
    assert stepped["episode_continues_after_trade"] is True
    assert stepped["episode"]["status"] == "active"
    # 00:10 bar fills and is observable at 00:11; same fill bar is excluded.
    # 00:20 TP bar is observable at 00:21; the trade closes there while the
    # requested Replay step continues causally through 01:00.
    assert stepped["advanced_minutes"] == 60
    assert stepped["episode"]["cursor_time"] == "2026-06-21 01:00:00"
    types = [event["event_type"] for event in stepped["trade_events"]]
    assert "ORDER_FILLED" in types
    assert "TAKE_PROFIT_HIT" in types
    assert "TRADE_CLOSED" in types
    assert "EPISODE_SUMMARY" not in types
    assert stepped["lifecycle_scan_passes"] <= 3


def test_event_pause_stops_at_fill_then_exit_without_exposing_future(tmp_path) -> None:
    app, episode_id = _app(tmp_path)
    setup = {"breakoutLiquidity": "swept", "setupNotes": "Target opposite liquidity"}
    app.trade(episode_id, {
        "side": "LONG", "timeframe": "1m", "order_type": "limit",
        "limit_price": 99.50, "stop_loss": 98.50, "take_profit": 101.0,
        "entry_context": {"setup": setup},
    })
    filled = app.step(episode_id, 60, ["1m", "30m"], pause_on_event=True)
    assert filled["episode"]["cursor_time"] == "2026-06-21 00:11:00"
    assert filled["paused_on_event"] is True
    assert filled["at_data_end"] is False
    assert filled["advanced_minutes"] == 11
    assert not filled["trade_closed"]
    opened = next(e for e in filled["trade_events"] if e["event_type"] == "TRADE_OPEN")
    assert opened["payload"]["entry_context"]["setup"] == setup
    assert all(bar["time"] <= filled["episode"]["cursor_time"] for bar in filled["updates"]["1m"])
    assert max(bar["high"] for bar in filled["updates"]["30m"]) < 101.0
    exited = app.step(episode_id, 60, ["1m"], pause_on_event=True)
    assert exited["episode"]["cursor_time"] == "2026-06-21 00:21:00"
    assert exited["trade_closed"] and exited["paused_on_event"]
    assert not exited["at_data_end"]
    assert exited["episode"]["status"] == "active"


def test_pause_mode_without_events_and_data_boundary(tmp_path) -> None:
    app, episode_id = _app(tmp_path)
    stepped = app.step(episode_id, 60, ["1m"], pause_on_event=True)
    assert stepped["advanced_minutes"] == 60
    assert not stepped["paused_on_event"]
    assert not stepped["at_data_end"]
    ended = app.step(episode_id, 1440, ["1m"], pause_on_event=True)
    assert ended["at_data_end"]
    assert not ended["paused_on_event"]
    again = app.step(episode_id, 1, ["1m"], pause_on_event=True)
    assert again["advanced_minutes"] == 0
    assert again["at_data_end"]


def test_market_order_keeps_pre_entry_thesis(tmp_path) -> None:
    app, episode_id = _app(tmp_path)
    setup = {"setupNotes": "Liquidity block invalidates the thesis"}
    result = app.trade(episode_id, {
        "side": "LONG", "timeframe": "1m", "order_type": "market",
        "stop_loss": 98.50, "take_profit": 101.0,
        "entry_context": {"setup": setup},
    })
    opened = next(e for e in result["events"] if e["event_type"] == "TRADE_OPEN")
    assert opened["payload"]["entry_context"]["setup"] == setup
    saved = app.store.list_events(episode_id)
    assert next(e for e in saved if e["event_type"] == "TRADE_OPEN")["payload"]["entry_context"]["setup"] == setup


def test_history_request_cannot_read_beyond_episode_cursor(tmp_path) -> None:
    app, episode_id = _app(tmp_path)
    history = app.history(episode_id, "1m", "2026-06-21 02:00:00", 100)
    assert history["before"] == "2026-06-21 00:00:00"
    assert history["bars"]
    assert all(bar["time"] < history["before"] for bar in history["bars"])
    assert max(bar["high"] for bar in history["bars"]) < 101.0
