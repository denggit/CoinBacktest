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


def test_60m_limit_fill_then_tp_stops_at_exact_causal_exit(tmp_path) -> None:
    app, episode_id = _app(tmp_path)
    pending = app.trade(episode_id, {
        "side": "LONG", "timeframe": "1m", "order_type": "limit",
        "limit_price": 99.50, "stop_loss": 98.50, "take_profit": 101.0,
    })
    assert pending["status"] == "pending"

    stepped = app.step(episode_id, 60, ["30m", "15m", "5m", "2m", "1m", "4H"])
    assert stepped["auto_finalized"] is True
    assert stepped["episode"]["status"] == "closed"
    # 00:10 bar fills and is observable at 00:11; same fill bar is excluded.
    # 00:20 TP bar is observable at 00:21. The remaining 39m must stay hidden.
    assert stepped["advanced_minutes"] == 21
    assert stepped["episode"]["cursor_time"] == "2026-06-21 00:21:00"
    types = [event["event_type"] for event in stepped["trade_events"]]
    assert "ORDER_FILLED" in types
    assert "TAKE_PROFIT_HIT" in types
    assert "TRADE_CLOSED" in types
    assert stepped["lifecycle_scan_passes"] <= 3
