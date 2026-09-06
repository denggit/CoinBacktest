from __future__ import annotations

import numpy as np
import pandas as pd

from human_replay_lab.data_service import ReplayDataService
from human_replay_lab.server import ReplayApplication
from human_replay_lab.store import ReplayStore
from src.data_feed.okx_loader import OKXDataLoader


def _seed_xau(tmp_path) -> None:
    # Project-local OKX timestamps are Beijing/source-wall time (normally UTC+8).
    idx = pd.date_range("2026-06-21 11:30:00", "2026-06-21 14:30:00", freq="1min")
    n = len(idx)
    open_ = np.full(n, 4600.0)
    high = np.full(n, 4600.5)
    low = np.full(n, 4599.5)
    close = np.full(n, 4600.0)
    # Beijing 12:01 == New York 00:01 EDT. TP is hit after entry.
    hit_i = int(np.where(idx == pd.Timestamp("2026-06-21 12:01:00"))[0][0])
    high[hit_i] = 4611.0
    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1.0},
        index=idx,
    )
    frame.index.name = "timestamp"
    OKXDataLoader(symbol="XAU-USDT-SWAP", timeframe="1m", db_dir=str(tmp_path)).save_local_data(frame)


def test_xau_is_discovered_and_uses_24x7_profile(tmp_path) -> None:
    _seed_xau(tmp_path)
    service = ReplayDataService(tmp_path)
    assert "XAU-USDT-SWAP" in service.available_symbols()
    assert service.is_24x7_symbol("XAU-USDT-SWAP") is True
    assert service.session_profile("XAU-USDT-SWAP") == "crypto_24x7_continuous_replay"
    assert service.auto_close_on_bracket_exit("XAU-USDT-SWAP") is False

    # Sunday is valid for OKX commodity-perp replay.
    cursor = service.cursor_for_start("XAU-USDT-SWAP", "2026-06-21T12:00")
    assert cursor == pd.Timestamp("2026-06-21 00:00:00")
    clock = service.clock_info(cursor, "XAU-USDT-SWAP")
    assert clock["market_phase"] == "24/7"
    assert clock["weekdays_only"] == "false"
    assert clock["episode_end_bjt"] == "MANUAL"


def test_xau_bracket_exit_keeps_episode_active_and_completes_step(tmp_path) -> None:
    _seed_xau(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))
    ep = app.create_episode({
        "symbol": "XAU-USDT-SWAP",
        "mode": "specific",
        "start_time": "2026-06-21T12:00",
    })
    result = app.trade(ep["id"], {
        "side": "LONG",
        "timeframe": "1m",
        "order_type": "market",
        "stop_loss": 4590.0,
        "take_profit": 4610.0,
    })
    assert result["status"] == "filled"

    stepped = app.step(ep["id"], 60, ["30m", "15m", "2m", "1m"])
    assert stepped["auto_finalized"] is False
    assert stepped["trade_closed"] is True
    assert stepped["episode_continues_after_trade"] is True
    assert stepped["episode"]["status"] == "active"
    assert stepped["advanced_minutes"] == 60
    assert stepped["episode"]["cursor_time"] == "2026-06-21 01:00:00"
    event_types = [event["event_type"] for event in stepped["trade_events"]]
    assert "TAKE_PROFIT_HIT" in event_types
    assert "TRADE_CLOSED" in event_types
    assert "EPISODE_SUMMARY" not in event_types
