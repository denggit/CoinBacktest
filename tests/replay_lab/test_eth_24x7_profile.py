from __future__ import annotations

import numpy as np
import pandas as pd

from human_replay_lab.data_service import ReplayDataService
from human_replay_lab.server import ReplayApplication
from human_replay_lab.store import ReplayStore
from src.data_feed.okx_loader import OKXDataLoader


def _seed_eth(tmp_path) -> None:
    # OKX local timestamps are project source-wall time (normally UTC+8 / Beijing).
    idx = pd.date_range("2026-06-21 11:30:00", "2026-06-21 14:30:00", freq="1min")
    n = len(idx)
    open_ = np.full(n, 100.0)
    high = np.full(n, 100.25)
    low = np.full(n, 99.75)
    close = np.full(n, 100.0)
    # Beijing 12:01 == New York 00:01 EDT. TP should be hit one minute after entry.
    hit_i = int(np.where(idx == pd.Timestamp("2026-06-21 12:01:00"))[0][0])
    high[hit_i] = 101.25
    frame = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": 1.0}, index=idx)
    frame.index.name = "timestamp"
    OKXDataLoader(symbol="ETH-USDT-SWAP", timeframe="1m", db_dir=str(tmp_path)).save_local_data(frame)


def test_eth_specific_start_is_beijing_24x7_and_weekend_allowed(tmp_path) -> None:
    _seed_eth(tmp_path)
    service = ReplayDataService(tmp_path)
    # 2026-06-21 is Sunday. Beijing 12:00 converts to New York 00:00 EDT.
    cursor = service.cursor_for_start("ETH-USDT-SWAP", "2026-06-21T12:00")
    assert cursor == pd.Timestamp("2026-06-21 00:00:00")
    assert cursor.dayofweek == 6
    assert service.validate_cursor("ETH-USDT-SWAP", cursor) == cursor
    clock = service.clock_info(cursor, "ETH-USDT-SWAP")
    assert clock["market_phase"] == "24/7"
    assert clock["weekdays_only"] == "false"
    assert clock["episode_end_bjt"] == "TP/SL"


def test_eth_tp_auto_finalizes_episode_at_hit_minute_even_with_60m_step(tmp_path) -> None:
    _seed_eth(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))
    ep = app.create_episode({
        "symbol": "ETH-USDT-SWAP",
        "mode": "specific",
        "start_time": "2026-06-21T12:00",
    })
    assert ep["status"] == "active"
    result = app.trade(ep["id"], {
        "side": "LONG",
        "timeframe": "1m",
        "order_type": "market",
        "stop_loss": 99.0,
        "take_profit": 101.0,
    })
    assert result["status"] == "filled"

    stepped = app.step(ep["id"], 60, ["30m", "15m", "2m", "1m"])
    assert stepped["auto_finalized"] is True
    assert stepped["episode"]["status"] == "closed"
    # Entry at 00:00. 00:01 bar closes at 00:02 and hits TP. Do not reveal the remaining 58 minutes.
    assert stepped["advanced_minutes"] == 2
    assert stepped["episode"]["cursor_time"] == "2026-06-21 00:02:00"
    types = [e["event_type"] for e in stepped["trade_events"]]
    assert "TAKE_PROFIT_HIT" in types
    assert "TRADE_CLOSED" in types
    assert "EPISODE_SUMMARY" in types
    assert "CLOSE" in types
    assert stepped["trade_summary"]["closed_trades"] == 1
    assert stepped["trade_summary"]["wins"] == 1


def test_soxl_profile_remains_weekday_session(tmp_path) -> None:
    # Profile dispatch itself must remain unchanged for SOXL.
    _seed_eth(tmp_path)
    service = ReplayDataService(tmp_path)
    assert service.session_profile("SOXL-USDT-SWAP") == "weekday_0730_1600_et"
    assert service.auto_close_on_bracket_exit("SOXL-USDT-SWAP") is False
