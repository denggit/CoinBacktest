from pathlib import Path

import pytest

from human_replay_lab.data_service import ReplayDataService
from human_replay_lab.server import ReplayApplication
from human_replay_lab.store import ReplayStore
from tests.replay_lab.test_data_service import seed_soxl


def test_manual_fvg_limit_order_rests_then_fills_on_next_closed_1m(tmp_path: Path) -> None:
    seed_soxl(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))
    episode = app.create_episode({"symbol": "SOXL-USDT-SWAP", "mode": "specific", "start_date": "2026-06-02"})

    # At 07:30 open is 102.10; 101.95 is a genuine resting buy limit. The
    # 07:30 bar low reaches 101.90, so it is only declared filled after the
    # replay advances to 07:31 and that 1m path is causally known.
    order = app.trade(episode["id"], {
        "side": "LONG",
        "timeframe": "1m",
        "order_type": "limit",
        "limit_price": 101.95,
        "entry_context": {
            "anchor_time": "2026-06-02 07:29:00",
            "anchor_timeframe": "1m",
            "snap_field": "L",
            "intent": "manual_limit_entry",
        },
    })
    assert order["status"] == "pending"
    assert order["fill_price"] is None
    assert len(order["active_limit_orders"]) == 1

    stepped = app.step(episode["id"], 1, ["1m", "2m", "15m", "30m"])
    types = [event["event_type"] for event in stepped["trade_events"]]
    assert types[:3] == ["LONG", "ORDER_FILLED", "TRADE_OPEN"]
    fill = stepped["trade_events"][0]
    assert fill["event_type"] == "LONG"
    assert fill["price"] == pytest.approx(101.95)
    assert fill["payload"]["order_type"] == "limit"
    assert fill["payload"]["entry_context"]["snap_field"] == "L"
    assert stepped["active_limit_orders"] == []


def test_limit_order_can_be_cancelled_before_fill(tmp_path: Path) -> None:
    seed_soxl(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))
    episode = app.create_episode({"symbol": "SOXL-USDT-SWAP", "mode": "specific", "start_date": "2026-06-02"})
    placed = app.trade(episode["id"], {
        "side": "LONG", "timeframe": "1m", "order_type": "limit", "limit_price": 90.0,
    })
    assert placed["status"] == "pending"
    cancelled = app.cancel_limit_order(episode["id"], {})
    assert cancelled["event"]["event_type"] == "LIMIT_CANCEL"
    assert cancelled["active_limit_orders"] == []


def test_limit_order_accepts_manual_entry_with_attached_sl_tp(tmp_path: Path) -> None:
    seed_soxl(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))
    episode = app.create_episode({"symbol": "SOXL-USDT-SWAP", "mode": "specific", "start_date": "2026-06-02"})

    order = app.trade(episode["id"], {
        "side": "LONG",
        "timeframe": "1m",
        "order_type": "limit",
        "limit_price": 101.95,
        "stop_loss": 101.40,
        "take_profit": 103.20,
        "entry_context": {"entry_price_source": "manual_input"},
    })
    assert order["status"] == "pending"
    pending = order["active_limit_orders"][0]
    assert pending["price"] == pytest.approx(101.95)
    assert pending["payload"]["stop_loss"] == pytest.approx(101.40)
    assert pending["payload"]["take_profit"] == pytest.approx(103.20)

    stepped = app.step(episode["id"], 1, ["1m"])
    types = [event["event_type"] for event in stepped["trade_events"]]
    assert types == ["LONG", "ORDER_FILLED", "TRADE_OPEN", "SL", "TP"]
    assert stepped["trade_events"][3]["price"] == pytest.approx(101.40)
    assert stepped["trade_events"][4]["price"] == pytest.approx(103.20)


def test_invalid_long_bracket_is_rejected(tmp_path: Path) -> None:
    seed_soxl(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))
    episode = app.create_episode({"symbol": "SOXL-USDT-SWAP", "mode": "specific", "start_date": "2026-06-02"})
    with pytest.raises(ValueError, match="stop_loss"):
        app.trade(episode["id"], {
            "side": "LONG", "timeframe": "1m", "order_type": "limit",
            "limit_price": 101.95, "stop_loss": 102.10, "take_profit": 103.0,
        })
