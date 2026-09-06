from pathlib import Path

import pytest

from human_replay_lab.data_service import ReplayDataService
from human_replay_lab.server import ReplayApplication
from human_replay_lab.store import ReplayStore
from tests.replay_lab.test_data_service import seed_soxl


def build_app(tmp_path: Path) -> tuple[ReplayApplication, str]:
    seed_soxl(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))
    episode = app.create_episode({"symbol": "SOXL-USDT-SWAP", "mode": "specific", "start_date": "2026-06-02"})
    return app, episode["id"]


def test_market_long_tp_hit_records_full_trade_outcome(tmp_path: Path) -> None:
    app, episode_id = build_app(tmp_path)
    opened = app.trade(episode_id, {
        "side": "LONG", "timeframe": "1m", "order_type": "market",
        "stop_loss": 101.50, "take_profit": 102.25,
    })
    assert opened["status"] == "filled"
    assert {event["event_type"] for event in opened["events"]} >= {"LONG", "ORDER_FILLED", "TRADE_OPEN", "SL", "TP"}

    stepped = app.step(episode_id, 1, ["1m"])
    types = [event["event_type"] for event in stepped["trade_events"]]
    assert "TAKE_PROFIT_HIT" in types
    assert "TRADE_CLOSED" in types
    closed = next(event for event in stepped["trade_events"] if event["event_type"] == "TRADE_CLOSED")
    assert closed["payload"]["exit_reason"] == "TAKE_PROFIT"
    assert closed["price"] == pytest.approx(102.25)
    assert closed["payload"]["gross_return_pct"] > 0
    assert closed["payload"]["net_return_pct"] > 0
    assert closed["payload"]["r_multiple"] == pytest.approx(
        closed["payload"]["net_pnl"] / closed["payload"]["planned_risk_amount"]
    )
    assert stepped["trade_summary"]["closed_trades"] == 1
    assert stepped["trade_summary"]["wins"] == 1
    assert stepped["trade_summary"]["active_trades"] == 0


def test_limit_fill_does_not_use_same_intrabar_path_for_tp(tmp_path: Path) -> None:
    app, episode_id = build_app(tmp_path)
    app.trade(episode_id, {
        "side": "LONG", "timeframe": "1m", "order_type": "limit",
        "limit_price": 101.95, "stop_loss": 101.40, "take_profit": 102.25,
    })
    first = app.step(episode_id, 1, ["1m"])
    first_types = [event["event_type"] for event in first["trade_events"]]
    assert "ORDER_FILLED" in first_types
    assert "TRADE_CLOSED" not in first_types
    assert first["trade_summary"]["active_trades"] == 1

    second = app.step(episode_id, 1, ["1m"])
    second_types = [event["event_type"] for event in second["trade_events"]]
    assert "TAKE_PROFIT_HIT" in second_types
    closed = next(event for event in second["trade_events"] if event["event_type"] == "TRADE_CLOSED")
    assert closed["payload"]["entry_bar_policy"] == "exclude_intrabar_limit_fill_bar"
    assert closed["payload"]["trigger_bar_time"] == "2026-06-02 07:31:00"


def test_same_1m_bar_sl_and_tp_is_flagged_and_resolved_conservatively(tmp_path: Path) -> None:
    app, episode_id = build_app(tmp_path)
    app.trade(episode_id, {
        "side": "LONG", "timeframe": "1m", "order_type": "market",
        "stop_loss": 101.95, "take_profit": 102.25,
    })
    stepped = app.step(episode_id, 1, ["1m"])
    types = [event["event_type"] for event in stepped["trade_events"]]
    assert "TRADE_EXIT_AMBIGUOUS" in types
    closed = next(event for event in stepped["trade_events"] if event["event_type"] == "TRADE_CLOSED")
    assert closed["payload"]["exit_reason"] == "AMBIGUOUS_BOTH_HIT"
    assert closed["payload"]["resolution"] == "conservative_stop_assumption"
    assert closed["payload"]["raw_exit_price"] == pytest.approx(101.95)
    assert closed["price"] == pytest.approx(101.95 * (1 - 0.0002))
    assert closed["payload"]["risk_overrun_amount"] > 0
    assert stepped["trade_summary"]["ambiguous"] == 1


def test_legacy_v16_entry_is_caught_up_without_deleting_old_episode(tmp_path: Path) -> None:
    app, episode_id = build_app(tmp_path)
    app.store.add_event(
        episode_id, "LONG", "2026-06-02 07:30:00", timeframe="1m", price=102.10,
        payload={"order_type": "market", "stop_loss": 101.50, "take_profit": 102.25},
    )
    app.store.update_cursor(episode_id, "2026-06-02 07:31:00")
    snap = app.snapshots(episode_id, ["1m"], 100)
    types = [event["event_type"] for event in snap["events"]]
    assert "TAKE_PROFIT_HIT" in types
    assert "TRADE_CLOSED" in types
    assert snap["trade_summary"]["closed_trades"] == 1


def test_events_autosave_before_end_episode_and_end_only_finalizes(tmp_path: Path) -> None:
    app, episode_id = build_app(tmp_path)
    app.add_event(episode_id, {
        "event_type": "LIQUIDITY", "timeframe": "15m", "price": 105.0,
        "payload": {"kind": "BSL", "importance": "normal"},
    })

    # Reopen the same SQLite file before pressing End Episode: the event is
    # already durable, proving End Episode is not the save operation.
    reopened = ReplayStore(tmp_path / "replay.sqlite3")
    assert any(event["event_type"] == "LIQUIDITY" for event in reopened.list_events(episode_id))
    assert reopened.get_episode(episode_id).status == "active"

    closed = app.close_episode(episode_id)
    assert closed["episode"]["status"] == "closed"
    assert closed["summary"]["closed_trades"] == 0
    events = reopened.list_events(episode_id)
    assert any(event["event_type"] == "EPISODE_SUMMARY" for event in events)
    assert reopened.get_episode(episode_id).status == "closed"


def test_end_episode_expires_unfilled_limit_instead_of_reporting_pending(tmp_path: Path) -> None:
    app, episode_id = build_app(tmp_path)
    pending = app.trade(episode_id, {
        "side": "LONG", "timeframe": "2m", "order_type": "limit",
        "limit_price": 90.0, "stop_loss": 80.0, "take_profit": 120.0,
    })
    assert pending["status"] == "pending"
    assert app.store.trade_summary(episode_id)["pending_orders"] == 1

    closed = app.close_episode(episode_id)
    assert closed["summary"]["pending_orders"] == 0
    assert closed["summary"]["unfilled_orders"] == 1
    types = [event["event_type"] for event in app.store.list_events(episode_id)]
    assert "LIMIT_EXPIRED" in types
    assert app.store.active_limit_orders(episode_id) == []


def test_legacy_closed_episode_unmatched_limit_is_not_pending(tmp_path: Path) -> None:
    app, episode_id = build_app(tmp_path)
    app.trade(episode_id, {
        "side": "LONG", "timeframe": "2m", "order_type": "limit",
        "limit_price": 90.0, "stop_loss": 80.0, "take_profit": 120.0,
    })
    # Simulate a V1.7.0 closed Episode whose old finalizer forgot to expire it.
    app.store.close_episode(episode_id)
    summary = app.store.trade_summary(episode_id)
    assert summary["pending_orders"] == 0
    assert summary["unfilled_orders"] == 1
    assert app.store.active_limit_orders(episode_id) == []
