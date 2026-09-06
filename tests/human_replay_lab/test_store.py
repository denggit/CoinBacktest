from human_replay_lab.server import ReplayApplication
from human_replay_lab.store import ReplayStore


class _DisplayData:
    @staticmethod
    def beijing_display(value):
        return str(value)


def test_episode_and_event_round_trip(tmp_path) -> None:
    store = ReplayStore(tmp_path / "replay.sqlite3")
    episode = store.create_episode("SOXL", "2025-01-02 07:30:00")
    event = store.add_event(episode.id, "LIQUIDITY", episode.cursor_time, timeframe="30m", price=40.5,
                            payload={"kind": "BSL", "snap_field": "H"})
    assert event["payload"]["snap_field"] == "H"
    exported = store.export_episode(episode.id)
    assert exported["episode"]["symbol"] == "SOXL"
    assert len(exported["events"]) == 1


def test_cursor_update_and_close(tmp_path) -> None:
    store = ReplayStore(tmp_path / "replay.sqlite3")
    episode = store.create_episode("SOXL", "2025-01-02 07:30:00")
    assert store.update_cursor(episode.id, "2025-01-02 07:35:00").cursor_time == "2025-01-02 07:35:00"
    assert store.close_episode(episode.id).status == "closed"


def test_limit_modify_replaces_resting_order(tmp_path) -> None:
    store = ReplayStore(tmp_path / "replay.sqlite3")
    episode = store.create_episode("ETH-USDT-SWAP", "2026-01-02 07:30:00")
    store.add_event(episode.id, "LIMIT_ORDER", episode.cursor_time, price=100.0, payload={"order_id": "o1", "side": "LONG", "stop_loss": 95.0, "take_profit": 110.0})
    store.add_event(episode.id, "LIMIT_MODIFY", "2026-01-02 07:31:00", price=101.0, payload={"order_id": "o1", "side": "LONG", "stop_loss": 96.0, "take_profit": 112.0})
    assert store.active_limit_orders(episode.id)[0]["price"] == 101.0


def test_open_trade_update_only_records_levels_that_really_changed(tmp_path) -> None:
    store = ReplayStore(tmp_path / "replay.sqlite3")
    app = ReplayApplication(_DisplayData(), store)  # type: ignore[arg-type]
    episode = store.create_episode("ETH-USDT-SWAP", "2026-01-02 07:30:00")
    store.add_event(
        episode.id, "TRADE_OPEN", episode.cursor_time, timeframe="5m", price=100.0,
        payload={
            "trade_id": "trade-1", "side": "LONG", "entry_price": 100.0,
            "initial_stop_loss": 95.0, "initial_take_profit": 110.0,
        },
    )

    unchanged = app.update_order(
        episode.id, {"trade_id": "trade-1", "stop_loss": 95.0, "take_profit": 110.0},
    )
    assert unchanged["events"] == []

    stop_only = app.update_order(
        episode.id, {"trade_id": "trade-1", "stop_loss": 96.0, "take_profit": 110.0},
    )
    assert [event["event_type"] for event in stop_only["events"]] == ["SL"]
    assert stop_only["events"][0]["payload"]["previous_price"] == 95.0
    active = store.active_trades(episode.id)[0]["payload"]
    assert active["current_stop_loss"] == 96.0
    assert active.get("current_take_profit", active["initial_take_profit"]) == 110.0
    assert not [event for event in store.list_events(episode.id) if event["event_type"] == "TP"]
