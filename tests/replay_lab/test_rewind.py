from pathlib import Path

from human_replay_lab.data_service import ReplayDataService
from human_replay_lab.server import ReplayApplication
from human_replay_lab.store import ReplayStore
from tests.replay_lab.test_data_service import seed_soxl


def test_rewind_archives_future_branch_and_restores_pending_order(tmp_path: Path) -> None:
    seed_soxl(tmp_path)
    store = ReplayStore(tmp_path / "replay.sqlite3")
    app = ReplayApplication(ReplayDataService(tmp_path), store)
    episode = app.create_episode({"symbol": "SOXL-USDT-SWAP", "mode": "specific", "start_date": "2026-06-02"})

    order = app.trade(episode["id"], {
        "side": "LONG", "timeframe": "1m", "order_type": "limit",
        "limit_price": 101.95, "stop_loss": 101.4, "take_profit": 103.2,
    })
    assert order["status"] == "pending"
    stepped = app.step(episode["id"], 2, ["30m", "15m", "2m", "1m"])
    assert stepped["episode"]["cursor_time"] == "2026-06-02 07:32:00"
    assert any(e["event_type"] == "LONG" for e in store.list_events(episode["id"]))

    rewound = app.rewind(episode["id"], 2, ["30m", "15m", "2m", "1m"])
    assert rewound["episode"]["cursor_time"] == "2026-06-02 07:30:00"
    assert rewound["discarded_event_count"] >= 3  # fill + SL + TP
    assert not any(e["event_type"] == "LONG" for e in rewound["events"])
    assert len(rewound["active_limit_orders"]) == 1
    assert rewound["events"][-1]["event_type"] == "REWIND"
    assert store.list_discarded_events(episode["id"])

    replayed = app.step(episode["id"], 1, ["1m"])
    assert [e["event_type"] for e in replayed["trade_events"]] == ["LONG", "ORDER_FILLED", "TRADE_OPEN", "SL", "TP"]


def test_rewind_never_moves_before_episode_start(tmp_path: Path) -> None:
    seed_soxl(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))
    episode = app.create_episode({"symbol": "SOXL-USDT-SWAP", "mode": "specific", "start_date": "2026-06-02"})
    app.step(episode["id"], 5, ["1m"])
    rewound = app.rewind(episode["id"], 15, ["1m"])
    assert rewound["episode"]["cursor_time"] == episode["start_time"]
    assert rewound["at_episode_start"] is True
