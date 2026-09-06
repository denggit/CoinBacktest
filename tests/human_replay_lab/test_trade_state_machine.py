from __future__ import annotations

import pytest

from human_replay_lab.server import ReplayApplication
from human_replay_lab.store import ReplayStore


def test_manual_close_is_rejected_without_appending_events(tmp_path) -> None:
    store = ReplayStore(tmp_path / "manual-close.sqlite3")
    app = ReplayApplication(None, store)  # type: ignore[arg-type]
    episode = store.create_episode("ETH-USDT-SWAP", "2026-08-01 00:00:00")

    with pytest.raises(ValueError, match="manual close is disabled"):
        app.trade(episode.id, {"side": "CLOSE"})

    assert store.list_events(episode.id) == []


@pytest.mark.parametrize("side", ["LONG", "SHORT"])
def test_closed_episode_rejects_new_positions_without_appending_events(tmp_path, side: str) -> None:
    store = ReplayStore(tmp_path / f"closed-{side.lower()}.sqlite3")
    app = ReplayApplication(None, store)  # type: ignore[arg-type]
    episode = store.create_episode("ETH-USDT-SWAP", "2026-08-01 00:00:00")
    store.close_episode(episode.id)

    with pytest.raises(ValueError, match="episode is not active"):
        app.trade(episode.id, {"side": side, "order_type": "market"})

    assert store.list_events(episode.id) == []


def test_episode_cannot_end_while_a_trade_is_open(tmp_path) -> None:
    store = ReplayStore(tmp_path / "open-trade.sqlite3")
    app = ReplayApplication(None, store)  # type: ignore[arg-type]
    episode = store.create_episode("ETH-USDT-SWAP", "2026-08-01 00:00:00")
    store.add_event(
        episode.id,
        "TRADE_OPEN",
        episode.cursor_time,
        timeframe="1m",
        price=100.0,
        payload={
            "trade_id": "trade-1",
            "side": "LONG",
            "entry_price": 100.0,
            "initial_stop_loss": 99.0,
            "initial_take_profit": 102.0,
        },
    )
    before = store.list_events(episode.id)

    with pytest.raises(ValueError, match="TP 或 SL"):
        app.close_episode(episode.id)

    assert store.get_episode(episode.id).status == "active"
    assert store.list_events(episode.id) == before
