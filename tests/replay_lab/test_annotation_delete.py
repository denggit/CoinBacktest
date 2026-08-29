from pathlib import Path

from human_replay_lab.data_service import ReplayDataService
from human_replay_lab.server import ReplayApplication
from human_replay_lab.store import ReplayStore
from tests.replay_lab.test_data_service import seed_soxl


def test_liquidity_delete_removes_line_but_keeps_correction_audit(tmp_path: Path) -> None:
    seed_soxl(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))
    episode = app.create_episode({"symbol": "SOXL-USDT-SWAP", "mode": "specific", "start_date": "2026-06-02"})
    liquidity = app.add_event(episode["id"], {
        "event_type": "LIQUIDITY",
        "timeframe": "30m",
        "price": 103.25,
        "payload": {"kind": "BSL", "importance": "high", "anchor_time": "2026-06-02 07:00:00"},
    })

    result = app.delete_annotation(episode["id"], {"event_id": liquidity["id"]})

    active = result["events"]
    assert all(event["id"] != liquidity["id"] for event in active)
    correction = result["correction_event"]
    assert correction["event_type"] == "ANNOTATION_DELETE"
    assert correction["payload"]["target_event_id"] == liquidity["id"]
    assert correction["payload"]["target_kind"] == "BSL"
    assert correction["payload"]["target_price"] == 103.25

    exported = app.store.export_episode(episode["id"])
    assert [event["id"] for event in exported["discarded_events"]] == [liquidity["id"]]


def test_trade_event_cannot_be_deleted_as_annotation(tmp_path: Path) -> None:
    seed_soxl(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))
    episode = app.create_episode({"symbol": "SOXL-USDT-SWAP", "mode": "specific", "start_date": "2026-06-02"})
    trade = app.trade(episode["id"], {"side": "LONG", "timeframe": "1m", "order_type": "market"})

    try:
        app.delete_annotation(episode["id"], {"event_id": trade["event"]["id"]})
    except ValueError as exc:
        assert "only Liquidity" in str(exc)
    else:
        raise AssertionError("trade event deletion must be rejected")
