from __future__ import annotations

from pathlib import Path

from human_replay_lab.data_service import ReplayDataService
from human_replay_lab.server import ReplayApplication
from human_replay_lab.store import ReplayStore
from tests.human_replay_lab.test_data_service import seed_soxl


def test_soxl_episode_is_0730_et_and_shared_marker_keeps_source_context(tmp_path: Path) -> None:
    seed_soxl(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))
    episode = app.create_episode({"symbol": "SOXL-USDT-SWAP", "mode": "specific", "start_date": "2026-06-02"})
    assert episode["cursor_time"] == "2026-06-02 07:30:00"

    app.add_event(episode["id"], {
        "event_type": "LIQUIDITY",
        "timeframe": "30m",
        "price": 101.25,
        "payload": {
            "kind": "BSL",
            "anchor_time": "2026-06-02 07:00:00",
            "anchor_timeframe": "30m",
            "source_pane": "setup30",
            "magnet_enabled": True,
            "snap_field": "H",
            "raw_clicked_price": 101.22,
        },
    })
    snapshot = app.snapshots(episode["id"], ["30m", "15m", "2m", "1m"], 100)
    assert set(snapshot["charts"]) == {"30m", "15m", "2m", "1m"}
    liq = [e for e in snapshot["events"] if e["event_type"] == "LIQUIDITY"][0]
    assert liq["payload"]["snap_field"] == "H"
    assert liq["payload"]["anchor_timeframe"] == "30m"
    assert snapshot["clock"]["market_phase"] == "PREMARKET"
    assert snapshot["clock"]["source"].startswith("OKX")


def test_step_returns_incremental_chart_updates_without_full_snapshot(tmp_path: Path) -> None:
    seed_soxl(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))
    episode = app.create_episode({"symbol": "SOXL-USDT-SWAP", "mode": "specific", "start_date": "2026-06-02"})
    result = app.step(episode["id"], 2, ["30m", "15m", "2m", "1m"])
    assert result["episode"]["cursor_time"] == "2026-06-02 07:32:00"
    assert len(result["updates"]["1m"]) == 2
    assert len(result["updates"]["2m"]) == 1
    assert result["updates"]["15m"] == []
    assert result["updates"]["30m"] == []
