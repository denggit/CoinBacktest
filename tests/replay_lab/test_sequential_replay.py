from __future__ import annotations

import numpy as np
import pandas as pd

from human_replay_lab.data_service import ReplayDataService
from human_replay_lab.server import ReplayApplication
from human_replay_lab.store import ReplayStore
from src.data_feed.okx_loader import OKXDataLoader


def _seed_eth(tmp_path) -> None:
    # OKX project-local rows are stored as source/Beijing wall time (+8).
    # Start at 00:02 on purpose so sequential start at 00:00 must skip a real local-data gap.
    idx = pd.date_range("2026-01-01 00:02:00", "2026-01-01 00:12:00", freq="1min")
    n = len(idx)
    base = np.arange(n, dtype=float) + 3000.0
    frame = pd.DataFrame(
        {
            "open": base,
            "high": base + 1.0,
            "low": base - 1.0,
            "close": base + 0.5,
            "volume": np.ones(n),
        },
        index=idx,
    )
    frame.index.name = "timestamp"
    OKXDataLoader(symbol="ETH-USDT-SWAP", timeframe="1m", db_dir=str(tmp_path)).save_local_data(frame)


def test_eth_sequential_start_uses_selected_beijing_date_and_first_available_bar(tmp_path) -> None:
    _seed_eth(tmp_path)
    service = ReplayDataService(tmp_path)

    cursor = service.sequential_start_cursor("ETH-USDT-SWAP", "2026-01-01T00:00")
    assert service.beijing_display(cursor) == "2026-01-01 00:02:00"


def test_eth_sequential_episode_continues_after_previous_closed_cursor(tmp_path) -> None:
    _seed_eth(tmp_path)
    app = ReplayApplication(ReplayDataService(tmp_path), ReplayStore(tmp_path / "replay.sqlite3"))

    first = app.create_episode(
        {
            "symbol": "ETH-USDT-SWAP",
            "mode": "sequential",
            "start_time": "2026-01-01T00:00",
        }
    )
    assert app.data.beijing_display(first["cursor_time"]) == "2026-01-01 00:02:00"

    finalized = app.close_episode(first["id"])
    assert finalized["episode"]["status"] == "closed"

    second = app.create_episode(
        {
            "symbol": "ETH-USDT-SWAP",
            "mode": "sequential",
            "previous_episode_id": first["id"],
        }
    )
    assert app.data.beijing_display(second["cursor_time"]) == "2026-01-01 00:03:00"

    start_event = app.store.list_events(second["id"])[0]
    assert start_event["event_type"] == "EPISODE_START"
    assert start_event["payload"]["mode"] == "sequential"
    assert start_event["payload"]["previous_episode_id"] == first["id"]
    assert start_event["payload"]["sequence_policy"] == "next_available_1m_after_previous_close"


def test_ui_exposes_from_date_sequential_mode_as_default() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    html = (root / "human_replay_lab" / "static" / "index.html").read_text(encoding="utf-8")
    js = (root / "human_replay_lab" / "static" / "app.js").read_text(encoding="utf-8")

    assert 'name="startMode" value="sequential" checked' in html
    assert 'id="sequentialStartDate"' in html
    assert 'id="sequentialStartTime"' in html
    assert "previous_episode_id" in js
    assert "继续下一 Episode" in js
