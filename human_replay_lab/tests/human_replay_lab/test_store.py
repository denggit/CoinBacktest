from human_replay_lab.store import ReplayStore


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
