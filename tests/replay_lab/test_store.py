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


def test_v15_database_migrates_is_active_without_losing_events(tmp_path) -> None:
    import sqlite3

    db_path = tmp_path / "legacy_v15.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE episodes (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                start_time TEXT NOT NULL,
                cursor_time TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT NOT NULL,
                event_time TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timeframe TEXT,
                price REAL,
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(episode_id) REFERENCES episodes(id)
            );
            CREATE INDEX idx_events_episode_id ON events(episode_id, id);
            """
        )
        conn.execute(
            "INSERT INTO episodes VALUES(?,?,?,?,?,?,?)",
            (
                "legacy001",
                "SOXL-USDT-SWAP",
                "2026-08-05 07:30:00",
                "2026-08-05 07:30:00",
                "active",
                "2026-08-22T15:00:00+08:00",
                "2026-08-22T15:00:00+08:00",
            ),
        )
        conn.execute(
            "INSERT INTO events(episode_id,event_time,event_type,timeframe,price,payload_json,created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (
                "legacy001",
                "2026-08-05 07:30:00",
                "LIQUIDITY",
                "30m",
                140.0,
                '{"kind":"BSL"}',
                "2026-08-22T15:00:00+08:00",
            ),
        )

    store = ReplayStore(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(events)").fetchall()}
        is_active = conn.execute("SELECT is_active FROM events WHERE id=1").fetchone()[0]

    assert "is_active" in columns
    assert "idx_events_episode_active_time" in indexes
    assert is_active == 1
    events = store.list_events("legacy001")
    assert len(events) == 1
    assert events[0]["event_type"] == "LIQUIDITY"


def test_single_annotation_can_be_archived_without_hard_delete(tmp_path) -> None:
    store = ReplayStore(tmp_path / "replay.sqlite3")
    episode = store.create_episode("SOXL-USDT-SWAP", "2026-08-05 07:30:00")
    event = store.add_event(
        episode.id,
        "LIQUIDITY",
        episode.cursor_time,
        timeframe="30m",
        price=140.85,
        payload={"kind": "BSL"},
    )

    archived = store.deactivate_event(episode.id, event["id"])

    assert archived["event_type"] == "LIQUIDITY"
    assert store.list_events(episode.id) == []
    discarded = store.list_discarded_events(episode.id)
    assert len(discarded) == 1
    assert discarded[0]["id"] == event["id"]
    assert discarded[0]["payload"]["kind"] == "BSL"
