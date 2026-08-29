from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_feed.okx_loader import OKXDataLoader


SCRIPT = Path(__file__).resolve().parents[2] / "research" / "human_trader_replay" / "01_manual_setup_execution_audit.py"
spec = importlib.util.spec_from_file_location("manual_setup_execution_audit", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _replay_db(path: Path) -> None:
    con = sqlite3.connect(path)
    try:
        con.executescript(
            """
            CREATE TABLE episodes(id TEXT PRIMARY KEY,symbol TEXT,start_time TEXT,cursor_time TEXT,status TEXT,created_at TEXT,updated_at TEXT);
            CREATE TABLE events(id INTEGER PRIMARY KEY AUTOINCREMENT,episode_id TEXT,event_time TEXT,event_type TEXT,timeframe TEXT,price REAL,payload_json TEXT,created_at TEXT,is_active INTEGER DEFAULT 1);
            """
        )
        con.execute(
            "INSERT INTO episodes VALUES(?,?,?,?,?,?,?)",
            ("ep1", "SOXL-USDT-SWAP", "2026-06-19 07:30:00", "2026-06-19 12:00:00", "closed", "x", "x"),
        )
        events = [
            ("2026-06-19 08:30:00", "BIAS", "15m", None, {"bias": "BULLISH"}),
            ("2026-06-19 08:45:00", "WATCH", "2m", None, {}),
            ("2026-06-19 09:00:00", "LONG", "2m", 100.0, {"trade_id": "t1"}),
            ("2026-06-19 09:00:00", "TRADE_OPEN", "2m", 100.0, {"trade_id": "t1", "side": "LONG", "entry_event_id": 3, "entry_price": 100.0, "entry_time": "2026-06-19 09:00:00", "initial_stop_loss": 99.0, "initial_take_profit": 103.0}),
            ("2026-06-19 09:11:00", "STOP_LOSS_HIT", "2m", 99.0, {"trade_id": "t1"}),
            ("2026-06-19 09:11:00", "TRADE_CLOSED", "2m", 99.0, {"trade_id": "t1", "side": "LONG", "entry_event_id": 3, "entry_price": 100.0, "entry_time": "2026-06-19 09:00:00", "exit_price": 99.0, "exit_time": "2026-06-19 09:11:00", "exit_reason": "STOP_LOSS", "r_multiple": -1.0, "net_return_pct": -1.11}),
        ]
        for when, typ, tf, price, payload in events:
            con.execute(
                "INSERT INTO events(episode_id,event_time,event_type,timeframe,price,payload_json,created_at,is_active) VALUES(?,?,?,?,?,?,?,1)",
                ("ep1", when, typ, tf, price, json.dumps(payload), "x"),
            )
        con.commit()
    finally:
        con.close()


def _market_data(data_dir: Path) -> None:
    # 2026-06-19 is EDT, Beijing is 12 hours ahead of New York.
    idx = pd.date_range("2026-06-19 20:30:00", "2026-06-20 04:00:00", freq="1min")
    frame = pd.DataFrame(index=idx, columns=["open", "high", "low", "close", "volume"], dtype=float)
    frame[["open", "high", "low", "close"]] = 100.0
    frame["volume"] = 1.0
    # NY 09:10 -> BJT 21:10: stop first.
    frame.loc[pd.Timestamp("2026-06-19 21:10:00"), ["open", "high", "low", "close"]] = [99.4, 99.5, 98.8, 99.1]
    # NY 09:30 -> BJT 21:30: original TP later.
    frame.loc[pd.Timestamp("2026-06-19 21:30:00"), ["open", "high", "low", "close"]] = [102.5, 103.2, 102.4, 103.0]
    frame.index.name = "timestamp"
    OKXDataLoader(symbol="SOXL-USDT-SWAP", timeframe="1m", db_dir=str(data_dir)).save_local_data(frame)


def test_manual_audit_labels_stop_first_then_target(tmp_path):
    replay = tmp_path / "replay.sqlite3"
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _replay_db(replay)
    _market_data(data_dir)
    out = tmp_path / "reports"
    args = mod.parse_args([
        "--replay-db", str(replay),
        "--data-dir", str(data_dir),
        "--out-dir", str(out),
        "--no-progress",
    ])
    audit, horizon, episodes, legacy = mod.run(args)
    assert len(audit) == 1
    row = audit.iloc[0]
    assert bool(row["stop_first_then_target"]) is True
    assert bool(row["premature_entry_candidate"]) is True
    assert row["primary_failure_taxonomy"] == "EXECUTION_STOP_FIRST_TARGET_LATER"
    assert row["survival_mae_pct"] >= 1.0
    assert row["bias_conflict"] is False or bool(row["bias_conflict"]) is False
    assert len(horizon) == 4
    assert legacy.empty
    assert (out / "trade_path_audit.csv").is_file()
    assert (out / "manual_setup_execution_audit.md").is_file()
