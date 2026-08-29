#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit Human Trader Replay Lab trades against the real local OKX 1m path.

The research intentionally does *not* optimize a stop parameter or invent a new
ICT setup.  It answers a narrower execution question using the trader's own
recorded decisions:

1. Was the original target eventually delivered after a stopped trade?
2. How much adverse excursion was required to survive until that target?
3. Was the trade aligned with the latest manually-recorded Bias?
4. Did a later same-side re-entry in the same Episode succeed?
5. Are failures more consistent with direction failure or early execution?

All decision/context fields come from active Replay Lab events only.  Future
1m data is used strictly as *post-trade outcome audit*; it is never used to
reconstruct or change the historical entry decision.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_loader import OKXDataLoader, TIMEZONE as OKX_LOADER_TIMEZONE  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402

NY_TZ = "America/New_York"
DEFAULT_REPLAY_DB = "data/human_replay_lab/replay.sqlite3"
DEFAULT_OUT_DIR = "data/reports/research/human_trader_replay/manual_setup_execution_audit"
DEFAULT_HORIZONS = (30, 60, 120, 240)
DEFAULT_ROUND_TRIP_COST = 0.0011


def _source_offset_hours(text: str) -> int:
    value = str(text or "+8").strip().upper().replace("UTC", "")
    try:
        return int(value)
    except ValueError:
        return 8


def _ny_wall(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(NY_TZ).tz_localize(None)
    return ts


def _ny_to_source_naive(value: Any) -> pd.Timestamp:
    wall = _ny_wall(value)
    aware_ny = pd.Timestamp(wall.to_pydatetime().replace(tzinfo=ZoneInfo(NY_TZ)))
    source_tz = timezone(timedelta(hours=_source_offset_hours(OKX_LOADER_TIMEZONE)))
    return aware_ny.tz_convert(source_tz).tz_localize(None)


def _source_frame_to_ny(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    idx = pd.DatetimeIndex(out.index)
    if idx.tz is None:
        source_tz = timezone(timedelta(hours=_source_offset_hours(OKX_LOADER_TIMEZONE)))
        idx = idx.tz_localize(source_tz)
    out.index = idx.tz_convert(NY_TZ).tz_localize(None)
    out.index.name = "timestamp_et"
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_index()


def _decode_payload(text: Any) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(str(text))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _load_replay_tables(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.is_file():
        raise FileNotFoundError(f"Replay database not found: {path}")
    conn = sqlite3.connect(str(path))
    try:
        episodes = pd.read_sql_query("SELECT * FROM episodes ORDER BY start_time,id", conn)
        events = pd.read_sql_query(
            "SELECT * FROM events WHERE COALESCE(is_active,1)=1 ORDER BY episode_id,event_time,id",
            conn,
        )
    finally:
        conn.close()
    if not events.empty:
        events["payload"] = events["payload_json"].map(_decode_payload)
        events["event_time_ts"] = pd.to_datetime(events["event_time"], errors="coerce")
    return episodes, events


def _event_payload_value(events: pd.DataFrame, event_type: str, key: str, *, trade_id: str | None = None) -> Any:
    if events.empty:
        return None
    subset = events[events["event_type"].eq(event_type)]
    if trade_id is not None:
        subset = subset[subset["payload"].map(lambda p: str(p.get("trade_id") or "") == trade_id)]
    if subset.empty:
        return None
    return subset.iloc[-1]["payload"].get(key)


def _latest_event_before(events: pd.DataFrame, event_type: str, at: pd.Timestamp) -> pd.Series | None:
    subset = events[(events["event_type"] == event_type) & (events["event_time_ts"] <= at)]
    if subset.empty:
        return None
    return subset.sort_values(["event_time_ts", "id"]).iloc[-1]


def _context_for_trade(episode_events: pd.DataFrame, entry_time: pd.Timestamp) -> dict[str, Any]:
    bias_event = _latest_event_before(episode_events, "BIAS", entry_time)
    watch_event = _latest_event_before(episode_events, "WATCH", entry_time)
    notes = episode_events[
        (episode_events["event_type"] == "NOTE") & (episode_events["event_time_ts"] <= entry_time)
    ].sort_values(["event_time_ts", "id"])
    liquidity = episode_events[
        (episode_events["event_type"] == "LIQUIDITY") & (episode_events["event_time_ts"] <= entry_time)
    ]
    bias = None if bias_event is None else str((bias_event["payload"] or {}).get("bias") or "").upper() or None
    watch_time = None if watch_event is None else pd.Timestamp(watch_event["event_time_ts"])
    note_text = " | ".join(str(p.get("text") or "").strip() for p in notes.tail(4)["payload"] if str(p.get("text") or "").strip())
    bsl = int(sum(1 for p in liquidity["payload"] if str(p.get("kind") or "").upper() == "BSL")) if not liquidity.empty else 0
    ssl = int(sum(1 for p in liquidity["payload"] if str(p.get("kind") or "").upper() == "SSL")) if not liquidity.empty else 0
    return {
        "bias_at_entry": bias,
        "watch_time": watch_time,
        "watch_to_entry_minutes": None if watch_time is None else float((entry_time - watch_time) / pd.Timedelta(minutes=1)),
        "notes_before_entry": note_text,
        "bsl_count_at_entry": bsl,
        "ssl_count_at_entry": ssl,
    }


def _extract_trades(episodes: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    episode_map = episodes.set_index("id").to_dict("index") if not episodes.empty else {}
    rows: list[dict[str, Any]] = []
    legacy: list[dict[str, Any]] = []

    closed = events[events["event_type"] == "TRADE_CLOSED"].copy()
    for _, event in closed.iterrows():
        payload = event["payload"] or {}
        episode_id = str(event["episode_id"])
        ep = episode_map.get(episode_id, {})
        ep_events = events[events["episode_id"] == episode_id]
        trade_id = str(payload.get("trade_id") or "")
        open_events = ep_events[
            (ep_events["event_type"] == "TRADE_OPEN")
            & ep_events["payload"].map(lambda p: str(p.get("trade_id") or "") == trade_id)
        ]
        open_payload = open_events.iloc[-1]["payload"] if not open_events.empty else {}
        entry_time = pd.Timestamp(payload.get("entry_time") or open_payload.get("entry_time") or event["event_time"])
        exit_time = pd.Timestamp(payload.get("exit_time") or event["event_time"])
        side = str(payload.get("side") or open_payload.get("side") or "").upper()
        entry = float(payload.get("entry_price") or open_payload.get("entry_price") or np.nan)
        exit_price = float(payload.get("exit_price") or event.get("price") or np.nan)
        stop = open_payload.get("initial_stop_loss")
        target = open_payload.get("initial_take_profit")
        if stop is None:
            stop = _event_payload_value(ep_events, "SL", "price", trade_id=trade_id)
        if target is None:
            target = _event_payload_value(ep_events, "TP", "price", trade_id=trade_id)
        # SL/TP price normally lives in the event column, not payload.
        if stop is None:
            s = ep_events[(ep_events["event_type"] == "SL") & ep_events["payload"].map(lambda p: str(p.get("trade_id") or "") == trade_id)]
            stop = None if s.empty else s.iloc[-1]["price"]
        if target is None:
            t = ep_events[(ep_events["event_type"] == "TP") & ep_events["payload"].map(lambda p: str(p.get("trade_id") or "") == trade_id)]
            target = None if t.empty else t.iloc[-1]["price"]
        context = _context_for_trade(ep_events, entry_time)
        rows.append({
            "episode_id": episode_id,
            "symbol": str(ep.get("symbol") or ""),
            "episode_start": ep.get("start_time"),
            "episode_status": ep.get("status"),
            "trade_id": trade_id,
            "side": side,
            "entry_time": entry_time,
            "entry_price": entry,
            "stop_loss": None if stop is None else float(stop),
            "take_profit": None if target is None else float(target),
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": str(payload.get("exit_reason") or "UNKNOWN"),
            "recorded_net_return_pct": payload.get("net_return_pct"),
            "recorded_r_multiple": payload.get("r_multiple"),
            "timeframe": event.get("timeframe"),
            **context,
        })

    # Explicitly report legacy LONG/SHORT fills that never received TRADE_CLOSED.
    for episode_id, ep_events in events.groupby("episode_id"):
        closed_entry_ids = {
            int((p or {}).get("entry_event_id") or 0)
            for p in ep_events.loc[ep_events["event_type"] == "TRADE_CLOSED", "payload"]
        }
        represented = {
            int((p or {}).get("entry_event_id") or 0)
            for p in ep_events.loc[ep_events["event_type"] == "TRADE_OPEN", "payload"]
        }
        for _, ev in ep_events[ep_events["event_type"].isin(["LONG", "SHORT"])].iterrows():
            eid = int(ev["id"])
            if eid in closed_entry_ids or eid in represented:
                continue
            legacy.append({
                "episode_id": episode_id,
                "event_id": eid,
                "event_time": ev["event_time"],
                "side": ev["event_type"],
                "price": ev["price"],
                "reason": "legacy fill has no TRADE_CLOSED; excluded from outcome statistics",
            })

    trades = pd.DataFrame(rows)
    if not trades.empty:
        trades["episode_day"] = pd.to_datetime(trades["episode_start"]).dt.strftime("%Y-%m-%d")
        key = trades["symbol"].astype(str) + "|" + trades["episode_day"].astype(str)
        counts = key.value_counts()
        trades["replayed_day_trade_count"] = key.map(counts).astype(int)
    return trades, pd.DataFrame(legacy)


def _load_market_windows(trades: pd.DataFrame, data_dir: Path, max_forward_minutes: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for symbol, group in trades.groupby("symbol"):
        if not symbol:
            continue
        start = pd.to_datetime(group["entry_time"]).min().normalize() - pd.Timedelta(days=1)
        end = pd.to_datetime(group["entry_time"]).max().normalize() + pd.Timedelta(days=1, minutes=max_forward_minutes)
        loader = OKXDataLoader(symbol=symbol, timeframe="1m", db_dir=str(data_dir))
        raw = loader.load_local_data_range(_ny_to_source_naive(start), _ny_to_source_naive(end))
        if raw.empty:
            raise RuntimeError(f"No local OKX {symbol} 1m data available for audit window {start} -> {end}")
        out[symbol] = _source_frame_to_ny(raw)
    return out


def _touch_target(row: pd.Series, side: str, target: float) -> bool:
    return float(row["high"]) >= target if side == "LONG" else float(row["low"]) <= target


def _touch_stop(row: pd.Series, side: str, stop: float) -> bool:
    return float(row["low"]) <= stop if side == "LONG" else float(row["high"]) >= stop


def _excursions(frame: pd.DataFrame, side: str, entry: float) -> tuple[float, float]:
    if frame.empty:
        return float("nan"), float("nan")
    if side == "LONG":
        mfe = (float(frame["high"].max()) - entry) / entry * 100.0
        mae = (entry - float(frame["low"].min())) / entry * 100.0
    else:
        mfe = (entry - float(frame["low"].min())) / entry * 100.0
        mae = (float(frame["high"].max()) - entry) / entry * 100.0
    return max(0.0, mfe), max(0.0, mae)


def _first_target_time(frame: pd.DataFrame, side: str, target: float) -> pd.Timestamp | None:
    if frame.empty:
        return None
    if side == "LONG":
        mask = frame["high"] >= target
    else:
        mask = frame["low"] <= target
    hits = frame.index[mask]
    return None if len(hits) == 0 else pd.Timestamp(hits[0])


def _day_end(entry_time: pd.Timestamp, max_forward_minutes: int) -> pd.Timestamp:
    session_end = entry_time.normalize() + pd.Timedelta(hours=16)
    horizon_end = entry_time + pd.Timedelta(minutes=max_forward_minutes)
    return max(entry_time, min(session_end, horizon_end))


def _audit_trade(trade: pd.Series, market: pd.DataFrame, horizons: Sequence[int], max_forward_minutes: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    side = str(trade["side"]).upper()
    entry_time = pd.Timestamp(trade["entry_time"])
    exit_time = pd.Timestamp(trade["exit_time"])
    entry = float(trade["entry_price"])
    stop = float(trade["stop_loss"]) if pd.notna(trade["stop_loss"]) else float("nan")
    target = float(trade["take_profit"]) if pd.notna(trade["take_profit"]) else float("nan")
    day_end = _day_end(entry_time, max_forward_minutes)

    # Exclude the 1m bar whose close made the exit known, matching the replay
    # engine's conservative entry/exit-bar policy for path statistics.
    pre_exit = market[(market.index >= entry_time) & (market.index < max(entry_time, exit_time - pd.Timedelta(minutes=1)))]
    mfe, mae = _excursions(pre_exit, side, entry)

    post_entry = market[(market.index >= entry_time) & (market.index < day_end)]
    target_any_time = None if not math.isfinite(target) else _first_target_time(post_entry, side, target)
    post_exit = market[(market.index >= exit_time) & (market.index < day_end)]
    target_after_stop_time = None
    if str(trade["exit_reason"]).upper() == "STOP_LOSS" and math.isfinite(target):
        target_after_stop_time = _first_target_time(post_exit, side, target)

    stop_first_then_target = target_after_stop_time is not None
    survival_mae_pct = float("nan")
    survival_mae_r = float("nan")
    extra_buffer_pct = float("nan")
    if stop_first_then_target:
        survival_end = target_after_stop_time + pd.Timedelta(minutes=1)
        survival = market[(market.index >= entry_time) & (market.index < survival_end)]
        _, survival_mae_pct = _excursions(survival, side, entry)
        risk_pct = abs(entry - stop) / entry * 100.0 if math.isfinite(stop) and stop != entry else float("nan")
        survival_mae_r = survival_mae_pct / risk_pct if math.isfinite(risk_pct) and risk_pct > 0 else float("nan")
        extra_buffer_pct = max(0.0, survival_mae_pct - risk_pct) if math.isfinite(risk_pct) else float("nan")

    risk_distance_pct = abs(entry - stop) / entry * 100.0 if math.isfinite(stop) else float("nan")
    target_distance_pct = abs(target - entry) / entry * 100.0 if math.isfinite(target) else float("nan")
    planned_rr = target_distance_pct / risk_distance_pct if math.isfinite(risk_distance_pct) and risk_distance_pct > 0 else float("nan")
    bias = str(trade.get("bias_at_entry") or "").upper()
    bias_conflict = (bias == "BULLISH" and side == "SHORT") or (bias == "BEARISH" and side == "LONG")
    recorded_r = pd.to_numeric(pd.Series([trade.get("recorded_r_multiple")]), errors="coerce").iloc[0]
    win = bool(pd.notna(recorded_r) and float(recorded_r) > 0) or str(trade["exit_reason"]).upper() == "TAKE_PROFIT"

    result = dict(trade)
    result.update({
        "risk_distance_pct": risk_distance_pct,
        "target_distance_pct": target_distance_pct,
        "planned_rr": planned_rr,
        "path_mfe_pct": mfe,
        "path_mae_pct": mae,
        "path_bars_used": int(len(pre_exit)),
        "target_eventually_hit_by_day_end": target_any_time is not None,
        "target_eventual_time": None if target_any_time is None else target_any_time.strftime("%Y-%m-%d %H:%M:%S"),
        "stop_first_then_target": stop_first_then_target,
        "target_after_stop_time": None if target_after_stop_time is None else target_after_stop_time.strftime("%Y-%m-%d %H:%M:%S"),
        "minutes_stop_to_target": None if target_after_stop_time is None else float((target_after_stop_time - exit_time) / pd.Timedelta(minutes=1)),
        "survival_mae_pct": survival_mae_pct,
        "survival_mae_r": survival_mae_r,
        "required_extra_buffer_pct": extra_buffer_pct,
        "bias_conflict": bool(bias_conflict),
        "is_win": bool(win),
    })

    horizon_rows: list[dict[str, Any]] = []
    for horizon in horizons:
        end = min(day_end, entry_time + pd.Timedelta(minutes=int(horizon)))
        segment = market[(market.index >= entry_time) & (market.index < end)]
        hmfe, hmae = _excursions(segment, side, entry)
        target_hit = False if not math.isfinite(target) else _first_target_time(segment, side, target) is not None
        stop_hit = False
        if math.isfinite(stop) and not segment.empty:
            stop_hit = bool(segment.apply(lambda r: _touch_stop(r, side, stop), axis=1).any())
        horizon_rows.append({
            "episode_id": trade["episode_id"],
            "trade_id": trade["trade_id"],
            "symbol": trade["symbol"],
            "side": side,
            "entry_time": entry_time,
            "horizon_minutes": int(horizon),
            "bars": int(len(segment)),
            "target_hit": bool(target_hit),
            "stop_touched": bool(stop_hit),
            "mfe_pct": hmfe,
            "mae_pct": hmae,
        })
    return result, horizon_rows


def _attach_reentry_flags(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return audit
    out = audit.sort_values(["episode_id", "entry_time", "trade_id"]).copy()
    later_any = []
    later_win = []
    for _, row in out.iterrows():
        later = out[
            (out["episode_id"] == row["episode_id"])
            & (out["side"] == row["side"])
            & (pd.to_datetime(out["entry_time"]) > pd.Timestamp(row["exit_time"]))
        ]
        later_any.append(not later.empty)
        later_win.append(bool((later["is_win"] == True).any()) if not later.empty else False)  # noqa: E712
    out["later_same_side_reentry"] = later_any
    out["later_same_side_winner"] = later_win
    out["premature_entry_candidate"] = out["stop_first_then_target"].astype(bool) | out["later_same_side_winner"].astype(bool)

    primary = []
    for _, row in out.iterrows():
        reason = str(row["exit_reason"]).upper()
        if bool(row["is_win"]):
            primary.append("WIN")
        elif reason == "TRADE_EXIT_AMBIGUOUS" or reason == "AMBIGUOUS":
            primary.append("AMBIGUOUS")
        elif bool(row["stop_first_then_target"]):
            primary.append("EXECUTION_STOP_FIRST_TARGET_LATER")
        elif bool(row["bias_conflict"]):
            primary.append("CONTEXT_BIAS_CONFLICT")
        elif reason == "STOP_LOSS":
            primary.append("DIRECTION_OR_TIMING_FAILURE")
        elif reason == "MANUAL_EXIT":
            primary.append("MANUAL_EXIT")
        else:
            primary.append(reason or "OTHER")
    out["primary_failure_taxonomy"] = primary
    return out


def _episode_audit(episodes: pd.DataFrame, trades: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    trade_groups = {k: v for k, v in trades.groupby("episode_id")} if not trades.empty else {}
    for _, ep in episodes.iterrows():
        eid = str(ep["id"])
        ev = events[events["episode_id"] == eid]
        tr = trade_groups.get(eid, pd.DataFrame())
        day = pd.Timestamp(ep["start_time"]).strftime("%Y-%m-%d")
        rows.append({
            "episode_id": eid,
            "symbol": ep["symbol"],
            "episode_day": day,
            "status": ep["status"],
            "event_count": int(len(ev)),
            "trade_count": int(len(tr)),
            "wins": int(tr["is_win"].sum()) if not tr.empty and "is_win" in tr else 0,
            "notes": int((ev["event_type"] == "NOTE").sum()) if not ev.empty else 0,
            "liquidity_marks": int((ev["event_type"] == "LIQUIDITY").sum()) if not ev.empty else 0,
            "rewinds": int((ev["event_type"] == "REWIND").sum()) if not ev.empty else 0,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        key = out["symbol"].astype(str) + "|" + out["episode_day"].astype(str)
        counts = key.value_counts()
        out["same_day_episode_count"] = key.map(counts).astype(int)
        out["duplicate_replay_day"] = out["same_day_episode_count"] > 1
    return out


def _safe_mean(series: Iterable[Any]) -> float:
    values = pd.to_numeric(pd.Series(list(series)), errors="coerce").dropna()
    return float(values.mean()) if len(values) else float("nan")


def _summary_table(audit: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    if audit.empty:
        return pd.DataFrame()
    for value, group in audit.groupby(key, dropna=False):
        rows.append({
            key: value,
            "trades": len(group),
            "wins": int(group["is_win"].sum()),
            "win_rate_pct": float(group["is_win"].mean() * 100.0),
            "mean_recorded_r": _safe_mean(group["recorded_r_multiple"]),
            "mean_risk_distance_pct": _safe_mean(group["risk_distance_pct"]),
            "stop_first_then_target": int(group["stop_first_then_target"].sum()),
            "premature_candidates": int(group["premature_entry_candidate"].sum()),
        })
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame, columns: Sequence[str] | None = None, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    view = df.loc[:, list(columns)] if columns else df
    view = view.head(max_rows).copy()
    for col in view.columns:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else (f"{x:.3f}" if isinstance(x, float) else str(x)))
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[c]).replace("|", "\\|") for c in headers) + " |")
    return "\n".join(lines)


def _write_report(out_dir: Path, audit: pd.DataFrame, horizon: pd.DataFrame, episodes: pd.DataFrame, legacy: pd.DataFrame, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_dir / "trade_path_audit.csv", index=False, encoding="utf-8-sig")
    horizon.to_csv(out_dir / "horizon_audit.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(out_dir / "episode_audit.csv", index=False, encoding="utf-8-sig")
    failures = audit[~audit["is_win"]].copy() if not audit.empty else pd.DataFrame()
    failures.to_csv(out_dir / "failure_taxonomy.csv", index=False, encoding="utf-8-sig")
    legacy.to_csv(out_dir / "legacy_unclosed_trade_candidates.csv", index=False, encoding="utf-8-sig")

    side = _summary_table(audit, "side")
    bias_state = audit.copy()
    if not bias_state.empty:
        bias_state["bias_relation"] = np.where(
            bias_state["bias_at_entry"].isna() | bias_state["bias_at_entry"].eq(""),
            "UNKNOWN",
            np.where(bias_state["bias_conflict"], "CONFLICT", "ALIGNED_OR_NEUTRAL"),
        )
    bias_summary = _summary_table(bias_state, "bias_relation") if not bias_state.empty else pd.DataFrame()
    taxonomy = audit.groupby("primary_failure_taxonomy", dropna=False).size().reset_index(name="trades") if not audit.empty else pd.DataFrame()

    total = len(audit)
    wins = int(audit["is_win"].sum()) if total else 0
    losses = total - wins
    sft = int(audit["stop_first_then_target"].sum()) if total else 0
    sft_losses = int((audit["stop_first_then_target"] & ~audit["is_win"]).sum()) if total else 0
    premature = int(audit["premature_entry_candidate"].sum()) if total else 0
    duplicates = int(episodes["duplicate_replay_day"].sum()) if not episodes.empty else 0

    loss_columns = [
        "episode_day", "side", "entry_price", "stop_loss", "take_profit", "exit_reason",
        "bias_at_entry", "bias_conflict", "stop_first_then_target", "minutes_stop_to_target",
        "risk_distance_pct", "survival_mae_pct", "survival_mae_r",
        "later_same_side_winner", "primary_failure_taxonomy",
    ]
    loss_table = _markdown_table(failures, loss_columns if not failures.empty else None, max_rows=50)
    report = f"""# Manual Setup Execution Audit

## Purpose

Use the trader's own Replay Lab decisions and local OKX 1m path to separate **direction failure** from **execution/timing/stop failure**. This is an outcome audit, not a parameter optimizer.

## Data

- Replay DB: `{args.replay_db}`
- Data dir: `{args.data_dir}`
- Complete lifecycle trades audited: **{total}**
- Wins / losses: **{wins} / {losses}**
- Win rate: **{(wins / total * 100.0 if total else 0):.1f}%**
- Legacy fills excluded because no TRADE_CLOSED: **{len(legacy)}**
- Duplicate replay-day episode rows: **{duplicates}** (do not treat repeated blind replays as independent samples)

## Core diagnostic

- Losses where original stop hit first but original target was delivered later: **{sft_losses}/{losses if losses else 0}**
- All stop-first-then-target cases: **{sft}**
- Premature-entry candidates (stop-first-target-later OR later same-side winner in same Episode): **{premature}/{total}**

These flags do **not** prove the stop should simply be widened. They identify trades where direction and execution should be studied separately.

## By side

{_markdown_table(side)}

## Bias relationship

{_markdown_table(bias_summary)}

## Failure taxonomy

{_markdown_table(taxonomy)}

## Loss audit

{loss_table}

## Interpretation rules

1. `EXECUTION_STOP_FIRST_TARGET_LATER`: the recorded stop was hit, then the original TP printed later inside the audit window. Direction eventually delivered, but the original execution did not survive.
2. `CONTEXT_BIAS_CONFLICT`: the trade direction contradicted the latest explicit manual Bullish/Bearish Bias and no stronger stop-first-target-later explanation was observed.
3. `DIRECTION_OR_TIMING_FAILURE`: stop loss with no later original-target delivery inside the audit window. More data is needed before calling it a pure direction error.
4. `premature_entry_candidate` is diagnostic only; it is **not** a new strategy rule.

## Causality / anti-overfit

- Replay decision events are read exactly as saved; discarded rewind branches are excluded.
- Local OKX 1m bars after entry are used only for ex-post path labeling.
- No stop-distance grid search, no parameter tuning, and no holdout optimization is performed.
- Same-bar path ambiguity is not resolved optimistically.

## Next step

Accumulate more blind Replay Episodes, then rerun this exact audit unchanged. Do not tune rules from this small sample. Once the sample is materially larger, compare stable failure mechanisms by side, Bias relation, setup timeframe, and liquidity context.
"""
    (out_dir / "manual_setup_execution_audit.md").write_text(report, encoding="utf-8")
    manifest = {
        "research": "01_manual_setup_execution_audit",
        "replay_db": str(args.replay_db),
        "data_dir": str(args.data_dir),
        "complete_trades": total,
        "legacy_unclosed": len(legacy),
        "horizons_minutes": list(args.horizons),
        "max_forward_minutes": int(args.max_forward_minutes),
        "round_trip_cost": float(args.round_trip_cost),
        "outputs": [
            "trade_path_audit.csv",
            "horizon_audit.csv",
            "failure_taxonomy.csv",
            "episode_audit.csv",
            "legacy_unclosed_trade_candidates.csv",
            "manual_setup_execution_audit.md",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_horizons(text: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x.strip()) for x in str(text).split(",") if x.strip()}))
    if not values or any(v <= 0 for v in values):
        raise ValueError("horizons must contain positive minute values")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit manual Replay Lab execution against local OKX 1m paths")
    p.add_argument("--replay-db", default=DEFAULT_REPLAY_DB)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--horizons", default=",".join(str(x) for x in DEFAULT_HORIZONS))
    p.add_argument("--max-forward-minutes", type=int, default=480)
    p.add_argument("--round-trip-cost", type=float, default=DEFAULT_ROUND_TRIP_COST)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)
    args.horizons = _parse_horizons(args.horizons)
    if args.max_forward_minutes <= 0:
        raise ValueError("max-forward-minutes must be > 0")
    return args


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    replay_db = Path(args.replay_db)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    episodes, events = _load_replay_tables(replay_db)
    trades, legacy = _extract_trades(episodes, events)
    if trades.empty:
        raise RuntimeError("No active TRADE_CLOSED lifecycle records found in Replay DB")
    markets = _load_market_windows(trades, data_dir, int(args.max_forward_minutes))

    audited: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    progress = ProgressReporter(
        label="[manual-audit] trades",
        total=len(trades),
        every=1,
        enabled=not args.no_progress,
    )
    for i, (_, trade) in enumerate(trades.iterrows(), start=1):
        market = markets.get(str(trade["symbol"]))
        if market is None or market.empty:
            raise RuntimeError(f"missing market frame for {trade['symbol']}")
        row, hrows = _audit_trade(trade, market, args.horizons, int(args.max_forward_minutes))
        audited.append(row)
        horizon_rows.extend(hrows)
        progress.update(i)
    progress.close()
    audit = _attach_reentry_flags(pd.DataFrame(audited))
    horizon = pd.DataFrame(horizon_rows)
    episode_audit = _episode_audit(episodes, audit, events)
    _write_report(out_dir, audit, horizon, episode_audit, legacy, args)
    return audit, horizon, episode_audit, legacy


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit, _horizon, episodes, legacy = run(args)
    print(f"[manual-audit] complete trades={len(audit)} legacy_unclosed={len(legacy)} episodes={len(episodes)}")
    print(f"[manual-audit] output={Path(args.out_dir).resolve()}")
    if len(audit):
        print(
            f"[manual-audit] win_rate={audit['is_win'].mean()*100:.1f}% "
            f"stop_first_then_target={int(audit['stop_first_then_target'].sum())} "
            f"premature_candidates={int(audit['premature_entry_candidate'].sum())}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
