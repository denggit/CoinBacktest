#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Range Footprint Failed Breakdown Reclaim V1.

Long-only order-flow prototype built for CoinBacktest.

Idea
----
Use tick-derived range bars and range-bar footprints to find sell-trap / failed
breakdown setups:

    1) Price breaks below a prior range-bar low.
    2) The breakdown bar contains a large sell footprint bucket near the bar low.
    3) Despite aggressive selling, the bar reclaims the breakdown level or closes
       high in its own range.
    4) A buy-stop is placed above the trap bar high. No same-bar retroactive entry.
    5) Stop is below the trap low. Target is fixed-R with fee/slippage-aware filters.

This is intentionally not an anchored volume profile strategy.  It borrows only
fast data-access patterns from the existing anchored-profile scripts: direct
SQLite footprint aggregation, shifted rolling thresholds, vectorized feature
construction, and a single state-machine scan.

Safety notes
------------
- No future function: all rolling thresholds and breakout levels are shifted by 1.
- No same-bar retroactive entry: signals created from bar i can only trigger from
  later bars through the pending buy-stop state machine.
- Fee-aware sizing: risk per unit includes estimated round-trip fee at stop.
- Conservative ambiguous-bar handling: if the entry bar also trades through the
  stop, the default is to skip that trade because intrabar order is unknown.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage  # noqa: E402
from src.backtest_common.reporting import build_report_trades, summarize_r_trades  # noqa: E402
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader  # noqa: E402
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

STRATEGY_NAME = "ETH_HF_RangeFootprint_FailedBreakdownReclaim_V1"
FILE_STEM = "eth_hf_range_footprint_failed_breakdown_reclaim_v1"


@dataclass
class Config:
    symbol: str = "ETH-USDT-SWAP"
    start_date: str = "2026-01-01"
    end_date: str = "2026-06-15"
    warmup_start_date: str = "2025-10-01"
    initial_capital: float = 1000.0

    # OKX fee model. Default is 0.055% per side so a full taker round-trip is 0.11%.
    fee_rate: float = 0.00055
    slippage_pct: float = 0.00015

    # Data.
    range_pct: float = 0.0020
    price_step: float = 1.0
    data_dir: str | None = None

    # Risk.
    unit_risk_per_trade: float = 0.0015
    max_notional_mult: float = 0.8
    target_r: float = 2.2
    min_net_reward_risk: float = 1.25
    min_target_distance_pct: float = 0.0055
    max_raw_stop_pct: float = 0.0060
    cooldown_bars: int = 20
    max_holding_bars: int = 120
    max_pending_bars: int = 40
    skip_ambiguous_entry_bar: bool = True

    # Failed-breakdown setup.
    structure_lookback_bars: int = 80
    min_structure_bars: int = 30
    breakdown_buffer_pct: float = 0.00025
    reclaim_buffer_pct: float = 0.00005
    min_close_pos: float = 0.58
    entry_buffer_pct: float = 0.00015
    stop_buffer_pct: float = 0.00045

    # Footprint / effort filters.
    footprint_quantile_window: int = 360
    max_sell_bucket_quantile: float = 0.90
    bar_sell_quantile: float = 0.80
    max_sell_bucket_low_steps: float = 4.0
    require_negative_delta: bool = True
    min_large_sell_share: float = 0.0

    out_dir: Path = Path("data/reports/hf/eth_hf_range_footprint_failed_breakdown_reclaim_v1")
    write_full_audit: bool = False


PRESETS: dict[str, dict[str, Any]] = {
    "stable": {
        "unit_risk_per_trade": 0.0010,
        "max_notional_mult": 0.6,
        "target_r": 2.4,
        "min_net_reward_risk": 1.35,
        "min_target_distance_pct": 0.0065,
        "cooldown_bars": 30,
    },
    "high": {
        "unit_risk_per_trade": 0.0015,
        "max_notional_mult": 0.8,
        "target_r": 2.2,
        "min_net_reward_risk": 1.25,
        "min_target_distance_pct": 0.0055,
        "cooldown_bars": 20,
    },
    "turbo": {
        "unit_risk_per_trade": 0.0020,
        "max_notional_mult": 1.0,
        "target_r": 2.0,
        "min_net_reward_risk": 1.20,
        "min_target_distance_pct": 0.0050,
        "cooldown_bars": 15,
    },
}


def _build_progress_marks(start: str, end: str, months: int = 1) -> list[np.datetime64]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    marks: list[np.datetime64] = []
    cur = start_ts + pd.DateOffset(months=months)
    while cur <= end_ts:
        marks.append(np.datetime64(cur.to_datetime64()))
        cur += pd.DateOffset(months=months)
    if not marks or marks[-1] < np.datetime64(end_ts.to_datetime64()):
        marks.append(np.datetime64(end_ts.to_datetime64()))
    return marks


def _format_progress_ts(ts: np.datetime64 | pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


class FootprintTrapStore:
    """SQLite-backed footprint access used only for performance.

    Loading the full footprint table for multi-year tests is slow and memory-heavy.
    This class asks SQLite for one row per range bar: the price bucket with the
    largest sell notional.  Signal logic still uses only historical rows and
    shifted rolling quantiles after this merge.
    """

    def __init__(self, cfg: Config):
        self.loader = OKXRangeFootprintLoader(
            symbol=cfg.symbol,
            range_pct=cfg.range_pct,
            price_step=cfg.price_step,
            data_dir=cfg.data_dir,
        )
        self.db_path = Path(self.loader.db_path)
        self.table_name = self.loader.table_name
        self._ensure_indexes()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    @staticmethod
    def _ts_text(value: Any) -> str:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _ensure_indexes(self) -> None:
        with self._connect() as conn:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_bar_id ON {self.table_name}(bar_id)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_end_ts ON {self.table_name}(end_ts)")
            conn.commit()

    def metadata(self, start_date: Any, end_date: Any) -> dict[str, Any]:
        sql = f"""
            SELECT COUNT(*) AS rows, MIN(end_ts) AS min_ts, MAX(end_ts) AS max_ts
            FROM {self.table_name}
            WHERE end_ts >= ? AND end_ts <= ?
        """
        with self._connect() as conn:
            row = conn.execute(sql, (self._ts_text(start_date), self._ts_text(end_date))).fetchone()
        return {"rows": int(row[0] or 0), "min_ts": row[1], "max_ts": row[2]}

    def load_max_sell_buckets(self, start_date: Any, end_date: Any) -> pd.DataFrame:
        sql = f"""
            SELECT bar_id,
                   price_bucket AS max_sell_bucket,
                   sell_notional AS max_bucket_sell_notional,
                   delta_notional AS max_bucket_delta_notional,
                   large_sell_notional AS max_bucket_large_sell_notional,
                   max_trade_notional AS max_bucket_trade_notional
            FROM (
                SELECT bar_id, price_bucket, sell_notional, delta_notional,
                       large_sell_notional, max_trade_notional,
                       ROW_NUMBER() OVER (
                           PARTITION BY bar_id
                           ORDER BY sell_notional DESC, price_bucket ASC
                       ) AS rn
                FROM {self.table_name}
                WHERE end_ts >= ? AND end_ts <= ? AND sell_notional > 0
            ) ranked
            WHERE rn = 1
            ORDER BY bar_id
        """
        with self._connect() as conn:
            return pd.read_sql_query(sql, conn, params=(self._ts_text(start_date), self._ts_text(end_date)))



def _normalize_loader_frame(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy().reset_index(drop=True)
    return out.sort_values(sort_cols).reset_index(drop=True)



def load_range_data(cfg: Config) -> tuple[pd.DataFrame, FootprintTrapStore, dict[str, Any]]:
    bars = OKXRangeBarLoader(
        symbol=cfg.symbol,
        range_pct=cfg.range_pct,
        data_dir=cfg.data_dir,
    ).load_local_data(start_date=cfg.warmup_start_date, end_date=cfg.end_date)
    if bars.empty:
        raise RuntimeError(
            "No range bar data loaded. Please prebuild range bars first, e.g. tools/prebuild_okx_range_all.py."
        )
    bars = _normalize_loader_frame(bars, ["end_ts", "bar_id"])
    bars["cvd_volume"] = bars["delta_volume"].cumsum()
    bars["cvd_notional"] = bars["delta_notional"].cumsum()

    fp_store = FootprintTrapStore(cfg)
    fp_meta = fp_store.metadata(cfg.warmup_start_date, cfg.end_date)
    if fp_meta["rows"] <= 0:
        raise RuntimeError(
            "No range footprint data loaded. Please prebuild range footprints first, e.g. tools/prebuild_okx_range_all.py."
        )
    return bars, fp_store, fp_meta



def add_features(bars: pd.DataFrame, fp_store: FootprintTrapStore, cfg: Config) -> pd.DataFrame:
    print(f"[{STRATEGY_NAME}][features] start | bars={len(bars):,} footprint_source=sqlite", flush=True)
    out = bars.copy().reset_index(drop=True)
    out["idx"] = np.arange(len(out), dtype=np.int64)

    print(f"[{STRATEGY_NAME}][features] loading max sell bucket per range bar...", flush=True)
    max_sell = fp_store.load_max_sell_buckets(cfg.warmup_start_date, cfg.end_date)
    if max_sell.empty:
        raise RuntimeError("No max sell footprint bucket rows loaded from SQLite.")
    out = out.merge(max_sell, on="bar_id", how="left")
    for col in [
        "max_sell_bucket",
        "max_bucket_sell_notional",
        "max_bucket_delta_notional",
        "max_bucket_large_sell_notional",
        "max_bucket_trade_notional",
    ]:
        if col not in out.columns:
            out[col] = np.nan if col == "max_sell_bucket" else 0.0
    out["max_sell_bucket"] = pd.to_numeric(out["max_sell_bucket"], errors="coerce")
    for col in ["max_bucket_sell_notional", "max_bucket_delta_notional", "max_bucket_large_sell_notional", "max_bucket_trade_notional"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)

    print(f"[{STRATEGY_NAME}][features] shifted rolling levels and quantiles...", flush=True)
    w = int(cfg.footprint_quantile_window)
    min_q = max(50, int(w * 0.25))
    out["prior_low"] = out["low"].rolling(cfg.structure_lookback_bars, min_periods=cfg.min_structure_bars).min().shift(1)
    out["prior_high"] = out["high"].rolling(cfg.structure_lookback_bars, min_periods=cfg.min_structure_bars).max().shift(1)
    out["max_sell_q"] = out["max_bucket_sell_notional"].rolling(w, min_periods=min_q).quantile(cfg.max_sell_bucket_quantile).shift(1)
    out["bar_sell_q"] = out["sell_notional"].rolling(w, min_periods=min_q).quantile(cfg.bar_sell_quantile).shift(1)
    out["range_height"] = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_pos"] = ((out["close"] - out["low"]) / out["range_height"]).clip(lower=0.0, upper=1.0)
    out["max_sell_bucket_dist_steps"] = (out["max_sell_bucket"] - out["low"]) / max(cfg.price_step, 1e-12)
    out["large_sell_share"] = out["max_bucket_large_sell_notional"] / out["max_bucket_sell_notional"].replace(0, np.nan)
    out["large_sell_share"] = out["large_sell_share"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # Fully historical signal ingredients; all thresholds/levels above are shifted by 1.
    out["breakdown_ok"] = out["low"] <= out["prior_low"] * (1.0 - cfg.breakdown_buffer_pct)
    out["reclaim_ok"] = (out["close"] >= out["prior_low"] * (1.0 + cfg.reclaim_buffer_pct)) | (out["close_pos"] >= cfg.min_close_pos)
    out["bucket_near_low_ok"] = out["max_sell_bucket_dist_steps"].between(0, cfg.max_sell_bucket_low_steps, inclusive="both")
    out["sell_bubble_ok"] = (out["max_bucket_sell_notional"] >= out["max_sell_q"]) & (out["sell_notional"] >= out["bar_sell_q"])
    out["negative_delta_ok"] = (out["delta_notional"] < 0) | (out["max_bucket_delta_notional"] < 0)
    out["large_sell_ok"] = out["large_sell_share"] >= cfg.min_large_sell_share
    out["raw_trap_setup"] = (
        out["breakdown_ok"]
        & out["reclaim_ok"]
        & out["bucket_near_low_ok"]
        & out["sell_bubble_ok"]
        & out["large_sell_ok"]
    )
    if cfg.require_negative_delta:
        out["raw_trap_setup"] &= out["negative_delta_ok"]

    print(f"[{STRATEGY_NAME}][features] finished | raw_trap_setup={int(out['raw_trap_setup'].sum()):,}", flush=True)
    return out



def _setup_passes_risk_filters(trigger: float, stop: float, cfg: Config) -> tuple[bool, dict[str, float]]:
    if not (np.isfinite(trigger) and np.isfinite(stop) and trigger > 0 and stop > 0):
        return False, {}
    raw_risk = trigger - stop
    if raw_risk <= 0:
        return False, {}
    raw_stop_pct = raw_risk / trigger
    if raw_stop_pct > cfg.max_raw_stop_pct:
        return False, {"raw_stop_pct": float(raw_stop_pct)}
    target = trigger + cfg.target_r * raw_risk
    target_distance_pct = (target - trigger) / trigger
    if target_distance_pct < cfg.min_target_distance_pct:
        return False, {"raw_stop_pct": float(raw_stop_pct), "target_distance_pct": float(target_distance_pct), "target": float(target)}

    # Net R filter approximates the expected stop/win after fees and slippage.
    roundtrip_cost = 2.0 * cfg.fee_rate + 2.0 * cfg.slippage_pct
    net_reward = (target - trigger) - trigger * roundtrip_cost
    net_risk = (trigger - stop) + trigger * roundtrip_cost
    net_rr = net_reward / net_risk if net_risk > 0 else np.nan
    if not (np.isfinite(net_rr) and net_rr >= cfg.min_net_reward_risk):
        return False, {
            "raw_stop_pct": float(raw_stop_pct),
            "target_distance_pct": float(target_distance_pct),
            "target": float(target),
            "net_reward_risk": float(net_rr) if np.isfinite(net_rr) else np.nan,
        }
    return True, {
        "target": float(target),
        "raw_stop_pct": float(raw_stop_pct),
        "target_distance_pct": float(target_distance_pct),
        "net_reward_risk": float(net_rr),
    }



def generate_signals(features: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = features.copy().reset_index(drop=True)
    n = len(b)
    if n == 0:
        raise RuntimeError("empty feature frame")

    end_ts_np = pd.to_datetime(b["end_ts"]).to_numpy(dtype="datetime64[ns]")
    start_ts64 = np.datetime64(pd.Timestamp(cfg.start_date).to_datetime64())
    start_candidates = np.flatnonzero(end_ts_np >= start_ts64)
    first_trade_i = int(start_candidates[0]) if len(start_candidates) else 0
    loop_start_i = max(0, first_trade_i - max(cfg.structure_lookback_bars + cfg.footprint_quantile_window + 50, 1000))

    print(
        f"[{STRATEGY_NAME}][signals] start scan | loop_start_i={loop_start_i:,} "
        f"loop_start_time={pd.Timestamp(end_ts_np[loop_start_i])} first_trade_i={first_trade_i:,} "
        f"first_trade_time={pd.Timestamp(end_ts_np[first_trade_i])}",
        flush=True,
    )

    open_a = b["open"].to_numpy(float)
    high_a = b["high"].to_numpy(float)
    low_a = b["low"].to_numpy(float)
    close_a = b["close"].to_numpy(float)
    bar_id_a = b["bar_id"].to_numpy(np.int64)
    raw_setup_a = b["raw_trap_setup"].to_numpy(bool)

    sig = np.zeros(n, dtype=np.int8)
    setup_id_arr = np.full(n, -1, dtype=np.int64)
    entry_trigger = np.full(n, np.nan)
    initial_stop = np.full(n, np.nan)
    target_price = np.full(n, np.nan)
    planned_net_rr = np.full(n, np.nan)
    target_distance_pct = np.full(n, np.nan)
    raw_stop_pct = np.full(n, np.nan)
    trap_low = np.full(n, np.nan)
    trap_high = np.full(n, np.nan)
    breakdown_level = np.full(n, np.nan)
    signal_reason = np.array([""] * n, dtype=object)

    events: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    setup_seq = 0
    cooldown_until = -1
    progress_marks = _build_progress_marks(cfg.start_date, cfg.end_date, months=1)
    progress_pos = 0

    def log(i: int, event: str, info: dict[str, Any]) -> None:
        row = {"event_time": pd.Timestamp(end_ts_np[i]), "bar_id": int(bar_id_a[i]), "event": event}
        row.update(info)
        events.append(row)

    for i in range(loop_start_i, n):
        if i > loop_start_i and (i - loop_start_i) % 25000 == 0:
            print(
                f"[{STRATEGY_NAME}][signals] bar_progress bar={i + 1:,}/{n:,} "
                f"current={pd.Timestamp(end_ts_np[i])} events={len(events):,} signals={int(sig.sum()):,} "
                f"pending={pending is not None}",
                flush=True,
            )
        while progress_pos < len(progress_marks) and end_ts_np[i] >= progress_marks[progress_pos]:
            print(
                f"[{STRATEGY_NAME}][signals] completed_to={_format_progress_ts(progress_marks[progress_pos])} "
                f"bar={i + 1:,}/{n:,} current={pd.Timestamp(end_ts_np[i])} "
                f"events={len(events):,} signals={int(sig.sum()):,} pending={pending is not None}",
                flush=True,
            )
            progress_pos += 1

        # Pending buy-stop state.  It can only trigger after the trap bar.
        if pending is not None and i > pending["active_from_i"]:
            if i - pending["active_from_i"] > cfg.max_pending_bars:
                log(i, "cancel_pending_timeout", pending)
                pending = None
                cooldown_until = i + cfg.cooldown_bars
            elif low_a[i] <= pending["stop"]:
                log(i, "cancel_pending_invalidated_before_entry", pending)
                pending = None
                cooldown_until = i + cfg.cooldown_bars
            elif high_a[i] >= pending["trigger"]:
                sig[i] = 1
                setup_id_arr[i] = pending["setup_id"]
                entry_trigger[i] = pending["trigger"]
                initial_stop[i] = pending["stop"]
                target_price[i] = pending["target"]
                planned_net_rr[i] = pending["net_reward_risk"]
                target_distance_pct[i] = pending["target_distance_pct"]
                raw_stop_pct[i] = pending["raw_stop_pct"]
                trap_low[i] = pending["trap_low"]
                trap_high[i] = pending["trap_high"]
                breakdown_level[i] = pending["breakdown_level"]
                signal_reason[i] = "failed_breakdown_sell_trap_reclaim"
                log(i, "entry_triggered", pending)
                pending = None
                cooldown_until = i + cfg.cooldown_bars
                continue

        if i < cooldown_until or pending is not None:
            continue

        if not raw_setup_a[i]:
            continue

        prior_low = float(b.loc[i, "prior_low"])
        if not np.isfinite(prior_low):
            continue
        trigger = float(high_a[i] * (1.0 + cfg.entry_buffer_pct))
        stop = float(low_a[i] * (1.0 - cfg.stop_buffer_pct))
        ok, risk_info = _setup_passes_risk_filters(trigger, stop, cfg)
        setup_info = {
            "trap_i": int(i),
            "trap_low": float(low_a[i]),
            "trap_high": float(high_a[i]),
            "breakdown_level": float(prior_low),
            "trigger": float(trigger),
            "stop": float(stop),
            "max_sell_bucket": float(b.loc[i, "max_sell_bucket"]) if np.isfinite(b.loc[i, "max_sell_bucket"]) else np.nan,
            "max_bucket_sell_notional": float(b.loc[i, "max_bucket_sell_notional"]),
            "max_sell_q": float(b.loc[i, "max_sell_q"]) if np.isfinite(b.loc[i, "max_sell_q"]) else np.nan,
            "close_pos": float(b.loc[i, "close_pos"]) if np.isfinite(b.loc[i, "close_pos"]) else np.nan,
            **risk_info,
        }
        if not ok:
            log(i, "reject_risk_filter", setup_info)
            cooldown_until = i + max(1, cfg.cooldown_bars // 2)
            continue

        setup_seq += 1
        pending = {
            **setup_info,
            "setup_id": int(setup_seq),
            "target": float(risk_info["target"]),
            "target_distance_pct": float(risk_info["target_distance_pct"]),
            "raw_stop_pct": float(risk_info["raw_stop_pct"]),
            "net_reward_risk": float(risk_info["net_reward_risk"]),
            "active_from_i": int(i),
        }
        log(i, "buy_stop_placed", pending)

    out = b.copy()
    out["signal"] = sig
    out["setup_id"] = setup_id_arr
    out["entry_trigger"] = entry_trigger
    out["initial_stop"] = initial_stop
    out["target_price"] = target_price
    out["planned_net_reward_risk"] = planned_net_rr
    out["target_distance_pct"] = target_distance_pct
    out["raw_stop_pct"] = raw_stop_pct
    out["trap_low"] = trap_low
    out["trap_high"] = trap_high
    out["breakdown_level"] = breakdown_level
    out["signal_reason"] = signal_reason
    return out, pd.DataFrame(events)



def run_backtest(signals: pd.DataFrame, cfg: Config) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    d = signals[(signals["end_ts"] >= pd.Timestamp(cfg.start_date)) & (signals["start_ts"] <= pd.Timestamp(cfg.end_date))].copy().reset_index(drop=True)
    if d.empty:
        raise RuntimeError("No bars inside backtest date range")

    n = len(d)
    open_a = d["open"].to_numpy(float)
    high_a = d["high"].to_numpy(float)
    low_a = d["low"].to_numpy(float)
    close_a = d["close"].to_numpy(float)
    ts_a = pd.to_datetime(d["end_ts"]).to_numpy(dtype="datetime64[ns]")
    sig_a = d["signal"].to_numpy(np.int8)
    trigger_a = d["entry_trigger"].to_numpy(float)
    stop_a = d["initial_stop"].to_numpy(float)
    target_a = d["target_price"].to_numpy(float)
    setup_id_a = d["setup_id"].to_numpy(np.int64)

    capital = float(cfg.initial_capital)
    peak = capital
    pos: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []
    skipped_ambiguous_entries = 0
    progress_marks = _build_progress_marks(cfg.start_date, cfg.end_date, months=1)
    progress_pos = 0

    for i in range(n):
        while progress_pos < len(progress_marks) and ts_a[i] >= progress_marks[progress_pos]:
            print(
                f"[{STRATEGY_NAME}][backtest] completed_to={_format_progress_ts(progress_marks[progress_pos])} "
                f"bar={i + 1:,}/{n:,} current={pd.Timestamp(ts_a[i])} trades={len(trades):,} "
                f"capital={capital:.2f} open_position={pos is not None}",
                flush=True,
            )
            progress_pos += 1

        ts = pd.Timestamp(ts_a[i])
        eq = capital if pos is None else capital + (close_a[i] - pos["entry"]) * pos["qty"]
        peak = max(peak, eq)
        equity_rows.append({"timestamp": ts, "equity": eq, "drawdown_pct": (peak - eq) / peak if peak > 0 else 0.0})

        if pos is not None:
            pos["mfe"] = max(pos["mfe"], high_a[i] - pos["entry"])
            pos["mae"] = max(pos["mae"], pos["entry"] - low_a[i])
            exit_price = None
            exit_reason = None
            # Conservative: if stop and target are both inside the same range bar, stop wins.
            if low_a[i] <= pos["stop"]:
                exit_price = pos["stop"]
                exit_reason = "stop_trap_low"
            elif high_a[i] >= pos["target"]:
                exit_price = pos["target"]
                exit_reason = "target_r"
            elif i - pos["entry_i"] >= cfg.max_holding_bars:
                exit_price = close_a[i]
                exit_reason = "time_bars"

            if exit_price is not None:
                fill = apply_exit_slippage(float(exit_price), 1, cfg.slippage_pct)
                exit_fee = abs(pos["qty"] * fill) * cfg.fee_rate
                fee = pos["entry_fee"] + exit_fee
                gross = (fill - pos["entry"]) * pos["qty"]
                pnl = gross - fee
                before = capital
                capital += pnl
                risk_dist = max(pos["risk_dist"], 1e-12)
                trades.append(
                    {
                        "entry_time": pos["entry_time"],
                        "exit_time": ts,
                        "type": "LONG",
                        "entry": pos["entry"],
                        "exit": fill,
                        "pnl": pnl,
                        "fee": fee,
                        "capital": capital,
                        "return_pct": pnl / max(before, 1e-12),
                        "mfe_r": pos["mfe"] / risk_dist,
                        "mae_r": pos["mae"] / risk_dist,
                        "holding_hours": (ts - pos["entry_time"]).total_seconds() / 3600.0,
                        "holding_bars": int(i - pos["entry_i"]),
                        "exit_reason": exit_reason,
                        "setup_id": pos["setup_id"],
                    }
                )
                pos = None
                continue

        if pos is None and sig_a[i] == 1:
            trigger = float(trigger_a[i])
            stop = float(stop_a[i])
            target = float(target_a[i])
            if not (np.isfinite(trigger) and np.isfinite(stop) and np.isfinite(target)):
                continue

            if cfg.skip_ambiguous_entry_bar and low_a[i] <= stop:
                skipped_ambiguous_entries += 1
                continue

            entry_raw = max(float(open_a[i]), trigger)
            entry = apply_entry_slippage(entry_raw, 1, cfg.slippage_pct)
            stop_fill_est = apply_exit_slippage(stop, 1, cfg.slippage_pct)
            target_fill_est = apply_exit_slippage(target, 1, cfg.slippage_pct)
            if not (entry > stop_fill_est and target_fill_est > entry):
                continue

            # Fee-aware position sizing: estimate loss if stop is hit.
            risk_per_unit = (entry - stop_fill_est) + entry * cfg.fee_rate + stop_fill_est * cfg.fee_rate
            if risk_per_unit <= 0:
                continue
            qty = capital * cfg.unit_risk_per_trade / risk_per_unit
            notional = abs(qty * entry)
            max_notional = capital * cfg.max_notional_mult
            if notional > max_notional:
                qty *= max_notional / notional
                notional = max_notional
            if qty <= 0 or notional <= 0:
                continue
            pos = {
                "entry": float(entry),
                "stop": float(stop),
                "target": float(target),
                "risk_dist": float(entry - stop_fill_est),
                "qty": float(qty),
                "entry_fee": float(notional * cfg.fee_rate),
                "entry_time": ts,
                "entry_i": int(i),
                "mfe": 0.0,
                "mae": 0.0,
                "setup_id": int(setup_id_a[i]),
            }

    if pos is not None:
        ts = pd.Timestamp(ts_a[-1])
        fill = apply_exit_slippage(float(close_a[-1]), 1, cfg.slippage_pct)
        fee = pos["entry_fee"] + abs(pos["qty"] * fill) * cfg.fee_rate
        gross = (fill - pos["entry"]) * pos["qty"]
        pnl = gross - fee
        before = capital
        capital += pnl
        risk_dist = max(pos["risk_dist"], 1e-12)
        trades.append(
            {
                "entry_time": pos["entry_time"],
                "exit_time": ts,
                "type": "LONG",
                "entry": pos["entry"],
                "exit": fill,
                "pnl": pnl,
                "fee": fee,
                "capital": capital,
                "return_pct": pnl / max(before, 1e-12),
                "mfe_r": pos["mfe"] / risk_dist,
                "mae_r": pos["mae"] / risk_dist,
                "holding_hours": (ts - pos["entry_time"]).total_seconds() / 3600.0,
                "holding_bars": int(n - 1 - pos["entry_i"]),
                "exit_reason": "final",
                "setup_id": pos["setup_id"],
            }
        )

    equity = pd.DataFrame(equity_rows).set_index("timestamp") if equity_rows else pd.DataFrame()
    summary = summarize_r_trades(trades, equity, cfg.initial_capital)
    summary.update(
        {
            "signal_count": int((d["signal"] == 1).sum()),
            "skipped_ambiguous_entries": int(skipped_ambiguous_entries),
            "range_pct": cfg.range_pct,
            "price_step": cfg.price_step,
            "fee_rate_per_side": cfg.fee_rate,
            "slippage_pct": cfg.slippage_pct,
            "unit_risk_per_trade": cfg.unit_risk_per_trade,
            "max_notional_mult": cfg.max_notional_mult,
            "target_r": cfg.target_r,
            "min_net_reward_risk": cfg.min_net_reward_risk,
            "min_target_distance_pct": cfg.min_target_distance_pct,
            "structure_lookback_bars": cfg.structure_lookback_bars,
            "footprint_quantile_window": cfg.footprint_quantile_window,
            "max_holding_bars": cfg.max_holding_bars,
        }
    )
    return trades, equity, summary



def write_outputs(signals: pd.DataFrame, events: pd.DataFrame, trades: list[dict[str, Any]], equity: pd.DataFrame, summary: dict[str, Any], cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(cfg.out_dir / f"{FILE_STEM}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(cfg.out_dir / f"{FILE_STEM}_equity.csv")
    events.to_csv(cfg.out_dir / f"{FILE_STEM}_setup_events.csv", index=False)
    if not events.empty:
        events["event"].value_counts().rename_axis("event").reset_index(name="count").to_csv(
            cfg.out_dir / f"{FILE_STEM}_event_counts.csv", index=False
        )

    sigs = signals[(signals["signal"] == 1) & (signals["end_ts"] >= pd.Timestamp(cfg.start_date))].copy()
    signal_cols = [
        "end_ts", "bar_id", "open", "high", "low", "close", "setup_id",
        "prior_low", "entry_trigger", "initial_stop", "target_price", "planned_net_reward_risk",
        "target_distance_pct", "raw_stop_pct", "trap_low", "trap_high", "breakdown_level",
        "max_sell_bucket", "max_bucket_sell_notional", "max_sell_q", "bar_sell_q", "close_pos",
        "max_sell_bucket_dist_steps", "signal_reason",
    ]
    sigs[[c for c in signal_cols if c in sigs.columns]].to_csv(cfg.out_dir / f"{FILE_STEM}_signal_audit.csv", index=False)
    if cfg.write_full_audit:
        signals.to_csv(cfg.out_dir / f"{FILE_STEM}_full_audit.csv", index=False)
    with (cfg.out_dir / f"{FILE_STEM}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)



def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 100)
    print(f"{STRATEGY_NAME} Backtest Summary")
    print("=" * 100)
    for k, v in summary.items():
        print(f"{k:>32}: {v}")
    print("-" * 100)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 100 + "\n")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=STRATEGY_NAME, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2026-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--warmup-start-date", default=None, help="Defaults to start-date minus --warmup-days.")
    p.add_argument("--warmup-days", type=int, default=120)
    p.add_argument("--preset", choices=sorted(PRESETS), default="high")
    p.add_argument("--range-pct", type=float, default=0.0020)
    p.add_argument("--price-step", type=float, default=1.0)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.00015)

    p.add_argument("--unit-risk-per-trade", type=float, default=None)
    p.add_argument("--max-notional-mult", type=float, default=None)
    p.add_argument("--target-r", type=float, default=None)
    p.add_argument("--min-net-reward-risk", type=float, default=None)
    p.add_argument("--min-target-distance-pct", type=float, default=None)
    p.add_argument("--max-raw-stop-pct", type=float, default=0.0060)
    p.add_argument("--max-holding-bars", type=int, default=120)
    p.add_argument("--max-pending-bars", type=int, default=40)
    p.add_argument("--cooldown-bars", type=int, default=None)
    p.add_argument("--allow-ambiguous-entry-bar", action="store_true")

    p.add_argument("--structure-lookback-bars", type=int, default=80)
    p.add_argument("--min-structure-bars", type=int, default=30)
    p.add_argument("--breakdown-buffer-pct", type=float, default=0.00025)
    p.add_argument("--reclaim-buffer-pct", type=float, default=0.00005)
    p.add_argument("--min-close-pos", type=float, default=0.58)
    p.add_argument("--entry-buffer-pct", type=float, default=0.00015)
    p.add_argument("--stop-buffer-pct", type=float, default=0.00045)

    p.add_argument("--footprint-quantile-window", type=int, default=360)
    p.add_argument("--max-sell-bucket-quantile", type=float, default=0.90)
    p.add_argument("--bar-sell-quantile", type=float, default=0.80)
    p.add_argument("--max-sell-bucket-low-steps", type=float, default=4.0)
    p.add_argument("--no-require-negative-delta", action="store_true")
    p.add_argument("--min-large-sell-share", type=float, default=0.0)
    p.add_argument("--write-full-audit", action="store_true")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    warmup_start = args.warmup_start_date
    if warmup_start is None:
        warmup_start = (pd.Timestamp(args.start_date) - pd.Timedelta(days=int(args.warmup_days))).strftime("%Y-%m-%d")

    cfg = Config(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        warmup_start_date=warmup_start,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        range_pct=args.range_pct,
        price_step=args.price_step,
        data_dir=args.data_dir,
        max_raw_stop_pct=args.max_raw_stop_pct,
        max_holding_bars=args.max_holding_bars,
        max_pending_bars=args.max_pending_bars,
        structure_lookback_bars=args.structure_lookback_bars,
        min_structure_bars=args.min_structure_bars,
        breakdown_buffer_pct=args.breakdown_buffer_pct,
        reclaim_buffer_pct=args.reclaim_buffer_pct,
        min_close_pos=args.min_close_pos,
        entry_buffer_pct=args.entry_buffer_pct,
        stop_buffer_pct=args.stop_buffer_pct,
        footprint_quantile_window=args.footprint_quantile_window,
        max_sell_bucket_quantile=args.max_sell_bucket_quantile,
        bar_sell_quantile=args.bar_sell_quantile,
        max_sell_bucket_low_steps=args.max_sell_bucket_low_steps,
        require_negative_delta=not args.no_require_negative_delta,
        min_large_sell_share=args.min_large_sell_share,
        skip_ambiguous_entry_bar=not args.allow_ambiguous_entry_bar,
        write_full_audit=args.write_full_audit,
    )
    for k, v in PRESETS[args.preset].items():
        setattr(cfg, k, v)
    for arg_name, cfg_name in [
        ("unit_risk_per_trade", "unit_risk_per_trade"),
        ("max_notional_mult", "max_notional_mult"),
        ("target_r", "target_r"),
        ("min_net_reward_risk", "min_net_reward_risk"),
        ("min_target_distance_pct", "min_target_distance_pct"),
        ("cooldown_bars", "cooldown_bars"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, cfg_name, value)

    if args.out_dir:
        cfg.out_dir = Path(args.out_dir)
    else:
        cfg.out_dir = Path(
            f"data/reports/hf/{FILE_STEM}/{args.preset}_r{int(cfg.range_pct * 10000):04d}_"
            f"step{cfg.price_step:g}_tr{cfg.target_r:g}_q{cfg.max_sell_bucket_quantile:g}"
        )

    bars, fp_store, fp_meta = load_range_data(cfg)
    print(f"Loaded range bars: {len(bars):,} | {bars['end_ts'].min()} -> {bars['end_ts'].max()}")
    print(f"Loaded footprints metadata: {fp_meta['rows']:,} | {fp_meta['min_ts']} -> {fp_meta['max_ts']} | source={fp_store.db_path}")

    features = add_features(bars, fp_store, cfg)
    signals, events = generate_signals(features, cfg)
    signal_count = int((signals["signal"] == 1).sum())
    print(f"Signals generated: {signal_count:,}")
    if not events.empty:
        print("Event counts:")
        print(events["event"].value_counts().to_string())

    trades, equity, summary = run_backtest(signals, cfg)
    write_outputs(signals, events, trades, equity, summary, cfg)
    print_summary(summary, cfg.out_dir)

    report_df = signals[(signals["end_ts"] >= pd.Timestamp(cfg.start_date)) & (signals["start_ts"] <= pd.Timestamp(cfg.end_date))].copy()
    report_df = report_df.set_index("end_ts", drop=False)
    final_capital = float(trades[-1]["capital"]) if trades else cfg.initial_capital
    total_days = max((report_df.index[-1] - report_df.index[0]).total_seconds() / 86400.0, 1.0) if len(report_df) > 1 else 1.0
    print_full_report(
        build_report_trades(trades),
        report_df,
        cfg.initial_capital,
        final_capital,
        STRATEGY_NAME,
        total_days,
        False,
        symbol=cfg.symbol,
        report_dir=cfg.out_dir,
    )


if __name__ == "__main__":
    main()
