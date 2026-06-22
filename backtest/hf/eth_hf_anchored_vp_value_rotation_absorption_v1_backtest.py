#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH HF Anchored VP Value Rotation Absorption V1.

Long-only range-bar + footprint strategy for CoinBacktest.

Concept
-------
This is not a simple failed-breakdown strategy.  It looks for a completed markdown
leg, freezes an anchored footprint volume profile, then waits for repeated sell
absorption between the lower HVN/POC and VAL.  Entry is only after value reclaim
and a buy-stop breakout; target is anchored VAH by default.

State chain:
    1) Confirm a swing-low after a meaningful down leg from a prior swing-high.
    2) Freeze anchored profile from the previous base to that confirmed swing-low.
    3) While price is around/below VAL, detect repeated sell-bubble absorption.
    4) Require close back above VAL before placing buy stop above absorption zone.
    5) Stop below lower HVN/POC zone; target VAH or POC.

No future function
------------------
- Swing points are confirmed only after pivot_lookback future bars have closed.
- Profiles end at the confirmed swing-low bar, not at the future final low.
- Rolling sell thresholds are shifted by one bar.
- Buy stops can trigger only after they are placed.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from bisect import bisect_left
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

STRATEGY_NAME = "ETH_HF_AnchoredVP_ValueRotationAbsorption_V1"
FILE_STEM = "eth_hf_anchored_vp_value_rotation_absorption_v1"


@dataclass
class Config:
    symbol: str = "ETH-USDT-SWAP"
    start_date: str = "2026-01-01"
    end_date: str = "2026-06-15"
    warmup_start_date: str = "2025-10-01"
    initial_capital: float = 1000.0

    # Fees. 0.055% per side approximates 0.11% round-trip taker cost.
    fee_rate: float = 0.00055
    slippage_pct: float = 0.00015

    # Data.
    range_pct: float = 0.0020
    price_step: float = 1.0
    data_dir: str | None = None

    # Risk.
    unit_risk_per_trade: float = 0.0015
    max_notional_mult: float = 0.8
    min_reward_risk: float = 1.60
    min_net_reward_risk: float = 1.25
    min_target_distance_pct: float = 0.0060
    max_raw_stop_pct: float = 0.0120
    cooldown_bars: int = 30
    max_holding_bars: int = 260
    max_active_setup_bars: int = 420
    max_pending_order_bars: int = 120
    skip_ambiguous_entry_bar: bool = True

    # Structure/profile.
    pivot_lookback: int = 2
    min_downmove_pct: float = 0.0060
    profile_padding_bars: int = 30
    max_profile_bars: int = 1400
    min_bars_between_profiles: int = 30
    value_area_pct: float = 0.70
    lower_poc_min_separation_pct: float = 0.0002
    lower_poc_zone_rel_vol: float = 0.30
    lower_poc_hvn_window_buckets: int = 2
    lower_poc_hvn_min_prominence: float = 1.15
    target_mode: str = "vah"  # vah | poc

    # Absorption.
    absorption_scan_back_bars: int = 260
    min_absorption_bubbles: int = 2
    bubble_quantile_window: int = 360
    bubble_sell_quantile: float = 0.90
    bubble_impact_quantile: float = 0.88
    max_absorption_zone_width_pct: float = 0.0080
    lower_poc_invalidate_buffer_pct: float = 0.0008
    val_reclaim_buffer_pct: float = 0.00010
    entry_buffer_pct: float = 0.00015
    stop_buffer_pct: float = 0.00035

    out_dir: Path = Path("data/reports/hf/eth_hf_anchored_vp_value_rotation_absorption_v1")
    write_full_audit: bool = False


PRESETS: dict[str, dict[str, Any]] = {
    "stable": {
        "unit_risk_per_trade": 0.0010,
        "max_notional_mult": 0.6,
        "min_reward_risk": 1.80,
        "min_net_reward_risk": 1.35,
        "min_target_distance_pct": 0.0070,
        "cooldown_bars": 40,
    },
    "high": {
        "unit_risk_per_trade": 0.0015,
        "max_notional_mult": 0.8,
        "min_reward_risk": 1.60,
        "min_net_reward_risk": 1.25,
        "min_target_distance_pct": 0.0060,
        "cooldown_bars": 30,
    },
    "turbo": {
        "unit_risk_per_trade": 0.0020,
        "max_notional_mult": 1.0,
        "min_reward_risk": 1.45,
        "min_net_reward_risk": 1.18,
        "min_target_distance_pct": 0.0055,
        "cooldown_bars": 20,
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


class FootprintProfileStore:
    """SQLite footprint access optimized for anchored profile backtests."""

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
                   large_sell_notional AS max_bucket_large_sell_notional
            FROM (
                SELECT bar_id, price_bucket, sell_notional, delta_notional, large_sell_notional,
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

    def compute_profile(self, start_bar_id: int, end_bar_id: int, cfg: Config) -> dict[str, float] | None:
        sql = f"""
            SELECT price_bucket,
                   SUM(volume) AS volume,
                   SUM(buy_notional) AS buy_notional,
                   SUM(sell_notional) AS sell_notional,
                   SUM(delta_notional) AS delta_notional,
                   SUM(large_sell_notional) AS large_sell_notional
            FROM {self.table_name}
            WHERE bar_id >= ? AND bar_id <= ?
            GROUP BY price_bucket
            ORDER BY price_bucket
        """
        with self._connect() as conn:
            prof = pd.read_sql_query(sql, conn, params=(int(start_bar_id), int(end_bar_id)))
        if prof.empty:
            return None
        va = _value_area(prof, cfg.value_area_pct)
        if va is None:
            return None
        lp = _lower_poc(prof, va["val"], cfg)
        if lp is None:
            return None
        return {**va, **lp, "profile_rows": int(len(prof))}



def _normalize_loader_frame(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.copy().reset_index(drop=True).sort_values(sort_cols).reset_index(drop=True)



def load_range_data(cfg: Config) -> tuple[pd.DataFrame, FootprintProfileStore, dict[str, Any]]:
    bars = OKXRangeBarLoader(symbol=cfg.symbol, range_pct=cfg.range_pct, data_dir=cfg.data_dir).load_local_data(
        start_date=cfg.warmup_start_date,
        end_date=cfg.end_date,
    )
    if bars.empty:
        raise RuntimeError("No range bar data loaded. Please prebuild range bars first.")
    bars = _normalize_loader_frame(bars, ["end_ts", "bar_id"])
    bars["cvd_volume"] = bars["delta_volume"].cumsum()
    bars["cvd_notional"] = bars["delta_notional"].cumsum()

    fp_store = FootprintProfileStore(cfg)
    fp_meta = fp_store.metadata(cfg.warmup_start_date, cfg.end_date)
    if fp_meta["rows"] <= 0:
        raise RuntimeError("No range footprint data loaded. Please prebuild range footprints first.")
    return bars, fp_store, fp_meta



def add_features(bars: pd.DataFrame, fp_store: FootprintProfileStore, cfg: Config) -> pd.DataFrame:
    print(f"[{STRATEGY_NAME}][features] start | bars={len(bars):,} footprint_source=sqlite", flush=True)
    out = bars.copy().reset_index(drop=True)
    out["idx"] = np.arange(len(out), dtype=np.int64)
    out["range_height"] = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_pos"] = ((out["close"] - out["low"]) / out["range_height"]).clip(lower=0.0, upper=1.0)

    print(f"[{STRATEGY_NAME}][features] max sell footprint bucket per range bar...", flush=True)
    max_sell = fp_store.load_max_sell_buckets(cfg.warmup_start_date, cfg.end_date)
    if max_sell.empty:
        raise RuntimeError("No max sell footprint buckets loaded.")
    out = out.merge(max_sell, on="bar_id", how="left")
    out["max_sell_bucket"] = pd.to_numeric(out["max_sell_bucket"], errors="coerce")
    for c in ["max_bucket_sell_notional", "max_bucket_delta_notional", "max_bucket_large_sell_notional"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    print(f"[{STRATEGY_NAME}][features] shifted rolling sell thresholds...", flush=True)
    w = int(cfg.bubble_quantile_window)
    min_q = max(60, int(w * 0.25))
    out["sell_q"] = out["max_bucket_sell_notional"].rolling(w, min_periods=min_q).quantile(cfg.bubble_sell_quantile).shift(1)
    price_drop = (out["close"].shift(1) - out["low"]).clip(lower=cfg.price_step)
    out["sell_impact"] = out["sell_notional"] / price_drop.replace(0, np.nan)
    out["sell_impact_q"] = out["sell_impact"].rolling(w, min_periods=min_q).quantile(cfg.bubble_impact_quantile).shift(1)
    print(f"[{STRATEGY_NAME}][features] finished", flush=True)
    return out



def _confirmed_pivots_at_i(high: np.ndarray, low: np.ndarray, i: int, k: int) -> tuple[int | None, int | None]:
    if i < 2 * k:
        return None, None
    c = i - k
    lo = i - 2 * k
    hi = i + 1
    ph = c if high[c] >= np.nanmax(high[lo:hi]) else None
    pl = c if low[c] <= np.nanmin(low[lo:hi]) else None
    return ph, pl



def _value_area(prof: pd.DataFrame, value_area_pct: float) -> dict[str, float] | None:
    if prof.empty or prof["volume"].sum() <= 0:
        return None
    p = prof.sort_values("price_bucket").reset_index(drop=True)
    prices = p["price_bucket"].to_numpy(float)
    vols = p["volume"].to_numpy(float)
    poc_i = int(np.argmax(vols))
    total = float(vols.sum())
    target = total * float(value_area_pct)
    lo = hi = poc_i
    cum = float(vols[poc_i])
    while cum < target and (lo > 0 or hi < len(vols) - 1):
        lv = vols[lo - 1] if lo > 0 else -1.0
        hv = vols[hi + 1] if hi < len(vols) - 1 else -1.0
        if hv >= lv:
            hi += 1
            cum += float(vols[hi])
        else:
            lo -= 1
            cum += float(vols[lo])
    return {
        "poc": float(prices[poc_i]),
        "vah": float(prices[hi]),
        "val": float(prices[lo]),
        "profile_low": float(prices[0]),
        "profile_high": float(prices[-1]),
        "profile_volume": total,
    }



def _lower_poc(prof: pd.DataFrame, val: float, cfg: Config) -> dict[str, float] | None:
    below = prof[prof["price_bucket"] < val * (1.0 - cfg.lower_poc_min_separation_pct)].copy()
    if below.empty:
        return None
    ordered = below.sort_values("price_bucket").reset_index(drop=True)
    prices = ordered["price_bucket"].to_numpy(float)
    vols = ordered["volume"].to_numpy(float)
    if len(prices) == 0:
        return None
    w = max(1, int(cfg.lower_poc_hvn_window_buckets))
    best_j: int | None = None
    best_score = -float("inf")
    best_prominence = np.nan
    for j0 in range(len(prices)):
        left = max(0, j0 - w)
        right = min(len(prices), j0 + w + 1)
        neigh = np.concatenate([vols[left:j0], vols[j0 + 1:right]])
        if len(neigh) == 0:
            continue
        neigh_mean = float(np.mean(neigh))
        neigh_max = float(np.max(neigh))
        if neigh_mean <= 0:
            continue
        prominence0 = float(vols[j0] / neigh_mean)
        if vols[j0] >= neigh_max and prominence0 >= cfg.lower_poc_hvn_min_prominence:
            score0 = float(vols[j0] * prominence0)
            if score0 > best_score:
                best_j = j0
                best_score = score0
                best_prominence = prominence0
    if best_j is None:
        return None
    j = int(best_j)
    lp = float(prices[j])
    lv = float(vols[j])
    lo = hi = j
    th = lv * cfg.lower_poc_zone_rel_vol
    while lo > 0 and vols[lo - 1] >= th:
        lo -= 1
    while hi < len(vols) - 1 and vols[hi + 1] >= th:
        hi += 1
    return {
        "lower_poc": lp,
        "lower_poc_volume": lv,
        "lower_poc_zone_low": float(prices[lo]),
        "lower_poc_zone_high": float(prices[hi] + cfg.price_step),
        "lower_poc_hvn_prominence": float(best_prominence) if np.isfinite(best_prominence) else np.nan,
        "lower_poc_hvn_score": float(best_score),
    }



def _find_markdown_profile_from_pivot_low(
    piv_highs: list[int],
    piv_lows: list[int],
    high: np.ndarray,
    low: np.ndarray,
    pivot_low_i: int,
    current_i: int,
    last_profile_i: int,
    cfg: Config,
) -> dict[str, int | float] | None:
    if current_i - last_profile_i < cfg.min_bars_between_profiles:
        return None
    if not piv_highs:
        return None
    # Use the highest confirmed pivot high before this pivot low within max_profile_bars.
    min_i = max(0, pivot_low_i - cfg.max_profile_bars)
    candidates = [h_i for h_i in piv_highs if min_i <= h_i < pivot_low_i]
    if not candidates:
        return None
    major_high_i = max(candidates, key=lambda x: high[x])
    if (high[major_high_i] - low[pivot_low_i]) / high[major_high_i] < cfg.min_downmove_pct:
        return None
    before_pos = bisect_left(piv_lows, major_high_i) - 1
    profile_start_i = piv_lows[before_pos] if before_pos >= 0 else max(0, major_high_i - cfg.profile_padding_bars)
    profile_start_i = max(0, profile_start_i - cfg.profile_padding_bars)
    if pivot_low_i - profile_start_i > cfg.max_profile_bars:
        profile_start_i = max(0, pivot_low_i - cfg.max_profile_bars)
    return {
        "major_high_i": int(major_high_i),
        "major_high_price": float(high[major_high_i]),
        "swing_low_i": int(pivot_low_i),
        "swing_low_price": float(low[pivot_low_i]),
        "profile_start_i": int(profile_start_i),
        "profile_end_i": int(pivot_low_i),
    }



def _cluster_absorption(bars_slice: pd.DataFrame, profile: dict[str, float], cfg: Config) -> dict[str, Any] | None:
    if bars_slice.empty:
        return None
    bucket = bars_slice["max_sell_bucket"]
    finite_bucket = np.isfinite(bucket.to_numpy(dtype=float, copy=False))
    in_region = (bucket >= profile["lower_poc_zone_low"]) & (bucket <= profile["val"])
    not_invalidated = bars_slice["low"] >= profile["lower_poc_zone_low"] * (1.0 - cfg.lower_poc_invalidate_buffer_pct)
    enough_sell = np.isfinite(bars_slice["sell_q"].to_numpy(dtype=float, copy=False)) & (bars_slice["max_bucket_sell_notional"] >= bars_slice["sell_q"])
    high_impact = np.isfinite(bars_slice["sell_impact_q"].to_numpy(dtype=float, copy=False)) & (bars_slice["sell_impact"] >= bars_slice["sell_impact_q"])
    negative_delta = (bars_slice["delta_notional"] < 0) | (bars_slice["max_bucket_delta_notional"] < 0)
    mask = finite_bucket & in_region.to_numpy(dtype=bool, copy=False) & not_invalidated.to_numpy(dtype=bool, copy=False)
    mask &= (enough_sell.to_numpy(dtype=bool, copy=False) | high_impact.to_numpy(dtype=bool, copy=False))
    mask &= negative_delta.to_numpy(dtype=bool, copy=False)
    bubbles = bars_slice.loc[mask]
    if len(bubbles) < cfg.min_absorption_bubbles:
        return None
    bucket_a = bubbles["max_sell_bucket"].to_numpy(dtype=float, copy=False)
    low_a = bubbles["low"].to_numpy(dtype=float, copy=False)
    open_a = bubbles["open"].to_numpy(dtype=float, copy=False)
    close_a = bubbles["close"].to_numpy(dtype=float, copy=False)
    zone_low_a = np.maximum(profile["lower_poc_zone_low"], np.minimum(low_a, bucket_a))
    zone_high_a = np.maximum.reduce([bucket_a + cfg.price_step, close_a, open_a])
    zone_low = float(np.min(zone_low_a))
    zone_high = float(np.max(zone_high_a))
    if zone_low <= 0 or (zone_high - zone_low) / zone_low > cfg.max_absorption_zone_width_pct:
        return None
    return {
        "bubble_count": int(len(bubbles)),
        "bubble_bar_ids": [int(x) for x in bubbles["bar_id"].to_list()],
        "absorption_zone_low": float(zone_low),
        "absorption_zone_high": float(zone_high),
        "bubble_total_sell_notional": float(bubbles["max_bucket_sell_notional"].sum()),
    }



def _risk_ok(trigger: float, stop: float, target: float, cfg: Config) -> tuple[bool, dict[str, float]]:
    if not (np.isfinite(trigger) and np.isfinite(stop) and np.isfinite(target)):
        return False, {}
    risk = trigger - stop
    reward = target - trigger
    if trigger <= 0 or stop <= 0 or risk <= 0 or reward <= 0:
        return False, {}
    raw_stop_pct = risk / trigger
    target_distance_pct = reward / trigger
    rr = reward / risk
    if raw_stop_pct > cfg.max_raw_stop_pct or rr < cfg.min_reward_risk or target_distance_pct < cfg.min_target_distance_pct:
        return False, {"raw_stop_pct": raw_stop_pct, "reward_risk": rr, "target_distance_pct": target_distance_pct}
    roundtrip_cost = 2.0 * cfg.fee_rate + 2.0 * cfg.slippage_pct
    net_reward = reward - trigger * roundtrip_cost
    net_risk = risk + trigger * roundtrip_cost
    net_rr = net_reward / net_risk if net_risk > 0 else np.nan
    if not (np.isfinite(net_rr) and net_rr >= cfg.min_net_reward_risk):
        return False, {"raw_stop_pct": raw_stop_pct, "reward_risk": rr, "target_distance_pct": target_distance_pct, "net_reward_risk": net_rr}
    return True, {"raw_stop_pct": raw_stop_pct, "reward_risk": rr, "target_distance_pct": target_distance_pct, "net_reward_risk": net_rr}



def generate_signals(bars: pd.DataFrame, fp_store: FootprintProfileStore, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = add_features(bars, fp_store, cfg).reset_index(drop=True)
    n = len(b)
    high = b["high"].to_numpy(float)
    low = b["low"].to_numpy(float)
    close = b["close"].to_numpy(float)
    end_ts_np = pd.to_datetime(b["end_ts"]).to_numpy(dtype="datetime64[ns]")
    bar_ids = b["bar_id"].to_numpy(np.int64)

    start_ts64 = np.datetime64(pd.Timestamp(cfg.start_date).to_datetime64())
    start_candidates = np.flatnonzero(end_ts_np >= start_ts64)
    first_trade_i = int(start_candidates[0]) if len(start_candidates) else 0
    warmup_scan_bars = max(cfg.max_profile_bars + cfg.absorption_scan_back_bars + cfg.bubble_quantile_window + 50, 2500)
    loop_start_i = max(0, first_trade_i - warmup_scan_bars)

    print(
        f"[{STRATEGY_NAME}][signals] start scan | loop_start_i={loop_start_i:,} "
        f"loop_start_time={pd.Timestamp(end_ts_np[loop_start_i])} first_trade_i={first_trade_i:,} "
        f"first_trade_time={pd.Timestamp(end_ts_np[first_trade_i])}",
        flush=True,
    )

    sig = np.zeros(n, dtype=np.int8)
    setup_id_arr = np.full(n, -1, dtype=int)
    entry_trigger = np.full(n, np.nan)
    initial_stop = np.full(n, np.nan)
    target_price = np.full(n, np.nan)
    planned_reward_risk = np.full(n, np.nan)
    planned_net_rr = np.full(n, np.nan)
    target_distance_pct = np.full(n, np.nan)
    profile_vah = np.full(n, np.nan)
    profile_val = np.full(n, np.nan)
    profile_poc = np.full(n, np.nan)
    lower_poc = np.full(n, np.nan)
    absorption_low = np.full(n, np.nan)
    absorption_high = np.full(n, np.nan)
    bubble_count = np.zeros(n, dtype=int)
    signal_reason = np.array([""] * n, dtype=object)

    events: list[dict[str, Any]] = []
    piv_highs: list[int] = []
    piv_lows: list[int] = []
    last_profile_i = -10**9
    setup_seq = 0
    active: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    cooldown_until = -1
    profile_cache: dict[tuple[int, int], dict[str, float] | None] = {}
    progress_marks = _build_progress_marks(cfg.start_date, cfg.end_date, months=1)
    progress_pos = 0

    def log(i: int, event: str, info: dict[str, Any]) -> None:
        row = {"event_time": pd.Timestamp(end_ts_np[i]), "bar_id": int(bar_ids[i]), "event": event}
        row.update(info)
        events.append(row)

    def prepare_order(i: int, setup: dict[str, Any], cluster: dict[str, Any]) -> dict[str, Any] | None:
        nonlocal setup_seq
        trigger = float(max(cluster["absorption_zone_high"], setup["val"]) * (1.0 + cfg.entry_buffer_pct))
        stop = float(setup["lower_poc_zone_low"] * (1.0 - cfg.stop_buffer_pct))
        target = float(setup["poc"] if cfg.target_mode == "poc" else setup["vah"])
        ok, risk_info = _risk_ok(trigger, stop, target, cfg)
        base = {**setup, **cluster, "entry_trigger": trigger, "initial_stop": stop, "target_price": target, **risk_info}
        if not ok:
            log(i, "reject_order_risk_filter", base)
            return None
        setup_seq += 1
        pending_order = {**base, "setup_id": int(setup_seq), "active_from_i": int(i)}
        log(i, "buy_stop_placed", pending_order)
        return pending_order

    for i in range(loop_start_i, n):
        if i > loop_start_i and (i - loop_start_i) % 25000 == 0:
            print(
                f"[{STRATEGY_NAME}][signals] bar_progress bar={i + 1:,}/{n:,} current={pd.Timestamp(end_ts_np[i])} "
                f"events={len(events):,} signals={int(sig.sum()):,} active={active is not None} pending={pending is not None}",
                flush=True,
            )
        while progress_pos < len(progress_marks) and end_ts_np[i] >= progress_marks[progress_pos]:
            print(
                f"[{STRATEGY_NAME}][signals] completed_to={_format_progress_ts(progress_marks[progress_pos])} "
                f"bar={i + 1:,}/{n:,} current={pd.Timestamp(end_ts_np[i])} events={len(events):,} "
                f"signals={int(sig.sum()):,} active={active is not None} pending={pending is not None}",
                flush=True,
            )
            progress_pos += 1

        ph, pl = _confirmed_pivots_at_i(high, low, i, cfg.pivot_lookback)
        if ph is not None:
            piv_highs.append(ph)
        if pl is not None:
            piv_lows.append(pl)

        if pending is not None and i > pending["active_from_i"]:
            if i - pending["active_from_i"] > cfg.max_pending_order_bars:
                log(i, "cancel_pending_timeout", pending)
                pending = None
                active = None
                cooldown_until = i + cfg.cooldown_bars
            elif low[i] < pending["lower_poc_zone_low"] * (1.0 - cfg.lower_poc_invalidate_buffer_pct):
                log(i, "cancel_pending_break_lower_poc", pending)
                pending = None
                active = None
                cooldown_until = i + cfg.cooldown_bars
            elif high[i] >= pending["entry_trigger"]:
                sig[i] = 1
                setup_id_arr[i] = pending["setup_id"]
                entry_trigger[i] = pending["entry_trigger"]
                initial_stop[i] = pending["initial_stop"]
                target_price[i] = pending["target_price"]
                planned_reward_risk[i] = pending.get("reward_risk", np.nan)
                planned_net_rr[i] = pending.get("net_reward_risk", np.nan)
                target_distance_pct[i] = pending.get("target_distance_pct", np.nan)
                profile_vah[i] = pending["vah"]
                profile_val[i] = pending["val"]
                profile_poc[i] = pending["poc"]
                lower_poc[i] = pending["lower_poc"]
                absorption_low[i] = pending["absorption_zone_low"]
                absorption_high[i] = pending["absorption_zone_high"]
                bubble_count[i] = pending["bubble_count"]
                signal_reason[i] = "anchored_vp_absorption_val_reclaim_rotation"
                log(i, "entry_triggered", pending)
                pending = None
                active = None
                cooldown_until = i + cfg.cooldown_bars
                continue

        if i < cooldown_until:
            continue

        if active is not None:
            if i - active["created_i"] > cfg.max_active_setup_bars:
                log(i, "cancel_active_timeout", active)
                active = None
                cooldown_until = i + cfg.cooldown_bars
            elif low[i] < active["lower_poc_zone_low"] * (1.0 - cfg.lower_poc_invalidate_buffer_pct):
                log(i, "cancel_active_break_lower_poc", active)
                active = None
                cooldown_until = i + cfg.cooldown_bars
            elif pending is None:
                scan_start = max(active["created_i"], i - cfg.absorption_scan_back_bars)
                cluster = _cluster_absorption(b.iloc[scan_start : i + 1], active, cfg)
                value_reclaimed = close[i] >= active["val"] * (1.0 + cfg.val_reclaim_buffer_pct)
                if cluster is not None and value_reclaimed:
                    pending = prepare_order(i, active, cluster)
                    if pending is None:
                        active = None
                        cooldown_until = i + max(1, cfg.cooldown_bars // 2)

        if active is not None or pending is not None or i < cooldown_until:
            continue

        if pl is None:
            continue
        md = _find_markdown_profile_from_pivot_low(piv_highs, piv_lows, high, low, pl, i, last_profile_i, cfg)
        if md is None:
            continue
        last_profile_i = i
        start_i = int(md["profile_start_i"])
        end_i = int(md["profile_end_i"])
        key = (int(bar_ids[start_i]), int(bar_ids[end_i]))
        if key not in profile_cache:
            profile_cache[key] = fp_store.compute_profile(key[0], key[1], cfg)
        prof = profile_cache[key]
        base_info = {**md, "profile_start_bar_id": key[0], "profile_end_bar_id": key[1]}
        if prof is None:
            log(i, "reject_profile_no_lower_poc_or_empty", base_info)
            continue
        active = {**base_info, **prof, "created_i": int(i)}
        log(i, "profile_frozen_wait_absorption", active)

    out = b.copy()
    out["signal"] = sig
    out["setup_id"] = setup_id_arr
    out["entry_trigger"] = entry_trigger
    out["initial_stop"] = initial_stop
    out["target_price"] = target_price
    out["planned_reward_risk"] = planned_reward_risk
    out["planned_net_reward_risk"] = planned_net_rr
    out["target_distance_pct"] = target_distance_pct
    out["profile_vah"] = profile_vah
    out["profile_val"] = profile_val
    out["profile_poc"] = profile_poc
    out["lower_poc"] = lower_poc
    out["absorption_low"] = absorption_low
    out["absorption_high"] = absorption_high
    out["bubble_count"] = bubble_count
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
    sig = d["signal"].to_numpy(int)
    trigger = d["entry_trigger"].to_numpy(float)
    stop_a = d["initial_stop"].to_numpy(float)
    target_a = d["target_price"].to_numpy(float)
    setup_id = d["setup_id"].to_numpy(int)

    capital = cfg.initial_capital
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
            # Conservative: stop before target if both touched inside same range bar.
            if low_a[i] <= pos["stop"]:
                exit_price = pos["stop"]
                exit_reason = "stop_lower_poc"
            elif high_a[i] >= pos["target"]:
                exit_price = pos["target"]
                exit_reason = f"target_{cfg.target_mode}"
            elif i - pos["entry_i"] >= cfg.max_holding_bars:
                exit_price = close_a[i]
                exit_reason = "time_bars"
            if exit_price is not None:
                fill = apply_exit_slippage(float(exit_price), 1, cfg.slippage_pct)
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
                        "holding_bars": int(i - pos["entry_i"]),
                        "exit_reason": exit_reason,
                        "setup_id": pos["setup_id"],
                    }
                )
                pos = None
                continue

        if pos is None and sig[i] == 1:
            if not (np.isfinite(trigger[i]) and np.isfinite(stop_a[i]) and np.isfinite(target_a[i])):
                continue
            if cfg.skip_ambiguous_entry_bar and low_a[i] <= stop_a[i]:
                skipped_ambiguous_entries += 1
                continue
            entry_raw = max(open_a[i], trigger[i])
            entry = apply_entry_slippage(float(entry_raw), 1, cfg.slippage_pct)
            stop_fill_est = apply_exit_slippage(float(stop_a[i]), 1, cfg.slippage_pct)
            target_fill_est = apply_exit_slippage(float(target_a[i]), 1, cfg.slippage_pct)
            if not (entry > stop_fill_est and target_fill_est > entry):
                continue
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
                "stop": float(stop_a[i]),
                "target": float(target_a[i]),
                "risk_dist": float(entry - stop_fill_est),
                "qty": float(qty),
                "entry_fee": float(notional * cfg.fee_rate),
                "entry_time": ts,
                "entry_i": i,
                "mfe": 0.0,
                "mae": 0.0,
                "setup_id": int(setup_id[i]),
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
            "target_mode": cfg.target_mode,
            "min_reward_risk": cfg.min_reward_risk,
            "min_net_reward_risk": cfg.min_net_reward_risk,
            "min_target_distance_pct": cfg.min_target_distance_pct,
            "max_holding_bars": cfg.max_holding_bars,
            "max_profile_bars": cfg.max_profile_bars,
            "min_absorption_bubbles": cfg.min_absorption_bubbles,
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
        events["event"].value_counts().rename_axis("event").reset_index(name="count").to_csv(cfg.out_dir / f"{FILE_STEM}_event_counts.csv", index=False)
    sigs = signals[(signals["signal"] == 1) & (signals["end_ts"] >= pd.Timestamp(cfg.start_date))].copy()
    cols = [
        "end_ts", "bar_id", "open", "high", "low", "close", "setup_id", "entry_trigger", "initial_stop", "target_price",
        "planned_reward_risk", "planned_net_reward_risk", "target_distance_pct", "profile_vah", "profile_val", "profile_poc",
        "lower_poc", "absorption_low", "absorption_high", "bubble_count", "signal_reason",
    ]
    sigs[[c for c in cols if c in sigs.columns]].to_csv(cfg.out_dir / f"{FILE_STEM}_signal_audit.csv", index=False)
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
    p.add_argument("--warmup-days", type=int, default=180)
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
    p.add_argument("--min-reward-risk", type=float, default=None)
    p.add_argument("--min-net-reward-risk", type=float, default=None)
    p.add_argument("--min-target-distance-pct", type=float, default=None)
    p.add_argument("--target-mode", choices=["vah", "poc"], default="vah")
    p.add_argument("--max-holding-bars", type=int, default=260)
    p.add_argument("--max-active-setup-bars", type=int, default=420)
    p.add_argument("--max-pending-order-bars", type=int, default=120)
    p.add_argument("--allow-ambiguous-entry-bar", action="store_true")

    p.add_argument("--pivot-lookback", type=int, default=2)
    p.add_argument("--min-downmove-pct", type=float, default=0.0060)
    p.add_argument("--max-profile-bars", type=int, default=1400)
    p.add_argument("--min-bars-between-profiles", type=int, default=30)
    p.add_argument("--min-absorption-bubbles", type=int, default=2)
    p.add_argument("--bubble-quantile-window", type=int, default=360)
    p.add_argument("--bubble-sell-quantile", type=float, default=0.90)
    p.add_argument("--bubble-impact-quantile", type=float, default=0.88)
    p.add_argument("--max-absorption-zone-width-pct", type=float, default=0.0080)
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
        target_mode=args.target_mode,
        max_holding_bars=args.max_holding_bars,
        max_active_setup_bars=args.max_active_setup_bars,
        max_pending_order_bars=args.max_pending_order_bars,
        skip_ambiguous_entry_bar=not args.allow_ambiguous_entry_bar,
        pivot_lookback=args.pivot_lookback,
        min_downmove_pct=args.min_downmove_pct,
        max_profile_bars=args.max_profile_bars,
        min_bars_between_profiles=args.min_bars_between_profiles,
        min_absorption_bubbles=args.min_absorption_bubbles,
        bubble_quantile_window=args.bubble_quantile_window,
        bubble_sell_quantile=args.bubble_sell_quantile,
        bubble_impact_quantile=args.bubble_impact_quantile,
        max_absorption_zone_width_pct=args.max_absorption_zone_width_pct,
        write_full_audit=args.write_full_audit,
    )
    for k, v in PRESETS[args.preset].items():
        setattr(cfg, k, v)
    for arg_name, cfg_name in [
        ("unit_risk_per_trade", "unit_risk_per_trade"),
        ("max_notional_mult", "max_notional_mult"),
        ("min_reward_risk", "min_reward_risk"),
        ("min_net_reward_risk", "min_net_reward_risk"),
        ("min_target_distance_pct", "min_target_distance_pct"),
    ]:
        value = getattr(args, arg_name)
        if value is not None:
            setattr(cfg, cfg_name, value)
    if args.out_dir:
        cfg.out_dir = Path(args.out_dir)
    else:
        cfg.out_dir = Path(
            f"data/reports/hf/{FILE_STEM}/{args.preset}_r{int(cfg.range_pct * 10000):04d}_"
            f"step{cfg.price_step:g}_{cfg.target_mode}_mb{cfg.min_absorption_bubbles}"
        )

    bars, fp_store, fp_meta = load_range_data(cfg)
    print(f"Loaded range bars: {len(bars):,} | {bars['end_ts'].min()} -> {bars['end_ts'].max()}")
    print(f"Loaded footprints metadata: {fp_meta['rows']:,} | {fp_meta['min_ts']} -> {fp_meta['max_ts']} | source={fp_store.db_path}")
    signals, events = generate_signals(bars, fp_store, cfg)
    print(f"Signals generated: {int((signals['signal'] == 1).sum()):,}")
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
