#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Range Anchored Volume Profile Absorption Breakout V9.

Fabio-style long-only prototype, rewritten chain:
    1) Detect completed downtrend structure and lower-high break.
    2) Only after lower-high break, build/freeze anchored volume profile.
    3) Profile spans prior swing-low/base -> major swing-high -> current lower-high-break bar.
    4) Look for repeated sell-bubble absorption anywhere between VAL and lower POC.
    5) Buy-stop above concentrated absorption zone; stop below lower POC; target anchored VAH.

No dynamic stop in V9. Dynamic structural trailing can be added only after this basic
state machine is validated.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage
from src.backtest_common.reporting import build_report_trades, summarize_r_trades
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader
from src.utils.report import print_full_report

STRATEGY_NAME = "ETH_Range_AnchoredVP_AbsorptionBreakout_V9"


def _build_progress_marks(start: str, end: str, months: int = 1) -> list[np.datetime64]:
    """Return calendar progress markers for long local backtests."""
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


@dataclass
class Config:
    symbol: str = "ETH-USDT-SWAP"
    start_date: str = "2023-01-01"
    end_date: str = "2026-06-15"
    warmup_start_date: str = "2022-01-01"
    initial_capital: float = 1000.0
    fee_rate: float = 0.00055
    slippage_pct: float = 0.00015
    range_pct: float = 0.0020
    price_step: float = 1.0
    data_dir: str | None = None

    unit_risk_per_trade: float = 0.0020
    max_notional_mult: float = 1.0

    # Market structure.
    pivot_lookback: int = 2
    min_downmove_pct: float = 0.004
    profile_padding_bars: int = 30
    max_profile_bars: int = 1200
    min_bars_between_lh_breaks: int = 20

    # Profile.
    value_area_pct: float = 0.70
    lower_poc_min_separation_pct: float = 0.0002
    lower_poc_zone_rel_vol: float = 0.30
    lower_poc_mode: str = "hvn"  # max_volume | hvn
    lower_poc_hvn_window_buckets: int = 2
    lower_poc_hvn_min_prominence: float = 1.15
    target_mode: str = "vah"  # vah | poc
    rr_filter_target: str = "selected"  # selected | vah

    # Bubble / absorption.
    absorption_scan_back_bars: int = 320
    max_active_setup_bars: int = 360
    max_pending_order_bars: int = 160
    min_absorption_bubbles: int = 2
    bubble_quantile_window: int = 300
    bubble_sell_quantile: float = 0.88
    bubble_impact_quantile: float = 0.88
    max_absorption_zone_width_pct: float = 0.010
    # V9: test whether absorption closer to VAL is stronger than lower-band absorption.
    # Band position: 0 = lower_poc_zone_low, 1 = VAL.
    absorption_location_mode: str = "upper_band"  # any | lower_band | upper_band
    absorption_max_band_position: float = 0.45
    absorption_min_band_position: float = 0.55
    lower_poc_invalidate_buffer_pct: float = 0.0008

    # Entry/exit.
    entry_buffer_pct: float = 0.00015
    stop_buffer_pct: float = 0.0003
    min_reward_risk: float = 2.0
    min_target_distance_pct: float = 0.015
    max_holding_bars: int = 260  # kept for compatibility; V8 uses max_holding_hours.
    max_holding_hours: float = 48.0
    cooldown_bars: int = 15

    out_dir: Path = Path("data/reports/mf/eth_range_anchored_vp_absorption_v9")
    write_full_audit: bool = False


PRESETS = {
    "stable": {"unit_risk_per_trade": 0.0015, "max_notional_mult": 0.8},
    "high": {"unit_risk_per_trade": 0.0020, "max_notional_mult": 1.0},
    "turbo": {"unit_risk_per_trade": 0.0030, "max_notional_mult": 1.3},
}


class FootprintStore:
    """SQLite-backed footprint access for fast, low-memory profile queries.

    The strategy still queries only explicit historical windows from the state
    machine. This class changes storage access, not signal logic.
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
            SELECT bar_id, price_bucket AS max_sell_bucket,
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
        return df
    out = df.copy().reset_index(drop=True)
    return out.sort_values(sort_cols).copy()


def _prepare_footprint_index(fps: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Sort footprint rows once and keep bar_id array for fast anchored-window slicing.

    This is a pure performance helper. It preserves the exact same rows that the
    original boolean mask selected, but avoids scanning the full footprint table
    for every candidate anchored profile.
    """
    if fps is None or fps.empty:
        return fps, np.array([], dtype=np.int64)
    out = fps.sort_values(["bar_id", "price_bucket"]).reset_index(drop=True)
    return out, out["bar_id"].to_numpy(dtype=np.int64, copy=False)


def _slice_footprint_by_bar_id(
    fps_sorted: pd.DataFrame,
    fp_bar_ids: np.ndarray,
    start_bar_id: int,
    end_bar_id: int,
) -> pd.DataFrame:
    if fps_sorted is None or fps_sorted.empty:
        return fps_sorted
    lo = int(np.searchsorted(fp_bar_ids, int(start_bar_id), side="left"))
    hi = int(np.searchsorted(fp_bar_ids, int(end_bar_id), side="right"))
    return fps_sorted.iloc[lo:hi]


def load_range_data(cfg: Config) -> tuple[pd.DataFrame, FootprintStore, dict[str, Any]]:
    bars = OKXRangeBarLoader(symbol=cfg.symbol, range_pct=cfg.range_pct, data_dir=cfg.data_dir).load_local_data(
        start_date=cfg.warmup_start_date,
        end_date=cfg.end_date,
    )
    if bars.empty:
        raise RuntimeError("No range bar data loaded. Please prebuild range bars first.")
    bars = _normalize_loader_frame(bars, ["end_ts", "bar_id"])
    bars["cvd_volume"] = bars["delta_volume"].cumsum()
    bars["cvd_notional"] = bars["delta_notional"].cumsum()

    fp_store = FootprintStore(cfg)
    fp_meta = fp_store.metadata(cfg.warmup_start_date, cfg.end_date)
    if fp_meta["rows"] <= 0:
        raise RuntimeError("No range footprint data loaded. Please prebuild range footprints first.")
    return bars, fp_store, fp_meta


def add_features(bars: pd.DataFrame, fp_store: FootprintStore, cfg: Config) -> pd.DataFrame:
    print(f"[{STRATEGY_NAME}][features] start add_features | bars={len(bars):,} footprint_source=sqlite", flush=True)
    out = bars.copy().reset_index(drop=True)
    out["idx"] = np.arange(len(out))

    print(f"[{STRATEGY_NAME}][features] rolling sell quantiles...", flush=True)
    out["sell_q"] = out["sell_notional"].rolling(cfg.bubble_quantile_window, min_periods=80).quantile(cfg.bubble_sell_quantile).shift(1)
    price_drop = (out["close"].shift(1) - out["low"]).clip(lower=cfg.price_step)
    out["sell_impact"] = out["sell_notional"] / price_drop.replace(0, np.nan)
    out["sell_impact_q"] = out["sell_impact"].rolling(cfg.bubble_quantile_window, min_periods=80).quantile(cfg.bubble_impact_quantile).shift(1)

    print(f"[{STRATEGY_NAME}][features] max sell footprint bucket per range bar from sqlite...", flush=True)
    max_sell = fp_store.load_max_sell_buckets(cfg.warmup_start_date, cfg.end_date)
    if max_sell.empty:
        out["max_sell_bucket"] = np.nan
        out["max_bucket_sell_notional"] = 0.0
        out["max_bucket_delta_notional"] = 0.0
        out["max_bucket_large_sell_notional"] = 0.0
        return out

    out = out.merge(max_sell, on="bar_id", how="left")
    for c in ["max_bucket_sell_notional", "max_bucket_delta_notional", "max_bucket_large_sell_notional"]:
        out[c] = out[c].fillna(0.0)
    print(f"[{STRATEGY_NAME}][features] finished add_features", flush=True)
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
    target = total * value_area_pct
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
    """Find lower POC below VAL.

    V8 change: default mode is local HVN instead of blindly taking the maximum
    single bucket below VAL. This tries to match the yellow-box small POC idea:
    a local accepted volume node, not a random isolated max bucket.
    """
    below = prof[prof["price_bucket"] < val * (1.0 - cfg.lower_poc_min_separation_pct)].copy()
    if below.empty:
        return None
    ordered = below.sort_values("price_bucket").reset_index(drop=True)
    prices = ordered["price_bucket"].to_numpy(float)
    vols = ordered["volume"].to_numpy(float)
    if len(prices) == 0:
        return None

    if cfg.lower_poc_mode == "max_volume":
        j = int(np.argmax(vols))
        prominence = np.nan
        score = float(vols[j])
    else:
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
            is_local_hvn = vols[j0] >= neigh_max and prominence0 >= cfg.lower_poc_hvn_min_prominence
            if not is_local_hvn:
                continue
            score0 = float(vols[j0] * prominence0)
            if score0 > best_score:
                best_j = j0
                best_score = score0
                best_prominence = prominence0
        if best_j is None:
            return None
        j = int(best_j)
        prominence = float(best_prominence)
        score = float(best_score)

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
        "lower_poc_hvn_prominence": float(prominence) if np.isfinite(prominence) else np.nan,
        "lower_poc_hvn_score": float(score),
    }


def compute_profile(fp_store: FootprintStore, start_bar_id: int, end_bar_id: int, cfg: Config) -> dict[str, float] | None:
    return fp_store.compute_profile(start_bar_id, end_bar_id, cfg)


def _find_latest_lh_break(
    piv_highs: list[int],
    piv_lows: list[int],
    high: np.ndarray,
    low: np.ndarray,
    close_i: float,
    i: int,
    last_break_i: int,
    cfg: Config,
) -> dict[str, int | float] | None:
    """Find the latest valid lower-high break.

    This is an algorithmic optimization of the original logic, not a strategy change.
    It preserves the same ordering and return semantics, but avoids repeated full-list
    comprehensions on every bar by:
    - checking close > lower_high before expensive context lookups;
    - scanning previous pivot highs backward only until the latest higher pivot is found;
    - using bisect on sorted pivot lows to inspect only lows in the relevant interval.
    """
    if i - last_break_i < cfg.min_bars_between_lh_breaks:
        return None
    if not piv_highs or not piv_lows:
        return None

    # Latest confirmed pivot high that acts as lower high. Iterate in the same
    # reverse order as the original implementation, so the chosen candidate is unchanged.
    for pos in range(len(piv_highs) - 1, -1, -1):
        lh_i = piv_highs[pos]
        if lh_i >= i - cfg.pivot_lookback:
            continue
        lh_price = high[lh_i]

        # Original code did this check after building several lists. It is a pure
        # short-circuit because the candidate can never be a break unless close_i > lh_price.
        if close_i <= lh_price:
            continue

        major_high_i: int | None = None
        for hp in range(pos - 1, -1, -1):
            h_i = piv_highs[hp]
            if high[h_i] > lh_price:
                major_high_i = h_i
                break
        if major_high_i is None:
            continue

        # Equivalent to lows_between + lows_after_lh in the original code:
        # all pivot lows with major_high_i < low_i < i, excluding the rare case
        # where the same bar is both the lower-high pivot and pivot low.
        lo_pos = bisect_right(piv_lows, major_high_i)
        hi_pos = bisect_left(piv_lows, i)
        if lo_pos >= hi_pos:
            continue

        leg_low_i: int | None = None
        leg_low_price = np.inf
        for l_i in piv_lows[lo_pos:hi_pos]:
            if l_i == lh_i:
                continue
            v = low[l_i]
            if v < leg_low_price:
                leg_low_price = v
                leg_low_i = l_i
        if leg_low_i is None:
            continue

        if (high[major_high_i] - low[leg_low_i]) / high[major_high_i] < cfg.min_downmove_pct:
            continue

        before_major_pos = bisect_left(piv_lows, major_high_i) - 1
        profile_start_i = piv_lows[before_major_pos] if before_major_pos >= 0 else max(0, major_high_i - cfg.profile_padding_bars)
        profile_start_i = max(0, profile_start_i - cfg.profile_padding_bars)
        if i - profile_start_i > cfg.max_profile_bars:
            profile_start_i = max(0, i - cfg.max_profile_bars)

        return {
            "lower_high_i": int(lh_i),
            "lower_high_price": float(high[lh_i]),
            "major_high_i": int(major_high_i),
            "major_high_price": float(high[major_high_i]),
            "leg_low_i": int(leg_low_i),
            "leg_low_price": float(low[leg_low_i]),
            "profile_start_i": int(profile_start_i),
            "profile_end_i": int(i),
        }
    return None


def _is_sell_bubble(row: pd.Series, profile: dict[str, float], cfg: Config) -> bool:
    bucket = row.get("max_sell_bucket", np.nan)
    if not np.isfinite(bucket):
        return False
    in_region = profile["lower_poc_zone_low"] <= bucket <= profile["val"]
    if not in_region:
        return False
    if row["low"] < profile["lower_poc_zone_low"] * (1.0 - cfg.lower_poc_invalidate_buffer_pct):
        return False
    enough_sell = np.isfinite(row["sell_q"]) and row["max_bucket_sell_notional"] >= row["sell_q"]
    high_impact = np.isfinite(row["sell_impact_q"]) and row["sell_impact"] >= row["sell_impact_q"]
    negative_delta = row["delta_notional"] < 0 or row["max_bucket_delta_notional"] < 0
    return bool((enough_sell or high_impact) and negative_delta)


def _cluster_absorption(bars_slice: pd.DataFrame, profile: dict[str, float], cfg: Config) -> dict[str, Any] | None:
    """Find repeated sell-bubble absorption in a bar slice.

    Vectorized equivalent of the original row-by-row implementation:
    - max sell bucket must be inside lower_poc_zone_low..VAL
    - bar low must not invalidate lower POC
    - sell size or sell-impact must be high
    - bar or max bucket delta must be negative
    """
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

    band_low = float(profile["lower_poc_zone_low"])
    band_high = float(profile["val"])
    band_width = band_high - band_low
    zone_mid = (zone_low + zone_high) / 2.0
    absorption_band_position = np.nan
    if band_width > 0:
        absorption_band_position = float((zone_mid - band_low) / band_width)

    if cfg.absorption_location_mode == "lower_band":
        # Lower-band control from V8. Kept for comparison only.
        if not (np.isfinite(absorption_band_position) and absorption_band_position <= cfg.absorption_max_band_position):
            return None
    elif cfg.absorption_location_mode == "upper_band":
        # V9 hypothesis: a long setup may be stronger when sell bubbles are absorbed
        # closer to VAL, meaning the market accepts/defends higher value instead of
        # only catching a falling knife near lower POC. This uses only frozen profile
        # and historical bubbles, so it does not introduce future leakage.
        if not (np.isfinite(absorption_band_position) and absorption_band_position >= cfg.absorption_min_band_position):
            return None

    return {
        "bubble_count": int(len(bubbles)),
        "bubble_bar_ids": [int(x) for x in bubbles["bar_id"].to_list()],
        "bubble_times": [pd.Timestamp(x) for x in bubbles["end_ts"].to_list()],
        "absorption_zone_low": float(zone_low),
        "absorption_zone_high": float(zone_high),
        "absorption_band_position": float(absorption_band_position) if np.isfinite(absorption_band_position) else np.nan,
        "bubble_total_sell_notional": float(bubbles["max_bucket_sell_notional"].sum()),
    }


def generate_signals(bars: pd.DataFrame, fp_store: FootprintStore, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = add_features(bars, fp_store, cfg).reset_index(drop=True)
    n = len(b)
    high = b["high"].to_numpy(float)
    low = b["low"].to_numpy(float)
    close = b["close"].to_numpy(float)
    end_ts_np = pd.to_datetime(b["end_ts"]).to_numpy(dtype="datetime64[ns]")
    bar_ids = b["bar_id"].to_numpy(np.int64)

    progress_marks = _build_progress_marks(cfg.start_date, cfg.end_date, months=1)
    progress_pos = 0
    start_ts64 = np.datetime64(pd.Timestamp(cfg.start_date).to_datetime64())
    start_candidates = np.flatnonzero(end_ts_np >= start_ts64)
    first_trade_i = int(start_candidates[0]) if len(start_candidates) else 0
    warmup_scan_bars = max(
        cfg.max_profile_bars + cfg.absorption_scan_back_bars + cfg.bubble_quantile_window + 50,
        2000,
    )
    loop_start_i = max(0, first_trade_i - warmup_scan_bars)
    print(
        f"[{STRATEGY_NAME}][signals] start scan | loop_start_i={loop_start_i:,} "
        f"loop_start_time={pd.Timestamp(end_ts_np[loop_start_i])} "
        f"first_trade_i={first_trade_i:,} first_trade_time={pd.Timestamp(end_ts_np[first_trade_i])} "
        f"monthly_progress_from={cfg.start_date}",
        flush=True,
    )

    sig = np.zeros(n, dtype=np.int8)
    setup_id_arr = np.full(n, -1, dtype=int)
    entry_trigger = np.full(n, np.nan)
    initial_stop = np.full(n, np.nan)
    target_price = np.full(n, np.nan)
    rr_filter_target_price = np.full(n, np.nan)
    planned_reward_risk = np.full(n, np.nan)
    target_distance_pct = np.full(n, np.nan)
    absorption_zone_width_pct = np.full(n, np.nan)
    absorption_band_position = np.full(n, np.nan)
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
    last_break_i = -10**9
    setup_seq = 0
    active: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    cooldown_until = -1
    profile_cache: dict[tuple[int, int], dict[str, float] | None] = {}

    def log(i: int, event: str, info: dict[str, Any]) -> None:
        row = {"event_time": pd.Timestamp(b.loc[i, "end_ts"]), "bar_id": int(bar_ids[i]), "event": event}
        row.update(info)
        events.append(row)

    def _prepare_order_or_retest(i: int, setup: dict[str, Any], cluster: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
        """Return (active_setup, pending_order, should_cooldown).

        V8 fix: if the absorption-zone trigger has already been crossed at the
        moment we recognize the setup, do not retroactively buy the old breakout.
        Instead wait for a retest and a fresh breakout trigger.
        """
        nonlocal setup_seq
        setup.update(cluster)
        trigger = setup["absorption_zone_high"] * (1.0 + cfg.entry_buffer_pct)
        stop = setup["lower_poc_zone_low"] * (1.0 - cfg.stop_buffer_pct)
        target = setup["poc"] if cfg.target_mode == "poc" else setup["vah"]
        rr_filter_target = setup["vah"] if cfg.rr_filter_target == "vah" else target
        risk = trigger - stop
        selected_reward = target - trigger
        filter_reward = rr_filter_target - trigger
        rr = filter_reward / risk if risk > 0 else np.nan
        target_distance_pct = selected_reward / trigger if trigger > 0 else np.nan
        absorption_zone_width_pct = (setup["absorption_zone_high"] - setup["absorption_zone_low"]) / setup["absorption_zone_low"] if setup.get("absorption_zone_low", 0) > 0 else np.nan
        base = {
            **setup,
            "entry_trigger": float(trigger),
            "initial_stop": float(stop),
            "target_price": float(target),
            "rr_filter_target_price": float(rr_filter_target),
            "reward_risk": float(rr) if np.isfinite(rr) else np.nan,
            "target_distance_pct": float(target_distance_pct) if np.isfinite(target_distance_pct) else np.nan,
            "absorption_zone_width_pct": float(absorption_zone_width_pct) if np.isfinite(absorption_zone_width_pct) else np.nan,
            "absorption_band_position": float(setup.get("absorption_band_position", np.nan)) if np.isfinite(setup.get("absorption_band_position", np.nan)) else np.nan,
            "target_mode": cfg.target_mode,
            "rr_filter_target": cfg.rr_filter_target,
        }
        if not (risk > 0 and selected_reward > 0):
            log(i, "reject_target_not_above_trigger", base)
            return None, None, True
        if not (np.isfinite(target_distance_pct) and target_distance_pct >= cfg.min_target_distance_pct):
            log(i, "reject_target_too_close", base)
            return None, None, True
        if not (filter_reward > 0 and rr >= cfg.min_reward_risk):
            log(i, "reject_order_low_rr", base)
            return None, None, True

        setup_seq += 1
        base["setup_id"] = setup_seq

        # Key V8 realism guard. A buy stop must be above the market when it is
        # placed. If current bar already traded through it, that breakout is
        # historical and cannot be retroactively entered.
        if high[i] >= trigger:
            wait = {
                **base,
                "wait_retest": True,
                "retest_seen": False,
                "created_i": i,
                "last_scan_i": i,
            }
            log(i, "wait_retest_after_trigger_already_crossed", wait)
            return wait, None, False

        pending_order = {**base, "active_from_i": i}
        log(i, "buy_stop_placed", pending_order)
        return setup, pending_order, False

    for i in range(loop_start_i, n):
        if i > loop_start_i and (i - loop_start_i) % 25000 == 0:
            print(
                f"[{STRATEGY_NAME}][signals] bar_progress "
                f"bar={i + 1:,}/{n:,} current={pd.Timestamp(end_ts_np[i])} "
                f"events={len(events):,} signals={int(sig.sum()):,} "
                f"active={active is not None} pending={pending is not None}",
                flush=True,
            )
        while progress_pos < len(progress_marks) and end_ts_np[i] >= progress_marks[progress_pos]:
            print(
                f"[{STRATEGY_NAME}][signals] completed_to={_format_progress_ts(progress_marks[progress_pos])} "
                f"bar={i + 1:,}/{n:,} current={pd.Timestamp(end_ts_np[i])} "
                f"events={len(events):,} signals={int(sig.sum()):,} "
                f"active={active is not None} pending={pending is not None}",
                flush=True,
            )
            progress_pos += 1

        ph, pl = _confirmed_pivots_at_i(high, low, i, cfg.pivot_lookback)
        if ph is not None:
            piv_highs.append(ph)
        if pl is not None:
            piv_lows.append(pl)

        # Pending buy-stop.
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
                rr_filter_target_price[i] = pending.get("rr_filter_target_price", pending["target_price"])
                planned_reward_risk[i] = pending.get("reward_risk", np.nan)
                target_distance_pct[i] = pending.get("target_distance_pct", np.nan)
                absorption_zone_width_pct[i] = pending.get("absorption_zone_width_pct", np.nan)
                absorption_band_position[i] = pending.get("absorption_band_position", np.nan)
                profile_vah[i] = pending["vah"]
                profile_val[i] = pending["val"]
                profile_poc[i] = pending["poc"]
                lower_poc[i] = pending["lower_poc"]
                absorption_low[i] = pending["absorption_zone_low"]
                absorption_high[i] = pending["absorption_zone_high"]
                bubble_count[i] = pending["bubble_count"]
                signal_reason[i] = "lh_break_anchored_vp_absorption_breakout"
                log(i, "entry_triggered", pending)
                pending = None
                active = None
                cooldown_until = i + cfg.cooldown_bars
                continue

        if i < cooldown_until:
            continue

        # If active setup exists, keep scanning for post-break bubbles, retest, or cancellation.
        if active is not None:
            if i - active["created_i"] > cfg.max_active_setup_bars:
                log(i, "cancel_active_timeout", active)
                active = None
                cooldown_until = i + cfg.cooldown_bars
            elif low[i] < active["lower_poc_zone_low"] * (1.0 - cfg.lower_poc_invalidate_buffer_pct):
                log(i, "cancel_active_break_lower_poc", active)
                active = None
                cooldown_until = i + cfg.cooldown_bars
            elif pending is None and active.get("wait_retest", False):
                # V8 retest mode: old trigger was already crossed. Wait for price
                # to retest the absorption zone from above, hold it, then place a
                # fresh buy stop above the retest bar high.
                retest_level = active["absorption_zone_high"]
                if low[i] <= active["entry_trigger"] and close[i] >= retest_level:
                    fresh_trigger = max(high[i], active["entry_trigger"]) * (1.0 + cfg.entry_buffer_pct)
                    fresh = {**active, "entry_trigger": float(fresh_trigger), "active_from_i": i, "wait_retest": False, "retest_seen": True}
                    pending = fresh
                    log(i, "buy_stop_placed_after_retest", pending)
                active["last_scan_i"] = i
            elif pending is None:
                # Check newly completed bars from profile break onward.
                scan_start = max(active["last_scan_i"] + 1, active["profile_end_i"])
                scan = b.iloc[scan_start : i + 1]
                cluster = _cluster_absorption(scan, active, cfg)
                active["last_scan_i"] = i
                if cluster is not None:
                    next_active, next_pending, should_cooldown = _prepare_order_or_retest(i, active, cluster)
                    active = next_active
                    pending = next_pending
                    if should_cooldown:
                        cooldown_until = i + cfg.cooldown_bars

        # New lower-high break creates a fresh anchored profile.
        lh = _find_latest_lh_break(piv_highs, piv_lows, high, low, close[i], i, last_break_i, cfg)
        if lh is None:
            continue
        last_break_i = i
        start_i = int(lh["profile_start_i"])
        end_i = int(lh["profile_end_i"])
        key = (int(bar_ids[start_i]), int(bar_ids[end_i]))
        if key not in profile_cache:
            profile_cache[key] = compute_profile(fp_store, key[0], key[1], cfg)
        prof = profile_cache[key]
        base_info = {**lh, "profile_start_bar_id": key[0], "profile_end_bar_id": key[1]}
        if prof is None:
            log(i, "reject_profile_no_lower_poc_or_empty", base_info)
            continue
        setup = {**base_info, **prof, "created_i": i, "last_scan_i": max(0, i - cfg.absorption_scan_back_bars)}
        log(i, "lower_high_broken_profile_created", setup)

        # Immediately look back before the lower-high break; Fabio often sees absorption before the break.
        scan_start = max(start_i, i - cfg.absorption_scan_back_bars)
        cluster = _cluster_absorption(b.iloc[scan_start : i + 1], setup, cfg)
        if cluster is None:
            active = setup
            log(i, "active_waiting_for_absorption", setup)
            continue
        next_active, next_pending, should_cooldown = _prepare_order_or_retest(i, setup, cluster)
        active = next_active
        pending = next_pending
        if should_cooldown:
            cooldown_until = i + cfg.cooldown_bars

    out = b.copy()
    out["signal"] = sig
    out["setup_id"] = setup_id_arr
    out["entry_trigger"] = entry_trigger
    out["initial_stop"] = initial_stop
    out["target_price"] = target_price
    out["rr_filter_target_price"] = rr_filter_target_price
    out["planned_reward_risk"] = planned_reward_risk
    out["target_distance_pct"] = target_distance_pct
    out["absorption_zone_width_pct"] = absorption_zone_width_pct
    out["absorption_band_position"] = absorption_band_position
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
    ts_a = pd.to_datetime(d["end_ts"]).to_numpy()
    progress_marks = _build_progress_marks(cfg.start_date, cfg.end_date, months=1)
    progress_pos = 0
    sig = d["signal"].to_numpy(int)
    trigger = d["entry_trigger"].to_numpy(float)
    stop_a = d["initial_stop"].to_numpy(float)
    target_a = d["target_price"].to_numpy(float)
    rr_filter_target_a = d["rr_filter_target_price"].to_numpy(float) if "rr_filter_target_price" in d.columns else target_a
    setup_id = d["setup_id"].to_numpy(int)

    capital = cfg.initial_capital
    peak = capital
    pos: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for i in range(n):
        while progress_pos < len(progress_marks) and np.datetime64(ts_a[i]) >= progress_marks[progress_pos]:
            print(
                f"[{STRATEGY_NAME}][backtest] completed_to={_format_progress_ts(progress_marks[progress_pos])} "
                f"bar={i + 1:,}/{n:,} current={pd.Timestamp(ts_a[i])} "
                f"trades={len(trades):,} capital={capital:.2f} open_position={pos is not None}",
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
            if low_a[i] <= pos["stop"]:
                exit_price = pos["stop"]
                exit_reason = "stop_lower_poc"
            elif high_a[i] >= pos["target"]:
                exit_price = pos["target"]
                exit_reason = f"target_{cfg.target_mode}"
            elif (ts - pos["entry_time"]).total_seconds() / 3600.0 >= cfg.max_holding_hours:
                exit_price = close_a[i]
                exit_reason = "time"
            if exit_price is not None:
                fill = apply_exit_slippage(float(exit_price), 1, cfg.slippage_pct)
                fee = pos["entry_fee"] + abs(pos["qty"] * fill) * cfg.fee_rate
                gross = (fill - pos["entry"]) * pos["qty"]
                pnl = gross - fee
                before = capital
                capital += pnl
                trades.append({"entry_time": pos["entry_time"], "exit_time": ts, "type": "LONG", "entry": pos["entry"], "exit": fill, "pnl": pnl, "fee": fee, "capital": capital, "return_pct": pnl / max(before, 1e-12), "mfe_r": pos["mfe"] / pos["risk_dist"], "mae_r": pos["mae"] / pos["risk_dist"], "holding_hours": (ts - pos["entry_time"]).total_seconds() / 3600, "exit_reason": exit_reason, "setup_id": pos["setup_id"]})
                pos = None
                continue

        if pos is None and sig[i] == 1:
            if not (np.isfinite(trigger[i]) and np.isfinite(stop_a[i]) and np.isfinite(target_a[i])):
                continue
            entry_raw = max(open_a[i], trigger[i])
            entry = apply_entry_slippage(float(entry_raw), 1, cfg.slippage_pct)
            risk = entry - stop_a[i]
            reward = target_a[i] - entry
            filter_reward = rr_filter_target_a[i] - entry
            if risk <= 0 or reward <= 0 or filter_reward <= 0 or filter_reward / risk < cfg.min_reward_risk:
                continue
            qty = capital * cfg.unit_risk_per_trade / risk
            notional = abs(qty * entry)
            max_notional = capital * cfg.max_notional_mult
            if notional > max_notional:
                qty *= max_notional / notional
                notional = max_notional
            if qty <= 0:
                continue
            pos = {"entry": float(entry), "stop": float(stop_a[i]), "target": float(target_a[i]), "risk_dist": float(risk), "qty": float(qty), "entry_fee": float(notional * cfg.fee_rate), "entry_time": ts, "entry_i": i, "mfe": 0.0, "mae": 0.0, "setup_id": int(setup_id[i])}

    if pos is not None:
        ts = pd.Timestamp(ts_a[-1])
        fill = apply_exit_slippage(float(close_a[-1]), 1, cfg.slippage_pct)
        fee = pos["entry_fee"] + abs(pos["qty"] * fill) * cfg.fee_rate
        gross = (fill - pos["entry"]) * pos["qty"]
        pnl = gross - fee
        before = capital
        capital += pnl
        trades.append({"entry_time": pos["entry_time"], "exit_time": ts, "type": "LONG", "entry": pos["entry"], "exit": fill, "pnl": pnl, "fee": fee, "capital": capital, "return_pct": pnl / max(before, 1e-12), "mfe_r": pos["mfe"] / pos["risk_dist"], "mae_r": pos["mae"] / pos["risk_dist"], "holding_hours": (ts - pos["entry_time"]).total_seconds() / 3600, "exit_reason": "final", "setup_id": pos["setup_id"]})

    equity = pd.DataFrame(equity_rows).set_index("timestamp")
    summary = summarize_r_trades(trades, equity, cfg.initial_capital)
    summary.update({"signal_count": int((d["signal"] == 1).sum()), "range_pct": cfg.range_pct, "price_step": cfg.price_step, "fee_rate_per_side": cfg.fee_rate, "slippage_pct": cfg.slippage_pct, "unit_risk_per_trade": cfg.unit_risk_per_trade, "max_holding_hours": cfg.max_holding_hours, "target_mode": cfg.target_mode, "rr_filter_target": cfg.rr_filter_target, "lower_poc_mode": cfg.lower_poc_mode, "min_reward_risk": cfg.min_reward_risk, "min_target_distance_pct": cfg.min_target_distance_pct, "max_absorption_zone_width_pct": cfg.max_absorption_zone_width_pct, "absorption_location_mode": cfg.absorption_location_mode, "absorption_max_band_position": cfg.absorption_max_band_position, "absorption_min_band_position": cfg.absorption_min_band_position})
    return trades, equity, summary


def write_outputs(signals: pd.DataFrame, events: pd.DataFrame, trades: list[dict[str, Any]], equity: pd.DataFrame, summary: dict[str, Any], cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(cfg.out_dir / "eth_range_anchored_vp_absorption_v9_trades.csv", index=False)
    equity.to_csv(cfg.out_dir / "eth_range_anchored_vp_absorption_v9_equity.csv")
    events.to_csv(cfg.out_dir / "eth_range_anchored_vp_absorption_v9_setup_events.csv", index=False)
    events["event"].value_counts().rename_axis("event").reset_index(name="count").to_csv(cfg.out_dir / "eth_range_anchored_vp_absorption_v9_event_counts.csv", index=False)
    sigs = signals[(signals["signal"] == 1) & (signals["end_ts"] >= pd.Timestamp(cfg.start_date))].copy()
    cols = ["end_ts", "bar_id", "open", "high", "low", "close", "setup_id", "entry_trigger", "initial_stop", "target_price", "rr_filter_target_price", "planned_reward_risk", "target_distance_pct", "absorption_zone_width_pct", "absorption_band_position", "profile_vah", "profile_val", "profile_poc", "lower_poc", "absorption_low", "absorption_high", "bubble_count", "signal_reason"]
    sigs[[c for c in cols if c in sigs.columns]].to_csv(cfg.out_dir / "eth_range_anchored_vp_absorption_v9_signal_audit.csv", index=False)
    if cfg.write_full_audit:
        signals.to_csv(cfg.out_dir / "eth_range_anchored_vp_absorption_v9_full_audit.csv", index=False)
    with (cfg.out_dir / "eth_range_anchored_vp_absorption_v9_summary.json").open("w", encoding="utf-8") as f:
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
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--preset", choices=sorted(PRESETS), default="high")
    p.add_argument("--range-pct", type=float, default=0.0020)
    p.add_argument("--price-step", type=float, default=1.0)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.00015)
    p.add_argument("--max-holding-hours", type=float, default=48.0)
    p.add_argument("--min-reward-risk", type=float, default=2.0)
    p.add_argument("--min-target-distance-pct", type=float, default=0.015)
    p.add_argument("--max-absorption-zone-width-pct", type=float, default=0.010)
    p.add_argument("--absorption-location-mode", choices=["any", "lower_band", "upper_band"], default="upper_band")
    p.add_argument("--absorption-max-band-position", type=float, default=0.45)
    p.add_argument("--absorption-min-band-position", type=float, default=0.55)
    p.add_argument("--target-mode", choices=["vah", "poc"], default="vah")
    p.add_argument("--lower-poc-mode", choices=["max_volume", "hvn"], default="hvn")
    p.add_argument("--lower-poc-hvn-min-prominence", type=float, default=1.15)
    p.add_argument("--rr-filter-target", choices=["selected", "vah"], default="selected")
    p.add_argument("--write-full-audit", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(symbol=args.symbol, start_date=args.start_date, end_date=args.end_date, warmup_start_date=args.warmup_start_date, initial_capital=args.initial_capital, fee_rate=args.fee_rate, slippage_pct=args.slippage_pct, range_pct=args.range_pct, price_step=args.price_step, data_dir=args.data_dir, max_holding_hours=args.max_holding_hours, min_reward_risk=args.min_reward_risk, min_target_distance_pct=args.min_target_distance_pct, max_absorption_zone_width_pct=args.max_absorption_zone_width_pct, absorption_location_mode=args.absorption_location_mode, absorption_max_band_position=args.absorption_max_band_position, absorption_min_band_position=args.absorption_min_band_position, target_mode=args.target_mode, lower_poc_mode=args.lower_poc_mode, lower_poc_hvn_min_prominence=args.lower_poc_hvn_min_prominence, rr_filter_target=args.rr_filter_target, write_full_audit=args.write_full_audit)
    for k, v in PRESETS[args.preset].items():
        setattr(cfg, k, v)
    cfg.out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/reports/mf/eth_range_anchored_vp_absorption_v9/{args.preset}_r{int(cfg.range_pct*10000):04d}_step{cfg.price_step:g}_{cfg.target_mode}_rr{cfg.rr_filter_target}_{cfg.lower_poc_mode}_struct")

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
    print_full_report(build_report_trades(trades), report_df, cfg.initial_capital, final_capital, STRATEGY_NAME, total_days, False, symbol=cfg.symbol, report_dir=cfg.out_dir)


if __name__ == "__main__":
    main()
