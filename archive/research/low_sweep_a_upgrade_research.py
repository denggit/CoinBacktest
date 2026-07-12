#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A-upgrade research lab for low-sweep panic reversal.

This script deliberately stays under ``research/``.  The current A edge has
survived sequential validation, but entry, support-zone definition, exit logic,
MAE control, and frequency expansion are still research questions.

What it tests, causally:
- support definition upgrades: single confirmed swing low vs equal-low clusters
  and aged liquidity pools with dynamic age thresholds;
- A0/A1/A2/B auxiliary layers: do not simply loosen A; add context/confirmation;
- entry upgrades: next-open baseline, reclaim confirmation, and conservative
  post-signal limit pullback fills;
- exit upgrades: fixed time exit, structural target, MFE protection, momentum
  exhaustion, and partial target + runner;
- optional range-bar and range-footprint context breakdowns using only closed
  range bars (merge_asof on range ``end_ts`` <= signal_time).

No direct SQLite/CSV/ZIP reading is performed here. Data access stays inside
``src.data_feed`` loaders.  All rolling filters inherited from the upstream
no-leakage probe use historical ``shift(1).rolling(...)`` thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.focused_low_sweep_reversal_event_lab import confirmed_swing_lows  # noqa: E402
from research.low_sweep_panic_reversal_strategy_backtest_probe import (  # noqa: E402
    StopSpec,
    _equity_and_dd,
    _profit_factor,
    _split_csv_names,
    _timeframe_to_minutes,
    build_variants as build_probe_variants,
    parse_args as _base_parse_args,
    parse_stop_specs,
    prepare_studied_events,
    run_variant_jobs as run_probe_variant_jobs,
    select_candidate_events,
    simulate_trade_path as simulate_probe_trade_path,
)
from research.low_sweep_panic_reversal_strategy_probe import build_fixed_candidate_masks  # noqa: E402
from research.low_sweep_scale_in_path_probe import get_scale_in_schemes  # noqa: E402
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader, range_code  # noqa: E402
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader, price_step_code  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402

SCRIPT_NAME = "low_sweep_a_upgrade_research"
DEFAULT_OUT_DIR = "data/reports/research/low_sweep_a_upgrade_research_tradebar_1m"


@dataclass(frozen=True)
class UpgradeVariant:
    variant_name: str
    candidate_layer: str
    support_mode: str
    entry_mode: str
    exit_mode: str
    stop_spec: StopSpec


@dataclass(frozen=True)
class MarketCache:
    """Immutable numeric arrays reused by every variant/job.

    The first version converted the full 2M+ row DataFrame and recomputed the
    reclaim rolling-volume baseline inside every single trade simulation.  That
    preserved logic but made broad research unbearably slow.  This cache is a
    pure performance layer: it does not change timestamps, signal generation,
    entry/exit rules, or any causal ordering.
    """

    index: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    reclaim_vol_base: np.ndarray


def build_market_cache(bars: pd.DataFrame, args: argparse.Namespace) -> MarketCache:
    volume = pd.to_numeric(bars.get("volume", pd.Series(np.nan, index=bars.index)), errors="coerce")
    vol_window = int(args.reclaim_volume_window)
    vol_min = max(3, vol_window // 3)
    return MarketCache(
        index=pd.DatetimeIndex(bars.index),
        open=pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float),
        high=pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float),
        low=pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float),
        close=pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float),
        volume=volume.to_numpy(dtype=float),
        reclaim_vol_base=volume.shift(1).rolling(vol_window, min_periods=vol_min).mean().to_numpy(dtype=float),
    )


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse base low-sweep args plus A-upgrade specific flags.

    We intentionally delegate the common low-sweep/event/stress arguments to the
    existing broad strategy probe parser, then append upgrade flags.  User args
    that belong to the base parser are passed through unchanged.
    """

    up = argparse.ArgumentParser(add_help=False)
    up.add_argument(
        "--candidate-layers",
        default=(
            "A1_current,A0_spike_ge_0100,A0_deep_spike_ge_0120,"
            "A2_soft_zone_v2,A2_reclaim_zone_v2,B_aux_session_spike_atr,"
            "A1_fp_low_delta_vneg,A0_fp_low_delta_vneg,"
            "A1_fp_abs_delta_high,A0_fp_abs_delta_high,"
            "A1_only_080_100,A1_only_fp_abs_delta_high,A1_only_fp_low_delta_vneg,"
            "A1_range_delta_mild_neg,A0_range_delta_mild_neg,"
            "A0_micro5_last20_buy_pressure,A0_micro10_last20_buy_pressure"
        ),
    )
    up.add_argument("--support-modes", default="single_swing,equal2_020,equal3_030,aged_ge_12,aged_ge_24,aged_ge_36")
    up.add_argument("--entry-modes", default="next_open,reclaim_swing_3,reclaim_signal_close_3,limit_low10_1,limit_low20_3")
    up.add_argument("--exit-modes", default="time48,time60,target_signal_open_or_time48,mfe_protect_15_05_time60,momentum_exhaust_time60,partial_target_trail_time96,swing_trail_after18_time96")
    up.add_argument("--upgrade-stop-specs", default="no_stop,fixed_0250,atr_6x")

    up.add_argument("--cluster-lookback-bars", type=int, default=1440)
    up.add_argument("--cluster-tolerance-pcts", default="0.0020,0.0030")
    up.add_argument("--aged-swing-min-age", type=int, default=60)
    up.add_argument("--soft-spike-pct", type=float, default=0.0060)
    up.add_argument("--soft-close-pos", type=float, default=0.35)
    up.add_argument("--limit-tick-size", type=float, default=0.01)
    up.add_argument("--reclaim-volume-window", type=int, default=20)
    up.add_argument("--reclaim-volume-mult", type=float, default=1.0)
    up.add_argument("--mfe-trigger-pct", type=float, default=0.015)
    up.add_argument("--mfe-lock-pct", type=float, default=0.005)
    up.add_argument("--trail-trigger-pct", type=float, default=0.025)
    up.add_argument("--trail-giveback-pct", type=float, default=0.010)
    up.add_argument("--swing-trail-start-bars", default="12,18,24", help="Post-entry bars after which newly confirmed swing lows may lift the trailing stop.")
    up.add_argument("--swing-trail-buffer-pct", type=float, default=0.0010, help="Buffer below newly confirmed swing lows for dynamic trailing stops.")
    up.add_argument("--momentum-exhaust-bars", type=int, default=2)
    up.add_argument("--momentum-exhaust-drop-pct", type=float, default=0.005)
    up.add_argument("--target-exit-next-open", action="store_true", default=True)
    up.add_argument("--target-entry-same-bar-min-delay", type=int, default=1, help="Earliest bar after entry where target exits may trigger.")

    up.add_argument("--context-sources", default="trade_bar,range_bar,footprint", help="Comma list: trade_bar, range_bar, footprint. Missing range/footprint cache errors are reported and skipped.")
    up.add_argument("--micro-timeframes", default="5s,10s", help="Optional micro trade bars to attach as closed signal-bar context. Missing/unsupported timeframes are skipped.")
    up.add_argument("--micro-last-seconds", type=int, default=20, help="Seconds at the end of the 1m signal bar used for micro exhaustion/turn filters.")
    up.add_argument("--micro-buy-sell-ratio-min", type=float, default=1.20, help="Micro buy/sell notional ratio threshold for buy-pressure candidate layers.")
    up.add_argument("--micro-load-mode", choices=["local", "fetch", "auto"], default="local", help="How to load micro trade bars by monthly sliding windows. local=cache-only load_local_data; fetch=monthly fetch_data_by_date_range; auto=local first, then monthly fetch fallback when local returns empty.")
    up.add_argument("--micro-debug-sample-limit", type=int, default=500, help="Maximum event-level micro attach debug rows to keep in 21_micro_match_debug.csv.")
    up.add_argument("--micro-sell-exhaustion-delta-min", type=float, default=-0.15, help="Micro delta-pressure threshold used as sell-exhaustion confirmation. Example: -0.15 means final seconds are no longer dominated by sellers.")
    up.add_argument("--micro-no-new-low-buffer-pct", type=float, default=0.0000, help="Micro no-new-low buffer versus the parent signal low. 0 means last micro window low must be >= signal low.")
    up.add_argument("--micro-large-delta-min", type=float, default=0.0, help="Micro large-trade delta pressure threshold for large-buy confirmation layers.")
    up.add_argument("--range-pcts", default="0.0015,0.0020,0.0025")
    up.add_argument("--footprint-range-pct", type=float, default=0.0020)
    up.add_argument("--footprint-price-step", type=float, default=1.0)
    up.add_argument("--context-breakdown-top-variants", type=int, default=25)
    up.add_argument("--fp-low-delta-vneg-threshold", type=float, default=-0.10, help="Fixed causal footprint filter threshold: low bucket delta pressure <= this value.")
    up.add_argument("--fp-abs-delta-high-threshold", type=float, default=0.60, help="Fixed causal footprint filter threshold: max bucket abs delta pressure >= this value.")
    up.add_argument("--range-delta-mild-neg-min", type=float, default=-0.35, help="Fixed causal range filter lower bound for mild negative delta pressure.")
    up.add_argument("--range-delta-mild-neg-max", type=float, default=0.05, help="Fixed causal range filter upper bound for mild negative delta pressure.")
    up.add_argument("--skip-consistency-audit", action="store_true", help="Skip baseline consistency audit against the existing A strategy-validation engine.")

    up.add_argument("--save-trades", type=int, default=0, help="Rows of detailed trades to save. Default 0 avoids huge upload-unfriendly reports; use e.g. 20000 when needed.")
    up.add_argument("--save-events", type=int, default=5000)
    up.add_argument("--min-trades-for-upgrade-edge", type=int, default=80)
    up.add_argument("--min-pf-for-upgrade-edge", type=float, default=1.25)
    up.add_argument("--max-top5-winner-share-for-edge", type=float, default=0.45)

    known, rest = up.parse_known_args(argv)
    # Broaden the event generation defaults so aged zones and soft variants have
    # a chance to exist. User-supplied base args in ``rest`` still win.
    base_defaults = [
        "--out-dir", DEFAULT_OUT_DIR,
        "--candidate-names", "A_spike_close_large_share",
        "--max-swing-ages", "12,24,48,96,240,1440",
        "--spike-pcts", "0.0060,0.0080,0.0100,0.0120",
        "--stop-specs", "no_stop,fixed_0250,atr_6x",
        "--entry-schemes", "full_entry",
        "--exit-horizons", "48",
        "--save-trade-sample", "200000",
    ]
    args = _base_parse_args(base_defaults + list(rest))
    for k, v in vars(known).items():
        setattr(args, k, v)
    if bool(getattr(args, "fast", False)):
        args.candidate_layers = "A1_current,A0_spike_ge_0100,A0_micro5_last20_buy_pressure"
        args.support_modes = "single_swing,equal2_020,aged_ge_12"
        args.entry_modes = "next_open,reclaim_swing_3,limit_low10_1"
        args.exit_modes = "time48,mfe_protect_15_05_time60,swing_trail_after18_time96"
        args.upgrade_stop_specs = "no_stop,fixed_0250"
        args.context_sources = "trade_bar"
        args.micro_timeframes = ""
    return args


# ---------------------------------------------------------------------------
# Utility stats
# ---------------------------------------------------------------------------


def _parse_float_list(raw: str) -> list[float]:
    vals: list[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return sorted(set(vals))


def _entry_cost(args: argparse.Namespace, cost_mult: float = 1.0) -> float:
    return float(args.entry_fee_rate + args.entry_slippage_pct) * float(cost_mult)


def _exit_cost(args: argparse.Namespace, cost_mult: float = 1.0) -> float:
    return float(args.exit_fee_rate + args.exit_slippage_pct) * float(cost_mult)


def _payoff_ratio(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    wins = vals[vals > 0]
    losses = vals[vals < 0]
    if wins.empty or losses.empty:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def _top_winner_share(x: pd.Series, top_n: int = 5) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    pos = vals[vals > 0].sort_values(ascending=False)
    if pos.empty:
        return float("nan")
    denom = float(pos.sum())
    return float(pos.head(top_n).sum() / denom) if denom > 0 else float("nan")


def _max_consecutive_losses(x: pd.Series) -> int:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    best = cur = 0
    for v in vals:
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _safe_num(s: pd.Series | object, index: pd.Index | None = None) -> pd.Series:
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    return pd.Series(s, index=index, dtype="float64")


# ---------------------------------------------------------------------------
# Support-zone upgrades
# ---------------------------------------------------------------------------


def extract_confirmed_swing_points(bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    swings = confirmed_swing_lows(bars, left=int(args.pivot_left), right=int(args.pivot_right))
    if swings.empty or "swing_low" not in swings:
        return pd.DataFrame(columns=["confirmed_time", "swing_low", "swing_low_pos"])
    tmp = swings.copy()
    changed = tmp["swing_low_pos"].notna() & (tmp["swing_low_pos"] != tmp["swing_low_pos"].shift(1))
    pts = tmp.loc[changed, ["swing_low", "swing_low_pos"]].copy()
    pts["confirmed_time"] = pts.index
    pts = pts.dropna(subset=["swing_low", "swing_low_pos"]).reset_index(drop=True)
    return pts[["confirmed_time", "swing_low", "swing_low_pos"]]


def attach_support_zone_metrics(events: pd.DataFrame, bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = events.copy().sort_values("signal_time").reset_index(drop=True)
    pivots = extract_confirmed_swing_points(bars, args)
    if pivots.empty:
        out["cluster_touch_count_020"] = 1
        out["cluster_touch_count_030"] = 1
        out["cluster_zone_low_020"] = out.get("swing_level", np.nan)
        out["cluster_zone_low_030"] = out.get("swing_level", np.nan)
        out["cluster_oldest_age_bars_020"] = out.get("swing_age", np.nan)
        out["cluster_oldest_age_bars_030"] = out.get("swing_age", np.nan)
        return out

    tf_minutes = _timeframe_to_minutes(str(args.timeframe))
    progress = ProgressReporter(label="[support] equal-low cluster metrics", total=len(out), every=max(1, int(args.progress_every)), enabled=not bool(args.no_progress))
    counts_by_tol: dict[str, list[int]] = {}
    lows_by_tol: dict[str, list[float]] = {}
    ages_by_tol: dict[str, list[float]] = {}
    tols = _parse_float_list(args.cluster_tolerance_pcts)
    for tol in tols:
        tag = f"{int(round(tol * 10000)):04d}"
        counts_by_tol[tag] = []
        lows_by_tol[tag] = []
        ages_by_tol[tag] = []

    p_times = pd.to_datetime(pivots["confirmed_time"])
    p_prices = pd.to_numeric(pivots["swing_low"], errors="coerce")
    p_pos = pd.to_numeric(pivots["swing_low_pos"], errors="coerce")
    for i, row in out.iterrows():
        ts = pd.Timestamp(row["signal_time"])
        level = float(row.get("swing_level", np.nan))
        sig_pos = bars.index.get_indexer([ts])[0] if ts in bars.index else np.nan
        lookback_start = ts - pd.Timedelta(minutes=int(args.cluster_lookback_bars) * tf_minutes)
        time_mask = (p_times < ts) & (p_times >= lookback_start)
        for tol in tols:
            tag = f"{int(round(tol * 10000)):04d}"
            if not math.isfinite(level) or level <= 0:
                mask = pd.Series(False, index=pivots.index)
            else:
                mask = time_mask & (p_prices >= level * (1.0 - tol)) & (p_prices <= level * (1.0 + tol))
            cnt = int(mask.sum())
            counts_by_tol[tag].append(cnt)
            lows_by_tol[tag].append(float(p_prices[mask].min()) if cnt else np.nan)
            if cnt and math.isfinite(float(sig_pos)):
                ages_by_tol[tag].append(float(np.nanmax(float(sig_pos) - p_pos[mask].to_numpy(dtype=float))))
            else:
                ages_by_tol[tag].append(np.nan)
        progress.update(i + 1)
    progress.close()

    for tag in counts_by_tol:
        out[f"cluster_touch_count_{tag}"] = counts_by_tol[tag]
        out[f"cluster_zone_low_{tag}"] = lows_by_tol[tag]
        out[f"cluster_oldest_age_bars_{tag}"] = ages_by_tol[tag]
    return out


# ---------------------------------------------------------------------------
# Candidate layers
# ---------------------------------------------------------------------------


def _bool_col(events: pd.DataFrame, col: str) -> pd.Series:
    if col not in events.columns:
        return pd.Series(False, index=events.index)
    return events[col].fillna(False).astype(bool)


def _first_existing_numeric(events: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    for col in candidates:
        if col in events.columns:
            return _safe_num(events[col], events.index)
    return pd.Series(np.nan, index=events.index, dtype="float64")


def _any_range_delta_pressure(events: pd.DataFrame) -> pd.Series:
    cols = [c for c in events.columns if c.startswith("range_") and c.endswith("_delta_pressure") and "large" not in c]
    if not cols:
        return pd.Series(np.nan, index=events.index, dtype="float64")
    frame = events[cols].apply(pd.to_numeric, errors="coerce")
    # Use the closed range-bar context with the strongest absolute pressure.
    idx = frame.abs().idxmax(axis=1)
    out = pd.Series(np.nan, index=events.index, dtype="float64")
    for col in cols:
        mask = idx.eq(col)
        out.loc[mask] = frame.loc[mask, col]
    return out


def build_candidate_layer_masks(events: pd.DataFrame, args: argparse.Namespace) -> dict[str, pd.Series]:
    fixed = build_fixed_candidate_masks(events)
    a = fixed.get("A_spike_close_large_share", {}).get("mask", pd.Series(False, index=events.index)).fillna(False).astype(bool)
    b = fixed.get("B_session_spike_atr", {}).get("mask", pd.Series(False, index=events.index)).fillna(False).astype(bool)
    spike = _safe_num(events.get("down_spike_pct", np.nan), events.index)
    close_pos = _safe_num(events.get("close_pos_in_bar", np.nan), events.index)
    large_q75 = _bool_col(events, "large_share_rq75_90d") | _bool_col(events, "large_share_rq75_30d")
    large_q80 = _bool_col(events, "large_share_rq80_90d") | _bool_col(events, "large_share_rq80_30d")
    cluster2 = _safe_num(events.get("cluster_touch_count_020", 0), events.index) >= 2
    cluster3 = _safe_num(events.get("cluster_touch_count_030", 0), events.index) >= 3
    aged12 = _safe_num(events.get("swing_age", np.nan), events.index) >= 12
    variant_close = events.get("variant", "").astype("object").eq("fade_close_through") if "variant" in events.columns else pd.Series(True, index=events.index)

    fp_low_delta = _first_existing_numeric(events, ["fp_low_bucket_delta_pressure"] + [c for c in events.columns if c.endswith("fp_low_bucket_delta_pressure")])
    # Loader currently emits this exact name for footprint context, but keep the
    # generic suffix fallback so future r/step-prefixed fields still work.
    if fp_low_delta.isna().all():
        fp_low_delta = _first_existing_numeric(events, [c for c in events.columns if c.endswith("low_bucket_delta_pressure") and c.startswith("fp_")])
    fp_abs_delta = _first_existing_numeric(events, ["fp_max_bucket_abs_delta_pressure"] + [c for c in events.columns if c.endswith("fp_max_bucket_abs_delta_pressure")])
    if fp_abs_delta.isna().all():
        fp_abs_delta = _first_existing_numeric(events, [c for c in events.columns if c.endswith("max_bucket_abs_delta_pressure") and c.startswith("fp_")])
    range_delta = _any_range_delta_pressure(events)
    fp_low_vneg = fp_low_delta <= float(args.fp_low_delta_vneg_threshold)
    fp_abs_high = fp_abs_delta >= float(args.fp_abs_delta_high_threshold)
    range_mild_neg = (range_delta >= float(args.range_delta_mild_neg_min)) & (range_delta <= float(args.range_delta_mild_neg_max))
    micro_last = int(getattr(args, "micro_last_seconds", 20))
    micro5_ratio = _safe_num(events.get(f"micro_5s_last{micro_last}_buy_sell_ratio", np.nan), events.index)
    micro10_ratio = _safe_num(events.get(f"micro_10s_last{micro_last}_buy_sell_ratio", np.nan), events.index)
    micro5_delta = _safe_num(events.get(f"micro_5s_last{micro_last}_delta_pressure", np.nan), events.index)
    micro10_delta = _safe_num(events.get(f"micro_10s_last{micro_last}_delta_pressure", np.nan), events.index)
    micro5_large_delta = _safe_num(events.get(f"micro_5s_last{micro_last}_large_delta_pressure", np.nan), events.index)
    micro10_large_delta = _safe_num(events.get(f"micro_10s_last{micro_last}_large_delta_pressure", np.nan), events.index)
    micro5_min_low = _safe_num(events.get(f"micro_5s_last{micro_last}_min_low", np.nan), events.index)
    micro10_min_low = _safe_num(events.get(f"micro_10s_last{micro_last}_min_low", np.nan), events.index)
    signal_low = _safe_num(events.get("low", events.get("signal_low", np.nan)), events.index)
    micro5_buy_pressure = micro5_ratio >= float(args.micro_buy_sell_ratio_min)
    micro10_buy_pressure = micro10_ratio >= float(args.micro_buy_sell_ratio_min)
    micro5_sell_exhaustion = micro5_delta >= float(args.micro_sell_exhaustion_delta_min)
    micro10_sell_exhaustion = micro10_delta >= float(args.micro_sell_exhaustion_delta_min)
    micro5_large_buy = micro5_large_delta >= float(args.micro_large_delta_min)
    micro10_large_buy = micro10_large_delta >= float(args.micro_large_delta_min)
    micro_no_low_buffer = float(args.micro_no_new_low_buffer_pct)
    micro5_no_new_low = micro5_min_low >= signal_low * (1.0 + micro_no_low_buffer)
    micro10_no_new_low = micro10_min_low >= signal_low * (1.0 + micro_no_low_buffer)

    a0_fp_abs = a & (spike >= 0.0100) & fp_abs_high
    a1_only_fp_abs = a & (spike < 0.0100) & fp_abs_high

    soft_base = variant_close & (spike >= float(args.soft_spike_pct)) & (close_pos <= float(args.soft_close_pos)) & large_q75
    soft_zone = soft_base & (cluster2 | aged12)
    soft_reclaim = soft_base & cluster2
    masks = {
        "A1_current": a,
        "A0_spike_ge_0100": a & (spike >= 0.0100),
        "A0_deep_spike_ge_0120": a & (spike >= 0.0120),
        # v2 avoids the earlier impossible/overly-strict version.  It is still
        # not a blind loosen: a soft spike needs an equal-low/aged liquidity zone.
        "A2_soft_zone": soft_zone,
        "A2_soft_zone_v2": soft_zone,
        "A2_reclaim_zone_v2": soft_reclaim,
        "A2_soft_large80": variant_close & (spike >= float(args.soft_spike_pct)) & (close_pos <= float(args.soft_close_pos)) & large_q80,
        "B_aux_session_spike_atr": b,
        # Fixed, interpretable context filters converted from the strongest
        # range/footprint breakdown buckets. These are still research-only and
        # causal because attached context bars are closed before signal_time.
        "A1_fp_low_delta_vneg": a & fp_low_vneg,
        "A0_fp_low_delta_vneg": a & (spike >= 0.0100) & fp_low_vneg,
        "A1_fp_abs_delta_high": a & fp_abs_high,
        "A0_fp_abs_delta_high": a & (spike >= 0.0100) & fp_abs_high,
        # Incremental A1-only layers answer the key portfolio question: after
        # reserving A0 (>=1.0% panic spike) as the main line, does the 0.8%-1.0%
        # slice provide useful extra trades or just dilute the cleaner A0 edge?
        "A1_only_080_100": a & (spike < 0.0100),
        "A1_only_fp_abs_delta_high": a1_only_fp_abs,
        "A1_only_fp_low_delta_vneg": a & (spike < 0.0100) & fp_low_vneg,
        "A1_range_delta_mild_neg": a & range_mild_neg,
        "A0_range_delta_mild_neg": a & (spike >= 0.0100) & range_mild_neg,
        "A0_micro5_last20_buy_pressure": a & (spike >= 0.0100) & micro5_buy_pressure,
        "A0_micro10_last20_buy_pressure": a & (spike >= 0.0100) & micro10_buy_pressure,
        # V2 micro-confirmed variants keep the parent 1m A0/A1 event definition.
        # Micro bars are used only as confirmation inside the already-closed
        # signal bar, not to redefine the 1m down-spike itself.
        "A0_fp_abs_delta_high_micro5_buy_pressure": a0_fp_abs & micro5_buy_pressure,
        "A0_fp_abs_delta_high_micro10_buy_pressure": a0_fp_abs & micro10_buy_pressure,
        "A0_fp_abs_delta_high_micro5_sell_exhaustion": a0_fp_abs & micro5_sell_exhaustion,
        "A0_fp_abs_delta_high_micro10_sell_exhaustion": a0_fp_abs & micro10_sell_exhaustion,
        "A0_fp_abs_delta_high_micro5_no_new_low": a0_fp_abs & micro5_no_new_low,
        "A0_fp_abs_delta_high_micro10_no_new_low": a0_fp_abs & micro10_no_new_low,
        "A0_fp_abs_delta_high_micro5_combo": a0_fp_abs & micro5_buy_pressure & micro5_no_new_low,
        "A0_fp_abs_delta_high_micro10_combo": a0_fp_abs & micro10_buy_pressure & micro10_no_new_low,
        "A0_fp_abs_delta_high_micro5_large_buy": a0_fp_abs & micro5_large_buy,
        "A0_fp_abs_delta_high_micro10_large_buy": a0_fp_abs & micro10_large_buy,
        "A1_only_fp_abs_delta_high_micro5_buy_pressure": a1_only_fp_abs & micro5_buy_pressure,
        "A1_only_fp_abs_delta_high_micro10_buy_pressure": a1_only_fp_abs & micro10_buy_pressure,
        "A1_only_fp_abs_delta_high_micro5_sell_exhaustion": a1_only_fp_abs & micro5_sell_exhaustion,
        "A1_only_fp_abs_delta_high_micro10_sell_exhaustion": a1_only_fp_abs & micro10_sell_exhaustion,
        "A1_only_fp_abs_delta_high_micro5_no_new_low": a1_only_fp_abs & micro5_no_new_low,
        "A1_only_fp_abs_delta_high_micro10_no_new_low": a1_only_fp_abs & micro10_no_new_low,
        "A1_only_fp_abs_delta_high_micro5_combo": a1_only_fp_abs & micro5_buy_pressure & micro5_no_new_low,
        "A1_only_fp_abs_delta_high_micro10_combo": a1_only_fp_abs & micro10_buy_pressure & micro10_no_new_low,
    }
    return masks

def build_support_mask(events: pd.DataFrame, mode: str, args: argparse.Namespace) -> pd.Series:
    mode = str(mode)
    if mode == "single_swing":
        return pd.Series(True, index=events.index)
    m = re.fullmatch(r"equal(\d+)_(\d{3,4})", mode)
    if m:
        need = int(m.group(1))
        tag = m.group(2).zfill(4)
        col = f"cluster_touch_count_{tag}"
        return _safe_num(events.get(col, 0), events.index) >= need
    m = re.fullmatch(r"aged_ge_(\d+)", mode)
    if m:
        return _safe_num(events.get("swing_age", np.nan), events.index) >= float(int(m.group(1)))
    m = re.fullmatch(r"equal2_020_or_aged(\d+)", mode)
    if m:
        return (_safe_num(events.get("cluster_touch_count_020", 0), events.index) >= 2) | (_safe_num(events.get("swing_age", np.nan), events.index) >= float(int(m.group(1))))
    raise ValueError(f"Unknown support_mode: {mode}")


# ---------------------------------------------------------------------------
# Optional range / footprint context
# ---------------------------------------------------------------------------


def _merge_asof_event_context(events: pd.DataFrame, ctx: pd.DataFrame, prefix: str, columns: Sequence[str]) -> pd.DataFrame:
    if events.empty or ctx.empty:
        return events
    left = events.sort_values("signal_time").copy()
    right = ctx.copy().sort_index()
    right = right[~right.index.duplicated(keep="last")]
    use_cols = [c for c in columns if c in right.columns]
    if not use_cols:
        return events
    r = right[use_cols].copy().add_prefix(prefix)
    r["ctx_time"] = r.index
    r = r.reset_index(drop=True).sort_values("ctx_time")
    merged = pd.merge_asof(
        left,
        r,
        left_on="signal_time",
        right_on="ctx_time",
        direction="backward",
        allow_exact_matches=True,
    )
    merged = merged.drop(columns=["ctx_time"], errors="ignore")
    return merged.sort_values("signal_time").reset_index(drop=True)


def attach_range_context(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = events.copy()
    if "range_bar" not in set(_split_csv_names(args.context_sources)):
        return out
    for rp in _parse_float_list(args.range_pcts):
        tag = range_code(float(rp))
        try:
            print(f"[context] loading range bars {tag}", flush=True)
            rb = OKXRangeBarLoader(symbol=args.symbol, range_pct=float(rp)).fetch_data_by_date_range(args.warmup_start_date, args.end_date)
            if rb.empty:
                print(f"[context] range {tag} empty; skip", flush=True)
                continue
            rb = rb.sort_index()
            rb[f"range_{tag}_delta_pressure"] = pd.to_numeric(rb.get("delta_notional", np.nan), errors="coerce") / (pd.to_numeric(rb.get("buy_notional", np.nan), errors="coerce") + pd.to_numeric(rb.get("sell_notional", np.nan), errors="coerce")).replace(0.0, np.nan)
            rb[f"range_{tag}_large_delta_pressure"] = pd.to_numeric(rb.get("large_delta_notional", np.nan), errors="coerce") / (pd.to_numeric(rb.get("large_buy_notional", np.nan), errors="coerce") + pd.to_numeric(rb.get("large_sell_notional", np.nan), errors="coerce")).replace(0.0, np.nan)
            cols = ["direction", "duration_seconds", "range_pct", "taker_buy_ratio", f"range_{tag}_delta_pressure", f"range_{tag}_large_delta_pressure"]
            rename = {"direction": f"range_{tag}_direction", "duration_seconds": f"range_{tag}_duration_seconds", "range_pct": f"range_{tag}_range_pct", "taker_buy_ratio": f"range_{tag}_taker_buy_ratio"}
            rb2 = rb.rename(columns=rename)
            out = _merge_asof_event_context(out, rb2, "", [rename.get(c, c) for c in cols])
        except Exception as exc:  # pragma: no cover - depends on local cache availability
            print(f"[context] range {tag} skipped: {exc}", flush=True)
    return out


def attach_footprint_context(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = events.copy()
    if "footprint" not in set(_split_csv_names(args.context_sources)):
        return out
    tag = range_code(float(args.footprint_range_pct))
    step_tag = price_step_code(float(args.footprint_price_step))
    try:
        print(f"[context] loading footprint {tag}_{step_tag}", flush=True)
        fp = OKXRangeFootprintLoader(symbol=args.symbol, range_pct=float(args.footprint_range_pct), price_step=float(args.footprint_price_step)).fetch_data_by_date_range(args.warmup_start_date, args.end_date)
        if fp.empty:
            print("[context] footprint empty; skip", flush=True)
            return out
        # Aggregate only closed range-bar footprints; index is end_ts from loader.
        fp = fp.copy().sort_index()
        denom = (pd.to_numeric(fp.get("buy_notional", np.nan), errors="coerce") + pd.to_numeric(fp.get("sell_notional", np.nan), errors="coerce")).replace(0.0, np.nan)
        fp["bucket_delta_pressure"] = pd.to_numeric(fp.get("delta_notional", np.nan), errors="coerce") / denom
        agg = fp.groupby("bar_id", dropna=False).agg(
            end_ts=("end_ts", "last") if "end_ts" in fp.columns else ("bucket_delta_pressure", "last"),
            fp_notional=("notional", "sum"),
            fp_delta_notional=("delta_notional", "sum"),
            fp_max_bucket_abs_delta_pressure=("bucket_delta_pressure", lambda s: float(pd.to_numeric(s, errors="coerce").abs().max())),
            fp_low_bucket_delta_pressure=("bucket_delta_pressure", "first"),
            fp_high_bucket_delta_pressure=("bucket_delta_pressure", "last"),
        ).reset_index()
        if "end_ts" not in agg or pd.to_datetime(agg["end_ts"], errors="coerce").isna().all():
            return out
        agg["end_ts"] = pd.to_datetime(agg["end_ts"])
        agg = agg.set_index("end_ts").sort_index()
        denom2 = pd.to_numeric(agg["fp_notional"], errors="coerce").replace(0.0, np.nan)
        agg[f"fp_{tag}_{step_tag}_delta_pressure"] = pd.to_numeric(agg["fp_delta_notional"], errors="coerce") / denom2
        out = _merge_asof_event_context(
            out,
            agg,
            "",
            [f"fp_{tag}_{step_tag}_delta_pressure", "fp_max_bucket_abs_delta_pressure", "fp_low_bucket_delta_pressure", "fp_high_bucket_delta_pressure"],
        )
    except Exception as exc:  # pragma: no cover - depends on local cache availability
        print(f"[context] footprint skipped: {exc}", flush=True)
    return out


def _seconds_from_timeframe(tf: str) -> int | None:
    m = re.fullmatch(r"(\d+)s", str(tf).strip())
    if m:
        return int(m.group(1))
    m = re.fullmatch(r"(\d+)m", str(tf).strip())
    if m:
        return int(m.group(1)) * 60
    return None


def _month_start(ts: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp()


def _next_month(ts: pd.Timestamp) -> pd.Timestamp:
    return _month_start(ts) + pd.offsets.MonthBegin(1)


def _month_end_inclusive(month_start: pd.Timestamp, tf_seconds: int) -> pd.Timestamp:
    """Last possible left-labeled micro bar timestamp inside ``month_start``."""
    return _next_month(month_start) - pd.Timedelta(seconds=max(1, int(tf_seconds)))


def _load_micro_trade_bar_month(
    loader: OKXTradeBarLoader,
    month_start: pd.Timestamp,
    tf_seconds: int,
    *,
    load_mode: str,
    max_exclusive_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Load one local month of cached/prebuilt micro bars.

    This is intentionally month-sized, not full-range.  ``local`` is cache-only
    and never triggers raw trade aggregation.  ``fetch`` is also safe when the
    user has prebuilt coverage and wants the loader's normal range interface,
    but it is still called one month at a time so it cannot materialize the
    2022-2026 second-level table in memory.
    """
    start = _month_start(month_start)
    end = _month_end_inclusive(start, tf_seconds)
    max_exclusive = pd.Timestamp(max_exclusive_ts) if max_exclusive_ts is not None and pd.notna(max_exclusive_ts) else None

    # Guard against post-backtest micro loading.  The sliding cache may keep
    # the next calendar month in memory for month-end continuity, but it must
    # never fetch/download bars after the last timestamp that can affect the
    # requested signal set.  If the next month starts at or after the exclusive
    # micro feature horizon, it is not needed at all.
    if max_exclusive is not None and start >= max_exclusive:
        return pd.DataFrame()

    if max_exclusive is not None:
        capped_end = min(end, max_exclusive - pd.Timedelta(seconds=max(1, int(tf_seconds))))
        if capped_end < start:
            return pd.DataFrame()
        end = capped_end

    mode = str(load_mode or "local").lower()
    if mode == "fetch":
        return loader.fetch_data_by_date_range(start, end, cvd_mode="range")
    local = loader.load_local_data(start_date=start, end_date=end)
    if mode == "auto" and (local is None or local.empty):
        return loader.fetch_data_by_date_range(start, end, cvd_mode="range")
    return local


def _coerce_micro_datetime(values: object) -> pd.DatetimeIndex:
    """Coerce loader timestamps to a naive DatetimeIndex.

    Different OKX trade-bar loaders may return cached rows with a DatetimeIndex,
    a string/object index, or an explicit timestamp-like column.  A previous V2
    draft loaded millions of 5s/10s rows correctly, but attached zero micro
    features because the matching function assumed the DataFrame index was
    already a clean DatetimeIndex.  This helper makes the matching layer robust
    without changing the causal slice: every feature still uses only the closed
    parent 1m signal bar interval.
    """
    ser = pd.Series(values)
    if pd.api.types.is_numeric_dtype(ser):
        finite = pd.to_numeric(ser, errors="coerce")
        med = float(finite.dropna().median()) if finite.notna().any() else np.nan
        if math.isfinite(med) and med > 1e17:
            dt = pd.to_datetime(finite, unit="ns", errors="coerce")
        elif math.isfinite(med) and med > 1e12:
            dt = pd.to_datetime(finite, unit="ms", errors="coerce")
        elif math.isfinite(med) and med > 1e9:
            dt = pd.to_datetime(finite, unit="s", errors="coerce")
        else:
            dt = pd.to_datetime(finite, errors="coerce")
    else:
        dt = pd.to_datetime(ser, errors="coerce")
    try:
        if getattr(dt.dt, "tz", None) is not None:
            dt = dt.dt.tz_convert(None)
    except Exception:
        pass
    return pd.DatetimeIndex(dt).astype("datetime64[ns]")


def _normalize_micro_bar_frame(mb: pd.DataFrame) -> pd.DataFrame:
    """Return micro bars indexed by timestamp, sorted and de-duplicated."""
    if mb is None or mb.empty:
        return pd.DataFrame()
    df = mb.copy()

    idx = _coerce_micro_datetime(df.index)
    # RangeIndex -> 1970 ns is not a real market timestamp; prefer explicit
    # timestamp columns whenever the index did not produce plausible dates.
    plausible = idx.notna() & (idx >= pd.Timestamp("2015-01-01")) & (idx <= pd.Timestamp("2035-01-01"))
    if plausible.sum() < max(1, int(len(df) * 0.5)):
        time_cols = [
            "timestamp", "ts", "datetime", "date_time", "time", "open_time",
            "start_time", "bar_start", "begin_time", "created_at", "ts_ms",
            "timestamp_ms", "timestamp_ns",
        ]
        for col in time_cols:
            if col in df.columns:
                cand = _coerce_micro_datetime(df[col])
                cand_plausible = cand.notna() & (cand >= pd.Timestamp("2015-01-01")) & (cand <= pd.Timestamp("2035-01-01"))
                if cand_plausible.sum() >= max(1, int(len(df) * 0.5)):
                    idx = cand
                    plausible = cand_plausible
                    break

    if plausible.sum() == 0:
        return pd.DataFrame()
    df = df.loc[np.asarray(plausible)].copy()
    df.index = pd.DatetimeIndex(idx[np.asarray(plausible)])
    df = df.sort_index(kind="stable")
    df = df[~df.index.duplicated(keep="last")]
    return df


def _compute_micro_rows_for_positions(
    *,
    mb: pd.DataFrame,
    signal_times: pd.Series,
    positions: Sequence[int],
    primary_delta: pd.Timedelta,
    last_seconds: int,
    tag: str,
    rows: list[dict[str, float]],
    progress: ProgressReporter,
    debug_rows: list[dict[str, object]] | None = None,
    debug_limit: int = 0,
    source_month: pd.Timestamp | None = None,
) -> None:
    """Attach micro features for the requested signal row positions.

    Loading a future month into memory does not make the calculation non-causal:
    every row still slices only the already-closed parent 1m signal bar interval
    ``[signal_time + 1m - last_seconds, signal_time + 1m)``.
    """
    prefix = f"micro_{tag}_last{last_seconds}_"
    if mb is None or mb.empty:
        for pos in positions:
            pos_i = int(pos)
            sig = pd.Timestamp(signal_times.iloc[pos_i])
            start = sig
            end = start + primary_delta
            last_start = max(start, end - pd.Timedelta(seconds=last_seconds))
            rows[pos_i] = {
                f"{prefix}matched_rows": 0,
                f"{prefix}window_start": last_start,
                f"{prefix}window_end_exclusive": end,
            }
            if debug_rows is not None and len(debug_rows) < int(debug_limit or 0):
                debug_rows.append({
                    "timeframe": tag,
                    "source_month": pd.Timestamp(source_month).strftime("%Y-%m") if source_month is not None and pd.notna(source_month) else "",
                    "event_row_pos": pos_i,
                    "signal_time": sig,
                    "window_start": last_start,
                    "window_end_exclusive": end,
                    "cache_rows": 0,
                    "matched_rows": 0,
                    "empty_reason": "month_cache_empty",
                })
            progress.step()
        return

    mb = _normalize_micro_bar_frame(mb)
    if mb.empty:
        for pos in positions:
            pos_i = int(pos)
            sig = pd.Timestamp(signal_times.iloc[pos_i])
            start = sig
            end = start + primary_delta
            last_start = max(start, end - pd.Timedelta(seconds=last_seconds))
            rows[pos_i] = {
                f"{prefix}matched_rows": 0,
                f"{prefix}window_start": last_start,
                f"{prefix}window_end_exclusive": end,
            }
            if debug_rows is not None and len(debug_rows) < int(debug_limit or 0):
                debug_rows.append({
                    "timeframe": tag,
                    "source_month": pd.Timestamp(source_month).strftime("%Y-%m") if source_month is not None and pd.notna(source_month) else "",
                    "event_row_pos": pos_i,
                    "signal_time": sig,
                    "window_start": last_start,
                    "window_end_exclusive": end,
                    "cache_rows": 0,
                    "matched_rows": 0,
                    "empty_reason": "normalized_cache_empty",
                })
            progress.step()
        return

    # Pandas 2.x can preserve DatetimeIndex units such as datetime64[s].
    # Timestamp.value is always nanoseconds; searching a seconds-based int view
    # makes every event window look after the cache and yields zero matches.
    # Force ns here before using integer searchsorted.
    mb_idx = pd.DatetimeIndex(pd.to_datetime(mb.index, errors="coerce")).astype("datetime64[ns]")
    mb = mb.copy()
    mb.index = mb_idx
    idx_values = mb_idx.view("int64")

    def _append_micro_debug(pos_i: int, start: pd.Timestamp, end: pd.Timestamp, last_start: pd.Timestamp, lo: int, hi: int) -> None:
        if debug_rows is None or len(debug_rows) >= int(debug_limit or 0):
            return
        prev_ts = mb_idx[lo - 1] if lo > 0 and len(mb_idx) else pd.NaT
        next_ts = mb_idx[lo] if lo < len(mb_idx) else pd.NaT
        try:
            gap_to_next_sec = float((pd.Timestamp(next_ts) - last_start).total_seconds()) if pd.notna(next_ts) else np.nan
        except Exception:
            gap_to_next_sec = np.nan
        debug_rows.append({
            "timeframe": tag,
            "source_month": pd.Timestamp(source_month).strftime("%Y-%m") if source_month is not None and pd.notna(source_month) else "",
            "event_row_pos": int(pos_i),
            "signal_time": pd.Timestamp(signal_times.iloc[pos_i]),
            "window_start": last_start,
            "window_end_exclusive": end,
            "cache_rows": int(len(mb_idx)),
            "cache_first_ts": mb_idx[0] if len(mb_idx) else pd.NaT,
            "cache_last_ts": mb_idx[-1] if len(mb_idx) else pd.NaT,
            "cache_index_dtype": str(mb_idx.dtype),
            "lo": int(lo),
            "hi": int(hi),
            "matched_rows": int(max(0, hi - lo)),
            "prev_cache_ts": prev_ts,
            "next_cache_ts": next_ts,
            "gap_to_next_sec": gap_to_next_sec,
        })

    notional = pd.to_numeric(mb.get("notional", np.nan), errors="coerce")
    if notional.isna().all():
        notional = pd.to_numeric(mb.get("buy_notional", np.nan), errors="coerce") + pd.to_numeric(mb.get("sell_notional", np.nan), errors="coerce")
    buy = pd.to_numeric(mb.get("buy_notional", np.nan), errors="coerce")
    sell = pd.to_numeric(mb.get("sell_notional", np.nan), errors="coerce")
    close = pd.to_numeric(mb.get("close", np.nan), errors="coerce")
    high = pd.to_numeric(mb.get("high", np.nan), errors="coerce")
    low = pd.to_numeric(mb.get("low", np.nan), errors="coerce")
    large_buy = pd.to_numeric(mb.get("large_buy_notional", np.nan), errors="coerce")
    large_sell = pd.to_numeric(mb.get("large_sell_notional", np.nan), errors="coerce")
    max_trade = pd.to_numeric(mb.get("max_trade_notional", np.nan), errors="coerce")

    for pos in positions:
        pos_i = int(pos)
        sig = pd.Timestamp(signal_times.iloc[pos_i])
        start = sig
        end = start + primary_delta
        last_start = max(start, end - pd.Timedelta(seconds=last_seconds))
        lo = np.searchsorted(idx_values, last_start.value, side="left")
        hi = np.searchsorted(idx_values, end.value, side="left")
        if hi <= lo:
            _append_micro_debug(pos_i, start, end, last_start, int(lo), int(hi))
            rows[pos_i] = {
                f"{prefix}matched_rows": 0,
                f"{prefix}window_start": last_start,
                f"{prefix}window_end_exclusive": end,
            }
            progress.step()
            continue
        sl = slice(lo, hi)
        nsum = float(notional.iloc[sl].sum(skipna=True))
        bsum = float(buy.iloc[sl].sum(skipna=True))
        ssum = float(sell.iloc[sl].sum(skipna=True))
        lbuy = float(large_buy.iloc[sl].sum(skipna=True))
        lsell = float(large_sell.iloc[sl].sum(skipna=True))
        last_close = float(close.iloc[hi - 1]) if hi - 1 < len(close) and pd.notna(close.iloc[hi - 1]) else np.nan
        min_low = float(low.iloc[sl].min(skipna=True))
        max_high = float(high.iloc[sl].max(skipna=True))
        _append_micro_debug(pos_i, start, end, last_start, int(lo), int(hi))
        rows[pos_i] = {
            f"{prefix}matched_rows": int(hi - lo),
            f"{prefix}window_start": last_start,
            f"{prefix}window_end_exclusive": end,
            f"{prefix}first_micro_ts": mb_idx[lo],
            f"{prefix}last_micro_ts": mb_idx[hi - 1],
            f"{prefix}notional": nsum,
            f"{prefix}buy_sell_ratio": bsum / ssum if ssum > 0 else np.nan,
            f"{prefix}buy_ratio": bsum / nsum if nsum > 0 else np.nan,
            f"{prefix}delta_pressure": (bsum - ssum) / nsum if nsum > 0 else np.nan,
            f"{prefix}large_delta_pressure": (lbuy - lsell) / (lbuy + lsell) if (lbuy + lsell) > 0 else np.nan,
            f"{prefix}max_trade_notional": float(max_trade.iloc[sl].max(skipna=True)),
            f"{prefix}last_close": last_close,
            f"{prefix}min_low": min_low,
            f"{prefix}max_high": max_high,
        }
        progress.step()


def attach_micro_trade_context(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = events.copy()
    micro_tfs = _split_csv_names(getattr(args, "micro_timeframes", ""))
    if out.empty or not micro_tfs:
        return out
    if str(getattr(args, "timeframe", "1m")) != "1m":
        print("[context] micro trade-bar context currently attaches only to 1m primary bars; skip", flush=True)
        return out

    signal_times = pd.to_datetime(out["signal_time"]).reset_index(drop=True)
    if signal_times.empty:
        return out
    last_seconds = int(getattr(args, "micro_last_seconds", 20))
    tf_minutes = _timeframe_to_minutes(str(args.timeframe))
    primary_delta = pd.Timedelta(minutes=tf_minutes)
    load_mode = str(getattr(args, "micro_load_mode", "local") or "local").lower()
    if not hasattr(args, "_micro_load_stats"):
        setattr(args, "_micro_load_stats", [])
    if not hasattr(args, "_micro_match_debug"):
        setattr(args, "_micro_match_debug", [])

    # Exclusive upper bound for micro features.  This is based on the actual
    # signal set, not wall-clock date, so a backtest ending on 2026-06-30 cannot
    # trigger 2026-07 raw tick downloads just because the current-month +
    # next-month cache sees July as a boundary month.
    max_signal_end = pd.Timestamp(signal_times.max()) + primary_delta
    arg_end = pd.to_datetime(getattr(args, "end_date", None), errors="coerce")
    if pd.notna(arg_end):
        # End date arguments are date-like and inclusive for the primary 1m
        # bars.  Convert to the exclusive timestamp immediately after that day.
        arg_end_exclusive = pd.Timestamp(arg_end).normalize() + pd.Timedelta(days=1)
        max_micro_exclusive_ts = min(max_signal_end, arg_end_exclusive)
    else:
        max_micro_exclusive_ts = max_signal_end

    # Group signals by their signal month, then process in time order with a
    # sliding current-month + next-month cache.  This keeps month-end windows
    # continuous without loading all 5s/10s bars into memory.  The next month may
    # be resident in RAM, but each feature row still slices only the closed
    # parent 1m signal bar, so this is not a future-function path.
    month_for_row = signal_times.map(_month_start)
    months = sorted(pd.Timestamp(m) for m in pd.Series(month_for_row).dropna().unique())
    row_positions_by_month: dict[pd.Timestamp, list[int]] = {}
    for i, m in enumerate(month_for_row):
        if pd.isna(m):
            continue
        row_positions_by_month.setdefault(pd.Timestamp(m), []).append(int(i))

    for tf in micro_tfs:
        tf_seconds = _seconds_from_timeframe(tf)
        if tf_seconds is None or tf_seconds <= 0 or tf_seconds >= tf_minutes * 60:
            print(f"[context] micro {tf} skipped: unsupported micro timeframe for 1m primary", flush=True)
            continue
        tag = str(tf).lower()
        loader = OKXTradeBarLoader(symbol=args.symbol, timeframe=str(tf), data_dir=getattr(args, "data_dir", None))
        rows: list[dict[str, float]] = [{} for _ in range(len(signal_times))]
        month_cache: dict[pd.Timestamp, pd.DataFrame] = {}
        print(f"[context] loading micro trade bars {tag} monthly sliding cache mode={load_mode}", flush=True)

        def get_month(month_start: pd.Timestamp) -> pd.DataFrame:
            m = _month_start(month_start)
            if m in month_cache:
                return month_cache[m]
            try:
                part = _load_micro_trade_bar_month(
                    loader,
                    m,
                    int(tf_seconds),
                    load_mode=load_mode,
                    max_exclusive_ts=max_micro_exclusive_ts,
                )
            except Exception as exc:  # pragma: no cover - local DB/schema dependent
                print(f"[context] micro {tag} month skipped {m.strftime('%Y-%m')}: {exc}", flush=True)
                part = pd.DataFrame()
            if part is None:
                part = pd.DataFrame()
            try:
                idx_min = part.index.min() if not part.empty else pd.NaT
                idx_max = part.index.max() if not part.empty else pd.NaT
                index_type = type(part.index).__name__ if part is not None else "None"
                normalized = _normalize_micro_bar_frame(part) if part is not None and not part.empty else pd.DataFrame()
                norm_min = normalized.index.min() if not normalized.empty else pd.NaT
                norm_max = normalized.index.max() if not normalized.empty else pd.NaT
            except Exception:
                idx_min = pd.NaT
                idx_max = pd.NaT
                index_type = "unknown"
                normalized = pd.DataFrame()
                norm_min = pd.NaT
                norm_max = pd.NaT
            getattr(args, "_micro_load_stats", []).append({
                "timeframe": tag,
                "month": m.strftime("%Y-%m"),
                "load_mode": load_mode,
                "max_micro_exclusive_ts": max_micro_exclusive_ts,
                "rows_loaded": int(len(part)),
                "normalized_rows": int(len(normalized)),
                "index_type": index_type,
                "first_ts": idx_min,
                "last_ts": idx_max,
                "normalized_first_ts": norm_min,
                "normalized_last_ts": norm_max,
                "columns": "|".join(map(str, list(part.columns)[:40])) if not part.empty else "",
                "cached_normalized": True,
            })
            # Cache the normalized DatetimeIndex frame, not the raw loader frame.
            # Diagnostics in V2 showed month-level data loaded correctly but event
            # windows matched zero rows; using one normalized representation for
            # both diagnostics and slicing removes index/column branch drift.
            month_cache[m] = normalized
            return normalized

        progress = ProgressReporter(
            label=f"[context] micro {tag} monthly signal rows",
            total=len(signal_times),
            every=max(1, int(getattr(args, "progress_every", 500))),
            enabled=not bool(getattr(args, "no_progress", False)),
        )
        for month in months:
            current_month = _month_start(month)
            next_month = _next_month(current_month)
            cur = get_month(current_month)
            nxt = get_month(next_month)
            parts = [p for p in [cur, nxt] if p is not None and not p.empty]
            if parts:
                mb = pd.concat(parts).sort_index(kind="stable")
                mb = mb[~mb.index.duplicated(keep="last")]
            else:
                mb = pd.DataFrame()
            _compute_micro_rows_for_positions(
                mb=mb,
                signal_times=signal_times,
                positions=row_positions_by_month.get(current_month, []),
                primary_delta=primary_delta,
                last_seconds=last_seconds,
                tag=tag,
                rows=rows,
                progress=progress,
                debug_rows=getattr(args, "_micro_match_debug", None),
                debug_limit=int(getattr(args, "micro_debug_sample_limit", 500)),
                source_month=current_month,
            )
            # Current month has been fully processed; retain the next month as
            # the next iteration's current cache and release older data.
            month_cache.pop(current_month, None)
        progress.close()

        mdf = pd.DataFrame(rows)
        if not mdf.empty:
            out = pd.concat([out.reset_index(drop=True), mdf.reset_index(drop=True)], axis=1)
    return out


# ---------------------------------------------------------------------------
# Entry / exit simulation
# ---------------------------------------------------------------------------


def _event_float(event: pd.Series | dict[str, object], key: str, default: float = np.nan) -> float:
    try:
        val = event.get(key, default)  # type: ignore[attr-defined]
    except AttributeError:
        val = default
    try:
        out = float(val)
    except (TypeError, ValueError):
        return float(default)
    return out if math.isfinite(out) else float(default)


def _compute_stop_price(event: pd.Series | dict[str, object], entry_price: float, stop_spec: StopSpec, args: argparse.Namespace) -> tuple[float | None, float | None, str]:
    if stop_spec.mode == "none":
        return None, None, "no_stop"
    if stop_spec.mode == "fixed_pct":
        pct = float(stop_spec.value or 0.0)
        return float(entry_price * (1.0 - pct)), pct, stop_spec.name
    if stop_spec.mode == "atr_mult":
        atr_pct = _event_float(event, "atr_pct", np.nan)
        if not math.isfinite(atr_pct) or atr_pct <= 0:
            atr_pct = float(args.min_atr_stop_pct)
        raw_pct = atr_pct * float(stop_spec.value or 1.0)
        pct = min(float(args.max_atr_stop_pct), max(float(args.min_atr_stop_pct), float(raw_pct)))
        return float(entry_price * (1.0 - pct)), pct, stop_spec.name
    if stop_spec.mode == "structural":
        raw_level = _event_float(event, "structural_stop_level", np.nan)
        buffer_pct = float(stop_spec.value or 0.0)
        min_pct = float(args.min_structural_stop_pct)
        fallback = float(entry_price * (1.0 - min_pct))
        if not math.isfinite(raw_level) or raw_level <= 0:
            return fallback, min_pct, f"{stop_spec.name}_fallback_min"
        stop_price = float(raw_level * (1.0 - buffer_pct))
        if stop_price >= entry_price:
            stop_price = fallback
        pct = max(0.0, float(1.0 - stop_price / entry_price))
        if pct < min_pct:
            stop_price = fallback
            pct = min_pct
        return float(stop_price), float(pct), stop_spec.name
    raise ValueError(f"Unsupported stop mode {stop_spec.mode!r}")


def _entry_from_mode(
    bars: pd.DataFrame,
    event: pd.Series | dict[str, object],
    signal_pos: int,
    entry_mode: str,
    args: argparse.Namespace,
    market: MarketCache | None = None,
) -> tuple[bool, int | None, float | None, str]:
    if market is None:
        market = build_market_cache(bars, args)
    opens = market.open
    highs = market.high
    lows = market.low
    closes = market.close
    volume = market.volume
    vol_base = market.reclaim_vol_base

    n = len(market.index)
    signal_pos = int(signal_pos)
    delay = int(args.entry_delay_bars)
    if entry_mode == "next_open":
        pos = signal_pos + delay
        if pos >= n:
            return False, None, None, "no_future_open"
        return True, int(pos), float(opens[pos]), "next_open"

    m = re.fullmatch(r"next_open_delay(\d+)", str(entry_mode))
    if m:
        extra_delay = int(m.group(1))
        pos = signal_pos + delay + extra_delay
        if pos >= n:
            return False, None, None, "no_future_open_delay"
        return True, int(pos), float(opens[pos]), f"next_open_delay{extra_delay}"

    m = re.fullmatch(r"reclaim_(swing|signal_close|signal_open)_(\d+)", str(entry_mode))
    if m:
        target_kind = m.group(1)
        valid_bars = int(m.group(2))
        if target_kind == "swing":
            target = float(event.get("swing_level", np.nan))
        elif target_kind == "signal_close":
            target = float(event.get("close", event.get("signal_close", np.nan)))
        else:
            target = float(event.get("open", np.nan))
        if not math.isfinite(target):
            return False, None, None, "missing_reclaim_target"
        for pos in range(signal_pos + 1, min(n - 1, signal_pos + valid_bars) + 1):
            vol_ok = True
            if math.isfinite(vol_base[pos]) and vol_base[pos] > 0:
                vol_ok = float(volume[pos]) >= float(args.reclaim_volume_mult) * float(vol_base[pos])
            if closes[pos] > target and vol_ok:
                entry_pos = pos + 1
                if entry_pos >= n:
                    return False, None, None, "reclaim_no_next_open"
                return True, int(entry_pos), float(opens[entry_pos]), f"reclaim_{target_kind}_confirmed_at_{market.index[pos]}"
        return False, None, None, "reclaim_not_confirmed"

    m = re.fullmatch(r"limit_low(\d{2})_(\d+)", str(entry_mode))
    if m:
        frac = int(m.group(1)) / 100.0
        valid_bars = int(m.group(2))
        sig_low = float(event.get("low", event.get("signal_low", np.nan)))
        sig_high = float(event.get("high", np.nan))
        if not (math.isfinite(sig_low) and math.isfinite(sig_high) and sig_high > sig_low):
            return False, None, None, "missing_limit_range"
        limit_price = sig_low + (sig_high - sig_low) * float(frac)
        fill_threshold = limit_price - float(args.limit_tick_size)
        for pos in range(signal_pos + 1, min(n - 1, signal_pos + valid_bars) + 1):
            if lows[pos] <= fill_threshold:
                return True, int(pos), float(limit_price), f"limit_fill_low{int(frac*100):02d}_bar{pos-signal_pos}"
        return False, None, None, "limit_not_filled"

    raise ValueError(f"Unknown entry_mode: {entry_mode}")


def _horizon_from_exit_mode(exit_mode: str) -> int:
    if exit_mode.startswith("time"):
        return int(exit_mode.replace("time", ""))
    m = re.search(r"time(\d+)", exit_mode)
    return int(m.group(1)) if m else 48


def _target_price_for_event(event: pd.Series) -> float:
    vals = [event.get("open", np.nan), event.get("swing_level", np.nan)]
    nums = [float(v) for v in vals if pd.notna(v) and math.isfinite(float(v)) and float(v) > 0]
    return max(nums) if nums else float("nan")




def _timing_stats_from_path(mtm_low: Sequence[float], mtm_high: Sequence[float]) -> dict[str, object]:
    low_arr = np.asarray(mtm_low, dtype=float)
    high_arr = np.asarray(mtm_high, dtype=float)
    out: dict[str, object] = {
        "mae_time_bars": np.nan,
        "mfe_time_bars": np.nan,
        "first_positive_high_bars": np.nan,
        "mae_before_mfe_flag": False,
    }
    if low_arr.size and np.isfinite(low_arr).any():
        out["mae_time_bars"] = int(np.nanargmin(low_arr))
    if high_arr.size and np.isfinite(high_arr).any():
        out["mfe_time_bars"] = int(np.nanargmax(high_arr))
        pos = np.flatnonzero(high_arr > 0)
        if pos.size:
            out["first_positive_high_bars"] = int(pos[0])
    if math.isfinite(float(out["mae_time_bars"])) and math.isfinite(float(out["mfe_time_bars"])):
        out["mae_before_mfe_flag"] = int(out["mae_time_bars"]) <= int(out["mfe_time_bars"])
    return out


def _timing_stats_from_bars(bars: pd.DataFrame, rec: dict[str, object]) -> dict[str, object]:
    try:
        entry_pos = int(rec.get("entry_pos"))
        exit_pos = int(rec.get("exit_pos"))
        entry_price = float(rec.get("initial_entry_price", rec.get("entry_price", rec.get("avg_entry_price", np.nan))))
    except Exception:
        return _timing_stats_from_path([], [])
    if not math.isfinite(entry_price) or entry_price <= 0 or exit_pos < entry_pos:
        return _timing_stats_from_path([], [])
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)[entry_pos : exit_pos + 1]
    highs = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)[entry_pos : exit_pos + 1]
    return _timing_stats_from_path(lows / entry_price - 1.0, highs / entry_price - 1.0)

def _reference_full_entry_scheme(args: argparse.Namespace):
    # Keep this helper compatible with the reference validation engine.
    # get_scale_in_schemes() takes no args; the previous call passed args and
    # also missed the import, causing the first probe-compatible variant to fail
    # before any results were written.
    schemes = get_scale_in_schemes()
    for scheme in schemes:
        if scheme.name == "full_entry":
            return scheme
    return schemes[0]


def _is_probe_compatible_variant(variant: UpgradeVariant) -> bool:
    return variant.entry_mode == "next_open" and bool(re.fullmatch(r"time\d+", str(variant.exit_mode)))


def simulate_probe_compatible_trade(
    bars: pd.DataFrame,
    event: pd.Series | dict[str, object],
    signal_pos: int,
    variant: UpgradeVariant,
    args: argparse.Namespace,
    *,
    cost_mult: float = 1.0,
) -> dict[str, object]:
    horizon = _horizon_from_exit_mode(variant.exit_mode)
    event_series = event if isinstance(event, pd.Series) else pd.Series(event)
    rec = simulate_probe_trade_path(
        bars,
        event_series,
        int(signal_pos),
        scheme=_reference_full_entry_scheme(args),
        exit_horizon=int(horizon),
        stop_spec=variant.stop_spec,
        args=args,
        cost_mult=float(cost_mult),
    )
    if not rec.get("valid"):
        return rec
    rec = dict(rec)
    rec.update({
        "variant_name": variant.variant_name,
        "candidate_layer": variant.candidate_layer,
        "support_mode": variant.support_mode,
        "entry_mode": variant.entry_mode,
        "exit_mode": variant.exit_mode,
        "entry_reason": "next_open_reference_engine",
        "target_hit": False,
        "partial_exit_done": False,
        "partial_exit_pos": np.nan,
        "partial_exit_price": np.nan,
        "target_price": np.nan,
        "entry_price": rec.get("initial_entry_price", rec.get("avg_entry_price", np.nan)),
        "entry_delay_bars_actual": rec.get("entry_delay_bars", np.nan),
    })
    # Keep legacy column names used by the upgrade summaries.
    for src, dst in [("candidate_name", "candidate_name")]:
        if src not in rec:
            rec[dst] = None
    rec.setdefault("signal_close", float(event_series.get("close", np.nan)))
    rec.setdefault("signal_open", float(event_series.get("open", np.nan)))
    rec.setdefault("signal_low", float(event_series.get("low", np.nan)))
    rec.setdefault("signal_high", float(event_series.get("high", np.nan)))
    rec.setdefault("swing_level", float(event_series.get("swing_level", np.nan)))
    rec.setdefault("swing_age", float(event_series.get("swing_age", np.nan)))
    rec.setdefault("down_spike_pct", float(event_series.get("down_spike_pct", np.nan)))
    rec.setdefault("close_pos_in_bar", float(event_series.get("close_pos_in_bar", np.nan)))
    rec.setdefault("large_trade_share", float(event_series.get("large_trade_share", np.nan)))
    rec.setdefault("atr_pct", float(event_series.get("atr_pct", np.nan)))
    rec.setdefault("session_bucket", event_series.get("session_bucket", "NA"))
    rec.setdefault("cluster_touch_count_020", event_series.get("cluster_touch_count_020", np.nan))
    rec.setdefault("cluster_touch_count_030", event_series.get("cluster_touch_count_030", np.nan))
    rec.update(_timing_stats_from_bars(bars, rec))
    return rec


def simulate_equal2_addon_trade(
    bars: pd.DataFrame,
    event: pd.Series | dict[str, object],
    signal_pos: int,
    variant: UpgradeVariant,
    args: argparse.Namespace,
    *,
    cost_mult: float = 1.0,
    market: MarketCache | None = None,
) -> dict[str, object]:
    """Initial next-open entry plus one causal equal-low confirmation add.

    This is not the same as filtering the initial signal by ``equal2_020``.  The
    initial single-swing A0/A1 trade is opened first; then, only if a new
    confirmed swing low forms near the original swept low after entry, a capped
    second leg is added at the following open.  It directly tests the user's
    idea of adding risk after equal-low confirmation instead of discarding the
    original single-swing trade.
    """
    if market is None:
        market = build_market_cache(bars, args)
    opens = market.open
    highs = market.high
    lows = market.low
    closes = market.close
    idx = market.index
    n = len(market.index)

    m_add = re.fullmatch(r"next_open_add_equal2_(\d{3})", str(variant.entry_mode))
    if not m_add:
        return {"valid": False, "invalid_reason": "not_equal2_addon_mode"}
    add_weight = int(m_add.group(1)) / 100.0
    signal_pos = int(signal_pos)
    delay = int(args.entry_delay_bars)
    entry_pos = signal_pos + delay
    horizon = _horizon_from_exit_mode(variant.exit_mode)
    planned_exit_pos = signal_pos + int(horizon)
    if entry_pos >= n or planned_exit_pos >= n or planned_exit_pos <= entry_pos:
        return {"valid": False, "invalid_reason": "insufficient_future_bars"}

    initial_entry_price = float(opens[entry_pos])
    stop_price, stop_pct, stop_mode_used = _compute_stop_price(event, initial_entry_price, variant.stop_spec, args)
    base_level = float(event.get("swing_level", event.get("signal_low", np.nan)))
    if not math.isfinite(base_level) or base_level <= 0:
        return {"valid": False, "invalid_reason": "missing_equal2_base_level"}
    tol = 0.0020
    left = int(args.pivot_left)
    right = int(args.pivot_right)

    legs: list[dict[str, object]] = [{"weight": 1.0, "entry_price": initial_entry_price, "entry_pos": int(entry_pos), "leg_type": "initial"}]
    add_filled = False
    add_confirm_pos = np.nan
    add_entry_pos = np.nan
    add_entry_price = np.nan
    stop_hit = False
    exit_pos = int(planned_exit_pos)
    exit_reason = f"time_exit_h{int(horizon)}"
    exit_price = float(closes[planned_exit_pos])
    mtm_low: list[float] = []
    mtm_high: list[float] = []

    for pos in range(int(entry_pos), int(planned_exit_pos) + 1):
        if stop_price is not None and float(lows[pos]) <= float(stop_price):
            stop_hit = True
            exit_pos = int(pos)
            exit_price = float(stop_price)
            exit_reason = stop_mode_used
            break

        # A new swing low is usable only after right-side bars have closed.  The
        # add is scheduled for the next open after confirmation, so this remains
        # causal even though the whole month may be loaded in memory.
        if not add_filled:
            center = int(pos) - right
            if center >= int(entry_pos) and center - left >= 0 and center + right <= int(pos):
                win = lows[center - left : center + right + 1]
                if len(win) == left + right + 1 and lows[center] <= np.nanmin(win):
                    near_base = abs(float(lows[center]) / base_level - 1.0) <= tol
                    if near_base:
                        candidate_add_pos = int(pos) + 1
                        if candidate_add_pos <= int(planned_exit_pos) and candidate_add_pos < n:
                            add_price = float(opens[candidate_add_pos])
                            legs.append({"weight": float(add_weight), "entry_price": add_price, "entry_pos": int(candidate_add_pos), "leg_type": "equal2_confirm_add"})
                            add_filled = True
                            add_confirm_pos = int(pos)
                            add_entry_pos = int(candidate_add_pos)
                            add_entry_price = float(add_price)

        active_legs = [leg for leg in legs if int(leg["entry_pos"]) <= int(pos)]
        if active_legs:
            low_pnl = sum(float(leg["weight"]) * (float(lows[pos]) / float(leg["entry_price"]) - 1.0) for leg in active_legs)
            high_pnl = sum(float(leg["weight"]) * (float(highs[pos]) / float(leg["entry_price"]) - 1.0) for leg in active_legs)
            mtm_low.append(float(low_pnl))
            mtm_high.append(float(high_pnl))

    filled_legs = [leg for leg in legs if int(leg["entry_pos"]) <= int(exit_pos)]
    filled_weight = float(sum(float(leg["weight"]) for leg in filled_legs))
    if filled_weight <= 0:
        return {"valid": False, "invalid_reason": "no_filled_weight"}
    gross = float(sum(float(leg["weight"]) * (float(exit_price) / float(leg["entry_price"]) - 1.0) for leg in filled_legs))
    net = gross - filled_weight * _entry_cost(args, cost_mult) - filled_weight * _exit_cost(args, cost_mult)
    avg_entry = float(sum(float(leg["weight"]) * float(leg["entry_price"]) for leg in filled_legs) / filled_weight)
    timing = _timing_stats_from_path(mtm_low, mtm_high)

    return {
        "valid": True,
        "variant_name": variant.variant_name,
        "candidate_layer": variant.candidate_layer,
        "support_mode": variant.support_mode,
        "entry_mode": variant.entry_mode,
        "exit_mode": variant.exit_mode,
        "stop_name": variant.stop_spec.name,
        "stop_mode": variant.stop_spec.mode,
        "cost_mult": float(cost_mult),
        "signal_time": event.get("signal_time"),
        "entry_time": idx[int(entry_pos)],
        "exit_time": idx[int(exit_pos)],
        "signal_pos": int(signal_pos),
        "entry_pos": int(entry_pos),
        "exit_pos": int(exit_pos),
        "entry_delay_bars_actual": int(entry_pos - signal_pos),
        "bars_held": int(exit_pos - entry_pos),
        "entry_reason": "next_open_with_equal2_addon",
        "exit_reason": exit_reason,
        "stop_hit": bool(stop_hit),
        "target_hit": False,
        "partial_exit_done": False,
        "partial_exit_pos": np.nan,
        "partial_exit_price": np.nan,
        "target_price": np.nan,
        "entry_price": float(initial_entry_price),
        "avg_entry_price": float(avg_entry),
        "exit_price": float(exit_price),
        "stop_price": float(stop_price) if stop_price is not None else np.nan,
        "stop_pct": float(stop_pct) if stop_pct is not None else np.nan,
        "filled_weight": float(filled_weight),
        "add_count": int(max(0, len(filled_legs) - 1)),
        "add_filled": bool(add_filled and filled_weight > 1.0),
        "add_confirm_pos": add_confirm_pos,
        "add_entry_pos": add_entry_pos,
        "add_entry_price": add_entry_price,
        "leg_weights": "|".join(f"{float(leg['weight']):.4f}" for leg in filled_legs),
        "leg_entry_prices": "|".join(f"{float(leg['entry_price']):.4f}" for leg in filled_legs),
        "leg_types": "|".join(str(leg["leg_type"]) for leg in filled_legs),
        "gross_return_on_equity": float(gross),
        "net_return_on_equity": float(net),
        "mae_on_equity": float(np.nanmin(mtm_low)) if mtm_low else np.nan,
        "mfe_on_equity": float(np.nanmax(mtm_high)) if mtm_high else np.nan,
        "mae_time_bars": timing["mae_time_bars"],
        "mfe_time_bars": timing["mfe_time_bars"],
        "first_positive_high_bars": timing["first_positive_high_bars"],
        "mae_before_mfe_flag": timing["mae_before_mfe_flag"],
        "signal_close": float(event.get("close", np.nan)),
        "signal_open": float(event.get("open", np.nan)),
        "signal_low": float(event.get("low", np.nan)),
        "signal_high": float(event.get("high", np.nan)),
        "swing_level": float(event.get("swing_level", np.nan)),
        "swing_age": float(event.get("swing_age", np.nan)),
        "down_spike_pct": float(event.get("down_spike_pct", np.nan)),
        "close_pos_in_bar": float(event.get("close_pos_in_bar", np.nan)),
        "large_trade_share": float(event.get("large_trade_share", np.nan)),
        "atr_pct": float(event.get("atr_pct", np.nan)),
        "session_bucket": event.get("session_bucket", "NA"),
        "cluster_touch_count_020": event.get("cluster_touch_count_020", np.nan),
        "cluster_touch_count_030": event.get("cluster_touch_count_030", np.nan),
    }

def simulate_upgrade_trade(
    bars: pd.DataFrame,
    event: pd.Series | dict[str, object],
    signal_pos: int,
    variant: UpgradeVariant,
    args: argparse.Namespace,
    *,
    cost_mult: float = 1.0,
    market: MarketCache | None = None,
) -> dict[str, object]:
    if market is None:
        market = build_market_cache(bars, args)
    opens = market.open
    highs = market.high
    lows = market.low
    closes = market.close
    idx = market.index
    n = len(market.index)

    if re.fullmatch(r"next_open_add_equal2_\d{3}", str(variant.entry_mode)):
        return simulate_equal2_addon_trade(bars, event, int(signal_pos), variant, args, cost_mult=cost_mult, market=market)

    ok, entry_pos, entry_price, entry_reason = _entry_from_mode(bars, event, int(signal_pos), variant.entry_mode, args, market)
    if not ok or entry_pos is None or entry_price is None:
        return {"valid": False, "invalid_reason": entry_reason}

    horizon = _horizon_from_exit_mode(variant.exit_mode)
    planned_exit_pos = int(signal_pos) + int(horizon)
    if planned_exit_pos >= n or planned_exit_pos <= int(entry_pos):
        return {"valid": False, "invalid_reason": "insufficient_future_bars"}

    stop_price, stop_pct, stop_mode_used = _compute_stop_price(event, float(entry_price), variant.stop_spec, args)
    current_stop = stop_price
    exit_pos = planned_exit_pos
    exit_price = float(closes[planned_exit_pos])
    exit_reason = f"time_exit_h{horizon}"
    stop_hit = False
    target_hit = False
    partial_exit_done = False
    partial_exit_price = np.nan
    partial_exit_pos = np.nan
    remaining_weight = 1.0
    realized_partial = 0.0
    high_water = float(entry_price)
    red_streak = 0
    red_drop_sum = 0.0
    target_price = _target_price_for_event(event)

    # For structure-confirmed trailing: do not move the stop immediately after
    # a noisy 1m swing low.  First store a confirmed swing low, then wait for a
    # later confirmed swing high to be broken; only then arm/raise the stop at
    # that prior swing low.  This tests whether delaying the stop update until
    # market structure turns up avoids 1m wick noise.
    pending_struct_low: tuple[int, float] | None = None
    pending_struct_high: tuple[int, float] | None = None

    mtm_low: list[float] = []
    mtm_high: list[float] = []

    for pos in range(int(entry_pos), int(planned_exit_pos) + 1):
        # Mark-to-market before decisions. For partial modes, realized component
        # plus remaining leg is tracked on equity.
        high_water = max(high_water, float(highs[pos]))
        low_unreal = remaining_weight * (float(lows[pos]) / float(entry_price) - 1.0) + realized_partial
        high_unreal = remaining_weight * (float(highs[pos]) / float(entry_price) - 1.0) + realized_partial
        mtm_low.append(float(low_unreal))
        mtm_high.append(float(high_unreal))

        # Conservative for long: stop before targets on the same bar.
        if current_stop is not None and lows[pos] <= float(current_stop):
            stop_hit = True
            exit_pos = int(pos)
            exit_price = float(current_stop)
            exit_reason = stop_mode_used if current_stop == stop_price else "dynamic_profit_stop"
            break

        # Dynamic MFE protection.
        if variant.exit_mode in {"mfe_protect_15_05_time60", "partial_target_trail_time96"}:
            if high_water / float(entry_price) - 1.0 >= float(args.mfe_trigger_pct):
                protect = float(entry_price) * (1.0 + float(args.mfe_lock_pct))
                current_stop = max(float(current_stop) if current_stop is not None else -np.inf, protect)
        m_mfe_lock = re.fullmatch(r"mfe_lock_(\d{2})_(\d{2})_time(\d+)", str(variant.exit_mode))
        if m_mfe_lock:
            trigger = int(m_mfe_lock.group(1)) / 1000.0
            lock = int(m_mfe_lock.group(2)) / 1000.0
            if high_water / float(entry_price) - 1.0 >= trigger:
                protect = float(entry_price) * (1.0 + lock)
                current_stop = max(float(current_stop) if current_stop is not None else -np.inf, protect)
        if variant.exit_mode == "partial_target_trail_time96":
            if high_water / float(entry_price) - 1.0 >= float(args.trail_trigger_pct):
                trail = high_water * (1.0 - float(args.trail_giveback_pct))
                current_stop = max(float(current_stop) if current_stop is not None else -np.inf, trail)

        # Early failure exits: if a panic-rebound trade does not produce even a
        # small MFE by N closed bars after entry, exit on the next open.  This
        # tests a comfort/failure stop without using future bars.
        m_fail = re.fullmatch(r"fail_no_mfe(\d+)_(\d{3})_time(\d+)", str(variant.exit_mode))
        if m_fail:
            fail_bars = int(m_fail.group(1))
            min_mfe = int(m_fail.group(2)) / 1000.0
            if pos >= int(entry_pos) + fail_bars and high_water / float(entry_price) - 1.0 < min_mfe:
                next_pos = min(pos + 1, planned_exit_pos)
                exit_pos = int(next_pos)
                exit_price = float(opens[next_pos]) if next_pos < n else float(closes[pos])
                exit_reason = f"early_fail_no_mfe{fail_bars}_{min_mfe:.3%}_next_open"
                break

        # Target exits. Use next open for full target modes unless user later
        # changes target-exit behavior; this is conservative and causal.
        target_eligible = pos >= int(entry_pos) + int(args.target_entry_same_bar_min_delay)
        m_partial_pct = re.fullmatch(r"partial_pct(\d{2})_(\d{2})_time(\d+)", str(variant.exit_mode))
        if target_eligible and m_partial_pct and not partial_exit_done:
            trigger_pct = int(m_partial_pct.group(1)) / 1000.0
            close_weight = int(m_partial_pct.group(2)) / 100.0
            if highs[pos] >= float(entry_price) * (1.0 + trigger_pct):
                partial_exit_done = True
                partial_exit_pos = int(min(pos + 1, planned_exit_pos))
                partial_exit_price = float(opens[int(partial_exit_pos)]) if int(partial_exit_pos) < n else float(entry_price) * (1.0 + trigger_pct)
                close_weight = min(max(close_weight, 0.0), remaining_weight)
                realized_partial += close_weight * (float(partial_exit_price) / float(entry_price) - 1.0)
                remaining_weight -= close_weight
                target_hit = True
        if target_eligible and math.isfinite(target_price) and highs[pos] >= target_price:
            if variant.exit_mode == "target_signal_open_or_time48":
                target_hit = True
                next_pos = min(pos + 1, planned_exit_pos)
                exit_pos = int(next_pos)
                exit_price = float(opens[next_pos]) if next_pos < n else float(target_price)
                exit_reason = "target_signal_open_or_swing_next_open"
                break
            if variant.exit_mode == "partial_target_trail_time96" and not partial_exit_done:
                partial_exit_done = True
                partial_exit_pos = int(min(pos + 1, planned_exit_pos))
                partial_exit_price = float(opens[int(partial_exit_pos)]) if int(partial_exit_pos) < n else float(target_price)
                realized_partial = 0.5 * (float(partial_exit_price) / float(entry_price) - 1.0)
                remaining_weight = 0.5
                target_hit = True

        # Momentum exhaustion: two red bars with combined drop > threshold exits
        # next open. This uses only closed bars before scheduling the exit.
        if variant.exit_mode == "momentum_exhaust_time60":
            red = closes[pos] < opens[pos]
            if red:
                red_streak += 1
                red_drop_sum += max(0.0, float(opens[pos] / closes[pos] - 1.0))
            else:
                red_streak = 0
                red_drop_sum = 0.0
            if red_streak >= int(args.momentum_exhaust_bars) and red_drop_sum >= float(args.momentum_exhaust_drop_pct):
                next_pos = min(pos + 1, planned_exit_pos)
                exit_pos = int(next_pos)
                exit_price = float(opens[next_pos]) if next_pos < n else float(closes[pos])
                exit_reason = "momentum_exhaust_next_open"
                break

        # Causal dynamic swing-low trailing stop: a low at ``center`` is only
        # usable after ``pivot_right`` bars have closed.  The updated stop is
        # applied only to later bars because stop checks happen at the top of
        # the loop before this update.
        m_trail = re.fullmatch(r"swing_trail_after(\d+)_time(\d+)", str(variant.exit_mode))
        if m_trail:
            start_after = int(m_trail.group(1))
            right = int(args.pivot_right)
            left = int(args.pivot_left)
            center = int(pos) - right
            if center >= int(entry_pos) + start_after and center - left >= 0 and center + right <= int(pos):
                win = lows[center - left : center + right + 1]
                if len(win) == left + right + 1 and lows[center] <= np.nanmin(win):
                    trail_stop = float(lows[center]) * (1.0 - float(args.swing_trail_buffer_pct))
                    if trail_stop < float(entry_price):
                        current_stop = max(float(current_stop) if current_stop is not None else -np.inf, trail_stop)

        # Structure-confirmed swing trailing stop.  Unlike swing_trail_afterX,
        # this does not arm a stop just because a 1m swing low has formed.  It
        # waits for a subsequent confirmed swing high and a later breakout above
        # that swing high.  Only after that higher-structure confirmation does
        # it lift the stop to the preceding swing low.
        m_struct = re.fullmatch(r"swing_struct_trail_after(\d+)_time(\d+)", str(variant.exit_mode))
        if m_struct:
            start_after = int(m_struct.group(1))
            right = int(args.pivot_right)
            left = int(args.pivot_left)
            center = int(pos) - right
            if center >= int(entry_pos) + start_after and center - left >= 0 and center + right <= int(pos):
                low_win = lows[center - left : center + right + 1]
                high_win = highs[center - left : center + right + 1]
                if len(low_win) == left + right + 1 and lows[center] <= np.nanmin(low_win):
                    pending_struct_low = (int(center), float(lows[center]))
                    pending_struct_high = None
                if pending_struct_low is not None and center > pending_struct_low[0] and len(high_win) == left + right + 1 and highs[center] >= np.nanmax(high_win):
                    pending_struct_high = (int(center), float(highs[center]))
            if pending_struct_low is not None and pending_struct_high is not None and int(pos) > pending_struct_high[0]:
                if float(highs[pos]) >= float(pending_struct_high[1]):
                    trail_stop = float(pending_struct_low[1]) * (1.0 - float(args.swing_trail_buffer_pct))
                    if trail_stop < float(entry_price):
                        current_stop = max(float(current_stop) if current_stop is not None else -np.inf, trail_stop)
                    pending_struct_low = None
                    pending_struct_high = None

    gross = realized_partial + remaining_weight * (float(exit_price) / float(entry_price) - 1.0)
    net = gross - _entry_cost(args, cost_mult) - _exit_cost(args, cost_mult)
    mae = float(np.nanmin(mtm_low)) if mtm_low else np.nan
    mfe = float(np.nanmax(mtm_high)) if mtm_high else np.nan
    timing = _timing_stats_from_path(mtm_low, mtm_high)

    return {
        "valid": True,
        "variant_name": variant.variant_name,
        "candidate_layer": variant.candidate_layer,
        "support_mode": variant.support_mode,
        "entry_mode": variant.entry_mode,
        "exit_mode": variant.exit_mode,
        "stop_name": variant.stop_spec.name,
        "stop_mode": variant.stop_spec.mode,
        "cost_mult": float(cost_mult),
        "signal_time": event.get("signal_time"),
        "entry_time": idx[int(entry_pos)],
        "exit_time": idx[int(exit_pos)],
        "signal_pos": int(signal_pos),
        "entry_pos": int(entry_pos),
        "exit_pos": int(exit_pos),
        "entry_delay_bars_actual": int(entry_pos - signal_pos),
        "bars_held": int(exit_pos - entry_pos),
        "entry_reason": entry_reason,
        "exit_reason": exit_reason,
        "stop_hit": bool(stop_hit),
        "target_hit": bool(target_hit),
        "partial_exit_done": bool(partial_exit_done),
        "partial_exit_pos": partial_exit_pos,
        "partial_exit_price": partial_exit_price,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "target_price": float(target_price) if math.isfinite(target_price) else np.nan,
        "stop_price": float(stop_price) if stop_price is not None else np.nan,
        "stop_pct": float(stop_pct) if stop_pct is not None else np.nan,
        "gross_return_on_equity": float(gross),
        "net_return_on_equity": float(net),
        "mae_on_equity": float(mae),
        "mfe_on_equity": float(mfe),
        "mae_time_bars": timing["mae_time_bars"],
        "mfe_time_bars": timing["mfe_time_bars"],
        "first_positive_high_bars": timing["first_positive_high_bars"],
        "mae_before_mfe_flag": timing["mae_before_mfe_flag"],
        "signal_close": float(event.get("close", np.nan)),
        "signal_open": float(event.get("open", np.nan)),
        "signal_low": float(event.get("low", np.nan)),
        "signal_high": float(event.get("high", np.nan)),
        "swing_level": float(event.get("swing_level", np.nan)),
        "swing_age": float(event.get("swing_age", np.nan)),
        "down_spike_pct": float(event.get("down_spike_pct", np.nan)),
        "close_pos_in_bar": float(event.get("close_pos_in_bar", np.nan)),
        "large_trade_share": float(event.get("large_trade_share", np.nan)),
        "atr_pct": float(event.get("atr_pct", np.nan)),
        "session_bucket": event.get("session_bucket", "NA"),
        "cluster_touch_count_020": event.get("cluster_touch_count_020", np.nan),
        "cluster_touch_count_030": event.get("cluster_touch_count_030", np.nan),
    }


def _event_positions(bars: pd.DataFrame, events: pd.DataFrame, max_horizon: int) -> tuple[pd.DataFrame, np.ndarray]:
    if events.empty:
        return events.copy(), np.asarray([], dtype=int)
    times = pd.DatetimeIndex(pd.to_datetime(events["signal_time"]))
    signal_pos = bars.index.get_indexer(times)
    valid = (signal_pos >= 0) & ((signal_pos + int(max_horizon) + 2) < len(bars))
    return events.loc[valid].copy().reset_index(drop=True), signal_pos[valid]


def simulate_upgrade_variant(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    variant: UpgradeVariant,
    args: argparse.Namespace,
    market: MarketCache | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if market is None:
        market = build_market_cache(bars, args)
    max_horizon = max(_horizon_from_exit_mode(x) for x in [variant.exit_mode])
    ev, positions = _event_positions(bars, events, max_horizon=max_horizon)
    rows: list[dict[str, object]] = []
    skipped_overlap = 0
    skipped_invalid = 0
    last_exit_pos = -1
    # ``to_dict('records')`` is intentionally done once per filtered event set.
    # Each record is then reused without constructing a pandas Series per trade.
    for event, signal_pos in zip(ev.to_dict("records"), positions):
        if int(signal_pos) <= last_exit_pos:
            skipped_overlap += 1
            continue
        if _is_probe_compatible_variant(variant):
            rec = simulate_probe_compatible_trade(bars, event, int(signal_pos), variant, args)
        else:
            rec = simulate_upgrade_trade(bars, event, int(signal_pos), variant, args, market=market)
        if not rec.get("valid"):
            skipped_invalid += 1
            continue
        last_exit_pos = int(rec.get("exit_pos", signal_pos))
        rows.append(rec)
    trades = pd.DataFrame(rows)
    return trades, {
        "skipped_overlap": int(skipped_overlap),
        "skipped_invalid": int(skipped_invalid),
        "input_events": int(len(ev)),
    }


def summarize_trades(trades: pd.DataFrame, args: argparse.Namespace, extra: dict[str, int] | None = None) -> dict[str, object]:
    extra = extra or {}
    if trades.empty:
        out: dict[str, object] = dict(extra)
        out.update({"trades": 0})
        return out
    x = pd.to_numeric(trades["net_return_on_equity"], errors="coerce").fillna(0.0)
    equity, dd = _equity_and_dd(x, float(args.starting_equity))
    first_entry = pd.Timestamp(trades["entry_time"].iloc[0])
    last_exit = pd.Timestamp(trades["exit_time"].iloc[-1])
    days = max(1e-9, (last_exit - first_entry).total_seconds() / 86400.0)
    total_ret = float(equity.iloc[-1] / float(args.starting_equity) - 1.0)
    ann_ret = float((1.0 + total_ret) ** (365.0 / days) - 1.0) if total_ret > -1.0 else -1.0
    wins = x[x > 0]
    losses = x[x < 0]
    out = {
        "trades": int(len(trades)),
        "return_total": total_ret,
        "return_annualized": ann_ret,
        "mean_return": float(x.mean()),
        "median_return": float(x.median()),
        "win_rate": float((x > 0).mean()),
        "avg_win": float(wins.mean()) if not wins.empty else np.nan,
        "avg_loss": float(losses.mean()) if not losses.empty else np.nan,
        "payoff_ratio": _payoff_ratio(x),
        "profit_factor": _profit_factor(x),
        "max_drawdown": float(dd.min()),
        "max_consecutive_losses": _max_consecutive_losses(x),
        "top5_winner_share": _top_winner_share(x),
        "worst_trade": float(x.min()),
        "best_trade": float(x.max()),
        "mae_mean": float(pd.to_numeric(trades["mae_on_equity"], errors="coerce").mean()),
        "mae_median": float(pd.to_numeric(trades["mae_on_equity"], errors="coerce").median()),
        "mae_p05": float(pd.to_numeric(trades["mae_on_equity"], errors="coerce").quantile(0.05)),
        "mfe_mean": float(pd.to_numeric(trades["mfe_on_equity"], errors="coerce").mean()),
        "mfe_median": float(pd.to_numeric(trades["mfe_on_equity"], errors="coerce").median()),
        "mae_time_median_bars": float(_safe_num(trades["mae_time_bars"] if "mae_time_bars" in trades else np.nan, trades.index).median()),
        "mfe_time_median_bars": float(_safe_num(trades["mfe_time_bars"] if "mfe_time_bars" in trades else np.nan, trades.index).median()),
        "first_positive_high_median_bars": float(_safe_num(trades["first_positive_high_bars"] if "first_positive_high_bars" in trades else np.nan, trades.index).median()),
        "mae_before_mfe_rate": float(_safe_num(trades["mae_before_mfe_flag"] if "mae_before_mfe_flag" in trades else np.nan, trades.index).mean()),
        "stop_hit_rate": float(pd.to_numeric(trades["stop_hit"], errors="coerce").mean()),
        "target_hit_rate": float(pd.to_numeric(trades["target_hit"], errors="coerce").mean()),
        "partial_exit_rate": float(pd.to_numeric(trades["partial_exit_done"], errors="coerce").mean()),
        "avg_bars_held": float(pd.to_numeric(trades["bars_held"], errors="coerce").mean()),
    }
    out.update(extra)
    return out


def summarize_by_period(trades: pd.DataFrame, period_col: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    tmp = trades.copy()
    tmp["exit_time"] = pd.to_datetime(tmp["exit_time"])
    if period_col == "year":
        tmp["year"] = tmp["exit_time"].dt.year
    elif period_col == "month":
        tmp["month"] = tmp["exit_time"].dt.to_period("M").astype(str)
    rows = []
    for (variant_name, period), grp in tmp.groupby(["variant_name", period_col], dropna=False):
        x = pd.to_numeric(grp["net_return_on_equity"], errors="coerce").fillna(0.0)
        equity, dd = _equity_and_dd(x, 1.0)
        rows.append({
            "variant_name": variant_name,
            period_col: period,
            "trades": int(len(grp)),
            "return_total": float(equity.iloc[-1] - 1.0) if not equity.empty else np.nan,
            "mean_return": float(x.mean()) if not x.empty else np.nan,
            "median_return": float(x.median()) if not x.empty else np.nan,
            "win_rate": float((x > 0).mean()) if not x.empty else np.nan,
            "profit_factor": _profit_factor(x),
            "max_drawdown": float(dd.min()) if not dd.empty else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["variant_name", period_col]).reset_index(drop=True)


def summarize_compare(summary: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in summary.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        top = grp.sort_values(["profit_factor", "return_total", "trades"], ascending=[False, False, False]).head(10)
        rec = {col: val for col, val in zip(group_cols, keys)}
        rec.update({
            "variants": int(len(grp)),
            "edge_variants": int((grp.get("edge_candidate", False) == True).sum()) if "edge_candidate" in grp else 0,
            "best_pf": float(pd.to_numeric(grp["profit_factor"], errors="coerce").max()),
            "best_return": float(pd.to_numeric(grp["return_total"], errors="coerce").max()),
            "best_median": float(pd.to_numeric(grp["median_return"], errors="coerce").max()),
            "best_mae_p05": float(pd.to_numeric(top["mae_p05"], errors="coerce").max()) if "mae_p05" in top else np.nan,
            "top_variant": str(top.iloc[0]["variant_name"]) if not top.empty else "",
        })
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["edge_variants", "best_pf", "best_return"], ascending=[False, False, False]).reset_index(drop=True)


def build_context_breakdown(trades: pd.DataFrame, context_cols: Sequence[str], min_trades: int = 20) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for col in context_cols:
        if col not in trades.columns:
            continue
        s = trades[col]
        if pd.api.types.is_numeric_dtype(s):
            try:
                bucket = pd.qcut(pd.to_numeric(s, errors="coerce"), q=4, duplicates="drop")
            except Exception:
                bucket = pd.cut(pd.to_numeric(s, errors="coerce"), bins=4)
        else:
            bucket = s.astype("object")
        tmp = trades.copy()
        tmp["context_bucket"] = bucket.astype("object")
        for (variant_name, bkt), grp in tmp.groupby(["variant_name", "context_bucket"], dropna=False):
            if len(grp) < int(min_trades):
                continue
            x = pd.to_numeric(grp["net_return_on_equity"], errors="coerce").fillna(0.0)
            rows.append({
                "context_col": col,
                "context_bucket": str(bkt),
                "variant_name": variant_name,
                "trades": int(len(grp)),
                "mean_return": float(x.mean()),
                "median_return": float(x.median()),
                "win_rate": float((x > 0).mean()),
                "profit_factor": _profit_factor(x),
                "mae_median": float(pd.to_numeric(grp["mae_on_equity"], errors="coerce").median()),
            })
    return pd.DataFrame(rows).sort_values(["context_col", "profit_factor", "trades"], ascending=[True, False, False]).reset_index(drop=True) if rows else pd.DataFrame()


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(out):
        return default
    return int(out)


def build_edge_registry(summary: pd.DataFrame, yearly: pd.DataFrame, monthly: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    yr = yearly.groupby("variant_name").agg(
        tested_years=("year", "count"),
        positive_years=("return_total", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        min_year_return=("return_total", "min"),
    ).reset_index() if not yearly.empty else pd.DataFrame(columns=["variant_name", "tested_years", "positive_years", "min_year_return"])
    mo = monthly.groupby("variant_name").agg(
        tested_months=("month", "count"),
        positive_months=("return_total", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
    ).reset_index() if not monthly.empty else pd.DataFrame(columns=["variant_name", "tested_months", "positive_months"])
    out = summary.merge(yr, on="variant_name", how="left").merge(mo, on="variant_name", how="left")
    # Variants with zero executed trades are expected to have no yearly/monthly rows.
    # After the left join these counts become NaN; normalize them to zero so report
    # generation never fails.  If a variant has trades>0 but period stats are still
    # missing, keep an explicit audit flag because that would indicate a reporting bug.
    for col in ["tested_years", "positive_years", "tested_months", "positive_months"]:
        if col not in out.columns:
            out[col] = 0
    out["period_stats_missing_flag"] = (
        (pd.to_numeric(out.get("trades", 0), errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(out["tested_years"], errors="coerce").fillna(0) <= 0)
    )
    out[["tested_years", "positive_years", "tested_months", "positive_months"]] = (
        out[["tested_years", "positive_years", "tested_months", "positive_months"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .astype(int)
    )
    rows = []
    for _, row in out.iterrows():
        trades = _safe_int(row.get("trades", 0), 0)
        ret = _safe_float(row.get("return_total", np.nan))
        pf = _safe_float(row.get("profit_factor", np.nan))
        median = _safe_float(row.get("median_return", np.nan))
        pos_years = _safe_int(row.get("positive_years", 0), 0)
        tested_years = _safe_int(row.get("tested_years", 0), 0)
        top5 = _safe_float(row.get("top5_winner_share", np.nan))
        period_missing = bool(row.get("period_stats_missing_flag", False))
        status = "research_only_not_promoted"
        if (
            trades >= int(args.min_trades_for_upgrade_edge)
            and math.isfinite(ret) and ret > 0
            and math.isfinite(median) and median > 0
            and math.isfinite(pf) and pf >= float(args.min_pf_for_upgrade_edge)
            and not period_missing
            and tested_years > 0
            and pos_years >= max(3, min(4, tested_years))
            and (not math.isfinite(top5) or top5 <= float(args.max_top5_winner_share_for_edge))
        ):
            status = "upgrade_edge_candidate_not_live_ready"
        rows.append({**row.to_dict(), "status": status})
    return pd.DataFrame(rows).sort_values(["status", "profit_factor", "return_total"], ascending=[True, False, False], na_position="last").reset_index(drop=True)



def build_baseline_consistency_audit(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    upgrade_summary: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Compare key upgrade baseline rows with the existing validation engine.

    This catches accidental divergence between this broad upgrade lab and the
    A-only validation/backtest probe.  It is a reporting/audit layer only; it
    does not feed results back into candidate selection.
    """
    if bool(getattr(args, "skip_consistency_audit", False)):
        return pd.DataFrame([{"audit_status": "skipped_by_user"}])
    baseline_names = {
        "no_stop": (
            "A1_current__single_swing__next_open__time48__no_stop",
            "A_spike_close_large_share__full_entry__h48__no_stop",
        ),
        "fixed_0250": (
            "A1_current__single_swing__next_open__time48__fixed_0250",
            "A_spike_close_large_share__full_entry__h48__fixed_0250",
        ),
        "atr_6x": (
            "A1_current__single_swing__next_open__time48__atr_6x",
            "A_spike_close_large_share__full_entry__h48__atr_6x",
        ),
    }
    # Build only the three baseline probe variants.  It is deliberately tiny and
    # acceptable even when the full upgrade grid is large.
    probe_variants = [v for v in build_probe_variants(args) if v.variant_name in {p for _, p in baseline_names.values()}]
    if not probe_variants:
        return pd.DataFrame([{"audit_status": "no_probe_variants_built"}])
    old_no_progress = getattr(args, "no_progress", False)
    setattr(args, "no_progress", True)
    try:
        _, probe_summary = run_probe_variant_jobs(
            bars,
            events,
            probe_variants,
            args,
            label="[consistency] baselines",
            keep_trades=False,
        )
    finally:
        setattr(args, "no_progress", old_no_progress)
    rows: list[dict[str, object]] = []
    for stop_name, (upgrade_name, probe_name) in baseline_names.items():
        up = upgrade_summary.loc[upgrade_summary["variant_name"].eq(upgrade_name)] if not upgrade_summary.empty else pd.DataFrame()
        pr = probe_summary.loc[probe_summary["variant_name"].eq(probe_name)] if not probe_summary.empty else pd.DataFrame()
        rec: dict[str, object] = {
            "stop_name": stop_name,
            "upgrade_variant_name": upgrade_name,
            "probe_variant_name": probe_name,
            "upgrade_found": bool(not up.empty),
            "probe_found": bool(not pr.empty),
        }
        metrics = ["trades", "return_total", "profit_factor", "max_drawdown", "worst_trade", "win_rate", "median_return"]
        for m in metrics:
            uv = _safe_float(up.iloc[0].get(m, np.nan)) if not up.empty else np.nan
            pv = _safe_float(pr.iloc[0].get(m, np.nan)) if not pr.empty else np.nan
            rec[f"upgrade_{m}"] = uv
            rec[f"probe_{m}"] = pv
            rec[f"diff_{m}"] = uv - pv if math.isfinite(uv) and math.isfinite(pv) else np.nan
        trades_ok = (not math.isfinite(rec.get("diff_trades", np.nan))) or abs(float(rec["diff_trades"])) <= 0
        ret_ok = (not math.isfinite(rec.get("diff_return_total", np.nan))) or abs(float(rec["diff_return_total"])) <= 1e-10
        pf_ok = (not math.isfinite(rec.get("diff_profit_factor", np.nan))) or abs(float(rec["diff_profit_factor"])) <= 1e-10
        rec["audit_pass"] = bool(rec["upgrade_found"] and rec["probe_found"] and trades_ok and ret_ok and pf_ok)
        rec["audit_status"] = "pass" if rec["audit_pass"] else "fail_check_stop_or_event_logic"
        rows.append(rec)
    return pd.DataFrame(rows)

def write_csv(df: pd.DataFrame, path: Path, name: str) -> None:
    print(f"[write] {name} rows={len(df):,} -> {path}", flush=True)
    df.to_csv(path, index=False)



def build_path_timing_compare(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "variant_name", "candidate_layer", "support_mode", "entry_mode", "exit_mode", "stop_name",
        "trades", "return_total", "profit_factor", "win_rate", "max_drawdown", "worst_trade",
        "mae_median", "mfe_median", "mae_time_median_bars", "mfe_time_median_bars",
        "first_positive_high_median_bars", "mae_before_mfe_rate",
    ]
    if summary.empty:
        return pd.DataFrame(columns=cols)
    use = [c for c in cols if c in summary.columns]
    return summary[use].sort_values(["profit_factor", "return_total", "trades"], ascending=[False, False, False]).reset_index(drop=True)



def build_micro_context_coverage(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Audit whether 5s/10s context actually attached and can form candidates."""
    if events.empty:
        return pd.DataFrame()
    fixed = build_fixed_candidate_masks(events)
    a = fixed.get("A_spike_close_large_share", {}).get("mask", pd.Series(False, index=events.index)).fillna(False).astype(bool)
    spike = _safe_num(events.get("down_spike_pct", np.nan), events.index)
    a0 = a & (spike >= 0.0100)
    last_seconds = int(getattr(args, "micro_last_seconds", 20))
    rows: list[dict[str, object]] = []
    for tf in _split_csv_names(getattr(args, "micro_timeframes", "")):
        tag = str(tf).lower()
        prefix = f"micro_{tag}_last{last_seconds}_"
        cols = [c for c in events.columns if c.startswith(prefix)]
        notional_col = f"{prefix}notional"
        ratio_col = f"{prefix}buy_sell_ratio"
        delta_col = f"{prefix}delta_pressure"
        valid_notional = pd.to_numeric(events[notional_col], errors="coerce") if notional_col in events.columns else pd.Series(np.nan, index=events.index)
        matched_col = f"{prefix}matched_rows"
        matched_rows = pd.to_numeric(events[matched_col], errors="coerce") if matched_col in events.columns else pd.Series(np.nan, index=events.index)
        ratio = pd.to_numeric(events[ratio_col], errors="coerce") if ratio_col in events.columns else pd.Series(np.nan, index=events.index)
        delta = pd.to_numeric(events[delta_col], errors="coerce") if delta_col in events.columns else pd.Series(np.nan, index=events.index)
        has_window = matched_rows.fillna(0.0) > 0
        if not has_window.any():
            has_window = valid_notional.fillna(0.0) > 0
        buy_pressure = ratio >= float(getattr(args, "micro_buy_sell_ratio_min", 1.20))
        rows.append({
            "timeframe": tag,
            "micro_columns_present": int(len(cols)),
            "events_total": int(len(events)),
            "events_with_micro_window": int(has_window.sum()),
            "events_with_micro_window_rate": float(has_window.mean()) if len(has_window) else np.nan,
            "ratio_non_null": int(ratio.notna().sum()),
            "delta_non_null": int(delta.notna().sum()),
            "a0_events": int(a0.sum()),
            "a0_events_with_micro_window": int((a0 & has_window).sum()),
            "a0_events_with_buy_pressure": int((a0 & buy_pressure).sum()),
            "a0_buy_pressure_rate_on_valid": float((a0 & buy_pressure).sum() / max(1, int((a0 & has_window).sum()))),
            "buy_sell_ratio_median": float(ratio[has_window].median()) if has_window.any() else np.nan,
            "buy_sell_ratio_p75": float(ratio[has_window].quantile(0.75)) if has_window.any() else np.nan,
            "delta_pressure_median": float(delta[has_window].median()) if has_window.any() else np.nan,
        })
    return pd.DataFrame(rows)


def _summarize_trade_slice_for_overlap(trades: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    if trades.empty:
        return {"trades": 0}
    return summarize_trades(trades.sort_values("exit_time"), args, {})



def build_micro_load_diagnostics(args: argparse.Namespace) -> pd.DataFrame:
    """Return month-level diagnostics for micro trade-bar loading.

    This is deliberately separate from coverage: if coverage is zero, this file
    tells us whether the loader returned zero rows, returned rows in a shifted
    timezone, or returned columns that do not include the expected notional
    fields.
    """
    rows = list(getattr(args, "_micro_load_stats", []) or [])
    if not rows:
        return pd.DataFrame(columns=["timeframe", "month", "load_mode", "rows_loaded", "first_ts", "last_ts", "columns"])
    return pd.DataFrame(rows)

def build_micro_match_debug(args: argparse.Namespace) -> pd.DataFrame:
    """Event-window diagnostics for micro context attachment.

    This answers a different question from the monthly load report: when a
    specific 1m signal asks for the last N seconds of 5s/10s bars, did the
    normalized monthly cache actually contain rows around that timestamp?
    """
    rows = list(getattr(args, "_micro_match_debug", []) or [])
    if not rows:
        return pd.DataFrame(columns=[
            "timeframe", "source_month", "event_row_pos", "signal_time",
            "window_start", "window_end_exclusive", "cache_rows",
            "cache_first_ts", "cache_last_ts", "lo", "hi", "matched_rows",
            "prev_cache_ts", "next_cache_ts", "gap_to_next_sec",
        ])
    return pd.DataFrame(rows)

def build_micro_feature_sample(events: pd.DataFrame, args: argparse.Namespace, limit: int = 200) -> pd.DataFrame:
    """Small event-level sample proving micro features are actually attached."""
    if events.empty:
        return pd.DataFrame()
    last_seconds = int(getattr(args, "micro_last_seconds", 20))
    base_cols = [
        "signal_time", "down_spike_pct", "close_pos_in_bar", "large_trade_share",
        "fp_max_bucket_abs_delta_pressure", "fp_low_bucket_delta_pressure",
    ]
    micro_cols: list[str] = []
    for tf in _split_csv_names(getattr(args, "micro_timeframes", "")):
        tag = str(tf).lower()
        prefix = f"micro_{tag}_last{last_seconds}_"
        wanted = [
            "matched_rows", "window_start", "window_end_exclusive", "first_micro_ts", "last_micro_ts",
            "notional", "buy_sell_ratio", "buy_ratio", "delta_pressure",
            "large_delta_pressure", "max_trade_notional", "min_low", "max_high", "last_close",
        ]
        micro_cols.extend([f"{prefix}{x}" for x in wanted if f"{prefix}{x}" in events.columns])
    cols = [c for c in base_cols + micro_cols if c in events.columns]
    if not cols:
        return pd.DataFrame()
    out = events[cols].copy()
    match_cols = [c for c in out.columns if c.endswith("_matched_rows")]
    if match_cols:
        out["any_micro_matched"] = False
        for c in match_cols:
            out["any_micro_matched"] = out["any_micro_matched"] | (pd.to_numeric(out[c], errors="coerce").fillna(0) > 0)
        out = out.sort_values(["any_micro_matched", "signal_time"], ascending=[False, True])
    return out.head(int(limit)).reset_index(drop=True)


def _quantile_record(series: pd.Series, prefix: str) -> dict[str, object]:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {f"{prefix}_{k}": np.nan for k in ["mean", "median", "p25", "p75", "p10", "p90"]}
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_median": float(x.median()),
        f"{prefix}_p25": float(x.quantile(0.25)),
        f"{prefix}_p75": float(x.quantile(0.75)),
        f"{prefix}_p10": float(x.quantile(0.10)),
        f"{prefix}_p90": float(x.quantile(0.90)),
    }


def build_a1_feature_distribution_report(events: pd.DataFrame, trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Distribution drilldown for A1-only rescue ideas.

    This does not select parameters.  It compares A0, A1-only, winners and
    losers across already available trade-bar/footprint/micro fields, so we can
    see whether features such as volume/notional pressure, CVD/delta, or future
    micro head/tail proxies are worth turning into causal filters.
    """
    if events.empty:
        return pd.DataFrame()
    fixed = build_fixed_candidate_masks(events)
    a = fixed.get("A_spike_close_large_share", {}).get("mask", pd.Series(False, index=events.index)).fillna(False).astype(bool)
    spike = _safe_num(events.get("down_spike_pct", np.nan), events.index)
    fp_abs = _safe_num(events.get("fp_max_bucket_abs_delta_pressure", np.nan), events.index) >= float(getattr(args, "fp_abs_delta_high_min", 0.60))
    masks = {
        "A0_all": a & (spike >= 0.0100),
        "A0_fp_abs_delta_high": a & (spike >= 0.0100) & fp_abs,
        "A1_only_080_100": a & (spike >= 0.0080) & (spike < 0.0100),
        "A1_only_fp_abs_delta_high": a & (spike >= 0.0080) & (spike < 0.0100) & fp_abs,
    }
    if not trades.empty and "variant_name" in trades.columns and "signal_time" in trades.columns:
        # Add realised outcomes for the baseline A1-only and A0 variants when available.
        outcome_map = trades.loc[trades["variant_name"].isin([
            "A0_fp_abs_delta_high__single_swing__next_open__time48__no_stop",
            "A1_only_fp_abs_delta_high__single_swing__next_open__time48__no_stop",
        ]), ["signal_time", "variant_name", "net_return_on_equity", "mae_on_equity", "mfe_on_equity"]].copy()
    else:
        outcome_map = pd.DataFrame()

    feature_cols = [
        "down_spike_pct", "close_pos_in_bar", "large_trade_share", "notional",
        "buy_notional", "sell_notional", "delta_notional", "cvd_notional",
        "buy_volume", "sell_volume", "delta_volume", "cvd_volume", "taker_buy_ratio",
        "fp_max_bucket_abs_delta_pressure", "fp_low_bucket_delta_pressure",
        "range_r0025_delta_pressure", "range_r0020_delta_pressure",
    ]
    for tf in _split_csv_names(getattr(args, "micro_timeframes", "")):
        tag = str(tf).lower()
        prefix = f"micro_{tag}_last{int(getattr(args, 'micro_last_seconds', 20))}_"
        feature_cols.extend([
            f"{prefix}buy_sell_ratio", f"{prefix}buy_ratio", f"{prefix}delta_pressure",
            f"{prefix}large_delta_pressure", f"{prefix}max_trade_notional", f"{prefix}matched_rows",
        ])
    feature_cols = [c for c in feature_cols if c in events.columns]

    rows: list[dict[str, object]] = []
    for name, mask in masks.items():
        part = events.loc[mask].copy()
        rec: dict[str, object] = {"bucket": name, "events": int(len(part))}
        for col in feature_cols:
            rec.update(_quantile_record(part[col], col))
        rows.append(rec)
        if not outcome_map.empty and "signal_time" in part.columns:
            merged = outcome_map.merge(part[["signal_time"]], on="signal_time", how="inner")
            if not merged.empty:
                for sub_name, sub_mask in {"winners": merged["net_return_on_equity"] > 0, "losers": merged["net_return_on_equity"] <= 0}.items():
                    times = set(pd.to_datetime(merged.loc[sub_mask, "signal_time"]).astype("int64"))
                    pp = part.loc[pd.to_datetime(part["signal_time"]).astype("int64").isin(times)]
                    rr: dict[str, object] = {"bucket": f"{name}_{sub_name}", "events": int(len(pp))}
                    for col in feature_cols:
                        rr.update(_quantile_record(pp[col], col))
                    rows.append(rr)
    return pd.DataFrame(rows)


def build_incremental_overlap_report(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Measure whether A1 variants are mostly overlapping with the A0 mainline."""
    if trades.empty or "variant_name" not in trades.columns or "signal_time" not in trades.columns:
        return pd.DataFrame()
    base_name = "A0_fp_abs_delta_high__single_swing__next_open__time48__no_stop"
    challengers = [
        "A1_fp_abs_delta_high__single_swing__next_open__swing_trail_after18_time96__no_stop",
        "A1_only_fp_abs_delta_high__single_swing__next_open__swing_trail_after18_time96__no_stop",
        "A1_only_fp_abs_delta_high__single_swing__next_open__time48__no_stop",
        "A0_fp_abs_delta_high__single_swing__next_open_add_equal2_050__time48__no_stop",
        "A0_fp_abs_delta_high__single_swing__next_open_delay1__time48__no_stop",
        "A0_fp_abs_delta_high__single_swing__next_open_delay2__time48__no_stop",
    ]
    base = trades.loc[trades["variant_name"].eq(base_name)].copy()
    base_times = set(pd.to_datetime(base["signal_time"]).astype("int64")) if not base.empty else set()
    rows: list[dict[str, object]] = []
    for name in challengers:
        cur = trades.loc[trades["variant_name"].eq(name)].copy()
        if cur.empty:
            rows.append({"base_variant": base_name, "challenger_variant": name, "challenger_found": False})
            continue
        cur_times = pd.to_datetime(cur["signal_time"]).astype("int64")
        overlap_mask = cur_times.isin(base_times).to_numpy(dtype=bool)
        unique = cur.loc[~overlap_mask].copy()
        overlap = cur.loc[overlap_mask].copy()
        all_stats = _summarize_trade_slice_for_overlap(cur, args)
        unique_stats = _summarize_trade_slice_for_overlap(unique, args)
        overlap_stats = _summarize_trade_slice_for_overlap(overlap, args)
        row = {
            "base_variant": base_name,
            "challenger_variant": name,
            "challenger_found": True,
            "base_trades": int(len(base)),
            "challenger_trades": int(len(cur)),
            "overlap_trades": int(len(overlap)),
            "unique_trades": int(len(unique)),
            "overlap_rate_vs_challenger": float(len(overlap) / max(1, len(cur))),
            "unique_rate_vs_challenger": float(len(unique) / max(1, len(cur))),
        }
        for prefix, stats in [("all", all_stats), ("unique", unique_stats), ("overlap", overlap_stats)]:
            for k in ["return_total", "profit_factor", "win_rate", "max_drawdown", "worst_trade", "median_return", "mae_median", "mfe_median"]:
                if k in stats:
                    row[f"{prefix}_{k}"] = stats[k]
        rows.append(row)
    return pd.DataFrame(rows)

def build_equal2_risk_scaling(summary: pd.DataFrame, max_scale: float = 3.0) -> pd.DataFrame:
    """Approximate risk-scaling report for equal2 liquidity-zone variants.

    This does not change strategy results.  It asks: if an equal2 version has
    materially lower drawdown than the matching single_swing variant, what
    simple scale multiplier would equalize drawdown, and what linearized return
    would that imply?  Real promotion still needs a dedicated position-sizing
    backtest.
    """
    if summary.empty:
        return pd.DataFrame()
    keys = ["candidate_layer", "entry_mode", "exit_mode", "stop_name"]
    base = summary.loc[summary.get("support_mode", "").eq("single_swing")].copy()
    eq2 = summary.loc[summary.get("support_mode", "").eq("equal2_020")].copy()
    if base.empty or eq2.empty:
        return pd.DataFrame()
    b = base.set_index(keys)
    rows: list[dict[str, object]] = []
    for _, row in eq2.iterrows():
        key = tuple(row.get(k) for k in keys)
        if key not in b.index:
            continue
        br = b.loc[key]
        if isinstance(br, pd.DataFrame):
            br = br.iloc[0]
        base_dd = abs(_safe_float(br.get("max_drawdown", np.nan)))
        eq_dd = abs(_safe_float(row.get("max_drawdown", np.nan)))
        if not math.isfinite(base_dd) or not math.isfinite(eq_dd) or eq_dd <= 0:
            continue
        scale_to_base_dd = min(float(max_scale), base_dd / eq_dd)
        rows.append({
            "equal2_variant_name": row.get("variant_name"),
            "matching_single_swing_variant": br.get("variant_name"),
            "candidate_layer": row.get("candidate_layer"),
            "entry_mode": row.get("entry_mode"),
            "exit_mode": row.get("exit_mode"),
            "stop_name": row.get("stop_name"),
            "equal2_trades": row.get("trades"),
            "single_trades": br.get("trades"),
            "equal2_return": row.get("return_total"),
            "single_return": br.get("return_total"),
            "equal2_pf": row.get("profit_factor"),
            "single_pf": br.get("profit_factor"),
            "equal2_max_drawdown": row.get("max_drawdown"),
            "single_max_drawdown": br.get("max_drawdown"),
            "scale_to_single_drawdown_cap3": scale_to_base_dd,
            "linearized_scaled_equal2_return_cap3": _safe_float(row.get("return_total", np.nan)) * scale_to_base_dd if math.isfinite(_safe_float(row.get("return_total", np.nan))) else np.nan,
            "note": "approximation only; requires real sizing backtest before use",
        })
    return pd.DataFrame(rows).sort_values(["linearized_scaled_equal2_return_cap3", "equal2_pf"], ascending=[False, False]).reset_index(drop=True) if rows else pd.DataFrame()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_variants(args: argparse.Namespace) -> list[UpgradeVariant]:
    layers = _split_csv_names(args.candidate_layers)
    supports = _split_csv_names(args.support_modes)
    entries = _split_csv_names(args.entry_modes)
    exits = _split_csv_names(args.exit_modes)
    stops = parse_stop_specs(args.upgrade_stop_specs)
    variants: list[UpgradeVariant] = []
    for layer in layers:
        for support in supports:
            for entry in entries:
                for exit_mode in exits:
                    for stop in stops:
                        name = f"{layer}__{support}__{entry}__{exit_mode}__{stop.name}"
                        variants.append(UpgradeVariant(name, layer, support, entry, exit_mode, stop))
    return variants


def run_upgrade_jobs(bars: pd.DataFrame, events: pd.DataFrame, variants: list[UpgradeVariant], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[cache] building reusable market arrays", flush=True)
    market = build_market_cache(bars, args)
    print("[cache] precomputing candidate/support masks", flush=True)
    layer_masks = build_candidate_layer_masks(events, args)
    layer_mask_cache = {k: v.fillna(False).astype(bool) for k, v in layer_masks.items()}
    support_mask_cache = {mode: build_support_mask(events, mode, args).fillna(False).astype(bool) for mode in sorted(set(v.support_mode for v in variants))}
    event_subset_cache: dict[tuple[str, str], pd.DataFrame] = {}

    summary_rows: list[dict[str, object]] = []
    trade_parts: list[pd.DataFrame] = []
    progress = ProgressReporter(label="[upgrade] variants", total=len(variants), every=1, enabled=not bool(args.no_progress))
    for i, variant in enumerate(variants, start=1):
        # Keep job visibility, but avoid making logs explode with repeated long lines
        # when users run thousands of variants.
        cache_key = (variant.candidate_layer, variant.support_mode)
        part = event_subset_cache.get(cache_key)
        if part is None:
            layer_mask = layer_mask_cache.get(variant.candidate_layer, pd.Series(False, index=events.index))
            support_mask = support_mask_cache.get(variant.support_mode, pd.Series(False, index=events.index))
            part = events.loc[layer_mask & support_mask].copy().sort_values("signal_time")
            event_subset_cache[cache_key] = part
        trades, counters = simulate_upgrade_variant(bars, part, variant, args, market=market)
        if not trades.empty:
            trade_parts.append(trades)
        rec = summarize_trades(trades, args, counters)
        rec.update({
            "variant_name": variant.variant_name,
            "candidate_layer": variant.candidate_layer,
            "support_mode": variant.support_mode,
            "entry_mode": variant.entry_mode,
            "exit_mode": variant.exit_mode,
            "stop_name": variant.stop_spec.name,
            "stop_mode": variant.stop_spec.mode,
        })
        summary_rows.append(rec)
        progress.update(i)
    progress.close()
    all_trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    if not all_trades.empty and "signal_time" in all_trades.columns:
        print("[cache] attaching event/context columns to all trades once", flush=True)
        cols = [c for c in events.columns if c not in all_trades.columns or c == "signal_time"]
        addon = events[cols].drop_duplicates("signal_time") if cols and "signal_time" in events.columns else pd.DataFrame()
        if not addon.empty:
            all_trades = all_trades.merge(addon, on="signal_time", how="left", suffixes=("", "_event"))
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["profit_factor", "return_total", "trades"], ascending=[False, False, False]).reset_index(drop=True)
    return all_trades, summary


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {SCRIPT_NAME}", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)
    from research.low_sweep_panic_reversal_strategy_backtest_probe import load_trade_bars  # local import keeps tests light

    bars = load_trade_bars(args)
    events = prepare_studied_events(bars, args)
    events = attach_support_zone_metrics(events, bars, args)
    events = attach_range_context(events, args)
    events = attach_footprint_context(events, args)
    events = attach_micro_trade_context(events, args)
    variants = build_variants(args)
    print(f"[variants] total={len(variants):,}", flush=True)
    trades, summary = run_upgrade_jobs(bars, events, variants, args)

    if not summary.empty:
        summary["edge_candidate"] = (
            (pd.to_numeric(summary["trades"], errors="coerce") >= int(args.min_trades_for_upgrade_edge))
            & (pd.to_numeric(summary["return_total"], errors="coerce") > 0)
            & (pd.to_numeric(summary["median_return"], errors="coerce") > 0)
            & (pd.to_numeric(summary["profit_factor"], errors="coerce") >= float(args.min_pf_for_upgrade_edge))
        )

    yearly = summarize_by_period(trades, "year")
    monthly = summarize_by_period(trades, "month")
    entry_compare = summarize_compare(summary, ["entry_mode"])
    exit_compare = summarize_compare(summary, ["exit_mode"])
    support_compare = summarize_compare(summary, ["support_mode"])
    layer_compare = summarize_compare(summary, ["candidate_layer"])
    mae_compare = summary[[c for c in ["variant_name", "candidate_layer", "support_mode", "entry_mode", "exit_mode", "stop_name", "trades", "return_total", "profit_factor", "win_rate", "mae_mean", "mae_median", "mae_p05", "mfe_mean", "mfe_median"] if c in summary.columns]].copy() if not summary.empty else pd.DataFrame()
    context_cols = [c for c in trades.columns if c.startswith("range_") or c.startswith("fp_") or c.startswith("micro_")]
    context_breakdown = build_context_breakdown(trades, context_cols, min_trades=20)
    registry = build_edge_registry(summary, yearly, monthly, args)
    consistency_audit = build_baseline_consistency_audit(bars, events, summary, args)
    path_timing_compare = build_path_timing_compare(summary)
    equal2_risk_scaling = build_equal2_risk_scaling(summary)
    micro_coverage = build_micro_context_coverage(events, args)
    micro_load_diagnostics = build_micro_load_diagnostics(args)
    micro_match_debug = build_micro_match_debug(args)
    micro_feature_sample = build_micro_feature_sample(events, args, limit=300)
    incremental_overlap = build_incremental_overlap_report(trades, args)
    a1_feature_distribution = build_a1_feature_distribution_report(events, trades, args)

    event_cols = [
        "signal_time", "event_name", "variant", "swing_level", "swing_age", "down_spike_pct", "close_pos_in_bar", "large_trade_share",
        "cluster_touch_count_020", "cluster_touch_count_030", "cluster_oldest_age_bars_020", "cluster_oldest_age_bars_030", "session_bucket",
    ]
    event_sample = events[[c for c in event_cols if c in events.columns]].head(int(args.save_events)).copy() if not events.empty else pd.DataFrame()
    trades_sample = trades.head(int(args.save_trades)).copy() if (not trades.empty and int(args.save_trades) > 0) else pd.DataFrame()

    write_csv(event_sample, out_dir / "01_events_enriched_sample.csv", "events_sample")
    write_csv(trades_sample, out_dir / "02_trades_sample.csv", "trades_sample")
    write_csv(summary, out_dir / "03_variant_summary.csv", "summary")
    write_csv(yearly, out_dir / "04_yearly.csv", "yearly")
    write_csv(monthly, out_dir / "05_monthly.csv", "monthly")
    write_csv(entry_compare, out_dir / "06_entry_mode_compare.csv", "entry_compare")
    write_csv(exit_compare, out_dir / "07_exit_mode_compare.csv", "exit_compare")
    write_csv(support_compare, out_dir / "08_support_mode_compare.csv", "support_compare")
    write_csv(layer_compare, out_dir / "09_candidate_layer_compare.csv", "layer_compare")
    write_csv(mae_compare, out_dir / "10_mae_mfe_compare.csv", "mae_compare")
    write_csv(context_breakdown, out_dir / "11_range_footprint_context_breakdown.csv", "context_breakdown")
    write_csv(registry, out_dir / "12_upgrade_edge_registry.csv", "edge_registry")
    write_csv(consistency_audit, out_dir / "13_baseline_consistency_audit.csv", "baseline_consistency_audit")
    write_csv(path_timing_compare, out_dir / "14_path_timing_compare.csv", "path_timing_compare")
    write_csv(equal2_risk_scaling, out_dir / "15_equal2_risk_scaling.csv", "equal2_risk_scaling")
    write_csv(micro_coverage, out_dir / "16_micro_context_coverage.csv", "micro_context_coverage")
    write_csv(micro_load_diagnostics, out_dir / "17_micro_load_diagnostics.csv", "micro_load_diagnostics")
    write_csv(micro_feature_sample, out_dir / "18_micro_feature_sample.csv", "micro_feature_sample")
    write_csv(incremental_overlap, out_dir / "19_incremental_overlap.csv", "incremental_overlap")
    write_csv(a1_feature_distribution, out_dir / "20_a1_feature_distribution.csv", "a1_feature_distribution")
    write_csv(micro_match_debug, out_dir / "21_micro_match_debug.csv", "micro_match_debug")

    meta = {
        "script": SCRIPT_NAME,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "variants": len(variants),
        "events": int(len(events)),
        "trades_rows_saved": int(len(trades_sample)),
        "context_sources": _split_csv_names(args.context_sources),
        "micro_timeframes": _split_csv_names(getattr(args, "micro_timeframes", "")),
        "causal_guards": [
            "confirmed swing lows use pivot_right confirmation plus one-bar delay upstream",
            "rolling thresholds inherited from no-leakage probe use shift(1).rolling(...)",
            "reclaim entries wait for a closed confirmation bar then enter next open",
            "range/footprint context uses merge_asof on closed range-bar end_ts <= signal_time",
            "micro trade-bar context only uses sub-bars inside the already-closed 1m signal bar",
            "micro trade bars are loaded by a sliding current-month plus next-month cache; next-month rows may be resident but every feature slices only the closed parent 1m signal bar",
            "--micro-load-mode local is cache-only; --micro-load-mode fetch may use monthly fetch_data_by_date_range after coverage is prebuilt",
            "no order book data is used",
        ],
        "consistency_audit": "13_baseline_consistency_audit.csv compares key A1 baseline rows against the existing validation engine",
        "support_modes_note": "aged_ge_N is now dynamic; defaults are aged_ge_12/24/36 because prior events had swing_age below 60",
        "context_filter_note": "footprint/range candidate layers use fixed interpretable thresholds, not full-sample qcut buckets",
        "performance_guards": [
            "market OHLCV arrays and reclaim rolling-volume baseline are cached once",
            "candidate/support masks are precomputed once and reused across variants",
            "event/context columns are merged into trade output once after all variants",
            "optimizations are performance-only and do not change signal timing or path logic",
            "micro 5s/10s context is loaded month-by-month, never as a 2022-2026 full table",
            "detailed trade sample output defaults to zero rows to avoid huge reports; use --save-trades when needed",
        ],
        "new_reports": [
            "14_path_timing_compare.csv",
            "15_equal2_risk_scaling.csv",
            "16_micro_context_coverage.csv",
            "17_micro_load_diagnostics.csv",
            "18_micro_feature_sample.csv",
            "19_incremental_overlap.csv",
            "20_a1_feature_distribution.csv",
            "21_micro_match_debug.csv",
        ],
    }
    print(f"[write] meta -> {out_dir / '19_lab_meta.json'}", flush=True)
    (out_dir / "19_lab_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[done] A upgrade research complete", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
