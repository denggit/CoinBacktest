#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unconsumed swing-low support-pool research for Low Sweep MF upgrade.

The current formal MF sleeve is based on one latest confirmed low swing.  This
lab asks a narrow question: if the signal bar can sweep one of several recently
confirmed low swings, can we add trades without degrading the existing A0
footprint edge?

Research-only guarantees:
- no portfolio/live strategy code is changed;
- pivot lows are available only after ``pivot_right`` bars plus one extra bar,
  matching ``confirmed_swing_lows`` from the existing no-leakage pipeline;
- entries still use closed signal bars and future opens through the existing
  low-sweep simulator;
- footprint/range context attachment reuses the existing causal merge_asof path;
- heavy work is cached: features, support events, candidate/support masks and
  market arrays are built once, then reused across variants/stress runs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.focused_low_sweep_reversal_event_lab import _parse_number_list  # noqa: E402
from research.low_sweep_a_upgrade_research import (  # noqa: E402
    UpgradeVariant,
    _profit_factor,
    _safe_float,
    _split_csv_names,
    attach_footprint_context,
    attach_micro_trade_context,
    attach_range_context,
    build_candidate_layer_masks,
    build_market_cache,
    parse_args as _upgrade_parse_args,
    parse_stop_specs,
    simulate_probe_compatible_trade,
    simulate_upgrade_trade,
    summarize_by_period,
    summarize_compare,
    summarize_trades,
    write_csv,
)
from research.low_sweep_panic_reversal_strategy_backtest_probe import load_trade_bars  # noqa: E402
from research.low_sweep_panic_reversal_strategy_probe import (  # noqa: E402
    add_filter_bins,
    attach_extra_features_to_events,
    build_enriched_features,
)
from src.research_common.progress import ProgressReporter  # noqa: E402

SCRIPT_NAME = "low_sweep_unconsumed_support_research"
DEFAULT_OUT_DIR = "data/reports/research/low_sweep_unconsumed_support_research_tradebar_1m"


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    own = argparse.ArgumentParser(add_help=False)
    own.add_argument(
        "--multi-support-modes",
        default="rank1_latest,first_swept_top2,latest_unconsumed_swept,nearest_unconsumed_swept,deepest_unconsumed_swept,all_unconsumed_swept,older_unconsumed_swept",
        help=(
            "Comma list. rank/first_swept_topN use recent confirmed lows as the old bounded baseline. "
            "*_unconsumed_* modes use a stateful support pool: a confirmed swing low is active until the first later bar consumes it."
        ),
    )
    own.add_argument("--max-support-rank", type=int, default=5, help="Number of most recent confirmed low pivots to inspect per signal bar for recent/topN baseline modes.")
    own.add_argument("--support-generation-mode", choices=["recent", "unconsumed", "both"], default="both", help="Generate recent/topN events, unconsumed support-pool events, or both for baseline comparison.")
    own.add_argument("--consume-breakout-pct", default="", help="Consumption threshold for unconsumed support pool. Empty uses min(--support-breakout-pcts or --breakout-pcts). A support is consumed when low <= level*(1-pct).")
    own.add_argument("--support-breakout-pcts", default="", help="Optional override. Empty inherits --breakout-pcts.")
    own.add_argument("--support-spike-pcts", default="", help="Optional override. Empty inherits --spike-pcts.")
    own.add_argument("--support-max-swing-ages", default="", help="Optional override. Empty inherits --max-swing-ages.")
    own.add_argument("--support-min-prominence-pcts", default="", help="Optional override. Empty inherits --min-swing-prominence-pcts.")
    own.add_argument("--stress-top-n", type=int, default=25, help="Run cost/delay stress for top N non-baseline variants plus baseline.")
    own.add_argument("--stress-cost-mults", default="1.5,2.0", help="Extra cost multipliers for selected variants. Base 1.0 is already in summary.")
    own.add_argument("--stress-delay-bars", default="1,2", help="Extra delay bars beyond next_open for selected next_open variants.")
    own.add_argument("--save-full-multi-events", action="store_true", help="Save full generated multi-swing event table. Default saves sample only.")
    known, rest = own.parse_known_args(argv)

    defaults = [
        "--out-dir",
        DEFAULT_OUT_DIR,
        "--candidate-layers",
        "A0_fp_abs_delta_high",
        "--support-modes",
        "single_swing",
        "--entry-modes",
        "next_open",
        "--exit-modes",
        "time48,mfe_lock_15_05_time48",
        "--upgrade-stop-specs",
        "no_stop",
        "--context-sources",
        "trade_bar,footprint",
        "--micro-timeframes",
        "",
        "--save-trades",
        "100000",
        "--save-events",
        "5000",
    ]
    args = _upgrade_parse_args(defaults + list(rest))
    for k, v in vars(known).items():
        setattr(args, k, v)
    # The old parser still has --support-modes; keep it out of this lab's
    # variant grid to avoid accidentally calling the old single/equal support
    # filters.  ``multi_support_modes`` is the only support dimension here.
    args.support_modes = args.multi_support_modes
    return args


# ---------------------------------------------------------------------------
# Causal multi-swing event generation
# ---------------------------------------------------------------------------


def _timeframe_to_minutes(timeframe: str) -> int:
    tf = str(timeframe).strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("H"):
        return int(tf[:-1]) * 60
    if tf.endswith("D"):
        return int(tf[:-1]) * 1440
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _parse_float_list_or_inherit(raw: str, fallback: str, *, name: str) -> list[float]:
    src = str(raw).strip() or str(fallback)
    return [float(x) for x in _parse_number_list(src, cast=float, name=name, allow_zero=True)]


def _parse_int_list_or_inherit(raw: str, fallback: str, *, name: str) -> list[int]:
    src = str(raw).strip() or str(fallback)
    return [int(x) for x in _parse_number_list(src, cast=int, name=name)]


def extract_causal_low_pivots(features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Return all low pivots with the first bar where they are tradably known.

    This intentionally mirrors ``confirmed_swing_lows``:
    - pivot center uses right-side bars for confirmation;
    - the level becomes usable only at ``pivot_pos + pivot_right + 1``.
    """
    low = pd.to_numeric(features["low"], errors="coerce")
    high = pd.to_numeric(features["high"], errors="coerce")
    left = int(args.pivot_left)
    right = int(args.pivot_right)
    pos = np.arange(len(features), dtype=int)

    left_low = low.shift(1).rolling(left, min_periods=left).min()
    right_low = low.iloc[::-1].shift(1).rolling(right, min_periods=right).min().iloc[::-1]
    pivot_mask = (low < left_low) & (low <= right_low)

    window = left + right + 1
    local_high = high.rolling(window, center=True, min_periods=window).max()
    prominence = local_high / low.replace(0.0, np.nan) - 1.0

    pivot_idx = np.flatnonzero(pivot_mask.fillna(False).to_numpy(dtype=bool))
    if pivot_idx.size == 0:
        return pd.DataFrame(columns=["pivot_pos", "available_pos", "pivot_time", "available_time", "swing_level", "swing_prominence_pct"])
    available_pos = pivot_idx + right + 1
    valid = available_pos < len(features)
    pivot_idx = pivot_idx[valid]
    available_pos = available_pos[valid]
    return pd.DataFrame(
        {
            "pivot_pos": pivot_idx.astype(int),
            "available_pos": available_pos.astype(int),
            "pivot_time": features.index[pivot_idx],
            "available_time": features.index[available_pos],
            "swing_level": low.iloc[pivot_idx].to_numpy(dtype=float),
            "swing_prominence_pct": prominence.iloc[pivot_idx].to_numpy(dtype=float),
        }
    ).dropna(subset=["swing_level", "swing_prominence_pct"]).reset_index(drop=True)


def _variant_mask(features: pd.DataFrame, variant: str) -> pd.Series:
    if variant == "fade_close_through":
        return pd.Series(True, index=features.index)  # support-level check is done per pivot below
    if variant == "reject":
        return pd.Series(True, index=features.index)
    if variant == "wick":
        return pd.to_numeric(features.get("lower_wick_frac", np.nan), errors="coerce") >= 0.0
    raise ValueError(f"Unsupported low-sweep variant: {variant}")


def _close_condition(row: pd.Series, variant: str, level: float, args: argparse.Namespace) -> bool:
    close = float(row.get("close", np.nan))
    low = float(row.get("low", np.nan))
    if not math.isfinite(close) or not math.isfinite(low) or not math.isfinite(level) or level <= 0:
        return False
    if variant == "fade_close_through":
        return close <= level * (1.0 - float(args.close_through_buffer_pct))
    if variant == "reject":
        return close > level
    if variant == "wick":
        wick = float(row.get("lower_wick_frac", np.nan))
        return math.isfinite(wick) and wick >= float(args.wick_min_frac)
    return False


def _event_from_feature_row(
    *,
    ts: pd.Timestamp,
    row: pd.Series,
    level: float,
    support_rank: int,
    swept_order: int,
    pivot: pd.Series,
    spike_pct: float,
    breakout_pct: float,
    max_age: int,
    min_prom: float,
    variant: str,
) -> dict[str, object]:
    pivot_pos = int(pivot["pivot_pos"])
    signal_pos = int(row["_bar_pos"])
    support_age = float(signal_pos - pivot_pos)
    suffix = (
        f"sp{int(spike_pct * 10000):04d}_br{int(breakout_pct * 10000):04d}_"
        f"age{int(max_age)}_prom{int(min_prom * 10000):04d}_rank{int(support_rank)}"
    )
    return {
        "signal_time": ts,
        "side": 1,
        "event_name": f"multi_low_{variant}_{suffix}",
        "event_family": f"multi_low_sweep_{variant}",
        "variant": variant,
        "support_rank": int(support_rank),
        "swept_order": int(swept_order),
        "support_source": f"rank{int(support_rank)}",
        "support_pivot_time": pivot["pivot_time"],
        "support_available_time": pivot["available_time"],
        "support_pivot_pos": int(pivot_pos),
        "support_available_pos": int(pivot["available_pos"]),
        "support_age": support_age,
        "support_prominence_pct": float(pivot["swing_prominence_pct"]),
        "swing_level": float(level),
        "sweep_extreme": float(row["low"]),
        "structural_stop_level": float(min(float(row["low"]), float(level))),
        "swing_age": support_age,
        "swing_prominence_pct": float(pivot["swing_prominence_pct"]),
        "down_spike_pct": float(row.get("down_spike_pct", np.nan)),
        "volume_ratio": float(row.get("volume_ratio", np.nan)),
        "trades_count_ratio": float(row.get("trades_count_ratio", np.nan)),
        "volume_spike": bool(row.get("volume_spike", False)),
        "atr_pct": float(row.get("atr_pct", np.nan)),
        "delta_notional": float(row.get("delta_notional", np.nan)),
        "delta_notional_ratio": float(row.get("delta_notional_ratio", np.nan)),
        "delta_notional_z": float(row.get("delta_notional_z", np.nan)),
        "cvd_notional_change": float(row.get("cvd_notional_change", np.nan)),
        "taker_buy_ratio": float(row.get("taker_buy_ratio", np.nan)),
        "large_delta_notional": float(row.get("large_delta_notional", np.nan)),
        "lower_wick_frac": float(row.get("lower_wick_frac", np.nan)),
        "close_pos_in_bar": float(row.get("close_pos_in_bar", np.nan)),
        "session_hour": int(row.get("session_hour", -1)) if pd.notna(row.get("session_hour", np.nan)) else -1,
        "session_bucket": str(row.get("session_bucket", "NA")),
        "weekday": int(row.get("weekday", -1)) if pd.notna(row.get("weekday", np.nan)) else -1,
        "close": float(row["close"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "spike_threshold_pct": float(spike_pct),
        "breakout_threshold_pct": float(breakout_pct),
        "max_swing_age": int(max_age),
        "min_swing_prominence_pct": float(min_prom),
    }


def build_multi_swing_events(features: pd.DataFrame, pivots: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if pivots.empty or features.empty:
        return pd.DataFrame()

    spike_pcts = _parse_float_list_or_inherit(getattr(args, "support_spike_pcts", ""), args.spike_pcts, name="support_spike_pcts")
    breakout_pcts = _parse_float_list_or_inherit(getattr(args, "support_breakout_pcts", ""), args.breakout_pcts, name="support_breakout_pcts")
    max_ages = _parse_int_list_or_inherit(getattr(args, "support_max_swing_ages", ""), args.max_swing_ages, name="support_max_swing_ages")
    min_proms = _parse_float_list_or_inherit(getattr(args, "support_min_prominence_pcts", ""), args.min_swing_prominence_pcts, name="support_min_prominence_pcts")
    variants = _split_csv_names(getattr(args, "variants", "fade_close_through"))
    max_rank = max(1, int(args.max_support_rank))

    f = features.copy()
    f["_bar_pos"] = np.arange(len(f), dtype=int)
    start_ts = pd.Timestamp(args.warmup_start_date)
    end_ts = pd.Timestamp(args.end_date) + pd.Timedelta(days=1)
    # Generate from warmup through end. Later we slice to formal start; warmup is
    # still needed only so rolling thresholds and support ages are correct.
    f = f.loc[(f.index >= start_ts) & (f.index < end_ts)].copy()

    min_spike = min(spike_pcts) if spike_pcts else 0.0
    broad = pd.to_numeric(f.get("down_spike_pct", np.nan), errors="coerce") >= float(min_spike)
    # Variant-specific close/reject/wick checks depend on the selected support
    # level, so the broad prefilter only uses bar-level panic information.
    candidate_positions = np.flatnonzero(broad.fillna(False).to_numpy(dtype=bool))

    pivot_avail = pd.to_numeric(pivots["available_pos"], errors="coerce").to_numpy(dtype=int)
    pivot_pos = pd.to_numeric(pivots["pivot_pos"], errors="coerce").to_numpy(dtype=int)
    pivot_level = pd.to_numeric(pivots["swing_level"], errors="coerce").to_numpy(dtype=float)
    pivot_prom = pd.to_numeric(pivots["swing_prominence_pct"], errors="coerce").to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    progress = ProgressReporter(
        label="[events] multi-swing supports",
        total=len(candidate_positions),
        every=max(1, int(getattr(args, "progress_every", 1000))),
        enabled=not bool(getattr(args, "no_progress", False)),
    )
    # Iterate only over broad panic bars.  For each bar we scan backward through
    # confirmed pivots and stop once max_rank recent supports have been checked.
    for done, bar_pos in enumerate(candidate_positions, start=1):
        ts = f.index[int(bar_pos)]
        row = f.iloc[int(bar_pos)]
        low = float(row.get("low", np.nan))
        close = float(row.get("close", np.nan))
        if not (math.isfinite(low) and math.isfinite(close)):
            progress.update(done)
            continue
        hi = int(np.searchsorted(pivot_avail, int(row["_bar_pos"]), side="right"))
        if hi <= 0:
            progress.update(done)
            continue

        checked = 0
        swept_for_bar: list[tuple[int, int, float]] = []  # pivot row, support_rank, level
        j = hi - 1
        while j >= 0 and checked < max_rank:
            level = float(pivot_level[j])
            if not math.isfinite(level) or level <= 0:
                j -= 1
                continue
            checked += 1
            support_rank = checked
            age = int(row["_bar_pos"]) - int(pivot_pos[j])
            # Broad pre-check only; exact max_age/min_prom/breakout threshold is
            # applied below so canonical specificity remains comparable to the
            # existing low-sweep event generator.
            if age >= int(args.min_swing_age) and low <= level:
                swept_for_bar.append((int(j), int(support_rank), float(level)))
            j -= 1
        if not swept_for_bar:
            progress.update(done)
            continue

        swept_order = 0
        for j, support_rank, level in swept_for_bar:
            swept_order += 1
            pivot = pivots.iloc[j]
            age = int(row["_bar_pos"]) - int(pivot_pos[j])
            prom = float(pivot_prom[j])
            for spike_pct in spike_pcts:
                if float(row.get("down_spike_pct", np.nan)) < float(spike_pct):
                    continue
                for breakout_pct in breakout_pcts:
                    if low > level * (1.0 - float(breakout_pct)):
                        continue
                    for max_age in max_ages:
                        if age > int(max_age):
                            continue
                        for min_prom in min_proms:
                            if prom < float(min_prom):
                                continue
                            for variant in variants:
                                if not _close_condition(row, variant, level, args):
                                    continue
                                rows.append(
                                    _event_from_feature_row(
                                        ts=ts,
                                        row=row,
                                        level=level,
                                        support_rank=support_rank,
                                        swept_order=swept_order,
                                        pivot=pivot,
                                        spike_pct=float(spike_pct),
                                        breakout_pct=float(breakout_pct),
                                        max_age=int(max_age),
                                        min_prom=float(min_prom),
                                        variant=variant,
                                    )
                                )
        progress.update(done)
    progress.close()

    events = pd.DataFrame(rows)
    if events.empty:
        return events
    variant_rank = {"fade_close_through": 0, "wick": 1, "reject": 2}
    events["signal_time"] = pd.to_datetime(events["signal_time"])
    events["variant_rank"] = events["variant"].map(variant_rank).fillna(9).astype(int)
    events["specificity_score"] = (
        pd.to_numeric(events["spike_threshold_pct"], errors="coerce").fillna(0) * 10_000
        + pd.to_numeric(events["min_swing_prominence_pct"], errors="coerce").fillna(0) * 10_000
        + pd.to_numeric(events["breakout_threshold_pct"], errors="coerce").fillna(0) * 10_000
        - pd.to_numeric(events["max_swing_age"], errors="coerce").fillna(999) * 0.01
        - pd.to_numeric(events["support_rank"], errors="coerce").fillna(99) * 0.0001
        - events["variant_rank"] * 0.001
    )
    # Keep one most-specific row per signal/support rank.  We intentionally do
    # not drop older ranks here; support modes decide whether rank2/3/etc are
    # traded.
    events = events.sort_values(["signal_time", "side", "support_rank", "specificity_score"], ascending=[True, True, True, False])
    events = events.drop_duplicates(["signal_time", "side", "support_rank"], keep="first").reset_index(drop=True)
    # Recompute swept_order after the exact event-definition filters and
    # canonical de-duplication.  Otherwise a broad rank1 sweep that later fails
    # max-age/prominence could incorrectly suppress a valid rank2 event in
    # first_swept_topN modes.
    events = events.sort_values(["signal_time", "side", "support_rank"]).reset_index(drop=True)
    events["swept_order"] = events.groupby(["signal_time", "side"], sort=False).cumcount() + 1
    start_ts = pd.Timestamp(args.start_date)
    end_ts = pd.Timestamp(args.end_date) + pd.Timedelta(days=1)
    events = events[(events["signal_time"] >= start_ts) & (events["signal_time"] < end_ts)].copy()
    events["year"] = events["signal_time"].dt.year
    return events.sort_values(["signal_time", "support_rank"]).reset_index(drop=True)



class _FirstLowLeIndex:
    """Segment tree for first bar index whose low <= threshold after start.

    This makes unconsumed-level detection O(pivots * log(bars)) instead of
    scanning every old support against every bar.
    """

    def __init__(self, lows: np.ndarray) -> None:
        vals = np.asarray(lows, dtype=float)
        vals = np.where(np.isfinite(vals), vals, np.inf)
        self.n = int(vals.size)
        size = 1
        while size < self.n:
            size <<= 1
        self.size = size
        self.tree = np.full(size * 2, np.inf, dtype=float)
        self.tree[size : size + self.n] = vals
        for i in range(size - 1, 0, -1):
            self.tree[i] = min(self.tree[i << 1], self.tree[(i << 1) | 1])

    def first_le(self, start: int, threshold: float) -> int:
        if self.n <= 0 or start >= self.n or not math.isfinite(float(threshold)):
            return -1
        return self._first_le(1, 0, self.size, max(0, int(start)), float(threshold))

    def _first_le(self, node: int, left: int, right: int, start: int, threshold: float) -> int:
        if right <= start or self.tree[node] > threshold:
            return -1
        if right - left == 1:
            return left if left < self.n else -1
        mid = (left + right) >> 1
        ans = self._first_le(node << 1, left, mid, start, threshold)
        if ans >= 0:
            return ans
        return self._first_le((node << 1) | 1, mid, right, start, threshold)


def _parse_consume_breakout_pct(args: argparse.Namespace, breakout_pcts: list[float]) -> float:
    raw = str(getattr(args, "consume_breakout_pct", "")).strip()
    if raw:
        return float(raw)
    return float(min(breakout_pcts) if breakout_pcts else 0.0)


def build_unconsumed_swing_events(features: pd.DataFrame, pivots: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Generate events from all confirmed low pivots that have not been consumed.

    A pivot enters the active support pool at ``available_pos``.  Its first later
    bar where ``low <= swing_level * (1 - consume_breakout_pct)`` is its only
    consumption bar.  If that bar also satisfies the low-sweep event definition,
    it can become a candidate.  This is different from topN/recent scanning:
    old unconsumed levels remain valid no matter how far back they are, subject
    only to the existing max_age/prominence filters in the variant grid.
    """
    if pivots.empty or features.empty:
        return pd.DataFrame()

    spike_pcts = _parse_float_list_or_inherit(getattr(args, "support_spike_pcts", ""), args.spike_pcts, name="support_spike_pcts")
    breakout_pcts = _parse_float_list_or_inherit(getattr(args, "support_breakout_pcts", ""), args.breakout_pcts, name="support_breakout_pcts")
    max_ages = _parse_int_list_or_inherit(getattr(args, "support_max_swing_ages", ""), args.max_swing_ages, name="support_max_swing_ages")
    min_proms = _parse_float_list_or_inherit(getattr(args, "support_min_prominence_pcts", ""), args.min_swing_prominence_pcts, name="support_min_prominence_pcts")
    variants = _split_csv_names(getattr(args, "variants", "fade_close_through"))
    consume_pct = _parse_consume_breakout_pct(args, breakout_pcts)

    f = features.copy().sort_index()
    f["_bar_pos"] = np.arange(len(f), dtype=int)
    low_arr = pd.to_numeric(f["low"], errors="coerce").to_numpy(dtype=float)
    first_low = _FirstLowLeIndex(low_arr)

    pivot_level = pd.to_numeric(pivots["swing_level"], errors="coerce").to_numpy(dtype=float)
    pivot_avail = pd.to_numeric(pivots["available_pos"], errors="coerce").to_numpy(dtype=int)
    pivot_pos = pd.to_numeric(pivots["pivot_pos"], errors="coerce").to_numpy(dtype=int)
    pivot_prom = pd.to_numeric(pivots["swing_prominence_pct"], errors="coerce").to_numpy(dtype=float)

    # Build the first-consumption map for every causal pivot once.
    consumed: list[tuple[int, int]] = []
    progress = ProgressReporter(
        label="[events] unconsumed support first-touch map",
        total=len(pivots),
        every=max(1, int(getattr(args, "progress_every", 1000))),
        enabled=not bool(getattr(args, "no_progress", False)),
    )
    for i in range(len(pivots)):
        level = float(pivot_level[i])
        avail = int(pivot_avail[i])
        if math.isfinite(level) and level > 0 and 0 <= avail < len(f):
            hit = first_low.first_le(avail, level * (1.0 - consume_pct))
            if hit >= 0:
                consumed.append((int(i), int(hit)))
        progress.update(i + 1)
    progress.close()
    if not consumed:
        return pd.DataFrame()

    start_ts = pd.Timestamp(args.start_date)
    end_ts = pd.Timestamp(args.end_date) + pd.Timedelta(days=1)
    min_spike = min(spike_pcts) if spike_pcts else 0.0

    rows: list[dict[str, object]] = []
    progress = ProgressReporter(
        label="[events] unconsumed support candidates",
        total=len(consumed),
        every=max(1, int(getattr(args, "progress_every", 1000))),
        enabled=not bool(getattr(args, "no_progress", False)),
    )
    for done, (pivot_i, signal_pos) in enumerate(consumed, start=1):
        if signal_pos < 0 or signal_pos >= len(f):
            progress.update(done)
            continue
        ts = f.index[int(signal_pos)]
        # Consumption before formal research start still removes the level.  It
        # just does not produce a tradable research event.
        if ts < start_ts or ts >= end_ts:
            progress.update(done)
            continue
        row = f.iloc[int(signal_pos)]
        if float(row.get("down_spike_pct", np.nan)) < float(min_spike):
            progress.update(done)
            continue
        level = float(pivot_level[pivot_i])
        age = int(signal_pos) - int(pivot_pos[pivot_i])
        prom = float(pivot_prom[pivot_i])
        if age < int(args.min_swing_age):
            progress.update(done)
            continue
        pivot = pivots.iloc[pivot_i]
        for spike_pct in spike_pcts:
            if float(row.get("down_spike_pct", np.nan)) < float(spike_pct):
                continue
            for breakout_pct in breakout_pcts:
                if float(row["low"]) > level * (1.0 - float(breakout_pct)):
                    continue
                for max_age in max_ages:
                    if age > int(max_age):
                        continue
                    for min_prom in min_proms:
                        if prom < float(min_prom):
                            continue
                        for variant in variants:
                            if not _close_condition(row, variant, level, args):
                                continue
                            rows.append(
                                _event_from_feature_row(
                                    ts=ts,
                                    row=row,
                                    level=level,
                                    support_rank=0,
                                    swept_order=0,
                                    pivot=pivot,
                                    spike_pct=float(spike_pct),
                                    breakout_pct=float(breakout_pct),
                                    max_age=int(max_age),
                                    min_prom=float(min_prom),
                                    variant=variant,
                                )
                            )
        progress.update(done)
    progress.close()

    events = pd.DataFrame(rows)
    if events.empty:
        return events
    variant_rank = {"fade_close_through": 0, "wick": 1, "reject": 2}
    events["signal_time"] = pd.to_datetime(events["signal_time"])
    events["variant_rank"] = events["variant"].map(variant_rank).fillna(9).astype(int)
    events["specificity_score"] = (
        pd.to_numeric(events["spike_threshold_pct"], errors="coerce").fillna(0) * 10_000
        + pd.to_numeric(events["min_swing_prominence_pct"], errors="coerce").fillna(0) * 10_000
        + pd.to_numeric(events["breakout_threshold_pct"], errors="coerce").fillna(0) * 10_000
        - pd.to_numeric(events["max_swing_age"], errors="coerce").fillna(999) * 0.01
        - events["variant_rank"] * 0.001
    )
    # One row per consumed support per signal bar after choosing the most
    # specific threshold combination.  We do not collapse different supports on
    # the same bar; support modes decide which one, if any, is traded.
    events = events.sort_values(["signal_time", "side", "support_pivot_pos", "specificity_score"], ascending=[True, True, False, False])
    events = events.drop_duplicates(["signal_time", "side", "support_pivot_pos"], keep="first").reset_index(drop=True)

    g = events.groupby(["signal_time", "side"], sort=False)
    events["support_rank"] = g["support_pivot_pos"].rank(method="first", ascending=False).astype(int)
    events["unconsumed_rank_by_recency"] = events["support_rank"]
    events["unconsumed_rank_price_high_to_low"] = g["swing_level"].rank(method="first", ascending=False).astype(int)
    events["unconsumed_rank_price_low_to_high"] = g["swing_level"].rank(method="first", ascending=True).astype(int)
    events["unconsumed_same_bar_count"] = g["support_pivot_pos"].transform("count").astype(int)
    events["swept_order"] = events["support_rank"]
    events["support_source"] = "unconsumed_recency" + events["support_rank"].astype(str)
    events["support_generation_mode"] = "unconsumed"
    events["consume_breakout_pct"] = float(consume_pct)
    events["year"] = events["signal_time"].dt.year
    return events.sort_values(["signal_time", "support_rank"]).reset_index(drop=True)

def prepare_multi_swing_events_and_context(bars: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[features] building enriched low-sweep feature frame once", flush=True)
    features = build_enriched_features(bars, args)
    print("[support] extracting all causal confirmed low pivots", flush=True)
    pivots = extract_causal_low_pivots(features, args)
    print(f"[support] pivots={len(pivots):,}", flush=True)

    generation_mode = str(getattr(args, "support_generation_mode", "both")).strip().lower()
    frames: list[pd.DataFrame] = []
    if generation_mode in {"recent", "both"}:
        print("[events] building recent/topN support events", flush=True)
        recent = build_multi_swing_events(features, pivots, args)
        if not recent.empty:
            recent["support_generation_mode"] = "recent_topn"
            frames.append(recent)
    if generation_mode in {"unconsumed", "both"}:
        print("[events] building unconsumed support-pool events", flush=True)
        unconsumed = build_unconsumed_swing_events(features, pivots, args)
        if not unconsumed.empty:
            frames.append(unconsumed)
    events = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if events.empty:
        return events, features, pivots

    # Hard causal audit: every selected support level must be known no later
    # than the closed signal bar.
    bad = pd.to_datetime(events["support_available_time"], errors="coerce") > pd.to_datetime(events["signal_time"], errors="coerce")
    if bool(bad.any()):
        sample = events.loc[bad, ["signal_time", "support_available_time", "support_rank", "support_generation_mode"]].head(5).to_dict("records")
        raise RuntimeError(f"support availability violation: {sample}")

    events = attach_extra_features_to_events(events, features)
    events = add_filter_bins(events)

    sources = set(_split_csv_names(getattr(args, "context_sources", "trade_bar,footprint")))
    if "range_bar" in sources:
        events = attach_range_context(events, args)
    if "footprint" in sources:
        events = attach_footprint_context(events, args)
    if _split_csv_names(getattr(args, "micro_timeframes", "")):
        events = attach_micro_trade_context(events, args)
    return events.sort_values(["signal_time", "support_generation_mode", "support_rank"]).reset_index(drop=True), features, pivots


# ---------------------------------------------------------------------------
# Multi-swing support masks / simulation
# ---------------------------------------------------------------------------


def build_multi_support_mask(events: pd.DataFrame, mode: str) -> pd.Series:
    rank = pd.to_numeric(events.get("support_rank", np.nan), errors="coerce")
    swept_order = pd.to_numeric(events.get("swept_order", np.nan), errors="coerce")
    generation = events.get("support_generation_mode", pd.Series("recent_topn", index=events.index)).astype(str)
    is_recent = generation.eq("recent_topn")
    is_unconsumed = generation.eq("unconsumed")
    mode = str(mode).strip()

    # Old bounded baseline modes: recent N confirmed swing lows.
    if mode in {"rank1_latest", "single_swing", "latest"}:
        return is_recent & rank.eq(1)
    m = re.fullmatch(r"rank(\d+)_(?:prev|exact)", mode)
    if m:
        return is_recent & rank.eq(int(m.group(1)))
    m = re.fullmatch(r"first_swept_top(\d+)", mode)
    if m:
        n = int(m.group(1))
        return is_recent & rank.le(n) & swept_order.eq(1)
    m = re.fullmatch(r"older_only_top(\d+)", mode)
    if m:
        n = int(m.group(1))
        return is_recent & rank.ge(2) & rank.le(n) & swept_order.eq(1)
    m = re.fullmatch(r"all_swept_top(\d+)", mode)
    if m:
        n = int(m.group(1))
        return is_recent & rank.le(n)

    # Stateful support-pool modes: all lows are kept until first market sweep.
    if mode in {"latest_unconsumed_swept", "first_unconsumed_swept"}:
        return is_unconsumed & rank.eq(1)
    if mode == "nearest_unconsumed_swept":
        price_rank = pd.to_numeric(events.get("unconsumed_rank_price_high_to_low", np.nan), errors="coerce")
        return is_unconsumed & price_rank.eq(1)
    if mode == "deepest_unconsumed_swept":
        price_rank = pd.to_numeric(events.get("unconsumed_rank_price_low_to_high", np.nan), errors="coerce")
        return is_unconsumed & price_rank.eq(1)
    if mode == "all_unconsumed_swept":
        return is_unconsumed
    if mode in {"older_unconsumed_swept", "unconsumed_rank2plus"}:
        return is_unconsumed & rank.ge(2)
    m = re.fullmatch(r"unconsumed_top(\d+)_by_recency", mode)
    if m:
        return is_unconsumed & rank.le(int(m.group(1)))
    raise ValueError(f"Unknown multi support mode: {mode}")


def build_variants(args: argparse.Namespace) -> list[UpgradeVariant]:
    layers = _split_csv_names(args.candidate_layers)
    supports = _split_csv_names(args.multi_support_modes)
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


def _simulate_variant_with_cost(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    variant: UpgradeVariant,
    args: argparse.Namespace,
    *,
    market,
    cost_mult: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if events.empty:
        return pd.DataFrame(), {"skipped_overlap": 0, "skipped_invalid": 0, "input_events": 0}
    from research.low_sweep_a_upgrade_research import _event_positions, _horizon_from_exit_mode, _is_probe_compatible_variant  # noqa: E402

    max_horizon = _horizon_from_exit_mode(variant.exit_mode)
    ev, positions = _event_positions(bars, events, max_horizon=max_horizon)
    rows: list[dict[str, object]] = []
    skipped_overlap = 0
    skipped_invalid = 0
    last_exit_pos = -1
    for event, signal_pos in zip(ev.to_dict("records"), positions):
        if int(signal_pos) <= last_exit_pos:
            skipped_overlap += 1
            continue
        if _is_probe_compatible_variant(variant):
            rec = simulate_probe_compatible_trade(bars, event, int(signal_pos), variant, args, cost_mult=float(cost_mult))
        else:
            rec = simulate_upgrade_trade(bars, event, int(signal_pos), variant, args, market=market, cost_mult=float(cost_mult))
        if not rec.get("valid"):
            skipped_invalid += 1
            continue
        last_exit_pos = int(rec.get("exit_pos", signal_pos))
        rows.append(rec)
    trades = pd.DataFrame(rows)
    return trades, {"skipped_overlap": int(skipped_overlap), "skipped_invalid": int(skipped_invalid), "input_events": int(len(ev))}


def run_multi_swing_jobs(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    variants: list[UpgradeVariant],
    args: argparse.Namespace,
    *,
    cost_mult: float = 1.0,
    progress_label: str = "[multi] variants",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[cache] building reusable market arrays", flush=True)
    market = build_market_cache(bars, args)
    print("[cache] precomputing candidate/support masks", flush=True)
    layer_masks = {k: v.fillna(False).astype(bool) for k, v in build_candidate_layer_masks(events, args).items()}
    support_masks = {m: build_multi_support_mask(events, m).fillna(False).astype(bool) for m in sorted(set(v.support_mode for v in variants))}
    subset_cache: dict[tuple[str, str], pd.DataFrame] = {}

    summary_rows: list[dict[str, object]] = []
    trade_parts: list[pd.DataFrame] = []
    progress = ProgressReporter(label=progress_label, total=len(variants), every=1, enabled=not bool(getattr(args, "no_progress", False)))
    for i, variant in enumerate(variants, start=1):
        key = (variant.candidate_layer, variant.support_mode)
        part = subset_cache.get(key)
        if part is None:
            layer = layer_masks.get(variant.candidate_layer, pd.Series(False, index=events.index))
            support = support_masks.get(variant.support_mode, pd.Series(False, index=events.index))
            part = events.loc[layer & support].copy().sort_values(["signal_time", "support_rank"])
            subset_cache[key] = part
        trades, counters = _simulate_variant_with_cost(bars, part, variant, args, market=market, cost_mult=float(cost_mult))
        if not trades.empty:
            trade_parts.append(trades)
        rec = summarize_trades(trades, args, counters)
        rec.update(
            {
                "variant_name": variant.variant_name,
                "candidate_layer": variant.candidate_layer,
                "support_mode": variant.support_mode,
                "entry_mode": variant.entry_mode,
                "exit_mode": variant.exit_mode,
                "stop_name": variant.stop_spec.name,
                "stop_mode": variant.stop_spec.mode,
                "cost_mult": float(cost_mult),
            }
        )
        summary_rows.append(rec)
        progress.update(i)
    progress.close()

    all_trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        for col, default in [("profit_factor", np.nan), ("return_total", 0.0), ("trades", 0)]:
            if col not in summary.columns:
                summary[col] = default
        summary = summary.sort_values(["profit_factor", "return_total", "trades"], ascending=[False, False, False]).reset_index(drop=True)
    return all_trades, summary


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def build_event_support_mode_counts(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if events.empty:
        return pd.DataFrame()
    layer_masks = {k: v.fillna(False).astype(bool) for k, v in build_candidate_layer_masks(events, args).items()}
    for mode in _split_csv_names(args.multi_support_modes):
        sm = build_multi_support_mask(events, mode).fillna(False).astype(bool)
        for layer, lm in layer_masks.items():
            part = events.loc[sm & lm]
            rows.append(
                {
                    "support_mode": mode,
                    "candidate_layer": layer,
                    "events": int(len(part)),
                    "unique_signal_time": int(part["signal_time"].nunique()) if not part.empty else 0,
                    "rank1_events": int((pd.to_numeric(part.get("support_rank", np.nan), errors="coerce") == 1).sum()) if not part.empty else 0,
                    "older_support_events": int((pd.to_numeric(part.get("support_rank", np.nan), errors="coerce") >= 2).sum()) if not part.empty else 0,
                    "median_support_rank": float(pd.to_numeric(part.get("support_rank", np.nan), errors="coerce").median()) if not part.empty else np.nan,
                    "median_support_age": float(pd.to_numeric(part.get("support_age", np.nan), errors="coerce").median()) if not part.empty else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["candidate_layer", "support_mode"]).reset_index(drop=True)


def build_incremental_vs_baseline(trades: pd.DataFrame, summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if trades.empty or summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    base_map: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    for _, s in summary.iterrows():
        if str(s.get("support_mode")) != "rank1_latest":
            continue
        key = (str(s.get("candidate_layer")), str(s.get("entry_mode")), str(s.get("exit_mode")), str(s.get("stop_name")))
        base_map[key] = trades.loc[trades["variant_name"].eq(str(s.get("variant_name")))].copy()

    for _, s in summary.iterrows():
        mode = str(s.get("support_mode"))
        if mode == "rank1_latest":
            continue
        key = (str(s.get("candidate_layer")), str(s.get("entry_mode")), str(s.get("exit_mode")), str(s.get("stop_name")))
        base = base_map.get(key, pd.DataFrame())
        cur = trades.loc[trades["variant_name"].eq(str(s.get("variant_name")))].copy()
        if cur.empty:
            continue
        base_times = set(pd.to_datetime(base.get("signal_time", pd.Series(dtype="datetime64[ns]"))).astype(str)) if not base.empty else set()
        cur_times = pd.to_datetime(cur["signal_time"], errors="coerce").astype(str)
        unique = cur.loc[~cur_times.isin(base_times)].copy()
        overlap = cur.loc[cur_times.isin(base_times)].copy()
        unique_ret = pd.to_numeric(unique.get("net_return_on_equity"), errors="coerce").dropna() if not unique.empty else pd.Series(dtype=float)
        overlap_ret = pd.to_numeric(overlap.get("net_return_on_equity"), errors="coerce").dropna() if not overlap.empty else pd.Series(dtype=float)
        base_summary = summary.loc[summary["variant_name"].isin([base["variant_name"].iloc[0] if not base.empty and "variant_name" in base else ""])]
        br = base_summary.iloc[0] if not base_summary.empty else pd.Series(dtype=object)
        rows.append(
            {
                "baseline_variant": br.get("variant_name", ""),
                "challenger_variant": s.get("variant_name", ""),
                "candidate_layer": s.get("candidate_layer", ""),
                "support_mode": mode,
                "entry_mode": s.get("entry_mode", ""),
                "exit_mode": s.get("exit_mode", ""),
                "stop_name": s.get("stop_name", ""),
                "baseline_trades": int(br.get("trades", 0) or 0),
                "challenger_trades": int(s.get("trades", 0) or 0),
                "trade_increase": int(s.get("trades", 0) or 0) - int(br.get("trades", 0) or 0),
                "trade_increase_pct": (float(s.get("trades", 0) or 0) / max(1.0, float(br.get("trades", 0) or 0)) - 1.0) if len(br) else np.nan,
                "unique_trades": int(len(unique)),
                "overlap_trades": int(len(overlap)),
                "unique_return_sum": float(unique_ret.sum()) if not unique_ret.empty else 0.0,
                "unique_pf": _profit_factor(unique_ret),
                "unique_win_rate": float((unique_ret > 0).mean()) if not unique_ret.empty else np.nan,
                "overlap_return_sum": float(overlap_ret.sum()) if not overlap_ret.empty else 0.0,
                "overlap_pf": _profit_factor(overlap_ret),
                "challenger_return_total": s.get("return_total", np.nan),
                "baseline_return_total": br.get("return_total", np.nan) if len(br) else np.nan,
                "challenger_pf": s.get("profit_factor", np.nan),
                "baseline_pf": br.get("profit_factor", np.nan) if len(br) else np.nan,
                "challenger_dd": s.get("max_drawdown", np.nan),
                "baseline_dd": br.get("max_drawdown", np.nan) if len(br) else np.nan,
                "not_worse_basic": bool(
                    _safe_float(s.get("trades", 0), 0) >= _safe_float(br.get("trades", 0), 0)
                    and _safe_float(s.get("return_total", np.nan)) >= _safe_float(br.get("return_total", np.nan))
                    and _safe_float(s.get("profit_factor", np.nan)) >= _safe_float(br.get("profit_factor", np.nan)) * 0.98
                    and abs(_safe_float(s.get("max_drawdown", np.nan))) <= abs(_safe_float(br.get("max_drawdown", np.nan))) * 1.10
                ) if len(br) else False,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["not_worse_basic", "trade_increase", "unique_return_sum"], ascending=[False, False, False]).reset_index(drop=True)


def build_stress_variants(summary: pd.DataFrame, args: argparse.Namespace) -> list[UpgradeVariant]:
    if summary.empty:
        return []
    stops = {s.name: s for s in parse_stop_specs(args.upgrade_stop_specs)}
    base = summary.loc[summary.get("support_mode", "").astype(str).eq("rank1_latest")].copy()
    top = summary.loc[~summary.get("support_mode", "").astype(str).eq("rank1_latest")].copy()
    if not top.empty:
        top = top.sort_values(["return_total", "profit_factor", "trades"], ascending=[False, False, False]).head(max(0, int(args.stress_top_n)))
    selected = pd.concat([base, top], ignore_index=True) if not base.empty else top
    variants: list[UpgradeVariant] = []
    seen: set[str] = set()
    for _, row in selected.iterrows():
        stop_name = str(row.get("stop_name", "no_stop"))
        if stop_name not in stops:
            continue
        v = UpgradeVariant(
            variant_name=str(row["variant_name"]),
            candidate_layer=str(row["candidate_layer"]),
            support_mode=str(row["support_mode"]),
            entry_mode=str(row["entry_mode"]),
            exit_mode=str(row["exit_mode"]),
            stop_spec=stops[stop_name],
        )
        if v.variant_name not in seen:
            variants.append(v)
            seen.add(v.variant_name)
    return variants


def run_cost_delay_stress(bars: pd.DataFrame, events: pd.DataFrame, base_summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    variants = build_stress_variants(base_summary, args)
    if not variants:
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    for mult in _parse_float_list_or_inherit(getattr(args, "stress_cost_mults", ""), "", name="stress_cost_mults"):
        _, s = run_multi_swing_jobs(bars, events, variants, args, cost_mult=float(mult), progress_label=f"[stress] cost {mult:g}x")
        if not s.empty:
            s["stress_type"] = "cost"
            s["stress_value"] = float(mult)
            parts.append(s)
    delay_bars = _parse_int_list_or_inherit(getattr(args, "stress_delay_bars", ""), "", name="stress_delay_bars")
    if delay_bars:
        delayed: list[UpgradeVariant] = []
        for v in variants:
            if v.entry_mode != "next_open":
                continue
            for d in delay_bars:
                delayed.append(replace(v, variant_name=f"{v.variant_name}__delay{int(d)}", entry_mode=f"next_open_delay{int(d)}"))
        if delayed:
            _, s = run_multi_swing_jobs(bars, events, delayed, args, cost_mult=1.0, progress_label="[stress] delay")
            if not s.empty:
                s["stress_type"] = "delay"
                s["stress_value"] = s["entry_mode"].map(lambda x: int(str(x).replace("next_open_delay", "")) if str(x).startswith("next_open_delay") else 0)
                parts.append(s)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_edge_registry(summary: pd.DataFrame, incremental: pd.DataFrame, stress: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    out = summary.copy()
    if not incremental.empty:
        inc_cols = [
            "challenger_variant",
            "trade_increase",
            "trade_increase_pct",
            "unique_trades",
            "unique_return_sum",
            "unique_pf",
            "unique_win_rate",
            "not_worse_basic",
        ]
        out = out.merge(incremental[[c for c in inc_cols if c in incremental.columns]], left_on="variant_name", right_on="challenger_variant", how="left")
    out["base_edge_candidate"] = (
        (pd.to_numeric(out.get("trades", 0), errors="coerce") >= int(args.min_trades_for_upgrade_edge))
        & (pd.to_numeric(out.get("return_total", np.nan), errors="coerce") > 0)
        & (pd.to_numeric(out.get("median_return", np.nan), errors="coerce") > 0)
        & (pd.to_numeric(out.get("profit_factor", np.nan), errors="coerce") >= float(args.min_pf_for_upgrade_edge))
    )
    if not stress.empty:
        st = stress.groupby("variant_name").agg(
            stress_cases=("stress_type", "count"),
            stress_min_return=("return_total", "min"),
            stress_min_pf=("profit_factor", "min"),
            stress_min_trades=("trades", "min"),
        ).reset_index()
        out = out.merge(st, on="variant_name", how="left")
    out["promotion_candidate"] = (
        out["base_edge_candidate"].fillna(False)
        & out.get("not_worse_basic", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        & (pd.to_numeric(out.get("stress_min_return", np.nan), errors="coerce").fillna(pd.to_numeric(out.get("return_total", np.nan), errors="coerce")) > 0)
        & (pd.to_numeric(out.get("stress_min_pf", np.nan), errors="coerce").fillna(pd.to_numeric(out.get("profit_factor", np.nan), errors="coerce")) >= 1.0)
    )
    return out.sort_values(["promotion_candidate", "base_edge_candidate", "return_total", "profit_factor"], ascending=[False, False, False, False]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {SCRIPT_NAME}", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)
    print(f"[load] trade bars {args.symbol} {args.timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    bars = load_trade_bars(args)
    events, features, pivots = prepare_multi_swing_events_and_context(bars, args)
    print(f"[events] multi_swing_events={len(events):,}", flush=True)
    if events.empty:
        write_csv(pd.DataFrame(), out_dir / "03_variant_summary.csv", "summary")
        return 0

    variants = build_variants(args)
    print(f"[variants] total={len(variants):,}", flush=True)
    trades, summary = run_multi_swing_jobs(bars, events, variants, args, cost_mult=1.0)

    yearly = summarize_by_period(trades, "year")
    monthly = summarize_by_period(trades, "month")
    support_counts = build_event_support_mode_counts(events, args)
    support_compare = summarize_compare(summary, ["support_mode"])
    layer_compare = summarize_compare(summary, ["candidate_layer"])
    incremental = build_incremental_vs_baseline(trades, summary, args)
    stress = run_cost_delay_stress(bars, events, summary, args)
    registry = build_edge_registry(summary, incremental, stress, args)

    event_cols = [
        "signal_time",
        "event_name",
        "variant",
        "support_generation_mode",
        "support_rank",
        "swept_order",
        "unconsumed_rank_by_recency",
        "unconsumed_rank_price_high_to_low",
        "unconsumed_rank_price_low_to_high",
        "unconsumed_same_bar_count",
        "support_pivot_time",
        "support_available_time",
        "swing_level",
        "swing_age",
        "down_spike_pct",
        "close_pos_in_bar",
        "large_trade_share",
        "fp_max_bucket_abs_delta_pressure",
        "session_bucket",
    ]
    event_sample = events[[c for c in event_cols if c in events.columns]].head(int(args.save_events)).copy()
    trades_sample = trades.head(int(args.save_trades)).copy() if not trades.empty and int(args.save_trades) > 0 else pd.DataFrame()
    pivots_sample = pivots.head(5000).copy()

    write_csv(event_sample, out_dir / "01_multi_swing_events_sample.csv", "events_sample")
    if bool(getattr(args, "save_full_multi_events", False)):
        write_csv(events, out_dir / "01b_multi_swing_events_full.csv", "events_full")
    write_csv(pivots_sample, out_dir / "02_causal_low_pivots_sample.csv", "pivots_sample")
    write_csv(summary, out_dir / "03_variant_summary.csv", "summary")
    write_csv(trades_sample, out_dir / "04_trades_sample.csv", "trades_sample")
    write_csv(yearly, out_dir / "05_yearly.csv", "yearly")
    write_csv(monthly, out_dir / "06_monthly.csv", "monthly")
    write_csv(support_counts, out_dir / "07_event_support_mode_counts.csv", "support_counts")
    write_csv(support_compare, out_dir / "08_support_mode_compare.csv", "support_compare")
    write_csv(layer_compare, out_dir / "09_candidate_layer_compare.csv", "layer_compare")
    write_csv(incremental, out_dir / "10_incremental_vs_rank1_baseline.csv", "incremental")
    write_csv(stress, out_dir / "11_cost_delay_stress.csv", "stress")
    write_csv(registry, out_dir / "12_multi_swing_edge_registry.csv", "edge_registry")

    meta = {
        "script": SCRIPT_NAME,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "events": int(len(events)),
        "pivots": int(len(pivots)),
        "variants": int(len(variants)),
        "multi_support_modes": _split_csv_names(args.multi_support_modes),
        "max_support_rank": int(args.max_support_rank),
        "support_generation_mode": str(getattr(args, "support_generation_mode", "both")),
        "consume_breakout_pct": str(getattr(args, "consume_breakout_pct", "")),
        "causal_guards": [
            "low pivot available_pos = pivot_pos + pivot_right + 1, matching existing confirmed_swing_lows availability",
            "support_available_time is written for audit and must be <= signal_time",
            "signals are generated from closed 1m trade bars only",
            "entry uses existing next-open/reclaim/limit simulation paths; default next_open means signal bar close then next bar open",
            "footprint/range context uses existing causal attach functions",
        ],
        "performance_guards": [
            "enriched features are built once",
            "all low pivots are extracted once",
            "recent/topN events are generated only from broad down-spike bars and scan at most max_support_rank pivots",
            "unconsumed support first-touch uses a segment tree: O(pivots * log(bars)), not O(pivots * bars)",
            "candidate/support masks are precomputed once per run",
            "market arrays are cached once per simulation batch",
            "stress is limited to baseline plus top --stress-top-n variants",
        ],
        "promotion_rule_summary": "12 registry marks promotion_candidate only when base metrics pass, multi-swing is not worse than rank1 baseline, and selected stress remains positive/PF>=1.",
    }
    print(f"[write] meta -> {out_dir / '13_lab_meta.json'}", flush=True)
    (out_dir / "13_lab_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("[done] multi-swing support research complete", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
