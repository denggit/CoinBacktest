#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Directional Impulse Continuation: CVD-conditioned path anatomy (round 08).

Research question
-----------------
Round 07 showed that a price impulse is followed by high two-sided excursion and
contains several materially different path types.  Round 08 does not propose a
new strategy rule.  It asks whether causal order-flow information already
available in local OKX trade bars can explain those path types:

1. At the closed impulse signal, does direction-adjusted CVD/delta support differ
   between immediate continuation, pullback continuation, failure, two-sided
   expansion, and muted paths?
2. During the first 1m/3m/5m/15m after the signal, do CVD pressure, acceleration,
   large-trade pressure, and price-flow alignment separate those paths?
3. Does the evidence support a multi-timescale architecture in which a broader
   price impulse is the context and smaller order-flow windows manage execution?
4. Are LONG and SHORT path mechanisms different?

This is descriptive path decomposition.  It does not select a CVD threshold,
filter entries, simulate exits, optimize TP/SL, or declare a tradable edge.

Causality
---------
- Signal bar t is fully closed before the event exists.
- Reference entry is bar t+1 open.
- Signal-stage flow windows end at t and are available at signal_time.
- A post-signal checkpoint k uses bars t+1..t+k and is available only after
  bar t+k closes.  If later used operationally, the earliest execution is the
  next bar open; this study does not pretend otherwise.
- Future path labels are ex-post research outcomes only and never input features.
- Any pre-signal, signal, entry, checkpoint, or 60m path touching a synthetic
  gap bar is excluded.

Performance design
------------------
- Local OKX trade bars only; build_missing=False.
- One price-feature build per impulse window.
- Prefix sums provide O(1), vectorized order-flow windows.
- Future paths are built in bounded float32 memmaps using the validated Round-07
  kernel; no per-event Python path loop.
- Nested thresholds and raw/deduplicated sets reuse the same event/path arrays.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402


def _load_round07_module():
    path = Path(__file__).resolve().with_name("07_impulse_path_anatomy_study.py")
    spec = importlib.util.spec_from_file_location("directional_impulse_round07_for_r08", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared round-07 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r07 = _load_round07_module()
r04 = r07.r04
r02 = r07.r02
r01 = r07.r01

SCRIPT_NAME = "08_cvd_path_regime_anatomy_study"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION_R08"
EDGE_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION"
TITLE = "ETH Directional Impulse Continuation - CVD Path-Regime Anatomy"
DEFAULT_OUT_DIR = (
    "data/reports/research/momentum/directional_impulse_continuation/"
    "08_cvd_path_regime_anatomy_study"
)

DEFAULT_WINDOWS = (1, 3, 5, 10, 15)
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
DEFAULT_FLOW_WINDOWS = (1, 3, 5, 15)
DEFAULT_CHECKPOINTS = (1, 3, 5, 15)
DEFAULT_MAX_PATH_MINUTES = 60
PATH_LABELS = (
    "immediate_runner",
    "pullback_runner",
    "one_sided_continuation",
    "directional_failure",
    "two_sided_expansion",
    "muted",
    "other",
)
PAIRWISE_CONTRASTS = (
    ("immediate_runner", "early_failure"),
    ("pullback_runner", "directional_failure"),
    ("one_sided_continuation", "two_sided_expansion"),
)
PROFILE_OFFSETS = tuple(range(-15, 16))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Descriptive CVD/order-flow anatomy of ETH impulse path regimes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--trade-bar-db-name", default="okx_trade_bars.db")
    p.add_argument("--impulse-windows", default=",".join(map(str, DEFAULT_WINDOWS)))
    p.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)))
    p.add_argument("--flow-windows", default=",".join(map(str, DEFAULT_FLOW_WINDOWS)))
    p.add_argument("--checkpoints", default=",".join(map(str, DEFAULT_CHECKPOINTS)))
    p.add_argument("--max-path-minutes", type=int, default=DEFAULT_MAX_PATH_MINUTES)
    p.add_argument("--vol-lookback-bars", type=int, default=1440)
    p.add_argument("--vol-min-periods", type=int, default=720)
    p.add_argument("--path-chunk-size", type=int, default=5000)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument(
        "--skip-events-csv",
        action="store_true",
        help="Development-only. Production stores minimum-threshold deduplicated event descriptors.",
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _parse_positive_ints(raw: str, *, name: str) -> tuple[int, ...]:
    vals = tuple(sorted(dict.fromkeys(int(x.strip()) for x in str(raw).split(",") if x.strip())))
    if not vals or any(v <= 0 for v in vals):
        raise ValueError(f"{name} must contain positive integers")
    return vals


def _safe_div(num: np.ndarray | float, den: np.ndarray | float) -> np.ndarray:
    a = np.asarray(num, dtype=float)
    b = np.asarray(den, dtype=float)
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan, dtype=float), where=np.isfinite(b) & (np.abs(b) > 1e-12))


def _prefix(values: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    out = np.zeros(len(x) + 1, dtype=np.float64)
    np.cumsum(x, out=out[1:])
    return out


def _window_sum(prefix: np.ndarray, end_pos: np.ndarray, window: int) -> np.ndarray:
    end = np.asarray(end_pos, dtype=np.int64) + 1
    start = end - int(window)
    out = np.full(len(end), np.nan, dtype=float)
    valid = start >= 0
    out[valid] = prefix[end[valid]] - prefix[start[valid]]
    return out


def _range_sum(prefix: np.ndarray, start_pos: np.ndarray, end_pos: np.ndarray) -> np.ndarray:
    start = np.asarray(start_pos, dtype=np.int64)
    end = np.asarray(end_pos, dtype=np.int64) + 1
    out = np.full(len(start), np.nan, dtype=float)
    valid = (start >= 0) & (end >= start) & (end < len(prefix) + 1)
    out[valid] = prefix[end[valid]] - prefix[start[valid]]
    return out


def _flow_arrays(bars: pd.DataFrame) -> dict[str, np.ndarray]:
    def col(name: str) -> np.ndarray:
        if name not in bars.columns:
            return np.zeros(len(bars), dtype=np.float64)
        return pd.to_numeric(bars[name], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    buy = col("buy_notional")
    sell = col("sell_notional")
    notional = col("notional")
    if not np.any(notional > 0):
        notional = buy + sell
    delta = col("delta_notional")
    recomputed_delta = buy - sell
    if not np.any(np.abs(delta) > 0) and np.any(np.abs(recomputed_delta) > 0):
        delta = recomputed_delta

    large_buy = col("large_buy_notional")
    large_sell = col("large_sell_notional")
    large_delta = col("large_delta_notional")
    recomputed_large_delta = large_buy - large_sell
    if not np.any(np.abs(large_delta) > 0) and np.any(np.abs(recomputed_large_delta) > 0):
        large_delta = recomputed_large_delta

    arrays = {
        "buy_notional": buy,
        "sell_notional": sell,
        "notional": notional,
        "delta_notional": delta,
        "large_buy_notional": large_buy,
        "large_sell_notional": large_sell,
        "large_delta_notional": large_delta,
        "trades_count": col("trades_count"),
        "open": pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=np.float64),
        "close": pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=np.float64),
        "observed": bars["source_bar_observed_flag"].astype(bool).to_numpy(),
    }
    arrays["large_notional"] = large_buy + large_sell
    arrays["observed_prefix"] = _prefix(arrays["observed"].astype(float))
    for key in (
        "buy_notional", "sell_notional", "notional", "delta_notional",
        "large_notional", "large_delta_notional", "trades_count",
    ):
        arrays[f"{key}_prefix"] = _prefix(arrays[key])
    return arrays


def _flow_data_audit(arr: dict[str, np.ndarray]) -> dict[str, Any]:
    notional = arr["notional"]
    delta = arr["delta_notional"]
    buy = arr["buy_notional"]
    sell = arr["sell_notional"]
    large = arr["large_notional"]
    valid = notional > 0
    delta_identity_err = np.abs(delta - (buy - sell))
    return {
        "rows": int(len(notional)),
        "positive_notional_rows": int(valid.sum()),
        "positive_notional_rate": float(valid.mean()) if len(valid) else float("nan"),
        "nonzero_delta_rows": int((np.abs(delta) > 0).sum()),
        "nonzero_large_trade_rows": int((large > 0).sum()),
        "large_trade_feature_available": bool(np.any(large > 0)),
        "max_delta_identity_abs_error": float(np.nanmax(delta_identity_err)) if len(delta_identity_err) else float("nan"),
        "mean_abs_delta_pressure": float(np.nanmean(np.abs(_safe_div(delta[valid], notional[valid])))) if valid.any() else float("nan"),
        "absolute_cvd_level_used": False,
        "cvd_definition": "window CVD change = sum(delta_notional); absolute cumulative CVD level is not used",
    }


def _observed_window_mask(arr: dict[str, np.ndarray], positions: np.ndarray, pre_bars: int, post_bars: int) -> np.ndarray:
    p = np.asarray(positions, dtype=np.int64)
    start = p - int(pre_bars) + 1
    end = p + int(post_bars)
    obs = _range_sum(arr["observed_prefix"], start, end)
    required = int(pre_bars + post_bars)
    return (start >= 0) & np.isfinite(obs) & (obs == float(required))


def _signal_flow_features(
    arr: dict[str, np.ndarray], positions: np.ndarray, side: int, flow_windows: tuple[int, ...]
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    p = np.asarray(positions, dtype=np.int64)
    for w in flow_windows:
        notional = _window_sum(arr["notional_prefix"], p, w)
        delta = _window_sum(arr["delta_notional_prefix"], p, w)
        large_total = _window_sum(arr["large_notional_prefix"], p, w)
        large_delta = _window_sum(arr["large_delta_notional_prefix"], p, w)
        trades = _window_sum(arr["trades_count_prefix"], p, w)
        start_close_pos = p - int(w)
        price_ret = np.full(len(p), np.nan, dtype=float)
        valid = start_close_pos >= 0
        price_ret[valid] = float(side) * (arr["close"][p[valid]] / arr["close"][start_close_pos[valid]] - 1.0)
        dir_pressure = float(side) * _safe_div(delta, notional)
        dir_large_pressure = float(side) * _safe_div(large_delta, large_total)
        out[f"signal_w{w}_dir_delta_pressure"] = dir_pressure
        out[f"signal_w{w}_dir_large_delta_pressure"] = dir_large_pressure
        out[f"signal_w{w}_large_notional_share"] = _safe_div(large_total, notional)
        out[f"signal_w{w}_notional_per_min"] = notional / float(w)
        out[f"signal_w{w}_trades_per_min"] = trades / float(w)
        out[f"signal_w{w}_dir_price_return"] = price_ret
        out[f"signal_w{w}_flow_price_alignment"] = ((dir_pressure > 0) & (price_ret > 0)).astype(float)
        out[f"signal_w{w}_absorption_proxy"] = ((dir_pressure > 0) & (price_ret <= 0)).astype(float)
    return out


def _checkpoint_flow_features(
    arr: dict[str, np.ndarray], positions: np.ndarray, side: int, checkpoints: tuple[int, ...]
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    p = np.asarray(positions, dtype=np.int64)
    entry_price = arr["open"][p + 1]
    for k in checkpoints:
        post_start = p + 1
        post_end = p + int(k)
        pre_start = p - int(k) + 1
        pre_end = p
        post_notional = _range_sum(arr["notional_prefix"], post_start, post_end)
        post_delta = _range_sum(arr["delta_notional_prefix"], post_start, post_end)
        post_large_total = _range_sum(arr["large_notional_prefix"], post_start, post_end)
        post_large_delta = _range_sum(arr["large_delta_notional_prefix"], post_start, post_end)
        post_trades = _range_sum(arr["trades_count_prefix"], post_start, post_end)
        pre_notional = _range_sum(arr["notional_prefix"], pre_start, pre_end)
        pre_delta = _range_sum(arr["delta_notional_prefix"], pre_start, pre_end)
        pre_trades = _range_sum(arr["trades_count_prefix"], pre_start, pre_end)
        post_pressure = float(side) * _safe_div(post_delta, post_notional)
        pre_pressure = float(side) * _safe_div(pre_delta, pre_notional)
        price_ret = float(side) * (arr["close"][post_end] / entry_price - 1.0)
        large_pressure = float(side) * _safe_div(post_large_delta, post_large_total)
        out[f"post{k}_dir_delta_pressure"] = post_pressure
        out[f"post{k}_dir_large_delta_pressure"] = large_pressure
        out[f"post{k}_large_notional_share"] = _safe_div(post_large_total, post_notional)
        out[f"post{k}_dir_price_return"] = price_ret
        out[f"post{k}_delta_pressure_accel_vs_pre"] = post_pressure - pre_pressure
        out[f"post{k}_notional_speed_ratio_vs_pre"] = _safe_div(post_notional, pre_notional)
        out[f"post{k}_trade_speed_ratio_vs_pre"] = _safe_div(post_trades, pre_trades)
        out[f"post{k}_flow_price_alignment"] = ((post_pressure > 0) & (price_ret > 0)).astype(float)
        out[f"post{k}_absorption_proxy"] = ((post_pressure > 0) & (price_ret <= 0)).astype(float)
        out[f"post{k}_vacuum_proxy"] = ((post_pressure <= 0) & (price_ret > 0)).astype(float)
    return out


def _path_flags(descriptors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    f15 = descriptors["close_favorable_first_15bps_min"].astype(int)
    a15 = descriptors["close_adverse_first_15bps_min"].astype(int)
    f25 = descriptors["close_favorable_first_25bps_min"].astype(int)
    a25 = descriptors["close_adverse_first_25bps_min"].astype(int)
    f50 = descriptors["close_favorable_first_50bps_min"].astype(int)

    favorable15_first = (f15 > 0) & ((a15 == 0) | (f15 < a15))
    adverse15_first = (a15 > 0) & ((f15 == 0) | (a15 < f15))
    favorable25_first = (f25 > 0) & ((a25 == 0) | (f25 < a25))
    adverse25_first = (a25 > 0) & ((f25 == 0) | (a25 < f25))

    immediate_runner = favorable15_first & (f50 > 0) & (f50 <= 5)
    pullback_runner = adverse15_first & (f50 > 0)
    early_failure = adverse25_first & (a25 <= 5)
    directional_failure = (a25 > 0) & (f25 == 0)
    one_sided_continuation = (f25 > 0) & (a25 == 0)
    two_sided_expansion = (f25 > 0) & (a25 > 0)
    muted = (f25 == 0) & (a25 == 0)

    label = np.full(len(f15), "other", dtype=object)
    label[muted] = "muted"
    label[two_sided_expansion] = "two_sided_expansion"
    label[directional_failure] = "directional_failure"
    label[one_sided_continuation] = "one_sided_continuation"
    label[pullback_runner] = "pullback_runner"
    label[immediate_runner] = "immediate_runner"
    return {
        "immediate_runner": immediate_runner,
        "pullback_runner": pullback_runner,
        "early_failure": early_failure,
        "directional_failure": directional_failure,
        "one_sided_continuation": one_sided_continuation,
        "two_sided_expansion": two_sided_expansion,
        "muted": muted,
        "primary_path_label": label,
        "favorable15_first": favorable15_first,
        "adverse15_first": adverse15_first,
        "favorable25_first": favorable25_first,
        "adverse25_first": adverse25_first,
    }


def _summary_stats(values: np.ndarray) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"mean": np.nan, "median": np.nan, "p25": np.nan, "p75": np.nan, "std": np.nan}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
    }


def _auc(values: np.ndarray, positive: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    y = np.asarray(positive, dtype=bool)
    valid = np.isfinite(x)
    x = x[valid]
    y = y[valid]
    n1 = int(y.sum())
    n0 = int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _path_distribution_rows(
    *, direction: str, window: int, threshold: float, event_set: str,
    selected: np.ndarray, flags: dict[str, np.ndarray], months: int,
) -> list[dict[str, Any]]:
    idx = np.flatnonzero(selected)
    total = len(idx)
    rows: list[dict[str, Any]] = []
    for label in PATH_LABELS:
        count = int(np.sum(flags["primary_path_label"][idx] == label))
        rows.append({
            "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
            "event_set": event_set, "path_label": label, "events": count,
            "path_rate": float(count / total) if total else np.nan,
            "events_per_month": float(count / max(1, months)),
        })
    for name in ("immediate_runner", "pullback_runner", "early_failure", "directional_failure", "one_sided_continuation", "two_sided_expansion", "muted"):
        count = int(np.sum(flags[name][idx]))
        rows.append({
            "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
            "event_set": event_set, "path_label": f"flag::{name}", "events": count,
            "path_rate": float(count / total) if total else np.nan,
            "events_per_month": float(count / max(1, months)),
        })
    return rows


def _feature_by_path_rows(
    *, direction: str, window: int, threshold: float, event_set: str,
    selected: np.ndarray, flags: dict[str, np.ndarray], features: dict[str, np.ndarray], stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base_idx = np.flatnonzero(selected)
    labels = flags["primary_path_label"]
    for label in PATH_LABELS:
        idx = base_idx[labels[base_idx] == label]
        for name, values in features.items():
            stat = _summary_stats(values[idx])
            rows.append({
                "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
                "event_set": event_set, "path_label": label, "feature_stage": stage,
                "feature_name": name, "events": int(len(idx)), **stat,
            })
    return rows


def _separation_rows(
    *, direction: str, window: int, threshold: float, event_set: str,
    selected: np.ndarray, flags: dict[str, np.ndarray], features: dict[str, np.ndarray], stage: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pos_name, neg_name in PAIRWISE_CONTRASTS:
        pos = selected & flags[pos_name]
        neg = selected & flags[neg_name]
        combined = pos | neg
        labels = pos[combined]
        for feature_name, values in features.items():
            x = values[combined]
            pos_x = values[pos]
            neg_x = values[neg]
            pos_stat = _summary_stats(pos_x)
            neg_stat = _summary_stats(neg_x)
            variances = np.asarray([pos_stat["std"] ** 2, neg_stat["std"] ** 2], dtype=float)
            pooled = np.sqrt(np.nanmean(variances)) if np.isfinite(variances).any() else np.nan
            smd = (pos_stat["mean"] - neg_stat["mean"]) / pooled if math.isfinite(pooled) and pooled > 0 else np.nan
            auc = _auc(x, labels)
            rows.append({
                "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
                "event_set": event_set, "feature_stage": stage, "positive_cohort": pos_name,
                "negative_cohort": neg_name, "feature_name": feature_name,
                "positive_events": int(pos.sum()), "negative_events": int(neg.sum()),
                "positive_mean": pos_stat["mean"], "negative_mean": neg_stat["mean"],
                "mean_difference": pos_stat["mean"] - neg_stat["mean"],
                "median_difference": pos_stat["median"] - neg_stat["median"],
                "standardized_mean_difference": smd, "auc_positive_high": auc,
                "auc_separation": abs(auc - 0.5) * 2.0 if math.isfinite(auc) else np.nan,
            })
    return rows


def _quintile_rows(
    *, direction: str, window: int, threshold: float, selected: np.ndarray,
    flags: dict[str, np.ndarray], features: dict[str, np.ndarray], years: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx = np.flatnonzero(selected)
    if len(idx) < 50:
        return rows
    target_names = ("immediate_runner", "pullback_runner", "early_failure", "directional_failure", "two_sided_expansion")
    for feature_name, values in features.items():
        x = values[idx]
        finite = np.isfinite(x)
        if finite.sum() < 50 or np.unique(x[finite]).size < 5:
            continue
        try:
            buckets = pd.qcut(pd.Series(x[finite]), q=5, labels=False, duplicates="drop").to_numpy()
        except ValueError:
            continue
        finite_idx = idx[finite]
        for bucket in np.unique(buckets):
            bidx = finite_idx[buckets == bucket]
            rec: dict[str, Any] = {
                "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
                "event_set": "deduplicated", "feature_name": feature_name, "quintile": int(bucket) + 1,
                "events": int(len(bidx)), "feature_min": float(np.nanmin(values[bidx])),
                "feature_max": float(np.nanmax(values[bidx])), "feature_median": float(np.nanmedian(values[bidx])),
            }
            for target in target_names:
                rec[f"{target}_rate"] = float(np.mean(flags[target][bidx])) if len(bidx) else np.nan
            year_rates = []
            for year in np.unique(years[bidx]):
                yi = bidx[years[bidx] == year]
                if len(yi):
                    year_rates.append(float(np.mean(flags["immediate_runner"][yi])))
            rec["immediate_runner_year_rate_min"] = min(year_rates) if year_rates else np.nan
            rec["immediate_runner_year_rate_max"] = max(year_rates) if year_rates else np.nan
            rows.append(rec)
    return rows


def _minute_profile_rows(
    *, bars: pd.DataFrame, arr: dict[str, np.ndarray], positions: np.ndarray, side: int,
    direction: str, window: int, threshold: float, selected: np.ndarray, flags: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx_base = np.flatnonzero(selected)
    labels = flags["primary_path_label"]
    p = positions
    baseline_notional = _window_sum(arr["notional_prefix"], p, 15) / 15.0
    bar_pressure = float(side) * _safe_div(arr["delta_notional"], arr["notional"])
    bar_price = float(side) * (arr["close"] / arr["open"] - 1.0)
    for label in PATH_LABELS:
        idx = idx_base[labels[idx_base] == label]
        if not len(idx):
            continue
        pos = p[idx]
        base = baseline_notional[idx]
        for offset in PROFILE_OFFSETS:
            loc = pos + int(offset)
            pressure = bar_pressure[loc]
            price = bar_price[loc]
            speed = _safe_div(arr["notional"][loc], base)
            rows.append({
                "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
                "event_set": "deduplicated", "path_label": label, "minute_offset_vs_signal_bar": int(offset),
                "events": int(len(idx)), "mean_dir_delta_pressure": float(np.nanmean(pressure)),
                "median_dir_delta_pressure": float(np.nanmedian(pressure)),
                "positive_dir_delta_pressure_rate": float(np.nanmean(pressure > 0)),
                "mean_dir_price_return_bps": float(np.nanmean(price) * 10_000.0),
                "median_dir_price_return_bps": float(np.nanmedian(price) * 10_000.0),
                "mean_notional_speed_ratio_vs_pre15": float(np.nanmean(speed)),
                "median_notional_speed_ratio_vs_pre15": float(np.nanmedian(speed)),
            })
    return rows


def _yearly_rows(
    *, direction: str, window: int, threshold: float, selected: np.ndarray,
    flags: dict[str, np.ndarray], signal_features: dict[str, np.ndarray], post_features: dict[str, np.ndarray], years: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    idx_all = np.flatnonzero(selected)
    key_features = [
        f"signal_w{min(DEFAULT_FLOW_WINDOWS)}_dir_delta_pressure",
        "signal_w5_dir_delta_pressure" if "signal_w5_dir_delta_pressure" in signal_features else next(iter(signal_features)),
        "post1_dir_delta_pressure" if "post1_dir_delta_pressure" in post_features else next(iter(post_features)),
        "post3_dir_delta_pressure" if "post3_dir_delta_pressure" in post_features else next(iter(post_features)),
    ]
    merged = {**signal_features, **post_features}
    for year in sorted(np.unique(years[idx_all])):
        idx = idx_all[years[idx_all] == year]
        rec: dict[str, Any] = {
            "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
            "event_set": "deduplicated", "year": int(year), "events": int(len(idx)),
        }
        for name in ("immediate_runner", "pullback_runner", "early_failure", "directional_failure", "two_sided_expansion", "muted"):
            rec[f"{name}_rate"] = float(np.mean(flags[name][idx])) if len(idx) else np.nan
        for name in dict.fromkeys(key_features):
            if name in merged:
                rec[f"mean_{name}"] = float(np.nanmean(merged[name][idx])) if len(idx) else np.nan
        rows.append(rec)
    return rows


def _compact_event_frame(
    *, bars: pd.DataFrame, positions: np.ndarray, selected: np.ndarray, direction: str, side: int,
    window: int, thresholds: tuple[float, ...], threshold_masks: dict[float, np.ndarray],
    dedup_masks: dict[float, np.ndarray], price_features: Any, path: dict[str, Any],
    descriptors: dict[str, np.ndarray], flags: dict[str, np.ndarray], signal_features: dict[str, np.ndarray],
    post_features: dict[str, np.ndarray], event_id_start: int,
) -> pd.DataFrame:
    idx = np.flatnonzero(selected)
    pos = positions[idx]
    signal_start = bars.index[pos]
    signal_end = signal_start + pd.Timedelta(minutes=1)
    entry_pos = pos + 1
    data: dict[str, Any] = {
        "event_id": np.arange(event_id_start, event_id_start + len(idx), dtype=np.int64),
        "direction": direction, "side": int(side), "impulse_window": int(window),
        "signal_bar_start": signal_start, "signal_bar_end": signal_end, "signal_time": signal_end,
        "entry_time": bars.index[entry_pos], "expected_entry_time": signal_end,
        "entry_price": pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)[entry_pos],
        "expected_entry_price": pd.to_numeric(bars["open"], errors="coerce").reindex(signal_end).to_numpy(dtype=float),
        "entry_not_next_open_flag": bars.index[entry_pos] != signal_end,
        "synthetic_bar_dependency_flag": np.zeros(len(idx), dtype=bool),
        "primary_path_label": flags["primary_path_label"][idx],
        "impulse_return": price_features.impulse_return[pos],
        "normalized_impulse": price_features.normalized_impulse[pos],
        "directional_efficiency": price_features.directional_efficiency[pos],
        "window_range_bps": price_features.window_range_bps[pos],
        "peak_mfe_minute": path["peak_mfe_minute"][idx],
        "close_peak_minute": path["close_peak_minute"][idx],
        "mfe_60m": descriptors["mfe_60m"][idx], "mae_60m": descriptors["mae_60m"][idx],
        "final_return_60m": descriptors["final_return_60m"][idx],
    }
    data["entry_price_mismatch_flag"] = ~np.isclose(data["entry_price"], data["expected_entry_price"], rtol=0.0, atol=1e-12)
    for name, values in flags.items():
        if name != "primary_path_label":
            data[f"path_{name}_flag"] = values[idx]
    for threshold in thresholds:
        tag = str(float(threshold)).replace(".", "p")
        data[f"event_{tag}_flag"] = threshold_masks[float(threshold)][idx]
        data[f"deduplicated_{tag}_flag"] = dedup_masks[float(threshold)][idx]
    for name, values in {**signal_features, **post_features}.items():
        data[name] = values[idx]
    return pd.DataFrame(data)


def _signal_audit(events: pd.DataFrame, checkpoints: tuple[int, ...]) -> pd.DataFrame:
    cols = [
        "event_id", "direction", "impulse_window", "signal_bar_start", "signal_bar_end", "signal_time",
        "entry_time", "expected_entry_time", "entry_price", "expected_entry_price",
        "entry_not_next_open_flag", "entry_price_mismatch_flag", "synthetic_bar_dependency_flag",
    ]
    out = events[cols].copy() if not events.empty else pd.DataFrame(columns=cols)
    if out.empty:
        out["lookahead_flag"] = pd.Series(dtype=bool)
        return out
    for k in checkpoints:
        out[f"post{k}_feature_available_time"] = pd.to_datetime(out["signal_time"]) + pd.Timedelta(minutes=int(k))
        out[f"post{k}_earliest_execution_time"] = pd.to_datetime(out["signal_time"]) + pd.Timedelta(minutes=int(k) + 1)
    out["lookahead_flag"] = (
        out["entry_not_next_open_flag"].astype(bool)
        | out["entry_price_mismatch_flag"].astype(bool)
        | out["synthetic_bar_dependency_flag"].astype(bool)
    )
    return out


def _build_brief(path_dist: pd.DataFrame, separation: pd.DataFrame, yearly: pd.DataFrame, meta: dict[str, Any]) -> str:
    lines = [
        "# Round 08 Research Brief", "",
        "## Scope", "",
        "This round is CVD/order-flow path anatomy only. It does not filter entries or simulate a strategy.", "",
        "## Path definitions", "",
        "- immediate_runner: favorable 15bps close milestone occurs first and favorable 50bps is reached within 5m.",
        "- pullback_runner: adverse 15bps occurs first, then favorable 50bps is reached within 60m.",
        "- one_sided_continuation: favorable 25bps occurs and adverse 25bps does not occur within 60m.",
        "- directional_failure: adverse 25bps occurs and favorable 25bps does not occur within 60m.",
        "- two_sided_expansion: both favorable and adverse 25bps occur within 60m.",
        "- muted: neither side reaches 25bps within 60m.", "",
        "These are descriptive labels, not optimized trading rules.", "",
        "## CVD interpretation", "",
        "Absolute cumulative CVD level is not used. All features use causal window changes in delta_notional, normalized by traded notional, plus large-trade pressure and price-flow alignment.", "",
    ]
    if not path_dist.empty:
        focus = path_dist[(path_dist["event_set"] == "deduplicated") & (~path_dist["path_label"].astype(str).str.startswith("flag::"))].copy()
        focus = focus.sort_values("events", ascending=False).head(12)
        lines.extend(["## Largest path cohorts", "", focus.to_markdown(index=False), ""])
    if not separation.empty:
        focus = separation[(separation["event_set"] == "deduplicated") & (separation["positive_events"] >= 100) & (separation["negative_events"] >= 100)].copy()
        focus = focus.sort_values("auc_separation", ascending=False).head(20)
        lines.extend(["## Strongest univariate descriptive separations", "", focus.to_markdown(index=False), ""])
    lines.extend([
        "## Interpretation guardrails", "",
        "- A high AUC or cohort difference is exploratory evidence, not an entry threshold.",
        "- Post1/post3/post5/post15 features are only available after those bars close.",
        "- Any later strategy must validate a small, mechanism-based rule on yearly splits, costs and delays.",
        "- LONG and SHORT must be evaluated separately.", "",
        "## Run metadata", "", "```json", json.dumps(meta, ensure_ascii=False, indent=2), "```", "",
    ])
    return "\n".join(lines)


def _update_log(log_path: Path, meta: dict[str, Any]) -> None:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# ETH Directional Impulse Continuation Research Log\n"
    marker = "## Round 08 — CVD Path-Regime Anatomy"
    if marker in text:
        return
    block = f"""

{marker}

- 研究问题：不同冲击后路径类型，是否在信号时及信号后 1m/3m/5m/15m 出现可解释的 CVD/主动成交差异？
- 研究假设：较长价格冲击可作为环境，较短 CVD 窗口刻画立即延续、回踩、衰竭、吸收和反向接管。
- 与上一轮相比改变：不再尝试退出规则；先将 Round 07 路径分类，并做因果订单流解剖。
- 使用数据：本地 OKX 1m trade bar；build_missing=False；不使用普通K线。
- 绝对 CVD：不使用。仅使用窗口 delta_notional 累计变化、delta pressure、large-trade pressure 和加速度。
- 时序：signal 特征在 signal bar close 可见；post-k 特征仅在 k 分钟闭合后可见，最早下一根 open 执行。
- 当前状态：等待生产运行。
- 禁止解释：本轮任何分层或 AUC 都不是策略阈值，不允许直接 promoted to backtest。
- 运行配置：windows={meta['impulse_windows']} thresholds={meta['thresholds']} flow_windows={meta['flow_windows']} checkpoints={meta['checkpoints']}。
"""
    log_path.write_text(text.rstrip() + block + "\n", encoding="utf-8")


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    windows = r01._parse_int_csv(args.impulse_windows)
    thresholds = r01._parse_float_csv(args.thresholds)
    flow_windows = _parse_positive_ints(args.flow_windows, name="flow-windows")
    checkpoints = _parse_positive_ints(args.checkpoints, name="checkpoints")
    max_path = int(args.max_path_minutes)
    if max_path < 60:
        raise ValueError("max-path-minutes must be at least 60 for fixed path labels")
    if max(flow_windows) > 60 or max(checkpoints) > max_path:
        raise ValueError("flow windows/checkpoints exceed supported path")
    if int(args.vol_min_periods) > int(args.vol_lookback_bars):
        raise ValueError("vol-min-periods cannot exceed vol-lookback-bars")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "11_events.csv"
    audit_path = out_dir / "12_signal_audit.csv"
    events_path.unlink(missing_ok=True)
    audit_path.unlink(missing_ok=True)

    validation = r01.validate_bars(bars, args)
    arr = _flow_arrays(bars)
    flow_audit = _flow_data_audit(arr)
    if flow_audit["positive_notional_rate"] < 0.90:
        raise RuntimeError(f"Trade-bar order-flow coverage is unexpectedly low: {flow_audit}")

    masks = r02._eligible_masks(bars, args, (max_path,))
    study_months = int(masks["study_months"])
    log_return, abs_price_change, historical_1m_vol = r01.build_base_volatility(
        bars, int(args.vol_lookback_bars), int(args.vol_min_periods)
    )
    # PROFILE_OFFSETS includes -15, so the observed pre-window must include p-15..p.
    max_pre = max(max(flow_windows), max(checkpoints), 16)
    n = len(bars)
    minimum_threshold = min(thresholds)

    count_rows: list[dict[str, Any]] = []
    path_dist_rows: list[dict[str, Any]] = []
    signal_path_rows: list[dict[str, Any]] = []
    post_path_rows: list[dict[str, Any]] = []
    separation_rows: list[dict[str, Any]] = []
    quintile_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    event_id_cursor = 1
    first_event_write = True
    first_audit_write = True
    compact_rows_written = 0

    print("[feature build] causal price and order-flow arrays", flush=True)
    with ProgressReporter(
        label="[cvd anatomy] direction/windows", total=len(windows) * 2,
        every=max(1, int(args.progress_every)), enabled=not args.no_progress,
    ) as progress:
        done = 0
        for window in windows:
            price_features = r01.build_window_features(
                bars, window, log_return, abs_price_change, historical_1m_vol
            )
            norm = price_features.normalized_impulse
            for direction, side in (("LONG", 1), ("SHORT", -1)):
                directed_norm = float(side) * norm
                all_min_positions = np.flatnonzero(
                    np.isfinite(directed_norm) & (directed_norm >= float(minimum_threshold))
                )
                base_eligible = all_min_positions[masks["eligible"][all_min_positions]]
                observed_ok = _observed_window_mask(arr, base_eligible, max_pre, max_path)
                eligible_positions = base_eligible[observed_ok]
                rows, threshold_masks, dedup_masks = r07._event_count_rows(
                    direction=direction, window=window, thresholds=thresholds,
                    all_min_positions=all_min_positions, eligible_positions=eligible_positions,
                    directed_norm=directed_norm, masks=masks, study_months=study_months, n=n,
                )
                # r07 helper builds the nested threshold/dedup masks. Recompute counts
                # on the stricter Round-08 eligible pool, which additionally requires
                # the full pre-signal order-flow window to be observed.
                for rec in rows:
                    threshold = float(rec["threshold"])
                    raw_count = int(threshold_masks[threshold].sum())
                    dedup_count = int(dedup_masks[threshold].sum())
                    rec["events"] = raw_count if rec["event_set"] == "raw" else dedup_count
                    rec["events_per_month"] = float(rec["events"] / max(1, study_months))
                    rec["overlap_ratio"] = 1.0 - dedup_count / raw_count if raw_count else np.nan
                    rec["pre_signal_orderflow_window_observed_required"] = int(max_pre)
                    rec["eligible_after_orderflow_gap_check"] = int(len(eligible_positions))
                count_rows.extend(rows)

                if len(eligible_positions):
                    signal_features = _signal_flow_features(arr, eligible_positions, side, flow_windows)
                    post_features = _checkpoint_flow_features(arr, eligible_positions, side, checkpoints)
                    with tempfile.TemporaryDirectory(prefix=f"dic_r08_{direction.lower()}_{window}m_") as tmp:
                        path = r07._build_path_memmaps(
                            bars, eligible_positions, side=side, max_path=max_path,
                            chunk_size=int(args.path_chunk_size), tmp_dir=Path(tmp),
                            label=f"[path build] {direction} {window}m chunks",
                            progress_enabled=not args.no_progress,
                        )
                        descriptors = r07._path_descriptor_arrays(
                            path["close_path"], path["running_mfe"], path["running_mae"],
                            activation_bps=(15, 25, 50), giveback_bps=(10,),
                        )
                        flags = _path_flags(descriptors)
                        signal_index = pd.to_datetime(bars.index[eligible_positions])
                        years = signal_index.year.to_numpy(dtype=int)

                        for threshold in thresholds:
                            for event_set, selected in (
                                ("raw", threshold_masks[float(threshold)]),
                                ("deduplicated", dedup_masks[float(threshold)]),
                            ):
                                path_dist_rows.extend(_path_distribution_rows(
                                    direction=direction, window=window, threshold=threshold,
                                    event_set=event_set, selected=selected, flags=flags, months=study_months,
                                ))
                                signal_path_rows.extend(_feature_by_path_rows(
                                    direction=direction, window=window, threshold=threshold,
                                    event_set=event_set, selected=selected, flags=flags,
                                    features=signal_features, stage="signal_closed_bar",
                                ))
                                post_path_rows.extend(_feature_by_path_rows(
                                    direction=direction, window=window, threshold=threshold,
                                    event_set=event_set, selected=selected, flags=flags,
                                    features=post_features, stage="post_signal_checkpoint",
                                ))
                                separation_rows.extend(_separation_rows(
                                    direction=direction, window=window, threshold=threshold,
                                    event_set=event_set, selected=selected, flags=flags,
                                    features=signal_features, stage="signal_closed_bar",
                                ))
                                separation_rows.extend(_separation_rows(
                                    direction=direction, window=window, threshold=threshold,
                                    event_set=event_set, selected=selected, flags=flags,
                                    features=post_features, stage="post_signal_checkpoint",
                                ))

                            dedup_selected = dedup_masks[float(threshold)]
                            compact_feature_set = {
                                **{k: v for k, v in signal_features.items() if ("dir_delta_pressure" in k or "dir_large_delta_pressure" in k or "absorption_proxy" in k)},
                                **{k: v for k, v in post_features.items() if ("dir_delta_pressure" in k or "delta_pressure_accel" in k or "notional_speed_ratio" in k or "absorption_proxy" in k)},
                            }
                            quintile_rows.extend(_quintile_rows(
                                direction=direction, window=window, threshold=threshold,
                                selected=dedup_selected, flags=flags, features=compact_feature_set, years=years,
                            ))
                            profile_rows.extend(_minute_profile_rows(
                                bars=bars, arr=arr, positions=eligible_positions, side=side,
                                direction=direction, window=window, threshold=threshold,
                                selected=dedup_selected, flags=flags,
                            ))
                            yearly_rows.extend(_yearly_rows(
                                direction=direction, window=window, threshold=threshold,
                                selected=dedup_selected, flags=flags, signal_features=signal_features,
                                post_features=post_features, years=years,
                            ))

                        if not args.skip_events_csv:
                            selected = dedup_masks[float(minimum_threshold)]
                            compact = _compact_event_frame(
                                bars=bars, positions=eligible_positions, selected=selected,
                                direction=direction, side=side, window=window, thresholds=thresholds,
                                threshold_masks=threshold_masks, dedup_masks=dedup_masks,
                                price_features=price_features, path=path, descriptors=descriptors,
                                flags=flags, signal_features=signal_features, post_features=post_features,
                                event_id_start=event_id_cursor,
                            )
                            event_id_cursor += len(compact)
                            compact_rows_written += len(compact)
                            r01._write_stream_csv(compact, events_path, first_write=first_event_write)
                            first_event_write = False
                            audit = _signal_audit(compact, checkpoints)
                            r01._write_stream_csv(audit, audit_path, first_write=first_audit_write)
                            first_audit_write = False
                            del compact, audit
                        for key in ("close_path", "running_mfe", "running_mae"):
                            path[key]._mmap.close()  # type: ignore[attr-defined]
                        del path, descriptors, flags
                    del signal_features, post_features
                done += 1
                progress.update(done)
            del price_features

    event_counts = pd.DataFrame(count_rows)
    path_distribution = pd.DataFrame(path_dist_rows)
    signal_by_path = pd.DataFrame(signal_path_rows)
    post_by_path = pd.DataFrame(post_path_rows)
    separation = pd.DataFrame(separation_rows)
    quintiles = pd.DataFrame(quintile_rows)
    minute_profile = pd.DataFrame(profile_rows)
    yearly = pd.DataFrame(yearly_rows)

    if first_event_write:
        pd.DataFrame(columns=["event_id", "direction", "impulse_window"]).to_csv(events_path, index=False)
        pd.DataFrame(columns=["event_id", "lookahead_flag"]).to_csv(audit_path, index=False)

    # Fixed, descriptive threshold-neighborhood table; no best-parameter selection.
    plateau = path_distribution[
        (path_distribution["event_set"] == "deduplicated")
        & path_distribution["path_label"].isin(["immediate_runner", "pullback_runner", "directional_failure", "two_sided_expansion", "muted"])
    ].copy()
    plateau = plateau.sort_values(["direction", "impulse_window", "path_label", "threshold"])

    # Direct long/short comparison of path rates.
    if not plateau.empty:
        keys = ["impulse_window", "threshold", "path_label"]
        long = plateau[plateau["direction"] == "LONG"][keys + ["events", "path_rate"]].rename(columns={"events": "long_events", "path_rate": "long_path_rate"})
        short = plateau[plateau["direction"] == "SHORT"][keys + ["events", "path_rate"]].rename(columns={"events": "short_events", "path_rate": "short_path_rate"})
        long_short = long.merge(short, on=keys, how="outer")
        long_short["long_minus_short_path_rate"] = long_short["long_path_rate"] - long_short["short_path_rate"]
    else:
        long_short = pd.DataFrame()

    meta = {
        "script_name": SCRIPT_NAME, "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID, "edge_id": EDGE_ID,
        "portfolio_plan": "ETH_NOVA_PORTFOLIO", "title": TITLE,
        "status": "research_only_cvd_path_anatomy", "symbol": args.symbol,
        "timeframe": args.timeframe, "data_source": "OKXTradeBarLoader local DB",
        "trade_bar_db_path": str(r01._trade_bar_db_path(args)), "build_missing": False,
        "ordinary_kline_download_enabled": False,
        "timezone_convention": "UTC+8 project convention; timezone-naive index",
        "warmup_start_date": args.warmup_start_date, "research_start": args.start_date,
        "research_end": args.end_date, "impulse_windows": list(windows),
        "thresholds": list(thresholds), "flow_windows": list(flow_windows),
        "checkpoints": list(checkpoints), "max_path_minutes": max_path,
        "path_labels_are_ex_post_only": True,
        "absolute_cvd_level_used": False,
        "signal_flow_available_time": "signal_bar_end",
        "post_flow_available_time": "signal_time + checkpoint minutes",
        "post_flow_earliest_execution": "next 1m open after checkpoint close",
        "deduplication": "threshold-specific cooldown=impulse_window",
        "path_chunk_size": int(args.path_chunk_size),
        "path_storage": "temporary float32 memmaps per direction/window",
        "compact_event_rows_written": int(compact_rows_written),
        "events_csv_skipped_for_development": bool(args.skip_events_csv),
        "research_month_count": study_months, "input_rows": int(len(bars)),
        "synthetic_gap_bar_count": int((~bars["source_bar_observed_flag"].astype(bool)).sum()),
        "gap_handling": str(bars.attrs.get("gap_policy", "")),
        "validation": validation, "flow_data_audit": flow_audit,
        "strategy_filter_added": False, "exit_rule_simulated": False,
        "parameter_optimization_performed": False,
        "research_boundary": "describe path cohorts and causal CVD/order-flow state only",
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    brief = _build_brief(path_distribution, separation, yearly, meta)

    artifacts = [
        (event_counts, out_dir / "01_event_counts.csv"),
        (path_distribution, out_dir / "02_path_class_distribution.csv"),
        (signal_by_path, out_dir / "03_signal_cvd_by_path.csv"),
        (post_by_path, out_dir / "04_post_signal_cvd_by_path.csv"),
        (separation, out_dir / "05_cvd_feature_separation.csv"),
        (quintiles, out_dir / "06_cvd_feature_quintiles.csv"),
        (minute_profile, out_dir / "07_cvd_minute_profile.csv"),
        (yearly, out_dir / "08_yearly_path_cvd.csv"),
        (plateau, out_dir / "09_threshold_plateau.csv"),
        (long_short, out_dir / "10_long_short_comparison.csv"),
    ]
    print("[artifacts] writing CVD path-anatomy report", flush=True)
    with ProgressReporter(label="[artifacts] tables", total=len(artifacts) + 3, every=1, enabled=not args.no_progress) as progress:
        done = 0
        for frame, path in artifacts:
            frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
            done += 1
            progress.update(done)
        r01._write_json(meta, out_dir / "13_run_meta.json")
        done += 1
        progress.update(done)
        (out_dir / "14_research_brief.md").write_text(brief, encoding="utf-8")
        done += 1
        progress.update(done)
        _update_log(Path(__file__).resolve().with_name("00_research_log.md"), meta)
        done += 1
        progress.update(done)

    if not args.skip_review_pack:
        print("[review pack] packaging CVD path-anatomy artifacts", flush=True)
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report_dir={out_dir}", flush=True)
    return {"report_dir": out_dir, "events": events_path, "audit": audit_path, "review_pack": out_dir / "gpt_review_pack.zip"}


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] deterministic CVD path-regime anatomy", flush=True)
    # Unit checks for causal prefix windows and path labels.
    idx = pd.date_range("2023-01-01", periods=80, freq="1min")
    base = np.linspace(100.0, 102.0, len(idx))
    bars = pd.DataFrame({
        "open": base, "high": base + 0.2, "low": base - 0.2, "close": base + 0.05,
        "volume": 1.0, "trades_count": 10,
        "buy_notional": np.where(np.arange(len(idx)) % 2 == 0, 700.0, 400.0),
        "sell_notional": np.where(np.arange(len(idx)) % 2 == 0, 300.0, 600.0),
        "notional": 1000.0,
        "delta_notional": np.where(np.arange(len(idx)) % 2 == 0, 400.0, -200.0),
        "large_buy_notional": 100.0, "large_sell_notional": 50.0,
        "large_delta_notional": 50.0,
        "source_bar_observed_flag": True,
    }, index=idx)
    arr = _flow_arrays(bars)
    pos = np.asarray([20], dtype=int)
    sig = _signal_flow_features(arr, pos, 1, (1, 3, 5, 15))
    post = _checkpoint_flow_features(arr, pos, 1, (1, 3, 5, 15))
    expected3 = np.sum(arr["delta_notional"][18:21]) / np.sum(arr["notional"][18:21])
    if not np.isclose(sig["signal_w3_dir_delta_pressure"][0], expected3):
        raise AssertionError("signal 3m delta pressure is not causal/window-correct")
    expected_post3 = np.sum(arr["delta_notional"][21:24]) / np.sum(arr["notional"][21:24])
    if not np.isclose(post["post3_dir_delta_pressure"][0], expected_post3):
        raise AssertionError("post 3m delta pressure window is wrong")

    desc = {
        "close_favorable_first_15bps_min": np.asarray([1, 4, 0, 0]),
        "close_adverse_first_15bps_min": np.asarray([0, 1, 2, 0]),
        "close_favorable_first_25bps_min": np.asarray([2, 8, 0, 0]),
        "close_adverse_first_25bps_min": np.asarray([0, 2, 3, 0]),
        "close_favorable_first_50bps_min": np.asarray([3, 12, 0, 0]),
    }
    flags = _path_flags(desc)
    expected = ["immediate_runner", "pullback_runner", "directional_failure", "muted"]
    if flags["primary_path_label"].tolist() != expected:
        raise AssertionError(f"path taxonomy mismatch: {flags['primary_path_label'].tolist()}")

    raw = r01._synthetic_bars().drop(r01._synthetic_bars().index[3700:3707])
    reg = r01._regularize_trade_bar_axis(raw)
    # Synthetic bars do not have order flow; add internally consistent fields.
    ret = pd.to_numeric(reg["close"], errors="coerce").pct_change().fillna(0.0).to_numpy()
    notion = np.full(len(reg), 1_000_000.0)
    delta = np.tanh(ret * 100.0) * notion * 0.4
    reg["notional"] = notion
    reg["buy_notional"] = (notion + delta) / 2.0
    reg["sell_notional"] = (notion - delta) / 2.0
    reg["delta_notional"] = delta
    reg["trades_count"] = 100
    reg["large_buy_notional"] = reg["buy_notional"] * 0.1
    reg["large_sell_notional"] = reg["sell_notional"] * 0.1
    reg["large_delta_notional"] = reg["large_buy_notional"] - reg["large_sell_notional"]

    log_path = Path(__file__).resolve().with_name("00_research_log.md")
    original_log = log_path.read_text(encoding="utf-8") if log_path.exists() else None
    try:
        with tempfile.TemporaryDirectory(prefix="dic_r08_") as tmp:
            args.out_dir = tmp
            args.warmup_start_date = "2022-12-20"
            args.start_date = "2022-12-23"
            args.end_date = "2022-12-24"
            args.vol_lookback_bars = 720
            args.vol_min_periods = 360
            args.impulse_windows = "1,3"
            args.thresholds = "1.0,1.5"
            args.flow_windows = "1,3,5,15"
            args.checkpoints = "1,3,5,15"
            args.max_path_minutes = 60
            args.path_chunk_size = 256
            args.skip_review_pack = True
            args.skip_events_csv = False
            args.no_progress = True
            result = run_research(reg, args)
            required = [
                "01_event_counts.csv", "02_path_class_distribution.csv", "03_signal_cvd_by_path.csv",
                "04_post_signal_cvd_by_path.csv", "05_cvd_feature_separation.csv",
                "06_cvd_feature_quintiles.csv", "07_cvd_minute_profile.csv", "08_yearly_path_cvd.csv",
                "09_threshold_plateau.csv", "10_long_short_comparison.csv", "11_events.csv",
                "12_signal_audit.csv", "13_run_meta.json", "14_research_brief.md",
            ]
            missing = [x for x in required if not (result["report_dir"] / x).exists()]
            if missing:
                raise AssertionError(f"missing self-test artifacts: {missing}")
            audit = pd.read_csv(result["audit"])
            if not audit.empty and audit["lookahead_flag"].astype(bool).any():
                raise AssertionError("causal audit contains flags")
            meta = json.loads((result["report_dir"] / "13_run_meta.json").read_text(encoding="utf-8"))
            if meta.get("absolute_cvd_level_used") is not False:
                raise AssertionError("absolute CVD must not be used")
            if int(meta.get("synthetic_gap_bar_count", 0)) != 7:
                raise AssertionError("seven-minute synthetic gap was not preserved")
    finally:
        if original_log is None:
            log_path.unlink(missing_ok=True)
        else:
            log_path.write_text(original_log, encoding="utf-8")
    print("[self-test] PASS", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args)
    bars = r01.load_bars(args)
    run_research(bars, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
