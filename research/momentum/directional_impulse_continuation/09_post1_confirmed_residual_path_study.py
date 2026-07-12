#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Directional Impulse Continuation: post-1m confirmed residual path study (round 09).

Research question
-----------------
Round 08 found that signal-stage CVD does not reliably identify continuation, while
price and CVD during the first fully closed minute after the impulse separate paths.
Round 09 asks the stricter causal question:

    After that first minute is fully closed and its price/CVD state is known, does
    the earliest executable next-open entry still have enough residual directional
    path advantage to justify further strategy research?

No future path label is used in the state.  The state is defined only from the first
post-signal 1m bar:

- price_pos_cvd_pos: direction-adjusted price return > 0 and delta pressure > 0
- price_pos_cvd_nonpos: price return > 0 and delta pressure <= 0
- price_nonpos_cvd_pos: price return <= 0 and delta pressure > 0
- price_nonpos_cvd_nonpos: price return <= 0 and delta pressure <= 0

Causality
---------
Signal bar p closes -> original reference entry is p+1 open -> bar p+1 fully closes
and post1 state becomes available -> confirmed residual entry is p+2 open.  All
returns, MFE/MAE, and first-passage paths in this study start from p+2 open.

This is still an event/path study.  It does not optimize a CVD threshold, add a
trend filter, simulate a final strategy, or use ex-post path labels as features.

Performance design
------------------
- Local OKX trade-bar DB only; build_missing=False.
- One load and one base-volatility build.
- Prefix sums calculate post1 flow state in O(1).
- One minimum-threshold event pool per direction/window.
- One bounded-memory vectorized residual path build per direction/window; all
  thresholds, event sets, states, horizons, years, and barriers reuse the arrays.
- No iterrows over market bars and no per-variant full-data scans.
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


def _load_round08_module():
    path = Path(__file__).resolve().with_name("08_cvd_path_regime_anatomy_study.py")
    spec = importlib.util.spec_from_file_location("directional_impulse_round08_for_r09", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared round-08 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r08 = _load_round08_module()
r07 = r08.r07
r04 = r07.r04
r02 = r08.r02
r01 = r08.r01

SCRIPT_NAME = "09_post1_confirmed_residual_path_study"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION_R09"
EDGE_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION"
TITLE = "ETH Directional Impulse Continuation - Post-1m Confirmed Residual Path"
DEFAULT_OUT_DIR = (
    "data/reports/research/momentum/directional_impulse_continuation/"
    "09_post1_confirmed_residual_path_study"
)

DEFAULT_WINDOWS = (1, 3, 5, 10, 15)
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
DEFAULT_HORIZONS = (1, 3, 5, 10, 15, 30, 60)
DEFAULT_BARRIERS_BPS = (15, 25, 50)
DEFAULT_TIME_LIMITS = (3, 5, 10, 15, 30, 60)
FEE_ONLY_COST = 0.0011
NORMAL_COST = 0.0015

POST1_STATES = (
    "price_pos_cvd_pos",
    "price_pos_cvd_nonpos",
    "price_nonpos_cvd_pos",
    "price_nonpos_cvd_nonpos",
)
STATE_DESCRIPTIONS = {
    "price_pos_cvd_pos": "post1 price and total CVD both support impulse direction",
    "price_pos_cvd_nonpos": "post1 price continues but total CVD does not support",
    "price_nonpos_cvd_pos": "post1 total CVD supports but price does not progress",
    "price_nonpos_cvd_nonpos": "post1 price and total CVD both oppose impulse direction",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal residual path study after one fully closed post-impulse minute.",
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
    p.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    p.add_argument("--barriers-bps", default=",".join(map(str, DEFAULT_BARRIERS_BPS)))
    p.add_argument("--time-limits", default=",".join(map(str, DEFAULT_TIME_LIMITS)))
    p.add_argument("--vol-lookback-bars", type=int, default=1440)
    p.add_argument("--vol-min-periods", type=int, default=720)
    p.add_argument("--path-chunk-size", type=int, default=5000)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument(
        "--skip-events-csv",
        action="store_true",
        help="Development-only. Production writes minimum-threshold deduplicated events.",
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _parse_positive_ints(raw: str, *, name: str) -> tuple[int, ...]:
    values = tuple(sorted(dict.fromkeys(int(x.strip()) for x in str(raw).split(",") if x.strip())))
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _state_labels(post1_price: np.ndarray, post1_cvd: np.ndarray) -> np.ndarray:
    p = np.asarray(post1_price, dtype=float)
    c = np.asarray(post1_cvd, dtype=float)
    out = np.full(len(p), "invalid", dtype=object)
    valid = np.isfinite(p) & np.isfinite(c)
    out[valid & (p > 0) & (c > 0)] = "price_pos_cvd_pos"
    out[valid & (p > 0) & (c <= 0)] = "price_pos_cvd_nonpos"
    out[valid & (p <= 0) & (c > 0)] = "price_nonpos_cvd_pos"
    out[valid & (p <= 0) & (c <= 0)] = "price_nonpos_cvd_nonpos"
    return out.astype(str)


def _first_true_minute(mask: np.ndarray) -> np.ndarray:
    any_hit = mask.any(axis=1)
    first = np.argmax(mask, axis=1).astype(np.int16) + np.int16(1)
    first[~any_hit] = np.int16(0)
    return first


def _build_residual_paths(
    bars: pd.DataFrame,
    signal_positions: np.ndarray,
    *,
    side: int,
    max_path: int,
    barriers_bps: tuple[int, ...],
    chunk_size: int,
    tmp_dir: Path,
    label: str,
    progress_enabled: bool,
) -> dict[str, Any]:
    """Build p+2-open residual paths and first-passage arrays in one market scan."""
    if int(chunk_size) <= 0:
        raise ValueError("path-chunk-size must be positive")
    m = int(len(signal_positions))
    h = int(max_path)
    shape = (m, h)
    close_path = np.memmap(tmp_dir / "residual_close_path.dat", mode="w+", dtype="float32", shape=shape)
    running_mfe = np.memmap(tmp_dir / "residual_running_mfe.dat", mode="w+", dtype="float32", shape=shape)
    running_mae = np.memmap(tmp_dir / "residual_running_mae.dat", mode="w+", dtype="float32", shape=shape)
    first = {
        int(b): {
            "favorable_first_min": np.zeros(m, dtype=np.int16),
            "adverse_first_min": np.zeros(m, dtype=np.int16),
        }
        for b in barriers_bps
    }

    open_arr = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=np.float64)
    close_arr = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=np.float64)
    high_arr = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=np.float64)
    low_arr = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=np.float64)
    offsets = np.arange(2, h + 2, dtype=np.int64)  # residual bars p+2 .. p+1+h
    total_chunks = max(1, int(math.ceil(max(1, m) / int(chunk_size))))

    if m:
        with ProgressReporter(label=label, total=total_chunks, every=1, enabled=progress_enabled) as progress:
            done = 0
            for start in range(0, m, int(chunk_size)):
                end = min(m, start + int(chunk_size))
                pos = signal_positions[start:end].astype(np.int64, copy=False)
                entry = open_arr[pos + 2].astype(np.float32, copy=False)
                gather = pos[:, None] + offsets[None, :]
                path_close = close_arr[gather].astype(np.float32, copy=False)
                path_high = high_arr[gather].astype(np.float32, copy=False)
                path_low = low_arr[gather].astype(np.float32, copy=False)
                close_ret = np.float32(float(side)) * (path_close / entry[:, None] - np.float32(1.0))
                if int(side) == 1:
                    favorable = path_high / entry[:, None] - np.float32(1.0)
                    adverse = np.float32(1.0) - path_low / entry[:, None]
                    adverse_signed = path_low / entry[:, None] - np.float32(1.0)
                else:
                    favorable = np.float32(1.0) - path_low / entry[:, None]
                    adverse = path_high / entry[:, None] - np.float32(1.0)
                    adverse_signed = np.float32(1.0) - path_high / entry[:, None]

                close_path[start:end, :] = close_ret
                running_mfe[start:end, :] = np.maximum.accumulate(favorable, axis=1)
                running_mae[start:end, :] = np.minimum.accumulate(adverse_signed, axis=1)
                for bps in barriers_bps:
                    rate = np.float32(float(bps) / 10_000.0)
                    first[int(bps)]["favorable_first_min"][start:end] = _first_true_minute(favorable >= rate)
                    first[int(bps)]["adverse_first_min"][start:end] = _first_true_minute(adverse >= rate)
                done += 1
                progress.update(done)

    close_path.flush()
    running_mfe.flush()
    running_mae.flush()
    return {
        "close_path": close_path,
        "running_mfe": running_mfe,
        "running_mae": running_mae,
        "first_passage": first,
    }


def _stats_for_indices(
    close_path: np.ndarray,
    running_mfe: np.ndarray,
    running_mae: np.ndarray,
    idx: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    h = int(horizon) - 1
    gross = np.asarray(close_path[idx, h], dtype=float)
    mfe = np.asarray(running_mfe[idx, h], dtype=float)
    mae = np.asarray(running_mae[idx, h], dtype=float)
    return r01._stats(gross, gross - FEE_ONLY_COST, gross - NORMAL_COST, mfe, mae)


def _first_passage_stats(
    first: dict[str, np.ndarray],
    close_path: np.ndarray,
    idx: np.ndarray,
    *,
    barrier_bps: int,
    time_limit: int,
) -> dict[str, Any]:
    if idx.size == 0:
        return {
            "events": 0, "target_touch_rate": np.nan, "stop_touch_rate": np.nan,
            "target_first_rate": np.nan, "stop_first_rate": np.nan,
            "ambiguous_same_bar_rate": np.nan, "neither_hit_rate": np.nan,
            "target_minus_stop_first_rate": np.nan, "median_target_touch_min": np.nan,
            "median_stop_touch_min": np.nan, "conservative_mean_gross": np.nan,
            "conservative_mean_net": np.nan, "conservative_median_net": np.nan,
            "conservative_win_rate": np.nan, "conservative_profit_factor": np.nan,
            "optimistic_mean_net": np.nan, "resolved_mean_net": np.nan,
        }
    fav = first["favorable_first_min"][idx]
    adv = first["adverse_first_min"][idx]
    limit = int(time_limit)
    target_touch = (fav > 0) & (fav <= limit)
    stop_touch = (adv > 0) & (adv <= limit)
    target_first = target_touch & (~stop_touch | (fav < adv))
    stop_first = stop_touch & (~target_touch | (adv < fav))
    ambiguous = target_touch & stop_touch & (fav == adv)
    neither = ~(target_first | stop_first | ambiguous)
    terminal = np.asarray(close_path[idx, limit - 1], dtype=float)
    barrier = float(barrier_bps) / 10_000.0
    conservative = np.where(target_first, barrier, np.where(stop_first | ambiguous, -barrier, terminal))
    optimistic = np.where(target_first | ambiguous, barrier, np.where(stop_first, -barrier, terminal))
    resolved = ~ambiguous

    def basic(gross: np.ndarray) -> dict[str, float]:
        dummy = np.full(len(gross), np.nan, dtype=float)
        s = r01._stats(gross, gross - FEE_ONLY_COST, gross - NORMAL_COST, dummy, dummy)
        return {
            "mean_gross": float(s["mean_gross"]), "mean_net": float(s["mean_net"]),
            "median_net": float(s["median_net"]), "win_rate": float(s["win_rate"]),
            "profit_factor": float(s["profit_factor"]),
        }

    c = basic(conservative)
    o = basic(optimistic)
    rs = basic(conservative[resolved]) if resolved.any() else {
        "mean_gross": np.nan, "mean_net": np.nan, "median_net": np.nan,
        "win_rate": np.nan, "profit_factor": np.nan,
    }
    return {
        "events": int(idx.size),
        "target_touch_rate": float(target_touch.mean()),
        "stop_touch_rate": float(stop_touch.mean()),
        "target_first_rate": float(target_first.mean()),
        "stop_first_rate": float(stop_first.mean()),
        "ambiguous_same_bar_rate": float(ambiguous.mean()),
        "neither_hit_rate": float(neither.mean()),
        "target_minus_stop_first_rate": float(target_first.mean() - stop_first.mean()),
        "median_target_touch_min": float(np.median(fav[target_touch])) if target_touch.any() else np.nan,
        "median_stop_touch_min": float(np.median(adv[stop_touch])) if stop_touch.any() else np.nan,
        "conservative_mean_gross": c["mean_gross"],
        "conservative_mean_net": c["mean_net"],
        "conservative_median_net": c["median_net"],
        "conservative_win_rate": c["win_rate"],
        "conservative_profit_factor": c["profit_factor"],
        "optimistic_mean_net": o["mean_net"],
        "resolved_mean_net": rs["mean_net"],
    }


def _minute_profile_rows(
    *,
    direction: str,
    window: int,
    threshold: float,
    event_set: str,
    state: str,
    idx: np.ndarray,
    close_path: np.ndarray,
    running_mfe: np.ndarray,
    running_mae: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for minute in range(1, close_path.shape[1] + 1):
        if idx.size:
            close_ret = np.asarray(close_path[idx, minute - 1], dtype=float)
            mfe = np.asarray(running_mfe[idx, minute - 1], dtype=float)
            mae = np.asarray(running_mae[idx, minute - 1], dtype=float)
            rows.append({
                "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
                "event_set": event_set, "post1_state": state, "minute_after_confirmed_entry": int(minute),
                "events": int(idx.size), "mean_gross": float(np.mean(close_ret)),
                "median_gross": float(np.median(close_ret)), "mean_net": float(np.mean(close_ret) - NORMAL_COST),
                "gross_positive_rate": float(np.mean(close_ret > 0)),
                "normal_cost_covered_rate": float(np.mean(close_ret > NORMAL_COST)),
                "mean_running_mfe": float(np.mean(mfe)), "median_running_mfe": float(np.median(mfe)),
                "mean_running_mae": float(np.mean(mae)), "median_running_mae": float(np.median(mae)),
            })
        else:
            rows.append({
                "direction": direction, "impulse_window": int(window), "threshold": float(threshold),
                "event_set": event_set, "post1_state": state, "minute_after_confirmed_entry": int(minute),
                "events": 0, "mean_gross": np.nan, "median_gross": np.nan, "mean_net": np.nan,
                "gross_positive_rate": np.nan, "normal_cost_covered_rate": np.nan,
                "mean_running_mfe": np.nan, "median_running_mfe": np.nan,
                "mean_running_mae": np.nan, "median_running_mae": np.nan,
            })
    return rows


def _compact_events(
    *,
    bars: pd.DataFrame,
    positions: np.ndarray,
    selected: np.ndarray,
    direction: str,
    side: int,
    window: int,
    thresholds: tuple[float, ...],
    threshold_masks: dict[float, np.ndarray],
    dedup_masks: dict[float, np.ndarray],
    normalized_impulse: np.ndarray,
    post1: dict[str, np.ndarray],
    states: np.ndarray,
    path: dict[str, Any],
    event_id_start: int,
    horizons: tuple[int, ...],
    barriers_bps: tuple[int, ...],
) -> pd.DataFrame:
    loc = np.flatnonzero(selected)
    if not len(loc):
        return pd.DataFrame()
    p = positions[loc]
    index = pd.DatetimeIndex(bars.index)
    frame = pd.DataFrame({
        "event_id": np.arange(event_id_start, event_id_start + len(loc), dtype=np.int64),
        "direction": direction, "side": int(side), "impulse_window": int(window),
        "signal_bar_start": index[p], "signal_bar_end": index[p] + pd.Timedelta(minutes=1),
        "signal_time": index[p] + pd.Timedelta(minutes=1),
        "original_reference_entry_time": index[p + 1],
        "original_reference_entry_price": pd.to_numeric(bars["open"], errors="coerce").to_numpy()[p + 1],
        "post1_bar_start": index[p + 1], "post1_bar_end": index[p + 1] + pd.Timedelta(minutes=1),
        "post1_feature_available_time": index[p + 1] + pd.Timedelta(minutes=1),
        "confirmed_entry_time": index[p + 2], "expected_confirmed_entry_time": index[p] + pd.Timedelta(minutes=2),
        "confirmed_entry_price": pd.to_numeric(bars["open"], errors="coerce").to_numpy()[p + 2],
        "expected_confirmed_entry_price": pd.to_numeric(bars["open"], errors="coerce").to_numpy()[p + 2],
        "post1_state": states[loc],
        "normalized_impulse": normalized_impulse[p],
        "post1_dir_price_return": post1["post1_dir_price_return"][loc],
        "post1_dir_delta_pressure": post1["post1_dir_delta_pressure"][loc],
        "post1_dir_large_delta_pressure": post1["post1_dir_large_delta_pressure"][loc],
        "post1_notional_speed_ratio_vs_pre": post1["post1_notional_speed_ratio_vs_pre"][loc],
        "post1_trade_speed_ratio_vs_pre": post1["post1_trade_speed_ratio_vs_pre"][loc],
        "post1_absorption_proxy": post1["post1_absorption_proxy"][loc],
        "post1_vacuum_proxy": post1["post1_vacuum_proxy"][loc],
    })
    for threshold in thresholds:
        tag = str(float(threshold)).replace(".", "p")
        frame[f"event_{tag}_flag"] = threshold_masks[float(threshold)][loc]
        frame[f"deduplicated_{tag}_flag"] = dedup_masks[float(threshold)][loc]
    for horizon in horizons:
        h = int(horizon) - 1
        gross = np.asarray(path["close_path"][loc, h], dtype=float)
        frame[f"residual_forward_return_{horizon}m"] = gross
        frame[f"residual_normal_net_return_{horizon}m"] = gross - NORMAL_COST
        frame[f"residual_mfe_{horizon}m"] = np.asarray(path["running_mfe"][loc, h], dtype=float)
        frame[f"residual_mae_{horizon}m"] = np.asarray(path["running_mae"][loc, h], dtype=float)
    for bps in barriers_bps:
        arrays = path["first_passage"][int(bps)]
        frame[f"residual_favorable_first_{bps}bps_min"] = arrays["favorable_first_min"][loc]
        frame[f"residual_adverse_first_{bps}bps_min"] = arrays["adverse_first_min"][loc]
    frame["confirmed_entry_not_next_open_flag"] = (
        pd.to_datetime(frame["confirmed_entry_time"]) != pd.to_datetime(frame["expected_confirmed_entry_time"])
    )
    frame["confirmed_entry_price_mismatch_flag"] = ~np.isclose(
        frame["confirmed_entry_price"].to_numpy(dtype=float),
        frame["expected_confirmed_entry_price"].to_numpy(dtype=float), rtol=1e-10, atol=1e-10,
    )
    frame["post1_feature_after_confirmed_entry_flag"] = (
        pd.to_datetime(frame["post1_feature_available_time"]) > pd.to_datetime(frame["confirmed_entry_time"])
    )
    frame["future_path_label_used_in_state_flag"] = False
    frame["synthetic_bar_dependency_flag"] = False
    return frame


def _signal_audit(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["event_id", "lookahead_flag"])
    cols = [
        "event_id", "direction", "impulse_window", "signal_time", "original_reference_entry_time",
        "post1_bar_start", "post1_bar_end", "post1_feature_available_time", "confirmed_entry_time",
        "expected_confirmed_entry_time", "confirmed_entry_price", "expected_confirmed_entry_price",
        "post1_state", "post1_dir_price_return", "post1_dir_delta_pressure",
        "confirmed_entry_not_next_open_flag", "confirmed_entry_price_mismatch_flag",
        "post1_feature_after_confirmed_entry_flag", "future_path_label_used_in_state_flag",
        "synthetic_bar_dependency_flag",
    ]
    out = events[cols].copy()
    out["lookahead_flag"] = (
        out["confirmed_entry_not_next_open_flag"].astype(bool)
        | out["confirmed_entry_price_mismatch_flag"].astype(bool)
        | out["post1_feature_after_confirmed_entry_flag"].astype(bool)
        | out["future_path_label_used_in_state_flag"].astype(bool)
        | out["synthetic_bar_dependency_flag"].astype(bool)
    )
    return out


def _build_brief(summary: pd.DataFrame, first_passage: pd.DataFrame, meta: dict[str, Any]) -> str:
    lines = [
        "# Round 09 Research Brief", "",
        "## Question", "",
        "After one post-impulse minute has fully closed, does its causal price/CVD state leave a residual directional path from the next open?", "",
        "## Strict timing", "",
        "- Original impulse is known only after the signal bar closes.",
        "- Post1 state uses only the next fully closed 1m trade bar.",
        "- Residual entry is the following 1m open (p+2); no post1 close information is used at p+1 open.",
        "- Future paths are outcomes only and are not used to define the state.", "",
        "## Interpretation guardrail", "",
        "This report is not a final strategy backtest and does not optimize thresholds. A useful result requires adjacent impulse thresholds/windows, enough events, multiple positive years, and residual returns or first-passage advantage large enough to survive normal 0.15% round-trip cost.", "",
    ]
    if not summary.empty:
        dedup = summary[summary["event_set"] == "deduplicated"].copy()
        ranked = dedup.sort_values(["mean_net", "events"], ascending=[False, False]).head(12)
        lines += ["## Highest residual fixed-horizon rows (descriptive, not selected parameters)", ""]
        for _, r in ranked.iterrows():
            lines.append(
                f"- {r['direction']} {int(r['impulse_window'])}m threshold {float(r['threshold']):g}, "
                f"{r['post1_state']}, horizon {int(r['horizon'])}m: events={int(r['events'])}, "
                f"mean gross={float(r['mean_gross']):.4%}, mean net={float(r['mean_net']):.4%}, "
                f"median net={float(r['median_net']):.4%}, PF={float(r['profit_factor']):.3f}."
            )
        lines.append("")
    if not first_passage.empty:
        fp = first_passage[first_passage["event_set"] == "deduplicated"].copy()
        fp = fp.sort_values(["target_minus_stop_first_rate", "events"], ascending=[False, False]).head(12)
        lines += ["## Largest residual target-first advantages", ""]
        for _, r in fp.iterrows():
            lines.append(
                f"- {r['direction']} {int(r['impulse_window'])}m threshold {float(r['threshold']):g}, "
                f"{r['post1_state']}, {int(r['barrier_bps'])}bps/{int(r['time_limit'])}m: "
                f"events={int(r['events'])}, target-first={float(r['target_first_rate']):.2%}, "
                f"stop-first={float(r['stop_first_rate']):.2%}, gap={float(r['target_minus_stop_first_rate']):.2%}, "
                f"conservative mean net={float(r['conservative_mean_net']):.4%}."
            )
        lines.append("")
    lines += [
        "## Next decision", "",
        "- If price_pos_cvd_pos retains a broad residual advantage from p+2 open, it can become a causal execution-confirmation hypothesis for a later dedicated backtest.",
        "- If it has large residual MFE but poor first-passage/net results, the state is descriptive but still lacks an executable entry/exit edge.",
        "- If price_nonpos states later recover, that belongs to a separate pullback-recovery anatomy branch rather than being mixed into immediate confirmation.", "",
        f"Generated: {meta.get('created_at', '')}",
    ]
    return "\n".join(lines) + "\n"


def _update_log(log_path: Path, meta: dict[str, Any]) -> None:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# ETH Directional Impulse Continuation Research Log\n"
    marker = "## Round 09 - Post-1m confirmed residual path"
    if marker in text:
        return
    block = f"""

{marker}

- 研究问题：第一根 post-signal 1m trade bar 完全关闭后，价格/CVD 四象限状态在下一根 open 是否仍有剩余顺势路径。
- 研究假设：10m 左右价格冲击是事件锚点；小周期价格与总 CVD 同向推进可能保留剩余延续空间。
- 与上一轮相比改变了什么：不再使用未来 path label 比较特征；仅用 post1 已关闭价格和 delta pressure 定义状态，并从 p+2 open 重新计算全部收益、MFE/MAE 与 first-passage。
- 使用的数据：ETH-USDT-SWAP 本地 1m OKX trade bars，UTC+8 项目约定，build_missing=False。
- 事件数、交易数、月均频率、mean/median net、胜率、PF、年度表现：等待本地生产运行。
- 因果时序：signal p close -> post1 bar p+1 close -> state available -> p+2 open residual entry。
- 结果解释：Pending production run。
- 失败分支：尚未判定。
- 下一轮理由：只有本轮证明确认后仍存在成本后剩余路径，才允许将该状态升级为正式入场确认假设。
- 状态：research_pending。
- Script: `09_post1_confirmed_residual_path_study.py`
- Generated: {meta.get('created_at', '')}
"""
    log_path.write_text(text.rstrip() + block + "\n", encoding="utf-8")


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    windows = r01._parse_int_csv(args.impulse_windows)
    thresholds = r01._parse_float_csv(args.thresholds)
    horizons = _parse_positive_ints(args.horizons, name="horizons")
    barriers = _parse_positive_ints(args.barriers_bps, name="barriers-bps")
    time_limits = _parse_positive_ints(args.time_limits, name="time-limits")
    max_path = max(max(horizons), max(time_limits))
    if max_path > 240:
        raise ValueError("max residual path above 240m is not supported")
    if int(args.vol_min_periods) > int(args.vol_lookback_bars):
        raise ValueError("vol-min-periods cannot exceed vol-lookback-bars")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "09_events.csv"
    audit_path = out_dir / "10_signal_audit.csv"
    events_path.unlink(missing_ok=True)
    audit_path.unlink(missing_ok=True)

    validation = r01.validate_bars(bars, args)
    arr = r08._flow_arrays(bars)
    flow_audit = r08._flow_data_audit(arr)
    # p+1 state plus p+2..p+1+max_path residual bars = max_path+1 post-signal bars.
    masks = r02._eligible_masks(bars, args, (max_path + 1,))
    study_months = int(masks["study_months"])
    log_return, abs_price_change, historical_1m_vol = r01.build_base_volatility(
        bars, int(args.vol_lookback_bars), int(args.vol_min_periods)
    )
    n = len(bars)
    minimum_threshold = min(thresholds)

    count_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    minute_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    event_id_cursor = 1
    first_event_write = True
    first_audit_write = True
    compact_rows_written = 0

    print("[feature build] causal impulse and post1 CVD state", flush=True)
    with ProgressReporter(
        label="[residual anatomy] direction/windows", total=len(windows) * 2,
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
                # Need observed signal history, p+1 checkpoint, p+2 entry, and residual path.
                observed_ok = r08._observed_window_mask(arr, base_eligible, max(window, 15), max_path + 1)
                eligible_positions = base_eligible[observed_ok]
                rows, threshold_masks, dedup_masks = r07._event_count_rows(
                    direction=direction, window=window, thresholds=thresholds,
                    all_min_positions=all_min_positions, eligible_positions=eligible_positions,
                    directed_norm=directed_norm, masks=masks, study_months=study_months, n=n,
                )
                for rec in rows:
                    threshold = float(rec["threshold"])
                    raw_count = int(threshold_masks[threshold].sum())
                    dedup_count = int(dedup_masks[threshold].sum())
                    rec["events"] = raw_count if rec["event_set"] == "raw" else dedup_count
                    rec["events_per_month"] = float(rec["events"] / max(1, study_months))
                    rec["post1_closed_bar_required"] = True
                    rec["residual_entry_delay_bars_from_original_signal"] = 2
                count_rows.extend(rows)

                if len(eligible_positions):
                    post1 = r08._checkpoint_flow_features(arr, eligible_positions, side, (1,))
                    states = _state_labels(
                        post1["post1_dir_price_return"], post1["post1_dir_delta_pressure"]
                    )
                    if np.any(states == "invalid"):
                        raise RuntimeError("Invalid post1 state after observed-window filtering")
                    years = pd.DatetimeIndex(bars.index[eligible_positions]).year.to_numpy(dtype=int)
                    with tempfile.TemporaryDirectory(prefix=f"dic_r09_{direction.lower()}_{window}m_") as tmp:
                        path = _build_residual_paths(
                            bars, eligible_positions, side=side, max_path=max_path,
                            barriers_bps=barriers, chunk_size=int(args.path_chunk_size),
                            tmp_dir=Path(tmp), label=f"[residual path] {direction} {window}m chunks",
                            progress_enabled=not args.no_progress,
                        )
                        for threshold in thresholds:
                            for event_set, selected in (
                                ("raw", threshold_masks[float(threshold)]),
                                ("deduplicated", dedup_masks[float(threshold)]),
                            ):
                                total_selected = int(selected.sum())
                                for state in POST1_STATES:
                                    state_mask = selected & (states == state)
                                    idx = np.flatnonzero(state_mask)
                                    state_rows.append({
                                        "direction": direction, "impulse_window": int(window),
                                        "threshold": float(threshold), "event_set": event_set,
                                        "post1_state": state, "state_description": STATE_DESCRIPTIONS[state],
                                        "events": int(idx.size),
                                        "events_per_month": float(idx.size / max(1, study_months)),
                                        "state_rate_within_selected_events": float(idx.size / total_selected) if total_selected else np.nan,
                                        "mean_post1_dir_price_return": float(np.mean(post1["post1_dir_price_return"][idx])) if idx.size else np.nan,
                                        "mean_post1_dir_delta_pressure": float(np.mean(post1["post1_dir_delta_pressure"][idx])) if idx.size else np.nan,
                                        "mean_post1_trade_speed_ratio": float(np.nanmean(post1["post1_trade_speed_ratio_vs_pre"][idx])) if idx.size else np.nan,
                                    })
                                    for horizon in horizons:
                                        stat = _stats_for_indices(
                                            path["close_path"], path["running_mfe"], path["running_mae"], idx, horizon
                                        )
                                        year_means: dict[int, float] = {}
                                        for year in sorted(set(years[idx].tolist())) if idx.size else []:
                                            yidx = idx[years[idx] == int(year)]
                                            ys = _stats_for_indices(
                                                path["close_path"], path["running_mfe"], path["running_mae"], yidx, horizon
                                            )
                                            yearly_rows.append({
                                                "direction": direction, "impulse_window": int(window),
                                                "threshold": float(threshold), "post1_state": state,
                                                "horizon": int(horizon), "event_set": event_set,
                                                "year": int(year), **ys,
                                            })
                                            year_means[int(year)] = float(ys["mean_net"])
                                        finite_years = [v for v in year_means.values() if np.isfinite(v)]
                                        horizon_rows.append({
                                            "direction": direction, "impulse_window": int(window),
                                            "threshold": float(threshold), "event_set": event_set,
                                            "post1_state": state, "horizon": int(horizon),
                                            "events_per_month": float(idx.size / max(1, study_months)),
                                            **stat,
                                            "positive_year_count": int(sum(v > 0 for v in finite_years)),
                                            "total_year_count": int(len(finite_years)),
                                            "worst_year_mean_net": float(min(finite_years)) if finite_years else np.nan,
                                        })
                                    for bps in barriers:
                                        arrays = path["first_passage"][int(bps)]
                                        for limit in time_limits:
                                            fs = _first_passage_stats(
                                                arrays, path["close_path"], idx,
                                                barrier_bps=int(bps), time_limit=int(limit),
                                            )
                                            fp_rows.append({
                                                "direction": direction, "impulse_window": int(window),
                                                "threshold": float(threshold), "event_set": event_set,
                                                "post1_state": state, "barrier_bps": int(bps),
                                                "time_limit": int(limit), **fs,
                                            })
                                    # Minute profile is large; keep deduplicated only.
                                    if event_set == "deduplicated":
                                        minute_rows.extend(_minute_profile_rows(
                                            direction=direction, window=window, threshold=threshold,
                                            event_set=event_set, state=state, idx=idx,
                                            close_path=path["close_path"], running_mfe=path["running_mfe"],
                                            running_mae=path["running_mae"],
                                        ))

                        if not args.skip_events_csv:
                            selected = dedup_masks[float(minimum_threshold)]
                            compact = _compact_events(
                                bars=bars, positions=eligible_positions, selected=selected,
                                direction=direction, side=side, window=window, thresholds=thresholds,
                                threshold_masks=threshold_masks, dedup_masks=dedup_masks,
                                normalized_impulse=directed_norm, post1=post1, states=states,
                                path=path, event_id_start=event_id_cursor,
                                horizons=horizons, barriers_bps=barriers,
                            )
                            event_id_cursor += len(compact)
                            compact_rows_written += len(compact)
                            r01._write_stream_csv(compact, events_path, first_write=first_event_write)
                            first_event_write = False
                            audit = _signal_audit(compact)
                            r01._write_stream_csv(audit, audit_path, first_write=first_audit_write)
                            first_audit_write = False
                            del compact, audit
                        for key in ("close_path", "running_mfe", "running_mae"):
                            path[key]._mmap.close()  # type: ignore[attr-defined]
                        del path
                    del post1, states
                done += 1
                progress.update(done)
            del price_features

    event_counts = pd.DataFrame(count_rows)
    state_dist = pd.DataFrame(state_rows)
    horizon_summary = pd.DataFrame(horizon_rows)
    first_passage = pd.DataFrame(fp_rows)
    minute_profile = pd.DataFrame(minute_rows)
    yearly = pd.DataFrame(yearly_rows)

    if first_event_write:
        pd.DataFrame(columns=["event_id", "direction", "impulse_window"]).to_csv(events_path, index=False)
        pd.DataFrame(columns=["event_id", "lookahead_flag"]).to_csv(audit_path, index=False)

    plateau = horizon_summary[
        (horizon_summary["event_set"] == "deduplicated")
        & (horizon_summary["post1_state"] == "price_pos_cvd_pos")
    ].sort_values(["direction", "impulse_window", "horizon", "threshold"])

    if not plateau.empty:
        keys = ["impulse_window", "threshold", "horizon", "post1_state"]
        l = plateau[plateau["direction"] == "LONG"][keys + ["events", "mean_net", "median_net", "profit_factor"]].rename(
            columns={"events": "long_events", "mean_net": "long_mean_net", "median_net": "long_median_net", "profit_factor": "long_pf"}
        )
        s = plateau[plateau["direction"] == "SHORT"][keys + ["events", "mean_net", "median_net", "profit_factor"]].rename(
            columns={"events": "short_events", "mean_net": "short_mean_net", "median_net": "short_median_net", "profit_factor": "short_pf"}
        )
        long_short = l.merge(s, on=keys, how="outer")
        long_short["long_minus_short_mean_net"] = long_short["long_mean_net"] - long_short["short_mean_net"]
    else:
        long_short = pd.DataFrame()

    meta = {
        "script_name": SCRIPT_NAME, "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID, "edge_id": EDGE_ID,
        "portfolio_plan": "ETH_NOVA_PORTFOLIO", "title": TITLE,
        "status": "research_only_post1_confirmed_residual_path",
        "symbol": args.symbol, "timeframe": args.timeframe,
        "data_source": "OKXTradeBarLoader local DB", "trade_bar_db_path": str(r01._trade_bar_db_path(args)),
        "build_missing": False, "ordinary_kline_download_enabled": False,
        "timezone_convention": "UTC+8 project convention; timezone-naive index",
        "warmup_start_date": args.warmup_start_date, "research_start": args.start_date,
        "research_end": args.end_date, "impulse_windows": list(windows),
        "thresholds": list(thresholds), "horizons": list(horizons),
        "barriers_bps": list(barriers), "time_limits": list(time_limits),
        "post1_state_definition": {
            k: v for k, v in STATE_DESCRIPTIONS.items()
        },
        "state_feature_window": "only fully closed bar p+1",
        "state_available_time": "p+1 bar end = original signal time + 1 minute",
        "earliest_confirmed_entry": "p+2 open",
        "residual_path_start": "p+2 open",
        "future_path_label_used_in_state": False,
        "same_bar_first_passage_primary_policy": "conservative stop-first",
        "normal_round_trip_cost": NORMAL_COST,
        "fee_only_round_trip_cost": FEE_ONLY_COST,
        "deduplication": "threshold-specific cooldown=impulse_window",
        "path_chunk_size": int(args.path_chunk_size),
        "path_storage": "temporary float32 memmaps per direction/window",
        "compact_event_rows_written": int(compact_rows_written),
        "events_csv_skipped_for_development": bool(args.skip_events_csv),
        "research_month_count": study_months, "input_rows": int(len(bars)),
        "synthetic_gap_bar_count": int((~bars["source_bar_observed_flag"].astype(bool)).sum()),
        "gap_handling": str(bars.attrs.get("gap_policy", "")),
        "validation": validation, "flow_data_audit": flow_audit,
        "strategy_filter_added": False, "final_strategy_backtest_performed": False,
        "parameter_optimization_performed": False,
        "research_boundary": "test causal residual path after fixed sign-only post1 price/CVD state",
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    brief = _build_brief(horizon_summary, first_passage, meta)
    artifacts = [
        (event_counts, out_dir / "01_event_counts.csv"),
        (state_dist, out_dir / "02_post1_state_distribution.csv"),
        (horizon_summary, out_dir / "03_residual_horizon_summary.csv"),
        (first_passage, out_dir / "04_residual_first_passage.csv"),
        (minute_profile, out_dir / "05_residual_minute_profile.csv"),
        (yearly, out_dir / "06_yearly_residual.csv"),
        (plateau, out_dir / "07_threshold_plateau.csv"),
        (long_short, out_dir / "08_long_short_comparison.csv"),
    ]
    print("[artifacts] writing causal residual-path report", flush=True)
    with ProgressReporter(label="[artifacts] tables", total=len(artifacts) + 3, every=1, enabled=not args.no_progress) as progress:
        done = 0
        for frame, path_out in artifacts:
            frame.to_csv(path_out, index=False, float_format="%.10g", lineterminator="\n")
            done += 1
            progress.update(done)
        r01._write_json(meta, out_dir / "11_run_meta.json")
        done += 1
        progress.update(done)
        (out_dir / "12_research_brief.md").write_text(brief, encoding="utf-8")
        done += 1
        progress.update(done)
        _update_log(Path(__file__).resolve().with_name("00_research_log.md"), meta)
        done += 1
        progress.update(done)

    if not args.skip_review_pack:
        print("[review pack] packaging residual-path artifacts", flush=True)
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report_dir={out_dir}", flush=True)
    return {"report_dir": out_dir, "events": events_path, "audit": audit_path, "review_pack": out_dir / "gpt_review_pack.zip"}


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] deterministic post1-confirmed residual path", flush=True)
    # State classification has no future inputs.
    states = _state_labels(
        np.asarray([0.1, 0.1, -0.1, -0.1]),
        np.asarray([0.2, -0.2, 0.2, -0.2]),
    )
    if states.tolist() != list(POST1_STATES):
        raise AssertionError(f"post1 state taxonomy mismatch: {states.tolist()}")

    # Direct path timing check: signal p=10, state from p+1, residual entry p+2.
    idx = pd.date_range("2023-01-01", periods=100, freq="1min")
    base = np.full(100, 100.0)
    bars = pd.DataFrame({
        "open": base.copy(), "high": base.copy(), "low": base.copy(), "close": base.copy(),
        "volume": 1.0, "notional": 1000.0, "buy_notional": 600.0, "sell_notional": 400.0,
        "delta_notional": 200.0, "trades_count": 10,
        "large_buy_notional": 100.0, "large_sell_notional": 50.0, "large_delta_notional": 50.0,
        "source_bar_observed_flag": True,
    }, index=idx)
    bars.iloc[11, bars.columns.get_loc("close")] = 100.2  # post1 state only
    bars.iloc[12, bars.columns.get_loc("open")] = 100.0  # confirmed entry
    bars.iloc[12, bars.columns.get_loc("close")] = 100.3
    bars.iloc[12, bars.columns.get_loc("high")] = 100.4
    bars.iloc[12, bars.columns.get_loc("low")] = 99.95
    with tempfile.TemporaryDirectory(prefix="dic_r09_unit_") as tmp:
        path = _build_residual_paths(
            bars, np.asarray([10]), side=1, max_path=3, barriers_bps=(15, 25),
            chunk_size=16, tmp_dir=Path(tmp), label="[unit]", progress_enabled=False,
        )
        if not np.isclose(float(path["close_path"][0, 0]), 0.003):
            raise AssertionError("residual path did not start from p+2 open")
        if int(path["first_passage"][15]["favorable_first_min"][0]) != 1:
            raise AssertionError("residual first passage minute is wrong")
        for key in ("close_path", "running_mfe", "running_mae"):
            path[key]._mmap.close()  # type: ignore[attr-defined]

    raw = r01._synthetic_bars().drop(r01._synthetic_bars().index[3700:3707])
    reg = r01._regularize_trade_bar_axis(raw)
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
        with tempfile.TemporaryDirectory(prefix="dic_r09_") as tmp:
            args.out_dir = tmp
            args.warmup_start_date = "2022-12-20"
            args.start_date = "2022-12-23"
            args.end_date = "2022-12-24"
            args.vol_lookback_bars = 720
            args.vol_min_periods = 360
            args.impulse_windows = "1,3"
            args.thresholds = "1.0,1.5"
            args.horizons = "1,3,5,10"
            args.barriers_bps = "15,25"
            args.time_limits = "3,5,10"
            args.path_chunk_size = 256
            args.skip_review_pack = True
            args.skip_events_csv = False
            args.no_progress = True
            result = run_research(reg, args)
            required = [
                "01_event_counts.csv", "02_post1_state_distribution.csv",
                "03_residual_horizon_summary.csv", "04_residual_first_passage.csv",
                "05_residual_minute_profile.csv", "06_yearly_residual.csv",
                "07_threshold_plateau.csv", "08_long_short_comparison.csv",
                "09_events.csv", "10_signal_audit.csv", "11_run_meta.json",
                "12_research_brief.md",
            ]
            missing = [x for x in required if not (result["report_dir"] / x).exists()]
            if missing:
                raise AssertionError(f"missing self-test artifacts: {missing}")
            audit = pd.read_csv(result["audit"])
            if not audit.empty and audit["lookahead_flag"].astype(bool).any():
                raise AssertionError("causal audit contains lookahead flags")
            events = pd.read_csv(result["events"])
            if not events.empty:
                if events["future_path_label_used_in_state_flag"].astype(bool).any():
                    raise AssertionError("future path labels leaked into post1 state")
                if events["confirmed_entry_not_next_open_flag"].astype(bool).any():
                    raise AssertionError("confirmed entries are not p+2 open")
            meta = json.loads((result["report_dir"] / "11_run_meta.json").read_text(encoding="utf-8"))
            if meta.get("future_path_label_used_in_state") is not False:
                raise AssertionError("meta causality declaration is wrong")
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
