#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Directional Impulse Continuation: post-entry path anatomy (round 07).

Research question
-----------------
Before proposing another entry filter or exit rule, describe the actual path
of the Round-01 basic impulse event, separately for LONG and SHORT:

1. When is favorable excursion created during the first 60 minutes?
2. When does the path reach its maximum favorable excursion?
3. After a causally observable close-profit milestone is reached, how quickly
   is that profit retained or surrendered?
4. Does the path hit a small favorable close milestone before the equal
   adverse milestone?
5. Are LONG and SHORT path shapes materially different across nearby impulse
   windows and thresholds?

This is a descriptive event study, not a strategy backtest. It does not add a
trend/session/activity/order-flow/range-bar/footprint filter, and it does not
optimize TP, SL, trailing stop, or position sizing.

Causality
---------
- The impulse signal is confirmed only after bar t closes.
- The simulated reference entry is bar t+1 open.
- All path measurements use bars t+1 onward.
- Close-based activation is observable only after that minute closes.
- The study reports what happens after activation; it does not pretend that an
  exit occurred at the activation bar open or at an ex-post path peak.
- Any signal, entry, or 60m path touching a synthetic gap bar is excluded.

Performance design
------------------
- Local OKX 1m trade bars only; build_missing=False.
- One data load and one feature build per impulse window.
- The 60m path is built in bounded vectorized chunks into temporary memmaps.
- Nested thresholds, raw/deduplicated sets, years, and all path tables reuse
  the same path arrays.
- No iterrows scan over market bars and no per-event Python path loop.
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


def _load_round04_module():
    path = Path(__file__).resolve().with_name("04_impulse_first_passage_path_study.py")
    spec = importlib.util.spec_from_file_location("directional_impulse_round04_for_r07", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared round-04 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r04 = _load_round04_module()
r02 = r04.r02
r01 = r04.r01

SCRIPT_NAME = "07_impulse_path_anatomy_study"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION_R07"
EDGE_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION"
TITLE = "ETH Directional Impulse Continuation - Post-Entry Path Anatomy"
DEFAULT_OUT_DIR = (
    "data/reports/research/momentum/directional_impulse_continuation/"
    "07_impulse_path_anatomy_study"
)

DEFAULT_WINDOWS = (1, 3, 5, 10, 15)
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
DEFAULT_ACTIVATION_BPS = (10, 15, 25, 50)
DEFAULT_GIVEBACK_BPS = (5, 10, 15, 25)
DEFAULT_POST_ACTIVATION_LAGS = (1, 2, 3, 5, 10, 15)
DEFAULT_MAX_PATH_MINUTES = 60
KEY_MINUTES = (1, 3, 5, 10, 15, 30, 60)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Descriptive ETH impulse post-entry path anatomy.",
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
    p.add_argument("--activation-bps", default=",".join(map(str, DEFAULT_ACTIVATION_BPS)))
    p.add_argument("--giveback-bps", default=",".join(map(str, DEFAULT_GIVEBACK_BPS)))
    p.add_argument(
        "--post-activation-lags",
        default=",".join(map(str, DEFAULT_POST_ACTIVATION_LAGS)),
    )
    p.add_argument("--max-path-minutes", type=int, default=DEFAULT_MAX_PATH_MINUTES)
    p.add_argument("--vol-lookback-bars", type=int, default=1440)
    p.add_argument("--vol-min-periods", type=int, default=720)
    p.add_argument("--path-chunk-size", type=int, default=5000)
    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage", type=float, default=0.00020)
    p.add_argument("--exit-slippage", type=float, default=0.00020)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument(
        "--skip-events-csv",
        action="store_true",
        help="Development-only. Production stores minimum-threshold deduplicated path descriptors.",
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _parse_positive_ints(raw: str, *, name: str) -> tuple[int, ...]:
    values = tuple(sorted(dict.fromkeys(int(x.strip()) for x in str(raw).split(",") if x.strip())))
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"{name} must contain positive integers")
    return values


def _threshold_tag(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _bps_tag(value: int) -> str:
    return f"{int(value)}bps"


def _first_true_minute(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    any_hit = mask.any(axis=1)
    first = np.argmax(mask, axis=1).astype(np.int16) + np.int16(1)
    first[~any_hit] = np.int16(0)
    return first


def _safe_quantile(values: np.ndarray, q: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.quantile(x, q)) if x.size else float("nan")


def _safe_mean(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def _safe_median(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _safe_rate(mask: np.ndarray) -> float:
    x = np.asarray(mask, dtype=bool)
    return float(np.mean(x)) if x.size else float("nan")


def _build_path_memmaps(
    bars: pd.DataFrame,
    positions: np.ndarray,
    *,
    side: int,
    max_path: int,
    chunk_size: int,
    tmp_dir: Path,
    label: str,
    progress_enabled: bool,
) -> dict[str, Any]:
    """Build bounded-memory close/MFE/MAE paths and compact descriptors."""
    if int(chunk_size) <= 0:
        raise ValueError("path-chunk-size must be positive")
    m = int(len(positions))
    h = int(max_path)
    shape = (m, h)
    close_path = np.memmap(tmp_dir / "close_path.dat", mode="w+", dtype="float32", shape=shape)
    running_mfe = np.memmap(tmp_dir / "running_mfe.dat", mode="w+", dtype="float32", shape=shape)
    running_mae = np.memmap(tmp_dir / "running_mae.dat", mode="w+", dtype="float32", shape=shape)

    open_arr = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=np.float64)
    close_arr = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=np.float64)
    high_arr = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=np.float64)
    low_arr = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=np.float64)
    offsets = np.arange(1, h + 1, dtype=np.int64)
    total_chunks = max(1, int(math.ceil(max(1, m) / int(chunk_size))))

    peak_mfe_minute = np.zeros(m, dtype=np.int16)
    close_peak_minute = np.zeros(m, dtype=np.int16)
    close_peak_return = np.full(m, np.nan, dtype=np.float32)

    if m:
        with ProgressReporter(label=label, total=total_chunks, every=1, enabled=progress_enabled) as progress:
            done = 0
            for start in range(0, m, int(chunk_size)):
                end = min(m, start + int(chunk_size))
                pos = positions[start:end].astype(np.int64, copy=False)
                entry = open_arr[pos + 1].astype(np.float32, copy=False)
                gather = pos[:, None] + offsets[None, :]
                path_close = close_arr[gather].astype(np.float32, copy=False)
                path_high = high_arr[gather].astype(np.float32, copy=False)
                path_low = low_arr[gather].astype(np.float32, copy=False)

                close_ret = np.float32(float(side)) * (path_close / entry[:, None] - np.float32(1.0))
                if int(side) == 1:
                    favorable = path_high / entry[:, None] - np.float32(1.0)
                    adverse_signed = path_low / entry[:, None] - np.float32(1.0)
                else:
                    favorable = np.float32(1.0) - path_low / entry[:, None]
                    adverse_signed = np.float32(1.0) - path_high / entry[:, None]

                close_path[start:end, :] = close_ret
                running_mfe[start:end, :] = np.maximum.accumulate(favorable, axis=1)
                running_mae[start:end, :] = np.minimum.accumulate(adverse_signed, axis=1)
                peak_mfe_minute[start:end] = np.argmax(favorable, axis=1).astype(np.int16) + np.int16(1)
                cp = np.argmax(close_ret, axis=1).astype(np.int16)
                close_peak_minute[start:end] = cp + np.int16(1)
                close_peak_return[start:end] = close_ret[np.arange(end - start), cp]
                done += 1
                progress.update(done)

    close_path.flush()
    running_mfe.flush()
    running_mae.flush()
    return {
        "close_path": close_path,
        "running_mfe": running_mfe,
        "running_mae": running_mae,
        "peak_mfe_minute": peak_mfe_minute,
        "close_peak_minute": close_peak_minute,
        "close_peak_return": close_peak_return,
    }


def _path_descriptor_arrays(
    close_path: np.ndarray,
    running_mfe: np.ndarray,
    running_mae: np.ndarray,
    *,
    activation_bps: tuple[int, ...],
    giveback_bps: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Compact event descriptors derived from the already-built path arrays."""
    m, max_path = close_path.shape
    out: dict[str, np.ndarray] = {
        "final_return_60m": np.asarray(close_path[:, -1], dtype=np.float32),
        "mfe_60m": np.asarray(running_mfe[:, -1], dtype=np.float32),
        "mae_60m": np.asarray(running_mae[:, -1], dtype=np.float32),
    }
    close_np = np.asarray(close_path)

    for bps in activation_bps:
        rate = np.float32(int(bps) / 10_000.0)
        tag = _bps_tag(bps)
        eps = np.float32(1e-7)
        favorable_first = _first_true_minute(close_np >= rate - eps)
        adverse_first = _first_true_minute(close_np <= -rate + eps)
        out[f"close_favorable_first_{tag}_min"] = favorable_first
        out[f"close_adverse_first_{tag}_min"] = adverse_first

        for giveback in giveback_bps:
            gb_rate = np.float32(int(giveback) / 10_000.0)
            gb_tag = _bps_tag(giveback)
            first_giveback = np.zeros(m, dtype=np.int16)
            first_breakeven = np.zeros(m, dtype=np.int16)
            for start in range(0, m, 5000):
                end = min(m, start + 5000)
                chunk = close_np[start:end]
                act = favorable_first[start:end].astype(np.int64)
                rows = np.arange(end - start)
                valid = act > 0
                if not valid.any():
                    continue
                activation_idx = np.maximum(act - 1, 0)
                after_mask = np.arange(max_path)[None, :] >= activation_idx[:, None]
                running_best = np.maximum.accumulate(
                    np.where(after_mask, chunk, -np.inf), axis=1
                )
                drawdown = running_best - chunk
                gb_mask = valid[:, None] & after_mask & (drawdown >= gb_rate - np.float32(1e-7))
                gb_abs = _first_true_minute(gb_mask)
                gb_rel = np.where(gb_abs > 0, gb_abs - act + 1, 0)
                first_giveback[start:end] = gb_rel.astype(np.int16)

                be_mask = valid[:, None] & after_mask & (chunk <= 0.0)
                be_abs = _first_true_minute(be_mask)
                be_rel = np.where(be_abs > 0, be_abs - act + 1, 0)
                first_breakeven[start:end] = be_rel.astype(np.int16)
            out[f"giveback_{gb_tag}_after_{tag}_min"] = first_giveback
            if int(giveback) == min(giveback_bps):
                out[f"breakeven_after_{tag}_min"] = first_breakeven
    return out


def _event_count_rows(
    *,
    direction: str,
    window: int,
    thresholds: tuple[float, ...],
    all_min_positions: np.ndarray,
    eligible_positions: np.ndarray,
    directed_norm: np.ndarray,
    masks: dict[str, Any],
    study_months: int,
    n: int,
) -> tuple[list[dict[str, Any]], dict[float, np.ndarray], dict[float, np.ndarray]]:
    rows: list[dict[str, Any]] = []
    threshold_masks: dict[float, np.ndarray] = {}
    dedup_masks: dict[float, np.ndarray] = {}
    for threshold in thresholds:
        all_t = all_min_positions[directed_norm[all_min_positions] >= float(threshold)]
        dedup_t = r01._deduplicate_positions(all_t, int(window))
        dedup_axis = np.zeros(n, dtype=bool)
        dedup_axis[all_t] = dedup_t
        event_threshold_mask = directed_norm[eligible_positions] >= float(threshold)
        event_dedup_mask = event_threshold_mask & dedup_axis[eligible_positions]
        threshold_masks[float(threshold)] = event_threshold_mask
        dedup_masks[float(threshold)] = event_dedup_mask

        research_t = masks["research_mask"][all_t] & masks["causal_next_bar"][all_t]
        eligible_t = masks["eligible"][all_t]
        raw_count = int(eligible_t.sum())
        dedup_count = int((eligible_t & dedup_t).sum())
        overlap_ratio = 1.0 - dedup_count / raw_count if raw_count else np.nan
        for event_set, count in (("raw", raw_count), ("deduplicated", dedup_count)):
            rows.append(
                {
                    "direction": direction,
                    "impulse_window": int(window),
                    "threshold": float(threshold),
                    "event_set": event_set,
                    "raw_detected_count_full_loaded_axis": int(len(all_t)),
                    "raw_detected_count_research_window_before_full_path_check": int(research_t.sum()),
                    "events": int(count),
                    "events_per_month": float(count / max(1, study_months)),
                    "overlap_ratio": overlap_ratio,
                    "cooldown_bars": int(window),
                }
            )
    return rows, threshold_masks, dedup_masks


def _minute_profile_rows(
    *,
    direction: str,
    window: int,
    threshold: float,
    event_set: str,
    indices: np.ndarray,
    close_path: np.ndarray,
    running_mfe: np.ndarray,
    running_mae: np.ndarray,
    fee_only_cost: float,
    normal_cost: float,
) -> list[dict[str, Any]]:
    if indices.size == 0:
        return []
    close = np.asarray(close_path[indices, :], dtype=np.float32)
    mfe = np.asarray(running_mfe[indices, :], dtype=np.float32)
    mae = np.asarray(running_mae[indices, :], dtype=np.float32)
    close_best = np.maximum.accumulate(close, axis=1)
    giveback = close_best - close
    q = np.quantile(close, [0.05, 0.25, 0.50, 0.75, 0.95], axis=0)
    rows: list[dict[str, Any]] = []
    for minute in range(1, close.shape[1] + 1):
        j = minute - 1
        rows.append(
            {
                "direction": direction,
                "impulse_window": int(window),
                "threshold": float(threshold),
                "event_set": event_set,
                "minute": int(minute),
                "events": int(indices.size),
                "mean_close_gross": float(np.mean(close[:, j])),
                "median_close_gross": float(q[2, j]),
                "p05_close_gross": float(q[0, j]),
                "p25_close_gross": float(q[1, j]),
                "p75_close_gross": float(q[3, j]),
                "p95_close_gross": float(q[4, j]),
                "positive_close_rate": float(np.mean(close[:, j] > 0.0)),
                "above_fee_only_rate": float(np.mean(close[:, j] > float(fee_only_cost))),
                "above_normal_cost_rate": float(np.mean(close[:, j] > float(normal_cost))),
                "mean_running_mfe": float(np.mean(mfe[:, j])),
                "median_running_mfe": float(np.median(mfe[:, j])),
                "mean_running_mae": float(np.mean(mae[:, j])),
                "median_running_mae": float(np.median(mae[:, j])),
                "mean_close_giveback_from_best": float(np.mean(giveback[:, j])),
                "median_close_giveback_from_best": float(np.median(giveback[:, j])),
            }
        )
    return rows


def _peak_timing_row(
    *,
    direction: str,
    window: int,
    threshold: float,
    event_set: str,
    indices: np.ndarray,
    path: dict[str, Any],
) -> dict[str, Any]:
    peak_mfe_min = path["peak_mfe_minute"][indices]
    close_peak_min = path["close_peak_minute"][indices]
    close_peak_return = path["close_peak_return"][indices]
    final_return = np.asarray(path["close_path"][indices, -1], dtype=float)
    mfe = np.asarray(path["running_mfe"][indices, -1], dtype=float)
    mae = np.asarray(path["running_mae"][indices, -1], dtype=float)
    row: dict[str, Any] = {
        "direction": direction,
        "impulse_window": int(window),
        "threshold": float(threshold),
        "event_set": event_set,
        "events": int(indices.size),
        "mean_mfe_60m": _safe_mean(mfe),
        "median_mfe_60m": _safe_median(mfe),
        "mean_mae_60m": _safe_mean(mae),
        "median_mae_60m": _safe_median(mae),
        "mean_final_return_60m": _safe_mean(final_return),
        "median_final_return_60m": _safe_median(final_return),
        "mean_close_peak_return": _safe_mean(close_peak_return),
        "median_close_peak_return": _safe_median(close_peak_return),
        "mean_mfe_giveback_to_60m_close": _safe_mean(mfe - final_return),
        "median_mfe_giveback_to_60m_close": _safe_median(mfe - final_return),
        "top_1_mfe_contribution": r01._top_positive_share(mfe, 1),
        "top_5_mfe_contribution": r01._top_positive_share(mfe, 5),
        "mean_close_peak_giveback_to_60m_close": _safe_mean(close_peak_return - final_return),
        "median_peak_mfe_minute": _safe_median(peak_mfe_min),
        "p25_peak_mfe_minute": _safe_quantile(peak_mfe_min, 0.25),
        "p75_peak_mfe_minute": _safe_quantile(peak_mfe_min, 0.75),
        "median_close_peak_minute": _safe_median(close_peak_min),
    }
    for minute in KEY_MINUTES:
        row[f"mfe_peak_by_{minute}m_rate"] = _safe_rate(peak_mfe_min <= int(minute))
        row[f"close_peak_by_{minute}m_rate"] = _safe_rate(close_peak_min <= int(minute))
    return row


def _activation_rows(
    *,
    direction: str,
    window: int,
    threshold: float,
    event_set: str,
    indices: np.ndarray,
    descriptors: dict[str, np.ndarray],
    activation_bps: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bps in activation_bps:
        tag = _bps_tag(bps)
        fav = descriptors[f"close_favorable_first_{tag}_min"][indices]
        adv = descriptors[f"close_adverse_first_{tag}_min"][indices]
        fav_hit = fav > 0
        adv_hit = adv > 0
        favorable_first = fav_hit & (~adv_hit | (fav < adv))
        adverse_first = adv_hit & (~fav_hit | (adv < fav))
        neither = ~fav_hit & ~adv_hit
        same_minute = fav_hit & adv_hit & (fav == adv)
        row: dict[str, Any] = {
            "direction": direction,
            "impulse_window": int(window),
            "threshold": float(threshold),
            "event_set": event_set,
            "activation_bps": int(bps),
            "events": int(indices.size),
            "favorable_close_hit_rate_60m": _safe_rate(fav_hit),
            "adverse_close_hit_rate_60m": _safe_rate(adv_hit),
            "favorable_first_rate": _safe_rate(favorable_first),
            "adverse_first_rate": _safe_rate(adverse_first),
            "same_minute_rate": _safe_rate(same_minute),
            "neither_close_hit_rate": _safe_rate(neither),
            "median_favorable_close_hit_minute": _safe_median(fav[fav_hit]),
            "p25_favorable_close_hit_minute": _safe_quantile(fav[fav_hit], 0.25),
            "p75_favorable_close_hit_minute": _safe_quantile(fav[fav_hit], 0.75),
            "median_adverse_close_hit_minute": _safe_median(adv[adv_hit]),
        }
        for minute in KEY_MINUTES:
            row[f"favorable_hit_by_{minute}m_rate"] = _safe_rate(fav_hit & (fav <= int(minute)))
            row[f"adverse_hit_by_{minute}m_rate"] = _safe_rate(adv_hit & (adv <= int(minute)))
        rows.append(row)
    return rows


def _post_activation_rows(
    *,
    direction: str,
    window: int,
    threshold: float,
    event_set: str,
    indices: np.ndarray,
    close_path: np.ndarray,
    descriptors: dict[str, np.ndarray],
    activation_bps: tuple[int, ...],
    lags: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    max_path = close_path.shape[1]
    selected_path = np.asarray(close_path[indices, :], dtype=np.float32) if indices.size else np.empty((0, max_path), dtype=np.float32)
    minute_axis = np.arange(max_path, dtype=np.int16)[None, :]
    for bps in activation_bps:
        rate = float(bps) / 10_000.0
        tag = _bps_tag(bps)
        first_all = descriptors[f"close_favorable_first_{tag}_min"]
        first = first_all[indices].astype(np.int64)
        for lag in lags:
            valid_local = (first > 0) & (first + int(lag) <= max_path)
            local_rows = np.flatnonzero(valid_local)
            if local_rows.size == 0:
                rows.append(
                    {
                        "direction": direction,
                        "impulse_window": int(window),
                        "threshold": float(threshold),
                        "event_set": event_set,
                        "activation_bps": int(bps),
                        "lag_after_activation_min": int(lag),
                        "events_reaching_activation_with_followup": 0,
                    }
                )
                continue
            activation_col = first[local_rows] - 1
            future_col = activation_col + int(lag)
            path = selected_path[local_rows, :]
            activation_ret = path[np.arange(local_rows.size), activation_col]
            future_ret = path[np.arange(local_rows.size), future_col]
            segment_mask = (minute_axis >= activation_col[:, None]) & (minute_axis <= future_col[:, None])
            segment_peak = np.max(np.where(segment_mask, path, -np.inf), axis=1)
            drawdown = segment_peak - future_ret
            rows.append(
                {
                    "direction": direction,
                    "impulse_window": int(window),
                    "threshold": float(threshold),
                    "event_set": event_set,
                    "activation_bps": int(bps),
                    "lag_after_activation_min": int(lag),
                    "events_reaching_activation_with_followup": int(local_rows.size),
                    "mean_return_at_activation_close": _safe_mean(activation_ret),
                    "mean_return_after_lag": _safe_mean(future_ret),
                    "median_return_after_lag": _safe_median(future_ret),
                    "mean_incremental_return_after_activation": _safe_mean(future_ret - activation_ret),
                    "median_incremental_return_after_activation": _safe_median(future_ret - activation_ret),
                    "still_above_activation_rate": _safe_rate(future_ret >= rate),
                    "still_above_half_activation_rate": _safe_rate(future_ret >= rate * 0.5),
                    "back_below_breakeven_rate": _safe_rate(future_ret <= 0.0),
                    "mean_close_drawdown_from_post_activation_peak": _safe_mean(drawdown),
                    "median_close_drawdown_from_post_activation_peak": _safe_median(drawdown),
                }
            )
    return rows


def _giveback_rows(
    *,
    direction: str,
    window: int,
    threshold: float,
    event_set: str,
    indices: np.ndarray,
    descriptors: dict[str, np.ndarray],
    activation_bps: tuple[int, ...],
    giveback_bps: tuple[int, ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for activation in activation_bps:
        atag = _bps_tag(activation)
        activated = descriptors[f"close_favorable_first_{atag}_min"][indices] > 0
        activated_count = int(activated.sum())
        for giveback in giveback_bps:
            gtag = _bps_tag(giveback)
            values = descriptors[f"giveback_{gtag}_after_{atag}_min"][indices]
            hit = activated & (values > 0)
            row: dict[str, Any] = {
                "direction": direction,
                "impulse_window": int(window),
                "threshold": float(threshold),
                "event_set": event_set,
                "activation_bps": int(activation),
                "giveback_bps": int(giveback),
                "activated_events": activated_count,
                "giveback_hit_rate_by_60m": float(hit.sum() / activated_count) if activated_count else np.nan,
                "median_minutes_from_activation_to_giveback": _safe_median(values[hit]),
                "p25_minutes_from_activation_to_giveback": _safe_quantile(values[hit], 0.25),
                "p75_minutes_from_activation_to_giveback": _safe_quantile(values[hit], 0.75),
            }
            for lag in (1, 2, 3, 5, 10, 15):
                row[f"giveback_within_{lag}m_rate"] = (
                    float(np.sum(hit & (values <= lag)) / activated_count) if activated_count else np.nan
                )
            rows.append(row)
    return rows


def _yearly_rows(
    *,
    direction: str,
    window: int,
    threshold: float,
    event_set: str,
    indices: np.ndarray,
    years: np.ndarray,
    path: dict[str, Any],
    descriptors: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if indices.size == 0:
        return rows
    selected_years = years[indices]
    for year in np.unique(selected_years):
        idx = indices[selected_years == year]
        key_cols = np.asarray([minute - 1 for minute in KEY_MINUTES], dtype=np.int64)
        close = np.asarray(path["close_path"][idx[:, None], key_cols[None, :]], dtype=float)
        mfe = np.asarray(path["running_mfe"][idx, -1], dtype=float)
        mae = np.asarray(path["running_mae"][idx, -1], dtype=float)
        fav15 = descriptors["close_favorable_first_15bps_min"][idx]
        row: dict[str, Any] = {
            "direction": direction,
            "impulse_window": int(window),
            "threshold": float(threshold),
            "event_set": event_set,
            "year": int(year),
            "events": int(idx.size),
            "mean_mfe_60m": _safe_mean(mfe),
            "median_mfe_60m": _safe_median(mfe),
            "mean_mae_60m": _safe_mean(mae),
            "median_peak_mfe_minute": _safe_median(path["peak_mfe_minute"][idx]),
            "close_15bps_hit_rate_60m": _safe_rate(fav15 > 0),
            "close_15bps_hit_rate_5m": _safe_rate((fav15 > 0) & (fav15 <= 5)),
        }
        for minute in KEY_MINUTES:
            col = KEY_MINUTES.index(minute)
            row[f"mean_close_return_{minute}m"] = _safe_mean(close[:, col])
            row[f"median_close_return_{minute}m"] = _safe_median(close[:, col])
        rows.append(row)
    return rows



def _monthly_rows(
    *,
    direction: str,
    window: int,
    threshold: float,
    event_set: str,
    indices: np.ndarray,
    months: np.ndarray,
    path: dict[str, Any],
    descriptors: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if indices.size == 0:
        return rows
    selected_months = months[indices]
    key_cols = np.asarray([minute - 1 for minute in KEY_MINUTES], dtype=np.int64)
    for month in np.unique(selected_months):
        idx = indices[selected_months == month]
        close = np.asarray(path["close_path"][idx[:, None], key_cols[None, :]], dtype=float)
        mfe = np.asarray(path["running_mfe"][idx, -1], dtype=float)
        mae = np.asarray(path["running_mae"][idx, -1], dtype=float)
        fav15 = descriptors["close_favorable_first_15bps_min"][idx]
        row: dict[str, Any] = {
            "direction": direction,
            "impulse_window": int(window),
            "threshold": float(threshold),
            "event_set": event_set,
            "month": str(month),
            "events": int(idx.size),
            "mean_mfe_60m": _safe_mean(mfe),
            "median_mfe_60m": _safe_median(mfe),
            "mean_mae_60m": _safe_mean(mae),
            "median_peak_mfe_minute": _safe_median(path["peak_mfe_minute"][idx]),
            "close_15bps_hit_rate_60m": _safe_rate(fav15 > 0),
            "close_15bps_hit_rate_5m": _safe_rate((fav15 > 0) & (fav15 <= 5)),
        }
        for col, minute in enumerate(KEY_MINUTES):
            row[f"mean_close_return_{minute}m"] = _safe_mean(close[:, col])
            row[f"median_close_return_{minute}m"] = _safe_median(close[:, col])
        rows.append(row)
    return rows

def _compact_event_frame(
    *,
    bars: pd.DataFrame,
    positions: np.ndarray,
    selected: np.ndarray,
    direction: str,
    side: int,
    window: int,
    features: Any,
    thresholds: tuple[float, ...],
    threshold_masks: dict[float, np.ndarray],
    dedup_masks: dict[float, np.ndarray],
    path: dict[str, Any],
    descriptors: dict[str, np.ndarray],
    event_id_start: int,
) -> pd.DataFrame:
    idx = np.flatnonzero(selected)
    pos = positions[idx]
    signal_start = bars.index[pos]
    signal_end = signal_start + pd.Timedelta(minutes=1)
    entry_pos = pos + 1
    entry_time = bars.index[entry_pos]
    entry_price = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)[entry_pos]
    expected_entry_price = pd.to_numeric(bars["open"], errors="coerce").reindex(signal_end).to_numpy(dtype=float)
    data: dict[str, Any] = {
        "event_id": np.arange(event_id_start, event_id_start + len(idx), dtype=np.int64),
        "direction": direction,
        "side": int(side),
        "impulse_window": int(window),
        "signal_time": signal_end,
        "signal_bar_start": signal_start,
        "signal_bar_end": signal_end,
        "entry_time": entry_time,
        "expected_entry_time": signal_end,
        "entry_price": entry_price,
        "expected_entry_price": expected_entry_price,
        "entry_not_next_open_flag": entry_time != signal_end,
        "entry_price_mismatch_flag": ~np.isclose(entry_price, expected_entry_price, rtol=0.0, atol=1e-12),
        "synthetic_bar_dependency_flag": np.zeros(len(idx), dtype=bool),
        "event_storage_scope": "minimum_threshold_deduplicated",
        "impulse_return": features.impulse_return[pos],
        "normalized_impulse": features.normalized_impulse[pos],
        "directional_efficiency": features.directional_efficiency[pos],
        "window_range_bps": features.window_range_bps[pos],
        "window_realized_vol": features.window_realized_vol[pos],
        "pre_impulse_volatility": features.pre_impulse_volatility[pos],
        "pre_impulse_return": features.pre_impulse_return[pos],
        "peak_mfe_minute": path["peak_mfe_minute"][idx],
        "close_peak_minute": path["close_peak_minute"][idx],
        "close_peak_return": path["close_peak_return"][idx],
    }
    for threshold in thresholds:
        tag = _threshold_tag(threshold)
        data[f"event_{tag}_flag"] = threshold_masks[float(threshold)][idx]
        data[f"deduplicated_{tag}_flag"] = dedup_masks[float(threshold)][idx]
    for key, values in descriptors.items():
        data[key] = values[idx]
    return pd.DataFrame(data)


def _signal_audit(events: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "event_id",
        "direction",
        "impulse_window",
        "signal_bar_start",
        "signal_bar_end",
        "signal_time",
        "entry_time",
        "expected_entry_time",
        "entry_price",
        "expected_entry_price",
        "entry_not_next_open_flag",
        "entry_price_mismatch_flag",
        "synthetic_bar_dependency_flag",
    ]
    audit = events[cols].copy() if not events.empty else pd.DataFrame(columns=cols)
    if audit.empty:
        audit["lookahead_flag"] = pd.Series(dtype=bool)
        return audit
    audit["lookahead_flag"] = (
        audit["entry_not_next_open_flag"].astype(bool)
        | audit["entry_price_mismatch_flag"].astype(bool)
        | audit["synthetic_bar_dependency_flag"].astype(bool)
    )
    return audit


def _threshold_plateau(peak: pd.DataFrame, activation: pd.DataFrame) -> pd.DataFrame:
    if peak.empty or activation.empty:
        return pd.DataFrame()
    act15 = activation[activation["activation_bps"] == 15].copy()
    cols = [
        "direction",
        "impulse_window",
        "threshold",
        "event_set",
        "events",
        "mean_mfe_60m",
        "median_mfe_60m",
        "mean_mae_60m",
        "mean_final_return_60m",
        "median_final_return_60m",
        "median_peak_mfe_minute",
        "mean_mfe_giveback_to_60m_close",
    ]
    out = peak[cols].merge(
        act15[
            [
                "direction",
                "impulse_window",
                "threshold",
                "event_set",
                "favorable_close_hit_rate_60m",
                "favorable_hit_by_5m_rate",
                "favorable_first_rate",
                "adverse_first_rate",
            ]
        ],
        on=["direction", "impulse_window", "threshold", "event_set"],
        how="left",
    )
    return out.sort_values(["direction", "event_set", "impulse_window", "threshold"])


def _long_short_comparison(plateau: pd.DataFrame) -> pd.DataFrame:
    if plateau.empty:
        return pd.DataFrame()
    keys = ["impulse_window", "threshold", "event_set"]
    metrics = [
        "events",
        "mean_mfe_60m",
        "mean_mae_60m",
        "mean_final_return_60m",
        "median_peak_mfe_minute",
        "mean_mfe_giveback_to_60m_close",
        "favorable_close_hit_rate_60m",
        "favorable_hit_by_5m_rate",
        "favorable_first_rate",
        "adverse_first_rate",
    ]
    long = plateau[plateau["direction"] == "LONG"][keys + metrics].copy()
    short = plateau[plateau["direction"] == "SHORT"][keys + metrics].copy()
    long = long.rename(columns={c: f"long_{c}" for c in metrics})
    short = short.rename(columns={c: f"short_{c}" for c in metrics})
    out = long.merge(short, on=keys, how="outer")
    for metric in metrics[1:]:
        out[f"long_minus_short_{metric}"] = out[f"long_{metric}"] - out[f"short_{metric}"]
    return out.sort_values(keys)


def _build_brief(
    peak: pd.DataFrame,
    activation: pd.DataFrame,
    post_activation: pd.DataFrame,
    giveback: pd.DataFrame,
    meta: dict[str, Any],
) -> str:
    lines = [
        "# Round 07 research brief — post-entry path anatomy",
        "",
        "## Boundary",
        "",
        "This round is descriptive. It does not claim a tradable edge and does not choose an exit rule.",
        "It measures when profit appears, peaks, and gives back after the basic next-open impulse entry.",
        "LONG and SHORT are reported separately.",
        "",
        "## Data and timing",
        "",
        f"- Rows: {meta['input_rows']:,}",
        f"- Research period: {meta['research_start']} to {meta['research_end']}",
        f"- Maximum path: {meta['max_path_minutes']} minutes",
        f"- Activation milestones: {meta['activation_bps']} bps",
        f"- Giveback milestones: {meta['giveback_bps']} bps",
        "- Signal: fully closed 1m bar; reference entry: next 1m open.",
        "- Close-based activation is only observable after that close; no ex-post peak is treated as executable.",
        "",
        "## What to inspect",
        "",
        "1. `02_minute_path_profile.csv`: minute-by-minute close return, running MFE/MAE, and giveback.",
        "2. `03_peak_timing_summary.csv`: when MFE and best close normally occur.",
        "3. `04_activation_summary.csv`: how often 10/15/25/50bps close profit is reached and whether it precedes equal adverse movement.",
        "4. `05_post_activation_decay.csv`: how much profit remains 1/2/3/5/10/15 minutes after a causal close milestone.",
        "5. `06_giveback_timing.csv`: how quickly fixed close drawdowns occur after activation.",
        "6. `07_yearly_key_path.csv`: whether path shape is consistent by year.",
        "7. `10_long_short_comparison.csv`: direct LONG versus SHORT path differences.",
        "",
    ]
    dedup_peak = peak[peak["event_set"] == "deduplicated"] if not peak.empty else peak
    dedup_act = activation[(activation["event_set"] == "deduplicated") & (activation["activation_bps"] == 15)] if not activation.empty else activation
    if not dedup_peak.empty and not dedup_act.empty:
        for direction in ("LONG", "SHORT"):
            p = dedup_peak[dedup_peak["direction"] == direction]
            a = dedup_act[dedup_act["direction"] == direction]
            if p.empty or a.empty:
                continue
            merged = p.merge(
                a[["direction", "impulse_window", "threshold", "favorable_hit_by_5m_rate", "favorable_close_hit_rate_60m"]],
                on=["direction", "impulse_window", "threshold"],
                how="left",
            )
            representative = merged.sort_values(
                ["favorable_hit_by_5m_rate", "mean_mfe_60m"], ascending=False
            ).iloc[0]
            lines.extend(
                [
                    f"## {direction} descriptive reference",
                    "",
                    f"- Window / threshold: {int(representative['impulse_window'])}m / {float(representative['threshold']):.1f}",
                    f"- Events: {int(representative['events']):,}",
                    f"- 15bps close hit by 5m: {float(representative['favorable_hit_by_5m_rate']):.2%}",
                    f"- 15bps close hit by 60m: {float(representative['favorable_close_hit_rate_60m']):.2%}",
                    f"- Median MFE peak minute: {float(representative['median_peak_mfe_minute']):.2f}",
                    f"- Mean 60m MFE: {float(representative['mean_mfe_60m']):.4%}",
                    f"- Mean 60m close return: {float(representative['mean_final_return_60m']):.4%}",
                    f"- Mean MFE giveback to 60m close: {float(representative['mean_mfe_giveback_to_60m_close']):.4%}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Interpretation rule",
            "",
            "Do not select a strategy from the largest MFE row. The next research step should be chosen only after checking:",
            "",
            "- whether profit forms early enough to be observable and executable;",
            "- whether post-activation retention is stable across nearby windows/thresholds and years;",
            "- whether giveback leaves enough reaction time for 1m bars, or requires trade/range-bar data;",
            "- whether LONG and SHORT require separate path models.",
            "",
            "Status: `research_only_path_anatomy`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _update_log(log_path: Path, meta: dict[str, Any]) -> None:
    section = f"""

## Round 07 — Post-entry path anatomy

### Research question

Before proposing another filter or exit rule, how does favorable excursion form, peak, and give back during the first {meta['max_path_minutes']} minutes after the Round-01 next-open entry, separately for LONG and SHORT?

### Changed from Round 06

Round 06's same-window impulse-retention exit is removed. Round 07 does not test a strategy rule. It decomposes the raw path so the next upgrade is evidence-led rather than another ungrounded parameter family.

### Fixed descriptive design

```text
maximum path             = {meta['max_path_minutes']}m
close-profit milestones  = {meta['activation_bps']} bps
close-giveback milestones= {meta['giveback_bps']} bps
post-activation lags     = {meta['post_activation_lags']} minutes
```

Reported separately for every direction, impulse window, threshold, and raw/deduplicated event set:

- minute-by-minute close-return distribution;
- running MFE and MAE;
- MFE peak time and close-path peak time;
- first favorable/adverse close milestone order;
- retention after a causally observable close-profit milestone;
- time from activation to fixed close giveback;
- yearly path stability;
- LONG versus SHORT differences.

### Boundary

No new entry condition, environment filter, trailing stop, TP/SL, range bar, footprint, order-flow feature, position sizing, or portfolio rule is introduced. Ex-post MFE and peak time are descriptive only and cannot be treated as executable profit.

### Performance design

- One local trade-bar load and one impulse feature build per window.
- A bounded chunk kernel writes the 60m close/MFE/MAE paths to temporary memmaps.
- All thresholds, event sets, years, activations, and giveback tables reuse those arrays.
- No per-event Python market-path loop and no repeated full-data scan per variant.
- Synthetic-gap-dependent events are excluded.

### Status

Pending production run.
"""
    existing = log_path.read_text(encoding="utf-8") if log_path.exists() else "# ETH Directional Impulse Continuation research log\n"
    if "## Round 07 — Post-entry path anatomy" not in existing:
        log_path.write_text(existing.rstrip() + section, encoding="utf-8")


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    windows = r01._parse_int_csv(args.impulse_windows)
    thresholds = r01._parse_float_csv(args.thresholds)
    activation_bps = _parse_positive_ints(args.activation_bps, name="activation-bps")
    giveback_bps = _parse_positive_ints(args.giveback_bps, name="giveback-bps")
    post_lags = _parse_positive_ints(args.post_activation_lags, name="post-activation-lags")
    max_path = int(args.max_path_minutes)
    if max_path <= 0 or max_path > np.iinfo(np.int16).max:
        raise ValueError("max-path-minutes must be positive and fit int16")
    if max(KEY_MINUTES) > max_path:
        raise ValueError(f"max-path-minutes must be at least {max(KEY_MINUTES)}")
    if max(post_lags) >= max_path:
        raise ValueError("post-activation lags must be smaller than max path")
    if int(args.vol_min_periods) > int(args.vol_lookback_bars):
        raise ValueError("vol-min-periods cannot exceed vol-lookback-bars")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "11_events.csv"
    audit_path = out_dir / "12_signal_audit.csv"
    for path in (events_path, audit_path):
        path.unlink(missing_ok=True)

    validation = r01.validate_bars(bars, args)
    masks = r02._eligible_masks(bars, args, (max_path,))
    study_months = int(masks["study_months"])
    fee_only_cost = float(args.entry_fee_rate + args.exit_fee_rate)
    normal_cost = float(fee_only_cost + args.entry_slippage + args.exit_slippage)
    log_return, abs_price_change, historical_1m_vol = r01.build_base_volatility(
        bars, int(args.vol_lookback_bars), int(args.vol_min_periods)
    )

    event_count_rows: list[dict[str, Any]] = []
    minute_rows: list[dict[str, Any]] = []
    peak_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    post_activation_rows: list[dict[str, Any]] = []
    giveback_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    event_id_cursor = 1
    first_event_write = True
    first_audit_write = True
    minimum_threshold = min(thresholds)
    n = len(bars)
    unique_raw_rows = 0
    compact_rows_written = 0

    print("[feature build] base impulse features", flush=True)
    total_groups = len(windows) * 2
    with ProgressReporter(
        label="[path anatomy] direction/windows",
        total=total_groups,
        every=max(1, int(args.progress_every)),
        enabled=not args.no_progress,
    ) as outer:
        done = 0
        for window in windows:
            features = r01.build_window_features(
                bars, window, log_return, abs_price_change, historical_1m_vol
            )
            norm = features.normalized_impulse
            for direction, side in (("LONG", 1), ("SHORT", -1)):
                directed_norm = float(side) * norm
                all_min_positions = np.flatnonzero(
                    np.isfinite(directed_norm) & (directed_norm >= float(minimum_threshold))
                )
                eligible_positions = all_min_positions[masks["eligible"][all_min_positions]]
                unique_raw_rows += int(len(eligible_positions))
                rows, threshold_masks, dedup_masks = _event_count_rows(
                    direction=direction,
                    window=window,
                    thresholds=thresholds,
                    all_min_positions=all_min_positions,
                    eligible_positions=eligible_positions,
                    directed_norm=directed_norm,
                    masks=masks,
                    study_months=study_months,
                    n=n,
                )
                event_count_rows.extend(rows)

                if eligible_positions.size:
                    with tempfile.TemporaryDirectory(
                        prefix=f"dic_r07_{direction.lower()}_{window}m_"
                    ) as tmp:
                        path = _build_path_memmaps(
                            bars,
                            eligible_positions,
                            side=side,
                            max_path=max_path,
                            chunk_size=int(args.path_chunk_size),
                            tmp_dir=Path(tmp),
                            label=f"[path build] {direction} {int(window)}m chunks",
                            progress_enabled=not args.no_progress,
                        )
                        descriptors = _path_descriptor_arrays(
                            path["close_path"],
                            path["running_mfe"],
                            path["running_mae"],
                            activation_bps=activation_bps,
                            giveback_bps=giveback_bps,
                        )
                        signal_index = pd.to_datetime(bars.index[eligible_positions])
                        years = signal_index.year.to_numpy(dtype=int)
                        months = signal_index.to_period("M").astype(str).to_numpy()

                        for threshold in thresholds:
                            for event_set, mask in (
                                ("raw", threshold_masks[float(threshold)]),
                                ("deduplicated", dedup_masks[float(threshold)]),
                            ):
                                idx = np.flatnonzero(mask)
                                minute_rows.extend(
                                    _minute_profile_rows(
                                        direction=direction,
                                        window=window,
                                        threshold=threshold,
                                        event_set=event_set,
                                        indices=idx,
                                        close_path=path["close_path"],
                                        running_mfe=path["running_mfe"],
                                        running_mae=path["running_mae"],
                                        fee_only_cost=fee_only_cost,
                                        normal_cost=normal_cost,
                                    )
                                )
                                peak_rows.append(
                                    _peak_timing_row(
                                        direction=direction,
                                        window=window,
                                        threshold=threshold,
                                        event_set=event_set,
                                        indices=idx,
                                        path=path,
                                    )
                                )
                                activation_rows.extend(
                                    _activation_rows(
                                        direction=direction,
                                        window=window,
                                        threshold=threshold,
                                        event_set=event_set,
                                        indices=idx,
                                        descriptors=descriptors,
                                        activation_bps=activation_bps,
                                    )
                                )
                                post_activation_rows.extend(
                                    _post_activation_rows(
                                        direction=direction,
                                        window=window,
                                        threshold=threshold,
                                        event_set=event_set,
                                        indices=idx,
                                        close_path=path["close_path"],
                                        descriptors=descriptors,
                                        activation_bps=activation_bps,
                                        lags=post_lags,
                                    )
                                )
                                giveback_rows.extend(
                                    _giveback_rows(
                                        direction=direction,
                                        window=window,
                                        threshold=threshold,
                                        event_set=event_set,
                                        indices=idx,
                                        descriptors=descriptors,
                                        activation_bps=activation_bps,
                                        giveback_bps=giveback_bps,
                                    )
                                )
                                yearly_rows.extend(
                                    _yearly_rows(
                                        direction=direction,
                                        window=window,
                                        threshold=threshold,
                                        event_set=event_set,
                                        indices=idx,
                                        years=years,
                                        path=path,
                                        descriptors=descriptors,
                                    )
                                )
                                monthly_rows.extend(
                                    _monthly_rows(
                                        direction=direction,
                                        window=window,
                                        threshold=threshold,
                                        event_set=event_set,
                                        indices=idx,
                                        months=months,
                                        path=path,
                                        descriptors=descriptors,
                                    )
                                )

                        if not args.skip_events_csv:
                            selected = dedup_masks[float(minimum_threshold)]
                            compact = _compact_event_frame(
                                bars=bars,
                                positions=eligible_positions,
                                selected=selected,
                                direction=direction,
                                side=side,
                                window=window,
                                features=features,
                                thresholds=thresholds,
                                threshold_masks=threshold_masks,
                                dedup_masks=dedup_masks,
                                path=path,
                                descriptors=descriptors,
                                event_id_start=event_id_cursor,
                            )
                            event_id_cursor += len(compact)
                            compact_rows_written += int(len(compact))
                            r01._write_stream_csv(compact, events_path, first_write=first_event_write)
                            first_event_write = False
                            audit = _signal_audit(compact)
                            r01._write_stream_csv(audit, audit_path, first_write=first_audit_write)
                            first_audit_write = False
                            del compact, audit
                        for key in ("close_path", "running_mfe", "running_mae"):
                            path[key]._mmap.close()  # type: ignore[attr-defined]
                        del path, descriptors
                done += 1
                outer.update(done)
            del features

    event_counts = pd.DataFrame(event_count_rows)
    minute_profile = pd.DataFrame(minute_rows)
    peak_timing = pd.DataFrame(peak_rows)
    activation = pd.DataFrame(activation_rows)
    post_activation = pd.DataFrame(post_activation_rows)
    giveback = pd.DataFrame(giveback_rows)
    yearly = pd.DataFrame(yearly_rows)
    monthly = pd.DataFrame(monthly_rows)
    plateau = _threshold_plateau(peak_timing, activation)
    long_short = _long_short_comparison(plateau)

    if first_event_write:
        pd.DataFrame(columns=["event_id", "direction", "impulse_window"]).to_csv(events_path, index=False)
        pd.DataFrame(columns=["event_id", "lookahead_flag"]).to_csv(audit_path, index=False)

    meta = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "portfolio_plan": "ETH_NOVA_PORTFOLIO",
        "title": TITLE,
        "status": "research_only_path_anatomy",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "data_source": "src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader",
        "trade_bar_db_path": str(r01._trade_bar_db_path(args)),
        "local_cache_only": True,
        "build_missing": False,
        "ordinary_kline_download_enabled": False,
        "timezone_convention": "UTC+8 project convention; timestamps remain timezone-naive",
        "warmup_start_date": args.warmup_start_date,
        "research_start": args.start_date,
        "research_end": args.end_date,
        "impulse_windows": list(windows),
        "thresholds": list(thresholds),
        "max_path_minutes": max_path,
        "activation_bps": list(activation_bps),
        "giveback_bps": list(giveback_bps),
        "post_activation_lags": list(post_lags),
        "entry_policy": "closed signal bar; next 1m open reference entry",
        "activation_policy": "first fully closed minute whose direction-adjusted close return reaches milestone",
        "path_boundary": "descriptive only; no ex-post MFE/peak is treated as executable",
        "fee_only_cost": fee_only_cost,
        "normal_execution_cost": normal_cost,
        "deduplication": "threshold-specific same-direction stream; cooldown=impulse_window",
        "path_chunk_size": int(args.path_chunk_size),
        "path_storage": "temporary float32 memmaps per direction/window; deleted after summaries",
        "research_month_count": study_months,
        "input_rows": int(len(bars)),
        "source_observed_rows": int(bars["source_bar_observed_flag"].sum()),
        "synthetic_gap_bar_count": int((~bars["source_bar_observed_flag"].astype(bool)).sum()),
        "source_gap_segment_count": int(bars.attrs.get("gap_segment_count", 0)),
        "max_gap_minutes": int(bars.attrs.get("max_gap_minutes", 0)),
        "gap_handling": str(bars.attrs.get("gap_policy", "")),
        "unique_raw_event_rows_analyzed": int(unique_raw_rows),
        "compact_event_rows_written": int(compact_rows_written),
        "events_csv_skipped_for_development": bool(args.skip_events_csv),
        "validation": validation,
        "round06_retention_exit_stacked": False,
        "trend_filter_stacked": False,
        "order_flow_filter_stacked": False,
        "range_bar_filter_stacked": False,
        "footprint_filter_stacked": False,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "research_boundary": "one descriptive question: post-entry profit formation, peak timing and giveback",
    }
    brief = _build_brief(peak_timing, activation, post_activation, giveback, meta)

    artifacts = [
        (event_counts, out_dir / "01_event_counts.csv"),
        (minute_profile, out_dir / "02_minute_path_profile.csv"),
        (peak_timing, out_dir / "03_peak_timing_summary.csv"),
        (activation, out_dir / "04_activation_summary.csv"),
        (post_activation, out_dir / "05_post_activation_decay.csv"),
        (giveback, out_dir / "06_giveback_timing.csv"),
        (yearly, out_dir / "07_yearly_key_path.csv"),
        (monthly, out_dir / "08_monthly_key_path.csv"),
        (plateau, out_dir / "09_threshold_plateau.csv"),
        (long_short, out_dir / "10_long_short_comparison.csv"),
    ]
    print("[artifacts] writing path-anatomy report", flush=True)
    with ProgressReporter(
        label="[artifacts] tables",
        total=len(artifacts) + 3,
        every=1,
        enabled=not args.no_progress,
    ) as progress:
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
        print("[review pack] packaging path-anatomy artifacts", flush=True)
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report_dir={out_dir}", flush=True)
    return {
        "report_dir": out_dir,
        "events": events_path,
        "audit": audit_path,
        "review_pack": out_dir / "gpt_review_pack.zip",
    }


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] deterministic path-anatomy study", flush=True)
    index = pd.date_range("2023-01-01", periods=8, freq="1min")
    kernel = pd.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [100.0, 100.12, 100.30, 100.40, 100.35, 100.20, 100.05, 99.95],
            "low": [100.0, 99.98, 100.05, 100.20, 100.10, 99.98, 99.90, 99.80],
            "close": [100.0, 100.10, 100.25, 100.35, 100.20, 100.05, 99.95, 99.85],
            "volume": [1.0] * 8,
            "source_bar_observed_flag": [True] * 8,
        },
        index=index,
    )
    with tempfile.TemporaryDirectory(prefix="dic_r07_kernel_") as tmp:
        path = _build_path_memmaps(
            kernel,
            np.asarray([0], dtype=int),
            side=1,
            max_path=6,
            chunk_size=8,
            tmp_dir=Path(tmp),
            label="[self-test kernel]",
            progress_enabled=False,
        )
        desc = _path_descriptor_arrays(
            path["close_path"], path["running_mfe"], path["running_mae"],
            activation_bps=(10, 25), giveback_bps=(10,)
        )
        if int(desc["close_favorable_first_10bps_min"][0]) != 1:
            raise AssertionError("10bps close activation minute is wrong")
        if int(desc["close_favorable_first_25bps_min"][0]) != 2:
            raise AssertionError("25bps close activation minute is wrong")
        if int(path["close_peak_minute"][0]) != 3:
            raise AssertionError("close peak minute is wrong")
        if int(desc["giveback_10bps_after_25bps_min"][0]) != 3:
            raise AssertionError("giveback timing is wrong")
        for key in ("close_path", "running_mfe", "running_mae"):
            path[key]._mmap.close()  # type: ignore[attr-defined]

    raw = r01._synthetic_bars().drop(r01._synthetic_bars().index[3700:3707])
    bars = r01._regularize_trade_bar_axis(raw)
    log_path = Path(__file__).resolve().with_name("00_research_log.md")
    original_log = log_path.read_text(encoding="utf-8") if log_path.exists() else None
    try:
        with tempfile.TemporaryDirectory(prefix="dic_r07_") as tmp:
            args.out_dir = tmp
            args.warmup_start_date = "2022-12-20"
            args.start_date = "2022-12-23"
            args.end_date = "2022-12-24"
            args.vol_lookback_bars = 720
            args.vol_min_periods = 360
            args.impulse_windows = "1,3"
            args.thresholds = "1.0,1.5"
            args.activation_bps = "10,15,25"
            args.giveback_bps = "5,10"
            args.post_activation_lags = "1,2,3,5,10,15"
            args.max_path_minutes = 60
            args.path_chunk_size = 256
            args.skip_review_pack = True
            args.skip_events_csv = False
            args.no_progress = True
            result = run_research(bars, args)
            required = [
                "01_event_counts.csv",
                "02_minute_path_profile.csv",
                "03_peak_timing_summary.csv",
                "04_activation_summary.csv",
                "05_post_activation_decay.csv",
                "06_giveback_timing.csv",
                "07_yearly_key_path.csv",
                "08_monthly_key_path.csv",
                "09_threshold_plateau.csv",
                "10_long_short_comparison.csv",
                "11_events.csv",
                "12_signal_audit.csv",
                "13_run_meta.json",
                "14_research_brief.md",
            ]
            missing = [name for name in required if not (result["report_dir"] / name).exists()]
            if missing:
                raise AssertionError(f"self-test missing artifacts: {missing}")
            audit = pd.read_csv(result["audit"])
            if not audit.empty and audit["lookahead_flag"].astype(bool).any():
                raise AssertionError("self-test causal audit contains flags")
            profile = pd.read_csv(result["report_dir"] / "02_minute_path_profile.csv")
            if profile.empty or int(profile["minute"].max()) != 60:
                raise AssertionError("minute profile is incomplete")
            meta = json.loads((result["report_dir"] / "13_run_meta.json").read_text(encoding="utf-8"))
            if int(meta.get("synthetic_gap_bar_count", 0)) != 7:
                raise AssertionError("self-test did not preserve seven-minute source gap")
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
