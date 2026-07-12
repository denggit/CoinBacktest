#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Directional Impulse Continuation: first-passage path study (round 04).

Research question
-----------------
The fixed-time studies found little cost-after continuation, while MFE and MAE
were much larger than terminal returns. This round asks one question only:

after a basic directional impulse, does price reach a fixed favorable barrier
before an equally distant adverse barrier often enough, and early enough, to
support a causal exit mechanism?

This is an event-path study, not a final strategy backtest. It does not add
trend, volume, persistence, session, order-flow, footprint, funding, OI,
liquidation, position sizing, or portfolio logic. It branches directly from the
Round-01 base impulse event.

Causal and ambiguity policy
---------------------------
- Local OKX 1m trade bars only; build_missing=False.
- Signal is confirmed after bar t closes; entry is bar t+1 open.
- Future high/low paths start at the entry bar and use only observed bars.
- If favorable and adverse barriers are both first touched in the same 1m bar,
  intrabar order is unknowable. The primary result is conservative (stop first),
  and optimistic plus ambiguity-excluded results are reported beside it.
- If neither barrier is touched before the fixed time limit, exit uses that
  time-limit bar close.
- Normal execution cost is subtracted from every completed event path.

Performance policy
------------------
- The 240m path is gathered in bounded chunks; no multi-year N x 240 matrix is
  materialized at once.
- First-passage times for all fixed barriers are computed once per minimum-
  threshold event pool and reused across thresholds, time limits, raw/dedup,
  years, and months.
- Window features and future MFE/MAE arrays are precomputed once and reused.
- No iterrows scan over market bars or per-event Python path loop is used.
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


def _load_round02_module():
    path = Path(__file__).resolve().with_name("02_impulse_persistence_event_study.py")
    spec = importlib.util.spec_from_file_location("directional_impulse_round02_for_r04", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared round-02 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r02 = _load_round02_module()
r01 = r02.r01

SCRIPT_NAME = "04_impulse_first_passage_path_study"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION_R04"
EDGE_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION"
TITLE = "ETH Directional Impulse Continuation - First-Passage Path Study"
DEFAULT_OUT_DIR = (
    "data/reports/research/momentum/directional_impulse_continuation/"
    "04_impulse_first_passage_path_study"
)

DEFAULT_WINDOWS = (1, 3, 5, 10, 15)
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
DEFAULT_BARRIERS_BPS = (25, 50, 75, 100)
DEFAULT_TIME_LIMITS = (5, 15, 30, 60, 120, 240)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal ETH impulse first-passage path study.",
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
    p.add_argument("--barriers-bps", default=",".join(map(str, DEFAULT_BARRIERS_BPS)))
    p.add_argument("--time-limits", default=",".join(map(str, DEFAULT_TIME_LIMITS)))
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
        help="Development-only. Production writes compact minimum-threshold deduplicated events.",
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _threshold_tag(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _barrier_tag(bps: int) -> str:
    return f"{int(bps)}bps"


def _parse_barriers(raw: str) -> tuple[int, ...]:
    values = tuple(sorted(dict.fromkeys(int(x.strip()) for x in str(raw).split(",") if x.strip())))
    if not values or any(v <= 0 or v >= 10_000 for v in values):
        raise ValueError("barriers-bps must contain positive values below 10000")
    return values


def _first_true_minute(mask: np.ndarray) -> np.ndarray:
    any_hit = mask.any(axis=1)
    first = np.argmax(mask, axis=1).astype(np.int16) + np.int16(1)
    first[~any_hit] = np.int16(0)
    return first


def _build_first_passage_arrays(
    bars: pd.DataFrame,
    positions: np.ndarray,
    *,
    side: int,
    barriers_bps: tuple[int, ...],
    max_horizon: int,
    chunk_size: int,
    label: str,
    progress_enabled: bool,
) -> dict[int, dict[str, np.ndarray]]:
    """Compute first favorable/adverse touch minutes in bounded vectorized chunks."""
    if int(chunk_size) <= 0:
        raise ValueError("path-chunk-size must be positive")
    m = int(len(positions))
    out = {
        int(bps): {
            "favorable_first_min": np.zeros(m, dtype=np.int16),
            "adverse_first_min": np.zeros(m, dtype=np.int16),
        }
        for bps in barriers_bps
    }
    if m == 0:
        return out

    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=np.float64)
    open_arr = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=np.float64)
    offsets = np.arange(1, int(max_horizon) + 1, dtype=np.int64)
    total_chunks = int(math.ceil(m / int(chunk_size)))

    with ProgressReporter(
        label=label,
        total=total_chunks,
        every=1,
        enabled=progress_enabled,
    ) as progress:
        done = 0
        for start in range(0, m, int(chunk_size)):
            end = min(m, start + int(chunk_size))
            pos = positions[start:end].astype(np.int64, copy=False)
            entry = open_arr[pos + 1].astype(np.float32, copy=False)
            gather = pos[:, None] + offsets[None, :]
            future_high = high[gather].astype(np.float32, copy=False)
            future_low = low[gather].astype(np.float32, copy=False)

            if int(side) == 1:
                favorable = future_high
                favorable /= entry[:, None]
                favorable -= np.float32(1.0)
                adverse = np.float32(1.0) - future_low / entry[:, None]
            else:
                favorable = np.float32(1.0) - future_low / entry[:, None]
                adverse = future_high
                adverse /= entry[:, None]
                adverse -= np.float32(1.0)

            for bps in barriers_bps:
                barrier = np.float32(float(bps) / 10_000.0)
                out[int(bps)]["favorable_first_min"][start:end] = _first_true_minute(
                    favorable >= barrier
                )
                out[int(bps)]["adverse_first_min"][start:end] = _first_true_minute(
                    adverse >= barrier
                )
            done += 1
            progress.update(done)
    return out


def _outcome_arrays(
    events: pd.DataFrame,
    *,
    barrier_bps: int,
    time_limit: int,
) -> dict[str, np.ndarray]:
    tag = _barrier_tag(barrier_bps)
    favorable_first = events[f"favorable_first_{tag}_min"].to_numpy(dtype=np.int16)
    adverse_first = events[f"adverse_first_{tag}_min"].to_numpy(dtype=np.int16)
    h = int(time_limit)
    barrier = float(barrier_bps) / 10_000.0

    target_touch = (favorable_first > 0) & (favorable_first <= h)
    stop_touch = (adverse_first > 0) & (adverse_first <= h)
    target_first = target_touch & (~stop_touch | (favorable_first < adverse_first))
    stop_first = stop_touch & (~target_touch | (adverse_first < favorable_first))
    ambiguous = target_touch & stop_touch & (favorable_first == adverse_first)
    neither = ~(target_first | stop_first | ambiguous)

    terminal = events[f"forward_return_{h}m"].to_numpy(dtype=float)
    conservative = np.where(target_first, barrier, np.where(stop_first | ambiguous, -barrier, terminal))
    optimistic = np.where(target_first | ambiguous, barrier, np.where(stop_first, -barrier, terminal))

    return {
        "favorable_first": favorable_first,
        "adverse_first": adverse_first,
        "target_touch": target_touch,
        "stop_touch": stop_touch,
        "target_first": target_first,
        "stop_first": stop_first,
        "ambiguous": ambiguous,
        "neither": neither,
        "terminal_gross": terminal,
        "conservative_gross": conservative,
        "optimistic_gross": optimistic,
    }


def _safe_rate(mask: np.ndarray) -> float:
    return float(np.mean(mask)) if len(mask) else float("nan")


def _safe_median_hit(values: np.ndarray, touched: np.ndarray) -> float:
    selected = values[touched]
    return float(np.median(selected)) if selected.size else float("nan")


def _path_stats(
    outcomes: dict[str, np.ndarray],
    idx: np.ndarray,
    *,
    fee_cost: float,
    normal_cost: float,
    mfe: np.ndarray,
    mae: np.ndarray,
) -> dict[str, Any]:
    if idx.size == 0:
        return {
            "events": 0,
            "target_touch_rate": np.nan,
            "stop_touch_rate": np.nan,
            "target_first_rate": np.nan,
            "stop_first_rate": np.nan,
            "ambiguous_same_bar_rate": np.nan,
            "neither_hit_rate": np.nan,
            "median_target_touch_min": np.nan,
            "median_stop_touch_min": np.nan,
            "fixed_time_mean_gross": np.nan,
            "fixed_time_mean_net": np.nan,
            "conservative_mean_gross": np.nan,
            "conservative_mean_net": np.nan,
            "conservative_median_net": np.nan,
            "conservative_win_rate": np.nan,
            "conservative_profit_factor": np.nan,
            "conservative_standard_deviation": np.nan,
            "conservative_p05": np.nan,
            "conservative_p95": np.nan,
            "conservative_top_1_event_contribution": np.nan,
            "conservative_top_5_event_contribution": np.nan,
            "optimistic_mean_net": np.nan,
            "optimistic_median_net": np.nan,
            "optimistic_profit_factor": np.nan,
            "resolved_events": 0,
            "resolved_mean_net": np.nan,
            "resolved_median_net": np.nan,
            "resolved_profit_factor": np.nan,
            "mean_mfe": np.nan,
            "mean_mae": np.nan,
            "excursion_advantage": np.nan,
            "path_capture_uplift_vs_fixed_time": np.nan,
        }

    target_touch = outcomes["target_touch"][idx]
    stop_touch = outcomes["stop_touch"][idx]
    target_first = outcomes["target_first"][idx]
    stop_first = outcomes["stop_first"][idx]
    ambiguous = outcomes["ambiguous"][idx]
    neither = outcomes["neither"][idx]
    terminal = outcomes["terminal_gross"][idx]
    conservative = outcomes["conservative_gross"][idx]
    optimistic = outcomes["optimistic_gross"][idx]
    mf = mfe[idx]
    ma = mae[idx]

    c = r01._stats(
        conservative,
        conservative - float(fee_cost),
        conservative - float(normal_cost),
        mf,
        ma,
    )
    o = r01._stats(
        optimistic,
        optimistic - float(fee_cost),
        optimistic - float(normal_cost),
        mf,
        ma,
    )
    fixed = r01._stats(
        terminal,
        terminal - float(fee_cost),
        terminal - float(normal_cost),
        mf,
        ma,
    )
    resolved_mask = ~ambiguous
    resolved_idx = np.flatnonzero(resolved_mask)
    resolved_gross = conservative[resolved_mask]
    resolved_mfe = mf[resolved_mask]
    resolved_mae = ma[resolved_mask]
    rs = r01._stats(
        resolved_gross,
        resolved_gross - float(fee_cost),
        resolved_gross - float(normal_cost),
        resolved_mfe,
        resolved_mae,
    )

    favorable_first = outcomes["favorable_first"][idx]
    adverse_first = outcomes["adverse_first"][idx]
    return {
        "events": int(idx.size),
        "target_touch_rate": _safe_rate(target_touch),
        "stop_touch_rate": _safe_rate(stop_touch),
        "target_first_rate": _safe_rate(target_first),
        "stop_first_rate": _safe_rate(stop_first),
        "ambiguous_same_bar_rate": _safe_rate(ambiguous),
        "neither_hit_rate": _safe_rate(neither),
        "median_target_touch_min": _safe_median_hit(favorable_first, target_touch),
        "median_stop_touch_min": _safe_median_hit(adverse_first, stop_touch),
        "fixed_time_mean_gross": float(fixed["mean_gross"]),
        "fixed_time_mean_net": float(fixed["mean_net"]),
        "conservative_mean_gross": float(c["mean_gross"]),
        "conservative_mean_net": float(c["mean_net"]),
        "conservative_median_net": float(c["median_net"]),
        "conservative_win_rate": float(c["win_rate"]),
        "conservative_profit_factor": float(c["profit_factor"]),
        "conservative_standard_deviation": float(c["standard_deviation"]),
        "conservative_p05": float(c["p05"]),
        "conservative_p95": float(c["p95"]),
        "conservative_top_1_event_contribution": float(c["top_1_event_contribution"]),
        "conservative_top_5_event_contribution": float(c["top_5_event_contribution"]),
        "optimistic_mean_net": float(o["mean_net"]),
        "optimistic_median_net": float(o["median_net"]),
        "optimistic_profit_factor": float(o["profit_factor"]),
        "resolved_events": int(rs["events"]),
        "resolved_mean_net": float(rs["mean_net"]),
        "resolved_median_net": float(rs["median_net"]),
        "resolved_profit_factor": float(rs["profit_factor"]),
        "mean_mfe": float(c["mean_mfe"]),
        "mean_mae": float(c["mean_mae"]),
        "excursion_advantage": float(c["mean_mfe"] + c["mean_mae"]),
        "path_capture_uplift_vs_fixed_time": float(c["mean_net"] - fixed["mean_net"]),
    }


def _top_dependency_rows(
    events: pd.DataFrame,
    idx: np.ndarray,
    outcomes: dict[str, np.ndarray],
    *,
    direction: str,
    window: int,
    threshold: float,
    barrier_bps: int,
    time_limit: int,
    event_set: str,
    normal_cost: float,
    top_n: int = 20,
) -> list[dict[str, Any]]:
    if idx.size == 0:
        return []
    net = outcomes["conservative_gross"][idx] - float(normal_cost)
    positive_local = np.flatnonzero(np.isfinite(net) & (net > 0))
    if positive_local.size == 0:
        return []
    values = net[positive_local]
    order = positive_local[np.argsort(values)[::-1][: int(top_n)]]
    total_positive = float(values.sum())
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for rank, local_pos in enumerate(order, start=1):
        base_pos = int(idx[int(local_pos)])
        value = float(net[int(local_pos)])
        contribution = value / total_positive if total_positive > 0 else np.nan
        cumulative += contribution if np.isfinite(contribution) else 0.0
        rows.append(
            {
                "direction": direction,
                "impulse_window": int(window),
                "threshold": float(threshold),
                "barrier_bps": int(barrier_bps),
                "time_limit": int(time_limit),
                "event_set": event_set,
                "rank": int(rank),
                "event_id": int(events.iloc[base_pos]["event_id"]),
                "signal_time": events.iloc[base_pos]["signal_time"],
                "conservative_normal_net_return": value,
                "contribution_to_positive_return": contribution,
                "cumulative_top_contribution": cumulative,
            }
        )
    return rows


def _summaries_for_event_frame(
    events: pd.DataFrame,
    *,
    direction: str,
    window: int,
    thresholds: tuple[float, ...],
    barriers_bps: tuple[int, ...],
    time_limits: tuple[int, ...],
    threshold_masks: dict[float, np.ndarray],
    dedup_masks: dict[float, np.ndarray],
    study_months: int,
    fee_cost: float,
    normal_cost: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []

    years = pd.to_datetime(events["signal_bar_start"]).dt.year.to_numpy()
    months = pd.to_datetime(events["signal_bar_start"]).dt.to_period("M").astype(str).to_numpy()
    # Precompute period positions once. Re-sorting all selected event labels for
    # every threshold/barrier/time-limit combination is materially slower.
    year_positions = {int(y): np.flatnonzero(years == y) for y in np.unique(years)}
    month_positions = {str(m): np.flatnonzero(months == m) for m in np.unique(months)}

    for barrier_bps in barriers_bps:
        for time_limit in time_limits:
            outcomes = _outcome_arrays(events, barrier_bps=barrier_bps, time_limit=time_limit)
            mfe = events[f"mfe_{int(time_limit)}m"].to_numpy(dtype=float)
            mae = events[f"mae_{int(time_limit)}m"].to_numpy(dtype=float)
            for threshold in thresholds:
                for event_set, base_mask in (
                    ("raw", threshold_masks[float(threshold)]),
                    ("deduplicated", dedup_masks[float(threshold)]),
                ):
                    idx = np.flatnonzero(base_mask)
                    stat = _path_stats(
                        outcomes,
                        idx,
                        fee_cost=fee_cost,
                        normal_cost=normal_cost,
                        mfe=mfe,
                        mae=mae,
                    )

                    year_means: dict[int, float] = {}
                    for year, all_period_idx in year_positions.items():
                        period_idx = all_period_idx[base_mask[all_period_idx]]
                        if period_idx.size == 0:
                            continue
                        period = _path_stats(
                            outcomes,
                            period_idx,
                            fee_cost=fee_cost,
                            normal_cost=normal_cost,
                            mfe=mfe,
                            mae=mae,
                        )
                        yearly_rows.append(
                            {
                                "direction": direction,
                                "impulse_window": int(window),
                                "threshold": float(threshold),
                                "barrier_bps": int(barrier_bps),
                                "time_limit": int(time_limit),
                                "event_set": event_set,
                                "year": int(year),
                                **period,
                            }
                        )
                        year_means[int(year)] = float(period["conservative_mean_net"])

                    for month, all_period_idx in month_positions.items():
                        period_idx = all_period_idx[base_mask[all_period_idx]]
                        if period_idx.size == 0:
                            continue
                        period = _path_stats(
                            outcomes,
                            period_idx,
                            fee_cost=fee_cost,
                            normal_cost=normal_cost,
                            mfe=mfe,
                            mae=mae,
                        )
                        monthly_rows.append(
                            {
                                "direction": direction,
                                "impulse_window": int(window),
                                "threshold": float(threshold),
                                "barrier_bps": int(barrier_bps),
                                "time_limit": int(time_limit),
                                "event_set": event_set,
                                "month": str(month),
                                **period,
                            }
                        )

                    finite_years = {y: v for y, v in year_means.items() if np.isfinite(v)}
                    worst_year = min(finite_years, key=finite_years.get) if finite_years else None
                    summary_rows.append(
                        {
                            "direction": direction,
                            "impulse_window": int(window),
                            "threshold": float(threshold),
                            "barrier_bps": int(barrier_bps),
                            "time_limit": int(time_limit),
                            "event_set": event_set,
                            "events_per_month": float(stat["events"] / max(1, int(study_months))),
                            **stat,
                            "positive_year_count": int(sum(v > 0 for v in finite_years.values())),
                            "total_year_count": int(len(finite_years)),
                            "worst_year": worst_year,
                            "worst_year_conservative_mean_net": (
                                float(finite_years[worst_year]) if worst_year is not None else np.nan
                            ),
                        }
                    )
                    top_rows.extend(
                        _top_dependency_rows(
                            events,
                            idx,
                            outcomes,
                            direction=direction,
                            window=window,
                            threshold=threshold,
                            barrier_bps=barrier_bps,
                            time_limit=time_limit,
                            event_set=event_set,
                            normal_cost=normal_cost,
                        )
                    )
    return summary_rows, yearly_rows, monthly_rows, top_rows


def _build_path_plateau(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []

    threshold_keys = ["direction", "impulse_window", "barrier_bps", "time_limit", "event_set"]
    for key, part in summary.groupby(threshold_keys, observed=False, dropna=False):
        ordered = part.sort_values("threshold")
        previous: pd.Series | None = None
        for _, row in ordered.iterrows():
            item = {k: v for k, v in zip(threshold_keys, key if isinstance(key, tuple) else (key,), strict=False)}
            item.update(
                {
                    "plateau_axis": "threshold",
                    "axis_value": float(row["threshold"]),
                    "threshold": float(row["threshold"]),
                    "barrier_bps": int(row["barrier_bps"]),
                    "events": int(row["events"]),
                    "conservative_mean_net": float(row["conservative_mean_net"]),
                    "conservative_median_net": float(row["conservative_median_net"]),
                    "conservative_profit_factor": float(row["conservative_profit_factor"]),
                    "positive_year_count": int(row["positive_year_count"]),
                    "total_year_count": int(row["total_year_count"]),
                    "previous_axis_value": None if previous is None else float(previous["threshold"]),
                    "event_retention_vs_previous": (
                        np.nan if previous is None or float(previous["events"]) <= 0 else float(row["events"] / previous["events"])
                    ),
                    "mean_net_change_vs_previous": (
                        np.nan if previous is None else float(row["conservative_mean_net"] - previous["conservative_mean_net"])
                    ),
                    "median_net_change_vs_previous": (
                        np.nan if previous is None else float(row["conservative_median_net"] - previous["conservative_median_net"])
                    ),
                }
            )
            rows.append(item)
            previous = row

    barrier_keys = ["direction", "impulse_window", "threshold", "time_limit", "event_set"]
    for key, part in summary.groupby(barrier_keys, observed=False, dropna=False):
        ordered = part.sort_values("barrier_bps")
        previous = None
        for _, row in ordered.iterrows():
            item = {k: v for k, v in zip(barrier_keys, key if isinstance(key, tuple) else (key,), strict=False)}
            item.update(
                {
                    "plateau_axis": "barrier_bps",
                    "axis_value": int(row["barrier_bps"]),
                    "barrier_bps": int(row["barrier_bps"]),
                    "events": int(row["events"]),
                    "conservative_mean_net": float(row["conservative_mean_net"]),
                    "conservative_median_net": float(row["conservative_median_net"]),
                    "conservative_profit_factor": float(row["conservative_profit_factor"]),
                    "positive_year_count": int(row["positive_year_count"]),
                    "total_year_count": int(row["total_year_count"]),
                    "previous_axis_value": None if previous is None else int(previous["barrier_bps"]),
                    "event_retention_vs_previous": 1.0 if previous is not None else np.nan,
                    "mean_net_change_vs_previous": (
                        np.nan if previous is None else float(row["conservative_mean_net"] - previous["conservative_mean_net"])
                    ),
                    "median_net_change_vs_previous": (
                        np.nan if previous is None else float(row["conservative_median_net"] - previous["conservative_median_net"])
                    ),
                }
            )
            rows.append(item)
            previous = row
    return pd.DataFrame(rows)


def _build_signal_audit(events: pd.DataFrame, barriers_bps: tuple[int, ...], max_horizon: int) -> pd.DataFrame:
    audit = r01._build_signal_audit(events)
    first_passage_invalid = np.zeros(len(events), dtype=bool)
    for bps in barriers_bps:
        tag = _barrier_tag(bps)
        f = events[f"favorable_first_{tag}_min"].to_numpy(dtype=int)
        a = events[f"adverse_first_{tag}_min"].to_numpy(dtype=int)
        first_passage_invalid |= (f < 0) | (f > int(max_horizon)) | (a < 0) | (a > int(max_horizon))
    audit["first_passage_range_invalid_flag"] = first_passage_invalid
    audit["lookahead_flag"] = audit["lookahead_flag"].astype(bool) | first_passage_invalid
    return audit


def _build_brief(summary: pd.DataFrame, meta: dict[str, Any]) -> str:
    lines = [
        "# ETH Directional Impulse Continuation - First-Passage Path Study",
        "",
        "> Automated descriptive brief. This is not an accepted Edge decision and not a final strategy backtest.",
        "",
        "## Research question",
        "",
        "Does an impulse reach a fixed favorable barrier before an equally distant adverse barrier often enough to support a causal path-based exit?",
        "",
        "## Primary interpretation rule",
        "",
        "The conservative result treats a same-1m-bar target/stop first touch as stop-first. Optimistic and ambiguity-excluded results are reported only as bounds.",
        "",
        "## Fixed path definitions",
        "",
        f"- Barriers: {', '.join(str(x) + ' bps' for x in meta['barriers_bps'])}.",
        f"- Time limits: {', '.join(str(x) + 'm' for x in meta['time_limits'])}.",
        f"- Normal round-trip cost: {float(meta['normal_execution_cost']):.4%}.",
        "- If neither barrier is touched, the event exits at the fixed time-limit close.",
        "",
    ]
    eligible = summary[(summary["event_set"] == "deduplicated") & (summary["events"] >= 300)].copy()
    for direction in ("LONG", "SHORT"):
        part = eligible[eligible["direction"] == direction].sort_values(
            ["conservative_mean_net", "resolved_mean_net", "events"], ascending=[False, False, False]
        )
        lines.extend([f"## {direction}", ""])
        if part.empty:
            lines.extend(["No eligible rows.", ""])
            continue
        lines.extend(
            [
                "Best conservative rows (not parameter selection):",
                "",
                "| window | threshold | barrier | limit | events | mean net | median net | PF | target-first | stop-first | ambiguous | fixed-time net | positive years | top5 |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in part.head(10).iterrows():
            lines.append(
                f"| {int(row['impulse_window'])}m | {float(row['threshold']):.2f} | {int(row['barrier_bps'])}bps | "
                f"{int(row['time_limit'])}m | {int(row['events'])} | {float(row['conservative_mean_net']):.4%} | "
                f"{float(row['conservative_median_net']):.4%} | {float(row['conservative_profit_factor']):.3f} | "
                f"{float(row['target_first_rate']):.2%} | {float(row['stop_first_rate']):.2%} | "
                f"{float(row['ambiguous_same_bar_rate']):.2%} | {float(row['fixed_time_mean_net']):.4%} | "
                f"{int(row['positive_year_count'])}/{int(row['total_year_count'])} | "
                f"{float(row['conservative_top_5_event_contribution']):.2%} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision rules",
            "",
            "A path mechanism is credible only if the conservative result is positive after normal cost, the ambiguity rate is not driving the conclusion, nearby barriers/thresholds remain stable, event frequency is reasonable, and most years agree.",
            "",
            "A positive optimistic result with a negative conservative result is not sufficient because 1m OHLC cannot reveal intrabar touch order.",
            "",
            "## Run facts",
            "",
            f"- Unique minimum-threshold raw event rows analyzed: {int(meta['unique_raw_event_rows_analyzed']):,}.",
            f"- Compact minimum-threshold deduplicated event rows written: {int(meta['compact_event_rows_written']):,}.",
            f"- Synthetic gap bars excluded: {int(meta['synthetic_gap_bar_count']):,}.",
        ]
    )
    return "\n".join(lines) + "\n"


def _update_log(log_path: Path, summary: pd.DataFrame, meta: dict[str, Any]) -> None:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# ETH Directional Impulse Continuation — Research Log\n"
    marker_start = "<!-- ROUND04_AUTO_RESULT_START -->"
    marker_end = "<!-- ROUND04_AUTO_RESULT_END -->"
    eligible = summary[(summary["event_set"] == "deduplicated") & (summary["events"] >= 300)]

    def row_text(direction: str) -> str:
        part = eligible[eligible["direction"] == direction].sort_values("conservative_mean_net", ascending=False)
        if part.empty:
            return "No eligible row"
        r = part.iloc[0]
        return (
            f"{int(r['impulse_window'])}m / threshold {float(r['threshold']):.2f} / "
            f"{int(r['barrier_bps'])}bps / {int(r['time_limit'])}m: events={int(r['events'])}, "
            f"events/month={float(r['events_per_month']):.2f}, conservative mean net={float(r['conservative_mean_net']):.4%}, "
            f"median net={float(r['conservative_median_net']):.4%}, PF={float(r['conservative_profit_factor']):.3f}, "
            f"target-first={float(r['target_first_rate']):.2%}, ambiguous={float(r['ambiguous_same_bar_rate']):.2%}, "
            f"positive years={int(r['positive_year_count'])}/{int(r['total_year_count'])}"
        )

    block = "\n".join(
        [
            marker_start,
            "## Round 04 generated result",
            "",
            f"- LONG best conservative descriptive row: {row_text('LONG')}",
            f"- SHORT best conservative descriptive row: {row_text('SHORT')}",
            f"- Raw event rows analyzed: {int(meta['unique_raw_event_rows_analyzed']):,}",
            f"- Compact deduplicated rows written: {int(meta['compact_event_rows_written']):,}",
            "- Final interpretation pending review of the generated Review Pack.",
            marker_end,
        ]
    )
    if marker_start in text and marker_end in text:
        before = text.split(marker_start, 1)[0]
        after = text.split(marker_end, 1)[1]
        text = before + block + after
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    log_path.write_text(text, encoding="utf-8")


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    windows = r01._parse_int_csv(args.impulse_windows)
    thresholds = r01._parse_float_csv(args.thresholds)
    barriers_bps = _parse_barriers(args.barriers_bps)
    time_limits = r01._parse_int_csv(args.time_limits)
    if int(args.vol_min_periods) > int(args.vol_lookback_bars):
        raise ValueError("vol-min-periods cannot exceed vol-lookback-bars")
    if max(time_limits) > np.iinfo(np.int16).max:
        raise ValueError("time limit exceeds int16 first-passage storage")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "05_events.csv"
    audit_path = out_dir / "06_signal_audit.csv"
    for path in (events_path, audit_path):
        if path.exists():
            path.unlink()

    validation = r01.validate_bars(bars, args)
    masks = r02._eligible_masks(bars, args, time_limits)
    study_months = int(masks["study_months"])
    fee_only_cost = float(args.entry_fee_rate + args.exit_fee_rate)
    normal_cost = float(fee_only_cost + args.entry_slippage + args.exit_slippage)
    if not math.isclose(fee_only_cost, 0.0011, rel_tol=0.0, abs_tol=1e-12):
        print(f"[warning] fee-only cost differs from 0.11%: {fee_only_cost:.6%}", flush=True)
    if not math.isclose(normal_cost, 0.0015, rel_tol=0.0, abs_tol=1e-12):
        print(f"[warning] normal cost differs from 0.15%: {normal_cost:.6%}", flush=True)

    path_cache = r01.build_path_cache(bars, time_limits, progress_enabled=not args.no_progress)
    log_return, abs_price_change, historical_1m_vol = r01.build_base_volatility(
        bars, int(args.vol_lookback_bars), int(args.vol_min_periods)
    )

    event_count_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    event_id_cursor = 1
    unique_raw_rows = 0
    compact_rows_written = 0
    first_event_write = True
    first_audit_write = True
    minimum_threshold = min(thresholds)
    n = len(bars)

    print("[feature build] base impulse features", flush=True)
    outer = ProgressReporter(
        label="[event detection + first-passage summaries] direction/windows",
        total=len(windows) * 2,
        every=max(1, int(args.progress_every)),
        enabled=not args.no_progress,
    )
    done = 0
    for window in windows:
        features = r01.build_window_features(bars, window, log_return, abs_price_change, historical_1m_vol)
        norm = features.normalized_impulse
        for direction, side in (("LONG", 1), ("SHORT", -1)):
            directed_norm = float(side) * norm
            all_min_positions = np.flatnonzero(
                np.isfinite(directed_norm) & (directed_norm >= float(minimum_threshold))
            )
            eligible_positions = all_min_positions[masks["eligible"][all_min_positions]]
            unique_raw_rows += int(len(eligible_positions))

            threshold_masks_event: dict[float, np.ndarray] = {}
            dedup_masks_event: dict[float, np.ndarray] = {}
            dedup_min_flags = np.zeros(len(eligible_positions), dtype=bool)
            for threshold in thresholds:
                all_t = all_min_positions[directed_norm[all_min_positions] >= float(threshold)]
                dedup_t = r01._deduplicate_positions(all_t, int(window))
                dedup_axis = np.zeros(n, dtype=bool)
                dedup_axis[all_t] = dedup_t
                event_threshold_mask = directed_norm[eligible_positions] >= float(threshold)
                event_dedup_mask = event_threshold_mask & dedup_axis[eligible_positions]
                threshold_masks_event[float(threshold)] = event_threshold_mask
                dedup_masks_event[float(threshold)] = event_dedup_mask
                if math.isclose(float(threshold), float(minimum_threshold), rel_tol=0.0, abs_tol=1e-12):
                    dedup_min_flags = event_dedup_mask

                research_t = masks["research_mask"][all_t] & masks["causal_next_bar"][all_t]
                eligible_t = masks["eligible"][all_t]
                raw_count = int(eligible_t.sum())
                dedup_count = int((eligible_t & dedup_t).sum())
                overlap_ratio = 1.0 - dedup_count / raw_count if raw_count else np.nan
                for event_set, count in (("raw", raw_count), ("deduplicated", dedup_count)):
                    event_count_rows.append(
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

            if eligible_positions.size:
                event_frame = r01._build_event_frame(
                    bars=bars,
                    positions=eligible_positions,
                    dedup_flags=dedup_min_flags,
                    direction=direction,
                    side=side,
                    window=window,
                    threshold=minimum_threshold,
                    features=features,
                    path_cache=path_cache,
                    full_forward_observed_mask=masks["full_forward_observed"],
                    horizons=time_limits,
                    fee_cost=fee_only_cost,
                    normal_cost=normal_cost,
                    event_id_start=event_id_cursor,
                )
                event_id_cursor += len(event_frame)
                event_frame["minimum_pool_threshold"] = float(minimum_threshold)
                directed_event_norm = directed_norm[eligible_positions]
                threshold_array = np.asarray(thresholds, dtype=float)
                threshold_idx = np.searchsorted(threshold_array, directed_event_norm, side="right") - 1
                threshold_idx = np.clip(threshold_idx, 0, len(threshold_array) - 1)
                event_frame["max_threshold_reached"] = threshold_array[threshold_idx]
                for threshold in thresholds:
                    tag = _threshold_tag(threshold)
                    event_frame[f"event_{tag}_flag"] = threshold_masks_event[float(threshold)]
                    event_frame[f"deduplicated_{tag}_flag"] = dedup_masks_event[float(threshold)]

                first_passage = _build_first_passage_arrays(
                    bars,
                    eligible_positions,
                    side=side,
                    barriers_bps=barriers_bps,
                    max_horizon=max(time_limits),
                    chunk_size=int(args.path_chunk_size),
                    label=f"[first passage] {direction} {int(window)}m chunks",
                    progress_enabled=not args.no_progress,
                )
                for bps in barriers_bps:
                    tag = _barrier_tag(bps)
                    event_frame[f"favorable_first_{tag}_min"] = first_passage[int(bps)][
                        "favorable_first_min"
                    ]
                    event_frame[f"adverse_first_{tag}_min"] = first_passage[int(bps)][
                        "adverse_first_min"
                    ]

                sr, yr, mr, tr = _summaries_for_event_frame(
                    event_frame,
                    direction=direction,
                    window=window,
                    thresholds=thresholds,
                    barriers_bps=barriers_bps,
                    time_limits=time_limits,
                    threshold_masks=threshold_masks_event,
                    dedup_masks=dedup_masks_event,
                    study_months=study_months,
                    fee_cost=fee_only_cost,
                    normal_cost=normal_cost,
                )
                summary_rows.extend(sr)
                yearly_rows.extend(yr)
                monthly_rows.extend(mr)
                top_rows.extend(tr)

                if not args.skip_events_csv:
                    compact = event_frame[dedup_min_flags].copy()
                    compact["event_storage_scope"] = "minimum_threshold_deduplicated"
                    compact_rows_written += int(len(compact))
                    r01._write_stream_csv(compact, events_path, first_write=first_event_write)
                    first_event_write = False
                    audit = _build_signal_audit(compact, barriers_bps, max(time_limits))
                    r01._write_stream_csv(audit, audit_path, first_write=first_audit_write)
                    first_audit_write = False
                    del compact, audit
                del event_frame, first_passage
            done += 1
            outer.update(done)
        del features
    outer.close()

    print("[summaries] assembling first-passage timing, path plateaus and dependency tables", flush=True)
    event_counts = pd.DataFrame(event_count_rows)
    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    monthly = pd.DataFrame(monthly_rows)
    timing_cols = [
        "direction",
        "impulse_window",
        "threshold",
        "barrier_bps",
        "time_limit",
        "event_set",
        "events",
        "events_per_month",
        "target_touch_rate",
        "stop_touch_rate",
        "target_first_rate",
        "stop_first_rate",
        "ambiguous_same_bar_rate",
        "neither_hit_rate",
        "median_target_touch_min",
        "median_stop_touch_min",
        "mean_mfe",
        "mean_mae",
        "excursion_advantage",
        "fixed_time_mean_net",
        "conservative_mean_net",
        "optimistic_mean_net",
        "resolved_mean_net",
        "path_capture_uplift_vs_fixed_time",
    ]
    timing = summary[timing_cols].copy() if not summary.empty else pd.DataFrame(columns=timing_cols)
    path_plateau = _build_path_plateau(summary)
    top_dependency = pd.DataFrame(top_rows)

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
        "status": "research_only_not_tradable",
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
        "barriers_bps": list(barriers_bps),
        "time_limits": list(time_limits),
        "path_definition": {
            "favorable_long": "future high / entry open - 1",
            "adverse_long": "1 - future low / entry open",
            "favorable_short": "1 - future low / entry open",
            "adverse_short": "future high / entry open - 1",
            "first_passage_axis": "entry bar through fixed time limit, inclusive",
            "same_bar_policy_primary": "conservative stop-first",
            "same_bar_bounds": "optimistic target-first and ambiguity-excluded results also reported",
            "neither_policy": "exit at fixed time-limit close",
        },
        "current_impulse_normalization": {
            "formula": "impulse_return / (historical rolling std shifted by impulse_window * sqrt(window))",
            "vol_lookback_bars": int(args.vol_lookback_bars),
            "vol_min_periods": int(args.vol_min_periods),
        },
        "round02_persistence_condition_stacked": False,
        "round03_notional_condition_stacked": False,
        "deduplication": "threshold-specific same-direction stream; cooldown=impulse_window",
        "event_storage": (
            "all minimum-threshold raw events analyzed in memory per direction/window; "
            "05_events.csv stores minimum-threshold deduplicated rows only with higher-threshold membership flags"
        ),
        "entry_policy": "closed signal bar; next 1m bar open",
        "fee_only_cost": fee_only_cost,
        "normal_execution_cost": normal_cost,
        "cost_components": {
            "entry_fee": float(args.entry_fee_rate),
            "exit_fee": float(args.exit_fee_rate),
            "entry_slippage": float(args.entry_slippage),
            "exit_slippage": float(args.exit_slippage),
        },
        "path_chunk_size": int(args.path_chunk_size),
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
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "research_boundary": "one mechanism only: symmetric first-passage path and causal time-stop exit",
    }

    brief = _build_brief(summary, meta)
    artifacts = [
        (event_counts, out_dir / "01_event_counts.csv"),
        (summary, out_dir / "02_path_summary.csv"),
        (yearly, out_dir / "03_yearly.csv"),
        (monthly, out_dir / "04_monthly.csv"),
        (timing, out_dir / "07_first_passage_timing.csv"),
        (path_plateau, out_dir / "08_path_plateau.csv"),
        (top_dependency, out_dir / "09_top_trade_dependency.csv"),
    ]
    print("[artifacts] writing report files", flush=True)
    with ProgressReporter(label="[artifacts] tables", total=len(artifacts) + 3, every=1, enabled=not args.no_progress) as p:
        count = 0
        for frame, path in artifacts:
            frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
            count += 1
            p.update(count)
        r01._write_json(meta, out_dir / "10_run_meta.json")
        count += 1
        p.update(count)
        (out_dir / "11_research_brief.md").write_text(brief, encoding="utf-8")
        count += 1
        p.update(count)
        _update_log(Path(__file__).resolve().with_name("00_research_log.md"), summary, meta)
        count += 1
        p.update(count)

    if not args.skip_review_pack:
        print("[review pack] packaging summary artifacts", flush=True)
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report_dir={out_dir}", flush=True)
    return {
        "report_dir": out_dir,
        "events": events_path,
        "audit": audit_path,
        "review_pack": out_dir / "gpt_review_pack.zip",
    }


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] deterministic first-passage path study", flush=True)
    # Exact kernel check: entry is bar 1 open at 100. A 25bps target and
    # stop are both first touched in bar 2, while the 50bps target is first
    # touched in bar 3 and its stop is never touched.
    kernel_index = pd.date_range("2023-01-01", periods=5, freq="1min")
    kernel = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [100.0, 100.20, 100.30, 100.60, 100.0],
            "low": [100.0, 99.90, 99.70, 99.80, 100.0],
            "close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "volume": [1.0] * 5,
            "source_bar_observed_flag": [True] * 5,
        },
        index=kernel_index,
    )
    kernel_result = _build_first_passage_arrays(
        kernel,
        np.asarray([0], dtype=int),
        side=1,
        barriers_bps=(25, 50),
        max_horizon=3,
        chunk_size=8,
        label="[self-test kernel]",
        progress_enabled=False,
    )
    if int(kernel_result[25]["favorable_first_min"][0]) != 2 or int(kernel_result[25]["adverse_first_min"][0]) != 2:
        raise AssertionError("same-bar first-passage kernel result is incorrect")
    if int(kernel_result[50]["favorable_first_min"][0]) != 3 or int(kernel_result[50]["adverse_first_min"][0]) != 0:
        raise AssertionError("single-sided first-passage kernel result is incorrect")

    raw = r01._synthetic_bars()
    raw = raw.drop(raw.index[3700:3707])
    bars = r01._regularize_trade_bar_axis(raw)
    log_path = Path(__file__).resolve().with_name("00_research_log.md")
    original_log = log_path.read_text(encoding="utf-8") if log_path.exists() else None
    try:
        with tempfile.TemporaryDirectory(prefix="dic_r04_") as tmp:
            args.out_dir = tmp
            args.warmup_start_date = "2022-12-20"
            args.start_date = "2022-12-23"
            args.end_date = "2022-12-24"
            args.vol_lookback_bars = 720
            args.vol_min_periods = 360
            args.impulse_windows = "1,3"
            args.thresholds = "1.0,1.5"
            args.barriers_bps = "25,50"
            args.time_limits = "5,15,30"
            args.path_chunk_size = 256
            args.skip_review_pack = True
            args.skip_events_csv = False
            result = run_research(bars, args)
            summary = pd.read_csv(result["report_dir"] / "02_path_summary.csv")
            audit = pd.read_csv(result["audit"])
            if summary.empty:
                raise AssertionError("self-test path summary is empty")
            if not audit.empty and audit["lookahead_flag"].astype(bool).any():
                raise AssertionError("self-test causal audit contains lookahead/data-integrity flags")
            rate_sum = (
                summary["target_first_rate"]
                + summary["stop_first_rate"]
                + summary["ambiguous_same_bar_rate"]
                + summary["neither_hit_rate"]
            )
            if not np.allclose(rate_sum.dropna().to_numpy(dtype=float), 1.0, rtol=0.0, atol=1e-9):
                raise AssertionError("first-passage outcome rates do not sum to one")
            if (summary["conservative_mean_net"] > summary["optimistic_mean_net"] + 1e-12).any():
                raise AssertionError("conservative result exceeds optimistic bound")
            meta = json.loads((result["report_dir"] / "10_run_meta.json").read_text(encoding="utf-8"))
            if int(meta.get("synthetic_gap_bar_count", 0)) != 7:
                raise AssertionError("self-test did not preserve the expected seven-minute source gap")
            events = pd.read_csv(result["events"], nrows=50)
            if not events.empty and not (events["event_storage_scope"] == "minimum_threshold_deduplicated").all():
                raise AssertionError("compact event storage scope is incorrect")
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
