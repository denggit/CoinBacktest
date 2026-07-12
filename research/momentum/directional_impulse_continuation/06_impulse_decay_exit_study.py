#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Directional Impulse Continuation: causal impulse-decay exit study (round 06).

Research question
-----------------
The basic event and post-signal confirmation studies show that waiting for a
confirmation bar consumes most of the small continuation edge. This round asks
one question only:

    after immediate next-open entry, can a causal exit triggered by decay of
    the same directional impulse retain enough of the observed favorable path
    to cover normal execution cost?

This round does not add trend, session, notional, persistence, order-flow,
range-bar, footprint, position sizing, or portfolio filters. Entry remains the
Round-01 basic impulse event. Only the exit mechanism changes.

Causal exit definition
----------------------
- Signal bar t must fully close.
- Entry is bar t+1 open.
- After every subsequent 1m bar closes, recompute the same-window directional
  impulse using only closed bars and the fixed pre-signal volatility baseline.
- live_retention = live_directional_impulse_z / original_directional_impulse_z.
- If live_retention falls to or below a fixed retention floor, exit at the next
  1m open.
- If no decay trigger occurs, exit after a fixed maximum number of fully closed
  holding bars, again at the next 1m open.
- MFE/MAE uses only bars actually held before that causal exit.

Performance policy
------------------
- Local OKX 1m trade bars only; build_missing=False.
- One data load and one feature build per impulse window.
- The maximum 60m holding path is processed in bounded vectorized chunks.
- All retention floors and maximum holds are computed from the same chunk path
  and reused for nested thresholds, raw/deduplicated sets, years and months.
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
    spec = importlib.util.spec_from_file_location("directional_impulse_round04_for_r06", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared round-04 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r04 = _load_round04_module()
r02 = r04.r02
r01 = r04.r01

SCRIPT_NAME = "06_impulse_decay_exit_study"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION_R06"
EDGE_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION"
TITLE = "ETH Directional Impulse Continuation - Causal Impulse-Decay Exit"
DEFAULT_OUT_DIR = (
    "data/reports/research/momentum/directional_impulse_continuation/"
    "06_impulse_decay_exit_study"
)

DEFAULT_WINDOWS = (1, 3, 5, 10, 15)
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
DEFAULT_RETENTION_FLOORS = (0.0, 0.25, 0.50, 0.75)
DEFAULT_MAX_HOLDS = (5, 15, 30, 60)
ANCHOR_RETENTION_FLOOR = 0.50
ANCHOR_MAX_HOLD = 30

REASON_DECAY = np.uint8(1)
REASON_MAX_HOLD = np.uint8(2)
REASON_LABELS = {1: "impulse_decay", 2: "max_hold"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal ETH impulse-decay exit study.",
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
    p.add_argument(
        "--retention-floors",
        default=",".join(map(str, DEFAULT_RETENTION_FLOORS)),
        help="Exit when live same-window impulse strength retains no more than this fraction of signal strength.",
    )
    p.add_argument("--max-holds", default=",".join(map(str, DEFAULT_MAX_HOLDS)))
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
        "--skip-trades-csv",
        action="store_true",
        help="Development-only. Production writes the fixed anchor-variant trade audit.",
    )
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _parse_retention_floors(raw: str) -> tuple[float, ...]:
    values = tuple(sorted(dict.fromkeys(float(x.strip()) for x in str(raw).split(",") if x.strip())))
    if not values or any((not math.isfinite(v)) or v < 0.0 or v >= 1.0 for v in values):
        raise ValueError("retention-floors must be finite values in [0, 1)")
    return values


def _floor_tag(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _threshold_tag(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _first_true_minute(mask: np.ndarray) -> np.ndarray:
    any_hit = mask.any(axis=1)
    first = np.argmax(mask, axis=1).astype(np.int16) + np.int16(1)
    first[~any_hit] = np.int16(0)
    return first


def _build_decay_outcomes(
    bars: pd.DataFrame,
    positions: np.ndarray,
    *,
    side: int,
    window: int,
    directed_signal_z: np.ndarray,
    pre_impulse_1m_vol: np.ndarray,
    retention_floors: tuple[float, ...],
    max_holds: tuple[int, ...],
    chunk_size: int,
    label: str,
    progress_enabled: bool,
) -> dict[tuple[float, int], dict[str, np.ndarray]]:
    """Compute all causal decay exits in bounded vectorized chunks."""
    if int(chunk_size) <= 0:
        raise ValueError("path-chunk-size must be positive")
    m = int(len(positions))
    max_hold = int(max(max_holds))
    result: dict[tuple[float, int], dict[str, np.ndarray]] = {}
    for floor in retention_floors:
        for hold in max_holds:
            result[(float(floor), int(hold))] = {
                "exit_minute": np.zeros(m, dtype=np.int16),
                "reason_code": np.zeros(m, dtype=np.uint8),
                "gross": np.full(m, np.nan, dtype=np.float32),
                "mfe": np.full(m, np.nan, dtype=np.float32),
                "mae": np.full(m, np.nan, dtype=np.float32),
                "exit_retention": np.full(m, np.nan, dtype=np.float32),
            }
    if m == 0:
        return result

    open_arr = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=np.float64)
    close_arr = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=np.float64)
    high_arr = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=np.float64)
    low_arr = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=np.float64)
    offsets = np.arange(1, max_hold + 1, dtype=np.int64)
    rows = np.arange(min(int(chunk_size), m), dtype=np.int64)
    total_chunks = int(math.ceil(m / int(chunk_size)))

    with ProgressReporter(label=label, total=total_chunks, every=1, enabled=progress_enabled) as progress:
        done = 0
        for start in range(0, m, int(chunk_size)):
            end = min(m, start + int(chunk_size))
            size = end - start
            row_idx = rows[:size]
            pos = positions[start:end].astype(np.int64, copy=False)
            entry = open_arr[pos + 1].astype(np.float32, copy=False)
            gather = pos[:, None] + offsets[None, :]
            previous = gather - int(window)

            path_close = close_arr[gather].astype(np.float32, copy=False)
            path_previous_close = close_arr[previous].astype(np.float32, copy=False)
            denom = (
                pre_impulse_1m_vol[pos].astype(np.float32, copy=False)
                * np.float32(math.sqrt(int(window)))
            )
            live_z = np.float32(float(side)) * (path_close / path_previous_close - np.float32(1.0))
            live_z /= denom[:, None]
            signal_strength = directed_signal_z[pos].astype(np.float32, copy=False)
            retention = live_z / signal_strength[:, None]

            path_high = high_arr[gather].astype(np.float32, copy=False)
            path_low = low_arr[gather].astype(np.float32, copy=False)
            if int(side) == 1:
                favorable = path_high / entry[:, None] - np.float32(1.0)
                adverse = np.float32(1.0) - path_low / entry[:, None]
            else:
                favorable = np.float32(1.0) - path_low / entry[:, None]
                adverse = path_high / entry[:, None] - np.float32(1.0)
            cumulative_mfe = np.maximum.accumulate(favorable, axis=1)
            cumulative_adverse = np.maximum.accumulate(adverse, axis=1)

            for floor in retention_floors:
                first_decay = _first_true_minute(retention <= np.float32(floor))
                for hold in max_holds:
                    h = int(hold)
                    decay_in_time = (first_decay > 0) & (first_decay <= h)
                    exit_minute = np.where(decay_in_time, first_decay, h).astype(np.int16)
                    exit_pos = pos + exit_minute.astype(np.int64) + 1
                    exit_price = open_arr[exit_pos].astype(np.float32, copy=False)
                    gross = np.float32(float(side)) * (exit_price / entry - np.float32(1.0))
                    path_col = exit_minute.astype(np.int64) - 1
                    key = (float(floor), h)
                    target = result[key]
                    target["exit_minute"][start:end] = exit_minute
                    target["reason_code"][start:end] = np.where(
                        decay_in_time, REASON_DECAY, REASON_MAX_HOLD
                    ).astype(np.uint8)
                    target["gross"][start:end] = gross
                    target["mfe"][start:end] = cumulative_mfe[row_idx, path_col]
                    target["mae"][start:end] = -cumulative_adverse[row_idx, path_col]
                    target["exit_retention"][start:end] = retention[row_idx, path_col]
            done += 1
            progress.update(done)
    return result


def _extra_stats(gross: np.ndarray, mfe: np.ndarray, exit_minute: np.ndarray, reason_code: np.ndarray) -> dict[str, float]:
    gross = np.asarray(gross, dtype=float)
    mfe = np.asarray(mfe, dtype=float)
    exit_minute = np.asarray(exit_minute, dtype=float)
    reason_code = np.asarray(reason_code)
    finite = np.isfinite(gross) & np.isfinite(mfe) & np.isfinite(exit_minute)
    if not finite.any():
        return {
            "mean_hold_minutes": np.nan,
            "median_hold_minutes": np.nan,
            "decay_exit_rate": np.nan,
            "max_hold_exit_rate": np.nan,
            "gross_positive_rate": np.nan,
            "mean_positive_mfe_capture_ratio": np.nan,
            "median_positive_mfe_capture_ratio": np.nan,
            "mean_mfe_giveback": np.nan,
        }
    g = gross[finite]
    m = mfe[finite]
    e = exit_minute[finite]
    r = reason_code[finite]
    capture_mask = (g > 0.0) & (m > 0.0)
    capture = g[capture_mask] / m[capture_mask] if capture_mask.any() else np.asarray([], dtype=float)
    return {
        "mean_hold_minutes": float(np.mean(e)),
        "median_hold_minutes": float(np.median(e)),
        "decay_exit_rate": float(np.mean(r == REASON_DECAY)),
        "max_hold_exit_rate": float(np.mean(r == REASON_MAX_HOLD)),
        "gross_positive_rate": float(np.mean(g > 0.0)),
        "mean_positive_mfe_capture_ratio": float(np.mean(capture)) if capture.size else np.nan,
        "median_positive_mfe_capture_ratio": float(np.median(capture)) if capture.size else np.nan,
        "mean_mfe_giveback": float(np.mean(m - g)),
    }


def _top_rows(
    event_frame: pd.DataFrame,
    *,
    indices: np.ndarray,
    gross: np.ndarray,
    normal_cost: float,
    direction: str,
    window: int,
    threshold: float,
    retention_floor: float,
    max_hold: int,
    event_set: str,
) -> list[dict[str, Any]]:
    net = np.asarray(gross, dtype=float) - float(normal_cost)
    positive = indices[np.isfinite(net[indices]) & (net[indices] > 0.0)]
    if not positive.size:
        return []
    ordered = positive[np.argsort(net[positive])[::-1][:5]]
    total_positive = float(net[positive].sum())
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for rank, idx in enumerate(ordered, start=1):
        contribution = float(net[idx] / total_positive) if total_positive > 0 else np.nan
        cumulative += contribution if math.isfinite(contribution) else 0.0
        rows.append(
            {
                "direction": direction,
                "impulse_window": int(window),
                "threshold": float(threshold),
                "retention_floor": float(retention_floor),
                "max_hold": int(max_hold),
                "event_set": event_set,
                "rank": int(rank),
                "event_id": int(event_frame.iloc[idx]["event_id"]),
                "signal_time": event_frame.iloc[idx]["signal_time"],
                "entry_time": event_frame.iloc[idx]["entry_time"],
                "normal_net_return": float(net[idx]),
                "contribution_to_positive_return": contribution,
                "cumulative_top_contribution": cumulative,
            }
        )
    return rows


def _summary_for_variant(
    event_frame: pd.DataFrame,
    *,
    mask: np.ndarray,
    outcome: dict[str, np.ndarray],
    direction: str,
    window: int,
    threshold: float,
    retention_floor: float,
    max_hold: int,
    event_set: str,
    study_months: int,
    fee_cost: float,
    normal_cost: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    indices = np.flatnonzero(mask)
    gross = outcome["gross"].astype(float, copy=False)
    fee = gross - float(fee_cost)
    net = gross - float(normal_cost)
    mfe = outcome["mfe"].astype(float, copy=False)
    mae = outcome["mae"].astype(float, copy=False)
    exit_minute = outcome["exit_minute"]
    reason_code = outcome["reason_code"]

    years = pd.to_datetime(event_frame["entry_time"]).dt.year.to_numpy()
    months = pd.to_datetime(event_frame["entry_time"]).dt.to_period("M").astype(str).to_numpy()
    year_groups = r01._group_indices(years, mask)
    month_groups = r01._group_indices(months, mask)

    stat = r01._stats(gross[indices], fee[indices], net[indices], mfe[indices], mae[indices])
    extra = _extra_stats(gross[indices], mfe[indices], exit_minute[indices], reason_code[indices])
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    year_means: dict[int, float] = {}
    for year, idx in year_groups.items():
        period = r01._stats(gross[idx], fee[idx], net[idx], mfe[idx], mae[idx])
        yearly_rows.append(
            {
                "direction": direction,
                "impulse_window": int(window),
                "threshold": float(threshold),
                "retention_floor": float(retention_floor),
                "max_hold": int(max_hold),
                "event_set": event_set,
                "year": int(year),
                **period,
                **_extra_stats(gross[idx], mfe[idx], exit_minute[idx], reason_code[idx]),
            }
        )
        year_means[int(year)] = float(period["mean_net"])
    for month, idx in month_groups.items():
        period = r01._stats(gross[idx], fee[idx], net[idx], mfe[idx], mae[idx])
        monthly_rows.append(
            {
                "direction": direction,
                "impulse_window": int(window),
                "threshold": float(threshold),
                "retention_floor": float(retention_floor),
                "max_hold": int(max_hold),
                "event_set": event_set,
                "month": str(month),
                **period,
                **_extra_stats(gross[idx], mfe[idx], exit_minute[idx], reason_code[idx]),
            }
        )
    finite_years = {y: v for y, v in year_means.items() if math.isfinite(v)}
    worst_year = min(finite_years, key=finite_years.get) if finite_years else None
    summary = {
        "direction": direction,
        "impulse_window": int(window),
        "threshold": float(threshold),
        "retention_floor": float(retention_floor),
        "max_hold": int(max_hold),
        "event_set": event_set,
        "events_per_month": float(stat["events"] / max(1, study_months)),
        **stat,
        **extra,
        "positive_year_count": int(sum(v > 0.0 for v in finite_years.values())),
        "total_year_count": int(len(finite_years)),
        "worst_year": worst_year,
        "worst_year_mean_net": finite_years.get(worst_year, np.nan) if worst_year is not None else np.nan,
    }
    top = _top_rows(
        event_frame,
        indices=indices,
        gross=gross,
        normal_cost=normal_cost,
        direction=direction,
        window=window,
        threshold=threshold,
        retention_floor=retention_floor,
        max_hold=max_hold,
        event_set=event_set,
    )
    return [summary], yearly_rows, monthly_rows, top


def _build_floor_plateau(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["direction", "impulse_window", "threshold", "max_hold", "event_set"]
    for key, part in summary.groupby(keys, observed=False, dropna=False):
        ordered = part.sort_values("retention_floor")
        previous: pd.Series | None = None
        for _, row in ordered.iterrows():
            item = {k: v for k, v in zip(keys, key if isinstance(key, tuple) else (key,), strict=False)}
            item.update(
                {
                    "retention_floor": float(row["retention_floor"]),
                    "events": int(row["events"]),
                    "mean_net": float(row["mean_net"]),
                    "median_net": float(row["median_net"]),
                    "profit_factor": float(row["profit_factor"]),
                    "mean_hold_minutes": float(row["mean_hold_minutes"]),
                    "decay_exit_rate": float(row["decay_exit_rate"]),
                    "mean_positive_mfe_capture_ratio": float(row["mean_positive_mfe_capture_ratio"]),
                    "positive_year_count": int(row["positive_year_count"]),
                    "total_year_count": int(row["total_year_count"]),
                }
            )
            if previous is None:
                item.update(
                    {
                        "previous_floor": np.nan,
                        "mean_net_change_vs_previous": np.nan,
                        "median_net_change_vs_previous": np.nan,
                        "hold_change_vs_previous": np.nan,
                    }
                )
            else:
                item.update(
                    {
                        "previous_floor": float(previous["retention_floor"]),
                        "mean_net_change_vs_previous": float(row["mean_net"] - previous["mean_net"]),
                        "median_net_change_vs_previous": float(row["median_net"] - previous["median_net"]),
                        "hold_change_vs_previous": float(row["mean_hold_minutes"] - previous["mean_hold_minutes"]),
                    }
                )
            rows.append(item)
            previous = row
    return pd.DataFrame(rows)


def _build_hold_plateau(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["direction", "impulse_window", "threshold", "retention_floor", "event_set"]
    for key, part in summary.groupby(keys, observed=False, dropna=False):
        ordered = part.sort_values("max_hold")
        previous: pd.Series | None = None
        for _, row in ordered.iterrows():
            item = {k: v for k, v in zip(keys, key if isinstance(key, tuple) else (key,), strict=False)}
            item.update(
                {
                    "max_hold": int(row["max_hold"]),
                    "events": int(row["events"]),
                    "mean_net": float(row["mean_net"]),
                    "median_net": float(row["median_net"]),
                    "profit_factor": float(row["profit_factor"]),
                    "mean_hold_minutes": float(row["mean_hold_minutes"]),
                    "decay_exit_rate": float(row["decay_exit_rate"]),
                    "mean_mfe": float(row["mean_mfe"]),
                    "mean_mfe_giveback": float(row["mean_mfe_giveback"]),
                    "positive_year_count": int(row["positive_year_count"]),
                    "total_year_count": int(row["total_year_count"]),
                }
            )
            if previous is None:
                item.update(
                    {
                        "previous_max_hold": np.nan,
                        "mean_net_change_vs_previous": np.nan,
                        "median_net_change_vs_previous": np.nan,
                    }
                )
            else:
                item.update(
                    {
                        "previous_max_hold": int(previous["max_hold"]),
                        "mean_net_change_vs_previous": float(row["mean_net"] - previous["mean_net"]),
                        "median_net_change_vs_previous": float(row["median_net"] - previous["median_net"]),
                    }
                )
            rows.append(item)
            previous = row
    return pd.DataFrame(rows)


def _build_brief(summary: pd.DataFrame, meta: dict[str, Any]) -> str:
    lines = [
        "# ETH Directional Impulse Continuation - Causal Impulse-Decay Exit",
        "",
        "> Automated descriptive brief. This is not an accepted Edge decision and not a final strategy backtest.",
        "",
        "## Research question",
        "",
        "After immediate next-open entry, can a causal exit based on decay of the same impulse retain enough favorable excursion to cover normal cost?",
        "",
        "## Exit rule",
        "",
        "- Entry: original impulse signal bar closes, then next 1m open.",
        "- State: same-window directional impulse recomputed after each fully closed holding bar.",
        "- Baseline: fixed historical volatility ending before the original impulse window.",
        "- Trigger: live impulse retention <= fixed retention floor.",
        "- Execution: next 1m open after trigger; otherwise next open after fixed max hold.",
        "",
        "## Best deduplicated rows by direction (not parameter selection)",
        "",
    ]
    if summary.empty:
        lines.append("No valid summary rows.")
    else:
        for direction in ("LONG", "SHORT"):
            part = summary[(summary["event_set"] == "deduplicated") & (summary["direction"] == direction)]
            lines.extend([f"### {direction}", ""])
            if part.empty:
                lines.extend(["No rows.", ""])
                continue
            best = part.sort_values(["mean_net", "events"], ascending=[False, False]).head(10)
            lines.append("| window | threshold | floor | max hold | events | mean hold | mean net | median net | PF | decay exits | positive years |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for _, row in best.iterrows():
                lines.append(
                    f"| {int(row['impulse_window'])}m | {row['threshold']:.2f} | {row['retention_floor']:.2f} | "
                    f"{int(row['max_hold'])}m | {int(row['events'])} | {row['mean_hold_minutes']:.2f}m | "
                    f"{row['mean_net']:.4%} | {row['median_net']:.4%} | {row['profit_factor']:.3f} | "
                    f"{row['decay_exit_rate']:.2%} | {int(row['positive_year_count'])}/{int(row['total_year_count'])} |"
                )
            lines.append("")
    lines.extend(
        [
            "## Decision rules",
            "",
            "A useful dynamic exit must improve cost-after mean and median without depending on one floor/hold point, retain reasonable frequency, reduce MFE giveback, and agree across years and nearby impulse thresholds.",
            "",
            "A high MFE capture ratio alone is insufficient if normal net remains negative or tail loss worsens.",
            "",
            "## Run facts",
            "",
            f"- Anchor trade rows written: {meta.get('anchor_trade_rows_written', 0):,}.",
            f"- Synthetic gap bars excluded: {meta.get('synthetic_gap_bar_count', 0):,}.",
        ]
    )
    return "\n".join(lines) + "\n"


def _update_log(log_path: Path, summary: pd.DataFrame) -> None:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Directional Impulse Continuation Research Log\n"
    marker = "## Round 06 — Causal impulse-decay exit"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n\n"
    lines = [
        marker,
        "",
        "### Research question",
        "",
        "After immediate next-open entry, can a causal exit based only on decay of the same directional impulse preserve enough of the transient favorable path to cover normal cost?",
        "",
        "### Changed from Round 05",
        "",
        "The delayed confirmation entry is removed. Entry returns to the Round-01 next-open event. The only new mechanism is a dynamic exit when live same-window impulse strength decays below a fixed fraction of original signal strength. No environment, order-flow, range-bar, footprint, volume, session or trend filter is stacked.",
        "",
        "### Fixed variants",
        "",
        "```text",
        f"retention floors = {', '.join(str(v) for v in DEFAULT_RETENTION_FLOORS)}",
        f"maximum holds    = {', '.join(str(v) + 'm' for v in DEFAULT_MAX_HOLDS)}",
        "```",
        "",
        "### Production result",
        "",
    ]
    if summary.empty:
        lines.append("Pending production run.")
    else:
        dedup = summary[summary["event_set"] == "deduplicated"]
        for direction in ("LONG", "SHORT"):
            part = dedup[dedup["direction"] == direction]
            if part.empty:
                continue
            row = part.sort_values(["mean_net", "events"], ascending=[False, False]).iloc[0]
            lines.append(
                f"- Best descriptive {direction}: {int(row['impulse_window'])}m impulse, threshold {row['threshold']:.2f}, "
                f"retention floor {row['retention_floor']:.2f}, max hold {int(row['max_hold'])}m, events={int(row['events'])}, "
                f"mean hold={row['mean_hold_minutes']:.2f}m, mean net={row['mean_net']:.4%}, median net={row['median_net']:.4%}, "
                f"PF={row['profit_factor']:.3f}, positive years={int(row['positive_year_count'])}/{int(row['total_year_count'])}."
            )
    lines.extend(
        [
            "",
            "### Boundary",
            "",
            "This round tests one price-only causal exit mechanism. It does not yet add risk sizing, a hard stop, higher-timeframe environment, trade-flow, range-bar or footprint state.",
            "",
        ]
    )
    log_path.write_text(text + "\n".join(lines), encoding="utf-8")


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    windows = r01._parse_int_csv(args.impulse_windows)
    thresholds = r01._parse_float_csv(args.thresholds)
    retention_floors = _parse_retention_floors(args.retention_floors)
    max_holds = r01._parse_int_csv(args.max_holds)
    if min(max_holds) <= 0:
        raise ValueError("max-holds must be positive")
    if int(args.vol_min_periods) > int(args.vol_lookback_bars):
        raise ValueError("vol-min-periods cannot exceed vol-lookback-bars")
    if float(ANCHOR_RETENTION_FLOOR) not in retention_floors or int(ANCHOR_MAX_HOLD) not in max_holds:
        raise ValueError("default anchor floor/hold must remain present for deterministic trade audit")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "05_trades.csv"
    audit_path = out_dir / "06_signal_audit.csv"
    for path in (trades_path, audit_path):
        path.unlink(missing_ok=True)

    validation = r01.validate_bars(bars, args)
    max_forward = max(max_holds) + 1
    masks = r02._eligible_masks(bars, args, (max_forward,))
    study_months = int(masks["study_months"])
    fee_cost = float(args.entry_fee_rate + args.exit_fee_rate)
    normal_cost = float(fee_cost + args.entry_slippage + args.exit_slippage)
    if not math.isclose(fee_cost, 0.0011, rel_tol=0.0, abs_tol=1e-12):
        print(f"[warning] fee-only cost differs from 0.11%: {fee_cost:.6%}", flush=True)
    if not math.isclose(normal_cost, 0.0015, rel_tol=0.0, abs_tol=1e-12):
        print(f"[warning] normal cost differs from 0.15%: {normal_cost:.6%}", flush=True)

    log_return, abs_price_change, historical_1m_vol = r01.build_base_volatility(
        bars, int(args.vol_lookback_bars), int(args.vol_min_periods)
    )
    event_count_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    event_id_cursor = 1
    anchor_rows_written = 0
    first_trade_write = True
    first_audit_write = True
    minimum_threshold = min(thresholds)
    n = len(bars)
    index = bars.index
    open_arr = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    observed = bars["source_bar_observed_flag"].to_numpy(dtype=bool)

    print("[feature build] base impulse + immediate-entry live decay paths", flush=True)
    progress = ProgressReporter(
        label="[event paths + summaries] direction/windows",
        total=len(windows) * 2,
        every=max(1, int(args.progress_every)),
        enabled=not args.no_progress,
    )
    done = 0
    for window in windows:
        features = r01.build_window_features(bars, window, log_return, abs_price_change, historical_1m_vol)
        norm = features.normalized_impulse
        pre_impulse_1m_vol = historical_1m_vol.shift(int(window)).to_numpy(dtype=float)
        for direction, side in (("LONG", 1), ("SHORT", -1)):
            directed_norm = float(side) * norm
            all_min_positions = np.flatnonzero(np.isfinite(directed_norm) & (directed_norm >= float(minimum_threshold)))
            eligible_positions = all_min_positions[masks["eligible"][all_min_positions]]
            if not eligible_positions.size:
                done += 1
                progress.update(done)
                continue

            threshold_masks: dict[float, np.ndarray] = {}
            dedup_masks: dict[float, np.ndarray] = {}
            for threshold in thresholds:
                all_t = all_min_positions[directed_norm[all_min_positions] >= float(threshold)]
                dedup_t = r01._deduplicate_positions(all_t, int(window))
                dedup_axis = np.zeros(n, dtype=bool)
                dedup_axis[all_t] = dedup_t
                threshold_mask = directed_norm[eligible_positions] >= float(threshold)
                dedup_mask = threshold_mask & dedup_axis[eligible_positions]
                threshold_masks[float(threshold)] = threshold_mask
                dedup_masks[float(threshold)] = dedup_mask
                raw_count = int(threshold_mask.sum())
                dedup_count = int(dedup_mask.sum())
                overlap = 1.0 - (dedup_count / raw_count) if raw_count else np.nan
                event_count_rows.extend(
                    [
                        {
                            "direction": direction,
                            "impulse_window": int(window),
                            "threshold": float(threshold),
                            "event_set": "raw",
                            "events": raw_count,
                            "events_per_month": float(raw_count / max(1, study_months)),
                            "overlap_ratio": overlap,
                            "cooldown_bars": int(window),
                        },
                        {
                            "direction": direction,
                            "impulse_window": int(window),
                            "threshold": float(threshold),
                            "event_set": "deduplicated",
                            "events": dedup_count,
                            "events_per_month": float(dedup_count / max(1, study_months)),
                            "overlap_ratio": overlap,
                            "cooldown_bars": int(window),
                        },
                    ]
                )

            event_frame = pd.DataFrame(
                {
                    "event_id": np.arange(event_id_cursor, event_id_cursor + len(eligible_positions), dtype=np.int64),
                    "direction": direction,
                    "side": int(side),
                    "impulse_window": int(window),
                    "minimum_pool_threshold": float(minimum_threshold),
                    "signal_bar_start": index[eligible_positions],
                    "signal_bar_end": index[eligible_positions] + r01.BAR_DELTA,
                    "signal_time": index[eligible_positions] + r01.BAR_DELTA,
                    "entry_time": index[eligible_positions + 1],
                    "expected_entry_time": index[eligible_positions] + r01.BAR_DELTA,
                    "entry_price": open_arr[eligible_positions + 1],
                    "expected_entry_price": pd.to_numeric(bars["open"], errors="coerce")
                    .reindex(index[eligible_positions] + r01.BAR_DELTA)
                    .to_numpy(dtype=float),
                    "signal_source_bar_observed_flag": observed[eligible_positions],
                    "impulse_source_window_observed_flag": features.source_window_valid[eligible_positions],
                    "full_exit_path_observed_flag": masks["full_forward_observed"][eligible_positions],
                    "impulse_return": features.impulse_return[eligible_positions],
                    "normalized_impulse": features.normalized_impulse[eligible_positions],
                    "directional_efficiency": features.directional_efficiency[eligible_positions],
                    "window_range_bps": features.window_range_bps[eligible_positions],
                    "window_realized_vol": features.window_realized_vol[eligible_positions],
                }
            )
            event_id_cursor += len(event_frame)
            event_frame["entry_not_next_open_flag"] = event_frame["entry_time"] != event_frame["expected_entry_time"]
            event_frame["entry_price_mismatch_flag"] = ~np.isclose(
                event_frame["entry_price"].to_numpy(dtype=float),
                event_frame["expected_entry_price"].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
                equal_nan=False,
            )
            event_frame["synthetic_bar_dependency_flag"] = ~(
                event_frame["signal_source_bar_observed_flag"].astype(bool)
                & event_frame["impulse_source_window_observed_flag"].astype(bool)
                & event_frame["full_exit_path_observed_flag"].astype(bool)
            )
            for threshold in thresholds:
                tag = _threshold_tag(threshold)
                event_frame[f"event_{tag}_flag"] = threshold_masks[float(threshold)]
                event_frame[f"deduplicated_{tag}_flag"] = dedup_masks[float(threshold)]

            outcomes = _build_decay_outcomes(
                bars,
                eligible_positions,
                side=int(side),
                window=int(window),
                directed_signal_z=directed_norm,
                pre_impulse_1m_vol=pre_impulse_1m_vol,
                retention_floors=retention_floors,
                max_holds=max_holds,
                chunk_size=int(args.path_chunk_size),
                label=f"[decay paths] {direction} {int(window)}m",
                progress_enabled=not args.no_progress,
            )

            for threshold in thresholds:
                for event_set, mask in (
                    ("raw", threshold_masks[float(threshold)]),
                    ("deduplicated", dedup_masks[float(threshold)]),
                ):
                    for floor in retention_floors:
                        for hold in max_holds:
                            s_rows, y_rows, m_rows, t_rows = _summary_for_variant(
                                event_frame,
                                mask=mask,
                                outcome=outcomes[(float(floor), int(hold))],
                                direction=direction,
                                window=int(window),
                                threshold=float(threshold),
                                retention_floor=float(floor),
                                max_hold=int(hold),
                                event_set=event_set,
                                study_months=study_months,
                                fee_cost=fee_cost,
                                normal_cost=normal_cost,
                            )
                            summary_rows.extend(s_rows)
                            yearly_rows.extend(y_rows)
                            monthly_rows.extend(m_rows)
                            top_rows.extend(t_rows)

            if not args.skip_trades_csv:
                min_dedup = dedup_masks[float(minimum_threshold)]
                idx = np.flatnonzero(min_dedup)
                anchor = outcomes[(float(ANCHOR_RETENTION_FLOOR), int(ANCHOR_MAX_HOLD))]
                exit_minute = anchor["exit_minute"][idx].astype(np.int64)
                exit_pos = eligible_positions[idx] + exit_minute + 1
                compact = event_frame.iloc[idx].copy()
                compact["anchor_retention_floor"] = float(ANCHOR_RETENTION_FLOOR)
                compact["anchor_max_hold"] = int(ANCHOR_MAX_HOLD)
                compact["exit_minute"] = exit_minute
                compact["exit_time"] = index[exit_pos]
                compact["expected_exit_time"] = pd.to_datetime(compact["entry_time"]) + pd.to_timedelta(exit_minute, unit="m")
                compact["exit_price"] = open_arr[exit_pos]
                compact["exit_reason"] = [REASON_LABELS[int(x)] for x in anchor["reason_code"][idx]]
                compact["exit_retention"] = anchor["exit_retention"][idx]
                compact["gross_return"] = anchor["gross"][idx]
                compact["fee_only_net_return"] = anchor["gross"][idx] - fee_cost
                compact["normal_net_return"] = anchor["gross"][idx] - normal_cost
                compact["mfe"] = anchor["mfe"][idx]
                compact["mae"] = anchor["mae"][idx]
                compact["exit_not_next_open_flag"] = pd.to_datetime(compact["exit_time"]) != pd.to_datetime(compact["expected_exit_time"])
                compact["event_storage_scope"] = "minimum_threshold_deduplicated_anchor_variant"
                anchor_rows_written += int(len(compact))
                r01._write_stream_csv(compact, trades_path, first_write=first_trade_write)
                first_trade_write = False
                audit = compact[
                    [
                        "event_id",
                        "direction",
                        "impulse_window",
                        "signal_time",
                        "entry_time",
                        "expected_entry_time",
                        "entry_not_next_open_flag",
                        "entry_price_mismatch_flag",
                        "exit_time",
                        "expected_exit_time",
                        "exit_not_next_open_flag",
                        "signal_source_bar_observed_flag",
                        "impulse_source_window_observed_flag",
                        "full_exit_path_observed_flag",
                        "synthetic_bar_dependency_flag",
                    ]
                ].copy()
                audit["lookahead_flag"] = (
                    audit["entry_not_next_open_flag"].astype(bool)
                    | audit["entry_price_mismatch_flag"].astype(bool)
                    | audit["exit_not_next_open_flag"].astype(bool)
                    | ~audit["signal_source_bar_observed_flag"].astype(bool)
                    | ~audit["impulse_source_window_observed_flag"].astype(bool)
                    | ~audit["full_exit_path_observed_flag"].astype(bool)
                    | audit["synthetic_bar_dependency_flag"].astype(bool)
                )
                r01._write_stream_csv(audit, audit_path, first_write=first_audit_write)
                first_audit_write = False
                del compact, audit
            del event_frame, outcomes
            done += 1
            progress.update(done)
        del features
    progress.close()

    print("[summaries] assembling decay-exit plateaus and dependencies", flush=True)
    event_counts = pd.DataFrame(event_count_rows)
    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    monthly = pd.DataFrame(monthly_rows)
    exit_reason = summary[
        [
            "direction",
            "impulse_window",
            "threshold",
            "retention_floor",
            "max_hold",
            "event_set",
            "events",
            "events_per_month",
            "mean_hold_minutes",
            "median_hold_minutes",
            "decay_exit_rate",
            "max_hold_exit_rate",
            "gross_positive_rate",
            "mean_positive_mfe_capture_ratio",
            "median_positive_mfe_capture_ratio",
            "mean_mfe",
            "mean_mae",
            "mean_mfe_giveback",
            "mean_net",
            "median_net",
            "profit_factor",
        ]
    ].copy() if not summary.empty else pd.DataFrame()
    floor_plateau = _build_floor_plateau(summary)
    hold_plateau = _build_hold_plateau(summary)
    top_dependency = pd.DataFrame(top_rows)

    if first_trade_write:
        pd.DataFrame(columns=["event_id", "direction", "impulse_window"]).to_csv(trades_path, index=False)
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
        "retention_floors": list(retention_floors),
        "max_holds": list(max_holds),
        "entry_policy": "signal bar fully closes; enter next 1m open",
        "live_state": "same-window directional impulse z using fixed pre-signal historical volatility baseline",
        "retention_definition": "live_directional_impulse_z / original_directional_impulse_z",
        "exit_policy": "after a fully closed bar, exit next 1m open when retention <= floor; otherwise next open after max hold",
        "anchor_trade_audit": {
            "retention_floor": ANCHOR_RETENTION_FLOOR,
            "max_hold": ANCHOR_MAX_HOLD,
            "scope": "minimum-threshold deduplicated events",
        },
        "round02_persistence_condition_stacked": False,
        "round03_notional_condition_stacked": False,
        "round04_barrier_condition_stacked": False,
        "round05_confirmation_condition_stacked": False,
        "higher_timeframe_filter_stacked": False,
        "order_flow_filter_stacked": False,
        "range_bar_filter_stacked": False,
        "footprint_filter_stacked": False,
        "deduplication": "threshold-specific original impulse stream; cooldown=impulse_window",
        "fee_only_cost": fee_cost,
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
        "anchor_trade_rows_written": int(anchor_rows_written),
        "trades_csv_skipped_for_development": bool(args.skip_trades_csv),
        "validation": validation,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "research_boundary": "one mechanism only: immediate entry plus causal same-impulse decay exit",
    }

    brief = _build_brief(summary, meta)
    artifacts = [
        (event_counts, out_dir / "01_event_counts.csv"),
        (summary, out_dir / "02_exit_summary.csv"),
        (yearly, out_dir / "03_yearly.csv"),
        (monthly, out_dir / "04_monthly.csv"),
        (exit_reason, out_dir / "07_exit_reason_summary.csv"),
        (floor_plateau, out_dir / "08_retention_floor_plateau.csv"),
        (hold_plateau, out_dir / "09_max_hold_plateau.csv"),
        (top_dependency, out_dir / "10_top_trade_dependency.csv"),
    ]
    print("[artifacts] writing report files", flush=True)
    with ProgressReporter(label="[artifacts] tables", total=len(artifacts) + 3, every=1, enabled=not args.no_progress) as p:
        count = 0
        for frame, path in artifacts:
            frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
            count += 1
            p.update(count)
        r01._write_json(meta, out_dir / "11_run_meta.json")
        count += 1
        p.update(count)
        (out_dir / "12_research_brief.md").write_text(brief, encoding="utf-8")
        count += 1
        p.update(count)
        _update_log(Path(__file__).resolve().with_name("00_research_log.md"), summary)
        count += 1
        p.update(count)

    if not args.skip_review_pack:
        print("[review pack] packaging summary artifacts", flush=True)
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report_dir={out_dir}", flush=True)
    return {"report_dir": out_dir, "trades": trades_path, "audit": audit_path, "review_pack": out_dir / "gpt_review_pack.zip"}


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] deterministic causal impulse-decay exit study", flush=True)
    test_mask = np.asarray([[False, True, True], [False, False, False], [True, False, False]], dtype=bool)
    if _first_true_minute(test_mask).tolist() != [2, 0, 1]:
        raise AssertionError("first-true kernel failed")

    raw = r01._synthetic_bars()
    raw = raw.drop(raw.index[3700:3707])
    bars = r01._regularize_trade_bar_axis(raw)
    log_path = Path(__file__).resolve().with_name("00_research_log.md")
    original_log = log_path.read_text(encoding="utf-8") if log_path.exists() else None
    try:
        with tempfile.TemporaryDirectory(prefix="dic_r06_") as tmp:
            args.out_dir = tmp
            args.warmup_start_date = "2022-12-20"
            args.start_date = "2022-12-23"
            args.end_date = "2022-12-24"
            args.impulse_windows = "1,3"
            args.thresholds = "0.5,1.0"
            args.retention_floors = "0.0,0.5"
            args.max_holds = "5,30"
            args.vol_lookback_bars = 720
            args.vol_min_periods = 360
            args.path_chunk_size = 256
            args.skip_review_pack = True
            args.no_progress = True
            result = run_research(bars, args)
            required = [
                "01_event_counts.csv",
                "02_exit_summary.csv",
                "03_yearly.csv",
                "04_monthly.csv",
                "05_trades.csv",
                "06_signal_audit.csv",
                "07_exit_reason_summary.csv",
                "08_retention_floor_plateau.csv",
                "09_max_hold_plateau.csv",
                "10_top_trade_dependency.csv",
                "11_run_meta.json",
                "12_research_brief.md",
            ]
            missing = [name for name in required if not (result["report_dir"] / name).exists()]
            if missing:
                raise AssertionError(f"self-test missing artifacts: {missing}")
            summary = pd.read_csv(result["report_dir"] / "02_exit_summary.csv")
            if summary.empty:
                raise AssertionError("self-test summary is empty")
            if not set(summary["retention_floor"].round(2)) == {0.0, 0.5}:
                raise AssertionError("self-test retention floors missing")
            if not set(summary["max_hold"].astype(int)) == {5, 30}:
                raise AssertionError("self-test max holds missing")
            trades = pd.read_csv(result["report_dir"] / "05_trades.csv")
            if trades.empty:
                raise AssertionError("self-test trades are empty")
            if not (pd.to_datetime(trades["entry_time"]) == pd.to_datetime(trades["expected_entry_time"])).all():
                raise AssertionError("self-test entry is not next open")
            if not (pd.to_datetime(trades["exit_time"]) == pd.to_datetime(trades["expected_exit_time"])).all():
                raise AssertionError("self-test exit is not next open after closed trigger/hold bar")
            audit = pd.read_csv(result["report_dir"] / "06_signal_audit.csv")
            if not audit.empty and audit["lookahead_flag"].astype(bool).any():
                raise AssertionError("self-test causal audit contains invalid events")
            if not trades["exit_minute"].between(1, ANCHOR_MAX_HOLD).all():
                raise AssertionError("self-test exit minute outside allowed hold")
            meta = json.loads((result["report_dir"] / "11_run_meta.json").read_text(encoding="utf-8"))
            if int(meta.get("synthetic_gap_bar_count", 0)) != 7:
                raise AssertionError("self-test did not preserve expected gap bars")
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
