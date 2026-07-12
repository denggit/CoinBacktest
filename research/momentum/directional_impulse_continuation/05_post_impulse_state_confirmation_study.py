#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Directional Impulse Continuation: post-impulse state confirmation (round 05).

Research question
-----------------
After a basic directional impulse is confirmed, does the first 1/3/5 minutes of
post-signal directional progress identify whether meaningful continuation still
remains, when entry is delayed causally until the next bar open after the
checkpoint?

This round is an event/state study, not a final strategy backtest. It branches
from the Round-01 base impulse and does not stack Round-02 persistence,
Round-03 notional expansion, or Round-04 barrier rules.

Causal policy
-------------
- Original impulse is confirmed after signal bar t closes.
- Checkpoint k uses only bars t+1..t+k after they have fully closed.
- A hypothetical confirmation entry executes at bar t+k+1 open.
- The checkpoint state denominator is a historical volatility estimate ending
  before the original impulse window; neither the impulse nor checkpoint bars
  enter the normalization baseline.
- Remaining forward return/MFE/MAE starts from the confirmation entry.
- Synthetic trade-bar gap rows are excluded from signal, checkpoint, entry and
  remaining path.

Performance policy
------------------
- Local OKX 1m trade bars only; build_missing=False.
- One load, one volatility build, one feature build per impulse window.
- Full-axis future high/low arrays are built once per distinct horizon and
  reused for every threshold, state bucket, year and month.
- Nested thresholds reuse one minimum-threshold event pool.
- No iterrows scan over market bars and no per-event future-path loop.
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
    spec = importlib.util.spec_from_file_location("directional_impulse_round04_for_r05", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared round-04 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r04 = _load_round04_module()
r02 = r04.r02
r01 = r04.r01

SCRIPT_NAME = "05_post_impulse_state_confirmation_study"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION_R05"
EDGE_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION"
TITLE = "ETH Directional Impulse Continuation - Post-Impulse State Confirmation"
DEFAULT_OUT_DIR = (
    "data/reports/research/momentum/directional_impulse_continuation/"
    "05_post_impulse_state_confirmation_study"
)

DEFAULT_WINDOWS = (1, 3, 5, 10, 15)
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
DEFAULT_CHECKPOINTS = (1, 3, 5)
DEFAULT_HORIZONS = (3, 5, 10, 15, 30, 60)

STATE_BANDS = (
    ("reversal_or_flat_le_0", -np.inf, 0.0),
    ("weak_continuation_0_0.5", 0.0, 0.5),
    ("moderate_continuation_0.5_1.0", 0.5, 1.0),
    ("strong_continuation_ge_1.0", 1.0, np.inf),
)
STATE_ORDER = {name: idx for idx, (name, _, _) in enumerate(STATE_BANDS)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal ETH post-impulse state confirmation study.",
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
    p.add_argument("--checkpoints", default=",".join(map(str, DEFAULT_CHECKPOINTS)))
    p.add_argument("--horizons", default=",".join(map(str, DEFAULT_HORIZONS)))
    p.add_argument("--vol-lookback-bars", type=int, default=1440)
    p.add_argument("--vol-min-periods", type=int, default=720)
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


def _state_labels(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    labels = np.full(len(x), "invalid", dtype=object)
    valid = np.isfinite(x)
    labels[valid & (x <= 0.0)] = "reversal_or_flat_le_0"
    labels[valid & (x > 0.0) & (x < 0.5)] = "weak_continuation_0_0.5"
    labels[valid & (x >= 0.5) & (x < 1.0)] = "moderate_continuation_0.5_1.0"
    labels[valid & (x >= 1.0)] = "strong_continuation_ge_1.0"
    return labels


def _checkpoint_arrays(
    bars: pd.DataFrame,
    positions: np.ndarray,
    *,
    side: int,
    checkpoint: int,
    pre_impulse_1m_vol: np.ndarray,
    path_cache: dict[int, Any],
) -> dict[str, np.ndarray]:
    p = positions.astype(np.int64, copy=False)
    k = int(checkpoint)
    open_arr = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    entry_price = open_arr[p + 1]
    checkpoint_close = close_arr[p + k]
    progress = float(side) * (checkpoint_close / entry_price - 1.0)
    denom = pre_impulse_1m_vol[p] * math.sqrt(k)
    progress_z = progress / np.where(np.isfinite(denom) & (denom > 0.0), denom, np.nan)

    future_high = path_cache[k].future_high[p]
    future_low = path_cache[k].future_low[p]
    if int(side) == 1:
        checkpoint_mfe = future_high / entry_price - 1.0
        checkpoint_mae = future_low / entry_price - 1.0
    else:
        checkpoint_mfe = 1.0 - future_low / entry_price
        checkpoint_mae = 1.0 - future_high / entry_price
    retention = progress / np.where(checkpoint_mfe > 0.0, checkpoint_mfe, np.nan)

    q = p + k
    confirmed_entry_pos = q + 1
    index = bars.index
    confirmed_entry_time = index[confirmed_entry_pos].to_numpy(dtype="datetime64[ns]")
    expected_entry_time = (
        index[p].to_numpy(dtype="datetime64[ns]")
        + np.timedelta64(k + 1, "m")
    )
    return {
        "checkpoint_progress": progress,
        "checkpoint_progress_z": progress_z,
        "checkpoint_mfe": checkpoint_mfe,
        "checkpoint_mae": checkpoint_mae,
        "checkpoint_mfe_retention": retention,
        "state_bucket": _state_labels(progress_z),
        "checkpoint_pos": q,
        "confirmed_entry_pos": confirmed_entry_pos,
        "confirmed_entry_time": confirmed_entry_time,
        "expected_confirmed_entry_time": expected_entry_time,
        "confirmed_entry_price": open_arr[confirmed_entry_pos],
    }


def _remaining_outcomes(
    bars: pd.DataFrame,
    checkpoint_pos: np.ndarray,
    *,
    side: int,
    horizons: tuple[int, ...],
    path_cache: dict[int, Any],
    fee_cost: float,
    normal_cost: float,
) -> dict[int, dict[str, np.ndarray]]:
    q = checkpoint_pos.astype(np.int64, copy=False)
    open_arr = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    entry = open_arr[q + 1]
    out: dict[int, dict[str, np.ndarray]] = {}
    for horizon in horizons:
        h = int(horizon)
        exit_price = close_arr[q + h]
        gross = float(side) * (exit_price / entry - 1.0)
        high = path_cache[h].future_high[q]
        low = path_cache[h].future_low[q]
        if int(side) == 1:
            mfe = high / entry - 1.0
            mae = low / entry - 1.0
        else:
            mfe = 1.0 - low / entry
            mae = 1.0 - high / entry
        out[h] = {
            "gross": gross,
            "fee": gross - float(fee_cost),
            "net": gross - float(normal_cost),
            "mfe": mfe,
            "mae": mae,
        }
    return out


def _top_rows(
    *,
    event_frame: pd.DataFrame,
    indices: np.ndarray,
    net: np.ndarray,
    direction: str,
    window: int,
    threshold: float,
    checkpoint: int,
    state_bucket: str,
    horizon: int,
    event_set: str,
) -> list[dict[str, Any]]:
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
                "checkpoint": int(checkpoint),
                "state_bucket": state_bucket,
                "horizon": int(horizon),
                "event_set": event_set,
                "rank": rank,
                "event_id": int(event_frame.iloc[idx]["event_id"]),
                "signal_time": event_frame.iloc[idx]["signal_time"],
                "confirmed_entry_time": event_frame.iloc[idx][f"confirmed_entry_time_{int(checkpoint)}m"],
                "normal_net_return": float(net[idx]),
                "contribution_to_positive_return": contribution,
                "cumulative_top_contribution": cumulative,
            }
        )
    return rows


def _summary_for_mask(
    event_frame: pd.DataFrame,
    *,
    indices: np.ndarray,
    outcomes: dict[int, dict[str, np.ndarray]],
    direction: str,
    window: int,
    threshold: float,
    checkpoint: int,
    state_bucket: str,
    event_set: str,
    horizons: tuple[int, ...],
    study_months: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []

    entry_times = pd.to_datetime(event_frame[f"confirmed_entry_time_{int(checkpoint)}m"])
    years = entry_times.dt.year.to_numpy()
    months = entry_times.dt.to_period("M").astype(str).to_numpy()
    mask = np.zeros(len(event_frame), dtype=bool)
    mask[indices] = True
    year_groups = r01._group_indices(years, mask)
    month_groups = r01._group_indices(months, mask)

    for horizon in horizons:
        h = int(horizon)
        arr = outcomes[h]
        stat = r01._stats(
            arr["gross"][indices], arr["fee"][indices], arr["net"][indices], arr["mfe"][indices], arr["mae"][indices]
        )
        year_means: dict[int, float] = {}
        for year, idx in year_groups.items():
            period = r01._stats(arr["gross"][idx], arr["fee"][idx], arr["net"][idx], arr["mfe"][idx], arr["mae"][idx])
            yearly_rows.append(
                {
                    "direction": direction,
                    "impulse_window": int(window),
                    "threshold": float(threshold),
                    "checkpoint": int(checkpoint),
                    "state_bucket": state_bucket,
                    "horizon": h,
                    "event_set": event_set,
                    "year": int(year),
                    **period,
                }
            )
            year_means[int(year)] = float(period["mean_net"])
        for month, idx in month_groups.items():
            period = r01._stats(arr["gross"][idx], arr["fee"][idx], arr["net"][idx], arr["mfe"][idx], arr["mae"][idx])
            monthly_rows.append(
                {
                    "direction": direction,
                    "impulse_window": int(window),
                    "threshold": float(threshold),
                    "checkpoint": int(checkpoint),
                    "state_bucket": state_bucket,
                    "horizon": h,
                    "event_set": event_set,
                    "month": str(month),
                    **period,
                }
            )
        finite_years = {y: v for y, v in year_means.items() if math.isfinite(v)}
        worst_year = min(finite_years, key=finite_years.get) if finite_years else None
        summary_rows.append(
            {
                "direction": direction,
                "impulse_window": int(window),
                "threshold": float(threshold),
                "checkpoint": int(checkpoint),
                "state_bucket": state_bucket,
                "horizon": h,
                "event_set": event_set,
                "events_per_month": float(stat["events"] / max(1, study_months)),
                **stat,
                "positive_year_count": int(sum(v > 0.0 for v in finite_years.values())),
                "total_year_count": int(len(finite_years)),
                "worst_year": worst_year,
                "worst_year_mean_net": finite_years.get(worst_year, np.nan) if worst_year is not None else np.nan,
            }
        )
        top_rows.extend(
            _top_rows(
                event_frame=event_frame,
                indices=indices,
                net=arr["net"],
                direction=direction,
                window=window,
                threshold=threshold,
                checkpoint=checkpoint,
                state_bucket=state_bucket,
                horizon=h,
                event_set=event_set,
            )
        )
    return summary_rows, yearly_rows, monthly_rows, top_rows


def _build_state_gradient(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["direction", "impulse_window", "threshold", "checkpoint", "horizon", "event_set"]
    valid_bands = list(STATE_ORDER)
    for key, part in summary[summary["state_bucket"].isin(valid_bands)].groupby(keys, observed=False, dropna=False):
        ordered = part.assign(_order=part["state_bucket"].map(STATE_ORDER)).sort_values("_order")
        total = float(ordered["events"].sum())
        previous: pd.Series | None = None
        for _, row in ordered.iterrows():
            item = {k: v for k, v in zip(keys, key if isinstance(key, tuple) else (key,), strict=False)}
            item.update(
                {
                    "state_bucket": row["state_bucket"],
                    "events": int(row["events"]),
                    "share_of_valid_events": float(row["events"] / total) if total > 0 else np.nan,
                    "mean_gross": float(row["mean_gross"]),
                    "median_gross": float(row["median_gross"]),
                    "mean_net": float(row["mean_net"]),
                    "median_net": float(row["median_net"]),
                    "profit_factor": float(row["profit_factor"]) if pd.notna(row["profit_factor"]) else np.nan,
                    "mean_mfe": float(row["mean_mfe"]),
                    "mean_mae": float(row["mean_mae"]),
                    "positive_year_count": int(row["positive_year_count"]),
                    "total_year_count": int(row["total_year_count"]),
                }
            )
            if previous is None:
                item.update(
                    {
                        "previous_bucket": None,
                        "mean_net_change_vs_previous": np.nan,
                        "median_net_change_vs_previous": np.nan,
                        "profit_factor_change_vs_previous": np.nan,
                    }
                )
            else:
                item.update(
                    {
                        "previous_bucket": previous["state_bucket"],
                        "mean_net_change_vs_previous": float(row["mean_net"] - previous["mean_net"]),
                        "median_net_change_vs_previous": float(row["median_net"] - previous["median_net"]),
                        "profit_factor_change_vs_previous": float(row["profit_factor"] - previous["profit_factor"]),
                    }
                )
            rows.append(item)
            previous = row
    return pd.DataFrame(rows)


def _build_threshold_plateau(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["direction", "impulse_window", "checkpoint", "state_bucket", "horizon", "event_set"]
    for key, part in summary.groupby(keys, observed=False, dropna=False):
        ordered = part.sort_values("threshold")
        previous: pd.Series | None = None
        for _, row in ordered.iterrows():
            item = {k: v for k, v in zip(keys, key if isinstance(key, tuple) else (key,), strict=False)}
            item.update(
                {
                    "threshold": float(row["threshold"]),
                    "events": int(row["events"]),
                    "events_per_month": float(row["events_per_month"]),
                    "mean_net": float(row["mean_net"]),
                    "median_net": float(row["median_net"]),
                    "profit_factor": float(row["profit_factor"]) if pd.notna(row["profit_factor"]) else np.nan,
                    "positive_year_count": int(row["positive_year_count"]),
                    "total_year_count": int(row["total_year_count"]),
                }
            )
            if previous is None:
                item.update(
                    {
                        "previous_threshold": np.nan,
                        "event_retention_vs_previous": np.nan,
                        "mean_net_change_vs_previous": np.nan,
                        "median_net_change_vs_previous": np.nan,
                    }
                )
            else:
                prev_events = float(previous["events"])
                item.update(
                    {
                        "previous_threshold": float(previous["threshold"]),
                        "event_retention_vs_previous": float(row["events"] / prev_events) if prev_events > 0 else np.nan,
                        "mean_net_change_vs_previous": float(row["mean_net"] - previous["mean_net"]),
                        "median_net_change_vs_previous": float(row["median_net"] - previous["median_net"]),
                    }
                )
            rows.append(item)
            previous = row
    return pd.DataFrame(rows)


def _build_transition_rows(
    event_frame: pd.DataFrame,
    *,
    direction: str,
    window: int,
    thresholds: tuple[float, ...],
    checkpoints: tuple[int, ...],
) -> list[dict[str, Any]]:
    if len(checkpoints) < 2 or event_frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        tag = _threshold_tag(threshold)
        for event_set, col in (("raw", f"event_{tag}_flag"), ("deduplicated", f"deduplicated_{tag}_flag")):
            base = event_frame[col].astype(bool).to_numpy()
            for a, b in zip(checkpoints[:-1], checkpoints[1:], strict=False):
                from_labels = event_frame[f"state_bucket_{int(a)}m"].to_numpy(dtype=object)
                to_labels = event_frame[f"state_bucket_{int(b)}m"].to_numpy(dtype=object)
                valid = base & np.isin(from_labels, list(STATE_ORDER)) & np.isin(to_labels, list(STATE_ORDER))
                total = int(valid.sum())
                for source in STATE_ORDER:
                    source_mask = valid & (from_labels == source)
                    source_total = int(source_mask.sum())
                    for target in STATE_ORDER:
                        count = int((source_mask & (to_labels == target)).sum())
                        rows.append(
                            {
                                "direction": direction,
                                "impulse_window": int(window),
                                "threshold": float(threshold),
                                "event_set": event_set,
                                "from_checkpoint": int(a),
                                "to_checkpoint": int(b),
                                "from_state": source,
                                "to_state": target,
                                "events": count,
                                "share_of_all_transitions": float(count / total) if total else np.nan,
                                "conditional_transition_rate": float(count / source_total) if source_total else np.nan,
                                "from_state_events": source_total,
                            }
                        )
    return rows


def _build_signal_audit(events: pd.DataFrame, checkpoints: tuple[int, ...]) -> pd.DataFrame:
    cols = [
        "event_id",
        "direction",
        "impulse_window",
        "signal_time",
        "signal_bar_start",
        "signal_bar_end",
        "base_entry_time",
        "base_expected_entry_time",
        "base_entry_not_next_open_flag",
        "base_entry_price_mismatch_flag",
        "signal_source_bar_observed_flag",
        "impulse_source_window_observed_flag",
        "full_confirmation_path_observed_flag",
        "synthetic_bar_dependency_flag",
    ]
    audit = events[cols].copy()
    lookahead = (
        audit["base_entry_not_next_open_flag"].astype(bool)
        | audit["base_entry_price_mismatch_flag"].astype(bool)
        | ~audit["signal_source_bar_observed_flag"].astype(bool)
        | ~audit["impulse_source_window_observed_flag"].astype(bool)
        | ~audit["full_confirmation_path_observed_flag"].astype(bool)
        | audit["synthetic_bar_dependency_flag"].astype(bool)
    )
    for checkpoint in checkpoints:
        k = int(checkpoint)
        names = [
            f"checkpoint_bar_end_{k}m",
            f"confirmed_entry_time_{k}m",
            f"expected_confirmed_entry_time_{k}m",
            f"confirmed_entry_not_next_open_flag_{k}m",
            f"confirmed_entry_price_mismatch_flag_{k}m",
            f"checkpoint_state_valid_flag_{k}m",
        ]
        for name in names:
            audit[name] = events[name].to_numpy()
        lookahead = (
            lookahead
            | audit[f"confirmed_entry_not_next_open_flag_{k}m"].astype(bool)
            | audit[f"confirmed_entry_price_mismatch_flag_{k}m"].astype(bool)
            | ~audit[f"checkpoint_state_valid_flag_{k}m"].astype(bool)
        )
    audit["lookahead_flag"] = lookahead
    return audit


def _build_brief(summary: pd.DataFrame, meta: dict[str, Any]) -> str:
    lines = [
        "# ETH Directional Impulse Continuation - Post-Impulse State Confirmation",
        "",
        "> Automated descriptive brief. This is not an accepted Edge decision and not a final strategy backtest.",
        "",
        "## Research question",
        "",
        "After a basic impulse, does causal 1m/3m/5m post-signal progress identify events with remaining continuation after a delayed next-open entry?",
        "",
        "## Causal interpretation",
        "",
        "- The checkpoint bar must close before its state is known.",
        "- Confirmation entry is the next 1m bar open after the checkpoint.",
        "- State normalization uses historical volatility ending before the original impulse window.",
        "- Results measure remaining return/MFE/MAE after confirmation, not the already-realized checkpoint move.",
        "",
        "## Fixed state bands",
        "",
    ]
    for name, lower, upper in STATE_BANDS:
        lines.append(f"- `{name}`: ({lower}, {upper})")
    lines.extend(["", "## Best deduplicated rows by direction (not parameter selection)", ""])
    if summary.empty:
        lines.append("No valid summary rows.")
    else:
        for direction in ("LONG", "SHORT"):
            part = summary[(summary["event_set"] == "deduplicated") & (summary["state_bucket"] != "ALL") & (summary["direction"] == direction)]
            lines.extend([f"### {direction}", ""])
            if part.empty:
                lines.append("No rows.")
                lines.append("")
                continue
            best = part.sort_values(["mean_net", "events"], ascending=[False, False]).head(10)
            lines.append("| window | threshold | checkpoint | state | horizon | events | mean net | median net | PF | positive years |")
            lines.append("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
            for _, row in best.iterrows():
                lines.append(
                    f"| {int(row['impulse_window'])}m | {row['threshold']:.2f} | {int(row['checkpoint'])}m | "
                    f"{row['state_bucket']} | {int(row['horizon'])}m | {int(row['events'])} | "
                    f"{row['mean_net']:.4%} | {row['median_net']:.4%} | {row['profit_factor']:.3f} | "
                    f"{int(row['positive_year_count'])}/{int(row['total_year_count'])} |"
                )
            lines.append("")
    lines.extend(
        [
            "## Decision rules",
            "",
            "A useful confirmation state should improve mean and median net monotonically across adjacent state bands, remain positive after 0.15% normal cost, retain reasonable monthly frequency, and agree across years and nearby impulse thresholds/checkpoints.",
            "",
            "A strong checkpoint move is not itself profit: the reported outcome begins only at the next-open confirmation entry.",
            "",
            "## Run facts",
            "",
            f"- Compact minimum-threshold deduplicated event rows written: {meta.get('compact_event_rows_written', 0):,}.",
            f"- Synthetic gap bars excluded: {meta.get('synthetic_gap_bar_count', 0):,}.",
        ]
    )
    return "\n".join(lines) + "\n"


def _update_log(log_path: Path, summary: pd.DataFrame, meta: dict[str, Any]) -> None:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Directional Impulse Continuation Research Log\n"
    marker = "## Round 05 — Post-impulse state confirmation"
    if marker in text:
        text = text.split(marker)[0].rstrip() + "\n\n"
    lines = [
        marker,
        "",
        "### Research question",
        "",
        "After the original impulse closes, can the first 1m/3m/5m of causal post-signal progress distinguish events that still have enough remaining continuation to trade after a next-open confirmation entry?",
        "",
        "### Changed from Round 04",
        "",
        "Symmetric TP/SL barriers are removed. This round delays entry until after a fully closed post-signal checkpoint and studies remaining return/MFE/MAE by normalized continuation state. No prior persistence, notional, trend, range-bar or footprint condition is stacked.",
        "",
        "### Production result",
        "",
    ]
    if summary.empty:
        lines.append("Pending production run.")
    else:
        dedup = summary[(summary["event_set"] == "deduplicated") & (summary["state_bucket"] != "ALL")]
        for direction in ("LONG", "SHORT"):
            part = dedup[dedup["direction"] == direction]
            if part.empty:
                continue
            row = part.sort_values(["mean_net", "events"], ascending=[False, False]).iloc[0]
            lines.append(
                f"- Best descriptive {direction}: {int(row['impulse_window'])}m impulse, threshold {row['threshold']:.2f}, "
                f"checkpoint {int(row['checkpoint'])}m, `{row['state_bucket']}`, remaining horizon {int(row['horizon'])}m, "
                f"events={int(row['events'])}, mean net={row['mean_net']:.4%}, median net={row['median_net']:.4%}, "
                f"PF={row['profit_factor']:.3f}, positive years={int(row['positive_year_count'])}/{int(row['total_year_count'])}."
            )
    lines.extend(
        [
            "",
            "### Boundary",
            "",
            "This round validates only post-signal price-state confirmation. It does not yet define a dynamic exit and does not use trade-flow, range-bar or footprint features.",
            "",
        ]
    )
    log_path.write_text(text + "\n".join(lines), encoding="utf-8")


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    windows = r01._parse_int_csv(args.impulse_windows)
    thresholds = r01._parse_float_csv(args.thresholds)
    checkpoints = r01._parse_int_csv(args.checkpoints)
    horizons = r01._parse_int_csv(args.horizons)
    if min(checkpoints) <= 0 or min(horizons) <= 0:
        raise ValueError("checkpoints and horizons must be positive")
    if int(args.vol_min_periods) > int(args.vol_lookback_bars):
        raise ValueError("vol-min-periods cannot exceed vol-lookback-bars")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    events_path = out_dir / "05_events.csv"
    audit_path = out_dir / "06_signal_audit.csv"
    for path in (events_path, audit_path):
        path.unlink(missing_ok=True)

    validation = r01.validate_bars(bars, args)
    max_forward = max(checkpoints) + max(horizons)
    masks = r02._eligible_masks(bars, args, (max_forward,))
    study_months = int(masks["study_months"])
    fee_only_cost = float(args.entry_fee_rate + args.exit_fee_rate)
    normal_cost = float(fee_only_cost + args.entry_slippage + args.exit_slippage)
    if not math.isclose(fee_only_cost, 0.0011, rel_tol=0.0, abs_tol=1e-12):
        print(f"[warning] fee-only cost differs from 0.11%: {fee_only_cost:.6%}", flush=True)
    if not math.isclose(normal_cost, 0.0015, rel_tol=0.0, abs_tol=1e-12):
        print(f"[warning] normal cost differs from 0.15%: {normal_cost:.6%}", flush=True)

    required_path_horizons = tuple(sorted(set(checkpoints) | set(horizons)))
    path_cache = r01.build_path_cache(bars, required_path_horizons, progress_enabled=not args.no_progress)
    log_return, abs_price_change, historical_1m_vol = r01.build_base_volatility(
        bars, int(args.vol_lookback_bars), int(args.vol_min_periods)
    )

    event_count_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    event_id_cursor = 1
    compact_rows_written = 0
    first_event_write = True
    first_audit_write = True
    minimum_threshold = min(thresholds)
    n = len(bars)
    index = bars.index
    open_arr = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    observed = bars["source_bar_observed_flag"].to_numpy(dtype=bool)

    print("[feature build] base impulse + causal post-signal checkpoint states", flush=True)
    progress = ProgressReporter(
        label="[event detection + state summaries] direction/windows",
        total=len(windows) * 2,
        every=max(1, int(args.progress_every)),
        enabled=not args.no_progress,
    )
    done = 0
    for window in windows:
        features = r01.build_window_features(bars, window, log_return, abs_price_change, historical_1m_vol)
        pre_impulse_1m_vol = historical_1m_vol.shift(int(window)).to_numpy(dtype=float)
        norm = features.normalized_impulse
        for direction, side in (("LONG", 1), ("SHORT", -1)):
            directed_norm = float(side) * norm
            all_min_positions = np.flatnonzero(np.isfinite(directed_norm) & (directed_norm >= float(minimum_threshold)))
            eligible_positions = all_min_positions[masks["eligible"][all_min_positions]]

            threshold_masks: dict[float, np.ndarray] = {}
            dedup_masks: dict[float, np.ndarray] = {}
            dedup_min_flags = np.zeros(len(eligible_positions), dtype=bool)
            for threshold in thresholds:
                all_t = all_min_positions[directed_norm[all_min_positions] >= float(threshold)]
                dedup_t = r01._deduplicate_positions(all_t, int(window))
                dedup_axis = np.zeros(n, dtype=bool)
                dedup_axis[all_t] = dedup_t
                threshold_mask = directed_norm[eligible_positions] >= float(threshold)
                dedup_mask = threshold_mask & dedup_axis[eligible_positions]
                threshold_masks[float(threshold)] = threshold_mask
                dedup_masks[float(threshold)] = dedup_mask
                if math.isclose(float(threshold), float(minimum_threshold), rel_tol=0.0, abs_tol=1e-12):
                    dedup_min_flags = dedup_mask

            if not eligible_positions.size:
                done += 1
                progress.update(done)
                continue

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
                    "base_entry_time": index[eligible_positions + 1],
                    "base_expected_entry_time": index[eligible_positions] + r01.BAR_DELTA,
                    "base_entry_price": open_arr[eligible_positions + 1],
                    "base_expected_entry_price": pd.to_numeric(bars["open"], errors="coerce").reindex(index[eligible_positions] + r01.BAR_DELTA).to_numpy(dtype=float),
                    "signal_source_bar_observed_flag": observed[eligible_positions],
                    "impulse_source_window_observed_flag": features.source_window_valid[eligible_positions],
                    "full_confirmation_path_observed_flag": masks["full_forward_observed"][eligible_positions],
                    "raw_event_flag": True,
                    "deduplicated_event_flag": dedup_min_flags,
                    "impulse_return": features.impulse_return[eligible_positions],
                    "normalized_impulse": features.normalized_impulse[eligible_positions],
                    "directional_efficiency": features.directional_efficiency[eligible_positions],
                    "window_range_bps": features.window_range_bps[eligible_positions],
                    "window_realized_vol": features.window_realized_vol[eligible_positions],
                }
            )
            event_id_cursor += len(event_frame)
            event_frame["base_entry_not_next_open_flag"] = event_frame["base_entry_time"] != event_frame["base_expected_entry_time"]
            event_frame["base_entry_price_mismatch_flag"] = ~np.isclose(
                event_frame["base_entry_price"].to_numpy(dtype=float),
                event_frame["base_expected_entry_price"].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )
            event_frame["synthetic_bar_dependency_flag"] = ~(
                event_frame["signal_source_bar_observed_flag"].astype(bool)
                & event_frame["impulse_source_window_observed_flag"].astype(bool)
                & event_frame["full_confirmation_path_observed_flag"].astype(bool)
            )
            threshold_array = np.asarray(thresholds, dtype=float)
            directed_event_norm = directed_norm[eligible_positions]
            threshold_idx = np.searchsorted(threshold_array, directed_event_norm, side="right") - 1
            threshold_idx = np.clip(threshold_idx, 0, len(threshold_array) - 1)
            event_frame["max_threshold_reached"] = threshold_array[threshold_idx]
            for threshold in thresholds:
                tag = _threshold_tag(threshold)
                event_frame[f"event_{tag}_flag"] = threshold_masks[float(threshold)]
                event_frame[f"deduplicated_{tag}_flag"] = dedup_masks[float(threshold)]

            checkpoint_data: dict[int, dict[str, np.ndarray]] = {}
            outcome_data: dict[int, dict[int, dict[str, np.ndarray]]] = {}
            for checkpoint in checkpoints:
                k = int(checkpoint)
                cp = _checkpoint_arrays(
                    bars,
                    eligible_positions,
                    side=side,
                    checkpoint=k,
                    pre_impulse_1m_vol=pre_impulse_1m_vol,
                    path_cache=path_cache,
                )
                checkpoint_data[k] = cp
                outcome_data[k] = _remaining_outcomes(
                    bars,
                    cp["checkpoint_pos"],
                    side=side,
                    horizons=horizons,
                    path_cache=path_cache,
                    fee_cost=fee_only_cost,
                    normal_cost=normal_cost,
                )
                event_frame[f"checkpoint_progress_{k}m"] = cp["checkpoint_progress"]
                event_frame[f"checkpoint_progress_z_{k}m"] = cp["checkpoint_progress_z"]
                event_frame[f"checkpoint_mfe_{k}m"] = cp["checkpoint_mfe"]
                event_frame[f"checkpoint_mae_{k}m"] = cp["checkpoint_mae"]
                event_frame[f"checkpoint_mfe_retention_{k}m"] = cp["checkpoint_mfe_retention"]
                event_frame[f"state_bucket_{k}m"] = cp["state_bucket"]
                event_frame[f"checkpoint_bar_end_{k}m"] = index[eligible_positions + k] + r01.BAR_DELTA
                event_frame[f"confirmed_entry_time_{k}m"] = cp["confirmed_entry_time"]
                event_frame[f"expected_confirmed_entry_time_{k}m"] = cp["expected_confirmed_entry_time"]
                event_frame[f"confirmed_entry_price_{k}m"] = cp["confirmed_entry_price"]
                expected_price = pd.to_numeric(bars["open"], errors="coerce").reindex(pd.DatetimeIndex(cp["expected_confirmed_entry_time"])).to_numpy(dtype=float)
                event_frame[f"expected_confirmed_entry_price_{k}m"] = expected_price
                event_frame[f"confirmed_entry_not_next_open_flag_{k}m"] = cp["confirmed_entry_time"] != cp["expected_confirmed_entry_time"]
                event_frame[f"confirmed_entry_price_mismatch_flag_{k}m"] = ~np.isclose(
                    cp["confirmed_entry_price"], expected_price, rtol=0.0, atol=1e-12, equal_nan=True
                )
                event_frame[f"checkpoint_state_valid_flag_{k}m"] = np.isfinite(cp["checkpoint_progress_z"])

            labels_by_cp = {k: checkpoint_data[k]["state_bucket"] for k in checkpoints}
            for threshold in thresholds:
                threshold_mask = threshold_masks[float(threshold)]
                dedup_mask = dedup_masks[float(threshold)]
                for checkpoint in checkpoints:
                    k = int(checkpoint)
                    labels = labels_by_cp[k]
                    for bucket in ["ALL", *list(STATE_ORDER)]:
                        bucket_mask = np.ones(len(event_frame), dtype=bool) if bucket == "ALL" else labels == bucket
                        for event_set, set_mask in (("raw", threshold_mask), ("deduplicated", dedup_mask)):
                            base_mask = set_mask & bucket_mask
                            indices = np.flatnonzero(base_mask)
                            count = int(indices.size)
                            raw_count = int((threshold_mask & bucket_mask).sum())
                            dedup_count = int((dedup_mask & bucket_mask).sum())
                            overlap_ratio = 1.0 - dedup_count / raw_count if raw_count else np.nan
                            event_count_rows.append(
                                {
                                    "direction": direction,
                                    "impulse_window": int(window),
                                    "threshold": float(threshold),
                                    "checkpoint": k,
                                    "state_bucket": bucket,
                                    "event_set": event_set,
                                    "events": count,
                                    "events_per_month": float(count / max(1, study_months)),
                                    "overlap_ratio": overlap_ratio,
                                    "cooldown_bars": int(window),
                                }
                            )
                            sr, yr, mr, tr = _summary_for_mask(
                                event_frame,
                                indices=indices,
                                outcomes=outcome_data[k],
                                direction=direction,
                                window=window,
                                threshold=threshold,
                                checkpoint=k,
                                state_bucket=bucket,
                                event_set=event_set,
                                horizons=horizons,
                                study_months=study_months,
                            )
                            summary_rows.extend(sr)
                            yearly_rows.extend(yr)
                            monthly_rows.extend(mr)
                            top_rows.extend(tr)

            transition_rows.extend(
                _build_transition_rows(
                    event_frame,
                    direction=direction,
                    window=window,
                    thresholds=thresholds,
                    checkpoints=checkpoints,
                )
            )

            if not args.skip_events_csv:
                compact = event_frame[dedup_min_flags].copy()
                compact["event_storage_scope"] = "minimum_threshold_deduplicated_no_forward_matrix"
                compact_rows_written += int(len(compact))
                r01._write_stream_csv(compact, events_path, first_write=first_event_write)
                first_event_write = False
                audit = _build_signal_audit(compact, checkpoints)
                r01._write_stream_csv(audit, audit_path, first_write=first_audit_write)
                first_audit_write = False
                del compact, audit
            del event_frame, checkpoint_data, outcome_data
            done += 1
            progress.update(done)
        del features
    progress.close()

    print("[summaries] assembling state gradients, transitions and parameter plateaus", flush=True)
    event_counts = pd.DataFrame(event_count_rows)
    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    monthly = pd.DataFrame(monthly_rows)
    state_gradient = _build_state_gradient(summary)
    transitions = pd.DataFrame(transition_rows)
    threshold_plateau = _build_threshold_plateau(summary)
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
        "checkpoints": list(checkpoints),
        "remaining_horizons": list(horizons),
        "state_definition": {
            "progress": "side * (checkpoint_close / original_next_open - 1)",
            "normalization": "pre-impulse historical_1m_vol * sqrt(checkpoint)",
            "baseline_end": "before original impulse window",
            "bands": [
                {"name": name, "lower": None if not math.isfinite(lower) else lower, "upper": None if not math.isfinite(upper) else upper}
                for name, lower, upper in STATE_BANDS
            ],
        },
        "confirmation_entry_policy": "checkpoint bar fully closes; enter next 1m bar open",
        "outcome_policy": "remaining return/MFE/MAE measured only after confirmation entry",
        "round02_persistence_condition_stacked": False,
        "round03_notional_condition_stacked": False,
        "round04_barrier_condition_stacked": False,
        "deduplication": "threshold-specific original impulse stream; cooldown=impulse_window",
        "event_storage": "minimum-threshold deduplicated base events written once with checkpoint state columns; no forward matrix",
        "fee_only_cost": fee_only_cost,
        "normal_execution_cost": normal_cost,
        "cost_components": {
            "entry_fee": float(args.entry_fee_rate),
            "exit_fee": float(args.exit_fee_rate),
            "entry_slippage": float(args.entry_slippage),
            "exit_slippage": float(args.exit_slippage),
        },
        "research_month_count": study_months,
        "input_rows": int(len(bars)),
        "source_observed_rows": int(bars["source_bar_observed_flag"].sum()),
        "synthetic_gap_bar_count": int((~bars["source_bar_observed_flag"].astype(bool)).sum()),
        "source_gap_segment_count": int(bars.attrs.get("gap_segment_count", 0)),
        "max_gap_minutes": int(bars.attrs.get("max_gap_minutes", 0)),
        "gap_handling": str(bars.attrs.get("gap_policy", "")),
        "compact_event_rows_written": int(compact_rows_written),
        "events_csv_skipped_for_development": bool(args.skip_events_csv),
        "validation": validation,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "research_boundary": "one mechanism only: causal post-signal price-state confirmation",
    }

    brief = _build_brief(summary, meta)
    artifacts = [
        (event_counts, out_dir / "01_event_counts.csv"),
        (summary, out_dir / "02_state_horizon_summary.csv"),
        (yearly, out_dir / "03_yearly.csv"),
        (monthly, out_dir / "04_monthly.csv"),
        (state_gradient, out_dir / "07_state_gradient.csv"),
        (transitions, out_dir / "08_state_transitions.csv"),
        (threshold_plateau, out_dir / "09_threshold_plateau.csv"),
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
        _update_log(Path(__file__).resolve().with_name("00_research_log.md"), summary, meta)
        count += 1
        p.update(count)

    if not args.skip_review_pack:
        print("[review pack] packaging summary artifacts", flush=True)
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report_dir={out_dir}", flush=True)
    return {"report_dir": out_dir, "events": events_path, "audit": audit_path, "review_pack": out_dir / "gpt_review_pack.zip"}


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] deterministic post-impulse state confirmation study", flush=True)
    labels = _state_labels(np.asarray([-1.0, 0.0, 0.25, 0.75, 1.25, np.nan]))
    expected = [
        "reversal_or_flat_le_0",
        "reversal_or_flat_le_0",
        "weak_continuation_0_0.5",
        "moderate_continuation_0.5_1.0",
        "strong_continuation_ge_1.0",
        "invalid",
    ]
    if labels.tolist() != expected:
        raise AssertionError("state bucket kernel failed")

    raw = r01._synthetic_bars()
    raw = raw.drop(raw.index[3700:3707])
    bars = r01._regularize_trade_bar_axis(raw)
    log_path = Path(__file__).resolve().with_name("00_research_log.md")
    original_log = log_path.read_text(encoding="utf-8") if log_path.exists() else None
    try:
        with tempfile.TemporaryDirectory(prefix="dic_r05_") as tmp:
            args.out_dir = tmp
            args.warmup_start_date = "2022-12-20"
            args.start_date = "2022-12-23"
            args.end_date = "2022-12-24"
            args.impulse_windows = "1,3"
            args.thresholds = "0.5,1.0"
            args.checkpoints = "1,3"
            args.horizons = "3,5"
            args.vol_lookback_bars = 720
            args.vol_min_periods = 360
            args.skip_review_pack = True
            args.no_progress = True
            result = run_research(bars, args)
            required = [
                "01_event_counts.csv",
                "02_state_horizon_summary.csv",
                "03_yearly.csv",
                "04_monthly.csv",
                "05_events.csv",
                "06_signal_audit.csv",
                "07_state_gradient.csv",
                "08_state_transitions.csv",
                "09_threshold_plateau.csv",
                "10_top_trade_dependency.csv",
                "11_run_meta.json",
                "12_research_brief.md",
            ]
            missing = [name for name in required if not (result["report_dir"] / name).exists()]
            if missing:
                raise AssertionError(f"self-test missing artifacts: {missing}")
            summary = pd.read_csv(result["report_dir"] / "02_state_horizon_summary.csv")
            if summary.empty:
                raise AssertionError("self-test summary is empty")
            events = pd.read_csv(result["report_dir"] / "05_events.csv")
            if events.empty:
                raise AssertionError("self-test events are empty")
            for checkpoint in (1, 3):
                expected_time = pd.to_datetime(events["signal_time"]) + pd.to_timedelta(checkpoint, unit="m")
                actual_time = pd.to_datetime(events[f"confirmed_entry_time_{checkpoint}m"])
                if not (expected_time == actual_time).all():
                    raise AssertionError("confirmation entry is not next open after checkpoint")
            audit = pd.read_csv(result["report_dir"] / "06_signal_audit.csv")
            if not audit.empty and audit["lookahead_flag"].astype(bool).any():
                raise AssertionError("self-test causal audit contains invalid events")
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
