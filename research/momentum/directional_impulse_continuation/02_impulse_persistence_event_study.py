#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Directional Impulse Continuation: adjacent-window persistence study (round 02).

Research question
-----------------
Does a directional impulse become a real continuation mechanism only when the
immediately preceding equal-length window was already moving in the same
direction? This round changes one mechanism only: impulse persistence across two
adjacent windows. It adds no trend, volume, session, order-flow, footprint,
funding, OI, liquidation, TP/SL, sizing, or portfolio logic.

Causal policy
-------------
- Local OKX 1m trade bars only; no ordinary K-line download.
- Closed signal bar; next calendar-minute open entry.
- Current impulse normalization excludes the complete current impulse window.
- Prior-window normalization uses a volatility baseline shifted by 2 * window,
  so the denominator excludes both the prior and current impulse windows.
- Synthetic gap rows are excluded from the current impulse, prior impulse,
  signal, entry, and full 240m forward path.
- Every threshold is retained. Persistence bands are fixed before production:
  opposite/flat, weak same-direction, moderate same-direction, strong
  same-direction.

Engineering policy
------------------
The minimum-threshold event pool is materialized once per direction/window.
Higher threshold membership and threshold-specific dedup flags are stored as
columns, avoiding one duplicate event row per nested threshold.
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


def _load_round01_module():
    path = Path(__file__).resolve().with_name("01_basic_impulse_event_study.py")
    spec = importlib.util.spec_from_file_location("directional_impulse_round01", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared round-01 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r01 = _load_round01_module()

SCRIPT_NAME = "02_impulse_persistence_event_study"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION_R02"
EDGE_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION"
TITLE = "ETH Directional Impulse Continuation - Adjacent-Window Persistence Study"
DEFAULT_OUT_DIR = "data/reports/research/momentum/directional_impulse_continuation/02_impulse_persistence_event_study"
BAR_DELTA = pd.Timedelta(minutes=1)

DEFAULT_WINDOWS = (1, 3, 5, 10, 15)
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
DEFAULT_HORIZONS = (1, 3, 5, 10, 15, 30, 60, 120, 240)

PERSISTENCE_BANDS = (
    ("opposite_or_flat", -np.inf, 0.0),
    ("weak_same_0_0.5", 0.0, 0.5),
    ("moderate_same_0.5_1.0", 0.5, 1.0),
    ("strong_same_ge_1.0", 1.0, np.inf),
)
PERSISTENCE_ORDER = {"ALL": -1, **{name: i for i, (name, _, _) in enumerate(PERSISTENCE_BANDS)}}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal ETH 1m adjacent-window impulse persistence event study.",
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
    p.add_argument("--vol-lookback-bars", type=int, default=1440)
    p.add_argument("--vol-min-periods", type=int, default=720)
    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage", type=float, default=0.00020)
    p.add_argument("--exit-slippage", type=float, default=0.00020)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--skip-events-csv", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _threshold_tag(value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "_")
    return f"t{text}"


def _persistence_arrays(
    bars: pd.DataFrame,
    *,
    window: int,
    historical_1m_vol: pd.Series,
    side: int,
) -> dict[str, np.ndarray]:
    close = pd.to_numeric(bars["close"], errors="coerce")
    observed = bars["source_bar_observed_flag"].astype(bool)
    w = int(window)

    two_window_source_valid = observed.rolling(2 * w + 1, min_periods=2 * w + 1).sum().eq(2 * w + 1)
    prior_return = close.shift(w) / close.shift(2 * w) - 1.0
    prior_expected_vol = historical_1m_vol.shift(2 * w) * math.sqrt(w)
    prior_normalized = prior_return / prior_expected_vol.replace(0.0, np.nan)
    direction_adjusted = float(side) * prior_normalized
    valid = (
        two_window_source_valid.to_numpy(dtype=bool)
        & np.isfinite(prior_return.to_numpy(dtype=float))
        & np.isfinite(prior_normalized.to_numpy(dtype=float))
    )
    labels = np.full(len(bars), "", dtype=object)
    x = direction_adjusted.to_numpy(dtype=float)
    labels[valid & (x <= 0.0)] = "opposite_or_flat"
    labels[valid & (x > 0.0) & (x < 0.5)] = "weak_same_0_0.5"
    labels[valid & (x >= 0.5) & (x < 1.0)] = "moderate_same_0.5_1.0"
    labels[valid & (x >= 1.0)] = "strong_same_ge_1.0"
    return {
        "prior_impulse_return": prior_return.to_numpy(dtype=float),
        "prior_expected_window_vol": prior_expected_vol.to_numpy(dtype=float),
        "prior_normalized_impulse": prior_normalized.to_numpy(dtype=float),
        "direction_adjusted_prior_normalized": x,
        "persistence_feature_valid": valid,
        "two_window_source_observed": two_window_source_valid.to_numpy(dtype=bool),
        "persistence_bucket": labels,
    }


def _eligible_masks(bars: pd.DataFrame, args: argparse.Namespace, horizons: tuple[int, ...]) -> dict[str, Any]:
    start, end_exclusive = r01._date_bounds(args.start_date, args.end_date)
    max_horizon = max(horizons)
    n = len(bars)
    research_mask = np.asarray((bars.index >= start) & (bars.index < end_exclusive), dtype=bool)
    observed = bars["source_bar_observed_flag"].to_numpy(dtype=bool)

    full_path_mask = np.arange(n, dtype=int) + max_horizon < n
    forward_observed_count = (
        pd.Series(observed, index=bars.index)
        .shift(-1)
        .iloc[::-1]
        .rolling(max_horizon, min_periods=max_horizon)
        .sum()
        .iloc[::-1]
        .to_numpy(dtype=float)
    )
    full_forward_observed = forward_observed_count == float(max_horizon)

    next_bar_mask = np.arange(n, dtype=int) + 1 < n
    expected_next_time = bars.index.to_numpy(dtype="datetime64[ns]") + np.timedelta64(1, "m")
    actual_next_time = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    actual_next_time[:-1] = bars.index.to_numpy(dtype="datetime64[ns]")[1:]
    next_source_observed = np.zeros(n, dtype=bool)
    next_source_observed[:-1] = observed[1:]
    causal_next_bar = next_bar_mask & (actual_next_time == expected_next_time) & next_source_observed

    eligible = research_mask & observed & full_path_mask & full_forward_observed & causal_next_bar
    months = len(
        pd.period_range(
            start=start.to_period("M"),
            end=(end_exclusive - BAR_DELTA).to_period("M"),
            freq="M",
        )
    )
    return {
        "start": start,
        "end_exclusive": end_exclusive,
        "research_mask": research_mask,
        "observed": observed,
        "full_forward_observed": full_forward_observed,
        "causal_next_bar": causal_next_bar,
        "eligible": eligible,
        "study_months": months,
    }


def _group_indices(labels: np.ndarray, mask: np.ndarray) -> dict[Any, np.ndarray]:
    return r01._group_indices(labels, mask)


def _summary_for_mask(
    events: pd.DataFrame,
    *,
    mask: np.ndarray,
    direction: str,
    window: int,
    threshold: float,
    persistence_bucket: str,
    event_set: str,
    horizons: tuple[int, ...],
    study_months: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []

    years = pd.to_datetime(events["signal_bar_start"]).dt.year.to_numpy()
    months = pd.to_datetime(events["signal_bar_start"]).dt.to_period("M").astype(str).to_numpy()
    base_indices = np.flatnonzero(mask)
    year_groups = _group_indices(years, mask)
    month_groups = _group_indices(months, mask)

    for horizon in horizons:
        h = int(horizon)
        gross = events[f"forward_return_{h}m"].to_numpy(dtype=float)
        fee = events[f"fee_only_net_return_{h}m"].to_numpy(dtype=float)
        net = events[f"normal_net_return_{h}m"].to_numpy(dtype=float)
        mfe = events[f"mfe_{h}m"].to_numpy(dtype=float)
        mae = events[f"mae_{h}m"].to_numpy(dtype=float)
        stat = r01._stats(gross[base_indices], fee[base_indices], net[base_indices], mfe[base_indices], mae[base_indices])

        year_means: dict[int, float] = {}
        for year, idx in year_groups.items():
            period = r01._stats(gross[idx], fee[idx], net[idx], mfe[idx], mae[idx])
            yearly_rows.append(
                {
                    "direction": direction,
                    "impulse_window": int(window),
                    "threshold": float(threshold),
                    "persistence_bucket": persistence_bucket,
                    "horizon": h,
                    "event_set": event_set,
                    "year": int(year),
                    **period,
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
                    "persistence_bucket": persistence_bucket,
                    "horizon": h,
                    "event_set": event_set,
                    "month": str(month),
                    **period,
                }
            )

        finite_years = {y: v for y, v in year_means.items() if math.isfinite(v)}
        if finite_years:
            worst_year = min(finite_years, key=finite_years.get)
            positive_years = sum(v > 0 for v in finite_years.values())
        else:
            worst_year = None
            positive_years = 0
        summary_rows.append(
            {
                "direction": direction,
                "impulse_window": int(window),
                "threshold": float(threshold),
                "persistence_bucket": persistence_bucket,
                "horizon": h,
                "event_set": event_set,
                "events_per_month": float(stat["events"] / max(1, study_months)),
                **stat,
                "positive_year_count": int(positive_years),
                "total_year_count": int(len(finite_years)),
                "worst_year": worst_year,
                "worst_year_mean_net": finite_years.get(worst_year, np.nan) if worst_year is not None else np.nan,
            }
        )

        positive = base_indices[np.isfinite(net[base_indices]) & (net[base_indices] > 0)]
        if positive.size:
            order = positive[np.argsort(net[positive])[::-1][:5]]
            total_positive = float(net[positive].sum())
            cumulative = 0.0
            for rank, row_idx in enumerate(order, start=1):
                contribution = float(net[row_idx] / total_positive) if total_positive > 0 else np.nan
                cumulative += contribution if math.isfinite(contribution) else 0.0
                top_rows.append(
                    {
                        "direction": direction,
                        "impulse_window": int(window),
                        "threshold": float(threshold),
                        "persistence_bucket": persistence_bucket,
                        "horizon": h,
                        "event_set": event_set,
                        "rank": rank,
                        "event_id": int(events.iloc[row_idx]["event_id"]),
                        "signal_time": events.iloc[row_idx]["signal_time"],
                        "normal_net_return": float(net[row_idx]),
                        "contribution_to_positive_return": contribution,
                        "cumulative_top_contribution": cumulative,
                    }
                )
    return summary_rows, yearly_rows, monthly_rows, top_rows


def _build_persistence_gradient(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["direction", "impulse_window", "threshold", "horizon", "event_set"]
    bands = [name for name, _, _ in PERSISTENCE_BANDS]
    for key, part in summary[summary["persistence_bucket"].isin(bands)].groupby(keys, observed=False, dropna=False):
        ordered = part.assign(_order=part["persistence_bucket"].map(PERSISTENCE_ORDER)).sort_values("_order")
        previous: pd.Series | None = None
        base_events = float(ordered["events"].sum())
        for _, row in ordered.iterrows():
            item = {k: v for k, v in zip(keys, key if isinstance(key, tuple) else (key,), strict=False)}
            item.update(
                {
                    "persistence_bucket": row["persistence_bucket"],
                    "events": int(row["events"]),
                    "share_of_valid_events": float(row["events"] / base_events) if base_events > 0 else np.nan,
                    "mean_gross": float(row["mean_gross"]),
                    "median_gross": float(row["median_gross"]),
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
                        "previous_bucket": None,
                        "mean_net_change_vs_previous": np.nan,
                        "median_net_change_vs_previous": np.nan,
                        "profit_factor_change_vs_previous": np.nan,
                    }
                )
            else:
                item.update(
                    {
                        "previous_bucket": previous["persistence_bucket"],
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
    keys = ["direction", "impulse_window", "persistence_bucket", "horizon", "event_set"]
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
                        "profit_factor_change_vs_previous": np.nan,
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
                        "profit_factor_change_vs_previous": float(row["profit_factor"] - previous["profit_factor"]),
                    }
                )
            rows.append(item)
            previous = row
    return pd.DataFrame(rows)


def _build_signal_audit(events: pd.DataFrame) -> pd.DataFrame:
    audit = r01._build_signal_audit(events)
    audit["prior_source_window_not_observed_flag"] = ~events["prior_source_window_observed_flag"].astype(bool).to_numpy()
    audit["persistence_feature_invalid_flag"] = ~events["persistence_feature_valid_flag"].astype(bool).to_numpy()
    audit["lookahead_flag"] = (
        audit["lookahead_flag"].astype(bool)
        | audit["prior_source_window_not_observed_flag"].astype(bool)
        | audit["persistence_feature_invalid_flag"].astype(bool)
    )
    return audit


def _build_brief(summary: pd.DataFrame, event_counts: pd.DataFrame, meta: dict[str, Any]) -> str:
    lines = [
        "# ETH Directional Impulse Continuation - Adjacent-Window Persistence Study",
        "",
        "> Automated descriptive brief. This is not an accepted Edge decision and not a strategy backtest.",
        "",
        "## Research question",
        "",
        "Does continuation improve monotonically when the immediately preceding equal-length window already moved in the impulse direction?",
        "",
        "## Fixed persistence bands",
        "",
        "- `opposite_or_flat`: direction-adjusted prior normalized impulse <= 0",
        "- `weak_same_0_0.5`: > 0 and < 0.5",
        "- `moderate_same_0.5_1.0`: >= 0.5 and < 1.0",
        "- `strong_same_ge_1.0`: >= 1.0",
        "",
        "The prior normalization denominator ends before both adjacent impulse windows; no current or prior impulse return is included in that baseline.",
        "",
    ]
    dedup = summary[(summary["event_set"] == "deduplicated") & (summary["persistence_bucket"] != "ALL") & (summary["events"] >= 100)]
    for direction in ("LONG", "SHORT"):
        part = dedup[dedup["direction"] == direction].sort_values(
            ["mean_net", "median_net", "events"], ascending=[False, False, False]
        )
        lines.extend([f"## {direction}", ""])
        if part.empty:
            lines.extend(["No eligible events.", ""])
            continue
        best = part.head(10)
        lines.extend(
            [
                "Best descriptive rows (not parameter selection):",
                "",
                "| window | threshold | persistence | horizon | events | events/month | mean net | median net | win rate | PF | positive years | top5 share |",
                "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in best.iterrows():
            lines.append(
                f"| {int(row['impulse_window'])}m | {float(row['threshold']):.2f} | {row['persistence_bucket']} | "
                f"{int(row['horizon'])}m | {int(row['events'])} | {float(row['events_per_month']):.2f} | "
                f"{float(row['mean_net']):.4%} | {float(row['median_net']):.4%} | {float(row['win_rate']):.2%} | "
                f"{float(row['profit_factor']):.3f} | {int(row['positive_year_count'])}/{int(row['total_year_count'])} | "
                f"{float(row['top_5_event_contribution']):.2%} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Decision rules",
            "",
            "A persistence mechanism is credible only if performance improves across adjacent persistence bands, survives normal cost, retains reasonable frequency, and is directionally consistent across years and nearby impulse thresholds/windows.",
            "",
            "An isolated strong bucket is not enough. If only long events improve, short continuation remains a separate failed branch rather than being forced into symmetric rules.",
            "",
            "## Run facts",
            "",
            f"- Unique minimum-threshold event rows written: {int(meta['unique_event_rows_written']):,}",
            f"- Synthetic gap bars excluded: {int(meta['synthetic_gap_bar_count']):,}",
            f"- Normal round-trip cost: {float(meta['normal_execution_cost']):.4%}",
        ]
    )
    return "\n".join(lines) + "\n"


def _update_log(log_path: Path, summary: pd.DataFrame, meta: dict[str, Any]) -> None:
    text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# ETH Directional Impulse Continuation — Research Log\n"
    marker_start = "<!-- ROUND02_AUTO_RESULT_START -->"
    marker_end = "<!-- ROUND02_AUTO_RESULT_END -->"
    dedup = summary[(summary["event_set"] == "deduplicated") & (summary["persistence_bucket"] != "ALL") & (summary["events"] >= 100)]
    best_long = dedup[dedup["direction"] == "LONG"].sort_values("mean_net", ascending=False).head(1)
    best_short = dedup[dedup["direction"] == "SHORT"].sort_values("mean_net", ascending=False).head(1)

    def row_text(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No eligible row"
        r = frame.iloc[0]
        return (
            f"{int(r['impulse_window'])}m / threshold {float(r['threshold']):.2f} / {r['persistence_bucket']} / "
            f"{int(r['horizon'])}m: events={int(r['events'])}, events/month={float(r['events_per_month']):.2f}, "
            f"mean net={float(r['mean_net']):.4%}, median net={float(r['median_net']):.4%}, "
            f"win rate={float(r['win_rate']):.2%}, PF={float(r['profit_factor']):.3f}, "
            f"positive years={int(r['positive_year_count'])}/{int(r['total_year_count'])}"
        )

    block = "\n".join(
        [
            marker_start,
            "## Round 02 generated result",
            "",
            f"- LONG best descriptive row: {row_text(best_long)}",
            f"- SHORT best descriptive row: {row_text(best_short)}",
            f"- Unique minimum-threshold event rows written: {int(meta['unique_event_rows_written']):,}",
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
    horizons = r01._parse_int_csv(args.horizons)
    if int(args.vol_min_periods) > int(args.vol_lookback_bars):
        raise ValueError("vol-min-periods cannot exceed vol-lookback-bars")

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
    masks = _eligible_masks(bars, args, horizons)
    study_months = int(masks["study_months"])
    fee_only_cost = float(args.entry_fee_rate + args.exit_fee_rate)
    normal_cost = float(fee_only_cost + args.entry_slippage + args.exit_slippage)
    if not math.isclose(fee_only_cost, 0.0011, rel_tol=0.0, abs_tol=1e-12):
        print(f"[warning] fee-only cost differs from 0.11%: {fee_only_cost:.6%}", flush=True)
    if not math.isclose(normal_cost, 0.0015, rel_tol=0.0, abs_tol=1e-12):
        print(f"[warning] normal cost differs from 0.15%: {normal_cost:.6%}", flush=True)

    path_cache = r01.build_path_cache(bars, horizons, progress_enabled=not args.no_progress)
    log_return, abs_price_change, historical_1m_vol = r01.build_base_volatility(
        bars, int(args.vol_lookback_bars), int(args.vol_min_periods)
    )

    event_count_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    event_id_cursor = 1
    first_event_write = True
    first_audit_write = True
    minimum_threshold = min(thresholds)
    n = len(bars)

    print("[feature build] current impulse + immediately preceding equal-length impulse", flush=True)
    progress = ProgressReporter(
        label="[event detection + persistence summaries] direction/windows",
        total=len(windows) * 2,
        every=max(1, int(args.progress_every)),
        enabled=not args.no_progress,
    )
    done = 0

    for window in windows:
        features = r01.build_window_features(bars, window, log_return, abs_price_change, historical_1m_vol)
        norm = features.normalized_impulse
        for direction, side in (("LONG", 1), ("SHORT", -1)):
            persistence = _persistence_arrays(
                bars,
                window=window,
                historical_1m_vol=historical_1m_vol,
                side=side,
            )
            directed_norm = float(side) * norm
            all_min_positions = np.flatnonzero(
                np.isfinite(directed_norm)
                & (directed_norm >= float(minimum_threshold))
                & persistence["persistence_feature_valid"]
            )
            eligible_positions = all_min_positions[masks["eligible"][all_min_positions]]

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
                if math.isclose(float(threshold), float(minimum_threshold), abs_tol=1e-12, rel_tol=0.0):
                    dedup_min_flags = event_dedup_mask

                research_t = (
                    masks["research_mask"][all_t]
                    & masks["causal_next_bar"][all_t]
                    & persistence["persistence_feature_valid"][all_t]
                )
                eligible_t = masks["eligible"][all_t] & persistence["persistence_feature_valid"][all_t]
                labels_t = persistence["persistence_bucket"][all_t]
                for bucket in ["ALL", *[name for name, _, _ in PERSISTENCE_BANDS]]:
                    bucket_all = np.ones(len(all_t), dtype=bool) if bucket == "ALL" else labels_t == bucket
                    raw_count = int((eligible_t & bucket_all).sum())
                    dedup_count = int((eligible_t & bucket_all & dedup_t).sum())
                    overlap_ratio = 1.0 - dedup_count / raw_count if raw_count else np.nan
                    for event_set, count in (("raw", raw_count), ("deduplicated", dedup_count)):
                        event_count_rows.append(
                            {
                                "direction": direction,
                                "impulse_window": int(window),
                                "threshold": float(threshold),
                                "persistence_bucket": bucket,
                                "event_set": event_set,
                                "raw_detected_count_full_loaded_axis": int(bucket_all.sum()),
                                "raw_detected_count_research_window_before_full_path_check": int(
                                    (research_t & bucket_all).sum()
                                ),
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
                    horizons=horizons,
                    fee_cost=fee_only_cost,
                    normal_cost=normal_cost,
                    event_id_start=event_id_cursor,
                )
                event_id_cursor += len(event_frame)
                event_frame["prior_impulse_return"] = persistence["prior_impulse_return"][eligible_positions]
                event_frame["prior_expected_window_vol"] = persistence["prior_expected_window_vol"][eligible_positions]
                event_frame["prior_normalized_impulse"] = persistence["prior_normalized_impulse"][eligible_positions]
                event_frame["direction_adjusted_prior_normalized"] = persistence[
                    "direction_adjusted_prior_normalized"
                ][eligible_positions]
                event_frame["persistence_bucket"] = persistence["persistence_bucket"][eligible_positions]
                event_frame["prior_source_window_observed_flag"] = persistence["two_window_source_observed"][
                    eligible_positions
                ]
                event_frame["persistence_feature_valid_flag"] = persistence["persistence_feature_valid"][
                    eligible_positions
                ]
                event_frame["minimum_pool_threshold"] = float(minimum_threshold)
                event_frame["max_threshold_reached"] = np.asarray(
                    [max(t for t in thresholds if value >= float(t)) for value in directed_norm[eligible_positions]],
                    dtype=float,
                )

                for threshold in thresholds:
                    tag = _threshold_tag(threshold)
                    event_frame[f"event_{tag}_flag"] = threshold_masks_event[float(threshold)]
                    event_frame[f"deduplicated_{tag}_flag"] = dedup_masks_event[float(threshold)]

                labels = event_frame["persistence_bucket"].to_numpy(dtype=object)
                for threshold in thresholds:
                    threshold_mask = threshold_masks_event[float(threshold)]
                    dedup_mask = dedup_masks_event[float(threshold)]
                    for bucket in ["ALL", *[name for name, _, _ in PERSISTENCE_BANDS]]:
                        bucket_mask = np.ones(len(event_frame), dtype=bool) if bucket == "ALL" else labels == bucket
                        for event_set, event_set_mask in (("raw", threshold_mask), ("deduplicated", dedup_mask)):
                            base_mask = event_set_mask & bucket_mask
                            sr, yr, mr, tr = _summary_for_mask(
                                event_frame,
                                mask=base_mask,
                                direction=direction,
                                window=window,
                                threshold=threshold,
                                persistence_bucket=bucket,
                                event_set=event_set,
                                horizons=horizons,
                                study_months=study_months,
                            )
                            summary_rows.extend(sr)
                            yearly_rows.extend(yr)
                            monthly_rows.extend(mr)
                            top_rows.extend(tr)

                if not args.skip_events_csv:
                    r01._write_stream_csv(event_frame, events_path, first_write=first_event_write)
                    first_event_write = False
                    audit = _build_signal_audit(event_frame)
                    r01._write_stream_csv(audit, audit_path, first_write=first_audit_write)
                    first_audit_write = False
                del event_frame
            done += 1
            progress.update(done)
        del features
    progress.close()

    print("[summaries] assembling persistence gradients, threshold plateaus and dependency tables", flush=True)
    event_counts = pd.DataFrame(event_count_rows)
    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    monthly = pd.DataFrame(monthly_rows)
    persistence_gradient = _build_persistence_gradient(summary)
    threshold_plateau = _build_threshold_plateau(summary)
    top_dependency = pd.DataFrame(top_rows)

    if first_event_write:
        pd.DataFrame(columns=["event_id", "direction", "persistence_bucket"]).to_csv(events_path, index=False)
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
        "horizons": list(horizons),
        "persistence_definition": {
            "prior_window": "immediately preceding equal-length window",
            "formula": "side * prior_window_return / (historical_1m_vol.shift(2*window) * sqrt(window))",
            "baseline_exclusion": "both prior and current impulse windows are excluded",
            "bands": [
                {"name": name, "lower": None if not math.isfinite(lower) else lower, "upper": None if not math.isfinite(upper) else upper}
                for name, lower, upper in PERSISTENCE_BANDS
            ],
        },
        "current_impulse_normalization": {
            "formula": "impulse_return / (historical rolling std shifted by impulse_window * sqrt(window))",
            "vol_lookback_bars": int(args.vol_lookback_bars),
            "vol_min_periods": int(args.vol_min_periods),
        },
        "deduplication": "threshold-specific same-direction stream; cooldown=impulse_window, then persistence buckets are subsets",
        "event_storage": "minimum-threshold event rows written once; higher threshold membership and dedup flags stored as columns",
        "entry_policy": "closed signal bar; next 1m bar open",
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
        "unique_event_rows_written": int(event_id_cursor - 1),
        "events_csv_skipped_for_development": bool(args.skip_events_csv),
        "validation": validation,
        "created_at": pd.Timestamp.utcnow().isoformat(),
        "research_boundary": "one mechanism only: adjacent-window impulse persistence",
    }

    brief = _build_brief(summary, event_counts, meta)
    artifacts = [
        (event_counts, out_dir / "01_event_counts.csv"),
        (summary, out_dir / "02_horizon_summary.csv"),
        (yearly, out_dir / "03_yearly.csv"),
        (monthly, out_dir / "04_monthly.csv"),
        (persistence_gradient, out_dir / "07_persistence_gradient.csv"),
        (threshold_plateau, out_dir / "08_threshold_plateau.csv"),
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
    return {"report_dir": out_dir, "events": events_path, "audit": audit_path, "review_pack": out_dir / "gpt_review_pack.zip"}


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] deterministic adjacent-window persistence study", flush=True)
    raw = r01._synthetic_bars()
    raw = raw.drop(raw.index[3700:3707])
    bars = r01._regularize_trade_bar_axis(raw)
    log_path = Path(__file__).resolve().with_name("00_research_log.md")
    original_log = log_path.read_text(encoding="utf-8") if log_path.exists() else None
    try:
        with tempfile.TemporaryDirectory(prefix="dic_r02_") as tmp:
            args.out_dir = tmp
            args.warmup_start_date = "2022-12-20"
            args.start_date = "2022-12-23"
            args.end_date = "2022-12-24"
            args.vol_lookback_bars = 720
            args.vol_min_periods = 360
            args.skip_review_pack = True
            args.no_progress = True
            result = run_research(bars, args)
            required = [
                "01_event_counts.csv",
                "02_horizon_summary.csv",
                "03_yearly.csv",
                "04_monthly.csv",
                "05_events.csv",
                "06_signal_audit.csv",
                "07_persistence_gradient.csv",
                "08_threshold_plateau.csv",
                "09_top_trade_dependency.csv",
                "10_run_meta.json",
                "11_research_brief.md",
            ]
            missing = [name for name in required if not (result["report_dir"] / name).exists()]
            if missing:
                raise AssertionError(f"self-test missing artifacts: {missing}")
            summary = pd.read_csv(result["report_dir"] / "02_horizon_summary.csv")
            if summary.empty:
                raise AssertionError("self-test summary is empty")
            expected_bands = {name for name, _, _ in PERSISTENCE_BANDS}
            if not expected_bands.issubset(set(summary["persistence_bucket"].dropna().unique())):
                raise AssertionError("self-test did not retain all fixed persistence bands")
            events = pd.read_csv(result["report_dir"] / "05_events.csv")
            if events.empty:
                raise AssertionError("self-test events are empty")
            parsed_thresholds = r01._parse_float_csv(args.thresholds)
            for threshold in parsed_thresholds:
                if f"event_{_threshold_tag(threshold)}_flag" not in events.columns:
                    raise AssertionError("threshold membership columns missing")
            for low, high in zip(parsed_thresholds[:-1], parsed_thresholds[1:], strict=False):
                low_flag = events[f"event_{_threshold_tag(low)}_flag"].astype(bool).to_numpy()
                high_flag = events[f"event_{_threshold_tag(high)}_flag"].astype(bool).to_numpy()
                if np.any(high_flag & ~low_flag):
                    raise AssertionError("nested threshold membership is not monotonic")
            if not np.allclose(
                events["max_threshold_reached"].to_numpy(dtype=float),
                np.max(
                    np.column_stack(
                        [
                            np.where(
                                events[f"event_{_threshold_tag(t)}_flag"].astype(bool).to_numpy(),
                                float(t),
                                -np.inf,
                            )
                            for t in parsed_thresholds
                        ]
                    ),
                    axis=1,
                ),
            ):
                raise AssertionError("max_threshold_reached does not match threshold flags")
            audit = pd.read_csv(result["report_dir"] / "06_signal_audit.csv")
            if not audit.empty and audit["lookahead_flag"].astype(bool).any():
                raise AssertionError("self-test causal audit contains invalid events")
            meta = json.loads((result["report_dir"] / "10_run_meta.json").read_text(encoding="utf-8"))
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
