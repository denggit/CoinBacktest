#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Directional Impulse Continuation: basic notional-expansion study (round 03).

Research question
-----------------
Does a directional price impulse continue more reliably when the same impulse
window also carries unusually high traded quote notional relative to its own
causal history?

This round changes one mechanism only: impulse-window quote-notional expansion.
It deliberately branches from the Round-01 base event and does not stack the
Round-02 persistence condition. No trend, EMA, VWAP, session, order-flow side,
footprint, funding, OI, liquidation, TP/SL, sizing, or portfolio logic is added.

Causal policy
-------------
- Local OKX 1m trade bars only; no ordinary K-line download.
- Closed signal bar; next calendar-minute open entry.
- Current price-impulse normalization excludes the complete current impulse
  window, as in Round 01.
- Current impulse-window quote notional is compared with a trailing median of
  equal-length window notionals shifted by the complete impulse window. The
  activity baseline therefore contains no bar from the current impulse window.
- Synthetic gap rows are excluded from activity features, signal, entry, and
  the complete 240m forward path.
- Every current-impulse threshold is retained. Activity bands are fixed before
  production and are not selected from the result.

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


def _load_round02_module():
    path = Path(__file__).resolve().with_name("02_impulse_persistence_event_study.py")
    spec = importlib.util.spec_from_file_location("directional_impulse_round02", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load shared round-02 implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


r02 = _load_round02_module()
r01 = r02.r01

SCRIPT_NAME = "03_impulse_notional_expansion_event_study"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION_R03"
EDGE_ID = "ETH_MOM_DIRECTIONAL_IMPULSE_CONTINUATION"
TITLE = "ETH Directional Impulse Continuation - Quote-Notional Expansion Study"
DEFAULT_OUT_DIR = (
    "data/reports/research/momentum/directional_impulse_continuation/"
    "03_impulse_notional_expansion_event_study"
)

DEFAULT_WINDOWS = (1, 3, 5, 10, 15)
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0, 2.5)
DEFAULT_HORIZONS = (1, 3, 5, 10, 15, 30, 60, 120, 240)

NOTIONAL_BANDS = (
    ("low_lt_0.75", -np.inf, 0.75),
    ("normal_0.75_1.25", 0.75, 1.25),
    ("elevated_1.25_2.0", 1.25, 2.0),
    ("extreme_ge_2.0", 2.0, np.inf),
)
NOTIONAL_ORDER = {"ALL": -1, **{name: i for i, (name, _, _) in enumerate(NOTIONAL_BANDS)}}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal ETH 1m impulse-window quote-notional expansion event study.",
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
    p.add_argument("--activity-lookback-bars", type=int, default=1440)
    p.add_argument("--activity-min-periods", type=int, default=720)
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
    return r02._threshold_tag(value)


def _notional_expansion_arrays(
    bars: pd.DataFrame,
    *,
    window: int,
    lookback_bars: int,
    min_periods: int,
) -> dict[str, np.ndarray]:
    """Build causal impulse-window activity features.

    The baseline series at bar ``t`` is the trailing median of equal-length
    window notionals whose latest member ends at ``t-window``. Thus no bar from
    the current impulse window is present in the denominator.
    """
    if "notional" not in bars.columns:
        raise RuntimeError(
            "OKX trade bars are missing the required 'notional' column. "
            "Round 03 uses quote notional from OKXTradeBarLoader and will not "
            "silently substitute ordinary-K-line volume."
        )
    if "trades_count" not in bars.columns:
        raise RuntimeError("OKX trade bars are missing the required 'trades_count' column")

    w = int(window)
    observed = bars["source_bar_observed_flag"].astype(bool)
    notional = pd.to_numeric(bars["notional"], errors="coerce")
    trades_count = pd.to_numeric(bars["trades_count"], errors="coerce")

    source_window_observed = observed.rolling(w, min_periods=w).sum().eq(w)
    window_notional = notional.rolling(w, min_periods=w).sum()
    window_trades_count = trades_count.rolling(w, min_periods=w).sum()

    historical_window_notional_median = (
        window_notional.shift(w)
        .rolling(int(lookback_bars), min_periods=int(min_periods))
        .median()
    )
    expansion_ratio = window_notional / historical_window_notional_median.replace(0.0, np.nan)

    valid = (
        source_window_observed.to_numpy(dtype=bool)
        & np.isfinite(window_notional.to_numpy(dtype=float))
        & np.isfinite(window_trades_count.to_numpy(dtype=float))
        & np.isfinite(historical_window_notional_median.to_numpy(dtype=float))
        & (historical_window_notional_median.to_numpy(dtype=float) > 0.0)
        & np.isfinite(expansion_ratio.to_numpy(dtype=float))
    )

    labels = np.full(len(bars), "", dtype=object)
    x = expansion_ratio.to_numpy(dtype=float)
    labels[valid & (x < 0.75)] = "low_lt_0.75"
    labels[valid & (x >= 0.75) & (x < 1.25)] = "normal_0.75_1.25"
    labels[valid & (x >= 1.25) & (x < 2.0)] = "elevated_1.25_2.0"
    labels[valid & (x >= 2.0)] = "extreme_ge_2.0"

    return {
        "impulse_window_notional": window_notional.to_numpy(dtype=float),
        "impulse_window_trades_count": window_trades_count.to_numpy(dtype=float),
        "historical_window_notional_median": historical_window_notional_median.to_numpy(dtype=float),
        "notional_expansion_ratio": x,
        "notional_feature_valid": valid,
        "notional_source_window_observed": source_window_observed.to_numpy(dtype=bool),
        "notional_bucket": labels,
    }


def _rename_bucket_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        if "persistence_bucket" in row:
            row["notional_bucket"] = row.pop("persistence_bucket")
    return rows


def _summary_for_mask(
    events: pd.DataFrame,
    *,
    mask: np.ndarray,
    direction: str,
    window: int,
    threshold: float,
    notional_bucket: str,
    event_set: str,
    horizons: tuple[int, ...],
    study_months: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sr, yr, mr, tr = r02._summary_for_mask(
        events,
        mask=mask,
        direction=direction,
        window=window,
        threshold=threshold,
        persistence_bucket=notional_bucket,
        event_set=event_set,
        horizons=horizons,
        study_months=study_months,
    )
    return tuple(_rename_bucket_rows(rows) for rows in (sr, yr, mr, tr))  # type: ignore[return-value]


def _build_notional_gradient(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["direction", "impulse_window", "threshold", "horizon", "event_set"]
    bands = [name for name, _, _ in NOTIONAL_BANDS]
    for key, part in summary[summary["notional_bucket"].isin(bands)].groupby(
        keys, observed=False, dropna=False
    ):
        ordered = part.assign(_order=part["notional_bucket"].map(NOTIONAL_ORDER)).sort_values("_order")
        previous: pd.Series | None = None
        base_events = float(ordered["events"].sum())
        for _, row in ordered.iterrows():
            item = {k: v for k, v in zip(keys, key if isinstance(key, tuple) else (key,), strict=False)}
            item.update(
                {
                    "notional_bucket": row["notional_bucket"],
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
                        "previous_bucket": previous["notional_bucket"],
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
    tmp = summary.rename(columns={"notional_bucket": "persistence_bucket"})
    out = r02._build_threshold_plateau(tmp)
    return out.rename(columns={"persistence_bucket": "notional_bucket"})


def _build_signal_audit(events: pd.DataFrame) -> pd.DataFrame:
    audit = r01._build_signal_audit(events)
    audit["notional_source_window_not_observed_flag"] = ~events[
        "notional_source_window_observed_flag"
    ].astype(bool).to_numpy()
    audit["notional_feature_invalid_flag"] = ~events["notional_feature_valid_flag"].astype(bool).to_numpy()
    audit["lookahead_flag"] = (
        audit["lookahead_flag"].astype(bool)
        | audit["notional_source_window_not_observed_flag"].astype(bool)
        | audit["notional_feature_invalid_flag"].astype(bool)
    )
    return audit


def _build_brief(summary: pd.DataFrame, meta: dict[str, Any]) -> str:
    lines = [
        "# ETH Directional Impulse Continuation - Quote-Notional Expansion Study",
        "",
        "> Automated descriptive brief. This is not an accepted Edge decision and not a strategy backtest.",
        "",
        "## Research question",
        "",
        "Does continuation improve monotonically when the impulse window carries more quote notional than its causal historical norm?",
        "",
        "## Fixed quote-notional expansion bands",
        "",
        "- `low_lt_0.75`: current window notional < 0.75x historical median",
        "- `normal_0.75_1.25`: 0.75x to < 1.25x",
        "- `elevated_1.25_2.0`: 1.25x to < 2.0x",
        "- `extreme_ge_2.0`: >= 2.0x",
        "",
        "The historical median is formed from equal-length window notionals shifted by the full impulse window, so it excludes every bar in the current impulse.",
        "",
    ]
    dedup = summary[
        (summary["event_set"] == "deduplicated")
        & (summary["notional_bucket"] != "ALL")
        & (summary["events"] >= 100)
    ]
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
                "| window | threshold | notional band | horizon | events | events/month | mean net | median net | win rate | PF | positive years | top5 share |",
                "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for _, row in best.iterrows():
            lines.append(
                f"| {int(row['impulse_window'])}m | {float(row['threshold']):.2f} | {row['notional_bucket']} | "
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
            "A quote-notional mechanism is credible only if performance improves across adjacent activity bands, survives normal cost, retains reasonable frequency, and remains consistent across years and nearby impulse thresholds/windows.",
            "",
            "An isolated extreme-volume bucket is not enough. LONG and SHORT are judged independently.",
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
    marker_start = "<!-- ROUND03_AUTO_RESULT_START -->"
    marker_end = "<!-- ROUND03_AUTO_RESULT_END -->"
    dedup = summary[
        (summary["event_set"] == "deduplicated")
        & (summary["notional_bucket"] != "ALL")
        & (summary["events"] >= 100)
    ]
    best_long = dedup[dedup["direction"] == "LONG"].sort_values("mean_net", ascending=False).head(1)
    best_short = dedup[dedup["direction"] == "SHORT"].sort_values("mean_net", ascending=False).head(1)

    def row_text(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No eligible row"
        r = frame.iloc[0]
        return (
            f"{int(r['impulse_window'])}m / threshold {float(r['threshold']):.2f} / {r['notional_bucket']} / "
            f"{int(r['horizon'])}m: events={int(r['events'])}, events/month={float(r['events_per_month']):.2f}, "
            f"mean net={float(r['mean_net']):.4%}, median net={float(r['median_net']):.4%}, "
            f"win rate={float(r['win_rate']):.2%}, PF={float(r['profit_factor']):.3f}, "
            f"positive years={int(r['positive_year_count'])}/{int(r['total_year_count'])}"
        )

    block = "\n".join(
        [
            marker_start,
            "## Round 03 generated result",
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
    if int(args.activity_min_periods) > int(args.activity_lookback_bars):
        raise ValueError("activity-min-periods cannot exceed activity-lookback-bars")

    required_activity_columns = ["notional", "trades_count"]
    missing_activity = [c for c in required_activity_columns if c not in bars.columns]
    if missing_activity:
        raise RuntimeError(f"Trade-bar activity data missing columns: {missing_activity}")
    for col in required_activity_columns:
        bars[col] = pd.to_numeric(bars[col], errors="coerce")

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
    masks = r02._eligible_masks(bars, args, horizons)
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

    print("[feature build] current impulse + causal quote-notional expansion", flush=True)
    progress = ProgressReporter(
        label="[event detection + notional summaries] direction/windows",
        total=len(windows) * 2,
        every=max(1, int(args.progress_every)),
        enabled=not args.no_progress,
    )
    done = 0

    for window in windows:
        features = r01.build_window_features(bars, window, log_return, abs_price_change, historical_1m_vol)
        activity = _notional_expansion_arrays(
            bars,
            window=window,
            lookback_bars=int(args.activity_lookback_bars),
            min_periods=int(args.activity_min_periods),
        )
        norm = features.normalized_impulse
        for direction, side in (("LONG", 1), ("SHORT", -1)):
            directed_norm = float(side) * norm
            all_min_positions = np.flatnonzero(
                np.isfinite(directed_norm)
                & (directed_norm >= float(minimum_threshold))
                & activity["notional_feature_valid"]
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

                research_t = masks["research_mask"][all_t] & masks["causal_next_bar"][all_t]
                eligible_t = masks["eligible"][all_t]
                labels_t = activity["notional_bucket"][all_t]
                for bucket in ["ALL", *[name for name, _, _ in NOTIONAL_BANDS]]:
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
                                "notional_bucket": bucket,
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
                event_frame["impulse_window_notional"] = activity["impulse_window_notional"][eligible_positions]
                event_frame["impulse_window_trades_count"] = activity[
                    "impulse_window_trades_count"
                ][eligible_positions]
                event_frame["historical_window_notional_median"] = activity[
                    "historical_window_notional_median"
                ][eligible_positions]
                event_frame["notional_expansion_ratio"] = activity["notional_expansion_ratio"][eligible_positions]
                event_frame["notional_bucket"] = activity["notional_bucket"][eligible_positions]
                event_frame["notional_source_window_observed_flag"] = activity[
                    "notional_source_window_observed"
                ][eligible_positions]
                event_frame["notional_feature_valid_flag"] = activity["notional_feature_valid"][eligible_positions]
                event_frame["minimum_pool_threshold"] = float(minimum_threshold)
                event_frame["max_threshold_reached"] = np.asarray(
                    [max(t for t in thresholds if value >= float(t)) for value in directed_norm[eligible_positions]],
                    dtype=float,
                )

                for threshold in thresholds:
                    tag = _threshold_tag(threshold)
                    event_frame[f"event_{tag}_flag"] = threshold_masks_event[float(threshold)]
                    event_frame[f"deduplicated_{tag}_flag"] = dedup_masks_event[float(threshold)]

                labels = event_frame["notional_bucket"].to_numpy(dtype=object)
                for threshold in thresholds:
                    threshold_mask = threshold_masks_event[float(threshold)]
                    dedup_mask = dedup_masks_event[float(threshold)]
                    for bucket in ["ALL", *[name for name, _, _ in NOTIONAL_BANDS]]:
                        bucket_mask = np.ones(len(event_frame), dtype=bool) if bucket == "ALL" else labels == bucket
                        for event_set, event_set_mask in (("raw", threshold_mask), ("deduplicated", dedup_mask)):
                            base_mask = event_set_mask & bucket_mask
                            sr, yr, mr, tr = _summary_for_mask(
                                event_frame,
                                mask=base_mask,
                                direction=direction,
                                window=window,
                                threshold=threshold,
                                notional_bucket=bucket,
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
        del features, activity
    progress.close()

    print("[summaries] assembling activity gradients, threshold plateaus and dependency tables", flush=True)
    event_counts = pd.DataFrame(event_count_rows)
    summary = pd.DataFrame(summary_rows)
    yearly = pd.DataFrame(yearly_rows)
    monthly = pd.DataFrame(monthly_rows)
    notional_gradient = _build_notional_gradient(summary)
    threshold_plateau = _build_threshold_plateau(summary)
    top_dependency = pd.DataFrame(top_rows)

    if first_event_write:
        pd.DataFrame(columns=["event_id", "direction", "notional_bucket"]).to_csv(events_path, index=False)
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
        "notional_expansion_definition": {
            "numerator": "sum of OKX trade-bar quote notional over the current impulse window",
            "denominator": "trailing median of equal-length window notionals shifted by the full impulse window",
            "formula": "current_window_notional / rolling_median(window_notional.shift(window), activity_lookback)",
            "baseline_exclusion": "the complete current impulse window is excluded",
            "activity_lookback_bars": int(args.activity_lookback_bars),
            "activity_min_periods": int(args.activity_min_periods),
            "bands": [
                {
                    "name": name,
                    "lower": None if not math.isfinite(lower) else lower,
                    "upper": None if not math.isfinite(upper) else upper,
                }
                for name, lower, upper in NOTIONAL_BANDS
            ],
        },
        "current_impulse_normalization": {
            "formula": "impulse_return / (historical rolling std shifted by impulse_window * sqrt(window))",
            "vol_lookback_bars": int(args.vol_lookback_bars),
            "vol_min_periods": int(args.vol_min_periods),
        },
        "round02_persistence_condition_stacked": False,
        "deduplication": "threshold-specific same-direction stream; cooldown=impulse_window, then notional buckets are subsets",
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
        "research_boundary": "one mechanism only: impulse-window quote-notional expansion",
    }

    brief = _build_brief(summary, meta)
    artifacts = [
        (event_counts, out_dir / "01_event_counts.csv"),
        (summary, out_dir / "02_horizon_summary.csv"),
        (yearly, out_dir / "03_yearly.csv"),
        (monthly, out_dir / "04_monthly.csv"),
        (notional_gradient, out_dir / "07_notional_expansion_gradient.csv"),
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
    print("[self-test] deterministic quote-notional expansion study", flush=True)
    raw = r01._synthetic_bars()
    rng = np.random.default_rng(20260711)
    raw["trades_count"] = np.maximum(1, np.rint(raw["volume"].to_numpy(dtype=float) / 2.0)).astype(int)
    raw["notional"] = raw["volume"].to_numpy(dtype=float) * raw["close"].to_numpy(dtype=float) * 0.1
    # Add deterministic activity bursts around the synthetic shocks so every
    # fixed activity band is represented without changing price timing.
    burst = np.ones(len(raw), dtype=float)
    burst[1780:1820] *= 3.0
    burst[2580:2620] *= 2.2
    burst[4080:4120] *= 1.6
    burst[6180:6220] *= 0.6
    raw["notional"] *= burst
    raw["trades_count"] = np.maximum(1, np.rint(raw["trades_count"].to_numpy(dtype=float) * burst)).astype(int)
    raw = raw.drop(raw.index[3700:3707])
    bars = r01._regularize_trade_bar_axis(raw)
    log_path = Path(__file__).resolve().with_name("00_research_log.md")
    original_log = log_path.read_text(encoding="utf-8") if log_path.exists() else None
    try:
        with tempfile.TemporaryDirectory(prefix="dic_r03_") as tmp:
            args.out_dir = tmp
            args.warmup_start_date = "2022-12-20"
            args.start_date = "2022-12-23"
            args.end_date = "2022-12-24"
            args.vol_lookback_bars = 720
            args.vol_min_periods = 360
            args.activity_lookback_bars = 720
            args.activity_min_periods = 360
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
                "07_notional_expansion_gradient.csv",
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
            expected_bands = {name for name, _, _ in NOTIONAL_BANDS}
            if not expected_bands.issubset(set(summary["notional_bucket"].dropna().unique())):
                raise AssertionError("self-test did not retain all fixed notional bands")
            events = pd.read_csv(result["report_dir"] / "05_events.csv")
            if events.empty:
                raise AssertionError("self-test events are empty")
            for col in (
                "impulse_window_notional",
                "historical_window_notional_median",
                "notional_expansion_ratio",
                "notional_bucket",
            ):
                if col not in events.columns:
                    raise AssertionError(f"self-test events missing {col}")
            parsed_thresholds = r01._parse_float_csv(args.thresholds)
            for low, high in zip(parsed_thresholds[:-1], parsed_thresholds[1:], strict=False):
                low_flag = events[f"event_{_threshold_tag(low)}_flag"].astype(bool).to_numpy()
                high_flag = events[f"event_{_threshold_tag(high)}_flag"].astype(bool).to_numpy()
                if np.any(high_flag & ~low_flag):
                    raise AssertionError("nested threshold membership is not monotonic")
            audit = pd.read_csv(result["report_dir"] / "06_signal_audit.csv")
            if not audit.empty and audit["lookahead_flag"].astype(bool).any():
                raise AssertionError("self-test causal audit contains invalid events")
            meta = json.loads((result["report_dir"] / "10_run_meta.json").read_text(encoding="utf-8"))
            if int(meta.get("synthetic_gap_bar_count", 0)) != 7:
                raise AssertionError("self-test did not preserve expected gap bars")
            if bool(meta.get("round02_persistence_condition_stacked", True)):
                raise AssertionError("Round 03 accidentally stacked the Round-02 persistence condition")
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
