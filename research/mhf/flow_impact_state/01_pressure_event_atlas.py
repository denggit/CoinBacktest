#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""OKX Flow-Impact State Strategy — Round 01 pressure-event atlas.

Research question
-----------------
When aggressive OKX buy/sell pressure becomes historically abnormal, does the
next path behave like effective continuation or pressure exhaustion/reversal?

This is an event study, not a strategy backtest.  It intentionally contains no
TP/SL optimisation, no 4H hard gate, no Liquidity assumption, no position sizing
and no machine-learning selection.  The first round creates a high-recall event
universe and measures symmetric continuation/reversal paths.

Causal timing
-------------
- Input is local rich OKX trade bars only; ordinary OHLCV fallback is forbidden.
- Bars are left-labelled by start time and become available at start + timeframe.
- Pressure features use the current closed bar and older bars.
- Historical baselines are shifted by the complete pressure window.
- Entry is the next bar open, exactly when the signal bar has become available.
- Synthetic gap rows are allowed only to regularize calendar time and are
  excluded from pressure windows, entries and complete forward paths.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.flow_impact import (  # noqa: E402
    FlowImpactConfig,
    assign_pressure_event_clusters,
    build_flow_impact_features,
    detect_pressure_events,
    flow_field_coverage,
    pressure_strength_labels,
    regularize_trade_bar_axis,
    response_state_labels,
)
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.flow_impact_outcomes import future_path_outcomes  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "01_pressure_event_atlas"
SCRIPT_VERSION = "1.0.1"
EXPERIMENT_ID = "ETH_MHF_FLOW_IMPACT_STATE_R01"
EDGE_ID = "ETH_MHF_FLOW_IMPACT_STATE"
TITLE = "OKX Active Flow Pressure - Price Impact State Atlas"
DEFAULT_OUT_DIR = "data/reports/research/mhf/flow_impact_state/01_pressure_event_atlas"
DEFAULT_WINDOWS = (1, 3, 5)
DEFAULT_HORIZON_MINUTES = (1, 2, 5, 15, 30)
DEFAULT_FIRST_TOUCH_BPS = (15.0, 25.0, 50.0)
DEFAULT_FREQUENCY_THRESHOLDS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal OKX active-flow pressure / price-response event atlas.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--timeframe", default="1m", help="Long-history base defaults to 1m; 5s can be run as a later micro validation.")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--trade-bar-db-name", default="okx_trade_bars.db")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--pressure-windows", default=",".join(map(str, DEFAULT_WINDOWS)))
    parser.add_argument("--horizon-minutes", default=",".join(map(str, DEFAULT_HORIZON_MINUTES)))
    parser.add_argument("--first-touch-bps", default=",".join(map(str, DEFAULT_FIRST_TOUCH_BPS)))
    parser.add_argument("--frequency-thresholds", default=",".join(map(str, DEFAULT_FREQUENCY_THRESHOLDS)), help="Fixed pressure-z thresholds evaluated by frequency only, never selected from forward returns.")
    parser.add_argument("--baseline-bars", type=int, default=1440)
    parser.add_argument("--baseline-min-periods", type=int, default=720)
    parser.add_argument("--min-pressure-z", type=float, default=1.5)
    parser.add_argument("--cooldown-multiplier", type=float, default=1.0)
    parser.add_argument("--release-pressure-z", type=float, default=0.5)
    parser.add_argument("--entry-fee-rate", type=float, default=0.00055)
    parser.add_argument("--exit-fee-rate", type=float, default=0.00055)
    parser.add_argument("--entry-slippage", type=float, default=0.00020)
    parser.add_argument("--exit-slippage", type=float, default=0.00020)
    parser.add_argument("--event-sample-rows", type=int, default=20_000)
    parser.add_argument("--write-full-events", action="store_true")
    parser.add_argument("--skip-review-pack", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def _parse_int_csv(value: str) -> tuple[int, ...]:
    parsed = tuple(sorted(set(int(part.strip()) for part in str(value).split(",") if part.strip())))
    if not parsed or any(v <= 0 for v in parsed):
        raise ValueError(f"expected positive integer CSV, got: {value!r}")
    return parsed


def _parse_float_csv(value: str) -> tuple[float, ...]:
    parsed = tuple(sorted(set(float(part.strip()) for part in str(value).split(",") if part.strip())))
    if not parsed or any(v <= 0 for v in parsed):
        raise ValueError(f"expected positive float CSV, got: {value!r}")
    return parsed


def _timeframe_delta(timeframe: str) -> pd.Timedelta:
    text = str(timeframe).strip()
    if len(text) < 2 or not text[:-1].isdigit():
        raise ValueError(f"invalid timeframe: {timeframe!r}")
    amount = int(text[:-1])
    unit = text[-1].lower()
    aliases = {"s": "s", "m": "min", "h": "h", "d": "D"}
    if unit not in aliases or amount <= 0:
        raise ValueError(f"unsupported timeframe: {timeframe!r}")
    return pd.Timedelta(amount, unit=aliases[unit])


def _inclusive_end(value: str | pd.Timestamp, bar_delta: pd.Timedelta) -> pd.Timestamp:
    if isinstance(value, str) and len(value.strip()) == 10:
        return pd.Timestamp(value) + pd.Timedelta(days=1) - bar_delta
    return pd.Timestamp(value)


def _study_months(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return max(1.0, (end - start + pd.Timedelta(days=1)).total_seconds() / (365.2425 / 12.0 * 86400.0))


def _profit_factor(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    gains = float(x[x > 0.0].sum())
    losses = float(-x[x <= 0.0].sum())
    if losses <= 0.0:
        return float("inf") if gains > 0.0 else float("nan")
    return gains / losses


def _return_stats(values: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {
            "events": 0,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
            "top5_winner_share": np.nan,
        }
    winners = x[x > 0.0].sort_values(ascending=False)
    top5_share = float(winners.head(5).sum() / winners.sum()) if float(winners.sum()) > 0.0 else np.nan
    return {
        "events": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "win_rate": float((x > 0.0).mean()),
        "profit_factor": _profit_factor(x),
        "p05": float(x.quantile(0.05)),
        "p25": float(x.quantile(0.25)),
        "p75": float(x.quantile(0.75)),
        "p95": float(x.quantile(0.95)),
        "top5_winner_share": top5_share,
    }


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    bar_delta = _timeframe_delta(args.timeframe)
    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    db_path = data_dir / args.trade_bar_db_name
    if not db_path.exists():
        raise FileNotFoundError(
            f"Local trade-bar DB not found: {db_path}. This research is cache-only and will not download/build missing data."
        )
    loader = OKXTradeBarLoader(
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_dir=data_dir,
        db_name=args.trade_bar_db_name,
    )
    load_end = _inclusive_end(args.end_date, bar_delta)
    print(f"[load] local OKX trade bars {args.warmup_start_date} -> {load_end} timeframe={args.timeframe}", flush=True)
    bars = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        load_end,
        build_missing=False,
        force_rebuild=False,
        cvd_mode="range",
    )
    if bars.empty:
        raise RuntimeError("Local OKX trade-bar query returned no rows")
    bars = bars.loc[(bars.index >= pd.Timestamp(args.warmup_start_date)) & (bars.index <= load_end)].copy()
    return regularize_trade_bar_axis(bars, bar_delta=bar_delta)


def _frequency_summary(events: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    calendar_days = int((end.normalize() - start.normalize()).days + 1)
    months = _study_months(start, end)
    rows: list[dict[str, Any]] = []
    groupings: list[tuple[str, list[str]]] = [
        ("all", []),
        ("window", ["pressure_window_bars"]),
        ("window_side", ["pressure_window_bars", "side_name"]),
        ("window_side_response", ["pressure_window_bars", "side_name", "response_state"]),
    ]
    for scope, columns in groupings:
        grouped: Iterable[tuple[Any, pd.DataFrame]]
        if columns:
            grouped = events.groupby(columns, dropna=False, observed=False)
        else:
            grouped = [((), events)]
        for key, part in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            row: dict[str, Any] = {"scope": scope}
            row.update(dict(zip(columns, key_tuple, strict=False)))
            dates = pd.to_datetime(part["signal_bar_start"]).dt.normalize().sort_values().drop_duplicates()
            gaps = dates.diff().dt.total_seconds().div(86400.0).dropna()
            row.update(
                {
                    "events": int(len(part)),
                    "events_per_calendar_day": float(len(part) / max(calendar_days, 1)),
                    "events_per_month": float(len(part) / months),
                    "active_dates": int(len(dates)),
                    "active_date_ratio": float(len(dates) / max(calendar_days, 1)),
                    "longest_gap_days_between_events": float(gaps.max()) if not gaps.empty else np.nan,
                    "median_events_on_active_date": float(part.groupby("date").size().median()),
                    "p95_events_on_active_date": float(part.groupby("date").size().quantile(0.95)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)



def _threshold_frequency_calibration(
    features: pd.DataFrame,
    *,
    windows: tuple[int, ...],
    thresholds: tuple[float, ...],
    cooldown_multiplier: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Evaluate fixed event thresholds using frequency only, without outcomes."""
    rows: list[dict[str, Any]] = []
    calendar_days = int((end.normalize() - start.normalize()).days + 1)
    months = _study_months(start, end)
    for threshold in thresholds:
        events = detect_pressure_events(
            features,
            windows=windows,
            min_pressure_z=float(threshold),
            cooldown_multiplier=float(cooldown_multiplier),
        )
        if events.empty:
            rows.append(
                {
                    "pressure_z_threshold": float(threshold),
                    "window_events": 0,
                    "unique_pressure_clusters": 0,
                    "events_per_calendar_day": 0.0,
                    "events_per_month": 0.0,
                    "active_dates": 0,
                    "active_date_ratio": 0.0,
                    "frequency_target_10_30_per_day": False,
                }
            )
            continue
        events = events.loc[
            (pd.to_datetime(events["signal_time"]) >= start)
            & (pd.to_datetime(events["signal_time"]) <= end)
        ].copy()
        events = assign_pressure_event_clusters(events, cluster_gap_bars=max(windows))
        primary = events.loc[events["cluster_primary_flag"].astype(bool)].copy()
        active_dates = int(pd.to_datetime(primary["signal_time"]).dt.normalize().nunique())
        per_day = float(len(primary) / max(calendar_days, 1))
        rows.append(
            {
                "pressure_z_threshold": float(threshold),
                "window_events": int(len(events)),
                "unique_pressure_clusters": int(len(primary)),
                "events_per_calendar_day": per_day,
                "events_per_month": float(len(primary) / months),
                "active_dates": active_dates,
                "active_date_ratio": float(active_dates / max(calendar_days, 1)),
                "frequency_target_10_30_per_day": bool(10.0 <= per_day <= 30.0),
            }
        )
    return pd.DataFrame(rows)


def _yearly_frequency(events: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for year, part in events.groupby("year", observed=False):
        year_start = max(start.normalize(), pd.Timestamp(year=int(year), month=1, day=1))
        year_end = min(end.normalize(), pd.Timestamp(year=int(year), month=12, day=31))
        calendar_days = int((year_end - year_start).days + 1)
        active_dates = int(pd.to_datetime(part["signal_bar_start"]).dt.normalize().nunique())
        dates = pd.to_datetime(part["signal_bar_start"]).dt.normalize().sort_values().drop_duplicates()
        gaps = dates.diff().dt.total_seconds().div(86400.0).dropna()
        rows.append(
            {
                "year": int(year),
                "events": int(len(part)),
                "events_per_calendar_day": float(len(part) / max(calendar_days, 1)),
                "active_dates": active_dates,
                "active_date_ratio": float(active_dates / max(calendar_days, 1)),
                "longest_gap_days_between_events": float(gaps.max()) if not gaps.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)

def _summarize_returns(
    events: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    group_cols: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped: Iterable[tuple[Any, pd.DataFrame]]
    if group_cols:
        grouped = events.groupby(list(group_cols), dropna=False, observed=False)
    else:
        grouped = [((), events)]
    for key, part in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        group_values = dict(zip(group_cols, key_tuple, strict=False))
        for horizon in horizons:
            for branch in ("continuation", "reversal"):
                for cost_name, column in (
                    ("gross", f"{branch}_gross_h{horizon}"),
                    ("fee_only", f"{branch}_fee_net_h{horizon}"),
                    ("normal", f"{branch}_net_h{horizon}"),
                ):
                    stats = _return_stats(part[column])
                    rows.append(
                        {
                            **group_values,
                            "horizon_bars": int(horizon),
                            "branch": branch,
                            "cost_model": cost_name,
                            **stats,
                        }
                    )
    return pd.DataFrame(rows)


def _first_touch_by_group(events: pd.DataFrame, levels: tuple[float, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["pressure_window_bars", "side_name", "response_state"]
    for key, part in events.groupby(group_cols, dropna=False, observed=False):
        base = dict(zip(group_cols, key, strict=False))
        for level in levels:
            column = f"first_touch_{level:g}bps"
            counts = part[column].value_counts()
            total = int(counts.drop(labels=["invalid_path"], errors="ignore").sum())
            rows.append(
                {
                    **base,
                    "touch_bps": float(level),
                    "events": total,
                    "favorable_first_rate": float(counts.get("favorable_first", 0) / total) if total else np.nan,
                    "adverse_first_rate": float(counts.get("adverse_first", 0) / total) if total else np.nan,
                    "both_same_bar_rate": float(counts.get("both_same_bar", 0) / total) if total else np.nan,
                    "none_rate": float(counts.get("none", 0) / total) if total else np.nan,
                    "directional_first_touch_gap": float((counts.get("favorable_first", 0) - counts.get("adverse_first", 0)) / total) if total else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _duration_summary(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["pressure_window_bars", "side_name", "response_state"]
    for key, part in events.groupby(group_cols, dropna=False, observed=False):
        values = pd.to_numeric(part["pressure_state_duration_minutes"], errors="coerce").dropna()
        rows.append(
            {
                **dict(zip(group_cols, key, strict=False)),
                "events": int(len(values)),
                "mean_duration_minutes": float(values.mean()) if len(values) else np.nan,
                "median_duration_minutes": float(values.median()) if len(values) else np.nan,
                "p75_duration_minutes": float(values.quantile(0.75)) if len(values) else np.nan,
                "p95_duration_minutes": float(values.quantile(0.95)) if len(values) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _feature_dictionary() -> pd.DataFrame:
    rows = [
        ("pressure_z", "log(abs rolling signed notional) versus a historical mean/std ending before the complete pressure window", "causal feature"),
        ("flow_ratio", "rolling signed notional / rolling total notional", "causal feature"),
        ("trade_imbalance", "rolling aggressive buy-minus-sell trade count / total trade count", "causal feature"),
        ("large_flow_ratio", "rolling large signed notional / rolling large total notional", "causal feature"),
        ("large_notional_share", "rolling large-trade notional / rolling total notional", "causal feature"),
        ("large_trade_share", "rolling large-trade count / rolling total trade count", "causal feature"),
        ("flow_concentration", "abs rolling net signed flow / rolling sum abs per-bar signed flow", "causal feature"),
        ("flow_persistence", "share-weighted direction agreement of per-bar signed delta with event direction", "causal feature"),
        ("notional_ratio", "rolling total notional / prior rolling median total notional", "causal feature"),
        ("avg_trade_notional_ratio", "rolling average trade notional / prior rolling median", "causal feature"),
        ("max_trade_notional_ratio", "rolling maximum trade notional / prior rolling median", "causal feature"),
        ("activity_z", "log rolling notional versus prior historical mean/std", "causal feature"),
        ("price_response", "event-direction adjusted return over the pressure window", "causal feature"),
        ("price_response_norm", "event-direction adjusted return / prior volatility scaled to pressure window", "causal feature"),
        ("pressure_effectiveness", "normalized price response / pressure-z magnitude", "causal feature"),
        ("response_state", "fixed semantic bucket of price_response_norm", "causal state"),
        ("post_flow_ratio_hN", "event-direction adjusted signed flow after next-open entry", "future outcome only"),
        ("pressure_state_duration", "future bars until pressure drops below release level or direction flips", "future outcome only"),
        ("continuation/reversal returns", "symmetric path returns from next bar open", "future outcome only"),
        ("MFE/MAE and first touch", "future high/low path from next bar open; same-bar dual touch is never resolved optimistically", "future outcome only"),
    ]
    return pd.DataFrame(rows, columns=["field", "definition", "role"])


def _deterministic_event_sample(events: pd.DataFrame, limit: int) -> pd.DataFrame:
    if len(events) <= int(limit):
        return events.copy()
    strongest = events.nlargest(max(1, int(limit) // 3), "pressure_z")
    remainder_limit = int(limit) - len(strongest)
    remainder = events.drop(index=strongest.index)
    sampled = remainder.sample(n=min(remainder_limit, len(remainder)), random_state=20260725)
    return pd.concat([strongest, sampled], ignore_index=True).sort_values("signal_bar_start")


def _build_brief(
    events: pd.DataFrame,
    frequency: pd.DataFrame,
    response_summary: pd.DataFrame,
    yearly: pd.DataFrame,
    *,
    normal_cost: float,
) -> str:
    all_freq = frequency[frequency["scope"] == "all"].iloc[0] if not frequency.empty else None
    normal = response_summary[response_summary["cost_model"] == "normal"].copy()
    viable = normal[
        (normal["events"] >= 1000)
        & (normal["mean"] > 0.0)
        & (normal["profit_factor"] > 1.0)
    ].copy()
    stable_rows: list[dict[str, Any]] = []
    if not viable.empty and not yearly.empty:
        yearly_normal = yearly[yearly["cost_model"] == "normal"].copy()
        key_cols = ["pressure_window_bars", "side_name", "response_state", "horizon_bars", "branch"]
        for row in viable.itertuples(index=False):
            mask = np.ones(len(yearly_normal), dtype=bool)
            for col in key_cols:
                mask &= yearly_normal[col].eq(getattr(row, col)).to_numpy()
            part = yearly_normal.loc[mask]
            positive_years = int((part["mean"] > 0.0).sum())
            if positive_years >= 3:
                stable_rows.append({**{col: getattr(row, col) for col in key_cols}, "events": int(row.events), "mean": float(row.mean), "median": float(row.median), "pf": float(row.profit_factor), "positive_years": positive_years})
    stable_rows = sorted(stable_rows, key=lambda item: (item["positive_years"], item["pf"], item["mean"]), reverse=True)

    lines = [
        "# Round 01 Research Brief",
        "",
        "## What this round does",
        "",
        "This round builds a high-recall active-flow pressure universe and measures both effective continuation and pressure-exhaustion reversal. It is not a tradable strategy and does not optimise exits.",
        "",
        "## Frequency",
        "",
    ]
    if all_freq is not None:
        lines.extend(
            [
                f"- Pressure events: **{int(all_freq.events):,}**",
                f"- Events/calendar day: **{float(all_freq.events_per_calendar_day):.2f}**",
                f"- Events/month: **{float(all_freq.events_per_month):.1f}**",
                f"- Active-date ratio: **{float(all_freq.active_date_ratio):.1%}**",
                f"- Longest gap between event dates: **{float(all_freq.longest_gap_days_between_events):.1f} days**",
            ]
        )
    lines.extend(
        [
            "",
            "The event universe is intentionally broader than the final 1-3 trades/day target. A useful first-round universe should normally produce roughly 10-30 candidate pressure events/day before branch selection.",
            "",
            "## Cost convention",
            "",
            f"Normal round-trip cost is **{normal_cost:.3%}** (OKX fee convention plus conservative slippage).",
            "",
            "## Stable descriptive cells",
            "",
        ]
    )
    if stable_rows:
        for row in stable_rows[:10]:
            lines.append(
                f"- {row['side_name']} w{row['pressure_window_bars']} {row['response_state']} -> {row['branch']} h{row['horizon_bars']}: "
                f"events={row['events']:,}, mean_net={row['mean']:.4%}, median_net={row['median']:.4%}, PF={row['pf']:.3f}, positive_years={row['positive_years']}."
            )
        lines.extend(
            [
                "",
                "These are mechanism candidates only. The next round must validate neighboring response states, first-touch ordering, pressure persistence and delayed execution before any TP/SL backtest.",
            ]
        )
    else:
        lines.extend(
            [
                "No response-state cell simultaneously reached 1,000 events, positive mean after normal costs, PF above 1, and at least three positive years.",
                "",
                "That does not yet reject the family: fixed-horizon return may miss a short-lived state transition. The first-touch and pressure-duration tables should determine whether a narrower causal trigger is justified or whether this family should stop.",
            ]
        )
    lines.extend(
        [
            "",
            "## Hard boundaries for Round 02",
            "",
            "1. Do not tune TP/SL from the best cell.",
            "2. Keep LONG/SHORT and continuation/reversal symmetric.",
            "3. Select at most one mechanism difference: impact efficiency, pressure persistence, or first-touch ordering.",
            "4. Liquidity primitives may be added only after the long-history mechanism exists, and only as an incremental comparison on 2025-11-02 to 2026-06-30.",
            "5. 4H context may only be a continuous risk feature, never a hard signal gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    windows = _parse_int_csv(args.pressure_windows)
    horizon_minutes = _parse_int_csv(args.horizon_minutes)
    touch_levels = _parse_float_csv(args.first_touch_bps)
    frequency_thresholds = _parse_float_csv(args.frequency_thresholds)
    bar_delta = _timeframe_delta(args.timeframe)
    bar_minutes = bar_delta.total_seconds() / 60.0
    horizons_bars = tuple(int(round(minutes / bar_minutes)) for minutes in horizon_minutes)
    if any(abs(horizon * bar_minutes - minutes) > 1e-9 for horizon, minutes in zip(horizons_bars, horizon_minutes, strict=True)):
        raise ValueError("Every horizon minute must be an exact multiple of the selected timeframe")

    cfg = FlowImpactConfig(
        pressure_windows=windows,
        baseline_bars=int(args.baseline_bars),
        baseline_min_periods=int(args.baseline_min_periods),
        min_pressure_z=float(args.min_pressure_z),
        event_cooldown_multiplier=float(args.cooldown_multiplier),
    )
    cfg.validate()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    coverage = flow_field_coverage(bars)
    print(f"[features] rows={len(bars):,} windows={windows}", flush=True)
    features = build_flow_impact_features(bars, cfg)
    print("[events] pressure onset detection", flush=True)
    events = detect_pressure_events(
        features,
        windows=windows,
        min_pressure_z=float(args.min_pressure_z),
        cooldown_multiplier=float(args.cooldown_multiplier),
    )
    if events.empty:
        raise RuntimeError("No pressure events detected; inspect field coverage and pressure-z distribution")

    start = pd.Timestamp(args.start_date)
    end = _inclusive_end(args.end_date, bar_delta)
    events = events.loc[(pd.to_datetime(events["signal_time"]) >= start) & (pd.to_datetime(events["signal_time"]) <= end)].copy()
    if events.empty:
        raise RuntimeError("Pressure events exist in warmup but none fall inside the research window")
    events = assign_pressure_event_clusters(events, cluster_gap_bars=max(windows))

    fee_only_cost = float(args.entry_fee_rate + args.exit_fee_rate)
    normal_cost = float(fee_only_cost + args.entry_slippage + args.exit_slippage)
    events, overall_first_touch, audit = future_path_outcomes(
        bars,
        features,
        events,
        horizons_bars=horizons_bars,
        touch_levels_bps=touch_levels,
        normal_cost=normal_cost,
        fee_only_cost=fee_only_cost,
        release_pressure_z=float(args.release_pressure_z),
        bar_delta=bar_delta,
        progress_enabled=not args.no_progress,
    )
    valid_events = events.loc[~audit["causal_or_data_fail_flag"].to_numpy(dtype=bool)].copy()
    if valid_events.empty:
        raise RuntimeError("All pressure events failed causal/data-path audit")
    valid_events = valid_events.sort_values(["signal_bar_pos", "pressure_window_bars", "event_id"], kind="stable").reset_index(drop=True)
    valid_events["cluster_primary_flag"] = ~valid_events["event_cluster_id"].duplicated(keep="first")
    valid_events["valid_cluster_size"] = valid_events.groupby("event_cluster_id", observed=False)["event_cluster_id"].transform("size").astype(int)
    primary_events = valid_events.loc[valid_events["cluster_primary_flag"].astype(bool)].copy()

    print(f"[summary] valid_events={len(valid_events):,} excluded={len(events) - len(valid_events):,}", flush=True)
    frequency = _frequency_summary(primary_events, start=start, end=end)
    threshold_frequency = _threshold_frequency_calibration(
        features,
        windows=windows,
        thresholds=frequency_thresholds,
        cooldown_multiplier=float(args.cooldown_multiplier),
        start=start,
        end=end,
    )
    yearly_frequency = _yearly_frequency(primary_events, start=start, end=end)
    unique_cluster_summary = _summarize_returns(
        primary_events,
        horizons=horizons_bars,
        group_cols=("side_name", "response_state"),
    )
    base_summary = _summarize_returns(
        valid_events,
        horizons=horizons_bars,
        group_cols=("pressure_window_bars", "side_name"),
    )
    response_summary = _summarize_returns(
        valid_events,
        horizons=horizons_bars,
        group_cols=("pressure_window_bars", "side_name", "response_state"),
    )
    strength_summary = _summarize_returns(
        valid_events,
        horizons=horizons_bars,
        group_cols=("pressure_window_bars", "side_name", "pressure_strength"),
    )
    yearly = _summarize_returns(
        valid_events,
        horizons=horizons_bars,
        group_cols=("pressure_window_bars", "side_name", "response_state", "year"),
    )
    monthly = _summarize_returns(
        valid_events,
        horizons=horizons_bars,
        group_cols=("pressure_window_bars", "side_name", "response_state", "month"),
    )
    first_touch = _first_touch_by_group(valid_events, touch_levels)
    duration = _duration_summary(valid_events)
    daily_counts = (
        valid_events.groupby(["date", "pressure_window_bars", "side_name"], observed=False)
        .size()
        .rename("events")
        .reset_index()
    )
    event_sample = _deterministic_event_sample(valid_events, int(args.event_sample_rows))
    feature_dictionary = _feature_dictionary()
    brief = _build_brief(valid_events, frequency, response_summary, yearly, normal_cost=normal_cost)

    artifact_frames = [
        (coverage, out_dir / "01_field_coverage.csv"),
        (frequency, out_dir / "02_event_frequency.csv"),
        (threshold_frequency, out_dir / "02b_threshold_frequency_calibration.csv"),
        (yearly_frequency, out_dir / "02c_yearly_event_frequency.csv"),
        (unique_cluster_summary, out_dir / "03_unique_cluster_path_summary.csv"),
        (base_summary, out_dir / "03b_window_path_summary.csv"),
        (response_summary, out_dir / "04_response_state_summary.csv"),
        (strength_summary, out_dir / "05_pressure_strength_summary.csv"),
        (yearly, out_dir / "06_yearly_response_summary.csv"),
        (monthly, out_dir / "07_monthly_response_summary.csv"),
        (first_touch, out_dir / "08_first_touch_by_state.csv"),
        (overall_first_touch, out_dir / "08b_first_touch_overall.csv"),
        (duration, out_dir / "09_pressure_duration_summary.csv"),
        (daily_counts, out_dir / "10_daily_event_counts.csv"),
        (event_sample, out_dir / "11_event_sample.csv"),
        (audit, out_dir / "12_signal_audit.csv"),
        (feature_dictionary, out_dir / "13_feature_dictionary.csv"),
    ]
    reporter = ProgressReporter("[artifacts] tables", len(artifact_frames), every=1, enabled=not args.no_progress)
    for done, (frame, path) in enumerate(artifact_frames, start=1):
        frame.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
        reporter.update(done)
    reporter.close()

    if args.write_full_events:
        events.to_csv(out_dir / "11b_full_events.csv.gz", index=False, compression="gzip", float_format="%.10g")

    meta = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "status": "research_only_not_tradable",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "bar_delta": str(bar_delta),
        "data_source": "src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader",
        "local_cache_only": True,
        "build_missing": False,
        "warmup_start_date": args.warmup_start_date,
        "research_start": args.start_date,
        "research_end": args.end_date,
        "pressure_windows": list(windows),
        "horizon_minutes": list(horizon_minutes),
        "horizon_bars": list(horizons_bars),
        "first_touch_bps": list(touch_levels),
        "frequency_thresholds": list(frequency_thresholds),
        "threshold_selection_policy": "frequency-only calibration; forward returns are forbidden for choosing the broad event threshold",
        "baseline_bars": int(args.baseline_bars),
        "baseline_min_periods": int(args.baseline_min_periods),
        "min_pressure_z": float(args.min_pressure_z),
        "cooldown_multiplier": float(args.cooldown_multiplier),
        "release_pressure_z": float(args.release_pressure_z),
        "fee_only_cost": fee_only_cost,
        "normal_execution_cost": normal_cost,
        "input_rows": int(len(bars)),
        "source_observed_rows": int(bars["source_bar_observed_flag"].sum()),
        "synthetic_gap_bars": int((~bars["source_bar_observed_flag"].astype(bool)).sum()),
        "raw_pressure_window_events": int(len(events)),
        "valid_pressure_window_events": int(len(valid_events)),
        "unique_pressure_clusters": int(len(primary_events)),
        "causal_or_data_fail_events": int(audit["causal_or_data_fail_flag"].sum()),
        "entry_policy": "closed pressure signal bar -> immediate next bar open",
        "event_definition": "pressure-z onset or direction flip above minimum threshold; no price-direction gate",
        "response_state_definition": "predeclared buckets of causal direction-adjusted price response normalized by prior volatility",
        "future_outcome_boundary": "post-flow, duration, fixed returns, MFE/MAE and first-touch fields are labels only",
        "ordinary_kline_fallback": False,
        "liquidity_used": False,
        "high_timeframe_hard_gate_used": False,
        "tp_sl_optimized": False,
        "created_at": pd.Timestamp.now("UTC").isoformat(),
    }
    (out_dir / "14_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "15_research_brief.md").write_text(brief, encoding="utf-8")
    if not args.skip_review_pack:
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report_dir={out_dir}", flush=True)
    return {"report_dir": out_dir, "review_pack": out_dir / "gpt_review_pack.zip"}


def _synthetic_bars() -> pd.DataFrame:
    rng = np.random.default_rng(20260725)
    n = 25_000
    index = pd.date_range("2022-12-01", periods=n, freq="1min")
    base_ret = rng.normal(0.0, 0.00018, n)
    delta = rng.normal(0.0, 80_000.0, n)
    notional = rng.lognormal(mean=13.0, sigma=0.45, size=n)
    patterns: list[tuple[int, int, str]] = []
    for i, pos in enumerate(range(2500, 22_000, 900)):
        side = 1 if i % 2 == 0 else -1
        kind = "effective" if i % 3 else "absorbed"
        delta[pos : pos + 3] += side * np.array([700_000.0, 1_200_000.0, 900_000.0])
        if kind == "effective":
            base_ret[pos : pos + 3] += side * np.array([0.0005, 0.0007, 0.0005])
            base_ret[pos + 3 : pos + 12] += side * 0.00012
        else:
            base_ret[pos : pos + 3] += side * np.array([0.00008, -0.00004, 0.00002])
            base_ret[pos + 3 : pos + 12] -= side * 0.00014
        patterns.append((pos, side, kind))
    close = 1800.0 * np.exp(np.cumsum(base_ret))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0.00012, 0.00004, n))
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    buy = np.maximum(notional * 0.05, (notional + delta) / 2.0)
    sell = np.maximum(notional * 0.05, (notional - delta) / 2.0)
    notional = buy + sell
    delta = buy - sell
    trades = np.maximum(10, (notional / 3000.0).astype(int))
    buy_ratio = np.clip(0.5 + 0.45 * delta / np.maximum(notional, 1.0), 0.02, 0.98)
    buy_trades = np.round(trades * buy_ratio).astype(int)
    sell_trades = trades - buy_trades
    large_share = np.clip(np.abs(delta) / np.maximum(notional, 1.0), 0.02, 0.70)
    large_buy = np.where(delta > 0, np.abs(delta) * large_share, np.abs(delta) * 0.08)
    large_sell = np.where(delta < 0, np.abs(delta) * large_share, np.abs(delta) * 0.08)
    bars = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": notional / close,
            "notional": notional,
            "buy_notional": buy,
            "sell_notional": sell,
            "delta_notional": delta,
            "trades_count": trades,
            "buy_trades_count": buy_trades,
            "sell_trades_count": sell_trades,
            "large_buy_notional": large_buy,
            "large_sell_notional": large_sell,
            "large_delta_notional": large_buy - large_sell,
            "large_trades_count": np.maximum(1, (trades * large_share * 0.15).astype(int)),
            "max_trade_notional": np.maximum(1000.0, np.abs(delta) * 0.2),
        },
        index=index,
    )
    bars = bars.drop(bars.index[12_000:12_004])
    return regularize_trade_bar_axis(bars, bar_delta=pd.Timedelta(minutes=1))


def run_self_test(args: argparse.Namespace) -> int:
    print("[self-test] deterministic synthetic flow-impact atlas", flush=True)
    bars = _synthetic_bars()
    with tempfile.TemporaryDirectory(prefix="flow_impact_r01_") as tmp:
        args.out_dir = tmp
        args.timeframe = "1m"
        args.warmup_start_date = "2022-12-01"
        args.start_date = "2022-12-06"
        args.end_date = "2022-12-18"
        args.baseline_bars = 720
        args.baseline_min_periods = 360
        args.skip_review_pack = True
        args.no_progress = True
        args.event_sample_rows = 1000
        result = run_research(bars, args)
        required = [
            "01_field_coverage.csv",
            "02_event_frequency.csv",
            "02b_threshold_frequency_calibration.csv",
            "02c_yearly_event_frequency.csv",
            "03_unique_cluster_path_summary.csv",
            "03b_window_path_summary.csv",
            "04_response_state_summary.csv",
            "06_yearly_response_summary.csv",
            "08_first_touch_by_state.csv",
            "09_pressure_duration_summary.csv",
            "11_event_sample.csv",
            "12_signal_audit.csv",
            "14_manifest.json",
            "15_research_brief.md",
        ]
        missing = [name for name in required if not (result["report_dir"] / name).exists()]
        if missing:
            raise AssertionError(f"missing self-test artifacts: {missing}")
        audit = pd.read_csv(result["report_dir"] / "12_signal_audit.csv")
        valid_audit = audit.loc[~audit["synthetic_bar_dependency_flag"].astype(bool)]
        if valid_audit["entry_not_next_open_flag"].astype(bool).any():
            raise AssertionError("next-open audit failed")
        if valid_audit["entry_before_signal_available_flag"].astype(bool).any():
            raise AssertionError("entry occurred before signal availability")
        sample = pd.read_csv(result["report_dir"] / "11_event_sample.csv")
        if sample.empty or set(sample["side_name"].unique()) != {"LONG", "SHORT"}:
            raise AssertionError("self-test did not create symmetric LONG/SHORT pressure events")
        if not {"opposite_or_absorbed", "effective_ge_0.75"}.intersection(set(sample["response_state"].unique())):
            raise AssertionError("self-test response-state atlas is unexpectedly empty")
    print("[self-test] PASS", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args)
    bars = load_bars(args)
    run_research(bars, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
