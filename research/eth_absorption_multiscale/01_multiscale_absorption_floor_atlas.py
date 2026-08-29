#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R01 multi-scale ETH absorption / floor-defense / spring path atlas.

This is an event study, not a strategy backtest.  It deliberately studies the
user's "跌不动 / 涨不动" hypothesis as a *process* from 5-second order flow up
to multi-day structure:

1. strong active pressure with weak price progress (stall / rejection);
2. adjacent pressure windows where similar pressure has declining impact;
3. repeated tests of a previously-known floor/ceiling;
4. failed breakdown / spring and failed breakout / upthrust after a defended
   area has existed for minutes, hours, or days depending on scale.

Causality:
- all signal features use only closed bars up to the signal bar;
- higher-timeframe bars are locally aggregated from closed 1m trade bars;
- signal_time = left-labelled bar_start + timeframe;
- all forward outcomes enter at the next bar open;
- no future label participates in event detection or threshold selection.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.flow_impact import regularize_trade_bar_axis  # noqa: E402
from src.research_common.multiscale_absorption import (  # noqa: E402
    AbsorptionFeatureConfig,
    attach_forward_outcomes,
    build_absorption_features,
    extract_events,
    resample_trade_bars,
)
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "01_multiscale_absorption_floor_atlas"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MULTISCALE_ABSORPTION_FLOOR_ATLAS_R01"
EDGE_ID = "RESEARCH_ONLY_ETH_ABSORPTION_FLOOR_DEFENSE"
TITLE = "ETH Multi-scale Absorption / Floor Defense / Spring Atlas R01"
DEFAULT_OUT_DIR = "data/reports/research/eth_absorption_multiscale/01_multiscale_absorption_floor_atlas"


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    bar_delta: pd.Timedelta
    process_windows: tuple[int, ...]
    baseline_bars: int
    baseline_min_periods: int
    floor_lookback: int
    defense_lookback: int
    reclaim_bars: int
    atr_lookback: int
    outcome_horizons: tuple[int, ...]
    floor_lookback_label: str
    source: str = "1m"
    resample_rule: str | None = None

    def horizon_label(self, bars: int) -> str:
        delta = self.bar_delta * int(bars)
        seconds = int(delta.total_seconds())
        if seconds < 60:
            return f"{seconds}s"
        if seconds < 3600:
            return f"{seconds // 60}m"
        if seconds < 86400:
            return f"{seconds / 3600:g}h"
        return f"{seconds / 86400:g}d"


SCALE_SPECS: tuple[ScaleSpec, ...] = (
    ScaleSpec(
        name="5s",
        bar_delta=pd.Timedelta(seconds=5),
        process_windows=(6, 12, 36),      # 30s / 1m / 3m
        baseline_bars=4320,               # 6h
        baseline_min_periods=2160,
        floor_lookback=360,               # 30m
        defense_lookback=360,
        reclaim_bars=12,                  # reclaim inside 60s
        atr_lookback=720,
        outcome_horizons=(6, 12, 60, 180),
        floor_lookback_label="30m",
        source="5s",
    ),
    ScaleSpec(
        name="1m",
        bar_delta=pd.Timedelta(minutes=1),
        process_windows=(3, 5, 15),
        baseline_bars=2880,               # 2d
        baseline_min_periods=1440,
        floor_lookback=360,               # 6h
        defense_lookback=360,
        reclaim_bars=3,
        atr_lookback=120,
        outcome_horizons=(5, 15, 60, 240),
        floor_lookback_label="6h",
        source="1m",
    ),
    ScaleSpec(
        name="5m",
        bar_delta=pd.Timedelta(minutes=5),
        process_windows=(3, 6, 12),        # 15m / 30m / 1h
        baseline_bars=2016,                # 7d
        baseline_min_periods=1008,
        floor_lookback=144,                # 12h
        defense_lookback=144,
        reclaim_bars=3,
        atr_lookback=72,
        outcome_horizons=(3, 12, 48, 144),
        floor_lookback_label="12h",
        source="1m",
        resample_rule="5min",
    ),
    ScaleSpec(
        name="15m",
        bar_delta=pd.Timedelta(minutes=15),
        process_windows=(3, 6, 16),        # 45m / 1.5h / 4h
        baseline_bars=1344,                # 14d
        baseline_min_periods=672,
        floor_lookback=96,                 # 1d
        defense_lookback=96,
        reclaim_bars=3,
        atr_lookback=48,
        outcome_horizons=(4, 16, 48, 96),
        floor_lookback_label="1d",
        source="1m",
        resample_rule="15min",
    ),
    ScaleSpec(
        name="1H",
        bar_delta=pd.Timedelta(hours=1),
        process_windows=(3, 6, 12),
        baseline_bars=720,                 # 30d
        baseline_min_periods=360,
        floor_lookback=72,                 # 3d
        defense_lookback=72,
        reclaim_bars=3,
        atr_lookback=48,
        outcome_horizons=(4, 12, 24, 72),
        floor_lookback_label="3d",
        source="1m",
        resample_rule="1h",
    ),
    ScaleSpec(
        name="4H",
        bar_delta=pd.Timedelta(hours=4),
        process_windows=(3, 6, 12),        # 12h / 1d / 2d
        baseline_bars=540,                 # 90d
        baseline_min_periods=270,
        floor_lookback=84,                 # 14d
        defense_lookback=84,
        reclaim_bars=2,
        atr_lookback=42,
        outcome_horizons=(3, 6, 18, 42),  # 12h / 1d / 3d / 7d
        floor_lookback_label="14d",
        source="1m",
        resample_rule="4h",
    ),
)

ABSORPTION_PATTERNS = {
    "strong_pressure_control_fade",
    "strong_pressure_control_follow",
    "pressure_stall",
    "pressure_rejection",
    "impact_decay",
}
FLOOR_PATTERNS = {
    "floor_retest",
    "ceiling_retest",
    "spring_same_bar",
    "upthrust_same_bar",
    "spring_reclaim",
    "upthrust_reclaim",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--trade-bar-db-name", default="okx_trade_bars.db")
    p.add_argument("--round-trip-cost", type=float, default=0.0011)
    p.add_argument("--scales", nargs="+", default=[spec.name for spec in SCALE_SPECS])
    p.add_argument("--skip-micro-if-missing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--event-sample-per-group", type=int, default=40)
    return p.parse_args(argv)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _research_end_exclusive(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if len(str(value).strip()) == 10:
        ts += pd.Timedelta(days=1)
    else:
        ts += pd.Timedelta(microseconds=1)
    return ts


def _month_ranges(start: pd.Timestamp, end_exclusive: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start.normalize().replace(day=1)
    while cursor < end_exclusive:
        next_month = cursor + pd.offsets.MonthBegin(1)
        left = max(start, cursor)
        right = min(end_exclusive, next_month)
        if left < right:
            ranges.append((left, right))
        cursor = next_month
    return ranges


def _db_has_table(db_path: Path, table_name: str) -> bool:
    if not db_path.exists():
        return False
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
            (table_name,),
        ).fetchone()
        if not row:
            return False
        count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
    return int(count) > 0


def _load_trade_bars(
    *,
    symbol: str,
    timeframe: str,
    start: pd.Timestamp,
    end_exclusive: pd.Timestamp,
    data_dir: Path,
    db_name: str,
) -> pd.DataFrame:
    loader = OKXTradeBarLoader(symbol=symbol, timeframe=timeframe, data_dir=data_dir, db_name=db_name)
    end_inclusive = end_exclusive - pd.Timedelta(microseconds=1)
    bars = loader.fetch_data_by_date_range(
        start,
        end_inclusive,
        build_missing=False,
        force_rebuild=False,
        cvd_mode="range",
    )
    if bars.empty:
        return bars
    bars = bars.sort_index()
    bars.index = pd.to_datetime(bars.index)
    bars = bars[~bars.index.duplicated(keep="last")]
    return bars


def _prepare_scale_bars(source_1m_regular: pd.DataFrame, spec: ScaleSpec) -> pd.DataFrame:
    if spec.name == "1m":
        return source_1m_regular
    if not spec.resample_rule:
        raise ValueError(f"missing resample rule for {spec.name}")
    resampled = resample_trade_bars(source_1m_regular, spec.resample_rule)
    return regularize_trade_bar_axis(resampled, bar_delta=spec.bar_delta)


def _feature_config(spec: ScaleSpec, process_window: int) -> AbsorptionFeatureConfig:
    return AbsorptionFeatureConfig(
        process_window=int(process_window),
        baseline_bars=int(spec.baseline_bars),
        baseline_min_periods=int(spec.baseline_min_periods),
        floor_lookback=int(spec.floor_lookback),
        defense_lookback=int(spec.defense_lookback),
        reclaim_bars=int(spec.reclaim_bars),
        atr_lookback=int(spec.atr_lookback),
    )


def _trim_events(events: pd.DataFrame, left: pd.Timestamp, right: pd.Timestamp) -> pd.DataFrame:
    if events.empty:
        return events
    signal = pd.to_datetime(events["signal_time"])
    return events.loc[(signal >= left) & (signal < right)].copy()


def _profit_factor(x: pd.Series) -> float:
    values = pd.to_numeric(x, errors="coerce").dropna()
    if values.empty:
        return np.nan
    gain = float(values[values > 0.0].sum())
    loss = float(-values[values <= 0.0].sum())
    if loss <= 0.0:
        return float("inf") if gain > 0.0 else np.nan
    return gain / loss


def _stat_row(values: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {
            "events": 0,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "p10": np.nan,
            "p90": np.nan,
        }
    return {
        "events": int(len(x)),
        "mean": float(x.mean()),
        "median": float(x.median()),
        "win_rate": float((x > 0.0).mean()),
        "profit_factor": _profit_factor(x),
        "p10": float(x.quantile(0.10)),
        "p90": float(x.quantile(0.90)),
    }


def _summarize(
    events: pd.DataFrame,
    spec: ScaleSpec,
    group_cols: Sequence[str],
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    grouped: Iterable[tuple[object, pd.DataFrame]]
    if group_cols:
        grouped = events.groupby(list(group_cols), dropna=False, observed=False)
    else:
        grouped = [((), events)]
    for key, part in grouped:
        key_tuple = key if isinstance(key, tuple) else (key,)
        group_values = dict(zip(group_cols, key_tuple, strict=False))
        for horizon in spec.outcome_horizons:
            gross = _stat_row(part[f"gross_h{horizon}"])
            net = _stat_row(part[f"net_h{horizon}"])
            mfe = pd.to_numeric(part[f"mfe_h{horizon}"], errors="coerce")
            mae = pd.to_numeric(part[f"mae_h{horizon}"], errors="coerce")
            rows.append(
                {
                    "scale": spec.name,
                    **group_values,
                    "horizon_bars": int(horizon),
                    "horizon": spec.horizon_label(horizon),
                    "gross_events": gross["events"],
                    "gross_mean": gross["mean"],
                    "gross_median": gross["median"],
                    "gross_win_rate": gross["win_rate"],
                    "gross_pf": gross["profit_factor"],
                    "net_mean": net["mean"],
                    "net_median": net["median"],
                    "net_win_rate": net["win_rate"],
                    "net_pf": net["profit_factor"],
                    "mean_mfe": float(mfe.mean()) if mfe.notna().any() else np.nan,
                    "mean_mae": float(mae.mean()) if mae.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _yearly_summary(events: pd.DataFrame, spec: ScaleSpec) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    part = events.copy()
    part["year"] = pd.to_datetime(part["signal_time"]).dt.year
    return _summarize(part, spec, ["year", "pattern", "trade_side", "process_window"])


def _hold_bucket(values: pd.Series) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    out = pd.Series("NA", index=x.index, dtype=object)
    out.loc[x < 0.60] = "lt0.60"
    out.loc[(x >= 0.60) & (x < 0.80)] = "0.60_0.80"
    out.loc[(x >= 0.80) & (x < 0.95)] = "0.80_0.95"
    out.loc[x >= 0.95] = "ge0.95"
    return out


def _run_scale_on_bars(
    bars: pd.DataFrame,
    *,
    spec: ScaleSpec,
    month_left: pd.Timestamp,
    month_right: pd.Timestamp,
    round_trip_cost: float,
) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for process_i, process_window in enumerate(spec.process_windows):
        features = build_absorption_features(bars, _feature_config(spec, process_window))
        events = extract_events(
            features,
            scale=spec.name,
            bar_delta=spec.bar_delta,
            floor_lookback_label=spec.floor_lookback_label,
        )
        if events.empty:
            continue
        # Floor morphology does not depend on process_window; retain it once.
        if process_i > 0:
            events = events.loc[events["pattern"].isin(ABSORPTION_PATTERNS)].copy()
        events = _trim_events(events, month_left, month_right)
        if events.empty:
            continue
        events = attach_forward_outcomes(
            events,
            bars,
            horizons=spec.outcome_horizons,
            round_trip_cost=float(round_trip_cost),
        )
        events["process_window_label"] = spec.horizon_label(int(process_window))
        outputs.append(events)
        del features, events
        gc.collect()
    if not outputs:
        return pd.DataFrame()
    out = pd.concat(outputs, ignore_index=True, sort=False)
    out["hold_ratio_bucket"] = _hold_bucket(out["hold_ratio"])
    return out


def _select_event_sample(events: pd.DataFrame, limit_per_group: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    sample_cols = [
        "scale",
        "pattern",
        "signal_bar_start",
        "signal_time",
        "entry_time",
        "entry_price",
        "trade_side",
        "process_window",
        "process_window_label",
        "floor_lookback_label",
        "pressure_z",
        "pressure",
        "flow_persistence",
        "price_response_norm",
        "pressure_retention",
        "response_retention",
        "prior_defense_count",
        "defense_count_bucket",
        "zone_stability_bucket",
        "hold_ratio",
        "prior_floor",
        "prior_ceiling",
    ]
    outcome_cols = [c for c in events.columns if c.startswith(("gross_h", "net_h", "mfe_h", "mae_h"))]
    sample_cols.extend(outcome_cols)
    sample_cols = [c for c in sample_cols if c in events.columns]
    ranked = events.copy()
    ranked["sample_score"] = pd.to_numeric(ranked.get("pressure_z"), errors="coerce").fillna(0.0).abs()
    sampled = (
        ranked.sort_values("sample_score", ascending=False)
        .groupby(["scale", "pattern"], observed=False, dropna=False)
        .head(max(1, int(limit_per_group)))
    )
    return sampled[sample_cols].sort_values(["scale", "pattern", "signal_time"])


def _causal_audit_samples(events: pd.DataFrame, specs: dict[str, ScaleSpec], limit: int = 60) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    sampled = events.sort_values("signal_time").groupby("scale", observed=False).head(max(1, limit // max(len(specs), 1))).copy()
    rows: list[dict[str, object]] = []
    for row in sampled.itertuples(index=False):
        spec = specs[str(row.scale)]
        signal_start = pd.Timestamp(row.signal_bar_start)
        expected_signal = signal_start + spec.bar_delta
        actual_signal = pd.Timestamp(row.signal_time)
        actual_entry = pd.Timestamp(row.entry_time)
        rows.append(
            {
                "scale": row.scale,
                "pattern": row.pattern,
                "signal_bar_start": signal_start,
                "expected_signal_time": expected_signal,
                "signal_time": actual_signal,
                "entry_time": actual_entry,
                "signal_available_time_ok": bool(actual_signal == expected_signal),
                "entry_not_before_signal_ok": bool(actual_entry >= actual_signal),
                "features_closed_bar_only_contract": True,
            }
        )
    return pd.DataFrame(rows)


def _write_methodology(path: Path, args: argparse.Namespace, specs: list[ScaleSpec]) -> None:
    lines = [
        "# R01 Methodology",
        "",
        "## Research question",
        "Study ETH 'cannot fall / cannot rise' as a causal process rather than a one-candle pattern.",
        "",
        "## Frozen semantic definitions",
        "- Strong pressure: pressure_z >= 1.5 and same-direction taker-flow persistence >= 0.60.",
        "- Pressure stall: strong pressure but normalized directional price response <= 0.25.",
        "- Pressure rejection: strong pressure but directional response < 0.",
        "- Impact decay: adjacent same-side pressure remains >= 80% as strong while normalized response falls to <= 50% and <= 0.50.",
        "- Floor/ceiling retest: distinct touch episode of a prior-only rolling extreme zone.",
        "- Spring/upthrust: price breaks a frozen prior extreme then has already reclaimed it before signal generation.",
        "",
        "## Timing",
        "- Every signal is generated only after its bar closes.",
        "- Higher scales are aggregated from closed local 1m trade bars.",
        "- Fixed-horizon outcomes enter at the next bar open.",
        "- No outcome is used to choose a threshold in R01.",
        "",
        f"## Economic diagnostic\n- Round-trip cost deducted from fixed-horizon net return: {float(args.round_trip_cost):.4%}.",
        "",
        "## Scales",
    ]
    for spec in specs:
        lines.append(
            f"- {spec.name}: process windows={','.join(spec.horizon_label(w) for w in spec.process_windows)}; "
            f"defended zone lookback={spec.floor_lookback_label}; outcomes={','.join(spec.horizon_label(h) for h in spec.outcome_horizons)}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    requested = set(args.scales)
    specs = [spec for spec in SCALE_SPECS if spec.name in requested]
    unknown = sorted(requested.difference({spec.name for spec in SCALE_SPECS}))
    if unknown:
        raise ValueError(f"unknown scales: {unknown}")
    if not specs:
        raise ValueError("no scales selected")

    data_dir = Path(args.data_dir) if args.data_dir else PROJECT_ROOT / "data"
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    research_start = pd.Timestamp(args.start_date)
    research_end = _research_end_exclusive(args.end_date)
    warmup_floor = pd.Timestamp(args.warmup_start_date)
    months = _month_ranges(research_start, research_end)
    spec_map = {spec.name: spec for spec in specs}

    print(f"[run] {TITLE}", flush=True)
    print(f"[window] warmup={warmup_floor} research={research_start} -> {research_end - pd.Timedelta(microseconds=1)}", flush=True)
    print(f"[cost] fixed-horizon round-trip={float(args.round_trip_cost):.4%}", flush=True)
    print(f"[scales] {', '.join(spec.name for spec in specs)}", flush=True)

    all_events: list[pd.DataFrame] = []
    skipped: list[dict[str, object]] = []
    progress = ProgressReporter(label="[months]", total=len(months), every=1)

    need_micro = any(spec.name == "5s" for spec in specs)
    high_specs = [spec for spec in specs if spec.name != "5s"]
    micro_available = True
    if need_micro:
        probe = OKXTradeBarLoader(symbol=args.symbol, timeframe="5s", data_dir=data_dir, db_name=args.trade_bar_db_name)
        micro_available = _db_has_table(data_dir / args.trade_bar_db_name, probe.table_name)
        if not micro_available:
            msg = f"local 5s table missing/empty: {probe.table_name}"
            if not args.skip_micro_if_missing:
                raise FileNotFoundError(msg)
            print(f"[skip] {msg}; continue with 1m+ scales", flush=True)
            skipped.append({"scale": "5s", "reason": msg})

    for month_i, (month_left, month_right) in enumerate(months, start=1):
        print(f"[month] {month_left.date()} -> {(month_right - pd.Timedelta(seconds=1)).date()}", flush=True)

        if need_micro and micro_available:
            spec = spec_map["5s"]
            load_left = max(warmup_floor, month_left - pd.Timedelta(days=1))
            load_right = month_right + max(spec.outcome_horizons) * spec.bar_delta + spec.bar_delta
            micro = _load_trade_bars(
                symbol=args.symbol,
                timeframe="5s",
                start=load_left,
                end_exclusive=load_right,
                data_dir=data_dir,
                db_name=args.trade_bar_db_name,
            )
            if micro.empty:
                skipped.append({"scale": "5s", "month": str(month_left.date()), "reason": "no local rows"})
            else:
                micro = regularize_trade_bar_axis(micro, bar_delta=spec.bar_delta)
                ev = _run_scale_on_bars(
                    micro,
                    spec=spec,
                    month_left=month_left,
                    month_right=month_right,
                    round_trip_cost=float(args.round_trip_cost),
                )
                if not ev.empty:
                    all_events.append(ev)
                del micro, ev
                gc.collect()

        if high_specs:
            # One 1m query feeds 1m/5m/15m/1H/4H. 120d covers the longest
            # prior-only baseline (4H 90d) and the largest 14d defended zone.
            max_forward = max(max(spec.outcome_horizons) * spec.bar_delta for spec in high_specs)
            load_left = max(warmup_floor, month_left - pd.Timedelta(days=120))
            load_right = month_right + max_forward + pd.Timedelta(hours=4)
            source_1m = _load_trade_bars(
                symbol=args.symbol,
                timeframe="1m",
                start=load_left,
                end_exclusive=load_right,
                data_dir=data_dir,
                db_name=args.trade_bar_db_name,
            )
            if source_1m.empty:
                raise RuntimeError(f"mandatory local 1m trade bars missing for month={month_left.date()}")
            source_1m = regularize_trade_bar_axis(source_1m, bar_delta=pd.Timedelta(minutes=1))
            for spec in high_specs:
                bars = _prepare_scale_bars(source_1m, spec)
                ev = _run_scale_on_bars(
                    bars,
                    spec=spec,
                    month_left=month_left,
                    month_right=month_right,
                    round_trip_cost=float(args.round_trip_cost),
                )
                if not ev.empty:
                    all_events.append(ev)
                del bars, ev
                gc.collect()
            del source_1m
            gc.collect()

        progress.update(month_i)
    progress.close()

    if not all_events:
        raise RuntimeError("No events produced on requested local data")
    events = pd.concat(all_events, ignore_index=True, sort=False)
    events = events.sort_values(["scale", "signal_time", "pattern", "process_window"], kind="stable").reset_index(drop=True)

    summary_parts: list[pd.DataFrame] = []
    yearly_parts: list[pd.DataFrame] = []
    response_parts: list[pd.DataFrame] = []
    defense_parts: list[pd.DataFrame] = []
    stability_parts: list[pd.DataFrame] = []
    hold_parts: list[pd.DataFrame] = []
    for spec in specs:
        part = events.loc[events["scale"] == spec.name].copy()
        if part.empty:
            continue
        summary_parts.append(_summarize(part, spec, ["pattern", "trade_side", "process_window", "process_window_label"]))
        yearly_parts.append(_yearly_summary(part, spec))

        control = part.loc[part["pattern"].isin({"strong_pressure_control_fade", "strong_pressure_control_follow"})]
        if not control.empty:
            response_parts.append(_summarize(control, spec, ["pattern", "trade_side", "response_state", "process_window"]))

        floor = part.loc[part["pattern"].isin(FLOOR_PATTERNS)]
        if not floor.empty:
            defense_parts.append(_summarize(floor, spec, ["pattern", "trade_side", "defense_count_bucket"]))
            stability_parts.append(_summarize(floor, spec, ["pattern", "trade_side", "zone_stability_bucket"]))
            hold_parts.append(_summarize(floor, spec, ["pattern", "trade_side", "hold_ratio_bucket"]))

    summary = pd.concat(summary_parts, ignore_index=True) if summary_parts else pd.DataFrame()
    yearly = pd.concat(yearly_parts, ignore_index=True) if yearly_parts else pd.DataFrame()
    response = pd.concat(response_parts, ignore_index=True) if response_parts else pd.DataFrame()
    defense = pd.concat(defense_parts, ignore_index=True) if defense_parts else pd.DataFrame()
    stability = pd.concat(stability_parts, ignore_index=True) if stability_parts else pd.DataFrame()
    hold = pd.concat(hold_parts, ignore_index=True) if hold_parts else pd.DataFrame()
    samples = _select_event_sample(events, int(args.event_sample_per_group))
    audit = _causal_audit_samples(events, spec_map)

    _write_csv(summary, out_dir / "01_pattern_horizon_summary.csv")
    _write_csv(yearly, out_dir / "02_yearly_stability.csv")
    _write_csv(response, out_dir / "03_pressure_response_state_comparison.csv")
    _write_csv(defense, out_dir / "04_floor_defense_count_comparison.csv")
    _write_csv(stability, out_dir / "05_zone_stability_comparison.csv")
    _write_csv(hold, out_dir / "06_hold_above_below_ratio_comparison.csv")
    _write_csv(samples, out_dir / "07_event_samples.csv")
    _write_csv(audit, out_dir / "08_causal_timing_audit.csv")
    _write_csv(pd.DataFrame(skipped), out_dir / "09_skipped_inputs.csv")

    manifest = {
        "script": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "research_start": str(research_start),
        "research_end_exclusive": str(research_end),
        "warmup_start": str(warmup_floor),
        "round_trip_cost": float(args.round_trip_cost),
        "threshold_policy": "predeclared semantic thresholds; no outcome-driven parameter selection in R01",
        "event_rows": int(len(events)),
        "scales_requested": [spec.name for spec in specs],
        "skipped": skipped,
        "causal_contract": {
            "closed_bar_features_only": True,
            "signal_available_at_bar_close": True,
            "next_bar_open_outcomes": True,
            "higher_timeframe_available_time_respected": True,
        },
        "scale_specs": [
            {
                "name": spec.name,
                "process_windows": list(spec.process_windows),
                "baseline_bars": spec.baseline_bars,
                "floor_lookback": spec.floor_lookback,
                "floor_lookback_label": spec.floor_lookback_label,
                "defense_lookback": spec.defense_lookback,
                "reclaim_bars": spec.reclaim_bars,
                "outcome_horizons": list(spec.outcome_horizons),
            }
            for spec in specs
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_methodology(out_dir / "10_methodology.md", args, specs)

    print(f"[events] total={len(events):,}", flush=True)
    if not summary.empty:
        display_cols = ["scale", "pattern", "trade_side", "process_window_label", "horizon", "gross_events", "gross_mean", "net_mean", "net_pf"]
        print(summary[display_cols].head(30).to_string(index=False), flush=True)
    print(f"[report] {out_dir}", flush=True)
    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
