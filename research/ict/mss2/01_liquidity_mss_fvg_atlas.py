#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R01 - ICT MSS2 causal liquidity taxonomy + 1m/2m execution atlas.

Research question
-----------------
Do ETH perpetuals show a measurable, tradable reversal edge when an *eligible
and still-unconsumed* higher-timeframe liquidity pool is swept, followed by a
causal lower-timeframe MSS with displacement and an FVG pullback?

This is intentionally broader than a single strategy backtest.  It compares:
- 15m / 30m / 1H / 4H liquidity sources;
- minor vs structural vs major/external vs equal-price/multi-TF pools;
- recent order-1 MSS reference vs already-confirmed structural reference;
- 1m vs 2m execution;
- recent vs old/remote liquidity;
- weekday vs weekend, UTC/NY weekday, Asia/London/New-York-open clocks;
- displacement, path efficiency, MSS delay and FVG width bins;
- FVG limit-fill rate and structural-stop/fixed-R/opposing-liquidity outcomes;
- baseline 0.11% round-trip cost plus 2x and 3x cost stress in summaries.

Causality
---------
Bars are left-labelled and only available after close.  HTF pivot order is
recorded with its first causal available timestamp.  The first actual sweep is
found on 1m data only after the level has become available.  2m execution is
armed only after the 2m bar containing the 1m sweep closes.  MSS requires a
close through a reference swing that itself existed before the sweep.  FVG
orders can first fill on a bar strictly after the MSS close.  Future eventual
pivot order and forward outcome labels are physically separated from features.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from config.loader import TIMEZONE  # type: ignore[attr-defined]  # noqa: E402
except ImportError:
    TIMEZONE = "+8"
from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.ict_mss2 import (  # noqa: E402
    MSS2Config,
    attach_execution_outcomes,
    attach_sweep_baseline_outcomes,
    build_first_sweep_lifecycle,
    build_liquidity_levels,
    build_mss_fvg_events,
    causal_audit,
    classify_liquidity,
    normalize_1m_bars,
    split_features_and_labels,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_LIQUIDITY_MSS_FVG_ATLAS_R01"
EDGE_ID = "RESEARCH_ONLY_ICT_MSS2_HTF_LIQUIDITY_SWEEP_MSS_FVG"
TITLE = "ETH ICT MSS2 Liquidity Taxonomy + 1m/2m MSS/FVG Atlas R01"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r01_liquidity_mss_fvg_atlas"
BASELINE_ROUNDTRIP_COST = 0.0011


def _comma_ints(text: str) -> tuple[int, ...]:
    values = tuple(sorted(set(int(v.strip()) for v in str(text).split(",") if v.strip())))
    if not values:
        raise argparse.ArgumentTypeError("expected comma-separated integers")
    return values


def _parse_tf_spec(text: str) -> tuple[tuple[str, int], ...]:
    mapping = {"15m": 15, "30m": 30, "1h": 60, "1H": 60, "4h": 240, "4H": 240}
    out: list[tuple[str, int]] = []
    for token in [v.strip() for v in str(text).split(",") if v.strip()]:
        if token not in mapping:
            raise argparse.ArgumentTypeError(f"unsupported liquidity timeframe: {token}")
        name = "1H" if mapping[token] == 60 else "4H" if mapping[token] == 240 else token
        pair = (name, mapping[token])
        if pair not in out:
            out.append(pair)
    if not out:
        raise argparse.ArgumentTypeError("no liquidity timeframes")
    return tuple(out)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal ICT MSS2 liquidity taxonomy and 1m/2m execution atlas",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-08-15 23:59:59")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--liquidity-timeframes", type=_parse_tf_spec, default=(("15m", 15), ("30m", 30), ("1H", 60), ("4H", 240)))
    parser.add_argument("--liquidity-orders", type=_comma_ints, default=(1, 2, 3, 5))
    parser.add_argument("--execution-orders", type=_comma_ints, default=(1, 2, 3))
    parser.add_argument("--execution-minutes", type=_comma_ints, default=(1, 2))
    parser.add_argument("--reference-modes", default="recent,structural")
    parser.add_argument("--confluence-tolerance-bps", type=float, default=10.0)
    parser.add_argument("--touch-tolerance-bps", type=float, default=5.0)
    parser.add_argument("--approach-tolerance-bps", type=float, default=25.0)
    parser.add_argument("--sweep-epsilon-bps", type=float, default=0.01)
    parser.add_argument("--mss-break-epsilon-bps", type=float, default=0.01)
    parser.add_argument("--max-mss-minutes", type=int, default=60)
    parser.add_argument("--max-entry-wait-minutes", type=int, default=60)
    parser.add_argument("--max-outcome-minutes", type=int, default=180)
    parser.add_argument("--stop-buffer-bps", type=float, default=2.0)
    parser.add_argument("--roundtrip-cost", type=float, default=BASELINE_ROUNDTRIP_COST)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--event-sample-size", type=int, default=25_000)
    return parser.parse_args(argv)


def _load_ohlcv(args: argparse.Namespace) -> pd.DataFrame:
    print(f"[load] official OKX naked 1m OHLCV | {args.symbol} | {args.warmup_start_date} -> {args.end_date}", flush=True)
    loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
    bars = loader.fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    keep = [name for name in ("open", "high", "low", "close", "volume") if name in bars.columns]
    bars = normalize_1m_bars(bars.loc[:, keep])
    print(f"[load] rows={len(bars):,} range={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _window(frame: pd.DataFrame, time_col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ts = pd.to_datetime(frame[time_col], errors="coerce")
    return frame.loc[(ts >= start) & (ts <= end)].reset_index(drop=True)


def _pf(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    wins = float(values.loc[values > 0].sum())
    losses = float(-values.loc[values < 0].sum())
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return wins / losses


def _metric_row(frame: pd.DataFrame, *, cost: float) -> dict[str, float | int]:
    row: dict[str, float | int] = {"events": int(len(frame))}
    if frame.empty:
        return row
    row["fvg_rate"] = float(pd.to_numeric(frame.get("has_displacement_fvg", 0), errors="coerce").fillna(0).mean())
    filled = frame.loc[pd.to_numeric(frame.get("filled_flag", 0), errors="coerce").fillna(0).eq(1)]
    row["filled"] = int(len(filled))
    row["fill_rate_of_mss"] = float(len(filled) / len(frame)) if len(frame) else np.nan
    valid = filled.loc[pd.to_numeric(filled.get("valid_risk_flag", 0), errors="coerce").fillna(0).eq(1)]
    row["valid_risk"] = int(len(valid))
    if valid.empty:
        return row
    row["mean_mfe_r"] = float(pd.to_numeric(valid.get("mfe_r_180m"), errors="coerce").mean())
    row["mean_mae_r"] = float(pd.to_numeric(valid.get("mae_r_180m"), errors="coerce").mean())
    row["median_risk_bps"] = float(pd.to_numeric(valid.get("risk_bps"), errors="coerce").median())
    for token in ("r1p0", "r2p0", "r3p0"):
        outcome_col = f"{token}_outcome"
        gross_col = f"{token}_gross_return"
        if outcome_col in valid.columns:
            outcome = valid[outcome_col].astype(str)
            row[f"{token}_target_rate"] = float(outcome.eq("target").mean())
            row[f"{token}_stop_rate"] = float(outcome.eq("stop").mean())
        if gross_col in valid.columns:
            gross = pd.to_numeric(valid[gross_col], errors="coerce")
            for mult in (1.0, 2.0, 3.0):
                net = gross - float(cost) * mult
                suffix = "base" if mult == 1.0 else f"cost{int(mult)}x"
                row[f"{token}_mean_net_{suffix}"] = float(net.mean())
                row[f"{token}_win_rate_{suffix}"] = float((net > 0).mean())
                row[f"{token}_pf_{suffix}"] = _pf(net)
    for token in ("liq15", "liqany"):
        outcome_col = f"{token}_outcome"
        gross_col = f"{token}_gross_return"
        if outcome_col in valid.columns:
            subset = valid.loc[~valid[outcome_col].astype(str).isin(["no_target", "invalid_target"])]
            row[f"{token}_target_available"] = int(len(subset))
            if len(subset):
                row[f"{token}_target_rate"] = float(subset[outcome_col].astype(str).eq("target").mean())
                gross = pd.to_numeric(subset[gross_col], errors="coerce")
                for mult in (1.0, 2.0, 3.0):
                    net = gross - float(cost) * mult
                    suffix = "base" if mult == 1.0 else f"cost{int(mult)}x"
                    row[f"{token}_mean_net_{suffix}"] = float(net.mean())
                    row[f"{token}_pf_{suffix}"] = _pf(net)
    return row


def _sweep_metric_row(frame: pd.DataFrame) -> dict[str, float | int]:
    row: dict[str, float | int] = {"sweeps": int(len(frame))}
    if frame.empty:
        return row
    for horizon in (5, 15, 30, 60, 120, 180):
        ret_col = f"sweep_close_return_{horizon}m"
        mfe_col = f"sweep_mfe_{horizon}m"
        mae_col = f"sweep_mae_{horizon}m"
        if ret_col in frame.columns:
            values = pd.to_numeric(frame[ret_col], errors="coerce")
            row[f"mean_close_return_{horizon}m"] = float(values.mean())
            row[f"positive_close_rate_{horizon}m"] = float((values > 0).mean())
        if mfe_col in frame.columns:
            row[f"mean_mfe_{horizon}m"] = float(pd.to_numeric(frame[mfe_col], errors="coerce").mean())
        if mae_col in frame.columns:
            row[f"mean_mae_{horizon}m"] = float(pd.to_numeric(frame[mae_col], errors="coerce").mean())
    return row


def _sweep_group_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    available = [c for c in group_cols if c in frame.columns]
    if not available:
        return pd.DataFrame([_sweep_metric_row(frame)])
    rows: list[dict[str, object]] = []
    for key, part in frame.groupby(available, dropna=False, observed=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(available, key)}
        row.update(_sweep_metric_row(part))
        rows.append(row)
    return pd.DataFrame(rows)


def _group_summary(frame: pd.DataFrame, group_cols: list[str], *, cost: float) -> pd.DataFrame:
    available = [c for c in group_cols if c in frame.columns]
    if not available:
        return pd.DataFrame([_metric_row(frame, cost=cost)])
    rows: list[dict[str, object]] = []
    for key, part in frame.groupby(available, dropna=False, observed=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(available, key)}
        row.update(_metric_row(part, cost=cost))
        rows.append(row)
    return pd.DataFrame(rows)


def _attach_fixed_bins(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "displacement_atr" in out.columns:
        out["displacement_atr_bin"] = pd.cut(
            pd.to_numeric(out["displacement_atr"], errors="coerce"),
            [-np.inf, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, np.inf],
            labels=["<0.5", "0.5-0.75", "0.75-1.0", "1.0-1.25", "1.25-1.5", "1.5-2.0", ">=2.0"],
            right=False,
        ).astype("string")
    if "path_efficiency" in out.columns:
        out["path_efficiency_bin"] = pd.cut(
            pd.to_numeric(out["path_efficiency"], errors="coerce"),
            [-np.inf, 0.30, 0.50, 0.65, 0.80, np.inf],
            labels=["<0.30", "0.30-0.50", "0.50-0.65", "0.65-0.80", ">=0.80"],
            right=False,
        ).astype("string")
    if "fvg_width_atr" in out.columns:
        out["fvg_width_atr_bin"] = pd.cut(
            pd.to_numeric(out["fvg_width_atr"], errors="coerce"),
            [-np.inf, 0.03, 0.05, 0.10, 0.20, 0.30, np.inf],
            labels=["<0.03", "0.03-0.05", "0.05-0.10", "0.10-0.20", "0.20-0.30", ">=0.30"],
            right=False,
        ).astype("string")
    if "minutes_to_mss" in out.columns:
        out["mss_delay_bin"] = pd.cut(
            pd.to_numeric(out["minutes_to_mss"], errors="coerce"),
            [-np.inf, 5, 10, 20, 30, 60, np.inf],
            labels=["<5m", "5-10m", "10-20m", "20-30m", "30-60m", ">=60m"],
            right=False,
        ).astype("string")
    return out


def _funnel(lifecycle: pd.DataFrame, event_sets: dict[tuple[int, str], pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    swept = _window(lifecycle.loc[pd.to_numeric(lifecycle["sweep_pos_1m"], errors="coerce").fillna(-1).ge(0)], "sweep_available_time_1m", start, end)
    rows = [{"execution_minutes": 0, "reference_mode": "liquidity", "stage": "first_sweeps", "count": len(swept)}]
    for (minutes, mode), events in event_sets.items():
        rows.extend(
            [
                {"execution_minutes": minutes, "reference_mode": mode, "stage": "mss", "count": len(events)},
                {"execution_minutes": minutes, "reference_mode": mode, "stage": "mss_with_fvg", "count": int(pd.to_numeric(events["has_displacement_fvg"], errors="coerce").fillna(0).sum()) if "has_displacement_fvg" in events.columns else 0},
                {"execution_minutes": minutes, "reference_mode": mode, "stage": "fvg_limit_filled", "count": int(pd.to_numeric(events["filled_flag"], errors="coerce").fillna(0).sum()) if "filled_flag" in events.columns else 0},
                {"execution_minutes": minutes, "reference_mode": mode, "stage": "valid_structural_risk", "count": int(pd.to_numeric(events["valid_risk_flag"], errors="coerce").fillna(0).sum()) if "valid_risk_flag" in events.columns else 0},
            ]
        )
    return pd.DataFrame(rows)


def _one_vs_two_minute_overlap(all_events: pd.DataFrame) -> pd.DataFrame:
    if all_events.empty or not {1, 2}.issubset(set(pd.to_numeric(all_events["execution_minutes"], errors="coerce").dropna().astype(int))):
        return pd.DataFrame()
    cols = ["level_id", "reference_mode", "has_displacement_fvg", "filled_flag", "valid_risk_flag", "r1p0_gross_return", "r2p0_gross_return", "displacement_atr", "path_efficiency", "minutes_to_mss"]
    left = all_events.loc[all_events["execution_minutes"].eq(1), [c for c in cols if c in all_events.columns]].copy()
    right = all_events.loc[all_events["execution_minutes"].eq(2), [c for c in cols if c in all_events.columns]].copy()
    left = left.add_suffix("_1m").rename(columns={"level_id_1m": "level_id", "reference_mode_1m": "reference_mode"})
    right = right.add_suffix("_2m").rename(columns={"level_id_2m": "level_id", "reference_mode_2m": "reference_mode"})
    merged = left.merge(right, on=["level_id", "reference_mode"], how="outer", indicator=True, validate="one_to_one")
    return merged


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_readme(out_dir: Path, summaries: dict[str, pd.DataFrame], audit: dict[str, int], config: MSS2Config) -> None:
    lines = [
        f"# {TITLE}",
        "",
        "## What this run tests",
        "- HTF swing candidates are not assumed to be equal liquidity.",
        "- Liquidity is stratified by causal order confirmation, externality, active same-price clustering, multi-timeframe confluence, source timeframe, and age.",
        "- Old/remote swings are retained until their first true sweep; there is no recent-N-bars expiry.",
        "- 1m and 2m execution are compared on the same underlying 1m first-sweep lifecycle.",
        "- MSS uses only a pre-sweep, already-known execution-TF pivot and requires a close break.",
        "- Displacement is measured over the whole sweep-to-MSS leg; FVG is measured inside that leg.",
        "- FVG limit orders activate only after MSS close. Same-bar target/stop ambiguity is stop-first.",
        "- Weekday/weekend and Asia/London/New-York clocks are diagnostics, not admission filters.",
        "",
        "## Causal audit",
    ]
    for key, value in audit.items():
        lines.append(f"- `{key}`: {value}")
    lines += ["", "## Config", "```json", json.dumps(asdict(config), ensure_ascii=False, indent=2), "```", ""]
    for name, frame in summaries.items():
        lines += [f"## {name}", f"Rows: {len(frame)}", ""]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    execution_minutes = tuple(v for v in args.execution_minutes if v in {1, 2})
    if not execution_minutes:
        raise ValueError("--execution-minutes must include 1 and/or 2")
    reference_modes = tuple(v.strip() for v in str(args.reference_modes).split(",") if v.strip())
    if any(v not in {"recent", "structural"} for v in reference_modes):
        raise ValueError("reference modes must be recent,structural")
    cfg = MSS2Config(
        liquidity_timeframes=tuple(args.liquidity_timeframes),
        liquidity_confirmation_orders=tuple(args.liquidity_orders),
        execution_confirmation_orders=tuple(args.execution_orders),
        confluence_tolerance_bps=float(args.confluence_tolerance_bps),
        touch_tolerance_bps=float(args.touch_tolerance_bps),
        approach_tolerance_bps=float(args.approach_tolerance_bps),
        sweep_epsilon_bps=float(args.sweep_epsilon_bps),
        mss_break_epsilon_bps=float(args.mss_break_epsilon_bps),
        max_mss_minutes=int(args.max_mss_minutes),
        max_entry_wait_minutes=int(args.max_entry_wait_minutes),
        max_outcome_minutes=int(args.max_outcome_minutes),
        stop_buffer_bps=float(args.stop_buffer_bps),
    ).validate()
    show_progress = not bool(args.no_progress)
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)

    bars = _load_ohlcv(args)
    print("[stage] causal HTF liquidity candidates 15m/30m/1H/4H", flush=True)
    levels = build_liquidity_levels(bars, cfg)
    print(f"[liquidity] candidates={len(levels):,}", flush=True)
    print("[stage] first true 1m sweep lifecycle", flush=True)
    lifecycle = build_first_sweep_lifecycle(bars, levels, cfg, show_progress=show_progress)
    lifecycle = classify_liquidity(lifecycle, cfg)
    swept_all = lifecycle.loc[pd.to_numeric(lifecycle["sweep_pos_1m"], errors="coerce").fillna(-1).ge(0)].copy()
    swept_research = _window(swept_all, "sweep_available_time_1m", start, end)
    print(f"[liquidity] first_sweeps research_window={len(swept_research):,}", flush=True)

    # Future-eventual order columns never enter event construction semantics; keep
    # a separate level file so reviewers can audit that distinction explicitly.
    future_cols = [c for c in lifecycle.columns if c.startswith("future_eventual_order_")]
    causal_level_cols = [c for c in lifecycle.columns if c not in future_cols]
    _write_csv(lifecycle.loc[:, causal_level_cols], out_dir / "01_liquidity_lifecycle_causal.csv")
    if future_cols:
        _write_csv(lifecycle.loc[:, ["level_id", *future_cols]], out_dir / "02_liquidity_future_order_audit_labels.csv")
    # The event engine receives only causal lifecycle columns.  Future eventual
    # order labels are physically removed before any MSS/FVG construction.
    causal_lifecycle = lifecycle.loc[:, causal_level_cols].copy()

    print("[stage] sweep-only forward baseline (control for MSS uplift)", flush=True)
    sweep_baseline = attach_sweep_baseline_outcomes(
        bars, causal_lifecycle, config=cfg, project_timezone=TIMEZONE, show_progress=show_progress
    )
    sweep_baseline = _window(sweep_baseline, "sweep_available_time_1m", start, end)
    sweep_label_prefixes = (
        "sweep_baseline_entry_",
        "sweep_close_return_",
        "sweep_mfe_",
        "sweep_mae_",
    )
    sweep_ids = [c for c in ("sweep_event_id", "level_id") if c in sweep_baseline.columns]
    sweep_label_cols = sweep_ids + [c for c in sweep_baseline.columns if c.startswith(sweep_label_prefixes)]
    sweep_feature_cols = [c for c in sweep_baseline.columns if c not in set(sweep_label_cols) or c in sweep_ids]
    _write_csv(sweep_baseline.loc[:, sweep_feature_cols], out_dir / "03_sweep_features_causal.csv")
    _write_csv(sweep_baseline.loc[:, sweep_label_cols], out_dir / "04_sweep_forward_labels.csv")

    event_sets: dict[tuple[int, str], pd.DataFrame] = {}
    combined: list[pd.DataFrame] = []
    for minutes in execution_minutes:
        for mode in reference_modes:
            print(f"[stage] execution={minutes}m reference={mode}: sweep -> MSS -> displacement/FVG -> limit outcome", flush=True)
            events = build_mss_fvg_events(
                bars,
                causal_lifecycle,
                execution_minutes=minutes,
                reference_mode=mode,
                config=cfg,
                project_timezone=TIMEZONE,
                show_progress=show_progress,
            )
            events = attach_execution_outcomes(
                bars,
                causal_lifecycle,
                events,
                execution_minutes=minutes,
                config=cfg,
                show_progress=show_progress,
            )
            events = _window(events, "mss_available_time", start, end)
            events = _attach_fixed_bins(events)
            event_sets[(minutes, mode)] = events
            combined.append(events)
            features, labels = split_features_and_labels(events)
            _write_csv(features, out_dir / f"10_features_{minutes}m_{mode}.csv")
            _write_csv(labels, out_dir / f"11_labels_{minutes}m_{mode}.csv")
            sample = events.head(max(0, int(args.event_sample_size)))
            _write_csv(sample, out_dir / f"12_review_sample_{minutes}m_{mode}.csv")
            print(f"[events] {minutes}m/{mode}: mss={len(events):,} fvg={int(events.get('has_displacement_fvg', pd.Series(dtype=int)).sum()) if len(events) else 0:,} filled={int(events.get('filled_flag', pd.Series(dtype=int)).sum()) if len(events) else 0:,}", flush=True)

    all_events = pd.concat(combined, ignore_index=True, sort=False) if combined else pd.DataFrame()
    audit = causal_audit(levels, all_events)
    if any(
        audit[key]
        for key in (
            "level_available_before_pivot_bar_end",
            "mss_available_before_sweep_exec_available",
            "entry_not_after_mss",
            "mss_reference_not_pre_sweep",
            "mss_reference_available_after_sweep_bar_start",
        )
    ):
        raise RuntimeError(f"causal audit failed: {audit}")

    summaries: dict[str, pd.DataFrame] = {}
    summaries["funnel"] = _funnel(lifecycle, event_sets, start, end)
    summaries["sweep_only_overall"] = _sweep_group_summary(sweep_baseline, ["trade_direction"])
    summaries["sweep_only_liquidity_class"] = _sweep_group_summary(sweep_baseline, ["liquidity_class", "trade_direction"])
    summaries["sweep_only_quality_tier"] = _sweep_group_summary(sweep_baseline, ["quality_tier", "trade_direction"])
    summaries["sweep_only_source_timeframe"] = _sweep_group_summary(sweep_baseline, ["source_timeframe", "trade_direction"])
    summaries["sweep_only_age_bucket"] = _sweep_group_summary(sweep_baseline, ["age_bucket", "trade_direction"])
    summaries["sweep_only_weekend"] = _sweep_group_summary(sweep_baseline, ["is_weekend_utc", "is_weekend_ny", "trade_direction"])
    summaries["sweep_only_session"] = _sweep_group_summary(sweep_baseline, ["session_primary", "trade_direction"])
    summaries["overall"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "trade_direction"], cost=args.roundtrip_cost)
    summaries["liquidity_class"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "liquidity_class", "trade_direction"], cost=args.roundtrip_cost)
    summaries["liquidity_structural_score"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "liquidity_structural_score", "trade_direction"], cost=args.roundtrip_cost)
    summaries["quality_tier"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "quality_tier", "trade_direction"], cost=args.roundtrip_cost)
    summaries["mss_has_fvg"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "has_displacement_fvg", "trade_direction"], cost=args.roundtrip_cost)
    summaries["source_timeframe"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "source_timeframe", "trade_direction"], cost=args.roundtrip_cost)
    summaries["age_bucket"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "age_bucket", "trade_direction"], cost=args.roundtrip_cost)
    summaries["prior_touch"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "pretested_before_sweep_flag", "trade_direction"], cost=args.roundtrip_cost)
    summaries["weekend"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "is_weekend_utc", "is_weekend_ny", "trade_direction"], cost=args.roundtrip_cost)
    summaries["weekday"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "ny_weekday", "trade_direction"], cost=args.roundtrip_cost)
    summaries["session"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "session_primary", "trade_direction"], cost=args.roundtrip_cost)
    summaries["ny_cash_open"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "ny_cash_open_30m", "trade_direction"], cost=args.roundtrip_cost)
    summaries["ny_open_90m"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "ny_open_90m", "trade_direction"], cost=args.roundtrip_cost)
    summaries["london_am_flag"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "london_am_0700_1100", "trade_direction"], cost=args.roundtrip_cost)
    summaries["asia_flag"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "asia_0800_1600_shanghai", "trade_direction"], cost=args.roundtrip_cost)
    summaries["ny_hour"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "ny_hour", "trade_direction"], cost=args.roundtrip_cost)
    summaries["london_hour"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "london_hour", "trade_direction"], cost=args.roundtrip_cost)
    summaries["shanghai_hour"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "shanghai_hour", "trade_direction"], cost=args.roundtrip_cost)
    dated_events = all_events.copy()
    if len(dated_events):
        mss_ts = pd.to_datetime(dated_events["mss_available_time"], errors="coerce")
        dated_events["year"] = mss_ts.dt.year
        dated_events["quarter"] = mss_ts.dt.to_period("Q").astype("string")
        dated_events["month"] = mss_ts.dt.month
    summaries["year"] = _group_summary(dated_events, ["execution_minutes", "reference_mode", "year", "trade_direction"], cost=args.roundtrip_cost)
    summaries["quarter"] = _group_summary(dated_events, ["execution_minutes", "reference_mode", "quarter", "trade_direction"], cost=args.roundtrip_cost)
    summaries["month"] = _group_summary(dated_events, ["execution_minutes", "reference_mode", "month", "trade_direction"], cost=args.roundtrip_cost)
    summaries["displacement_bin"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "displacement_atr_bin", "trade_direction"], cost=args.roundtrip_cost)
    summaries["efficiency_bin"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "path_efficiency_bin", "trade_direction"], cost=args.roundtrip_cost)
    summaries["fvg_width_bin"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "fvg_width_atr_bin", "trade_direction"], cost=args.roundtrip_cost)
    summaries["mss_delay_bin"] = _group_summary(all_events, ["execution_minutes", "reference_mode", "mss_delay_bin", "trade_direction"], cost=args.roundtrip_cost)
    summaries["one_vs_two_minute_overlap"] = _one_vs_two_minute_overlap(all_events)
    summaries["liquidity_inventory_by_class"] = swept_research.groupby(["source_timeframe", "liquidity_class", "quality_tier", "pivot_side"], dropna=False, observed=False).size().rename("sweeps").reset_index()

    for name, frame in summaries.items():
        _write_csv(frame, out_dir / f"20_summary_{name}.csv")
    meta = {
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "script_version": SCRIPT_VERSION,
        "symbol": args.symbol,
        "warmup_start_date": args.warmup_start_date,
        "research_start_date": args.start_date,
        "research_end_date": args.end_date,
        "roundtrip_cost": float(args.roundtrip_cost),
        "project_timezone": str(TIMEZONE),
        "config": asdict(cfg),
        "causal_audit": audit,
        "important_semantics": [
            "future eventual swing order is audit-only and never used to construct events",
            "liquidity levels do not expire by age; first true sweep consumes them",
            "2m signals use the same 1m first-sweep lifecycle and cannot react before the containing 2m bar closes; resting-order fills/stops/targets are evaluated on original 1m OHLC",
            "MSS reference pivot must predate the sweep and already be causally confirmed",
            "MSS is close-confirmed, not wick-only",
            "FVG order starts strictly after MSS close",
            "same-bar stop/target is stop-first",
            "session/weekend fields are post-event stratification, not filters",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_readme(out_dir, summaries, audit, cfg)
    print(f"[audit] {audit}", flush=True)
    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
