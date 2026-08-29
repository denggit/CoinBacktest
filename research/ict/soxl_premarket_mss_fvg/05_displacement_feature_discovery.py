#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SOXL ICT R05: liquidity sweep -> structural MSS -> displacement leg -> FVG.

This revision corrects R02's concept error: the MSS break candle is NOT required
itself to be a large displacement candle or the third candle of an FVG.  MSS,
displacement and FVG are modeled as related but distinct parts of the reversal
process.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.alpaca_stock_loader import AlpacaStockLoader  # noqa: E402
from src.data_feed.okx_loader import OKXDataLoader, TIMEZONE as OKX_LOADER_TIMEZONE  # noqa: E402
from src.research_common.ict.premarket_mss_fvg import (  # noqa: E402
    BASE_SCENARIO,
    NY_TZ,
    ReplayScenario,
    ResearchConfig,
    add_analysis_dimensions,
    build_data_quality_table,
    compound_account,
    eligible_ny_dates,
    enforce_single_lifecycle,
    make_synthetic_ict_day,
    ny_date_bounds_to_source_naive,
    replay_attempts,
    report_trade_history,
    source_naive_to_new_york,
    summarize_variant,
)
from src.research_common.ict.premarket_mss_fvg_v2 import (  # noqa: E402
    SweepEpisodeConfig,
    build_all_premarket_levels_v2,
    build_sweep_events_v2,
    filter_liquidity_mode_v2,
    summarize_by_groups,
)
from src.research_common.ict.premarket_mss_fvg_v4 import (  # noqa: E402
    ICTDisplacementDiscoveryConfig,
    build_causal_audit_v4,
    build_signal_attempts_v4,
)
from src.research_common.ict.spot_perp_overlap import (  # noqa: E402
    build_equity_proxy_data_quality_table,
    densify_equity_minutes_causally,
)
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

DEFAULT_SYMBOL = "SOXL-USDT-SWAP"
DEFAULT_START_DATE = "2026-05-20"
DEFAULT_END_DATE = "2026-06-30"
DEFAULT_OUT_DIR = "data/reports/research/ict/soxl/mss/r05_displacement_discovery"
LIQUIDITY_MODES = ("extremes_only", "extremes_plus_strong_15m_swing")


def _csv_numbers(text: str, *, cast=float) -> list[Any]:
    out = [cast(x.strip()) for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError(f"empty numeric list: {text!r}")
    return out


def _csv_names(text: str) -> list[str]:
    out = [x.strip() for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("empty name list")
    return out


def _source_offset_hours(text: str) -> int:
    value = str(text).strip().upper().replace("UTC", "")
    try:
        return int(value)
    except ValueError:
        return 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SOXL ICT R05 path-based MSS/displacement/FVG causal research",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-source", choices=("okx", "alpaca"), default="okx")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--alpaca-symbol", default="SOXL")
    p.add_argument("--alpaca-feed", choices=("sip", "iex", "boats"), default="sip")
    p.add_argument("--alpaca-adjustment", default="split")
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--include-us-equity-holidays", action="store_true")
    p.add_argument("--required-day-coverage", type=float, default=0.995)
    p.add_argument("--execution-timeframes", default="1,2,5")
    p.add_argument("--liquidity-modes", default=",".join(LIQUIDITY_MODES))
    p.add_argument("--premarket-pivot-left", type=int, default=2)
    p.add_argument("--premarket-pivot-right", type=int, default=2)
    p.add_argument("--mss-pivot-left", type=int, default=1)
    p.add_argument("--mss-pivot-right", type=int, default=1)
    p.add_argument("--round-trip-cost", type=float, default=0.0011)
    p.add_argument("--cost-multipliers", default="1,2,3")
    p.add_argument("--order-delay-minutes", default="0,1,2")
    p.add_argument("--risk-fraction", type=float, default=0.01)
    p.add_argument("--max-notional-multiple", type=float, default=2.0)
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-platform-reports", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    tfs = _csv_numbers(args.execution_timeframes, cast=int)
    if any(tf not in {1, 2, 5} for tf in tfs):
        raise ValueError("execution_timeframes must be a subset of 1,2,5")
    invalid = set(_csv_names(args.liquidity_modes)) - set(LIQUIDITY_MODES)
    if invalid:
        raise ValueError(f"invalid liquidity modes: {sorted(invalid)}")
    if not 0.90 <= float(args.required_day_coverage) <= 1.0:
        raise ValueError("required_day_coverage must be in [0.90,1.0]")


def _base_cfg(args: argparse.Namespace) -> ResearchConfig:
    return ResearchConfig(
        execution_timeframes=tuple(_csv_numbers(args.execution_timeframes, cast=int)),
        mss_pivot_left=int(args.mss_pivot_left),
        mss_pivot_right=int(args.mss_pivot_right),
        premarket_pivot_left=int(args.premarket_pivot_left),
        premarket_pivot_right=int(args.premarket_pivot_right),
        required_day_coverage=float(args.required_day_coverage),
        round_trip_cost=float(args.round_trip_cost),
        risk_fraction=float(args.risk_fraction),
        max_notional_multiple=float(args.max_notional_multiple),
    )


def _path_cfg(args: argparse.Namespace) -> ICTDisplacementDiscoveryConfig:
    return ICTDisplacementDiscoveryConfig(
        execution_timeframes=tuple(_csv_numbers(args.execution_timeframes, cast=int)),
        mss_pivot_left=int(args.mss_pivot_left),
        mss_pivot_right=int(args.mss_pivot_right),
    )


def _load_1m(args: argparse.Namespace) -> pd.DataFrame:
    if args.data_source == "alpaca":
        start_ny = pd.Timestamp(args.start_date).normalize().tz_localize(NY_TZ)
        end_ny = (pd.Timestamp(args.end_date).normalize() + pd.Timedelta(days=1)).tz_localize(NY_TZ)
        loader = AlpacaStockLoader(
            symbol=args.alpaca_symbol,
            timeframe="1Min",
            feed=args.alpaca_feed,
            adjustment=args.alpaca_adjustment,
            data_dir=args.data_dir,
        )
        print(f"[load] Alpaca {args.alpaca_symbol} 1Min {args.alpaca_feed}/{args.alpaca_adjustment} NY={args.start_date}->{args.end_date}", flush=True)
        raw = loader.fetch_data_by_date_range(
            start_ny.tz_convert("UTC"),
            end_ny.tz_convert("UTC") - pd.Timedelta(minutes=1),
            local_only=bool(args.local_only),
        )
        if raw.empty:
            raise RuntimeError("Alpaca loader returned no data")
        idx = pd.DatetimeIndex(raw.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        bars = raw.copy()
        bars.index = idx.tz_convert(NY_TZ)
        bars.index.name = "bar_start_ny"
    else:
        offset = _source_offset_hours(OKX_LOADER_TIMEZONE)
        start_src, end_src = ny_date_bounds_to_source_naive(args.start_date, args.end_date, source_offset_hours=offset)
        loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
        print(f"[load] OKX {args.symbol} 1m NY={args.start_date}->{args.end_date}", flush=True)
        if args.local_only:
            raw = loader.load_local_data()
            if not raw.empty:
                raw = raw.loc[(raw.index >= start_src) & (raw.index <= end_src)].copy()
        else:
            raw = loader.fetch_data_by_date_range(start_src, end_src)
        if raw.empty:
            raise RuntimeError("OKX loader returned no data")
        bars = source_naive_to_new_york(raw, source_offset_hours=offset)

    mins = bars.index.hour * 60 + bars.index.minute
    bars = bars.loc[(mins >= 240) & (mins < 990)].copy()  # NY 04:00-16:30
    if args.data_source == "alpaca":
        bars = densify_equity_minutes_causally(bars)
    print(f"[load] rows={len(bars):,} NY={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path.name} rows={len(df):,}", flush=True)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _design(args: argparse.Namespace, cfg: ResearchConfig, valid_days: int) -> str:
    return f"""# SOXL ICT R05 — displacement feature discovery

## Purpose
R05 does **not** decide in advance what a "strong displacement" must be.  It first builds the causal ICT path, replays every MSS+FVG candidate, and then studies which measurable displacement shapes are associated with better or worse realized outcomes.

## ICT process modeled
1. Freeze NY 04:00-08:30 external premarket high/low plus genuinely strong causal 15m internal swings.
2. Trade only 08:30-16:30. A fresh liquidity sweep opens an episode and tracks the current terminal extreme.
3. MSS is the first completed close through the latest causally confirmed opposing STH/STL. A small STH/STL formed *after* the terminal extreme is valid and preferred once confirmed.
4. MSS, displacement and FVG are distinct. No body multiplier, close-location rule, speed-ratio threshold, or "MSS bar must be FVG candle 3" rule is used.
5. A directional FVG may be known before/on MSS, or complete after MSS while the same terminal extreme remains intact. The order cannot activate until both MSS and FVG are actually known.
6. Stop = terminal sweep extreme; target = opposite fresh absolute premarket extreme.

## Displacement discovery
The following are diagnostics, **not entry gates**: relative speed vs inbound leg, absolute/percentage delivery speed, terminal->MSS duration, path efficiency, directional-body share, directional-bar share, max directional body/range vs prior-20-bar median, MSS overshoot, FVG size, FVG timing relative to MSS, and FVG entry depth within the MSS leg.

Quartile boundaries are learned only from the discovery period through 2024, then frozen and applied to 2025 forward and 2026 late holdout. This is diagnostic research, not parameter selection.

## Causality
All higher-timeframe bars use available_time. Orders activate only after both MSS and the selected FVG are closed/known. No post-trade feature is used for signal generation.

## Frozen costs
Round trip={cfg.round_trip_cost:.4%}; cost stress={args.cost_multipliers}x; delay stress={args.order_delay_minutes} minutes. Valid sessions={valid_days}.
"""

def _replay_summaries(bars: pd.DataFrame, attempts: pd.DataFrame, args: argparse.Namespace, cfg: ResearchConfig):
    modes = _csv_names(args.liquidity_modes)
    costs = _csv_numbers(args.cost_multipliers, cast=float)
    delays = _csv_numbers(args.order_delay_minutes, cast=int)
    lifecycle_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    total = len(cfg.execution_timeframes) * len(modes) * (len(costs) + max(0, len(delays) - 1))
    prog = ProgressReporter(label="[replay] variants/stress", total=max(1, total), every=1, enabled=not args.no_progress)
    done = 0
    for tf in cfg.execution_timeframes:
        for mode in modes:
            subset = attempts.loc[attempts["execution_tf_minutes"] == int(tf)].copy()
            subset = filter_liquidity_mode_v2(subset, mode)
            scenarios = []
            for c in costs:
                scenarios.append((ReplayScenario("base" if c == 1 else f"cost_{c:g}x", c, 0), "cost"))
            scenarios += [(ReplayScenario(f"delay_{d}m", 1.0, d), "delay") for d in delays if d != 0]
            for scenario, family in scenarios:
                replayed = replay_attempts(
                    bars, subset, scenario=scenario, round_trip_cost=cfg.round_trip_cost,
                    risk_fraction=cfg.risk_fraction, max_notional_multiple=cfg.max_notional_multiple,
                )
                kept, skipped = enforce_single_lifecycle(replayed)
                if not kept.empty:
                    kept["liquidity_mode"] = mode
                    kept["stress_family"] = family
                    kept["variant_key"] = f"tf={tf}m|liq={mode}|disp=discovery|scenario={scenario.name}"
                    lifecycle_parts.append(kept)
                summaries.append({
                    "execution_tf": f"{tf}m", "execution_tf_minutes": tf, "liquidity_mode": mode,
                    "displacement_model": "ungated_feature_discovery", "scenario": scenario.name,
                    "stress_family": family, "cost_multiple": scenario.cost_multiple,
                    "order_delay_minutes": scenario.order_delay_minutes,
                    **summarize_variant(kept, skipped_overlap=skipped, initial_capital=float(args.initial_capital)),
                })
                done += 1
                prog.update(done)
    prog.close()
    lifecycle = pd.concat(lifecycle_parts, ignore_index=True) if lifecycle_parts else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    base = summary.loc[(summary["scenario"] == "base") & (summary["stress_family"] == "cost")].copy() if not summary.empty else pd.DataFrame()
    return lifecycle, base, summary.loc[summary["stress_family"] == "cost"].copy() if not summary.empty else pd.DataFrame(), summary.loc[summary["stress_family"] == "delay"].copy() if not summary.empty else pd.DataFrame()



DISCOVERY_FEATURES = (
    "displacement_speed_ratio",
    "mss_outbound_speed_pct_per_min",
    "terminal_to_mss_minutes",
    "reversal_path_efficiency",
    "directional_body_share",
    "directional_bar_fraction",
    "max_directional_body_vs_pre20_median",
    "max_leg_range_vs_pre20_median",
    "mss_overshoot_pct",
    "fvg_size_pct",
    "fvg_entry_depth_vs_mss_leg",
    "mss_to_fvg_minutes",
)


def _analysis_period(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates, errors="coerce")
    return pd.Series(
        np.select(
            [d.dt.year <= 2024, d.dt.year == 2025, d.dt.year >= 2026],
            ["discovery_through_2024", "forward_2025", "late_holdout_2026"],
            default="unknown",
        ),
        index=dates.index,
        dtype="object",
    )


def _profit_factor(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return np.nan
    gains = float(x.loc[x > 0].sum())
    losses = float(-x.loc[x < 0].sum())
    if losses <= 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def _trade_group_summary(group: pd.DataFrame) -> dict[str, object]:
    x = pd.to_numeric(group["net_return"], errors="coerce").dropna()
    return {
        "trades": int(len(group)),
        "win_rate": float((x > 0).mean()) if len(x) else np.nan,
        "mean_net_return": float(x.mean()) if len(x) else np.nan,
        "median_net_return": float(x.median()) if len(x) else np.nan,
        "profit_factor": _profit_factor(x),
        "target_hit_rate": float((group["exit_reason"] == "opposite_premarket_extreme_target").mean()) if len(group) else np.nan,
        "stop_hit_rate": float(group["exit_reason"].astype(str).str.contains("stop").mean()) if len(group) else np.nan,
        "median_mfe_r": float(pd.to_numeric(group.get("mfe_r"), errors="coerce").median()) if "mfe_r" in group else np.nan,
        "median_mae_r": float(pd.to_numeric(group.get("mae_r"), errors="coerce").median()) if "mae_r" in group else np.nan,
        "median_planned_rr": float(pd.to_numeric(group.get("planned_rr"), errors="coerce").median()) if "planned_rr" in group else np.nan,
    }


def _build_displacement_discovery_tables(base_lifecycle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Outcome diagnostics using frozen discovery-period quartiles.

    Only the full liquidity universe is used so premarket-extreme trades are not
    duplicated by the nested liquidity-mode variants.  Quartile cut points are
    estimated per execution timeframe from filled discovery trades through 2024
    and then held fixed for 2025/2026.  These tables are descriptive research,
    never an in-sample optimization loop.
    """
    if base_lifecycle.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    work = base_lifecycle.loc[
        (base_lifecycle["liquidity_mode"] == "extremes_plus_strong_15m_swing")
        & base_lifecycle["filled"].fillna(False).astype(bool)
    ].copy()
    if work.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    work["analysis_period"] = _analysis_period(work["ny_date"])
    work["year"] = pd.to_datetime(work["ny_date"], errors="coerce").dt.year.astype("Int64")

    edge_rows: list[dict[str, object]] = []
    perf_rows: list[dict[str, object]] = []
    bucketed_parts: list[pd.DataFrame] = []

    for tf, tf_group in work.groupby("execution_tf", sort=True):
        discovery = tf_group.loc[tf_group["analysis_period"] == "discovery_through_2024"].copy()
        tf_bucketed = tf_group.copy()
        for feature in DISCOVERY_FEATURES:
            if feature not in tf_group.columns:
                continue
            vals = pd.to_numeric(discovery[feature], errors="coerce").dropna()
            if len(vals) < 24:
                continue
            q25, q50, q75 = [float(x) for x in vals.quantile([0.25, 0.50, 0.75]).to_numpy()]
            if not (np.isfinite(q25) and np.isfinite(q50) and np.isfinite(q75)):
                continue
            edge_rows.append({
                "execution_tf": tf, "feature": feature,
                "discovery_trades": int(len(vals)), "q25": q25, "q50": q50, "q75": q75,
            })
            x = pd.to_numeric(tf_bucketed[feature], errors="coerce")
            bucket_col = f"__bucket__{feature}"
            tf_bucketed[bucket_col] = np.select(
                [x <= q25, (x > q25) & (x <= q50), (x > q50) & (x <= q75), x > q75],
                ["Q1", "Q2", "Q3", "Q4"], default="NA",
            )
            for (period, bucket), g in tf_bucketed.loc[tf_bucketed[bucket_col] != "NA"].groupby(["analysis_period", bucket_col], sort=True):
                perf_rows.append({
                    "execution_tf": tf, "feature": feature, "analysis_period": period,
                    "bucket": bucket, **_trade_group_summary(g),
                    "feature_median": float(pd.to_numeric(g[feature], errors="coerce").median()),
                })
        bucketed_parts.append(tf_bucketed)

    edges = pd.DataFrame(edge_rows)
    performance = pd.DataFrame(perf_rows)
    bucketed = pd.concat(bucketed_parts, ignore_index=True) if bucketed_parts else pd.DataFrame()

    yearly_rows: list[dict[str, object]] = []
    for (tf, year), g in work.groupby(["execution_tf", "year"], dropna=False, sort=True):
        yearly_rows.append({"execution_tf": tf, "year": year, **_trade_group_summary(g)})
    yearly = pd.DataFrame(yearly_rows)

    fvg_rows: list[dict[str, object]] = []
    if "fvg_relation_to_mss" in work.columns:
        for (tf, period, relation), g in work.groupby(["execution_tf", "analysis_period", "fvg_relation_to_mss"], sort=True):
            fvg_rows.append({"execution_tf": tf, "analysis_period": period, "fvg_relation_to_mss": relation, **_trade_group_summary(g)})
    fvg_timing = pd.DataFrame(fvg_rows)

    ref_rows: list[dict[str, object]] = []
    if "mss_reference_source" in work.columns:
        for (tf, period, source), g in work.groupby(["execution_tf", "analysis_period", "mss_reference_source"], sort=True):
            ref_rows.append({"execution_tf": tf, "analysis_period": period, "mss_reference_source": source, **_trade_group_summary(g)})
    reference = pd.DataFrame(ref_rows)
    return edges, performance, yearly, fvg_timing, reference

def _emit_platform_reports(base_lifecycle: pd.DataFrame, bars: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> None:
    if base_lifecycle.empty or args.skip_platform_reports:
        return
    root = out_dir / "platform_full_reports"
    for (tf, mode), group in base_lifecycle.groupby(["execution_tf", "liquidity_mode"], sort=True):
        filled = group.loc[group["filled"].fillna(False).astype(bool)].copy()
        if filled.empty:
            continue
        account = compound_account(filled, initial_capital=float(args.initial_capital))
        history = report_trade_history(account)
        if not history:
            continue
        print_full_report(
            trade_history=history, df=bars, initial_capital=float(args.initial_capital),
            capital=float(account["capital"].iloc[-1]), strategy_name=f"SOXL_ICT_R05_{tf}_{mode}",
            total_days=max((bars.index.max() - bars.index.min()).total_seconds() / 86400.0, 1.0),
            ai_enabled=False, symbol=args.symbol, report_dir=root / f"{tf}_{mode}",
        )


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    _validate_args(args)
    cfg = _base_cfg(args)
    path_cfg = _path_cfg(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    days = eligible_ny_dates(bars, start_date=args.start_date, end_date=args.end_date,
                             exclude_equity_holidays=not args.include_us_equity_holidays)
    quality = build_equity_proxy_data_quality_table(bars, days) if args.data_source == "alpaca" else build_data_quality_table(bars, days, required_coverage=cfg.required_day_coverage)
    valid_text = set(quality.loc[quality["coverage_pass"], "ny_date"].astype(str)) if not quality.empty else set()
    valid_days = [pd.Timestamp(x).date() for x in sorted(valid_text)]
    if not valid_days:
        raise RuntimeError("No valid sessions after coverage gate")
    print(f"[coverage] eligible={len(days)} valid={len(valid_days)}", flush=True)

    stage = ProgressReporter(label="[research] build stages", total=5, every=1, enabled=not args.no_progress)
    levels = build_all_premarket_levels_v2(bars, valid_days, pivot_left=cfg.premarket_pivot_left,
                                            pivot_right=cfg.premarket_pivot_right, episode_config=SweepEpisodeConfig())
    stage.update(1)
    sweeps = build_sweep_events_v2(bars, levels)
    stage.update(2)
    attempts, funnel = build_signal_attempts_v4(bars, sweeps, config=path_cfg, progress_enabled=not args.no_progress)
    stage.update(3)
    lifecycle, base_summary, cost_stress, delay_stress = _replay_summaries(bars, attempts, args, cfg) if not attempts.empty else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    stage.update(4)
    base_lifecycle = add_analysis_dimensions(lifecycle.loc[(lifecycle["scenario"] == "base") & (lifecycle["stress_family"] == "cost")].copy()) if not lifecycle.empty else pd.DataFrame()
    audit = build_causal_audit_v4(attempts)
    displacement_edges, displacement_perf, yearly_perf, fvg_timing_perf, reference_perf = _build_displacement_discovery_tables(base_lifecycle)
    stage.update(5); stage.close()

    (out_dir / "00_research_design.md").write_text(_design(args, cfg, len(valid_days)), encoding="utf-8")
    _write_csv(quality, out_dir / "01_data_quality.csv")
    _write_csv(levels, out_dir / "02_premarket_liquidity_levels.csv")
    _write_csv(sweeps, out_dir / "03_sweep_events.csv")
    _write_csv(funnel, out_dir / "04_mss_displacement_funnel.csv")
    _write_csv(attempts, out_dir / "05_signal_attempts.csv")
    _write_csv(base_lifecycle, out_dir / "06_base_trade_lifecycle.csv")
    _write_csv(base_summary, out_dir / "07_base_variant_summary.csv")
    _write_csv(cost_stress, out_dir / "08_cost_stress.csv")
    _write_csv(delay_stress, out_dir / "09_order_delay_stress.csv")
    filled = base_lifecycle.loc[base_lifecycle["filled"].fillna(False).astype(bool)].copy() if not base_lifecycle.empty else pd.DataFrame()
    _write_csv(summarize_by_groups(filled, ["liquidity_mode", "execution_tf"]), out_dir / "10_execution_timeframe_compare.csv")
    _write_csv(summarize_by_groups(filled, ["liquidity_mode", "level_type"]), out_dir / "11_liquidity_type_compare.csv")
    _write_csv(summarize_by_groups(filled, ["liquidity_mode", "trade_side"]), out_dir / "12_long_short_compare.csv")
    _write_csv(summarize_by_groups(filled, ["liquidity_mode", "weekday"]), out_dir / "13_weekday_compare.csv")
    _write_csv(summarize_by_groups(filled, ["liquidity_mode", "sweep_time_bucket"]), out_dir / "14_sweep_time_compare.csv")
    _write_csv(audit, out_dir / "15_causal_audit.csv")
    _write_csv(displacement_edges, out_dir / "18_displacement_feature_quartile_edges.csv")
    _write_csv(displacement_perf, out_dir / "19_displacement_feature_performance.csv")
    _write_csv(yearly_perf, out_dir / "20_yearly_performance.csv")
    _write_csv(fvg_timing_perf, out_dir / "21_fvg_timing_vs_mss_performance.csv")
    _write_csv(reference_perf, out_dir / "22_mss_reference_source_performance.csv")

    findings = [
        "# R05 Findings", "",
        f"- Valid sessions: **{len(valid_days)}**.",
        f"- Fresh/eligible sweeps: **{int(sweeps.get('setup_eligible_at_sweep', pd.Series(dtype=bool)).fillna(False).sum()) if not sweeps.empty else 0}**.",
        f"- MSS+FVG attempts without displacement-strength gate: **{len(attempts)}**.",
        "- R05 does not impose a displacement-strength threshold before observing outcomes.",
        "- Displacement characteristics are retained as continuous research features; FVG may complete before/on/after MSS while the terminal extreme remains intact.",
    ]
    if not funnel.empty:
        by = funnel.groupby("execution_tf").agg(sweeps=("fresh_sweep","sum"), mss=("mss_found","sum"), fvg=("fvg_after_mss_or_in_reversal_found","sum"), attempts=("attempt_emitted","sum")).reset_index()
        findings += ["", "## Funnel"] + [f"- `{r.execution_tf}`: sweeps={int(r.sweeps)}, MSS={int(r.mss)}, MSS+FVG={int(r.fvg)}, attempts={int(r.attempts)}." for r in by.itertuples(index=False)]
    if not base_summary.empty:
        ranked = base_summary.sort_values(["profit_factor","mean_net_return"], ascending=False, na_position="last")
        findings += ["", "## Base variants"]
        for r in ranked.head(8).to_dict("records"):
            findings.append(f"- `{r['execution_tf']} / {r['liquidity_mode']}`: trades={int(r.get('filled_trades',0) or 0)}, win={float(r.get('win_rate',np.nan)):.1%}, PF={float(r.get('profit_factor',np.nan)):.3f}, mean_net={float(r.get('mean_net_return',np.nan)):.3%}.")
    if not displacement_perf.empty:
        findings += ["", "## Displacement discovery", "- Strength variables are descriptive only; no quartile is promoted to a rule in R05.", "- Quartile edges come only from data through 2024 and are reused unchanged for 2025 and 2026."]
    (out_dir / "16_findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")

    manifest = {
        "experiment_id": "SOXL_ICT_MSS_R05_DISPLACEMENT_DISCOVERY",
        "edge_id": "SOXL_ICT_PM_SWEEP_MSS_FVG",
        "data_source": args.data_source,
        "start_date": args.start_date, "end_date": args.end_date,
        "valid_sessions": len(valid_days),
        "session_ny": "04:00-16:30", "premarket_ny": "04:00-08:30", "trade_ny": "08:30-16:30",
        "mss_definition": "liquidity sweep then close through latest causally confirmed opposing short-term pivot; post-terminal STH/STL is valid",
        "displacement_definition": "ungated terminal-extreme to MSS path features; strength discovered from outcomes, not pre-imposed",
        "fvg_definition": "directional 3-candle imbalance in reversal/continuation path; can complete before/on/after MSS if terminal extreme remains intact",
        "entry": "latest FVG already known at MSS, otherwise first directional FVG completed after MSS while terminal extreme remains intact; third-candle near edge",
        "round_trip_cost": cfg.round_trip_cost,
        "causal_audit_passed": bool(not audit.empty and audit["passed"].fillna(False).all()),
    }
    _write_json(manifest, out_dir / "17_manifest.json")
    _emit_platform_reports(base_lifecycle, bars, args, out_dir)
    if not args.skip_review_pack:
        finalize_research_report(out_dir, experiment_id=manifest["experiment_id"], edge_id=manifest["edge_id"], title="SOXL ICT R05 Displacement Discovery", print_log=True)
    return {"report_dir": out_dir, "review_pack": out_dir / "gpt_review_pack.zip"}


def run_self_test(args: argparse.Namespace) -> int:
    bars = make_synthetic_ict_day()
    with tempfile.TemporaryDirectory(prefix="soxl_ict_r05_") as tmp:
        args.start_date = args.end_date = "2026-06-02"
        args.out_dir = tmp
        args.include_us_equity_holidays = True
        args.required_day_coverage = 1.0
        args.skip_platform_reports = True
        args.skip_review_pack = True
        args.no_progress = True
        result = run_research(bars, args)
        if not (result["report_dir"] / "15_causal_audit.csv").exists():
            raise AssertionError("missing causal audit")
    print("[self-test] PASS", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _validate_args(args)
    if args.self_test:
        return run_self_test(args)
    run_research(_load_1m(args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
