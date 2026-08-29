#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SOXL ICT R09: mechanism archetype validation + causal EQH/EQL pools for Sweep -> MSS -> FVG.

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
    summarize_by_groups,
)
from src.research_common.ict.premarket_mss_fvg_v4 import (  # noqa: E402
    ICTDisplacementDiscoveryConfig,
    build_causal_audit_v4,
    build_signal_attempts_v4,
)
from src.research_common.ict.htf_liquidity import (  # noqa: E402
    HTFLiquidityConfig,
    attach_first_consumption_time,
    build_htf_swing_catalog,
    build_remote_htf_levels_for_days,
    dedupe_same_family_sweeps,
)
from src.research_common.ict.spot_perp_overlap import (  # noqa: E402
    build_equity_proxy_data_quality_table,
    densify_equity_minutes_causally,
)
from src.research_common.ict.semantic_gap import (  # noqa: E402
    SemanticGapConfig,
    attach_causal_semantic_features,
    attach_outcome_path_labels,
    build_entry_failure_atlas,
    build_mfe_transition_table,
    build_semantic_category_atlas,
    build_semantic_causal_audit,
    build_semantic_feature_atlas,
)
from src.research_common.ict.liquidity_maturity import (  # noqa: E402
    LiquidityMaturityConfig,
    attach_liquidity_maturity_features,
    build_maturity_causal_audit,
    build_maturity_feature_atlas,
    build_maturity_pair_atlas,
    build_opportunity_frequency_atlas,
)
from src.research_common.ict.equal_liquidity import (  # noqa: E402
    EqualLiquidityConfig,
    attach_equal_pool_context,
    build_equal_liquidity_pools,
    build_equal_pool_causal_audit,
)
from src.research_common.ict.mechanism_archetypes import (  # noqa: E402
    MechanismArchetypeConfig,
    attach_mechanism_archetypes,
    build_equal_pool_performance_atlas,
    build_mechanism_archetype_atlas,
    build_mechanism_causal_audit,
    build_mechanism_combination_atlas,
    fit_mechanism_distribution_edges,
)
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

DEFAULT_SYMBOL = "SOXL-USDT-SWAP"
DEFAULT_START_DATE = "2023-07-01"
DEFAULT_END_DATE = "2026-06-30"
DEFAULT_OUT_DIR = "data/reports/research/ict/soxl/mss/r09_mechanism_archetype_validation"
LIQUIDITY_FAMILIES = ("premarket_extreme", "major_15m_swing", "remote_1h_swing", "remote_4h_swing", "remote_1d_swing", "equal_liquidity_pool")


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
        description="SOXL ICT R09 mechanism archetype validation for Sweep -> MSS -> FVG research",
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
    p.add_argument("--liquidity-families", default=",".join(LIQUIDITY_FAMILIES))
    p.add_argument("--htf-timeframes", default="1h,4h,1d")
    p.add_argument("--htf-pivot-left", type=int, default=2)
    p.add_argument("--htf-pivot-right", type=int, default=2)
    p.add_argument("--premarket-pivot-left", type=int, default=2)
    p.add_argument("--premarket-pivot-right", type=int, default=2)
    p.add_argument("--mss-pivot-left", type=int, default=1)
    p.add_argument("--mss-pivot-right", type=int, default=1)
    p.add_argument("--semantic-retest-tolerance-bp", type=float, default=10.0)
    p.add_argument("--semantic-min-discovery-samples", type=int, default=40)
    p.add_argument("--semantic-quantile-bins", type=int, default=5)
    p.add_argument("--maturity-presweep-lookback-minutes", type=int, default=60)
    p.add_argument("--maturity-near-touch-tolerance-bp", type=float, default=10.0)
    p.add_argument("--maturity-min-discovery-samples", type=int, default=40)
    p.add_argument("--maturity-quantile-bins", type=int, default=5)
    p.add_argument("--eq-source-timeframes", default="1,5,15")
    p.add_argument("--eq-pivot-left", type=int, default=1)
    p.add_argument("--eq-pivot-right", type=int, default=1)
    p.add_argument("--eq-min-members", type=int, default=2)
    p.add_argument("--eq-min-tolerance-bp", type=float, default=5.0)
    p.add_argument("--eq-range-tolerance-fraction", type=float, default=0.25)
    p.add_argument("--mechanism-min-discovery-samples", type=int, default=40)
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
    invalid = set(_csv_names(args.liquidity_families)) - set(LIQUIDITY_FAMILIES)
    if invalid:
        raise ValueError(f"invalid liquidity families: {sorted(invalid)}")
    htf = set(_csv_names(args.htf_timeframes))
    if not htf.issubset({"1h", "4h", "1d"}):
        raise ValueError(f"invalid htf timeframes: {sorted(htf)}")
    if int(args.htf_pivot_left) < 1 or int(args.htf_pivot_right) < 1:
        raise ValueError("htf pivot left/right must be >= 1")
    if not 0.90 <= float(args.required_day_coverage) <= 1.0:
        raise ValueError("required_day_coverage must be in [0.90,1.0]")
    if float(args.semantic_retest_tolerance_bp) <= 0:
        raise ValueError("semantic_retest_tolerance_bp must be > 0")
    if int(args.semantic_min_discovery_samples) < 20:
        raise ValueError("semantic_min_discovery_samples must be >= 20")
    if int(args.semantic_quantile_bins) not in {4, 5, 6, 8, 10}:
        raise ValueError("semantic_quantile_bins must be one of 4,5,6,8,10")
    if int(args.maturity_presweep_lookback_minutes) < 15:
        raise ValueError("maturity_presweep_lookback_minutes must be >= 15")
    if float(args.maturity_near_touch_tolerance_bp) <= 0:
        raise ValueError("maturity_near_touch_tolerance_bp must be > 0")
    if int(args.maturity_min_discovery_samples) < 20:
        raise ValueError("maturity_min_discovery_samples must be >= 20")
    if int(args.maturity_quantile_bins) not in {4, 5, 6, 8, 10}:
        raise ValueError("maturity_quantile_bins must be one of 4,5,6,8,10")
    eq_tfs = _csv_numbers(args.eq_source_timeframes, cast=int)
    if any(tf not in {1, 5, 15} for tf in eq_tfs):
        raise ValueError("eq_source_timeframes must be a subset of 1,5,15")
    if int(args.eq_pivot_left) < 1 or int(args.eq_pivot_right) < 1:
        raise ValueError("eq pivot left/right must be >= 1")
    if int(args.eq_min_members) < 2:
        raise ValueError("eq_min_members must be >= 2")
    if float(args.eq_min_tolerance_bp) <= 0 or float(args.eq_range_tolerance_fraction) <= 0:
        raise ValueError("equal-pool tolerance parameters must be > 0")
    if int(args.mechanism_min_discovery_samples) < 20:
        raise ValueError("mechanism_min_discovery_samples must be >= 20")


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


def _semantic_cfg(args: argparse.Namespace) -> SemanticGapConfig:
    return SemanticGapConfig(
        terminal_retest_tolerance_bp=float(args.semantic_retest_tolerance_bp),
        min_discovery_samples=int(args.semantic_min_discovery_samples),
        quantile_bins=int(args.semantic_quantile_bins),
    )


def _maturity_cfg(args: argparse.Namespace) -> LiquidityMaturityConfig:
    return LiquidityMaturityConfig(
        discovery_end_date="2024-12-31",
        forward_start_date="2025-01-01",
        forward_end_date="2025-12-31",
        holdout_start_date="2026-01-01",
        presweep_lookback_minutes=int(args.maturity_presweep_lookback_minutes),
        near_touch_tolerance_bp=float(args.maturity_near_touch_tolerance_bp),
        min_discovery_samples=int(args.maturity_min_discovery_samples),
        quantile_bins=int(args.maturity_quantile_bins),
    )


def _eq_cfg(args: argparse.Namespace) -> EqualLiquidityConfig:
    return EqualLiquidityConfig(
        source_timeframes=tuple(_csv_numbers(args.eq_source_timeframes, cast=int)),
        pivot_left=int(args.eq_pivot_left),
        pivot_right=int(args.eq_pivot_right),
        min_members=int(args.eq_min_members),
        min_tolerance_bp=float(args.eq_min_tolerance_bp),
        range_tolerance_fraction=float(args.eq_range_tolerance_fraction),
    )


def _mechanism_cfg(args: argparse.Namespace) -> MechanismArchetypeConfig:
    return MechanismArchetypeConfig(
        discovery_end_date="2024-12-31",
        forward_start_date="2025-01-01",
        forward_end_date="2025-12-31",
        holdout_start_date="2026-01-01",
        min_discovery_samples=int(args.mechanism_min_discovery_samples),
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
    return f"""# SOXL ICT R09 — Mechanism Archetype Validation + True EQH/EQL Pools

## Purpose
R09 does not promote R08's profitable quantile buckets into a strict strategy. It keeps the broad causal Sweep -> MSS -> FVG universe and validates overlapping path mechanisms: fast rejection, sustained consumption, deep/progressive flush, equal-high/equal-low stop runs, clean reversal delivery and moderate/extended MSS delivery.

## True EQH/EQL liquidity
R08's `near_touch_count` was only a coarse bar-proximity proxy. R09 constructs actual equal-high/equal-low pools from causally confirmed 1m/5m/15m same-side swing clusters. A pool is available only after all member pivots are confirmed and remains active at 08:30 only if price has not strictly traded through its outer boundary. All active pools are retained; no nearest-only rule is used.

## Archetype protocol
Archetype tags are overlapping diagnostics, not entry gates. Distribution landmarks (Q25/Q50/Q75) are fitted from 2023H2-2024 attempts only and frozen for 2025 forward and 2026 late holdout. PnL is never used to fit those landmarks. Existing premarket/15m/1H/4H/1D attempts are preserved; EQH/EQL pools are evaluated as an additional liquidity family and also as context around existing levels.

## Frozen execution
MSS/FVG/entry/terminal stop/opposite premarket target/cost/replay semantics remain unchanged from R08. No archetype, equal-pool context or maturity feature can filter a candidate in R09.

Round trip={cfg.round_trip_cost:.4%}; stress={args.cost_multipliers}x; delay={args.order_delay_minutes} minutes. Valid sessions={valid_days}.
"""

def _replay_summaries(bars: pd.DataFrame, attempts: pd.DataFrame, args: argparse.Namespace, cfg: ResearchConfig):
    families = _csv_names(args.liquidity_families)
    costs = _csv_numbers(args.cost_multipliers, cast=float)
    delays = _csv_numbers(args.order_delay_minutes, cast=int)
    lifecycle_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    total = len(cfg.execution_timeframes) * len(families) * (len(costs) + max(0, len(delays) - 1))
    prog = ProgressReporter(label="[replay] family variants/stress", total=max(1, total), every=1, enabled=not args.no_progress)
    done = 0
    for tf in cfg.execution_timeframes:
        for family in families:
            subset = attempts.loc[
                (attempts["execution_tf_minutes"] == int(tf))
                & (attempts["liquidity_family"].astype(str) == family)
            ].copy()
            scenarios = [
                (ReplayScenario("base" if c == 1 else f"cost_{c:g}x", c, 0), "cost")
                for c in costs
            ]
            scenarios += [(ReplayScenario(f"delay_{d}m", 1.0, d), "delay") for d in delays if d != 0]
            for scenario, stress_family in scenarios:
                replayed = replay_attempts(
                    bars, subset, scenario=scenario, round_trip_cost=cfg.round_trip_cost,
                    risk_fraction=cfg.risk_fraction, max_notional_multiple=cfg.max_notional_multiple,
                )
                kept, skipped = enforce_single_lifecycle(replayed)
                if not kept.empty:
                    kept["liquidity_family"] = family
                    kept["stress_family"] = stress_family
                    kept["variant_key"] = f"tf={tf}m|family={family}|disp=discovery|scenario={scenario.name}"
                    lifecycle_parts.append(kept)
                summaries.append({
                    "execution_tf": f"{tf}m", "execution_tf_minutes": tf,
                    "liquidity_family": family,
                    "displacement_model": "ungated_feature_discovery", "scenario": scenario.name,
                    "stress_family": stress_family, "cost_multiple": scenario.cost_multiple,
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
    """Frozen-quartile displacement diagnostics, kept separate by liquidity family."""
    if base_lifecycle.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    work = base_lifecycle.loc[base_lifecycle["filled"].fillna(False).astype(bool)].copy()
    if work.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    work["analysis_period"] = _analysis_period(work["ny_date"])
    work["year"] = pd.to_datetime(work["ny_date"], errors="coerce").dt.year.astype("Int64")

    edge_rows: list[dict[str, object]] = []
    perf_rows: list[dict[str, object]] = []
    for (tf, family), fam_group in work.groupby(["execution_tf", "liquidity_family"], sort=True):
        discovery = fam_group.loc[fam_group["analysis_period"] == "discovery_through_2024"].copy()
        for feature in DISCOVERY_FEATURES:
            if feature not in fam_group.columns:
                continue
            vals = pd.to_numeric(discovery[feature], errors="coerce").dropna()
            if len(vals) < 24:
                continue
            q25, q50, q75 = [float(x) for x in vals.quantile([0.25, 0.50, 0.75]).to_numpy()]
            if not (np.isfinite(q25) and np.isfinite(q50) and np.isfinite(q75)):
                continue
            edge_rows.append({
                "execution_tf": tf, "liquidity_family": family, "feature": feature,
                "discovery_trades": int(len(vals)), "q25": q25, "q50": q50, "q75": q75,
            })
            x = pd.to_numeric(fam_group[feature], errors="coerce")
            bucket = np.select(
                [x <= q25, (x > q25) & (x <= q50), (x > q50) & (x <= q75), x > q75],
                ["Q1", "Q2", "Q3", "Q4"], default="NA",
            )
            tmp = fam_group.assign(__bucket=bucket)
            for (period, b), g in tmp.loc[tmp["__bucket"] != "NA"].groupby(["analysis_period", "__bucket"], sort=True):
                perf_rows.append({
                    "execution_tf": tf, "liquidity_family": family, "feature": feature,
                    "analysis_period": period, "bucket": b, **_trade_group_summary(g),
                    "feature_median": float(pd.to_numeric(g[feature], errors="coerce").median()),
                })

    yearly_rows: list[dict[str, object]] = []
    for (tf, family, year), g in work.groupby(["execution_tf", "liquidity_family", "year"], dropna=False, sort=True):
        yearly_rows.append({"execution_tf": tf, "liquidity_family": family, "year": year, **_trade_group_summary(g)})

    fvg_rows: list[dict[str, object]] = []
    if "fvg_relation_to_mss" in work.columns:
        for (tf, family, period, relation), g in work.groupby(["execution_tf", "liquidity_family", "analysis_period", "fvg_relation_to_mss"], sort=True):
            fvg_rows.append({"execution_tf": tf, "liquidity_family": family, "analysis_period": period, "fvg_relation_to_mss": relation, **_trade_group_summary(g)})

    ref_rows: list[dict[str, object]] = []
    if "mss_reference_source" in work.columns:
        for (tf, family, period, source), g in work.groupby(["execution_tf", "liquidity_family", "analysis_period", "mss_reference_source"], sort=True):
            ref_rows.append({"execution_tf": tf, "liquidity_family": family, "analysis_period": period, "mss_reference_source": source, **_trade_group_summary(g)})

    return pd.DataFrame(edge_rows), pd.DataFrame(perf_rows), pd.DataFrame(yearly_rows), pd.DataFrame(fvg_rows), pd.DataFrame(ref_rows)

def _emit_platform_reports(base_lifecycle: pd.DataFrame, bars: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> None:
    if base_lifecycle.empty or args.skip_platform_reports:
        return
    root = out_dir / "platform_full_reports"
    for (tf, family), group in base_lifecycle.groupby(["execution_tf", "liquidity_family"], sort=True):
        filled = group.loc[group["filled"].fillna(False).astype(bool)].copy()
        if filled.empty:
            continue
        account = compound_account(filled, initial_capital=float(args.initial_capital))
        history = report_trade_history(account)
        if not history:
            continue
        print_full_report(
            trade_history=history, df=bars, initial_capital=float(args.initial_capital),
            capital=float(account["capital"].iloc[-1]), strategy_name=f"SOXL_ICT_R08_{tf}_{family}",
            total_days=max((bars.index.max() - bars.index.min()).total_seconds() / 86400.0, 1.0),
            ai_enabled=False, symbol=args.symbol, report_dir=root / f"{tf}_{family}",
        )


def _family_summary_tables(base_lifecycle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if base_lifecycle.empty:
        e = pd.DataFrame()
        return e, e, e, e
    filled = base_lifecycle.loc[base_lifecycle["filled"].fillna(False).astype(bool)].copy()
    if not filled.empty:
        filled["year"] = pd.to_datetime(filled["ny_date"], errors="coerce").dt.year.astype("Int64")
    if filled.empty:
        e = pd.DataFrame()
        return e, e, e, e
    family = summarize_by_groups(filled, ["liquidity_family", "execution_tf"])
    yearly = summarize_by_groups(filled, ["liquidity_family", "execution_tf", "year"])
    side = summarize_by_groups(filled, ["liquidity_family", "execution_tf", "trade_side"])
    conf = summarize_by_groups(filled, ["execution_tf", "htf_confluence_count"]) if "htf_confluence_count" in filled.columns else pd.DataFrame()
    return family, yearly, side, conf


def _htf_structure_diagnostics(base_lifecycle: pd.DataFrame) -> pd.DataFrame:
    if base_lifecycle.empty:
        return pd.DataFrame()
    work = base_lifecycle.loc[
        base_lifecycle["filled"].fillna(False).astype(bool)
        & base_lifecycle["liquidity_family"].astype(str).str.startswith("remote_")
    ].copy()
    if work.empty:
        return pd.DataFrame()
    age = pd.to_numeric(work.get("htf_age_calendar_days"), errors="coerce")
    dist = pd.to_numeric(work.get("htf_distance_from_premarket_close_pct"), errors="coerce")
    rank = pd.to_numeric(work.get("active_rank_nearest"), errors="coerce")
    work["htf_age_bucket"] = pd.cut(age, [-np.inf, 2, 5, 20, 60, np.inf], labels=["0-2d", "2-5d", "5-20d", "20-60d", "60d+"])
    work["htf_distance_bucket"] = pd.cut(dist, [-np.inf, .005, .01, .02, .05, np.inf], labels=["<=0.5%", "0.5-1%", "1-2%", "2-5%", ">5%"])
    work["htf_rank_bucket"] = np.select([rank <= 1, rank <= 3, rank <= 5], ["nearest", "rank2-3", "rank4-5"], default="rank6+")
    rows: list[dict[str, object]] = []
    for dim in ("htf_age_bucket", "htf_distance_bucket", "htf_rank_bucket"):
        for (family, tf, bucket), g in work.groupby(["liquidity_family", "execution_tf", dim], observed=True, dropna=False, sort=True):
            rows.append({"dimension": dim, "bucket": str(bucket), "liquidity_family": family, "execution_tf": tf, **_trade_group_summary(g)})
    return pd.DataFrame(rows)


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    _validate_args(args)
    cfg = _base_cfg(args)
    path_cfg = _path_cfg(args)
    sem_cfg = _semantic_cfg(args)
    maturity_cfg = _maturity_cfg(args)
    eq_cfg = _eq_cfg(args)
    mechanism_cfg = _mechanism_cfg(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    days = eligible_ny_dates(
        bars, start_date=args.start_date, end_date=args.end_date,
        exclude_equity_holidays=not args.include_us_equity_holidays,
    )
    quality = (
        build_equity_proxy_data_quality_table(bars, days)
        if args.data_source == "alpaca"
        else build_data_quality_table(bars, days, required_coverage=cfg.required_day_coverage)
    )
    valid_text = set(quality.loc[quality["coverage_pass"], "ny_date"].astype(str)) if not quality.empty else set()
    valid_days = [pd.Timestamp(x).date() for x in sorted(valid_text)]
    if not valid_days:
        raise RuntimeError("No valid sessions after coverage gate")
    print(f"[coverage] eligible={len(days)} valid={len(valid_days)}", flush=True)

    stage = ProgressReporter(label="[research] build stages", total=14, every=1, enabled=not args.no_progress)

    premarket_levels = build_all_premarket_levels_v2(
        bars, valid_days, pivot_left=cfg.premarket_pivot_left,
        pivot_right=cfg.premarket_pivot_right, episode_config=SweepEpisodeConfig(),
    )
    if not premarket_levels.empty:
        premarket_levels = premarket_levels.copy()
        premarket_levels["liquidity_family"] = premarket_levels["level_type"].astype(str)
    stage.update(1)

    equal_pools = build_equal_liquidity_pools(
        bars, valid_days, premarket_levels, config=eq_cfg,
    )
    stage.update(2)

    htf_cfg = HTFLiquidityConfig(
        timeframes=tuple(_csv_names(args.htf_timeframes)),
        pivot_left=int(args.htf_pivot_left), pivot_right=int(args.htf_pivot_right),
    )
    htf_catalog = build_htf_swing_catalog(bars, config=htf_cfg)
    htf_catalog = attach_first_consumption_time(bars, htf_catalog)
    htf_levels = build_remote_htf_levels_for_days(htf_catalog, premarket_levels, valid_days)
    stage.update(3)

    level_parts = [x for x in (premarket_levels, htf_levels, equal_pools) if not x.empty]
    levels = pd.concat(level_parts, ignore_index=True, sort=False) if level_parts else pd.DataFrame()
    allowed_families = set(_csv_names(args.liquidity_families))
    if not levels.empty:
        levels = levels.loc[levels["liquidity_family"].astype(str).isin(allowed_families)].copy()
    stage.update(4)

    sweeps_raw = build_sweep_events_v2(bars, levels)
    sweeps = dedupe_same_family_sweeps(sweeps_raw)
    stage.update(5)

    attempts, funnel = build_signal_attempts_v4(
        bars, sweeps, config=path_cfg, progress_enabled=not args.no_progress,
    )
    stage.update(6)

    # Descriptive features only: no semantic, maturity, equal-pool context or
    # archetype tag is allowed to filter the broad causal candidate universe.
    attempts = attach_causal_semantic_features(attempts, bars, config=sem_cfg) if not attempts.empty else attempts
    attempts = attach_liquidity_maturity_features(attempts, bars, config=maturity_cfg) if not attempts.empty else attempts
    attempts = attach_equal_pool_context(attempts, equal_pools) if not attempts.empty else attempts
    stage.update(7)

    mechanism_edges = fit_mechanism_distribution_edges(attempts, config=mechanism_cfg) if not attempts.empty else pd.DataFrame()
    attempts_before_tags = attempts["attempt_id"].astype(str).tolist() if not attempts.empty else []
    attempts = attach_mechanism_archetypes(attempts, mechanism_edges) if not attempts.empty else attempts
    if attempts_before_tags and attempts["attempt_id"].astype(str).tolist() != attempts_before_tags:
        raise AssertionError("R09 mechanism tags changed the candidate attempt universe")
    stage.update(8)

    lifecycle, base_summary, cost_stress, delay_stress = (
        _replay_summaries(bars, attempts, args, cfg)
        if not attempts.empty
        else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    )
    stage.update(9)

    base_lifecycle = (
        add_analysis_dimensions(lifecycle.loc[(lifecycle["scenario"] == "base") & (lifecycle["stress_family"] == "cost")].copy())
        if not lifecycle.empty else pd.DataFrame()
    )
    base_lifecycle = attach_outcome_path_labels(base_lifecycle) if not base_lifecycle.empty else base_lifecycle

    causal_audit = build_causal_audit_v4(attempts)
    semantic_audit = build_semantic_causal_audit(attempts)
    maturity_audit = build_maturity_causal_audit(attempts)
    equal_audit = build_equal_pool_causal_audit(equal_pools, attempts)
    mechanism_audit = build_mechanism_causal_audit(attempts)
    audit = pd.concat([causal_audit, semantic_audit, maturity_audit, equal_audit, mechanism_audit], ignore_index=True, sort=False)
    stage.update(10)

    displacement_edges, displacement_perf, yearly_disp, fvg_timing_perf, reference_perf = _build_displacement_discovery_tables(base_lifecycle)
    family_perf, family_yearly, family_side, confluence_perf = _family_summary_tables(base_lifecycle)
    htf_structure = _htf_structure_diagnostics(base_lifecycle)
    semantic_edges, semantic_perf = build_semantic_feature_atlas(base_lifecycle, config=sem_cfg)
    semantic_categories = build_semantic_category_atlas(base_lifecycle, config=sem_cfg)
    mfe_transitions = build_mfe_transition_table(base_lifecycle, config=sem_cfg)
    failure_atlas = build_entry_failure_atlas(base_lifecycle, config=sem_cfg)
    stage.update(11)

    maturity_edges, maturity_perf = build_maturity_feature_atlas(base_lifecycle, config=maturity_cfg)
    maturity_pairs = build_maturity_pair_atlas(base_lifecycle, config=maturity_cfg)
    frequency_atlas = build_opportunity_frequency_atlas(base_lifecycle, valid_days, config=maturity_cfg)
    stage.update(12)

    archetype_perf = build_mechanism_archetype_atlas(base_lifecycle, config=mechanism_cfg)
    archetype_combos = build_mechanism_combination_atlas(base_lifecycle, config=mechanism_cfg)
    equal_perf = build_equal_pool_performance_atlas(base_lifecycle, config=mechanism_cfg)
    stage.update(13)

    # Explicit coverage check: R09 may add equal-pool opportunities, but all
    # existing non-EQ base families must still remain present in the output.
    base_family_attempts = attempts.loc[attempts["liquidity_family"].astype(str) != "equal_liquidity_pool"] if not attempts.empty else pd.DataFrame()
    coverage_rows = []
    if not base_family_attempts.empty:
        for (tf, fam), g in base_family_attempts.groupby(["execution_tf", "liquidity_family"], sort=True):
            coverage_rows.append({"execution_tf": tf, "liquidity_family": fam, "attempts": int(len(g)), "status": "preserved_broad_universe"})
    opportunity_preservation = pd.DataFrame(coverage_rows)
    stage.update(14)
    stage.close()

    (out_dir / "00_research_design.md").write_text(_design(args, cfg, len(valid_days)), encoding="utf-8")
    _write_csv(quality, out_dir / "01_data_quality.csv")
    _write_csv(premarket_levels, out_dir / "02_premarket_liquidity_levels.csv")
    _write_csv(htf_catalog, out_dir / "03_htf_swing_catalog.csv")
    _write_csv(htf_levels, out_dir / "04_active_unconsumed_htf_levels.csv")
    _write_csv(equal_pools, out_dir / "05_equal_high_low_pool_catalog.csv")
    _write_csv(sweeps_raw, out_dir / "06_sweep_events_raw.csv")
    _write_csv(sweeps, out_dir / "07_sweep_events_deduped.csv")
    _write_csv(funnel, out_dir / "08_mss_displacement_funnel.csv")
    _write_csv(attempts, out_dir / "09_signal_attempts.csv")
    _write_csv(base_lifecycle, out_dir / "10_base_trade_lifecycle.csv")
    _write_csv(base_summary, out_dir / "11_base_variant_summary.csv")
    _write_csv(cost_stress, out_dir / "12_cost_stress.csv")
    _write_csv(delay_stress, out_dir / "13_order_delay_stress.csv")
    _write_csv(family_perf, out_dir / "14_liquidity_family_compare.csv")
    _write_csv(family_yearly, out_dir / "15_liquidity_family_yearly.csv")
    _write_csv(family_side, out_dir / "16_liquidity_family_long_short.csv")
    _write_csv(confluence_perf, out_dir / "17_htf_confluence_compare.csv")
    _write_csv(htf_structure, out_dir / "18_htf_age_distance_rank_compare.csv")
    _write_csv(audit, out_dir / "19_causal_audit.csv")
    _write_csv(displacement_edges, out_dir / "20_displacement_feature_frozen_edges.csv")
    _write_csv(displacement_perf, out_dir / "21_displacement_feature_performance.csv")
    _write_csv(yearly_disp, out_dir / "22_displacement_yearly_performance.csv")
    _write_csv(fvg_timing_perf, out_dir / "23_fvg_timing_vs_mss_performance.csv")
    _write_csv(reference_perf, out_dir / "24_mss_reference_source_performance.csv")
    _write_csv(semantic_edges, out_dir / "25_semantic_feature_frozen_edges.csv")
    _write_csv(semantic_perf, out_dir / "26_semantic_feature_atlas.csv")
    _write_csv(semantic_categories, out_dir / "27_semantic_category_atlas.csv")
    _write_csv(mfe_transitions, out_dir / "28_mfe_transition_atlas.csv")
    _write_csv(failure_atlas, out_dir / "29_entry_failure_vs_favorable_failure.csv")
    _write_csv(maturity_edges, out_dir / "30_maturity_feature_frozen_edges.csv")
    _write_csv(maturity_perf, out_dir / "31_liquidity_consumption_maturity_atlas.csv")
    _write_csv(maturity_pairs, out_dir / "32_liquidity_consumption_pair_atlas.csv")
    _write_csv(frequency_atlas, out_dir / "33_opportunity_frequency_atlas.csv")
    _write_csv(mechanism_edges, out_dir / "34_mechanism_distribution_edges.csv")
    _write_csv(archetype_perf, out_dir / "35_mechanism_archetype_atlas.csv")
    _write_csv(archetype_combos, out_dir / "36_mechanism_combination_atlas.csv")
    _write_csv(equal_perf, out_dir / "37_equal_pool_performance_atlas.csv")
    _write_csv(opportunity_preservation, out_dir / "38_opportunity_preservation_audit.csv")

    findings = [
        "# R09 Mechanism Archetype Validation Findings", "",
        f"- Valid sessions: **{len(valid_days)}**.",
        f"- Active causal EQH/EQL pools at 08:30: **{len(equal_pools)}**.",
        f"- Deduped liquidity sweeps across all requested families: **{len(sweeps)}**.",
        f"- MSS+FVG attempts: **{len(attempts)}**.",
        "- Archetypes are overlapping research tags, not entry filters. Distribution landmarks are fitted without PnL using 2023H2-2024 only and frozen for 2025/2026.",
        "- R09 adds true EQH/EQL swing pools while retaining the prior broad premarket/15m/1H/4H/1D universe.",
    ]
    if not archetype_perf.empty:
        ranked = archetype_perf.loc[archetype_perf["analysis_period"].astype(str).str.contains("2025|2026", regex=True)].copy()
        ranked = ranked.sort_values(["profit_factor", "trades"], ascending=[False, False], na_position="last")
        findings += ["", "## Forward/holdout archetype observations (diagnostic only)"]
        for r in ranked.head(15).to_dict("records"):
            findings.append(
                f"- `{r.get('archetype')} / {r.get('liquidity_family')} / {r.get('execution_tf')} / {r.get('analysis_period')}`: "
                f"trades={int(r.get('trades', 0) or 0)}, PF={float(r.get('profit_factor', np.nan)):.3f}, "
                f"mean_net={float(r.get('mean_net_return', np.nan)):.3%}."
            )
    (out_dir / "39_findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")

    manifest = {
        "experiment_id": "SOXL_ICT_MSS_R09_MECHANISM_ARCHETYPE_VALIDATION",
        "edge_id": "SOXL_ICT_SWEEP_MSS_FVG_MECHANISM_ARCHETYPES",
        "data_source": args.data_source,
        "start_date": args.start_date, "end_date": args.end_date,
        "valid_sessions": len(valid_days),
        "session_ny": "04:00-16:30", "premarket_ny": "04:00-08:30", "trade_ny": "08:30-16:30",
        "liquidity_families": list(_csv_names(args.liquidity_families)),
        "equal_pool_source_timeframes": list(_csv_numbers(args.eq_source_timeframes, cast=int)),
        "equal_pool_definition": "causally confirmed same-side swing clusters, volatility-scaled tolerance, all active pools retained if unconsumed at 08:30",
        "archetype_protocol": "overlapping non-PnL tags; Q25/Q50/Q75 landmarks fit from 2023H2-2024 attempts and frozen for 2025/2026; no tag gates entry",
        "mss_definition": "liquidity sweep then close through latest causally confirmed opposing short-term pivot; post-terminal STH/STL valid",
        "round_trip_cost": cfg.round_trip_cost,
        "causal_audit_passed": bool(not audit.empty and audit["passed"].fillna(False).all()),
    }
    _write_json(manifest, out_dir / "40_manifest.json")
    _emit_platform_reports(base_lifecycle, bars, args, out_dir)
    if not args.skip_review_pack:
        finalize_research_report(
            out_dir, experiment_id=manifest["experiment_id"], edge_id=manifest["edge_id"],
            title="SOXL ICT R09 Mechanism Archetype Validation", print_log=True,
        )
    return {"report_dir": out_dir, "review_pack": out_dir / "gpt_review_pack.zip"}

def run_self_test(args: argparse.Namespace) -> int:
    bars = make_synthetic_ict_day()
    with tempfile.TemporaryDirectory(prefix="soxl_ict_r09_") as tmp:
        args.start_date = args.end_date = "2026-06-02"
        args.out_dir = tmp
        args.include_us_equity_holidays = True
        args.required_day_coverage = 1.0
        args.skip_platform_reports = True
        args.skip_review_pack = True
        args.no_progress = True
        result = run_research(bars, args)
        if not (result["report_dir"] / "19_causal_audit.csv").exists():
            raise AssertionError("missing combined causal/semantic/maturity/equal/mechanism audit")
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
