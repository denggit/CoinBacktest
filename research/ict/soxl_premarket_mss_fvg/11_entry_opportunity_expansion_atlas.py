#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SOXL ICT R11: entry opportunity expansion atlas.

R11 keeps the broad causal Sweep -> MSS -> FVG logic and expands *where* a
new liquidity cycle may start after the premarket story is partly/fully spent.
It also compares retracement execution models on the exact same frozen MSS
signals.  It is a discovery atlas, not a strict final strategy.
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
    NY_TZ, ReplayScenario, ResearchConfig, add_analysis_dimensions,
    build_data_quality_table, compound_account, eligible_ny_dates,
    enforce_single_lifecycle, make_synthetic_ict_day, ny_date_bounds_to_source_naive,
    replay_attempts, source_naive_to_new_york, summarize_variant,
)
from src.research_common.ict.premarket_mss_fvg_v2 import (  # noqa: E402
    SweepEpisodeConfig, build_all_premarket_levels_v2, build_sweep_events_v2,
)
from src.research_common.ict.premarket_mss_fvg_v4 import (  # noqa: E402
    ICTDisplacementDiscoveryConfig, build_causal_audit_v4, build_signal_attempts_v4,
)
from src.research_common.ict.htf_liquidity import (  # noqa: E402
    HTFLiquidityConfig, attach_first_consumption_time, build_htf_swing_catalog,
    build_remote_htf_levels_for_days, dedupe_same_family_sweeps,
)
from src.research_common.ict.equal_liquidity import (  # noqa: E402
    EqualLiquidityConfig, build_equal_liquidity_pools,
)
from src.research_common.ict.semantic_gap import (  # noqa: E402
    SemanticGapConfig, attach_causal_semantic_features,
)
from src.research_common.ict.liquidity_maturity import (  # noqa: E402
    LiquidityMaturityConfig, attach_liquidity_maturity_features,
)
from src.research_common.ict.entry_expansion import (  # noqa: E402
    EntryExpansionConfig, build_intraday_15m_swing_catalog,
    build_intraday_15m_sweep_events, entry_expansion_causal_audit,
    expand_entry_models, expand_intraday_target_models,
)
from src.research_common.ict.spot_perp_overlap import (  # noqa: E402
    build_equity_proxy_data_quality_table, densify_equity_minutes_causally,
)
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

DEFAULT_SYMBOL = "SOXL-USDT-SWAP"
DEFAULT_START_DATE = "2023-07-01"
DEFAULT_END_DATE = "2026-06-30"
DEFAULT_OUT_DIR = "data/reports/research/ict/soxl/mss/r11_entry_opportunity_expansion"
BASE_FAMILIES = (
    "premarket_extreme", "major_15m_swing", "remote_1h_swing",
    "remote_4h_swing", "remote_1d_swing", "equal_liquidity_pool",
)
ALL_FAMILIES = BASE_FAMILIES + ("intraday_15m_swing",)


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
    try:
        return int(str(text).strip().upper().replace("UTC", ""))
    except ValueError:
        return 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SOXL ICT R11 entry opportunity expansion atlas",
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
    p.add_argument("--base-liquidity-families", default=",".join(BASE_FAMILIES))
    p.add_argument("--htf-timeframes", default="1h,4h,1d")
    p.add_argument("--htf-pivot-left", type=int, default=2)
    p.add_argument("--htf-pivot-right", type=int, default=2)
    p.add_argument("--premarket-pivot-left", type=int, default=2)
    p.add_argument("--premarket-pivot-right", type=int, default=2)
    p.add_argument("--mss-pivot-left", type=int, default=1)
    p.add_argument("--mss-pivot-right", type=int, default=1)
    p.add_argument("--intraday-pivot-left", type=int, default=1)
    p.add_argument("--intraday-pivot-right", type=int, default=1)
    p.add_argument("--intraday-obvious-excursion-multiple", type=float, default=1.0)
    p.add_argument("--entry-models", default="fvg_near_edge,fvg_ce_50,order_block_open_proxy,order_block_midpoint_proxy")
    p.add_argument("--intraday-target-models", default="local_equilibrium_50,local_opposite_15m_swing")
    p.add_argument("--eq-source-timeframes", default="1,5,15")
    p.add_argument("--eq-pivot-left", type=int, default=1)
    p.add_argument("--eq-pivot-right", type=int, default=1)
    p.add_argument("--eq-min-members", type=int, default=2)
    p.add_argument("--eq-min-tolerance-bp", type=float, default=5.0)
    p.add_argument("--eq-range-tolerance-fraction", type=float, default=0.25)
    p.add_argument("--round-trip-cost", type=float, default=0.0011)
    p.add_argument("--cost-multipliers", default="1,2,3")
    p.add_argument("--order-delay-minutes", default="0,1,2")
    p.add_argument("--risk-fraction", type=float, default=0.01)
    p.add_argument("--max-notional-multiple", type=float, default=2.0)
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if any(x not in {1, 2, 5} for x in _csv_numbers(args.execution_timeframes, cast=int)):
        raise ValueError("execution_timeframes must be subset of 1,2,5")
    invalid = set(_csv_names(args.base_liquidity_families)) - set(BASE_FAMILIES)
    if invalid:
        raise ValueError(f"invalid base liquidity families: {sorted(invalid)}")
    if any(x not in {1, 5, 15} for x in _csv_numbers(args.eq_source_timeframes, cast=int)):
        raise ValueError("eq source timeframes must be subset of 1,5,15")
    if not 0.90 <= float(args.required_day_coverage) <= 1.0:
        raise ValueError("required_day_coverage must be in [0.90,1.0]")


def _base_cfg(args: argparse.Namespace) -> ResearchConfig:
    return ResearchConfig(
        execution_timeframes=tuple(_csv_numbers(args.execution_timeframes, cast=int)),
        mss_pivot_left=int(args.mss_pivot_left), mss_pivot_right=int(args.mss_pivot_right),
        premarket_pivot_left=int(args.premarket_pivot_left), premarket_pivot_right=int(args.premarket_pivot_right),
        required_day_coverage=float(args.required_day_coverage), round_trip_cost=float(args.round_trip_cost),
        risk_fraction=float(args.risk_fraction), max_notional_multiple=float(args.max_notional_multiple),
    )


def _path_cfg(args: argparse.Namespace) -> ICTDisplacementDiscoveryConfig:
    return ICTDisplacementDiscoveryConfig(
        execution_timeframes=tuple(_csv_numbers(args.execution_timeframes, cast=int)),
        mss_pivot_left=int(args.mss_pivot_left), mss_pivot_right=int(args.mss_pivot_right),
    )


def _entry_cfg(args: argparse.Namespace) -> EntryExpansionConfig:
    return EntryExpansionConfig(
        intraday_pivot_left=int(args.intraday_pivot_left),
        intraday_pivot_right=int(args.intraday_pivot_right),
        obvious_excursion_multiple=float(args.intraday_obvious_excursion_multiple),
        entry_models=tuple(_csv_names(args.entry_models)),
        target_models=tuple(_csv_names(args.intraday_target_models)),
    )


def _load_1m(args: argparse.Namespace) -> pd.DataFrame:
    if args.data_source == "alpaca":
        start_ny = pd.Timestamp(args.start_date).normalize().tz_localize(NY_TZ)
        end_ny = (pd.Timestamp(args.end_date).normalize() + pd.Timedelta(days=1)).tz_localize(NY_TZ)
        loader = AlpacaStockLoader(
            symbol=args.alpaca_symbol, timeframe="1Min", feed=args.alpaca_feed,
            adjustment=args.alpaca_adjustment, data_dir=args.data_dir,
        )
        print(f"[load] Alpaca {args.alpaca_symbol} 1Min {args.alpaca_feed}/{args.alpaca_adjustment} NY={args.start_date}->{args.end_date}", flush=True)
        raw = loader.fetch_data_by_date_range(
            start_ny.tz_convert("UTC"), end_ny.tz_convert("UTC") - pd.Timedelta(minutes=1),
            local_only=bool(args.local_only),
        )
        if raw.empty:
            raise RuntimeError("Alpaca loader returned no data")
        idx = pd.DatetimeIndex(raw.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        bars = raw.copy(); bars.index = idx.tz_convert(NY_TZ); bars.index.name = "bar_start_ny"
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
    bars = bars.loc[(mins >= 240) & (mins < 990)].copy()
    if args.data_source == "alpaca":
        bars = densify_equity_minutes_causally(bars)
    print(f"[load] rows={len(bars):,} NY={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path.name} rows={len(df):,}", flush=True)


def _period(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates, errors="coerce")
    return pd.Series(np.select(
        [d <= pd.Timestamp("2024-12-31"), d.dt.year == 2025, d.dt.year >= 2026],
        ["discovery_2023H2_2024", "forward_2025", "late_holdout_2026"], default="unknown",
    ), index=dates.index, dtype="object")


def _profit_factor(x: pd.Series) -> float:
    v = pd.to_numeric(x, errors="coerce").dropna()
    gains = float(v[v > 0].sum()); losses = float(-v[v < 0].sum())
    return gains / losses if losses > 0 else (np.inf if gains > 0 else np.nan)


def _positive_month_rate(group: pd.DataFrame) -> float:
    if group.empty or "fill_time" not in group or "account_return" not in group:
        return np.nan
    g = group.copy()
    ft = pd.to_datetime(g["fill_time"], errors="coerce", utc=True)
    g["month"] = ft.dt.strftime("%Y-%m")
    vals = []
    for _, m in g.groupby("month", sort=True):
        r = pd.to_numeric(m["account_return"], errors="coerce").dropna()
        if len(r):
            vals.append(float((1.0 + r).prod() - 1.0))
    return float(np.mean(np.asarray(vals) > 0)) if vals else np.nan


def _group_summary(g: pd.DataFrame, initial_capital: float) -> dict[str, object]:
    filled = g.loc[g["filled"].fillna(False).astype(bool)].copy() if "filled" in g else g.copy()
    if filled.empty:
        return {"trades": 0}
    net = pd.to_numeric(filled["net_return"], errors="coerce")
    gross = pd.to_numeric(filled["gross_return"], errors="coerce")
    account = compound_account(filled, initial_capital=float(initial_capital))
    return {
        "trades": int(len(filled)), "win_rate": float((net > 0).mean()),
        "mean_net_return": float(net.mean()), "median_net_return": float(net.median()),
        "profit_factor": _profit_factor(net), "gross_profit_factor": _profit_factor(gross),
        "mean_net_r": float(pd.to_numeric(filled["net_r"], errors="coerce").mean()),
        "median_mfe_r": float(pd.to_numeric(filled["mfe_r"], errors="coerce").median()),
        "median_mae_r": float(pd.to_numeric(filled["mae_r"], errors="coerce").median()),
        "median_planned_rr": float(pd.to_numeric(filled["planned_rr"], errors="coerce").median()),
        "target_hit_rate": float(filled["exit_reason"].astype(str).str.contains("target").mean()),
        "stop_hit_rate": float(filled["exit_reason"].astype(str).str.contains("stop").mean()),
        "positive_month_rate": _positive_month_rate(filled),
        "account_total_return": float(account["capital"].iloc[-1] / initial_capital - 1.0) if not account.empty else 0.0,
        "account_max_drawdown": float(pd.to_numeric(account["drawdown"], errors="coerce").min()) if not account.empty else 0.0,
    }


def _replay_expanded(
    bars: pd.DataFrame, attempts: pd.DataFrame, args: argparse.Namespace, cfg: ResearchConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if attempts.empty:
        return pd.DataFrame(), pd.DataFrame()
    scenarios: list[tuple[ReplayScenario, str]] = [
        (ReplayScenario("base" if c == 1 else f"cost_{c:g}x", c, 0), "cost")
        for c in _csv_numbers(args.cost_multipliers, cast=float)
    ]
    scenarios += [
        (ReplayScenario(f"delay_{d}m", 1.0, d), "delay")
        for d in _csv_numbers(args.order_delay_minutes, cast=int) if d != 0
    ]
    prog = ProgressReporter(label="[replay] entry expansion scenarios", total=len(scenarios), every=1, enabled=not args.no_progress)
    lifecycle_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    group_cols = ["execution_tf", "liquidity_family", "target_model", "entry_model"]
    for n, (scenario, stress_family) in enumerate(scenarios, start=1):
        replayed = replay_attempts(
            bars, attempts, scenario=scenario, round_trip_cost=cfg.round_trip_cost,
            risk_fraction=cfg.risk_fraction, max_notional_multiple=cfg.max_notional_multiple,
        )
        if not replayed.empty:
            for keys, group in replayed.groupby(group_cols, sort=True, dropna=False):
                kept, skipped = enforce_single_lifecycle(group)
                if not kept.empty:
                    kept["stress_family"] = stress_family
                    lifecycle_parts.append(kept)
                summary_rows.append({
                    **dict(zip(group_cols, keys)), "scenario": scenario.name,
                    "stress_family": stress_family, "cost_multiple": float(scenario.cost_multiple),
                    "order_delay_minutes": int(scenario.order_delay_minutes),
                    "skipped_overlap": int(skipped),
                    **summarize_variant(kept, skipped_overlap=skipped, initial_capital=float(args.initial_capital)),
                })
        prog.update(n)
    prog.close()
    lifecycle = pd.concat(lifecycle_parts, ignore_index=True) if lifecycle_parts else pd.DataFrame()
    return lifecycle, pd.DataFrame(summary_rows)


def _period_validation(base_lifecycle: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if base_lifecycle.empty:
        return pd.DataFrame()
    work = base_lifecycle.loc[base_lifecycle["filled"].fillna(False).astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["analysis_period"] = _period(work["ny_date"])
    rows: list[dict[str, object]] = []
    dims = ["analysis_period", "execution_tf", "liquidity_family", "target_model", "entry_model"]
    for keys, g in work.groupby(dims, sort=True, dropna=False):
        rows.append({**dict(zip(dims, keys)), **_group_summary(g, initial_capital)})
    return pd.DataFrame(rows)


def _intraday_state_validation(base_lifecycle: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if base_lifecycle.empty:
        return pd.DataFrame()
    work = base_lifecycle.loc[
        base_lifecycle["filled"].fillna(False).astype(bool)
        & (base_lifecycle["liquidity_family"].astype(str) == "intraday_15m_swing")
    ].copy()
    if work.empty:
        return pd.DataFrame()
    work["analysis_period"] = _period(work["ny_date"])
    rows = []
    dims = [
        "analysis_period", "execution_tf", "premarket_consumption_state_at_sweep",
        "target_model", "entry_model",
    ]
    for keys, g in work.groupby(dims, sort=True, dropna=False):
        rows.append({**dict(zip(dims, keys)), **_group_summary(g, initial_capital)})
    return pd.DataFrame(rows)


def _intraday_strength_atlas(base_lifecycle: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if base_lifecycle.empty:
        return pd.DataFrame()
    work = base_lifecycle.loc[
        base_lifecycle["filled"].fillna(False).astype(bool)
        & (base_lifecycle["liquidity_family"].astype(str) == "intraday_15m_swing")
    ].copy()
    if work.empty:
        return pd.DataFrame()
    x = pd.to_numeric(work["intraday_excursion_vs_known_median_15m_range"], errors="coerce")
    work["intraday_swing_strength_bucket"] = pd.cut(
        x, [-np.inf, 0.5, 1.0, 2.0, np.inf],
        labels=["<0.5x", "0.5-1.0x", "1.0-2.0x", ">2.0x"], right=False,
    ).astype(str)
    work["analysis_period"] = _period(work["ny_date"])
    rows = []
    dims = ["analysis_period", "execution_tf", "intraday_swing_strength_bucket", "target_model", "entry_model"]
    for keys, g in work.groupby(dims, sort=True, dropna=False):
        rows.append({**dict(zip(dims, keys)), **_group_summary(g, initial_capital)})
    return pd.DataFrame(rows)


def _entry_model_atlas(base_lifecycle: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if base_lifecycle.empty:
        return pd.DataFrame()
    work = base_lifecycle.loc[base_lifecycle["filled"].fillna(False).astype(bool)].copy()
    if work.empty:
        return pd.DataFrame()
    work["analysis_period"] = _period(work["ny_date"])
    rows = []
    dims = ["analysis_period", "execution_tf", "liquidity_family", "entry_model"]
    for keys, g in work.groupby(dims, sort=True, dropna=False):
        rows.append({**dict(zip(dims, keys)), **_group_summary(g, initial_capital)})
    return pd.DataFrame(rows)


def _target_model_atlas(base_lifecycle: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if base_lifecycle.empty:
        return pd.DataFrame()
    work = base_lifecycle.loc[
        base_lifecycle["filled"].fillna(False).astype(bool)
        & (base_lifecycle["liquidity_family"].astype(str) == "intraday_15m_swing")
    ].copy()
    if work.empty:
        return pd.DataFrame()
    work["analysis_period"] = _period(work["ny_date"])
    rows = []
    dims = ["analysis_period", "execution_tf", "target_model", "entry_model"]
    for keys, g in work.groupby(dims, sort=True, dropna=False):
        rows.append({**dict(zip(dims, keys)), **_group_summary(g, initial_capital)})
    return pd.DataFrame(rows)


def _opportunity_frequency(
    physical_base_sweeps: pd.DataFrame,
    intraday_sweeps: pd.DataFrame,
    valid_days: Sequence,
) -> pd.DataFrame:
    total_days = max(1, len(valid_days))
    rows = []
    for name, frame in (("base", physical_base_sweeps), ("intraday", intraday_sweeps)):
        if frame.empty:
            continue
        for family, g in frame.groupby("liquidity_family", sort=True):
            rows.append({
                "source": name, "liquidity_family": family,
                "physical_sweep_events": int(len(g)),
                "unique_days": int(g["ny_date"].astype(str).nunique()),
                "events_per_valid_day": float(len(g) / total_days),
                "day_coverage_rate": float(g["ny_date"].astype(str).nunique() / total_days),
            })
    if not intraday_sweeps.empty and "premarket_consumption_state_at_sweep" in intraday_sweeps:
        for state, g in intraday_sweeps.groupby("premarket_consumption_state_at_sweep", sort=True):
            rows.append({
                "source": "intraday_state", "liquidity_family": str(state),
                "physical_sweep_events": int(len(g)), "unique_days": int(g["ny_date"].astype(str).nunique()),
                "events_per_valid_day": float(len(g) / total_days),
                "day_coverage_rate": float(g["ny_date"].astype(str).nunique() / total_days),
            })
    return pd.DataFrame(rows)


def _design(args: argparse.Namespace, valid_days: int) -> str:
    return f"""# SOXL ICT R11 — Entry Opportunity Expansion Atlas

## Goal
Do **not** solve R10's weak wide-universe PF by stacking stricter filters. Expand and classify legitimate ICT entry opportunities while preserving causal execution.

## New liquidity cycle
A causally confirmed intraday 15m swing may become new liquidity after 08:30, including after both premarket high and low have already been consumed. R11 records the premarket-consumption state at both level confirmation and sweep instead of assuming the day's only valid liquidity was frozen at 08:30.

All causal intraday 15m pivots are catalogued; `intraday_excursion_vs_known_median_15m_range` is a descriptive quality feature only. The median 15m range uses only bars known by pivot confirmation.

## Local target research
For intraday 15m liquidity, target variants are separated:
- `local_equilibrium_50`: midpoint of the swept 15m swing and latest fresh opposite 15m swing. For shorts this is the first transition into the discount half; for longs it is the transition back through equilibrium toward the upper half.
- `local_opposite_15m_swing`: full local opposite swing target.

## Entry research
The MSS definition remains frozen. On the same causal signal R11 compares:
- FVG near edge (existing baseline)
- FVG 50% CE
- latest opposite-close displacement candle open (explicit quantitative Order Block proxy)
- midpoint of that proxy candle

Order Block variants are deliberately labelled `proxy`; they are research formulas, not a claim that discretion can be reduced to one canonical candle rule.

## Validation
2023-07-01..2024-12-31 discovery / 2025 forward / 2026 late holdout. No R11 quality bucket gates entry. Round-trip cost={args.round_trip_cost:.4%}; 2x/3x cost and 1m/2m delay are reported. Valid sessions={valid_days}.
"""


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    _validate_args(args)
    cfg = _base_cfg(args); path_cfg = _path_cfg(args); entry_cfg = _entry_cfg(args)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
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

    stage = ProgressReporter(label="[research] R11 stages", total=13, every=1, enabled=not args.no_progress)
    premarket_levels = build_all_premarket_levels_v2(
        bars, valid_days, pivot_left=cfg.premarket_pivot_left, pivot_right=cfg.premarket_pivot_right,
        episode_config=SweepEpisodeConfig(),
    )
    if not premarket_levels.empty:
        premarket_levels = premarket_levels.copy(); premarket_levels["liquidity_family"] = premarket_levels["level_type"].astype(str)
    stage.update(1)

    eq_cfg = EqualLiquidityConfig(
        source_timeframes=tuple(_csv_numbers(args.eq_source_timeframes, cast=int)),
        pivot_left=int(args.eq_pivot_left), pivot_right=int(args.eq_pivot_right), min_members=int(args.eq_min_members),
        min_tolerance_bp=float(args.eq_min_tolerance_bp), range_tolerance_fraction=float(args.eq_range_tolerance_fraction),
    )
    equal_pools = build_equal_liquidity_pools(bars, valid_days, premarket_levels, config=eq_cfg)
    htf_cfg = HTFLiquidityConfig(
        timeframes=tuple(_csv_names(args.htf_timeframes)), pivot_left=int(args.htf_pivot_left), pivot_right=int(args.htf_pivot_right),
    )
    htf_catalog = attach_first_consumption_time(bars, build_htf_swing_catalog(bars, config=htf_cfg))
    htf_levels = build_remote_htf_levels_for_days(htf_catalog, premarket_levels, valid_days)
    stage.update(2)

    parts = [x for x in (premarket_levels, htf_levels, equal_pools) if not x.empty]
    base_levels = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    allowed = set(_csv_names(args.base_liquidity_families))
    if not base_levels.empty:
        base_levels = base_levels.loc[base_levels["liquidity_family"].astype(str).isin(allowed)].copy()
    base_sweeps_raw = build_sweep_events_v2(bars, base_levels)
    base_sweeps = dedupe_same_family_sweeps(base_sweeps_raw)
    if not base_sweeps.empty:
        base_sweeps = base_sweeps.copy(); base_sweeps["target_model"] = "premarket_opposite_external"
    stage.update(3)

    intraday_catalog = build_intraday_15m_swing_catalog(bars, valid_days, premarket_levels, config=entry_cfg)
    intraday_sweeps = build_intraday_15m_sweep_events(bars, intraday_catalog, config=entry_cfg)
    intraday_target_sweeps = expand_intraday_target_models(intraday_sweeps, config=entry_cfg)
    stage.update(4)

    target_sweeps = pd.concat(
        [x for x in (base_sweeps, intraday_target_sweeps) if not x.empty], ignore_index=True, sort=False,
    ) if (not base_sweeps.empty or not intraday_target_sweeps.empty) else pd.DataFrame()
    attempts, funnel = build_signal_attempts_v4(bars, target_sweeps, config=path_cfg, progress_enabled=not args.no_progress)
    stage.update(5)

    sem_cfg = SemanticGapConfig(discovery_end_year=2024, forward_year=2025, holdout_year=2026)
    mat_cfg = LiquidityMaturityConfig(discovery_end_date="2024-12-31", forward_start_date="2025-01-01", forward_end_date="2025-12-31", holdout_start_date="2026-01-01")
    if not attempts.empty:
        attempts = attach_causal_semantic_features(attempts, bars, config=sem_cfg)
        attempts = attach_liquidity_maturity_features(attempts, bars, config=mat_cfg)
    stage.update(6)

    expanded_attempts, entry_variant_audit = expand_entry_models(attempts, bars, config=entry_cfg)
    stage.update(7)

    lifecycle, scenario_summary = _replay_expanded(bars, expanded_attempts, args, cfg)
    stage.update(8)
    base_lifecycle = add_analysis_dimensions(
        lifecycle.loc[(lifecycle["scenario"] == "base") & (lifecycle["stress_family"] == "cost")].copy()
    ) if not lifecycle.empty else pd.DataFrame()

    causal_base = build_causal_audit_v4(attempts)
    causal_expand = entry_expansion_causal_audit(intraday_catalog, intraday_sweeps, expanded_attempts)
    audit = pd.concat([causal_base, causal_expand], ignore_index=True, sort=False)
    stage.update(9)

    entry_atlas = _entry_model_atlas(base_lifecycle, float(args.initial_capital))
    target_atlas = _target_model_atlas(base_lifecycle, float(args.initial_capital))
    state_atlas = _intraday_state_validation(base_lifecycle, float(args.initial_capital))
    strength_atlas = _intraday_strength_atlas(base_lifecycle, float(args.initial_capital))
    period_validation = _period_validation(base_lifecycle, float(args.initial_capital))
    stage.update(10)

    frequency = _opportunity_frequency(base_sweeps, intraday_sweeps, valid_days)
    ob_audit = entry_variant_audit.loc[entry_variant_audit["entry_model"].astype(str).str.startswith("order_block")].copy() if not entry_variant_audit.empty else pd.DataFrame()
    if not ob_audit.empty:
        ob_audit = ob_audit.groupby("entry_model", sort=True).agg(
            variants=("attempt_id", "size"), valid_variants=("valid_entry_variant", "sum"),
            proxy_available_rate=("ob_proxy_available", "mean"), mitigated_before_signal_rate=("ob_proxy_mitigated_before_signal", "mean"),
        ).reset_index()
    stage.update(11)

    # Baseline preservation: every original R09 attempt must still exist as the
    # FVG-near-edge/premarket-target variant. R11 may add opportunities, never
    # silently delete the frozen base signal universe.
    preservation_rows = []
    if not attempts.empty and not expanded_attempts.empty:
        base_attempt_ids = set(attempts.loc[attempts["liquidity_family"].astype(str) != "intraday_15m_swing", "attempt_id"].astype(str))
        near_ids = set(
            expanded_attempts.loc[
                (expanded_attempts["liquidity_family"].astype(str) != "intraday_15m_swing")
                & (expanded_attempts["entry_model"] == "fvg_near_edge"), "attempt_id"
            ].astype(str).str.replace(r"\|entry=fvg_near_edge$", "", regex=True)
        )
        preservation_rows.append({
            "check": "base_attempts_preserved_in_fvg_near_edge_variant",
            "base_attempts": len(base_attempt_ids), "preserved_attempts": len(base_attempt_ids & near_ids),
            "passed": base_attempt_ids.issubset(near_ids),
        })
    preservation = pd.DataFrame(preservation_rows)
    stage.update(12)

    (out_dir / "00_research_design.md").write_text(_design(args, len(valid_days)), encoding="utf-8")
    _write_csv(quality, out_dir / "01_data_quality.csv")
    _write_csv(premarket_levels, out_dir / "02_premarket_liquidity_levels.csv")
    _write_csv(htf_catalog, out_dir / "03_htf_swing_catalog.csv")
    _write_csv(htf_levels, out_dir / "04_active_unconsumed_htf_levels.csv")
    _write_csv(equal_pools, out_dir / "05_equal_liquidity_pool_catalog.csv")
    _write_csv(base_sweeps, out_dir / "06_base_sweep_events.csv")
    _write_csv(intraday_catalog, out_dir / "07_intraday_15m_swing_catalog.csv")
    _write_csv(intraday_sweeps, out_dir / "08_intraday_15m_physical_sweeps.csv")
    _write_csv(intraday_target_sweeps, out_dir / "09_intraday_target_variants.csv")
    _write_csv(funnel, out_dir / "10_mss_fvg_funnel.csv")
    _write_csv(attempts, out_dir / "11_frozen_mss_fvg_attempts.csv")
    _write_csv(entry_variant_audit, out_dir / "12_entry_variant_construction_audit.csv")
    _write_csv(expanded_attempts, out_dir / "13_expanded_entry_attempts.csv")
    _write_csv(base_lifecycle, out_dir / "14_base_cost_trade_lifecycle.csv")
    _write_csv(scenario_summary, out_dir / "15_cost_delay_scenario_summary.csv")
    _write_csv(entry_atlas, out_dir / "16_entry_model_atlas.csv")
    _write_csv(target_atlas, out_dir / "17_intraday_target_model_atlas.csv")
    _write_csv(state_atlas, out_dir / "18_premarket_consumption_state_atlas.csv")
    _write_csv(strength_atlas, out_dir / "19_intraday_swing_strength_atlas.csv")
    _write_csv(period_validation, out_dir / "20_period_validation.csv")
    _write_csv(frequency, out_dir / "21_opportunity_frequency_expanded.csv")
    _write_csv(ob_audit, out_dir / "22_order_block_proxy_audit.csv")
    _write_csv(audit, out_dir / "23_causal_audit.csv")
    _write_csv(preservation, out_dir / "24_base_universe_preservation_audit.csv")

    findings = [
        "# R11 Entry Opportunity Expansion Findings", "",
        f"- Valid sessions: **{len(valid_days)}**.",
        f"- Existing physical liquidity sweeps: **{len(base_sweeps)}**.",
        f"- New causal intraday 15m swing levels: **{len(intraday_catalog)}**.",
        f"- New physical intraday 15m sweeps: **{len(intraday_sweeps)}**.",
        f"- Frozen MSS+FVG attempts before entry-price expansion: **{len(attempts)}**.",
        f"- Entry-price/target research attempts: **{len(expanded_attempts)}**.",
        "- Intraday 15m swing strength is descriptive only; R11 does not require an excursion threshold to enter.",
        "- EQH/EQL remains ordinary liquidity/context; R11 does not promote it to a special gate.",
        "- Order Block formulas are explicitly labelled quantitative proxies; use results to decide whether the proxy deserves deeper ICT-specific refinement.",
    ]
    both = intraday_sweeps.loc[intraday_sweeps.get("premarket_consumption_state_at_sweep", pd.Series(index=intraday_sweeps.index, dtype=str)).astype(str) == "both_premarket_sides_consumed"] if not intraday_sweeps.empty else pd.DataFrame()
    findings.append(f"- Intraday 15m physical sweeps occurring after **both** premarket sides were consumed: **{len(both)}**.")
    if not state_atlas.empty:
        ranked = state_atlas.loc[state_atlas["premarket_consumption_state_at_sweep"] == "both_premarket_sides_consumed"].copy()
        ranked = ranked.loc[ranked["trades"] >= 10].sort_values(["profit_factor", "trades"], ascending=[False, False])
        if not ranked.empty:
            findings += ["", "## Best both-premarket-consumed intraday rows (diagnostic, not a frozen rule)"]
            for r in ranked.head(12).to_dict("records"):
                findings.append(
                    f"- {r['analysis_period']} {r['execution_tf']} {r['target_model']} {r['entry_model']}: "
                    f"trades={int(r['trades'])}, PF={float(r['profit_factor']):.3f}, grossPF={float(r['gross_profit_factor']):.3f}, "
                    f"MDD={float(r['account_max_drawdown']):.2%}."
                )
    (out_dir / "25_findings.md").write_text("\n".join(findings) + "\n", encoding="utf-8")

    manifest = {
        "experiment_id": "SOXL_ICT_MSS_R11_ENTRY_OPPORTUNITY_EXPANSION",
        "edge_id": "SOXL_ICT_SWEEP_MSS_FVG_ENTRY_EXPANSION",
        "data_source": args.data_source, "start_date": args.start_date, "end_date": args.end_date,
        "valid_sessions": len(valid_days), "round_trip_cost": cfg.round_trip_cost,
        "new_liquidity_family": "intraday_15m_swing",
        "intraday_target_models": list(entry_cfg.target_models),
        "entry_models": list(entry_cfg.entry_models),
        "causal_audit_passed": bool(not audit.empty and audit["passed"].fillna(False).all()),
        "base_universe_preserved": bool(preservation.empty or preservation["passed"].fillna(False).all()),
        "protocol": "broad entry discovery; no R11 swing-quality/consumption/OB/FVG-CE feature gates entry",
    }
    (out_dir / "26_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    stage.update(13); stage.close()

    if not args.skip_review_pack:
        finalize_research_report(
            out_dir, experiment_id=manifest["experiment_id"], edge_id=manifest["edge_id"],
            title="SOXL ICT R11 Entry Opportunity Expansion Atlas", print_log=True,
        )
    return {"report_dir": out_dir, "review_pack": out_dir / "gpt_review_pack.zip"}


def run_self_test(args: argparse.Namespace) -> int:
    # Two synthetic sessions: the inherited ICT day validates base preservation;
    # second day is deliberately warped to create both-sided premarket sweeps
    # and later intraday 15m structure. Self-test checks report/audits, while
    # unit tests below cover exact intraday-event semantics.
    bars = make_synthetic_ict_day("2026-06-02")
    with tempfile.TemporaryDirectory(prefix="soxl_ict_r11_") as tmp:
        args.start_date = args.end_date = "2026-06-02"
        args.out_dir = tmp; args.include_us_equity_holidays = True
        args.required_day_coverage = 1.0; args.skip_review_pack = True; args.no_progress = True
        result = run_research(bars, args)
        audit = pd.read_csv(result["report_dir"] / "23_causal_audit.csv")
        if audit.empty or not audit["passed"].fillna(False).all():
            raise AssertionError("R11 causal audit failed")
        pres = pd.read_csv(result["report_dir"] / "24_base_universe_preservation_audit.csv")
        if not pres.empty and not pres["passed"].fillna(False).all():
            raise AssertionError("R11 removed the frozen base FVG-near entry universe")
        if not (result["report_dir"] / "16_entry_model_atlas.csv").exists():
            raise AssertionError("missing entry model atlas")
    print("[self-test] PASS", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); _validate_args(args)
    if args.self_test:
        return run_self_test(args)
    run_research(_load_1m(args), args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
