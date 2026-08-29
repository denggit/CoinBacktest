#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ICT MSS R02: causal liquidity taxonomy + 1m/2m execution/session atlas.

R01 showed that unconditional HTF-sweep -> MSS -> displacement -> FVG is not an
edge.  R02 therefore studies *which already-confirmed swing levels actually
behave like meaningful liquidity* without outcome-fitted thresholds.

Signal-time safety
------------------
- HTF pivots are usable only after right-side confirmation bars close.
- Liquidity quality uses data ending at sweep-bar open; sweep-bar OHLC is not
  used to decide whether the level was high quality.
- 1m and native 2m execution bars are both derived from the same 1m bare OHLC.
- A MSS may break only a micro pivot already confirmed before displacement.
- FVG is actionable only after its third candle closes; the limit starts on the
  next execution bar.
- Fill-bar favorable extreme cannot hit TP; same-bar ambiguity is stop-first.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Keep the same compatibility semantics as src.data_feed.okx_loader.
    from config.loader import TIMEZONE  # type: ignore[attr-defined]  # noqa: E402
except (ImportError, AttributeError):  # pragma: no cover - repo-version compatibility
    TIMEZONE = "+8"
from research.ict.mss.common.evaluation import profit_factor  # noqa: E402
from research.ict.mss.common.execution import attach_limit_entry_and_outcomes  # noqa: E402
from research.ict.mss.common.liquidity import (  # noqa: E402
    aggregate_quality_to_sweep_episodes,
    enrich_level_sweeps_with_causal_quality,
    liquidity_taxonomy_counts,
)
from research.ict.mss.common.structure import (  # noqa: E402
    aggregate_timeframe,
    build_displacement_fvgs,
    build_htf_liquidity_levels,
    build_micro_structure_context,
    build_sweep_episodes,
    normalize_bars,
    pair_sweeps_with_mss_fvgs,
)
from research.ict.mss.common.time_context import (  # noqa: E402
    add_calendar_session_context,
    parse_project_timezone_offset_hours,
)
from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "02_ict_mss_liquidity_taxonomy"
SCRIPT_VERSION = "2.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS_LIQUIDITY_TAXONOMY_R02"
EDGE_ID = "ICT_MSS_CAUSAL_LIQUIDITY_QUALITY_SESSION_EXEC_TF"
TITLE = "ETH ICT MSS Liquidity Taxonomy R02"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss/02_ict_mss_liquidity_taxonomy"
HTF_TIMEFRAMES = (("15m", 15), ("30m", 30), ("1H", 60), ("4H", 240))
HTF_CONFIRMATION_ORDERS = (1, 2, 3, 5)
MICRO_ORDERS = (2, 3, 5)
EXECUTION_TIMEFRAMES = (1, 2)
TARGET_RS = (1.0, 2.0, 3.0)
ROUND_TRIP_COST = 0.0011


@dataclass(frozen=True)
class CandidateRule:
    candidate_id: str
    execution_minutes: int
    micro_order: int
    liquidity_rule: str
    time_rule: str


LIQUIDITY_RULES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "ALL": lambda x: pd.Series(True, index=x.index),
    "STRUCTURAL_MAJOR": lambda x: x["lq_structural_major"].astype(bool),
    "MATURE_24H": lambda x: x["lq_mature_24h"].astype(bool),
    "MATURE_72H": lambda x: x["lq_mature_72h"].astype(bool),
    "REMOTE_100BP": lambda x: x["lq_remote_100bp"].astype(bool),
    "REMOTE_200BP": lambda x: x["lq_remote_200bp"].astype(bool),
    "STACKED_10BP": lambda x: x["lq_stacked_10bp"].astype(bool),
    "STACKED_MULTI_TF": lambda x: x["lq_stacked_multi_tf_10bp"].astype(bool),
    "MAJOR_REMOTE": lambda x: x["lq_major_remote"].astype(bool),
    "MAJOR_REMOTE_STACKED": lambda x: x["lq_major_remote_stacked"].astype(bool),
    "4H_MATURE": lambda x: x["lq_4h_mature"].astype(bool),
    "1HPLUS_REMOTE": lambda x: x["lq_1hplus_remote"].astype(bool),
}

TIME_RULES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "ALL": lambda x: pd.Series(True, index=x.index),
    "WEEKDAY": lambda x: x["is_weekday_utc"].astype(bool),
    "WEEKEND": lambda x: x["is_weekend_utc"].astype(bool),
    "ASIA": lambda x: x["session_asia"].astype(bool),
    "LONDON": lambda x: x["session_london"].astype(bool),
    "NEW_YORK": lambda x: x["session_new_york"].astype(bool),
    "LONDON_KZ": lambda x: x["ict_london_kill_zone"].astype(bool),
    "NY_KZ": lambda x: x["ict_new_york_kill_zone"].astype(bool),
    "US_CASH_OPEN_90M": lambda x: x["us_cash_open_90m"].astype(bool),
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--round-trip-cost-pct", type=float, default=ROUND_TRIP_COST)
    p.add_argument("--sweep-epsilon-bp", type=float, default=0.01)
    p.add_argument("--max-search-minutes", type=int, default=180)
    p.add_argument("--max-fill-wait-minutes", type=int, default=120)
    p.add_argument("--outcome-horizon-minutes", type=int, default=240)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--write-full-setups", action="store_true")
    return p.parse_args(argv)


def _date_end(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts + pd.Timedelta(days=1) - pd.Timedelta(minutes=1) if len(str(value).strip()) <= 10 else ts


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _load_1m(args: argparse.Namespace) -> pd.DataFrame:
    end = _date_end(args.end_date)
    print(f"[load] src.data_feed.OKXDataLoader {args.symbol} 1m {args.warmup_start_date} -> {end}", flush=True)
    loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
    raw = loader.fetch_data_by_date_range(pd.Timestamp(args.warmup_start_date), end)
    if raw.empty:
        raise RuntimeError("No 1m OHLC loaded through src.data_feed.OKXDataLoader")
    bars = normalize_bars(raw[["open", "high", "low", "close"]])
    print(f"[load] rows={len(bars):,} {bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _exec_bars(bars_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    if minutes == 1:
        return bars_1m
    h = aggregate_timeframe(bars_1m, minutes)
    return normalize_bars(h[["open", "high", "low", "close"]])


def _core_displacement_mask(x: pd.DataFrame) -> pd.Series:
    return (
        x["displacement_body_vs_past_median"].ge(2.0)
        & x["displacement_range_vs_past_median"].ge(1.8)
        & x["displacement_body_fraction"].ge(0.64)
        & x["displacement_close_from_extreme_fraction"].le(0.18)
        & x["fvg_size_bp"].ge(3.0)
    )


def _dedup_base_setups(frame: pd.DataFrame, *, research_start: pd.Timestamp, research_end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    x = frame.loc[
        frame["structure_mode"].eq("pre_sweep")
        & _core_displacement_mask(frame)
        & frame["entry_structure_valid"].astype(bool)
    ].copy()
    # Physical windows, not bar-count windows, so 1m and 2m are comparable.
    x = x.loc[x["sweep_to_displacement_minutes"].le(120.0) & x["fill_wait_minutes"].le(120.0)].copy()
    x = x.sort_values(
        [
            "execution_minutes", "micro_order", "fvg_completion_time", "side",
            "max_timeframe_min", "max_confirmed_order", "oldest_level_age_minutes_q",
            "max_excursion_away_bp_before_sweep", "max_cluster_count_10bp", "sweep_pos",
        ],
        ascending=[True, True, True, True, False, False, False, False, False, False],
        kind="mergesort",
    )
    x = x.drop_duplicates(["execution_minutes", "micro_order", "side", "fvg_completion_time"], keep="first")
    signal = pd.to_datetime(x["fvg_available_time"], errors="coerce")
    x = x.loc[(signal >= research_start) & (signal <= research_end)].copy()
    return x.reset_index(drop=True)


def _period(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts, errors="coerce")
    out = pd.Series("OUTSIDE", index=ts.index, dtype="object")
    out[(t >= "2023-01-01") & (t < "2024-01-01")] = "2023"
    out[(t >= "2024-01-01") & (t < "2025-01-01")] = "2024"
    out[(t >= "2025-01-01") & (t < "2026-01-01")] = "2025"
    out[(t >= "2026-01-01") & (t < "2026-07-01")] = "2026H1"
    return out


def _metric_row(x: pd.DataFrame, target_r: float, cost_mult: float = 1.0) -> dict[str, object]:
    token = str(float(target_r)).replace(".", "p")
    gross_col = f"gross_r_r{token}"
    gross_pct_col = f"gross_return_pct_r{token}"
    result_col = f"result_r{token}"
    if x.empty or gross_col not in x:
        return {"trades": 0, "mean_net_r": np.nan, "profit_factor": np.nan, "win_rate": np.nan, "target_rate": np.nan, "positive_month_share": np.nan, "median_risk_pct": np.nan}
    risk = pd.to_numeric(x["risk_pct"], errors="coerce")
    gross = pd.to_numeric(x[gross_col], errors="coerce")
    cost_r = ROUND_TRIP_COST * float(cost_mult) / risk
    net = gross - cost_r
    valid = np.isfinite(net) & np.isfinite(risk) & risk.gt(0)
    net = net.loc[valid]
    if net.empty:
        return {"trades": 0, "mean_net_r": np.nan, "profit_factor": np.nan, "win_rate": np.nan, "target_rate": np.nan, "positive_month_share": np.nan, "median_risk_pct": np.nan}
    xx = x.loc[net.index]
    gross = gross.loc[net.index]
    cost_r = cost_r.loc[net.index]
    gross_pct = pd.to_numeric(xx[gross_pct_col], errors="coerce") if gross_pct_col in xx.columns else pd.Series(np.nan, index=xx.index)
    net_pct = gross_pct - ROUND_TRIP_COST * float(cost_mult)
    months = pd.to_datetime(xx["first_fill_time"]).dt.to_period("M").astype(str)
    monthly = net.groupby(months).sum()
    return {
        "trades": int(len(net)),
        "mean_gross_r": float(gross.mean()),
        "mean_cost_r": float(cost_r.mean()),
        "mean_net_r": float(net.mean()),
        "mean_gross_return_pct": float(gross_pct.mean()),
        "mean_net_return_pct": float(net_pct.mean()),
        "median_net_r": float(net.median()),
        "profit_factor": float(profit_factor(net)),
        "win_rate": float((net > 0).mean()),
        "target_rate": float(xx[result_col].eq("TARGET").mean()),
        "positive_month_share": float((monthly > 0).mean()) if len(monthly) else np.nan,
        "median_risk_pct": float(risk.loc[net.index].median()),
        "top10_removed_mean_net_r": float(net.sort_values(ascending=False).iloc[min(10, len(net)):].mean()) if len(net) > 10 else np.nan,
        "top10_removed_profit_factor": float(profit_factor(net.sort_values(ascending=False).iloc[min(10, len(net)):])) if len(net) > 10 else np.nan,
    }


def _atlas_rows(
    setups: pd.DataFrame,
    *,
    dimension: str,
    values: pd.Series,
    target_r: float = 2.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    tmp = setups.copy()
    tmp["_bucket"] = values.astype("object")
    tmp["_period"] = _period(tmp["first_fill_time"])
    for (emin, micro, bucket), part in tmp.groupby(
        ["execution_minutes", "micro_order", "_bucket"],
        dropna=False,
        observed=False,
    ):
        for period_name, p in [("FULL", part), *list(part.groupby("_period", sort=False, observed=False))]:
            rows.append({
                "execution_minutes": int(emin),
                "micro_order": int(micro),
                "dimension": dimension,
                "bucket": str(bucket),
                "period": str(period_name),
                "target_r": target_r,
                **_metric_row(p, target_r),
            })
    return rows


def _build_candidate_rules() -> tuple[CandidateRule, ...]:
    rows: list[CandidateRule] = []
    for execution_minutes in EXECUTION_TIMEFRAMES:
        for micro_order in (2, 5):
            for liquidity_rule in LIQUIDITY_RULES:
                for time_rule in TIME_RULES:
                    rows.append(CandidateRule(
                        candidate_id=f"E{execution_minutes}M_M{micro_order}_{liquidity_rule}_{time_rule}",
                        execution_minutes=execution_minutes,
                        micro_order=micro_order,
                        liquidity_rule=liquidity_rule,
                        time_rule=time_rule,
                    ))
    return tuple(rows)


def _candidate_tables(setups: pd.DataFrame, *, cost_pct: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global ROUND_TRIP_COST
    ROUND_TRIP_COST = float(cost_pct)
    if setups.empty or "execution_minutes" not in setups.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    metric_rows: list[dict[str, object]] = []
    stress_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    rules = _build_candidate_rules()
    for rule in rules:
        base = setups.loc[
            setups["execution_minutes"].eq(rule.execution_minutes)
            & setups["micro_order"].eq(rule.micro_order)
        ].copy()
        if not base.empty:
            base = base.loc[LIQUIDITY_RULES[rule.liquidity_rule](base) & TIME_RULES[rule.time_rule](base)].copy()
            base["period"] = _period(base["first_fill_time"])
        for target_r in TARGET_RS:
            full = _metric_row(base, target_r)
            periods: dict[str, dict[str, object]] = {}
            for pname in ("2023", "2024", "2025", "2026H1"):
                part = base.loc[base["period"].eq(pname)] if not base.empty else base
                pm = _metric_row(part, target_r)
                periods[pname] = pm
                metric_rows.append({
                    "candidate_id": rule.candidate_id,
                    "execution_minutes": rule.execution_minutes,
                    "micro_order": rule.micro_order,
                    "liquidity_rule": rule.liquidity_rule,
                    "time_rule": rule.time_rule,
                    "target_r": target_r,
                    "period": pname,
                    **pm,
                })
            metric_rows.append({
                "candidate_id": rule.candidate_id,
                "execution_minutes": rule.execution_minutes,
                "micro_order": rule.micro_order,
                "liquidity_rule": rule.liquidity_rule,
                "time_rule": rule.time_rule,
                "target_r": target_r,
                "period": "FULL",
                **full,
            })
            pre = base.loc[pd.to_datetime(base["first_fill_time"], errors="coerce") < pd.Timestamp("2026-01-01")] if not base.empty else base
            stress = {m: _metric_row(pre, target_r, m) for m in (1.0, 2.0, 3.0)}
            for mult, sm in stress.items():
                stress_rows.append({"candidate_id": rule.candidate_id, "target_r": target_r, "scope": "2023_2025", "cost_multiplier": mult, **sm})

            y23, y24, y25, y26 = (periods[p] for p in ("2023", "2024", "2025", "2026H1"))
            dev_pass = all(
                int(y["trades"]) >= 30
                and np.isfinite(y["mean_net_r"]) and float(y["mean_net_r"]) >= 0.05
                and np.isfinite(y["profit_factor"]) and float(y["profit_factor"]) >= 1.10
                for y in (y23, y24)
            )
            validation_pass = (
                int(y25["trades"]) >= 20 and np.isfinite(y25["mean_net_r"]) and float(y25["mean_net_r"]) > 0.0
                and np.isfinite(y25["profit_factor"]) and float(y25["profit_factor"]) >= 1.05
            )
            cost2 = stress[2.0]
            cost2_pass = (
                int(cost2["trades"]) >= 100 and np.isfinite(cost2["mean_net_r"]) and float(cost2["mean_net_r"]) > 0.0
                and np.isfinite(cost2["profit_factor"]) and float(cost2["profit_factor"]) >= 1.0
            )
            top10_pass = (
                int(stress[1.0]["trades"]) >= 100
                and np.isfinite(stress[1.0]["top10_removed_mean_net_r"])
                and float(stress[1.0]["top10_removed_mean_net_r"]) > 0.0
                and np.isfinite(stress[1.0]["top10_removed_profit_factor"])
                and float(stress[1.0]["top10_removed_profit_factor"]) >= 1.0
            )
            forward_pass = (
                int(y26["trades"]) >= 10 and np.isfinite(y26["mean_net_r"]) and float(y26["mean_net_r"]) >= 0.0
                and np.isfinite(y26["profit_factor"]) and float(y26["profit_factor"]) >= 1.0
            )
            robust_full = (
                int(full["trades"]) >= 150
                and np.isfinite(full["mean_net_r"]) and float(full["mean_net_r"]) >= 0.08
                and np.isfinite(full["profit_factor"]) and float(full["profit_factor"]) >= 1.25
                and np.isfinite(full["positive_month_share"]) and float(full["positive_month_share"]) >= 0.60
            )
            gate_rows.append({
                "candidate_id": rule.candidate_id,
                "execution_minutes": rule.execution_minutes,
                "micro_order": rule.micro_order,
                "liquidity_rule": rule.liquidity_rule,
                "time_rule": rule.time_rule,
                "target_r": target_r,
                "development_2023_2024_pass": bool(dev_pass),
                "validation_2025_pass": bool(validation_pass),
                "cost_2x_2023_2025_pass": bool(cost2_pass),
                "top10_removed_2023_2025_pass": bool(top10_pass),
                "forward_2026h1_pass": bool(forward_pass),
                "strong_full_sample_pass": bool(robust_full),
                "edge_found": bool(dev_pass and validation_pass and cost2_pass and top10_pass and forward_pass and robust_full),
            })
    return pd.DataFrame(metric_rows), pd.DataFrame(stress_rows), pd.DataFrame(gate_rows)


def _causal_audit(levels: pd.DataFrame, level_sweeps: pd.DataFrame, setups: pd.DataFrame, execution_minutes: int) -> pd.DataFrame:
    checks: list[dict[str, object]] = []
    checks.append({
        "execution_minutes": execution_minutes,
        "check": "htf_level_available_no_later_than_sweep_bar_open",
        "violations": int((pd.to_datetime(level_sweeps["initial_available_time"]) > pd.to_datetime(level_sweeps["sweep_bar_time"])).sum()) if not level_sweeps.empty else 0,
    })
    checks.append({
        "execution_minutes": execution_minutes,
        "check": "confirmed_liquidity_order_available_by_sweep_bar_open",
        "violations": int((pd.to_datetime(level_sweeps["confirmed_order_available_time"]) > pd.to_datetime(level_sweeps["sweep_bar_time"])).sum()) if not level_sweeps.empty else 0,
    })
    checks.append({
        "execution_minutes": execution_minutes,
        "check": "liquidity_excursion_excludes_sweep_bar",
        "violations": int((level_sweeps["quality_feature_last_pos"] >= level_sweeps["sweep_pos"]).sum()) if not level_sweeps.empty else 0,
    })
    if not setups.empty:
        checks.extend([
            {
                "execution_minutes": execution_minutes,
                "check": "micro_structure_available_before_displacement",
                "violations": int((setups["micro_structure_available_pos"] > setups["displacement_pos"]).sum()),
            },
            {
                "execution_minutes": execution_minutes,
                "check": "fvg_available_after_completion",
                "violations": int((pd.to_datetime(setups["fvg_available_time"]) <= pd.to_datetime(setups["fvg_completion_time"])).sum()),
            },
            {
                "execution_minutes": execution_minutes,
                "check": "limit_active_after_fvg_completion",
                "violations": int((setups["order_active_pos"] <= setups["fvg_completion_pos"]).sum()),
            },
            {
                "execution_minutes": execution_minutes,
                "check": "fill_not_before_limit_activation",
                "violations": int(((setups["first_fill_pos"] >= 0) & (setups["first_fill_pos"] < setups["order_active_pos"])).sum()),
            },
        ])
    return pd.DataFrame(checks)


def run(args: argparse.Namespace) -> Path:
    global ROUND_TRIP_COST
    ROUND_TRIP_COST = float(args.round_trip_cost_pct)
    show_progress = not args.no_progress
    start = pd.Timestamp(args.start_date)
    end = _date_end(args.end_date)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bars_1m = _load_1m(args)
    print("[stage] shared causal HTF liquidity levels", flush=True)
    levels = build_htf_liquidity_levels(bars_1m, timeframes=HTF_TIMEFRAMES, confirmation_orders=HTF_CONFIRMATION_ORDERS)
    print(f"[levels] {len(levels):,}", flush=True)

    all_setups: list[pd.DataFrame] = []
    taxonomy_counts: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    level_quality_samples: list[pd.DataFrame] = []
    mechanism_rows: list[dict[str, object]] = []
    tz_offset = parse_project_timezone_offset_hours(TIMEZONE, default=8)

    for execution_minutes in EXECUTION_TIMEFRAMES:
        label = f"{execution_minutes}m"
        print(f"[stage:{label}] build complete execution bars", flush=True)
        bars = _exec_bars(bars_1m, execution_minutes)
        print(f"[{label}] rows={len(bars):,}", flush=True)

        print(f"[stage:{label}] first sweeps + causal liquidity quality", flush=True)
        level_sweeps, episodes = build_sweep_episodes(
            bars,
            levels,
            confirmation_orders=HTF_CONFIRMATION_ORDERS,
            sweep_epsilon_bp=float(args.sweep_epsilon_bp),
            bar_minutes=execution_minutes,
            show_progress=show_progress,
        )
        enriched_levels = enrich_level_sweeps_with_causal_quality(
            bars, levels, level_sweeps, show_progress=show_progress
        )
        episodes_q = aggregate_quality_to_sweep_episodes(episodes, enriched_levels)
        taxonomy_counts.append(liquidity_taxonomy_counts(episodes_q, execution_timeframe=label))
        sample_cols = [
            "level_id", "liquidity_side", "source_timeframe", "level_price", "pivot_time",
            "initial_available_time", "sweep_bar_time", "confirmed_order_at_sweep",
            "confirmed_order_available_time", "confirmed_order_age_minutes_at_sweep",
            "confirmed_prominence_bp_at_sweep", "pivot_range_bp", "pivot_rejection_fraction",
            "level_age_minutes_at_sweep", "activation_distance_bp", "pre_sweep_distance_bp",
            "max_excursion_away_bp_before_sweep", "cluster_count_5bp", "cluster_count_10bp",
            "cluster_count_25bp", "cluster_timeframe_count_10bp",
        ]
        qsample = enriched_levels[[c for c in sample_cols if c in enriched_levels]].copy()
        qsample.insert(0, "execution_timeframe", label)
        level_quality_samples.append(qsample.head(5000))

        print(f"[stage:{label}] native {label} MSS + displacement + FVG", flush=True)
        micro = build_micro_structure_context(bars, orders=MICRO_ORDERS)
        rolling_bars = max(20, int(round(60 / execution_minutes)))
        fvgs = build_displacement_fvgs(bars, rolling_window=rolling_bars, bar_minutes=execution_minutes)
        pairs = pair_sweeps_with_mss_fvgs(
            bars,
            episodes_q,
            fvgs,
            micro,
            max_search_bars=max(1, int(np.ceil(args.max_search_minutes / execution_minutes))),
            show_progress=show_progress,
        )
        print(f"[{label}] pairs={len(pairs):,}", flush=True)
        attached = attach_limit_entry_and_outcomes(
            bars,
            pairs,
            max_fill_wait_bars=max(1, int(np.ceil(args.max_fill_wait_minutes / execution_minutes))),
            outcome_horizon_bars=max(1, int(np.ceil(args.outcome_horizon_minutes / execution_minutes))),
            target_rs=TARGET_RS,
            round_trip_cost_pct=float(args.round_trip_cost_pct),
            show_progress=show_progress,
        )
        if not attached.empty:
            attached["execution_minutes"] = execution_minutes
            attached["execution_timeframe"] = label
            attached["sweep_to_displacement_minutes"] = attached["sweep_to_displacement_bars"] * execution_minutes
            attached["fill_wait_minutes"] = attached["fill_wait_bars"] * execution_minutes
            attached = add_calendar_session_context(
                attached, timestamp_col="sweep_bar_time", project_offset_hours=tz_offset
            )
            attached["mss_delay_bucket"] = pd.cut(
                attached["sweep_to_displacement_minutes"],
                [-np.inf, 5, 15, 30, 60, 120, np.inf],
                labels=["<=5m", "5-15m", "15-30m", "30-60m", "60-120m", "120m+"],
            ).astype("object")
            all_setups.append(attached)
        audits.append(_causal_audit(levels, enriched_levels, attached, execution_minutes))
        mechanism_rows.append({
            "execution_timeframe": label,
            "bars": len(bars),
            "level_sweeps": len(level_sweeps),
            "sweep_episodes": len(episodes_q),
            "directional_fvgs": len(fvgs),
            "mss_fvg_pairs": len(pairs),
            "filled_pairs": int(attached["first_fill_pos"].ge(0).sum()) if not attached.empty else 0,
        })

    setups = pd.concat(all_setups, ignore_index=True, sort=False) if all_setups else pd.DataFrame()
    base = _dedup_base_setups(setups, research_start=start, research_end=end)
    print(f"[stage] deduplicated core setups={len(base):,}", flush=True)

    # Full execution-timeframe comparison.
    exec_rows: list[dict[str, object]] = []
    if not base.empty:
        base["period"] = _period(base["first_fill_time"])
        for (emin, micro), part in base.groupby(["execution_minutes", "micro_order"], observed=False):
            for target_r in TARGET_RS:
                exec_rows.append({"execution_minutes": int(emin), "micro_order": int(micro), "target_r": target_r, "period": "FULL", **_metric_row(part, target_r)})
                for pname, pp in part.groupby("period", sort=False, observed=False):
                    exec_rows.append({"execution_minutes": int(emin), "micro_order": int(micro), "target_r": target_r, "period": pname, **_metric_row(pp, target_r)})

    liquidity_atlas_rows: list[dict[str, object]] = []
    session_atlas_rows: list[dict[str, object]] = []
    if not base.empty:
        liquidity_dims = {
            "max_timeframe_min": base["max_timeframe_min"].astype(str),
            "max_confirmed_order": base["max_confirmed_order"].astype(str),
            "liquidity_age_bucket": base["liquidity_age_bucket"],
            "confirmed_order_age_bucket": base["confirmed_order_age_bucket"],
            "liquidity_excursion_bucket": base["liquidity_excursion_bucket"],
            "cluster_10bp": pd.cut(base["max_cluster_count_10bp"], [-np.inf, 1, 2, 4, np.inf], labels=["1", "2", "3-4", "5+"]).astype("object"),
            "prominence_bp": pd.cut(base["max_confirmed_prominence_bp_q"], [-np.inf, 1, 3, 10, 30, np.inf], labels=["<1", "1-3", "3-10", "10-30", "30+"]).astype("object"),
        }
        for dim, vals in liquidity_dims.items():
            liquidity_atlas_rows.extend(_atlas_rows(base, dimension=dim, values=vals, target_r=2.0))

        session_dims = {
            "weekday_weekend": np.where(base["is_weekend_utc"], "WEEKEND", "WEEKDAY"),
            "utc_day_of_week": base["utc_day_of_week"],
            "ny_2h_bucket": (np.floor(base["new_york_hour"] / 2.0) * 2.0).astype(int).astype(str) + "-" + ((np.floor(base["new_york_hour"] / 2.0) * 2.0 + 2) % 24).astype(int).astype(str),
        }
        for flag in ("session_asia", "session_london", "session_new_york", "ict_london_kill_zone", "ict_new_york_kill_zone", "us_cash_open_90m"):
            session_dims[flag] = np.where(base[flag], "IN", "OUT")
        for dim, vals in session_dims.items():
            session_atlas_rows.extend(_atlas_rows(base, dimension=dim, values=pd.Series(vals, index=base.index), target_r=2.0))

    candidate_metrics, cost_stress, edge_gate = _candidate_tables(base, cost_pct=float(args.round_trip_cost_pct))
    audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    violations = int(audit["violations"].sum()) if not audit.empty else 0
    if violations:
        raise RuntimeError(f"Causal audit failed with {violations} violations\n{audit.to_string(index=False)}")

    # Cross-execution support is reported, not made a hard requirement.
    if not edge_gate.empty:
        peer_key = ["micro_order", "liquidity_rule", "time_rule", "target_r"]
        pass_map = edge_gate.set_index(peer_key + ["execution_minutes"])["edge_found"]
        peer_support = []
        for row in edge_gate.itertuples(index=False):
            other = 2 if int(row.execution_minutes) == 1 else 1
            key = (int(row.micro_order), str(row.liquidity_rule), str(row.time_rule), float(row.target_r), other)
            peer_support.append(bool(pass_map.get(key, False)))
        edge_gate["other_execution_tf_edge_support"] = peer_support

    _write_csv(pd.DataFrame(mechanism_rows), out_dir / "01_mechanism_counts.csv")
    _write_csv(audit, out_dir / "02_causal_audit.csv")
    _write_csv(pd.concat(taxonomy_counts, ignore_index=True) if taxonomy_counts else pd.DataFrame(), out_dir / "03_liquidity_taxonomy_counts.csv")
    _write_csv(pd.concat(level_quality_samples, ignore_index=True) if level_quality_samples else pd.DataFrame(), out_dir / "04_liquidity_quality_audit_sample.csv")
    _write_csv(pd.DataFrame(exec_rows), out_dir / "05_execution_1m_vs_2m.csv")
    _write_csv(pd.DataFrame(liquidity_atlas_rows), out_dir / "06_liquidity_outcome_atlas.csv")
    _write_csv(pd.DataFrame(session_atlas_rows), out_dir / "07_calendar_session_atlas.csv")
    _write_csv(candidate_metrics, out_dir / "08_candidate_period_metrics.csv")
    _write_csv(cost_stress, out_dir / "09_candidate_cost_stress.csv")
    _write_csv(edge_gate, out_dir / "10_edge_gate.csv")

    if not base.empty:
        replay_cols = [
            "execution_timeframe", "micro_order", "side", "sweep_bar_time", "sweep_available_time",
            "swept_timeframes", "max_timeframe_min", "max_confirmed_order", "oldest_level_age_minutes_q",
            "max_excursion_away_bp_before_sweep", "max_cluster_count_10bp", "max_cluster_timeframe_count_10bp",
            "liquidity_age_bucket", "liquidity_excursion_bucket", "lq_structural_major", "lq_major_remote",
            "lq_major_remote_stacked", "utc_day_of_week", "is_weekend_utc", "session_asia", "session_london",
            "session_new_york", "ict_london_kill_zone", "ict_new_york_kill_zone", "us_cash_open_90m",
            "micro_structure_pivot_time", "micro_structure_available_time", "fvg_completion_time", "fvg_available_time",
            "order_active_time", "first_fill_time", "entry_price", "stop_price", "risk_pct", "fill_wait_minutes",
            "sweep_to_displacement_minutes", "fvg_size_bp", "mfe_r_horizon", "mae_r_horizon",
        ]
        replay_cols += [c for c in base if c.startswith(("result_r", "gross_r_r", "net_r_r"))]
        _write_csv(base[[c for c in replay_cols if c in base]].sort_values("fvg_completion_time").head(10000), out_dir / "11_replay_audit_sample.csv")
        if args.write_full_setups:
            _write_csv(base, out_dir / "11b_full_base_setups.csv")

    passes = edge_gate.loc[edge_gate["edge_found"]] if not edge_gate.empty else pd.DataFrame()
    summary = [
        f"# {TITLE}", "",
        f"- Data: 1m bare OHLC only, loaded through `src.data_feed.OKXDataLoader`.",
        f"- Shared causal HTF levels: `{len(levels):,}`.",
        f"- Deduplicated core 1m/2m setups: `{len(base):,}`.",
        f"- Causal audit violations: `{violations}`.",
        f"- Candidate rules tested: `{len(_build_candidate_rules())}` x `{len(TARGET_RS)}` fixed-R targets.",
        f"- Strict edge-gate passes: `{len(passes)}`.", "",
        "## Liquidity definition discipline", "",
        "R02 does not assume every confirmed swing is real liquidity. It separately tests structural scale/order, age, pre-sweep excursion, equal-level clustering, multi-timeframe stacking and pivot prominence. All classifier inputs are observable before the sweep bar begins; sweep depth and post-sweep outcomes are excluded from liquidity quality.", "",
        "## Time/session discipline", "",
        "Weekday/weekend, Asia, London, New York, ICT London/NY kill zones and the 09:30-11:00 New York cash-open window are reported separately. London/New York clocks use real DST conversions from the project's configured candle timestamp offset.", "",
        "## Holdout note", "",
        "Because R01 aggregate 2026H1 results have already been observed, R02 labels 2026H1 as forward evidence rather than pretending it is a pristine sealed holdout. Candidate thresholds and session windows are fixed in code before R02 outcomes are inspected, and a pass must remain positive across 2023, 2024, 2025, 2026H1 plus 2x cost and top-10-winner removal.", "",
    ]
    if passes.empty:
        summary += ["## Decision", "", "No R02 candidate passed the strict gate. Continue research; do not promote a strategy merely because one exploratory slice looks good."]
    else:
        summary += ["## Decision", "", "At least one fixed R02 candidate passed all gates. Freeze those definitions before any further tuning and run a separate portfolio-style backtest with order overlap/capital constraints.", "", "Passing candidates:"]
        for r in passes.sort_values(["candidate_id", "target_r"]).itertuples(index=False):
            summary.append(f"- `{r.candidate_id}` @ `{r.target_r:g}R`")
    (out_dir / "12_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    manifest = {
        "script": SCRIPT_NAME,
        "version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "symbol": args.symbol,
        "data_source": "src.data_feed.okx_loader.OKXDataLoader",
        "input": "1m bare OHLC; 2m derived causally from complete 1m bars",
        "timeframes": {"liquidity": HTF_TIMEFRAMES, "execution_minutes": EXECUTION_TIMEFRAMES},
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "project_timezone": str(TIMEZONE),
        "rows": {"levels": len(levels), "base_setups": len(base)},
        "causal_audit_violations": violations,
        "edge_found_count": int(len(passes)),
        "notes": [
            "No future_max_eventual_order_label is used by the liquidity classifier.",
            "Liquidity excursion and clustering stop at sweep-bar open.",
            "2m bars are complete 1m aggregates; no separate data interface was added.",
            "2026H1 is forward evidence, not represented as pristine after R01 was observed.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pack = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE, print_log=True)
    print(f"[done] report_dir={out_dir}", flush=True)
    print(f"[done] edge_found={len(passes)}", flush=True)
    print(f"[done] review_pack={pack.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
