#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SOXL ICT R14: executable profitability freeze.

R14 is the first study in this line that deliberately stops behaving like an
atlas.  It converts R13's causal semantic candidates into one executable setup
per physical liquidity sweep, then enforces one pending/position lifecycle for
one account.

The goal is not to prove a final production strategy from already-inspected
history.  The goal is to answer whether the currently understood ICT mechanism
can produce a realistic, costed, non-overlapping strategy candidate worth
paper/live shadow testing.

Frozen R14 principles:
* 1m is primary research execution; 2m is a secondary comparison; 5m does not
  independently enter in R14.
* micro (<P50 causal visibility) structure is not executable.  Visible/strong
  structure is descriptive/causal, not PnL-ranked inside R14.
* Swing +/- $0.10 is NOT a gate.
* an opposite target does NOT need to be equal-like.  The narrow profit core
  includes both shallow/equal-like and partial-consumed external targets.
  Fresh/deep target management is deliberately deferred rather than forced into
  this first executable profitability freeze.
* one physical sweep -> at most one setup.  The earliest eligible 1m/2m signal
  wins; later MSS from the same sweep cannot create another trade.
* one account lifecycle at a time; overlapping later events are skipped.
* baseline round-trip cost 0.11%, plus cost/delay stress.

R14 may optionally reuse R13's 07/08/04/01 report files.  This is only a cache
of causal intermediates; it does not read R13 PnL tables or select a strategy by
R13 performance ranks.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.alpaca_stock_loader import AlpacaStockLoader
from src.data_feed.okx_loader import OKXDataLoader, TIMEZONE as OKX_LOADER_TIMEZONE
from src.research_common.ict.entry_expansion import (
    EntryExpansionConfig,
    build_intraday_15m_swing_catalog,
    build_intraday_15m_sweep_events,
)
from src.research_common.ict.executable_profitability import (
    DEFAULT_POLICIES,
    account_curve,
    annual_table,
    build_policy_attempts,
    contribution_table,
    monthly_table,
    opportunity_metrics,
    period_label,
    summarize_lifecycle,
)
from src.research_common.ict.premarket_mss_fvg import (
    NY_TZ,
    ReplayScenario,
    build_data_quality_table,
    eligible_ny_dates,
    enforce_single_lifecycle,
    ny_date_bounds_to_source_naive,
    replay_attempts,
    slice_ny_day,
    source_naive_to_new_york,
)
from src.research_common.ict.premarket_mss_fvg_v2 import SweepEpisodeConfig, build_all_premarket_levels_v2
from src.research_common.ict.semantic_consolidation import (
    LiquidityConsumptionConfig,
    attach_consumption_state_to_fvg_rows,
    build_liquidity_consumption_query_index,
)
from src.research_common.ict.spot_perp_overlap import build_equity_proxy_data_quality_table, densify_equity_minutes_causally
from src.research_common.ict.structure_entry_semantics import (
    StructureSemanticConfig,
    build_causal_sweep_events_for_levels,
    build_dual_session_liquidity_levels,
    build_r13_primary_break_fvg_compact,
    build_visible_swing_catalog,
)
from src.research_common.progress import ProgressReporter
from src.research_common.review_pack import finalize_research_report

DEFAULT_START_DATE = "2023-07-01"
DEFAULT_END_DATE = "2026-08-14"
DEFAULT_OUT_DIR = "data/reports/research/ict/soxl/mss/r14_executable_profitability_freeze"


def _source_offset_hours(text: str) -> int:
    try:
        return int(str(text).strip().upper().replace("UTC", ""))
    except ValueError:
        return 8


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SOXL ICT R14 executable profitability freeze")
    p.add_argument("--data-source", choices=("okx", "alpaca"), default="alpaca")
    p.add_argument("--symbol", default="SOXL-USDT-SWAP")
    p.add_argument("--alpaca-symbol", default="SOXL")
    p.add_argument("--alpaca-feed", default="sip")
    p.add_argument("--alpaca-adjustment", default="split")
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--r13-cache-dir", default="", help="optional R13 report dir containing 01/04/07/08 files")
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--include-us-equity-holidays", action="store_true")
    p.add_argument("--required-day-coverage", type=float, default=0.995)
    p.add_argument("--round-trip-cost", type=float, default=0.0011)
    p.add_argument("--risk-fraction", type=float, default=0.01)
    p.add_argument("--max-notional-multiple", type=float, default=2.0)
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--golden-date", default="2026-08-05")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--self-test", action="store_true")
    return p.parse_args(argv)


def _load_1m(args: argparse.Namespace) -> pd.DataFrame:
    if args.data_source == "alpaca":
        start_ny = pd.Timestamp(args.start_date).normalize().tz_localize(NY_TZ)
        end_ny = (pd.Timestamp(args.end_date).normalize() + pd.Timedelta(days=1)).tz_localize(NY_TZ)
        loader = AlpacaStockLoader(
            symbol=args.alpaca_symbol, timeframe="1Min", feed=args.alpaca_feed,
            adjustment=args.alpaca_adjustment, data_dir=args.data_dir,
        )
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
        raw = loader.load_local_data() if args.local_only else loader.fetch_data_by_date_range(start_src, end_src)
        if args.local_only and not raw.empty:
            raw = raw.loc[(raw.index >= start_src) & (raw.index <= end_src)].copy()
        if raw.empty:
            raise RuntimeError("OKX loader returned no data")
        bars = source_naive_to_new_york(raw, source_offset_hours=offset)
    mins = bars.index.hour * 60 + bars.index.minute
    bars = bars.loc[(mins >= 240) & (mins < 990)].copy()
    if args.data_source == "alpaca":
        bars = densify_equity_minutes_causally(bars)
    print(f"[load] source={args.data_source} rows={len(bars):,} NY={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path.name} rows={len(df):,}", flush=True)


def _load_r13_cache(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    if not str(args.r13_cache_dir).strip():
        return None
    root = Path(args.r13_cache_dir)
    required = {
        "quality": root / "01_data_quality.csv",
        "sweeps": root / "04_physical_sweep_events.csv",
        "primary": root / "07_r13_primary_mss_narratives.csv",
        "fvgs": root / "08_fvg_train_with_liquidity_state.csv",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"R13 cache missing: {missing}")
    manifest = root / "20_manifest.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        for key, expected in (("data_source", args.data_source), ("start_date", args.start_date), ("end_date", args.end_date)):
            actual = str(meta.get(key, ""))
            if actual and actual != str(expected):
                raise ValueError(f"R13 cache {key} mismatch: cache={actual} requested={expected}")
    print(f"[cache] reusing R13 causal intermediates from {root}", flush=True)
    quality = pd.read_csv(required["quality"], low_memory=False)
    sweeps = pd.read_csv(required["sweeps"], low_memory=False)
    primary = pd.read_csv(required["primary"], low_memory=False)
    fvgs = pd.read_csv(required["fvgs"], low_memory=False)
    for frame in (sweeps, primary, fvgs):
        for c in frame.columns:
            if c.endswith("_time") or c.endswith("_available_time") or c in {"signal_time", "break_available_time"}:
                # R13 CSV spans both EST and EDT, so mixed -05:00/-04:00
                # offsets must be normalised through UTC before returning to
                # New York time.  Plain pd.to_datetime on mixed offsets is
                # object-dtype today and becomes an error in newer pandas.
                try:
                    parsed = pd.to_datetime(frame[c], errors="coerce", utc=True)
                    frame[c] = parsed.dt.tz_convert(NY_TZ)
                except Exception:
                    pass
    return quality, sweeps, primary, fvgs


def _build_r13_intermediates(bars: pd.DataFrame, args: argparse.Namespace, days: list) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = StructureSemanticConfig(execution_timeframes=(1, 2), structure_lookback_minutes=150, absolute_entry_buffer=0.10)
    c_cfg = LiquidityConsumptionConfig()
    pm = build_all_premarket_levels_v2(bars, days, pivot_left=2, pivot_right=2, episode_config=SweepEpisodeConfig())
    major = pm.loc[pm["level_type"].eq("major_15m_swing")].copy() if not pm.empty else pd.DataFrame()
    if not major.empty:
        major["liquidity_family"] = "major_15m_swing"
    dual, _ = build_dual_session_liquidity_levels(bars, days, config=cfg)
    levels = pd.concat([x for x in (dual, major) if not x.empty], ignore_index=True, sort=False) if (not dual.empty or not major.empty) else pd.DataFrame()
    sweeps = build_causal_sweep_events_for_levels(bars, levels)
    intraday_cfg = EntryExpansionConfig(intraday_pivot_left=1, intraday_pivot_right=1)
    intraday_catalog = build_intraday_15m_swing_catalog(bars, days, pm, config=intraday_cfg)
    intraday_sweeps = build_intraday_15m_sweep_events(bars, intraday_catalog, config=intraday_cfg)
    if not intraday_sweeps.empty:
        intraday_sweeps = intraday_sweeps.copy(); intraday_sweeps["setup_eligible_at_sweep"] = True
    all_sweeps = pd.concat([x for x in (sweeps, intraday_sweeps) if not x.empty], ignore_index=True, sort=False) if (not sweeps.empty or not intraday_sweeps.empty) else pd.DataFrame()
    swing_catalog = build_visible_swing_catalog(bars, days, config=cfg)
    primary_parts: list[pd.DataFrame] = []; fvg_parts: list[pd.DataFrame] = []
    sweep_groups = {str(k): g for k, g in all_sweeps.groupby("ny_date", sort=True)} if not all_sweeps.empty else {}
    swing_groups = {str(k): g for k, g in swing_catalog.groupby("ny_date", sort=False)} if not swing_catalog.empty else {}
    prog = ProgressReporter(label="[r14-semantic] days", total=max(1, len(sweep_groups)), every=10, enabled=not args.no_progress)
    for day_text, day_sweeps in sweep_groups.items():
        day_swings = swing_groups.get(str(day_text), pd.DataFrame())
        if not day_swings.empty:
            p_day, f_day, _ = build_r13_primary_break_fvg_compact(bars, day_sweeps, day_swings, config=cfg)
            if not p_day.empty: primary_parts.append(p_day)
            if not f_day.empty: fvg_parts.append(f_day)
        prog.update(1)
    prog.close()
    primary = pd.concat(primary_parts, ignore_index=True, sort=False) if primary_parts else pd.DataFrame()
    fvgs = pd.concat(fvg_parts, ignore_index=True, sort=False) if fvg_parts else pd.DataFrame()
    all_levels = pd.concat([x for x in (levels, intraday_catalog) if not x.empty], ignore_index=True, sort=False) if (not levels.empty or not intraday_catalog.empty) else pd.DataFrame()
    state_index = build_liquidity_consumption_query_index(bars, all_levels, config=c_cfg)
    fvgs_state = attach_consumption_state_to_fvg_rows(fvgs, state_index=state_index, config=c_cfg)
    return all_sweeps, primary, fvgs_state


def _period_summary(lifecycle: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    q = lifecycle.copy()
    q["period_r14"] = [period_label(x) for x in q["signal_time"]]
    rows = []
    for (policy, scenario, period), g in q.groupby(["policy_id_r14", "scenario", "period_r14"], sort=True):
        s = summarize_lifecycle(g, initial_capital=initial_capital)
        rows.append({"policy_id_r14": policy, "scenario": scenario, "period_r14": period, **s})
    return pd.DataFrame(rows)


def _top10_removal(lifecycle: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    rows = []
    base = lifecycle.loc[lifecycle["scenario"].eq("base")].copy() if not lifecycle.empty else pd.DataFrame()
    for policy, g in base.groupby("policy_id_r14", sort=True):
        f = g.loc[g["filled"].fillna(False).astype(bool)].copy()
        if f.empty:
            continue
        ranked = f.assign(_net=pd.to_numeric(f["net_return"], errors="coerce")).sort_values("_net", ascending=False)
        for k in (0, 5, 10):
            kept = ranked.iloc[k:].drop(columns=["_net"]) if k else ranked.drop(columns=["_net"])
            s = summarize_lifecycle(kept, initial_capital=initial_capital)
            rows.append({"policy_id_r14": policy, "removed_top_winners": k, **s})
    return pd.DataFrame(rows)


def _frequency_table(attempts: pd.DataFrame, lifecycle: pd.DataFrame, days: list) -> pd.DataFrame:
    rows = []
    base_life = lifecycle.loc[lifecycle["scenario"].eq("base")].copy() if not lifecycle.empty else pd.DataFrame()
    for policy, a in attempts.groupby("policy_id_r14", sort=True):
        l = base_life.loc[base_life["policy_id_r14"].eq(policy)].copy() if not base_life.empty else pd.DataFrame()
        rows.append({"policy_id_r14": policy, **opportunity_metrics(a, l, days)})
    return pd.DataFrame(rows)


def run_research(bars: pd.DataFrame, args: argparse.Namespace) -> dict[str, Path]:
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    stages = ProgressReporter(label="[research] R14 stages", total=9, every=1, enabled=not args.no_progress)

    cache = _load_r13_cache(args)
    if cache is not None:
        quality, sweeps, primary, fvgs_state = cache
        valid = quality.loc[quality["coverage_pass"].fillna(False).astype(bool), "ny_date"].astype(str).tolist()
        days = [pd.Timestamp(x).date() for x in valid]
    else:
        days = eligible_ny_dates(bars, start_date=args.start_date, end_date=args.end_date, exclude_equity_holidays=not args.include_us_equity_holidays)
        quality = build_equity_proxy_data_quality_table(bars, days) if args.data_source == "alpaca" else build_data_quality_table(bars, days, required_coverage=float(args.required_day_coverage))
        valid = set(quality.loc[quality["coverage_pass"], "ny_date"].astype(str)); days = [pd.Timestamp(x).date() for x in sorted(valid)]
        if not days:
            raise RuntimeError("no valid sessions")
        sweeps, primary, fvgs_state = _build_r13_intermediates(bars, args, days)
    stages.update(1)

    attempts, rejections, fvg_one = build_policy_attempts(primary, fvgs_state, policies=DEFAULT_POLICIES)
    if attempts.empty:
        raise RuntimeError("R14 produced no executable setup attempts")
    attempts["sweep_to_signal_minutes"] = (
        pd.to_datetime(attempts["signal_time"]) - pd.to_datetime(attempts["sweep_time"])
    ).dt.total_seconds() / 60.0
    stages.update(2)

    # Cost-only stress does not change fill/exit path or account-overlap timing.
    # Replay each policy only for base/delay paths, then reprice the frozen base
    # lifecycle for 1.5x/2x cost.  This cuts two full intraday path scans per
    # policy without changing any trading semantics.
    path_scenarios = (
        ReplayScenario("base", 1.0, 0),
        ReplayScenario("delay_1m", 1.0, 1),
        ReplayScenario("delay_2m", 1.0, 2),
    )
    cost_clones = (("cost_1p5x", 1.5), ("cost_2x", 2.0))
    life_parts: list[pd.DataFrame] = []; summaries: list[dict[str, object]] = []; overlap_rows: list[dict[str, object]] = []
    total = len(DEFAULT_POLICIES) * len(path_scenarios)
    replay_prog = ProgressReporter(label="[r14] policy/path replay", total=total, every=1, enabled=not args.no_progress)

    def register(policy, scenario_name: str, kept: pd.DataFrame, skipped: int, replayed_setups: int) -> None:
        if not kept.empty:
            kept = kept.copy(); kept["policy_id_r14"] = policy.policy_id; kept["scenario"] = scenario_name
            life_parts.append(kept)
        s = summarize_lifecycle(kept, initial_capital=float(args.initial_capital))
        summaries.append({
            "policy_id_r14": policy.policy_id, "scenario": scenario_name,
            "target_router_r14": "profit_core_state_specific",
            "allowed_execution_tfs": "+".join(dict.fromkeys(leg.execution_tf for leg in policy.legs)),
            "skipped_account_overlap": int(skipped), **s,
        })
        overlap_rows.append({"policy_id_r14": policy.policy_id, "scenario": scenario_name, "replayed_setups": replayed_setups, "kept_lifecycles": len(kept), "skipped_account_overlap": int(skipped)})

    for policy in DEFAULT_POLICIES:
        a = attempts.loc[attempts["policy_id_r14"].eq(policy.policy_id)].copy()
        base_kept = pd.DataFrame(); base_skipped = 0
        for scenario in path_scenarios:
            replay = replay_attempts(
                bars, a, scenario=scenario, round_trip_cost=float(args.round_trip_cost),
                risk_fraction=float(args.risk_fraction), max_notional_multiple=float(args.max_notional_multiple),
            )
            kept, skipped = enforce_single_lifecycle(replay)
            register(policy, scenario.name, kept, skipped, len(replay))
            if scenario.name == "base":
                base_kept = kept.copy(); base_skipped = int(skipped)
            replay_prog.update(1)
        if not base_kept.empty:
            for scenario_name, multiple in cost_clones:
                clone = base_kept.copy()
                gross = pd.to_numeric(clone["gross_return"], errors="coerce")
                cost = float(args.round_trip_cost) * float(multiple)
                net = gross - cost
                entry = pd.to_numeric(clone["entry_price"], errors="coerce")
                stop = pd.to_numeric(clone["stop_price"], errors="coerce")
                is_long = clone["trade_side"].astype(str).eq("LONG")
                risk_abs = pd.Series(np.where(is_long, entry-stop, stop-entry), index=clone.index, dtype=float)
                risk_pct = risk_abs / entry
                clone["cost_multiple"] = float(multiple)
                clone["round_trip_cost"] = cost
                clone["net_return"] = net
                clone["net_r"] = net / risk_pct
                clone["account_return"] = net * pd.to_numeric(clone["notional_multiple"], errors="coerce")
                register(policy, scenario_name, clone, base_skipped, len(a))
        else:
            for scenario_name, _ in cost_clones:
                register(policy, scenario_name, pd.DataFrame(), base_skipped, len(a))
    replay_prog.close()
    lifecycle = pd.concat(life_parts, ignore_index=True, sort=False) if life_parts else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    overlap = pd.DataFrame(overlap_rows)
    stages.update(3)

    period = _period_summary(lifecycle, float(args.initial_capital))
    top10 = _top10_removal(lifecycle, float(args.initial_capital))
    freq = _frequency_table(attempts, lifecycle, days)
    stages.update(4)

    base_life = lifecycle.loc[lifecycle["scenario"].eq("base")].copy() if not lifecycle.empty else pd.DataFrame()
    monthly_parts = []; annual_parts = []; curve_parts = []
    for policy, g in base_life.groupby("policy_id_r14", sort=True):
        m = monthly_table(g); y = annual_table(g); c = account_curve(g, initial_capital=float(args.initial_capital))
        if not m.empty:
            if "policy_id_r14" not in m.columns: m.insert(0, "policy_id_r14", policy)
            monthly_parts.append(m)
        if not y.empty:
            if "policy_id_r14" not in y.columns: y.insert(0, "policy_id_r14", policy)
            annual_parts.append(y)
        if not c.empty:
            if "policy_id_r14" not in c.columns: c.insert(0, "policy_id_r14", policy)
            curve_parts.append(c)
    monthly = pd.concat(monthly_parts, ignore_index=True, sort=False) if monthly_parts else pd.DataFrame()
    annual = pd.concat(annual_parts, ignore_index=True, sort=False) if annual_parts else pd.DataFrame()
    curves = pd.concat(curve_parts, ignore_index=True, sort=False) if curve_parts else pd.DataFrame()
    stages.update(5)

    tf_contrib = contribution_table(base_life, ["policy_id_r14", "execution_tf"]) if not base_life.empty else pd.DataFrame()
    target_contrib = contribution_table(base_life, ["policy_id_r14", "target_liquidity_state", "target_model_r14"]) if not base_life.empty else pd.DataFrame()
    family_contrib = contribution_table(base_life, ["policy_id_r14", "liquidity_family"]) if not base_life.empty else pd.DataFrame()
    visibility_contrib = contribution_table(base_life, ["policy_id_r14", "structure_visibility_tier_r13"]) if not base_life.empty else pd.DataFrame()
    stages.update(6)

    funnel = []
    for policy in DEFAULT_POLICIES:
        pid = policy.policy_id
        a = attempts.loc[attempts["policy_id_r14"].eq(pid)]
        l = base_life.loc[base_life["policy_id_r14"].eq(pid)] if not base_life.empty else pd.DataFrame()
        funnel.append({
            "policy_id_r14": pid,
            "physical_sweeps_total": int(sweeps["event_id"].nunique()) if not sweeps.empty else 0,
            "selected_one_per_sweep": int(len(a)),
            "selected_unique_sweeps": int(a["event_id"].nunique()) if not a.empty else 0,
            "account_lifecycles_after_overlap": int(len(l)),
            "filled_trades": int(l["filled"].fillna(False).astype(bool).sum()) if not l.empty else 0,
        })
    funnel = pd.DataFrame(funnel)
    stages.update(7)

    golden = pd.DataFrame()
    for frame, name in ((sweeps, "physical_sweep"), (attempts, "selected_setup"), (base_life, "base_lifecycle")):
        if not frame.empty and "ny_date" in frame:
            g = frame.loc[frame["ny_date"].astype(str).eq(str(args.golden_date))].copy()
            if not g.empty:
                g.insert(0, "golden_record_type", name); golden = pd.concat([golden, g], ignore_index=True, sort=False)
    stages.update(8)

    design = f"""# R14 Executable Profitability Freeze\n\n- Source: {args.data_source}\n- Window: {args.start_date} -> {args.end_date}\n- R13 cache: {args.r13_cache_dir or 'not used'}\n- One physical liquidity sweep produces at most one executable setup per policy.\n- Earliest eligible signal wins; later MSS/re-entry from the same sweep is intentionally suppressed in R14.\n- Executable structure tiers: visible P50-P80 and strong >=P80. Micro pivots remain research context but cannot independently open R14 trades.\n- 1m and 2m are studied; 5m does not independently enter.\n- Swing +/- $0.10 is not a gate.\n- Target does not have to be equal-like: the core contains shallow/equal-like AND partial-consumed external targets.\n- Core break-middle policy: 1m shallow/equal-like -> break-middle near; 2m partial-consumed -> break-middle CE.\n- A second core policy swaps the 1m leg to first-train near while keeping the 2m partial CE leg.\n- Fresh and accepted/deep external targets are not declared invalid liquidity; they are deferred from this narrow profit-core freeze because their current full-target management is not yet stable.\n- No PnL rank is used to arbitrate simultaneous setups inside R14.\n- One account lifecycle at a time after setup selection.\n- Baseline fee is {args.round_trip_cost:.4%}; cost 1.5x/2x and 1m/2m order delay are replayed.\n- 2026 has already been inspected in earlier research; R14 is a candidate freeze/robustness study, not a claim of untouched holdout validation.\n"""
    (out / "00_research_design.md").write_text(design, encoding="utf-8")
    _write(quality, out / "01_data_quality.csv")
    _write(sweeps, out / "02_physical_sweep_events.csv")
    _write(fvg_one, out / "03_fixed_fvg_entry_catalog.csv")
    _write(attempts, out / "04_selected_one_setup_per_sweep.csv")
    _write(rejections, out / "05_setup_rejections.csv")
    _write(lifecycle, out / "06_account_lifecycle_all_stress.csv")
    _write(summary, out / "07_strategy_summary.csv")
    _write(period, out / "08_period_summary.csv")
    _write(monthly, out / "09_monthly_performance.csv")
    _write(annual, out / "10_annual_performance.csv")
    _write(curves, out / "11_base_account_curve.csv")
    _write(overlap, out / "12_account_overlap_audit.csv")
    _write(tf_contrib, out / "13_execution_tf_contribution.csv")
    _write(target_contrib, out / "14_target_state_contribution.csv")
    _write(family_contrib, out / "15_liquidity_family_contribution.csv")
    _write(visibility_contrib, out / "16_visibility_contribution.csv")
    _write(freq, out / "17_real_opportunity_frequency.csv")
    _write(top10, out / "18_top_winner_removal.csv")
    _write(funnel, out / "19_selection_funnel.csv")
    _write(golden, out / f"20_golden_replay_{args.golden_date}.csv")

    manifest = {
        "experiment_id": "SOXL_ICT_MSS_R14_EXECUTABLE_PROFITABILITY_FREEZE",
        "data_source": args.data_source, "start_date": args.start_date, "end_date": args.end_date,
        "valid_sessions": len(days), "physical_sweeps": int(sweeps["event_id"].nunique()) if not sweeps.empty else 0,
        "policies": [p.policy_id for p in DEFAULT_POLICIES],
        "round_trip_cost": float(args.round_trip_cost), "risk_fraction": float(args.risk_fraction),
        "max_notional_multiple": float(args.max_notional_multiple),
        "protocol": "one physical sweep -> one causal setup -> one account lifecycle; no $0.10 gate; target not required to be equal-like",
    }
    (out / "21_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    stages.update(9); stages.close()
    if not args.skip_review_pack:
        finalize_research_report(out, experiment_id=manifest["experiment_id"], edge_id="SOXL_ICT_SWEEP_MSS_EXECUTABLE_R14", title="SOXL ICT R14 Executable Profitability Freeze", print_log=True)
    return {"report_dir": out, "review_pack": out / "gpt_review_pack.zip"}


def _synthetic_r14_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    day = "2026-06-02"
    idx = pd.date_range(f"{day} 08:30", f"{day} 16:29", freq="1min", tz=NY_TZ)
    px = np.linspace(100.0, 108.0, len(idx))
    bars = pd.DataFrame({"open": px, "high": px + 0.2, "low": px - 0.2, "close": px + 0.05, "volume": 100.0}, index=idx)
    # Force a retracement fill then target for the selected long setup.
    bars.loc[pd.Timestamp(f"{day} 10:05", tz=NY_TZ), "low"] = 101.0
    bars.loc[pd.Timestamp(f"{day} 10:20", tz=NY_TZ), "high"] = 104.5
    common = {
        "ny_date": day, "event_id": "e1", "trade_side": "LONG", "liquidity_family": "major_15m_swing",
        "sweep_time": pd.Timestamp(f"{day} 09:50", tz=NY_TZ), "level_available_time": pd.Timestamp(f"{day} 08:30", tz=NY_TZ),
        "execution_tf": "1m", "execution_tf_minutes": 1, "terminal_version": 1,
        "terminal_extreme_price": 99.0, "mss_reference_time": pd.Timestamp(f"{day} 09:55", tz=NY_TZ),
        "mss_reference_price": 100.5, "mss_reference_available_time": pd.Timestamp(f"{day} 09:57", tz=NY_TZ),
        "break_available_time": pd.Timestamp(f"{day} 10:00", tz=NY_TZ), "break_close_cross": True,
        "structure_visibility_tier_r13": "visible_p50_p80", "causal_visibility_percentile": 0.7,
        "target_price": 106.0, "nearest_internal_target_price": 104.0,
        "target_liquidity_state": "shallow_probe_equal_like", "source_liquidity_state": "partial_consumed",
        "reference_model_r13": "outermost_barrier_tiered_primary",
    }
    primary = pd.DataFrame([common])
    fvg_common = {k: v for k, v in common.items() if k != "reference_model_r13"}
    fvg = pd.DataFrame([{**fvg_common,
        "fvg_available_time": pd.Timestamp(f"{day} 10:01", tz=NY_TZ), "signal_time": pd.Timestamp(f"{day} 10:01", tz=NY_TZ),
        "fvg_near_edge_entry": 101.0, "fvg_far_edge": 100.8, "fvg_train_sequence": 1,
        "fvg_third_pos": 1, "fvg_middle_relation_to_break": "break_bar_middle",
        "entry_distance_from_broken_swing": 0.5, "swing_buffer_cap_pass": False,
        "stop_price": 99.0,
    }])
    sweeps = pd.DataFrame([{**common, "level_price": 99.5}])
    return bars, sweeps, primary, fvg


def run_self_test(args: argparse.Namespace) -> int:
    import tempfile
    bars, sweeps, primary, fvgs = _synthetic_r14_frames()
    quality = pd.DataFrame([{"ny_date": "2026-06-02", "coverage_pass": True}])
    # Exercise the cache path because it is the recommended long-history R14 path.
    with tempfile.TemporaryDirectory(prefix="soxl_r14_") as tmp:
        cache = Path(tmp) / "r13"; cache.mkdir(parents=True)
        quality.to_csv(cache / "01_data_quality.csv", index=False)
        sweeps.to_csv(cache / "04_physical_sweep_events.csv", index=False)
        primary.to_csv(cache / "07_r13_primary_mss_narratives.csv", index=False)
        fvgs.to_csv(cache / "08_fvg_train_with_liquidity_state.csv", index=False)
        (cache / "20_manifest.json").write_text(json.dumps({"data_source":"alpaca","start_date":"2026-06-02","end_date":"2026-06-02"}), encoding="utf-8")
        args.data_source = "alpaca"; args.start_date = args.end_date = "2026-06-02"
        args.r13_cache_dir = str(cache); args.out_dir = str(Path(tmp) / "out")
        args.skip_review_pack = True; args.no_progress = True
        result = run_research(bars, args)
        summary = pd.read_csv(result["report_dir"] / "07_strategy_summary.csv")
        if summary.empty:
            raise AssertionError("R14 summary empty")
        freq = pd.read_csv(result["report_dir"] / "17_real_opportunity_frequency.csv")
        if not (freq["selected_setups"] <= 1).all():
            raise AssertionError("one-sweep-one-setup regression")
    print("[self-test] PASS", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test(args)
    return 0 if run_research(_load_1m(args), args) else 1


if __name__ == "__main__":
    raise SystemExit(main())
