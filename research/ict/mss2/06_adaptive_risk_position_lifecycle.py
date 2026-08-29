#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R06: ETH liquidity reversal adaptive risk + protected position lifecycle.

R06 is deliberately a *systemization* study, not another hard-filter search.
It freezes R05's broad N>=3 + (4H OR LT) Long episode-reclaim family and asks:

1. Can setup quality scale account risk instead of deleting lower-tier trades?
2. Can 5m/15m structural lows wait for later protection before moving the SL?
3. Can one risk-recycled add-on improve capital efficiency without raising the
   setup's worst-case risk budget?
4. Can a 5m protection phase slow to 15m only after the trade causally upgrades
   into a >=3% major move, preserving the long right tail?
5. Does the resulting single-ETH-position equity curve rise smoothly across
   years/months after realistic cost stress and winner-concentration tests?

There is no fixed TP and no time stop.  The data end is a right-edge censor.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.ict_mss2.r06 import (  # noqa: E402
    R06Config,
    attach_risk_sized_trade_returns,
    build_adaptive_base_universe,
    build_daily_mtm_equity,
    build_protected_structure_events,
    r06_causal_audit,
    select_single_position_trades,
    simulate_adaptive_trade_paths,
    summarize_portfolio,
)
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_VERSION = "6.0.0"
EXPERIMENT_ID = "ETH_ICT_MSS2_ADAPTIVE_RISK_POSITION_LIFECYCLE_R06"
EDGE_ID = "RESEARCH_ONLY_ETH_LIQUIDITY_REVERSAL_ENGINE"
TITLE = "ETH ICT MSS2 R06 Adaptive Risk + Protected Position Lifecycle"
DEFAULT_R05_DIR = "data/reports/research/ict/mss2/r05_entry_timing_structural_stop_runner_atlas"
DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r06_adaptive_risk_position_lifecycle"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=TITLE, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-08-15 23:59:59")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--r05-report-dir", default=DEFAULT_R05_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--market-roundtrip-cost", type=float, default=0.0011)
    p.add_argument("--stop-buffer-bps", type=float, default=2.0)
    p.add_argument("--max-notional-multiple", type=float, default=3.0)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)




def _safe_median(values) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(x.median()) if not x.empty else np.nan


def _safe_quantile(values, q: float) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(x.quantile(float(q))) if not x.empty else np.nan

def _months_between(start: str, end: str) -> float:
    a = pd.Timestamp(start)
    b = pd.Timestamp(end)
    return max(1.0 / 30.4375, float((b - a) / pd.Timedelta(days=30.4375)))


def _tier_summary(base: pd.DataFrame, months: float) -> pd.DataFrame:
    rows = []
    for key, part in base.groupby(["execution_minutes", "setup_tier"], dropna=False, sort=True):
        rows.append({
            "execution_minutes": int(key[0]),
            "setup_tier": str(key[1]),
            "opportunities": len(part),
            "episodes": int(part["episode_id"].nunique()),
            "opportunities_per_month": float(len(part) / months),
            "median_pool_count_at_entry": float(pd.to_numeric(part["ict_price_pools_cum"], errors="coerce").median()),
            "both_4h_lt_rate": float(pd.to_numeric(part["both_4h_lt_at_entry_flag"], errors="coerce").mean()),
            "median_initial_risk_pct": float(pd.to_numeric(part["initial_risk_return"], errors="coerce").median() * 100.0) if "initial_risk_return" in part else np.nan,
        })
    return pd.DataFrame(rows)


def _protected_event_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    e = events.copy()
    e["promotion_delay_minutes"] = (
        pd.to_datetime(e["promotion_time"], errors="coerce") - pd.to_datetime(e["candidate_activation_time"], errors="coerce")
    ) / pd.Timedelta(minutes=1)
    rows = []
    for key, part in e.groupby(["trail_tf_min", "event_type", "promotion_reason"], dropna=False, sort=True):
        d = pd.to_numeric(part["promotion_delay_minutes"], errors="coerce")
        rows.append({
            "trail_tf_min": int(key[0]),
            "event_type": str(key[1]),
            "promotion_reason": str(key[2]),
            "events": len(part),
            "median_promotion_delay_minutes": _safe_median(d),
            "p75_promotion_delay_minutes": _safe_quantile(d, 0.75),
            "p90_promotion_delay_minutes": _safe_quantile(d, 0.90),
        })
    return pd.DataFrame(rows)


def _path_summary(paths: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if paths.empty:
        return pd.DataFrame(), pd.DataFrame()
    p = paths.copy()
    p["year"] = pd.to_datetime(p["entry_time"], errors="coerce").dt.year
    keys = ["execution_minutes", "setup_tier", "management_variant"]
    def summ(part: pd.DataFrame) -> dict[str, object]:
        return {
            "opportunities": len(part),
            "resolved": int(pd.to_datetime(part["exit_time"], errors="coerce").notna().sum()),
            "right_edge_open_rate": float(pd.to_numeric(part["right_edge_open_flag"], errors="coerce").mean()),
            "median_initial_risk_pct": _safe_median(part["initial_risk_return"]) * 100.0,
            "median_trail_updates": _safe_median(part["trail_updates"]),
            "median_protected_updates": _safe_median(part["protected_updates"]),
            "median_first_promotion_minutes": _safe_median(part["first_promotion_minutes"]),
            "major_state_rate": float(pd.to_numeric(part["major_state_reached_flag"], errors="coerce").mean()),
            "median_holding_hours": _safe_median(part["holding_minutes"]) / 60.0,
            "median_mfe_pct": _safe_median(part["mfe_until_exit_or_data_end"]) * 100.0,
            "reached_3pct_rate": float(pd.to_numeric(part["reached_3pct_before_exit_flag"], errors="coerce").mean()),
            "reached_5pct_rate": float(pd.to_numeric(part["reached_5pct_before_exit_flag"], errors="coerce").mean()),
            "reached_10pct_rate": float(pd.to_numeric(part["reached_10pct_before_exit_flag"], errors="coerce").mean()),
            "addon_candidate_rate": float(pd.to_numeric(part["addon_pos_1m"], errors="coerce").ge(0).mean()),
        }
    overall, yearly = [], []
    for key, part in p.groupby(keys, dropna=False, sort=True):
        overall.append(dict(zip(keys, key)) | summ(part))
    for key, part in p.groupby(keys + ["year"], dropna=False, sort=True):
        yearly.append(dict(zip(keys + ["year"], key)) | summ(part))
    return pd.DataFrame(overall), pd.DataFrame(yearly)


def _risk_sizing_summary(sized: pd.DataFrame) -> pd.DataFrame:
    if sized.empty:
        return pd.DataFrame()
    rows = []
    keys = ["execution_minutes", "setup_tier", "management_variant", "addon_variant", "risk_schedule", "cost_scale"]
    for key, part in sized.groupby(keys, dropna=False, sort=True):
        x = pd.to_numeric(part["strategy_equity_return"], errors="coerce")
        rows.append(dict(zip(keys, key)) | {
            "candidate_trades": len(part),
            "resolved": int(x.notna().sum()),
            "mean_risk_budget_pct": float(pd.to_numeric(part["risk_budget_fraction"], errors="coerce").mean() * 100.0),
            "median_base_notional_x": _safe_median(part["base_notional_multiple"]),
            "addon_use_rate": float(pd.to_numeric(part["addon_used_flag"], errors="coerce").mean()),
            "median_addon_notional_x": _safe_median(part.loc[pd.to_numeric(part["addon_used_flag"], errors="coerce").eq(1), "addon_notional_multiple"]),
        })
    return pd.DataFrame(rows)




def _year_equity_summary(curve: pd.DataFrame, scenario: dict[str, object]) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    e = curve.copy()
    e["date"] = pd.to_datetime(e["date"], errors="coerce")
    e["year"] = e["date"].dt.year
    rows = []
    for year, part in e.groupby("year", sort=True):
        part = part.sort_values("date")
        eq = pd.to_numeric(part["equity"], errors="coerce").dropna()
        if eq.empty:
            continue
        start_eq = float(eq.iloc[0])
        end_eq = float(eq.iloc[-1])
        peak = eq.cummax()
        dd = eq / peak - 1.0
        me = part.set_index("date")["equity"].resample("ME").last().dropna()
        mr = me.pct_change().dropna()
        rows.append(scenario | {
            "year": int(year),
            "year_return": end_eq / start_eq - 1.0 if start_eq > 0 else np.nan,
            "year_max_drawdown": float(dd.min()),
            "positive_month_rate_in_year": float((mr > 0).mean()) if len(mr) else np.nan,
            "median_month_return_in_year": float(mr.median()) if len(mr) else np.nan,
            "year_end_equity": end_eq,
        })
    return pd.DataFrame(rows)

def _monthly_returns(curve: pd.DataFrame, scenario: dict[str, object]) -> pd.DataFrame:
    if curve.empty:
        return pd.DataFrame()
    x = curve.set_index("date")["equity"].resample("ME").last().pct_change().dropna().rename("monthly_return").reset_index()
    for k, v in scenario.items():
        x[k] = v
    return x


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = R06Config(
        market_roundtrip_cost=float(args.market_roundtrip_cost),
        stop_buffer_bps=float(args.stop_buffer_bps),
        max_notional_multiple=float(args.max_notional_multiple),
    ).validate()
    months = _months_between(args.start_date, args.end_date)
    r05_dir = Path(args.r05_report_dir)

    print("[r06] load frozen R05 causal opportunity + trailing data", flush=True)
    r05_opps = _read_csv(r05_dir / "03_entry_opportunities_causal.csv.gz")
    r05_events = _read_csv(r05_dir / "10_trailing_anchor_events_causal.csv.gz")
    base = build_adaptive_base_universe(r05_opps)
    if base.empty:
        raise RuntimeError("R06 built zero frozen N>=3 key-liquidity opportunities")
    base["initial_risk_return"] = (
        pd.to_numeric(base["entry_price"], errors="coerce") - pd.to_numeric(base["stop_episode_extreme"], errors="coerce")
    ) / pd.to_numeric(base["entry_price"], errors="coerce")

    print("[r06] load bare 1m K", flush=True)
    loader = OKXDataLoader(symbol=args.symbol, timeframe="1m", db_dir=args.data_dir)
    bars = loader.fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    if bars.empty:
        raise RuntimeError("R06 bare 1m K is empty")

    print("[r06] build delayed protected 5m/15m structural anchors", flush=True)
    protected = build_protected_structure_events(bars, r05_events, config=cfg)
    if protected.empty:
        raise RuntimeError("R06 built zero protected structure events")

    print("[r06] adaptive structural lifecycle + add-on candidates", flush=True)
    paths = simulate_adaptive_trade_paths(
        base, bars, r05_events, protected, config=cfg, show_progress=not args.no_progress
    )
    if paths.empty:
        raise RuntimeError("R06 adaptive path simulator returned zero rows")

    print("[r06] risk schedules + conservative risk-recycling add-on sizing", flush=True)
    sized = attach_risk_sized_trade_returns(paths, config=cfg)
    executed, overlap = select_single_position_trades(sized)
    if executed.empty:
        raise RuntimeError("R06 single-position allocator executed zero trades")

    print("[r06] capital/equity scorecards", flush=True)
    score_rows: list[dict[str, object]] = []
    curve_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    year_parts: list[pd.DataFrame] = []
    group_cols = ["execution_minutes", "management_variant", "addon_variant", "risk_schedule", "cost_scale"]
    for key, part in executed.groupby(group_cols, dropna=False, sort=True):
        scenario = dict(zip(group_cols, key))
        curve = build_daily_mtm_equity(part, bars, market_roundtrip_cost=cfg.market_roundtrip_cost)
        summary = summarize_portfolio(part, curve, months=months)
        if not summary:
            continue
        score_rows.append(scenario | summary)
        # Save all primary 2m curves; save 1m/5m only for 2x conservative
        # sensitivity so report size remains manageable.
        save_curve = int(scenario["execution_minutes"]) == cfg.primary_entry_minutes or (
            float(scenario["cost_scale"]) == 2.0 and str(scenario["risk_schedule"]) == "tiered_conservative"
        )
        if save_curve and not curve.empty:
            c = curve.copy()
            for k, v in scenario.items():
                c[k] = v
            curve_parts.append(c)
            monthly_parts.append(_monthly_returns(curve, scenario))
            year_parts.append(_year_equity_summary(curve, scenario))
    portfolio = pd.DataFrame(score_rows)
    curves = pd.concat(curve_parts, ignore_index=True, sort=False) if curve_parts else pd.DataFrame()
    monthly = pd.concat(monthly_parts, ignore_index=True, sort=False) if monthly_parts else pd.DataFrame()
    yearly_equity = pd.concat(year_parts, ignore_index=True, sort=False) if year_parts else pd.DataFrame()

    causal = r06_causal_audit(base, protected, paths, sized)
    violations = int(pd.to_numeric(causal["violations"], errors="coerce").fillna(0).sum())
    if violations:
        raise RuntimeError(f"R06 causal audit failed violations={violations}")

    tier_summary = _tier_summary(base, months)
    protected_summary = _protected_event_summary(protected)
    path_summary, path_year = _path_summary(paths)
    risk_summary = _risk_sizing_summary(sized)

    manifest = {
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "warmup_start_date": args.warmup_start_date,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "frozen_base_universe": "Long n3_4h_or_lt first causal stage from R05; lower tier trades are not deleted.",
        "risk_schedules": [list(x) for x in cfg.risk_schedules],
        "market_roundtrip_cost": cfg.market_roundtrip_cost,
        "cost_scales": list(cfg.cost_scales),
        "max_notional_multiple": cfg.max_notional_multiple,
        "major_upgrade_return": cfg.major_upgrade_return,
        "management_variants": sorted(paths["management_variant"].astype(str).unique().tolist()),
        "addon_variants": sorted(sized["addon_variant"].astype(str).unique().tolist()),
        "semantics": [
            "R06 does not add a new hard entry filter; quality changes risk budget, not whether the base N>=3 key-liquidity trade exists.",
            "A/A+ initial tier can only come from the first N>=3 causal stage itself already containing >=4 pools; a later N=4 stage never backfills earlier risk.",
            "ITL/LTL formation and stop promotion are separate. Protected structural lows require a later HTF close above a frozen, already-known confirmation high.",
            "1m is allowed for entry/path execution but never for trailing structure.",
            "One optional add-on is risk recycling after a protected 5m LTL higher-high confirmation; no averaging down.",
            "Risk-recycled add-on sizing keeps open risk to the common stop inside the configured setup risk budget and caps total notional.",
            "No fixed TP and no time stop. +3/+5/+10% are diagnostics/state milestones only; data end is right-edge censoring.",
            "Single-ETH-position portfolio semantics skip overlapping new base episodes while a position is open; an internal add-on belongs to the same setup.",
            "Equity quality is judged by daily mark-to-market MDD, positive month/quarter rates, rolling-90d positivity, log-equity R2, Ulcer index and top-winner removal, not PF alone.",
        ],
        "base_rows": int(len(base)),
        "base_episodes": int(base["episode_id"].nunique()),
        "protected_events": int(len(protected)),
        "adaptive_path_rows": int(len(paths)),
        "sized_rows": int(len(sized)),
        "executed_rows_all_scenarios": int(len(executed)),
        "causal_violations": violations,
    }
    (out / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([
        {"metric": "base_rows", "value": len(base)},
        {"metric": "base_episodes", "value": base["episode_id"].nunique()},
        {"metric": "protected_events", "value": len(protected)},
        {"metric": "adaptive_path_rows", "value": len(paths)},
        {"metric": "sized_rows", "value": len(sized)},
        {"metric": "portfolio_scenarios", "value": len(portfolio)},
        {"metric": "causal_violations", "value": violations},
    ]).to_csv(out / "01_engineering_audit.csv", index=False)
    (out / "02_frozen_design.json").write_text(json.dumps({
        "base": "n3_4h_or_lt, Long, first causal qualifying stage; 1m/2m/5m reclaim compared",
        "initial_risk_tiers": {
            "B": "first N>=3 key stage is N=3",
            "A": "first N>=3 key stage itself already jumps to >=4 pools",
            "A_plus": "same fast >=4 stage and both 4H + LT liquidity are already present",
        },
        "management": [
            "r05_immediate_ltl5 baseline",
            "protected 5m LTL only after later HH close",
            "protected 5m LTL OR 15m q95 bullish displacement+FVG",
            "protected 5m LTL, then after +3% use protected 15m ITL/LTL",
            "protected 5m LTL, then after +3% use protected 15m LTL only",
        ],
        "addon": "at most one risk-recycled add-on after protected 5m LTL+HH; common stop; no averaging down",
        "portfolio": "one ETH base position at a time; overlap is skipped and reported",
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    base.to_csv(out / "03_base_opportunities_causal.csv.gz", index=False, compression="gzip")
    tier_summary.to_csv(out / "04_setup_tier_frequency_summary.csv", index=False)
    protected.to_csv(out / "05_protected_structure_events_causal.csv.gz", index=False, compression="gzip")
    protected_summary.to_csv(out / "06_protected_structure_promotion_summary.csv", index=False)
    path_summary.to_csv(out / "07_adaptive_path_summary.csv", index=False)
    path_year.to_csv(out / "08_adaptive_path_year_summary.csv", index=False)
    paths.to_csv(out / "09_adaptive_path_trade_rows.csv.gz", index=False, compression="gzip")
    risk_summary.to_csv(out / "10_risk_sizing_summary.csv", index=False)
    overlap.to_csv(out / "11_single_position_overlap_audit.csv", index=False)
    portfolio.to_csv(out / "12_portfolio_equity_scorecard.csv", index=False)
    monthly.to_csv(out / "13_portfolio_monthly_returns.csv.gz", index=False, compression="gzip")
    yearly_equity.to_csv(out / "13b_portfolio_year_equity_summary.csv", index=False)
    curves.to_csv(out / "14_daily_mtm_equity_curves.csv.gz", index=False, compression="gzip")
    causal.to_csv(out / "15_causal_audit.csv", index=False)

    # Compact 2x-cost focus table: no winner is chosen automatically.
    focus = portfolio.loc[pd.to_numeric(portfolio.get("cost_scale"), errors="coerce").eq(2.0)].copy()
    if not focus.empty:
        focus = focus.sort_values(
            ["positive_month_rate", "max_drawdown_daily_mtm", "log_equity_r2"],
            ascending=[False, False, False], kind="stable"
        )
    focus.to_csv(out / "16_cost2x_equity_focus.csv", index=False)

    readme = "# R06 Adaptive Risk + Protected Position Lifecycle\n\n"
    readme += "R06 intentionally stops adding hard entry filters. It freezes the broad R05 `N>=3 + (4H OR LT)` Long family and studies whether account risk, stop promotion, one conservative add-on and a slower major-runner state can turn the edge into a smoother capital curve.\n\n"
    readme += "## Read this report in this order\n\n1. `04_setup_tier_frequency_summary.csv` — did quality-tiering preserve trade frequency?\n2. `06_protected_structure_promotion_summary.csv` — how long do ITL/LTL anchors wait before becoming protected?\n3. `07/08_adaptive_path_*` — do delayed promotions preserve 3/5/10% right tails across years?\n4. `11_single_position_overlap_audit.csv` — how many nominal signals can actually be executed on one ETH position?\n5. `12_portfolio_equity_scorecard.csv` — primary system result. Prioritize positive-month/quarter rate, daily MTM MDD, drawdown duration, rolling-90d positivity, log-equity R2, Ulcer index and no-top-winner robustness; PF alone is insufficient.\n\n"
    readme += "No scenario is auto-promoted. Risk schedules and management variants are a small frozen comparison set; do not select one solely because its full-sample return is largest.\n"
    (out / "README.md").write_text(readme, encoding="utf-8")

    finalize_research_report(out, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[r06] done -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
