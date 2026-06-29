#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10 Momentum Short Fast-Speed Robustness Lab
===========================================

Research-only robustness verification for the strongest V10 microstructure-router
candidate found by `v10_microstructure_router_variants_lab.py`:

    Current V10 baseline:
        Block independent MOMENTUM_V3 LONG entries when the same-side range-footprint
        micro action is NOT_ALIGNED_RISK_REDUCED.

    Candidate A:
        Current V10 + block independent MOMENTUM_V3 SHORT entries when range speed is FAST_Q4.

It also probes the second candidate family from the same discovery run:

    Bear high-volume dependency:
        - Hard block BEAR_V3_ONLY when it is NOT high-volume past Q75.
        - Risk-down BEAR_V3_ONLY when it is NOT high-volume past Q75, using 0.50 / 0.35 / 0.25 scales.

Purpose
-------
Answer whether Momentum Short FAST range-speed filtering is stable enough to promote
beyond research, and whether Bear high-volume should be handled by risk-down rather
than hard-block.

Lookahead safety
----------------
- `mom_short_fast_speed` is derived from `rf_speed_bin`, which is built by
  `v10_microstructure_feature_discovery_lab.add_past_only_micro_features` using
  completed signal-bar range context plus shifted rolling past-only thresholds.
- Bear high-volume is also past-only (`high_volume_past_q75` uses shifted rolling
  thresholds).
- Future returns / trade outcomes are never used in router decisions.

Outputs
-------
- v10_mom_short_fast_speed_robustness_score.csv
- v10_mom_short_fast_speed_robustness_summary.csv
- v10_mom_short_fast_speed_robustness_compare_to_baseline.csv
- v10_mom_short_fast_speed_robustness_yearly.csv
- v10_mom_short_fast_speed_robustness_condition_counts.csv
- v10_mom_short_fast_speed_robustness_flags.csv
- optional trades/equity for key periods with --write-trades
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from research import v9e_engine_router_variants_lab as router  # noqa: E402
from research import v10_microstructure_router_variants_lab as micro_router  # noqa: E402

ENGINE_MOM = router.ENGINE_MOM
ENGINE_BEAR = router.ENGINE_BEAR
ENGINE_BULL = router.ENGINE_BULL

BASELINE = "baseline_v10_micro_filter"
CANDIDATE = "v10_plus_mom_short_fast_speed_block"
PRIMARY_SCENARIOS = [
    BASELINE,
    CANDIDATE,
    "v10_plus_mom_short_lower_wick_block",
    "v10_plus_mom_short_fast_speed_and_lower_wick_block",
    "v10_plus_bear_not_high_volume_block",
    "v10_plus_bear_not_high_volume_risk_down_50",
    "v10_plus_bear_not_high_volume_risk_down_35",
    "v10_plus_bear_not_high_volume_risk_down_25",
]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _copy_with_cost(cfg: Any, fee_mult: float = 1.0, slip_mult: float = 1.0) -> Any:
    if cfg is None:
        return cfg
    fee = _safe_float(getattr(cfg, "fee_rate", np.nan), np.nan)
    slip = _safe_float(getattr(cfg, "slippage_pct", np.nan), np.nan)
    updates: dict[str, Any] = {}
    if np.isfinite(fee):
        updates["fee_rate"] = fee * float(fee_mult)
    if np.isfinite(slip):
        updates["slippage_pct"] = slip * float(slip_mult)
    if dataclasses.is_dataclass(cfg):
        return dataclasses.replace(cfg, **updates)
    out = copy.copy(cfg)
    for k, v in updates.items():
        try:
            setattr(out, k, v)
        except Exception:
            pass
    return out


def _scale_costs(cfg: Any, engine_cfgs: dict[str, Any], fee_mult: float, slip_mult: float) -> tuple[Any, dict[str, Any]]:
    cfg2 = _copy_with_cost(cfg, fee_mult=fee_mult, slip_mult=slip_mult)
    engine_cfgs2 = {k: _copy_with_cost(v, fee_mult=fee_mult, slip_mult=slip_mult) for k, v in engine_cfgs.items()}
    return cfg2, engine_cfgs2


def _closed_trade_frame(trades: list[dict[str, Any]]) -> pd.DataFrame:
    tdf = pd.DataFrame(trades).copy() if trades else pd.DataFrame()
    if tdf.empty:
        return tdf
    if "note" in tdf.columns:
        tdf = tdf.loc[~tdf["note"].astype(str).eq("FORCE_CLOSE_END")].copy()
    for col in ["pnl", "return_pct", "capital"]:
        if col in tdf.columns:
            tdf[col] = pd.to_numeric(tdf[col], errors="coerce")
    return tdf


def _top_dependency(trades: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    closed = _closed_trade_frame(trades)
    out: dict[str, Any] = {
        "closed_net_pnl": 0.0,
        "top1_closed_pnl": 0.0,
        "top3_closed_pnl": 0.0,
        "closed_final_capital_minus_top1_pnl": float(initial_capital),
        "closed_final_capital_minus_top3_pnl": float(initial_capital),
        "top1_pnl_share_of_net_pnl_pct": np.nan,
        "top3_pnl_share_of_net_pnl_pct": np.nan,
        "worst1_closed_pnl": 0.0,
    }
    if closed.empty or "pnl" not in closed.columns:
        return out
    pnl = closed["pnl"].fillna(0.0).astype(float)
    net = float(pnl.sum())
    winners = pnl[pnl > 0].sort_values(ascending=False)
    top1 = float(winners.iloc[0]) if len(winners) else 0.0
    top3 = float(winners.head(3).sum()) if len(winners) else 0.0
    final_cap = float(closed["capital"].dropna().iloc[-1]) if "capital" in closed.columns and closed["capital"].notna().any() else float(initial_capital + net)
    out.update({
        "closed_net_pnl": net,
        "top1_closed_pnl": top1,
        "top3_closed_pnl": top3,
        "closed_final_capital_minus_top1_pnl": final_cap - top1,
        "closed_final_capital_minus_top3_pnl": final_cap - top3,
        "top1_pnl_share_of_net_pnl_pct": (top1 / net * 100.0) if net > 0 else np.nan,
        "top3_pnl_share_of_net_pnl_pct": (top3 / net * 100.0) if net > 0 else np.nan,
        "worst1_closed_pnl": float(pnl.min()) if len(pnl) else 0.0,
    })
    return out


def _slice_features(features: pd.DataFrame, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    out = features.loc[(features.index >= s) & (features.index <= e)].copy()
    return out.sort_index()


def _periods_from_index(idx: pd.Index) -> list[dict[str, str]]:
    ts = pd.to_datetime(idx)
    start = ts.min().normalize()
    end = ts.max().normalize()
    periods: list[dict[str, str]] = [
        {"period_name": "full_sample", "start": str(start.date()), "end": str(end.date()), "test_type": "full"},
    ]
    pre_2026_end = pd.Timestamp("2025-12-31")
    if start <= pre_2026_end and end >= pre_2026_end:
        periods.append({"period_name": "pre_2026_only", "start": str(start.date()), "end": str(pre_2026_end.date()), "test_type": "holdout_sanity"})
    if start <= pd.Timestamp("2024-12-31") and end >= pd.Timestamp("2025-01-01"):
        periods.append({"period_name": "early_2023_2024", "start": str(start.date()), "end": "2024-12-31", "test_type": "early_window"})
        periods.append({"period_name": "forward_2025_2026", "start": "2025-01-01", "end": str(end.date()), "test_type": "forward_window"})
    if start <= pd.Timestamp("2025-12-31") and end >= pd.Timestamp("2026-01-01"):
        periods.append({"period_name": "trainlike_2023_2025", "start": str(start.date()), "end": "2025-12-31", "test_type": "trainlike_window"})
        periods.append({"period_name": "holdout_2026", "start": "2026-01-01", "end": str(end.date()), "test_type": "latest_holdout"})
    for year in sorted(pd.Series(ts.year).dropna().unique().tolist()):
        y_start = max(start, pd.Timestamp(f"{int(year)}-01-01"))
        y_end = min(end, pd.Timestamp(f"{int(year)}-12-31"))
        periods.append({"period_name": f"year_reset_{int(year)}", "start": str(y_start.date()), "end": str(y_end.date()), "test_type": "year_reset"})
    return periods


def _run_one(
    *,
    scenario: str,
    features: pd.DataFrame,
    cfg: Any,
    engine_cfgs: dict[str, Any],
    args: Any,
    period_name: str,
    start: str,
    end: str,
    test_type: str,
    fee_mult: float = 1.0,
    slip_mult: float = 1.0,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    feats = _slice_features(features, start, end)
    meta = {
        "period_name": period_name,
        "period_start": str(pd.Timestamp(start).date()),
        "period_end": str(pd.Timestamp(end).date()),
        "test_type": test_type,
        "fee_mult": float(fee_mult),
        "slippage_mult": float(slip_mult),
        "row_count": int(len(feats)),
    }
    if extra:
        meta.update(extra)
    if feats.empty:
        summary = {"scenario": scenario, **meta, "empty_period": True}
        return summary, pd.DataFrame(), [], pd.DataFrame()
    cfg2, engine_cfgs2 = _scale_costs(cfg, engine_cfgs, fee_mult, slip_mult)
    run_name = f"{scenario}__{period_name}__fee{fee_mult:g}_slip{slip_mult:g}"
    summary, trades_df, trades, equity = router.run_variant(run_name, feats, cfg2, engine_cfgs2, args, addon_mode="off", extra=meta)
    summary["run_name"] = summary.get("scenario", run_name)
    summary["scenario"] = scenario
    summary.update(_top_dependency(trades, getattr(cfg2, "initial_capital", args.initial_capital)))
    return summary, trades_df, trades, equity


def _make_v10_base(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: Any, cond: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    block = {ENGINE_MOM: cond["v10_mom_long_not_aligned"]}
    features = micro_router.make_features_for_block_map(
        baseline,
        raw,
        args,
        scenario=BASELINE,
        block_map=block,
    )
    extra = {
        "variant_type": "v10_baseline",
        "rule_note": "Current V10: block Momentum Long only when micro action is NOT_ALIGNED_RISK_REDUCED.",
        "block_momentum_count": int(block[ENGINE_MOM].sum()),
        "block_bull_count": 0,
        "block_bear_count": 0,
        "block_total_count": int(block[ENGINE_MOM].sum()),
        "risk_down_engine": "NONE",
        "risk_down_scale": 1.0,
        "risk_down_bar_count": 0,
    }
    return features, extra


def _block_scenario(
    baseline: pd.DataFrame,
    raw: dict[str, pd.DataFrame],
    args: Any,
    cond: pd.DataFrame,
    *,
    name: str,
    block_map: dict[str, pd.Series],
    note: str,
    variant_type: str = "candidate",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features = micro_router.make_features_for_block_map(
        baseline,
        raw,
        args,
        scenario=name,
        block_map=block_map,
    )
    extra = {
        "variant_type": variant_type,
        "rule_note": note,
        "block_momentum_count": int(block_map.get(ENGINE_MOM, pd.Series(False, index=cond.index)).sum()),
        "block_bull_count": int(block_map.get(ENGINE_BULL, pd.Series(False, index=cond.index)).sum()),
        "block_bear_count": int(block_map.get(ENGINE_BEAR, pd.Series(False, index=cond.index)).sum()),
        "block_total_count": int(sum(int(m.sum()) for m in block_map.values())),
        "risk_down_engine": "NONE",
        "risk_down_scale": 1.0,
        "risk_down_bar_count": 0,
    }
    return features, extra


def _risk_down_bear_scenario(
    baseline: pd.DataFrame,
    raw: dict[str, pd.DataFrame],
    args: Any,
    cond: pd.DataFrame,
    *,
    name: str,
    scale: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    v10_features, _ = _make_v10_base(baseline, raw, args, cond)
    features = v10_features.copy()
    mask = cond["bear_not_high_volume_past_q75"].reindex(features.index).fillna(False).astype(bool)
    selected_bear = features.get("selected_engine", pd.Series("", index=features.index)).astype(str).eq(ENGINE_BEAR)
    apply_mask = mask & selected_bear
    if "risk_mult" in features.columns and bool(apply_mask.any()):
        features.loc[apply_mask, "risk_mult"] = pd.to_numeric(features.loc[apply_mask, "risk_mult"], errors="coerce").fillna(1.0) * float(scale)
        if hasattr(args, "min_risk_mult"):
            features.loc[apply_mask, "risk_mult"] = pd.to_numeric(features.loc[apply_mask, "risk_mult"], errors="coerce").clip(float(args.min_risk_mult), float(args.max_risk_mult or 10.0))
    features["router_variant"] = name
    features["bear_risk_down"] = False
    features.loc[apply_mask, "bear_risk_down"] = True
    features.loc[apply_mask, "router_note"] = f"BEAR_NOT_HIGH_VOLUME_RISK_DOWN_{float(scale):.2f}"
    features.loc[apply_mask, "router_risk_adjustment"] = pd.to_numeric(features.get("router_risk_adjustment", pd.Series(1.0, index=features.index)), errors="coerce").fillna(1.0).loc[apply_mask] * float(scale)
    extra = {
        "variant_type": "risk_down_candidate",
        "rule_note": f"Current V10 + Bear entries that are not high-volume past Q75 keep entry but risk_mult is multiplied by {float(scale):.2f}.",
        "block_momentum_count": int(cond["v10_mom_long_not_aligned"].sum()),
        "block_bull_count": 0,
        "block_bear_count": 0,
        "block_total_count": int(cond["v10_mom_long_not_aligned"].sum()),
        "risk_down_engine": ENGINE_BEAR,
        "risk_down_scale": float(scale),
        "risk_down_bar_count": int(apply_mask.sum()),
    }
    return features, extra


def build_scenarios(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: Any, cond: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    scenarios: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    v10_block = {ENGINE_MOM: cond["v10_mom_long_not_aligned"]}
    scenarios[BASELINE] = _make_v10_base(baseline, raw, args, cond)

    scenarios["v10_plus_mom_short_fast_speed_block"] = _block_scenario(
        baseline, raw, args, cond,
        name="v10_plus_mom_short_fast_speed_block",
        block_map=micro_router._merge_blocks(v10_block, {ENGINE_MOM: cond["mom_short_fast_speed"]}),
        note="Current V10 + block Momentum Short entries when range speed is FAST_Q4.",
    )
    scenarios["v10_plus_mom_short_lower_wick_block"] = _block_scenario(
        baseline, raw, args, cond,
        name="v10_plus_mom_short_lower_wick_block",
        block_map=micro_router._merge_blocks(v10_block, {ENGINE_MOM: cond["mom_short_lower_wick_big"]}),
        note="Current V10 + block Momentum Short entries with large lower wick.",
    )
    scenarios["v10_plus_mom_short_fast_speed_and_lower_wick_block"] = _block_scenario(
        baseline, raw, args, cond,
        name="v10_plus_mom_short_fast_speed_and_lower_wick_block",
        block_map=micro_router._merge_blocks(v10_block, {ENGINE_MOM: cond["mom_short_fast_speed"] | cond["mom_short_lower_wick_big"]}),
        note="Current V10 + block Momentum Short FAST_Q4 range speed or large lower wick.",
        variant_type="combo_candidate",
    )
    scenarios["v10_plus_bear_not_high_volume_block"] = _block_scenario(
        baseline, raw, args, cond,
        name="v10_plus_bear_not_high_volume_block",
        block_map=micro_router._merge_blocks(v10_block, {ENGINE_BEAR: cond["bear_not_high_volume_past_q75"]}),
        note="Current V10 + hard block Bear unless high-volume past Q75.",
        variant_type="aggressive_candidate",
    )
    for scale in [0.50, 0.35, 0.25]:
        name = f"v10_plus_bear_not_high_volume_risk_down_{int(scale * 100)}"
        scenarios[name] = _risk_down_bear_scenario(baseline, raw, args, cond, name=name, scale=scale)
    return {k: scenarios[k] for k in PRIMARY_SCENARIOS if k in scenarios}


def _add_pairwise_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    keys = ["period_name", "fee_mult", "slippage_mult"]
    base_rows = summary_df[summary_df["scenario"].eq(BASELINE)].copy()
    for _, row in summary_df.iterrows():
        if row["scenario"] == BASELINE:
            continue
        mask = pd.Series(True, index=base_rows.index)
        for k in keys:
            mask &= base_rows[k].astype(str).eq(str(row[k]))
        if not bool(mask.any()):
            continue
        b = base_rows.loc[mask].iloc[0]
        out = {
            "scenario": row["scenario"],
            "baseline_scenario": BASELINE,
            "period_name": row["period_name"],
            "test_type": row.get("test_type", ""),
            "period_start": row.get("period_start", ""),
            "period_end": row.get("period_end", ""),
            "fee_mult": row.get("fee_mult", 1.0),
            "slippage_mult": row.get("slippage_mult", 1.0),
        }
        for col in [
            "closed_final_capital", "closed_profit_factor", "closed_win_rate", "max_drawdown_pct",
            "closed_expectancy_pct", "closed_total_trades", "closed_final_capital_minus_top1_pnl",
            "closed_final_capital_minus_top3_pnl", "force_close_count", "force_close_pnl",
        ]:
            if col in summary_df.columns and col in b.index:
                rv = _safe_float(row.get(col, np.nan), np.nan)
                bv = _safe_float(b.get(col, np.nan), np.nan)
                out[col] = rv
                out[f"baseline_{col}"] = bv
                out[f"delta_{col}"] = rv - bv if np.isfinite(rv) and np.isfinite(bv) else np.nan
                if col in ["closed_final_capital", "closed_final_capital_minus_top1_pnl", "closed_final_capital_minus_top3_pnl"] and np.isfinite(rv) and np.isfinite(bv) and abs(bv) > 1e-12:
                    out[f"delta_{col}_pct"] = (rv / bv - 1.0) * 100.0
        rows.append(out)
    return pd.DataFrame(rows)


def _score_candidates(summary_df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in [s for s in PRIMARY_SCENARIOS if s != BASELINE]:
        d = delta_df[delta_df["scenario"].eq(scenario)].copy()
        s = summary_df[summary_df["scenario"].eq(scenario)].copy()
        if d.empty:
            continue
        full = d[d["period_name"].eq("full_sample")]
        pre = d[d["period_name"].eq("pre_2026_only")]
        holdout = d[d["period_name"].eq("holdout_2026")]
        year = d[d["test_type"].eq("year_reset")]
        stress = d[(d["fee_mult"].gt(1.0)) | (d["slippage_mult"].gt(1.0))]
        year_s = s[s["test_type"].eq("year_reset")]

        def first_pct(frame: pd.DataFrame, col: str) -> float:
            return _safe_float(frame[col].iloc[0], np.nan) if not frame.empty and col in frame.columns else np.nan

        full_delta_cap_pct = first_pct(full, "delta_closed_final_capital_pct")
        pre_delta_cap_pct = first_pct(pre, "delta_closed_final_capital_pct")
        holdout_delta_cap_pct = first_pct(holdout, "delta_closed_final_capital_pct")
        full_delta_pf = first_pct(full, "delta_closed_profit_factor")
        full_delta_dd = first_pct(full, "delta_max_drawdown_pct")
        year_win_count = int((year["delta_closed_final_capital"].fillna(0.0) > 0).sum()) if "delta_closed_final_capital" in year.columns else 0
        year_count = int(len(year))
        positive_year_return_count = int((year_s["closed_total_return_pct"].fillna(0.0) > 0).sum()) if "closed_total_return_pct" in year_s.columns else 0
        stress_win_count = int((stress["delta_closed_final_capital"].fillna(0.0) > 0).sum()) if "delta_closed_final_capital" in stress.columns else 0
        stress_count = int(len(stress))

        warnings: list[str] = []
        if np.isfinite(full_delta_cap_pct) and full_delta_cap_pct <= 0:
            warnings.append("full_sample_not_better")
        if np.isfinite(pre_delta_cap_pct) and pre_delta_cap_pct <= 0:
            warnings.append("pre_2026_not_better")
        if year_count and year_win_count < year_count:
            warnings.append(f"not_better_all_years:{year_win_count}/{year_count}")
        if year_count and positive_year_return_count < year_count:
            warnings.append(f"not_positive_all_years:{positive_year_return_count}/{year_count}")
        if stress_count and stress_win_count < stress_count:
            warnings.append(f"not_better_all_stress:{stress_win_count}/{stress_count}")
        if np.isfinite(full_delta_dd) and full_delta_dd > 0:
            warnings.append("max_dd_worse_full_sample")
        if np.isfinite(holdout_delta_cap_pct) and np.isfinite(pre_delta_cap_pct) and holdout_delta_cap_pct > 0 and pre_delta_cap_pct <= 0:
            warnings.append("latest_year_only_warning")

        pass_flag = (
            (np.isfinite(full_delta_cap_pct) and full_delta_cap_pct > 0)
            and (not np.isfinite(pre_delta_cap_pct) or pre_delta_cap_pct > 0)
            and (year_count == 0 or year_win_count == year_count)
            and (year_count == 0 or positive_year_return_count == year_count)
            and (stress_count == 0 or stress_win_count == stress_count)
            and (not np.isfinite(full_delta_dd) or full_delta_dd <= 0)
        )
        rows.append({
            "scenario": scenario,
            "primary_candidate": scenario == CANDIDATE,
            "pass_flag": bool(pass_flag),
            "full_delta_closed_final_capital_pct": full_delta_cap_pct,
            "pre_2026_delta_closed_final_capital_pct": pre_delta_cap_pct,
            "holdout_2026_delta_closed_final_capital_pct": holdout_delta_cap_pct,
            "full_delta_profit_factor": full_delta_pf,
            "full_delta_max_drawdown_pct": full_delta_dd,
            "year_reset_better_count": year_win_count,
            "year_reset_count": year_count,
            "positive_year_return_count": positive_year_return_count,
            "stress_better_count": stress_win_count,
            "stress_count": stress_count,
            "warnings": ";".join(warnings) if warnings else "",
        })
    return pd.DataFrame(rows).sort_values(["pass_flag", "primary_candidate", "full_delta_closed_final_capital_pct"], ascending=[False, False, False])


def main() -> int:
    args = router.parse_args()
    args = micro_router.ensure_discovery_defaults(args)
    if str(args.out_dir).replace("\\", "/").endswith("v9e_engine_router_variants_lab"):
        args.out_dir = "data/reports/research/v10_mom_short_fast_speed_robustness_lab"
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(PROJECT_ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline, raw, cfg, engine_cfgs = router.build_features(args)
    cond = micro_router.build_microstructure_flags(baseline, raw, args)
    cond.to_csv(out_dir / "v10_mom_short_fast_speed_robustness_flags.csv", encoding="utf-8-sig")
    micro_router.build_condition_counts(cond).to_csv(out_dir / "v10_mom_short_fast_speed_robustness_condition_counts.csv", index=False, encoding="utf-8-sig")

    scenarios = build_scenarios(baseline, raw, args, cond)
    periods = _periods_from_index(baseline.index)
    cost_tests = [
        {"cost_name": "normal_cost", "fee_mult": 1.0, "slip_mult": 1.0, "stress_only_full": False},
        {"cost_name": "fee2x", "fee_mult": 2.0, "slip_mult": 1.0, "stress_only_full": True},
        {"cost_name": "slippage2x", "fee_mult": 1.0, "slip_mult": 2.0, "stress_only_full": True},
        {"cost_name": "fee2x_slippage2x", "fee_mult": 2.0, "slip_mult": 2.0, "stress_only_full": True},
    ]

    summaries: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []

    for cost in cost_tests:
        for period in periods:
            if cost["stress_only_full"] and period["period_name"] != "full_sample":
                continue
            for scenario, (features, extra) in scenarios.items():
                print(
                    f"Running robustness: scenario={scenario} period={period['period_name']} "
                    f"fee_mult={cost['fee_mult']} slip_mult={cost['slip_mult']}",
                    flush=True,
                )
                meta_extra = dict(extra)
                meta_extra["cost_name"] = cost["cost_name"]
                summary, trades_df, trades, equity = _run_one(
                    scenario=scenario,
                    features=features,
                    cfg=cfg,
                    engine_cfgs=engine_cfgs,
                    args=args,
                    period_name=period["period_name"],
                    start=period["start"],
                    end=period["end"],
                    test_type=period["test_type"],
                    fee_mult=cost["fee_mult"],
                    slip_mult=cost["slip_mult"],
                    extra=meta_extra,
                )
                summaries.append(summary)
                if equity is not None and not equity.empty:
                    y = router.yearly_metrics(trades, equity, f"{scenario}__{period['period_name']}__{cost['cost_name']}", getattr(cfg, "initial_capital", args.initial_capital))
                    if not y.empty:
                        y["scenario"] = scenario
                        y["period_name"] = period["period_name"]
                        y["cost_name"] = cost["cost_name"]
                        yearly_frames.append(y)
                if args.write_trades and period["period_name"] in {"full_sample", "pre_2026_only", "holdout_2026"} and cost["cost_name"] == "normal_cost":
                    safe = f"{scenario}__{period['period_name']}"
                    trades_df.to_csv(out_dir / f"{safe}_trades.csv", index=False, encoding="utf-8-sig")
                    if equity is not None and not equity.empty:
                        equity.to_csv(out_dir / f"{safe}_equity.csv", encoding="utf-8-sig")

    summary_df = pd.DataFrame(summaries)
    sort_cols = [c for c in ["period_name", "fee_mult", "slippage_mult", "scenario"] if c in summary_df.columns]
    summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else summary_df
    summary_df.to_csv(out_dir / "v10_mom_short_fast_speed_robustness_summary.csv", index=False, encoding="utf-8-sig")

    delta_df = _add_pairwise_deltas(summary_df)
    delta_df.to_csv(out_dir / "v10_mom_short_fast_speed_robustness_compare_to_baseline.csv", index=False, encoding="utf-8-sig")

    score_df = _score_candidates(summary_df, delta_df)
    score_df.to_csv(out_dir / "v10_mom_short_fast_speed_robustness_score.csv", index=False, encoding="utf-8-sig")

    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    yearly_df.to_csv(out_dir / "v10_mom_short_fast_speed_robustness_yearly.csv", index=False, encoding="utf-8-sig")

    meta = {
        "script": "v10_mom_short_fast_speed_robustness_lab.py",
        "baseline": BASELINE,
        "primary_candidate": CANDIDATE,
        "primary_scenarios": PRIMARY_SCENARIOS,
        "periods": periods,
        "cost_tests": cost_tests,
        "risk_down_explanation": "Bear risk-down keeps BEAR_V3_ONLY entries in not-high-volume regimes but scales risk_mult by 0.50/0.35/0.25 instead of blocking them.",
        "no_lookahead_note": "All candidate masks use completed signal-bar data and shifted rolling past-only thresholds from the microstructure feature discovery helper.",
        "pass_rule": "Candidate should beat current V10 on full sample, pre-2026, every year-reset period, and full-sample fee/slippage stress while not worsening full-sample max drawdown.",
        "args": vars(args),
        "output_dir": str(out_dir),
    }
    (out_dir / "v10_mom_short_fast_speed_robustness_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 96)
    print("V10 Momentum Short Fast-Speed Robustness Lab completed")
    print("=" * 96)
    print(f"Output directory: {out_dir.resolve()}")
    print("Key files:")
    print("  - v10_mom_short_fast_speed_robustness_score.csv")
    print("  - v10_mom_short_fast_speed_robustness_summary.csv")
    print("  - v10_mom_short_fast_speed_robustness_compare_to_baseline.csv")
    print("  - v10_mom_short_fast_speed_robustness_yearly.csv")
    print("  - v10_mom_short_fast_speed_robustness_condition_counts.csv")
    print("=" * 96 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
