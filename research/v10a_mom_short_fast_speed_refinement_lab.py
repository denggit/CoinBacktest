#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10A Momentum Short Fast-Speed Refinement Lab
============================================

Research-only lab for understanding and refining the V10A candidate:

    Current V10 + block MOMENTUM_V3 SHORT independent entries when range speed is FAST_Q4.

Why this lab exists
-------------------
The first V10A robustness run showed strong full-sample improvement, but 2024 year-reset
was slightly worse than V10 because one true crash/continuation Momentum Short was blocked
and later re-entered by BEAR_V3_ONLY with less profit. This script asks a narrower question:

    Can we keep the certainty benefit of blocking most fast-speed Momentum Shorts,
    while identifying broad, lookahead-safe exception/risk-down forms that avoid
    over-filtering true crash-breakdown events?

Lookahead safety
----------------
All rules use only:
    - completed signal-bar OHLCV;
    - completed signal-bar range/footprint aggregates;
    - shifted rolling past-only thresholds from `add_past_only_micro_features`.

No future return, MFE/MAE, actual trade PnL, or later BEAR signal is used by any router rule.
Future outcomes are only exported for diagnostics/event-study evaluation.

Outputs
-------
- v10a_fast_speed_refinement_score.csv
- v10a_fast_speed_refinement_summary.csv
- v10a_fast_speed_refinement_compare_to_v10.csv
- v10a_fast_speed_refinement_yearly.csv
- v10a_fast_speed_refinement_flags.csv
- v10a_fast_speed_refinement_condition_counts.csv
- v10a_fast_speed_refinement_event_study.csv
- optional per-scenario trades/equity for key periods with --write-trades
"""
from __future__ import annotations

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
from research import v10_microstructure_feature_discovery_lab as discovery  # noqa: E402
from research import v10_microstructure_router_variants_lab as micro_router  # noqa: E402
from research import v10_mom_short_fast_speed_robustness_lab as robustness  # noqa: E402

ENGINE_MOM = router.ENGINE_MOM
ENGINE_BEAR = router.ENGINE_BEAR
ENGINE_BULL = router.ENGINE_BULL

BASELINE = "baseline_v10_micro_filter"
PRIMARY = "v10a_mom_short_fast_speed_block"

SCENARIOS = [
    BASELINE,
    PRIMARY,
    "v10a_mom_short_fast_speed_risk_down_50",
    "v10a_mom_short_fast_speed_risk_down_35",
    "v10a_mom_short_fast_speed_risk_down_25",
    "v10a_fast_block_allow_strong_breakdown",
    "v10a_fast_block_allow_sell_imbalance_good_close",
    "v10a_fast_block_allow_close_below_prior_low",
    "v10a_fast_block_allow_high_volume",
    "v10a_fast_block_allow_bear_same_bar",
    "v10a_fast_block_allow_broad_breakdown",
]


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype("boolean").fillna(False).astype(bool)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.astype(float) / b.replace(0, np.nan).astype(float)


def _merge_blocks(base: dict[str, pd.Series], extra: dict[str, pd.Series]) -> dict[str, pd.Series]:
    return micro_router._merge_blocks(base, extra)  # intentionally reusing tested helper


def build_refinement_flags(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: Any) -> pd.DataFrame:
    """Build V10/V10A/refinement flags. Router masks are current-bar or past-only only."""
    args = micro_router.ensure_discovery_defaults(args)
    cond = micro_router.build_microstructure_flags(baseline, raw, args)

    base = discovery.add_past_only_micro_features(baseline, args)
    base = micro_router.add_engine_raw_signals(base, raw, args)

    idx = baseline.index
    out = cond.copy().reindex(idx)

    mom_short_fast = out["mom_short_fast_speed"].astype("boolean").fillna(False).astype(bool)
    bear_short_same_bar = base.get("bear_short", pd.Series(False, index=idx)).astype("boolean").fillna(False).astype(bool)

    close_pos = _num(base, "candle_close_pos")
    body_pct = _num(base, "candle_body_pct")
    lower_wick = _num(base, "lower_wick_pct")
    rf_imb = _num(base, "rf_imbalance")
    rf_close_pos = _num(base, "rf_close_pos")

    # Broad, pre-defined exception hypotheses. These are deliberately simple and not
    # fitted to one trade. They describe crash-breakdown quality at the signal close.
    out["fast_exception_strong_breakdown"] = (
        mom_short_fast
        & base["candle_body_dir"].astype(str).eq("DOWN")
        & body_pct.ge(0.45)
        & close_pos.le(0.35)
        & lower_wick.le(0.35)
    )
    out["fast_exception_sell_imbalance_good_close"] = (
        mom_short_fast
        & rf_imb.le(-0.05)
        & rf_close_pos.le(0.35)
    )
    out["fast_exception_close_below_prior_low"] = mom_short_fast & _bool(base, "close_below_prev_low_n")
    out["fast_exception_high_volume"] = mom_short_fast & _bool(base, "high_volume_past_q75")
    out["fast_exception_bear_same_bar"] = mom_short_fast & bear_short_same_bar
    out["fast_exception_broad_breakdown"] = (
        out["fast_exception_strong_breakdown"]
        | out["fast_exception_sell_imbalance_good_close"]
        | out["fast_exception_close_below_prior_low"]
    )

    # Audit columns.
    audit_cols = [
        "mom_signal", "bear_signal", "rf_speed_bin", "rf_bar_count", "rf_bar_count_ratio_past",
        "candle_close_pos", "candle_body_pct", "upper_wick_pct", "lower_wick_pct", "candle_body_dir",
        "volume_ratio_past", "volume_ratio_bin", "high_volume_past_q75",
        "break_prev_low_n", "close_below_prev_low_n", "failed_down_break_n",
        "rf_imbalance", "rf_close_pos", "rf_imbalance_bin", "rf_close_pos_bin",
        "mom_micro_action", "bear_short",
    ]
    for col in audit_cols:
        if col in base.columns and col not in out.columns:
            out[col] = base[col]
    return out


def _v10_base(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: Any, flags: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    return robustness._make_v10_base(baseline, raw, args, flags)


def _block_scenario(
    baseline: pd.DataFrame,
    raw: dict[str, pd.DataFrame],
    args: Any,
    flags: pd.DataFrame,
    *,
    name: str,
    block_mask: pd.Series,
    note: str,
    variant_type: str = "candidate",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    v10_block = {ENGINE_MOM: flags["v10_mom_long_not_aligned"]}
    block_map = _merge_blocks(v10_block, {ENGINE_MOM: block_mask})
    features = micro_router.make_features_for_block_map(baseline, raw, args, scenario=name, block_map=block_map)
    extra = {
        "variant_type": variant_type,
        "rule_note": note,
        "block_momentum_count": int(block_map[ENGINE_MOM].sum()),
        "block_bull_count": 0,
        "block_bear_count": 0,
        "block_total_count": int(block_map[ENGINE_MOM].sum()),
        "risk_down_engine": "NONE",
        "risk_down_scale": 1.0,
        "risk_down_bar_count": 0,
    }
    return features, extra


def _risk_down_fast_scenario(
    baseline: pd.DataFrame,
    raw: dict[str, pd.DataFrame],
    args: Any,
    flags: pd.DataFrame,
    *,
    name: str,
    scale: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    features, _ = _v10_base(baseline, raw, args, flags)
    features = features.copy()
    mask = flags["mom_short_fast_speed"].reindex(features.index).fillna(False).astype(bool)
    selected_mom_short = (
        features.get("selected_engine", pd.Series("", index=features.index)).astype(str).eq(ENGINE_MOM)
        & pd.to_numeric(features.get("signal", pd.Series(0, index=features.index)), errors="coerce").fillna(0).astype(int).eq(-1)
    )
    apply_mask = mask & selected_mom_short
    if "risk_mult" in features.columns and bool(apply_mask.any()):
        features.loc[apply_mask, "risk_mult"] = pd.to_numeric(features.loc[apply_mask, "risk_mult"], errors="coerce").fillna(1.0) * float(scale)
        if hasattr(args, "min_risk_mult"):
            features.loc[apply_mask, "risk_mult"] = pd.to_numeric(features.loc[apply_mask, "risk_mult"], errors="coerce").clip(float(args.min_risk_mult), float(args.max_risk_mult or 10.0))
    features["router_variant"] = name
    features["mom_short_fast_speed_risk_down"] = False
    features.loc[apply_mask, "mom_short_fast_speed_risk_down"] = True
    features.loc[apply_mask, "router_note"] = f"MOM_SHORT_FAST_SPEED_RISK_DOWN_{float(scale):.2f}"
    features.loc[apply_mask, "router_risk_adjustment"] = pd.to_numeric(features.get("router_risk_adjustment", pd.Series(1.0, index=features.index)), errors="coerce").fillna(1.0).loc[apply_mask] * float(scale)
    extra = {
        "variant_type": "risk_down_candidate",
        "rule_note": f"Current V10 + Momentum Short FAST_Q4 keeps entry but risk_mult is multiplied by {float(scale):.2f} instead of blocking.",
        "block_momentum_count": int(flags["v10_mom_long_not_aligned"].sum()),
        "block_bull_count": 0,
        "block_bear_count": 0,
        "block_total_count": int(flags["v10_mom_long_not_aligned"].sum()),
        "risk_down_engine": ENGINE_MOM,
        "risk_down_scale": float(scale),
        "risk_down_bar_count": int(apply_mask.sum()),
    }
    return features, extra


def build_scenarios(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: Any, flags: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    scenarios: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    scenarios[BASELINE] = _v10_base(baseline, raw, args, flags)

    fast = flags["mom_short_fast_speed"].astype("boolean").fillna(False).astype(bool)
    scenarios[PRIMARY] = _block_scenario(
        baseline, raw, args, flags,
        name=PRIMARY,
        block_mask=fast,
        note="Current V10 + block independent MOMENTUM_V3 SHORT entries when range speed is FAST_Q4.",
        variant_type="primary_candidate",
    )
    for scale in [0.50, 0.35, 0.25]:
        name = f"v10a_mom_short_fast_speed_risk_down_{int(scale * 100)}"
        scenarios[name] = _risk_down_fast_scenario(baseline, raw, args, flags, name=name, scale=scale)

    exception_specs = [
        ("v10a_fast_block_allow_strong_breakdown", flags["fast_exception_strong_breakdown"], "Block fast Momentum Short except when the signal bar is a broad strong down breakdown candle."),
        ("v10a_fast_block_allow_sell_imbalance_good_close", flags["fast_exception_sell_imbalance_good_close"], "Block fast Momentum Short except when range footprint has sell imbalance and closes near lows."),
        ("v10a_fast_block_allow_close_below_prior_low", flags["fast_exception_close_below_prior_low"], "Block fast Momentum Short except when the signal bar closes below the prior lookback low."),
        ("v10a_fast_block_allow_high_volume", flags["fast_exception_high_volume"], "Block fast Momentum Short except when current volume is above shifted rolling past Q75."),
        ("v10a_fast_block_allow_bear_same_bar", flags["fast_exception_bear_same_bar"], "Block fast Momentum Short except when BEAR_V3_ONLY also signals on the same completed bar."),
        ("v10a_fast_block_allow_broad_breakdown", flags["fast_exception_broad_breakdown"], "Block fast Momentum Short except broad crash-breakdown evidence: strong candle or sell-imbalance close or close-below-prior-low."),
    ]
    for name, exception_mask, note in exception_specs:
        block_mask = fast & (~exception_mask.reindex(flags.index).fillna(False).astype(bool))
        scenarios[name] = _block_scenario(
            baseline, raw, args, flags,
            name=name,
            block_mask=block_mask,
            note=note,
            variant_type="exception_candidate",
        )
    return {k: scenarios[k] for k in SCENARIOS if k in scenarios}


def build_condition_counts(flags: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "v10_mom_long_not_aligned",
        "mom_short_fast_speed",
        "fast_exception_strong_breakdown",
        "fast_exception_sell_imbalance_good_close",
        "fast_exception_close_below_prior_low",
        "fast_exception_high_volume",
        "fast_exception_bear_same_bar",
        "fast_exception_broad_breakdown",
    ]
    rows = []
    for col in cols:
        if col in flags.columns:
            rows.append({"condition": col, "count": int(flags[col].astype("boolean").fillna(False).sum())})
    return pd.DataFrame(rows)


def build_event_study(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: Any, flags: pd.DataFrame) -> pd.DataFrame:
    """Export independent signal-level diagnostics for fast Momentum Shorts.

    Future return columns are labels for analysis only and must not be used in rules.
    """
    events = discovery.build_signal_events(baseline, raw, args, horizons=[1, 3, 6, 12])
    if events.empty:
        return events
    m = events[events["engine"].eq(ENGINE_MOM) & events["side"].eq("SHORT")].copy()
    if m.empty:
        return m
    f = flags.reindex(pd.to_datetime(m["timestamp"]).values)
    f.index = m.index
    copy_cols = [
        "mom_short_fast_speed",
        "fast_exception_strong_breakdown",
        "fast_exception_sell_imbalance_good_close",
        "fast_exception_close_below_prior_low",
        "fast_exception_high_volume",
        "fast_exception_bear_same_bar",
        "fast_exception_broad_breakdown",
    ]
    for col in copy_cols:
        if col in f.columns:
            m[col] = f[col].astype("boolean").fillna(False).astype(bool)
    m["v10a_fast_blocked"] = m.get("mom_short_fast_speed", False)
    m["v10a_block_with_broad_exception"] = m.get("mom_short_fast_speed", False) & (~m.get("fast_exception_broad_breakdown", False))
    return m.sort_values(["timestamp"]).reset_index(drop=True)


def main() -> int:
    args = router.parse_args()
    args = micro_router.ensure_discovery_defaults(args)
    if str(args.out_dir).replace("\\", "/").endswith("v9e_engine_router_variants_lab"):
        args.out_dir = "data/reports/research/v10a_mom_short_fast_speed_refinement_lab"
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(PROJECT_ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline, raw, cfg, engine_cfgs = router.build_features(args)
    flags = build_refinement_flags(baseline, raw, args)
    flags.to_csv(out_dir / "v10a_fast_speed_refinement_flags.csv", encoding="utf-8-sig")
    build_condition_counts(flags).to_csv(out_dir / "v10a_fast_speed_refinement_condition_counts.csv", index=False, encoding="utf-8-sig")

    event_study = build_event_study(baseline, raw, args, flags)
    event_study.to_csv(out_dir / "v10a_fast_speed_refinement_event_study.csv", index=False, encoding="utf-8-sig")

    scenarios = build_scenarios(baseline, raw, args, flags)
    periods = robustness._periods_from_index(baseline.index)
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
                    f"Running V10A refinement: scenario={scenario} period={period['period_name']} "
                    f"fee_mult={cost['fee_mult']} slip_mult={cost['slip_mult']}",
                    flush=True,
                )
                meta_extra = dict(extra)
                meta_extra["cost_name"] = cost["cost_name"]
                summary, trades_df, trades, equity = robustness._run_one(
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
    summary_df.to_csv(out_dir / "v10a_fast_speed_refinement_summary.csv", index=False, encoding="utf-8-sig")

    delta_df = robustness._add_pairwise_deltas(summary_df)
    delta_df.to_csv(out_dir / "v10a_fast_speed_refinement_compare_to_v10.csv", index=False, encoding="utf-8-sig")

    # The robustness scorer was written for a fixed earlier scenario set.
    # Refinement labs have additional scenario names, so fall back to the
    # generic scorer when the imported scorer cannot build a complete table.
    try:
        score_df = robustness._score_candidates(summary_df, delta_df)
    except Exception as exc:  # scoring-only fallback; backtest outputs above remain valid
        print(f"Imported robustness scorer failed; using generic refinement scorer: {exc}", flush=True)
        score_df = pd.DataFrame()
    if score_df.empty or not set(SCENARIOS[1:]).issubset(set(score_df.get("scenario", pd.Series(dtype=str)).astype(str))):
        score_df = _generic_score(summary_df, delta_df)
    score_df.to_csv(out_dir / "v10a_fast_speed_refinement_score.csv", index=False, encoding="utf-8-sig")

    yearly_df = pd.concat(yearly_frames, ignore_index=True) if yearly_frames else pd.DataFrame()
    yearly_df.to_csv(out_dir / "v10a_fast_speed_refinement_yearly.csv", index=False, encoding="utf-8-sig")

    meta = {
        "script": "v10a_mom_short_fast_speed_refinement_lab.py",
        "baseline": BASELINE,
        "primary_candidate": PRIMARY,
        "scenarios": SCENARIOS,
        "rule_safety": "All router rules use completed signal-bar data and shifted rolling past-only thresholds only. Future returns are exported solely as labels in event_study.",
        "research_question": "Does hard-blocking fast Momentum Short remain best, or is risk-down / a broad crash-breakdown exception better without overfitting to one 2024 trade?",
        "args": vars(args),
        "output_dir": str(out_dir),
    }
    (out_dir / "v10a_fast_speed_refinement_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 96)
    print("V10A Momentum Short Fast-Speed Refinement Lab completed")
    print("=" * 96)
    print(f"Output directory: {out_dir.resolve()}")
    print("Key files:")
    print("  - v10a_fast_speed_refinement_score.csv")
    print("  - v10a_fast_speed_refinement_summary.csv")
    print("  - v10a_fast_speed_refinement_compare_to_v10.csv")
    print("  - v10a_fast_speed_refinement_yearly.csv")
    print("  - v10a_fast_speed_refinement_event_study.csv")
    print("=" * 96 + "\n")
    return 0


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _generic_score(summary_df: pd.DataFrame, delta_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario in [s for s in SCENARIOS if s != BASELINE]:
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
        if stress_count and stress_win_count < stress_count:
            warnings.append(f"not_better_all_stress:{stress_win_count}/{stress_count}")
        if np.isfinite(full_delta_dd) and full_delta_dd > 0:
            warnings.append("max_dd_worse_full_sample")
        pass_flag = (
            (np.isfinite(full_delta_cap_pct) and full_delta_cap_pct > 0)
            and (not np.isfinite(pre_delta_cap_pct) or pre_delta_cap_pct > 0)
            and (year_count == 0 or year_win_count == year_count)
            and (stress_count == 0 or stress_win_count == stress_count)
            and (not np.isfinite(full_delta_dd) or full_delta_dd <= 0)
        )
        rows.append({
            "scenario": scenario,
            "primary_candidate": scenario == PRIMARY,
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


if __name__ == "__main__":
    raise SystemExit(main())
