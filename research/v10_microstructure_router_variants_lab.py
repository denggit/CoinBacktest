#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10 Microstructure Router Variants Lab
======================================

Research-only chronological portfolio backtest lab for V10-style ETH LF portfolio
microstructure filters.

Purpose
-------
Test a small set of engine-specific, lookahead-safe microstructure block rules found by
`v10_microstructure_feature_discovery_lab.py`:

    - Momentum Long medium-volume fake-breakout / wick / weak-close candidates
    - Bull Reclaim upper-wick / failed-breakout candidates
    - Bear low-volume / weak-speed candidates
    - Momentum Short lower-wick candidate

Important safety design
-----------------------
1. All candidate block masks use only completed signal-bar data and shifted rolling
   past-only thresholds from `v10_microstructure_feature_discovery_lab.add_past_only_micro_features`.
2. No future return, MFE, MAE, or final-trade outcome is used in any router rule.
3. This script is a research probe, not a final strategy. Any promising candidate must
   still pass robustness / walk-forward / stress checks before being promoted.

Outputs
-------
- v10_microstructure_router_variant_summary.csv
- v10_microstructure_router_compare_to_baseline.csv
- v10_microstructure_router_variant_yearly.csv
- v10_microstructure_router_condition_counts.csv
- v10_microstructure_router_flags.csv
- optional per-variant trades/equity/audit with --write-trades
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

ENGINE_MOM = router.ENGINE_MOM
ENGINE_BEAR = router.ENGINE_BEAR
ENGINE_BULL = router.ENGINE_BULL
CURRENT_PRIORITY = (ENGINE_BULL, ENGINE_MOM, ENGINE_BEAR)  # V9E/V10 default reclaim_first order


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _bool(s: pd.Series | None, index: pd.Index) -> pd.Series:
    if s is None:
        return pd.Series(False, index=index)
    return s.reindex(index).astype("boolean").fillna(False).astype(bool)


def ensure_discovery_defaults(args: Any) -> Any:
    """Add discovery-lab args when this script is launched via router.parse_args()."""
    defaults = {
        "rolling_window_bars": 1080,        # ~180 days of 4H bars
        "volume_median_window": 120,
        "atr_window": 42,
        "ema_fast": 20,
        "ema_slow": 50,
        "prev_breakout_window": 20,
        "range_pct": getattr(args, "range_pct", 0.002),
    }
    for k, v in defaults.items():
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    return args


def _engine_micro_action(sig: pd.Series, baseline: pd.DataFrame, args: Any) -> pd.Series:
    """Same-side micro action, using only the completed signal bar footprint context."""
    sig = pd.to_numeric(sig, errors="coerce").fillna(0).astype(int)
    has_ctx = baseline.get("micro_context_available", pd.Series(False, index=baseline.index)).astype("boolean").fillna(False).astype(bool)
    imb = _num(baseline, "rf_imbalance")
    pos = _num(baseline, "rf_close_pos")
    aligned_imb = abs(float(getattr(args, "micro_aligned_imbalance", 0.05) or 0.05))
    contra_imb = abs(float(getattr(args, "micro_contra_imbalance", 0.05) or 0.05))
    good_pos = float(getattr(args, "micro_good_close_pos", 0.65) or 0.65)
    bad_pos = float(getattr(args, "micro_bad_close_pos", 0.35) or 0.35)

    long_sig = sig.eq(1)
    short_sig = sig.eq(-1)
    aligned = (
        (long_sig & has_ctx & imb.ge(aligned_imb) & pos.ge(good_pos))
        | (short_sig & has_ctx & imb.le(-aligned_imb) & pos.le(bad_pos))
    )
    contra = (
        (long_sig & has_ctx & imb.le(-contra_imb) & pos.le(bad_pos))
        | (short_sig & has_ctx & imb.ge(contra_imb) & pos.ge(good_pos))
    )
    action = pd.Series("NEUTRAL", index=baseline.index, dtype="object")
    action.loc[aligned] = "ALIGNED"
    action.loc[contra] = "CONTRA"
    action.loc[sig.ne(0) & has_ctx & (~aligned) & (~contra)] = "NOT_ALIGNED_RISK_REDUCED"
    action.loc[sig.eq(0)] = "NO_SIGNAL"
    return action


def add_engine_raw_signals(base: pd.DataFrame, raw: dict[str, pd.DataFrame], args: Any) -> pd.DataFrame:
    out = base.copy()
    for engine, prefix in [(ENGINE_MOM, "mom"), (ENGINE_BULL, "bull"), (ENGINE_BEAR, "bear")]:
        rdf = raw[engine].reindex(out.index)
        sig = _num(rdf, "signal", 0.0).fillna(0).astype(int)
        out[f"{prefix}_signal"] = sig
        out[f"{prefix}_long"] = sig.eq(1)
        out[f"{prefix}_short"] = sig.eq(-1)
        out[f"{prefix}_active"] = sig.ne(0)
        out[f"{prefix}_micro_action"] = _engine_micro_action(sig, out, args)
        out[f"{prefix}_not_aligned"] = out[f"{prefix}_active"] & out[f"{prefix}_micro_action"].eq("NOT_ALIGNED_RISK_REDUCED")
    return out


def build_microstructure_flags(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: Any) -> pd.DataFrame:
    """Build candidate block flags. All input features are signal-bar or past-only."""
    args = ensure_discovery_defaults(args)
    base = discovery.add_past_only_micro_features(baseline, args)
    base = add_engine_raw_signals(base, raw, args)

    idx = base.index
    cond = pd.DataFrame(index=idx)

    # Current formal V10 filter: no lookahead, already validated.
    cond["v10_mom_long_not_aligned"] = base["mom_long"] & base["mom_not_aligned"]

    # Momentum Long candidates from feature discovery.
    cond["mom_long_volume_high_1p0_1p5"] = base["mom_long"] & base["volume_ratio_bin"].astype(str).eq("HIGH_1P0_1P5")
    cond["mom_long_not_very_high_volume"] = base["mom_long"] & (~base["volume_ratio_bin"].astype(str).eq("VERY_HIGH_GT_1P5"))
    cond["mom_long_signal_wick_bad"] = base["mom_long"] & base["upper_wick_big"].astype("boolean").fillna(False).astype(bool)
    cond["mom_long_signal_close_weak"] = base["mom_long"] & _num(base, "candle_close_pos").le(0.50)
    cond["mom_long_wick_or_weak_close"] = cond["mom_long_signal_wick_bad"] | cond["mom_long_signal_close_weak"]
    cond["mom_long_failed_up_breakout"] = base["mom_long"] & base["failed_up_break_n"].astype("boolean").fillna(False).astype(bool)
    cond["mom_long_rf_no_buy_imbalance"] = base["mom_long"] & _num(base, "rf_imbalance").lt(0.05)
    cond["mom_long_rf_buy_absorption"] = base["mom_long"] & base["rf_buy_absorption"].astype("boolean").fillna(False).astype(bool)

    # Bull Reclaim candidates: do not use global micro filters; only side-specific candle structure.
    cond["bull_upper_wick_big"] = base["bull_long"] & base["upper_wick_big"].astype("boolean").fillna(False).astype(bool)
    cond["bull_failed_up_breakout"] = base["bull_long"] & base["failed_up_break_n"].astype("boolean").fillna(False).astype(bool)
    cond["bull_wick_or_failed_breakout"] = cond["bull_upper_wick_big"] | cond["bull_failed_up_breakout"]
    cond["bull_signal_close_weak"] = base["bull_long"] & _num(base, "candle_close_pos").le(0.50)

    # Bear candidates: low-volume / non-fast-breakdown ideas. These are weaker hypotheses.
    cond["bear_low_volume_past_q25"] = base["bear_short"] & base["low_volume_past_q25"].astype("boolean").fillna(False).astype(bool)
    cond["bear_not_high_volume_past_q75"] = base["bear_short"] & (~base["high_volume_past_q75"].astype("boolean").fillna(False).astype(bool))
    cond["bear_slow_range_speed"] = base["bear_short"] & base["rf_speed_bin"].astype(str).eq("SLOW_Q1")
    cond["bear_lower_wick_big"] = base["bear_short"] & base["lower_wick_big"].astype("boolean").fillna(False).astype(bool)

    # Momentum Short candidates.
    cond["mom_short_lower_wick_big"] = base["mom_short"] & base["lower_wick_big"].astype("boolean").fillna(False).astype(bool)
    cond["mom_short_fast_speed"] = base["mom_short"] & base["rf_speed_bin"].astype(str).eq("FAST_Q4")
    cond["mom_short_not_normal_speed"] = base["mom_short"] & (~base["rf_speed_bin"].astype(str).eq("NORMAL_Q2_Q3"))

    # Useful audit fields.
    for col in [
        "mom_signal", "bull_signal", "bear_signal", "volume_ratio_past", "volume_ratio_bin",
        "candle_close_pos", "upper_wick_pct", "lower_wick_pct", "rf_imbalance", "rf_close_pos",
        "rf_speed_bin", "low_volume_past_q25", "high_volume_past_q75",
    ]:
        if col in base.columns:
            cond[col] = base[col]
    return cond


def _assign_blocked_zero(df: pd.DataFrame, mask: pd.Series, cols: list[str]) -> None:
    mask = mask.reindex(df.index).fillna(False).astype(bool)
    if not bool(mask.any()):
        return
    for col in cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_bool_dtype(df[col].dtype):
            df.loc[mask, col] = False
        else:
            df.loc[mask, col] = 0


def _copy_raw_with_block_map(raw: dict[str, pd.DataFrame], block_map: dict[str, pd.Series]) -> dict[str, pd.DataFrame]:
    out = {k: v.copy() for k, v in raw.items()}
    for engine, mask in block_map.items():
        if engine not in out:
            continue
        _assign_blocked_zero(
            out[engine],
            mask,
            ["signal", "long_signal", "short_signal", "momentum_signal", "bull_signal", "bear_signal"],
        )
    return out


def make_features_for_block_map(
    baseline: pd.DataFrame,
    raw: dict[str, pd.DataFrame],
    args: Any,
    *,
    scenario: str,
    block_map: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    raw2 = raw if not block_map else _copy_raw_with_block_map(raw, block_map)
    features = router.make_engine_aware_router(
        baseline,
        raw2,
        args,
        scenario=scenario,
        momentum_entry_mode="all",
        priority_order=CURRENT_PRIORITY,
    )
    features["router_variant"] = scenario
    features["microstructure_blocked"] = False
    features["microstructure_block_note"] = "NONE"
    if block_map:
        any_block = pd.Series(False, index=features.index)
        notes: list[str] = []
        for engine, mask in block_map.items():
            m = mask.reindex(features.index).fillna(False).astype(bool)
            any_block = any_block | m
            if bool(m.any()):
                notes.append(f"{engine}:{int(m.sum())}")
        features.loc[any_block, "microstructure_blocked"] = True
        features.loc[any_block, "microstructure_block_note"] = ";".join(notes) if notes else "BLOCKED"
    return features


def _merge_blocks(*maps: dict[str, pd.Series]) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    for mp in maps:
        for engine, mask in mp.items():
            if engine in out:
                out[engine] = out[engine] | mask
            else:
                out[engine] = mask.copy()
    return out


def build_scenarios(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: Any, cond: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    scenarios: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}

    # V9E reference, included only for context.
    scenarios["baseline_v9e_reference"] = (baseline.copy(), {"variant_type": "baseline_reference", "rule_note": "Original V9E-style router before V10 micro filter."})

    v10_base = {ENGINE_MOM: cond["v10_mom_long_not_aligned"]}
    scenarios["baseline_v10_micro_filter"] = (
        make_features_for_block_map(baseline, raw, args, scenario="baseline_v10_micro_filter", block_map=v10_base),
        {
            "variant_type": "v10_baseline",
            "rule_note": "Current V10: block Momentum Long only when micro action is NOT_ALIGNED_RISK_REDUCED.",
            "block_momentum_count": int(v10_base[ENGINE_MOM].sum()),
            "block_bull_count": 0,
            "block_bear_count": 0,
            "block_total_count": int(v10_base[ENGINE_MOM].sum()),
        },
    )

    def mom(mask_col: str) -> dict[str, pd.Series]:
        return _merge_blocks(v10_base, {ENGINE_MOM: cond[mask_col]})

    def bull(mask_col: str) -> dict[str, pd.Series]:
        return _merge_blocks(v10_base, {ENGINE_BULL: cond[mask_col]})

    def bear(mask_col: str) -> dict[str, pd.Series]:
        return _merge_blocks(v10_base, {ENGINE_BEAR: cond[mask_col]})

    def mom_short(mask_col: str) -> dict[str, pd.Series]:
        return _merge_blocks(v10_base, {ENGINE_MOM: cond[mask_col]})

    specs: list[tuple[str, dict[str, pd.Series], str, str]] = [
        # Momentum Long candidates.
        ("v10_plus_mom_long_volume_high_1p0_1p5_block", mom("mom_long_volume_high_1p0_1p5"), "candidate", "Block V10 + Momentum Long medium-high but not extreme volume ratio bin HIGH_1P0_1P5."),
        ("v10_plus_mom_long_wick_or_weak_close_block", mom("mom_long_wick_or_weak_close"), "candidate", "Block V10 + Momentum Long with large upper wick or weak signal close position."),
        ("v10_plus_mom_long_upper_wick_block", mom("mom_long_signal_wick_bad"), "candidate", "Block V10 + Momentum Long with large upper wick."),
        ("v10_plus_mom_long_weak_close_block", mom("mom_long_signal_close_weak"), "candidate", "Block V10 + Momentum Long with weak signal-bar close position."),
        ("v10_plus_mom_long_failed_breakout_block", mom("mom_long_failed_up_breakout"), "candidate", "Block V10 + Momentum Long that broke prior high but failed to close above it."),
        ("v10_plus_mom_long_rf_no_buy_imbalance_block", mom("mom_long_rf_no_buy_imbalance"), "candidate", "Block V10 + Momentum Long without positive range-footprint buy imbalance."),
        ("v10_plus_mom_long_rf_buy_absorption_block", mom("mom_long_rf_buy_absorption"), "candidate", "Block V10 + Momentum Long with buy imbalance but weak range close, possible absorption."),
        # Bull candidates.
        ("v10_plus_bull_upper_wick_block", bull("bull_upper_wick_big"), "candidate", "Block V10 + Bull Reclaim with large upper wick on signal bar."),
        ("v10_plus_bull_failed_breakout_block", bull("bull_failed_up_breakout"), "candidate", "Block V10 + Bull Reclaim failed prior high breakout."),
        ("v10_plus_bull_wick_or_failed_breakout_block", bull("bull_wick_or_failed_breakout"), "combo_candidate", "Block V10 + Bull Reclaim upper wick or failed breakout."),
        ("v10_plus_bull_weak_close_block", bull("bull_signal_close_weak"), "candidate", "Block V10 + Bull Reclaim when signal bar closes weakly."),
        # Bear candidates.
        ("v10_plus_bear_low_volume_block", bear("bear_low_volume_past_q25"), "candidate", "Block V10 + Bear when 4H volume ratio is below shifted rolling past Q25."),
        ("v10_plus_bear_not_high_volume_block", bear("bear_not_high_volume_past_q75"), "aggressive_candidate", "Block V10 + Bear unless signal bar is high-volume past Q75. Aggressive hypothesis."),
        ("v10_plus_bear_slow_range_speed_block", bear("bear_slow_range_speed"), "candidate", "Block V10 + Bear when range-bar count is slow past Q25."),
        ("v10_plus_bear_lower_wick_block", bear("bear_lower_wick_big"), "candidate", "Block V10 + Bear with large lower wick."),
        # Momentum Short candidates.
        ("v10_plus_mom_short_lower_wick_block", mom_short("mom_short_lower_wick_big"), "candidate", "Block V10 + Momentum Short with large lower wick."),
        ("v10_plus_mom_short_fast_speed_block", mom_short("mom_short_fast_speed"), "candidate", "Block V10 + Momentum Short when range speed is FAST_Q4."),
        ("v10_plus_mom_short_not_normal_speed_block", mom_short("mom_short_not_normal_speed"), "aggressive_candidate", "Block V10 + Momentum Short unless range speed is NORMAL_Q2_Q3."),
    ]

    # Small combinations. These are deliberately labelled as combo/high-overfit-risk until robustness proves otherwise.
    combo_1 = _merge_blocks(v10_base, {ENGINE_MOM: cond["mom_long_volume_high_1p0_1p5"], ENGINE_BULL: cond["bull_upper_wick_big"]})
    specs.append(("v10_combo_mom_long_volume_plus_bull_upper_wick", combo_1, "combo_candidate", "V10 + Momentum Long medium-volume block + Bull upper-wick block."))
    combo_2 = _merge_blocks(v10_base, {ENGINE_MOM: cond["mom_long_volume_high_1p0_1p5"] | cond["mom_short_lower_wick_big"]})
    specs.append(("v10_combo_mom_long_volume_plus_mom_short_lower_wick", combo_2, "combo_candidate", "V10 + Momentum Long medium-volume block + Momentum Short lower-wick block."))
    combo_3 = _merge_blocks(v10_base, {ENGINE_MOM: cond["mom_long_volume_high_1p0_1p5"] | cond["mom_short_lower_wick_big"], ENGINE_BULL: cond["bull_upper_wick_big"]})
    specs.append(("v10_combo_mom_volume_bull_wick_momshort_wick", combo_3, "combo_high_overfit_risk", "Compact three-rule combo. Included for observation only; high overfit risk."))
    combo_4 = _merge_blocks(v10_base, {ENGINE_MOM: cond["mom_long_wick_or_weak_close"] | cond["mom_short_lower_wick_big"], ENGINE_BULL: cond["bull_wick_or_failed_breakout"], ENGINE_BEAR: cond["bear_low_volume_past_q25"]})
    specs.append(("v10_combo_all_discovered_blocks_high_overfit_risk", combo_4, "combo_high_overfit_risk", "All first-pass discovered blocks. Observation only; very high overfit risk."))

    for name, block_map, variant_type, note in specs:
        features = make_features_for_block_map(baseline, raw, args, scenario=name, block_map=block_map)
        extra = {
            "variant_type": variant_type,
            "rule_note": note,
            "block_momentum_count": int(block_map.get(ENGINE_MOM, pd.Series(False, index=cond.index)).sum()),
            "block_bull_count": int(block_map.get(ENGINE_BULL, pd.Series(False, index=cond.index)).sum()),
            "block_bear_count": int(block_map.get(ENGINE_BEAR, pd.Series(False, index=cond.index)).sum()),
            "block_total_count": int(sum(int(m.sum()) for m in block_map.values())),
        }
        scenarios[name] = (features, extra)
    return scenarios


def build_condition_counts(cond: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    condition_cols = [
        "v10_mom_long_not_aligned",
        "mom_long_volume_high_1p0_1p5",
        "mom_long_not_very_high_volume",
        "mom_long_signal_wick_bad",
        "mom_long_signal_close_weak",
        "mom_long_wick_or_weak_close",
        "mom_long_failed_up_breakout",
        "mom_long_rf_no_buy_imbalance",
        "mom_long_rf_buy_absorption",
        "bull_upper_wick_big",
        "bull_failed_up_breakout",
        "bull_wick_or_failed_breakout",
        "bull_signal_close_weak",
        "bear_low_volume_past_q25",
        "bear_not_high_volume_past_q75",
        "bear_slow_range_speed",
        "bear_lower_wick_big",
        "mom_short_lower_wick_big",
        "mom_short_fast_speed",
        "mom_short_not_normal_speed",
    ]
    for col in condition_cols:
        if col in cond.columns:
            rows.append({"condition": col, "count": int(cond[col].astype("boolean").fillna(False).sum())})
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


def main() -> int:
    args = router.parse_args()
    args = ensure_discovery_defaults(args)
    if str(args.out_dir).replace("\\", "/").endswith("v9e_engine_router_variants_lab"):
        args.out_dir = "data/reports/research/v10_microstructure_router_variants_lab"
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(PROJECT_ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline, raw, cfg, engine_cfgs = router.build_features(args)
    cond = build_microstructure_flags(baseline, raw, args)
    cond.to_csv(out_dir / "v10_microstructure_router_flags.csv", encoding="utf-8-sig")
    build_condition_counts(cond).to_csv(out_dir / "v10_microstructure_router_condition_counts.csv", index=False, encoding="utf-8-sig")

    scenarios = build_scenarios(baseline, raw, args, cond)
    summaries: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []

    for name, (features, extra) in scenarios.items():
        print(f"Running microstructure router variant: {name}", flush=True)
        summary, trades_df, trades, equity = router.run_variant(name, features, cfg, engine_cfgs, args, addon_mode="off", extra=extra)
        selected_engine = features.get("selected_engine", pd.Series("", index=features.index)).astype(str)
        sig = pd.to_numeric(features.get("signal", pd.Series(0, index=features.index)), errors="coerce").fillna(0).astype(int)
        summary["selected_momentum_count"] = int(selected_engine.eq(ENGINE_MOM).sum())
        summary["selected_momentum_long_count"] = int((selected_engine.eq(ENGINE_MOM) & sig.eq(1)).sum())
        summary["selected_momentum_short_count"] = int((selected_engine.eq(ENGINE_MOM) & sig.eq(-1)).sum())
        summary["selected_bull_count"] = int(selected_engine.eq(ENGINE_BULL).sum())
        summary["selected_bear_count"] = int(selected_engine.eq(ENGINE_BEAR).sum())
        summary["microstructure_blocked_bar_count"] = int(features.get("microstructure_blocked", pd.Series(False, index=features.index)).astype("boolean").fillna(False).sum())
        summaries.append(summary)
        yearly_frames.append(router.yearly_metrics(trades, equity, name, cfg.initial_capital))

        if args.write_trades:
            trades_df.to_csv(out_dir / f"{name}_trades.csv", index=False, encoding="utf-8-sig")
            if not equity.empty:
                equity.to_csv(out_dir / f"{name}_equity.csv", encoding="utf-8-sig")
            audit_cols = [
                "open", "high", "low", "close", "volume", "signal", "selected_engine",
                "momentum_signal", "bull_signal", "bear_signal", "risk_mult", "quality_mult",
                "micro_filter_action", "rf_imbalance", "rf_close_pos", "rf_bar_count",
                "microstructure_blocked", "microstructure_block_note", "router_note",
            ]
            features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{name}_signal_audit.csv", encoding="utf-8-sig")

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty and "closed_final_capital" in summary_df.columns:
        summary_df = summary_df.sort_values("closed_final_capital", ascending=False)
    summary_df.to_csv(out_dir / "v10_microstructure_router_variant_summary.csv", index=False, encoding="utf-8-sig")

    yearly_df = pd.concat([f for f in yearly_frames if not f.empty], ignore_index=True) if yearly_frames else pd.DataFrame()
    yearly_df.to_csv(out_dir / "v10_microstructure_router_variant_yearly.csv", index=False, encoding="utf-8-sig")

    if not summary_df.empty:
        # Compare to current V10 baseline, not to V9E reference.
        base = summary_df[summary_df["scenario"].eq("baseline_v10_micro_filter")]
        if not base.empty:
            b = base.iloc[0]
            comp = summary_df.copy()
            for col in [
                "closed_final_capital", "closed_profit_factor", "closed_win_rate",
                "max_drawdown_pct", "closed_expectancy_pct", "closed_total_trades",
                "selected_momentum_count", "selected_momentum_long_count", "selected_momentum_short_count",
                "selected_bull_count", "selected_bear_count", "force_close_count", "force_close_pnl",
            ]:
                if col in comp.columns and col in b.index:
                    comp[f"delta_{col}"] = pd.to_numeric(comp[col], errors="coerce") - float(b[col])
            comp.to_csv(out_dir / "v10_microstructure_router_compare_to_baseline.csv", index=False, encoding="utf-8-sig")

    meta = {
        "script": "v10_microstructure_router_variants_lab.py",
        "mode": "chronological_portfolio_backtest_with_engine_specific_microstructure_blocks",
        "baseline_for_comparison": "baseline_v10_micro_filter",
        "args": vars(args),
        "scenarios": list(scenarios.keys()),
        "no_lookahead_note": "Candidate masks use completed signal-bar features and shifted rolling past-only thresholds only. Future returns are not used here.",
        "overfit_note": "Combo scenarios are exploratory and must not be promoted without robustness/walk-forward validation.",
        "output_dir": str(out_dir),
    }
    (out_dir / "v10_microstructure_router_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 96)
    print("V10 Microstructure Router Variants Lab completed")
    print("=" * 96)
    print(f"Output directory: {out_dir.resolve()}")
    print("Key files:")
    print("  - v10_microstructure_router_variant_summary.csv")
    print("  - v10_microstructure_router_compare_to_baseline.csv")
    print("  - v10_microstructure_router_variant_yearly.csv")
    print("  - v10_microstructure_router_condition_counts.csv")
    print("=" * 96 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
