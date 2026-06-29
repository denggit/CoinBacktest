#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9E Momentum Conditional Router Lab
===================================

Research-only probe for ETH_LF_Portfolio_V9E_RangeExitOverlay.

Purpose:
    Test the safer idea: do NOT ban Momentum Long globally. Instead:
        - block or risk-down Momentum only in bad regimes
        - keep Momentum in good/neutral regimes
        - optionally risk-down weak Momentum Short regimes

This script does NOT modify V9E strategy logic and does NOT place orders.
It reuses the V9E closed-bar -> next-open timing through the router lab executor.
"""
from __future__ import annotations

import json
import math
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

# Reuse the tested V9E router research executor/helpers.
from research import v9e_engine_router_variants_lab as router  # noqa: E402

ENGINE_MOM = router.ENGINE_MOM
ENGINE_BEAR = router.ENGINE_BEAR
ENGINE_BULL = router.ENGINE_BULL
CURRENT_PRIORITY = (ENGINE_BULL, ENGINE_MOM, ENGINE_BEAR)  # reclaim_first-compatible priority used in prior labs.


def _qcut_label(s: pd.Series, q: int = 4, prefix: str = "Q") -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series("NA", index=s.index, dtype="object")
    valid = x.dropna()
    if valid.nunique() < 2 or len(valid) < q:
        return out
    try:
        binned = pd.qcut(valid, q=q, duplicates="drop")
    except ValueError:
        return out
    cats = binned.cat.categories
    label_map = {cat: f"{prefix}{i + 1}" for i, cat in enumerate(cats)}
    out.loc[valid.index] = binned.map(label_map).astype(str)
    return out


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _first_existing(df: pd.DataFrame, names: list[str], default: float = np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return _num(df, name, default)
    return pd.Series(default, index=df.index, dtype="float64")


def _bool_mask(s: pd.Series) -> pd.Series:
    return s.astype("boolean").fillna(False).astype(bool)


def build_momentum_regime_conditions(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build per-bar Momentum regime flags for conditional entry rules.

    Important:
        Conditions are computed from raw Momentum signal context, not from final portfolio-selected signal.
        This avoids banning all Momentum Long blindly and lets us test only specific weak environments.
    """
    mom = raw[ENGINE_MOM].reindex(baseline.index).copy()
    out = pd.DataFrame(index=baseline.index)
    out["momentum_signal"] = _num(mom, "signal", 0.0).fillna(0).astype(int)
    out["momentum_long"] = out["momentum_signal"].eq(1)
    out["momentum_short"] = out["momentum_signal"].eq(-1)

    # Raw Momentum factors.
    out["mom_adx"] = _first_existing(mom, ["adx"])
    out["mom_atr_pct"] = _first_existing(mom, ["atr_pct", "atr_pct_4h"])
    out["mom_quality_mult"] = _first_existing(mom, ["quality_mult"])
    out["mom_risk_mult"] = _first_existing(mom, ["risk_mult"])
    out["mom_volume_ratio"] = _first_existing(mom, ["volume_ratio", "vol_ratio", "volume_ratio_20", "volume_z"])

    # Range/footprint context copied from baseline after V9E micro context load.
    out["rf_imbalance"] = _num(baseline, "rf_imbalance")
    out["rf_close_pos"] = _num(baseline, "rf_close_pos")
    if "micro_context_available" in baseline.columns:
        out["micro_context_available"] = _bool_mask(baseline["micro_context_available"])
    else:
        out["micro_context_available"] = False

    # Recompute micro action for raw Momentum side. Default thresholds align with V9E defaults.
    imb = out["rf_imbalance"]
    pos = out["rf_close_pos"]
    has_ctx = _bool_mask(out["micro_context_available"])
    long_sig = out["momentum_long"]
    short_sig = out["momentum_short"]
    aligned = (
        (long_sig & has_ctx & (imb >= 0.05) & (pos >= 0.65))
        | (short_sig & has_ctx & (imb <= -0.05) & (pos <= 0.35))
    )
    contra = (
        (long_sig & has_ctx & (imb <= -0.05) & (pos <= 0.35))
        | (short_sig & has_ctx & (imb >= 0.05) & (pos >= 0.65))
    )
    action = pd.Series("NEUTRAL", index=out.index, dtype="object")
    action.loc[aligned] = "ALIGNED"
    action.loc[contra] = "CONTRA"
    action.loc[(out["momentum_signal"].ne(0)) & has_ctx & (~aligned) & (~contra)] = "NOT_ALIGNED_RISK_REDUCED"
    out["mom_micro_action"] = action
    out["mom_micro_aligned"] = aligned
    out["mom_micro_contra"] = contra

    # Quantile labels are based on all valid rows to match earlier regime diagnostics style.
    out["mom_adx_q"] = _qcut_label(out["mom_adx"], 4, "ADX_Q")
    out["mom_atr_q"] = _qcut_label(out["mom_atr_pct"], 4, "ATR_Q")
    out["mom_quality_q"] = _qcut_label(out["mom_quality_mult"], 4, "QUALITY_Q")
    out["mom_risk_q"] = _qcut_label(out["mom_risk_mult"], 4, "RISK_Q")
    out["mom_volume_q"] = _qcut_label(out["mom_volume_ratio"], 4, "VOL_Q")
    out["rf_imbalance_q"] = _qcut_label(out["rf_imbalance"], 4, "RFIMB_Q")
    out["rf_close_pos_q"] = _qcut_label(out["rf_close_pos"], 4, "RFCLOSE_Q")

    # Candidate bad/good regimes from the previous Momentum regime diagnostics.
    out["mom_long_bad_low_volume"] = out["momentum_long"] & out["mom_volume_q"].eq("VOL_Q1")
    out["mom_long_bad_not_aligned"] = out["momentum_long"] & out["mom_micro_action"].eq("NOT_ALIGNED_RISK_REDUCED")
    out["mom_long_bad_quality_q2"] = out["momentum_long"] & out["mom_quality_q"].eq("QUALITY_Q2")
    out["mom_long_bad_any"] = out["mom_long_bad_low_volume"] | out["mom_long_bad_not_aligned"] | out["mom_long_bad_quality_q2"]
    out["mom_long_bad_low_volume_or_not_aligned"] = out["mom_long_bad_low_volume"] | out["mom_long_bad_not_aligned"]

    # More permissive than "Long must be neutral only": allow non-bad Long, do not globally ban Long.
    out["mom_long_good_neutral_non_low_volume"] = out["momentum_long"] & out["mom_micro_action"].eq("NEUTRAL") & (~out["mom_volume_q"].eq("VOL_Q1"))
    out["mom_long_good_not_bad"] = out["momentum_long"] & (~out["mom_long_bad_any"])

    out["mom_short_good_adx_q2"] = out["momentum_short"] & out["mom_adx_q"].eq("ADX_Q2")
    out["mom_short_good_rfclose_q2"] = out["momentum_short"] & out["rf_close_pos_q"].eq("RFCLOSE_Q2")
    out["mom_short_good_rfimb_q2_rfclose_q2"] = out["momentum_short"] & out["rf_imbalance_q"].eq("RFIMB_Q2") & out["rf_close_pos_q"].eq("RFCLOSE_Q2")
    out["mom_short_good_any"] = out["mom_short_good_adx_q2"] | out["mom_short_good_rfclose_q2"] | out["mom_short_good_rfimb_q2_rfclose_q2"]
    out["mom_short_bad_rfclose_q3"] = out["momentum_short"] & out["rf_close_pos_q"].eq("RFCLOSE_Q3")
    out["mom_short_weak_not_good"] = out["momentum_short"] & (~out["mom_short_good_any"])
    return out


def _set_blocked_signal_value(df: pd.DataFrame, mask: pd.Series, col: str) -> None:
    """Disable a signal column without triggering pandas bool/int dtype upcast errors."""
    if col not in df.columns:
        return
    if pd.api.types.is_bool_dtype(df[col].dtype):
        df.loc[mask, col] = False
    else:
        df.loc[mask, col] = 0


def _copy_raw_with_momentum_block(raw: dict[str, pd.DataFrame], block_mask: pd.Series) -> dict[str, pd.DataFrame]:
    out = {k: v.copy() for k, v in raw.items()}
    mom = out[ENGINE_MOM]
    mask = block_mask.reindex(mom.index).fillna(False).astype(bool)
    for col in ["signal", "long_signal", "short_signal", "momentum_signal"]:
        _set_blocked_signal_value(mom, mask, col)
    return out


def make_conditional_features(
    baseline: pd.DataFrame,
    raw: dict[str, pd.DataFrame],
    args: Any,
    cond: pd.DataFrame,
    *,
    scenario: str,
    block_mask: pd.Series | None = None,
    risk_down_mask: pd.Series | None = None,
    risk_down_scale: float = 1.0,
    short_risk_down_mask: pd.Series | None = None,
    short_risk_down_scale: float = 1.0,
) -> pd.DataFrame:
    raw2 = raw if block_mask is None else _copy_raw_with_momentum_block(raw, block_mask)
    features = router.make_engine_aware_router(
        baseline,
        raw2,
        args,
        scenario=scenario,
        momentum_entry_mode="all",
        priority_order=CURRENT_PRIORITY,
    )

    features["router_variant"] = scenario
    features["momentum_conditional_blocked"] = False
    features["momentum_conditional_risk_down"] = False
    features["momentum_conditional_short_risk_down"] = False
    features["momentum_conditional_note"] = "NONE"

    if block_mask is not None:
        b = block_mask.reindex(features.index).fillna(False).astype(bool)
        features.loc[b, "momentum_conditional_blocked"] = True

    selected_mom = features["selected_engine"].astype(str).eq(ENGINE_MOM)
    if risk_down_mask is not None and float(risk_down_scale) != 1.0:
        m = selected_mom & risk_down_mask.reindex(features.index).fillna(False).astype(bool)
        if bool(m.any()):
            features.loc[m, "risk_mult"] = (router._num(features, "risk_mult", 1.0).loc[m] * float(risk_down_scale)).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
            features.loc[m, "router_risk_adjustment"] = router._num(features, "router_risk_adjustment", 1.0).loc[m] * float(risk_down_scale)
            features.loc[m, "router_note"] = f"MOMENTUM_CONDITIONAL_RISK_DOWN_{risk_down_scale:.2f}"
            features.loc[m, "momentum_conditional_risk_down"] = True
            features.loc[m, "momentum_conditional_note"] = f"RISK_DOWN_{risk_down_scale:.2f}"

    if short_risk_down_mask is not None and float(short_risk_down_scale) != 1.0:
        m = selected_mom & short_risk_down_mask.reindex(features.index).fillna(False).astype(bool)
        if bool(m.any()):
            features.loc[m, "risk_mult"] = (router._num(features, "risk_mult", 1.0).loc[m] * float(short_risk_down_scale)).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
            features.loc[m, "router_risk_adjustment"] = router._num(features, "router_risk_adjustment", 1.0).loc[m] * float(short_risk_down_scale)
            features.loc[m, "router_note"] = f"MOMENTUM_SHORT_WEAK_RISK_DOWN_{short_risk_down_scale:.2f}"
            features.loc[m, "momentum_conditional_short_risk_down"] = True
            features.loc[m, "momentum_conditional_note"] = f"SHORT_RISK_DOWN_{short_risk_down_scale:.2f}"
    return features


def build_scenarios(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: Any, cond: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    scenarios: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    scenarios["baseline_v9e"] = (baseline.copy(), {"variant_type": "baseline"})

    specs = [
        (
            "mom_long_bad_any_hard_block",
            cond["mom_long_bad_any"], None, 1.0, None, 1.0,
            "Block Momentum Long only when low volume OR not-aligned micro OR quality_q2. Momentum Short unchanged.",
        ),
        (
            "mom_long_bad_any_risk_down_50",
            None, cond["mom_long_bad_any"], 0.50, None, 1.0,
            "Risk down Momentum Long bad-any to 50%, do not block.",
        ),
        (
            "mom_long_bad_any_risk_down_35",
            None, cond["mom_long_bad_any"], 0.35, None, 1.0,
            "Risk down Momentum Long bad-any to 35%, do not block.",
        ),
        (
            "mom_long_bad_low_volume_or_not_aligned_hard_block",
            cond["mom_long_bad_low_volume_or_not_aligned"], None, 1.0, None, 1.0,
            "Block Momentum Long only when low volume OR not-aligned micro.",
        ),
        (
            "mom_long_bad_low_volume_or_not_aligned_risk_down_50",
            None, cond["mom_long_bad_low_volume_or_not_aligned"], 0.50, None, 1.0,
            "Risk down Momentum Long low-volume/not-aligned to 50%.",
        ),
        (
            "mom_long_not_aligned_hard_block",
            cond["mom_long_bad_not_aligned"], None, 1.0, None, 1.0,
            "Block only Momentum Long not-aligned micro.",
        ),
        (
            "mom_long_not_aligned_risk_down_50",
            None, cond["mom_long_bad_not_aligned"], 0.50, None, 1.0,
            "Risk down only Momentum Long not-aligned micro to 50%.",
        ),
        (
            "mom_long_low_volume_hard_block",
            cond["mom_long_bad_low_volume"], None, 1.0, None, 1.0,
            "Block only Momentum Long low volume.",
        ),
        (
            "mom_long_low_volume_risk_down_50",
            None, cond["mom_long_bad_low_volume"], 0.50, None, 1.0,
            "Risk down only Momentum Long low volume to 50%.",
        ),
        (
            "mom_long_quality_q2_risk_down_50",
            None, cond["mom_long_bad_quality_q2"], 0.50, None, 1.0,
            "Risk down only Momentum Long quality_q2 to 50%.",
        ),
        (
            "mom_long_bad_any_risk_down_50_short_weak_risk_down_50",
            None, cond["mom_long_bad_any"], 0.50, cond["mom_short_weak_not_good"], 0.50,
            "Risk down bad Momentum Long and weak-not-good Momentum Short to 50%.",
        ),
        (
            "mom_long_bad_any_hard_block_short_bad_rfclose_q3_risk_down_50",
            cond["mom_long_bad_any"], None, 1.0, cond["mom_short_bad_rfclose_q3"], 0.50,
            "Block bad Momentum Long; risk down only Short rf_close_pos_q3.",
        ),
        (
            "mom_long_bad_lowvol_notaligned_risk_down_50_short_weak_risk_down_50",
            None, cond["mom_long_bad_low_volume_or_not_aligned"], 0.50, cond["mom_short_weak_not_good"], 0.50,
            "Risk down Long low-volume/not-aligned and weak Short to 50%.",
        ),
    ]

    for name, block_mask, risk_mask, risk_scale, short_mask, short_scale, note in specs:
        features = make_conditional_features(
            baseline,
            raw,
            args,
            cond,
            scenario=name,
            block_mask=block_mask,
            risk_down_mask=risk_mask,
            risk_down_scale=risk_scale,
            short_risk_down_mask=short_mask,
            short_risk_down_scale=short_scale,
        )
        extra = {
            "variant_type": "momentum_conditional_entry",
            "rule_note": note,
            "block_condition_count": int(block_mask.sum()) if block_mask is not None else 0,
            "risk_down_condition_count": int(risk_mask.sum()) if risk_mask is not None else 0,
            "short_risk_down_condition_count": int(short_mask.sum()) if short_mask is not None else 0,
            "risk_down_scale": risk_scale,
            "short_risk_down_scale": short_scale,
        }
        scenarios[name] = (features, extra)
    return scenarios


def main() -> int:
    args = router.parse_args()
    # If caller did not override previous router default, use this lab's directory.
    if str(args.out_dir).replace("\\", "/").endswith("v9e_engine_router_variants_lab"):
        args.out_dir = "data/reports/research/v9e_momentum_conditional_router_lab"
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(PROJECT_ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline, raw, cfg, engine_cfgs = router.build_features(args)
    cond = build_momentum_regime_conditions(baseline, raw)
    cond.to_csv(out_dir / "v9e_momentum_conditional_regime_flags.csv")

    condition_counts = []
    for col in cond.columns:
        if col.startswith("mom_long_bad") or col.startswith("mom_long_good") or col.startswith("mom_short_good") or col.startswith("mom_short_bad") or col.startswith("mom_short_weak"):
            condition_counts.append({"condition": col, "count": int(cond[col].sum())})
    pd.DataFrame(condition_counts).sort_values("count", ascending=False).to_csv(out_dir / "v9e_momentum_condition_counts.csv", index=False)

    scenarios = build_scenarios(baseline, raw, args, cond)
    summaries: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []

    for name, (features, extra) in scenarios.items():
        print(f"Running conditional Momentum variant: {name}", flush=True)
        summary, trades_df, trades, equity = router.run_variant(name, features, cfg, engine_cfgs, args, addon_mode="off", extra=extra)
        selected_mom = features.get("selected_engine", pd.Series("", index=features.index)).astype(str).eq(ENGINE_MOM)
        summary["momentum_selected_count"] = int(selected_mom.sum())
        summary["momentum_selected_long_count"] = int((selected_mom & features.get("signal", pd.Series(0, index=features.index)).astype(int).eq(1)).sum())
        summary["momentum_selected_short_count"] = int((selected_mom & features.get("signal", pd.Series(0, index=features.index)).astype(int).eq(-1)).sum())
        summary["momentum_conditional_blocked_count"] = int(features.get("momentum_conditional_blocked", pd.Series(False, index=features.index)).astype("boolean").fillna(False).sum())
        summary["momentum_conditional_risk_down_selected_count"] = int(features.get("momentum_conditional_risk_down", pd.Series(False, index=features.index)).astype("boolean").fillna(False).sum())
        summary["momentum_conditional_short_risk_down_selected_count"] = int(features.get("momentum_conditional_short_risk_down", pd.Series(False, index=features.index)).astype("boolean").fillna(False).sum())
        summaries.append(summary)
        yearly_frames.append(router.yearly_metrics(trades, equity, name, cfg.initial_capital))
        if args.write_trades:
            trades_df.to_csv(out_dir / f"{name}_trades.csv", index=False)
            if not equity.empty:
                equity.to_csv(out_dir / f"{name}_equity.csv")
            audit_cols = [
                "open", "high", "low", "close", "atr", "atr_pct", "adx", "signal", "selected_engine",
                "momentum_signal", "bear_signal", "bull_signal", "risk_mult", "quality_mult", "micro_entry_risk_scale",
                "micro_filter_action", "rf_imbalance", "rf_close_pos", "momentum_conditional_blocked",
                "momentum_conditional_risk_down", "momentum_conditional_short_risk_down", "momentum_conditional_note", "router_note",
            ]
            features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{name}_signal_audit.csv")

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty and "closed_final_capital" in summary_df.columns:
        summary_df = summary_df.sort_values("closed_final_capital", ascending=False)
    summary_df.to_csv(out_dir / "v9e_momentum_conditional_variant_summary.csv", index=False)

    yearly_df = pd.concat([f for f in yearly_frames if not f.empty], ignore_index=True) if yearly_frames else pd.DataFrame()
    yearly_df.to_csv(out_dir / "v9e_momentum_conditional_variant_yearly.csv", index=False)

    if not summary_df.empty:
        base = summary_df[summary_df["scenario"].eq("baseline_v9e")]
        if not base.empty:
            b = base.iloc[0]
            comp = summary_df.copy()
            for col in ["closed_final_capital", "closed_profit_factor", "closed_win_rate", "max_drawdown_pct", "closed_expectancy_pct", "closed_total_trades"]:
                if col in comp.columns and col in b.index:
                    comp[f"delta_{col}"] = pd.to_numeric(comp[col], errors="coerce") - float(b[col])
            comp.to_csv(out_dir / "v9e_momentum_conditional_compare_to_baseline.csv", index=False)

    with (out_dir / "v9e_momentum_conditional_router_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "scenarios": list(scenarios.keys()), "output_dir": str(out_dir)}, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 92)
    print("V9E Momentum Conditional Router Lab completed")
    print("=" * 92)
    print(f"Output directory: {out_dir.resolve()}")
    print("Key files:")
    print("  - v9e_momentum_conditional_variant_summary.csv")
    print("  - v9e_momentum_conditional_compare_to_baseline.csv")
    print("  - v9e_momentum_conditional_variant_yearly.csv")
    print("  - v9e_momentum_condition_counts.csv")
    print("=" * 92 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
