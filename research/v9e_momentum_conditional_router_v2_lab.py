#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9E Momentum Conditional Router V2 Lab
======================================

Research-only probe for ETH_LF_Portfolio_V9E_RangeExitOverlay.

Purpose:
    Validate whether specific bad environments should block independent entries:
        - Momentum Long + micro NOT_ALIGNED_RISK_REDUCED
        - Momentum Long + low volume (VOL_Q1 among Momentum signal events)
        - Momentum Long + either of the above
    Also run deliberately broad/global filters as counterexamples:
        - block all engine signals in their own micro NOT_ALIGNED environment
        - block all engine signals in their own low-volume environment

This script does NOT modify V9E strategy logic and DOES NOT place orders.
It performs full chronological portfolio backtests through the existing V9E router lab executor.
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

ENGINE_MOM = router.ENGINE_MOM
ENGINE_BEAR = router.ENGINE_BEAR
ENGINE_BULL = router.ENGINE_BULL
ENGINES = (ENGINE_BULL, ENGINE_BEAR, ENGINE_MOM)
CURRENT_PRIORITY = (ENGINE_BULL, ENGINE_MOM, ENGINE_BEAR)  # match previous conditional-router lab


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _bool_mask(s: pd.Series) -> pd.Series:
    return s.astype("boolean").fillna(False).astype(bool)


def _first_num(df: pd.DataFrame, names: list[str], default: float = np.nan) -> pd.Series:
    for name in names:
        if name in df.columns:
            return _num(df, name, default)
    return pd.Series(default, index=df.index, dtype="float64")


def _qcut_label_on_mask(s: pd.Series, eligible: pd.Series, q: int = 4, prefix: str = "Q") -> pd.Series:
    """Quantile-label only eligible rows; all non-eligible rows remain NA.

    This intentionally mirrors the signal-event-study semantics: VOL_Q1 for Momentum
    means the bottom quartile *among Momentum signal events*, not among every 4H bar.
    """
    x = pd.to_numeric(s, errors="coerce")
    eligible = eligible.reindex(x.index).fillna(False).astype(bool)
    out = pd.Series("NA", index=x.index, dtype="object")
    valid = x.loc[eligible].dropna()
    if valid.nunique() < 2 or len(valid) < q:
        return out
    try:
        binned = pd.qcut(valid, q=q, duplicates="drop")
    except ValueError:
        return out
    label_map = {cat: f"{prefix}{i + 1}" for i, cat in enumerate(binned.cat.categories)}
    out.loc[valid.index] = binned.map(label_map).astype(str)
    return out


def _volume_ratio_for_engine(raw_engine: pd.DataFrame, baseline: pd.DataFrame) -> pd.Series:
    """Return a stable per-bar volume ratio.

    Previous v1 looked for precomputed volume_ratio-like columns, which are not always
    present in the raw Momentum frame. The event-study lab showed the reliable formula
    is volume / volume_median when available.
    """
    df = raw_engine.reindex(baseline.index)
    for name in ["volume_ratio", "vol_ratio", "volume_ratio_20"]:
        if name in df.columns:
            out = _num(df, name)
            if out.notna().sum() > 0:
                return out.replace([np.inf, -np.inf], np.nan)
    for source in [df, baseline]:
        if "volume" in source.columns and "volume_median" in source.columns:
            vol = _num(source, "volume")
            med = _num(source, "volume_median")
            out = vol / med.replace(0, np.nan)
            return out.replace([np.inf, -np.inf], np.nan)
    # Fallback: rolling median from baseline volume. This should rarely be needed.
    if "volume" in baseline.columns:
        vol = _num(baseline, "volume")
        med = vol.rolling(100, min_periods=20).median()
        return (vol / med.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return pd.Series(np.nan, index=baseline.index, dtype="float64")


def _engine_micro_action(sig: pd.Series, baseline: pd.DataFrame, args: Any) -> pd.Series:
    sig = pd.to_numeric(sig, errors="coerce").fillna(0).astype(int)
    if "micro_context_available" in baseline.columns:
        has_ctx = _bool_mask(baseline["micro_context_available"])
    else:
        has_ctx = pd.Series(False, index=baseline.index)
    imb = _num(baseline, "rf_imbalance")
    pos = _num(baseline, "rf_close_pos")

    aligned_imb = float(getattr(args, "micro_aligned_imbalance", 0.05) or 0.05)
    contra_imb = float(getattr(args, "micro_contra_imbalance", 0.05) or 0.05)
    good_pos = float(getattr(args, "micro_good_close_pos", 0.65) or 0.65)
    bad_pos = float(getattr(args, "micro_bad_close_pos", 0.35) or 0.35)

    long_sig = sig.eq(1)
    short_sig = sig.eq(-1)
    aligned = (
        (long_sig & has_ctx & (imb >= aligned_imb) & (pos >= good_pos))
        | (short_sig & has_ctx & (imb <= -aligned_imb) & (pos <= bad_pos))
    )
    contra = (
        (long_sig & has_ctx & (imb <= -contra_imb) & (pos <= bad_pos))
        | (short_sig & has_ctx & (imb >= contra_imb) & (pos >= good_pos))
    )
    action = pd.Series("NEUTRAL", index=baseline.index, dtype="object")
    action.loc[aligned] = "ALIGNED"
    action.loc[contra] = "CONTRA"
    action.loc[sig.ne(0) & has_ctx & (~aligned) & (~contra)] = "NOT_ALIGNED_RISK_REDUCED"
    action.loc[sig.eq(0)] = "NO_SIGNAL"
    return action


def build_engine_condition_flags(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: Any) -> pd.DataFrame:
    out = pd.DataFrame(index=baseline.index)
    for engine in ENGINES:
        df = raw[engine].reindex(baseline.index)
        sig = _num(df, "signal", 0.0).fillna(0).astype(int)
        active = sig.ne(0)
        long_sig = sig.eq(1)
        short_sig = sig.eq(-1)
        micro_action = _engine_micro_action(sig, baseline, args)
        vol_ratio = _volume_ratio_for_engine(df, baseline)
        vol_q = _qcut_label_on_mask(vol_ratio, active, 4, "VOL_Q")

        prefix = engine.lower().replace("_v3_only", "").replace("_v2", "").replace("_v3", "")
        # Stable short prefixes for readable CSV columns.
        if engine == ENGINE_MOM:
            prefix = "mom"
        elif engine == ENGINE_BULL:
            prefix = "bull"
        elif engine == ENGINE_BEAR:
            prefix = "bear"

        out[f"{prefix}_signal"] = sig
        out[f"{prefix}_active"] = active
        out[f"{prefix}_long"] = long_sig
        out[f"{prefix}_short"] = short_sig
        out[f"{prefix}_micro_action"] = micro_action
        out[f"{prefix}_micro_not_aligned"] = active & micro_action.eq("NOT_ALIGNED_RISK_REDUCED")
        out[f"{prefix}_volume_ratio"] = vol_ratio
        out[f"{prefix}_volume_q"] = vol_q
        out[f"{prefix}_low_volume"] = active & vol_q.eq("VOL_Q1")

    out["mom_long_not_aligned"] = out["mom_long"] & out["mom_micro_not_aligned"]
    out["mom_long_low_volume"] = out["mom_long"] & out["mom_low_volume"]
    out["mom_long_not_aligned_or_low_volume"] = out["mom_long_not_aligned"] | out["mom_long_low_volume"]

    out["global_any_micro_not_aligned"] = out["mom_micro_not_aligned"] | out["bull_micro_not_aligned"] | out["bear_micro_not_aligned"]
    out["global_any_low_volume"] = out["mom_low_volume"] | out["bull_low_volume"] | out["bear_low_volume"]
    out["global_any_not_aligned_or_low_volume"] = out["global_any_micro_not_aligned"] | out["global_any_low_volume"]
    return out


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
        df = out[engine]
        _assign_blocked_zero(df, mask, ["signal", "long_signal", "short_signal", "momentum_signal", "bull_signal", "bear_signal"])
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
    features["conditional_v2_blocked"] = False
    features["conditional_v2_block_note"] = "NONE"
    if block_map:
        # Mark bars where any raw candidate was blocked. This is audit metadata only;
        # final selected entry may still come from a different engine on that bar.
        any_block = pd.Series(False, index=features.index)
        notes = []
        for engine, mask in block_map.items():
            m = mask.reindex(features.index).fillna(False).astype(bool)
            any_block = any_block | m
            if bool(m.any()):
                notes.append(f"{engine}:{int(m.sum())}")
        features.loc[any_block, "conditional_v2_blocked"] = True
        features.loc[any_block, "conditional_v2_block_note"] = ";".join(notes) if notes else "BLOCKED"
    return features


def build_scenarios(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: Any, cond: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, dict[str, Any]]]:
    scenarios: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {}
    scenarios["baseline_v9e"] = (baseline.copy(), {"variant_type": "baseline"})

    # Engine-specific candidate rules: only Momentum Long is filtered.
    specs: list[tuple[str, dict[str, pd.Series], str]] = [
        (
            "mom_long_not_aligned_block",
            {ENGINE_MOM: cond["mom_long_not_aligned"]},
            "Block only raw Momentum Long when Momentum micro action is NOT_ALIGNED_RISK_REDUCED.",
        ),
        (
            "mom_long_low_volume_block",
            {ENGINE_MOM: cond["mom_long_low_volume"]},
            "Block only raw Momentum Long when volume ratio is VOL_Q1 among Momentum signal events.",
        ),
        (
            "mom_long_not_aligned_or_low_volume_block",
            {ENGINE_MOM: cond["mom_long_not_aligned_or_low_volume"]},
            "Block raw Momentum Long when micro not-aligned OR low volume.",
        ),
        # Deliberately broad/global counterexamples. These are NOT primary candidates.
        (
            "global_micro_not_aligned_block_counterexample",
            {
                ENGINE_MOM: cond["mom_micro_not_aligned"],
                ENGINE_BULL: cond["bull_micro_not_aligned"],
                ENGINE_BEAR: cond["bear_micro_not_aligned"],
            },
            "Counterexample: block every engine's raw signal when that engine is micro not-aligned.",
        ),
        (
            "global_low_volume_block_counterexample",
            {
                ENGINE_MOM: cond["mom_low_volume"],
                ENGINE_BULL: cond["bull_low_volume"],
                ENGINE_BEAR: cond["bear_low_volume"],
            },
            "Counterexample: block every engine's raw signal when that engine is VOL_Q1 among its own signal events.",
        ),
        (
            "global_not_aligned_or_low_volume_block_counterexample",
            {
                ENGINE_MOM: cond["mom_micro_not_aligned"] | cond["mom_low_volume"],
                ENGINE_BULL: cond["bull_micro_not_aligned"] | cond["bull_low_volume"],
                ENGINE_BEAR: cond["bear_micro_not_aligned"] | cond["bear_low_volume"],
            },
            "Counterexample: block every engine's raw signal when micro not-aligned OR low volume.",
        ),
    ]

    for name, block_map, note in specs:
        features = make_features_for_block_map(baseline, raw, args, scenario=name, block_map=block_map)
        extra = {
            "variant_type": "conditional_v2_block",
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
        "mom_long_not_aligned",
        "mom_long_low_volume",
        "mom_long_not_aligned_or_low_volume",
        "mom_micro_not_aligned",
        "mom_low_volume",
        "bull_micro_not_aligned",
        "bull_low_volume",
        "bear_micro_not_aligned",
        "bear_low_volume",
        "global_any_micro_not_aligned",
        "global_any_low_volume",
        "global_any_not_aligned_or_low_volume",
    ]
    for col in condition_cols:
        if col not in cond.columns:
            continue
        rows.append({"condition": col, "count": int(cond[col].sum())})
    for prefix in ["mom", "bull", "bear"]:
        active = cond.get(f"{prefix}_active", pd.Series(False, index=cond.index)).astype(bool)
        rows.append({"condition": f"{prefix}_active", "count": int(active.sum())})
        for q in sorted(cond.loc[active, f"{prefix}_volume_q"].dropna().astype(str).unique().tolist()) if f"{prefix}_volume_q" in cond.columns else []:
            rows.append({"condition": f"{prefix}_volume_q_{q}", "count": int((active & cond[f"{prefix}_volume_q"].eq(q)).sum())})
    return pd.DataFrame(rows).sort_values("condition").reset_index(drop=True)


def main() -> int:
    args = router.parse_args()
    if str(args.out_dir).replace("\\", "/").endswith("v9e_engine_router_variants_lab"):
        args.out_dir = "data/reports/research/v9e_momentum_conditional_router_v2_lab"
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(PROJECT_ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline, raw, cfg, engine_cfgs = router.build_features(args)
    cond = build_engine_condition_flags(baseline, raw, args)
    cond.to_csv(out_dir / "v9e_momentum_conditional_v2_regime_flags.csv", encoding="utf-8-sig")
    build_condition_counts(cond).to_csv(out_dir / "v9e_momentum_conditional_v2_condition_counts.csv", index=False, encoding="utf-8-sig")

    scenarios = build_scenarios(baseline, raw, args, cond)
    summaries: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []

    for name, (features, extra) in scenarios.items():
        print(f"Running conditional router v2 variant: {name}", flush=True)
        summary, trades_df, trades, equity = router.run_variant(name, features, cfg, engine_cfgs, args, addon_mode="off", extra=extra)
        selected_engine = features.get("selected_engine", pd.Series("", index=features.index)).astype(str)
        sig = pd.to_numeric(features.get("signal", pd.Series(0, index=features.index)), errors="coerce").fillna(0).astype(int)
        summary["selected_momentum_count"] = int(selected_engine.eq(ENGINE_MOM).sum())
        summary["selected_momentum_long_count"] = int((selected_engine.eq(ENGINE_MOM) & sig.eq(1)).sum())
        summary["selected_momentum_short_count"] = int((selected_engine.eq(ENGINE_MOM) & sig.eq(-1)).sum())
        summary["selected_bull_count"] = int(selected_engine.eq(ENGINE_BULL).sum())
        summary["selected_bear_count"] = int(selected_engine.eq(ENGINE_BEAR).sum())
        summary["conditional_v2_blocked_bar_count"] = int(features.get("conditional_v2_blocked", pd.Series(False, index=features.index)).astype("boolean").fillna(False).sum())
        summaries.append(summary)
        yearly_frames.append(router.yearly_metrics(trades, equity, name, cfg.initial_capital))

        if args.write_trades:
            trades_df.to_csv(out_dir / f"{name}_trades.csv", index=False, encoding="utf-8-sig")
            if not equity.empty:
                equity.to_csv(out_dir / f"{name}_equity.csv", encoding="utf-8-sig")
            audit_cols = [
                "open", "high", "low", "close", "volume", "signal", "selected_engine",
                "momentum_signal", "bull_signal", "bear_signal", "risk_mult", "quality_mult",
                "micro_filter_action", "rf_imbalance", "rf_close_pos",
                "conditional_v2_blocked", "conditional_v2_block_note", "router_note",
            ]
            features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{name}_signal_audit.csv", encoding="utf-8-sig")

    summary_df = pd.DataFrame(summaries)
    if not summary_df.empty and "closed_final_capital" in summary_df.columns:
        summary_df = summary_df.sort_values("closed_final_capital", ascending=False)
    summary_df.to_csv(out_dir / "v9e_momentum_conditional_v2_variant_summary.csv", index=False, encoding="utf-8-sig")

    yearly_df = pd.concat([f for f in yearly_frames if not f.empty], ignore_index=True) if yearly_frames else pd.DataFrame()
    yearly_df.to_csv(out_dir / "v9e_momentum_conditional_v2_variant_yearly.csv", index=False, encoding="utf-8-sig")

    if not summary_df.empty:
        base = summary_df[summary_df["scenario"].eq("baseline_v9e")]
        if not base.empty:
            b = base.iloc[0]
            comp = summary_df.copy()
            for col in [
                "closed_final_capital", "closed_profit_factor", "closed_win_rate",
                "max_drawdown_pct", "closed_expectancy_pct", "closed_total_trades",
                "selected_momentum_count", "selected_bull_count", "selected_bear_count",
            ]:
                if col in comp.columns and col in b.index:
                    comp[f"delta_{col}"] = pd.to_numeric(comp[col], errors="coerce") - float(b[col])
            comp.to_csv(out_dir / "v9e_momentum_conditional_v2_compare_to_baseline.csv", index=False, encoding="utf-8-sig")

    meta = {
        "script": "v9e_momentum_conditional_router_v2_lab.py",
        "mode": "chronological_portfolio_backtest_with_conditional_raw_signal_blocks",
        "args": vars(args),
        "scenarios": list(scenarios.keys()),
        "important_note": "Primary candidates are Momentum-Long-specific blocks. Global filters are included as counterexamples and should not be treated as candidates unless they robustly improve portfolio metrics.",
        "output_dir": str(out_dir),
    }
    (out_dir / "v9e_momentum_conditional_v2_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 96)
    print("V9E Momentum Conditional Router V2 Lab completed")
    print("=" * 96)
    print(f"Output directory: {out_dir.resolve()}")
    print("Key files:")
    print("  - v9e_momentum_conditional_v2_variant_summary.csv")
    print("  - v9e_momentum_conditional_v2_compare_to_baseline.csv")
    print("  - v9e_momentum_conditional_v2_variant_yearly.csv")
    print("  - v9e_momentum_conditional_v2_condition_counts.csv")
    print("=" * 96 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
