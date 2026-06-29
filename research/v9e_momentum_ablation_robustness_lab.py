#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9E Momentum Ablation Robustness Lab
====================================

Research-only robustness check for the hypothesis:
    "Momentum Short looks good only because 2026 is a down year and capital is largest."

This lab intentionally avoids optimizing a new rule. It reuses the exact router variants
from v9e_engine_router_variants_lab.py, then evaluates them with anti-overfit views:
    1) Full-sample compounded result.
    2) Per-year reset-to-initial-capital result, so 2026 does not dominate because of compounding.
    3) Pre-2026 only result.
    4) Year-by-year delta versus baseline.
    5) Simple robustness score: equal-year average, positive-year count, improvement-year count.

It does NOT modify V9E and does NOT place orders.
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
RESEARCH_DIR = os.path.dirname(CURRENT_FILE)
for p in (PROJECT_ROOT, RESEARCH_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import v9e_engine_router_variants_lab as router  # noqa: E402


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clone_args(args: Any, **updates: Any) -> Any:
    """Create a shallow argparse-like clone without mutating the caller's args."""
    from argparse import Namespace

    data = dict(vars(args))
    data.update(updates)
    return Namespace(**data)


def _subset_by_date(df: pd.DataFrame, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    out = df
    if start:
        out = out[out.index >= pd.Timestamp(start)]
    if end:
        out = out[out.index <= pd.Timestamp(end)]
    return out.copy()


def _year_bounds(features: pd.DataFrame) -> list[int]:
    years = sorted({int(x) for x in pd.DatetimeIndex(features.index).year})
    return years


def _select_scenarios(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: Any) -> dict[str, tuple[pd.DataFrame, dict[str, Any], str]]:
    """Return only variants useful for testing momentum long/short overfit."""
    scenarios = router.scenario_features(baseline, raw, args)

    # Add the opposite ablation: disable Momentum Short but keep Momentum Long under current-like priority.
    scenarios["router_momentum_short_disabled_keep_long_current_priority"] = (
        router.make_engine_aware_router(
            baseline,
            raw,
            args,
            scenario="router_momentum_short_disabled_keep_long_current_priority",
            momentum_entry_mode="long_only",
            priority_order=(router.ENGINE_BULL, router.ENGINE_MOM, router.ENGINE_BEAR),
        ),
        {
            "variant_type": "entry_router",
            "momentum_entry_mode": "long_only",
            "priority_order": "BULL>MOM_LONG>BEAR",
            "purpose": "opposite_ablation_disable_momentum_short",
        },
        "off",
    )

    wanted = [
        "baseline_v9e",
        "router_momentum_long_disabled_keep_short_current_priority",
        "router_momentum_short_only_after_bull_bear",
        "router_momentum_short_disabled_keep_long_current_priority",
        "router_momentum_long_risk_down_current_priority",
        "router_momentum_all_after_bull_bear_risk_down",
        "router_bull_bear_primary_momentum_disabled",
    ]
    return {name: scenarios[name] for name in wanted if name in scenarios}


def _run_one(name: str, features: pd.DataFrame, cfg: Any, engine_cfgs: dict[str, Any], args: Any, addon_mode: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    summary, trades_df, trades, equity = router.run_variant(name, features, cfg, engine_cfgs, args, addon_mode=addon_mode, extra=extra or {})
    summary["trade_rows"] = int(len(trades_df))
    return summary


def _run_scenario_set(
    scenarios: dict[str, tuple[pd.DataFrame, dict[str, Any], str]],
    *,
    cfg: Any,
    engine_cfgs: dict[str, Any],
    args: Any,
    label: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, (features, extra, addon_mode) in scenarios.items():
        f = _subset_by_date(features, start, end)
        if f.empty:
            continue
        run_name = f"{label}__{name}"
        row = _run_one(run_name, f, cfg, engine_cfgs, args, addon_mode=addon_mode, extra=dict(extra, eval_label=label, eval_start=start, eval_end=end))
        row["base_scenario"] = name
        row["eval_label"] = label
        row["eval_start"] = start or str(f.index.min())
        row["eval_end"] = end or str(f.index.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _append_baseline_deltas(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty or "base_scenario" not in df.columns:
        return df
    out = df.copy()
    metric_cols = [
        "final_capital",
        "total_return_pct",
        "closed_final_capital",
        "closed_total_return_pct",
        "closed_profit_factor",
        "closed_win_rate",
        "closed_expectancy_pct",
        "max_drawdown_pct",
        "closed_total_trades",
        "force_close_pnl",
    ]
    existing = [c for c in metric_cols if c in out.columns]
    for _, g in out.groupby(group_cols, dropna=False):
        base = g[g["base_scenario"].eq("baseline_v9e")]
        if base.empty:
            continue
        base_idx = base.index[0]
        for idx in g.index:
            for col in existing:
                b = _safe_float(out.at[base_idx, col])
                v = _safe_float(out.at[idx, col])
                out.at[idx, f"delta_{col}"] = v - b if math.isfinite(v) and math.isfinite(b) else float("nan")
        b_final = _safe_float(out.at[base_idx, "closed_final_capital"] if "closed_final_capital" in out.columns else np.nan)
        for idx in g.index:
            v_final = _safe_float(out.at[idx, "closed_final_capital"] if "closed_final_capital" in out.columns else np.nan)
            if math.isfinite(v_final) and math.isfinite(b_final) and b_final != 0:
                out.at[idx, "delta_closed_final_capital_pct"] = (v_final / b_final - 1.0) * 100.0
    return out


def _robustness_score(year_reset_df: pd.DataFrame, full_df: pd.DataFrame, no2026_df: pd.DataFrame) -> pd.DataFrame:
    if year_reset_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    grouped = year_reset_df.groupby("base_scenario", dropna=False)
    for scenario, g in grouped:
        if scenario == "baseline_v9e":
            continue
        closed_returns = pd.to_numeric(g.get("closed_total_return_pct"), errors="coerce")
        marked_returns = pd.to_numeric(g.get("total_return_pct"), errors="coerce")
        delta_closed = pd.to_numeric(g.get("delta_closed_final_capital_pct"), errors="coerce")
        delta_pf = pd.to_numeric(g.get("delta_closed_profit_factor"), errors="coerce")
        delta_dd = pd.to_numeric(g.get("delta_max_drawdown_pct"), errors="coerce")
        years = int(g["year"].nunique()) if "year" in g.columns else int(len(g))

        full_row = full_df[full_df["base_scenario"].eq(scenario)].head(1)
        no2026_row = no2026_df[no2026_df["base_scenario"].eq(scenario)].head(1)
        full_delta_pct = _safe_float(full_row["delta_closed_final_capital_pct"].iloc[0]) if not full_row.empty and "delta_closed_final_capital_pct" in full_row.columns else float("nan")
        no2026_delta_pct = _safe_float(no2026_row["delta_closed_final_capital_pct"].iloc[0]) if not no2026_row.empty and "delta_closed_final_capital_pct" in no2026_row.columns else float("nan")

        rows.append(
            {
                "scenario": scenario,
                "year_count": years,
                "avg_year_reset_closed_return_pct": float(closed_returns.mean()) if not closed_returns.empty else float("nan"),
                "median_year_reset_closed_return_pct": float(closed_returns.median()) if not closed_returns.empty else float("nan"),
                "worst_year_reset_closed_return_pct": float(closed_returns.min()) if not closed_returns.empty else float("nan"),
                "positive_closed_year_count": int((closed_returns > 0).sum()),
                "positive_marked_year_count": int((marked_returns > 0).sum()),
                "improved_closed_cap_year_count": int((delta_closed > 0).sum()),
                "avg_delta_closed_cap_pct_vs_baseline": float(delta_closed.mean()) if not delta_closed.empty else float("nan"),
                "median_delta_closed_cap_pct_vs_baseline": float(delta_closed.median()) if not delta_closed.empty else float("nan"),
                "worst_delta_closed_cap_pct_vs_baseline": float(delta_closed.min()) if not delta_closed.empty else float("nan"),
                "avg_delta_pf_vs_baseline": float(delta_pf.mean()) if not delta_pf.empty else float("nan"),
                "avg_delta_max_dd_pct_vs_baseline": float(delta_dd.mean()) if not delta_dd.empty else float("nan"),
                "full_sample_delta_closed_cap_pct": full_delta_pct,
                "pre2026_delta_closed_cap_pct": no2026_delta_pct,
                "latest_year_overfit_warning": bool(math.isfinite(full_delta_pct) and math.isfinite(no2026_delta_pct) and full_delta_pct > 0 and no2026_delta_pct < 0),
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty:
        # Conservative sorting: prefer robust yearly improvement, not terminal wealth.
        out = out.sort_values(
            ["improved_closed_cap_year_count", "positive_closed_year_count", "avg_delta_closed_cap_pct_vs_baseline"],
            ascending=[False, False, False],
        )
    return out


def _add_year_col(df: pd.DataFrame, year: int) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "year", int(year))
    out["eval_label"] = f"year_reset_{year}"
    return out


def main() -> int:
    args = router.parse_args()
    # Re-route default output dir to this lab if user did not pass one explicitly.
    old_default = "data/reports/research/v9e_engine_router_variants_lab"
    if str(args.out_dir).replace("\\", "/") == old_default:
        args.out_dir = "data/reports/research/v9e_momentum_ablation_robustness_lab"

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(PROJECT_ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building V9E features once...", flush=True)
    baseline, raw, cfg, engine_cfgs = router.build_features(args)
    scenarios = _select_scenarios(baseline, raw, args)
    print(f"Scenarios: {list(scenarios)}", flush=True)

    # Full-sample compounded view, same as previous lab.
    print("Running full-sample compounded view...", flush=True)
    full_df = _run_scenario_set(scenarios, cfg=cfg, engine_cfgs=engine_cfgs, args=args, label="full_sample", start=None, end=None)
    full_df = _append_baseline_deltas(full_df, ["eval_label"])
    full_df.to_csv(out_dir / "v9e_momentum_ablation_full_sample.csv", index=False)

    # Pre-2026 view directly addresses the user's concern.
    print("Running pre-2026 view...", flush=True)
    no2026_df = _run_scenario_set(scenarios, cfg=cfg, engine_cfgs=engine_cfgs, args=args, label="pre_2026_only", start=str(args.start_date), end="2025-12-31 23:59:59")
    no2026_df = _append_baseline_deltas(no2026_df, ["eval_label"])
    no2026_df.to_csv(out_dir / "v9e_momentum_ablation_pre_2026_only.csv", index=False)

    # Year-reset view removes compounding domination by later years.
    print("Running year-reset equal-capital view...", flush=True)
    year_frames: list[pd.DataFrame] = []
    for year in _year_bounds(baseline):
        if year < pd.Timestamp(args.start_date).year:
            continue
        start = f"{year}-01-01 00:00:00"
        end = f"{year}-12-31 23:59:59"
        # Cap last year by actual end date.
        if pd.Timestamp(end) > pd.Timestamp(args.end_date):
            end = str(args.end_date)
        print(f"  year reset: {year} {start} -> {end}", flush=True)
        ydf = _run_scenario_set(scenarios, cfg=cfg, engine_cfgs=engine_cfgs, args=args, label=f"year_reset_{year}", start=start, end=end)
        ydf = _append_baseline_deltas(ydf, ["eval_label"])
        year_frames.append(_add_year_col(ydf, year))
    year_reset_df = pd.concat(year_frames, ignore_index=True) if year_frames else pd.DataFrame()
    year_reset_df.to_csv(out_dir / "v9e_momentum_ablation_year_reset.csv", index=False)

    score_df = _robustness_score(year_reset_df, full_df, no2026_df)
    score_df.to_csv(out_dir / "v9e_momentum_ablation_robustness_score.csv", index=False)

    with (out_dir / "v9e_momentum_ablation_robustness_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "args": vars(args),
                "output_dir": str(out_dir.resolve()),
                "scenarios": list(scenarios),
                "purpose": "Check whether Momentum Short-only advantage is merely latest-year/down-year/compounding overfit.",
                "key_files": [
                    "v9e_momentum_ablation_full_sample.csv",
                    "v9e_momentum_ablation_pre_2026_only.csv",
                    "v9e_momentum_ablation_year_reset.csv",
                    "v9e_momentum_ablation_robustness_score.csv",
                ],
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    print("\n" + "=" * 96)
    print("V9E Momentum Ablation Robustness Lab completed")
    print("=" * 96)
    print(f"Output directory: {out_dir.resolve()}")
    print("Key files:")
    print("  - v9e_momentum_ablation_robustness_score.csv")
    print("  - v9e_momentum_ablation_year_reset.csv")
    print("  - v9e_momentum_ablation_pre_2026_only.csv")
    print("  - v9e_momentum_ablation_full_sample.csv")
    print("=" * 96 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
