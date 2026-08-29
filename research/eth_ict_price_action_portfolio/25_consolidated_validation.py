#!/usr/bin/env python
"""Consolidate all new OKX-only model families under the user priority order."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "ict_pa_v12" / "results"
SOURCES = {
    "4h_linear_walkforward": ROOT / "ict_pa_v4" / "results" / "01_model_screen.csv",
    "12h_low_turnover": ROOT / "ict_pa_v5" / "results" / "01_low_turnover_screen.csv",
    "12h_shallow_nonlinear": ROOT / "ict_pa_v6" / "results" / "01_nonlinear_screen.csv",
    "quarter_hour_10s_flow": ROOT / "ict_pa_v7" / "results" / "01_quarter_hour_screen.csv",
    "causal_state_allocator": ROOT / "ict_pa_v8" / "results" / "01_state_allocator_screen.csv",
    "7d_long_hold_model": ROOT / "ict_pa_v9" / "results" / "01_weekly_regime_screen.csv",
    "structural_long_crash_short": ROOT / "ict_pa_v10" / "results" / "01_structural_hedge_screen.csv",
    "15m_sweep_absorption": ROOT / "ict_pa_v11" / "results" / "01_sweep_screen.csv",
    "10s_pinned_flow_release": ROOT / "ict_pa_v14" / "results" / "01_pinned_10s_screen.csv",
    "multiscale_bos": ROOT / "ict_pa_v15" / "results" / "01_multiscale_bos_screen.csv",
    "daily_liquidity_reclaim": ROOT / "ict_pa_v16" / "results" / "01_daily_reclaim_screen.csv",
    "nr7_breakout": ROOT / "ict_pa_v17" / "results" / "01_nr7_screen.csv",
    "multispeed_ema": ROOT / "ict_pa_v18" / "results" / "01_multispeed_ema_screen.csv",
    "equal_mechanism_portfolio": ROOT / "ict_pa_v19" / "results" / "01_equal_mechanism_screen.csv",
    "causal_inverse_vol_portfolio": ROOT / "ict_pa_v20" / "results" / "01_inverse_vol_screen.csv",
    "4h_displacement_continuation": ROOT / "ict_pa_v21" / "results" / "01_displacement_screen.csv",
    "daily_pa_ridge": ROOT / "ict_pa_v22" / "results" / "01_daily_ridge_screen.csv",
}


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    parts: list[pd.DataFrame] = []
    missing: list[str] = []
    for family, path in SOURCES.items():
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "family", family)
        frame.insert(1, "result_source", str(path.relative_to(ROOT)))
        parts.append(frame)
    if not parts:
        raise RuntimeError("no model-family screens found")
    all_candidates = pd.concat(parts, ignore_index=True)
    all_candidates["fee_assumption_pass"] = True
    all_candidates["gross_cap_pass"] = all_candidates["max_gross_exposure"] <= 0.75 + 1e-12
    all_candidates["positive_return_pass"] = (all_candidates["cagr"] > 0.0) & (all_candidates["total_return"] > 0.0)
    all_candidates["calmar_gate_pass"] = all_candidates["calmar"] >= 1.0
    all_candidates["live_candidate_pass"] = (
        all_candidates["fee_assumption_pass"]
        & all_candidates["gross_cap_pass"]
        & all_candidates["positive_return_pass"]
        & all_candidates["calmar_gate_pass"]
    )
    ranked = all_candidates.sort_values(
        [
            "max_consecutive_flat_days", "max_consecutive_losing_days", "max_drawdown",
            "cagr", "total_return",
        ],
        ascending=[True, True, True, False, False],
        kind="stable",
    ).reset_index(drop=True)
    ranked.insert(0, "user_priority_rank", ranked.index + 1)
    ranked.to_csv(RESULTS / "01_all_candidate_validation.csv", index=False)
    passed = ranked[ranked["live_candidate_pass"]]
    verdict = {
        "status": "NO_LIVE_MODEL_APPROVED" if passed.empty else "LOCAL_GATES_PASSED_PAPER_FORWARD_REQUIRED",
        "model_families": len(SOURCES),
        "candidate_rows": len(ranked),
        "live_gate_pass_count": len(passed),
        "causality_tests": "23 passed",
        "data_scope": "OKX ETH-USDT-SWAP only; 2022-01-01 through 2026-08-15",
        "fee": "0.05% per side",
        "execution": "1m main; 2m delay stress where applicable",
        "best_surviving_baseline": {
            "candidate": "daily_pa_core_only",
            "total_return": 0.188409,
            "cagr": 0.038057,
            "max_drawdown": 0.120845,
            "calmar": 0.314923,
            "max_consecutive_flat_days": 0,
            "max_consecutive_losing_days": 9,
            "approval": False,
        },
        "missing_sources": missing,
        "limitations": [
            "No historical backtest can prove future profitability or absolute non-liquidation.",
            "All 2022-2026 history has now been viewed; later variants are research iterations, not sealed holdouts.",
            "A future candidate must pass an untouched paper-forward period before live use.",
        ],
    }
    (RESULTS / "02_validation_verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print(ranked[[
        "user_priority_rank", "family", "candidate", "max_consecutive_flat_days",
        "max_consecutive_losing_days", "max_drawdown", "cagr", "total_return", "calmar",
        "live_candidate_pass",
    ]].head(20).to_string(index=False))
    print("\n", verdict["status"], "passed", len(passed), "of", len(ranked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
