#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RDPOS-03: mature-expansion path atlas + path-aware position counterfactual.

This is still research.  The path definition was discovered using the same
2023-2026 sample, so even a better counterfactual is NOT an independent
holdout result and cannot be promoted directly to live trading.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.research_common.eth_dynamic_position_path import (  # noqa: E402
    PathRule,
    account_summary,
    add_forward_market_labels,
    build_path_table,
    exposure_matched_base,
    path_neighborhood_table,
    replay_targets,
    summarize_neighborhood,
    validate_report_inputs,
    yearly_summary,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        default="data/reports/research/eth_dynamic_positioning/01_trend_location_vol_positioning",
    )
    p.add_argument(
        "--output-dir",
        default="data/reports/research/eth_dynamic_positioning/03_path_aware_positioning_counterfactual",
    )
    return p.parse_args()


def _load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, object]]:
    dec = pd.read_csv(root / "decision_audit.csv", parse_dates=["timestamp", "available_time"])
    eq = pd.read_csv(root / "equity_hourly.csv", parse_dates=["timestamp", "next_timestamp"])
    cfg = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    dec = dec.sort_values("available_time", kind="stable").reset_index(drop=True)
    eq = eq.sort_values("timestamp", kind="stable").reset_index(drop=True)
    validate_report_inputs(dec, eq)
    return dec, eq, cfg, summary


def _assert_base_replay_matches(original: dict[str, object], replay: dict[str, float]) -> None:
    expected = original["summary"]
    for key in ("total_return", "cagr", "max_drawdown", "mean_gross_exposure"):
        if not np.isclose(float(expected[key]), float(replay[key]), atol=1e-10, rtol=1e-10):
            raise RuntimeError(
                f"base replay mismatch for {key}: expected={expected[key]} replay={replay[key]}"
            )


def _scenario_row(name: str, frame: pd.DataFrame) -> dict[str, object]:
    return {"scenario": name, **account_summary(frame)}


def _central_path_yearly(path: pd.DataFrame, trade_start: pd.Timestamp) -> pd.DataFrame:
    q = path[path["mature_expansion"] & (path["available_time"] >= trade_start)].copy()
    q = q.dropna(subset=["aligned_return_12h", "aligned_return_24h", "aligned_return_72h"])
    rows = []
    for year, g in q.groupby(q["available_time"].dt.year):
        rows.append({
            "year": int(year),
            "n": int(len(g)),
            "aligned_return_12h": float(g["aligned_return_12h"].mean()),
            "aligned_return_24h": float(g["aligned_return_24h"].mean()),
            "aligned_return_72h": float(g["aligned_return_72h"].mean()),
            "mean_state_age_hours": float(g["state_age_hours"].mean()),
            "mean_aligned_extension": float(g["aligned_extension_mean"].mean()),
            "mean_strong_share_72h": float(g["strong_share_72h"].mean()),
        })
    return pd.DataFrame(rows)


def _write_report(
    out: Path,
    *,
    rule: PathRule,
    scenarios: pd.DataFrame,
    path_yearly: pd.DataFrame,
    neighborhood_summary: pd.DataFrame,
    risk_matches: dict[str, object],
    verdict: dict[str, object],
) -> None:
    lines = [
        "# ETH Dynamic Positioning RDPOS-03 — Path-Aware Positioning Counterfactual",
        "",
        "## Purpose",
        "",
        "Test the specific path hypothesis discovered in RDPOS-02: a mature, persistent trend expansion may deserve **more exposure than the frozen location penalty currently allows**.",
        "",
        "This study does **not** tune trend horizons or create entry/TP/SL trades.",
        "",
        "## Frozen central path definition",
        "",
        "```json",
        json.dumps(rule.__dict__, indent=2),
        "```",
        "",
        "A qualifying state requires medium/slow strong agreement, enough state age, positive aligned extension, and a sufficiently high share of strong-agreement observations over the past 72h.",
        "",
        "## Counterfactual scenarios",
        "",
        scenarios.to_markdown(index=False),
        "",
        "`mature_expansion_no_penalty` only removes the old extension penalty while the mature-expansion path is present. `mature_expansion_reward` is exploratory and restores plus rewards aligned expansion using the already-frozen 0.25 location magnitude.",
        "",
        "## Central path forward market outcomes",
        "",
        path_yearly.to_markdown(index=False),
        "",
        "## Exposure-matched controls",
        "",
        "```json",
        json.dumps(risk_matches, indent=2, default=str),
        "```",
        "",
        "The static control scale is selected **only** to match mean gross exposure. PnL is not used to choose the scale.",
        "",
        "## Neighborhood robustness",
        "",
        f"Pre-specified combinations tested: {len(neighborhood_summary)}",
        f"All-year positive 24h combinations: {int(neighborhood_summary['all_years_positive_24h'].sum()) if not neighborhood_summary.empty else 0}",
        f"All-year positive 72h combinations: {int(neighborhood_summary['all_years_positive_72h'].sum()) if not neighborhood_summary.empty else 0}",
        "",
        "## Verdict",
        "",
        "```json",
        json.dumps(verdict, indent=2, default=str),
        "```",
        "",
        "## Guardrails",
        "",
        "- This path definition is **post-discovery on the same historical sample**; it is not an independent holdout.",
        "- A higher CAGR caused only by higher average exposure is not path alpha; exposure-matched controls are mandatory.",
        "- If CAGR still does not exceed |MDD|, the account is not promoted even if the path adds return.",
        "- The next research target should be false-expansion / deterioration detection, not a larger leverage multiplier, when extra exposure increases drawdown.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    if not inp.is_absolute():
        inp = Path(PROJECT_ROOT) / inp
    if not out.is_absolute():
        out = Path(PROJECT_ROOT) / out
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {inp}", flush=True)
    decisions, equity, config, original_summary = _load(inp)

    rule = PathRule()
    path = build_path_table(decisions, rule)
    path = add_forward_market_labels(path, equity)
    path.to_csv(out / "decision_path_states.csv", index=False, encoding="utf-8-sig")

    central_yearly = _central_path_yearly(path, pd.Timestamp(equity["timestamp"].min()))
    central_yearly.to_csv(out / "central_path_yearly.csv", index=False, encoding="utf-8-sig")

    print("[atlas] path-neighborhood robustness 3x3x3x3", flush=True)
    neighborhood = path_neighborhood_table(decisions, equity)
    neighborhood.to_csv(out / "path_neighborhood_yearly.csv", index=False, encoding="utf-8-sig")
    neighborhood_summary = summarize_neighborhood(neighborhood)
    neighborhood_summary.to_csv(out / "path_neighborhood_summary.csv", index=False, encoding="utf-8-sig")

    print("[replay] frozen base", flush=True)
    base = replay_targets(path, equity, config, target_suffix="base")
    base_summary = account_summary(base)
    _assert_base_replay_matches(original_summary, base_summary)

    print("[replay] mature expansion no-penalty", flush=True)
    no_penalty = replay_targets(path, equity, config, target_suffix="no_penalty")
    reward = replay_targets(path, equity, config, target_suffix="reward")
    no_penalty_cost2 = replay_targets(
        path, equity, config, target_suffix="no_penalty", cost_multiplier=2.0
    )
    no_penalty_cost3 = replay_targets(
        path, equity, config, target_suffix="no_penalty", cost_multiplier=3.0
    )

    no_summary = account_summary(no_penalty)
    reward_summary = account_summary(reward)

    print("[control] exposure-matched static base", flush=True)
    no_match_frame, no_match, no_grid = exposure_matched_base(
        path, equity, config, target_mean_gross=float(no_summary["mean_gross_exposure"])
    )
    reward_match_frame, reward_match, reward_grid = exposure_matched_base(
        path, equity, config, target_mean_gross=float(reward_summary["mean_gross_exposure"])
    )
    no_grid.assign(control_for="no_penalty").to_csv(
        out / "risk_match_grid_no_penalty.csv", index=False, encoding="utf-8-sig"
    )
    reward_grid.assign(control_for="reward").to_csv(
        out / "risk_match_grid_reward.csv", index=False, encoding="utf-8-sig"
    )

    scenario_frames = {
        "frozen_base": base,
        "mature_expansion_no_penalty": no_penalty,
        "mature_expansion_reward": reward,
        "base_exposure_matched_no_penalty": no_match_frame,
        "base_exposure_matched_reward": reward_match_frame,
        "mature_expansion_no_penalty_cost2x": no_penalty_cost2,
        "mature_expansion_no_penalty_cost3x": no_penalty_cost3,
    }
    scenarios = pd.DataFrame([_scenario_row(name, frame) for name, frame in scenario_frames.items()])
    scenarios.to_csv(out / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(
        [yearly_summary(frame, name) for name, frame in scenario_frames.items()], ignore_index=True
    ).to_csv(out / "scenario_yearly.csv", index=False, encoding="utf-8-sig")

    no_match_summary = account_summary(no_match_frame)
    checks = {
        "central_path_positive_24h_every_year": bool(
            not central_yearly.empty and (central_yearly["aligned_return_24h"] > 0).all()
        ),
        "central_path_positive_72h_every_year": bool(
            not central_yearly.empty and (central_yearly["aligned_return_72h"] > 0).all()
        ),
        "no_penalty_beats_exposure_matched_calmar": bool(
            no_summary["calmar"] > no_match_summary["calmar"]
        ),
        "no_penalty_cagr_gt_abs_mdd": bool(
            no_summary["cagr"] > abs(no_summary["max_drawdown"])
        ),
        "no_penalty_cost2x_positive": bool(account_summary(no_penalty_cost2)["total_return"] > 0),
    }
    verdict = {
        "pass": False,
        "decision": "CONTINUE_FALSE_EXPANSION_DIAGNOSTICS_NOT_LIVE_PROMOTION",
        "checks": checks,
        "reason": (
            "Path-aware exposure adds economic value versus an exposure-matched static control, "
            "but capital efficiency still fails because CAGR does not exceed |MDD|.  The path was "
            "also discovered on this same sample, so no live claim is permitted."
        ),
        "path_rule": rule.__dict__,
    }
    (out / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    (out / "risk_match_selected.json").write_text(
        json.dumps({"no_penalty": no_match, "reward": reward_match}, indent=2, default=str),
        encoding="utf-8",
    )
    (out / "run_config.json").write_text(
        json.dumps({"source_config": config, "path_rule": rule.__dict__}, indent=2), encoding="utf-8"
    )

    _write_report(
        out,
        rule=rule,
        scenarios=scenarios,
        path_yearly=central_yearly,
        neighborhood_summary=neighborhood_summary,
        risk_matches={"no_penalty": no_match, "reward": reward_match},
        verdict=verdict,
    )

    print("=" * 96)
    print("RDPOS-03 PATH-AWARE POSITIONING COUNTERFACTUAL")
    print("=" * 96)
    print(scenarios[["scenario", "total_return", "cagr", "max_drawdown", "calmar", "mean_gross_exposure"]].to_string(index=False))
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
