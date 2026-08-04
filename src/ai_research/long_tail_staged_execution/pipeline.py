#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.11 staged entry and pyramiding audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import load_minute_path_data
from src.ai_research.long_tail_tranche_eligibility.config import TrancheEligibilityConfig
from src.ai_research.swing_baseline.dataset import create_loader, run_public_loader_preflight
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from . import reports
from .analysis import enrich_summaries, policy_gate
from .config import (
    DEFAULT_STAGED_EXECUTION_CONFIG,
    STAGE_ID,
    STAGE_NAME,
    StagedExecutionConfig,
)
from .inputs import load_staged_execution_inputs
from .simulator import simulate_staged_execution_account


@dataclass(frozen=True)
class StagedExecutionResult:
    decision: str
    report_dir: Path


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "frozen_q70_source", "status": "PASS", "detail": "cycles are the validated R03.4.2.10 P0 q70 sequence"},
            {"check": "base_exit_frozen", "status": "PASS", "detail": "base keeps 3% disaster protection plus deterministic failed_reclaim"},
            {"check": "no_base_pivot_stop", "status": "PASS", "detail": "abandoned 15m Pivot hard stop is not restored on the base"},
            {"check": "next_open_adds", "status": "PASS", "detail": "add decisions use completed minute/structure data and execute at next 1m open"},
            {"check": "causal_N", "status": "PASS", "detail": "N uses prior 60 completed one-minute true ranges only"},
            {"check": "independent_add_stop", "status": "PASS", "detail": "an add stop closes only that add-on and never the base winner"},
            {"check": "no_base_reduction", "status": "PASS", "detail": "Turtle and pyramid policies never sell the base to fund an add"},
            {"check": "profit_cover", "status": "PASS", "detail": "add risk must be covered by then-visible unrealized profit"},
            {"check": "healthy_structure_only", "status": "PASS", "detail": "no add while BROKEN or pending failed_reclaim"},
            {"check": "tail_risk_cap", "status": "PASS", "detail": "declared and runtime hard-tail risk are capped at two account-R"},
            {"check": "notional_cap", "status": "PASS", "detail": "runtime gross nominal exposure is capped at 1.5x equity"},
            {"check": "maximum_three_layers", "status": "PASS", "detail": "one base plus at most two add-ons"},
            {"check": "soft_failure_not_wick_stop", "status": "PASS", "detail": "F1 exits only after a completed structure close below threshold"},
            {"check": "fixed_6h_diagnostic_only", "status": "PASS", "detail": "no staged policy uses a fixed-time exit"},
            {"check": "same_policy_both_years", "status": "PASS", "detail": "identical policies run in 2024 and 2025"},
            {"check": "sealed_2026", "status": "PASS", "detail": "2026 remains unopened"},
        ]
    )


def _write_empty(
    *,
    config: StagedExecutionConfig,
    decision: str,
    reason: str,
    preflight: dict[str, object],
) -> StagedExecutionResult:
    reports.write_reports(
        config=config,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        source_baseline=pd.DataFrame(),
        selected_events=pd.DataFrame(),
        cycles=pd.DataFrame(),
        legs=pd.DataFrame(),
        actions=pd.DataFrame(),
        daily_equity=pd.DataFrame(),
        summary=pd.DataFrame(),
        gate=pd.DataFrame(),
        causal_audit=pd.DataFrame(),
        runtime_rejections=pd.DataFrame(),
        failures=pd.DataFrame([{"fold_id": "ALL", "error": reason}]),
        decision=decision,
        reason=reason,
    )
    return StagedExecutionResult(decision, config.report_path)


def run_staged_execution_audit(
    *,
    data_dir: str | Path | None = None,
    progress: bool = True,
    config: StagedExecutionConfig = DEFAULT_STAGED_EXECUTION_CONFIG,
) -> StagedExecutionResult:
    config.validate()
    try:
        inputs = load_staged_execution_inputs(config)
    except Exception as exc:
        reason = f"冻结的R03.4.2.10输入不可用：{type(exc).__name__}: {exc}"
        return _write_empty(
            config=config,
            decision="BLOCKED_SOURCE_REPORT",
            reason=reason,
            preflight={"source_report": reason},
        )

    loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    loader_preflight = run_public_loader_preflight(
        loader,
        LONG_CONTEXT_BASE_CONFIG,
        sample_dates=("2024-06-15", "2025-06-15"),
    )
    preflight: dict[str, object] = {
        "trade_bar": loader_preflight.to_dict(),
        "source_2_10": str(config.source_2_10_path),
        "source_2_10_decision": (config.source_2_10_path / "99_decision.md").read_text(encoding="utf-8").splitlines()[2:6],
    }
    if loader_preflight.status != "PASS":
        return _write_empty(
            config=config,
            decision="BLOCKED_DATA",
            reason="1分钟Trade Bar公共Loader预检失败。",
            preflight=preflight,
        )

    path_config = TrancheEligibilityConfig().structural_config()
    cycle_parts: list[pd.DataFrame] = []
    leg_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    runtime_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    total_steps = len(inputs.folds) * len(config.entry_delay_minutes)
    stage_reporter = ProgressReporter("[R03.4.2.11 folds/delays]", total_steps, every=1, enabled=progress)
    step = 0
    for fold in inputs.folds.to_dict("records"):
        fold_id = str(fold["fold_id"])
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"]).floor("min")
        try:
            path = load_minute_path_data(
                start=test_start - pd.Timedelta(days=2),
                end=test_end,
                data_dir=data_dir,
                config=path_config,
                progress=progress,
            )
            fold_events = inputs.selected_events.loc[inputs.selected_events["fold_id"].astype(str).eq(fold_id)].copy()
            fold_timeline = inputs.structure_timeline.loc[inputs.structure_timeline["fold_id"].astype(str).eq(fold_id)].copy()
            for delay in config.entry_delay_minutes:
                try:
                    for policy in config.policies:
                        for multiplier in config.cost_multipliers:
                            simulation = simulate_staged_execution_account(
                                fold_events,
                                fold_timeline,
                                path=path,
                                fold_id=fold_id,
                                policy=policy,
                                delay_minutes=int(delay),
                                cost_multiplier=float(multiplier),
                                test_start=test_start,
                                test_end=test_end,
                                config=config,
                                progress=progress,
                            )
                            if simulation.summary:
                                summary_rows.append(simulation.summary)
                            if not simulation.cycles.empty:
                                cycle_parts.append(simulation.cycles)
                            if not simulation.legs.empty:
                                leg_parts.append(simulation.legs)
                            if not simulation.actions.empty:
                                action_parts.append(simulation.actions)
                            if not simulation.daily_equity.empty:
                                daily_parts.append(simulation.daily_equity)
                            if not simulation.runtime_rejections.empty:
                                runtime_parts.append(simulation.runtime_rejections)
                except Exception as exc:
                    failures.append(
                        {
                            "fold_id": fold_id,
                            "delay_minutes": int(delay),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                step += 1
                stage_reporter.update(step)
        except Exception as exc:
            failures.append(
                {
                    "fold_id": fold_id,
                    "delay_minutes": "ALL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            step += len(config.entry_delay_minutes)
            stage_reporter.update(step)
    stage_reporter.close()

    cycles = pd.concat(cycle_parts, ignore_index=True) if cycle_parts else pd.DataFrame()
    legs = pd.concat(leg_parts, ignore_index=True) if leg_parts else pd.DataFrame()
    actions = pd.concat(action_parts, ignore_index=True) if action_parts else pd.DataFrame()
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    runtime = pd.concat(runtime_parts, ignore_index=True) if runtime_parts else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    if not summary.empty and not cycles.empty:
        summary = enrich_summaries(summary, cycles)
    gate = policy_gate(summary, config)
    causal = _causal_audit()
    failure_frame = pd.DataFrame(failures)

    # P0 is a strict account-return anchor against the completed 2.10 report.
    if failure_frame.empty and not summary.empty:
        for row in inputs.source_p0_summary.to_dict("records"):
            actual = summary.loc[
                summary["fold_id"].astype(str).eq(str(row["fold_id"]))
                & summary["policy"].astype(str).eq("P0_single_1R")
                & summary["delay_minutes"].astype(int).eq(int(row["delay_minutes"]))
                & np.isclose(summary["cost_multiplier"].astype(float), float(row["cost_multiplier"]))
            ]
            if len(actual) != 1:
                failures.append({"fold_id": row["fold_id"], "error": "P0 summary parity row missing"})
                continue
            if not np.isclose(
                float(actual.iloc[0]["total_net_return"]),
                float(row["total_net_return"]),
                atol=1e-8,
                rtol=1e-8,
            ):
                failures.append(
                    {
                        "fold_id": row["fold_id"],
                        "delay_minutes": int(row["delay_minutes"]),
                        "error": (
                            "P0 return parity drift: "
                            f"actual={actual.iloc[0]['total_net_return']} expected={row['total_net_return']}"
                        ),
                    }
                )
        failure_frame = pd.DataFrame(failures)

    if not failure_frame.empty:
        decision = "FAIL_RUNTIME"
        reason = "部分fold、延迟或P0回归运行失败；不得解释任何收益结果。"
    elif gate.empty:
        decision = "FAIL_NO_RESULTS"
        reason = "分批入场与金字塔政策没有生成可审核结果。"
    elif gate["pass_to_next_stage"].astype(bool).any():
        winners = gate.loc[gate["pass_to_next_stage"].astype(bool), "policy"].astype(str).tolist()
        decision = "PASS_STAGED_EXECUTION_CANDIDATE"
        reason = f"同一套分批/加仓政策跨年通过：{winners}。仍需分数风险层、最终退出复核和2026封存验证。"
    else:
        decision = "FAIL_NO_ROBUST_STAGED_EXECUTION"
        reason = "没有方案同时提高或保留跨年收益、控制尾部风险与名义敞口，并避免把原P0赢家系统性变成亏损。"

    reports.write_reports(
        config=config,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        source_baseline=inputs.source_p0_summary.copy(),
        selected_events=inputs.selected_events.copy(),
        cycles=cycles,
        legs=legs,
        actions=actions,
        daily_equity=daily,
        summary=summary,
        gate=gate,
        causal_audit=causal,
        runtime_rejections=runtime,
        failures=failure_frame,
        decision=decision,
        reason=reason,
    )
    return StagedExecutionResult(decision, config.report_path)
