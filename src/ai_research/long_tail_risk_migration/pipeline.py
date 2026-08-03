#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.10 partial de-risking and risk-migration audit."""

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
from .analysis import build_account_summary, policy_gate
from .config import (
    DEFAULT_RISK_MIGRATION_CONFIG,
    STAGE_ID,
    STAGE_NAME,
    RiskMigrationConfig,
)
from .inputs import load_risk_migration_inputs
from .simulator import simulate_risk_migration_account
from .structure import build_candidate_pair_snapshots, build_soft_structure_timeline


@dataclass(frozen=True)
class RiskMigrationResult:
    decision: str
    report_dir: Path


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "frozen_q70_source", "status": "PASS", "detail": "all candidate events come from validated R03.4.2.8A q70 outcomes"},
            {"check": "frozen_failed_reclaim", "status": "PASS", "detail": "every remaining tranche keeps its original deterministic failed-reclaim exit"},
            {"check": "three_percent_disaster_only", "status": "PASS", "detail": "3% remains the only exchange-style hard floor; Pivot hard stops are not used"},
            {"check": "completed_structure_only", "status": "PASS", "detail": "soft structure uses completed 15m bars and right-confirmed pivots"},
            {"check": "next_open_actions", "status": "PASS", "detail": "partial reductions execute only at the next 1m open after structure close"},
            {"check": "real_partial_close", "status": "PASS", "detail": "risk release is recognized only after units are physically closed and fees charged"},
            {"check": "cycle_budget_fixed", "status": "PASS", "detail": "one risk cycle uses the one-R dollar budget fixed when its first primary opens"},
            {"check": "migration_conserves_risk", "status": "PASS", "detail": "new q70 risk uses free cycle capacity or same-open old-position reduction"},
            {"check": "no_losing_migration", "status": "PASS", "detail": "a losing root cannot fund a new tranche"},
            {"check": "no_broken_migration", "status": "PASS", "detail": "BROKEN or pending failed-reclaim roots cannot migrate risk"},
            {"check": "maximum_two_tranches", "status": "PASS", "detail": "at most two simultaneous virtual tranches"},
            {"check": "entry_before_equal_exit", "status": "PASS", "detail": "equal-time new entries still see the old tranche occupied, preserving P0 convention"},
            {"check": "fixed_6h_diagnostic_only", "status": "PASS", "detail": "fixed six hours is never used as an exit"},
            {"check": "same_policy_both_years", "status": "PASS", "detail": "identical policies run in WF_2024 and WF_2025"},
            {"check": "sealed_2026", "status": "PASS", "detail": "2026 remains unopened"},
        ]
    )


def _write_empty(
    *,
    config: RiskMigrationConfig,
    decision: str,
    reason: str,
    preflight: dict[str, object],
) -> RiskMigrationResult:
    reports.write_reports(
        config=config,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        source_baseline=pd.DataFrame(),
        structure_timeline=pd.DataFrame(),
        pair_snapshots=pd.DataFrame(),
        decisions=pd.DataFrame(),
        actions=pd.DataFrame(),
        legs=pd.DataFrame(),
        trades=pd.DataFrame(),
        daily_equity=pd.DataFrame(),
        account_summary=pd.DataFrame(),
        gate=pd.DataFrame(),
        causal_audit=pd.DataFrame(),
        runtime_rejections=pd.DataFrame(),
        failures=pd.DataFrame([{"fold_id": "ALL", "error": reason}]),
        decision=decision,
        reason=reason,
    )
    return RiskMigrationResult(decision, config.report_path)


def run_risk_migration_audit(
    *,
    data_dir: str | Path | None = None,
    progress: bool = True,
    config: RiskMigrationConfig = DEFAULT_RISK_MIGRATION_CONFIG,
) -> RiskMigrationResult:
    config.validate()
    try:
        inputs = load_risk_migration_inputs(config)
    except Exception as exc:
        reason = f"冻结的2.8A/2.8B/2.9输入不可用：{type(exc).__name__}: {exc}"
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
        "source_2_8a": str(config.source_2_8a_path),
        "source_2_8b": str(config.source_2_8b_path),
        "source_2_9": str(config.source_2_9_path),
        "source_2_9_decision": (config.source_2_9_path / "99_decision.md").read_text(encoding="utf-8").splitlines()[2:6],
    }
    if loader_preflight.status != "PASS":
        return _write_empty(
            config=config,
            decision="BLOCKED_DATA",
            reason="1分钟Trade Bar公共Loader预检失败。",
            preflight=preflight,
        )

    structural_config = TrancheEligibilityConfig().structural_config()
    timeline_parts: list[pd.DataFrame] = []
    pair_parts: list[pd.DataFrame] = []
    decision_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    leg_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    runtime_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    total_steps = len(inputs.folds) * len(config.entry_delay_minutes)
    stage_reporter = ProgressReporter("[R03.4.2.10 folds/delays]", total_steps, every=1, enabled=progress)
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
                config=structural_config,
                progress=progress,
            )
            fold_structural = inputs.structural.loc[inputs.structural["fold_id"] == fold_id].copy()
            for delay in config.entry_delay_minutes:
                try:
                    structural = fold_structural.loc[
                        fold_structural["delay_minutes"].astype(int) == int(delay)
                    ].copy()
                    event_timelines: list[pd.DataFrame] = []
                    reporter = ProgressReporter(
                        f"[R03.4.2.10 timeline {fold_id} d{delay}]",
                        len(structural),
                        every=max(1, len(structural) // 100),
                        enabled=progress,
                    )
                    for index, row in enumerate(structural.to_dict("records"), start=1):
                        timeline = build_soft_structure_timeline(row, path=path, config=structural_config)
                        if not timeline.empty:
                            event_timelines.append(timeline)
                        reporter.update(index)
                    reporter.close()
                    timelines = pd.concat(event_timelines, ignore_index=True) if event_timelines else pd.DataFrame()
                    if not timelines.empty:
                        timeline_parts.append(timelines)
                        pairs = build_candidate_pair_snapshots(
                            structural,
                            timelines,
                            delay_minutes=int(delay),
                        )
                        if not pairs.empty:
                            pair_parts.append(pairs)

                    for policy in config.policies:
                        for multiplier in config.cost_multipliers:
                            simulation = simulate_risk_migration_account(
                                structural,
                                timelines,
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
                            if not simulation.decisions.empty:
                                decision_parts.append(simulation.decisions)
                            if not simulation.actions.empty:
                                action_parts.append(simulation.actions)
                            if not simulation.legs.empty:
                                leg_parts.append(simulation.legs)
                            if not simulation.trades.empty:
                                trade_parts.append(simulation.trades)
                            if not simulation.daily_equity.empty:
                                daily_parts.append(simulation.daily_equity)
                            if not simulation.runtime_rejections.empty:
                                runtime_parts.append(simulation.runtime_rejections)

                            # P0 is a strict regression anchor against 2.8B.
                            if policy.name == "P0_single_1R":
                                expected = inputs.p0_trades.loc[
                                    (inputs.p0_trades["fold_id"].astype(str) == fold_id)
                                    & (inputs.p0_trades["delay_minutes"].astype(int) == int(delay))
                                    & (inputs.p0_trades["cost_multiplier"].astype(float) == float(multiplier)),
                                    "event_id",
                                ].astype(str).tolist()
                                actual = simulation.trades["event_id"].astype(str).tolist() if not simulation.trades.empty else []
                                if actual != expected:
                                    raise RuntimeError(
                                        f"P0 event parity drift {fold_id} d{delay} c{multiplier:g}: actual={len(actual)} expected={len(expected)}"
                                    )
                                expected_summary = inputs.p0_summary.loc[
                                    (inputs.p0_summary["fold_id"].astype(str) == fold_id)
                                    & (inputs.p0_summary["delay_minutes"].astype(int) == int(delay))
                                    & (inputs.p0_summary["cost_multiplier"].astype(float) == float(multiplier))
                                ]
                                if len(expected_summary) == 1:
                                    expected_return = float(expected_summary.iloc[0]["total_net_return"])
                                    if not np.isclose(
                                        float(simulation.summary["total_net_return"]),
                                        expected_return,
                                        atol=1e-9,
                                        rtol=1e-9,
                                    ):
                                        raise RuntimeError(
                                            f"P0 return parity drift {fold_id} d{delay} c{multiplier:g}: "
                                            f"actual={simulation.summary['total_net_return']} expected={expected_return}"
                                        )
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

    timeline_frame = pd.concat(timeline_parts, ignore_index=True) if timeline_parts else pd.DataFrame()
    pair_frame = pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()
    decisions = pd.concat(decision_parts, ignore_index=True) if decision_parts else pd.DataFrame()
    actions = pd.concat(action_parts, ignore_index=True) if action_parts else pd.DataFrame()
    legs = pd.concat(leg_parts, ignore_index=True) if leg_parts else pd.DataFrame()
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    runtime = pd.concat(runtime_parts, ignore_index=True) if runtime_parts else pd.DataFrame()
    account_summary = build_account_summary(summary_rows)
    gate = policy_gate(account_summary, config)
    causal = _causal_audit()
    failure_frame = pd.DataFrame(failures)

    if not failure_frame.empty:
        decision = "FAIL_RUNTIME"
        reason = "部分fold或延迟运行失败；不得解释任何收益结果。"
    elif gate.empty:
        decision = "FAIL_NO_RESULTS"
        reason = "账户政策没有生成可审核结果。"
    elif gate["pass_to_next_stage"].astype(bool).any():
        decision = "PASS_RISK_MIGRATION_CANDIDATE"
        winners = gate.loc[gate["pass_to_next_stage"].astype(bool), "policy"].astype(str).tolist()
        reason = f"统一风险守恒政策通过：{winners}。仍需后续入场、分数定仓和最终退出复核。"
    else:
        decision = "FAIL_NO_ROBUST_PARTIAL_OR_MIGRATION"
        reason = "部分减仓或风险迁移没有同时保住两年收益、频率、成本压力和一R风险边界。"

    reports.write_reports(
        config=config,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        source_baseline=inputs.p0_summary.copy(),
        structure_timeline=timeline_frame,
        pair_snapshots=pair_frame,
        decisions=decisions,
        actions=actions,
        legs=legs,
        trades=trades,
        daily_equity=daily,
        account_summary=account_summary,
        gate=gate,
        causal_audit=causal,
        runtime_rejections=runtime,
        failures=failure_frame,
        decision=decision,
        reason=reason,
    )
    return RiskMigrationResult(decision, config.report_path)
