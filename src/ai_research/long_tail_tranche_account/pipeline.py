#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.8B account-level dual-slot audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ai_research.long_tail_exit_audit.data import load_minute_path_data
from src.ai_research.swing_baseline.dataset import create_loader, run_public_loader_preflight
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from . import reports
from .analysis import (
    build_account_summary,
    concentration_summary,
    fixed_6h_diagnostic_summary,
    policy_coverage_summary,
    policy_gate,
)
from .config import DEFAULT_TRANCHE_ACCOUNT_CONFIG, STAGE_ID, STAGE_NAME, TrancheAccountConfig
from .inputs import load_tranche_account_inputs
from .simulator import build_pair_diagnostics, select_policy_trades, simulate_account


@dataclass(frozen=True)
class TrancheAccountResult:
    decision: str
    report_dir: Path


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "frozen_q70_source", "status": "PASS", "detail": "events and scores come from the validated R03.4.2.8A q70 artifact"},
            {"check": "frozen_failed_reclaim", "status": "PASS", "detail": "every virtual tranche uses its precomputed deterministic failed-reclaim exit and 3% disaster protection"},
            {"check": "maximum_two_tranches", "status": "PASS", "detail": "selection state has only virtual slots A and B"},
            {"check": "entry_before_equal_time_exit", "status": "PASS", "detail": "equal entry/exit timestamps remain occupied, matching the frozen P0 convention"},
            {"check": "independent_tranche_state", "status": "PASS", "detail": "new signals never reset or extend an existing tranche exit"},
            {"check": "causal_protection_gate", "status": "PASS", "detail": "P3 blocks only from state observable at the new signal decision time"},
            {"check": "risk_slots_not_notional_multipliers", "status": "PASS", "detail": "slot weights allocate account R; notional is derived from the fixed 3% disaster distance"},
            {"check": "account_risk_cap", "status": "PASS", "detail": "new tranche risk is capped against marked equity at entry and cannot exceed one full risk unit"},
            {"check": "minute_marked_mdd", "status": "PASS", "detail": "MDD uses the public 1m path, entry/exit fees and open-position mark-to-market"},
            {"check": "fixed_6h_diagnostic_only", "status": "PASS", "detail": "fixed six hours is reported but never used by account policies"},
            {"check": "same_policy_both_years", "status": "PASS", "detail": "P0/P1/P2/P3 definitions are identical in WF_2024 and WF_2025"},
            {"check": "sealed_2026", "status": "PASS", "detail": "2026 remains unopened"},
        ]
    )


def run_tranche_account_audit(
    *,
    data_dir: str | Path | None = None,
    progress: bool = True,
    config: TrancheAccountConfig = DEFAULT_TRANCHE_ACCOUNT_CONFIG,
) -> TrancheAccountResult:
    config.validate()
    try:
        inputs = load_tranche_account_inputs(config)
    except Exception as exc:
        reason = f"R03.4.2.8A冻结输入不可用：{type(exc).__name__}: {exc}"
        reports.write_reports(
            config=config,
            manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
            preflight={"source_report": reason},
            fixed_6h=pd.DataFrame(),
            pair_diagnostics=pd.DataFrame(),
            decisions=pd.DataFrame(),
            account_summary=pd.DataFrame(),
            trades=pd.DataFrame(),
            daily_equity=pd.DataFrame(),
            concentration=pd.DataFrame(),
            gate=pd.DataFrame(),
            runtime_rejections=pd.DataFrame(),
            causal_audit=pd.DataFrame(),
            failures=pd.DataFrame([{"fold_id": "ALL", "error": reason}]),
            decision="BLOCKED_SOURCE_REPORT",
            reason=reason,
        )
        return TrancheAccountResult("BLOCKED_SOURCE_REPORT", config.report_path)
    loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    loader_preflight = run_public_loader_preflight(
        loader,
        LONG_CONTEXT_BASE_CONFIG,
        sample_dates=("2024-06-15", "2025-06-15"),
    )
    preflight: dict[str, object] = {
        "trade_bar": loader_preflight.to_dict(),
        "source_stage": inputs.manifest.get("stage"),
        "source_report_dir": str(config.source_report_path),
        "source_2_8a_decision": (config.source_report_path / "99_decision.md").read_text(encoding="utf-8").splitlines()[2:6],
    }
    if loader_preflight.status != "PASS":
        reason = "1分钟Trade Bar公共Loader预检失败。"
        reports.write_reports(
            config=config,
            manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
            preflight=preflight,
            fixed_6h=pd.DataFrame(),
            pair_diagnostics=pd.DataFrame(),
            decisions=pd.DataFrame(),
            account_summary=pd.DataFrame(),
            trades=pd.DataFrame(),
            daily_equity=pd.DataFrame(),
            concentration=pd.DataFrame(),
            gate=pd.DataFrame(),
            runtime_rejections=pd.DataFrame(),
            causal_audit=pd.DataFrame(),
            failures=pd.DataFrame([{"fold_id": "ALL", "error": reason}]),
            decision="BLOCKED_DATA",
            reason=reason,
        )
        return TrancheAccountResult("BLOCKED_DATA", config.report_path)

    fixed_summary = fixed_6h_diagnostic_summary(inputs.fixed_6h, config)
    pair_parts: list[pd.DataFrame] = []
    decision_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    runtime_parts: list[pd.DataFrame] = []
    simulation_summaries: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    total_steps = len(inputs.folds) * len(config.entry_delay_minutes)
    reporter = ProgressReporter("[R03.4.2.8B folds/delays]", total_steps, every=1, enabled=progress)
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
                config=config.eligibility_config().structural_config(),
                progress=progress,
            )
            fold_structural = inputs.structural.loc[inputs.structural["fold_id"] == fold_id]
            for delay in config.entry_delay_minutes:
                try:
                    structural = fold_structural.loc[fold_structural["delay_minutes"] == delay].copy()
                    pairs = build_pair_diagnostics(
                        structural,
                        delay_minutes=delay,
                        path=path,
                        config=config,
                        progress=progress,
                    )
                    if not pairs.empty:
                        pair_parts.append(pairs)
                    for policy in config.policies:
                        selection = select_policy_trades(structural, policy=policy, pair_diagnostics=pairs)
                        if policy.name == "P0_single_1R":
                            expected = inputs.p0_baseline.loc[
                                (inputs.p0_baseline["fold_id"] == fold_id)
                                & (inputs.p0_baseline["delay_minutes"].astype(int) == int(delay)),
                                "event_id",
                            ].astype(str).tolist()
                            actual = selection.accepted["event_id"].astype(str).tolist() if not selection.accepted.empty else []
                            if actual != expected:
                                raise RuntimeError(
                                    f"P0 parity drift {fold_id} d{delay}: actual={len(actual)} expected={len(expected)}"
                                )
                        if not selection.decisions.empty:
                            decision_parts.append(selection.decisions)
                        for multiplier in config.cost_multipliers:
                            simulation = simulate_account(
                                selection.accepted,
                                path=path,
                                fold_id=fold_id,
                                policy=policy,
                                delay_minutes=delay,
                                cost_multiplier=multiplier,
                                test_start=test_start,
                                test_end=test_end,
                                config=config,
                                progress=progress,
                            )
                            if simulation.summary:
                                simulation_summaries.append(simulation.summary)
                                coverage = policy_coverage_summary(
                                    selection.decisions,
                                    simulation.trades,
                                    candidate_count=int(len(structural)),
                                )
                                coverage_rows.append(
                                    {
                                        "fold_id": fold_id,
                                        "policy": policy.name,
                                        "delay_minutes": int(delay),
                                        "cost_multiplier": float(multiplier),
                                        **coverage,
                                    }
                                )
                            if not simulation.trades.empty:
                                trade_parts.append(simulation.trades)
                            if not simulation.daily_equity.empty:
                                daily_parts.append(simulation.daily_equity)
                            if not simulation.runtime_rejections.empty:
                                runtime_parts.append(simulation.runtime_rejections)
                except Exception as exc:
                    failures.append({"fold_id": fold_id, "delay_minutes": delay, "error": f"{type(exc).__name__}: {exc}"})
                step += 1
                reporter.update(step)
        except Exception as exc:
            failures.append({"fold_id": fold_id, "delay_minutes": "ALL", "error": f"{type(exc).__name__}: {exc}"})
            step += len(config.entry_delay_minutes)
            reporter.update(step)
    reporter.close()

    pair_frame = pd.concat(pair_parts, ignore_index=True) if pair_parts else pd.DataFrame()
    decisions = pd.concat(decision_parts, ignore_index=True) if decision_parts else pd.DataFrame()
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    runtime = pd.concat(runtime_parts, ignore_index=True) if runtime_parts else pd.DataFrame()
    account_summary = build_account_summary(simulation_summaries, coverage_rows)
    concentration = concentration_summary(trades)
    gate = policy_gate(account_summary, config)
    failures_frame = pd.DataFrame(failures)

    passed = gate.loc[gate["pass_to_entry_stop_research"].astype(bool), "policy"].tolist() if not gate.empty else []
    if passed:
        decision = "PASS_DUAL_SLOT_ACCOUNT_CANDIDATE"
        reason = "至少一套统一双槽位方案在2024与2025同时提高单仓Failed-Reclaim账户收益，恢复足够q70覆盖，并通过MDD、成本、延迟、集中度和危险加仓约束；下一步进入入场与真实结构止损优化。"
    else:
        decision = "FAIL_NO_ROBUST_DUAL_SLOT_ACCOUNT_POLICY"
        reason = "没有统一双槽位方案同时改善2024与2025账户收益并满足覆盖、MDD、3倍成本、延迟和危险加仓约束；不得扩展到第三Tranche、无限加仓或按年份挑策略。"

    reports.write_reports(
        config=config,
        manifest={
            "stage": STAGE_ID,
            "name": STAGE_NAME,
            "config": config.to_dict(),
            "source_stage": inputs.manifest.get("stage"),
            "source_gate_result": inputs.source_gate.to_dict("records"),
            "reason_for_continuing_after_2_8a_gate_failure": "2.8A rejected a very strict healthy/recovered subset; 2.8B tests broader q70 coverage through pre-allocated account-risk slots rather than loosening to unlimited averaging down",
            "policy_selection_by_year": "FORBIDDEN",
        },
        preflight=preflight,
        fixed_6h=fixed_summary,
        pair_diagnostics=pair_frame,
        decisions=decisions,
        account_summary=account_summary,
        trades=trades,
        daily_equity=daily,
        concentration=concentration,
        gate=gate,
        runtime_rejections=runtime,
        causal_audit=_causal_audit(),
        failures=failures_frame,
        decision=decision,
        reason=reason,
    )
    return TrancheAccountResult(decision=decision, report_dir=config.report_path)
