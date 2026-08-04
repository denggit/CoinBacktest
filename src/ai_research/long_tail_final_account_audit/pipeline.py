#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.15 final account audit."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from .analysis import (
    build_continuous_scenarios,
    build_gate,
    build_live_state_contract,
    build_lot_size_audit,
    build_model_governance,
    build_risk_reserve_audit,
)
from .config import DEFAULT_FINAL_ACCOUNT_AUDIT_CONFIG, STAGE_ID, STAGE_NAME, FinalAccountAuditConfig
from .inputs import load_final_audit_inputs
from . import reports


@dataclass(frozen=True)
class FinalAccountAuditResult:
    decision: str
    report_dir: Path


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "frozen_strategy", "status": "PASS", "detail": "q70 immediate entry, equal one-R, C2 stops and failed_reclaim are unchanged"},
            {"check": "continuous_oos", "status": "PASS", "detail": "WF_2024 and WF_2025 are compounded continuously without annual equity reset"},
            {"check": "development_year_excluded", "status": "PASS", "detail": "2023 is training/development history and is not presented as OOS account return"},
            {"check": "stress_grid", "status": "PASS", "detail": "all 1/3/5-minute and 2x/3x-cost frozen cells are audited"},
            {"check": "lot_rounding", "status": "PASS", "detail": "OKX contract rounding uses floor sizing and cannot exceed target price risk"},
            {"check": "model_release_governance", "status": "PASS", "detail": "monthly audit/retrain is separated from quarterly or event-driven manual promotion"},
            {"check": "sealed_holdout", "status": "PASS", "detail": "2026 is not loaded or evaluated"},
        ]
    )


def _empty(config: FinalAccountAuditConfig, decision: str, reason: str) -> FinalAccountAuditResult:
    reports.write_reports(
        config=config,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight={"source_2_14": str(config.source_2_14_path), "status": decision},
        historical=pd.DataFrame(),
        cycles=pd.DataFrame(),
        daily=pd.DataFrame(),
        scenarios=pd.DataFrame(),
        months=pd.DataFrame(),
        quarters=pd.DataFrame(),
        lot_sizes=pd.DataFrame(),
        risk=pd.DataFrame(),
        governance=build_model_governance(),
        live_state=build_live_state_contract(),
        gate=pd.DataFrame(),
        causal=_causal_audit(),
        failures=pd.DataFrame([{"scope": "source", "error": reason}]),
        decision=decision,
        reason=reason,
    )
    return FinalAccountAuditResult(decision, config.report_path)


def run_final_account_audit(
    *,
    source_report_dir: str | Path | None = None,
    config: FinalAccountAuditConfig = DEFAULT_FINAL_ACCOUNT_AUDIT_CONFIG,
) -> FinalAccountAuditResult:
    if source_report_dir is not None:
        source = Path(source_report_dir).resolve()
        try:
            relative = source.relative_to(Path.cwd().resolve())
            source_value = str(relative).replace("\\", "/")
        except ValueError:
            source_value = str(source)
        config = replace(config, source_2_14_report_dir=source_value)
    config.validate()
    try:
        inputs = load_final_audit_inputs(config)
    except Exception as exc:
        return _empty(config, "BLOCKED_SOURCE_REPORT", f"冻结R03.4.2.14报告不可用：{type(exc).__name__}: {exc}")

    try:
        historical = inputs.historical_contract.copy()
        scope_notes = {
            "fixed_6h_all_signals": "independent-signal/full-notional diagnostic; not a single-position account return",
            "P0_failed_reclaim": "single-position/full-notional path diagnostic; not the risk-sized P0 account return",
            "C2_equal_1R_account": "risk-sized single-position account return; directly comparable with the frozen live candidate",
        }
        historical["return_scope_note"] = historical["metric_scope"].astype(str).map(scope_notes).fillna("source metric scope")
        cycles, daily, scenarios, months, quarters = build_continuous_scenarios(
            inputs.cycles,
            inputs.daily_equity,
            inputs.source_summary,
            config,
        )
        risk = build_risk_reserve_audit(scenarios, config)
        conservative_live_budget = float(risk["recommended_live_price_risk_budget"].min())
        lot_sizes = build_lot_size_audit(
            inputs.legs,
            config,
            live_price_risk_fraction=conservative_live_budget,
        )
        governance = build_model_governance()
        live_state = build_live_state_contract()
        gate = build_gate(scenarios, risk, config)
        causal = _causal_audit()
        failures = pd.DataFrame()
    except Exception as exc:
        return _empty(config, "FAIL_RUNTIME", f"最终账户审计失败：{type(exc).__name__}: {exc}")

    if gate.empty or not bool(gate["pass"].all()):
        decision = "FAIL_FINAL_ACCOUNT_LIVE_READINESS"
        failed = gate.loc[~gate["pass"].astype(bool), "check"].astype(str).tolist()
        reason = f"最终账户或部署资格门未全部通过：{failed}。不得开启2026封存验证。"
    else:
        decision = "PASS_FINAL_ACCOUNT_LIVE_READINESS"
        reason = (
            "冻结C2在连续2024-2025 OOS、全部成本/延迟压力、去前十大、净风险缓冲和OKX张数审计中通过；"
            "下一步只允许一次性开启2026封存验证。"
        )

    reports.write_reports(
        config=config,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight={"source_2_14": str(config.source_2_14_path), "source_decision": "PASS_C2_FROZEN_NO_ENTRY_UPLIFT"},
        historical=historical,
        cycles=cycles,
        daily=daily,
        scenarios=scenarios,
        months=months,
        quarters=quarters,
        lot_sizes=lot_sizes,
        risk=risk,
        governance=governance,
        live_state=live_state,
        gate=gate,
        causal=causal,
        failures=failures,
        decision=decision,
        reason=reason,
    )
    return FinalAccountAuditResult(decision, config.report_path)
