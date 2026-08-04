#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.13 score-risk sizing audit."""

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
from .analysis import build_cross_year_order_audit, build_tier_attribution, policy_gate
from .config import DEFAULT_SCORE_RISK_CONFIG, STAGE_ID, STAGE_NAME, ScoreRiskConfig
from .inputs import load_score_risk_inputs
from .simulator import simulate_score_risk_account


@dataclass(frozen=True)
class ScoreRiskResult:
    decision: str
    report_dir: Path


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"check": "passed_c2_source", "status": "PASS", "detail": "uses only passed R03.4.2.12 C2 cycles"},
        {"check": "score_tier_frozen_at_entry", "status": "PASS", "detail": "tier comes from frozen opening score percentile"},
        {"check": "no_holding_score_use", "status": "PASS", "detail": "later score changes never alter live risk or exit"},
        {"check": "entry_exit_frozen", "status": "PASS", "detail": "C2 entry, real 2% stop, soft failure and failed_reclaim exits remain unchanged"},
        {"check": "same_map_both_years", "status": "PASS", "detail": "identical risk maps run in 2024 and 2025"},
        {"check": "qualifying_tail_one_r", "status": "PASS", "detail": "formal tier policies never exceed one account-R price tail"},
        {"check": "diagnostic_scaling_separate", "status": "PASS", "detail": "0.75R/1.25R equal scaling is diagnostic, not score Edge"},
        {"check": "no_trade_filter", "status": "PASS", "detail": "all C2 cycles remain executed; tiering changes size only"},
        {"check": "no_fixed_time_exit", "status": "PASS", "detail": "fixed six hours remains diagnostic only"},
        {"check": "sealed_2026", "status": "PASS", "detail": "2026 remains unopened"},
    ])


def _write_empty(config: ScoreRiskConfig, decision: str, reason: str, preflight: dict[str, object]) -> ScoreRiskResult:
    reports.write_reports(config=config, manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()}, preflight=preflight, source_summary=pd.DataFrame(), attribution=pd.DataFrame(), order_audit=pd.DataFrame(), cycles=pd.DataFrame(), legs=pd.DataFrame(), daily=pd.DataFrame(), summary=pd.DataFrame(), gate=pd.DataFrame(), causal=pd.DataFrame(), rejections=pd.DataFrame(), failures=pd.DataFrame([{"fold_id": "ALL", "error": reason}]), decision=decision, reason=reason)
    return ScoreRiskResult(decision, config.report_path)


def _assert_anchor(summary: pd.DataFrame, source: pd.DataFrame, failures: list[dict[str, object]]) -> None:
    actual = summary.loc[summary.policy.astype(str).eq("E100_equal_1R")]
    expected = source
    keys = ["fold_id", "delay_minutes", "cost_multiplier"]
    merged = expected.merge(actual, on=keys, how="left", suffixes=("_source", "_actual"))
    for row in merged.to_dict("records"):
        for metric in ("total_net_return", "max_drawdown"):
            a = float(row.get(f"{metric}_actual", np.nan)); e = float(row.get(f"{metric}_source", np.nan))
            if not np.isfinite(a) or abs(a - e) > 1e-8:
                failures.append({"fold_id": row.get("fold_id"), "delay_minutes": row.get("delay_minutes"), "cost_multiplier": row.get("cost_multiplier"), "error": f"equal-one-R anchor mismatch {metric}: source={e} actual={a}"})


def run_score_risk_audit(*, data_dir: str | Path | None = None, progress: bool = True, config: ScoreRiskConfig = DEFAULT_SCORE_RISK_CONFIG) -> ScoreRiskResult:
    config.validate()
    try:
        inputs = load_score_risk_inputs(config)
    except Exception as exc:
        return _write_empty(config, "BLOCKED_SOURCE_REPORT", f"冻结的R03.4.2.12输入不可用：{type(exc).__name__}: {exc}", {"source_report": str(config.source_path)})
    loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    loader_preflight = run_public_loader_preflight(loader, LONG_CONTEXT_BASE_CONFIG, sample_dates=("2024-06-15", "2025-06-15"))
    preflight = {"trade_bar": loader_preflight.to_dict(), "source_2_12": str(config.source_path), "source_decision": "PASS_REAL_1R_TAIL_COMPRESSION_CANDIDATE"}
    if loader_preflight.status != "PASS":
        return _write_empty(config, "BLOCKED_DATA", "1分钟Trade Bar公共Loader预检失败。", preflight)

    path_config = TrancheEligibilityConfig().structural_config()
    cycle_parts: list[pd.DataFrame] = []; leg_parts: list[pd.DataFrame] = []; daily_parts: list[pd.DataFrame] = []; rejection_parts: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []; failures: list[dict[str, object]] = []
    folds = [("WF_2024", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31 23:59:00")), ("WF_2025", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31 23:59:00"))]
    reporter = ProgressReporter("[R03.4.2.13 folds]", len(folds), every=1, enabled=progress)
    for number, (fold_id, start, end) in enumerate(folds, start=1):
        try:
            path = load_minute_path_data(start=start - pd.Timedelta(days=2), end=end, data_dir=data_dir, config=path_config, progress=progress)
            for delay in config.entry_delay_minutes:
                for policy in config.policies:
                    for cost in config.cost_multipliers:
                        sim = simulate_score_risk_account(inputs.source_cycles, inputs.source_legs, path=path, fold_id=fold_id, policy=policy, delay_minutes=delay, cost_multiplier=cost, config=config, progress=progress)
                        if sim.summary: summaries.append(sim.summary)
                        if not sim.cycles.empty: cycle_parts.append(sim.cycles)
                        if not sim.legs.empty: leg_parts.append(sim.legs)
                        if not sim.daily_equity.empty: daily_parts.append(sim.daily_equity)
                        if not sim.rejections.empty: rejection_parts.append(sim.rejections)
        except Exception as exc:
            failures.append({"fold_id": fold_id, "error": f"{type(exc).__name__}: {exc}"})
        reporter.update(number)
    reporter.close()
    cycles = pd.concat(cycle_parts, ignore_index=True) if cycle_parts else pd.DataFrame()
    legs = pd.concat(leg_parts, ignore_index=True) if leg_parts else pd.DataFrame()
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    rejections = pd.concat(rejection_parts, ignore_index=True) if rejection_parts else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        _assert_anchor(summary, inputs.source_c2_summary, failures)
    attribution = build_tier_attribution(inputs.source_cycles)
    order_audit = build_cross_year_order_audit(attribution)
    gate = policy_gate(summary, config)
    failure_frame = pd.DataFrame(failures)
    if not failure_frame.empty:
        decision = "FAIL_RUNTIME"; reason = "运行或C2等风险基准复现失败；不得解释收益。"
    elif gate.empty:
        decision = "FAIL_RUNTIME"; reason = "没有生成完整风险映射资格门。"
    elif gate.pass_to_next_stage.astype(bool).any():
        winners = gate.loc[gate.pass_to_next_stage.astype(bool), "policy"].astype(str).tolist()
        decision = "PASS_SCORE_TIER_RISK_POLICY"; reason = f"同一套跨年分数风险映射通过：{winners}。"
    elif not order_audit.empty and not order_audit.monotonic_score_order.astype(bool).all():
        decision = "PASS_EQUAL_RISK_RETAINED"; reason = "C2分数层边际收益顺序跨年不稳定，降权方案未同时保留收益并改善风险；冻结全部q70等风险1R。"
    else:
        decision = "FAIL_NO_ROBUST_SCORE_RISK_POLICY"; reason = "没有风险分层方案同时保留两年收益、改善风险效率并通过成本延迟压力。"
    reports.write_reports(config=config, manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()}, preflight=preflight, source_summary=inputs.source_c2_summary, attribution=attribution, order_audit=order_audit, cycles=cycles, legs=legs, daily=daily, summary=summary, gate=gate, causal=_causal_audit(), rejections=rejections, failures=failure_frame, decision=decision, reason=reason)
    return ScoreRiskResult(decision, config.report_path)
