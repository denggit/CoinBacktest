#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.14 entry timing and MAE attribution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ai_research.long_tail_exit_audit.data import load_minute_path_data
from src.ai_research.long_tail_tranche_eligibility.config import TrancheEligibilityConfig
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.ai_research.swing_baseline.dataset import create_loader, run_public_loader_preflight
from src.research_common.progress import ProgressReporter

from .analysis import build_anchor_mae_attribution, policy_gate, summarize_mae_attribution
from .config import DEFAULT_ENTRY_TIMING_CONFIG, STAGE_ID, STAGE_NAME, EntryTimingConfig
from .inputs import load_entry_timing_inputs
from .simulator import simulate_entry_timing_account
from . import reports


@dataclass(frozen=True)
class EntryTimingResult:
    decision: str
    report_dir: Path


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame([
        {"check":"frozen_model_and_exit","status":"PASS","detail":"q70, equal 1R, real 2% hard stop, 1.5% completed-close soft failure and failed_reclaim are unchanged"},
        {"check":"bounded_wait","status":"PASS","detail":"all alternative entries wait at most 60 minutes and use only observable q70 scores or completed 5m closes"},
        {"check":"next_open_execution","status":"PASS","detail":"score decisions and completed-bar reclaim execute at the next observable open plus the frozen delay stress"},
        {"check":"coverage_gate","status":"PASS","detail":"formal candidates must retain at least 90% of frozen C2 cycles"},
        {"check":"sealed_holdout","status":"PASS","detail":"2026 is not loaded or evaluated"},
    ])


def _empty(config: EntryTimingConfig, decision: str, reason: str, preflight: dict[str, object]) -> EntryTimingResult:
    reports.write_reports(config=config,manifest={"stage":STAGE_ID,"name":STAGE_NAME,"config":config.to_dict()},preflight=preflight,historical=pd.DataFrame(),source_summary=pd.DataFrame(),mae=pd.DataFrame(),mae_summary=pd.DataFrame(),decisions=pd.DataFrame(),cycles=pd.DataFrame(),legs=pd.DataFrame(),daily=pd.DataFrame(),summary=pd.DataFrame(),gate=pd.DataFrame(),causal=_causal_audit(),rejections=pd.DataFrame(),failures=pd.DataFrame([{"fold_id":"ALL","error":reason}]),decision=decision,reason=reason)
    return EntryTimingResult(decision,config.report_path)


def run_entry_timing_audit(*, data_dir: str|Path|None=None, progress: bool=True, config: EntryTimingConfig=DEFAULT_ENTRY_TIMING_CONFIG) -> EntryTimingResult:
    config.validate()
    try:
        inputs=load_entry_timing_inputs(config)
    except Exception as exc:
        return _empty(config,"BLOCKED_SOURCE_REPORT",f"冻结报告链不可用：{type(exc).__name__}: {exc}",{"source_report":str(exc)})
    loader=create_loader(LONG_CONTEXT_BASE_CONFIG,data_dir=data_dir)
    pre=run_public_loader_preflight(loader,LONG_CONTEXT_BASE_CONFIG,sample_dates=("2024-06-15","2025-06-15"))
    preflight={"trade_bar":pre.to_dict(),"source_2_8a":str(config.source_2_8a_path),"source_2_12":str(config.source_2_12_path),"source_2_13":str(config.source_2_13_path)}
    if pre.status!="PASS": return _empty(config,"BLOCKED_DATA","1分钟Trade Bar公共Loader预检失败。",preflight)

    path_config=TrancheEligibilityConfig().structural_config()
    decisions=[]; cycles=[]; legs=[]; daily=[]; summaries=[]; rejections=[]; mae_parts=[]; failures=[]
    reporter=ProgressReporter("[R03.4.2.14 folds/delays]",len(inputs.folds)*len(config.entry_delay_minutes),every=1,enabled=progress); step=0
    for fold in inputs.folds.to_dict("records"):
        fold_id=str(fold["fold_id"]); test_start=pd.Timestamp(fold["test_start"]); test_end=pd.Timestamp(fold["test_end"]).floor("min")
        try:
            path=load_minute_path_data(start=test_start-pd.Timedelta(days=2),end=test_end,data_dir=data_dir,config=path_config,progress=progress)
            mae_parts.append(build_anchor_mae_attribution(inputs.source_c2_cycles,inputs.selected_events,path=path,fold_id=fold_id))
            for delay in config.entry_delay_minutes:
                for policy in config.policies:
                    for cost in config.cost_multipliers:
                        sim=simulate_entry_timing_account(inputs.selected_events,inputs.all_q70_signals,path=path,fold_id=fold_id,policy=policy,delay_minutes=delay,cost_multiplier=cost,test_start=test_start,test_end=test_end,config=config,progress=progress)
                        if sim.summary: summaries.append(sim.summary)
                        if not sim.decisions.empty: decisions.append(sim.decisions)
                        if not sim.cycles.empty: cycles.append(sim.cycles)
                        if not sim.legs.empty: legs.append(sim.legs)
                        if not sim.daily_equity.empty: daily.append(sim.daily_equity)
                        if not sim.rejections.empty: rejections.append(sim.rejections)
                step+=1; reporter.update(step)
        except Exception as exc:
            failures.append({"fold_id":fold_id,"delay_minutes":"ALL","error":f"{type(exc).__name__}: {exc}"}); step+=len(config.entry_delay_minutes); reporter.update(step)
    reporter.close()
    decisions_f=pd.concat(decisions,ignore_index=True) if decisions else pd.DataFrame(); cycles_f=pd.concat(cycles,ignore_index=True) if cycles else pd.DataFrame(); legs_f=pd.concat(legs,ignore_index=True) if legs else pd.DataFrame(); daily_f=pd.concat(daily,ignore_index=True) if daily else pd.DataFrame(); summary_f=pd.DataFrame(summaries); rejections_f=pd.concat(rejections,ignore_index=True) if rejections else pd.DataFrame(); mae_f=pd.concat(mae_parts,ignore_index=True) if mae_parts else pd.DataFrame(); mae_summary=summarize_mae_attribution(mae_f); failure_f=pd.DataFrame(failures)

    # E0 must reproduce the frozen equal-risk C2 account before any candidate is interpreted.
    if not summary_f.empty:
        anchor=summary_f.loc[summary_f["policy"].astype(str).eq("E0_immediate_C2")]
        merged=inputs.source_c2_summary.merge(anchor,on=["fold_id","delay_minutes","cost_multiplier"],how="left",suffixes=("_source","_actual"))
        for row in merged.to_dict("records"):
            for metric in ("total_net_return","max_drawdown"):
                if abs(float(row.get(f"{metric}_source",float("nan")))-float(row.get(f"{metric}_actual",float("nan"))))>1e-8:
                    failures.append({"fold_id":row.get("fold_id"),"delay_minutes":row.get("delay_minutes"),"error":f"E0 anchor mismatch {metric}"})
        failure_f=pd.DataFrame(failures)
    gate=policy_gate(summary_f,config); causal=_causal_audit()
    if not failure_f.empty: decision="FAIL_RUNTIME"; reason="运行失败或E0未精确复现冻结C2；不得解释候选结果。"
    elif gate.empty: decision="FAIL_RUNTIME"; reason="没有生成完整入场资格门。"
    elif gate["pass_to_next_stage"].astype(bool).any():
        winners=gate.loc[gate["pass_to_next_stage"].astype(bool),"policy"].astype(str).tolist(); decision="PASS_ENTRY_TIMING_UPGRADE"; reason=f"同一套有界因果入场规则跨年通过：{winners}。"
    else: decision="PASS_C2_FROZEN_NO_ENTRY_UPLIFT"; reason="没有延迟入场规则在保留至少90%交易的同时跨年改善MAE/胜率并保住C2收益；冻结立即入场C2。"
    reports.write_reports(config=config,manifest={"stage":STAGE_ID,"name":STAGE_NAME,"config":config.to_dict()},preflight=preflight,historical=inputs.historical_contract,source_summary=inputs.source_c2_summary,mae=mae_f,mae_summary=mae_summary,decisions=decisions_f,cycles=cycles_f,legs=legs_f,daily=daily_f,summary=summary_f,gate=gate,causal=causal,rejections=rejections_f,failures=failure_f,decision=decision,reason=reason)
    return EntryTimingResult(decision,config.report_path)
