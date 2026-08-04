#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.14."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import EntryTimingConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str)+"\n", encoding="utf-8")


def decision_markdown(decision: str, reason: str, *, historical: pd.DataFrame, mae_summary: pd.DataFrame, summary: pd.DataFrame, gate: pd.DataFrame) -> str:
    lines=["# R03.4.2.14 入场时机与MAE归因","",f"## 决策：`{decision}`","",reason,"","## 冻结边界","","- 冻结q70模型、等风险1R、C2真实2%硬止损、1.5%完成收盘软失败和`failed_reclaim`。","- 本阶段只改变首次入场时机；不加仓、不删信号、不修改退出参数。","- 所有等待均有30至60分钟因果上限，且正式候选至少保留90%的冻结C2交易。","- 固定6小时只保留历史口径对照，绝不作为最终退出。","- 2026继续封存。","","## 历史口径对照",""]
    for row in historical.to_dict("records"):
        lines.append(f"- {row['fold_id']}/{row['metric_scope']}: 交易{int(row['trades'])}，胜率{float(row['win_rate']):.1%}，PF={float(row['profit_factor']):.2f}，收益{float(row['total_return']):.1%}，MDD={float(row['max_drawdown']):.1%}。")
    if not mae_summary.empty:
        lines += ["","## C2初始MAE归因",""]
        for row in mae_summary.to_dict("records"):
            lines.append(f"- {row['fold_id']}/{row['mae_class']}: {int(row['events'])}笔，占比{float(row['share']):.1%}，平均60分钟MAE={float(row['mean_mae_60m']):.2%}，平均收益={float(row['mean_cycle_return']):.2%}。")
    focus=summary.loc[summary["delay_minutes"].astype(int).eq(1) & summary["cost_multiplier"].astype(float).eq(2.0)] if not summary.empty else pd.DataFrame()
    if not focus.empty:
        lines += ["","## 账户结果（1分钟延迟、2倍成本）",""]
        for row in focus.to_dict("records"):
            lines.append(f"- {row['policy']}/{row['fold_id']}: 执行{int(row['executed_cycles'])}笔，覆盖{float(row['coverage_ratio']):.1%}，收益{float(row['total_net_return']):.1%}，MDD={float(row['max_drawdown']):.1%}，胜率{float(row['win_rate']):.1%}，PF={float(row['profit_factor']):.2f}，平均60分钟MAE={float(row['mean_mae_60m']):.2%}，平均等待{float(row['mean_wait_minutes']):.1f}分钟。")
    if not gate.empty:
        lines += ["","## 资格门",""]
        for row in gate.to_dict("records"):
            lines.append(f"- {row['policy']}: 通过={bool(row['pass_to_next_stage'])}，最低覆盖={float(row['minimum_coverage_ratio']):.1%}，最低收益保留={float(row['minimum_return_retention']):.1%}，组合收益比={float(row['combined_return_ratio']):.3f}。")
    lines += ["","## 后续",""]
    if decision=="PASS_ENTRY_TIMING_UPGRADE": lines.append("冻结通过的同一入场规则，随后进行完整C2最终账户鲁棒性与2026封存前审计。")
    else: lines.append("若没有统一规则通过，则冻结立即入场C2，不继续微调等待分钟或追高阈值，进入最终完整策略审计。")
    return "\n".join(lines)+"\n"


def write_reports(*, config: EntryTimingConfig, manifest: dict[str, object], preflight: dict[str, object], historical: pd.DataFrame, source_summary: pd.DataFrame, mae: pd.DataFrame, mae_summary: pd.DataFrame, decisions: pd.DataFrame, cycles: pd.DataFrame, legs: pd.DataFrame, daily: pd.DataFrame, summary: pd.DataFrame, gate: pd.DataFrame, causal: pd.DataFrame, rejections: pd.DataFrame, failures: pd.DataFrame, decision: str, reason: str) -> None:
    root=config.report_path; root.mkdir(parents=True, exist_ok=True)
    _json(root/"00_run_manifest.json",manifest); _json(root/"01_preflight.json",preflight)
    for name,frame in (("02_historical_metric_contract.csv",historical),("03_source_c2_summary.csv",source_summary),("04_mae_attribution.csv",mae),("05_mae_attribution_summary.csv",mae_summary),("06_entry_decisions.csv",decisions),("07_account_cycles.csv",cycles),("08_account_legs.csv",legs),("09_daily_equity.csv",daily),("10_policy_summary.csv",summary),("11_policy_gate.csv",gate),("12_causal_audit.csv",causal),("13_runtime_rejections.csv",rejections),("14_failures.csv",failures)):
        _csv(root/name,frame)
    (root/"99_decision.md").write_text(decision_markdown(decision,reason,historical=historical,mae_summary=mae_summary,summary=summary,gate=gate),encoding="utf-8")
    prompt=("Review R03.4.2.14 as a causal entry-timing audit. Check that q70/C2 exits/risk are frozen, waiting is bounded, coverage cannot be optimized away, and 2026 remains sealed. Focus on whether delayed entry improves MAE or win rate without sacrificing annual return.\n")
    (root/"GPT_REVIEW_PROMPT.md").write_text(prompt,encoding="utf-8")
    write_gpt_review_pack(ReviewPackConfig(report_dir=root, edge_id="eth_ai_r03_4_2_14_entry_timing_mae", title="ETH AI R03.4.2.14 entry timing and MAE attribution", decision_focus="whether bounded causal entry timing improves frozen C2 without filtering away trades"))
