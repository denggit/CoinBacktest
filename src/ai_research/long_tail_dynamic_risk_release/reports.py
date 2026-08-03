#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.9."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import DynamicRiskReleaseConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    *,
    source_baseline: pd.DataFrame,
    protection_summary: pd.DataFrame,
    account_summary: pd.DataFrame,
    gate: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.9 结构保护止损与动态风险释放研究",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 冻结边界",
        "",
        "- q70 ML开仓模型、分数阈值和跨年口径不变。",
        "- Failed-Reclaim仍是确定性非时间结构退出，不重新调参。",
        "- 3%硬止损保留为灾难保护；结构保护位只能上移，不能放宽。",
        "- 固定6小时仅为诊断基准，不作为最终出场。",
        "- 首仓始终以1R开始，不再静态预留第二槽位。",
        "- 第二Tranche只能使用真实已挂保护止损释放的风险，最多两个虚拟Tranche。",
        "- 2026继续封存，禁止按年份挑不同规则。",
        "",
    ]
    if not source_baseline.empty:
        lines.extend(["## 来源基准（R03.4.2.8B P0，1分钟延迟、2倍成本）", ""])
        focus = source_baseline.loc[
            (source_baseline["delay_minutes"] == 1)
            & (source_baseline["cost_multiplier"] == 2.0)
        ]
        for row in focus.sort_values("fold_id").itertuples():
            lines.append(
                f"- {row.fold_id}: {int(row.executed_events)}笔，覆盖{row.coverage_ratio:.1%}，"
                f"总收益{row.total_net_return:.1%}，MDD={row.max_drawdown:.1%}。"
            )
        lines.append("")
    if not protection_summary.empty:
        lines.extend(["## 结构保护行为（1分钟延迟）", ""])
        focus = protection_summary.loc[protection_summary["delay_minutes"] == 1]
        for row in focus.sort_values(["protection_policy", "fold_id"]).itertuples():
            lines.append(
                f"- {row.protection_policy} / {row.fold_id}: 硬结构止损占比{row.hard_stop_share:.1%}，"
                f"平均最大释放风险{row.mean_maximum_released_risk_fraction:.1%}，"
                f"持仓中位数{row.median_holding_minutes:.0f}分钟。"
            )
        lines.append("")
    if not account_summary.empty:
        lines.extend(["## 账户级核心结果（1分钟延迟、2倍成本）", ""])
        focus = account_summary.loc[
            (account_summary["delay_minutes"] == 1)
            & (account_summary["cost_multiplier"] == 2.0)
        ]
        for row in focus.sort_values(["protection_policy", "dynamic_policy", "fold_id"]).itertuples():
            lines.append(
                f"- {row.protection_policy} + {row.dynamic_policy} / {row.fold_id}: "
                f"{int(row.executed_tranches)}笔，覆盖{row.coverage_ratio:.1%}，月均{row.monthly_tranches:.1f}，"
                f"收益{row.total_net_return:.1%}，MDD={row.max_drawdown:.1%}，"
                f"最大实时剩余风险={row.max_live_remaining_r:.2f}R，第二Tranche={int(row.secondary_tranches)}笔。"
            )
        lines.append("")
    if not gate.empty:
        lines.extend(["## 预注册资格门", ""])
        for row in gate.itertuples():
            lines.append(
                f"- {row.protection_policy} + {row.dynamic_policy}: "
                f"保护门={bool(row.protection_gate_pass)}，压力门={bool(row.stress_gate_pass)}，"
                f"跨年总收益不低于P0={bool(row.cross_year_total_improvement)}，"
                f"进入下一阶段={bool(row.pass_to_next_stage)}。"
            )
        lines.append("")
    lines.extend(
        [
            "## 决策解释",
            "",
            "- 最新确认低点保护若过早砍掉大赢家，应正式舍弃，不能因MDD更低而保留。",
            "- 落后一层结构保护只有在两年均保留足够P0收益后，才有资格用于动态释放风险。",
            "- 动态Tranche若不能恢复覆盖且至少接近P0收益，不通过降低收益门槛或增加第三Tranche补救。",
            "- 本阶段通过也不代表完整策略完成；后续仍需入场MAE、分数定仓、最终非时间退出复核和2026封存验证。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    *,
    config: DynamicRiskReleaseConfig,
    manifest: dict[str, object],
    preflight: dict[str, object],
    source_baseline: pd.DataFrame,
    protection_summary: pd.DataFrame,
    protection_trades: pd.DataFrame,
    stop_updates: pd.DataFrame,
    pair_diagnostics: pd.DataFrame,
    decisions: pd.DataFrame,
    account_summary: pd.DataFrame,
    account_trades: pd.DataFrame,
    daily_equity: pd.DataFrame,
    gate: pd.DataFrame,
    runtime_rejections: pd.DataFrame,
    causal_audit: pd.DataFrame,
    failures: pd.DataFrame,
    decision: str,
    reason: str,
) -> None:
    root = config.report_path
    root.mkdir(parents=True, exist_ok=True)
    _json(root / "00_run_manifest.json", manifest)
    _json(root / "01_preflight.json", preflight)
    _csv(root / "02_source_p0_baseline.csv", source_baseline)
    _csv(root / "03_protection_summary.csv", protection_summary)
    _csv(root / "04_protection_trades.csv", protection_trades)
    _csv(root / "05_stop_updates.csv", stop_updates)
    _csv(root / "06_release_pair_diagnostics.csv", pair_diagnostics)
    _csv(root / "07_dynamic_event_decisions.csv", decisions)
    _csv(root / "08_account_policy_summary.csv", account_summary)
    _csv(root / "09_account_trades.csv", account_trades)
    _csv(root / "10_daily_equity.csv", daily_equity)
    _csv(root / "11_policy_gate.csv", gate)
    _csv(root / "12_runtime_rejections.csv", runtime_rejections)
    _csv(root / "13_causal_audit.csv", causal_audit)
    _csv(root / "14_failures.csv", failures)
    (root / "99_decision.md").write_text(
        decision_markdown(
            decision,
            reason,
            source_baseline=source_baseline,
            protection_summary=protection_summary,
            account_summary=account_summary,
            gate=gate,
        ),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=root,
            experiment_id="R03.4.2.9",
            edge_id="q70_dynamic_structural_risk_release",
            stage="research",
            title="ETH AI R03.4.2.9 structural protection and dynamic risk release",
            decision_focus="whether a monotone enforceable structural stop preserves failed-reclaim returns and can fund a second q70 tranche without static primary-risk dilution or more than one live R",
            print_log=True,
        )
    )
