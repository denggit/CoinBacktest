#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.10."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import RiskMigrationConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    *,
    source_baseline: pd.DataFrame,
    account_summary: pd.DataFrame,
    gate: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.10 结构驱动部分减仓与q70风险迁移研究",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 冻结边界",
        "",
        "- q70 ML开仓池、分数阈值和跨年口径不变。",
        "- 3%硬止损只作为灾难保护；不再把15m Pivot低点直接挂成硬止损。",
        "- `failed_reclaim`继续作为每个Tranche独立的确定性非时间退出。",
        "- 固定6小时仅为诊断基准，不参与最终退出。",
        "- 新周期首仓始终从1R开始，不静态预留第二槽位。",
        "- 部分减仓必须是真实成交；风险迁移必须先减少旧仓或使用已真实释放的风险容量。",
        "- 同一风险周期同时最多两个虚拟Tranche，总初始亏损风险预算不超过1R。",
        "- 亏损旧仓、BROKEN状态或Failed-Reclaim确认过程中禁止迁移新风险。",
        "- 2026继续封存；禁止按年份挑不同政策。",
        "",
        "## 预注册政策",
        "",
        "- P0：1R单仓基准。",
        "- R1/R2：已证明趋势结构第一次进入软BROKEN且旧仓不亏时，分别减仓25%/50%。",
        "- M1/M2：新q70出现时，在健康且不亏的旧仓与新机会之间迁移最多0.35R/0.50R。",
        "- H1：先允许25%结构减仓，再将真实释放的容量迁移给后续q70，最高0.35R。",
        "",
    ]
    if not source_baseline.empty:
        lines.extend(["## 来源P0基准（1分钟延迟、2倍成本）", ""])
        focus = source_baseline.loc[
            (source_baseline["delay_minutes"].astype(int) == 1)
            & (source_baseline["cost_multiplier"].astype(float) == 2.0)
        ]
        for row in focus.sort_values("fold_id").itertuples():
            lines.append(
                f"- {row.fold_id}: {int(row.executed_tranches)}笔，总收益{row.total_net_return:.1%}，"
                f"MDD={row.max_drawdown:.1%}，覆盖{row.coverage_ratio:.1%}。"
            )
        lines.append("")
    if not account_summary.empty:
        lines.extend(["## 账户级核心结果（1分钟延迟、2倍成本）", ""])
        focus = account_summary.loc[
            (account_summary["delay_minutes"].astype(int) == 1)
            & (account_summary["cost_multiplier"].astype(float) == 2.0)
        ]
        for row in focus.sort_values(["policy", "fold_id"]).itertuples():
            lines.append(
                f"- {row.policy} / {row.fold_id}: {int(row.executed_tranches)}个Tranche，"
                f"覆盖{row.coverage_ratio:.1%}，月均{row.monthly_tranches:.1f}，"
                f"收益{row.total_net_return:.1%}，MDD={row.max_drawdown:.1%}，PF={row.profit_factor:.2f}，"
                f"结构减仓{int(row.partial_reduce_actions)}次，迁移释放{int(row.migration_release_actions)}次。"
            )
        lines.append("")
    if not gate.empty:
        lines.extend(["## 统一资格门", ""])
        for row in gate.itertuples():
            lines.append(
                f"- {row.policy}: 最低年度收益保留{row.minimum_return_retention:.1%}，"
                f"跨年收益比{row.combined_return_ratio:.2f}，压力门={bool(row.stress_gate_pass)}，"
                f"进入下一阶段={bool(row.pass_to_next_stage)}。"
            )
        lines.append("")
    lines.extend(
        [
            "## 决策解释",
            "",
            "- 部分减仓的价值必须来自真实降低风险和改善周转，不能只因回撤更低就接受大幅收益损失。",
            "- 风险迁移不是加总风险：旧仓减少多少初始亏损风险，新Tranche最多获得多少。",
            "- 若迁移政策无法接近或超过P0收益，不增加第三Tranche、不放宽亏损仓迁移，也不按年份切规则。",
            "- 通过也不代表完整策略结束；后续仍需入场MAE、分数定仓、最终退出链复核和2026封存验证。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    *,
    config: RiskMigrationConfig,
    manifest: dict[str, object],
    preflight: dict[str, object],
    source_baseline: pd.DataFrame,
    structure_timeline: pd.DataFrame,
    pair_snapshots: pd.DataFrame,
    decisions: pd.DataFrame,
    actions: pd.DataFrame,
    legs: pd.DataFrame,
    trades: pd.DataFrame,
    daily_equity: pd.DataFrame,
    account_summary: pd.DataFrame,
    gate: pd.DataFrame,
    causal_audit: pd.DataFrame,
    runtime_rejections: pd.DataFrame,
    failures: pd.DataFrame,
    decision: str,
    reason: str,
) -> None:
    report_dir = config.report_path
    report_dir.mkdir(parents=True, exist_ok=True)
    _json(report_dir / "00_run_manifest.json", manifest)
    _json(report_dir / "01_preflight.json", preflight)
    _csv(report_dir / "02_source_p0_baseline.csv", source_baseline)
    _csv(report_dir / "03_soft_structure_timeline.csv", structure_timeline)
    _csv(report_dir / "04_candidate_pair_snapshots.csv", pair_snapshots)
    _csv(report_dir / "05_event_decisions.csv", decisions)
    _csv(report_dir / "06_account_actions.csv", actions)
    _csv(report_dir / "07_account_legs.csv", legs)
    _csv(report_dir / "08_account_trades.csv", trades)
    _csv(report_dir / "09_daily_equity.csv", daily_equity)
    _csv(report_dir / "10_account_policy_summary.csv", account_summary)
    _csv(report_dir / "11_policy_gate.csv", gate)
    _csv(report_dir / "12_causal_audit.csv", causal_audit)
    _csv(report_dir / "13_runtime_rejections.csv", runtime_rejections)
    _csv(report_dir / "14_failures.csv", failures)
    (report_dir / "99_decision.md").write_text(
        decision_markdown(
            decision,
            reason,
            source_baseline=source_baseline,
            account_summary=account_summary,
            gate=gate,
        ),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.2.10",
            edge_id="q70_partial_derisk_risk_migration",
            stage="research",
            title="ETH AI R03.4.2.10 partial de-risking and q70 risk migration",
            decision_focus="whether real partial closes and one-R-conserving risk migration can preserve failed-reclaim returns while restoring q70 coverage without a hard Pivot stop, time exit, averaging down or more than two tranches",
            print_log=True,
        )
    )
