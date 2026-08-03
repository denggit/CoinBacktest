#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.8B."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import TrancheAccountConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    *,
    fixed_6h: pd.DataFrame,
    account_summary: pd.DataFrame,
    gate: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.8B 双风险槽位账户级回测",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 冻结边界",
        "",
        "- q70开仓模型不变。",
        "- 每个Tranche独立使用Failed-Reclaim结构退出与3%灾难保护。",
        "- 固定6小时只作为全部q70信号的诊断基准，不作为最终退出。",
        "- 最多两个虚拟Tranche；交易所层仍是一个ETH净仓位。",
        "- P0=1R单槽；P1=0.5R+0.5R；P2=0.65R+0.35R；P3=0.65R+0.35R并阻止危险摊低/结构破坏时的第二槽。",
        "- 1R默认等于账户权益的1%，仓位按账户风险除以3%灾难距离计算。",
        "- 新信号不会重置旧Tranche的Failed-Reclaim状态，也不会放宽旧仓保护。",
        "- 2026继续封存。",
        "",
    ]
    if not fixed_6h.empty:
        lines.extend(["## 固定6小时诊断基准（1分钟延迟）", ""])
        focus = fixed_6h.loc[fixed_6h["delay_minutes"] == 1]
        for row in focus.sort_values(["fold_id", "cost_multiplier"]).itertuples():
            lines.append(
                f"- {row.fold_id} / {row.cost_multiplier:.0f}x成本：{int(row.signals)}个独立诊断信号，"
                f"净期望{row.mean_net_return:.3%}，PF={row.profit_factor:.2f}，"
                f"独立复合{row.independent_compounded_return:.1%}。"
            )
        lines.append("")
    if not account_summary.empty:
        lines.extend(["## 账户级核心结果（1分钟延迟、2倍成本）", ""])
        focus = account_summary.loc[
            (account_summary["delay_minutes"] == 1)
            & (account_summary["cost_multiplier"] == 2.0)
        ]
        for row in focus.sort_values(["policy", "fold_id"]).itertuples():
            lines.append(
                f"- {row.policy} / {row.fold_id}: {int(row.executed_events)}个Tranche，"
                f"覆盖{row.coverage_ratio:.1%}，月均{row.monthly_tranches:.1f}，"
                f"总收益{row.total_net_return:.1%}，MDD={row.max_drawdown:.1%}，"
                f"PF={row.profit_factor:.2f}，第二槽危险/亏损占比="
                f"{row.dangerous_second_add_share:.1%}/{row.losing_active_second_add_share:.1%}。"
            )
        lines.append("")
    if not gate.empty:
        lines.extend(["## 统一策略资格门", ""])
        for row in gate.sort_values("policy").itertuples():
            lines.append(
                f"- {row.policy}: 主门={bool(row.primary_gate_pass)}，"
                f"3/5分钟延迟压力={bool(row.delay_stress_pass)}，"
                f"进入下一阶段={bool(row.pass_to_entry_stop_research)}。"
            )
        lines.append("")
    lines.extend(
        [
            "## 决策解释",
            "",
            "- 通过不等于最终策略完成，只证明双槽位能在不突破1R预设槽位上限的情况下，改善单仓Failed-Reclaim的机会覆盖和账户收益。",
            "- 若全部失败，不通过放宽到无限加仓、第三Tranche或按年份挑策略来补救。",
            "- 若有统一方案通过，下一步进入入场、真实结构止损与MAE优化；之后才正式冻结分数分层仓位。",
            "- 最终仍需非时间退出复核、完整真实成本账户回测和2026封存验证。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    *,
    config: TrancheAccountConfig,
    manifest: dict[str, object],
    preflight: dict[str, object],
    fixed_6h: pd.DataFrame,
    pair_diagnostics: pd.DataFrame,
    decisions: pd.DataFrame,
    account_summary: pd.DataFrame,
    trades: pd.DataFrame,
    daily_equity: pd.DataFrame,
    concentration: pd.DataFrame,
    gate: pd.DataFrame,
    runtime_rejections: pd.DataFrame,
    causal_audit: pd.DataFrame,
    failures: pd.DataFrame,
    decision: str,
    reason: str,
) -> None:
    report_dir = config.report_path
    report_dir.mkdir(parents=True, exist_ok=True)
    _json(report_dir / "00_run_manifest.json", manifest)
    _json(report_dir / "01_preflight.json", preflight)
    _csv(report_dir / "02_fixed_6h_diagnostic_summary.csv", fixed_6h)
    _csv(report_dir / "03_causal_pair_diagnostics.csv", pair_diagnostics)
    _csv(report_dir / "04_policy_event_decisions.csv", decisions)
    _csv(report_dir / "05_account_policy_summary.csv", account_summary)
    _csv(report_dir / "06_account_trades.csv", trades)
    _csv(report_dir / "07_daily_equity.csv", daily_equity)
    _csv(report_dir / "08_concentration_summary.csv", concentration)
    _csv(report_dir / "09_policy_gate.csv", gate)
    _csv(report_dir / "10_runtime_rejections.csv", runtime_rejections)
    _csv(report_dir / "11_causal_audit.csv", causal_audit)
    _csv(report_dir / "12_failures.csv", failures)
    (report_dir / "99_decision.md").write_text(
        decision_markdown(
            decision,
            reason,
            fixed_6h=fixed_6h,
            account_summary=account_summary,
            gate=gate,
        ),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.2.8B",
            edge_id="q70_dual_risk_slot_account",
            stage="research",
            title="ETH AI R03.4.2.8B dual risk-slot account audit",
            decision_focus="whether a unified maximum-two-tranche account policy restores q70 coverage and beats the single-position failed-reclaim baseline in both 2024 and 2025 without exceeding the risk cap or becoming averaging down",
            print_log=True,
        )
    )
