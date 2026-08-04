#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.11."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import StagedExecutionConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    *,
    source_baseline: pd.DataFrame,
    summary: pd.DataFrame,
    gate: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.11 分批入场、软失败止损与非对称金字塔研究",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 冻结边界",
        "",
        "- q70 ML开仓池、分数阈值和2024/2025跨年口径不变。",
        "- 基础仓继续使用3%灾难硬止损与确定性`failed_reclaim`非时间退出。",
        "- 不再把15m Pivot直接作为基础仓硬止损。",
        "- 固定6小时只保留为诊断基准，不参与最终退出。",
        "- 加仓层不会减仓、重置或放宽基础仓；基础赢家必须独立保留。",
        "- 加仓层使用独立止损；被扫只关闭加仓层，不关闭基础仓。",
        "- 每层数量按该层账户风险除以该层真实止损距离计算。",
        "- 账户硬尾部风险最多2R，最大名义仓位最多1.5倍权益，最多基础仓加两个加仓层。",
        "- 所有触发使用已完成分钟/结构信息，并在下一根1m open执行。",
        "- 2026继续封存；禁止按年份挑不同政策。",
        "",
        "## 预注册政策",
        "",
        "- P0：基础仓1R，3%灾难止损，`failed_reclaim`退出。",
        "- F1：按1.5%软失败距离定仓，15m完成收盘确认失败后主动退出，3%仍是2R尾部灾难保护。",
        "- S1：0.60R基础仓；回踩收回或顺势确认后补0.40R，总硬风险不超过1R。",
        "- T1：完整1R基础仓；价格顺势走出一个因果N后增加0.35R，独立止损，总硬风险不超过1.35R。",
        "- P1：完整1R基础仓；在一个N和两个N位置各增加0.35R，最多三层，总硬风险不超过1.70R。",
        "",
    ]
    if not source_baseline.empty:
        lines.extend(["## 来源P0基准（1分钟延迟、2倍成本）", ""])
        focus = source_baseline.loc[
            source_baseline["delay_minutes"].astype(int).eq(1)
            & source_baseline["cost_multiplier"].astype(float).eq(2.0)
        ]
        for row in focus.sort_values("fold_id").itertuples():
            lines.append(
                f"- {row.fold_id}: 收益{row.total_net_return:.1%}，MDD={row.max_drawdown:.1%}，"
                f"最大名义仓位{row.max_notional_to_equity:.2f}倍。"
            )
        lines.append("")
    if not summary.empty:
        lines.extend(["## 核心账户结果（1分钟延迟、2倍成本）", ""])
        focus = summary.loc[
            summary["delay_minutes"].astype(int).eq(1)
            & summary["cost_multiplier"].astype(float).eq(2.0)
        ]
        for row in focus.sort_values(["policy", "fold_id"]).itertuples():
            lines.append(
                f"- {row.policy} / {row.fold_id}: 收益{row.total_net_return:.1%}，MDD={row.max_drawdown:.1%}，"
                f"PF={row.profit_factor:.2f}，加仓层{int(row.add_tranches)}个，"
                f"最大硬尾部风险{row.max_hard_tail_r:.2f}R，最大名义仓位{row.max_notional_to_equity:.2f}倍，"
                f"基准赢家转亏比例{row.winner_to_loser_share:.1%}。"
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
            "- 提高名义仓位不是目标本身；必须证明新增仓位带来的收益超过手续费、止损和尾部风险成本。",
            "- 软失败定仓只有在两年都减少平均亏损、同时不放大灾难尾部时才可保留。",
            "- 海龟/金字塔层允许被独立扫损，但不得系统性把原P0盈利周期变成亏损周期。",
            "- 如果加仓层不能超过完整基础仓P0，不继续微调N倍数、止损百分比或按年份选规则。",
            "- 通过也不代表完整策略结束；仍需分数风险层、最终退出链复核、2026封存验证和Portfolio组合。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    *,
    config: StagedExecutionConfig,
    manifest: dict[str, object],
    preflight: dict[str, object],
    source_baseline: pd.DataFrame,
    selected_events: pd.DataFrame,
    cycles: pd.DataFrame,
    legs: pd.DataFrame,
    actions: pd.DataFrame,
    daily_equity: pd.DataFrame,
    summary: pd.DataFrame,
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
    _csv(report_dir / "03_selected_p0_cycles.csv", selected_events)
    _csv(report_dir / "04_account_cycles.csv", cycles)
    _csv(report_dir / "05_account_legs.csv", legs)
    _csv(report_dir / "06_account_actions.csv", actions)
    _csv(report_dir / "07_daily_equity.csv", daily_equity)
    _csv(report_dir / "08_policy_summary.csv", summary)
    _csv(report_dir / "09_policy_gate.csv", gate)
    _csv(report_dir / "10_causal_audit.csv", causal_audit)
    _csv(report_dir / "11_runtime_rejections.csv", runtime_rejections)
    _csv(report_dir / "12_failures.csv", failures)
    (report_dir / "99_decision.md").write_text(
        decision_markdown(
            decision,
            reason,
            source_baseline=source_baseline,
            summary=summary,
            gate=gate,
        ),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.2.11",
            edge_id="q70_staged_entry_asymmetric_pyramid",
            stage="research",
            title="ETH AI R03.4.2.11 staged entry and asymmetric pyramiding",
            decision_focus=(
                "whether soft-failure sizing, split entry, Turtle-style adds or independent-stop "
                "pyramiding can raise executable notional exposure and total return without reducing "
                "the frozen q70 base winner, exceeding two-R tail risk, turning baseline winners into "
                "losers, using a fixed-time exit or opening 2026"
            ),
            print_log=True,
        )
    )
