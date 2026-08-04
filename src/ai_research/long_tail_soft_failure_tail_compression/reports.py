#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.12."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import TailCompressionConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    *,
    source_p0: pd.DataFrame,
    source_f1: pd.DataFrame,
    attribution_summary: pd.DataFrame,
    summary: pd.DataFrame,
    gate: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.12 软失败止损归因与真实尾部风险压缩",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 冻结边界",
        "",
        "- q70 ML开仓、下一根1m open、`failed_reclaim`盈利退出保持不变。",
        "- 固定6小时仍只作为诊断，不参与最终退出。",
        "- 不恢复15m Pivot硬止损，不研究海龟/金字塔或重复q70加仓。",
        "- 2026继续封存；同一政策必须同时用于2024和2025。",
        "- F1参考组保留3%灾难尾部，因此是2R风险参考，不能作为1R候选通过。",
        "- 合格候选必须按真实可执行硬止损定仓，最坏价格尾部不超过1R。",
        "- 软失败只使用已完成结构收盘，并在下一根1m open执行。",
        "",
        "## 预注册政策",
        "",
        "- P0：3%真实硬止损，约0.33倍平均基础名义仓位。",
        "- F1 reference：按1.5%定仓、1.5%完成收盘软失败、3%灾难尾部；只用于拆分风险放大与退出增量。",
        "- C2：2%真实硬止损，1.5%完成收盘软失败，尾部1R。",
        "- C15 hard：1.5%真实硬止损，无软确认，直接测试币圈噪声扫损。",
        "- C15 soft：1.5%真实硬止损，1.0%完成收盘软失败，尾部1R。",
        "- V1：入场时用前60根已完成1m ATR冻结止损，`2×ATR%`限制在1.5%至3%，软阈值为硬止损的75%。",
        "",
    ]
    if not source_p0.empty or not source_f1.empty:
        lines.extend(["## 来源基准（1分钟延迟、2倍成本）", ""])
        joined = pd.concat([source_p0.assign(source="P0"), source_f1.assign(source="F1")], ignore_index=True)
        focus = joined.loc[
            joined["delay_minutes"].astype(int).eq(1)
            & joined["cost_multiplier"].astype(float).eq(2.0)
        ]
        for row in focus.sort_values(["source", "fold_id"]).itertuples():
            lines.append(
                f"- {row.source}/{row.fold_id}: 收益{row.total_net_return:.1%}，MDD={row.max_drawdown:.1%}，"
                f"平均基础名义仓位{row.mean_base_notional_to_equity:.2f}倍，最大硬尾部{row.max_hard_tail_r:.2f}R。"
            )
        lines.append("")
    if not attribution_summary.empty:
        lines.extend(["## F1软失败归因（1分钟延迟、2倍成本）", ""])
        focus = attribution_summary.loc[
            attribution_summary["delay_minutes"].astype(int).eq(1)
            & attribution_summary["cost_multiplier"].astype(float).eq(2.0)
        ]
        for row in focus.sort_values(["fold_id", "attribution_class"]).itertuples():
            lines.append(
                f"- {row.fold_id}/{row.attribution_class}: {int(row.events)}笔，"
                f"1R归一化退出增量{row.mean_exit_edge_1r:+.3%}，"
                f"软退出后重新回到入场价上方比例{row.recovered_above_entry_share:.1%}。"
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
                f"- {row.policy}/{row.fold_id}: 收益{row.total_net_return:.1%}，MDD={row.max_drawdown:.1%}，"
                f"PF={row.profit_factor:.2f}，平均名义仓位{row.mean_base_notional_to_equity:.2f}倍，"
                f"硬止损{int(row.hard_stop_exits)}笔，软失败{int(row.soft_failure_exits)}笔，"
                f"最坏尾部{row.max_hard_tail_r:.2f}R，基准赢家转亏{row.winner_to_loser_share:.1%}。"
            )
        lines.append("")
    if not gate.empty:
        lines.extend(["## 统一资格门", ""])
        for row in gate.itertuples():
            lines.append(
                f"- {row.policy}: 候选={bool(row.qualifying_candidate)}，最低年度收益保留"
                f"{row.minimum_return_retention:.1%}，跨年收益比{row.combined_return_ratio:.2f}，"
                f"压力门={bool(row.stress_gate_pass)}，通过={bool(row.pass_to_next_stage)}。"
            )
        lines.append("")
    lines.extend(
        [
            "## 决策解释",
            "",
            "- F1的高收益必须先按2R尾部归一化，不能把风险翻倍当成止损Edge。",
            "- 真实硬止损若导致大量P0赢家转亏，说明名义仓位提升只是被更频繁扫损抵消。",
            "- 只有同一套1R真实尾部规则在两年、成本和延迟压力下都保留P0并提高跨年收益，才能进入风险分层。",
            "- 通过也不代表完整策略完成；仍需分数风险层、最终退出链、2026封存和Portfolio组合验证。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    *,
    config: TailCompressionConfig,
    manifest: dict[str, object],
    preflight: dict[str, object],
    source_p0: pd.DataFrame,
    source_f1: pd.DataFrame,
    selected_events: pd.DataFrame,
    attribution: pd.DataFrame,
    attribution_summary: pd.DataFrame,
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
    _csv(report_dir / "02_source_p0_baseline.csv", source_p0)
    _csv(report_dir / "03_source_f1_reference.csv", source_f1)
    _csv(report_dir / "04_selected_p0_cycles.csv", selected_events)
    _csv(report_dir / "05_f1_exit_attribution.csv", attribution)
    _csv(report_dir / "06_f1_attribution_summary.csv", attribution_summary)
    _csv(report_dir / "07_account_cycles.csv", cycles)
    _csv(report_dir / "08_account_legs.csv", legs)
    _csv(report_dir / "09_account_actions.csv", actions)
    _csv(report_dir / "10_daily_equity.csv", daily_equity)
    _csv(report_dir / "11_policy_summary.csv", summary)
    _csv(report_dir / "12_policy_gate.csv", gate)
    _csv(report_dir / "13_causal_audit.csv", causal_audit)
    _csv(report_dir / "14_runtime_rejections.csv", runtime_rejections)
    _csv(report_dir / "15_failures.csv", failures)
    (report_dir / "99_decision.md").write_text(
        decision_markdown(
            decision,
            reason,
            source_p0=source_p0,
            source_f1=source_f1,
            attribution_summary=attribution_summary,
            summary=summary,
            gate=gate,
        ),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.2.12",
            edge_id="q70_soft_failure_real_tail_compression",
            stage="research",
            title="ETH AI R03.4.2.12 soft-failure attribution and real tail compression",
            decision_focus=(
                "whether the F1 completed-close failure logic has true exit value after removing its "
                "two-R sizing effect, and whether a fixed 2%, fixed 1.5%, or causally volatility-adaptive "
                "real hard stop can increase q70 nominal exposure while keeping worst price risk near one-R, "
                "preserving failed_reclaim winners, avoiding fixed-time exits and keeping 2026 sealed"
            ),
            print_log=True,
        )
    )
