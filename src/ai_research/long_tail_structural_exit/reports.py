#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.7."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import StructuralExitConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    *,
    comparison: pd.DataFrame,
    stable: pd.DataFrame,
    exits: pd.DataFrame,
    censoring: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.7 q70因果结构状态机与非时间退出审核",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 核心研究边界",
        "",
        "- WF_2024与WF_2025使用完全相同的预注册结构规则，不允许按年份选择不同模型或规则。",
        "- 候选策略没有固定6小时、24小时、5天或最大持仓时间退出。",
        "- 15分钟只是完成结构确认的观察粒度；结构事件在下一分钟开盘执行。",
        "- OOS结束和数据缺口只记录为右删失并按最后价格保守盯市，不属于策略退出。",
        "- 固定6小时只作为冻结开仓Edge的对照基准。",
        "- q70-q80、q80-q90、q90+全部保留并分层报告，但退出逻辑完全相同。",
        "- 开仓分数在持仓中不用于续期、退出或加仓。",
        "",
    ]
    if not comparison.empty:
        lines.extend(["## 相对固定6小时基准（1分钟延迟、2倍成本）", ""])
        for row in comparison.sort_values(["policy", "fold_id"]).itertuples():
            lines.append(
                f"- {row.policy} / {row.fold_id}: 交易={int(row.trades)}（基准{int(row.fixed_trades)}），"
                f"净期望={row.mean_net_return:.3%}（基准{row.fixed_mean_net_return:.3%}），"
                f"PF={row.profit_factor:.2f}（基准{row.fixed_profit_factor:.2f}），"
                f"复合收益差={row.total_return_delta:.1%}，MDD改善={row.relative_mdd_improvement:.1%}，"
                f"删失={row.censored_share:.1%}，中位持仓={row.median_holding_minutes:.0f}分钟。"
            )
        lines.append("")
    if not stable.empty:
        lines.extend(["## 跨年统一规则门槛", ""])
        for row in stable.itertuples():
            lines.append(
                f"- {row.policy}: 稳健={bool(row.base_robustness_pass)}，"
                f"利润升级={bool(row.passes_profit_upgrade)}，风险升级={bool(row.passes_risk_upgrade)}，"
                f"2024/2025净期望={row.mean_net_2x_2024:.3%}/{row.mean_net_2x_2025:.3%}，"
                f"PF={row.pf_2x_2024:.2f}/{row.pf_2x_2025:.2f}，"
                f"MDD={row.mdd_2024:.1%}/{row.mdd_2025:.1%}。"
            )
        lines.append("")
    if not exits.empty:
        focus = exits.loc[(exits["delay_minutes"] == 1) & exits["policy_kind"].eq("non_time_structural_candidate")]
        lines.extend(["## 非时间退出原因", ""])
        for row in focus.sort_values(["policy", "fold_id", "exit_reason"]).itertuples():
            lines.append(
                f"- {row.policy} / {row.fold_id} / {row.exit_reason}: {int(row.count)}笔，"
                f"占比{row.share:.1%}，平均毛收益{row.mean_gross_return:.3%}，"
                f"中位持仓{row.median_holding_minutes:.0f}分钟。"
            )
        lines.append("")
    if not censoring.empty:
        focus = censoring.loc[(censoring["delay_minutes"] == 1) & censoring["policy"].isin(stable["policy"] if not stable.empty else [])]
        if not focus.empty:
            lines.extend(["## 右删失审核", ""])
            for row in focus.sort_values(["policy", "fold_id"]).itertuples():
                lines.append(
                    f"- {row.policy} / {row.fold_id}: 删失{int(row.censored_trades)}/{int(row.trades)} "
                    f"({row.censored_share:.1%})，删失盯市均值{row.censored_mean_mark_return:.3%}。"
                )
            lines.append("")
    lines.extend(
        [
            "## 后续决策",
            "",
            "- 只有同一套规则同时通过2024与2025，才能进入分数分层风险配置。",
            "- 若结构状态机失败，停止继续堆退出机器学习或参数网格，转向更精确的入场与独立趋势持仓Sleeve。",
            "- 加仓研究必须等退出和真实止损距离冻结后进行，且禁止对失控亏损仓摊低。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    *,
    config: StructuralExitConfig,
    manifest: dict[str, object],
    preflight: dict[str, object],
    score_audit: pd.DataFrame,
    event_audit: pd.DataFrame,
    summary: pd.DataFrame,
    periods: pd.DataFrame,
    tiers: pd.DataFrame,
    exits: pd.DataFrame,
    censoring: pd.DataFrame,
    overlap: pd.DataFrame,
    comparison: pd.DataFrame,
    stable: pd.DataFrame,
    causal_audit: pd.DataFrame,
    failures: pd.DataFrame,
    trades: pd.DataFrame,
    decision: str,
    reason: str,
) -> None:
    report_dir = config.report_path
    report_dir.mkdir(parents=True, exist_ok=True)
    _json(report_dir / "00_run_manifest.json", manifest)
    _json(report_dir / "01_preflight.json", preflight)
    _csv(report_dir / "02_score_threshold_audit.csv", score_audit)
    _csv(report_dir / "03_event_execution_audit.csv", event_audit)
    _csv(report_dir / "04_policy_summary.csv", summary)
    _csv(report_dir / "05_quarter_summary.csv", periods)
    _csv(report_dir / "06_score_tier_summary.csv", tiers)
    _csv(report_dir / "07_exit_reason_summary.csv", exits)
    _csv(report_dir / "08_censoring_audit.csv", censoring)
    _csv(report_dir / "09_overlap_audit.csv", overlap)
    _csv(report_dir / "10_vs_fixed6h_comparison.csv", comparison)
    _csv(report_dir / "11_stable_candidates.csv", stable)
    _csv(report_dir / "12_causal_audit.csv", causal_audit)
    _csv(report_dir / "13_failures.csv", failures)
    _csv(report_dir / "14_trade_details.csv", trades)
    (report_dir / "99_decision.md").write_text(
        decision_markdown(
            decision,
            reason,
            comparison=comparison,
            stable=stable,
            exits=exits,
            censoring=censoring,
        ),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.2.7",
            edge_id="q70_causal_non_time_structural_exit",
            stage="research",
            title="ETH AI R03.4.2.7 causal non-time structural exit audit",
            decision_focus="whether one unified causal structural state machine can replace fixed-time holding across both 2024 and 2025 without losing positive expectancy or total profit",
            print_log=True,
        )
    )
