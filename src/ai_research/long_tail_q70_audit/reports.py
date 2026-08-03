#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import Q70CrossYearAuditConfig


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
    band_summary: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.4 q70跨年开仓池审核",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 研究边界",
        "",
        "- q70与q90使用完全相同的冻结R03.4.1开仓模型。",
        "- 固定6小时仅是开仓Edge的冻结诊断基准，不是最终实盘退出方案。",
        "- 本阶段不依赖恢复模型、续持模型或已舍弃的市场状态模型，因此WF_2024不能再被持仓模型样本不足阻塞。",
        "- 2026继续封存；信号在闭合决策点生成，下一分钟开盘执行。",
        "",
    ]
    if not comparison.empty:
        lines.extend(["## q70相对q90（1分钟延迟、2倍成本）", ""])
        for row in comparison.itertuples():
            lines.append(
                f"- {row.fold_id}: q70/q90交易={int(row.q70_trades)}/{int(row.q90_trades)}, "
                f"单笔净期望={row.q70_mean_net:.3%}/{row.q90_mean_net:.3%}, "
                f"PF={row.q70_pf:.2f}/{row.q90_pf:.2f}, "
                f"复合收益={row.q70_total_return:.1%}/{row.q90_total_return:.1%}, "
                f"MDD={row.q70_mdd:.1%}/{row.q90_mdd:.1%}。"
            )
        lines.append("")
    if not band_summary.empty:
        lines.extend(["## q70新增分数带", ""])
        focus = band_summary.loc[
            (band_summary["delay_minutes"] == 1)
            & (band_summary["cost_multiplier"] == 2.0)
            & (band_summary["score_band"] == "q70_to_q90")
        ]
        for row in focus.sort_values("fold_id").itertuples():
            lines.append(
                f"- {row.fold_id}: 新增带交易={int(row.trades)}, 净期望={row.mean_net_return:.3%}, "
                f"PF={row.profit_factor:.2f}, 胜率={row.win_rate:.1%}, 去Top10={row.mean_net_without_top10:.3%}。"
            )
        lines.append("")
    if not stable.empty:
        row = stable.iloc[0]
        lines.extend(
            [
                "## 硬门槛",
                "",
                f"- q70跨年2倍成本正期望：{bool(row['positive_expectancy_2x_both_years'])}",
                f"- q70跨年PF≥1.40：{bool(row['pf_2x_both_years'])}",
                f"- q70新增分数带跨年正期望：{bool(row['incremental_band_positive_both_years'])}",
                f"- q70两年总复合收益均超过q90：{bool(row['beats_q90_total_return_both_years'])}",
                f"- 5分钟延迟与3倍成本仍为正：{bool(row['delay_and_cost_stress_pass'])}",
                f"- 去Top10、季度、MDD与集中度通过：{bool(row['robustness_pass'])}",
                "",
            ]
        )
    lines.extend(
        [
            "## 下一阶段约束",
            "",
            "- 只有q70通过跨年审核后，才进入坏单退出Overlay和选择性长持研究。",
            "- 后续退出研究不得再把整条策略强制依赖稀缺的恢复标签；持续失败Overlay必须能够独立评估。",
            "- 最终目标是无机械持仓时限的结构化管理，但任何新退出都必须超过本报告的q70固定6小时收益基准。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    *,
    config: Q70CrossYearAuditConfig,
    manifest: dict[str, object],
    preflight: dict[str, object],
    score_audit: pd.DataFrame,
    execution_audit: pd.DataFrame,
    summary: pd.DataFrame,
    periods: pd.DataFrame,
    bands: pd.DataFrame,
    score_deciles: pd.DataFrame,
    comparison: pd.DataFrame,
    overlap: pd.DataFrame,
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
    _csv(report_dir / "03_execution_audit.csv", execution_audit)
    _csv(report_dir / "04_policy_summary.csv", summary)
    _csv(report_dir / "05_period_summary.csv", periods)
    _csv(report_dir / "06_q70_score_band_summary.csv", bands)
    _csv(report_dir / "07_q70_score_deciles.csv", score_deciles)
    _csv(report_dir / "08_q70_vs_q90_comparison.csv", comparison)
    _csv(report_dir / "09_overlap_audit.csv", overlap)
    _csv(report_dir / "10_stable_candidate.csv", stable)
    _csv(report_dir / "11_causal_audit.csv", causal_audit)
    _csv(report_dir / "12_failures.csv", failures)
    _csv(report_dir / "13_trade_details.csv", trades)
    (report_dir / "99_decision.md").write_text(
        decision_markdown(decision, reason, comparison=comparison, stable=stable, band_summary=bands),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.2.4",
            edge_id="frozen_long_q70_cross_year_audit",
            stage="research",
            title="ETH AI R03.4.2.4 q70 cross-year opening-pool audit",
            decision_focus="whether q70 robustly increases total cost-adjusted profit versus q90 in both 2024 and 2025 before non-time exit research",
            print_log=True,
        )
    )
