#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.8A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import TrancheEligibilityConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    *,
    baseline_summary: pd.DataFrame,
    occupancy: pd.DataFrame,
    classes: pd.DataFrame,
    score_price: pd.DataFrame,
    gate: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.8A 持仓中新q70信号图谱与Tranche资格门",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 本阶段边界",
        "",
        "- 冻结q70开仓模型、Failed-Reclaim结构退出和3%灾难保护参数。",
        "- 本阶段不增加仓位、不执行第二Tranche、不声称账户风险已经释放。",
        "- candidate_hard_stop仅用于判断未来是否存在可研究的保护位，尚不是实际止损。",
        "- 新信号分为健康趋势、回撤修复、危险摊低和不明确四类。",
        "- 分数升高本身永远不能触发加仓。",
        "- 固定6小时仍仅用于衡量新信号后是否存在独立开仓Edge。",
        "- 2026继续封存。",
        "",
    ]
    if not baseline_summary.empty:
        lines.extend(["## 两个冻结基准（1分钟延迟、2倍成本）", ""])
        focus = baseline_summary.loc[
            (baseline_summary["delay_minutes"] == 1)
            & (baseline_summary["cost_multiplier"] == 2.0)
        ]
        for row in focus.sort_values(["baseline", "fold_id"]).itertuples():
            lines.append(
                f"- {row.baseline} / {row.fold_id}: {int(row.signals)}笔，"
                f"净期望{row.mean_net_return:.3%}，PF={row.profit_factor:.2f}，"
                f"胜率{row.win_rate:.1%}，诊断MDD={row.max_drawdown_diagnostic:.1%}。"
            )
        lines.append("")
    if not occupancy.empty:
        lines.extend(["## Failed-Reclaim占仓情况", ""])
        focus = occupancy.loc[occupancy["delay_minutes"] == 1]
        for row in focus.sort_values("fold_id").itertuples():
            lines.append(
                f"- {row.fold_id}: 完整信号{int(row.complete_events)}，单仓执行{int(row.executed_events)}，"
                f"占仓跳过{int(row.occupied_events)}（{row.occupied_share_of_complete:.1%}）。"
            )
        lines.append("")
    if not classes.empty:
        lines.extend(["## 新信号分类结果（1分钟延迟、固定6小时、2倍成本）", ""])
        focus = classes.loc[
            (classes["delay_minutes"] == 1)
            & (classes["outcome"] == "fixed6h_gross_return")
            & (classes["cost_multiplier"] == 2.0)
        ]
        for row in focus.sort_values(["fold_id", "signal_class"]).itertuples():
            lines.append(
                f"- {row.fold_id} / {row.signal_class}: {int(row.signals)}个，"
                f"净期望{row.mean_net_return:.3%}，PF={row.profit_factor:.2f}，"
                f"胜率{row.win_rate:.1%}，亏损仓出现占比{row.losing_position_share:.1%}，"
                f"候选风险释放均值{row.mean_released_risk_fraction:.1%}。"
            )
        lines.append("")
    if not score_price.empty:
        lines.extend(["## 越跌分数越高诊断", ""])
        focus = score_price.loc[score_price["delay_minutes"] == 1]
        for row in focus.sort_values(["fold_id", "signal_class"]).itertuples():
            lines.append(
                f"- {row.fold_id} / {row.signal_class}: score-up/price-down={int(row.score_up_price_down_count)}/"
                f"{int(row.signals)}（{row.score_up_price_down_share:.1%}）。"
            )
        lines.append("")
    if not gate.empty:
        lines.extend(["## 进入P2/P3模拟的资格门", ""])
        for row in gate.sort_values("delay_minutes").itertuples():
            lines.append(
                f"- 延迟{int(row.delay_minutes)}分钟: 通过={bool(row.pass_to_tranche_simulation)}；"
                f"2024/2025合格信号={int(row.eligible_signals_2024)}/{int(row.eligible_signals_2025)}，"
                f"2x净期望={row.mean_net_2x_2024:.3%}/{row.mean_net_2x_2025:.3%}，"
                f"PF={row.pf_2x_2024:.2f}/{row.pf_2x_2025:.2f}，"
                f"3x净期望={row.mean_net_3x_2024:.3%}/{row.mean_net_3x_2025:.3%}，"
                f"独立Failed-Reclaim 2x净期望={row.structural_mean_net_2x_2024:.3%}/"
                f"{row.structural_mean_net_2x_2025:.3%}。"
            )
        lines.append("")
    lines.extend(
        [
            "## 后续规则",
            "",
            "- 只有1分钟主口径同时通过2024和2025，才开发R03.4.2.8B。",
            "- R03.4.2.8B必须用真实可执行保护位计算账户风险，比较P0/P1/P2/P3。",
            "- 若本阶段失败，停止Tranche方向，转向入场MAE优化或独立长趋势Sleeve。",
            "- 即使通过，本阶段也没有证明加仓提高总利润、降低MDD或维持总风险；这些必须由下一阶段账户级回测回答。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    *,
    config: TrancheEligibilityConfig,
    manifest: dict[str, object],
    preflight: dict[str, object],
    score_audit: pd.DataFrame,
    event_audit: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    occupancy: pd.DataFrame,
    atlas: pd.DataFrame,
    classes: pd.DataFrame,
    quarters: pd.DataFrame,
    tiers: pd.DataFrame,
    score_price: pd.DataFrame,
    risk_release: pd.DataFrame,
    gate: pd.DataFrame,
    causal_audit: pd.DataFrame,
    failures: pd.DataFrame,
    baseline_trades: pd.DataFrame,
    standalone_outcomes: pd.DataFrame,
    decision: str,
    reason: str,
) -> None:
    report_dir = config.report_path
    report_dir.mkdir(parents=True, exist_ok=True)
    _json(report_dir / "00_run_manifest.json", manifest)
    _json(report_dir / "01_preflight.json", preflight)
    _csv(report_dir / "02_score_threshold_audit.csv", score_audit)
    _csv(report_dir / "03_event_execution_audit.csv", event_audit)
    _csv(report_dir / "04_frozen_baseline_summary.csv", baseline_summary)
    _csv(report_dir / "05_occupancy_summary.csv", occupancy)
    _csv(report_dir / "06_occupied_signal_atlas.csv", atlas)
    _csv(report_dir / "07_signal_class_summary.csv", classes)
    _csv(report_dir / "08_eligible_quarter_summary.csv", quarters)
    _csv(report_dir / "09_score_tier_summary.csv", tiers)
    _csv(report_dir / "10_score_price_diagnostic.csv", score_price)
    _csv(report_dir / "11_candidate_risk_release_distribution.csv", risk_release)
    _csv(report_dir / "12_tranche_eligibility_gate.csv", gate)
    _csv(report_dir / "13_causal_audit.csv", causal_audit)
    _csv(report_dir / "14_failures.csv", failures)
    _csv(report_dir / "15_p0_failed_reclaim_trades.csv", baseline_trades)
    _csv(report_dir / "16_standalone_signal_outcomes.csv", standalone_outcomes)
    (report_dir / "99_decision.md").write_text(
        decision_markdown(
            decision,
            reason,
            baseline_summary=baseline_summary,
            occupancy=occupancy,
            classes=classes,
            score_price=score_price,
            gate=gate,
        ),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.2.8A",
            edge_id="q70_occupied_signal_tranche_eligibility",
            stage="research",
            title="ETH AI R03.4.2.8A occupied-signal atlas and tranche eligibility gate",
            decision_focus="whether occupied q70 signals contain one causal healthy/recovered subset that is independently positive in both 2024 and 2025 and therefore justifies a later account-risk tranche simulation",
            print_log=True,
        )
    )
