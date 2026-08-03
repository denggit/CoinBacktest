#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.5."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import FailureOverlayConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    *,
    stable: pd.DataFrame,
    policy_summary: pd.DataFrame,
    tier_summary: pd.DataFrame,
    upgrade_diagnostics: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.5 q70极高置信坏单退出Overlay",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 冻结边界",
        "",
        "- q70已经通过R03.4.2.4跨年审核，本研究不删除q70-q80、q80-q90或q90+任何分数层。",
        "- 分数层只决定需要多强的坏单证据；更高分事件获得更高容忍度，不直接映射实盘仓位。",
        "- T+60只预警；T+180必须同时出现极高失败概率与多项价格结构恶化才允许退出。",
        "- 3%安全底线只用于黑天鹅风险诊断，触发后按下一分钟开盘成交，不假设完美止损价。",
        "- 固定6小时仍然只是收益基准，不是最终实盘退出。",
        "- 持仓中评分升级只做加仓价值诊断，本阶段不执行加仓。",
        "",
    ]
    if not stable.empty:
        lines.extend(["## 跨年候选", ""])
        for row in stable.itertuples():
            lines.append(
                f"- {row.policy}: 2024/2025 2x净期望={row.mean_net_2x_2024:.3%}/{row.mean_net_2x_2025:.3%}, "
                f"PF={row.pf_2x_2024:.2f}/{row.pf_2x_2025:.2f}, "
                f"退出占比={row.overlay_exit_share_2024:.1%}/{row.overlay_exit_share_2025:.1%}, "
                f"单次退出增量={row.overlay_uplift_2024:.3%}/{row.overlay_uplift_2025:.3%}, "
                f"超过原基准={bool(row.beats_fixed6h_both_years)}, 超过安全底线基准={bool(row.beats_safety_baseline_both_years)}。"
            )
        lines.append("")
    if not policy_summary.empty:
        lines.extend(["## 主要策略结果（1分钟延迟、2倍成本）", ""])
        focus = policy_summary.loc[(policy_summary["delay_minutes"] == 1) & (policy_summary["cost_multiplier"] == 2.0)]
        for row in focus.sort_values(["policy", "fold_id"]).itertuples():
            lines.append(
                f"- {row.fold_id} {row.policy}: trades={int(row.trades)}, mean={row.mean_net_return:.3%}, "
                f"PF={row.profit_factor:.2f}, win={row.win_rate:.1%}, MDD={row.max_drawdown:.1%}, "
                f"overlay={row.overlay_exit_share:.1%}, false-exit={row.overlay_false_exit_share:.1%}。"
            )
        lines.append("")
    if not tier_summary.empty:
        lines.extend(["## 分数层保留审核", ""])
        focus = tier_summary.loc[
            (tier_summary["policy"] == "tiered_failure_overlay")
            & (tier_summary["delay_minutes"] == 1)
            & (tier_summary["cost_multiplier"] == 2.0)
        ]
        for row in focus.sort_values(["fold_id", "score_tier"]).itertuples():
            lines.append(
                f"- {row.fold_id} {row.score_tier}: trades={int(row.trades)}, mean={row.mean_net_return:.3%}, "
                f"PF={row.profit_factor:.2f}, overlay={row.overlay_exit_share:.1%}。"
            )
        lines.append("")
    if not upgrade_diagnostics.empty:
        lines.extend(["## 持仓中评分升级诊断", ""])
        for row in upgrade_diagnostics.sort_values(["fold_id", "checkpoint_minutes", "score_tier", "score_upgraded"]).itertuples():
            lines.append(
                f"- {row.fold_id} T+{int(row.checkpoint_minutes)}m {row.score_tier} upgraded={bool(row.score_upgraded)}: events={int(row.events)}, "
                f"6h毛收益={row.mean_fixed6h_gross_return:.3%}, 胜率={row.fixed6h_win_rate_1x:.1%}, "
                f"持续失败率={row.persistent_failure_rate:.1%}。"
            )
        lines.append("")
    lines.extend(
        [
            "## 下一阶段",
            "",
            "- 若Overlay跨年提高利润：先冻结坏单退出，再独立研究第6小时后的增量持仓价值。",
            "- 若Overlay只降低尾部风险但不提高利润：可以保留为风险保险候选，但不能替代主要退出逻辑。",
            "- 最终策略将使用结构失效、风险底线、利润保护与选择性长持，不采用机械持仓时间上限。",
            "",
        ]
    )
    return "\n".join(lines)


def empty(config: FailureOverlayConfig, preflight: dict[str, object], decision: str, reason: str) -> None:
    frame = pd.DataFrame()
    write_reports(
        config=config,
        preflight=preflight,
        manifest={"stage": "R03.4.2.5", "config": config.to_dict()},
        entry_oof_audit=frame,
        extraction_audit=frame,
        model_selection=frame,
        model_metrics=frame,
        thresholds=frame,
        importance=frame,
        predictions=frame,
        policy_summary=frame,
        period_summary=frame,
        tier_summary=frame,
        exit_summary=frame,
        overlap_audit=frame,
        upgrade_diagnostics=frame,
        stable=frame,
        causal_audit=frame,
        failures=frame,
        trades=frame,
        decision=decision,
        reason=reason,
    )


def write_reports(
    *,
    config: FailureOverlayConfig,
    preflight: dict[str, object],
    manifest: dict[str, object],
    entry_oof_audit: pd.DataFrame,
    extraction_audit: pd.DataFrame,
    model_selection: pd.DataFrame,
    model_metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    importance: pd.DataFrame,
    predictions: pd.DataFrame,
    policy_summary: pd.DataFrame,
    period_summary: pd.DataFrame,
    tier_summary: pd.DataFrame,
    exit_summary: pd.DataFrame,
    overlap_audit: pd.DataFrame,
    upgrade_diagnostics: pd.DataFrame,
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
    _csv(report_dir / "02_entry_oof_audit.csv", entry_oof_audit)
    _csv(report_dir / "03_event_extraction_audit.csv", extraction_audit)
    _csv(report_dir / "04_model_selection_audit.csv", model_selection)
    _csv(report_dir / "05_model_metrics.csv", model_metrics)
    _csv(report_dir / "06_probability_thresholds.csv", thresholds)
    _csv(report_dir / "07_feature_importance.csv", importance)
    _csv(report_dir / "08_prediction_samples.csv", predictions)
    _csv(report_dir / "09_policy_summary.csv", policy_summary)
    _csv(report_dir / "10_quarter_summary.csv", period_summary)
    _csv(report_dir / "11_score_tier_policy_summary.csv", tier_summary)
    _csv(report_dir / "12_exit_reason_summary.csv", exit_summary)
    _csv(report_dir / "13_overlap_audit.csv", overlap_audit)
    _csv(report_dir / "14_score_upgrade_diagnostics.csv", upgrade_diagnostics)
    _csv(report_dir / "15_stable_candidates.csv", stable)
    _csv(report_dir / "16_causal_audit.csv", causal_audit)
    _csv(report_dir / "17_failures.csv", failures)
    _csv(report_dir / "18_trade_details.csv", trades)
    (report_dir / "99_decision.md").write_text(
        decision_markdown(
            decision,
            reason,
            stable=stable,
            policy_summary=policy_summary,
            tier_summary=tier_summary,
            upgrade_diagnostics=upgrade_diagnostics,
        ),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.2.5",
            edge_id="q70_high_confidence_failure_overlay",
            stage="research",
            title="ETH AI R03.4.2.5 q70 high-confidence persistent-failure overlay",
            decision_focus="whether causal T+60 warning plus score-tier T+180 path confirmation can cut only the most certain losing q70 trades without sacrificing positive expectancy or total profit",
            print_log=True,
        )
    )
