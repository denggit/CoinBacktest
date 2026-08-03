#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.6."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import IncrementalHoldConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    *,
    stable: pd.DataFrame,
    metrics: pd.DataFrame,
    tier_diagnostics: pd.DataFrame,
    score_ablation: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.6 增量持仓价值与非时间退出信号研究",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 本阶段回答的问题",
        "",
        "- 不是预测‘持有多久’，而是比较在当前检查点立即平仓与继续持有到下一个决策节点的增量价值。",
        "- 检查点只是重新观察，不是强制退出时间；120小时只是标签与审计窗口，不能作为最终实盘时间止损。",
        "- q70完整保留，并持续拆分q70-q80、q80-q90与q90+三个风险层。",
        "- 3%灾难保护继续作为独立风险保险，不允许机器学习模型取代交易所硬止损。",
        "",
    ]
    if not stable.empty:
        lines.extend(["## 跨年候选", ""])
        for row in stable.head(20).itertuples():
            lines.append(
                f"- T+{int(row.checkpoint_minutes)}m {row.target} / {row.feature_set} / {row.scope}: "
                f"2024/2025 Rank IC={float(getattr(row, 'WF_2024_rank_ic', float('nan'))):.3f}/"
                f"{float(getattr(row, 'WF_2025_rank_ic', float('nan'))):.3f}, "
                f"top-bottom spread={float(getattr(row, 'WF_2024_top_bottom_spread', float('nan'))):.3%}/"
                f"{float(getattr(row, 'WF_2025_top_bottom_spread', float('nan'))):.3%}, "
                f"pass={bool(row.passes_cross_year)}。"
            )
        lines.append("")
    if not metrics.empty:
        lines.extend(["## OOS模型概览", ""])
        focus = metrics.loc[metrics["scope"] == "broad_q70"].sort_values(
            ["checkpoint_minutes", "target", "fold_id", "rank_ic"], ascending=[True, True, True, False]
        )
        for row in focus.groupby(["fold_id", "checkpoint_minutes", "target"], sort=False).head(1).itertuples():
            lines.append(
                f"- {row.fold_id} T+{int(row.checkpoint_minutes)}m {row.target}: {row.feature_set}, "
                f"Rank IC={row.rank_ic:.3f}, sign AUC={row.sign_auc:.3f}, "
                f"top-bottom={row.top_bottom_spread:.3%}。"
            )
        lines.append("")
    if not tier_diagnostics.empty:
        lines.extend(["## 分数层增量价值", ""])
        focus = tier_diagnostics.loc[
            (tier_diagnostics["scope"] == "broad_q70")
            & (tier_diagnostics["target"] == "next_incremental_utility")
        ]
        for row in focus.sort_values(["fold_id", "checkpoint_minutes", "score_tier"]).itertuples():
            lines.append(
                f"- {row.fold_id} T+{int(row.checkpoint_minutes)}m {row.score_tier}: n={int(row.rows)}, "
                f"实际增量价值均值={row.actual_mean:.3%}, 正增量率={row.actual_positive_rate:.1%}。"
            )
        lines.append("")
    if not score_ablation.empty:
        lines.extend(["## 开仓分数是否适合持仓", ""])
        for row in score_ablation.sort_values(["fold_id", "checkpoint_minutes", "target"]).itertuples():
            lines.append(
                f"- {row.fold_id} T+{int(row.checkpoint_minutes)}m {row.target}: path IC={row.path_rank_ic:.3f}, "
                f"path+score IC={row.path_plus_score_rank_ic:.3f}, score-only IC={row.score_only_rank_ic:.3f}, "
                f"增量={row.score_increment_rank_ic:.3f}。"
            )
        lines.append("")
    lines.extend(
        [
            "## 后续计划",
            "",
            "1. 只有本阶段出现跨年稳定的增量价值排序，下一阶段才构建循环持仓状态机。",
            "2. 状态机将反复执行：继续持有价值为正则持有；结构失效或增量价值为负则退出；盈利回吐风险上升则保护利润。",
            "3. q70-q80、q80-q90、q90+将分别配置初始风险；新高分信号只有在原仓风险已降低且结构确认后才允许研究加仓。",
            "4. 最终策略不采用机械持仓时间上限；研究窗口末端的未结束仓位必须作为右删失样本处理，不能伪装成正常时间退出。",
            "",
        ]
    )
    return "\n".join(lines)


def empty(config: IncrementalHoldConfig, preflight: dict[str, object], decision: str, reason: str) -> None:
    frame = pd.DataFrame()
    write_reports(
        config=config,
        preflight=preflight,
        manifest={"stage": "R03.4.2.6", "config": config.to_dict()},
        entry_oof_audit=frame,
        extraction_audit=frame,
        model_selection=frame,
        model_metrics=frame,
        deciles=frame,
        thresholds=frame,
        action_diagnostics=frame,
        tier_diagnostics=frame,
        score_ablation=frame,
        importance=frame,
        predictions=frame,
        stable=frame,
        causal_audit=frame,
        failures=frame,
        decision=decision,
        reason=reason,
    )


def write_reports(
    *,
    config: IncrementalHoldConfig,
    preflight: dict[str, object],
    manifest: dict[str, object],
    entry_oof_audit: pd.DataFrame,
    extraction_audit: pd.DataFrame,
    model_selection: pd.DataFrame,
    model_metrics: pd.DataFrame,
    deciles: pd.DataFrame,
    thresholds: pd.DataFrame,
    action_diagnostics: pd.DataFrame,
    tier_diagnostics: pd.DataFrame,
    score_ablation: pd.DataFrame,
    importance: pd.DataFrame,
    predictions: pd.DataFrame,
    stable: pd.DataFrame,
    causal_audit: pd.DataFrame,
    failures: pd.DataFrame,
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
    _csv(report_dir / "06_prediction_deciles.csv", deciles)
    _csv(report_dir / "07_oof_decision_thresholds.csv", thresholds)
    _csv(report_dir / "08_action_diagnostics.csv", action_diagnostics)
    _csv(report_dir / "09_score_tier_diagnostics.csv", tier_diagnostics)
    _csv(report_dir / "10_score_ablation.csv", score_ablation)
    _csv(report_dir / "11_feature_importance.csv", importance)
    _csv(report_dir / "12_prediction_samples.csv", predictions)
    _csv(report_dir / "13_stable_candidates.csv", stable)
    _csv(report_dir / "14_causal_audit.csv", causal_audit)
    _csv(report_dir / "15_failures.csv", failures)
    (report_dir / "99_decision.md").write_text(
        decision_markdown(
            decision,
            reason,
            stable=stable,
            metrics=model_metrics,
            tier_diagnostics=tier_diagnostics,
            score_ablation=score_ablation,
        ),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.2.6",
            edge_id="q70_incremental_holding_value",
            stage="research",
            title="ETH AI R03.4.2.6 incremental holding value and non-time exit signal research",
            decision_focus="whether causal price-path structure can predict the incremental utility of continuing an existing q70 long position versus exiting now, without treating time as the exit rule",
            print_log=True,
        )
    )
