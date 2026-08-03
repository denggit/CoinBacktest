#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.2 causal path recognition."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import LongTailPathRecognitionConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    *,
    metrics: pd.DataFrame,
    stable: pd.DataFrame,
    action_diagnostics: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.2 q90/q70多头事件早期因果路径识别",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 研究目标",
        "",
        "- 开仓模型只负责识别‘此刻新开多是否有未来空间’，不再被当成持仓续期模型。",
        "- 在T+60/T+180/T+360分钟，使用当时已经发生的价格路径与结构判断持续失败、可恢复回撤、冲高回吐和6小时后续涨。",
        "- 允许健康交易持有24至48小时；重点是尽早识别真正会持续亏损的交易，而不是强制短持仓。",
        "- q90仍是主要OOS基准；q70只作为扩展机会池，检验是否能在不依赖极高开仓分数的情况下找到健康事件。",
        "",
        "## 不允许的解释",
        "",
        "- 本阶段只识别路径健康度，不生成最终止损、止盈或续期策略。",
        "- 未来24/48小时只用于历史标签，不能作为检查点特征。",
        "- 高风险概率不能直接视为实盘平仓指令；必须在下一阶段冻结规则后重新做完整OOS交易回测。",
        "- 已舍弃的市场状态模型不被加载。",
        "",
    ]
    if not stable.empty:
        passed = stable.loc[stable["stable_signal"] == True]  # noqa: E712
        lines.extend(["## 跨年稳定信号", ""])
        if passed.empty:
            lines.append("- 没有任务在2024和2025同时通过预注册门槛。")
        else:
            for row in passed.itertuples():
                lines.append(
                    f"- {row.task} @ T+{int(row.checkpoint_minutes)}m / {row.feature_set}: "
                    f"最低AUC={row.minimum_auc:.3f}, 最低Top-Decile Lift={row.minimum_top_decile_lift:.2f}。"
                )
        lines.append("")
    if not action_diagnostics.empty:
        lines.extend(["## 检查点诊断", ""])
        primary = action_diagnostics.loc[action_diagnostics["scope"] == "primary_q90"]
        for row in primary.head(12).itertuples():
            lines.append(
                f"- {row.fold_id} {row.task} T+{int(row.checkpoint_minutes)}m {row.feature_set}: "
                f"flagged={int(row.flagged_rows)}, precision={row.flagged_positive_rate:.2%}, "
                f"checkpoint-vs-hold6h={row.mean_checkpoint_advantage_vs_6h:.3%}。"
            )
        lines.append("")
    lines.extend(
        [
            "## 下一步",
            "",
            "若持续失败、恢复或续持任务有跨年稳定信号，R03.4.2.3才允许把概率转成少量冻结的差异化退出/续期规则，并重新验证正期望、PF、回撤、成本和延迟。",
            "若只有q70安全桶有正期望，也只能把它视为增加交易次数的候选，不能在本阶段直接扩仓或上线。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    report_dir: Path,
    *,
    manifest: dict[str, object],
    preflight: dict[str, object],
    contract: dict[str, object],
    extraction_audit: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    metrics: pd.DataFrame,
    deciles: pd.DataFrame,
    action_diagnostics: pd.DataFrame,
    broad_pool_diagnostics: pd.DataFrame,
    score_ablation: pd.DataFrame,
    importance: pd.DataFrame,
    predictions: pd.DataFrame,
    representatives: pd.DataFrame,
    stable: pd.DataFrame,
    causal_audit: pd.DataFrame,
    failures: pd.DataFrame,
    decision: str,
    reason: str,
    config: LongTailPathRecognitionConfig,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _json(report_dir / "00_run_manifest.json", manifest)
    _json(report_dir / "01_preflight.json", preflight)
    _json(report_dir / "02_path_recognition_contract.json", contract)
    _csv(report_dir / "03_event_extraction_audit.csv", extraction_audit)
    _csv(report_dir / "04_task_dataset_summary.csv", dataset_summary)
    _csv(report_dir / "05_model_metrics.csv", metrics)
    _csv(report_dir / "06_probability_deciles.csv", deciles)
    _csv(report_dir / "07_checkpoint_action_diagnostics.csv", action_diagnostics)
    _csv(report_dir / "08_broad_q70_safe_bucket.csv", broad_pool_diagnostics)
    _csv(report_dir / "09_score_path_ablation.csv", score_ablation)
    _csv(report_dir / "10_feature_importance.csv", importance)
    _csv(report_dir / "11_prediction_samples.csv", predictions)
    _csv(report_dir / "12_representative_cases.csv", representatives)
    _csv(report_dir / "13_stable_candidates.csv", stable)
    _csv(report_dir / "14_causal_audit.csv", causal_audit)
    _csv(report_dir / "15_failures.csv", failures)
    (report_dir / "99_decision.md").write_text(
        decision_markdown(
            decision,
            reason,
            metrics=metrics,
            stable=stable,
            action_diagnostics=action_diagnostics,
        ),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.2.2",
            edge_id="long_tail_causal_path_health_recognition",
            stage="research",
            title="ETH AI R03.4.2.2 causal path-health and long-hold recognition",
            decision_focus="whether early causal price structure can separate persistent failures, recoverable drawdowns and safe long holds without relying on entry score persistence",
            print_log=True,
        )
    )
