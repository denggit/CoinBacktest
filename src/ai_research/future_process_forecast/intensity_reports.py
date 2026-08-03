#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Durable reports for R03.3.2 continuous intensity research."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .intensity_config import CURRENT_STATE_DEFINITION, FutureIntensityConfig


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def decision_markdown(
    *,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    config: FutureIntensityConfig,
) -> str:
    lines = [
        "# R03.3.2 连续未来机会强度预测结论",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 当下市场状态如何定义",
        "",
        "- 当下状态不是一个由未来结果反推的离散标签，而是决策时刻已经可见的因果状态向量。",
        "- 状态向量包含长期结构、中周期推动/回调、短周期位置、波动阶段、订单流冲击和成交活跃度。",
        "- 模型训练目标是未来6/12小时的连续机会强度；因此不会因为刚好没达到某个人工事件阈值就把整段行情判成无机会。",
        "- 2026年上半年继续封存，不参与训练、选型或展示。",
        "",
        "## 连续目标",
        "",
        "- `future_range_pct`：未来窗口最高价到最低价的完整区间。",
        "- `future_max_directional_pct`：未来向上或向下空间中较大的一侧。",
        "- `future_two_sided_pct`：未来向上与向下空间中较小的一侧，用于识别双向短线机会。",
        "- `future_range_atr_multiple`：未来完整区间相对当前因果4H ATR尺度的倍数。",
        "",
        "## 通过标准",
        "",
        f"- 2024与2025 Rank IC都至少为 {config.minimum_rank_ic:.2f}。",
        f"- 两年Top Decile实际机会强度都至少为全样本的 {config.minimum_top_decile_lift:.2f} 倍。",
        f"- 两年预测十分位与实际均值的单调性都至少为 {config.minimum_decile_monotonicity:.2f}。",
        "- 重点判断排序是否跨年稳定，而不是要求绝对点预测完全准确。",
        "",
    ]
    if champion:
        lines.extend(
            [
                "## 最佳候选",
                "",
                f"- 架构：`{champion.get('architecture')}`",
                f"- 目标：`{champion.get('target')}`",
                f"- 2024 Rank IC：{float(champion.get('WF_2024_rank_ic', float('nan'))):.4f}",
                f"- 2025 Rank IC：{float(champion.get('WF_2025_rank_ic', float('nan'))):.4f}",
                f"- 2024 Top Decile Lift：{float(champion.get('WF_2024_top_decile_lift', float('nan'))):.3f}",
                f"- 2025 Top Decile Lift：{float(champion.get('WF_2025_top_decile_lift', float('nan'))):.3f}",
                "",
            ]
        )
    lines.extend(
        [
            "## 研究纪律",
            "",
            "- 若连续强度排序仍无法跨年稳定，停止继续堆普通价格/Trade Bar特征。",
            "- 若强度排序稳定，再把它作为AI Bot的环境评分，不直接据此开仓。",
            "- 方向、具体入场和退出仍需后续独立模型或结构规则。",
            "",
        ]
    )
    return "\n".join(lines)


def write_intensity_reports(
    report_dir: Path,
    *,
    manifest: dict[str, object],
    preflight: dict[str, object],
    target_distribution: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    bucket_metrics: pd.DataFrame,
    quantile_metrics: pd.DataFrame,
    candidates: pd.DataFrame,
    micro_increment: pd.DataFrame,
    feature_importance: pd.DataFrame,
    prediction_samples: pd.DataFrame,
    failures: pd.DataFrame,
    causal_audit: pd.DataFrame,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    config: FutureIntensityConfig,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _json(report_dir / "00_run_manifest.json", manifest)
    _json(report_dir / "01_preflight.json", preflight)
    _json(report_dir / "02_current_state_definition.json", CURRENT_STATE_DEFINITION)
    _csv(report_dir / "03_target_distribution.csv", target_distribution)
    _csv(report_dir / "04_regression_metrics.csv", regression_metrics)
    _csv(report_dir / "05_prediction_decile_curve.csv", bucket_metrics)
    _csv(report_dir / "06_calibration_threshold_metrics.csv", quantile_metrics)
    _csv(report_dir / "07_stable_candidates.csv", candidates)
    _csv(report_dir / "08_micro_increment.csv", micro_increment)
    _csv(report_dir / "09_feature_importance.csv", feature_importance)
    _csv(report_dir / "10_prediction_samples.csv", prediction_samples)
    _csv(report_dir / "11_model_failures.csv", failures)
    _csv(report_dir / "12_causal_audit.csv", causal_audit)
    _json(report_dir / "98_champion.json", champion or {})
    (report_dir / "99_decision.md").write_text(
        decision_markdown(decision=decision, reason=reason, champion=champion, config=config),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.3.2",
            edge_id="future_opportunity_intensity",
            stage="research",
            title="ETH AI R03.3.2 continuous future opportunity intensity",
            decision_focus="whether continuous future opportunity ranking is stable across years",
            print_log=True,
        )
    )
