#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import LongStateCalibrationConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    config: LongStateCalibrationConfig,
) -> str:
    lines = [
        "# R03.4.1 多头机会二阶段软状态校准结论",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 研究边界",
        "",
        "- 第一阶段只用原R03.4多周期基础特征预测未来6小时多头价值。",
        "- 第二阶段只接收冻结基础分数和少量软状态，不再把战术/入场离散多空标签塞回主模型。",
        "- 元模型使用扩展窗口OOF基础预测训练，不能看到同一行基础模型的拟合内预测。",
        "- 2024、2025分别纯OOS，2026继续封存。",
        "- 固定6小时收盘和成本结果仍是诊断，不是最终退出策略。",
        "",
        "## 软状态",
        "",
        "- 战略：连续分数、状态年龄、边界距离、24小时翻转率。",
        "- 活跃度：1D/4H/1H/15m活跃分数、年龄、边界距离、6/24小时翻转率。",
        "- 明确排除战术与入场离散方向标签及其方向分数。",
        "",
        "## 严格对照",
        "",
        "- `base_identity`：冻结基础多头分数。",
        "- `score_only_meta`：只对基础分数做非线性校准，作为真正的元模型控制组。",
        "- 状态版本必须稳定超过`score_only_meta`，而不是只超过未校准原分数。",
        "- 另用共同基础候选池重排，确保状态只筛选基础模型已经发现的机会。",
        "- 仓位倍率只在完全相同的基础事件上比较，不允许靠改变信号数量制造提升。",
        "",
        "## 通过门槛",
        "",
        f"- 两个OOS年份多头价值Rank IC增量均至少 {config.minimum_long_utility_ic_increment:.3f}。",
        "- 共同候选池重排后的多头价值不得下降。",
        f"- MAE最多允许恶化 {config.maximum_mae_worsening:.4%}。",
        f"- 每年独立事件至少 {config.minimum_independent_events} 个。",
        "",
    ]
    if champion:
        lines.extend(
            [
                "## 最佳候选",
                "",
                f"- 版本：`{champion.get('variant')}`",
                f"- 稳定性分数：{float(champion.get('stability_score', float('nan'))):.6f}",
                "",
            ]
        )
    lines.extend(
        [
            "## 解释限制",
            "",
            "- 通过只表示状态对多头机会排序或风险倍率有稳定增量，不表示已经得到可实盘策略。",
            "- 若只通过风险倍率，不允许把状态升级成开仓触发器。",
            "- 若两个年份不一致，应停止继续堆状态特征，转向真正的方向与低MAE启动模型。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    report_dir: Path,
    *,
    manifest: dict[str, object],
    preflight: dict[str, object],
    oof_audit: pd.DataFrame,
    model_metrics: pd.DataFrame,
    signal_metrics: pd.DataFrame,
    rerank_metrics: pd.DataFrame,
    multiplier_metrics: pd.DataFrame,
    uplift: pd.DataFrame,
    stable: pd.DataFrame,
    importance: pd.DataFrame,
    samples: pd.DataFrame,
    causal_audit: pd.DataFrame,
    failures: pd.DataFrame,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    config: LongStateCalibrationConfig,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _json(report_dir / "00_run_manifest.json", manifest)
    _json(report_dir / "01_preflight.json", preflight)
    _json(
        report_dir / "02_calibration_contract.json",
        {
            "variants": list(config.variants),
            "base_target": "long_utility_h6 = long_mfe_h6 - 1.25 * long_mae_h6",
            "state_role": "post-hoc calibration and fixed-candidate position multiplier only",
            "excluded": ["tactical_state", "entry_state", "tactical_score", "entry_score"],
        },
    )
    _csv(report_dir / "03_oof_stacking_audit.csv", oof_audit)
    _csv(report_dir / "04_model_metrics.csv", model_metrics)
    _csv(report_dir / "05_variant_threshold_event_metrics.csv", signal_metrics)
    _csv(report_dir / "06_common_candidate_rerank_metrics.csv", rerank_metrics)
    _csv(report_dir / "07_fixed_candidate_multiplier_metrics.csv", multiplier_metrics)
    _csv(report_dir / "08_uplift_vs_controls.csv", uplift)
    _csv(report_dir / "09_stable_candidates.csv", stable)
    _csv(report_dir / "10_meta_feature_importance.csv", importance)
    _csv(report_dir / "11_prediction_samples.csv", samples)
    _csv(report_dir / "12_causal_audit.csv", causal_audit)
    _csv(report_dir / "13_model_failures.csv", failures)
    _json(report_dir / "98_champion.json", champion or {})
    (report_dir / "99_decision.md").write_text(
        decision_markdown(decision, reason, champion, config), encoding="utf-8"
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.1",
            edge_id="long_opportunity_soft_state_meta_calibration",
            stage="research",
            title="ETH AI R03.4.1 long opportunity soft-state meta calibration",
            decision_focus="whether soft strategic/activity context improves long opportunity ranking or risk sizing after the base model",
            print_log=True,
        )
    )
