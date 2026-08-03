#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reports for R03.4 state-context opening-value ablation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import StateContextAblationConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _decision_markdown(
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    config: StateContextAblationConfig,
) -> str:
    lines = [
        "# R03.4 市场状态上下文对开仓价值模型的增量结论",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 本研究回答什么",
        "",
        "- 在完全相同的训练窗口、LightGBM参数、未来目标、成本和信号阈值下，比较是否加入冻结市场状态上下文。",
        "- 预测目标是未来6小时多头/空头可交易价值：MFE减去风险惩罚后的MAE。",
        "- 信号统一按下一分钟开盘进入；6小时收盘收益只作为诊断，不是最终实盘退出设计。",
        "- 2024和2025分别作为纯OOS；2026继续封存。",
        "",
        "## 消融组",
        "",
        "- `base_multiframe`：原多周期开仓特征。",
        "- `base_plus_activity`：仅增加活跃度状态。",
        "- `base_plus_directional_state`：增加战略、战术、入场及对齐关系。",
        "- `base_plus_all_state`：加入全部确定性状态上下文。",
        "- `base_plus_all_state_and_activity_persist`：再加入因果嵌套训练的活跃持续概率。",
        "- `state_only`：只使用压缩后的状态向量，作为解释性基准。",
        "",
        "## 判定原则",
        "",
        f"- 状态版本必须在2024和2025都至少提高方向Rank IC {config.minimum_rank_ic_increment:.3f}。",
        "- 校准期固定90%阈值后的成本后平均收益不得低于基线。",
        f"- 每年信号至少 {config.minimum_signal_count} 个，避免靠极低频样本通过。",
        "- 2倍成本结果单独报告，不因基础成本通过就宣称可交易。",
        "",
    ]
    if champion:
        lines.extend(
            [
                "## 最佳稳定增量候选",
                "",
                f"- 版本：`{champion.get('variant')}`",
                f"- 稳定性分数：{float(champion.get('stability_score', float('nan'))):.6f}",
                "",
            ]
        )
    lines.extend(
        [
            "## 重要限制",
            "",
            "- 这仍是开仓价值/方向诊断，不是最终策略回测。",
            "- 6小时固定收盘只用于同口径比较；最终退出必须继续研究结构止损、保护利润和持仓管理。",
            "- 只有状态上下文在两个OOS年份都产生增量，才允许进入正式低MAE开仓模型。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    report_dir: Path,
    *,
    manifest: dict[str, object],
    preflight: dict[str, object],
    target_distribution: pd.DataFrame,
    model_metrics: pd.DataFrame,
    signal_metrics: pd.DataFrame,
    uplift: pd.DataFrame,
    stable_candidates: pd.DataFrame,
    feature_importance: pd.DataFrame,
    samples: pd.DataFrame,
    failures: pd.DataFrame,
    causal_audit: pd.DataFrame,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    config: StateContextAblationConfig,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _json(report_dir / "00_run_manifest.json", manifest)
    _json(report_dir / "01_preflight.json", preflight)
    _json(
        report_dir / "02_ablation_contract.json",
        {
            "variants": list(config.variants),
            "primary_horizon_hours": config.primary_horizon_hours,
            "risk_penalty": config.risk_penalty,
            "base_round_trip_cost": config.base_round_trip_cost,
            "note": "Only the state-context feature set changes between variants.",
        },
    )
    _csv(report_dir / "03_target_distribution.csv", target_distribution)
    _csv(report_dir / "04_model_metrics.csv", model_metrics)
    _csv(report_dir / "05_cost_aware_signal_metrics.csv", signal_metrics)
    _csv(report_dir / "06_uplift_vs_base.csv", uplift)
    _csv(report_dir / "07_stable_uplift_candidates.csv", stable_candidates)
    _csv(report_dir / "08_feature_importance.csv", feature_importance)
    _csv(report_dir / "09_prediction_samples.csv", samples)
    _csv(report_dir / "10_model_failures.csv", failures)
    _csv(report_dir / "11_causal_audit.csv", causal_audit)
    _json(report_dir / "98_champion.json", champion or {})
    (report_dir / "99_decision.md").write_text(
        _decision_markdown(decision, reason, champion, config), encoding="utf-8"
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4",
            edge_id="state_context_opening_value_ablation",
            stage="research",
            title="ETH AI R03.4 state-context opening-value ablation",
            decision_focus="whether frozen market-state context improves OOS directional opening value",
            print_log=True,
        )
    )
