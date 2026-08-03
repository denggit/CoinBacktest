#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Durable reports for R03.3.3.1 market-state calibration and audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import MarketStateContinuityConfig


STATE_DEFINITION = {
    "purpose": "auxiliary market context for future direction, entry, holding and risk models; never a direct order trigger",
    "strategic_layer": {
        "timeframes": ["1D", "4H"],
        "meaning": "long direction, trend age, long drawdown/rebound and slow volatility regime",
        "expected_duration": "days to months",
    },
    "tactical_layer": {
        "timeframes": ["4H", "1H"],
        "meaning": "impulse, pullback, recovery, compression, expansion and exhaustion",
        "expected_duration": "hours to days",
    },
    "entry_layer": {
        "timeframes": ["30m", "15m", "5m", "1m"],
        "meaning": "local decline, stabilisation, recovery, momentum and short-term instability",
        "expected_duration": "minutes to hours",
    },
    "activity_layer": {
        "meaning": "relative realised activity from strategic through entry horizons",
        "expected_duration": "hours to days",
    },
    "representation": "continuous scores plus causal hysteretic states, ages, flip rates and cross-layer alignment",
    "state_count": {
        "layers": 4,
        "discrete_states_per_layer": 3,
        "layer_level_named_states": 12,
        "theoretical_joint_combinations": 81,
        "note": "The model does not treat 81 combinations as 81 separate classes; four state dimensions coexist and are also retained as continuous scores.",
    },
    "discrete_state_meanings": {
        "strategic": {"-1": "long-horizon bearish", "0": "strategic neutral/range", "1": "long-horizon bullish"},
        "tactical": {"-1": "downward impulse/pullback", "0": "tactical neutral/compression", "1": "upward impulse/recovery"},
        "entry": {"-1": "local bearish/declining", "0": "local stabilising/neutral", "1": "local bullish/recovering"},
        "activity": {"-1": "low/compressed activity", "0": "normal activity", "1": "high/expanding activity"},
    },
    "future_usage": [
        "direction model context",
        "entry model context",
        "position-management context",
        "risk scaling and state-transition warning",
    ],
}


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def decision_markdown(
    *,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    config: MarketStateContinuityConfig,
) -> str:
    lines = [
        "# R03.3.3.1 市场状态连续性小修正与审计结论",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 研究定位",
        "",
        "- 市场状态模型不是开仓器。",
        "- 它向未来的方向、入场、持仓管理和风险模型提供战略/战术/入场层上下文。",
        "- 战略层使用前一日及更早历史的因果分位阈值；其他层使用冻结迟滞阈值，避免固定战略阈值长期不可达。\n- 状态使用连续强度、因果迟滞、状态年龄、边界距离、翻转率和多周期对齐表达。",
        "- 所有训练目标仍围绕交易价值：状态是否持续、何时可能转换，以及不同状态组合对应的未来6小时机会厚度。",
        "",
        "## 数据合同",
        "",
        "- 2020—2021：通过 `src.data_feed.okx_loader.OKXDataLoader` 读取普通1m OHLCV。",
        "- 2022—2025：通过 `src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader` 读取1m Trade Bar。",
        "- Universal分支只使用所有年份共有的OHLCV因果特征。",
        "- Trade-enhanced分支仅在真实Trade特征存在的2022年以后训练，不把早期缺失字段填0。",
        "- 2026年上半年保持封存，不参与训练、选型或报告展示。",
        "",
        "## 验收标准",
        "",
        f"- 同一连续性任务在2024与2025 AUC都至少为 {config.minimum_auc:.2f}。",
        "- 两年Brier Skill均不低于常数概率基准。",
        f"- 最低持续概率十分位的状态转换Lift两年都至少为 {config.minimum_transition_lift:.2f}。",
        "- 持续标签要求整个预测窗口内没有任何状态切换，不能只比较起点和终点。\n- 完整模型必须与“状态年龄/边界距离/当前状态”机械基准对照。\n- 连续低持续概率信号必须合并为独立转换预警段再审核。",
        "- 最终仍需后续方向/入场模型证明真实成本后的PF和净期望改善。",
        "",
    ]
    if champion:
        lines.extend(
            [
                "## 最佳连续性候选",
                "",
                f"- 架构：`{champion.get('architecture')}`",
                f"- 目标：`{champion.get('target')}`",
                f"- 2024 AUC：{float(champion.get('WF_2024_auc', float('nan'))):.4f}",
                f"- 2025 AUC：{float(champion.get('WF_2025_auc', float('nan'))):.4f}",
                f"- 2024转换Lift：{float(champion.get('WF_2024_transition_lift', float('nan'))):.3f}",
                f"- 2025转换Lift：{float(champion.get('WF_2025_transition_lift', float('nan'))):.3f}",
                "",
            ]
        )
    lines.extend(
        [
            "## 下一步纪律",
            "",
            "- 若状态持续与转换不可跨年预测，不继续堆更多硬状态标签。",
            "- 若通过，下一阶段把状态向量作为方向和低MAE入场模型的辅助输入，不直接触发订单。",
            "- 强化学习仍留到已有稳定Edge后的持仓与执行优化阶段。",
            "",
        ]
    )
    return "\n".join(lines)


def write_state_continuity_reports(
    report_dir: Path,
    *,
    manifest: dict[str, object],
    preflight: dict[str, object],
    duration_atlas: pd.DataFrame,
    target_distribution: pd.DataFrame,
    opportunity_link: pd.DataFrame,
    model_metrics: pd.DataFrame,
    decile_curves: pd.DataFrame,
    candidates: pd.DataFrame,
    attribution: pd.DataFrame,
    trade_increment: pd.DataFrame,
    feature_importance: pd.DataFrame,
    state_samples: pd.DataFrame,
    prediction_samples: pd.DataFrame,
    strategic_threshold_audit: pd.DataFrame,
    mechanical_baseline_metrics: pd.DataFrame,
    mechanical_increment_audit: pd.DataFrame,
    transition_alert_metrics: pd.DataFrame,
    transition_alert_episodes: pd.DataFrame,
    failures: pd.DataFrame,
    causal_audit: pd.DataFrame,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    config: MarketStateContinuityConfig,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _json(report_dir / "00_run_manifest.json", manifest)
    _json(report_dir / "01_preflight.json", preflight)
    _json(report_dir / "02_state_definition.json", STATE_DEFINITION)
    _csv(report_dir / "03_state_duration_atlas.csv", duration_atlas)
    _csv(report_dir / "04_continuity_target_distribution.csv", target_distribution)
    _csv(report_dir / "05_state_opportunity_link.csv", opportunity_link)
    _csv(report_dir / "06_model_metrics.csv", model_metrics)
    _csv(report_dir / "07_prediction_decile_curve.csv", decile_curves)
    _csv(report_dir / "08_stable_candidates.csv", candidates)
    _csv(report_dir / "09_training_year_attribution.csv", attribution)
    _csv(report_dir / "10_trade_feature_increment.csv", trade_increment)
    _csv(report_dir / "11_feature_importance.csv", feature_importance)
    _csv(report_dir / "12_state_samples.csv", state_samples)
    _csv(report_dir / "13_prediction_samples.csv", prediction_samples)
    _csv(report_dir / "14_model_failures.csv", failures)
    _csv(report_dir / "15_causal_audit.csv", causal_audit)
    _csv(report_dir / "16_strategic_threshold_audit.csv", strategic_threshold_audit)
    _csv(report_dir / "17_mechanical_baseline_metrics.csv", mechanical_baseline_metrics)
    _csv(report_dir / "18_mechanical_increment_audit.csv", mechanical_increment_audit)
    _csv(report_dir / "19_transition_alert_metrics.csv", transition_alert_metrics)
    _csv(report_dir / "20_transition_alert_episodes.csv", transition_alert_episodes)
    _json(report_dir / "98_champion.json", champion or {})
    (report_dir / "99_decision.md").write_text(
        decision_markdown(decision=decision, reason=reason, champion=champion, config=config),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.3.3.1",
            edge_id="market_state_continuity",
            stage="research",
            title="ETH AI R03.3.3.1 market-state continuity calibration and audit",
            decision_focus="whether calibrated states add information beyond mechanical persistence and produce usable independent transition warnings",
            print_log=True,
        )
    )
