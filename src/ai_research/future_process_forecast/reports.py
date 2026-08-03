#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Durable R03.3 reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import FutureProcessForecastConfig


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def decision_markdown(
    *,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    event_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    config: FutureProcessForecastConfig,
) -> str:
    lines = [
        "# R03.3 未来市场过程预测结论",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 这次和旧市场状态地图的区别",
        "",
        "- 旧地图主要识别已经出现的趋势、突破或高波动；R03.3预测未来6/12/24小时内是否出现新的过程启动点。",
        "- 上涨扩张、下跌扩张和高波动震荡都先被切成独立事件；一个持续行情不会贡献几千个伪独立事件。",
        "- 正样本必须位于启动点之前；启动后信号单独计入ongoing和tail-car，不会算作预测成功。",
        "- 报告核心不是交易PF，而是跨年AUC/AP、Top分位Lift、真实领先小时和尾班车率。",
        "- 2026-01-01至2026-06-30保持封存，本轮不会训练、选型或展示其结果。",
        "",
        "## 数据与模型",
        "",
        "- 长周期：1D最长365天、4H约120天、1H约30天。",
        "- 入场上下文：30m、15m、5m、1m连续特征。",
        f"- 微观：公共 `{config.micro_timeframe}` Trade Bar，只做缓存读取；不访问或重建Raw Trades。",
        "- 模型：Macro LightGBM、全多周期LightGBM、全多周期+微观LightGBM、标准化Logistic基准。",
        "",
    ]
    if not event_summary.empty:
        lines.extend(["## 独立事件数量", ""])
        for row in event_summary.itertuples(index=False):
            lines.append(f"- {row.process} / {int(row.start_year)}：{int(row.events)}个")
        lines.append("")
    if champion:
        lines.extend(
            [
                "## 最佳跨期候选",
                "",
                f"- 架构：`{champion.get('architecture')}`",
                f"- 过程：`{champion.get('process')}`",
                f"- 预测窗口：未来{champion.get('horizon_hours')}小时",
                f"- 2024 AUC：{float(champion.get('WF_2024_roc_auc', float('nan'))):.4f}",
                f"- 2025 AUC：{float(champion.get('WF_2025_roc_auc', float('nan'))):.4f}",
                f"- 2024 Top5% Lift：{float(champion.get('WF_2024_lift', float('nan'))):.3f}",
                f"- 2025 Top5% Lift：{float(champion.get('WF_2025_lift', float('nan'))):.3f}",
                f"- 2024尾班车率：{float(champion.get('WF_2024_tail_car_rate_progress30', float('nan'))):.2%}",
                f"- 2025尾班车率：{float(champion.get('WF_2025_tail_car_rate_progress30', float('nan'))):.2%}",
                "",
            ]
        )
    if not candidates.empty:
        passed = int(candidates["passes"].sum()) if "passes" in candidates else 0
        lines.extend(["## 稳定性审核", "", f"- 同配置同时通过2024和2025的候选：{passed}个。", ""])
    lines.extend(
        [
            "## 下一步纪律",
            "",
            "- 通过：只进入R03.4低MAE入场模型；R03.3本身不直接交易。",
            "- 未通过：先检查事件定义与可预测性，不通过增加参数网格或提前打开2026来救结果。",
            "- 高波动震荡若可预测，应交给短线流动性/反转Sleeve；单边扩张交给Swing方向与入场Sleeve。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(
    report_dir: Path,
    *,
    manifest: dict[str, object],
    preflight: dict[str, object],
    event_definition: dict[str, object],
    events: pd.DataFrame,
    event_summary: pd.DataFrame,
    event_path_summary: pd.DataFrame,
    label_rates: pd.DataFrame,
    probability_metrics: pd.DataFrame,
    quantile_metrics: pd.DataFrame,
    candidates: pd.DataFrame,
    micro_increment: pd.DataFrame,
    feature_importance: pd.DataFrame,
    pre_event_uplift: pd.DataFrame,
    signal_samples: pd.DataFrame,
    failures: pd.DataFrame,
    causal_audit: pd.DataFrame,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    config: FutureProcessForecastConfig,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": report_dir / "00_run_manifest.json",
        "preflight": report_dir / "01_preflight.json",
        "event_definition": report_dir / "02_event_definition.json",
        "events": report_dir / "03_event_atlas.csv",
        "event_summary": report_dir / "04_event_yearly_summary.csv",
        "event_paths": report_dir / "05_event_path_summary.csv",
        "label_rates": report_dir / "06_forecast_label_rates.csv",
        "probability": report_dir / "07_probability_metrics.csv",
        "quantiles": report_dir / "08_top_quantile_forecast_metrics.csv",
        "candidates": report_dir / "09_stable_candidates.csv",
        "micro": report_dir / "10_micro_increment.csv",
        "importance": report_dir / "11_feature_importance.csv",
        "uplift": report_dir / "12_pre_event_feature_uplift.csv",
        "samples": report_dir / "13_signal_samples.csv",
        "failures": report_dir / "14_model_failures.csv",
        "audit": report_dir / "15_causal_tail_car_audit.csv",
        "champion": report_dir / "98_champion.json",
        "decision": report_dir / "99_decision.md",
    }
    _json(paths["manifest"], manifest)
    _json(paths["preflight"], preflight)
    _json(paths["event_definition"], event_definition)
    for key, frame in (
        ("events", events),
        ("event_summary", event_summary),
        ("event_paths", event_path_summary),
        ("label_rates", label_rates),
        ("probability", probability_metrics),
        ("quantiles", quantile_metrics),
        ("candidates", candidates),
        ("micro", micro_increment),
        ("importance", feature_importance),
        ("uplift", pre_event_uplift),
        ("samples", signal_samples),
        ("failures", failures),
        ("audit", causal_audit),
    ):
        _csv(paths[key], frame)
    _json(paths["champion"], {"champion": champion, "decision": decision, "reason": reason})
    paths["decision"].write_text(
        decision_markdown(
            decision=decision,
            reason=reason,
            champion=champion,
            event_summary=event_summary,
            candidates=candidates,
            config=config,
        ),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="ETH-AI-R03.3",
            edge_id="future-market-process-forecast",
            stage="research",
            title="R03.3 Future Market Process Forecast",
            decision_focus="forecast_validity_and_tail_car_risk",
            print_log=True,
        )
    )
    return paths
