#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writers for R03 swing research."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import SwingBaselineConfig


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _format_percent(value: object) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def build_decision_markdown(
    *,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    holdout: dict[str, object] | None,
    config: SwingBaselineConfig,
) -> str:
    lines = [
        "# R03 中线 Swing 监督学习研究结论",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 研究定义",
        "",
        "- 日线、4H、1H用于方向与趋势背景。",
        "- 30m、15m、5m、1m用于回踩、结构和订单流入场质量。",
        "- 训练目标不是固定时间收益，而是未来3%–5%潜在空间且MAE受限。",
        "- 退出不是固定持仓时间：结构止损、趋势失效、反向模型证据、盈亏平衡保护与MFE回吐追踪共同决定退出。",
        f"- `{config.max_hold_hours}h` 仅为安全上限，不是主退出条件。",
        f"- 基础完整成本：{config.base_round_trip_cost:.2%}；压力测试包含2x/3x成本和最多5分钟延迟。",
        "",
    ]
    if champion:
        lines.extend(
            [
                "## 2025验证期冠军",
                "",
                f"- 架构：`{champion.get('architecture')}`",
                f"- 目标：`{champion.get('target_id')}`",
                f"- 信号分位数：`{champion.get('quantile')}`",
                f"- 交易数：{champion.get('trades')}",
                f"- 总收益：{_format_percent(champion.get('total_return'))}",
                f"- 单笔净期望：{_format_percent(champion.get('mean_net_return'))}",
                f"- PF：{float(champion.get('profit_factor', 0.0)):.3f}",
                f"- 最大回撤：{_format_percent(champion.get('max_drawdown'))}",
                f"- 2x成本收益：{_format_percent(champion.get('return_2x'))}",
                f"- 5分钟延迟收益：{_format_percent(champion.get('return_delay5'))}",
                f"- 去掉前5大盈利：{_format_percent(champion.get('top5_removed_total_return'))}",
                "",
            ]
        )
    if holdout:
        lines.extend(
            [
                "## 2026锁定样本外",
                "",
                "> 2026H1 已在早期 R01 报告中被整体观察过，因此它不再是项目级“从未看过”的纯封存集；R03 仍禁止用它选择模型、目标或分位数，只作为锁定样本外复核。",
                "",
                f"- 交易数：{holdout.get('trades')}",
                f"- 总收益：{_format_percent(holdout.get('total_return'))}",
                f"- 单笔净期望：{_format_percent(holdout.get('mean_net_return'))}",
                f"- PF：{float(holdout.get('profit_factor', 0.0)):.3f}",
                f"- 最大回撤：{_format_percent(holdout.get('max_drawdown'))}",
                f"- 2x成本收益：{_format_percent(holdout.get('return_2x'))}",
                f"- 5分钟延迟收益：{_format_percent(holdout.get('return_delay5'))}",
                "",
            ]
        )
    lines.extend(
        [
            "## 后续动作",
            "",
            "- `PASS_SWING_EDGE`：进入R04日内趋势，同时保留Swing为第一主Sleeve。",
            "- `FAIL_VALIDATION`：不查看2026来救模型，先诊断高周期方向是否有效、低周期入场是否真正降低MAE。",
            "- `FAIL_LOCKED_HOLDOUT`：承认跨期失效，停止直接迁移实盘；只能从市场状态或数据增量做明确的新假设。",
            "- 所有交易明细、退出原因、MFE/MAE和压力矩阵以CSV为准。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(
    report_dir: Path,
    *,
    run_manifest: dict[str, object],
    preflight: dict[str, object],
    cache_manifest: dict[str, object],
    label_balance: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
    scenario_summaries: pd.DataFrame,
    trades: pd.DataFrame,
    feature_importance: pd.DataFrame,
    champion: dict[str, object] | None,
    holdout: dict[str, object] | None,
    decision: str,
    reason: str,
    config: SwingBaselineConfig,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": report_dir / "00_run_manifest.json",
        "preflight": report_dir / "01_preflight.json",
        "cache": report_dir / "02_cache_manifest.json",
        "labels": report_dir / "03_label_balance.csv",
        "predictions": report_dir / "04_prediction_metrics.csv",
        "scenarios": report_dir / "05_trade_stress_matrix.csv",
        "trades": report_dir / "06_trades.csv",
        "importance": report_dir / "07_feature_importance.csv",
        "champion": report_dir / "08_champion.json",
        "decision": report_dir / "99_decision.md",
    }
    write_json(paths["manifest"], run_manifest)
    write_json(paths["preflight"], preflight)
    write_json(paths["cache"], cache_manifest)
    write_csv(paths["labels"], label_balance)
    write_csv(paths["predictions"], prediction_metrics)
    write_csv(paths["scenarios"], scenario_summaries)
    write_csv(paths["trades"], trades)
    write_csv(paths["importance"], feature_importance)
    write_json(paths["champion"], {"validation_champion": champion, "locked_holdout": holdout})
    paths["decision"].write_text(
        build_decision_markdown(
            decision=decision,
            reason=reason,
            champion=champion,
            holdout=holdout,
            config=config,
        ),
        encoding="utf-8",
    )
    return paths
