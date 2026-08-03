#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reports for R03.1 exact-path swing entry research."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import SwingEntryMvpConfig


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _pct(value: object) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def decision_markdown(
    *,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    holdout: dict[str, object] | None,
    config: SwingEntryMvpConfig,
) -> str:
    lines = [
        "# R03.1 3%–5% Swing 开仓 MVP 结论",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 本轮实际验证什么",
        "",
        "- 不要求至少持仓十几个小时；目标触发可能发生在几十分钟、数小时或数天。",
        "- 模型标签改成真实1分钟路径上的“目标先于风险线”，不再只比较未来最高价和最低价。",
        "- 同一分钟同时触发止盈和风险线时，按风险线先触发处理。",
        "- 多单和空单分开验证，允许只保留一个方向作为实盘MVP。",
        "- 出场只包含目标、初始风险、因果利润保护和最长研究窗口；不再使用15分钟趋势失效或模型反转强平。",
        f"- 基础完整交易成本：{config.base.base_round_trip_cost:.2%}。",
        "",
    ]
    if champion:
        lines.extend(
            [
                "## 2024支持 + 2025验证冠军",
                "",
                f"- 架构：`{champion.get('architecture')}`",
                f"- 方向：`{champion.get('direction')}`",
                f"- 目标：`{champion.get('target_id')}`",
                f"- 出场规则：`{champion.get('exit_policy')}`",
                f"- 分位数：`{champion.get('quantile')}`",
                f"- 2025交易数：{champion.get('trades')}",
                f"- 2025目标命中率：{_pct(champion.get('target_hit_rate'))}",
                f"- 2025总收益：{_pct(champion.get('total_return'))}",
                f"- 2025单笔净期望：{_pct(champion.get('mean_net_return'))}",
                f"- 2025 PF：{float(champion.get('profit_factor', 0.0)):.3f}",
                f"- 2025最大回撤：{_pct(champion.get('max_drawdown'))}",
                f"- 2025中位持仓：{float(champion.get('median_hold_hours', 0.0)):.2f}小时",
                f"- 2025 2x成本收益：{_pct(champion.get('return_2x'))}",
                f"- 2025最大延迟收益：{_pct(champion.get('return_delay_max'))}",
                f"- 2024支持期收益：{_pct(champion.get('dev_total_return'))}",
                "",
            ]
        )
    if holdout:
        lines.extend(
            [
                "## 2026锁定样本外",
                "",
                f"- 交易数：{holdout.get('trades')}",
                f"- 目标命中率：{_pct(holdout.get('target_hit_rate'))}",
                f"- 总收益：{_pct(holdout.get('total_return'))}",
                f"- 单笔净期望：{_pct(holdout.get('mean_net_return'))}",
                f"- PF：{float(holdout.get('profit_factor', 0.0)):.3f}",
                f"- 最大回撤：{_pct(holdout.get('max_drawdown'))}",
                f"- 2x成本收益：{_pct(holdout.get('return_2x'))}",
                f"- 最大延迟收益：{_pct(holdout.get('return_delay_max'))}",
                "",
            ]
        )
    lines.extend(
        [
            "## 下一步",
            "",
            "- `PASS_SWING_ENTRY_MVP`：导出模型合同，进入AetherEdge影子推理；持仓管理模型仍不在本轮范围内。",
            "- `FAIL_VALIDATION`：说明当前多周期特征无法稳定识别3%–5%开仓机会，不再通过改退出规则救模型。",
            "- `FAIL_LOCKED_HOLDOUT`：说明跨期失效，不能实盘；后续只能提出新的市场状态或数据增量假设。",
            "- 交易明细中的 `exit_reason`、`target_hit`、MFE、MAE和持仓时长用于判断到底是开仓失败还是利润保护过早。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_reports(
    report_dir: Path,
    *,
    run_manifest: dict[str, object],
    preflight: dict[str, object],
    base_cache_manifest: dict[str, object],
    exact_label_diagnostics: pd.DataFrame,
    prediction_metrics: pd.DataFrame,
    scenario_summaries: pd.DataFrame,
    trades: pd.DataFrame,
    feature_importance: pd.DataFrame,
    champion: dict[str, object] | None,
    holdout: dict[str, object] | None,
    decision: str,
    reason: str,
    config: SwingEntryMvpConfig,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": report_dir / "00_run_manifest.json",
        "preflight": report_dir / "01_preflight.json",
        "base_cache": report_dir / "02_base_cache_manifest.json",
        "labels": report_dir / "03_exact_label_diagnostics.csv",
        "predictions": report_dir / "04_prediction_metrics.csv",
        "scenarios": report_dir / "05_trade_stress_matrix.csv",
        "trades": report_dir / "06_trades.csv",
        "importance": report_dir / "07_feature_importance.csv",
        "champion": report_dir / "08_champion.json",
        "mvp_contract": report_dir / "09_mvp_contract.json",
        "decision": report_dir / "99_decision.md",
    }
    _json(paths["manifest"], run_manifest)
    _json(paths["preflight"], preflight)
    _json(paths["base_cache"], base_cache_manifest)
    _csv(paths["labels"], exact_label_diagnostics)
    _csv(paths["predictions"], prediction_metrics)
    _csv(paths["scenarios"], scenario_summaries)
    _csv(paths["trades"], trades)
    _csv(paths["importance"], feature_importance)
    _json(paths["champion"], {"validation_champion": champion, "locked_holdout": holdout})
    contract = None
    if decision == "PASS_SWING_ENTRY_MVP" and champion:
        contract = {
            "stage": "R03.1",
            "status": "RESEARCH_CANDIDATE_NOT_LIVE_MODEL_BINARY",
            "architecture": champion.get("architecture"),
            "direction": champion.get("direction"),
            "target_id": champion.get("target_id"),
            "exit_policy": champion.get("exit_policy"),
            "score_quantile": champion.get("quantile"),
            "score_threshold_2025": champion.get("score_threshold"),
            "decision_interval_minutes": config.base.decision_interval_minutes,
            "base_execution_delay_minutes": config.base.execution_delay_minutes,
            "hard_contracts": [
                "Use only completed causal timeframe features.",
                "Never exceed deterministic account risk limits.",
                "No model binary is exported until offline/live feature parity is proven.",
            ],
        }
    _json(paths["mvp_contract"], contract)
    paths["decision"].write_text(
        decision_markdown(
            decision=decision,
            reason=reason,
            champion=champion,
            holdout=holdout,
            config=config,
        ),
        encoding="utf-8",
    )
    return paths
