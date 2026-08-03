#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03.2 reports focused on long causal context and continuous market process."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.swing_entry_mvp.config import SwingEntryMvpConfig


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _pct(value: object) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return "n/a"


def _feature_summary(manifest: dict[str, object]) -> dict[str, object]:
    files = manifest.get("files", [])
    return {
        "feature_profile": manifest.get("feature_profile"),
        "cache_signature": manifest.get("cache_signature"),
        "feature_lookback_days": (manifest.get("config") or {}).get("feature_lookback_days")
        if isinstance(manifest.get("config"), dict)
        else None,
        "year_shards": len(files) if isinstance(files, list) else None,
    }


def decision_markdown(
    *,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    holdout: dict[str, object] | None,
    config: SwingEntryMvpConfig,
    feature_importance: pd.DataFrame,
) -> str:
    lines = [
        "# R03.2 长上下文 3%–5% Swing 开仓研究结论",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 本轮只改变什么",
        "",
        "- 标签、真实路径回放、费用、退出规则和LightGBM参数均沿用R03.1。",
        "- 日线上下文扩展到365天，4H扩展到约120天，1H扩展到约30天。",
        "- 新增长期高低点位置、趋势年龄、结构抬升/降低、推动与回调、波动率生命周期、订单流持续性。",
        "- 新增日线/4H/1H之间的方向一致性和战术回调关系。",
        "- 仍是因果滚动特征模型，不是单根K线，也不是TCN/Transformer序列模型。",
        "- 多空独立训练和验证；允许最终只保留一个方向。",
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
                f"- 出场：`{champion.get('exit_policy')}`",
                f"- 信号分位：`{champion.get('quantile')}`",
                f"- 2025交易数：{champion.get('trades')}",
                f"- 2025目标命中率：{_pct(champion.get('target_hit_rate'))}",
                f"- 2025总收益：{_pct(champion.get('total_return'))}",
                f"- 2025净期望：{_pct(champion.get('mean_net_return'))}",
                f"- 2025 PF：{float(champion.get('profit_factor', 0.0)):.3f}",
                f"- 2025最大回撤：{_pct(champion.get('max_drawdown'))}",
                f"- 2025 2x成本：{_pct(champion.get('return_2x'))}",
                f"- 2024支持期：{_pct(champion.get('dev_total_return'))}",
                "",
            ]
        )
    if holdout:
        lines.extend(
            [
                "## 2026锁定样本外",
                "",
                f"- 交易数：{holdout.get('trades')}",
                f"- 总收益：{_pct(holdout.get('total_return'))}",
                f"- PF：{float(holdout.get('profit_factor', 0.0)):.3f}",
                f"- 最大回撤：{_pct(holdout.get('max_drawdown'))}",
                f"- 2x成本：{_pct(holdout.get('return_2x'))}",
                "",
            ]
        )
    if not feature_importance.empty and "feature" in feature_importance.columns:
        ranking = (
            feature_importance.groupby("feature", as_index=False)["importance"]
            .mean()
            .sort_values("importance", ascending=False, kind="stable")
            .head(20)
        )
        lines.extend(["## 平均特征重要性前20", ""])
        for row in ranking.itertuples(index=False):
            lines.append(f"- `{row.feature}`：{float(row.importance):.6g}")
        lines.append("")
    lines.extend(
        [
            "## 下一步纪律",
            "",
            "- 通过：才进入模型导出、特征在线一致性和AetherEdge影子推理。",
            "- 未通过：不靠继续调退出或扩大参数网格救模型；先判断长期上下文是否改善了跨期稳定性。",
            "- 本轮不开发持仓管理模型、强化学习、短线Sleeve或执行优化。",
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
        "feature_summary": report_dir / "02_1_long_context_summary.json",
        "labels": report_dir / "03_exact_label_diagnostics.csv",
        "predictions": report_dir / "04_prediction_metrics.csv",
        "scenarios": report_dir / "05_trade_stress_matrix.csv",
        "trades": report_dir / "06_trades.csv",
        "importance": report_dir / "07_feature_importance.csv",
        "champion": report_dir / "08_champion.json",
        "decision": report_dir / "99_decision.md",
    }
    _json(paths["manifest"], run_manifest)
    _json(paths["preflight"], preflight)
    _json(paths["base_cache"], base_cache_manifest)
    _json(paths["feature_summary"], _feature_summary(base_cache_manifest))
    _csv(paths["labels"], exact_label_diagnostics)
    _csv(paths["predictions"], prediction_metrics)
    _csv(paths["scenarios"], scenario_summaries)
    _csv(paths["trades"], trades)
    _csv(paths["importance"], feature_importance)
    _json(paths["champion"], {"validation_champion": champion, "locked_holdout": holdout})
    paths["decision"].write_text(
        decision_markdown(
            decision=decision,
            reason=reason,
            champion=champion,
            holdout=holdout,
            config=config,
            feature_importance=feature_importance,
        ),
        encoding="utf-8",
    )
    return paths
