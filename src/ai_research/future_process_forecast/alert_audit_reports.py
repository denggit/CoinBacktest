#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Durable reports for R03.3.1 actionable alert-value audit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .alert_audit_config import ProcessAlertAuditConfig


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def decision_markdown(
    *,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    candidates: pd.DataFrame,
    config: ProcessAlertAuditConfig,
) -> str:
    lines = [
        "# R03.3.1 独立预警与剩余交易空间审核",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 本轮回答的问题",
        "",
        "- 连续15分钟高分不再逐点计数，而是合并成一个独立预警段。",
        "- 每个预警段只使用第一次高分，避免行情走了很久后重复报高分美化命中率。",
        f"- 允许启动后{config.early_start_grace_hours:g}小时内、且过程完成不超过{config.max_actionable_progress:.0%}的早期确认。",
        "- 预警即使时间够早，若剩余波动空间不足，也不算可交易命中。",
        "- 报告同时给出独立预警成功率和事件级覆盖率，不再只看每15分钟点位精度。",
        "- 2026H1继续封存。",
        "",
        "## 可交易剩余空间门槛",
        "",
        f"- 上涨/下跌扩张：第一次预警后至少还剩{config.min_remaining_directional_move:.2%}到正式目标。",
        f"- 高波动双向震荡：第一次预警后至少还剩{config.min_remaining_range_move:.2%}可实现双向区间。",
        "",
    ]
    if champion:
        lines.extend(
            [
                "## 最佳跨期候选",
                "",
                f"- 架构：`{champion.get('architecture')}`",
                f"- 过程：`{champion.get('process')}`",
                f"- 窗口：{champion.get('horizon_hours')}小时",
                f"- 分位：{float(champion.get('quantile', float('nan'))):.1%}",
                f"- 2024独立预警可交易命中率：{float(champion.get('WF_2024_actionable_alert_precision', float('nan'))):.2%}",
                f"- 2025独立预警可交易命中率：{float(champion.get('WF_2025_actionable_alert_precision', float('nan'))):.2%}",
                f"- 2024事件覆盖率：{float(champion.get('WF_2024_event_coverage', float('nan'))):.2%}",
                f"- 2025事件覆盖率：{float(champion.get('WF_2025_event_coverage', float('nan'))):.2%}",
                f"- 2024首次预警剩余空间：{float(champion.get('WF_2024_median_first_alert_remaining_opportunity', float('nan'))):.2%}",
                f"- 2025首次预警剩余空间：{float(champion.get('WF_2025_median_first_alert_remaining_opportunity', float('nan'))):.2%}",
                "",
            ]
        )
    if not candidates.empty:
        passed = int(candidates["passes_actionability"].sum()) if "passes_actionability" in candidates else 0
        lines.extend(["## 跨期审核", "", f"- 同配置同时通过2024与2025可交易审核：{passed}个。", ""])
    lines.extend(
        [
            "## 下一步纪律",
            "",
            "- 通过：把该概率作为策略环境开关，下一步研究高波动环境内部的趋势/双向流动性执行。",
            "- 未通过且首次预警剩余空间不足：停止状态预测路线，说明模型主要是晚确认。",
            "- 未通过但剩余空间充足：问题在误报或覆盖率，先改善标签/概率稳定性，不直接做入场回测。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_alert_audit_reports(
    report_dir: Path,
    *,
    manifest: dict[str, object],
    preflight: dict[str, object],
    audit_definition: dict[str, object],
    episode_metrics: pd.DataFrame,
    event_metrics: pd.DataFrame,
    first_alerts: pd.DataFrame,
    event_coverage: pd.DataFrame,
    candidates: pd.DataFrame,
    failures: pd.DataFrame,
    causal_audit: pd.DataFrame,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    config: ProcessAlertAuditConfig,
) -> dict[str, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": report_dir / "00_run_manifest.json",
        "preflight": report_dir / "01_preflight.json",
        "definition": report_dir / "02_alert_value_definition.json",
        "episode_metrics": report_dir / "03_independent_alert_episode_metrics.csv",
        "event_metrics": report_dir / "04_event_level_coverage_metrics.csv",
        "first_alerts": report_dir / "05_first_alert_episodes.csv",
        "coverage": report_dir / "06_event_first_alert_coverage.csv",
        "candidates": report_dir / "07_actionability_candidates.csv",
        "failures": report_dir / "08_model_failures.csv",
        "audit": report_dir / "09_causal_audit.csv",
        "champion": report_dir / "98_champion.json",
        "decision": report_dir / "99_decision.md",
    }
    _json(paths["manifest"], manifest)
    _json(paths["preflight"], preflight)
    _json(paths["definition"], audit_definition)
    for key, frame in (
        ("episode_metrics", episode_metrics),
        ("event_metrics", event_metrics),
        ("first_alerts", first_alerts),
        ("coverage", event_coverage),
        ("candidates", candidates),
        ("failures", failures),
        ("audit", causal_audit),
    ):
        _csv(paths[key], frame)
    _json(paths["champion"], {"champion": champion, "decision": decision, "reason": reason})
    paths["decision"].write_text(
        decision_markdown(decision=decision, reason=reason, champion=champion, candidates=candidates, config=config),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="ETH-AI-R03.3.1",
            edge_id="future-process-actionable-alert-audit",
            stage="research",
            title="R03.3.1 Actionable Process Alert Audit",
            decision_focus="first_alert_timing_remaining_opportunity_and_event_coverage",
            print_log=True,
        )
    )
    return paths
