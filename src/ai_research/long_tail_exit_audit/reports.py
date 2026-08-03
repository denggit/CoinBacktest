#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import LongTailExitAuditConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    config: LongTailExitAuditConfig,
) -> str:
    lines = [
        "# R03.4.2 q90基础多头高分尾部事件策略化审核",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 核心原则",
        "",
        "- 正期望优先于胜率、交易次数、利润厚度或任一单项指标。",
        "- 冻结R03.4.1基础多头LightGBM及q90校准方式，不再调模型。",
        "- 状态模型已从交易决策链正式舍弃；本研究不加载状态缓存、不使用状态字段。",
        "- q95只作为质量控制，不替代q90主线。",
        "- 所有入场使用信号后的1/3/5分钟开盘；2024和2025分别纯OOS，2026继续封存。",
        "",
        "## 退出族",
        "",
        "- 固定6小时退出只作为原始诊断基准。",
        "- 结构止损：入场前60或180分钟低点，加固定bps/ATR缓冲；结构过远则放弃交易。",
        "- 固定R目标：1.5R或2R。",
        "- 利润保护：达到1R或1.5R后，以0.5R或0.75R回吐追踪；新追踪价从下一分钟起生效。",
        "- 滚动续期：每6小时用最新基础模型分数重新验证；未达到校准期q60/q70则退出。",
        "- 模型失效：至少持仓1小时后，连续两次低于校准期q50才退出。",
        "- 24/36/48小时只是安全上限；候选若主要依赖安全时间退出则不能通过。",
        "",
        "## 正期望通过门槛",
        "",
        "- q90、1分钟延迟下，2024和2025在1倍与2倍成本后均为正期望。",
        f"- 两年2倍成本PF均不低于 {config.minimum_2x_profit_factor:.2f}。",
        f"- 每年交易数至少 {config.minimum_trades_per_year}。",
        f"- 8个季度至少 {config.minimum_positive_quarters} 个正收益季度。",
        "- 去掉每年前10大盈利交易后仍为正期望。",
        "- 3分钟延迟后两年仍为正期望。",
        f"- 风险定仓权益曲线MDD不超过 {config.maximum_risk_sized_drawdown:.0%}。",
        f"- 安全时间上限退出占比不超过 {config.maximum_safety_cap_share:.0%}。",
        "",
    ]
    if champion:
        lines.extend(
            [
                "## 最佳候选",
                "",
                f"- 退出方案：`{champion.get('recipe')}`",
                f"- 最弱年份2倍成本净期望：{float(champion.get('minimum_2x_expectancy', float('nan'))):.4%}",
                f"- 最弱年份2倍成本PF：{float(champion.get('minimum_2x_profit_factor', float('nan'))):.3f}",
                f"- 正收益季度：{int(champion.get('positive_quarters', 0))}/8",
                "",
            ]
        )
    lines.extend(
        [
            "## 解释限制",
            "",
            "- 通过表示退出机制把冻结q90候选转成跨年、成本后仍为正期望的路径策略候选；仍需再做最终执行与组合回测。",
            "- 胜率下降不自动淘汰，只要净期望、PF、回撤和成本韧性改善。",
            "- 交易次数增加不自动加分；若新增交易稀释净期望则应拒绝。",
            "- 若所有路径退出都不如固定6小时诊断，应保留开仓Edge结论，但停止把它策略化为当前退出结构。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    report_dir: Path,
    *,
    manifest: dict[str, object],
    preflight: dict[str, object],
    exit_contract: dict[str, object],
    signal_audit: pd.DataFrame,
    threshold_audit: pd.DataFrame,
    trade_summary: pd.DataFrame,
    period_summary: pd.DataFrame,
    exit_reason_summary: pd.DataFrame,
    duration_summary: pd.DataFrame,
    stress_summary: pd.DataFrame,
    concentration: pd.DataFrame,
    stable: pd.DataFrame,
    trades: pd.DataFrame,
    causal_audit: pd.DataFrame,
    failures: pd.DataFrame,
    decision: str,
    reason: str,
    champion: dict[str, object] | None,
    config: LongTailExitAuditConfig,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _json(report_dir / "00_run_manifest.json", manifest)
    _json(report_dir / "01_preflight.json", preflight)
    _json(report_dir / "02_exit_contract.json", exit_contract)
    _csv(report_dir / "03_signal_event_audit.csv", signal_audit)
    _csv(report_dir / "04_score_threshold_audit.csv", threshold_audit)
    _csv(report_dir / "05_trade_summary.csv", trade_summary)
    _csv(report_dir / "06_period_summary.csv", period_summary)
    _csv(report_dir / "07_exit_reason_summary.csv", exit_reason_summary)
    _csv(report_dir / "08_duration_summary.csv", duration_summary)
    _csv(report_dir / "09_cost_delay_stress.csv", stress_summary)
    _csv(report_dir / "10_top10_concentration.csv", concentration)
    _csv(report_dir / "11_stable_candidates.csv", stable)
    _csv(report_dir / "12_trade_details.csv", trades)
    _csv(report_dir / "13_causal_audit.csv", causal_audit)
    _csv(report_dir / "14_model_failures.csv", failures)
    _json(report_dir / "98_champion.json", champion or {})
    (report_dir / "99_decision.md").write_text(
        decision_markdown(decision, reason, champion, config), encoding="utf-8"
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.2",
            edge_id="frozen_q90_long_tail_path_exit",
            stage="research",
            title="ETH AI R03.4.2 frozen q90 long-tail path exit audit",
            decision_focus="whether structural stops, profit protection and rolling model renewal preserve positive expectancy better than fixed six-hour exit",
            print_log=True,
        )
    )
