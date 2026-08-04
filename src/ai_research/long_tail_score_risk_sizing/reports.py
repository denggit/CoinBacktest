#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.13."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import ScoreRiskConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(decision: str, reason: str, *, attribution: pd.DataFrame, order_audit: pd.DataFrame, summary: pd.DataFrame, gate: pd.DataFrame) -> str:
    lines = [
        "# R03.4.2.13 分数层风险定仓与账户缩放", "", f"## 决策：`{decision}`", "", reason, "",
        "## 冻结主线", "",
        "- q70 ML开仓、下一根1m open不变。",
        "- C2真实2%硬止损 + 1.5%完成收盘软失败不变。",
        "- 盈利继续由确定性`failed_reclaim`非时间退出。",
        "- 2026继续封存；禁止按年份选择不同风险映射。",
        "- 分数只在开仓时确定风险倍率，持仓期间分数变化不能续命、加仓或退出。",
        "- 所有正式候选单笔最大价格风险仍不超过1R；1.25R只做账户缩放诊断。", "",
    ]
    if not attribution.empty:
        lines.extend(["## C2分数层归因（1分钟延迟、2倍成本）", ""])
        focus = attribution.loc[attribution.delay_minutes.astype(int).eq(1) & attribution.cost_multiplier.astype(float).eq(2.0)]
        for row in focus.sort_values(["fold_id", "score_tier"]).itertuples():
            lines.append(f"- {row.fold_id}/{row.score_tier}: {int(row.events)}笔，均值{row.mean_cycle_return:+.3%}，胜率{row.win_rate:.1%}，PF={row.profit_factor:.2f}。")
        lines.append("")
    if not order_audit.empty:
        lines.extend(["## 跨年分数顺序审计", ""])
        for row in order_audit.itertuples():
            lines.append(f"- {row.fold_id}: 单调={bool(row.monotonic_score_order)}，排序={row.return_ranking}。")
        lines.append("")
    if not summary.empty:
        lines.extend(["## 账户结果（1分钟延迟、2倍成本）", ""])
        focus = summary.loc[summary.delay_minutes.astype(int).eq(1) & summary.cost_multiplier.astype(float).eq(2.0)]
        for row in focus.sort_values(["policy", "fold_id"]).itertuples():
            lines.append(f"- {row.policy}/{row.fold_id}: 收益{row.total_net_return:.1%}，MDD={row.max_drawdown:.1%}，PF={row.profit_factor:.2f}，平均风险{row.mean_risk_multiplier:.2f}R，平均名义仓位{row.mean_base_notional_to_equity:.2f}倍。")
        lines.append("")
    if not gate.empty:
        lines.extend(["## 资格门", ""])
        for row in gate.itertuples():
            lines.append(f"- {row.policy}: 年度最低保留{row.minimum_return_retention:.1%}，最大MDD比{row.maximum_mdd_ratio:.2f}，最低Calmar比{row.minimum_calmar_ratio:.2f}，压力门={bool(row.stress_gate_pass)}，通过={bool(row.pass_to_next_stage)}。")
        lines.append("")
    lines.extend([
        "## 解释原则", "",
        "- 如果分数层收益顺序跨年漂移，不能因为2025的q90很强就追溯性放大q90风险。",
        "- 低分层仍有正Edge时，简单降权会降低绝对收益；只有收益基本保留且回撤/Calmar显著改善才值得采用。",
        "- 等风险若胜出，结论是分数负责准入和质量描述，不负责仓位倍率。",
        "- 1.25R等风险只展示进取账户缩放的收益/MDD，不是本阶段正式通过候选。", "",
    ])
    return "\n".join(lines)


def write_reports(*, config: ScoreRiskConfig, manifest: dict[str, object], preflight: dict[str, object], source_summary: pd.DataFrame, attribution: pd.DataFrame, order_audit: pd.DataFrame, cycles: pd.DataFrame, legs: pd.DataFrame, daily: pd.DataFrame, summary: pd.DataFrame, gate: pd.DataFrame, causal: pd.DataFrame, rejections: pd.DataFrame, failures: pd.DataFrame, decision: str, reason: str) -> None:
    root = config.report_path
    root.mkdir(parents=True, exist_ok=True)
    _json(root / "00_run_manifest.json", manifest)
    _json(root / "01_preflight.json", preflight)
    _csv(root / "02_source_c2_summary.csv", source_summary)
    _csv(root / "03_score_tier_attribution.csv", attribution)
    _csv(root / "04_cross_year_score_order.csv", order_audit)
    _csv(root / "05_account_cycles.csv", cycles)
    _csv(root / "06_account_legs.csv", legs)
    _csv(root / "07_daily_equity.csv", daily)
    _csv(root / "08_policy_summary.csv", summary)
    _csv(root / "09_policy_gate.csv", gate)
    _csv(root / "10_causal_audit.csv", causal)
    _csv(root / "11_runtime_rejections.csv", rejections)
    _csv(root / "12_failures.csv", failures)
    (root / "99_decision.md").write_text(decision_markdown(decision, reason, attribution=attribution, order_audit=order_audit, summary=summary, gate=gate), encoding="utf-8")
    write_gpt_review_pack(ReviewPackConfig(
        report_dir=root, experiment_id="R03.4.2.13", edge_id="q70_c2_score_risk_sizing", stage="research",
        title="ETH AI R03.4.2.13 score-tier risk sizing and account scaling",
        decision_focus="whether frozen q70/q80/q90 score tiers justify one same cross-year risk map on top of the passed C2 real-one-R stop, without reducing trade coverage, tuning exits, opening 2026 or confusing global risk leverage with score Edge",
        print_log=True,
    ))
