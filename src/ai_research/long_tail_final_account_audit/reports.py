#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.15."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import FinalAccountAuditConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    *,
    historical: pd.DataFrame,
    scenarios: pd.DataFrame,
    lot_sizes: pd.DataFrame,
    risk: pd.DataFrame,
    gate: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.15 冻结策略最终账户与实盘准备审计",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 冻结策略",
        "",
        "- q70信号后下一根1m open立即入场。",
        "- 所有q70统一等风险；不按分数加仓或调整持仓。",
        "- 2%真实交易所硬止损，1.5%完成15m收盘软失败。",
        "- `failed_reclaim`为非时间结构退出；没有固定止盈。",
        "- 2026继续封存。2023属于训练/开发历史，不伪装成OOS账户收益。",
        "",
        "## 历史指标口径",
        "",
    ]
    for row in historical.to_dict("records"):
        lines.append(
            f"- {row['fold_id']}/{row['metric_scope']}: 交易{int(row['trades'])}，"
            f"胜率{float(row['win_rate']):.1%}，PF={float(row['profit_factor']):.2f}，"
            f"收益{float(row['total_return']):.1%}，MDD={float(row['max_drawdown']):.1%}；"
            f"口径：{row.get('return_scope_note', 'source metric scope')}。"
        )
    lines += ["", "## 连续2024-2025 OOS账户", ""]
    ordered_scenarios = scenarios.sort_values(["delay_minutes", "cost_multiplier"]) if not scenarios.empty else scenarios
    for row in ordered_scenarios.to_dict("records"):
        lines.append(
            f"- 延迟{int(row['delay_minutes'])}m/成本{float(row['cost_multiplier']):.0f}x："
            f"交易{int(row['trades'])}笔，收益{float(row['total_return']):.1%}，CAGR={float(row['cagr']):.1%}，"
            f"MDD={float(row['max_drawdown']):.1%}，PF={float(row['profit_factor']):.2f}，"
            f"正收益月{int(row['positive_months'])}/{int(row['months'])}，"
            f"去前10大收益{float(row['total_return_without_top10']):.1%}。"
        )
    anchor = (
        scenarios.loc[
            scenarios["delay_minutes"].astype(int).eq(1)
            & scenarios["cost_multiplier"].astype(float).eq(2.0)
        ]
        if not scenarios.empty
        else pd.DataFrame()
    )
    if not anchor.empty:
        row = anchor.iloc[0]
        lines += [
            "",
            "## 主场景执行特征",
            "",
            f"- 月均交易：{float(row['trades_per_month']):.1f}笔。",
            f"- 持仓中位数：{float(row['median_hold_hours']):.1f}小时；P90={float(row['p90_hold_hours']):.1f}小时。",
            f"- 最长无新开仓：{float(row['max_entry_gap_hours']) / 24.0:.1f}天。",
            f"- 最长连续亏损：{int(row['longest_losing_streak'])}笔。",
            f"- 最长日频回撤期：{int(row['max_drawdown_duration_days'])}天。",
            f"- 历史最差净亏损：{float(row['worst_net_r']):.3f}R。",
        ]
    if not risk.empty:
        focus = risk.loc[
            risk["delay_minutes"].astype(int).eq(1)
            & risk["cost_multiplier"].astype(float).eq(2.0)
        ]
        if not focus.empty:
            row = focus.iloc[0]
            conservative = float(risk["recommended_live_price_risk_budget"].min())
            lines += [
                "",
                "## 实盘风险预算",
                "",
                f"- 若账户净尾部目标为1%，历史主场景允许的价格风险预算约{float(row['maximum_price_risk_budget_for_1pct_net_tail']):.2%}。",
                f"- 覆盖全部3倍成本压力时，保守价格风险预算约{conservative:.2%}。",
                f"- 建议部署初期使用0.83%至0.85%价格风险，保留约0.15%至0.17%给手续费、滑点与跳价。",
            ]
    lines += ["", "## OKX最小张数", ""]
    for row in lot_sizes.to_dict("records"):
        lines.append(
            f"- {row.get('sizing_profile', 'default')}/初始权益{float(row['initial_equity_usdt']):,.0f}U："
            f"目标名义{float(row['target_notional_multiple']):.3f}x，不可交易占比{float(row['untradable_share']):.1%}，"
            f"平均实际价格风险{float(row['mean_actual_price_risk_fraction']):.2%}，"
            f"定仓效率{float(row['mean_sizing_efficiency']):.1%}。"
        )
    lines += ["", "## 资格门", ""]
    for row in gate.to_dict("records"):
        lines.append(f"- {row['check']}: {bool(row['pass'])}")
    lines += [
        "",
        "## 模型训练与发布制度",
        "",
        "- 每月做数据、分数分布、q70频率、成本后表现与校准漂移审计。",
        "- 可以每月训练影子候选，但绝不自动替换冠军模型。",
        "- 正常发布按季度或显著漂移事件触发；必须通过冻结OOS、压力、影子运行与回滚门。",
        "- 实盘持续使用不可变模型版本、特征Schema哈希与训练截止时间，保证可恢复和可审计。",
        "",
        "## 后续",
        "",
    ]
    if decision == "PASS_FINAL_ACCOUNT_LIVE_READINESS":
        lines.append("冻结C2规则与风险预算，下一步只允许一次性开启2026封存验证；在看到2026前不得再调参数。")
    else:
        lines.append("先修复失败的账户、风险或部署门，不得开启2026封存集。")
    return "\n".join(lines) + "\n"


def write_reports(
    *,
    config: FinalAccountAuditConfig,
    manifest: dict[str, object],
    preflight: dict[str, object],
    historical: pd.DataFrame,
    cycles: pd.DataFrame,
    daily: pd.DataFrame,
    scenarios: pd.DataFrame,
    months: pd.DataFrame,
    quarters: pd.DataFrame,
    lot_sizes: pd.DataFrame,
    risk: pd.DataFrame,
    governance: pd.DataFrame,
    live_state: pd.DataFrame,
    gate: pd.DataFrame,
    causal: pd.DataFrame,
    failures: pd.DataFrame,
    decision: str,
    reason: str,
) -> None:
    root = config.report_path
    root.mkdir(parents=True, exist_ok=True)
    _json(root / "00_run_manifest.json", manifest)
    _json(root / "01_preflight.json", preflight)
    outputs = (
        ("02_historical_metric_contract.csv", historical),
        ("03_continuous_account_cycles.csv", cycles),
        ("04_continuous_daily_equity.csv", daily),
        ("05_continuous_scenario_summary.csv", scenarios),
        ("06_monthly_returns.csv", months),
        ("07_quarterly_returns.csv", quarters),
        ("08_okx_lot_size_audit.csv", lot_sizes),
        ("09_net_risk_reserve.csv", risk),
        ("10_model_governance.csv", governance),
        ("11_live_state_contract.csv", live_state),
        ("12_final_gate.csv", gate),
        ("13_causal_audit.csv", causal),
        ("14_failures.csv", failures),
    )
    for name, frame in outputs:
        _csv(root / name, frame)
    (root / "99_decision.md").write_text(
        decision_markdown(
            decision,
            reason,
            historical=historical,
            scenarios=scenarios,
            lot_sizes=lot_sizes,
            risk=risk,
            gate=gate,
        ),
        encoding="utf-8",
    )
    (root / "GPT_REVIEW_PROMPT.md").write_text(
        "Review R03.4.2.15 as the final pre-holdout account and live-readiness audit. "
        "Verify the continuous 2024-2025 OOS compounding, stress cells, top-10 removal, net-R reserve, "
        "OKX lot sizing, immutable model governance, restart-safe state contract and sealed 2026 boundary.\n",
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=root,
            edge_id="eth_ai_r03_4_2_15_final_account_live_readiness",
            title="ETH AI R03.4.2.15 final account and live-readiness audit",
            decision_focus="whether the frozen C2 sleeve is ready for one-time 2026 sealed validation",
        )
    )
