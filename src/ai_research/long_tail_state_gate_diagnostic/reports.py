#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.17."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import StateGateDiagnosticConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def state_definition_frame(config: StateGateDiagnosticConfig) -> pd.DataFrame:
    return pd.DataFrame([
        {"dimension": "trend_1d/trend_4h", "state": "UP", "definition": "completed close > EMA20 > EMA50 and EMA20 slope over 3 completed bars > 0", "causal": True},
        {"dimension": "trend_1d/trend_4h", "state": "DOWN", "definition": "completed close < EMA20 < EMA50 and EMA20 slope over 3 completed bars < 0", "causal": True},
        {"dimension": "trend_1d/trend_4h", "state": "MIXED", "definition": "neither strict UP nor strict DOWN", "causal": True},
        {"dimension": "combined_state", "state": "BULL_ALIGNED", "definition": "1D UP and 4H UP", "causal": True},
        {"dimension": "combined_state", "state": "BULL_TACTICAL", "definition": "1D not DOWN and 4H UP", "causal": True},
        {"dimension": "combined_state", "state": "BEAR_ALIGNED", "definition": "1D DOWN and 4H DOWN", "causal": True},
        {"dimension": "combined_state", "state": "BEAR_TACTICAL", "definition": "1D not UP and 4H DOWN", "causal": True},
        {"dimension": "combined_state", "state": "MIXED", "definition": "all remaining alignments", "causal": True},
        {"dimension": "drawdown_state", "state": "NEAR_90D_HIGH", "definition": f"completed 1D close drawdown from rolling 90D high >= {config.near_90d_high_drawdown:.0%}", "causal": True},
        {"dimension": "drawdown_state", "state": "CORRECTION", "definition": f"90D drawdown between {config.deep_90d_drawdown:.0%} and {config.near_90d_high_drawdown:.0%}", "causal": True},
        {"dimension": "drawdown_state", "state": "DEEP_DRAWDOWN", "definition": f"90D drawdown <= {config.deep_90d_drawdown:.0%}", "causal": True},
        {"dimension": "vol_state", "state": "VOL_EXPANDING", "definition": f"4H ATR14/ATR60 >= {config.high_vol_ratio:.2f}", "causal": True},
        {"dimension": "vol_state", "state": "VOL_COMPRESSED", "definition": f"4H ATR14/ATR60 <= {config.low_vol_ratio:.2f}", "causal": True},
        {"dimension": "vol_state", "state": "VOL_NORMAL", "definition": "between fixed volatility lifecycle boundaries", "causal": True},
    ])


def _period_block(summary: pd.DataFrame, period: str) -> list[str]:
    lines: list[str] = []
    focus = summary.loc[(summary["analysis_period"] == period) & (summary["state_dimension"] == "combined_state")]
    for row in focus.sort_values("state_value").to_dict("records"):
        lines.append(
            f"- {row['state_value']}: {int(row['trades'])}笔，收益{float(row['total_return']):.1%}，"
            f"均值{float(row['mean_return']):.3%}，PF={float(row['profit_factor']):.2f}，"
            f"胜率{float(row['win_rate']):.1%}，硬止损{float(row['hard_stop_share']):.1%}，"
            f"软失败{float(row['soft_failure_share']):.1%}。"
        )
    return lines


def _decision_markdown(
    decision: str,
    reason: str,
    *,
    c2_summary: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    score_summary: pd.DataFrame,
    gate_summary: pd.DataFrame,
    findings: pd.DataFrame,
    monthly: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.17 2026封存失败归因与Long市场状态诊断",
        "",
        f"## 诊断结论：`{decision}`",
        "",
        reason,
        "",
        "## 约束与解释边界",
        "",
        "- R03.4.2.16的`FAIL_2026_SEALED_HOLDOUT`仍然有效；V1不得直接实盘。",
        "- 本阶段不修改q70、模型、止损、软失败、failed_reclaim、仓位或加仓规则。",
        "- 状态门控的反事实结果是在已经看过2026后形成的开发证据，不能视为OOS通过。",
        "- 任何V2门控必须重新版本化，并由未来尚未观察的数据验证。",
        "",
        "## 冻结C2在因果1D/4H状态下的表现",
    ]
    if c2_summary.empty:
        lines += ["", "无完整状态归因结果。"]
    else:
        for period in ("2024", "2025", "2026_H1", "2026_JULY"):
            lines += ["", f"### {period}", ""]
            lines += _period_block(c2_summary, period) or ["- 无样本。"]
    if not fixed_summary.empty:
        lines += ["", "## 2026固定6小时开仓Edge按状态", ""]
        for row in fixed_summary.sort_values(["analysis_period", "combined_state"]).to_dict("records"):
            lines.append(
                f"- {row['analysis_period']}/{row['combined_state']}: {int(row['trades'])}个信号，"
                f"均值{float(row['mean_return']):.3%}，PF={float(row['profit_factor']):.2f}，"
                f"胜率{float(row['win_rate']):.1%}，MFE={float(row['mean_mfe']):.2%}，MAE={float(row['mean_mae']):.2%}。"
            )
    if not score_summary.empty:
        lines += ["", "## 冻结分数的状态条件漂移", ""]
        for period in ("CAL_Q4_2025", "2026_Q1", "2026_Q2", "2026_JULY"):
            focus = score_summary.loc[score_summary["analysis_period"] == period]
            if focus.empty:
                continue
            weighted = np.average(focus["q70_exceedance_rate"], weights=focus["decision_rows"])
            lines.append(f"- {period}: 总体状态加权q70超阈率约{float(weighted):.1%}。")
            for row in focus.sort_values("combined_state").to_dict("records"):
                lines.append(
                    f"  - {row['combined_state']}: {int(row['decision_rows'])}行，"
                    f"超阈{float(row['q70_exceedance_rate']):.1%}，中位分数{float(row['median_score']):.6f}。"
                )
    if not monthly.empty:
        lines += ["", "## 月度市场与C2", ""]
        for row in monthly.tail(10).to_dict("records"):
            lines.append(
                f"- {row['month']}: ETH {float(row['eth_month_return']):.1%}，"
                f"C2账户月收益{float(row['c2_account_return']):.1%}，"
                f"当月入场交易{int(row['c2_entry_trades']) if pd.notna(row.get('c2_entry_trades')) else 0}，"
                f"q70超阈{float(row['q70_exceedance_rate']):.1%}，主状态{row['dominant_state']}。"
            )
    lines += ["", "## 预定义门控反事实（仅开发诊断）", ""]
    if gate_summary.empty:
        lines.append("- 无结果。")
    else:
        for gate, group in gate_summary.groupby("gate", sort=False):
            positives = int((group["total_return"] > 0).sum())
            min_coverage = float(group["coverage"].min())
            lines.append(f"- {gate}: 4个时期中正收益{positives}/4，最低覆盖{min_coverage:.1%}。")
    lines += ["", "## 关键归因", ""]
    for row in findings.to_dict("records"):
        lines.append(f"- {row['finding']}: {bool(row['supported'])}；{row['detail']}")
    lines += [
        "",
        "## 后续纪律",
        "",
        "- 如果状态依赖得到支持：只允许设计独立V2门控规范，不得把本报告当作验证。",
        "- 如果分数漂移占主导：优先研究滚动校准/漂移检测，但不能在2026上宣布修复。",
        "- 如果没有简单分离：结束C2主线，把资源转向独立Short、长趋势和流动性模型。",
        "- 无论本阶段结论如何，V1都保持`NOT LIVE APPROVED`。",
    ]
    return "\n".join(lines) + "\n"


def write_reports(
    *,
    config: StateGateDiagnosticConfig,
    manifest: dict[str, object],
    preflight: dict[str, object],
    source_integrity: pd.DataFrame,
    state_definition: pd.DataFrame,
    state_timeline: pd.DataFrame,
    cycles: pd.DataFrame,
    c2_state_summary: pd.DataFrame,
    fixed_trades: pd.DataFrame,
    fixed_state_summary: pd.DataFrame,
    scores: pd.DataFrame,
    score_summary: pd.DataFrame,
    monthly: pd.DataFrame,
    gate_summary: pd.DataFrame,
    findings: pd.DataFrame,
    model_audit: pd.DataFrame,
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
        ("02_source_integrity.csv", source_integrity),
        ("03_state_definition.csv", state_definition),
        ("04_state_timeline_15m.csv", state_timeline),
        ("05_c2_cycles_with_state.csv", cycles),
        ("06_c2_state_summary.csv", c2_state_summary),
        ("07_fixed6h_with_state.csv", fixed_trades),
        ("08_fixed6h_state_summary.csv", fixed_state_summary),
        ("09_score_state_rows.csv", scores),
        ("10_score_state_summary.csv", score_summary),
        ("11_monthly_market_vs_c2.csv", monthly),
        ("12_counterfactual_gate_summary.csv", gate_summary),
        ("13_attribution_findings.csv", findings),
        ("14_model_recipe_audit.csv", model_audit),
        ("15_causal_audit.csv", causal),
        ("16_failures.csv", failures),
    )
    for name, frame in outputs:
        _csv(root / name, frame)
    (root / "99_decision.md").write_text(
        _decision_markdown(
            decision, reason, c2_summary=c2_state_summary, fixed_summary=fixed_state_summary,
            score_summary=score_summary, gate_summary=gate_summary, findings=findings, monthly=monthly,
        ),
        encoding="utf-8",
    )
    (root / "GPT_REVIEW_PROMPT.md").write_text(
        "Review R03.4.2.17 strictly as a post-seal diagnostic. Verify completed-bar 1D/4H availability, exact frozen model recipe, source-report integrity, separation of fixed-6h entry Edge from C2 exit-overlay returns, and explicit disclosure that counterfactual gates are not validated or live-approved.\n",
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=root,
            edge_id="eth_ai_r03_4_2_17_state_gate_diagnostic",
            title="ETH AI R03.4.2.17 sealed-failure state attribution",
            decision_focus="whether V1 failure is better explained by Long-regime dependence, score drift, exit-overlay concentration, or no simple causal gate",
        )
    )
