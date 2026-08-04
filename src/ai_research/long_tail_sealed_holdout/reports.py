#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.16 sealed validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack
from .config import SealedHoldoutConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(decision: str, reason: str, *, config: SealedHoldoutConfig, score_audit: pd.DataFrame, fixed_summary: pd.DataFrame, summary: pd.DataFrame, extended: pd.DataFrame, gate: pd.DataFrame, seal_check: dict[str, object]) -> str:
    lines = [
        "# R03.4.2.16 2026一次性纯封存验证", "", f"## 决策：`{decision}`", "", reason, "",
        "## 不可变冻结策略", "", "- q70信号后下一根1m open立即入场。", "- 所有q70统一等风险；持仓中不加仓。",
        "- 2%真实硬止损；1.5%不利波动后由完成15m收盘确认软失败。", "- `failed_reclaim`为非时间结构退出；没有固定止盈。",
        "- 2026仅用于一次性推理与评分，禁止用于训练、阈值、规则选择或事后修补。", "", "## 封存边界", "",
        f"- 训练：{config.fit_start} → {config.fit_end}。", f"- q70校准：{config.calibration_start} → {config.calibration_end}。",
        f"- 纯封存：{config.holdout_start} → {config.holdout_end}（半年，不是完整年度）。", f"- 封印状态：{seal_check.get('status', 'NOT_RUN')}。",
    ]
    if not score_audit.empty:
        row = score_audit.iloc[0]
        lines += ["", "## 模型与阈值", "", f"- 训练行数：{int(row['fit_rows']):,}；校准行数：{int(row['calibration_rows']):,}；封存推理行数：{int(row['test_rows']):,}。", f"- Q4 2025冻结q70阈值：{float(row['calibration_threshold']):.8f}。", f"- 2026超过q70的决策比例：{float(row['test_exceedance_rate']):.2%}。", f"- 特征Schema与历史一致：{bool(row['feature_schema_matches_history'])}。"]
    if not fixed_summary.empty:
        lines += ["", "## 固定6小时信号诊断（非最终账户退出）", ""]
        for row in fixed_summary.sort_values(["delay_minutes", "cost_multiplier"]).to_dict("records"):
            lines.append(f"- 延迟{int(row['delay_minutes'])}m/成本{float(row['cost_multiplier']):.0f}x：信号{int(row['signals'])}，均值{float(row['mean_net_return']):.3%}，胜率{float(row['win_rate']):.1%}，PF={float(row['profit_factor']):.2f}。")
    if not summary.empty:
        lines += ["", "## 2026冻结C2账户", ""]
        for row in summary.sort_values(["delay_minutes", "cost_multiplier"]).to_dict("records"):
            lines.append(f"- 延迟{int(row['delay_minutes'])}m/成本{float(row['cost_multiplier']):.0f}x：交易{int(row['executed_cycles'])}，收益{float(row['total_net_return']):.1%}，MDD={float(row['max_drawdown']):.1%}，PF={float(row['profit_factor']):.2f}，胜率{float(row['win_rate']):.1%}，正收益月{int(row['positive_months'])}/6，去前10大{float(row['total_return_without_top10']):.1%}。")
    if not extended.empty:
        lines += ["", "## 2024至2026-06拼接OOS诊断", ""]
        for row in extended.sort_values(["delay_minutes", "cost_multiplier"]).to_dict("records"):
            lines.append(f"- 延迟{int(row['delay_minutes'])}m/成本{float(row['cost_multiplier']):.0f}x：累计交易{int(row['trades'])}，最终权益{float(row['final_equity']):.3f}，累计收益{float(row['total_return']):.1%}。")
    lines += ["", "## 预注册资格门", ""]
    for row in gate.to_dict("records"):
        lines.append(f"- [{row.get('gate_class', 'unknown')}] {row['check']}: {bool(row['pass'])} (value={row.get('value')}, threshold={row.get('threshold')})")
    lines += ["", "## 后续纪律", ""]
    if decision.startswith("PASS_2026_SEALED_HOLDOUT"):
        lines += ["- 正式冻结为 `MF Long Sleeve V1`；不再利用2026调整本Sleeve。", "- 下一阶段只能做模型工件导出、AetherEdge影子实盘、状态恢复和实时一致性验证。", "- 其他8大模型与多种开仓模型必须作为独立Sleeve重新研究，不能污染本Sleeve的封存结论。"]
    elif decision == "FAIL_2026_SEALED_HOLDOUT":
        lines += ["- 诚实归档失败；禁止在本2026封存集上继续改参数。", "- 后续只能提出新一代独立假设，并使用未来尚未观察的数据做新封存验证。"]
    else:
        lines.append("- 先修复数据或运行完整性问题；不得解释不完整收益。")
    return "\n".join(lines) + "\n"


def write_reports(*, config: SealedHoldoutConfig, manifest: dict[str, object], preflight: dict[str, object], pre_open_seal: dict[str, object], holdout_open_log: dict[str, object], historical: pd.DataFrame, score_audit: pd.DataFrame, fixed_trades: pd.DataFrame, fixed_summary: pd.DataFrame, selected_events: pd.DataFrame, structure_timeline: pd.DataFrame, cycles: pd.DataFrame, legs: pd.DataFrame, actions: pd.DataFrame, daily: pd.DataFrame, summary: pd.DataFrame, months: pd.DataFrame, quarters: pd.DataFrame, extended: pd.DataFrame, gate: pd.DataFrame, causal: pd.DataFrame, rejections: pd.DataFrame, failures: pd.DataFrame, seal_check: dict[str, object], decision: str, reason: str) -> None:
    root = config.report_path; root.mkdir(parents=True, exist_ok=True)
    _json(root / "00_run_manifest.json", manifest)
    if pre_open_seal:
        seal_path = root / "00_pre_open_seal.json"
        if not seal_path.exists(): _json(seal_path, pre_open_seal)
    _json(root / "01_preflight.json", preflight)
    if holdout_open_log: _json(root / "01_holdout_open_log.json", holdout_open_log)
    _json(root / "18_post_run_seal_check.json", seal_check)
    outputs = (
        ("02_historical_metric_contract.csv", historical), ("03_model_threshold_audit.csv", score_audit), ("04_fixed_6h_trades.csv", fixed_trades),
        ("05_fixed_6h_summary.csv", fixed_summary), ("06_selected_failed_reclaim_events.csv", selected_events), ("07_soft_structure_timeline.csv", structure_timeline),
        ("08_account_cycles.csv", cycles), ("09_account_legs.csv", legs), ("10_account_actions.csv", actions), ("11_daily_equity.csv", daily),
        ("12_holdout_scenario_summary.csv", summary), ("13_monthly_returns.csv", months), ("14_quarterly_returns.csv", quarters),
        ("15_extended_oos_summary.csv", extended), ("16_sealed_gate.csv", gate), ("17_causal_audit.csv", causal),
        ("19_runtime_rejections.csv", rejections), ("20_failures.csv", failures),
    )
    for name, frame in outputs: _csv(root / name, frame)
    (root / "99_decision.md").write_text(decision_markdown(decision, reason, config=config, score_audit=score_audit, fixed_summary=fixed_summary, summary=summary, extended=extended, gate=gate, seal_check=seal_check), encoding="utf-8")
    (root / "GPT_REVIEW_PROMPT.md").write_text("Review R03.4.2.16 as a one-time 2026 sealed holdout. Verify the pre-open SHA-256 seal, pre-2026 fit/calibration boundaries, unchanged C2 execution, complete 1/3/5m and 2x/3x grid, partial-year disclosure, and the prohibition on post-2026 tuning.\n", encoding="utf-8")
    write_gpt_review_pack(ReviewPackConfig(report_dir=root, edge_id="eth_ai_r03_4_2_16_2026_sealed_validation", title="ETH AI R03.4.2.16 one-time 2026 sealed validation", decision_focus="whether frozen C2 passes the untouched 2026-01-01 through 2026-06-30 holdout"))
