#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reports for the July-2026 forward extension."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import FOLD_ID, ForwardExtensionConfig


def _csv(path, frame: pd.DataFrame) -> None:
    out = frame.copy()
    if "fold_id" in out.columns:
        out["fold_id"] = out["fold_id"].astype(str).replace({"WF_2026_SEALED": FOLD_ID})
    out.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def map_decision(decision: str) -> str:
    return {
        "PASS_2026_SEALED_HOLDOUT": "JULY_FORWARD_SUPPORTS_FROZEN_C2",
        "PASS_2026_SEALED_HOLDOUT_WITH_CAVEATS": "JULY_FORWARD_MIXED_SUPPORT",
        "FAIL_2026_SEALED_HOLDOUT": "JULY_FORWARD_DOES_NOT_SUPPORT_FROZEN_C2",
    }.get(decision, decision)


def _comparison(config: ForwardExtensionConfig, score_audit: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    h1_score = pd.read_csv(config.source_2_16_path / "03_model_threshold_audit.csv")
    h1_summary = pd.read_csv(config.source_2_16_path / "12_holdout_scenario_summary.csv")
    rows: list[dict[str, object]] = []
    july_exceedance = float(score_audit.iloc[0]["test_exceedance_rate"]) if not score_audit.empty else np.nan
    h1_exceedance = float(h1_score.iloc[0]["test_exceedance_rate"]) if not h1_score.empty else np.nan
    for delay in config.entry_delay_minutes:
        for cost in config.cost_multipliers:
            old = h1_summary.loc[
                h1_summary["delay_minutes"].astype(int).eq(int(delay))
                & np.isclose(h1_summary["cost_multiplier"].astype(float), float(cost))
            ]
            new = summary.loc[
                summary["delay_minutes"].astype(int).eq(int(delay))
                & np.isclose(summary["cost_multiplier"].astype(float), float(cost))
            ]
            if len(old) != 1 or len(new) != 1:
                continue
            a, b = old.iloc[0], new.iloc[0]
            rows.append(
                {
                    "delay_minutes": int(delay),
                    "cost_multiplier": float(cost),
                    "h1_trades": int(a["executed_cycles"]),
                    "july_trades": int(b["executed_cycles"]),
                    "h1_total_return": float(a["total_net_return"]),
                    "july_total_return": float(b["total_net_return"]),
                    "h1_profit_factor": float(a["profit_factor"]),
                    "july_profit_factor": float(b["profit_factor"]),
                    "h1_win_rate": float(a["win_rate"]),
                    "july_win_rate": float(b["win_rate"]),
                    "h1_max_drawdown": float(a["max_drawdown"]),
                    "july_max_drawdown": float(b["max_drawdown"]),
                    "h1_q70_exceedance": h1_exceedance,
                    "july_q70_exceedance": july_exceedance,
                }
            )
    return pd.DataFrame(rows)


def _decision_markdown(
    decision: str,
    reason: str,
    *,
    config: ForwardExtensionConfig,
    score_audit: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    summary: pd.DataFrame,
    extended: pd.DataFrame,
    gate: pd.DataFrame,
    seal_check: dict[str, object],
    comparison: pd.DataFrame,
) -> str:
    mapped = map_decision(decision)
    lines = [
        "# R03.4.2.16.1 2026年7月前向扩展验证",
        "",
        f"## 决策：`{mapped}`",
        "",
        reason,
        "",
        "## 解释边界",
        "",
        "- R03.4.2.16的2026年1—6月纯封存失败结论保持不变。",
        "- 本阶段不修复、不重训、不重新校准，也不把7月包装成新的半年封存。",
        "- 7月只用于判断冻结C2在新的市场月份是否恢复，以及熊市/状态不适配假设是否获得支持。",
        "- 单月样本有限；即使7月很好，也不能推翻上半年封存失败或直接批准实盘。",
        "",
        "## 不可变策略",
        "",
        "- q70信号后下一根1m open立即入场；所有q70统一等风险。",
        "- 2%真实硬止损；1.5%完成15m收盘软失败。",
        "- `failed_reclaim`非时间结构退出；不加仓、无固定止盈。",
        "",
        "## 时间边界",
        "",
        f"- 训练：{config.fit_start} → {config.fit_end}。",
        f"- q70校准：{config.calibration_start} → {config.calibration_end}。",
        f"- 新前向窗口：{config.holdout_start} → {config.holdout_end}。",
        f"- 封印状态：{seal_check.get('status', 'NOT_RUN')}。",
    ]
    if not score_audit.empty:
        row = score_audit.iloc[0]
        lines += [
            "",
            "## 模型分数",
            "",
            f"- 冻结q70阈值：{float(row['calibration_threshold']):.8f}。",
            f"- 7月超过q70比例：{float(row['test_exceedance_rate']):.2%}。",
            f"- 特征Schema一致：{bool(row['feature_schema_matches_history'])}。",
        ]
    if not fixed_summary.empty:
        lines += ["", "## 7月固定6小时开仓Edge诊断", ""]
        for row in fixed_summary.sort_values(["delay_minutes", "cost_multiplier"]).to_dict("records"):
            lines.append(
                f"- 延迟{int(row['delay_minutes'])}m/成本{float(row['cost_multiplier']):.0f}x："
                f"信号{int(row['signals'])}，均值{float(row['mean_net_return']):.3%}，"
                f"胜率{float(row['win_rate']):.1%}，PF={float(row['profit_factor']):.2f}。"
            )
    if not summary.empty:
        lines += ["", "## 7月冻结C2账户", ""]
        for row in summary.sort_values(["delay_minutes", "cost_multiplier"]).to_dict("records"):
            lines.append(
                f"- 延迟{int(row['delay_minutes'])}m/成本{float(row['cost_multiplier']):.0f}x："
                f"交易{int(row['executed_cycles'])}，收益{float(row['total_net_return']):.1%}，"
                f"MDD={float(row['max_drawdown']):.1%}，PF={float(row['profit_factor']):.2f}，"
                f"胜率{float(row['win_rate']):.1%}。"
            )
    if not comparison.empty:
        anchor = comparison.loc[
            comparison["delay_minutes"].astype(int).eq(config.anchor_delay_minutes)
            & np.isclose(comparison["cost_multiplier"].astype(float), config.anchor_cost_multiplier)
        ]
        if len(anchor) == 1:
            row = anchor.iloc[0]
            lines += [
                "",
                "## 与2026上半年对比（1m延迟/2x成本）",
                "",
                f"- q70超阈率：上半年{float(row['h1_q70_exceedance']):.2%} → 7月{float(row['july_q70_exceedance']):.2%}。",
                f"- PF：上半年{float(row['h1_profit_factor']):.2f} → 7月{float(row['july_profit_factor']):.2f}。",
                f"- 胜率：上半年{float(row['h1_win_rate']):.1%} → 7月{float(row['july_win_rate']):.1%}。",
                f"- 收益：上半年{float(row['h1_total_return']):.1%} → 7月{float(row['july_total_return']):.1%}。",
            ]
    if not extended.empty:
        lines += ["", "## 2024至2026-07拼接诊断", ""]
        for row in extended.sort_values(["delay_minutes", "cost_multiplier"]).to_dict("records"):
            lines.append(
                f"- 延迟{int(row['delay_minutes'])}m/成本{float(row['cost_multiplier']):.0f}x："
                f"累计交易{int(row['trades'])}，累计收益{float(row['total_return']):.1%}。"
            )
    lines += ["", "## 诊断门", ""]
    for row in gate.to_dict("records"):
        lines.append(
            f"- [{row.get('gate_class', 'unknown')}] {row['check']}: "
            f"{bool(row['pass'])} (value={row.get('value')}, threshold={row.get('threshold')})"
        )
    lines += [
        "",
        "## 后续纪律",
        "",
        "- 不允许利用7月调整q70阈值、模型、止损、软失败或退出。",
        "- 7月改善只能支持市场状态不适配假设，不能洗掉上半年封存失败。",
        "- 7月仍弱则进一步支持模型漂移/Long状态门控缺失，下一步应做失败归因或独立V2。",
    ]
    return "\n".join(lines) + "\n"


def write_reports(
    *, config: ForwardExtensionConfig, manifest: dict[str, object], preflight: dict[str, object],
    pre_open_seal: dict[str, object], holdout_open_log: dict[str, object], historical: pd.DataFrame,
    score_audit: pd.DataFrame, fixed_trades: pd.DataFrame, fixed_summary: pd.DataFrame,
    selected_events: pd.DataFrame, structure_timeline: pd.DataFrame, cycles: pd.DataFrame,
    legs: pd.DataFrame, actions: pd.DataFrame, daily: pd.DataFrame, summary: pd.DataFrame,
    months: pd.DataFrame, quarters: pd.DataFrame, extended: pd.DataFrame, gate: pd.DataFrame,
    causal: pd.DataFrame, rejections: pd.DataFrame, failures: pd.DataFrame,
    seal_check: dict[str, object], decision: str, reason: str,
) -> None:
    root = config.report_path
    root.mkdir(parents=True, exist_ok=True)
    manifest = dict(manifest)
    manifest["stage"] = "R03.4.2.16.1"
    manifest["name"] = "July-2026 forward extension of unchanged frozen C2"
    _json(root / "00_run_manifest.json", manifest)
    if pre_open_seal:
        seal_path = root / "00_pre_open_seal.json"
        if not seal_path.exists():
            _json(seal_path, pre_open_seal)
    _json(root / "01_preflight.json", preflight)
    if holdout_open_log:
        open_log = dict(holdout_open_log)
        open_log["stage"] = "R03.4.2.16.1"
        open_log["status"] = "JULY_FORWARD_OPENED"
        _json(root / "01_holdout_open_log.json", open_log)
    _json(root / "18_post_run_seal_check.json", seal_check)

    outputs = (
        ("02_historical_metric_contract.csv", historical),
        ("03_model_threshold_audit.csv", score_audit),
        ("04_fixed_6h_trades.csv", fixed_trades),
        ("05_fixed_6h_summary.csv", fixed_summary),
        ("06_selected_failed_reclaim_events.csv", selected_events),
        ("07_soft_structure_timeline.csv", structure_timeline),
        ("08_account_cycles.csv", cycles),
        ("09_account_legs.csv", legs),
        ("10_account_actions.csv", actions),
        ("11_daily_equity.csv", daily),
        ("12_july_scenario_summary.csv", summary),
        ("13_monthly_returns.csv", months),
        ("14_quarterly_returns.csv", quarters),
        ("15_extended_oos_through_july.csv", extended),
        ("16_forward_diagnostic_gate.csv", gate),
        ("17_causal_audit.csv", causal),
        ("19_runtime_rejections.csv", rejections),
        ("20_failures.csv", failures),
    )
    for name, frame in outputs:
        _csv(root / name, frame)
    comparison = _comparison(config, score_audit, summary) if failures.empty else pd.DataFrame()
    _csv(root / "21_h1_vs_july_comparison.csv", comparison)
    mapped = map_decision(decision)
    mapped_reason = {
        "JULY_FORWARD_SUPPORTS_FROZEN_C2": "冻结C2在7月新前向窗口恢复并通过单月诊断门；这支持市场状态不适配假设，但不推翻上半年封存失败。",
        "JULY_FORWARD_MIXED_SUPPORT": "冻结C2在7月出现部分恢复，但单月质量或集中度仍有警告；结论只能视为混合证据。",
        "JULY_FORWARD_DOES_NOT_SUPPORT_FROZEN_C2": "冻结C2在7月新前向窗口仍未恢复；市场状态不适配不足以单独解释上半年失败。",
    }.get(mapped, reason)
    (root / "99_decision.md").write_text(
        _decision_markdown(
            decision, mapped_reason, config=config, score_audit=score_audit,
            fixed_summary=fixed_summary, summary=summary, extended=extended,
            gate=gate, seal_check=seal_check, comparison=comparison,
        ),
        encoding="utf-8",
    )
    (root / "GPT_REVIEW_PROMPT.md").write_text(
        "Review R03.4.2.16.1 as a July-only forward extension. Confirm that the pre-2026 fit and Q4-2025 q70 threshold are unchanged, January-June 2026 is comparison-only, July is not used for tuning, and a good July result is not misrepresented as reversing the failed R03.4.2.16 seal.\n",
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=root,
            edge_id="eth_ai_r03_4_2_16_1_2026_july_forward_extension",
            title="ETH AI R03.4.2.16.1 July-2026 forward extension",
            decision_focus="whether unchanged frozen C2 recovers in the new July window and what that implies about regime dependence",
        )
    )
