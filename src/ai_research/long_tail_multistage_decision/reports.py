#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.3."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import LongTailMultistageConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    *,
    stable: pd.DataFrame,
    policy_summary: pd.DataFrame,
    model_metrics: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.3 多阶段持仓决策与q70扩展",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 冻结研究原则",
        "",
        "- 开仓模型仍是R03.4.1的六小时多头效用模型，不重新调参。",
        "- 持仓模型训练事件来自整个测试年前的严格滚动OOF q50池，而不是单一季度。",
        "- T+60只观察；T+180只有‘失败风险高且恢复概率低’才允许提前退出。",
        "- T+360决定退出或继续到24小时；T+24小时决定退出或最多继续到5天。",
        "- q70、q90分别纯OOS审核；增加次数不能以负期望为代价。",
        "- 市场状态模型已舍弃，不参与本研究。",
        "",
    ]
    if not stable.empty:
        lines.extend(["## 稳健策略候选", ""])
        passed = stable.loc[stable["stable_positive_expectancy"] == True]  # noqa: E712
        if passed.empty:
            lines.append("- 没有策略通过2倍成本、跨年PF、去Top10、季度和回撤硬门槛。")
        else:
            for row in passed.itertuples():
                lines.append(
                    f"- {row.scope} / {row.policy}: 2024净期望={row.mean_net_2x_2024:.3%}, "
                    f"2025净期望={row.mean_net_2x_2025:.3%}, PF={row.pf_2x_2024:.2f}/{row.pf_2x_2025:.2f}, "
                    f"交易={int(row.trades_2024)}/{int(row.trades_2025)}, "
                    f"同池利润升级={bool(row.beats_same_scope_fixed_both_years)}, q70总利润升级={bool(row.q70_expands_total_profit_both_years)}。"
                )
        lines.append("")
    if not policy_summary.empty:
        lines.extend(["## 主要策略结果（1分钟延迟、2倍成本）", ""])
        primary = policy_summary.loc[(policy_summary["delay_minutes"] == 1) & (policy_summary["cost_multiplier"] == 2.0)]
        for row in primary.sort_values(["scope", "policy", "fold_id"]).itertuples():
            lines.append(
                f"- {row.fold_id} {row.scope} {row.policy}: trades={int(row.trades)}, "
                f"mean={row.mean_net_return:.3%}, PF={row.profit_factor:.2f}, win={row.win_rate:.1%}, "
                f"MDD={row.max_drawdown:.1%}, hold={row.median_holding_minutes:.0f}m。"
            )
        lines.append("")
    if not model_metrics.empty:
        lines.extend(["## 持仓模型OOS概览", ""])
        for row in model_metrics.sort_values(["task", "checkpoint_minutes", "scope", "fold_id"]).head(24).itertuples():
            lines.append(
                f"- {row.fold_id} {row.scope} {row.task}@T+{int(row.checkpoint_minutes)}m "
                f"{row.feature_set}: AUC={row.roc_auc:.3f}, AP lift={row.average_precision_lift:.2f}, "
                f"Top-decile lift={row.top_decile_lift:.2f}。"
            )
        lines.append("")
    lines.extend(
        [
            "## 解释边界",
            "",
            "- 本研究中的5天是安全评估上限，不代表最终实盘必须按时间退出。",
            "- 多阶段方案只有通过正期望硬门槛，才允许进入完整策略回测。",
            "- 若q70仅增加次数但PF、净期望或回撤恶化，则必须维持q90。",
            "",
        ]
    )
    return "\n".join(lines)


def empty(
    config: LongTailMultistageConfig,
    preflight: dict[str, object],
    decision: str,
    reason: str,
) -> None:
    frame = pd.DataFrame()
    write_reports(
        config=config,
        preflight=preflight,
        manifest={"stage": "R03.4.2.3", "config": config.to_dict()},
        contract={},
        entry_oof_audit=frame,
        extraction_audit=frame,
        dataset_summary=frame,
        model_selection=frame,
        model_metrics=frame,
        thresholds=frame,
        importance=frame,
        predictions=frame,
        policy_summary=frame,
        period_summary=frame,
        exit_summary=frame,
        overlap_audit=frame,
        stable=frame,
        causal_audit=frame,
        failures=frame,
        trades=frame,
        decision=decision,
        reason=reason,
    )


def write_reports(
    *,
    config: LongTailMultistageConfig,
    preflight: dict[str, object],
    manifest: dict[str, object],
    contract: dict[str, object],
    entry_oof_audit: pd.DataFrame,
    extraction_audit: pd.DataFrame,
    dataset_summary: pd.DataFrame,
    model_selection: pd.DataFrame,
    model_metrics: pd.DataFrame,
    thresholds: pd.DataFrame,
    importance: pd.DataFrame,
    predictions: pd.DataFrame,
    policy_summary: pd.DataFrame,
    period_summary: pd.DataFrame,
    exit_summary: pd.DataFrame,
    overlap_audit: pd.DataFrame,
    stable: pd.DataFrame,
    causal_audit: pd.DataFrame,
    failures: pd.DataFrame,
    trades: pd.DataFrame,
    decision: str,
    reason: str,
) -> None:
    report_dir = config.report_path
    report_dir.mkdir(parents=True, exist_ok=True)
    _json(report_dir / "00_run_manifest.json", manifest)
    _json(report_dir / "01_preflight.json", preflight)
    _json(report_dir / "02_multistage_contract.json", contract)
    _csv(report_dir / "03_entry_oof_audit.csv", entry_oof_audit)
    _csv(report_dir / "04_event_extraction_audit.csv", extraction_audit)
    _csv(report_dir / "05_dataset_summary.csv", dataset_summary)
    _csv(report_dir / "06_model_selection_audit.csv", model_selection)
    _csv(report_dir / "07_model_metrics.csv", model_metrics)
    _csv(report_dir / "08_probability_thresholds.csv", thresholds)
    _csv(report_dir / "09_feature_importance.csv", importance)
    _csv(report_dir / "10_prediction_samples.csv", predictions)
    _csv(report_dir / "11_policy_summary.csv", policy_summary)
    _csv(report_dir / "12_quarter_summary.csv", period_summary)
    _csv(report_dir / "13_exit_reason_summary.csv", exit_summary)
    _csv(report_dir / "14_overlap_and_skip_audit.csv", overlap_audit)
    _csv(report_dir / "15_stable_candidates.csv", stable)
    _csv(report_dir / "16_causal_audit.csv", causal_audit)
    _csv(report_dir / "17_failures.csv", failures)
    _csv(report_dir / "18_trade_details.csv", trades)
    (report_dir / "99_decision.md").write_text(
        decision_markdown(decision, reason, stable=stable, policy_summary=policy_summary, model_metrics=model_metrics),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.2.3",
            edge_id="long_tail_multistage_q70_expansion",
            stage="research",
            title="ETH AI R03.4.2.3 multi-stage holding and q70 expansion",
            decision_focus="whether expanded causal path training can separate persistent failure, recoverable drawdown and healthy long holds while increasing total positive expectancy with q70",
            print_log=True,
        )
    )
