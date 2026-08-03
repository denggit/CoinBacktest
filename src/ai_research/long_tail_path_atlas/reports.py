#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writer for R03.4.2.1 path atlas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.research_common.review_pack import ReviewPackConfig, write_gpt_review_pack

from .config import LongTailPathAtlasConfig


def _csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def decision_markdown(
    decision: str,
    reason: str,
    config: LongTailPathAtlasConfig,
    oos_features: pd.DataFrame,
    path_type_summary: pd.DataFrame,
) -> str:
    lines = [
        "# R03.4.2.1 q90多头事件完整路径图谱与类型研究",
        "",
        f"## 决策：`{decision}`",
        "",
        reason,
        "",
        "## 本阶段不做什么",
        "",
        "- 不修改冻结的q90基础多头LightGBM、训练区间或阈值方法。",
        "- 不尝试止损、止盈、追踪、续期参数，也不选出交易退出冠军。",
        "- 不使用已舍弃的战略、战术、入场或活跃状态模型。",
        "- 不只研究盈利事件；所有q90独立事件都进入图谱，避免幸存者偏差。",
        "- 路径类型使用未来轨迹，只能作为历史研究标签，不能直接在实盘时刻使用。",
        "",
        "## 因果与OOS边界",
        "",
        "- 入场固定为15分钟决策后的下一分钟开盘。",
        "- 2024路径类型模型只使用2023Q4校准期事件建立。",
        "- 2025路径类型模型使用截至2024Q4以前可获得的校准路径建立。",
        "- 2024/2025完整路径只用于OOS分布与稳定性审核，不用于回头修改类型定义。",
        "- 2026继续封存；缺少完整48小时路径的年末事件被排除。",
        "",
        "## 输出解释",
        "",
        "- `02_oos_path_features.csv`：每笔OOS事件的一行完整路径特征。",
        "- `03_path_type_assignments.csv`：逐笔语义类型、聚类类型与重要路径旗标。",
        "- `04_path_type_summary.csv`：各类型的胜率、MFE、MAE、回吐、晚续涨和最佳历史退出时点。",
        "- `08_representative_events.csv`：每类典型、最好、最差及最大MFE事件，便于逐笔检查。",
        "- `event_paths/*.csv.gz`：每笔事件48小时完整1分钟轨迹；大文件不进入GPT review pack。",
        "- `11_oracle_exit_envelope.csv`：事后最佳固定观察时点，仅用于判断退出空间，绝非可交易结果。",
        "",
    ]
    if not oos_features.empty:
        lines.extend(
            [
                "## OOS事件规模",
                "",
            ]
        )
        for fold_id, group in oos_features.groupby("fold_id", sort=False):
            lines.append(
                f"- {fold_id}: {len(group)}笔完整48小时事件；固定6小时成本后盈利率 {group['fixed6h_positive_expectancy_event'].mean():.2%}；"
                f"平均6小时净收益 {group['fixed6h_net_1x'].mean():.4%}。"
            )
        lines.append("")
    if not path_type_summary.empty:
        lines.extend(["## 类型覆盖", ""])
        for fold_id, group in path_type_summary.groupby("fold_id", sort=False):
            eligible = group.loc[group["events"] >= config.minimum_type_samples]
            labels = ", ".join(f"{row.semantic_path_type}={int(row.events)}" for row in eligible.itertuples())
            lines.append(f"- {fold_id}: {labels or '没有达到最低样本数的稳定类型'}")
        lines.append("")
    lines.extend(
        [
            "## 下一阶段允许做的事",
            "",
            "只有当某类路径在2024和2025都存在、规模足够且行为相似时，下一阶段才能研究当时可观测的早期预测特征。",
            "R03.4.2.2应预测‘这笔事件正在走向哪种路径’，而不是直接使用未来路径标签；随后才可冻结差异化退出逻辑并重新做纯OOS交易验证。",
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    report_dir: Path,
    *,
    manifest: dict[str, object],
    preflight: dict[str, object],
    contract: dict[str, object],
    event_audit: pd.DataFrame,
    discovery_features: pd.DataFrame,
    oos_features: pd.DataFrame,
    assignments: pd.DataFrame,
    path_type_summary: pd.DataFrame,
    path_type_period: pd.DataFrame,
    target_timing: pd.DataFrame,
    winner_loser_contrast: pd.DataFrame,
    representatives: pd.DataFrame,
    cluster_centroids: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    oracle_exit: pd.DataFrame,
    score_bins: pd.DataFrame,
    path_file_manifest: dict[str, object],
    causal_audit: pd.DataFrame,
    failures: pd.DataFrame,
    decision: str,
    reason: str,
    config: LongTailPathAtlasConfig,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    _json(report_dir / "00_run_manifest.json", manifest)
    _json(report_dir / "01_preflight.json", preflight)
    _json(report_dir / "02_path_atlas_contract.json", contract)
    _csv(report_dir / "03_event_extraction_audit.csv", event_audit)
    _csv(report_dir / "04_discovery_path_features.csv", discovery_features)
    _csv(report_dir / "05_oos_path_features.csv", oos_features)
    _csv(report_dir / "06_path_type_assignments.csv", assignments)
    _csv(report_dir / "07_path_type_summary.csv", path_type_summary)
    _csv(report_dir / "08_path_type_by_quarter.csv", path_type_period)
    _csv(report_dir / "09_target_hit_timing.csv", target_timing)
    _csv(report_dir / "10_winner_loser_path_contrast.csv", winner_loser_contrast)
    _csv(report_dir / "11_representative_events.csv", representatives)
    _csv(report_dir / "12_discovery_cluster_centroids.csv", cluster_centroids)
    _csv(report_dir / "13_oos_cluster_summary.csv", cluster_summary)
    _csv(report_dir / "14_oracle_exit_envelope.csv", oracle_exit)
    _csv(report_dir / "15_score_path_bins.csv", score_bins)
    _json(report_dir / "16_event_path_files.json", path_file_manifest)
    _csv(report_dir / "17_causal_audit.csv", causal_audit)
    _csv(report_dir / "18_failures.csv", failures)
    (report_dir / "99_decision.md").write_text(
        decision_markdown(decision, reason, config, oos_features, path_type_summary),
        encoding="utf-8",
    )
    write_gpt_review_pack(
        ReviewPackConfig(
            report_dir=report_dir,
            experiment_id="R03.4.2.1",
            edge_id="frozen_q90_long_event_path_atlas",
            stage="research",
            title="ETH AI R03.4.2.1 frozen q90 long event complete path atlas",
            decision_focus="what path families produce the q90 long edge before designing any new exit mechanism",
            print_log=True,
        )
    )
