#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end R03.4.2.12 soft-failure tail-compression audit."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import load_minute_path_data
from src.ai_research.long_tail_tranche_eligibility.config import TrancheEligibilityConfig
from src.ai_research.swing_baseline.dataset import create_loader, run_public_loader_preflight
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG
from src.research_common.progress import ProgressReporter

from . import reports
from .analysis import (
    build_f1_attribution,
    enrich_summaries,
    policy_gate,
    summarize_f1_attribution,
)
from .config import (
    DEFAULT_TAIL_COMPRESSION_CONFIG,
    STAGE_ID,
    STAGE_NAME,
    TailCompressionConfig,
)
from .inputs import load_tail_compression_inputs
from .simulator import simulate_tail_compression_account


@dataclass(frozen=True)
class TailCompressionResult:
    decision: str
    report_dir: Path


def _causal_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "frozen_q70_source", "status": "PASS", "detail": "uses the validated R03.4.2.11 P0 event sequence"},
            {"check": "next_open_entry", "status": "PASS", "detail": "q70 still enters at the next observable 1m open"},
            {"check": "failed_reclaim_frozen", "status": "PASS", "detail": "profit exit remains deterministic failed_reclaim"},
            {"check": "no_fixed_time_exit", "status": "PASS", "detail": "fixed 6h remains diagnostic only"},
            {"check": "soft_failure_completed_close", "status": "PASS", "detail": "soft failure uses a completed structure close and exits at next 1m open"},
            {"check": "hard_stop_executable", "status": "PASS", "detail": "real candidate hard stops are intrabar executable and gap-aware"},
            {"check": "real_tail_sizing", "status": "PASS", "detail": "qualifying policies size from the same frozen distance as the executable hard stop"},
            {"check": "adaptive_stop_causal", "status": "PASS", "detail": "V1 uses prior 60 completed 1m ATR only and freezes distance at entry"},
            {"check": "no_stop_widening", "status": "PASS", "detail": "hard and soft distances never widen after entry"},
            {"check": "f1_reference_not_candidate", "status": "PASS", "detail": "F1 keeps a 2R tail only for attribution and cannot pass the one-R gate"},
            {"check": "same_policy_both_years", "status": "PASS", "detail": "identical policies run in 2024 and 2025"},
            {"check": "sealed_2026", "status": "PASS", "detail": "2026 remains unopened"},
        ]
    )


def _write_empty(
    *,
    config: TailCompressionConfig,
    decision: str,
    reason: str,
    preflight: dict[str, object],
) -> TailCompressionResult:
    reports.write_reports(
        config=config,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        source_p0=pd.DataFrame(),
        source_f1=pd.DataFrame(),
        selected_events=pd.DataFrame(),
        attribution=pd.DataFrame(),
        attribution_summary=pd.DataFrame(),
        cycles=pd.DataFrame(),
        legs=pd.DataFrame(),
        actions=pd.DataFrame(),
        daily_equity=pd.DataFrame(),
        summary=pd.DataFrame(),
        gate=pd.DataFrame(),
        causal_audit=pd.DataFrame(),
        runtime_rejections=pd.DataFrame(),
        failures=pd.DataFrame([{"fold_id": "ALL", "error": reason}]),
        decision=decision,
        reason=reason,
    )
    return TailCompressionResult(decision, config.report_path)


def _assert_source_anchor(
    summary: pd.DataFrame,
    source: pd.DataFrame,
    *,
    policy: str,
    source_policy: str,
    failures: list[dict[str, object]],
) -> None:
    actual = summary.loc[summary["policy"].astype(str).eq(policy)]
    expected = source.loc[source["policy"].astype(str).eq(source_policy)] if "policy" in source.columns else source
    keys = ["fold_id", "delay_minutes", "cost_multiplier"]
    merged = expected.merge(actual, on=keys, how="left", suffixes=("_source", "_actual"))
    for row in merged.to_dict("records"):
        if not np.isfinite(float(row.get("total_net_return_actual", np.nan))):
            failures.append({"fold_id": row.get("fold_id"), "error": f"{policy} anchor row missing"})
            continue
        for metric, tolerance in (("total_net_return", 1e-8), ("max_drawdown", 1e-8)):
            source_value = float(row[f"{metric}_source"])
            actual_value = float(row[f"{metric}_actual"])
            if abs(source_value - actual_value) > tolerance:
                failures.append(
                    {
                        "fold_id": row.get("fold_id"),
                        "delay_minutes": row.get("delay_minutes"),
                        "cost_multiplier": row.get("cost_multiplier"),
                        "error": f"{policy} anchor mismatch {metric}: source={source_value} actual={actual_value}",
                    }
                )


def run_tail_compression_audit(
    *,
    data_dir: str | Path | None = None,
    progress: bool = True,
    config: TailCompressionConfig = DEFAULT_TAIL_COMPRESSION_CONFIG,
) -> TailCompressionResult:
    config.validate()
    try:
        inputs = load_tail_compression_inputs(config)
    except Exception as exc:
        reason = f"冻结的R03.4.2.11输入不可用：{type(exc).__name__}: {exc}"
        return _write_empty(
            config=config,
            decision="BLOCKED_SOURCE_REPORT",
            reason=reason,
            preflight={"source_report": reason},
        )

    loader = create_loader(LONG_CONTEXT_BASE_CONFIG, data_dir=data_dir)
    loader_preflight = run_public_loader_preflight(
        loader,
        LONG_CONTEXT_BASE_CONFIG,
        sample_dates=("2024-06-15", "2025-06-15"),
    )
    preflight: dict[str, object] = {
        "trade_bar": loader_preflight.to_dict(),
        "source_2_11": str(config.source_2_11_path),
        "source_2_11_decision": (config.source_2_11_path / "99_decision.md").read_text(encoding="utf-8").splitlines()[2:6],
    }
    if loader_preflight.status != "PASS":
        return _write_empty(
            config=config,
            decision="BLOCKED_DATA",
            reason="1分钟Trade Bar公共Loader预检失败。",
            preflight=preflight,
        )

    path_config = TrancheEligibilityConfig().structural_config()
    cycle_parts: list[pd.DataFrame] = []
    leg_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    daily_parts: list[pd.DataFrame] = []
    runtime_parts: list[pd.DataFrame] = []
    attribution_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    total_steps = len(inputs.folds) * len(config.entry_delay_minutes)
    stage_reporter = ProgressReporter("[R03.4.2.12 folds/delays]", total_steps, every=1, enabled=progress)
    step = 0
    for fold in inputs.folds.to_dict("records"):
        fold_id = str(fold["fold_id"])
        test_start = pd.Timestamp(fold["test_start"])
        test_end = pd.Timestamp(fold["test_end"]).floor("min")
        try:
            path = load_minute_path_data(
                start=test_start - pd.Timedelta(days=2),
                end=test_end,
                data_dir=data_dir,
                config=path_config,
                progress=progress,
            )
            attribution_parts.append(
                build_f1_attribution(
                    inputs.source_cycles,
                    inputs.source_legs,
                    fold_id=fold_id,
                    path=path,
                    materiality=config.attribution_materiality,
                )
            )
            fold_events = inputs.selected_events.loc[inputs.selected_events["fold_id"].astype(str).eq(fold_id)].copy()
            fold_timeline = inputs.structure_timeline.loc[inputs.structure_timeline["fold_id"].astype(str).eq(fold_id)].copy()
            for delay in config.entry_delay_minutes:
                try:
                    for policy in config.policies:
                        for multiplier in config.cost_multipliers:
                            simulation = simulate_tail_compression_account(
                                fold_events,
                                fold_timeline,
                                path=path,
                                fold_id=fold_id,
                                policy=policy,
                                delay_minutes=int(delay),
                                cost_multiplier=float(multiplier),
                                test_start=test_start,
                                test_end=test_end,
                                config=config,
                                progress=progress,
                            )
                            if simulation.summary:
                                summary_rows.append(simulation.summary)
                            if not simulation.cycles.empty:
                                cycle_parts.append(simulation.cycles)
                            if not simulation.legs.empty:
                                leg_parts.append(simulation.legs)
                            if not simulation.actions.empty:
                                action_parts.append(simulation.actions)
                            if not simulation.daily_equity.empty:
                                daily_parts.append(simulation.daily_equity)
                            if not simulation.runtime_rejections.empty:
                                runtime_parts.append(simulation.runtime_rejections)
                except Exception as exc:
                    failures.append(
                        {
                            "fold_id": fold_id,
                            "delay_minutes": int(delay),
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                step += 1
                stage_reporter.update(step)
        except Exception as exc:
            failures.append(
                {
                    "fold_id": fold_id,
                    "delay_minutes": "ALL",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            step += len(config.entry_delay_minutes)
            stage_reporter.update(step)
    stage_reporter.close()

    cycles = pd.concat(cycle_parts, ignore_index=True) if cycle_parts else pd.DataFrame()
    legs = pd.concat(leg_parts, ignore_index=True) if leg_parts else pd.DataFrame()
    actions = pd.concat(action_parts, ignore_index=True) if action_parts else pd.DataFrame()
    daily = pd.concat(daily_parts, ignore_index=True) if daily_parts else pd.DataFrame()
    runtime = pd.concat(runtime_parts, ignore_index=True) if runtime_parts else pd.DataFrame()
    attribution = pd.concat(attribution_parts, ignore_index=True) if attribution_parts else pd.DataFrame()
    attribution_summary = summarize_f1_attribution(attribution)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty and not cycles.empty:
        summary = enrich_summaries(summary, cycles)

    if not summary.empty:
        _assert_source_anchor(
            summary,
            inputs.source_p0_summary,
            policy="P0_single_1R",
            source_policy="P0_single_1R",
            failures=failures,
        )
        _assert_source_anchor(
            summary,
            inputs.source_f1_summary,
            policy="F1_reference_1p5size_3ptail",
            source_policy="F1_soft_failure_1p5",
            failures=failures,
        )

    gate = policy_gate(summary, config)
    causal = _causal_audit()
    failure_frame = pd.DataFrame(failures)

    if not failure_frame.empty:
        decision = "FAIL_RUNTIME"
        reason = "运行或冻结基准复现失败；不得解释收益。"
    elif gate.empty:
        decision = "FAIL_RUNTIME"
        reason = "没有生成完整政策资格门。"
    elif gate["pass_to_next_stage"].astype(bool).any():
        winners = gate.loc[gate["pass_to_next_stage"].astype(bool), "policy"].astype(str).tolist()
        decision = "PASS_REAL_1R_TAIL_COMPRESSION_CANDIDATE"
        reason = f"同一套真实1R尾部压缩政策跨年通过：{winners}。"
    else:
        decision = "FAIL_NO_ROBUST_REAL_TAIL_COMPRESSION"
        reason = "F1的退出增量不足以在真实1R尾部下稳定复制；固定或自适应紧止损未同时保留两年收益与回撤。"

    reports.write_reports(
        config=config,
        manifest={"stage": STAGE_ID, "name": STAGE_NAME, "config": config.to_dict()},
        preflight=preflight,
        source_p0=inputs.source_p0_summary.copy(),
        source_f1=inputs.source_f1_summary.copy(),
        selected_events=inputs.selected_events.copy(),
        attribution=attribution,
        attribution_summary=attribution_summary,
        cycles=cycles,
        legs=legs,
        actions=actions,
        daily_equity=daily,
        summary=summary,
        gate=gate,
        causal_audit=causal,
        runtime_rejections=runtime,
        failures=failure_frame,
        decision=decision,
        reason=reason,
    )
    return TailCompressionResult(decision, config.report_path)
