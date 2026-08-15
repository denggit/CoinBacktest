#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02.4 pipeline: ask whether the latent-liquidity reversal mechanism contains enough money at all."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .audit import (
    attach_oracle_metrics,
    causal_audit,
    ceiling_distribution,
    decision_from_episode_metrics,
    fixed_r_performance,
    yearly_ceiling,
)
from .config import DEFAULT_CONFIG, MODEL_NAME, STAGE_ID, EconomicCeilingConfig
from .reports import write_reports
from .source import load_release_episode_source


@dataclass(frozen=True)
class EconomicCeilingResult:
    decision: str
    report_dir: Path
    release_episodes: int


def run_economic_ceiling_audit(
    *,
    progress: bool = True,
    skip_review_pack: bool = False,
    config: EconomicCeilingConfig = DEFAULT_CONFIG,
) -> EconomicCeilingResult:
    config.validate()
    print(f"[run] {MODEL_NAME} {STAGE_ID}", flush=True)
    print("[purpose] stop/go audit: prove economic room exists before another identification model is allowed", flush=True)
    print("[warning] oracle future information is intentional and may never become a live rule", flush=True)
    source = load_release_episode_source(config, progress=progress)
    print(f"[source] scanned={source.scanned_rows:,} release_episodes={len(source.episodes):,}", flush=True)
    print("[stage] attach 60/180/300/600s MFE/MAE ceilings and fixed-R oracle realizations", flush=True)
    episodes = attach_oracle_metrics(source.episodes, config)
    distribution = ceiling_distribution(episodes, config)
    performance = fixed_r_performance(episodes, config)
    yearly = yearly_ceiling(episodes, config)
    decision, gate = decision_from_episode_metrics(episodes, config)
    causal = causal_audit(episodes, source.source_gate, config)
    failures = causal.loc[causal["status"].astype(str).eq("FAIL"), "check"].tolist()
    if failures:
        decision = "BLOCKED_R02_4_SOURCE_OR_ORACLE_AUDIT_FAILED"
    print("[stage] write compact economic-ceiling report", flush=True)
    report_dir = write_reports(
        config=config,
        source_gate=source.source_gate,
        episodes=episodes,
        distribution=distribution,
        performance=performance,
        yearly=yearly,
        decision_gate=gate,
        causal=causal,
        decision=decision,
        scanned_rows=source.scanned_rows,
        skip_review_pack=skip_review_pack,
    )
    print(f"[decision] {decision}", flush=True)
    print(f"[done] report={report_dir}", flush=True)
    return EconomicCeilingResult(decision=decision, report_dir=report_dir, release_episodes=len(episodes))
