#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R01.2 pipeline: explain stable clusters and audit fixed causal confirmations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DEFAULT_CONFIG, MODEL_NAME, STAGE_ID, StablePathExecutionAuditConfig
from .cache import (
    load_replay_cache,
    load_scan_cache,
    replay_cache_key,
    replay_cache_path,
    save_replay_cache,
    save_scan_cache,
    scan_cache_path,
    source_cache_key,
)
from .replay import ReplayResult, replay_samples
from .reports import (
    causal_audit,
    confirmation_detection_summary,
    confirmation_period_stability,
    confirmation_rule_summary,
    write_reports,
)
from .source import resolve_source_paths, scan_source_tables
from src.ai_research.config import PROJECT_ROOT
from .statistics import (
    block_bootstrap_ci,
    cluster_feature_profiles,
    cluster_registry,
    daily_stability,
    feature_family_profiles,
    monthly_stability,
    runtime_signature,
    stability_scorecard,
)


@dataclass(frozen=True)
class StablePathAuditResult:
    decision: str
    report_dir: Path
    source_rows_scanned: int
    replay_episodes: int


def run_stable_path_execution_audit(
    *,
    data_dir: str | Path | None = None,
    db_name: str = "okx_trade_bars.db",
    progress: bool = True,
    skip_review_pack: bool = False,
    skip_micro_replay: bool = False,
    use_cache: bool = True,
    config: StablePathExecutionAuditConfig = DEFAULT_CONFIG,
) -> StablePathAuditResult:
    config.validate()
    print(f"[run] {MODEL_NAME} {STAGE_ID}", flush=True)
    print(f"[source] {config.source_report_path}", flush=True)
    print(
        "[design] liquidity-first; Cluster IDs explain R01.1 discovery only; Swing is never an admission gate",
        flush=True,
    )
    print("[stage] stream aligned R01.1 feature/label/assignment tables", flush=True)
    source_paths = resolve_source_paths(config)
    source_key = source_cache_key(config, source_paths)
    scan_path = scan_cache_path(config, source_key)
    if use_cache and scan_path.exists():
        try:
            scan = load_scan_cache(scan_path)
            print(f"[source-cache] loaded {scan_path}", flush=True)
        except (OSError, ValueError, EOFError):
            scan_path.unlink(missing_ok=True)
            scan = scan_source_tables(config, progress=progress)
            save_scan_cache(scan_path, scan)
    else:
        scan = scan_source_tables(config, progress=progress)
        if use_cache:
            save_scan_cache(scan_path, scan)
    if scan.episode_rows.empty:
        raise RuntimeError("R01.2 found no first-event representatives for target liquidity-release clusters")
    print(
        f"[source-complete] rows={scan.scanned_rows:,} target_episodes={len(scan.episode_rows):,} "
        f"profile_strata={len(scan.profile_samples):,} replay_sample={len(scan.replay_samples):,}",
        flush=True,
    )
    print("[stage] Episode-level stability and day-block bootstrap", flush=True)
    registry = cluster_registry(config, scan.episode_rows)
    stability = stability_scorecard(scan.episode_rows, config)
    daily = daily_stability(scan.episode_rows)
    monthly = monthly_stability(scan.episode_rows)
    bootstrap = block_bootstrap_ci(daily, config)
    print("[stage] explain target clusters against same-side same-period baseline", flush=True)
    profiles = cluster_feature_profiles(scan.profile_samples, scan.feature_columns, config)
    families = feature_family_profiles(profiles)
    signatures = runtime_signature(profiles)
    if skip_micro_replay:
        replay = ReplayResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(
            [{"check": "micro_replay_skipped", "value": 1, "status": "WARN"}]
        ))
    else:
        print("[stage] fixed causal 1s confirmation replay", flush=True)
        db_root = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data"
        replay_key = replay_cache_key(config, source_key, db_root / db_name)
        replay_path = replay_cache_path(config, replay_key)
        if use_cache and replay_path.exists():
            try:
                replay = load_replay_cache(replay_path)
                print(f"[replay-cache] loaded {replay_path}", flush=True)
            except (OSError, ValueError, EOFError):
                replay_path.unlink(missing_ok=True)
                replay = replay_samples(
                    scan.replay_samples, config, data_dir=data_dir, db_name=db_name, progress=progress
                )
                save_replay_cache(replay_path, replay)
        else:
            replay = replay_samples(
                scan.replay_samples, config, data_dir=data_dir, db_name=db_name, progress=progress
            )
            if use_cache:
                save_replay_cache(replay_path, replay)
    detection = confirmation_detection_summary(replay.confirmation_events)
    confirmation_summary = confirmation_rule_summary(replay.confirmation_events, config)
    period_execution = confirmation_period_stability(replay.confirmation_events, config)
    causal = causal_audit(config, scan.source_gate, replay.replay_quality)
    print("[stage] write compact R01.2 report and review pack", flush=True)
    report_dir, decision = write_reports(
        config=config,
        source_gate_frame=scan.source_gate,
        registry=registry,
        stability=stability,
        daily=daily,
        monthly=monthly,
        bootstrap=bootstrap,
        feature_profiles=profiles,
        family_profiles=families,
        runtime_signature=signatures,
        aligned_price=replay.aligned_price_path,
        aligned_flow=replay.aligned_flow_path,
        replay_quality=replay.replay_quality,
        detection=detection,
        confirmation_summary=confirmation_summary,
        period_execution=period_execution,
        causal=causal,
        scanned_rows=scan.scanned_rows,
        skip_review_pack=skip_review_pack,
    )
    print(f"[decision] {decision}", flush=True)
    print(f"[done] report={report_dir}", flush=True)
    return StablePathAuditResult(
        decision=decision,
        report_dir=report_dir,
        source_rows_scanned=scan.scanned_rows,
        replay_episodes=int(scan.replay_samples["event_id"].nunique()) if not scan.replay_samples.empty else 0,
    )
