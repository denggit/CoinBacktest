#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R01.3 pipeline: learn absorption completion and remaining executable space."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

from .cache import (
    load_source_scan,
    replay_key,
    save_source_scan,
    snapshot_root,
    source_key,
    source_scan_path,
)
from .config import DEFAULT_CONFIG, MODEL_NAME, STAGE_ID, AbsorptionModelConfig
from .evaluation import (
    attach_trade_stress,
    calibration_thresholds,
    monthly_summary,
    score_deciles,
    select_first_snapshot,
    threshold_audit,
    trade_summary,
)
from .modeling import feature_importance, fit_models, metric_table, predict
from .replay import build_snapshot_dataset
from .reports import causal_audit, label_summary, selected_cluster_summary, write_reports
from .source import scan_sources, source_paths


@dataclass(frozen=True)
class AbsorptionModelResult:
    decision: str
    report_dir: Path
    source_rows_scanned: int
    snapshot_rows: int
    episodes: int


def run_absorption_remaining_space_model(
    *,
    data_dir: str | Path | None = None,
    db_name: str = "okx_trade_bars.db",
    progress: bool = True,
    skip_review_pack: bool = False,
    use_cache: bool = True,
    config: AbsorptionModelConfig = DEFAULT_CONFIG,
) -> AbsorptionModelResult:
    config.validate()
    print(f"[run] {MODEL_NAME} {STAGE_ID}", flush=True)
    print(f"[source] {config.source_report_path}", flush=True)
    print(
        "[design] Cluster is upstream liquidity-path context; R01.3 learns causal absorption and remaining space; Swing is not a gate",
        flush=True,
    )
    print("[stage] deterministic Episode sample from aligned R01.1 tables", flush=True)
    paths = source_paths(config)
    src_key = source_key(config, paths)
    scan_path = source_scan_path(config, src_key)
    if use_cache and scan_path.exists():
        try:
            scan = load_source_scan(scan_path)
            print(f"[source-cache] loaded {scan_path}", flush=True)
        except (OSError, ValueError, EOFError):
            scan_path.unlink(missing_ok=True)
            scan = scan_sources(config, progress=progress)
            save_source_scan(scan_path, scan)
    else:
        scan = scan_sources(config, progress=progress)
        if use_cache:
            save_source_scan(scan_path, scan)
    if scan.replay_samples.empty:
        raise RuntimeError("R01.3 found no deterministic Episode samples")
    print(
        f"[source-complete] rows={scan.scanned_rows:,} sampled_episodes={scan.replay_samples['event_id'].nunique():,}",
        flush=True,
    )
    print("[stage] causal 1-second multi-checkpoint snapshots and future labels", flush=True)
    db_root = Path(data_dir) if data_dir is not None else PROJECT_ROOT / "data"
    snap_key = replay_key(config, src_key, db_root / db_name)
    build = build_snapshot_dataset(
        scan.replay_samples,
        config,
        cache_root=snapshot_root(config, snap_key),
        data_dir=data_dir,
        db_name=db_name,
        progress=progress,
        use_cache=use_cache,
    )
    snapshots = build.snapshots
    if snapshots.empty:
        raise RuntimeError("R01.3 produced no complete causal snapshots")
    required_periods = set(config.periods)
    observed_periods = set(snapshots["period"].astype(str).unique())
    if not required_periods <= observed_periods:
        raise RuntimeError(f"R01.3 missing frozen periods: {sorted(required_periods - observed_periods)}")
    print(
        f"[snapshots] rows={len(snapshots):,} episodes={snapshots['event_id'].nunique():,} "
        f"train={int(snapshots['period'].eq(config.train_period).sum()):,} "
        f"calibration={int(snapshots['period'].eq(config.calibration_period).sum()):,} "
        f"holdout={int(snapshots['period'].eq(config.holdout_period).sum()):,}",
        flush=True,
    )
    print("[stage] fixed multi-task LightGBM fit on 2023-2024 only", flush=True)
    models = fit_models(snapshots, config)
    predictions = predict(snapshots, models)
    metrics = metric_table(predictions, config)
    importance = feature_importance(models)
    deciles = score_deciles(predictions, config)
    print("[stage] freeze q90 score threshold on 2025Q1-Q3 and select first causal snapshot per Episode", flush=True)
    full_threshold = calibration_thresholds(predictions, config, "trade_score")
    baseline_threshold = calibration_thresholds(predictions, config, "p_tradeable_baseline")
    thresholds = pd.concat([full_threshold, baseline_threshold], ignore_index=True, copy=False)
    full_selected = select_first_snapshot(predictions, full_threshold, score_column="trade_score", model_name="FULL")
    baseline_selected = select_first_snapshot(
        predictions,
        baseline_threshold,
        score_column="p_tradeable_baseline",
        model_name="BASELINE",
    )
    selected = pd.concat([full_selected, baseline_selected], ignore_index=True, copy=False)
    trades = attach_trade_stress(selected, config)
    summary = trade_summary(trades)
    monthly = monthly_summary(trades)
    cluster_summary = selected_cluster_summary(trades)
    threshold_frame = threshold_audit(thresholds, config)
    labels = label_summary(snapshots)
    causal = causal_audit(
        snapshots,
        models.feature_columns,
        scan.source_gate,
        build.quality,
        threshold_frame,
        config,
    )
    print("[stage] write compact R01.3 commercial-gate report", flush=True)
    report_dir, decision = write_reports(
        config=config,
        source_gate=scan.source_gate,
        replay_quality=build.quality,
        snapshots=snapshots,
        label_frame=labels,
        metrics=metrics,
        deciles=deciles,
        importance=importance,
        thresholds=threshold_frame,
        selected_trades=trades,
        trade_summary=summary,
        cluster_summary=cluster_summary,
        monthly=monthly,
        causal=causal,
        source_rows_scanned=scan.scanned_rows,
        feature_columns=models.feature_columns,
        skip_review_pack=skip_review_pack,
    )
    print(f"[decision] {decision}", flush=True)
    print(f"[done] report={report_dir}", flush=True)
    return AbsorptionModelResult(
        decision=decision,
        report_dir=report_dir,
        source_rows_scanned=int(scan.scanned_rows),
        snapshot_rows=int(len(snapshots)),
        episodes=int(snapshots["event_id"].nunique()),
    )
