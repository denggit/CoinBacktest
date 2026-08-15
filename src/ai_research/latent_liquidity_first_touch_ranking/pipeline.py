#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02.2 pipeline: exact first-touch labels + cross-sectional liquidity ranking."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .cache import dataset_cache_path, load_frame, save_frame
from .config import DEFAULT_CONFIG, MODEL_NAME, STAGE_ID, FirstTouchLiquidityRankingConfig
from .labels import add_relative_relevance, build_first_touch_dataset
from .modeling import feature_importance, fit_models, predict, ranking_metrics, top_zone_summary
from .reports import causal_audit, write_reports
from .source import load_r02_audit_lattice_and_episodes


@dataclass(frozen=True)
class FirstTouchRankingResult:
    decision: str
    report_dir: Path
    rows: int


def run_first_touch_liquidity_ranking(
    *,
    data_dir: str | Path | None = None,
    db_name: str = "okx_trade_bars.db",
    skip_review_pack: bool = False,
    use_cache: bool = True,
    progress: bool = True,
    config: FirstTouchLiquidityRankingConfig = DEFAULT_CONFIG,
) -> FirstTouchRankingResult:
    config.validate()
    print(f"[run] {MODEL_NAME} {STAGE_ID}", flush=True)
    print("[design] exact first touch -> fixed 30/60/180/300s labels -> within-snapshot relative ranking; no absolute strength threshold", flush=True)
    audit, episodes, source_gate = load_r02_audit_lattice_and_episodes()
    print(
        f"[source] complete-lattice rows={len(audit):,} groups={audit.groupby(['decision_time','zone_side']).ngroups:,} Episodes={len(episodes):,}",
        flush=True,
    )
    cache = dataset_cache_path(config)
    if use_cache and cache.exists():
        frame = load_frame(cache)
        quality = pd.DataFrame([{"cache": True, "rows": len(frame)}])
        print(f"[first-touch-cache] rows={len(frame):,}", flush=True)
    else:
        print("[stage] resolve exact first touch from 1m -> 1s and build equal-duration post-touch labels", flush=True)
        built = build_first_touch_dataset(
            audit,
            episodes,
            config,
            use_cache=use_cache,
            data_dir=data_dir,
            db_name=db_name,
            progress=progress,
        )
        frame, quality = built.frame, built.quality
        frame = add_relative_relevance(frame, config)
        if use_cache:
            save_frame(cache, frame)
    if frame.empty:
        raise RuntimeError("R02.2 produced no rows")
    if "ranking_group_eligible" not in frame:
        frame = add_relative_relevance(frame, config)
    print(
        f"[dataset] rows={len(frame):,} exact_touch={int(frame['first_touch_observed'].sum()):,} "
        f"complete={int(frame['first_touch_label_complete'].sum()):,} rank_groups={frame.loc[frame['ranking_group_eligible'], 'ranking_group'].nunique():,}",
        flush=True,
    )
    print("[stage] fit path-no-Swing PRIMARY ranker vs full-with-Swing ablation; distance is mechanical baseline", flush=True)
    models = fit_models(frame, config)
    pred = predict(frame, models)
    metrics = ranking_metrics(pred, config)
    top = top_zone_summary(pred, config)
    importance = feature_importance(models)
    causal = causal_audit(pred, models, source_gate, config)
    print("[stage] write compact R02.2 report and review pack", flush=True)
    report_dir, decision = write_reports(
        config=config,
        source_gate=source_gate,
        quality=quality,
        frame=pred,
        metrics=metrics,
        top=top,
        importance=importance,
        causal=causal,
        skip_review_pack=skip_review_pack,
    )
    print(f"[decision] {decision}", flush=True)
    print(f"[done] report={report_dir}", flush=True)
    return FirstTouchRankingResult(decision=decision, report_dir=report_dir, rows=len(frame))
