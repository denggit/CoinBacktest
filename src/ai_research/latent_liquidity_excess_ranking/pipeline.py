#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02.3 pipeline: train-only distance normalization + separate spatial rankers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .cache import dataset_cache_path, load_frame, save_frame
from .config import DEFAULT_CONFIG, MODEL_NAME, STAGE_ID, ExcessLiquidityRankingConfig
from .labels import DistanceNormalizer, attach_excess_and_reversal_targets, fit_distance_normalizer
from .modeling import feature_importance, fit_models, predict, ranking_metrics, regression_metrics, top_zone_summary
from .reports import causal_audit, write_reports
from .source import load_r02_2_exact_first_touch_dataset


@dataclass(frozen=True)
class ExcessLiquidityRankingResult:
    decision: str
    report_dir: Path
    rows: int


def run_excess_liquidity_ranking(
    *,
    skip_review_pack: bool = False,
    use_cache: bool = True,
    progress: bool = True,
    config: ExcessLiquidityRankingConfig = DEFAULT_CONFIG,
) -> ExcessLiquidityRankingResult:
    del progress  # Label transformation is vectorized and fast; no hidden long loop exists in R02.3.
    config.validate()
    print(f"[run] {MODEL_NAME} {STAGE_ID}", flush=True)
    print(
        "[design] exact first touch -> TRAIN-only side x distance robust normalization -> "
        "Excess Liquidity rank + separate Reversal Quality rank; PRIMARY excludes Swing",
        flush=True,
    )
    source, source_gate = load_r02_2_exact_first_touch_dataset()
    mismatch = int((source["first_touch_label_complete"].astype(bool) & ~source["r02_touch_consistent"].astype(bool)).sum())
    same_bar = int((source["first_touch_observed"].astype(bool) & source["first_touch_time"].eq(source["decision_time"])).sum())
    print(
        f"[source] R02.2 rows={len(source):,} exact_complete={int(source['first_touch_label_complete'].sum()):,} "
        f"same-bar-start={same_bar:,} quarantined-r02-touch-mismatch={mismatch:,}",
        flush=True,
    )
    normalizer = fit_distance_normalizer(source, config)
    cache = dataset_cache_path(config)
    if use_cache and cache.exists():
        frame = load_frame(cache)
        print(f"[label-cache] rows={len(frame):,}", flush=True)
    else:
        print("[stage] vectorized distance normalization + excess/reversal target construction", flush=True)
        frame = attach_excess_and_reversal_targets(source, normalizer, config)
        if use_cache:
            save_frame(cache, frame)
    if frame.empty:
        raise RuntimeError("R02.3 produced no rows")
    print(
        f"[dataset] source_eligible={int(frame['r02_3_source_eligible'].sum()):,} "
        f"excess_rank_groups={frame.loc[frame['excess_group_eligible'], 'ranking_group'].nunique():,} "
        f"reversal_rank_groups={frame.loc[frame['reversal_group_eligible'], 'ranking_group'].nunique():,}",
        flush=True,
    )
    print("[stage] fit No-Swing PRIMARY excess/reversal rankers + Swing ablation + sweep geometry", flush=True)
    models = fit_models(frame, config)
    pred = predict(frame, models)
    metrics = ranking_metrics(pred, config)
    regression = regression_metrics(pred, config)
    top = top_zone_summary(pred, config)
    importance = feature_importance(models)
    causal = causal_audit(pred, models, source_gate, normalizer, config)
    print("[stage] write compact R02.3 report, worklog evidence and GPT review pack", flush=True)
    report_dir, decision = write_reports(
        config=config,
        source_gate=source_gate,
        frame=pred,
        normalizer=normalizer,
        metrics=metrics,
        regression=regression,
        top=top,
        importance=importance,
        causal=causal,
        skip_review_pack=skip_review_pack,
    )
    print(f"[decision] {decision}", flush=True)
    print(f"[done] report={report_dir}", flush=True)
    return ExcessLiquidityRankingResult(decision=decision, report_dir=report_dir, rows=len(frame))
