#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02.3.1 pipeline: hurdle nuisance residualization before any new data family."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cache import dataset_cache_path, load_frame, save_frame
from .config import DEFAULT_CONFIG, MODEL_NAME, STAGE_ID, HurdleResidualizationConfig
from .labels import attach_ranking_targets
from .modeling import feature_importance, fit_models, predict, ranking_metrics, regression_metrics, top_zone_summary
from .nuisance import attach_past_only_nuisance_predictions, nuisance_metric_table
from .reports import causal_audit, write_reports
from .source import load_source


@dataclass(frozen=True)
class HurdleResidualizationResult:
    decision: str
    report_dir: Path
    rows: int


def run_hurdle_residualization(
    *,
    skip_review_pack: bool = False,
    use_cache: bool = True,
    progress: bool = True,
    config: HurdleResidualizationConfig = DEFAULT_CONFIG,
) -> HurdleResidualizationResult:
    config.validate()
    print(f"[run] {MODEL_NAME} {STAGE_ID}", flush=True)
    print(
        "[design] exact first touch -> two-part nuisance expectation(distance + broad activity) -> "
        "Residual Liquidity / Residual Reversal ranking; PRIMARY excludes raw distance, nuisance activity and Swing",
        flush=True,
    )
    source, source_gate = load_source()
    print(
        f"[source] R02.2 rows={len(source):,} exact_complete={int(source['first_touch_label_complete'].sum()):,} "
        f"upstream-eligible={int(source['r02_3_1_upstream_eligible'].sum()):,}",
        flush=True,
    )
    cache = dataset_cache_path(config)
    nuisance_feature_audit = None
    nuisance_fold_audit = None
    if use_cache and cache.exists():
        frame = load_frame(cache)
        # Audits are deterministic and cheap; rebuild only audit metadata, not nuisance predictions.
        print(f"[label-cache] rows={len(frame):,}", flush=True)
        # Cache stores audit metadata columns only in the frame; for report-level audit tables,
        # rebuild from the cached schema using tiny vectorized checks below in nuisance module via full source if necessary.
        from .nuisance import nuisance_feature_columns, nuisance_feature_audit as build_feature_audit
        nuisance_feature_audit = build_feature_audit(frame, nuisance_feature_columns(frame))
        # Fold audit is persisted as attrs when supported; fall back to a compact reconstruction from source labels.
        raw = frame.attrs.get("r02_3_1_fold_audit")
        if raw is not None:
            import pandas as pd
            nuisance_fold_audit = pd.DataFrame(raw)
        else:
            # Force a cheap rebuild only if an old cache lacks audit metadata. This path should not occur for current cache version.
            raise RuntimeError("R02.3.1 cache missing fold-audit metadata; rerun with --no-cache once")
    else:
        print("[stage] fit past-only expanding hurdle nuisance models and construct residual targets", flush=True)
        result = attach_past_only_nuisance_predictions(source, config, progress=progress)
        frame = attach_ranking_targets(result.frame, config)
        nuisance_feature_audit = result.feature_audit
        nuisance_fold_audit = result.fold_audit
        if use_cache:
            frame.attrs["r02_3_1_fold_audit"] = nuisance_fold_audit.to_dict(orient="records")
            save_frame(cache, frame)
    if frame.empty:
        raise RuntimeError("R02.3.1 produced no rows")
    if "excess_residual_group_eligible" not in frame:
        frame = attach_ranking_targets(frame, config)
    print(
        f"[dataset] source_eligible={int(frame['r02_3_1_source_eligible'].sum()):,} "
        f"excess_groups={frame.loc[frame['excess_residual_group_eligible'], 'ranking_group'].nunique():,} "
        f"reversal_groups={frame.loc[frame['reversal_residual_group_eligible'], 'ranking_group'].nunique():,}",
        flush=True,
    )
    print("[stage] fit residual No-Swing rankers + Swing ablation + retained sweep geometry", flush=True)
    models = fit_models(frame, config)
    pred = predict(frame, models)
    nuisance_metrics = nuisance_metric_table(pred, config)
    metrics = ranking_metrics(pred, config)
    regression = regression_metrics(pred, config)
    top = top_zone_summary(pred, config)
    importance = feature_importance(models)
    causal = causal_audit(pred, models, source_gate, nuisance_feature_audit, nuisance_fold_audit, config)
    print("[stage] write R02.3.1 residualization report and GPT review pack", flush=True)
    report_dir, decision = write_reports(
        config=config,
        source_gate=source_gate,
        frame=pred,
        nuisance_feature_audit=nuisance_feature_audit,
        nuisance_fold_audit=nuisance_fold_audit,
        nuisance_metrics=nuisance_metrics,
        metrics=metrics,
        regression=regression,
        top=top,
        importance=importance,
        causal=causal,
        skip_review_pack=skip_review_pack,
    )
    print(f"[decision] {decision}", flush=True)
    print(f"[done] report={report_dir}", flush=True)
    return HurdleResidualizationResult(decision=decision, report_dir=report_dir, rows=len(frame))
