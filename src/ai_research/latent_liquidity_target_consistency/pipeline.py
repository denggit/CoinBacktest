#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02.3.1b pipeline: target consistency before any path-model or data-family expansion."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .audit import causal_audit
from .cache import dataset_cache_path, load_frame, save_frame
from .config import DEFAULT_CONFIG, MODEL_NAME, STAGE_ID, TargetConsistencyConfig
from .nuisance import (
    attach_past_only_target_consistency_predictions,
    nuisance_feature_audit,
    nuisance_feature_columns,
    nuisance_metric_table,
)
from .reports import write_reports
from .source import load_source


@dataclass(frozen=True)
class TargetConsistencyResult:
    decision: str
    report_dir: Path
    rows: int


def run_target_consistency_audit(
    *,
    skip_review_pack: bool = False,
    use_cache: bool = True,
    progress: bool = True,
    config: TargetConsistencyConfig = DEFAULT_CONFIG,
) -> TargetConsistencyResult:
    config.validate()
    print(f"[run] {MODEL_NAME} {STAGE_ID}", flush=True)
    print(
        "[design] audit-only: exact first touch -> past-only hurdle nuisance -> "
        "legacy vs formula-only vs L2-mean-aligned log residual; no PATH ranker / no new data family",
        flush=True,
    )
    source, source_gate = load_source()
    print(
        f"[source] R02.2 rows={len(source):,} exact_complete={int(source['first_touch_label_complete'].sum()):,} "
        f"upstream-eligible={int(source['r02_3_1b_upstream_eligible'].sum()):,}",
        flush=True,
    )

    cache = dataset_cache_path(config)
    feature_audit: pd.DataFrame
    fold_audit: pd.DataFrame
    if use_cache and cache.exists():
        frame = load_frame(cache)
        print(f"[label-cache] rows={len(frame):,}", flush=True)
        feature_audit = nuisance_feature_audit(frame, nuisance_feature_columns(frame))
        raw = frame.attrs.get("r02_3_1b_fold_audit")
        if raw is None:
            raise RuntimeError("R02.3.1b cache missing fold-audit metadata; rerun once with --no-cache")
        fold_audit = pd.DataFrame(raw)
    else:
        print("[stage] fit expanding/frozen hurdle nuisance models on consistent log scale", flush=True)
        result = attach_past_only_target_consistency_predictions(source, config, progress=progress)
        frame = result.frame
        feature_audit = result.feature_audit
        fold_audit = result.fold_audit
        if use_cache:
            frame.attrs["r02_3_1b_fold_audit"] = fold_audit.to_dict(orient="records")
            save_frame(cache, frame)

    if frame.empty:
        raise RuntimeError("R02.3.1b produced no rows")
    print(
        f"[dataset] source_eligible={int(frame['r02_3_1b_source_eligible'].sum()):,} "
        f"train_oos={int((frame['r02_3_1b_source_eligible'] & frame['nuisance_prediction_source'].eq('TRAIN_EXPANDING_OOS')).sum()):,}",
        flush=True,
    )
    print("[stage] audit target identity, distance cells, yearly stability and nuisance drift", flush=True)
    metrics = nuisance_metric_table(frame, config)
    causal = causal_audit(frame, source_gate, feature_audit, fold_audit, config)
    print("[stage] write R02.3.1b report and GPT review pack", flush=True)
    report_dir, decision = write_reports(
        config=config,
        source_gate=source_gate,
        frame=frame,
        nuisance_feature_audit=feature_audit,
        nuisance_fold_audit=fold_audit,
        nuisance_metrics=metrics,
        causal=causal,
        skip_review_pack=skip_review_pack,
    )
    print(f"[decision] {decision}", flush=True)
    print(f"[done] report={report_dir}", flush=True)
    return TargetConsistencyResult(decision=decision, report_dir=report_dir, rows=len(frame))
