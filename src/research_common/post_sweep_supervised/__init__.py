#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R13 post-sweep supervised meta-labeling research components."""

from .config import PostSweepSupervisedConfig
from .data import R13DataBundle, load_r13_data
from .features import (
    FeatureModuleResult,
    build_footprint_features,
    build_oi_features,
    build_range_features,
    build_trade_1s_features,
    cache_module,
    checkpoint_index,
    load_cached_module,
)
from .modeling import ModelingResult, run_supervised_ablation
from .reports import (
    ablation_delta,
    causal_audit,
    data_quality_report,
    decision_counts,
    manifest,
    module_coverage_report,
    research_brief,
    write_csv,
    write_manifest,
)

__all__ = [
    "PostSweepSupervisedConfig", "R13DataBundle", "FeatureModuleResult", "ModelingResult",
    "load_r13_data", "checkpoint_index", "build_trade_1s_features", "build_range_features",
    "build_footprint_features", "build_oi_features", "cache_module", "load_cached_module",
    "run_supervised_ablation", "data_quality_report", "module_coverage_report", "causal_audit",
    "ablation_delta", "decision_counts", "manifest", "research_brief", "write_csv", "write_manifest",
]
