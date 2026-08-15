#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02.1 pipeline: arrival-independent conditional pool-strength learning."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_pool_forecast.cache import (
    dataset_cache_path as r02_dataset_cache_path,
    episode_cache_path as r02_episode_cache_path,
    load_frame as load_r02_frame,
    save_frame as save_r02_frame,
)
from src.ai_research.latent_liquidity_pool_forecast.config import DEFAULT_CONFIG as R02_CONFIG
from src.ai_research.latent_liquidity_pool_forecast.source import load_episode_table, source_gate_only

from .cache import dataset_cache_path, load_frame, save_frame
from .config import DEFAULT_CONFIG, MODEL_NAME, STAGE_ID, LatentLiquidityPoolStrengthConfig
from .labels import attach_strength_labels, attach_train_frozen_strength_thresholds
from .modeling import feature_importance, fit_models, metric_table, predict
from .reports import causal_audit, write_reports


@dataclass(frozen=True)
class LatentLiquidityPoolStrengthResult:
    decision: str
    report_dir: Path
    rows: int


def _load_r02_inputs(*, use_cache: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    spatial_path = r02_dataset_cache_path(R02_CONFIG)
    episode_path = r02_episode_cache_path(R02_CONFIG)
    if not spatial_path.exists():
        raise RuntimeError(
            "R02.1 requires the completed R02 spatial cache. Run once: "
            "python research\\eth_ai_trading\\eth_latent_liquidity_path_v1\\02_latent_pool_location_depth_model.py"
        )
    spatial = load_r02_frame(spatial_path)
    source_gate, _ = source_gate_only(R02_CONFIG)
    failures = source_gate.loc[source_gate["status"].astype(str).eq("FAIL"), "check"].tolist()
    if failures:
        raise RuntimeError(f"R02.1 source gate failed: {failures}")
    if episode_path.exists():
        episodes = load_r02_frame(episode_path)
    else:
        episodes, gate, _ = load_episode_table(R02_CONFIG, progress=True)
        failures = gate.loc[gate["status"].astype(str).eq("FAIL"), "check"].tolist()
        if failures:
            raise RuntimeError(f"R02.1 episode source gate failed: {failures}")
        if use_cache:
            save_r02_frame(episode_path, episodes)
    return spatial, episodes, source_gate


def _build_strength_dataset(spatial: pd.DataFrame, episodes: pd.DataFrame, config: LatentLiquidityPoolStrengthConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = spatial.copy(deep=False)
    # Remove old R02 future-release labels/predictions; all pre-event features and
    # completed-minute Touch labels remain untouched.
    drop_prefixes = (
        "release_", "favorable_", "continuation_", "time_to_release_", "sweep_depth_",
        "reversal_after_", "time_to_extreme_", "p_touch", "p_release_", "p_favorable_",
        "pred_sweep_", "pred_reversal_", "pred_room_", "pool_score",
    )
    drop = [name for name in frame.columns if name.startswith(drop_prefixes)]
    frame = frame.drop(columns=drop, errors="ignore")
    frame = attach_strength_labels(
        frame, episodes,
        horizon_minutes=config.primary_horizon_minutes,
        zone_offsets_bp=tuple(float(x) for x in R02_CONFIG.zone_offsets_bp),
        zone_half_width_bp=float(R02_CONFIG.zone_half_width_bp),
        decision_chunk_size=config.aggregation_decision_chunk_size,
        progress=True,
    )
    frame, thresholds = attach_train_frozen_strength_thresholds(
        frame, train_period=config.train_period, quantile=config.strength_quantile,
    )
    for name in frame.columns:
        if pd.api.types.is_float_dtype(frame[name].dtype):
            frame[name] = pd.to_numeric(frame[name], downcast="float")
        elif pd.api.types.is_integer_dtype(frame[name].dtype) and not pd.api.types.is_bool_dtype(frame[name].dtype):
            frame[name] = pd.to_numeric(frame[name], downcast="integer")
    return frame, thresholds


def _eval_cap(frame: pd.DataFrame, config: LatentLiquidityPoolStrengthConfig) -> pd.DataFrame:
    model = frame.loc[frame.get("model_sample_keep", pd.Series(True, index=frame.index)).astype(bool)].copy()
    parts = []
    for period, group in model.groupby("period", sort=True):
        cap = config.model_train_cap_rows if period == config.train_period else config.model_eval_cap_rows_per_period
        touched = group.loc[group["touch_720m"].astype(bool)]
        # R02.1 is conditional-on-touch; untouched rows never enter model metrics.
        if len(touched) <= cap:
            parts.append(touched); continue
        h = pd.util.hash_pandas_object(touched["zone_id"].astype(str), index=False).to_numpy(dtype=np.uint64)
        idx = np.argpartition(h, cap - 1)[:cap]
        parts.append(touched.iloc[np.sort(idx)])
    return pd.concat(parts, ignore_index=True, copy=False) if parts else pd.DataFrame()


def run_latent_liquidity_pool_strength(*, skip_review_pack: bool = False, use_cache: bool = True, config: LatentLiquidityPoolStrengthConfig = DEFAULT_CONFIG) -> LatentLiquidityPoolStrengthResult:
    config.validate()
    print(f"[run] {MODEL_NAME} {STAGE_ID}", flush=True)
    print("[design] conditional pool strength != arrival probability; primary score excludes Touch and Swing", flush=True)
    spatial, episodes, source_gate = _load_r02_inputs(use_cache=use_cache)
    print(f"[source] R02 spatial rows={len(spatial):,} Episodes={len(episodes):,}", flush=True)
    cpath = dataset_cache_path(config)
    if use_cache and cpath.exists():
        frame = load_frame(cpath)
        # Thresholds are deterministic from the cached train data.
        _, thresholds_train = attach_train_frozen_strength_thresholds(frame, train_period=config.train_period, quantile=config.strength_quantile)
        print(f"[strength-cache] rows={len(frame):,}", flush=True)
    else:
        print("[stage] aggregate all future release Episodes into touched-zone strength labels", flush=True)
        frame, thresholds_train = _build_strength_dataset(spatial, episodes, config)
        if use_cache:
            save_frame(cpath, frame)
    if frame.empty:
        raise RuntimeError("R02.1 produced no strength rows")
    required = set(config.periods); observed = set(frame["period"].astype(str).unique())
    if not required <= observed:
        raise RuntimeError(f"R02.1 missing frozen periods: {sorted(required - observed)}")
    modeling = _eval_cap(frame, config)
    if modeling.empty:
        raise RuntimeError("R02.1 conditional touched-zone modeling sample is empty")
    print(f"[dataset] rows={len(frame):,} touched_modeling={len(modeling):,} releases={int(frame['release_episode_count'].gt(0).sum()):,}", flush=True)
    print("[stage] fit strict distance baseline vs path-no-Swing PRIMARY vs full-with-Swing ablation", flush=True)
    models = fit_models(modeling, config)
    pred = predict(modeling, models)
    metrics = metric_table(pred, config)
    importance = feature_importance(models)
    audit = frame.loc[frame["full_lattice_audit_group"].astype(bool)].copy()
    if audit.empty:
        raise RuntimeError("R02.1 full-lattice audit sample is empty")
    audit_pred = predict(audit, models)
    causal = causal_audit(frame, audit_pred, models.path_columns, source_gate, config)
    print(f"[audit-lattice] rows={len(audit_pred):,} groups={audit_pred.groupby(['decision_time','zone_side']).ngroups:,}", flush=True)
    print("[stage] write compact R02.1 report", flush=True)
    report_dir, decision = write_reports(
        config=config, source_gate=source_gate, frame=pred, audit=audit_pred,
        thresholds_train=thresholds_train, metrics=metrics, importance=importance,
        causal=causal, skip_review_pack=skip_review_pack,
    )
    print(f"[decision] {decision}", flush=True)
    print(f"[done] report={report_dir}", flush=True)
    return LatentLiquidityPoolStrengthResult(decision=decision, report_dir=report_dir, rows=len(frame))
