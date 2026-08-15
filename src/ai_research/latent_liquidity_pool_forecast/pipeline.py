#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02 pipeline: pre-event price-zone liquidity and sweep-depth prediction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.config import PROJECT_ROOT
from src.ai_research.latent_liquidity_path_atlas.candidates import normalize_second_bars
from src.ai_research.latent_liquidity_path_atlas.config import DEFAULT_CONFIG as ATLAS_CONFIG
from src.ai_research.latent_liquidity_path_atlas.unswept_swings import load_swing_lifecycle
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter

from .cache import chunk_cache_path, dataset_cache_path, episode_cache_path, load_frame, save_frame
from .config import DEFAULT_CONFIG, MODEL_NAME, STAGE_ID, LatentLiquidityPoolForecastConfig
from .labels import attach_release_labels, attach_touch_labels, deterministic_control_sample
from .modeling import feature_importance, fit_models, metric_table, predict
from .reports import calibration_thresholds, causal_audit, score_deciles, top_zone_summary, write_reports
from .source import load_episode_table, source_gate_only
from .spatial import attach_swing_spatial_features, build_snapshot_context, expand_zone_lattice


@dataclass(frozen=True)
class LatentLiquidityPoolForecastResult:
    decision: str
    report_dir: Path
    spatial_rows: int
    snapshots: int


def _chunks(start: pd.Timestamp, end: pd.Timestamp, days: int):
    current = start
    while current <= end:
        stop = min(end, current + pd.Timedelta(days=days) - pd.Timedelta(seconds=1))
        yield current, stop
        current = stop + pd.Timedelta(seconds=1)


def _load_swing_lifecycle(config: LatentLiquidityPoolForecastConfig) -> pd.DataFrame:
    # R01.1 lifecycle is built to the same research end and deliberately keeps all 15m+ unswept levels.
    symbol = config.symbol.replace("-", "_")
    end = pd.Timestamp(config.research_end).strftime("%Y%m%d")
    path = PROJECT_ROOT / ATLAS_CONFIG.swing_cache_dir / f"{symbol}_unswept_swing_lifecycle_to_{end}.csv.gz"
    if not path.exists():
        return pd.DataFrame()
    return load_swing_lifecycle(path)


def _build_dataset(config: LatentLiquidityPoolForecastConfig, episodes: pd.DataFrame, *, data_dir: str | Path | None, db_name: str, progress: bool, use_cache: bool) -> pd.DataFrame:
    start = pd.Timestamp(config.research_start)
    requested_end = pd.Timestamp(config.research_end)
    # Every candidate requires the full frozen primary future horizon.  The tail
    # of the requested research window is label-only support, never a negative
    # training region.
    end = requested_end - pd.Timedelta(minutes=config.primary_horizon_minutes)
    if end < start:
        raise RuntimeError("R02 research window is shorter than the primary future-label horizon")
    minute_loader = OKXTradeBarLoader(symbol=config.symbol, timeframe="1m", data_dir=data_dir, db_name=db_name)
    second_loader = OKXTradeBarLoader(symbol=config.symbol, timeframe="1s", data_dir=data_dir, db_name=db_name)
    lifecycle = _load_swing_lifecycle(config)
    if lifecycle.empty:
        raise RuntimeError(
            "R02 requires the R01.1 all-unswept 15m+ Swing lifecycle cache for the explicit Swing ablation; "
            "the Swing family remains supplemental and is never an admission gate."
        )
    print(f"[swing-supplement] active-lifecycle-source rows={len(lifecycle):,} (15m+ only; all ages retained until sweep)", flush=True)
    windows = list(_chunks(start, end, config.chunk_days))
    reporter = ProgressReporter("[latent-liquidity-r02] spatial chunks", len(windows), every=1, enabled=progress)
    parts: list[pd.DataFrame] = []
    for i, (core_start, core_end) in enumerate(windows, 1):
        cpath = chunk_cache_path(config, core_start, core_end)
        if use_cache and cpath.exists():
            try:
                cached = load_frame(cpath)
                parts.append(cached)
                print(f"[r02-chunk-cache] {core_start.date()}->{core_end.date()} rows={len(cached):,}", flush=True)
                reporter.update(i)
                continue
            except (OSError, ValueError, EOFError):
                cpath.unlink(missing_ok=True)
        minute_start = core_start - pd.Timedelta(minutes=config.macro_context_minutes + 5)
        minute_end = core_end + pd.Timedelta(minutes=config.primary_horizon_minutes + 5)
        second_start = core_start - pd.Timedelta(seconds=config.micro_context_seconds + 5)
        second_end = core_end
        minute = minute_loader.fetch_data_by_date_range(minute_start, minute_end, build_missing=False, cvd_mode="range")
        second = second_loader.fetch_data_by_date_range(second_start, second_end, build_missing=False, cvd_mode="range")
        if minute.empty or second.empty:
            print(f"[r02-chunk] no local data {core_start}->{core_end}", flush=True)
            reporter.update(i); continue
        second = normalize_second_bars(second, ATLAS_CONFIG)
        snapshots = build_snapshot_context(minute, second, core_start, core_end, config)
        if snapshots.empty:
            reporter.update(i); continue
        zones = expand_zone_lattice(snapshots, config)
        zones = attach_swing_spatial_features(zones, lifecycle, config)
        zones = attach_touch_labels(zones, minute, config)
        incomplete = ~zones["primary_touch_label_complete"].astype(bool)
        if incomplete.any():
            zones = zones.loc[~incomplete].reset_index(drop=True)
        if zones.empty:
            reporter.update(i)
            continue
        zones = attach_release_labels(zones, episodes, config)
        zones = deterministic_control_sample(zones, config)
        # Downcast to keep full-history memory bounded.
        for name in zones.columns:
            if pd.api.types.is_float_dtype(zones[name].dtype):
                zones[name] = pd.to_numeric(zones[name], downcast="float")
            elif pd.api.types.is_integer_dtype(zones[name].dtype) and not pd.api.types.is_bool_dtype(zones[name].dtype):
                zones[name] = pd.to_numeric(zones[name], downcast="integer")
        # Drop raw macro notional now that zone buildup proxies have been constructed.
        zones = zones.drop(columns=[f"macro_notional_{w}m" for w in config.macro_windows_minutes if f"macro_notional_{w}m" in zones], errors="ignore")
        if use_cache:
            save_frame(cpath, zones)
        parts.append(zones)
        print(f"[r02-chunk] {core_start.date()}->{core_end.date()} snapshots={len(snapshots):,} sampled_zones={len(zones):,} releases={int(zones['release_within_horizon'].sum()):,}", flush=True)
        reporter.update(i)
    reporter.close()
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True, copy=False).sort_values(["decision_time", "zone_side", "zone_distance_bp"], kind="mergesort").reset_index(drop=True)


def _eval_cap(frame: pd.DataFrame, config: LatentLiquidityPoolForecastConfig) -> pd.DataFrame:
    if "model_sample_keep" in frame:
        frame = frame.loc[frame["model_sample_keep"].astype(bool)].copy()
    parts = []
    for period, group in frame.groupby("period", sort=True):
        cap = config.model_train_cap_rows if period == config.train_period else config.model_eval_cap_rows_per_period
        if len(group) <= cap:
            parts.append(group); continue
        # Always retain release rows, deterministic hash sample the rest.
        pos = group.loc[group["release_within_horizon"].astype(bool)]
        neg = group.loc[~group["release_within_horizon"].astype(bool)]
        remaining = max(0, cap - len(pos))
        if remaining and len(neg) > remaining:
            h = pd.util.hash_pandas_object(neg["zone_id"].astype(str), index=False).to_numpy(dtype=np.uint64)
            take = np.argpartition(h, remaining - 1)[:remaining]
            neg = neg.iloc[np.sort(take)]
        parts.append(pd.concat([pos, neg.head(remaining)], ignore_index=True, copy=False))
    return pd.concat(parts, ignore_index=True, copy=False)


def run_latent_liquidity_pool_forecast(*, data_dir: str | Path | None = None, db_name: str = "okx_trade_bars.db", progress: bool = True, skip_review_pack: bool = False, use_cache: bool = True, config: LatentLiquidityPoolForecastConfig = DEFAULT_CONFIG) -> LatentLiquidityPoolForecastResult:
    config.validate()
    print(f"[run] {MODEL_NAME} {STAGE_ID}", flush=True)
    print("[design] pre-event time-price liquidity pool forecast; Swing is 15m+ supplemental only", flush=True)
    ep_path = episode_cache_path(config)
    source_gate = pd.DataFrame(); source_rows = 0
    if use_cache and ep_path.exists():
        episodes = load_frame(ep_path)
        source_gate, source_rows = source_gate_only(config)
        failures = source_gate.loc[source_gate["status"].astype(str).eq("FAIL"), "check"].tolist()
        if failures:
            raise RuntimeError(f"R02 source gate failed: {failures}")
        print(f"[episode-cache] rows={len(episodes):,}", flush=True)
    else:
        episodes, source_gate, source_rows = load_episode_table(config, progress=progress)
        if use_cache: save_frame(ep_path, episodes)
    if episodes.empty:
        raise RuntimeError("R02 found no R01.1 Episode labels")
    data_path = dataset_cache_path(config)
    if use_cache and data_path.exists():
        spatial = load_frame(data_path)
        print(f"[dataset-cache] rows={len(spatial):,}", flush=True)
    else:
        print("[stage] build causal 15m decision x price-zone lattice from local 1m/1s data", flush=True)
        spatial = _build_dataset(config, episodes, data_dir=data_dir, db_name=db_name, progress=progress, use_cache=use_cache)
        if use_cache and not spatial.empty: save_frame(data_path, spatial)
    if spatial.empty:
        raise RuntimeError("R02 produced no spatial rows")
    required = set(config.periods); observed = set(spatial["period"].astype(str).unique())
    if not required <= observed:
        raise RuntimeError(f"R02 missing frozen periods: {sorted(required - observed)}")
    modeling = _eval_cap(spatial, config)
    print(f"[dataset] full_sampled={len(spatial):,} modeling={len(modeling):,} snapshots={spatial['decision_time'].nunique():,} releases={int(spatial['release_within_horizon'].sum()):,}", flush=True)
    print("[stage] fit distance baseline vs liquidity-path-no-Swing vs full-with-15m+-Swing", flush=True)
    models = fit_models(modeling, config)
    pred = predict(modeling, models)
    metrics = metric_table(pred, config)
    importance = feature_importance(models)
    audit_lattice = spatial.loc[spatial["full_lattice_audit_group"].astype(bool)].copy()
    if audit_lattice.empty:
        raise RuntimeError("R02 full-lattice audit sample is empty")
    audit_pred = predict(audit_lattice, models)
    # Calibration thresholds and location-quality summaries must use complete
    # price lattices, not the inverse-probability model-control sample.
    deciles = score_deciles(audit_pred, config)
    thresholds = calibration_thresholds(audit_pred, config)
    top = top_zone_summary(audit_pred, thresholds, config)
    causal = causal_audit(pred, models.full_columns, source_gate, config, audit_frame=audit_pred)
    print(f"[audit-lattice] rows={len(audit_pred):,} complete_groups={audit_pred.groupby(['decision_time','zone_side']).ngroups:,}", flush=True)
    print("[stage] write compact R02 report", flush=True)
    report_dir, decision = write_reports(config=config, source_gate=source_gate, frame=pred, audit_frame=audit_pred, spatial_rows=len(spatial), spatial_snapshots=spatial["decision_time"].nunique(), metrics=metrics, deciles=deciles, importance=importance, thresholds=thresholds, top=top, causal=causal, source_rows_scanned=source_rows, feature_columns=models.full_columns, skip_review_pack=skip_review_pack)
    print(f"[decision] {decision}", flush=True); print(f"[done] report={report_dir}", flush=True)
    return LatentLiquidityPoolForecastResult(decision, report_dir, len(spatial), spatial["decision_time"].nunique())
