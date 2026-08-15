#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chunked pipeline for the liquidity-first latent-liquidity path atlas."""
from __future__ import annotations

from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter
from src.research_common.review_pack import finalize_research_report

from .candidates import build_candidate_frame, normalize_second_bars, select_candidates
from .clustering import assign_path_clusters, fit_path_clusters
from .config import DEFAULT_CONFIG, MODEL_NAME, STAGE_ID, LatentLiquidityPathAtlasConfig
from .features import event_feature_table
from .macro import attach_macro_path_context, build_macro_path_context
from .outcomes import attach_outcomes
from .reports import write_all_reports
from .unswept_swings import (
    attach_unswept_swing_inventory,
    build_unswept_swing_lifecycle,
    load_swing_lifecycle,
    save_swing_lifecycle,
)


@dataclass(frozen=True)
class LatentLiquidityPathAtlasResult:
    decision: str
    report_dir: Path
    feature_rows: int
    label_rows: int


def _chunks(start: pd.Timestamp, end: pd.Timestamp, days: int):
    cursor = start
    delta = pd.Timedelta(days=days)
    while cursor <= end:
        core_end = min(end, cursor + delta - pd.Timedelta(seconds=1))
        yield cursor, core_end
        cursor = core_end + pd.Timedelta(seconds=1)


def _swing_signature(config: LatentLiquidityPathAtlasConfig) -> str:
    payload = {
        "symbol": config.symbol,
        "warmup_start": config.warmup_start,
        "research_end": config.research_end,
        "timeframes": config.swing_timeframes,
        "confirmation_order": config.swing_confirmation_order,
        "sweep_epsilon_bp": config.swing_sweep_epsilon_bp,
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_or_build_swing_lifecycle(
    loader: OKXTradeBarLoader,
    config: LatentLiquidityPathAtlasConfig,
    *,
    build_missing: bool,
    force_rebuild: bool,
) -> pd.DataFrame:
    cache_path = config.swing_cache_path
    meta_path = cache_path.with_suffix(cache_path.suffix + ".json")
    signature = _swing_signature(config)
    if not force_rebuild and cache_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("signature") == signature:
                levels = load_swing_lifecycle(cache_path)
                print(f"[swing-inventory] cache rows={len(levels):,} path={cache_path}", flush=True)
                return levels
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    print(
        f"[swing-inventory] build all confirmed unswept 15m+ levels "
        f"{config.warmup_start} -> {config.research_end}",
        flush=True,
    )
    minute = loader.fetch_data_by_date_range(
        pd.Timestamp(config.warmup_start),
        pd.Timestamp(config.research_end),
        force_rebuild=force_rebuild,
        build_missing=build_missing,
        cvd_mode="range",
    )
    if minute.empty:
        raise RuntimeError("R01.1 cannot build 15m+ unswept Swing inventory: 1m data are empty")
    keep = [
        name
        for name in (
            "open",
            "high",
            "low",
            "close",
            "notional",
            "trades_count",
            "buy_notional",
            "sell_notional",
            "delta_notional",
        )
        if name in minute.columns
    ]
    levels = build_unswept_swing_lifecycle(minute.loc[:, keep], config)
    save_swing_lifecycle(levels, cache_path)
    meta_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "rows": len(levels),
                "created_at_utc": pd.Timestamp.now("UTC").isoformat(),
                "stage": STAGE_ID,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[swing-inventory] built rows={len(levels):,} cache={cache_path}", flush=True)
    return levels


def _compact_chunk_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Consolidate a small chunk and downcast safe numeric columns.

    The full run creates millions of rows and hundreds of features.  Compacting
    each small chunk prevents the final frame from retaining one float64 block
    per feature and cuts numeric memory roughly in half.
    """
    if frame.empty:
        return frame.copy()
    data: dict[str, pd.Series] = {}
    for name in frame.columns:
        values = frame[name]
        if pd.api.types.is_float_dtype(values.dtype):
            data[name] = pd.to_numeric(values, downcast="float")
        elif pd.api.types.is_integer_dtype(values.dtype) and not pd.api.types.is_bool_dtype(values.dtype):
            data[name] = pd.to_numeric(values, downcast="integer")
        else:
            data[name] = values
    return pd.DataFrame(data, index=frame.index).reset_index(drop=True)


def _assign_global_release_episodes(
    features: pd.DataFrame,
    gap_seconds: int,
) -> pd.DataFrame:
    """Assign cross-chunk episodes in place without copying the wide frame.

    Chunk windows are produced chronologically and candidate rows are ordered by
    event time.  Sorting/copying the full 300+ column frame caused a 6.96 GiB
    consolidation allocation on the full dataset.  We therefore verify order and
    operate only on the two narrow key arrays.
    """
    if features.empty:
        return features
    event_times = pd.to_datetime(features["event_time"]).to_numpy(dtype="datetime64[ns]")
    if len(event_times) > 1 and np.any(event_times[1:] < event_times[:-1]):
        raise RuntimeError("global episode assignment requires chronological chunk output")
    sides = features["event_side"].astype(str).to_numpy(copy=False)
    episode_number = np.zeros(len(features), dtype=np.int32)
    episode_ordinal = np.zeros(len(features), dtype=np.int32)
    serial = 0
    last_ns_by_side: dict[str, int] = {}
    active_by_side: dict[str, int] = {}
    ordinal_by_episode: list[int] = [0]
    first_ns: list[int] = [0]
    first_side: list[str] = [""]
    gap_ns = int(pd.Timedelta(seconds=int(gap_seconds)).value)
    time_ns = event_times.astype(np.int64, copy=False)
    for pos, (ts_ns, side) in enumerate(zip(time_ns, sides)):
        previous = last_ns_by_side.get(side)
        if previous is None or int(ts_ns) - previous > gap_ns:
            serial += 1
            active_by_side[side] = serial
            ordinal_by_episode.append(0)
            first_ns.append(int(ts_ns))
            first_side.append(side)
        episode = active_by_side[side]
        ordinal_by_episode[episode] += 1
        episode_number[pos] = episode
        episode_ordinal[pos] = ordinal_by_episode[episode]
        last_ns_by_side[side] = int(ts_ns)
    sizes = np.bincount(episode_number, minlength=serial + 1).astype(np.int32)
    ids = np.empty(serial + 1, dtype=object)
    ids[0] = ""
    for episode in range(1, serial + 1):
        ts = pd.Timestamp(first_ns[episode])
        ids[episode] = f"LLE_{ts:%Y%m%d_%H%M%S}_{first_side[episode]}"
    episode_columns = [
        "release_episode_number",
        "release_episode_id",
        "release_episode_ordinal",
        "release_episode_size",
        "release_episode_weight",
    ]
    base = features.drop(columns=[name for name in episode_columns if name in features], errors="ignore")
    episode_frame = pd.DataFrame(
        {
            "release_episode_number": episode_number,
            "release_episode_id": ids[episode_number],
            "release_episode_ordinal": episode_ordinal,
            "release_episode_size": sizes[episode_number],
            "release_episode_weight": (1.0 / sizes[episode_number]).astype(np.float32),
        },
        index=base.index,
    )
    # axis=1/copy=False attaches five narrow blocks without consolidating/copying
    # the hundreds of existing feature columns.
    return pd.concat([base, episode_frame], axis=1, copy=False)


def _stratified_cap(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cap smoke-test rows across dates/sides instead of truncating earliest data."""
    if cap <= 0 or len(features) <= cap:
        return features, labels
    work = features.copy()
    work["_sample_day"] = pd.to_datetime(work["event_time"]).dt.floor("D")
    groups = list(work.groupby(["_sample_day", "event_side"], sort=True, dropna=False))
    per_group = max(1, cap // max(1, len(groups)))
    positions: list[int] = []
    for _, group in groups:
        take = min(len(group), per_group)
        selected = np.linspace(0, len(group) - 1, take, dtype=int)
        positions.extend(group.iloc[selected].index.tolist())
    if len(positions) < cap:
        remaining = work.index.difference(pd.Index(positions))
        if len(remaining):
            selected = np.linspace(0, len(remaining) - 1, min(cap - len(positions), len(remaining)), dtype=int)
            positions.extend(remaining[selected].tolist())
    keep = work.loc[sorted(set(positions))].head(cap).drop(columns="_sample_day")
    keep_ids = set(keep["event_id"].astype(str))
    kept_labels = labels.loc[labels["event_id"].astype(str).isin(keep_ids)].copy()
    return keep.reset_index(drop=True), kept_labels.reset_index(drop=True)


def _chunk_cache_signature(config: LatentLiquidityPathAtlasConfig) -> str:
    payload = config.to_dict().copy()
    for key in (
        "candidate_cap",
        "cluster_train_sample_cap",
        "cluster_assign_batch_rows",
        "descriptive_sample_cap",
        "csv_write_chunk_rows",
        "report_dir",
    ):
        payload.pop(key, None)
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:20]


def _chunk_cache_path(
    config: LatentLiquidityPathAtlasConfig,
    core_start: pd.Timestamp,
    core_end: pd.Timestamp,
) -> Path:
    root = Path(config.chunk_cache_dir)
    if not root.is_absolute():
        from src.ai_research.config import PROJECT_ROOT

        root = PROJECT_ROOT / root
    name = f"{core_start:%Y%m%d_%H%M%S}_{core_end:%Y%m%d_%H%M%S}.pkl.gz"
    return root / _chunk_cache_signature(config) / name


def _save_chunk_cache(path: Path, features: pd.DataFrame, labels: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.to_pickle(
        {"features": features, "labels": labels},
        temp,
        compression={"method": "gzip", "compresslevel": 1, "mtime": 1},
    )
    temp.replace(path)


def _load_chunk_cache(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = pd.read_pickle(path, compression="gzip")
    if not isinstance(payload, dict) or "features" not in payload or "labels" not in payload:
        raise ValueError(f"invalid latent-liquidity chunk cache: {path}")
    features = payload["features"]
    labels = payload["labels"]
    if not isinstance(features, pd.DataFrame) or not isinstance(labels, pd.DataFrame):
        raise ValueError(f"invalid latent-liquidity chunk cache frames: {path}")
    return features, labels


def run_latent_liquidity_path_atlas(
    *,
    data_dir: str | Path | None = None,
    db_name: str = "okx_trade_bars.db",
    build_missing: bool = False,
    force_rebuild: bool = False,
    progress: bool = True,
    skip_review_pack: bool = False,
    use_chunk_cache: bool = True,
    config: LatentLiquidityPathAtlasConfig = DEFAULT_CONFIG,
) -> LatentLiquidityPathAtlasResult:
    config.validate()
    start = pd.Timestamp(config.research_start)
    end = pd.Timestamp(config.research_end)
    loader = OKXTradeBarLoader(
        symbol=config.symbol,
        timeframe=config.micro_timeframe,
        data_dir=data_dir,
        db_name=db_name,
    )
    macro_loader = OKXTradeBarLoader(
        symbol=config.symbol,
        timeframe="1m",
        data_dir=data_dir,
        db_name=db_name,
    )
    swing_levels = _load_or_build_swing_lifecycle(
        macro_loader,
        config,
        build_missing=build_missing,
        force_rebuild=force_rebuild,
    )
    chunk_windows = list(_chunks(start, end, config.chunk_days))
    reporter = ProgressReporter(
        label="[latent-liquidity-atlas] chunks",
        total=len(chunk_windows),
        every=1,
        enabled=progress,
    )
    feature_parts: list[pd.DataFrame] = []
    label_parts: list[pd.DataFrame] = []
    print(f"[run] {MODEL_NAME} {STAGE_ID}", flush=True)
    print(f"[window] {start} -> {end} micro={config.micro_timeframe} chunks={len(chunk_windows)}", flush=True)
    print(
        "[design] liquidity-first broad union; 15m+ all-unswept Swing inventory is supplementary only",
        flush=True,
    )

    for idx, (core_start, core_end) in enumerate(chunk_windows, start=1):
        cache_path = _chunk_cache_path(config, core_start, core_end)
        if use_chunk_cache and not force_rebuild and cache_path.exists():
            try:
                cached_features, cached_labels = _load_chunk_cache(cache_path)
                if not cached_features.empty:
                    feature_parts.append(cached_features)
                if not cached_labels.empty:
                    label_parts.append(cached_labels)
                print(
                    f"[chunk-cache] {core_start.date()}->{core_end.date()} "
                    f"features={len(cached_features):,} labels={len(cached_labels):,}",
                    flush=True,
                )
                reporter.update(idx)
                continue
            except (OSError, ValueError, EOFError):
                cache_path.unlink(missing_ok=True)
        load_start = core_start - pd.Timedelta(seconds=config.pre_context_seconds + config.baseline_seconds)
        load_end = core_end + pd.Timedelta(seconds=config.post_label_seconds)
        bars = loader.fetch_data_by_date_range(
            load_start,
            load_end,
            force_rebuild=force_rebuild,
            build_missing=build_missing,
            cvd_mode="range",
        )
        if bars.empty:
            print(f"[chunk] no data {core_start}->{core_end}", flush=True)
            reporter.update(idx)
            continue
        bars = normalize_second_bars(bars, config)
        if bars.empty:
            reporter.update(idx)
            continue
        candidate_frame = build_candidate_frame(bars, config)
        events = select_candidates(candidate_frame, core_start, core_end, config)
        if events.empty:
            reporter.update(idx)
            continue
        features = event_feature_table(candidate_frame, events, config)
        macro_start = core_start - pd.Timedelta(minutes=config.macro_context_minutes)
        macro_bars = macro_loader.fetch_data_by_date_range(
            macro_start,
            core_end,
            force_rebuild=force_rebuild,
            build_missing=build_missing,
            cvd_mode="range",
        )
        macro_context = build_macro_path_context(macro_bars, config)
        features = attach_macro_path_context(features, macro_context, config)
        features = attach_unswept_swing_inventory(features, swing_levels, config)
        labels = attach_outcomes(candidate_frame, features[["event_id", "event_time", "event_side"]], config)
        episodes = features.get("release_episode_id", pd.Series(dtype=str)).nunique()
        if not features.empty and not labels.empty:
            label_ids = pd.Index(labels["event_id"].astype(str))
            complete_mask = features["event_id"].astype(str).isin(label_ids)
            features = features.loc[complete_mask].reset_index(drop=True)
            labels = labels.loc[labels["event_id"].astype(str).isin(pd.Index(features["event_id"].astype(str)))].reset_index(drop=True)
            # Chunk-local episode metadata is replaced globally after concatenation.
            local_episode_columns = [
                "release_episode_id",
                "release_episode_number",
                "release_episode_ordinal",
                "release_episode_size",
                "release_episode_weight",
            ]
            features = features.drop(columns=[c for c in local_episode_columns if c in features], errors="ignore")
            compact_features = _compact_chunk_frame(features)
            compact_labels = _compact_chunk_frame(labels)
            feature_parts.append(compact_features)
            label_parts.append(compact_labels)
        else:
            compact_features = pd.DataFrame()
            compact_labels = pd.DataFrame()
        if use_chunk_cache:
            _save_chunk_cache(cache_path, compact_features, compact_labels)
        print(
            f"[chunk] {core_start.date()}->{core_end.date()} bars={len(bars):,} "
            f"events={len(events):,} episodes={episodes:,} labels={len(labels):,}",
            flush=True,
        )
        reporter.update(idx)
    reporter.close()

    print("[stage] assemble compact full-history feature/label frames", flush=True)
    features = pd.concat(feature_parts, ignore_index=True, copy=False) if feature_parts else pd.DataFrame()
    labels = pd.concat(label_parts, ignore_index=True, copy=False) if label_parts else pd.DataFrame()
    feature_parts.clear()
    label_parts.clear()
    gc.collect()
    if not features.empty:
        if features["event_id"].duplicated().any():
            raise RuntimeError("duplicate feature event_id across non-overlapping chunks")
        if labels["event_id"].duplicated().any():
            raise RuntimeError("duplicate label event_id across non-overlapping chunks")
        if len(features) != len(labels) or not np.array_equal(
            features["event_id"].astype(str).to_numpy(),
            labels["event_id"].astype(str).to_numpy(),
        ):
            raise RuntimeError("chunk-level complete feature/label alignment was lost")
        print(
            f"[assemble] rows={len(features):,} cols={len(features.columns):,} "
            f"feature_memory_gib={features.memory_usage(deep=True).sum() / (1024 ** 3):.2f}",
            flush=True,
        )
        print("[stage] assign cross-chunk liquidity-release episodes", flush=True)
        features = _assign_global_release_episodes(features, config.release_episode_gap_seconds)

    features, labels = _stratified_cap(features, labels, config.candidate_cap)
    if config.candidate_cap > 0 and not features.empty:
        features = _assign_global_release_episodes(features, config.release_episode_gap_seconds)
    print(
        f"[stage] fit frozen path clusters cap={config.cluster_train_sample_cap:,}",
        flush=True,
    )
    cluster_model = fit_path_clusters(features, config)
    if cluster_model is not None:
        print(
            f"[cluster-fit] eligible={cluster_model.eligible_train_rows:,} "
            f"fit_rows={cluster_model.train_rows:,} features={len(cluster_model.columns):,}",
            flush=True,
        )
    assignments = assign_path_clusters(
        features,
        cluster_model,
        batch_rows=config.cluster_assign_batch_rows,
        progress=progress,
    )
    print("[stage] build summaries and stream full report tables", flush=True)
    reports = write_all_reports(
        config.report_path,
        config,
        features,
        labels,
        assignments,
        swing_levels,
        0 if cluster_model is None else cluster_model.train_rows,
        0 if cluster_model is None else cluster_model.eligible_train_rows,
    )
    failures: list[str] = []
    for name in ("01_data_quality.csv", "09_causal_audit.csv"):
        frame = reports[name]
        if "status" in frame:
            failures.extend(frame.loc[frame["status"].eq("FAIL"), "check"].astype(str).tolist())
    if failures:
        raise RuntimeError(f"R01.1 quality/causal gate failed: {failures}")
    if not skip_review_pack:
        finalize_research_report(
            config.report_path,
            experiment_id="ETH_LATENT_LIQUIDITY_PATH_R01_1",
            edge_id="RESEARCH_ONLY_LATENT_LIQUIDITY_PATH",
            title=f"{MODEL_NAME} {STAGE_ID}",
        )
    print(
        f"[done] report={config.report_path} features={len(features):,} labels={len(labels):,} "
        f"clusters={'READY' if cluster_model is not None else 'INSUFFICIENT_SAMPLE'}",
        flush=True,
    )
    return LatentLiquidityPathAtlasResult(
        decision="COMPLETE_LIQUIDITY_FIRST_DISCOVERY_ATLAS_NO_TRADING_CLAIM",
        report_dir=config.report_path,
        feature_rows=len(features),
        label_rows=len(labels),
    )
