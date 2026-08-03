#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal pre-sweep liquidity-pool approach universe for R11."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter
from src.research_common.structured_stop_pool import (
    FAMILY_COLUMNS,
    StructuredStopPoolConfig,
    build_level_structure_features,
    load_or_build_r02,
)
from src.research_common.swing_liquidity_atlas.lifecycle import attach_active_confluence
from src.research_common.swing_liquidity_atlas.config import AtlasConfig
from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

from .config import LiquidityMagnetConfig


_TIME_COLUMNS = (
    "pivot_time",
    "pivot_bar_end_time",
    "initial_available_time",
    "order_1_available_time",
    "order_2_available_time",
    "order_3_available_time",
    "order_5_available_time",
    "active_bar_time",
    "approach_available_time",
    "touch_available_time",
    "sweep_available_time",
)


def _read_csv(path: Path, *, compression: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, compression=compression, low_memory=False)
    for name in _TIME_COLUMNS:
        if name in frame.columns:
            frame[name] = pd.to_datetime(frame[name], errors="coerce")
    return frame


def load_r02_and_r09_levels(
    *,
    r02_dir: Path,
    r09_dir: Path,
    primary: pd.DataFrame,
    rebuild_r02_if_missing: bool,
    rebuild_r09_if_missing: bool,
    show_progress: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Load causal R02 lifecycle and R09 formation-time features.

    The R09 cache is optional.  If absent, features are rebuilt from the R02
    level table and current primary bars without importing a research script.
    """

    r09_cfg = StructuredStopPoolConfig().validate()
    levels, lifecycle, r02_source = load_or_build_r02(
        r02_dir,
        primary,
        r09_cfg,
        rebuild_if_missing=bool(rebuild_r02_if_missing),
        show_progress=bool(show_progress),
    )
    feature_path = r09_dir / "18_level_structure_feature_table.csv.gz"
    if feature_path.exists():
        features = _read_csv(feature_path, compression="gzip")
        r09_source = "R09_REPORT_CACHE"
    else:
        if not rebuild_r09_if_missing:
            raise FileNotFoundError(
                f"R09 level feature cache missing: {feature_path}. "
                "Run R09 first or pass --rebuild-r09-if-missing."
            )
        features, _ = build_level_structure_features(levels, primary, r09_cfg)
        r09_source = "R09_REBUILT_COMMON_MODULE"
    if features.empty:
        raise RuntimeError("R09 level structure feature table is empty")
    if features["level_id"].duplicated().any():
        raise RuntimeError("duplicate level_id in R09 level features")
    return levels, lifecycle, features, r02_source, r09_source


def _period(ts: pd.Series) -> pd.Series:
    value = pd.to_datetime(ts, errors="coerce")
    return pd.Series(
        np.select(
            [value < pd.Timestamp("2025-01-01"), value < pd.Timestamp("2025-10-01")],
            ["EARLY_2023_2024", "MID_2025Q1_Q3"],
            default="LATE_2025Q4_2026H1",
        ),
        index=ts.index,
        dtype="object",
    )


def _first_band_crossings(
    lifecycle: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: LiquidityMagnetConfig,
    *,
    research_start: pd.Timestamp,
    research_end_exclusive: pd.Timestamp,
    show_progress: bool,
) -> pd.DataFrame:
    bars = normalize_primary_bars(bars)
    index = pd.DatetimeIndex(bars.index)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    low_index = SegmentThresholdIndex(low)
    n = len(bars)
    rows: list[dict[str, object]] = []
    reporter = ProgressReporter(
        label="[r11] first distance-band crossings",
        total=len(lifecycle),
        every=max(1, len(lifecycle) // 200),
        enabled=bool(show_progress),
    )
    bands = tuple(sorted((float(v) for v in cfg.distance_bands_bp), reverse=True))
    keep = [
        "level_id", "source_timeframe", "source_timeframe_min", "level_price",
        "pivot_time", "initial_available_time", "active_pos", "sweep_pos",
        "confirmed_order_at_sweep", "confirmation_reaction_close_bp",
        "confirmation_reaction_high_bp", "left_high_range_20_bp",
        "left_low_gap_20_bp", "pivot_notional_vs_past20",
        "pivot_trades_count_vs_past20",
    ]
    for ordinal, row in enumerate(lifecycle.itertuples(index=False), start=1):
        source = row._asdict()
        active_pos = int(source.get("active_pos", -1))
        sweep_raw = source.get("sweep_pos", -1)
        sweep_pos = int(sweep_raw) if pd.notna(sweep_raw) else -1
        if active_pos < 0 or active_pos >= n - 1:
            reporter.update(ordinal)
            continue
        end_pos = min(n - 2, sweep_pos - 1 if sweep_pos >= 0 else n - 2)
        if end_pos < active_pos:
            reporter.update(ordinal)
            continue
        level_price = float(source["level_price"])
        for band in bands:
            threshold = level_price * (1.0 + band / 10_000.0)
            pos = low_index.first_leq(active_pos, end_pos, threshold)
            if pos < 0 or pos + 1 >= n:
                continue
            available_time = pd.Timestamp(index[pos] + pd.Timedelta(minutes=1))
            entry_time = pd.Timestamp(index[pos + 1])
            # A missing 1m bar leaves the path between signal and observed next
            # open unknown.  Do not reinterpret a later bar as the strict next
            # executable minute; discard the candidate instead.
            if entry_time != available_time:
                continue
            if available_time < research_start or available_time >= research_end_exclusive:
                continue
            entry_price = float(bars["open"].iloc[pos + 1])
            if not np.isfinite(entry_price) or entry_price <= level_price:
                continue
            record = {name: source.get(name, np.nan) for name in keep}
            record.update(
                {
                    "distance_band_bp": float(band),
                    "event_pos": int(pos),
                    "event_bar_time": pd.Timestamp(index[pos]),
                    "event_available_time": available_time,
                    "entry_pos": int(pos + 1),
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "signal_close": float(close[pos]),
                    "distance_close_to_level_bp": (float(close[pos]) / level_price - 1.0) * 10_000.0,
                    "distance_entry_to_level_bp": (entry_price / level_price - 1.0) * 10_000.0,
                    "level_age_minutes_at_signal": int(pos - active_pos),
                    # Labels kept separate by the entrypoint.
                    "future_sweep_pos": int(sweep_pos),
                    "future_sweep_available_time": source.get("sweep_available_time", pd.NaT),
                }
            )
            rows.append(record)
        reporter.update(ordinal)
    reporter.close()
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["event_pos", "distance_band_bp", "level_price", "level_id"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _group_same_bar_pools(frame: pd.DataFrame, cfg: LiquidityMagnetConfig) -> pd.DataFrame:
    """Merge same-band, same-closed-bar levels whose prices are within 10bp."""

    if frame.empty:
        return frame.copy()
    tolerance = float(cfg.zone_merge_tolerance_bp) / 10_000.0
    rows: list[dict[str, object]] = []
    group_id = 0
    for (event_pos, band), part in frame.groupby(["event_pos", "distance_band_bp"], sort=True):
        part = part.sort_values(["level_price", "level_id"], ascending=[False, True], kind="mergesort")
        current: list[dict[str, object]] = []
        last_price = np.nan

        def emit(items: list[dict[str, object]]) -> None:
            nonlocal group_id
            if not items:
                return
            group_id += 1
            member = pd.DataFrame(items)
            prices = pd.to_numeric(member["level_price"], errors="coerce")
            ceiling = float(prices.max())
            floor = float(prices.min())
            anchor = member.sort_values(
                ["source_timeframe_min", "confirmation_reaction_high_bp", "level_id"],
                ascending=[False, False, True],
                kind="mergesort",
            ).iloc[0].to_dict()
            record = dict(anchor)
            record.update(
                {
                    "pool_event_id": f"LM_{int(event_pos):08d}_{int(round(float(band))):03d}_{group_id:06d}",
                    "pool_floor": floor,
                    "pool_ceiling": ceiling,
                    "pool_center": float(np.sqrt(max(floor, 1e-12) * max(ceiling, 1e-12))),
                    "pool_width_bp": (ceiling / floor - 1.0) * 10_000.0 if floor > 0 else np.nan,
                    "pool_member_count": int(len(member)),
                    "pool_timeframe_count": int(member["source_timeframe"].astype(str).nunique()),
                    "pool_max_timeframe_min": int(pd.to_numeric(member["source_timeframe_min"], errors="coerce").max()),
                    "pool_member_level_ids": "|".join(str(int(v)) for v in sorted(member["level_id"].astype(int).unique())),
                    "pool_timeframes": "|".join(sorted(member["source_timeframe"].astype(str).unique())),
                    "pool_reaction_high_bp_max": float(pd.to_numeric(member["confirmation_reaction_high_bp"], errors="coerce").max()),
                    "pool_left_prominence_bp_max": float(pd.to_numeric(member["left_high_range_20_bp"], errors="coerce").max()),
                    "pool_age_minutes_min": float(pd.to_numeric(member["level_age_minutes_at_signal"], errors="coerce").min()),
                    "pool_age_minutes_max": float(pd.to_numeric(member["level_age_minutes_at_signal"], errors="coerce").max()),
                    "pool_member_initial_available_time_max": pd.to_datetime(member["initial_available_time"], errors="coerce").max(),
                    "future_any_member_swept": bool(pd.to_numeric(member["future_sweep_pos"], errors="coerce").ge(0).any()),
                }
            )
            rows.append(record)

        for item in part.to_dict("records"):
            price = float(item["level_price"])
            if current and np.isfinite(last_price) and (last_price / price - 1.0) > tolerance:
                emit(current)
                current = []
            current.append(item)
            last_price = price
        emit(current)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if out["pool_event_id"].duplicated().any():
        raise RuntimeError("duplicate pool_event_id")
    return out.sort_values(["event_pos", "distance_band_bp", "pool_ceiling"], kind="mergesort").reset_index(drop=True)


def _attach_market_state(frame: pd.DataFrame, bars: pd.DataFrame, cfg: LiquidityMagnetConfig) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    out = frame.copy()
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    prev_close = np.r_[np.nan, close[:-1]]
    tr = np.nanmax(
        np.vstack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)]),
        axis=0,
    )
    atr60 = pd.Series(tr).shift(1).rolling(60, min_periods=15).mean().to_numpy(dtype=float)
    positions = pd.to_numeric(out["event_pos"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    entry_pos = pd.to_numeric(out["entry_pos"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
    n = len(out)
    for lookback in (5, 15, 60):
        values = np.full(n, np.nan, dtype=float)
        for i, pos in enumerate(positions):
            start = int(pos) - int(lookback)
            if start >= 0 and np.isfinite(close[start]) and close[start] > 0 and np.isfinite(close[pos]):
                values[i] = (close[pos] / close[start] - 1.0) * 10_000.0
        out[f"pre_return_{lookback}m_bp"] = values
    out["signal_atr60_bp"] = np.where(
        (positions >= 0) & np.isfinite(atr60[np.clip(positions, 0, len(atr60) - 1)]) & (close[np.clip(positions, 0, len(close) - 1)] > 0),
        atr60[np.clip(positions, 0, len(atr60) - 1)] / close[np.clip(positions, 0, len(close) - 1)] * 10_000.0,
        np.nan,
    )
    for window in cfg.local_high_windows_minutes:
        stop = np.full(n, np.nan, dtype=float)
        for i, pos in enumerate(positions):
            start = max(0, int(pos) - int(window) + 1)
            if pos >= 0:
                value = float(np.nanmax(high[start : int(pos) + 1]))
                stop[i] = value * (1.0 + float(cfg.local_high_buffer_bp) / 10_000.0)
        out[f"stop_local_high_{int(window)}m"] = stop
    target = pd.to_numeric(out["pool_ceiling"], errors="coerce").to_numpy(dtype=float) * (
        1.0 + float(cfg.front_run_buffer_bp) / 10_000.0
    )
    entry = np.where(
        (entry_pos >= 0) & (entry_pos < len(open_)),
        open_[np.clip(entry_pos, 0, len(open_) - 1)],
        np.nan,
    )
    out["entry_price"] = entry
    out["front_run_target_price"] = target
    out["tradable_target_distance_bp"] = (entry / target - 1.0) * 10_000.0
    minimum_stop = entry * (1.0 + float(cfg.local_high_buffer_bp) / 10_000.0)
    for window in cfg.local_high_windows_minutes:
        column = f"stop_local_high_{int(window)}m"
        out[column] = np.maximum(pd.to_numeric(out[column], errors="coerce").to_numpy(dtype=float), minimum_stop)
    out["stop_equal_distance"] = entry * (1.0 + np.maximum(out["tradable_target_distance_bp"].to_numpy(dtype=float), 0.0) / 10_000.0)
    out["period"] = _period(out["event_available_time"])
    return out


def _attach_r09_families(frame: pd.DataFrame, level_features: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    cols = ["level_id", *[name for name in FAMILY_COLUMNS if name in level_features.columns]]
    member_features = level_features.loc[:, cols].copy()
    member_features["level_id"] = pd.to_numeric(member_features["level_id"], errors="coerce").astype("Int64")
    lookup = member_features.set_index("level_id")
    out = frame.copy()
    for family in FAMILY_COLUMNS:
        values = []
        for token in out["pool_member_level_ids"].astype(str):
            ids = [int(v) for v in token.split("|") if v]
            present = [bool(lookup.loc[level_id, family]) for level_id in ids if level_id in lookup.index] if family in lookup.columns else []
            values.append(any(present))
        out[family] = np.asarray(values, dtype=bool)
    out["structured_family_count"] = out.loc[:, FAMILY_COLUMNS].astype(bool).sum(axis=1).astype(np.int16)
    out["has_any_structured_family"] = out["structured_family_count"].gt(0)
    return out


def build_liquidity_magnet_universe(
    lifecycle: pd.DataFrame,
    level_features: pd.DataFrame,
    primary: pd.DataFrame,
    config: LiquidityMagnetConfig,
    *,
    research_start: pd.Timestamp,
    research_end_exclusive: pd.Timestamp,
    max_candidates: int = 0,
    show_progress: bool = True,
) -> pd.DataFrame:
    cfg = config.validate()
    crossings = _first_band_crossings(
        lifecycle,
        primary,
        cfg,
        research_start=research_start,
        research_end_exclusive=research_end_exclusive,
        show_progress=show_progress,
    )
    pools = _group_same_bar_pools(crossings, cfg)
    if pools.empty:
        return pools
    pools = _attach_r09_families(pools, level_features)
    confluence_input = pools.loc[:, ["pool_event_id", "event_pos", "pool_center"]].rename(
        columns={"pool_event_id": "event_id", "pool_center": "level_price"}
    )
    confluence_input["event_pos"] = pd.to_numeric(confluence_input["event_pos"], errors="coerce").astype(int)
    atlas_cfg = AtlasConfig(confluence_tolerances_bp=(10.0, 25.0, 50.0)).validate()
    confluence = attach_active_confluence(confluence_input, lifecycle, atlas_cfg).rename(
        columns={"event_id": "pool_event_id"}
    )
    count_cols = [
        name for name in confluence.columns
        if name.startswith("active_level_count_") or name.startswith("active_timeframe_count_")
    ]
    pools = pools.merge(
        confluence.loc[:, ["pool_event_id", *count_cols]],
        on="pool_event_id",
        how="left",
        validate="one_to_one",
    )
    pools = _attach_market_state(pools, normalize_primary_bars(primary), cfg)
    pools = pools.loc[
        pd.to_numeric(pools["tradable_target_distance_bp"], errors="coerce").gt(0)
        & pd.to_datetime(pools["entry_time"], errors="coerce").notna()
    ].copy()
    pools = pools.sort_values(["event_available_time", "distance_band_bp", "pool_event_id"], kind="mergesort").reset_index(drop=True)
    if int(max_candidates) > 0:
        pools = pools.head(int(max_candidates)).copy()
    return pools
