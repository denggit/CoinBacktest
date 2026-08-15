#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Exact first-touch labels with fixed post-touch observation windows.

The purpose of R02.2 is to remove the exposure-time bias from R02.1.  A zone
that is touched after 11 hours receives the same 30/60/180/300-second label
window as a zone touched after 10 minutes.  First touch is resolved from 1m
bars and refined to the exact 1-second bar.  Release Episode and flow labels
are then anchored on that exact first-touch second.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_execution_audit.replay import _normalize_bars
from src.ai_research.latent_liquidity_path_atlas.time_axis import as_datetime_ns
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter

from .cache import chunk_cache_path, load_frame, save_frame
from .config import FirstTouchLiquidityRankingConfig




def _normalize_touch_second_bars(bars: pd.DataFrame, max_fill_gap_seconds: int) -> pd.DataFrame:
    """Regularize only the six 1s columns needed by R02.2.

    This intentionally avoids the wider R01 feature normalizer so full-history
    first-touch replay does not materialize unused microstructure columns.
    """
    if bars.empty:
        return pd.DataFrame()
    out = bars.loc[:, [c for c in ("close", "low", "high", "notional", "trades_count", "delta_notional") if c in bars.columns]].copy()
    out.index = as_datetime_ns(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index(kind="mergesort")
    out = out.loc[~out.index.duplicated(keep="last")]
    if out.empty:
        return out
    full_index = pd.date_range(out.index.min().floor("s"), out.index.max().floor("s"), freq="1s")
    observed = pd.Series(True, index=out.index).reindex(full_index, fill_value=False)
    out = out.reindex(full_index)
    close = pd.to_numeric(out.get("close"), errors="coerce").ffill()
    out["low"] = pd.to_numeric(out.get("low"), errors="coerce").fillna(close)
    out["high"] = pd.to_numeric(out.get("high"), errors="coerce").fillna(close)
    for name in ("notional", "trades_count", "delta_notional"):
        if name in out:
            out[name] = pd.to_numeric(out[name], errors="coerce").fillna(0.0)
        else:
            out[name] = 0.0
    missing = (~observed).astype(np.int8)
    run_id = (missing != missing.shift(fill_value=0)).cumsum()
    missing_run = missing.groupby(run_id).transform("sum")
    out["unsafe_gap"] = (missing.eq(1) & missing_run.gt(int(max_fill_gap_seconds))).astype(np.int8)
    out.index.name = "timestamp"
    return out.loc[:, ["low", "high", "notional", "trades_count", "delta_notional", "unsafe_gap"]]

@dataclass(frozen=True)
class FirstTouchBuildResult:
    frame: pd.DataFrame
    quality: pd.DataFrame


def _first_touch_minute_positions(
    minute_frame: pd.DataFrame,
    *,
    side: str,
    thresholds: np.ndarray,
    zero_distance: np.ndarray,
) -> np.ndarray:
    """Vectorized first crossing position for many ordered zone thresholds."""
    n = len(thresholds)
    out = np.full(n, -1, dtype=np.int32)
    if minute_frame.empty or n == 0:
        return out
    if side == "DOWN":
        extrema = np.minimum.accumulate(pd.to_numeric(minute_frame["low"], errors="coerce").to_numpy(dtype=float))
        seq = -extrema
        for i, threshold in enumerate(np.asarray(thresholds, dtype=float)):
            if not np.isfinite(threshold):
                continue
            pos = int(np.searchsorted(seq, -threshold, side="right" if bool(zero_distance[i]) else "left"))
            if pos < len(seq):
                out[i] = pos
    else:
        extrema = np.maximum.accumulate(pd.to_numeric(minute_frame["high"], errors="coerce").to_numpy(dtype=float))
        seq = extrema
        for i, threshold in enumerate(np.asarray(thresholds, dtype=float)):
            if not np.isfinite(threshold):
                continue
            pos = int(np.searchsorted(seq, threshold, side="right" if bool(zero_distance[i]) else "left"))
            if pos < len(seq):
                out[i] = pos
    return out


@dataclass(frozen=True)
class _SecondArrays:
    time_ns: np.ndarray
    low: np.ndarray
    high: np.ndarray
    notional_prefix: np.ndarray
    trades_prefix: np.ndarray
    abs_delta_prefix: np.ndarray
    unsafe_prefix: np.ndarray


def _prefix(values: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values) + 1, dtype=np.float64)
    out[1:] = np.cumsum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), dtype=np.float64)
    return out


def _prepare_second_arrays(second_frame: pd.DataFrame) -> _SecondArrays | None:
    if second_frame.empty:
        return None
    time_ns = second_frame.index.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    low = pd.to_numeric(second_frame["low"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    high = pd.to_numeric(second_frame["high"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    notional = pd.to_numeric(second_frame["notional"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    trades = pd.to_numeric(second_frame["trades_count"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    delta = pd.to_numeric(second_frame["delta_notional"], errors="coerce").fillna(0.0).abs().to_numpy(dtype=np.float64, copy=False)
    unsafe = pd.to_numeric(second_frame.get("unsafe_gap", 0), errors="coerce")
    if not isinstance(unsafe, pd.Series):
        unsafe = pd.Series(np.zeros(len(second_frame), dtype=np.int8), index=second_frame.index)
    unsafe_arr = unsafe.fillna(1).astype(bool).to_numpy(dtype=np.int8, copy=False)
    return _SecondArrays(
        time_ns=time_ns, low=low, high=high,
        notional_prefix=_prefix(notional),
        trades_prefix=_prefix(trades),
        abs_delta_prefix=_prefix(delta),
        unsafe_prefix=np.concatenate(([0], np.cumsum(unsafe_arr, dtype=np.int64))),
    )


def _exact_touch_seconds(
    second: _SecondArrays | None,
    minute_starts: np.ndarray,
    *,
    side: str,
    thresholds: np.ndarray,
    strict: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(thresholds)
    touch_ns = np.full(n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    touch_pos = np.full(n, -1, dtype=np.int64)
    if second is None or n == 0:
        return touch_ns, touch_pos
    for i, minute64 in enumerate(minute_starts):
        if np.isnat(minute64) or not np.isfinite(thresholds[i]):
            continue
        start_ns = np.asarray(minute64, dtype="datetime64[ns]").astype(np.int64)
        left = int(np.searchsorted(second.time_ns, start_ns, side="left"))
        right = int(np.searchsorted(second.time_ns, start_ns + 60_000_000_000, side="left"))
        if right <= left:
            continue
        values = second.low[left:right] if side == "DOWN" else second.high[left:right]
        if side == "DOWN":
            hit = values < thresholds[i] if bool(strict[i]) else values <= thresholds[i]
        else:
            hit = values > thresholds[i] if bool(strict[i]) else values >= thresholds[i]
        hits = np.flatnonzero(hit & np.isfinite(values))
        if hits.size == 0:
            continue
        pos = left + int(hits[0])
        # Long missing-trade gaps before the exact crossing second make the
        # second-level first-touch timestamp ambiguous, so reject that label.
        if int(second.unsafe_prefix[pos + 1] - second.unsafe_prefix[left]) > 0:
            continue
        touch_pos[i] = pos
        touch_ns[i] = np.datetime64(int(second.time_ns[pos]), "ns")
    return touch_ns, touch_pos

def _numeric_series(frame: pd.DataFrame, name: str, default: float) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.full(len(frame), float(default)), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _episode_arrays(episodes: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for side, group in episodes.groupby("event_side", sort=False):
        g = group.sort_values("event_time", kind="mergesort")
        result[str(side)] = {
            "time": g["event_time"].to_numpy(dtype="datetime64[ns]"),
            "reference": _numeric_series(g, "event_reference_price", np.nan).to_numpy(dtype=float),
            "density": _numeric_series(g, "release_density_proxy", 0.0).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float),
            "size": _numeric_series(g, "release_episode_size", 0.0).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float),
            "score": _numeric_series(g, "release_score", 0.0).fillna(0.0).clip(lower=0.0).to_numpy(dtype=float),
            "favorable": g.get("favorable_reversal", pd.Series(False, index=g.index)).fillna(False).astype(bool).to_numpy(),
            "continuation": g.get("outcome_type", pd.Series("", index=g.index)).astype(str).eq("ACCEPT_CONTINUATION").to_numpy(),
            "sweep": _numeric_series(g, "future_extension_bp", np.nan).to_numpy(dtype=float),
            "room": _numeric_series(g, "future_reversal_after_extreme_bp", np.nan).to_numpy(dtype=float),
        }
    return result


def _aggregate_episode_windows_for_group(
    rows: pd.DataFrame,
    touch_times: np.ndarray,
    arrays: dict[str, np.ndarray] | None,
    *,
    side: str,
    windows: tuple[int, ...],
) -> dict[str, np.ndarray]:
    n = len(rows)
    result: dict[str, np.ndarray] = {}
    for window in windows:
        for name in (
            "release_episode_count", "release_density_sum", "release_density_max", "release_episode_size_sum",
            "release_score_sum", "favorable_episode_count", "continuation_episode_count",
            "favorable_density_sum", "continuation_density_sum", "sweep_depth_weighted_bp", "reversal_room_weighted_bp",
        ):
            result[f"ft_{name}_{window}s"] = np.zeros(n, dtype=np.float64)
    if arrays is None or n == 0:
        return result
    event_times = arrays["time"]
    if event_times.size == 0:
        return result
    offsets = pd.to_numeric(rows["zone_distance_bp"], errors="coerce").to_numpy(dtype=float)
    current = float(pd.to_numeric(rows["current_price"], errors="coerce").iloc[0])
    half = float(np.nanmedian(
        (pd.to_numeric(rows["zone_far_distance_bp"], errors="coerce").to_numpy(dtype=float)
         - pd.to_numeric(rows["zone_near_distance_bp"], errors="coerce").to_numpy(dtype=float)) / 2.0
    ))
    if len(offsets) > 1:
        midpoints = (offsets[:-1] + offsets[1:]) * 0.5
    else:
        midpoints = np.empty(0, dtype=float)
    for i, touch in enumerate(touch_times):
        if np.isnat(touch):
            continue
        left = int(np.searchsorted(event_times, touch, side="left"))
        max_end = touch + np.timedelta64(int(max(windows)), "s")
        right = int(np.searchsorted(event_times, max_end, side="left"))
        if right <= left:
            continue
        ref = arrays["reference"][left:right]
        distance = (current - ref) / current * 1e4 if side == "DOWN" else (ref - current) / current * 1e4
        if len(offsets) == 1:
            nearest = np.zeros(len(distance), dtype=np.int32)
        else:
            nearest = np.searchsorted(midpoints, distance, side="left").astype(np.int32, copy=False)
        valid = (
            np.isfinite(distance) & (distance > 0.0)
            & (nearest >= 0) & (nearest < len(offsets))
            & (np.abs(distance - offsets[np.clip(nearest, 0, len(offsets) - 1)]) <= half + 1e-12)
        )
        zone_mask = valid & (nearest == i)
        if not zone_mask.any():
            continue
        rel_idx = np.flatnonzero(zone_mask) + left
        rel_time = event_times[rel_idx]
        for window in windows:
            within = rel_time < touch + np.timedelta64(int(window), "s")
            idx = rel_idx[within]
            if idx.size == 0:
                continue
            density = arrays["density"][idx]
            positive_density = np.where(np.isfinite(density) & (density > 0), density, 0.0)
            result[f"ft_release_episode_count_{window}s"][i] = float(len(idx))
            result[f"ft_release_density_sum_{window}s"][i] = float(np.sum(positive_density))
            result[f"ft_release_density_max_{window}s"][i] = float(np.max(positive_density)) if len(positive_density) else 0.0
            result[f"ft_release_episode_size_sum_{window}s"][i] = float(np.nansum(arrays["size"][idx]))
            result[f"ft_release_score_sum_{window}s"][i] = float(np.nansum(arrays["score"][idx]))
            fav = arrays["favorable"][idx]
            cont = arrays["continuation"][idx]
            result[f"ft_favorable_episode_count_{window}s"][i] = float(np.sum(fav))
            result[f"ft_continuation_episode_count_{window}s"][i] = float(np.sum(cont))
            result[f"ft_favorable_density_sum_{window}s"][i] = float(np.sum(positive_density[fav]))
            result[f"ft_continuation_density_sum_{window}s"][i] = float(np.sum(positive_density[cont]))
            denom = float(np.sum(positive_density))
            if denom > 1e-12:
                sweep = arrays["sweep"][idx]
                room = arrays["room"][idx]
                sv = np.isfinite(sweep)
                rv = np.isfinite(room)
                result[f"ft_sweep_depth_weighted_bp_{window}s"][i] = float(np.sum(positive_density[sv] * sweep[sv]) / np.sum(positive_density[sv])) if sv.any() and np.sum(positive_density[sv]) > 0 else 0.0
                result[f"ft_reversal_room_weighted_bp_{window}s"][i] = float(np.sum(positive_density[rv] * room[rv]) / np.sum(positive_density[rv])) if rv.any() and np.sum(positive_density[rv]) > 0 else 0.0
    return result


def _range_sum(prefix: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return prefix[right] - prefix[left]


def _flow_labels(
    second: _SecondArrays | None,
    touch_pos: np.ndarray,
    *,
    windows: tuple[int, ...],
    pre_seconds: int,
) -> dict[str, np.ndarray]:
    n = len(touch_pos)
    result: dict[str, np.ndarray] = {}
    for window in windows:
        result[f"ft_notional_ratio_{window}s"] = np.full(n, np.nan, dtype=float)
        result[f"ft_trades_ratio_{window}s"] = np.full(n, np.nan, dtype=float)
        result[f"ft_abs_delta_ratio_{window}s"] = np.full(n, np.nan, dtype=float)
    micro_complete = np.zeros(n, dtype=bool)
    post_complete = np.zeros(n, dtype=bool)
    result["ft_micro_label_complete"] = micro_complete
    result["ft_post_label_complete"] = post_complete
    if second is None or n == 0:
        return result
    pos = np.asarray(touch_pos, dtype=np.int64)
    post_valid = (pos >= 0) & (pos + max(windows) <= len(second.time_ns))
    post_idx = np.flatnonzero(post_valid)
    if len(post_idx):
        pp = pos[post_idx]
        post_unsafe = second.unsafe_prefix[pp + max(windows)] - second.unsafe_prefix[pp]
        post_idx = post_idx[post_unsafe == 0]
        post_complete[post_idx] = True
    valid = post_complete & (pos >= pre_seconds)
    if not valid.any():
        result["ft_post_label_complete"] = post_complete
        return result
    idx = np.flatnonzero(valid)
    p = pos[idx]
    pre_left = p - int(pre_seconds)
    pre_right = p
    unsafe_pre = second.unsafe_prefix[p] - second.unsafe_prefix[pre_left]
    safe_mask = unsafe_pre == 0
    idx = idx[safe_mask]
    p = p[safe_mask]
    pre_left = pre_left[safe_mask]
    pre_right = pre_right[safe_mask]
    if len(idx) == 0:
        result["ft_post_label_complete"] = post_complete
        return result
    pre_notional = _range_sum(second.notional_prefix, pre_left, pre_right)
    pre_trades = _range_sum(second.trades_prefix, pre_left, pre_right)
    pre_abs_delta = _range_sum(second.abs_delta_prefix, pre_left, pre_right)
    for window in windows:
        right = p + int(window)
        post_notional = _range_sum(second.notional_prefix, p, right)
        post_trades = _range_sum(second.trades_prefix, p, right)
        post_abs_delta = _range_sum(second.abs_delta_prefix, p, right)
        scale = float(window) / float(pre_seconds)
        result[f"ft_notional_ratio_{window}s"][idx] = post_notional / np.maximum(pre_notional * scale, 1e-9)
        result[f"ft_trades_ratio_{window}s"][idx] = post_trades / np.maximum(pre_trades * scale, 1e-9)
        result[f"ft_abs_delta_ratio_{window}s"][idx] = post_abs_delta / np.maximum(pre_abs_delta * scale, 1e-9)
    micro_complete[idx] = True
    result["ft_micro_label_complete"] = micro_complete
    result["ft_post_label_complete"] = post_complete
    return result

def _period_chunk_bounds(times: pd.Series, days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(times.min()).floor("D")
    end = pd.Timestamp(times.max()).floor("D")
    result = []
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + pd.Timedelta(days=days - 1), end)
        result.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(days=1)
    return result


def build_first_touch_dataset(
    audit_lattice: pd.DataFrame,
    episodes: pd.DataFrame,
    config: FirstTouchLiquidityRankingConfig,
    *,
    use_cache: bool = True,
    data_dir: str | Path | None = None,
    db_name: str = "okx_trade_bars.db",
    progress: bool = True,
) -> FirstTouchBuildResult:
    if audit_lattice.empty:
        return FirstTouchBuildResult(pd.DataFrame(), pd.DataFrame())
    work = audit_lattice.copy()
    work["decision_time"] = pd.to_datetime(work["decision_time"], errors="coerce")
    work = work.loc[work["decision_time"].notna()].copy()
    work["decision_day"] = work["decision_time"].dt.floor("D")
    episode_by_side = _episode_arrays(episodes)
    minute_loader = OKXTradeBarLoader(symbol=config.symbol, timeframe="1m", data_dir=data_dir, db_name=db_name)
    second_loader = OKXTradeBarLoader(symbol=config.symbol, timeframe="1s", data_dir=data_dir, db_name=db_name)
    chunks = _period_chunk_bounds(work["decision_time"], config.touch_replay_chunk_days)
    reporter = ProgressReporter("[latent-liquidity-r02.2] exact first-touch chunks", len(chunks), every=1, enabled=progress)
    parts: list[pd.DataFrame] = []
    quality_rows: list[dict[str, object]] = []
    for number, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        cache = chunk_cache_path(config, chunk_start, chunk_end)
        chunk_rows = work.loc[(work["decision_day"] >= chunk_start) & (work["decision_day"] <= chunk_end)].copy()
        if chunk_rows.empty:
            reporter.update(number)
            continue
        if use_cache and cache.exists():
            try:
                cached = load_frame(cache)
                parts.append(cached)
                quality_rows.append({"chunk_start": chunk_start, "chunk_end": chunk_end, "rows": len(cached), "cache": True})
                reporter.update(number)
                continue
            except (OSError, ValueError, EOFError):
                cache.unlink(missing_ok=True)
        load_start = chunk_rows["decision_time"].min() - pd.Timedelta(seconds=config.pre_touch_baseline_seconds + 5)
        load_end = chunk_rows["decision_time"].max() + pd.Timedelta(minutes=config.primary_horizon_minutes, seconds=max(config.label_windows_seconds) + 5)
        minute = minute_loader.fetch_data_by_date_range(load_start.floor("min"), load_end.ceil("min"), build_missing=False, force_rebuild=False, cvd_mode="range")
        minute = _normalize_bars(minute)
        second = second_loader.fetch_data_by_date_range(load_start.floor("s"), load_end.ceil("s"), build_missing=False, force_rebuild=False, cvd_mode="range")
        second = _normalize_touch_second_bars(second, config.max_fill_gap_seconds)
        second_arrays = _prepare_second_arrays(second)
        out_parts: list[pd.DataFrame] = []
        for (decision_time, side), rows in chunk_rows.groupby(["decision_time", "zone_side"], sort=True):
            rows = rows.sort_values("zone_distance_bp", kind="mergesort").copy()
            start = pd.Timestamp(decision_time)
            future_minute = minute.loc[(minute.index >= start) & (minute.index < start + pd.Timedelta(minutes=config.primary_horizon_minutes))]
            thresholds = pd.to_numeric(rows["zone_near_price"], errors="coerce").to_numpy(dtype=float)
            zero = pd.to_numeric(rows["zone_near_distance_bp"], errors="coerce").fillna(0.0).to_numpy(dtype=float) <= 1e-9
            positions = _first_touch_minute_positions(future_minute, side=str(side), thresholds=thresholds, zero_distance=zero)
            minute_starts = np.full(len(rows), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
            valid_positions = np.flatnonzero(positions >= 0)
            if len(valid_positions):
                minute_starts[valid_positions] = future_minute.index.to_numpy(dtype="datetime64[ns]")[positions[valid_positions]]
            touch, touch_pos = _exact_touch_seconds(
                second_arrays, minute_starts, side=str(side), thresholds=thresholds, strict=zero,
            )
            rows["first_touch_time"] = pd.to_datetime(touch)
            rows["first_touch_observed"] = rows["first_touch_time"].notna()
            rows["time_to_first_touch_minutes"] = (
                (rows["first_touch_time"] - start).dt.total_seconds() / 60.0
            )
            ep_labels = _aggregate_episode_windows_for_group(
                rows, touch, episode_by_side.get(str(side)), side=str(side), windows=config.label_windows_seconds,
            )
            for name, values in ep_labels.items():
                rows[name] = values
            flow = _flow_labels(
                second_arrays, touch_pos, windows=config.label_windows_seconds, pre_seconds=config.pre_touch_baseline_seconds,
            )
            for name, values in flow.items():
                rows[name] = values
            rows["first_touch_label_complete"] = rows["first_touch_observed"] & rows["ft_post_label_complete"].astype(bool)
            out_parts.append(rows)
        chunk_frame = pd.concat(out_parts, ignore_index=True, copy=False) if out_parts else pd.DataFrame()
        if use_cache:
            save_frame(cache, chunk_frame)
        parts.append(chunk_frame)
        quality_rows.append({
            "chunk_start": chunk_start, "chunk_end": chunk_end, "rows": len(chunk_frame),
            "touched": int(chunk_frame.get("first_touch_observed", pd.Series(dtype=bool)).sum()) if not chunk_frame.empty else 0,
            "complete": int(chunk_frame.get("first_touch_label_complete", pd.Series(dtype=bool)).sum()) if not chunk_frame.empty else 0,
            "cache": False,
        })
        reporter.update(number)
    reporter.close()
    frame = pd.concat([p for p in parts if not p.empty], ignore_index=True, copy=False) if any(not p.empty for p in parts) else pd.DataFrame()
    return FirstTouchBuildResult(frame=frame, quality=pd.DataFrame(quality_rows))


def add_relative_relevance(frame: pd.DataFrame, config: FirstTouchLiquidityRankingConfig) -> pd.DataFrame:
    out = frame.copy()
    target = f"ft_release_density_sum_{config.primary_label_window_seconds}s"
    out["ranking_target"] = pd.to_numeric(out[target], errors="coerce")
    out["ranking_group"] = out["decision_time"].astype(str) + "|" + out["zone_side"].astype(str)
    eligible = out["first_touch_label_complete"].astype(bool) & out["ranking_target"].notna()
    out["ranking_relevance"] = -1
    out["ranking_group_eligible"] = False
    for _, idx in out.loc[eligible].groupby("ranking_group", sort=False).groups.items():
        values = out.loc[idx, "ranking_target"].to_numpy(dtype=float)
        if len(values) < 2 or not np.isfinite(values).all() or float(np.max(values) - np.min(values)) <= 1e-12:
            continue
        ranks = pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)
        grades = np.floor(np.clip(ranks - 1e-12, 0.0, 0.999999) * config.rank_relevance_grades).astype(np.int16)
        out.loc[idx, "ranking_relevance"] = grades
        out.loc[idx, "ranking_group_eligible"] = True
    return out
