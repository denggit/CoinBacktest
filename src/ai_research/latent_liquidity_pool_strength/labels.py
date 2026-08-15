#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Conditional-on-touch realized liquidity-strength labels for R02.1.

R02.1 deliberately separates two questions:
1) arrival/touch probability, and
2) how much liquidity is actually released if the price reaches a zone.

Strength is therefore evaluated conditionally on touched zones. Untouched zones
are not silently labeled as "no liquidity".

The full-history labeler is intentionally NumPy-first.  The previous reference
implementation iterated every decision timestamp with pandas ``groupby`` /
``iloc.copy`` operations and rebuilt an Episode x zone-distance matrix inside
the loop.  On the 2023-2026 dataset that means millions of rows and tens of
millions of decision/Episode pairs, so the Python/Pandas object overhead can
look like a hang for hours.  The implementation below preserves the exact
causal window and zone mapping while aggregating bounded decision chunks with
vectorized NumPy reductions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter


_NUMERIC_ZERO_COLUMNS = (
    "release_episode_count",
    "release_density_sum",
    "release_density_max",
    "release_episode_size_sum",
    "release_score_sum",
    "favorable_episode_count",
    "continuation_episode_count",
    "favorable_density_sum",
    "continuation_density_sum",
    "sweep_depth_weighted_bp",
    "reversal_room_weighted_bp",
)


@dataclass(frozen=True)
class _EpisodeArrays:
    time_ns: np.ndarray
    reference_price: np.ndarray
    density: np.ndarray
    size: np.ndarray
    score: np.ndarray
    favorable: np.ndarray
    continuation: np.ndarray
    sweep_depth: np.ndarray
    reversal_room: np.ndarray


def _numeric_column(frame: pd.DataFrame, name: str, *, default: float) -> np.ndarray:
    if name not in frame.columns:
        return np.full(len(frame), float(default), dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64, copy=False)


def _safe_positive_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(arr) & (arr > 0.0), arr, 0.0)


def _prepare_episode_arrays(episodes: pd.DataFrame) -> dict[str, _EpisodeArrays]:
    if episodes.empty:
        return {}
    event_time = pd.to_datetime(episodes["event_time"], errors="coerce")
    time_ns = event_time.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    reference = _numeric_column(episodes, "event_reference_price", default=np.nan)
    side = episodes["event_side"].astype(str).to_numpy(copy=False)
    density = _safe_positive_array(_numeric_column(episodes, "release_density_proxy", default=0.0))
    size = _safe_positive_array(_numeric_column(episodes, "release_episode_size", default=1.0))
    score = _safe_positive_array(_numeric_column(episodes, "release_score", default=0.0))
    favorable = (
        episodes["favorable_reversal"].fillna(False).astype(bool).to_numpy(copy=False)
        if "favorable_reversal" in episodes.columns
        else np.zeros(len(episodes), dtype=bool)
    )
    continuation = (
        episodes["outcome_type"].astype(str).eq("ACCEPT_CONTINUATION").to_numpy(copy=False)
        if "outcome_type" in episodes.columns
        else np.zeros(len(episodes), dtype=bool)
    )
    sweep = _numeric_column(episodes, "future_extension_bp", default=np.nan)
    room = _numeric_column(episodes, "future_reversal_after_extreme_bp", default=np.nan)
    nat_i64 = np.datetime64("NaT", "ns").astype(np.int64)
    valid = (time_ns != nat_i64) & np.isfinite(reference) & (reference > 0.0)

    result: dict[str, _EpisodeArrays] = {}
    for side_name in ("DOWN", "UP"):
        idx = np.flatnonzero(valid & (side == side_name))
        if idx.size == 0:
            continue
        order = np.argsort(time_ns[idx], kind="stable")
        idx = idx[order]
        result[side_name] = _EpisodeArrays(
            time_ns=time_ns[idx],
            reference_price=reference[idx],
            density=density[idx],
            size=size[idx],
            score=score[idx],
            favorable=favorable[idx],
            continuation=continuation[idx],
            sweep_depth=sweep[idx],
            reversal_room=room[idx],
        )
    return result


def _zone_positions(distance_bp: np.ndarray, offsets: np.ndarray, half: float) -> tuple[np.ndarray, np.ndarray]:
    """Map distances to the same nearest-zone semantics as ``argmin``.

    ``searchsorted(..., side='left')`` on adjacent midpoints intentionally sends
    an exact midpoint tie to the lower zone, matching ``np.argmin``.
    """
    distance = np.asarray(distance_bp, dtype=np.float64)
    if offsets.size == 1:
        pos = np.zeros(distance.size, dtype=np.int32)
    else:
        midpoints = (offsets[:-1] + offsets[1:]) * 0.5
        pos = np.searchsorted(midpoints, distance, side="left").astype(np.int32, copy=False)
    valid = (
        np.isfinite(distance)
        & (distance > 0.0)
        & (distance <= float(offsets[-1]) + float(half))
        & (np.abs(distance - offsets[pos]) <= float(half) + 1e-12)
    )
    return pos, valid


def _expanded_ranges(starts: np.ndarray, ends: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(decision_local_idx, episode_idx)`` for interval slices.

    Chunking keeps this pair expansion bounded.  The helper avoids any pandas
    object creation inside the full-history hot loop.
    """
    counts = np.maximum(0, np.asarray(ends, dtype=np.int64) - np.asarray(starts, dtype=np.int64))
    total = int(counts.sum())
    if total <= 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int64)
    decision_idx = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    blocks = [np.arange(int(start), int(end), dtype=np.int64) for start, end in zip(starts, ends) if end > start]
    episode_idx = np.concatenate(blocks) if blocks else np.empty(0, dtype=np.int64)
    return decision_idx, episode_idx


def _aggregate_side_chunk(
    decision_time_ns: np.ndarray,
    current_price: np.ndarray,
    eps: _EpisodeArrays,
    *,
    side: str,
    offsets: np.ndarray,
    half: float,
    horizon_ns: int,
) -> dict[str, np.ndarray]:
    n_decisions = len(decision_time_ns)
    n_zones = len(offsets)
    size = n_decisions * n_zones
    zeros = lambda: np.zeros(size, dtype=np.float64)
    result = {name: zeros() for name in _NUMERIC_ZERO_COLUMNS}
    result["first_release_minutes"] = np.full(size, np.nan, dtype=np.float64)
    result["release_within_horizon"] = np.zeros(size, dtype=bool)
    if n_decisions == 0 or eps.time_ns.size == 0:
        return result

    starts = np.searchsorted(eps.time_ns, decision_time_ns, side="right")
    # Exclusive right edge: (decision_time, decision_time + horizon).
    ends = np.searchsorted(eps.time_ns, decision_time_ns + int(horizon_ns), side="left")
    dec_local, ep_idx = _expanded_ranges(starts, ends)
    if ep_idx.size == 0:
        return result

    ref = eps.reference_price[ep_idx]
    curr = current_price[dec_local]
    if side == "DOWN":
        distance = (curr - ref) / curr * 1e4
    else:
        distance = (ref - curr) / curr * 1e4
    zone_pos, valid = _zone_positions(distance, offsets, half)
    valid &= np.isfinite(curr) & (curr > 0.0)
    if not valid.any():
        return result

    dec_local = dec_local[valid]
    ep_idx = ep_idx[valid]
    zone_pos = zone_pos[valid]
    flat = dec_local.astype(np.int64) * n_zones + zone_pos.astype(np.int64)

    density = eps.density[ep_idx]
    episode_size = eps.size[ep_idx]
    score = eps.score[ep_idx]
    favorable = eps.favorable[ep_idx]
    continuation = eps.continuation[ep_idx]
    sweep = eps.sweep_depth[ep_idx]
    room = eps.reversal_room[ep_idx]
    weight = np.where(density > 0.0, density, 1.0)

    count = np.bincount(flat, minlength=size).astype(np.float64, copy=False)
    result["release_episode_count"] = count
    result["release_density_sum"] = np.bincount(flat, weights=density, minlength=size)
    max_density = zeros()
    np.maximum.at(max_density, flat, density)
    result["release_density_max"] = max_density
    result["release_episode_size_sum"] = np.bincount(flat, weights=episode_size, minlength=size)
    result["release_score_sum"] = np.bincount(flat, weights=score, minlength=size)
    result["favorable_episode_count"] = np.bincount(flat, weights=favorable.astype(np.float64), minlength=size)
    result["continuation_episode_count"] = np.bincount(flat, weights=continuation.astype(np.float64), minlength=size)
    result["favorable_density_sum"] = np.bincount(flat, weights=density * favorable, minlength=size)
    result["continuation_density_sum"] = np.bincount(flat, weights=density * continuation, minlength=size)

    valid_sweep = np.isfinite(sweep)
    if valid_sweep.any():
        den = np.bincount(flat[valid_sweep], weights=weight[valid_sweep], minlength=size)
        num = np.bincount(flat[valid_sweep], weights=sweep[valid_sweep] * weight[valid_sweep], minlength=size)
        np.divide(num, den, out=result["sweep_depth_weighted_bp"], where=den > 0.0)
    valid_room = np.isfinite(room)
    if valid_room.any():
        den = np.bincount(flat[valid_room], weights=weight[valid_room], minlength=size)
        num = np.bincount(flat[valid_room], weights=room[valid_room] * weight[valid_room], minlength=size)
        np.divide(num, den, out=result["reversal_room_weighted_bp"], where=den > 0.0)

    first = np.full(size, np.inf, dtype=np.float64)
    elapsed_minutes = (eps.time_ns[ep_idx] - decision_time_ns[dec_local]).astype(np.float64) / 60_000_000_000.0
    np.minimum.at(first, flat, elapsed_minutes)
    first[~np.isfinite(first)] = np.nan
    result["first_release_minutes"] = first
    result["release_within_horizon"] = count > 0.0
    return result


def attach_strength_labels(
    zones: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    horizon_minutes: int,
    zone_offsets_bp: tuple[float, ...],
    zone_half_width_bp: float,
    decision_chunk_size: int = 1024,
    progress: bool = False,
) -> pd.DataFrame:
    """Aggregate *all* future release Episodes mapped to each price zone.

    Causal semantics are unchanged from R02.1:
    - future window is ``(decision_time, decision_time + horizon)``;
    - the right edge is exclusive;
    - each Episode maps to the nearest configured price zone, with midpoint ties
      assigned to the lower-distance zone;
    - only rows that actually exist in the sampled/full-lattice spatial frame are
      populated.

    The implementation runs in bounded decision-time chunks and reports progress
    when requested, avoiding the old per-decision pandas copies.
    """
    if zones.empty:
        return zones.copy()
    if decision_chunk_size < 1:
        raise ValueError("decision_chunk_size must be positive")

    # Shallow copy is sufficient: R02.1 only appends label columns and never
    # mutates pre-event spatial feature columns.
    out = zones.copy(deep=False)
    n_rows = len(out)
    label_arrays: dict[str, np.ndarray] = {
        name: np.zeros(n_rows, dtype=np.float64) for name in _NUMERIC_ZERO_COLUMNS
    }
    label_arrays["first_release_minutes"] = np.full(n_rows, np.nan, dtype=np.float64)
    release_flag = np.zeros(n_rows, dtype=bool)

    decision_dt = pd.to_datetime(out["decision_time"], errors="coerce")
    if decision_dt.isna().any():
        raise ValueError("R02.1 decision_time contains NaT")
    decision_row_ns = decision_dt.to_numpy(dtype="datetime64[ns]").astype(np.int64, copy=False)
    current_row = pd.to_numeric(out["current_price"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    unique_time_ns, first_row, decision_id = np.unique(
        decision_row_ns, return_index=True, return_inverse=True
    )
    current_by_decision = current_row[first_row]
    n_decisions = len(unique_time_ns)

    side_text = out["zone_side"].astype(str).to_numpy(copy=False)
    side_code = np.full(n_rows, -1, dtype=np.int8)
    side_code[side_text == "DOWN"] = 0
    side_code[side_text == "UP"] = 1
    offsets = np.asarray(zone_offsets_bp, dtype=np.float64)
    zone_distance = pd.to_numeric(out["zone_distance_bp"], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    zone_pos, zone_valid = _zone_positions(zone_distance, offsets, float(zone_half_width_bp))

    # One stable sort lets each decision chunk find its existing sampled lattice
    # rows without scanning all ~2M rows again.
    row_order = np.argsort(decision_id, kind="stable")
    rows_per_decision = np.bincount(decision_id, minlength=n_decisions)
    row_boundaries = np.concatenate(([0], np.cumsum(rows_per_decision, dtype=np.int64)))

    eps_by_side = _prepare_episode_arrays(episodes)
    horizon_ns = int(pd.Timedelta(minutes=int(horizon_minutes)).value)
    chunk_size = int(decision_chunk_size)
    n_chunks = (n_decisions + chunk_size - 1) // chunk_size
    if progress:
        print(
            f"[strength-aggregation] decision_times={n_decisions:,} chunks={n_chunks:,} "
            f"chunk_size={chunk_size:,} Episodes={sum(len(x.time_ns) for x in eps_by_side.values()):,}",
            flush=True,
        )
    reporter = ProgressReporter(
        label="[latent-liquidity-r02.1] strength aggregation",
        total=n_chunks,
        every=1,
        enabled=progress,
    )

    for chunk_no, lo in enumerate(range(0, n_decisions, chunk_size), start=1):
        hi = min(n_decisions, lo + chunk_size)
        dec_times = unique_time_ns[lo:hi]
        dec_prices = current_by_decision[lo:hi]
        row_slice = row_order[row_boundaries[lo] : row_boundaries[hi]]
        local_dec = decision_id[row_slice] - lo
        local_zone = zone_pos[row_slice]
        local_side = side_code[row_slice]
        local_valid = zone_valid[row_slice] & (local_side >= 0)

        for side_name, code in (("DOWN", 0), ("UP", 1)):
            eps = eps_by_side.get(side_name)
            if eps is None:
                continue
            agg = _aggregate_side_chunk(
                dec_times,
                dec_prices,
                eps,
                side=side_name,
                offsets=offsets,
                half=float(zone_half_width_bp),
                horizon_ns=horizon_ns,
            )
            choose = local_valid & (local_side == code)
            if not choose.any():
                continue
            rows = row_slice[choose]
            flat = local_dec[choose].astype(np.int64) * len(offsets) + local_zone[choose].astype(np.int64)
            for name in _NUMERIC_ZERO_COLUMNS:
                label_arrays[name][rows] = agg[name][flat]
            label_arrays["first_release_minutes"][rows] = agg["first_release_minutes"][flat]
            release_flag[rows] = agg["release_within_horizon"][flat]
        reporter.update(chunk_no)
    reporter.close()

    for name, values in label_arrays.items():
        out[name] = values
    out["release_within_horizon"] = release_flag
    out["release_density_log"] = np.log1p(np.clip(label_arrays["release_density_sum"], 0.0, None))
    out["release_count_log"] = np.log1p(np.clip(label_arrays["release_episode_count"], 0.0, None))
    out["release_size_log"] = np.log1p(np.clip(label_arrays["release_episode_size_sum"], 0.0, None))
    out["release_peak_log"] = np.log1p(np.clip(label_arrays["release_density_max"], 0.0, None))
    return out


def attach_train_frozen_strength_thresholds(
    frame: pd.DataFrame,
    *,
    train_period: str,
    quantile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze side-specific high-strength thresholds on TRAIN touched zones only."""
    # Only append one target column; avoid a deep copy of the multi-million-row
    # spatial feature table.
    out = frame.copy(deep=False)
    rows: list[dict[str, object]] = []
    high_strength = np.zeros(len(out), dtype=bool)
    period = out["period"].astype(str)
    zone_side = out["zone_side"].astype(str)
    touched = out["touch_720m"].astype(bool)
    density_log = pd.to_numeric(out["release_density_log"], errors="coerce")
    for side in ("DOWN", "UP"):
        mask = period.eq(train_period) & zone_side.eq(side) & touched
        if "full_lattice_audit_group" in out.columns:
            audit_mask = mask & out["full_lattice_audit_group"].astype(bool)
            train_mask = audit_mask if audit_mask.any() else mask
        else:
            train_mask = mask
        values = density_log.loc[train_mask]
        values = values[np.isfinite(values)]
        threshold = float(values.quantile(quantile)) if len(values) else np.nan
        rows.append(
            {
                "zone_side": side,
                "train_strength_quantile": quantile,
                "release_density_log_threshold": threshold,
                "train_touched_rows": int(len(values)),
            }
        )
        if np.isfinite(threshold):
            side_mask = zone_side.eq(side) & touched
            high_strength[side_mask.to_numpy()] = density_log.loc[side_mask].ge(threshold).to_numpy()
    out["high_strength_label"] = high_strength
    return out, pd.DataFrame(rows)
