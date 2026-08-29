#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02 primitives for causal ETH liquidity-pool / stack exhaustion research.

R01 deliberately treated each confirmed HTF swing as an individual lifecycle
object.  That is useful for taxonomy, but a single 1m liquidation impulse can
sweep several nearby swings at once.  R02 therefore changes the statistical
unit from ``level_event`` to two causal units:

* ``sweep_stage``: one 1m bar consuming one or more already-active levels;
* ``sweep_episode``: consecutive same-direction stages that keep extending the
  sweep extreme within a short, fixed causal gap.

Nearby swept levels are additionally merged into price pools at multiple fixed
5/10/20bp tolerances.  These tolerances are *descriptive sensitivity checks*,
not a fitted entry threshold.

Trade management is also changed.  Time is only a research censoring horizon;
it is never a forced profit-taking exit.  Trades use a structural stop beyond
the sweep/confirmation extreme and frozen opposing-liquidity targets selected
from the active book at entry time.  Unresolved trades are marked ``censored``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter
from src.research_common.swing_liquidity_atlas.thresholds import FenwickTree, SegmentThresholdIndex
from src.research_common.swing_liquidity_zone_study.outcomes import RangeMinMaxIndex

from .core import (
    EPS,
    MSS2Config,
    _PivotReferenceIndex,
    _first_fvg_in_range,
    _fvg_leg_stats,
    _range_extreme,
    _true_range,
    aggregate_bars,
    attach_session_context,
    build_execution_pivots,
    normalize_1m_bars,
)


@dataclass(frozen=True)
class R02Config:
    """Configuration for R02 pool/episode and structural-exit research."""

    pool_tolerances_bps: tuple[float, ...] = (5.0, 10.0, 20.0)
    episode_gap_minutes: int = 15
    max_confirmation_minutes: int = 180
    max_fvg_wait_minutes: int = 180
    exit_censor_minutes: int = 10_080  # 7 days; research censor, not a forced exit
    path_horizons_minutes: tuple[int, ...] = (60, 360, 720, 1_440, 2_880, 4_320, 10_080)
    target_pool_tolerance_bps: float = 10.0
    target_candidate_scan_limit: int = 64
    stop_buffer_bps: float = 2.0
    mss_break_epsilon_bps: float = 0.01
    fixed_r_targets: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0)

    def validate(self) -> "R02Config":
        if not self.pool_tolerances_bps or any(float(x) <= 0 for x in self.pool_tolerances_bps):
            raise ValueError("pool_tolerances_bps must contain positive values")
        if tuple(sorted(set(float(x) for x in self.pool_tolerances_bps))) != tuple(float(x) for x in self.pool_tolerances_bps):
            raise ValueError("pool_tolerances_bps must be sorted unique")
        if self.episode_gap_minutes <= 0:
            raise ValueError("episode_gap_minutes must be positive")
        if self.max_confirmation_minutes <= 0 or self.max_fvg_wait_minutes <= 0:
            raise ValueError("confirmation/wait minutes must be positive")
        if self.exit_censor_minutes <= 0:
            raise ValueError("exit_censor_minutes must be positive")
        if not self.path_horizons_minutes or max(self.path_horizons_minutes) > self.exit_censor_minutes:
            raise ValueError("path horizons must be non-empty and <= exit_censor_minutes")
        if self.target_pool_tolerance_bps <= 0 or self.target_candidate_scan_limit <= 0:
            raise ValueError("target pool settings must be positive")
        if self.stop_buffer_bps < 0:
            raise ValueError("stop_buffer_bps cannot be negative")
        return self


def _pool_token(value: float) -> str:
    return str(float(value)).replace(".", "p")


def _cluster_sorted_prices(prices: Iterable[float], tolerance_bps: float) -> list[np.ndarray]:
    """Transitive adjacent-price clustering at a fixed bp tolerance."""
    arr = np.asarray(list(prices), dtype=float)
    arr = np.sort(arr[np.isfinite(arr)])
    if not len(arr):
        return []
    tol = float(tolerance_bps)
    groups: list[list[float]] = [[float(arr[0])]]
    prev = float(arr[0])
    for value in arr[1:]:
        price = float(value)
        gap_bp = abs(price / prev - 1.0) * 10_000.0 if abs(prev) > EPS else np.inf
        if gap_bp <= tol:
            groups[-1].append(price)
        else:
            groups.append([price])
        prev = price
    return [np.asarray(group, dtype=float) for group in groups]


def _count_price_pools(prices: Iterable[float], tolerance_bps: float) -> int:
    return int(len(_cluster_sorted_prices(prices, tolerance_bps)))


def build_sweep_stages(
    primary_1m: pd.DataFrame,
    classified_lifecycle: pd.DataFrame,
    *,
    config: R02Config | None = None,
    project_timezone: str | int | float | None = "+8",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Collapse level-level first sweeps into unique 1m sweep stages.

    All fields are known by the 1m sweep close.  The function never looks at a
    level whose ``initial_available_time`` is after that close.
    """
    cfg = (config or R02Config()).validate()
    bars = normalize_1m_bars(primary_1m)
    if classified_lifecycle.empty:
        return pd.DataFrame()
    swept = classified_lifecycle.loc[
        pd.to_numeric(classified_lifecycle.get("sweep_pos_1m"), errors="coerce").fillna(-1).ge(0)
    ].copy()
    if swept.empty:
        return pd.DataFrame()
    swept["sweep_pos_1m"] = pd.to_numeric(swept["sweep_pos_1m"], errors="raise").astype(np.int64)
    swept["trade_direction"] = pd.to_numeric(swept["trade_direction"], errors="raise").astype(np.int8)
    swept = swept.sort_values(["sweep_pos_1m", "trade_direction", "level_price", "level_id"], kind="stable")

    low = bars["low"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    grouped = list(swept.groupby(["sweep_pos_1m", "trade_direction"], sort=True, observed=False))
    reporter = ProgressReporter(
        "[r02-sweep-stages]", total=len(grouped), every=max(1, len(grouped) // 100), enabled=show_progress
    )
    for loop_i, ((sweep_pos, direction), group) in enumerate(grouped, start=1):
        reporter.update(loop_i)
        pos = int(sweep_pos)
        direction_i = int(direction)
        if pos < 0 or pos >= len(bars):
            continue
        available_time = bars.index[pos] + pd.Timedelta(minutes=1)
        initial_available = pd.to_datetime(group["initial_available_time"], errors="coerce")
        if (initial_available > available_time).any():
            raise RuntimeError("R02 stage included liquidity not available by sweep close")
        prices = pd.to_numeric(group["level_price"], errors="coerce").dropna().to_numpy(dtype=float)
        if not len(prices):
            continue
        tf_min = pd.to_numeric(group["source_timeframe_min"], errors="coerce")
        order = pd.to_numeric(group["confirmed_order_at_sweep"], errors="coerce").fillna(0)
        age = pd.to_numeric(group.get("age_minutes_since_pivot_at_sweep"), errors="coerce")
        record: dict[str, object] = {
            "sweep_pos_1m": pos,
            "sweep_bar_time_1m": bars.index[pos],
            "sweep_available_time_1m": available_time,
            "trade_direction": direction_i,
            "liquidity_side": "sell_side" if direction_i > 0 else "buy_side",
            "levels_consumed_stage": int(len(group)),
            "distinct_timeframes_stage": int(group["source_timeframe"].astype(str).nunique()),
            "timeframe_signature_stage": "+".join(
                str(x) for x in sorted(group["source_timeframe"].astype(str).unique(), key=lambda name: int(group.loc[group["source_timeframe"].astype(str).eq(name), "source_timeframe_min"].iloc[0]))
            ),
            "min_source_timeframe_min_stage": int(tf_min.min()) if tf_min.notna().any() else -1,
            "max_source_timeframe_min_stage": int(tf_min.max()) if tf_min.notna().any() else -1,
            "order_sum_stage": float(order.sum()),
            "order_ge2_stage": int((order >= 2).sum()),
            "order_ge3_stage": int((order >= 3).sum()),
            "order_ge5_stage": int((order >= 5).sum()),
            "htf_60m_plus_levels_stage": int((tf_min >= 60).sum()),
            "htf_240m_plus_levels_stage": int((tf_min >= 240).sum()),
            "htf_1440m_plus_levels_stage": int((tf_min >= 1440).sum()),
            "external20_levels_stage": int(pd.to_numeric(group.get("external_20_flag", 0), errors="coerce").fillna(0).sum()),
            "external50_levels_stage": int(pd.to_numeric(group.get("external_50_flag", 0), errors="coerce").fillna(0).sum()),
            "clean_levels_stage": int(pd.to_numeric(group.get("clean_sweep_no_prior_touch_flag", 0), errors="coerce").fillna(0).sum()),
            "pretested_levels_stage": int(pd.to_numeric(group.get("pretested_before_sweep_flag", 0), errors="coerce").fillna(0).sum()),
            "old_24h_levels_stage": int(pd.to_numeric(group.get("old_remote_flag_24h", 0), errors="coerce").fillna(0).sum()),
            "old_72h_levels_stage": int(pd.to_numeric(group.get("old_remote_flag_72h", 0), errors="coerce").fillna(0).sum()),
            "mean_age_hours_stage": float(age.mean() / 60.0) if age.notna().any() else np.nan,
            "max_age_hours_stage": float(age.max() / 60.0) if age.notna().any() else np.nan,
            "min_consumed_level_price_stage": float(np.nanmin(prices)),
            "max_consumed_level_price_stage": float(np.nanmax(prices)),
            "sweep_extreme_stage": float(low[pos] if direction_i > 0 else high[pos]),
            # object tuple is intentionally kept internally for causal episode accumulation;
            # research scripts should drop/serialize it before writing CSV.
            "_consumed_level_prices": tuple(float(x) for x in prices),
        }
        for tolerance in cfg.pool_tolerances_bps:
            token = _pool_token(tolerance)
            clusters = _cluster_sorted_prices(prices, tolerance)
            record[f"price_pools_{token}bp_stage"] = int(len(clusters))
            record[f"largest_pool_levels_{token}bp_stage"] = int(max((len(x) for x in clusters), default=0))
        for tf_value in sorted(int(v) for v in tf_min.dropna().unique()):
            record[f"levels_tf_{tf_value}m_stage"] = int((tf_min == tf_value).sum())
        rows.append(record)
    reporter.close()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["sweep_pos_1m", "trade_direction"], kind="stable").reset_index(drop=True)
    out.insert(0, "stage_id", [f"STACK_STAGE_{i+1:08d}" for i in range(len(out))])
    out = attach_session_context(out, "sweep_available_time_1m", project_timezone=project_timezone)
    return out


def build_sweep_episodes(
    stages: pd.DataFrame,
    *,
    config: R02Config | None = None,
) -> pd.DataFrame:
    """Build causal directional sweep episodes and cumulative stack features.

    An episode continues only if the next sweep stage:
    1) is in the same direction,
    2) arrives within ``episode_gap_minutes``, and
    3) extends the prior sweep extreme in the same direction.

    Any opposite-side stage or non-extension starts a new episode.  Every row is
    a *causal episode stage*: cumulative columns only include current/past stages.
    """
    cfg = (config or R02Config()).validate()
    if stages.empty:
        return stages.copy()
    frame = stages.sort_values(["sweep_pos_1m", "trade_direction", "stage_id"], kind="stable").reset_index(drop=True).copy()
    rows: list[dict[str, object]] = []
    episode_number = 0
    current_direction: int | None = None
    last_pos = -10**12
    episode_extreme = np.nan
    episode_start_pos = -1
    episode_start_time = pd.NaT
    cumulative_prices: list[float] = []
    cumulative_levels = 0
    cumulative_tf_names: set[int] = set()
    cumulative_order_sum = 0.0
    cumulative_order_ge2 = 0
    cumulative_order_ge3 = 0
    cumulative_htf60 = 0
    cumulative_htf240 = 0
    cumulative_htf1440 = 0
    cumulative_old24 = 0
    cumulative_old72 = 0
    cumulative_clean = 0
    cumulative_pretested = 0
    stage_no = 0

    def starts_new(row: pd.Series) -> bool:
        nonlocal current_direction, last_pos, episode_extreme
        direction = int(row["trade_direction"])
        pos = int(row["sweep_pos_1m"])
        extreme = float(row["sweep_extreme_stage"])
        if current_direction is None or direction != current_direction:
            return True
        if pos - last_pos > int(cfg.episode_gap_minutes):
            return True
        if direction > 0 and extreme > float(episode_extreme) + EPS:
            return True
        if direction < 0 and extreme < float(episode_extreme) - EPS:
            return True
        return False

    for _, source in frame.iterrows():
        row = source.to_dict()
        if starts_new(source):
            episode_number += 1
            current_direction = int(source["trade_direction"])
            episode_start_pos = int(source["sweep_pos_1m"])
            episode_start_time = pd.Timestamp(source["sweep_bar_time_1m"])
            episode_extreme = float(source["sweep_extreme_stage"])
            cumulative_prices = []
            cumulative_levels = 0
            cumulative_tf_names = set()
            cumulative_order_sum = 0.0
            cumulative_order_ge2 = 0
            cumulative_order_ge3 = 0
            cumulative_htf60 = 0
            cumulative_htf240 = 0
            cumulative_htf1440 = 0
            cumulative_old24 = 0
            cumulative_old72 = 0
            cumulative_clean = 0
            cumulative_pretested = 0
            stage_no = 0
        stage_no += 1
        prices = [float(x) for x in row.get("_consumed_level_prices", ()) if np.isfinite(float(x))]
        cumulative_prices.extend(prices)
        cumulative_levels += int(row["levels_consumed_stage"])
        for column in [name for name in row if name.startswith("levels_tf_") and name.endswith("m_stage")]:
            value = row.get(column, 0)
            if pd.notna(value) and int(value) > 0:
                token = column.removeprefix("levels_tf_").removesuffix("m_stage")
                try:
                    cumulative_tf_names.add(int(token))
                except ValueError:
                    pass
        cumulative_order_sum += float(row["order_sum_stage"])
        cumulative_order_ge2 += int(row["order_ge2_stage"])
        cumulative_order_ge3 += int(row["order_ge3_stage"])
        cumulative_htf60 += int(row["htf_60m_plus_levels_stage"])
        cumulative_htf240 += int(row["htf_240m_plus_levels_stage"])
        cumulative_htf1440 += int(row["htf_1440m_plus_levels_stage"])
        cumulative_old24 += int(row["old_24h_levels_stage"])
        cumulative_old72 += int(row["old_72h_levels_stage"])
        cumulative_clean += int(row["clean_levels_stage"])
        cumulative_pretested += int(row["pretested_levels_stage"])
        direction = int(row["trade_direction"])
        current_extreme = float(row["sweep_extreme_stage"])
        episode_extreme = min(float(episode_extreme), current_extreme) if direction > 0 else max(float(episode_extreme), current_extreme)
        last_pos = int(row["sweep_pos_1m"])
        row.update(
            {
                "episode_id": f"STACK_EP_{episode_number:08d}",
                "episode_stage_no": int(stage_no),
                "episode_start_pos_1m": int(episode_start_pos),
                "episode_start_time_1m": episode_start_time,
                "episode_elapsed_minutes": int(last_pos - episode_start_pos),
                "episode_extreme_so_far": float(episode_extreme),
                "levels_consumed_cum": int(cumulative_levels),
                "distinct_timeframes_cum": int(len(cumulative_tf_names)),
                "timeframe_signature_cum": "+".join(f"{tf}m" for tf in sorted(cumulative_tf_names)),
                "min_source_timeframe_min_cum": int(min(cumulative_tf_names)) if cumulative_tf_names else -1,
                "max_source_timeframe_min_cum": int(max(cumulative_tf_names)) if cumulative_tf_names else -1,
                "order_sum_cum": float(cumulative_order_sum),
                "order_ge2_cum": int(cumulative_order_ge2),
                "order_ge3_cum": int(cumulative_order_ge3),
                "htf_60m_plus_levels_cum": int(cumulative_htf60),
                "htf_240m_plus_levels_cum": int(cumulative_htf240),
                "htf_1440m_plus_levels_cum": int(cumulative_htf1440),
                "old_24h_levels_cum": int(cumulative_old24),
                "old_72h_levels_cum": int(cumulative_old72),
                "clean_levels_cum": int(cumulative_clean),
                "pretested_levels_cum": int(cumulative_pretested),
                "min_consumed_level_price_cum": float(np.nanmin(cumulative_prices)),
                "max_consumed_level_price_cum": float(np.nanmax(cumulative_prices)),
                "_consumed_level_prices_cum": tuple(cumulative_prices),
            }
        )
        first_price = float(cumulative_prices[0]) if cumulative_prices else np.nan
        if np.isfinite(first_price) and abs(first_price) > EPS:
            if direction > 0:
                row["episode_consumption_depth_bp"] = max(0.0, (first_price / float(episode_extreme) - 1.0) * 10_000.0)
            else:
                row["episode_consumption_depth_bp"] = max(0.0, (float(episode_extreme) / first_price - 1.0) * 10_000.0)
        else:
            row["episode_consumption_depth_bp"] = np.nan
        elapsed = max(1, int(row["episode_elapsed_minutes"]) + 1)
        row["levels_consumed_per_min_cum"] = float(cumulative_levels / elapsed)
        for tolerance in cfg.pool_tolerances_bps:
            token = _pool_token(tolerance)
            clusters = _cluster_sorted_prices(cumulative_prices, tolerance)
            row[f"price_pools_{token}bp_cum"] = int(len(clusters))
            row[f"largest_pool_levels_{token}bp_cum"] = int(max((len(x) for x in clusters), default=0))
            row[f"pools_per_min_{token}bp_cum"] = float(len(clusters) / elapsed)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["sweep_pos_1m", "stage_id"], kind="stable").reset_index(drop=True)


def attach_stage_forward_paths(
    primary_1m: pd.DataFrame,
    episode_stages: pd.DataFrame,
    *,
    config: R02Config | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Attach long-horizon *labels* to unique sweep stages from next 1m open."""
    cfg = (config or R02Config()).validate()
    bars = normalize_1m_bars(primary_1m)
    if episode_stages.empty:
        return episode_stages.copy()
    open_ = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    high_range = RangeMinMaxIndex(high)
    low_range = RangeMinMaxIndex(low)
    out = episode_stages.copy().reset_index(drop=True)
    n = len(out)
    extra: dict[str, np.ndarray] = {
        "path_entry_pos_1m": np.full(n, -1, dtype=np.int64),
        "path_entry_time": np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]"),
        "path_entry_price": np.full(n, np.nan, dtype=float),
    }
    for horizon in cfg.path_horizons_minutes:
        h = int(horizon)
        extra[f"path_close_return_{h}m"] = np.full(n, np.nan, dtype=float)
        extra[f"path_mfe_{h}m"] = np.full(n, np.nan, dtype=float)
        extra[f"path_mae_{h}m"] = np.full(n, np.nan, dtype=float)
    reporter = ProgressReporter(
        "[r02-stage-paths]", total=n, every=max(1, n // 100), enabled=show_progress
    )
    for i, row in out.iterrows():
        reporter.update(i + 1)
        sweep_pos = int(row["sweep_pos_1m"])
        entry_pos = sweep_pos + 1
        if entry_pos >= len(bars):
            continue
        direction = int(row["trade_direction"])
        entry = float(open_[entry_pos])
        if entry <= EPS:
            continue
        extra["path_entry_pos_1m"][i] = entry_pos
        extra["path_entry_time"][i] = bars.index[entry_pos].to_datetime64()
        extra["path_entry_price"][i] = entry
        for horizon in cfg.path_horizons_minutes:
            h = int(horizon)
            end_pos = min(len(bars) - 1, entry_pos + h - 1)
            if end_pos < entry_pos:
                continue
            min_low, _ = low_range.query(entry_pos, end_pos)
            _, max_high = high_range.query(entry_pos, end_pos)
            extra[f"path_close_return_{h}m"][i] = direction * (float(close[end_pos]) / entry - 1.0)
            if direction > 0:
                extra[f"path_mfe_{h}m"][i] = max(0.0, max_high / entry - 1.0) if np.isfinite(max_high) else np.nan
                extra[f"path_mae_{h}m"][i] = max(0.0, 1.0 - min_low / entry) if np.isfinite(min_low) else np.nan
            else:
                extra[f"path_mfe_{h}m"][i] = max(0.0, 1.0 - min_low / entry) if np.isfinite(min_low) else np.nan
                extra[f"path_mae_{h}m"][i] = max(0.0, max_high / entry - 1.0) if np.isfinite(max_high) else np.nan
    reporter.close()
    return pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)

def _market_entry_after_signal(
    bars1: pd.DataFrame,
    signal_available_time: pd.Timestamp,
) -> tuple[int, pd.Timestamp, float]:
    pos = int(bars1.index.searchsorted(pd.Timestamp(signal_available_time), side="left"))
    if pos < 0 or pos >= len(bars1):
        return -1, pd.NaT, np.nan
    return pos, pd.Timestamp(bars1.index[pos]), float(bars1["open"].iloc[pos])


def _structural_stop_before_entry(
    low: np.ndarray,
    high: np.ndarray,
    *,
    direction: int,
    start_pos: int,
    end_pos: int,
    buffer_bps: float,
) -> tuple[float, float]:
    """Return the causal structural extreme on a valid in-bounds slice.

    Position-bearing R02/R03 artifacts are tied to the exact 1m bar origin
    used to build them.  A caller with a shifted bar window must not allow
    Python slicing to silently turn an out-of-range interval into an empty
    array (or, worse, a wrong interval via negative indexing).  Invalid or
    all-NaN intervals are therefore rejected explicitly.
    """

    n = min(len(low), len(high))
    if n <= 0:
        return np.nan, np.nan
    left = max(0, int(start_pos))
    right = min(n - 1, int(end_pos))
    if right < left:
        return np.nan, np.nan
    values = low[left : right + 1] if direction > 0 else high[left : right + 1]
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan
    if direction > 0:
        extreme = float(finite.min())
        stop = extreme * (1.0 - float(buffer_bps) / 10_000.0)
    else:
        extreme = float(finite.max())
        stop = extreme * (1.0 + float(buffer_bps) / 10_000.0)
    return extreme, stop


def _dynamic_post_sweep_st_reference(
    pivots: pd.DataFrame,
    exec_bars: pd.DataFrame,
    *,
    direction: int,
    sweep_exec_pos: int,
    end_pos: int,
    break_epsilon: float,
) -> tuple[float, int, int, pd.Timestamp, int]:
    """Find a causal post-sweep short-term swing MSS reference.

    After the liquidity sweep, a new small ST swing can form.  The reference is
    allowed to be created *after* the sweep, but it may only be used from the
    first execution bar whose start time is at or after the pivot's causal
    ``initial_available_time``.  As new post-sweep ST pivots become available,
    the latest one replaces the older local reference.

    Returns ``(price, pivot_pos, confirmed_order, available_time, mss_pos)``.
    ``mss_pos`` is the first close that breaks the then-current causal ST
    reference.  Same-bar pivot confirmation/break is impossible by construction.
    """
    side = "high" if int(direction) > 0 else "low"
    if pivots.empty or sweep_exec_pos >= end_pos:
        return np.nan, -1, 0, pd.NaT, -1
    part = pivots.loc[
        pivots["pivot_side"].astype(str).eq(side)
        & pd.to_numeric(pivots["pivot_pos_htf"], errors="coerce").gt(int(sweep_exec_pos))
        & pd.to_numeric(pivots["pivot_pos_htf"], errors="coerce").lt(int(end_pos)),
        ["pivot_pos_htf", "level_price", "initial_available_time"],
    ].copy()
    if part.empty:
        return np.nan, -1, 0, pd.NaT, -1
    part["pivot_pos_htf"] = pd.to_numeric(part["pivot_pos_htf"], errors="coerce").astype(int)
    part["level_price"] = pd.to_numeric(part["level_price"], errors="coerce")
    part["initial_available_time"] = pd.to_datetime(part["initial_available_time"], errors="coerce")
    part = part.dropna(subset=["level_price", "initial_available_time"]).sort_values(
        ["initial_available_time", "pivot_pos_htf"], kind="stable"
    )
    if part.empty:
        return np.nan, -1, 0, pd.NaT, -1

    exec_start = pd.DatetimeIndex(exec_bars.index)
    close = pd.to_numeric(exec_bars["close"], errors="coerce").to_numpy(dtype=float)
    candidates = list(part.itertuples(index=False))
    ptr = 0
    current = None
    for pos in range(int(sweep_exec_pos) + 1, min(int(end_pos), len(exec_bars) - 1) + 1):
        bar_start = pd.Timestamp(exec_start[pos])
        while ptr < len(candidates) and pd.Timestamp(candidates[ptr].initial_available_time) <= bar_start:
            cand = candidates[ptr]
            # Latest post-sweep pivot is the local structure reference once it is known.
            if current is None or int(cand.pivot_pos_htf) >= int(current.pivot_pos_htf):
                current = cand
            ptr += 1
        if current is None:
            continue
        price = float(current.level_price)
        if int(direction) > 0:
            broken = np.isfinite(close[pos]) and close[pos] >= price * (1.0 + float(break_epsilon))
        else:
            broken = np.isfinite(close[pos]) and close[pos] <= price * (1.0 - float(break_epsilon))
        if broken:
            return (
                price, int(current.pivot_pos_htf), 1,
                pd.Timestamp(current.initial_available_time), int(pos),
            )
    return np.nan, -1, 0, pd.NaT, -1


def build_stack_execution_triggers(
    primary_1m: pd.DataFrame,
    episode_stages: pd.DataFrame,
    *,
    execution_minutes: int,
    base_config: MSS2Config | None = None,
    config: R02Config | None = None,
    reference_modes: tuple[str, ...] = ("structural",),
    include_reclaims: bool = True,
    include_mss_market: bool = True,
    include_mss_fvg: bool = True,
    project_timezone: str | int | float | None = "+8",
    show_progress: bool = False,
) -> pd.DataFrame:
    """Create causal reclaim / MSS entries from each episode stage.

    ``stage_reclaim`` and ``episode_reclaim`` can confirm on the execution bar
    containing the sweep, but the market order is always the next available 1m
    open after that execution bar closes.

    ``recent`` / ``structural`` MSS references are pre-sweep references and must
    already be confirmed before the execution bar containing the sweep begins.
    ``post_sweep_st`` is deliberately different: a new causal short-term swing
    may form after the sweep and becomes eligible only after its right-hand
    confirmation bar has closed.  MSS then requires a later close through the
    latest such known ST reference.  The R02 FVG limit order is cancelled if the
    frozen structural stop is breached *before* the limit fill.
    """
    cfg = (config or R02Config()).validate()
    base_cfg = (base_config or MSS2Config()).validate()
    if episode_stages.empty:
        return pd.DataFrame()
    modes = tuple(dict.fromkeys(str(x).strip().lower() for x in reference_modes if str(x).strip()))
    if any(mode not in {"recent", "structural", "post_sweep_st"} for mode in modes):
        raise ValueError("reference_modes must be recent/structural/post_sweep_st")
    minutes = int(execution_minutes)
    if minutes <= 0:
        raise ValueError("execution_minutes must be positive")

    bars1 = normalize_1m_bars(primary_1m)
    exec_bars = aggregate_bars(bars1, minutes)
    if exec_bars.empty:
        return pd.DataFrame()
    pivots = build_execution_pivots(exec_bars, minutes, base_cfg)
    high_refs = _PivotReferenceIndex(pivots, "high", base_cfg.execution_confirmation_orders)
    low_refs = _PivotReferenceIndex(pivots, "low", base_cfg.execution_confirmation_orders)
    close_index = SegmentThresholdIndex(exec_bars["close"].to_numpy(dtype=float))
    base_low_index = SegmentThresholdIndex(bars1["low"].to_numpy(dtype=float))
    base_high_index = SegmentThresholdIndex(bars1["high"].to_numpy(dtype=float))
    low1 = bars1["low"].to_numpy(dtype=float)
    high1 = bars1["high"].to_numpy(dtype=float)
    high = exec_bars["high"].to_numpy(dtype=float)
    low = exec_bars["low"].to_numpy(dtype=float)
    open_ = exec_bars["open"].to_numpy(dtype=float)
    close = exec_bars["close"].to_numpy(dtype=float)
    tr = _true_range(exec_bars)
    atr_pre = tr.shift(1).rolling(base_cfg.atr_window, min_periods=max(5, base_cfg.atr_window // 2)).mean().to_numpy(dtype=float)
    abs_close_change = np.abs(np.diff(close, prepend=np.nan))
    path_cum = np.nancumsum(np.where(np.isfinite(abs_close_change), abs_close_change, 0.0))
    exec_index = exec_bars.index
    max_bars = max(1, int(np.ceil(cfg.max_confirmation_minutes / minutes)))
    break_eps = float(cfg.mss_break_epsilon_bps) / 10_000.0

    rows: list[dict[str, object]] = []
    stages = episode_stages.sort_values(["sweep_pos_1m", "stage_id"], kind="stable").reset_index(drop=True)
    reporter = ProgressReporter(
        f"[r02-triggers-{minutes}m]", total=len(stages), every=max(1, len(stages) // 100), enabled=show_progress
    )

    def add_market_event(base: dict[str, object], *, trigger_type: str, signal_pos: int, threshold: float, ref_mode: str | None, ref_meta: dict[str, object] | None = None) -> None:
        signal_time = pd.Timestamp(exec_bars["bar_end_time"].iloc[signal_pos])
        entry_pos, entry_time, entry_price = _market_entry_after_signal(bars1, signal_time)
        if entry_pos < 0:
            return
        direction = int(base["trade_direction"])
        episode_start = int(base["episode_start_pos_1m"])
        pre_entry_end = max(episode_start, entry_pos - 1)
        structural_extreme, stop = _structural_stop_before_entry(
            low1, high1, direction=direction, start_pos=episode_start, end_pos=pre_entry_end, buffer_bps=cfg.stop_buffer_bps
        )
        record = dict(base)
        record.update(
            {
                "execution_minutes": minutes,
                "trigger_type": trigger_type,
                "reference_mode": ref_mode or "none",
                "signal_exec_pos": int(signal_pos),
                "signal_bar_time": exec_index[signal_pos],
                "signal_available_time": signal_time,
                "trigger_threshold_price": float(threshold),
                "entry_kind": "market_next_open",
                "entry_fill_flag": 1,
                "entry_pos_1m": int(entry_pos),
                "entry_time": entry_time,
                "entry_price": float(entry_price),
                "structural_extreme_pre_entry": float(structural_extreme),
                "stop_price": float(stop),
                "limit_cancelled_pre_fill_flag": 0,
            }
        )
        if ref_meta:
            record.update(ref_meta)
        rows.append(record)

    for loop_i, source in stages.iterrows():
        reporter.update(loop_i + 1)
        base = source.to_dict()
        sweep_time = pd.Timestamp(base["sweep_bar_time_1m"])
        sweep_exec_pos = int(exec_index.searchsorted(sweep_time, side="right")) - 1
        if sweep_exec_pos < 0 or sweep_exec_pos >= len(exec_bars):
            continue
        direction = int(base["trade_direction"])
        end_pos = min(len(exec_bars) - 1, sweep_exec_pos + max_bars)
        sweep_exec_available = pd.Timestamp(exec_bars["bar_end_time"].iloc[sweep_exec_pos])
        base.update(
            {
                "sweep_exec_pos": int(sweep_exec_pos),
                "sweep_exec_bar_time": exec_index[sweep_exec_pos],
                "sweep_exec_available_time": sweep_exec_available,
            }
        )

        if include_reclaims:
            stage_threshold = float(base["max_consumed_level_price_stage"] if direction > 0 else base["min_consumed_level_price_stage"])
            episode_threshold = float(base["max_consumed_level_price_cum"] if direction > 0 else base["min_consumed_level_price_cum"])
            for trigger_type, threshold in (("stage_reclaim", stage_threshold), ("episode_reclaim", episode_threshold)):
                if direction > 0:
                    signal_pos = close_index.first_geq(sweep_exec_pos, end_pos, threshold)
                else:
                    signal_pos = close_index.first_leq(sweep_exec_pos, end_pos, threshold)
                if signal_pos >= 0:
                    add_market_event(base, trigger_type=trigger_type, signal_pos=signal_pos, threshold=threshold, ref_mode=None)

        if not (include_mss_market or include_mss_fvg):
            continue
        arm_pos = sweep_exec_pos + 1
        if arm_pos > end_pos:
            continue
        reference_known_cutoff = pd.Timestamp(exec_index[sweep_exec_pos])
        for mode in modes:
            ref_index = high_refs if direction > 0 else low_refs
            if mode == "post_sweep_st":
                ref_price, ref_pivot_pos, ref_order, ref_available_time, mss_pos = _dynamic_post_sweep_st_reference(
                    pivots, exec_bars, direction=direction, sweep_exec_pos=sweep_exec_pos,
                    end_pos=end_pos, break_epsilon=break_eps,
                )
            else:
                min_order = 1 if mode == "recent" else 2
                ref_price, ref_pivot_pos, ref_order, ref_available_time = ref_index.latest_before(
                    sweep_pos=sweep_exec_pos,
                    known_time=reference_known_cutoff,
                    min_order=min_order,
                )
                if direction > 0:
                    mss_pos = close_index.first_geq(arm_pos, end_pos, ref_price * (1.0 + break_eps)) if np.isfinite(ref_price) else -1
                else:
                    mss_pos = close_index.first_leq(arm_pos, end_pos, ref_price * (1.0 - break_eps)) if np.isfinite(ref_price) else -1
            if not np.isfinite(ref_price) or mss_pos < 0:
                continue
            pre_atr = float(atr_pre[sweep_exec_pos]) if sweep_exec_pos < len(atr_pre) else np.nan
            if direction > 0:
                sweep_extreme_exec = _range_extreme(low, sweep_exec_pos, mss_pos, "min")
                directional_move = float(close[mss_pos] - sweep_extreme_exec)
                break_distance = float(close[mss_pos] - ref_price)
            else:
                sweep_extreme_exec = _range_extreme(high, sweep_exec_pos, mss_pos, "max")
                directional_move = float(sweep_extreme_exec - close[mss_pos])
                break_distance = float(ref_price - close[mss_pos])
            path_start = max(1, sweep_exec_pos)
            path_len = float(path_cum[mss_pos] - path_cum[path_start - 1]) if mss_pos >= path_start else 0.0
            body = abs(float(close[mss_pos] - open_[mss_pos]))
            candle_range = max(float(high[mss_pos] - low[mss_pos]), EPS)
            fvg_pos, fvg_lower, fvg_upper, fvg_proximal = _first_fvg_in_range(exec_bars, direction, sweep_exec_pos, mss_pos)
            fvg_count, largest_fvg_width, mss_bar_fvg_flag = _fvg_leg_stats(exec_bars, direction, sweep_exec_pos, mss_pos, mss_pos)
            fvg_width = float(fvg_upper - fvg_lower) if fvg_pos >= 0 else np.nan

            # Displacement is intentionally measured as a family of continuous
            # path attributes, not reduced to a hard "strong / weak" formula.
            # This lets downstream research discover non-monotonic payoff
            # regions (for example, medium displacement outperforming extremes).
            leg_slice = slice(int(sweep_exec_pos), int(mss_pos) + 1)
            signed_body = float(direction) * (close[leg_slice] - open_[leg_slice])
            abs_body = np.abs(close[leg_slice] - open_[leg_slice])
            directional_body = np.clip(signed_body, 0.0, None)
            leg_bars = max(1, int(mss_pos - sweep_exec_pos + 1))
            minutes_to_signal = max(int(minutes), int((mss_pos - sweep_exec_pos) * minutes))
            displacement_atr = directional_move / pre_atr if np.isfinite(pre_atr) and pre_atr > EPS else np.nan
            displacement_speed = displacement_atr / minutes_to_signal if np.isfinite(displacement_atr) else np.nan
            max_directional_body_atr = (
                float(np.nanmax(directional_body)) / pre_atr
                if directional_body.size and np.isfinite(pre_atr) and pre_atr > EPS else np.nan
            )
            directional_body_share = (
                float(np.nansum(directional_body)) / float(np.nansum(abs_body))
                if abs_body.size and float(np.nansum(abs_body)) > EPS else np.nan
            )
            leg_range = float(np.nanmax(high[leg_slice]) - np.nanmin(low[leg_slice])) if leg_bars else np.nan

            # The attack leg is a comparison variable only.  We deliberately do
            # NOT require the reversal to be stronger/faster than the move into
            # the sweep extreme.  The latest pre-sweep causal ST opposite pivot
            # supplies a structural attack anchor where available.
            attack_anchor_price, attack_anchor_pos, _, attack_anchor_available = ref_index.latest_before(
                sweep_pos=sweep_exec_pos, known_time=reference_known_cutoff, min_order=1
            )
            attack_move = np.nan
            attack_path = np.nan
            attack_atr = np.nan
            attack_efficiency = np.nan
            attack_speed = np.nan
            attack_minutes = np.nan
            if np.isfinite(attack_anchor_price) and 0 <= int(attack_anchor_pos) < int(sweep_exec_pos):
                if direction > 0:
                    attack_move = float(attack_anchor_price - sweep_extreme_exec)
                else:
                    attack_move = float(sweep_extreme_exec - attack_anchor_price)
                attack_move = attack_move if attack_move > 0 else np.nan
                attack_path = float(path_cum[sweep_exec_pos] - path_cum[int(attack_anchor_pos)])
                attack_atr = attack_move / pre_atr if np.isfinite(attack_move) and np.isfinite(pre_atr) and pre_atr > EPS else np.nan
                attack_efficiency = attack_move / attack_path if np.isfinite(attack_move) and attack_path > EPS else np.nan
                attack_minutes = max(int(minutes), int((sweep_exec_pos - int(attack_anchor_pos)) * minutes))
                attack_speed = attack_atr / attack_minutes if np.isfinite(attack_atr) else np.nan
            reversal_attack_distance_ratio = (
                displacement_atr / attack_atr if np.isfinite(displacement_atr) and np.isfinite(attack_atr) and attack_atr > EPS else np.nan
            )
            reversal_attack_speed_ratio = (
                displacement_speed / attack_speed if np.isfinite(displacement_speed) and np.isfinite(attack_speed) and attack_speed > EPS else np.nan
            )

            ref_meta = {
                "mss_reference_price": float(ref_price),
                "mss_reference_pivot_pos": int(ref_pivot_pos),
                "mss_reference_confirmed_order": int(ref_order),
                "mss_reference_available_time": ref_available_time,
                "mss_reference_known_cutoff": reference_known_cutoff,
                "minutes_to_signal": int((mss_pos - sweep_exec_pos) * minutes),
                "displacement_atr": displacement_atr,
                "displacement_speed_atr_per_min": displacement_speed,
                "displacement_bars": int(leg_bars),
                "displacement_leg_range_atr": leg_range / pre_atr if np.isfinite(leg_range) and np.isfinite(pre_atr) and pre_atr > EPS else np.nan,
                "max_directional_body_atr": max_directional_body_atr,
                "directional_body_share": directional_body_share,
                "break_distance_atr": break_distance / pre_atr if np.isfinite(pre_atr) and pre_atr > EPS else np.nan,
                "path_efficiency": directional_move / path_len if path_len > EPS else np.nan,
                "mss_body_atr": body / pre_atr if np.isfinite(pre_atr) and pre_atr > EPS else np.nan,
                "mss_body_ratio": body / candle_range,
                "fvg_pos": int(fvg_pos),
                "fvg_lower": fvg_lower,
                "fvg_upper": fvg_upper,
                "fvg_proximal": fvg_proximal,
                "fvg_width_atr": fvg_width / pre_atr if np.isfinite(fvg_width) and np.isfinite(pre_atr) and pre_atr > EPS else np.nan,
                "fvg_count_in_leg": int(fvg_count),
                "largest_fvg_width_atr": largest_fvg_width / pre_atr if np.isfinite(largest_fvg_width) and np.isfinite(pre_atr) and pre_atr > EPS else np.nan,
                "mss_bar_fvg_flag": int(mss_bar_fvg_flag),
                "has_displacement_fvg": int(fvg_pos >= 0),
                "fvg_density_per_bar": float(fvg_count) / float(leg_bars),
                "attack_anchor_price": attack_anchor_price,
                "attack_anchor_pivot_pos": int(attack_anchor_pos),
                "attack_anchor_available_time": attack_anchor_available,
                "attack_displacement_atr": attack_atr,
                "attack_path_efficiency": attack_efficiency,
                "attack_speed_atr_per_min": attack_speed,
                "attack_minutes": attack_minutes,
                "reversal_attack_distance_ratio": reversal_attack_distance_ratio,
                "reversal_attack_speed_ratio": reversal_attack_speed_ratio,
                "reversal_weaker_than_attack_flag": int(np.isfinite(reversal_attack_distance_ratio) and reversal_attack_distance_ratio < 1.0),
            }
            if include_mss_market:
                add_market_event(
                    base,
                    trigger_type=f"mss_{mode}_market",
                    signal_pos=int(mss_pos),
                    threshold=float(ref_price),
                    ref_mode=mode,
                    ref_meta=ref_meta,
                )
            if include_mss_fvg and fvg_pos >= 0 and np.isfinite(fvg_proximal):
                signal_time = pd.Timestamp(exec_bars["bar_end_time"].iloc[mss_pos])
                entry_start = int(bars1.index.searchsorted(signal_time, side="left"))
                entry_end = min(len(bars1) - 1, entry_start + int(cfg.max_fvg_wait_minutes) - 1)
                if entry_start > entry_end:
                    continue
                if direction > 0:
                    fill_pos = base_low_index.first_leq(entry_start, entry_end, float(fvg_proximal))
                else:
                    fill_pos = base_high_index.first_geq(entry_start, entry_end, float(fvg_proximal))
                if fill_pos < 0 or not (low1[fill_pos] <= float(fvg_proximal) <= high1[fill_pos]):
                    continue
                episode_start = int(base["episode_start_pos_1m"])
                structural_extreme, stop = _structural_stop_before_entry(
                    low1,
                    high1,
                    direction=direction,
                    start_pos=episode_start,
                    end_pos=max(episode_start, entry_start - 1),
                    buffer_bps=cfg.stop_buffer_bps,
                )
                # Strict causal validity of a resting limit: if the thesis stop is
                # breached on a completed 1m bar before the fill bar, the order is
                # cancelled.  A stop/fill collision on the fill bar is left for the
                # pessimistic outcome resolver (stop may count, target may not).
                cancelled = False
                if fill_pos > entry_start and np.isfinite(stop):
                    if direction > 0:
                        stop_before_fill = base_low_index.first_leq(entry_start, fill_pos - 1, stop)
                    else:
                        stop_before_fill = base_high_index.first_geq(entry_start, fill_pos - 1, stop)
                    cancelled = stop_before_fill >= 0
                if cancelled:
                    continue
                record = dict(base)
                record.update(ref_meta)
                record.update(
                    {
                        "execution_minutes": minutes,
                        "trigger_type": f"mss_{mode}_fvg_limit",
                        "reference_mode": mode,
                        "signal_exec_pos": int(mss_pos),
                        "signal_bar_time": exec_index[mss_pos],
                        "signal_available_time": signal_time,
                        "trigger_threshold_price": float(ref_price),
                        "entry_kind": "fvg_limit",
                        "entry_fill_flag": 1,
                        "entry_pos_1m": int(fill_pos),
                        "entry_time": bars1.index[fill_pos],
                        "entry_price": float(fvg_proximal),
                        "structural_extreme_pre_entry": float(structural_extreme),
                        "stop_price": float(stop),
                        "limit_cancelled_pre_fill_flag": 0,
                    }
                )
                rows.append(record)
    reporter.close()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["entry_pos_1m", "episode_id", "stage_id", "trigger_type"], kind="stable").reset_index(drop=True)
    out.insert(0, "trade_event_id", [f"R02_{minutes}M_TRADE_{i+1:09d}" for i in range(len(out))])
    # First entry per episode/config is the independent statistical unit used by
    # default summaries.  Later stages remain available for path diagnostics.
    group_cols = ["episode_id", "execution_minutes", "trigger_type"]
    out["episode_first_entry_flag"] = (~out.duplicated(group_cols, keep="first")).astype(np.int8)
    out = attach_session_context(out, "entry_time", project_timezone=project_timezone)
    return out


class _DynamicActiveLiquidityBook:
    """Causal active-level book for frozen opposing-liquidity target selection."""

    def __init__(self, lifecycle: pd.DataFrame, *, side: str):
        part = lifecycle.loc[lifecycle["pivot_side"].astype(str).eq(side)].copy()
        part = part.loc[pd.to_numeric(part.get("active_pos_1m"), errors="coerce").fillna(-1).ge(0)].copy()
        self.part = part
        self.prices = np.sort(pd.to_numeric(part.get("level_price"), errors="coerce").dropna().astype(float).unique())
        self.rank = {float(price): i for i, price in enumerate(self.prices)}
        self.all_tree = FenwickTree(len(self.prices))
        self.htf60_tree = FenwickTree(len(self.prices))
        self.htf240_tree = FenwickTree(len(self.prices))
        self.htf1440_tree = FenwickTree(len(self.prices))
        tf_values = sorted(int(v) for v in pd.to_numeric(part.get("source_timeframe_min"), errors="coerce").dropna().unique())
        self.tf_trees = {tf: FenwickTree(len(self.prices)) for tf in tf_values}
        self.additions: dict[int, list[tuple[float, int]]] = {}
        self.removals: dict[int, list[tuple[float, int]]] = {}
        for row in part.itertuples(index=False):
            price = float(row.level_price)
            tf = int(row.source_timeframe_min)
            active_pos = int(row.active_pos_1m)
            self.additions.setdefault(active_pos, []).append((price, tf))
            sweep_pos = int(getattr(row, "sweep_pos_1m", -1))
            if sweep_pos >= 0:
                # Removal happens at next bar start.  At the start of the sweep
                # bar the level was still active; using sweep_pos itself would
                # peek at that bar's future path.
                self.removals.setdefault(sweep_pos + 1, []).append((price, tf))
        self.update_positions = sorted(set(self.additions) | set(self.removals))
        self.pointer = 0
        self.current_pos = -1

    def _update_tree(self, price: float, tf: int, delta: int) -> None:
        if price not in self.rank:
            return
        idx = self.rank[price]
        self.all_tree.add(idx, delta)
        if tf >= 60:
            self.htf60_tree.add(idx, delta)
        if tf >= 240:
            self.htf240_tree.add(idx, delta)
        if tf >= 1440:
            self.htf1440_tree.add(idx, delta)
        if tf in self.tf_trees:
            self.tf_trees[tf].add(idx, delta)

    def advance(self, pos: int) -> None:
        if int(pos) < self.current_pos:
            raise ValueError("active liquidity book requires nondecreasing query positions")
        while self.pointer < len(self.update_positions) and self.update_positions[self.pointer] <= int(pos):
            update_pos = self.update_positions[self.pointer]
            # remove stale first, then add newly confirmed levels at this bar start
            for price, tf in self.removals.get(update_pos, []):
                self._update_tree(price, tf, -1)
            for price, tf in self.additions.get(update_pos, []):
                self._update_tree(price, tf, +1)
            self.pointer += 1
        self.current_pos = int(pos)

    @staticmethod
    def _range_count(tree: FenwickTree, left: int, right: int) -> int:
        return int(tree.range_sum(int(left), int(right)))

    def _nearest_rank(self, price: float, *, above: bool, tree: FenwickTree) -> int:
        n = len(self.prices)
        if n == 0:
            return -1
        if above:
            left = int(np.searchsorted(self.prices, float(price), side="right"))
            if left >= n or self._range_count(tree, left, n) <= 0:
                return -1
            lo, hi = left, n - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if self._range_count(tree, left, mid + 1) > 0:
                    hi = mid
                else:
                    lo = mid + 1
            return int(lo)
        right = int(np.searchsorted(self.prices, float(price), side="left"))
        if right <= 0 or self._range_count(tree, 0, right) <= 0:
            return -1
        lo, hi = 0, right - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._range_count(tree, mid, right) > 0:
                lo = mid
            else:
                hi = mid - 1
        return int(lo)

    def nearest_category(self, price: float, *, above: bool, category: str) -> float:
        tree = {
            "any": self.all_tree,
            "htf60": self.htf60_tree,
            "htf240": self.htf240_tree,
            "htf1440": self.htf1440_tree,
        }[category]
        rank = self._nearest_rank(price, above=above, tree=tree)
        return float(self.prices[rank]) if rank >= 0 else np.nan

    def nearest_pool(
        self,
        price: float,
        *,
        above: bool,
        tolerance_bps: float,
        min_levels: int,
        min_timeframes: int,
        scan_limit: int,
    ) -> tuple[float, int, int]:
        if not len(self.prices):
            return np.nan, 0, 0
        probe = float(price)
        for _ in range(int(scan_limit)):
            rank = self._nearest_rank(probe, above=above, tree=self.all_tree)
            if rank < 0:
                return np.nan, 0, 0
            candidate = float(self.prices[rank])
            tol = float(tolerance_bps) / 10_000.0
            left = int(np.searchsorted(self.prices, candidate * (1.0 - tol), side="left"))
            right = int(np.searchsorted(self.prices, candidate * (1.0 + tol), side="right"))
            count = self._range_count(self.all_tree, left, right)
            tf_count = sum(self._range_count(tree, left, right) > 0 for tree in self.tf_trees.values())
            if count >= int(min_levels) and tf_count >= int(min_timeframes):
                return candidate, int(count), int(tf_count)
            # Step strictly past this active candidate.  np.nextafter avoids
            # skipping a distinct price that is extremely close.
            probe = np.nextafter(candidate, np.inf if above else -np.inf)
        return np.nan, 0, 0


def _first_competing_outcome(
    *,
    direction: int,
    entry_kind: str,
    entry_pos: int,
    end_pos: int,
    stop: float,
    target: float,
    low_index: SegmentThresholdIndex,
    high_index: SegmentThresholdIndex,
) -> tuple[str, int]:
    if not np.isfinite(stop) or not np.isfinite(target):
        return "invalid", -1
    if direction > 0:
        stop_pos = low_index.first_leq(entry_pos, end_pos, float(stop))
        target_start = entry_pos + 1 if entry_kind == "fvg_limit" else entry_pos
        target_pos = high_index.first_geq(target_start, end_pos, float(target)) if target_start <= end_pos else -1
    else:
        stop_pos = high_index.first_geq(entry_pos, end_pos, float(stop))
        target_start = entry_pos + 1 if entry_kind == "fvg_limit" else entry_pos
        target_pos = low_index.first_leq(target_start, end_pos, float(target)) if target_start <= end_pos else -1
    if stop_pos < 0 and target_pos < 0:
        return "censored", -1
    if stop_pos >= 0 and (target_pos < 0 or stop_pos <= target_pos):
        return "stop", int(stop_pos)
    return "target", int(target_pos)


def attach_structural_exit_outcomes(
    primary_1m: pd.DataFrame,
    classified_lifecycle: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    config: R02Config | None = None,
    roundtrip_cost: float = 0.0011,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Freeze opposing-liquidity targets at entry and run competing-risk exits.

    Primary exit semantics:
    - structural stop beyond the sweep/confirmation extreme;
    - no time-based TP;
    - target is an opposing active level/pool selected causally at entry;
    - 7d (default) is only right-censoring; unresolved trades are not force-closed.
    """
    cfg = (config or R02Config()).validate()
    if trades.empty:
        return trades.copy()
    bars = normalize_1m_bars(primary_1m)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    high_index = SegmentThresholdIndex(high)
    low_index = SegmentThresholdIndex(low)
    high_range = RangeMinMaxIndex(high)
    low_range = RangeMinMaxIndex(low)
    buy_book = _DynamicActiveLiquidityBook(classified_lifecycle, side="high")
    sell_book = _DynamicActiveLiquidityBook(classified_lifecycle, side="low")

    out = trades.sort_values(["entry_pos_1m", "trade_event_id"], kind="stable").reset_index(drop=True).copy()
    n = len(out)
    target_names = ["any", "pool2", "pool2tf", "htf60", "htf240", "htf1440"] + [
        f"r{str(float(r)).replace('.', 'p')}" for r in cfg.fixed_r_targets
    ]
    extra: dict[str, np.ndarray] = {
        "risk_price": np.full(n, np.nan, dtype=float),
        "risk_bps": np.full(n, np.nan, dtype=float),
        "valid_risk_flag": np.zeros(n, dtype=np.int8),
    }
    for name in target_names:
        extra[f"target_{name}_price"] = np.full(n, np.nan, dtype=float)
        extra[f"target_{name}_pool_levels"] = np.zeros(n, dtype=np.int16)
        extra[f"target_{name}_pool_timeframes"] = np.zeros(n, dtype=np.int8)
        extra[f"target_{name}_outcome"] = np.full(n, "", dtype=object)
        extra[f"target_{name}_r_multiple"] = np.full(n, np.nan, dtype=float)
        extra[f"target_{name}_censored_flag"] = np.zeros(n, dtype=np.int8)
        extra[f"target_{name}_exit_pos"] = np.full(n, -1, dtype=np.int64)
        extra[f"target_{name}_exit_time"] = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
        extra[f"target_{name}_holding_minutes"] = np.full(n, np.nan, dtype=float)
        extra[f"target_{name}_gross_return"] = np.full(n, np.nan, dtype=float)
        extra[f"target_{name}_net_return_base"] = np.full(n, np.nan, dtype=float)
        extra[f"target_{name}_net_return_cost2x"] = np.full(n, np.nan, dtype=float)
        extra[f"target_{name}_net_return_cost3x"] = np.full(n, np.nan, dtype=float)
    for horizon in cfg.path_horizons_minutes:
        h = int(horizon)
        extra[f"mark_return_{h}m"] = np.full(n, np.nan, dtype=float)
        extra[f"mfe_{h}m"] = np.full(n, np.nan, dtype=float)
        extra[f"mae_{h}m"] = np.full(n, np.nan, dtype=float)

    reporter = ProgressReporter(
        "[r02-structural-exits]", total=n, every=max(1, n // 100), enabled=show_progress
    )
    for i, row in out.iterrows():
        reporter.update(i + 1)
        entry_pos = int(row["entry_pos_1m"])
        if entry_pos < 0 or entry_pos >= len(bars):
            continue
        direction = int(row["trade_direction"])
        entry = float(row["entry_price"])
        stop = float(row["stop_price"])
        risk = (entry - stop) if direction > 0 else (stop - entry)
        extra["risk_price"][i] = risk
        extra["risk_bps"][i] = risk / entry * 10_000.0 if entry > EPS else np.nan
        if not np.isfinite(risk) or risk <= EPS or entry <= EPS:
            continue
        extra["valid_risk_flag"][i] = 1
        book = buy_book if direction > 0 else sell_book
        book.advance(entry_pos)
        above = direction > 0
        target_map: dict[str, tuple[float, int, int]] = {
            "any": (book.nearest_category(entry, above=above, category="any"), 1, 1),
            "htf60": (book.nearest_category(entry, above=above, category="htf60"), 1, 1),
            "htf240": (book.nearest_category(entry, above=above, category="htf240"), 1, 1),
            "htf1440": (book.nearest_category(entry, above=above, category="htf1440"), 1, 1),
        }
        target_map["pool2"] = book.nearest_pool(
            entry, above=above, tolerance_bps=cfg.target_pool_tolerance_bps,
            min_levels=2, min_timeframes=1, scan_limit=cfg.target_candidate_scan_limit,
        )
        target_map["pool2tf"] = book.nearest_pool(
            entry, above=above, tolerance_bps=cfg.target_pool_tolerance_bps,
            min_levels=2, min_timeframes=2, scan_limit=cfg.target_candidate_scan_limit,
        )
        for r_target in cfg.fixed_r_targets:
            token = f"r{str(float(r_target)).replace('.', 'p')}"
            target_map[token] = (entry + direction * float(r_target) * risk, 0, 0)

        end_pos = min(len(bars) - 1, entry_pos + int(cfg.exit_censor_minutes) - 1)
        for name, (target, pool_levels, pool_tfs) in target_map.items():
            extra[f"target_{name}_price"][i] = target
            extra[f"target_{name}_pool_levels"][i] = int(pool_levels)
            extra[f"target_{name}_pool_timeframes"][i] = int(pool_tfs)
            if not np.isfinite(target) or (direction > 0 and target <= entry) or (direction < 0 and target >= entry):
                extra[f"target_{name}_outcome"][i] = "no_target"
                continue
            target_r = direction * (float(target) - entry) / risk
            extra[f"target_{name}_r_multiple"][i] = target_r
            outcome, exit_pos = _first_competing_outcome(
                direction=direction, entry_kind=str(row["entry_kind"]), entry_pos=entry_pos,
                end_pos=end_pos, stop=stop, target=float(target), low_index=low_index, high_index=high_index,
            )
            extra[f"target_{name}_outcome"][i] = outcome
            if outcome == "censored":
                extra[f"target_{name}_censored_flag"][i] = 1
                continue
            if outcome in {"target", "stop"} and exit_pos >= 0:
                exit_price = float(target) if outcome == "target" else stop
                gross = direction * (exit_price / entry - 1.0)
                extra[f"target_{name}_exit_pos"][i] = int(exit_pos)
                extra[f"target_{name}_exit_time"][i] = bars.index[exit_pos].to_datetime64()
                extra[f"target_{name}_holding_minutes"][i] = int(exit_pos - entry_pos + 1)
                extra[f"target_{name}_gross_return"][i] = gross
                extra[f"target_{name}_net_return_base"][i] = gross - float(roundtrip_cost)
                extra[f"target_{name}_net_return_cost2x"][i] = gross - float(roundtrip_cost) * 2.0
                extra[f"target_{name}_net_return_cost3x"][i] = gross - float(roundtrip_cost) * 3.0

        for horizon in cfg.path_horizons_minutes:
            h = int(horizon)
            end_h = min(len(bars) - 1, entry_pos + h - 1)
            min_low, _ = low_range.query(entry_pos, end_h)
            _, max_high = high_range.query(entry_pos, end_h)
            extra[f"mark_return_{h}m"][i] = direction * (float(close[end_h]) / entry - 1.0)
            if direction > 0:
                extra[f"mfe_{h}m"][i] = max(0.0, max_high / entry - 1.0) if np.isfinite(max_high) else np.nan
                extra[f"mae_{h}m"][i] = max(0.0, 1.0 - min_low / entry) if np.isfinite(min_low) else np.nan
            else:
                extra[f"mfe_{h}m"][i] = max(0.0, 1.0 - min_low / entry) if np.isfinite(min_low) else np.nan
                extra[f"mae_{h}m"][i] = max(0.0, max_high / entry - 1.0) if np.isfinite(max_high) else np.nan
    reporter.close()
    return pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1).sort_values(
        ["entry_time", "trade_event_id"], kind="stable"
    ).reset_index(drop=True)

def split_r02_features_and_labels(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split causal-at-entry fields from long-horizon/exit labels."""
    if frame.empty:
        return frame.copy(), frame.copy()
    ids = [name for name in ("trade_event_id", "stage_id", "episode_id") if name in frame.columns]
    label_prefixes = (
        "target_",
        "mark_return_",
        "mfe_",
        "mae_",
    )
    labels = ids + [name for name in frame.columns if name.startswith(label_prefixes)]
    labels = list(dict.fromkeys(labels))
    feature_cols = [name for name in frame.columns if name not in set(labels) or name in ids]
    features = frame.loc[:, feature_cols].copy()
    label_frame = frame.loc[:, labels].copy()
    forbidden = [name for name in features.columns if name.startswith(label_prefixes)]
    if forbidden:
        raise RuntimeError(f"R02 future outcome columns leaked into features: {forbidden}")
    return features, label_frame


def r02_causal_audit(stages: pd.DataFrame, trades: pd.DataFrame) -> dict[str, int]:
    result = {
        "stages": int(len(stages)),
        "trades": int(len(trades)),
        "stage_available_before_sweep_close": 0,
        "signal_before_sweep_exec_available": 0,
        "entry_before_signal_available": 0,
        "mss_reference_available_after_known_cutoff": 0,
        "episode_start_after_stage": 0,
    }
    if not stages.empty:
        stage_available = pd.to_datetime(stages["sweep_available_time_1m"], errors="coerce")
        stage_bar = pd.to_datetime(stages["sweep_bar_time_1m"], errors="coerce") + pd.Timedelta(minutes=1)
        result["stage_available_before_sweep_close"] = int((stage_available < stage_bar).fillna(False).sum())
        if "episode_start_pos_1m" in stages.columns:
            result["episode_start_after_stage"] = int(
                (pd.to_numeric(stages["episode_start_pos_1m"], errors="coerce") > pd.to_numeric(stages["sweep_pos_1m"], errors="coerce")).fillna(False).sum()
            )
    if not trades.empty:
        signal = pd.to_datetime(trades["signal_available_time"], errors="coerce")
        sweep_exec = pd.to_datetime(trades["sweep_exec_available_time"], errors="coerce")
        result["signal_before_sweep_exec_available"] = int((signal < sweep_exec).fillna(False).sum())
        entry = pd.to_datetime(trades["entry_time"], errors="coerce")
        result["entry_before_signal_available"] = int((entry < signal).fillna(False).sum())
        mss = trades.loc[trades["reference_mode"].astype(str).isin(["recent", "structural"])].copy()
        if not mss.empty:
            ref_available = pd.to_datetime(mss.get("mss_reference_available_time"), errors="coerce")
            cutoff = pd.to_datetime(mss.get("mss_reference_known_cutoff"), errors="coerce")
            result["mss_reference_available_after_known_cutoff"] = int((ref_available > cutoff).fillna(False).sum())
    return result
