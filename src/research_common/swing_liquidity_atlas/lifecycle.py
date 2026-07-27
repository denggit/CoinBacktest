#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unconsumed swing-low lifecycle and broad event construction."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter

from .config import AtlasConfig
from .pivots import normalize_primary_bars
from .thresholds import FenwickTree, SegmentThresholdIndex

EPS = 1e-12


def _ns(values: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    return pd.to_datetime(values, errors="coerce").to_numpy(dtype="datetime64[ns]").astype(np.int64)


def _position_time(index: pd.DatetimeIndex, pos: int, bar_delta: pd.Timedelta) -> pd.Timestamp | pd.NaT:
    if int(pos) < 0 or int(pos) >= len(index):
        return pd.NaT
    return pd.Timestamp(index[int(pos)] + bar_delta)


def _first_run_below(close: np.ndarray, start: int, end: int, level: float, run: int) -> int:
    count = 0
    for pos in range(max(0, int(start)), min(len(close) - 1, int(end)) + 1):
        if np.isfinite(close[pos]) and close[pos] < level:
            count += 1
            if count >= int(run):
                return pos
        else:
            count = 0
    return -1


def _count_episodes(
    low_index: SegmentThresholdIndex,
    *,
    start: int,
    end: int,
    enter_threshold: float,
    exit_threshold: float,
) -> tuple[int, int]:
    if start > end:
        return 0, -1
    episodes = 0
    last_enter = -1
    cursor = int(start)
    while cursor <= int(end):
        enter = low_index.first_leq(cursor, end, enter_threshold)
        if enter < 0:
            break
        episodes += 1
        last_enter = enter
        exit_pos = low_index.first_geq(enter + 1, end, exit_threshold)
        if exit_pos < 0:
            break
        cursor = exit_pos + 1
    return episodes, last_enter


def _confirmed_order(level: pd.Series, event_time: pd.Timestamp, orders: Iterable[int]) -> int:
    out = 1
    for order in sorted(int(v) for v in orders):
        value = level.get(f"order_{order}_available_time", pd.NaT)
        if pd.notna(value) and pd.Timestamp(value) <= event_time:
            out = order
    return out


def build_level_lifecycle(
    primary: pd.DataFrame,
    levels: pd.DataFrame,
    config: AtlasConfig,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Track every level from causal activation to its first true sweep.

    A first sweep consumes the stop-liquidity pool.  The subsequent support
    outcome is recorded independently as reclaim/acceptance/unresolved; a
    reclaimed level is not put back into the *unconsumed stop* pool.
    """

    cfg = config.validate()
    bars = normalize_primary_bars(primary)
    if levels.empty:
        return pd.DataFrame()
    index = pd.DatetimeIndex(bars.index)
    bar_delta = pd.Timedelta(minutes=1)
    index_ns = _ns(index)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    low_index = SegmentThresholdIndex(low)
    close_index = SegmentThresholdIndex(close)

    reporter = ProgressReporter(
        label="[atlas] level lifecycle",
        total=len(levels),
        every=max(1, len(levels) // 200),
        enabled=bool(show_progress),
    )
    rows: list[dict[str, object]] = []
    n = len(index)
    for ordinal, level in enumerate(levels.itertuples(index=False), start=1):
        level_price = float(level.level_price)
        available_time = pd.Timestamp(level.initial_available_time)
        active_pos = int(np.searchsorted(index_ns, available_time.value, side="left"))
        if active_pos >= n:
            # A pivot can become known exactly when the final source bar closes.
            # It is part of the causal level universe even though no later bar
            # exists to approach or sweep it inside this dataset.
            row = dict(level._asdict())
            row.update(
                {
                    "active_pos": n,
                    "active_bar_time": pd.NaT,
                    "approach_pos": -1,
                    "approach_available_time": pd.NaT,
                    "touch_pos": -1,
                    "touch_available_time": pd.NaT,
                    "sweep_pos": -1,
                    "sweep_available_time": pd.NaT,
                    "first_reclaim_pos": -1,
                    "first_reclaim_available_time": pd.NaT,
                    "accept_depth_pos": -1,
                    "accept_closes_pos": -1,
                    "stop_liquidity_state": "active_unconsumed",
                    "support_resolution_180m": "unswept_active_at_end",
                    "age_minutes_at_approach": np.nan,
                    "age_minutes_at_touch": np.nan,
                    "age_minutes_at_sweep": np.nan,
                    "approach_episode_count_before_sweep": 0,
                    "touch_episode_count_before_sweep": 0,
                    "last_approach_pos_before_sweep": -1,
                    "last_touch_pos_before_sweep": -1,
                    "minutes_since_last_touch_at_sweep": np.nan,
                    "sweep_depth_bp": np.nan,
                    "sweep_close_vs_level_bp": np.nan,
                    "max_penetration_bp_180": np.nan,
                    "reclaim_lag_minutes": np.nan,
                    "confirmed_order_at_sweep": _confirmed_order(
                        pd.Series(level._asdict()), pd.Timestamp(index[-1] + bar_delta), cfg.confirmation_orders
                    ),
                }
            )
            for horizon in cfg.forward_horizons:
                row[f"close_revisited_level_by_{int(horizon)}m"] = False
                row[f"clean_reclaim_by_{int(horizon)}m"] = False
            rows.append(row)
            reporter.update(ordinal)
            continue

        approach_threshold = level_price * (1.0 + cfg.approach_distance_bp / 10_000.0)
        touch_threshold = level_price * (1.0 + cfg.touch_distance_bp / 10_000.0)
        sweep_threshold = level_price * (1.0 - cfg.sweep_epsilon_bp / 10_000.0)
        approach_pos = low_index.first_leq(active_pos, n - 1, approach_threshold)
        touch_pos = low_index.first_leq(active_pos, n - 1, touch_threshold)
        sweep_pos = low_index.first_leq(active_pos, n - 1, sweep_threshold)
        pre_sweep_end = (sweep_pos - 1) if sweep_pos >= 0 else (n - 1)
        approach_episodes, last_approach_pos = _count_episodes(
            low_index,
            start=active_pos,
            end=pre_sweep_end,
            enter_threshold=approach_threshold,
            exit_threshold=approach_threshold * (1.0 + 1e-10),
        )
        touch_episodes, last_touch_pos = _count_episodes(
            low_index,
            start=active_pos,
            end=pre_sweep_end,
            enter_threshold=touch_threshold,
            exit_threshold=approach_threshold,
        )

        first_reclaim_pos = -1
        accept_depth_pos = -1
        accept_closes_pos = -1
        resolution_class = "unswept_active_at_end"
        first_accept = -1
        max_penetration_bp_180 = np.nan
        sweep_depth_bp = np.nan
        close_vs_level_bp = np.nan
        if sweep_pos >= 0:
            sweep_depth_bp = max((level_price - low[sweep_pos]) / level_price * 10_000.0, 0.0)
            close_vs_level_bp = (close[sweep_pos] / level_price - 1.0) * 10_000.0
            resolution_end = min(n - 1, sweep_pos + int(cfg.resolution_horizon_bars))
            if close[sweep_pos] >= level_price:
                first_reclaim_pos = sweep_pos
            else:
                first_reclaim_pos = close_index.first_geq(sweep_pos + 1, n - 1, level_price)
            accept_depth_pos = low_index.first_leq(
                sweep_pos,
                resolution_end,
                level_price * (1.0 - cfg.acceptance_depth_bp / 10_000.0),
            )
            accept_closes_pos = _first_run_below(
                close,
                sweep_pos,
                resolution_end,
                level_price,
                int(cfg.acceptance_consecutive_closes),
            )
            min_low = float(np.nanmin(low[sweep_pos : resolution_end + 1]))
            max_penetration_bp_180 = max((level_price - min_low) / level_price * 10_000.0, 0.0)
            reclaim_lag = first_reclaim_pos - sweep_pos if first_reclaim_pos >= 0 else np.inf
            accept_candidates = [value for value in (accept_depth_pos, accept_closes_pos) if value >= 0]
            first_accept = min(accept_candidates) if accept_candidates else -1
            if first_reclaim_pos == sweep_pos:
                resolution_class = "same_bar_reclaim"
            elif first_reclaim_pos >= 0 and (first_accept < 0 or first_reclaim_pos < first_accept):
                if reclaim_lag <= 5:
                    resolution_class = "reclaim_by_5m"
                elif reclaim_lag <= 30:
                    resolution_class = "reclaim_by_30m"
                elif reclaim_lag <= cfg.resolution_horizon_bars:
                    resolution_class = "reclaim_by_180m"
                else:
                    resolution_class = "late_reclaim_after_180m"
            elif first_accept >= 0:
                resolution_class = "accepted_below"
            else:
                resolution_class = "unresolved_180m"

        sweep_available = _position_time(index, sweep_pos, bar_delta)
        event_time_for_order = pd.Timestamp(sweep_available) if pd.notna(sweep_available) else pd.Timestamp(index[-1] + bar_delta)
        level_series = pd.Series(level._asdict())
        row = dict(level._asdict())
        row.update(
            {
                "active_pos": active_pos,
                "active_bar_time": index[active_pos],
                "approach_pos": approach_pos,
                "approach_available_time": _position_time(index, approach_pos, bar_delta),
                "touch_pos": touch_pos,
                "touch_available_time": _position_time(index, touch_pos, bar_delta),
                "sweep_pos": sweep_pos,
                "sweep_available_time": sweep_available,
                "first_reclaim_pos": first_reclaim_pos,
                "first_reclaim_available_time": _position_time(index, first_reclaim_pos, bar_delta),
                "accept_depth_pos": accept_depth_pos,
                "accept_closes_pos": accept_closes_pos,
                "stop_liquidity_state": "consumed_first_sweep" if sweep_pos >= 0 else "active_unconsumed",
                "support_resolution_180m": resolution_class,
                "age_minutes_at_approach": (approach_pos - active_pos) if approach_pos >= 0 else np.nan,
                "age_minutes_at_touch": (touch_pos - active_pos) if touch_pos >= 0 else np.nan,
                "age_minutes_at_sweep": (sweep_pos - active_pos) if sweep_pos >= 0 else np.nan,
                "approach_episode_count_before_sweep": int(approach_episodes),
                "touch_episode_count_before_sweep": int(touch_episodes),
                "last_approach_pos_before_sweep": int(last_approach_pos),
                "last_touch_pos_before_sweep": int(last_touch_pos),
                "minutes_since_last_touch_at_sweep": (sweep_pos - last_touch_pos) if sweep_pos >= 0 and last_touch_pos >= 0 else np.nan,
                "sweep_depth_bp": sweep_depth_bp,
                "sweep_close_vs_level_bp": close_vs_level_bp,
                "max_penetration_bp_180": max_penetration_bp_180,
                "reclaim_lag_minutes": (first_reclaim_pos - sweep_pos) if first_reclaim_pos >= 0 and sweep_pos >= 0 else np.nan,
                "confirmed_order_at_sweep": _confirmed_order(level_series, event_time_for_order, cfg.confirmation_orders),
            }
        )
        for horizon in cfg.forward_horizons:
            revisited = bool(
                sweep_pos >= 0 and first_reclaim_pos >= sweep_pos and first_reclaim_pos - sweep_pos <= int(horizon)
            )
            clean_reclaim = bool(
                revisited
                and (first_reclaim_pos == sweep_pos or first_accept < 0 or first_reclaim_pos < first_accept)
            )
            row[f"close_revisited_level_by_{int(horizon)}m"] = revisited
            row[f"clean_reclaim_by_{int(horizon)}m"] = clean_reclaim
        rows.append(row)
        reporter.update(ordinal)
    reporter.close()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if out["level_id"].duplicated().any():
        raise RuntimeError("duplicate level_id in lifecycle")
    valid = out["sweep_pos"].ge(0)
    if valid.any() and (
        pd.to_datetime(out.loc[valid, "initial_available_time"]) > pd.to_datetime(out.loc[valid, "sweep_available_time"])
    ).any():
        raise RuntimeError("a level was swept before it became causally available")
    return out.sort_values(["initial_available_time", "level_id"], kind="mergesort").reset_index(drop=True)


def build_event_table(lifecycle: pd.DataFrame, primary: pd.DataFrame, config: AtlasConfig) -> pd.DataFrame:
    cfg = config.validate()
    if lifecycle.empty:
        return pd.DataFrame()
    bars = normalize_primary_bars(primary)
    index = pd.DatetimeIndex(bars.index)
    bar_delta = pd.Timedelta(minutes=1)
    stages = (
        ("approach", "approach_pos"),
        ("touch", "touch_pos"),
        ("sweep", "sweep_pos"),
        ("reclaim", "first_reclaim_pos"),
    )
    rows: list[dict[str, object]] = []
    base_columns = [
        "level_id", "source_timeframe", "source_timeframe_min", "pivot_time", "level_price",
        "initial_available_time", "pivot_range_bp", "pivot_close_location",
        "pivot_lower_wick_fraction", "confirmation_reaction_close_bp", "confirmation_reaction_high_bp",
        "left_low_gap_3_bp", "left_low_gap_8_bp", "left_low_gap_20_bp",
        "left_high_range_3_bp", "left_high_range_8_bp", "left_high_range_20_bp",
        "pivot_notional_vs_past20", "pivot_trades_count_vs_past20", "pivot_delta_ratio",
        "approach_episode_count_before_sweep", "touch_episode_count_before_sweep",
        "minutes_since_last_touch_at_sweep", "sweep_depth_bp", "sweep_close_vs_level_bp",
        "support_resolution_180m", "reclaim_lag_minutes", "confirmed_order_at_sweep",
        "max_penetration_bp_180",
        "close_revisited_level_by_5m", "close_revisited_level_by_15m",
        "close_revisited_level_by_30m", "close_revisited_level_by_60m",
        "close_revisited_level_by_180m",
        "clean_reclaim_by_5m", "clean_reclaim_by_15m",
        "clean_reclaim_by_30m", "clean_reclaim_by_60m", "clean_reclaim_by_180m",
    ]
    for row in lifecycle.itertuples(index=False):
        source = row._asdict()
        for stage, pos_column in stages:
            raw_pos = source.get(pos_column, -1)
            pos = int(raw_pos) if pd.notna(raw_pos) else -1
            if pos < 0 or pos >= len(index):
                continue
            event_available = index[pos] + bar_delta
            record = {name: source.get(name, np.nan) for name in base_columns}
            record.update(
                {
                    "event_id": f"SL_{int(source['level_id']):08d}_{stage.upper()}",
                    "event_stage": stage,
                    "event_pos": pos,
                    "event_bar_time": index[pos],
                    "event_available_time": event_available,
                    "level_age_minutes": pos - int(source["active_pos"]),
                    "distance_low_to_level_bp": (float(bars["low"].iloc[pos]) / float(source["level_price"]) - 1.0) * 10_000.0,
                    "distance_close_to_level_bp": (float(bars["close"].iloc[pos]) / float(source["level_price"]) - 1.0) * 10_000.0,
                }
            )
            level_series = pd.Series(source)
            record["confirmed_order_at_event"] = _confirmed_order(level_series, event_available, cfg.confirmation_orders)
            rows.append(record)
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    if events["event_id"].duplicated().any():
        raise RuntimeError("duplicate event_id")
    return events.sort_values(["event_pos", "event_stage", "level_id"], kind="mergesort").reset_index(drop=True)


def attach_active_confluence(events: pd.DataFrame, lifecycle: pd.DataFrame, config: AtlasConfig) -> pd.DataFrame:
    """Attach active unconsumed-level counts with an O((L+E)logL) sweep line."""

    cfg = config.validate()
    if events.empty or lifecycle.empty:
        return events.copy()
    out = events.copy().sort_values(["event_pos", "event_id"], kind="mergesort").reset_index(drop=True)
    levels = lifecycle[["level_id", "level_price", "source_timeframe", "active_pos", "sweep_pos"]].copy()
    prices = np.sort(levels["level_price"].dropna().unique().astype(float))
    rank = {float(value): int(i) for i, value in enumerate(prices)}
    timeframes = sorted(levels["source_timeframe"].astype(str).unique())
    total_tree = FenwickTree(len(prices))
    tf_trees = {tf: FenwickTree(len(prices)) for tf in timeframes}

    additions: dict[int, list[tuple[float, str]]] = defaultdict(list)
    removals: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for row in levels.itertuples(index=False):
        price = float(row.level_price)
        tf = str(row.source_timeframe)
        additions[int(row.active_pos)].append((price, tf))
        sweep_pos = int(row.sweep_pos)
        if sweep_pos >= 0:
            removals[sweep_pos + 1].append((price, tf))

    update_positions = sorted(set(additions) | set(removals))
    update_pointer = 0
    result = {float(bp): np.zeros(len(out), dtype=np.int32) for bp in cfg.confluence_tolerances_bp}
    tf_result = {float(bp): np.zeros(len(out), dtype=np.int8) for bp in cfg.confluence_tolerances_bp}
    for event_index, event in enumerate(out.itertuples(index=False)):
        event_pos = int(event.event_pos)
        while update_pointer < len(update_positions) and update_positions[update_pointer] <= event_pos:
            position = update_positions[update_pointer]
            for price, tf in removals.get(position, []):
                idx = rank[price]
                total_tree.add(idx, -1)
                tf_trees[tf].add(idx, -1)
            for price, tf in additions.get(position, []):
                idx = rank[price]
                total_tree.add(idx, 1)
                tf_trees[tf].add(idx, 1)
            update_pointer += 1
        center = float(event.level_price)
        for bp in cfg.confluence_tolerances_bp:
            tolerance = float(bp) / 10_000.0
            left = int(np.searchsorted(prices, center * (1.0 - tolerance), side="left"))
            right = int(np.searchsorted(prices, center * (1.0 + tolerance), side="right"))
            result[float(bp)][event_index] = total_tree.range_sum(left, right)
            tf_result[float(bp)][event_index] = sum(tree.range_sum(left, right) > 0 for tree in tf_trees.values())

    for bp in cfg.confluence_tolerances_bp:
        token = str(float(bp)).replace(".", "p")
        out[f"active_level_count_{token}bp"] = result[float(bp)]
        out[f"active_timeframe_count_{token}bp"] = tf_result[float(bp)]
    return out
