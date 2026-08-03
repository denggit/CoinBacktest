#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Target-before-stop replay for R11 short trades toward lower liquidity pools."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter
from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex
from src.research_common.swing_liquidity_zone_study.outcomes import RangeMinMaxIndex

from .config import LiquidityMagnetConfig, stop_model_definitions


def attach_risk_frontier_outcomes(
    candidates: pd.DataFrame,
    primary: pd.DataFrame,
    config: LiquidityMagnetConfig,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    cfg = config.validate()
    if candidates.empty:
        return pd.DataFrame()
    bars = normalize_primary_bars(primary)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    high_index = SegmentThresholdIndex(high)
    low_index = SegmentThresholdIndex(low)
    high_range = RangeMinMaxIndex(high)
    low_range = RangeMinMaxIndex(low)
    definitions = stop_model_definitions()
    total = len(candidates) * len(definitions)
    reporter = ProgressReporter(
        label="[r11] target/stop frontier",
        total=total,
        every=max(1, total // 200),
        enabled=bool(show_progress),
    )
    rows: list[dict[str, object]] = []
    done = 0
    cost_1x = 2.0 * (float(cfg.fee_rate_per_side) + float(cfg.slippage_rate_per_side))
    cost_2x = cost_1x * float(cfg.stressed_cost_multiplier)
    for candidate in candidates.itertuples(index=False):
        source = candidate._asdict()
        entry_pos = int(source["entry_pos"])
        entry = float(source["entry_price"])
        target = float(source["front_run_target_price"])
        end = min(len(bars) - 1, entry_pos + int(cfg.horizon_minutes) - 1)
        min_low, _ = low_range.query(entry_pos, end)
        _, max_high = high_range.query(entry_pos, end)
        favorable_bp = (entry - min_low) / entry * 10_000.0 if np.isfinite(min_low) and entry > 0 else np.nan
        adverse_bp = (max_high / entry - 1.0) * 10_000.0 if np.isfinite(max_high) and entry > 0 else np.nan
        for definition in definitions:
            model = str(definition["stop_model"])
            if model == "EQUAL_DISTANCE":
                stop = float(source["stop_equal_distance"])
            elif model == "LOCAL_HIGH_15M":
                stop = float(source["stop_local_high_15m"])
            elif model == "LOCAL_HIGH_60M":
                stop = float(source["stop_local_high_60m"])
            else:
                raise ValueError(f"unsupported stop model: {model}")
            valid = (
                0 <= entry_pos <= end < len(bars)
                and np.isfinite(entry)
                and np.isfinite(target)
                and np.isfinite(stop)
                and entry > target > 0
                and stop > entry
            )
            target_pos = -1
            stop_pos = -1
            outcome = "INVALID"
            exit_pos = -1
            exit_price = np.nan
            if valid:
                target_pos = low_index.first_leq(entry_pos, end, target)
                stop_pos = high_index.first_geq(entry_pos, end, stop)
                if target_pos >= 0 and stop_pos >= 0 and target_pos == stop_pos:
                    outcome = "STOP_CONSERVATIVE_SAME_BAR"
                    exit_pos = stop_pos
                    exit_price = stop
                elif stop_pos >= 0 and (target_pos < 0 or stop_pos < target_pos):
                    outcome = "STOP"
                    exit_pos = stop_pos
                    exit_price = stop
                elif target_pos >= 0:
                    outcome = "TARGET"
                    exit_pos = target_pos
                    exit_price = target
                else:
                    outcome = "TIME"
                    exit_pos = end
                    exit_price = float(close[end])
            gross = (entry - exit_price) / entry if valid and np.isfinite(exit_price) and entry > 0 else np.nan
            stop_distance_bp = (stop / entry - 1.0) * 10_000.0 if valid else np.nan
            target_distance_bp = (entry / target - 1.0) * 10_000.0 if valid else np.nan
            row = dict(source)
            row.update(
                {
                    "stop_model": model,
                    "stop_price": stop,
                    "stop_distance_bp": stop_distance_bp,
                    "target_distance_bp": target_distance_bp,
                    "nominal_reward_risk": target_distance_bp / stop_distance_bp if np.isfinite(stop_distance_bp) and stop_distance_bp > 0 else np.nan,
                    "horizon_end_pos": int(end),
                    "target_hit_pos": int(target_pos),
                    "stop_hit_pos": int(stop_pos),
                    "outcome": outcome,
                    "exit_pos": int(exit_pos),
                    "exit_time": bars.index[exit_pos] if 0 <= exit_pos < len(bars) else pd.NaT,
                    "exit_price": exit_price,
                    "gross_return": gross,
                    "net_return_1x_cost": gross - cost_1x if np.isfinite(gross) else np.nan,
                    "net_return_2x_cost": gross - cost_2x if np.isfinite(gross) else np.nan,
                    "target_before_stop": bool(outcome == "TARGET"),
                    "stopped": bool(outcome.startswith("STOP")),
                    "time_exit": bool(outcome == "TIME"),
                    "mfe_short_bp": favorable_bp,
                    "mae_short_bp": adverse_bp,
                }
            )
            rows.append(row)
            done += 1
            reporter.update(done)
    reporter.close()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["entry_time", "distance_band_bp", "pool_event_id", "stop_model"],
        kind="mergesort",
    ).reset_index(drop=True)
