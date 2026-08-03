#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Conservative next-open long/short replay after causal R12 checkpoints."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter
from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex
from src.research_common.swing_liquidity_zone_study.outcomes import RangeMinMaxIndex

from .config import PostSweepAcceptanceConfig


def _trade_result(
    *,
    direction: str,
    entry_pos: int,
    end_pos: int,
    entry: float,
    stop: float,
    target: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    high_index: SegmentThresholdIndex,
    low_index: SegmentThresholdIndex,
) -> tuple[str, int, float, int, int]:
    if direction == "LONG":
        target_pos = high_index.first_geq(entry_pos, end_pos, target)
        stop_pos = low_index.first_leq(entry_pos, end_pos, stop)
    else:
        target_pos = low_index.first_leq(entry_pos, end_pos, target)
        stop_pos = high_index.first_geq(entry_pos, end_pos, stop)
    if target_pos >= 0 and stop_pos >= 0 and target_pos == stop_pos:
        return "STOP_CONSERVATIVE_SAME_BAR", stop_pos, stop, target_pos, stop_pos
    if stop_pos >= 0 and (target_pos < 0 or stop_pos < target_pos):
        return "STOP", stop_pos, stop, target_pos, stop_pos
    if target_pos >= 0:
        return "TARGET", target_pos, target, target_pos, stop_pos
    return "TIME", end_pos, float(close[end_pos]), target_pos, stop_pos


def attach_checkpoint_outcomes(
    checkpoints: pd.DataFrame,
    primary: pd.DataFrame,
    config: PostSweepAcceptanceConfig,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    cfg = config.validate()
    if checkpoints.empty:
        return checkpoints.copy()
    bars = normalize_primary_bars(primary)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    high_index = SegmentThresholdIndex(high)
    low_index = SegmentThresholdIndex(low)
    high_range = RangeMinMaxIndex(high)
    low_range = RangeMinMaxIndex(low)
    cost_1x = 2.0 * (float(cfg.fee_rate_per_side) + float(cfg.slippage_rate_per_side))
    cost_2x = cost_1x * float(cfg.stressed_cost_multiplier)
    rows: list[dict[str, object]] = []
    total = len(checkpoints) * 2
    reporter = ProgressReporter(
        label="[r12] rejection/acceptance replay",
        total=total,
        every=max(1, total // 200),
        enabled=bool(show_progress),
    )
    done = 0
    n = len(bars)
    for item in checkpoints.itertuples(index=False):
        source = item._asdict()
        entry_pos = int(source["entry_pos"])
        entry = float(source["entry_price"])
        end_pos = min(n - 1, entry_pos + int(cfg.horizon_minutes) - 1)
        min_low, _ = low_range.query(entry_pos, end_pos)
        _, max_high = high_range.query(entry_pos, end_pos)
        for direction in ("LONG", "SHORT"):
            if direction == "LONG":
                raw_stop = float(source["path_low_visible"]) * (1.0 - float(cfg.stop_buffer_bp) / 10_000.0)
                stop = min(raw_stop, entry * (1.0 - float(cfg.stop_buffer_bp) / 10_000.0))
                stop_distance = entry - stop
                mfe_bp = (max_high / entry - 1.0) * 10_000.0 if entry > 0 and np.isfinite(max_high) else np.nan
                mae_bp = (entry / min_low - 1.0) * 10_000.0 if entry > 0 and np.isfinite(min_low) and min_low > 0 else np.nan
            else:
                raw_stop = float(source["path_high_visible"]) * (1.0 + float(cfg.stop_buffer_bp) / 10_000.0)
                stop = max(raw_stop, entry * (1.0 + float(cfg.stop_buffer_bp) / 10_000.0))
                stop_distance = stop - entry
                mfe_bp = (entry / min_low - 1.0) * 10_000.0 if entry > 0 and np.isfinite(min_low) and min_low > 0 else np.nan
                mae_bp = (max_high / entry - 1.0) * 10_000.0 if entry > 0 and np.isfinite(max_high) else np.nan
            valid = (
                0 <= entry_pos <= end_pos < n
                and np.isfinite(entry)
                and entry > 0
                and np.isfinite(stop)
                and stop_distance > 0
            )
            base = dict(source)
            base.update(
                {
                    "trade_direction": direction,
                    "natural_stop_price": stop,
                    "natural_stop_distance_bp": stop_distance / entry * 10_000.0 if valid else np.nan,
                    "mfe_bp": mfe_bp,
                    "mae_bp": mae_bp,
                    "horizon_end_pos": int(end_pos),
                }
            )
            for r_multiple in cfg.target_r_multiples:
                token = str(float(r_multiple)).replace(".", "p")
                target = entry + float(r_multiple) * stop_distance if direction == "LONG" else entry - float(r_multiple) * stop_distance
                outcome = "INVALID"
                exit_pos = -1
                exit_price = np.nan
                target_pos = -1
                stop_pos = -1
                if valid and target > 0:
                    outcome, exit_pos, exit_price, target_pos, stop_pos = _trade_result(
                        direction=direction,
                        entry_pos=entry_pos,
                        end_pos=end_pos,
                        entry=entry,
                        stop=stop,
                        target=target,
                        high=high,
                        low=low,
                        close=close,
                        high_index=high_index,
                        low_index=low_index,
                    )
                gross = (
                    (exit_price / entry - 1.0) if direction == "LONG" else (entry - exit_price) / entry
                ) if valid and np.isfinite(exit_price) else np.nan
                risk_return = stop_distance / entry if valid else np.nan
                gross_r = gross / risk_return if np.isfinite(gross) and np.isfinite(risk_return) and risk_return > 0 else np.nan
                net_1x_r = (gross - cost_1x) / risk_return if np.isfinite(gross) and np.isfinite(risk_return) and risk_return > 0 else np.nan
                net_2x_r = (gross - cost_2x) / risk_return if np.isfinite(gross) and np.isfinite(risk_return) and risk_return > 0 else np.nan
                base.update(
                    {
                        f"r{token}_target_price": target,
                        f"r{token}_outcome": outcome,
                        f"r{token}_exit_pos": int(exit_pos),
                        f"r{token}_target_hit_pos": int(target_pos),
                        f"r{token}_stop_hit_pos": int(stop_pos),
                        f"r{token}_target_before_stop": bool(outcome == "TARGET"),
                        f"r{token}_stopped": bool(outcome.startswith("STOP")),
                        f"r{token}_gross_return": gross,
                        f"r{token}_gross_r": gross_r,
                        f"r{token}_net_1x_r": net_1x_r,
                        f"r{token}_net_2x_r": net_2x_r,
                        f"r{token}_exit_time": bars.index[exit_pos] if 0 <= exit_pos < n else pd.NaT,
                    }
                )
            rows.append(base)
            done += 1
            reporter.update(done)
    reporter.close()
    return pd.DataFrame(rows).sort_values(
        ["entry_time", "zone_event_id", "checkpoint_minutes", "trade_direction"],
        kind="mergesort",
    ).reset_index(drop=True)
