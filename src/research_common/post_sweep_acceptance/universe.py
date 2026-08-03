#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal post-sweep checkpoints and rejection/acceptance state features."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter
from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars

from .config import PostSweepAcceptanceConfig, state_direction

_TIME_COLUMNS = (
    "event_bar_time",
    "event_available_time",
    "zone_latest_level_available_time",
    "zone_earliest_pivot_time",
    "zone_latest_pivot_time",
    "zone_member_structure_available_time_max",
    "entry_reference_time",
    "r09_entry_time",
)


def _read_csv(path: Path, *, compression: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path, compression=compression, low_memory=False)
    for name in _TIME_COLUMNS:
        if name in frame.columns:
            frame[name] = pd.to_datetime(frame[name], errors="coerce")
    return frame


def resolve_r09_dir(path: Path) -> Path:
    if path.exists():
        return path
    alternatives = (
        path.parent / "09_structured_swing_stop_pool_hypotheses_r09",
        path.parent / "structured_swing_stop_pool_hypotheses_r09",
    )
    for candidate in alternatives:
        if candidate.exists():
            return candidate
    return path


def load_r09_zone_events(r09_dir: Path) -> tuple[pd.DataFrame, str]:
    """Load R09 zone features and sweep-time release labels without importing R09."""
    directory = resolve_r09_dir(r09_dir)
    feature_path = directory / "16_zone_feature_table.csv.gz"
    label_path = directory / "17_zone_label_table.csv.gz"
    if not feature_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"R09 zone cache missing: {feature_path} / {label_path}. Run full R09 first."
        )
    features = _read_csv(feature_path, compression="gzip")
    labels = _read_csv(label_path, compression="gzip")
    if features.empty or labels.empty:
        raise RuntimeError("R09 zone feature/label cache is empty")
    if features["zone_event_id"].duplicated().any() or labels["zone_event_id"].duplicated().any():
        raise RuntimeError("duplicate zone_event_id in R09 cache")
    keep_labels = [
        "zone_event_id",
        "stop_release_score",
        "high_stop_release_label",
        "release_event_bar_downside_bp",
        "release_event_bar_close_off_low_bp",
        "release_sell_notional_1m_vs_prior60",
        "release_trades_count_1m_vs_prior60",
        "release_sell_notional_5m_vs_prior60",
        "release_trades_count_5m_vs_prior60",
        "release_negative_delta_ratio_1m",
        "release_negative_delta_ratio_5m",
        "release_sell_impact_bp_per_million_1m",
        "release_sell_impact_bp_per_million_5m",
    ]
    keep_labels = [name for name in keep_labels if name in labels.columns]
    merged = features.merge(
        labels.loc[:, keep_labels],
        on="zone_event_id",
        how="left",
        validate="one_to_one",
    )
    merged = merged.loc[merged["event_kind"].eq("swing_zone_sweep")].copy()
    merged["event_pos"] = pd.to_numeric(merged["event_pos"], errors="coerce").astype("Int64")
    merged = merged.loc[merged["event_pos"].notna()].copy()
    merged["event_pos"] = merged["event_pos"].astype(np.int64)
    return merged.sort_values(["event_pos", "zone_event_id"], kind="mergesort").reset_index(drop=True), "R09_REPORT_CACHE"


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


def _consecutive_true(values: np.ndarray) -> int:
    total = 0
    for value in values[::-1]:
        if not bool(value):
            break
        total += 1
    return total


def _safe_sum(values: np.ndarray) -> float:
    return float(np.nansum(values)) if values.size else np.nan


def _impact_bp_per_million(start_price: float, end_low: float, sell_notional: float) -> float:
    if not np.isfinite(start_price) or start_price <= 0 or not np.isfinite(end_low) or not np.isfinite(sell_notional) or sell_notional <= 0:
        return np.nan
    downside_bp = max(0.0, (start_price - end_low) / start_price * 10_000.0)
    return downside_bp / (sell_notional / 1_000_000.0)


def build_post_sweep_checkpoints(
    zones: pd.DataFrame,
    primary: pd.DataFrame,
    config: PostSweepAcceptanceConfig,
    *,
    research_start: pd.Timestamp,
    research_end_exclusive: pd.Timestamp,
    max_events: int = 0,
    show_progress: bool = True,
) -> pd.DataFrame:
    cfg = config.validate()
    if zones.empty:
        return pd.DataFrame()
    bars = normalize_primary_bars(primary)
    index = pd.DatetimeIndex(bars.index)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    def numeric_column(name: str) -> np.ndarray:
        if name not in bars.columns:
            return np.full(len(bars), np.nan, dtype=float)
        return pd.to_numeric(bars[name], errors="coerce").to_numpy(dtype=float)

    notional = numeric_column("notional")
    sell = numeric_column("sell_notional")
    delta = numeric_column("delta_notional")
    trades = numeric_column("trades_count")
    work = zones.sort_values(["event_pos", "zone_event_id"], kind="mergesort")
    if int(max_events) > 0:
        work = work.head(int(max_events)).copy()
    rows: list[dict[str, object]] = []
    total = len(work) * len(cfg.checkpoints_minutes)
    reporter = ProgressReporter(
        label="[r12] post-sweep checkpoints",
        total=total,
        every=max(1, total // 200),
        enabled=bool(show_progress),
    )
    done = 0
    n = len(bars)
    for zone in work.itertuples(index=False):
        source = zone._asdict()
        event_pos = int(source["event_pos"])
        floor = float(source["zone_floor_price"])
        ceiling = float(source["zone_ceiling_price"])
        sweep_low = float(source.get("sweep_low", low[event_pos]))
        for checkpoint in cfg.checkpoints_minutes:
            checkpoint_pos = event_pos + int(checkpoint)
            entry_pos = checkpoint_pos + 1
            if event_pos < 0 or entry_pos >= n:
                done += 1
                reporter.update(done)
                continue
            checkpoint_bar_time = pd.Timestamp(index[checkpoint_pos])
            checkpoint_available_time = checkpoint_bar_time + pd.Timedelta(minutes=1)
            entry_time = pd.Timestamp(index[entry_pos])
            # Every visible post-sweep minute and the entry minute must be contiguous.
            expected_checkpoint = pd.Timestamp(index[event_pos]) + pd.Timedelta(minutes=int(checkpoint))
            if checkpoint_bar_time != expected_checkpoint or entry_time != checkpoint_available_time:
                done += 1
                reporter.update(done)
                continue
            if checkpoint_available_time < research_start or checkpoint_available_time >= research_end_exclusive:
                done += 1
                reporter.update(done)
                continue
            post_start = event_pos + 1
            post_end = checkpoint_pos
            post_close = close[post_start : post_end + 1]
            post_low = low[post_start : post_end + 1]
            post_high = high[post_start : post_end + 1]
            visible_low = low[event_pos : post_end + 1]
            visible_high = high[event_pos : post_end + 1]
            below = post_close < floor
            above_floor = post_close >= floor
            above_ceiling = post_close >= ceiling
            first_floor_rel = int(np.flatnonzero(above_floor)[0]) if above_floor.any() else -1
            first_ceiling_rel = int(np.flatnonzero(above_ceiling)[0]) if above_ceiling.any() else -1
            first_floor_pos = post_start + first_floor_rel if first_floor_rel >= 0 else -1
            first_ceiling_pos = post_start + first_ceiling_rel if first_ceiling_rel >= 0 else -1
            pre_reclaim_low = np.nan
            after_reclaim_low = np.nan
            after_reclaim_delta_sum = np.nan
            after_reclaim_sell_sum = np.nan
            second_wave_new_low = False
            if first_floor_pos >= 0:
                pre_reclaim_low = float(np.nanmin(low[event_pos : first_floor_pos + 1]))
                if first_floor_pos < post_end:
                    after_reclaim_low = float(np.nanmin(low[first_floor_pos + 1 : post_end + 1]))
                    after_reclaim_delta_sum = _safe_sum(delta[first_floor_pos + 1 : post_end + 1])
                    after_reclaim_sell_sum = _safe_sum(sell[first_floor_pos + 1 : post_end + 1])
                    second_wave_new_low = bool(np.isfinite(after_reclaim_low) and np.isfinite(pre_reclaim_low) and after_reclaim_low < pre_reclaim_low)
            path_low = float(np.nanmin(visible_low))
            path_high = float(np.nanmax(visible_high))
            checkpoint_close = float(close[checkpoint_pos])
            below_share = float(np.mean(below)) if len(below) else np.nan
            pressure_test_reject = bool(
                first_floor_pos >= 0
                and first_floor_pos < post_end
                and np.isfinite(after_reclaim_delta_sum)
                and after_reclaim_delta_sum < 0
                and not second_wave_new_low
                and checkpoint_close >= floor
            )
            if pressure_test_reject:
                state = "PRESSURE_TEST_REJECT"
            elif checkpoint_close >= ceiling:
                state = "STRONG_REJECT"
            elif checkpoint_close >= floor:
                state = "REJECT"
            elif first_floor_pos >= 0:
                state = "RECLAIM_FAILED"
            elif np.isfinite(below_share) and below_share >= float(cfg.persistent_accept_share):
                state = "PERSISTENT_ACCEPT"
            else:
                state = "MIXED_BELOW"
            split = max(1, len(post_close) // 2)
            first_slice = slice(post_start, post_start + split)
            second_slice = slice(post_start + split, post_end + 1)
            first_low = float(np.nanmin(low[first_slice])) if post_start + split > post_start else np.nan
            second_values = low[second_slice]
            second_low = float(np.nanmin(second_values)) if second_values.size else np.nan
            first_sell = _safe_sum(sell[first_slice])
            second_sell = _safe_sum(sell[second_slice])
            first_impact = _impact_bp_per_million(sweep_low, first_low, first_sell)
            second_anchor = first_low if np.isfinite(first_low) else sweep_low
            second_impact = _impact_bp_per_million(second_anchor, second_low, second_sell)
            immediate_entry_pos = event_pos + 1
            immediate_entry_price = float(open_[immediate_entry_pos]) if immediate_entry_pos < n else np.nan
            pre_entry_high = float(np.nanmax(high[immediate_entry_pos : entry_pos + 1])) if immediate_entry_pos <= entry_pos else np.nan
            pre_entry_low = float(np.nanmin(low[immediate_entry_pos : entry_pos + 1])) if immediate_entry_pos <= entry_pos else np.nan
            checkpoint_entry_price = float(open_[entry_pos])
            record = dict(source)
            record.update(
                {
                    "checkpoint_minutes": int(checkpoint),
                    "checkpoint_pos": int(checkpoint_pos),
                    "checkpoint_bar_time": checkpoint_bar_time,
                    "checkpoint_available_time": checkpoint_available_time,
                    "entry_pos": int(entry_pos),
                    "entry_time": entry_time,
                    "entry_price": checkpoint_entry_price,
                    "immediate_entry_pos": int(immediate_entry_pos),
                    "immediate_entry_time": pd.Timestamp(index[immediate_entry_pos]),
                    "immediate_entry_price": immediate_entry_price,
                    "long_entry_delay_bp": (checkpoint_entry_price / immediate_entry_price - 1.0) * 10_000.0 if immediate_entry_price > 0 else np.nan,
                    "short_entry_delay_bp": (immediate_entry_price / checkpoint_entry_price - 1.0) * 10_000.0 if checkpoint_entry_price > 0 else np.nan,
                    "pre_entry_mfe_long_bp": (pre_entry_high / immediate_entry_price - 1.0) * 10_000.0 if immediate_entry_price > 0 else np.nan,
                    "pre_entry_mae_long_bp": (immediate_entry_price / pre_entry_low - 1.0) * 10_000.0 if immediate_entry_price > 0 and pre_entry_low > 0 else np.nan,
                    "checkpoint_close": checkpoint_close,
                    "path_low_visible": path_low,
                    "path_high_visible": path_high,
                    "close_vs_floor_bp": (checkpoint_close / floor - 1.0) * 10_000.0 if floor > 0 else np.nan,
                    "close_vs_ceiling_bp": (checkpoint_close / ceiling - 1.0) * 10_000.0 if ceiling > 0 else np.nan,
                    "path_low_extension_below_sweep_bp": max(0.0, (sweep_low / path_low - 1.0) * 10_000.0) if path_low > 0 else np.nan,
                    "close_recovery_from_path_low_bp": (checkpoint_close / path_low - 1.0) * 10_000.0 if path_low > 0 else np.nan,
                    "post_close_below_floor_count": int(below.sum()),
                    "post_close_below_floor_share": below_share,
                    "post_close_above_floor_count": int(above_floor.sum()),
                    "post_close_above_ceiling_count": int(above_ceiling.sum()),
                    "terminal_consecutive_closes_above_floor": int(_consecutive_true(above_floor)),
                    "terminal_consecutive_closes_below_floor": int(_consecutive_true(below)),
                    "first_floor_reclaim_pos_visible": int(first_floor_pos),
                    "first_ceiling_reclaim_pos_visible": int(first_ceiling_pos),
                    "bars_to_floor_reclaim_visible": int(first_floor_pos - event_pos) if first_floor_pos >= 0 else np.nan,
                    "bars_to_ceiling_reclaim_visible": int(first_ceiling_pos - event_pos) if first_ceiling_pos >= 0 else np.nan,
                    "pre_reclaim_low_visible": pre_reclaim_low,
                    "after_reclaim_low_visible": after_reclaim_low,
                    "after_reclaim_delta_notional_sum": after_reclaim_delta_sum,
                    "after_reclaim_sell_notional_sum": after_reclaim_sell_sum,
                    "second_wave_new_low_visible": bool(second_wave_new_low),
                    "pressure_test_reject_visible": bool(pressure_test_reject),
                    "post_notional_sum": _safe_sum(notional[post_start : post_end + 1]),
                    "post_sell_notional_sum": _safe_sum(sell[post_start : post_end + 1]),
                    "post_delta_notional_sum": _safe_sum(delta[post_start : post_end + 1]),
                    "post_trades_count_sum": _safe_sum(trades[post_start : post_end + 1]),
                    "first_half_sell_notional": first_sell,
                    "second_half_sell_notional": second_sell,
                    "first_half_sell_impact_bp_per_million": first_impact,
                    "second_half_sell_impact_bp_per_million": second_impact,
                    "second_vs_first_sell_impact_ratio": second_impact / first_impact if np.isfinite(second_impact) and np.isfinite(first_impact) and first_impact > 0 else np.nan,
                    "state": state,
                    "state_direction": state_direction(state),
                }
            )
            rows.append(record)
            done += 1
            reporter.update(done)
    reporter.close()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["period"] = _period(out["checkpoint_available_time"])
    out["high_timeframe_zone"] = pd.to_numeric(out["zone_max_timeframe_min"], errors="coerce").ge(60)
    out["multitimeframe_zone"] = pd.to_numeric(out["zone_timeframe_count"], errors="coerce").ge(2)
    if "high_stop_release_label" in out.columns:
        raw_release = out["high_stop_release_label"]
        if pd.api.types.is_bool_dtype(raw_release):
            out["high_release"] = raw_release.fillna(False).astype(bool)
        else:
            out["high_release"] = raw_release.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    else:
        out["high_release"] = False
    return out.sort_values(["checkpoint_available_time", "zone_event_id", "checkpoint_minutes"], kind="mergesort").reset_index(drop=True)
