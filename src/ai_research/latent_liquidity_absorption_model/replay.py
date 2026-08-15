#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal 1-second snapshot construction for R01.3."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_execution_audit.replay import _normalize_bars
from src.ai_research.latent_liquidity_path_atlas.candidates import normalize_second_bars
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter

from .cache import load_snapshot_day, save_snapshot_day, snapshot_day_path
from .config import AbsorptionModelConfig


@dataclass(frozen=True)
class SnapshotBuildResult:
    snapshots: pd.DataFrame
    quality: pd.DataFrame


def _safe_median(values: np.ndarray, fallback: float = 1.0) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return float(fallback)
    value = float(np.median(finite))
    return value if abs(value) > 1e-12 else float(fallback)


def _signed_move(current: float, reference: float, side: str) -> float:
    if side == "DOWN":
        return (reference - current) / reference * 1e4
    return (current - reference) / reference * 1e4


def _favorable_move(current: float, entry: float, side: str) -> float:
    if side == "DOWN":  # long after downward release
        return (current - entry) / entry * 1e4
    return (entry - current) / entry * 1e4


def _known_extreme(frame: pd.DataFrame, side: str) -> float:
    return float(frame["low"].min()) if side == "DOWN" else float(frame["high"].max())


def _extreme_positions(frame: pd.DataFrame, side: str) -> np.ndarray:
    values = frame["low"].to_numpy(dtype=float) if side == "DOWN" else frame["high"].to_numpy(dtype=float)
    if side == "DOWN":
        running = np.minimum.accumulate(values)
    else:
        running = np.maximum.accumulate(values)
    return np.flatnonzero(values == running)


def _window(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    return frame.iloc[max(0, len(frame) - int(count)) :]


def _burst_count(values: np.ndarray, baseline: float, threshold: float) -> int:
    if baseline <= 0:
        return 0
    return int(np.sum(values >= baseline * threshold))


def _barrier_order(
    future: pd.DataFrame,
    side: str,
    stop_price: float,
    target_price: float,
) -> str:
    for row in future.itertuples(index=False):
        if side == "DOWN":
            stop_hit = float(row.low) <= stop_price
            target_hit = float(row.high) >= target_price
        else:
            stop_hit = float(row.high) >= stop_price
            target_hit = float(row.low) <= target_price
        if stop_hit:
            return "STOP"
        if target_hit:
            return "TARGET"
    return "NONE"


def snapshot_rows_for_event(
    path: pd.DataFrame,
    event: object,
    config: AbsorptionModelConfig,
) -> pd.DataFrame:
    """Build causal decision snapshots and future labels for one Episode representative."""
    event_time = pd.Timestamp(getattr(event, "event_time"))
    side = str(getattr(event, "event_side"))
    event_ref = float(getattr(event, "event_reference_price"))
    event_pos = int(config.pre_replay_seconds)
    pre = path.iloc[:event_pos]
    after = path.iloc[event_pos + 1 :].reset_index(drop=False).rename(columns={path.index.name or "index": "bar_time"})
    if after.empty:
        return pd.DataFrame()
    base_notional = max(_safe_median(pre["notional"].to_numpy(dtype=float)), 1e-9)
    base_trades = max(_safe_median(pre["trades_count"].to_numpy(dtype=float)), 1e-9)
    rows: list[dict[str, object]] = []
    for offset in config.decision_offsets_seconds:
        decision_pos = int(offset) - 1
        entry_pos = decision_pos + 1
        future_end = entry_pos + int(config.label_horizon_seconds)
        if decision_pos < 0 or future_end > len(after):
            continue
        observed = after.iloc[: decision_pos + 1]
        future = after.iloc[entry_pos:future_end]
        if observed.empty or future.empty:
            continue
        current_close = float(observed.iloc[-1]["close"])
        current_extreme = _known_extreme(observed, side)
        extreme_positions = _extreme_positions(observed, side)
        last_extreme_pos = int(extreme_positions[-1]) if len(extreme_positions) else decision_pos
        seconds_since_extreme = int(decision_pos - last_extreme_pos)
        extension_bp = _signed_move(current_extreme, event_ref, side)
        if side == "DOWN":
            reclaim_bp = (current_close / current_extreme - 1.0) * 1e4
        else:
            reclaim_bp = (current_extreme / current_close - 1.0) * 1e4
        observed_notional = observed["notional"].to_numpy(dtype=float)
        observed_trades = observed["trades_count"].to_numpy(dtype=float)
        observed_delta = observed["delta_notional"].to_numpy(dtype=float)
        aligned_delta = -observed_delta if side == "DOWN" else observed_delta
        feature: dict[str, object] = {
            "event_id": str(getattr(event, "event_id")),
            "release_episode_id": str(getattr(event, "release_episode_id")),
            "event_time": event_time,
            "decision_time": event_time + pd.Timedelta(seconds=int(offset)),
            "entry_time": event_time + pd.Timedelta(seconds=int(offset) + 1),
            "event_side": side,
            "period": str(getattr(event, "period")),
            "path_cluster": int(getattr(event, "path_cluster")),
            "cluster_distance": float(getattr(event, "cluster_distance")),
            "decision_offset_seconds": int(offset),
            "event_reference_price": event_ref,
            "known_extreme_price": current_extreme,
            "current_close": current_close,
            "extension_from_reference_bp": extension_bp,
            "reclaim_from_known_extreme_bp": reclaim_bp,
            "seconds_since_known_extreme": seconds_since_extreme,
            "notional_intensity_cum": float(np.sum(observed_notional) / (base_notional * len(observed))),
            "trades_intensity_cum": float(np.sum(observed_trades) / (base_trades * len(observed))),
            "aligned_delta_share_cum": float(np.sum(aligned_delta) / max(np.sum(observed_notional), 1e-9)),
            "notional_burst_count_2x": _burst_count(observed_notional, base_notional, 2.0),
            "notional_burst_count_3x": _burst_count(observed_notional, base_notional, 3.0),
            "notional_burst_count_5x": _burst_count(observed_notional, base_notional, 5.0),
        }
        for window in config.recent_windows_seconds:
            recent = _window(observed, int(window))
            close = recent["close"].to_numpy(dtype=float)
            notional = recent["notional"].to_numpy(dtype=float)
            trades = recent["trades_count"].to_numpy(dtype=float)
            delta = recent["delta_notional"].to_numpy(dtype=float)
            aligned = -delta if side == "DOWN" else delta
            start_price = float(recent.iloc[0]["open"])
            end_price = float(recent.iloc[-1]["close"])
            signed_release = _signed_move(end_price, start_price, side)
            travel = float(np.sum(np.abs(np.diff(close) / close[:-1]) * 1e4)) if len(close) > 1 else 0.0
            price_range = float((recent["high"].max() - recent["low"].min()) / max(start_price, 1e-9) * 1e4)
            feature[f"release_move_{window}s_bp"] = signed_release
            feature[f"path_travel_{window}s_bp"] = travel
            feature[f"range_{window}s_bp"] = price_range
            feature[f"price_efficiency_{window}s"] = abs(signed_release) / max(travel, 1e-9)
            feature[f"notional_intensity_{window}s"] = float(np.mean(notional) / base_notional)
            feature[f"trades_intensity_{window}s"] = float(np.mean(trades) / base_trades)
            feature[f"aligned_delta_share_{window}s"] = float(np.sum(aligned) / max(np.sum(notional), 1e-9))
            feature[f"impact_bp_per_million_{window}s"] = abs(signed_release) / max(np.sum(notional) / 1_000_000.0, 1e-6)
        # Pressure that fails to create more release-direction extension is a
        # causal absorption proxy, not a hard confirmation rule.
        feature["pressure_no_progress_15s"] = float(
            max(feature.get("aligned_delta_share_15s", 0.0), 0.0)
            * max(feature.get("notional_intensity_15s", 0.0), 0.0)
            / max(abs(float(feature.get("release_move_15s_bp", 0.0))), 1.0)
        )
        feature["efficiency_decay_5s_vs_30s"] = float(feature.get("price_efficiency_5s", 0.0)) - float(
            feature.get("price_efficiency_30s", 0.0)
        )
        feature["impact_decay_5s_vs_30s"] = float(feature.get("impact_bp_per_million_5s", 0.0)) - float(
            feature.get("impact_bp_per_million_30s", 0.0)
        )
        # Rebound/retest structure is represented continuously rather than as
        # one frozen hand-written confirmation.
        close_seen = observed["close"].to_numpy(dtype=float)
        if side == "DOWN":
            reclaim_series = (close_seen / current_extreme - 1.0) * 1e4
        else:
            reclaim_series = (current_extreme / close_seen - 1.0) * 1e4
        max_reclaim = float(np.nanmax(reclaim_series)) if len(reclaim_series) else 0.0
        feature["max_reclaim_seen_bp"] = max_reclaim
        feature["reclaim_giveback_bp"] = max(0.0, max_reclaim - reclaim_bp)
        feature["extreme_updates_count"] = int(len(extreme_positions))

        # Labels are generated for several fixed costs and execution delays.
        # The primary supervised target remains 1-second delay / 1x cost.
        primary_entry_price = float(future.iloc[0]["open"])
        future_extreme = float(future["low"].min()) if side == "DOWN" else float(future["high"].max())
        if side == "DOWN":
            additional_extension_bp = max(0.0, (current_extreme - future_extreme) / current_extreme * 1e4)
            stop_price = current_extreme * (1.0 - config.structural_stop_buffer_bp / 1e4)
        else:
            additional_extension_bp = max(0.0, (future_extreme - current_extreme) / current_extreme * 1e4)
            stop_price = current_extreme * (1.0 + config.structural_stop_buffer_bp / 1e4)
        absorption_end = min(len(future), int(config.absorption_lookahead_seconds))
        near_future = future.iloc[:absorption_end]
        near_extreme = float(near_future["low"].min()) if side == "DOWN" else float(near_future["high"].max())
        if side == "DOWN":
            near_extension = max(0.0, (current_extreme - near_extreme) / current_extreme * 1e4)
        else:
            near_extension = max(0.0, (near_extreme - current_extreme) / current_extreme * 1e4)
        feature["future_additional_extension_bp"] = float(additional_extension_bp)
        feature["absorption_complete_target"] = bool(near_extension <= config.absorption_extension_tolerance_bp)
        feature["feature_available_time"] = event_time + pd.Timedelta(seconds=int(offset))
        primary_barrier = "NONE"
        for delay in config.entry_delay_seconds:
            delayed_entry_pos = decision_pos + int(delay)
            delayed_end = delayed_entry_pos + int(config.label_horizon_seconds)
            if delayed_entry_pos >= len(after) or delayed_end > len(after):
                continue
            delayed_future = after.iloc[delayed_entry_pos:delayed_end]
            entry_price = float(delayed_future.iloc[0]["open"])
            if side == "DOWN":
                favorable_path = (delayed_future["high"].to_numpy(dtype=float) - entry_price) / entry_price * 1e4
                adverse_path = (entry_price - delayed_future["low"].to_numpy(dtype=float)) / entry_price * 1e4
            else:
                favorable_path = (entry_price - delayed_future["low"].to_numpy(dtype=float)) / entry_price * 1e4
                adverse_path = (delayed_future["high"].to_numpy(dtype=float) - entry_price) / entry_price * 1e4
            terminal_close = float(delayed_future.iloc[-1]["close"])
            terminal_gross = _favorable_move(terminal_close, entry_price, side)
            feature[f"entry_price_d{delay}"] = entry_price
            if side == "DOWN":
                stop_distance_bp = max(0.0, (entry_price - stop_price) / entry_price * 1e4)
            else:
                stop_distance_bp = max(0.0, (stop_price - entry_price) / entry_price * 1e4)
            feature[f"structural_stop_distance_bp_d{delay}"] = float(stop_distance_bp)
            feature[f"future_favorable_mfe_bp_d{delay}"] = float(max(0.0, np.nanmax(favorable_path)))
            feature[f"future_adverse_mae_bp_d{delay}"] = float(max(0.0, np.nanmax(adverse_path)))
            feature[f"future_terminal_gross_bp_d{delay}"] = float(terminal_gross)
            for cost_multiple in config.cost_multipliers:
                cost_bp = float(config.roundtrip_cost_bp * cost_multiple)
                target_gross_bp = float(cost_bp + config.minimum_net_room_bp)
                if side == "DOWN":
                    target_price = entry_price * (1.0 + target_gross_bp / 1e4)
                else:
                    target_price = entry_price * (1.0 - target_gross_bp / 1e4)
                barrier = _barrier_order(delayed_future, side, stop_price, target_price)
                key = f"d{delay}_c{int(cost_multiple)}x"
                feature[f"barrier_result_{key}"] = barrier
                feature[f"tradeable_before_stop_{key}"] = bool(barrier == "TARGET")
                feature[f"future_terminal_net_bp_{key}"] = float(terminal_gross - cost_bp)
                if delay == 1 and float(cost_multiple) == 1.0:
                    primary_barrier = barrier
        feature["entry_price"] = primary_entry_price
        feature["future_favorable_mfe_bp"] = float(feature.get("future_favorable_mfe_bp_d1", np.nan))
        feature["future_adverse_mae_bp"] = float(feature.get("future_adverse_mae_bp_d1", np.nan))
        feature["future_terminal_gross_bp"] = float(feature.get("future_terminal_gross_bp_d1", np.nan))
        feature["future_terminal_net_bp_1x"] = float(feature.get("future_terminal_net_bp_d1_c1x", np.nan))
        feature["tradeable_before_stop_target"] = bool(feature.get("tradeable_before_stop_d1_c1x", False))
        feature["barrier_result"] = primary_barrier
        rows.append(feature)
    return pd.DataFrame(rows)


def build_snapshot_dataset(
    samples: pd.DataFrame,
    config: AbsorptionModelConfig,
    *,
    cache_root: Path,
    data_dir: str | Path | None = None,
    db_name: str = "okx_trade_bars.db",
    progress: bool = True,
    use_cache: bool = True,
) -> SnapshotBuildResult:
    if samples.empty:
        return SnapshotBuildResult(pd.DataFrame(), pd.DataFrame())
    work = samples.copy()
    work["event_time"] = pd.to_datetime(work["event_time"], errors="coerce")
    work = work.loc[work["event_time"].notna()].drop_duplicates("event_id")
    work["event_day"] = work["event_time"].dt.floor("D")
    groups = list(work.groupby("event_day", sort=True))
    reporter = ProgressReporter(
        label="[latent-liquidity-r01.3] causal snapshot replay days",
        total=len(groups),
        every=1,
        enabled=progress,
    )
    loader = OKXTradeBarLoader(symbol=config.symbol, timeframe="1s", data_dir=data_dir, db_name=db_name)
    parts: list[pd.DataFrame] = []
    quality_rows: list[dict[str, object]] = []
    for number, (day, events) in enumerate(groups, start=1):
        cache_path = snapshot_day_path(cache_root, pd.Timestamp(day))
        if use_cache and cache_path.exists():
            try:
                day_frame = load_snapshot_day(cache_path)
                parts.append(day_frame)
                quality_rows.append({"day": day, "requested_events": len(events), "complete_events": day_frame["event_id"].nunique() if not day_frame.empty else 0, "cache": True})
                reporter.update(number)
                continue
            except (OSError, ValueError, EOFError):
                cache_path.unlink(missing_ok=True)
        pad = int(config.replay_max_fill_gap_seconds)
        start = events["event_time"].min() - pd.Timedelta(seconds=config.pre_replay_seconds + pad)
        end = events["event_time"].max() + pd.Timedelta(seconds=config.post_replay_seconds + pad)
        bars = loader.fetch_data_by_date_range(start, end, build_missing=False, force_rebuild=False, cvd_mode="range")
        bars = _normalize_bars(bars)
        if not bars.empty:
            bars = normalize_second_bars(bars, SimpleNamespace(max_fill_gap_seconds=config.replay_max_fill_gap_seconds))
        day_parts: list[pd.DataFrame] = []
        complete = 0
        for event in events.itertuples(index=False):
            event_time = pd.Timestamp(event.event_time)
            expected = pd.date_range(
                event_time - pd.Timedelta(seconds=config.pre_replay_seconds),
                event_time + pd.Timedelta(seconds=config.post_replay_seconds),
                freq="1s",
            )
            path = bars.reindex(expected)
            required = ["open", "high", "low", "close", "notional", "trades_count", "delta_notional"]
            unsafe = bool(path.get("unsafe_gap", pd.Series(1, index=path.index)).fillna(1).astype(bool).any())
            if path[required].isna().any().any() or unsafe:
                continue
            rows = snapshot_rows_for_event(path, event, config)
            if not rows.empty:
                complete += 1
                day_parts.append(rows)
        day_frame = pd.concat(day_parts, ignore_index=True, copy=False) if day_parts else pd.DataFrame()
        if use_cache:
            save_snapshot_day(cache_path, day_frame)
        parts.append(day_frame)
        quality_rows.append({"day": day, "requested_events": len(events), "complete_events": complete, "cache": False})
        reporter.update(number)
    reporter.close()
    snapshots = pd.concat([part for part in parts if not part.empty], ignore_index=True, copy=False) if any(not part.empty for part in parts) else pd.DataFrame()
    quality = pd.DataFrame(quality_rows)
    if not quality.empty:
        quality["completion_rate"] = quality["complete_events"] / quality["requested_events"].clip(lower=1)
    return SnapshotBuildResult(snapshots=snapshots, quality=quality)
