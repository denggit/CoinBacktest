#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build causal post-sweep checkpoints and future-only labels for R04."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars

from .config import PostSweepConfig

EPS = 1e-12
REQUIRED_FLOW_COLUMNS = (
    "notional",
    "buy_notional",
    "sell_notional",
    "delta_notional",
)
OPTIONAL_FLOW_COLUMNS = (
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
    "trades_count",
    "buy_trades_count",
    "sell_trades_count",
    "max_trade_notional",
)


def _safe_divide(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(float(den)) <= EPS:
        return np.nan
    return float(num) / float(den)


def _prefix(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(values.astype(float, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    return np.r_[0.0, np.cumsum(values, dtype=np.float64)]


def _window_sum(prefix: np.ndarray, left: int, right_inclusive: int) -> float:
    if right_inclusive < left:
        return 0.0
    return float(prefix[right_inclusive + 1] - prefix[left])


def _forward_rolling(values: np.ndarray, window: int, mode: str) -> np.ndarray:
    series = pd.Series(values, copy=False)
    rolling = series.rolling(int(window), min_periods=int(window))
    if mode == "max":
        out = rolling.max().shift(-(int(window) - 1))
    elif mode == "min":
        out = rolling.min().shift(-(int(window) - 1))
    else:
        raise ValueError(f"unsupported mode: {mode}")
    return out.to_numpy(dtype=float)


def _prepare_forward_labels(bars: pd.DataFrame, config: PostSweepConfig) -> dict[int, dict[str, np.ndarray]]:
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    out: dict[int, dict[str, np.ndarray]] = {}
    for horizon in config.future_horizons:
        h = int(horizon)
        out[h] = {
            "max_high": _forward_rolling(high, h, "max"),
            "min_low": _forward_rolling(low, h, "min"),
            "end_close": pd.Series(close, copy=False).shift(-(h - 1)).to_numpy(dtype=float),
        }
    return out


def _prior_high_arrays(high: np.ndarray, windows: Iterable[int]) -> dict[int, np.ndarray]:
    series = pd.Series(high, copy=False)
    return {
        int(window): series.shift(1).rolling(int(window), min_periods=int(window)).max().to_numpy(dtype=float)
        for window in windows
    }


def _fixed_period(timestamp: pd.Timestamp) -> str:
    if timestamp < pd.Timestamp("2025-01-01"):
        return "EARLY_2023_2024"
    if timestamp < pd.Timestamp("2025-10-01"):
        return "MID_2025Q1_Q3"
    return "BOOKS_2025Q4_2026H1"


def _required_orderflow(bars: pd.DataFrame) -> None:
    missing = [name for name in REQUIRED_FLOW_COLUMNS if name not in bars.columns]
    if missing:
        raise ValueError(
            "R04 requires trade-bar order-flow columns; missing=" + ",".join(missing)
            + ". Use OKXTradeBarLoader, not plain OHLCV."
        )


def _checkpoint_offsets(
    low_segment: np.ndarray,
    sweep_low: float,
    config: PostSweepConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return selected offsets and causal new-low state arrays for one event."""

    if not len(low_segment):
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, empty, np.array([], dtype=float)
    prior_running = np.empty(len(low_segment), dtype=float)
    running = np.empty(len(low_segment), dtype=float)
    new_low = np.zeros(len(low_segment), dtype=bool)
    attempt_index = np.zeros(len(low_segment), dtype=np.int64)
    current = float(sweep_low)
    attempts = 0
    epsilon_fraction = float(config.new_low_epsilon_bp) / 10_000.0
    for i, value in enumerate(low_segment):
        prior_running[i] = current
        threshold = current * (1.0 - epsilon_fraction)
        if np.isfinite(value) and value < threshold:
            new_low[i] = True
            attempts += 1
            current = float(value)
        elif np.isfinite(value) and value < current:
            # Sub-epsilon movement remains part of running risk, but is not a
            # separate attempt when a positive epsilon is requested.
            current = float(value)
        running[i] = current
        attempt_index[i] = attempts

    dense_end = min(int(config.dense_checkpoint_bars), len(low_segment))
    selected = set(range(1, dense_end + 1))
    selected.update(int(v) for v in config.fixed_checkpoint_bars if int(v) <= len(low_segment))
    selected.update((np.flatnonzero(new_low) + 1).astype(int).tolist())
    offsets = np.asarray(sorted(selected), dtype=np.int64) - 1
    if len(offsets) > int(config.max_rows_per_event):
        # Preserve dense/fixed schedule, then the earliest additional attempts.
        mandatory = sorted(
            set(range(1, dense_end + 1))
            | {int(v) for v in config.fixed_checkpoint_bars if int(v) <= len(low_segment)}
        )
        extra = [int(v) for v in (np.flatnonzero(new_low) + 1) if int(v) not in set(mandatory)]
        kept = (mandatory + extra)[: int(config.max_rows_per_event)]
        offsets = np.asarray(sorted(set(kept)), dtype=np.int64) - 1

    last_attempt_pos = np.full(len(low_segment), -1, dtype=np.int64)
    last = -1
    for i in range(len(low_segment)):
        if new_low[i]:
            last = i
        last_attempt_pos[i] = last
    bars_since = np.arange(len(low_segment), dtype=np.int64) - last_attempt_pos
    bars_since[last_attempt_pos < 0] = np.arange(len(low_segment), dtype=np.int64)[last_attempt_pos < 0] + 1
    return offsets, new_low, attempt_index, bars_since, prior_running


def build_post_sweep_checkpoint_table(
    zone_events: pd.DataFrame,
    primary: pd.DataFrame,
    config: PostSweepConfig,
    *,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build sparse causal checkpoints after each first Swing-zone sweep.

    Each checkpoint is a closed 1m bar after the closed sweep bar. Every feature
    uses data available by ``checkpoint_available_time``. Future path fields are
    labels and are split out by :func:`split_checkpoint_features_labels`.
    """

    cfg = config.validate()
    if zone_events.empty:
        return pd.DataFrame()
    bars = normalize_primary_bars(primary)
    _required_orderflow(bars)
    index = pd.DatetimeIndex(bars.index)
    n = len(bars)

    arrays: dict[str, np.ndarray] = {}
    for name in ("open", "high", "low", "close", *REQUIRED_FLOW_COLUMNS, *OPTIONAL_FLOW_COLUMNS):
        if name in bars.columns:
            arrays[name] = pd.to_numeric(bars[name], errors="coerce").to_numpy(dtype=float)
        else:
            arrays[name] = np.zeros(n, dtype=float)
    prefixes = {name: _prefix(arrays[name]) for name in (*REQUIRED_FLOW_COLUMNS, *OPTIONAL_FLOW_COLUMNS)}
    prior_high = _prior_high_arrays(arrays["high"], cfg.micro_break_windows)
    forward = _prepare_forward_labels(bars, cfg)

    rows: list[dict[str, object]] = []
    events = zone_events.sort_values("event_pos", kind="mergesort").reset_index(drop=True)
    total = len(events)
    progress_step = max(1, total // 20)

    for event_number, event in events.iterrows():
        event_pos = int(event["event_pos"])
        segment_start = event_pos + 1
        segment_end = min(n, segment_start + int(cfg.observation_horizon_bars))
        if segment_start >= segment_end:
            continue
        sweep_low = float(event.get("sweep_low", arrays["low"][event_pos]))
        zone_floor = float(event.get("zone_floor_price", sweep_low))
        zone_ceiling = float(event.get("zone_ceiling_price", zone_floor))
        zone_center = float(event.get("zone_center_price", (zone_floor + zone_ceiling) / 2.0))
        pre_atr_240 = float(event.get("pre_atr_240m_abs", np.nan))
        low_segment = arrays["low"][segment_start:segment_end]
        offsets, new_low_flags, attempt_indices, bars_since_new_low, prior_running = _checkpoint_offsets(
            low_segment, sweep_low, cfg
        )
        running_low = np.minimum.accumulate(np.r_[sweep_low, low_segment])[1:]
        delta_segment = arrays["delta_notional"][segment_start:segment_end]
        cum_delta = np.cumsum(np.nan_to_num(delta_segment, nan=0.0), dtype=np.float64)
        prev_cum_min = np.r_[0.0, np.minimum.accumulate(cum_delta)[:-1]]
        cvd_new_low = cum_delta < prev_cum_min

        attempt_positions = np.flatnonzero(new_low_flags)
        attempt_extension_bp = np.full(len(low_segment), np.nan, dtype=float)
        attempt_delta = np.full(len(low_segment), np.nan, dtype=float)
        attempt_sell = np.full(len(low_segment), np.nan, dtype=float)
        attempt_extension_ratio = np.full(len(low_segment), np.nan, dtype=float)
        previous_attempt_offset = -1
        previous_extension = np.nan
        for offset in attempt_positions:
            global_pos = segment_start + int(offset)
            left = segment_start if previous_attempt_offset < 0 else segment_start + previous_attempt_offset + 1
            extension = max(0.0, float(prior_running[offset] - arrays["low"][global_pos])) / max(zone_center, EPS) * 10_000.0
            attempt_extension_bp[offset] = extension
            attempt_delta[offset] = _window_sum(prefixes["delta_notional"], left, global_pos)
            attempt_sell[offset] = _window_sum(prefixes["sell_notional"], left, global_pos)
            if np.isfinite(previous_extension) and previous_extension > EPS:
                attempt_extension_ratio[offset] = extension / previous_extension
            previous_extension = extension
            previous_attempt_offset = int(offset)

        for offset in offsets:
            checkpoint_pos = segment_start + int(offset)
            elapsed = int(offset) + 1
            checkpoint_time = pd.Timestamp(index[checkpoint_pos])
            checkpoint_available_time = checkpoint_time + pd.Timedelta(minutes=1)
            close_now = float(arrays["close"][checkpoint_pos])
            low_now = float(arrays["low"][checkpoint_pos])
            current_running_low = float(running_low[offset])
            current_new_low = bool(new_low_flags[offset])
            current_attempt = int(attempt_indices[offset])
            current_bars_since = int(bars_since_new_low[offset])

            row: dict[str, object] = {
                "checkpoint_id": f"{event['zone_event_id']}_C{elapsed:03d}",
                "zone_event_id": str(event["zone_event_id"]),
                "event_kind": str(event.get("event_kind", "swing_zone_sweep")),
                "period": _fixed_period(checkpoint_available_time),
                "event_pos": event_pos,
                "event_available_time": pd.Timestamp(event["event_available_time"]),
                "checkpoint_pos": checkpoint_pos,
                "checkpoint_time": checkpoint_time,
                "checkpoint_available_time": checkpoint_available_time,
                "elapsed_bars": elapsed,
                "zone_floor_price": zone_floor,
                "zone_ceiling_price": zone_ceiling,
                "zone_center_price": zone_center,
                "sweep_low": sweep_low,
                "checkpoint_open": float(arrays["open"][checkpoint_pos]),
                "checkpoint_high": float(arrays["high"][checkpoint_pos]),
                "checkpoint_low": low_now,
                "checkpoint_close": close_now,
                "running_low_since_sweep": current_running_low,
                "new_low_attempt_flag": current_new_low,
                "new_low_attempt_index": current_attempt,
                "bars_since_new_low_attempt": current_bars_since,
                "new_low_extension_bp": float(attempt_extension_bp[offset]) if current_new_low else 0.0,
                "new_low_extension_to_pre_atr_240m": _safe_divide(
                    float(prior_running[offset] - low_now) if current_new_low else 0.0,
                    pre_atr_240,
                ),
                "attempt_delta_notional": float(attempt_delta[offset]) if current_new_low else np.nan,
                "attempt_sell_notional": float(attempt_sell[offset]) if current_new_low else np.nan,
                "attempt_extension_vs_previous": float(attempt_extension_ratio[offset]) if current_new_low else np.nan,
                "close_vs_zone_floor_bp": (close_now / max(zone_floor, EPS) - 1.0) * 10_000.0,
                "close_vs_zone_ceiling_bp": (close_now / max(zone_ceiling, EPS) - 1.0) * 10_000.0,
                "close_vs_running_low_bp": (close_now / max(current_running_low, EPS) - 1.0) * 10_000.0,
                "running_low_vs_zone_floor_bp": (current_running_low / max(zone_floor, EPS) - 1.0) * 10_000.0,
                "running_low_vs_sweep_low_bp": (current_running_low / max(sweep_low, EPS) - 1.0) * 10_000.0,
                "zone_floor_reclaimed": bool(close_now > zone_floor),
                "zone_ceiling_reclaimed": bool(close_now > zone_ceiling),
                "cum_delta_since_sweep": float(cum_delta[offset]),
                "cum_delta_ratio_since_sweep": _safe_divide(
                    float(cum_delta[offset]),
                    _window_sum(prefixes["notional"], segment_start, checkpoint_pos),
                ),
                "cvd_new_low_flag": bool(cvd_new_low[offset]),
                "cvd_new_low_without_price_new_low": bool(cvd_new_low[offset] and not current_new_low),
                "negative_delta_without_price_new_low": bool(
                    arrays["delta_notional"][checkpoint_pos] < 0 and not current_new_low
                ),
            }

            for window in cfg.no_new_low_windows:
                row[f"no_new_low_{int(window)}bars"] = bool(current_bars_since >= int(window))
            for window in cfg.micro_break_windows:
                threshold = prior_high[int(window)][checkpoint_pos]
                row[f"micro_high_break_{int(window)}bars"] = bool(np.isfinite(threshold) and close_now > threshold)

            for window in cfg.flow_windows:
                w = int(window)
                left = max(segment_start, checkpoint_pos - w + 1)
                base_pos = left - 1
                base_close = float(arrays["close"][base_pos]) if base_pos >= 0 else float(arrays["open"][left])
                buy = _window_sum(prefixes["buy_notional"], left, checkpoint_pos)
                sell = _window_sum(prefixes["sell_notional"], left, checkpoint_pos)
                notional = _window_sum(prefixes["notional"], left, checkpoint_pos)
                delta = _window_sum(prefixes["delta_notional"], left, checkpoint_pos)
                large_buy = _window_sum(prefixes["large_buy_notional"], left, checkpoint_pos)
                large_sell = _window_sum(prefixes["large_sell_notional"], left, checkpoint_pos)
                large_delta = _window_sum(prefixes["large_delta_notional"], left, checkpoint_pos)
                price_change_bp = (close_now / max(base_close, EPS) - 1.0) * 10_000.0
                downside_bp = max(0.0, -price_change_bp)
                row[f"buy_notional_{w}m"] = buy
                row[f"sell_notional_{w}m"] = sell
                row[f"delta_notional_{w}m"] = delta
                row[f"delta_ratio_{w}m"] = _safe_divide(delta, notional)
                row[f"sell_share_{w}m"] = _safe_divide(sell, buy + sell)
                row[f"large_delta_ratio_{w}m"] = _safe_divide(large_delta, large_buy + large_sell)
                row[f"price_change_{w}m_bp"] = price_change_bp
                row[f"downside_bp_per_sell_million_{w}m"] = _safe_divide(downside_bp, sell / 1_000_000.0)
                row[f"downside_bp_per_abs_negative_delta_million_{w}m"] = _safe_divide(
                    downside_bp,
                    max(0.0, -delta) / 1_000_000.0,
                )
                row[f"price_to_delta_slope_{w}m"] = _safe_divide(price_change_bp, delta / 1_000_000.0)

            # Future-only labels begin at the next open after this closed checkpoint.
            entry_pos = checkpoint_pos + 1
            row["entry_reference_pos"] = entry_pos if entry_pos < n else -1
            row["entry_reference_time"] = pd.Timestamp(index[entry_pos]) if entry_pos < n else pd.NaT
            entry_price = float(arrays["open"][entry_pos]) if entry_pos < n else np.nan
            row["entry_reference_price"] = entry_price
            for horizon in cfg.future_horizons:
                h = int(horizon)
                label_data = forward[h]
                if entry_pos >= n or not np.isfinite(label_data["max_high"][entry_pos]):
                    row[f"future_label_complete_{h}m"] = False
                    row[f"future_mfe_{h}m"] = np.nan
                    row[f"future_mae_{h}m"] = np.nan
                    row[f"future_close_return_{h}m"] = np.nan
                    row[f"future_no_lower_low_{h}m"] = pd.NA
                    row[f"future_reversal_dominant_{h}m"] = pd.NA
                    row[f"future_continuation_dominant_{h}m"] = pd.NA
                    continue
                mfe = label_data["max_high"][entry_pos] / max(entry_price, EPS) - 1.0
                mae = label_data["min_low"][entry_pos] / max(entry_price, EPS) - 1.0
                close_return = label_data["end_close"][entry_pos] / max(entry_price, EPS) - 1.0
                no_lower_low = bool(label_data["min_low"][entry_pos] >= current_running_low)
                row[f"future_label_complete_{h}m"] = True
                row[f"future_mfe_{h}m"] = float(mfe)
                row[f"future_mae_{h}m"] = float(mae)
                row[f"future_close_return_{h}m"] = float(close_return)
                row[f"future_no_lower_low_{h}m"] = no_lower_low
                row[f"future_reversal_dominant_{h}m"] = bool(
                    mfe >= float(cfg.reversal_mfe_return)
                    and mfe >= abs(mae) * float(cfg.dominance_ratio)
                )
                row[f"future_continuation_dominant_{h}m"] = bool(
                    mae <= -float(cfg.continuation_mae_return)
                    and abs(mae) >= mfe * float(cfg.dominance_ratio)
                )
            max_horizon = max(cfg.future_horizons)
            for threshold in cfg.large_mfe_returns:
                tag = str(float(threshold) * 100.0).rstrip("0").rstrip(".").replace(".", "p")
                value = row.get(f"future_mfe_{max_horizon}m", np.nan)
                row[f"future_large_mfe_{tag}_{max_horizon}m"] = (
                    bool(float(value) >= float(threshold)) if np.isfinite(value) else pd.NA
                )
            rows.append(row)

        if show_progress and ((event_number + 1) % progress_step == 0 or event_number + 1 == total):
            pct = (event_number + 1) / max(total, 1) * 100.0
            print(
                f"[post-sweep] events={event_number + 1:,}/{total:,} ({pct:5.1f}%) checkpoints={len(rows):,}",
                flush=True,
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["checkpoint_available_time", "zone_event_id", "elapsed_bars"], kind="mergesort").reset_index(drop=True)
    return out


def split_checkpoint_features_labels(checkpoints: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Physically separate future labels from causal checkpoint features."""

    if checkpoints.empty:
        return checkpoints.copy(), checkpoints.copy()
    identifiers = [
        name
        for name in (
            "checkpoint_id",
            "zone_event_id",
            "event_kind",
            "period",
            "event_pos",
            "event_available_time",
            "checkpoint_pos",
            "checkpoint_time",
            "checkpoint_available_time",
            "elapsed_bars",
        )
        if name in checkpoints.columns
    ]
    label_prefixes = ("entry_reference_", "future_")
    label_columns = identifiers + [
        name for name in checkpoints.columns if name.startswith(label_prefixes)
    ]
    label_columns = list(dict.fromkeys(label_columns))
    labels = checkpoints.loc[:, label_columns].copy()
    label_set = set(label_columns) - set(identifiers)
    features = checkpoints.loc[:, [name for name in checkpoints.columns if name not in label_set]].copy()
    forbidden = [
        name for name in features.columns
        if name.startswith("future_") or name.startswith("entry_reference_") or "oracle" in name.lower()
    ]
    if forbidden:
        raise RuntimeError(f"future label leakage in checkpoint features: {forbidden}")
    return features, labels
