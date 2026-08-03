#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Broad 1m stop-release labels and frozen control-calibrated release score."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars

from .config import StructuredStopPoolConfig

EPS = 1e-12


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.full(np.broadcast_shapes(num.shape, den.shape), np.nan, dtype=float), where=np.isfinite(den) & (np.abs(den) > EPS))


def _prefix(values: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.r_[0.0, np.cumsum(arr, dtype=float)]


def _window_sum(prefix: np.ndarray, positions: np.ndarray, horizon: int, n: int) -> np.ndarray:
    start = positions.astype(np.int64)
    end = np.minimum(start + int(horizon), int(n))
    valid = (start >= 0) & (start < n) & (end > start)
    out = np.full(len(start), np.nan, dtype=float)
    out[valid] = prefix[end[valid]] - prefix[start[valid]]
    return out


def _window_terminal(values: np.ndarray, positions: np.ndarray, horizon: int) -> np.ndarray:
    n = len(values)
    end = np.minimum(positions + int(horizon) - 1, n - 1)
    valid = (positions >= 0) & (positions < n)
    out = np.full(len(positions), np.nan, dtype=float)
    out[valid] = values[end[valid]]
    return out


def _window_min(values: np.ndarray, positions: np.ndarray, horizon: int) -> np.ndarray:
    n = len(values)
    out = np.full(len(positions), np.nan, dtype=float)
    for i, pos in enumerate(positions):
        start = int(pos)
        if start < 0 or start >= n:
            continue
        end = min(n, start + int(horizon))
        segment = values[start:end]
        if len(segment):
            out[i] = float(np.nanmin(segment))
    return out


def _period(ts: pd.Series) -> pd.Series:
    value = pd.to_datetime(ts, errors="coerce")
    return pd.Series(
        np.select(
            [value < pd.Timestamp("2025-01-01"), value < pd.Timestamp("2025-10-01")],
            ["EARLY_2023_2024", "MID_2025Q1_Q3"],
            default="BOOKS_2025Q4_2026H1",
        ),
        index=ts.index,
        dtype="object",
    )


def attach_stop_release_labels(events: pd.DataFrame, primary: pd.DataFrame, config: StructuredStopPoolConfig) -> pd.DataFrame:
    """Attach event-bar and 5m/15m order-flow release targets.

    These are labels describing what happened when/after the first sweep. They
    must never be used to classify a level before the sweep.
    """
    cfg = config.validate()
    if events.empty:
        return events.copy()
    bars = normalize_primary_bars(primary)
    out = events.copy().reset_index(drop=True)
    positions = pd.to_numeric(out["event_pos"], errors="raise").astype(np.int64).to_numpy()
    n = len(bars)
    if np.any((positions < 0) | (positions >= n)):
        raise ValueError("event_pos outside primary bars")
    out["period"] = _period(out["event_available_time"])

    numeric: dict[str, np.ndarray] = {}
    for name in (
        "open", "high", "low", "close", "notional", "buy_notional", "sell_notional",
        "delta_notional", "trades_count", "large_sell_notional", "large_sell_trades_count",
        "max_trade_notional",
    ):
        if name in bars.columns:
            numeric[name] = pd.to_numeric(bars[name], errors="coerce").to_numpy(dtype=float)
        else:
            numeric[name] = np.full(n, np.nan, dtype=float)

    baseline_fields = {
        "sell_notional": "median",
        "trades_count": "median",
        "large_sell_notional": "mean",
        "large_sell_trades_count": "mean",
        "max_trade_notional": "median",
    }
    baselines: dict[str, np.ndarray] = {}
    for name, method in baseline_fields.items():
        series = pd.Series(numeric[name])
        roller = series.shift(1).rolling(int(cfg.release_baseline_minutes), min_periods=20)
        base = roller.median() if method == "median" else roller.mean()
        baselines[name] = base.to_numpy(dtype=float)
        long_roller = series.shift(1).rolling(int(cfg.release_long_baseline_minutes), min_periods=60)
        long_base = long_roller.median() if method == "median" else long_roller.mean()
        baselines[f"{name}_long"] = long_base.to_numpy(dtype=float)

    prefixes = {name: _prefix(values) for name, values in numeric.items() if name not in {"open", "high", "low", "close", "max_trade_notional"}}
    open_event = numeric["open"][positions]
    low_event = numeric["low"][positions]
    close_event = numeric["close"][positions]
    out["release_event_bar_downside_bp"] = _safe_ratio(np.maximum(open_event - low_event, 0.0), open_event) * 10_000.0
    out["release_event_bar_close_off_low_bp"] = _safe_ratio(np.maximum(close_event - low_event, 0.0), open_event) * 10_000.0

    for horizon in cfg.release_windows_minutes:
        h = int(horizon)
        sell = _window_sum(prefixes["sell_notional"], positions, h, n)
        buy = _window_sum(prefixes["buy_notional"], positions, h, n)
        notional = _window_sum(prefixes["notional"], positions, h, n)
        delta = _window_sum(prefixes["delta_notional"], positions, h, n)
        trades = _window_sum(prefixes["trades_count"], positions, h, n)
        large_sell = _window_sum(prefixes["large_sell_notional"], positions, h, n)
        large_sell_count = _window_sum(prefixes["large_sell_trades_count"], positions, h, n)
        base_sell = baselines["sell_notional"][positions] * h
        base_trades = baselines["trades_count"][positions] * h
        base_large_sell = baselines["large_sell_notional"][positions] * h
        base_large_count = baselines["large_sell_trades_count"][positions] * h
        out[f"release_sell_notional_{h}m"] = sell
        out[f"release_trades_count_{h}m"] = trades
        out[f"release_large_sell_notional_{h}m"] = large_sell
        out[f"release_large_sell_trades_count_{h}m"] = large_sell_count
        out[f"release_sell_notional_{h}m_vs_prior60"] = _safe_ratio(sell, base_sell)
        out[f"release_trades_count_{h}m_vs_prior60"] = _safe_ratio(trades, base_trades)
        out[f"release_large_sell_notional_{h}m_vs_prior60"] = _safe_ratio(large_sell, base_large_sell)
        out[f"release_large_sell_count_{h}m_vs_prior60"] = _safe_ratio(large_sell_count, base_large_count)
        out[f"release_sell_share_{h}m"] = _safe_ratio(sell, buy + sell)
        out[f"release_negative_delta_ratio_{h}m"] = np.maximum(-_safe_ratio(delta, notional), 0.0)
        path_low = _window_min(numeric["low"], positions, h)
        terminal = _window_terminal(numeric["close"], positions, h)
        downside_bp = _safe_ratio(np.maximum(open_event - path_low, 0.0), open_event) * 10_000.0
        out[f"release_price_downside_{h}m_bp"] = downside_bp
        out[f"release_terminal_return_{h}m_bp"] = _safe_ratio(terminal - open_event, open_event) * 10_000.0
        out[f"release_sell_impact_bp_per_million_{h}m"] = _safe_ratio(downside_bp, sell / 1_000_000.0)

    max_trade = numeric["max_trade_notional"][positions]
    out["release_max_trade_notional_1m"] = max_trade
    out["release_max_trade_notional_1m_vs_prior60"] = _safe_ratio(max_trade, baselines["max_trade_notional"][positions])
    out["release_baseline_available"] = np.isfinite(baselines["sell_notional"][positions]) & np.isfinite(baselines["trades_count"][positions])
    return out


def calibrate_release_score(frame: pd.DataFrame, config: StructuredStopPoolConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit a robust score only on early matched controls and freeze it forward."""
    cfg = config.validate()
    if frame.empty:
        return frame.copy(), pd.DataFrame()
    out = frame.copy()
    components = (
        "release_sell_notional_5m_vs_prior60",
        "release_trades_count_5m_vs_prior60",
        "release_large_sell_notional_5m_vs_prior60",
        "release_max_trade_notional_1m_vs_prior60",
        "release_negative_delta_ratio_5m",
    )
    early = out["period"].astype(str).eq("EARLY_2023_2024")
    if "event_kind" in out.columns:
        control = out["event_kind"].astype(str).eq("non_zone_downside_control")
    else:
        control = pd.Series(False, index=out.index)
    calibration_mask = early & control & out["release_baseline_available"].fillna(False).astype(bool)
    calibration_source = "EARLY_MATCHED_CONTROLS"
    if int(calibration_mask.sum()) < 100:
        calibration_mask = early & out["release_baseline_available"].fillna(False).astype(bool)
        calibration_source = "EARLY_ALL_EVENTS_FALLBACK"
    rows: list[dict[str, Any]] = []
    z_columns: list[str] = []
    for name in components:
        raw = pd.to_numeric(out.get(name), errors="coerce")
        transformed = np.log1p(raw.clip(lower=0.0)) if "ratio" not in name or name != "release_negative_delta_ratio_5m" else raw
        sample = transformed.loc[calibration_mask].dropna()
        center = float(sample.median()) if len(sample) else np.nan
        q25 = float(sample.quantile(0.25)) if len(sample) else np.nan
        q75 = float(sample.quantile(0.75)) if len(sample) else np.nan
        scale = (q75 - q25) / 1.349 if np.isfinite(q75 - q25) and (q75 - q25) > EPS else float(sample.std(ddof=0)) if len(sample) else np.nan
        z_name = f"_release_z_{name}"
        if np.isfinite(center) and np.isfinite(scale) and scale > EPS:
            out[z_name] = ((transformed - center) / scale).clip(-5.0, 5.0)
        else:
            out[z_name] = np.nan
        z_columns.append(z_name)
        rows.append(
            {
                "component": name,
                "calibration_source": calibration_source,
                "calibration_rows": int(len(sample)),
                "center": center,
                "scale": scale,
                "q25": q25,
                "q75": q75,
            }
        )
    out["stop_release_score"] = out[z_columns].mean(axis=1, skipna=True)
    score_sample = pd.to_numeric(out.loc[calibration_mask, "stop_release_score"], errors="coerce").dropna()
    threshold = float(score_sample.quantile(cfg.release_score_quantile)) if len(score_sample) else np.nan
    out["high_stop_release_label"] = pd.to_numeric(out["stop_release_score"], errors="coerce").ge(threshold) if np.isfinite(threshold) else False
    out = out.drop(columns=z_columns)
    rows.append(
        {
            "component": "STOP_RELEASE_SCORE_THRESHOLD",
            "calibration_source": calibration_source,
            "calibration_rows": int(len(score_sample)),
            "center": threshold,
            "scale": np.nan,
            "q25": float(score_sample.quantile(0.25)) if len(score_sample) else np.nan,
            "q75": float(score_sample.quantile(0.75)) if len(score_sample) else np.nan,
        }
    )
    return out, pd.DataFrame(rows)
