#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reusable event-study runner."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .causal import audit_context_available_times, audit_next_open_entries, ensure_datetime_index
from .models import EventStudyConfig, EventStudyResult
from .outcomes import cost_adjust_return, normalize_side
from .stats import summarize_many
from src.research_common.progress import ProgressReporter

_REQUIRED_BAR_COLS = ("open", "high", "low", "close")


def _require_bar_columns(bars: pd.DataFrame) -> None:
    missing = [col for col in _REQUIRED_BAR_COLS if col not in bars.columns]
    if missing:
        raise KeyError(f"bars is missing required columns: {missing}")


def _normalize_events(events: pd.DataFrame, cfg: EventStudyConfig) -> pd.DataFrame:
    out = events.copy()
    if cfg.signal_time_col not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out[cfg.signal_time_col] = out.index
        else:
            raise KeyError(f"events must contain {cfg.signal_time_col} or use a DatetimeIndex")
    out[cfg.signal_time_col] = pd.to_datetime(out[cfg.signal_time_col], errors="coerce")
    if cfg.side_col not in out.columns:
        raise KeyError(f"events is missing side column: {cfg.side_col}")
    out["side"] = normalize_side(out[cfg.side_col])
    out = out[out["side"] != 0].copy()
    out = out.dropna(subset=[cfg.signal_time_col]).sort_values(cfg.signal_time_col)
    if cfg.event_name_col and cfg.event_name_col not in out.columns:
        out[cfg.event_name_col] = "event"
    return out


def _attach_bar_positions(bars: pd.DataFrame, events: pd.DataFrame, cfg: EventStudyConfig) -> pd.DataFrame:
    out = events.copy()
    signal_times = pd.DatetimeIndex(out[cfg.signal_time_col])
    positions = bars.index.get_indexer(signal_times)
    out["signal_bar_pos"] = positions
    out["signal_on_bar_index_flag"] = positions >= 0
    return out


def _attach_return_labels(frame: pd.DataFrame, enriched: pd.DataFrame, cfg: EventStudyConfig) -> pd.DataFrame:
    out = enriched.copy()
    close = pd.to_numeric(frame["close"], errors="coerce")
    open_ = pd.to_numeric(frame["open"], errors="coerce")
    valid_mask = out["signal_on_bar_index_flag"].astype(bool)
    valid_index = out.index[valid_mask]
    if len(valid_index) == 0:
        for horizon in cfg.horizons:
            h = int(horizon)
            out[f"close_to_close_ret_h{h}"] = pd.NA
            out[f"next_open_ret_h{h}_gross"] = pd.NA
            out[f"next_open_ret_h{h}_net"] = pd.NA
        return out

    pos = out.loc[valid_index, "signal_bar_pos"].astype(int).to_numpy()
    side = out.loc[valid_index, "side"].astype(float).to_numpy()
    for horizon in cfg.horizons:
        h = int(horizon)
        close_to_close_col = f"close_to_close_ret_h{h}"
        gross_col = f"next_open_ret_h{h}_gross"
        net_col = f"next_open_ret_h{h}_net"
        out[close_to_close_col] = pd.NA
        out[gross_col] = pd.NA
        out[net_col] = pd.NA

        future_close = close.shift(-h).to_numpy(dtype=float)
        signal_close = close.to_numpy(dtype=float)
        entry_open = open_.shift(-int(cfg.entry_delay_bars)).to_numpy(dtype=float)

        ctc = (future_close[pos] / signal_close[pos] - 1.0) * side
        gross = (future_close[pos] / entry_open[pos] - 1.0) * side
        net = cost_adjust_return(pd.Series(gross, index=valid_index), cfg.cost)
        out.loc[valid_index, close_to_close_col] = ctc
        out.loc[valid_index, gross_col] = gross
        out.loc[valid_index, net_col] = net.to_numpy()
    return out


def _attach_mfe_mae_labels(frame: pd.DataFrame, enriched: pd.DataFrame, cfg: EventStudyConfig) -> pd.DataFrame:
    out = enriched.copy()
    mfe_col = f"mfe_h{int(cfg.mfe_mae_horizon)}"
    mae_col = f"mae_h{int(cfg.mfe_mae_horizon)}"
    out[mfe_col] = pd.NA
    out[mae_col] = pd.NA
    valid = out[out["signal_on_bar_index_flag"]].copy()
    if valid.empty:
        return out

    highs = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    h = int(cfg.mfe_mae_horizon)
    delay = int(cfg.entry_delay_bars)
    progress = ProgressReporter(
        label="[event-study] MFE/MAE",
        total=len(valid),
        every=int(cfg.progress_every),
        enabled=bool(int(cfg.progress_every) > 0),
    )
    for done, (row_idx, row) in enumerate(valid.iterrows(), start=1):
        pos = int(row["signal_bar_pos"])
        side = int(row["side"])
        entry_pos = pos + delay
        if entry_pos < len(frame):
            end = min(len(frame), pos + h + 1)
            if end > entry_pos:
                entry = opens[entry_pos]
                if np.isfinite(entry) and entry > 0:
                    high_path = highs[entry_pos:end]
                    low_path = lows[entry_pos:end]
                    if side == 1:
                        path_high = np.nanmax(high_path) if high_path.size else np.nan
                        path_low = np.nanmin(low_path) if low_path.size else np.nan
                        mfe = path_high / entry - 1.0 if np.isfinite(path_high) else np.nan
                        mae = path_low / entry - 1.0 if np.isfinite(path_low) else np.nan
                    else:
                        path_low = np.nanmin(low_path) if low_path.size else np.nan
                        path_high = np.nanmax(high_path) if high_path.size else np.nan
                        mfe = entry / path_low - 1.0 if np.isfinite(path_low) and path_low > 0 else np.nan
                        mae = entry / path_high - 1.0 if np.isfinite(path_high) and path_high > 0 else np.nan
                    out.loc[row_idx, mfe_col] = mfe
                    out.loc[row_idx, mae_col] = mae
        progress.update(done)
    progress.close()
    return out


def run_event_study(bars: pd.DataFrame, events: pd.DataFrame, config: EventStudyConfig | None = None) -> EventStudyResult:
    """Run a closed-bar signal / next-open event study.

    Parameters
    ----------
    bars:
        OHLCV execution axis indexed by bar timestamp. Required columns:
        open/high/low/close.
    events:
        Event rows with signal_time and side columns. If signal_time is missing,
        a DatetimeIndex is used as signal_time. side supports +1/-1, LONG/SHORT,
        BUY/SELL, UP/DOWN.
    config:
        EventStudyConfig. Defaults are safe for ETH perpetual research using the
        project's round-trip fee convention.
    """
    cfg = config or EventStudyConfig()
    frame = ensure_datetime_index(bars, name="bars")
    _require_bar_columns(frame)
    ev = _normalize_events(events, cfg)
    enriched = _attach_bar_positions(frame, ev, cfg)
    enriched = _attach_return_labels(frame, enriched, cfg)
    enriched = _attach_mfe_mae_labels(frame, enriched, cfg)

    enriched["entry_bar_pos"] = enriched["signal_bar_pos"] + int(cfg.entry_delay_bars)
    in_entry_range = enriched["entry_bar_pos"].between(0, len(frame) - 1)
    enriched["entry_time"] = pd.NaT
    enriched["entry_price"] = pd.NA
    if in_entry_range.any():
        entry_pos = enriched.loc[in_entry_range, "entry_bar_pos"].astype(int).to_numpy()
        enriched.loc[in_entry_range, "entry_time"] = frame.index[entry_pos]
        enriched.loc[in_entry_range, "entry_price"] = pd.to_numeric(frame["open"], errors="coerce").iloc[entry_pos].to_numpy()
    enriched["side_name"] = enriched["side"].map({1: "LONG", -1: "SHORT"}).fillna("FLAT")
    enriched["year"] = pd.to_datetime(enriched[cfg.signal_time_col]).dt.year
    enriched["round_trip_cost_pct"] = float(cfg.cost.round_trip_cost_pct)

    next_open_audit = audit_next_open_entries(enriched, signal_time_col=cfg.signal_time_col, entry_time_col="entry_time")
    context_audit = audit_context_available_times(
        enriched,
        signal_time_col=cfg.signal_time_col,
        context_available_time_cols=cfg.context_available_time_cols,
    )
    causal_audit = pd.concat(
        [
            next_open_audit.drop(columns=["signal_time"], errors="ignore"),
            context_audit.drop(columns=["signal_time"], errors="ignore"),
            enriched[["signal_on_bar_index_flag"]],
        ],
        axis=1,
    )
    causal_audit["causal_fail_flag"] = (
        (~causal_audit["signal_on_bar_index_flag"].astype(bool))
        | causal_audit.get("entry_not_after_signal_flag", False).astype(bool)
        | causal_audit.get("context_available_time_flag", False).astype(bool)
    )

    return_cols = [f"next_open_ret_h{int(h)}_net" for h in cfg.horizons]
    overview = summarize_many(enriched, return_cols, min_count=cfg.min_count)
    yearly = summarize_many(enriched, return_cols, group_cols=["year"], min_count=cfg.min_count)
    side_stats = summarize_many(enriched, return_cols, group_cols=["side_name"], min_count=cfg.min_count)
    horizon_stats = overview.copy()

    meta: dict[str, Any] = {
        "event_count": int(len(enriched)),
        "valid_signal_count": int(enriched["signal_on_bar_index_flag"].sum()) if not enriched.empty else 0,
        "entry_assumption": cfg.entry_assumption,
        "entry_delay_bars": int(cfg.entry_delay_bars),
        "horizons": tuple(int(h) for h in cfg.horizons),
        "mfe_mae_horizon": int(cfg.mfe_mae_horizon),
        "round_trip_cost_pct": float(cfg.cost.round_trip_cost_pct),
        "causal_fail_count": int(causal_audit["causal_fail_flag"].sum()) if not causal_audit.empty else 0,
    }
    return EventStudyResult(
        events=enriched,
        overview=overview,
        yearly=yearly,
        side_stats=side_stats,
        horizon_stats=horizon_stats,
        causal_audit=causal_audit,
        meta=meta,
    )
