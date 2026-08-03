#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast causal limit-fill and structural-stop replay for R10."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter
from src.research_common.swing_liquidity_atlas.pivots import normalize_primary_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex
from src.research_common.swing_liquidity_zone_study.outcomes import RangeMinMaxIndex

from .config import StructuredPullbackConfig, target_specs


def _datetime_ns(values: pd.Series | pd.Index) -> np.ndarray:
    """Return nanosecond epochs independent of pandas datetime storage unit."""

    parsed = pd.to_datetime(values, errors="coerce")
    return np.asarray(parsed, dtype="datetime64[ns]").astype(np.int64, copy=False)


def _position_at_or_after(index_ns: np.ndarray, values: pd.Series) -> np.ndarray:
    return np.searchsorted(index_ns, _datetime_ns(values), side="left").astype(np.int64)


def _safe_profit_factor(values: pd.Series) -> float:
    data = pd.to_numeric(values, errors="coerce").dropna()
    gains = float(data.loc[data > 0].sum())
    losses = float(-data.loc[data < 0].sum())
    if losses <= 0:
        return np.inf if gains > 0 else np.nan
    return gains / losses


def attach_limit_fills(
    family_candidates: pd.DataFrame,
    primary: pd.DataFrame,
    config: StructuredPullbackConfig,
    *,
    research_end_exclusive: pd.Timestamp,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Find the first legal retest fill before the next same-timeframe structure.

    The limit becomes active at ``structure_available_time``. The old order is
    cancelled when the next same-timeframe Swing Low becomes causally available.
    No arbitrary age expiry is introduced.
    """

    cfg = config.validate()
    if family_candidates.empty:
        return family_candidates.copy()
    bars = normalize_primary_bars(primary)
    index = pd.DatetimeIndex(bars.index)
    index_ns = _datetime_ns(index)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low_index = SegmentThresholdIndex(low)
    high_index = SegmentThresholdIndex(high)
    n = len(bars)
    research_end_pos = int(np.searchsorted(index_ns, pd.Timestamp(research_end_exclusive).value, side="left"))
    research_end_pos = min(n, max(0, research_end_pos))

    out = family_candidates.copy().reset_index(drop=True)
    signal_pos = _position_at_or_after(index_ns, out["structure_available_time"])
    next_signal = pd.to_datetime(
        out["next_same_timeframe_structure_available_time"], errors="coerce"
    )
    next_signal_pos = np.full(len(out), research_end_pos, dtype=np.int64)
    has_next = next_signal.notna().to_numpy()
    if has_next.any():
        next_signal_pos[has_next] = _position_at_or_after(index_ns, next_signal.loc[has_next])
    order_end_pos = np.minimum(next_signal_pos - 1, research_end_pos - 1)
    order_end_pos = np.minimum(order_end_pos, n - 1)

    # Fill is identical across families for the same level. Search once per level.
    unique = out.loc[:, ["level_id", "entry_limit_price"]].copy()
    unique["order_active_pos"] = signal_pos
    unique["order_end_pos"] = order_end_pos
    unique = unique.drop_duplicates("level_id", keep="first").reset_index(drop=True)
    fill_lookup: dict[int, dict[str, object]] = {}
    reporter = ProgressReporter(
        label="[r10] limit fills",
        total=len(unique),
        every=max(1, len(unique) // 200),
        enabled=show_progress,
    )
    for ordinal, row in enumerate(unique.itertuples(index=False), start=1):
        level_id = int(row.level_id)
        start = int(row.order_active_pos)
        end = int(row.order_end_pos)
        limit_price = float(row.entry_limit_price)
        fill_pos = -1
        status = "INVALID"
        fill_price = np.nan
        h0_before_fill_pos = -1
        if 0 <= start < n and start <= end and np.isfinite(limit_price) and limit_price > 0:
            fill_pos = low_index.first_leq(start, end, limit_price)
            if fill_pos >= 0:
                # A resting buy limit receives price improvement if the bar opens below it.
                fill_price = min(limit_price, float(open_[fill_pos]))
                status = "FILLED"
            else:
                status = "UNFILLED_NEXT_STRUCTURE_OR_END"
        fill_lookup[level_id] = {
            "order_active_pos": start,
            "order_end_pos": end,
            "fill_pos": fill_pos,
            "fill_price": fill_price,
            "fill_status": status,
            "fill_at_bar_open_or_better": bool(
                fill_pos >= 0 and np.isfinite(fill_price) and float(open_[fill_pos]) <= limit_price
            ),
            "h0_before_fill_pos_placeholder": h0_before_fill_pos,
        }
        reporter.update(ordinal)
    reporter.close()

    fill_frame = pd.DataFrame.from_dict(fill_lookup, orient="index")
    fill_frame.index.name = "level_id"
    fill_frame = fill_frame.reset_index()
    out = out.merge(fill_frame, on="level_id", how="left", validate="many_to_one")

    valid_active = out["order_active_pos"].between(0, n - 1)
    out["order_active_time"] = pd.NaT
    if valid_active.any():
        positions = out.loc[valid_active, "order_active_pos"].astype(np.int64).to_numpy()
        out.loc[valid_active, "order_active_time"] = index[positions].to_numpy()
    valid_end = out["order_end_pos"].between(0, n - 1)
    out["order_end_time"] = pd.NaT
    if valid_end.any():
        positions = out.loc[valid_end, "order_end_pos"].astype(np.int64).to_numpy()
        out.loc[valid_end, "order_end_time"] = index[positions].to_numpy()
    valid_fill = out["fill_pos"].ge(0)
    out["fill_time"] = pd.NaT
    if valid_fill.any():
        positions = out.loc[valid_fill, "fill_pos"].astype(np.int64).to_numpy()
        out.loc[valid_fill, "fill_time"] = index[positions].to_numpy()
    out["order_age_minutes_to_fill"] = np.where(
        valid_fill,
        pd.to_numeric(out["fill_pos"], errors="coerce")
        - pd.to_numeric(out["order_active_pos"], errors="coerce"),
        np.nan,
    )
    out["order_lifetime_minutes"] = (
        pd.to_numeric(out["order_end_pos"], errors="coerce")
        - pd.to_numeric(out["order_active_pos"], errors="coerce")
        + 1
    )

    # Opportunity-cost diagnostic: did H0 trade before the retest order filled?
    # H0 and fill timing are level-specific, so compute once per level and merge
    # back to overlapping hypothesis families.
    audit_unique = out.loc[
        :,
        [
            "level_id",
            "order_active_pos",
            "order_end_pos",
            "fill_pos",
            "entry_limit_price",
            "structural_target_h0_price",
        ],
    ].drop_duplicates("level_id", keep="first").reset_index(drop=True)
    h0_before = np.zeros(len(audit_unique), dtype=bool)
    h0_first_pos = np.full(len(audit_unique), -1, dtype=np.int64)
    reporter = ProgressReporter(
        label="[r10] missed breakout audit",
        total=len(audit_unique),
        every=max(1, len(audit_unique) // 200),
        enabled=show_progress,
    )
    for i, row in enumerate(audit_unique.itertuples(index=False)):
        start = int(row.order_active_pos)
        end = int(row.order_end_pos)
        target = float(row.structural_target_h0_price)
        fill_pos = int(row.fill_pos)
        if 0 <= start < n and start <= end and np.isfinite(target):
            pos = high_index.first_geq(start, end, target)
            h0_first_pos[i] = pos
            if pos >= 0:
                if fill_pos < 0 or pos < fill_pos:
                    h0_before[i] = True
                elif pos == fill_pos and float(open_[pos]) > float(row.entry_limit_price):
                    # Intrabar high may have happened before a later low-limit fill.
                    h0_before[i] = True
        reporter.update(i + 1)
    reporter.close()
    audit_unique["first_h0_trade_pos_from_activation"] = h0_first_pos
    audit_unique["h0_traded_before_fill_flag"] = h0_before
    out = out.drop(
        columns=["first_h0_trade_pos_from_activation", "h0_traded_before_fill_flag"],
        errors="ignore",
    ).merge(
        audit_unique.loc[
            :,
            ["level_id", "first_h0_trade_pos_from_activation", "h0_traded_before_fill_flag"],
        ],
        on="level_id",
        how="left",
        validate="many_to_one",
    )
    out = out.drop(columns=["h0_before_fill_pos_placeholder"], errors="ignore")
    return out


def attach_trade_outcomes(
    filled_candidates: pd.DataFrame,
    primary: pd.DataFrame,
    config: StructuredPullbackConfig,
    *,
    research_end_exclusive: pd.Timestamp,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Replay H0/1R/2R/3R targets with a fixed structural stop.

    Same-bar TP/SL is a stop. If the limit is touched intrabar rather than at
    the open, the fill bar cannot earn a TP because the bar high may have
    occurred before the entry. The stop remains valid because a lower stop must
    be crossed after the entry level.
    """

    cfg = config.validate()
    if filled_candidates.empty:
        return filled_candidates.copy()
    bars = normalize_primary_bars(primary)
    index = pd.DatetimeIndex(bars.index)
    index_ns = _datetime_ns(index)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    high_index = SegmentThresholdIndex(high)
    low_index = SegmentThresholdIndex(low)
    high_range = RangeMinMaxIndex(high)
    low_range = RangeMinMaxIndex(low)
    n = len(bars)
    research_end_pos = int(np.searchsorted(index_ns, pd.Timestamp(research_end_exclusive).value, side="left"))
    research_end_pos = min(n, max(0, research_end_pos))

    out = filled_candidates.copy().reset_index(drop=True)
    # ``Series.to_numpy()`` may expose a read-only view when pandas
    # Copy-on-Write is enabled.  This mask is refined below, so make the
    # ownership/writeability explicit instead of relying on pandas internals.
    valid_fill = (
        out["fill_status"].eq("FILLED")
        & out["family_geometry_valid"].astype(bool)
        & pd.to_numeric(out["fill_pos"], errors="coerce").ge(0)
    ).to_numpy(dtype=bool, copy=True)
    fill_pos = pd.to_numeric(out["fill_pos"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    entry = pd.to_numeric(out["fill_price"], errors="coerce").to_numpy(dtype=float)
    stop = pd.to_numeric(out["stop_price"], errors="coerce").to_numpy(dtype=float)
    source_tf = pd.to_numeric(out["source_timeframe_min"], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    risk_return = (entry - stop) / entry
    valid_fill = valid_fill & (
        np.isfinite(entry) & np.isfinite(stop) & (entry > stop) & (stop > 0)
    )
    out["actual_risk_distance_return"] = risk_return
    out["actual_risk_distance_bp"] = risk_return * 10_000.0
    out["holding_horizon_minutes"] = [
        cfg.holding_minutes(int(value)) if int(value) > 0 else 0 for value in source_tf
    ]

    target_columns: dict[str, np.ndarray] = {}
    for spec in target_specs():
        if spec.mode == "STRUCTURAL_H0":
            target = pd.to_numeric(out["structural_target_h0_price"], errors="coerce").to_numpy(dtype=float)
        else:
            target = entry + float(spec.r_multiple or 0.0) * (entry - stop)
        target_columns[spec.name] = target
        out[f"{spec.name.lower()}_target_price"] = target

    fee_cost = float(cfg.fee_round_trip)
    realistic_cost = float(cfg.realistic_round_trip_cost)
    stressed_cost = float(cfg.stressed_round_trip_cost)

    reporter = ProgressReporter(
        label="[r10] target/stop replay",
        total=int(valid_fill.sum()) * len(target_specs()),
        every=max(1, int(valid_fill.sum()) * len(target_specs()) // 250),
        enabled=show_progress,
    )
    completed = 0
    for spec in target_specs():
        token = spec.name.lower()
        target = target_columns[spec.name]
        outcome = np.full(len(out), "NOT_FILLED_OR_INVALID", dtype=object)
        exit_pos = np.full(len(out), -1, dtype=np.int64)
        first_tp = np.full(len(out), -1, dtype=np.int64)
        first_sl = np.full(len(out), -1, dtype=np.int64)
        same_bar_both = np.zeros(len(out), dtype=bool)
        gross = np.full(len(out), np.nan, dtype=float)
        mfe = np.full(len(out), np.nan, dtype=float)
        mae = np.full(len(out), np.nan, dtype=float)

        for i in np.flatnonzero(valid_fill):
            start = int(fill_pos[i])
            horizon = int(out.at[i, "holding_horizon_minutes"])
            end = min(n - 1, research_end_pos - 1, start + max(1, horizon) - 1)
            price = float(entry[i])
            stop_price = float(stop[i])
            target_price = float(target[i])
            if end < start or not np.isfinite(target_price) or target_price <= price:
                outcome[i] = "INVALID_TARGET"
                completed += 1
                reporter.update(completed)
                continue

            stop_pos = low_index.first_leq(start, end, stop_price)
            # If the order filled below its limit at the open, it was active at bar start.
            # Otherwise the bar high is not safely available after an intrabar low fill.
            target_start = start if float(open_[start]) <= float(out.at[i, "entry_limit_price"]) else start + 1
            tp_pos = high_index.first_geq(target_start, end, target_price) if target_start <= end else -1
            first_tp[i] = tp_pos
            first_sl[i] = stop_pos

            if stop_pos >= 0 and tp_pos >= 0 and stop_pos == tp_pos:
                same_bar_both[i] = True
                exit_pos[i] = stop_pos
                gross[i] = stop_price / price - 1.0
                outcome[i] = "SL_CONSERVATIVE_SAME_BAR"
            elif stop_pos >= 0 and (tp_pos < 0 or stop_pos < tp_pos):
                exit_pos[i] = stop_pos
                gross[i] = stop_price / price - 1.0
                outcome[i] = "SL"
            elif tp_pos >= 0:
                exit_pos[i] = tp_pos
                gross[i] = target_price / price - 1.0
                outcome[i] = "TP"
            else:
                exit_pos[i] = end
                gross[i] = close[end] / price - 1.0
                outcome[i] = "TIME"

            low_value, _ = low_range.query(start, int(exit_pos[i]))
            if float(open_[start]) <= float(out.at[i, "entry_limit_price"]):
                _, high_value = high_range.query(start, int(exit_pos[i]))
            else:
                # The fill happened after the bar opened. Its earlier high is not
                # safely post-entry, so use the closing price for the fill bar and
                # start high-range measurement from the next bar.
                safe_fill_bar_high = max(price, float(close[start]))
                _, later_high = high_range.query(start + 1, int(exit_pos[i]))
                high_value = max(safe_fill_bar_high, later_high) if np.isfinite(later_high) else safe_fill_bar_high
            mae[i] = low_value / price - 1.0 if np.isfinite(low_value) else np.nan
            mfe[i] = high_value / price - 1.0 if np.isfinite(high_value) else np.nan
            completed += 1
            reporter.update(completed)

        out[f"{token}_first_tp_pos"] = first_tp
        out[f"{token}_first_sl_pos"] = first_sl
        out[f"{token}_exit_pos"] = exit_pos
        out[f"{token}_outcome"] = outcome
        out[f"{token}_same_bar_both_flag"] = same_bar_both
        out[f"{token}_tp_before_sl"] = outcome == "TP"
        out[f"{token}_gross_return"] = gross
        out[f"{token}_net_return_fee_only"] = gross - fee_cost
        out[f"{token}_net_return_realistic"] = gross - realistic_cost
        out[f"{token}_net_return_2x_cost"] = gross - stressed_cost
        out[f"{token}_gross_r"] = gross / risk_return
        out[f"{token}_net_r_fee_only"] = (gross - fee_cost) / risk_return
        out[f"{token}_net_r_realistic"] = (gross - realistic_cost) / risk_return
        out[f"{token}_net_r_2x_cost"] = (gross - stressed_cost) / risk_return
        out[f"{token}_mae_return"] = mae
        out[f"{token}_mfe_return"] = mfe
        out[f"{token}_bars_to_exit"] = np.where(exit_pos >= 0, exit_pos - fill_pos + 1, np.nan)
        out[f"{token}_exit_time"] = pd.NaT
        mask = exit_pos >= 0
        if mask.any():
            out.loc[mask, f"{token}_exit_time"] = index[exit_pos[mask]].to_numpy()

    reporter.close()
    out["valid_filled_trade"] = valid_fill
    return out


__all__ = [
    "attach_limit_fills",
    "attach_trade_outcomes",
    "_safe_profit_factor",
]
