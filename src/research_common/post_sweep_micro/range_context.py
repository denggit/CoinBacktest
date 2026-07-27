#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal Range-Bar context extraction around R06 attempt minutes."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader

from .config import PostSweepMicroConfig

EPS = 1e-12


def _ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= EPS:
        return np.nan
    return float(a / b)


def _bar_features(prefix: str, row: pd.Series | None) -> dict[str, Any]:
    if row is None:
        return {
            f"{prefix}_present": False,
            f"{prefix}_bar_id": np.nan,
            f"{prefix}_end_ts": pd.NaT,
            f"{prefix}_direction": np.nan,
            f"{prefix}_duration_seconds": np.nan,
            f"{prefix}_range_bp": np.nan,
            f"{prefix}_delta_ratio": np.nan,
            f"{prefix}_sell_share": np.nan,
            f"{prefix}_downside_bp_per_sell_million": np.nan,
            f"{prefix}_notional": np.nan,
            f"{prefix}_max_trade_notional": np.nan,
        }
    notional = float(row.get("notional", np.nan))
    buy = float(row.get("buy_notional", np.nan))
    sell = float(row.get("sell_notional", np.nan))
    delta = float(row.get("delta_notional", np.nan))
    open_price = float(row.get("open", np.nan))
    close_price = float(row.get("close", np.nan))
    direction = float(row.get("direction", np.nan))
    range_bp = abs(close_price / open_price - 1.0) * 10_000.0 if open_price > 0 else np.nan
    downside = range_bp if direction < 0 else 0.0
    return {
        f"{prefix}_present": True,
        f"{prefix}_bar_id": float(row.get("bar_id", np.nan)),
        f"{prefix}_end_ts": pd.Timestamp(row.get("end_ts")),
        f"{prefix}_direction": direction,
        f"{prefix}_duration_seconds": float(row.get("duration_seconds", np.nan)),
        f"{prefix}_range_bp": range_bp,
        f"{prefix}_delta_ratio": _ratio(delta, notional),
        f"{prefix}_sell_share": _ratio(sell, buy + sell),
        f"{prefix}_downside_bp_per_sell_million": _ratio(downside, sell / 1_000_000.0),
        f"{prefix}_notional": notional,
        f"{prefix}_max_trade_notional": float(row.get("max_trade_notional", np.nan)),
    }


def analyze_event_range_context(event: pd.Series, bars: pd.DataFrame, range_pct: float) -> dict[str, Any]:
    anchor = pd.Timestamp(event["checkpoint_time"])
    minute_end = anchor + pd.Timedelta(minutes=1)
    result: dict[str, Any] = {
        "window_id": event["window_id"],
        "checkpoint_id": event["checkpoint_id"],
        "zone_event_id": event["zone_event_id"],
        "pair_id": event["pair_id"],
        "cohort": event["cohort"],
        "period": event["period"],
        "checkpoint_time": anchor,
        "range_pct": float(range_pct),
    }
    if bars.empty:
        result["status"] = "no_range_bars"
        return result
    local = bars.copy().reset_index(drop=True)
    local["end_ts"] = pd.to_datetime(local["end_ts"], errors="coerce")
    local = local.dropna(subset=["end_ts"]).sort_values(["end_ts", "bar_id"], kind="mergesort")
    before = local.loc[local["end_ts"] <= anchor]
    through_minute = local.loc[local["end_ts"] <= minute_end]
    after_anchor = local.loc[(local["end_ts"] > anchor) & (local["end_ts"] <= anchor + pd.Timedelta(minutes=15))]

    last_before = before.iloc[-1] if not before.empty else None
    down_through = through_minute.loc[pd.to_numeric(through_minute["direction"], errors="coerce") < 0]
    last_down = down_through.iloc[-1] if not down_through.empty else None
    previous_down = down_through.iloc[-2] if len(down_through) >= 2 else None
    up_after = after_anchor.loc[pd.to_numeric(after_anchor["direction"], errors="coerce") > 0]
    first_up = up_after.iloc[0] if not up_after.empty else None

    result.update(_bar_features("last_before", last_before))
    result.update(_bar_features("last_down", last_down))
    result.update(_bar_features("previous_down", previous_down))
    result.update(_bar_features("first_up", first_up))
    result["range_bars_ending_in_attempt_minute"] = int(
        ((local["end_ts"] > anchor) & (local["end_ts"] <= minute_end)).sum()
    )
    result["down_range_bars_ending_in_attempt_minute"] = int(
        ((local["end_ts"] > anchor) & (local["end_ts"] <= minute_end) & (pd.to_numeric(local["direction"]) < 0)).sum()
    )
    result["up_range_bars_ending_in_attempt_minute"] = int(
        ((local["end_ts"] > anchor) & (local["end_ts"] <= minute_end) & (pd.to_numeric(local["direction"]) > 0)).sum()
    )
    if first_up is not None:
        first_up_end = pd.Timestamp(first_up["end_ts"])
        result["first_up_delay_seconds"] = float((first_up_end - anchor).total_seconds())
        preceding = local.loc[(local["end_ts"] > anchor) & (local["end_ts"] < first_up_end)]
        result["down_bars_before_first_up"] = int((pd.to_numeric(preceding["direction"]) < 0).sum())
        result["first_up_within_60s"] = bool(first_up_end <= minute_end)
        result["first_up_within_120s"] = bool(first_up_end <= anchor + pd.Timedelta(seconds=120))
    else:
        result["first_up_delay_seconds"] = np.nan
        result["down_bars_before_first_up"] = np.nan
        result["first_up_within_60s"] = False
        result["first_up_within_120s"] = False
    result["last_down_impact_ratio_vs_previous_down"] = _ratio(
        float(result.get("last_down_downside_bp_per_sell_million", np.nan)),
        float(result.get("previous_down_downside_bp_per_sell_million", np.nan)),
    )
    result["last_down_duration_ratio_vs_previous_down"] = _ratio(
        float(result.get("last_down_duration_seconds", np.nan)),
        float(result.get("previous_down_duration_seconds", np.nan)),
    )
    result["range_impact_weaker_flag"] = bool(
        np.isfinite(result["last_down_impact_ratio_vs_previous_down"])
        and result["last_down_impact_ratio_vs_previous_down"] <= 0.67
    )
    result["status"] = "complete"
    return result



def _take_numeric(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    out = np.full(len(indices), np.nan, dtype=float)
    valid = indices >= 0
    if valid.any():
        out[valid] = values[indices[valid]]
    return out


def _take_datetime_ns(values: np.ndarray, indices: np.ndarray) -> pd.Series:
    out = np.full(len(indices), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid = indices >= 0
    if valid.any():
        out[valid] = values[indices[valid]]
    return pd.Series(out)


def _vectorized_bar_features(
    prefix: str,
    indices: np.ndarray,
    *,
    end_ns: np.ndarray,
    bar_id: np.ndarray,
    direction: np.ndarray,
    duration: np.ndarray,
    open_price: np.ndarray,
    close_price: np.ndarray,
    notional: np.ndarray,
    buy: np.ndarray,
    sell: np.ndarray,
    delta: np.ndarray,
    max_trade_notional: np.ndarray,
) -> dict[str, Any]:
    present = indices >= 0
    selected_bar_id = _take_numeric(bar_id, indices)
    selected_direction = _take_numeric(direction, indices)
    selected_duration = _take_numeric(duration, indices)
    selected_open = _take_numeric(open_price, indices)
    selected_close = _take_numeric(close_price, indices)
    selected_notional = _take_numeric(notional, indices)
    selected_buy = _take_numeric(buy, indices)
    selected_sell = _take_numeric(sell, indices)
    selected_delta = _take_numeric(delta, indices)
    selected_max_trade = _take_numeric(max_trade_notional, indices)

    range_bp = np.full(len(indices), np.nan, dtype=float)
    valid_price = present & np.isfinite(selected_open) & (selected_open > 0) & np.isfinite(selected_close)
    range_bp[valid_price] = np.abs(selected_close[valid_price] / selected_open[valid_price] - 1.0) * 10_000.0
    delta_ratio = np.divide(
        selected_delta, selected_notional,
        out=np.full(len(indices), np.nan, dtype=float),
        where=np.isfinite(selected_delta) & np.isfinite(selected_notional) & (np.abs(selected_notional) > EPS),
    )
    total_side = selected_buy + selected_sell
    sell_share = np.divide(
        selected_sell, total_side,
        out=np.full(len(indices), np.nan, dtype=float),
        where=np.isfinite(selected_sell) & np.isfinite(total_side) & (np.abs(total_side) > EPS),
    )
    downside = np.where(selected_direction < 0, range_bp, 0.0)
    sell_million = selected_sell / 1_000_000.0
    downside_per_sell = np.divide(
        downside, sell_million,
        out=np.full(len(indices), np.nan, dtype=float),
        where=np.isfinite(downside) & np.isfinite(sell_million) & (np.abs(sell_million) > EPS),
    )
    return {
        f"{prefix}_present": present,
        f"{prefix}_bar_id": selected_bar_id,
        f"{prefix}_end_ts": _take_datetime_ns(end_ns.astype("datetime64[ns]"), indices).to_numpy(),
        f"{prefix}_direction": selected_direction,
        f"{prefix}_duration_seconds": selected_duration,
        f"{prefix}_range_bp": range_bp,
        f"{prefix}_delta_ratio": delta_ratio,
        f"{prefix}_sell_share": sell_share,
        f"{prefix}_downside_bp_per_sell_million": downside_per_sell,
        f"{prefix}_notional": selected_notional,
        f"{prefix}_max_trade_notional": selected_max_trade,
    }


def _analyze_chunk_vectorized(events: pd.DataFrame, bars: pd.DataFrame, range_pct: float) -> pd.DataFrame:
    """Extract Range context for one calendar chunk in O(events log bars).

    The prior implementation sliced the full Range frame for every event, which
    caused tens of thousands of repeated Boolean scans. This implementation
    sorts once and uses searchsorted against all/down/up bar timelines.
    """

    meta_columns = [
        "window_id", "checkpoint_id", "zone_event_id", "pair_id",
        "cohort", "period", "checkpoint_time",
    ]
    out = events.loc[:, meta_columns].copy().reset_index(drop=True)
    out["checkpoint_time"] = pd.to_datetime(out["checkpoint_time"], errors="coerce")
    out["range_pct"] = float(range_pct)
    if bars.empty:
        out["status"] = "no_range_bars"
        return out

    local = bars.copy().reset_index(drop=True)
    local["end_ts"] = pd.to_datetime(local["end_ts"], errors="coerce")
    local = local.dropna(subset=["end_ts"]).sort_values(["end_ts", "bar_id"], kind="mergesort").reset_index(drop=True)
    if local.empty:
        out["status"] = "no_range_bars"
        return out

    end_ns = local["end_ts"].to_numpy(dtype="datetime64[ns]").astype("int64")
    bar_id = pd.to_numeric(local["bar_id"], errors="coerce").to_numpy(dtype=float)
    direction = pd.to_numeric(local["direction"], errors="coerce").to_numpy(dtype=float)
    duration = pd.to_numeric(local.get("duration_seconds"), errors="coerce").to_numpy(dtype=float)
    open_price = pd.to_numeric(local["open"], errors="coerce").to_numpy(dtype=float)
    close_price = pd.to_numeric(local["close"], errors="coerce").to_numpy(dtype=float)
    notional = pd.to_numeric(local.get("notional"), errors="coerce").to_numpy(dtype=float)
    buy = pd.to_numeric(local.get("buy_notional"), errors="coerce").to_numpy(dtype=float)
    sell = pd.to_numeric(local.get("sell_notional"), errors="coerce").to_numpy(dtype=float)
    delta = pd.to_numeric(local.get("delta_notional"), errors="coerce").to_numpy(dtype=float)
    max_trade = pd.to_numeric(local.get("max_trade_notional"), errors="coerce").to_numpy(dtype=float)

    anchors_ns = out["checkpoint_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    minute_ns = anchors_ns + int(pd.Timedelta(minutes=1).value)
    horizon_15m_ns = anchors_ns + int(pd.Timedelta(minutes=15).value)

    down_idx = np.flatnonzero(direction < 0)
    up_idx = np.flatnonzero(direction > 0)
    down_ns = end_ns[down_idx]
    up_ns = end_ns[up_idx]

    last_before = np.searchsorted(end_ns, anchors_ns, side="right") - 1
    last_before[last_before < 0] = -1

    last_down_pos = np.searchsorted(down_ns, minute_ns, side="right") - 1
    last_down = np.full(len(out), -1, dtype=int)
    valid_last_down = last_down_pos >= 0
    last_down[valid_last_down] = down_idx[last_down_pos[valid_last_down]]

    previous_down_pos = last_down_pos - 1
    previous_down = np.full(len(out), -1, dtype=int)
    valid_previous_down = previous_down_pos >= 0
    previous_down[valid_previous_down] = down_idx[previous_down_pos[valid_previous_down]]

    first_up_pos = np.searchsorted(up_ns, anchors_ns, side="right")
    first_up = np.full(len(out), -1, dtype=int)
    valid_first_up_pos = first_up_pos < len(up_ns)
    candidate_times = np.full(len(out), np.iinfo(np.int64).max, dtype=np.int64)
    candidate_times[valid_first_up_pos] = up_ns[first_up_pos[valid_first_up_pos]]
    valid_first_up = valid_first_up_pos & (candidate_times <= horizon_15m_ns)
    first_up[valid_first_up] = up_idx[first_up_pos[valid_first_up]]

    feature_args = {
        "end_ns": end_ns, "bar_id": bar_id, "direction": direction, "duration": duration,
        "open_price": open_price, "close_price": close_price, "notional": notional,
        "buy": buy, "sell": sell, "delta": delta, "max_trade_notional": max_trade,
    }
    for prefix, indices in (
        ("last_before", last_before),
        ("last_down", last_down),
        ("previous_down", previous_down),
        ("first_up", first_up),
    ):
        for column, values in _vectorized_bar_features(prefix, indices, **feature_args).items():
            out[column] = values

    all_left = np.searchsorted(end_ns, anchors_ns, side="right")
    all_right = np.searchsorted(end_ns, minute_ns, side="right")
    down_left = np.searchsorted(down_ns, anchors_ns, side="right")
    down_right = np.searchsorted(down_ns, minute_ns, side="right")
    up_left = np.searchsorted(up_ns, anchors_ns, side="right")
    up_right = np.searchsorted(up_ns, minute_ns, side="right")
    out["range_bars_ending_in_attempt_minute"] = all_right - all_left
    out["down_range_bars_ending_in_attempt_minute"] = down_right - down_left
    out["up_range_bars_ending_in_attempt_minute"] = up_right - up_left

    first_up_time_ns = np.full(len(out), np.iinfo(np.int64).min, dtype=np.int64)
    first_up_time_ns[valid_first_up] = end_ns[first_up[valid_first_up]]
    delay = np.full(len(out), np.nan, dtype=float)
    delay[valid_first_up] = (first_up_time_ns[valid_first_up] - anchors_ns[valid_first_up]) / 1_000_000_000.0
    out["first_up_delay_seconds"] = delay
    down_before = np.full(len(out), np.nan, dtype=float)
    if valid_first_up.any():
        before_right = np.searchsorted(down_ns, first_up_time_ns[valid_first_up], side="left")
        down_before[valid_first_up] = before_right - down_left[valid_first_up]
    out["down_bars_before_first_up"] = down_before
    out["first_up_within_60s"] = valid_first_up & (first_up_time_ns <= minute_ns)
    out["first_up_within_120s"] = valid_first_up & (first_up_time_ns <= anchors_ns + int(pd.Timedelta(seconds=120).value))

    last_impact = pd.to_numeric(out["last_down_downside_bp_per_sell_million"], errors="coerce").to_numpy(dtype=float)
    previous_impact = pd.to_numeric(out["previous_down_downside_bp_per_sell_million"], errors="coerce").to_numpy(dtype=float)
    impact_ratio = np.divide(
        last_impact, previous_impact,
        out=np.full(len(out), np.nan, dtype=float),
        where=np.isfinite(last_impact) & np.isfinite(previous_impact) & (np.abs(previous_impact) > EPS),
    )
    out["last_down_impact_ratio_vs_previous_down"] = impact_ratio
    last_duration = pd.to_numeric(out["last_down_duration_seconds"], errors="coerce").to_numpy(dtype=float)
    previous_duration = pd.to_numeric(out["previous_down_duration_seconds"], errors="coerce").to_numpy(dtype=float)
    out["last_down_duration_ratio_vs_previous_down"] = np.divide(
        last_duration, previous_duration,
        out=np.full(len(out), np.nan, dtype=float),
        where=np.isfinite(last_duration) & np.isfinite(previous_duration) & (np.abs(previous_duration) > EPS),
    )
    out["range_impact_weaker_flag"] = np.isfinite(impact_ratio) & (impact_ratio <= 0.67)
    out["status"] = "complete"
    return out

def extract_range_context(
    events: pd.DataFrame,
    config: PostSweepMicroConfig,
    *,
    symbol: str,
    data_dir: str | None,
    db_name: str,
    chunk_days: int = 30,
    progress_callback: Callable[[int, int, float, pd.Timestamp, pd.Timestamp], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load bounded calendar chunks and extract all configured Range-Bar contexts."""

    cfg = config.validate()
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    required_event_columns = [
        "window_id", "checkpoint_id", "zone_event_id", "pair_id",
        "cohort", "period", "checkpoint_time",
    ]
    missing_event_columns = sorted(set(required_event_columns) - set(events.columns))
    if missing_event_columns:
        raise ValueError(f"range context events missing columns: {missing_event_columns}")
    event_frame = events.loc[:, required_event_columns].copy()
    anchors = pd.to_datetime(event_frame["checkpoint_time"], errors="coerce")
    start = anchors.min().normalize()
    end = (anchors.max().normalize() + pd.Timedelta(days=1))
    chunk_delta = pd.Timedelta(days=max(1, int(chunk_days)))
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    while cur < end:
        nxt = min(cur + chunk_delta, end)
        chunks.append((cur, nxt))
        cur = nxt

    output_parts: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    total = len(cfg.range_pcts) * len(chunks)
    done = 0
    loaders = [
        OKXRangeBarLoader(
            symbol=symbol,
            range_pct=float(range_pct),
            data_dir=data_dir,
            db_name=db_name,
            initialize_db=False,
        )
        for range_pct in cfg.range_pcts
    ]
    shared_connection = loaders[0].open_read_connection()
    try:
        for range_pct, loader in zip(cfg.range_pcts, loaders, strict=True):
            for core_start, core_end in chunks:
                done += 1
                mask = (anchors >= core_start) & (anchors < core_end)
                subset = event_frame.loc[mask].copy()
                if subset.empty:
                    if progress_callback:
                        progress_callback(done, total, float(range_pct), core_start, core_end)
                    continue
                query_start = core_start - pd.Timedelta(minutes=cfg.range_lookback_minutes)
                query_end = core_end + pd.Timedelta(minutes=cfg.range_lookforward_minutes)
                bars = loader.load_local_data(
                    query_start,
                    query_end,
                    connection=shared_connection,
                    columns=(
                        "bar_id", "end_ts", "direction", "duration_seconds",
                        "open", "close", "notional", "buy_notional",
                        "sell_notional", "delta_notional", "max_trade_notional",
                    ),
                )
                audits.append(
                    {
                        "range_pct": float(range_pct),
                        "core_start": core_start,
                        "core_end": core_end,
                        "events": int(len(subset)),
                        "range_rows": int(len(bars)),
                        "status": "loaded" if not bars.empty else "missing_range_cache",
                    }
                )
                chunk_output = _analyze_chunk_vectorized(subset, bars, float(range_pct))
                output_parts.append(chunk_output)
                if progress_callback:
                    progress_callback(done, total, float(range_pct), core_start, core_end)
                del bars, chunk_output, subset
    finally:
        shared_connection.close()

    output = pd.concat(output_parts, ignore_index=True, sort=False) if output_parts else pd.DataFrame()
    return output, pd.DataFrame(audits)


__all__ = ["analyze_event_range_context", "extract_range_context"]
