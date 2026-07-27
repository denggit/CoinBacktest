#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Second-level features and causal trigger/path evaluation for R06."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import PostSweepMicroConfig

EPS = 1e-12
FLOW_COLUMNS: tuple[str, ...] = (
    "volume", "trades_count", "buy_volume", "sell_volume", "notional",
    "buy_notional", "sell_notional", "buy_trades_count", "sell_trades_count",
    "delta_volume", "delta_notional", "large_buy_notional",
    "large_sell_notional", "large_delta_notional", "large_buy_trades_count",
    "large_sell_trades_count", "large_trades_count",
)

TRIGGER_NAMES: tuple[str, ...] = (
    "FIRST_NEW_LOW",
    "IMPACT_COLLAPSE_67",
    "IMPACT_COLLAPSE_50",
    "IMPACT_COLLAPSE_50_HIGH_BREAK",
    "MICRO_RECLAIM_5S",
    "MINUTE_CLOSE",
    "ORACLE_LOW_PLUS_1S",
)


def _prefix(values: np.ndarray) -> np.ndarray:
    out = np.empty(len(values) + 1, dtype=np.float64)
    out[0] = 0.0
    np.cumsum(np.nan_to_num(values, nan=0.0), out=out[1:])
    return out


def _window_sum(prefix: np.ndarray, end_inclusive: int, bars: int) -> float:
    end = end_inclusive + 1
    start = max(0, end - int(bars))
    return float(prefix[end] - prefix[start])


def _safe_ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= EPS:
        return np.nan
    return float(a / b)


def regularize_window(
    raw: pd.DataFrame,
    *,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Regularize missing seconds using only the last observed trade price."""

    full_index = pd.date_range(start_time, end_time, freq="1s", inclusive="left")
    if raw.empty or len(full_index) == 0:
        return pd.DataFrame(), {"regularized_rows": 0, "observed_rows": 0, "leading_gap_seconds": np.nan}
    frame = raw.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last").set_index("timestamp").sort_index()
    frame = frame.loc[(frame.index >= start_time) & (frame.index < end_time)]
    if frame.empty:
        return pd.DataFrame(), {"regularized_rows": 0, "observed_rows": 0, "leading_gap_seconds": np.nan}
    observed = frame.index
    leading_gap = max(0.0, float((observed.min() - start_time).total_seconds()))
    out = frame.reindex(full_index)
    out["source_bar_observed_flag"] = out["close"].notna()

    # Forward fill prices only from an already observed second.  A leading gap
    # is backfilled solely to make array math possible and is explicitly
    # audited; windows with a material leading gap are excluded downstream.
    seed_price = float(frame.iloc[0]["open"])
    close = pd.to_numeric(out["close"], errors="coerce").ffill().fillna(seed_price)
    for name in ("open", "high", "low", "close", "vwap"):
        values = pd.to_numeric(out.get(name), errors="coerce")
        if name == "close":
            out[name] = close
        elif name == "vwap":
            out[name] = values.fillna(close)
        else:
            out[name] = values.fillna(close)
    for name in FLOW_COLUMNS:
        out[name] = pd.to_numeric(out.get(name), errors="coerce").fillna(0.0)
    out["max_trade_notional"] = pd.to_numeric(out.get("max_trade_notional"), errors="coerce").fillna(0.0)
    out["max_trade_size"] = pd.to_numeric(out.get("max_trade_size"), errors="coerce").fillna(0.0)
    out["taker_buy_ratio"] = np.divide(
        out["buy_notional"].to_numpy(dtype=float),
        out["notional"].to_numpy(dtype=float),
        out=np.full(len(out), np.nan),
        where=out["notional"].to_numpy(dtype=float) > EPS,
    )
    out.index.name = "timestamp"
    return out, {
        "regularized_rows": int(len(out)),
        "observed_rows": int(out["source_bar_observed_flag"].sum()),
        "leading_gap_seconds": leading_gap,
        "observed_share": float(out["source_bar_observed_flag"].mean()),
    }


def _feature_at(
    *,
    i: int,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    p_buy: np.ndarray,
    p_sell: np.ndarray,
    p_delta: np.ndarray,
    p_large_delta: np.ndarray,
    p_notional: np.ndarray,
    p_trades: np.ndarray,
    running_low: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for w in (3, 5, 10, 15, 30):
        buy = _window_sum(p_buy, i, w)
        sell = _window_sum(p_sell, i, w)
        delta = _window_sum(p_delta, i, w)
        large_delta = _window_sum(p_large_delta, i, w)
        notional = _window_sum(p_notional, i, w)
        trades = _window_sum(p_trades, i, w)
        start = max(0, i - w + 1)
        start_price = float(open_arr[start])
        price_change_bp = (float(close_arr[i]) / start_price - 1.0) * 10_000.0 if start_price > 0 else np.nan
        downside = max(0.0, -price_change_bp) if np.isfinite(price_change_bp) else np.nan
        result[f"buy_notional_{w}s"] = buy
        result[f"sell_notional_{w}s"] = sell
        result[f"delta_ratio_{w}s"] = _safe_ratio(delta, buy + sell)
        result[f"sell_share_{w}s"] = _safe_ratio(sell, buy + sell)
        result[f"large_delta_ratio_{w}s"] = _safe_ratio(large_delta, notional)
        result[f"price_change_{w}s_bp"] = price_change_bp
        result[f"downside_bp_per_sell_million_{w}s"] = _safe_ratio(downside, sell / 1_000_000.0)
        result[f"downside_bp_per_abs_negative_delta_million_{w}s"] = _safe_ratio(
            downside, max(0.0, -delta) / 1_000_000.0
        )
        result[f"notional_{w}s"] = notional
        result[f"trades_{w}s"] = trades
    result["close_off_running_low_bp"] = (
        (float(close_arr[i]) / float(running_low[i]) - 1.0) * 10_000.0 if running_low[i] > 0 else np.nan
    )
    previous_high_start = max(0, i - 5)
    previous_high = float(np.max(high_arr[previous_high_start:i])) if i > previous_high_start else np.nan
    result["micro_high_break_5s"] = float(np.isfinite(previous_high) and close_arr[i] > previous_high)
    result["impact_ratio_5s_vs_prior15s"] = np.nan
    cur_impact = result["downside_bp_per_sell_million_5s"]
    prior_end = i - 5
    if prior_end >= 0:
        prior_start = max(0, prior_end - 14)
        prior_sell = float(p_sell[prior_end + 1] - p_sell[prior_start])
        prior_price = (float(close_arr[prior_end]) / float(open_arr[prior_start]) - 1.0) * 10_000.0
        prior_impact = _safe_ratio(max(0.0, -prior_price), prior_sell / 1_000_000.0)
        result["prior15s_downside_bp_per_sell_million"] = prior_impact
        result["impact_ratio_5s_vs_prior15s"] = _safe_ratio(cur_impact, prior_impact)
        prior_delta = float(p_delta[prior_end + 1] - p_delta[prior_start])
        result["prior15s_delta_ratio"] = _safe_ratio(prior_delta, float(p_notional[prior_end + 1] - p_notional[prior_start]))
    else:
        result["prior15s_downside_bp_per_sell_million"] = np.nan
        result["prior15s_delta_ratio"] = np.nan
    result["delta_improvement_5s_vs_prior15s"] = (
        result["delta_ratio_5s"] - result["prior15s_delta_ratio"]
        if np.isfinite(result["delta_ratio_5s"]) and np.isfinite(result["prior15s_delta_ratio"])
        else np.nan
    )
    return result


def _first_true(mask: np.ndarray, valid_start: int, valid_end: int) -> int | None:
    lo = max(0, int(valid_start))
    hi = min(len(mask), int(valid_end))
    found = np.flatnonzero(mask[lo:hi])
    return int(lo + found[0]) if len(found) else None


def _path_metrics(
    *,
    trigger_name: str,
    signal_idx: int,
    index: pd.DatetimeIndex,
    open_arr: np.ndarray,
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    horizons: tuple[int, ...],
    barriers: tuple[float, ...],
    round_trip_cost: float,
    signal_uses_future: bool,
) -> dict[str, Any] | None:
    entry_idx = signal_idx + 1
    if entry_idx >= len(index):
        return None
    entry_price = float(open_arr[entry_idx])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    row: dict[str, Any] = {
        "trigger_name": trigger_name,
        "signal_bar_start_time": index[signal_idx],
        "signal_time": index[signal_idx] + pd.Timedelta(seconds=1),
        "entry_time": index[entry_idx],
        "entry_price": entry_price,
        "signal_uses_future": bool(signal_uses_future),
        "entry_is_next_bar_open": bool(index[entry_idx] == index[signal_idx] + pd.Timedelta(seconds=1)),
    }
    max_horizon = 0
    for horizon in horizons:
        end = entry_idx + int(horizon)
        complete = end <= len(index)
        row[f"path_complete_{horizon}s"] = complete
        if not complete:
            for metric in ("mfe", "mae", "close_return", "net_close_return"):
                row[f"{metric}_{horizon}s"] = np.nan
            continue
        max_horizon = max(max_horizon, int(horizon))
        h = high_arr[entry_idx:end]
        l = low_arr[entry_idx:end]
        row[f"mfe_{horizon}s"] = float(np.nanmax(h / entry_price - 1.0))
        row[f"mae_{horizon}s"] = float(np.nanmin(l / entry_price - 1.0))
        gross = float(close_arr[end - 1] / entry_price - 1.0)
        row[f"close_return_{horizon}s"] = gross
        row[f"net_close_return_{horizon}s"] = gross - float(round_trip_cost)
    if max_horizon > 0:
        end = entry_idx + max_horizon
        h = high_arr[entry_idx:end]
        l = low_arr[entry_idx:end]
        for barrier in barriers:
            up = np.flatnonzero(h >= entry_price * (1.0 + float(barrier) / 10_000.0))
            down = np.flatnonzero(l <= entry_price * (1.0 - float(barrier) / 10_000.0))
            up_i = int(up[0]) if len(up) else -1
            down_i = int(down[0]) if len(down) else -1
            tag = str(float(barrier)).replace(".", "p")
            target_first = up_i >= 0 and (down_i < 0 or up_i < down_i)
            stop_first = down_i >= 0 and (up_i < 0 or down_i <= up_i)
            row[f"target_first_{tag}bp"] = bool(target_first)
            row[f"stop_first_{tag}bp"] = bool(stop_first)
            row[f"target_first_seconds_{tag}bp"] = up_i + 1 if up_i >= 0 else np.nan
            row[f"stop_first_seconds_{tag}bp"] = down_i + 1 if down_i >= 0 else np.nan
    return row


def analyze_micro_window(
    raw_bars: pd.DataFrame,
    event: pd.Series,
    config: PostSweepMicroConfig,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    cfg = config.validate()
    start = pd.Timestamp(event["start_time"])
    end = pd.Timestamp(event["end_time"])
    micro, quality = regularize_window(raw_bars, start_time=start, end_time=end)
    audit = {
        "window_id": event["window_id"],
        "checkpoint_id": event["checkpoint_id"],
        "cohort": event["cohort"],
        **quality,
    }
    if micro.empty or quality.get("leading_gap_seconds", np.inf) > 2:
        audit["status"] = "insufficient_micro_data"
        return None, [], audit

    index = pd.DatetimeIndex(micro.index)
    anchor = pd.Timestamp(event["checkpoint_time"])
    anchor_pos = int(index.searchsorted(anchor, side="left"))
    minute_end_pos = int(index.searchsorted(anchor + pd.Timedelta(minutes=1), side="left"))
    if anchor_pos >= len(index) or minute_end_pos <= anchor_pos:
        audit["status"] = "anchor_outside_window"
        return None, [], audit

    open_arr = micro["open"].to_numpy(dtype=np.float64)
    high_arr = micro["high"].to_numpy(dtype=np.float64)
    low_arr = micro["low"].to_numpy(dtype=np.float64)
    close_arr = micro["close"].to_numpy(dtype=np.float64)
    buy = micro["buy_notional"].to_numpy(dtype=np.float64)
    sell = micro["sell_notional"].to_numpy(dtype=np.float64)
    delta = micro["delta_notional"].to_numpy(dtype=np.float64)
    large_delta = micro["large_delta_notional"].to_numpy(dtype=np.float64)
    notional = micro["notional"].to_numpy(dtype=np.float64)
    trades = micro["trades_count"].to_numpy(dtype=np.float64)
    p_buy, p_sell, p_delta = _prefix(buy), _prefix(sell), _prefix(delta)
    p_large_delta, p_notional, p_trades = _prefix(large_delta), _prefix(notional), _prefix(trades)
    running_low = np.minimum.accumulate(low_arr)

    minute_low_slice = low_arr[anchor_pos:minute_end_pos]
    if len(minute_low_slice) == 0 or not np.isfinite(minute_low_slice).any():
        audit["status"] = "minute_low_missing"
        return None, [], audit
    low_rel = int(np.nanargmin(minute_low_slice))
    low_idx = anchor_pos + low_rel
    prior_running_low = float(event.get("prior_running_low_before_attempt", np.nan))
    if not np.isfinite(prior_running_low) or prior_running_low <= 0:
        prior_running_low = float(np.nanmin(low_arr[:anchor_pos])) if anchor_pos > 0 else float(low_arr[anchor_pos])
    new_low_seen = np.minimum.accumulate(low_arr) < prior_running_low - max(1e-12, prior_running_low * 1e-10)

    feature_rows: list[dict[str, Any]] = []
    feature_cache: dict[int, dict[str, float]] = {}
    trigger_search_end = min(len(index) - 1, anchor_pos + 180)
    for i in range(max(anchor_pos, 30), trigger_search_end):
        feature_cache[i] = _feature_at(
            i=i, open_arr=open_arr, high_arr=high_arr, low_arr=low_arr, close_arr=close_arr,
            p_buy=p_buy, p_sell=p_sell, p_delta=p_delta, p_large_delta=p_large_delta,
            p_notional=p_notional, p_trades=p_trades, running_low=running_low,
        )

    low_features = _feature_at(
        i=low_idx, open_arr=open_arr, high_arr=high_arr, low_arr=low_arr, close_arr=close_arr,
        p_buy=p_buy, p_sell=p_sell, p_delta=p_delta, p_large_delta=p_large_delta,
        p_notional=p_notional, p_trades=p_trades, running_low=running_low,
    )
    window_feature: dict[str, Any] = {
        "window_id": event["window_id"],
        "checkpoint_id": event["checkpoint_id"],
        "zone_event_id": event["zone_event_id"],
        "pair_id": event["pair_id"],
        "cohort": event["cohort"],
        "period": event["period"],
        "checkpoint_time": anchor,
        "micro_minute_low_time": index[low_idx],
        "micro_minute_low_price": float(low_arr[low_idx]),
        "micro_low_offset_seconds": int(low_idx - anchor_pos),
        "prior_running_low_before_attempt": prior_running_low,
        "micro_new_low_confirmed": bool(np.any(new_low_seen[anchor_pos:minute_end_pos])),
        **{f"low_{key}": value for key, value in low_features.items()},
    }

    # Predeclared natural trigger variants.  Threshold neighborhoods are fixed
    # before seeing outcomes; R06 does not grid-search a winner.
    masks: dict[str, np.ndarray] = {name: np.zeros(len(index), dtype=bool) for name in TRIGGER_NAMES}
    masks["FIRST_NEW_LOW"] = new_low_seen & (np.arange(len(index)) >= anchor_pos)
    for i, feat in feature_cache.items():
        sell_share = feat.get("sell_share_5s", np.nan)
        delta_ratio = feat.get("delta_ratio_5s", np.nan)
        impact_ratio = feat.get("impact_ratio_5s_vs_prior15s", np.nan)
        off_low = feat.get("close_off_running_low_bp", np.nan)
        high_break = bool(feat.get("micro_high_break_5s", 0.0) > 0.5)
        base = bool(
            new_low_seen[i]
            and np.isfinite(sell_share) and sell_share >= 0.55
            and np.isfinite(delta_ratio) and delta_ratio < 0.0
            and np.isfinite(impact_ratio)
        )
        masks["IMPACT_COLLAPSE_67"][i] = base and impact_ratio <= 0.67 and off_low >= 2.0
        masks["IMPACT_COLLAPSE_50"][i] = base and impact_ratio <= 0.50 and off_low >= 3.0
        masks["IMPACT_COLLAPSE_50_HIGH_BREAK"][i] = masks["IMPACT_COLLAPSE_50"][i] and high_break
        delta_improvement = feat.get("delta_improvement_5s_vs_prior15s", np.nan)
        masks["MICRO_RECLAIM_5S"][i] = bool(
            new_low_seen[i] and off_low >= 5.0 and high_break
            and np.isfinite(delta_improvement) and delta_improvement >= 0.05
        )

    # Minute-close baseline is causal after the full attempt minute closes.
    minute_signal_idx = minute_end_pos - 1
    if 0 <= minute_signal_idx < len(index) - 1:
        masks["MINUTE_CLOSE"][minute_signal_idx] = True
    masks["ORACLE_LOW_PLUS_1S"][low_idx] = True

    trigger_rows: list[dict[str, Any]] = []
    for trigger_name in TRIGGER_NAMES:
        if trigger_name == "ORACLE_LOW_PLUS_1S":
            signal_idx = low_idx
            uses_future = True
        elif trigger_name == "MINUTE_CLOSE":
            signal_idx = minute_signal_idx
            uses_future = False
        else:
            signal_idx = _first_true(masks[trigger_name], anchor_pos, trigger_search_end)
            uses_future = False
            if signal_idx is None:
                continue
        path = _path_metrics(
            trigger_name=trigger_name,
            signal_idx=int(signal_idx),
            index=index,
            open_arr=open_arr,
            high_arr=high_arr,
            low_arr=low_arr,
            close_arr=close_arr,
            horizons=cfg.future_horizons_seconds,
            barriers=cfg.first_passage_barriers_bp,
            round_trip_cost=cfg.round_trip_cost,
            signal_uses_future=uses_future,
        )
        if path is None:
            continue
        feat = feature_cache.get(int(signal_idx)) or _feature_at(
            i=int(signal_idx), open_arr=open_arr, high_arr=high_arr, low_arr=low_arr, close_arr=close_arr,
            p_buy=p_buy, p_sell=p_sell, p_delta=p_delta, p_large_delta=p_large_delta,
            p_notional=p_notional, p_trades=p_trades, running_low=running_low,
        )
        path.update(
            {
                "window_id": event["window_id"],
                "checkpoint_id": event["checkpoint_id"],
                "zone_event_id": event["zone_event_id"],
                "pair_id": event["pair_id"],
                "cohort": event["cohort"],
                "period": event["period"],
                "checkpoint_time": anchor,
                "signal_delay_from_minute_start_seconds": float((path["signal_time"] - anchor).total_seconds()),
                **feat,
            }
        )
        trigger_rows.append(path)

    audit["status"] = "complete"
    audit["trigger_rows"] = len(trigger_rows)
    audit["micro_new_low_confirmed"] = window_feature["micro_new_low_confirmed"]
    return window_feature, trigger_rows, audit


__all__ = ["TRIGGER_NAMES", "analyze_micro_window", "regularize_window"]
