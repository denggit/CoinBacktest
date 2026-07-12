#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal trade-bar order-flow features for panic/liquidity research.

The module deliberately requires trade-derived fields instead of silently
falling back to OHLCV. Rolling baselines use only earlier closed bars. Episode
aggregates for a green recovery signal are bounded by ``episode_start_time`` and
``signal_time``; therefore they are available when the signal bar closes.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter


CORE_ORDERFLOW_COLUMNS: tuple[str, ...] = (
    "notional",
    "buy_notional",
    "sell_notional",
    "delta_notional",
    "trades_count",
    "buy_trades_count",
    "sell_trades_count",
    "taker_buy_ratio",
)

LARGE_ORDERFLOW_COLUMNS: tuple[str, ...] = (
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
    "large_trades_count",
    "max_trade_notional",
)

OPTIONAL_ORDERFLOW_COLUMNS: tuple[str, ...] = (
    "buy_volume",
    "sell_volume",
    "delta_volume",
    "avg_trade_size",
    "vwap",
    "large_buy_trades_count",
    "large_sell_trades_count",
    "max_trade_size",
)


def _numeric(series: pd.Series | Any, index: pd.Index, default: float = np.nan) -> pd.Series:
    if isinstance(series, pd.Series):
        out = pd.to_numeric(series, errors="coerce").reindex(index)
    else:
        out = pd.Series(default, index=index, dtype=float)
    return out.astype(float)


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _prior_median(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    return series.shift(1).rolling(window, min_periods=min_periods).median()


def _rolling_flow_ratio(delta: pd.Series, notional: pd.Series, window: int) -> pd.Series:
    return _safe_divide(
        delta.rolling(window, min_periods=1).sum(),
        notional.rolling(window, min_periods=1).sum(),
    ).clip(-1.0, 1.0)


def trade_bar_field_coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Report whether rich trade-bar fields are actually populated."""
    rows: list[dict[str, Any]] = []
    wanted = CORE_ORDERFLOW_COLUMNS + LARGE_ORDERFLOW_COLUMNS + OPTIONAL_ORDERFLOW_COLUMNS
    for field in wanted:
        present = field in df.columns
        s = pd.to_numeric(df[field], errors="coerce") if present else pd.Series(dtype=float)
        non_null_pct = float(s.notna().mean()) if present and len(s) else 0.0
        non_zero_pct = float((s.fillna(0.0).abs() > 1e-12).mean()) if present and len(s) else 0.0
        unique_values = int(s.nunique(dropna=True)) if present else 0
        rows.append(
            {
                "field": field,
                "group": (
                    "core" if field in CORE_ORDERFLOW_COLUMNS
                    else "large" if field in LARGE_ORDERFLOW_COLUMNS
                    else "optional"
                ),
                "present": bool(present),
                "non_null_pct": non_null_pct,
                "non_zero_pct": non_zero_pct,
                "unique_values": unique_values,
                "usable": bool(present and non_null_pct >= 0.90 and unique_values > 1),
            }
        )
    return pd.DataFrame(rows)


def validate_trade_bar_orderflow(
    df: pd.DataFrame,
    *,
    require_large_fields: bool = True,
    min_non_null_pct: float = 0.90,
) -> pd.DataFrame:
    """Fail loudly when 02 research is accidentally run on plain OHLCV."""
    coverage = trade_bar_field_coverage(df)
    required = set(CORE_ORDERFLOW_COLUMNS)
    if require_large_fields:
        required.update(LARGE_ORDERFLOW_COLUMNS)
    bad = coverage[
        coverage["field"].isin(required)
        & (
            (~coverage["present"])
            | (coverage["non_null_pct"] < float(min_non_null_pct))
            | (coverage["unique_values"] <= 1)
        )
    ]
    if not bad.empty:
        detail = ", ".join(
            f"{row.field}(present={row.present}, non_null={row.non_null_pct:.1%}, unique={row.unique_values})"
            for row in bad.itertuples(index=False)
        )
        raise RuntimeError(
            "02 order-flow research requires populated OKX trade-bar fields; "
            f"unusable fields: {detail}. Rebuild/backfill okx_trade_bars.db instead of falling back to OHLCV."
        )
    return coverage


def build_trade_bar_orderflow_features(
    bars: pd.DataFrame,
    *,
    baseline_window: int = 240,
) -> pd.DataFrame:
    """Build causal closed-bar order-flow, activity, impact and absorption features."""
    if baseline_window < 60:
        raise ValueError("baseline_window must be >= 60")
    validate_trade_bar_orderflow(bars)

    out = bars.copy().sort_index()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]
    idx = out.index

    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            raise ValueError(f"missing required bar field: {col}")
        out[col] = _numeric(out[col], idx)

    for col in CORE_ORDERFLOW_COLUMNS + LARGE_ORDERFLOW_COLUMNS + OPTIONAL_ORDERFLOW_COLUMNS:
        if col in out.columns:
            out[col] = _numeric(out[col], idx)

    # Reconcile mathematically linked fields without replacing valid source data.
    buy_notional = _numeric(out.get("buy_notional"), idx, 0.0).clip(lower=0.0)
    sell_notional = _numeric(out.get("sell_notional"), idx, 0.0).clip(lower=0.0)
    notional_sum = buy_notional + sell_notional
    notional = _numeric(out.get("notional"), idx)
    notional = notional.where(notional > 0, notional_sum)
    delta_notional = _numeric(out.get("delta_notional"), idx)
    delta_notional = delta_notional.where(delta_notional.notna(), buy_notional - sell_notional)

    large_buy = _numeric(out.get("large_buy_notional"), idx, 0.0).clip(lower=0.0)
    large_sell = _numeric(out.get("large_sell_notional"), idx, 0.0).clip(lower=0.0)
    large_notional = large_buy + large_sell
    large_delta = _numeric(out.get("large_delta_notional"), idx)
    large_delta = large_delta.where(large_delta.notna(), large_buy - large_sell)

    trades = _numeric(out.get("trades_count"), idx, 0.0).clip(lower=0.0)
    buy_trades = _numeric(out.get("buy_trades_count"), idx, 0.0).clip(lower=0.0)
    sell_trades = _numeric(out.get("sell_trades_count"), idx, 0.0).clip(lower=0.0)
    large_trades = _numeric(out.get("large_trades_count"), idx, 0.0).clip(lower=0.0)
    max_trade_notional = _numeric(out.get("max_trade_notional"), idx, 0.0).clip(lower=0.0)

    out["notional"] = notional
    out["buy_notional"] = buy_notional
    out["sell_notional"] = sell_notional
    out["delta_notional"] = delta_notional
    out["large_notional"] = large_notional
    out["large_delta_notional"] = large_delta

    out["delta_ratio"] = _safe_divide(delta_notional, notional).clip(-1.0, 1.0)
    out["large_delta_ratio"] = _safe_divide(large_delta, large_notional).clip(-1.0, 1.0)
    out["taker_buy_ratio_raw"] = _safe_divide(buy_notional, notional).clip(0.0, 1.0)
    source_taker = _numeric(out.get("taker_buy_ratio"), idx)
    out["taker_buy_ratio"] = source_taker.where(source_taker.between(0.0, 1.0), out["taker_buy_ratio_raw"])
    out["buy_trade_ratio"] = _safe_divide(buy_trades, trades).clip(0.0, 1.0)
    out["large_trade_share"] = _safe_divide(large_notional, notional).clip(0.0, 1.0)
    out["large_sell_share_of_sell"] = _safe_divide(large_sell, sell_notional).clip(0.0, 1.0)
    out["large_buy_share_of_buy"] = _safe_divide(large_buy, buy_notional).clip(0.0, 1.0)
    out["large_trade_count_share"] = _safe_divide(large_trades, trades).clip(0.0, 1.0)
    out["max_trade_share"] = _safe_divide(max_trade_notional, notional).clip(0.0, 1.0)

    min_periods = max(60, baseline_window // 4)
    activity_sources = {
        "notional": notional,
        "buy_notional": buy_notional,
        "sell_notional": sell_notional,
        "trades": trades,
        "large_notional": large_notional,
        "max_trade_notional": max_trade_notional,
    }
    for name, series in activity_sources.items():
        baseline = _prior_median(series, baseline_window, min_periods)
        out[f"{name}_ratio_base"] = _safe_divide(series, baseline).clip(lower=0.0, upper=50.0)

    avg_trade_notional = _safe_divide(notional, trades)
    avg_trade_base = _prior_median(avg_trade_notional, baseline_window, min_periods)
    out["avg_trade_notional"] = avg_trade_notional
    out["avg_trade_notional_ratio_base"] = _safe_divide(avg_trade_notional, avg_trade_base).clip(0.0, 50.0)

    out["ret_1"] = out["close"].pct_change()
    abs_ret_base = _prior_median(out["ret_1"].abs(), baseline_window, min_periods).clip(lower=1e-8)
    out["down_move_norm"] = (-out["ret_1"]).clip(lower=0.0) / abs_ret_base
    out["up_move_norm"] = out["ret_1"].clip(lower=0.0) / abs_ret_base
    out["sell_impact_per_intensity"] = _safe_divide(
        out["down_move_norm"],
        out["sell_notional_ratio_base"].clip(lower=0.10),
    ).clip(0.0, 50.0)

    bar_range = (out["high"] - out["low"]).clip(lower=out["close"].abs() * 1e-9)
    out["close_pos"] = ((out["close"] - out["low"]) / bar_range).clip(0.0, 1.0)
    out["lower_wick_frac"] = (
        (out[["open", "close"]].min(axis=1) - out["low"]).clip(lower=0.0) / bar_range
    ).clip(0.0, 1.0)

    # High selling activity with unexpectedly small price damage is an absorption proxy.
    out["absorption_score"] = (
        out["sell_notional_ratio_base"].fillna(1.0)
        + 0.40 * out["trades_ratio_base"].fillna(1.0)
        + 0.50 * out["large_sell_share_of_sell"].fillna(0.0)
        - out["down_move_norm"].fillna(0.0)
        + 0.50 * out["close_pos"].fillna(0.5)
    )

    for window in (2, 3, 5):
        out[f"delta_ratio_{window}"] = _rolling_flow_ratio(delta_notional, notional, window)
        out[f"large_delta_ratio_{window}"] = _rolling_flow_ratio(large_delta, large_notional, window)
        out[f"taker_buy_ratio_{window}"] = _safe_divide(
            buy_notional.rolling(window, min_periods=1).sum(),
            notional.rolling(window, min_periods=1).sum(),
        ).clip(0.0, 1.0)
        out[f"price_return_{window}"] = out["close"] / out["close"].shift(window) - 1.0

    out["delta_reversal_short"] = out["delta_ratio_2"] - out["delta_ratio_5"].shift(2)
    out["large_delta_reversal_short"] = out["large_delta_ratio_2"] - out["large_delta_ratio_5"].shift(2)
    out["flow_price_divergence_3"] = out["delta_ratio_3"] - (
        out["price_return_3"] / abs_ret_base.replace(0.0, np.nan)
    ).clip(-10.0, 10.0) / 10.0

    return out.replace([np.inf, -np.inf], np.nan)


def _value(frame: pd.DataFrame, ts: pd.Timestamp, col: str, default: float = np.nan) -> float:
    if col not in frame.columns or ts not in frame.index:
        return float(default)
    try:
        value = float(frame.at[ts, col])
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _segment_stat(segment: pd.DataFrame, col: str, op: str, default: float = np.nan) -> float:
    if segment.empty or col not in segment.columns:
        return float(default)
    s = pd.to_numeric(segment[col], errors="coerce")
    if not s.notna().any():
        return float(default)
    value = getattr(s, op)()
    return float(value) if np.isfinite(value) else float(default)


def summarize_episode_orderflow(
    stage_events: pd.DataFrame,
    orderflow: pd.DataFrame,
    *,
    progress_every: int = 250,
    progress_enabled: bool = True,
) -> pd.DataFrame:
    """Create one rich order-flow row per episode.

    For signal episodes every aggregate ends at ``signal_time``. The actual low
    timestamp is therefore already known when the green signal bar closes.
    """
    if stage_events.empty:
        return pd.DataFrame()

    episodes = (
        stage_events.sort_values(["episode_id", "event_time"])
        .drop_duplicates("episode_id")
        .reset_index(drop=True)
    )
    index_positions = pd.Series(np.arange(len(orderflow), dtype=int), index=orderflow.index)
    rows: list[dict[str, Any]] = []
    reporter = ProgressReporter(
        "[orderflow] episode features",
        len(episodes),
        every=max(1, int(progress_every)),
        enabled=progress_enabled,
    )

    for done, ep in enumerate(episodes.itertuples(index=False), start=1):
        start = pd.Timestamp(ep.episode_start_time)
        signal_time = pd.Timestamp(ep.signal_time) if pd.notna(ep.signal_time) else pd.Timestamp(ep.episode_end_time)
        if start not in orderflow.index or signal_time not in orderflow.index or signal_time < start:
            reporter.update(done)
            continue
        window = orderflow.loc[start:signal_time]
        if window.empty:
            reporter.update(done)
            continue

        low_time = pd.Timestamp(window["low"].idxmin())
        panic = window.loc[start:low_time]
        recovery = window.loc[low_time:signal_time]
        before_low = panic.iloc[:-1]

        start_pos = int(index_positions.at[start])
        low_pos = int(index_positions.at[low_time])
        signal_pos = int(index_positions.at[signal_time])

        panic_min_delta = _segment_stat(panic, "delta_ratio", "min")
        panic_min_large_delta = _segment_stat(panic, "large_delta_ratio", "min")
        panic_min_taker = _segment_stat(panic, "taker_buy_ratio", "min")
        panic_max_sell_ratio = _segment_stat(panic, "sell_notional_ratio_base", "max")
        panic_max_notional_ratio = _segment_stat(panic, "notional_ratio_base", "max")
        panic_max_trades_ratio = _segment_stat(panic, "trades_ratio_base", "max")
        panic_max_large_share = _segment_stat(panic, "large_trade_share", "max")
        panic_max_large_sell_share = _segment_stat(panic, "large_sell_share_of_sell", "max")
        panic_max_trade_share = _segment_stat(panic, "max_trade_share", "max")
        panic_max_absorption = _segment_stat(panic, "absorption_score", "max")

        low_delta = _value(orderflow, low_time, "delta_ratio")
        low_large_delta = _value(orderflow, low_time, "large_delta_ratio")
        low_taker = _value(orderflow, low_time, "taker_buy_ratio")
        low_absorption = _value(orderflow, low_time, "absorption_score")
        low_close_pos = _value(orderflow, low_time, "close_pos")
        low_lower_wick = _value(orderflow, low_time, "lower_wick_frac")
        low_sell_ratio = _value(orderflow, low_time, "sell_notional_ratio_base")
        low_trades_ratio = _value(orderflow, low_time, "trades_ratio_base")
        low_large_sell_share = _value(orderflow, low_time, "large_sell_share_of_sell")
        prior_min_delta = _segment_stat(before_low, "delta_ratio", "min")
        prior_min_large_delta = _segment_stat(before_low, "large_delta_ratio", "min")

        signal_delta = _value(orderflow, signal_time, "delta_ratio_2")
        signal_large_delta = _value(orderflow, signal_time, "large_delta_ratio_2")
        signal_taker = _value(orderflow, signal_time, "taker_buy_ratio_2")
        signal_sell_ratio = _value(orderflow, signal_time, "sell_notional_ratio_base")
        signal_trades_ratio = _value(orderflow, signal_time, "trades_ratio_base")
        signal_notional_ratio = _value(orderflow, signal_time, "notional_ratio_base")
        signal_absorption = _value(orderflow, signal_time, "absorption_score")
        signal_flow_reversal = _value(orderflow, signal_time, "delta_reversal_short")
        signal_large_reversal = _value(orderflow, signal_time, "large_delta_reversal_short")

        sell_intensity_decay = (
            signal_sell_ratio / panic_max_sell_ratio
            if np.isfinite(signal_sell_ratio) and np.isfinite(panic_max_sell_ratio) and panic_max_sell_ratio > 0
            else np.nan
        )
        delta_recovery = signal_delta - panic_min_delta if np.isfinite(signal_delta) and np.isfinite(panic_min_delta) else np.nan
        large_delta_recovery = (
            signal_large_delta - panic_min_large_delta
            if np.isfinite(signal_large_delta) and np.isfinite(panic_min_large_delta)
            else np.nan
        )
        taker_recovery = signal_taker - panic_min_taker if np.isfinite(signal_taker) and np.isfinite(panic_min_taker) else np.nan
        low_delta_divergence = (
            low_delta - prior_min_delta
            if np.isfinite(low_delta) and np.isfinite(prior_min_delta)
            else np.nan
        )
        low_large_delta_divergence = (
            low_large_delta - prior_min_large_delta
            if np.isfinite(low_large_delta) and np.isfinite(prior_min_large_delta)
            else np.nan
        )
        flow_recovery_score = (
            (delta_recovery if np.isfinite(delta_recovery) else 0.0)
            + 0.75 * (large_delta_recovery if np.isfinite(large_delta_recovery) else 0.0)
            + 0.50 * (taker_recovery if np.isfinite(taker_recovery) else 0.0)
            + 0.25 * (1.0 - sell_intensity_decay if np.isfinite(sell_intensity_decay) else 0.0)
        )
        low_absorption_rejection_score = (
            (low_absorption if np.isfinite(low_absorption) else 0.0)
            + 0.75 * (low_close_pos if np.isfinite(low_close_pos) else 0.0)
            + 0.50 * (low_lower_wick if np.isfinite(low_lower_wick) else 0.0)
        )

        rows.append(
            {
                "episode_id": int(ep.episode_id),
                "orderflow_window_start": start,
                "orderflow_window_end": signal_time,
                "actual_low_time": low_time,
                "bars_start_to_low": low_pos - start_pos,
                "bars_low_to_signal": signal_pos - low_pos,
                "panic_min_delta_ratio": panic_min_delta,
                "panic_min_large_delta_ratio": panic_min_large_delta,
                "panic_min_taker_buy_ratio": panic_min_taker,
                "panic_max_sell_notional_ratio": panic_max_sell_ratio,
                "panic_max_notional_ratio": panic_max_notional_ratio,
                "panic_max_trades_ratio": panic_max_trades_ratio,
                "panic_max_large_trade_share": panic_max_large_share,
                "panic_max_large_sell_share": panic_max_large_sell_share,
                "panic_max_trade_share": panic_max_trade_share,
                "panic_max_absorption_score": panic_max_absorption,
                "low_delta_ratio": low_delta,
                "low_large_delta_ratio": low_large_delta,
                "low_taker_buy_ratio": low_taker,
                "low_absorption_score": low_absorption,
                "low_close_pos": low_close_pos,
                "low_lower_wick_frac": low_lower_wick,
                "low_sell_notional_ratio": low_sell_ratio,
                "low_trades_ratio": low_trades_ratio,
                "low_large_sell_share": low_large_sell_share,
                "low_delta_divergence": low_delta_divergence,
                "low_large_delta_divergence": low_large_delta_divergence,
                "signal_delta_ratio_2": signal_delta,
                "signal_large_delta_ratio_2": signal_large_delta,
                "signal_taker_buy_ratio_2": signal_taker,
                "signal_sell_notional_ratio": signal_sell_ratio,
                "signal_trades_ratio": signal_trades_ratio,
                "signal_notional_ratio": signal_notional_ratio,
                "signal_absorption_score": signal_absorption,
                "signal_delta_reversal_short": signal_flow_reversal,
                "signal_large_delta_reversal_short": signal_large_reversal,
                "delta_recovery_from_panic": delta_recovery,
                "large_delta_recovery_from_panic": large_delta_recovery,
                "taker_recovery_from_panic": taker_recovery,
                "sell_intensity_decay": sell_intensity_decay,
                "flow_recovery_score": flow_recovery_score,
                "low_absorption_rejection_score": low_absorption_rejection_score,
                "recovery_min_delta_ratio": _segment_stat(recovery, "delta_ratio", "min"),
                "recovery_max_delta_ratio": _segment_stat(recovery, "delta_ratio", "max"),
                "recovery_max_taker_buy_ratio": _segment_stat(recovery, "taker_buy_ratio", "max"),
            }
        )
        if done < len(episodes):
            reporter.update(done)

    reporter.close()
    return pd.DataFrame(rows)


def attach_orderflow_to_stage_events(
    stage_events: pd.DataFrame,
    orderflow: pd.DataFrame,
    episode_orderflow: pd.DataFrame,
) -> pd.DataFrame:
    """Attach episode aggregates and same-node closed-bar fields."""
    if stage_events.empty:
        return stage_events.copy()
    out = stage_events.merge(episode_orderflow, on="episode_id", how="left", validate="many_to_one")
    node_fields = (
        "delta_ratio",
        "large_delta_ratio",
        "taker_buy_ratio",
        "buy_trade_ratio",
        "notional_ratio_base",
        "buy_notional_ratio_base",
        "sell_notional_ratio_base",
        "trades_ratio_base",
        "large_notional_ratio_base",
        "large_trade_share",
        "large_sell_share_of_sell",
        "max_trade_share",
        "absorption_score",
        "sell_impact_per_intensity",
        "delta_ratio_2",
        "delta_ratio_3",
        "large_delta_ratio_2",
        "taker_buy_ratio_2",
        "delta_reversal_short",
        "large_delta_reversal_short",
    )
    for field in node_fields:
        values = orderflow[field].reindex(pd.to_datetime(out["event_time"])) if field in orderflow.columns else np.nan
        if isinstance(values, pd.Series):
            out[f"node_{field}"] = values.to_numpy()
        else:
            out[f"node_{field}"] = values
    return out
