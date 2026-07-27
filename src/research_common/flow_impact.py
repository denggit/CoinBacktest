#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal active-flow pressure and price-impact research primitives.

This module is deliberately research-facing and symmetric for buy/sell pressure.
It consumes OKX trade-derived bars with explicit taker-side fields; it never
infers aggressive direction from candle colour.  All historical normalizers end
before the complete current pressure window, so a feature on a left-labelled bar
is usable only after that bar closes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

REQUIRED_PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
REQUIRED_FLOW_COLUMNS: tuple[str, ...] = (
    "notional",
    "buy_notional",
    "sell_notional",
    "delta_notional",
    "trades_count",
    "buy_trades_count",
    "sell_trades_count",
)
OPTIONAL_LARGE_COLUMNS: tuple[str, ...] = (
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
    "large_trades_count",
    "max_trade_notional",
)
_EPS = 1e-12


@dataclass(frozen=True)
class FlowImpactConfig:
    """Configuration for causal pressure/impact feature construction."""

    pressure_windows: tuple[int, ...] = (1, 3, 5)
    baseline_bars: int = 1440
    baseline_min_periods: int = 720
    min_pressure_z: float = 1.0
    event_cooldown_multiplier: float = 1.0

    def validate(self) -> None:
        if not self.pressure_windows or any(int(v) <= 0 for v in self.pressure_windows):
            raise ValueError("pressure_windows must contain positive bar counts")
        if int(self.baseline_bars) < 60:
            raise ValueError("baseline_bars must be >= 60")
        if int(self.baseline_min_periods) < 30:
            raise ValueError("baseline_min_periods must be >= 30")
        if int(self.baseline_min_periods) > int(self.baseline_bars):
            raise ValueError("baseline_min_periods must be <= baseline_bars")
        if float(self.min_pressure_z) <= 0:
            raise ValueError("min_pressure_z must be positive")
        if float(self.event_cooldown_multiplier) < 0:
            raise ValueError("event_cooldown_multiplier must be >= 0")


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _prior_mean_std(values: pd.Series, *, exclude_bars: int, window: int, min_periods: int) -> tuple[pd.Series, pd.Series]:
    history = pd.to_numeric(values, errors="coerce").shift(int(exclude_bars))
    mean = history.rolling(int(window), min_periods=int(min_periods)).mean()
    std = history.rolling(int(window), min_periods=int(min_periods)).std(ddof=0)
    return mean, std


def _prior_median(values: pd.Series, *, exclude_bars: int, window: int, min_periods: int) -> pd.Series:
    return (
        pd.to_numeric(values, errors="coerce")
        .shift(int(exclude_bars))
        .rolling(int(window), min_periods=int(min_periods))
        .median()
    )


def flow_field_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    """Return field-level diagnostics for rich OKX trade-bar inputs."""
    rows: list[dict[str, Any]] = []
    for column in REQUIRED_PRICE_COLUMNS + REQUIRED_FLOW_COLUMNS + OPTIONAL_LARGE_COLUMNS:
        present = column in frame.columns
        values = _numeric(frame, column) if present else pd.Series(dtype=float)
        rows.append(
            {
                "field": column,
                "group": (
                    "price"
                    if column in REQUIRED_PRICE_COLUMNS
                    else "core_flow"
                    if column in REQUIRED_FLOW_COLUMNS
                    else "large_flow"
                ),
                "present": bool(present),
                "non_null_ratio": float(values.notna().mean()) if present and len(values) else 0.0,
                "non_zero_ratio": float((values.fillna(0.0).abs() > _EPS).mean()) if present and len(values) else 0.0,
                "unique_values": int(values.nunique(dropna=True)) if present else 0,
            }
        )
    return pd.DataFrame(rows)


def validate_flow_input(frame: pd.DataFrame, *, min_non_null_ratio: float = 0.90) -> pd.DataFrame:
    """Fail loudly instead of silently falling back to OHLCV-derived direction."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("flow input must use a pandas DatetimeIndex")
    coverage = flow_field_coverage(frame)
    required = set(REQUIRED_PRICE_COLUMNS + REQUIRED_FLOW_COLUMNS)
    bad = coverage[
        coverage["field"].isin(required)
        & (
            (~coverage["present"])
            | (coverage["non_null_ratio"] < float(min_non_null_ratio))
            | (coverage["unique_values"] <= 1)
        )
    ]
    if not bad.empty:
        details = ", ".join(
            f"{row.field}(present={row.present}, non_null={row.non_null_ratio:.1%}, unique={row.unique_values})"
            for row in bad.itertuples(index=False)
        )
        raise RuntimeError(
            "Flow-impact research requires populated OKX trade-bar order-flow fields; "
            f"unusable fields: {details}. Use OKXTradeBarLoader cache and do not fall back to ordinary OHLCV."
        )
    return coverage


def regularize_trade_bar_axis(frame: pd.DataFrame, *, bar_delta: pd.Timedelta) -> pd.DataFrame:
    """Regularize missing calendar bars while preserving an observation mask.

    Missing buckets are represented as previous-close flat bars with zero flow.
    They are never intended to become eligible pressure, entry or outcome rows;
    callers must use ``source_bar_observed_flag`` and the window-valid columns
    produced by :func:`build_flow_impact_features`.
    """
    if bar_delta <= pd.Timedelta(0):
        raise ValueError("bar_delta must be positive")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame must use a pandas DatetimeIndex")
    if frame.empty:
        return frame.copy()

    source = frame.copy().sort_index()
    source.index = pd.to_datetime(source.index)
    source = source[~source.index.duplicated(keep="last")]
    full_index = pd.date_range(source.index.min(), source.index.max(), freq=bar_delta)
    out = source.reindex(full_index)
    observed = out["close"].notna() if "close" in out.columns else pd.Series(False, index=full_index)
    out["source_bar_observed_flag"] = observed.astype(bool)

    close = pd.to_numeric(out.get("close"), errors="coerce").ffill()
    for column in ("open", "high", "low", "close"):
        values = pd.to_numeric(out.get(column), errors="coerce")
        out[column] = values.where(observed, close)
    zero_columns = set(REQUIRED_FLOW_COLUMNS + OPTIONAL_LARGE_COLUMNS + ("volume",))
    zero_columns.update(
        {
            "buy_volume",
            "sell_volume",
            "delta_volume",
            "cvd_volume",
            "cvd_notional",
            "taker_buy_ratio",
            "avg_trade_size",
            "vwap",
            "large_buy_trades_count",
            "large_sell_trades_count",
            "max_trade_size",
        }
    )
    for column in zero_columns:
        if column not in out.columns:
            out[column] = 0.0
        else:
            values = pd.to_numeric(out[column], errors="coerce")
            out[column] = values.where(observed, 0.0).fillna(0.0)
    out.index.name = frame.index.name or "timestamp"
    out.attrs["source_rows"] = int(len(source))
    out.attrs["regularized_rows"] = int(len(out))
    out.attrs["synthetic_gap_bars"] = int((~observed).sum())
    return out


def build_flow_impact_features(frame: pd.DataFrame, config: FlowImpactConfig | None = None) -> pd.DataFrame:
    """Build symmetric pressure, persistence and price-response features."""
    cfg = config or FlowImpactConfig()
    cfg.validate()
    validate_flow_input(frame)

    out = frame.copy().sort_index()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]
    observed = (
        out["source_bar_observed_flag"].astype(bool)
        if "source_bar_observed_flag" in out.columns
        else pd.Series(True, index=out.index, dtype=bool)
    )

    open_ = _numeric(out, "open")
    high = _numeric(out, "high")
    low = _numeric(out, "low")
    close = _numeric(out, "close")
    notional = _numeric(out, "notional", 0.0).clip(lower=0.0)
    buy = _numeric(out, "buy_notional", 0.0).clip(lower=0.0)
    sell = _numeric(out, "sell_notional", 0.0).clip(lower=0.0)
    delta = _numeric(out, "delta_notional")
    delta = delta.where(delta.notna(), buy - sell)
    notional = notional.where(notional > 0.0, buy + sell)
    trades = _numeric(out, "trades_count", 0.0).clip(lower=0.0)
    buy_trades = _numeric(out, "buy_trades_count", 0.0).clip(lower=0.0)
    sell_trades = _numeric(out, "sell_trades_count", 0.0).clip(lower=0.0)

    large_buy = _numeric(out, "large_buy_notional", 0.0).clip(lower=0.0)
    large_sell = _numeric(out, "large_sell_notional", 0.0).clip(lower=0.0)
    large_delta = _numeric(out, "large_delta_notional")
    large_delta = large_delta.where(large_delta.notna(), large_buy - large_sell)
    large_total = large_buy + large_sell
    large_trades = _numeric(out, "large_trades_count", 0.0).clip(lower=0.0)
    max_trade_notional = _numeric(out, "max_trade_notional", 0.0).clip(lower=0.0)

    out["ret_1"] = close.pct_change()
    out["bar_range_pct"] = _safe_divide(high - low, close.shift(1).abs()).clip(lower=0.0)
    out["delta_ratio_1"] = _safe_divide(delta, notional).clip(-1.0, 1.0)
    out["trade_imbalance_1"] = _safe_divide(buy_trades - sell_trades, trades).clip(-1.0, 1.0)
    out["large_delta_ratio_1"] = _safe_divide(large_delta, large_total).clip(-1.0, 1.0)

    for window in sorted(set(int(v) for v in cfg.pressure_windows)):
        suffix = f"w{window}"
        observed_count = observed.astype(np.int16).rolling(window, min_periods=window).sum()
        window_valid = observed_count.eq(window)

        notional_sum = notional.rolling(window, min_periods=window).sum()
        delta_sum = delta.rolling(window, min_periods=window).sum()
        trades_sum = trades.rolling(window, min_periods=window).sum()
        trade_delta_sum = (buy_trades - sell_trades).rolling(window, min_periods=window).sum()
        large_total_sum = large_total.rolling(window, min_periods=window).sum()
        large_delta_sum = large_delta.rolling(window, min_periods=window).sum()
        large_trades_sum = large_trades.rolling(window, min_periods=window).sum()
        max_trade_window = max_trade_notional.rolling(window, min_periods=window).max()
        abs_delta_sum = delta.abs().rolling(window, min_periods=window).sum()

        direction = np.sign(delta_sum).astype(float)
        flow_ratio = _safe_divide(delta_sum, notional_sum).clip(-1.0, 1.0)
        trade_imbalance = _safe_divide(trade_delta_sum, trades_sum).clip(-1.0, 1.0)
        large_flow_ratio = _safe_divide(large_delta_sum, large_total_sum).clip(-1.0, 1.0)
        large_notional_share = _safe_divide(large_total_sum, notional_sum).clip(0.0, 1.0)
        large_trade_share = _safe_divide(large_trades_sum, trades_sum).clip(0.0, 1.0)
        flow_concentration = _safe_divide(delta_sum.abs(), abs_delta_sum).clip(0.0, 1.0)
        signed_flow_persistence = np.sign(delta).rolling(window, min_periods=window).mean()
        persistence = (signed_flow_persistence * direction).clip(-1.0, 1.0)

        pressure_log = np.log1p(delta_sum.abs())
        pressure_mean, pressure_std = _prior_mean_std(
            pressure_log,
            exclude_bars=window,
            window=cfg.baseline_bars,
            min_periods=cfg.baseline_min_periods,
        )
        pressure_z = ((pressure_log - pressure_mean) / pressure_std.replace(0.0, np.nan)).clip(-8.0, 12.0)

        activity_log = np.log1p(notional_sum)
        activity_mean, activity_std = _prior_mean_std(
            activity_log,
            exclude_bars=window,
            window=cfg.baseline_bars,
            min_periods=cfg.baseline_min_periods,
        )
        activity_z = ((activity_log - activity_mean) / activity_std.replace(0.0, np.nan)).clip(-8.0, 12.0)

        window_return = close / close.shift(window) - 1.0
        log_ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan)
        prior_vol = (
            log_ret.shift(window)
            .rolling(cfg.baseline_bars, min_periods=cfg.baseline_min_periods)
            .std(ddof=0)
            * np.sqrt(float(window))
        )
        response_norm = _safe_divide(direction * window_return, prior_vol).clip(-12.0, 12.0)
        pressure_effectiveness = _safe_divide(
            response_norm,
            pressure_z.clip(lower=0.25),
        ).clip(-12.0, 12.0)

        delta_million = delta_sum.abs() / 1_000_000.0
        impact_bps_per_million = _safe_divide(direction * window_return * 10_000.0, delta_million).clip(-500.0, 500.0)
        prior_notional_median = _prior_median(
            notional_sum,
            exclude_bars=window,
            window=cfg.baseline_bars,
            min_periods=cfg.baseline_min_periods,
        )
        notional_ratio = _safe_divide(notional_sum, prior_notional_median).clip(0.0, 100.0)
        avg_trade_notional = _safe_divide(notional_sum, trades_sum)
        prior_avg_trade_median = _prior_median(
            avg_trade_notional,
            exclude_bars=window,
            window=cfg.baseline_bars,
            min_periods=cfg.baseline_min_periods,
        )
        avg_trade_notional_ratio = _safe_divide(avg_trade_notional, prior_avg_trade_median).clip(0.0, 100.0)
        prior_max_trade_median = _prior_median(
            max_trade_window,
            exclude_bars=window,
            window=cfg.baseline_bars,
            min_periods=cfg.baseline_min_periods,
        )
        max_trade_notional_ratio = _safe_divide(max_trade_window, prior_max_trade_median).clip(0.0, 100.0)

        bar_range = (high - low).clip(lower=close.abs() * 1e-9)
        close_pos = ((close - low) / bar_range).clip(0.0, 1.0)
        direction_close_location = pd.Series(
            np.where(direction >= 0.0, close_pos, 1.0 - close_pos),
            index=out.index,
            dtype=float,
        )

        ready = (
            window_valid
            & notional_sum.gt(0.0)
            & delta_sum.ne(0.0)
            & pressure_z.notna()
            & prior_vol.gt(0.0)
        )
        fields = {
            f"source_window_valid_{suffix}": window_valid,
            f"feature_ready_{suffix}": ready,
            f"pressure_direction_{suffix}": direction,
            f"pressure_notional_{suffix}": delta_sum.abs(),
            f"pressure_z_{suffix}": pressure_z,
            f"flow_ratio_{suffix}": flow_ratio,
            f"trade_imbalance_{suffix}": trade_imbalance,
            f"large_flow_ratio_{suffix}": large_flow_ratio,
            f"large_notional_share_{suffix}": large_notional_share,
            f"large_trade_share_{suffix}": large_trade_share,
            f"flow_concentration_{suffix}": flow_concentration,
            f"flow_persistence_{suffix}": persistence,
            f"notional_ratio_{suffix}": notional_ratio,
            f"avg_trade_notional_ratio_{suffix}": avg_trade_notional_ratio,
            f"max_trade_notional_ratio_{suffix}": max_trade_notional_ratio,
            f"activity_z_{suffix}": activity_z,
            f"price_response_{suffix}": direction * window_return,
            f"price_response_norm_{suffix}": response_norm,
            f"pressure_effectiveness_{suffix}": pressure_effectiveness,
            f"impact_bps_per_million_{suffix}": impact_bps_per_million,
            f"direction_close_location_{suffix}": direction_close_location,
        }
        for column, values in fields.items():
            out[column] = values.where(ready) if column not in {f"source_window_valid_{suffix}", f"feature_ready_{suffix}"} else values.astype(bool)

    return out.replace([np.inf, -np.inf], np.nan)


def _cooldown_mask(candidate: np.ndarray, *, cooldown_bars: int) -> np.ndarray:
    keep = np.zeros(len(candidate), dtype=bool)
    last = -10**12
    for pos in np.flatnonzero(candidate):
        if int(pos) - int(last) > int(cooldown_bars):
            keep[int(pos)] = True
            last = int(pos)
    return keep


def detect_pressure_events(
    features: pd.DataFrame,
    *,
    windows: Iterable[int],
    min_pressure_z: float = 1.0,
    cooldown_multiplier: float = 1.0,
) -> pd.DataFrame:
    """Detect high-recall pressure onsets without price-direction filtering.

    An event begins when pressure crosses above ``min_pressure_z`` or when the
    signed pressure direction flips while pressure remains above the threshold.
    A short per-window cooldown suppresses threshold chatter but does not merge
    opposite-direction events.
    """
    rows: list[pd.DataFrame] = []
    index = pd.DatetimeIndex(features.index)
    for window in sorted(set(int(v) for v in windows)):
        suffix = f"w{window}"
        required = [
            f"feature_ready_{suffix}",
            f"pressure_direction_{suffix}",
            f"pressure_z_{suffix}",
        ]
        missing = [column for column in required if column not in features.columns]
        if missing:
            raise KeyError(f"missing flow-impact feature columns for window={window}: {missing}")

        ready = features[f"feature_ready_{suffix}"].fillna(False).astype(bool)
        direction = pd.to_numeric(features[f"pressure_direction_{suffix}"], errors="coerce").fillna(0.0)
        pressure_z = pd.to_numeric(features[f"pressure_z_{suffix}"], errors="coerce")
        active = ready & pressure_z.ge(float(min_pressure_z)) & direction.ne(0.0)
        onset = active & ((~active.shift(1, fill_value=False)) | direction.ne(direction.shift(1)))

        cooldown = max(0, int(round(float(cooldown_multiplier) * window)))
        keep = np.zeros(len(features), dtype=bool)
        for side in (-1, 1):
            candidate = (onset & direction.eq(side)).to_numpy(dtype=bool)
            keep |= _cooldown_mask(candidate, cooldown_bars=cooldown)
        positions = np.flatnonzero(keep)
        if not len(positions):
            continue

        columns = [
            f"pressure_z_{suffix}",
            f"pressure_notional_{suffix}",
            f"flow_ratio_{suffix}",
            f"trade_imbalance_{suffix}",
            f"large_flow_ratio_{suffix}",
            f"large_notional_share_{suffix}",
            f"large_trade_share_{suffix}",
            f"flow_concentration_{suffix}",
            f"flow_persistence_{suffix}",
            f"notional_ratio_{suffix}",
            f"avg_trade_notional_ratio_{suffix}",
            f"max_trade_notional_ratio_{suffix}",
            f"activity_z_{suffix}",
            f"price_response_{suffix}",
            f"price_response_norm_{suffix}",
            f"pressure_effectiveness_{suffix}",
            f"impact_bps_per_million_{suffix}",
            f"direction_close_location_{suffix}",
        ]
        part = features.iloc[positions][columns].copy()
        part.columns = [column.removesuffix(f"_{suffix}") for column in columns]
        part.insert(0, "signal_bar_pos", positions.astype(np.int64))
        part.insert(0, "signal_time", index[positions])
        part.insert(0, "pressure_window_bars", int(window))
        part.insert(0, "side", direction.iloc[positions].astype(int).to_numpy())
        rows.append(part.reset_index(drop=True))

    if not rows:
        return pd.DataFrame()
    events = pd.concat(rows, ignore_index=True)
    events = events.sort_values(["signal_time", "pressure_window_bars", "side"]).reset_index(drop=True)
    events.insert(0, "event_id", np.arange(1, len(events) + 1, dtype=np.int64))
    events["side_name"] = events["side"].map({1: "LONG", -1: "SHORT"})
    return events



def assign_pressure_event_clusters(events: pd.DataFrame, *, cluster_gap_bars: int) -> pd.DataFrame:
    """Assign cross-window cluster ids to the same observable pressure process.

    Events are sorted by signal-bar position. A new cluster starts when pressure
    direction changes or the gap from the previous event exceeds
    ``cluster_gap_bars``. The earliest event is the primary representative, so
    unique-event frequency never benefits from waiting for a later window.
    """
    if events.empty:
        out = events.copy()
        out["event_cluster_id"] = pd.Series(dtype="int64")
        out["cluster_primary_flag"] = pd.Series(dtype="bool")
        out["cluster_size"] = pd.Series(dtype="int64")
        return out
    if int(cluster_gap_bars) < 0:
        raise ValueError("cluster_gap_bars must be >= 0")
    required = {"signal_bar_pos", "side", "pressure_window_bars"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise KeyError(f"events missing cluster fields: {missing}")

    out = events.copy().sort_values(
        ["signal_bar_pos", "pressure_window_bars", "event_id"],
        kind="stable",
    ).reset_index(drop=True)
    cluster_ids = np.zeros(len(out), dtype=np.int64)
    current_cluster = 0
    previous_pos: int | None = None
    previous_side: int | None = None
    for i, row in enumerate(out.itertuples(index=False)):
        pos = int(row.signal_bar_pos)
        side = int(row.side)
        if (
            previous_pos is None
            or previous_side is None
            or side != previous_side
            or pos - previous_pos > int(cluster_gap_bars)
        ):
            current_cluster += 1
        cluster_ids[i] = current_cluster
        previous_pos = pos
        previous_side = side
    out["event_cluster_id"] = cluster_ids
    out["cluster_primary_flag"] = ~out["event_cluster_id"].duplicated(keep="first")
    sizes = out.groupby("event_cluster_id", observed=False)["event_cluster_id"].transform("size")
    out["cluster_size"] = sizes.astype(np.int64)
    return out.sort_values(["signal_time", "pressure_window_bars", "side"]).reset_index(drop=True)

def response_state_labels(values: pd.Series) -> pd.Series:
    """Predeclared semantic buckets for direction-adjusted response in vol units."""
    x = pd.to_numeric(values, errors="coerce")
    out = pd.Series("NA", index=values.index, dtype="object")
    out.loc[x < 0.0] = "opposite_or_absorbed"
    out.loc[x.between(0.0, 0.25, inclusive="left")] = "flat_0_0.25"
    out.loc[x.between(0.25, 0.75, inclusive="left")] = "moderate_0.25_0.75"
    out.loc[x >= 0.75] = "effective_ge_0.75"
    return out


def pressure_strength_labels(values: pd.Series) -> pd.Series:
    """Fixed pressure-z buckets retained across all years and directions."""
    x = pd.to_numeric(values, errors="coerce")
    out = pd.Series("NA", index=values.index, dtype="object")
    out.loc[x.between(1.0, 1.5, inclusive="left")] = "z1.0_1.5"
    out.loc[x.between(1.5, 2.0, inclusive="left")] = "z1.5_2.0"
    out.loc[x.between(2.0, 2.5, inclusive="left")] = "z2.0_2.5"
    out.loc[x >= 2.5] = "z_ge_2.5"
    return out
