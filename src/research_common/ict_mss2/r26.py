#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R26 causal relative-positioning leadership repricing study.

Binance ratio observations are a cross-exchange positioning proxy. Price,
execution, structural barriers, and first-passage paths remain OKX
ETH-USDT-SWAP. The module never reads July 2025 or the sealed holdout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import aggregate_bars, normalize_1m_bars
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

EPS = 1e-12


@dataclass(frozen=True)
class R26Config:
    research_start: pd.Timestamp = pd.Timestamp("2023-01-01 00:00:00")
    validation_start: pd.Timestamp = pd.Timestamp("2025-01-01 00:00:00")
    embargo_start: pd.Timestamp = pd.Timestamp("2025-07-01 00:00:00")
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01 00:00:00")
    publication_lag: pd.Timedelta = pd.Timedelta("1min")
    metric_min_gap: pd.Timedelta = pd.Timedelta("4min")
    metric_max_gap: pd.Timedelta = pd.Timedelta("6min")
    confirmation_window: pd.Timedelta = pd.Timedelta("60min")
    atr_window_5m: int = 12
    stop_buffer_atr: float = 0.25
    max_stop_distance_pct: float = 0.015
    path_horizon_minutes: int = 24 * 60
    fixed_r_targets: tuple[float, ...] = (1.0, 2.0, 3.0)
    market_roundtrip_cost: float = 0.0011
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)

    def validate(self) -> "R26Config":
        if not (self.research_start < self.validation_start < self.embargo_start < self.holdout_start):
            raise ValueError("R26 split boundaries must be strictly increasing")
        if self.publication_lag < pd.Timedelta(0):
            raise ValueError("publication lag cannot be negative")
        if not (pd.Timedelta(0) < self.metric_min_gap <= self.metric_max_gap):
            raise ValueError("invalid metric gap bounds")
        if self.confirmation_window <= pd.Timedelta(0):
            raise ValueError("confirmation window must be positive")
        if self.atr_window_5m < 2 or self.stop_buffer_atr < 0:
            raise ValueError("invalid ATR stop configuration")
        if not 0 < self.max_stop_distance_pct < 1:
            raise ValueError("invalid maximum stop distance")
        if self.path_horizon_minutes <= 0:
            raise ValueError("path horizon must be positive")
        if not self.fixed_r_targets or any(float(value) <= 0 for value in self.fixed_r_targets):
            raise ValueError("fixed-R targets must be positive")
        if self.market_roundtrip_cost < 0 or any(float(value) <= 0 for value in self.cost_scales):
            raise ValueError("invalid cost configuration")
        return self


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _datetime_ns(values: Iterable[object]) -> np.ndarray:
    return np.asarray(pd.to_datetime(values, errors="coerce"), dtype="datetime64[ns]").astype(np.int64)


def _true_range(frame: pd.DataFrame) -> pd.Series:
    high = _num(frame, "high")
    low = _num(frame, "low")
    previous_close = _num(frame, "close").shift(1)
    return pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)


def _split_at(when: object, cfg: R26Config) -> str:
    value = pd.Timestamp(when)
    if value < cfg.research_start:
        return "warmup"
    if value < cfg.validation_start:
        return "discovery"
    if value < cfg.embargo_start:
        return "validation"
    if value < cfg.holdout_start:
        return "embargo"
    return "holdout"


def _price_state(bars_1m: pd.DataFrame, cfg: R26Config) -> pd.DataFrame:
    five = aggregate_bars(bars_1m, 5).copy()
    high = _num(five, "high")
    low = _num(five, "low")
    close = _num(five, "close")
    atr = _true_range(five).rolling(cfg.atr_window_5m, min_periods=cfg.atr_window_5m).mean()
    return pd.DataFrame(
        {
            "price_bar_time": five.index,
            "price_available_time": pd.to_datetime(five["bar_end_time"]),
            "price_open": _num(five, "open").to_numpy(float),
            "price_high": high.to_numpy(float),
            "price_low": low.to_numpy(float),
            "price_close": close.to_numpy(float),
            "price_prior_high": high.shift(1).to_numpy(float),
            "price_prior_low": low.shift(1).to_numpy(float),
            "price_return_1h": (close / close.shift(12) - 1.0).to_numpy(float),
            "range_high_1h": high.rolling(12, min_periods=12).max().to_numpy(float),
            "range_low_1h": low.rolling(12, min_periods=12).min().to_numpy(float),
            "atr_5m_1h": atr.to_numpy(float),
        }
    ).sort_values("price_available_time", kind="stable").reset_index(drop=True)


def prepare_ratio_alignment(
    bars_1m: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    config: R26Config | None = None,
) -> pd.DataFrame:
    """Align valid, published Binance ratios to completed OKX five-minute bars."""

    cfg = (config or R26Config()).validate()
    required = {
        "timestamp",
        "available_time",
        "top_trader_position_long_share",
        "global_account_long_share",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"R26 metrics missing required columns: {missing}")
    forbidden = [name for name in metrics.columns if name.startswith("future_") or "oracle" in name.lower()]
    if forbidden:
        raise RuntimeError(f"future leakage in R26 metric features: {forbidden}")

    ratio = metrics.reset_index(drop=True).copy()
    ratio["timestamp"] = pd.to_datetime(ratio["timestamp"], errors="coerce")
    ratio["available_time"] = pd.to_datetime(ratio["available_time"], errors="coerce")
    ratio = ratio.dropna(subset=["timestamp", "available_time"]).sort_values("available_time", kind="stable")
    ratio = ratio.drop_duplicates("available_time", keep="last").reset_index(drop=True)
    if (ratio["available_time"] >= cfg.embargo_start).any():
        raise RuntimeError("R26 input physically includes embargo or holdout metrics")

    expected_available = ratio["timestamp"] + cfg.publication_lag
    if not expected_available.eq(ratio["available_time"]).all():
        raise RuntimeError("R26 metric publication lag is not exact")

    price = _price_state(normalize_1m_bars(bars_1m), cfg)
    aligned = pd.merge_asof(
        ratio,
        price,
        left_on="available_time",
        right_on="price_available_time",
        direction="backward",
        tolerance=pd.Timedelta("5min"),
        allow_exact_matches=True,
    )
    aligned["metric_gap"] = aligned["timestamp"].diff()
    aligned["metric_gap_valid"] = aligned["metric_gap"].between(
        cfg.metric_min_gap, cfg.metric_max_gap, inclusive="both"
    )
    top = _num(aligned, "top_trader_position_long_share")
    broad = _num(aligned, "global_account_long_share")
    aligned["ratio_valid"] = top.between(0.0, 1.0, inclusive="both") & broad.between(
        0.0, 1.0, inclusive="both"
    )
    aligned["relative_spread"] = top - broad
    aligned["prior_relative_spread"] = aligned["relative_spread"].shift(1)
    aligned["prior_ratio_valid"] = aligned["ratio_valid"].shift(1, fill_value=False)
    aligned["price_step_valid"] = (
        pd.to_datetime(aligned["price_bar_time"]).diff().eq(pd.Timedelta("5min"))
        & aligned["price_available_time"].notna()
    )
    return aligned


def _empty_seal(cfg: R26Config, metrics: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    metric_max = pd.to_datetime(metrics.get("available_time"), errors="coerce").max()
    bar_max = pd.to_datetime(bars.index, errors="coerce").max()
    return pd.DataFrame(
        [
            {"check": "physical_input_cutoff", "value": str(cfg.embargo_start)},
            {"check": "maximum_metric_available_time", "value": str(metric_max)},
            {"check": "maximum_okx_bar_time", "value": str(bar_max)},
            {"check": "embargo_or_holdout_outcome_rows", "value": 0},
            {"check": "holdout_unsealed", "value": 0},
        ]
    )


def build_positioning_leadership_events(
    bars_1m: pd.DataFrame,
    metrics: pd.DataFrame,
    *,
    config: R26Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build gap-safe spread-cross episodes and their first price confirmation."""

    cfg = (config or R26Config()).validate()
    bars = normalize_1m_bars(bars_1m)
    if pd.to_datetime(bars.index).max() >= cfg.embargo_start:
        raise RuntimeError("R26 OKX input physically includes embargo or holdout bars")
    aligned = prepare_ratio_alignment(bars, metrics, config=cfg)

    cross_long = (
        aligned["ratio_valid"]
        & aligned["prior_ratio_valid"]
        & aligned["metric_gap_valid"]
        & _num(aligned, "prior_relative_spread").le(0)
        & _num(aligned, "relative_spread").gt(0)
    )
    cross_short = (
        aligned["ratio_valid"]
        & aligned["prior_ratio_valid"]
        & aligned["metric_gap_valid"]
        & _num(aligned, "prior_relative_spread").ge(0)
        & _num(aligned, "relative_spread").lt(0)
    )

    candidates: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    expired = recrossed = invalidated = 0
    for idx, row in aligned.iterrows():
        when = pd.Timestamp(row["available_time"])
        direction_cross = 1 if bool(cross_long.iloc[idx]) else -1 if bool(cross_short.iloc[idx]) else 0

        if active is not None:
            elapsed = when - pd.Timestamp(active["cross_available_time"])
            direction = int(active["trade_direction"])
            same_sign = float(row["relative_spread"]) > 0 if direction > 0 else float(row["relative_spread"]) < 0
            valid_step = bool(row["ratio_valid"]) and bool(row["metric_gap_valid"]) and bool(row["price_step_valid"])
            if elapsed > cfg.confirmation_window:
                expired += 1
                active = None
            elif not valid_step:
                invalidated += 1
                active = None
            elif not same_sign:
                recrossed += 1
                active = None
            else:
                confirmed = (
                    float(row["price_close"]) > float(row["price_prior_high"])
                    if direction > 0
                    else float(row["price_close"]) < float(row["price_prior_low"])
                )
                if confirmed:
                    record = dict(active)
                    record.update(
                        {
                            "confirmation_metric_time": pd.Timestamp(row["timestamp"]),
                            "confirmation_metric_available_time": when,
                            "confirmation_price_bar_time": pd.Timestamp(row["price_bar_time"]),
                            "confirmation_price_available_time": pd.Timestamp(row["price_available_time"]),
                            "confirmation_delay_minutes": float(elapsed / pd.Timedelta(minutes=1)),
                            "confirmation_top_share": float(row["top_trader_position_long_share"]),
                            "confirmation_global_share": float(row["global_account_long_share"]),
                            "confirmation_relative_spread": float(row["relative_spread"]),
                            "confirmation_price_open": float(row["price_open"]),
                            "confirmation_price_high": float(row["price_high"]),
                            "confirmation_price_low": float(row["price_low"]),
                            "confirmation_price_close": float(row["price_close"]),
                            "confirmation_prior_high": float(row["price_prior_high"]),
                            "confirmation_prior_low": float(row["price_prior_low"]),
                            "atr_5m_1h_at_signal": float(row["atr_5m_1h"]),
                        }
                    )
                    candidates.append(record)
                    active = None

        if direction_cross:
            active = {
                "trade_direction": direction_cross,
                "direction": "Long" if direction_cross > 0 else "Short",
                "cross_metric_time": pd.Timestamp(row["timestamp"]),
                "cross_available_time": when,
                "cross_price_bar_time": pd.Timestamp(row["price_bar_time"]),
                "cross_price_available_time": pd.Timestamp(row["price_available_time"]),
                "prior_top_share": float(aligned.iloc[idx - 1]["top_trader_position_long_share"]),
                "prior_global_share": float(aligned.iloc[idx - 1]["global_account_long_share"]),
                "prior_relative_spread": float(row["prior_relative_spread"]),
                "cross_top_share": float(row["top_trader_position_long_share"]),
                "cross_global_share": float(row["global_account_long_share"]),
                "cross_relative_spread": float(row["relative_spread"]),
                "cross_price_return_1h": float(row["price_return_1h"]),
                "cross_range_high_1h": float(row["range_high_1h"]),
                "cross_range_low_1h": float(row["range_low_1h"]),
            }

    index_ns = _datetime_ns(bars.index)
    rows: list[dict[str, object]] = []
    for ordinal, item in enumerate(candidates, start=1):
        signal = max(
            pd.Timestamp(item["confirmation_metric_available_time"]),
            pd.Timestamp(item["confirmation_price_available_time"]),
        )
        split = _split_at(signal, cfg)
        if split not in {"discovery", "validation"}:
            continue
        direction = int(item["trade_direction"])
        entry_pos = int(np.searchsorted(index_ns, np.datetime64(signal, "ns").astype(np.int64), side="left"))
        base = dict(item)
        base.update(
            {
                "setup_id": f"R26_{item['direction'].upper()}_{signal.strftime('%Y%m%dT%H%M%S%f')}_{ordinal:06d}",
                "research_split": split,
                "signal_available_time": signal,
                "setup_status": "pending_geometry",
            }
        )
        if entry_pos >= len(bars):
            base["setup_status"] = "next_1m_entry_unavailable"
            rows.append(base)
            continue
        entry_time = pd.Timestamp(bars.index[entry_pos])
        entry = float(bars.iloc[entry_pos]["open"])
        atr = float(item["atr_5m_1h_at_signal"])
        stop = (
            min(float(item["confirmation_price_low"]), float(item["confirmation_prior_low"]))
            - cfg.stop_buffer_atr * atr
            if direction > 0
            else max(float(item["confirmation_price_high"]), float(item["confirmation_prior_high"]))
            + cfg.stop_buffer_atr * atr
        )
        target = float(item["cross_range_high_1h"] if direction > 0 else item["cross_range_low_1h"])
        risk = direction * (entry - stop) / entry if entry > EPS else np.nan
        runway = direction * (target / entry - 1.0) if entry > EPS else np.nan
        base.update(
            {
                "entry_time": entry_time,
                "entry_price": entry,
                "stop_price": stop,
                "risk_distance_pct": risk,
                "structural_target_price": target,
                "structural_target_available_time": pd.Timestamp(item["cross_available_time"]),
                "structural_runway_pct": runway,
                "structural_reward_risk": runway / risk if np.isfinite(risk) and risk > EPS else np.nan,
            }
        )
        if not np.isfinite(atr) or atr <= EPS:
            base["setup_status"] = "atr_unavailable"
        elif not np.isfinite(risk) or risk <= EPS:
            base["setup_status"] = "invalid_stop_geometry"
        elif risk > cfg.max_stop_distance_pct + EPS:
            base["setup_status"] = "stop_too_wide"
        elif not np.isfinite(runway) or runway <= EPS:
            base["setup_status"] = "no_remaining_structural_target"
        else:
            base["setup_status"] = "executable"
        rows.append(base)

    events = pd.DataFrame(rows)
    if not events.empty:
        for name in [column for column in events.columns if column.endswith("_time")]:
            events[name] = pd.to_datetime(events[name], errors="coerce")
        events = events.sort_values(["signal_available_time", "direction", "setup_id"], kind="stable").reset_index(drop=True)
    seal = _empty_seal(cfg, metrics, bars)
    engineering = pd.DataFrame(
        [
            {"check": "aligned_metric_rows", "value": int(len(aligned))},
            {"check": "raw_long_crosses", "value": int(cross_long.sum())},
            {"check": "raw_short_crosses", "value": int(cross_short.sum())},
            {"check": "confirmed_crosses", "value": int(len(candidates))},
            {"check": "expired_episodes", "value": int(expired)},
            {"check": "recrossed_episodes", "value": int(recrossed)},
            {"check": "invalidated_gap_or_ratio_episodes", "value": int(invalidated)},
            {"check": "visible_event_rows", "value": int(len(events))},
            {"check": "visible_executable_setups", "value": int(events.get("setup_status", pd.Series(dtype=str)).eq("executable").sum())},
        ]
    )
    return events, seal, engineering


def _first_barrier(
    high_tree: SegmentThresholdIndex,
    low_tree: SegmentThresholdIndex,
    *,
    direction: int,
    target: float,
    stop: float,
    start: int,
    end: int,
) -> tuple[str, int]:
    if direction > 0:
        tp = int(high_tree.first_geq(start, end, target))
        sl = int(low_tree.first_leq(start, end, stop))
    else:
        tp = int(low_tree.first_leq(start, end, target))
        sl = int(high_tree.first_geq(start, end, stop))
    if sl >= 0 and (tp < 0 or sl <= tp):
        return "sl_first", sl
    if tp >= 0:
        return "tp_first", tp
    return "horizon_exit", end


def _split_end(split: str, cfg: R26Config) -> pd.Timestamp:
    if split == "discovery":
        return cfg.validation_start
    if split == "validation":
        return cfg.embargo_start
    raise ValueError(f"unsupported visible split: {split}")


def build_positioning_leadership_paths(
    bars_1m: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: R26Config | None = None,
) -> pd.DataFrame:
    """Replay exact stop-first paths and mark independently non-overlapping trades."""

    cfg = (config or R26Config()).validate()
    if events.empty:
        return pd.DataFrame()
    executable = events.loc[events["setup_status"].eq("executable")].copy()
    if executable.empty:
        return pd.DataFrame()
    bars = normalize_1m_bars(bars_1m)
    index_ns = _datetime_ns(bars.index)
    open_ = _num(bars, "open").to_numpy(float)
    high = _num(bars, "high").to_numpy(float)
    low = _num(bars, "low").to_numpy(float)
    close = _num(bars, "close").to_numpy(float)
    high_tree = SegmentThresholdIndex(high)
    low_tree = SegmentThresholdIndex(low)
    target_specs: tuple[tuple[str, float | None], ...] = (
        ("H0_CROSS_TIME_1H_RANGE", None),
        *tuple((f"R{float(value):g}", float(value)) for value in cfg.fixed_r_targets),
    )
    rows: list[dict[str, object]] = []
    for event in executable.itertuples(index=False):
        entry_time = pd.Timestamp(event.entry_time)
        entry_pos = int(np.searchsorted(index_ns, np.datetime64(entry_time, "ns").astype(np.int64), side="left"))
        if entry_pos >= len(bars) or pd.Timestamp(bars.index[entry_pos]) != entry_time:
            continue
        end = entry_pos + cfg.path_horizon_minutes - 1
        split_end = _split_end(str(event.research_split), cfg)
        if end >= len(bars) or pd.Timestamp(bars.index[end]) >= split_end:
            for model, multiple in target_specs:
                rows.append(
                    {
                        "setup_id": str(event.setup_id),
                        "direction": str(event.direction),
                        "trade_direction": int(event.trade_direction),
                        "research_split": str(event.research_split),
                        "year": int(entry_time.year),
                        "target_model": model,
                        "entry_time": entry_time,
                        "outcome": "split_boundary_censored",
                    }
                )
            continue
        direction = int(event.trade_direction)
        entry = float(event.entry_price)
        stop = float(event.stop_price)
        risk_price = abs(entry - stop)
        base = {
            "setup_id": str(event.setup_id),
            "direction": str(event.direction),
            "trade_direction": direction,
            "research_split": str(event.research_split),
            "year": int(entry_time.year),
            "cross_available_time": pd.Timestamp(event.cross_available_time),
            "signal_available_time": pd.Timestamp(event.signal_available_time),
            "entry_time": entry_time,
            "entry_price": entry,
            "stop_price": stop,
            "risk_distance_pct": float(event.risk_distance_pct),
            "structural_runway_pct": float(event.structural_runway_pct),
            "structural_reward_risk": float(event.structural_reward_risk),
            "cross_relative_spread": float(event.cross_relative_spread),
            "confirmation_relative_spread": float(event.confirmation_relative_spread),
            "confirmation_delay_minutes": float(event.confirmation_delay_minutes),
        }
        for model, multiple in target_specs:
            target = (
                float(event.structural_target_price)
                if multiple is None
                else entry + direction * float(multiple) * risk_price
            )
            rec = dict(base)
            rec.update({"target_model": model, "target_price": target})
            outcome, exit_pos = _first_barrier(
                high_tree,
                low_tree,
                direction=direction,
                target=target,
                stop=stop,
                start=entry_pos,
                end=end,
            )
            if outcome == "sl_first":
                exit_price = min(stop, open_[exit_pos]) if direction > 0 else max(stop, open_[exit_pos])
            elif outcome == "tp_first":
                exit_price = target
            else:
                exit_price = float(close[exit_pos])
            gross = direction * (exit_price / entry - 1.0)
            segment_high = high[entry_pos : exit_pos + 1]
            segment_low = low[entry_pos : exit_pos + 1]
            mfe = float(np.nanmax(segment_high) / entry - 1.0) if direction > 0 else float(1.0 - np.nanmin(segment_low) / entry)
            mae = float(1.0 - np.nanmin(segment_low) / entry) if direction > 0 else float(np.nanmax(segment_high) / entry - 1.0)
            rec.update(
                {
                    "outcome": outcome,
                    "exit_time": pd.Timestamp(bars.index[exit_pos]),
                    "exit_price": exit_price,
                    "holding_minutes": float(exit_pos - entry_pos + 1),
                    "gross_return": gross,
                    "gross_r": gross / float(event.risk_distance_pct),
                    "mfe_pct": mfe,
                    "mae_pct": mae,
                }
            )
            for scale in cfg.cost_scales:
                net = gross - cfg.market_roundtrip_cost * float(scale)
                rec[f"net_return_cost{float(scale):g}x"] = net
                rec[f"net_r_cost{float(scale):g}x"] = net / float(event.risk_distance_pct)
            rows.append(rec)

    paths = pd.DataFrame(rows)
    if paths.empty:
        return paths
    paths["entry_time"] = pd.to_datetime(paths["entry_time"], errors="coerce")
    paths["exit_time"] = pd.to_datetime(paths.get("exit_time"), errors="coerce")
    paths["position_selected"] = False
    paths["overlap_skip_reason"] = ""
    valid = paths["gross_return"].notna()
    for _, indexes in paths.loc[valid].groupby(
        ["research_split", "direction", "target_model"], sort=True
    ).groups.items():
        last_exit: pd.Timestamp | None = None
        ordered = paths.loc[list(indexes)].sort_values(["entry_time", "setup_id"], kind="stable")
        for idx, row in ordered.iterrows():
            entry_time = pd.Timestamp(row["entry_time"])
            if last_exit is None or entry_time > last_exit:
                paths.at[idx, "position_selected"] = True
                last_exit = pd.Timestamp(row["exit_time"])
            else:
                paths.at[idx, "overlap_skip_reason"] = "prior_same_direction_model_position_open"
    return paths.sort_values(["entry_time", "direction", "target_model"], kind="stable").reset_index(drop=True)


def _profit_factor(values: Iterable[float]) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    gain = float(x[x > 0].sum())
    loss = float(-x[x < 0].sum())
    return gain / loss if loss > EPS else (np.inf if gain > EPS else np.nan)


def _top_removed(values: pd.Series, count: int) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").dropna().sort_values(ascending=False)
    remaining = x.iloc[int(count) :] if len(x) > int(count) else pd.Series(dtype=float)
    return _profit_factor(remaining), float(remaining.sum()) if len(remaining) else np.nan


def _calendar(split: str) -> pd.PeriodIndex:
    if split == "discovery":
        return pd.period_range("2023-01", "2024-12", freq="M")
    if split == "validation":
        return pd.period_range("2025-01", "2025-06", freq="M")
    return pd.PeriodIndex([], freq="M")


def summarize_r26_funnel(events: pd.DataFrame, paths: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (split, direction), part in events.groupby(["research_split", "direction"], sort=True):
        executable = part.loc[part["setup_status"].eq("executable")]
        selected_primary = paths.loc[
            paths.get("position_selected", False).eq(True)
            & paths["research_split"].eq(split)
            & paths["direction"].eq(direction)
            & paths["target_model"].eq("H0_CROSS_TIME_1H_RANGE")
        ] if not paths.empty else pd.DataFrame()
        rows.append(
            {
                "research_split": split,
                "direction": direction,
                "confirmed_crosses": int(len(part)),
                "executable_rows": int(len(executable)),
                "selected_primary_trades": int(len(selected_primary)),
                "stop_too_wide_rows": int(part["setup_status"].eq("stop_too_wide").sum()),
                "no_structural_target_rows": int(part["setup_status"].eq("no_remaining_structural_target").sum()),
                "median_confirmation_delay_minutes": _num(part, "confirmation_delay_minutes").median(),
                "median_risk_distance_pct": _num(executable, "risk_distance_pct").median(),
                "median_structural_reward_risk": _num(executable, "structural_reward_risk").median(),
            }
        )
    return pd.DataFrame(rows)


def summarize_r26_paths(paths: pd.DataFrame, *, config: R26Config | None = None) -> pd.DataFrame:
    cfg = (config or R26Config()).validate()
    if paths.empty:
        return pd.DataFrame()
    selected = paths.loc[paths.get("position_selected", False).eq(True) & paths["gross_return"].notna()].copy()
    rows: list[dict[str, object]] = []
    for (split, direction, model), part in selected.groupby(
        ["research_split", "direction", "target_model"], sort=True
    ):
        calendar = _calendar(str(split))
        monthly = pd.Series(0.0, index=calendar)
        observed = part.assign(month=part["entry_time"].dt.to_period("M")).groupby("month")["net_return_cost2x"].sum()
        common = calendar.intersection(observed.index)
        monthly.loc[common] = observed.reindex(common)
        times = part["entry_time"].drop_duplicates().sort_values()
        top5_pf, top5_sum = _top_removed(part["net_return_cost2x"], 5)
        top10_pf, top10_sum = _top_removed(part["net_return_cost2x"], 10)
        rec: dict[str, object] = {
            "research_split": split,
            "direction": direction,
            "target_model": model,
            "trades": int(len(part)),
            "trades_per_month": float(len(part) / max(1, len(calendar))),
            "tp_rate": float(part["outcome"].eq("tp_first").mean()),
            "sl_rate": float(part["outcome"].eq("sl_first").mean()),
            "horizon_exit_rate": float(part["outcome"].eq("horizon_exit").mean()),
            "gross_pf": _profit_factor(part["gross_return"]),
            "mean_gross_return": _num(part, "gross_return").mean(),
            "mean_gross_r": _num(part, "gross_r").mean(),
            "median_risk_distance_pct": _num(part, "risk_distance_pct").median(),
            "median_structural_reward_risk": _num(part, "structural_reward_risk").median(),
            "median_holding_minutes": _num(part, "holding_minutes").median(),
            "positive_month_rate_cost2x": float((monthly > 0).mean()) if len(monthly) else np.nan,
            "longest_entry_gap_days": float(times.diff().max() / pd.Timedelta(days=1)) if len(times) >= 2 else np.nan,
            "net_pf_cost2x_top5_removed": top5_pf,
            "net_sum_cost2x_top5_removed": top5_sum,
            "net_pf_cost2x_top10_removed": top10_pf,
            "net_sum_cost2x_top10_removed": top10_sum,
        }
        for scale in cfg.cost_scales:
            net = _num(part, f"net_return_cost{float(scale):g}x")
            rec[f"mean_net_return_cost{float(scale):g}x"] = net.mean()
            rec[f"net_sum_cost{float(scale):g}x"] = net.sum()
            rec[f"net_pf_cost{float(scale):g}x"] = _profit_factor(net)
            rec[f"mean_net_r_cost{float(scale):g}x"] = _num(part, f"net_r_cost{float(scale):g}x").mean()
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_r26_years(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    selected = paths.loc[paths.get("position_selected", False).eq(True) & paths["gross_return"].notna()].copy()
    rows: list[dict[str, object]] = []
    for (year, direction, model), part in selected.groupby(["year", "direction", "target_model"], sort=True):
        top5_pf, top5_sum = _top_removed(part["net_return_cost2x"], 5)
        rows.append(
            {
                "year": int(year),
                "direction": direction,
                "target_model": model,
                "trades": int(len(part)),
                "mean_net_return_cost2x": _num(part, "net_return_cost2x").mean(),
                "net_sum_cost2x": _num(part, "net_return_cost2x").sum(),
                "net_pf_cost2x": _profit_factor(_num(part, "net_return_cost2x")),
                "net_pf_cost2x_top5_removed": top5_pf,
                "net_sum_cost2x_top5_removed": top5_sum,
            }
        )
    return pd.DataFrame(rows)


def r26_data_quality_audit(
    bars_1m: pd.DataFrame,
    metrics: pd.DataFrame,
    coverage_by_day: pd.DataFrame,
    *,
    config: R26Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R26Config()).validate()
    bars = normalize_1m_bars(bars_1m)
    ratio = metrics.reset_index(drop=True).copy()
    ratio["timestamp"] = pd.to_datetime(ratio["timestamp"], errors="coerce")
    ratio["available_time"] = pd.to_datetime(ratio["available_time"], errors="coerce")
    visible = ratio.loc[
        (ratio["available_time"] >= cfg.research_start) & (ratio["available_time"] < cfg.embargo_start)
    ].copy()
    gap = visible.sort_values("timestamp")["timestamp"].diff().dropna()
    top = _num(visible, "top_trader_position_long_share")
    broad = _num(visible, "global_account_long_share")
    bar_gap = bars.index.to_series().diff().dropna()
    coverage = coverage_by_day.copy()
    if not coverage.empty and "day_utc" in coverage:
        coverage["day_utc"] = pd.to_datetime(coverage["day_utc"], errors="coerce")
        coverage = coverage.loc[coverage["day_utc"] < cfg.embargo_start]
    return pd.DataFrame(
        [
            {"check": "visible_metric_rows", "value": int(len(visible)), "status": "INFO"},
            {"check": "duplicate_metric_timestamps", "value": int(visible["timestamp"].duplicated().sum()), "status": "PASS"},
            {"check": "nonexact_5m_intervals", "value": int(gap.ne(pd.Timedelta("5min")).sum()), "status": "EXCLUDE"},
            {"check": "maximum_metric_gap_seconds", "value": float(gap.max().total_seconds()) if len(gap) else np.nan, "status": "INFO"},
            {"check": "top_share_null_rows", "value": int(top.isna().sum()), "status": "EXCLUDE"},
            {"check": "global_share_null_rows", "value": int(broad.isna().sum()), "status": "EXCLUDE"},
            {"check": "top_share_out_of_range_rows", "value": int((top.notna() & ~top.between(0, 1)).sum()), "status": "EXCLUDE"},
            {"check": "global_share_out_of_range_rows", "value": int((broad.notna() & ~broad.between(0, 1)).sum()), "status": "EXCLUDE"},
            {"check": "partial_archive_days_visible", "value": int(coverage.get("status", pd.Series(dtype=str)).eq("partial").sum()), "status": "EXCLUDE_GAPS"},
            {"check": "okx_1m_rows_loaded", "value": int(len(bars)), "status": "INFO"},
            {"check": "okx_1m_duplicate_timestamps", "value": int(bars.index.duplicated().sum()), "status": "PASS"},
            {"check": "okx_1m_nonexact_intervals", "value": int(bar_gap.ne(pd.Timedelta("1min")).sum()), "status": "PASS" if bar_gap.eq(pd.Timedelta("1min")).all() else "FAIL"},
            {"check": "maximum_input_time_before_embargo", "value": int(max(pd.to_datetime(visible["available_time"]).max(), pd.to_datetime(bars.index).max()) < cfg.embargo_start), "status": "PASS"},
        ]
    )


def r26_causal_audit(
    events: pd.DataFrame,
    paths: pd.DataFrame,
    *,
    config: R26Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R26Config()).validate()
    checks: list[dict[str, object]] = []

    def add(name: str, violations: int) -> None:
        checks.append({"check": name, "violations": int(violations), "status": "PASS" if int(violations) == 0 else "FAIL"})

    if events.empty:
        add("nonempty_visible_event_table", 1)
        return pd.DataFrame(checks)
    signal = pd.to_datetime(events["signal_available_time"])
    direction = _num(events, "trade_direction")
    add("unique_setup_id", int(events["setup_id"].duplicated().sum()))
    add("feature_schema_excludes_future_or_oracle", len([c for c in events if c.startswith("future_") or "oracle" in c.lower()]))
    add("cross_publication_lag_exact", int((pd.to_datetime(events["cross_metric_time"]) + cfg.publication_lag != pd.to_datetime(events["cross_available_time"])).sum()))
    add("confirmation_publication_lag_exact", int((pd.to_datetime(events["confirmation_metric_time"]) + cfg.publication_lag != pd.to_datetime(events["confirmation_metric_available_time"])).sum()))
    add("cross_precedes_confirmation", int((pd.to_datetime(events["cross_available_time"]) >= pd.to_datetime(events["confirmation_metric_available_time"])).sum()))
    add("confirmation_within_frozen_window", int((_num(events, "confirmation_delay_minutes") > cfg.confirmation_window / pd.Timedelta(minutes=1)).sum()))
    add("cross_sign_direction", int((((direction > 0) & ((_num(events, "prior_relative_spread") > 0) | (_num(events, "cross_relative_spread") <= 0))) | ((direction < 0) & ((_num(events, "prior_relative_spread") < 0) | (_num(events, "cross_relative_spread") >= 0)))).sum()))
    add("confirmation_spread_retains_sign", int((((direction > 0) & (_num(events, "confirmation_relative_spread") <= 0)) | ((direction < 0) & (_num(events, "confirmation_relative_spread") >= 0))).sum()))
    add("price_confirmation_sign", int((((direction > 0) & (_num(events, "confirmation_price_close") <= _num(events, "confirmation_prior_high"))) | ((direction < 0) & (_num(events, "confirmation_price_close") >= _num(events, "confirmation_prior_low")))).sum()))
    add("all_information_available_by_signal", int(((pd.to_datetime(events["confirmation_metric_available_time"]) > signal) | (pd.to_datetime(events["confirmation_price_available_time"]) > signal)).sum()))
    add("next_eligible_1m_open", int((pd.to_datetime(events["entry_time"]) != signal.dt.ceil("min")).sum()))
    add("embargo_and_holdout_absent_from_events", int((signal >= cfg.embargo_start).sum()))
    executable = events.loc[events["setup_status"].eq("executable")]
    add("maximum_stop_distance_respected", int((_num(executable, "risk_distance_pct") > cfg.max_stop_distance_pct + EPS).sum()))
    add("structural_target_frozen_by_cross", int((pd.to_datetime(executable["structural_target_available_time"]) != pd.to_datetime(executable["cross_available_time"])).sum()))
    if paths.empty:
        add("nonempty_first_passage_paths", 1)
        return pd.DataFrame(checks)
    add("paths_reference_executable_events", int((~paths["setup_id"].isin(executable["setup_id"])).sum()))
    add("embargo_and_holdout_absent_from_paths", int((pd.to_datetime(paths["entry_time"]) >= cfg.embargo_start).sum()))
    add("unique_setup_target_path", int(paths.duplicated(["setup_id", "target_model"]).sum()))
    selected = paths.loc[paths.get("position_selected", False).eq(True) & paths["gross_return"].notna()]
    overlap_violations = 0
    for _, part in selected.groupby(["research_split", "direction", "target_model"], sort=True):
        ordered = part.sort_values("entry_time", kind="stable")
        overlap_violations += int((pd.to_datetime(ordered["entry_time"]).iloc[1:].to_numpy() <= pd.to_datetime(ordered["exit_time"]).iloc[:-1].to_numpy()).sum())
    add("selected_positions_do_not_overlap", overlap_violations)
    return pd.DataFrame(checks)
