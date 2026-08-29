#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal market-structure construction for ICT MSS research.

The module intentionally contains no data access.  The entrypoint loads 1m bare
candles through ``src.data_feed`` and hands a normalized OHLC frame here.

Timing convention
-----------------
All candle timestamps are bar-start times.  A 1m candle at 10:00 is observable
only at 10:01.  A left-labelled 15m candle at 10:00 is observable at 10:15.
A pivot is observable only after all right-confirmation candles have closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter
from src.research_common.swing_liquidity_atlas.thresholds import SegmentThresholdIndex

EPS = 1e-12
_REQUIRED = ("open", "high", "low", "close")


def normalize_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise ValueError("1m bars are empty")
    missing = [name for name in _REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError(f"1m bars missing required columns: {missing}")
    out = frame.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" not in out.columns:
            raise TypeError("1m bars require DatetimeIndex or timestamp column")
        out.index = pd.to_datetime(out.pop("timestamp"), errors="coerce")
    else:
        out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out = out.loc[~out.index.isna()].sort_index()
    out = out.loc[~out.index.duplicated(keep="last")]
    for name in _REQUIRED:
        out[name] = pd.to_numeric(out[name], errors="coerce")
    out = out.dropna(subset=list(_REQUIRED))
    if len(out) < 10:
        raise ValueError("fewer than 10 valid 1m bars")
    return out


def _rule(minutes: int) -> str:
    return f"{int(minutes)}min"


def aggregate_timeframe(primary: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Aggregate complete left-labelled HTF candles from the 1m execution axis."""

    bars = normalize_bars(primary)
    working = bars.loc[:, list(_REQUIRED)].copy()
    working["_source_bar_count"] = 1
    htf = working.resample(
        _rule(minutes),
        label="left",
        closed="left",
        origin="start_day",
    ).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        _source_bar_count=("_source_bar_count", "sum"),
    )
    htf = htf.dropna(subset=list(_REQUIRED))
    htf = htf.loc[htf["_source_bar_count"].eq(int(minutes))].copy()
    source_available_end = bars.index[-1] + pd.Timedelta(minutes=1)
    delta = pd.Timedelta(minutes=int(minutes))
    htf = htf.loc[(htf.index + delta) <= source_available_end].copy()
    htf["bar_end_time"] = htf.index + delta
    return htf


def _pivot_mask(values: np.ndarray, order: int, kind: str) -> np.ndarray:
    """Historical pivot identity; caller enforces causal availability separately."""

    arr = np.asarray(values, dtype=float)
    n = len(arr)
    k = int(order)
    if k < 1:
        raise ValueError("pivot order must be >= 1")
    if kind not in {"low", "high"}:
        raise ValueError("kind must be low/high")
    if n < 2 * k + 1:
        return np.zeros(n, dtype=bool)
    mask = np.isfinite(arr)
    mask[:k] = False
    mask[n - k :] = False
    for lag in range(1, k + 1):
        left = np.full(n, np.nan, dtype=float)
        right = np.full(n, np.nan, dtype=float)
        left[lag:] = arr[:-lag]
        right[:-lag] = arr[lag:]
        if kind == "low":
            mask &= arr < left
            mask &= arr <= right
        else:
            mask &= arr > left
            mask &= arr >= right
    return mask


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    return np.divide(num, den, out=np.full_like(num, np.nan, dtype=float), where=np.isfinite(den) & (np.abs(den) > EPS))


def build_htf_liquidity_levels(
    primary: pd.DataFrame,
    *,
    timeframes: Iterable[tuple[str, int]] = (("15m", 15), ("30m", 30), ("1H", 60), ("4H", 240)),
    confirmation_orders: Iterable[int] = (1, 2, 3, 5),
) -> pd.DataFrame:
    """Build causal swing-high/low liquidity levels from complete HTF candles.

    One physical pivot produces one level row.  Stronger eventual pivot orders
    are stored as later ``order_N_available_time`` fields instead of duplicate
    levels, so a level can only be classified as "obvious order-3" after that
    confirmation really existed in real time.
    """

    bars = normalize_bars(primary)
    orders = tuple(sorted(set(int(v) for v in confirmation_orders)))
    if not orders or orders[0] != 1:
        raise ValueError("confirmation_orders must include 1")
    rows: list[pd.DataFrame] = []
    for timeframe, minutes in timeframes:
        htf = aggregate_timeframe(bars, int(minutes))
        if htf.empty:
            continue
        delta = pd.Timedelta(minutes=int(minutes))
        for kind, price_col in (("low", "low"), ("high", "high")):
            values = pd.to_numeric(htf[price_col], errors="coerce").to_numpy(dtype=float)
            base_mask = _pivot_mask(values, 1, kind)
            positions = np.flatnonzero(base_mask)
            if not len(positions):
                continue
            masks = {order: _pivot_mask(values, order, kind) for order in orders}
            open_ = htf["open"].to_numpy(dtype=float)
            high = htf["high"].to_numpy(dtype=float)
            low = htf["low"].to_numpy(dtype=float)
            close = htf["close"].to_numpy(dtype=float)
            bar_range = np.maximum(high[positions] - low[positions], EPS)
            level = values[positions]
            frame = pd.DataFrame(
                {
                    "liquidity_side": "sell_side" if kind == "low" else "buy_side",
                    "pivot_kind": kind,
                    "source_timeframe": str(timeframe),
                    "source_timeframe_min": int(minutes),
                    "pivot_pos_htf": positions.astype(np.int64),
                    "pivot_time": htf.index[positions],
                    "pivot_bar_end_time": htf.index[positions] + delta,
                    "level_price": level,
                    "initial_available_time": htf.index[positions] + 2 * delta,
                    "pivot_range_bp": _safe_ratio(bar_range, close[positions]) * 10_000.0,
                    "pivot_body_fraction": _safe_ratio(np.abs(close[positions] - open_[positions]), bar_range),
                    # This is fully known by the time even the order-1 pivot is
                    # confirmed.  It measures whether the pivot candle itself
                    # closed away from the liquidity extreme rather than at it.
                    "pivot_rejection_fraction": np.where(
                        kind == "low",
                        _safe_ratio(close[positions] - low[positions], bar_range),
                        _safe_ratio(high[positions] - close[positions], bar_range),
                    ),
                }
            )
            max_eventual = np.ones(len(frame), dtype=np.int16)
            for order in orders:
                qualified = masks[order][positions]
                available = np.full(len(positions), np.datetime64("NaT"), dtype="datetime64[ns]")
                if np.any(qualified):
                    available[qualified] = (htf.index[positions[qualified]] + (order + 1) * delta).to_numpy(dtype="datetime64[ns]")
                frame[f"order_{order}_available_time"] = pd.to_datetime(available)
                frame[f"eventual_order_{order}"] = qualified.astype(np.int8)
                prominence = np.full(len(positions), np.nan, dtype=float)
                if np.any(qualified):
                    for local_idx in np.flatnonzero(qualified):
                        p = int(positions[local_idx])
                        left = values[p - order : p]
                        right = values[p + 1 : p + order + 1]
                        neighbors = np.concatenate((left, right))
                        if kind == "low":
                            clearance = float(np.nanmin(neighbors) - values[p])
                        else:
                            clearance = float(values[p] - np.nanmax(neighbors))
                        prominence[local_idx] = max(0.0, clearance) / max(abs(values[p]), EPS) * 10_000.0
                frame[f"order_{order}_prominence_bp"] = prominence
                max_eventual = np.where(qualified, int(order), max_eventual)
            frame["future_max_eventual_order_label"] = max_eventual
            rows.append(frame)
    if not rows:
        return pd.DataFrame()
    levels = pd.concat(rows, ignore_index=True, sort=False)
    levels = levels.sort_values(
        ["initial_available_time", "source_timeframe_min", "pivot_time", "liquidity_side", "level_price"],
        kind="mergesort",
    ).reset_index(drop=True)
    levels.insert(0, "level_id", np.arange(1, len(levels) + 1, dtype=np.int64))
    if (pd.to_datetime(levels["initial_available_time"]) <= pd.to_datetime(levels["pivot_bar_end_time"])).any():
        raise RuntimeError("HTF pivot became available before its right-confirmation bar closed")
    return levels


def _confirmed_order_before_bar(level: object, bar_start_time: pd.Timestamp, orders: tuple[int, ...]) -> int:
    confirmed = 0
    for order in orders:
        value = getattr(level, f"order_{order}_available_time", pd.NaT)
        if pd.notna(value) and pd.Timestamp(value) <= bar_start_time:
            confirmed = int(order)
    return confirmed


def build_sweep_episodes(
    primary: pd.DataFrame,
    levels: pd.DataFrame,
    *,
    confirmation_orders: Iterable[int] = (1, 2, 3, 5),
    sweep_epsilon_bp: float = 0.01,
    bar_minutes: int = 1,
    show_progress: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find each level's first causal sweep and collapse same-bar levels to one episode."""

    bars = normalize_bars(primary)
    if levels.empty:
        return pd.DataFrame(), pd.DataFrame()
    orders = tuple(sorted(set(int(v) for v in confirmation_orders)))
    if int(bar_minutes) < 1:
        raise ValueError("bar_minutes must be >= 1")
    bar_delta = pd.Timedelta(minutes=int(bar_minutes))
    idx = pd.DatetimeIndex(bars.index)
    idx_ns = idx.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    low = bars["low"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low_index = SegmentThresholdIndex(low)
    high_index = SegmentThresholdIndex(high)
    n = len(bars)
    reporter = ProgressReporter(
        label="[ict-mss] HTF liquidity first sweeps",
        total=len(levels),
        every=max(1, len(levels) // 200),
        enabled=bool(show_progress),
    )
    rows: list[dict[str, object]] = []
    for done, level in enumerate(levels.itertuples(index=False), start=1):
        available = pd.Timestamp(level.initial_available_time)
        active_pos = int(np.searchsorted(idx_ns, available.value, side="left"))
        if active_pos >= n:
            reporter.update(done)
            continue
        price = float(level.level_price)
        if str(level.liquidity_side) == "sell_side":
            threshold = price * (1.0 - float(sweep_epsilon_bp) / 10_000.0)
            sweep_pos = low_index.first_leq(active_pos, n - 1, threshold)
            extreme = low[sweep_pos] if sweep_pos >= 0 else np.nan
            depth_bp = (price - extreme) / price * 10_000.0 if sweep_pos >= 0 else np.nan
        else:
            threshold = price * (1.0 + float(sweep_epsilon_bp) / 10_000.0)
            sweep_pos = high_index.first_geq(active_pos, n - 1, threshold)
            extreme = high[sweep_pos] if sweep_pos >= 0 else np.nan
            depth_bp = (extreme - price) / price * 10_000.0 if sweep_pos >= 0 else np.nan
        if sweep_pos < 0:
            reporter.update(done)
            continue
        sweep_bar_time = idx[sweep_pos]
        confirmed = _confirmed_order_before_bar(level, sweep_bar_time, orders)
        if confirmed < 1:
            # Defensive: initial order-1 availability must predate the swept bar.
            reporter.update(done)
            continue
        rows.append(
            {
                **level._asdict(),
                "active_pos": active_pos,
                "sweep_pos": int(sweep_pos),
                "sweep_bar_time": sweep_bar_time,
                "sweep_available_time": sweep_bar_time + bar_delta,
                "confirmed_order_at_sweep": int(confirmed),
                "confirmed_order_available_time": pd.Timestamp(
                    getattr(level, f"order_{confirmed}_available_time")
                ),
                "confirmed_order_age_minutes_at_sweep": float(
                    (
                        sweep_bar_time
                        - pd.Timestamp(getattr(level, f"order_{confirmed}_available_time"))
                    ).total_seconds()
                    / 60.0
                ),
                "confirmed_prominence_bp_at_sweep": float(
                    getattr(level, f"order_{confirmed}_prominence_bp", np.nan)
                ),
                "level_age_minutes_at_sweep": float((sweep_bar_time - available).total_seconds() / 60.0),
                "level_sweep_depth_bp": float(depth_bp),
            }
        )
        reporter.update(done)
    reporter.close()
    level_sweeps = pd.DataFrame(rows)
    if level_sweeps.empty:
        return level_sweeps, pd.DataFrame()

    episode_rows: list[dict[str, object]] = []
    grouped = level_sweeps.groupby(["liquidity_side", "sweep_pos"], sort=True, observed=False)
    for (liq_side, sweep_pos), part in grouped:
        tf_values = sorted(set(int(v) for v in part["source_timeframe_min"]))
        tf_names = [tf for _, tf in sorted(zip(part["source_timeframe_min"], part["source_timeframe"], strict=False))]
        side = 1 if liq_side == "sell_side" else -1
        pos = int(sweep_pos)
        episode_rows.append(
            {
                "sweep_id": f"{'SSL' if side == 1 else 'BSL'}_{pos:09d}",
                "side": side,
                "liquidity_side": str(liq_side),
                "sweep_pos": pos,
                "sweep_bar_time": idx[pos],
                "sweep_available_time": idx[pos] + bar_delta,
                "sweep_extreme": float(low[pos] if side == 1 else high[pos]),
                "swept_level_count": int(len(part)),
                "swept_timeframe_count": int(len(tf_values)),
                "max_timeframe_min": int(max(tf_values)),
                "max_confirmed_order": int(part["confirmed_order_at_sweep"].max()),
                "max_confirmed_prominence_bp": float(part["confirmed_prominence_bp_at_sweep"].max()),
                "max_pivot_range_bp": float(part["pivot_range_bp"].max()),
                "max_pivot_rejection_fraction": float(part["pivot_rejection_fraction"].max()),
                "max_level_sweep_depth_bp": float(part["level_sweep_depth_bp"].max()),
                "median_level_age_minutes": float(part["level_age_minutes_at_sweep"].median()),
                "oldest_level_age_minutes": float(part["level_age_minutes_at_sweep"].max()),
                "swept_timeframes": "|".join(dict.fromkeys(tf_names)),
                "swept_15m": bool((part["source_timeframe_min"] == 15).any()),
                "swept_30m": bool((part["source_timeframe_min"] == 30).any()),
                "swept_1h": bool((part["source_timeframe_min"] == 60).any()),
                "swept_4h": bool((part["source_timeframe_min"] == 240).any()),
                "level_ids": "|".join(str(int(v)) for v in part["level_id"].tolist()),
            }
        )
    episodes = pd.DataFrame(episode_rows).sort_values(["sweep_pos", "side"], kind="mergesort").reset_index(drop=True)
    return level_sweeps, episodes


@dataclass(frozen=True)
class MicroStructureContext:
    order: int
    last_high_level: np.ndarray
    last_high_pivot_pos: np.ndarray
    last_low_level: np.ndarray
    last_low_pivot_pos: np.ndarray


def _last_confirmed_pivot_arrays(values: np.ndarray, order: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(values)
    mask = _pivot_mask(values, int(order), kind)
    pivot_positions = np.flatnonzero(mask)
    activations = pivot_positions + int(order) + 1
    valid = activations < n
    pivot_positions = pivot_positions[valid]
    activations = activations[valid]
    pivot_pos = np.full(n, -1, dtype=np.int32)
    if len(activations):
        event_pivot = np.full(n, -1, dtype=np.int32)
        event_pivot[activations.astype(np.int64)] = pivot_positions.astype(np.int32)
        # Pivot positions are strictly increasing, so maximum-accumulate is an
        # O(n) vectorized causal forward-fill of the latest confirmed pivot.
        pivot_pos = np.maximum.accumulate(event_pivot)
    level = np.full(n, np.nan, dtype=float)
    valid_latest = pivot_pos >= 0
    if np.any(valid_latest):
        level[valid_latest] = values[pivot_pos[valid_latest].astype(np.int64)]
    return level, pivot_pos


def build_micro_structure_context(primary: pd.DataFrame, orders: Iterable[int] = (2, 3, 5)) -> dict[int, MicroStructureContext]:
    bars = normalize_bars(primary)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    out: dict[int, MicroStructureContext] = {}
    for order in sorted(set(int(v) for v in orders)):
        high_level, high_pos = _last_confirmed_pivot_arrays(high, order, "high")
        low_level, low_pos = _last_confirmed_pivot_arrays(low, order, "low")
        out[order] = MicroStructureContext(
            order=order,
            last_high_level=high_level,
            last_high_pivot_pos=high_pos,
            last_low_level=low_level,
            last_low_pivot_pos=low_pos,
        )
    return out


def build_displacement_fvgs(
    primary: pd.DataFrame,
    *,
    rolling_window: int = 60,
    bar_minutes: int = 1,
) -> pd.DataFrame:
    """Build all three-candle FVGs and causal displacement diagnostics.

    For completion position ``i`` the displacement candle is ``i-1`` and the
    first candle is ``i-2``.  The FVG is not actionable until bar ``i`` closes;
    therefore a limit order may only start on ``i+1``.
    """

    bars = normalize_bars(primary)
    if int(bar_minutes) < 1:
        raise ValueError("bar_minutes must be >= 1")
    bar_delta = pd.Timedelta(minutes=int(bar_minutes))
    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    body = np.abs(close - open_)
    rng = np.maximum(high - low, EPS)
    body_s = pd.Series(body, index=bars.index)
    range_s = pd.Series(rng, index=bars.index)
    min_periods = max(10, int(rolling_window) // 3)
    prior_body = body_s.shift(1).rolling(int(rolling_window), min_periods=min_periods).median().to_numpy(dtype=float)
    prior_range = range_s.shift(1).rolling(int(rolling_window), min_periods=min_periods).median().to_numpy(dtype=float)

    completion = np.arange(2, len(bars), dtype=np.int64)
    middle = completion - 1
    first = completion - 2
    bullish_fvg = low[completion] > high[first]
    bearish_fvg = high[completion] < low[first]
    valid = bullish_fvg | bearish_fvg
    completion = completion[valid]
    middle = middle[valid]
    first = first[valid]
    if not len(completion):
        return pd.DataFrame()
    side = np.where(low[completion] > high[first], 1, -1).astype(np.int8)
    fvg_near = np.where(side == 1, low[completion], high[completion])
    fvg_far = np.where(side == 1, high[first], low[first])
    fvg_size = np.abs(fvg_near - fvg_far)
    middle_range = rng[middle]
    close_from_extreme = np.where(side == 1, high[middle] - close[middle], close[middle] - low[middle])
    direction_ok = np.where(side == 1, close[middle] > open_[middle], close[middle] < open_[middle])
    out = pd.DataFrame(
        {
            "side": side,
            "fvg_first_pos": first,
            "displacement_pos": middle,
            "fvg_completion_pos": completion,
            "fvg_completion_time": bars.index[completion],
            "fvg_available_time": bars.index[completion] + bar_delta,
            "fvg_near_price": fvg_near,
            "fvg_far_price": fvg_far,
            "fvg_size_bp": _safe_ratio(fvg_size, close[middle]) * 10_000.0,
            "displacement_direction_ok": direction_ok,
            "displacement_body_vs_past_median": _safe_ratio(body[middle], prior_body[middle]),
            "displacement_range_vs_past_median": _safe_ratio(rng[middle], prior_range[middle]),
            "displacement_body_fraction": _safe_ratio(body[middle], middle_range),
            "displacement_close_from_extreme_fraction": _safe_ratio(close_from_extreme, middle_range),
            "displacement_open": open_[middle],
            "displacement_close": close[middle],
            "displacement_high": high[middle],
            "displacement_low": low[middle],
        }
    )
    return out.loc[out["displacement_direction_ok"]].sort_values(["fvg_completion_pos", "side"], kind="mergesort").reset_index(drop=True)


def _candidate_slice_positions(values: np.ndarray, start: int, end: int) -> tuple[int, int]:
    left = int(np.searchsorted(values, int(start), side="left"))
    right = int(np.searchsorted(values, int(end), side="right"))
    return left, right


def pair_sweeps_with_mss_fvgs(
    primary: pd.DataFrame,
    sweep_episodes: pd.DataFrame,
    fvgs: pd.DataFrame,
    micro_context: dict[int, MicroStructureContext],
    *,
    max_search_bars: int = 180,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Pair HTF liquidity sweeps with the first causal MSS+FVG for each context.

    Two predeclared structural interpretations are retained for research:
    ``pre_sweep`` freezes the latest confirmed micro swing before the liquidity
    sweep; ``rolling`` lets a new micro swing become confirmed after the sweep
    before displacement breaks it.  Neither path can use an unconfirmed pivot.
    """

    bars = normalize_bars(primary)
    if sweep_episodes.empty or fvgs.empty:
        return pd.DataFrame()
    close = bars["close"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    fvg_by_side = {
        side: part.sort_values("fvg_completion_pos", kind="mergesort").reset_index(drop=True)
        for side, part in fvgs.groupby("side", sort=False)
    }
    positions_by_side = {side: part["fvg_completion_pos"].to_numpy(dtype=np.int64) for side, part in fvg_by_side.items()}
    contexts = sorted(micro_context)
    total = len(sweep_episodes) * len(contexts) * 2
    reporter = ProgressReporter(
        label="[ict-mss] sweep -> MSS/FVG pairing",
        total=total,
        every=max(1, total // 200),
        enabled=bool(show_progress),
    )
    done = 0
    rows: list[dict[str, object]] = []
    n = len(bars)
    for sweep in sweep_episodes.itertuples(index=False):
        side = int(sweep.side)
        if side not in fvg_by_side:
            done += len(contexts) * 2
            reporter.update(done)
            continue
        part = fvg_by_side[side]
        fvg_positions = positions_by_side[side]
        sweep_pos = int(sweep.sweep_pos)
        # FVG completion must occur after the sweep bar; allowing a same-bar
        # completion would require intrabar ordering that 1m OHLC cannot prove.
        left, right = _candidate_slice_positions(fvg_positions, sweep_pos + 1, min(n - 1, sweep_pos + int(max_search_bars) + 1))
        candidates = part.iloc[left:right]
        for order in contexts:
            ctx = micro_context[order]
            for mode in ("pre_sweep", "rolling"):
                chosen: dict[str, object] | None = None
                if not candidates.empty:
                    for fvg in candidates.itertuples(index=False):
                        m = int(fvg.displacement_pos)
                        if m <= sweep_pos or m <= 0:
                            continue
                        if side == 1:
                            if mode == "pre_sweep":
                                structure_level = float(ctx.last_high_level[sweep_pos])
                                structure_pos = int(ctx.last_high_pivot_pos[sweep_pos])
                                structure_available_pos = int(structure_pos + int(order) + 1)
                            else:
                                structure_level = float(ctx.last_high_level[m])
                                structure_pos = int(ctx.last_high_pivot_pos[m])
                                structure_available_pos = int(structure_pos + int(order) + 1)
                            if not np.isfinite(structure_level) or structure_pos < 0 or structure_available_pos < 0:
                                continue
                            # Pivot must already be known before the displacement
                            # bar opens, and the displacement candle must close
                            # through it from the other side.
                            if structure_available_pos > m:
                                continue
                            if not (close[m - 1] <= structure_level and close[m] > structure_level):
                                continue
                            break_margin_bp = (close[m] / structure_level - 1.0) * 10_000.0
                            sweep_extreme = float(np.min(low[sweep_pos : int(fvg.fvg_completion_pos) + 1]))
                        else:
                            if mode == "pre_sweep":
                                structure_level = float(ctx.last_low_level[sweep_pos])
                                structure_pos = int(ctx.last_low_pivot_pos[sweep_pos])
                                structure_available_pos = int(structure_pos + int(order) + 1)
                            else:
                                structure_level = float(ctx.last_low_level[m])
                                structure_pos = int(ctx.last_low_pivot_pos[m])
                                structure_available_pos = int(structure_pos + int(order) + 1)
                            if not np.isfinite(structure_level) or structure_pos < 0 or structure_available_pos < 0:
                                continue
                            if structure_available_pos > m:
                                continue
                            if not (close[m - 1] >= structure_level and close[m] < structure_level):
                                continue
                            break_margin_bp = (1.0 - close[m] / structure_level) * 10_000.0
                            sweep_extreme = float(np.max(high[sweep_pos : int(fvg.fvg_completion_pos) + 1]))
                        chosen = {
                            **sweep._asdict(),
                            **fvg._asdict(),
                            "micro_order": int(order),
                            "structure_mode": mode,
                            "micro_structure_level": float(structure_level),
                            "micro_structure_pivot_pos": int(structure_pos),
                            "micro_structure_available_pos": int(structure_available_pos),
                            "micro_structure_pivot_time": bars.index[int(structure_pos)],
                            "micro_structure_available_time": bars.index[int(structure_available_pos)],
                            "mss_break_margin_bp": float(break_margin_bp),
                            "sweep_to_displacement_bars": int(m - sweep_pos),
                            "sweep_to_fvg_completion_bars": int(fvg.fvg_completion_pos - sweep_pos),
                            "stop_extreme": sweep_extreme,
                        }
                        break
                if chosen is not None:
                    rows.append(chosen)
                done += 1
                reporter.update(done)
    reporter.close()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["setup_key"] = (
        out["side"].astype(str)
        + "_"
        + out["micro_order"].astype(str)
        + "_"
        + out["structure_mode"].astype(str)
        + "_"
        + out["fvg_completion_pos"].astype(str)
    )
    return out.sort_values(["fvg_completion_pos", "side", "micro_order", "structure_mode", "sweep_pos"], kind="mergesort").reset_index(drop=True)
