#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal multi-bar detector for this panic-recovery research line.

The implementation lives beside the research that owns the event definition.
It is reused by the numbered research scripts and ``analyze_tool``.
It does not define an event from a single candle.  Instead it walks forward as a
small state machine:

1. multi-bar sell pressure starts an observation episode;
2. price/volume pressure expands into panic acceleration;
3. the market stops making lows and selling pressure decays;
4. price reclaims part of the panic range and produces a causal recovery signal.

All state transitions use only the current bar and earlier bars.  Forward
returns attached to a completed signal are diagnostics only and never feed back
into signal generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanicEpisodeConfig:
    baseline_window: int = 60
    selloff_window: int = 5
    min_red_bars: int = 3
    observe_drop_pct: float = 0.0045
    observe_drop_vol_mult: float = 2.5
    observe_volume_ratio: float = 1.10
    panic_drop_pct: float = 0.0075
    panic_volume_ratio: float = 1.35
    stabilization_bars: int = 2
    min_rebound_from_low_pct: float = 0.0020
    pressure_decay_ratio: float = 0.68
    reclaim_fraction: float = 0.35
    breakout_lookback: int = 2
    max_episode_bars: int = 30
    cooldown_bars: int = 8
    outcome_horizons: tuple[int, ...] = (5, 15, 30)

    def validated(self) -> "PanicEpisodeConfig":
        if self.baseline_window < 10:
            raise ValueError("baseline_window must be >= 10")
        if self.selloff_window < 2:
            raise ValueError("selloff_window must be >= 2")
        if self.min_red_bars < 1 or self.min_red_bars > self.selloff_window:
            raise ValueError("min_red_bars must be within [1, selloff_window]")
        if self.observe_drop_pct <= 0 or self.panic_drop_pct <= 0:
            raise ValueError("drop thresholds must be > 0")
        if self.panic_drop_pct < self.observe_drop_pct:
            raise ValueError("panic_drop_pct must be >= observe_drop_pct")
        if self.observe_drop_vol_mult <= 0:
            raise ValueError("observe_drop_vol_mult must be > 0")
        if self.observe_volume_ratio <= 0 or self.panic_volume_ratio <= 0:
            raise ValueError("volume ratios must be > 0")
        if self.stabilization_bars < 1:
            raise ValueError("stabilization_bars must be >= 1")
        if not 0 < self.pressure_decay_ratio <= 1:
            raise ValueError("pressure_decay_ratio must be within (0, 1]")
        if not 0 < self.reclaim_fraction < 1:
            raise ValueError("reclaim_fraction must be within (0, 1)")
        if self.breakout_lookback < 1:
            raise ValueError("breakout_lookback must be >= 1")
        if self.max_episode_bars < self.stabilization_bars + 2:
            raise ValueError("max_episode_bars is too small")
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must be >= 0")
        return self


@dataclass(frozen=True)
class PanicNode:
    timestamp: pd.Timestamp
    kind: str
    label: str
    price: float | None = None
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PanicEpisode:
    episode_id: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    status: str
    nodes: tuple[PanicNode, ...]
    reference_price: float
    episode_low: float
    signal_time: pd.Timestamp | None = None
    signal_price: float | None = None
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PanicEpisodeResult:
    episodes: tuple[PanicEpisode, ...]
    feature_frame: pd.DataFrame

    @property
    def signal_count(self) -> int:
        return sum(ep.signal_time is not None for ep in self.episodes)


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _prepare_features(df: pd.DataFrame, cfg: PanicEpisodeConfig) -> pd.DataFrame:
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"panic episode detector missing columns: {sorted(missing)}")

    out = df.copy().sort_index()
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")]
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        return out

    prev_close = out["close"].shift(1)
    out["ret_1"] = out["close"].pct_change()
    out["selloff_return"] = out["close"] / out["close"].shift(cfg.selloff_window) - 1.0
    out["red_bar"] = out["close"] < prev_close
    out["red_count"] = out["red_bar"].rolling(cfg.selloff_window, min_periods=cfg.selloff_window).sum()

    abs_ret_base = out["ret_1"].abs().shift(1).rolling(
        cfg.baseline_window,
        min_periods=max(10, cfg.baseline_window // 3),
    ).median()
    abs_ret_floor = out["close"].abs() * 1e-8
    out["abs_ret_base"] = abs_ret_base.clip(lower=abs_ret_floor)
    out["drop_vol_mult"] = (-out["selloff_return"]) / (
        out["abs_ret_base"] * np.sqrt(float(cfg.selloff_window))
    )

    vol_base = out["volume"].shift(1).rolling(
        cfg.baseline_window,
        min_periods=max(10, cfg.baseline_window // 3),
    ).median()
    out["volume_ratio"] = _safe_divide(out["volume"], vol_base)
    window_volume = out["volume"].rolling(cfg.selloff_window, min_periods=cfg.selloff_window).sum()
    window_volume_base = vol_base * cfg.selloff_window
    out["window_volume_ratio"] = _safe_divide(window_volume, window_volume_base)

    bar_range = (out["high"] - out["low"]).clip(lower=out["close"].abs() * 1e-9)
    out["close_pos"] = ((out["close"] - out["low"]) / bar_range).clip(0.0, 1.0)
    out["lower_wick_frac"] = (
        (out[["open", "close"]].min(axis=1) - out["low"]).clip(lower=0.0) / bar_range
    ).clip(0.0, 1.0)

    flow_pressure = pd.Series(np.nan, index=out.index, dtype=float)
    if {"delta_notional", "notional"}.issubset(out.columns):
        flow_pressure = _safe_divide(out["delta_notional"], out["notional"])
    elif {"delta_volume", "volume"}.issubset(out.columns):
        flow_pressure = _safe_divide(out["delta_volume"], out["volume"])
    out["flow_pressure"] = flow_pressure.clip(-1.0, 1.0)
    out["flow_pressure_2"] = out["flow_pressure"].rolling(2, min_periods=1).mean()

    speed = (-out["ret_1"] / out["abs_ret_base"]).clip(lower=0.0, upper=20.0)
    cumulative = out["drop_vol_mult"].clip(lower=0.0, upper=20.0)
    volume = (out["volume_ratio"].fillna(1.0) - 1.0).clip(lower=0.0, upper=5.0)
    flow = (-out["flow_pressure"].fillna(0.0)).clip(lower=0.0, upper=1.0)
    out["sell_pressure_score"] = speed + 0.8 * cumulative + 1.2 * volume + 2.0 * flow
    return out


def _float(value: Any, default: float = np.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _outcome_fields(features: pd.DataFrame, signal_pos: int, signal_price: float, horizons: tuple[int, ...]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    n = len(features)
    for horizon in horizons:
        if horizon <= 0:
            continue
        end = signal_pos + horizon
        key = str(int(horizon))
        if end >= n:
            fields[f"outcome_return_{key}b"] = None
            fields[f"outcome_mfe_{key}b"] = None
            fields[f"outcome_mae_{key}b"] = None
            continue
        window = features.iloc[signal_pos + 1 : end + 1]
        fields[f"outcome_return_{key}b"] = float(features["close"].iloc[end] / signal_price - 1.0)
        fields[f"outcome_mfe_{key}b"] = float(window["high"].max() / signal_price - 1.0)
        fields[f"outcome_mae_{key}b"] = float(window["low"].min() / signal_price - 1.0)
    return fields


def detect_panic_episodes(df: pd.DataFrame, config: PanicEpisodeConfig | None = None) -> PanicEpisodeResult:
    """Detect panic episodes with a causal forward state machine.

    Performance note: observation candidates are identified vectorially first.
    The Python state machine then scans only the short candidate episodes rather
    than every bar in the full history. This is logically equivalent to the
    previous full-frame loop because no new episode can start while another is
    active or during cooldown.
    """
    cfg = (config or PanicEpisodeConfig()).validated()
    features = _prepare_features(df, cfg)
    if features.empty:
        return PanicEpisodeResult(episodes=tuple(), feature_frame=features)

    index = features.index
    lows = features["low"].to_numpy(dtype=float)
    highs = features["high"].to_numpy(dtype=float)
    closes = features["close"].to_numpy(dtype=float)
    selloff_return = features["selloff_return"].to_numpy(dtype=float)
    drop_vol_mult = features["drop_vol_mult"].to_numpy(dtype=float)
    red_count = features["red_count"].to_numpy(dtype=float)
    window_volume_ratio = features["window_volume_ratio"].to_numpy(dtype=float)
    volume_ratio = features["volume_ratio"].to_numpy(dtype=float)
    flow_pressure = features["flow_pressure"].to_numpy(dtype=float)
    flow_pressure_2 = features["flow_pressure_2"].to_numpy(dtype=float)
    sell_pressure_score = features["sell_pressure_score"].to_numpy(dtype=float)
    close_pos = features["close_pos"].to_numpy(dtype=float)
    lower_wick_frac = features["lower_wick_frac"].to_numpy(dtype=float)
    ret_1 = features["ret_1"].to_numpy(dtype=float)

    observe_mask = (
        (selloff_return <= -cfg.observe_drop_pct)
        & (drop_vol_mult >= cfg.observe_drop_vol_mult)
        & (red_count >= cfg.min_red_bars)
        & (window_volume_ratio >= cfg.observe_volume_ratio)
    )
    warmup_pos = max(cfg.baseline_window // 3, cfg.selloff_window)
    if warmup_pos > 0:
        observe_mask[:warmup_pos] = False
    candidates = np.flatnonzero(observe_mask)

    episodes: list[PanicEpisode] = []
    cooldown_until = -1
    episode_id = 0
    candidate_cursor = 0
    n = len(features)

    def finite(value: float, default: float) -> float:
        return float(value) if np.isfinite(value) else float(default)

    def upsert_low_candidate(active: dict[str, Any]) -> None:
        """Keep the purple marker on the latest low known by the state machine.

        The marker is visual/diagnostic, not an entry signal. Every replacement
        occurs only after that low has printed on a closed bar, so no future data
        is introduced.
        """
        low_pos = int(active["low_pos"])
        node = PanicNode(
            pd.Timestamp(index[low_pos]),
            "low_candidate",
            "恐慌低点候选",
            price=float(active["low"]),
            fields={
                "episode_id": active["id"],
                "episode_low": float(active["low"]),
                "known_at_or_before_pos": low_pos,
                "note": "候选低点；若后续创新低，紫灯会因果地移动到新低",
            },
        )
        for i, existing in enumerate(active["nodes"]):
            if existing.kind == "low_candidate":
                active["nodes"][i] = node
                return
        active["nodes"].append(node)

    while candidate_cursor < len(candidates):
        candidate_cursor = int(np.searchsorted(candidates, cooldown_until + 1, side="left"))
        if candidate_cursor >= len(candidates):
            break
        start_pos = int(candidates[candidate_cursor])
        start_ts = pd.Timestamp(index[start_pos])
        ref_pos = max(0, start_pos - cfg.selloff_window)
        reference_price = float(closes[ref_pos])
        episode_id += 1
        start_fields = {
            "episode_id": episode_id,
            "selloff_return": finite(selloff_return[start_pos], np.nan),
            "drop_vol_mult": finite(drop_vol_mult[start_pos], np.nan),
            "red_count": int(finite(red_count[start_pos], 0.0)),
            "window_volume_ratio": finite(window_volume_ratio[start_pos], np.nan),
            "flow_pressure": finite(flow_pressure[start_pos], np.nan),
            "reference_price": reference_price,
        }
        active: dict[str, Any] = {
            "id": episode_id,
            "start_pos": start_pos,
            "start_time": start_ts,
            "reference_price": reference_price,
            "low": float(lows[start_pos]),
            "low_pos": start_pos,
            "panic": False,
            "exhaustion": False,
            "worst_pressure": max(0.0, finite(sell_pressure_score[start_pos], 0.0)),
            "panic_max_volume_ratio": max(1.0, finite(volume_ratio[start_pos], 1.0)),
            "panic_flow_pressure": finite(flow_pressure[start_pos], np.nan),
            "nodes": [
                PanicNode(
                    start_ts,
                    "start",
                    "开始观察",
                    price=float(highs[start_pos]),
                    fields=start_fields,
                )
            ],
        }

        completed = False
        final_pos = min(n - 1, start_pos + cfg.max_episode_bars)
        for pos in range(start_pos + 1, final_pos + 1):
            ts = pd.Timestamp(index[pos])
            elapsed = pos - start_pos
            current_low = float(lows[pos])
            made_new_low = current_low < float(active["low"])
            if made_new_low:
                active["low"] = current_low
                active["low_pos"] = pos
                if active["panic"]:
                    upsert_low_candidate(active)

            pressure = max(0.0, finite(sell_pressure_score[pos], 0.0))
            active["worst_pressure"] = max(float(active["worst_pressure"]), pressure)
            active["panic_max_volume_ratio"] = max(
                float(active["panic_max_volume_ratio"]),
                max(1.0, finite(volume_ratio[pos], 1.0)),
            )
            drop_from_reference = float(active["low"]) / reference_price - 1.0

            if not active["panic"]:
                panic = (
                    drop_from_reference <= -cfg.panic_drop_pct
                    and (
                        finite(volume_ratio[pos], 0.0) >= cfg.panic_volume_ratio
                        or finite(window_volume_ratio[pos], 0.0) >= cfg.observe_volume_ratio
                    )
                )
                if panic:
                    active["panic"] = True
                    active["panic_flow_pressure"] = finite(flow_pressure[pos], np.nan)
                    active["nodes"].append(
                        PanicNode(
                            ts,
                            "acceleration",
                            "卖压加速",
                            price=float(lows[pos]),
                            fields={
                                "episode_id": active["id"],
                                "drop_from_reference": drop_from_reference,
                                "volume_ratio": finite(volume_ratio[pos], np.nan),
                                "window_volume_ratio": finite(window_volume_ratio[pos], np.nan),
                                "sell_pressure_score": pressure,
                                "flow_pressure": finite(flow_pressure[pos], np.nan),
                            },
                        )
                    )
                    upsert_low_candidate(active)

            if active["panic"] and not active["exhaustion"]:
                bars_since_low = pos - int(active["low_pos"])
                rebound = float(closes[pos]) / float(active["low"]) - 1.0
                pressure_decayed = pressure <= max(
                    0.25,
                    float(active["worst_pressure"]) * cfg.pressure_decay_ratio,
                )
                flow_now = finite(flow_pressure_2[pos], np.nan)
                flow_then = finite(active.get("panic_flow_pressure", np.nan), np.nan)
                flow_improved = np.isfinite(flow_now) and (
                    not np.isfinite(flow_then) or flow_now >= flow_then + 0.08
                )
                rejection_shape = (
                    finite(close_pos[pos], 0.0) >= 0.58
                    or finite(lower_wick_frac[pos], 0.0) >= 0.28
                    or finite(ret_1[pos], -1.0) > 0.0
                )
                volume_cooled = finite(volume_ratio[pos], 1.0) <= float(active["panic_max_volume_ratio"]) * 0.92
                exhaustion = (
                    bars_since_low >= cfg.stabilization_bars
                    and rebound >= cfg.min_rebound_from_low_pct
                    and rejection_shape
                    and (pressure_decayed or flow_improved or volume_cooled)
                )
                if exhaustion:
                    active["exhaustion"] = True
                    active["exhaustion_pos"] = pos
                    active["nodes"].append(
                        PanicNode(
                            ts,
                            "exhaustion",
                            "卖压衰减 / 拒绝",
                            price=float(lows[pos]),
                            fields={
                                "episode_id": active["id"],
                                "bars_since_low": bars_since_low,
                                "rebound_from_low": rebound,
                                "pressure_ratio": pressure / max(float(active["worst_pressure"]), 1e-9),
                                "flow_pressure_2": flow_now,
                                "close_pos": finite(close_pos[pos], np.nan),
                                "lower_wick_frac": finite(lower_wick_frac[pos], np.nan),
                            },
                        )
                    )

            if active["panic"] and active["exhaustion"]:
                low = float(active["low"])
                reclaim_level = low + cfg.reclaim_fraction * max(reference_price - low, 0.0)
                left = max(0, pos - cfg.breakout_lookback)
                prior_high = float(np.max(highs[left:pos])) if pos > left else np.nan
                momentum_ok = finite(ret_1[pos], 0.0) > 0 and closes[pos] > closes[pos - 1]
                reclaim_ok = closes[pos] >= reclaim_level
                breakout_ok = np.isfinite(prior_high) and closes[pos] >= prior_high
                signal = momentum_ok and (reclaim_ok or breakout_ok)
                if signal:
                    signal_price = float(closes[pos])
                    outcome = _outcome_fields(features, pos, signal_price, cfg.outcome_horizons)
                    node_fields = {
                        "episode_id": active["id"],
                        "signal_is_causal": True,
                        "signal_price": signal_price,
                        "episode_low": low,
                        "reference_price": reference_price,
                        "reclaim_level": reclaim_level,
                        "reclaim_fraction": cfg.reclaim_fraction,
                        "breakout_level": prior_high,
                        "bars_from_start": elapsed,
                        "bars_from_low": pos - int(active["low_pos"]),
                        **outcome,
                    }
                    active["nodes"].append(
                        PanicNode(
                            ts,
                            "signal",
                            "恢复确认 · 做多观察",
                            price=signal_price,
                            fields=node_fields,
                        )
                    )
                    episode_fields = {
                        "bars": elapsed + 1,
                        "drop_from_reference": low / reference_price - 1.0,
                        "recovery_to_signal": signal_price / low - 1.0,
                        **outcome,
                    }
                    episodes.append(
                        PanicEpisode(
                            episode_id=int(active["id"]),
                            start_time=start_ts,
                            end_time=ts,
                            status="signal",
                            nodes=tuple(active["nodes"]),
                            reference_price=reference_price,
                            episode_low=low,
                            signal_time=ts,
                            signal_price=signal_price,
                            fields=episode_fields,
                        )
                    )
                    cooldown_until = pos + cfg.cooldown_bars
                    completed = True
                    break

            if elapsed >= cfg.max_episode_bars:
                status = "timeout_after_panic" if active["panic"] else "observe_failed"
                episodes.append(
                    PanicEpisode(
                        episode_id=int(active["id"]),
                        start_time=start_ts,
                        end_time=ts,
                        status=status,
                        nodes=tuple(active["nodes"]),
                        reference_price=reference_price,
                        episode_low=float(active["low"]),
                        fields={
                            "bars": elapsed + 1,
                            "drop_from_reference": float(active["low"]) / reference_price - 1.0,
                        },
                    )
                )
                cooldown_until = pos + cfg.cooldown_bars
                completed = True
                break

        if not completed:
            last_pos = n - 1
            last_ts = pd.Timestamp(index[last_pos])
            episodes.append(
                PanicEpisode(
                    episode_id=int(active["id"]),
                    start_time=start_ts,
                    end_time=last_ts,
                    status="incomplete",
                    nodes=tuple(active["nodes"]),
                    reference_price=reference_price,
                    episode_low=float(active["low"]),
                    fields={
                        "bars": n - start_pos,
                        "drop_from_reference": float(active["low"]) / reference_price - 1.0,
                    },
                )
            )
            break

        candidate_cursor = int(np.searchsorted(candidates, cooldown_until + 1, side="left"))

    return PanicEpisodeResult(episodes=tuple(episodes), feature_frame=features)

