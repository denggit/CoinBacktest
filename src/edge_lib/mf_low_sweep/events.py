#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Event construction for ETH MF Low Sweep."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader


def parse_number_list(raw: str, *, cast: Callable[[str], Any], name: str, allow_zero: bool = False) -> list[Any]:
    out: list[Any] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = cast(part)
        if not allow_zero and float(value) <= 0:
            raise ValueError(f"{name} must be positive: {value}")
        out.append(value)
    if not out:
        raise ValueError(f"{name} is empty")
    return sorted(set(out))


def parse_variant_list(raw: str) -> list[str]:
    variants = [p.strip() for p in str(raw).split(",") if p.strip()]
    allowed = {"fade_close_through", "reject", "wick"}
    bad = [v for v in variants if v not in allowed]
    if bad:
        raise ValueError(f"Unknown low-sweep variants: {bad}")
    return variants


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return pd.to_numeric(tr, errors="coerce")


def load_trade_bars(args: Any) -> pd.DataFrame:
    print(f"[mf] load 1m trade bars {args.symbol} {args.warmup_start_date}->{args.end_date}", flush=True)
    df = OKXTradeBarLoader(symbol=args.symbol, timeframe=args.timeframe).fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
    )
    if df.empty:
        raise RuntimeError(f"No trade_bar data loaded for {args.symbol} {args.timeframe}")
    out = df.copy().sort_index()
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"Loaded trade bars missing required columns: {missing}")
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    print(f"[mf] 1m rows={len(out):,} range={out.index[0]} -> {out.index[-1]}", flush=True)
    return out


def confirmed_swing_lows(df: pd.DataFrame, *, left: int, right: int) -> pd.DataFrame:
    low = pd.to_numeric(df["low"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    pos = pd.Series(np.arange(len(df), dtype=float), index=df.index)

    left_low = low.shift(1).rolling(left, min_periods=left).min()
    right_low = low.iloc[::-1].shift(1).rolling(right, min_periods=right).min().iloc[::-1]
    pivot_low = (low < left_low) & (low <= right_low)

    window = left + right + 1
    local_high = high.rolling(window, center=True, min_periods=window).max()
    low_prom = safe_divide(local_high, low) - 1.0

    swing_low = low.where(pivot_low).shift(right).shift(1).ffill()
    swing_low_pos = pos.where(pivot_low).shift(right).shift(1).ffill()
    swing_low_prom = low_prom.where(pivot_low).shift(right).shift(1).ffill()

    return pd.DataFrame(
        {
            "swing_low": swing_low,
            "swing_low_pos": swing_low_pos,
            "swing_low_age": pos - swing_low_pos,
            "swing_low_prominence_pct": swing_low_prom,
        },
        index=df.index,
    )


def build_features(bars: pd.DataFrame, args: Any) -> pd.DataFrame:
    df = bars.copy().sort_index()
    df["prev_close"] = df["close"].shift(1)
    df["ret_close_to_close"] = df["close"].pct_change()
    df["bar_body_ret"] = df["close"] / df["open"] - 1.0
    df["down_spike_pct"] = df["prev_close"] / df["low"] - 1.0
    df["tr"] = true_range(df)
    df["atr"] = df["tr"].rolling(int(args.atr_window), min_periods=int(args.atr_window)).mean()
    df["atr_pct"] = safe_divide(df["atr"], df["close"])

    vol_min_periods = min(int(args.volume_window), max(10, int(args.volume_window) // 3))
    vol_base = df["volume"].shift(1).rolling(int(args.volume_window), min_periods=vol_min_periods).median()
    df["volume_ratio"] = safe_divide(df["volume"], vol_base)
    df["volume_spike"] = df["volume_ratio"] >= float(args.volume_spike_threshold)

    if "trades_count" in df.columns:
        trade_min_periods = min(int(args.volume_window), max(10, int(args.volume_window) // 3))
        trade_base = pd.to_numeric(df["trades_count"], errors="coerce").shift(1).rolling(
            int(args.volume_window),
            min_periods=trade_min_periods,
        ).median()
        df["trades_count_ratio"] = safe_divide(pd.to_numeric(df["trades_count"], errors="coerce"), trade_base)
    else:
        df["trades_count_ratio"] = np.nan

    if "delta_notional" in df.columns:
        df["delta_notional"] = pd.to_numeric(df["delta_notional"], errors="coerce")
        notional_base = (
            pd.to_numeric(df.get("buy_notional", np.nan), errors="coerce")
            + pd.to_numeric(df.get("sell_notional", np.nan), errors="coerce")
        ).replace(0.0, np.nan)
        df["delta_notional_ratio"] = df["delta_notional"] / notional_base
        cvd_min_periods = min(int(args.cvd_window), max(10, int(args.cvd_window) // 3))
        roll = df["delta_notional"].shift(1).rolling(int(args.cvd_window), min_periods=cvd_min_periods)
        df["delta_notional_z"] = (df["delta_notional"] - roll.mean()) / roll.std(ddof=0).replace(0.0, np.nan)
    else:
        df["delta_notional"] = np.nan
        df["delta_notional_ratio"] = np.nan
        df["delta_notional_z"] = np.nan

    if "cvd_notional" in df.columns:
        cvd = pd.to_numeric(df["cvd_notional"], errors="coerce")
        df["cvd_notional_change"] = cvd.diff(int(args.cvd_window))
    else:
        df["cvd_notional_change"] = np.nan

    df["taker_buy_ratio"] = pd.to_numeric(df["taker_buy_ratio"], errors="coerce") if "taker_buy_ratio" in df.columns else np.nan
    df["large_delta_notional"] = pd.to_numeric(df["large_delta_notional"], errors="coerce") if "large_delta_notional" in df.columns else np.nan

    bar_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    df["lower_wick_frac"] = safe_divide(df[["open", "close"]].min(axis=1) - df["low"], bar_range)
    df["close_pos_in_bar"] = safe_divide(df["close"] - df["low"], bar_range)

    swings = confirmed_swing_lows(df, left=int(args.pivot_left), right=int(args.pivot_right))
    out = pd.concat([df, swings], axis=1)
    out["session_hour"] = out.index.hour
    out["session_bucket"] = pd.cut(
        out["session_hour"],
        bins=[-1, 7, 15, 23],
        labels=["S0_00_07", "S1_08_15", "S2_16_23"],
    ).astype("object").fillna("NA")
    out["weekday"] = out.index.dayofweek
    return out


def event_row(ts: pd.Timestamp, event_name: str, family: str, variant: str, row: pd.Series, extra: dict[str, object]) -> dict[str, object]:
    return {
        "signal_time": ts,
        "side": 1,
        "event_name": event_name,
        "event_family": family,
        "variant": variant,
        "swing_level": float(row["swing_low"]),
        "sweep_extreme": float(row["low"]),
        "structural_stop_level": float(min(float(row["low"]), float(row["swing_low"]))),
        "swing_age": float(row["swing_low_age"]),
        "swing_prominence_pct": float(row["swing_low_prominence_pct"]),
        "down_spike_pct": float(row.get("down_spike_pct", np.nan)),
        "volume_ratio": float(row.get("volume_ratio", np.nan)),
        "trades_count_ratio": float(row.get("trades_count_ratio", np.nan)),
        "volume_spike": bool(row.get("volume_spike", False)),
        "atr_pct": float(row.get("atr_pct", np.nan)),
        "delta_notional": float(row.get("delta_notional", np.nan)),
        "delta_notional_ratio": float(row.get("delta_notional_ratio", np.nan)),
        "delta_notional_z": float(row.get("delta_notional_z", np.nan)),
        "cvd_notional_change": float(row.get("cvd_notional_change", np.nan)),
        "taker_buy_ratio": float(row.get("taker_buy_ratio", np.nan)),
        "large_delta_notional": float(row.get("large_delta_notional", np.nan)),
        "lower_wick_frac": float(row.get("lower_wick_frac", np.nan)),
        "close_pos_in_bar": float(row.get("close_pos_in_bar", np.nan)),
        "session_hour": int(row.get("session_hour", -1)) if pd.notna(row.get("session_hour", np.nan)) else -1,
        "session_bucket": str(row.get("session_bucket", "NA")),
        "weekday": int(row.get("weekday", -1)) if pd.notna(row.get("weekday", np.nan)) else -1,
        "close": float(row["close"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        **extra,
    }


def build_low_sweep_events(features: pd.DataFrame, args: Any) -> pd.DataFrame:
    spike_pcts = parse_number_list(args.spike_pcts, cast=float, name="spike_pcts")
    breakout_pcts = parse_number_list(args.breakout_pcts, cast=float, name="breakout_pcts", allow_zero=True)
    max_ages = parse_number_list(args.max_swing_ages, cast=int, name="max_swing_ages")
    min_proms = parse_number_list(args.min_swing_prominence_pcts, cast=float, name="min_swing_prominence_pcts")
    variants = parse_variant_list(args.variants)

    rows: list[dict[str, object]] = []
    for spike_pct in spike_pcts:
        for breakout_pct in breakout_pcts:
            for max_age in max_ages:
                for min_prom in min_proms:
                    base = (
                        features["swing_low"].notna()
                        & features["swing_low_age"].between(int(args.min_swing_age), int(max_age), inclusive="both")
                        & (features["swing_low_prominence_pct"] >= float(min_prom))
                        & (features["down_spike_pct"] >= float(spike_pct))
                        & (features["low"] <= features["swing_low"] * (1.0 - float(breakout_pct)))
                    )
                    variant_masks = {
                        "fade_close_through": base
                        & (features["close"] <= features["swing_low"] * (1.0 - float(args.close_through_buffer_pct))),
                        "reject": base & (features["close"] > features["swing_low"]),
                        "wick": base & (features["lower_wick_frac"] >= float(args.wick_min_frac)),
                    }
                    for variant in variants:
                        mask = variant_masks[variant]
                        family = f"low_sweep_{variant}"
                        suffix = f"sp{int(spike_pct * 10000):04d}_br{int(breakout_pct * 10000):04d}_age{max_age}_prom{int(min_prom * 10000):04d}"
                        name = f"low_{variant}_{suffix}"
                        for ts, row in features.loc[mask].iterrows():
                            rows.append(
                                event_row(
                                    ts,
                                    name,
                                    family,
                                    variant,
                                    row,
                                    {
                                        "spike_threshold_pct": float(spike_pct),
                                        "breakout_threshold_pct": float(breakout_pct),
                                        "max_swing_age": int(max_age),
                                        "min_swing_prominence_pct": float(min_prom),
                                    },
                                )
                            )
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    return events.sort_values(["signal_time", "event_name"]).reset_index(drop=True)


def build_canonical_events(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = events.copy()
    variant_rank = {"fade_close_through": 0, "wick": 1, "reject": 2}
    out["variant_rank"] = out["variant"].map(variant_rank).fillna(9).astype(int)
    out["specificity_score"] = (
        pd.to_numeric(out["spike_threshold_pct"], errors="coerce").fillna(0) * 10_000
        + pd.to_numeric(out["min_swing_prominence_pct"], errors="coerce").fillna(0) * 10_000
        + pd.to_numeric(out["breakout_threshold_pct"], errors="coerce").fillna(0) * 10_000
        - pd.to_numeric(out["max_swing_age"], errors="coerce").fillna(999) * 0.01
        - out["variant_rank"] * 0.001
    )
    out = out.sort_values(["signal_time", "side", "specificity_score"], ascending=[True, True, False])
    return out.drop_duplicates(["signal_time", "side"], keep="first").reset_index(drop=True)

