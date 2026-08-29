#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R23 frozen panic-wick structural Long helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

EPS = 1e-12


@dataclass(frozen=True)
class R23Config:
    wick_share_min: float = 0.50
    wick_atr_min: float = 0.55
    volume_ratio_min: float = 2.0
    reclaim_close_pos: float = 0.66
    soft_reclaim_close_pos: float = 0.55
    prior_flush_30_min: float = -0.005
    prior_flush_120_min: float = -0.010
    delta_absorption_max: float = -0.10
    taker_buy_absorption_max: float = 0.45
    deeper_sweep_buffer_pct: float = 0.0015
    entry_extra_delay_bars: int = 2
    feature_observed_minutes: int = 240
    market_roundtrip_cost: float = 0.0011
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)
    discovery_start: pd.Timestamp = pd.Timestamp("2023-01-01")
    validation_start: pd.Timestamp = pd.Timestamp("2025-01-01")
    embargo_start: pd.Timestamp = pd.Timestamp("2025-07-01")
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01")

    def validate(self) -> "R23Config":
        positive = (
            self.wick_share_min, self.wick_atr_min, self.volume_ratio_min,
            self.reclaim_close_pos, self.soft_reclaim_close_pos,
            self.deeper_sweep_buffer_pct, self.feature_observed_minutes,
            self.market_roundtrip_cost,
        )
        if any(value <= 0 for value in positive) or self.entry_extra_delay_bars < 0:
            raise ValueError("invalid R23 contract")
        if not (self.discovery_start < self.validation_start < self.embargo_start < self.holdout_start):
            raise ValueError("invalid R23 splits")
        return self


def regularize_trade_bars(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("empty trade bars")
    out = frame.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        if "timestamp" not in out.columns:
            raise ValueError("trade bars require timestamps")
        out.index = pd.to_datetime(out.pop("timestamp"))
    out.index = pd.to_datetime(out.index).tz_localize(None) if out.index.tz is not None else pd.to_datetime(out.index)
    out = out.sort_index().loc[lambda x: ~x.index.duplicated(keep="last")]
    required = ["open", "high", "low", "close", "volume"]
    if any(column not in out.columns for column in required):
        raise ValueError("trade bars missing OHLCV")
    full_index = pd.date_range(out.index.min(), out.index.max(), freq="1min")
    observed = pd.Series(1, index=out.index, dtype=np.int8).reindex(full_index, fill_value=0)
    out = out.reindex(full_index)
    previous_close = pd.to_numeric(out["close"], errors="coerce").ffill()
    for column in ("open", "high", "low", "close"):
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(previous_close)
    zero_columns = [column for column in out.columns if column not in ("open", "high", "low", "close")]
    for column in zero_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    out["source_bar_observed_flag"] = observed.to_numpy(dtype=np.int8)
    out.index.name = "timestamp"
    return out


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)


def build_panic_features(bars: pd.DataFrame, *, config: R23Config | None = None) -> pd.DataFrame:
    cfg = (config or R23Config()).validate()
    df = regularize_trade_bars(bars)
    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_low = df[["open", "close"]].min(axis=1)
    body_high = df[["open", "close"]].max(axis=1)
    out = df.copy()
    out["close_pos"] = _safe_divide(df["close"] - df["low"], rng)
    out["lower_wick"] = (body_low - df["low"]).clip(lower=0.0)
    out["upper_wick"] = (df["high"] - body_high).clip(lower=0.0)
    out["lower_wick_share"] = _safe_divide(out["lower_wick"], rng)
    out["upper_wick_share"] = _safe_divide(out["upper_wick"], rng)
    previous_close = df["close"].shift(1)
    out["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    out["atr"] = out["tr"].rolling(60, min_periods=30).mean()
    out["atr_pct"] = _safe_divide(out["atr"], df["close"])
    out["lower_wick_atr"] = _safe_divide(out["lower_wick"], out["atr"])
    volume_base = df["volume"].shift(1).rolling(240, min_periods=60).median()
    out["volume_ratio"] = _safe_divide(df["volume"], volume_base)
    out["ret_30"] = df["close"].pct_change(30)
    out["ret_120"] = df["close"].pct_change(120)
    out["ema_240"] = df["close"].ewm(span=240, adjust=False, min_periods=240).mean()
    out["ema240_slope_60"] = out["ema_240"] / out["ema_240"].shift(60) - 1.0
    delta = pd.to_numeric(df.get("delta_notional", np.nan), errors="coerce")
    notional = pd.to_numeric(df.get("notional", np.nan), errors="coerce").abs()
    out["delta_ratio"] = _safe_divide(delta, notional)
    if "taker_buy_ratio" in df.columns:
        out["taker_buy_ratio"] = pd.to_numeric(df["taker_buy_ratio"], errors="coerce")
    else:
        out["taker_buy_ratio"] = _safe_divide(pd.to_numeric(df.get("buy_volume", np.nan), errors="coerce"), df["volume"])
    out["vol_regime"] = pd.cut(
        out["atr_pct"], [-np.inf, 0.0015, 0.0030, 0.0050, np.inf],
        labels=["very_low_vol", "low_mid_vol", "mid_high_vol", "extreme_vol"],
    ).astype("object").fillna("NA")
    out["trend_down"] = (df["close"] < out["ema_240"]) & (out["ema240_slope_60"] < -0.0005)
    out["observed_240"] = out["source_bar_observed_flag"].rolling(
        cfg.feature_observed_minutes, min_periods=cfg.feature_observed_minutes
    ).sum().eq(cfg.feature_observed_minutes)
    return out


def build_priority_union_events(features: pd.DataFrame, *, config: R23Config | None = None) -> pd.DataFrame:
    cfg = (config or R23Config()).validate()
    lower_wick = (
        features["lower_wick_share"].ge(cfg.wick_share_min)
        & features["lower_wick_atr"].ge(cfg.wick_atr_min)
        & features["upper_wick_share"].lt(cfg.wick_share_min)
    )
    context = (
        lower_wick
        & features["volume_ratio"].ge(cfg.volume_ratio_min)
        & features["vol_regime"].isin(["mid_high_vol", "extreme_vol"])
        & (features["ret_30"].le(cfg.prior_flush_30_min) | features["ret_120"].le(cfg.prior_flush_120_min))
        & features["trend_down"]
        & features["observed_240"]
        & features["source_bar_observed_flag"].eq(1)
    )
    flow = context & features["close_pos"].ge(cfg.soft_reclaim_close_pos) & (
        features["delta_ratio"].le(cfg.delta_absorption_max)
        | features["taker_buy_ratio"].le(cfg.taker_buy_absorption_max)
    )
    reclaim = context & features["close_pos"].ge(cfg.reclaim_close_pos)
    union = flow | reclaim
    selected = features.loc[union].copy()
    if selected.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "event_bar_time": selected.index,
        "signal_available_time": selected.index + pd.Timedelta(minutes=1),
        "entry_time": selected.index + pd.Timedelta(minutes=1 + cfg.entry_extra_delay_bars),
        "source_event": np.where(flow.loc[selected.index], "strict_flow", "strict_reclaim"),
        "event_low": selected["low"].to_numpy(dtype=float),
        "event_high": selected["high"].to_numpy(dtype=float),
        "event_close": selected["close"].to_numpy(dtype=float),
        "lower_wick_share": selected["lower_wick_share"].to_numpy(dtype=float),
        "lower_wick_atr": selected["lower_wick_atr"].to_numpy(dtype=float),
        "volume_ratio": selected["volume_ratio"].to_numpy(dtype=float),
        "ret_30": selected["ret_30"].to_numpy(dtype=float),
        "ret_120": selected["ret_120"].to_numpy(dtype=float),
        "delta_ratio": selected["delta_ratio"].to_numpy(dtype=float),
        "taker_buy_ratio": selected["taker_buy_ratio"].to_numpy(dtype=float),
    })
    out["event_id"] = [f"R23_EVENT_{i:06d}" for i in range(1, len(out) + 1)]
    return out


def _confirmed_higher_low(low: np.ndarray, pos: int, event_low: float) -> float | None:
    if pos < 2:
        return None
    pivot = pos - 1
    if low[pivot] <= low[pivot - 1] and low[pivot] <= low[pos] and low[pivot] > event_low:
        return float(low[pivot])
    return None


def simulate_frozen_panic_long(
    features: pd.DataFrame,
    events: pd.DataFrame,
    *,
    split: str,
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    config: R23Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R23Config()).validate()
    frame = features.sort_index()
    index = frame.index
    open_ = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    observed = frame["source_bar_observed_flag"].to_numpy(dtype=np.int8)
    candidates = events.loc[
        pd.to_datetime(events["event_bar_time"]).ge(split_start)
        & pd.to_datetime(events["event_bar_time"]).lt(split_end)
    ].sort_values("entry_time")
    next_free_pos = int(index.searchsorted(split_start, side="left")) - 1
    split_end_pos = int(index.searchsorted(split_end, side="left"))
    rows = []
    ordinal = 0
    for event in candidates.itertuples(index=False):
        entry_time = pd.Timestamp(event.entry_time)
        entry_pos = int(index.searchsorted(entry_time, side="left"))
        if entry_pos <= next_free_pos or entry_pos >= split_end_pos or index[entry_pos] != entry_time or observed[entry_pos] != 1:
            continue
        entry_price = float(open_[entry_pos])
        event_low = float(event.event_low)
        event_high = float(event.event_high)
        deeper_threshold = event_low * (1.0 - cfg.deeper_sweep_buffer_pct)
        sweep_active = False
        sweep_count = 0
        previous_below = False
        event_high_reclaimed = False
        trail_stop = np.nan
        trail_updates = 0
        exit_pos: int | None = None
        decision_pos: int | None = None
        exit_reason = "SPLIT_BOUNDARY_CENSORED"
        path_status = "boundary_censored"
        gap_pos: int | None = None
        for pos in range(entry_pos, split_end_pos - 1):
            if observed[pos] != 1 or observed[pos + 1] != 1:
                gap_pos = pos if observed[pos] != 1 else pos + 1
                exit_reason = "DATA_GAP_CENSORED"
                path_status = "data_gap_censored"
                break
            bar_low = float(low[pos])
            bar_close = float(close[pos])
            if bar_low < event_low:
                if not previous_below:
                    sweep_count += 1
                previous_below = True
                sweep_active = True
            else:
                previous_below = False
            if bar_close >= event_low:
                sweep_active = False
            if bar_close >= event_high:
                event_high_reclaimed = True
                if not np.isfinite(trail_stop):
                    trail_stop = event_low
            pivot = _confirmed_higher_low(low, pos, event_low)
            if event_high_reclaimed and pivot is not None and (not np.isfinite(trail_stop) or pivot > trail_stop):
                trail_stop = pivot
                trail_updates += 1
            reason = None
            if sweep_count >= 2 and bar_low < deeper_threshold and bar_close < event_low:
                reason = "MULTI_SWEEP_DEEPER_FAIL"
            elif event_high_reclaimed and np.isfinite(trail_stop) and bar_close < trail_stop:
                reason = "HIGHER_LOW_TRAIL_BREAK"
            if reason is not None:
                decision_pos = pos
                exit_pos = pos + 1
                exit_reason = reason
                path_status = "included"
                break
        ordinal += 1
        if exit_pos is not None:
            exit_time = index[exit_pos]
            exit_price = float(open_[exit_pos])
            gross = exit_price / entry_price - 1.0
            next_free_pos = exit_pos
        else:
            exit_time = pd.NaT
            exit_price = np.nan
            gross = np.nan
            if gap_pos is not None:
                resumed = np.flatnonzero(observed[gap_pos:split_end_pos] == 1)
                next_free_pos = gap_pos + int(resumed[0]) if len(resumed) else split_end_pos
            else:
                next_free_pos = split_end_pos
        row = {
            "trade_id": f"R23_{split}_LONG_{ordinal:04d}",
            "event_id": event.event_id,
            "source_event": event.source_event,
            "research_split": split,
            "direction": "Long",
            "event_bar_time": event.event_bar_time,
            "signal_available_time": event.signal_available_time,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "event_low": event_low,
            "event_high": event_high,
            "deeper_failure_price": deeper_threshold,
            "exit_decision_bar_time": index[decision_pos] if decision_pos is not None else pd.NaT,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "path_status": path_status,
            "sweep_count": sweep_count,
            "event_high_reclaimed": int(event_high_reclaimed),
            "trail_updates": trail_updates,
            "final_trail_stop": trail_stop,
            "holding_hours": float((exit_time - entry_time) / pd.Timedelta(hours=1)) if pd.notna(exit_time) else np.nan,
            "gross_return": gross,
        }
        for scale in cfg.cost_scales:
            row[f"net_return_cost{int(scale)}x"] = gross - scale * cfg.market_roundtrip_cost if np.isfinite(gross) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _pf(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gain = float(x[x > 0].sum())
    loss = float(-x[x < 0].sum())
    return gain / loss if loss > EPS else (np.inf if gain > EPS else np.nan)


def summarize_r23(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, all_part in trades.groupby("research_split", sort=True):
        part = all_part.loc[all_part["path_status"].eq("included")]
        net2 = pd.to_numeric(part["net_return_cost2x"], errors="coerce")
        top = net2.sort_values(ascending=False)
        without5 = net2.drop(index=top.head(5).index)
        without10 = net2.drop(index=top.head(10).index)
        equity = (1.0 + net2).cumprod()
        drawdown = equity / equity.cummax() - 1.0
        entries = pd.to_datetime(part["entry_time"]).sort_values()
        months = 24 if split == "discovery" else 6
        rows.append({
            "research_split": split,
            "trades": len(part),
            "censored": int(all_part["path_status"].ne("included").sum()),
            "censor_rate": float(all_part["path_status"].ne("included").mean()),
            "trades_per_month": len(part) / months,
            "win_rate": float(pd.to_numeric(part["gross_return"], errors="coerce").gt(0).mean()),
            "gross_pf": _pf(part["gross_return"]),
            "mean_gross_return": float(pd.to_numeric(part["gross_return"], errors="coerce").mean()),
            "net_pf_cost1x": _pf(part["net_return_cost1x"]),
            "mean_net_return_cost1x": float(pd.to_numeric(part["net_return_cost1x"], errors="coerce").mean()),
            "net_pf_cost2x": _pf(net2),
            "mean_net_return_cost2x": float(net2.mean()),
            "net_pf_cost3x": _pf(part["net_return_cost3x"]),
            "mean_net_return_cost3x": float(pd.to_numeric(part["net_return_cost3x"], errors="coerce").mean()),
            "compounded_return_cost2x": float(equity.iloc[-1] - 1.0) if len(equity) else np.nan,
            "max_drawdown_cost2x": float(drawdown.min()) if len(drawdown) else np.nan,
            "median_holding_hours": float(pd.to_numeric(part["holding_hours"], errors="coerce").median()),
            "longest_entry_gap_days": float(entries.diff().max() / pd.Timedelta(days=1)) if len(entries) > 1 else np.nan,
            "net_pf_cost2x_top5_removed": _pf(without5),
            "net_sum_cost2x_top5_removed": float(without5.sum()),
            "net_pf_cost2x_top10_removed": _pf(without10),
            "net_sum_cost2x_top10_removed": float(without10.sum()),
        })
    return pd.DataFrame(rows)


def summarize_r23_years(trades: pd.DataFrame) -> pd.DataFrame:
    closed = trades.loc[trades["path_status"].eq("included")].copy()
    closed["year"] = pd.to_datetime(closed["exit_time"]).dt.year
    rows = []
    for year, part in closed.groupby("year", sort=True):
        net2 = pd.to_numeric(part["net_return_cost2x"], errors="coerce")
        rows.append({"year": int(year), "trades": len(part), "net_pf_cost2x": _pf(net2), "mean_net_return_cost2x": float(net2.mean()), "net_sum_cost2x": float(net2.sum())})
    return pd.DataFrame(rows)


def build_r23_gate(score: pd.DataFrame, years: pd.DataFrame) -> pd.DataFrame:
    reasons = []
    indexed = score.set_index("research_split")
    for split, minimum in (("discovery", 100), ("validation", 20)):
        if split not in indexed.index:
            reasons.append(f"missing_{split}")
            continue
        row = indexed.loc[split]
        if int(row["trades"]) < minimum:
            reasons.append(f"{split}_sample")
        if float(row["net_pf_cost2x"]) < 1.4:
            reasons.append(f"{split}_pf2x")
        if float(row["mean_net_return_cost2x"]) <= 0:
            reasons.append(f"{split}_expectancy")
        if float(row["censor_rate"]) > 0.05:
            reasons.append(f"{split}_censor")
    if "discovery" in indexed.index and float(indexed.loc["discovery", "net_sum_cost2x_top10_removed"]) <= 0:
        reasons.append("discovery_top10")
    year_index = years.set_index("year")
    for year in (2023, 2024, 2025):
        if year not in year_index.index or float(year_index.loc[year, "net_sum_cost2x"]) <= 0:
            reasons.append(f"year_{year}")
    return pd.DataFrame([{"direction": "Long", "research_candidate": int(not reasons), "reason": "PASS" if not reasons else "FAIL_" + ",".join(reasons)}])


def r23_causal_audit(trades: pd.DataFrame, events: pd.DataFrame, *, config: R23Config | None = None) -> pd.DataFrame:
    cfg = (config or R23Config()).validate()
    closed = trades.loc[trades["path_status"].eq("included")]
    rows = [
        {"check": "unique_event_id", "violations": int(events["event_id"].duplicated().sum())},
        {"check": "unique_trade_id", "violations": int(trades["trade_id"].duplicated().sum())},
        {"check": "signal_after_event_close", "violations": int((pd.to_datetime(events["signal_available_time"]) != pd.to_datetime(events["event_bar_time"]) + pd.Timedelta(minutes=1)).sum())},
        {"check": "frozen_delay_two", "violations": int((pd.to_datetime(trades["entry_time"]) != pd.to_datetime(trades["event_bar_time"]) + pd.Timedelta(minutes=3)).sum())},
        {"check": "exit_next_open", "violations": int((pd.to_datetime(closed["exit_time"]) != pd.to_datetime(closed["exit_decision_bar_time"]) + pd.Timedelta(minutes=1)).sum())},
        {"check": "exit_after_entry", "violations": int((pd.to_datetime(closed["exit_time"]) <= pd.to_datetime(closed["entry_time"])).sum())},
        {"check": "discovery_boundary", "violations": int((closed["research_split"].eq("discovery") & pd.to_datetime(closed["exit_time"]).ge(cfg.validation_start)).sum())},
        {"check": "validation_boundary", "violations": int((closed["research_split"].eq("validation") & pd.to_datetime(closed["exit_time"]).ge(cfg.embargo_start)).sum())},
        {"check": "holdout_absent", "violations": int(trades["research_split"].isin(["embargo", "holdout"]).sum())},
    ]
    for scale in cfg.cost_scales:
        expected = pd.to_numeric(trades["gross_return"], errors="coerce") - scale * cfg.market_roundtrip_cost
        actual = pd.to_numeric(trades[f"net_return_cost{int(scale)}x"], errors="coerce")
        rows.append({"check": f"cost{int(scale)}x_formula", "violations": int((actual - expected).abs().dropna().gt(1e-12).sum())})
    out = pd.DataFrame(rows)
    out["status"] = np.where(out["violations"].eq(0), "PASS", "FAIL")
    return out

