#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R21 canonical daily channel trend-following research helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import normalize_1m_bars

EPS = 1e-12


@dataclass(frozen=True)
class R21Model:
    name: str
    entry_window: int
    exit_window: int


@dataclass(frozen=True)
class R21Config:
    atr_window: int = 20
    stop_atr: float = 2.0
    market_roundtrip_cost: float = 0.0011
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)
    models: tuple[R21Model, ...] = (
        R21Model("D20_X10", 20, 10),
        R21Model("D55_X20", 55, 20),
    )
    discovery_start: pd.Timestamp = pd.Timestamp("2023-01-01")
    validation_start: pd.Timestamp = pd.Timestamp("2025-01-01")
    embargo_start: pd.Timestamp = pd.Timestamp("2025-07-01")
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01")

    def validate(self) -> "R21Config":
        if self.atr_window <= 0 or self.stop_atr <= 0 or self.market_roundtrip_cost <= 0:
            raise ValueError("invalid R21 risk/cost contract")
        if not (self.discovery_start < self.validation_start < self.embargo_start < self.holdout_start):
            raise ValueError("invalid R21 splits")
        if any(model.entry_window <= model.exit_window or model.exit_window <= 0 for model in self.models):
            raise ValueError("entry channel must exceed exit channel")
        return self


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame.get(column, pd.Series(np.nan, index=frame.index)), errors="coerce")


def _pf(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gain = float(x[x > 0].sum())
    loss = float(-x[x < 0].sum())
    return gain / loss if loss > EPS else (np.inf if gain > EPS else np.nan)


def build_daily_channel_features(bars_1m: pd.DataFrame, *, config: R21Config | None = None) -> pd.DataFrame:
    cfg = (config or R21Config()).validate()
    bars = normalize_1m_bars(bars_1m)
    daily = bars.resample("1D", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum")
    ).dropna(subset=["open", "high", "low", "close"])
    prev_close = daily["close"].shift(1)
    tr = pd.concat(
        [daily["high"] - daily["low"], (daily["high"] - prev_close).abs(), (daily["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    daily["atr20"] = tr.ewm(alpha=1.0 / cfg.atr_window, adjust=False, min_periods=cfg.atr_window).mean()
    for model in cfg.models:
        daily[f"entry_high_{model.entry_window}"] = daily["high"].rolling(model.entry_window, min_periods=model.entry_window).max().shift(1)
        daily[f"entry_low_{model.entry_window}"] = daily["low"].rolling(model.entry_window, min_periods=model.entry_window).min().shift(1)
        daily[f"exit_high_{model.exit_window}"] = daily["high"].rolling(model.exit_window, min_periods=model.exit_window).max().shift(1)
        daily[f"exit_low_{model.exit_window}"] = daily["low"].rolling(model.exit_window, min_periods=model.exit_window).min().shift(1)
    daily["available_time"] = daily.index + pd.Timedelta(days=1)
    return daily


def _first_stop_touch(
    bars: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    direction: int,
    stop: float,
) -> tuple[pd.Timestamp, float] | None:
    section = bars.loc[(bars.index >= start) & (bars.index < end)]
    if section.empty:
        return None
    hit = section["low"].le(stop) if direction > 0 else section["high"].ge(stop)
    if not bool(hit.any()):
        return None
    timestamp = pd.Timestamp(hit.index[np.flatnonzero(hit.to_numpy(bool))[0]])
    row = section.loc[timestamp]
    raw_open = float(row["open"])
    price = min(raw_open, stop) if direction > 0 else max(raw_open, stop)
    return timestamp, price


def simulate_daily_channel(
    bars_1m: pd.DataFrame,
    daily: pd.DataFrame,
    *,
    model: R21Model,
    direction: int,
    split: str,
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    config: R21Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R21Config()).validate()
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    bars = normalize_1m_bars(bars_1m)
    days = daily.loc[(daily.index >= split_start) & (daily.index < split_end)].copy()
    rows: list[dict[str, object]] = []
    position: dict[str, object] | None = None
    pending_entry: dict[str, object] | None = None
    pending_exit_signal: pd.Timestamp | None = None
    ordinal = 0

    for day_time, day in days.iterrows():
        day_time = pd.Timestamp(day_time)
        next_day = day_time + pd.Timedelta(days=1)

        if position is not None and pending_exit_signal is not None:
            ordinal += 1
            entry = float(position["entry_price"])
            exit_price = float(day["open"])
            gross = direction * (exit_price / entry - 1.0)
            item = dict(position)
            item.update(
                {
                    "trade_id": f"R21_{model.name}_{split}_{'LONG' if direction > 0 else 'SHORT'}_{ordinal:04d}",
                    "exit_signal_available_time": pending_exit_signal,
                    "exit_time": day_time,
                    "exit_price": exit_price,
                    "exit_reason": "CHANNEL_EXIT_NEXT_OPEN",
                    "path_status": "included",
                    "gross_return": gross,
                    "holding_days": float((day_time - pd.Timestamp(position["entry_time"])) / pd.Timedelta(days=1)),
                }
            )
            rows.append(item)
            position = None
            pending_exit_signal = None

        if position is None and pending_entry is not None:
            atr_value = float(pending_entry["atr_at_signal"])
            if np.isfinite(atr_value) and atr_value > EPS:
                entry_price = float(day["open"])
                stop = entry_price - direction * cfg.stop_atr * atr_value
                position = {
                    "model": model.name,
                    "research_split": split,
                    "direction": "Long" if direction > 0 else "Short",
                    "trade_direction": direction,
                    "entry_signal_bar_time": pending_entry["signal_bar_time"],
                    "entry_signal_available_time": day_time,
                    "entry_time": day_time,
                    "entry_price": entry_price,
                    "initial_stop_price": stop,
                    "atr20_at_signal": atr_value,
                    "risk_distance_pct": abs(entry_price - stop) / entry_price,
                }
            pending_entry = None

        stopped = False
        if position is not None:
            stop = float(position["initial_stop_price"])
            daily_touch = float(day["low"]) <= stop if direction > 0 else float(day["high"]) >= stop
            if daily_touch:
                touch = _first_stop_touch(bars, day_time, min(next_day, split_end), direction, stop)
                if touch is not None:
                    ordinal += 1
                    exit_time, exit_price = touch
                    entry = float(position["entry_price"])
                    gross = direction * (exit_price / entry - 1.0)
                    item = dict(position)
                    item.update(
                        {
                            "trade_id": f"R21_{model.name}_{split}_{'LONG' if direction > 0 else 'SHORT'}_{ordinal:04d}",
                            "exit_signal_available_time": pd.NaT,
                            "exit_time": exit_time,
                            "exit_price": exit_price,
                            "exit_reason": "INITIAL_ATR_STOP",
                            "path_status": "included",
                            "gross_return": gross,
                            "holding_days": float((exit_time - pd.Timestamp(position["entry_time"])) / pd.Timedelta(days=1)),
                        }
                    )
                    rows.append(item)
                    position = None
                    pending_exit_signal = None
                    stopped = True

        close = float(day["close"])
        if position is not None and not stopped:
            exit_level = float(day[f"exit_low_{model.exit_window}"] if direction > 0 else day[f"exit_high_{model.exit_window}"])
            exit_signal = close < exit_level if direction > 0 else close > exit_level
            if np.isfinite(exit_level) and exit_signal:
                pending_exit_signal = next_day

        if position is None and pending_entry is None:
            entry_level = float(day[f"entry_high_{model.entry_window}"] if direction > 0 else day[f"entry_low_{model.entry_window}"])
            entry_signal = close > entry_level if direction > 0 else close < entry_level
            if np.isfinite(entry_level) and entry_signal and next_day < split_end:
                pending_entry = {"signal_bar_time": day_time, "atr_at_signal": float(day["atr20"])}

    if position is not None:
        ordinal += 1
        item = dict(position)
        item.update(
            {
                "trade_id": f"R21_{model.name}_{split}_{'LONG' if direction > 0 else 'SHORT'}_{ordinal:04d}",
                "exit_signal_available_time": pending_exit_signal,
                "exit_time": pd.NaT,
                "exit_price": np.nan,
                "exit_reason": "SPLIT_BOUNDARY_CENSORED",
                "path_status": "boundary_censored",
                "gross_return": np.nan,
                "holding_days": float((split_end - pd.Timestamp(position["entry_time"])) / pd.Timedelta(days=1)),
            }
        )
        rows.append(item)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for scale in cfg.cost_scales:
        out[f"net_return_cost{int(scale)}x"] = _num(out, "gross_return") - scale * cfg.market_roundtrip_cost
    return out


def summarize_r21(trades: pd.DataFrame, *, config: R21Config | None = None) -> pd.DataFrame:
    cfg = (config or R21Config()).validate()
    closed = trades.loc[trades.get("path_status", pd.Series(dtype=str)).eq("included")].copy()
    if closed.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (model, split, direction), part in closed.groupby(["model", "research_split", "direction"], sort=True):
        months = 24 if split == "discovery" else 6
        month_index = pd.period_range(
            cfg.discovery_start if split == "discovery" else cfg.validation_start,
            (cfg.validation_start if split == "discovery" else cfg.embargo_start) - pd.Timedelta(seconds=1), freq="M"
        )
        net2 = _num(part, "net_return_cost2x")
        top = net2.sort_values(ascending=False)
        without5 = net2.drop(index=top.head(5).index)
        without10 = net2.drop(index=top.head(10).index)
        monthly = net2.groupby(pd.to_datetime(part["exit_time"]).dt.to_period("M")).sum().reindex(month_index, fill_value=0.0)
        entries = pd.to_datetime(part["entry_time"]).sort_values()
        exits = pd.to_datetime(part.sort_values("entry_time")["exit_time"]).reset_index(drop=True)
        ordered_entries = pd.to_datetime(part.sort_values("entry_time")["entry_time"]).reset_index(drop=True)
        flat = (ordered_entries.iloc[1:].reset_index(drop=True) - exits.iloc[:-1].reset_index(drop=True)) / pd.Timedelta(days=1)
        rows.append(
            {
                "model": model, "research_split": split, "direction": direction, "trades": len(part),
                "trades_per_month": len(part) / months, "win_rate": float(_num(part, "gross_return").gt(0).mean()),
                "gross_pf": _pf(_num(part, "gross_return")), "mean_gross_return": float(_num(part, "gross_return").mean()),
                "net_pf_cost1x": _pf(_num(part, "net_return_cost1x")), "mean_net_return_cost1x": float(_num(part, "net_return_cost1x").mean()),
                "net_pf_cost2x": _pf(net2), "mean_net_return_cost2x": float(net2.mean()),
                "net_pf_cost3x": _pf(_num(part, "net_return_cost3x")), "mean_net_return_cost3x": float(_num(part, "net_return_cost3x").mean()),
                "positive_month_rate_cost2x": float(monthly.gt(0).mean()),
                "longest_entry_gap_days": float(entries.diff().max() / pd.Timedelta(days=1)) if len(entries) > 1 else np.nan,
                "longest_flat_days": float(flat.clip(lower=0).max()) if len(flat) else np.nan,
                "median_flat_days": float(flat.clip(lower=0).median()) if len(flat) else np.nan,
                "p90_flat_days": float(flat.clip(lower=0).quantile(0.9)) if len(flat) else np.nan,
                "median_holding_days": float(_num(part, "holding_days").median()),
                "median_risk_distance_pct": float(_num(part, "risk_distance_pct").median()),
                "net_pf_cost2x_top5_removed": _pf(without5), "net_sum_cost2x_top5_removed": float(without5.sum()),
                "net_pf_cost2x_top10_removed": _pf(without10), "net_sum_cost2x_top10_removed": float(without10.sum()),
            }
        )
    return pd.DataFrame(rows)


def summarize_r21_years(trades: pd.DataFrame) -> pd.DataFrame:
    closed = trades.loc[trades.get("path_status", pd.Series(dtype=str)).eq("included")].copy()
    if closed.empty:
        return pd.DataFrame()
    closed["year"] = pd.to_datetime(closed["exit_time"]).dt.year
    rows = []
    for (model, direction, year), part in closed.groupby(["model", "direction", "year"], sort=True):
        net2 = _num(part, "net_return_cost2x")
        rows.append({"model": model, "direction": direction, "year": int(year), "trades": len(part), "net_pf_cost2x": _pf(net2), "mean_net_return_cost2x": float(net2.mean()), "net_sum_cost2x": float(net2.sum())})
    return pd.DataFrame(rows)


def r21_causal_audit(trades: pd.DataFrame, *, config: R21Config | None = None) -> pd.DataFrame:
    cfg = (config or R21Config()).validate()
    closed = trades.loc[trades["path_status"].eq("included")].copy()
    stop_expected = _num(trades, "entry_price") - _num(trades, "trade_direction") * cfg.stop_atr * _num(trades, "atr20_at_signal")
    rows = [
        {"check": "unique_trade_id", "violations": int(trades["trade_id"].duplicated().sum())},
        {"check": "entry_after_closed_signal", "violations": int((pd.to_datetime(trades["entry_time"]) != pd.to_datetime(trades["entry_signal_bar_time"]) + pd.Timedelta(days=1)).sum())},
        {"check": "exit_after_entry", "violations": int((pd.to_datetime(closed["exit_time"]) < pd.to_datetime(closed["entry_time"])).sum())},
        {"check": "fixed_atr_stop_formula", "violations": int((_num(trades, "initial_stop_price") - stop_expected).abs().gt(1e-10).sum())},
        {"check": "channel_exit_next_open", "violations": int((pd.to_datetime(closed.loc[closed["exit_reason"].eq("CHANNEL_EXIT_NEXT_OPEN"), "exit_time"]) != pd.to_datetime(closed.loc[closed["exit_reason"].eq("CHANNEL_EXIT_NEXT_OPEN"), "exit_signal_available_time"])).sum())},
        {"check": "discovery_boundary", "violations": int((closed["research_split"].eq("discovery") & pd.to_datetime(closed["exit_time"]).ge(cfg.validation_start)).sum())},
        {"check": "validation_boundary", "violations": int((closed["research_split"].eq("validation") & pd.to_datetime(closed["exit_time"]).ge(cfg.embargo_start)).sum())},
        {"check": "embargo_or_holdout_absent", "violations": int(trades["research_split"].isin(["embargo", "holdout"]).sum())},
    ]
    for scale in cfg.cost_scales:
        expected = _num(trades, "gross_return") - scale * cfg.market_roundtrip_cost
        actual = _num(trades, f"net_return_cost{int(scale)}x")
        rows.append({"check": f"cost{int(scale)}x_formula", "violations": int((actual - expected).abs().dropna().gt(1e-12).sum())})
    audit = pd.DataFrame(rows)
    audit["status"] = np.where(audit["violations"].eq(0), "PASS", "FAIL")
    return audit

