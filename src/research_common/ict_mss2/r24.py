#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R24 scheduled funding-window unwind helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import normalize_1m_bars

EPS = 1e-12


@dataclass(frozen=True)
class R24Config:
    sigma_hours: int = 720
    impulse_z: float = 1.5
    atr_window: int = 20
    stop_atr: float = 1.5
    target_rs: tuple[float, ...] = (1.0, 2.0)
    hold_hours: int = 8
    schedule_hours: tuple[int, ...] = (0, 8, 16)
    market_roundtrip_cost: float = 0.0011
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)
    discovery_start: pd.Timestamp = pd.Timestamp("2023-01-01")
    validation_start: pd.Timestamp = pd.Timestamp("2025-01-01")
    embargo_start: pd.Timestamp = pd.Timestamp("2025-07-01")
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01")

    def validate(self) -> "R24Config":
        if min(self.sigma_hours, self.atr_window, self.impulse_z, self.stop_atr, self.hold_hours, self.market_roundtrip_cost) <= 0:
            raise ValueError("invalid R24 contract")
        if self.schedule_hours != (0, 8, 16) or any(value <= 0 for value in self.target_rs):
            raise ValueError("invalid R24 schedule/targets")
        if not (self.discovery_start < self.validation_start < self.embargo_start < self.holdout_start):
            raise ValueError("invalid R24 splits")
        return self


def build_funding_window_events(bars_1m: pd.DataFrame, *, config: R24Config | None = None) -> pd.DataFrame:
    cfg = (config or R24Config()).validate()
    bars = normalize_1m_bars(bars_1m)
    work = bars.copy()
    work["_count"] = 1
    hourly = work.resample("1h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), count=("_count", "sum")
    )
    hourly = hourly.loc[hourly["count"].eq(60)].drop(columns="count")
    hourly["return_1h"] = hourly["close"].pct_change()
    hourly["sigma_prior"] = hourly["return_1h"].shift(1).rolling(cfg.sigma_hours, min_periods=cfg.sigma_hours).std()
    previous_close = hourly["close"].shift(1)
    tr = pd.concat([
        hourly["high"] - hourly["low"],
        (hourly["high"] - previous_close).abs(),
        (hourly["low"] - previous_close).abs(),
    ], axis=1).max(axis=1)
    hourly["atr20"] = tr.ewm(alpha=1.0 / cfg.atr_window, adjust=False, min_periods=cfg.atr_window).mean()
    hourly["pre_settlement_z"] = hourly["return_1h"] / hourly["sigma_prior"]
    hourly["event_bar_time"] = hourly.index
    hourly["entry_time"] = hourly.index + pd.Timedelta(hours=1)
    schedule = pd.to_datetime(hourly["entry_time"]).dt.hour.isin(cfg.schedule_hours)
    valid = np.isfinite(hourly[["pre_settlement_z", "atr20", "return_1h"]]).all(axis=1)
    events = hourly.loc[schedule & valid & hourly["pre_settlement_z"].abs().ge(cfg.impulse_z)].copy().reset_index(drop=True)
    events["trade_direction"] = -np.sign(events["return_1h"]).astype(int)
    events["direction"] = np.where(events["trade_direction"].gt(0), "Long", "Short")
    events["signal_available_time"] = events["entry_time"]
    events["event_id"] = [f"R24_EVENT_{i:06d}" for i in range(1, len(events) + 1)]
    return events


def _passage(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, direction: int, stop: float, target: float) -> tuple[str, pd.Timestamp, float] | None:
    section = bars.loc[(bars.index >= start) & (bars.index < end)]
    for timestamp, row in section.iterrows():
        stop_hit = float(row["low"]) <= stop if direction > 0 else float(row["high"]) >= stop
        target_hit = float(row["high"]) >= target if direction > 0 else float(row["low"]) <= target
        if stop_hit:
            raw_open = float(row["open"])
            return "STOP", pd.Timestamp(timestamp), min(raw_open, stop) if direction > 0 else max(raw_open, stop)
        if target_hit:
            return "TARGET", pd.Timestamp(timestamp), target
    return None


def simulate_funding_unwind(
    bars_1m: pd.DataFrame,
    events: pd.DataFrame,
    *,
    target_r: float,
    direction: int,
    split: str,
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    config: R24Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R24Config()).validate()
    if direction not in (-1, 1) or target_r not in cfg.target_rs:
        raise ValueError("invalid R24 path")
    bars = normalize_1m_bars(bars_1m)
    candidates = events.loc[
        events["trade_direction"].eq(direction)
        & pd.to_datetime(events["entry_time"]).ge(split_start)
        & pd.to_datetime(events["entry_time"]).lt(split_end)
    ].sort_values("entry_time")
    rows = []
    available = split_start
    ordinal = 0
    for event in candidates.itertuples(index=False):
        entry_time = pd.Timestamp(event.entry_time)
        if entry_time < available or entry_time not in bars.index:
            continue
        entry_price = float(bars.loc[entry_time, "open"])
        risk = cfg.stop_atr * float(event.atr20)
        stop = entry_price - direction * risk
        target = entry_price + direction * float(target_r) * risk
        deadline = entry_time + pd.Timedelta(hours=cfg.hold_hours)
        path_end = min(deadline, split_end)
        hit = _passage(bars, entry_time, path_end, direction, stop, target)
        if hit is not None:
            reason, exit_time, exit_price = hit
            status = "included"
            available = exit_time + pd.Timedelta(minutes=1)
        elif deadline < split_end and deadline in bars.index:
            reason, exit_time, exit_price = "TIME_EXIT", deadline, float(bars.loc[deadline, "open"])
            status = "included"
            available = deadline
        else:
            reason, exit_time, exit_price = "SPLIT_BOUNDARY_CENSORED", pd.NaT, np.nan
            status = "boundary_censored"
            available = split_end
        ordinal += 1
        gross = direction * (exit_price / entry_price - 1.0) if np.isfinite(exit_price) else np.nan
        row = {
            "trade_id": f"R24_R{int(target_r)}_{split}_{'LONG' if direction > 0 else 'SHORT'}_{ordinal:04d}",
            "event_id": event.event_id, "target_r": float(target_r), "research_split": split,
            "direction": "Long" if direction > 0 else "Short", "trade_direction": direction,
            "event_bar_time": event.event_bar_time, "signal_available_time": event.signal_available_time,
            "entry_time": entry_time, "entry_price": entry_price, "pre_settlement_return": event.return_1h,
            "pre_settlement_z": event.pre_settlement_z, "atr20": event.atr20,
            "risk_distance_pct": risk / entry_price, "stop_price": stop, "target_price": target,
            "exit_time": exit_time, "exit_price": exit_price, "exit_reason": reason, "path_status": status,
            "holding_hours": float((exit_time - entry_time) / pd.Timedelta(hours=1)) if pd.notna(exit_time) else np.nan,
            "gross_return": gross,
        }
        for scale in cfg.cost_scales:
            row[f"net_return_cost{int(scale)}x"] = gross - scale * cfg.market_roundtrip_cost if np.isfinite(gross) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _pf(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gain = float(x[x > 0].sum()); loss = float(-x[x < 0].sum())
    return gain / loss if loss > EPS else (np.inf if gain > EPS else np.nan)


def summarize_r24(trades: pd.DataFrame) -> pd.DataFrame:
    closed = trades.loc[trades["path_status"].eq("included")]
    rows = []
    for (target_r, split, direction), part in closed.groupby(["target_r", "research_split", "direction"], sort=True):
        net2 = pd.to_numeric(part["net_return_cost2x"], errors="coerce")
        top = net2.sort_values(ascending=False)
        without10 = net2.drop(index=top.head(10).index)
        entries = pd.to_datetime(part["entry_time"]).sort_values()
        months = 24 if split == "discovery" else 6
        rows.append({
            "target_r": target_r, "research_split": split, "direction": direction, "trades": len(part),
            "trades_per_month": len(part) / months, "win_rate": float(pd.to_numeric(part["gross_return"]).gt(0).mean()),
            "gross_pf": _pf(part["gross_return"]), "mean_gross_return": float(pd.to_numeric(part["gross_return"]).mean()),
            "net_pf_cost1x": _pf(part["net_return_cost1x"]), "mean_net_return_cost1x": float(pd.to_numeric(part["net_return_cost1x"]).mean()),
            "net_pf_cost2x": _pf(net2), "mean_net_return_cost2x": float(net2.mean()),
            "net_pf_cost3x": _pf(part["net_return_cost3x"]), "mean_net_return_cost3x": float(pd.to_numeric(part["net_return_cost3x"]).mean()),
            "timeout_rate": float(part["exit_reason"].eq("TIME_EXIT").mean()),
            "median_holding_hours": float(pd.to_numeric(part["holding_hours"]).median()),
            "median_risk_distance_pct": float(pd.to_numeric(part["risk_distance_pct"]).median()),
            "longest_entry_gap_days": float(entries.diff().max() / pd.Timedelta(days=1)) if len(entries) > 1 else np.nan,
            "net_pf_cost2x_top10_removed": _pf(without10), "net_sum_cost2x_top10_removed": float(without10.sum()),
        })
    return pd.DataFrame(rows)


def summarize_r24_years(trades: pd.DataFrame) -> pd.DataFrame:
    closed = trades.loc[trades["path_status"].eq("included")].copy()
    closed["year"] = pd.to_datetime(closed["exit_time"]).dt.year
    rows = []
    for (target_r, direction, year), part in closed.groupby(["target_r", "direction", "year"], sort=True):
        net2 = pd.to_numeric(part["net_return_cost2x"])
        rows.append({"target_r": target_r, "direction": direction, "year": int(year), "trades": len(part), "net_pf_cost2x": _pf(net2), "mean_net_return_cost2x": float(net2.mean()), "net_sum_cost2x": float(net2.sum())})
    return pd.DataFrame(rows)


def build_r24_gate(score: pd.DataFrame, years: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for direction in ("Long", "Short"):
        reasons = []
        primary = score.loc[score["target_r"].eq(1.0) & score["direction"].eq(direction)].set_index("research_split")
        for split, minimum in (("discovery", 50), ("validation", 10)):
            if split not in primary.index:
                reasons.append(f"missing_{split}"); continue
            row = primary.loc[split]
            if int(row["trades"]) < minimum: reasons.append(f"{split}_sample")
            if float(row["net_pf_cost2x"]) < 1.4: reasons.append(f"{split}_pf2x")
            if float(row["mean_net_return_cost2x"]) <= 0: reasons.append(f"{split}_expectancy")
            if float(row["timeout_rate"]) > 0.20: reasons.append(f"{split}_timeout")
        if "discovery" in primary.index and float(primary.loc["discovery", "net_sum_cost2x_top10_removed"]) <= 0:
            reasons.append("discovery_top10")
        yp = years.loc[years["target_r"].eq(1.0) & years["direction"].eq(direction)].set_index("year")
        for year in (2023, 2024, 2025):
            if year not in yp.index or float(yp.loc[year, "net_sum_cost2x"]) <= 0: reasons.append(f"year_{year}")
        r2 = score.loc[score["target_r"].eq(2.0) & score["direction"].eq(direction) & score["research_split"].eq("discovery")]
        if r2.empty or float(r2.iloc[0]["mean_net_return_cost2x"]) <= 0: reasons.append("r2_discovery")
        rows.append({"direction": direction, "research_candidate": int(not reasons), "reason": "PASS" if not reasons else "FAIL_" + ",".join(reasons)})
    return pd.DataFrame(rows)


def r24_causal_audit(trades: pd.DataFrame, *, config: R24Config | None = None) -> pd.DataFrame:
    cfg = (config or R24Config()).validate(); closed = trades.loc[trades["path_status"].eq("included")]
    expected_stop = pd.to_numeric(trades["entry_price"]) - pd.to_numeric(trades["trade_direction"]) * cfg.stop_atr * pd.to_numeric(trades["atr20"])
    expected_target = pd.to_numeric(trades["entry_price"]) + pd.to_numeric(trades["trade_direction"]) * pd.to_numeric(trades["target_r"]) * cfg.stop_atr * pd.to_numeric(trades["atr20"])
    rows = [
        {"check":"unique_trade_id","violations":int(trades["trade_id"].duplicated().sum())},
        {"check":"scheduled_entry","violations":int((~pd.to_datetime(trades["entry_time"]).dt.hour.isin(cfg.schedule_hours)).sum())},
        {"check":"entry_after_closed_hour","violations":int((pd.to_datetime(trades["entry_time"]) != pd.to_datetime(trades["event_bar_time"]) + pd.Timedelta(hours=1)).sum())},
        {"check":"reversal_direction","violations":int((np.sign(pd.to_numeric(trades["pre_settlement_return"])) == pd.to_numeric(trades["trade_direction"])).sum())},
        {"check":"impulse_threshold","violations":int(pd.to_numeric(trades["pre_settlement_z"]).abs().lt(cfg.impulse_z).sum())},
        {"check":"stop_formula","violations":int((pd.to_numeric(trades["stop_price"])-expected_stop).abs().gt(1e-10).sum())},
        {"check":"target_formula","violations":int((pd.to_numeric(trades["target_price"])-expected_target).abs().gt(1e-10).sum())},
        {"check":"exit_after_entry","violations":int((pd.to_datetime(closed["exit_time"]) < pd.to_datetime(closed["entry_time"])).sum())},
        {"check":"discovery_boundary","violations":int((closed["research_split"].eq("discovery") & pd.to_datetime(closed["exit_time"]).ge(cfg.validation_start)).sum())},
        {"check":"validation_boundary","violations":int((closed["research_split"].eq("validation") & pd.to_datetime(closed["exit_time"]).ge(cfg.embargo_start)).sum())},
        {"check":"holdout_absent","violations":int(trades["research_split"].isin(["embargo","holdout"]).sum())},
    ]
    for scale in cfg.cost_scales:
        expected = pd.to_numeric(trades["gross_return"], errors="coerce") - scale * cfg.market_roundtrip_cost
        actual = pd.to_numeric(trades[f"net_return_cost{int(scale)}x"], errors="coerce")
        rows.append({"check":f"cost{int(scale)}x_formula","violations":int((actual-expected).abs().dropna().gt(1e-12).sum())})
    out=pd.DataFrame(rows); out["status"]=np.where(out["violations"].eq(0),"PASS","FAIL"); return out

