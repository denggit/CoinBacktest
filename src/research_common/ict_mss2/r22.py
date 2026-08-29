#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R22 BTC-led ETH catch-up research helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import normalize_1m_bars

EPS = 1e-12


@dataclass(frozen=True)
class R22Config:
    beta_hours: int = 720
    btc_sigma_hours: int = 168
    residual_sigma_hours: int = 720
    impulse_z: float = 2.0
    lag_z: float = 0.75
    atr_window: int = 20
    stop_atr: float = 1.5
    target_rs: tuple[float, ...] = (1.0, 2.0)
    max_hold_hours: int = 24
    market_roundtrip_cost: float = 0.0011
    cost_scales: tuple[float, ...] = (1.0, 2.0, 3.0)
    discovery_start: pd.Timestamp = pd.Timestamp("2023-01-01")
    validation_start: pd.Timestamp = pd.Timestamp("2025-01-01")
    embargo_start: pd.Timestamp = pd.Timestamp("2025-07-01")
    holdout_start: pd.Timestamp = pd.Timestamp("2025-08-01")

    def validate(self) -> "R22Config":
        if min(self.beta_hours, self.btc_sigma_hours, self.residual_sigma_hours, self.atr_window) < 2:
            raise ValueError("invalid R22 lookback")
        if min(self.impulse_z, self.lag_z, self.stop_atr, self.max_hold_hours, self.market_roundtrip_cost) <= 0:
            raise ValueError("invalid R22 threshold/risk contract")
        if not self.target_rs or any(value <= 0 for value in self.target_rs):
            raise ValueError("target Rs must be positive")
        if not (self.discovery_start < self.validation_start < self.embargo_start < self.holdout_start):
            raise ValueError("invalid R22 splits")
        return self


def _pf(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gain = float(x[x > 0].sum())
    loss = float(-x[x < 0].sum())
    return gain / loss if loss > EPS else (np.inf if gain > EPS else np.nan)


def _hourly(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    bars = normalize_1m_bars(frame)
    work = bars.copy()
    work["_count"] = 1
    out = work.resample("1h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"), count=("_count", "sum")
    )
    out = out.loc[out["count"].eq(60)].drop(columns="count")
    return out.add_prefix(f"{prefix}_")


def build_cross_market_features(
    eth_1m: pd.DataFrame,
    btc_1m: pd.DataFrame,
    *,
    config: R22Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R22Config()).validate()
    hourly = _hourly(eth_1m, "eth").join(_hourly(btc_1m, "btc"), how="inner")
    hourly["eth_return_1h"] = hourly["eth_close"].pct_change()
    hourly["btc_return_1h"] = hourly["btc_close"].pct_change()
    eth_prior = hourly["eth_return_1h"].shift(1)
    btc_prior = hourly["btc_return_1h"].shift(1)
    covariance = eth_prior.rolling(cfg.beta_hours, min_periods=cfg.beta_hours).cov(btc_prior)
    variance = btc_prior.rolling(cfg.beta_hours, min_periods=cfg.beta_hours).var()
    hourly["beta_prior"] = covariance / variance.where(variance.abs().gt(EPS))
    hourly["btc_sigma_prior"] = btc_prior.rolling(cfg.btc_sigma_hours, min_periods=cfg.btc_sigma_hours).std()
    hourly["residual"] = hourly["eth_return_1h"] - hourly["beta_prior"] * hourly["btc_return_1h"]
    hourly["residual_sigma_prior"] = hourly["residual"].shift(1).rolling(
        cfg.residual_sigma_hours, min_periods=cfg.residual_sigma_hours
    ).std()
    previous_close = hourly["eth_close"].shift(1)
    true_range = pd.concat(
        [
            hourly["eth_high"] - hourly["eth_low"],
            (hourly["eth_high"] - previous_close).abs(),
            (hourly["eth_low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    hourly["eth_atr20"] = true_range.ewm(
        alpha=1.0 / cfg.atr_window, adjust=False, min_periods=cfg.atr_window
    ).mean()
    hourly["btc_impulse_z"] = hourly["btc_return_1h"] / hourly["btc_sigma_prior"]
    direction = np.sign(hourly["btc_return_1h"])
    hourly["trade_direction"] = direction.astype(float)
    hourly["eth_signed_return"] = direction * hourly["eth_return_1h"]
    hourly["signed_lag"] = direction * (
        hourly["beta_prior"] * hourly["btc_return_1h"] - hourly["eth_return_1h"]
    )
    hourly["lag_z"] = hourly["signed_lag"] / hourly["residual_sigma_prior"]
    required = ["beta_prior", "btc_sigma_prior", "residual_sigma_prior", "eth_atr20", "btc_impulse_z", "lag_z"]
    valid = np.isfinite(hourly[required]).all(axis=1)
    hourly["event_flag"] = (
        valid
        & hourly["btc_impulse_z"].abs().ge(cfg.impulse_z)
        & hourly["eth_signed_return"].ge(0.0)
        & hourly["lag_z"].ge(cfg.lag_z)
    ).astype(np.int8)
    hourly["signal_bar_time"] = hourly.index
    hourly["signal_available_time"] = hourly.index + pd.Timedelta(hours=1)
    return hourly


def build_catchup_events(features: pd.DataFrame) -> pd.DataFrame:
    events = features.loc[features["event_flag"].eq(1)].copy().reset_index(drop=True)
    if events.empty:
        return events
    events["event_id"] = [f"R22_EVENT_{i:06d}" for i in range(1, len(events) + 1)]
    events["direction"] = np.where(events["trade_direction"].gt(0), "Long", "Short")
    return events


def _first_passage(
    bars: pd.DataFrame,
    *,
    entry_time: pd.Timestamp,
    deadline: pd.Timestamp,
    direction: int,
    stop: float,
    target: float,
) -> tuple[str, pd.Timestamp, float] | None:
    section = bars.loc[(bars.index >= entry_time) & (bars.index < deadline)]
    for timestamp, row in section.iterrows():
        stop_hit = float(row["low"]) <= stop if direction > 0 else float(row["high"]) >= stop
        target_hit = float(row["high"]) >= target if direction > 0 else float(row["low"]) <= target
        if stop_hit:
            raw_open = float(row["open"])
            price = min(raw_open, stop) if direction > 0 else max(raw_open, stop)
            return "STOP", pd.Timestamp(timestamp), price
        if target_hit:
            return "TARGET", pd.Timestamp(timestamp), target
    return None


def simulate_catchup(
    eth_1m: pd.DataFrame,
    events: pd.DataFrame,
    *,
    target_r: float,
    direction: int,
    split: str,
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    config: R22Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R22Config()).validate()
    if direction not in (-1, 1) or target_r not in cfg.target_rs:
        raise ValueError("invalid R22 direction/target")
    bars = normalize_1m_bars(eth_1m)
    candidates = events.loc[
        events["trade_direction"].eq(direction)
        & pd.to_datetime(events["signal_bar_time"]).ge(split_start)
        & pd.to_datetime(events["signal_bar_time"]).lt(split_end)
    ].sort_values("signal_available_time")
    rows: list[dict[str, object]] = []
    sleeve_available = split_start
    ordinal = 0
    for event in candidates.itertuples(index=False):
        entry_time = pd.Timestamp(event.signal_available_time)
        if entry_time < sleeve_available or entry_time >= split_end or entry_time not in bars.index:
            continue
        entry_price = float(bars.loc[entry_time, "open"])
        risk = cfg.stop_atr * float(event.eth_atr20)
        if not np.isfinite(risk) or risk <= EPS:
            continue
        stop = entry_price - direction * risk
        target = entry_price + direction * float(target_r) * risk
        deadline = entry_time + pd.Timedelta(hours=cfg.max_hold_hours)
        path_end = min(deadline, split_end)
        passage = _first_passage(
            bars, entry_time=entry_time, deadline=path_end, direction=direction, stop=stop, target=target
        )
        if passage is not None:
            outcome, exit_time, exit_price = passage
            path_status = "included"
            sleeve_available = exit_time + pd.Timedelta(minutes=1)
        elif deadline < split_end and deadline in bars.index:
            outcome, exit_time, exit_price = "TIME_EXIT", deadline, float(bars.loc[deadline, "open"])
            path_status = "included"
            sleeve_available = deadline
        else:
            outcome, exit_time, exit_price = "SPLIT_BOUNDARY_CENSORED", pd.NaT, np.nan
            path_status = "boundary_censored"
            sleeve_available = split_end
        ordinal += 1
        gross = direction * (exit_price / entry_price - 1.0) if np.isfinite(exit_price) else np.nan
        row = {
            "trade_id": f"R22_R{int(target_r)}_{split}_{'LONG' if direction > 0 else 'SHORT'}_{ordinal:04d}",
            "event_id": event.event_id,
            "target_r": float(target_r),
            "research_split": split,
            "direction": "Long" if direction > 0 else "Short",
            "trade_direction": direction,
            "signal_bar_time": event.signal_bar_time,
            "signal_available_time": event.signal_available_time,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "beta_prior": event.beta_prior,
            "btc_return_1h": event.btc_return_1h,
            "eth_return_1h": event.eth_return_1h,
            "btc_impulse_z": event.btc_impulse_z,
            "lag_z": event.lag_z,
            "eth_atr20": event.eth_atr20,
            "risk_distance_pct": risk / entry_price,
            "stop_price": stop,
            "target_price": target,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "exit_reason": outcome,
            "path_status": path_status,
            "holding_hours": float((path_end - entry_time) / pd.Timedelta(hours=1)) if pd.isna(exit_time) else float((exit_time - entry_time) / pd.Timedelta(hours=1)),
            "gross_return": gross,
        }
        for scale in cfg.cost_scales:
            row[f"net_return_cost{int(scale)}x"] = gross - scale * cfg.market_roundtrip_cost if np.isfinite(gross) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_r22(trades: pd.DataFrame, *, config: R22Config | None = None) -> pd.DataFrame:
    cfg = (config or R22Config()).validate()
    closed = trades.loc[trades["path_status"].eq("included")].copy()
    rows = []
    for (target_r, split, direction), part in closed.groupby(["target_r", "research_split", "direction"], sort=True):
        net2 = pd.to_numeric(part["net_return_cost2x"], errors="coerce")
        top = net2.sort_values(ascending=False)
        without5 = net2.drop(index=top.head(5).index)
        without10 = net2.drop(index=top.head(10).index)
        entries = pd.to_datetime(part["entry_time"]).sort_values()
        months = 24 if split == "discovery" else 6
        rows.append({
            "target_r": target_r, "research_split": split, "direction": direction,
            "trades": len(part), "trades_per_month": len(part) / months,
            "win_rate": float(pd.to_numeric(part["gross_return"], errors="coerce").gt(0).mean()),
            "gross_pf": _pf(part["gross_return"]), "mean_gross_return": float(pd.to_numeric(part["gross_return"], errors="coerce").mean()),
            "net_pf_cost1x": _pf(part["net_return_cost1x"]), "mean_net_return_cost1x": float(pd.to_numeric(part["net_return_cost1x"], errors="coerce").mean()),
            "net_pf_cost2x": _pf(net2), "mean_net_return_cost2x": float(net2.mean()),
            "net_pf_cost3x": _pf(part["net_return_cost3x"]), "mean_net_return_cost3x": float(pd.to_numeric(part["net_return_cost3x"], errors="coerce").mean()),
            "timeout_rate": float(part["exit_reason"].eq("TIME_EXIT").mean()),
            "median_holding_hours": float(pd.to_numeric(part["holding_hours"], errors="coerce").median()),
            "median_risk_distance_pct": float(pd.to_numeric(part["risk_distance_pct"], errors="coerce").median()),
            "longest_entry_gap_days": float(entries.diff().max() / pd.Timedelta(days=1)) if len(entries) > 1 else np.nan,
            "net_pf_cost2x_top5_removed": _pf(without5), "net_sum_cost2x_top5_removed": float(without5.sum()),
            "net_pf_cost2x_top10_removed": _pf(without10), "net_sum_cost2x_top10_removed": float(without10.sum()),
        })
    return pd.DataFrame(rows)


def summarize_r22_years(trades: pd.DataFrame) -> pd.DataFrame:
    closed = trades.loc[trades["path_status"].eq("included")].copy()
    closed["year"] = pd.to_datetime(closed["exit_time"]).dt.year
    rows = []
    for (target_r, direction, year), part in closed.groupby(["target_r", "direction", "year"], sort=True):
        net2 = pd.to_numeric(part["net_return_cost2x"], errors="coerce")
        rows.append({"target_r": target_r, "direction": direction, "year": int(year), "trades": len(part), "net_pf_cost2x": _pf(net2), "mean_net_return_cost2x": float(net2.mean()), "net_sum_cost2x": float(net2.sum())})
    return pd.DataFrame(rows)


def build_r22_gate(score: pd.DataFrame, years: pd.DataFrame) -> pd.DataFrame:
    rows = []
    primary = score.loc[score["target_r"].eq(1.0)]
    sensitivity = score.loc[score["target_r"].eq(2.0)]
    for direction in ("Long", "Short"):
        reasons: list[str] = []
        direction_primary = primary.loc[primary["direction"].eq(direction)].set_index("research_split")
        for split, minimum in (("discovery", 50), ("validation", 10)):
            if split not in direction_primary.index:
                reasons.append(f"missing_{split}")
                continue
            row = direction_primary.loc[split]
            if int(row["trades"]) < minimum:
                reasons.append(f"{split}_sample")
            if float(row["net_pf_cost2x"]) < 1.4:
                reasons.append(f"{split}_pf2x")
            if float(row["mean_net_return_cost2x"]) <= 0:
                reasons.append(f"{split}_expectancy")
            if float(row["timeout_rate"]) > 0.20:
                reasons.append(f"{split}_timeout")
        if "discovery" in direction_primary.index and float(direction_primary.loc["discovery", "net_sum_cost2x_top10_removed"]) <= 0:
            reasons.append("discovery_top10")
        year_part = years.loc[years["target_r"].eq(1.0) & years["direction"].eq(direction)].set_index("year")
        for year in (2023, 2024, 2025):
            if year not in year_part.index or float(year_part.loc[year, "net_sum_cost2x"]) <= 0:
                reasons.append(f"year_{year}")
        sensitivity_discovery = sensitivity.loc[
            sensitivity["direction"].eq(direction) & sensitivity["research_split"].eq("discovery")
        ]
        if sensitivity_discovery.empty or float(sensitivity_discovery.iloc[0]["mean_net_return_cost2x"]) <= 0:
            reasons.append("r2_discovery")
        rows.append({
            "direction": direction,
            "research_candidate": int(not reasons),
            "reason": "PASS" if not reasons else "FAIL_" + ",".join(reasons),
        })
    return pd.DataFrame(rows)


def r22_causal_audit(trades: pd.DataFrame, *, config: R22Config | None = None) -> pd.DataFrame:
    cfg = (config or R22Config()).validate()
    expected_stop = pd.to_numeric(trades["entry_price"]) - pd.to_numeric(trades["trade_direction"]) * cfg.stop_atr * pd.to_numeric(trades["eth_atr20"])
    expected_target = pd.to_numeric(trades["entry_price"]) + pd.to_numeric(trades["trade_direction"]) * pd.to_numeric(trades["target_r"]) * cfg.stop_atr * pd.to_numeric(trades["eth_atr20"])
    closed = trades.loc[trades["path_status"].eq("included")]
    rows = [
        {"check": "unique_trade_id", "violations": int(trades["trade_id"].duplicated().sum())},
        {"check": "entry_after_closed_hour", "violations": int((pd.to_datetime(trades["entry_time"]) != pd.to_datetime(trades["signal_bar_time"]) + pd.Timedelta(hours=1)).sum())},
        {"check": "entry_matches_available_time", "violations": int((pd.to_datetime(trades["entry_time"]) != pd.to_datetime(trades["signal_available_time"])).sum())},
        {"check": "finite_prior_features", "violations": int((~np.isfinite(trades[["beta_prior", "btc_impulse_z", "lag_z", "eth_atr20"]]).all(axis=1)).sum())},
        {"check": "impulse_threshold", "violations": int(pd.to_numeric(trades["btc_impulse_z"]).abs().lt(cfg.impulse_z).sum())},
        {"check": "lag_threshold", "violations": int(pd.to_numeric(trades["lag_z"]).lt(cfg.lag_z).sum())},
        {"check": "stop_formula", "violations": int((pd.to_numeric(trades["stop_price"]) - expected_stop).abs().gt(1e-10).sum())},
        {"check": "target_formula", "violations": int((pd.to_numeric(trades["target_price"]) - expected_target).abs().gt(1e-10).sum())},
        {"check": "exit_not_before_entry", "violations": int((pd.to_datetime(closed["exit_time"]) < pd.to_datetime(closed["entry_time"])).sum())},
        {"check": "discovery_boundary", "violations": int((closed["research_split"].eq("discovery") & pd.to_datetime(closed["exit_time"]).ge(cfg.validation_start)).sum())},
        {"check": "validation_boundary", "violations": int((closed["research_split"].eq("validation") & pd.to_datetime(closed["exit_time"]).ge(cfg.embargo_start)).sum())},
        {"check": "embargo_holdout_absent", "violations": int(trades["research_split"].isin(["embargo", "holdout"]).sum())},
    ]
    for scale in cfg.cost_scales:
        expected = pd.to_numeric(trades["gross_return"], errors="coerce") - scale * cfg.market_roundtrip_cost
        actual = pd.to_numeric(trades[f"net_return_cost{int(scale)}x"], errors="coerce")
        rows.append({"check": f"cost{int(scale)}x_formula", "violations": int((actual - expected).abs().dropna().gt(1e-12).sum())})
    out = pd.DataFrame(rows)
    out["status"] = np.where(out["violations"].eq(0), "PASS", "FAIL")
    return out
