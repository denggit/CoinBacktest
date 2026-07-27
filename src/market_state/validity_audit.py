#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal forward-path audit for Market State Map labels and transitions.

This module does not fit a predictor and does not create an executable trading
strategy.  It asks a narrower question: after a state becomes observable, does
the subsequent path differ from an appropriate baseline in a stable way?

All event entries use the next bar open.  Current-bar high/low/close and all
future path values are used only as post-event labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


_EPS = 1e-12


@dataclass(frozen=True)
class ValidityAuditConfig:
    horizons_bars: tuple[int, ...] = (5, 15, 30, 60, 180)
    event_cooldown_bars: int = 5
    transition_lookback_bars: int = 15
    trap_horizon_bars: int = 60
    trap_tolerance_atr: float = 0.25
    round_trip_cost: float = 0.0011
    minimum_events: int = 80
    holdout_start: str | None = "2025-07-01"

    def validate(self) -> None:
        if not self.horizons_bars:
            raise ValueError("horizons_bars cannot be empty")
        if any(int(h) < 1 for h in self.horizons_bars):
            raise ValueError("all horizons_bars must be >= 1")
        if self.event_cooldown_bars < 0:
            raise ValueError("event_cooldown_bars must be >= 0")
        if self.transition_lookback_bars < 1:
            raise ValueError("transition_lookback_bars must be >= 1")
        if self.trap_horizon_bars < 1:
            raise ValueError("trap_horizon_bars must be >= 1")
        if self.trap_tolerance_atr < 0:
            raise ValueError("trap_tolerance_atr must be >= 0")
        if self.round_trip_cost < 0:
            raise ValueError("round_trip_cost must be >= 0")
        if self.minimum_events < 1:
            raise ValueError("minimum_events must be >= 1")


@dataclass(frozen=True)
class EventDefinition:
    event_name: str
    direction: int
    mask: pd.Series
    family: str
    description: str


DIRECTION_LABEL = {1: "long", -1: "short", 0: "neutral"}


def _state_start(series: pd.Series, value: str) -> pd.Series:
    current = series.astype(str)
    return current.eq(value) & current.shift(1).ne(value)


def _recently(series: pd.Series, value: str, lookback: int) -> pd.Series:
    hit = series.astype(str).eq(value).shift(1, fill_value=False)
    return hit.rolling(int(lookback), min_periods=1).max().fillna(0.0).astype(bool)


def _apply_cooldown(mask: pd.Series, cooldown: int) -> pd.Series:
    values = mask.fillna(False).to_numpy(dtype=bool)
    positions = np.flatnonzero(values)
    if cooldown <= 0 or positions.size <= 1:
        return pd.Series(values, index=mask.index, dtype=bool)
    keep = np.zeros_like(values, dtype=bool)
    last = -10**18
    for pos in positions:
        if int(pos) - int(last) >= int(cooldown):
            keep[pos] = True
            last = int(pos)
    return pd.Series(keep, index=mask.index, dtype=bool)


def build_event_definitions(
    frame: pd.DataFrame,
    *,
    transition_lookback_bars: int = 15,
    event_cooldown_bars: int = 5,
) -> list[EventDefinition]:
    """Build deterministic event-start and transition masks.

    Event definitions are fixed before looking at forward returns.  They are
    deliberately broad so the audit can reject weak labels instead of tuning
    around individual losing examples.
    """

    required = {
        "trend_state",
        "flow_state",
        "impact_state",
        "location_state",
        "trade_context_state",
        "data_ready",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"market-state frame missing columns: {missing}")

    ready = frame["data_ready"].fillna(False).astype(bool)
    flow_ready = frame.get("orderflow_available", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    location_ready = frame.get("location_available", pd.Series(False, index=frame.index)).fillna(False).astype(bool)

    specs: list[tuple[str, int, pd.Series, str, str]] = [
        ("trend_up_start", 1, _state_start(frame["trend_state"], "up") & ready, "trend", "稳定上涨状态首次确认"),
        ("trend_down_start", -1, _state_start(frame["trend_state"], "down") & ready, "trend", "稳定下跌状态首次确认"),
        ("flow_buy_building_start", 1, _state_start(frame["flow_state"], "buy_building") & flow_ready, "orderflow", "主动买压开始增强"),
        ("flow_sell_building_start", -1, _state_start(frame["flow_state"], "sell_building") & flow_ready, "orderflow", "主动卖压开始增强"),
        ("flow_buy_persistent_start", 1, _state_start(frame["flow_state"], "buy_persistent") & flow_ready, "orderflow", "主动买压转为持续"),
        ("flow_sell_persistent_start", -1, _state_start(frame["flow_state"], "sell_persistent") & flow_ready, "orderflow", "主动卖压转为持续"),
        ("impact_buy_effective_start", 1, _state_start(frame["impact_state"], "buy_effective") & flow_ready, "impact", "主动买入开始有效推动价格"),
        ("impact_sell_effective_start", -1, _state_start(frame["impact_state"], "sell_effective") & flow_ready, "impact", "主动卖出开始有效推动价格"),
        ("impact_sell_absorbed_start", 1, _state_start(frame["impact_state"], "sell_absorbed") & flow_ready, "absorption", "卖压开始被吸收，多头反转上下文"),
        ("impact_buy_absorbed_start", -1, _state_start(frame["impact_state"], "buy_absorbed") & flow_ready, "absorption", "买压开始被吸收，空头反转上下文"),
        ("location_downside_sweep_reclaim", 1, _state_start(frame["location_state"], "downside_sweep_reclaim") & location_ready, "location", "向下扫过已知支撑后收回"),
        ("location_upside_sweep_reject", -1, _state_start(frame["location_state"], "upside_sweep_reject") & location_ready, "location", "向上扫过已知阻力后拒绝"),
        ("location_breakout_accept", 1, _state_start(frame["location_state"], "breakout_accept") & location_ready, "location", "突破已知阻力并收盘接受"),
        ("location_breakdown_accept", -1, _state_start(frame["location_state"], "breakdown_accept") & location_ready, "location", "跌破已知支撑并收盘接受"),
        ("context_long_reversal_watch", 1, _state_start(frame["trade_context_state"], "long_reversal_watch") & ready, "context", "卖压吸收与低位位置共振"),
        ("context_short_reversal_watch", -1, _state_start(frame["trade_context_state"], "short_reversal_watch") & ready, "context", "买压吸收与高位位置共振"),
        ("context_long_continuation_watch", 1, _state_start(frame["trade_context_state"], "long_continuation_watch") & ready, "context", "上涨背景、买压持续且价格响应有效"),
        ("context_short_continuation_watch", -1, _state_start(frame["trade_context_state"], "short_continuation_watch") & ready, "context", "下跌背景、卖压持续且价格响应有效"),
    ]

    sell_effective_recent = _recently(frame["impact_state"], "sell_effective", transition_lookback_bars)
    buy_effective_recent = _recently(frame["impact_state"], "buy_effective", transition_lookback_bars)
    specs.extend(
        [
            (
                "transition_sell_effective_to_absorbed",
                1,
                _state_start(frame["impact_state"], "sell_absorbed") & sell_effective_recent & flow_ready,
                "transition",
                "最近卖压有效，随后转为卖压吸收",
            ),
            (
                "transition_buy_effective_to_absorbed",
                -1,
                _state_start(frame["impact_state"], "buy_absorbed") & buy_effective_recent & flow_ready,
                "transition",
                "最近买压有效，随后转为买压吸收",
            ),
        ]
    )

    definitions: list[EventDefinition] = []
    for name, direction, mask, family, description in specs:
        definitions.append(
            EventDefinition(
                event_name=name,
                direction=direction,
                mask=_apply_cooldown(mask, event_cooldown_bars),
                family=family,
                description=description,
            )
        )
    return definitions


def _forward_window_extreme(series: pd.Series, horizon: int, method: str) -> pd.Series:
    shifted = pd.to_numeric(series, errors="coerce").shift(-1)
    reversed_series = shifted.iloc[::-1]
    roller = reversed_series.rolling(int(horizon), min_periods=int(horizon))
    if method == "max":
        out = roller.max()
    elif method == "min":
        out = roller.min()
    else:
        raise ValueError("method must be max or min")
    return out.iloc[::-1]


def build_forward_path_frame(
    frame: pd.DataFrame,
    config: ValidityAuditConfig,
) -> pd.DataFrame:
    """Attach next-open causal entry labels and future path outcomes."""

    config.validate()
    required = {"open", "high", "low", "close", "available_time", "volatility_state"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"market-state frame missing path columns: {missing}")

    out = frame.copy()
    open_ = pd.to_numeric(out["open"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    out["entry_time"] = pd.Series(out.index, index=out.index).shift(-1)
    out["entry_price"] = open_.shift(-1)
    out["signal_year"] = pd.DatetimeIndex(out["available_time"]).year
    out["signal_month"] = pd.DatetimeIndex(out["available_time"]).to_period("M").astype(str)

    atr_pct = pd.to_numeric(out.get("atr_pct", np.nan), errors="coerce")
    for horizon in sorted(set((*config.horizons_bars, config.trap_horizon_bars))):
        exit_close = close.shift(-int(horizon))
        future_high = _forward_window_extreme(high, int(horizon), "max")
        future_low = _forward_window_extreme(low, int(horizon), "min")
        entry = out["entry_price"]

        out[f"exit_time_h{horizon}"] = pd.Series(out.index, index=out.index).shift(-int(horizon))
        out[f"exit_close_h{horizon}"] = exit_close
        out[f"long_return_h{horizon}"] = exit_close / entry - 1.0
        out[f"short_return_h{horizon}"] = (entry - exit_close) / entry
        out[f"long_mfe_h{horizon}"] = future_high / entry - 1.0
        out[f"long_mae_h{horizon}"] = future_low / entry - 1.0
        out[f"short_mfe_h{horizon}"] = (entry - future_low) / entry
        out[f"short_mae_h{horizon}"] = (entry - future_high) / entry

        tolerance = (config.trap_tolerance_atr * atr_pct).clip(lower=0.0)
        out[f"long_trap_h{horizon}"] = (
            ((future_high - entry) / entry <= tolerance)
            & (out[f"long_return_h{horizon}"] < 0.0)
        )
        out[f"short_trap_h{horizon}"] = (
            ((entry - future_low) / entry <= tolerance)
            & (out[f"short_return_h{horizon}"] < 0.0)
        )
    return out


def _profit_factor(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    gross_profit = float(clean[clean > 0.0].sum())
    gross_loss = float(-clean[clean < 0.0].sum())
    if gross_loss <= _EPS:
        return float("inf") if gross_profit > 0.0 else float("nan")
    return gross_profit / gross_loss


def _safe_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else float("nan")


def _matched_baseline_table(path_frame: pd.DataFrame, horizon: int, direction: int) -> pd.DataFrame:
    ret_col = f"{'long' if direction > 0 else 'short'}_return_h{horizon}"
    base = path_frame.loc[path_frame[ret_col].notna(), ["signal_year", "volatility_state", ret_col]].copy()
    if base.empty:
        return pd.DataFrame(columns=["signal_year", "volatility_state", "baseline_mean", "baseline_win_rate"])
    return (
        base.groupby(["signal_year", "volatility_state"], dropna=False)[ret_col]
        .agg(baseline_mean="mean", baseline_win_rate=lambda s: float((s > 0.0).mean()))
        .reset_index()
    )


def extract_event_rows(
    path_frame: pd.DataFrame,
    definitions: Iterable[EventDefinition],
    config: ValidityAuditConfig,
    *,
    profile: str = "base",
) -> pd.DataFrame:
    """Return one row per event and horizon with matched-baseline uplift."""

    records: list[pd.DataFrame] = []
    baselines: dict[tuple[int, int], pd.DataFrame] = {}
    for definition in definitions:
        positions = np.flatnonzero(definition.mask.to_numpy(dtype=bool))
        if positions.size == 0:
            continue
        selected = path_frame.iloc[positions].copy()
        for horizon in config.horizons_bars:
            direction_name = "long" if definition.direction > 0 else "short"
            ret_col = f"{direction_name}_return_h{horizon}"
            mfe_col = f"{direction_name}_mfe_h{horizon}"
            mae_col = f"{direction_name}_mae_h{horizon}"
            trap_col = f"{direction_name}_trap_h{config.trap_horizon_bars}"
            sample = selected.loc[selected[ret_col].notna()].copy()
            if sample.empty:
                continue
            key = (int(horizon), int(definition.direction))
            if key not in baselines:
                baselines[key] = _matched_baseline_table(path_frame, int(horizon), int(definition.direction))
            baseline = baselines[key]
            sample = sample.reset_index(names="signal_time")
            sample = sample.merge(baseline, on=["signal_year", "volatility_state"], how="left")
            sample["profile"] = profile
            sample["event_name"] = definition.event_name
            sample["event_family"] = definition.family
            sample["event_description"] = definition.description
            sample["direction"] = DIRECTION_LABEL[definition.direction]
            sample["horizon_bars"] = int(horizon)
            sample["exit_time"] = sample[f"exit_time_h{horizon}"]
            sample["exit_price"] = sample[f"exit_close_h{horizon}"]
            sample["gross_return"] = sample[ret_col]
            sample["net_return"] = sample[ret_col] - float(config.round_trip_cost)
            sample["mfe"] = sample[mfe_col]
            sample["mae"] = sample[mae_col]
            sample["trap_flag"] = sample[trap_col].fillna(False).astype(bool)
            sample["excess_return_vs_matched"] = sample["gross_return"] - sample["baseline_mean"]
            sample["period"] = "all"
            if config.holdout_start:
                cutoff = pd.Timestamp(config.holdout_start)
                sample["period"] = np.where(pd.to_datetime(sample["available_time"]) >= cutoff, "holdout", "pre_holdout")
            keep = [
                "profile",
                "event_name",
                "event_family",
                "event_description",
                "direction",
                "horizon_bars",
                "signal_time",
                "available_time",
                "entry_time",
                "entry_price",
                "exit_time",
                "exit_price",
                "signal_year",
                "signal_month",
                "period",
                "trend_state",
                "trend_phase",
                "volatility_state",
                "flow_state",
                "impact_state",
                "location_state",
                "trade_context_state",
                "gross_return",
                "net_return",
                "mfe",
                "mae",
                "trap_flag",
                "baseline_mean",
                "baseline_win_rate",
                "excess_return_vs_matched",
            ]
            records.append(sample[keep])
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def summarize_event_rows(event_rows: pd.DataFrame, config: ValidityAuditConfig) -> pd.DataFrame:
    if event_rows.empty:
        return pd.DataFrame()
    keys = ["profile", "event_name", "event_family", "direction", "horizon_bars"]
    grouped = event_rows.groupby(keys, dropna=False)
    summary = grouped.agg(
        events=("gross_return", "size"),
        mean_gross_return=("gross_return", "mean"),
        median_gross_return=("gross_return", "median"),
        mean_net_return=("net_return", "mean"),
        gross_win_rate=("gross_return", lambda s: float((s > 0.0).mean())),
        net_win_rate=("net_return", lambda s: float((s > 0.0).mean())),
        mean_mfe=("mfe", "mean"),
        mean_mae=("mae", "mean"),
        trap_rate=("trap_flag", "mean"),
        mean_matched_baseline=("baseline_mean", "mean"),
        mean_excess_return=("excess_return_vs_matched", "mean"),
    ).reset_index()
    pf = grouped["net_return"].apply(_profit_factor).rename("net_profit_factor").reset_index()
    summary = summary.merge(pf, on=keys, how="left")
    summary["sample_ok"] = summary["events"] >= int(config.minimum_events)
    summary["directional_value_flag"] = (
        summary["sample_ok"]
        & (summary["mean_excess_return"] > 0.0)
        & (summary["mean_net_return"] > 0.0)
        & (summary["net_profit_factor"] > 1.0)
    )
    return summary.sort_values(["horizon_bars", "mean_excess_return"], ascending=[True, False]).reset_index(drop=True)


def summarize_breakdowns(event_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if event_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["profile", "event_name", "direction", "horizon_bars"]
    yearly = (
        event_rows.groupby([*keys, "signal_year"], dropna=False)
        .agg(
            events=("gross_return", "size"),
            mean_net_return=("net_return", "mean"),
            net_win_rate=("net_return", lambda s: float((s > 0.0).mean())),
            mean_excess_return=("excess_return_vs_matched", "mean"),
            trap_rate=("trap_flag", "mean"),
        )
        .reset_index()
    )
    period = (
        event_rows.groupby([*keys, "period"], dropna=False)
        .agg(
            events=("gross_return", "size"),
            mean_net_return=("net_return", "mean"),
            net_win_rate=("net_return", lambda s: float((s > 0.0).mean())),
            mean_excess_return=("excess_return_vs_matched", "mean"),
            trap_rate=("trap_flag", "mean"),
        )
        .reset_index()
    )
    return yearly, period


def build_profile_stability(summary: pd.DataFrame, yearly: pd.DataFrame, period: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    base = summary.copy()
    year_stats = (
        yearly.groupby(["profile", "event_name", "direction", "horizon_bars"], dropna=False)
        .agg(
            years=("signal_year", "nunique"),
            positive_years=("mean_net_return", lambda s: int((s > 0.0).sum())),
            positive_excess_years=("mean_excess_return", lambda s: int((s > 0.0).sum())),
            worst_year_net=("mean_net_return", "min"),
        )
        .reset_index()
    )
    out = base.merge(year_stats, on=["profile", "event_name", "direction", "horizon_bars"], how="left")
    holdout = period.loc[period["period"].eq("holdout"), ["profile", "event_name", "direction", "horizon_bars", "events", "mean_net_return", "mean_excess_return"]].copy()
    holdout = holdout.rename(
        columns={
            "events": "holdout_events",
            "mean_net_return": "holdout_mean_net_return",
            "mean_excess_return": "holdout_mean_excess_return",
        }
    )
    out = out.merge(holdout, on=["profile", "event_name", "direction", "horizon_bars"], how="left")
    out["year_positive_ratio"] = out["positive_years"] / out["years"].replace(0, np.nan)

    neighborhood_keys = ["event_name", "direction", "horizon_bars"]
    profile_eval = out.assign(
        profile_positive=(
            out["sample_ok"]
            & (out["mean_net_return"] > 0.0)
            & (out["mean_excess_return"] > 0.0)
            & (out["net_profit_factor"] > 1.0)
        ),
        profile_holdout_positive=(
            out["holdout_events"].fillna(0).ge(20)
            & out["holdout_mean_excess_return"].fillna(-np.inf).gt(0.0)
        ),
    )
    neighborhood = (
        profile_eval.groupby(neighborhood_keys, dropna=False)
        .agg(
            profiles_tested=("profile", "nunique"),
            positive_profiles=("profile_positive", "sum"),
            holdout_positive_profiles=("profile_holdout_positive", "sum"),
        )
        .reset_index()
    )
    neighborhood["required_positive_profiles"] = np.ceil(0.67 * neighborhood["profiles_tested"]).astype(int).clip(lower=1)
    neighborhood["profile_neighborhood_ratio"] = (
        neighborhood["positive_profiles"] / neighborhood["profiles_tested"].replace(0, np.nan)
    )
    neighborhood["neighborhood_flag"] = (
        neighborhood["positive_profiles"] >= neighborhood["required_positive_profiles"]
    )
    out = out.merge(neighborhood, on=neighborhood_keys, how="left")
    out["robust_flag"] = (
        out["sample_ok"]
        & out["directional_value_flag"]
        & (out["year_positive_ratio"] >= 0.60)
        & (out["holdout_events"].fillna(0) >= 20)
        & (out["holdout_mean_excess_return"].fillna(-np.inf) > 0.0)
        & out["neighborhood_flag"].fillna(False)
    )
    return out


def build_verdict(stability: pd.DataFrame) -> Mapping[str, object]:
    if stability.empty:
        return {
            "decision": "rejected",
            "reason": "No eligible events were produced.",
            "robust_candidates": [],
        }
    robust = stability.loc[stability["robust_flag"]].copy()
    trend_rows = stability.loc[stability["event_name"].isin({"trend_up_start", "trend_down_start"})]
    trend_valid = bool(trend_rows["robust_flag"].any())
    transition_valid = bool(stability.loc[stability["event_family"].eq("transition"), "robust_flag"].any())
    context_valid = bool(stability.loc[stability["event_family"].eq("context"), "robust_flag"].any())
    if robust.empty:
        decision = "rejected"
        reason = "No state or transition passed cost, matched-baseline, yearly and holdout checks."
    elif transition_valid or context_valid:
        decision = "research_continue"
        reason = "At least one transition/context has robust forward-path separation; build a strict sequential backtest next."
    else:
        decision = "research_continue"
        reason = "Some static states show separation, but no transition/context is yet robust enough for execution logic."
    candidate_cols = [
        "profile",
        "event_name",
        "direction",
        "horizon_bars",
        "events",
        "mean_net_return",
        "mean_excess_return",
        "net_profit_factor",
        "year_positive_ratio",
        "holdout_mean_excess_return",
    ]
    candidates = robust.sort_values("mean_excess_return", ascending=False).head(20)
    return {
        "decision": decision,
        "reason": reason,
        "trend_start_direction_valid": trend_valid,
        "transition_valid": transition_valid,
        "context_valid": context_valid,
        "robust_candidate_count": int(len(robust)),
        "robust_candidates": candidates[candidate_cols].replace({np.nan: None, np.inf: None, -np.inf: None}).to_dict("records"),
    }


def build_naive_fixed_horizon_trades(
    event_rows: pd.DataFrame,
    *,
    event_names: set[str],
    horizon_bars: int,
    initial_capital: float = 10_000.0,
) -> tuple[list[dict[str, object]], float]:
    """Create non-overlapping diagnostic trades for print_full_report.

    This is explicitly a diagnostic replay, not a strategy candidate.  It uses
    fixed-horizon exits and accepts only the first event while flat.
    """

    if event_rows.empty:
        return [], initial_capital
    rows = event_rows.loc[
        event_rows["event_name"].isin(event_names)
        & event_rows["horizon_bars"].eq(int(horizon_bars))
        & event_rows["profile"].eq("base")
    ].copy()
    if rows.empty:
        return [], initial_capital
    rows = rows.sort_values(["entry_time", "signal_time", "event_name"])
    capital = float(initial_capital)
    trades: list[dict[str, object]] = []
    blocked_until = pd.Timestamp.min
    for row in rows.itertuples(index=False):
        entry_time = pd.Timestamp(row.entry_time)
        if entry_time <= blocked_until:
            continue
        exit_time = pd.Timestamp(row.exit_time)
        net_return = float(row.net_return)
        capital_before = capital
        pnl = capital_before * net_return
        fee = capital_before * 0.0011
        capital += pnl
        trades.append(
            {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "type": f"DIAGNOSTIC_{str(row.event_name)}_{str(row.direction).upper()}",
                "entry": float(row.entry_price),
                "exit": float(row.exit_price),
                "pnl": pnl,
                "fee": fee,
                "capital": capital,
                "mfe_r": float(row.mfe),
                "mae_r": float(row.mae),
            }
        )
        blocked_until = exit_time
    return trades, capital
