#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal strategy replay and metrics for the R02 environment lab.

Minute OHLC arrays are converted once per chunk and shared by every candidate
and stress scenario.  Intrabar ambiguity is pessimistic and trailing changes
become active only on the next bar.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.research_common.progress import ProgressReporter

@dataclass(frozen=True)
class Scenario:
    name: str
    delay_bars: int = 0
    fee_multiple: float = 1.0
    slippage_per_side: float = 0.0


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("base"),
    Scenario("fee_2x", fee_multiple=2.0),
    Scenario("delay_1m", delay_bars=1),
    Scenario("delay_3m", delay_bars=3),
    Scenario("slip_2bps", slippage_per_side=0.0002),
)


@dataclass(frozen=True)
class LabConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "1m"
    start: str = "2023-01-01"
    end: str = "2026-06-30 23:59:59"
    range_pct: float = 0.0020
    baseline_window: int = 240
    warmup_days: int = 8
    tail_minutes: int = 245
    round_trip_cost: float = 0.0011
    risk_fraction: float = 0.01
    max_notional_multiple: float = 2.0
    min_initial_risk_pct: float = 0.0015
    max_initial_risk_pct: float = 0.0150
    cooldown_minutes: int = 15


@dataclass(frozen=True)
class FamilyExit:
    max_hold_bars: int
    target_r: float
    arm_r_1: float
    lock_r_1: float
    arm_r_2: float
    lock_r_2: float




@dataclass(frozen=True)
class BarArrays:
    index: pd.DatetimeIndex
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray


def build_bar_arrays(bars: pd.DataFrame) -> BarArrays:
    """Convert a chunk to shared contiguous replay arrays exactly once."""
    return BarArrays(
        index=pd.DatetimeIndex(bars.index),
        opens=np.ascontiguousarray(pd.to_numeric(bars["open"], errors="coerce").to_numpy(float)),
        highs=np.ascontiguousarray(pd.to_numeric(bars["high"], errors="coerce").to_numpy(float)),
        lows=np.ascontiguousarray(pd.to_numeric(bars["low"], errors="coerce").to_numpy(float)),
        closes=np.ascontiguousarray(pd.to_numeric(bars["close"], errors="coerce").to_numpy(float)),
    )

FAMILY_EXITS: Mapping[str, FamilyExit] = {
    "compression_breakout": FamilyExit(180, 1.80, 1.00, 0.00, 1.50, 0.50),
    "expansion_exhaustion": FamilyExit(120, 1.50, 0.90, 0.00, 1.30, 0.35),
    "balance_failed_auction": FamilyExit(180, 1.20, 0.80, 0.00, 1.10, 0.30),
}



def enforce_nonoverlap(trades: pd.DataFrame) -> pd.DataFrame:
    """Enforce sleeve-level non-overlap across yearly chunk boundaries."""
    if trades.empty:
        return trades.copy()
    work = trades.sort_values(
        ["scenario", "definition", "family", "entry_time", "signal_time"],
        kind="stable",
    )
    kept: list[int] = []
    for _, group in work.groupby(["scenario", "definition", "family"], sort=False):
        next_free = pd.Timestamp.min
        for idx, entry, exit_time in zip(
            group.index,
            pd.to_datetime(group["entry_time"]),
            pd.to_datetime(group["exit_time"]),
        ):
            if entry > next_free:
                kept.append(int(idx))
                next_free = exit_time
    return work.loc[sorted(kept)].reset_index(drop=True)


def _adverse_fill(price: float, side: int, *, is_entry: bool, slippage: float) -> float:
    if slippage <= 0:
        return float(price)
    sign = side if is_entry else -side
    return float(price) * (1.0 + sign * float(slippage))


def _resolve_target(
    family: str,
    side: int,
    entry: float,
    risk: float,
    target_reference: float,
    exit_cfg: FamilyExit,
) -> float:
    fixed = entry + side * exit_cfg.target_r * risk
    if not np.isfinite(target_reference):
        return fixed
    if family == "compression_breakout":
        return fixed
    if family == "expansion_exhaustion":
        structural = target_reference
        if side == 1 and structural > entry:
            return min(max(fixed, structural), entry + 2.20 * risk)
        if side == -1 and structural < entry:
            return max(min(fixed, structural), entry - 2.20 * risk)
        return fixed
    # A failed auction should return toward the balance midpoint.  Skip logic
    # later rejects a midpoint that does not offer at least 0.8R.
    return float(target_reference)


def simulate_candidate(
    candidate: Mapping[str, object],
    market: BarArrays,
    scenario: Scenario,
    cfg: LabConfig,
) -> dict[str, object] | None:
    """Replay one trade with pessimistic intrabar ambiguity handling.

    The signal bar is closed.  Entry is the next bar open plus the scenario
    delay.  Trailing-stop changes derived from a bar become active only on the
    following bar.  If stop and target are both touched in one bar, the stop is
    assumed first.
    """
    idx = market.index
    signal_time = pd.Timestamp(candidate["signal_time"])
    signal_pos_value = candidate.get("_signal_pos")
    signal_pos = int(signal_pos_value) if signal_pos_value is not None else int(idx.searchsorted(signal_time, side="left"))
    if signal_pos < 0 or signal_pos >= len(idx) or idx[signal_pos] != signal_time:
        return None
    entry_pos = signal_pos + 1 + int(scenario.delay_bars)
    if entry_pos >= len(idx):
        return None

    opens = market.opens
    highs = market.highs
    lows = market.lows
    closes = market.closes
    side = int(candidate["side"])
    raw_entry = float(opens[entry_pos])
    structural_stop = float(candidate["structural_stop"])
    if not np.isfinite(raw_entry) or raw_entry <= 0 or not np.isfinite(structural_stop):
        return None
    risk = (raw_entry - structural_stop) * side
    risk_pct = risk / raw_entry
    if not np.isfinite(risk_pct) or not (cfg.min_initial_risk_pct <= risk_pct <= cfg.max_initial_risk_pct):
        return None

    family = str(candidate["family"])
    exit_cfg = FAMILY_EXITS[family]
    target = _resolve_target(
        family,
        side,
        raw_entry,
        risk,
        float(candidate.get("target_reference", np.nan)),
        exit_cfg,
    )
    reward_r = ((target - raw_entry) * side) / risk
    if not np.isfinite(reward_r) or reward_r < 0.80 or reward_r > 3.00:
        return None

    current_stop = structural_stop
    max_favourable_r = 0.0
    exit_pos: int | None = None
    raw_exit = np.nan
    exit_reason = ""
    ambiguous = False
    end_pos = min(len(idx) - 1, entry_pos + exit_cfg.max_hold_bars)

    for pos in range(entry_pos, end_pos + 1):
        if pos - entry_pos >= exit_cfg.max_hold_bars:
            exit_pos = pos
            raw_exit = float(opens[pos])
            exit_reason = "time_fail_safe_open"
            break

        if side == 1:
            stop_hit = lows[pos] <= current_stop
            target_hit = highs[pos] >= target
            favourable = (highs[pos] - raw_entry) / risk
        else:
            stop_hit = highs[pos] >= current_stop
            target_hit = lows[pos] <= target
            favourable = (raw_entry - lows[pos]) / risk

        if stop_hit and target_hit:
            ambiguous = True
            exit_pos = pos
            raw_exit = current_stop
            exit_reason = "same_bar_stop_first"
            break
        if stop_hit:
            exit_pos = pos
            raw_exit = current_stop
            exit_reason = "structural_or_trailing_stop"
            break
        if target_hit:
            exit_pos = pos
            raw_exit = target
            exit_reason = "mechanism_target"
            break

        max_favourable_r = max(max_favourable_r, float(favourable))
        # Updates below become active on the next bar only.
        new_stop = current_stop
        if max_favourable_r >= exit_cfg.arm_r_1:
            new_stop = raw_entry + side * exit_cfg.lock_r_1 * risk
        if max_favourable_r >= exit_cfg.arm_r_2:
            stronger = raw_entry + side * exit_cfg.lock_r_2 * risk
            new_stop = max(new_stop, stronger) if side == 1 else min(new_stop, stronger)
        current_stop = max(current_stop, new_stop) if side == 1 else min(current_stop, new_stop)

    if exit_pos is None:
        exit_pos = end_pos
        raw_exit = float(closes[end_pos])
        exit_reason = "data_end_close"

    entry_fill = _adverse_fill(raw_entry, side, is_entry=True, slippage=scenario.slippage_per_side)
    exit_fill = _adverse_fill(raw_exit, side, is_entry=False, slippage=scenario.slippage_per_side)
    gross_return = (exit_fill / entry_fill - 1.0) * side
    net_return = gross_return - cfg.round_trip_cost * scenario.fee_multiple
    net_r = net_return / risk_pct
    notional_multiple = min(cfg.max_notional_multiple, cfg.risk_fraction / risk_pct)
    equity_return = net_return * notional_multiple

    exported_candidate = {key: value for key, value in candidate.items() if not str(key).startswith("_")}
    return {
        **exported_candidate,
        "scenario": scenario.name,
        "entry_time": idx[entry_pos],
        "entry_price_raw": raw_entry,
        "entry_price_fill": entry_fill,
        "exit_time": idx[exit_pos],
        "exit_price_raw": raw_exit,
        "exit_price_fill": exit_fill,
        "exit_reason": exit_reason,
        "initial_stop": structural_stop,
        "final_stop": current_stop,
        "target_price": target,
        "risk_pct": risk_pct,
        "reward_r_planned": reward_r,
        "max_favourable_r": max_favourable_r,
        "bars_held": int(exit_pos - entry_pos),
        "gross_return": gross_return,
        "net_return": net_return,
        "net_r": net_r,
        "notional_multiple": notional_multiple,
        "equity_return": equity_return,
        "same_bar_ambiguous": ambiguous,
        "causal_entry": idx[entry_pos] > signal_time,
        "range_context_causal": bool(candidate.get("range_context_causal", False)),
    }


def simulate_candidates(
    candidates: pd.DataFrame,
    bars: pd.DataFrame,
    scenarios: Iterable[Scenario],
    cfg: LabConfig,
    *,
    progress_label: str,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()
    scenarios_list = list(scenarios)
    market = build_bar_arrays(bars)
    prepared = candidates.copy()
    prepared["_signal_pos"] = market.index.get_indexer(pd.DatetimeIndex(prepared["signal_time"]))
    records = prepared.to_dict(orient="records")
    total = len(records) * len(scenarios_list)
    reporter = ProgressReporter(progress_label, total=total, every=max(1, total // 50))
    rows: list[dict[str, object]] = []
    done = 0
    # Each family/definition/scenario is an independent sleeve.  Overlapping
    # signals inside the same sleeve are skipped until the previous trade exits.
    next_free: dict[tuple[str, str, str], pd.Timestamp] = {}
    for scenario in scenarios_list:
        for candidate in records:
            done += 1
            key = (str(candidate["definition"]), str(candidate["family"]), scenario.name)
            signal_time = pd.Timestamp(candidate["signal_time"])
            if signal_time < next_free.get(key, pd.Timestamp.min):
                reporter.update(done)
                continue
            trade = simulate_candidate(candidate, market, scenario, cfg)
            if trade is not None:
                rows.append(trade)
                next_free[key] = pd.Timestamp(trade["exit_time"])
            reporter.update(done)
    reporter.close()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["scenario", "definition", "family", "entry_time"], kind="stable")


def _profit_factor(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gains = x[x > 0].sum()
    losses = -x[x < 0].sum()
    if losses <= 0:
        return math.inf if gains > 0 else math.nan
    return float(gains / losses)


def _equity_metrics(frame: pd.DataFrame) -> tuple[float, float]:
    if frame.empty:
        return np.nan, np.nan
    returns = pd.to_numeric(frame.sort_values("entry_time")["equity_return"], errors="coerce").fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(equity.iloc[-1] - 1.0), float(drawdown.min())


def _remove_top_n_return(frame: pd.DataFrame, n: int = 10) -> float:
    if frame.empty:
        return np.nan
    trimmed = frame.drop(frame.nlargest(min(n, len(frame)), "equity_return").index)
    if trimmed.empty:
        return 0.0
    return float((1.0 + pd.to_numeric(trimmed["equity_return"], errors="coerce").fillna(0.0)).prod() - 1.0)


def summarize_trades(trades: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame()
    grouped = trades.groupby(group_cols, dropna=False, sort=True)
    for keys, frame in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {name: value for name, value in zip(group_cols, keys)}
        equity_return, max_drawdown = _equity_metrics(frame)
        ret = pd.to_numeric(frame["equity_return"], errors="coerce").dropna()
        net_r = pd.to_numeric(frame["net_r"], errors="coerce").dropna()
        months = max(1, pd.to_datetime(frame["entry_time"]).dt.to_period("M").nunique())
        monthly = (
            frame.assign(month=pd.to_datetime(frame["entry_time"]).dt.to_period("M"))
            .groupby("month")["equity_return"]
            .apply(lambda x: float((1.0 + pd.to_numeric(x, errors="coerce").fillna(0.0)).prod() - 1.0))
        )
        row.update(
            {
                "trades": int(len(frame)),
                "trades_per_month": float(len(frame) / months),
                "win_rate": float((ret > 0).mean()) if len(ret) else np.nan,
                "mean_equity_return": float(ret.mean()) if len(ret) else np.nan,
                "median_equity_return": float(ret.median()) if len(ret) else np.nan,
                "mean_net_r": float(net_r.mean()) if len(net_r) else np.nan,
                "profit_factor": _profit_factor(ret),
                "total_return": equity_return,
                "max_drawdown": max_drawdown,
                "positive_month_rate": float((monthly > 0).mean()) if len(monthly) else np.nan,
                "return_without_top10": _remove_top_n_return(frame, 10),
                "same_bar_ambiguous_rate": float(frame["same_bar_ambiguous"].mean()),
                "causal_entry_rate": float(frame["causal_entry"].mean()),
                "causal_range_rate": float(frame["range_context_causal"].mean()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_period_tables(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    out = trades.copy()
    ts = pd.to_datetime(out["entry_time"])
    out["year"] = ts.dt.year
    out["quarter"] = ts.dt.to_period("Q").astype(str)
    out["month"] = ts.dt.to_period("M").astype(str)
    yearly = summarize_trades(out, ["scenario", "definition", "family", "year"])
    quarterly = summarize_trades(out, ["scenario", "definition", "family", "quarter"])
    monthly = summarize_trades(out, ["scenario", "definition", "family", "month"])
    return yearly, quarterly, monthly


def _promotion_table(overview: pd.DataFrame, yearly: pd.DataFrame) -> pd.DataFrame:
    if overview.empty:
        return pd.DataFrame()
    primary = overview[(overview["scenario"] == "base") & (overview["definition"] == "base")].copy()
    fee = overview[(overview["scenario"] == "fee_2x") & (overview["definition"] == "base")][
        ["family", "profit_factor", "total_return"]
    ].rename(columns={"profit_factor": "fee2x_pf", "total_return": "fee2x_return"})
    delay = overview[(overview["scenario"] == "delay_1m") & (overview["definition"] == "base")][
        ["family", "profit_factor", "total_return"]
    ].rename(columns={"profit_factor": "delay1_pf", "total_return": "delay1_return"})
    holdout = yearly[
        (yearly["scenario"] == "base")
        & (yearly["definition"] == "base")
        & (yearly["year"] == 2026)
    ][["family", "profit_factor", "total_return", "trades"]].rename(
        columns={"profit_factor": "holdout_2026_pf", "total_return": "holdout_2026_return", "trades": "holdout_2026_trades"}
    )
    result = primary.merge(fee, on="family", how="left").merge(delay, on="family", how="left").merge(holdout, on="family", how="left")
    result["screen_pass"] = (
        (result["profit_factor"] >= 1.15)
        & (result["total_return"] > 0)
        & (result["return_without_top10"] > -0.02)
        & (result["fee2x_return"] > 0)
        & (result["delay1_return"] > 0)
        & (result["holdout_2026_return"] > 0)
        & (result["trades"] >= 120)
    )
    return result


