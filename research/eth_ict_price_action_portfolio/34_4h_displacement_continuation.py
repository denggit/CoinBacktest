#!/usr/bin/env python
"""4H price-action displacement plus OKX order-flow continuation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio import _okx_walkforward_bridge as base


RESULTS = Path(__file__).resolve().parent / "ict_pa_v21" / "results"
RANGE_LOOKBACK_BARS = 6 * 180
RANGE_QUANTILE = 0.95
CLOSE_LOCATION_GATE = 0.80
MAX_HOLD = pd.Timedelta(hours=12)
RISK_PER_TRADE = 0.005
NOTIONAL_CAP = 0.30
REWARD_RISK = 2.0
MIN_TARGET_DISTANCE = 1.5 * 2.0 * base.ONE_WAY_COST


def build_displacement_events(minute: pd.DataFrame, trade15: pd.DataFrame) -> pd.DataFrame:
    price = minute.resample("4h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        source_minutes=("close", "size"),
    )
    flow = trade15.resample("4h", label="left", closed="left").agg(
        delta_notional=("delta_notional", "sum"), notional=("notional", "sum"), source_trade_bars=("close", "size")
    )
    bars = price.join(flow, how="inner")
    bars = bars[(bars["source_minutes"] == 240) & (bars["source_trade_bars"] == 16)].copy()
    bar_range = bars["high"] / bars["low"] - 1.0
    prior_range_gate = bar_range.shift(1).rolling(RANGE_LOOKBACK_BARS, min_periods=6 * 30).quantile(RANGE_QUANTILE)
    close_location = (bars["close"] - bars["low"]) / (bars["high"] - bars["low"]).replace(0.0, np.nan)
    delta_ratio = bars["delta_notional"] / bars["notional"].replace(0.0, np.nan)
    long_event = (
        (bar_range >= prior_range_gate) & (bars["close"] > bars["open"])
        & (close_location >= CLOSE_LOCATION_GATE) & (delta_ratio > 0.0)
    )
    short_event = (
        (bar_range >= prior_range_gate) & (bars["close"] < bars["open"])
        & (close_location <= 1.0 - CLOSE_LOCATION_GATE) & (delta_ratio < 0.0)
    )
    side = pd.Series(pd.NA, index=bars.index, dtype="object")
    side.loc[long_event] = "long"
    side.loc[short_event] = "short"
    raw = pd.DataFrame(
        {
            "side": side, "bar_time": bars.index, "bar_open": bars["open"], "bar_high": bars["high"],
            "bar_low": bars["low"], "bar_close": bars["close"], "bar_range": bar_range,
            "prior_range_q95": prior_range_gate, "close_location": close_location, "delta_ratio": delta_ratio,
        }, index=bars.index
    ).dropna(subset=["side"])
    # [T,T+4H) is available only at T+4H.
    available = pd.DataFrame(raw.to_numpy(), columns=raw.columns, index=raw.index + pd.Timedelta(hours=4))
    available["available_time"] = available.index
    return available


def build_side_positions(
    events: pd.DataFrame, minute: pd.DataFrame, side: str, delay_minutes: int
) -> tuple[pd.Series, pd.DataFrame, dict[str, int]]:
    is_long = side == "long"
    candidates = events[events["side"] == side].sort_values("available_time")
    sparse: dict[pd.Timestamp, float] = {}
    trades: list[dict[str, object]] = []
    next_entry_after = pd.Timestamp.min
    audit = {"candidate_events": len(candidates), "skipped_overlap": 0, "skipped_invalid_stop": 0, "skipped_cost_distance": 0}
    for _, event in candidates.iterrows():
        entry_time = pd.Timestamp(event["available_time"]) + pd.Timedelta(minutes=delay_minutes)
        if entry_time <= next_entry_after:
            audit["skipped_overlap"] += 1
            continue
        if entry_time not in minute.index:
            continue
        entry_price = float(minute.at[entry_time, "open"])
        midpoint = 0.5 * (float(event["bar_high"]) + float(event["bar_low"]))
        stop_price = midpoint
        risk_distance = entry_price - stop_price if is_long else stop_price - entry_price
        if not np.isfinite(risk_distance) or risk_distance <= 0.0:
            audit["skipped_invalid_stop"] += 1
            continue
        stop_fraction = risk_distance / entry_price
        if REWARD_RISK * stop_fraction < MIN_TARGET_DISTANCE:
            audit["skipped_cost_distance"] += 1
            continue
        notional = min(NOTIONAL_CAP, RISK_PER_TRADE / stop_fraction)
        target_price = entry_price + REWARD_RISK * risk_distance if is_long else entry_price - REWARD_RISK * risk_distance
        deadline = min(entry_time + MAX_HOLD, minute.index.max())
        trigger_time: pd.Timestamp | None = None
        exit_reason = "time"
        ambiguous = False
        for bar_time, bar in minute.loc[entry_time:deadline].iterrows():
            stop_hit = float(bar["low"]) <= stop_price if is_long else float(bar["high"]) >= stop_price
            target_hit = float(bar["high"]) >= target_price if is_long else float(bar["low"]) <= target_price
            if stop_hit or target_hit:
                trigger_time = bar_time
                ambiguous = bool(stop_hit and target_hit)
                exit_reason = "stop" if stop_hit else "target"
                break
        exit_time = deadline if trigger_time is None else min(trigger_time + pd.Timedelta(minutes=1), minute.index.max())
        notional_signed = notional if is_long else -notional
        sparse[entry_time] = notional_signed
        sparse[exit_time] = 0.0
        exit_price = float(minute.at[exit_time, "open"])
        underlying_return = (exit_price / entry_price - 1.0) * (1.0 if is_long else -1.0)
        trades.append(
            {
                "side": side, "bar_time": event["bar_time"], "entry_time": entry_time, "exit_time": exit_time,
                "entry_price": entry_price, "exit_price": exit_price, "stop_price": stop_price, "target_price": target_price,
                "notional": notional, "underlying_return": underlying_return,
                "net_account_contribution": notional * underlying_return - 2.0 * notional * base.ONE_WAY_COST,
                "exit_reason": exit_reason, "ambiguous_bar_loss_first": ambiguous,
            }
        )
        next_entry_after = exit_time
    audit["trades"] = len(trades)
    series = pd.Series(sparse, dtype=float).sort_index()
    position = series.reindex(minute.index, method="ffill").fillna(0.0) if not series.empty else pd.Series(0.0, index=minute.index)
    return position, pd.DataFrame(trades), audit


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, trade15 = base.load_inputs()
    events = build_displacement_events(minute, trade15)
    positions = {}
    ledgers = {}
    audits = []
    for delay in (1, 2):
        for side in ("long", "short"):
            pos, ledger, audit = build_side_positions(events, minute, side, delay)
            positions[f"{side}_{delay}m"] = pos
            ledgers[f"{side}_{delay}m"] = ledger
            audits.append({"delay_minutes": delay, "side": side, **audit})
    disp1 = pd.DataFrame({"disp_long": positions["long_1m"], "disp_short": positions["short_1m"]}, index=minute.index)
    disp2 = pd.DataFrame({"disp_long": positions["long_2m"], "disp_short": positions["short_2m"]}, index=minute.index)
    core = pd.DataFrame({"core": base.core_state(minute) * 0.75}, index=minute.index)
    variants = {
        "displacement_only_1m": disp1, "displacement_only_2m": disp2,
        "core_plus_displacement_1m": pd.concat([core, disp1], axis=1),
        "core_plus_displacement_2m": pd.concat([core.shift(1).fillna(0.0), disp2], axis=1),
    }
    replays = {name: base.simulate_minute(minute, pos) for name, pos in variants.items()}
    screen = pd.DataFrame([base.metrics(replay, name) for name, replay in replays.items()])
    screen.to_csv(RESULTS / "01_displacement_screen.csv", index=False)
    events.to_csv(RESULTS / "02_events.csv")
    for delay in (1, 2):
        ledger = pd.concat([ledgers[f"long_{delay}m"], ledgers[f"short_{delay}m"]], ignore_index=True)
        if not ledger.empty:
            ledger = ledger.sort_values("entry_time")
        ledger.to_csv(RESULTS / f"03_trades_{delay}m.csv", index=False)
    pd.DataFrame(audits).to_csv(RESULTS / "04_execution_diagnostics.csv", index=False)
    yearly = []
    for name, replay in replays.items():
        for year, group in replay.groupby(replay.index.year):
            local = group.copy(); local["equity"] = (1 + local["net_return"]).cumprod(); local["drawdown"] = local["equity"] / local["equity"].cummax() - 1
            row = base.metrics(local, name); row["year"] = int(year); yearly.append(row)
    pd.DataFrame(yearly).to_csv(RESULTS / "05_yearly.csv", index=False)
    (RESULTS / "run_config.json").write_text(json.dumps({
        "source": "OKX ETH-USDT-SWAP perpetual 1m K-lines and 15m trade bars only",
        "mechanism": "4H range displacement continuation with same-sign OKX taker delta",
        "range_gate": "rolling prior 180D 95th percentile", "close_location_gate": CLOSE_LOCATION_GATE,
        "invalidation": "50% displacement retracement", "reward_risk": REWARD_RISK, "max_hold": "12H",
        "execution": "complete 4H bar +1m; +2m stress", "one_way_cost": base.ONE_WAY_COST,
        "parameter_search": "none",
    }, indent=2), encoding="utf-8")
    print(screen.to_string(index=False)); print("\nEVENTS", len(events)); print(pd.DataFrame(audits).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
