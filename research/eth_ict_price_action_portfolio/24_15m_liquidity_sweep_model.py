#!/usr/bin/env python
"""15m liquidity sweep/reclaim with OKX flow absorption and 1m execution."""

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


RESULTS = Path(__file__).resolve().parent / "ict_pa_v11" / "results"
RISK_PER_TRADE = 0.005
NOTIONAL_CAP = 0.30
MAX_HOLD = pd.Timedelta(hours=12)
REWARD_RISK = 2.0


def sweep_events(trade15: pd.DataFrame) -> pd.DataFrame:
    prior_high = trade15["high"].shift(1).rolling(96, min_periods=96).max()
    prior_low = trade15["low"].shift(1).rolling(96, min_periods=96).min()
    candle_range = (trade15["high"] - trade15["low"]).replace(0.0, np.nan)
    close_location = (trade15["close"] - trade15["low"]) / candle_range
    long_event = (
        (trade15["low"] < prior_low)
        & (trade15["close"] > prior_low)
        & (close_location >= 0.60)
        & (trade15["delta_notional"] < 0.0)
    )
    short_event = (
        (trade15["high"] > prior_high)
        & (trade15["close"] < prior_high)
        & (close_location <= 0.40)
        & (trade15["delta_notional"] > 0.0)
    )
    raw = pd.DataFrame(
        {
            "long_event": long_event,
            "short_event": short_event,
            "sweep_low": trade15["low"],
            "sweep_high": trade15["high"],
            "close_location": close_location,
            "delta_ratio": trade15["delta_notional"] / trade15["notional"].replace(0.0, np.nan),
        },
        index=trade15.index,
    )
    # The 15m bar labelled T completes at T+15m.
    return pd.DataFrame(raw.to_numpy(), columns=raw.columns, index=raw.index + pd.Timedelta(minutes=15))


def build_direction_positions(
    events: pd.DataFrame,
    minute: pd.DataFrame,
    side: str,
    delay_minutes: int,
) -> tuple[pd.Series, pd.DataFrame]:
    is_long = side == "long"
    event_column = "long_event" if is_long else "short_event"
    event_rows = events[events[event_column].astype(bool)]
    position_events: dict[pd.Timestamp, float] = {}
    trades: list[dict[str, object]] = []
    next_entry = pd.Timestamp.min
    for available_time, row in event_rows.iterrows():
        entry_time = available_time + pd.Timedelta(minutes=delay_minutes)
        if entry_time < next_entry or entry_time not in minute.index:
            continue
        entry_price = float(minute.at[entry_time, "open"])
        stop_price = float(row["sweep_low"] if is_long else row["sweep_high"])
        risk_distance = entry_price - stop_price if is_long else stop_price - entry_price
        if not np.isfinite(risk_distance) or risk_distance <= 0.0:
            continue
        stop_fraction = risk_distance / entry_price
        notional = min(NOTIONAL_CAP, RISK_PER_TRADE / stop_fraction)
        if notional <= 0.0:
            continue
        target_price = entry_price + REWARD_RISK * risk_distance if is_long else entry_price - REWARD_RISK * risk_distance
        deadline = min(entry_time + MAX_HOLD, minute.index.max())
        path = minute.loc[entry_time:deadline]
        trigger_time: pd.Timestamp | None = None
        exit_reason = "time"
        for timestamp, bar in path.iterrows():
            stop_hit = float(bar["low"]) <= stop_price if is_long else float(bar["high"]) >= stop_price
            target_hit = float(bar["high"]) >= target_price if is_long else float(bar["low"]) <= target_price
            if stop_hit or target_hit:
                # Intraminute ordering is unknown; treat a double touch as stop-first.
                trigger_time = timestamp
                exit_reason = "stop" if stop_hit else "target"
                break
        if trigger_time is None:
            exit_time = deadline
        else:
            exit_time = min(trigger_time + pd.Timedelta(minutes=1), minute.index.max())
        signed_notional = notional if is_long else -notional
        position_events[entry_time] = signed_notional
        position_events[exit_time] = 0.0
        exit_price = float(minute.at[exit_time, "open"])
        gross_return = (exit_price / entry_price - 1.0) * (1.0 if is_long else -1.0)
        trades.append(
            {
                "side": side, "available_time": available_time, "entry_time": entry_time,
                "exit_time": exit_time, "entry_price": entry_price, "exit_price": exit_price,
                "stop_price": stop_price, "target_price": target_price, "notional": notional,
                "stop_fraction": stop_fraction, "gross_underlying_return": gross_return,
                "net_account_contribution": notional * gross_return - 2.0 * notional * base.ONE_WAY_COST,
                "exit_reason": exit_reason,
            }
        )
        next_entry = exit_time
    sparse = pd.Series(position_events, dtype=float).sort_index()
    aligned = sparse.reindex(minute.index, method="ffill").fillna(0.0) if not sparse.empty else pd.Series(0.0, index=minute.index)
    return aligned, pd.DataFrame(trades)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, trade15 = base.load_inputs()
    events = sweep_events(trade15)
    long1, long_trades1 = build_direction_positions(events, minute, "long", 1)
    short1, short_trades1 = build_direction_positions(events, minute, "short", 1)
    long2, long_trades2 = build_direction_positions(events, minute, "long", 2)
    short2, short_trades2 = build_direction_positions(events, minute, "short", 2)
    pos1 = pd.DataFrame({"sweep_long": long1, "sweep_short": short1}, index=minute.index)
    pos2 = pd.DataFrame({"sweep_long": long2, "sweep_short": short2}, index=minute.index)
    core = base.core_state(minute) * 0.75
    variants = {
        "sweep_absorption_1m": pos1,
        "sweep_absorption_2m": pos2,
        "daily_pa_core_only": pd.DataFrame({"core": core}, index=minute.index),
        "core_plus_sweep_1m": pd.concat([pd.DataFrame({"core": core}, index=minute.index), pos1], axis=1),
        "core_plus_sweep_2m": pd.concat([pd.DataFrame({"core": core}, index=minute.index), pos2], axis=1),
    }
    rows: list[dict[str, object]] = []
    replays: dict[str, pd.DataFrame] = {}
    for name, positions in variants.items():
        replay = base.simulate_minute(minute, positions)
        rows.append(base.metrics(replay, name))
        replays[name] = replay
    screen = pd.DataFrame(rows)
    screen.to_csv(RESULTS / "01_sweep_screen.csv", index=False)
    pd.concat([long_trades1, short_trades1]).sort_values("entry_time").to_csv(RESULTS / "02_trades_1m.csv", index=False)
    pd.concat([long_trades2, short_trades2]).sort_values("entry_time").to_csv(RESULTS / "03_trades_2m.csv", index=False)
    event_rows = events[events["long_event"].astype(bool) | events["short_event"].astype(bool)]
    event_rows.to_csv(RESULTS / "04_events.csv")
    yearly: list[dict[str, object]] = []
    for name, replay in replays.items():
        for year, group in replay.groupby(replay.index.year):
            local = group.copy()
            local["equity"] = (1.0 + local["net_return"]).cumprod()
            local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
            row = base.metrics(local, name)
            row["year"] = year
            yearly.append(row)
    pd.DataFrame(yearly).to_csv(RESULTS / "05_yearly.csv", index=False)
    (RESULTS / "run_config.json").write_text(
        json.dumps(
            {
                "source": "OKX ETH-USDT-SWAP only",
                "event": "15m sweep of prior 24H high/low, close reclaim, opposing taker flow absorption",
                "availability": "15m bar end",
                "execution": "bar end +1m; +2m delay stress",
                "risk_per_trade": RISK_PER_TRADE,
                "notional_cap_per_side": NOTIONAL_CAP,
                "reward_risk": REWARD_RISK,
                "max_hold": "12H",
                "one_way_cost": base.ONE_WAY_COST,
                "portfolio_gross_cap": 0.75,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(screen.to_string(index=False))
    print("\nEVENTS", len(event_rows), "TRADES_1M", len(long_trades1), len(short_trades1))
    if len(long_trades1) + len(short_trades1):
        trade_summary = pd.concat([long_trades1, short_trades1]).groupby(["side", "exit_reason"]).agg(
            trades=("net_account_contribution", "size"), mean_net=("net_account_contribution", "mean"), sum_net=("net_account_contribution", "sum")
        )
        print(trade_summary.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

