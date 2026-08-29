#!/usr/bin/env python
"""Canonical NR7 volatility-contraction breakout on OKX ETH perpetual."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio import _multiscale_bos_bridge as multibos
from research.eth_ict_price_action_portfolio import _okx_walkforward_bridge as base


RESULTS = Path(__file__).resolve().parent / "ict_pa_v17" / "results"
NR_DAYS = 7
BREAKOUT_WINDOW = pd.Timedelta(days=1)
MAX_HOLD = pd.Timedelta(days=2)
RISK_PER_TRADE = 0.005
NOTIONAL_CAP = 0.30
REWARD_RISK = 2.0
ROUND_TRIP_COST = 2.0 * base.ONE_WAY_COST
MIN_TARGET_DISTANCE = 1.5 * ROUND_TRIP_COST


def build_nr7_setups(minute: pd.DataFrame) -> pd.DataFrame:
    daily = minute.resample("1D", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        source_minutes=("close", "size"),
    )
    daily = daily[daily["source_minutes"] == 1440].dropna(subset=["open", "high", "low", "close"])
    day_range = daily["high"] / daily["low"] - 1.0
    is_nr7 = day_range <= day_range.rolling(NR_DAYS, min_periods=NR_DAYS).min()
    raw = pd.DataFrame(
        {"setup_day": daily.index, "setup_high": daily["high"], "setup_low": daily["low"], "setup_range": day_range},
        index=daily.index,
    ).loc[is_nr7]
    # Setup day is complete at the next midnight.
    return pd.DataFrame(raw.to_numpy(), columns=raw.columns, index=raw.index + pd.Timedelta(days=1))


def build_breakout_events(setups: pd.DataFrame, minute: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for available_time, setup in setups.iterrows():
        end = min(available_time + BREAKOUT_WINDOW - pd.Timedelta(minutes=1), minute.index.max())
        path = minute.loc[available_time:end]
        if path.empty:
            continue
        long_break = path["close"] > float(setup["setup_high"])
        short_break = path["close"] < float(setup["setup_low"])
        eligible = long_break | short_break
        if not eligible.any():
            continue
        release_bar = eligible[eligible].index[0]
        side = "long" if bool(long_break.at[release_bar]) else "short"
        rows.append(
            {
                "setup_day": setup["setup_day"], "setup_available_time": available_time,
                "release_bar_time": release_bar, "available_time": release_bar + pd.Timedelta(minutes=1),
                "side": side, "setup_high": float(setup["setup_high"]), "setup_low": float(setup["setup_low"]),
                "setup_range": float(setup["setup_range"]),
            }
        )
    return pd.DataFrame(rows)


def build_side_positions(
    events: pd.DataFrame, minute: pd.DataFrame, side: str, delay_minutes: int
) -> tuple[pd.Series, pd.DataFrame, dict[str, int]]:
    if delay_minutes not in (1, 2):
        raise ValueError("delay_minutes must be 1m baseline or 2m stress")
    is_long = side == "long"
    candidates = events[events["side"] == side].sort_values("available_time") if not events.empty else events
    sparse: dict[pd.Timestamp, float] = {}
    trades: list[dict[str, object]] = []
    next_entry_after = pd.Timestamp.min
    audit = {"candidate_events": int(len(candidates)), "skipped_overlap": 0, "skipped_invalid_stop": 0, "skipped_cost_distance": 0}
    for _, event in candidates.iterrows():
        entry_time = pd.Timestamp(event["available_time"]) + pd.Timedelta(minutes=delay_minutes - 1)
        if entry_time <= next_entry_after:
            audit["skipped_overlap"] += 1
            continue
        if entry_time not in minute.index:
            continue
        entry_price = float(minute.at[entry_time, "open"])
        stop_price = float(event["setup_low"] if is_long else event["setup_high"])
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
        signed_notional = notional if is_long else -notional
        sparse[entry_time] = signed_notional
        sparse[exit_time] = 0.0
        exit_price = float(minute.at[exit_time, "open"])
        underlying_return = (exit_price / entry_price - 1.0) * (1.0 if is_long else -1.0)
        trades.append(
            {
                "side": side, "setup_day": event["setup_day"], "release_bar_time": event["release_bar_time"],
                "entry_time": entry_time, "exit_time": exit_time, "entry_price": entry_price, "exit_price": exit_price,
                "stop_price": stop_price, "target_price": target_price, "notional": notional,
                "underlying_return": underlying_return,
                "net_account_contribution": notional * underlying_return - 2.0 * notional * base.ONE_WAY_COST,
                "exit_reason": exit_reason, "ambiguous_bar_loss_first": ambiguous,
            }
        )
        next_entry_after = exit_time
    audit["trades"] = len(trades)
    series = pd.Series(sparse, dtype=float).sort_index()
    position = series.reindex(minute.index, method="ffill").fillna(0.0) if not series.empty else pd.Series(0.0, index=minute.index)
    return position, pd.DataFrame(trades), audit


def yearly_metrics(replays: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, replay in replays.items():
        for year, group in replay.groupby(replay.index.year):
            local = group.copy()
            local["equity"] = (1.0 + local["net_return"]).cumprod()
            local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
            row = base.metrics(local, name)
            row["year"] = int(year)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, _ = base.load_inputs()
    setups = build_nr7_setups(minute)
    events = build_breakout_events(setups, minute)
    positions: dict[str, pd.Series] = {}
    trades: dict[str, pd.DataFrame] = {}
    diagnostics: list[dict[str, object]] = []
    for delay in (1, 2):
        for side in ("long", "short"):
            position, ledger, audit = build_side_positions(events, minute, side, delay)
            positions[f"{side}_{delay}m"] = position
            trades[f"{side}_{delay}m"] = ledger
            diagnostics.append({"delay_minutes": delay, "side": side, **audit})
    nr7_1m = pd.DataFrame({"nr7_long": positions["long_1m"], "nr7_short": positions["short_1m"]}, index=minute.index)
    nr7_2m = pd.DataFrame({"nr7_long": positions["long_2m"], "nr7_short": positions["short_2m"]}, index=minute.index)
    bos_features = multibos.build_daily_bos_features(minute)
    bos_1m = multibos.positions_from_features(bos_features, minute.index, 1)
    bos_2m = multibos.positions_from_features(bos_features, minute.index, 2)
    variants = {
        "nr7_breakout_only_1m": nr7_1m,
        "nr7_breakout_only_2m": nr7_2m,
        "bos_plus_nr7_breakout_1m": pd.concat([bos_1m, nr7_1m], axis=1),
        "bos_plus_nr7_breakout_2m": pd.concat([bos_2m, nr7_2m], axis=1),
    }
    replays = {name: base.simulate_minute(minute, pos) for name, pos in variants.items()}
    pd.DataFrame([base.metrics(replay, name) for name, replay in replays.items()]).to_csv(RESULTS / "01_nr7_screen.csv", index=False)
    setups.loc[base.START:base.END].to_csv(RESULTS / "02_nr7_setups.csv")
    events.to_csv(RESULTS / "03_breakout_events.csv", index=False)
    for delay in (1, 2):
        ledger = pd.concat([trades[f"long_{delay}m"], trades[f"short_{delay}m"]], ignore_index=True)
        if not ledger.empty:
            ledger = ledger.sort_values("entry_time")
        ledger.to_csv(RESULTS / f"04_trades_{delay}m.csv", index=False)
    yearly_metrics(replays).to_csv(RESULTS / "05_yearly.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(RESULTS / "06_execution_diagnostics.csv", index=False)
    config = {
        "source": "OKX ETH-USDT-SWAP perpetual K-lines only", "mechanism": "canonical NR7 daily contraction breakout",
        "nr_days": NR_DAYS, "breakout_observation": "first 1m close outside completed NR7 day range during next 24H",
        "execution": "next 1m open; +2m stress", "risk_per_trade": RISK_PER_TRADE,
        "notional_cap_per_side": NOTIONAL_CAP, "reward_risk": REWARD_RISK, "max_hold": "48H",
        "minimum_2R_target_fraction": MIN_TARGET_DISTANCE, "one_way_cost": base.ONE_WAY_COST,
        "same_bar_stop_and_target": "loss-first", "parameter_search": "none",
    }
    (RESULTS / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    screen = pd.read_csv(RESULTS / "01_nr7_screen.csv")
    print(screen.to_string(index=False))
    print("\nSETUPS", len(setups.loc[base.START:base.END]), "EVENTS", len(events))
    print(pd.DataFrame(diagnostics).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
