#!/usr/bin/env python
"""Daily liquidity-sweep reclaim with one-minute causal execution.

This is a frozen low-frequency ICT/Price Action mechanism, not a refinement of
the failed intraday sweep screen.  A completed daily candle must trade beyond
the prior completed 20-day range and close back inside it.  Entry occurs after
the daily bar is known; the sweep extreme is the structural stop, the target
is 2R, and maximum holding time is seven calendar days.
"""

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


RESULTS = Path(__file__).resolve().parent / "ict_pa_v16" / "results"
LIQUIDITY_LOOKBACK_DAYS = 20
MAX_HOLD = pd.Timedelta(days=7)
RISK_PER_TRADE = 0.005
NOTIONAL_CAP = 0.30
REWARD_RISK = 2.0
ROUND_TRIP_COST = 2.0 * base.ONE_WAY_COST
MIN_TARGET_DISTANCE = 1.5 * ROUND_TRIP_COST


def build_daily_reclaim_events(minute: pd.DataFrame) -> pd.DataFrame:
    daily = minute.resample("1D", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        source_minutes=("close", "size"),
    )
    daily = daily[daily["source_minutes"] == 1440].dropna(subset=["open", "high", "low", "close"])
    prior_high = daily["high"].shift(1).rolling(LIQUIDITY_LOOKBACK_DAYS, min_periods=LIQUIDITY_LOOKBACK_DAYS).max()
    prior_low = daily["low"].shift(1).rolling(LIQUIDITY_LOOKBACK_DAYS, min_periods=LIQUIDITY_LOOKBACK_DAYS).min()
    sweep_low = (daily["low"] < prior_low) & (daily["close"] > prior_low)
    sweep_high = (daily["high"] > prior_high) & (daily["close"] < prior_high)
    unambiguous = sweep_low ^ sweep_high
    side = pd.Series(pd.NA, index=daily.index, dtype="object")
    side.loc[sweep_low & unambiguous] = "long"
    side.loc[sweep_high & unambiguous] = "short"
    raw = pd.DataFrame(
        {
            "side": side,
            "sweep_low": sweep_low,
            "sweep_high": sweep_high,
            "prior_high": prior_high,
            "prior_low": prior_low,
            "bar_high": daily["high"],
            "bar_low": daily["low"],
            "bar_close": daily["close"],
        },
        index=daily.index,
    )
    # The daily candle [D,D+1) becomes public at D+1 00:00.
    available = pd.DataFrame(raw.to_numpy(), columns=raw.columns, index=raw.index + pd.Timedelta(days=1))
    available["event_day"] = raw.index.to_numpy()
    available["available_time"] = available.index
    return available.dropna(subset=["side"])


def build_side_positions(
    events: pd.DataFrame,
    minute: pd.DataFrame,
    side: str,
    delay_minutes: int,
) -> tuple[pd.Series, pd.DataFrame, dict[str, int]]:
    if delay_minutes not in (1, 2):
        raise ValueError("delay_minutes must be the frozen 1m baseline or 2m stress")
    is_long = side == "long"
    candidates = events[events["side"] == side].sort_values("available_time")
    sparse: dict[pd.Timestamp, float] = {}
    trades: list[dict[str, object]] = []
    next_entry_after = pd.Timestamp.min
    diagnostics = {"candidate_events": int(len(candidates)), "skipped_overlap": 0, "skipped_invalid_stop": 0, "skipped_cost_distance": 0}
    for _, event in candidates.iterrows():
        entry_time = pd.Timestamp(event["available_time"]) + pd.Timedelta(minutes=delay_minutes)
        if entry_time <= next_entry_after:
            diagnostics["skipped_overlap"] += 1
            continue
        if entry_time not in minute.index:
            continue
        entry_price = float(minute.at[entry_time, "open"])
        stop_price = float(event["bar_low"] if is_long else event["bar_high"])
        risk_distance = entry_price - stop_price if is_long else stop_price - entry_price
        if not np.isfinite(risk_distance) or risk_distance <= 0.0:
            diagnostics["skipped_invalid_stop"] += 1
            continue
        stop_fraction = risk_distance / entry_price
        if REWARD_RISK * stop_fraction < MIN_TARGET_DISTANCE:
            diagnostics["skipped_cost_distance"] += 1
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
                "side": side, "event_day": event["event_day"], "entry_time": entry_time, "exit_time": exit_time,
                "entry_price": entry_price, "exit_price": exit_price, "stop_price": stop_price, "target_price": target_price,
                "stop_fraction": stop_fraction, "notional": notional, "underlying_return": underlying_return,
                "net_account_contribution": notional * underlying_return - 2.0 * notional * base.ONE_WAY_COST,
                "exit_reason": exit_reason, "ambiguous_bar_loss_first": ambiguous,
            }
        )
        next_entry_after = exit_time
    diagnostics["trades"] = len(trades)
    event_series = pd.Series(sparse, dtype=float).sort_index()
    position = event_series.reindex(minute.index, method="ffill").fillna(0.0) if not event_series.empty else pd.Series(0.0, index=minute.index)
    return position, pd.DataFrame(trades), diagnostics


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
    events = build_daily_reclaim_events(minute)
    positions: dict[str, pd.Series] = {}
    trades: dict[str, pd.DataFrame] = {}
    diagnostics: list[dict[str, object]] = []
    for delay in (1, 2):
        for side in ("long", "short"):
            position, ledger, audit = build_side_positions(events, minute, side, delay)
            positions[f"{side}_{delay}m"] = position
            trades[f"{side}_{delay}m"] = ledger
            diagnostics.append({"delay_minutes": delay, "side": side, **audit})

    reclaim_1m = pd.DataFrame({"reclaim_long": positions["long_1m"], "reclaim_short": positions["short_1m"]}, index=minute.index)
    reclaim_2m = pd.DataFrame({"reclaim_long": positions["long_2m"], "reclaim_short": positions["short_2m"]}, index=minute.index)
    bos_features = multibos.build_daily_bos_features(minute)
    bos_1m = multibos.positions_from_features(bos_features, minute.index, 1)
    bos_2m = multibos.positions_from_features(bos_features, minute.index, 2)
    variants = {
        "daily_reclaim_only_1m": reclaim_1m,
        "daily_reclaim_only_2m": reclaim_2m,
        "bos_plus_daily_reclaim_1m": pd.concat([bos_1m, reclaim_1m], axis=1),
        "bos_plus_daily_reclaim_2m": pd.concat([bos_2m, reclaim_2m], axis=1),
    }
    replays = {name: base.simulate_minute(minute, position) for name, position in variants.items()}
    screen = pd.DataFrame([base.metrics(replay, name) for name, replay in replays.items()])
    screen.to_csv(RESULTS / "01_daily_reclaim_screen.csv", index=False)
    events.loc[base.START:base.END].to_csv(RESULTS / "02_reclaim_events.csv")
    for delay in (1, 2):
        ledger = pd.concat([trades[f"long_{delay}m"], trades[f"short_{delay}m"]], ignore_index=True)
        if not ledger.empty:
            ledger = ledger.sort_values("entry_time")
        ledger.to_csv(RESULTS / f"03_trades_{delay}m.csv", index=False)
    yearly_metrics(replays).to_csv(RESULTS / "04_yearly.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(RESULTS / "05_execution_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["OKX ETH-USDT-SWAP 1m K-lines", "complete daily bars", "reclaim events"],
            "start": [minute.index.min(), bos_features.index.min(), events.index.min()],
            "end": [minute.index.max(), bos_features.index.max(), events.index.max()],
            "rows": [len(minute), len(bos_features), len(events)],
        }
    ).to_csv(RESULTS / "06_data_quality.csv", index=False)
    config = {
        "source": "OKX ETH-USDT-SWAP perpetual K-lines only",
        "mechanism": "completed daily sweep of prior 20-day range followed by close reclaim",
        "liquidity_lookback_days": LIQUIDITY_LOOKBACK_DAYS,
        "execution": "daily bar available next midnight; +1m baseline; +2m latency stress",
        "risk_per_trade": RISK_PER_TRADE,
        "notional_cap_per_side": NOTIONAL_CAP,
        "reward_risk": REWARD_RISK,
        "max_hold": "7D",
        "economic_filter_minimum_2R_fraction": MIN_TARGET_DISTANCE,
        "one_way_cost": base.ONE_WAY_COST,
        "same_bar_stop_and_target": "loss-first",
        "parameter_search": "none",
    }
    (RESULTS / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(screen.to_string(index=False))
    print("\nEVENTS", len(events.loc[base.START:base.END]))
    print(pd.DataFrame(diagnostics).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
