#!/usr/bin/env python
"""Pinned order-sign imbalance -> balanced-flow price release model.

Mechanism source: Patzelt & Bouchaud (2018).  The paper freezes the empirical
hypothesis that extreme sign imbalance can coexist with a pinned price because
liquidity is replenished.  This OKX-only strategy waits for the dominant flow
to return to balance and for price to break away from the pinned range before
entering opposite the previously absorbed flow.
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

from research.eth_ict_price_action_portfolio import _okx_walkforward_bridge as base
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader


RESULTS = Path(__file__).resolve().parent / "ict_pa_v13" / "results"
PIN_SIGN_BIAS = 0.80
BALANCED_FLOW = 0.20
RELEASE_WINDOW = pd.Timedelta(minutes=30)
MAX_HOLD = pd.Timedelta(hours=4)
RISK_PER_TRADE = 0.005
NOTIONAL_CAP = 0.30
REWARD_RISK = 2.0


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def build_pinned_events(trade1: pd.DataFrame) -> pd.DataFrame:
    bars = trade1.resample("5min", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        notional=("notional", "sum"), buy_trades=("buy_trades_count", "sum"),
        sell_trades=("sell_trades_count", "sum"), trades=("trades_count", "sum"), source_minutes=("close", "size"),
    )
    bars = bars[bars["source_minutes"] == 5]
    sign_imbalance = safe_divide(bars["buy_trades"] - bars["sell_trades"], bars["buy_trades"] + bars["sell_trades"])
    net_return = bars["close"] / bars["open"] - 1.0
    bar_range = bars["high"] / bars["low"] - 1.0
    lookback = 12 * 24 * 7  # one calendar week of prior 5m observations
    prior_abs_return_median = net_return.abs().shift(1).rolling(lookback, min_periods=12 * 24).median()
    prior_range_median = bar_range.shift(1).rolling(lookback, min_periods=12 * 24).median()
    prior_trades_median = bars["trades"].shift(1).rolling(lookback, min_periods=12 * 24).median()
    pinned = (
        (sign_imbalance.abs() >= PIN_SIGN_BIAS)
        & (net_return.abs() <= prior_abs_return_median)
        & (bar_range <= prior_range_median)
        & (bars["trades"] >= prior_trades_median)
    )
    raw = pd.DataFrame(
        {
            "pinned": pinned,
            "dominant_sign": np.sign(sign_imbalance),
            "sign_imbalance": sign_imbalance,
            "pin_open": bars["open"], "pin_high": bars["high"], "pin_low": bars["low"], "pin_close": bars["close"],
            "pin_return": net_return, "pin_range": bar_range, "trades": bars["trades"],
        },
        index=bars.index,
    )
    # The [T,T+5m) pinning bar is known only at T+5m.
    return pd.DataFrame(raw.to_numpy(), columns=raw.columns, index=raw.index + pd.Timedelta(minutes=5))


def build_release_events(pins: pd.DataFrame, trade1: pd.DataFrame, minute: pd.DataFrame) -> pd.DataFrame:
    one_minute_sign = safe_divide(
        trade1["buy_trades_count"] - trade1["sell_trades_count"],
        trade1["buy_trades_count"] + trade1["sell_trades_count"],
    ).reindex(minute.index)
    rows: list[dict[str, object]] = []
    for pin_time, pin in pins[pins["pinned"].astype(bool)].iterrows():
        end = min(pin_time + RELEASE_WINDOW, minute.index.max())
        price_path = minute.loc[pin_time:end]
        if price_path.empty:
            continue
        dominant = int(pin["dominant_sign"])
        for timestamp, price_bar in price_path.iterrows():
            flow = one_minute_sign.get(timestamp, np.nan)
            if pd.isna(flow) or abs(float(flow)) > BALANCED_FLOW:
                continue
            if dominant < 0 and float(price_bar["close"]) > float(pin["pin_high"]):
                side = "long"
            elif dominant > 0 and float(price_bar["close"]) < float(pin["pin_low"]):
                side = "short"
            else:
                continue
            rows.append(
                {
                    "pin_time": pin_time, "release_bar_time": timestamp,
                    "available_time": timestamp + pd.Timedelta(minutes=1), "side": side,
                    "dominant_sign": dominant, "pin_sign_imbalance": pin["sign_imbalance"],
                    "pin_high": pin["pin_high"], "pin_low": pin["pin_low"],
                    "release_flow": flow,
                }
            )
            break
    return pd.DataFrame(rows)


def build_side_positions(
    releases: pd.DataFrame,
    minute: pd.DataFrame,
    side: str,
    delay_minutes: int,
) -> tuple[pd.Series, pd.DataFrame]:
    is_long = side == "long"
    candidates = releases[releases["side"] == side].sort_values("available_time")
    sparse: dict[pd.Timestamp, float] = {}
    trades: list[dict[str, object]] = []
    next_entry = pd.Timestamp.min
    for _, event in candidates.iterrows():
        # available_time is already the end of the 1m release bar.  Delay 1
        # means the immediately following 1m open; delay 2 is one extra minute.
        entry_time = pd.Timestamp(event["available_time"]) + pd.Timedelta(minutes=delay_minutes - 1)
        if entry_time < next_entry or entry_time not in minute.index:
            continue
        entry_price = float(minute.at[entry_time, "open"])
        stop_price = float(event["pin_low"] if is_long else event["pin_high"])
        risk_distance = entry_price - stop_price if is_long else stop_price - entry_price
        if not np.isfinite(risk_distance) or risk_distance <= 0.0:
            continue
        stop_fraction = risk_distance / entry_price
        notional = min(NOTIONAL_CAP, RISK_PER_TRADE / stop_fraction)
        target_price = entry_price + REWARD_RISK * risk_distance if is_long else entry_price - REWARD_RISK * risk_distance
        deadline = min(entry_time + MAX_HOLD, minute.index.max())
        trigger_time: pd.Timestamp | None = None
        exit_reason = "time"
        for timestamp, bar in minute.loc[entry_time:deadline].iterrows():
            stop_hit = float(bar["low"]) <= stop_price if is_long else float(bar["high"]) >= stop_price
            target_hit = float(bar["high"]) >= target_price if is_long else float(bar["low"]) <= target_price
            if stop_hit or target_hit:
                trigger_time = timestamp
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
                "side": side, "pin_time": event["pin_time"], "release_bar_time": event["release_bar_time"],
                "entry_time": entry_time, "exit_time": exit_time, "entry_price": entry_price, "exit_price": exit_price,
                "stop_price": stop_price, "target_price": target_price, "notional": notional,
                "underlying_return": underlying_return,
                "net_account_contribution": notional * underlying_return - 2.0 * notional * base.ONE_WAY_COST,
                "exit_reason": exit_reason,
            }
        )
        next_entry = exit_time
    event_series = pd.Series(sparse, dtype=float).sort_index()
    position = event_series.reindex(minute.index, method="ffill").fillna(0.0) if not event_series.empty else pd.Series(0.0, index=minute.index)
    return position, pd.DataFrame(trades)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, _ = base.load_inputs()
    trade1 = OKXTradeBarLoader(symbol="ETH-USDT-SWAP", timeframe="1m").load_local_data(base.START, base.END)
    pins = build_pinned_events(trade1)
    releases = build_release_events(pins, trade1, minute)
    long1, long_trades1 = build_side_positions(releases, minute, "long", 1)
    short1, short_trades1 = build_side_positions(releases, minute, "short", 1)
    long2, long_trades2 = build_side_positions(releases, minute, "long", 2)
    short2, short_trades2 = build_side_positions(releases, minute, "short", 2)
    pos1 = pd.DataFrame({"pin_long": long1, "pin_short": short1}, index=minute.index)
    pos2 = pd.DataFrame({"pin_long": long2, "pin_short": short2}, index=minute.index)
    core = base.core_state(minute) * 0.75
    variants = {
        "pinned_flow_release_1m": pos1,
        "pinned_flow_release_2m": pos2,
        "daily_pa_core_only": pd.DataFrame({"core": core}, index=minute.index),
        "core_plus_pinned_release_1m": pd.concat([pd.DataFrame({"core": core}, index=minute.index), pos1], axis=1),
        "core_plus_pinned_release_2m": pd.concat([pd.DataFrame({"core": core}, index=minute.index), pos2], axis=1),
    }
    rows: list[dict[str, object]] = []
    replays: dict[str, pd.DataFrame] = {}
    for name, positions in variants.items():
        replay = base.simulate_minute(minute, positions)
        rows.append(base.metrics(replay, name))
        replays[name] = replay
    screen = pd.DataFrame(rows)
    screen.to_csv(RESULTS / "01_pinned_release_screen.csv", index=False)
    pins[pins["pinned"].astype(bool)].to_csv(RESULTS / "02_pinned_events.csv")
    releases.to_csv(RESULTS / "03_release_events.csv", index=False)
    trades1 = pd.concat([long_trades1, short_trades1]).sort_values("entry_time")
    trades2 = pd.concat([long_trades2, short_trades2]).sort_values("entry_time")
    trades1.to_csv(RESULTS / "04_trades_1m.csv", index=False)
    trades2.to_csv(RESULTS / "05_trades_2m.csv", index=False)
    yearly: list[dict[str, object]] = []
    for name, replay in replays.items():
        for year, group in replay.groupby(replay.index.year):
            local = group.copy()
            local["equity"] = (1.0 + local["net_return"]).cumprod()
            local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
            row = base.metrics(local, name)
            row["year"] = year
            yearly.append(row)
    pd.DataFrame(yearly).to_csv(RESULTS / "06_yearly.csv", index=False)
    (RESULTS / "run_config.json").write_text(
        json.dumps(
            {
                "source": "OKX ETH-USDT-SWAP price and trades only",
                "mechanism_source": "Patzelt & Bouchaud, Universal scaling and nonlinearity of aggregate price impact, arXiv:1706.04163",
                "pin": "5m absolute trade-sign imbalance >=80%, below prior-week median return/range, trades >= median",
                "release": "within 30m, 1m trade-sign imbalance <=20% and price breaks opposite absorbed flow",
                "execution": "release bar completion then next 1m open; 2m delay stress",
                "risk_per_trade": RISK_PER_TRADE, "notional_cap": NOTIONAL_CAP,
                "reward_risk": REWARD_RISK, "max_hold": "4H",
                "one_way_cost": base.ONE_WAY_COST, "gross_cap": 0.75,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(screen.to_string(index=False))
    print("\nPINS", int(pins["pinned"].astype(bool).sum()), "RELEASES", len(releases), "TRADES", len(trades1))
    if not trades1.empty:
        print(trades1.groupby(["side", "exit_reason"]).agg(trades=("net_account_contribution", "size"), mean_net=("net_account_contribution", "mean"), sum_net=("net_account_contribution", "sum")).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

