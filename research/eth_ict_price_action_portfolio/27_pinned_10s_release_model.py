#!/usr/bin/env python
"""Paper-scale pinned-flow release test on OKX ETH-USDT-SWAP.

The mechanism is frozen before observing results:

* a completed 10-second bar contains at least 50 trades;
* absolute trade-sign imbalance is at least 80%;
* its price return and range are no larger than the prior 60-minute median
  one-minute diffusion scale divided by sqrt(6);
* after the pin, a complete one-minute bar must show balanced trade signs and
  close outside the pin range opposite the absorbed flow;
* execution is the following one-minute open, with a fixed two-minute latency
  stress variant.

Only local OKX perpetual K-lines and OKX perpetual trade bars are used.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio import _okx_walkforward_bridge as base
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader


RESULTS = Path(__file__).resolve().parent / "ict_pa_v14" / "results"
PIN_SIGN_BIAS = 0.80
BALANCED_FLOW = 0.20
MIN_PIN_TRADES = 50
DIFFUSION_DIVISOR = float(np.sqrt(6.0))
RELEASE_WINDOW = pd.Timedelta(minutes=30)
MAX_HOLD = pd.Timedelta(hours=4)
RISK_PER_TRADE = 0.005
NOTIONAL_CAP = 0.30
REWARD_RISK = 2.0
ROUND_TRIP_COST = 2.0 * base.ONE_WAY_COST
MIN_TARGET_DISTANCE = 1.5 * ROUND_TRIP_COST


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def load_extreme_ten_second_bars() -> pd.DataFrame:
    """SQL prefilter implementing the paper's fixed N>=50 scale."""
    loader = OKXTradeBarLoader(symbol="ETH-USDT-SWAP", timeframe="10s")
    quoted_table = loader.table_name.replace('"', '""')
    query = f"""
        SELECT timestamp, open, high, low, close,
               buy_trades_count, sell_trades_count, trades_count
        FROM \"{quoted_table}\"
        WHERE timestamp >= ? AND timestamp <= ?
          AND trades_count >= ?
          AND (buy_trades_count + sell_trades_count) > 0
          AND abs(buy_trades_count - sell_trades_count) * 1.0
              / (buy_trades_count + sell_trades_count) >= ?
        ORDER BY timestamp
    """
    with sqlite3.connect(loader.db_path) as connection:
        frame = pd.read_sql_query(
            query,
            connection,
            params=(str(base.START), str(base.END + pd.Timedelta(seconds=59)), MIN_PIN_TRADES, PIN_SIGN_BIAS),
        )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.set_index("timestamp").sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def filter_pinned_candidates(extreme: pd.DataFrame, minute: pd.DataFrame) -> pd.DataFrame:
    """Apply causal price-pinning filters to SQL-prefiltered 10s bars."""
    minute_return = minute["close"] / minute["open"] - 1.0
    minute_range = minute["high"] / minute["low"] - 1.0
    prior_abs_return = minute_return.abs().shift(1).rolling(60, min_periods=60).median()
    prior_range = minute_range.shift(1).rolling(60, min_periods=60).median()

    out = extreme.copy()
    denominator = out["buy_trades_count"] + out["sell_trades_count"]
    out["sign_imbalance"] = safe_divide(
        out["buy_trades_count"] - out["sell_trades_count"], denominator
    )
    out["pin_return"] = out["close"] / out["open"] - 1.0
    out["pin_range"] = out["high"] / out["low"] - 1.0

    minute_key = out.index.floor("min")
    out["prior_60m_abs_return_median"] = prior_abs_return.reindex(minute_key).to_numpy()
    out["prior_60m_range_median"] = prior_range.reindex(minute_key).to_numpy()
    out["return_gate_10s"] = out["prior_60m_abs_return_median"] / DIFFUSION_DIVISOR
    out["range_gate_10s"] = out["prior_60m_range_median"] / DIFFUSION_DIVISOR
    out["pinned"] = (
        (out["trades_count"] >= MIN_PIN_TRADES)
        & (out["sign_imbalance"].abs() >= PIN_SIGN_BIAS)
        & (out["pin_return"].abs() <= out["return_gate_10s"])
        & (out["pin_range"] <= out["range_gate_10s"])
    )
    out["dominant_sign"] = np.sign(out["sign_imbalance"]).astype(int)
    out["available_time"] = out.index + pd.Timedelta(seconds=10)
    return out.replace([np.inf, -np.inf], np.nan)


def build_release_events(pins: pd.DataFrame, trade1: pd.DataFrame, minute: pd.DataFrame) -> pd.DataFrame:
    """Find the first fully observed balanced-flow release after each pin."""
    one_minute_sign = safe_divide(
        trade1["buy_trades_count"] - trade1["sell_trades_count"],
        trade1["buy_trades_count"] + trade1["sell_trades_count"],
    ).reindex(minute.index)
    rows: list[dict[str, object]] = []
    selected = pins[pins["pinned"].fillna(False).astype(bool)]
    for pin_time, pin in selected.iterrows():
        available = pd.Timestamp(pin["available_time"])
        # Never use a partial minute containing the 10s pin.  The first
        # eligible release window begins at the next one-minute boundary.
        first_bar = available.ceil("min")
        last_bar = min(first_bar + RELEASE_WINDOW - pd.Timedelta(minutes=1), minute.index.max())
        price_path = minute.loc[first_bar:last_bar]
        if price_path.empty:
            continue
        dominant = int(pin["dominant_sign"])
        for bar_time, price_bar in price_path.iterrows():
            flow = one_minute_sign.get(bar_time, np.nan)
            if pd.isna(flow) or abs(float(flow)) > BALANCED_FLOW:
                continue
            if dominant < 0 and float(price_bar["close"]) > float(pin["high"]):
                side = "long"
            elif dominant > 0 and float(price_bar["close"]) < float(pin["low"]):
                side = "short"
            else:
                continue
            rows.append(
                {
                    "pin_time": pin_time,
                    "pin_available_time": available,
                    "release_bar_time": bar_time,
                    "available_time": bar_time + pd.Timedelta(minutes=1),
                    "side": side,
                    "dominant_sign": dominant,
                    "pin_sign_imbalance": float(pin["sign_imbalance"]),
                    "pin_high": float(pin["high"]),
                    "pin_low": float(pin["low"]),
                    "release_flow": float(flow),
                }
            )
            break
    return pd.DataFrame(rows)


def build_side_positions(
    releases: pd.DataFrame,
    minute: pd.DataFrame,
    side: str,
    delay_minutes: int,
) -> tuple[pd.Series, pd.DataFrame, dict[str, int]]:
    """Convert releases into a non-overlapping sleeve and auditable trades."""
    if delay_minutes not in (1, 2):
        raise ValueError("delay_minutes must be the frozen 1m baseline or 2m stress")
    is_long = side == "long"
    candidates = releases[releases["side"] == side].sort_values("available_time") if not releases.empty else releases
    sparse: dict[pd.Timestamp, float] = {}
    trades: list[dict[str, object]] = []
    next_entry_after = pd.Timestamp.min
    skipped_overlap = 0
    skipped_invalid_stop = 0
    skipped_cost_distance = 0
    for _, event in candidates.iterrows():
        entry_time = pd.Timestamp(event["available_time"]) + pd.Timedelta(minutes=delay_minutes - 1)
        if entry_time <= next_entry_after:
            skipped_overlap += 1
            continue
        if entry_time not in minute.index:
            continue
        entry_price = float(minute.at[entry_time, "open"])
        stop_price = float(event["pin_low"] if is_long else event["pin_high"])
        risk_distance = entry_price - stop_price if is_long else stop_price - entry_price
        if not np.isfinite(risk_distance) or risk_distance <= 0.0:
            skipped_invalid_stop += 1
            continue
        stop_fraction = risk_distance / entry_price
        if REWARD_RISK * stop_fraction < MIN_TARGET_DISTANCE:
            skipped_cost_distance += 1
            continue
        notional = min(NOTIONAL_CAP, RISK_PER_TRADE / stop_fraction)
        target_price = entry_price + REWARD_RISK * risk_distance if is_long else entry_price - REWARD_RISK * risk_distance
        deadline = min(entry_time + MAX_HOLD, minute.index.max())
        trigger_time: pd.Timestamp | None = None
        exit_reason = "time"
        ambiguous_bar = False
        for bar_time, bar in minute.loc[entry_time:deadline].iterrows():
            stop_hit = float(bar["low"]) <= stop_price if is_long else float(bar["high"]) >= stop_price
            target_hit = float(bar["high"]) >= target_price if is_long else float(bar["low"]) <= target_price
            if stop_hit or target_hit:
                trigger_time = bar_time
                ambiguous_bar = bool(stop_hit and target_hit)
                # With one-minute OHLC, same-bar ordering is unknowable.  Use
                # the loss-first convention rather than optimistic ordering.
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
                "side": side,
                "pin_time": event["pin_time"],
                "release_bar_time": event["release_bar_time"],
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "stop_fraction": stop_fraction,
                "notional": notional,
                "underlying_return": underlying_return,
                "net_account_contribution": notional * underlying_return - 2.0 * notional * base.ONE_WAY_COST,
                "exit_reason": exit_reason,
                "ambiguous_bar_loss_first": ambiguous_bar,
            }
        )
        next_entry_after = exit_time
    events = pd.Series(sparse, dtype=float).sort_index()
    position = events.reindex(minute.index, method="ffill").fillna(0.0) if not events.empty else pd.Series(0.0, index=minute.index)
    diagnostics = {
        "candidate_releases": int(len(candidates)),
        "trades": int(len(trades)),
        "skipped_overlap": skipped_overlap,
        "skipped_invalid_stop": skipped_invalid_stop,
        "skipped_cost_distance": skipped_cost_distance,
    }
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
    trade1 = OKXTradeBarLoader(symbol="ETH-USDT-SWAP", timeframe="1m").load_local_data(base.START, base.END)
    extreme = load_extreme_ten_second_bars()
    pins = filter_pinned_candidates(extreme, minute)
    releases = build_release_events(pins, trade1, minute)

    positions: dict[str, pd.Series] = {}
    trade_frames: dict[str, pd.DataFrame] = {}
    diagnostic_rows: list[dict[str, object]] = []
    for delay in (1, 2):
        for side in ("long", "short"):
            position, trades, diagnostics = build_side_positions(releases, minute, side, delay)
            positions[f"{side}_{delay}m"] = position
            trade_frames[f"{side}_{delay}m"] = trades
            diagnostic_rows.append({"delay_minutes": delay, "side": side, **diagnostics})

    pos1 = pd.DataFrame({"pin_long": positions["long_1m"], "pin_short": positions["short_1m"]}, index=minute.index)
    pos2 = pd.DataFrame({"pin_long": positions["long_2m"], "pin_short": positions["short_2m"]}, index=minute.index)
    core = base.core_state(minute) * 0.75
    variants = {
        "pinned_10s_release_1m": pos1,
        "pinned_10s_release_2m": pos2,
        "daily_pa_core_only": pd.DataFrame({"core": core}, index=minute.index),
        "core_plus_pinned_10s_1m": pd.concat([pd.DataFrame({"core": core}, index=minute.index), pos1], axis=1),
        "core_plus_pinned_10s_2m": pd.concat([pd.DataFrame({"core": core}, index=minute.index), pos2], axis=1),
    }
    replays = {name: base.simulate_minute(minute, position) for name, position in variants.items()}
    screen = pd.DataFrame([base.metrics(replay, name) for name, replay in replays.items()])

    screen.to_csv(RESULTS / "01_pinned_10s_screen.csv", index=False)
    extreme.to_csv(RESULTS / "02_extreme_10s_candidates.csv")
    pins[pins["pinned"].fillna(False).astype(bool)].to_csv(RESULTS / "03_pinned_10s_events.csv")
    releases.to_csv(RESULTS / "04_release_events.csv", index=False)
    for delay in (1, 2):
        combined = pd.concat([trade_frames[f"long_{delay}m"], trade_frames[f"short_{delay}m"]], ignore_index=True)
        if not combined.empty:
            combined = combined.sort_values("entry_time")
        combined.to_csv(RESULTS / f"05_trades_{delay}m.csv", index=False)
    yearly_metrics(replays).to_csv(RESULTS / "06_yearly.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(RESULTS / "07_execution_diagnostics.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["OKX 1m K-lines", "OKX 1m trade bars", "OKX extreme 10s trade bars"],
            "start": [minute.index.min(), trade1.index.min(), extreme.index.min() if len(extreme) else pd.NaT],
            "end": [minute.index.max(), trade1.index.max(), extreme.index.max() if len(extreme) else pd.NaT],
            "rows": [len(minute.loc[base.START:base.END]), len(trade1), len(extreme)],
            "duplicate_timestamps": [minute.index.duplicated().sum(), trade1.index.duplicated().sum(), extreme.index.duplicated().sum()],
        }
    ).to_csv(RESULTS / "08_data_quality.csv", index=False)
    config = {
        "source": "OKX ETH-USDT-SWAP perpetual price and trades only",
        "mechanism_source": "Patzelt & Bouchaud, Universal scaling and nonlinearity of aggregate price impact, arXiv:1706.04163",
        "pin": {
            "window": "10s",
            "minimum_trades": MIN_PIN_TRADES,
            "absolute_trade_sign_imbalance": PIN_SIGN_BIAS,
            "price_gate": "prior 60 complete 1m median absolute return and range divided by sqrt(6)",
        },
        "release": "next full 1m bar through 30m; abs trade-sign imbalance <=20%; close breaks opposite absorbed flow",
        "execution": "first open after completed release bar; 2m fixed latency stress",
        "economic_filter": {"minimum_2R_target_fraction": MIN_TARGET_DISTANCE, "round_trip_cost": ROUND_TRIP_COST},
        "risk_per_trade": RISK_PER_TRADE,
        "notional_cap_per_side": NOTIONAL_CAP,
        "portfolio_gross_cap": 0.75,
        "reward_risk": REWARD_RISK,
        "max_hold": "4H",
        "one_way_cost": base.ONE_WAY_COST,
        "same_bar_stop_and_target": "loss-first",
        "parameter_search": "none",
    }
    (RESULTS / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(screen.to_string(index=False))
    print(
        "\nEXTREME", len(extreme),
        "PINS", int(pins["pinned"].fillna(False).sum()),
        "RELEASES", len(releases),
    )
    print(pd.DataFrame(diagnostic_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
