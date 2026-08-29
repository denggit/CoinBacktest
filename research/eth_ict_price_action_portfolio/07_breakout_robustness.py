#!/usr/bin/env python
"""Causality, neighbourhood and concentration audit for the 4H breakout sleeve."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio import _stable_portfolio_bridge as bridge


RESULTS = Path(__file__).resolve().parent / "ict_pa_v2" / "results"


def aggregate_four_hour(trade: pd.DataFrame) -> pd.DataFrame:
    return trade.resample("4h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        notional=("notional", "sum"), delta_notional=("delta_notional", "sum"),
        large_delta_notional=("large_delta_notional", "sum"), trades=("trades_count", "sum"),
    ).dropna(subset=["open", "high", "low", "close"])


def definition(four: pd.DataFrame, channel_bars: int, tail: float, expansion: float, hold_bars: int) -> pd.DataFrame:
    candle_range = (four["high"] - four["low"]).replace(0.0, np.nan)
    close_location = (four["close"] - four["low"]) / candle_range
    delta_ratio = four["delta_notional"] / four["notional"].replace(0.0, np.nan)
    prior_low = four["low"].shift(1).rolling(channel_bars, min_periods=channel_bars).min()
    prior_high = four["high"].shift(1).rolling(channel_bars, min_periods=channel_bars).max()
    q_low = delta_ratio.shift(1).rolling(180, min_periods=60).quantile(tail)
    q_high = delta_ratio.shift(1).rolling(180, min_periods=60).quantile(1.0 - tail)
    tr = pd.concat(
        [four["high"] - four["low"], (four["high"] - four["close"].shift(1)).abs(), (four["low"] - four["close"].shift(1)).abs()], axis=1
    ).max(axis=1)
    atr = tr.shift(1).rolling(42, min_periods=42).median()
    long_event = (four["close"] > prior_high) & (close_location >= 0.70) & (tr >= expansion * atr) & (delta_ratio >= q_high)
    short_event = (four["close"] < prior_low) & (close_location <= 0.30) & (tr >= expansion * atr) & (delta_ratio <= q_low)
    long_pos, short_pos = bridge.hold_events(long_event, short_event, bars=hold_bars, exposure=0.10)
    return pd.DataFrame(
        {
            # Positional copy is essential because the availability index is
            # shifted by 4H.  Timestamp alignment here would be a 4H leak.
            "long_event": long_event.to_numpy(),
            "short_event": short_event.to_numpy(),
            "long": long_pos.to_numpy(),
            "short": short_pos.to_numpy(),
        },
        index=four.index + pd.Timedelta(hours=4),
    )


def align(position: pd.DataFrame, target: pd.DatetimeIndex) -> pd.DataFrame:
    out = position[["long", "short"]].reindex(target, method="ffill").fillna(0.0)
    if len(position):
        out.loc[out.index >= position.index.max() + pd.Timedelta(hours=4), :] = 0.0
    return out


def event_study(events: pd.DataFrame, price_15m: pd.DataFrame, label: str) -> pd.DataFrame:
    open_px = price_15m["open"]
    rows: list[dict[str, object]] = []
    for side, sign in (("long", 1.0), ("short", -1.0)):
        for signal_time in events.index[events[f"{side}_event"].fillna(False)]:
            entry_time = pd.Timestamp(signal_time)
            exit_time = entry_time + pd.Timedelta(hours=24)
            if entry_time not in open_px.index or exit_time not in open_px.index:
                continue
            raw = sign * (float(open_px.loc[exit_time]) / float(open_px.loc[entry_time]) - 1.0)
            rows.append(
                {"definition": label, "side": side, "signal_available_time": entry_time, "exit_time": exit_time, "gross_return_24h": raw, "net_return_24h": raw - 0.001}
            )
    return pd.DataFrame(rows)


def bootstrap_daily(daily: pd.Series, simulations: int = 2000, block: int = 7, seed: int = 20260817) -> pd.DataFrame:
    values = daily.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    blocks_needed = int(np.ceil(len(values) / block))
    for _ in range(simulations):
        starts = rng.integers(0, max(1, len(values) - block + 1), size=blocks_needed)
        sample = np.concatenate([values[s:s + block] for s in starts])[:len(values)]
        equity = np.cumprod(1.0 + sample)
        dd = equity / np.maximum.accumulate(equity) - 1.0
        years = len(sample) / 365.25
        cagr = equity[-1] ** (1.0 / years) - 1.0
        max_dd = abs(float(dd.min()))
        rows.append({"cagr": cagr, "max_drawdown": max_dd, "calmar": cagr / max_dd if max_dd > 0 else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    price, trade = bridge.load_inputs()
    four = aggregate_four_hour(trade)
    specs = {
        "adjacent_fast": (24, 0.15, 1.00, 4),
        "frozen_base": (30, 0.20, 1.25, 6),
        "adjacent_slow": (36, 0.25, 1.50, 8),
    }
    core_daily = bridge.build_daily_core(price)
    core = core_daily["position"].reindex(price.index, method="ffill").fillna(0.0)
    frames: dict[str, pd.DataFrame] = {}
    positions: dict[str, pd.DataFrame] = {}
    events: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    for name, spec in specs.items():
        raw = definition(four, *spec)
        event_frame = event_study(raw, price, name)
        events.append(event_frame)
        pos = align(raw, price.index)
        positions[name] = pos
        combined = pd.concat([core.rename("core"), pos], axis=1)
        frame = bridge.simulate(price, combined)
        frames[name] = frame
        rows.append(bridge.metrics(frame, name))

    # Equal-weight adjacent definitions are the deployable tactical sleeve.
    ensemble_micro = sum((positions[name] for name in specs)) / len(specs)
    ensemble = pd.concat([core.rename("core"), ensemble_micro], axis=1)
    ensemble_frame = bridge.simulate(price, ensemble)
    rows.append(bridge.metrics(ensemble_frame, "adjacent_definition_ensemble"))
    pd.DataFrame(rows).to_csv(RESULTS / "breakout_neighbourhood.csv", index=False)

    event_table = pd.concat(events, ignore_index=True)
    event_table.to_csv(RESULTS / "breakout_event_study.csv", index=False)
    event_summary = event_table.groupby(["definition", "side", event_table["signal_available_time"].dt.year]).agg(
        events=("net_return_24h", "size"), mean_net_24h=("net_return_24h", "mean"), median_net_24h=("net_return_24h", "median"), win_rate=("net_return_24h", lambda x: float((x > 0).mean()))
    ).reset_index().rename(columns={"signal_available_time": "year"})
    event_summary.to_csv(RESULTS / "breakout_event_summary_by_year.csv", index=False)

    # Sequence-independent block bootstrap quantifies whether the fixed
    # portfolio's Calmar>1 conclusion depends on the observed ordering alone.
    daily = (1.0 + ensemble_frame["net_return"]).groupby(ensemble_frame.index.floor("D")).prod() - 1.0
    boot = bootstrap_daily(daily)
    quantiles = boot.quantile([0.025, 0.05, 0.50, 0.95, 0.975]).reset_index(names="quantile")
    quantiles["probability_calmar_ge_1"] = float((boot["calmar"] >= 1.0).mean())
    quantiles["probability_cagr_positive"] = float((boot["cagr"] > 0.0).mean())
    quantiles.to_csv(RESULTS / "block_bootstrap.csv", index=False)

    # Future perturbation audit: changing all trade fields strictly after each
    # cutoff must not alter any event available on or before the cutoff.
    audit_rows = []
    for cutoff in (pd.Timestamp("2023-12-31 20:00"), pd.Timestamp("2024-12-31 20:00"), pd.Timestamp("2025-12-31 20:00")):
        base = definition(four, *specs["frozen_base"])
        changed = four.copy()
        future = changed.index > cutoff
        for column in ("open", "high", "low", "close", "delta_notional", "large_delta_notional"):
            changed.loc[future, column] = changed.loc[future, column] * 1.37 + 17.0
        altered = definition(changed, *specs["frozen_base"])
        available_cutoff = cutoff + pd.Timedelta(hours=4)
        left = base.loc[:available_cutoff, ["long_event", "short_event", "long", "short"]]
        right = altered.reindex(left.index).loc[:, left.columns]
        audit_rows.append({"cutoff": cutoff, "rows_compared": len(left), "passed": bool(left.equals(right)), "differences": int((left != right).to_numpy().sum())})
    pd.DataFrame(audit_rows).to_csv(RESULTS / "future_perturbation_audit.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nBOOTSTRAP\n", quantiles.to_string(index=False))
    print("\nFUTURE AUDIT\n", pd.DataFrame(audit_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
