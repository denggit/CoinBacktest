#!/usr/bin/env python
"""Structural long-hold sleeve plus independent crash-short hedge.

Calendar-semantic horizons are frozen before evaluation: six months for the
macro growth state, one month for downside structure, and two weeks for hedge
release.  Both sleeves may coexist in hedge mode and are charged separately.
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


RESULTS = Path(__file__).resolve().parent / "ict_pa_v10" / "results"
BASE_LONG = 0.10
GROWTH_LONG = 0.20
CRASH_SHORT = -0.30


def daily_states(minute: pd.DataFrame) -> pd.DataFrame:
    daily = minute.resample("1D", label="left", closed="left").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"), source_minutes=("close", "size")
    )
    daily = daily[daily["source_minutes"] == 1440]
    close = daily["close"]
    prior_high_90 = daily["high"].shift(1).rolling(90, min_periods=90).max()
    prior_low_90 = daily["low"].shift(1).rolling(90, min_periods=90).min()
    macro_midpoint = (prior_high_90 + prior_low_90) / 2.0
    momentum_180 = np.log(close).diff(180)
    momentum_30 = np.log(close).diff(30)
    prior_low_30 = daily["low"].shift(1).rolling(30, min_periods=30).min()
    prior_high_14 = daily["high"].shift(1).rolling(14, min_periods=14).max()

    growth_on = (momentum_180 > 0.0) & (close > macro_midpoint)
    hedge = False
    hedge_values: list[float] = []
    for price, low30, high14, mom30 in zip(close, prior_low_30, prior_high_14, momentum_30):
        if not hedge and pd.notna(low30) and price < low30 and mom30 < 0.0:
            hedge = True
        elif hedge and ((pd.notna(high14) and price > high14) or mom30 > 0.0):
            hedge = False
        hedge_values.append(CRASH_SHORT if hedge else 0.0)

    raw = pd.DataFrame(
        {
            "base_long": BASE_LONG,
            "growth_long": np.where(growth_on, GROWTH_LONG, 0.0),
            "crash_short": hedge_values,
            "growth_on": growth_on.astype(float),
            "hedge_on": (np.asarray(hedge_values) != 0.0).astype(float),
        },
        index=daily.index,
    )
    # State computed from day D is first executable at D+1.
    return pd.DataFrame(raw.to_numpy(), columns=raw.columns, index=raw.index + pd.Timedelta(days=1))


def positions(states: pd.DataFrame, minute_index: pd.DatetimeIndex, delay_minutes: int) -> pd.DataFrame:
    execution = states[["base_long", "growth_long", "crash_short"]].copy()
    execution.index = execution.index + pd.Timedelta(minutes=delay_minutes)
    return execution.reindex(minute_index, method="ffill").fillna({"base_long": BASE_LONG, "growth_long": 0.0, "crash_short": 0.0})


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, _ = base.load_inputs()
    states = daily_states(minute)
    pos1 = positions(states, minute.index, 1)
    pos2 = positions(states, minute.index, 2)
    frozen_core = base.core_state(minute) * 0.75
    variants = {
        "structural_long_crash_hedge_1m": pos1,
        "structural_long_crash_hedge_2m": pos2,
        "frozen_daily_pa_core": pd.DataFrame({"core": frozen_core}, index=minute.index),
        "pa_core_plus_structural_hedge_1m": pd.concat([pd.DataFrame({"core": frozen_core}, index=minute.index), pos1], axis=1),
        "pa_core_plus_structural_hedge_2m": pd.concat([pd.DataFrame({"core": frozen_core}, index=minute.index), pos2], axis=1),
    }
    rows: list[dict[str, object]] = []
    replays: dict[str, pd.DataFrame] = {}
    for name, position in variants.items():
        replay = base.simulate_minute(minute, position)
        rows.append(base.metrics(replay, name))
        replays[name] = replay
    screen = pd.DataFrame(rows)
    screen.to_csv(RESULTS / "01_structural_hedge_screen.csv", index=False)
    states.to_csv(RESULTS / "02_daily_states.csv")

    yearly: list[dict[str, object]] = []
    for name, replay in replays.items():
        for year, group in replay.groupby(replay.index.year):
            local = group.copy()
            local["equity"] = (1.0 + local["net_return"]).cumprod()
            local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
            row = base.metrics(local, name)
            row["year"] = year
            yearly.append(row)
    pd.DataFrame(yearly).to_csv(RESULTS / "03_yearly.csv", index=False)
    for name, replay in replays.items():
        daily = replay.groupby(replay.index.floor("D")).agg(
            equity=("equity", "last"), drawdown=("drawdown", "last"),
            gross_exposure=("gross_exposure", "max"), trading_cost=("trading_cost", "sum"),
        )
        daily["net_return"] = (1.0 + replay["net_return"]).groupby(replay.index.floor("D")).prod() - 1.0
        daily.to_csv(RESULTS / f"daily_{name}.csv")
    (RESULTS / "run_config.json").write_text(
        json.dumps(
            {
                "source": "OKX ETH-USDT-SWAP only",
                "sleeves": {"permanent_long": BASE_LONG, "six_month_growth_add": GROWTH_LONG, "one_month_breakdown_short": CRASH_SHORT},
                "hedge_release": "two-week high break or 30D momentum positive",
                "execution": "completed daily state at D+1 plus 1m; 2m delay stress",
                "one_way_cost": base.ONE_WAY_COST,
                "gross_cap": 0.75,
                "exchange_leverage_cap": 15.0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(screen.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

