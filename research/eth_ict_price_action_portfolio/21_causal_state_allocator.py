#!/usr/bin/env python
"""Causal state allocator for the long-hold daily PA core.

The allocator does not change entries or fit market returns.  It changes only
the capital released to the frozen daily PA/BOS core using information known at
the prior daily close.  A non-zero risk floor preserves daily market presence.
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


RESULTS = Path(__file__).resolve().parent / "ict_pa_v8" / "results"
RISK_ON_MULTIPLIER = 2.0
RISK_FLOOR_MULTIPLIER = 0.25


def daily_market_efficiency(minute: pd.DataFrame) -> pd.Series:
    daily = minute.resample("1D", label="left", closed="left").agg(close=("close", "last"))
    log_return = np.log(daily["close"]).diff()
    efficiency = np.log(daily["close"]).diff(90).abs() / log_return.abs().rolling(90, min_periods=90).sum()
    # The comparison distribution excludes the just-completed observation.
    prior_median = efficiency.shift(1).rolling(365, min_periods=180).median()
    risk_on = (efficiency > prior_median).astype(float)
    available = pd.Series(risk_on.to_numpy(), index=risk_on.index + pd.Timedelta(days=1), name="market_efficiency_on")
    return available


def daily_core_health(base_replay: pd.DataFrame) -> pd.Series:
    daily_return = (1.0 + base_replay["net_return"]).groupby(base_replay.index.floor("D")).prod() - 1.0
    trailing_return = (1.0 + daily_return).rolling(90, min_periods=90).apply(np.prod, raw=True) - 1.0
    # Day D's completed strategy return first controls risk at D+1 00:01.
    available = pd.Series((trailing_return > 0.0).astype(float).to_numpy(), index=trailing_return.index + pd.Timedelta(days=1), name="core_health_on")
    return available


def minute_multiplier(state: pd.Series, minute_index: pd.DatetimeIndex) -> pd.Series:
    multiplier = state.replace({0.0: RISK_FLOOR_MULTIPLIER, 1.0: RISK_ON_MULTIPLIER})
    execution_index = multiplier.index + pd.Timedelta(minutes=1)
    return pd.Series(multiplier.to_numpy(), index=execution_index).reindex(minute_index, method="ffill").fillna(RISK_FLOOR_MULTIPLIER)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, _ = base.load_inputs()
    frozen_core = base.core_state(minute) * 0.75
    base_positions = pd.DataFrame({"core": frozen_core}, index=minute.index)
    base_replay = base.simulate_minute(minute, base_positions)

    efficiency_state = daily_market_efficiency(minute)
    health_state = daily_core_health(base_replay)
    efficiency_multiplier = minute_multiplier(efficiency_state, minute.index)
    health_multiplier = minute_multiplier(health_state, minute.index)
    consensus_state = (efficiency_state.reindex(health_state.index, method="ffill") * health_state).rename("consensus_on")
    consensus_multiplier = minute_multiplier(consensus_state, minute.index)

    variants = {
        "frozen_daily_pa_core": base_positions,
        "market_efficiency_allocator": pd.DataFrame({"core": frozen_core * efficiency_multiplier}, index=minute.index),
        "core_health_allocator": pd.DataFrame({"core": frozen_core * health_multiplier}, index=minute.index),
        "efficiency_health_consensus": pd.DataFrame({"core": frozen_core * consensus_multiplier}, index=minute.index),
    }
    rows: list[dict[str, object]] = []
    replays: dict[str, pd.DataFrame] = {}
    for name, positions in variants.items():
        replay = base.simulate_minute(minute, positions)
        rows.append(base.metrics(replay, name))
        replays[name] = replay
    screen = pd.DataFrame(rows)
    screen.to_csv(RESULTS / "01_state_allocator_screen.csv", index=False)

    yearly: list[dict[str, object]] = []
    for name, replay in replays.items():
        for year, group in replay.groupby(replay.index.year):
            local = group.copy()
            local["equity"] = (1.0 + local["net_return"]).cumprod()
            local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
            row = base.metrics(local, name)
            row["year"] = year
            yearly.append(row)
    pd.DataFrame(yearly).to_csv(RESULTS / "02_yearly.csv", index=False)

    state_daily = pd.concat([efficiency_state, health_state, consensus_state], axis=1)
    state_daily.to_csv(RESULTS / "03_daily_states.csv")
    for name, replay in replays.items():
        daily = replay.groupby(replay.index.floor("D")).agg(
            equity=("equity", "last"),
            drawdown=("drawdown", "last"),
            gross_exposure=("gross_exposure", "max"),
            trading_cost=("trading_cost", "sum"),
        )
        daily["net_return"] = (1.0 + replay["net_return"]).groupby(replay.index.floor("D")).prod() - 1.0
        daily.to_csv(RESULTS / f"daily_{name}.csv")
    (RESULTS / "run_config.json").write_text(
        json.dumps(
            {
                "source": "OKX ETH-USDT-SWAP only",
                "core": "frozen daily PA/BOS core",
                "states": ["90D price efficiency above its trailing prior-365D median", "prior completed 90D core return positive"],
                "availability": "daily close +1 day boundary +1 minute execution",
                "risk_on_multiplier": RISK_ON_MULTIPLIER,
                "risk_floor_multiplier": RISK_FLOOR_MULTIPLIER,
                "one_way_cost": base.ONE_WAY_COST,
                "gross_cap": 0.75,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(screen.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

