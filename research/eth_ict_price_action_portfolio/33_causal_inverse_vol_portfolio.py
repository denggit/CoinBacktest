#!/usr/bin/env python
"""Causal inverse-volatility allocation across three frozen PA mechanisms."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio import _multiscale_bos_bridge as bos
from research.eth_ict_price_action_portfolio import _multispeed_ema_bridge as ema
from research.eth_ict_price_action_portfolio import _okx_walkforward_bridge as base


RESULTS = Path(__file__).resolve().parent / "ict_pa_v20" / "results"
WEIGHT_LOOKBACK_DAYS = 90


def daily_inverse_vol_weights(
    minute: pd.DataFrame, mechanisms: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    daily_returns: dict[str, pd.Series] = {}
    for name, positions in mechanisms.items():
        replay = base.simulate_minute(minute, positions, cost=0.0)
        daily_returns[name] = (1.0 + replay["net_return"]).groupby(replay.index.floor("D")).prod() - 1.0
    returns = pd.DataFrame(daily_returns)
    trailing_vol = returns.rolling(WEIGHT_LOOKBACK_DAYS, min_periods=WEIGHT_LOOKBACK_DAYS).std(ddof=0)
    inverse = 1.0 / trailing_vol.replace(0.0, np.nan)
    weights = inverse.div(inverse.sum(axis=1), axis=0).fillna(1.0 / len(mechanisms))
    # Completed day D's mechanism PnL controls D+1, never D itself.
    return pd.DataFrame(weights.to_numpy(), columns=weights.columns, index=weights.index + pd.Timedelta(days=1))


def allocate(
    mechanisms: dict[str, pd.DataFrame], weights: pd.DataFrame, minute_index: pd.DatetimeIndex, delay_minutes: int
) -> pd.DataFrame:
    minute_weights = pd.DataFrame(
        weights.to_numpy(), columns=weights.columns, index=weights.index + pd.Timedelta(minutes=delay_minutes)
    ).reindex(minute_index, method="ffill").fillna(1.0 / len(mechanisms))
    parts = []
    for name, positions in mechanisms.items():
        part = positions.mul(minute_weights[name], axis=0)
        part.columns = [f"{name}_{column}" for column in part.columns]
        parts.append(part)
    return pd.concat(parts, axis=1)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, _ = base.load_inputs()
    core = pd.DataFrame({"core": base.core_state(minute)}, index=minute.index)
    bos_features = bos.build_daily_bos_features(minute)
    ema_features = ema.build_daily_ema_features(minute)
    mechanisms_1m = {
        "pa_core": core,
        "bos": bos.positions_from_features(bos_features, minute.index, 1),
        "ema": ema.positions_from_features(ema_features, minute.index, 1),
    }
    mechanisms_2m = {
        "pa_core": core.shift(1).fillna(0.0),
        "bos": bos.positions_from_features(bos_features, minute.index, 2),
        "ema": ema.positions_from_features(ema_features, minute.index, 2),
    }
    weights = daily_inverse_vol_weights(minute, mechanisms_1m)
    pos1 = allocate(mechanisms_1m, weights, minute.index, 1)
    pos2 = allocate(mechanisms_2m, weights, minute.index, 2)
    variants = {
        "causal_inverse_vol_1m": (pos1, base.ONE_WAY_COST),
        "causal_inverse_vol_2m": (pos2, base.ONE_WAY_COST),
        "causal_inverse_vol_1m_double_cost": (pos1, 2.0 * base.ONE_WAY_COST),
    }
    replays = {name: base.simulate_minute(minute, pos, cost=cost) for name, (pos, cost) in variants.items()}
    pd.DataFrame([base.metrics(replay, name) for name, replay in replays.items()]).to_csv(RESULTS / "01_inverse_vol_screen.csv", index=False)
    weights.loc[base.START:base.END].to_csv(RESULTS / "02_daily_weights.csv")
    yearly = []
    for name, replay in replays.items():
        for year, group in replay.groupby(replay.index.year):
            local = group.copy()
            local["equity"] = (1.0 + local["net_return"]).cumprod()
            local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
            row = base.metrics(local, name)
            row["year"] = int(year)
            yearly.append(row)
    pd.DataFrame(yearly).to_csv(RESULTS / "03_yearly.csv", index=False)
    (RESULTS / "run_config.json").write_text(
        json.dumps(
            {
                "source": "OKX ETH-USDT-SWAP perpetual K-lines only",
                "mechanisms": ["causal daily PA core", "7/28/91D BOS", "8/32-16/64-32/128 EMA"],
                "allocator": "nonnegative inverse trailing realized volatility; weights sum to one; no expected-return estimate",
                "weight_lookback_days": WEIGHT_LOOKBACK_DAYS,
                "availability": "completed mechanism daily return at D controls D+1 + execution delay",
                "execution": "1m baseline; 2m stress", "one_way_cost": base.ONE_WAY_COST,
                "parameter_or_weight_search": "none",
            }, indent=2
        ), encoding="utf-8"
    )
    print(pd.read_csv(RESULTS / "01_inverse_vol_screen.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
