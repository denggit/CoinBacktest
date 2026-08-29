#!/usr/bin/env python
"""Frozen low-turnover multi-scale Price Action BOS portfolio.

Three independent breakout-state sleeves represent weekly, monthly, and
quarterly market structure.  A sleeve changes direction only when a completed
daily close breaks the prior completed range for its horizon.  Therefore a
quarterly bullish structure can coexist with a weekly bearish structure.

The horizons (7/28/91 calendar days), equal sleeve weights, 10% ex-ante
portfolio volatility target, and 60% gross ceiling are fixed before results.
There is no parameter search or candidate selection inside this script.
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


RESULTS = Path(__file__).resolve().parent / "ict_pa_v15" / "results"
HORIZONS = (7, 28, 91)
VOL_LOOKBACK_DAYS = 30
TARGET_VOLATILITY = 0.10
GROSS_CAP = 0.60


def build_daily_bos_features(minute: pd.DataFrame) -> pd.DataFrame:
    daily = minute.resample("1D", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        source_minutes=("close", "size"),
    )
    daily = daily[daily["source_minutes"] == 1440].dropna(subset=["open", "high", "low", "close"])
    states: dict[str, pd.Series] = {}
    for horizon in HORIZONS:
        prior_high = daily["high"].shift(1).rolling(horizon, min_periods=horizon).max()
        prior_low = daily["low"].shift(1).rolling(horizon, min_periods=horizon).min()
        event = pd.Series(np.nan, index=daily.index)
        event.loc[daily["close"] > prior_high] = 1.0
        event.loc[daily["close"] < prior_low] = -1.0
        states[f"bos_{horizon}d"] = event.ffill().fillna(0.0)

    log_return = np.log(daily["close"]).diff()
    realized_vol = log_return.rolling(VOL_LOOKBACK_DAYS, min_periods=VOL_LOOKBACK_DAYS).std(ddof=0) * np.sqrt(365.25)
    gross_budget = (TARGET_VOLATILITY / realized_vol.replace(0.0, np.nan)).clip(upper=GROSS_CAP).fillna(0.0)
    raw = pd.DataFrame(states, index=daily.index)
    raw["realized_vol_30d"] = realized_vol
    raw["gross_budget"] = gross_budget
    raw["daily_close"] = daily["close"]
    # [D,D+1) is fully known only at D+1 00:00.  Positional construction is
    # intentional and protects against pandas label alignment leakage.
    return pd.DataFrame(raw.to_numpy(), columns=raw.columns, index=raw.index + pd.Timedelta(days=1))


def positions_from_features(
    features: pd.DataFrame,
    minute_index: pd.DatetimeIndex,
    delay_minutes: int,
) -> pd.DataFrame:
    if delay_minutes not in (1, 2, 5):
        raise ValueError("delay must be one of the frozen execution stresses: 1m, 2m, 5m")
    event_index = features.index + pd.Timedelta(minutes=delay_minutes)
    per_sleeve = features["gross_budget"] / len(HORIZONS)
    events = pd.DataFrame(
        {
            f"bos_{horizon}d": features[f"bos_{horizon}d"].to_numpy() * per_sleeve.to_numpy()
            for horizon in HORIZONS
        },
        index=event_index,
    )
    return events.reindex(minute_index, method="ffill").fillna(0.0)


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
    features = build_daily_bos_features(minute)
    positions_1m = positions_from_features(features, minute.index, 1)
    positions_2m = positions_from_features(features, minute.index, 2)
    variants = {
        "multiscale_bos_1m": (positions_1m, base.ONE_WAY_COST),
        "multiscale_bos_2m": (positions_2m, base.ONE_WAY_COST),
        "multiscale_bos_1m_double_cost": (positions_1m, 2.0 * base.ONE_WAY_COST),
    }
    replays = {name: base.simulate_minute(minute, position, cost=cost) for name, (position, cost) in variants.items()}
    screen = pd.DataFrame([base.metrics(replay, name) for name, replay in replays.items()])
    screen.to_csv(RESULTS / "01_multiscale_bos_screen.csv", index=False)
    features.loc[base.START:base.END].to_csv(RESULTS / "02_daily_features.csv")
    yearly_metrics(replays).to_csv(RESULTS / "03_yearly.csv", index=False)

    audit_rows: list[dict[str, object]] = []
    evaluation = features.loc[base.START:base.END]
    for horizon in HORIZONS:
        state = evaluation[f"bos_{horizon}d"]
        audit_rows.append(
            {
                "sleeve": f"bos_{horizon}d",
                "long_days": int((state > 0).sum()),
                "short_days": int((state < 0).sum()),
                "flat_days": int((state == 0).sum()),
                "state_changes": int(state.ne(state.shift(1)).sum() - 1),
            }
        )
    pd.DataFrame(audit_rows).to_csv(RESULTS / "04_sleeve_state_audit.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["OKX ETH-USDT-SWAP 1m K-lines", "complete daily PA bars"],
            "start": [minute.index.min(), features.index.min()],
            "end": [minute.index.max(), features.index.max()],
            "rows": [len(minute), len(features)],
            "duplicate_timestamps": [minute.index.duplicated().sum(), features.index.duplicated().sum()],
        }
    ).to_csv(RESULTS / "05_data_quality.csv", index=False)
    config = {
        "source": "OKX ETH-USDT-SWAP perpetual K-lines only",
        "architecture": "three independent daily-close Price Action break-of-structure state sleeves",
        "horizons_days": HORIZONS,
        "state_rule": "long after close > prior completed range high; short after close < prior completed range low; otherwise retain state",
        "sleeve_weights": "equal gross budget",
        "volatility_lookback_days": VOL_LOOKBACK_DAYS,
        "target_annual_volatility": TARGET_VOLATILITY,
        "gross_cap": GROSS_CAP,
        "exchange_max_leverage": 15.0,
        "execution": "completed daily bar available at next midnight; enter/rebalance at +1m; +2m stress",
        "one_way_cost": base.ONE_WAY_COST,
        "double_cost_stress": 2.0 * base.ONE_WAY_COST,
        "parameter_search": "none",
        "hedge_mode": "independent horizons may hold opposing signs simultaneously",
    }
    (RESULTS / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(screen.to_string(index=False))
    print("\nSLEEVES\n", pd.DataFrame(audit_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
