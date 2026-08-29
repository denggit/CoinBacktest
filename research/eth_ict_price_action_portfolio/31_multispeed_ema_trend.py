#!/usr/bin/env python
"""Frozen multi-speed EMA trend sleeves with 1m/2m execution."""

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


RESULTS = Path(__file__).resolve().parent / "ict_pa_v18" / "results"
EMA_PAIRS = ((8, 32), (16, 64), (32, 128))
VOL_LOOKBACK_DAYS = 30
TARGET_VOLATILITY = 0.10
GROSS_CAP = 0.60


def build_daily_ema_features(minute: pd.DataFrame) -> pd.DataFrame:
    daily = minute.resample("1D", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        source_minutes=("close", "size"),
    )
    daily = daily[daily["source_minutes"] == 1440].dropna(subset=["open", "high", "low", "close"])
    raw = pd.DataFrame(index=daily.index)
    for fast, slow in EMA_PAIRS:
        fast_ema = daily["close"].ewm(span=fast, adjust=False, min_periods=slow).mean()
        slow_ema = daily["close"].ewm(span=slow, adjust=False, min_periods=slow).mean()
        raw[f"ema_{fast}_{slow}"] = np.sign(fast_ema - slow_ema).fillna(0.0)
    realized_vol = np.log(daily["close"]).diff().rolling(VOL_LOOKBACK_DAYS, min_periods=VOL_LOOKBACK_DAYS).std(ddof=0) * np.sqrt(365.25)
    raw["realized_vol_30d"] = realized_vol
    raw["gross_budget"] = (TARGET_VOLATILITY / realized_vol.replace(0.0, np.nan)).clip(upper=GROSS_CAP).fillna(0.0)
    raw["daily_close"] = daily["close"]
    return pd.DataFrame(raw.to_numpy(), columns=raw.columns, index=raw.index + pd.Timedelta(days=1))


def positions_from_features(features: pd.DataFrame, minute_index: pd.DatetimeIndex, delay_minutes: int) -> pd.DataFrame:
    if delay_minutes not in (1, 2, 5):
        raise ValueError("delay must be 1m, 2m, or 5m")
    per_sleeve = features["gross_budget"] / len(EMA_PAIRS)
    events = pd.DataFrame(
        {
            f"ema_{fast}_{slow}": features[f"ema_{fast}_{slow}"].to_numpy() * per_sleeve.to_numpy()
            for fast, slow in EMA_PAIRS
        },
        index=features.index + pd.Timedelta(minutes=delay_minutes),
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
    features = build_daily_ema_features(minute)
    pos1 = positions_from_features(features, minute.index, 1)
    pos2 = positions_from_features(features, minute.index, 2)
    variants = {
        "multispeed_ema_1m": (pos1, base.ONE_WAY_COST),
        "multispeed_ema_2m": (pos2, base.ONE_WAY_COST),
        "multispeed_ema_1m_double_cost": (pos1, 2.0 * base.ONE_WAY_COST),
    }
    replays = {name: base.simulate_minute(minute, pos, cost=cost) for name, (pos, cost) in variants.items()}
    screen = pd.DataFrame([base.metrics(replay, name) for name, replay in replays.items()])
    screen.to_csv(RESULTS / "01_multispeed_ema_screen.csv", index=False)
    features.loc[base.START:base.END].to_csv(RESULTS / "02_daily_features.csv")
    yearly_metrics(replays).to_csv(RESULTS / "03_yearly.csv", index=False)
    rows = []
    evaluation = features.loc[base.START:base.END]
    for fast, slow in EMA_PAIRS:
        state = evaluation[f"ema_{fast}_{slow}"]
        rows.append(
            {"sleeve": f"ema_{fast}_{slow}", "long_days": int((state > 0).sum()), "short_days": int((state < 0).sum()),
             "flat_days": int((state == 0).sum()), "state_changes": int(state.ne(state.shift(1)).sum() - 1)}
        )
    pd.DataFrame(rows).to_csv(RESULTS / "04_sleeve_state_audit.csv", index=False)
    (RESULTS / "run_config.json").write_text(
        json.dumps(
            {
                "source": "OKX ETH-USDT-SWAP perpetual K-lines only", "architecture": "three independent sign(EMA fast - EMA slow) sleeves",
                "ema_pairs_days": EMA_PAIRS, "volatility_lookback_days": VOL_LOOKBACK_DAYS,
                "target_annual_volatility": TARGET_VOLATILITY, "gross_cap": GROSS_CAP,
                "execution": "completed daily bar, next midnight +1m; +2m stress", "one_way_cost": base.ONE_WAY_COST,
                "double_cost_stress": 2.0 * base.ONE_WAY_COST, "parameter_search": "none",
                "hedge_mode": "independent speed sleeves may oppose one another",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(screen.to_string(index=False))
    print("\nSLEEVES\n", pd.DataFrame(rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
