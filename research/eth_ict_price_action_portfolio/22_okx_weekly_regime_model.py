#!/usr/bin/env python
"""OKX-only long-hold PA/order-flow regime model with seven-day target."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio import _okx_walkforward_bridge as base


RESULTS = Path(__file__).resolve().parent / "ict_pa_v9" / "results"
HORIZON = pd.Timedelta(days=7)
RETURN_GATE = 1.5 * 2.0 * base.ONE_WAY_COST
SIZE = 0.30
FEATURES = (
    "return_7d", "return_30d", "return_90d",
    "efficiency_30d", "range_position_20d", "range_position_60d",
    "realized_vol_30d", "volume_relative_30d",
    "delta_ratio_1d", "delta_ratio_7d", "large_delta_ratio_7d",
    "flow_price_divergence_7d",
)


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def build_daily_features(minute: pd.DataFrame, trade15: pd.DataFrame) -> pd.DataFrame:
    price = minute.resample("1D", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"), source_minutes=("close", "size"),
    )
    flow = trade15.resample("1D", label="left", closed="left").agg(
        notional=("notional", "sum"), delta_notional=("delta_notional", "sum"),
        large_delta_notional=("large_delta_notional", "sum"), source_bars=("close", "size"),
    )
    daily = price.join(flow, how="inner")
    daily = daily[(daily["source_minutes"] == 1440) & (daily["source_bars"] == 96)]
    close = daily["close"]
    log_return = np.log(close).diff()
    data: dict[str, pd.Series] = {
        "return_7d": np.log(close).diff(7),
        "return_30d": np.log(close).diff(30),
        "return_90d": np.log(close).diff(90),
    }
    data["efficiency_30d"] = safe_divide(data["return_30d"].abs(), log_return.abs().rolling(30, min_periods=30).sum())
    for lookback in (20, 60):
        prior_high = daily["high"].shift(1).rolling(lookback, min_periods=lookback).max()
        prior_low = daily["low"].shift(1).rolling(lookback, min_periods=lookback).min()
        data[f"range_position_{lookback}d"] = safe_divide(close - prior_low, prior_high - prior_low)
    data["realized_vol_30d"] = log_return.shift(1).rolling(30, min_periods=30).std(ddof=0) * np.sqrt(365.25)
    data["volume_relative_30d"] = safe_divide(daily["notional"], daily["notional"].shift(1).rolling(30, min_periods=30).median())
    data["delta_ratio_1d"] = safe_divide(daily["delta_notional"], daily["notional"])
    data["delta_ratio_7d"] = safe_divide(daily["delta_notional"].rolling(7).sum(), daily["notional"].rolling(7).sum())
    data["large_delta_ratio_7d"] = safe_divide(daily["large_delta_notional"].rolling(7).sum(), daily["notional"].rolling(7).sum())
    data["flow_price_divergence_7d"] = data["return_7d"] * -data["delta_ratio_7d"]
    raw = pd.DataFrame(data, index=daily.index).replace([np.inf, -np.inf], np.nan)
    # The daily bar labelled D is completed only at D+1.  Positional transfer
    # avoids pandas label alignment with the shifted availability index.
    return pd.DataFrame(raw.to_numpy(), columns=raw.columns, index=raw.index + pd.Timedelta(days=1))


def build_samples(features: pd.DataFrame, minute: pd.DataFrame) -> pd.DataFrame:
    sample = features.copy()
    sample["available_time"] = sample.index
    sample["execution_time_1m"] = sample.index + pd.Timedelta(minutes=1)
    sample["exit_time_1m"] = sample["execution_time_1m"] + HORIZON
    opens = minute["open"]
    entry = opens.reindex(pd.DatetimeIndex(sample["execution_time_1m"]))
    exit_ = opens.reindex(pd.DatetimeIndex(sample["exit_time_1m"]))
    sample["future_return_7d"] = exit_.to_numpy() / entry.to_numpy() - 1.0
    return sample.dropna(subset=[*FEATURES, "future_return_7d"])


def walk_forward(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parts: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    for test_start in pd.date_range("2023-01-01", "2026-08-01", freq="MS"):
        test_end = test_start + pd.offsets.MonthBegin(1)
        train = samples[
            (samples.index >= test_start - pd.Timedelta(days=730))
            & (samples["exit_time_1m"] < test_start)
        ]
        test = samples[(samples.index >= test_start) & (samples.index < test_end)]
        if len(train) < 300 or test.empty:
            continue
        target = train["future_return_7d"]
        lo, hi = target.quantile([0.01, 0.99])
        model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        model.fit(train[list(FEATURES)], target.clip(lo, hi))
        expected = model.predict(test[list(FEATURES)])
        part = test[["available_time", "execution_time_1m", "exit_time_1m", "future_return_7d"]].copy()
        part["expected_return"] = expected
        part["fold"] = str(test_start.date())
        parts.append(part)
        folds.append(
            {
                "test_month": str(test_start.date()),
                "train_start": str(train.index.min()),
                "train_label_exit_max": str(train["exit_time_1m"].max()),
                "train_rows": len(train), "test_rows": len(test),
                "return_correlation": test["future_return_7d"].corr(pd.Series(expected, index=test.index)),
            }
        )
    if not parts:
        raise RuntimeError("no weekly-regime folds")
    return pd.concat(parts), pd.DataFrame(folds)


def sticky_state(predictions: pd.DataFrame, minute_index: pd.DatetimeIndex, delay_minutes: int) -> pd.Series:
    state = 0.0
    values: list[float] = []
    for expected in predictions["expected_return"]:
        if expected >= RETURN_GATE:
            state = SIZE
        elif expected <= -RETURN_GATE:
            state = -SIZE
        elif state > 0 and expected < 0.0:
            state = 0.0
        elif state < 0 and expected > 0.0:
            state = 0.0
        values.append(state)
    events = pd.Series(values, index=predictions.index + pd.Timedelta(minutes=delay_minutes))
    return events.reindex(minute_index, method="ffill").fillna(0.0)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, trade15 = base.load_inputs()
    features = build_daily_features(minute, trade15)
    samples = build_samples(features, minute)
    predictions, folds = walk_forward(samples)
    model_1m = sticky_state(predictions, minute.index, 1)
    model_2m = sticky_state(predictions, minute.index, 2)
    core = base.core_state(minute) * 0.75
    variants = {
        "daily_pa_core_only": pd.DataFrame({"core": core}, index=minute.index),
        "weekly_regime_only_1m": pd.DataFrame({"regime": model_1m}, index=minute.index),
        "core_plus_weekly_regime_1m": pd.DataFrame({"core": core, "regime": model_1m}, index=minute.index),
        "core_plus_weekly_regime_2m": pd.DataFrame({"core": core, "regime": model_2m}, index=minute.index),
    }
    rows: list[dict[str, object]] = []
    replays: dict[str, pd.DataFrame] = {}
    for name, positions in variants.items():
        replay = base.simulate_minute(minute, positions)
        rows.append(base.metrics(replay, name))
        replays[name] = replay
    screen = pd.DataFrame(rows)
    screen.to_csv(RESULTS / "01_weekly_regime_screen.csv", index=False)
    folds.to_csv(RESULTS / "02_walkforward_folds.csv", index=False)
    predictions.reset_index(names="feature_available_time").to_csv(RESULTS / "03_oos_predictions.csv", index=False)
    yearly: list[dict[str, object]] = []
    for name, replay in replays.items():
        for year, group in replay.groupby(replay.index.year):
            local = group.copy()
            local["equity"] = (1.0 + local["net_return"]).cumprod()
            local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
            row = base.metrics(local, name)
            row["year"] = year
            yearly.append(row)
    pd.DataFrame(yearly).to_csv(RESULTS / "04_yearly.csv", index=False)
    (RESULTS / "run_config.json").write_text(
        json.dumps(
            {
                "source": "OKX ETH-USDT-SWAP only",
                "feature_availability": "completed daily price and OKX trade-flow bar, positionally shifted to D+1",
                "target": "next 7D return from D+1m open",
                "model": "fixed standardized Ridge(alpha=10), trailing 730D, monthly refit, 7D label purge",
                "execution": "1m main, 2m delay stress",
                "entry_gate": RETURN_GATE,
                "one_way_cost": base.ONE_WAY_COST,
                "size": SIZE,
                "gross_cap": 0.75,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(screen.to_string(index=False))
    print("\nFOLDS\n", folds["return_correlation"].describe().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
