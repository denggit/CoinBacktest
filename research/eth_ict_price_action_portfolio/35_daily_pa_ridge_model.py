#!/usr/bin/env python
"""Strict walk-forward daily Price Action Ridge model on OKX only."""

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


RESULTS = Path(__file__).resolve().parent / "ict_pa_v22" / "results"
FEATURES = (
    "return_1d", "return_3d", "return_7d", "return_14d", "return_30d", "return_90d",
    "close_location", "body_fraction", "range_over_20d", "realized_vol_30d",
    "distance_high_20d", "distance_low_20d",
)
TRAIN_DAYS = 730
RIDGE_ALPHA = 10.0
RETURN_GATE = 1.5 * 2.0 * base.ONE_WAY_COST
NOTIONAL = 0.30


def build_samples(minute: pd.DataFrame) -> pd.DataFrame:
    daily = minute.resample("1D", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        source_minutes=("close", "size"),
    )
    daily = daily[daily["source_minutes"] == 1440].dropna(subset=["open", "high", "low", "close"])
    log_close = np.log(daily["close"])
    candle_range = (daily["high"] - daily["low"]).replace(0.0, np.nan)
    sample = pd.DataFrame(index=daily.index)
    for horizon in (1, 3, 7, 14, 30, 90):
        sample[f"return_{horizon}d"] = log_close.diff(horizon)
    sample["close_location"] = (daily["close"] - daily["low"]) / candle_range
    sample["body_fraction"] = (daily["close"] - daily["open"]).abs() / candle_range
    sample["range_over_20d"] = (daily["high"] / daily["low"] - 1.0) / (daily["high"] / daily["low"] - 1.0).shift(1).rolling(20).median()
    sample["realized_vol_30d"] = log_close.diff().rolling(30).std(ddof=0) * np.sqrt(365.25)
    sample["distance_high_20d"] = daily["close"] / daily["high"].shift(1).rolling(20).max() - 1.0
    sample["distance_low_20d"] = daily["close"] / daily["low"].shift(1).rolling(20).min() - 1.0
    # Features from day D are available D+1 00:00; execution waits one minute.
    available_index = sample.index + pd.Timedelta(days=1)
    out = pd.DataFrame(sample.to_numpy(), columns=sample.columns, index=available_index)
    out["available_time"] = out.index
    out["execution_time"] = out.index + pd.Timedelta(minutes=1)
    out["exit_time"] = out["execution_time"] + pd.Timedelta(days=1)
    opens = minute["open"]
    entry = opens.reindex(pd.DatetimeIndex(out["execution_time"]))
    exit_ = opens.reindex(pd.DatetimeIndex(out["exit_time"]))
    out["future_return_1d"] = exit_.to_numpy() / entry.to_numpy() - 1.0
    return out.replace([np.inf, -np.inf], np.nan).dropna(subset=[*FEATURES, "future_return_1d"])


def walk_forward(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    folds = []
    coefficients = []
    for test_start in pd.date_range("2022-01-01", "2026-08-01", freq="MS"):
        test_end = test_start + pd.offsets.MonthBegin(1)
        train = samples[(samples.index >= test_start - pd.Timedelta(days=TRAIN_DAYS)) & (samples["exit_time"] < test_start)]
        test = samples[(samples.index >= test_start) & (samples.index < test_end)]
        if len(train) < 400 or test.empty:
            continue
        target = train["future_return_1d"]
        lo, hi = target.quantile([0.01, 0.99])
        model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=RIDGE_ALPHA))])
        model.fit(train[list(FEATURES)], target.clip(lo, hi))
        expected = model.predict(test[list(FEATURES)])
        part = test[["available_time", "execution_time", "exit_time", "future_return_1d"]].copy()
        part["expected_return"] = expected
        part["fold"] = str(test_start.date())
        predictions.append(part)
        folds.append(
            {
                "test_month": str(test_start.date()), "train_start": train.index.min(), "train_label_exit_max": train["exit_time"].max(),
                "train_rows": len(train), "test_rows": len(test),
                "return_correlation": test["future_return_1d"].corr(pd.Series(expected, index=test.index)),
                "mean_expected": float(np.mean(expected)), "std_expected": float(np.std(expected)),
            }
        )
        for feature, value in zip(FEATURES, model.named_steps["ridge"].coef_):
            coefficients.append({"test_month": str(test_start.date()), "feature": feature, "coefficient": float(value)})
    if not predictions:
        raise RuntimeError("no valid daily walk-forward folds")
    return pd.concat(predictions), pd.DataFrame(folds), pd.DataFrame(coefficients)


def prediction_positions(predictions: pd.DataFrame, minute_index: pd.DatetimeIndex, delay_minutes: int) -> pd.Series:
    expected = predictions["expected_return"]
    desired = pd.Series(0.0, index=predictions.index)
    desired.loc[expected >= RETURN_GATE] = NOTIONAL
    desired.loc[expected <= -RETURN_GATE] = -NOTIONAL
    event_index = pd.DatetimeIndex(predictions["available_time"]) + pd.Timedelta(minutes=delay_minutes)
    events = pd.Series(desired.to_numpy(), index=event_index)
    aligned = events.reindex(minute_index, method="ffill").fillna(0.0)
    if len(events):
        aligned.loc[aligned.index >= events.index.max() + pd.Timedelta(days=1)] = 0.0
    return aligned


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, _ = base.load_inputs()
    samples = build_samples(minute)
    predictions, folds, coefficients = walk_forward(samples)
    model1 = prediction_positions(predictions, minute.index, 1)
    model2 = prediction_positions(predictions, minute.index, 2)
    core = base.core_state(minute) * 0.75
    variants = {
        "daily_pa_ridge_only_1m": pd.DataFrame({"model": model1}, index=minute.index),
        "daily_pa_ridge_only_2m": pd.DataFrame({"model": model2}, index=minute.index),
        "core_plus_daily_pa_ridge_1m": pd.DataFrame({"core": core, "model": model1}, index=minute.index),
        "core_plus_daily_pa_ridge_2m": pd.DataFrame({"core": core.shift(1).fillna(0.0), "model": model2}, index=minute.index),
    }
    replays = {name: base.simulate_minute(minute, pos) for name, pos in variants.items()}
    screen = pd.DataFrame([base.metrics(replay, name) for name, replay in replays.items()])
    screen.to_csv(RESULTS / "01_daily_ridge_screen.csv", index=False)
    folds.to_csv(RESULTS / "02_walkforward_folds.csv", index=False)
    predictions.reset_index(names="feature_available_time").to_csv(RESULTS / "03_oos_predictions.csv", index=False)
    coefficients.to_csv(RESULTS / "04_coefficients.csv", index=False)
    yearly = []
    for name, replay in replays.items():
        for year, group in replay.groupby(replay.index.year):
            local = group.copy(); local["equity"] = (1 + local["net_return"]).cumprod(); local["drawdown"] = local["equity"] / local["equity"].cummax() - 1
            row = base.metrics(local, name); row["year"] = int(year); yearly.append(row)
    pd.DataFrame(yearly).to_csv(RESULTS / "05_yearly.csv", index=False)
    (RESULTS / "run_config.json").write_text(json.dumps({
        "source": "OKX ETH-USDT-SWAP perpetual 1m K-lines only", "features": FEATURES,
        "model": "StandardScaler + Ridge(alpha=10)", "training": "trailing 730D; monthly refit; 1D label purge",
        "prediction_horizon": "next natural-day open-to-open", "economic_gate": RETURN_GATE,
        "execution": "feature day complete, next midnight +1m; +2m stress", "notional": NOTIONAL,
        "one_way_cost": base.ONE_WAY_COST, "parameter_search": "none",
    }, indent=2), encoding="utf-8")
    print(screen.to_string(index=False)); print("\nFOLDS\n", folds[["return_correlation", "mean_expected", "std_expected"]].describe().to_string())
    print("ACTIVE DAYS", int((predictions["expected_return"].abs() >= RETURN_GATE).sum()), "OF", len(predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
