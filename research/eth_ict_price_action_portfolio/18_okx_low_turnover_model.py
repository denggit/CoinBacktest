#!/usr/bin/env python
"""Low-turnover OKX PA/microstructure models with strict 1m execution.

This is an independent, mechanism-led follow-up to the rejected 4H direction
classifier.  It fixes the economic horizon at 12H (two decisions per day) so a
signal must clear the 0.10% round-trip fee hurdle instead of optimizing a
high-frequency classification score.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio import _okx_walkforward_bridge as base


RESULTS = Path(__file__).resolve().parent / "ict_pa_v5" / "results"
HORIZON_HOURS = 12
ROUND_TRIP_COST = 2.0 * base.ONE_WAY_COST
RETURN_GATE = 1.5 * ROUND_TRIP_COST
TACTICAL_SIZE = 0.20


def build_samples(features: pd.DataFrame, minute: pd.DataFrame) -> pd.DataFrame:
    """Create non-overlapping 12H observations from causally available bars."""
    sample = features[(features.index.minute == 0) & features.index.hour.isin([0, 12])].copy()
    sample["available_time"] = sample.index
    sample["execution_time"] = sample.index + pd.Timedelta(minutes=1)
    sample["exit_time"] = sample["execution_time"] + pd.Timedelta(hours=HORIZON_HOURS)
    opens = minute["open"]
    sample["entry_price"] = opens.reindex(pd.DatetimeIndex(sample["execution_time"])).to_numpy()
    sample["exit_price"] = opens.reindex(pd.DatetimeIndex(sample["exit_time"])).to_numpy()
    sample["future_return"] = sample["exit_price"] / sample["entry_price"] - 1.0
    sample["label_up"] = (sample["future_return"] > 0.0).astype(int)
    return sample.dropna(subset=[*base.FEATURE_COLUMNS, "future_return"])


def walk_forward(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Monthly OOS predictions with training-only winsorization and label purge."""
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    first_test = pd.Timestamp("2023-01-01")
    last_test = min(pd.Timestamp("2026-08-01"), samples.index.max().to_period("M").start_time)
    for test_start in pd.date_range(first_test, last_test, freq="MS"):
        test_end = test_start + pd.offsets.MonthBegin(1)
        train_start = test_start - pd.Timedelta(days=730)
        train = samples[(samples.index >= train_start) & (samples["exit_time"] < test_start)]
        test = samples[(samples.index >= test_start) & (samples.index < test_end)]
        if len(train) < 500 or test.empty or train["label_up"].nunique() < 2:
            continue

        direction = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.10, solver="lbfgs", max_iter=1000, random_state=18)),
            ]
        )
        magnitude = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))])
        target = train["future_return"]
        lo, hi = target.quantile([0.01, 0.99])
        direction.fit(train[list(base.FEATURE_COLUMNS)], train["label_up"])
        magnitude.fit(train[list(base.FEATURE_COLUMNS)], target.clip(lo, hi))
        probability = direction.predict_proba(test[list(base.FEATURE_COLUMNS)])[:, 1]
        expected_return = magnitude.predict(test[list(base.FEATURE_COLUMNS)])

        part = test[["available_time", "execution_time", "exit_time", "future_return", "label_up"]].copy()
        part["probability_up"] = probability
        part["expected_return"] = expected_return
        part["fold"] = str(test_start.date())
        predictions.append(part)
        auc = roc_auc_score(test["label_up"], probability) if test["label_up"].nunique() > 1 else np.nan
        fold_rows.append(
            {
                "test_month": str(test_start.date()),
                "train_start": str(train.index.min()),
                "train_label_exit_max": str(train["exit_time"].max()),
                "train_rows": len(train),
                "test_rows": len(test),
                "auc": auc,
                "brier": brier_score_loss(test["label_up"], probability),
                "return_correlation": test["future_return"].corr(pd.Series(expected_return, index=test.index)),
            }
        )
    if not predictions:
        raise RuntimeError("no valid 12H walk-forward folds")
    return pd.concat(predictions), pd.DataFrame(fold_rows)


def model_events(predictions: pd.DataFrame, delay_minutes: int) -> pd.Series:
    """Direction and magnitude must independently clear frozen fee-aware gates."""
    long = (predictions["probability_up"] >= 0.55) & (predictions["expected_return"] >= RETURN_GATE)
    short = (predictions["probability_up"] <= 0.45) & (predictions["expected_return"] <= -RETURN_GATE)
    values = np.select([long, short], [TACTICAL_SIZE, -TACTICAL_SIZE], default=0.0)
    index = pd.DatetimeIndex(predictions["available_time"]) + pd.Timedelta(minutes=delay_minutes)
    return pd.Series(values, index=index, name="model")


def pa_event_features(features: pd.DataFrame) -> pd.DataFrame:
    """Frozen PA/MM confirmation events evaluated only on available features."""
    # Extreme location plus opposing aggressive flow defines absorption.  The
    # fixed 0.20/0.80 candle locations are semantic fifths, not fitted cutoffs.
    long_reclaim = (
        (features["sweep_low_24h"] > 0.5)
        & (features["close_location"] >= 0.80)
        & (features["delta_ratio_1h"] < 0.0)
        & (features["body_fraction"] >= 0.50)
    )
    short_reject = (
        (features["sweep_high_24h"] > 0.5)
        & (features["close_location"] <= 0.20)
        & (features["delta_ratio_1h"] > 0.0)
        & (features["body_fraction"] >= 0.50)
    )
    return pd.DataFrame({"long": long_reclaim, "short": short_reject}, index=features.index)


def rule_events(features: pd.DataFrame, delay_minutes: int) -> pd.Series:
    events = pa_event_features(features)
    values = np.select([events["long"], events["short"]], [TACTICAL_SIZE, -TACTICAL_SIZE], default=np.nan)
    index = events.index + pd.Timedelta(minutes=delay_minutes)
    # Each confirmed liquidity-reclaim event has a fixed 24H maximum life.  A
    # later event can replace it; absence of an event does not cause churn.
    sparse = pd.Series(values, index=index).dropna()
    if sparse.empty:
        return sparse
    expiry = pd.Series(0.0, index=sparse.index + pd.Timedelta(hours=24))
    combined = pd.concat([expiry.rename("value"), sparse.rename("value")]).sort_index(kind="stable")
    combined = combined[~combined.index.duplicated(keep="last")]
    return combined


def align_events(events: pd.Series, minute_index: pd.DatetimeIndex) -> pd.Series:
    if events.empty:
        return pd.Series(0.0, index=minute_index)
    return events.reindex(minute_index, method="ffill").fillna(0.0)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, trade = base.load_inputs()
    features = base.build_hourly_features(trade)
    samples = build_samples(features, minute)
    predictions, folds = walk_forward(samples)

    model_1m = align_events(model_events(predictions, 1), minute.index)
    model_2m = align_events(model_events(predictions, 2), minute.index)
    rule_1m = align_events(rule_events(features, 1), minute.index)
    rule_2m = align_events(rule_events(features, 2), minute.index)
    core = base.core_state(minute) * 0.75

    variants = {
        "daily_pa_core_only": pd.DataFrame({"core": core}, index=minute.index),
        "fee_aware_model_only_1m": pd.DataFrame({"model": model_1m}, index=minute.index),
        "pa_absorption_rule_only_1m": pd.DataFrame({"rule": rule_1m}, index=minute.index),
        "core_plus_model_1m": pd.DataFrame({"core": core, "model": model_1m}, index=minute.index),
        "core_plus_model_2m": pd.DataFrame({"core": core, "model": model_2m}, index=minute.index),
        "core_plus_rule_1m": pd.DataFrame({"core": core, "rule": rule_1m}, index=minute.index),
        "core_plus_rule_2m": pd.DataFrame({"core": core, "rule": rule_2m}, index=minute.index),
        "core_plus_model_rule_1m": pd.DataFrame({"core": core, "model": model_1m, "rule": rule_1m}, index=minute.index),
    }
    screens: list[dict[str, object]] = []
    replays: dict[str, pd.DataFrame] = {}
    for name, positions in variants.items():
        replay = base.simulate_minute(minute, positions)
        screens.append(base.metrics(replay, name))
        replays[name] = replay

    screen = pd.DataFrame(screens)
    screen.to_csv(RESULTS / "01_low_turnover_screen.csv", index=False)
    folds.to_csv(RESULTS / "02_walkforward_folds.csv", index=False)
    predictions.reset_index(drop=True).to_csv(RESULTS / "03_oos_predictions.csv", index=False)
    pd.DataFrame(
        {
            "feature_time": features.index,
            "long_event": pa_event_features(features)["long"].to_numpy(),
            "short_event": pa_event_features(features)["short"].to_numpy(),
        }
    ).to_csv(RESULTS / "04_pa_events.csv", index=False)

    yearly_rows: list[dict[str, object]] = []
    for name, replay in replays.items():
        for year, group in replay.groupby(replay.index.year):
            local = group.copy()
            local["equity"] = (1.0 + local["net_return"]).cumprod()
            local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
            row = base.metrics(local, name)
            row["year"] = year
            yearly_rows.append(row)
    pd.DataFrame(yearly_rows).to_csv(RESULTS / "05_yearly.csv", index=False)

    config = {
        "source": "OKX ETH-USDT-SWAP only",
        "feature_availability": "completed 1H features shifted to bar-end; incomplete hours dropped",
        "decision_horizon": "12H, non-overlapping, 00:00 and 12:00 local loader time",
        "execution": "available time +1m; +2m fixed delay stress",
        "model": "L2 logistic direction AND Ridge expected-return confirmation",
        "training": "trailing 730 days, monthly retrain, test-boundary label purge, training-only 1/99% target winsorization",
        "probability_gates": {"long": 0.55, "short": 0.45},
        "expected_return_gate": RETURN_GATE,
        "round_trip_cost": ROUND_TRIP_COST,
        "tactical_size": TACTICAL_SIZE,
        "strategy_gross_cap": 0.75,
        "exchange_leverage_cap": 15.0,
    }
    (RESULTS / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(screen.to_string(index=False))
    print("\nFOLDS\n", folds[["auc", "brier", "return_correlation"]].describe().to_string())
    print("\nEVENT COUNTS", int(pa_event_features(features)["long"].sum()), int(pa_event_features(features)["short"].sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
