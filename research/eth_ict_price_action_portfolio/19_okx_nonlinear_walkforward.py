#!/usr/bin/env python
"""Strongly regularized nonlinear OKX PA/microstructure walk-forward model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio import _okx_low_turnover_bridge as low_turnover
from research.eth_ict_price_action_portfolio import _okx_walkforward_bridge as base


RESULTS = Path(__file__).resolve().parent / "ict_pa_v6" / "results"


def walk_forward(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    last_test = min(pd.Timestamp("2026-08-01"), samples.index.max().to_period("M").start_time)
    for test_start in pd.date_range("2023-01-01", last_test, freq="MS"):
        test_end = test_start + pd.offsets.MonthBegin(1)
        train = samples[
            (samples.index >= test_start - pd.Timedelta(days=730))
            & (samples["exit_time"] < test_start)
        ]
        test = samples[(samples.index >= test_start) & (samples.index < test_end)]
        if len(train) < 500 or test.empty or train["label_up"].nunique() < 2:
            continue

        # Complexity is frozen before evaluation: at most seven leaves, at
        # least 100 samples per leaf, slow learning, and strong L2 shrinkage.
        direction = HistGradientBoostingClassifier(
            learning_rate=0.03,
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=100,
            l2_regularization=10.0,
            random_state=19,
        )
        magnitude = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.03,
            max_iter=100,
            max_leaf_nodes=7,
            min_samples_leaf=100,
            l2_regularization=10.0,
            random_state=19,
        )
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
        prediction_parts.append(part)
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
    if not prediction_parts:
        raise RuntimeError("no nonlinear walk-forward folds")
    return pd.concat(prediction_parts), pd.DataFrame(fold_rows)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, trade = base.load_inputs()
    features = base.build_hourly_features(trade)
    samples = low_turnover.build_samples(features, minute)
    predictions, folds = walk_forward(samples)
    tactical_1m = low_turnover.align_events(low_turnover.model_events(predictions, 1), minute.index)
    tactical_2m = low_turnover.align_events(low_turnover.model_events(predictions, 2), minute.index)
    core = base.core_state(minute) * 0.75

    variants = {
        "daily_pa_core_only": pd.DataFrame({"core": core}, index=minute.index),
        "nonlinear_model_only_1m": pd.DataFrame({"tactical": tactical_1m}, index=minute.index),
        "core_plus_nonlinear_1m": pd.DataFrame({"core": core, "tactical": tactical_1m}, index=minute.index),
        "core_plus_nonlinear_2m": pd.DataFrame({"core": core, "tactical": tactical_2m}, index=minute.index),
    }
    rows: list[dict[str, object]] = []
    replays: dict[str, pd.DataFrame] = {}
    for name, positions in variants.items():
        replay = base.simulate_minute(minute, positions)
        rows.append(base.metrics(replay, name))
        replays[name] = replay
    screen = pd.DataFrame(rows)
    screen.to_csv(RESULTS / "01_nonlinear_screen.csv", index=False)
    folds.to_csv(RESULTS / "02_walkforward_folds.csv", index=False)
    predictions.reset_index(drop=True).to_csv(RESULTS / "03_oos_predictions.csv", index=False)

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
                "model": "frozen shallow histogram gradient boosting direction and absolute-return models",
                "complexity": {"max_leaf_nodes": 7, "min_samples_leaf": 100, "max_iter": 100, "learning_rate": 0.03, "l2": 10.0},
                "training": "trailing 730 days, monthly, purged 12H labels",
                "execution": "1m main and 2m delay stress",
                "expected_return_gate": low_turnover.RETURN_GATE,
                "one_way_cost": base.ONE_WAY_COST,
                "strategy_gross_cap": 0.75,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(screen.to_string(index=False))
    print("\nFOLDS\n", folds[["auc", "brier", "return_correlation"]].describe().to_string())
    print("\nACTIVE DECISIONS", int((tactical_1m.groupby(tactical_1m.index.floor("12h")).last() != 0).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

