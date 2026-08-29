#!/usr/bin/env python
"""OKX-only test of quarter-hour opening order-flow predictability.

The external paper is used only to freeze the hypothesis.  Every feature,
label, execution price, and validation result comes from OKX ETH-USDT-SWAP.
The first 10 seconds of each quarter-hour are observed; execution occurs at
the next 1m open (with 2m as a latency stress), never inside the observed bar.
"""

from __future__ import annotations

import json
import sqlite3
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
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader


RESULTS = Path(__file__).resolve().parent / "ict_pa_v7" / "results"
HORIZON = pd.Timedelta(hours=8)
ROUND_TRIP_COST = 2.0 * base.ONE_WAY_COST
RETURN_GATE = 1.5 * ROUND_TRIP_COST
SIZE = 0.20
QH_FEATURES = (
    "opening_oi",
    "opening_return",
    "opening_notional_relative",
    "oi_lag_1",
    "oi_lag_4",
    "oi_lag_12",
    "return_lag_1",
    "return_lag_4",
    "return_24h",
    "return_72h",
    "close_location",
    "realized_vol_24h",
    "delta_ratio_24h",
    "large_delta_ratio_4h",
)


def load_quarter_hour_openings() -> pd.DataFrame:
    loader = OKXTradeBarLoader(symbol="ETH-USDT-SWAP", timeframe="10s")
    columns = [
        "timestamp", "open", "high", "low", "close", "notional",
        "delta_notional", "large_delta_notional", "trades_count",
    ]
    quoted_table = loader.table_name.replace('"', '""')
    query = f"""
        SELECT {', '.join(columns)}
        FROM \"{quoted_table}\"
        WHERE timestamp >= ? AND timestamp <= ?
          AND CAST(strftime('%M', timestamp) AS INTEGER) % 15 = 0
          AND CAST(strftime('%S', timestamp) AS INTEGER) = 0
        ORDER BY timestamp
    """
    with sqlite3.connect(loader.db_path) as connection:
        frame = pd.read_sql_query(query, connection, params=("2022-01-01 00:00:00", "2026-08-15 23:59:59"))
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.set_index("timestamp").sort_index()
    return frame[~frame.index.duplicated(keep="last")]


def build_samples(opening: pd.DataFrame, hourly: pd.DataFrame, minute: pd.DataFrame) -> pd.DataFrame:
    sample = pd.DataFrame(index=opening.index)
    sample["opening_oi"] = opening["delta_notional"] / opening["notional"].replace(0.0, np.nan)
    sample["opening_return"] = opening["close"] / opening["open"] - 1.0
    prior_median = opening["notional"].shift(1).rolling(96 * 7, min_periods=96).median()
    sample["opening_notional_relative"] = opening["notional"] / prior_median.replace(0.0, np.nan)
    for lag in (1, 4, 12):
        sample[f"oi_lag_{lag}"] = sample["opening_oi"].shift(lag)
    for lag in (1, 4):
        sample[f"return_lag_{lag}"] = sample["opening_return"].shift(lag)
    public = hourly.reindex(sample.index, method="ffill")
    for name in ("return_24h", "return_72h", "close_location", "realized_vol_24h", "delta_ratio_24h", "large_delta_ratio_4h"):
        sample[name] = public[name].to_numpy()

    sample["available_time"] = sample.index + pd.Timedelta(seconds=10)
    sample["execution_time_1m"] = sample.index + pd.Timedelta(minutes=1)
    sample["exit_time_1m"] = sample["execution_time_1m"] + HORIZON
    opens = minute["open"]
    entry = opens.reindex(pd.DatetimeIndex(sample["execution_time_1m"]))
    exit_ = opens.reindex(pd.DatetimeIndex(sample["exit_time_1m"]))
    sample["future_return_8h"] = exit_.to_numpy() / entry.to_numpy() - 1.0
    return sample.replace([np.inf, -np.inf], np.nan).dropna(subset=[*QH_FEATURES, "future_return_8h"])


def walk_forward(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for test_start in pd.date_range("2023-01-01", "2026-08-01", freq="MS"):
        test_end = test_start + pd.offsets.MonthBegin(1)
        train = samples[
            (samples.index >= test_start - pd.Timedelta(days=365))
            & (samples["exit_time_1m"] < test_start)
        ]
        test = samples[(samples.index >= test_start) & (samples.index < test_end)]
        if len(train) < 10_000 or test.empty:
            continue
        target = train["future_return_8h"]
        lo, hi = target.quantile([0.01, 0.99])
        model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
        model.fit(train[list(QH_FEATURES)], target.clip(lo, hi))
        expected = model.predict(test[list(QH_FEATURES)])
        part = test[["available_time", "execution_time_1m", "exit_time_1m", "future_return_8h", "opening_oi", "return_24h"]].copy()
        part["expected_return"] = expected
        part["fold"] = str(test_start.date())
        prediction_parts.append(part)
        fold_rows.append(
            {
                "test_month": str(test_start.date()),
                "train_start": str(train.index.min()),
                "train_label_exit_max": str(train["exit_time_1m"].max()),
                "train_rows": len(train),
                "test_rows": len(test),
                "return_correlation": test["future_return_8h"].corr(pd.Series(expected, index=test.index)),
                "mean_prediction": float(np.mean(expected)),
                "prediction_std": float(np.std(expected)),
            }
        )
    if not prediction_parts:
        raise RuntimeError("no quarter-hour walk-forward folds")
    return pd.concat(prediction_parts), pd.DataFrame(fold_rows)


def non_overlapping_model_events(predictions: pd.DataFrame, delay_minutes: int) -> pd.Series:
    events: dict[pd.Timestamp, float] = {}
    next_available = pd.Timestamp.min
    for timestamp, row in predictions.sort_index().iterrows():
        execution = timestamp + pd.Timedelta(minutes=delay_minutes)
        if execution < next_available:
            continue
        expected = float(row["expected_return"])
        if abs(expected) < RETURN_GATE:
            continue
        events[execution] = SIZE if expected > 0 else -SIZE
        expiry = execution + HORIZON
        events[expiry] = 0.0
        next_available = expiry
    return pd.Series(events, dtype=float).sort_index()


def extreme_oi_events(samples: pd.DataFrame, delay_minutes: int) -> pd.Series:
    threshold = samples["opening_oi"].abs().shift(1).rolling(96 * 180, min_periods=96 * 60).quantile(0.99)
    events: dict[pd.Timestamp, float] = {}
    next_available = pd.Timestamp.min
    for timestamp, oi, gate in zip(samples.index, samples["opening_oi"], threshold):
        execution = timestamp + pd.Timedelta(minutes=delay_minutes)
        if execution < next_available or pd.isna(gate) or abs(oi) < gate:
            continue
        events[execution] = SIZE if oi > 0 else -SIZE
        expiry = execution + HORIZON
        events[expiry] = 0.0
        next_available = expiry
    return pd.Series(events, dtype=float).sort_index()


def align(events: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    if events.empty:
        return pd.Series(0.0, index=index)
    return events.reindex(index, method="ffill").fillna(0.0)


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, trade15 = base.load_inputs()
    hourly = base.build_hourly_features(trade15)
    opening = load_quarter_hour_openings()
    samples = build_samples(opening, hourly, minute)
    predictions, folds = walk_forward(samples)
    model_1m = align(non_overlapping_model_events(predictions, 1), minute.index)
    model_2m = align(non_overlapping_model_events(predictions, 2), minute.index)
    rule_1m = align(extreme_oi_events(samples, 1), minute.index)
    rule_2m = align(extreme_oi_events(samples, 2), minute.index)
    core = base.core_state(minute) * 0.75
    variants = {
        "daily_pa_core_only": pd.DataFrame({"core": core}, index=minute.index),
        "qh_ridge_only_1m": pd.DataFrame({"qh": model_1m}, index=minute.index),
        "qh_extreme_oi_only_1m": pd.DataFrame({"qh": rule_1m}, index=minute.index),
        "core_plus_qh_ridge_1m": pd.DataFrame({"core": core, "qh": model_1m}, index=minute.index),
        "core_plus_qh_ridge_2m": pd.DataFrame({"core": core, "qh": model_2m}, index=minute.index),
        "core_plus_qh_extreme_1m": pd.DataFrame({"core": core, "qh": rule_1m}, index=minute.index),
        "core_plus_qh_extreme_2m": pd.DataFrame({"core": core, "qh": rule_2m}, index=minute.index),
    }
    rows: list[dict[str, object]] = []
    replays: dict[str, pd.DataFrame] = {}
    for name, positions in variants.items():
        replay = base.simulate_minute(minute, positions)
        rows.append(base.metrics(replay, name))
        replays[name] = replay
    screen = pd.DataFrame(rows)
    screen.to_csv(RESULTS / "01_quarter_hour_screen.csv", index=False)
    folds.to_csv(RESULTS / "02_walkforward_folds.csv", index=False)
    predictions.reset_index(names="boundary_time").to_csv(RESULTS / "03_oos_predictions.csv", index=False)
    pd.DataFrame(
        {
            "dataset": ["OKX 1m K-lines", "OKX 10s trade bars", "quarter-hour opening 10s"],
            "start": [minute.index.min(), opening.index.min(), samples.index.min()],
            "end": [minute.index.max(), opening.index.max(), samples.index.max()],
            "rows": [len(minute), len(opening), len(samples)],
            "duplicates": [minute.index.duplicated().sum(), opening.index.duplicated().sum(), samples.index.duplicated().sum()],
        }
    ).to_csv(RESULTS / "04_data_quality.csv", index=False)
    yearly: list[dict[str, object]] = []
    for name, replay in replays.items():
        for year, group in replay.groupby(replay.index.year):
            local = group.copy()
            local["equity"] = (1.0 + local["net_return"]).cumprod()
            local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
            row = base.metrics(local, name)
            row["year"] = year
            yearly.append(row)
    pd.DataFrame(yearly).to_csv(RESULTS / "05_yearly.csv", index=False)
    (RESULTS / "run_config.json").write_text(
        json.dumps(
            {
                "source": "OKX ETH-USDT-SWAP only",
                "external_hypothesis_source": "Kim and Hansen (2026), arXiv:2607.09426; no external market data used",
                "observed_window": "first completed 10 seconds of minutes 00/15/30/45",
                "execution": "next 1m open; 2m delay stress",
                "horizon": "8H, non-overlapping positions",
                "model": "fixed standardized Ridge(alpha=10), trailing 365D, monthly refit, label purge",
                "economic_gate": RETURN_GATE,
                "rule_control": "rolling prior-180D absolute-OI 99th percentile, no future rows",
                "one_way_cost": base.ONE_WAY_COST,
                "strategy_gross_cap": 0.75,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(screen.to_string(index=False))
    print("\nFOLDS\n", folds[["return_correlation", "mean_prediction", "prediction_std"]].describe().to_string())
    print("\nROWS", len(opening), len(samples), "MODEL EVENTS", len(non_overlapping_model_events(predictions, 1)) // 2, "RULE EVENTS", len(extreme_oi_events(samples, 1)) // 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

