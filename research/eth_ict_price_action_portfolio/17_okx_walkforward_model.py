#!/usr/bin/env python
"""OKX-only causal walk-forward ETH perpetual model with 1m execution."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.eth_ict_price_action_portfolio.ict_pa_model import IctPaConfig, build_daily_structure_core, resample_ohlcv
from src.data_feed.okx_loader import OKXDataLoader
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader


RESULTS = Path(__file__).resolve().parent / "ict_pa_v4" / "results"
START = pd.Timestamp("2022-01-01")
END = pd.Timestamp("2026-08-15 23:59:00")
ONE_WAY_COST = 0.0005
FEATURE_COLUMNS = (
    "return_1h", "return_3h", "return_6h", "return_12h", "return_24h", "return_72h",
    "close_location", "body_fraction", "range_over_atr", "distance_high_24h", "distance_low_24h",
    "realized_vol_24h", "volume_relative_24h", "trades_relative_24h",
    "delta_ratio_1h", "delta_ratio_4h", "delta_ratio_24h",
    "large_delta_ratio_1h", "large_delta_ratio_4h", "taker_buy_ratio_4h",
    "sell_absorption", "buy_absorption", "price_flow_efficiency_4h", "max_trade_share",
)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    minute = OKXDataLoader(symbol="ETH-USDT-SWAP", timeframe="1m").load_local_data()
    minute = minute.loc["2020-01-01":END, ["open", "high", "low", "close", "volume"]].copy()
    trade = OKXTradeBarLoader(symbol="ETH-USDT-SWAP", timeframe="15m").load_local_data("2022-01-01", END)
    return minute, trade


def safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def build_hourly_features(trade: pd.DataFrame) -> pd.DataFrame:
    hour = trade.resample("1h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), notional=("notional", "sum"), trades=("trades_count", "sum"),
        buy_notional=("buy_notional", "sum"), sell_notional=("sell_notional", "sum"),
        delta_notional=("delta_notional", "sum"), large_delta_notional=("large_delta_notional", "sum"),
        max_trade_notional=("max_trade_notional", "max"), source_bars=("close", "size"),
    )
    # A partial hour is not equivalent to a completed 1H feature bar.  Drop it
    # instead of silently treating one to three 15m bars as a full observation.
    hour = hour.loc[hour["source_bars"] == 4].dropna(subset=["open", "high", "low", "close"])
    close = hour["close"]
    candle_range = (hour["high"] - hour["low"]).replace(0.0, np.nan)
    true_range = pd.concat(
        [hour["high"] - hour["low"], (hour["high"] - close.shift(1)).abs(), (hour["low"] - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    atr24 = true_range.shift(1).rolling(24, min_periods=24).median()
    data: dict[str, pd.Series] = {}
    for lag in (1, 3, 6, 12, 24, 72):
        data[f"return_{lag}h"] = np.log(close).diff(lag)
    data["close_location"] = safe_divide(close - hour["low"], candle_range)
    data["body_fraction"] = safe_divide((close - hour["open"]).abs(), candle_range)
    data["range_over_atr"] = safe_divide(true_range, atr24)
    high24 = hour["high"].shift(1).rolling(24, min_periods=24).max()
    low24 = hour["low"].shift(1).rolling(24, min_periods=24).min()
    data["distance_high_24h"] = close / high24 - 1.0
    data["distance_low_24h"] = close / low24 - 1.0
    data["sweep_high_24h"] = ((hour["high"] > high24) & (close < high24)).astype(float)
    data["sweep_low_24h"] = ((hour["low"] < low24) & (close > low24)).astype(float)
    data["realized_vol_24h"] = np.log(close).diff().shift(1).rolling(24, min_periods=24).std(ddof=0) * np.sqrt(24 * 365.25)
    data["volume_relative_24h"] = safe_divide(hour["volume"], hour["volume"].shift(1).rolling(24, min_periods=24).median())
    data["trades_relative_24h"] = safe_divide(hour["trades"], hour["trades"].shift(1).rolling(24, min_periods=24).median())
    data["delta_ratio_1h"] = safe_divide(hour["delta_notional"], hour["notional"])
    data["delta_ratio_4h"] = safe_divide(hour["delta_notional"].rolling(4).sum(), hour["notional"].rolling(4).sum())
    data["delta_ratio_24h"] = safe_divide(hour["delta_notional"].rolling(24).sum(), hour["notional"].rolling(24).sum())
    data["large_delta_ratio_1h"] = safe_divide(hour["large_delta_notional"], hour["notional"])
    data["large_delta_ratio_4h"] = safe_divide(hour["large_delta_notional"].rolling(4).sum(), hour["notional"].rolling(4).sum())
    data["taker_buy_ratio_4h"] = safe_divide(hour["buy_notional"].rolling(4).sum(), hour["notional"].rolling(4).sum())
    negative_flow = (-data["delta_ratio_1h"]).clip(lower=0.0)
    positive_flow = data["delta_ratio_1h"].clip(lower=0.0)
    data["sell_absorption"] = negative_flow * data["close_location"]
    data["buy_absorption"] = positive_flow * (1.0 - data["close_location"])
    data["price_flow_efficiency_4h"] = data["return_3h"] / (data["delta_ratio_4h"].abs() + 1e-4)
    data["max_trade_share"] = safe_divide(hour["max_trade_notional"], hour["notional"])
    features = pd.DataFrame(data, index=hour.index).replace([np.inf, -np.inf], np.nan)
    # Positional shift: the [T,T+1H) bar becomes available at T+1H.
    available = pd.DataFrame(features.to_numpy(), columns=features.columns, index=features.index + pd.Timedelta(hours=1))
    return available


def build_samples(features: pd.DataFrame, minute: pd.DataFrame) -> pd.DataFrame:
    # Six non-overlapping decisions per day.  One minute is reserved between
    # completed feature bar and execution.
    sample = features[(features.index.minute == 0) & (features.index.hour % 4 == 0)].copy()
    sample["available_time"] = sample.index
    sample["execution_time"] = sample.index + pd.Timedelta(minutes=1)
    sample["exit_time"] = sample["execution_time"] + pd.Timedelta(hours=4)
    open_price = minute["open"]
    sample["entry_price"] = open_price.reindex(pd.DatetimeIndex(sample["execution_time"])).to_numpy()
    sample["exit_price"] = open_price.reindex(pd.DatetimeIndex(sample["exit_time"])).to_numpy()
    sample["future_return_4h"] = sample["exit_price"] / sample["entry_price"] - 1.0
    sample["label_up"] = (sample["future_return_4h"] > 0.0).astype(int)
    return sample.dropna(subset=[*FEATURE_COLUMNS, "future_return_4h"])


def walk_forward_predictions(samples: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts = []
    fold_rows = []
    coefficient_rows = []
    first_test = pd.Timestamp("2022-07-01")
    last_test = min(pd.Timestamp("2026-07-01"), samples.index.max().to_period("M").start_time)
    for test_start in pd.date_range(first_test, last_test, freq="MS"):
        test_end = test_start + pd.offsets.MonthBegin(1)
        train_start = test_start - pd.Timedelta(days=365)
        # Purge all labels whose 4H outcome touches the test month.
        train = samples[(samples.index >= train_start) & (samples["exit_time"] < test_start)]
        test = samples[(samples.index >= test_start) & (samples.index < test_end)]
        if len(train) < 600 or len(test) == 0 or train["label_up"].nunique() < 2:
            continue
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("logit", LogisticRegression(C=0.10, penalty="l2", solver="lbfgs", max_iter=1000, random_state=17)),
            ]
        )
        model.fit(train[list(FEATURE_COLUMNS)], train["label_up"])
        probability = model.predict_proba(test[list(FEATURE_COLUMNS)])[:, 1]
        part = test[["available_time", "execution_time", "exit_time", "future_return_4h", "label_up"]].copy()
        part["probability_up"] = probability
        part["fold"] = str(test_start.date())
        prediction_parts.append(part)
        auc = roc_auc_score(test["label_up"], probability) if test["label_up"].nunique() > 1 else np.nan
        fold_rows.append(
            {
                "test_month": str(test_start.date()), "train_start": str(train.index.min()), "train_end_label_exit": str(train["exit_time"].max()),
                "train_rows": len(train), "test_rows": len(test), "auc": auc,
                "accuracy_050": accuracy_score(test["label_up"], probability >= 0.5),
                "brier": brier_score_loss(test["label_up"], probability), "label_up_rate": float(test["label_up"].mean()),
            }
        )
        coefficients = model.named_steps["logit"].coef_[0]
        for feature, value in zip(FEATURE_COLUMNS, coefficients):
            coefficient_rows.append({"test_month": str(test_start.date()), "feature": feature, "standardized_coefficient": float(value)})
    if not prediction_parts:
        raise RuntimeError("no valid walk-forward folds")
    return pd.concat(prediction_parts), pd.DataFrame(fold_rows), pd.DataFrame(coefficient_rows)


def tactical_state(predictions: pd.DataFrame, minute_index: pd.DatetimeIndex, delay_minutes: int = 1) -> pd.Series:
    state = 0.0
    values = []
    for probability in predictions["probability_up"]:
        if probability >= 0.56:
            state = 0.20
        elif probability <= 0.44:
            state = -0.20
        elif state > 0 and probability < 0.50:
            state = 0.0
        elif state < 0 and probability > 0.50:
            state = 0.0
        values.append(state)
    execution_index = pd.DatetimeIndex(predictions["available_time"]) + pd.Timedelta(minutes=delay_minutes)
    events = pd.Series(values, index=execution_index)
    aligned = events.reindex(minute_index, method="ffill").fillna(0.0)
    if len(events):
        aligned.loc[aligned.index >= events.index.max() + pd.Timedelta(hours=4)] = 0.0
    return aligned


def core_state(minute: pd.DataFrame) -> pd.Series:
    bars15 = resample_ohlcv(minute, "15min")
    cfg = IctPaConfig(core_mode="daily_12m_blend", core_target_volatility=0.08, core_notional_cap=0.40)
    core = build_daily_structure_core(bars15, cfg)["core_desired_close"]
    aligned = core.reindex(minute.index, method="ffill").fillna(0.0)
    # Daily boundary is the moment the close is known; execute one minute later.
    return aligned.shift(1).fillna(0.0)


def simulate_minute(minute: pd.DataFrame, positions: pd.DataFrame, cost: float = ONE_WAY_COST) -> pd.DataFrame:
    pos = positions.loc[START:END].copy()
    price = minute["open"].reindex(pos.index)
    price_return = minute["open"].shift(-1).reindex(pos.index) / price - 1.0
    valid = price_return.notna()
    pos, price_return = pos.loc[valid], price_return.loc[valid]
    requested_gross = pos.abs().sum(axis=1)
    scale = (0.75 / requested_gross.replace(0.0, np.nan)).clip(upper=1.0).fillna(1.0)
    pos = pos.mul(scale, axis=0)
    gross = pos.abs().sum(axis=1)
    turnover = pos.diff().abs().sum(axis=1)
    turnover.iloc[0] = pos.iloc[0].abs().sum()
    net_exposure = pos.sum(axis=1)
    trading_cost = turnover * cost
    net_return = net_exposure * price_return - trading_cost
    equity = (1.0 + net_return).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return pd.concat(
        [pos.add_prefix("position_"), price_return.rename("price_return"), turnover.rename("turnover"), trading_cost.rename("trading_cost"), gross.rename("gross_exposure"), net_exposure.rename("net_exposure"), net_return.rename("net_return"), equity.rename("equity"), drawdown.rename("drawdown")], axis=1
    )


def streak(values: pd.Series) -> int:
    best = current = 0
    for value in values.fillna(False).astype(bool):
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def metrics(frame: pd.DataFrame, name: str) -> dict[str, object]:
    daily = (1.0 + frame["net_return"]).groupby(frame.index.floor("D")).prod() - 1.0
    daily_gross = frame["gross_exposure"].groupby(frame.index.floor("D")).max()
    total = float(frame["equity"].iloc[-1] - 1.0)
    years = (frame.index[-1] - frame.index[0]).total_seconds() / (365.25 * 86400)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    dd = abs(float(frame["drawdown"].min()))
    long_gross = frame.filter(regex=r"^position_").clip(lower=0).sum(axis=1)
    short_gross = -frame.filter(regex=r"^position_").clip(upper=0).sum(axis=1)
    return {
        "candidate": name, "total_return": total, "cagr": cagr, "max_drawdown": dd, "calmar": cagr / dd if dd else np.nan,
        "max_consecutive_flat_days": streak(daily_gross <= 1e-12), "max_consecutive_losing_days": streak(daily < 0),
        "positive_month_rate": float(((1.0 + frame["net_return"]).groupby(frame.index.to_period("M")).prod() - 1.0 > 0).mean()),
        "annual_volatility": float(frame["net_return"].std(ddof=0) * np.sqrt(365.25 * 1440)),
        "max_gross_exposure": float(frame["gross_exposure"].max()), "hedged_bar_rate": float(((long_gross > 0) & (short_gross > 0)).mean()),
        "total_cost": float(frame["trading_cost"].sum()),
    }


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    minute, trade = load_inputs()
    features = build_hourly_features(trade)
    samples = build_samples(features, minute)
    predictions, folds, coefficients = walk_forward_predictions(samples)
    core = core_state(minute) * 0.75
    tactical_1m = tactical_state(predictions, minute.index, delay_minutes=1)
    tactical_2m = tactical_state(predictions, minute.index, delay_minutes=2)
    position_variants = {
        "daily_pa_core_only": pd.DataFrame({"core": core}, index=minute.index),
        "walkforward_tactical_only": pd.DataFrame({"tactical": tactical_1m}, index=minute.index),
        "okx_pa_flow_walkforward_1m": pd.DataFrame({"core": core, "tactical": tactical_1m}, index=minute.index),
        "okx_pa_flow_walkforward_2m": pd.DataFrame({"core": core, "tactical": tactical_2m}, index=minute.index),
    }
    rows = []
    selected = None
    for name, position in position_variants.items():
        replay = simulate_minute(minute, position)
        rows.append(metrics(replay, name))
        if name == "okx_pa_flow_walkforward_1m":
            selected = replay
    assert selected is not None
    pd.DataFrame(rows).to_csv(RESULTS / "01_model_screen.csv", index=False)
    folds.to_csv(RESULTS / "02_walkforward_folds.csv", index=False)
    coefficients.to_csv(RESULTS / "03_fold_coefficients.csv", index=False)
    predictions.reset_index(drop=True).to_csv(RESULTS / "04_oos_predictions.csv", index=False)
    period_rows = []
    for year, group in selected.groupby(selected.index.year):
        local = group.copy()
        local["equity"] = (1.0 + local["net_return"]).cumprod()
        local["drawdown"] = local["equity"] / local["equity"].cummax() - 1.0
        period_rows.append(metrics(local, str(year)))
    pd.DataFrame(period_rows).to_csv(RESULTS / "05_yearly.csv", index=False)
    daily = selected.groupby(selected.index.floor("D")).agg(equity=("equity", "last"), drawdown=("drawdown", "last"), max_gross_exposure=("gross_exposure", "max"), end_net_exposure=("net_exposure", "last"), trading_cost=("trading_cost", "sum"))
    daily["net_return"] = (1.0 + selected["net_return"]).groupby(selected.index.floor("D")).prod() - 1.0
    daily.to_csv(RESULTS / "06_daily_equity.csv")
    config = {
        "source": "OKX ETH-USDT-SWAP only",
        "feature_frequency": "1H completed bars from OKX 15m trade bars",
        "decision_frequency": "4H non-overlapping",
        "execution": "next 1m open; 2m is delay stress",
        "model": "fixed StandardScaler + L2 LogisticRegression(C=0.10)",
        "training": "trailing 365 days, retrained monthly, 4H label purge before each test month",
        "features": list(FEATURE_COLUMNS), "entry_thresholds": {"long": 0.56, "short": 0.44},
        "one_way_cost": ONE_WAY_COST, "exchange_leverage_cap": 15.0, "strategy_gross_cap": 0.75,
    }
    (RESULTS / "run_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nFOLDS\n", folds[["auc", "accuracy_050", "brier"]].describe().to_string())
    print("\nYEARLY\n", pd.DataFrame(period_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
