#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Concrete trade-opportunity targets and fixed-model wrappers for R01."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradeTemplate:
    name: str
    horizon_minutes: int
    take_profit: float
    stop_loss: float

    def validate(self) -> None:
        if self.horizon_minutes <= 0:
            raise ValueError("horizon_minutes must be positive")
        if self.take_profit <= 0 or self.stop_loss <= 0:
            raise ValueError("take_profit and stop_loss must be positive")


DEFAULT_TRADE_TEMPLATES: tuple[TradeTemplate, ...] = (
    TradeTemplate("H60_TP060_SL040", 60, 0.0060, 0.0040),
    TradeTemplate("H180_TP100_SL060", 180, 0.0100, 0.0060),
    TradeTemplate("H360_TP150_SL080", 360, 0.0150, 0.0080),
)


def required_label_names(template: TradeTemplate) -> tuple[str, ...]:
    h = int(template.horizon_minutes)
    return (
        f"h{h}__final_return",
        f"h{h}__long_mfe",
        f"h{h}__long_mae",
        f"h{h}__short_mfe",
        f"h{h}__short_mae",
    )


def conservative_template_returns(
    labels: np.ndarray,
    label_names: Iterable[str],
    template: TradeTemplate,
    *,
    round_trip_cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative long/short template net returns per unit notional.

    R00 contains MFE/MAE but not first-hit order.  When TP and SL are both
    touched inside the horizon, R01 deliberately labels the sample as a stop.
    This prevents optimistic path assumptions in model training.  Final
    calibration/OOS strategy replay uses the actual 1m path and therefore
    resolves first-hit order exactly, with same-minute TP+SL still adverse-first.
    """

    template.validate()
    names = tuple(label_names)
    lookup = {name: i for i, name in enumerate(names)}
    needed = required_label_names(template)
    missing = [name for name in needed if name not in lookup]
    if missing:
        raise KeyError(f"required labels missing: {missing}")
    arr = np.asarray(labels, dtype=np.float64)
    final = arr[:, lookup[needed[0]]]
    long_mfe = arr[:, lookup[needed[1]]]
    long_mae = arr[:, lookup[needed[2]]]
    short_mfe = arr[:, lookup[needed[3]]]
    short_mae = arr[:, lookup[needed[4]]]

    tp = float(template.take_profit)
    sl = float(template.stop_loss)
    cost = float(round_trip_cost)

    long_tp = long_mfe >= tp
    long_sl = long_mae <= -sl
    short_tp = short_mfe >= tp
    short_sl = short_mae <= -sl

    long_gross = final.copy()
    long_gross[long_tp & ~long_sl] = tp
    long_gross[long_sl] = -sl  # includes both-hit => conservative stop

    short_gross = -final
    short_gross[short_tp & ~short_sl] = tp
    short_gross[short_sl] = -sl

    long_net = long_gross - cost
    short_net = short_gross - cost
    invalid = ~np.isfinite(final)
    long_net[invalid] = np.nan
    short_net[invalid] = np.nan
    return long_net.astype(np.float32), short_net.astype(np.float32)


def feature_groups(feature_names: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Frozen economically meaningful ablations; availability flags are audit-only."""

    names = tuple(feature_names)
    usable = [name for name in names if not name.startswith("availability__")]
    kline = tuple(name for name in usable if name.startswith("kline_"))
    trade = tuple(name for name in usable if name.startswith("trade_"))
    full = tuple(usable)
    return {
        "KLINE_ONLY": kline,
        "KLINE_TRADE": tuple([*kline, *trade]),
        "FULL": full,
    }


class FixedOpportunityRegressor:
    """LightGBM baseline with a deterministic sklearn fallback."""

    def __init__(self, *, random_state: int = 20260817) -> None:
        self.random_state = int(random_state)
        self.backend = ""
        self.model = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "FixedOpportunityRegressor":
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        valid = np.isfinite(y)
        if valid.sum() < 100:
            raise ValueError("fewer than 100 finite training targets")
        x_fit = x[valid]
        y_fit = y[valid]
        try:
            from lightgbm import LGBMRegressor

            self.backend = "lightgbm"
            self.model = LGBMRegressor(
                objective="regression_l2",
                n_estimators=240,
                learning_rate=0.035,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=120,
                subsample=0.85,
                subsample_freq=1,
                colsample_bytree=0.80,
                reg_alpha=0.05,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=-1,
            )
        except ImportError:
            from sklearn.ensemble import HistGradientBoostingRegressor

            self.backend = "sklearn_hist_gradient_boosting"
            self.model = HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_iter=220,
                max_leaf_nodes=31,
                min_samples_leaf=120,
                l2_regularization=1.0,
                random_state=self.random_state,
            )
        self.model.fit(x_fit, y_fit)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        values = np.asarray(x, dtype=np.float32)
        booster = getattr(self.model, "booster_", None)
        if booster is not None:
            return np.asarray(booster.predict(values), dtype=np.float32)
        return np.asarray(self.model.predict(values), dtype=np.float32)

    def feature_importance(self, feature_names: Iterable[str]) -> pd.DataFrame:
        names = tuple(feature_names)
        if self.model is None:
            return pd.DataFrame(columns=["feature", "importance"])
        raw = getattr(self.model, "feature_importances_", None)
        if raw is None:
            return pd.DataFrame(columns=["feature", "importance"])
        frame = pd.DataFrame({"feature": names, "importance": np.asarray(raw, dtype=float)})
        return frame.sort_values("importance", ascending=False, kind="stable").reset_index(drop=True)

    def save(self, path: str | Path) -> str:
        if self.model is None:
            raise RuntimeError("model is not fitted")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        booster = getattr(self.model, "booster_", None)
        if booster is not None:
            booster.save_model(str(p.with_suffix(".txt")))
            return str(p.with_suffix(".txt"))
        # Avoid opaque pickle artifacts for the fallback; report metadata only.
        return ""


def prediction_quality(pred: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    valid = np.isfinite(pred) & np.isfinite(actual)
    if valid.sum() < 10:
        return {"spearman_ic": np.nan, "top_decile_mean": np.nan, "all_mean": np.nan, "top_decile_lift": np.nan}
    p = pd.Series(pred[valid])
    a = pd.Series(actual[valid])
    ic = float(p.rank(pct=True).corr(a.rank(pct=True)))
    cutoff = float(np.nanquantile(p.to_numpy(), 0.90))
    top = a[p >= cutoff]
    all_mean = float(a.mean())
    top_mean = float(top.mean()) if len(top) else np.nan
    return {
        "spearman_ic": ic,
        "top_decile_mean": top_mean,
        "all_mean": all_mean,
        "top_decile_lift": top_mean - all_mean if np.isfinite(top_mean) else np.nan,
    }
