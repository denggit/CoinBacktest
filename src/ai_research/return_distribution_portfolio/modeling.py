from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_pinball_loss

from .config import ReturnDistributionConfig
from .dataset import feature_columns


@dataclass
class QuantileBundle:
    horizon: int
    features: list[str]
    medians: pd.Series
    models: dict[float, object]

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        clean = frame.loc[:, self.features].replace([np.inf, -np.inf], np.nan).fillna(self.medians)
        x = clean.to_numpy(dtype=np.float32, copy=False)
        out = pd.DataFrame(index=frame.index)
        for q, model in sorted(self.models.items()):
            out[f"q{int(round(q * 100)):02d}"] = np.asarray(model.predict(x), dtype=float)
        return out


def _lightgbm_regressor(config: ReturnDistributionConfig, quantile: float):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "lightgbm is required for RDP V1 because the existing CoinBacktest AI research stack already uses it"
        ) from exc
    return LGBMRegressor(
        objective="quantile",
        alpha=float(quantile),
        n_estimators=config.lightgbm_n_estimators,
        learning_rate=config.lightgbm_learning_rate,
        num_leaves=config.lightgbm_num_leaves,
        min_child_samples=config.lightgbm_min_child_samples,
        colsample_bytree=config.feature_fraction,
        subsample=0.90,
        subsample_freq=1,
        reg_lambda=1.0,
        random_state=20260817,
        n_jobs=-1,
        verbosity=-1,
    )


def prepare_training_frame(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, list[str], pd.Series]:
    target = f"ret_h{horizon}"
    candidates = feature_columns(frame)
    source = frame.loc[:, candidates + [target]].replace([np.inf, -np.inf], np.nan)
    coverage = source[candidates].notna().mean()
    features = [c for c in candidates if coverage[c] >= 0.95 and source[c].nunique(dropna=True) > 1]
    if not features:
        raise RuntimeError(f"no usable causal features for horizon {horizon}")
    medians = source[features].median(axis=0, skipna=True)
    features = [c for c in features if pd.notna(medians[c])]
    medians = medians.loc[features]
    clean = source.loc[source[target].notna(), features + [target]].copy()
    clean.loc[:, features] = clean[features].fillna(medians)
    return clean, features, medians


def fit_quantile_bundle(frame: pd.DataFrame, horizon: int, config: ReturnDistributionConfig) -> QuantileBundle:
    clean, features, medians = prepare_training_frame(frame, horizon)
    if clean.empty:
        raise RuntimeError(f"no training rows for horizon {horizon}")
    sampled = clean.iloc[:: config.train_stride].copy()
    if len(sampled) > config.train_sample_cap:
        # Deterministic evenly spaced cap; no random post-hoc sampling.
        pos = np.linspace(0, len(sampled) - 1, config.train_sample_cap, dtype=int)
        sampled = sampled.iloc[pos]
    x = sampled.loc[:, features].to_numpy(dtype=np.float32, copy=False)
    y = sampled[f"ret_h{horizon}"].to_numpy(dtype=float)
    models: dict[float, object] = {}
    for q in config.quantiles:
        model = _lightgbm_regressor(config, q)
        model.fit(x, y)
        models[q] = model
    return QuantileBundle(horizon=horizon, features=features, medians=medians, models=models)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    valid = a.notna() & b.notna()
    if int(valid.sum()) < 3:
        return float("nan")
    return float(a.loc[valid].rank(method="average").corr(b.loc[valid].rank(method="average")))


def evaluate_bundle(
    bundle: QuantileBundle,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    fold_id: str,
    config: ReturnDistributionConfig,
) -> tuple[dict[str, float | int | str], pd.DataFrame, pd.DataFrame]:
    target = f"ret_h{bundle.horizon}"
    valid_cols = bundle.features + [target]
    test = test_frame.loc[:, valid_cols].replace([np.inf, -np.inf], np.nan)
    test = test.loc[test[target].notna()].copy()
    if test.empty:
        raise RuntimeError(f"no test rows for {fold_id} horizon={bundle.horizon}")
    pred = bundle.predict(test)
    actual = test[target].astype(float)
    quantile_names = [f"q{int(round(q * 100)):02d}" for q in config.quantiles]

    # LightGBM quantile models are independent; report crossing instead of
    # silently sorting predictions and hiding calibration defects.
    pmat = pred.loc[:, quantile_names].to_numpy(dtype=float)
    crossing = np.any(np.diff(pmat, axis=1) < 0.0, axis=1)

    q50 = pred["q50"]
    metrics: dict[str, float | int | str] = {
        "fold_id": fold_id,
        "horizon_minutes": bundle.horizon,
        "rows": int(len(test)),
        "rank_ic_q50": _spearman(q50, actual),
        "sign_accuracy_q50": float((np.sign(q50) == np.sign(actual)).mean()),
        "mean_actual_return": float(actual.mean()),
        "mean_abs_actual_return": float(actual.abs().mean()),
        "quantile_crossing_rate": float(crossing.mean()),
    }
    q10 = pred["q10"]
    q90 = pred["q90"]
    metrics["q10_q90_coverage"] = float(((actual >= q10) & (actual <= q90)).mean())
    metrics["q50_mae"] = float((actual - q50).abs().mean())
    baseline_median = float(train_frame[target].replace([np.inf, -np.inf], np.nan).dropna().median())
    metrics["baseline_median_mae"] = float((actual - baseline_median).abs().mean())
    for q in config.quantiles:
        name = f"q{int(round(q * 100)):02d}"
        loss = mean_pinball_loss(actual.to_numpy(), pred[name].to_numpy(), alpha=q)
        baseline_q = float(train_frame[target].replace([np.inf, -np.inf], np.nan).dropna().quantile(q))
        baseline_loss = mean_pinball_loss(actual.to_numpy(), np.full(len(actual), baseline_q), alpha=q)
        metrics[f"pinball_{name}"] = float(loss)
        metrics[f"pinball_skill_{name}"] = float(1.0 - loss / max(baseline_loss, 1e-12))

    decile = pd.qcut(q50.rank(method="first"), 10, labels=False, duplicates="drop")
    decile_rows = []
    for d in sorted(pd.Series(decile).dropna().unique()):
        mask = decile == d
        decile_rows.append(
            {
                "fold_id": fold_id,
                "horizon_minutes": bundle.horizon,
                "decile": int(d) + 1,
                "rows": int(mask.sum()),
                "predicted_q50_mean": float(q50.loc[mask].mean()),
                "actual_return_mean": float(actual.loc[mask].mean()),
                "actual_return_median": float(actual.loc[mask].median()),
            }
        )
    deciles = pd.DataFrame(decile_rows)
    if len(deciles) >= 2:
        metrics["top_bottom_decile_spread"] = float(deciles.iloc[-1]["actual_return_mean"] - deciles.iloc[0]["actual_return_mean"])
        metrics["decile_monotonicity"] = _spearman(deciles["decile"], deciles["actual_return_mean"])
    else:
        metrics["top_bottom_decile_spread"] = float("nan")
        metrics["decile_monotonicity"] = float("nan")

    sample = pd.DataFrame(index=test.index)
    sample["fold_id"] = fold_id
    sample["horizon_minutes"] = bundle.horizon
    sample["actual_return"] = actual
    for name in quantile_names:
        sample[name] = pred[name]
    sample["quantile_crossing"] = crossing
    return metrics, deciles, sample
