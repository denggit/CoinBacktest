#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Modeling and ablation diagnostics for R03.4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.market_state_continuity.config import (
    DEFAULT_MARKET_STATE_CONTINUITY_CONFIG,
)
from src.ai_research.market_state_continuity.modeling import (
    collect_continuity_period_data,
    fit_continuity_model,
)
from src.ai_research.market_state_continuity.state_cache import load_state_year_shard
from src.ai_research.swing_baseline.dataset import load_year_shard

from .config import StateContextAblationConfig
from .outcomes import load_outcome_year_shard

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None  # type: ignore[assignment]


ACTIVITY_STATE_COLUMNS = (
    "strategic_activity_score",
    "tactical_activity_score",
    "entry_activity_score",
    "activity_score",
    "activity_raw_state",
    "activity_state",
    "activity_boundary_margin",
    "activity_age_bars",
    "activity_flip_rate_6h",
    "activity_flip_rate_24h",
)

DIRECTIONAL_STATE_COLUMNS = (
    "strategic_score",
    "strategic_raw_state",
    "strategic_state",
    "strategic_boundary_margin",
    "strategic_age_bars",
    "strategic_flip_rate_6h",
    "strategic_flip_rate_24h",
    "tactical_score",
    "tactical_raw_state",
    "tactical_state",
    "tactical_boundary_margin",
    "tactical_age_bars",
    "tactical_flip_rate_6h",
    "tactical_flip_rate_24h",
    "entry_score",
    "entry_raw_state",
    "entry_state",
    "entry_boundary_margin",
    "entry_age_bars",
    "entry_flip_rate_6h",
    "entry_flip_rate_24h",
    "strategic_tactical_alignment",
    "tactical_entry_alignment",
    "all_direction_alignment",
    "long_pullback_setup",
    "short_pullback_setup",
    "trend_momentum_long",
    "trend_momentum_short",
)
ALL_STATE_COLUMNS = tuple(dict.fromkeys((*DIRECTIONAL_STATE_COLUMNS, *ACTIVITY_STATE_COLUMNS)))


@dataclass(frozen=True)
class AblationFold:
    fold_id: str
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, object]:
        return {
            key: str(value) if isinstance(value, pd.Timestamp) else value
            for key, value in asdict(self).items()
        }


def default_ablation_folds(config: StateContextAblationConfig) -> tuple[AblationFold, ...]:
    embargo = pd.Timedelta(hours=max(config.horizons_hours) + 12)
    return (
        AblationFold(
            "WF_2024",
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-10-01") - embargo,
            pd.Timestamp("2023-10-01"),
            pd.Timestamp("2023-12-31 23:59:59"),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-31 23:59:59"),
        ),
        AblationFold(
            "WF_2025",
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2024-10-01") - embargo,
            pd.Timestamp("2024-10-01"),
            pd.Timestamp("2024-12-31 23:59:59"),
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31 23:59:59"),
        ),
    )


@dataclass(frozen=True)
class AblationPeriodData:
    timestamps_ns: np.ndarray
    base_x: np.ndarray
    state_x: np.ndarray
    outcomes: dict[str, np.ndarray]
    base_columns: tuple[str, ...]
    state_columns: tuple[str, ...]
    activity_persist_probability: np.ndarray | None = None

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.timestamps_ns, unit="ns")

    def with_activity_probability(self, values: np.ndarray) -> "AblationPeriodData":
        array = np.asarray(values, dtype=float)
        if len(array) != len(self.timestamps_ns):
            raise ValueError("activity probability must share the decision axis")
        return AblationPeriodData(
            timestamps_ns=self.timestamps_ns,
            base_x=self.base_x,
            state_x=self.state_x,
            outcomes=self.outcomes,
            base_columns=self.base_columns,
            state_columns=self.state_columns,
            activity_persist_probability=array,
        )


def _year_from_base(path: Path) -> int:
    shard = load_year_shard(path)
    return int(pd.to_datetime(np.asarray(shard.decision_times_ns[:1], dtype=np.int64), unit="ns")[0].year)


def collect_ablation_period_data(
    base_paths: list[Path],
    state_paths: list[Path],
    outcome_paths: list[Path],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: StateContextAblationConfig,
) -> AblationPeriodData:
    base_map = {_year_from_base(path): path for path in base_paths}
    state_map = {load_state_year_shard(path).year: path for path in state_paths}
    outcome_map = {load_outcome_year_shard(path).year: path for path in outcome_paths}
    time_parts: list[np.ndarray] = []
    base_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    outcome_parts: dict[str, list[np.ndarray]] = {name: [] for name in config.outcome_columns()}
    expected_base: tuple[str, ...] | None = None
    expected_state: tuple[str, ...] | None = None

    for year in sorted(base_map):
        if year not in state_map or year not in outcome_map:
            continue
        base = load_year_shard(base_map[year])
        state = load_state_year_shard(state_map[year])
        outcome = load_outcome_year_shard(outcome_map[year])
        common, base_pos, state_pos = np.intersect1d(
            np.asarray(base.decision_times_ns, dtype=np.int64),
            np.asarray(state.decision_times_ns, dtype=np.int64),
            assume_unique=True,
            return_indices=True,
        )
        common2, common_pos, outcome_pos = np.intersect1d(
            common,
            np.asarray(outcome.decision_times_ns, dtype=np.int64),
            assume_unique=True,
            return_indices=True,
        )
        if not len(common2):
            continue
        base_pos = base_pos[common_pos]
        state_pos = state_pos[common_pos]
        left = int(np.searchsorted(common2, int(pd.Timestamp(start).value), side="left"))
        right = int(np.searchsorted(common2, int(pd.Timestamp(end).value), side="right"))
        if right <= left:
            continue
        base_positions = base_pos[left:right]
        state_positions = state_pos[left:right]
        outcome_positions = outcome_pos[left:right]
        if expected_base is None:
            expected_base = tuple(base.full_feature_columns)
            expected_state = tuple(state.state_columns)
            missing = sorted(set(ALL_STATE_COLUMNS) - set(expected_state))
            if missing:
                raise RuntimeError(f"R03.4 required state columns missing: {missing}")
        elif tuple(base.full_feature_columns) != expected_base or tuple(state.state_columns) != expected_state:
            raise RuntimeError(f"R03.4 feature schema drift in year {year}")
        time_parts.append(common2[left:right])
        base_parts.append(np.asarray(base.features[base_positions], dtype=np.float32))
        state_parts.append(np.asarray(state.states[state_positions], dtype=np.float32))
        for name in config.outcome_columns():
            outcome_parts[name].append(
                np.asarray(outcome.outcomes[outcome_positions, outcome.outcome_index[name]], dtype=float)
            )

    if not time_parts or expected_base is None or expected_state is None:
        raise RuntimeError(f"R03.4 no aligned data for {start} -> {end}")
    return AblationPeriodData(
        timestamps_ns=np.concatenate(time_parts),
        base_x=np.concatenate(base_parts, axis=0),
        state_x=np.concatenate(state_parts, axis=0),
        outcomes={name: np.concatenate(parts) for name, parts in outcome_parts.items()},
        base_columns=expected_base,
        state_columns=expected_state,
    )


def _state_subset(data: AblationPeriodData, columns: tuple[str, ...]) -> np.ndarray:
    positions = [data.state_columns.index(column) for column in columns]
    return np.asarray(data.state_x[:, positions], dtype=np.float32)


def variant_matrix(variant: str, data: AblationPeriodData) -> tuple[np.ndarray, tuple[str, ...]]:
    if variant == "base_multiframe":
        return data.base_x, data.base_columns
    if variant == "base_plus_activity":
        state = _state_subset(data, ACTIVITY_STATE_COLUMNS)
        return np.concatenate([data.base_x, state], axis=1), tuple(data.base_columns) + tuple(
            f"state::{name}" for name in ACTIVITY_STATE_COLUMNS
        )
    if variant == "base_plus_directional_state":
        state = _state_subset(data, DIRECTIONAL_STATE_COLUMNS)
        return np.concatenate([data.base_x, state], axis=1), tuple(data.base_columns) + tuple(
            f"state::{name}" for name in DIRECTIONAL_STATE_COLUMNS
        )
    if variant in {"base_plus_all_state", "base_plus_all_state_and_activity_persist"}:
        state = _state_subset(data, ALL_STATE_COLUMNS)
        matrix = np.concatenate([data.base_x, state], axis=1)
        columns = tuple(data.base_columns) + tuple(f"state::{name}" for name in ALL_STATE_COLUMNS)
        if variant.endswith("activity_persist"):
            if data.activity_persist_probability is None:
                raise RuntimeError("nested activity persistence probability is unavailable")
            probability = np.asarray(data.activity_persist_probability, dtype=np.float32).reshape(-1, 1)
            matrix = np.concatenate([matrix, probability], axis=1)
            columns = (*columns, "state::activity_persist_h3_probability")
        return matrix, columns
    if variant == "state_only":
        return _state_subset(data, ALL_STATE_COLUMNS), ALL_STATE_COLUMNS
    raise ValueError(f"unsupported R03.4 variant: {variant}")


def _align_probability(timestamps_ns: np.ndarray, source_times: np.ndarray, values: np.ndarray) -> np.ndarray:
    index = pd.Index(np.asarray(source_times, dtype=np.int64))
    positions = index.get_indexer(np.asarray(timestamps_ns, dtype=np.int64))
    output = np.full(len(timestamps_ns), np.nan, dtype=float)
    valid = positions >= 0
    output[valid] = np.asarray(values, dtype=float)[positions[valid]]
    return output


def activity_persistence_feature(
    state_paths: list[Path],
    trade_paths: list[Path],
    *,
    prediction_times_ns: np.ndarray,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    prediction_start: pd.Timestamp,
    prediction_end: pd.Timestamp,
) -> np.ndarray:
    """Generate a causal nested activity-persistence probability feature."""
    target = "activity_persist_h3"
    architecture = "universal_ohlcv_lightgbm"
    train = collect_continuity_period_data(
        state_paths,
        trade_paths,
        start=train_start,
        end=train_end,
        target=target,
        architecture=architecture,
    )
    prediction = collect_continuity_period_data(
        state_paths,
        trade_paths,
        start=prediction_start,
        end=prediction_end,
        target=target,
        architecture=architecture,
    )
    model = fit_continuity_model(train, DEFAULT_MARKET_STATE_CONTINUITY_CONFIG)
    values = np.asarray(model.predict_proba(prediction.x)[:, 1], dtype=float)
    return _align_probability(prediction_times_ns, prediction.timestamps_ns, values)


@dataclass
class OpeningValueModel:
    long_model: object
    short_model: object
    long_clip: tuple[float, float]
    short_clip: tuple[float, float]

    def predict(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        long = np.asarray(self.long_model.predict(matrix), dtype=float)
        short = np.asarray(self.short_model.predict(matrix), dtype=float)
        return long, short


def _sample_positions(valid: np.ndarray, cap: int) -> np.ndarray:
    positions = np.flatnonzero(valid)
    if len(positions) <= cap:
        return positions
    return positions[np.linspace(0, len(positions) - 1, cap, dtype=np.int64)]


def fit_opening_value_model(
    variant: str,
    data: AblationPeriodData,
    config: StateContextAblationConfig,
) -> tuple[OpeningValueModel, tuple[str, ...], dict[str, object]]:
    if LGBMRegressor is None:
        raise RuntimeError("R03.4 requires lightgbm")
    matrix, columns = variant_matrix(variant, data)
    horizon = config.primary_horizon_hours
    long_y = np.asarray(data.outcomes[f"long_utility_h{horizon}"], dtype=float)
    short_y = np.asarray(data.outcomes[f"short_utility_h{horizon}"], dtype=float)
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(long_y) & np.isfinite(short_y)
    positions = _sample_positions(valid, config.train_sample_cap)
    if len(positions) < 1_000:
        raise RuntimeError(f"R03.4 insufficient train rows variant={variant}: {len(positions)}")

    def fit_one(values: np.ndarray):
        low, high = (float(np.quantile(values[positions], q)) for q in (0.0025, 0.9975))
        target = np.clip(values[positions], low, high)
        model = LGBMRegressor(
            objective="regression_l1",
            n_estimators=config.lightgbm_n_estimators,
            learning_rate=config.lightgbm_learning_rate,
            num_leaves=config.lightgbm_num_leaves,
            min_child_samples=config.lightgbm_min_child_samples,
            colsample_bytree=config.lightgbm_feature_fraction,
            subsample=0.85,
            subsample_freq=1,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(matrix[positions], target)
        return model, (low, high)

    long_model, long_clip = fit_one(long_y)
    short_model, short_clip = fit_one(short_y)
    return (
        OpeningValueModel(long_model, short_model, long_clip, short_clip),
        columns,
        {"variant": variant, "train_rows": int(len(positions)), "features": int(len(columns))},
    )


def _rank_ic(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return np.nan
    x = pd.Series(left[valid]).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(right[valid]).rank(method="average").to_numpy(dtype=float)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _profit_factor(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    gains = float(array[array > 0].sum())
    losses = float(-array[array < 0].sum())
    return gains / losses if losses > 0 else np.inf if gains > 0 else np.nan


def evaluate_opening_value_model(
    *,
    fold_id: str,
    variant: str,
    model: OpeningValueModel,
    calibration: AblationPeriodData,
    test: AblationPeriodData,
    config: StateContextAblationConfig,
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame]:
    cal_x, _ = variant_matrix(variant, calibration)
    test_x, _ = variant_matrix(variant, test)
    cal_long, cal_short = model.predict(cal_x)
    pred_long, pred_short = model.predict(test_x)
    horizon = config.primary_horizon_hours
    actual_long = np.asarray(test.outcomes[f"long_utility_h{horizon}"], dtype=float)
    actual_short = np.asarray(test.outcomes[f"short_utility_h{horizon}"], dtype=float)
    close_ret = np.asarray(test.outcomes[f"future_close_return_h{horizon}"], dtype=float)
    long_mfe = np.asarray(test.outcomes[f"long_mfe_h{horizon}"], dtype=float)
    long_mae = np.asarray(test.outcomes[f"long_mae_h{horizon}"], dtype=float)
    valid = (
        np.isfinite(test_x).all(axis=1)
        & np.isfinite(pred_long)
        & np.isfinite(pred_short)
        & np.isfinite(actual_long)
        & np.isfinite(actual_short)
        & np.isfinite(close_ret)
    )
    if int(valid.sum()) < config.minimum_test_rows:
        raise RuntimeError(f"R03.4 insufficient test rows {fold_id}/{variant}: {int(valid.sum())}")
    p_long, p_short = pred_long[valid], pred_short[valid]
    a_long, a_short = actual_long[valid], actual_short[valid]
    realized_close = close_ret[valid]
    times = np.asarray(test.timestamps_ns, dtype=np.int64)[valid]
    mfe_long, mae_long = long_mfe[valid], long_mae[valid]
    pred_direction = p_long - p_short
    actual_direction = a_long - a_short
    meaningful = np.abs(realized_close) > config.base_round_trip_cost
    direction_accuracy = float(np.mean(np.sign(pred_direction[meaningful]) == np.sign(realized_close[meaningful]))) if np.any(meaningful) else np.nan
    metric = {
        "fold_id": fold_id,
        "variant": variant,
        "rows": int(valid.sum()),
        "direction_rank_ic": _rank_ic(pred_direction, realized_close),
        "utility_direction_rank_ic": _rank_ic(pred_direction, actual_direction),
        "long_utility_rank_ic": _rank_ic(p_long, a_long),
        "short_utility_rank_ic": _rank_ic(p_short, a_short),
        "mean_utility_rank_ic": float(np.nanmean([_rank_ic(p_long, a_long), _rank_ic(p_short, a_short)])),
        "meaningful_direction_accuracy": direction_accuracy,
    }

    cal_valid = np.isfinite(cal_x).all(axis=1) & np.isfinite(cal_long) & np.isfinite(cal_short)
    cal_opportunity = np.maximum(cal_long[cal_valid], cal_short[cal_valid])
    opportunity = np.maximum(p_long, p_short)
    choose_long = p_long >= p_short
    selected_utility = np.where(choose_long, a_long, a_short)
    selected_mfe = np.where(choose_long, mfe_long, mae_long)
    selected_mae = np.where(choose_long, mae_long, mfe_long)
    signed_close = np.where(choose_long, realized_close, -realized_close)
    signal_rows: list[dict[str, object]] = []
    sample_parts: list[pd.DataFrame] = []
    for quantile in config.signal_quantiles:
        threshold = float(np.quantile(cal_opportunity, quantile))
        signal = opportunity >= threshold
        for multiplier in config.cost_stress_multipliers:
            net = signed_close[signal] - config.base_round_trip_cost * float(multiplier)
            signal_rows.append(
                {
                    "fold_id": fold_id,
                    "variant": variant,
                    "quantile": quantile,
                    "cost_multiplier": multiplier,
                    "threshold": threshold,
                    "signals": int(signal.sum()),
                    "signal_rate": float(np.mean(signal)),
                    "long_share": float(np.mean(choose_long[signal])) if np.any(signal) else np.nan,
                    "mean_selected_utility": float(np.mean(selected_utility[signal])) if np.any(signal) else np.nan,
                    "mean_mfe": float(np.mean(selected_mfe[signal])) if np.any(signal) else np.nan,
                    "mean_mae": float(np.mean(selected_mae[signal])) if np.any(signal) else np.nan,
                    "mfe_mae_ratio": float(np.mean(selected_mfe[signal]) / max(np.mean(selected_mae[signal]), 1e-12)) if np.any(signal) else np.nan,
                    "mean_net_close_return": float(np.mean(net)) if len(net) else np.nan,
                    "median_net_close_return": float(np.median(net)) if len(net) else np.nan,
                    "win_rate": float(np.mean(net > 0)) if len(net) else np.nan,
                    "profit_factor": _profit_factor(net),
                }
            )
        positions = np.flatnonzero(signal)
        if len(positions):
            sample_parts.append(
                pd.DataFrame(
                    {
                        "fold_id": fold_id,
                        "variant": variant,
                        "quantile": quantile,
                        "decision_time": pd.to_datetime(times[positions], unit="ns"),
                        "direction": np.where(choose_long[positions], "LONG", "SHORT"),
                        "predicted_long_utility": p_long[positions],
                        "predicted_short_utility": p_short[positions],
                        "actual_selected_utility": selected_utility[positions],
                        "mfe": selected_mfe[positions],
                        "mae": selected_mae[positions],
                        "gross_close_return": signed_close[positions],
                    }
                )
            )
    samples = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    return metric, signal_rows, samples


def build_uplift_table(metrics: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    base_metrics = metrics.loc[metrics["variant"] == "base_multiframe"].copy()
    merged = metrics.merge(base_metrics, on="fold_id", suffixes=("", "_base"))
    merged = merged.loc[merged["variant"] != "base_multiframe"].copy()
    for column in (
        "direction_rank_ic",
        "utility_direction_rank_ic",
        "mean_utility_rank_ic",
        "meaningful_direction_accuracy",
    ):
        merged[f"delta_{column}"] = merged[column] - merged[f"{column}_base"]
    if not signals.empty:
        primary = signals.loc[(signals["quantile"] == 0.90) & (signals["cost_multiplier"] == 1.0)].copy()
        base_signal = primary.loc[primary["variant"] == "base_multiframe"].copy()
        signal_merge = primary.merge(base_signal, on="fold_id", suffixes=("", "_base"))
        signal_merge = signal_merge.loc[signal_merge["variant"] != "base_multiframe"]
        keep = [
            "fold_id",
            "variant",
            "signals",
            "mean_net_close_return",
            "profit_factor",
            "mean_mae",
            "mean_mfe",
        ]
        signal_merge = signal_merge[keep + [f"{name}_base" for name in keep[2:]]]
        signal_merge["delta_mean_net_close_return"] = signal_merge["mean_net_close_return"] - signal_merge["mean_net_close_return_base"]
        signal_merge["delta_profit_factor"] = signal_merge["profit_factor"] - signal_merge["profit_factor_base"]
        signal_merge["delta_mean_mae"] = signal_merge["mean_mae"] - signal_merge["mean_mae_base"]
        signal_merge["delta_mean_mfe"] = signal_merge["mean_mfe"] - signal_merge["mean_mfe_base"]
        merged = merged.merge(signal_merge, on=["fold_id", "variant"], how="left")
    return merged.reset_index(drop=True)


def select_stable_uplift(uplift: pd.DataFrame, config: StateContextAblationConfig) -> pd.DataFrame:
    if uplift.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for variant, group in uplift.groupby("variant", sort=False):
        row: dict[str, object] = {"variant": variant}
        passed = True
        for fold in ("WF_2024", "WF_2025"):
            subset = group.loc[group["fold_id"] == fold]
            if subset.empty:
                passed = False
                continue
            item = subset.iloc[0]
            for column in (
                "delta_direction_rank_ic",
                "delta_mean_utility_rank_ic",
                "delta_mean_net_close_return",
                "delta_profit_factor",
                "delta_mean_mae",
            ):
                row[f"{fold}_{column}"] = item.get(column, np.nan)
            fold_pass = (
                float(item["delta_direction_rank_ic"]) >= config.minimum_rank_ic_increment
                and float(item.get("delta_mean_net_close_return", np.nan)) >= config.minimum_net_expectancy_increment
                and int(item.get("signals", 0)) >= config.minimum_signal_count
            )
            row[f"{fold}_passes"] = bool(fold_pass)
            passed = passed and fold_pass
        row["passes"] = passed
        rank_deltas = [float(row.get(f"{fold}_delta_direction_rank_ic", np.nan)) for fold in ("WF_2024", "WF_2025")]
        net_deltas = [float(row.get(f"{fold}_delta_mean_net_close_return", np.nan)) for fold in ("WF_2024", "WF_2025")]
        row["stability_score"] = float(np.nanmin(rank_deltas) + 20.0 * np.nanmin(net_deltas))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["passes", "stability_score"], ascending=[False, False], kind="stable")
