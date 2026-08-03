#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Two-stage long-only state meta calibration for R03.4.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.ai_research.state_context_ablation.modeling import AblationPeriodData

from .config import LongStateCalibrationConfig, STATE_META_VARIANTS

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None  # type: ignore[assignment]


STRATEGIC_SOFT_COLUMNS = (
    "strategic_score",
    "strategic_boundary_margin",
    "strategic_age_bars",
    "strategic_flip_rate_24h",
)
ACTIVITY_SOFT_COLUMNS = (
    "strategic_activity_score",
    "tactical_activity_score",
    "entry_activity_score",
    "activity_score",
    "activity_boundary_margin",
    "activity_age_bars",
    "activity_flip_rate_6h",
    "activity_flip_rate_24h",
)
ALL_SOFT_COLUMNS = (*STRATEGIC_SOFT_COLUMNS, *ACTIVITY_SOFT_COLUMNS)


@dataclass(frozen=True)
class OOFBlock:
    block_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    prediction_start: pd.Timestamp
    prediction_end: pd.Timestamp

    def to_dict(self) -> dict[str, object]:
        return {
            "block_id": self.block_id,
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "prediction_start": str(self.prediction_start),
            "prediction_end": str(self.prediction_end),
        }


@dataclass
class LongBaseModel:
    model: object
    clip: tuple[float, float]

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(matrix), dtype=float)


@dataclass
class MetaModel:
    variant: str
    model: object | None
    columns: tuple[str, ...]

    def predict(self, base_score: np.ndarray, data: AblationPeriodData) -> np.ndarray:
        if self.variant == "base_identity":
            return np.asarray(base_score, dtype=float)
        matrix, _ = meta_matrix(self.variant, base_score, data)
        if self.model is None:
            raise RuntimeError(f"meta model missing for {self.variant}")
        return np.asarray(self.model.predict(matrix), dtype=float)


def subset_period_data(data: AblationPeriodData, positions: np.ndarray) -> AblationPeriodData:
    pos = np.asarray(positions, dtype=np.int64)
    return AblationPeriodData(
        timestamps_ns=np.asarray(data.timestamps_ns, dtype=np.int64)[pos],
        base_x=np.asarray(data.base_x)[pos],
        state_x=np.asarray(data.state_x)[pos],
        outcomes={name: np.asarray(values)[pos] for name, values in data.outcomes.items()},
        base_columns=data.base_columns,
        state_columns=data.state_columns,
        activity_persist_probability=None,
    )


def _state_matrix(data: AblationPeriodData, columns: Iterable[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    names = tuple(columns)
    missing = sorted(set(names) - set(data.state_columns))
    if missing:
        raise RuntimeError(f"R03.4.1 required soft-state columns missing: {missing}")
    positions = [data.state_columns.index(name) for name in names]
    matrix = np.asarray(data.state_x[:, positions], dtype=np.float32).copy()
    output_names: list[str] = []
    for index, name in enumerate(names):
        if name.endswith("_age_bars"):
            matrix[:, index] = np.log1p(np.maximum(matrix[:, index], 0.0))
            output_names.append(f"state::log1p_{name}")
        else:
            output_names.append(f"state::{name}")
    return matrix, tuple(output_names)


def meta_matrix(
    variant: str,
    base_score: np.ndarray,
    data: AblationPeriodData,
) -> tuple[np.ndarray, tuple[str, ...]]:
    score = np.asarray(base_score, dtype=np.float32).reshape(-1, 1)
    if len(score) != len(data.timestamps_ns):
        raise ValueError("base score and period data must share the decision axis")
    if variant == "base_identity":
        return score, ("base_long_score",)
    if variant == "score_only_meta":
        return score, ("base_long_score",)
    if variant == "score_plus_activity_meta":
        state, names = _state_matrix(data, ACTIVITY_SOFT_COLUMNS)
        return np.concatenate([score, state], axis=1), ("base_long_score", *names)
    if variant == "score_plus_strategic_meta":
        state, names = _state_matrix(data, STRATEGIC_SOFT_COLUMNS)
        return np.concatenate([score, state], axis=1), ("base_long_score", *names)
    if variant == "score_plus_strategic_activity_meta":
        state, names = _state_matrix(data, ALL_SOFT_COLUMNS)
        return np.concatenate([score, state], axis=1), ("base_long_score", *names)
    if variant == "soft_state_only_meta":
        return _state_matrix(data, ALL_SOFT_COLUMNS)
    raise ValueError(f"unsupported R03.4.1 variant: {variant}")


def build_expanding_oof_blocks(
    index: pd.DatetimeIndex,
    config: LongStateCalibrationConfig,
) -> tuple[OOFBlock, ...]:
    if len(index) < 2 or not index.is_monotonic_increasing:
        raise ValueError("OOF decision index must be monotonic and non-empty")
    first_prediction = index[0] + pd.Timedelta(days=config.oof_min_train_days)
    prediction_positions = np.flatnonzero(index >= first_prediction)
    if len(prediction_positions) < config.oof_blocks * 100:
        raise RuntimeError("insufficient rows for blocked OOF stacking")
    chunks = [chunk for chunk in np.array_split(prediction_positions, config.oof_blocks) if len(chunk)]
    embargo = pd.Timedelta(hours=config.oof_embargo_hours)
    blocks: list[OOFBlock] = []
    for block_id, chunk in enumerate(chunks, start=1):
        prediction_start = index[int(chunk[0])]
        prediction_end = index[int(chunk[-1])]
        train_end = prediction_start - embargo
        if train_end <= index[0]:
            continue
        blocks.append(OOFBlock(block_id, index[0], train_end, prediction_start, prediction_end))
    if len(blocks) < 3:
        raise RuntimeError("fewer than three valid OOF blocks")
    return tuple(blocks)


def _sample_positions(valid: np.ndarray, cap: int) -> np.ndarray:
    positions = np.flatnonzero(valid)
    if len(positions) <= cap:
        return positions
    return positions[np.linspace(0, len(positions) - 1, cap, dtype=np.int64)]


def fit_base_long_model(data: AblationPeriodData, config: LongStateCalibrationConfig) -> LongBaseModel:
    if LGBMRegressor is None:
        raise RuntimeError("R03.4.1 requires lightgbm")
    target = np.asarray(data.outcomes[f"long_utility_h{config.primary_horizon_hours}"], dtype=float)
    matrix = np.asarray(data.base_x, dtype=np.float32)
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(target)
    positions = _sample_positions(valid, config.train_sample_cap)
    if len(positions) < 1_000:
        raise RuntimeError(f"insufficient base long train rows: {len(positions)}")
    low, high = (float(np.quantile(target[positions], q)) for q in (0.0025, 0.9975))
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=config.base_n_estimators,
        learning_rate=config.base_learning_rate,
        num_leaves=config.base_num_leaves,
        min_child_samples=config.base_min_child_samples,
        colsample_bytree=0.80,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=config.random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(matrix[positions], np.clip(target[positions], low, high))
    return LongBaseModel(model=model, clip=(low, high))


def generate_oof_base_scores(
    fit: AblationPeriodData,
    config: LongStateCalibrationConfig,
) -> tuple[np.ndarray, tuple[OOFBlock, ...], list[dict[str, object]]]:
    index = fit.index
    blocks = build_expanding_oof_blocks(index, config)
    scores = np.full(len(index), np.nan, dtype=float)
    rows: list[dict[str, object]] = []
    for block in blocks:
        train_mask = (index >= block.train_start) & (index <= block.train_end)
        prediction_mask = (index >= block.prediction_start) & (index <= block.prediction_end)
        train_pos = np.flatnonzero(train_mask)
        prediction_pos = np.flatnonzero(prediction_mask)
        model = fit_base_long_model(subset_period_data(fit, train_pos), config)
        values = model.predict(np.asarray(fit.base_x[prediction_pos], dtype=np.float32))
        scores[prediction_pos] = values
        rows.append(
            {
                **block.to_dict(),
                "train_rows": int(len(train_pos)),
                "prediction_rows": int(len(prediction_pos)),
                "maximum_train_time_ns": int(np.asarray(fit.timestamps_ns)[train_pos].max()),
                "minimum_prediction_time_ns": int(np.asarray(fit.timestamps_ns)[prediction_pos].min()),
                "embargo_hours": config.oof_embargo_hours,
            }
        )
    return scores, blocks, rows


def fit_meta_model(
    variant: str,
    oof_base_score: np.ndarray,
    fit: AblationPeriodData,
    config: LongStateCalibrationConfig,
) -> MetaModel:
    if variant == "base_identity":
        return MetaModel(variant, None, ("base_long_score",))
    if LGBMRegressor is None:
        raise RuntimeError("R03.4.1 requires lightgbm")
    matrix, columns = meta_matrix(variant, oof_base_score, fit)
    target = np.asarray(fit.outcomes[f"long_utility_h{config.primary_horizon_hours}"], dtype=float)
    valid = (
        np.isfinite(matrix).all(axis=1)
        & np.isfinite(target)
        & np.isfinite(np.asarray(oof_base_score, dtype=float))
    )
    positions = _sample_positions(valid, config.train_sample_cap)
    if len(positions) < 5_000:
        raise RuntimeError(f"insufficient OOF meta rows variant={variant}: {len(positions)}")
    low, high = (float(np.quantile(target[positions], q)) for q in (0.005, 0.995))
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=config.meta_n_estimators,
        learning_rate=config.meta_learning_rate,
        num_leaves=config.meta_num_leaves,
        min_child_samples=config.meta_min_child_samples,
        colsample_bytree=1.0,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=1.0,
        reg_lambda=4.0,
        random_state=config.random_state + 17,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(matrix[positions], np.clip(target[positions], low, high))
    return MetaModel(variant=variant, model=model, columns=columns)


def rank_ic(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return np.nan
    x = pd.Series(np.asarray(left)[valid]).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(np.asarray(right)[valid]).rank(method="average").to_numpy(dtype=float)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def profit_factor(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    array = np.asarray(values, dtype=float)
    weight = np.ones(len(array), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    valid = np.isfinite(array) & np.isfinite(weight) & (weight >= 0)
    array, weight = array[valid], weight[valid]
    gains = float((array[array > 0] * weight[array > 0]).sum())
    losses = float((-array[array < 0] * weight[array < 0]).sum())
    return gains / losses if losses > 0 else np.inf if gains > 0 else np.nan


def empirical_percentile(calibration_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    calibration = np.sort(np.asarray(calibration_values, dtype=float)[np.isfinite(calibration_values)])
    output = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values)
    if not len(calibration):
        return output
    output[valid] = np.searchsorted(calibration, np.asarray(values)[valid], side="right") / len(calibration)
    return output


def state_rank_multiplier(
    calibration_base: np.ndarray,
    calibration_variant: np.ndarray,
    test_base: np.ndarray,
    test_variant: np.ndarray,
) -> np.ndarray:
    base_pct = empirical_percentile(calibration_base, test_base)
    variant_pct = empirical_percentile(calibration_variant, test_variant)
    return np.clip(1.0 + (variant_pct - base_pct), 0.5, 1.5)


def select_episode_peaks(
    timestamps_ns: np.ndarray,
    scores: np.ndarray,
    signal: np.ndarray,
    *,
    merge_gap_minutes: int,
    cooldown_hours: int,
) -> np.ndarray:
    times = np.asarray(timestamps_ns, dtype=np.int64)
    values = np.asarray(scores, dtype=float)
    active = np.flatnonzero(np.asarray(signal, dtype=bool) & np.isfinite(values))
    if not len(active):
        return np.empty(0, dtype=np.int64)
    merge_gap = int(pd.Timedelta(minutes=merge_gap_minutes).value)
    cooldown = int(pd.Timedelta(hours=cooldown_hours).value)
    peaks: list[int] = []
    episode: list[int] = [int(active[0])]
    for position in active[1:]:
        pos = int(position)
        if times[pos] - times[episode[-1]] <= merge_gap:
            episode.append(pos)
        else:
            peaks.append(max(episode, key=lambda item: values[item]))
            episode = [pos]
    peaks.append(max(episode, key=lambda item: values[item]))
    # Select the strongest episode peaks subject to a true pairwise cooldown.
    # A weak intermediate episode must not bridge two otherwise independent events.
    selected: list[int] = []
    for peak in sorted(peaks, key=lambda item: values[item], reverse=True):
        if all(abs(times[peak] - times[chosen]) >= cooldown for chosen in selected):
            selected.append(peak)
    selected.sort(key=lambda item: times[item])
    return np.asarray(selected, dtype=np.int64)


def _event_metrics(
    *,
    fold_id: str,
    variant: str,
    evaluation_mode: str,
    score: np.ndarray,
    positions: np.ndarray,
    test: AblationPeriodData,
    config: LongStateCalibrationConfig,
    threshold: float | None,
) -> list[dict[str, object]]:
    horizon = config.primary_horizon_hours
    utility = np.asarray(test.outcomes[f"long_utility_h{horizon}"], dtype=float)
    mfe = np.asarray(test.outcomes[f"long_mfe_h{horizon}"], dtype=float)
    mae = np.asarray(test.outcomes[f"long_mae_h{horizon}"], dtype=float)
    close_return = np.asarray(test.outcomes[f"future_close_return_h{horizon}"], dtype=float)
    pos = np.asarray(positions, dtype=np.int64)
    rows: list[dict[str, object]] = []
    for multiplier in config.cost_stress_multipliers:
        net = close_return[pos] - config.base_round_trip_cost * float(multiplier)
        rows.append(
            {
                "fold_id": fold_id,
                "variant": variant,
                "evaluation_mode": evaluation_mode,
                "cost_multiplier": multiplier,
                "threshold": threshold,
                "independent_events": int(len(pos)),
                "mean_score": float(np.nanmean(score[pos])) if len(pos) else np.nan,
                "mean_long_utility": float(np.nanmean(utility[pos])) if len(pos) else np.nan,
                "mean_mfe": float(np.nanmean(mfe[pos])) if len(pos) else np.nan,
                "mean_mae": float(np.nanmean(mae[pos])) if len(pos) else np.nan,
                "mfe_mae_ratio": float(np.nanmean(mfe[pos]) / max(np.nanmean(mae[pos]), 1e-12)) if len(pos) else np.nan,
                "mean_net_close_return": float(np.nanmean(net)) if len(pos) else np.nan,
                "win_rate": float(np.nanmean(net > 0)) if len(pos) else np.nan,
                "profit_factor": profit_factor(net),
            }
        )
    return rows


def evaluate_fold_models(
    *,
    fold_id: str,
    models: dict[str, MetaModel],
    base_model: LongBaseModel,
    calibration: AblationPeriodData,
    test: AblationPeriodData,
    config: LongStateCalibrationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cal_base = base_model.predict(np.asarray(calibration.base_x, dtype=np.float32))
    test_base = base_model.predict(np.asarray(test.base_x, dtype=np.float32))
    predictions_cal = {name: model.predict(cal_base, calibration) for name, model in models.items()}
    predictions_test = {name: model.predict(test_base, test) for name, model in models.items()}
    horizon = config.primary_horizon_hours
    actual_utility = np.asarray(test.outcomes[f"long_utility_h{horizon}"], dtype=float)
    actual_mfe = np.asarray(test.outcomes[f"long_mfe_h{horizon}"], dtype=float)
    actual_mae = np.asarray(test.outcomes[f"long_mae_h{horizon}"], dtype=float)
    close_return = np.asarray(test.outcomes[f"future_close_return_h{horizon}"], dtype=float)
    valid = (
        np.isfinite(test_base)
        & np.isfinite(actual_utility)
        & np.isfinite(actual_mfe)
        & np.isfinite(actual_mae)
        & np.isfinite(close_return)
    )
    if int(valid.sum()) < config.minimum_test_rows:
        raise RuntimeError(f"insufficient test rows {fold_id}: {int(valid.sum())}")
    times = np.asarray(test.timestamps_ns, dtype=np.int64)
    metric_rows: list[dict[str, object]] = []
    signal_rows: list[dict[str, object]] = []
    sample_parts: list[pd.DataFrame] = []

    for variant, score in predictions_test.items():
        score_valid = valid & np.isfinite(score)
        metric_rows.append(
            {
                "fold_id": fold_id,
                "variant": variant,
                "rows": int(score_valid.sum()),
                "long_utility_rank_ic": rank_ic(score[score_valid], actual_utility[score_valid]),
                "long_mfe_rank_ic": rank_ic(score[score_valid], actual_mfe[score_valid]),
                "negative_mae_rank_ic": rank_ic(score[score_valid], -actual_mae[score_valid]),
                "close_return_rank_ic": rank_ic(score[score_valid], close_return[score_valid]),
            }
        )
        cal_score = predictions_cal[variant]
        for quantile in config.signal_quantiles:
            threshold = float(np.nanquantile(cal_score, quantile))
            signal = score >= threshold
            positions = select_episode_peaks(
                times,
                score,
                signal & valid,
                merge_gap_minutes=config.episode_merge_gap_minutes,
                cooldown_hours=config.independent_event_cooldown_hours,
            )
            rows = _event_metrics(
                fold_id=fold_id,
                variant=variant,
                evaluation_mode=f"variant_q{int(quantile * 100)}",
                score=score,
                positions=positions,
                test=test,
                config=config,
                threshold=threshold,
            )
            for row in rows:
                row["dense_signal_count"] = int(np.sum(signal & valid))
                row["dense_signal_rate"] = float(np.mean(signal[valid]))
            signal_rows.extend(rows)
            if len(positions):
                sample_parts.append(
                    pd.DataFrame(
                        {
                            "fold_id": fold_id,
                            "variant": variant,
                            "evaluation_mode": f"variant_q{int(quantile * 100)}",
                            "decision_time": pd.to_datetime(times[positions], unit="ns"),
                            "score": score[positions],
                            "base_score": test_base[positions],
                            "long_utility": actual_utility[positions],
                            "mfe": actual_mfe[positions],
                            "mae": actual_mae[positions],
                            "gross_close_return": close_return[positions],
                        }
                    )
                )

    cal_candidate_threshold = float(np.nanquantile(cal_base, config.common_candidate_quantile))
    cal_candidate = cal_base >= cal_candidate_threshold
    test_candidate = test_base >= cal_candidate_threshold
    rerank_rows: list[dict[str, object]] = []
    for variant, score in predictions_test.items():
        cal_score = predictions_cal[variant]
        within_threshold = float(np.nanquantile(cal_score[cal_candidate], config.common_rerank_quantile))
        signal = test_candidate & (score >= within_threshold) & valid
        positions = select_episode_peaks(
            times,
            score,
            signal,
            merge_gap_minutes=config.episode_merge_gap_minutes,
            cooldown_hours=config.independent_event_cooldown_hours,
        )
        rows = _event_metrics(
            fold_id=fold_id,
            variant=variant,
            evaluation_mode="common_base_q80_rerank_top50pct",
            score=score,
            positions=positions,
            test=test,
            config=config,
            threshold=within_threshold,
        )
        for row in rows:
            row["base_candidate_threshold"] = cal_candidate_threshold
            row["dense_signal_count"] = int(np.sum(signal))
            row["dense_signal_rate"] = float(np.mean(signal[valid]))
        rerank_rows.extend(rows)

    base_q90 = float(np.nanquantile(cal_base, 0.90))
    base_signal = (test_base >= base_q90) & valid
    base_events = select_episode_peaks(
        times,
        test_base,
        base_signal,
        merge_gap_minutes=config.episode_merge_gap_minutes,
        cooldown_hours=config.independent_event_cooldown_hours,
    )
    multiplier_rows: list[dict[str, object]] = []
    for variant in ("score_only_meta", *STATE_META_VARIANTS):
        multiplier = state_rank_multiplier(
            predictions_cal["base_identity"],
            predictions_cal[variant],
            predictions_test["base_identity"],
            predictions_test[variant],
        )
        for cost_multiplier in config.cost_stress_multipliers:
            net = close_return[base_events] - config.base_round_trip_cost * float(cost_multiplier)
            weights = multiplier[base_events]
            multiplier_rows.append(
                {
                    "fold_id": fold_id,
                    "variant": variant,
                    "cost_multiplier": cost_multiplier,
                    "fixed_base_events": int(len(base_events)),
                    "mean_multiplier": float(np.nanmean(weights)) if len(weights) else np.nan,
                    "weighted_mean_net_close_return": float(np.nansum(net * weights) / np.nansum(weights)) if len(weights) and np.nansum(weights) > 0 else np.nan,
                    "weighted_profit_factor": profit_factor(net, weights),
                    "weighted_mean_mae": float(np.nansum(actual_mae[base_events] * weights) / np.nansum(weights)) if len(weights) and np.nansum(weights) > 0 else np.nan,
                    "weighted_mean_mfe": float(np.nansum(actual_mfe[base_events] * weights) / np.nansum(weights)) if len(weights) and np.nansum(weights) > 0 else np.nan,
                }
            )

    samples = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    return pd.DataFrame(metric_rows), pd.DataFrame(signal_rows), pd.DataFrame(rerank_rows), pd.DataFrame(multiplier_rows), samples


def build_uplift_tables(
    metrics: pd.DataFrame,
    rerank: pd.DataFrame,
    multipliers: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    controls = ("base_identity", "score_only_meta")
    for fold_id in sorted(metrics["fold_id"].unique()):
        fold_metrics = metrics.loc[metrics["fold_id"] == fold_id].set_index("variant")
        fold_rerank = rerank.loc[(rerank["fold_id"] == fold_id) & (rerank["cost_multiplier"] == 1.0)].set_index("variant")
        fold_multiplier = multipliers.loc[(multipliers["fold_id"] == fold_id) & (multipliers["cost_multiplier"] == 1.0)].set_index("variant")
        for variant in STATE_META_VARIANTS:
            if variant not in fold_metrics.index:
                continue
            row: dict[str, object] = {"fold_id": fold_id, "variant": variant}
            for control in controls:
                if control not in fold_metrics.index:
                    continue
                for column in ("long_utility_rank_ic", "long_mfe_rank_ic", "negative_mae_rank_ic", "close_return_rank_ic"):
                    row[f"delta_{column}_vs_{control}"] = float(fold_metrics.loc[variant, column] - fold_metrics.loc[control, column])
                if variant in fold_rerank.index and control in fold_rerank.index:
                    for column in ("mean_long_utility", "mean_mfe", "mean_mae", "mean_net_close_return", "profit_factor"):
                        row[f"delta_rerank_{column}_vs_{control}"] = float(fold_rerank.loc[variant, column] - fold_rerank.loc[control, column])
            if variant in fold_multiplier.index and "score_only_meta" in fold_multiplier.index:
                for column in ("weighted_mean_net_close_return", "weighted_profit_factor", "weighted_mean_mae", "weighted_mean_mfe"):
                    row[f"delta_{column}_vs_score_only_meta"] = float(fold_multiplier.loc[variant, column] - fold_multiplier.loc["score_only_meta", column])
            if variant in fold_rerank.index:
                row["independent_events"] = int(fold_rerank.loc[variant, "independent_events"])
            rows.append(row)
    return pd.DataFrame(rows)


def select_stable_candidates(uplift: pd.DataFrame, config: LongStateCalibrationConfig) -> pd.DataFrame:
    if uplift.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for variant, group in uplift.groupby("variant", sort=False):
        row: dict[str, object] = {"variant": variant}
        calibration_pass = True
        risk_pass = True
        for fold in ("WF_2024", "WF_2025"):
            subset = group.loc[group["fold_id"] == fold]
            if subset.empty:
                calibration_pass = False
                risk_pass = False
                continue
            item = subset.iloc[0]
            utility_delta = float(item.get("delta_long_utility_rank_ic_vs_score_only_meta", np.nan))
            rerank_utility = float(item.get("delta_rerank_mean_long_utility_vs_score_only_meta", np.nan))
            rerank_mae = float(item.get("delta_rerank_mean_mae_vs_score_only_meta", np.nan))
            weighted_net = float(item.get("delta_weighted_mean_net_close_return_vs_score_only_meta", np.nan))
            weighted_mae = float(item.get("delta_weighted_mean_mae_vs_score_only_meta", np.nan))
            events = int(item.get("independent_events", 0))
            row[f"{fold}_utility_ic_delta"] = utility_delta
            row[f"{fold}_rerank_utility_delta"] = rerank_utility
            row[f"{fold}_rerank_mae_delta"] = rerank_mae
            row[f"{fold}_weighted_net_delta"] = weighted_net
            row[f"{fold}_weighted_mae_delta"] = weighted_mae
            row[f"{fold}_events"] = events
            fold_calibration = (
                utility_delta >= config.minimum_long_utility_ic_increment
                and rerank_utility >= 0.0
                and rerank_mae <= config.maximum_mae_worsening
                and events >= config.minimum_independent_events
            )
            fold_risk = (
                weighted_net >= 0.0
                and weighted_mae <= 0.0
                and events >= config.minimum_independent_events
            )
            row[f"{fold}_calibration_pass"] = bool(fold_calibration)
            row[f"{fold}_risk_pass"] = bool(fold_risk)
            calibration_pass = calibration_pass and fold_calibration
            risk_pass = risk_pass and fold_risk
        row["passes_calibration"] = calibration_pass
        row["passes_risk_scaling"] = risk_pass
        utility_values = [float(row.get(f"{fold}_utility_ic_delta", np.nan)) for fold in ("WF_2024", "WF_2025")]
        risk_values = [float(row.get(f"{fold}_weighted_net_delta", np.nan)) for fold in ("WF_2024", "WF_2025")]
        row["stability_score"] = float(np.nanmin(utility_values) + 20.0 * np.nanmin(risk_values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["passes_calibration", "passes_risk_scaling", "stability_score"],
        ascending=[False, False, False],
        kind="stable",
    )
