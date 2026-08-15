#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02.3.1 residual rankers and retained sweep-geometry regressions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

try:
    from lightgbm import LGBMRanker, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRanker = None
    LGBMRegressor = None

from src.ai_research.latent_liquidity_first_touch_ranking.modeling import feature_columns as r02_2_feature_columns
from src.ai_research.latent_liquidity_excess_ranking.modeling import feature_columns as r02_3_feature_columns
from .config import HurdleResidualizationConfig

_NUISANCE_ACTIVITY_PREFIXES = (
    "macro_notional_intensity_", "macro_trades_intensity_", "macro_realized_vol_", "macro_range_bp_",
    "micro_path_notional_intensity_", "micro_path_trades_intensity_", "micro_path_realized_vol_", "micro_path_range_bp_",
)
_TARGET_PREFIXES = (
    "nuisance_", "raw_release_", "raw_log_release_", "raw_reversal_", "release_observed_",
    "excess_liquidity_", "density_vs_nuisance_", "release_probability_surprise",
    "positive_log_density_residual", "reversal_quality_residual", "sweep_depth_target_",
    "reversal_room_target_", "r02_3_1_", "excess_residual_", "reversal_residual_",
)
_QUALITY_METADATA_COLUMNS = {
    "r02_touch_consistent",
    "r02_3_source_eligible",
    "split_purge_eligible",
}
_GEOMETRY_EXCLUDED_PREFIXES = (
    "nuisance_", "raw_", "r02_3_1_", "excess_", "positive_log_density_residual",
    "reversal_quality_residual", "release_probability_surprise",
)


def residual_feature_columns(frame: pd.DataFrame, *, include_swing: bool) -> tuple[str, ...]:
    """Zone-specific path structure only; no raw distance or nuisance activity."""
    cols: list[str] = []
    for name in r02_2_feature_columns(frame, include_swing=include_swing):
        if name == "zone_distance_bp" or name in _QUALITY_METADATA_COLUMNS or name.startswith(_TARGET_PREFIXES):
            continue
        if name.startswith(_NUISANCE_ACTIVITY_PREFIXES):
            continue
        if name in {"nuisance_hour_sin", "nuisance_hour_cos", "nuisance_dow_sin", "nuisance_dow_cos"}:
            continue
        cols.append(name)
    return tuple(cols)


def geometry_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    # Keep the already-audited R02.3 No-Swing geometry schema for continuity,
    # but never allow R02.3.1 quality-control / nuisance outputs to leak in.
    return tuple(
        name
        for name in r02_3_feature_columns(frame, include_swing=False)
        if name not in _QUALITY_METADATA_COLUMNS
        and not name.startswith(_GEOMETRY_EXCLUDED_PREFIXES)
    )


def _ranker(config: HurdleResidualizationConfig) -> LGBMRanker:
    if LGBMRanker is None:
        raise RuntimeError("lightgbm is required for R02.3.1")
    gain = [0] + [(2 ** i) - 1 for i in range(1, config.rank_relevance_grades)]
    return LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=gain,
        n_estimators=config.model_n_estimators, learning_rate=config.model_learning_rate,
        num_leaves=config.model_num_leaves, min_child_samples=config.model_min_child_samples,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=1.5, reg_lambda=6.0,
        random_state=config.random_state, n_jobs=-1, verbosity=-1,
    )


def _regressor(config: HurdleResidualizationConfig) -> LGBMRegressor:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for R02.3.1")
    return LGBMRegressor(
        objective="huber", n_estimators=config.model_n_estimators,
        learning_rate=config.model_learning_rate, num_leaves=config.model_num_leaves,
        min_child_samples=config.model_min_child_samples, subsample=0.85,
        colsample_bytree=0.80, reg_alpha=1.0, reg_lambda=4.0,
        random_state=config.random_state, n_jobs=-1, verbosity=-1,
    )


def _cap_groups(frame: pd.DataFrame, cap_rows: int) -> pd.DataFrame:
    if len(frame) <= cap_rows:
        return frame
    groups = frame[["ranking_group"]].drop_duplicates().copy()
    groups["hash"] = pd.util.hash_pandas_object(groups["ranking_group"].astype(str), index=False).to_numpy(dtype=np.uint64)
    groups = groups.sort_values("hash", kind="mergesort")
    sizes = frame.groupby("ranking_group", sort=False).size()
    chosen: list[str] = []
    total = 0
    for group in groups["ranking_group"].astype(str):
        size = int(sizes.get(group, 0))
        if chosen and total + size > cap_rows:
            break
        chosen.append(group)
        total += size
    return frame.loc[frame["ranking_group"].astype(str).isin(chosen)].copy()


def _fit_ranker(frame: pd.DataFrame, cols: tuple[str, ...], relevance_col: str, config: HurdleResidualizationConfig) -> LGBMRanker:
    work = frame.sort_values(["ranking_group", "zone_distance_bp"], kind="mergesort")
    sizes = work.groupby("ranking_group", sort=False).size().to_numpy(dtype=np.int32)
    if len(sizes) < config.minimum_rank_groups:
        raise RuntimeError(f"R02.3.1 insufficient rank groups for {relevance_col}: {len(sizes)}")
    model = _ranker(config)
    model.fit(work.loc[:, cols], work[relevance_col].to_numpy(dtype=np.int16), group=sizes.tolist())
    return model


def _fit_regressor(frame: pd.DataFrame, cols: tuple[str, ...], target: str, config: HurdleResidualizationConfig) -> LGBMRegressor:
    work = frame.loc[pd.to_numeric(frame[target], errors="coerce").notna()].copy()
    if len(work) < config.minimum_regression_rows:
        raise RuntimeError(f"R02.3.1 insufficient regression rows for {target}: {len(work)}")
    if len(work) > config.model_train_cap_rows_per_side:
        key = work.get("zone_id", work.index.to_series()).astype(str)
        priority = pd.util.hash_pandas_object(key, index=False).to_numpy(dtype=np.uint64)
        keep = np.argpartition(priority, config.model_train_cap_rows_per_side - 1)[: config.model_train_cap_rows_per_side]
        work = work.iloc[np.sort(keep)].copy()
    model = _regressor(config)
    model.fit(work.loc[:, cols], pd.to_numeric(work[target], errors="coerce").to_numpy(dtype=float))
    return model


@dataclass
class SideModels:
    side: str
    path_columns: tuple[str, ...]
    full_columns: tuple[str, ...]
    geometry_columns: tuple[str, ...]
    excess_path: LGBMRanker
    excess_full: LGBMRanker
    reversal_path: LGBMRanker
    reversal_full: LGBMRanker
    sweep_depth: LGBMRegressor
    reversal_room: LGBMRegressor


@dataclass
class ModelBundle:
    by_side: dict[str, SideModels]


def fit_models(frame: pd.DataFrame, config: HurdleResidualizationConfig) -> ModelBundle:
    by_side: dict[str, SideModels] = {}
    for side in ("DOWN", "UP"):
        train = frame.loc[
            frame["period"].astype(str).eq(config.train_period)
            & frame["zone_side"].astype(str).eq(side)
            & frame["r02_3_1_source_eligible"].astype(bool)
            & frame["nuisance_prediction_source"].astype(str).eq("TRAIN_EXPANDING_OOS")
        ].copy()
        path_cols = residual_feature_columns(train, include_swing=False)
        full_cols = residual_feature_columns(train, include_swing=True)
        geom_cols = geometry_feature_columns(train)
        if not path_cols or not full_cols or not geom_cols:
            raise RuntimeError(f"R02.3.1 empty feature schema for {side}")
        excess = train.loc[train["excess_residual_group_eligible"].astype(bool) & train["excess_residual_relevance"].ge(0)].copy()
        reversal = train.loc[train["reversal_residual_group_eligible"].astype(bool) & train["reversal_residual_relevance"].ge(0)].copy()
        excess = _cap_groups(excess, config.model_train_cap_rows_per_side)
        reversal = _cap_groups(reversal, config.model_train_cap_rows_per_side)
        release = train.loc[train["release_observed_180s"].astype(bool)].copy()
        by_side[side] = SideModels(
            side=side,
            path_columns=path_cols,
            full_columns=full_cols,
            geometry_columns=geom_cols,
            excess_path=_fit_ranker(excess, path_cols, "excess_residual_relevance", config),
            excess_full=_fit_ranker(excess, full_cols, "excess_residual_relevance", config),
            reversal_path=_fit_ranker(reversal, path_cols, "reversal_residual_relevance", config),
            reversal_full=_fit_ranker(reversal, full_cols, "reversal_residual_relevance", config),
            sweep_depth=_fit_regressor(release, geom_cols, "sweep_depth_target_bp", config),
            reversal_room=_fit_regressor(release, geom_cols, "reversal_room_target_bp", config),
        )
    return ModelBundle(by_side=by_side)


def _group_percentile(values: pd.Series, groups: pd.Series) -> pd.Series:
    work = pd.DataFrame({"v": pd.to_numeric(values, errors="coerce"), "g": groups}, index=values.index)
    return work.groupby("g", sort=False)["v"].rank(method="average", pct=True)


def predict(frame: pd.DataFrame, models: ModelBundle) -> pd.DataFrame:
    out = frame.copy()
    for name in (
        "score_excess_path_no_swing", "score_excess_full_with_swing",
        "score_reversal_path_no_swing", "score_reversal_full_with_swing",
        "pred_sweep_depth_bp", "pred_reversal_room_bp",
    ):
        out[name] = np.nan
    out["score_distance_near"] = -pd.to_numeric(out["zone_distance_bp"], errors="coerce")
    out["score_distance_far"] = pd.to_numeric(out["zone_distance_bp"], errors="coerce")
    out["score_nuisance_expected_density"] = pd.to_numeric(out["nuisance_expected_density"], errors="coerce")
    out["score_nuisance_expected_reversal"] = pd.to_numeric(out["nuisance_expected_reversal_quality"], errors="coerce")
    for side, bundle in models.by_side.items():
        mask = out["zone_side"].astype(str).eq(side)
        if not mask.any():
            continue
        xp = out.loc[mask, bundle.path_columns]
        xf = out.loc[mask, bundle.full_columns]
        xg = out.loc[mask, bundle.geometry_columns]
        out.loc[mask, "score_excess_path_no_swing"] = bundle.excess_path.predict(xp)
        out.loc[mask, "score_excess_full_with_swing"] = bundle.excess_full.predict(xf)
        out.loc[mask, "score_reversal_path_no_swing"] = bundle.reversal_path.predict(xp)
        out.loc[mask, "score_reversal_full_with_swing"] = bundle.reversal_full.predict(xf)
        out.loc[mask, "pred_sweep_depth_bp"] = np.maximum(0.0, bundle.sweep_depth.predict(xg))
        out.loc[mask, "pred_reversal_room_bp"] = np.maximum(0.0, bundle.reversal_room.predict(xg))
    ex_pct = _group_percentile(out["score_excess_path_no_swing"], out["ranking_group"])
    rv_pct = _group_percentile(out["score_reversal_path_no_swing"], out["ranking_group"])
    out["score_joint_residual_path_no_swing"] = 0.5 * ex_pct + 0.5 * rv_pct
    return out


def _ndcg(target: np.ndarray, score: np.ndarray, k: int) -> float:
    valid = np.isfinite(target) & np.isfinite(score)
    y, s = target[valid], score[valid]
    if len(y) < 2 or float(np.max(y) - np.min(y)) <= 1e-12:
        return np.nan
    y = (y - np.min(y)) / max(float(np.max(y) - np.min(y)), 1e-12)
    kk = min(int(k), len(y))
    order = np.argsort(-s, kind="stable")[:kk]
    ideal = np.argsort(-y, kind="stable")[:kk]
    discount = np.log2(np.arange(2, kk + 2, dtype=float))
    dcg = float(np.sum(y[order] / discount))
    idcg = float(np.sum(y[ideal] / discount))
    return dcg / idcg if idcg > 1e-12 else np.nan


def _within_group(group: pd.DataFrame, score_col: str, target_col: str, eligible_col: str) -> dict[str, float]:
    g = group.loc[group[eligible_col].astype(bool)].copy()
    y = pd.to_numeric(g[target_col], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(g[score_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(s)
    y, s = y[valid], s[valid]
    if len(y) < 2 or float(np.max(y) - np.min(y)) <= 1e-12:
        return {"spearman": np.nan, "ndcg1": np.nan, "ndcg3": np.nan, "pairwise_accuracy": np.nan}
    rho = spearmanr(y, s).statistic if np.nanstd(s) > 1e-12 else np.nan
    dy = y[:, None] - y[None, :]
    ds = s[:, None] - s[None, :]
    tri = np.triu(np.ones_like(dy, dtype=bool), 1) & (np.abs(dy) > 1e-12)
    acc = float(np.mean(np.sign(dy[tri]) == np.sign(ds[tri]))) if tri.any() else np.nan
    return {
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
        "ndcg1": _ndcg(y, s, 1),
        "ndcg3": _ndcg(y, s, 3),
        "pairwise_accuracy": acc,
    }


def ranking_metrics(pred: pd.DataFrame, config: HurdleResidualizationConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tasks = {
        "EXCESS_RESIDUAL": (
            "excess_liquidity_residual", "excess_residual_group_eligible",
            {
                "PATH_NO_SWING": "score_excess_path_no_swing",
                "FULL_WITH_SWING": "score_excess_full_with_swing",
                "NUISANCE_EXPECTED": "score_nuisance_expected_density",
                "DISTANCE_NEAR": "score_distance_near",
                "DISTANCE_FAR": "score_distance_far",
            },
        ),
        "REVERSAL_RESIDUAL": (
            "reversal_quality_residual", "reversal_residual_group_eligible",
            {
                "PATH_NO_SWING": "score_reversal_path_no_swing",
                "FULL_WITH_SWING": "score_reversal_full_with_swing",
                "NUISANCE_EXPECTED": "score_nuisance_expected_reversal",
                "DISTANCE_NEAR": "score_distance_near",
                "DISTANCE_FAR": "score_distance_far",
            },
        ),
    }
    for (period, side), sf in pred.groupby(["period", "zone_side"], sort=True):
        for task, (target, eligible_col, score_map) in tasks.items():
            for model_name, score_col in score_map.items():
                stats: list[dict[str, float]] = []
                for _, group in sf.groupby("ranking_group", sort=False):
                    values = _within_group(group, score_col, target, eligible_col)
                    if any(np.isfinite(v) for v in values.values()):
                        stats.append(values)
                if not stats:
                    rows.append({"period": period, "zone_side": side, "task": task, "model": model_name, "rank_groups": 0})
                    continue
                t = pd.DataFrame(stats)
                rows.append({
                    "period": period, "zone_side": side, "task": task, "model": model_name,
                    "rank_groups": int(len(t)),
                    "mean_group_spearman": float(t["spearman"].mean()),
                    "median_group_spearman": float(t["spearman"].median()),
                    "mean_ndcg1": float(t["ndcg1"].mean()),
                    "mean_ndcg3": float(t["ndcg3"].mean()),
                    "mean_pairwise_accuracy": float(t["pairwise_accuracy"].mean()),
                })
    return pd.DataFrame(rows)


def regression_metrics(pred: pd.DataFrame, config: HurdleResidualizationConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (period, side), sf in pred.groupby(["period", "zone_side"], sort=True):
        work = sf.loc[sf["r02_3_1_source_eligible"].astype(bool) & sf["release_observed_180s"].astype(bool)].copy()
        for task, target, score in (
            ("SWEEP_DEPTH", "sweep_depth_target_bp", "pred_sweep_depth_bp"),
            ("REVERSAL_ROOM", "reversal_room_target_bp", "pred_reversal_room_bp"),
        ):
            y = pd.to_numeric(work[target], errors="coerce").to_numpy(dtype=float)
            p = pd.to_numeric(work[score], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(y) & np.isfinite(p)
            rho = spearmanr(y[valid], p[valid]).statistic if int(valid.sum()) >= 20 and np.nanstd(y[valid]) > 1e-12 and np.nanstd(p[valid]) > 1e-12 else np.nan
            rows.append({
                "period": period, "zone_side": side, "task": task, "rows": int(valid.sum()),
                "mae_bp": float(mean_absolute_error(y[valid], p[valid])) if valid.any() else np.nan,
                "spearman": float(rho) if np.isfinite(rho) else np.nan,
            })
    return pd.DataFrame(rows)


def top_zone_summary(pred: pd.DataFrame, config: HurdleResidualizationConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    task_defs = {
        "EXCESS_RESIDUAL": (
            "excess_liquidity_residual", "excess_residual_group_eligible",
            {"PATH_NO_SWING": "score_excess_path_no_swing", "FULL_WITH_SWING": "score_excess_full_with_swing", "NUISANCE_EXPECTED": "score_nuisance_expected_density", "DISTANCE_NEAR": "score_distance_near", "DISTANCE_FAR": "score_distance_far"},
        ),
        "REVERSAL_RESIDUAL": (
            "reversal_quality_residual", "reversal_residual_group_eligible",
            {"PATH_NO_SWING": "score_reversal_path_no_swing", "FULL_WITH_SWING": "score_reversal_full_with_swing", "NUISANCE_EXPECTED": "score_nuisance_expected_reversal", "DISTANCE_NEAR": "score_distance_near", "DISTANCE_FAR": "score_distance_far"},
        ),
    }
    w = config.primary_label_window_seconds
    for (period, side), sf in pred.groupby(["period", "zone_side"], sort=True):
        sf = sf.loc[sf["split_purge_eligible"].astype(bool)].copy()
        for task, (target, eligible_col, score_map) in task_defs.items():
            all_eligible = sf.loc[sf[eligible_col].astype(bool)].copy()
            baseline_target = float(pd.to_numeric(all_eligible[target], errors="coerce").mean()) if not all_eligible.empty else np.nan
            for model_name, score_col in score_map.items():
                selected: list[pd.Series] = []
                oracle_hits: list[bool] = []
                for _, group in sf.groupby("ranking_group", sort=False):
                    scores = pd.to_numeric(group[score_col], errors="coerce")
                    if scores.notna().any():
                        selected.append(group.loc[scores.idxmax()])
                    eligible = group.loc[group[eligible_col].astype(bool)].copy()
                    if len(eligible) < 2:
                        continue
                    target_values = pd.to_numeric(eligible[target], errors="coerce")
                    if target_values.notna().sum() < 2 or float(target_values.max() - target_values.min()) <= 1e-12:
                        continue
                    all_scores = pd.to_numeric(group[score_col], errors="coerce")
                    top3 = set(group.loc[all_scores.nlargest(min(3, len(group))).index, "zone_id"].astype(str))
                    oracle_hits.append(str(eligible.loc[target_values.idxmax(), "zone_id"]) in top3)
                sel = pd.DataFrame(selected) if selected else pd.DataFrame()
                touched = sel.loc[sel.get("r02_3_1_source_eligible", pd.Series(False, index=sel.index)).astype(bool)].copy() if not sel.empty else pd.DataFrame()
                released = touched.loc[touched.get("release_observed_180s", pd.Series(False, index=touched.index)).astype(bool)].copy() if not touched.empty else pd.DataFrame()
                actual = float(pd.to_numeric(touched.get(target), errors="coerce").mean()) if not touched.empty else np.nan
                ratio = float(pd.to_numeric(touched.get("density_vs_nuisance_expected_ratio"), errors="coerce").mean()) if not touched.empty else np.nan
                rows.append({
                    "period": period, "zone_side": side, "task": task, "model": model_name,
                    "groups": int(sf["ranking_group"].nunique()),
                    "top1_touched": int(len(touched)),
                    "top1_touch_rate": float(len(touched) / max(len(sel), 1)),
                    "top1_mean_target": actual,
                    "all_eligible_mean_target": baseline_target,
                    "top1_mean_actual_to_nuisance_expected_density_ratio": ratio,
                    "top1_median_actual_to_nuisance_expected_density_ratio": float(pd.to_numeric(touched.get("density_vs_nuisance_expected_ratio"), errors="coerce").median()) if not touched.empty else np.nan,
                    "top1_release_rate": float(touched["release_observed_180s"].astype(bool).mean()) if not touched.empty else np.nan,
                    "top1_favorable_rate": float(pd.to_numeric(touched.get(f"ft_favorable_episode_count_{w}s"), errors="coerce").gt(0).mean()) if not touched.empty else np.nan,
                    "top1_continuation_rate": float(pd.to_numeric(touched.get(f"ft_continuation_episode_count_{w}s"), errors="coerce").gt(0).mean()) if not touched.empty else np.nan,
                    "top1_mean_distance_bp": float(pd.to_numeric(sel.get("zone_distance_bp"), errors="coerce").mean()) if not sel.empty else np.nan,
                    "top1_mean_sweep_depth_bp": float(pd.to_numeric(released.get("sweep_depth_target_bp"), errors="coerce").mean()) if not released.empty else np.nan,
                    "top1_mean_reversal_room_bp": float(pd.to_numeric(released.get("reversal_room_target_bp"), errors="coerce").mean()) if not released.empty else np.nan,
                    "oracle_strongest_zone_in_top3_rate": float(np.mean(oracle_hits)) if oracle_hits else np.nan,
                })
    return pd.DataFrame(rows)


def feature_importance(models: ModelBundle) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for side, bundle in models.by_side.items():
        for task, model_name, model, cols in (
            ("EXCESS", "PATH_NO_SWING", bundle.excess_path, bundle.path_columns),
            ("EXCESS", "FULL_WITH_SWING", bundle.excess_full, bundle.full_columns),
            ("REVERSAL", "PATH_NO_SWING", bundle.reversal_path, bundle.path_columns),
            ("REVERSAL", "FULL_WITH_SWING", bundle.reversal_full, bundle.full_columns),
            ("SWEEP_DEPTH", "PATH_NO_SWING", bundle.sweep_depth, bundle.geometry_columns),
            ("REVERSAL_ROOM", "PATH_NO_SWING", bundle.reversal_room, bundle.geometry_columns),
        ):
            raw = np.asarray(model.feature_importances_, dtype=float)[: len(cols)]
            denom = max(float(np.sum(raw)), 1e-12)
            for name, value in zip(cols, raw, strict=True):
                rows.append({
                    "zone_side": side, "task": task, "model": model_name, "feature": name,
                    "feature_family": "SWING_SUPPLEMENT" if name.startswith("swing_") else "LIQUIDITY_PATH",
                    "importance": float(value), "importance_share": float(value / denom),
                })
    return pd.DataFrame(rows).sort_values(["zone_side", "task", "model", "importance"], ascending=[True, True, True, False]).reset_index(drop=True)
