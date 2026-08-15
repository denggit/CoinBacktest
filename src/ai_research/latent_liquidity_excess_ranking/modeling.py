#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02.3 cross-sectional rankers and sweep-geometry regressors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from lightgbm import LGBMRanker, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRanker = None
    LGBMRegressor = None

from src.ai_research.latent_liquidity_first_touch_ranking.modeling import feature_columns as r02_2_feature_columns
from .config import ExcessLiquidityRankingConfig

_NEW_TARGET_PREFIXES = (
    "expected_", "excess_", "density_vs_expected", "reversal_quality", "release_observed_",
    "favorable_observed_", "continuation_observed_", "sweep_depth_target", "reversal_room_target",
    "first_touch_available_time", "r02_touch_consistent", "r02_3_source_eligible",
)


def feature_columns(frame: pd.DataFrame, *, include_swing: bool) -> tuple[str, ...]:
    return tuple(
        name for name in r02_2_feature_columns(frame, include_swing=include_swing)
        if not name.startswith(_NEW_TARGET_PREFIXES)
        and name not in {"excess_relevance", "reversal_relevance", "excess_group_eligible", "reversal_group_eligible", "robust_scale", "zone_distance_bp"}
    )


def _ranker(config: ExcessLiquidityRankingConfig) -> LGBMRanker:
    if LGBMRanker is None:
        raise RuntimeError("lightgbm is required for R02.3")
    gain = [0] + [(2 ** i) - 1 for i in range(1, config.rank_relevance_grades)]
    return LGBMRanker(
        objective="lambdarank", metric="ndcg", label_gain=gain,
        n_estimators=config.model_n_estimators, learning_rate=config.model_learning_rate,
        num_leaves=config.model_num_leaves, min_child_samples=config.model_min_child_samples,
        subsample=0.85, colsample_bytree=0.80, reg_alpha=1.0, reg_lambda=4.0,
        random_state=config.random_state, n_jobs=-1, verbosity=-1,
    )


def _regressor(config: ExcessLiquidityRankingConfig) -> LGBMRegressor:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for R02.3")
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
        chosen.append(group); total += size
    return frame.loc[frame["ranking_group"].astype(str).isin(chosen)].copy()


def _fit_ranker(
    frame: pd.DataFrame,
    *, cols: tuple[str, ...], relevance_col: str, config: ExcessLiquidityRankingConfig,
) -> LGBMRanker:
    work = frame.sort_values(["ranking_group", "zone_distance_bp"], kind="mergesort")
    sizes = work.groupby("ranking_group", sort=False).size().to_numpy(dtype=np.int32)
    if len(sizes) < config.minimum_rank_groups:
        raise RuntimeError(f"R02.3 insufficient rank groups for {relevance_col}: {len(sizes)}")
    model = _ranker(config)
    model.fit(work.loc[:, cols], work[relevance_col].to_numpy(dtype=np.int16), group=sizes.tolist())
    return model


def _fit_regressor(frame: pd.DataFrame, cols: tuple[str, ...], target: str, config: ExcessLiquidityRankingConfig) -> LGBMRegressor:
    work = frame.loc[pd.to_numeric(frame[target], errors="coerce").notna()].copy()
    if len(work) < config.minimum_regression_rows:
        raise RuntimeError(f"R02.3 insufficient regression rows for {target}: {len(work)}")
    if len(work) > config.model_train_cap_rows_per_side:
        priority = pd.util.hash_pandas_object(work["zone_id"].astype(str), index=False).to_numpy(dtype=np.uint64)
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
    excess_path: LGBMRanker
    excess_full: LGBMRanker
    reversal_path: LGBMRanker
    reversal_full: LGBMRanker
    sweep_depth_path: LGBMRegressor
    reversal_room_path: LGBMRegressor


@dataclass
class ModelBundle:
    by_side: dict[str, SideModels]


def fit_models(frame: pd.DataFrame, config: ExcessLiquidityRankingConfig) -> ModelBundle:
    by_side: dict[str, SideModels] = {}
    for side in ("DOWN", "UP"):
        base = frame.loc[
            frame["period"].astype(str).eq(config.train_period)
            & frame["zone_side"].astype(str).eq(side)
            & frame["r02_3_source_eligible"].astype(bool)
        ].copy()
        path_cols = feature_columns(base, include_swing=False)
        full_cols = feature_columns(base, include_swing=True)
        if not path_cols or not full_cols:
            raise RuntimeError(f"R02.3 empty feature schema for {side}")

        excess = base.loc[base["excess_group_eligible"].astype(bool) & base["excess_relevance"].ge(0)].copy()
        reversal = base.loc[base["reversal_group_eligible"].astype(bool) & base["reversal_relevance"].ge(0)].copy()
        excess = _cap_groups(excess, config.model_train_cap_rows_per_side)
        reversal = _cap_groups(reversal, config.model_train_cap_rows_per_side)
        release = base.loc[base["release_observed_180s"].astype(bool)].copy()
        by_side[side] = SideModels(
            side=side,
            path_columns=path_cols,
            full_columns=full_cols,
            excess_path=_fit_ranker(excess, cols=path_cols, relevance_col="excess_relevance", config=config),
            excess_full=_fit_ranker(excess, cols=full_cols, relevance_col="excess_relevance", config=config),
            reversal_path=_fit_ranker(reversal, cols=path_cols, relevance_col="reversal_relevance", config=config),
            reversal_full=_fit_ranker(reversal, cols=full_cols, relevance_col="reversal_relevance", config=config),
            sweep_depth_path=_fit_regressor(release, path_cols, "sweep_depth_target_bp", config),
            reversal_room_path=_fit_regressor(release, path_cols, "reversal_room_target_bp", config),
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
    for side, bundle in models.by_side.items():
        mask = out["zone_side"].astype(str).eq(side)
        if not mask.any():
            continue
        xp = out.loc[mask, bundle.path_columns]
        xf = out.loc[mask, bundle.full_columns]
        out.loc[mask, "score_excess_path_no_swing"] = bundle.excess_path.predict(xp)
        out.loc[mask, "score_excess_full_with_swing"] = bundle.excess_full.predict(xf)
        out.loc[mask, "score_reversal_path_no_swing"] = bundle.reversal_path.predict(xp)
        out.loc[mask, "score_reversal_full_with_swing"] = bundle.reversal_full.predict(xf)
        out.loc[mask, "pred_sweep_depth_bp"] = np.maximum(0.0, bundle.sweep_depth_path.predict(xp))
        out.loc[mask, "pred_reversal_room_bp"] = np.maximum(0.0, bundle.reversal_room_path.predict(xp))
    ex_pct = _group_percentile(out["score_excess_path_no_swing"], out["ranking_group"])
    rv_pct = _group_percentile(out["score_reversal_path_no_swing"], out["ranking_group"])
    out["score_joint_path_no_swing"] = 0.5 * ex_pct + 0.5 * rv_pct
    return out


def _ndcg(target: np.ndarray, score: np.ndarray, k: int) -> float:
    valid = np.isfinite(target) & np.isfinite(score)
    y = target[valid]; s = score[valid]
    if len(y) < 2 or float(np.max(y) - np.min(y)) <= 1e-12:
        return np.nan
    y = (y - np.min(y)) / max(float(np.max(y) - np.min(y)), 1e-12)
    kk = min(int(k), len(y))
    order = np.argsort(-s, kind="stable")[:kk]
    ideal = np.argsort(-y, kind="stable")[:kk]
    discount = np.log2(np.arange(2, kk + 2, dtype=float))
    dcg = float(np.sum(y[order] / discount)); idcg = float(np.sum(y[ideal] / discount))
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
    dy = y[:, None] - y[None, :]; ds = s[:, None] - s[None, :]
    tri = np.triu(np.ones_like(dy, dtype=bool), 1) & (np.abs(dy) > 1e-12)
    acc = float(np.mean(np.sign(dy[tri]) == np.sign(ds[tri]))) if tri.any() else np.nan
    return {
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
        "ndcg1": _ndcg(y, s, 1), "ndcg3": _ndcg(y, s, 3), "pairwise_accuracy": acc,
    }


def ranking_metrics(pred: pd.DataFrame, config: ExcessLiquidityRankingConfig) -> pd.DataFrame:
    tasks = {
        "EXCESS_LIQUIDITY": ("excess_liquidity_z", "excess_group_eligible", {
            "PATH_NO_SWING": "score_excess_path_no_swing",
            "FULL_WITH_SWING": "score_excess_full_with_swing",
            "DISTANCE_NEAR": "score_distance_near",
            "DISTANCE_FAR": "score_distance_far",
        }),
        "REVERSAL_QUALITY": ("reversal_quality_target", "reversal_group_eligible", {
            "PATH_NO_SWING": "score_reversal_path_no_swing",
            "FULL_WITH_SWING": "score_reversal_full_with_swing",
            "DISTANCE_NEAR": "score_distance_near",
            "DISTANCE_FAR": "score_distance_far",
        }),
    }
    rows: list[dict[str, object]] = []
    for (period, side), sf in pred.groupby(["period", "zone_side"], sort=True):
        for task, (target, eligible, score_map) in tasks.items():
            for model_name, score_col in score_map.items():
                stats = []
                for _, group in sf.groupby("ranking_group", sort=False):
                    m = _within_group(group, score_col, target, eligible)
                    if any(np.isfinite(v) for v in m.values()):
                        stats.append(m)
                if not stats:
                    rows.append({"period": period, "zone_side": side, "task": task, "model": model_name, "rank_groups": 0})
                    continue
                tab = pd.DataFrame(stats)
                rows.append({
                    "period": period, "zone_side": side, "task": task, "model": model_name,
                    "rank_groups": int(len(tab)),
                    "mean_group_spearman": float(tab["spearman"].mean()),
                    "median_group_spearman": float(tab["spearman"].median()),
                    "mean_ndcg1": float(tab["ndcg1"].mean()),
                    "mean_ndcg3": float(tab["ndcg3"].mean()),
                    "mean_pairwise_accuracy": float(tab["pairwise_accuracy"].mean()),
                })
    return pd.DataFrame(rows)


def regression_metrics(pred: pd.DataFrame, config: ExcessLiquidityRankingConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (period, side), sf in pred.groupby(["period", "zone_side"], sort=True):
        release = sf.loc[sf["r02_3_source_eligible"].astype(bool) & sf["release_observed_180s"].astype(bool)]
        for task, target, score in (
            ("SWEEP_DEPTH", "sweep_depth_target_bp", "pred_sweep_depth_bp"),
            ("REVERSAL_ROOM", "reversal_room_target_bp", "pred_reversal_room_bp"),
        ):
            y = pd.to_numeric(release[target], errors="coerce").to_numpy(dtype=float)
            p = pd.to_numeric(release[score], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(y) & np.isfinite(p)
            y, p = y[valid], p[valid]
            rho = spearmanr(y, p).statistic if len(y) >= 2 and np.nanstd(y) > 1e-12 and np.nanstd(p) > 1e-12 else np.nan
            rows.append({
                "period": period, "zone_side": side, "task": task, "rows": int(len(y)),
                "spearman": float(rho) if np.isfinite(rho) else np.nan,
                "mae_bp": float(np.mean(np.abs(y - p))) if len(y) else np.nan,
                "actual_mean_bp": float(np.mean(y)) if len(y) else np.nan,
                "pred_mean_bp": float(np.mean(p)) if len(y) else np.nan,
            })
    return pd.DataFrame(rows)


def top_zone_summary(pred: pd.DataFrame, config: ExcessLiquidityRankingConfig) -> pd.DataFrame:
    score_map = {
        "EXCESS_PATH_NO_SWING": "score_excess_path_no_swing",
        "EXCESS_FULL_WITH_SWING": "score_excess_full_with_swing",
        "REVERSAL_PATH_NO_SWING": "score_reversal_path_no_swing",
        "JOINT_PATH_NO_SWING": "score_joint_path_no_swing",
        "DISTANCE_NEAR": "score_distance_near",
        "DISTANCE_FAR": "score_distance_far",
    }
    rows: list[dict[str, object]] = []
    w = config.primary_label_window_seconds
    for (period, side), sf in pred.groupby(["period", "zone_side"], sort=True):
        eligible = sf.loc[sf["r02_3_source_eligible"].astype(bool)].copy()
        all_excess = float(pd.to_numeric(eligible["excess_liquidity_z"], errors="coerce").mean()) if not eligible.empty else np.nan
        all_ratio = float(pd.to_numeric(eligible["density_vs_expected_ratio"], errors="coerce").mean()) if not eligible.empty else np.nan
        all_density = float(pd.to_numeric(eligible[f"ft_release_density_sum_{w}s"], errors="coerce").mean()) if not eligible.empty else np.nan
        for model_name, score_col in score_map.items():
            selected = []
            oracle_hits = []
            for _, group in sf.groupby("ranking_group", sort=False):
                score = pd.to_numeric(group[score_col], errors="coerce")
                if score.notna().any():
                    selected.append(group.loc[score.idxmax()])
                touched = group.loc[group["r02_3_source_eligible"].astype(bool)].copy()
                if len(touched) < 2:
                    continue
                target = pd.to_numeric(touched["excess_liquidity_z"], errors="coerce")
                if target.notna().sum() < 2:
                    continue
                top3_idx = score.nlargest(min(3, len(group))).index
                top3 = set(group.loc[top3_idx, "zone_id"].astype(str))
                oracle_hits.append(str(touched.loc[target.idxmax(), "zone_id"]) in top3)
            sel = pd.DataFrame(selected) if selected else pd.DataFrame()
            touched_sel = sel.loc[sel.get("r02_3_source_eligible", pd.Series(False, index=sel.index)).astype(bool)].copy() if not sel.empty else pd.DataFrame()
            if touched_sel.empty:
                metrics = {name: np.nan for name in (
                    "top1_mean_excess_z", "top1_mean_density_ratio", "top1_mean_raw_density",
                    "top1_favorable_rate", "top1_continuation_rate", "top1_mean_reversal_quality", "top1_mean_distance_bp",
                )}
            else:
                metrics = {
                    "top1_mean_excess_z": float(pd.to_numeric(touched_sel["excess_liquidity_z"], errors="coerce").mean()),
                    "top1_mean_density_ratio": float(pd.to_numeric(touched_sel["density_vs_expected_ratio"], errors="coerce").mean()),
                    "top1_mean_raw_density": float(pd.to_numeric(touched_sel[f"ft_release_density_sum_{w}s"], errors="coerce").mean()),
                    "top1_favorable_rate": float(touched_sel["favorable_observed_180s"].astype(bool).mean()),
                    "top1_continuation_rate": float(touched_sel["continuation_observed_180s"].astype(bool).mean()),
                    "top1_mean_reversal_quality": float(pd.to_numeric(touched_sel["reversal_quality_target"], errors="coerce").mean()),
                    "top1_mean_distance_bp": float(pd.to_numeric(touched_sel["zone_distance_bp"], errors="coerce").mean()),
                }
            rows.append({
                "period": period, "zone_side": side, "model": model_name,
                "groups": int(sf["ranking_group"].nunique()), "top1_touched": int(len(touched_sel)),
                "top1_touch_rate": float(len(touched_sel) / max(len(sel), 1)),
                **metrics,
                "all_touched_mean_excess_z": all_excess,
                "all_touched_mean_density_ratio": all_ratio,
                "all_touched_mean_raw_density": all_density,
                "top1_excess_z_uplift": metrics["top1_mean_excess_z"] - all_excess if np.isfinite(metrics["top1_mean_excess_z"]) and np.isfinite(all_excess) else np.nan,
                "top1_density_ratio_lift": metrics["top1_mean_density_ratio"] / all_ratio if np.isfinite(metrics["top1_mean_density_ratio"]) and np.isfinite(all_ratio) and all_ratio > 1e-12 else np.nan,
                "oracle_strongest_excess_zone_in_top3_rate": float(np.mean(oracle_hits)) if oracle_hits else np.nan,
            })
    return pd.DataFrame(rows)


def feature_importance(models: ModelBundle) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for side, b in models.by_side.items():
        specs = (
            ("EXCESS", "PATH_NO_SWING", b.excess_path, b.path_columns),
            ("EXCESS", "FULL_WITH_SWING", b.excess_full, b.full_columns),
            ("REVERSAL", "PATH_NO_SWING", b.reversal_path, b.path_columns),
            ("REVERSAL", "FULL_WITH_SWING", b.reversal_full, b.full_columns),
            ("SWEEP_DEPTH", "PATH_NO_SWING", b.sweep_depth_path, b.path_columns),
            ("REVERSAL_ROOM", "PATH_NO_SWING", b.reversal_room_path, b.path_columns),
        )
        for task, model_name, model, cols in specs:
            raw = np.asarray(model.feature_importances_, dtype=float)[: len(cols)]
            denom = max(float(np.sum(raw)), 1e-12)
            for name, value in zip(cols, raw, strict=True):
                rows.append({
                    "zone_side": side, "task": task, "model": model_name, "feature": name,
                    "feature_family": "SWING_SUPPLEMENT" if name.startswith("swing_") else "LIQUIDITY_PATH",
                    "importance": float(value), "importance_share": float(value / denom),
                })
    return pd.DataFrame(rows).sort_values(["zone_side", "task", "model", "importance"], ascending=[True, True, True, False]).reset_index(drop=True)
