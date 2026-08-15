#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-sectional first-touch liquidity ranking models for R02.2."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from lightgbm import LGBMRanker
except ImportError:  # pragma: no cover
    LGBMRanker = None

from src.ai_research.latent_liquidity_pool_forecast.modeling import feature_columns as r02_feature_columns
from .config import FirstTouchLiquidityRankingConfig

_FUTURE_PREFIXES = (
    "touch_", "release_", "favorable_", "continuation_", "time_to_", "sweep_", "reversal_",
    "p_touch", "p_release", "p_favorable", "pred_", "pool_score", "high_strength", "ft_",
    "first_touch", "ranking_",
)
_FUTURE_EXACT = {"primary_touch_label_complete", "model_sample_keep", "sample_weight", "full_lattice_audit_group"}


def feature_columns(frame: pd.DataFrame, *, include_swing: bool) -> tuple[str, ...]:
    cols: list[str] = []
    for name in r02_feature_columns(frame, include_swing=include_swing):
        if name in _FUTURE_EXACT or name.startswith(_FUTURE_PREFIXES):
            continue
        cols.append(name)
    return tuple(cols)


def _ranker(config: FirstTouchLiquidityRankingConfig) -> LGBMRanker:
    if LGBMRanker is None:
        raise RuntimeError("lightgbm is required for R02.2")
    gain = [0]
    for i in range(1, config.rank_relevance_grades):
        gain.append((2 ** i) - 1)
    return LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=gain,
        n_estimators=config.model_n_estimators,
        learning_rate=config.model_learning_rate,
        num_leaves=config.model_num_leaves,
        min_child_samples=config.model_min_child_samples,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=1.0,
        reg_lambda=4.0,
        random_state=config.random_state,
        n_jobs=-1,
        verbosity=-1,
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


def _fit_one(frame: pd.DataFrame, cols: tuple[str, ...], config: FirstTouchLiquidityRankingConfig) -> LGBMRanker:
    work = frame.sort_values(["ranking_group", "zone_distance_bp"], kind="mergesort")
    sizes = work.groupby("ranking_group", sort=False).size().to_numpy(dtype=np.int32)
    if len(sizes) < config.minimum_rank_groups:
        raise RuntimeError(f"R02.2 insufficient rank groups: {len(sizes)}")
    model = _ranker(config)
    model.fit(
        work.loc[:, cols],
        work["ranking_relevance"].to_numpy(dtype=np.int16),
        group=sizes.tolist(),
    )
    return model


@dataclass
class SideRankModel:
    side: str
    path_columns: tuple[str, ...]
    full_columns: tuple[str, ...]
    path_model: LGBMRanker
    full_model: LGBMRanker


@dataclass
class RankModelBundle:
    by_side: dict[str, SideRankModel]


def fit_models(frame: pd.DataFrame, config: FirstTouchLiquidityRankingConfig) -> RankModelBundle:
    by_side: dict[str, SideRankModel] = {}
    for side in ("DOWN", "UP"):
        train = frame.loc[
            frame["period"].astype(str).eq(config.train_period)
            & frame["zone_side"].astype(str).eq(side)
            & frame["ranking_group_eligible"].astype(bool)
            & frame["first_touch_label_complete"].astype(bool)
            & frame["ranking_relevance"].ge(0)
        ].copy()
        train = _cap_groups(train, config.model_train_cap_rows_per_side)
        path = feature_columns(train, include_swing=False)
        full = feature_columns(train, include_swing=True)
        if not path or not full:
            raise RuntimeError(f"R02.2 feature schema empty for {side}")
        by_side[side] = SideRankModel(
            side=side,
            path_columns=path,
            full_columns=full,
            path_model=_fit_one(train, path, config),
            full_model=_fit_one(train, full, config),
        )
    return RankModelBundle(by_side=by_side)


def predict(frame: pd.DataFrame, models: RankModelBundle) -> pd.DataFrame:
    out = frame.copy()
    out["rank_score_path_no_swing"] = np.nan
    out["rank_score_full_with_swing"] = np.nan
    out["rank_score_distance_baseline"] = -pd.to_numeric(out["zone_distance_bp"], errors="coerce")
    for side, bundle in models.by_side.items():
        mask = out["zone_side"].astype(str).eq(side)
        if not mask.any():
            continue
        out.loc[mask, "rank_score_path_no_swing"] = bundle.path_model.predict(out.loc[mask, bundle.path_columns])
        out.loc[mask, "rank_score_full_with_swing"] = bundle.full_model.predict(out.loc[mask, bundle.full_columns])
    return out


def _ndcg(relevance: np.ndarray, score: np.ndarray, k: int) -> float:
    valid = np.isfinite(relevance) & np.isfinite(score)
    rel = np.asarray(relevance[valid], dtype=float)
    scr = np.asarray(score[valid], dtype=float)
    if len(rel) < 2 or float(np.max(rel) - np.min(rel)) <= 1e-12:
        return np.nan
    rel = (rel - np.min(rel)) / max(float(np.max(rel) - np.min(rel)), 1e-12)
    kk = min(int(k), len(rel))
    order = np.argsort(-scr, kind="stable")[:kk]
    ideal = np.argsort(-rel, kind="stable")[:kk]
    discount = np.log2(np.arange(2, kk + 2, dtype=float))
    dcg = float(np.sum(rel[order] / discount))
    idcg = float(np.sum(rel[ideal] / discount))
    return dcg / idcg if idcg > 1e-12 else np.nan


def _within_group_metrics(group: pd.DataFrame, score_col: str, target_col: str) -> dict[str, float]:
    g = group.loc[group["first_touch_label_complete"].astype(bool)].copy()
    if len(g) < 2:
        return {"spearman": np.nan, "ndcg1": np.nan, "ndcg3": np.nan, "pairwise_accuracy": np.nan}
    y = pd.to_numeric(g[target_col], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(g[score_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(s)
    y, s = y[valid], s[valid]
    if len(y) < 2 or float(np.max(y) - np.min(y)) <= 1e-12:
        return {"spearman": np.nan, "ndcg1": np.nan, "ndcg3": np.nan, "pairwise_accuracy": np.nan}
    rho = spearmanr(y, s).statistic if np.nanstd(s) > 1e-12 else np.nan
    diff_y = y[:, None] - y[None, :]
    diff_s = s[:, None] - s[None, :]
    tri = np.triu(np.ones_like(diff_y, dtype=bool), 1) & (np.abs(diff_y) > 1e-12)
    acc = float(np.mean(np.sign(diff_y[tri]) == np.sign(diff_s[tri]))) if tri.any() else np.nan
    return {
        "spearman": float(rho) if np.isfinite(rho) else np.nan,
        "ndcg1": _ndcg(y, s, 1),
        "ndcg3": _ndcg(y, s, 3),
        "pairwise_accuracy": acc,
    }


def ranking_metrics(pred: pd.DataFrame, config: FirstTouchLiquidityRankingConfig) -> pd.DataFrame:
    target = "ranking_target"
    rows: list[dict[str, object]] = []
    score_map = {
        "PATH_NO_SWING": "rank_score_path_no_swing",
        "FULL_WITH_SWING": "rank_score_full_with_swing",
        "DISTANCE_BASELINE": "rank_score_distance_baseline",
    }
    for (period, side), sf in pred.groupby(["period", "zone_side"], sort=True):
        for model_name, score_col in score_map.items():
            group_stats = []
            for _, group in sf.groupby("ranking_group", sort=False):
                values = _within_group_metrics(group, score_col, target)
                if any(np.isfinite(v) for v in values.values()):
                    group_stats.append(values)
            if not group_stats:
                rows.append({"period": period, "zone_side": side, "model": model_name, "rank_groups": 0})
                continue
            stats = pd.DataFrame(group_stats)
            rows.append({
                "period": period,
                "zone_side": side,
                "model": model_name,
                "rank_groups": int(len(stats)),
                "mean_group_spearman": float(stats["spearman"].mean()),
                "median_group_spearman": float(stats["spearman"].median()),
                "mean_ndcg1": float(stats["ndcg1"].mean()),
                "mean_ndcg3": float(stats["ndcg3"].mean()),
                "mean_pairwise_accuracy": float(stats["pairwise_accuracy"].mean()),
            })
    return pd.DataFrame(rows)


def top_zone_summary(pred: pd.DataFrame, config: FirstTouchLiquidityRankingConfig) -> pd.DataFrame:
    target = "ranking_target"
    w = config.primary_label_window_seconds
    score_map = {
        "PATH_NO_SWING": "rank_score_path_no_swing",
        "FULL_WITH_SWING": "rank_score_full_with_swing",
        "DISTANCE_BASELINE": "rank_score_distance_baseline",
    }
    rows: list[dict[str, object]] = []
    for (period, side), sf in pred.groupby(["period", "zone_side"], sort=True):
        complete_touched = sf.loc[sf["first_touch_label_complete"].astype(bool)].copy()
        baseline_density = float(pd.to_numeric(complete_touched[target], errors="coerce").mean()) if not complete_touched.empty else np.nan
        baseline_flow = float(pd.to_numeric(complete_touched.get("ft_notional_ratio_60s"), errors="coerce").mean()) if not complete_touched.empty else np.nan
        for model_name, score_col in score_map.items():
            selected: list[pd.Series] = []
            oracle_hits = []
            touched_rank_density = []
            touched_group_mean = []
            groups_seen = 0
            for _, group in sf.groupby("ranking_group", sort=False):
                groups_seen += 1
                scores = pd.to_numeric(group[score_col], errors="coerce")
                if scores.notna().any():
                    selected.append(group.loc[scores.idxmax()])
                touched = group.loc[group["first_touch_label_complete"].astype(bool)].copy()
                if len(touched) < 2:
                    continue
                ty = pd.to_numeric(touched[target], errors="coerce")
                if ty.notna().sum() < 2 or float(ty.max() - ty.min()) <= 1e-12:
                    continue
                ts = pd.to_numeric(touched[score_col], errors="coerce")
                if ts.notna().any():
                    touched_rank_density.append(float(ty.loc[ts.idxmax()]))
                    touched_group_mean.append(float(ty.mean()))
                all_scores = pd.to_numeric(group[score_col], errors="coerce")
                top3 = set(group.loc[all_scores.nlargest(min(3, len(group))).index, "zone_id"].astype(str))
                oracle_zone = str(touched.loc[ty.idxmax(), "zone_id"])
                oracle_hits.append(oracle_zone in top3)
            sel = pd.DataFrame(selected) if selected else pd.DataFrame()
            touched_sel = sel.loc[sel.get("first_touch_label_complete", pd.Series(False, index=sel.index)).astype(bool)].copy() if not sel.empty else pd.DataFrame()
            mean_density = float(pd.to_numeric(touched_sel.get(target), errors="coerce").mean()) if not touched_sel.empty else np.nan
            mean_flow = float(pd.to_numeric(touched_sel.get("ft_notional_ratio_60s"), errors="coerce").mean()) if not touched_sel.empty else np.nan
            release_rate = float(pd.to_numeric(touched_sel.get(f"ft_release_episode_count_{w}s"), errors="coerce").gt(0).mean()) if not touched_sel.empty else np.nan
            favorable_rate = float(pd.to_numeric(touched_sel.get(f"ft_favorable_episode_count_{w}s"), errors="coerce").gt(0).mean()) if not touched_sel.empty else np.nan
            continuation_rate = float(pd.to_numeric(touched_sel.get(f"ft_continuation_episode_count_{w}s"), errors="coerce").gt(0).mean()) if not touched_sel.empty else np.nan
            rows.append({
                "period": period,
                "zone_side": side,
                "model": model_name,
                "groups": int(groups_seen),
                "top1_touched": int(len(touched_sel)),
                "top1_touch_rate": float(len(touched_sel) / max(len(sel), 1)),
                "top1_mean_density": mean_density,
                "all_touched_mean_density": baseline_density,
                "top1_density_lift": mean_density / baseline_density if np.isfinite(mean_density) and np.isfinite(baseline_density) and baseline_density > 1e-12 else np.nan,
                "top1_mean_flow_ratio_60s": mean_flow,
                "all_touched_mean_flow_ratio_60s": baseline_flow,
                "top1_flow_lift": mean_flow / baseline_flow if np.isfinite(mean_flow) and np.isfinite(baseline_flow) and baseline_flow > 1e-12 else np.nan,
                "top1_release_rate": release_rate,
                "top1_favorable_rate": favorable_rate,
                "top1_continuation_rate": continuation_rate,
                "oracle_strongest_zone_in_top3_rate": float(np.mean(oracle_hits)) if oracle_hits else np.nan,
                "within_touched_top1_density_lift": float(np.mean(touched_rank_density) / np.mean(touched_group_mean)) if touched_rank_density and np.mean(touched_group_mean) > 1e-12 else np.nan,
                "top1_mean_distance_bp": float(pd.to_numeric(sel.get("zone_distance_bp"), errors="coerce").mean()) if not sel.empty else np.nan,
            })
    return pd.DataFrame(rows)


def feature_importance(models: RankModelBundle) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for side, bundle in models.by_side.items():
        for model_name, model, cols in (
            ("PATH_NO_SWING", bundle.path_model, bundle.path_columns),
            ("FULL_WITH_SWING", bundle.full_model, bundle.full_columns),
        ):
            raw = np.asarray(model.feature_importances_, dtype=float)[: len(cols)]
            denom = max(float(np.sum(raw)), 1e-12)
            for name, value in zip(cols, raw, strict=True):
                rows.append({
                    "zone_side": side,
                    "model": model_name,
                    "feature": name,
                    "feature_family": "SWING_SUPPLEMENT" if name.startswith("swing_") else "LIQUIDITY_PATH",
                    "importance": float(value),
                    "importance_share": float(value / denom),
                })
    return pd.DataFrame(rows).sort_values(["zone_side", "model", "importance"], ascending=[True, True, False]).reset_index(drop=True)
