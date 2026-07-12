#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train-only unsupervised clustering and interpretable cluster descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import (
        adjusted_rand_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        silhouette_score,
    )
    from sklearn.preprocessing import RobustScaler
    from sklearn.tree import DecisionTreeClassifier, export_text
    from threadpoolctl import threadpool_limits
except ImportError as exc:  # pragma: no cover - local environment dependency
    raise ImportError(
        "Swing-low typology research requires scikit-learn. Install it with: pip install scikit-learn"
    ) from exc


@dataclass
class FrozenTypology:
    feature_columns: list[str]
    medians: pd.Series
    lower_bounds: pd.Series
    upper_bounds: pd.Series
    scaler: RobustScaler
    pca: PCA
    kmeans: KMeans
    label_map: dict[int, str]
    train_scaled_original: pd.DataFrame
    all_scaled_original: pd.DataFrame
    transformed_all: np.ndarray
    labels_raw_all: np.ndarray
    labels_all: np.ndarray
    selected_k: int


def _fit_clean_matrix(
    train: pd.DataFrame,
    all_rows: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    max_missing_ratio: float = 0.20,
    correlation_threshold: float = 0.97,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    train_raw = train[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    all_raw = all_rows[list(feature_columns)].apply(pd.to_numeric, errors="coerce")

    keep = [c for c in feature_columns if float(train_raw[c].isna().mean()) <= max_missing_ratio]
    if len(keep) < 5:
        raise RuntimeError(f"Too few usable causal features after missing-value filter: {len(keep)}")
    train_raw = train_raw[keep]
    all_raw = all_raw[keep]

    medians = train_raw.median(axis=0, skipna=True)
    train_imp = train_raw.fillna(medians)
    all_imp = all_raw.fillna(medians)
    varying = [c for c in keep if float(train_imp[c].std(ddof=0)) > 1e-12]
    train_imp = train_imp[varying]
    all_imp = all_imp[varying]
    medians = medians[varying]

    lower = train_imp.quantile(0.005)
    upper = train_imp.quantile(0.995)
    train_clip = train_imp.clip(lower=lower, upper=upper, axis=1)
    all_clip = all_imp.clip(lower=lower, upper=upper, axis=1)

    corr = train_clip.corr().abs()
    dropped: set[str] = set()
    for i, col in enumerate(corr.columns):
        if col in dropped:
            continue
        for other in corr.columns[i + 1 :]:
            if other in dropped:
                continue
            value = corr.at[col, other]
            if np.isfinite(value) and value >= correlation_threshold:
                dropped.add(other)
    selected = [c for c in train_clip.columns if c not in dropped]
    if len(selected) < 5:
        raise RuntimeError(f"Too few features after correlation filter: {len(selected)}")
    return train_clip[selected], all_clip[selected], medians[selected], lower[selected], upper[selected], selected


def _stability_score(x: np.ndarray, k: int, seeds: Sequence[int]) -> float:
    label_sets: list[np.ndarray] = []
    for seed in seeds:
        model = KMeans(n_clusters=k, n_init=3, max_iter=300, random_state=int(seed), algorithm="lloyd")
        with threadpool_limits(limits=1):
            label_sets.append(model.fit_predict(x))
    if len(label_sets) < 2:
        return float("nan")
    scores = [adjusted_rand_score(a, b) for a, b in combinations(label_sets, 2)]
    return float(np.mean(scores)) if scores else float("nan")


def _stress_order(profile: pd.DataFrame) -> list[int]:
    candidates = {
        "drawdown_from_high_60": -1.0,
        "delta_ratio_30": -0.8,
        "current_notional_vs_prev30_median": 0.4,
        "current_range_pct": 0.4,
        "negative_delta_share_30": 0.5,
    }
    score = pd.Series(0.0, index=profile.index)
    used = 0
    for feature, weight in candidates.items():
        if feature in profile.columns:
            score = score + profile[feature].fillna(0.0) * weight
            used += 1
    if used == 0:
        return sorted(int(x) for x in profile.index)
    return [int(x) for x in score.sort_values(ascending=False).index]


def fit_frozen_typology(
    feature_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    train_mask: pd.Series,
    *,
    k_min: int,
    k_max: int,
    random_state: int,
    silhouette_sample_size: int = 3000,
) -> tuple[FrozenTypology, pd.DataFrame]:
    """Fit all preprocessing and clustering on the development period only."""

    train = feature_frame.loc[train_mask].copy()
    if len(train) < 200:
        raise RuntimeError(f"Need at least 200 development events, got {len(train)}")
    train_clean, all_clean, medians, lower, upper, selected = _fit_clean_matrix(train, feature_frame, feature_columns)

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    train_scaled_np = scaler.fit_transform(train_clean)
    all_scaled_np = scaler.transform(all_clean)
    train_scaled = pd.DataFrame(train_scaled_np, index=train.index, columns=selected)
    all_scaled = pd.DataFrame(all_scaled_np, index=feature_frame.index, columns=selected)

    max_pca = min(15, train_scaled.shape[1], max(2, len(train_scaled) - 1))
    pca_probe = PCA(n_components=max_pca, random_state=random_state).fit(train_scaled_np)
    cumulative = np.cumsum(pca_probe.explained_variance_ratio_)
    component_count = int(np.searchsorted(cumulative, 0.88) + 1)
    component_count = max(3, min(max_pca, component_count))
    pca = PCA(n_components=component_count, random_state=random_state)
    train_pca = pca.fit_transform(train_scaled_np)
    all_pca = pca.transform(all_scaled_np)

    selection_rows: list[dict[str, object]] = []
    best_score = -np.inf
    best_model: KMeans | None = None
    valid_k_max = min(int(k_max), max(2, len(train) // 50))
    for k in range(max(2, int(k_min)), valid_k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=random_state, algorithm="lloyd")
        with threadpool_limits(limits=1):
            labels = model.fit_predict(train_pca)
        counts = np.bincount(labels, minlength=k)
        min_share = float(counts.min() / len(labels))
        sample_size = min(int(silhouette_sample_size), len(labels))
        silhouette = float(
            silhouette_score(
                train_pca,
                labels,
                sample_size=sample_size if sample_size < len(labels) else None,
                random_state=random_state,
            )
        )
        stability = _stability_score(train_pca, k, [random_state + i for i in range(3)])
        penalty = 0.35 if min_share < 0.04 else 0.0
        score = silhouette + 0.20 * stability + 0.10 * min_share - penalty
        selection_rows.append(
            {
                "k": k,
                "silhouette_train": silhouette,
                "seed_stability_ari": stability,
                "minimum_cluster_share_train": min_share,
                "selection_score": score,
                "selected": False,
            }
        )
        if score > best_score:
            best_score = score
            best_model = model

    if best_model is None:
        raise RuntimeError("No clustering candidate was fitted")
    selected_k = int(best_model.n_clusters)
    for row in selection_rows:
        row["selected"] = int(row["k"]) == selected_k

    raw_all = best_model.predict(all_pca)
    raw_train = raw_all[train_mask.to_numpy()]
    profile_raw = train_scaled.assign(_cluster=raw_train).groupby("_cluster").median()
    ordered = _stress_order(profile_raw)
    label_map = {raw: f"C{i + 1}" for i, raw in enumerate(ordered)}
    labels_all = np.asarray([label_map[int(x)] for x in raw_all], dtype=object)

    frozen = FrozenTypology(
        feature_columns=selected,
        medians=medians,
        lower_bounds=lower,
        upper_bounds=upper,
        scaler=scaler,
        pca=pca,
        kmeans=best_model,
        label_map=label_map,
        train_scaled_original=train_scaled,
        all_scaled_original=all_scaled,
        transformed_all=all_pca,
        labels_raw_all=raw_all,
        labels_all=labels_all,
        selected_k=selected_k,
    )
    return frozen, pd.DataFrame(selection_rows)


def build_cluster_profiles(
    frozen: FrozenTypology,
    feature_frame: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    train_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.Series(frozen.labels_all, index=feature_frame.index, name="cluster_id")
    train_scaled = frozen.all_scaled_original.loc[train_mask].copy()
    train_scaled["cluster_id"] = labels.loc[train_mask]
    profile = train_scaled.groupby("cluster_id").median().sort_index()
    overall = frozen.all_scaled_original.loc[train_mask].median()
    differential = profile.subtract(overall, axis=1)

    dictionary = feature_dictionary.set_index("feature") if not feature_dictionary.empty else pd.DataFrame()
    descriptor_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for cluster_id in profile.index:
        diffs = differential.loc[cluster_id].dropna().sort_values(key=lambda s: s.abs(), ascending=False)
        top = diffs.head(8)
        phrases: list[str] = []
        for rank, (feature, value) in enumerate(top.items(), start=1):
            label = str(dictionary.at[feature, "label"]) if feature in dictionary.index else feature
            direction = "高" if value > 0 else "低"
            phrases.append(f"{direction}{label}")
            descriptor_rows.append(
                {
                    "cluster_id": cluster_id,
                    "rank": rank,
                    "feature": feature,
                    "label": label,
                    "median_robust_z_vs_overall": float(value),
                    "direction": direction,
                }
            )
        mask = labels.loc[train_mask] == cluster_id
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "train_count": int(mask.sum()),
                "train_share": float(mask.mean()),
                "descriptor": "；".join(phrases[:5]),
            }
        )

    profile_long = (
        differential.reset_index()
        .melt(id_vars="cluster_id", var_name="feature", value_name="median_robust_z_vs_overall")
        .merge(feature_dictionary, on="feature", how="left")
        .sort_values(["cluster_id", "median_robust_z_vs_overall"], ascending=[True, False])
    )
    return pd.DataFrame(summary_rows), pd.DataFrame(descriptor_rows), profile_long


def build_rule_cards(
    frozen: FrozenTypology,
    feature_frame: pd.DataFrame,
    train_mask: pd.Series,
    *,
    random_state: int,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    x = frozen.all_scaled_original
    y = pd.Series(frozen.labels_all, index=feature_frame.index)
    min_leaf = max(30, int(train_mask.sum() * 0.04))
    text_parts: list[str] = []
    fidelity_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    for cluster_id in sorted(y.unique()):
        y_binary = (y == cluster_id).astype(int)
        tree = DecisionTreeClassifier(
            max_depth=2,
            min_samples_leaf=min_leaf,
            class_weight="balanced",
            random_state=random_state,
        )
        tree.fit(x.loc[train_mask], y_binary.loc[train_mask])
        text_parts.append(f"\n## {cluster_id}\n" + export_text(tree, feature_names=list(x.columns), decimals=3))
        for split_name, mask in (("train", train_mask), ("holdout", ~train_mask)):
            if int(mask.sum()) == 0:
                continue
            pred = tree.predict(x.loc[mask])
            true = y_binary.loc[mask]
            fidelity_rows.append(
                {
                    "cluster_id": cluster_id,
                    "split": split_name,
                    "count": int(mask.sum()),
                    "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
                    "precision": float(precision_score(true, pred, zero_division=0)),
                    "recall": float(recall_score(true, pred, zero_division=0)),
                    "f1": float(f1_score(true, pred, zero_division=0)),
                }
            )
        for feature, importance in zip(x.columns, tree.feature_importances_):
            if importance > 0:
                importance_rows.append(
                    {"cluster_id": cluster_id, "feature": feature, "tree_importance": float(importance)}
                )
    return "\n".join(text_parts).strip() + "\n", pd.DataFrame(fidelity_rows), pd.DataFrame(importance_rows)


def build_assignments(
    frozen: FrozenTypology,
    feature_frame: pd.DataFrame,
    train_mask: pd.Series,
) -> pd.DataFrame:
    out = feature_frame[
        [
            "event_id",
            "extreme_time",
            "feature_available_time",
            "extreme_pos",
            "extreme_price",
            "confirmation_time",
            "confirmation_available_time",
            "completion_bars",
            "realized_confirmation_move_pct",
        ]
    ].copy()
    out["split"] = np.where(train_mask.to_numpy(), "train", "holdout")
    out["cluster_id"] = frozen.labels_all
    raw_distances = frozen.kmeans.transform(frozen.transformed_all)
    assigned_raw = frozen.labels_raw_all.astype(int)
    out["distance_to_train_centroid"] = raw_distances[np.arange(len(out)), assigned_raw]
    out["year"] = pd.to_datetime(out["extreme_time"]).dt.year
    return out.sort_values("extreme_time").reset_index(drop=True)


def build_cluster_stability(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        assignments.groupby(["cluster_id", "split"], as_index=False)
        .agg(
            count=("event_id", "size"),
            median_centroid_distance=("distance_to_train_centroid", "median"),
            p90_centroid_distance=("distance_to_train_centroid", lambda s: float(np.nanquantile(s, 0.90))),
        )
    )
    summary["share_within_split"] = summary["count"] / summary.groupby("split")["count"].transform("sum")

    yearly = (
        assignments.groupby(["year", "cluster_id"], as_index=False)
        .agg(
            count=("event_id", "size"),
            median_centroid_distance=("distance_to_train_centroid", "median"),
        )
    )
    yearly["share_within_year"] = yearly["count"] / yearly.groupby("year")["count"].transform("sum")
    return summary, yearly


def build_post_label_diagnostics(assignments: pd.DataFrame) -> pd.DataFrame:
    """Future confirmation metrics are reported only after clustering."""
    return (
        assignments.groupby(["cluster_id", "split"], as_index=False)
        .agg(
            count=("event_id", "size"),
            median_completion_bars=("completion_bars", "median"),
            p25_completion_bars=("completion_bars", lambda s: float(np.nanquantile(s, 0.25))),
            p75_completion_bars=("completion_bars", lambda s: float(np.nanquantile(s, 0.75))),
            median_confirmation_move_pct=("realized_confirmation_move_pct", "median"),
        )
    )


def representative_events(
    assignments: pd.DataFrame,
    *,
    per_cluster: int = 12,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for cluster_id, group in assignments.groupby("cluster_id", sort=True):
        rows.append(group.nsmallest(per_cluster, "distance_to_train_centroid"))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
