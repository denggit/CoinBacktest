#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Family-balanced frozen clustering for second-stage C3 swing-low research."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
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

EPS = 1e-12


@dataclass
class FamilyTransform:
    family: str
    feature_columns: list[str]
    medians: pd.Series
    lower_bounds: pd.Series
    upper_bounds: pd.Series
    scaler: RobustScaler
    pca: PCA
    block_weight: float


@dataclass
class FrozenC3Typology:
    transforms: list[FamilyTransform]
    embedding_columns: list[str]
    train_embedding: np.ndarray
    all_embedding: np.ndarray
    kmeans: KMeans
    raw_labels_all: np.ndarray
    labels_all: np.ndarray
    label_map: dict[int, str]
    selected_k: int


def _clean_family(
    train: pd.DataFrame,
    full: pd.DataFrame,
    columns: Sequence[str],
    *,
    minimum_non_null: float = 0.90,
    minimum_unique: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, list[str]]:
    keep: list[str] = []
    for col in columns:
        values = pd.to_numeric(train[col], errors="coerce")
        if float(values.notna().mean()) >= minimum_non_null and int(values.nunique(dropna=True)) >= minimum_unique:
            keep.append(col)
    if not keep:
        raise RuntimeError("No usable features in family")
    train_num = train[keep].apply(pd.to_numeric, errors="coerce")
    full_num = full[keep].apply(pd.to_numeric, errors="coerce")
    medians = train_num.median().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    train_num = train_num.replace([np.inf, -np.inf], np.nan).fillna(medians)
    full_num = full_num.replace([np.inf, -np.inf], np.nan).fillna(medians)
    lower = train_num.quantile(0.005)
    upper = train_num.quantile(0.995)
    train_num = train_num.clip(lower=lower, upper=upper, axis=1)
    full_num = full_num.clip(lower=lower, upper=upper, axis=1)

    # Remove near-constant and highly redundant columns inside each family only.
    variable = [c for c in keep if float(train_num[c].std(ddof=0)) > 1e-10]
    train_num = train_num[variable]
    full_num = full_num[variable]
    medians, lower, upper = medians[variable], lower[variable], upper[variable]
    if len(variable) > 1:
        corr = train_num.corr().abs()
        dropped: set[str] = set()
        for i, col in enumerate(corr.columns):
            if col in dropped:
                continue
            for other in corr.columns[i + 1 :]:
                if other in dropped:
                    continue
                value = corr.at[col, other]
                if np.isfinite(value) and value >= 0.985:
                    dropped.add(other)
        selected = [c for c in variable if c not in dropped]
    else:
        selected = variable
    return train_num[selected], full_num[selected], medians[selected], lower[selected], upper[selected], selected


def _seed_stability(x: np.ndarray, k: int, seeds: Sequence[int]) -> float:
    labels: list[np.ndarray] = []
    for seed in seeds:
        model = KMeans(n_clusters=k, n_init=5, max_iter=300, random_state=int(seed), algorithm="lloyd")
        with threadpool_limits(limits=1):
            labels.append(model.fit_predict(x))
    scores = [adjusted_rand_score(a, b) for a, b in combinations(labels, 2)]
    return float(np.mean(scores)) if scores else float("nan")


def fit_family_balanced_typology(
    features: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    train_mask: pd.Series,
    *,
    k_min: int = 4,
    k_max: int = 10,
    random_state: int = 42,
    family_variance_target: float = 0.85,
    max_components_per_family: int = 5,
) -> tuple[FrozenC3Typology, pd.DataFrame, pd.DataFrame]:
    """Fit transforms and clusters using the development period only.

    Each feature family receives its own robust scaler and PCA block.  Every
    block is divided by sqrt(component_count), so a large family cannot drown
    out smaller but economically distinct families.
    """

    if int(train_mask.sum()) < 300:
        raise RuntimeError(f"Need at least 300 train C3 events, got {int(train_mask.sum())}")
    dictionary = feature_dictionary.set_index("feature")
    transforms: list[FamilyTransform] = []
    train_blocks: list[np.ndarray] = []
    all_blocks: list[np.ndarray] = []
    embedding_columns: list[str] = []
    family_rows: list[dict[str, object]] = []

    families = list(dict.fromkeys(feature_dictionary["family"].astype(str).tolist()))
    for family in families:
        cols = [c for c in feature_dictionary.loc[feature_dictionary["family"] == family, "feature"] if c in features]
        if not cols:
            continue
        try:
            train_clean, full_clean, medians, lower, upper, selected = _clean_family(
                features.loc[train_mask], features, cols
            )
        except RuntimeError:
            continue
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        train_scaled = scaler.fit_transform(train_clean)
        all_scaled = scaler.transform(full_clean)
        max_components = min(max_components_per_family, train_scaled.shape[1], max(1, len(train_clean) - 1))
        probe = PCA(n_components=max_components, random_state=random_state).fit(train_scaled)
        cumulative = np.cumsum(probe.explained_variance_ratio_)
        component_count = int(np.searchsorted(cumulative, family_variance_target) + 1)
        component_count = max(1, min(max_components, component_count))
        pca = PCA(n_components=component_count, whiten=True, random_state=random_state)
        train_block = pca.fit_transform(train_scaled)
        all_block = pca.transform(all_scaled)
        weight = 1.0 / np.sqrt(component_count)
        train_block *= weight
        all_block *= weight
        train_blocks.append(train_block)
        all_blocks.append(all_block)
        embedding_columns.extend([f"{family}_pc{i + 1}" for i in range(component_count)])
        transforms.append(
            FamilyTransform(
                family=family,
                feature_columns=selected,
                medians=medians,
                lower_bounds=lower,
                upper_bounds=upper,
                scaler=scaler,
                pca=pca,
                block_weight=weight,
            )
        )
        family_rows.append(
            {
                "family": family,
                "input_features": len(cols),
                "retained_features": len(selected),
                "pca_components": component_count,
                "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
                "block_weight": weight,
            }
        )

    if len(train_blocks) < 4:
        raise RuntimeError(f"Too few usable feature families: {len(train_blocks)}")
    train_embedding = np.concatenate(train_blocks, axis=1)
    all_embedding = np.concatenate(all_blocks, axis=1)

    rows: list[dict[str, object]] = []
    best_score = -np.inf
    best_model: KMeans | None = None
    valid_max = min(int(k_max), max(int(k_min), int(train_mask.sum()) // 80))
    for k in range(int(k_min), valid_max + 1):
        model = KMeans(n_clusters=k, n_init=20, max_iter=400, random_state=random_state, algorithm="lloyd")
        with threadpool_limits(limits=1):
            labels = model.fit_predict(train_embedding)
        counts = np.bincount(labels, minlength=k)
        min_share = float(counts.min() / len(labels))
        sample_size = min(3000, len(labels))
        silhouette = float(
            silhouette_score(
                train_embedding,
                labels,
                sample_size=sample_size if sample_size < len(labels) else None,
                random_state=random_state,
            )
        )
        stability = _seed_stability(train_embedding, k, [random_state + i for i in range(4)])
        tiny_penalty = 0.45 if min_share < 0.025 else (0.15 if min_share < 0.04 else 0.0)
        complexity_penalty = 0.01 * max(0, k - 6)
        score = silhouette + 0.30 * stability + 0.12 * min_share - tiny_penalty - complexity_penalty
        rows.append(
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
        raise RuntimeError("No valid second-stage clustering model")
    selected_k = int(best_model.n_clusters)
    for row in rows:
        row["selected"] = int(row["k"]) == selected_k

    raw_all = best_model.predict(all_embedding)
    raw_train = raw_all[train_mask.to_numpy()]

    # Deterministic ordering by structural stress and activity, not future labels.
    train_raw = features.loc[train_mask].copy()
    train_raw["_raw"] = raw_train
    severity_features = [
        c for c in (
            "drawdown_from_high_120",
            "drawdown_from_high_240",
            "realized_vol_60",
            "current_notional_intensity",
            "notional_hhi_60",
            "cvd_ratio_60",
        )
        if c in train_raw
    ]
    severity = pd.Series(0.0, index=sorted(train_raw["_raw"].unique()))
    grouped = train_raw.groupby("_raw")
    for col in severity_features:
        med = grouped[col].median()
        center = float(train_raw[col].median())
        scale = float(train_raw[col].quantile(0.75) - train_raw[col].quantile(0.25))
        if not np.isfinite(scale) or abs(scale) <= EPS:
            continue
        z = (med - center) / scale
        if col.startswith("drawdown") or col.startswith("cvd_ratio"):
            z = -z
        severity = severity.add(z.reindex(severity.index).fillna(0.0), fill_value=0.0)
    ordered_raw = [int(x) for x in severity.sort_values(ascending=False).index]
    label_map = {raw: f"C3-{chr(65 + i)}" for i, raw in enumerate(ordered_raw)}
    labels_all = np.asarray([label_map[int(x)] for x in raw_all], dtype=object)

    frozen = FrozenC3Typology(
        transforms=transforms,
        embedding_columns=embedding_columns,
        train_embedding=train_embedding,
        all_embedding=all_embedding,
        kmeans=best_model,
        raw_labels_all=raw_all,
        labels_all=labels_all,
        label_map=label_map,
        selected_k=selected_k,
    )
    return frozen, pd.DataFrame(rows), pd.DataFrame(family_rows)


def build_assignments(frozen: FrozenC3Typology, features: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    cols = [
        "event_id",
        "extreme_time",
        "feature_available_time",
        "extreme_pos",
        "extreme_price",
        "confirmation_time",
        "confirmation_available_time",
        "completion_bars",
        "realized_confirmation_move_pct",
        "parent_cluster_id",
        "parent_distance_to_centroid",
    ]
    out = features[cols].copy()
    out["split"] = np.where(train_mask.to_numpy(), "train", "holdout")
    out["subcluster_id"] = frozen.labels_all
    distances = frozen.kmeans.transform(frozen.all_embedding)
    assigned_raw = frozen.raw_labels_all.astype(int)
    out["distance_to_train_centroid"] = distances[np.arange(len(out)), assigned_raw]
    out["year"] = pd.to_datetime(out["extreme_time"]).dt.year
    return out.sort_values("extreme_time").reset_index(drop=True)


def build_stability(assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = (
        assignments.groupby(["subcluster_id", "split"], as_index=False)
        .agg(
            count=("event_id", "size"),
            median_centroid_distance=("distance_to_train_centroid", "median"),
            p90_centroid_distance=("distance_to_train_centroid", lambda s: float(np.nanquantile(s, 0.90))),
        )
    )
    split["share_within_split"] = split["count"] / split.groupby("split")["count"].transform("sum")
    yearly = (
        assignments.groupby(["year", "subcluster_id"], as_index=False)
        .agg(count=("event_id", "size"), median_centroid_distance=("distance_to_train_centroid", "median"))
    )
    yearly["share_within_year"] = yearly["count"] / yearly.groupby("year")["count"].transform("sum")
    return split, yearly


def build_profiles(
    features: pd.DataFrame,
    dictionary: pd.DataFrame,
    assignments: pd.DataFrame,
    train_mask: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_cols = dictionary["feature"].tolist()
    train = features.loc[train_mask, ["event_id", *feature_cols]].merge(
        assignments[["event_id", "subcluster_id"]], on="event_id", how="left"
    )
    overall = train[feature_cols].median()
    iqr = train[feature_cols].quantile(0.75) - train[feature_cols].quantile(0.25)
    iqr = iqr.replace(0.0, np.nan)
    medians = train.groupby("subcluster_id")[feature_cols].median()
    robust_diff = medians.subtract(overall, axis=1).divide(iqr, axis=1)
    dictionary_index = dictionary.set_index("feature")
    descriptor_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for cluster_id in robust_diff.index:
        ranked = robust_diff.loc[cluster_id].dropna().sort_values(key=lambda s: s.abs(), ascending=False).head(10)
        phrases: list[str] = []
        for rank, (feature, value) in enumerate(ranked.items(), start=1):
            label = str(dictionary_index.at[feature, "label"]) if feature in dictionary_index.index else feature
            direction = "高" if value > 0 else "低"
            phrases.append(f"{direction}{label}")
            descriptor_rows.append(
                {
                    "subcluster_id": cluster_id,
                    "rank": rank,
                    "feature": feature,
                    "family": dictionary_index.at[feature, "family"] if feature in dictionary_index.index else "",
                    "label": label,
                    "median_iqr_z_vs_parent_c3": float(value),
                    "direction": direction,
                }
            )
        count = int((assignments.loc[train_mask.to_numpy(), "subcluster_id"] == cluster_id).sum())
        summary_rows.append(
            {
                "subcluster_id": cluster_id,
                "train_count": count,
                "train_share": float(count / int(train_mask.sum())),
                "descriptor": "；".join(phrases[:6]),
            }
        )
    long = (
        robust_diff.reset_index()
        .melt(id_vars="subcluster_id", var_name="feature", value_name="median_iqr_z_vs_parent_c3")
        .merge(dictionary, on="feature", how="left")
    )
    return pd.DataFrame(summary_rows), pd.DataFrame(descriptor_rows), long


def build_rule_cards(
    features: pd.DataFrame,
    dictionary: pd.DataFrame,
    assignments: pd.DataFrame,
    train_mask: pd.Series,
    *,
    random_state: int = 42,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    feature_cols = dictionary["feature"].tolist()
    x = features[feature_cols].apply(pd.to_numeric, errors="coerce")
    train_median = x.loc[train_mask].median().fillna(0.0)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(train_median)
    # Limit rules to the most globally discriminative features for readability.
    cluster_labels = assignments["subcluster_id"].astype(str)
    between = x.loc[train_mask].groupby(cluster_labels.loc[train_mask.to_numpy()].to_numpy()).median().var(axis=0)
    within = x.loc[train_mask].var(axis=0).replace(0.0, np.nan)
    score = (between / within).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    rule_features = score.head(30).index.tolist()
    if len(rule_features) < 5:
        rule_features = feature_cols[: min(30, len(feature_cols))]
    x_rule = x[rule_features]
    text_parts: list[str] = []
    fidelity_rows: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []
    min_leaf = max(25, int(train_mask.sum() * 0.025))
    for cluster_id in sorted(cluster_labels.unique()):
        y = (cluster_labels == cluster_id).astype(int)
        tree = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=min_leaf,
            class_weight="balanced",
            random_state=random_state,
        )
        tree.fit(x_rule.loc[train_mask], y.loc[train_mask])
        text_parts.append(f"\n## {cluster_id}\n" + export_text(tree, feature_names=rule_features, decimals=4))
        for split_name, mask in (("train", train_mask), ("holdout", ~train_mask)):
            if int(mask.sum()) == 0:
                continue
            pred = tree.predict(x_rule.loc[mask])
            true = y.loc[mask]
            fidelity_rows.append(
                {
                    "subcluster_id": cluster_id,
                    "split": split_name,
                    "count": int(mask.sum()),
                    "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
                    "precision": float(precision_score(true, pred, zero_division=0)),
                    "recall": float(recall_score(true, pred, zero_division=0)),
                    "f1": float(f1_score(true, pred, zero_division=0)),
                }
            )
        for feature, importance in zip(rule_features, tree.feature_importances_):
            if importance > 0:
                importance_rows.append(
                    {"subcluster_id": cluster_id, "feature": feature, "tree_importance": float(importance)}
                )
    return "\n".join(text_parts).strip() + "\n", pd.DataFrame(fidelity_rows), pd.DataFrame(importance_rows)


def representative_events(assignments: pd.DataFrame, per_cluster: int = 15) -> pd.DataFrame:
    rows = [group.nsmallest(per_cluster, "distance_to_train_centroid") for _, group in assignments.groupby("subcluster_id")]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_post_label_diagnostics(assignments: pd.DataFrame) -> pd.DataFrame:
    return (
        assignments.groupby(["subcluster_id", "split"], as_index=False)
        .agg(
            count=("event_id", "size"),
            median_completion_bars=("completion_bars", "median"),
            p25_completion_bars=("completion_bars", lambda s: float(np.nanquantile(s, 0.25))),
            p75_completion_bars=("completion_bars", lambda s: float(np.nanquantile(s, 0.75))),
            median_confirmation_move_pct=("realized_confirmation_move_pct", "median"),
        )
    )
