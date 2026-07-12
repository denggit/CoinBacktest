#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Weakly supervised mechanism scoring for causal swing-low typology.

This module deliberately avoids another unconstrained KMeans pass.  Type names
are defined by causal market-mechanism hypotheses, while train-only robust
normalization and empirical score calibration freeze how those hypotheses are
applied to the holdout period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier, export_text

EPS = 1e-12


@dataclass(frozen=True)
class ScoreTerm:
    feature: str
    direction: float = 1.0
    weight: float = 1.0
    rationale: str = ""


@dataclass(frozen=True)
class FrozenScoreModel:
    name: str
    terms: Mapping[str, tuple[ScoreTerm, ...]]
    feature_columns: tuple[str, ...]
    medians: pd.Series
    scales: pd.Series
    lower_bounds: pd.Series
    upper_bounds: pd.Series
    calibrate_percentiles: bool
    calibration_values: Mapping[str, np.ndarray]
    ambiguity_margin_threshold: float
    labels: tuple[str, ...]

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        numeric = features.reindex(columns=self.feature_columns).apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(self.medians)
        numeric = numeric.clip(lower=self.lower_bounds, upper=self.upper_bounds, axis=1)
        z = numeric.subtract(self.medians, axis=1).divide(self.scales, axis=1).clip(-6.0, 6.0)
        raw = _raw_scores(z, self.terms)
        calibrated = _calibrate_scores(raw, self.calibration_values) if self.calibrate_percentiles else raw
        probabilities = _softmax(calibrated)
        values = calibrated.to_numpy(dtype=float)
        order = np.argsort(values, axis=1)
        best_idx = order[:, -1]
        second_idx = order[:, -2] if values.shape[1] > 1 else best_idx
        labels = np.asarray(self.labels, dtype=object)
        best = labels[best_idx]
        second = labels[second_idx]
        top = values[np.arange(len(values)), best_idx]
        second_score = values[np.arange(len(values)), second_idx]
        margin = top - second_score
        out = pd.DataFrame(index=features.index)
        for label in self.labels:
            out[f"score_{label}"] = calibrated[label].to_numpy()
            out[f"probability_{label}"] = probabilities[label].to_numpy()
        out["primary_type"] = best
        out["secondary_type"] = second
        out["top_score"] = top
        out["score_margin"] = margin
        out["confidence"] = probabilities.to_numpy()[np.arange(len(values)), best_idx]
        out["ambiguous"] = margin < float(self.ambiguity_margin_threshold)
        return out


BROAD_MECHANISM_TERMS: dict[str, tuple[ScoreTerm, ...]] = {
    "shock": (
        ScoreTerm("current_notional_intensity", 1, 1.0, "低点成交额爆发"),
        ScoreTerm("current_trades_intensity", 1, 1.0, "低点成交笔数爆发"),
        ScoreTerm("current_range_pct", 1, 0.9, "当前振幅扩张"),
        ScoreTerm("realized_vol_15", 1, 0.7, "短周期波动冲击"),
        ScoreTerm("activity_burst_share_30", 1, 0.7, "成交爆发集中"),
        ScoreTerm("notional_hhi_30", 1, 0.6, "成交集中于少数K线"),
        ScoreTerm("close_return_15", -1, 0.6, "末段快速下跌"),
        ScoreTerm("price_decline_acceleration", 1, 0.5, "末段下跌加速"),
    ),
    "trend": (
        ScoreTerm("price_trend_slope_240", -1, 1.0, "长周期价格趋势向下"),
        ScoreTerm("price_trend_r2_240", 1, 0.8, "价格趋势持续稳定"),
        ScoreTerm("cvd_slope_240", -1, 0.8, "累计CVD持续向下"),
        ScoreTerm("cvd_r2_240", 1, 0.5, "CVD趋势持续稳定"),
        ScoreTerm("down_bar_share_240", 1, 0.7, "下跌K线占比高"),
        ScoreTerm("path_efficiency_240", 1, 0.7, "价格路径方向性强"),
        ScoreTerm("drawdown_from_high_240", -1, 0.7, "长周期回撤深"),
        ScoreTerm("price_decline_uniformity", 1, 0.5, "下跌路径均匀"),
        ScoreTerm("large_sell_phase_persistence", 1, 0.4, "大单卖压跨阶段持续"),
    ),
    "base": (
        ScoreTerm("prior_low_test_count_10bp_120", 1, 0.8, "相近低点反复测试"),
        ScoreTerm("near_floor_dwell_share_25bp_240", 1, 0.8, "长期停留底部附近"),
        ScoreTerm("support_prior_test_count", 1, 1.0, "事件级支撑测试次数"),
        ScoreTerm("support_low_dispersion_bp", -1, 0.7, "测试低点集中"),
        ScoreTerm("negative_delta_no_new_low_share", 1, 0.7, "负Delta但价格不创新低"),
        ScoreTerm("absorption_observable", 1, 0.8, "卖压吸收"),
        ScoreTerm("repeated_support_observable", 1, 0.7, "重复支撑测试"),
        ScoreTerm("direction_change_rate_120", 1, 0.4, "底部来回震荡"),
        ScoreTerm("range_compression", 1, 0.4, "振幅逐步压缩"),
    ),
}

TREND_ARCHETYPE_TERMS: dict[str, tuple[ScoreTerm, ...]] = {
    "T1_uniform_decline": (
        ScoreTerm("price_decline_uniformity", 1, 1.0, "阶段跌速均匀"),
        ScoreTerm("price_trend_r2_240", 1, 0.9, "长周期价格线性趋势清晰"),
        ScoreTerm("path_efficiency_240", 1, 0.8, "路径方向性强"),
        ScoreTerm("price_phase_direction_changes", -1, 0.7, "阶段方向切换少"),
        ScoreTerm("price_decline_acceleration", -1, 0.6, "不是末段突然加速"),
        ScoreTerm("down_bar_share_240", 1, 0.5, "持续下跌K线占比高"),
    ),
    "T2_staged_acceleration": (
        ScoreTerm("price_decline_acceleration", 1, 1.0, "末段下跌速度高于前段"),
        ScoreTerm("close_return_30", -1, 0.9, "最后30分钟下跌明显"),
        ScoreTerm("activity_acceleration_30", 1, 0.7, "末段成交活动加速"),
        ScoreTerm("delta_acceleration_30", -1, 0.7, "末段主动卖压增强"),
        ScoreTerm("phase_price_10", 1, 0.5, "末段前价格仍明显高于最终低点"),
        ScoreTerm("phase_price_11", 1, 0.5, "最后阶段快速靠近低点"),
    ),
    "T3_sync_persistent_selling": (
        ScoreTerm("price_cvd_path_sync", 1, 1.0, "价格与CVD同步走弱"),
        ScoreTerm("return_delta_correlation_240", 1, 0.8, "收益与Delta同向"),
        ScoreTerm("large_sell_phase_persistence", 1, 0.9, "大单卖压持续"),
        ScoreTerm("large_cvd_ratio_240", -1, 0.8, "大单累计净卖出"),
        ScoreTerm("cvd_r2_240", 1, 0.6, "CVD趋势稳定"),
        ScoreTerm("delta_sign_change_rate_120", -1, 0.5, "订单流方向切换少"),
    ),
    "T4_price_cvd_divergence": (
        ScoreTerm("price_cvd_path_divergence", 1, 1.0, "CVD弱于价格"),
        ScoreTerm("price_cvd_dislocation_240", 1, 0.9, "价格与累计Delta错位"),
        ScoreTerm("negative_delta_no_new_low_share", 1, 0.8, "负Delta未推动新低"),
        ScoreTerm("sell_impact_decay", 1, 0.8, "卖压价格冲击下降"),
        ScoreTerm("sell_pressure_decay", 1, 0.6, "主动卖压开始衰减"),
        ScoreTerm("negative_delta_no_down_share_120", 1, 0.5, "负Delta时价格抗跌"),
    ),
    "T5_temporary_trend_pause": (
        ScoreTerm("rebound_attempt_count_60", 1, 0.9, "趋势中出现多次短反弹"),
        ScoreTerm("max_rebound_before_low_60", 1, 0.8, "低点前存在明显反弹尝试"),
        ScoreTerm("direction_change_rate_60", 1, 0.7, "短周期路径来回切换"),
        ScoreTerm("price_trend_r2_240", 1, 0.5, "仍处于长周期趋势"),
        ScoreTerm("price_trend_r2_30", -1, 0.5, "短周期趋势不再顺畅"),
        ScoreTerm("sell_pressure_decay", 1, 0.6, "卖压阶段性衰减"),
    ),
}

BASE_ARCHETYPE_TERMS: dict[str, tuple[ScoreTerm, ...]] = {
    "B1_absorption": (
        ScoreTerm("absorption_observable", 1, 1.0, "综合吸收观测"),
        ScoreTerm("negative_delta_no_new_low_share", 1, 0.9, "负Delta未推动新低"),
        ScoreTerm("test_sell_impact_decay_slope", 1, 0.8, "多次测试中卖压冲击下降"),
        ScoreTerm("price_cvd_path_divergence", 1, 0.7, "CVD继续弱而价格抗跌"),
        ScoreTerm("sell_flow_without_new_low_share_120", 1, 0.6, "卖流持续但不创新低"),
    ),
    "B2_compression": (
        ScoreTerm("compression_observable", 1, 1.0, "综合压缩观测"),
        ScoreTerm("range_compression", 1, 0.9, "价格振幅收缩"),
        ScoreTerm("activity_compression", 1, 0.8, "成交活动收缩"),
        ScoreTerm("support_low_dispersion_bp", -1, 0.8, "测试低点聚集"),
        ScoreTerm("realized_vol_30", -1, 0.6, "末段波动较低"),
        ScoreTerm("near_floor_dwell_share_25bp_120", 1, 0.5, "长期贴近底部"),
    ),
    "B3_spring_false_breakdown": (
        ScoreTerm("spring_observable", 1, 1.0, "假跌破并收复"),
        ScoreTerm("final_support_break_bp", 1, 0.9, "最终跌破旧支撑"),
        ScoreTerm("final_close_reclaim_bp", 1, 0.9, "收盘重新站回支撑"),
        ScoreTerm("current_lower_wick_share", 1, 0.7, "低点K线下影明显"),
        ScoreTerm("current_close_position", 1, 0.6, "低点K线收盘位置较高"),
    ),
    "B4_repeated_support_test": (
        ScoreTerm("repeated_support_observable", 1, 1.0, "综合重复测试观测"),
        ScoreTerm("support_prior_test_count", 1, 1.0, "历史测试次数多"),
        ScoreTerm("support_low_dispersion_bp", -1, 0.9, "测试位置集中"),
        ScoreTerm("support_test_interval_cv", -1, 0.6, "测试节奏稳定"),
        ScoreTerm("near_floor_dwell_share_25bp_240", 1, 0.7, "价格长时间停留支撑附近"),
    ),
    "B5_slow_accumulation": (
        ScoreTerm("slow_accumulation_observable", 1, 1.0, "综合缓慢吸筹观测"),
        ScoreTerm("test_rebound_strengthening_slope", 1, 0.8, "每次测试后反弹增强"),
        ScoreTerm("test_sell_pressure_decay_slope", 1, 0.8, "每次测试卖压衰减"),
        ScoreTerm("test_large_sell_decay_slope", 1, 0.7, "大单卖压衰减"),
        ScoreTerm("test_activity_u_shape", 1, 0.7, "成交先缩后放"),
        ScoreTerm("delta_acceleration_120", 1, 0.5, "订单流逐步改善"),
    ),
}


def terms_to_frame(model_name: str, terms: Mapping[str, Sequence[ScoreTerm]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, label_terms in terms.items():
        for rank, term in enumerate(label_terms, start=1):
            rows.append(
                {
                    "model": model_name,
                    "type_id": label,
                    "rank": rank,
                    "feature": term.feature,
                    "direction": term.direction,
                    "weight": term.weight,
                    "rationale": term.rationale,
                }
            )
    return pd.DataFrame(rows)


def _fit_normalizer(features: pd.DataFrame, train_mask: pd.Series, feature_columns: Sequence[str]) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    train = features.loc[train_mask, list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    train = train.replace([np.inf, -np.inf], np.nan)
    medians = train.median().fillna(0.0)
    filled = train.fillna(medians)
    lower = filled.quantile(0.005)
    upper = filled.quantile(0.995)
    clipped = filled.clip(lower=lower, upper=upper, axis=1)
    scales = (clipped.quantile(0.75) - clipped.quantile(0.25)).replace(0.0, np.nan)
    fallback = clipped.std(ddof=0).replace(0.0, np.nan)
    scales = scales.fillna(fallback).fillna(1.0)
    return medians, scales, lower, upper


def _normalised_frame(
    features: pd.DataFrame,
    feature_columns: Sequence[str],
    medians: pd.Series,
    scales: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
) -> pd.DataFrame:
    numeric = features.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(medians)
    numeric = numeric.clip(lower=lower, upper=upper, axis=1)
    return numeric.subtract(medians, axis=1).divide(scales, axis=1).clip(-6.0, 6.0)


def _raw_scores(z: pd.DataFrame, terms: Mapping[str, Sequence[ScoreTerm]]) -> pd.DataFrame:
    scores: dict[str, pd.Series] = {}
    for label, label_terms in terms.items():
        numerator = pd.Series(0.0, index=z.index)
        denominator = 0.0
        for term in label_terms:
            if term.feature not in z.columns:
                continue
            numerator = numerator + z[term.feature] * float(term.direction) * float(term.weight)
            denominator += abs(float(term.weight))
        if denominator <= EPS:
            raise RuntimeError(f"No usable score terms for {label}")
        scores[label] = numerator / denominator
    return pd.DataFrame(scores, index=z.index)


def _empirical_percentile(values: np.ndarray, sorted_train: np.ndarray) -> np.ndarray:
    train = np.asarray(sorted_train, dtype=float)
    train = train[np.isfinite(train)]
    if train.size == 0:
        return np.full(len(values), 0.5, dtype=float)
    positions = np.searchsorted(train, values, side="right")
    return (positions + 0.5) / (len(train) + 1.0)


def _calibrate_scores(raw: pd.DataFrame, calibration_values: Mapping[str, np.ndarray]) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    for label in raw.columns:
        out[label] = _empirical_percentile(raw[label].to_numpy(dtype=float), calibration_values[label])
    return out


def _softmax(scores: pd.DataFrame) -> pd.DataFrame:
    values = scores.to_numpy(dtype=float)
    values = values - np.nanmax(values, axis=1, keepdims=True)
    exp = np.exp(np.clip(values, -30.0, 30.0))
    den = np.sum(exp, axis=1, keepdims=True)
    den = np.where(den <= EPS, 1.0, den)
    return pd.DataFrame(exp / den, index=scores.index, columns=scores.columns)


def fit_score_model(
    features: pd.DataFrame,
    train_mask: pd.Series,
    terms: Mapping[str, Sequence[ScoreTerm]],
    *,
    name: str,
    calibrate_percentiles: bool,
    ambiguity_quantile: float = 0.20,
) -> FrozenScoreModel:
    """Fit robust normalization and score calibration on train rows only."""

    if int(train_mask.sum()) < 100:
        raise RuntimeError(f"{name} needs at least 100 train rows, got {int(train_mask.sum())}")
    feature_columns = tuple(dict.fromkeys(term.feature for label_terms in terms.values() for term in label_terms))
    missing = [column for column in feature_columns if column not in features.columns]
    if missing:
        raise RuntimeError(f"{name} missing required causal features: {missing}")
    medians, scales, lower, upper = _fit_normalizer(features, train_mask, feature_columns)
    z = _normalised_frame(features, feature_columns, medians, scales, lower, upper)
    raw = _raw_scores(z, terms)
    calibration: dict[str, np.ndarray] = {}
    for label in raw.columns:
        calibration[label] = np.sort(raw.loc[train_mask, label].dropna().to_numpy(dtype=float))
    calibrated = _calibrate_scores(raw, calibration) if calibrate_percentiles else raw
    values = calibrated.loc[train_mask].to_numpy(dtype=float)
    ordered = np.sort(values, axis=1)
    margins = ordered[:, -1] - ordered[:, -2] if values.shape[1] > 1 else np.ones(len(values))
    threshold = float(np.nanquantile(margins, float(ambiguity_quantile)))
    return FrozenScoreModel(
        name=name,
        terms={key: tuple(value) for key, value in terms.items()},
        feature_columns=feature_columns,
        medians=medians,
        scales=scales,
        lower_bounds=lower,
        upper_bounds=upper,
        calibrate_percentiles=bool(calibrate_percentiles),
        calibration_values=calibration,
        ambiguity_margin_threshold=threshold,
        labels=tuple(terms.keys()),
    )


def build_bootstrap_stability(
    features: pd.DataFrame,
    train_mask: pd.Series,
    base_model: FrozenScoreModel,
    *,
    seeds: Sequence[int] = (41, 42, 43, 44, 45),
) -> pd.DataFrame:
    """Refit train-only calibration on bootstrap samples and compare assignments."""

    base = base_model.transform(features)["primary_type"].astype(str)
    train_indices = np.flatnonzero(train_mask.to_numpy())
    rows: list[dict[str, object]] = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        sampled = rng.choice(train_indices, size=len(train_indices), replace=True)
        # A frequency-weighted bootstrap is represented by repeating sampled rows.
        boot_features = features.iloc[sampled].reset_index(drop=True)
        boot_train = pd.Series(True, index=boot_features.index)
        model = fit_score_model(
            boot_features,
            boot_train,
            base_model.terms,
            name=base_model.name,
            calibrate_percentiles=base_model.calibrate_percentiles,
            ambiguity_quantile=0.20,
        )
        pred = model.transform(features)["primary_type"].astype(str)
        for split_name, mask in (("train", train_mask), ("holdout", ~train_mask)):
            if int(mask.sum()) == 0:
                continue
            rows.append(
                {
                    "model": base_model.name,
                    "seed": int(seed),
                    "split": split_name,
                    "count": int(mask.sum()),
                    "adjusted_rand_index_vs_primary": float(adjusted_rand_score(base.loc[mask], pred.loc[mask])),
                    "exact_assignment_rate_vs_primary": float((base.loc[mask].to_numpy() == pred.loc[mask].to_numpy()).mean()),
                }
            )
    return pd.DataFrame(rows)


def build_type_summary(assignments: pd.DataFrame, *, type_column: str, confidence_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = (
        assignments.groupby([type_column, "split"], as_index=False)
        .agg(
            count=("event_id", "size"),
            median_confidence=(confidence_column, "median"),
            ambiguity_share=("ambiguous", "mean"),
        )
    )
    split["share_within_split"] = split["count"] / split.groupby("split")["count"].transform("sum")
    yearly = assignments.copy()
    yearly["year"] = pd.to_datetime(yearly["extreme_time"]).dt.year
    yearly = yearly.groupby(["year", type_column], as_index=False).agg(count=("event_id", "size"))
    yearly["share_within_year"] = yearly["count"] / yearly.groupby("year")["count"].transform("sum")
    return split, yearly


def representative_events(
    assignments: pd.DataFrame,
    *,
    type_column: str,
    per_type: int = 12,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for _, group in assignments.groupby(type_column, sort=True):
        ordered = group.sort_values(["ambiguous", "score_margin", "confidence"], ascending=[True, False, False])
        rows.append(ordered.head(int(per_type)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_feature_descriptors(
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    feature_dictionary: pd.DataFrame,
    train_mask: pd.Series,
    *,
    type_column: str,
    top_n: int = 12,
) -> pd.DataFrame:
    dictionary = feature_dictionary.drop_duplicates("feature").set_index("feature")
    feature_columns = [column for column in dictionary.index if column in features.columns]
    train = features.loc[train_mask, ["event_id", *feature_columns]].merge(
        assignments[["event_id", type_column]], on="event_id", how="inner"
    )
    overall = train[feature_columns].median()
    scale = (train[feature_columns].quantile(0.75) - train[feature_columns].quantile(0.25)).replace(0.0, np.nan)
    medians = train.groupby(type_column)[feature_columns].median()
    diff = medians.subtract(overall, axis=1).divide(scale, axis=1)
    rows: list[dict[str, object]] = []
    for type_id in diff.index:
        ranked = diff.loc[type_id].dropna().sort_values(key=lambda series: series.abs(), ascending=False).head(int(top_n))
        for rank, (feature, value) in enumerate(ranked.items(), start=1):
            rows.append(
                {
                    "type_id": type_id,
                    "rank": rank,
                    "feature": feature,
                    "family": dictionary.at[feature, "family"] if feature in dictionary.index else "",
                    "label": dictionary.at[feature, "label"] if feature in dictionary.index else feature,
                    "direction": "高" if value > 0 else "低",
                    "median_iqr_z": float(value),
                }
            )
    return pd.DataFrame(rows)


def build_rule_cards(
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    train_mask: pd.Series,
    feature_columns: Sequence[str],
    *,
    target_column: str,
    random_state: int = 42,
    max_depth: int = 4,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """Fit shallow train-only rules that explain, but do not create, types."""

    columns = [column for column in feature_columns if column in features.columns]
    x = features[columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x.loc[train_mask].median().fillna(0.0)
    x = x.fillna(medians)
    labels = assignments.set_index("event_id").reindex(features["event_id"])[target_column].astype(str).reset_index(drop=True)

    between = x.loc[train_mask].groupby(labels.loc[train_mask.to_numpy()].to_numpy()).median().var(axis=0)
    within = x.loc[train_mask].var(axis=0).replace(0.0, np.nan)
    discriminative = (between / within).replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    rule_features = discriminative.head(40).index.tolist()
    if len(rule_features) < 5:
        rule_features = columns[: min(40, len(columns))]
    min_leaf = max(20, int(train_mask.sum() * 0.02))
    tree = DecisionTreeClassifier(
        max_depth=int(max_depth),
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=int(random_state),
    )
    tree.fit(x.loc[train_mask, rule_features], labels.loc[train_mask])
    text = export_text(tree, feature_names=rule_features, decimals=4)

    fidelity_rows: list[dict[str, object]] = []
    for split_name, mask in (("train", train_mask), ("holdout", ~train_mask)):
        if int(mask.sum()) == 0:
            continue
        pred = tree.predict(x.loc[mask, rule_features])
        true = labels.loc[mask]
        fidelity_rows.append(
            {
                "target": target_column,
                "split": split_name,
                "count": int(mask.sum()),
                "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
                "macro_precision": float(precision_score(true, pred, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(true, pred, average="macro", zero_division=0)),
                "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
            }
        )
    importance = pd.DataFrame(
        [
            {"target": target_column, "feature": feature, "tree_importance": float(value)}
            for feature, value in zip(rule_features, tree.feature_importances_)
            if value > 0
        ]
    ).sort_values("tree_importance", ascending=False)
    return text, pd.DataFrame(fidelity_rows), importance


def build_weak_anchor_agreement(assignments: pd.DataFrame) -> pd.DataFrame:
    anchor_map = {
        "C3-A": "shock",
        "C3-B": "shock",
        "C3-C": "trend",
        "C3-D": "base",
        "C3-E": "base",
    }
    frame = assignments.copy()
    frame["weak_anchor"] = frame["source_subcluster_id"].map(anchor_map)
    grouped = frame.groupby(["weak_anchor", "mechanism", "split"], as_index=False).agg(count=("event_id", "size"))
    grouped["share_within_anchor_split"] = grouped["count"] / grouped.groupby(["weak_anchor", "split"])["count"].transform("sum")
    return grouped


def perturb_future_metadata(features: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Change future-only metadata without touching any causal feature column."""

    out = features.copy()
    rng = np.random.default_rng(int(random_state))
    if "completion_bars" in out:
        out["completion_bars"] = rng.integers(1, 10_000, size=len(out))
    if "realized_confirmation_move_pct" in out:
        out["realized_confirmation_move_pct"] = rng.normal(0.0, 100.0, size=len(out))
    if "confirmation_time" in out:
        out["confirmation_time"] = pd.to_datetime(out["feature_available_time"]) + pd.to_timedelta(
            rng.integers(1, 100_000, size=len(out)), unit="m"
        )
    if "confirmation_available_time" in out:
        out["confirmation_available_time"] = pd.to_datetime(out["confirmation_time"]) + pd.Timedelta(minutes=1)
    return out
