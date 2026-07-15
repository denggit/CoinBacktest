#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Event-level online reversal-opportunity research.

Research 05 intentionally stops predicting the exact retrospective Swing Low
minute.  It asks a more tradable question:

    after the current 1m bar closes, does entry at the next-bar open reach a
    +1% future *closed-bar close* before suffering material adverse movement?

The script is research only.  It does not place orders, apply fees, choose a
stop, size positions, or run a portfolio backtest.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from research.market_structure.swing_low_typology.common.online_recognizability import (  # noqa: E402
    CandidateGateConfig,
    attach_temporal_split,
    binary_metrics,
    build_candidate_episodes,
    build_online_candidate_events,
    fit_binary_model,
    prepare_feature_matrix,
    purge_temporal_label_overlap,
)
from research.market_structure.swing_low_typology.common.reversal_opportunity import (  # noqa: E402
    ADVERSE_LEVELS_PCT,
    FEATURE_GROUP_ORDER,
    FeatureBuildResult,
    attach_nearest_swing_distance,
    build_reversal_candidate_features,
    build_reversal_forward_labels,
    choose_validation_event_spec,
    empirical_percentile,
    opportunity_event_metrics,
    select_first_crossing_events,
    select_usable_features,
    threshold_cooldown_grid,
)
from research.market_structure.swing_low_typology.common.swing_low_dataset import (  # noqa: E402
    validate_trade_bar_fields,
)

SCRIPT_NAME = "05_online_reversal_opportunity_research"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_1M_ONLINE_REVERSAL_OPPORTUNITY_05"
EDGE_ID = "RESEARCH_ONLY_ETH_ONLINE_REVERSAL_OPPORTUNITY"
TITLE = "ETH Online Reversal Opportunity Research 05"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/05_online_reversal_opportunity"
DEFAULT_STAGE1_DIR = "data/reports/research/market_structure/swing_low_typology/01_causal_typology"

MODEL_FAMILIES: tuple[str, ...] = ("logistic", "hist_gbdt")
ARCHITECTURES: tuple[str, ...] = ("tp_only", "two_stage", "direct_clean")
TOP_FRACTIONS: tuple[float, ...] = (0.001, 0.0025, 0.005, 0.01, 0.02, 0.05)
COOLDOWNS: tuple[int, ...] = (5, 10, 15, 30)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal event-level +1% reversal-opportunity recognizability research.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--fit-end-date", default="2023-12-31 23:59:59")
    p.add_argument("--validation-end-date", default="2024-12-31 23:59:59")
    p.add_argument("--target-move-pct", type=float, default=1.0)
    p.add_argument("--forward-horizon-bars", type=int, default=60)
    p.add_argument("--clean-adverse-pct", type=float, default=0.25)
    p.add_argument("--stage1-report-dir", default=DEFAULT_STAGE1_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--lookback", type=int, default=240)
    p.add_argument("--candidate-new-low-window", type=int, default=5)
    p.add_argument("--candidate-near-floor-window", type=int, default=60)
    p.add_argument("--candidate-position-window", type=int, default=120)
    p.add_argument("--candidate-near-floor-tolerance-bp", type=float, default=20.0)
    p.add_argument("--candidate-max-position-in-range", type=float, default=0.55)
    p.add_argument("--training-episode-gap-bars", type=int, default=1)
    p.add_argument("--label-vectorized-chunk-size", type=int, default=50_000)
    p.add_argument("--model-min-samples-leaf", type=int, default=100)
    p.add_argument("--minimum-temporal-split-rows", type=int, default=1_000)
    p.add_argument("--minimum-validation-events", type=int, default=30)
    p.add_argument("--causal-audit-sample-size", type=int, default=8)
    p.add_argument("--seed-stability-count", type=int, default=3)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args(argv)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _end_exclusive(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if len(str(value).strip()) <= 10:
        timestamp += pd.Timedelta(days=1)
    return timestamp


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    print(
        f"[load] source=trade_bar {args.symbol} {args.timeframe} "
        f"{args.warmup_start_date}->{args.end_date}",
        flush=True,
    )
    loader = OKXTradeBarLoader(
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_dir=args.data_dir,
        db_name=args.db_name,
    )
    bars = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
        chunksize=int(args.chunksize),
        force_rebuild=bool(args.force_rebuild),
        build_missing=not bool(args.no_build_missing),
    )
    if bars.empty:
        raise RuntimeError("No trade-bar data loaded")
    bars = bars.sort_index()
    if not isinstance(bars.index, pd.DatetimeIndex):
        bars.index = pd.to_datetime(bars.index, errors="coerce")
    bars = bars[~bars.index.isna()]
    bars = bars[~bars.index.duplicated(keep="last")]
    print(f"       rows={len(bars):,} range={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _rebuild_split_episodes(frame: pd.DataFrame, *, max_gap_bars: int) -> pd.DataFrame:
    """Build duplicate-control episodes independently inside each time split.

    This prevents validation/holdout candidate runs from changing fit-period
    sample weights at a calendar boundary.  Episode metadata is training/evaluation
    bookkeeping only and is never included in model features.
    """

    if frame.empty:
        return frame.copy()
    parts: list[pd.DataFrame] = []
    for split_name, group in frame.groupby("split", sort=False, dropna=False):
        base = group.drop(columns=["episode_id", "episode_size", "episode_weight"], errors="ignore")
        rebuilt = build_candidate_episodes(base, max_gap_bars=int(max_gap_bars))
        rebuilt["episode_id"] = str(split_name) + "_" + rebuilt["episode_id"].astype(str)
        parts.append(rebuilt)
    return pd.concat(parts, ignore_index=True).sort_values("extreme_pos").reset_index(drop=True)


def _candidate_config(args: argparse.Namespace) -> CandidateGateConfig:
    return CandidateGateConfig(
        lookback=int(args.lookback),
        horizon=int(args.forward_horizon_bars),
        new_low_window=int(args.candidate_new_low_window),
        near_floor_window=int(args.candidate_near_floor_window),
        position_window=int(args.candidate_position_window),
        near_floor_tolerance_bp=float(args.candidate_near_floor_tolerance_bp),
        max_position_in_range=float(args.candidate_max_position_in_range),
    )


def _load_optional_stage1(stage1_dir: Path) -> tuple[pd.DataFrame, dict[str, object] | None]:
    events_path = stage1_dir / "02_swing_low_events.csv"
    manifest_path = stage1_dir / "00_manifest.json"
    if not events_path.exists() or not manifest_path.exists():
        return pd.DataFrame(), None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "swing_extreme_price_source": "low",
        "swing_entry_price_source": "next_bar_open",
        "swing_target_observation_source": "future_closed_bar_close",
    }
    mismatches = [f"{key}={manifest.get(key)}" for key, value in expected.items() if manifest.get(key) != value]
    if mismatches:
        raise RuntimeError("Stage1 label policy mismatch: " + ", ".join(mismatches))
    events = pd.read_csv(events_path, usecols=["event_id", "extreme_pos", "extreme_time"])
    return events, manifest


def _target_columns(clean_adverse_pct: float) -> tuple[str, str]:
    suffix = str(float(clean_adverse_pct)).replace(".", "p")
    clean = f"tp_before_adverse_{suffix}pct"
    return clean, "tp_hit_1pct"


def _requested_group_features(membership: pd.DataFrame, group: str) -> tuple[str, ...]:
    return tuple(membership.loc[membership["feature_group"].eq(group), "feature"].astype(str))


def _fit_model_triplet(
    train: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    family: str,
    clean_target: str,
    random_state: int,
    min_samples_leaf: int,
) -> dict[str, object]:
    tp_model = fit_binary_model(
        train,
        feature_columns=feature_columns,
        target_column="tp_hit_1pct",
        family=family,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
    )
    tp_positive = train[train["tp_hit_1pct"].astype(bool)].copy()
    if len(tp_positive) < 100 or tp_positive[clean_target].nunique() < 2:
        raise RuntimeError(
            f"conditional path-quality target is not trainable: rows={len(tp_positive)}, "
            f"classes={tp_positive[clean_target].nunique()}"
        )
    quality_model = fit_binary_model(
        tp_positive,
        feature_columns=feature_columns,
        target_column=clean_target,
        family=family,
        random_state=random_state,
        min_samples_leaf=max(20, min_samples_leaf // 2),
    )
    direct_clean_model = fit_binary_model(
        train,
        feature_columns=feature_columns,
        target_column=clean_target,
        family=family,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
    )
    return {
        "tp": tp_model,
        "quality": quality_model,
        "direct_clean": direct_clean_model,
    }


def _score_model_triplet(models: dict[str, object], frame: pd.DataFrame) -> pd.DataFrame:
    tp = models["tp"].predict_proba(frame)  # type: ignore[attr-defined]
    quality = models["quality"].predict_proba(frame)  # type: ignore[attr-defined]
    direct = models["direct_clean"].predict_proba(frame)  # type: ignore[attr-defined]
    return pd.DataFrame(
        {
            "tp_probability": tp,
            "path_quality_probability": quality,
            "score_tp_only": tp,
            "score_two_stage": tp * quality,
            "score_direct_clean": direct,
        },
        index=frame.index,
    )


def _architecture_score_column(architecture: str) -> str:
    mapping = {
        "tp_only": "score_tp_only",
        "two_stage": "score_two_stage",
        "direct_clean": "score_direct_clean",
    }
    if architecture not in mapping:
        raise ValueError(f"unknown architecture: {architecture}")
    return mapping[architecture]


def _model_selection(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    feature_groups: dict[str, tuple[str, ...]],
    args: argparse.Namespace,
    clean_target: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_jobs = len(feature_groups) * len(MODEL_FAMILIES)
    done = 0
    reporter = ProgressReporter("[models] validation ablation", total=total_jobs, every=1)
    for group in FEATURE_GROUP_ORDER:
        requested = feature_groups[group]
        selected = select_usable_features(fit, requested)
        if len(selected) < 20:
            raise RuntimeError(f"feature group {group} has only {len(selected)} usable fit-period features")
        for family in MODEL_FAMILIES:
            models = _fit_model_triplet(
                fit,
                feature_columns=selected,
                family=family,
                clean_target=clean_target,
                random_state=int(args.random_state),
                min_samples_leaf=int(args.model_min_samples_leaf),
            )
            scores = _score_model_triplet(models, validation)
            tp_metrics = binary_metrics(validation["tp_hit_1pct"], scores["tp_probability"])
            conditional = validation[validation["tp_hit_1pct"].astype(bool)]
            conditional_metrics = binary_metrics(
                conditional[clean_target],
                scores.loc[conditional.index, "path_quality_probability"],
            )
            for architecture in ARCHITECTURES:
                score_column = _architecture_score_column(architecture)
                clean_metrics = binary_metrics(validation[clean_target], scores[score_column])
                rows.append(
                    {
                        "feature_group": group,
                        "family": family,
                        "architecture": architecture,
                        "feature_count": len(selected),
                        "clean_pr_auc": clean_metrics["pr_auc"],
                        "clean_roc_auc": clean_metrics["roc_auc"],
                        "clean_brier": clean_metrics["brier"],
                        "clean_precision_top_1pct": clean_metrics["precision_top_1pct"],
                        "clean_lift_top_1pct": clean_metrics["lift_top_1pct"],
                        "tp_pr_auc": tp_metrics["pr_auc"],
                        "tp_roc_auc": tp_metrics["roc_auc"],
                        "tp_precision_top_1pct": tp_metrics["precision_top_1pct"],
                        "conditional_quality_pr_auc": conditional_metrics["pr_auc"],
                        "conditional_quality_roc_auc": conditional_metrics["roc_auc"],
                    }
                )
            done += 1
            if done < total_jobs:
                reporter.update(done)
            del models, scores
            gc.collect()
    reporter.close()
    table = pd.DataFrame(rows)
    return table.sort_values(
        ["clean_pr_auc", "clean_brier", "tp_pr_auc", "feature_count"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)


def _scoring_metadata_columns(frame: pd.DataFrame, clean_target: str) -> list[str]:
    preferred = (
        "event_id", "extreme_time", "feature_available_time", "extreme_pos", "extreme_price",
        "split", "year", "episode_id", "episode_size", "episode_weight", "entry_time",
        "entry_price", "label_end_time", "forward_horizon_bars", "tp_hit_1pct", clean_target,
        "tp_before_adverse_0p5pct", "tp_before_adverse_0p75pct", "tp_before_adverse_1p0pct",
        "tp_within_15", "tp_within_30", "tp_within_45", "mfe_pct", "mae_horizon_pct",
        "mae_before_tp_pct", "terminal_return_pct", "tp_first_touch_bar",
    )
    return [column for column in preferred if column in frame.columns]


def _fit_selected_validation_scores(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    selected_spec: pd.Series,
    feature_groups: dict[str, tuple[str, ...]],
    args: argparse.Namespace,
    clean_target: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    group = str(selected_spec["feature_group"])
    family = str(selected_spec["family"])
    architecture = str(selected_spec["architecture"])
    feature_columns = select_usable_features(fit, feature_groups[group])
    models = _fit_model_triplet(
        fit,
        feature_columns=feature_columns,
        family=family,
        clean_target=clean_target,
        random_state=int(args.random_state),
        min_samples_leaf=int(args.model_min_samples_leaf),
    )
    scores = _score_model_triplet(models, validation)
    result = validation[_scoring_metadata_columns(validation, clean_target)].copy()
    for column in scores:
        result[column] = scores[column].to_numpy()
    result["opportunity_raw_score"] = result[_architecture_score_column(architecture)]
    return result, feature_columns


def _fit_final_scores(
    train_all: pd.DataFrame,
    holdout: pd.DataFrame,
    selected_spec: pd.Series,
    feature_columns: Sequence[str],
    args: argparse.Namespace,
    clean_target: str,
    *,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    seed = int(args.random_state if random_state is None else random_state)
    family = str(selected_spec["family"])
    architecture = str(selected_spec["architecture"])
    models = _fit_model_triplet(
        train_all,
        feature_columns=feature_columns,
        family=family,
        clean_target=clean_target,
        random_state=seed,
        min_samples_leaf=int(args.model_min_samples_leaf),
    )
    train_scores = _score_model_triplet(models, train_all)
    holdout_scores = _score_model_triplet(models, holdout)
    train_scored = train_all[_scoring_metadata_columns(train_all, clean_target)].copy()
    holdout_scored = holdout[_scoring_metadata_columns(holdout, clean_target)].copy()
    for column in train_scores:
        train_scored[column] = train_scores[column].to_numpy()
        holdout_scored[column] = holdout_scores[column].to_numpy()
    selected_column = _architecture_score_column(architecture)
    train_scored["opportunity_raw_score"] = train_scored[selected_column]
    holdout_scored["opportunity_raw_score"] = holdout_scored[selected_column]
    holdout_scored["opportunity_score_0_100"] = empirical_percentile(
        train_scored["opportunity_raw_score"],
        holdout_scored["opportunity_raw_score"],
    )
    return train_scored, holdout_scored, models


def _efficient_feature_importance(
    model: object,
    validation: pd.DataFrame,
    *,
    target_column: str,
    random_state: int,
    top_n: int = 30,
    max_rows: int = 2_000,
) -> pd.DataFrame:
    """Bounded-cost importance diagnostic for the final selected model.

    Logistic models expose standardized coefficients directly.  Histogram GBDT
    models do not expose stable public split importances, so we run a single-pass
    permutation diagnostic on a bounded, class-preserving sample.  This report is
    post-selection interpretation only and never feeds model/threshold selection.
    """

    if validation.empty:
        return pd.DataFrame()
    family = str(getattr(model, "family", ""))
    feature_columns = tuple(getattr(model, "feature_columns", ()))
    estimator = getattr(model, "model", None)
    if not feature_columns or estimator is None:
        return pd.DataFrame()

    if family == "logistic" and hasattr(estimator, "named_steps"):
        fitted = estimator.named_steps.get("model")
        coefficients = np.asarray(getattr(fitted, "coef_", []), dtype=float)
        if coefficients.ndim == 2 and coefficients.shape[1] == len(feature_columns):
            rows = pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance_mean": np.abs(coefficients[0]),
                    "importance_std": 0.0,
                    "signed_effect": coefficients[0],
                    "importance_method": "absolute standardized logistic coefficient",
                    "diagnostic_rows": int(len(validation)),
                }
            )
            return rows.sort_values("importance_mean", ascending=False).head(int(top_n)).reset_index(drop=True)

    target = validation[target_column].astype(bool)
    if target.nunique() < 2:
        return pd.DataFrame()
    positive = validation[target]
    negative = validation[~target]
    if len(validation) > int(max_rows):
        positive_take = min(len(positive), max(1, int(max_rows) // 2))
        negative_take = min(len(negative), int(max_rows) - positive_take)
        if positive_take + negative_take < int(max_rows):
            positive_take = min(len(positive), int(max_rows) - negative_take)
        sample = pd.concat(
            [
                positive.sample(positive_take, random_state=int(random_state)) if len(positive) > positive_take else positive,
                negative.sample(negative_take, random_state=int(random_state) + 1) if len(negative) > negative_take else negative,
            ],
            ignore_index=True,
        ).sample(frac=1.0, random_state=int(random_state)).reset_index(drop=True)
    else:
        sample = validation.reset_index(drop=True)

    x = prepare_feature_matrix(sample, feature_columns, getattr(model, "medians", None))
    values = x.to_numpy(dtype=float, copy=True)
    y = sample[target_column].astype(int).to_numpy()
    prediction_frame = pd.DataFrame(values, columns=feature_columns, copy=False)
    baseline_probability = np.asarray(estimator.predict_proba(prediction_frame)[:, 1], dtype=float)
    baseline_ap = float(average_precision_score(y, baseline_probability))
    rng = np.random.default_rng(int(random_state))
    rows: list[dict[str, object]] = []
    for column_index, feature in enumerate(feature_columns):
        original = values[:, column_index].copy()
        values[:, column_index] = original[rng.permutation(len(original))]
        probability = np.asarray(estimator.predict_proba(prediction_frame)[:, 1], dtype=float)
        values[:, column_index] = original
        rows.append(
            {
                "feature": feature,
                "importance_mean": baseline_ap - float(average_precision_score(y, probability)),
                "importance_std": 0.0,
                "signed_effect": np.nan,
                "prediction_shift_mean": float(np.mean(np.abs(probability - baseline_probability))),
                "importance_method": "single-pass permutation AP drop on bounded class-preserving sample",
                "diagnostic_rows": int(len(sample)),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["importance_mean", "prediction_shift_mean"], ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )


def _label_diagnostics(frame: pd.DataFrame, clean_target: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(["split", "year"], dropna=False):
        split, year = keys
        tp = group["tp_hit_1pct"].astype(bool)
        rows.append(
            {
                "split": split,
                "year": int(year),
                "candidate_count": len(group),
                "tp_rate": float(tp.mean()),
                "clean_0p25_rate": float(group[clean_target].astype(bool).mean()),
                "tp_within_15_rate": float(group["tp_within_15"].astype(bool).mean()),
                "tp_within_30_rate": float(group["tp_within_30"].astype(bool).mean()),
                "median_mfe_pct": float(pd.to_numeric(group["mfe_pct"], errors="coerce").median()),
                "median_mae_horizon_pct": float(pd.to_numeric(group["mae_horizon_pct"], errors="coerce").median()),
                "median_mae_before_tp_pct": float(pd.to_numeric(group.loc[tp, "mae_before_tp_pct"], errors="coerce").median()) if tp.any() else np.nan,
                "median_tp_bars": float(pd.to_numeric(group.loc[tp, "tp_first_touch_bar"], errors="coerce").median()) if tp.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _score_bucket_table(frame: pd.DataFrame, clean_target: str, buckets: int = 20) -> pd.DataFrame:
    data = frame.copy()
    rank = pd.to_numeric(data["opportunity_raw_score"], errors="coerce").rank(method="first", pct=True)
    data["score_bucket"] = np.minimum(np.ceil(rank * buckets), buckets).astype(int)
    data["tp_bar_or_nan"] = pd.to_numeric(data["tp_first_touch_bar"], errors="coerce").where(data["tp_hit_1pct"].astype(bool))
    result = (
        data.groupby("score_bucket", as_index=False)
        .agg(
            count=("event_id", "size"),
            mean_score=("opportunity_score_0_100", "mean"),
            tp_rate=("tp_hit_1pct", "mean"),
            clean_0p25_rate=(clean_target, "mean"),
            clean_0p50_rate=("tp_before_adverse_0p5pct", "mean"),
            median_mfe_pct=("mfe_pct", "median"),
            median_mae_horizon_pct=("mae_horizon_pct", "median"),
            median_mae_before_tp_pct=("mae_before_tp_pct", "median"),
            median_tp_bars=("tp_bar_or_nan", "median"),
        )
    )
    return result


def _group_event_metrics(events: pd.DataFrame, period: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    data = events.copy()
    timestamp = pd.to_datetime(data["extreme_time"])
    if period == "year":
        data["period"] = timestamp.dt.year.astype(str)
    elif period == "month":
        data["period"] = timestamp.dt.to_period("M").astype(str)
    else:
        raise ValueError(period)
    rows: list[dict[str, object]] = []
    for label, group in data.groupby("period"):
        rows.append({"period": label, **opportunity_event_metrics(group)})
    return pd.DataFrame(rows)


def _swing_proximity_table(frame: pd.DataFrame) -> pd.DataFrame:
    if "nearest_swing_abs_distance_bars" not in frame.columns:
        return pd.DataFrame()
    data = frame.copy()
    bins = [-0.1, 0, 1, 3, 5, 10, 30, 60, np.inf]
    labels = ["0", "1", "2-3", "4-5", "6-10", "11-30", "31-60", ">60"]
    data["distance_bucket"] = pd.cut(
        pd.to_numeric(data["nearest_swing_abs_distance_bars"], errors="coerce"),
        bins=bins,
        labels=labels,
    )
    return (
        data.groupby("distance_bucket", observed=True, as_index=False)
        .agg(
            count=("event_id", "size"),
            mean_opportunity_score=("opportunity_score_0_100", "mean"),
            tp_rate=("tp_hit_1pct", "mean"),
            clean_0p25_rate=("tp_before_adverse_0p25pct", "mean"),
        )
    )


def _raw_future_perturbation_audit(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    feature_columns: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame([{"event_id": "", "passed": False, "maximum_absolute_difference": np.nan, "detail": "no candidates"}])
    audit_lookback = max(int(args.lookback), 60 * 12 + 60)
    valid = candidates[
        (pd.to_numeric(candidates["extreme_pos"], errors="coerce") >= audit_lookback)
        & (pd.to_numeric(candidates["extreme_pos"], errors="coerce") + int(args.forward_horizon_bars) < len(bars))
    ]
    sample = valid.sample(min(int(args.causal_audit_sample_size), len(valid)), random_state=int(args.random_state))
    rows: list[dict[str, object]] = []
    numeric_columns = [
        column
        for column in (
            "open", "high", "low", "close", "volume", "trades_count", "notional", "buy_notional",
            "sell_notional", "delta_notional", "large_buy_notional", "large_sell_notional",
            "large_delta_notional", "large_trades_count", "max_trade_notional", "avg_trade_size", "vwap",
        )
        if column in bars.columns
    ]
    for source in sample.itertuples(index=False):
        global_pos = int(source.extreme_pos)
        start = global_pos - audit_lookback
        end = global_pos + int(args.forward_horizon_bars) + 1
        local = bars.iloc[start:end].copy()
        local_pos = global_pos - start
        timestamp = local.index[local_pos]
        event = pd.DataFrame(
            {
                "event_id": [str(source.event_id)],
                "extreme_time": [timestamp],
                "feature_available_time": [timestamp + pd.Timedelta(minutes=1)],
                "extreme_pos": [local_pos],
                "extreme_price": [float(local.iloc[local_pos]["low"])],
                "split": ["audit"],
                "year": [timestamp.year],
                "candidate_new_low": [True],
                "candidate_near_floor": [True],
                "candidate_range_position": [0.0],
            }
        )
        original = build_reversal_candidate_features(local, event, show_progress=False).frame
        perturbed = local.copy()
        rng = np.random.default_rng(int(args.random_state) + global_pos)
        future_start = local_pos + 1
        for column in numeric_columns:
            values = pd.to_numeric(perturbed[column], errors="coerce").to_numpy(dtype=float, copy=True)
            segment = values[future_start:]
            values[future_start:] = np.where(
                np.isfinite(segment),
                segment * rng.uniform(0.2, 4.0, len(segment)),
                segment,
            )
            perturbed[column] = values
        # Restore valid OHLC ordering in future rows after the arbitrary perturbation.
        if future_start < len(perturbed):
            future = perturbed.iloc[future_start:].copy()
            center = pd.to_numeric(future["close"], errors="coerce").to_numpy(dtype=float, copy=True)
            spread = np.maximum(np.abs(center) * rng.uniform(0.0002, 0.01, len(center)), 1e-9)
            open_future = center * rng.uniform(0.995, 1.005, len(center))
            close_future = center * rng.uniform(0.995, 1.005, len(center))
            perturbed.iloc[future_start:, perturbed.columns.get_loc("open")] = open_future
            perturbed.iloc[future_start:, perturbed.columns.get_loc("close")] = close_future
            perturbed.iloc[future_start:, perturbed.columns.get_loc("high")] = np.maximum(open_future, close_future) + spread
            perturbed.iloc[future_start:, perturbed.columns.get_loc("low")] = np.minimum(open_future, close_future) - spread
        changed = build_reversal_candidate_features(perturbed, event, show_progress=False).frame
        comparable = [column for column in feature_columns if column in original.columns and column in changed.columns]
        a = original[comparable].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        b = changed[comparable].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        diff = np.abs(a - b)
        finite = np.isfinite(diff)
        maximum = float(np.nanmax(diff[finite])) if finite.any() else 0.0
        rows.append(
            {
                "event_id": str(source.event_id),
                "maximum_absolute_difference": maximum,
                "feature_count": len(comparable),
                "passed": bool(maximum <= 1e-10),
                "detail": "all rows strictly after current bar were perturbed",
            }
        )
    return pd.DataFrame(rows)


def _causal_audit(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    alignment: pd.DataFrame,
    raw_audit: pd.DataFrame,
    *,
    fit_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.DataFrame:
    forbidden_tokens = (
        "future", "forward", "label", "mfe", "mae", "tp_hit", "adverse", "entry_price",
        "nearest_swing", "reference", "confirmation", "completion",
    )
    forbidden = [column for column in feature_columns if any(token in column.lower() for token in forbidden_tokens)]
    feature_time = pd.to_datetime(frame["feature_available_time"])
    entry_time = pd.to_datetime(frame["entry_time"])
    fit_labels = frame[frame["split"].eq("fit")]
    validation_labels = frame[frame["split"].eq("validation")]
    return pd.DataFrame(
        [
            {
                "check": "next_open_after_feature_available",
                "passed": bool((feature_time <= entry_time).all()),
                "detail": f"minimum_lag={(entry_time - feature_time).min()}",
            },
            {
                "check": "future_labels_excluded_from_features",
                "passed": not forbidden,
                "detail": ",".join(forbidden),
            },
            {
                "check": "fit_labels_do_not_cross_2023",
                "passed": bool((pd.to_datetime(fit_labels["label_end_time"]) <= fit_end).all()),
                "detail": f"rows={len(fit_labels)}",
            },
            {
                "check": "validation_labels_do_not_cross_2024",
                "passed": bool((pd.to_datetime(validation_labels["label_end_time"]) <= validation_end).all()),
                "detail": f"rows={len(validation_labels)}",
            },
            {
                "check": "causal_high_timeframe_available_time",
                "passed": bool(not alignment.empty and alignment["passed"].all()),
                "detail": alignment.to_json(orient="records"),
            },
            {
                "check": "raw_future_feature_perturbation",
                "passed": bool(not raw_audit.empty and raw_audit["passed"].all()),
                "detail": f"audited={len(raw_audit)}, max_diff={raw_audit.get('maximum_absolute_difference', pd.Series([np.nan])).max()}",
            },
            {
                "check": "close_only_path_policy",
                "passed": True,
                "detail": "entry=next bar open; target/MAE/first touch=future closed-bar closes; future high/low unused",
            },
        ]
    )


def _fit_selected_architecture_score(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    selected_spec: pd.Series,
    feature_columns: Sequence[str],
    args: argparse.Namespace,
    clean_target: str,
    *,
    random_state: int,
) -> np.ndarray:
    family = str(selected_spec["family"])
    architecture = str(selected_spec["architecture"])
    if architecture == "tp_only":
        model = fit_binary_model(
            train,
            feature_columns=feature_columns,
            target_column="tp_hit_1pct",
            family=family,
            random_state=random_state,
            min_samples_leaf=int(args.model_min_samples_leaf),
        )
        return model.predict_proba(evaluation)
    if architecture == "direct_clean":
        model = fit_binary_model(
            train,
            feature_columns=feature_columns,
            target_column=clean_target,
            family=family,
            random_state=random_state,
            min_samples_leaf=int(args.model_min_samples_leaf),
        )
        return model.predict_proba(evaluation)
    tp_model = fit_binary_model(
        train,
        feature_columns=feature_columns,
        target_column="tp_hit_1pct",
        family=family,
        random_state=random_state,
        min_samples_leaf=int(args.model_min_samples_leaf),
    )
    tp_positive = train[train["tp_hit_1pct"].astype(bool)]
    quality_model = fit_binary_model(
        tp_positive,
        feature_columns=feature_columns,
        target_column=clean_target,
        family=family,
        random_state=random_state,
        min_samples_leaf=max(20, int(args.model_min_samples_leaf) // 2),
    )
    return tp_model.predict_proba(evaluation) * quality_model.predict_proba(evaluation)


def _seed_stability(
    train_all: pd.DataFrame,
    holdout: pd.DataFrame,
    selected_spec: pd.Series,
    feature_columns: Sequence[str],
    args: argparse.Namespace,
    clean_target: str,
) -> pd.DataFrame:
    family = str(selected_spec["family"])
    # LBFGS logistic regression is deterministic for fixed data. Re-fitting it
    # several times only burns minutes without testing a real random component.
    if family == "logistic":
        return pd.DataFrame(
            [
                {
                    "seed_left": int(args.random_state),
                    "seed_right": int(args.random_state),
                    "spearman": 1.0,
                    "top_1pct_overlap": 1.0,
                    "detail": "logistic/LBFGS is deterministic; no redundant refits",
                }
            ]
        )

    seeds = [int(args.random_state)]
    for candidate in (7, 19, 73, 101):
        if candidate not in seeds and len(seeds) < int(args.seed_stability_count):
            seeds.append(candidate)
    # Stability is a model-randomness diagnostic, not a final performance
    # estimate. Bound its data size so it cannot dominate the actual research.
    train_sample = train_all
    if len(train_sample) > 60_000:
        train_sample = train_sample.sample(60_000, random_state=int(args.random_state))
    holdout_sample = holdout
    if len(holdout_sample) > 40_000:
        holdout_sample = holdout_sample.sample(40_000, random_state=int(args.random_state))
    scores: dict[int, np.ndarray] = {}
    reporter = ProgressReporter("[models] seed stability", total=len(seeds), every=1)
    for i, seed in enumerate(seeds, start=1):
        scores[seed] = _fit_selected_architecture_score(
            train_sample,
            holdout_sample,
            selected_spec,
            feature_columns,
            args,
            clean_target,
            random_state=seed,
        )
        if i < len(seeds):
            reporter.update(i)
    reporter.close()
    rows: list[dict[str, object]] = []
    for i, left in enumerate(seeds):
        for right in seeds[i + 1 :]:
            a = scores[left]
            b = scores[right]
            n = max(1, int(np.ceil(len(a) * 0.01)))
            top_a = set(np.argpartition(a, -n)[-n:].tolist())
            top_b = set(np.argpartition(b, -n)[-n:].tolist())
            rows.append(
                {
                    "seed_left": left,
                    "seed_right": right,
                    "spearman": float(pd.Series(a).corr(pd.Series(b), method="spearman")),
                    "top_1pct_overlap": float(len(top_a & top_b) / n),
                    "detail": f"bounded stability sample train={len(train_sample)}, holdout={len(holdout_sample)}",
                }
            )
    return pd.DataFrame(rows)

def _plot_score_buckets(table: pd.DataFrame, path: Path) -> None:
    if table.empty:
        return
    import matplotlib.pyplot as plt

    data = table.sort_values("score_bucket")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(data["score_bucket"], data["tp_rate"] * 100.0, marker="o", label="TP +1%")
    ax.plot(data["score_bucket"], data["clean_0p25_rate"] * 100.0, marker="o", label="TP before -0.25%")
    ax.set_xlabel("Opportunity score bucket (higher is better)")
    ax.set_ylabel("Outcome rate (%)")
    ax.set_title("Frozen holdout reversal-opportunity ranking")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _summary(
    args: argparse.Namespace,
    selected_spec: pd.Series,
    selected_event_spec: pd.Series,
    holdout_metrics: pd.DataFrame,
    selected_events: pd.DataFrame,
    validation_selection: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    metrics = holdout_metrics.iloc[0].to_dict() if not holdout_metrics.empty else {}
    event_metrics = opportunity_event_metrics(selected_events)
    m0 = validation_selection[validation_selection["feature_group"].eq("M0_core")]["clean_pr_auc"].max()
    m1 = validation_selection[validation_selection["feature_group"].eq("M1_session")]["clean_pr_auc"].max()
    m2 = validation_selection[validation_selection["feature_group"].eq("M2_causal_htf")]["clean_pr_auc"].max()
    lines = [
        f"# {TITLE}",
        "",
        "## Scope",
        "",
        "- Research only: no fees, stop, position sizing, order execution, or portfolio backtest.",
        "- The target is no longer the exact retrospective Swing Low minute.",
        "- Signal information ends at the current closed 1m bar; the executable reference is next-bar open.",
        f"- Success is a future closed-bar close reaching +{args.target_move_pct:g}% within {args.forward_horizon_bars} bars.",
        f"- Primary path-quality target is TP before a -{args.clean_adverse_pct:g}% future-close adverse move.",
        "",
        "## Selected validation design",
        "",
        f"- Feature group: `{selected_spec['feature_group']}`.",
        f"- Model family: `{selected_spec['family']}`.",
        f"- Architecture: `{selected_spec['architecture']}`.",
        f"- Frozen top fraction: {float(selected_event_spec['top_fraction']):.3%}.",
        f"- Cooldown: {int(selected_event_spec['cooldown_bars'])} bars.",
        "- Scores are ranking scores, not calibrated success probabilities.",
        "",
        "## Feature ablation on 2024 validation",
        "",
        f"- M0 core best clean PR-AUC: {m0:.4f}.",
        f"- M1 core + session best clean PR-AUC: {m1:.4f}.",
        f"- M2 + causal 5m/15m/1H best clean PR-AUC: {m2:.4f}.",
        "",
        "## Frozen 2025-2026H1 holdout",
        "",
        f"- Bar-level TP PR-AUC: {metrics.get('tp_pr_auc', np.nan):.4f}.",
        f"- Bar-level clean-path PR-AUC: {metrics.get('clean_pr_auc', np.nan):.4f}.",
        f"- Selected independent events: {int(event_metrics['event_count'])}.",
        f"- Event TP rate: {event_metrics['tp_rate']:.2%}.",
        f"- Event TP-before--0.25% rate: {event_metrics['clean_0p25_rate']:.2%}.",
        f"- Event TP-before--0.50% rate: {event_metrics['clean_0p50_rate']:.2%}.",
        f"- Median TP time: {event_metrics['median_tp_bars']:.1f} bars.",
        f"- Median MAE before TP: {event_metrics['median_mae_before_tp_pct']:.3f}%.",
        "",
        "## Interpretation rule",
        "",
        "- A usable result requires holdout event lift, reasonable monthly frequency, stable 2025/2026 performance, and seed stability.",
        "- Session or high-timeframe context is retained only when 2024 validation improves; holdout is never used to choose the feature group.",
        "- Range bar, range footprint, and funding remain optional later-stage additions, not dependencies of this baseline.",
        "",
        "## Causal status",
        "",
        f"- All causal checks passed: {bool(not audit.empty and audit['passed'].all())}.",
    ]
    return "\n".join(lines) + "\n"


def run_research(args: argparse.Namespace) -> Path:
    out_dir = PROJECT_ROOT / args.out_dir
    stage1_dir = PROJECT_ROOT / args.stage1_report_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_target, tp_target = _target_columns(float(args.clean_adverse_pct))
    expected_clean = "tp_before_adverse_0p25pct"
    if clean_target != expected_clean:
        raise ValueError(
            "Research 05 currently freezes path-quality model selection at 0.25%; "
            "other adverse levels remain report diagnostics."
        )

    bars = load_bars(args)
    coverage = validate_trade_bar_fields(bars)
    _write_csv(coverage, out_dir / "01_trade_bar_field_coverage.csv")

    research_start = pd.Timestamp(args.start_date)
    research_end = _end_exclusive(args.end_date)
    print("[stage] build broad causal candidate universe", flush=True)
    candidates, gate_summary = build_online_candidate_events(
        bars,
        research_start=research_start,
        research_end_exclusive=research_end,
        config=_candidate_config(args),
    )
    candidates = attach_temporal_split(
        candidates,
        fit_end=pd.Timestamp(args.fit_end_date),
        validation_end=pd.Timestamp(args.validation_end_date),
    )
    candidates = _rebuild_split_episodes(candidates, max_gap_bars=int(args.training_episode_gap_bars))
    _write_csv(gate_summary, out_dir / "02_candidate_gate_summary.csv")
    if candidates.empty:
        raise RuntimeError("candidate gate produced no rows")

    # Cheap preflight catches pandas mutability and available-time regressions
    # before the full 2.36m-row vectorized feature build.
    print("[stage] causal feature preflight", flush=True)
    preflight_args = argparse.Namespace(**vars(args))
    preflight_args.causal_audit_sample_size = 1
    preflight_pool = candidates[["event_id", "extreme_pos"]].sample(min(64, len(candidates)), random_state=int(args.random_state))
    preflight = _raw_future_perturbation_audit(bars, preflight_pool, ["current_return_1", "tf5m_return_3"], preflight_args)
    if preflight.empty or not preflight["passed"].all():
        raise RuntimeError("causal feature preflight failed")

    print(f"[stage] vectorized causal features candidates={len(candidates):,}", flush=True)
    feature_result: FeatureBuildResult = build_reversal_candidate_features(bars, candidates)
    features = feature_result.frame
    feature_dictionary = feature_result.dictionary
    group_membership = feature_result.group_membership
    alignment_audit = feature_result.alignment_audit
    _write_csv(feature_dictionary, out_dir / "03_causal_feature_dictionary.csv")
    _write_csv(group_membership, out_dir / "04_feature_group_membership.csv")
    _write_csv(alignment_audit, out_dir / "05_high_timeframe_available_time_audit.csv")
    if not alignment_audit["passed"].all():
        raise RuntimeError("high-timeframe available_time audit failed")
    del candidates
    gc.collect()

    print("[stage] next-open / future-close path labels", flush=True)
    labels = build_reversal_forward_labels(
        bars,
        features[["event_id", "extreme_pos"]],
        horizon=int(args.forward_horizon_bars),
        target_move_pct=float(args.target_move_pct),
        adverse_levels_pct=ADVERSE_LEVELS_PCT,
        vectorized_chunk_size=int(args.label_vectorized_chunk_size),
        progress_every=max(int(args.label_vectorized_chunk_size), 50_000),
    )
    frame = features.merge(labels, on="event_id", how="inner", validate="one_to_one")
    del features, labels, feature_result
    gc.collect()
    frame, temporal_purge = purge_temporal_label_overlap(
        frame,
        fit_end=pd.Timestamp(args.fit_end_date),
        validation_end=pd.Timestamp(args.validation_end_date),
    )
    frame = _rebuild_split_episodes(frame, max_gap_bars=int(args.training_episode_gap_bars))
    _write_csv(temporal_purge, out_dir / "06_temporal_label_boundary_purge.csv")
    label_definitions = pd.DataFrame(
        [
            {
                "label": "tp_hit_1pct",
                "definition": "from next-bar open, any of the next 60 closed-bar closes reaches +1%",
                "future_high_low_used": False,
                "role": "primary opportunity target",
            },
            {
                "label": "tp_before_adverse_0p25pct",
                "definition": "future close reaches +1% before any future close reaches -0.25%",
                "future_high_low_used": False,
                "role": "primary clean-path target",
            },
            {
                "label": "mae_before_tp_pct",
                "definition": "maximum adverse future-close return from next open, ending at first TP close",
                "future_high_low_used": False,
                "role": "path-quality diagnostic",
            },
            {
                "label": "tp_first_touch_bar",
                "definition": "first future closed bar reaching +1%; success ends immediately",
                "future_high_low_used": False,
                "role": "speed diagnostic",
            },
        ]
    )
    _write_csv(label_definitions, out_dir / "06b_label_definitions.csv")
    if clean_target not in frame.columns or tp_target not in frame.columns:
        raise RuntimeError("required reversal targets were not generated")
    _write_csv(_label_diagnostics(frame, clean_target), out_dir / "07_label_diagnostics.csv")

    audit_frame = frame[["event_id", "extreme_time", "feature_available_time", "entry_time", "label_end_time", "split"]].copy()
    fit = frame[frame["split"].eq("fit")].copy()
    validation = frame[frame["split"].eq("validation")].copy()
    holdout = frame[frame["split"].eq("holdout")].copy()
    fit_count = int(len(fit))
    validation_count = int(len(validation))
    holdout_count = int(len(holdout))
    del frame
    gc.collect()
    if min(len(fit), len(validation), len(holdout)) < int(args.minimum_temporal_split_rows):
        raise RuntimeError(
            f"temporal split too small: fit={len(fit)}, validation={len(validation)}, holdout={len(holdout)}"
        )
    for split_name, split_frame in (("fit", fit), ("validation", validation), ("holdout", holdout)):
        if split_frame[tp_target].nunique() < 2 or split_frame[clean_target].nunique() < 2:
            raise RuntimeError(f"{split_name} split has a one-class target")

    feature_groups = {
        group: _requested_group_features(group_membership, group)
        for group in FEATURE_GROUP_ORDER
    }
    print("[stage] 2023 fit / 2024 model and feature-group selection", flush=True)
    model_selection = _model_selection(fit, validation, feature_groups, args, clean_target)
    _write_csv(model_selection, out_dir / "08_validation_model_feature_ablation.csv")
    selected_spec = model_selection.iloc[0]
    _write_csv(pd.DataFrame([selected_spec]), out_dir / "09_selected_model_spec.csv")

    validation_scored, selected_features = _fit_selected_validation_scores(
        fit,
        validation,
        selected_spec,
        feature_groups,
        args,
        clean_target,
    )
    _write_csv(pd.DataFrame({"feature": selected_features}), out_dir / "10_selected_model_features.csv")
    validation_grid, _ = threshold_cooldown_grid(
        validation_scored,
        validation_scored,
        score_column="opportunity_raw_score",
        fractions=TOP_FRACTIONS,
        cooldowns=COOLDOWNS,
        threshold_source="2024_validation_score_quantile",
    )
    selected_event_spec = choose_validation_event_spec(
        validation_grid,
        minimum_events=int(args.minimum_validation_events),
    )
    validation_grid["selected"] = (
        np.isclose(validation_grid["top_fraction"], float(selected_event_spec["top_fraction"]))
        & validation_grid["cooldown_bars"].eq(int(selected_event_spec["cooldown_bars"]))
    )
    _write_csv(validation_grid, out_dir / "11_validation_threshold_cooldown_grid.csv")
    _write_csv(pd.DataFrame([selected_event_spec]), out_dir / "12_selected_event_spec.csv")

    print("[stage] refit selected design on 2023-2024 and freeze 2025-2026", flush=True)
    train_all = pd.concat([fit, validation], ignore_index=True)
    del fit, validation, validation_scored
    gc.collect()
    final_features = tuple(selected_features)
    train_scored, holdout_scored, final_models = _fit_final_scores(
        train_all,
        holdout,
        selected_spec,
        final_features,
        args,
        clean_target,
    )
    holdout_grid, holdout_event_sets = threshold_cooldown_grid(
        train_scored,
        holdout_scored,
        score_column="opportunity_raw_score",
        fractions=TOP_FRACTIONS,
        cooldowns=COOLDOWNS,
        threshold_source="2023_2024_refit_score_quantile",
    )
    chosen_key = (float(selected_event_spec["top_fraction"]), int(selected_event_spec["cooldown_bars"]))
    selected_events = holdout_event_sets[chosen_key].copy()
    holdout_grid["selected_validation_spec"] = (
        np.isclose(holdout_grid["top_fraction"], chosen_key[0])
        & holdout_grid["cooldown_bars"].eq(chosen_key[1])
    )
    _write_csv(holdout_grid, out_dir / "13_frozen_holdout_threshold_cooldown_grid.csv")

    stage1_events, stage1_manifest = _load_optional_stage1(stage1_dir)
    if not stage1_events.empty:
        holdout_scored = attach_nearest_swing_distance(holdout_scored, stage1_events)
        selected_events = attach_nearest_swing_distance(selected_events, stage1_events)
    _write_csv(selected_events, out_dir / "14_selected_holdout_events.csv")
    _write_csv(_group_event_metrics(selected_events, "year"), out_dir / "15_selected_holdout_yearly_metrics.csv")
    _write_csv(_group_event_metrics(selected_events, "month"), out_dir / "16_selected_holdout_monthly_metrics.csv")

    tp_holdout_metrics = binary_metrics(holdout_scored[tp_target], holdout_scored["tp_probability"])
    clean_holdout_metrics = binary_metrics(holdout_scored[clean_target], holdout_scored["opportunity_raw_score"])
    holdout_metrics = pd.DataFrame(
        [
            {
                "model": "selected_opportunity_design",
                "tp_pr_auc": tp_holdout_metrics["pr_auc"],
                "tp_roc_auc": tp_holdout_metrics["roc_auc"],
                "tp_base_rate": tp_holdout_metrics["base_rate"],
                "tp_precision_top_1pct": tp_holdout_metrics["precision_top_1pct"],
                "clean_pr_auc": clean_holdout_metrics["pr_auc"],
                "clean_roc_auc": clean_holdout_metrics["roc_auc"],
                "clean_base_rate": clean_holdout_metrics["base_rate"],
                "clean_precision_top_1pct": clean_holdout_metrics["precision_top_1pct"],
                "clean_lift_top_1pct": clean_holdout_metrics["lift_top_1pct"],
            }
        ]
    )
    _write_csv(holdout_metrics, out_dir / "17_frozen_holdout_bar_metrics.csv")

    buckets = _score_bucket_table(holdout_scored, clean_target)
    _write_csv(buckets, out_dir / "18_holdout_score_buckets.csv")
    _plot_score_buckets(buckets, out_dir / "18_holdout_score_buckets.png")
    proximity = _swing_proximity_table(holdout_scored)
    if not proximity.empty:
        _write_csv(proximity, out_dir / "19_post_label_swing_proximity_diagnostic_NOT_FEATURES.csv")

    architecture = str(selected_spec["architecture"])
    if architecture == "tp_only":
        importance_model = final_models["tp"]
        importance_target = tp_target
        importance_frame = holdout
    elif architecture == "direct_clean":
        importance_model = final_models["direct_clean"]
        importance_target = clean_target
        importance_frame = holdout
    else:
        # For a two-stage score, report both components separately.
        tp_importance = _efficient_feature_importance(
            final_models["tp"],  # type: ignore[arg-type]
            holdout,
            target_column=tp_target,
            random_state=int(args.random_state),
        )
        tp_importance.insert(0, "component", "tp")
        quality_validation = holdout[holdout[tp_target].astype(bool)]
        quality_importance = _efficient_feature_importance(
            final_models["quality"],  # type: ignore[arg-type]
            quality_validation,
            target_column=clean_target,
            random_state=int(args.random_state),
        )
        quality_importance.insert(0, "component", "conditional_path_quality")
        _write_csv(pd.concat([tp_importance, quality_importance], ignore_index=True), out_dir / "20_permutation_importance.csv")
        importance_model = None
        importance_target = ""
        importance_frame = pd.DataFrame()
    if importance_model is not None:
        importance = _efficient_feature_importance(
            importance_model,  # type: ignore[arg-type]
            importance_frame,
            target_column=importance_target,
            random_state=int(args.random_state),
        )
        importance.insert(0, "component", architecture)
        _write_csv(importance, out_dir / "20_permutation_importance.csv")

    print("[stage] selected-design random-seed stability", flush=True)
    stability = _seed_stability(train_all, holdout, selected_spec, final_features, args, clean_target)
    _write_csv(stability, out_dir / "21_random_seed_stability.csv")

    representative = pd.concat(
        [
            holdout_scored.nlargest(25, "opportunity_raw_score").assign(case="highest_score"),
            holdout_scored[holdout_scored[clean_target].astype(bool)].nlargest(25, "opportunity_raw_score").assign(case="high_score_clean_success"),
            holdout_scored[~holdout_scored[clean_target].astype(bool)].nlargest(25, "opportunity_raw_score").assign(case="high_score_failure"),
        ],
        ignore_index=True,
    ).drop_duplicates("event_id")
    representative_columns = [
        column
        for column in (
            "case", "event_id", "extreme_time", "feature_available_time", "entry_time", "label_end_time",
            "opportunity_score_0_100", "opportunity_raw_score", "tp_probability", "path_quality_probability",
            "score_direct_clean", "tp_hit_1pct", clean_target, "tp_before_adverse_0p5pct",
            "mfe_pct", "mae_horizon_pct", "mae_before_tp_pct", "tp_first_touch_bar",
            "nearest_swing_signed_distance_bars", "nearest_swing_abs_distance_bars",
        )
        if column in representative.columns
    ]
    _write_csv(representative[representative_columns], out_dir / "22_representative_holdout_cases.csv")

    prediction_columns = [
        column
        for column in (
            "event_id", "extreme_time", "feature_available_time", "extreme_pos", "entry_time", "label_end_time",
            "split", "year", "opportunity_score_0_100", "opportunity_raw_score", "tp_probability",
            "path_quality_probability", "score_direct_clean", "tp_hit_1pct", clean_target,
            "tp_before_adverse_0p5pct", "tp_before_adverse_0p75pct", "tp_before_adverse_1p0pct",
            "mfe_pct", "mae_horizon_pct", "mae_before_tp_pct", "tp_first_touch_bar",
            "nearest_swing_signed_distance_bars", "nearest_swing_abs_distance_bars",
        )
        if column in holdout_scored.columns
    ]
    _write_csv(holdout_scored[prediction_columns], out_dir / "23_holdout_predictions.csv")

    print("[stage] raw future perturbation causal audit", flush=True)
    audit_pool = holdout_scored[["event_id", "extreme_pos"]].sample(
        min(max(64, int(args.causal_audit_sample_size) * 8), len(holdout_scored)),
        random_state=int(args.random_state),
    )
    raw_audit = _raw_future_perturbation_audit(bars, audit_pool, final_features, args)
    _write_csv(raw_audit, out_dir / "24_raw_future_perturbation_audit.csv")
    audit = _causal_audit(
        audit_frame,
        final_features,
        alignment_audit,
        raw_audit,
        fit_end=pd.Timestamp(args.fit_end_date),
        validation_end=pd.Timestamp(args.validation_end_date),
    )
    _write_csv(audit, out_dir / "25_causal_audit.csv")
    if not audit["passed"].all():
        failed = audit.loc[~audit["passed"], "check"].tolist()
        raise RuntimeError(f"causal audit failed: {failed}")

    manifest = {
        "script": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "fit_end_date": args.fit_end_date,
        "validation_end_date": args.validation_end_date,
        "target_move_pct": float(args.target_move_pct),
        "forward_horizon_bars": int(args.forward_horizon_bars),
        "clean_adverse_pct": float(args.clean_adverse_pct),
        "entry_price_source": "next_bar_open",
        "path_observation_source": "future_closed_bar_close",
        "future_high_low_used_for_labels": False,
        "candidate_count": int(len(audit_frame)),
        "fit_count": fit_count,
        "validation_count": validation_count,
        "holdout_count": holdout_count,
        "feature_group_order": list(FEATURE_GROUP_ORDER),
        "selected_feature_group": str(selected_spec["feature_group"]),
        "selected_family": str(selected_spec["family"]),
        "selected_architecture": str(selected_spec["architecture"]),
        "selected_feature_count": int(len(final_features)),
        "selected_top_fraction": float(selected_event_spec["top_fraction"]),
        "selected_cooldown_bars": int(selected_event_spec["cooldown_bars"]),
        "selected_holdout_event_count": int(len(selected_events)),
        "stage1_proximity_diagnostic_available": bool(stage1_manifest is not None),
        "stage1_manifest_summary": None if stage1_manifest is None else {
            "experiment_id": stage1_manifest.get("experiment_id"),
            "event_count": stage1_manifest.get("event_count"),
            "swing_extreme_price_source": stage1_manifest.get("swing_extreme_price_source"),
            "swing_entry_price_source": stage1_manifest.get("swing_entry_price_source"),
            "swing_target_observation_source": stage1_manifest.get("swing_target_observation_source"),
        },
        "causal_policy": "features=current closed 1m or older; HTF context joined only when bar_available_time <= feature_available_time; entry=next open; labels=future closes only",
        "strategy_or_backtest": False,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _summary(
        args,
        selected_spec,
        selected_event_spec,
        holdout_metrics,
        selected_events,
        model_selection,
        audit,
    )
    (out_dir / "26_RESEARCH_SUMMARY.md").write_text(summary, encoding="utf-8")

    result = finalize_research_report(
        out_dir,
        experiment_id=EXPERIMENT_ID,
        edge_id=EDGE_ID,
        title=TITLE,
    )
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={result.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
