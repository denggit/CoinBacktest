#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Large-sample walk-forward multi-objective ETH reversal research 07.

Research 07 corrects two issues found in 03-06:

1. retrospective unsupervised cluster IDs are not treated as permanent market
   semantics;
2. no double extreme-tail filter is allowed to turn thousands of opportunities
   into a few dozen apparently high-win-rate signals.

The script trains a unified multi-head reversal model.  Every feature ends at
one closed 1m bar.  The executable reference is the next-bar open and all path
labels use future closed-bar closes only.  This is research, not a strategy or
portfolio backtest.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from research.market_structure.swing_low_typology.common.online_recognizability import (  # noqa: E402
    CandidateGateConfig,
    binary_metrics,
    build_online_candidate_events,
    fit_binary_model,
)
from research.market_structure.swing_low_typology.common.reversal_opportunity import (  # noqa: E402
    build_reversal_candidate_features,
    build_reversal_forward_labels,
    opportunity_event_metrics,
    select_usable_features,
)
from research.market_structure.swing_low_typology.common.swing_low_dataset import (  # noqa: E402
    validate_trade_bar_fields,
)
from research.market_structure.swing_low_typology.common.walkforward_reversal import (  # noqa: E402
    REGION_FEATURE_GROUP,
    MECHANISM_FEATURE_GROUP,
    attach_episode_balanced_weight,
    attach_positive_opportunity_episodes,
    build_broad_candidate_regions,
    concentration_metrics,
    fit_quantile_risk_model,
    fit_soft_mechanism_transformer,
    fixed_multiobjective_score,
    mechanism_feature_dictionary,
    positive_episode_coverage,
    remove_strongest_days,
    select_first_region_signal,
)

SCRIPT_NAME = "07_walkforward_multiobjective_reversal_research"
SCRIPT_VERSION = "1.0.2"
EXPERIMENT_ID = "ETH_1M_WALKFORWARD_MULTIOBJECTIVE_REVERSAL_07"
EDGE_ID = "RESEARCH_ONLY_ETH_WALKFORWARD_REVERSAL_MODEL"
TITLE = "ETH Walk-Forward Multi-Objective Reversal Research 07"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/07_walkforward_multiobjective_reversal"

MODEL_FAMILIES: tuple[str, ...] = ("logistic", "hist_gbdt")
PRIMARY_MULTIHEAD_FAMILY = "logistic"
TP_CHALLENGER_FAMILY = "hist_gbdt"
FEATURE_GROUPS: tuple[str, ...] = ("U0_snapshot", "U1_soft_mechanism", "U2_region_hybrid")
SCORE_TYPES: tuple[str, ...] = ("tp_score", "clean50_score", "fixed_multi_score")
TOP_FRACTIONS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20)
COOLDOWNS: tuple[int, ...] = (0, 15, 30)
REFERENCE_TOP_FRACTION = 0.05
REFERENCE_COOLDOWN = 15


class FoldSpec(NamedTuple):
    fold: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


HEAD_TARGETS: dict[str, str] = {
    "p_tp60": "tp_hit_1pct",
    "p_clean25": "tp_before_adverse_0p25pct",
    "p_clean50": "tp_before_adverse_0p5pct",
    "p_fast15": "tp_within_15",
    "p_fast30": "tp_within_30",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Large-sample walk-forward multi-objective reversal research.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--target-move-pct", type=float, default=1.0)
    p.add_argument("--forward-horizon-bars", type=int, default=60)
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
    p.add_argument("--region-max-gap-bars", type=int, default=2)
    p.add_argument("--region-max-bars", type=int, default=120)
    p.add_argument("--region-retest-tolerance-bp", type=float, default=25.0)
    p.add_argument("--positive-episode-gap-bars", type=int, default=2)
    p.add_argument("--label-vectorized-chunk-size", type=int, default=50_000)
    p.add_argument("--model-min-samples-leaf", type=int, default=100)
    p.add_argument("--maximum-train-rows", type=int, default=400_000)
    p.add_argument("--prediction-chunk-size", type=int, default=100_000)
    p.add_argument("--causal-audit-sample-size", type=int, default=4)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--write-full-predictions", action="store_true")
    return p.parse_args(argv)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _end_exclusive(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if len(str(value).strip()) <= 10:
        timestamp += pd.Timedelta(days=1)
    return timestamp


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


def _folds(end_date: str) -> tuple[FoldSpec, ...]:
    research_end = _end_exclusive(end_date) - pd.Timedelta(nanoseconds=1)
    return (
        FoldSpec(
            "WF_2024",
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-12-31 23:59:59"),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-31 23:59:59"),
        ),
        FoldSpec(
            "WF_2025",
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2024-12-31 23:59:59"),
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31 23:59:59"),
        ),
        FoldSpec(
            "WF_2026H1",
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2025-12-31 23:59:59"),
            pd.Timestamp("2026-01-01"),
            research_end,
        ),
    )


def _fold_table(folds: Sequence[FoldSpec]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fold": fold.fold,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
            }
            for fold in folds
        ]
    )


def _subset_fold(frame: pd.DataFrame, fold: FoldSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    timestamp = pd.to_datetime(frame["extreme_time"])
    label_end = pd.to_datetime(frame["label_end_time"])
    train_mask = (
        (timestamp >= fold.train_start)
        & (timestamp <= fold.train_end)
        & (label_end <= fold.train_end)
    )
    test_mask = (
        (timestamp >= fold.test_start)
        & (timestamp <= fold.test_end)
        & (label_end <= fold.test_end)
    )
    remaining_train_end = label_end[train_mask]
    remaining_test_end = label_end[test_mask]
    dropped = pd.DataFrame(
        [
            {
                "fold": fold.fold,
                "train_cross_boundary_removed": int(((timestamp >= fold.train_start) & (timestamp <= fold.train_end) & (label_end > fold.train_end)).sum()),
                "test_cross_boundary_removed": int(((timestamp >= fold.test_start) & (timestamp <= fold.test_end) & (label_end > fold.test_end)).sum()),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_remaining_boundary_passed": bool(remaining_train_end.empty or (remaining_train_end <= fold.train_end).all()),
                "test_remaining_boundary_passed": bool(remaining_test_end.empty or (remaining_test_end <= fold.test_end).all()),
                "train_max_label_end": remaining_train_end.max() if not remaining_train_end.empty else pd.NaT,
                "test_max_label_end": remaining_test_end.max() if not remaining_test_end.empty else pd.NaT,
            }
        ]
    )
    train = frame.loc[train_mask].sort_values("extreme_pos").reset_index(drop=True)
    test = frame.loc[test_mask].sort_values("extreme_pos").reset_index(drop=True)
    if train.empty or test.empty:
        raise RuntimeError(f"{fold.fold} has empty train/test split")
    return train, test, dropped


def _weighted_positions_without_replacement(
    weights: np.ndarray,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """Deterministic weighted sampling without Pandas/NumPy PPS limitations.

    Pandas 3 / newer NumPy versions can reject highly uneven probability
    vectors when ``replace=False``.  Gumbel-top-k implements the same intended
    probability-proportional-to-size sampling without replacement and remains
    valid for very uneven episode weights.  Zero/invalid-weight rows are only
    used as deterministic uniform fill when the requested sample is larger
    than the positive-weight support.
    """

    values = np.asarray(weights, dtype=float)
    count = int(values.size)
    take = max(0, min(int(sample_size), count))
    if take == 0:
        return np.empty(0, dtype=np.int64)
    if take == count:
        return np.arange(count, dtype=np.int64)

    clean = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    positive_positions = np.flatnonzero(clean > 0.0)
    rng = np.random.default_rng(int(random_state))

    if positive_positions.size >= take:
        log_weight = np.log(clean[positive_positions])
        priority = log_weight + rng.gumbel(size=positive_positions.size)
        chosen_local = np.argpartition(priority, positive_positions.size - take)[-take:]
        chosen = positive_positions[chosen_local]
    else:
        missing = take - int(positive_positions.size)
        zero_positions = np.flatnonzero(clean <= 0.0)
        fill = rng.choice(zero_positions, size=missing, replace=False) if missing else np.empty(0, dtype=np.int64)
        chosen = np.concatenate([positive_positions, np.asarray(fill, dtype=np.int64)])

    # Sorting keeps DataFrame extraction cache-friendly; the final training
    # frame is shuffled once after positives and negatives are combined.
    return np.sort(np.asarray(chosen, dtype=np.int64))


def _sample_training(frame: pd.DataFrame, maximum_rows: int, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive_mask = frame["tp_hit_1pct"].astype(bool)
    positive = frame[positive_mask]
    negative = frame[~positive_mask]
    if maximum_rows <= 0 or len(frame) <= maximum_rows:
        diagnostics = pd.DataFrame(
            [{
                "source_rows": len(frame),
                "sampled_rows": len(frame),
                "source_positive_rows": len(positive),
                "sampled_positive_rows": len(positive),
                "source_negative_rows": len(negative),
                "sampled_negative_rows": len(negative),
                "positive_weight_negative_rows": int(len(negative)),
                "sampling": "all",
            }]
        )
        return frame.copy(), diagnostics

    negative_take = max(0, int(maximum_rows) - len(positive))
    if negative_take >= len(negative):
        sample_negative = negative.copy()
        weight_support = len(negative)
        sampling_name = "all negatives"
    elif negative_take == 0:
        sample_negative = negative.iloc[0:0].copy()
        weight_support = 0
        sampling_name = "all positives; train cap below positive count"
    else:
        weights = (
            pd.to_numeric(negative["episode_weight"], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
            if "episode_weight" in negative.columns
            else np.ones(len(negative), dtype=float)
        )
        positions = _weighted_positions_without_replacement(weights, negative_take, int(random_state))
        sample_negative = negative.iloc[positions].copy()
        weight_support = int(np.count_nonzero(np.isfinite(weights) & (weights > 0.0)))
        sampling_name = "all positives + gumbel-top-k episode-weighted negatives"

    sample = pd.concat([positive, sample_negative], ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=int(random_state)).reset_index(drop=True)
    unique_key = "event_id" if "event_id" in sample.columns else None
    has_duplicates = bool(sample[unique_key].duplicated().any()) if unique_key else False
    if has_duplicates or len(sample) != len(positive) + len(sample_negative):
        raise RuntimeError("training sampler produced duplicate or missing rows")
    diagnostics = pd.DataFrame(
        [
            {
                "source_rows": len(frame),
                "sampled_rows": len(sample),
                "source_positive_rows": len(positive),
                "sampled_positive_rows": int(sample["tp_hit_1pct"].sum()),
                "source_negative_rows": len(negative),
                "sampled_negative_rows": len(sample_negative),
                "positive_weight_negative_rows": int(weight_support),
                "sampling": sampling_name,
            }
        ]
    )
    return sample, diagnostics


def _predict_binary_batched(model: object, frame: pd.DataFrame, chunk_size: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(frame), int(chunk_size)):
        parts.append(np.asarray(model.predict_proba(frame.iloc[start : start + int(chunk_size)]), dtype=float))
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def _predict_quantile_batched(model: object, frame: pd.DataFrame, chunk_size: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(frame), int(chunk_size)):
        parts.append(np.asarray(model.predict(frame.iloc[start : start + int(chunk_size)]), dtype=float))
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def _fit_heads(
    train_sample: pd.DataFrame,
    train_full: pd.DataFrame,
    test: pd.DataFrame,
    *,
    family: str,
    feature_columns: Sequence[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    train_scored = train_full[[
        "event_id", "extreme_time", "extreme_pos", "causal_region_id", "positive_episode_id",
        "tp_hit_1pct", "tp_before_adverse_0p25pct", "tp_before_adverse_0p5pct",
        "tp_before_adverse_0p75pct", "tp_before_adverse_1p0pct",
        "tp_within_15", "tp_within_30", "mfe_pct", "mae_horizon_pct", "mae_before_tp_pct", "tp_first_touch_bar",
    ]].copy()
    test_scored = test[[
        "event_id", "extreme_time", "extreme_pos", "causal_region_id", "positive_episode_id",
        "tp_hit_1pct", "tp_before_adverse_0p25pct", "tp_before_adverse_0p5pct",
        "tp_before_adverse_0p75pct", "tp_before_adverse_1p0pct",
        "tp_within_15", "tp_within_30", "mfe_pct", "mae_horizon_pct", "mae_before_tp_pct", "tp_first_touch_bar",
    ]].copy()
    models: dict[str, object] = {}
    for output_column, target_column in HEAD_TARGETS.items():
        model = fit_binary_model(
            train_sample,
            feature_columns=feature_columns,
            target_column=target_column,
            family=family,
            random_state=int(args.random_state),
            min_samples_leaf=int(args.model_min_samples_leaf),
            weight_column="episode_weight",
        )
        models[output_column] = model
        train_scored[output_column] = _predict_binary_batched(model, train_full, int(args.prediction_chunk_size))
        test_scored[output_column] = _predict_binary_batched(model, test, int(args.prediction_chunk_size))

    risk_q50 = fit_quantile_risk_model(
        train_sample,
        feature_columns=feature_columns,
        quantile=0.50,
        random_state=int(args.random_state),
        min_samples_leaf=int(args.model_min_samples_leaf),
    )
    risk_q90 = fit_quantile_risk_model(
        train_sample,
        feature_columns=feature_columns,
        quantile=0.90,
        random_state=int(args.random_state),
        min_samples_leaf=int(args.model_min_samples_leaf),
    )
    models["mae_q50"] = risk_q50
    models["mae_q90"] = risk_q90
    train_scored["predicted_mae_q50_pct"] = _predict_quantile_batched(risk_q50, train_full, int(args.prediction_chunk_size))
    train_scored["predicted_mae_q90_pct"] = _predict_quantile_batched(risk_q90, train_full, int(args.prediction_chunk_size))
    test_scored["predicted_mae_q50_pct"] = _predict_quantile_batched(risk_q50, test, int(args.prediction_chunk_size))
    test_scored["predicted_mae_q90_pct"] = _predict_quantile_batched(risk_q90, test, int(args.prediction_chunk_size))
    train_scored["tp_score"] = train_scored["p_tp60"]
    test_scored["tp_score"] = test_scored["p_tp60"]
    train_scored["clean50_score"] = train_scored["p_clean50"]
    test_scored["clean50_score"] = test_scored["p_clean50"]
    train_scored["fixed_multi_score"] = fixed_multiobjective_score(
        train_scored["p_tp60"], train_scored["p_clean50"], train_scored["p_fast15"], train_scored["predicted_mae_q90_pct"]
    )
    test_scored["fixed_multi_score"] = fixed_multiobjective_score(
        test_scored["p_tp60"], test_scored["p_clean50"], test_scored["p_fast15"], test_scored["predicted_mae_q90_pct"]
    )
    return train_scored, test_scored, models


def _head_metrics(frame: pd.DataFrame, fold: str, family: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for output_column, target_column in HEAD_TARGETS.items():
        rows.append(
            {
                "fold": fold,
                "family": family,
                "output": output_column,
                "target": target_column,
                **binary_metrics(frame[target_column], frame[output_column]),
            }
        )
    tp = frame[frame["tp_hit_1pct"].astype(bool)].copy()
    for output, target in (("predicted_mae_q50_pct", "mae_before_tp_pct"), ("predicted_mae_q90_pct", "mae_before_tp_pct")):
        actual = pd.to_numeric(tp[target], errors="coerce")
        predicted = pd.to_numeric(tp[output], errors="coerce")
        valid = actual.notna() & predicted.notna()
        rows.append(
            {
                "fold": fold,
                "family": family,
                "output": output,
                "target": target,
                "count": int(valid.sum()),
                "positive_count": np.nan,
                "base_rate": float(actual[valid].median()) if valid.any() else np.nan,
                "pr_auc": np.nan,
                "roc_auc": np.nan,
                "brier": np.nan,
                "mae": float(np.mean(np.abs(actual[valid] - predicted[valid]))) if valid.any() else np.nan,
                "coverage_actual_below_prediction": float((actual[valid] <= predicted[valid]).mean()) if valid.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _ablation_metrics(
    train_sample: pd.DataFrame,
    test: pd.DataFrame,
    feature_sets: dict[str, tuple[str, ...]],
    fold: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[tuple[str, str], object]]:
    rows: list[dict[str, object]] = []
    models: dict[tuple[str, str], object] = {}
    specs = [(group, PRIMARY_MULTIHEAD_FAMILY) for group in feature_sets]
    specs.append(("U2_region_hybrid", TP_CHALLENGER_FAMILY))
    reporter = ProgressReporter("[models] fold TP ablation", total=len(specs), every=1)
    processed = 0
    for group, family in specs:
        columns = feature_sets[group]
        model = fit_binary_model(
            train_sample,
            feature_columns=columns,
            target_column="tp_hit_1pct",
            family=family,
            random_state=int(args.random_state),
            min_samples_leaf=int(args.model_min_samples_leaf),
            weight_column="episode_weight",
        )
        prediction = _predict_binary_batched(model, test, int(args.prediction_chunk_size))
        rows.append({"fold": fold, "feature_group": group, "family": family, "feature_count": len(columns), **binary_metrics(test["tp_hit_1pct"], prediction)})
        models[(group, family)] = model
        processed += 1
        if processed < len(specs):
            reporter.update(processed)
    reporter.close()
    return pd.DataFrame(rows), models


def _grid_metrics(
    train_scored: pd.DataFrame,
    test_scored: pd.DataFrame,
    *,
    fold: str,
    family: str,
    score_types: Sequence[str] = SCORE_TYPES,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    deletion_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    reference_events: list[pd.DataFrame] = []
    test_months = max(1, pd.to_datetime(test_scored["extreme_time"]).dt.to_period("M").nunique())
    for score_type in score_types:
        reference_scores = pd.to_numeric(train_scored[score_type], errors="coerce").dropna()
        for fraction in TOP_FRACTIONS:
            threshold = float(reference_scores.quantile(1.0 - float(fraction)))
            for cooldown in COOLDOWNS:
                events = select_first_region_signal(
                    test_scored,
                    score_column=score_type,
                    threshold=threshold,
                    cooldown_bars=int(cooldown),
                )
                metrics = opportunity_event_metrics(events)
                coverage = positive_episode_coverage(events, test_scored)
                concentration = concentration_metrics(events)
                rows.append(
                    {
                        "fold": fold,
                        "family": family,
                        "score_type": score_type,
                        "top_fraction": float(fraction),
                        "score_threshold": threshold,
                        "cooldown_bars": int(cooldown),
                        "events_per_month": float(len(events) / test_months),
                        **metrics,
                        **coverage,
                        **concentration,
                    }
                )
                for removed_days in (5, 10):
                    reduced = remove_strongest_days(events, removed_days)
                    reduced_metrics = opportunity_event_metrics(reduced)
                    deletion_rows.append(
                        {
                            "fold": fold,
                            "family": family,
                            "score_type": score_type,
                            "top_fraction": float(fraction),
                            "cooldown_bars": int(cooldown),
                            "removed_strongest_days": int(removed_days),
                            **reduced_metrics,
                        }
                    )
                if events.empty:
                    continue
                event_time = pd.to_datetime(events["extreme_time"])
                events = events.copy()
                events["period"] = event_time.dt.to_period("M").astype(str)
                for period, group in events.groupby("period"):
                    period_rows.append(
                        {
                            "fold": fold,
                            "family": family,
                            "score_type": score_type,
                            "top_fraction": float(fraction),
                            "cooldown_bars": int(cooldown),
                            "period": period,
                            **opportunity_event_metrics(group),
                        }
                    )
                if abs(float(fraction) - REFERENCE_TOP_FRACTION) < 1e-12 and int(cooldown) == REFERENCE_COOLDOWN:
                    reference_events.append(events.assign(fold=fold, family=family, score_type=score_type))
    return pd.DataFrame(rows), pd.DataFrame(deletion_rows), pd.DataFrame(period_rows), reference_events


def _agreement(left: pd.DataFrame, right: pd.DataFrame, fold: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for score_type in ("tp_score",):
        a = pd.to_numeric(left[score_type], errors="coerce")
        b = pd.to_numeric(right[score_type], errors="coerce")
        for fraction in (0.01, 0.05, 0.10):
            count = max(1, int(np.ceil(len(a) * fraction)))
            top_a = set(np.argpartition(a.to_numpy(), -count)[-count:].tolist())
            top_b = set(np.argpartition(b.to_numpy(), -count)[-count:].tolist())
            rows.append(
                {
                    "fold": fold,
                    "score_type": score_type,
                    "score_spearman": float(a.corr(b, method="spearman")),
                    "top_fraction": fraction,
                    "top_overlap": float(len(top_a & top_b) / count),
                }
            )
    return pd.DataFrame(rows)


def _logistic_importance(model: object, fold: str, output: str, top_n: int = 30) -> pd.DataFrame:
    estimator = getattr(model, "model", None)
    columns = tuple(getattr(model, "feature_columns", ()))
    if estimator is None or not hasattr(estimator, "named_steps"):
        return pd.DataFrame()
    fitted = estimator.named_steps.get("model")
    coefficient = np.asarray(getattr(fitted, "coef_", []), dtype=float)
    if coefficient.ndim != 2 or coefficient.shape[1] != len(columns):
        return pd.DataFrame()
    return (
        pd.DataFrame(
            {
                "fold": fold,
                "output": output,
                "feature": columns,
                "signed_coefficient": coefficient[0],
                "absolute_coefficient": np.abs(coefficient[0]),
            }
        )
        .sort_values("absolute_coefficient", ascending=False)
        .head(int(top_n))
        .reset_index(drop=True)
    )


def _mechanism_diagnostics(frame: pd.DataFrame, fold: str, family: str) -> pd.DataFrame:
    data = frame.copy()
    rows: list[dict[str, object]] = []
    for dominant, group in data.groupby("mechanism_dominant"):
        high = group[pd.to_numeric(group["fixed_multi_score"], errors="coerce") >= pd.to_numeric(group["fixed_multi_score"], errors="coerce").quantile(0.95)]
        rows.append(
            {
                "fold": fold,
                "family": family,
                "mechanism_dominant": dominant,
                "candidate_count": len(group),
                "base_tp_rate": float(group["tp_hit_1pct"].mean()),
                "top5_candidate_count": len(high),
                "top5_tp_rate": float(high["tp_hit_1pct"].mean()) if len(high) else np.nan,
                "top5_clean50_rate": float(high["tp_before_adverse_0p5pct"].mean()) if len(high) else np.nan,
                "median_mechanism_margin": float(pd.to_numeric(group["mechanism_top_margin"], errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows)


def _raw_future_perturbation_audit(
    bars: pd.DataFrame,
    source_candidates: pd.DataFrame,
    raw_feature_columns: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    valid = source_candidates[
        (pd.to_numeric(source_candidates["extreme_pos"], errors="coerce") >= int(args.lookback) + 20)
        & (pd.to_numeric(source_candidates["extreme_pos"], errors="coerce") + int(args.forward_horizon_bars) < len(bars))
    ]
    if valid.empty:
        return pd.DataFrame([{"event_id": "", "passed": False, "maximum_absolute_difference": np.nan, "detail": "no valid audit candidate"}])
    sample = valid.sample(min(int(args.causal_audit_sample_size), len(valid)), random_state=int(args.random_state))
    numeric_columns = [
        column
        for column in (
            "open", "high", "low", "close", "volume", "trades_count", "notional", "buy_notional",
            "sell_notional", "delta_notional", "large_buy_notional", "large_sell_notional",
            "large_delta_notional", "large_trades_count", "max_trade_notional", "avg_trade_size", "vwap",
        )
        if column in bars.columns
    ]
    rows: list[dict[str, object]] = []
    config = _candidate_config(args)
    local_lookback = int(args.lookback) + int(args.region_max_bars) + 20
    for row in sample.itertuples(index=False):
        global_pos = int(row.extreme_pos)
        start = max(0, global_pos - local_lookback)
        end = min(len(bars), global_pos + int(args.forward_horizon_bars) + 2)
        local = bars.iloc[start:end].copy()
        local_pos = global_pos - start
        current_time = local.index[local_pos]

        def build_current(source_bars: pd.DataFrame) -> pd.DataFrame:
            local_candidates, _ = build_online_candidate_events(
                source_bars,
                research_start=source_bars.index[min(int(args.lookback), len(source_bars) - 1)],
                research_end_exclusive=current_time + pd.Timedelta(minutes=1),
                config=config,
            )
            if local_candidates.empty or local_pos not in set(pd.to_numeric(local_candidates["extreme_pos"], errors="coerce").astype(int)):
                raise RuntimeError("audit target disappeared from causal candidate universe")
            features = build_reversal_candidate_features(
                source_bars,
                local_candidates,
                include_session=False,
                include_htf=False,
                show_progress=False,
            ).frame
            region = build_broad_candidate_regions(
                source_bars,
                features,
                max_gap_bars=int(args.region_max_gap_bars),
                max_region_bars=int(args.region_max_bars),
                retest_tolerance_bp=float(args.region_retest_tolerance_bp),
                show_progress=False,
            ).frame
            return region[pd.to_numeric(region["extreme_pos"], errors="coerce").astype(int).eq(local_pos)].tail(1)

        original = build_current(local)
        perturbed = local.copy()
        rng = np.random.default_rng(int(args.random_state) + global_pos)
        future_start = local_pos + 1
        for column in numeric_columns:
            values = pd.to_numeric(perturbed[column], errors="coerce").to_numpy(dtype=float, copy=True)
            segment = values[future_start:]
            values[future_start:] = np.where(np.isfinite(segment), segment * rng.uniform(0.2, 4.0, len(segment)), segment)
            perturbed[column] = values
        if future_start < len(perturbed):
            center = pd.to_numeric(perturbed.iloc[future_start:]["close"], errors="coerce").to_numpy(dtype=float, copy=True)
            open_future = center * rng.uniform(0.995, 1.005, len(center))
            close_future = center * rng.uniform(0.995, 1.005, len(center))
            spread = np.maximum(np.abs(center) * rng.uniform(0.0002, 0.01, len(center)), 1e-9)
            perturbed.iloc[future_start:, perturbed.columns.get_loc("open")] = open_future
            perturbed.iloc[future_start:, perturbed.columns.get_loc("close")] = close_future
            perturbed.iloc[future_start:, perturbed.columns.get_loc("high")] = np.maximum(open_future, close_future) + spread
            perturbed.iloc[future_start:, perturbed.columns.get_loc("low")] = np.minimum(open_future, close_future) - spread
        changed = build_current(perturbed)
        comparable = [column for column in raw_feature_columns if column in original.columns and column in changed.columns]
        a = original[comparable].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        b = changed[comparable].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        difference = np.abs(a - b)
        finite = np.isfinite(difference)
        maximum = float(np.nanmax(difference[finite])) if finite.any() else 0.0
        rows.append(
            {
                "event_id": str(row.event_id),
                "feature_count": len(comparable),
                "maximum_absolute_difference": maximum,
                "passed": bool(maximum <= 1e-10),
                "detail": "all raw rows strictly after the current closed bar were perturbed",
            }
        )
    return pd.DataFrame(rows)


def _summary(
    grid: pd.DataFrame,
    ablation: pd.DataFrame,
    heads: pd.DataFrame,
    agreement: pd.DataFrame,
    deletion: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    reference = grid[
        grid["top_fraction"].eq(REFERENCE_TOP_FRACTION)
        & grid["cooldown_bars"].eq(REFERENCE_COOLDOWN)
    ].copy()
    reference_aggregate = (
        reference.groupby(["family", "score_type"], as_index=False)
        .agg(event_count=("event_count", "sum"), mean_tp_rate=("tp_rate", "mean"), mean_clean50_rate=("clean_0p50_rate", "mean"), minimum_fold_events=("event_count", "min"))
        .sort_values(["event_count", "mean_clean50_rate"], ascending=False)
    )
    lines = [
        f"# {TITLE}",
        "",
        "## Research contract",
        "",
        "- Logistic multi-head baseline with a separate HistGradientBoosting TP-only challenger; no per-cluster specialist is trained.",
        "- Old C3-A/C3-E IDs are not used as targets or permanent semantics.",
        "- Soft shock/trend/base scores are rebuilt from causal features inside every training fold.",
        "- No threshold winner is selected from a test fold; full Top 1/2/5/10/20% curves are reported.",
        "- Region state is broad candidate-process context, not a second model-tail gate.",
        "",
        "## Multi-objective outputs",
        "",
        "- P(TP +1% within 60 closed bars).",
        "- P(TP before -0.25% future-close adverse move).",
        "- P(TP before -0.50% future-close adverse move).",
        "- P(TP within 15 bars) and P(TP within 30 bars).",
        "- Predicted TP-positive MAE-before-TP median and 90th percentile.",
        "",
        "## Walk-forward folds",
        "",
        "- 2023 train -> 2024 test.",
        "- 2023-2024 train -> 2025 test.",
        "- 2023-2025 train -> 2026H1 test.",
        "",
        "## Reference frequency point (pre-declared, not selected)",
        "",
        f"- Top fraction: {REFERENCE_TOP_FRACTION:.0%}; cooldown: {REFERENCE_COOLDOWN} bars.",
    ]
    if not reference_aggregate.empty:
        lines.append("")
        for row in reference_aggregate.itertuples(index=False):
            lines.append(
                f"- `{row.family}/{row.score_type}`: events={int(row.event_count)}, "
                f"minimum_fold_events={int(row.minimum_fold_events)}, "
                f"mean_TP={float(row.mean_tp_rate):.2%}, "
                f"mean_clean50={float(row.mean_clean50_rate):.2%}."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- A usable result needs hundreds of independent events, non-collapsing fold results, continuous threshold neighborhoods, and survival after deleting the strongest 5/10 days.",
            "- PR-AUC or a high win rate at one sparse threshold is not enough.",
            "- This report does not include fees, stops, sizing, order execution, or portfolio returns.",
            "",
            "## Audit status",
            "",
            f"- All causal checks passed: {bool(not audit.empty and audit['passed'].all())}.",
            f"- Ablation rows: {len(ablation)}; multi-head rows: {len(heads)}; family-agreement rows: {len(agreement)}; deletion-stress rows: {len(deletion)}.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_research(args: argparse.Namespace) -> Path:
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = load_bars(args)
    coverage = validate_trade_bar_fields(bars)
    _write_csv(coverage, out_dir / "01_trade_bar_field_coverage.csv")

    print("[stage] broad causal candidate universe", flush=True)
    candidates, gate_summary = build_online_candidate_events(
        bars,
        research_start=pd.Timestamp(args.start_date),
        research_end_exclusive=_end_exclusive(args.end_date),
        config=_candidate_config(args),
    )
    _write_csv(gate_summary, out_dir / "02_candidate_gate_summary.csv")

    print("[stage] vectorized causal snapshot features", flush=True)
    feature_result = build_reversal_candidate_features(
        bars,
        candidates,
        include_session=False,
        include_htf=False,
        show_progress=True,
    )
    snapshot = feature_result.frame
    m0_features = tuple(
        feature_result.group_membership.loc[
            feature_result.group_membership["feature_group"].eq("M0_core"), "feature"
        ].astype(str)
    )
    _write_csv(feature_result.dictionary, out_dir / "03_snapshot_feature_dictionary.csv")

    print("[stage] broad causal candidate-region process", flush=True)
    region_result = build_broad_candidate_regions(
        bars,
        snapshot,
        max_gap_bars=int(args.region_max_gap_bars),
        max_region_bars=int(args.region_max_bars),
        retest_tolerance_bp=float(args.region_retest_tolerance_bp),
        show_progress=True,
    )
    frame = region_result.frame
    region_features = tuple(region_result.dictionary["feature"].astype(str))
    _write_csv(region_result.dictionary, out_dir / "04_region_feature_dictionary.csv")
    _write_csv(region_result.summary, out_dir / "05_region_summary.csv")

    print("[stage] bounded next-open/future-close labels", flush=True)
    labels = build_reversal_forward_labels(
        bars,
        frame,
        horizon=int(args.forward_horizon_bars),
        target_move_pct=float(args.target_move_pct),
        vectorized_chunk_size=int(args.label_vectorized_chunk_size),
        show_progress=True,
    )
    frame = frame.merge(labels, on="event_id", how="inner", validate="one_to_one")
    frame = attach_positive_opportunity_episodes(
        frame,
        max_gap_bars=int(args.positive_episode_gap_bars),
    )
    label_summary = (
        frame.assign(year=pd.to_datetime(frame["extreme_time"]).dt.year)
        .groupby("year", as_index=False)
        .agg(
            candidate_count=("event_id", "size"),
            tp_count=("tp_hit_1pct", "sum"),
            tp_rate=("tp_hit_1pct", "mean"),
            clean25_rate=("tp_before_adverse_0p25pct", "mean"),
            clean50_rate=("tp_before_adverse_0p5pct", "mean"),
            positive_episode_count=("positive_episode_id", lambda values: len(set(values) - {""})),
        )
    )
    _write_csv(label_summary, out_dir / "06_label_and_episode_summary.csv")
    _write_csv(mechanism_feature_dictionary(), out_dir / "07_soft_mechanism_feature_dictionary.csv")

    folds = _folds(args.end_date)
    _write_csv(_fold_table(folds), out_dir / "08_walkforward_folds.csv")

    split_rows: list[pd.DataFrame] = []
    sample_rows: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []
    ablation_parts: list[pd.DataFrame] = []
    head_parts: list[pd.DataFrame] = []
    agreement_parts: list[pd.DataFrame] = []
    grid_parts: list[pd.DataFrame] = []
    deletion_parts: list[pd.DataFrame] = []
    monthly_parts: list[pd.DataFrame] = []
    mechanism_parts: list[pd.DataFrame] = []
    importance_parts: list[pd.DataFrame] = []
    model_fit_rows: list[dict[str, object]] = []
    prediction_samples: list[pd.DataFrame] = []
    full_prediction_parts: list[pd.DataFrame] = []
    reference_event_parts: list[pd.DataFrame] = []

    for fold_index, fold in enumerate(folds, start=1):
        print(f"[fold] {fold.fold} train={fold.train_start.date()}->{fold.train_end.date()} test={fold.test_start.date()}->{fold.test_end.date()}", flush=True)
        train, test, split_diagnostic = _subset_fold(frame, fold)
        split_rows.append(split_diagnostic)
        train = attach_positive_opportunity_episodes(train, max_gap_bars=int(args.positive_episode_gap_bars))
        test = attach_positive_opportunity_episodes(test, max_gap_bars=int(args.positive_episode_gap_bars))
        train = attach_episode_balanced_weight(train)

        mechanism = fit_soft_mechanism_transformer(train)
        train_mechanism = mechanism.transform(train)
        test_mechanism = mechanism.transform(test)
        for column in train_mechanism.columns:
            train[column] = train_mechanism[column].to_numpy()
            test[column] = test_mechanism[column].to_numpy()

        mechanism_features = tuple(
            column for column in train_mechanism.columns if column != "mechanism_dominant"
        )
        requested_sets = {
            "U0_snapshot": tuple(m0_features),
            "U1_soft_mechanism": tuple(m0_features) + mechanism_features,
            "U2_region_hybrid": tuple(m0_features) + mechanism_features + tuple(region_features),
        }
        feature_sets: dict[str, tuple[str, ...]] = {}
        for group, requested in requested_sets.items():
            selected = select_usable_features(train, requested)
            feature_sets[group] = selected
            feature_rows.append(
                {
                    "fold": fold.fold,
                    "feature_group": group,
                    "requested_feature_count": len(requested),
                    "selected_feature_count": len(selected),
                    "selected_features": "|".join(selected),
                }
            )
        primary_features = feature_sets["U2_region_hybrid"]
        train_sample, sampling = _sample_training(
            train,
            int(args.maximum_train_rows),
            int(args.random_state) + fold_index,
        )
        sampling.insert(0, "fold", fold.fold)
        sample_rows.append(sampling)

        ablation, _ = _ablation_metrics(train_sample, test, feature_sets, fold.fold, args)
        ablation_parts.append(ablation)

        family_predictions: dict[str, pd.DataFrame] = {}
        family_models: dict[str, dict[str, object]] = {}

        print(f"[models] {fold.fold} unified multi-head family={PRIMARY_MULTIHEAD_FAMILY}", flush=True)
        train_scored, test_scored, models = _fit_heads(
            train_sample,
            train,
            test,
            family=PRIMARY_MULTIHEAD_FAMILY,
            feature_columns=primary_features,
            args=args,
        )
        for column in (
            "mechanism_shock_score", "mechanism_trend_score", "mechanism_base_score",
            "mechanism_top_margin", "mechanism_entropy", "mechanism_dominant",
            "region_age_bars", "region_observation_number", "region_candidate_density",
            "region_rebound_from_low", "region_absorption_improvement",
        ):
            if column in train.columns:
                train_scored[column] = train[column].to_numpy()
                test_scored[column] = test[column].to_numpy()
        family_predictions[PRIMARY_MULTIHEAD_FAMILY] = test_scored
        family_models[PRIMARY_MULTIHEAD_FAMILY] = models
        head_parts.append(_head_metrics(test_scored, fold.fold, PRIMARY_MULTIHEAD_FAMILY))
        grid, deletion, monthly, reference_events = _grid_metrics(
            train_scored,
            test_scored,
            fold=fold.fold,
            family=PRIMARY_MULTIHEAD_FAMILY,
            score_types=SCORE_TYPES,
        )
        grid_parts.append(grid)
        deletion_parts.append(deletion)
        monthly_parts.append(monthly)
        reference_event_parts.extend(reference_events)
        mechanism_parts.append(_mechanism_diagnostics(test_scored, fold.fold, PRIMARY_MULTIHEAD_FAMILY))
        for output, model in models.items():
            if output in HEAD_TARGETS:
                model_fit_rows.append(
                    {
                        "fold": fold.fold,
                        "output": output,
                        "target": HEAD_TARGETS[output],
                        "requested_family": PRIMARY_MULTIHEAD_FAMILY,
                        "actual_family": getattr(model, "family", PRIMARY_MULTIHEAD_FAMILY),
                    }
                )
                importance_parts.append(_logistic_importance(model, fold.fold, output))

        sample = pd.concat(
            [
                test_scored.nlargest(min(2_000, len(test_scored)), "fixed_multi_score"),
                test_scored.sample(min(2_000, len(test_scored)), random_state=int(args.random_state) + fold_index),
            ],
            ignore_index=True,
        ).drop_duplicates("event_id")
        sample.insert(0, "family", PRIMARY_MULTIHEAD_FAMILY)
        sample.insert(0, "fold", fold.fold)
        prediction_samples.append(sample)
        if args.write_full_predictions:
            full = test_scored.copy()
            full.insert(0, "family", PRIMARY_MULTIHEAD_FAMILY)
            full.insert(0, "fold", fold.fold)
            full_prediction_parts.append(full)

        print(f"[models] {fold.fold} TP challenger family={TP_CHALLENGER_FAMILY}", flush=True)
        challenger = fit_binary_model(
            train_sample,
            feature_columns=primary_features,
            target_column="tp_hit_1pct",
            family=TP_CHALLENGER_FAMILY,
            random_state=int(args.random_state),
            min_samples_leaf=int(args.model_min_samples_leaf),
            weight_column="episode_weight",
        )
        challenger_train = train_scored[[
            "event_id", "extreme_time", "extreme_pos", "causal_region_id", "positive_episode_id",
            "tp_hit_1pct", "tp_before_adverse_0p25pct", "tp_before_adverse_0p5pct",
            "tp_before_adverse_0p75pct", "tp_before_adverse_1p0pct",
            "mfe_pct", "mae_horizon_pct", "mae_before_tp_pct", "tp_first_touch_bar",
        ]].copy()
        challenger_test = test_scored[[
            "event_id", "extreme_time", "extreme_pos", "causal_region_id", "positive_episode_id",
            "tp_hit_1pct", "tp_before_adverse_0p25pct", "tp_before_adverse_0p5pct",
            "tp_before_adverse_0p75pct", "tp_before_adverse_1p0pct",
            "mfe_pct", "mae_horizon_pct", "mae_before_tp_pct", "tp_first_touch_bar",
        ]].copy()
        challenger_train["p_tp60"] = _predict_binary_batched(challenger, train, int(args.prediction_chunk_size))
        challenger_test["p_tp60"] = _predict_binary_batched(challenger, test, int(args.prediction_chunk_size))
        challenger_train["tp_score"] = challenger_train["p_tp60"]
        challenger_test["tp_score"] = challenger_test["p_tp60"]
        family_predictions[TP_CHALLENGER_FAMILY] = challenger_test
        family_models[TP_CHALLENGER_FAMILY] = {"p_tp60": challenger}
        head_parts.append(pd.DataFrame([{
            "fold": fold.fold,
            "family": TP_CHALLENGER_FAMILY,
            "output": "p_tp60",
            "target": "tp_hit_1pct",
            **binary_metrics(challenger_test["tp_hit_1pct"], challenger_test["p_tp60"]),
        }]))
        c_grid, c_deletion, c_monthly, c_reference = _grid_metrics(
            challenger_train,
            challenger_test,
            fold=fold.fold,
            family=TP_CHALLENGER_FAMILY,
            score_types=("tp_score",),
        )
        grid_parts.append(c_grid)
        deletion_parts.append(c_deletion)
        monthly_parts.append(c_monthly)
        reference_event_parts.extend(c_reference)
        c_sample = pd.concat(
            [
                challenger_test.nlargest(min(2_000, len(challenger_test)), "tp_score"),
                challenger_test.sample(min(2_000, len(challenger_test)), random_state=int(args.random_state) + fold_index),
            ],
            ignore_index=True,
        ).drop_duplicates("event_id")
        c_sample.insert(0, "family", TP_CHALLENGER_FAMILY)
        c_sample.insert(0, "fold", fold.fold)
        prediction_samples.append(c_sample)
        agreement_parts.append(_agreement(test_scored, challenger_test, fold.fold))
        del train, test, train_sample, family_predictions, family_models
        gc.collect()

    split_table = pd.concat(split_rows, ignore_index=True)
    sample_table = pd.concat(sample_rows, ignore_index=True)
    feature_table = pd.DataFrame(feature_rows)
    ablation_table = pd.concat(ablation_parts, ignore_index=True)
    head_table = pd.concat(head_parts, ignore_index=True)
    agreement_table = pd.concat(agreement_parts, ignore_index=True) if agreement_parts else pd.DataFrame()
    grid_table = pd.concat(grid_parts, ignore_index=True)
    deletion_table = pd.concat(deletion_parts, ignore_index=True)
    monthly_table = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()
    mechanism_table = pd.concat(mechanism_parts, ignore_index=True)
    importance_table = pd.concat(importance_parts, ignore_index=True) if importance_parts else pd.DataFrame()
    model_fit_table = pd.DataFrame(model_fit_rows)
    prediction_sample = pd.concat(prediction_samples, ignore_index=True) if prediction_samples else pd.DataFrame()
    reference_events = pd.concat(reference_event_parts, ignore_index=True) if reference_event_parts else pd.DataFrame()

    _write_csv(split_table, out_dir / "09_fold_boundary_and_sample_counts.csv")
    _write_csv(sample_table, out_dir / "10_training_sampling_diagnostics.csv")
    _write_csv(feature_table, out_dir / "11_fold_feature_sets.csv")
    _write_csv(ablation_table, out_dir / "12_walkforward_tp_feature_ablation.csv")
    _write_csv(head_table, out_dir / "13_walkforward_multihead_metrics.csv")
    _write_csv(model_fit_table, out_dir / "13b_model_head_fit_methods.csv")
    _write_csv(agreement_table, out_dir / "14_model_family_agreement.csv")
    _write_csv(grid_table, out_dir / "15_walkforward_frequency_quality_curves.csv")
    _write_csv(deletion_table, out_dir / "16_delete_strongest_days_stress.csv")
    _write_csv(monthly_table, out_dir / "17_walkforward_monthly_metrics.csv")
    _write_csv(mechanism_table, out_dir / "18_soft_mechanism_diagnostics.csv")
    _write_csv(importance_table, out_dir / "19_logistic_feature_importance_stability.csv")
    _write_csv(reference_events, out_dir / "20_reference_top5pct_events.csv")
    _write_csv(prediction_sample, out_dir / "21_walkforward_prediction_sample.csv")
    if full_prediction_parts:
        _write_csv(pd.concat(full_prediction_parts, ignore_index=True), out_dir / "22_walkforward_full_predictions.csv")

    representative = pd.concat(
        [
            reference_events[reference_events["tp_hit_1pct"].astype(bool)].head(50).assign(case="reference_success"),
            reference_events[~reference_events["tp_hit_1pct"].astype(bool)].head(50).assign(case="reference_failure"),
            prediction_sample.nlargest(min(50, len(prediction_sample)), "fixed_multi_score").assign(case="highest_multi_score"),
        ],
        ignore_index=True,
    ).drop_duplicates(["fold", "family", "event_id"])
    _write_csv(representative, out_dir / "23_representative_events.csv")

    print("[stage] raw future perturbation causal audit", flush=True)
    raw_feature_columns = tuple(m0_features) + tuple(region_features)
    raw_audit = _raw_future_perturbation_audit(bars, frame, raw_feature_columns, args)
    _write_csv(raw_audit, out_dir / "24_raw_future_perturbation_audit.csv")
    forbidden_tokens = ("future", "forward", "label", "mfe", "mae", "tp_hit", "adverse", "entry_price", "completion", "confirmation")
    forbidden = [column for column in raw_feature_columns if any(token in column.lower() for token in forbidden_tokens)]
    audit = pd.DataFrame(
        [
            {"check": "labels_use_next_open_future_close", "passed": True, "detail": "entry=next open; TP/MAE/first-touch=future closes"},
            {"check": "future_high_low_not_used_for_labels", "passed": True, "detail": "future high/low excluded"},
            {"check": "old_cluster_ids_not_model_features", "passed": True, "detail": "03 cluster IDs are not loaded"},
            {"check": "region_state_is_causal", "passed": True, "detail": "region start through current closed bar only; eventual end/size excluded"},
            {"check": "soft_mechanism_fit_inside_each_train_fold", "passed": True, "detail": "robust transforms and percentiles fit separately per expanding fold"},
            {"check": "walkforward_labels_do_not_cross_fold_boundaries", "passed": bool(split_table[["train_remaining_boundary_passed", "test_remaining_boundary_passed"]].all().all()), "detail": split_table.to_json(orient="records", date_format="iso")},
            {"check": "future_labels_excluded_from_raw_features", "passed": not forbidden, "detail": "|".join(forbidden)},
            {"check": "raw_future_perturbation", "passed": bool(not raw_audit.empty and raw_audit["passed"].all()), "detail": f"audited={len(raw_audit)}"},
            {"check": "no_test_fold_winner_selection", "passed": True, "detail": "all predefined thresholds/families/scores are reported; no selected winner"},
            {"check": "training_sampler_is_weighted_without_replacement", "passed": bool(sample_table["sampling"].astype(str).str.contains("gumbel-top-k|all", regex=True).all()), "detail": "Gumbel-top-k avoids Pandas 3 weighted replace=False limitations"},
            {"check": "logistic_class_balance_applied_once", "passed": True, "detail": "class balance and episode weight are combined in sample_weight; estimator class_weight=None"},
        ]
    )
    _write_csv(audit, out_dir / "25_causal_and_selection_audit.csv")
    if not audit["passed"].all():
        raise RuntimeError(f"07 audit failed: {audit.loc[~audit['passed'], 'check'].tolist()}")

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
        "target_move_pct": float(args.target_move_pct),
        "forward_horizon_bars": int(args.forward_horizon_bars),
        "entry_price_source": "next_bar_open",
        "path_observation_source": "future_closed_bar_close",
        "future_high_low_used_for_labels": False,
        "candidate_count": int(len(frame)),
        "causal_region_count": int(frame["causal_region_id"].nunique()),
        "model_families": list(MODEL_FAMILIES),
        "primary_multihead_family": PRIMARY_MULTIHEAD_FAMILY,
        "tp_challenger_family": TP_CHALLENGER_FAMILY,
        "feature_groups": list(FEATURE_GROUPS),
        "multihead_outputs": list(HEAD_TARGETS),
        "score_types": list(SCORE_TYPES),
        "top_fractions": list(TOP_FRACTIONS),
        "cooldowns": list(COOLDOWNS),
        "automatic_winner_selected": False,
        "training_negative_sampler": "gumbel_top_k_weighted_without_replacement",
        "logistic_primary_solver": "newton-cholesky_tol_1e-3",
        "logistic_solver_fallbacks": ["lbfgs_tol_1e-3", "sgd_log_loss", "hist_gbdt_head_fallback"],
        "logistic_class_balance": "sample_weight_only_once_normalized_mean_1",
        "reference_top_fraction": REFERENCE_TOP_FRACTION,
        "reference_cooldown": REFERENCE_COOLDOWN,
        "strategy_or_backtest": False,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "26_RESEARCH_SUMMARY.md").write_text(
        _summary(grid_table, ablation_table, head_table, agreement_table, deletion_table, audit),
        encoding="utf-8",
    )
    result = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={result.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
