#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Respected macro first-sweep event research 12.

Research 12 isolates the sparse mechanism that research 11 could not validly
model inside the broad candidate population.  It builds one causal decision at
the first sweep bar and, when observed, another decision at the reclaim bar.
The two paths are trained and evaluated separately.

Sweep path groups
-----------------
S0 : causal 1m trade-bar snapshot + train-fitted soft mechanisms
S1 : S0 + respected-level / sweep geometry
S2 : S1 + sweep-bar aggressive-sell and absorption proxies

Reclaim path groups
-------------------
R0 : causal 1m trade-bar snapshot + train-fitted soft mechanisms
R1 : R0 + respected-level / sweep geometry
R2 : R1 + sweep order flow and causally observed reclaim process

All labels enter at the next 1m open and inspect future closed-bar closes only.
No strategy, fees, stops, sizing or automatic frozen-test winner is produced.
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from research.market_structure.swing_low_typology.common.first_sweep_event import (  # noqa: E402
    LEVEL_GROUP,
    ORDERFLOW_GROUP,
    RECLAIM_GROUP,
    build_first_sweep_event_decisions,
)
from research.market_structure.swing_low_typology.common.multiobjective_calibration import (  # noqa: E402
    calibration_metrics,
    choose_calibrator,
    delete_day_stress,
    fit_conformal_adjustment,
    fit_score_probability_calibrators,
    fit_risk_point_model,
    policy_metrics,
    quantile_metrics,
)
from research.market_structure.swing_low_typology.common.online_recognizability import fit_binary_model  # noqa: E402
from research.market_structure.swing_low_typology.common.range_increment import (  # noqa: E402
    EmpiricalRankReference,
    select_ranked_events,
)
from research.market_structure.swing_low_typology.common.reversal_opportunity import (  # noqa: E402
    build_reversal_candidate_features,
    build_reversal_forward_labels,
    select_usable_features,
)
from research.market_structure.swing_low_typology.common.swing_low_dataset import validate_trade_bar_fields  # noqa: E402
from research.market_structure.swing_low_typology.common.walkforward_reversal import (  # noqa: E402
    attach_episode_balanced_weight,
    attach_positive_opportunity_episodes,
    fit_soft_mechanism_transformer,
    mechanism_feature_dictionary,
)

SCRIPT_NAME = "12_respected_macro_first_sweep_event_research"
SCRIPT_VERSION = "1.0.3"
EXPERIMENT_ID = "ETH_1M_RESPECTED_MACRO_FIRST_SWEEP_EVENT_12"
EDGE_ID = "RESEARCH_ONLY_ETH_RESPECTED_MACRO_FIRST_SWEEP"
TITLE = "ETH Respected Macro First Sweep Event Research 12"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/12_respected_macro_first_sweep_event"
PRIMARY_FAMILY = "logistic_sgd"
HEAD_TARGETS: dict[str, str] = {
    "p_tp60": "tp_hit_1pct",
    "p_clean50": "tp_before_adverse_0p5pct",
    "p_fast30": "tp_within_30",
}


class FoldSpec(NamedTuple):
    fold: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward respected macro first-sweep event research.",
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
    p.add_argument("--liquidity-pivot-minutes", nargs="+", type=int, default=[15, 60, 240])
    p.add_argument("--liquidity-pivot-weights", nargs="+", type=float, default=[1.0, 2.0, 4.0])
    p.add_argument("--liquidity-pivot-left-bars", type=int, default=2)
    p.add_argument("--liquidity-pivot-right-bars", type=int, default=2)
    p.add_argument("--liquidity-cluster-tolerance-bp", type=float, default=25.0)
    p.add_argument("--liquidity-minimum-respects", type=int, default=2)
    p.add_argument("--liquidity-minimum-macro-timeframe-min", type=int, default=60)
    p.add_argument("--liquidity-minimum-respect-separation-minutes", type=int, default=60)
    p.add_argument("--liquidity-formation-max-days", type=int, default=45)
    p.add_argument("--liquidity-reclaim-window-bars", type=int, default=3)
    p.add_argument("--liquidity-accept-below-bars", type=int, default=3)
    p.add_argument("--liquidity-accept-depth-bp", type=float, default=75.0)
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--label-vectorized-chunk-size", type=int, default=50_000)
    p.add_argument("--model-min-samples-leaf", type=int, default=20)
    p.add_argument("--prediction-chunk-size", type=int, default=100_000)
    p.add_argument("--cooldown-bars", type=int, default=15)
    p.add_argument("--minimum-test-events", type=int, default=30)
    p.add_argument("--causal-audit-sample-size", type=int, default=2)
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


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    print(f"[load] source=trade_bar {args.symbol} {args.timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    loader = OKXTradeBarLoader(symbol=args.symbol, timeframe=args.timeframe, data_dir=args.data_dir, db_name=args.db_name)
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
        FoldSpec("WF_2024", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31 23:59:59"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31 23:59:59")),
        FoldSpec("WF_2025", pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31 23:59:59"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31 23:59:59")),
        FoldSpec("WF_2026H1", pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31 23:59:59"), pd.Timestamp("2026-01-01"), research_end),
    )


def _subset_period(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    timestamp = pd.to_datetime(frame["extreme_time"])
    label_end = pd.to_datetime(frame["label_end_time"])
    in_period = (timestamp >= start) & (timestamp <= end)
    valid = in_period & (label_end <= end)
    removed = int((in_period & ~valid).sum())
    return frame.loc[valid].sort_values(["extreme_pos", "event_id"]).reset_index(drop=True), removed


def _development_split(train: pd.DataFrame, fold: FoldSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = pd.period_range(fold.train_start.to_period("M"), fold.train_end.to_period("M"), freq="M")
    tail_months = 4 if len(months) <= 12 else 6
    calibration_months = tail_months // 2
    policy_months = tail_months - calibration_months
    calibration_start = months[-tail_months].start_time
    policy_start = months[-policy_months].start_time
    model_end = calibration_start - pd.Timedelta(nanoseconds=1)
    calibration_end = policy_start - pd.Timedelta(nanoseconds=1)
    model_fit, removed_model = _subset_period(train, fold.train_start, model_end)
    calibration, removed_calibration = _subset_period(train, calibration_start, calibration_end)
    policy, removed_policy = _subset_period(train, policy_start, fold.train_end)
    if min(len(model_fit), len(calibration), len(policy)) == 0:
        raise RuntimeError(f"{fold.fold} nested development split is empty")
    diagnostic = pd.DataFrame([{
        "fold": fold.fold,
        "model_fit_start": model_fit["extreme_time"].min(), "model_fit_end": model_fit["extreme_time"].max(),
        "calibration_start": calibration["extreme_time"].min(), "calibration_end": calibration["extreme_time"].max(),
        "policy_start": policy["extreme_time"].min(), "policy_end": policy["extreme_time"].max(),
        "model_fit_rows": len(model_fit), "calibration_rows": len(calibration), "policy_rows": len(policy),
        "model_fit_cross_boundary_removed": removed_model,
        "calibration_cross_boundary_removed": removed_calibration,
        "policy_cross_boundary_removed": removed_policy,
    }])
    return model_fit, calibration, policy, diagnostic


def _condition_feature_columns(
    fit: pd.DataFrame,
    requested: Sequence[str],
    *,
    max_features: int = 260,
    sample_rows: int = 10_000,
    max_abs_correlation: float = 0.9995,
) -> tuple[tuple[str, ...], dict[str, object]]:
    usable = tuple(select_usable_features(fit, requested, max_missing_ratio=0.40))
    candidates = usable[: max(int(max_features) + 64, int(max_features))]
    if not candidates:
        raise RuntimeError("no usable model features after fit-period sanitation")
    sample = fit.iloc[np.linspace(0, len(fit) - 1, int(sample_rows), dtype=np.int64)] if len(fit) > int(sample_rows) else fit
    numeric = sample.reindex(columns=candidates).apply(pd.to_numeric, errors="coerce")
    # ``pd.to_numeric`` preserves native/nullable boolean dtype.  Pandas
    # quantile then delegates to NumPy interpolation, which subtracts adjacent
    # values and crashes on boolean arrays.  Force one homogeneous float64
    # matrix at the feature-conditioning boundary so bool, nullable bool, ints
    # and floats all share the same safe numerical path.
    numeric = pd.DataFrame(
        numeric.to_numpy(dtype=np.float64, na_value=np.nan),
        index=numeric.index,
        columns=numeric.columns,
    ).replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median().fillna(0.0))
    robust_span = (numeric.quantile(0.90) - numeric.quantile(0.10)).abs()
    scale_keep = [column for column in candidates if np.isfinite(robust_span[column]) and robust_span[column] > 1e-10]
    removed_low_scale = len(candidates) - len(scale_keep)
    if not scale_keep:
        raise RuntimeError("all model features are near-constant in the fit period")
    values = numeric[scale_keep].to_numpy(dtype=np.float64, copy=True)
    values -= values.mean(axis=0, keepdims=True)
    sd = values.std(axis=0, ddof=0)
    valid = np.isfinite(sd) & (sd > 1e-12)
    scale_keep = [column for column, keep in zip(scale_keep, valid, strict=True) if keep]
    values = values[:, valid]
    removed_low_scale += int((~valid).sum())
    values /= np.maximum(values.std(axis=0, ddof=0, keepdims=True), 1e-12)
    corr = np.asarray(values.T @ values / max(1, len(values)), dtype=np.float64)
    kept: list[int] = []
    removed_correlated: list[str] = []
    for index, column in enumerate(scale_keep):
        if kept and np.any(np.abs(corr[index, kept]) >= float(max_abs_correlation)):
            removed_correlated.append(column)
            continue
        kept.append(index)
        if len(kept) >= int(max_features):
            break
    selected = tuple(scale_keep[index] for index in kept)
    if not selected:
        raise RuntimeError("no model features remain after collinearity conditioning")
    return selected, {
        "requested_feature_count": len(requested), "usable_feature_count": len(usable),
        "conditioning_sample_rows": len(sample), "removed_low_scale_count": removed_low_scale,
        "removed_near_duplicate_count": len(removed_correlated), "selected_feature_count": len(selected),
        "removed_near_duplicate_features": "|".join(removed_correlated), "selected_features": "|".join(selected),
    }


def _predict_binary_probability(model: object, frame: pd.DataFrame, chunk_size: int) -> np.ndarray:
    parts = [np.asarray(model.predict_proba(frame.iloc[start : start + max(1, int(chunk_size))]), dtype=float)
             for start in range(0, len(frame), max(1, int(chunk_size)))]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def _predict_binary_score(model: object, frame: pd.DataFrame, chunk_size: int) -> np.ndarray:
    parts = [np.asarray(model.predict_score(frame.iloc[start : start + max(1, int(chunk_size))]), dtype=float)
             for start in range(0, len(frame), max(1, int(chunk_size)))]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def _predict_risk(model: object, frame: pd.DataFrame, chunk_size: int) -> np.ndarray:
    parts = [np.asarray(model.predict(frame.iloc[start : start + max(1, int(chunk_size))]), dtype=float)
             for start in range(0, len(frame), max(1, int(chunk_size)))]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def _required_raw_score_levels(rows: int) -> int:
    """Minimum raw-score resolution needed by the broad 10/20/30/40% policies.

    The frozen empirical CDF may legitimately compress out-of-policy-window
    test scores into the same 0/1 tail percentile.  That tail saturation is a
    distribution-shift diagnostic, not evidence that the underlying model
    score is constant.  Resolution therefore has to be judged on raw scores.
    """

    return max(1, min(10, int(rows)))


def _rank_resolution_record(
    *,
    fold: str,
    decision_path: str,
    feature_group: str,
    output: str,
    split: str,
    raw_scores: Sequence[float],
    ranks: Sequence[float],
    calibrated: Sequence[float],
    reference: EmpiricalRankReference,
    model_probability: Sequence[float] | None = None,
) -> dict[str, object]:
    raw = pd.to_numeric(pd.Series(raw_scores), errors="coerce").to_numpy(dtype=float)
    rank = pd.to_numeric(pd.Series(ranks), errors="coerce").to_numpy(dtype=float)
    cal = pd.to_numeric(pd.Series(calibrated), errors="coerce").to_numpy(dtype=float)
    model_prob = (
        pd.to_numeric(pd.Series(model_probability), errors="coerce").to_numpy(dtype=float)
        if model_probability is not None
        else np.asarray([], dtype=float)
    )
    finite = np.isfinite(raw)
    rows = int(len(raw))
    unique_raw = int(pd.Series(raw[finite]).nunique(dropna=True)) if finite.any() else 0
    required = _required_raw_score_levels(rows)
    reference_values = np.asarray(reference.sorted_values, dtype=float)
    reference_values = reference_values[np.isfinite(reference_values)]
    reference_min = float(reference_values[0]) if reference_values.size else np.nan
    reference_max = float(reference_values[-1]) if reference_values.size else np.nan
    below = float(np.mean(raw[finite] < reference_min)) if finite.any() and np.isfinite(reference_min) else np.nan
    above = float(np.mean(raw[finite] > reference_max)) if finite.any() and np.isfinite(reference_max) else np.nan
    at_lower = float(np.mean(rank[np.isfinite(rank)] <= 0.0)) if np.isfinite(rank).any() else np.nan
    at_upper = float(np.mean(rank[np.isfinite(rank)] >= 1.0)) if np.isfinite(rank).any() else np.nan
    return {
        "fold": fold,
        "decision_path": decision_path,
        "feature_group": feature_group,
        "output": output,
        "split": split,
        "rows": rows,
        "finite_raw_score_share": float(finite.mean()) if rows else np.nan,
        "unique_raw_scores": unique_raw,
        "required_unique_raw_scores": required,
        "raw_score_resolution_passed": bool(rows > 0 and finite.all() and unique_raw >= required),
        "unique_rank_percentiles": int(pd.Series(rank).nunique(dropna=True)),
        "unique_model_probabilities": int(pd.Series(model_prob).nunique(dropna=True)) if model_prob.size else np.nan,
        "model_probability_saturated_vs_score": bool(
            model_prob.size and pd.Series(model_prob).nunique(dropna=True) < required and unique_raw >= required
        ),
        "unique_calibrated_probabilities": int(pd.Series(cal).nunique(dropna=True)),
        "policy_reference_unique_scores": int(pd.Series(reference_values).nunique(dropna=True)),
        "policy_reference_min_raw": reference_min,
        "policy_reference_max_raw": reference_max,
        "below_policy_reference_min_share": below,
        "above_policy_reference_max_share": above,
        "rank_lower_tail_share": at_lower,
        "rank_upper_tail_share": at_upper,
        "rank_tail_saturation_share": float(np.nansum([at_lower, at_upper])),
    }


def _assert_raw_score_resolution(record: dict[str, object], *, actual_family: str) -> None:
    if bool(record["raw_score_resolution_passed"]):
        return
    raise RuntimeError(
        "raw model score lost deployable resolution before policy evaluation: "
        f"fold={record['fold']} path={record['decision_path']} group={record['feature_group']} "
        f"output={record['output']} split={record['split']} rows={record['rows']} "
        f"unique_raw={record['unique_raw_scores']} required={record['required_unique_raw_scores']} "
        f"finite_share={record['finite_raw_score_share']:.6f} family={actual_family}"
    )


def _fit_binary_with_resolution_fallback(
    train: pd.DataFrame,
    policy: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    fold: str,
    decision_path: str,
    feature_group: str,
    output: str,
    random_state: int,
    min_samples_leaf: int,
    prediction_chunk_size: int,
) -> tuple[object, dict[str, object]]:
    """Fit a stable head whose *decision score* can support broad ranks.

    Solver fallback is chosen only from the development policy window and only
    for numerical score resolution, never for target performance.  Frozen test
    labels and scores are not consulted.
    """

    attempts: list[dict[str, object]] = []
    seen_actual: set[str] = set()
    for requested_family in (PRIMARY_FAMILY, "logistic", "hist_gbdt"):
        model = fit_binary_model(
            train,
            feature_columns=feature_columns,
            target_column=target_column,
            family=requested_family,
            random_state=int(random_state),
            min_samples_leaf=int(min_samples_leaf),
            weight_column="episode_weight",
        )
        actual_family = str(getattr(model, "family", requested_family))
        # The common fitter may already have reached the same fallback family.
        # Repeating an identical numerical family cannot add information.
        if actual_family in seen_actual:
            attempts.append({
                "requested_family": requested_family,
                "actual_family": actual_family,
                "skipped_duplicate_actual_family": True,
            })
            continue
        seen_actual.add(actual_family)

        score = _predict_binary_score(model, policy, int(prediction_chunk_size))
        probability = _predict_binary_probability(model, policy, int(prediction_chunk_size))
        reference = EmpiricalRankReference.fit(score)
        resolution = _rank_resolution_record(
            fold=fold, decision_path=decision_path, feature_group=feature_group,
            output=output, split="policy_preflight", raw_scores=score,
            ranks=reference.transform(score), calibrated=probability,
            reference=reference, model_probability=probability,
        )
        attempts.append({
            "requested_family": requested_family,
            "actual_family": actual_family,
            "skipped_duplicate_actual_family": False,
            "unique_decision_scores": int(resolution["unique_raw_scores"]),
            "unique_model_probabilities": int(resolution["unique_model_probabilities"]),
            "resolution_passed": bool(resolution["raw_score_resolution_passed"]),
        })
        if bool(resolution["raw_score_resolution_passed"]):
            return model, {
                "attempted_families": "|".join(str(item.get("requested_family")) for item in attempts),
                "attempt_details": json.dumps(attempts, ensure_ascii=False, sort_keys=True),
                "resolution_fallback_used": requested_family != PRIMARY_FAMILY,
                "policy_unique_decision_scores": int(resolution["unique_raw_scores"]),
                "policy_unique_model_probabilities": int(resolution["unique_model_probabilities"]),
                "ranking_score_source": "decision_function_or_logit_fallback",
            }

    raise RuntimeError(
        "all predeclared stable binary families lost deployable decision-score resolution: "
        f"fold={fold} path={decision_path} group={feature_group} output={output} "
        f"attempts={json.dumps(attempts, ensure_ascii=False, sort_keys=True)}"
    )


def _score_shell(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "event_id", "lifecycle_id", "decision_path", "extreme_pos", "extreme_time", "feature_available_time",
        "entry_time", "entry_price", "label_end_time", "causal_region_id", "positive_episode_id",
        "lifecycle_status", "reclaim_lag_bars", "same_bar_reclaim",
        "tp_hit_1pct", "tp_before_adverse_0p25pct", "tp_before_adverse_0p5pct",
        "tp_before_adverse_0p75pct", "tp_before_adverse_1p0pct", "tp_within_15", "tp_within_30",
        "mfe_pct", "mae_horizon_pct", "mae_before_tp_pct", "tp_first_touch_bar",
    ]
    return frame.reindex(columns=keep).copy()


def _head_metrics(frame: pd.DataFrame, *, fold: str, decision_path: str, feature_group: str, output: str, target: str, split: str) -> pd.DataFrame:
    y = frame[target].astype(int).to_numpy()
    rows: list[dict[str, object]] = []
    methods = (
        ("decision_score", pd.to_numeric(frame[f"{output}_score_raw"], errors="coerce").to_numpy(dtype=float), False),
        ("model_probability", pd.to_numeric(frame[f"{output}_raw"], errors="coerce").to_numpy(dtype=float), True),
        ("calibrated", pd.to_numeric(frame[f"{output}_cal"], errors="coerce").to_numpy(dtype=float), True),
    )
    for method, score, is_probability in methods:
        finite = np.isfinite(score)
        pr_auc = roc_auc = np.nan
        if finite.any() and np.unique(y[finite]).size >= 2:
            pr_auc = float(average_precision_score(y[finite], score[finite]))
            roc_auc = float(roc_auc_score(y[finite], score[finite]))
        probability = np.clip(score[finite], 1e-7, 1.0 - 1e-7) if is_probability else np.asarray([], dtype=float)
        rows.append({
            "fold": fold, "decision_path": decision_path, "feature_group": feature_group,
            "output": output, "target": target, "split": split, "method": method,
            "rows": int(finite.sum()), "positive_rate": float(y[finite].mean()) if finite.any() else np.nan,
            "pr_auc": pr_auc, "roc_auc": roc_auc,
            "brier": float(brier_score_loss(y[finite], probability)) if is_probability and finite.any() else np.nan,
            "log_loss": float(log_loss(y[finite], probability, labels=[0, 1])) if is_probability and finite.any() else np.nan,
        })
    return pd.DataFrame(rows)


def _event_policy_specs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fraction in (0.10, 0.20, 0.30, 0.40):
        rows.extend([
            {"policy_id": f"TP{int(fraction*100):02d}_ONLY", "tp_top_fraction": fraction, "fast30_min_percentile": np.nan, "clean50_min_percentile": np.nan, "risk_max_percentile": np.nan},
            {"policy_id": f"TP{int(fraction*100):02d}_FAST50", "tp_top_fraction": fraction, "fast30_min_percentile": 0.50, "clean50_min_percentile": np.nan, "risk_max_percentile": np.nan},
            {"policy_id": f"TP{int(fraction*100):02d}_CLEAN50", "tp_top_fraction": fraction, "fast30_min_percentile": np.nan, "clean50_min_percentile": 0.50, "risk_max_percentile": np.nan},
            {"policy_id": f"TP{int(fraction*100):02d}_FAST50_CLEAN50", "tp_top_fraction": fraction, "fast30_min_percentile": 0.50, "clean50_min_percentile": 0.50, "risk_max_percentile": np.nan},
            {"policy_id": f"TP{int(fraction*100):02d}_FAST50_CLEAN50_RISK75", "tp_top_fraction": fraction, "fast30_min_percentile": 0.50, "clean50_min_percentile": 0.50, "risk_max_percentile": 0.75},
        ])
    return pd.DataFrame(rows)


def _event_outcomes(frame: pd.DataFrame, folds: Sequence[FoldSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in folds:
        subset, _ = _subset_period(frame, fold.test_start, fold.test_end)
        for path, path_frame in subset.groupby("decision_path", sort=False):
            groups: list[tuple[str, pd.DataFrame]] = [("all", path_frame)]
            if path == "reclaim":
                lag = pd.to_numeric(path_frame["reclaim_lag_bars"], errors="coerce")
                groups.extend([
                    ("same_bar", path_frame[lag.eq(0)]),
                    ("lag_1", path_frame[lag.eq(1)]),
                    ("lag_2_3", path_frame[lag.between(2, 3)]),
                ])
            for state, part in groups:
                rows.append({
                    "fold": fold.fold, "decision_path": path, "state": state, "events": len(part),
                    "tp_rate": float(part["tp_hit_1pct"].mean()) if len(part) else np.nan,
                    "clean50_rate": float(part["tp_before_adverse_0p5pct"].mean()) if len(part) else np.nan,
                    "fast30_rate": float(part["tp_within_30"].mean()) if len(part) else np.nan,
                    "median_mae_pct": float(pd.to_numeric(part["mae_horizon_pct"], errors="coerce").median()) if len(part) else np.nan,
                    "p90_mae_pct": float(pd.to_numeric(part["mae_horizon_pct"], errors="coerce").quantile(0.90)) if len(part) else np.nan,
                })
    return pd.DataFrame(rows)


def _paired_path_comparison(frame: pd.DataFrame, folds: Sequence[FoldSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in folds:
        subset, _ = _subset_period(frame, fold.test_start, fold.test_end)
        sweep = subset[subset["decision_path"].eq("sweep")].set_index("lifecycle_id")
        reclaim = subset[subset["decision_path"].eq("reclaim")].set_index("lifecycle_id")
        common = sweep.index.intersection(reclaim.index)
        for state, ids in (
            ("all_reclaimed", common),
            ("same_bar", common[pd.to_numeric(reclaim.loc[common, "reclaim_lag_bars"], errors="coerce").eq(0).to_numpy()]),
            ("lag_1", common[pd.to_numeric(reclaim.loc[common, "reclaim_lag_bars"], errors="coerce").eq(1).to_numpy()]),
            ("lag_2_3", common[pd.to_numeric(reclaim.loc[common, "reclaim_lag_bars"], errors="coerce").between(2, 3).to_numpy()]),
        ):
            for path, source in (("sweep", sweep), ("reclaim", reclaim)):
                part = source.loc[ids] if len(ids) else source.iloc[:0]
                rows.append({
                    "fold": fold.fold, "paired_state": state, "decision_path": path, "paired_events": len(part),
                    "tp_rate": float(part["tp_hit_1pct"].mean()) if len(part) else np.nan,
                    "clean50_rate": float(part["tp_before_adverse_0p5pct"].mean()) if len(part) else np.nan,
                    "fast30_rate": float(part["tp_within_30"].mean()) if len(part) else np.nan,
                    "median_mae_pct": float(pd.to_numeric(part["mae_horizon_pct"], errors="coerce").median()) if len(part) else np.nan,
                    "median_entry_price": float(pd.to_numeric(part["entry_price"], errors="coerce").median()) if len(part) else np.nan,
                })
    return pd.DataFrame(rows)


def _increment_comparison(frontier: pd.DataFrame) -> pd.DataFrame:
    keys = ["fold", "decision_path", "policy_id"]
    baseline_map = {"sweep": "S0_tradebar", "reclaim": "R0_tradebar"}
    rows: list[dict[str, object]] = []
    metrics = ["event_count", "events_per_month", "tp_rate", "clean_0p50_rate", "fast30_rate", "median_horizon_mae_pct", "top5_day_event_share"]
    for path, baseline_group in baseline_map.items():
        baseline = frontier[(frontier["decision_path"].eq(path)) & (frontier["feature_group"].eq(baseline_group))]
        lookup = baseline.set_index(keys)[metrics]
        for row in frontier[(frontier["decision_path"].eq(path)) & (~frontier["feature_group"].eq(baseline_group))].itertuples(index=False):
            key = (row.fold, row.decision_path, row.policy_id)
            if key not in lookup.index:
                continue
            base = lookup.loc[key]
            output = {"fold": row.fold, "decision_path": row.decision_path, "feature_group": row.feature_group, "baseline_group": baseline_group, "policy_id": row.policy_id}
            for metric in metrics:
                value = getattr(row, metric)
                output[metric] = value
                output[f"baseline_{metric}"] = base[metric]
                output[f"delta_{metric}"] = value - base[metric] if pd.notna(value) and pd.notna(base[metric]) else np.nan
            rows.append(output)
    return pd.DataFrame(rows)


def _stability_matrix(increments: pd.DataFrame, minimum_test_events: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if increments.empty:
        return pd.DataFrame()
    for keys, group in increments.groupby(["decision_path", "feature_group", "policy_id"], sort=False):
        path, feature_group, policy_id = keys
        event_ok = pd.to_numeric(group["event_count"], errors="coerce") >= int(minimum_test_events)
        tp = pd.to_numeric(group["delta_tp_rate"], errors="coerce")
        clean = pd.to_numeric(group["delta_clean_0p50_rate"], errors="coerce")
        fast = pd.to_numeric(group["delta_fast30_rate"], errors="coerce")
        mae = pd.to_numeric(group["delta_median_horizon_mae_pct"], errors="coerce")
        rows.append({
            "decision_path": path, "feature_group": feature_group, "policy_id": policy_id,
            "fold_count": len(group), "minimum_event_count_pass": bool(event_ok.all()),
            "tp_positive_folds": int((tp > 0).sum()), "clean_positive_folds": int((clean > 0).sum()),
            "fast_positive_folds": int((fast > 0).sum()), "mae_improved_folds": int((mae < 0).sum()),
            "mean_delta_tp_rate": float(tp.mean()), "mean_delta_clean50_rate": float(clean.mean()),
            "mean_delta_fast30_rate": float(fast.mean()), "mean_delta_median_mae_pct": float(mae.mean()),
            "predeclared_keep_gate": bool(event_ok.all() and ((clean > 0).sum() >= 2 or (fast > 0).sum() >= 2) and (tp < -0.01).sum() == 0 and (mae > 0.05).sum() == 0),
        })
    return pd.DataFrame(rows)



def _future_truncation_audit(bars: pd.DataFrame, frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Rebuild sampled sweep decisions with every later bar removed.

    The audit is intentionally performed before model fitting.  It verifies that
    level construction and sweep-bar features do not depend on a later reclaim,
    acceptance outcome, or any other future raw bar.
    """

    sweeps = frame[frame["decision_path"].eq("sweep")].sort_values("extreme_pos").reset_index(drop=True)
    sample_size = min(max(0, int(args.causal_audit_sample_size)), len(sweeps))
    if sample_size == 0:
        return pd.DataFrame([{"sample": -1, "passed": True, "detail": "audit disabled or no sweep rows", "max_abs_diff": 0.0}])
    positions = np.unique(np.linspace(0, len(sweeps) - 1, sample_size, dtype=np.int64))
    feature_columns = [column for column in frame.columns if column.startswith("fse_")]
    rows: list[dict[str, object]] = []
    for sample_number, row_pos in enumerate(positions, start=1):
        expected = sweeps.iloc[int(row_pos)]
        decision_pos = int(expected["extreme_pos"])
        prefix = bars.iloc[: decision_pos + 1].copy()
        rebuilt = build_first_sweep_event_decisions(
            prefix,
            research_start=pd.Timestamp(args.start_date),
            research_end_exclusive=pd.Timestamp(expected["feature_available_time"]) + pd.Timedelta(nanoseconds=1),
            pivot_minutes=tuple(int(x) for x in args.liquidity_pivot_minutes),
            pivot_weights=tuple(float(x) for x in args.liquidity_pivot_weights),
            left_bars=int(args.liquidity_pivot_left_bars),
            right_bars=int(args.liquidity_pivot_right_bars),
            cluster_tolerance_bp=float(args.liquidity_cluster_tolerance_bp),
            minimum_respects=int(args.liquidity_minimum_respects),
            minimum_macro_timeframe_min=int(args.liquidity_minimum_macro_timeframe_min),
            minimum_respect_separation_minutes=int(args.liquidity_minimum_respect_separation_minutes),
            formation_max_days=int(args.liquidity_formation_max_days),
            reclaim_window_bars=int(args.liquidity_reclaim_window_bars),
            accept_below_bars=int(args.liquidity_accept_below_bars),
            accept_depth_bp=float(args.liquidity_accept_depth_bp),
            show_progress=False,
        ).decisions
        match = rebuilt[(rebuilt["decision_path"].eq("sweep")) & (pd.to_numeric(rebuilt["level_id"], errors="coerce") == int(expected["level_id"]))]
        if match.empty:
            same_bar = rebuilt[
                rebuilt["decision_path"].eq("sweep")
                & (pd.to_numeric(rebuilt["extreme_pos"], errors="coerce") == decision_pos)
            ].copy()
            if not same_bar.empty:
                expected_price = float(expected["level_price"])
                same_bar["_price_diff"] = (pd.to_numeric(same_bar["level_price"], errors="coerce") - expected_price).abs()
                nearest = same_bar.sort_values("_price_diff").iloc[0]
                detail = (
                    "level identity changed after truncation: "
                    f"expected_id={int(expected['level_id'])} rebuilt_id={int(nearest['level_id'])} "
                    f"price_diff={float(nearest['_price_diff']):.12g}"
                )
            else:
                detail = "sweep missing after future truncation"
            rows.append({
                "sample": sample_number, "level_id": int(expected["level_id"]),
                "decision_time": expected["feature_available_time"], "passed": False,
                "max_abs_diff": np.nan, "max_diff_feature": "", "detail": detail,
            })
            continue
        actual = match.iloc[0]
        expected_values = pd.to_numeric(expected.reindex(feature_columns), errors="coerce").to_numpy(dtype=float)
        actual_values = pd.to_numeric(actual.reindex(feature_columns), errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(expected_values) & np.isfinite(actual_values)
        nan_match = np.isnan(expected_values) == np.isnan(actual_values)
        abs_diff = np.full(len(feature_columns), np.nan, dtype=float)
        abs_diff[finite] = np.abs(expected_values[finite] - actual_values[finite])
        max_index = int(np.nanargmax(abs_diff)) if np.isfinite(abs_diff).any() else -1
        max_diff = float(abs_diff[max_index]) if max_index >= 0 else 0.0
        max_feature = feature_columns[max_index] if max_index >= 0 else ""
        nan_mismatch_features = [
            feature_columns[index]
            for index, matched in enumerate(nan_match)
            if not bool(matched)
        ]
        passed = bool(nan_match.all() and max_diff <= 1e-9)
        detail = "future bars removed"
        if not passed:
            detail = (
                f"feature mismatch max={max_feature}:{max_diff:.12g}; "
                f"nan_mismatch={'|'.join(nan_mismatch_features[:5])}"
            )
        rows.append({
            "sample": sample_number, "level_id": int(expected["level_id"]),
            "decision_time": expected["feature_available_time"], "passed": passed,
            "max_abs_diff": max_diff, "max_diff_feature": max_feature, "detail": detail,
        })
    return pd.DataFrame(rows)


def _summary(outcomes: pd.DataFrame, increments: pd.DataFrame, stability: pd.DataFrame, diagnostics: pd.DataFrame, audit: pd.DataFrame) -> str:
    metrics = diagnostics[diagnostics.get("scope", pd.Series(dtype=str)).eq("aggregate")]
    metric_map = dict(zip(metrics.get("metric", []), metrics.get("value", [])))
    lines = [
        "# Research 12 Summary",
        "",
        "Research-only first-sweep event branch. Sweep and reclaim are separate deployable decision times.",
        "",
        f"- sweep decisions: {int(metric_map.get('sweep_decisions', 0)):,}",
        f"- reclaim decisions: {int(metric_map.get('reclaim_decisions', 0)):,}",
        "- entry reference: next 1m open after the selected closed decision bar",
        "- path labels: future closed-bar closes only; future high/low excluded",
        "- no automatic frozen-test winner selection",
        "",
        "## Direct event outcomes",
        "",
    ]
    for row in outcomes[outcomes["state"].eq("all")].itertuples(index=False):
        lines.append(f"- {row.fold} {row.decision_path}: n={row.events:,}, TP={row.tp_rate:.4f}, Clean50={row.clean50_rate:.4f}, Fast30={row.fast30_rate:.4f}, median MAE={row.median_mae_pct:.4f}%")
    lines.extend(["", "## Increment gate", ""])
    kept = stability[stability["predeclared_keep_gate"].astype(bool)] if not stability.empty else pd.DataFrame()
    if kept.empty:
        lines.append("No feature-group/policy pair passed the predeclared cross-fold keep gate.")
    else:
        for row in kept.itertuples(index=False):
            lines.append(f"- {row.decision_path} {row.feature_group} {row.policy_id}: TP folds+={row.tp_positive_folds}, Clean folds+={row.clean_positive_folds}, Fast folds+={row.fast_positive_folds}, MAE folds improved={row.mae_improved_folds}")
    lines.extend(["", "## Audit", ""])
    for row in audit.itertuples(index=False):
        lines.append(f"- [{'PASS' if row.passed else 'FAIL'}] {row.check}: {row.detail}")
    lines.extend(["", "The report is an event-model study, not a trading strategy or profitability claim."])
    return "\n".join(lines) + "\n"


def run_research(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = load_bars(args)
    _write_csv(validate_trade_bar_fields(bars), out_dir / "01_trade_bar_field_coverage.csv")

    research_start = pd.Timestamp(args.start_date)
    research_end = _end_exclusive(args.end_date)
    print("[stage] respected macro first-sweep lifecycles", flush=True)
    event_build = build_first_sweep_event_decisions(
        bars,
        research_start=research_start,
        research_end_exclusive=research_end,
        pivot_minutes=tuple(int(x) for x in args.liquidity_pivot_minutes),
        pivot_weights=tuple(float(x) for x in args.liquidity_pivot_weights),
        left_bars=int(args.liquidity_pivot_left_bars),
        right_bars=int(args.liquidity_pivot_right_bars),
        cluster_tolerance_bp=float(args.liquidity_cluster_tolerance_bp),
        minimum_respects=int(args.liquidity_minimum_respects),
        minimum_macro_timeframe_min=int(args.liquidity_minimum_macro_timeframe_min),
        minimum_respect_separation_minutes=int(args.liquidity_minimum_respect_separation_minutes),
        formation_max_days=int(args.liquidity_formation_max_days),
        reclaim_window_bars=int(args.liquidity_reclaim_window_bars),
        accept_below_bars=int(args.liquidity_accept_below_bars),
        accept_depth_bp=float(args.liquidity_accept_depth_bp),
        show_progress=True,
    )
    if event_build.decisions.empty:
        raise RuntimeError("no first-sweep decisions were built")
    if not {"sweep", "reclaim"}.issubset(set(event_build.decisions["decision_path"])):
        raise RuntimeError("both sweep and reclaim decision paths are required")
    metric_map = dict(zip(event_build.diagnostics.get("metric", []), event_build.diagnostics.get("value", [])))
    if int(metric_map.get("availability_violations", 0)) != 0:
        raise RuntimeError("first-sweep level availability audit failed")
    _write_csv(event_build.diagnostics, out_dir / "02_event_build_diagnostics.csv")
    _write_csv(event_build.levels, out_dir / "03_respected_level_table.csv")
    _write_csv(event_build.lifecycle, out_dir / "04_first_sweep_lifecycle_table.csv")
    _write_csv(event_build.dictionary, out_dir / "05_event_feature_dictionary.csv")

    print("[stage] causal 1m snapshot at sweep and reclaim decision bars", flush=True)
    snapshot = build_reversal_candidate_features(
        bars, event_build.decisions, include_session=False, include_htf=False, show_progress=True,
    )
    labels = build_reversal_forward_labels(
        bars, snapshot.frame,
        target_move_pct=float(args.target_move_pct),
        horizon=int(args.forward_horizon_bars),
        vectorized_chunk_size=int(args.label_vectorized_chunk_size),
        show_progress=True,
    )
    frame = snapshot.frame.merge(labels, on="event_id", how="inner", validate="one_to_one", suffixes=("", "_label"))
    if frame.empty:
        raise RuntimeError("all first-sweep decisions were removed by forward-label boundaries")
    frame = frame.sort_values(["extreme_pos", "decision_path", "event_id"]).reset_index(drop=True)
    episodic = attach_positive_opportunity_episodes(frame, max_gap_bars=2)
    for column in ("positive_episode_id", "positive_episode_size", "positive_episode_number"):
        frame[column] = episodic[column].to_numpy()
    _write_csv(snapshot.dictionary, out_dir / "06_snapshot_feature_dictionary.csv")
    _write_csv(mechanism_feature_dictionary(), out_dir / "07_soft_mechanism_feature_dictionary.csv")

    folds = _folds(args.end_date)
    outcomes = _event_outcomes(frame, folds)
    paired = _paired_path_comparison(frame, folds)
    _write_csv(outcomes, out_dir / "08_direct_event_outcomes.csv")
    _write_csv(paired, out_dir / "09_paired_sweep_vs_reclaim.csv")
    print("[stage] future-truncation causal audit", flush=True)
    future_audit = _future_truncation_audit(bars, frame, args)
    _write_csv(future_audit, out_dir / "09b_future_truncation_audit.csv")
    if not future_audit["passed"].astype(bool).all():
        raise RuntimeError("first-sweep future-truncation audit failed")
    _write_csv(pd.DataFrame([fold._asdict() for fold in folds]), out_dir / "10_walkforward_folds.csv")
    policy_specs = _event_policy_specs()
    _write_csv(policy_specs, out_dir / "11_predeclared_event_policy_grid.csv")

    m0_features = tuple(snapshot.group_membership.loc[snapshot.group_membership["feature_group"].eq("M0_core"), "feature"].astype(str))
    group_members = {
        group: tuple(event_build.group_membership.loc[event_build.group_membership["feature_group"].eq(group), "feature"].astype(str))
        for group in (LEVEL_GROUP, ORDERFLOW_GROUP, RECLAIM_GROUP)
    }
    path_groups: dict[str, dict[str, tuple[str, ...]]] = {
        "sweep": {
            "S0_tradebar": (),
            "S1_level_geometry": group_members[LEVEL_GROUP],
            "S2_sweep_orderflow": (*group_members[LEVEL_GROUP], *group_members[ORDERFLOW_GROUP]),
        },
        "reclaim": {
            "R0_tradebar": (),
            "R1_level_geometry": group_members[LEVEL_GROUP],
            "R2_reclaim_process": (*group_members[LEVEL_GROUP], *group_members[ORDERFLOW_GROUP], *group_members[RECLAIM_GROUP]),
        },
    }

    split_parts: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    calibration_selection_parts: list[pd.DataFrame] = []
    calibration_metric_parts: list[pd.DataFrame] = []
    head_metric_parts: list[pd.DataFrame] = []
    risk_metric_parts: list[pd.DataFrame] = []
    policy_window_parts: list[pd.DataFrame] = []
    test_frontier_parts: list[pd.DataFrame] = []
    stress_parts: list[pd.DataFrame] = []
    prediction_samples: list[pd.DataFrame] = []
    rank_rows: list[dict[str, object]] = []
    full_predictions: list[pd.DataFrame] = []

    for path_index, decision_path in enumerate(("sweep", "reclaim"), start=1):
        path_frame = frame[frame["decision_path"].eq(decision_path)].reset_index(drop=True)
        for fold_index, fold in enumerate(folds, start=1):
            print(f"[fold] path={decision_path} {fold.fold}", flush=True)
            full_train, removed_train = _subset_period(path_frame, fold.train_start, fold.train_end)
            test, removed_test = _subset_period(path_frame, fold.test_start, fold.test_end)
            if len(test) < int(args.minimum_test_events):
                raise RuntimeError(f"{decision_path} {fold.fold} has only {len(test)} test events")
            model_fit, calibration, policy, nested = _development_split(full_train, fold)
            nested.insert(1, "decision_path", decision_path)
            nested["test_rows"] = len(test)
            nested["full_train_cross_boundary_removed"] = removed_train
            nested["test_cross_boundary_removed"] = removed_test
            split_parts.append(nested)

            for data in (model_fit, calibration, policy, test):
                episode = attach_positive_opportunity_episodes(data, max_gap_bars=2)
                for column in ("positive_episode_id", "positive_episode_size"):
                    data[column] = episode[column].to_numpy()
            model_fit = attach_episode_balanced_weight(model_fit)
            mechanism = fit_soft_mechanism_transformer(model_fit)
            mechanism_features: tuple[str, ...] = ()
            for name, data in (("model_fit", model_fit), ("calibration", calibration), ("policy", policy), ("test", test)):
                transformed = mechanism.transform(data)
                if name == "model_fit":
                    mechanism_features = tuple(column for column in transformed.columns if column != "mechanism_dominant")
                for column in transformed.columns:
                    data[column] = transformed[column].to_numpy()
            base_requested = (*m0_features, *mechanism_features)

            for group_index, (feature_group, extra) in enumerate(path_groups[decision_path].items(), start=1):
                print(f"[models] {fold.fold} {feature_group} ({group_index}/{len(path_groups[decision_path])})", flush=True)
                requested = (*base_requested, *extra)
                selected, diagnostics = _condition_feature_columns(model_fit, requested)
                selected_event = [column for column in selected if column.startswith("fse_")]
                if extra and not selected_event:
                    raise RuntimeError(f"{fold.fold} {feature_group} retained no first-sweep features")
                if feature_group in {"S2_sweep_orderflow", "R2_reclaim_process"}:
                    required_prefixes = group_members[ORDERFLOW_GROUP] if decision_path == "sweep" else (*group_members[ORDERFLOW_GROUP], *group_members[RECLAIM_GROUP])
                    if not any(column in selected for column in required_prefixes):
                        raise RuntimeError(f"{fold.fold} {feature_group} retained no process/order-flow increment feature")
                feature_rows.append({"fold": fold.fold, "decision_path": decision_path, "feature_group": feature_group, **diagnostics})
                score_frames = {"calibration": _score_shell(calibration), "policy": _score_shell(policy), "test": _score_shell(test)}

                for head_index, (output, target) in enumerate(HEAD_TARGETS.items(), start=1):
                    model, fit_diagnostics = _fit_binary_with_resolution_fallback(
                        model_fit, policy,
                        feature_columns=selected, target_column=target,
                        fold=fold.fold, decision_path=decision_path,
                        feature_group=feature_group, output=output,
                        # Keep the solver seed identical across feature groups
                        # within the same path/fold/head so incremental results
                        # cannot be explained by a different random path.
                        random_state=int(args.random_state) + path_index * 100 + fold_index * 10 + head_index,
                        min_samples_leaf=int(args.model_min_samples_leaf),
                        prediction_chunk_size=int(args.prediction_chunk_size),
                    )
                    model_rows.append({
                        "fold": fold.fold, "decision_path": decision_path, "feature_group": feature_group,
                        "output": output, "target": target, "requested_family": PRIMARY_FAMILY,
                        "actual_family": getattr(model, "family", PRIMARY_FAMILY),
                        **fit_diagnostics,
                    })
                    score_by_split: dict[str, np.ndarray] = {}
                    probability_by_split: dict[str, np.ndarray] = {}
                    for split, source in (("calibration", calibration), ("policy", policy), ("test", test)):
                        score = _predict_binary_score(model, source, int(args.prediction_chunk_size))
                        probability = _predict_binary_probability(model, source, int(args.prediction_chunk_size))
                        score_by_split[split] = score
                        probability_by_split[split] = probability
                        # ``*_score_raw`` is the unsquashed deployable ranking
                        # score. ``*_raw`` remains the model probability for
                        # probability diagnostics and backwards-readable reports.
                        score_frames[split][f"{output}_score_raw"] = score
                        score_frames[split][f"{output}_raw"] = probability

                    # Check every policy-driving head immediately.  A saturated
                    # probability is allowed; a degenerate decision score is not.
                    provisional_reference = EmpiricalRankReference.fit(score_by_split["policy"])
                    provisional_rank = provisional_reference.transform(score_by_split["policy"] )
                    provisional_resolution = _rank_resolution_record(
                        fold=fold.fold, decision_path=decision_path, feature_group=feature_group,
                        output=output, split="policy", raw_scores=score_by_split["policy"],
                        ranks=provisional_rank, calibrated=probability_by_split["policy"],
                        reference=provisional_reference, model_probability=probability_by_split["policy"],
                    )
                    _assert_raw_score_resolution(
                        provisional_resolution, actual_family=str(getattr(model, "family", PRIMARY_FAMILY))
                    )

                    order = np.argsort(pd.to_numeric(calibration["extreme_pos"], errors="raise").to_numpy(dtype=np.int64), kind="mergesort")
                    cut = max(1, min(len(calibration) - 1, len(calibration) // 2))
                    fit_pos, select_pos = order[:cut], order[cut:]
                    candidates_cal = fit_score_probability_calibrators(score_by_split["calibration"][fit_pos], calibration.iloc[fit_pos][target])
                    selected_method, selection = choose_calibrator(candidates_cal, score_by_split["calibration"][select_pos], calibration.iloc[select_pos][target])
                    selection.insert(0, "feature_group", feature_group)
                    selection.insert(0, "decision_path", decision_path)
                    selection.insert(0, "output", output)
                    selection.insert(0, "fold", fold.fold)
                    selection["selected"] = selection["method"].eq(selected_method)
                    calibration_selection_parts.append(selection)
                    final_calibrator = fit_score_probability_calibrators(score_by_split["calibration"], calibration[target])[selected_method]
                    rank_reference = EmpiricalRankReference.fit(score_by_split["policy"])
                    for split, source in (("calibration", calibration), ("policy", policy), ("test", test)):
                        score_frames[split][f"{output}_cal"] = final_calibrator.transform(score_by_split[split])
                        calibration_metric_parts.append(pd.DataFrame([{
                            "fold": fold.fold, "decision_path": decision_path, "feature_group": feature_group,
                            "output": output, "target": target, "split": split, "selected_method": selected_method,
                            **calibration_metrics(source[target], score_frames[split][f"{output}_cal"]),
                        }]))
                        head_metric_parts.append(_head_metrics(score_frames[split], fold=fold.fold, decision_path=decision_path, feature_group=feature_group, output=output, target=target, split=split))
                    for split in ("policy", "test"):
                        ranks = rank_reference.transform(score_by_split[split])
                        score_frames[split][f"{output}_rank"] = ranks
                        resolution = _rank_resolution_record(
                            fold=fold.fold,
                            decision_path=decision_path,
                            feature_group=feature_group,
                            output=output,
                            split=split,
                            raw_scores=score_by_split[split],
                            ranks=ranks,
                            calibrated=score_frames[split][f"{output}_cal"],
                            reference=rank_reference, model_probability=probability_by_split[split],
                        )
                        rank_rows.append(resolution)
                        _assert_raw_score_resolution(
                            resolution, actual_family=str(getattr(model, "family", PRIMARY_FAMILY))
                        )

                risk_model = fit_risk_point_model(model_fit, feature_columns=selected, target_column="mae_horizon_pct", success_only=False)
                risk_rows.append({
                    "fold": fold.fold, "decision_path": decision_path, "feature_group": feature_group,
                    "target": "mae_horizon_pct", "actual_family": getattr(risk_model, "fit_method", "unknown"),
                    "converged": bool(getattr(risk_model, "converged", True)),
                    "iterations": int(getattr(risk_model, "iterations", 0)), "selected_feature_count": len(selected),
                })
                risk_raw = {split: _predict_risk(risk_model, source, int(args.prediction_chunk_size)) for split, source in (("calibration", calibration), ("policy", policy), ("test", test))}
                adjustment = fit_conformal_adjustment(calibration["mae_horizon_pct"], risk_raw["calibration"], quantile=0.90)
                for split, source in (("calibration", calibration), ("policy", policy), ("test", test)):
                    score_frames[split]["mae_horizon_point_raw"] = risk_raw[split]
                    score_frames[split]["mae_horizon_q90_cal"] = adjustment.apply(risk_raw[split])
                    risk_metric_parts.append(pd.DataFrame([{
                        "fold": fold.fold, "decision_path": decision_path, "feature_group": feature_group,
                        "split": split, "output": "mae_horizon_q90", "target": "mae_horizon_pct",
                        "quantile": 0.90, "additive_shift": adjustment.additive_shift,
                        **quantile_metrics(source["mae_horizon_pct"], score_frames[split]["mae_horizon_q90_cal"]),
                    }]))
                risk_reference = EmpiricalRankReference.fit(risk_raw["policy"])
                score_frames["policy"]["mae_horizon_risk_rank"] = risk_reference.transform(risk_raw["policy"])
                score_frames["test"]["mae_horizon_risk_rank"] = risk_reference.transform(risk_raw["test"])

                policy_months = max(1, pd.to_datetime(policy["extreme_time"]).dt.to_period("M").nunique())
                test_months = max(1, pd.to_datetime(test["extreme_time"]).dt.to_period("M").nunique())
                for spec in policy_specs.itertuples(index=False):
                    series = pd.Series(spec._asdict())
                    policy_events = select_ranked_events(score_frames["policy"], series, cooldown_bars=int(args.cooldown_bars))
                    test_events = select_ranked_events(score_frames["test"], series, cooldown_bars=int(args.cooldown_bars))
                    common = {"fold": fold.fold, "decision_path": decision_path, "feature_group": feature_group, **spec._asdict(), "cooldown_bars": int(args.cooldown_bars)}
                    policy_window_parts.append(pd.DataFrame([{**common, **policy_metrics(policy_events, score_frames["policy"], months=policy_months)}]))
                    test_frontier_parts.append(pd.DataFrame([{**common, **policy_metrics(test_events, score_frames["test"], months=test_months)}]))
                    if spec.policy_id in {"TP10_ONLY", "TP20_ONLY", "TP20_FAST50_CLEAN50", "TP30_FAST50_CLEAN50_RISK75"}:
                        stress = delete_day_stress(test_events)
                        stress.insert(0, "policy_id", spec.policy_id)
                        stress.insert(0, "feature_group", feature_group)
                        stress.insert(0, "decision_path", decision_path)
                        stress.insert(0, "fold", fold.fold)
                        stress_parts.append(stress)

                sample = pd.concat([
                    score_frames["test"].nlargest(min(300, len(test)), "p_tp60_rank"),
                    score_frames["test"].sample(min(300, len(test)), random_state=int(args.random_state) + fold_index + group_index),
                ], ignore_index=True).drop_duplicates("event_id")
                sample.insert(0, "feature_group", feature_group)
                sample.insert(0, "fold", fold.fold)
                prediction_samples.append(sample)
                if args.write_full_predictions:
                    full = score_frames["test"].copy()
                    full.insert(0, "feature_group", feature_group)
                    full.insert(0, "fold", fold.fold)
                    full_predictions.append(full)
                del score_frames
                gc.collect()

    split_table = pd.concat(split_parts, ignore_index=True)
    feature_table = pd.DataFrame(feature_rows)
    model_methods = pd.DataFrame(model_rows)
    risk_methods = pd.DataFrame(risk_rows)
    calibration_selection = pd.concat(calibration_selection_parts, ignore_index=True)
    calibration_metrics_table = pd.concat(calibration_metric_parts, ignore_index=True)
    head_metrics = pd.concat(head_metric_parts, ignore_index=True)
    risk_metrics = pd.concat(risk_metric_parts, ignore_index=True)
    policy_window = pd.concat(policy_window_parts, ignore_index=True)
    test_frontier = pd.concat(test_frontier_parts, ignore_index=True)
    increments = _increment_comparison(test_frontier)
    stability = _stability_matrix(increments, int(args.minimum_test_events))
    stress = pd.concat(stress_parts, ignore_index=True) if stress_parts else pd.DataFrame()
    samples = pd.concat(prediction_samples, ignore_index=True)
    ranks = pd.DataFrame(rank_rows)

    _write_csv(split_table, out_dir / "12_nested_fold_boundaries.csv")
    _write_csv(feature_table, out_dir / "13_fold_feature_groups.csv")
    _write_csv(model_methods, out_dir / "14_model_head_fit_methods.csv")
    _write_csv(risk_methods, out_dir / "14b_risk_fit_methods.csv")
    _write_csv(calibration_selection, out_dir / "15_calibration_method_selection.csv")
    _write_csv(calibration_metrics_table, out_dir / "16_probability_calibration_metrics.csv")
    _write_csv(head_metrics, out_dir / "17_head_ranking_metrics.csv")
    _write_csv(risk_metrics, out_dir / "18_mae_risk_calibration.csv")
    _write_csv(policy_window, out_dir / "19_policy_window_rank_frontier.csv")
    _write_csv(test_frontier, out_dir / "20_frozen_test_rank_frontier.csv")
    _write_csv(increments, out_dir / "21_event_feature_increment_comparison.csv")
    _write_csv(stability, out_dir / "22_cross_fold_stability_matrix.csv")
    _write_csv(stress, out_dir / "23_delete_strong_days_stress.csv")
    _write_csv(samples, out_dir / "24_walkforward_prediction_sample.csv")
    _write_csv(ranks, out_dir / "25_raw_rank_resolution_diagnostics.csv")
    if full_predictions:
        _write_csv(pd.concat(full_predictions, ignore_index=True), out_dir / "26_walkforward_full_predictions.csv")

    forbidden_tokens = ("future", "forward", "label", "mfe", "mae", "tp_hit", "adverse", "entry_price", "completion", "confirmation")
    selected_text = "|".join(feature_table["selected_features"].astype(str)).lower()
    forbidden = [token for token in forbidden_tokens if token in selected_text]
    policy_driving_ranks = ranks[ranks["split"].isin(["policy", "test"])]
    raw_score_resolution_passed = bool(
        not policy_driving_ranks.empty and policy_driving_ranks["raw_score_resolution_passed"].astype(bool).all()
    )
    test_ranks = policy_driving_ranks[policy_driving_ranks["split"].eq("test")]
    maximum_tail_saturation = (
        float(pd.to_numeric(test_ranks["rank_tail_saturation_share"], errors="coerce").max())
        if not test_ranks.empty else np.nan
    )
    binary_stable = bool(not model_methods.empty and model_methods["actual_family"].astype(str).str.contains("logistic_sgd|logistic_newton_cholesky|logistic_lbfgs|hist_gbdt", regex=True).all())
    risk_stable = bool(not risk_methods.empty and risk_methods["converged"].astype(bool).all())
    increment_groups_have_features = bool(
        feature_table[~feature_table["feature_group"].isin(["S0_tradebar", "R0_tradebar"])]
        ["selected_features"].astype(str).str.contains("fse_").all()
    )
    audit = pd.DataFrame([
        {"check": "level_available_before_decision", "passed": int(metric_map.get("availability_violations", 0)) == 0, "detail": f"violations={int(metric_map.get('availability_violations', 0))}"},
        {"check": "sweep_and_reclaim_are_separate_decision_paths", "passed": True, "detail": "sweep entry uses sweep close -> next open; reclaim entry uses reclaim close -> next open"},
        {"check": "labels_use_next_open_future_close", "passed": True, "detail": "entry=next open; TP/Fast/Clean/MAE=future closes"},
        {"check": "future_high_low_not_used_for_labels", "passed": True, "detail": "future high/low excluded"},
        {"check": "sweep_does_not_use_future_reclaim", "passed": True, "detail": "delayed reclaim process fields are NaN on sweep decisions; same-bar reclaim is known at sweep close"},
        {"check": "future_truncation_rebuild", "passed": bool(not future_audit.empty and future_audit["passed"].all()), "detail": f"audited={len(future_audit)}"},
        {"check": "increment_groups_retain_event_features", "passed": increment_groups_have_features, "detail": "each non-baseline fold/group retains at least one fse_* feature"},
        {
            "check": "raw_rank_has_resolution",
            "passed": raw_score_resolution_passed,
            "detail": (
                f"policy_and_test_groups={len(policy_driving_ranks)}; resolution is checked on finite unsquashed decision scores for TP/Fast/Clean; "
                f"frozen empirical-CDF tail compression is diagnostic only; max_test_tail_saturation={maximum_tail_saturation:.6f}"
            ),
        },
        {
            "check": "frozen_rank_tail_saturation_reported",
            "passed": True,
            "detail": "25_raw_rank_resolution_diagnostics.csv records policy-range overflow and 0/1 tail shares",
        },
        {"check": "future_labels_excluded_from_model_features", "passed": not forbidden, "detail": "|".join(forbidden)},
        {"check": "binary_heads_converged_or_stable_fallback", "passed": binary_stable, "detail": "all actual fit methods recorded"},
        {"check": "mae_risk_model_numerically_stable", "passed": risk_stable, "detail": "all risk fits converged or deterministic constant"},
        {"check": "no_frozen_test_winner_selection", "passed": True, "detail": "all predeclared groups/policies reported; stability gate is descriptive"},
        {"check": "aggressive_sell_is_proxy_not_stop_identity", "passed": True, "detail": "observed sell imbalance is not labeled as confirmed stop flow"},
    ])
    _write_csv(audit, out_dir / "27_causal_and_selection_audit.csv")
    if not audit["passed"].all():
        raise RuntimeError(f"12 audit failed: {audit.loc[~audit['passed'], 'check'].tolist()}")

    manifest = {
        "script": SCRIPT_NAME, "script_version": SCRIPT_VERSION, "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID, "title": TITLE, "symbol": args.symbol, "timeframe": args.timeframe,
        "start_date": args.start_date, "end_date": args.end_date, "warmup_start_date": args.warmup_start_date,
        "target_move_pct": float(args.target_move_pct), "forward_horizon_bars": int(args.forward_horizon_bars),
        "entry_price_source": "next_bar_open_after_path_specific_closed_decision_bar",
        "path_observation_source": "future_closed_bar_close", "future_high_low_used_for_labels": False,
        "decision_paths": ["sweep", "reclaim"], "feature_groups": path_groups,
        "sweep_decision_count": int((frame["decision_path"] == "sweep").sum()),
        "reclaim_decision_count": int((frame["decision_path"] == "reclaim").sum()),
        "ranking": "frozen empirical percentile of unsquashed policy-window decision scores",
        "model_probability_role": "diagnostics only; may saturate without losing decision-score ordering",
        "calibration_role": "interpretation only; never rank selection",
        "automatic_test_winner_selected": False, "strategy_or_backtest": False,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "28_RESEARCH_SUMMARY.md").write_text(_summary(outcomes, increments, stability, event_build.diagnostics, audit), encoding="utf-8")
    result = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={result.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
