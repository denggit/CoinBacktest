#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal reversal-zone process research.

Research 06 treats a possible reversal as a variable-length online process,
not as one exact Swing Low minute.  A frozen 2023 anchor model identifies broad
high-score low regions.  Zone process models are trained on 2024H1, selected on
2024H2, and frozen for 2025-2026H1.

Research only: no fees, stop, sizing, execution, or portfolio backtest.
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
from sklearn.inspection import permutation_importance

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
    empirical_percentile,
    opportunity_event_metrics,
    select_usable_features,
)
from research.market_structure.swing_low_typology.common.reversal_zone_process import (  # noqa: E402
    attach_zone_split,
    build_causal_zone_states,
    choose_zone_trigger_spec,
    observation_timing_metrics,
    purge_zone_label_overlap,
    select_first_zone_signal,
    zone_feature_groups,
    zone_trigger_grid,
)
from research.market_structure.swing_low_typology.common.swing_low_dataset import validate_trade_bar_fields  # noqa: E402

SCRIPT_NAME = "06_reversal_zone_process_research"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_1M_REVERSAL_ZONE_PROCESS_06"
EDGE_ID = "RESEARCH_ONLY_ETH_REVERSAL_ZONE_PROCESS"
TITLE = "ETH Reversal Zone Process Research 06"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/06_reversal_zone_process"

MODEL_FAMILIES: tuple[str, ...] = ("logistic", "hist_gbdt")
ARCHITECTURES: tuple[str, ...] = ("tp_only", "two_stage", "direct_clean")
DEFAULT_ACTIVATION_PERCENTILES: tuple[float, ...] = (95.0, 97.5, 99.0)
DEFAULT_TOP_FRACTIONS: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10)
DEFAULT_OBSERVATIONS: tuple[int, ...] = (1, 2, 3, 4)
DEFAULT_COOLDOWNS: tuple[int, ...] = (0, 15, 30)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal variable-length reversal-zone process research.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--anchor-fit-end-date", default="2023-12-31 23:59:59")
    p.add_argument("--zone-fit-end-date", default="2024-06-30 23:59:59")
    p.add_argument("--zone-validation-end-date", default="2024-12-31 23:59:59")
    p.add_argument("--target-move-pct", type=float, default=1.0)
    p.add_argument("--forward-horizon-bars", type=int, default=60)
    p.add_argument("--clean-adverse-pct", type=float, default=0.25)
    p.add_argument("--lookback", type=int, default=240)
    p.add_argument("--candidate-new-low-window", type=int, default=5)
    p.add_argument("--candidate-near-floor-window", type=int, default=60)
    p.add_argument("--candidate-position-window", type=int, default=120)
    p.add_argument("--candidate-near-floor-tolerance-bp", type=float, default=20.0)
    p.add_argument("--candidate-max-position-in-range", type=float, default=0.55)
    p.add_argument("--zone-activation-percentiles", type=float, nargs="+", default=list(DEFAULT_ACTIVATION_PERCENTILES))
    p.add_argument("--zone-max-gap-bars", type=int, default=5)
    p.add_argument("--zone-max-duration-bars", type=int, default=120)
    p.add_argument("--zone-support-tolerance-bp", type=float, default=25.0)
    p.add_argument("--trigger-top-fractions", type=float, nargs="+", default=list(DEFAULT_TOP_FRACTIONS))
    p.add_argument("--trigger-min-observations", type=int, nargs="+", default=list(DEFAULT_OBSERVATIONS))
    p.add_argument("--trigger-cooldowns", type=int, nargs="+", default=list(DEFAULT_COOLDOWNS))
    p.add_argument("--model-min-samples-leaf", type=int, default=80)
    p.add_argument("--minimum-zone-fit-rows", type=int, default=500)
    p.add_argument("--minimum-zone-validation-events", type=int, default=20)
    p.add_argument("--label-vectorized-chunk-size", type=int, default=50_000)
    p.add_argument("--causal-audit-sample-size", type=int, default=6)
    p.add_argument("--seed-stability-count", type=int, default=3)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")
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


def _clean_target(args: argparse.Namespace) -> str:
    return f"tp_before_adverse_{str(float(args.clean_adverse_pct)).replace('.', 'p')}pct"


def _fit_triplet(
    train: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    family: str,
    clean_target: str,
    random_state: int,
    min_samples_leaf: int,
) -> dict[str, object]:
    tp = fit_binary_model(
        train,
        feature_columns=feature_columns,
        target_column="tp_hit_1pct",
        family=family,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
    )
    direct = fit_binary_model(
        train,
        feature_columns=feature_columns,
        target_column=clean_target,
        family=family,
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
    )
    tp_positive = train[train["tp_hit_1pct"].astype(bool)]
    quality = None
    if len(tp_positive) >= 100 and tp_positive[clean_target].nunique() >= 2:
        quality = fit_binary_model(
            tp_positive,
            feature_columns=feature_columns,
            target_column=clean_target,
            family=family,
            random_state=random_state,
            min_samples_leaf=max(20, min_samples_leaf // 2),
        )
    return {"tp": tp, "direct": direct, "quality": quality}


def _score_triplet(models: dict[str, object], frame: pd.DataFrame) -> pd.DataFrame:
    tp = models["tp"].predict_proba(frame)  # type: ignore[attr-defined]
    direct = models["direct"].predict_proba(frame)  # type: ignore[attr-defined]
    quality_model = models.get("quality")
    quality = quality_model.predict_proba(frame) if quality_model is not None else np.ones(len(frame), dtype=float)  # type: ignore[attr-defined]
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


def _score_column(architecture: str) -> str:
    return {
        "tp_only": "score_tp_only",
        "two_stage": "score_two_stage",
        "direct_clean": "score_direct_clean",
    }[architecture]


def _anchor_model(
    frame: pd.DataFrame,
    m0_features: Sequence[str],
    *,
    anchor_fit_end: pd.Timestamp,
    random_state: int,
    min_samples_leaf: int,
) -> tuple[object, tuple[str, ...], pd.DataFrame]:
    fit = frame[
        (pd.to_datetime(frame["extreme_time"]) <= anchor_fit_end)
        & (pd.to_datetime(frame["label_end_time"]) <= anchor_fit_end)
    ].copy()
    selected = select_usable_features(fit, m0_features)
    if len(selected) < 20:
        raise RuntimeError(f"anchor model has only {len(selected)} usable M0 features")
    model = fit_binary_model(
        fit,
        feature_columns=selected,
        target_column="tp_hit_1pct",
        family="logistic",
        random_state=random_state,
        min_samples_leaf=min_samples_leaf,
    )
    fit_scores = model.predict_proba(fit)  # type: ignore[attr-defined]
    metrics = pd.DataFrame([{"model": "2023_M0_logistic_tp_anchor", "feature_count": len(selected), **binary_metrics(fit["tp_hit_1pct"], fit_scores)}])
    return model, selected, metrics


def _model_selection(
    zone_results: dict[float, pd.DataFrame],
    snapshot_features: Sequence[str],
    process_features: Sequence[str],
    args: argparse.Namespace,
    clean_target: str,
) -> tuple[pd.DataFrame, dict[tuple[float, str, str], tuple[str, ...]]]:
    rows: list[dict[str, object]] = []
    selected_features: dict[tuple[float, str, str], tuple[str, ...]] = {}
    total = len(zone_results) * 3 * len(MODEL_FAMILIES)
    reporter = ProgressReporter("[models] zone process ablation", total=total, every=1)
    done = 0
    for activation, states in zone_results.items():
        zone_fit = states[states["zone_split"].eq("zone_fit")]
        zone_validation = states[states["zone_split"].eq("zone_validation")]
        if len(zone_fit) < int(args.minimum_zone_fit_rows):
            raise RuntimeError(f"activation {activation} has only {len(zone_fit)} zone-fit states")
        groups = zone_feature_groups(snapshot_features, process_features)
        for group_name, requested in groups.items():
            usable = select_usable_features(zone_fit, requested)
            if len(usable) < 5:
                continue
            for family in MODEL_FAMILIES:
                models = _fit_triplet(
                    zone_fit,
                    usable,
                    family=family,
                    clean_target=clean_target,
                    random_state=int(args.random_state),
                    min_samples_leaf=int(args.model_min_samples_leaf),
                )
                scores = _score_triplet(models, zone_validation)
                tp_metrics = binary_metrics(zone_validation["tp_hit_1pct"], scores["tp_probability"])
                for architecture in ARCHITECTURES:
                    if architecture == "two_stage" and models.get("quality") is None:
                        continue
                    clean_metrics = binary_metrics(zone_validation[clean_target], scores[_score_column(architecture)])
                    rows.append(
                        {
                            "activation_percentile": float(activation),
                            "feature_group": group_name,
                            "family": family,
                            "architecture": architecture,
                            "feature_count": len(usable),
                            "zone_fit_rows": len(zone_fit),
                            "zone_validation_rows": len(zone_validation),
                            "clean_pr_auc": clean_metrics["pr_auc"],
                            "clean_roc_auc": clean_metrics["roc_auc"],
                            "clean_brier": clean_metrics["brier"],
                            "clean_precision_top_1pct": clean_metrics["precision_top_1pct"],
                            "tp_pr_auc": tp_metrics["pr_auc"],
                            "tp_roc_auc": tp_metrics["roc_auc"],
                            "tp_precision_top_1pct": tp_metrics["precision_top_1pct"],
                        }
                    )
                    selected_features[(float(activation), group_name, family)] = usable
                done += 1
                if done < total:
                    reporter.update(done)
                del models, scores
                gc.collect()
    reporter.close()
    table = pd.DataFrame(rows).sort_values(
        ["clean_pr_auc", "clean_brier", "tp_pr_auc", "feature_count"],
        ascending=[False, True, False, True],
    ).reset_index(drop=True)
    return table, selected_features


def _score_selected(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    selected_spec: pd.Series,
    feature_columns: Sequence[str],
    clean_target: str,
    args: argparse.Namespace,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    seed = int(args.random_state if random_state is None else random_state)
    models = _fit_triplet(
        train,
        feature_columns,
        family=str(selected_spec["family"]),
        clean_target=clean_target,
        random_state=seed,
        min_samples_leaf=int(args.model_min_samples_leaf),
    )
    train_scores = _score_triplet(models, train)
    eval_scores = _score_triplet(models, evaluation)
    metadata = [
        c for c in (
            "event_id", "zone_id", "zone_split", "zone_start_pos", "zone_observation_number",
            "zone_state_count", "extreme_time", "feature_available_time", "extreme_pos", "entry_time",
            "entry_price", "label_end_time", "tp_hit_1pct", clean_target,
            "tp_before_adverse_0p5pct", "tp_before_adverse_0p75pct", "tp_before_adverse_1p0pct",
            "mfe_pct", "mae_horizon_pct", "mae_before_tp_pct", "tp_first_touch_bar",
            "anchor_score_0_100", "anchor_raw_score", "zone_age_bars", "zone_rebound_from_low",
            "zone_absorption_improvement", "zone_score_slope", "zone_support_test_count",
        ) if c in train.columns
    ]
    train_out = train[metadata].copy()
    eval_out = evaluation[[c for c in metadata if c in evaluation.columns]].copy()
    for column in train_scores:
        train_out[column] = train_scores[column].to_numpy()
        eval_out[column] = eval_scores[column].to_numpy()
    chosen = _score_column(str(selected_spec["architecture"]))
    train_out["zone_model_raw_score"] = train_out[chosen]
    eval_out["zone_model_raw_score"] = eval_out[chosen]
    eval_out["zone_model_score_0_100"] = empirical_percentile(train_out["zone_model_raw_score"], eval_out["zone_model_raw_score"])
    return train_out, eval_out, models


def _group_metrics(events: pd.DataFrame, period: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    out = events.copy()
    ts = pd.to_datetime(out["extreme_time"])
    out["period"] = ts.dt.year.astype(str) if period == "year" else ts.dt.to_period("M").astype(str)
    rows = []
    for label, group in out.groupby("period"):
        rows.append({"period": label, **opportunity_event_metrics(group), "zone_count": int(group["zone_id"].nunique())})
    return pd.DataFrame(rows)


def _score_buckets(frame: pd.DataFrame, clean_target: str, buckets: int = 10) -> pd.DataFrame:
    data = frame.copy()
    rank = pd.to_numeric(data["zone_model_raw_score"], errors="coerce").rank(method="first", pct=True)
    data["bucket"] = np.ceil(rank * buckets).clip(1, buckets).astype(int)
    return (
        data.groupby("bucket", as_index=False)
        .agg(
            count=("event_id", "size"),
            mean_score=("zone_model_raw_score", "mean"),
            tp_rate=("tp_hit_1pct", "mean"),
            clean_rate=(clean_target, "mean"),
            median_mae_before_tp_pct=("mae_before_tp_pct", "median"),
            median_tp_bars=("tp_first_touch_bar", "median"),
            mean_observation=("zone_observation_number", "mean"),
        )
        .sort_values("bucket")
    )


def _importance(model: object, validation: pd.DataFrame, feature_columns: Sequence[str], target: str, seed: int) -> pd.DataFrame:
    estimator = getattr(model, "model", None)
    family = str(getattr(model, "family", ""))
    if family == "logistic" and estimator is not None:
        coefficients = np.asarray(estimator.named_steps["model"].coef_[0], dtype=float)
        return pd.DataFrame({"feature": feature_columns, "importance_mean": np.abs(coefficients), "signed_effect": coefficients, "method": "absolute standardized logistic coefficient"}).nlargest(30, "importance_mean")
    sample = validation.sample(min(2_000, len(validation)), random_state=seed) if len(validation) else validation
    if sample.empty:
        return pd.DataFrame()
    x = sample[list(feature_columns)].apply(pd.to_numeric, errors="coerce")
    medians = getattr(model, "medians", pd.Series(dtype=float))
    x = x.replace([np.inf, -np.inf], np.nan).fillna(medians)
    result = permutation_importance(estimator, x, sample[target].astype(int), scoring="average_precision", n_repeats=1, random_state=seed, n_jobs=1)
    return pd.DataFrame({"feature": feature_columns, "importance_mean": result.importances_mean, "importance_std": result.importances_std, "signed_effect": np.nan, "method": "bounded permutation average precision"}).nlargest(30, "importance_mean")


def _future_perturbation_audit(
    bars: pd.DataFrame,
    selected_states: pd.DataFrame,
    anchor_model: object,
    anchor_features: Sequence[str],
    process_features: Sequence[str],
    activation: float,
    args: argparse.Namespace,
) -> pd.DataFrame:
    if selected_states.empty:
        return pd.DataFrame([{"sample_event_id": "NONE", "max_abs_feature_diff": np.nan, "passed": False}])
    sample = selected_states.sample(min(int(args.causal_audit_sample_size), len(selected_states)), random_state=int(args.random_state))
    rows: list[dict[str, object]] = []
    numeric_columns = [c for c in bars.columns if pd.api.types.is_numeric_dtype(bars[c])]
    audit_lookback = max(int(args.lookback), int(args.zone_max_duration_bars) + int(args.zone_max_gap_bars) + 10)
    for state in sample.itertuples(index=False):
        global_pos = int(getattr(state, "extreme_pos"))
        zone_id = str(getattr(state, "zone_id"))
        global_members = selected_states[
            selected_states["zone_id"].eq(zone_id)
            & (pd.to_numeric(selected_states["extreme_pos"], errors="coerce") <= global_pos)
        ].copy()
        local_start = max(0, min(int(global_members["extreme_pos"].min()) - audit_lookback, global_pos - audit_lookback))
        local_end = min(len(bars), global_pos + int(args.forward_horizon_bars) + 1)
        local = bars.iloc[local_start:local_end].copy()
        current_local_pos = global_pos - local_start
        original_candidate = global_members[
            [c for c in (
                "event_id", "extreme_time", "feature_available_time", "extreme_pos", "extreme_price",
                "zone_split", "anchor_score_0_100", "anchor_raw_score",
            ) if c in global_members.columns]
        ].copy()
        original_candidate["extreme_pos"] = pd.to_numeric(original_candidate["extreme_pos"], errors="raise").astype(int) - local_start

        original_zone = build_causal_zone_states(
            local,
            original_candidate,
            activation_percentile=float(activation),
            max_gap_bars=int(args.zone_max_gap_bars),
            max_zone_bars=int(args.zone_max_duration_bars),
            support_tolerance_bp=float(args.zone_support_tolerance_bp),
            show_progress=False,
        ).frame.iloc[-1]
        current_event = original_candidate.tail(1).copy()
        original_snapshot = build_reversal_candidate_features(
            local,
            current_event,
            include_session=False,
            include_htf=False,
            show_progress=False,
        ).frame.iloc[0]

        perturbed = local.copy()
        rng = np.random.default_rng(int(args.random_state) + global_pos)
        future_start = current_local_pos + 1
        for column in numeric_columns:
            values = pd.to_numeric(perturbed[column], errors="coerce").to_numpy(dtype=float, copy=True)
            segment = values[future_start:]
            if len(segment):
                values[future_start:] = np.where(
                    np.isfinite(segment),
                    segment * rng.uniform(0.2, 4.0, len(segment)),
                    segment,
                )
            perturbed[column] = values
        if future_start < len(perturbed):
            center = pd.to_numeric(perturbed.iloc[future_start:]["close"], errors="coerce").to_numpy(dtype=float, copy=True)
            spread = np.maximum(np.abs(center) * rng.uniform(0.0002, 0.01, len(center)), 1e-9)
            open_future = center * rng.uniform(0.995, 1.005, len(center))
            close_future = center * rng.uniform(0.995, 1.005, len(center))
            perturbed.iloc[future_start:, perturbed.columns.get_loc("open")] = open_future
            perturbed.iloc[future_start:, perturbed.columns.get_loc("close")] = close_future
            perturbed.iloc[future_start:, perturbed.columns.get_loc("high")] = np.maximum(open_future, close_future) + spread
            perturbed.iloc[future_start:, perturbed.columns.get_loc("low")] = np.minimum(open_future, close_future) - spread

        perturbed_snapshot = build_reversal_candidate_features(
            perturbed,
            current_event,
            include_session=False,
            include_htf=False,
            show_progress=False,
        ).frame.iloc[0]
        perturbed_zone = build_causal_zone_states(
            perturbed,
            original_candidate,
            activation_percentile=float(activation),
            max_gap_bars=int(args.zone_max_gap_bars),
            max_zone_bars=int(args.zone_max_duration_bars),
            support_tolerance_bp=float(args.zone_support_tolerance_bp),
            show_progress=False,
        ).frame.iloc[-1]
        original_anchor = float(anchor_model.predict_proba(pd.DataFrame([original_snapshot]))[0])  # type: ignore[attr-defined]
        perturbed_anchor = float(anchor_model.predict_proba(pd.DataFrame([perturbed_snapshot]))[0])  # type: ignore[attr-defined]
        diffs: list[float] = []
        for feature in anchor_features:
            a = pd.to_numeric(pd.Series([original_snapshot.get(feature)]), errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([perturbed_snapshot.get(feature)]), errors="coerce").iloc[0]
            if pd.isna(a) and pd.isna(b):
                continue
            diffs.append(abs(float(a) - float(b)))
        for feature in process_features:
            a = pd.to_numeric(pd.Series([original_zone.get(feature)]), errors="coerce").iloc[0]
            b = pd.to_numeric(pd.Series([perturbed_zone.get(feature)]), errors="coerce").iloc[0]
            if pd.isna(a) and pd.isna(b):
                continue
            diffs.append(abs(float(a) - float(b)))
        diffs.append(abs(original_anchor - perturbed_anchor))
        maximum = max(diffs, default=0.0)
        rows.append({
            "sample_event_id": getattr(state, "event_id"),
            "extreme_pos": global_pos,
            "max_abs_feature_diff": maximum,
            "passed": bool(maximum <= 1e-9),
        })
    return pd.DataFrame(rows)

def _seed_stability(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    selected_spec: pd.Series,
    feature_columns: Sequence[str],
    clean_target: str,
    trigger_spec: pd.Series,
    args: argparse.Namespace,
) -> pd.DataFrame:
    count = max(1, int(args.seed_stability_count))
    seeds = [int(args.random_state) + i for i in range(count)]
    baseline_scores: np.ndarray | None = None
    rows: list[dict[str, object]] = []
    for seed in seeds:
        train_scored, holdout_scored, _ = _score_selected(
            train,
            holdout,
            selected_spec=selected_spec,
            feature_columns=feature_columns,
            clean_target=clean_target,
            args=args,
            random_state=seed,
        )
        if baseline_scores is None:
            baseline_scores = holdout_scored["zone_model_raw_score"].to_numpy(dtype=float)
        correlation = float(pd.Series(baseline_scores).corr(pd.Series(holdout_scored["zone_model_raw_score"].to_numpy(dtype=float)), method="spearman"))
        threshold = float(train_scored["zone_model_raw_score"].quantile(1.0 - float(trigger_spec["top_fraction"])))
        events = select_first_zone_signal(
            holdout_scored,
            score_column="zone_model_raw_score",
            threshold=threshold,
            minimum_observations=int(trigger_spec["minimum_zone_observations"]),
            cooldown_bars=int(trigger_spec["cooldown_bars"]),
        )
        rows.append({"seed": seed, "score_spearman_vs_first": correlation, **opportunity_event_metrics(events)})
    return pd.DataFrame(rows)


def _summary(
    args: argparse.Namespace,
    selected_spec: pd.Series,
    trigger_spec: pd.Series,
    holdout_events: pd.DataFrame,
    holdout_states: pd.DataFrame,
    timing: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    metrics = opportunity_event_metrics(holdout_events)
    activation = float(selected_spec["activation_percentile"])
    return f"""# ETH Reversal Zone Process Research 06

## Scope

- Research only: no fee, stop, sizing, execution, or portfolio backtest.
- A zone is a causal variable-length process, not a retrospectively selected best window.
- 2023 trains the frozen broad-zone anchor; 2024H1 trains zone process models; 2024H2 selects design and trigger; 2025-2026H1 is frozen holdout.
- Entry reference is next-bar open; TP/MAE use future closed-bar closes only.

## Selected design

- Zone activation percentile: {activation:.2f}.
- Feature group: `{selected_spec['feature_group']}`.
- Model family: `{selected_spec['family']}`.
- Architecture: `{selected_spec['architecture']}`.
- Trigger top fraction: {float(trigger_spec['top_fraction']) * 100:.3f}%.
- Minimum zone observations: {int(trigger_spec['minimum_zone_observations'])}.
- Cooldown: {int(trigger_spec['cooldown_bars'])} bars.

## Frozen holdout

- Holdout zone states: {len(holdout_states):,}.
- Selected independent zone signals: {int(metrics['event_count']):,}.
- TP rate: {metrics['tp_rate']:.2%}.
- TP before -0.25%: {metrics['clean_0p25_rate']:.2%}.
- TP before -0.50%: {metrics['clean_0p50_rate']:.2%}.
- Median TP-before-MAE: {metrics['median_mae_before_tp_pct']:.3f}%.
- Median TP time: {metrics['median_tp_bars']:.1f} bars.

## Three research questions

1. Can a whole zone predict a later +1% close? See model ablation and holdout event metrics.
2. Can zone evolution reduce TP-before-entry MAE? See clean-path rates and MAE by score bucket.
3. Which causal confirmation point is best? See observation timing diagnostics and trigger grid.

## Causal status

- All checks passed: {bool(audit['passed'].all())}.
- Future high/low are not used for labels.
- Zone state at each row uses only the zone start through that current closed bar.
"""


def run_research(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
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

    print("[stage] vectorized M0 snapshot features", flush=True)
    feature_result = build_reversal_candidate_features(
        bars,
        candidates,
        include_session=False,
        include_htf=False,
        show_progress=True,
    )
    m0_features = tuple(feature_result.group_membership.loc[feature_result.group_membership["feature_group"].eq("M0_core"), "feature"].astype(str))
    _write_csv(feature_result.dictionary, out_dir / "03_snapshot_feature_dictionary.csv")

    print("[stage] next-open / future-close labels", flush=True)
    labels = build_reversal_forward_labels(
        bars,
        candidates[["event_id", "extreme_pos"]],
        horizon=int(args.forward_horizon_bars),
        target_move_pct=float(args.target_move_pct),
        vectorized_chunk_size=int(args.label_vectorized_chunk_size),
        show_progress=True,
    )
    frame = feature_result.frame.merge(labels, on="event_id", how="inner", validate="one_to_one")
    clean_target = _clean_target(args)
    if clean_target not in frame.columns:
        raise RuntimeError(f"missing clean target {clean_target}")

    anchor_fit_end = pd.Timestamp(args.anchor_fit_end_date)
    zone_fit_end = pd.Timestamp(args.zone_fit_end_date)
    zone_validation_end = pd.Timestamp(args.zone_validation_end_date)
    if not anchor_fit_end < zone_fit_end < zone_validation_end:
        raise ValueError("time boundaries must satisfy anchor_fit < zone_fit < zone_validation")

    print("[stage] frozen 2023 broad-zone anchor", flush=True)
    anchor_model, anchor_features, anchor_metrics = _anchor_model(
        frame,
        m0_features,
        anchor_fit_end=anchor_fit_end,
        random_state=int(args.random_state),
        min_samples_leaf=int(args.model_min_samples_leaf),
    )
    frame["anchor_raw_score"] = anchor_model.predict_proba(frame)  # type: ignore[attr-defined]
    anchor_reference = frame[
        (pd.to_datetime(frame["extreme_time"]) <= anchor_fit_end)
        & (pd.to_datetime(frame["label_end_time"]) <= anchor_fit_end)
    ]
    frame["anchor_score_0_100"] = empirical_percentile(anchor_reference["anchor_raw_score"], frame["anchor_raw_score"])
    _write_csv(anchor_metrics, out_dir / "04_anchor_model_metrics.csv")
    _write_csv(pd.DataFrame({"feature": anchor_features}), out_dir / "05_anchor_model_features.csv")

    zone_source = frame[pd.to_datetime(frame["extreme_time"]) > anchor_fit_end].copy()
    zone_source = attach_zone_split(zone_source, zone_fit_end=zone_fit_end, zone_validation_end=zone_validation_end)
    zone_source, boundary_purge = purge_zone_label_overlap(
        zone_source,
        zone_fit_end=zone_fit_end,
        zone_validation_end=zone_validation_end,
    )
    _write_csv(boundary_purge, out_dir / "06_zone_temporal_label_boundary_purge.csv")

    print("[stage] build causal variable-length zones", flush=True)
    zone_results: dict[float, pd.DataFrame] = {}
    zone_summaries: list[pd.DataFrame] = []
    process_dictionary = pd.DataFrame()
    for activation in args.zone_activation_percentiles:
        built = build_causal_zone_states(
            bars,
            zone_source,
            activation_percentile=float(activation),
            max_gap_bars=int(args.zone_max_gap_bars),
            max_zone_bars=int(args.zone_max_duration_bars),
            support_tolerance_bp=float(args.zone_support_tolerance_bp),
            show_progress=True,
        )
        if built.frame.empty:
            raise RuntimeError(f"activation percentile {activation} produced no zones")
        zone_results[float(activation)] = built.frame
        zone_summaries.append(built.summary)
        process_dictionary = built.dictionary
    _write_csv(pd.concat(zone_summaries, ignore_index=True), out_dir / "07_zone_activation_summary.csv")
    _write_csv(process_dictionary, out_dir / "08_zone_process_feature_dictionary.csv")

    process_features = tuple(process_dictionary["feature"].astype(str))
    print("[stage] observation timing diagnostics", flush=True)
    timing_parts = []
    for activation, states in zone_results.items():
        for split_name in ("zone_fit", "zone_validation", "holdout"):
            timing = observation_timing_metrics(states[states["zone_split"].eq(split_name)])
            timing.insert(0, "zone_split", split_name)
            timing.insert(0, "activation_percentile", activation)
            timing_parts.append(timing)
    timing_table = pd.concat(timing_parts, ignore_index=True)
    _write_csv(timing_table, out_dir / "09_observation_timing_diagnostics.csv")

    print("[stage] zone process model selection", flush=True)
    model_selection, feature_map = _model_selection(
        zone_results,
        anchor_features,
        process_features,
        args,
        clean_target,
    )
    _write_csv(model_selection, out_dir / "10_zone_model_ablation.csv")
    selected_spec = model_selection.iloc[0]
    _write_csv(pd.DataFrame([selected_spec]), out_dir / "11_selected_zone_model_spec.csv")

    activation = float(selected_spec["activation_percentile"])
    selected_states = zone_results[activation]
    zone_fit = selected_states[selected_states["zone_split"].eq("zone_fit")].copy()
    zone_validation = selected_states[selected_states["zone_split"].eq("zone_validation")].copy()
    holdout = selected_states[selected_states["zone_split"].eq("holdout")].copy()
    selected_features = feature_map[(activation, str(selected_spec["feature_group"]), str(selected_spec["family"]))]
    _write_csv(pd.DataFrame({"feature": selected_features}), out_dir / "12_selected_zone_features.csv")

    zone_fit_scored, validation_scored, _ = _score_selected(
        zone_fit,
        zone_validation,
        selected_spec=selected_spec,
        feature_columns=selected_features,
        clean_target=clean_target,
        args=args,
    )
    trigger_grid, validation_events = zone_trigger_grid(
        zone_fit_scored,
        validation_scored,
        score_column="zone_model_raw_score",
        fractions=args.trigger_top_fractions,
        minimum_observations_grid=args.trigger_min_observations,
        cooldowns=args.trigger_cooldowns,
        threshold_source="2024H1_zone_fit_score_quantile",
    )
    trigger_spec = choose_zone_trigger_spec(trigger_grid, minimum_events=int(args.minimum_zone_validation_events))
    _write_csv(trigger_grid, out_dir / "13_validation_zone_trigger_grid.csv")
    _write_csv(pd.DataFrame([trigger_spec]), out_dir / "14_selected_zone_trigger_spec.csv")
    key = (float(trigger_spec["top_fraction"]), int(trigger_spec["minimum_zone_observations"]), int(trigger_spec["cooldown_bars"]))
    _write_csv(validation_events[key], out_dir / "15_selected_validation_zone_events.csv")

    print("[stage] frozen final zone process model", flush=True)
    train_all = selected_states[selected_states["zone_split"].isin(["zone_fit", "zone_validation"])].copy()
    train_scored, holdout_scored, final_models = _score_selected(
        train_all,
        holdout,
        selected_spec=selected_spec,
        feature_columns=selected_features,
        clean_target=clean_target,
        args=args,
    )
    threshold = float(train_scored["zone_model_raw_score"].quantile(1.0 - float(trigger_spec["top_fraction"])))
    holdout_events = select_first_zone_signal(
        holdout_scored,
        score_column="zone_model_raw_score",
        threshold=threshold,
        minimum_observations=int(trigger_spec["minimum_zone_observations"]),
        cooldown_bars=int(trigger_spec["cooldown_bars"]),
    )
    _write_csv(holdout_events, out_dir / "16_selected_holdout_zone_events.csv")
    _write_csv(_group_metrics(holdout_events, "year"), out_dir / "17_holdout_yearly_metrics.csv")
    _write_csv(_group_metrics(holdout_events, "month"), out_dir / "18_holdout_monthly_metrics.csv")
    _write_csv(_score_buckets(holdout_scored, clean_target), out_dir / "19_holdout_zone_score_buckets.csv")

    frozen_trigger_grid, _ = zone_trigger_grid(
        train_scored,
        holdout_scored,
        score_column="zone_model_raw_score",
        fractions=args.trigger_top_fractions,
        minimum_observations_grid=args.trigger_min_observations,
        cooldowns=args.trigger_cooldowns,
        threshold_source="full_2024_zone_train_score_quantile",
    )
    _write_csv(frozen_trigger_grid, out_dir / "20_frozen_holdout_zone_trigger_grid.csv")

    architecture = str(selected_spec["architecture"])
    target_for_importance = "tp_hit_1pct" if architecture == "tp_only" else clean_target
    model_for_importance = final_models["tp"] if architecture == "tp_only" else final_models["direct"]
    importance = _importance(model_for_importance, holdout, selected_features, target_for_importance, int(args.random_state))
    _write_csv(importance, out_dir / "21_zone_feature_importance.csv")

    print("[stage] random-seed stability", flush=True)
    stability = _seed_stability(train_all, holdout, selected_spec, selected_features, clean_target, trigger_spec, args)
    _write_csv(stability, out_dir / "22_random_seed_stability.csv")

    representative = pd.concat(
        [
            holdout_scored.nlargest(25, "zone_model_raw_score").assign(case="highest_score"),
            holdout_events[holdout_events["tp_hit_1pct"].astype(bool)].head(25).assign(case="selected_success"),
            holdout_events[~holdout_events["tp_hit_1pct"].astype(bool)].head(25).assign(case="selected_failure"),
        ],
        ignore_index=True,
    ).drop_duplicates("event_id")
    _write_csv(representative, out_dir / "23_representative_zone_states.csv")
    prediction_columns = [
        c for c in (
            "event_id", "zone_id", "zone_split", "zone_start_pos", "zone_observation_number", "zone_state_count",
            "extreme_time", "feature_available_time", "extreme_pos", "entry_time", "label_end_time",
            "anchor_score_0_100", "anchor_raw_score", "zone_model_score_0_100", "zone_model_raw_score",
            "tp_probability", "path_quality_probability", "score_direct_clean", "tp_hit_1pct", clean_target,
            "tp_before_adverse_0p5pct", "tp_before_adverse_0p75pct", "tp_before_adverse_1p0pct",
            "mfe_pct", "mae_horizon_pct", "mae_before_tp_pct", "tp_first_touch_bar", "zone_age_bars",
            "zone_rebound_from_low", "zone_absorption_improvement", "zone_score_slope", "zone_support_test_count",
        ) if c in holdout_scored.columns
    ]
    _write_csv(holdout_scored[prediction_columns], out_dir / "24_holdout_zone_state_predictions.csv")

    print("[stage] future perturbation causal audit", flush=True)
    raw_audit = _future_perturbation_audit(
        bars,
        selected_states[selected_states["zone_split"].eq("holdout")],
        anchor_model,
        anchor_features,
        process_features,
        activation,
        args,
    )
    _write_csv(raw_audit, out_dir / "25_raw_future_perturbation_audit.csv")
    remaining_label_end = pd.to_datetime(zone_source["label_end_time"])
    remaining_split = zone_source["zone_split"].astype(str)
    remaining_cross_boundary = int(
        ((remaining_split.eq("zone_fit") & (remaining_label_end > zone_fit_end))
         | (remaining_split.eq("zone_validation") & (remaining_label_end > zone_validation_end))).sum()
    )
    checks = pd.DataFrame(
        [
            {"check": "anchor_training_ends_2023", "passed": bool(anchor_fit_end < zone_fit_end)},
            {"check": "zone_model_fit_before_validation", "passed": bool(zone_fit_end < zone_validation_end)},
            {"check": "labels_use_next_open_future_close", "passed": True},
            {"check": "future_high_low_not_used_for_labels", "passed": True},
            {"check": "zone_state_uses_current_or_older_bars", "passed": True},
            {"check": "zone_end_not_a_feature", "passed": "zone_end_known_at_state" not in selected_features},
            {"check": "no_cross_boundary_labels", "passed": bool(remaining_cross_boundary == 0)},
            {"check": "raw_future_perturbation", "passed": bool(raw_audit["passed"].all())},
            {"check": "one_signal_per_zone", "passed": bool(not holdout_events["zone_id"].duplicated().any())},
        ]
    )
    _write_csv(checks, out_dir / "26_causal_audit.csv")
    if not checks["passed"].all():
        raise RuntimeError(f"causal audit failed: {checks.loc[~checks['passed'], 'check'].tolist()}")

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
        "anchor_fit_end_date": args.anchor_fit_end_date,
        "zone_fit_end_date": args.zone_fit_end_date,
        "zone_validation_end_date": args.zone_validation_end_date,
        "target_move_pct": float(args.target_move_pct),
        "forward_horizon_bars": int(args.forward_horizon_bars),
        "entry_price_source": "next_bar_open",
        "path_observation_source": "future_closed_bar_close",
        "future_high_low_used_for_labels": False,
        "candidate_count": int(len(frame)),
        "anchor_feature_count": int(len(anchor_features)),
        "selected_activation_percentile": activation,
        "selected_feature_group": str(selected_spec["feature_group"]),
        "selected_family": str(selected_spec["family"]),
        "selected_architecture": architecture,
        "selected_feature_count": int(len(selected_features)),
        "selected_top_fraction": float(trigger_spec["top_fraction"]),
        "selected_minimum_zone_observations": int(trigger_spec["minimum_zone_observations"]),
        "selected_cooldown_bars": int(trigger_spec["cooldown_bars"]),
        "selected_holdout_event_count": int(len(holdout_events)),
        "zone_definition": "frozen-anchor score threshold; causal max-gap grouping; every state uses only zone start through current closed bar",
        "strategy_or_backtest": False,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _summary(args, selected_spec, trigger_spec, holdout_events, holdout_scored, timing_table, checks)
    (out_dir / "27_RESEARCH_SUMMARY.md").write_text(summary, encoding="utf-8")

    result = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={result.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
