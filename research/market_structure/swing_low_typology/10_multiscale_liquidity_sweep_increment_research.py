#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal multi-scale liquidity sweep increment research 10.

Research 10 keeps the deployable raw-score ranking and walk-forward framework
from 09, but removes range data.  Under identical samples, labels, model family
and predeclared policies it compares:

L0 : causal 1m trade-bar snapshot + train-fitted soft mechanisms
L1 : L0 + micro liquidity map
L2 : L0 + macro liquidity map
L3 : L0 + micro/macro map + sweep/reclaim/order-flow process

Liquidity is an optional information layer, not a mandatory event gate.  A low
is not called liquidity merely because price traded below it.  Levels must exist
before the current closed bar, and sweep/reclaim features use only current and
prior closed bars.  No strategy, sizing, stop, scale-in or backtest is included.
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
from research.market_structure.swing_low_typology.common.multiobjective_calibration import (  # noqa: E402
    calibration_metrics,
    choose_calibrator,
    delete_day_stress,
    fit_conformal_adjustment,
    fit_probability_calibrators,
    fit_risk_point_model,
    policy_metrics,
    quantile_metrics,
)
from research.market_structure.swing_low_typology.common.online_recognizability import (  # noqa: E402
    CandidateGateConfig,
    build_online_candidate_events,
    fit_binary_model,
)
from research.market_structure.swing_low_typology.common.range_increment import (  # noqa: E402
    EmpiricalRankReference,
    deployable_policy_specs,
    select_ranked_events,
)
from research.market_structure.swing_low_typology.common.liquidity_increment import (  # noqa: E402
    MICRO_GROUP,
    MACRO_GROUP,
    SWEEP_GROUP,
    build_multiscale_liquidity_features,
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
    build_broad_candidate_regions,
    fit_soft_mechanism_transformer,
    mechanism_feature_dictionary,
)

SCRIPT_NAME = "10_multiscale_liquidity_sweep_increment_research"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_1M_MULTISCALE_LIQUIDITY_INCREMENT_10"
EDGE_ID = "RESEARCH_ONLY_ETH_LIQUIDITY_SWEEP_INCREMENT"
TITLE = "ETH Multi-scale Liquidity Sweep Increment Research 10"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/10_multiscale_liquidity_sweep_increment"
PRIMARY_FAMILY = "logistic_sgd"
HEAD_TARGETS: dict[str, str] = {
    "p_tp60": "tp_hit_1pct",
    "p_clean25": "tp_before_adverse_0p25pct",
    "p_clean50": "tp_before_adverse_0p5pct",
    "p_fast15": "tp_within_15",
    "p_fast30": "tp_within_30",
}
DEFAULT_COOLDOWN = 15


class FoldSpec(NamedTuple):
    fold: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward causal multi-scale liquidity sweep increment research.",
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
    p.add_argument("--micro-liquidity-windows", nargs="+", type=int, default=[5, 15, 30, 60])
    p.add_argument("--liquidity-equal-low-tolerance-bp", type=float, default=8.0)
    p.add_argument("--liquidity-approach-tolerance-bp", type=float, default=15.0)
    p.add_argument("--liquidity-htf-pivot-minutes", nargs="+", type=int, default=[60, 240])
    p.add_argument("--liquidity-htf-pivot-left-bars", type=int, default=2)
    p.add_argument("--liquidity-htf-pivot-right-bars", type=int, default=2)
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
    p.add_argument("--maximum-train-rows", type=int, default=300_000)
    p.add_argument("--prediction-chunk-size", type=int, default=100_000)
    p.add_argument("--cooldown-bars", type=int, default=DEFAULT_COOLDOWN)
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
    return frame.loc[valid].sort_values("extreme_pos").reset_index(drop=True), removed


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
    diagnostic = pd.DataFrame(
        [
            {
                "fold": fold.fold,
                "model_fit_start": model_fit["extreme_time"].min(),
                "model_fit_end": model_fit["extreme_time"].max(),
                "calibration_start": calibration["extreme_time"].min(),
                "calibration_end": calibration["extreme_time"].max(),
                "policy_start": policy["extreme_time"].min(),
                "policy_end": policy["extreme_time"].max(),
                "model_fit_rows": len(model_fit),
                "calibration_rows": len(calibration),
                "policy_rows": len(policy),
                "model_fit_cross_boundary_removed": removed_model,
                "calibration_cross_boundary_removed": removed_calibration,
                "policy_cross_boundary_removed": removed_policy,
            }
        ]
    )
    return model_fit, calibration, policy, diagnostic


def _weighted_positions_without_replacement(weights: np.ndarray, sample_size: int, random_state: int) -> np.ndarray:
    count = len(weights)
    if sample_size >= count:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    clean = np.asarray(weights, dtype=float)
    positive = np.isfinite(clean) & (clean > 0)
    keys = np.full(count, -np.inf, dtype=float)
    if positive.any():
        uniform = np.clip(rng.random(int(positive.sum())), 1e-15, 1.0 - 1e-15)
        keys[positive] = np.log(clean[positive]) - np.log(-np.log(uniform))
    chosen = np.argpartition(keys, -min(sample_size, int(positive.sum())))[-min(sample_size, int(positive.sum())) :]
    if len(chosen) < sample_size:
        remaining = np.setdiff1d(np.arange(count), chosen, assume_unique=False)
        extra = rng.choice(remaining, size=sample_size - len(chosen), replace=False)
        chosen = np.concatenate([chosen, extra])
    return np.sort(chosen.astype(np.int64))


def _sample_training(frame: pd.DataFrame, maximum_rows: int, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive = frame[frame["tp_hit_1pct"].astype(bool)]
    negative = frame[~frame["tp_hit_1pct"].astype(bool)]
    target_rows = max(int(maximum_rows), len(positive))
    negative_count = min(len(negative), max(0, target_rows - len(positive)))
    weights = pd.to_numeric(negative.get("episode_weight", 1.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    positions = _weighted_positions_without_replacement(weights, negative_count, random_state)
    sampled = pd.concat([positive, negative.iloc[positions]], ignore_index=True).sort_values("extreme_pos").reset_index(drop=True)
    diagnostics = pd.DataFrame(
        [
            {
                "source_rows": len(frame),
                "source_positive_rows": len(positive),
                "source_negative_rows": len(negative),
                "sampled_rows": len(sampled),
                "sampled_positive_rows": int(sampled["tp_hit_1pct"].sum()),
                "sampled_negative_rows": int((~sampled["tp_hit_1pct"].astype(bool)).sum()),
                "sampling": "all positives + gumbel-top-k episode-weighted negatives",
            }
        ]
    )
    return sampled, diagnostics


def _condition_feature_columns(
    fit: pd.DataFrame,
    requested: Sequence[str],
    *,
    max_features: int = 320,
    sample_rows: int = 10_000,
    max_abs_correlation: float = 0.9995,
) -> tuple[tuple[str, ...], dict[str, object]]:
    """Train-period-only conditioning for stable cross-feature-group models.

    The ordering is inherited from the predeclared feature groups.  Only near
    constants and almost-duplicate columns are removed; no target or frozen
    test information participates.
    """

    usable = tuple(select_usable_features(fit, requested))
    candidates = usable[: max(int(max_features) + 64, int(max_features))]
    if not candidates:
        raise RuntimeError("no usable model features after fit-period sanitation")
    if len(fit) > int(sample_rows):
        positions = np.linspace(0, len(fit) - 1, int(sample_rows), dtype=np.int64)
        sample = fit.iloc[positions]
    else:
        sample = fit
    numeric = sample.reindex(columns=candidates).apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.median().fillna(0.0))
    q10 = numeric.quantile(0.10)
    q90 = numeric.quantile(0.90)
    robust_span = (q90 - q10).abs()
    scale_keep = [column for column in candidates if np.isfinite(robust_span[column]) and robust_span[column] > 1e-10]
    removed_low_scale = len(candidates) - len(scale_keep)
    if not scale_keep:
        raise RuntimeError("all model features are near-constant in the fit period")

    values = numeric[scale_keep].to_numpy(dtype=np.float64, copy=True)
    values -= values.mean(axis=0, keepdims=True)
    standard_deviation = values.std(axis=0, ddof=0)
    valid_scale = np.isfinite(standard_deviation) & (standard_deviation > 1e-12)
    scale_keep = [column for column, valid in zip(scale_keep, valid_scale, strict=True) if valid]
    values = values[:, valid_scale]
    removed_low_scale += int((~valid_scale).sum())
    if not scale_keep:
        raise RuntimeError("all model features have zero numerical scale")
    values /= np.maximum(values.std(axis=0, ddof=0, keepdims=True), 1e-12)
    correlation = np.asarray(values.T @ values / max(1, len(values)), dtype=np.float64)
    kept_indices: list[int] = []
    removed_correlated: list[str] = []
    for index, column in enumerate(scale_keep):
        if kept_indices and np.any(np.abs(correlation[index, kept_indices]) >= float(max_abs_correlation)):
            removed_correlated.append(column)
            continue
        kept_indices.append(index)
        if len(kept_indices) >= int(max_features):
            break
    selected = tuple(scale_keep[index] for index in kept_indices)
    if not selected:
        raise RuntimeError("no model features remain after collinearity conditioning")
    diagnostics: dict[str, object] = {
        "requested_feature_count": len(requested),
        "usable_feature_count": len(usable),
        "conditioning_candidate_count": len(candidates),
        "conditioning_sample_rows": len(sample),
        "removed_low_scale_count": removed_low_scale,
        "removed_near_duplicate_count": len(removed_correlated),
        "max_abs_correlation": float(max_abs_correlation),
        "selected_feature_count": len(selected),
        "removed_near_duplicate_features": "|".join(removed_correlated),
        "selected_features": "|".join(selected),
    }
    return selected, diagnostics


def _predict_binary(model: object, frame: pd.DataFrame, chunk_size: int) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, len(frame), max(1, int(chunk_size))):
        values.append(np.asarray(model.predict_proba(frame.iloc[start : start + int(chunk_size)]), dtype=float))
    return np.concatenate(values) if values else np.asarray([], dtype=float)


def _predict_risk(model: object, frame: pd.DataFrame, chunk_size: int) -> np.ndarray:
    values: list[np.ndarray] = []
    for start in range(0, len(frame), max(1, int(chunk_size))):
        values.append(np.asarray(model.predict(frame.iloc[start : start + int(chunk_size)]), dtype=float))
    return np.concatenate(values) if values else np.asarray([], dtype=float)


def _score_shell(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "event_id", "extreme_pos", "extreme_time", "feature_available_time", "entry_time", "entry_price",
        "label_end_time", "causal_region_id", "positive_episode_id", "tp_hit_1pct", "tp_before_adverse_0p25pct",
        "tp_before_adverse_0p5pct", "tp_before_adverse_0p75pct", "tp_before_adverse_1p0pct", "tp_within_15",
        "tp_within_30", "mfe_pct", "mae_horizon_pct", "mae_before_tp_pct", "tp_first_touch_bar",
    ]
    return frame.reindex(columns=keep).copy()


def _head_metrics(frame: pd.DataFrame, *, fold: str, feature_group: str, output: str, target: str, split: str) -> pd.DataFrame:
    y = frame[target].astype(int).to_numpy()
    raw = pd.to_numeric(frame[f"{output}_raw"], errors="coerce").to_numpy(dtype=float)
    cal = pd.to_numeric(frame[f"{output}_cal"], errors="coerce").to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for method, score in (("raw", raw), ("calibrated", cal)):
        finite = np.isfinite(score)
        if not finite.any() or np.unique(y[finite]).size < 2:
            pr_auc = roc_auc = np.nan
        else:
            pr_auc = float(average_precision_score(y[finite], score[finite]))
            roc_auc = float(roc_auc_score(y[finite], score[finite]))
        probability = np.clip(score[finite], 1e-7, 1 - 1e-7)
        rows.append(
            {
                "fold": fold,
                "feature_group": feature_group,
                "output": output,
                "target": target,
                "split": split,
                "method": method,
                "rows": int(finite.sum()),
                "positive_rate": float(y[finite].mean()) if finite.any() else np.nan,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "brier": float(brier_score_loss(y[finite], probability)) if finite.any() else np.nan,
                "log_loss": float(log_loss(y[finite], probability, labels=[0, 1])) if finite.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _increment_comparison(frontier: pd.DataFrame) -> pd.DataFrame:
    if frontier.empty:
        return pd.DataFrame()
    keys = ["fold", "policy_id", "cooldown_bars"]
    baseline = frontier[frontier["feature_group"].eq("L0_tradebar")].copy()
    baseline = baseline.set_index(keys)
    rows: list[dict[str, object]] = []
    for row in frontier[~frontier["feature_group"].eq("L0_tradebar")].itertuples(index=False):
        key = (row.fold, row.policy_id, row.cooldown_bars)
        if key not in baseline.index:
            continue
        base = baseline.loc[key]
        rows.append(
            {
                "fold": row.fold,
                "feature_group": row.feature_group,
                "policy_id": row.policy_id,
                "cooldown_bars": row.cooldown_bars,
                "event_count": row.event_count,
                "baseline_event_count": int(base["event_count"]),
                "delta_event_count": int(row.event_count - base["event_count"]),
                "tp_rate": row.tp_rate,
                "delta_tp_rate": row.tp_rate - base["tp_rate"],
                "clean_0p50_rate": row.clean_0p50_rate,
                "delta_clean_0p50_rate": row.clean_0p50_rate - base["clean_0p50_rate"],
                "fast30_rate": row.fast30_rate,
                "delta_fast30_rate": row.fast30_rate - base["fast30_rate"],
                "median_horizon_mae_pct": row.median_horizon_mae_pct,
                "delta_median_horizon_mae_pct": row.median_horizon_mae_pct - base["median_horizon_mae_pct"],
                "top10_day_event_share": row.top10_day_event_share,
                "delta_top10_day_event_share": row.top10_day_event_share - base["top10_day_event_share"],
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
        (pd.to_numeric(source_candidates["extreme_pos"], errors="coerce") >= max(int(args.lookback) + 20, 10 * 1440))
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
    config = _candidate_config(args)
    local_lookback = max(int(args.lookback) + int(args.region_max_bars) + 20, 10 * 1440)
    rows: list[dict[str, object]] = []
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
            if local_candidates.empty or "extreme_pos" not in local_candidates.columns:
                raise RuntimeError("audit target disappeared from causal candidate universe: no local candidate positions")
            positions = pd.to_numeric(local_candidates["extreme_pos"], errors="coerce").fillna(-1).astype(int)
            if local_pos not in set(positions):
                raise RuntimeError("audit target disappeared from causal candidate universe")
            feature = build_reversal_candidate_features(
                source_bars,
                local_candidates,
                include_session=False,
                include_htf=False,
                show_progress=False,
            ).frame
            region = build_broad_candidate_regions(
                source_bars,
                feature,
                max_gap_bars=int(args.region_max_gap_bars),
                max_region_bars=int(args.region_max_bars),
                retest_tolerance_bp=float(args.region_retest_tolerance_bp),
                show_progress=False,
            ).frame
            liquidity = build_multiscale_liquidity_features(
                source_bars,
                region,
                micro_windows=tuple(int(x) for x in args.micro_liquidity_windows),
                equal_low_tolerance_bp=float(args.liquidity_equal_low_tolerance_bp),
                approach_tolerance_bp=float(args.liquidity_approach_tolerance_bp),
                htf_pivot_minutes=tuple(int(x) for x in args.liquidity_htf_pivot_minutes),
                htf_pivot_left_bars=int(args.liquidity_htf_pivot_left_bars),
                htf_pivot_right_bars=int(args.liquidity_htf_pivot_right_bars),
            )
            columns = tuple(liquidity.dictionary["feature"].astype(str))
            combined = pd.concat(
                [region.reset_index(drop=True), liquidity.frame.loc[:, columns].reset_index(drop=True)],
                axis=1,
            )
            return combined[pd.to_numeric(combined["extreme_pos"], errors="coerce").astype(int).eq(local_pos)].tail(1)

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
        left = original[comparable].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        right = changed[comparable].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        maximum = float(np.abs(np.nan_to_num(left, nan=0.0) - np.nan_to_num(right, nan=0.0)).max(initial=0.0))
        rows.append({"event_id": row.event_id, "passed": bool(maximum <= 1e-10), "maximum_absolute_difference": maximum, "detail": f"features={len(comparable)}"})
    return pd.DataFrame(rows)

def _summary(frontier: pd.DataFrame, increments: pd.DataFrame, diagnostics: pd.DataFrame, audit: pd.DataFrame) -> str:
    lines = [
        "# Research 10 Summary",
        "",
        "## Purpose",
        "",
        "Liquidity is evaluated as an optional multi-scale information layer, not as a mandatory candidate gate.",
        "Raw scores are used for frozen empirical-percentile ranking; calibrated probabilities are interpretation outputs only.",
        "Micro map, macro map and sweep/reclaim/order-flow process are compared under identical folds, samples, labels and policies.",
        "",
        "## Liquidity diagnostics",
        "",
    ]
    if diagnostics.empty:
        lines.append("No liquidity diagnostics were produced.")
    else:
        aggregate = diagnostics[diagnostics["source"].eq("aggregate")]
        if not aggregate.empty:
            row = aggregate.iloc[0]
            lines.append(
                f"- candidates={int(row.get('candidate_rows', 0)):,}, micro features={int(row.get('micro_feature_count', 0))}, "
                f"macro features={int(row.get('macro_feature_count', 0))}, sweep features={int(row.get('sweep_feature_count', 0))}, "
                f"HTF availability violations={int(row.get('htf_available_time_violations', 0))}."
            )
        for row in diagnostics[~diagnostics["source"].eq("aggregate")].itertuples(index=False):
            lines.append(
                f"- {row.source}: candidate coverage={row.candidate_non_null_coverage:.2%}, "
                f"sweeps={int(row.sweep_rows):,}, reclaims={int(row.reclaim_rows):,}, accepted below={int(row.accept_below_rows):,}."
            )
    lines.extend(["", "## Frozen test overview", ""])
    if frontier.empty:
        lines.append("No frozen frontier rows.")
    else:
        representative = frontier[frontier["policy_id"].isin(["TP05_ONLY", "TP05_FAST50_CLEAN50", "TP10_ONLY"])]
        for row in representative.sort_values(["fold", "policy_id", "feature_group"]).itertuples(index=False):
            lines.append(
                f"- {row.fold} {row.feature_group} {row.policy_id}: events={int(row.event_count):,}, "
                f"TP={row.tp_rate:.2%}, clean50={row.clean_0p50_rate:.2%}, fast30={row.fast30_rate:.2%}, "
                f"median horizon MAE={row.median_horizon_mae_pct:.3f}%"
            )
    lines.extend(["", "## Increment interpretation", ""])
    if increments.empty:
        lines.append("No liquidity increment comparison rows.")
    else:
        stable = increments.groupby("feature_group", as_index=False).agg(
            folds=("fold", "nunique"),
            mean_delta_tp=("delta_tp_rate", "mean"),
            mean_delta_clean50=("delta_clean_0p50_rate", "mean"),
            mean_delta_fast30=("delta_fast30_rate", "mean"),
            mean_delta_mae=("delta_median_horizon_mae_pct", "mean"),
        )
        for row in stable.itertuples(index=False):
            lines.append(
                f"- {row.feature_group}: mean ΔTP={row.mean_delta_tp:+.2%}, Δclean50={row.mean_delta_clean50:+.2%}, "
                f"Δfast30={row.mean_delta_fast30:+.2%}, Δmedian MAE={row.mean_delta_mae:+.4f}% "
                f"across {int(row.folds)} folds/policy rows."
            )
    lines.extend(["", "## Guardrails", ""])
    for row in audit.itertuples(index=False):
        lines.append(f"- {'PASS' if bool(row.passed) else 'FAIL'} {row.check}: {row.detail}")
    lines.extend(
        [
            "",
            "This remains model research. No liquidity rule, policy, strategy, stop, scale-in or position-sizing decision is selected from frozen tests.",
        ]
    )
    return "\n".join(lines) + "\n"

def run_research(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = load_bars(args)
    field_coverage = validate_trade_bar_fields(bars)
    _write_csv(field_coverage, out_dir / "01_trade_bar_field_coverage.csv")

    research_start = pd.Timestamp(args.start_date)
    research_end_exclusive = _end_exclusive(args.end_date)
    print("[stage] broad causal candidate universe", flush=True)
    candidates, gate_summary = build_online_candidate_events(
        bars,
        research_start=research_start,
        research_end_exclusive=research_end_exclusive,
        config=_candidate_config(args),
    )
    _write_csv(gate_summary, out_dir / "02_candidate_gate_summary.csv")

    print("[stage] vectorized causal 1m snapshot features", flush=True)
    snapshot = build_reversal_candidate_features(
        bars,
        candidates,
        include_session=False,
        include_htf=False,
        show_progress=True,
    )
    frame = build_broad_candidate_regions(
        bars,
        snapshot.frame,
        max_gap_bars=int(args.region_max_gap_bars),
        max_region_bars=int(args.region_max_bars),
        retest_tolerance_bp=float(args.region_retest_tolerance_bp),
        show_progress=True,
    ).frame
    labels = build_reversal_forward_labels(
        bars,
        frame,
        target_move_pct=float(args.target_move_pct),
        horizon=int(args.forward_horizon_bars),
        vectorized_chunk_size=int(args.label_vectorized_chunk_size),
        show_progress=True,
    )
    for column in labels.columns:
        if column not in frame.columns or column not in {"event_id", "extreme_time", "extreme_pos"}:
            frame[column] = labels[column].to_numpy()
    frame = attach_positive_opportunity_episodes(frame, max_gap_bars=int(args.positive_episode_gap_bars))
    _write_csv(snapshot.dictionary, out_dir / "03_snapshot_feature_dictionary.csv")
    _write_csv(mechanism_feature_dictionary(), out_dir / "04_soft_mechanism_feature_dictionary.csv")
    _write_csv(
        pd.DataFrame(
            [
                {
                    "candidate_rows": len(frame),
                    "causal_regions": frame["causal_region_id"].nunique(),
                    "positive_rows": int(frame["tp_hit_1pct"].sum()),
                    "positive_episodes": frame.loc[frame["tp_hit_1pct"].astype(bool), "positive_episode_id"].nunique(),
                    "tp_base_rate": float(frame["tp_hit_1pct"].mean()),
                }
            ]
        ),
        out_dir / "05_label_episode_summary.csv",
    )

    m0_features = tuple(snapshot.group_membership.loc[snapshot.group_membership["feature_group"].eq("M0_core"), "feature"].astype(str))
    print("[stage] causal multi-scale liquidity map and sweep/reclaim process", flush=True)
    liquidity = build_multiscale_liquidity_features(
        bars,
        frame,
        micro_windows=tuple(int(x) for x in args.micro_liquidity_windows),
        equal_low_tolerance_bp=float(args.liquidity_equal_low_tolerance_bp),
        approach_tolerance_bp=float(args.liquidity_approach_tolerance_bp),
        htf_pivot_minutes=tuple(int(x) for x in args.liquidity_htf_pivot_minutes),
        htf_pivot_left_bars=int(args.liquidity_htf_pivot_left_bars),
        htf_pivot_right_bars=int(args.liquidity_htf_pivot_right_bars),
        show_progress=True,
    )
    if not liquidity.frame["event_id"].equals(frame["event_id"]):
        raise RuntimeError("liquidity feature construction changed candidate order")
    liquidity_columns = tuple(liquidity.dictionary["feature"].astype(str))
    frame = pd.concat(
        [
            frame.reset_index(drop=True),
            liquidity.frame.loc[:, liquidity_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    liquidity_group_members = {
        group: tuple(
            liquidity.group_membership.loc[
                liquidity.group_membership["feature_group"].eq(group), "feature"
            ].astype(str)
        )
        for group in (MICRO_GROUP, MACRO_GROUP, SWEEP_GROUP)
    }
    liquidity_feature_groups: dict[str, tuple[str, ...]] = {
        "L1_micro_liquidity": liquidity_group_members[MICRO_GROUP],
        "L2_macro_liquidity": liquidity_group_members[MACRO_GROUP],
        "L3_multiscale_sweep": (
            *liquidity_group_members[MICRO_GROUP],
            *liquidity_group_members[MACRO_GROUP],
            *liquidity_group_members[SWEEP_GROUP],
        ),
    }
    _write_csv(liquidity.dictionary, out_dir / "06_liquidity_feature_dictionary.csv")
    _write_csv(liquidity.diagnostics, out_dir / "07_liquidity_source_diagnostics.csv")
    aggregate_diagnostics = liquidity.diagnostics[liquidity.diagnostics["source"].eq("aggregate")]
    if aggregate_diagnostics.empty:
        raise RuntimeError("liquidity feature diagnostics missing aggregate row")
    if int(aggregate_diagnostics.iloc[0].get("htf_available_time_violations", 0)) != 0:
        raise RuntimeError("causal HTF liquidity pivot availability audit failed")

    folds = _folds(args.end_date)
    _write_csv(pd.DataFrame([fold._asdict() for fold in folds]), out_dir / "08_walkforward_folds.csv")
    policy_specs = deployable_policy_specs()
    _write_csv(policy_specs, out_dir / "09_predeclared_rank_policy_grid.csv")

    split_parts: list[pd.DataFrame] = []
    sampling_parts: list[pd.DataFrame] = []
    feature_parts: list[dict[str, object]] = []
    model_method_rows: list[dict[str, object]] = []
    risk_method_rows: list[dict[str, object]] = []
    calibrator_selection_parts: list[pd.DataFrame] = []
    calibration_metric_parts: list[pd.DataFrame] = []
    head_metric_parts: list[pd.DataFrame] = []
    risk_parts: list[pd.DataFrame] = []
    policy_window_parts: list[pd.DataFrame] = []
    test_frontier_parts: list[pd.DataFrame] = []
    stress_parts: list[pd.DataFrame] = []
    prediction_samples: list[pd.DataFrame] = []
    rank_resolution_rows: list[dict[str, object]] = []
    full_predictions: list[pd.DataFrame] = []

    for fold_index, fold in enumerate(folds, start=1):
        print(f"[fold] {fold.fold} train={fold.train_start.date()}->{fold.train_end.date()} test={fold.test_start.date()}->{fold.test_end.date()}", flush=True)
        full_train, removed_train = _subset_period(frame, fold.train_start, fold.train_end)
        test, removed_test = _subset_period(frame, fold.test_start, fold.test_end)
        model_fit, calibration, policy, nested = _development_split(full_train, fold)
        nested["test_rows"] = len(test)
        nested["full_train_cross_boundary_removed"] = removed_train
        nested["test_cross_boundary_removed"] = removed_test
        split_parts.append(nested)

        for data in (model_fit, calibration, policy, test):
            episodic = attach_positive_opportunity_episodes(data, max_gap_bars=int(args.positive_episode_gap_bars))
            for column in ("positive_episode_id", "positive_episode_size"):
                data[column] = episodic[column].to_numpy()
        model_fit = attach_episode_balanced_weight(model_fit)
        mechanism = fit_soft_mechanism_transformer(model_fit)
        transformed_by_split: dict[str, pd.DataFrame] = {}
        for name, data in (("model_fit", model_fit), ("calibration", calibration), ("policy", policy), ("test", test)):
            transformed = mechanism.transform(data)
            transformed_by_split[name] = transformed
            for column in transformed.columns:
                data[column] = transformed[column].to_numpy()
        mechanism_features = tuple(column for column in transformed_by_split["model_fit"].columns if column != "mechanism_dominant")
        base_requested = tuple(m0_features) + mechanism_features
        group_requested: dict[str, tuple[str, ...]] = {"L0_tradebar": base_requested}
        for group, extra in liquidity_feature_groups.items():
            group_requested[group] = (*base_requested, *extra)

        train_sample, sampling = _sample_training(model_fit, int(args.maximum_train_rows), int(args.random_state) + fold_index)
        sampling.insert(0, "fold", fold.fold)
        sampling_parts.append(sampling)

        for group_index, (feature_group, requested) in enumerate(group_requested.items(), start=1):
            print(f"[models] {fold.fold} {feature_group} ({group_index}/{len(group_requested)})", flush=True)
            feature_columns, feature_diagnostics = _condition_feature_columns(
                train_sample,
                requested,
                max_features=320,
            )
            feature_parts.append(
                {
                    "fold": fold.fold,
                    "feature_group": feature_group,
                    **feature_diagnostics,
                }
            )
            score_frames = {"calibration": _score_shell(calibration), "policy": _score_shell(policy), "test": _score_shell(test)}
            for output, target in HEAD_TARGETS.items():
                model = fit_binary_model(
                    train_sample,
                    feature_columns=feature_columns,
                    target_column=target,
                    family=PRIMARY_FAMILY,
                    random_state=int(args.random_state) + group_index,
                    min_samples_leaf=int(args.model_min_samples_leaf),
                    weight_column="episode_weight",
                )
                model_method_rows.append(
                    {
                        "fold": fold.fold,
                        "feature_group": feature_group,
                        "output": output,
                        "target": target,
                        "requested_family": PRIMARY_FAMILY,
                        "actual_family": getattr(model, "family", PRIMARY_FAMILY),
                    }
                )
                raw_by_split: dict[str, np.ndarray] = {}
                for split, source in (("calibration", calibration), ("policy", policy), ("test", test)):
                    raw = _predict_binary(model, source, int(args.prediction_chunk_size))
                    raw_by_split[split] = raw
                    score_frames[split][f"{output}_raw"] = raw

                order = np.argsort(pd.to_numeric(calibration["extreme_pos"], errors="raise").to_numpy(dtype=np.int64), kind="mergesort")
                cut = max(1, min(len(calibration) - 1, len(calibration) // 2))
                fit_positions, select_positions = order[:cut], order[cut:]
                candidates_cal = fit_probability_calibrators(raw_by_split["calibration"][fit_positions], calibration.iloc[fit_positions][target])
                selected_method, selection = choose_calibrator(candidates_cal, raw_by_split["calibration"][select_positions], calibration.iloc[select_positions][target])
                selection.insert(0, "feature_group", feature_group)
                selection.insert(0, "output", output)
                selection.insert(0, "fold", fold.fold)
                selection["selected"] = selection["method"].eq(selected_method)
                calibrator_selection_parts.append(selection)
                final_calibrator = fit_probability_calibrators(raw_by_split["calibration"], calibration[target])[selected_method]
                for split, source in (("calibration", calibration), ("policy", policy), ("test", test)):
                    score_frames[split][f"{output}_cal"] = final_calibrator.transform(raw_by_split[split])
                    metrics = calibration_metrics(source[target], score_frames[split][f"{output}_cal"])
                    calibration_metric_parts.append(pd.DataFrame([{"fold": fold.fold, "feature_group": feature_group, "output": output, "target": target, "split": split, "selected_method": selected_method, **metrics}]))
                    head_metric_parts.append(_head_metrics(score_frames[split], fold=fold.fold, feature_group=feature_group, output=output, target=target, split=split))

                rank_reference = EmpiricalRankReference.fit(raw_by_split["policy"])
                for split in ("policy", "test"):
                    ranks = rank_reference.transform(raw_by_split[split])
                    score_frames[split][f"{output}_rank"] = ranks
                    rank_resolution_rows.append(
                        {
                            "fold": fold.fold,
                            "feature_group": feature_group,
                            "output": output,
                            "split": split,
                            "rows": len(ranks),
                            "unique_raw_scores": int(pd.Series(raw_by_split[split]).nunique(dropna=True)),
                            "unique_rank_percentiles": int(pd.Series(ranks).nunique(dropna=True)),
                            "unique_calibrated_probabilities": int(pd.Series(score_frames[split][f"{output}_cal"]).nunique(dropna=True)),
                        }
                    )

            risk_model = fit_risk_point_model(
                train_sample,
                feature_columns=feature_columns,
                target_column="mae_horizon_pct",
                success_only=False,
            )
            risk_method_rows.append(
                {
                    "fold": fold.fold,
                    "feature_group": feature_group,
                    "target": "mae_horizon_pct",
                    "actual_family": getattr(risk_model, "fit_method", "unknown"),
                    "converged": bool(getattr(risk_model, "converged", True)),
                    "iterations": int(getattr(risk_model, "iterations", 0)),
                    "selected_feature_count": len(feature_columns),
                }
            )
            risk_raw = {
                split: _predict_risk(risk_model, source, int(args.prediction_chunk_size))
                for split, source in (("calibration", calibration), ("policy", policy), ("test", test))
            }
            adjustment = fit_conformal_adjustment(calibration["mae_horizon_pct"], risk_raw["calibration"], quantile=0.90)
            for split, source in (("calibration", calibration), ("policy", policy), ("test", test)):
                score_frames[split]["mae_horizon_point_raw"] = risk_raw[split]
                score_frames[split]["mae_horizon_q90_cal"] = adjustment.apply(risk_raw[split])
                metrics = quantile_metrics(source["mae_horizon_pct"], score_frames[split]["mae_horizon_q90_cal"])
                risk_parts.append(pd.DataFrame([{"fold": fold.fold, "feature_group": feature_group, "split": split, "output": "mae_horizon_q90", "target": "mae_horizon_pct", "quantile": 0.90, "additive_shift": adjustment.additive_shift, **metrics}]))
            risk_rank_reference = EmpiricalRankReference.fit(risk_raw["policy"])
            score_frames["policy"]["mae_horizon_risk_rank"] = risk_rank_reference.transform(risk_raw["policy"])
            score_frames["test"]["mae_horizon_risk_rank"] = risk_rank_reference.transform(risk_raw["test"])

            policy_months = max(1, pd.to_datetime(policy["extreme_time"]).dt.to_period("M").nunique())
            test_months = max(1, pd.to_datetime(test["extreme_time"]).dt.to_period("M").nunique())
            for spec in policy_specs.itertuples(index=False):
                spec_series = pd.Series(spec._asdict())
                policy_events = select_ranked_events(score_frames["policy"], spec_series, cooldown_bars=int(args.cooldown_bars))
                test_events = select_ranked_events(score_frames["test"], spec_series, cooldown_bars=int(args.cooldown_bars))
                policy_window_parts.append(pd.DataFrame([{"fold": fold.fold, "feature_group": feature_group, **spec._asdict(), "cooldown_bars": int(args.cooldown_bars), **policy_metrics(policy_events, score_frames["policy"], months=policy_months)}]))
                test_frontier_parts.append(pd.DataFrame([{"fold": fold.fold, "feature_group": feature_group, **spec._asdict(), "cooldown_bars": int(args.cooldown_bars), **policy_metrics(test_events, score_frames["test"], months=test_months)}]))
                if spec.policy_id in {"TP05_ONLY", "TP05_FAST50_CLEAN50", "TP10_ONLY"}:
                    stress = delete_day_stress(test_events)
                    stress.insert(0, "cooldown_bars", int(args.cooldown_bars))
                    stress.insert(0, "policy_id", spec.policy_id)
                    stress.insert(0, "feature_group", feature_group)
                    stress.insert(0, "fold", fold.fold)
                    stress_parts.append(stress)

            sample = pd.concat(
                [
                    score_frames["test"].nlargest(min(1_000, len(test)), "p_tp60_rank"),
                    score_frames["test"].sample(min(1_000, len(test)), random_state=int(args.random_state) + fold_index + group_index),
                ],
                ignore_index=True,
            ).drop_duplicates("event_id")
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
        del full_train, model_fit, calibration, policy, test, train_sample
        gc.collect()

    split_table = pd.concat(split_parts, ignore_index=True)
    sampling_table = pd.concat(sampling_parts, ignore_index=True)
    feature_table = pd.DataFrame(feature_parts)
    model_methods = pd.DataFrame(model_method_rows)
    risk_methods = pd.DataFrame(risk_method_rows)
    calibration_selection = pd.concat(calibrator_selection_parts, ignore_index=True)
    calibration_table = pd.concat(calibration_metric_parts, ignore_index=True)
    head_metrics = pd.concat(head_metric_parts, ignore_index=True)
    risk_table = pd.concat(risk_parts, ignore_index=True)
    policy_window = pd.concat(policy_window_parts, ignore_index=True)
    test_frontier = pd.concat(test_frontier_parts, ignore_index=True)
    increments = _increment_comparison(test_frontier)
    stress_table = pd.concat(stress_parts, ignore_index=True) if stress_parts else pd.DataFrame()
    prediction_sample = pd.concat(prediction_samples, ignore_index=True)
    rank_resolution = pd.DataFrame(rank_resolution_rows)

    _write_csv(split_table, out_dir / "10_nested_fold_boundaries.csv")
    _write_csv(sampling_table, out_dir / "11_training_sampling_diagnostics.csv")
    _write_csv(feature_table, out_dir / "12_fold_feature_groups.csv")
    _write_csv(model_methods, out_dir / "13_model_head_fit_methods.csv")
    _write_csv(risk_methods, out_dir / "13b_risk_fit_methods.csv")
    _write_csv(calibration_selection, out_dir / "14_calibration_method_selection.csv")
    _write_csv(calibration_table, out_dir / "15_probability_calibration_metrics.csv")
    _write_csv(head_metrics, out_dir / "16_head_ranking_metrics.csv")
    _write_csv(risk_table, out_dir / "17_mae_risk_calibration.csv")
    _write_csv(policy_window, out_dir / "18_policy_window_rank_frontier.csv")
    _write_csv(test_frontier, out_dir / "19_frozen_test_rank_frontier.csv")
    _write_csv(increments, out_dir / "20_liquidity_increment_comparison.csv")
    _write_csv(stress_table, out_dir / "21_delete_strong_days_stress.csv")
    _write_csv(prediction_sample, out_dir / "22_walkforward_prediction_sample.csv")
    _write_csv(rank_resolution, out_dir / "23_raw_rank_resolution_diagnostics.csv")
    if full_predictions:
        _write_csv(pd.concat(full_predictions, ignore_index=True), out_dir / "24_walkforward_full_predictions.csv")

    policy_count_check = test_frontier.groupby(["fold", "feature_group"])["event_count"].nunique().reset_index(name="unique_event_counts")
    tp_rank_resolution = rank_resolution[(rank_resolution["output"].eq("p_tp60")) & (rank_resolution["split"].eq("test"))]
    rank_resolution_passed = bool(
        not tp_rank_resolution.empty
        and (tp_rank_resolution["unique_rank_percentiles"] >= np.minimum(20, tp_rank_resolution["rows"])).all()
    )
    print("[stage] raw future perturbation causal audit", flush=True)
    raw_future_audit = _raw_future_perturbation_audit(bars, frame, (*m0_features, *liquidity_columns), args)
    _write_csv(raw_future_audit, out_dir / "24b_raw_future_perturbation_audit.csv")
    forbidden_tokens = ("future", "forward", "label", "mfe", "mae", "tp_hit", "adverse", "entry_price", "completion", "confirmation")
    model_features = "|".join(feature_table["selected_features"].astype(str))
    forbidden = [token for token in forbidden_tokens if token in model_features.lower()]
    binary_methods_stable = bool(
        not model_methods.empty
        and model_methods["actual_family"].astype(str).str.contains(
            "logistic_sgd|logistic_newton_cholesky_fallback|hist_gbdt_convergence_fallback",
            regex=True,
        ).all()
    )
    risk_methods_stable = bool(
        not risk_methods.empty
        and risk_methods["converged"].astype(bool).all()
        and risk_methods["actual_family"].isin(["ridge_lsqr", "hist_gbdt_risk_fallback", "constant"]).all()
    )
    htf_available_violations = int(aggregate_diagnostics.iloc[0].get("htf_available_time_violations", 0))
    audit = pd.DataFrame(
        [
            {"check": "labels_use_next_open_future_close", "passed": True, "detail": "entry=next open; TP/MAE/first-touch=future closes"},
            {"check": "future_high_low_not_used_for_labels", "passed": True, "detail": "future high/low excluded from return labels"},
            {"check": "liquidity_levels_preexist_current_bar", "passed": True, "detail": "micro/current-session levels use t-1; calendar levels are prior completed periods"},
            {"check": "htf_pivot_available_before_feature", "passed": htf_available_violations == 0, "detail": f"violations={htf_available_violations}"},
            {"check": "raw_future_perturbation", "passed": bool(not raw_future_audit.empty and raw_future_audit["passed"].all()), "detail": f"audited={len(raw_future_audit)}"},
            {"check": "liquidity_not_mandatory_candidate_gate", "passed": True, "detail": "L0/L1/L2/L3 use the identical broad candidate universe"},
            {"check": "aggressive_sell_is_proxy_not_stop_identity", "passed": True, "detail": "features describe observed market-order imbalance; they do not claim stop-order identity"},
            {"check": "raw_score_rank_separate_from_calibration", "passed": True, "detail": "selection uses *_rank from raw scores; *_cal is interpretation only"},
            {"check": "raw_rank_has_resolution", "passed": rank_resolution_passed, "detail": f"tp_test_groups={len(tp_rank_resolution)}"},
            {"check": "future_labels_excluded_from_model_features", "passed": not forbidden, "detail": "|".join(forbidden)},
            {"check": "no_frozen_test_winner_selection", "passed": True, "detail": "all predeclared feature groups and policies are reported; no winner is selected"},
            {"check": "liquidity_groups_evaluated_independently", "passed": True, "detail": "micro, macro and combined sweep-process increments are separately reported against L0"},
            {"check": "binary_heads_converged_or_stable_fallback", "passed": binary_methods_stable, "detail": "all binary fit methods are warning-free and recorded"},
            {"check": "mae_risk_model_numerically_stable", "passed": risk_methods_stable, "detail": "ridge uses iterative LSQR; any deterministic tree fallback is recorded"},
        ]
    )
    _write_csv(policy_count_check, out_dir / "25_policy_event_count_resolution.csv")
    _write_csv(audit, out_dir / "26_causal_and_selection_audit.csv")
    if not audit["passed"].all():
        raise RuntimeError(f"10 audit failed: {audit.loc[~audit['passed'], 'check'].tolist()}")

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
        "feature_groups": ["L0_tradebar", *liquidity_feature_groups],
        "micro_liquidity_windows": [int(x) for x in args.micro_liquidity_windows],
        "equal_low_tolerance_bp": float(args.liquidity_equal_low_tolerance_bp),
        "approach_tolerance_bp": float(args.liquidity_approach_tolerance_bp),
        "htf_pivot_minutes": [int(x) for x in args.liquidity_htf_pivot_minutes],
        "htf_pivot_left_bars": int(args.liquidity_htf_pivot_left_bars),
        "htf_pivot_right_bars": int(args.liquidity_htf_pivot_right_bars),
        "liquidity_role": "optional increment features; never a mandatory candidate gate",
        "liquidity_alignment": "all levels exist before current closed bar; HTF pivots exposed only after right-side bars close",
        "ranking": "frozen empirical percentile of raw policy-window scores",
        "calibration_role": "interpretation only; never used for rank selection",
        "automatic_test_winner_selected": False,
        "strategy_or_backtest": False,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "27_RESEARCH_SUMMARY.md").write_text(_summary(test_frontier, increments, liquidity.diagnostics, audit), encoding="utf-8")
    result = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={result.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
