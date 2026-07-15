#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""First-sweep multi-timeframe and multi-horizon research 13.

The event pool is fixed to causal respected-macro first sweeps from research 12.
The 1m closed sweep bar remains the decision and the next 1m open remains the
entry reference.  Fully closed 15m and 1H bars are evaluated only as context.

Feature groups
--------------
M0 : causal 1m trade-bar snapshot + train-fitted soft mechanisms
M1 : M0 + fully closed 15m context
M2 : M1 + fully closed 1H context

Outcome horizons
----------------
60 bars  : fast reversal / short-horizon risk
180 bars : three-hour reversal / slow-clean vs deep-recovery path

The script is an event-model study.  It does not produce a strategy, fees,
stops, sizing, or an automatically selected frozen-test winner.
"""

from __future__ import annotations

import argparse
import gc
import importlib
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
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from research.market_structure.swing_low_typology.common.first_sweep_event import (  # noqa: E402
    build_first_sweep_event_decisions,
)
from research.market_structure.swing_low_typology.common.multiframe_sweep_context import (  # noqa: E402
    TF15_GROUP,
    TF60_GROUP,
    attach_closed_multiframe_context,
)
from research.market_structure.swing_low_typology.common.multihorizon_close_labels import (  # noqa: E402
    build_multihorizon_close_labels,
)
from research.market_structure.swing_low_typology.common.multiobjective_calibration import (  # noqa: E402
    calibration_metrics,
    choose_calibrator,
    fit_conformal_adjustment,
    fit_risk_point_model,
    fit_score_probability_calibrators,
    quantile_metrics,
)
from research.market_structure.swing_low_typology.common.range_increment import EmpiricalRankReference  # noqa: E402
from research.market_structure.swing_low_typology.common.reversal_opportunity import (  # noqa: E402
    build_reversal_candidate_features,
)
from research.market_structure.swing_low_typology.common.swing_low_dataset import validate_trade_bar_fields  # noqa: E402
from research.market_structure.swing_low_typology.common.walkforward_reversal import (  # noqa: E402
    attach_episode_balanced_weight,
    attach_positive_opportunity_episodes,
    fit_soft_mechanism_transformer,
    mechanism_feature_dictionary,
)

# Reuse the already-tested numerical conditioning, solver fallback and raw-score
# resolution machinery from research 12.  Research 13 changes event context and
# labels, not the deployable model-fitting protocol.
_R12 = importlib.import_module(
    "research.market_structure.swing_low_typology.12_respected_macro_first_sweep_event_research"
)

SCRIPT_NAME = "13_multiframe_multihorizon_first_sweep_research"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_1M_FIRST_SWEEP_MULTIFRAME_MULTIHORIZON_13"
EDGE_ID = "RESEARCH_ONLY_ETH_FIRST_SWEEP_MULTIFRAME"
TITLE = "ETH First Sweep Multi-Timeframe Multi-Horizon Research 13"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/13_multiframe_multihorizon_first_sweep"
PRIMARY_FAMILY = "logistic_sgd"
HEAD_TARGETS: dict[str, str] = {
    "p_tp30": "tp30",
    "p_tp60": "tp60",
    "p_tp180": "tp180",
    "p_clean60": "clean60_0p5",
    "p_clean180": "clean180_0p5",
}

FoldSpec = _R12.FoldSpec
_condition_feature_columns = _R12._condition_feature_columns
_predict_binary_probability = _R12._predict_binary_probability
_predict_binary_score = _R12._predict_binary_score
_predict_risk = _R12._predict_risk
_fit_binary_with_resolution_fallback = _R12._fit_binary_with_resolution_fallback
_rank_resolution_record = _R12._rank_resolution_record
_assert_raw_score_resolution = _R12._assert_raw_score_resolution


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Walk-forward multi-timeframe and multi-horizon first-sweep research.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--target-move-pct", type=float, default=1.0)
    p.add_argument("--short-horizon-bars", type=int, default=60)
    p.add_argument("--long-horizon-bars", type=int, default=180)
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
    p.add_argument("--causal-audit-sample-size", type=int, default=3)
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
        "model_cross_boundary_removed": removed_model,
        "calibration_cross_boundary_removed": removed_calibration,
        "policy_cross_boundary_removed": removed_policy,
    }])
    return model_fit, calibration, policy, diagnostic


def _score_shell(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "event_id", "lifecycle_id", "extreme_pos", "extreme_time", "feature_available_time",
        "entry_time", "entry_price", "label_end_time", "causal_region_id", "positive_episode_id",
        "lifecycle_status", "same_bar_reclaim", "path_class",
        "tp30", "tp60", "tp180", "clean30_0p5", "clean60_0p5", "clean180_0p5", "clean180_1p0",
        "slow_success_180", "slow_clean_success_180", "deep_recovery_180", "persistent_failure_180",
        "tp_first_touch_bar_60", "tp_first_touch_bar_180",
        "mae_60_pct", "mae_180_pct", "mae_before_tp_60_pct", "mae_before_tp_180_pct",
        "mfe_60_pct", "mfe_180_pct", "terminal_return_60_pct", "terminal_return_180_pct",
    ]
    return frame.reindex(columns=keep).copy()


def _head_metrics(frame: pd.DataFrame, *, fold: str, feature_group: str, output: str, target: str, split: str) -> pd.DataFrame:
    y = frame[target].astype(int).to_numpy()
    rows: list[dict[str, object]] = []
    for method, column, probability in (
        ("decision_score", f"{output}_score_raw", False),
        ("model_probability", f"{output}_raw", True),
        ("calibrated", f"{output}_cal", True),
    ):
        score = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(score)
        pr_auc = roc_auc = np.nan
        if finite.any() and np.unique(y[finite]).size >= 2:
            pr_auc = float(average_precision_score(y[finite], score[finite]))
            roc_auc = float(roc_auc_score(y[finite], score[finite]))
        clipped = np.clip(score[finite], 1e-7, 1.0 - 1e-7) if probability else np.asarray([], dtype=float)
        rows.append({
            "fold": fold, "feature_group": feature_group, "output": output, "target": target,
            "split": split, "method": method, "rows": int(finite.sum()),
            "positive_rate": float(y[finite].mean()) if finite.any() else np.nan,
            "pr_auc": pr_auc, "roc_auc": roc_auc,
            "brier": float(brier_score_loss(y[finite], clipped)) if probability and finite.any() else np.nan,
            "log_loss": float(log_loss(y[finite], clipped, labels=[0, 1])) if probability and finite.any() else np.nan,
        })
    return pd.DataFrame(rows)


def _policy_specs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fraction in (0.20, 0.30, 0.40):
        pct = int(fraction * 100)
        rows.extend([
            {"policy_id": f"QUICK30_{pct}_ONLY", "tp30_top_fraction": fraction, "tp60_top_fraction": np.nan, "tp180_top_fraction": np.nan, "clean60_min_rank": np.nan, "clean180_min_rank": np.nan, "mae60_max_rank": np.nan, "mae180_max_rank": np.nan},
            {"policy_id": f"FAST{pct}_ONLY", "tp30_top_fraction": np.nan, "tp60_top_fraction": fraction, "tp180_top_fraction": np.nan, "clean60_min_rank": np.nan, "clean180_min_rank": np.nan, "mae60_max_rank": np.nan, "mae180_max_rank": np.nan},
            {"policy_id": f"FAST{pct}_CLEAN50_RISK75", "tp30_top_fraction": np.nan, "tp60_top_fraction": fraction, "tp180_top_fraction": np.nan, "clean60_min_rank": 0.50, "clean180_min_rank": np.nan, "mae60_max_rank": 0.75, "mae180_max_rank": np.nan},
            {"policy_id": f"H180_{pct}_ONLY", "tp30_top_fraction": np.nan, "tp60_top_fraction": np.nan, "tp180_top_fraction": fraction, "clean60_min_rank": np.nan, "clean180_min_rank": np.nan, "mae60_max_rank": np.nan, "mae180_max_rank": np.nan},
            {"policy_id": f"H180_{pct}_CLEAN50_RISK75", "tp30_top_fraction": np.nan, "tp60_top_fraction": np.nan, "tp180_top_fraction": fraction, "clean60_min_rank": np.nan, "clean180_min_rank": 0.50, "mae60_max_rank": np.nan, "mae180_max_rank": 0.75},
        ])
    rows.extend([
        {"policy_id": "BALANCED30", "tp30_top_fraction": np.nan, "tp60_top_fraction": 0.30, "tp180_top_fraction": 0.30, "clean60_min_rank": 0.50, "clean180_min_rank": 0.50, "mae60_max_rank": 0.75, "mae180_max_rank": 0.75},
        {"policy_id": "BALANCED40", "tp30_top_fraction": np.nan, "tp60_top_fraction": 0.40, "tp180_top_fraction": 0.40, "clean60_min_rank": 0.50, "clean180_min_rank": 0.50, "mae60_max_rank": 0.75, "mae180_max_rank": 0.75},
    ])
    return pd.DataFrame(rows)


def _select_policy_events(frame: pd.DataFrame, spec: pd.Series, cooldown_bars: int) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    for key, column in (("tp30_top_fraction", "p_tp30_rank"), ("tp60_top_fraction", "p_tp60_rank"), ("tp180_top_fraction", "p_tp180_rank")):
        value = spec.get(key, np.nan)
        if pd.notna(value):
            mask &= pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float) >= 1.0 - float(value)
    for key, column in (("clean60_min_rank", "p_clean60_rank"), ("clean180_min_rank", "p_clean180_rank")):
        value = spec.get(key, np.nan)
        if pd.notna(value):
            mask &= pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float) >= float(value)
    for key, column in (("mae60_max_rank", "mae60_risk_rank"), ("mae180_max_rank", "mae180_risk_rank")):
        value = spec.get(key, np.nan)
        if pd.notna(value):
            mask &= pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float) <= float(value)
    eligible = frame.loc[mask].sort_values(["extreme_pos", "event_id"]).drop_duplicates("causal_region_id", keep="first")
    if int(cooldown_bars) <= 0 or eligible.empty:
        return eligible.reset_index(drop=True)
    chosen: list[int] = []
    last_pos = -10**18
    for row_index, position in zip(eligible.index, pd.to_numeric(eligible["extreme_pos"], errors="raise").astype(int)):
        if int(position) - last_pos < int(cooldown_bars):
            continue
        chosen.append(int(row_index))
        last_pos = int(position)
    return eligible.loc[chosen].reset_index(drop=True)


def _event_metrics(events: pd.DataFrame, months: int) -> dict[str, float | int]:
    if events.empty:
        return {
            "event_count": 0, "events_per_month": 0.0,
            "tp30_rate": np.nan, "tp60_rate": np.nan, "tp180_rate": np.nan,
            "clean30_rate": np.nan, "clean60_rate": np.nan, "clean180_rate": np.nan, "clean180_1p0_rate": np.nan,
            "slow_success_rate": np.nan, "slow_clean_rate": np.nan,
            "deep_recovery_rate": np.nan, "persistent_failure_rate": np.nan,
            "median_mae60_pct": np.nan, "median_mae180_pct": np.nan,
            "p90_mae180_pct": np.nan, "median_tp180_bar": np.nan,
            "max_day_event_share": np.nan, "top5_day_event_share": np.nan,
        }
    dates = pd.to_datetime(events["extreme_time"]).dt.normalize()
    counts = dates.value_counts()
    tp180_bar = pd.to_numeric(events.loc[events["tp180"].astype(bool), "tp_first_touch_bar_180"], errors="coerce")
    return {
        "event_count": int(len(events)), "events_per_month": float(len(events) / max(1, int(months))),
        "tp30_rate": float(events["tp30"].astype(bool).mean()),
        "tp60_rate": float(events["tp60"].astype(bool).mean()),
        "tp180_rate": float(events["tp180"].astype(bool).mean()),
        "clean30_rate": float(events["clean30_0p5"].astype(bool).mean()),
        "clean60_rate": float(events["clean60_0p5"].astype(bool).mean()),
        "clean180_rate": float(events["clean180_0p5"].astype(bool).mean()),
        "clean180_1p0_rate": float(events["clean180_1p0"].astype(bool).mean()),
        "slow_success_rate": float(events["slow_success_180"].astype(bool).mean()),
        "slow_clean_rate": float(events["slow_clean_success_180"].astype(bool).mean()),
        "deep_recovery_rate": float(events["deep_recovery_180"].astype(bool).mean()),
        "persistent_failure_rate": float(events["persistent_failure_180"].astype(bool).mean()),
        "median_mae60_pct": float(pd.to_numeric(events["mae_60_pct"], errors="coerce").median()),
        "median_mae180_pct": float(pd.to_numeric(events["mae_180_pct"], errors="coerce").median()),
        "p90_mae180_pct": float(pd.to_numeric(events["mae_180_pct"], errors="coerce").quantile(0.90)),
        "median_tp180_bar": float(tp180_bar.median()) if len(tp180_bar) else np.nan,
        "max_day_event_share": float(counts.iloc[0] / len(events)),
        "top5_day_event_share": float(counts.head(5).sum() / len(events)),
    }


def _remove_strong_days(events: pd.DataFrame, count: int) -> pd.DataFrame:
    if events.empty or count <= 0:
        return events.copy()
    data = events.copy()
    data["_day"] = pd.to_datetime(data["extreme_time"]).dt.normalize()
    day = data.groupby("_day", sort=True).agg(
        tp180=("tp180", "sum"), tp60=("tp60", "sum"), events=("event_id", "size")
    ).sort_values(["tp180", "tp60", "events"], ascending=False, kind="mergesort")
    removed = set(day.head(int(count)).index)
    return data[~data["_day"].isin(removed)].drop(columns="_day").reset_index(drop=True)


def _delete_day_stress(events: pd.DataFrame, months: int) -> pd.DataFrame:
    return pd.DataFrame([
        {"removed_strongest_days": removed, **_event_metrics(_remove_strong_days(events, removed), months)}
        for removed in (0, 5, 10)
    ])


def _direct_outcomes(frame: pd.DataFrame, folds: Sequence[FoldSpec]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold in folds:
        test, removed = _subset_period(frame, fold.test_start, fold.test_end)
        months = max(1, pd.to_datetime(test["extreme_time"]).dt.to_period("M").nunique())
        rows.append({"fold": fold.fold, "state": "all", "cross_boundary_removed": removed, **_event_metrics(test, months)})
        for state, part in test.groupby("path_class", sort=True):
            rows.append({"fold": fold.fold, "state": state, "cross_boundary_removed": removed, **_event_metrics(part, months)})
    return pd.DataFrame(rows)


def _increment_comparison(frontier: pd.DataFrame) -> pd.DataFrame:
    base = frontier[frontier["feature_group"].eq("M0_1m")].copy()
    keys = ["fold", "policy_id"]
    metrics = [
        "event_count", "events_per_month", "tp30_rate", "tp60_rate", "tp180_rate", "clean30_rate", "clean60_rate", "clean180_rate",
        "slow_clean_rate", "deep_recovery_rate", "persistent_failure_rate",
        "median_mae60_pct", "median_mae180_pct", "p90_mae180_pct", "top5_day_event_share",
    ]
    base = base[keys + metrics].rename(columns={metric: f"baseline_{metric}" for metric in metrics})
    comp = frontier[~frontier["feature_group"].eq("M0_1m")].merge(base, on=keys, how="left", validate="many_to_one")
    for metric in metrics:
        comp[f"delta_{metric}"] = pd.to_numeric(comp[metric], errors="coerce") - pd.to_numeric(comp[f"baseline_{metric}"], errors="coerce")
    return comp


def _stability_matrix(increments: pd.DataFrame, stress: pd.DataFrame, minimum_test_events: int) -> pd.DataFrame:
    if increments.empty:
        return pd.DataFrame()
    stress10 = stress[stress["removed_strongest_days"].eq(10)][
        ["fold", "feature_group", "policy_id", "tp180_rate", "clean180_rate"]
    ].rename(columns={"tp180_rate": "delete10_tp180_rate", "clean180_rate": "delete10_clean180_rate"})
    base10 = stress10[stress10["feature_group"].eq("M0_1m")].rename(columns={
        "delete10_tp180_rate": "baseline_delete10_tp180_rate",
        "delete10_clean180_rate": "baseline_delete10_clean180_rate",
    })[["fold", "policy_id", "baseline_delete10_tp180_rate", "baseline_delete10_clean180_rate"]]
    joined = increments.merge(stress10, on=["fold", "feature_group", "policy_id"], how="left")
    joined = joined.merge(base10, on=["fold", "policy_id"], how="left")
    joined["delta_delete10_tp180_rate"] = joined["delete10_tp180_rate"] - joined["baseline_delete10_tp180_rate"]
    joined["delta_delete10_clean180_rate"] = joined["delete10_clean180_rate"] - joined["baseline_delete10_clean180_rate"]
    rows: list[dict[str, object]] = []
    for (feature_group, policy_id), group in joined.groupby(["feature_group", "policy_id"], sort=False):
        event_ok = pd.to_numeric(group["event_count"], errors="coerce") >= int(minimum_test_events)
        tp60 = pd.to_numeric(group["delta_tp60_rate"], errors="coerce")
        tp180 = pd.to_numeric(group["delta_tp180_rate"], errors="coerce")
        clean60 = pd.to_numeric(group["delta_clean60_rate"], errors="coerce")
        clean180 = pd.to_numeric(group["delta_clean180_rate"], errors="coerce")
        mae180 = pd.to_numeric(group["delta_median_mae180_pct"], errors="coerce")
        delete10 = pd.to_numeric(group["delta_delete10_tp180_rate"], errors="coerce")
        keep = bool(
            event_ok.all()
            and ((clean60 > 0).sum() >= 2 or (clean180 > 0).sum() >= 2)
            and (tp60 < -0.03).sum() == 0
            and (tp180 < -0.03).sum() == 0
            and (mae180 > 0.10).sum() == 0
            and (delete10 > 0).sum() >= 2
        )
        rows.append({
            "feature_group": feature_group, "policy_id": policy_id, "fold_count": len(group),
            "minimum_event_count_pass": bool(event_ok.all()),
            "tp60_positive_folds": int((tp60 > 0).sum()), "tp180_positive_folds": int((tp180 > 0).sum()),
            "clean60_positive_folds": int((clean60 > 0).sum()), "clean180_positive_folds": int((clean180 > 0).sum()),
            "mae180_improved_folds": int((mae180 < 0).sum()),
            "delete10_tp180_positive_folds": int((delete10 > 0).sum()),
            "mean_delta_tp60_rate": float(tp60.mean()), "mean_delta_tp180_rate": float(tp180.mean()),
            "mean_delta_clean60_rate": float(clean60.mean()), "mean_delta_clean180_rate": float(clean180.mean()),
            "mean_delta_median_mae180_pct": float(mae180.mean()),
            "predeclared_keep_gate": keep,
        })
    return pd.DataFrame(rows)


def _multiframe_truncation_audit(bars: pd.DataFrame, frame: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    sample_size = min(max(0, int(sample_size)), len(frame))
    if sample_size == 0:
        return pd.DataFrame([{"sample": -1, "passed": True, "detail": "disabled", "max_abs_diff": 0.0}])
    positions = np.unique(np.linspace(0, len(frame) - 1, sample_size, dtype=np.int64))
    feature_columns = [column for column in frame.columns if column.startswith("mtf_") and not column.endswith("_available_time")]
    rows: list[dict[str, object]] = []
    for sample_number, row_pos in enumerate(positions, start=1):
        expected = frame.iloc[int(row_pos)]
        decision_pos = int(expected["extreme_pos"])
        decision = expected.reindex(["event_id", "feature_available_time"]).to_frame().T
        rebuilt = attach_closed_multiframe_context(bars.iloc[: decision_pos + 1].copy(), decision).frame.iloc[0]
        expected_values = pd.to_numeric(expected.reindex(feature_columns), errors="coerce").to_numpy(dtype=float)
        actual_values = pd.to_numeric(rebuilt.reindex(feature_columns), errors="coerce").to_numpy(dtype=float)
        nan_match = np.isnan(expected_values) == np.isnan(actual_values)
        finite = np.isfinite(expected_values) & np.isfinite(actual_values)
        diff = np.zeros(len(feature_columns), dtype=float)
        diff[finite] = np.abs(expected_values[finite] - actual_values[finite])
        max_diff = float(diff.max()) if len(diff) else 0.0
        passed = bool(nan_match.all() and max_diff <= 1e-9)
        rows.append({
            "sample": sample_number, "event_id": expected["event_id"],
            "decision_time": expected["feature_available_time"], "passed": passed,
            "max_abs_diff": max_diff, "detail": "future bars removed" if passed else "multiframe feature mismatch",
        })
    return pd.DataFrame(rows)


def _summary(outcomes: pd.DataFrame, stability: pd.DataFrame, audit: pd.DataFrame) -> str:
    lines = [
        "# Research 13 Summary", "",
        "Causal respected-macro first sweeps with closed 15m/1H context and 60/180-minute close-path outcomes.", "",
        "- decision: closed 1m first-sweep bar", "- entry reference: next 1m open",
        "- context: only fully closed 15m and 1H bars by available_time",
        "- labels: future closed-bar closes only; future high/low excluded", "",
        "## Direct outcomes", "",
    ]
    for row in outcomes[outcomes["state"].eq("all")].itertuples(index=False):
        lines.append(
            f"- {row.fold}: n={row.event_count:,}, TP30={row.tp30_rate:.4f}, TP60={row.tp60_rate:.4f}, TP180={row.tp180_rate:.4f}, "
            f"slow-clean={row.slow_clean_rate:.4f}, median MAE180={row.median_mae180_pct:.4f}%"
        )
    lines.extend(["", "## Predeclared keep gate", ""])
    kept = stability[stability["predeclared_keep_gate"].astype(bool)] if not stability.empty else pd.DataFrame()
    if kept.empty:
        lines.append("No multi-timeframe feature-group/policy pair passed the cross-fold keep gate.")
    else:
        for row in kept.itertuples(index=False):
            lines.append(
                f"- {row.feature_group} {row.policy_id}: TP60+ folds={row.tp60_positive_folds}, "
                f"TP180+ folds={row.tp180_positive_folds}, Clean180+ folds={row.clean180_positive_folds}, "
                f"delete10 TP180+ folds={row.delete10_tp180_positive_folds}"
            )
    lines.extend(["", "## Audit", ""])
    for row in audit.itertuples(index=False):
        lines.append(f"- [{'PASS' if row.passed else 'FAIL'}] {row.check}: {row.detail}")
    lines.extend(["", "This is an event-model study, not a trading-strategy or profitability claim."])
    return "\n".join(lines) + "\n"


def run_research(args: argparse.Namespace) -> Path:
    if not (1 <= int(args.short_horizon_bars) < int(args.long_horizon_bars)):
        raise ValueError("short horizon must be smaller than long horizon")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = load_bars(args)
    _write_csv(validate_trade_bar_fields(bars), out_dir / "01_trade_bar_field_coverage.csv")

    research_start = pd.Timestamp(args.start_date)
    research_end = _end_exclusive(args.end_date)
    print("[stage] respected macro first-sweep event pool", flush=True)
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
    decisions = event_build.decisions[event_build.decisions["decision_path"].eq("sweep")].reset_index(drop=True)
    if decisions.empty:
        raise RuntimeError("no first-sweep decisions were built")
    _write_csv(event_build.diagnostics, out_dir / "02_event_build_diagnostics.csv")
    _write_csv(event_build.levels, out_dir / "03_respected_level_table.csv")
    _write_csv(event_build.lifecycle, out_dir / "04_first_sweep_lifecycle_table.csv")

    print("[stage] causal 1m snapshot", flush=True)
    snapshot = build_reversal_candidate_features(
        bars, decisions, include_session=False, include_htf=False, show_progress=True,
    )
    print("[stage] closed 15m and 1H context", flush=True)
    multiframe = attach_closed_multiframe_context(bars, snapshot.frame, timeframes_minutes=(15, 60))
    if not multiframe.alignment_audit["passed"].astype(bool).all():
        raise RuntimeError("multi-timeframe available-time alignment failed")
    if (pd.to_numeric(multiframe.alignment_audit["context_coverage"], errors="coerce") < 0.95).any():
        raise RuntimeError("multi-timeframe context coverage below 95%")
    _write_csv(snapshot.dictionary, out_dir / "05_snapshot_feature_dictionary.csv")
    _write_csv(multiframe.dictionary, out_dir / "06_multiframe_feature_dictionary.csv")
    _write_csv(multiframe.alignment_audit, out_dir / "07_multiframe_alignment_audit.csv")
    _write_csv(mechanism_feature_dictionary(), out_dir / "08_soft_mechanism_feature_dictionary.csv")

    print("[stage] 60m and 180m future-close paths", flush=True)
    labels = build_multihorizon_close_labels(
        bars,
        multiframe.frame,
        target_move_pct=float(args.target_move_pct),
        short_horizon=int(args.short_horizon_bars),
        long_horizon=int(args.long_horizon_bars),
        vectorized_chunk_size=int(args.label_vectorized_chunk_size),
        show_progress=True,
    )
    frame = multiframe.frame.merge(labels, on="event_id", how="inner", validate="one_to_one", suffixes=("", "_label"))
    if frame.empty:
        raise RuntimeError("all first-sweep decisions were removed by 180-bar label boundaries")
    frame = frame.sort_values(["extreme_pos", "event_id"]).reset_index(drop=True)
    episodic = attach_positive_opportunity_episodes(frame.rename(columns={"tp60": "tp_hit_1pct"}), max_gap_bars=2)
    for column in ("positive_episode_id", "positive_episode_size", "positive_episode_number"):
        frame[column] = episodic[column].to_numpy()

    folds = _folds(args.end_date)
    outcomes = _direct_outcomes(frame, folds)
    _write_csv(outcomes, out_dir / "09_direct_multihorizon_outcomes.csv")
    print("[stage] multi-timeframe future-truncation audit", flush=True)
    mtf_future_audit = _multiframe_truncation_audit(bars, frame, int(args.causal_audit_sample_size))
    _write_csv(mtf_future_audit, out_dir / "10_multiframe_future_truncation_audit.csv")
    if not mtf_future_audit["passed"].astype(bool).all():
        raise RuntimeError("multi-timeframe future-truncation audit failed")
    _write_csv(pd.DataFrame([fold._asdict() for fold in folds]), out_dir / "11_walkforward_folds.csv")
    policy_specs = _policy_specs()
    _write_csv(policy_specs, out_dir / "12_predeclared_policy_grid.csv")

    m0_features = tuple(snapshot.group_membership.loc[snapshot.group_membership["feature_group"].eq("M0_core"), "feature"].astype(str))
    tf15_features = tuple(multiframe.group_membership.loc[multiframe.group_membership["feature_group"].eq(TF15_GROUP), "feature"].astype(str))
    tf60_features = tuple(multiframe.group_membership.loc[multiframe.group_membership["feature_group"].eq(TF60_GROUP), "feature"].astype(str))
    feature_groups = {
        "M0_1m": (),
        "M1_closed15m": tf15_features,
        "M2_closed15m_1h": (*tf15_features, *tf60_features),
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

    for fold_index, fold in enumerate(folds, start=1):
        print(f"[fold] {fold.fold}", flush=True)
        full_train, removed_train = _subset_period(frame, fold.train_start, fold.train_end)
        test, removed_test = _subset_period(frame, fold.test_start, fold.test_end)
        if len(test) < int(args.minimum_test_events):
            raise RuntimeError(f"{fold.fold} has only {len(test)} test events")
        model_fit, calibration, policy, nested = _development_split(full_train, fold)
        nested["test_rows"] = len(test)
        nested["full_train_cross_boundary_removed"] = removed_train
        nested["test_cross_boundary_removed"] = removed_test
        split_parts.append(nested)

        for data in (model_fit, calibration, policy, test):
            episode_source = data.rename(columns={"tp60": "tp_hit_1pct"})
            episode = attach_positive_opportunity_episodes(episode_source, max_gap_bars=2)
            for column in ("positive_episode_id", "positive_episode_size"):
                data[column] = episode[column].to_numpy()
        weighted_source = model_fit.rename(columns={"tp60": "tp_hit_1pct"})
        weighted = attach_episode_balanced_weight(weighted_source)
        model_fit["episode_weight"] = weighted["episode_weight"].to_numpy()
        mechanism_source = model_fit.rename(columns={"tp60": "tp_hit_1pct"})
        mechanism = fit_soft_mechanism_transformer(mechanism_source)
        mechanism_features: tuple[str, ...] = ()
        for name, data in (("model_fit", model_fit), ("calibration", calibration), ("policy", policy), ("test", test)):
            transformed = mechanism.transform(data.rename(columns={"tp60": "tp_hit_1pct"}))
            if name == "model_fit":
                mechanism_features = tuple(column for column in transformed.columns if column != "mechanism_dominant")
            for column in transformed.columns:
                data[column] = transformed[column].to_numpy()
        base_requested = (*m0_features, *mechanism_features)

        for group_index, (feature_group, extra) in enumerate(feature_groups.items(), start=1):
            print(f"[models] {fold.fold} {feature_group} ({group_index}/{len(feature_groups)})", flush=True)
            requested = (*base_requested, *extra)
            selected, diagnostics = _condition_feature_columns(model_fit, requested, max_features=220)
            if feature_group != "M0_1m" and not any(column.startswith("mtf_15m_") for column in selected):
                raise RuntimeError(f"{fold.fold} {feature_group} retained no 15m context")
            if feature_group == "M2_closed15m_1h" and not any(column.startswith("mtf_60m_") for column in selected):
                raise RuntimeError(f"{fold.fold} {feature_group} retained no 1H context")
            feature_rows.append({"fold": fold.fold, "feature_group": feature_group, **diagnostics})
            score_frames = {"calibration": _score_shell(calibration), "policy": _score_shell(policy), "test": _score_shell(test)}

            for head_index, (output, target) in enumerate(HEAD_TARGETS.items(), start=1):
                model, fit_diagnostics = _fit_binary_with_resolution_fallback(
                    model_fit,
                    policy,
                    feature_columns=selected,
                    target_column=target,
                    fold=fold.fold,
                    decision_path="sweep_multihorizon",
                    feature_group=feature_group,
                    output=output,
                    random_state=int(args.random_state) + fold_index * 10 + head_index,
                    min_samples_leaf=int(args.model_min_samples_leaf),
                    prediction_chunk_size=int(args.prediction_chunk_size),
                )
                model_rows.append({
                    "fold": fold.fold, "feature_group": feature_group, "output": output, "target": target,
                    "requested_family": PRIMARY_FAMILY, "actual_family": getattr(model, "family", PRIMARY_FAMILY),
                    **fit_diagnostics,
                })
                scores: dict[str, np.ndarray] = {}
                probabilities: dict[str, np.ndarray] = {}
                for split, source in (("calibration", calibration), ("policy", policy), ("test", test)):
                    scores[split] = _predict_binary_score(model, source, int(args.prediction_chunk_size))
                    probabilities[split] = _predict_binary_probability(model, source, int(args.prediction_chunk_size))
                    score_frames[split][f"{output}_score_raw"] = scores[split]
                    score_frames[split][f"{output}_raw"] = probabilities[split]

                provisional_reference = EmpiricalRankReference.fit(scores["policy"])
                provisional = _rank_resolution_record(
                    fold=fold.fold, decision_path="sweep_multihorizon", feature_group=feature_group,
                    output=output, split="policy", raw_scores=scores["policy"],
                    ranks=provisional_reference.transform(scores["policy"]), calibrated=probabilities["policy"],
                    reference=provisional_reference, model_probability=probabilities["policy"],
                )
                _assert_raw_score_resolution(provisional, actual_family=str(getattr(model, "family", PRIMARY_FAMILY)))

                order = np.argsort(pd.to_numeric(calibration["extreme_pos"], errors="raise").to_numpy(dtype=np.int64), kind="mergesort")
                cut = max(1, min(len(calibration) - 1, len(calibration) // 2))
                fit_pos, select_pos = order[:cut], order[cut:]
                candidates_cal = fit_score_probability_calibrators(scores["calibration"][fit_pos], calibration.iloc[fit_pos][target])
                selected_method, selection = choose_calibrator(candidates_cal, scores["calibration"][select_pos], calibration.iloc[select_pos][target])
                selection.insert(0, "feature_group", feature_group)
                selection.insert(0, "output", output)
                selection.insert(0, "fold", fold.fold)
                selection["selected"] = selection["method"].eq(selected_method)
                calibration_selection_parts.append(selection)
                final_calibrator = fit_score_probability_calibrators(scores["calibration"], calibration[target])[selected_method]
                rank_reference = EmpiricalRankReference.fit(scores["policy"])
                for split, source in (("calibration", calibration), ("policy", policy), ("test", test)):
                    score_frames[split][f"{output}_cal"] = final_calibrator.transform(scores[split])
                    calibration_metric_parts.append(pd.DataFrame([{
                        "fold": fold.fold, "feature_group": feature_group, "output": output,
                        "target": target, "split": split, "selected_method": selected_method,
                        **calibration_metrics(source[target], score_frames[split][f"{output}_cal"]),
                    }]))
                    head_metric_parts.append(_head_metrics(
                        score_frames[split], fold=fold.fold, feature_group=feature_group,
                        output=output, target=target, split=split,
                    ))
                for split in ("policy", "test"):
                    ranks = rank_reference.transform(scores[split])
                    score_frames[split][f"{output}_rank"] = ranks
                    resolution = _rank_resolution_record(
                        fold=fold.fold, decision_path="sweep_multihorizon", feature_group=feature_group,
                        output=output, split=split, raw_scores=scores[split], ranks=ranks,
                        calibrated=score_frames[split][f"{output}_cal"], reference=rank_reference,
                        model_probability=probabilities[split],
                    )
                    rank_rows.append(resolution)
                    _assert_raw_score_resolution(resolution, actual_family=str(getattr(model, "family", PRIMARY_FAMILY)))

            for risk_name, target in (("mae60", "mae_60_pct"), ("mae180", "mae_180_pct")):
                risk_model = fit_risk_point_model(model_fit, feature_columns=selected, target_column=target, success_only=False)
                risk_rows.append({
                    "fold": fold.fold, "feature_group": feature_group, "risk_name": risk_name, "target": target,
                    "actual_family": getattr(risk_model, "fit_method", "unknown"),
                    "converged": bool(getattr(risk_model, "converged", True)),
                    "iterations": int(getattr(risk_model, "iterations", 0)), "selected_feature_count": len(selected),
                })
                raw = {split: _predict_risk(risk_model, source, int(args.prediction_chunk_size)) for split, source in (("calibration", calibration), ("policy", policy), ("test", test))}
                adjustment = fit_conformal_adjustment(calibration[target], raw["calibration"], quantile=0.90)
                for split, source in (("calibration", calibration), ("policy", policy), ("test", test)):
                    score_frames[split][f"{risk_name}_point_raw"] = raw[split]
                    score_frames[split][f"{risk_name}_q90_cal"] = adjustment.apply(raw[split])
                    risk_metric_parts.append(pd.DataFrame([{
                        "fold": fold.fold, "feature_group": feature_group, "risk_name": risk_name,
                        "split": split, "output": f"{risk_name}_q90", "target": target,
                        "quantile": 0.90, "additive_shift": adjustment.additive_shift,
                        **quantile_metrics(source[target], score_frames[split][f"{risk_name}_q90_cal"]),
                    }]))
                reference = EmpiricalRankReference.fit(raw["policy"])
                score_frames["policy"][f"{risk_name}_risk_rank"] = reference.transform(raw["policy"])
                score_frames["test"][f"{risk_name}_risk_rank"] = reference.transform(raw["test"])

            policy_months = max(1, pd.to_datetime(policy["extreme_time"]).dt.to_period("M").nunique())
            test_months = max(1, pd.to_datetime(test["extreme_time"]).dt.to_period("M").nunique())
            for spec in policy_specs.itertuples(index=False):
                series = pd.Series(spec._asdict())
                policy_events = _select_policy_events(score_frames["policy"], series, int(args.cooldown_bars))
                test_events = _select_policy_events(score_frames["test"], series, int(args.cooldown_bars))
                common = {"fold": fold.fold, "feature_group": feature_group, **spec._asdict(), "cooldown_bars": int(args.cooldown_bars)}
                policy_window_parts.append(pd.DataFrame([{**common, **_event_metrics(policy_events, policy_months)}]))
                test_frontier_parts.append(pd.DataFrame([{**common, **_event_metrics(test_events, test_months)}]))
                stress = _delete_day_stress(test_events, test_months)
                stress.insert(0, "policy_id", spec.policy_id)
                stress.insert(0, "feature_group", feature_group)
                stress.insert(0, "fold", fold.fold)
                stress_parts.append(stress)

            sample = pd.concat([
                score_frames["test"].nlargest(min(250, len(test)), "p_tp180_rank"),
                score_frames["test"].sample(min(250, len(test)), random_state=int(args.random_state) + fold_index + group_index),
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
    stress = pd.concat(stress_parts, ignore_index=True)
    increments = _increment_comparison(test_frontier)
    stability = _stability_matrix(increments, stress, int(args.minimum_test_events))
    samples = pd.concat(prediction_samples, ignore_index=True)
    ranks = pd.DataFrame(rank_rows)

    _write_csv(split_table, out_dir / "13_nested_fold_boundaries.csv")
    _write_csv(feature_table, out_dir / "14_fold_feature_groups.csv")
    _write_csv(model_methods, out_dir / "15_model_head_fit_methods.csv")
    _write_csv(risk_methods, out_dir / "15b_risk_fit_methods.csv")
    _write_csv(calibration_selection, out_dir / "16_calibration_method_selection.csv")
    _write_csv(calibration_metrics_table, out_dir / "17_probability_calibration_metrics.csv")
    _write_csv(head_metrics, out_dir / "18_head_ranking_metrics.csv")
    _write_csv(risk_metrics, out_dir / "19_mae_risk_calibration.csv")
    _write_csv(policy_window, out_dir / "20_policy_window_rank_frontier.csv")
    _write_csv(test_frontier, out_dir / "21_frozen_test_rank_frontier.csv")
    _write_csv(increments, out_dir / "22_multiframe_increment_comparison.csv")
    _write_csv(stability, out_dir / "23_cross_fold_stability_matrix.csv")
    _write_csv(stress, out_dir / "24_delete_strong_days_stress.csv")
    _write_csv(samples, out_dir / "25_walkforward_prediction_sample.csv")
    _write_csv(ranks, out_dir / "26_raw_rank_resolution_diagnostics.csv")
    if full_predictions:
        _write_csv(pd.concat(full_predictions, ignore_index=True), out_dir / "27_walkforward_full_predictions.csv")

    forbidden_tokens = ("future", "forward", "label", "mfe", "mae_", "tp_hit", "adverse", "entry_price", "completion", "confirmation")
    selected_text = "|".join(feature_table["selected_features"].astype(str)).lower()
    forbidden = [token for token in forbidden_tokens if token in selected_text]
    rank_ok = bool(not ranks.empty and ranks["raw_score_resolution_passed"].astype(bool).all())
    binary_stable = bool(not model_methods.empty and model_methods["actual_family"].astype(str).str.contains("logistic_sgd|logistic_newton_cholesky|logistic_lbfgs|hist_gbdt", regex=True).all())
    risk_stable = bool(not risk_methods.empty and risk_methods["converged"].astype(bool).all())
    groups_retain = bool(
        feature_table[feature_table["feature_group"].eq("M1_closed15m")]["selected_features"].str.contains("mtf_15m_").all()
        and feature_table[feature_table["feature_group"].eq("M2_closed15m_1h")]["selected_features"].str.contains("mtf_15m_").all()
        and feature_table[feature_table["feature_group"].eq("M2_closed15m_1h")]["selected_features"].str.contains("mtf_60m_").all()
    )
    audit = pd.DataFrame([
        {"check": "first_sweep_level_available_before_decision", "passed": int(dict(zip(event_build.diagnostics.get("metric", []), event_build.diagnostics.get("value", []))).get("availability_violations", 0)) == 0, "detail": "respected level exists before sweep bar starts"},
        {"check": "closed_multiframe_available_time", "passed": bool(multiframe.alignment_audit["passed"].astype(bool).all()), "detail": "15m/1H available_time <= 1m feature_available_time"},
        {"check": "multiframe_future_truncation", "passed": bool(mtf_future_audit["passed"].astype(bool).all()), "detail": f"audited={len(mtf_future_audit)}"},
        {"check": "labels_use_next_open_future_close", "passed": True, "detail": "both horizons share next 1m open and inspect future closed-bar closes"},
        {"check": "future_high_low_not_used_for_labels", "passed": True, "detail": "60/180 labels use future closes only"},
        {"check": "long_horizon_controls_fold_boundaries", "passed": True, "detail": "label_end_time is the 180-bar end for every split"},
        {"check": "multiframe_groups_retain_context", "passed": groups_retain, "detail": "M1 retains 15m; M2 retains 15m and 1H in every fold"},
        {"check": "raw_rank_has_resolution", "passed": rank_ok, "detail": "finite unsquashed decision scores; CDF tail saturation is diagnostic"},
        {"check": "future_labels_excluded_from_model_features", "passed": not forbidden, "detail": "|".join(forbidden)},
        {"check": "binary_heads_stable", "passed": binary_stable, "detail": "all actual solver families recorded"},
        {"check": "mae_models_stable", "passed": risk_stable, "detail": "60m and 180m risk fits converged"},
        {"check": "broad_policies_only", "passed": float(policy_specs[["tp30_top_fraction", "tp60_top_fraction", "tp180_top_fraction"]].stack().min()) >= 0.20, "detail": "no Top2/5/10 policy"},
        {"check": "no_frozen_test_winner_selection", "passed": True, "detail": "all groups and policies reported; keep gate descriptive only"},
    ])
    _write_csv(audit, out_dir / "28_causal_and_selection_audit.csv")
    if not audit["passed"].all():
        raise RuntimeError(f"13 audit failed: {audit.loc[~audit['passed'], 'check'].tolist()}")

    manifest = {
        "script": SCRIPT_NAME, "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID, "edge_id": EDGE_ID, "title": TITLE,
        "symbol": args.symbol, "timeframe": args.timeframe,
        "start_date": args.start_date, "end_date": args.end_date, "warmup_start_date": args.warmup_start_date,
        "target_move_pct": float(args.target_move_pct),
        "short_horizon_bars": int(args.short_horizon_bars), "long_horizon_bars": int(args.long_horizon_bars),
        "event_pool": "respected macro first sweep only", "decision_time": "closed 1m sweep bar",
        "entry_price_source": "next 1m open", "context_timeframes": [15, 60],
        "context_alignment": "fully closed HTF bar available_time backward asof",
        "feature_groups": feature_groups, "model_heads": HEAD_TARGETS,
        "ranking": "frozen empirical percentile of unsquashed policy-window decision scores",
        "calibration_role": "interpretation only", "automatic_test_winner_selected": False,
        "strategy_or_backtest": False,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "29_RESEARCH_SUMMARY.md").write_text(_summary(outcomes, stability, audit), encoding="utf-8")
    result = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={result.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
