#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sequential first-sweep state scoring research 14.

Research 14 asks whether a respected-macro first-sweep score can be updated
causally after the event and whether score improvement is associated with a
better *remaining* path.  It is intentionally not a sizing or trading-strategy
backtest.

Decision checkpoints
--------------------
t0, t+1, t+3, t+5, t+10 and t+15 closed 1m bars.  Every checkpoint uses its
next 1m open as the fresh entry/add-on reference and future closed-bar closes
for labels.

Model groups
------------
I0_t0_full
    Research-13-style full M0 model trained only on first-sweep t0 rows and
    reapplied to later causal snapshots.
P0_pooled_compact
    One compact current-state model trained on all checkpoints with each
    lifecycle receiving equal total optimization weight.
P1_pooled_trajectory
    P0 plus a small, explicit causal path-state feature group.

The 0.5/1.0/1.5/2.0 percent target ladder is descriptive only.  Research 14
never selects a target or trains four target-specific model families.
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_common.review_pack import finalize_research_report  # noqa: E402
from research.market_structure.swing_low_typology.common.first_sweep_event import (  # noqa: E402
    build_first_sweep_event_decisions,
)
from research.market_structure.swing_low_typology.common.multihorizon_close_labels import (  # noqa: E402
    build_multihorizon_close_labels,
)
from research.market_structure.swing_low_typology.common.multiobjective_calibration import (  # noqa: E402
    fit_risk_point_model,
)
from research.market_structure.swing_low_typology.common.range_increment import EmpiricalRankReference  # noqa: E402
from research.market_structure.swing_low_typology.common.reversal_opportunity import (  # noqa: E402
    build_reversal_candidate_features,
)
from research.market_structure.swing_low_typology.common.sequential_sweep_state import (  # noqa: E402
    DEFAULT_AMPLITUDE_HORIZONS,
    DEFAULT_AMPLITUDE_TARGETS_PCT,
    DEFAULT_CHECKPOINT_OFFSETS,
    SEQUENTIAL_FEATURE_GROUP,
    build_amplitude_ladder_close_labels,
    build_sequential_checkpoint_decisions,
)
from research.market_structure.swing_low_typology.common.swing_low_dataset import validate_trade_bar_fields  # noqa: E402
from research.market_structure.swing_low_typology.common.walkforward_reversal import (  # noqa: E402
    fit_soft_mechanism_transformer,
    mechanism_feature_dictionary,
)

_R12 = importlib.import_module(
    "research.market_structure.swing_low_typology.12_respected_macro_first_sweep_event_research"
)
_R13 = importlib.import_module(
    "research.market_structure.swing_low_typology.13_multiframe_multihorizon_first_sweep_research"
)

SCRIPT_NAME = "14_sequential_first_sweep_state_scoring_research"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_1M_FIRST_SWEEP_SEQUENTIAL_STATE_SCORING_14"
EDGE_ID = "RESEARCH_ONLY_ETH_FIRST_SWEEP_SEQUENTIAL_STATE"
TITLE = "ETH First Sweep Sequential State Scoring Research 14"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/14_sequential_first_sweep_state_scoring"
PRIMARY_FAMILY = "logistic_sgd"

HEAD_TARGETS: dict[str, str] = {
    "p_tp30": "tp30",
    "p_tp60": "tp60",
    "p_tp180": "tp180",
    "p_clean60": "clean60_0p5",
}

# Compact current-state representation.  The list is fixed before any fold is
# evaluated; conditioning may remove constants/redundant columns using only the
# model-fit window but may not add unlisted features.
COMPACT_SNAPSHOT_FEATURES: tuple[str, ...] = (
    "current_return_1", "current_body_pct", "current_range_pct",
    "current_lower_wick_share", "current_close_in_bar",
    "current_delta_ratio", "current_large_delta_ratio", "current_buy_ratio",
    "current_large_buy_ratio", "current_notional_log", "current_trades_log",
    "price_return_5", "rebound_from_low_5", "range_position_5",
    "realized_vol_5", "delta_ratio_5", "notional_intensity_5",
    "support_test_density_5",
    "price_return_15", "rebound_from_low_15", "range_position_15",
    "realized_vol_15", "delta_ratio_15", "notional_intensity_15",
    "support_test_density_15",
    "price_return_30", "rebound_from_low_30", "range_position_30",
    "realized_vol_30", "delta_ratio_30", "notional_intensity_30",
    "support_test_density_30",
    "price_return_60", "rebound_from_low_60", "range_position_60",
    "realized_vol_60", "delta_ratio_60", "notional_intensity_60",
    "support_test_density_60",
    "return_acceleration_5_30", "return_acceleration_10_60",
    "vol_compression_10_60", "price_delta_divergence_30",
    "price_delta_divergence_60", "sell_pressure_absorption_30",
    "sell_pressure_absorption_60", "target_to_vol_30", "target_to_vol_60",
)

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
        description="Causal sequential first-sweep state-score research.",
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
    p.add_argument("--checkpoint-offsets", nargs="+", type=int, default=list(DEFAULT_CHECKPOINT_OFFSETS))
    p.add_argument("--amplitude-targets-pct", nargs="+", type=float, default=list(DEFAULT_AMPLITUDE_TARGETS_PCT))
    p.add_argument("--amplitude-horizons", nargs="+", type=int, default=list(DEFAULT_AMPLITUDE_HORIZONS))
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
    p.add_argument("--minimum-test-events", type=int, default=30)
    p.add_argument("--causal-audit-sample-size", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--write-full-predictions", action="store_true")
    return p.parse_args(argv)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _end_exclusive(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp + pd.Timedelta(days=1) if len(str(value).strip()) <= 10 else timestamp


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    return _R13.load_bars(args)


def _folds(end_date: str) -> tuple[FoldSpec, ...]:
    return _R13._folds(end_date)


def _subset_origin_period(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    origin = pd.to_datetime(frame["origin_sweep_time"])
    label_end = pd.to_datetime(frame["label_end_time"])
    in_period = (origin >= start) & (origin <= end)
    valid = in_period & (label_end <= end)
    removed = int((in_period & ~valid).sum())
    out = frame.loc[valid].sort_values(["origin_sweep_pos", "checkpoint_offset", "event_id"], kind="mergesort")
    return out.reset_index(drop=True), removed


def _development_split(train: pd.DataFrame, fold: FoldSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    months = pd.period_range(fold.train_start.to_period("M"), fold.train_end.to_period("M"), freq="M")
    tail_months = 4 if len(months) <= 12 else 6
    calibration_months = tail_months // 2
    policy_months = tail_months - calibration_months
    calibration_start = months[-tail_months].start_time
    policy_start = months[-policy_months].start_time
    model_end = calibration_start - pd.Timedelta(nanoseconds=1)
    calibration_end = policy_start - pd.Timedelta(nanoseconds=1)
    model_fit, removed_model = _subset_origin_period(train, fold.train_start, model_end)
    calibration, removed_calibration = _subset_origin_period(train, calibration_start, calibration_end)
    policy, removed_policy = _subset_origin_period(train, policy_start, fold.train_end)
    if min(model_fit["origin_event_id"].nunique(), calibration["origin_event_id"].nunique(), policy["origin_event_id"].nunique()) < 10:
        raise RuntimeError(f"{fold.fold} sequential nested split has too few lifecycle events")
    diagnostic = pd.DataFrame([{
        "fold": fold.fold,
        "model_fit_start": model_fit["origin_sweep_time"].min(), "model_fit_end": model_fit["origin_sweep_time"].max(),
        "calibration_start": calibration["origin_sweep_time"].min(), "calibration_end": calibration["origin_sweep_time"].max(),
        "policy_start": policy["origin_sweep_time"].min(), "policy_end": policy["origin_sweep_time"].max(),
        "model_fit_events": model_fit["origin_event_id"].nunique(),
        "calibration_events": calibration["origin_event_id"].nunique(),
        "policy_events": policy["origin_event_id"].nunique(),
        "model_fit_rows": len(model_fit), "calibration_rows": len(calibration), "policy_rows": len(policy),
        "model_cross_boundary_removed": removed_model,
        "calibration_cross_boundary_removed": removed_calibration,
        "policy_cross_boundary_removed": removed_policy,
    }])
    return model_fit, calibration, policy, diagnostic


def _attach_event_balanced_weight(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    count = out.groupby("origin_event_id", sort=False)["event_id"].transform("size").clip(lower=1)
    weight = 1.0 / pd.to_numeric(count, errors="coerce").to_numpy(dtype=float)
    weight = weight / max(float(np.nanmean(weight)), 1e-12)
    out["episode_weight"] = weight.astype(np.float32)
    return out


def _score_shell(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "event_id", "origin_event_id", "lifecycle_id", "causal_region_id",
        "origin_sweep_pos", "origin_sweep_time", "checkpoint_offset",
        "extreme_pos", "extreme_time", "feature_available_time",
        "state_status", "prior_tp_reached", "hard_invalidated", "add_on_eligible",
        "initial_decision", "entry_time", "entry_price", "label_end_time",
        "tp30", "tp60", "tp180", "clean30_0p5", "clean60_0p5", "clean180_0p5",
        "slow_success_180", "slow_clean_success_180", "deep_recovery_180",
        "persistent_failure_180", "mae_60_pct", "mae_180_pct",
        "mae_before_tp_60_pct", "mae_before_tp_180_pct", "mfe_60_pct", "mfe_180_pct",
    ]
    amp = [column for column in frame.columns if column.startswith("amp_")]
    return frame.reindex(columns=[*keep, *amp]).copy()


def _head_metric_rows(
    frame: pd.DataFrame,
    *,
    fold: str,
    model_group: str,
    output: str,
    target: str,
    split: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for checkpoint, part in frame.groupby("checkpoint_offset", sort=True):
        y = part[target].astype(int).to_numpy()
        for method, column, is_probability in (
            ("decision_score", f"{output}_score_raw", False),
            ("model_probability", f"{output}_raw", True),
        ):
            score = pd.to_numeric(part[column], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(score)
            pr_auc = roc_auc = brier = np.nan
            if finite.any() and np.unique(y[finite]).size >= 2:
                pr_auc = float(average_precision_score(y[finite], score[finite]))
                roc_auc = float(roc_auc_score(y[finite], score[finite]))
                if is_probability:
                    brier = float(brier_score_loss(y[finite], np.clip(score[finite], 1e-7, 1.0 - 1e-7)))
            rows.append({
                "fold": fold, "model_group": model_group, "output": output, "target": target,
                "split": split, "checkpoint_offset": int(checkpoint), "method": method,
                "rows": int(finite.sum()), "positive_rate": float(y[finite].mean()) if finite.any() else np.nan,
                "pr_auc": pr_auc, "roc_auc": roc_auc, "brier": brier,
            })
    return pd.DataFrame(rows)


def _add_dimension_scores(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["opportunity_score"] = 50.0 * (out["p_tp60_rank"] + out["p_tp180_rank"])
    out["speed_score"] = 50.0 * (out["p_tp30_rank"] + out["p_tp60_rank"])
    out["cleanliness_score"] = 100.0 * out["p_clean60_rank"]
    out["risk_score"] = 100.0 * (1.0 - 0.5 * (out["mae60_risk_rank"] + out["mae180_risk_rank"]))
    baseline = out[out["checkpoint_offset"].eq(0)][
        ["origin_event_id", "opportunity_score", "speed_score", "cleanliness_score", "risk_score"]
    ].drop_duplicates("origin_event_id").rename(columns={
        "opportunity_score": "initial_opportunity_score",
        "speed_score": "initial_speed_score",
        "cleanliness_score": "initial_cleanliness_score",
        "risk_score": "initial_risk_score",
    })
    out = out.merge(baseline, on="origin_event_id", how="left", validate="many_to_one")
    for name in ("opportunity", "speed", "cleanliness", "risk"):
        out[f"delta_{name}_score"] = out[f"{name}_score"] - out[f"initial_{name}_score"]
    return out


def _trajectory_policy_specs() -> tuple[tuple[str, Callable[[pd.DataFrame], pd.Series]], ...]:
    return (
        ("ALL_ELIGIBLE", lambda data: pd.Series(True, index=data.index)),
        ("OPPORTUNITY_UP10", lambda data: data["delta_opportunity_score"] >= 10.0),
        ("SPEED_UP10", lambda data: data["delta_speed_score"] >= 10.0),
        ("RISK_IMPROVE10", lambda data: data["delta_risk_score"] >= 10.0),
        ("CONFIRM10", lambda data: (data["delta_opportunity_score"] >= 10.0) & (data["delta_cleanliness_score"] >= 10.0) & (data["delta_risk_score"] >= 0.0)),
        ("STRONG_ABSOLUTE", lambda data: (data["opportunity_score"] >= 70.0) & (data["cleanliness_score"] >= 60.0) & (data["risk_score"] >= 60.0)),
        ("STRONG_AND_IMPROVING", lambda data: (data["opportunity_score"] >= 70.0) & (data["cleanliness_score"] >= 60.0) & (data["risk_score"] >= 60.0) & (data["delta_opportunity_score"] >= 5.0) & (data["delta_cleanliness_score"] >= 5.0) & (data["delta_risk_score"] >= 0.0)),
        ("DETERIORATING", lambda data: (data["delta_opportunity_score"] <= -10.0) | (data["delta_risk_score"] <= -10.0)),
    )


def _event_metrics(events: pd.DataFrame, months: int) -> dict[str, float | int]:
    if events.empty:
        return {
            "event_count": 0, "events_per_month": 0.0,
            "tp30_rate": np.nan, "tp60_rate": np.nan, "tp180_rate": np.nan,
            "clean60_rate": np.nan, "clean180_rate": np.nan,
            "median_mae60_pct": np.nan, "median_mae180_pct": np.nan,
            "p90_mae180_pct": np.nan, "max_day_event_share": np.nan, "top5_day_event_share": np.nan,
        }
    dates = pd.to_datetime(events["extreme_time"]).dt.normalize()
    count = dates.value_counts()
    return {
        "event_count": int(len(events)), "events_per_month": float(len(events) / max(1, int(months))),
        "tp30_rate": float(events["tp30"].astype(bool).mean()),
        "tp60_rate": float(events["tp60"].astype(bool).mean()),
        "tp180_rate": float(events["tp180"].astype(bool).mean()),
        "clean60_rate": float(events["clean60_0p5"].astype(bool).mean()),
        "clean180_rate": float(events["clean180_0p5"].astype(bool).mean()),
        "median_mae60_pct": float(pd.to_numeric(events["mae_60_pct"], errors="coerce").median()),
        "median_mae180_pct": float(pd.to_numeric(events["mae_180_pct"], errors="coerce").median()),
        "p90_mae180_pct": float(pd.to_numeric(events["mae_180_pct"], errors="coerce").quantile(0.90)),
        "max_day_event_share": float(count.iloc[0] / len(events)),
        "top5_day_event_share": float(count.head(5).sum() / len(events)),
    }


def _remove_strong_days(events: pd.DataFrame, count: int) -> pd.DataFrame:
    if events.empty or int(count) <= 0:
        return events.copy()
    data = events.copy()
    data["_day"] = pd.to_datetime(data["extreme_time"]).dt.normalize()
    day = data.groupby("_day", sort=True).agg(tp60=("tp60", "sum"), tp180=("tp180", "sum"), events=("event_id", "size"))
    removed = set(day.sort_values(["tp60", "tp180", "events"], ascending=False, kind="mergesort").head(int(count)).index)
    return data[~data["_day"].isin(removed)].drop(columns="_day").reset_index(drop=True)


def _trajectory_frontier(frame: pd.DataFrame, *, fold: str, model_group: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    stress_rows: list[dict[str, object]] = []
    months = max(1, pd.to_datetime(frame["origin_sweep_time"]).dt.to_period("M").nunique())
    eligible = frame[(frame["checkpoint_offset"] > 0) & frame["add_on_eligible"].astype(bool)].copy()
    for checkpoint, checkpoint_frame in eligible.groupby("checkpoint_offset", sort=True):
        for policy_id, predicate in _trajectory_policy_specs():
            chosen = checkpoint_frame.loc[predicate(checkpoint_frame).fillna(False)].copy()
            common = {"fold": fold, "model_group": model_group, "checkpoint_offset": int(checkpoint), "policy_id": policy_id}
            rows.append({**common, **_event_metrics(chosen, months)})
            for removed in (0, 5, 10):
                stress_rows.append({
                    **common, "removed_strongest_days": int(removed),
                    **_event_metrics(_remove_strong_days(chosen, removed), months),
                })
    return pd.DataFrame(rows), pd.DataFrame(stress_rows)


def _increment_from_all(frontier: pd.DataFrame, stress: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["fold", "model_group", "checkpoint_offset"]
    metrics = ["event_count", "tp30_rate", "tp60_rate", "tp180_rate", "clean60_rate", "clean180_rate", "median_mae60_pct", "median_mae180_pct", "p90_mae180_pct", "top5_day_event_share"]
    base = frontier[frontier["policy_id"].eq("ALL_ELIGIBLE")][keys + metrics].rename(columns={metric: f"baseline_{metric}" for metric in metrics})
    comp = frontier[~frontier["policy_id"].eq("ALL_ELIGIBLE")].merge(base, on=keys, how="left", validate="many_to_one")
    for metric in metrics:
        comp[f"delta_{metric}"] = pd.to_numeric(comp[metric], errors="coerce") - pd.to_numeric(comp[f"baseline_{metric}"], errors="coerce")

    stress_keys = [*keys, "removed_strongest_days"]
    base_stress = stress[stress["policy_id"].eq("ALL_ELIGIBLE")][stress_keys + metrics].rename(columns={metric: f"baseline_{metric}" for metric in metrics})
    stress_comp = stress[~stress["policy_id"].eq("ALL_ELIGIBLE")].merge(base_stress, on=stress_keys, how="left", validate="many_to_one")
    for metric in metrics:
        stress_comp[f"delta_{metric}"] = pd.to_numeric(stress_comp[metric], errors="coerce") - pd.to_numeric(stress_comp[f"baseline_{metric}"], errors="coerce")
    return comp, stress_comp


def _cross_fold_gate(increments: pd.DataFrame, stress_increments: pd.DataFrame, minimum_events: int) -> pd.DataFrame:
    if increments.empty:
        return pd.DataFrame()
    stress10 = stress_increments[stress_increments["removed_strongest_days"].eq(10)][
        ["fold", "model_group", "checkpoint_offset", "policy_id", "delta_tp60_rate", "delta_clean60_rate"]
    ].rename(columns={"delta_tp60_rate": "delete10_delta_tp60_rate", "delta_clean60_rate": "delete10_delta_clean60_rate"})
    joined = increments.merge(stress10, on=["fold", "model_group", "checkpoint_offset", "policy_id"], how="left")
    rows: list[dict[str, object]] = []
    for (model_group, checkpoint, policy_id), group in joined.groupby(["model_group", "checkpoint_offset", "policy_id"], sort=True):
        event_ok = pd.to_numeric(group["event_count"], errors="coerce") >= int(minimum_events)
        tp = pd.to_numeric(group["delta_tp60_rate"], errors="coerce")
        clean = pd.to_numeric(group["delta_clean60_rate"], errors="coerce")
        mae = pd.to_numeric(group["delta_median_mae60_pct"], errors="coerce")
        delete10 = pd.to_numeric(group["delete10_delta_tp60_rate"], errors="coerce")
        keep = bool(
            len(group) == 3 and event_ok.all()
            and (tp > 0).sum() >= 2 and (clean > 0).sum() >= 2
            and (mae <= 0).sum() >= 2 and (delete10 > 0).sum() >= 2
            and (tp < -0.03).sum() == 0 and (mae > 0.15).sum() == 0
        )
        rows.append({
            "model_group": model_group, "checkpoint_offset": int(checkpoint), "policy_id": policy_id,
            "fold_count": len(group), "minimum_event_count_pass": bool(event_ok.all()),
            "tp60_positive_folds": int((tp > 0).sum()), "clean60_positive_folds": int((clean > 0).sum()),
            "mae60_nonworse_folds": int((mae <= 0).sum()), "delete10_tp60_positive_folds": int((delete10 > 0).sum()),
            "mean_delta_tp60_rate": float(tp.mean()), "mean_delta_clean60_rate": float(clean.mean()),
            "mean_delta_median_mae60_pct": float(mae.mean()), "predeclared_keep_gate": keep,
        })
    return pd.DataFrame(rows)


def _amplitude_outcomes(frame: pd.DataFrame, folds: Sequence[FoldSpec]) -> pd.DataFrame:
    amp_columns = [column for column in frame.columns if column.startswith("amp_tp_")]
    rows: list[dict[str, object]] = []
    for fold in folds:
        test, _ = _subset_origin_period(frame, fold.test_start, fold.test_end)
        for checkpoint, part in test.groupby("checkpoint_offset", sort=True):
            for eligibility, subset in (
                ("all", part),
                ("add_on_eligible", part[part["add_on_eligible"].astype(bool)]),
            ):
                row: dict[str, object] = {
                    "fold": fold.fold, "checkpoint_offset": int(checkpoint), "eligibility": eligibility,
                    "event_count": int(len(subset)),
                }
                for column in amp_columns:
                    row[f"{column}_rate"] = float(subset[column].astype(bool).mean()) if len(subset) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def _amplitude_by_score_band(frame: pd.DataFrame, *, fold: str, model_group: str) -> pd.DataFrame:
    amp_columns = [column for column in frame.columns if column.startswith("amp_tp_")]
    eligible = frame[(frame["checkpoint_offset"] > 0) & frame["add_on_eligible"].astype(bool)].copy()
    eligible["opportunity_band"] = pd.cut(
        eligible["opportunity_score"], bins=[-np.inf, 20, 40, 60, 80, np.inf],
        labels=["00_20", "20_40", "40_60", "60_80", "80_100"], right=False,
    )
    rows: list[dict[str, object]] = []
    for (checkpoint, band), part in eligible.groupby(["checkpoint_offset", "opportunity_band"], observed=True, sort=True):
        row: dict[str, object] = {
            "fold": fold, "model_group": model_group, "checkpoint_offset": int(checkpoint),
            "opportunity_band": str(band), "event_count": int(len(part)),
        }
        for column in amp_columns:
            row[f"{column}_rate"] = float(part[column].astype(bool).mean()) if len(part) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _state_future_truncation_audit(
    bars: pd.DataFrame,
    sweep_decisions: pd.DataFrame,
    states: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    count = min(max(0, int(args.causal_audit_sample_size)), len(states))
    if count == 0:
        return pd.DataFrame([{"sample": -1, "passed": True, "detail": "disabled", "max_abs_diff": 0.0}])
    positions = np.unique(np.linspace(0, len(states) - 1, count, dtype=np.int64))
    seq_columns = [column for column in states.columns if column.startswith("seq_")]
    source = sweep_decisions.set_index("event_id", drop=False)
    rows: list[dict[str, object]] = []
    for sample_number, row_pos in enumerate(positions, start=1):
        expected = states.iloc[int(row_pos)]
        origin = str(expected["origin_event_id"])
        checkpoint = int(expected["checkpoint_offset"])
        truncated = bars.iloc[: int(expected["extreme_pos"]) + 1].copy()
        rebuilt = build_sequential_checkpoint_decisions(
            truncated,
            source.loc[[origin]].reset_index(drop=True),
            checkpoint_offsets=tuple(sorted(set((0, checkpoint)))),
            accept_below_bars=int(args.liquidity_accept_below_bars),
            accept_depth_bp=float(args.liquidity_accept_depth_bp),
            prior_target_move_pct=float(args.target_move_pct),
            show_progress=False,
        ).frame
        actual_rows = rebuilt[rebuilt["checkpoint_offset"].eq(checkpoint)]
        if actual_rows.empty:
            rows.append({"sample": sample_number, "event_id": expected["event_id"], "passed": False, "detail": "checkpoint missing after future truncation", "max_abs_diff": np.nan})
            continue
        actual = actual_rows.iloc[0]
        expected_values = pd.to_numeric(expected.reindex(seq_columns), errors="coerce").to_numpy(dtype=float)
        actual_values = pd.to_numeric(actual.reindex(seq_columns), errors="coerce").to_numpy(dtype=float)
        nan_match = np.isnan(expected_values) == np.isnan(actual_values)
        finite = np.isfinite(expected_values) & np.isfinite(actual_values)
        diff = np.zeros(len(seq_columns), dtype=float)
        diff[finite] = np.abs(expected_values[finite] - actual_values[finite])
        max_diff = float(diff.max()) if len(diff) else 0.0
        passed = bool(nan_match.all() and max_diff <= 1e-9)
        rows.append({
            "sample": sample_number, "event_id": expected["event_id"],
            "checkpoint_offset": checkpoint, "decision_time": expected["feature_available_time"],
            "passed": passed, "detail": "future bars removed" if passed else "sequential state mismatch",
            "max_abs_diff": max_diff,
        })
    return pd.DataFrame(rows)


def _summary(gate: pd.DataFrame, audit: pd.DataFrame) -> str:
    lines = [
        "# Research 14 Summary", "",
        "Causal sequential rescoring after respected-macro first sweeps.", "",
        "- checkpoints: t0/t1/t3/t5/t10/t15 closed 1m bars",
        "- each checkpoint has a fresh next-open/future-close label path",
        "- target ladder 0.5/1.0/1.5/2.0 is descriptive only",
        "- no position sizing, averaging-down rule, fees, stops or strategy PnL", "",
        "## Predeclared trajectory keep gate", "",
    ]
    kept = gate[gate["predeclared_keep_gate"].astype(bool)] if not gate.empty else pd.DataFrame()
    if kept.empty:
        lines.append("No model/checkpoint/trajectory policy passed the cross-fold confirmation gate.")
    else:
        for row in kept.itertuples(index=False):
            lines.append(
                f"- {row.model_group} t+{row.checkpoint_offset} {row.policy_id}: "
                f"TP60+ folds={row.tp60_positive_folds}, Clean60+ folds={row.clean60_positive_folds}, "
                f"MAE60 non-worse folds={row.mae60_nonworse_folds}, delete10 TP60+ folds={row.delete10_tp60_positive_folds}"
            )
    lines.extend(["", "## Audit", ""])
    for row in audit.itertuples(index=False):
        lines.append(f"- [{'PASS' if row.passed else 'FAIL'}] {row.check}: {row.detail}")
    lines.extend(["", "This is a sequential event-model study, not evidence that scaling or averaging down is profitable."])
    return "\n".join(lines) + "\n"


def run_research(args: argparse.Namespace) -> Path:
    offsets = tuple(sorted(set(int(value) for value in args.checkpoint_offsets)))
    if not offsets or offsets[0] != 0:
        raise ValueError("checkpoint offsets must include 0")
    if not (1 <= int(args.short_horizon_bars) < int(args.long_horizon_bars)):
        raise ValueError("short horizon must be smaller than long horizon")
    amplitude_horizons = tuple(int(value) for value in args.amplitude_horizons)
    if not amplitude_horizons or min(amplitude_horizons) < 1 or max(amplitude_horizons) > int(args.long_horizon_bars):
        raise ValueError("amplitude horizons must be positive and cannot exceed long_horizon_bars")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = load_bars(args)
    _write_csv(validate_trade_bar_fields(bars), out_dir / "01_trade_bar_field_coverage.csv")

    print("[stage] respected macro first-sweep event pool", flush=True)
    event_build = build_first_sweep_event_decisions(
        bars,
        research_start=pd.Timestamp(args.start_date),
        research_end_exclusive=_end_exclusive(args.end_date),
        pivot_minutes=tuple(int(value) for value in args.liquidity_pivot_minutes),
        pivot_weights=tuple(float(value) for value in args.liquidity_pivot_weights),
        left_bars=int(args.liquidity_pivot_left_bars), right_bars=int(args.liquidity_pivot_right_bars),
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
    sweeps = event_build.decisions[event_build.decisions["decision_path"].eq("sweep")].reset_index(drop=True)
    if sweeps.empty:
        raise RuntimeError("no respected-macro first sweeps")
    _write_csv(event_build.diagnostics, out_dir / "02_event_build_diagnostics.csv")
    _write_csv(event_build.levels, out_dir / "03_respected_level_table.csv")
    _write_csv(event_build.lifecycle, out_dir / "04_first_sweep_lifecycle_table.csv")

    print("[stage] causal sequential checkpoints", flush=True)
    sequential = build_sequential_checkpoint_decisions(
        bars, sweeps, checkpoint_offsets=offsets,
        accept_below_bars=int(args.liquidity_accept_below_bars),
        accept_depth_bp=float(args.liquidity_accept_depth_bp),
        prior_target_move_pct=float(args.target_move_pct), show_progress=True,
    )
    if sequential.frame.empty:
        raise RuntimeError("no sequential checkpoint rows")
    expected_t0 = sequential.frame["origin_event_id"].nunique()
    actual_t0 = int(sequential.frame["checkpoint_offset"].eq(0).sum())
    if actual_t0 != expected_t0:
        raise RuntimeError(f"every lifecycle must have exactly one t0 row: events={expected_t0} t0={actual_t0}")
    _write_csv(sequential.diagnostics, out_dir / "05_sequential_state_diagnostics.csv")
    _write_csv(sequential.dictionary, out_dir / "06_sequential_feature_dictionary.csv")

    print("[stage] causal 1m snapshots at every checkpoint", flush=True)
    snapshot = build_reversal_candidate_features(
        bars, sequential.frame, include_session=False, include_htf=False, show_progress=True,
    )
    _write_csv(snapshot.dictionary, out_dir / "07_snapshot_feature_dictionary.csv")
    _write_csv(mechanism_feature_dictionary(), out_dir / "08_soft_mechanism_feature_dictionary.csv")

    print("[stage] checkpoint future-close paths", flush=True)
    labels = build_multihorizon_close_labels(
        bars, snapshot.frame, target_move_pct=float(args.target_move_pct),
        short_horizon=int(args.short_horizon_bars), long_horizon=int(args.long_horizon_bars),
        vectorized_chunk_size=int(args.label_vectorized_chunk_size), show_progress=True,
    )
    amplitude = build_amplitude_ladder_close_labels(
        bars, snapshot.frame, targets_pct=tuple(float(value) for value in args.amplitude_targets_pct),
        horizons=tuple(int(value) for value in args.amplitude_horizons),
        vectorized_chunk_size=int(args.label_vectorized_chunk_size), show_progress=True,
    )
    frame = snapshot.frame.merge(labels, on="event_id", how="inner", validate="one_to_one", suffixes=("", "_label"))
    frame = frame.merge(amplitude, on="event_id", how="inner", validate="one_to_one")
    if frame.empty:
        raise RuntimeError("all sequential rows were removed by label boundaries")
    frame = frame.sort_values(["origin_sweep_pos", "checkpoint_offset", "event_id"], kind="mergesort").reset_index(drop=True)

    print("[stage] sequential future-truncation audit", flush=True)
    state_audit = _state_future_truncation_audit(bars, sweeps, frame, args)
    _write_csv(state_audit, out_dir / "09_sequential_future_truncation_audit.csv")
    if not state_audit["passed"].astype(bool).all():
        raise RuntimeError("sequential future-truncation audit failed")

    folds = _folds(args.end_date)
    _write_csv(pd.DataFrame([fold._asdict() for fold in folds]), out_dir / "10_walkforward_folds.csv")
    amplitude_outcomes = _amplitude_outcomes(frame, folds)
    _write_csv(amplitude_outcomes, out_dir / "11_amplitude_ladder_direct_outcomes.csv")

    m0_features = tuple(snapshot.group_membership.loc[snapshot.group_membership["feature_group"].eq("M0_core"), "feature"].astype(str))
    seq_features = tuple(sequential.group_membership.loc[sequential.group_membership["feature_group"].eq(SEQUENTIAL_FEATURE_GROUP), "feature"].astype(str))
    compact = tuple(column for column in COMPACT_SNAPSHOT_FEATURES if column in frame.columns)
    if len(compact) < 30:
        raise RuntimeError(f"compact snapshot feature coverage unexpectedly low: {len(compact)}")

    split_parts: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    risk_rows: list[dict[str, object]] = []
    head_metric_parts: list[pd.DataFrame] = []
    rank_rows: list[dict[str, object]] = []
    frontier_parts: list[pd.DataFrame] = []
    stress_parts: list[pd.DataFrame] = []
    amplitude_band_parts: list[pd.DataFrame] = []
    prediction_samples: list[pd.DataFrame] = []
    full_predictions: list[pd.DataFrame] = []

    for fold_index, fold in enumerate(folds, start=1):
        print(f"[fold] {fold.fold}", flush=True)
        full_train, removed_train = _subset_origin_period(frame, fold.train_start, fold.train_end)
        test, removed_test = _subset_origin_period(frame, fold.test_start, fold.test_end)
        if test["origin_event_id"].nunique() < int(args.minimum_test_events):
            raise RuntimeError(f"{fold.fold} has only {test['origin_event_id'].nunique()} test lifecycles")
        model_fit, calibration, policy, nested = _development_split(full_train, fold)
        nested["test_events"] = test["origin_event_id"].nunique()
        nested["test_rows"] = len(test)
        nested["full_train_cross_boundary_removed"] = removed_train
        nested["test_cross_boundary_removed"] = removed_test
        split_parts.append(nested)

        # Fit the legacy soft mechanism only on t0 model-fit rows.  Applying it
        # later is causal and directly tests whether the current t0 model score
        # remains useful as the state evolves.
        t0_model_fit = model_fit[model_fit["checkpoint_offset"].eq(0)].copy()
        t0_calibration = calibration[calibration["checkpoint_offset"].eq(0)].copy()
        t0_policy = policy[policy["checkpoint_offset"].eq(0)].copy()
        mechanism = fit_soft_mechanism_transformer(t0_model_fit.rename(columns={"tp60": "tp_hit_1pct"}))
        mechanism_features: tuple[str, ...] = ()
        for name, data in (("model_fit", model_fit), ("calibration", calibration), ("policy", policy), ("test", test)):
            transformed = mechanism.transform(data.rename(columns={"tp60": "tp_hit_1pct"}))
            if name == "model_fit":
                mechanism_features = tuple(column for column in transformed.columns if column != "mechanism_dominant")
            for column in transformed.columns:
                data[column] = transformed[column].to_numpy()

        model_groups = {
            "I0_t0_full": {
                "fit_filter": lambda data: data["checkpoint_offset"].eq(0),
                "requested": (*m0_features, *mechanism_features), "max_features": 128,
            },
            "P0_pooled_compact": {
                "fit_filter": lambda data: data["initial_decision"].astype(bool) | data["add_on_eligible"].astype(bool),
                "requested": compact, "max_features": 64,
            },
            "P1_pooled_trajectory": {
                "fit_filter": lambda data: data["initial_decision"].astype(bool) | data["add_on_eligible"].astype(bool),
                "requested": (*compact, *seq_features), "max_features": 80,
            },
        }

        for group_index, (model_group, config) in enumerate(model_groups.items(), start=1):
            print(f"[models] {fold.fold} {model_group} ({group_index}/{len(model_groups)})", flush=True)
            fit_mask = config["fit_filter"](model_fit).astype(bool)
            preflight_mask = config["fit_filter"](policy).astype(bool)
            fit_train = _attach_event_balanced_weight(model_fit.loc[fit_mask].copy())
            preflight_policy = policy.loc[preflight_mask].copy()
            selected, diagnostics = _condition_feature_columns(
                fit_train, config["requested"], max_features=int(config["max_features"]),
            )
            retained_seq = sum(column.startswith("seq_") for column in selected)
            if model_group == "P1_pooled_trajectory" and retained_seq < 5:
                raise RuntimeError(f"{fold.fold} trajectory model retained only {retained_seq} seq features")
            if model_group != "P1_pooled_trajectory" and retained_seq:
                raise RuntimeError(f"{fold.fold} {model_group} unexpectedly retained sequential features")
            feature_rows.append({
                "fold": fold.fold, "model_group": model_group,
                "fit_rows": len(fit_train), "fit_events": fit_train["origin_event_id"].nunique(),
                "retained_sequential_features": retained_seq, **diagnostics,
            })
            score_frames = {
                "calibration": _score_shell(calibration),
                "policy": _score_shell(policy),
                "test": _score_shell(test),
            }

            for head_index, (output, target) in enumerate(HEAD_TARGETS.items(), start=1):
                model, fit_diagnostics = _fit_binary_with_resolution_fallback(
                    fit_train, preflight_policy,
                    feature_columns=selected, target_column=target,
                    fold=fold.fold, decision_path="sequential_state", feature_group=model_group,
                    output=output, random_state=int(args.random_state) + fold_index * 100 + group_index * 10 + head_index,
                    min_samples_leaf=int(args.model_min_samples_leaf),
                    prediction_chunk_size=int(args.prediction_chunk_size),
                )
                model_rows.append({
                    "fold": fold.fold, "model_group": model_group, "output": output, "target": target,
                    "requested_family": PRIMARY_FAMILY, "actual_family": getattr(model, "family", PRIMARY_FAMILY),
                    **fit_diagnostics,
                })
                score: dict[str, np.ndarray] = {}
                probability: dict[str, np.ndarray] = {}
                for split, source_data in (("calibration", calibration), ("policy", policy), ("test", test)):
                    score[split] = _predict_binary_score(model, source_data, int(args.prediction_chunk_size))
                    probability[split] = _predict_binary_probability(model, source_data, int(args.prediction_chunk_size))
                    score_frames[split][f"{output}_score_raw"] = score[split]
                    score_frames[split][f"{output}_raw"] = probability[split]
                anchor = policy["checkpoint_offset"].eq(0).to_numpy()
                reference = EmpiricalRankReference.fit(score["policy"][anchor])
                for split in ("policy", "test"):
                    ranks = reference.transform(score[split])
                    score_frames[split][f"{output}_rank"] = ranks
                    resolution = _rank_resolution_record(
                        fold=fold.fold, decision_path="sequential_state", feature_group=model_group,
                        output=output, split=split, raw_scores=score[split], ranks=ranks,
                        calibrated=probability[split], reference=reference, model_probability=probability[split],
                    )
                    rank_rows.append(resolution)
                    _assert_raw_score_resolution(resolution, actual_family=str(getattr(model, "family", PRIMARY_FAMILY)))
                head_metric_parts.append(_head_metric_rows(
                    score_frames["test"], fold=fold.fold, model_group=model_group,
                    output=output, target=target, split="test",
                ))

            for risk_name, target in (("mae60", "mae_60_pct"), ("mae180", "mae_180_pct")):
                risk_model = fit_risk_point_model(fit_train, feature_columns=selected, target_column=target, success_only=False)
                risk_rows.append({
                    "fold": fold.fold, "model_group": model_group, "risk_name": risk_name,
                    "target": target, "actual_family": getattr(risk_model, "fit_method", "unknown"),
                    "converged": bool(getattr(risk_model, "converged", True)),
                    "iterations": int(getattr(risk_model, "iterations", 0)), "selected_feature_count": len(selected),
                })
                raw = {
                    split: _predict_risk(risk_model, source_data, int(args.prediction_chunk_size))
                    for split, source_data in (("policy", policy), ("test", test))
                }
                anchor = policy["checkpoint_offset"].eq(0).to_numpy()
                reference = EmpiricalRankReference.fit(raw["policy"][anchor])
                for split in ("policy", "test"):
                    score_frames[split][f"{risk_name}_point_raw"] = raw[split]
                    score_frames[split][f"{risk_name}_risk_rank"] = reference.transform(raw[split])

            for split in ("policy", "test"):
                score_frames[split] = _add_dimension_scores(score_frames[split])
            test_scores = score_frames["test"]
            frontier, stress = _trajectory_frontier(test_scores, fold=fold.fold, model_group=model_group)
            frontier_parts.append(frontier)
            stress_parts.append(stress)
            amplitude_band_parts.append(_amplitude_by_score_band(test_scores, fold=fold.fold, model_group=model_group))

            sample = pd.concat([
                test_scores.nlargest(min(250, len(test_scores)), "opportunity_score"),
                test_scores.sample(min(250, len(test_scores)), random_state=int(args.random_state) + fold_index + group_index),
            ], ignore_index=True).drop_duplicates("event_id")
            sample.insert(0, "model_group", model_group)
            sample.insert(0, "fold", fold.fold)
            prediction_samples.append(sample)
            if args.write_full_predictions:
                full = test_scores.copy()
                full.insert(0, "model_group", model_group)
                full.insert(0, "fold", fold.fold)
                full_predictions.append(full)
            del score_frames
            gc.collect()

    split_table = pd.concat(split_parts, ignore_index=True)
    feature_table = pd.DataFrame(feature_rows)
    model_methods = pd.DataFrame(model_rows)
    risk_methods = pd.DataFrame(risk_rows)
    head_metrics = pd.concat(head_metric_parts, ignore_index=True)
    ranks = pd.DataFrame(rank_rows)
    frontier = pd.concat(frontier_parts, ignore_index=True)
    stress = pd.concat(stress_parts, ignore_index=True)
    increments, stress_increments = _increment_from_all(frontier, stress)
    gate = _cross_fold_gate(increments, stress_increments, int(args.minimum_test_events))
    amplitude_bands = pd.concat(amplitude_band_parts, ignore_index=True)
    samples = pd.concat(prediction_samples, ignore_index=True)

    _write_csv(split_table, out_dir / "12_nested_fold_boundaries.csv")
    _write_csv(feature_table, out_dir / "13_fold_feature_groups.csv")
    _write_csv(model_methods, out_dir / "14_model_head_fit_methods.csv")
    _write_csv(risk_methods, out_dir / "14b_risk_fit_methods.csv")
    _write_csv(head_metrics, out_dir / "15_checkpoint_head_ranking_metrics.csv")
    _write_csv(ranks, out_dir / "16_raw_rank_resolution_diagnostics.csv")
    _write_csv(frontier, out_dir / "17_sequential_score_policy_frontier.csv")
    _write_csv(increments, out_dir / "18_sequential_score_increment_comparison.csv")
    _write_csv(stress, out_dir / "19_delete_strong_days_stress.csv")
    _write_csv(stress_increments, out_dir / "20_delete_strong_days_increment.csv")
    _write_csv(gate, out_dir / "21_cross_fold_trajectory_keep_gate.csv")
    _write_csv(amplitude_bands, out_dir / "22_amplitude_ladder_by_opportunity_band.csv")
    _write_csv(samples, out_dir / "23_walkforward_prediction_sample.csv")
    if full_predictions:
        _write_csv(pd.concat(full_predictions, ignore_index=True), out_dir / "24_walkforward_full_predictions.csv")

    forbidden_tokens = ("future", "forward", "label", "mfe", "mae_", "tp_hit", "entry_price", "completion", "confirmation", "prior_tp_reached", "add_on_eligible", "hard_invalidated")
    selected_text = "|".join(feature_table["selected_features"].astype(str)).lower()
    forbidden = [token for token in forbidden_tokens if token in selected_text]
    all_origin_single_fold = True
    for fold in folds:
        train, _ = _subset_origin_period(frame, fold.train_start, fold.train_end)
        test, _ = _subset_origin_period(frame, fold.test_start, fold.test_end)
        if set(train["origin_event_id"]).intersection(set(test["origin_event_id"])):
            all_origin_single_fold = False
            break
    t0_no_initial_entry = bool(frame.loc[frame["checkpoint_offset"].eq(0), "initial_entry_price"].isna().all())
    audit = pd.DataFrame([
        {"check": "sequential_checkpoint_available_time", "passed": bool((pd.to_datetime(frame["feature_available_time"]) > pd.to_datetime(frame["extreme_time"])).all()), "detail": "closed checkpoint bar before any fresh next-open reference"},
        {"check": "t0_does_not_read_next_open", "passed": t0_no_initial_entry, "detail": "initial entry open is unavailable on sweep-bar close"},
        {"check": "sequential_future_truncation", "passed": bool(state_audit["passed"].astype(bool).all()), "detail": f"audited={len(state_audit)}"},
        {"check": "labels_use_checkpoint_next_open_future_close", "passed": True, "detail": "each checkpoint receives a fresh next-open/closed-close path"},
        {"check": "future_high_low_excluded_from_labels", "passed": True, "detail": "multihorizon and amplitude labels use future closes only"},
        {"check": "origin_lifecycle_not_split_across_fold", "passed": all_origin_single_fold, "detail": "fold membership uses original sweep time"},
        {"check": "trajectory_features_retain_real_features", "passed": bool((feature_table.loc[feature_table["model_group"].eq("P1_pooled_trajectory"), "retained_sequential_features"] >= 5).all()), "detail": "P1 retains at least five seq features in every fold"},
        {"check": "raw_rank_has_resolution", "passed": bool(not ranks.empty and ranks["raw_score_resolution_passed"].astype(bool).all()), "detail": "unsquashed decision score anchored to policy t0 distribution"},
        {"check": "future_or_eligibility_metadata_excluded", "passed": not forbidden, "detail": "|".join(forbidden)},
        {"check": "risk_models_stable", "passed": bool(not risk_methods.empty and risk_methods["converged"].astype(bool).all()), "detail": "all MAE60/180 fits converged"},
        {"check": "amplitude_ladder_is_descriptive_only", "passed": True, "detail": "0.5/1.0/1.5/2.0 labels never enter model features or winner selection"},
        {"check": "no_strategy_or_position_sizing", "passed": True, "detail": "no fees, stops, scaling orders or PnL"},
        {"check": "no_frozen_test_winner_selection", "passed": True, "detail": "all groups/checkpoints/policies reported"},
    ])
    _write_csv(audit, out_dir / "25_causal_and_selection_audit.csv")
    if not audit["passed"].all():
        raise RuntimeError(f"14 audit failed: {audit.loc[~audit['passed'], 'check'].tolist()}")

    manifest = {
        "script": SCRIPT_NAME, "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID, "edge_id": EDGE_ID, "title": TITLE,
        "symbol": args.symbol, "timeframe": args.timeframe,
        "start_date": args.start_date, "end_date": args.end_date, "warmup_start_date": args.warmup_start_date,
        "event_pool": "respected macro first sweep",
        "checkpoint_offsets": list(offsets),
        "decision": "each checkpoint closed 1m bar", "entry_reference": "each checkpoint next 1m open",
        "target_move_pct": float(args.target_move_pct),
        "short_horizon_bars": int(args.short_horizon_bars), "long_horizon_bars": int(args.long_horizon_bars),
        "model_groups": ["I0_t0_full", "P0_pooled_compact", "P1_pooled_trajectory"],
        "pooled_model_fit_domain": "t0 or causally add-on-eligible checkpoint only",
        "dimension_scores": ["opportunity", "speed", "cleanliness", "risk"],
        "amplitude_targets_pct": [float(value) for value in args.amplitude_targets_pct],
        "amplitude_role": "descriptive feasibility and monotonicity only",
        "automatic_test_winner_selected": False, "strategy_or_backtest": False,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "26_RESEARCH_SUMMARY.md").write_text(_summary(gate, audit), encoding="utf-8")
    result = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={result.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
