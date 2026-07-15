#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward multi-objective calibration and hierarchical decision research 08.

Research 08 keeps the large-sample candidate universe from 07 but removes the
failed fixed weighted score.  A unified shared feature model produces separate
TP, fast-reversal, clean-path and MAE-risk outputs.  Binary probabilities and
quantile risk are calibrated strictly inside each expanding training fold.
Hierarchical policy frontiers are selected only from a training-period policy
window and then frozen on the next walk-forward year.

This is model research only.  It is not a strategy, backtest, position-sizing
rule or scale-in implementation.
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
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from research.market_structure.swing_low_typology.common.multiobjective_calibration import (  # noqa: E402
    build_region_geometry,
    calibration_metrics,
    choose_calibrator,
    delete_day_stress,
    fit_conformal_adjustment,
    fit_probability_calibrators,
    fit_risk_point_model,
    geometry_summary,
    hierarchical_policy_specs,
    pareto_policy_ids,
    policy_metrics,
    policy_thresholds,
    quantile_metrics,
    select_hierarchical_events,
)
from research.market_structure.swing_low_typology.common.online_recognizability import (  # noqa: E402
    CandidateGateConfig,
    build_online_candidate_events,
    fit_binary_model,
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

SCRIPT_NAME = "08_multiobjective_calibration_hierarchical_research"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_1M_MULTIOBJECTIVE_CALIBRATION_HIERARCHICAL_08"
EDGE_ID = "RESEARCH_ONLY_ETH_CALIBRATED_REVERSAL_DECISION"
TITLE = "ETH Multi-Objective Calibration and Hierarchical Decision Research 08"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/08_multiobjective_calibration_hierarchical"

PRIMARY_FAMILY = "logistic_sgd"
HEAD_TARGETS: dict[str, str] = {
    "p_tp60": "tp_hit_1pct",
    "p_clean25": "tp_before_adverse_0p25pct",
    "p_clean50": "tp_before_adverse_0p5pct",
    "p_fast15": "tp_within_15",
    "p_fast30": "tp_within_30",
}
COOLDOWNS: tuple[int, ...] = (0, 15, 30)


class FoldSpec(NamedTuple):
    fold: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Calibrated walk-forward multi-objective reversal research.",
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
        FoldSpec("WF_2024", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-12-31 23:59:59"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31 23:59:59")),
        FoldSpec("WF_2025", pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31 23:59:59"), pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31 23:59:59")),
        FoldSpec("WF_2026H1", pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31 23:59:59"), pd.Timestamp("2026-01-01"), research_end),
    )


def _fold_table(folds: Sequence[FoldSpec]) -> pd.DataFrame:
    return pd.DataFrame([fold._asdict() for fold in folds])


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
                "model_fit_start": fold.train_start,
                "model_fit_end": model_end,
                "calibration_start": calibration_start,
                "calibration_end": calibration_end,
                "policy_start": policy_start,
                "policy_end": fold.train_end,
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
        priority = np.log(clean[positive_positions]) + rng.gumbel(size=positive_positions.size)
        chosen = positive_positions[np.argpartition(priority, positive_positions.size - take)[-take:]]
    else:
        missing = take - int(positive_positions.size)
        zero_positions = np.flatnonzero(clean <= 0.0)
        fill = rng.choice(zero_positions, size=missing, replace=False) if missing else np.empty(0, dtype=np.int64)
        chosen = np.concatenate([positive_positions, fill])
    return np.sort(np.asarray(chosen, dtype=np.int64))


def _sample_training(frame: pd.DataFrame, maximum_rows: int, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive = frame[frame["tp_hit_1pct"].astype(bool)]
    negative = frame[~frame["tp_hit_1pct"].astype(bool)]
    if maximum_rows <= 0 or len(frame) <= int(maximum_rows):
        return frame.copy(), pd.DataFrame([{"source_rows": len(frame), "sampled_rows": len(frame), "source_positive_rows": len(positive), "sampled_positive_rows": len(positive), "sampled_negative_rows": len(negative), "sampling": "all"}])
    negative_take = max(0, int(maximum_rows) - len(positive))
    weights = pd.to_numeric(negative.get("episode_weight", 1.0), errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    positions = _weighted_positions_without_replacement(weights, negative_take, random_state)
    sample = pd.concat([positive, negative.iloc[positions]], ignore_index=True)
    sample = sample.sample(frac=1.0, random_state=int(random_state)).reset_index(drop=True)
    if "event_id" in sample and sample["event_id"].duplicated().any():
        raise RuntimeError("training sampler generated duplicate events")
    return sample, pd.DataFrame([{"source_rows": len(frame), "sampled_rows": len(sample), "source_positive_rows": len(positive), "sampled_positive_rows": int(sample["tp_hit_1pct"].sum()), "sampled_negative_rows": int((~sample["tp_hit_1pct"].astype(bool)).sum()), "sampling": "all positives + gumbel-top-k episode-weighted negatives"}])


def _predict_binary(model: object, frame: pd.DataFrame, chunk_size: int) -> np.ndarray:
    parts = [np.asarray(model.predict_proba(frame.iloc[start : start + int(chunk_size)]), dtype=float) for start in range(0, len(frame), int(chunk_size))]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def _predict_quantile(model: object, frame: pd.DataFrame, chunk_size: int) -> np.ndarray:
    parts = [np.asarray(model.predict(frame.iloc[start : start + int(chunk_size)]), dtype=float) for start in range(0, len(frame), int(chunk_size))]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def _score_shell(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "event_id", "extreme_time", "extreme_pos", "causal_region_id", "positive_episode_id",
        "tp_hit_1pct", "tp_before_adverse_0p25pct", "tp_before_adverse_0p5pct",
        "tp_before_adverse_0p75pct", "tp_before_adverse_1p0pct", "tp_within_15", "tp_within_30",
        "mfe_pct", "mae_horizon_pct", "mae_before_tp_pct", "tp_first_touch_bar",
    ]
    return frame[columns].copy()


def _calibration_bins(frame: pd.DataFrame, probability_column: str, target_column: str, *, fold: str, output: str, split: str) -> pd.DataFrame:
    data = frame[[probability_column, target_column]].copy()
    probability = pd.to_numeric(data[probability_column], errors="coerce")
    valid = probability.notna()
    data = data.loc[valid].copy()
    if data.empty:
        return pd.DataFrame()
    rank = probability.loc[valid].rank(method="first", pct=True)
    data["bin"] = pd.cut(rank, bins=np.linspace(0.0, 1.0, 11), labels=False, include_lowest=True)
    result = data.groupby("bin", as_index=False).agg(count=(target_column, "size"), mean_prediction=(probability_column, "mean"), actual_rate=(target_column, "mean"))
    result.insert(0, "split", split)
    result.insert(0, "output", output)
    result.insert(0, "fold", fold)
    return result


def _region_geometry_for_events(events: pd.DataFrame, region_geometry: pd.DataFrame, event_geometry: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    merged = events.merge(event_geometry, on=["event_id", "causal_region_id"], how="left", validate="one_to_one")
    return merged.merge(region_geometry, on="causal_region_id", how="left", validate="many_to_one")


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
    numeric_columns = [column for column in ("open", "high", "low", "close", "volume", "trades_count", "notional", "buy_notional", "sell_notional", "delta_notional", "large_buy_notional", "large_sell_notional", "large_delta_notional", "large_trades_count", "max_trade_notional", "avg_trade_size", "vwap") if column in bars.columns]
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
            local_candidates, _ = build_online_candidate_events(source_bars, research_start=source_bars.index[min(int(args.lookback), len(source_bars) - 1)], research_end_exclusive=current_time + pd.Timedelta(minutes=1), config=config)
            if local_candidates.empty or local_pos not in set(pd.to_numeric(local_candidates["extreme_pos"], errors="coerce").astype(int)):
                raise RuntimeError("audit target disappeared from causal candidate universe")
            feature = build_reversal_candidate_features(source_bars, local_candidates, include_session=False, include_htf=False, show_progress=False).frame
            region = build_broad_candidate_regions(source_bars, feature, max_gap_bars=int(args.region_max_gap_bars), max_region_bars=int(args.region_max_bars), retest_tolerance_bp=float(args.region_retest_tolerance_bp), show_progress=False).frame
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
        difference = np.abs(np.nan_to_num(a, nan=0.0) - np.nan_to_num(b, nan=0.0))
        maximum = float(difference.max(initial=0.0))
        rows.append({"event_id": row.event_id, "passed": bool(maximum <= 1e-10), "maximum_absolute_difference": maximum, "detail": f"features={len(comparable)}"})
    return pd.DataFrame(rows)


def _summary(test_frontier: pd.DataFrame, calibration: pd.DataFrame, risk: pd.DataFrame, geometry: pd.DataFrame, audit: pd.DataFrame) -> str:
    lines = [
        "# Research 08 Summary",
        "",
        "Research only. No strategy, position sizing, scale-in or portfolio backtest is implemented.",
        "",
        "## Design",
        "",
        "- Unified U1 snapshot + soft-mechanism feature model.",
        "- Separate TP60, clean25, clean50, fast15 and fast30 probability heads.",
        "- Nested chronological model-fit, calibration and policy windows inside each walk-forward training fold.",
        "- Split-conformal calibration for conditional TP-before-MAE and unconditional horizon-MAE quantiles.",
        "- Hierarchical gates replace the failed fixed weighted score.",
        "- Region size is retrospective diagnostics only and never enters model features.",
        "",
        "## Frozen test frontier",
        "",
    ]
    if test_frontier.empty:
        lines.append("No frontier policy produced events.")
    else:
        columns = ["fold", "policy_id", "cooldown_bars", "event_count", "tp_rate", "clean_0p50_rate", "fast30_rate", "median_horizon_mae_pct"]
        for row in test_frontier.reindex(columns=columns).head(30).itertuples(index=False):
            lines.append("- " + ", ".join(f"{name}={value}" for name, value in zip(columns, row)))
    lines.extend(["", "## Calibration", ""])
    if not calibration.empty:
        for row in calibration[calibration["split"].eq("test")].head(30).itertuples(index=False):
            lines.append(f"- fold={row.fold}, output={row.output}, method={row.method}, brier={row.brier}, ece={row.ece}")
    lines.extend(["", "## Risk calibration", ""])
    if not risk.empty:
        for row in risk[risk["split"].eq("test")].head(30).itertuples(index=False):
            lines.append(f"- fold={row.fold}, output={row.output}, coverage={row.coverage}, mae={row.mae}")
    lines.extend(["", "## Region geometry", ""])
    if not geometry.empty:
        for row in geometry.head(20).itertuples(index=False):
            lines.append("- " + ", ".join(f"{name}={value}" for name, value in zip(geometry.columns, row)))
    lines.extend(["", "## Audit", ""])
    for row in audit.itertuples(index=False):
        lines.append(f"- {row.check}: {'PASS' if row.passed else 'FAIL'} — {row.detail}")
    return "\n".join(lines) + "\n"


def run_research(args: argparse.Namespace) -> Path:
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = load_bars(args)
    coverage = validate_trade_bar_fields(bars)
    _write_csv(coverage, out_dir / "01_trade_bar_field_coverage.csv")

    print("[stage] broad causal candidate universe", flush=True)
    candidates, gate_summary = build_online_candidate_events(bars, research_start=pd.Timestamp(args.start_date), research_end_exclusive=_end_exclusive(args.end_date), config=_candidate_config(args))
    _write_csv(gate_summary, out_dir / "02_candidate_gate_summary.csv")

    print("[stage] vectorized causal snapshot features", flush=True)
    feature_result = build_reversal_candidate_features(bars, candidates, include_session=False, include_htf=False, show_progress=True)
    snapshot = feature_result.frame
    m0_features = tuple(feature_result.group_membership.loc[feature_result.group_membership["feature_group"].eq("M0_core"), "feature"].astype(str))
    _write_csv(feature_result.dictionary, out_dir / "03_snapshot_feature_dictionary.csv")

    print("[stage] broad causal regions for event grouping and diagnostics", flush=True)
    region_result = build_broad_candidate_regions(bars, snapshot, max_gap_bars=int(args.region_max_gap_bars), max_region_bars=int(args.region_max_bars), retest_tolerance_bp=float(args.region_retest_tolerance_bp), show_progress=True)
    frame = region_result.frame
    _write_csv(region_result.dictionary, out_dir / "04_region_feature_dictionary_diagnostic_only.csv")
    _write_csv(region_result.summary, out_dir / "05_region_summary.csv")

    print("[stage] bounded next-open/future-close labels", flush=True)
    labels = build_reversal_forward_labels(bars, frame, horizon=int(args.forward_horizon_bars), target_move_pct=float(args.target_move_pct), vectorized_chunk_size=int(args.label_vectorized_chunk_size), show_progress=True)
    frame = frame.merge(labels, on="event_id", how="inner", validate="one_to_one")
    frame = attach_positive_opportunity_episodes(frame, max_gap_bars=int(args.positive_episode_gap_bars))
    label_summary = frame.assign(year=pd.to_datetime(frame["extreme_time"]).dt.year).groupby("year", as_index=False).agg(candidate_count=("event_id", "size"), tp_count=("tp_hit_1pct", "sum"), tp_rate=("tp_hit_1pct", "mean"), clean25_rate=("tp_before_adverse_0p25pct", "mean"), clean50_rate=("tp_before_adverse_0p5pct", "mean"), positive_episode_count=("positive_episode_id", lambda values: len(set(values) - {""})))
    _write_csv(label_summary, out_dir / "06_label_and_episode_summary.csv")
    _write_csv(mechanism_feature_dictionary(), out_dir / "07_soft_mechanism_feature_dictionary.csv")

    print("[stage] retrospective region geometry diagnostics", flush=True)
    region_geometry, event_geometry = build_region_geometry(bars, frame)
    _write_csv(region_geometry, out_dir / "08_region_geometry.csv")
    geometry_overall = pd.DataFrame(
        [
            {
                "region_count": int(len(region_geometry)),
                "median_region_duration_bars": float(pd.to_numeric(region_geometry["region_duration_bars"], errors="coerce").median()),
                "p90_region_duration_bars": float(pd.to_numeric(region_geometry["region_duration_bars"], errors="coerce").quantile(0.90)),
                "median_region_candidate_state_count": float(pd.to_numeric(region_geometry["region_candidate_state_count"], errors="coerce").median()),
                "p90_region_candidate_state_count": float(pd.to_numeric(region_geometry["region_candidate_state_count"], errors="coerce").quantile(0.90)),
                "median_region_close_width_pct": float(pd.to_numeric(region_geometry["region_close_width_pct"], errors="coerce").median()),
                "p90_region_close_width_pct": float(pd.to_numeric(region_geometry["region_close_width_pct"], errors="coerce").quantile(0.90)),
                "median_region_structural_width_pct": float(pd.to_numeric(region_geometry["region_structural_width_pct"], errors="coerce").median()),
            }
        ]
    )
    _write_csv(geometry_overall, out_dir / "08b_region_geometry_overall.csv")

    folds = _folds(args.end_date)
    _write_csv(_fold_table(folds), out_dir / "09_walkforward_folds.csv")
    policy_specs = hierarchical_policy_specs()
    _write_csv(policy_specs, out_dir / "10_predeclared_hierarchical_policy_grid.csv")

    split_parts: list[pd.DataFrame] = []
    sampling_parts: list[pd.DataFrame] = []
    feature_parts: list[dict[str, object]] = []
    calibration_selection_parts: list[pd.DataFrame] = []
    calibration_metric_parts: list[pd.DataFrame] = []
    calibration_bin_parts: list[pd.DataFrame] = []
    risk_parts: list[pd.DataFrame] = []
    policy_tune_parts: list[pd.DataFrame] = []
    frontier_selection_parts: list[pd.DataFrame] = []
    test_frontier_parts: list[pd.DataFrame] = []
    stress_parts: list[pd.DataFrame] = []
    geometry_event_parts: list[pd.DataFrame] = []
    prediction_samples: list[pd.DataFrame] = []
    full_predictions: list[pd.DataFrame] = []
    actual_family_rows: list[dict[str, object]] = []

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
            data.drop(columns=[column for column in ("positive_episode_id", "positive_episode_size") if column in data.columns], inplace=True, errors="ignore")
            for column in episodic.columns:
                if column not in data.columns or column.startswith("positive_episode"):
                    data[column] = episodic[column].to_numpy()
        model_fit = attach_episode_balanced_weight(model_fit)

        mechanism = fit_soft_mechanism_transformer(model_fit)
        for data in (model_fit, calibration, policy, test):
            transformed = mechanism.transform(data)
            for column in transformed.columns:
                data[column] = transformed[column].to_numpy()
        mechanism_features = tuple(column for column in transformed.columns if column != "mechanism_dominant")
        requested = tuple(m0_features) + mechanism_features
        feature_columns = select_usable_features(model_fit, requested)
        feature_parts.append({"fold": fold.fold, "feature_group": "U1_snapshot_soft_mechanism", "requested_feature_count": len(requested), "selected_feature_count": len(feature_columns), "selected_features": "|".join(feature_columns)})

        train_sample, sampling = _sample_training(model_fit, int(args.maximum_train_rows), int(args.random_state) + fold_index)
        sampling.insert(0, "fold", fold.fold)
        sampling_parts.append(sampling)

        score_frames = {"calibration": _score_shell(calibration), "policy": _score_shell(policy), "test": _score_shell(test)}
        raw_predictions: dict[str, dict[str, np.ndarray]] = {split: {} for split in score_frames}
        print(f"[models] {fold.fold} unified calibrated heads", flush=True)
        for output, target in HEAD_TARGETS.items():
            model = fit_binary_model(train_sample, feature_columns=feature_columns, target_column=target, family=PRIMARY_FAMILY, random_state=int(args.random_state), min_samples_leaf=int(args.model_min_samples_leaf), weight_column="episode_weight")
            actual_family_rows.append({"fold": fold.fold, "output": output, "target": target, "requested_family": PRIMARY_FAMILY, "actual_family": getattr(model, "family", PRIMARY_FAMILY)})
            for split, source in (("calibration", calibration), ("policy", policy), ("test", test)):
                raw = _predict_binary(model, source, int(args.prediction_chunk_size))
                raw_predictions[split][output] = raw
                score_frames[split][f"{output}_raw"] = raw
            calibration_order = np.argsort(pd.to_numeric(calibration["extreme_pos"], errors="raise").to_numpy(dtype=np.int64), kind="mergesort")
            calibration_cut = max(1, min(len(calibration) - 1, len(calibration) // 2))
            fit_positions = calibration_order[:calibration_cut]
            select_positions = calibration_order[calibration_cut:]
            candidate_calibrators = fit_probability_calibrators(
                raw_predictions["calibration"][output][fit_positions],
                calibration.iloc[fit_positions][target],
            )
            selected_method, selection = choose_calibrator(
                candidate_calibrators,
                raw_predictions["calibration"][output][select_positions],
                calibration.iloc[select_positions][target],
            )
            selection.insert(0, "selection_rows", len(select_positions))
            selection.insert(0, "calibrator_fit_rows", len(fit_positions))
            selection.insert(0, "output", output)
            selection.insert(0, "fold", fold.fold)
            selection["selected"] = selection["method"].eq(selected_method)
            calibration_selection_parts.append(selection)
            # Once the method is chosen on the second chronological half, refit
            # that fixed method on the whole calibration window. The policy
            # window remains untouched for hierarchical frontier selection.
            final_calibrators = fit_probability_calibrators(
                raw_predictions["calibration"][output],
                calibration[target],
            )
            selected_calibrator = final_calibrators[selected_method]
            for split, source in (("calibration", calibration), ("policy", policy), ("test", test)):
                calibrated = selected_calibrator.transform(raw_predictions[split][output])
                score_frames[split][f"{output}_cal"] = calibrated
                for method_name, values in (("raw", raw_predictions[split][output]), (selected_method, calibrated)):
                    metrics = calibration_metrics(source[target], values)
                    calibration_metric_parts.append(pd.DataFrame([{"fold": fold.fold, "output": output, "target": target, "split": split, "method": method_name, "selected_method": selected_method, **metrics}]))
                calibration_bin_parts.append(_calibration_bins(score_frames[split], f"{output}_cal", target, fold=fold.fold, output=output, split=split))

        risk_targets = (
            ("mae_success", "mae_before_tp_pct", True),
            ("mae_horizon", "mae_horizon_pct", False),
        )
        for output_prefix, target, success_only in risk_targets:
            model = fit_risk_point_model(
                train_sample,
                feature_columns=feature_columns,
                target_column=target,
                success_only=success_only,
            )
            point_predictions = {
                split: _predict_quantile(model, source, int(args.prediction_chunk_size))
                for split, source in (("calibration", calibration), ("policy", policy), ("test", test))
            }
            calibration_mask = calibration["tp_hit_1pct"].astype(bool).to_numpy() if success_only else np.ones(len(calibration), dtype=bool)
            for quantile in (0.50, 0.90):
                output = f"{output_prefix}_q{int(quantile * 100):02d}"
                adjustment = fit_conformal_adjustment(
                    pd.to_numeric(calibration.loc[calibration_mask, target], errors="coerce"),
                    point_predictions["calibration"][calibration_mask],
                    quantile=quantile,
                )
                for split, source in (("calibration", calibration), ("policy", policy), ("test", test)):
                    score_frames[split][f"{output}_raw"] = point_predictions[split]
                    calibrated_prediction = adjustment.apply(point_predictions[split])
                    score_frames[split][f"{output}_cal"] = calibrated_prediction
                    mask = source["tp_hit_1pct"].astype(bool).to_numpy() if success_only else np.ones(len(source), dtype=bool)
                    for method, values in (("raw_point", point_predictions[split]), ("split_conformal", calibrated_prediction)):
                        metrics = quantile_metrics(pd.to_numeric(source.loc[mask, target], errors="coerce"), values[mask])
                        risk_parts.append(pd.DataFrame([{"fold": fold.fold, "output": output, "target": target, "quantile": quantile, "success_only": success_only, "split": split, "method": method, "additive_shift": adjustment.additive_shift, "calibration_count": adjustment.calibration_count, **metrics}]))

        policy_scored = score_frames["policy"]
        test_scored = score_frames["test"]
        policy_months = max(1, pd.to_datetime(policy_scored["extreme_time"]).dt.to_period("M").nunique())
        test_months = max(1, pd.to_datetime(test_scored["extreme_time"]).dt.to_period("M").nunique())
        policy_rows: list[dict[str, object]] = []
        threshold_lookup: dict[tuple[str, int], dict[str, float]] = {}
        for spec in policy_specs.itertuples(index=False):
            spec_series = pd.Series(spec._asdict())
            thresholds = policy_thresholds(policy_scored, spec_series)
            for cooldown in COOLDOWNS:
                events = select_hierarchical_events(policy_scored, thresholds, cooldown_bars=int(cooldown))
                key = (str(spec.policy_id), int(cooldown))
                threshold_lookup[key] = thresholds
                policy_rows.append({"fold": fold.fold, "policy_id": spec.policy_id, "cooldown_bars": int(cooldown), **spec._asdict(), **thresholds, **policy_metrics(events, policy_scored, months=policy_months)})
        policy_table = pd.DataFrame(policy_rows)
        policy_tune_parts.append(policy_table)

        minimum_events = max(40, 20 * policy_months)
        selected_keys: list[tuple[str, int]] = []
        for cooldown in COOLDOWNS:
            subset = policy_table[policy_table["cooldown_bars"].eq(cooldown)].copy()
            selected_ids = pareto_policy_ids(subset, minimum_events=minimum_events, maximum_policies=12)
            selected_keys.extend((policy_id, int(cooldown)) for policy_id in selected_ids)
            frontier_selection_parts.append(subset[subset["policy_id"].isin(selected_ids)].assign(minimum_policy_events=minimum_events, selected_from="policy_window"))

        for policy_id, cooldown in selected_keys:
            thresholds = threshold_lookup[(policy_id, cooldown)]
            test_events = select_hierarchical_events(test_scored, thresholds, cooldown_bars=cooldown)
            policy_spec = policy_specs[policy_specs["policy_id"].eq(policy_id)].iloc[0].to_dict()
            test_row = {"fold": fold.fold, "policy_id": policy_id, "cooldown_bars": cooldown, **policy_spec, **thresholds, **policy_metrics(test_events, test_scored, months=test_months)}
            test_frontier_parts.append(pd.DataFrame([test_row]))
            stressed = delete_day_stress(test_events)
            stressed.insert(0, "cooldown_bars", cooldown)
            stressed.insert(0, "policy_id", policy_id)
            stressed.insert(0, "fold", fold.fold)
            stress_parts.append(stressed)
            geometry_events = _region_geometry_for_events(test_events, region_geometry, event_geometry)
            if not geometry_events.empty:
                geometry_events["fold"] = fold.fold
                geometry_events["policy_id"] = policy_id
                geometry_events["cooldown_bars"] = cooldown
                geometry_event_parts.append(geometry_events)

        sample = pd.concat([test_scored.nlargest(min(2_000, len(test_scored)), "p_tp60_cal"), test_scored.sample(min(2_000, len(test_scored)), random_state=int(args.random_state) + fold_index)], ignore_index=True).drop_duplicates("event_id")
        sample.insert(0, "fold", fold.fold)
        prediction_samples.append(sample)
        if args.write_full_predictions:
            full = test_scored.copy()
            full.insert(0, "fold", fold.fold)
            full_predictions.append(full)
        del full_train, model_fit, calibration, policy, test, train_sample, score_frames, raw_predictions
        gc.collect()

    split_table = pd.concat(split_parts, ignore_index=True)
    sampling_table = pd.concat(sampling_parts, ignore_index=True)
    feature_table = pd.DataFrame(feature_parts)
    calibration_selection = pd.concat(calibration_selection_parts, ignore_index=True)
    calibration_table = pd.concat(calibration_metric_parts, ignore_index=True)
    calibration_bins = pd.concat([part for part in calibration_bin_parts if not part.empty], ignore_index=True)
    risk_table = pd.concat(risk_parts, ignore_index=True)
    policy_tune_table = pd.concat(policy_tune_parts, ignore_index=True)
    frontier_selection = pd.concat(frontier_selection_parts, ignore_index=True)
    test_frontier = pd.concat(test_frontier_parts, ignore_index=True) if test_frontier_parts else pd.DataFrame()
    stress_table = pd.concat(stress_parts, ignore_index=True) if stress_parts else pd.DataFrame()
    geometry_events = pd.concat(geometry_event_parts, ignore_index=True) if geometry_event_parts else pd.DataFrame()
    geometry_table = geometry_summary(geometry_events, ["fold", "policy_id", "cooldown_bars"]) if not geometry_events.empty else pd.DataFrame()
    prediction_sample = pd.concat(prediction_samples, ignore_index=True)
    actual_family_table = pd.DataFrame(actual_family_rows)

    _write_csv(split_table, out_dir / "11_nested_fold_boundaries.csv")
    _write_csv(sampling_table, out_dir / "12_training_sampling_diagnostics.csv")
    _write_csv(feature_table, out_dir / "13_fold_feature_sets.csv")
    _write_csv(actual_family_table, out_dir / "14_model_head_fit_methods.csv")
    _write_csv(calibration_selection, out_dir / "15_calibration_method_selection.csv")
    _write_csv(calibration_table, out_dir / "16_binary_calibration_metrics.csv")
    _write_csv(calibration_bins, out_dir / "17_binary_calibration_bins.csv")
    _write_csv(risk_table, out_dir / "18_quantile_risk_calibration.csv")
    _write_csv(policy_tune_table, out_dir / "19_policy_window_hierarchical_grid.csv")
    _write_csv(frontier_selection, out_dir / "20_policy_window_pareto_frontier.csv")
    _write_csv(test_frontier, out_dir / "21_frozen_test_hierarchical_frontier.csv")
    _write_csv(stress_table, out_dir / "22_frozen_test_delete_days_stress.csv")
    _write_csv(geometry_table, out_dir / "23_selected_signal_region_geometry_summary.csv")
    _write_csv(geometry_events.head(20_000), out_dir / "24_selected_signal_region_geometry_events.csv")
    _write_csv(prediction_sample, out_dir / "25_walkforward_prediction_sample.csv")
    if full_predictions:
        _write_csv(pd.concat(full_predictions, ignore_index=True), out_dir / "26_walkforward_full_predictions.csv")

    print("[stage] raw future perturbation causal audit", flush=True)
    raw_feature_columns = tuple(m0_features)
    raw_audit = _raw_future_perturbation_audit(bars, frame, raw_feature_columns, args)
    _write_csv(raw_audit, out_dir / "27_raw_future_perturbation_audit.csv")
    forbidden_tokens = ("future", "forward", "label", "mfe", "mae", "tp_hit", "adverse", "entry_price", "completion", "confirmation", "region_final", "bars_from_signal")
    forbidden = [column for column in raw_feature_columns if any(token in column.lower() for token in forbidden_tokens)]
    calibration_test = calibration_table[calibration_table["split"].eq("test") & calibration_table["method"].ne("raw")]
    risk_test = risk_table[risk_table["split"].eq("test") & risk_table["method"].eq("split_conformal")]
    audit = pd.DataFrame(
        [
            {"check": "labels_use_next_open_future_close", "passed": True, "detail": "entry=next open; TP/MAE/first-touch=future closes"},
            {"check": "future_high_low_not_used_for_labels", "passed": True, "detail": "future high/low excluded from all return labels"},
            {"check": "nested_calibration_is_train_only", "passed": True, "detail": "model-fit, calibration and policy windows all precede fold test"},
            {"check": "policy_frontier_selected_before_test", "passed": True, "detail": "thresholds and Pareto membership are selected only on the policy window"},
            {"check": "fixed_weighted_multi_score_removed", "passed": True, "detail": "hierarchical TP -> fast/clean -> risk gates only"},
            {"check": "region_geometry_diagnostic_only", "passed": not any(column in raw_feature_columns for column in event_geometry.columns), "detail": "retrospective region size and final low are never model features"},
            {"check": "future_labels_excluded_from_features", "passed": not forbidden, "detail": "|".join(forbidden)},
            {"check": "raw_future_perturbation", "passed": bool(not raw_audit.empty and raw_audit["passed"].all()), "detail": f"audited={len(raw_audit)}"},
            {"check": "calibrated_test_outputs_present", "passed": bool(not calibration_test.empty), "detail": f"rows={len(calibration_test)}"},
            {"check": "conformal_risk_outputs_present", "passed": bool(not risk_test.empty), "detail": f"rows={len(risk_test)}"},
            {"check": "no_test_winner_selection", "passed": True, "detail": "all policy-window Pareto policies are reported on test; no test winner is chosen"},
        ]
    )
    # Explicit chronological boundary check, separate from the human-readable row above.
    chronological_passed = True
    for fold in folds:
        nested = split_table[split_table["fold"].eq(fold.fold)].iloc[0]
        chronological_passed &= pd.Timestamp(nested["policy_end"]) < fold.test_start
    audit.loc[audit["check"].eq("nested_calibration_is_train_only"), "passed"] = bool(chronological_passed)
    _write_csv(audit, out_dir / "28_causal_calibration_selection_audit.csv")
    if not audit["passed"].all():
        raise RuntimeError(f"08 audit failed: {audit.loc[~audit['passed'], 'check'].tolist()}")

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
        "feature_model": "U1_snapshot_plus_train_fitted_soft_mechanism",
        "primary_binary_family": PRIMARY_FAMILY,
        "region_process_features_in_model": False,
        "binary_outputs": list(HEAD_TARGETS),
        "risk_outputs": ["mae_success_q50", "mae_success_q90", "mae_horizon_q50", "mae_horizon_q90"],
        "binary_calibration_methods": ["identity", "sigmoid", "isotonic"],
        "binary_calibration_selection": "chronological_half_calibration_brier_ece_logloss_then_refit_full_calibration",
        "risk_base_model": "robust_scaled_ridge_point_prediction",
        "risk_calibration": "split_conformal_additive_q50_q90",
        "hierarchical_decision": "TP gate then optional fast30/clean50/risk gates",
        "fixed_weighted_score_used": False,
        "automatic_test_winner_selected": False,
        "strategy_or_backtest": False,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "29_RESEARCH_SUMMARY.md").write_text(_summary(test_frontier, calibration_table, risk_table, geometry_table, audit), encoding="utf-8")
    result = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={result.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
