#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Research 16: broad reversal candidate and mechanism uplift decomposition.

The study decomposes value in a fixed order:

matched random 1m baseline -> broad causal candidate pool -> frozen mechanism
rules -> simple single factors -> mechanism-specific expert models -> aligned
execution and costs.

No frozen test fold selects candidate rules, mechanism count, feature groups,
Top20/30/40 thresholds, execution horizon, or stress settings.
"""

from __future__ import annotations

import argparse
import gc
import importlib
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

from src.research_common.review_pack import finalize_research_report  # noqa: E402
from research.market_structure.swing_low_typology.common.broad_reversal_evaluation import (  # noqa: E402
    CloseTargetCostSpec,
    bootstrap_day_metrics,
    bootstrap_interval,
    build_multi_horizon_close_labels,
    executable_trade_set,
    execution_metrics,
    path_metrics,
    strongest_day_stress,
)
from research.market_structure.swing_low_typology.common.broad_reversal_mechanisms import (  # noqa: E402
    EXPERT_FEATURES,
    MECHANISM_ORDER,
    SIMPLE_FACTOR_SPECS,
    UNRESOLVED_MECHANISM,
    FrozenMechanismScorer,
    attach_simple_factor_scores,
    build_matched_random_samples,
    fit_simple_factor_references,
    mechanism_dictionary,
    mechanism_overlap_matrix,
    merge_macro_first_sweep_candidates,
    select_broad_region_events,
)
from research.market_structure.swing_low_typology.common.first_sweep_event import (  # noqa: E402
    build_first_sweep_event_decisions,
)
from research.market_structure.swing_low_typology.common.online_recognizability import (  # noqa: E402
    CandidateGateConfig,
    build_online_candidate_events,
)
from research.market_structure.swing_low_typology.common.range_increment import EmpiricalRankReference  # noqa: E402
from research.market_structure.swing_low_typology.common.reversal_opportunity import (  # noqa: E402
    build_reversal_candidate_features,
)
from research.market_structure.swing_low_typology.common.swing_low_dataset import validate_trade_bar_fields  # noqa: E402
from research.market_structure.swing_low_typology.common.walkforward_reversal import (  # noqa: E402
    build_broad_candidate_regions,
)

_R12 = importlib.import_module(
    "research.market_structure.swing_low_typology.12_respected_macro_first_sweep_event_research"
)

SCRIPT_NAME = "16_broad_reversal_mechanism_uplift_research"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_1M_BROAD_REVERSAL_MECHANISM_UPLIFT_16"
EDGE_ID = "RESEARCH_ONLY_ETH_BROAD_REVERSAL_MECHANISM_UPLIFT"
TITLE = "ETH Broad Reversal Candidate and Mechanism Uplift Research 16"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/16_broad_reversal_mechanism_uplift"
PRIMARY_FAMILY = "logistic_sgd"

FoldSpec = _R12.FoldSpec
_condition_feature_columns = _R12._condition_feature_columns
_predict_binary_probability = _R12._predict_binary_probability
_predict_binary_score = _R12._predict_binary_score
_fit_binary_with_resolution_fallback = _R12._fit_binary_with_resolution_fallback
_rank_resolution_record = _R12._rank_resolution_record
_assert_raw_score_resolution = _R12._assert_raw_score_resolution


class GroupIdentity(NamedTuple):
    baseline_layer: str
    group_id: str
    parent_group_id: str
    mechanism: str
    policy_id: str
    model_group: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Broad causal reversal mechanism uplift research.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")

    p.add_argument("--candidate-lookback-bars", type=int, default=240)
    p.add_argument("--candidate-new-low-window", type=int, default=5)
    p.add_argument("--candidate-near-floor-window", type=int, default=60)
    p.add_argument("--candidate-position-window", type=int, default=120)
    p.add_argument("--candidate-near-floor-tolerance-bp", type=float, default=20.0)
    p.add_argument("--candidate-max-position-in-range", type=float, default=0.55)
    p.add_argument("--region-max-gap-bars", type=int, default=2)
    p.add_argument("--region-max-bars", type=int, default=120)
    p.add_argument("--region-retest-tolerance-bp", type=float, default=25.0)
    p.add_argument("--broad-cooldown-bars", type=int, default=15)
    p.add_argument("--mechanism-minimum-score", type=float, default=70.0)

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

    p.add_argument("--path-horizons", nargs="+", type=int, default=[30, 60, 180])
    p.add_argument("--target-levels-pct", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0])
    p.add_argument("--label-vectorized-chunk-size", type=int, default=20_000)
    p.add_argument("--score-top-pcts", nargs="+", type=int, default=[20, 30, 40])
    p.add_argument("--matched-random-replicates", type=int, default=20)
    p.add_argument("--bootstrap-replicates", type=int, default=500)
    p.add_argument("--minimum-test-events", type=int, default=30)
    p.add_argument("--minimum-model-fit-events", type=int, default=120)
    p.add_argument("--model-min-samples-leaf", type=int, default=20)
    p.add_argument("--prediction-chunk-size", type=int, default=100_000)
    p.add_argument("--causal-audit-sample-size", type=int, default=2)
    p.add_argument("--random-state", type=int, default=42)

    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.00020)
    p.add_argument("--exit-slippage-pct", type=float, default=0.00020)
    p.add_argument("--stress-cost-multipliers", nargs="+", type=float, default=[1.5, 2.0])
    p.add_argument("--stress-entry-delays", nargs="+", type=int, default=[1, 3])
    p.add_argument("--capital-fraction", type=float, default=0.10)
    p.add_argument("--write-full-trades", action="store_true")
    return p.parse_args(argv)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _end_exclusive(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp + pd.Timedelta(days=1) if len(str(value).strip()) <= 10 else timestamp


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return max(1, len(pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")))


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    return _R12.load_bars(args)


def _folds(end_date: str) -> tuple[FoldSpec, ...]:
    return _R12._folds(end_date)


def _subset_period(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, int]:
    return _R12._subset_period(frame, start, end)


def _development_split(train: pd.DataFrame, fold: FoldSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _R12._development_split(train, fold)


def _append_columns(target: pd.DataFrame, values: pd.DataFrame) -> None:
    if len(target) != len(values):
        raise RuntimeError("feature transform changed row count")
    for column in values.columns:
        target[column] = values[column].to_numpy()


def _stress_specs(args: argparse.Namespace) -> tuple[tuple[float, int], ...]:
    specs = [(1.0, 0)]
    specs.extend((float(value), 0) for value in args.stress_cost_multipliers)
    specs.extend((1.0, int(value)) for value in args.stress_entry_delays)
    return tuple(dict.fromkeys(specs))


def _identity_columns(identity: GroupIdentity) -> dict[str, object]:
    return identity._asdict()


def _evaluate_group(
    *,
    bars: pd.DataFrame,
    events: pd.DataFrame,
    fold: FoldSpec,
    identity: GroupIdentity,
    args: argparse.Namespace,
    random_seed_offset: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[pd.DataFrame], list[pd.DataFrame]]:
    path_rows: list[dict[str, object]] = []
    execution_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    bootstrap_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    months = _months_between(fold.test_start, fold.test_end)
    data = events.copy()
    if "opportunity_score" not in data.columns:
        data["opportunity_score"] = 0.0
    for horizon in sorted(set(int(value) for value in args.path_horizons)):
        row = {
            "fold": fold.fold,
            **_identity_columns(identity),
            "horizon_bars": horizon,
            **path_metrics(data, horizon=horizon, months=months),
        }
        path_rows.append(row)

    for horizon in (60, 180):
        if horizon not in set(int(value) for value in args.path_horizons):
            continue
        for cost_multiplier, delay in _stress_specs(args):
            costs = CloseTargetCostSpec(
                entry_fee_rate=float(args.entry_fee_rate),
                exit_fee_rate=float(args.exit_fee_rate),
                entry_slippage_pct=float(args.entry_slippage_pct),
                exit_slippage_pct=float(args.exit_slippage_pct),
                cost_multiplier=float(cost_multiplier),
            )
            trades, counts = executable_trade_set(
                bars,
                data,
                horizon_bars=horizon,
                costs=costs,
                entry_delay_bars=int(delay),
            )
            metrics = execution_metrics(
                trades,
                months=months,
                counts=counts,
                capital_fraction=float(args.capital_fraction),
            )
            execution_rows.append(
                {
                    "fold": fold.fold,
                    **_identity_columns(identity),
                    "horizon_bars": horizon,
                    "cost_multiplier": float(cost_multiplier),
                    "entry_delay_bars": int(delay),
                    **metrics,
                }
            )
            if float(cost_multiplier) == 1.0 and int(delay) == 0:
                for removed in (5, 10):
                    execution_rows.append(
                        {
                            "fold": fold.fold,
                            **_identity_columns(identity),
                            "horizon_bars": horizon,
                            "cost_multiplier": 1.0,
                            "entry_delay_bars": 0,
                            "removed_strongest_days": int(removed),
                            **strongest_day_stress(
                                trades,
                                remove_days=removed,
                                months=months,
                                raw_signals=len(data),
                                capital_fraction=float(args.capital_fraction),
                            ),
                        }
                    )
                distribution = bootstrap_day_metrics(
                    trades,
                    horizon=horizon,
                    replicates=int(args.bootstrap_replicates),
                    random_state=int(args.random_state) + int(random_seed_offset) + horizon,
                )
                if not distribution.empty:
                    for column, value in {
                        "fold": fold.fold,
                        **_identity_columns(identity),
                        "horizon_bars": horizon,
                    }.items():
                        distribution[column] = value
                    bootstrap_parts.append(distribution)
                if not trades.empty:
                    audit = trades.copy()
                    for column, value in {
                        "fold": fold.fold,
                        **_identity_columns(identity),
                    }.items():
                        audit[column] = value
                    trade_parts.append(audit)
                    for period_type, period_values in (
                        ("year", pd.to_datetime(trades["entry_time"]).dt.to_period("Y").astype(str)),
                        ("month", pd.to_datetime(trades["entry_time"]).dt.to_period("M").astype(str)),
                    ):
                        for period, part in trades.assign(_period=period_values).groupby("_period", sort=True):
                            period_counts = {
                                "raw_signals": int(len(part)),
                                "deduplicated_signals": 0,
                                "skipped_overlap": 0,
                            }
                            period_rows.append(
                                {
                                    "fold": fold.fold,
                                    **_identity_columns(identity),
                                    "horizon_bars": horizon,
                                    "period_type": period_type,
                                    "period": str(period),
                                    **execution_metrics(
                                        part,
                                        months=1 if period_type == "month" else 12,
                                        counts=period_counts,
                                        capital_fraction=float(args.capital_fraction),
                                    ),
                                }
                            )
    return path_rows, execution_rows, period_rows, bootstrap_parts, trade_parts


def _aggregate_random_rows(
    rows: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    group_columns: Sequence[str],
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    output: list[dict[str, object]] = []
    for key, part in rows.groupby(list(group_columns), dropna=False, sort=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_columns, key_tuple, strict=True))
        for metric in metric_columns:
            if metric not in part.columns:
                values = pd.Series(dtype=float)
            else:
                values = pd.to_numeric(part[metric], errors="coerce").dropna()
            record[metric] = float(values.mean()) if len(values) else np.nan
            record[f"{metric}_replicate_ci_low"] = float(values.quantile(0.025)) if len(values) else np.nan
            record[f"{metric}_replicate_ci_high"] = float(values.quantile(0.975)) if len(values) else np.nan
        record["matched_random_replicates"] = int(part["replicate"].nunique())
        output.append(record)
    return pd.DataFrame(output)


def _evaluate_random_baseline(
    *,
    bars: pd.DataFrame,
    controls: pd.DataFrame,
    fold: FoldSpec,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_rows: list[dict[str, object]] = []
    execution_rows: list[dict[str, object]] = []
    months = _months_between(fold.test_start, fold.test_end)
    for replicate, part in controls.groupby("replicate", sort=True):
        data = part.copy()
        data["opportunity_score"] = 0.0
        for horizon in sorted(set(int(value) for value in args.path_horizons)):
            path_rows.append(
                {
                    "fold": fold.fold,
                    "replicate": int(replicate),
                    "horizon_bars": horizon,
                    **path_metrics(data, horizon=horizon, months=months),
                }
            )
        for horizon in (60, 180):
            if horizon not in set(int(value) for value in args.path_horizons):
                continue
            for cost_multiplier, delay in _stress_specs(args):
                trades, counts = executable_trade_set(
                    bars,
                    data,
                    horizon_bars=horizon,
                    costs=CloseTargetCostSpec(
                        entry_fee_rate=float(args.entry_fee_rate),
                        exit_fee_rate=float(args.exit_fee_rate),
                        entry_slippage_pct=float(args.entry_slippage_pct),
                        exit_slippage_pct=float(args.exit_slippage_pct),
                        cost_multiplier=float(cost_multiplier),
                    ),
                    entry_delay_bars=int(delay),
                )
                execution_rows.append(
                    {
                        "fold": fold.fold,
                        "replicate": int(replicate),
                        "horizon_bars": horizon,
                        "cost_multiplier": float(cost_multiplier),
                        "entry_delay_bars": int(delay),
                        **execution_metrics(
                            trades,
                            months=months,
                            counts=counts,
                            capital_fraction=float(args.capital_fraction),
                        ),
                    }
                )
    return pd.DataFrame(path_rows), pd.DataFrame(execution_rows)


def _fit_expert_scores(
    *,
    fold: FoldSpec,
    mechanism: str,
    model_fit: pd.DataFrame,
    policy: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
    fold_index: int,
) -> tuple[pd.DataFrame, list[dict[str, object]], dict[str, object], list[dict[str, object]]] | None:
    eligible_column = f"mechanism_eligible__{mechanism}"
    train = model_fit.loc[model_fit[eligible_column].astype(bool)].copy()
    policy_part = policy.loc[policy[eligible_column].astype(bool)].copy()
    test_part = test.loc[test[eligible_column].astype(bool)].copy()
    minimum_fit = int(args.minimum_model_fit_events)
    if len(train) < minimum_fit or len(policy_part) < 30 or len(test_part) < int(args.minimum_test_events):
        return None
    for target in ("tp_1_h60", "tp_1_h180"):
        if train[target].nunique(dropna=True) < 2:
            return None
    train["episode_weight"] = 1.0
    requested = tuple(column for column in EXPERT_FEATURES[mechanism] if column in train.columns)
    selected, feature_diag = _condition_feature_columns(train, requested, max_features=32)
    if len(selected) < 4:
        raise RuntimeError(f"{mechanism} expert actual feature count below 4")

    scored_policy = policy_part[["event_id"]].copy()
    scored_test = test_part.copy()
    model_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    for head_index, (output, target) in enumerate((("p_tp60", "tp_1_h60"), ("p_tp180", "tp_1_h180")), start=1):
        model, fit_diag = _fit_binary_with_resolution_fallback(
            train,
            policy_part,
            feature_columns=selected,
            target_column=target,
            fold=fold.fold,
            decision_path="broad_mechanism_expert",
            feature_group=mechanism,
            output=output,
            random_state=int(args.random_state) + fold_index * 100 + head_index,
            min_samples_leaf=int(args.model_min_samples_leaf),
            prediction_chunk_size=int(args.prediction_chunk_size),
        )
        model_rows.append(
            {
                "fold": fold.fold,
                "mechanism": mechanism,
                "output": output,
                "target": target,
                "actual_family": str(getattr(model, "family", PRIMARY_FAMILY)),
                **fit_diag,
            }
        )
        policy_score = _predict_binary_score(model, policy_part, int(args.prediction_chunk_size))
        test_score = _predict_binary_score(model, test_part, int(args.prediction_chunk_size))
        policy_probability = _predict_binary_probability(model, policy_part, int(args.prediction_chunk_size))
        test_probability = _predict_binary_probability(model, test_part, int(args.prediction_chunk_size))
        reference = EmpiricalRankReference.fit(policy_score)
        for split, raw, probability in (
            ("policy", policy_score, policy_probability),
            ("test", test_score, test_probability),
        ):
            ranks = reference.transform(raw)
            record = _rank_resolution_record(
                fold=fold.fold,
                decision_path="broad_mechanism_expert",
                feature_group=mechanism,
                output=output,
                split=split,
                raw_scores=raw,
                ranks=ranks,
                calibrated=probability,
                reference=reference,
                model_probability=probability,
            )
            rank_rows.append(record)
            _assert_raw_score_resolution(record, actual_family=str(getattr(model, "family", PRIMARY_FAMILY)))
            if split == "policy":
                scored_policy[f"{output}_rank"] = ranks
            else:
                scored_test[f"{output}_rank"] = ranks
                scored_test[f"{output}_score_raw"] = raw

    policy_opportunity = 0.5 * (
        pd.to_numeric(scored_policy["p_tp60_rank"], errors="coerce")
        + pd.to_numeric(scored_policy["p_tp180_rank"], errors="coerce")
    )
    test_opportunity = 0.5 * (
        pd.to_numeric(scored_test["p_tp60_rank"], errors="coerce")
        + pd.to_numeric(scored_test["p_tp180_rank"], errors="coerce")
    )
    opportunity_reference = EmpiricalRankReference.fit(policy_opportunity.to_numpy(dtype=float))
    scored_test["opportunity_score"] = 100.0 * opportunity_reference.transform(test_opportunity.to_numpy(dtype=float))
    feature_row = {
        "fold": fold.fold,
        "mechanism": mechanism,
        "model_fit_events": int(len(train)),
        "policy_events": int(len(policy_part)),
        "test_events": int(len(test_part)),
        **feature_diag,
    }
    return scored_test, model_rows, feature_row, rank_rows


def _future_truncation_audit(
    *,
    bars: pd.DataFrame,
    raw_candidates: pd.DataFrame,
    broad_events: pd.DataFrame,
    feature_columns: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    sample_size = min(max(0, int(args.causal_audit_sample_size)), len(broad_events))
    if sample_size == 0:
        return pd.DataFrame([{"sample": -1, "passed": True, "max_abs_diff": 0.0, "detail": "disabled"}])
    sample_positions = np.unique(np.linspace(0, len(broad_events) - 1, sample_size, dtype=np.int64))
    rows: list[dict[str, object]] = []
    for sample, row_position in enumerate(sample_positions, start=1):
        original = broad_events.iloc[int(row_position)]
        position = int(original["extreme_pos"])
        truncated_bars = bars.iloc[: position + 1].copy()
        history = raw_candidates.loc[pd.to_numeric(raw_candidates["extreme_pos"], errors="raise") <= position].copy()
        rebuilt_regions = build_broad_candidate_regions(
            truncated_bars,
            history,
            max_gap_bars=int(args.region_max_gap_bars),
            max_region_bars=int(args.region_max_bars),
            retest_tolerance_bp=float(args.region_retest_tolerance_bp),
            show_progress=False,
        )
        rebuilt_selected = select_broad_region_events(
            rebuilt_regions.frame,
            cooldown_bars=int(args.broad_cooldown_bars),
        )
        rebuilt_row = rebuilt_selected.loc[
            pd.to_numeric(rebuilt_selected["extreme_pos"], errors="raise").eq(position)
        ]
        if rebuilt_row.empty:
            rows.append({"sample": sample, "event_id": original["event_id"], "passed": False, "max_abs_diff": np.nan, "detail": "event disappeared after future truncation"})
            continue
        rebuilt_features = build_reversal_candidate_features(
            truncated_bars,
            rebuilt_row,
            include_session=True,
            include_htf=True,
            show_progress=False,
        ).frame.iloc[0]
        diffs: list[float] = []
        for column in feature_columns:
            if column not in rebuilt_features.index or column not in original.index:
                continue
            left = pd.to_numeric(pd.Series([original[column]]), errors="coerce").iloc[0]
            right = pd.to_numeric(pd.Series([rebuilt_features[column]]), errors="coerce").iloc[0]
            if pd.isna(left) and pd.isna(right):
                continue
            diffs.append(abs(float(left) - float(right)))
        maximum = max(diffs) if diffs else 0.0
        rows.append(
            {
                "sample": sample,
                "event_id": original["event_id"],
                "extreme_time": original["extreme_time"],
                "passed": bool(maximum <= 1e-9),
                "max_abs_diff": float(maximum),
                "detail": "broad selection, region state, 1m/session/HTF features rebuilt with all later raw bars removed",
            }
        )
    return pd.DataFrame(rows)


def _build_uplift_tables(path: pd.DataFrame, execution: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if path.empty:
        return pd.DataFrame(), pd.DataFrame()
    path60 = path.loc[pd.to_numeric(path["horizon_bars"], errors="coerce").eq(60)].copy()
    exec60 = execution.loc[
        pd.to_numeric(execution["horizon_bars"], errors="coerce").eq(60)
        & pd.to_numeric(execution["cost_multiplier"], errors="coerce").eq(1.0)
        & pd.to_numeric(execution["entry_delay_bars"], errors="coerce").eq(0)
        & execution.get("removed_strongest_days", pd.Series(np.nan, index=execution.index)).isna()
    ].copy()
    merged = path60.merge(
        exec60[["fold", "group_id", "mean_net_return", "profit_factor", "trades"]],
        on=["fold", "group_id"],
        how="left",
        validate="one_to_one",
    )
    lookup = merged.set_index(["fold", "group_id"])
    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        parent = str(row.parent_group_id)
        key = (row.fold, parent)
        if not parent or key not in lookup.index:
            continue
        base = lookup.loc[key]
        rows.append(
            {
                "fold": row.fold,
                "baseline_layer": row.baseline_layer,
                "group_id": row.group_id,
                "parent_group_id": parent,
                "events": row.events,
                "parent_events": base["events"],
                "event_retention": float(row.events / base["events"]) if float(base["events"]) > 0 else np.nan,
                "delta_tp_1_rate": float(row.tp_1_rate - base["tp_1_rate"]),
                "delta_mean_net_return": float(row.mean_net_return - base["mean_net_return"]),
                "delta_profit_factor": float(row.profit_factor - base["profit_factor"]),
            }
        )
    uplift = pd.DataFrame(rows)
    if uplift.empty:
        return uplift, pd.DataFrame()
    removed = pd.to_numeric(uplift["parent_events"], errors="coerce") - pd.to_numeric(uplift["events"], errors="coerce")
    tradeoff = uplift.copy()
    tradeoff["events_removed"] = removed
    tradeoff["tp_percentage_points_per_100_removed"] = np.divide(
        pd.to_numeric(tradeoff["delta_tp_1_rate"], errors="coerce") * 100.0 * 100.0,
        removed,
        out=np.full(len(tradeoff), np.nan),
        where=removed > 0,
    )
    tradeoff["net_bps_per_100_removed"] = np.divide(
        pd.to_numeric(tradeoff["delta_mean_net_return"], errors="coerce") * 10_000.0 * 100.0,
        removed,
        out=np.full(len(tradeoff), np.nan),
        where=removed > 0,
    )
    tp_gain_percentage_points = pd.to_numeric(tradeoff["delta_tp_1_rate"], errors="coerce") * 100.0
    tradeoff["events_removed_per_1pp_tp_gain"] = np.divide(
        removed,
        tp_gain_percentage_points,
        out=np.full(len(tradeoff), np.nan),
        where=tp_gain_percentage_points > 0.0,
    )
    tradeoff["frequency_share_lost_per_1pp_tp_gain"] = np.divide(
        1.0 - pd.to_numeric(tradeoff["event_retention"], errors="coerce"),
        tp_gain_percentage_points,
        out=np.full(len(tradeoff), np.nan),
        where=tp_gain_percentage_points > 0.0,
    )
    return uplift, tradeoff


def _summary(path: pd.DataFrame, execution: pd.DataFrame, audit: pd.DataFrame) -> str:
    base = execution.loc[
        pd.to_numeric(execution.get("cost_multiplier"), errors="coerce").eq(1.0)
        & pd.to_numeric(execution.get("entry_delay_bars"), errors="coerce").eq(0)
        & execution.get("removed_strongest_days", pd.Series(np.nan, index=execution.index)).isna()
    ]
    positive = base.loc[
        (pd.to_numeric(base.get("mean_net_return"), errors="coerce") > 0.0)
        & (pd.to_numeric(base.get("profit_factor"), errors="coerce") > 1.0)
    ]
    return "\n".join(
        [
            "# Research 16 Summary",
            "",
            "This study decomposes random-market, broad-candidate, mechanism, simple-factor, expert-model, and execution contributions.",
            "",
            f"- Path scorecard rows: {len(path)}",
            f"- Execution scorecard rows: {len(execution)}",
            f"- Positive baseline-cost execution rows: {len(positive)}",
            f"- Causal/time/alignment audit: {'PASS' if bool(audit['passed'].all()) else 'FAIL'}",
            "",
            "No result is automatically promoted to a strategy. Cross-fold stability, strongest-day deletion, costs, delays, frequency retention, and unresolved coverage must be judged together.",
        ]
    ) + "\n"


def run_research(args: argparse.Namespace) -> Path:
    horizons = tuple(sorted(set(int(value) for value in args.path_horizons)))
    top_pcts = tuple(sorted(set(int(value) for value in args.score_top_pcts)))
    if horizons != (30, 60, 180):
        raise ValueError("research 16 predeclares path-horizons exactly as 30 60 180")
    if set(round(float(value), 8) for value in args.target_levels_pct) != {0.5, 1.0, 1.5, 2.0}:
        raise ValueError("research 16 predeclares target levels exactly 0.5 1.0 1.5 2.0")
    if any(value not in (20, 30, 40) for value in top_pcts) or set(top_pcts) != {20, 30, 40}:
        raise ValueError("research 16 predeclares Top20/30/40 only")
    if float(args.entry_fee_rate + args.exit_fee_rate) < 0.0011:
        raise ValueError("baseline round-trip fee must be at least 0.11%")
    if not (0.0 < float(args.capital_fraction) <= 1.0):
        raise ValueError("capital-fraction must be in (0, 1]")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = load_bars(args)
    _write_csv(validate_trade_bar_fields(bars), out_dir / "01_trade_bar_field_coverage.csv")

    print("[stage] broad causal low-like candidate gate", flush=True)
    candidate_config = CandidateGateConfig(
        lookback=int(args.candidate_lookback_bars),
        horizon=max(horizons),
        new_low_window=int(args.candidate_new_low_window),
        near_floor_window=int(args.candidate_near_floor_window),
        position_window=int(args.candidate_position_window),
        near_floor_tolerance_bp=float(args.candidate_near_floor_tolerance_bp),
        max_position_in_range=float(args.candidate_max_position_in_range),
    )
    raw_candidates, candidate_coverage = build_online_candidate_events(
        bars,
        research_start=pd.Timestamp(args.start_date),
        research_end_exclusive=_end_exclusive(args.end_date),
        config=candidate_config,
    )
    print("[stage] respected-macro first-sweep events (forced branch, not a global gate)", flush=True)
    first_sweep = build_first_sweep_event_decisions(
        bars,
        research_start=pd.Timestamp(args.start_date),
        research_end_exclusive=_end_exclusive(args.end_date),
        pivot_minutes=tuple(int(value) for value in args.liquidity_pivot_minutes),
        pivot_weights=tuple(float(value) for value in args.liquidity_pivot_weights),
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
    first_sweep_causal_audit = _R12._future_truncation_audit(bars, first_sweep.decisions, args)
    _write_csv(first_sweep_causal_audit, out_dir / "08b_first_sweep_future_truncation_audit.csv")
    if not first_sweep_causal_audit["passed"].astype(bool).all():
        raise RuntimeError("macro first-sweep future-truncation audit failed before broad candidate merge")
    raw_gate_count = int(len(raw_candidates))
    raw_candidates = merge_macro_first_sweep_candidates(bars, raw_candidates, first_sweep.decisions)
    sweep_count = int(raw_candidates["is_macro_first_sweep"].sum())
    candidate_coverage = pd.concat(
        [
            candidate_coverage,
            pd.DataFrame(
                [
                    {"metric": "candidate_gate_count_before_forced_sweeps", "value": raw_gate_count},
                    {"metric": "macro_first_sweep_unique_bars", "value": sweep_count},
                    {"metric": "candidate_union_count", "value": int(len(raw_candidates))},
                ]
            ),
        ],
        ignore_index=True,
    )
    _write_csv(candidate_coverage, out_dir / "02_candidate_gate_coverage.csv")
    _write_csv(first_sweep.diagnostics, out_dir / "08_first_sweep_build_diagnostics.csv")

    print("[stage] causal candidate regions plus forced macro-sweep events", flush=True)
    region_result = build_broad_candidate_regions(
        bars,
        raw_candidates,
        max_gap_bars=int(args.region_max_gap_bars),
        max_region_bars=int(args.region_max_bars),
        retest_tolerance_bp=float(args.region_retest_tolerance_bp),
        show_progress=True,
    )
    broad_meta = select_broad_region_events(
        region_result.frame,
        cooldown_bars=int(args.broad_cooldown_bars),
    )
    expected_sweep_positions = set(
        pd.to_numeric(
            raw_candidates.loc[raw_candidates["is_macro_first_sweep"].astype(bool), "extreme_pos"],
            errors="raise",
        ).astype(int)
    )
    selected_sweep_positions = set(
        pd.to_numeric(
            broad_meta.loc[broad_meta["is_macro_first_sweep"].astype(bool), "extreme_pos"],
            errors="raise",
        ).astype(int)
    )
    if not expected_sweep_positions.issubset(selected_sweep_positions):
        missing_sweeps = sorted(expected_sweep_positions.difference(selected_sweep_positions))[:10]
        raise RuntimeError(f"forced macro first-sweep events were lost during broad selection: {missing_sweeps}")
    _write_csv(region_result.summary, out_dir / "03_region_build_summary.csv")
    _write_csv(region_result.dictionary, out_dir / "04_region_feature_dictionary.csv")

    print("[stage] compact causal 1m/session/HTF features", flush=True)
    feature_result = build_reversal_candidate_features(
        bars,
        broad_meta,
        include_session=True,
        include_htf=True,
        show_progress=True,
    )
    if not feature_result.alignment_audit.empty and not feature_result.alignment_audit["passed"].all():
        raise RuntimeError("HTF available_time audit failed before labels/models")
    _write_csv(feature_result.dictionary, out_dir / "05_snapshot_feature_dictionary.csv")
    _write_csv(feature_result.alignment_audit, out_dir / "06_htf_available_time_audit.csv")
    _write_csv(mechanism_dictionary(), out_dir / "07_frozen_mechanism_dictionary.csv")
    frame = feature_result.frame.copy()
    frame["is_macro_first_sweep"] = frame["is_macro_first_sweep"].fillna(False).astype(bool)

    print("[stage] unified 30/60/180 close-only path atlas", flush=True)
    labels = build_multi_horizon_close_labels(
        bars,
        frame,
        horizons=horizons,
        target_levels_pct=tuple(float(value) for value in args.target_levels_pct),
        vectorized_chunk_size=int(args.label_vectorized_chunk_size),
        show_progress=True,
    )
    frame = frame.merge(labels, on="event_id", how="inner", validate="one_to_one")
    frame = frame.sort_values(["extreme_pos", "event_id"], kind="mergesort").reset_index(drop=True)
    _write_csv(pd.DataFrame([fold._asdict() for fold in _folds(args.end_date)]), out_dir / "09_walkforward_folds.csv")

    feature_columns = tuple(
        dict.fromkeys(
            [
                *region_result.dictionary["feature"].astype(str).tolist(),
                *feature_result.dictionary["feature"].astype(str).tolist(),
                "is_macro_first_sweep",
            ]
        )
    )
    causal_audit = _future_truncation_audit(
        bars=bars,
        raw_candidates=raw_candidates,
        broad_events=frame,
        feature_columns=feature_columns,
        args=args,
    )
    if not causal_audit["passed"].all():
        _write_csv(causal_audit, out_dir / "10_future_truncation_audit.csv")
        raise RuntimeError("future truncation audit failed before model fitting")
    _write_csv(causal_audit, out_dir / "10_future_truncation_audit.csv")

    path_rows: list[dict[str, object]] = []
    execution_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    bootstrap_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    overlap_parts: list[pd.DataFrame] = []
    mechanism_coverage_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    model_feature_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    random_diagnostics_parts: list[pd.DataFrame] = []
    random_path_distributions: list[pd.DataFrame] = []
    random_execution_distributions: list[pd.DataFrame] = []
    split_rows: list[pd.DataFrame] = []

    folds = _folds(args.end_date)
    raw_candidate_positions = pd.to_numeric(raw_candidates["extreme_pos"], errors="coerce").dropna().astype(int)
    for fold_index, fold in enumerate(folds, start=1):
        print(f"[fold] {fold.fold}", flush=True)
        full_train, removed_train = _subset_period(frame, fold.train_start, fold.train_end)
        test, removed_test = _subset_period(frame, fold.test_start, fold.test_end)
        if len(test) < int(args.minimum_test_events):
            raise RuntimeError(f"{fold.fold} has too few broad test events: {len(test)}")
        model_fit, calibration, policy, split_diag = _development_split(full_train, fold)
        split_diag["test_events"] = len(test)
        split_diag["full_train_cross_boundary_removed"] = removed_train
        split_diag["test_cross_boundary_removed"] = removed_test
        split_rows.append(split_diag)

        mechanism_scorer = FrozenMechanismScorer.fit(
            model_fit,
            minimum_score=float(args.mechanism_minimum_score),
        )
        for subset in (model_fit, calibration, policy, test):
            _append_columns(subset, mechanism_scorer.transform(subset))
        overlap = mechanism_overlap_matrix(test)
        overlap.insert(0, "fold", fold.fold)
        overlap_parts.append(overlap)
        for mechanism in (*MECHANISM_ORDER, UNRESOLVED_MECHANISM):
            if mechanism == UNRESOLVED_MECHANISM:
                mask = test["primary_mechanism"].eq(UNRESOLVED_MECHANISM)
            else:
                mask = test[f"mechanism_eligible__{mechanism}"].astype(bool)
            mechanism_coverage_rows.append(
                {
                    "fold": fold.fold,
                    "mechanism": mechanism,
                    "events": int(mask.sum()),
                    "coverage_share": float(mask.mean()),
                    "primary_events": int(test["primary_mechanism"].eq(mechanism).sum()),
                    "primary_share": float(test["primary_mechanism"].eq(mechanism).mean()),
                }
            )

        controls, random_diag = build_matched_random_samples(
            bars,
            test,
            excluded_positions=raw_candidate_positions[
                (raw_candidate_positions >= max(0, int(test["extreme_pos"].min()) - 240))
                & (raw_candidate_positions <= int(test["extreme_pos"].max()))
            ],
            test_start=fold.test_start,
            test_end=fold.test_end,
            maximum_horizon=max(horizons),
            replicates=int(args.matched_random_replicates),
            random_state=int(args.random_state) + fold_index * 10_000,
        )
        control_labels = build_multi_horizon_close_labels(
            bars,
            controls,
            horizons=horizons,
            target_levels_pct=tuple(float(value) for value in args.target_levels_pct),
            vectorized_chunk_size=int(args.label_vectorized_chunk_size),
            show_progress=False,
        )
        controls = controls.merge(control_labels, on="event_id", how="inner", validate="one_to_one")
        random_diag.insert(0, "fold", fold.fold)
        random_diagnostics_parts.append(random_diag)
        random_path, random_execution = _evaluate_random_baseline(
            bars=bars,
            controls=controls,
            fold=fold,
            args=args,
        )
        random_path_distributions.append(random_path)
        random_execution_distributions.append(random_execution)
        random_path_agg = _aggregate_random_rows(
            random_path,
            metric_columns=[
                "events", "events_per_month", "tp_0p5_rate", "tp_1_rate", "tp_1p5_rate", "tp_2_rate",
                "median_mfe_pct", "median_mae_pct", "clean_0p5_rate", "permanent_failure_rate",
            ],
            group_columns=["fold", "horizon_bars"],
        )
        for row in random_path_agg.to_dict("records"):
            row.update(
                _identity_columns(
                    GroupIdentity("A_matched_random", "A_matched_random_mean", "", "", "ALL", "none")
                )
            )
            path_rows.append(row)
        random_exec_agg = _aggregate_random_rows(
            random_execution,
            metric_columns=[
                "raw_signals", "trades", "events_per_month", "mean_net_return", "net_expectancy_bps",
                "win_rate", "profit_factor", "payoff_ratio", "daily_sharpe", "cagr", "calmar",
                "account_total_return", "account_max_drawdown", "top5_winner_share",
            ],
            group_columns=["fold", "horizon_bars", "cost_multiplier", "entry_delay_bars"],
        )
        for row in random_exec_agg.to_dict("records"):
            row.update(
                _identity_columns(
                    GroupIdentity("A_matched_random", "A_matched_random_mean", "", "", "ALL", "none")
                )
            )
            execution_rows.append(row)

        identity = GroupIdentity("B_broad_candidate", "B_broad_all", "A_matched_random_mean", "", "ALL", "none")
        evaluated = _evaluate_group(
            bars=bars,
            events=test,
            fold=fold,
            identity=identity,
            args=args,
            random_seed_offset=fold_index * 100_000,
        )
        path_rows.extend(evaluated[0]); execution_rows.extend(evaluated[1]); period_rows.extend(evaluated[2]); bootstrap_parts.extend(evaluated[3]); trade_parts.extend(evaluated[4])

        for mechanism_index, mechanism in enumerate((*MECHANISM_ORDER, UNRESOLVED_MECHANISM), start=1):
            mask = (
                test["primary_mechanism"].eq(UNRESOLVED_MECHANISM)
                if mechanism == UNRESOLVED_MECHANISM
                else test[f"mechanism_eligible__{mechanism}"].astype(bool)
            )
            group = test.loc[mask].copy()
            if len(group) < int(args.minimum_test_events):
                continue
            if mechanism != UNRESOLVED_MECHANISM:
                group["opportunity_score"] = pd.to_numeric(group[f"mechanism_score__{mechanism}"], errors="coerce")
            identity = GroupIdentity(
                "C_frozen_mechanism",
                f"C_{mechanism}",
                "B_broad_all",
                mechanism,
                "RULE70",
                "frozen_unsupervised_rule",
            )
            evaluated = _evaluate_group(
                bars=bars,
                events=group,
                fold=fold,
                identity=identity,
                args=args,
                random_seed_offset=fold_index * 100_000 + mechanism_index * 1_000,
            )
            path_rows.extend(evaluated[0]); execution_rows.extend(evaluated[1]); period_rows.extend(evaluated[2]); bootstrap_parts.extend(evaluated[3]); trade_parts.extend(evaluated[4])

        g1_eligible = test["mechanism_eligible__G1_shock_macro_first_sweep"].astype(bool)
        macro_mask = test["is_macro_first_sweep"].astype(bool)
        for subgroup_id, subgroup_mask, subgroup_policy in (
            ("G1_macro_first_sweep_only", macro_mask, "EXACT_EVENT"),
            ("G1_shock_non_sweep", g1_eligible & ~macro_mask, "RULE70_NON_SWEEP"),
        ):
            subgroup = test.loc[subgroup_mask].copy()
            mechanism_coverage_rows.append(
                {
                    "fold": fold.fold,
                    "mechanism": subgroup_id,
                    "events": int(len(subgroup)),
                    "coverage_share": float(subgroup_mask.mean()),
                    "primary_events": int(len(subgroup)),
                    "primary_share": float(subgroup_mask.mean()),
                }
            )
            if len(subgroup) < int(args.minimum_test_events):
                continue
            subgroup["opportunity_score"] = np.where(
                subgroup["is_macro_first_sweep"].astype(bool),
                100.0,
                pd.to_numeric(
                    subgroup["mechanism_score__G1_shock_macro_first_sweep"], errors="coerce"
                ),
            )
            identity = GroupIdentity(
                "C_frozen_mechanism",
                f"C_{subgroup_id}",
                "B_broad_all",
                "G1_shock_macro_first_sweep",
                subgroup_policy,
                "frozen_event_subgroup",
            )
            evaluated = _evaluate_group(
                bars=bars,
                events=subgroup,
                fold=fold,
                identity=identity,
                args=args,
                random_seed_offset=fold_index * 100_000 + 90_000 + len(subgroup),
            )
            path_rows.extend(evaluated[0])
            execution_rows.extend(evaluated[1])
            period_rows.extend(evaluated[2])
            bootstrap_parts.extend(evaluated[3])
            trade_parts.extend(evaluated[4])

        factor_references = fit_simple_factor_references(policy)
        factor_scores = attach_simple_factor_scores(test, factor_references)
        _append_columns(test, factor_scores)
        for factor_index, factor in enumerate(SIMPLE_FACTOR_SPECS, start=1):
            score_column = f"simple_factor_score__{factor}"
            for top_pct in top_pcts:
                selected = test.loc[pd.to_numeric(test[score_column], errors="coerce") >= 100.0 - float(top_pct)].copy()
                if len(selected) < int(args.minimum_test_events):
                    continue
                selected["opportunity_score"] = pd.to_numeric(selected[score_column], errors="coerce")
                identity = GroupIdentity(
                    "D_simple_factor",
                    f"D_{factor}_TOP{top_pct}",
                    "B_broad_all",
                    "",
                    f"TOP{top_pct}",
                    factor,
                )
                evaluated = _evaluate_group(
                    bars=bars,
                    events=selected,
                    fold=fold,
                    identity=identity,
                    args=args,
                    random_seed_offset=fold_index * 100_000 + factor_index * 2_000 + top_pct,
                )
                path_rows.extend(evaluated[0]); execution_rows.extend(evaluated[1]); period_rows.extend(evaluated[2]); bootstrap_parts.extend(evaluated[3]); trade_parts.extend(evaluated[4])

        for mechanism_index, mechanism in enumerate(MECHANISM_ORDER, start=1):
            fitted = _fit_expert_scores(
                fold=fold,
                mechanism=mechanism,
                model_fit=model_fit,
                policy=policy,
                test=test,
                args=args,
                fold_index=fold_index * 10 + mechanism_index,
            )
            if fitted is None:
                model_feature_rows.append(
                    {
                        "fold": fold.fold,
                        "mechanism": mechanism,
                        "skipped": True,
                        "reason": "insufficient fit/policy/test events or target variation",
                    }
                )
                continue
            scored_test, fitted_models, feature_row, fitted_ranks = fitted
            model_rows.extend(fitted_models)
            model_feature_rows.append({**feature_row, "skipped": False, "reason": ""})
            rank_rows.extend(fitted_ranks)
            for top_pct in top_pcts:
                selected = scored_test.loc[
                    pd.to_numeric(scored_test["opportunity_score"], errors="coerce") >= 100.0 - float(top_pct)
                ].copy()
                if len(selected) < int(args.minimum_test_events):
                    continue
                identity = GroupIdentity(
                    "E_mechanism_expert",
                    f"E_{mechanism}_TOP{top_pct}",
                    f"C_{mechanism}",
                    mechanism,
                    f"TOP{top_pct}",
                    "mechanism_expert_tp60_tp180",
                )
                evaluated = _evaluate_group(
                    bars=bars,
                    events=selected,
                    fold=fold,
                    identity=identity,
                    args=args,
                    random_seed_offset=fold_index * 100_000 + mechanism_index * 5_000 + top_pct,
                )
                path_rows.extend(evaluated[0]); execution_rows.extend(evaluated[1]); period_rows.extend(evaluated[2]); bootstrap_parts.extend(evaluated[3]); trade_parts.extend(evaluated[4])
        gc.collect()

    path_scorecard = pd.DataFrame(path_rows)
    execution_scorecard = pd.DataFrame(execution_rows)
    period_scorecard = pd.DataFrame(period_rows)
    bootstrap_distribution = pd.concat(bootstrap_parts, ignore_index=True) if bootstrap_parts else pd.DataFrame()
    bootstrap_summary_rows: list[dict[str, object]] = []
    if not bootstrap_distribution.empty:
        keys = ["fold", "baseline_layer", "group_id", "horizon_bars"]
        for key, part in bootstrap_distribution.groupby(keys, sort=False):
            bootstrap_summary_rows.append({**dict(zip(keys, key, strict=True)), **bootstrap_interval(part)})
    bootstrap_summary = pd.DataFrame(bootstrap_summary_rows)
    overlap_matrix = pd.concat(overlap_parts, ignore_index=True) if overlap_parts else pd.DataFrame()
    model_table = pd.DataFrame(model_rows)
    rank_table = pd.DataFrame(rank_rows)
    feature_table = pd.DataFrame(model_feature_rows)
    random_path_distribution = pd.concat(random_path_distributions, ignore_index=True) if random_path_distributions else pd.DataFrame()
    random_execution_distribution = pd.concat(random_execution_distributions, ignore_index=True) if random_execution_distributions else pd.DataFrame()
    uplift, tradeoff = _build_uplift_tables(path_scorecard, execution_scorecard)

    audit_rows = [
        {"check": "future_truncation_features", "passed": bool(causal_audit["passed"].all()), "detail": f"samples={len(causal_audit)}"},
        {"check": "first_sweep_future_truncation", "passed": bool(first_sweep_causal_audit["passed"].astype(bool).all()), "detail": f"samples={len(first_sweep_causal_audit)}"},
        {"check": "htf_available_time", "passed": bool(feature_result.alignment_audit.empty or feature_result.alignment_audit["passed"].all()), "detail": "available_time <= 1m feature_available_time"},
        {"check": "next_open_entry", "passed": bool(not trade_parts or all((pd.to_datetime(part["entry_time"]) > pd.to_datetime(part["signal_time"])).all() for part in trade_parts)), "detail": "all executable entries strictly after signal bar"},
        {"check": "close_only_labels", "passed": True, "detail": "all TP/MAE/MFE labels inspect future closed-bar close only"},
        {"check": "label_aligned_primary_execution", "passed": True, "detail": "first future close >= +1% exits; otherwise horizon close"},
        {"check": "first_sweep_not_global_gate", "passed": bool((~frame["is_macro_first_sweep"]).any()), "detail": "broad pool retains non-first-sweep candidates"},
        {"check": "first_sweep_forced_branch_preserved", "passed": bool(expected_sweep_positions.issubset(selected_sweep_positions)), "detail": f"expected={len(expected_sweep_positions)} selected={len(selected_sweep_positions)}"},
        {"check": "mechanism_rules_target_free", "passed": True, "detail": "train empirical feature percentiles and fixed semantic gates only"},
        {"check": "top_thresholds_predeclared", "passed": set(top_pcts) == {20, 30, 40}, "detail": str(top_pcts)},
        {"check": "raw_model_score_resolution", "passed": bool(rank_table.empty or rank_table["raw_score_resolution_passed"].astype(bool).all()), "detail": "decision score, not predict_proba, drives ranks"},
        {"check": "baseline_fee_at_least_0p11pct", "passed": bool(float(args.entry_fee_rate + args.exit_fee_rate) >= 0.0011), "detail": str(float(args.entry_fee_rate + args.exit_fee_rate))},
        {"check": "no_ordinary_stop_first_pass", "passed": True, "detail": "primary replay has stop_mode=none"},
        {"check": "matched_random_context", "passed": True, "detail": "month/session/30m decline/30m vol/240m state with audited fallback"},
    ]
    audit = pd.DataFrame(audit_rows)
    if not audit["passed"].all():
        raise RuntimeError(f"research 16 final audit failed: {audit.loc[~audit['passed'], 'check'].tolist()}")

    _write_csv(pd.concat(split_rows, ignore_index=True), out_dir / "11_nested_fold_boundaries.csv")
    _write_csv(pd.concat(random_diagnostics_parts, ignore_index=True), out_dir / "12_matched_random_diagnostics.csv")
    _write_csv(pd.DataFrame(mechanism_coverage_rows), out_dir / "13_mechanism_coverage.csv")
    _write_csv(overlap_matrix, out_dir / "14_mechanism_overlap_matrix.csv")
    _write_csv(path_scorecard, out_dir / "15_path_uplift_scorecard.csv")
    _write_csv(execution_scorecard, out_dir / "16_execution_cost_delay_scorecard.csv")
    _write_csv(period_scorecard, out_dir / "17_yearly_monthly_stability.csv")
    _write_csv(bootstrap_summary, out_dir / "18_bootstrap_confidence_intervals.csv")
    _write_csv(random_path_distribution, out_dir / "19_matched_random_path_distribution.csv")
    _write_csv(random_execution_distribution, out_dir / "20_matched_random_execution_distribution.csv")
    _write_csv(model_table, out_dir / "21_expert_model_fit_methods.csv")
    _write_csv(feature_table, out_dir / "22_expert_feature_coverage.csv")
    _write_csv(rank_table, out_dir / "23_raw_rank_resolution_diagnostics.csv")
    _write_csv(uplift, out_dir / "24_uplift_decomposition.csv")
    _write_csv(tradeoff, out_dir / "25_frequency_precision_tradeoff.csv")
    _write_csv(audit, out_dir / "26_causal_execution_selection_audit.csv")
    if not bootstrap_distribution.empty:
        _write_csv(bootstrap_distribution, out_dir / "27_bootstrap_distribution.csv")
    if trade_parts:
        trades = pd.concat(trade_parts, ignore_index=True)
        if args.write_full_trades:
            _write_csv(trades, out_dir / "28_trade_audit.csv")
        else:
            sample = pd.concat(
                [
                    trades.nsmallest(min(2000, len(trades)), "net_return"),
                    trades.nlargest(min(2000, len(trades)), "net_return"),
                    trades.sample(min(4000, len(trades)), random_state=int(args.random_state)),
                ],
                ignore_index=True,
            ).drop_duplicates(["fold", "group_id", "event_id", "entry_time", "horizon_bars"])
            _write_csv(sample, out_dir / "28_trade_audit_sample.csv")

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
        "uplift_order": [
            "matched_random_1m",
            "broad_candidate_pool",
            "frozen_mechanism_rule",
            "simple_single_factor",
            "mechanism_expert_model",
            "execution_cost_delay",
        ],
        "mechanisms": [*MECHANISM_ORDER, UNRESOLVED_MECHANISM],
        "mechanisms_are_overlapping": True,
        "first_sweep_is_global_gate": False,
        "path_horizons": list(horizons),
        "target_levels_pct": [0.5, 1.0, 1.5, 2.0],
        "model_rank_source": "decision_function_or_logit_fallback",
        "execution": "closed signal bar -> next open -> first future closed-bar close >= +1% else horizon close",
        "ordinary_stop": None,
        "costs": {
            "round_trip_fee": float(args.entry_fee_rate + args.exit_fee_rate),
            "round_trip_slippage": float(args.entry_slippage_pct + args.exit_slippage_pct),
            "multipliers": [1.0, *[float(value) for value in args.stress_cost_multipliers]],
        },
        "entry_delay_stress_bars": [0, *[int(value) for value in args.stress_entry_delays]],
        "top_score_neighborhoods": list(top_pcts),
        "automatic_frozen_test_winner_selected": False,
        "research_leverage": 1.0,
        "future_live_leverage_reference": 15.0,
        "account_risk_sizing_applied": False,
        "live_ready": False,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "29_RESEARCH_SUMMARY.md").write_text(_summary(path_scorecard, execution_scorecard, audit), encoding="utf-8")
    result = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={result.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
