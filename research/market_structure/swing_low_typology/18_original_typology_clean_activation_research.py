#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Research 18: integrity-audited clean-path and causal-activation research.

Research 17 proved that causal online features can rank future C3-D/C3-E Swing
Low contexts and future +1% touches, but the selected paths suffered larger MAE
and remained unprofitable.  Research 18 freezes that candidate/type/opportunity
architecture.  It first audits label availability and apparent AUC, then tests
one additional clean-path head, one predeclared causal activation rule, and a
small frozen set of TP+1% exit structures.  Frequency remains a hard gate.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_common.review_pack import finalize_research_report  # noqa: E402
from research.market_structure.swing_low_typology.common.broad_reversal_evaluation import (  # noqa: E402
    CloseTargetCostSpec,
    build_multi_horizon_close_labels,
    execution_metrics,
    strongest_day_stress,
)
from research.market_structure.swing_low_typology.common.broad_reversal_mechanisms import (  # noqa: E402
    merge_macro_first_sweep_candidates,
    select_broad_region_events,
)
from research.market_structure.swing_low_typology.common.clean_activation import (  # noqa: E402
    ActivationSpec,
    FROZEN_EXIT_POLICIES,
    activation_state_table,
    attach_clean_path_targets,
    attach_signal_region_low,
    build_activation_map,
    executable_tp1_policy,
    first_policy_event_per_region,
    materialize_activation_events,
)
from research.market_structure.swing_low_typology.common.first_sweep_event import (  # noqa: E402
    build_first_sweep_event_decisions,
)
from research.market_structure.swing_low_typology.common.online_recognizability import (  # noqa: E402
    CandidateGateConfig,
    fit_binary_model,
)
from research.market_structure.swing_low_typology.common.original_typology_bridge import (  # noqa: E402
    map_candidates_to_future_typology,
    typology_inventory,
)
from research.market_structure.swing_low_typology.common.original_typology_experts import (  # noqa: E402
    COMPACT_EXPERT_FEATURES,
    EXPERT_SPECS,
    EmpiricalRankReference,
    ExpertModelUnavailableError,
    add_episode_weight,
    binary_ranking_metrics,
    build_expert_targets,
    build_fold_split,
    combine_component_ranks,
    fit_resolved_binary_model,
    folds_for_end_date,
    frequency_guard,
    months_in_fold,
    predict_binary_probability_chunked,
    predict_binary_score_chunked,
    raw_score_resolution,
    signed_feature_rank,
    target_capture_metrics,
    top_rank_mask,
    validate_feature_names,
)
from research.market_structure.swing_low_typology.common.reversal_opportunity import (  # noqa: E402
    build_reversal_candidate_features,
)
from research.market_structure.swing_low_typology.common.swing_low_dataset import (  # noqa: E402
    validate_trade_bar_fields,
)
from research.market_structure.swing_low_typology.common.typology_integrity import (  # noqa: E402
    FROZEN_PLACEBO_SPECS,
    attach_true_type_label_availability,
    coefficient_stability_summary,
    episode_ranking_metrics,
    extract_linear_coefficients,
    feature_group_map,
    label_availability_audit,
    stratified_permutation_target,
    time_shift_placebo_target,
)
from research.market_structure.swing_low_typology.common.walkforward_reversal import (  # noqa: E402
    build_broad_candidate_regions,
)

_R16R = importlib.import_module(
    "research.market_structure.swing_low_typology.16r_original_typology_online_bridge_audit"
)
_R17 = importlib.import_module(
    "research.market_structure.swing_low_typology.17_original_typology_online_expert_research"
)

SCRIPT_NAME = "18_original_typology_clean_activation_research"
SCRIPT_VERSION = "1.0.5"
EXPERIMENT_ID = "ETH_1M_ORIGINAL_TYPOLOGY_CLEAN_ACTIVATION_18"
EDGE_ID = "RESEARCH_ONLY_ETH_ORIGINAL_TYPOLOGY_CLEAN_ACTIVATION"
TITLE = "ETH Original Swing Low Clean Path and Activation Research 18"
DEFAULT_OUT_DIR = (
    "data/reports/research/market_structure/swing_low_typology/"
    "18_original_typology_clean_activation"
)
FROZEN_TARGET_MOVE_PCT = 1.0
FROZEN_HORIZON_BARS = 60
R17_TYPE_OPPORTUNITY_WEIGHTS = {"type": 0.60, "opportunity": 0.40}
R18_CLEAN_WEIGHTS = {"type": 0.45, "opportunity": 0.25, "clean": 0.30}
BROAD_CLEAN_WEIGHTS = {"opportunity": 0.50, "clean": 0.50}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit Research 17 AUC and test clean-path/causal-activation TP1% execution.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s v{SCRIPT_VERSION}")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--stage1-report-dir", default=_R16R.DEFAULT_STAGE1_DIR)
    p.add_argument("--stage2-report-dir", default=_R16R.DEFAULT_STAGE2_DIR)
    p.add_argument("--stage3-report-dir", default=_R16R.DEFAULT_STAGE3_DIR)
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

    p.add_argument("--reference-maximum-lead-bars", type=int, default=15)
    p.add_argument("--reference-price-tolerance-bp", type=float, default=75.0)
    p.add_argument("--rank-top-pcts", nargs="+", type=int, default=[20, 30, 40])
    p.add_argument("--policy-window-days", type=int, default=90)
    p.add_argument("--model-min-samples-leaf", type=int, default=50)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--prediction-chunk-size", type=int, default=20_000)
    p.add_argument("--label-vectorized-chunk-size", type=int, default=20_000)

    p.add_argument("--activation-maximum-wait-bars", type=int, default=10)
    p.add_argument("--activation-minimum-reclaim-bp", type=float, default=10.0)
    p.add_argument("--clean-maximum-mae-before-tp-pct", type=float, default=0.50)
    p.add_argument("--adverse-path-mae-pct", type=float, default=0.75)
    p.add_argument("--future-truncation-samples", type=int, default=6)

    p.add_argument("--minimum-annualized-raw-events", type=float, default=600.0)
    p.add_argument("--minimum-annualized-executable-trades", type=float, default=240.0)
    p.add_argument("--minimum-raw-events-per-fold", type=int, default=100)
    p.add_argument("--capital-fraction", type=float, default=0.10)

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


def _validate_args(args: argparse.Namespace) -> None:
    if args.symbol != "ETH-USDT-SWAP" or str(args.timeframe).lower() != "1m":
        raise ValueError("Research 18 is frozen to OKX ETH-USDT-SWAP 1m")
    if int(args.reference_maximum_lead_bars) != 15:
        raise ValueError("Research 18 freezes the original-type bridge at 15 bars")
    if not np.isclose(float(args.reference_price_tolerance_bp), 75.0):
        raise ValueError("Research 18 freezes the original-type price zone at 75bp")
    if tuple(sorted(set(map(int, args.rank_top_pcts)))) != (20, 30, 40):
        raise ValueError("Research 18 permits only Top20/Top30/Top40")
    if int(args.activation_maximum_wait_bars) != 10:
        raise ValueError("Research 18 freezes activation wait at 10 bars")
    if not np.isclose(float(args.activation_minimum_reclaim_bp), 10.0):
        raise ValueError("Research 18 freezes activation reclaim at 10bp")
    if not np.isclose(float(args.clean_maximum_mae_before_tp_pct), 0.50):
        raise ValueError("Research 18 freezes clean TP MAE-before-TP at 0.50%")
    validate_feature_names(COMPACT_EXPERT_FEATURES)


def _build_dataset(
    args: argparse.Namespace,
    bars: pd.DataFrame,
    historical: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, pd.DataFrame]]:
    print("[stage] frozen broad causal candidate universe", flush=True)
    gate_config = CandidateGateConfig(
        lookback=int(args.candidate_lookback_bars),
        horizon=60,
        new_low_window=int(args.candidate_new_low_window),
        near_floor_window=int(args.candidate_near_floor_window),
        position_window=int(args.candidate_position_window),
        near_floor_tolerance_bp=float(args.candidate_near_floor_tolerance_bp),
        max_position_in_range=float(args.candidate_max_position_in_range),
    )
    raw_gate, gate_coverage = _R17.build_online_candidate_events(
        bars,
        research_start=pd.Timestamp(args.start_date),
        research_end_exclusive=_end_exclusive(args.end_date),
        config=gate_config,
    )
    print("[stage] frozen First Sweep control branch", flush=True)
    first_sweep = build_first_sweep_event_decisions(
        bars,
        research_start=pd.Timestamp(args.start_date),
        research_end_exclusive=_end_exclusive(args.end_date),
        pivot_minutes=tuple(map(int, args.liquidity_pivot_minutes)),
        pivot_weights=tuple(map(float, args.liquidity_pivot_weights)),
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
    raw_union = merge_macro_first_sweep_candidates(bars, raw_gate, first_sweep.decisions)
    print("[stage] frozen causal regions and spaced events", flush=True)
    region_result = build_broad_candidate_regions(
        bars,
        raw_union,
        max_gap_bars=int(args.region_max_gap_bars),
        max_region_bars=int(args.region_max_bars),
        retest_tolerance_bp=float(args.region_retest_tolerance_bp),
        show_progress=True,
    )
    selected_meta = select_broad_region_events(
        region_result.frame, cooldown_bars=int(args.broad_cooldown_bars)
    )
    print("[stage] frozen causal online feature matrix", flush=True)
    feature_result = build_reversal_candidate_features(
        bars, selected_meta, include_session=True, include_htf=True, show_progress=True
    )
    if not feature_result.alignment_audit.empty and not feature_result.alignment_audit["passed"].all():
        raise RuntimeError("HTF available-time audit failed")
    frame = feature_result.frame.copy()
    frame["is_macro_first_sweep"] = frame.get("is_macro_first_sweep", False)
    frame["is_macro_first_sweep"] = frame["is_macro_first_sweep"].fillna(False).astype(bool)
    print("[stage] TP1% path-quality labels", flush=True)
    labels = build_multi_horizon_close_labels(
        bars,
        frame,
        horizons=(60,),
        target_levels_pct=(0.5, 1.0, 1.5, 2.0),
        vectorized_chunk_size=int(args.label_vectorized_chunk_size),
        show_progress=True,
    )
    frame = frame.merge(labels, on="event_id", how="inner", validate="one_to_one")
    frame = map_candidates_to_future_typology(
        frame,
        historical,
        maximum_lead_bars=int(args.reference_maximum_lead_bars),
        price_tolerance_bp=float(args.reference_price_tolerance_bp),
    )
    frame = build_expert_targets(
        frame, bridge_maximum_lead_bars=int(args.reference_maximum_lead_bars)
    )
    frame = attach_true_type_label_availability(
        frame,
        historical,
        bridge_maximum_lead_bars=int(args.reference_maximum_lead_bars),
    )
    frame = attach_clean_path_targets(
        frame,
        horizon=60,
        maximum_mae_before_tp_pct=float(args.clean_maximum_mae_before_tp_pct),
        adverse_mae_pct=float(args.adverse_path_mae_pct),
    )
    missing = sorted(set(COMPACT_EXPERT_FEATURES).difference(frame.columns))
    if missing:
        raise RuntimeError(f"Research 18 features missing: {missing}")

    print("[stage] causal activation map and activation-entry labels", flush=True)
    activation_spec = ActivationSpec(
        maximum_wait_bars=int(args.activation_maximum_wait_bars),
        minimum_bars_since_low=1,
        minimum_reclaim_bp=float(args.activation_minimum_reclaim_bp),
    )
    activation_states = activation_state_table(region_result.frame, bars, spec=activation_spec)
    activation_map = build_activation_map(
        frame, activation_states, maximum_wait_bars=int(args.activation_maximum_wait_bars)
    )
    activation_events = materialize_activation_events(frame, activation_map)
    label_columns = {
        column
        for column in activation_events.columns
        if column == "entry_time"
        or column == "entry_price"
        or column == "label_end_time"
        or column.startswith(("tp_", "time_to_tp_", "mae_", "mfe_", "terminal_", "clean_", "deep_sweep_", "permanent_failure_", "target_clean_", "target_fast_clean_", "target_adverse_"))
    }
    activation_core = activation_events.drop(columns=sorted(label_columns), errors="ignore")
    if activation_core.empty:
        activation_frame = activation_core.copy()
    else:
        activation_labels = build_multi_horizon_close_labels(
            bars,
            activation_core,
            horizons=(60,),
            target_levels_pct=(0.5, 1.0, 1.5, 2.0),
            vectorized_chunk_size=int(args.label_vectorized_chunk_size),
            show_progress=True,
        )
        activation_frame = activation_core.merge(
            activation_labels, on="event_id", how="inner", validate="one_to_one"
        )
        activation_frame = attach_clean_path_targets(
            activation_frame,
            horizon=60,
            maximum_mae_before_tp_pct=float(args.clean_maximum_mae_before_tp_pct),
            adverse_mae_pct=float(args.adverse_path_mae_pct),
        )
        # Activation changes the next-open reference, so its TP opportunity
        # label must be owned by the activation event rather than inherited
        # from the earlier armed candidate.  Original-type targets remain tied
        # to the armed event and are intentionally preserved for recall audit.
        activation_frame["target_tp60"] = (
            activation_frame["tp_1_h60"].fillna(False).astype(bool)
        )
    diagnostics = {
        "gate_coverage": gate_coverage,
        "first_sweep": first_sweep.diagnostics,
        "region_summary": region_result.summary,
        "region_dictionary": region_result.dictionary,
        "feature_dictionary": feature_result.dictionary,
        "alignment_audit": feature_result.alignment_audit,
        "label_availability": label_availability_audit(frame),
        "activation_summary": pd.DataFrame(
            [
                {"metric": "armed_events", "value": int(len(frame))},
                {"metric": "activation_found", "value": int(activation_map["activation_found"].sum())},
                {"metric": "activation_labeled", "value": int(len(activation_frame))},
                {"metric": "activation_found_rate", "value": float(activation_map["activation_found"].mean())},
            ]
        ),
        "activation_map": activation_map,
        "selected_meta": selected_meta,
    }
    return frame, activation_frame, diagnostics


def _ranking_row(
    *,
    fold: str,
    expert_id: str,
    head: str,
    target_column: str,
    test: pd.DataFrame,
    score: np.ndarray,
) -> dict[str, object]:
    base = binary_ranking_metrics(test[target_column], score)
    episode = episode_ranking_metrics(
        test, target_column=target_column, score=score
    )
    return {
        "fold": fold,
        "expert_id": expert_id,
        "head": head,
        "target_column": target_column,
        "test_rows": int(len(test)),
        "test_positives": int(test[target_column].fillna(False).astype(bool).sum()),
        **base,
        **episode,
        **{f"score_{key}": value for key, value in raw_score_resolution(score).items()},
    }


def _rank_pair(
    policy_components: Mapping[str, np.ndarray],
    test_components: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    policy_rank, _, _ = combine_component_ranks(
        policy_components, policy_components, weights=weights
    )
    _, test_rank, _ = combine_component_ranks(
        policy_components, test_components, weights=weights
    )
    return policy_rank, test_rank


def _placebo_metrics(
    *,
    fold: str,
    expert_id: str,
    head: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    feature_columns: Sequence[str],
    real_test_score: np.ndarray,
    random_state: int,
    min_samples_leaf: int,
) -> list[dict[str, object]]:
    """Evaluate frozen OOS scores against relabeled placebo outcomes.

    The model is not refitted on noise.  This audit asks the relevant leakage
    question directly: does the reported OOS ranking remain associated after
    month/volatility-preserving permutations or calendar shifts?  Avoiding
    placebo refits keeps the integrity stage practical on the full 81k-event
    candidate set.
    """

    rows: list[dict[str, object]] = []
    for spec in FROZEN_PLACEBO_SPECS:
        train_copy = train.copy()
        test_copy = test.copy()
        if spec.kind == "stratified_permutation":
            train_copy["_placebo_target"] = stratified_permutation_target(
                train_copy, target_column, random_state=int(spec.random_state or random_state)
            )
            test_copy["_placebo_target"] = stratified_permutation_target(
                test_copy, target_column, random_state=int(spec.random_state or random_state) + 100
            )
            valid_test = pd.Series(True, index=test_copy.index, dtype=bool)
        elif spec.kind == "time_shift":
            train_target, train_valid = time_shift_placebo_target(
                train_copy, target_column, shift_days=int(spec.shift_days or 1)
            )
            test_target, test_valid = time_shift_placebo_target(
                test_copy, target_column, shift_days=int(spec.shift_days or 1)
            )
            train_copy = train_copy.loc[train_valid].copy()
            test_copy["_placebo_target"] = test_target.astype(bool)
            train_copy["_placebo_target"] = train_target.loc[train_copy.index].astype(bool)
            valid_test = test_valid.astype(bool)
        else:  # pragma: no cover
            raise ValueError(spec.kind)

        score = np.asarray(real_test_score, dtype=float)[valid_test.to_numpy()]
        target = test_copy.loc[valid_test, "_placebo_target"]
        if len(target) < 100 or target.nunique() < 2:
            rows.append(
                {
                    "fold": fold,
                    "expert_id": expert_id,
                    "head": head,
                    "placebo_id": spec.placebo_id,
                    "placebo_mode": "frozen_score_relabel",
                    "status": "insufficient_placebo_classes",
                    "train_rows": int(len(train_copy)),
                    "test_rows": int(len(target)),
                }
            )
            continue
        metrics = binary_ranking_metrics(target, score)
        rows.append(
            {
                "fold": fold,
                "expert_id": expert_id,
                "head": head,
                "placebo_id": spec.placebo_id,
                "placebo_mode": "frozen_score_relabel",
                "status": "ok",
                "train_rows": int(len(train_copy)),
                "test_rows": int(len(target)),
                **metrics,
            }
        )

    return rows


def _ablation_metrics(
    *,
    fold: str,
    expert_id: str,
    head: str,
    fitted_model: object,
    test: pd.DataFrame,
    target_column: str,
    feature_columns: Sequence[str],
    full_auc: float,
) -> list[dict[str, object]]:
    """Neutralize one feature group in the frozen fitted model.

    This is an OOS reliance audit rather than a new model search.  Replacing a
    group with its training median preserves every other coefficient and avoids
    thirty expensive refits across the three folds.
    """

    rows: list[dict[str, object]] = []
    groups = feature_group_map(feature_columns)
    model = getattr(fitted_model, "model", fitted_model)
    medians = getattr(model, "medians", pd.Series(dtype=float))
    for group_id, removed in groups.items():
        affected = tuple(column for column in removed if column in feature_columns)
        if not affected:
            rows.append(
                {
                    "fold": fold,
                    "expert_id": expert_id,
                    "head": head,
                    "ablation_group": group_id,
                    "ablation_mode": "frozen_model_median_neutralization",
                    "status": "no_selected_features_in_group",
                    "removed_features": 0,
                }
            )
            continue
        neutralized = test.copy()
        for column in affected:
            neutralized[column] = float(medians.get(column, 0.0))
        try:
            score = np.asarray(model.predict_score(neutralized), dtype=float)
            metrics = binary_ranking_metrics(test[target_column], score)
            rows.append(
                {
                    "fold": fold,
                    "expert_id": expert_id,
                    "head": head,
                    "ablation_group": group_id,
                    "ablation_mode": "frozen_model_median_neutralization",
                    "status": "ok",
                    "removed_features": len(affected),
                    **metrics,
                    "roc_auc_change": float(metrics["roc_auc"] - full_auc),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "fold": fold,
                    "expert_id": expert_id,
                    "head": head,
                    "ablation_group": group_id,
                    "ablation_mode": "frozen_model_median_neutralization",
                    "status": "unavailable",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "removed_features": len(affected),
                }
            )
    return rows


def _future_truncation_audit(
    bars: pd.DataFrame,
    frame: pd.DataFrame,
    selected_meta: pd.DataFrame,
    *,
    samples: int,
) -> pd.DataFrame:
    if int(samples) <= 0 or frame.empty:
        return pd.DataFrame(
            [{"check": "future_truncation_rebuild", "passed": True, "detail": "disabled"}]
        )
    ordered = frame.sort_values(["extreme_pos", "event_id"], kind="mergesort")
    positions = np.linspace(0, len(ordered) - 1, num=min(int(samples), len(ordered)), dtype=int)
    meta = selected_meta.set_index("event_id", drop=False)
    rows: list[dict[str, object]] = []
    for location in positions:
        full = ordered.iloc[int(location)]
        event_id = str(full["event_id"])
        if event_id not in meta.index:
            rows.append({"event_id": event_id, "passed": False, "detail": "missing selected_meta"})
            continue
        extreme_pos = int(full["extreme_pos"])
        prefix = bars.iloc[: extreme_pos + 1].copy()
        rebuilt = build_reversal_candidate_features(
            prefix,
            meta.loc[[event_id]].copy(),
            include_session=True,
            include_htf=True,
            show_progress=False,
        ).frame
        if len(rebuilt) != 1:
            rows.append({"event_id": event_id, "passed": False, "detail": "rebuild row count"})
            continue
        mismatches: list[str] = []
        for column in COMPACT_EXPERT_FEATURES:
            left = pd.to_numeric(pd.Series([full[column]]), errors="coerce").iloc[0]
            right = pd.to_numeric(rebuilt[column], errors="coerce").iloc[0]
            if pd.isna(left) and pd.isna(right):
                continue
            if not (np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, rtol=1e-9, atol=1e-11)):
                mismatches.append(f"{column}:{left}->{right}")
        rows.append(
            {
                "event_id": event_id,
                "extreme_time": full["extreme_time"],
                "extreme_pos": extreme_pos,
                "checked_features": len(COMPACT_EXPERT_FEATURES),
                "mismatch_count": len(mismatches),
                "passed": not mismatches,
                "detail": "|".join(mismatches[:10]),
            }
        )
    return pd.DataFrame(rows)


def _entry_frames(
    bars: pd.DataFrame,
    selected: pd.DataFrame,
    activation_frame: pd.DataFrame,
    *,
    fold_test_end: pd.Timestamp,
) -> Mapping[str, pd.DataFrame]:
    armed = first_policy_event_per_region(selected)
    base = attach_signal_region_low(armed, bars)
    if activation_frame.empty or armed.empty:
        activation = activation_frame.iloc[0:0].copy()
    else:
        required = {"base_event_id", "extreme_time", "label_end_time"}
        missing = sorted(required.difference(activation_frame.columns))
        if missing:
            raise RuntimeError(f"activation frame missing columns: {missing}")
        activation_ids = set(armed["event_id"].astype(str))
        activation = activation_frame[
            activation_frame["base_event_id"].astype(str).isin(activation_ids)
        ].copy()
        activation = activation[
            (pd.to_datetime(activation["extreme_time"], errors="raise") <= fold_test_end)
            & (pd.to_datetime(activation["label_end_time"], errors="raise") <= fold_test_end)
        ]
        if not activation.empty:
            rank_lookup = armed.set_index(armed["event_id"].astype(str))["opportunity_score"]
            activation["opportunity_score"] = activation["base_event_id"].astype(str).map(rank_lookup)
            if activation["opportunity_score"].isna().any():
                raise RuntimeError("activation score mapping produced NA")
    return {
        "BASE_NEXT_OPEN": base.reset_index(drop=True),
        "CAUSAL_ACTIVATION_WAIT10": activation.reset_index(drop=True),
    }


def _exit_stress(exit_policy_id: str) -> tuple[tuple[float, int], ...]:
    # Every exit structure receives at least 1x/1.5x cost stress.  The frozen
    # 60-bar baseline additionally receives 2x cost and 1/3-bar delay stress.
    if exit_policy_id == "TP1_TIME60":
        return ((1.0, 0), (1.5, 0), (2.0, 0), (1.0, 1), (1.0, 3))
    return ((1.0, 0), (1.5, 0))


def _path_row(selected: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, object]:
    if selected.empty:
        return {
            "tp60_rate": np.nan,
            "tp60_uplift_pp": np.nan,
            "clean_tp60_rate": np.nan,
            "clean_tp60_uplift_pp": np.nan,
            "adverse_path60_rate": np.nan,
            "adverse_path60_change_pp": np.nan,
            "mean_mae60_pct": np.nan,
            "mae60_change_pp": np.nan,
            "median_time_to_tp60": np.nan,
        }
    tp = float(selected["tp_1_h60"].astype(bool).mean())
    base_tp = float(baseline["tp_1_h60"].astype(bool).mean()) if len(baseline) else np.nan
    clean = float(selected["target_clean_tp60"].astype(bool).mean())
    base_clean = float(baseline["target_clean_tp60"].astype(bool).mean()) if len(baseline) else np.nan
    adverse = float(selected["target_adverse_path60"].astype(bool).mean())
    base_adverse = float(baseline["target_adverse_path60"].astype(bool).mean()) if len(baseline) else np.nan
    mae = float(pd.to_numeric(selected["mae_h60_pct"], errors="coerce").mean())
    base_mae = float(pd.to_numeric(baseline["mae_h60_pct"], errors="coerce").mean()) if len(baseline) else np.nan
    return {
        "tp60_rate": tp,
        "tp60_uplift_pp": (tp - base_tp) * 100.0,
        "clean_tp60_rate": clean,
        "clean_tp60_uplift_pp": (clean - base_clean) * 100.0,
        "adverse_path60_rate": adverse,
        "adverse_path60_change_pp": (adverse - base_adverse) * 100.0,
        "mean_mae60_pct": mae,
        "mae60_change_pp": mae - base_mae,
        "median_time_to_tp60": float(pd.to_numeric(selected["time_to_tp_1_h60"], errors="coerce").median()),
    }


def _evaluate_selected(
    *,
    bars: pd.DataFrame,
    fold,
    expert_id: str,
    policy_id: str,
    top_pct: int,
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
    activation_frame: pd.DataFrame,
    target_column: str | None,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[pd.DataFrame]]:
    selected_modes = _entry_frames(
        bars, selected, activation_frame, fold_test_end=fold.test_end
    )
    baseline_ranked = baseline.copy()
    baseline_ranked["opportunity_score"] = 50.0
    baseline_modes = _entry_frames(
        bars, baseline_ranked, activation_frame, fold_test_end=fold.test_end
    )
    rows: list[dict[str, object]] = []
    strong_rows: list[dict[str, object]] = []
    trade_parts: list[pd.DataFrame] = []
    months = months_in_fold(fold)
    for entry_mode, entry_events in selected_modes.items():
        entry_baseline = baseline_modes[entry_mode]
        path = _path_row(entry_events, entry_baseline)
        capture = (
            target_capture_metrics(entry_events, entry_baseline, target_column=target_column)
            if target_column is not None and target_column in entry_events.columns
            else {}
        )
        for exit_policy in FROZEN_EXIT_POLICIES:
            for cost_multiplier, delay in _exit_stress(exit_policy.policy_id):
                costs = CloseTargetCostSpec(cost_multiplier=float(cost_multiplier))
                trades, counts = executable_tp1_policy(
                    bars,
                    entry_events,
                    policy=exit_policy,
                    costs=costs,
                    entry_delay_bars=int(delay),
                )
                frequency = frequency_guard(
                    raw_events=int(len(entry_events)),
                    executable_trades=int(len(trades)),
                    months=months,
                    minimum_annualized_raw_events=float(args.minimum_annualized_raw_events),
                    minimum_annualized_executable_trades=float(args.minimum_annualized_executable_trades),
                    minimum_raw_events=int(args.minimum_raw_events_per_fold),
                )
                metrics = execution_metrics(
                    trades,
                    months=months,
                    counts=counts,
                    capital_fraction=float(args.capital_fraction),
                )
                row = {
                    "fold": fold.fold,
                    "expert_id": expert_id,
                    "policy_id": policy_id,
                    "top_pct": int(top_pct),
                    "entry_mode": entry_mode,
                    "exit_policy_id": exit_policy.policy_id,
                    "cost_multiplier": float(cost_multiplier),
                    "entry_delay_bars": int(delay),
                    "broad_test_candidates": int(len(baseline)),
                    "selected_share_of_broad_candidates": float(len(entry_events) / max(1, len(baseline))),
                    **path,
                    **frequency,
                    "net_edge_throughput_bps_per_month": float(
                        metrics.get("net_expectancy_bps", np.nan)
                        * metrics.get("events_per_month", 0.0)
                    ),
                    **metrics,
                    **capture,
                }
                rows.append(row)
                # Metrics and strongest-day stress use the complete trade set
                # immediately.  Persist only a bounded primary-scenario audit
                # sample; retaining every duplicated cost/delay trade would
                # multiply memory into millions of rows on the full history.
                if (
                    not trades.empty
                    and float(cost_multiplier) == 1.0
                    and int(delay) == 0
                ):
                    seed = (
                        int(top_pct)
                        + sum(map(ord, str(policy_id)))
                        + sum(map(ord, str(entry_mode)))
                        + sum(map(ord, str(exit_policy.policy_id)))
                    ) % (2**32 - 1)
                    part = (
                        trades
                        if len(trades) <= 200
                        else trades.sample(200, random_state=seed)
                    ).copy()
                    part["fold"] = fold.fold
                    part["expert_id"] = expert_id
                    part["policy_id"] = policy_id
                    part["top_pct"] = int(top_pct)
                    part["entry_mode"] = entry_mode
                    part["exit_policy_id"] = exit_policy.policy_id
                    part["trade_storage_scope"] = "bounded_primary_scenario_audit_sample"
                    trade_parts.append(part)
                if (
                    exit_policy.policy_id == "TP1_TIME60"
                    and float(cost_multiplier) == 1.0
                    and int(delay) == 0
                ):
                    for remove_days in (5, 10):
                        stressed = strongest_day_stress(
                            trades,
                            remove_days=remove_days,
                            months=months,
                            raw_signals=int(len(entry_events)),
                            capital_fraction=float(args.capital_fraction),
                        )
                        strong_rows.append(
                            {
                                "fold": fold.fold,
                                "expert_id": expert_id,
                                "policy_id": policy_id,
                                "top_pct": int(top_pct),
                                "entry_mode": entry_mode,
                                "exit_policy_id": exit_policy.policy_id,
                                "remove_strongest_days": int(remove_days),
                                **stressed,
                            }
                        )
    return rows, strong_rows, trade_parts


def _finite_median(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(finite.median()) if len(finite) else np.nan


def _finite_minimum(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(finite.min()) if len(finite) else np.nan


def _crossfold_summary(scorecard: pd.DataFrame) -> pd.DataFrame:
    if scorecard.empty:
        return pd.DataFrame()
    groups = [
        "expert_id",
        "policy_id",
        "top_pct",
        "entry_mode",
        "exit_policy_id",
        "cost_multiplier",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in scorecard.groupby(groups, sort=True, dropna=False):
        values = dict(zip(groups, keys))
        primary = group[group["entry_delay_bars"].eq(0)]
        if primary.empty:
            continue
        folds = int(primary["fold"].nunique())
        candidate = bool(
            folds == 3
            and int(primary["frequency_guard_passed"].fillna(False).sum()) == 3
            and int((primary["net_expectancy_bps"] > 0.0).sum()) == 3
            and int((primary["clean_tp60_uplift_pp"] > 0.0).sum()) == 3
            and int((primary["mae60_change_pp"] <= 0.05).sum()) == 3
            and int(primary["executable_trades"].sum()) >= 600
            and _finite_minimum(primary["profit_factor"]) > 1.0
            and _finite_median(primary["profit_factor"]) >= 1.05
        )
        rows.append(
            {
                **values,
                "folds": folds,
                "frequency_pass_folds": int(primary["frequency_guard_passed"].fillna(False).sum()),
                "positive_net_expectancy_folds": int((primary["net_expectancy_bps"] > 0.0).sum()),
                "positive_clean_uplift_folds": int((primary["clean_tp60_uplift_pp"] > 0.0).sum()),
                "nonworse_mae_folds": int((primary["mae60_change_pp"] <= 0.05).sum()),
                "total_oos_raw_events": int(primary["raw_events"].sum()),
                "total_oos_executable_trades": int(primary["executable_trades"].sum()),
                "minimum_annualized_executable_trades": _finite_minimum(primary["annualized_executable_trades"]),
                "median_tp60_uplift_pp": _finite_median(primary["tp60_uplift_pp"]),
                "median_clean_tp60_uplift_pp": _finite_median(primary["clean_tp60_uplift_pp"]),
                "median_adverse_path_change_pp": _finite_median(primary["adverse_path60_change_pp"]),
                "median_mae60_change_pp": _finite_median(primary["mae60_change_pp"]),
                "median_net_expectancy_bps": _finite_median(primary["net_expectancy_bps"]),
                "minimum_profit_factor": _finite_minimum(primary["profit_factor"]),
                "median_profit_factor": _finite_median(primary["profit_factor"]),
                "research_candidate_status": "candidate" if candidate else "not_supported",
            }
        )
    return pd.DataFrame(rows)


def _neighborhood_summary(crossfold: pd.DataFrame) -> pd.DataFrame:
    if crossfold.empty:
        return pd.DataFrame()
    groups = ["expert_id", "policy_id", "entry_mode", "exit_policy_id", "cost_multiplier"]
    rows: list[dict[str, object]] = []
    for keys, group in crossfold.groupby(groups, sort=True, dropna=False):
        values = dict(zip(groups, keys))
        ranked = group[group["top_pct"].isin([20, 30, 40])]
        candidate_tops = sorted(
            ranked.loc[ranked["research_candidate_status"].eq("candidate"), "top_pct"]
            .astype(int)
            .tolist()
        )
        rows.append(
            {
                **values,
                "evaluated_top_pcts": "|".join(map(str, sorted(ranked["top_pct"].astype(int).unique()))),
                "candidate_top_pcts": "|".join(map(str, candidate_tops)),
                "top30_supported": 30 in candidate_tops,
                "adjacent_neighbor_supported": bool(20 in candidate_tops or 40 in candidate_tops),
                "neighborhood_status": (
                    "candidate"
                    if 30 in candidate_tops and (20 in candidate_tops or 40 in candidate_tops)
                    else "not_supported"
                ),
            }
        )
    return pd.DataFrame(rows)


def _robust_policy_summary(
    crossfold: pd.DataFrame,
    neighborhood: pd.DataFrame,
    scorecard: pd.DataFrame,
    strong_days: pd.DataFrame,
) -> pd.DataFrame:
    """Apply deployment gates with vectorized groupby/merge operations."""

    if crossfold.empty:
        return pd.DataFrame()
    keys = ["expert_id", "policy_id", "top_pct", "entry_mode", "exit_policy_id"]
    group_keys = ["expert_id", "policy_id", "entry_mode", "exit_policy_id"]
    base = crossfold.loc[crossfold["cost_multiplier"].eq(1.0), keys + ["research_candidate_status"]].copy()
    base = base.rename(columns={"research_candidate_status": "base_1x_status"})

    def cost_status(multiplier: float, output: str) -> pd.DataFrame:
        subset = crossfold.loc[
            crossfold["cost_multiplier"].eq(float(multiplier)),
            keys + ["research_candidate_status"],
        ].copy()
        return subset.rename(columns={"research_candidate_status": output})

    result = base.merge(cost_status(1.5, "cost_1p5x_status"), on=keys, how="left", validate="one_to_one")
    result = result.merge(cost_status(2.0, "cost_2x_status"), on=keys, how="left", validate="one_to_one")

    neighbor = neighborhood.loc[
        neighborhood["cost_multiplier"].eq(1.0),
        group_keys + ["neighborhood_status"],
    ].copy()
    result = result.merge(neighbor, on=group_keys, how="left", validate="many_to_one")

    if strong_days.empty:
        strong = pd.DataFrame(columns=keys + ["strongest10_days_rows", "strongest10_days_positive_folds"] )
    else:
        strong_source = strong_days.loc[
            strong_days["remove_strongest_days"].eq(10),
            keys + ["fold", "net_expectancy_bps"],
        ].copy()
        strong_source["positive"] = pd.to_numeric(
            strong_source["net_expectancy_bps"], errors="coerce"
        ).gt(0.0)
        strong = (
            strong_source.groupby(keys, sort=False, dropna=False)
            .agg(
                strongest10_days_rows=("fold", "nunique"),
                strongest10_days_positive_folds=("positive", "sum"),
            )
            .reset_index()
        )
    result = result.merge(strong, on=keys, how="left", validate="one_to_one")

    delay_source = scorecard.loc[
        scorecard["cost_multiplier"].eq(1.0)
        & scorecard["entry_delay_bars"].isin([1, 3]),
        keys + ["fold", "entry_delay_bars", "net_expectancy_bps"],
    ].copy()
    if delay_source.empty:
        delay = pd.DataFrame(columns=keys + ["delay1_positive_folds", "delay3_positive_folds_diagnostic"] )
    else:
        delay_source["positive"] = pd.to_numeric(
            delay_source["net_expectancy_bps"], errors="coerce"
        ).gt(0.0)
        delay = (
            delay_source.groupby(keys + ["entry_delay_bars"], sort=False, dropna=False)["positive"]
            .sum()
            .unstack("entry_delay_bars", fill_value=0)
            .rename(columns={1: "delay1_positive_folds", 3: "delay3_positive_folds_diagnostic"})
            .reset_index()
        )
    result = result.merge(delay, on=keys, how="left", validate="one_to_one")

    for column in (
        "strongest10_days_rows",
        "strongest10_days_positive_folds",
        "delay1_positive_folds",
        "delay3_positive_folds_diagnostic",
    ):
        if column not in result.columns:
            result[column] = 0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)

    result["base_1x_supported"] = result["base_1x_status"].eq("candidate")
    result["cost_1p5x_supported"] = result["cost_1p5x_status"].eq("candidate")
    result["cost_2x_supported_diagnostic"] = result["cost_2x_status"].eq("candidate")
    result["top_neighborhood_supported"] = result["neighborhood_status"].eq("candidate")
    result["strongest10_days_supported"] = (
        result["strongest10_days_rows"].eq(3)
        & result["strongest10_days_positive_folds"].eq(3)
    )
    result["delay1_required"] = result["exit_policy_id"].eq("TP1_TIME60")
    delay_supported = ~result["delay1_required"] | result["delay1_positive_folds"].eq(3)
    final = (
        result["base_1x_supported"]
        & result["cost_1p5x_supported"]
        & result["top_neighborhood_supported"]
        & result["strongest10_days_supported"]
        & delay_supported
    )
    result["final_policy_status"] = np.where(final, "candidate", "not_supported")
    return result[
        keys
        + [
            "base_1x_supported",
            "cost_1p5x_supported",
            "cost_2x_supported_diagnostic",
            "top_neighborhood_supported",
            "strongest10_days_positive_folds",
            "strongest10_days_supported",
            "delay1_positive_folds",
            "delay3_positive_folds_diagnostic",
            "delay1_required",
            "final_policy_status",
        ]
    ].sort_values(keys, kind="mergesort").reset_index(drop=True)


def _run_fold(
    *,
    bars: pd.DataFrame,
    frame: pd.DataFrame,
    activation_frame: pd.DataFrame,
    fold,
    args: argparse.Namespace,
) -> Mapping[str, list]:
    print(f"[fold] {fold.fold}", flush=True)
    split_tp = build_fold_split(
        frame, fold, policy_days=int(args.policy_window_days), fit_label_end_column="label_end_time"
    )
    split_type = build_fold_split(
        frame, fold, policy_days=int(args.policy_window_days), fit_label_end_column="type_label_end_time"
    )
    policy = frame.loc[split_tp.policy_mask].reset_index(drop=True)
    test = frame.loc[split_tp.test_mask].reset_index(drop=True)
    if len(policy) < 500 or len(test) < 500:
        raise RuntimeError(f"{fold.fold} has insufficient policy/test rows")

    result: dict[str, list] = {
        "split": [
            {
                "fold": fold.fold,
                "train_start": fold.train_start,
                "model_fit_end_tp": split_tp.model_fit_end,
                "model_fit_end_type": split_type.model_fit_end,
                "policy_start": split_tp.policy_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "policy_rows": int(len(policy)),
                "test_rows": int(len(test)),
            }
        ],
        "incidence": [],
        "features": [],
        "fits": [],
        "metrics": [],
        "coefficients": [],
        "placebos": [],
        "ablations": [],
        "predictions": [],
        "scorecard": [],
        "strong_days": [],
        "trades": [],
    }
    for target in (
        "target_tp60",
        "target_clean_tp60",
        "target_adverse_path60",
        "target_E1_C3D_PRICE_RESPONSE",
        "target_E2_C3E_EARLY_RECOVERY",
    ):
        result["incidence"].append(
            {
                "fold": fold.fold,
                "target": target,
                "policy_rows": int(len(policy)),
                "policy_positives": int(policy[target].astype(bool).sum()),
                "test_rows": int(len(test)),
                "test_positives": int(test[target].astype(bool).sum()),
            }
        )

    # Shared +1% opportunity and clean-path heads.
    head_specs = (
        ("B0_BROAD", "global_opportunity", "target_tp60"),
        ("B0_BROAD", "clean_path", "target_clean_tp60"),
    )
    fitted: dict[str, object] = {}
    policy_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}
    policy_ranks: dict[str, np.ndarray] = {}
    test_ranks: dict[str, np.ndarray] = {}
    probabilities: dict[str, np.ndarray] = {}
    for expert_id, head, target in head_specs:
        train = add_episode_weight(frame.loc[split_tp.model_fit_mask].copy())
        fit, feature_diag = fit_resolved_binary_model(
            train,
            policy,
            requested_features=COMPACT_EXPERT_FEATURES,
            required_features=("sell_pressure_absorption_30", "region_rebound_from_low"),
            target_column=target,
            random_state=int(args.random_state),
            min_samples_leaf=int(args.model_min_samples_leaf),
            prediction_chunk_size=int(args.prediction_chunk_size),
        )
        feature_diag.insert(0, "head", head)
        feature_diag.insert(0, "expert_id", expert_id)
        feature_diag.insert(0, "fold", fold.fold)
        result["features"].append(feature_diag)
        result["fits"].append(
            {
                "fold": fold.fold,
                "expert_id": expert_id,
                "head": head,
                "model_fit_rows": int(len(train)),
                "model_fit_positives": int(train[target].sum()),
                "policy_rows": int(len(policy)),
                **fit.diagnostics,
            }
        )
        policy_score = predict_binary_score_chunked(
            fit.model, policy, chunk_size=int(args.prediction_chunk_size)
        )
        test_score = predict_binary_score_chunked(
            fit.model, test, chunk_size=int(args.prediction_chunk_size)
        )
        ref = EmpiricalRankReference.fit(policy_score)
        fitted[head] = fit
        policy_scores[head] = policy_score
        test_scores[head] = test_score
        policy_ranks[head] = ref.transform(policy_score)
        test_ranks[head] = ref.transform(test_score)
        probabilities[head] = predict_binary_probability_chunked(
            fit.model, test, chunk_size=int(args.prediction_chunk_size)
        )
        result["metrics"].append(
            _ranking_row(
                fold=fold.fold,
                expert_id=expert_id,
                head=head,
                target_column=target,
                test=test,
                score=test_score,
            )
        )
        result["coefficients"].append(
            extract_linear_coefficients(
                fit.model, fold=fold.fold, expert_id=expert_id, head=head
            )
        )
        full_auc = float(result["metrics"][-1]["roc_auc"])
        result["placebos"].extend(
            _placebo_metrics(
                fold=fold.fold,
                expert_id=expert_id,
                head=head,
                train=train,
                test=test,
                target_column=target,
                feature_columns=fit.selected_features,
                real_test_score=test_score,
                random_state=int(args.random_state),
                min_samples_leaf=int(args.model_min_samples_leaf),
            )
        )
        result["ablations"].extend(
            _ablation_metrics(
                fold=fold.fold,
                expert_id=expert_id,
                head=head,
                fitted_model=fit,
                test=test,
                target_column=target,
                feature_columns=fit.selected_features,
                full_auc=full_auc,
            )
        )

    broad_combined_policy, broad_combined_test = _rank_pair(
        {"opportunity": policy_ranks["global_opportunity"], "clean": policy_ranks["clean_path"]},
        {"opportunity": test_ranks["global_opportunity"], "clean": test_ranks["clean_path"]},
        BROAD_CLEAN_WEIGHTS,
    )
    broad_prediction = test[
        [
            "event_id",
            "extreme_time",
            "extreme_pos",
            "causal_region_id",
            "is_macro_first_sweep",
            "tp_1_h60",
            "target_clean_tp60",
            "target_adverse_path60",
            "mae_h60_pct",
            "reference_stage2_type",
            "reference_swing_event_id",
        ]
    ].copy()
    broad_prediction.insert(0, "expert_id", "B0_BROAD")
    broad_prediction.insert(0, "fold", fold.fold)
    broad_prediction["opportunity_raw_score"] = test_scores["global_opportunity"]
    broad_prediction["opportunity_probability"] = probabilities["global_opportunity"]
    broad_prediction["opportunity_rank"] = test_ranks["global_opportunity"]
    broad_prediction["clean_raw_score"] = test_scores["clean_path"]
    broad_prediction["clean_probability"] = probabilities["clean_path"]
    broad_prediction["clean_rank"] = test_ranks["clean_path"]
    broad_prediction["opportunity_clean_rank"] = broad_combined_test
    result["predictions"].append(broad_prediction)

    broad_policies = {
        "B0_GLOBAL_OPPORTUNITY": test_ranks["global_opportunity"],
        "B0_OPPORTUNITY_CLEAN": broad_combined_test,
    }
    for policy_id, rank in broad_policies.items():
        for top_pct in sorted(set(map(int, args.rank_top_pcts))):
            mask = top_rank_mask(rank, top_pct)
            selected = test.loc[mask].copy()
            selected["opportunity_score"] = rank[mask]
            rows, strong, trades = _evaluate_selected(
                bars=bars,
                fold=fold,
                expert_id="B0_BROAD",
                policy_id=policy_id,
                top_pct=top_pct,
                selected=selected,
                baseline=test,
                activation_frame=activation_frame,
                target_column=None,
                args=args,
            )
            result["scorecard"].extend(rows)
            result["strong_days"].extend(strong)
            result["trades"].extend(trades)

    # Sparse type-likelihood heads, with corrected confirmation-time purge.
    for spec in EXPERT_SPECS[:2]:
        target = f"target_{spec.expert_id}"
        train = add_episode_weight(
            frame.loc[split_type.model_fit_mask].copy(), positive_target_column=target
        )
        positives = int(train[target].sum())
        episodes = int(
            train.loc[train[target].astype(bool), "reference_swing_event_id"]
            .dropna()
            .astype(str)
            .nunique()
        )
        if positives < 20 or episodes < 10 or int((~train[target].astype(bool)).sum()) < 200:
            result["fits"].append(
                {
                    "fold": fold.fold,
                    "expert_id": spec.expert_id,
                    "head": "type_likelihood",
                    "status": "insufficient_type_supervision",
                    "model_fit_rows": int(len(train)),
                    "model_fit_positives": positives,
                    "model_fit_positive_episodes": episodes,
                }
            )
            continue
        try:
            fit, feature_diag = fit_resolved_binary_model(
                train,
                policy,
                requested_features=COMPACT_EXPERT_FEATURES,
                required_features=(str(spec.special_feature),),
                target_column=target,
                random_state=int(args.random_state),
                min_samples_leaf=int(args.model_min_samples_leaf),
                prediction_chunk_size=int(args.prediction_chunk_size),
            )
        except ExpertModelUnavailableError as exc:
            result["fits"].append(
                {
                    "fold": fold.fold,
                    "expert_id": spec.expert_id,
                    "head": "type_likelihood",
                    "status": "model_unavailable",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "model_fit_rows": int(len(train)),
                    "model_fit_positives": positives,
                    "model_fit_positive_episodes": episodes,
                }
            )
            continue
        feature_diag.insert(0, "head", "type_likelihood")
        feature_diag.insert(0, "expert_id", spec.expert_id)
        feature_diag.insert(0, "fold", fold.fold)
        result["features"].append(feature_diag)
        result["fits"].append(
            {
                "fold": fold.fold,
                "expert_id": spec.expert_id,
                "head": "type_likelihood",
                "status": "fitted",
                "model_fit_rows": int(len(train)),
                "model_fit_positives": positives,
                "model_fit_positive_episodes": episodes,
                **fit.diagnostics,
            }
        )
        type_policy_score = predict_binary_score_chunked(
            fit.model, policy, chunk_size=int(args.prediction_chunk_size)
        )
        type_test_score = predict_binary_score_chunked(
            fit.model, test, chunk_size=int(args.prediction_chunk_size)
        )
        ref = EmpiricalRankReference.fit(type_policy_score)
        type_policy_rank = ref.transform(type_policy_score)
        type_test_rank = ref.transform(type_test_score)
        try:
            special_policy_rank, special_test_rank, _ = signed_feature_rank(
                policy[str(spec.special_feature)],
                test[str(spec.special_feature)],
                direction=float(spec.special_direction),
            )
        except Exception as exc:
            # The predeclared condition is diagnostic only in Research 18 and
            # must not terminate an otherwise valid type/clean model fold.
            special_policy_rank = np.full(len(policy), np.nan, dtype=float)
            special_test_rank = np.full(len(test), np.nan, dtype=float)
            result["fits"].append(
                {
                    "fold": fold.fold,
                    "expert_id": spec.expert_id,
                    "head": "special_condition_rank",
                    "status": "unavailable",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )
        r17_policy_rank, r17_test_rank = _rank_pair(
            {
                "type": type_policy_rank,
                "opportunity": policy_ranks["global_opportunity"],
            },
            {
                "type": type_test_rank,
                "opportunity": test_ranks["global_opportunity"],
            },
            R17_TYPE_OPPORTUNITY_WEIGHTS,
        )
        clean_policy_rank, clean_test_rank = _rank_pair(
            {
                "type": type_policy_rank,
                "opportunity": policy_ranks["global_opportunity"],
                "clean": policy_ranks["clean_path"],
            },
            {
                "type": type_test_rank,
                "opportunity": test_ranks["global_opportunity"],
                "clean": test_ranks["clean_path"],
            },
            R18_CLEAN_WEIGHTS,
        )
        result["metrics"].append(
            _ranking_row(
                fold=fold.fold,
                expert_id=spec.expert_id,
                head="type_likelihood",
                target_column=target,
                test=test,
                score=type_test_score,
            )
        )
        result["coefficients"].append(
            extract_linear_coefficients(
                fit.model,
                fold=fold.fold,
                expert_id=spec.expert_id,
                head="type_likelihood",
            )
        )
        full_auc = float(result["metrics"][-1]["roc_auc"])
        result["placebos"].extend(
            _placebo_metrics(
                fold=fold.fold,
                expert_id=spec.expert_id,
                head="type_likelihood",
                train=train,
                test=test,
                target_column=target,
                feature_columns=fit.selected_features,
                real_test_score=type_test_score,
                random_state=int(args.random_state),
                min_samples_leaf=int(args.model_min_samples_leaf),
            )
        )
        result["ablations"].extend(
            _ablation_metrics(
                fold=fold.fold,
                expert_id=spec.expert_id,
                head="type_likelihood",
                fitted_model=fit,
                test=test,
                target_column=target,
                feature_columns=fit.selected_features,
                full_auc=full_auc,
            )
        )
        prediction = test[
            [
                "event_id",
                "extreme_time",
                "extreme_pos",
                "causal_region_id",
                "is_macro_first_sweep",
                target,
                "tp_1_h60",
                "target_clean_tp60",
                "target_adverse_path60",
                "mae_h60_pct",
                "reference_stage2_type",
                "reference_swing_event_id",
                str(spec.special_feature),
            ]
        ].copy()
        prediction.insert(0, "expert_id", spec.expert_id)
        prediction.insert(0, "fold", fold.fold)
        prediction["type_raw_score"] = type_test_score
        prediction["type_probability"] = predict_binary_probability_chunked(
            fit.model, test, chunk_size=int(args.prediction_chunk_size)
        )
        prediction["type_rank"] = type_test_rank
        prediction["special_rank"] = special_test_rank
        prediction["opportunity_rank"] = test_ranks["global_opportunity"]
        prediction["clean_rank"] = test_ranks["clean_path"]
        prediction["r17_type_opportunity_rank"] = r17_test_rank
        prediction["r18_clean_rank"] = clean_test_rank
        result["predictions"].append(prediction)
        for policy_id, rank in {
            f"{spec.expert_id[:2]}_R17_TYPE_OPPORTUNITY": r17_test_rank,
            f"{spec.expert_id[:2]}_R18_TYPE_OPPORTUNITY_CLEAN": clean_test_rank,
        }.items():
            for top_pct in sorted(set(map(int, args.rank_top_pcts))):
                mask = top_rank_mask(rank, top_pct)
                selected = test.loc[mask].copy()
                selected["opportunity_score"] = rank[mask]
                rows, strong, trades = _evaluate_selected(
                    bars=bars,
                    fold=fold,
                    expert_id=spec.expert_id,
                    policy_id=policy_id,
                    top_pct=top_pct,
                    selected=selected,
                    baseline=test,
                    activation_frame=activation_frame,
                    target_column=target,
                    args=args,
                )
                result["scorecard"].extend(rows)
                result["strong_days"].extend(strong)
                result["trades"].extend(trades)

    # First Sweep remains a control, not a gate.
    sweep_test = test[test["target_E3_FIRST_SWEEP_CONTROL"].astype(bool)].copy()
    if not sweep_test.empty:
        sweep_rank = broad_combined_test[test["target_E3_FIRST_SWEEP_CONTROL"].astype(bool).to_numpy()]
        for policy_id, rank, top_values in (
            ("E3_EVENT_ALL", np.full(len(sweep_test), 100.0), (100,)),
            ("E3_EVENT_OPPORTUNITY_CLEAN", sweep_rank, tuple(sorted(set(map(int, args.rank_top_pcts))))),
        ):
            for top_pct in top_values:
                mask = np.ones(len(sweep_test), dtype=bool) if top_pct == 100 else top_rank_mask(rank, top_pct)
                selected = sweep_test.loc[mask].copy()
                selected["opportunity_score"] = rank[mask]
                rows, strong, trades = _evaluate_selected(
                    bars=bars,
                    fold=fold,
                    expert_id="E3_FIRST_SWEEP_CONTROL",
                    policy_id=policy_id,
                    top_pct=int(top_pct),
                    selected=selected,
                    baseline=test,
                    activation_frame=activation_frame,
                    target_column=None,
                    args=args,
                )
                result["scorecard"].extend(rows)
                result["strong_days"].extend(strong)
                result["trades"].extend(trades)
    return result


def _concat(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    valid = [part for part in parts if isinstance(part, pd.DataFrame) and not part.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()


def _integrity_gate(
    *,
    label_audit: pd.DataFrame,
    truncation: pd.DataFrame,
    placebo: pd.DataFrame,
    htf_audit: pd.DataFrame,
) -> pd.DataFrame:
    valid = placebo[placebo["status"].eq("ok")].copy() if not placebo.empty else pd.DataFrame()
    permutations = (
        valid[valid["placebo_id"].astype(str).str.startswith("PERMUTE_")].copy()
        if not valid.empty
        else valid
    )

    def placebo_check(head: str, minimum_rows: int) -> dict[str, object]:
        sample = permutations[permutations["head"].eq(head)] if not permutations.empty else permutations
        auc = pd.to_numeric(sample.get("roc_auc"), errors="coerce").dropna() if len(sample) else pd.Series(dtype=float)
        median = float(auc.median()) if len(auc) else np.nan
        p90 = float(auc.quantile(0.90)) if len(auc) else np.nan
        passed = bool(
            len(auc) >= int(minimum_rows)
            and np.isfinite(median)
            and median <= 0.57
            and p90 <= 0.64
        )
        return {
            "check": f"{head}_permutation_auc_collapses",
            "passed": passed,
            "detail": f"rows={len(auc)} median_auc={median:.4f} p90_auc={p90:.4f}",
        }

    shifted = (
        valid[valid["placebo_id"].astype(str).str.startswith("SHIFT_")]
        if not valid.empty
        else valid
    )
    shifted_auc = pd.to_numeric(shifted.get("roc_auc"), errors="coerce").dropna() if len(shifted) else pd.Series(dtype=float)
    rows = [
        *label_audit.to_dict(orient="records"),
        {
            "check": "future_truncation_all_sampled_features_invariant",
            "passed": bool(truncation["passed"].fillna(False).all()) if not truncation.empty else False,
            "detail": f"samples={len(truncation)} failures={int((~truncation['passed'].fillna(False)).sum()) if not truncation.empty else -1}",
        },
        {
            "check": "htf_available_time_alignment",
            "passed": bool(htf_audit.empty or htf_audit["passed"].fillna(False).all()),
            "detail": f"rows={len(htf_audit)}",
        },
        placebo_check("type_likelihood", minimum_rows=18),
        placebo_check("global_opportunity", minimum_rows=9),
        placebo_check("clean_path", minimum_rows=9),
        {
            "check": "time_shift_placebo_reported_diagnostic",
            "passed": bool(len(shifted_auc) >= 6),
            "detail": (
                f"rows={len(shifted_auc)} median_auc={float(shifted_auc.median()) if len(shifted_auc) else np.nan:.4f}; "
                "time-shift persistence is diagnostic, not a hard 0.5 requirement"
            ),
        },
    ]
    return pd.DataFrame(rows)


def _annotate_integrity_audit(audit: pd.DataFrame) -> pd.DataFrame:
    """Classify final audits without turning research findings into crashes.

    Causal/contract checks and statistical placebo diagnostics both block a
    trading conclusion when they fail.  They differ only in interpretation:
    a placebo failure is evidence to investigate, not a Python runtime error.
    All final reports must therefore be written before the run returns.
    """

    required = {"check", "passed", "detail"}
    missing = sorted(required.difference(audit.columns))
    if missing:
        raise RuntimeError(f"integrity audit missing columns: {missing}")
    out = audit.copy()
    out["check"] = out["check"].astype(str)
    out["passed"] = out["passed"].fillna(False).astype(bool)
    diagnostic = (
        out["check"].str.contains("permutation_auc_collapses", regex=False)
        | out["check"].eq("time_shift_placebo_reported_diagnostic")
    )
    out["severity"] = np.where(diagnostic, "diagnostic", "blocking")
    out["blocks_trading_conclusion"] = ~out["passed"]
    return out


def _integrity_outcome(audit: pd.DataFrame) -> tuple[str, list[str]]:
    annotated = _annotate_integrity_audit(audit)
    failed = annotated.loc[~annotated["passed"], ["check", "severity"]]
    if failed.empty:
        return "passed", []
    failed_checks = [
        f"{row.check}({row.severity})" for row in failed.itertuples(index=False)
    ]
    if failed["severity"].eq("blocking").any():
        return "causal_or_contract_failed", failed_checks
    return "diagnostic_failed", failed_checks


def _apply_integrity_status_to_policies(
    policies: pd.DataFrame,
    *,
    integrity_status: str,
    failed_checks: Sequence[str],
) -> pd.DataFrame:
    out = policies.copy()
    if "final_policy_status" not in out.columns:
        out["final_policy_status"] = pd.Series(dtype="string")
    out["pre_integrity_policy_status"] = out["final_policy_status"].astype("string")
    out["integrity_status"] = str(integrity_status)
    out["integrity_failed_checks"] = "; ".join(map(str, failed_checks))
    if integrity_status != "passed" and not out.empty:
        out["final_policy_status"] = "integrity_blocked"
    return out


def _build_summary(
    ranking: pd.DataFrame,
    placebo: pd.DataFrame,
    crossfold: pd.DataFrame,
    robust_policies: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    lines = [
        "# Research 18 Summary",
        "",
        "## Scope",
        "- Frozen Research 17 broad candidates, C3-D/C3-E type heads, and TP +1% opportunity target.",
        "- Corrected original-type label availability uses the later of bridge-window end and historical +1% confirmation availability.",
        "- New research layers are one clean-path head, one causal activation rule, and frozen TP1% exit structures.",
        "- Top20/30/40 and hard frequency gates remain mandatory.",
        "",
        "## Integrity",
    ]
    integrity_status, failed_checks = _integrity_outcome(audit)
    lines.append(f"- Overall status: {integrity_status}.")
    if failed_checks:
        lines.append(f"- Failed checks: {'; '.join(failed_checks)}.")
    for row in audit.itertuples(index=False):
        severity = getattr(row, "severity", "unclassified")
        lines.append(
            f"- {row.check}: {'PASS' if bool(row.passed) else 'FAIL'} "
            f"[{severity}] — {row.detail}"
        )
    if not ranking.empty:
        lines.extend(["", "## OOS ranking snapshot"])
        for row in ranking.sort_values(["expert_id", "head", "fold"]).itertuples(index=False):
            lines.append(
                f"- {row.fold} {row.expert_id}/{row.head}: AUC={row.roc_auc:.4f}, "
                f"AP={row.average_precision:.4f}, episode_AUC={row.episode_roc_auc:.4f}."
            )
    if not placebo.empty:
        valid = placebo[placebo["status"].eq("ok")]
        lines.extend(
            [
                "",
                "## Placebo",
                f"- Valid placebo fits: {len(valid)}; median AUC={valid['roc_auc'].median() if len(valid) else np.nan:.4f}.",
            ]
        )
    if not crossfold.empty:
        candidates = crossfold[crossfold["research_candidate_status"].eq("candidate")]
        robust = (
            robust_policies[robust_policies["final_policy_status"].eq("candidate")]
            if not robust_policies.empty
            else robust_policies
        )
        lines.extend(
            [
                "",
                "## Trading result",
                f"- Cross-fold policy rows: {len(crossfold)}; preliminary candidates: {len(candidates)}; robust candidates: {len(robust)}.",
                "- Final support additionally requires Top30+neighbor stability, 1.5x cost support, positive results after deleting the strongest 10 days, and 1-bar delay support for the TP1_TIME60 baseline.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- High ROC-AUC alone is not accepted as an edge; PR-AUC, episode metrics, placebo collapse, frequency, path quality, and costed execution are evaluated separately.",
            "- If the integrity gate fails, all trading results are research-invalid even when returns look attractive.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_research(args: argparse.Namespace) -> Path:
    _validate_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    historical, compatibility, hierarchy_audit = _R17._load_original_reports(args)
    _write_csv(compatibility, out_dir / "01_frozen_report_compatibility.csv")
    _write_csv(typology_inventory(historical), out_dir / "02_original_typology_inventory.csv")
    bars = _R16R.load_bars(args)
    _write_csv(validate_trade_bar_fields(bars), out_dir / "03_trade_bar_field_coverage.csv")
    frame, activation_frame, diagnostics = _build_dataset(args, bars, historical)
    _write_csv(diagnostics["gate_coverage"], out_dir / "04_candidate_gate_coverage.csv")
    _write_csv(diagnostics["first_sweep"], out_dir / "05_first_sweep_diagnostics.csv")
    _write_csv(diagnostics["region_summary"], out_dir / "06_region_build_summary.csv")
    _write_csv(diagnostics["feature_dictionary"], out_dir / "07_online_feature_dictionary.csv")
    _write_csv(diagnostics["alignment_audit"], out_dir / "08_htf_available_time_audit.csv")
    _write_csv(diagnostics["label_availability"], out_dir / "09_type_label_availability_audit.csv")
    _write_csv(diagnostics["activation_summary"], out_dir / "10_activation_build_summary.csv")
    truncation = _future_truncation_audit(
        bars,
        frame,
        diagnostics["selected_meta"],
        samples=int(args.future_truncation_samples),
    )
    _write_csv(truncation, out_dir / "11_future_truncation_feature_audit.csv")

    all_results: dict[str, list] = {
        "split": [], "incidence": [], "features": [], "fits": [], "metrics": [],
        "coefficients": [], "placebos": [], "ablations": [], "predictions": [],
        "scorecard": [], "strong_days": [], "trades": [],
    }
    for fold in folds_for_end_date(args.end_date):
        fold_result = _run_fold(
            bars=bars,
            frame=frame,
            activation_frame=activation_frame,
            fold=fold,
            args=args,
        )
        for key, values in fold_result.items():
            all_results[key].extend(values)

    print("[stage] aggregate walk-forward outputs", flush=True)
    split = pd.DataFrame(all_results["split"])
    incidence = pd.DataFrame(all_results["incidence"])
    features = _concat(all_results["features"])
    fits = pd.DataFrame(all_results["fits"])
    ranking = pd.DataFrame(all_results["metrics"])
    coefficients = _concat(all_results["coefficients"])
    coefficient_stability = coefficient_stability_summary(coefficients)
    placebo = pd.DataFrame(all_results["placebos"])
    ablations = pd.DataFrame(all_results["ablations"])
    predictions = _concat(all_results["predictions"])
    scorecard = pd.DataFrame(all_results["scorecard"])
    strong_days = pd.DataFrame(all_results["strong_days"])
    trades = _concat(all_results["trades"])
    crossfold = _crossfold_summary(scorecard)
    neighborhood = _neighborhood_summary(crossfold)
    robust_policies = _robust_policy_summary(
        crossfold, neighborhood, scorecard, strong_days
    )

    print("[stage] write research tables", flush=True)
    _write_csv(split, out_dir / "12_walkforward_boundaries.csv")
    _write_csv(incidence, out_dir / "13_target_incidence.csv")
    _write_csv(features, out_dir / "14_feature_conditioning_diagnostics.csv")
    _write_csv(fits, out_dir / "15_model_fit_and_resolution.csv")
    _write_csv(ranking, out_dir / "16_model_ranking_and_episode_metrics.csv")
    _write_csv(placebo, out_dir / "17_placebo_auc_audit.csv")
    _write_csv(ablations, out_dir / "18_feature_group_ablation.csv")
    _write_csv(coefficients, out_dir / "19_linear_coefficients.csv")
    _write_csv(coefficient_stability, out_dir / "20_coefficient_stability.csv")
    _write_csv(scorecard, out_dir / "21_path_entry_exit_execution_scorecard.csv")
    _write_csv(crossfold, out_dir / "22_crossfold_policy_summary.csv")
    _write_csv(neighborhood, out_dir / "23_policy_neighborhood_summary.csv")
    _write_csv(strong_days, out_dir / "24_delete_strong_days_stress.csv")
    if bool(args.write_full_predictions):
        _write_csv(predictions, out_dir / "26_oos_full_predictions.csv")
    elif not predictions.empty:
        sample_parts = []
        for _, group in predictions.groupby(["fold", "expert_id"], sort=True):
            sample_parts.append(group.sample(min(2_000, len(group)), random_state=42))
        _write_csv(_concat(sample_parts), out_dir / "26_oos_prediction_sample.csv")
    if not trades.empty:
        sample = trades if len(trades) <= 20_000 else trades.sample(20_000, random_state=42)
        _write_csv(sample.sort_values(["entry_time", "event_id"], kind="mergesort"), out_dir / "27_execution_trade_sample.csv")

    print("[stage] final causal, placebo, frequency, and robustness audit", flush=True)
    audit = pd.concat(
        [
            hierarchy_audit,
            _integrity_gate(
                label_audit=diagnostics["label_availability"],
                truncation=truncation,
                placebo=placebo,
                htf_audit=diagnostics["alignment_audit"],
            ),
            pd.DataFrame(
                [
                    {
                        "check": "closed_bar_signal_next_open_entry",
                        "passed": bool((pd.to_datetime(frame["entry_time"]) > pd.to_datetime(frame["extreme_time"])).all()),
                        "detail": f"base_rows={len(frame):,} activation_rows={len(activation_frame):,}",
                    },
                    {
                        "check": "only_top20_top30_top40_and_explicit_all_control",
                        "passed": bool(set(scorecard["top_pct"].astype(int).unique()).issubset({20, 30, 40, 100})),
                        "detail": "no Top10/Top5 search",
                    },
                    {
                        "check": "frequency_guard_present",
                        "passed": bool({"frequency_guard_passed", "annualized_executable_trades"}.issubset(scorecard.columns)),
                        "detail": f"minimum annualized executable={float(args.minimum_annualized_executable_trades):.1f}",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    audit = _annotate_integrity_audit(audit)
    integrity_status, failed_checks = _integrity_outcome(audit)
    robust_policies = _apply_integrity_status_to_policies(
        robust_policies,
        integrity_status=integrity_status,
        failed_checks=failed_checks,
    )
    _write_csv(robust_policies, out_dir / "25_robust_policy_decision.csv")
    _write_csv(audit, out_dir / "28_causal_integrity_frequency_audit.csv")
    if integrity_status == "passed":
        print("[audit] status=PASS all final integrity checks passed", flush=True)
    else:
        print(
            f"[audit] status={integrity_status} failed_checks={' | '.join(failed_checks)}",
            flush=True,
        )
        print(
            "[audit] report will be completed; all trading candidates are marked integrity_blocked",
            flush=True,
        )

    manifest = {
        "script": SCRIPT_NAME,
        "version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "target_move_pct": FROZEN_TARGET_MOVE_PCT,
        "forward_horizon_bars": FROZEN_HORIZON_BARS,
        "research_start": args.start_date,
        "research_end": args.end_date,
        "candidate_count": int(len(frame)),
        "activation_labeled_count": int(len(activation_frame)),
        "rank_top_pcts": [20, 30, 40],
        "feature_columns": list(COMPACT_EXPERT_FEATURES),
        "r17_type_opportunity_weights": R17_TYPE_OPPORTUNITY_WEIGHTS,
        "r18_clean_weights": R18_CLEAN_WEIGHTS,
        "broad_clean_weights": BROAD_CLEAN_WEIGHTS,
        "clean_target": "TP +1% within 60 closed bars and MAE-before-TP <=0.50%",
        "activation": "first closed-bar state within 10 bars with >=1 bar since region low, positive response, close above previous, and >=10bp reclaim",
        "exit_policies": [policy.__dict__ for policy in FROZEN_EXIT_POLICIES],
        "fees": "0.11% roundtrip at 1x",
        "slippage": "0.04% roundtrip at 1x",
        "integrity": {
            "status": integrity_status,
            "failed_checks": failed_checks,
            "trading_conclusion_allowed": integrity_status == "passed",
            "true_type_label_end": "max(candidate+15m, historical confirmation_available_time)",
            "placebos": [spec.__dict__ for spec in FROZEN_PLACEBO_SPECS],
            "future_truncation_samples": int(args.future_truncation_samples),
        },
    }
    (out_dir / "00_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "29_RESEARCH_SUMMARY.md").write_text(
        _build_summary(ranking, placebo, crossfold, robust_policies, audit), encoding="utf-8"
    )
    print("[stage] finalize review pack", flush=True)
    result = finalize_research_report(
        out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE
    )
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={result.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
