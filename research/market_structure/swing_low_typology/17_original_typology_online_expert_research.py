#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Research 17: causal online experts for original C3-D/C3-E Swing Low types.

Research 16R restored the original Research 01-03 typology and found two
non-First-Sweep branches worth testing online:

* C3-D candidates with price-response failure;
* C3-E candidates with early recovery.

This research trains sparse type-likelihood heads on *all* online candidates,
then combines them with a type-conditional +1% opportunity head and the one
predeclared condition from Research 16R.  First Sweep remains a control branch.
No Top threshold narrower than Top20 is permitted.  Candidate quality and trade
frequency are reported together; low-frequency high-looking results fail the
research gate.
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
    executable_trade_set,
    execution_metrics,
    strongest_day_stress,
)
from research.market_structure.swing_low_typology.common.broad_reversal_mechanisms import (  # noqa: E402
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
from research.market_structure.swing_low_typology.common.original_typology_bridge import (  # noqa: E402
    build_historical_typology_table,
    map_candidates_to_future_typology,
    typology_inventory,
)
from research.market_structure.swing_low_typology.common.original_typology_experts import (  # noqa: E402
    COMPACT_EXPERT_FEATURES,
    EXPERT_SPECS,
    EmpiricalRankReference,
    ExpertModelUnavailableError,
    ExpertRankUnavailableError,
    add_episode_weight,
    binary_ranking_metrics,
    build_expert_targets,
    build_fold_split,
    combine_component_ranks,
    crossfold_policy_summary,
    expert_spec_table,
    fit_resolved_binary_model,
    folds_for_end_date,
    frequency_guard,
    months_in_fold,
    policy_neighborhood_summary,
    policy_path_metrics,
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
from research.market_structure.swing_low_typology.common.walkforward_reversal import (  # noqa: E402
    build_broad_candidate_regions,
)

_R16R = importlib.import_module(
    "research.market_structure.swing_low_typology.16r_original_typology_online_bridge_audit"
)

SCRIPT_NAME = "17_original_typology_online_expert_research"
SCRIPT_VERSION = "1.0.3"
EXPERIMENT_ID = "ETH_1M_ORIGINAL_TYPOLOGY_ONLINE_EXPERTS_17"
EDGE_ID = "RESEARCH_ONLY_ETH_ORIGINAL_TYPOLOGY_ONLINE_EXPERTS"
TITLE = "ETH Original Swing Low Typology Online Expert Research 17"
DEFAULT_OUT_DIR = (
    "data/reports/research/market_structure/swing_low_typology/"
    "17_original_typology_online_experts"
)
DEFAULT_STAGE1_DIR = _R16R.DEFAULT_STAGE1_DIR
DEFAULT_STAGE2_DIR = _R16R.DEFAULT_STAGE2_DIR
DEFAULT_STAGE3_DIR = _R16R.DEFAULT_STAGE3_DIR

FROZEN_TARGET_MOVE_PCT = 1.0
FROZEN_FORWARD_HORIZON_BARS = 60


PRIMARY_POLICY_IDS = {
    "B0_BROAD_ALL",
    "B0_GLOBAL_OPPORTUNITY",
    "E1_TYPE_GLOBAL_OPPORTUNITY_SPECIAL",
    "E2_TYPE_GLOBAL_OPPORTUNITY_SPECIAL",
    "E3_EVENT_ALL",
    "E3_EVENT_OPPORTUNITY",
}

# The original-type labels are sparse by construction.  Research 17 therefore
# uses one deployable opportunity head trained on the full candidate universe
# and shares it across E1/E2.  A separate type-conditional TP model is not fit:
# the earliest fold contains only dozens of positive-type training rows and
# would be an overfit rather than a valid expert head.
OPPORTUNITY_HEAD_SOURCE = "B0_BROAD.global_opportunity"
MIN_TYPE_MODEL_POSITIVE_ROWS = 20
MIN_TYPE_MODEL_NEGATIVE_ROWS = 200
MIN_TYPE_MODEL_POSITIVE_EPISODES = 10


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train causal C3-D/C3-E online expert models with hard frequency guardrails.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"%(prog)s v{SCRIPT_VERSION}")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--stage1-report-dir", default=DEFAULT_STAGE1_DIR)
    p.add_argument("--stage2-report-dir", default=DEFAULT_STAGE2_DIR)
    p.add_argument("--stage3-report-dir", default=DEFAULT_STAGE3_DIR)
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

    p.add_argument("--minimum-annualized-raw-events", type=float, default=600.0)
    p.add_argument("--minimum-annualized-executable-trades", type=float, default=240.0)
    p.add_argument("--minimum-raw-events-per-fold", type=int, default=100)
    p.add_argument("--cost-multipliers", nargs="+", type=float, default=[1.0, 1.5, 2.0])
    p.add_argument("--entry-delays", nargs="+", type=int, default=[0, 1, 3])
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
        raise ValueError("Research 17 is frozen to OKX ETH-USDT-SWAP 1m")
    if int(args.reference_maximum_lead_bars) != 15:
        raise ValueError("Research 17 freezes the original-type bridge at 15 bars")
    if not np.isclose(float(args.reference_price_tolerance_bp), 75.0):
        raise ValueError("Research 17 freezes the original-type bridge price zone at 75bp")
    top_pcts = tuple(sorted(set(int(value) for value in args.rank_top_pcts)))
    if top_pcts != (20, 30, 40):
        raise ValueError("Research 17 permits only broad Top20/Top30/Top40 policies")
    costs = tuple(sorted(set(float(value) for value in args.cost_multipliers)))
    if costs != (1.0, 1.5, 2.0):
        raise ValueError("Research 17 freezes cost stress at 1x/1.5x/2x")
    delays = tuple(sorted(set(int(value) for value in args.entry_delays)))
    if delays != (0, 1, 3):
        raise ValueError("Research 17 freezes entry delays at 0/1/3 bars")
    validate_feature_names(COMPACT_EXPERT_FEATURES)


def _original_report_validation_args(args: argparse.Namespace) -> argparse.Namespace:
    """Build the explicit Research 16R compatibility contract.

    Research 17 intentionally fixes the original Swing Low label at +1% within
    60 closed bars.  Passing the whole Research 17 namespace directly into the
    Research 16R validator previously leaked an implicit interface dependency:
    17 did not define ``target_move_pct`` or ``forward_horizon_bars`` and failed
    before reading any report.  Keep the adapter explicit so future argument
    changes fail here rather than during a multi-year run.
    """

    required = (
        "symbol",
        "timeframe",
        "start_date",
        "end_date",
        "stage1_report_dir",
        "stage2_report_dir",
        "stage3_report_dir",
    )
    missing = [name for name in required if not hasattr(args, name)]
    if missing:
        raise RuntimeError(f"Research 17 report-validation adapter missing fields: {missing}")
    return argparse.Namespace(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        stage1_report_dir=args.stage1_report_dir,
        stage2_report_dir=args.stage2_report_dir,
        stage3_report_dir=args.stage3_report_dir,
        target_move_pct=FROZEN_TARGET_MOVE_PCT,
        forward_horizon_bars=FROZEN_FORWARD_HORIZON_BARS,
    )


def _load_original_reports(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_args = _original_report_validation_args(args)
    stage1_dir, stage2_dir, stage3_dir, compatibility = _R16R._validate_original_reports(
        validation_args
    )
    historical, hierarchy_audit = _R16R._load_historical(stage1_dir, stage2_dir, stage3_dir)
    return historical, compatibility, hierarchy_audit


def _build_dataset(
    args: argparse.Namespace,
    bars: pd.DataFrame,
    historical: pd.DataFrame,
) -> tuple[pd.DataFrame, Mapping[str, pd.DataFrame]]:
    print("[stage] broad causal candidate universe", flush=True)
    candidate_config = CandidateGateConfig(
        lookback=int(args.candidate_lookback_bars),
        horizon=60,
        new_low_window=int(args.candidate_new_low_window),
        near_floor_window=int(args.candidate_near_floor_window),
        position_window=int(args.candidate_position_window),
        near_floor_tolerance_bp=float(args.candidate_near_floor_tolerance_bp),
        max_position_in_range=float(args.candidate_max_position_in_range),
    )
    raw_gate, gate_coverage = build_online_candidate_events(
        bars,
        research_start=pd.Timestamp(args.start_date),
        research_end_exclusive=_end_exclusive(args.end_date),
        config=candidate_config,
    )

    print("[stage] First Sweep control branch", flush=True)
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
    raw_union = merge_macro_first_sweep_candidates(bars, raw_gate, first_sweep.decisions)

    print("[stage] causal regions and spaced online events", flush=True)
    region_result = build_broad_candidate_regions(
        bars,
        raw_union,
        max_gap_bars=int(args.region_max_gap_bars),
        max_region_bars=int(args.region_max_bars),
        retest_tolerance_bp=float(args.region_retest_tolerance_bp),
        show_progress=True,
    )
    selected_meta = select_broad_region_events(
        region_result.frame,
        cooldown_bars=int(args.broad_cooldown_bars),
    )

    print("[stage] causal online feature matrix", flush=True)
    feature_result = build_reversal_candidate_features(
        bars,
        selected_meta,
        include_session=True,
        include_htf=True,
        show_progress=True,
    )
    if not feature_result.alignment_audit.empty and not feature_result.alignment_audit["passed"].all():
        raise RuntimeError("HTF available-time audit failed")
    frame = feature_result.frame.copy()
    frame["is_macro_first_sweep"] = frame.get("is_macro_first_sweep", False)
    frame["is_macro_first_sweep"] = frame["is_macro_first_sweep"].fillna(False).astype(bool)

    print("[stage] next-open +1% close-only labels", flush=True)
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
        frame,
        bridge_maximum_lead_bars=int(args.reference_maximum_lead_bars),
    )
    missing_features = sorted(set(COMPACT_EXPERT_FEATURES).difference(frame.columns))
    if missing_features:
        raise RuntimeError(f"predeclared Research 17 features missing: {missing_features}")

    diagnostics = {
        "gate_coverage": gate_coverage,
        "first_sweep": first_sweep.diagnostics,
        "region_summary": region_result.summary,
        "region_dictionary": region_result.dictionary,
        "feature_dictionary": feature_result.dictionary,
        "alignment_audit": feature_result.alignment_audit,
    }
    return frame, diagnostics


def _model_metric_row(
    *,
    fold: str,
    expert_id: str,
    head: str,
    target_column: str,
    test: pd.DataFrame,
    score: np.ndarray,
) -> dict[str, object]:
    metrics = binary_ranking_metrics(test[target_column], score)
    return {
        "fold": fold,
        "expert_id": expert_id,
        "head": head,
        "target_column": target_column,
        "test_rows": int(len(test)),
        "test_positives": int(test[target_column].fillna(False).astype(bool).sum()),
        **metrics,
        **{f"score_{key}": value for key, value in raw_score_resolution(score).items()},
    }


def _execution_stress_specs(policy_id: str) -> tuple[tuple[float, int], ...]:
    base = [(1.0, 0)]
    if policy_id in PRIMARY_POLICY_IDS:
        base.extend([(1.5, 0), (2.0, 0), (1.0, 1), (1.0, 3)])
    return tuple(base)


def _evaluate_policy(
    *,
    bars: pd.DataFrame,
    fold_name: str,
    expert_id: str,
    policy_id: str,
    top_pct: int,
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
    expert_pool: pd.DataFrame,
    target_column: str | None,
    months: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    strong_rows: list[dict[str, object]] = []
    trade_parts: list[pd.DataFrame] = []
    path = policy_path_metrics(selected, baseline, horizon=60)
    expert_pool_tp = float(expert_pool["tp_1_h60"].astype(bool).mean()) if len(expert_pool) else np.nan
    selected_tp = float(selected["tp_1_h60"].astype(bool).mean()) if len(selected) else np.nan
    pool_uplift = (selected_tp - expert_pool_tp) * 100.0 if np.isfinite(selected_tp) else np.nan
    capture = (
        target_capture_metrics(selected, baseline, target_column=target_column)
        if target_column is not None
        else {}
    )

    primary_executable_count: int | None = None
    for cost_multiplier, entry_delay in _execution_stress_specs(policy_id):
        costs = CloseTargetCostSpec(cost_multiplier=float(cost_multiplier))
        trades, counts = executable_trade_set(
            bars,
            selected,
            horizon_bars=60,
            costs=costs,
            entry_delay_bars=int(entry_delay),
        )
        trades = trades.copy()
        if not trades.empty:
            trades["fold"] = fold_name
            trades["expert_id"] = expert_id
            trades["policy_id"] = policy_id
            trades["top_pct"] = int(top_pct)
            trades["cost_multiplier"] = float(cost_multiplier)
            trades["entry_delay_bars"] = int(entry_delay)
            trade_parts.append(trades)
        metrics = execution_metrics(
            trades,
            months=months,
            counts=counts,
            capital_fraction=float(args.capital_fraction),
        )
        if float(cost_multiplier) == 1.0 and int(entry_delay) == 0:
            primary_executable_count = int(len(trades))
        guard = frequency_guard(
            raw_events=int(len(selected)),
            executable_trades=int(len(trades)),
            months=months,
            minimum_annualized_raw_events=float(args.minimum_annualized_raw_events),
            minimum_annualized_executable_trades=float(args.minimum_annualized_executable_trades),
            minimum_raw_events=int(args.minimum_raw_events_per_fold),
        )
        net_expectancy_bps = float(metrics.get("net_expectancy_bps", np.nan))
        throughput = (
            net_expectancy_bps * float(len(trades)) / max(1, int(months))
            if np.isfinite(net_expectancy_bps)
            else np.nan
        )
        rows.append(
            {
                "fold": fold_name,
                "expert_id": expert_id,
                "policy_id": policy_id,
                "top_pct": int(top_pct),
                "cost_multiplier": float(cost_multiplier),
                "entry_delay_bars": int(entry_delay),
                "broad_test_candidates": int(len(baseline)),
                "selected_share_of_broad_candidates": float(len(selected) / max(1, len(baseline))),
                "expert_pool_events": int(len(expert_pool)),
                "expert_pool_tp60_rate": expert_pool_tp,
                "selected_vs_expert_pool_tp_uplift_pp": pool_uplift,
                **path,
                **capture,
                **guard,
                "net_edge_throughput_bps_per_month": throughput,
                **metrics,
            }
        )
        if float(cost_multiplier) == 1.0 and int(entry_delay) == 0 and policy_id in PRIMARY_POLICY_IDS:
            for remove_days in (5, 10):
                stressed = strongest_day_stress(
                    trades,
                    remove_days=remove_days,
                    months=months,
                    raw_signals=int(len(selected)),
                    capital_fraction=float(args.capital_fraction),
                )
                strong_rows.append(
                    {
                        "fold": fold_name,
                        "expert_id": expert_id,
                        "policy_id": policy_id,
                        "top_pct": int(top_pct),
                        "remove_strongest_days": remove_days,
                        **stressed,
                    }
                )
    if primary_executable_count is None:
        raise RuntimeError("primary execution row was not generated")
    return rows, strong_rows, trade_parts


def _policy_rank(
    policy_components: Mapping[str, np.ndarray],
    test_components: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    _, test_rank, _ = combine_component_ranks(
        policy_components,
        test_components,
        weights=weights,
    )
    policy_rank, _, _ = combine_component_ranks(
        policy_components,
        policy_components,
        weights=weights,
    )
    return policy_rank, test_rank


def _type_supervision_summary(train: pd.DataFrame, target_column: str) -> dict[str, object]:
    """Validate sparse one-vs-rest type supervision without hiding scarcity."""

    if target_column not in train.columns:
        raise RuntimeError(f"type target column missing: {target_column}")
    target = train[target_column]
    missing = int(target.isna().sum())
    if missing:
        raise RuntimeError(
            f"type target {target_column} contains {missing} NA rows before fold fit"
        )
    positive = target.astype(bool)
    positive_rows = int(positive.sum())
    negative_rows = int((~positive).sum())
    reference = train.get(
        "reference_swing_event_id",
        pd.Series(pd.NA, index=train.index, dtype="string"),
    ).astype("string")
    positive_episodes = int(reference[positive & reference.notna()].nunique())
    checks = {
        "positive_rows_passed": positive_rows >= MIN_TYPE_MODEL_POSITIVE_ROWS,
        "negative_rows_passed": negative_rows >= MIN_TYPE_MODEL_NEGATIVE_ROWS,
        "positive_episodes_passed": positive_episodes >= MIN_TYPE_MODEL_POSITIVE_EPISODES,
    }
    return {
        "rows": int(len(train)),
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "positive_episodes": positive_episodes,
        **checks,
        "passed": bool(all(checks.values())),
    }


def _conditional_tp_diagnostic(
    frame: pd.DataFrame,
    *,
    target_column: str,
) -> dict[str, object]:
    """Report type-conditional TP sample size; never fit a sparse second head."""

    subset = frame[frame[target_column].astype(bool)]
    classes = int(subset["target_tp60"].astype(bool).nunique()) if len(subset) else 0
    positive_episodes = int(
        subset.get(
            "reference_swing_event_id",
            pd.Series(pd.NA, index=subset.index, dtype="string"),
        )
        .astype("string")
        .dropna()
        .nunique()
    )
    return {
        "model_fit_rows": int(len(subset)),
        "model_fit_positives": int(subset["target_tp60"].astype(bool).sum()) if len(subset) else 0,
        "model_fit_classes": classes,
        "positive_type_episodes": positive_episodes,
        "status": "not_fit_shared_global_opportunity",
        "opportunity_head_source": OPPORTUNITY_HEAD_SOURCE,
    }


def _run_fold(
    *,
    bars: pd.DataFrame,
    frame: pd.DataFrame,
    fold,
    args: argparse.Namespace,
) -> Mapping[str, list]:
    fold_name = fold.fold
    print(f"[fold] {fold_name}", flush=True)
    split_tp = build_fold_split(
        frame,
        fold,
        policy_days=int(args.policy_window_days),
        fit_label_end_column="label_end_time",
    )
    split_type = build_fold_split(
        frame,
        fold,
        policy_days=int(args.policy_window_days),
        fit_label_end_column="type_label_end_time",
    )
    policy = frame.loc[split_tp.policy_mask].reset_index(drop=True)
    test = frame.loc[split_tp.test_mask].reset_index(drop=True)
    if len(policy) < 500 or len(test) < 500:
        raise RuntimeError(f"{fold_name} has insufficient policy/test candidates")
    months = months_in_fold(fold)

    feature_rows: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    model_metric_rows: list[dict[str, object]] = []
    prediction_parts: list[pd.DataFrame] = []
    scorecard_rows: list[dict[str, object]] = []
    strong_rows: list[dict[str, object]] = []
    trade_parts: list[pd.DataFrame] = []
    incidence_rows: list[dict[str, object]] = []

    for target in ["target_tp60", *[f"target_{spec.expert_id}" for spec in EXPERT_SPECS]]:
        incidence_rows.append(
            {
                "fold": fold_name,
                "target": target,
                "policy_rows": int(len(policy)),
                "policy_positives": int(policy[target].fillna(False).astype(bool).sum()),
                "test_rows": int(len(test)),
                "test_positives": int(test[target].fillna(False).astype(bool).sum()),
            }
        )

    # Broad global opportunity baseline.
    global_train = add_episode_weight(frame.loc[split_tp.model_fit_mask].copy())
    global_fit, feature_diag = fit_resolved_binary_model(
        global_train,
        policy,
        requested_features=COMPACT_EXPERT_FEATURES,
        required_features=("sell_pressure_absorption_30", "region_rebound_from_low"),
        target_column="target_tp60",
        random_state=int(args.random_state),
        min_samples_leaf=int(args.model_min_samples_leaf),
        prediction_chunk_size=int(args.prediction_chunk_size),
    )
    feature_diag.insert(0, "head", "global_opportunity")
    feature_diag.insert(0, "expert_id", "B0_BROAD")
    feature_diag.insert(0, "fold", fold_name)
    feature_rows.append(feature_diag)
    fit_rows.append(
        {
            "fold": fold_name,
            "expert_id": "B0_BROAD",
            "head": "global_opportunity",
            "model_fit_rows": int(len(global_train)),
            "model_fit_positives": int(global_train["target_tp60"].sum()),
            "policy_rows": int(len(policy)),
            **global_fit.diagnostics,
        }
    )
    global_policy_score = predict_binary_score_chunked(
        global_fit.model, policy, chunk_size=int(args.prediction_chunk_size)
    )
    global_test_score = predict_binary_score_chunked(
        global_fit.model, test, chunk_size=int(args.prediction_chunk_size)
    )
    global_policy_probability = predict_binary_probability_chunked(
        global_fit.model, policy, chunk_size=int(args.prediction_chunk_size)
    )
    global_test_probability = predict_binary_probability_chunked(
        global_fit.model, test, chunk_size=int(args.prediction_chunk_size)
    )
    global_reference = EmpiricalRankReference.fit(global_policy_score)
    global_policy_rank = global_reference.transform(global_policy_score)
    global_test_rank = global_reference.transform(global_test_score)
    model_metric_rows.append(
        _model_metric_row(
            fold=fold_name,
            expert_id="B0_BROAD",
            head="global_opportunity",
            target_column="target_tp60",
            test=test,
            score=global_test_score,
        )
    )
    broad_prediction = test[
        [
            "event_id",
            "extreme_time",
            "extreme_pos",
            "is_macro_first_sweep",
            "tp_1_h60",
            "mae_h60_pct",
            "reference_stage2_type",
            "reference_swing_event_id",
        ]
    ].copy()
    broad_prediction.insert(0, "expert_id", "B0_BROAD")
    broad_prediction.insert(0, "fold", fold_name)
    broad_prediction["opportunity_raw_score"] = global_test_score
    broad_prediction["opportunity_probability"] = global_test_probability
    broad_prediction["opportunity_rank"] = global_test_rank
    prediction_parts.append(broad_prediction)
    broad_all = test.copy()
    broad_all["opportunity_score"] = global_test_rank
    rows, stressed, trades = _evaluate_policy(
        bars=bars,
        fold_name=fold_name,
        expert_id="B0_BROAD",
        policy_id="B0_BROAD_ALL",
        top_pct=100,
        selected=broad_all,
        baseline=test,
        expert_pool=test,
        target_column=None,
        months=months,
        args=args,
    )
    scorecard_rows.extend(rows)
    strong_rows.extend(stressed)
    trade_parts.extend(trades)
    for top_pct in sorted(set(int(value) for value in args.rank_top_pcts)):
        mask = top_rank_mask(global_test_rank, top_pct)
        selected = test.loc[mask].copy()
        selected["opportunity_score"] = global_test_rank[mask]
        rows, stressed, trades = _evaluate_policy(
            bars=bars,
            fold_name=fold_name,
            expert_id="B0_BROAD",
            policy_id="B0_GLOBAL_OPPORTUNITY",
            top_pct=top_pct,
            selected=selected,
            baseline=test,
            expert_pool=test,
            target_column=None,
            months=months,
            args=args,
        )
        scorecard_rows.extend(rows)
        strong_rows.extend(stressed)
        trade_parts.extend(trades)

    # Original C3-D / C3-E experts.  The type head is sparse one-vs-rest,
    # while the +1% opportunity head is the shared full-universe model fitted
    # above.  Do not fit a second opportunity model on only dozens of historical
    # type-positive rows: that would be fold-specific overfitting.
    for spec in EXPERT_SPECS[:2]:
        target_column = f"target_{spec.expert_id}"
        type_train = add_episode_weight(
            frame.loc[split_type.model_fit_mask].copy(),
            positive_target_column=target_column,
        )
        supervision = _type_supervision_summary(type_train, target_column)
        conditional_diag = _conditional_tp_diagnostic(
            frame.loc[split_tp.model_fit_mask].copy(),
            target_column=target_column,
        )
        print(
            f"[fold] {fold_name} {spec.expert_id} "
            f"type_rows={supervision['rows']:,} "
            f"type_positive_rows={supervision['positive_rows']:,} "
            f"type_positive_episodes={supervision['positive_episodes']:,} "
            f"conditional_tp_rows={conditional_diag['model_fit_rows']:,} "
            f"opportunity_head=global_shared",
            flush=True,
        )
        fit_rows.append(
            {
                "fold": fold_name,
                "expert_id": spec.expert_id,
                "head": "type_conditional_opportunity_sample_diagnostic",
                "policy_rows": int(len(policy)),
                **conditional_diag,
            }
        )
        if not bool(supervision["passed"]):
            fit_rows.append(
                {
                    "fold": fold_name,
                    "expert_id": spec.expert_id,
                    "head": "type_likelihood",
                    "model_fit_rows": int(len(type_train)),
                    "model_fit_positives": int(supervision["positive_rows"]),
                    "model_fit_positive_episodes": int(supervision["positive_episodes"]),
                    "policy_rows": int(len(policy)),
                    "status": "insufficient_type_supervision",
                    "passed": np.nan,
                    **supervision,
                }
            )
            continue

        try:
            type_fit, type_feature_diag = fit_resolved_binary_model(
                type_train,
                policy,
                requested_features=COMPACT_EXPERT_FEATURES,
                required_features=(str(spec.special_feature),),
                target_column=target_column,
                random_state=int(args.random_state),
                min_samples_leaf=int(args.model_min_samples_leaf),
                prediction_chunk_size=int(args.prediction_chunk_size),
            )
        except ExpertModelUnavailableError as exc:
            fit_rows.append(
                {
                    "fold": fold_name,
                    "expert_id": spec.expert_id,
                    "head": "type_likelihood",
                    "model_fit_rows": int(len(type_train)),
                    "model_fit_positives": int(supervision["positive_rows"]),
                    "model_fit_positive_episodes": int(supervision["positive_episodes"]),
                    "policy_rows": int(len(policy)),
                    "status": "model_unavailable",
                    "unavailable_reason": type(exc).__name__,
                    "detail": str(exc),
                    "passed": np.nan,
                }
            )
            continue
        type_feature_diag.insert(0, "head", "type_likelihood")
        type_feature_diag.insert(0, "expert_id", spec.expert_id)
        type_feature_diag.insert(0, "fold", fold_name)
        feature_rows.append(type_feature_diag)
        fit_rows.append(
            {
                "fold": fold_name,
                "expert_id": spec.expert_id,
                "head": "type_likelihood",
                "model_fit_rows": int(len(type_train)),
                "model_fit_positives": int(type_train[target_column].sum()),
                "model_fit_positive_episodes": int(supervision["positive_episodes"]),
                "policy_rows": int(len(policy)),
                "status": "fitted",
                **type_fit.diagnostics,
            }
        )

        type_policy_score = predict_binary_score_chunked(
            type_fit.model, policy, chunk_size=int(args.prediction_chunk_size)
        )
        type_test_score = predict_binary_score_chunked(
            type_fit.model, test, chunk_size=int(args.prediction_chunk_size)
        )
        type_policy_probability = predict_binary_probability_chunked(
            type_fit.model, policy, chunk_size=int(args.prediction_chunk_size)
        )
        type_test_probability = predict_binary_probability_chunked(
            type_fit.model, test, chunk_size=int(args.prediction_chunk_size)
        )
        type_reference = EmpiricalRankReference.fit(type_policy_score)
        type_policy_rank = type_reference.transform(type_policy_score)
        type_test_rank = type_reference.transform(type_test_score)

        # Reuse the full-universe opportunity head identically in every fold.
        # This preserves sample size, avoids a sparse second-stage overfit, and
        # keeps the architecture deployable when the true future type is unknown.
        opp_policy_score = global_policy_score
        opp_test_score = global_test_score
        opp_policy_probability = global_policy_probability
        opp_test_probability = global_test_probability
        opp_policy_rank = global_policy_rank
        opp_test_rank = global_test_rank
        _, type_opp_rank = _policy_rank(
            {"type": type_policy_rank, "opportunity": opp_policy_rank},
            {"type": type_test_rank, "opportunity": opp_test_rank},
            {"type": 0.60, "opportunity": 0.40},
        )
        special_available = True
        try:
            special_policy_rank, special_test_rank, _ = signed_feature_rank(
                policy[str(spec.special_feature)],
                test[str(spec.special_feature)],
                direction=float(spec.special_direction),
            )
            _, type_opp_special_rank = _policy_rank(
                {
                    "type": type_policy_rank,
                    "opportunity": opp_policy_rank,
                    "special": special_policy_rank,
                },
                {
                    "type": type_test_rank,
                    "opportunity": opp_test_rank,
                    "special": special_test_rank,
                },
                {"type": 0.50, "opportunity": 0.30, "special": 0.20},
            )
        except ExpertRankUnavailableError as exc:
            special_available = False
            special_policy_rank = np.full(len(policy), np.nan, dtype=float)
            special_test_rank = np.full(len(test), np.nan, dtype=float)
            type_opp_special_rank = np.full(len(test), np.nan, dtype=float)
            fit_rows.append(
                {
                    "fold": fold_name,
                    "expert_id": spec.expert_id,
                    "head": "special_condition_rank",
                    "model_fit_rows": 0,
                    "model_fit_positives": 0,
                    "policy_rows": int(len(policy)),
                    "status": "insufficient_policy_variation",
                    "unavailable_reason": type(exc).__name__,
                    "detail": str(exc),
                    "passed": np.nan,
                }
            )
        model_metric_rows.append(
            _model_metric_row(
                fold=fold_name,
                expert_id=spec.expert_id,
                head="type_likelihood",
                target_column=target_column,
                test=test,
                score=type_test_score,
            )
        )
        actual_type_mask = test[target_column].astype(bool).to_numpy()
        actual_type_test = test.loc[actual_type_mask].copy()
        actual_type_score = global_test_score[actual_type_mask]
        model_metric_rows.append(
            _model_metric_row(
                fold=fold_name,
                expert_id=spec.expert_id,
                head="shared_global_opportunity_within_type",
                target_column="target_tp60",
                test=actual_type_test,
                score=actual_type_score,
            )
        )

        prediction = test[
            [
                "event_id",
                "extreme_time",
                "extreme_pos",
                "is_macro_first_sweep",
                target_column,
                "tp_1_h60",
                "mae_h60_pct",
                "reference_stage2_type",
                "reference_swing_event_id",
                str(spec.special_feature),
            ]
        ].copy()
        prediction.insert(0, "expert_id", spec.expert_id)
        prediction.insert(0, "fold", fold_name)
        prediction["type_raw_score"] = type_test_score
        prediction["type_probability"] = type_test_probability
        prediction["type_rank"] = type_test_rank
        prediction["opportunity_head_source"] = OPPORTUNITY_HEAD_SOURCE
        prediction["opportunity_raw_score"] = opp_test_score
        prediction["opportunity_probability"] = opp_test_probability
        prediction["opportunity_rank"] = opp_test_rank
        prediction["special_rank"] = special_test_rank
        prediction["type_global_opportunity_rank"] = type_opp_rank
        prediction["type_global_opportunity_special_rank"] = type_opp_special_rank
        prediction_parts.append(prediction)

        policies = {
            f"{spec.expert_id[:2]}_TYPE_ONLY": type_test_rank,
            f"{spec.expert_id[:2]}_TYPE_GLOBAL_OPPORTUNITY": type_opp_rank,
        }
        if special_available:
            policies.update(
                {
                    f"{spec.expert_id[:2]}_SPECIAL_ONLY": special_test_rank,
                    f"{spec.expert_id[:2]}_TYPE_GLOBAL_OPPORTUNITY_SPECIAL": type_opp_special_rank,
                }
            )
        expert_pool = test[test[target_column].astype(bool)]
        for policy_id, rank in policies.items():
            for top_pct in sorted(set(int(value) for value in args.rank_top_pcts)):
                mask = top_rank_mask(rank, top_pct)
                selected = test.loc[mask].copy()
                selected["opportunity_score"] = rank[mask]
                rows, stressed, trades = _evaluate_policy(
                    bars=bars,
                    fold_name=fold_name,
                    expert_id=spec.expert_id,
                    policy_id=policy_id,
                    top_pct=top_pct,
                    selected=selected,
                    baseline=test,
                    expert_pool=expert_pool,
                    target_column=target_column,
                    months=months,
                    args=args,
                )
                scorecard_rows.extend(rows)
                strong_rows.extend(stressed)
                trade_parts.extend(trades)

    # First Sweep control.  The event itself is causal and always evaluated.
    # Ranking within the event pool reuses the same full-universe opportunity
    # model as E1/E2; no sparse event-conditional model is force-fit.
    sweep_spec = EXPERT_SPECS[2]
    sweep_target = f"target_{sweep_spec.expert_id}"
    sweep_train_source = frame.loc[
        split_tp.model_fit_mask & frame[sweep_target].astype(bool).to_numpy()
    ].copy()
    sweep_policy = policy[policy[sweep_target].astype(bool)].reset_index(drop=True)
    sweep_test = test[test[sweep_target].astype(bool)].reset_index(drop=True)
    sweep_classes = (
        int(sweep_train_source["target_tp60"].astype(bool).nunique())
        if len(sweep_train_source)
        else 0
    )
    fit_rows.append(
        {
            "fold": fold_name,
            "expert_id": sweep_spec.expert_id,
            "head": "event_conditional_opportunity_sample_diagnostic",
            "model_fit_rows": int(len(sweep_train_source)),
            "model_fit_positives": (
                int(sweep_train_source["target_tp60"].sum())
                if len(sweep_train_source)
                else 0
            ),
            "model_fit_classes": sweep_classes,
            "policy_rows": int(len(sweep_policy)),
            "test_rows": int(len(sweep_test)),
            "status": "not_fit_shared_global_opportunity",
            "opportunity_head_source": OPPORTUNITY_HEAD_SOURCE,
        }
    )

    if not sweep_test.empty:
        sweep_test_all = sweep_test.copy()
        sweep_test_all["opportunity_score"] = 100.0
        rows, stressed, trades = _evaluate_policy(
            bars=bars,
            fold_name=fold_name,
            expert_id=sweep_spec.expert_id,
            policy_id="E3_EVENT_ALL",
            top_pct=100,
            selected=sweep_test_all,
            baseline=test,
            expert_pool=sweep_test,
            target_column=None,
            months=months,
            args=args,
        )
        scorecard_rows.extend(rows)
        strong_rows.extend(stressed)
        trade_parts.extend(trades)

    sweep_rank_available = len(sweep_policy) >= 20 and len(sweep_test) >= 1
    if sweep_rank_available:
        sweep_policy_score = predict_binary_score_chunked(
            global_fit.model, sweep_policy, chunk_size=int(args.prediction_chunk_size)
        )
        sweep_test_score = predict_binary_score_chunked(
            global_fit.model, sweep_test, chunk_size=int(args.prediction_chunk_size)
        )
        sweep_policy_probability = predict_binary_probability_chunked(
            global_fit.model, sweep_policy, chunk_size=int(args.prediction_chunk_size)
        )
        sweep_test_probability = predict_binary_probability_chunked(
            global_fit.model, sweep_test, chunk_size=int(args.prediction_chunk_size)
        )
        try:
            sweep_reference = EmpiricalRankReference.fit(sweep_policy_score)
        except ExpertRankUnavailableError as exc:
            sweep_rank_available = False
            fit_rows.append(
                {
                    "fold": fold_name,
                    "expert_id": sweep_spec.expert_id,
                    "head": "shared_global_opportunity_event_rank",
                    "model_fit_rows": 0,
                    "model_fit_positives": 0,
                    "policy_rows": int(len(sweep_policy)),
                    "test_rows": int(len(sweep_test)),
                    "status": "insufficient_policy_variation",
                    "unavailable_reason": type(exc).__name__,
                    "detail": str(exc),
                    "passed": np.nan,
                }
            )
    if sweep_rank_available:
        sweep_test_rank = sweep_reference.transform(sweep_test_score)
        sweep_resolution = raw_score_resolution(sweep_policy_score)
        fit_rows.append(
            {
                "fold": fold_name,
                "expert_id": sweep_spec.expert_id,
                "head": "shared_global_opportunity_event_rank",
                "model_fit_rows": 0,
                "model_fit_positives": 0,
                "policy_rows": int(len(sweep_policy)),
                "test_rows": int(len(sweep_test)),
                "status": "shared_global_rank_available",
                "opportunity_head_source": OPPORTUNITY_HEAD_SOURCE,
                **sweep_resolution,
            }
        )
        model_metric_rows.append(
            _model_metric_row(
                fold=fold_name,
                expert_id=sweep_spec.expert_id,
                head="shared_global_opportunity_within_event",
                target_column="target_tp60",
                test=sweep_test,
                score=sweep_test_score,
            )
        )
        prediction = sweep_test[
            [
                "event_id",
                "extreme_time",
                "extreme_pos",
                "is_macro_first_sweep",
                "tp_1_h60",
                "mae_h60_pct",
                "reference_stage2_type",
                "reference_swing_event_id",
            ]
        ].copy()
        prediction.insert(0, "expert_id", sweep_spec.expert_id)
        prediction.insert(0, "fold", fold_name)
        prediction["opportunity_head_source"] = OPPORTUNITY_HEAD_SOURCE
        prediction["opportunity_raw_score"] = sweep_test_score
        prediction["opportunity_probability"] = sweep_test_probability
        prediction["opportunity_rank"] = sweep_test_rank
        prediction_parts.append(prediction)

        for top_pct in sorted(set(int(value) for value in args.rank_top_pcts)):
            mask = top_rank_mask(sweep_test_rank, top_pct)
            selected = sweep_test.loc[mask].copy()
            selected["opportunity_score"] = sweep_test_rank[mask]
            rows, stressed, trades = _evaluate_policy(
                bars=bars,
                fold_name=fold_name,
                expert_id=sweep_spec.expert_id,
                policy_id="E3_EVENT_OPPORTUNITY",
                top_pct=top_pct,
                selected=selected,
                baseline=test,
                expert_pool=sweep_test,
                target_column=None,
                months=months,
                args=args,
            )
            scorecard_rows.extend(rows)
            strong_rows.extend(stressed)
            trade_parts.extend(trades)
    elif len(sweep_policy) < 20 or len(sweep_test) < 1:
        fit_rows.append(
            {
                "fold": fold_name,
                "expert_id": sweep_spec.expert_id,
                "head": "shared_global_opportunity_event_rank",
                "model_fit_rows": 0,
                "model_fit_positives": 0,
                "policy_rows": int(len(sweep_policy)),
                "test_rows": int(len(sweep_test)),
                "status": "insufficient_control_rows",
                "passed": np.nan,
            }
        )

    split_row = {
        "fold": fold_name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "policy_start": split_tp.policy_start,
        "model_fit_end": split_tp.model_fit_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "model_fit_rows_tp": int(split_tp.model_fit_mask.sum()),
        "model_fit_rows_type": int(split_type.model_fit_mask.sum()),
        "policy_rows": int(split_tp.policy_mask.sum()),
        "test_rows": int(split_tp.test_mask.sum()),
        "test_months": months,
    }
    return {
        "split": [split_row],
        "incidence": incidence_rows,
        "features": feature_rows,
        "fits": fit_rows,
        "metrics": model_metric_rows,
        "predictions": prediction_parts,
        "scorecard": scorecard_rows,
        "strong_days": strong_rows,
        "trades": trade_parts,
    }


def _concat(parts: list) -> pd.DataFrame:
    frames = [part for part in parts if isinstance(part, pd.DataFrame) and not part.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_summary(
    inventory: pd.DataFrame,
    incidence: pd.DataFrame,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    stage1 = inventory[inventory["hierarchy_level"].eq("stage1_type")]
    lines = [
        f"# {TITLE}",
        "",
        "## Frozen objective",
        "",
        "- Long-only online candidates for future +1% moves from next-bar open.",
        "- E1 identifies original C3-D; E2 identifies original C3-E; E3 is the First Sweep control.",
        "- Original types are future labels only. No type label or TP path enters a feature.",
        "- E1/E2 share one +1% opportunity head trained on the full candidate universe; sparse type-conditional TP rows are diagnostics only and are never force-fit.",
        "- Top20/30/40 are the only permitted ranked policies. Narrower tails are prohibited.",
        "- A result cannot be retained when annualized executable frequency is below the hard guardrail, even if expectancy looks high.",
        "",
        "## Original type universe",
        "",
    ]
    for row in stage1.itertuples(index=False):
        lines.append(f"- {row.type_id}: {int(row.count):,} ({float(row.share_within_level):.2%}).")
    lines.extend(["", "## Sparse target incidence", ""])
    if incidence.empty:
        lines.append("- No incidence rows generated.")
    else:
        for row in incidence[incidence["target"].str.contains("C3D|C3E")].itertuples(index=False):
            lines.append(
                f"- {row.fold} {row.target}: test {int(row.test_positives):,}/{int(row.test_rows):,}."
            )
    lines.extend(["", "## Model ranking", ""])
    for row in metrics.itertuples(index=False):
        if row.head == "type_likelihood":
            lines.append(
                f"- {row.fold} {row.expert_id}: type AP={row.average_precision:.4f}, AUC={row.roc_auc:.4f}, positives={int(row.test_positives):,}."
            )
    lines.extend(["", "## Frequency-aware economic candidates", ""])
    if summary.empty:
        lines.append("- No policy summary generated.")
    else:
        for row in summary.head(20).itertuples(index=False):
            lines.append(
                f"- {row.expert_id} {row.policy_id} Top{int(row.top_pct)} cost={float(row.cost_multiplier):g}x: "
                f"status={row.research_candidate_status}, executable={int(row.total_oos_executable_trades):,}, "
                f"min annualized executable={float(row.minimum_annualized_executable_trades):.1f}, "
                f"median net={float(row.median_net_expectancy_bps):+.2f}bp, "
                f"median TP uplift={float(row.median_tp60_uplift_pp):+.2f}pp."
            )
    lines.extend(["", "## Causal and selection audit", ""])
    for row in audit.itertuples(index=False):
        lines.append(f"- {'PASS' if bool(row.passed) else 'FAIL'} `{row.check}`: {row.detail}")
    lines.extend(
        [
            "",
            "## Decision rule",
            "",
            "- Statistical type recognition alone is insufficient.",
            "- A Research 17 candidate must pass all three folds, broad Top20/30/40 neighborhood checks, frequency guardrails, cost stress and strong-day stress before a dedicated deployable backtest is justified.",
            "- If E1/E2 type precision remains too low across all broad thresholds, stop adding features and revisit the online candidate/region representation.",
            "",
        ]
    )
    return "\n".join(lines)



def _special_feature_entry_audit(
    feature_diagnostics: pd.DataFrame,
    fit_diagnostics: pd.DataFrame,
    fold_names: Sequence[str],
) -> dict[str, object]:
    expected = {
        *{
            (str(fold), "B0_BROAD", "global_opportunity", feature)
            for fold in fold_names
            for feature in ("sell_pressure_absorption_30", "region_rebound_from_low")
        },
        *{
            (str(fold), expert_id, "type_likelihood", feature)
            for fold in fold_names
            for expert_id, feature in (
                ("E1_C3D_PRICE_RESPONSE", "sell_pressure_absorption_30"),
                ("E2_C3E_EARLY_RECOVERY", "region_rebound_from_low"),
            )
        },
    }
    declared_unavailable: set[tuple[str, str, str, str]] = set()
    if not fit_diagnostics.empty:
        for row in fit_diagnostics.itertuples(index=False):
            if str(getattr(row, "status", "")) not in {
                "insufficient_type_supervision",
                "model_unavailable",
            }:
                continue
            expert_id = str(row.expert_id)
            feature = {
                "E1_C3D_PRICE_RESPONSE": "sell_pressure_absorption_30",
                "E2_C3E_EARLY_RECOVERY": "region_rebound_from_low",
            }.get(expert_id)
            if feature is not None:
                declared_unavailable.add(
                    (str(row.fold), expert_id, "type_likelihood", feature)
                )
    observed: set[tuple[str, str, str, str]] = set()
    wrong_status: list[str] = []
    if not feature_diagnostics.empty:
        required_rows = feature_diagnostics[
            feature_diagnostics["feature"].isin(
                ["sell_pressure_absorption_30", "region_rebound_from_low"]
            )
        ]
        for row in required_rows.itertuples(index=False):
            key = (str(row.fold), str(row.expert_id), str(row.head), str(row.feature))
            if key in expected:
                if str(row.status) == "selected_pre_correlation":
                    observed.add(key)
                else:
                    wrong_status.append(f"{key}:{row.status}")
    required_after_declared_skip = expected.difference(declared_unavailable)
    missing = sorted(required_after_declared_skip.difference(observed))
    return {
        "passed": bool(not missing and not wrong_status),
        "expected_rows": len(expected),
        "required_after_declared_skip": len(required_after_declared_skip),
        "observed_selected_rows": len(observed),
        "declared_unavailable": sorted(declared_unavailable),
        "missing": missing,
        "wrong_status": wrong_status,
    }


def _raw_score_fit_audit(
    fit_diagnostics: pd.DataFrame,
    fold_names: Sequence[str],
) -> dict[str, object]:
    hard_required = {
        (str(fold), "B0_BROAD", "global_opportunity")
        for fold in fold_names
    }
    type_heads = {
        (str(fold), expert_id, "type_likelihood")
        for fold in fold_names
        for expert_id in ("E1_C3D_PRICE_RESPONSE", "E2_C3E_EARLY_RECOVERY")
    }
    observed: set[tuple[str, str, str]] = set()
    declared_unavailable: set[tuple[str, str, str]] = set()
    failed: list[str] = []
    if not fit_diagnostics.empty:
        for row in fit_diagnostics.itertuples(index=False):
            key = (str(row.fold), str(row.expert_id), str(row.head))
            status = str(getattr(row, "status", ""))
            passed_value = getattr(row, "passed", np.nan)
            if key in hard_required or key in type_heads:
                if pd.notna(passed_value) and bool(passed_value):
                    observed.add(key)
                elif key in type_heads and status in {
                    "insufficient_type_supervision",
                    "model_unavailable",
                }:
                    declared_unavailable.add(key)
                else:
                    failed.append(f"{key}:passed={passed_value}:status={status}")
            elif (
                str(row.expert_id) == "E3_FIRST_SWEEP_CONTROL"
                and str(row.head) == "shared_global_opportunity_event_rank"
            ):
                if status in {"insufficient_control_rows", "insufficient_policy_variation"}:
                    declared_unavailable.add(key)
                elif not (pd.notna(passed_value) and bool(passed_value)):
                    failed.append(f"{key}:passed={passed_value}:status={status}")
    missing_hard = sorted(hard_required.difference(observed))
    unresolved_type = sorted(type_heads.difference(observed).difference(declared_unavailable))
    return {
        "passed": bool(not missing_hard and not unresolved_type and not failed),
        "expected_hard_heads": len(hard_required),
        "expected_type_heads": len(type_heads),
        "observed_passed_heads": len(observed),
        "declared_unavailable": sorted(declared_unavailable),
        "missing_hard": missing_hard,
        "unresolved_type": unresolved_type,
        "failed": failed,
    }


def run_research(args: argparse.Namespace) -> Path:
    _validate_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    historical, compatibility, hierarchy_audit = _load_original_reports(args)
    _write_csv(compatibility, out_dir / "01_frozen_report_compatibility.csv")
    inventory = typology_inventory(historical)
    _write_csv(inventory, out_dir / "02_original_typology_inventory.csv")
    _write_csv(expert_spec_table(), out_dir / "03_frozen_expert_specs.csv")

    bars = _R16R.load_bars(args)
    _write_csv(validate_trade_bar_fields(bars), out_dir / "04_trade_bar_field_coverage.csv")
    frame, diagnostics = _build_dataset(args, bars, historical)
    _write_csv(diagnostics["gate_coverage"], out_dir / "05_candidate_gate_coverage.csv")
    _write_csv(diagnostics["first_sweep"], out_dir / "06_first_sweep_diagnostics.csv")
    _write_csv(diagnostics["region_summary"], out_dir / "07_region_build_summary.csv")
    _write_csv(diagnostics["region_dictionary"], out_dir / "08_region_feature_dictionary.csv")
    _write_csv(diagnostics["feature_dictionary"], out_dir / "09_online_feature_dictionary.csv")
    _write_csv(diagnostics["alignment_audit"], out_dir / "10_htf_available_time_audit.csv")

    all_results: dict[str, list] = {
        "split": [],
        "incidence": [],
        "features": [],
        "fits": [],
        "metrics": [],
        "predictions": [],
        "scorecard": [],
        "strong_days": [],
        "trades": [],
    }
    for fold in folds_for_end_date(args.end_date):
        fold_result = _run_fold(bars=bars, frame=frame, fold=fold, args=args)
        for key, values in fold_result.items():
            all_results[key].extend(values)

    split_table = pd.DataFrame(all_results["split"])
    incidence = pd.DataFrame(all_results["incidence"])
    feature_diagnostics = _concat(all_results["features"])
    fit_diagnostics = pd.DataFrame(all_results["fits"])
    model_metrics = pd.DataFrame(all_results["metrics"])
    predictions = _concat(all_results["predictions"])
    scorecard = pd.DataFrame(all_results["scorecard"])
    strong_days = pd.DataFrame(all_results["strong_days"])
    trades = _concat(all_results["trades"])
    crossfold = crossfold_policy_summary(scorecard)
    neighborhood = policy_neighborhood_summary(crossfold)

    _write_csv(split_table, out_dir / "11_walkforward_boundaries.csv")
    _write_csv(incidence, out_dir / "12_target_incidence.csv")
    _write_csv(feature_diagnostics, out_dir / "13_feature_conditioning_diagnostics.csv")
    _write_csv(fit_diagnostics, out_dir / "14_model_fit_and_resolution.csv")
    _write_csv(model_metrics, out_dir / "15_model_ranking_metrics.csv")
    _write_csv(scorecard, out_dir / "16_frequency_path_execution_scorecard.csv")
    _write_csv(crossfold, out_dir / "17_crossfold_policy_summary.csv")
    _write_csv(neighborhood, out_dir / "18_policy_neighborhood_summary.csv")
    _write_csv(strong_days, out_dir / "19_delete_strong_days_stress.csv")

    if bool(args.write_full_predictions):
        _write_csv(predictions, out_dir / "20_oos_full_predictions.csv")
    elif not predictions.empty:
        sample_parts = []
        for _, group in predictions.groupby(["fold", "expert_id"], sort=True):
            sample_parts.append(group.sample(min(2_000, len(group)), random_state=42))
        sample = pd.concat(sample_parts, ignore_index=True).sort_values(
            ["fold", "expert_id", "extreme_time", "event_id"], kind="mergesort"
        )
        _write_csv(sample, out_dir / "20_oos_prediction_sample.csv")
    if not trades.empty:
        sample = trades
        if len(sample) > 20_000:
            sample = sample.sample(20_000, random_state=42)
        _write_csv(
            sample.sort_values(["entry_time", "event_id"], kind="mergesort"),
            out_dir / "21_execution_trade_sample.csv",
        )

    fold_names = split_table["fold"].astype(str).tolist() if not split_table.empty else []
    special_feature_audit = _special_feature_entry_audit(
        feature_diagnostics, fit_diagnostics, fold_names
    )
    raw_score_audit = _raw_score_fit_audit(fit_diagnostics, fold_names)
    target_columns = {f"target_{spec.expert_id}" for spec in EXPERT_SPECS}
    future_feature_overlap = sorted(target_columns.intersection(COMPACT_EXPERT_FEATURES))
    audit = pd.concat(
        [
            hierarchy_audit,
            pd.DataFrame(
                [
                    {
                        "check": "original_type_and_tp_targets_are_labels_only",
                        "passed": bool(not future_feature_overlap),
                        "detail": f"target_feature_overlap={future_feature_overlap}",
                    },
                    {
                        "check": "closed_bar_to_next_open_label_timing",
                        "passed": bool(
                            (
                                pd.to_datetime(frame["entry_time"], errors="raise")
                                > pd.to_datetime(frame["extreme_time"], errors="raise")
                            ).all()
                        ),
                        "detail": f"rows={len(frame):,}",
                    },
                    {
                        "check": "htf_available_time_alignment",
                        "passed": bool(
                            diagnostics["alignment_audit"].empty
                            or diagnostics["alignment_audit"]["passed"].fillna(False).astype(bool).all()
                        ),
                        "detail": f"rows={len(diagnostics['alignment_audit']):,}",
                    },
                    {
                        "check": "special_features_really_enter_conditioned_models",
                        "passed": bool(special_feature_audit["passed"]),
                        "detail": (
                            f"selected={special_feature_audit['observed_selected_rows']}/"
                            f"{special_feature_audit['required_after_declared_skip']} required; "
                            f"declared_unavailable={special_feature_audit['declared_unavailable']}; "
                            f"missing={special_feature_audit['missing']}; "
                            f"wrong_status={special_feature_audit['wrong_status']}"
                        ),
                    },
                    {
                        "check": "raw_decision_score_resolution",
                        "passed": bool(raw_score_audit["passed"]),
                        "detail": (
                            f"passed_heads={raw_score_audit['observed_passed_heads']}; "
                            f"hard={raw_score_audit['expected_hard_heads']}; "
                            f"type={raw_score_audit['expected_type_heads']}; "
                            f"declared_unavailable={raw_score_audit['declared_unavailable']}; "
                            f"missing_hard={raw_score_audit['missing_hard']}; "
                            f"unresolved_type={raw_score_audit['unresolved_type']}; "
                            f"failed={raw_score_audit['failed']}; "
                            "ranking uses decision_function or logit fallback"
                        ),
                    },
                    {
                        "check": "only_broad_rank_thresholds",
                        "passed": bool(set(scorecard["top_pct"].astype(int).unique()).issubset({20, 30, 40, 100})),
                        "detail": "Top20/30/40 plus explicit First Sweep all-event control",
                    },
                    {
                        "check": "frequency_guard_is_mandatory",
                        "passed": bool(
                            {
                                "annualized_raw_frequency_passed",
                                "annualized_executable_frequency_passed",
                                "frequency_guard_passed",
                            }.issubset(scorecard.columns)
                        ),
                        "detail": (
                            f"minimum annualized raw={float(args.minimum_annualized_raw_events):.1f}; "
                            f"minimum annualized executable={float(args.minimum_annualized_executable_trades):.1f}"
                        ),
                    },
                    {
                        "check": "first_sweep_not_global_gate",
                        "passed": bool((~frame["is_macro_first_sweep"]).any()),
                        "detail": f"first_sweep={int(frame['is_macro_first_sweep'].sum()):,}/{len(frame):,}",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    _write_csv(audit, out_dir / "22_causal_selection_frequency_audit.csv")
    if not audit["passed"].fillna(False).astype(bool).all():
        raise RuntimeError("Research 17 audit failed; inspect 22_causal_selection_frequency_audit.csv")

    manifest = {
        "script": SCRIPT_NAME,
        "version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "scope": "research_only_original_C3D_C3E_online_experts_shared_global_opportunity_with_frequency_guard",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "target_move_pct": FROZEN_TARGET_MOVE_PCT,
        "forward_horizon_bars": FROZEN_FORWARD_HORIZON_BARS,
        "research_start": args.start_date,
        "research_end": args.end_date,
        "warmup_start": args.warmup_start_date,
        "candidate_count": int(len(frame)),
        "historical_swing_low_count": int(len(historical)),
        "rank_top_pcts": sorted(set(int(value) for value in args.rank_top_pcts)),
        "policy_window_days": int(args.policy_window_days),
        "minimum_annualized_raw_events": float(args.minimum_annualized_raw_events),
        "minimum_annualized_executable_trades": float(args.minimum_annualized_executable_trades),
        "minimum_total_oos_executable_for_candidate": 600,
        "expert_specs": expert_spec_table().to_dict(orient="records"),
        "feature_columns": list(COMPACT_EXPERT_FEATURES),
        "opportunity_head_source": OPPORTUNITY_HEAD_SOURCE,
        "type_global_opportunity_weights": {"type": 0.60, "opportunity": 0.40},
        "type_global_opportunity_special_weights": {
            "type": 0.50,
            "opportunity": 0.30,
            "special": 0.20,
        },
        "execution": {
            "direction": "long",
            "signal": "current closed 1m bar",
            "entry": "next bar open",
            "take_profit": "+1% on first future closed-bar close",
            "time_exit": "60 closed bars",
            "ordinary_stop": "none",
            "roundtrip_fee": "0.11% at 1x",
            "roundtrip_slippage": "0.04% at 1x",
            "cost_multipliers": [1.0, 1.5, 2.0],
            "entry_delays": [0, 1, 3],
        },
        "causal_policy": "future original types and future +1% paths are targets only; E1/E2 share the full-universe +1% opportunity head; all ranking references use trailing policy scores without labels",
    }
    (out_dir / "00_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (out_dir / "23_RESEARCH_SUMMARY.md").write_text(
        _build_summary(inventory, incidence, model_metrics, crossfold, audit),
        encoding="utf-8",
    )
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
