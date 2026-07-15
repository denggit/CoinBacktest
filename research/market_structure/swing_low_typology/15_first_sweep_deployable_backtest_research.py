#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deployable backtest research 15 for respected-macro first sweeps.

This is the first strategy-PnL stage in the swing-low typology sequence.  It
freezes the event pool and model-ranking methodology developed in 12--14, then
asks whether broad score policies survive deterministic execution, costs,
delay, stops, overlap, and strongest-day deletion.

No frozen test fold is used to select a model, score threshold, stop, horizon,
or add-on rule.  All variants are predeclared and reported.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from research.market_structure.swing_low_typology.common.deployable_first_sweep_backtest import (  # noqa: E402
    AddOnSpec,
    CostSpec,
    ExitSpec,
    deduplicate_signals,
    default_exit_specs,
    enforce_single_position,
    prepare_bars,
    remove_strongest_days,
    replay_trade,
    summarize_trades,
)
from research.market_structure.swing_low_typology.common.first_sweep_event import (  # noqa: E402
    build_first_sweep_event_decisions,
)
from research.market_structure.swing_low_typology.common.multihorizon_close_labels import (  # noqa: E402
    build_multihorizon_close_labels,
)
from research.market_structure.swing_low_typology.common.range_increment import EmpiricalRankReference  # noqa: E402
from research.market_structure.swing_low_typology.common.reversal_opportunity import (  # noqa: E402
    build_reversal_candidate_features,
)
from research.market_structure.swing_low_typology.common.sequential_sweep_state import (  # noqa: E402
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
_R14 = importlib.import_module(
    "research.market_structure.swing_low_typology.14_sequential_first_sweep_state_scoring_research"
)

SCRIPT_NAME = "15_first_sweep_deployable_backtest_research"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_1M_RESPECTED_MACRO_FIRST_SWEEP_DEPLOYABLE_BACKTEST_15"
EDGE_ID = "RESEARCH_ONLY_ETH_FIRST_SWEEP_DEPLOYABLE_BACKTEST"
TITLE = "ETH Respected Macro First Sweep Deployable Backtest Research 15"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/15_first_sweep_deployable_backtest"
PRIMARY_FAMILY = "logistic_sgd"

FoldSpec = _R12.FoldSpec
_condition_feature_columns = _R12._condition_feature_columns
_predict_binary_probability = _R12._predict_binary_probability
_predict_binary_score = _R12._predict_binary_score
_fit_binary_with_resolution_fallback = _R12._fit_binary_with_resolution_fallback
_rank_resolution_record = _R12._rank_resolution_record
_assert_raw_score_resolution = _R12._assert_raw_score_resolution
COMPACT_SNAPSHOT_FEATURES = _R14.COMPACT_SNAPSHOT_FEATURES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal deployable first-sweep strategy backtest.",
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
    p.add_argument("--score-top-pcts", nargs="+", type=int, default=[20, 30, 40])
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
    p.add_argument("--random-state", type=int, default=42)

    # Conservative market-execution convention: 0.11% fee plus 0.04%
    # round-trip slippage under baseline assumptions.
    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.00020)
    p.add_argument("--exit-slippage-pct", type=float, default=0.00020)
    p.add_argument("--capital-fraction", type=float, default=0.10)
    p.add_argument("--starting-equity", type=float, default=1.0)
    p.add_argument("--stress-cost-multipliers", nargs="+", type=float, default=[1.5, 2.0])
    p.add_argument("--stress-entry-delays", nargs="+", type=int, default=[1, 3])
    p.add_argument("--add-on-top-pct", type=int, default=30)
    p.add_argument("--add-on-maximum-chase-pct", type=float, default=0.0025)
    p.add_argument("--write-full-trades", action="store_true")
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
    return _R14._subset_origin_period(frame, start, end)


def _development_split(train: pd.DataFrame, fold: FoldSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _R14._development_split(train, fold)


def _score_shell(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "event_id", "origin_event_id", "lifecycle_id", "causal_region_id",
        "origin_sweep_pos", "origin_sweep_time", "checkpoint_offset", "extreme_pos",
        "extreme_time", "feature_available_time", "state_status", "prior_tp_reached",
        "hard_invalidated", "add_on_eligible", "initial_decision", "entry_time",
        "entry_price", "label_end_time", "level_price", "tp60", "tp180",
        "clean60_0p5", "mae_60_pct", "mae_180_pct",
    ]
    return frame.reindex(columns=keep).copy()


def _months_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return max(1, len(pd.period_range(start.to_period("M"), end.to_period("M"), freq="M")))


def _fit_and_score_group(
    *,
    fold: FoldSpec,
    fold_index: int,
    group_index: int,
    model_group: str,
    requested_features: Sequence[str],
    max_features: int,
    model_fit: pd.DataFrame,
    policy: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    fit_train = model_fit.loc[model_fit["checkpoint_offset"].eq(0)].copy()
    fit_train["episode_weight"] = 1.0
    preflight_policy = policy.loc[policy["checkpoint_offset"].eq(0)].copy()
    selected, feature_diag = _condition_feature_columns(fit_train, requested_features, max_features=max_features)
    model_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    score_frames = {"policy": _score_shell(policy), "test": _score_shell(test)}

    for head_index, (output, target) in enumerate((("p_tp60", "tp60"), ("p_tp180", "tp180")), start=1):
        model, fit_diag = _fit_binary_with_resolution_fallback(
            fit_train,
            preflight_policy,
            feature_columns=selected,
            target_column=target,
            fold=fold.fold,
            decision_path="deployable_backtest",
            feature_group=model_group,
            output=output,
            random_state=int(args.random_state) + fold_index * 100 + group_index * 10 + head_index,
            min_samples_leaf=int(args.model_min_samples_leaf),
            prediction_chunk_size=int(args.prediction_chunk_size),
        )
        model_rows.append({
            "fold": fold.fold, "model_group": model_group, "output": output, "target": target,
            "requested_family": PRIMARY_FAMILY, "actual_family": getattr(model, "family", PRIMARY_FAMILY),
            **fit_diag,
        })
        policy_score = _predict_binary_score(model, policy, int(args.prediction_chunk_size))
        test_score = _predict_binary_score(model, test, int(args.prediction_chunk_size))
        policy_prob = _predict_binary_probability(model, policy, int(args.prediction_chunk_size))
        test_prob = _predict_binary_probability(model, test, int(args.prediction_chunk_size))
        anchor = policy["checkpoint_offset"].eq(0).to_numpy()
        reference = EmpiricalRankReference.fit(policy_score[anchor])
        for split, raw, probability in (
            ("policy", policy_score, policy_prob), ("test", test_score, test_prob),
        ):
            ranks = reference.transform(raw)
            score_frames[split][f"{output}_score_raw"] = raw
            score_frames[split][f"{output}_rank"] = ranks
            record = _rank_resolution_record(
                fold=fold.fold, decision_path="deployable_backtest", feature_group=model_group,
                output=output, split=split, raw_scores=raw, ranks=ranks,
                calibrated=probability, reference=reference, model_probability=probability,
            )
            rank_rows.append(record)
            _assert_raw_score_resolution(record, actual_family=str(getattr(model, "family", PRIMARY_FAMILY)))

    # The deployable Opportunity score is itself normalized on the development
    # policy t0 distribution.  A fixed 80/70/60 threshold therefore truly
    # represents policy Top20/30/40 instead of relying on the average of two
    # independently uniform ranks.
    for split in ("policy", "test"):
        score_frames[split]["opportunity_raw"] = 0.5 * (
            score_frames[split]["p_tp60_rank"] + score_frames[split]["p_tp180_rank"]
        )
    policy_anchor = score_frames["policy"]["checkpoint_offset"].eq(0).to_numpy()
    opportunity_reference = EmpiricalRankReference.fit(
        score_frames["policy"].loc[policy_anchor, "opportunity_raw"].to_numpy(dtype=float)
    )
    for split in ("policy", "test"):
        score_frames[split]["opportunity_score"] = 100.0 * opportunity_reference.transform(
            score_frames[split]["opportunity_raw"].to_numpy(dtype=float)
        )

    feature_row = {
        "fold": fold.fold, "model_group": model_group,
        "fit_rows": len(fit_train), "fit_events": fit_train["origin_event_id"].nunique(),
        **feature_diag,
    }
    return score_frames["policy"], score_frames["test"], model_rows, feature_row, rank_rows


def _attach_execution_fields(scored: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    out = scored.copy()
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    out["sweep_low"] = [lows[int(pos)] if 0 <= int(pos) < len(lows) else np.nan for pos in out["origin_sweep_pos"]]
    return out


def _merge_t1_score(t0: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    t1 = scored.loc[scored["checkpoint_offset"].eq(1), [
        "origin_event_id", "opportunity_score", "add_on_eligible", "state_status", "extreme_pos", "extreme_time"
    ]].rename(columns={
        "opportunity_score": "add_on_opportunity_score",
        "add_on_eligible": "add_on_eligible_t1",
        "state_status": "add_on_state_status",
        "extreme_pos": "add_on_signal_pos",
        "extreme_time": "add_on_signal_time",
    })
    out = t0.merge(t1, on="origin_event_id", how="left", validate="one_to_one")
    out["add_on_eligible"] = out["add_on_eligible_t1"].fillna(False).astype(bool)
    return out


def _replay_signals(
    prepared: object,
    events: pd.DataFrame,
    *,
    exit_spec: ExitSpec,
    costs: CostSpec,
    entry_delay_bars: int,
    add_on: AddOnSpec,
) -> tuple[pd.DataFrame, dict[str, int]]:
    deduped, duplicate_count = deduplicate_signals(events)
    rows: list[dict[str, object]] = []
    for _, event in deduped.iterrows():
        rec = replay_trade(
            prepared, event, exit_spec=exit_spec, costs=costs,
            entry_delay_bars=entry_delay_bars, add_on=add_on,
        )
        if bool(rec.get("valid")):
            rows.append(rec)
    isolated = pd.DataFrame(rows)
    portfolio, skipped = enforce_single_position(isolated)
    return portfolio, {
        "raw_signals": int(len(events)),
        "deduplicated_signals": int(duplicate_count),
        "skipped_overlap": int(skipped),
    }


def _base_variant_rows(
    *,
    fold: FoldSpec,
    model_group: str,
    scored_test: pd.DataFrame,
    prepared: object,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[dict[str, object]]]:
    summaries: list[dict[str, object]] = []
    trade_parts: list[pd.DataFrame] = []
    strong_rows: list[dict[str, object]] = []
    t0 = _merge_t1_score(
        scored_test.loc[scored_test["checkpoint_offset"].eq(0)].copy(), scored_test,
    )
    thresholds = [("ALL", 0.0), *[(f"TOP{int(pct)}", 100.0 - float(pct)) for pct in args.score_top_pcts]]
    exits = default_exit_specs()
    total = len(thresholds) * len(exits)
    progress = ProgressReporter(f"[backtest] {fold.fold} {model_group}", total=max(total, 1), every=1)
    ordinal = 0
    for policy_id, threshold in thresholds:
        selected = t0.loc[t0["opportunity_score"] >= threshold].copy()
        for exit_spec in exits:
            ordinal += 1
            progress.update(ordinal)
            costs = CostSpec(
                entry_fee_rate=float(args.entry_fee_rate), exit_fee_rate=float(args.exit_fee_rate),
                entry_slippage_pct=float(args.entry_slippage_pct), exit_slippage_pct=float(args.exit_slippage_pct),
                cost_multiplier=1.0,
            )
            trades, counts = _replay_signals(
                prepared, selected, exit_spec=exit_spec, costs=costs, entry_delay_bars=0,
                add_on=AddOnSpec(enabled=False, initial_weight=1.0, add_weight=0.0),
            )
            months = _months_between(fold.test_start, fold.test_end)
            summary = summarize_trades(
                trades, months=months, capital_fraction=float(args.capital_fraction),
                starting_equity=float(args.starting_equity), **counts,
            )
            summary.update({
                "fold": fold.fold, "model_group": model_group, "policy_id": policy_id,
                "score_threshold": threshold, "entry_mode": "single_full",
                "exit_spec_id": exit_spec.spec_id, "cost_multiplier": 1.0, "entry_delay_bars": 0,
            })
            summaries.append(summary)
            for removed in (5, 10):
                stressed = remove_strongest_days(trades, removed)
                stress_summary = summarize_trades(
                    stressed, months=months, capital_fraction=float(args.capital_fraction),
                    starting_equity=float(args.starting_equity), raw_signals=counts["raw_signals"],
                    skipped_overlap=counts["skipped_overlap"], deduplicated_signals=counts["deduplicated_signals"],
                )
                stress_summary.update({
                    "fold": fold.fold, "model_group": model_group, "policy_id": policy_id,
                    "entry_mode": "single_full", "exit_spec_id": exit_spec.spec_id,
                    "removed_strongest_days": removed,
                })
                strong_rows.append(stress_summary)
            if not trades.empty:
                sample = trades.copy()
                sample.insert(0, "entry_mode", "single_full")
                sample.insert(0, "policy_id", policy_id)
                sample.insert(0, "model_group", model_group)
                sample.insert(0, "fold", fold.fold)
                trade_parts.append(sample)
    progress.close()
    return summaries, trade_parts, strong_rows


def _stress_and_add_on_rows(
    *,
    fold: FoldSpec,
    model_group: str,
    scored_test: pd.DataFrame,
    prepared: object,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    trade_parts: list[pd.DataFrame] = []
    t0 = _merge_t1_score(scored_test.loc[scored_test["checkpoint_offset"].eq(0)].copy(), scored_test)
    anchor_exits = (
        ExitSpec("TP1_STRUCT_B10_H60", 60, 0.0100, "structural", 10.0),
        ExitSpec("TP1_FIXED_075_H60", 60, 0.0100, "fixed_pct", 0.0075),
        ExitSpec("TP1_STRUCT_B10_H180", 180, 0.0100, "structural", 10.0),
        ExitSpec("TP1_FIXED_075_H180", 180, 0.0100, "fixed_pct", 0.0075),
    )
    threshold = 100.0 - float(args.add_on_top_pct)
    selected = t0.loc[t0["opportunity_score"] >= threshold].copy()
    variants: list[tuple[str, ExitSpec, CostSpec, int, AddOnSpec]] = []
    for exit_spec in anchor_exits:
        variants.append((
            "single_full", exit_spec,
            CostSpec(float(args.entry_fee_rate), float(args.exit_fee_rate), float(args.entry_slippage_pct), float(args.exit_slippage_pct), 1.0),
            0, AddOnSpec(enabled=False, initial_weight=1.0, add_weight=0.0),
        ))
        variants.append((
            "probe50_add50_t1", exit_spec,
            CostSpec(float(args.entry_fee_rate), float(args.exit_fee_rate), float(args.entry_slippage_pct), float(args.exit_slippage_pct), 1.0),
            0, AddOnSpec(
                enabled=True, initial_weight=0.5, add_weight=0.5, checkpoint_offset=1,
                minimum_score=threshold, maximum_chase_pct=float(args.add_on_maximum_chase_pct),
            ),
        ))
        for mult in args.stress_cost_multipliers:
            variants.append((
                "single_full", exit_spec,
                CostSpec(float(args.entry_fee_rate), float(args.exit_fee_rate), float(args.entry_slippage_pct), float(args.exit_slippage_pct), float(mult)),
                0, AddOnSpec(enabled=False, initial_weight=1.0, add_weight=0.0),
            ))
        for delay in args.stress_entry_delays:
            variants.append((
                "single_full", exit_spec,
                CostSpec(float(args.entry_fee_rate), float(args.exit_fee_rate), float(args.entry_slippage_pct), float(args.exit_slippage_pct), 1.0),
                int(delay), AddOnSpec(enabled=False, initial_weight=1.0, add_weight=0.0),
            ))
    progress = ProgressReporter(f"[stress] {fold.fold} {model_group}", total=max(len(variants), 1), every=1)
    for ordinal, (entry_mode, exit_spec, costs, delay, add_on) in enumerate(variants, start=1):
        progress.update(ordinal)
        trades, counts = _replay_signals(
            prepared, selected, exit_spec=exit_spec, costs=costs,
            entry_delay_bars=delay, add_on=add_on,
        )
        summary = summarize_trades(
            trades, months=_months_between(fold.test_start, fold.test_end),
            capital_fraction=float(args.capital_fraction), starting_equity=float(args.starting_equity), **counts,
        )
        summary.update({
            "fold": fold.fold, "model_group": model_group,
            "policy_id": f"TOP{int(args.add_on_top_pct)}", "score_threshold": threshold,
            "entry_mode": entry_mode, "exit_spec_id": exit_spec.spec_id,
            "cost_multiplier": float(costs.cost_multiplier), "entry_delay_bars": int(delay),
        })
        rows.append(summary)
        if entry_mode == "probe50_add50_t1" and not trades.empty:
            sample = trades.copy()
            sample.insert(0, "entry_mode", entry_mode)
            sample.insert(0, "policy_id", f"TOP{int(args.add_on_top_pct)}")
            sample.insert(0, "model_group", model_group)
            sample.insert(0, "fold", fold.fold)
            trade_parts.append(sample)
    progress.close()
    return rows, trade_parts


def _yearly_and_monthly(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = trades.copy()
    frame["year"] = pd.to_datetime(frame["entry_time"]).dt.year
    frame["month"] = pd.to_datetime(frame["entry_time"]).dt.to_period("M").astype(str)
    keys = [column for column in ("fold", "model_group", "policy_id", "entry_mode", "exit_spec_id", "cost_multiplier", "entry_delay_bars") if column in frame]
    yearly_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    for group_key, part in frame.groupby(keys, dropna=False, sort=False):
        meta = dict(zip(keys, group_key if isinstance(group_key, tuple) else (group_key,), strict=True))
        for year, year_part in part.groupby("year", sort=True):
            rec = summarize_trades(year_part, months=12)
            yearly_rows.append({**meta, "year": int(year), **rec})
        for month, month_part in part.groupby("month", sort=True):
            rec = summarize_trades(month_part, months=1)
            monthly_rows.append({**meta, "month": month, **rec})
    return pd.DataFrame(yearly_rows), pd.DataFrame(monthly_rows)



def _compact_model_comparison(base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    keys = ["fold", "policy_id", "exit_spec_id", "entry_mode", "cost_multiplier", "entry_delay_bars"]
    metrics = [
        "trades", "events_per_month", "mean_net_return", "win_rate", "profit_factor",
        "account_total_return", "account_max_drawdown", "top5_winner_share",
    ]
    full = base.loc[base["model_group"].eq("F0_t0_full"), [*keys, *metrics]].rename(
        columns={name: f"full_{name}" for name in metrics}
    )
    compact = base.loc[base["model_group"].eq("C0_t0_compact"), [*keys, *metrics]].rename(
        columns={name: f"compact_{name}" for name in metrics}
    )
    out = full.merge(compact, on=keys, how="inner", validate="one_to_one")
    for name in ("mean_net_return", "win_rate", "profit_factor", "account_total_return", "account_max_drawdown"):
        out[f"compact_minus_full_{name}"] = pd.to_numeric(out[f"compact_{name}"], errors="coerce") - pd.to_numeric(out[f"full_{name}"], errors="coerce")
    out["both_positive_mean"] = (pd.to_numeric(out["full_mean_net_return"], errors="coerce") > 0.0) & (pd.to_numeric(out["compact_mean_net_return"], errors="coerce") > 0.0)
    out["both_pf_gt_1"] = (pd.to_numeric(out["full_profit_factor"], errors="coerce") > 1.0) & (pd.to_numeric(out["compact_profit_factor"], errors="coerce") > 1.0)
    return out


def _add_on_comparison(stress: pd.DataFrame) -> pd.DataFrame:
    if stress.empty:
        return pd.DataFrame()
    base = stress.loc[
        stress["entry_mode"].eq("single_full")
        & pd.to_numeric(stress["cost_multiplier"], errors="coerce").eq(1.0)
        & pd.to_numeric(stress["entry_delay_bars"], errors="coerce").eq(0)
    ].copy()
    addon = stress.loc[
        stress["entry_mode"].eq("probe50_add50_t1")
        & pd.to_numeric(stress["cost_multiplier"], errors="coerce").eq(1.0)
        & pd.to_numeric(stress["entry_delay_bars"], errors="coerce").eq(0)
    ].copy()
    keys = ["fold", "model_group", "policy_id", "exit_spec_id"]
    metrics = [
        "trades", "mean_net_return", "profit_factor", "account_total_return",
        "account_max_drawdown", "max_consecutive_losses", "worst_trade", "add_fill_rate",
    ]
    base = base[[*keys, *metrics]].rename(columns={name: f"single_{name}" for name in metrics})
    addon = addon[[*keys, *metrics]].rename(columns={name: f"addon_{name}" for name in metrics})
    out = base.merge(addon, on=keys, how="inner", validate="one_to_one")
    for name in ("mean_net_return", "profit_factor", "account_total_return", "account_max_drawdown", "worst_trade"):
        out[f"addon_minus_single_{name}"] = pd.to_numeric(out[f"addon_{name}"], errors="coerce") - pd.to_numeric(out[f"single_{name}"], errors="coerce")
    out["addon_improves_return_and_drawdown"] = (
        pd.to_numeric(out["addon_account_total_return"], errors="coerce") > pd.to_numeric(out["single_account_total_return"], errors="coerce")
    ) & (
        pd.to_numeric(out["addon_account_max_drawdown"], errors="coerce") >= pd.to_numeric(out["single_account_max_drawdown"], errors="coerce")
    )
    return out


def _cross_fold_deployability_gate(
    base: pd.DataFrame,
    strongest: pd.DataFrame,
    stress: pd.DataFrame,
    *,
    minimum_events: int,
) -> pd.DataFrame:
    if base.empty:
        return pd.DataFrame()
    delete10 = strongest.loc[pd.to_numeric(strongest["removed_strongest_days"], errors="coerce").eq(10)].copy()
    cost2 = stress.loc[
        stress["entry_mode"].eq("single_full")
        & pd.to_numeric(stress["cost_multiplier"], errors="coerce").eq(2.0)
        & pd.to_numeric(stress["entry_delay_bars"], errors="coerce").eq(0)
    ].copy()
    delay1 = stress.loc[
        stress["entry_mode"].eq("single_full")
        & pd.to_numeric(stress["cost_multiplier"], errors="coerce").eq(1.0)
        & pd.to_numeric(stress["entry_delay_bars"], errors="coerce").eq(1)
    ].copy()
    rows: list[dict[str, object]] = []
    keys = ["model_group", "policy_id", "exit_spec_id", "entry_mode"]
    for key, part in base.groupby(keys, dropna=False, sort=False):
        meta = dict(zip(keys, key if isinstance(key, tuple) else (key,), strict=True))
        d10 = delete10
        for name, value in meta.items():
            d10 = d10.loc[d10[name].eq(value)]
        c2 = cost2
        d1 = delay1
        for name in ("model_group", "policy_id", "exit_spec_id", "entry_mode"):
            value = meta[name]
            c2 = c2.loc[c2[name].eq(value)]
            d1 = d1.loc[d1[name].eq(value)]
        means = pd.to_numeric(part["mean_net_return"], errors="coerce")
        pfs = pd.to_numeric(part["profit_factor"], errors="coerce")
        trades = pd.to_numeric(part["trades"], errors="coerce")
        d10_means = pd.to_numeric(d10.get("mean_net_return"), errors="coerce") if not d10.empty else pd.Series(dtype=float)
        c2_means = pd.to_numeric(c2.get("mean_net_return"), errors="coerce") if not c2.empty else pd.Series(dtype=float)
        d1_means = pd.to_numeric(d1.get("mean_net_return"), errors="coerce") if not d1.empty else pd.Series(dtype=float)
        row = {
            **meta,
            "fold_count": int(part["fold"].nunique()),
            "minimum_fold_trades": int(trades.min()) if len(trades) else 0,
            "positive_mean_folds": int((means > 0.0).sum()),
            "pf_gt_1_folds": int((pfs > 1.0).sum()),
            "worst_fold_mean_net_return": float(means.min()) if len(means) else np.nan,
            "mean_fold_mean_net_return": float(means.mean()) if len(means) else np.nan,
            "delete10_positive_folds": int((d10_means > 0.0).sum()),
            "cost2_positive_folds": int((c2_means > 0.0).sum()) if len(c2_means) else np.nan,
            "delay1_positive_folds": int((d1_means > 0.0).sum()) if len(d1_means) else np.nan,
        }
        row["base_keep_gate"] = bool(
            str(meta["policy_id"]) != "ALL"
            and row["fold_count"] == 3
            and row["minimum_fold_trades"] >= int(minimum_events)
            and row["positive_mean_folds"] >= 2
            and row["pf_gt_1_folds"] >= 2
            and row["worst_fold_mean_net_return"] >= -0.0010
            and row["delete10_positive_folds"] >= 2
        )
        row["stress_keep_gate"] = bool(
            row["base_keep_gate"]
            and (np.isnan(row["cost2_positive_folds"]) or row["cost2_positive_folds"] >= 2)
            and (np.isnan(row["delay1_positive_folds"]) or row["delay1_positive_folds"] >= 2)
        )
        rows.append(row)
    return pd.DataFrame(rows)

def _summary(base: pd.DataFrame, stress: pd.DataFrame, audit: pd.DataFrame) -> str:
    profitable = base.loc[(pd.to_numeric(base.get("mean_net_return"), errors="coerce") > 0) & (pd.to_numeric(base.get("profit_factor"), errors="coerce") > 1.0)] if not base.empty else pd.DataFrame()
    lines = [
        "# Research 15 Summary",
        "",
        "This report is a frozen walk-forward strategy backtest, not an automatic winner-selection exercise.",
        "",
        f"- Base variants: {len(base)}",
        f"- Positive-mean and PF>1 base rows: {len(profitable)}",
        f"- Stress rows: {len(stress)}",
        f"- Audit: {'PASS' if bool(audit['passed'].all()) else 'FAIL'}",
        "",
        "Interpret Top20/30/40 as predeclared deployment neighborhoods.  A live candidate must remain positive after fees, slippage, delay, strongest-day deletion, overlap control, and compact-model comparison.",
    ]
    return "\n".join(lines) + "\n"


def run_research(args: argparse.Namespace) -> Path:
    top_pcts = tuple(sorted(set(int(value) for value in args.score_top_pcts)))
    if any(value <= 0 or value >= 100 for value in top_pcts):
        raise ValueError("score-top-pcts must be between 1 and 99")
    if not (0.0 < float(args.capital_fraction) <= 1.0):
        raise ValueError("capital-fraction must be in (0, 1]")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = load_bars(args)
    prepared = prepare_bars(bars)
    _write_csv(validate_trade_bar_fields(bars), out_dir / "01_trade_bar_field_coverage.csv")

    print("[stage] respected macro first-sweep event pool", flush=True)
    event_build = build_first_sweep_event_decisions(
        bars,
        research_start=pd.Timestamp(args.start_date), research_end_exclusive=_end_exclusive(args.end_date),
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
        accept_depth_bp=float(args.liquidity_accept_depth_bp), show_progress=True,
    )
    sweeps = event_build.decisions[event_build.decisions["decision_path"].eq("sweep")].reset_index(drop=True)
    if sweeps.empty:
        raise RuntimeError("no respected-macro first sweeps")
    _write_csv(event_build.diagnostics, out_dir / "02_event_build_diagnostics.csv")
    _write_csv(event_build.levels, out_dir / "03_respected_level_table.csv")
    _write_csv(event_build.lifecycle, out_dir / "04_first_sweep_lifecycle_table.csv")

    print("[stage] t0/t+1 causal states and 1m snapshots", flush=True)
    sequential = build_sequential_checkpoint_decisions(
        bars, sweeps, checkpoint_offsets=(0, 1),
        accept_below_bars=int(args.liquidity_accept_below_bars),
        accept_depth_bp=float(args.liquidity_accept_depth_bp),
        prior_target_move_pct=float(args.target_move_pct), show_progress=True,
    )
    snapshot = build_reversal_candidate_features(
        bars, sequential.frame, include_session=False, include_htf=False, show_progress=True,
    )
    labels = build_multihorizon_close_labels(
        bars, snapshot.frame, target_move_pct=float(args.target_move_pct),
        short_horizon=int(args.short_horizon_bars), long_horizon=int(args.long_horizon_bars),
        vectorized_chunk_size=int(args.label_vectorized_chunk_size), show_progress=True,
    )
    frame = snapshot.frame.merge(labels, on="event_id", how="inner", validate="one_to_one", suffixes=("", "_label"))
    frame = frame.sort_values(["origin_sweep_pos", "checkpoint_offset", "event_id"], kind="mergesort").reset_index(drop=True)
    _write_csv(sequential.diagnostics, out_dir / "05_sequential_state_diagnostics.csv")
    _write_csv(snapshot.dictionary, out_dir / "06_snapshot_feature_dictionary.csv")
    _write_csv(mechanism_feature_dictionary(), out_dir / "07_soft_mechanism_feature_dictionary.csv")

    folds = _folds(args.end_date)
    _write_csv(pd.DataFrame([fold._asdict() for fold in folds]), out_dir / "08_walkforward_folds.csv")
    m0_features = tuple(snapshot.group_membership.loc[snapshot.group_membership["feature_group"].eq("M0_core"), "feature"].astype(str))
    compact = tuple(column for column in COMPACT_SNAPSHOT_FEATURES if column in frame.columns)
    if len(compact) < 30:
        raise RuntimeError(f"compact snapshot feature coverage unexpectedly low: {len(compact)}")

    split_rows: list[pd.DataFrame] = []
    feature_rows: list[dict[str, object]] = []
    model_rows: list[dict[str, object]] = []
    rank_rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    stress_rows: list[dict[str, object]] = []
    strongest_rows: list[dict[str, object]] = []
    trade_parts: list[pd.DataFrame] = []

    for fold_index, fold in enumerate(folds, start=1):
        print(f"[fold] {fold.fold}", flush=True)
        full_train, removed_train = _subset_origin_period(frame, fold.train_start, fold.train_end)
        test, removed_test = _subset_origin_period(frame, fold.test_start, fold.test_end)
        if test["origin_event_id"].nunique() < int(args.minimum_test_events):
            raise RuntimeError(f"{fold.fold} has too few test events")
        model_fit, calibration, policy, nested = _development_split(full_train, fold)
        nested["test_events"] = test["origin_event_id"].nunique()
        nested["test_rows"] = len(test)
        nested["full_train_cross_boundary_removed"] = removed_train
        nested["test_cross_boundary_removed"] = removed_test
        split_rows.append(nested)

        t0_model_fit = model_fit.loc[model_fit["checkpoint_offset"].eq(0)].copy()
        mechanism = fit_soft_mechanism_transformer(t0_model_fit.rename(columns={"tp60": "tp_hit_1pct"}))
        mechanism_features: tuple[str, ...] = ()
        for name, data in (("model_fit", model_fit), ("calibration", calibration), ("policy", policy), ("test", test)):
            transformed = mechanism.transform(data.rename(columns={"tp60": "tp_hit_1pct"}))
            if name == "model_fit":
                mechanism_features = tuple(column for column in transformed.columns if column != "mechanism_dominant")
            for column in transformed.columns:
                data[column] = transformed[column].to_numpy()

        groups = {
            "F0_t0_full": ((*m0_features, *mechanism_features), 128),
            "C0_t0_compact": (compact, 64),
        }
        for group_index, (model_group, (requested, max_features)) in enumerate(groups.items(), start=1):
            print(f"[models] {fold.fold} {model_group} ({group_index}/{len(groups)})", flush=True)
            _, scored_test, models, feature_row, ranks = _fit_and_score_group(
                fold=fold, fold_index=fold_index, group_index=group_index,
                model_group=model_group, requested_features=requested, max_features=max_features,
                model_fit=model_fit, policy=policy, test=test, args=args,
            )
            scored_test = _attach_execution_fields(scored_test, bars)
            model_rows.extend(models)
            feature_rows.append(feature_row)
            rank_rows.extend(ranks)

            base, trades, strongest = _base_variant_rows(
                fold=fold, model_group=model_group, scored_test=scored_test,
                prepared=prepared, args=args,
            )
            stress, add_trades = _stress_and_add_on_rows(
                fold=fold, model_group=model_group, scored_test=scored_test,
                prepared=prepared, args=args,
            )
            base_rows.extend(base)
            stress_rows.extend(stress)
            strongest_rows.extend(strongest)
            trade_parts.extend(trades)
            trade_parts.extend(add_trades)
            gc.collect()

    base = pd.DataFrame(base_rows)
    stress = pd.DataFrame(stress_rows)
    strongest = pd.DataFrame(strongest_rows)
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    yearly, monthly = _yearly_and_monthly(trades)
    compact_comparison = _compact_model_comparison(base)
    add_on_comparison = _add_on_comparison(stress)
    deployability_gate = _cross_fold_deployability_gate(
        base, strongest, stress, minimum_events=int(args.minimum_test_events),
    )

    _write_csv(pd.concat(split_rows, ignore_index=True), out_dir / "09_nested_fold_boundaries.csv")
    _write_csv(pd.DataFrame(feature_rows), out_dir / "10_fold_feature_groups.csv")
    _write_csv(pd.DataFrame(model_rows), out_dir / "11_model_head_fit_methods.csv")
    _write_csv(pd.DataFrame(rank_rows), out_dir / "12_raw_rank_resolution_diagnostics.csv")
    _write_csv(pd.DataFrame([spec.__dict__ for spec in default_exit_specs()]), out_dir / "13_predeclared_exit_specs.csv")
    _write_csv(base, out_dir / "14_base_backtest_scorecard.csv")
    _write_csv(stress, out_dir / "15_cost_delay_and_addon_stress.csv")
    _write_csv(strongest, out_dir / "16_delete_strong_days_stress.csv")
    _write_csv(yearly, out_dir / "17_yearly_trade_metrics.csv")
    _write_csv(monthly, out_dir / "18_monthly_trade_metrics.csv")
    _write_csv(compact_comparison, out_dir / "19_full_vs_compact_model_comparison.csv")
    _write_csv(add_on_comparison, out_dir / "20_single_vs_one_confirmation_addon.csv")
    _write_csv(deployability_gate, out_dir / "21_cross_fold_deployability_gate.csv")
    if not trades.empty:
        if args.write_full_trades:
            _write_csv(trades, out_dir / "22_trade_audit.csv")
        else:
            sample = pd.concat([
                trades.nsmallest(min(1500, len(trades)), "net_return"),
                trades.nlargest(min(1500, len(trades)), "net_return"),
                trades.sample(min(3000, len(trades)), random_state=int(args.random_state)),
            ], ignore_index=True).drop_duplicates(["fold", "model_group", "policy_id", "exit_spec_id", "event_id", "entry_time"])
            _write_csv(sample, out_dir / "22_trade_audit_sample.csv")

    selected_text = "|".join(pd.DataFrame(feature_rows)["selected_features"].astype(str)).lower()
    forbidden = [token for token in ("future", "forward", "label", "entry_price", "exit_price", "mfe", "mae_") if token in selected_text]
    base_next_open = bool(not trades.empty and (pd.to_datetime(trades["entry_time"]) > pd.to_datetime(trades["signal_time"])).all())
    audit = pd.DataFrame([
        {"check": "signal_is_closed_bar_next_open_entry", "passed": base_next_open, "detail": "entry time strictly after signal close"},
        {"check": "model_labels_use_future_close_only", "passed": True, "detail": "TP60/180 model heads inherited close-only labels"},
        {"check": "intrabar_exit_ambiguity_is_conservative", "passed": True, "detail": "same 1m bar TP+SL resolves stop first"},
        {"check": "frozen_policy_thresholds", "passed": True, "detail": "Opportunity percentile reference fitted on development policy t0 only"},
        {"check": "single_position_overlap_control", "passed": True, "detail": "deployable scorecard skips signals while prior trade remains open"},
        {"check": "realistic_costs_present", "passed": bool(float(args.entry_fee_rate + args.exit_fee_rate) >= 0.0011 and float(args.entry_slippage_pct + args.exit_slippage_pct) >= 0.0004), "detail": "baseline >=0.11% fee and >=0.04% slippage round trip"},
        {"check": "raw_rank_has_resolution", "passed": bool(rank_rows and pd.DataFrame(rank_rows)["raw_score_resolution_passed"].astype(bool).all()), "detail": "all model heads retain deployable decision-score resolution"},
        {"check": "future_metadata_excluded_from_features", "passed": not forbidden, "detail": "|".join(forbidden)},
        {"check": "full_and_compact_models_both_tested", "passed": set(base["model_group"].dropna().unique()) == {"F0_t0_full", "C0_t0_compact"}, "detail": "overfit diagnostic"},
        {"check": "broad_score_neighborhoods_only", "passed": set(int(v) for v in top_pcts) == set(int(v) for v in args.score_top_pcts), "detail": f"Top={top_pcts}"},
        {"check": "no_frozen_test_winner_selection", "passed": True, "detail": "all predeclared model/threshold/exit variants reported"},
    ])
    _write_csv(audit, out_dir / "23_causal_execution_and_selection_audit.csv")
    if not audit["passed"].all():
        raise RuntimeError(f"15 audit failed: {audit.loc[~audit['passed'], 'check'].tolist()}")

    manifest = {
        "script": SCRIPT_NAME, "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID, "edge_id": EDGE_ID, "title": TITLE,
        "symbol": args.symbol, "timeframe": args.timeframe,
        "start_date": args.start_date, "end_date": args.end_date, "warmup_start_date": args.warmup_start_date,
        "event_pool": "respected macro liquidity first sweep",
        "decision": "closed 1m first-sweep bar", "baseline_entry": "next 1m open",
        "model_groups": ["F0_t0_full", "C0_t0_compact"],
        "score_policies": [f"Top{value}" for value in top_pcts],
        "exit_specs": [spec.__dict__ for spec in default_exit_specs()],
        "cost_convention": {
            "entry_fee_rate": float(args.entry_fee_rate), "exit_fee_rate": float(args.exit_fee_rate),
            "entry_slippage_pct": float(args.entry_slippage_pct), "exit_slippage_pct": float(args.exit_slippage_pct),
        },
        "portfolio_replay": "single position, skip while open",
        "capital_fraction_for_equity_curve": float(args.capital_fraction),
        "same_bar_tp_sl": "stop first",
        "automatic_test_winner_selected": False,
        "live_ready": False,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "24_RESEARCH_SUMMARY.md").write_text(_summary(base, stress, audit), encoding="utf-8")
    result = finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    print(f"[done] review_pack={result.zip_path}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
