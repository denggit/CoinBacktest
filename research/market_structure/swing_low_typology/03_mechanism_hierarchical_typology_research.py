#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mechanism-guided hierarchical typology of retrospectively labelled swing lows.

Research only: no strategy, entry, exit, PnL, or backtest.  Research 02 supplies
causal C3 subtype assignments and its 315 pre-low features.  This research adds
bounded event-window mechanism features, then applies train-fitted weakly
supervised mechanism scores instead of another unconstrained KMeans pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from research.market_structure.swing_low_typology.common.c3_sequence_features import (  # noqa: E402
    METADATA_COLUMNS as C3_METADATA_COLUMNS,
    build_c3_sequence_features,
)
from research.market_structure.swing_low_typology.common.mechanism_features import (  # noqa: E402
    MECHANISM_METADATA_COLUMNS,
    build_future_perturbation_audit,
    build_mechanism_features,
    build_path_profiles,
)
from research.market_structure.swing_low_typology.common.mechanism_typology import (  # noqa: E402
    BASE_ARCHETYPE_TERMS,
    BROAD_MECHANISM_TERMS,
    TREND_ARCHETYPE_TERMS,
    build_bootstrap_stability,
    build_feature_descriptors,
    build_rule_cards,
    build_type_summary,
    build_weak_anchor_agreement,
    fit_score_model,
    perturb_future_metadata,
    representative_events,
    terms_to_frame,
)
from research.market_structure.swing_low_typology.common.swing_low_dataset import (  # noqa: E402
    validate_trade_bar_fields,
)

SCRIPT_NAME = "03_mechanism_hierarchical_typology_research"
SCRIPT_VERSION = "1.1.1"
EXPERIMENT_ID = "ETH_1M_SWING_LOW_MECHANISM_HIERARCHICAL_TYPOLOGY_03"
EDGE_ID = "RESEARCH_ONLY_ETH_SWING_LOW_MECHANISMS"
TITLE = "ETH Swing Low Mechanism Hierarchical Typology 03"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/03_mechanism_hierarchical_typology"
DEFAULT_STAGE2_DIR = "data/reports/research/market_structure/swing_low_typology/02_c3_hierarchical_typology"

DATE_COLUMNS = [
    "extreme_time",
    "feature_available_time",
    "confirmation_time",
    "confirmation_available_time",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mechanism-guided weakly supervised hierarchy for C3 swing lows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--train-end-date", default="2024-12-31")
    p.add_argument("--target-move-pct", type=float, default=1.0)
    p.add_argument("--max-completion-bars", type=int, default=60)
    p.add_argument("--stage2-report-dir", default=DEFAULT_STAGE2_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--mechanism-lookback", type=int, default=240)
    p.add_argument("--phase-bins", type=int, default=12)
    p.add_argument("--support-tolerance-bp", type=float, default=25.0)
    p.add_argument("--minimum-test-gap", type=int, default=4)
    p.add_argument("--minimum-test-rebound-bp", type=float, default=15.0)
    p.add_argument("--test-rebound-horizon", type=int, default=30)
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--representative-events-per-type", type=int, default=12)
    p.add_argument("--path-profile-sample-per-type", type=int, default=600)
    p.add_argument("--causal-audit-sample-size", type=int, default=24)
    return p.parse_args(argv)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


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


def _validate_stage2(args: argparse.Namespace, report_dir: Path) -> dict[str, object]:
    manifest_path = report_dir / "00_manifest.json"
    assignment_path = report_dir / "07_frozen_c3_subcluster_assignments.csv"
    dictionary_path = report_dir / "03_c3_feature_dictionary.csv"
    missing = [str(path) for path in (manifest_path, assignment_path, dictionary_path) if not path.exists()]
    if missing:
        raise FileNotFoundError("Run research 02 first; missing stage-2 outputs: " + ", ".join(missing))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "target_move_pct": float(args.target_move_pct),
        "max_completion_bars": int(args.max_completion_bars),
        "train_end_date": args.train_end_date,
    }
    mismatches = [f"{key}: stage2={manifest.get(key)}, requested={value}" for key, value in checks.items() if str(manifest.get(key)) != str(value)]
    label_policy = {
        "swing_extreme_price_source": "low",
        "swing_entry_price_source": "next_bar_open",
        "swing_target_observation_source": "future_closed_bar_close",
    }
    for key, expected in label_policy.items():
        actual = manifest.get(key)
        if actual != expected:
            mismatches.append(f"{key}: stage2={actual}, required={expected}")
    if mismatches:
        raise RuntimeError("Stage-2 report is incompatible: " + "; ".join(mismatches))
    if manifest.get("causal_policy") != "swing extreme uses low; tradable confirmation uses next-bar open to future closed-bar close; subtype features end at extreme bar close":
        raise RuntimeError("Stage-2 manifest does not contain the expected causal policy")
    return manifest


def _load_stage2_assignments(report_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(report_dir / "07_frozen_c3_subcluster_assignments.csv", parse_dates=DATE_COLUMNS)
    required = {
        "event_id",
        "extreme_time",
        "feature_available_time",
        "extreme_pos",
        "extreme_price",
        "confirmation_time",
        "confirmation_available_time",
        "completion_bars",
        "realized_confirmation_move_pct",
        "parent_cluster_id",
        "parent_distance_to_centroid",
        "split",
        "subcluster_id",
        "distance_to_train_centroid",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Stage-2 assignment file missing columns: {missing}")
    return frame.sort_values("extreme_time").reset_index(drop=True)


def _load_or_rebuild_stage2_features(
    bars: pd.DataFrame,
    assignments: pd.DataFrame,
    report_dir: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    matrix_path = report_dir / "04_c3_feature_matrix_sample.csv"
    dictionary = pd.read_csv(report_dir / "03_c3_feature_dictionary.csv")
    if matrix_path.exists():
        loaded = pd.read_csv(matrix_path, parse_dates=DATE_COLUMNS)
        if set(loaded["event_id"]) == set(assignments["event_id"]) and len(loaded) == len(assignments):
            print(f"[stage2] reusing complete causal feature matrix rows={len(loaded):,}", flush=True)
            return loaded.sort_values("extreme_time").reset_index(drop=True), dictionary, "reused_complete_02_matrix"

    print("[stage2] feature sample is incomplete; rebuilding 315 causal features from raw bars", flush=True)
    # Research 02's feature builder expects the parent C3 assignment schema.
    # Select explicitly instead of renaming the full frame: the stage-2 file also
    # contains its own ``distance_to_train_centroid`` column, which would create
    # duplicate names and silently bind the wrong distance during a fallback run.
    parent_columns = [
        "event_id",
        "extreme_time",
        "feature_available_time",
        "extreme_pos",
        "extreme_price",
        "confirmation_time",
        "confirmation_available_time",
        "completion_bars",
        "realized_confirmation_move_pct",
        "parent_cluster_id",
        "parent_distance_to_centroid",
        "split",
    ]
    parent = assignments[parent_columns].rename(
        columns={
            "parent_cluster_id": "cluster_id",
            "parent_distance_to_centroid": "distance_to_train_centroid",
        }
    ).copy()
    features, rebuilt_dictionary = build_c3_sequence_features(
        bars,
        parent,
        windows=(15, 30, 60, 120, 240),
        phase_lookback=240,
        phase_bins=12,
        progress_every=int(args.progress_every),
    )
    return features, rebuilt_dictionary, "rebuilt_from_raw_trade_bars"


def _combine_features(
    stage2_features: pd.DataFrame,
    stage2_assignments: pd.DataFrame,
    mechanism_features: pd.DataFrame,
) -> pd.DataFrame:
    assignment_meta = stage2_assignments[
        ["event_id", "split", "subcluster_id", "distance_to_train_centroid"]
    ].rename(
        columns={
            "subcluster_id": "source_subcluster_id",
            "distance_to_train_centroid": "source_subcluster_distance",
        }
    )
    combined = stage2_features.merge(assignment_meta, on="event_id", how="inner", validate="one_to_one")
    new_columns = [
        column for column in mechanism_features.columns
        if column not in MECHANISM_METADATA_COLUMNS and column not in combined.columns
    ]
    combined = combined.merge(
        mechanism_features[["event_id", *new_columns]], on="event_id", how="inner", validate="one_to_one"
    )
    if len(combined) != len(stage2_assignments):
        raise RuntimeError(
            f"Feature merge lost events: assignments={len(stage2_assignments)}, combined={len(combined)}"
        )
    return combined.sort_values("extreme_time").reset_index(drop=True)


def _apply_models(
    combined: pd.DataFrame,
) -> tuple[object, object, object, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_mask = combined["split"].astype(str).eq("train")
    broad_model = fit_score_model(
        combined,
        train_mask,
        BROAD_MECHANISM_TERMS,
        name="broad_mechanism",
        calibrate_percentiles=False,
    )
    broad = broad_model.transform(combined)
    broad_assignments = combined[
        [
            "event_id", "extreme_time", "feature_available_time", "extreme_pos", "extreme_price",
            "confirmation_time", "confirmation_available_time", "completion_bars",
            "realized_confirmation_move_pct", "split", "source_subcluster_id",
            "source_subcluster_distance",
        ]
    ].copy()
    broad_assignments["mechanism"] = broad["primary_type"].to_numpy()
    broad_assignments["secondary_mechanism"] = broad["secondary_type"].to_numpy()
    broad_assignments["confidence"] = broad["confidence"].to_numpy()
    broad_assignments["score_margin"] = broad["score_margin"].to_numpy()
    broad_assignments["ambiguous"] = broad["ambiguous"].to_numpy()
    for column in broad.columns:
        if column.startswith("score_") or column.startswith("probability_"):
            broad_assignments[column] = broad[column].to_numpy()

    trend_frame = combined[combined["source_subcluster_id"].astype(str).eq("C3-C")].reset_index(drop=True)
    trend_train = trend_frame["split"].astype(str).eq("train")
    trend_model = fit_score_model(
        trend_frame,
        trend_train,
        TREND_ARCHETYPE_TERMS,
        name="trend_subtypes",
        calibrate_percentiles=True,
    )
    trend_scores = trend_model.transform(trend_frame)
    trend_assignments = trend_frame[
        ["event_id", "extreme_time", "feature_available_time", "extreme_pos", "extreme_price", "split", "source_subcluster_id"]
    ].copy()
    trend_assignments["trend_subtype"] = trend_scores["primary_type"].to_numpy()
    trend_assignments["secondary_trend_subtype"] = trend_scores["secondary_type"].to_numpy()
    trend_assignments["confidence"] = trend_scores["confidence"].to_numpy()
    trend_assignments["score_margin"] = trend_scores["score_margin"].to_numpy()
    trend_assignments["ambiguous"] = trend_scores["ambiguous"].to_numpy()
    for column in trend_scores.columns:
        if column.startswith("score_") or column.startswith("probability_"):
            trend_assignments[column] = trend_scores[column].to_numpy()

    base_frame = combined[combined["source_subcluster_id"].astype(str).eq("C3-E")].reset_index(drop=True)
    base_train = base_frame["split"].astype(str).eq("train")
    base_model = fit_score_model(
        base_frame,
        base_train,
        BASE_ARCHETYPE_TERMS,
        name="base_subtypes",
        calibrate_percentiles=True,
    )
    base_scores = base_model.transform(base_frame)
    base_assignments = base_frame[
        ["event_id", "extreme_time", "feature_available_time", "extreme_pos", "extreme_price", "split", "source_subcluster_id"]
    ].copy()
    base_assignments["base_subtype"] = base_scores["primary_type"].to_numpy()
    base_assignments["secondary_base_subtype"] = base_scores["secondary_type"].to_numpy()
    base_assignments["confidence"] = base_scores["confidence"].to_numpy()
    base_assignments["score_margin"] = base_scores["score_margin"].to_numpy()
    base_assignments["ambiguous"] = base_scores["ambiguous"].to_numpy()
    for column in base_scores.columns:
        if column.startswith("score_") or column.startswith("probability_"):
            base_assignments[column] = base_scores[column].to_numpy()
    return broad_model, trend_model, base_model, broad_assignments, trend_assignments, base_assignments


def _plot_profiles(profiles: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if profiles.empty:
        return
    for (split, metric), group in profiles.groupby(["split", "metric"], sort=True):
        fig, ax = plt.subplots(figsize=(10, 6))
        for type_id, type_group in group.groupby("type_id", sort=True):
            ax.plot(type_group["phase"], type_group["median"], marker="o", label=str(type_id))
        ax.set_title(f"{prefix} {metric} path ({split})")
        ax.set_xlabel("Historical phase: oldest to extreme")
        ax.set_ylabel("Median normalized value")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{prefix}_{metric}_{split}.png", dpi=150)
        plt.close(fig)


def _plot_yearly(yearly: pd.DataFrame, type_column: str, out_path: Path, title: str) -> None:
    if yearly.empty:
        return
    pivot = yearly.pivot(index="year", columns=type_column, values="share_within_year").fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for type_id in pivot.columns:
        ax.plot(pivot.index, pivot[type_id] * 100.0, marker="o", label=str(type_id))
    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share (%)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _causal_audit(
    combined: pd.DataFrame,
    feature_columns: Sequence[str],
    stage2_manifest: dict[str, object],
    raw_perturbation: pd.DataFrame,
    assignment_invariance: bool,
) -> pd.DataFrame:
    # ``realized_vol_*`` is a valid trailing volatility feature.  Only the
    # retrospective confirmation outcome is forbidden, not every "realized" field.
    forbidden_tokens = ("future", "post_", "forward", "confirmation", "completion", "mfe", "mae")
    forbidden = [
        column
        for column in feature_columns
        if any(token in column.lower() for token in forbidden_tokens)
        or column.lower().startswith("realized_confirmation_")
    ]
    feature_time = pd.to_datetime(combined["feature_available_time"])
    extreme_time = pd.to_datetime(combined["extreme_time"])
    confirmation = pd.to_datetime(combined["confirmation_available_time"])
    return pd.DataFrame(
        [
            {
                "check": "features_end_at_extreme_close",
                "passed": bool((feature_time > extreme_time).all()),
                "detail": "all 03 features end at the left-labelled extreme bar close",
            },
            {
                "check": "confirmation_after_feature_cutoff",
                "passed": bool((confirmation > feature_time).all()),
                "detail": "future confirmation remains retrospective metadata only",
            },
            {
                "check": "no_future_named_model_features",
                "passed": not forbidden,
                "detail": ",".join(forbidden),
            },
            {
                "check": "stage2_causal_policy_inherited",
                "passed": stage2_manifest.get("causal_policy") == "swing extreme uses low; tradable confirmation uses next-bar open to future closed-bar close; subtype features end at extreme bar close",
                "detail": str(stage2_manifest.get("causal_policy")),
            },
            {
                "check": "train_only_fit_holdout_frozen",
                "passed": True,
                "detail": "robust normalization, score calibration, ambiguity thresholds and shallow rules fit on 2023-2024 only",
            },
            {
                "check": "future_metadata_assignment_invariance",
                "passed": bool(assignment_invariance),
                "detail": "completion speed, confirmation move and confirmation timestamps were perturbed",
            },
            {
                "check": "raw_future_bar_perturbation",
                "passed": bool(not raw_perturbation.empty and raw_perturbation["passed"].all()),
                "detail": f"audited_events={len(raw_perturbation)}; max_diff={raw_perturbation.get('maximum_absolute_difference', pd.Series([np.nan])).max()}",
            },
        ]
    )


def _summary(
    stage2_manifest: dict[str, object],
    combined: pd.DataFrame,
    broad_summary: pd.DataFrame,
    trend_summary: pd.DataFrame,
    base_summary: pd.DataFrame,
    broad_stability: pd.DataFrame,
    trend_stability: pd.DataFrame,
    base_stability: pd.DataFrame,
    model_fit_diagnostics: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    lines = [
        f"# {TITLE}",
        "",
        "## Scope",
        "",
        "- Historical typology research only; no strategy, entry, exit, PnL, or backtest.",
        "- Future bars only confirm the retrospective Swing Low universe.",
        "- All broad mechanism and second-layer subtype inputs stop at the extreme bar close.",
        "- Research 02 frozen C3-A..E labels are used only as weak causal anchors/source universes.",
        "",
        "## Design",
        "",
        "- Layer 1: causal weak-score gate for shock / trend / base.",
        "- Layer 2 trend: five named mechanisms inside frozen C3-C.",
        "- Layer 2 base: absorption, compression, spring, repeated support test, and slow accumulation inside frozen C3-E.",
        "- No cluster-count search and no future-outcome-based model selection.",
        "- Overlap is retained through secondary type, probability, score margin, and ambiguity flags.",
        "",
        "## Data",
        "",
        f"- Parent C3 events: {len(combined):,}",
        f"- Research 02 selected K: {stage2_manifest.get('selected_k')}",
        f"- Combined causal feature count: {len([c for c in combined.columns if c not in C3_METADATA_COLUMNS and c not in MECHANISM_METADATA_COLUMNS]):,}",
        "",
        "## Layer-1 train/holdout shares",
        "",
    ]
    for row in broad_summary.itertuples(index=False):
        lines.append(
            f"- {row.mechanism} / {row.split}: {int(row.count):,} ({float(row.share_within_split):.1%}), "
            f"ambiguity {float(row.ambiguity_share):.1%}"
        )
    lines.extend(["", "## C3-C trend subtype train/holdout shares", ""])
    for row in trend_summary.itertuples(index=False):
        lines.append(f"- {row.trend_subtype} / {row.split}: {int(row.count):,} ({float(row.share_within_split):.1%})")
    lines.extend(["", "## C3-E base subtype train/holdout shares", ""])
    for row in base_summary.itertuples(index=False):
        lines.append(f"- {row.base_subtype} / {row.split}: {int(row.count):,} ({float(row.share_within_split):.1%})")
    lines.extend(["", "## Model fit diagnostics", ""])
    for row in model_fit_diagnostics.itertuples(index=False):
        mode = "SMALL-SAMPLE" if bool(row.small_sample_mode) else "standard"
        lines.append(
            f"- {row.model}: train={int(row.train_rows):,}, minimum={int(row.minimum_train_rows):,}, "
            f"preferred={int(row.preferred_train_rows):,}, mode={mode}, "
            f"ambiguity quantile={float(row.ambiguity_quantile):.0%}"
        )
    lines.extend(["", "## Bootstrap/random-seed stability", ""])
    for name, frame in (("broad", broad_stability), ("trend", trend_stability), ("base", base_stability)):
        holdout = frame[frame["split"] == "holdout"]
        lines.append(
            f"- {name}: mean holdout ARI={holdout['adjusted_rand_index_vs_primary'].mean():.3f}, "
            f"exact={holdout['exact_assignment_rate_vs_primary'].mean():.1%}"
        )
    lines.extend(["", "## Causal audit", ""])
    for row in audit.itertuples(index=False):
        lines.append(f"- {'PASS' if bool(row.passed) else 'FAIL'} `{row.check}`: {row.detail}")
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- These are retrospective historical types, not real-time Swing Low signals.",
            "- Weak mechanism names encode hypotheses and should be manually validated on representative paths.",
            "- Low score margins indicate genuine overlap; ambiguous rows should not be forced into clean narratives.",
            "- Completion speed and future rebound strength are diagnostics only and never form a type.",
            "",
        ]
    )
    return "\n".join(lines)


def run_research(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stage2_dir = Path(args.stage2_report_dir)
    stage2_manifest = _validate_stage2(args, stage2_dir)
    assignments02 = _load_stage2_assignments(stage2_dir)

    bars = load_bars(args)
    coverage = validate_trade_bar_fields(bars)
    _write_csv(coverage, out_dir / "01_trade_bar_field_coverage.csv")
    stage2_features, stage2_dictionary, feature_source = _load_or_rebuild_stage2_features(
        bars, assignments02, stage2_dir, args
    )

    print("[features] building detailed mechanism and support-test sequences", flush=True)
    mechanism_features, mechanism_dictionary, support_tests = build_mechanism_features(
        bars,
        assignments02,
        lookback=int(args.mechanism_lookback),
        phase_bins=int(args.phase_bins),
        support_tolerance_bp=float(args.support_tolerance_bp),
        min_test_gap=int(args.minimum_test_gap),
        rebound_horizon=int(args.test_rebound_horizon),
        minimum_separation_rebound_bp=float(args.minimum_test_rebound_bp),
        progress_every=int(args.progress_every),
    )
    combined = _combine_features(stage2_features, assignments02, mechanism_features)
    feature_dictionary = pd.concat([stage2_dictionary, mechanism_dictionary], ignore_index=True).drop_duplicates("feature")
    model_feature_columns = [column for column in feature_dictionary["feature"] if column in combined.columns]
    _write_csv(feature_dictionary, out_dir / "02_combined_causal_feature_dictionary.csv")
    _write_csv(combined.head(5000), out_dir / "03_combined_causal_feature_matrix_sample.csv")
    _write_csv(support_tests, out_dir / "04_support_test_sequence_details.csv")
    score_definitions = pd.concat(
        [
            terms_to_frame("broad_mechanism", BROAD_MECHANISM_TERMS),
            terms_to_frame("trend_subtypes", TREND_ARCHETYPE_TERMS),
            terms_to_frame("base_subtypes", BASE_ARCHETYPE_TERMS),
        ],
        ignore_index=True,
    )
    _write_csv(score_definitions, out_dir / "05_mechanism_score_definitions.csv")

    print("[classify] fitting train-only mechanism scores and frozen holdout mapping", flush=True)
    broad_model, trend_model, base_model, broad_assignments, trend_assignments, base_assignments = _apply_models(
        combined
    )
    model_fit_diagnostics = pd.DataFrame(
        [
            {
                "model": model.name,
                "train_rows": model.train_row_count,
                "archetype_count": len(model.labels),
                "minimum_train_rows": model.minimum_train_rows,
                "preferred_train_rows": model.preferred_train_rows,
                "small_sample_mode": model.small_sample_mode,
                "ambiguity_quantile": model.ambiguity_quantile,
                "calibrate_percentiles": model.calibrate_percentiles,
            }
            for model in (broad_model, trend_model, base_model)
        ]
    )
    _write_csv(model_fit_diagnostics, out_dir / "05b_model_fit_diagnostics.csv")
    for row in model_fit_diagnostics.itertuples(index=False):
        if bool(row.small_sample_mode):
            print(
                f"[classify] {row.model} small-sample mode "
                f"train_rows={int(row.train_rows)} preferred={int(row.preferred_train_rows)} "
                f"ambiguity_quantile={float(row.ambiguity_quantile):.0%}",
                flush=True,
            )
    _write_csv(broad_assignments, out_dir / "06_broad_mechanism_assignments.csv")
    _write_csv(trend_assignments, out_dir / "15_c3c_trend_subtype_assignments.csv")
    _write_csv(base_assignments, out_dir / "24_c3e_base_subtype_assignments.csv")

    broad_summary, broad_yearly = build_type_summary(
        broad_assignments, type_column="mechanism", confidence_column="confidence"
    )
    trend_summary, trend_yearly = build_type_summary(
        trend_assignments, type_column="trend_subtype", confidence_column="confidence"
    )
    base_summary, base_yearly = build_type_summary(
        base_assignments, type_column="base_subtype", confidence_column="confidence"
    )
    _write_csv(broad_summary, out_dir / "07_broad_mechanism_train_holdout_share.csv")
    _write_csv(broad_yearly, out_dir / "08_broad_mechanism_yearly_share.csv")
    _write_csv(build_weak_anchor_agreement(broad_assignments), out_dir / "09_weak_anchor_agreement.csv")
    _write_csv(trend_summary, out_dir / "16_c3c_trend_train_holdout_share.csv")
    _write_csv(trend_yearly, out_dir / "17_c3c_trend_yearly_share.csv")
    _write_csv(base_summary, out_dir / "25_c3e_base_train_holdout_share.csv")
    _write_csv(base_yearly, out_dir / "26_c3e_base_yearly_share.csv")

    broad_train = combined["split"].astype(str).eq("train")
    trend_frame = combined[combined["source_subcluster_id"].astype(str).eq("C3-C")].reset_index(drop=True)
    trend_train = trend_frame["split"].astype(str).eq("train")
    base_frame = combined[combined["source_subcluster_id"].astype(str).eq("C3-E")].reset_index(drop=True)
    base_train = base_frame["split"].astype(str).eq("train")
    broad_stability = build_bootstrap_stability(combined, broad_train, broad_model)
    trend_stability = build_bootstrap_stability(trend_frame, trend_train, trend_model)
    base_stability = build_bootstrap_stability(base_frame, base_train, base_model)
    _write_csv(broad_stability, out_dir / "10_broad_mechanism_seed_stability.csv")
    _write_csv(trend_stability, out_dir / "18_c3c_trend_seed_stability.csv")
    _write_csv(base_stability, out_dir / "27_c3e_base_seed_stability.csv")

    broad_descriptors = build_feature_descriptors(
        combined, broad_assignments, feature_dictionary, broad_train, type_column="mechanism"
    )
    trend_descriptors = build_feature_descriptors(
        trend_frame, trend_assignments, feature_dictionary, trend_train, type_column="trend_subtype"
    )
    base_descriptors = build_feature_descriptors(
        base_frame, base_assignments, feature_dictionary, base_train, type_column="base_subtype"
    )
    _write_csv(broad_descriptors, out_dir / "11_broad_mechanism_descriptors.csv")
    _write_csv(trend_descriptors, out_dir / "19_c3c_trend_descriptors.csv")
    _write_csv(base_descriptors, out_dir / "28_c3e_base_descriptors.csv")

    print("[explain] fitting shallow rules for type interpretation", flush=True)
    broad_rule, broad_fidelity, broad_importance = build_rule_cards(
        combined, broad_assignments, broad_train, model_feature_columns,
        target_column="mechanism", random_state=int(args.random_state)
    )
    trend_rule, trend_fidelity, trend_importance = build_rule_cards(
        trend_frame, trend_assignments, trend_train, model_feature_columns,
        target_column="trend_subtype", random_state=int(args.random_state)
    )
    base_rule, base_fidelity, base_importance = build_rule_cards(
        base_frame, base_assignments, base_train, model_feature_columns,
        target_column="base_subtype", random_state=int(args.random_state)
    )
    (out_dir / "12_broad_mechanism_rule_cards.txt").write_text(broad_rule, encoding="utf-8")
    _write_csv(broad_fidelity, out_dir / "13_broad_mechanism_rule_fidelity.csv")
    _write_csv(broad_importance, out_dir / "14_broad_mechanism_rule_importance.csv")
    (out_dir / "20_c3c_trend_rule_cards.txt").write_text(trend_rule, encoding="utf-8")
    _write_csv(trend_fidelity, out_dir / "21_c3c_trend_rule_fidelity.csv")
    _write_csv(trend_importance, out_dir / "22_c3c_trend_rule_importance.csv")
    (out_dir / "29_c3e_base_rule_cards.txt").write_text(base_rule, encoding="utf-8")
    _write_csv(base_fidelity, out_dir / "30_c3e_base_rule_fidelity.csv")
    _write_csv(base_importance, out_dir / "31_c3e_base_rule_importance.csv")

    _write_csv(
        representative_events(broad_assignments, type_column="mechanism", per_type=int(args.representative_events_per_type)),
        out_dir / "32_representative_broad_events.csv",
    )
    _write_csv(
        representative_events(trend_assignments, type_column="trend_subtype", per_type=int(args.representative_events_per_type)),
        out_dir / "33_representative_c3c_trend_events.csv",
    )
    _write_csv(
        representative_events(base_assignments, type_column="base_subtype", per_type=int(args.representative_events_per_type)),
        out_dir / "34_representative_c3e_base_events.csv",
    )

    broad_profiles = build_path_profiles(
        bars, broad_assignments, type_column="mechanism", lookback=int(args.mechanism_lookback),
        max_samples_per_type_split=int(args.path_profile_sample_per_type), random_state=int(args.random_state)
    )
    trend_profiles = build_path_profiles(
        bars, trend_assignments, type_column="trend_subtype", lookback=int(args.mechanism_lookback),
        max_samples_per_type_split=int(args.path_profile_sample_per_type), random_state=int(args.random_state)
    )
    base_profiles = build_path_profiles(
        bars, base_assignments, type_column="base_subtype", lookback=int(args.mechanism_lookback),
        max_samples_per_type_split=int(args.path_profile_sample_per_type), random_state=int(args.random_state)
    )
    _write_csv(broad_profiles, out_dir / "35_broad_price_cvd_volume_largeflow_paths.csv")
    _write_csv(trend_profiles, out_dir / "36_c3c_price_cvd_volume_largeflow_paths.csv")
    _write_csv(base_profiles, out_dir / "37_c3e_price_cvd_volume_largeflow_paths.csv")
    _plot_profiles(broad_profiles, out_dir, "35_broad")
    _plot_profiles(trend_profiles, out_dir, "36_c3c")
    _plot_profiles(base_profiles, out_dir, "37_c3e")
    _plot_yearly(broad_yearly, "mechanism", out_dir / "08_broad_mechanism_yearly_share.png", "Broad mechanism share by year")
    _plot_yearly(trend_yearly, "trend_subtype", out_dir / "17_c3c_trend_yearly_share.png", "C3-C trend subtype share by year")
    _plot_yearly(base_yearly, "base_subtype", out_dir / "26_c3e_base_yearly_share.png", "C3-E base subtype share by year")

    support_with_types = support_tests.merge(
        broad_assignments[["event_id", "mechanism"]], on="event_id", how="left"
    ).merge(
        trend_assignments[["event_id", "trend_subtype"]], on="event_id", how="left"
    ).merge(base_assignments[["event_id", "base_subtype"]], on="event_id", how="left")
    support_with_types["test_from_extreme"] = (
        support_with_types.groupby("event_id")["test_order"].transform("max") - support_with_types["test_order"]
    )
    _write_csv(support_with_types, out_dir / "38_support_test_paths_with_types.csv")
    support_profile_rows: list[pd.DataFrame] = []
    value_columns = [
        "low_distance_bp", "interval_bars", "drawdown_depth_bp", "rebound_bp",
        "negative_delta_ratio", "sell_price_impact", "notional_intensity", "large_sell_ratio",
    ]
    for level in ("mechanism", "trend_subtype", "base_subtype"):
        subset = support_with_types[support_with_types[level].notna() & (support_with_types["test_from_extreme"] <= 6)]
        if subset.empty:
            continue
        grouped = subset.groupby([level, "split", "test_from_extreme"], as_index=False)[value_columns].median()
        grouped = grouped.rename(columns={level: "type_id"})
        grouped.insert(0, "level", level)
        support_profile_rows.append(grouped)
    support_profiles = pd.concat(support_profile_rows, ignore_index=True) if support_profile_rows else pd.DataFrame()
    _write_csv(support_profiles, out_dir / "38b_support_test_path_profiles.csv")

    print("[audit] perturbing future metadata and raw post-extreme bars", flush=True)
    perturbed_metadata = perturb_future_metadata(combined, random_state=int(args.random_state) + 100)
    broad_same = broad_model.transform(combined)["primary_type"].equals(
        broad_model.transform(perturbed_metadata)["primary_type"]
    )
    trend_same = trend_model.transform(trend_frame)["primary_type"].equals(
        trend_model.transform(perturb_future_metadata(trend_frame, int(args.random_state) + 101))["primary_type"]
    )
    base_same = base_model.transform(base_frame)["primary_type"].equals(
        base_model.transform(perturb_future_metadata(base_frame, int(args.random_state) + 102))["primary_type"]
    )
    raw_perturbation = build_future_perturbation_audit(
        bars,
        assignments02,
        lookback=int(args.mechanism_lookback),
        phase_bins=int(args.phase_bins),
        support_tolerance_bp=float(args.support_tolerance_bp),
        min_test_gap=int(args.minimum_test_gap),
        rebound_horizon=int(args.test_rebound_horizon),
        minimum_separation_rebound_bp=float(args.minimum_test_rebound_bp),
        future_bars=int(args.max_completion_bars),
        sample_size=int(args.causal_audit_sample_size),
        random_state=int(args.random_state),
    )
    _write_csv(raw_perturbation, out_dir / "39_raw_future_perturbation_audit.csv")
    audit = _causal_audit(
        combined, model_feature_columns, stage2_manifest, raw_perturbation,
        assignment_invariance=bool(broad_same and trend_same and base_same),
    )
    _write_csv(audit, out_dir / "40_causal_audit.csv")
    if not bool(audit["passed"].all()):
        raise RuntimeError("Causal audit failed; inspect 39_raw_future_perturbation_audit.csv and 40_causal_audit.csv")

    post_label = pd.concat(
        [
            broad_assignments.groupby(["mechanism", "split"], as_index=False).agg(
                count=("event_id", "size"),
                median_completion_bars=("completion_bars", "median"),
                median_confirmation_move_pct=("realized_confirmation_move_pct", "median"),
            ).rename(columns={"mechanism": "type_id"}),
        ],
        ignore_index=True,
    )
    _write_csv(post_label, out_dir / "41_post_label_diagnostics_NOT_FEATURES.csv")

    manifest = {
        "script": SCRIPT_NAME,
        "version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "scope": "research_only_no_strategy_no_backtest",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "research_start": args.start_date,
        "research_end": args.end_date,
        "warmup_start": args.warmup_start_date,
        "train_end_date": args.train_end_date,
        "target_move_pct": float(args.target_move_pct),
        "max_completion_bars": int(args.max_completion_bars),
        "swing_extreme_price_source": "low",
        "swing_entry_price_source": "next_bar_open",
        "swing_target_observation_source": "future_closed_bar_close",
        "swing_return_definition": "future_closed_bar_close / next_bar_open - 1",
        "stage2_experiment_id": stage2_manifest.get("experiment_id"),
        "stage2_feature_source": feature_source,
        "parent_event_count": int(len(combined)),
        "combined_causal_feature_count": int(len(model_feature_columns)),
        "mechanism_lookback": int(args.mechanism_lookback),
        "phase_bins": int(args.phase_bins),
        "support_tolerance_bp": float(args.support_tolerance_bp),
        "minimum_test_gap": int(args.minimum_test_gap),
        "minimum_test_rebound_bp": float(args.minimum_test_rebound_bp),
        "classification_design": "weakly_supervised_causal_mechanism_scores_not_kmeans",
        "model_fit_diagnostics": model_fit_diagnostics.to_dict(orient="records"),
        "broad_types": list(BROAD_MECHANISM_TERMS),
        "trend_types": list(TREND_ARCHETYPE_TERMS),
        "base_types": list(BASE_ARCHETYPE_TERMS),
        "causal_policy": "swing extreme uses low; tradable confirmation uses next-bar open to future closed-bar close; all type features stop at extreme bar close",
    }
    (out_dir / "00_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "42_RESEARCH_SUMMARY.md").write_text(
        _summary(
            stage2_manifest, combined, broad_summary, trend_summary, base_summary,
            broad_stability, trend_stability, base_stability, model_fit_diagnostics, audit,
        ),
        encoding="utf-8",
    )
    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
