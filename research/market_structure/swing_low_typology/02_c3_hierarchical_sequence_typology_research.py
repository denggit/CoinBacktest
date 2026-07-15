#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Second-stage causal decomposition of the broad C3 swing-low family.

This remains pure descriptive research.  The historical swing-low universe is
retrospectively labelled, while every second-stage feature stops at the C3
extreme bar close.  No strategy, entry, exit, PnL, or parameter optimization is
performed.
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
    METADATA_COLUMNS,
    build_c3_sequence_features,
    build_causal_audit,
    build_sequence_profiles,
)
from research.market_structure.swing_low_typology.common.clustering import (  # noqa: E402
    build_assignments as build_stage1_assignments,
    fit_frozen_typology,
)
from research.market_structure.swing_low_typology.common.hierarchical_clustering import (  # noqa: E402
    build_assignments,
    build_post_label_diagnostics,
    build_profiles,
    build_rule_cards,
    build_stability,
    fit_family_balanced_typology,
    representative_events,
)
from research.market_structure.swing_low_typology.common.swing_low_dataset import (  # noqa: E402
    build_causal_features as build_stage1_features,
    detect_swing_lows,
    validate_trade_bar_fields,
)

SCRIPT_NAME = "02_c3_hierarchical_sequence_typology_research"
SCRIPT_VERSION = "1.1.0"
EXPERIMENT_ID = "ETH_1M_SWING_LOW_C3_HIERARCHICAL_TYPOLOGY_02"
EDGE_ID = "RESEARCH_ONLY_ETH_SWING_LOW_C3_SUBTYPES"
TITLE = "ETH Swing Low C3 Hierarchical Sequence Typology 02"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/02_c3_hierarchical_typology"
DEFAULT_STAGE1_DIR = "data/reports/research/market_structure/swing_low_typology/01_causal_typology"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Second-stage causal clustering inside the broad C3 swing-low family.",
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
    p.add_argument("--parent-cluster", default="C3")
    p.add_argument("--stage1-report-dir", default=DEFAULT_STAGE1_DIR)
    p.add_argument("--rebuild-stage1", action="store_true")
    p.add_argument("--stage1-feature-windows", default="5,15,30,60,120")
    p.add_argument("--feature-windows", default="15,30,60,120,240")
    p.add_argument("--phase-lookback", type=int, default=240)
    p.add_argument("--phase-bins", type=int, default=12)
    p.add_argument("--k-min", type=int, default=4)
    p.add_argument("--k-max", type=int, default=10)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--representative-events-per-cluster", type=int, default=15)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")
    return p.parse_args(argv)


def _parse_windows(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x.strip()) for x in str(raw).split(",") if x.strip()}))
    if not values:
        raise ValueError("window list cannot be empty")
    return values


def _end_exclusive(value: str, timeframe: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if len(str(value).strip()) <= 10 and ts == ts.normalize():
        return ts + pd.Timedelta(days=1)
    unit = str(timeframe).strip()
    if unit.endswith("m"):
        return ts + pd.Timedelta(minutes=int(unit[:-1]))
    if unit.endswith("s"):
        return ts + pd.Timedelta(seconds=int(unit[:-1]))
    if unit.endswith("h") or unit.endswith("H"):
        return ts + pd.Timedelta(hours=int(unit[:-1]))
    return ts + pd.Timedelta(minutes=1)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


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


def _validate_stage1_manifest(args: argparse.Namespace, report_dir: Path) -> None:
    manifest_path = report_dir / "00_manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    checks = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "target_move_pct": float(args.target_move_pct),
        "max_completion_bars": int(args.max_completion_bars),
    }
    for key, expected in checks.items():
        actual = manifest.get(key)
        if str(actual) != str(expected):
            mismatches.append(f"{key}: stage1={actual}, requested={expected}")
    label_policy = {
        "swing_extreme_price_source": "low",
        "swing_entry_price_source": "next_bar_open",
        "swing_target_observation_source": "future_closed_bar_close",
    }
    for key, expected in label_policy.items():
        actual = manifest.get(key)
        if actual != expected:
            mismatches.append(f"{key}: stage1={actual}, required={expected}")
    if mismatches:
        raise RuntimeError("Stage-1 report is incompatible: " + "; ".join(mismatches))


def _load_or_rebuild_stage1(args: argparse.Namespace, bars: pd.DataFrame) -> pd.DataFrame:
    report_dir = Path(args.stage1_report_dir)
    assignment_path = report_dir / "06_frozen_cluster_assignments.csv"
    if assignment_path.exists() and not args.rebuild_stage1:
        _validate_stage1_manifest(args, report_dir)
        print(f"[stage1] loading frozen assignments: {assignment_path}", flush=True)
        assignments = pd.read_csv(
            assignment_path,
            parse_dates=[
                "extreme_time",
                "feature_available_time",
                "confirmation_time",
                "confirmation_available_time",
            ],
        )
    else:
        print("[stage1] report missing/rebuild requested; reconstructing train-only parent typology", flush=True)
        stage1_windows = _parse_windows(args.stage1_feature_windows)
        events = detect_swing_lows(
            bars,
            target_move_pct=float(args.target_move_pct),
            max_completion_bars=int(args.max_completion_bars),
            research_start=pd.Timestamp(args.start_date),
            research_end_exclusive=_end_exclusive(args.end_date, args.timeframe),
            minimum_history_bars=max(stage1_windows),
        )
        stage1_features, _ = build_stage1_features(
            bars,
            events,
            windows=stage1_windows,
            progress_every=int(args.progress_every),
        )
        metadata = {
            "event_id",
            "extreme_time",
            "feature_available_time",
            "extreme_pos",
            "extreme_price",
            "confirmation_time",
            "confirmation_available_time",
            "completion_bars",
            "realized_confirmation_move_pct",
        }
        stage1_columns = [c for c in stage1_features.columns if c not in metadata]
        train_end = pd.Timestamp(args.train_end_date) + pd.Timedelta(days=1)
        train_mask = pd.to_datetime(stage1_features["extreme_time"]) < train_end
        frozen, _ = fit_frozen_typology(
            stage1_features,
            stage1_columns,
            train_mask,
            k_min=3,
            k_max=7,
            random_state=int(args.random_state),
        )
        assignments = build_stage1_assignments(frozen, stage1_features, train_mask)

    parent = assignments[assignments["cluster_id"].astype(str) == str(args.parent_cluster)].copy()
    if len(parent) < 500:
        raise RuntimeError(f"Too few {args.parent_cluster} events for second-stage clustering: {len(parent)}")
    parent = parent.sort_values("extreme_time").reset_index(drop=True)
    print(f"       parent_cluster={args.parent_cluster} events={len(parent):,}", flush=True)
    return parent


def _plot_sequence_profiles(profiles: pd.DataFrame, out_dir: Path, split: str) -> None:
    subset = profiles[(profiles["split"] == split) & profiles["metric"].isin(["price", "cvd", "activity"])]
    if subset.empty:
        return
    for metric, group_metric in subset.groupby("metric"):
        fig, ax = plt.subplots(figsize=(10, 6))
        for cluster_id, group in group_metric.groupby("subcluster_id", sort=True):
            ax.plot(group["phase"], group["median"], marker="o", label=str(cluster_id))
        ax.set_title(f"C3 pre-low {metric} phase profile ({split})")
        ax.set_xlabel("Historical phase (oldest to extreme)")
        ax.set_ylabel("Median normalized value")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"17_{metric}_phase_profile_{split}.png", dpi=150)
        plt.close(fig)


def _plot_yearly_share(yearly: pd.DataFrame, out_dir: Path) -> None:
    if yearly.empty:
        return
    pivot = yearly.pivot(index="year", columns="subcluster_id", values="share_within_year").fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster_id in pivot.columns:
        ax.plot(pivot.index, pivot[cluster_id] * 100.0, marker="o", label=str(cluster_id))
    ax.set_title("Frozen C3 subtype share by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share within C3 (%)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "18_c3_subtype_share_by_year.png", dpi=150)
    plt.close(fig)


def _build_summary(
    args: argparse.Namespace,
    parent: pd.DataFrame,
    selected_k: int,
    selection: pd.DataFrame,
    family_summary: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    stability: pd.DataFrame,
    rule_fidelity: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    selected = selection[selection["selected"]].iloc[0]
    lines = [
        f"# {TITLE}",
        "",
        "## Scope",
        "",
        "- Pure descriptive second-stage research inside parent cluster C3; no strategy or PnL backtest.",
        "- Future prices only define the historical swing-low universe and confirmation metadata.",
        "- All subtype features end at the extreme bar close or earlier.",
        "- 2023-2024 fits every cleaner, family scaler, PCA block, K choice, and centroid; holdout only receives frozen assignments.",
        "",
        "## Why this differs from a naive second KMeans",
        "",
        "- Rich 240-bar price structure, cumulative CVD, large-flow, volume/activity, and price-response paths.",
        "- Twelve ordered historical phase bins preserve sequence shape instead of reducing each event to one point.",
        "- Each feature family has an independent robust scaler/PCA block and equalized block weight.",
        "- Cluster count is selected using train silhouette, seed ARI, and minimum-share penalties only.",
        "",
        "## Model selection",
        "",
        f"- Parent C3 events: {len(parent):,}",
        f"- Selected C3 subtypes: {selected_k}",
        f"- Train silhouette: {float(selected['silhouette_train']):.4f}",
        f"- Seed stability ARI: {float(selected['seed_stability_ari']):.4f}",
        f"- Minimum train subtype share: {float(selected['minimum_cluster_share_train']):.2%}",
        f"- Usable feature families: {len(family_summary)}",
        "",
        "## Subtype descriptions",
        "",
    ]
    holdout_share = stability[stability["split"] == "holdout"].set_index("subcluster_id")["share_within_split"]
    for row in cluster_summary.itertuples(index=False):
        share = float(holdout_share.get(row.subcluster_id, np.nan))
        lines.append(
            f"- **{row.subcluster_id}**: train {int(row.train_count):,} ({float(row.train_share):.1%}), "
            f"holdout {share:.1%}; {row.descriptor}"
        )
    lines.extend(["", "## Rule-card fidelity", ""])
    if rule_fidelity.empty:
        lines.append("- No rule-card output.")
    else:
        for row in rule_fidelity[rule_fidelity["split"] == "holdout"].itertuples(index=False):
            lines.append(
                f"- {row.subcluster_id}: holdout F1 {float(row.f1):.3f}, "
                f"balanced accuracy {float(row.balanced_accuracy):.3f}."
            )
    lines.extend(["", "## Causal audit", ""])
    for row in audit.itertuples(index=False):
        lines.append(f"- {'PASS' if bool(row.passed) else 'FAIL'} `{row.check}`: {row.detail}")
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "- The subtype cannot be known at the extreme timestamp because the swing-low label itself is retrospective.",
            "- Completion speed is reported separately and never used to form or select subtypes.",
            "- A visually coherent subtype is not automatically a predictive edge.",
            "",
        ]
    )
    return "\n".join(lines)


def run_research(args: argparse.Namespace) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_windows = _parse_windows(args.feature_windows)
    bars = load_bars(args)

    print("[validate] checking rich trade-bar fields", flush=True)
    coverage = validate_trade_bar_fields(bars)
    _write_csv(coverage, out_dir / "01_trade_bar_field_coverage.csv")

    parent = _load_or_rebuild_stage1(args, bars)
    _write_csv(parent, out_dir / "02_parent_c3_events.csv")

    print("[features] building structural, CVD, activity, response, and ordered-phase features", flush=True)
    features, dictionary = build_c3_sequence_features(
        bars,
        parent,
        windows=feature_windows,
        phase_lookback=int(args.phase_lookback),
        phase_bins=int(args.phase_bins),
        progress_every=int(args.progress_every),
    )
    feature_columns = [c for c in features.columns if c not in METADATA_COLUMNS]
    _write_csv(dictionary, out_dir / "03_c3_feature_dictionary.csv")
    _write_csv(features.head(5000), out_dir / "04_c3_feature_matrix_sample.csv")

    train_end = pd.Timestamp(args.train_end_date) + pd.Timedelta(days=1)
    train_mask = pd.to_datetime(features["extreme_time"]) < train_end
    if int(train_mask.sum()) < 500 or int((~train_mask).sum()) < 200:
        raise RuntimeError(
            f"Insufficient split: train={int(train_mask.sum())}, holdout={int((~train_mask).sum())}"
        )
    print(
        f"[cluster] family-balanced train={int(train_mask.sum()):,} holdout={int((~train_mask).sum()):,} "
        f"k={args.k_min}..{args.k_max}",
        flush=True,
    )
    frozen, selection, family_summary = fit_family_balanced_typology(
        features,
        dictionary,
        train_mask,
        k_min=int(args.k_min),
        k_max=int(args.k_max),
        random_state=int(args.random_state),
    )
    _write_csv(family_summary, out_dir / "05_family_embedding_summary.csv")
    _write_csv(selection, out_dir / "06_subcluster_selection_train_only.csv")

    assignments = build_assignments(frozen, features, train_mask)
    _write_csv(assignments, out_dir / "07_frozen_c3_subcluster_assignments.csv")
    cluster_summary, descriptors, profiles = build_profiles(features, dictionary, assignments, train_mask)
    stability, yearly = build_stability(assignments)
    _write_csv(cluster_summary, out_dir / "08_c3_subcluster_summary.csv")
    _write_csv(stability, out_dir / "09_c3_subcluster_train_holdout_stability.csv")
    _write_csv(yearly, out_dir / "10_c3_subcluster_yearly_stability.csv")
    _write_csv(profiles, out_dir / "11_c3_subcluster_feature_profiles.csv")
    _write_csv(descriptors, out_dir / "12_c3_subcluster_top_descriptors.csv")

    print("[explain] fitting shallow rules to describe frozen subtypes", flush=True)
    rule_text, rule_fidelity, rule_importance = build_rule_cards(
        features,
        dictionary,
        assignments,
        train_mask,
        random_state=int(args.random_state),
    )
    (out_dir / "13_c3_subcluster_rule_cards.txt").write_text(rule_text, encoding="utf-8")
    _write_csv(rule_fidelity, out_dir / "14_c3_subcluster_rule_fidelity.csv")
    _write_csv(rule_importance, out_dir / "15_c3_subcluster_rule_feature_importance.csv")
    _write_csv(
        representative_events(assignments, per_cluster=int(args.representative_events_per_cluster)),
        out_dir / "16_representative_c3_swing_lows.csv",
    )

    sequence_profiles = build_sequence_profiles(features, assignments)
    _write_csv(sequence_profiles, out_dir / "17_c3_sequence_phase_profiles.csv")
    _plot_sequence_profiles(sequence_profiles, out_dir, "train")
    _plot_sequence_profiles(sequence_profiles, out_dir, "holdout")
    _plot_yearly_share(yearly, out_dir)

    post_diag = build_post_label_diagnostics(assignments)
    _write_csv(post_diag, out_dir / "19_post_label_confirmation_diagnostics_NOT_FEATURES.csv")
    audit = build_causal_audit(features, feature_columns)
    audit = pd.concat(
        [
            audit,
            pd.DataFrame(
                [
                    {
                        "check": "train_only_model_fit",
                        "passed": True,
                        "detail": f"train through {args.train_end_date}; holdout receives frozen transforms and centroids",
                    },
                    {
                        "check": "post_label_metrics_excluded_from_model_selection",
                        "passed": True,
                        "detail": "completion bars and realized confirmation move appear only in metadata/diagnostics",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    _write_csv(audit, out_dir / "20_causal_audit.csv")
    if not bool(audit["passed"].all()):
        raise RuntimeError("Causal audit failed; inspect 20_causal_audit.csv")

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
        "train_end_date": args.train_end_date,
        "target_move_pct": float(args.target_move_pct),
        "max_completion_bars": int(args.max_completion_bars),
        "swing_extreme_price_source": "low",
        "swing_entry_price_source": "next_bar_open",
        "swing_target_observation_source": "future_closed_bar_close",
        "swing_return_definition": "future_closed_bar_close / next_bar_open - 1",
        "parent_cluster": args.parent_cluster,
        "parent_event_count": int(len(parent)),
        "second_stage_feature_count": int(len(feature_columns)),
        "feature_windows": list(feature_windows),
        "phase_lookback": int(args.phase_lookback),
        "phase_bins": int(args.phase_bins),
        "selected_k": int(frozen.selected_k),
        "feature_families": family_summary["family"].tolist(),
        "causal_policy": "swing extreme uses low; tradable confirmation uses next-bar open to future closed-bar close; subtype features end at extreme bar close",
    }
    (out_dir / "00_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = _build_summary(
        args,
        parent,
        frozen.selected_k,
        selection,
        family_summary,
        cluster_summary,
        stability,
        rule_fidelity,
        audit,
    )
    (out_dir / "21_RESEARCH_SUMMARY.md").write_text(summary, encoding="utf-8")

    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(f"[done] report={out_dir}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
