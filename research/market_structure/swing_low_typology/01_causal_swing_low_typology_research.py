#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Research-only causal typology of retrospectively confirmed ETH swing lows.

This is not a strategy or a backtest. It answers a descriptive question:

    What recurring pre-low market structures and order-flow paths exist among
    ETH swing lows that later complete a configured rebound?

Future data is used only to label the historical swing-low universe. Cluster
features stop at the extreme bar close. Preprocessing, PCA, K selection,
centroids, and rule cards are fitted on 2023-2024 by default and frozen before
2025-2026H1 assignment. Future rebound speed is reported separately and never
participates in clustering or model selection.
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
from research.market_structure.swing_low_typology.common.clustering import (  # noqa: E402
    build_assignments,
    build_cluster_profiles,
    build_cluster_stability,
    build_post_label_diagnostics,
    build_rule_cards,
    fit_frozen_typology,
    representative_events,
)
from research.market_structure.swing_low_typology.common.swing_low_dataset import (  # noqa: E402
    build_causal_audit,
    build_causal_features,
    build_pre_low_path_profiles,
    detect_swing_lows,
    validate_trade_bar_fields,
)

SCRIPT_NAME = "01_causal_swing_low_typology_research"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_1M_SWING_LOW_CAUSAL_TYPOLOGY_01"
EDGE_ID = "RESEARCH_ONLY_ETH_SWING_LOW_TYPOLOGY"
TITLE = "ETH Swing Low Causal Typology Research 01"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/01_causal_typology"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unsupervised causal clustering of retrospectively confirmed ETH swing lows.",
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
    p.add_argument("--feature-windows", default="5,15,30,60,120")
    p.add_argument("--k-min", type=int, default=3)
    p.add_argument("--k-max", type=int, default=7)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--path-profile-lookback", type=int, default=120)
    p.add_argument("--path-profile-sample-per-cluster", type=int, default=800)
    p.add_argument("--representative-events-per-cluster", type=int, default=12)
    return p.parse_args(argv)


def _parse_windows(raw: str) -> tuple[int, ...]:
    values = tuple(sorted({int(x.strip()) for x in str(raw).split(",") if x.strip()}))
    if not values or min(values) < 2:
        raise ValueError("feature-windows must contain integers >= 2")
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
    if unit.endswith("H") or unit.endswith("h"):
        return ts + pd.Timedelta(hours=int(unit[:-1]))
    return ts + pd.Timedelta(minutes=1)


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


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _plot_pre_low_paths(profiles: pd.DataFrame, out_dir: Path, split: str) -> None:
    subset = profiles[profiles["split"] == split]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster_id, group in subset.groupby("cluster_id", sort=True):
        ax.plot(group["offset_bars"], group["median_close_vs_extreme"] * 100.0, label=str(cluster_id))
    ax.axvline(0, linewidth=1)
    ax.set_title(f"Pre-low median price paths ({split})")
    ax.set_xlabel("Bars before labelled extreme")
    ax.set_ylabel("Median close vs extreme low (%)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / f"17_pre_low_price_paths_{split}.png", dpi=150)
    plt.close(fig)


def _plot_cluster_shares(yearly: pd.DataFrame, out_dir: Path) -> None:
    if yearly.empty:
        return
    pivot = yearly.pivot(index="year", columns="cluster_id", values="share_within_year").fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 6))
    for cluster_id in pivot.columns:
        ax.plot(pivot.index, pivot[cluster_id] * 100.0, marker="o", label=str(cluster_id))
    ax.set_title("Frozen cluster share by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of swing lows (%)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "18_cluster_share_by_year.png", dpi=150)
    plt.close(fig)


def _build_summary(
    args: argparse.Namespace,
    events: pd.DataFrame,
    frozen,
    selection: pd.DataFrame,
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
        "- Pure descriptive research; no entries, exits, PnL, or strategy backtest.",
        f"- Swing-low label: at least {args.target_move_pct:g}% rebound completed within {args.max_completion_bars} bars.",
        "- Future prices are used only for the historical label and confirmation timestamp.",
        "- Every clustering feature uses the extreme bar close or older trade bars only; left-labelled bars become available after close.",
        "- Preprocessing, PCA, K selection, centroids, and rule cards are fit on the development period only.",
        "",
        "## Model selection",
        "",
        f"- Events: {len(events):,}",
        f"- Selected clusters: {frozen.selected_k}",
        f"- Causal features retained after train-only cleaning: {len(frozen.feature_columns)}",
        f"- PCA components: {frozen.pca.n_components_}",
        f"- Train silhouette: {float(selected['silhouette_train']):.4f}",
        f"- Seed stability ARI: {float(selected['seed_stability_ari']):.4f}",
        f"- Minimum train cluster share: {float(selected['minimum_cluster_share_train']):.2%}",
        "",
        "## Cluster descriptions",
        "",
    ]
    for row in cluster_summary.itertuples(index=False):
        holdout_row = stability[(stability["cluster_id"] == row.cluster_id) & (stability["split"] == "holdout")]
        holdout_share = float(holdout_row["share_within_split"].iloc[0]) if not holdout_row.empty else np.nan
        lines.append(
            f"- **{row.cluster_id}**: train {int(row.train_count):,} ({float(row.train_share):.1%}), "
            f"holdout share {holdout_share:.1%}; {row.descriptor}"
        )
    lines.extend(["", "## Interpretability fidelity", ""])
    if rule_fidelity.empty:
        lines.append("- No rule-card fidelity output.")
    else:
        for row in rule_fidelity[rule_fidelity["split"] == "holdout"].itertuples(index=False):
            lines.append(
                f"- {row.cluster_id}: holdout shallow-rule F1 {float(row.f1):.3f}, "
                f"balanced accuracy {float(row.balanced_accuracy):.3f}."
            )
    lines.extend(
        [
            "",
            "## Causal audit",
            "",
            *[
                f"- {'PASS' if bool(r.passed) else 'FAIL'} `{r.check}`: {r.detail}"
                for r in audit.itertuples(index=False)
            ],
            "",
            "## Interpretation limits",
            "",
            "- A swing low is not knowable at the extreme timestamp; it is confirmed later and retrospectively labelled.",
            "- Cluster IDs describe recurring historical pre-low states, not profitable trades or real-time signals.",
            "- Future confirmation speed is included only in a separate diagnostics file and did not affect clustering.",
            "- A useful next step is visual/manual validation of representative events before any signal-prediction research.",
            "",
        ]
    )
    return "\n".join(lines)


def run_research(args: argparse.Namespace) -> Path:
    windows = _parse_windows(args.feature_windows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = load_bars(args)

    print("[validate] checking rich trade-bar field coverage", flush=True)
    coverage = validate_trade_bar_fields(bars)
    _write_csv(coverage, out_dir / "01_trade_bar_field_coverage.csv")

    research_start = pd.Timestamp(args.start_date)
    research_end_exclusive = _end_exclusive(args.end_date, args.timeframe)
    print("[labels] detecting retrospective percentage-confirmed swing lows", flush=True)
    events = detect_swing_lows(
        bars,
        target_move_pct=float(args.target_move_pct),
        max_completion_bars=int(args.max_completion_bars),
        research_start=research_start,
        research_end_exclusive=research_end_exclusive,
        minimum_history_bars=max(windows),
    )
    if len(events) < 300:
        raise RuntimeError(f"Too few swing-low events for clustering: {len(events)}")
    _write_csv(events, out_dir / "02_swing_low_events.csv")
    print(f"       swing_lows={len(events):,}", flush=True)

    print("[features] building current-bar and multi-window causal features", flush=True)
    features, feature_dictionary = build_causal_features(
        bars,
        events,
        windows=windows,
        progress_every=int(args.progress_every),
    )
    metadata_columns = {
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
    feature_columns = [c for c in features.columns if c not in metadata_columns]
    _write_csv(feature_dictionary, out_dir / "03_causal_feature_dictionary.csv")
    _write_csv(features.head(5000), out_dir / "04_causal_feature_matrix_sample.csv")

    train_end_exclusive = pd.Timestamp(args.train_end_date) + pd.Timedelta(days=1)
    train_mask = pd.to_datetime(features["extreme_time"]) < train_end_exclusive
    if int((~train_mask).sum()) < 100:
        raise RuntimeError(f"Too few holdout events: {int((~train_mask).sum())}")
    print(
        f"[cluster] train={int(train_mask.sum()):,} holdout={int((~train_mask).sum()):,} "
        f"k={args.k_min}..{args.k_max}",
        flush=True,
    )
    frozen, selection = fit_frozen_typology(
        features,
        feature_columns,
        train_mask,
        k_min=int(args.k_min),
        k_max=int(args.k_max),
        random_state=int(args.random_state),
    )
    _write_csv(selection, out_dir / "05_cluster_selection_train_only.csv")

    assignments = build_assignments(frozen, features, train_mask)
    _write_csv(assignments, out_dir / "06_frozen_cluster_assignments.csv")
    cluster_summary, descriptors, profiles = build_cluster_profiles(
        frozen,
        features,
        feature_dictionary,
        train_mask,
    )
    stability, yearly = build_cluster_stability(assignments)
    cluster_summary = cluster_summary.merge(
        stability.pivot(index="cluster_id", columns="split", values="share_within_split")
        .rename(columns={"train": "train_share_check", "holdout": "holdout_share"})
        .reset_index(),
        on="cluster_id",
        how="left",
    )
    _write_csv(cluster_summary, out_dir / "07_cluster_summary.csv")
    _write_csv(stability, out_dir / "08_cluster_train_holdout_stability.csv")
    _write_csv(yearly, out_dir / "09_cluster_yearly_stability.csv")
    _write_csv(profiles, out_dir / "10_cluster_feature_profiles.csv")
    _write_csv(descriptors, out_dir / "11_cluster_top_descriptors.csv")

    print("[explain] fitting shallow train-only rule cards for cluster descriptions", flush=True)
    rule_text, rule_fidelity, rule_importance = build_rule_cards(
        frozen,
        features,
        train_mask,
        random_state=int(args.random_state),
    )
    (out_dir / "12_cluster_rule_cards.txt").write_text(rule_text, encoding="utf-8")
    _write_csv(rule_fidelity, out_dir / "13_cluster_rule_fidelity.csv")
    _write_csv(rule_importance, out_dir / "14_cluster_rule_feature_importance.csv")
    _write_csv(
        representative_events(assignments, per_cluster=int(args.representative_events_per_cluster)),
        out_dir / "15_representative_swing_lows.csv",
    )

    print("[paths] building pre-low-only normalized path profiles", flush=True)
    path_profiles = build_pre_low_path_profiles(
        bars,
        assignments,
        lookback_bars=int(args.path_profile_lookback),
        max_samples_per_cluster_split=int(args.path_profile_sample_per_cluster),
        random_state=int(args.random_state),
    )
    _write_csv(path_profiles, out_dir / "16_pre_low_path_profiles.csv")
    _plot_pre_low_paths(path_profiles, out_dir, "train")
    _plot_pre_low_paths(path_profiles, out_dir, "holdout")
    _plot_cluster_shares(yearly, out_dir)

    post_diag = build_post_label_diagnostics(assignments)
    _write_csv(post_diag, out_dir / "19_post_label_confirmation_diagnostics_NOT_FEATURES.csv")
    audit = build_causal_audit(features, frozen.feature_columns)
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
        "feature_windows": list(windows),
        "event_count": int(len(events)),
        "train_count": int(train_mask.sum()),
        "holdout_count": int((~train_mask).sum()),
        "selected_k": int(frozen.selected_k),
        "selected_features": list(frozen.feature_columns),
        "pca_components": int(frozen.pca.n_components_),
        "causal_policy": "features use extreme bar or older; future only labels the swing-low universe",
    }
    (out_dir / "00_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = _build_summary(args, events, frozen, selection, cluster_summary, stability, rule_fidelity, audit)
    (out_dir / "21_RESEARCH_SUMMARY.md").write_text(summary, encoding="utf-8")

    finalize_research_report(
        out_dir,
        experiment_id=EXPERIMENT_ID,
        edge_id=EDGE_ID,
        title=TITLE,
    )
    print(f"[done] report={out_dir}", flush=True)
    return out_dir


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
