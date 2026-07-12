#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""03 pre-orange environment contrast research for panic recovery.

Goal
----
The orange node is *not* an entry. It is an early observation gate. This pass
asks whether information already visible before orange can improve the quality
of the later causal green entry.

Research flow
-------------
1. Detect the same causal multi-bar panic episodes as 01/02.
2. Build rich trade-bar features.
3. For every orange node, sample two explicitly separated feature groups:
   - ``pre_*``: ends at orange_time - 1 bar; strict pre-orange environment;
   - ``orange_*``: the closed orange bar itself; secondary diagnostic only.
4. Join those orange features to the later green signal by ``episode_id``.
5. Label green trades with future outcomes only for winner/loser analysis.
6. Learn tail thresholds on 2023-2024 only and apply unchanged to
   2025-2026H1 holdout.
7. Re-test train-selected orange gates on the actual green entry using the
   known episode low as a structural stop.

Causal boundary
---------------
- No future outcome, final-low distance, or post-orange episode aggregate is a
  candidate feature.
- ``pre_*`` rolling features use ``shift(1)`` before every rolling operation.
- ``orange_*`` uses the orange closed bar and is never available before close.
- Green entries remain next-bar open.
- Structural stop uses ``episode_low`` only after green confirmation; that low
  has already printed and is known at the green signal time.
- Train thresholds and candidate selection never inspect holdout performance.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_common.review_pack import finalize_research_report  # noqa: E402
from src.research_common.trade_bar_orderflow import (  # noqa: E402
    attach_orderflow_to_stage_events,
    build_trade_bar_orderflow_features,
    summarize_episode_orderflow,
    validate_trade_bar_orderflow,
)


SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}


def _load_sibling(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared research helpers: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


V1 = _load_sibling("01_environment_and_cluster_scale_in_research.py", "panic_recovery_01_shared_for_03")
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="03 pre-orange winner/loser contrast and green-quality gate research",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", choices=sorted(SUPPORTED_TIMEFRAMES), default="1m")
    p.add_argument("--data-source", choices=["trade_bar"], default="trade_bar")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--train-end-date", default="2024-12-31 23:59:59")
    p.add_argument(
        "--out-dir",
        default="data/reports/research/liquidity/panic_selloff_rejection_recovery_long/03_pre_orange_winner_loser_contrast",
    )

    # Keep detector baseline unchanged so 03 isolates the orange environment.
    p.add_argument("--baseline-window", type=int, default=60)
    p.add_argument("--selloff-window", type=int, default=5)
    p.add_argument("--min-red-bars", type=int, default=3)
    p.add_argument("--observe-drop-pct", type=float, default=0.0045)
    p.add_argument("--observe-drop-vol-mult", type=float, default=2.5)
    p.add_argument("--observe-volume-ratio", type=float, default=1.10)
    p.add_argument("--panic-drop-pct", type=float, default=0.0075)
    p.add_argument("--panic-volume-ratio", type=float, default=1.35)
    p.add_argument("--stabilization-bars", type=int, default=2)
    p.add_argument("--min-rebound-from-low-pct", type=float, default=0.0020)
    p.add_argument("--pressure-decay-ratio", type=float, default=0.68)
    p.add_argument("--reclaim-fraction", type=float, default=0.35)
    p.add_argument("--breakout-lookback", type=int, default=2)
    p.add_argument("--max-episode-bars", type=int, default=30)
    p.add_argument("--cooldown-bars", type=int, default=8)

    p.add_argument("--orderflow-baseline-window", type=int, default=240)
    p.add_argument("--pre-windows", default="5,15,30,60,120,240")
    p.add_argument("--horizons", default="5,15,30,60,120,240")
    p.add_argument("--candidate-horizon", type=int, default=60)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--winner-threshold", type=float, default=0.0)
    p.add_argument("--strong-winner-threshold", type=float, default=0.0025)
    p.add_argument("--strong-loser-threshold", type=float, default=-0.0015)

    # Train-only tail mining. Boundaries are then frozen for holdout.
    p.add_argument("--tail-quantiles", default="0.20,0.30")
    p.add_argument("--min-filter-train", type=int, default=80)
    p.add_argument("--min-filter-holdout", type=int, default=35)
    p.add_argument("--top-atomic-for-pairs", type=int, default=10)
    p.add_argument("--top-train-candidates", type=int, default=8)
    p.add_argument("--max-score-features", type=int, default=6)

    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.00020)
    p.add_argument("--exit-slippage-pct", type=float, default=0.00020)
    p.add_argument("--cost-multipliers", default="1.0,2.0")
    p.add_argument("--cluster-gap-bars", default="30")
    p.add_argument("--stop-buffer-pct", type=float, default=0.0005)
    p.add_argument("--target-r-list", default="0.75,1.0,1.5")
    p.add_argument("--save-trade-sample", type=int, default=30000)
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _parse_list(text: str, *, cast: Callable[[str], Any], name: str) -> list[Any]:
    values: list[Any] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        value = cast(token)
        if float(value) <= 0:
            raise ValueError(f"{name} must contain positive values")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return sorted(set(values))


from research.liquidity.panic_selloff_rejection_recovery_long.common.pre_orange_environment import (  # noqa: E402
    build_pre_orange_features,
    feature_id,
    join_orange_features_to_green,
    numeric_series,
    winner_loser_contrast,
)


def _selection_score(row: dict[str, Any] | pd.Series) -> float:
    count = float(row.get("train_count", 0) or 0)
    mean = float(row.get("train_mean_net", np.nan))
    pf = float(row.get("train_profit_factor", np.nan))
    if count <= 0 or not np.isfinite(mean) or not np.isfinite(pf):
        return np.nan
    return float(mean * math.sqrt(count) + 0.0005 * np.clip(pf - 1.0, -1.0, 2.0) * math.sqrt(count))


def _holdout_pass(row: dict[str, Any] | pd.Series, args: argparse.Namespace) -> bool:
    return bool(
        float(row.get("train_count", 0) or 0) >= int(args.min_filter_train)
        and float(row.get("holdout_count", 0) or 0) >= int(args.min_filter_holdout)
        and float(row.get("train_mean_net", -np.inf)) > 0
        and float(row.get("holdout_mean_net", -np.inf)) > 0
        and float(row.get("train_profit_factor", 0)) > 1.0
        and float(row.get("holdout_profit_factor", 0)) > 1.0
    )


def _candidate_mask(values: pd.Series, direction: str, threshold: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if direction == "le":
        return numeric <= threshold
    if direction == "ge":
        return numeric >= threshold
    raise ValueError(direction)


def evaluate_train_tail_filters(
    signals: pd.DataFrame,
    feature_meta: pd.DataFrame,
    args: argparse.Namespace,
    *,
    scope: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(args.train_end_date)
    train = signals[pd.to_datetime(signals["event_time"]) <= train_end]
    return_col = f"ret_h{int(args.candidate_horizon)}_net"
    quantiles = _parse_list(args.tail_quantiles, cast=float, name="tail_quantiles")
    if any(q >= 0.5 for q in quantiles):
        raise ValueError("tail_quantiles must be below 0.5")
    meta = feature_meta[feature_meta["scope"] == scope].copy()
    meta_map = meta.set_index("feature").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    masks: dict[str, pd.Series] = {}

    for feature in meta["feature"].tolist():
        train_values = numeric_series(train, feature).dropna()
        if len(train_values) < int(args.min_filter_train) or train_values.nunique() < 12:
            continue
        for q in quantiles:
            for direction, quantile in (("le", q), ("ge", 1.0 - q)):
                threshold = float(train_values.quantile(quantile))
                mask = _candidate_mask(numeric_series(signals, feature), direction, threshold).fillna(False)
                candidate_id = feature_id(f"{feature}__{direction}__q{quantile:.2f}")
                part = signals[mask]
                row = {
                    "candidate_id": candidate_id,
                    "feature": feature,
                    "family": meta_map[feature]["family"],
                    "scope": scope,
                    "description": meta_map[feature]["description"],
                    "direction": direction,
                    "train_quantile": quantile,
                    "threshold": threshold,
                    **V1._split_stats(part, return_col, train_end),
                }
                row["train_score"] = _selection_score(row)
                row["holdout_pass"] = _holdout_pass(row, args)
                rows.append(row)
                masks[candidate_id] = mask
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["holdout_pass", "train_score", "train_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    mask_frame = pd.DataFrame(masks, index=signals.index)
    return out, mask_frame


def evaluate_train_selected_pairs(
    signals: pd.DataFrame,
    atomic: pd.DataFrame,
    atomic_masks: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if atomic.empty:
        return pd.DataFrame(), pd.DataFrame(index=signals.index)
    eligible = atomic[
        (atomic["train_count"] >= int(args.min_filter_train))
        & np.isfinite(pd.to_numeric(atomic["train_score"], errors="coerce"))
    ].sort_values("train_score", ascending=False)
    # Deduplicate by feature before pairing so one variable cannot dominate with
    # several nearby quantile cuts.
    eligible = eligible.drop_duplicates("feature").head(int(args.top_atomic_for_pairs))
    return_col = f"ret_h{int(args.candidate_horizon)}_net"
    train_end = pd.Timestamp(args.train_end_date)
    rows: list[dict[str, Any]] = []
    masks: dict[str, pd.Series] = {}
    records = eligible.to_dict(orient="records")
    for left, right in itertools.combinations(records, 2):
        if left["family"] == right["family"]:
            continue
        left_id = str(left["candidate_id"])
        right_id = str(right["candidate_id"])
        mask = atomic_masks[left_id] & atomic_masks[right_id]
        candidate_id = feature_id(f"PAIR__{left_id}__{right_id}")
        part = signals[mask]
        row = {
            "candidate_id": candidate_id,
            "left_candidate": left_id,
            "right_candidate": right_id,
            "left_feature": left["feature"],
            "right_feature": right["feature"],
            "family": f"{left['family']}+{right['family']}",
            "scope": "pre_orange_pair",
            **V1._split_stats(part, return_col, train_end),
        }
        row["train_score"] = _selection_score(row)
        row["holdout_pass"] = _holdout_pass(row, args)
        rows.append(row)
        masks[candidate_id] = mask
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["holdout_pass", "train_score", "train_count"],
            ascending=[False, False, False],
        ).reset_index(drop=True)
    return out, pd.DataFrame(masks, index=signals.index)


def build_train_selected_score(
    signals: pd.DataFrame,
    atomic: pd.DataFrame,
    atomic_masks: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if atomic.empty:
        return pd.DataFrame(), pd.DataFrame(index=signals.index), []
    selected = atomic[
        (atomic["train_count"] >= int(args.min_filter_train))
        & (pd.to_numeric(atomic["train_mean_net"], errors="coerce") > 0)
        & (pd.to_numeric(atomic["train_profit_factor"], errors="coerce") > 1.0)
    ].sort_values("train_score", ascending=False)
    selected = selected.drop_duplicates("family").head(int(args.max_score_features))
    selected_ids = selected["candidate_id"].astype(str).tolist()
    if not selected_ids:
        return pd.DataFrame(), pd.DataFrame(index=signals.index), []
    score = atomic_masks[selected_ids].fillna(False).astype(int).sum(axis=1)
    return_col = f"ret_h{int(args.candidate_horizon)}_net"
    train_end = pd.Timestamp(args.train_end_date)
    rows: list[dict[str, Any]] = []
    masks: dict[str, pd.Series] = {}
    for threshold in range(1, min(4, len(selected_ids)) + 1):
        candidate_id = f"PRE_ORANGE_SCORE_GE_{threshold}"
        mask = score >= threshold
        row = {
            "candidate_id": candidate_id,
            "score_threshold": threshold,
            "selected_features": ";".join(selected_ids),
            "family": "train_selected_score",
            "scope": "pre_orange_score",
            **V1._split_stats(signals[mask], return_col, train_end),
        }
        row["train_score"] = _selection_score(row)
        row["holdout_pass"] = _holdout_pass(row, args)
        rows.append(row)
        masks[candidate_id] = mask
    return pd.DataFrame(rows).sort_values("train_score", ascending=False), pd.DataFrame(masks, index=signals.index), selected_ids


def build_structural_candidates(
    signals: pd.DataFrame,
    atomic: pd.DataFrame,
    atomic_masks: pd.DataFrame,
    pairs: pd.DataFrame,
    pair_masks: pd.DataFrame,
    scores: pd.DataFrame,
    score_masks: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [{"candidate_name": "ALL_GREEN", "source": "baseline", "filter_expression": "ALL", "train_score": np.nan}]
    combined_masks = pd.DataFrame(index=signals.index)

    selections: list[tuple[pd.DataFrame, pd.DataFrame, str]] = [
        (atomic, atomic_masks, "pre_orange_atomic"),
        (pairs, pair_masks, "pre_orange_pair"),
        (scores, score_masks, "pre_orange_score"),
    ]
    pool_rows: list[dict[str, Any]] = []
    for frame, masks, source in selections:
        if frame.empty:
            continue
        for row in frame.to_dict(orient="records"):
            cid = str(row["candidate_id"])
            if cid not in masks.columns:
                continue
            pool_rows.append({**row, "_source": source})
    if pool_rows:
        pool = pd.DataFrame(pool_rows).sort_values("train_score", ascending=False)
        pool = pool.drop_duplicates("candidate_id").head(int(args.top_train_candidates))
        for row in pool.to_dict(orient="records"):
            cid = str(row["candidate_id"])
            source = str(row["_source"])
            source_masks = atomic_masks if source == "pre_orange_atomic" else pair_masks if source == "pre_orange_pair" else score_masks
            col = f"filter__{cid}"
            signals[col] = source_masks[cid].fillna(False).astype(bool).to_numpy()
            combined_masks[cid] = signals[col]
            rows.append(
                {
                    "candidate_name": cid,
                    "source": source,
                    "filter_expression": cid,
                    "train_score": float(row.get("train_score", np.nan)),
                }
            )
    return pd.DataFrame(rows), combined_masks


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _baseline_row(signals: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    return_col = f"ret_h{int(args.candidate_horizon)}_net"
    row = {"candidate_id": "ALL_GREEN", "scope": "baseline"}
    row.update(V1._split_stats(signals, return_col, pd.Timestamp(args.train_end_date)))
    row["train_score"] = _selection_score(row)
    row["holdout_pass"] = _holdout_pass(row, args)
    return row


def write_summary(
    out_dir: Path,
    signals: pd.DataFrame,
    contrast: pd.DataFrame,
    atomic: pd.DataFrame,
    pairs: pd.DataFrame,
    scores: pd.DataFrame,
    structural: pd.DataFrame,
    selected_score_features: list[str],
    args: argparse.Namespace,
) -> None:
    baseline = _baseline_row(signals, args)
    lines = [
        "# 03 Pre-Orange Winner / Loser Contrast Summary",
        "",
        "橙灯不作为入场。03只研究橙灯出现前的环境，能否提高后续绿灯入场质量。",
        "",
        "## Baseline green",
        f"- train n={baseline['train_count']}, mean={baseline['train_mean_net']:.4%}, PF={baseline['train_profit_factor']:.3f}",
        f"- holdout n={baseline['holdout_count']}, mean={baseline['holdout_mean_net']:.4%}, PF={baseline['holdout_profit_factor']:.3f}",
        "",
        "## Stable winner / loser differences",
    ]
    stable = contrast[
        contrast["direction_stable"]
        & (pd.to_numeric(contrast["train_directional_strength"], errors="coerce") >= 0.08)
        & (pd.to_numeric(contrast["holdout_directional_strength"], errors="coerce") >= 0.05)
    ].head(10)
    if stable.empty:
        lines.append("- None with meaningful train/holdout direction stability.")
    else:
        for row in stable.itertuples(index=False):
            lines.append(
                f"- {row.feature} ({row.scope}/{row.family}): train AUC={row.train_auc:.3f}, "
                f"holdout AUC={row.holdout_auc:.3f}"
            )

    lines.extend(["", "## Pre-orange filters passing holdout"])
    passed = pd.concat(
        [
            atomic[atomic.get("holdout_pass", False)] if not atomic.empty else pd.DataFrame(),
            pairs[pairs.get("holdout_pass", False)] if not pairs.empty else pd.DataFrame(),
            scores[scores.get("holdout_pass", False)] if not scores.empty else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    if passed.empty:
        lines.append("- None. Pre-orange environment did not produce a stable gate.")
    else:
        passed = passed.sort_values("holdout_mean_net", ascending=False).head(10)
        for row in passed.itertuples(index=False):
            cid = getattr(row, "candidate_id")
            lines.append(
                f"- {cid}: train n={int(row.train_count)}, mean={row.train_mean_net:.4%}, "
                f"PF={row.train_profit_factor:.3f}; holdout n={int(row.holdout_count)}, "
                f"mean={row.holdout_mean_net:.4%}, PF={row.holdout_profit_factor:.3f}"
            )

    lines.extend(["", "## Train-selected transparent score"])
    if selected_score_features:
        lines.append("- components: " + ", ".join(selected_score_features))
    else:
        lines.append("- No positive train-only components met the minimum requirements.")

    lines.extend(["", "## Green entry with purple-low structural stop"])
    single = structural[structural.get("scheme", "") == "single_full"] if not structural.empty else pd.DataFrame()
    if single.empty:
        lines.append("- No completed structural simulations.")
    else:
        top = single.sort_values("profit_factor_on_max", ascending=False).head(10)
        for row in top.itertuples(index=False):
            lines.append(
                f"- {row.candidate_name}/{row.target_name}/cost={row.cost_mult:.1f}x: "
                f"n={int(row.trades)}, mean={row.mean_net_on_max:.4%}, PF={row.profit_factor_on_max:.3f}, "
                f"DD={row.max_drawdown_on_max:.2%}"
            )

    lines.extend(
        [
            "",
            "## Causal and research limits",
            "- pre_* features end before orange; orange_* uses only the closed orange bar.",
            "- final-low distance and future returns are labels/diagnostics only.",
            "- threshold boundaries and candidate ranking are learned on train only.",
            "- holdout is evaluation only; it does not select candidates.",
            "- green entry is next-bar open; purple-low stop is known by green time.",
            "- same-bar stop/target collision is stop-first.",
            "- many candidate tails are diagnostic multiple testing; passing 03 is not final strategy approval.",
        ]
    )
    (out_dir / "15_RESEARCH_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research(args: argparse.Namespace) -> dict[str, Any]:
    if args.data_source != "trade_bar":
        raise ValueError("03 requires --data-source trade_bar")
    horizons = tuple(int(x) for x in _parse_list(args.horizons, cast=int, name="horizons"))
    if int(args.candidate_horizon) not in horizons:
        raise ValueError("candidate_horizon must be included in horizons")
    windows = [int(x) for x in _parse_list(args.pre_windows, cast=int, name="pre_windows")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = V1.load_bars(args)
    coverage = validate_trade_bar_orderflow(bars)
    print("[features] building causal trade-bar order-flow features", flush=True)
    orderflow = build_trade_bar_orderflow_features(bars, baseline_window=int(args.orderflow_baseline_window))
    context = V1.build_context_features(bars)
    stage_events, detector_features = V1.build_stage_events(bars, context, args, horizons)
    if stage_events.empty:
        raise RuntimeError("No panic episode stages detected")
    stage_events, causal_audit = V1.attach_next_open_outcomes(stage_events, bars, args, horizons)

    episode_orderflow = summarize_episode_orderflow(
        stage_events,
        orderflow,
        progress_every=int(args.progress_every),
        progress_enabled=not bool(args.no_progress),
    )
    enriched = attach_orderflow_to_stage_events(stage_events, orderflow, episode_orderflow)
    starts = enriched[enriched["stage"] == "start"].copy()
    if starts.empty:
        raise RuntimeError("No orange/start nodes detected")

    print("[orange] building strict pre-orange environment table", flush=True)
    orange_features, feature_meta = build_pre_orange_features(
        bars,
        orderflow,
        starts,
        windows=windows,
        progress_enabled=not bool(args.no_progress),
    )
    signals = join_orange_features_to_green(
        enriched,
        orange_features,
        candidate_horizon=int(args.candidate_horizon),
        winner_threshold=float(args.winner_threshold),
        strong_winner_threshold=float(args.strong_winner_threshold),
        strong_loser_threshold=float(args.strong_loser_threshold),
    )
    if signals.empty:
        raise RuntimeError("No green signals available for orange gating")

    train_end = pd.Timestamp(args.train_end_date)
    contrast = winner_loser_contrast(signals, feature_meta, label_col="label_winner", train_end=train_end)
    strong_signals = signals[signals["label_strong_contrast"].notna()].copy()
    strong_contrast = winner_loser_contrast(
        strong_signals,
        feature_meta,
        label_col="label_strong_contrast",
        train_end=train_end,
    ) if not strong_signals.empty else pd.DataFrame()

    pre_atomic, pre_atomic_masks = evaluate_train_tail_filters(
        signals, feature_meta, args, scope="pre_orange"
    )
    orange_atomic, _ = evaluate_train_tail_filters(
        signals, feature_meta, args, scope="orange_closed"
    )
    pre_pairs, pre_pair_masks = evaluate_train_selected_pairs(
        signals, pre_atomic, pre_atomic_masks, args
    )
    score_summary, score_masks, selected_score_features = build_train_selected_score(
        signals, pre_atomic, pre_atomic_masks, args
    )

    structural_candidates, _ = build_structural_candidates(
        signals,
        pre_atomic,
        pre_atomic_masks,
        pre_pairs,
        pre_pair_masks,
        score_summary,
        score_masks,
        args,
    )
    trades, structural_summary_all, structural_yearly_all = V1.simulate_cluster_variants(
        bars,
        signals,
        structural_candidates,
        args,
    )
    structural_summary = structural_summary_all[
        structural_summary_all["scheme"] == "single_full"
    ].copy() if not structural_summary_all.empty else pd.DataFrame()
    structural_yearly = structural_yearly_all[
        structural_yearly_all["scheme"] == "single_full"
    ].copy() if not structural_yearly_all.empty else pd.DataFrame()
    structural_trades = trades[trades["scheme"] == "single_full"].copy() if not trades.empty else pd.DataFrame()

    write_csv(coverage, out_dir / "01_trade_bar_field_coverage.csv")
    write_csv(feature_meta, out_dir / "02_feature_dictionary.csv")
    write_csv(orange_features, out_dir / "03_orange_pre_features.csv")
    write_csv(signals, out_dir / "04_green_signals_with_orange_features.csv")
    write_csv(contrast, out_dir / "05_winner_loser_contrast_train_holdout.csv")
    write_csv(strong_contrast, out_dir / "06_strong_winner_loser_contrast.csv")
    write_csv(pre_atomic, out_dir / "07_pre_orange_tail_filters_train_holdout.csv")
    write_csv(orange_atomic, out_dir / "08_orange_closed_tail_filters_secondary.csv")
    write_csv(pre_pairs, out_dir / "09_pre_orange_pairs_train_holdout.csv")
    write_csv(score_summary, out_dir / "10_train_selected_score_train_holdout.csv")
    write_csv(structural_candidates, out_dir / "11_structural_filter_candidates.csv")
    write_csv(structural_summary, out_dir / "12_green_structural_stop_summary.csv")
    write_csv(structural_yearly, out_dir / "13_green_structural_stop_yearly.csv")
    trade_out = structural_trades
    if int(args.save_trade_sample) > 0 and len(trade_out) > int(args.save_trade_sample):
        trade_out = trade_out.sort_values(["candidate_name", "entry_time"]).head(int(args.save_trade_sample))
    write_csv(trade_out, out_dir / "14_green_structural_trades_sample.csv")
    write_csv(causal_audit, out_dir / "16_causal_audit.csv")
    write_summary(
        out_dir,
        signals,
        contrast,
        pre_atomic,
        pre_pairs,
        score_summary,
        structural_summary,
        selected_score_features,
        args,
    )

    meta = {
        "script": Path(__file__).name,
        "research_family": "liquidity/panic_selloff_rejection_recovery_long",
        "research_question": "Can pre-orange environment improve later green-entry quality?",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "data_source": args.data_source,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "train_end_date": args.train_end_date,
        "bar_rows": int(len(bars)),
        "detector_feature_rows": int(len(detector_features)),
        "orange_count": int(len(orange_features)),
        "green_signal_count": int(len(signals)),
        "pre_orange_feature_count": int((feature_meta["scope"] == "pre_orange").sum()),
        "orange_closed_feature_count": int((feature_meta["scope"] == "orange_closed").sum()),
        "selected_score_features": selected_score_features,
        "structural_candidate_count": int(len(structural_candidates)),
        "cost_convention": {
            "round_trip_fee": float(args.entry_fee_rate + args.exit_fee_rate),
            "round_trip_slippage": float(args.entry_slippage_pct + args.exit_slippage_pct),
            "cost_multipliers": _parse_list(args.cost_multipliers, cast=float, name="cost_multipliers"),
        },
        "causal_guards": [
            "all pre_orange rolling features shift by one closed bar",
            "orange_closed features use only the closed orange bar",
            "diagnostic final-low fields are excluded from feature metadata and masks",
            "future green outcomes are labels only",
            "tail thresholds and candidate ranking use train only",
            "holdout never selects candidates",
            "green entry executes next-bar open",
            "episode low used as stop is already known at green signal time",
            "same-bar stop and target collision is stop-first",
        ],
        "params": vars(args),
    }
    write_json(out_dir / "00_manifest.json", meta)
    finalize_research_report(
        out_dir,
        title="03 Panic Recovery Pre-Orange Winner/Loser Contrast",
        print_log=True,
    )
    print(f"[done] reports -> {out_dir}", flush=True)
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
