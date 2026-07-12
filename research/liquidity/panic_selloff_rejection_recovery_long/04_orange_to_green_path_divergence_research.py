#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""04 causal orange-to-green path divergence research.

Orange remains an observation gate and green remains the only tested entry.
This pass compares the complete *causal* path available when green closes:
price trajectory, repeated lows, recovery shape, CVD/large-flow evolution,
activity decay and absorption. Future bars are kept in a separate ``post_*``
diagnostic namespace and can never become filter features.
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
from research.liquidity.panic_selloff_rejection_recovery_long.common.orange_to_green_path import (  # noqa: E402
    attach_post_green_path_diagnostics,
    build_orange_to_green_path_features,
)
from research.liquidity.panic_selloff_rejection_recovery_long.common.pre_orange_environment import (  # noqa: E402
    numeric_series,
    winner_loser_contrast,
)

SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}


def _load_sibling(filename: str, module_name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load research helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


V1 = _load_sibling("01_environment_and_cluster_scale_in_research.py", "panic_recovery_01_shared_for_04")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="04 orange-to-green causal path divergence and green-quality filters",
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
        default="data/reports/research/liquidity/panic_selloff_rejection_recovery_long/04_orange_to_green_path_divergence",
    )

    # Detector stays unchanged from 01-03.
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
    p.add_argument("--horizons", default="5,15,30,60,120,240")
    p.add_argument("--candidate-horizon", type=int, default=60)
    p.add_argument("--post-path-horizon", type=int, default=240)
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--low-retest-tolerance-pct", type=float, default=0.0008)
    p.add_argument("--winner-threshold", type=float, default=0.0)
    p.add_argument("--strong-winner-threshold", type=float, default=0.0025)
    p.add_argument("--strong-loser-threshold", type=float, default=-0.0015)

    # Thresholds are learned on 2023-2024 only, then frozen.
    p.add_argument("--tail-quantiles", default="0.20,0.30")
    p.add_argument("--min-filter-train", type=int, default=80)
    p.add_argument("--min-filter-holdout", type=int, default=35)
    p.add_argument("--top-atomic-for-pairs", type=int, default=12)
    p.add_argument("--max-score-features", type=int, default=6)
    p.add_argument("--top-train-candidates", type=int, default=10)

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
        if token:
            value = cast(token)
            if float(value) <= 0:
                raise ValueError(f"{name} must contain positive values")
            values.append(value)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return sorted(set(values))


def _safe_id(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in str(text))


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
    values = pd.to_numeric(values, errors="coerce")
    if direction == "le":
        return values <= threshold
    if direction == "ge":
        return values >= threshold
    raise ValueError(direction)


def evaluate_path_tail_filters(
    signals: pd.DataFrame,
    feature_meta: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = pd.Timestamp(args.train_end_date)
    train = signals[pd.to_datetime(signals["event_time"]) <= train_end]
    return_col = f"ret_h{int(args.candidate_horizon)}_net"
    quantiles = _parse_list(args.tail_quantiles, cast=float, name="tail_quantiles")
    if any(q >= 0.5 for q in quantiles):
        raise ValueError("tail_quantiles must stay below 0.5")
    meta = feature_meta[feature_meta["scope"] == "orange_to_green_path"].copy()
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
                cid = _safe_id(f"{feature}__{direction}__q{quantile:.2f}")
                row = {
                    "candidate_id": cid,
                    "feature": feature,
                    "family": meta_map[feature]["family"],
                    "scope": "orange_to_green_path",
                    "description": meta_map[feature]["description"],
                    "direction": direction,
                    "train_quantile": quantile,
                    "threshold": threshold,
                    **V1._split_stats(signals[mask], return_col, train_end),
                }
                row["train_score"] = _selection_score(row)
                row["holdout_pass"] = _holdout_pass(row, args)
                rows.append(row)
                masks[cid] = mask
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["holdout_pass", "train_score", "train_count"], ascending=[False, False, False]).reset_index(drop=True)
    return out, pd.DataFrame(masks, index=signals.index)


def evaluate_path_pairs(
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
    eligible = eligible.drop_duplicates("feature").head(int(args.top_atomic_for_pairs))
    records = eligible.to_dict(orient="records")
    return_col = f"ret_h{int(args.candidate_horizon)}_net"
    train_end = pd.Timestamp(args.train_end_date)
    rows: list[dict[str, Any]] = []
    masks: dict[str, pd.Series] = {}
    for left, right in itertools.combinations(records, 2):
        if left["family"] == right["family"]:
            continue
        left_id, right_id = str(left["candidate_id"]), str(right["candidate_id"])
        mask = atomic_masks[left_id] & atomic_masks[right_id]
        cid = _safe_id(f"PATH_PAIR__{left_id}__{right_id}")
        row = {
            "candidate_id": cid,
            "left_candidate": left_id,
            "right_candidate": right_id,
            "left_feature": left["feature"],
            "right_feature": right["feature"],
            "family": f"{left['family']}+{right['family']}",
            "scope": "path_pair",
            **V1._split_stats(signals[mask], return_col, train_end),
        }
        row["train_score"] = _selection_score(row)
        row["holdout_pass"] = _holdout_pass(row, args)
        rows.append(row)
        masks[cid] = mask
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["holdout_pass", "train_score", "train_count"], ascending=[False, False, False]).reset_index(drop=True)
    return out, pd.DataFrame(masks, index=signals.index)


def build_path_score(
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
    ids = selected["candidate_id"].astype(str).tolist()
    if not ids:
        return pd.DataFrame(), pd.DataFrame(index=signals.index), []
    score = atomic_masks[ids].fillna(False).astype(int).sum(axis=1)
    rows: list[dict[str, Any]] = []
    masks: dict[str, pd.Series] = {}
    return_col = f"ret_h{int(args.candidate_horizon)}_net"
    train_end = pd.Timestamp(args.train_end_date)
    for threshold in range(1, min(4, len(ids)) + 1):
        cid = f"PATH_SCORE_GE_{threshold}"
        mask = score >= threshold
        row = {
            "candidate_id": cid,
            "score_threshold": threshold,
            "selected_features": ";".join(ids),
            "family": "path_score",
            "scope": "path_score",
            **V1._split_stats(signals[mask], return_col, train_end),
        }
        row["train_score"] = _selection_score(row)
        row["holdout_pass"] = _holdout_pass(row, args)
        rows.append(row)
        masks[cid] = mask
    return pd.DataFrame(rows).sort_values("train_score", ascending=False), pd.DataFrame(masks, index=signals.index), ids


def build_structural_candidates(
    signals: pd.DataFrame,
    atomic: pd.DataFrame,
    atomic_masks: pd.DataFrame,
    pairs: pd.DataFrame,
    pair_masks: pd.DataFrame,
    scores: pd.DataFrame,
    score_masks: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows = [{"candidate_name": "ALL_GREEN", "source": "baseline", "filter_expression": "ALL", "train_score": np.nan}]
    pools = [
        (atomic, atomic_masks, "path_atomic"),
        (pairs, pair_masks, "path_pair"),
        (scores, score_masks, "path_score"),
    ]
    candidates: list[dict[str, Any]] = []
    for frame, masks, source in pools:
        if frame.empty:
            continue
        for row in frame.to_dict(orient="records"):
            cid = str(row["candidate_id"])
            if cid in masks.columns:
                candidates.append({**row, "_source": source})
    if candidates:
        pool = pd.DataFrame(candidates).sort_values("train_score", ascending=False)
        pool = pool.drop_duplicates("candidate_id").head(int(args.top_train_candidates))
        for row in pool.to_dict(orient="records"):
            cid = str(row["candidate_id"])
            source = str(row["_source"])
            masks = atomic_masks if source == "path_atomic" else pair_masks if source == "path_pair" else score_masks
            signals[f"filter__{cid}"] = masks[cid].fillna(False).astype(bool).to_numpy()
            rows.append({
                "candidate_name": cid,
                "source": source,
                "filter_expression": cid,
                "train_score": float(row.get("train_score", np.nan)),
            })
    return pd.DataFrame(rows)


def phase_profile(signals: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    train_end = pd.Timestamp(args.train_end_date)
    phase_metrics = ["price_return", "delta_ratio", "large_delta_ratio", "sell_intensity", "absorption"]
    for split_name, split_mask in (
        ("train", pd.to_datetime(signals["event_time"]) <= train_end),
        ("holdout", pd.to_datetime(signals["event_time"]) > train_end),
    ):
        for outcome, outcome_mask in (("winner", signals["label_horizon_winner"]), ("loser", ~signals["label_horizon_winner"])):
            part = signals[split_mask & outcome_mask]
            for q in range(1, 5):
                row: dict[str, Any] = {"split": split_name, "outcome": outcome, "phase": q, "count": len(part)}
                for metric in phase_metrics:
                    row[metric] = pd.to_numeric(part.get(f"path_q{q}_{metric}"), errors="coerce").mean()
                rows.append(row)
    return pd.DataFrame(rows)


def post_path_summary(signals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "post_risk_pct", "post_mfe_r", "post_mae_r", "post_time_to_mfe_bars",
        "post_time_to_mae_bars", "post_horizon_net", "post_close_peak_giveback",
    ]
    rows: list[dict[str, Any]] = []
    yearly: list[dict[str, Any]] = []
    for outcome, part in signals.groupby("post_outcome_class", dropna=False):
        row = {"outcome_class": outcome, "count": len(part), "share": len(part) / max(1, len(signals))}
        for metric in metrics:
            s = pd.to_numeric(part.get(metric), errors="coerce")
            row[f"{metric}_mean"] = s.mean()
            row[f"{metric}_median"] = s.median()
        rows.append(row)
    years = pd.to_datetime(signals["event_time"]).dt.year
    for (year, outcome), part in signals.assign(_year=years).groupby(["_year", "post_outcome_class"], dropna=False):
        yearly.append({
            "year": int(year), "outcome_class": outcome, "count": len(part),
            "share_in_year": len(part) / max(1, int((years == year).sum())),
            "mean_horizon_net": pd.to_numeric(part["post_horizon_net"], errors="coerce").mean(),
            "mean_mfe_r": pd.to_numeric(part["post_mfe_r"], errors="coerce").mean(),
        })
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _baseline(signals: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    row = V1._split_stats(signals, f"ret_h{int(args.candidate_horizon)}_net", pd.Timestamp(args.train_end_date))
    return row


def write_summary(
    out_dir: Path,
    signals: pd.DataFrame,
    contrast: pd.DataFrame,
    structural_contrast: pd.DataFrame,
    atomic: pd.DataFrame,
    pairs: pd.DataFrame,
    scores: pd.DataFrame,
    structural: pd.DataFrame,
    post_summary: pd.DataFrame,
    score_features: list[str],
    args: argparse.Namespace,
) -> None:
    base = _baseline(signals, args)
    lines = [
        "# 04 Orange-to-Green Path Divergence Summary", "",
        "橙灯不入场；所有候选路径特征只使用橙灯至绿灯收盘之间已知的数据。", "",
        "## Baseline green",
        f"- train n={base['train_count']}, mean={base['train_mean_net']:.4%}, PF={base['train_profit_factor']:.3f}",
        f"- holdout n={base['holdout_count']}, mean={base['holdout_mean_net']:.4%}, PF={base['holdout_profit_factor']:.3f}",
        "", "## Stable causal path differences",
    ]
    stable = contrast[
        contrast["direction_stable"]
        & (pd.to_numeric(contrast["train_directional_strength"], errors="coerce") >= 0.08)
        & (pd.to_numeric(contrast["holdout_directional_strength"], errors="coerce") >= 0.05)
    ].head(12)
    if stable.empty:
        lines.append("- None with meaningful train/holdout direction stability.")
    else:
        for row in stable.itertuples(index=False):
            lines.append(f"- {row.feature} ({row.family}): train AUC={row.train_auc:.3f}, holdout AUC={row.holdout_auc:.3f}")

    lines.extend(["", "## Structural 1.5R winner differences"])
    stable_struct = structural_contrast[
        structural_contrast["direction_stable"]
        & (pd.to_numeric(structural_contrast["train_directional_strength"], errors="coerce") >= 0.08)
        & (pd.to_numeric(structural_contrast["holdout_directional_strength"], errors="coerce") >= 0.05)
    ].head(10)
    if stable_struct.empty:
        lines.append("- None with meaningful train/holdout direction stability.")
    else:
        for row in stable_struct.itertuples(index=False):
            lines.append(f"- {row.feature}: train AUC={row.train_auc:.3f}, holdout AUC={row.holdout_auc:.3f}")

    lines.extend(["", "## Path filters passing holdout"])
    passed_parts = []
    for frame in (atomic, pairs, scores):
        if not frame.empty and "holdout_pass" in frame:
            passed_parts.append(frame[frame["holdout_pass"]])
    passed = pd.concat(passed_parts, ignore_index=True, sort=False) if passed_parts else pd.DataFrame()
    if passed.empty:
        lines.append("- None. Orange-to-green path did not produce a stable green gate.")
    else:
        for row in passed.sort_values("holdout_mean_net", ascending=False).head(10).itertuples(index=False):
            lines.append(
                f"- {row.candidate_id}: train n={int(row.train_count)}, mean={row.train_mean_net:.4%}, PF={row.train_profit_factor:.3f}; "
                f"holdout n={int(row.holdout_count)}, mean={row.holdout_mean_net:.4%}, PF={row.holdout_profit_factor:.3f}"
            )

    lines.extend(["", "## Train-selected path score"])
    lines.append("- components: " + ", ".join(score_features) if score_features else "- No positive train-only components.")

    lines.extend(["", "## Green entry with purple-low structural stop"])
    single = structural[structural.get("scheme", "") == "single_full"] if not structural.empty else pd.DataFrame()
    if single.empty:
        lines.append("- No completed structural simulations.")
    else:
        for row in single.sort_values("profit_factor_on_max", ascending=False).head(10).itertuples(index=False):
            lines.append(
                f"- {row.candidate_name}/{row.target_name}/cost={row.cost_mult:.1f}x: n={int(row.trades)}, "
                f"mean={row.mean_net_on_max:.4%}, PF={row.profit_factor_on_max:.3f}, DD={row.max_drawdown_on_max:.2%}"
            )

    lines.extend(["", "## Post-green path classes (diagnostic only)"])
    if post_summary.empty:
        lines.append("- No completed post-green path diagnostics.")
    else:
        for row in post_summary.sort_values("count", ascending=False).itertuples(index=False):
            lines.append(f"- {row.outcome_class}: n={int(row.count)}, share={row.share:.1%}, mean MFE={row.post_mfe_r_mean:.2f}R")

    lines.extend([
        "", "## Causal limits",
        "- Candidate path_window_end always equals the closed green event_time.",
        "- Purple low, retests and recovery aggregates are all already visible by green close.",
        "- post_* columns use future bars only for labels and path diagnosis; they are excluded from feature metadata and masks.",
        "- Tail boundaries, pair selection and score components use train only; holdout never selects.",
        "- Entry is next-bar open; structural stop is below the already-known purple low.",
        "- Same-bar stop/target collision is stop-first.",
    ])
    (out_dir / "19_RESEARCH_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_research(args: argparse.Namespace) -> dict[str, Any]:
    if args.data_source != "trade_bar":
        raise ValueError("04 requires --data-source trade_bar")
    horizons = tuple(int(x) for x in _parse_list(args.horizons, cast=int, name="horizons"))
    if int(args.candidate_horizon) not in horizons:
        raise ValueError("candidate_horizon must be included in horizons")
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
        stage_events, orderflow,
        progress_every=int(args.progress_every),
        progress_enabled=not bool(args.no_progress),
    )
    enriched = attach_orderflow_to_stage_events(stage_events, orderflow, episode_orderflow)
    print("[path] building causal orange-to-green trajectories", flush=True)
    path_features, feature_meta = build_orange_to_green_path_features(
        bars, orderflow, enriched,
        low_retest_tolerance_pct=float(args.low_retest_tolerance_pct),
        progress_enabled=not bool(args.no_progress),
        progress_every=int(args.progress_every),
    )
    signals = enriched[enriched["stage"] == "signal"].copy().sort_values("event_time").reset_index(drop=True)
    signals = signals.merge(path_features, on="episode_id", how="inner", validate="one_to_one")
    if signals.empty:
        raise RuntimeError("No green signals with complete orange-to-green path")
    signals = attach_post_green_path_diagnostics(
        signals, bars,
        horizon=int(args.post_path_horizon),
        entry_delay_bars=int(args.entry_delay_bars),
        stop_buffer_pct=float(args.stop_buffer_pct),
        entry_fee_rate=float(args.entry_fee_rate),
        exit_fee_rate=float(args.exit_fee_rate),
        entry_slippage_pct=float(args.entry_slippage_pct),
        exit_slippage_pct=float(args.exit_slippage_pct),
    )
    return_col = f"ret_h{int(args.candidate_horizon)}_net"
    signals["label_horizon_winner"] = pd.to_numeric(signals[return_col], errors="coerce") > float(args.winner_threshold)
    signals["label_strong_winner"] = pd.to_numeric(signals[return_col], errors="coerce") >= float(args.strong_winner_threshold)
    signals["label_strong_loser"] = pd.to_numeric(signals[return_col], errors="coerce") <= float(args.strong_loser_threshold)
    signals["label_structural_1_5R_winner"] = signals["post_target_1_5R_before_stop"].fillna(False).astype(bool)

    train_end = pd.Timestamp(args.train_end_date)
    contrast = winner_loser_contrast(signals, feature_meta, label_col="label_horizon_winner", train_end=train_end)
    structural_contrast = winner_loser_contrast(signals, feature_meta, label_col="label_structural_1_5R_winner", train_end=train_end)
    atomic, atomic_masks = evaluate_path_tail_filters(signals, feature_meta, args)
    pairs, pair_masks = evaluate_path_pairs(signals, atomic, atomic_masks, args)
    scores, score_masks, score_features = build_path_score(signals, atomic, atomic_masks, args)
    candidates = build_structural_candidates(signals, atomic, atomic_masks, pairs, pair_masks, scores, score_masks, args)
    trades, structural_all, yearly_all = V1.simulate_cluster_variants(bars, signals, candidates, args)
    structural = structural_all[structural_all["scheme"] == "single_full"].copy() if not structural_all.empty else pd.DataFrame()
    yearly = yearly_all[yearly_all["scheme"] == "single_full"].copy() if not yearly_all.empty else pd.DataFrame()
    trade_sample = trades[trades["scheme"] == "single_full"].copy() if not trades.empty else pd.DataFrame()
    if int(args.save_trade_sample) > 0 and len(trade_sample) > int(args.save_trade_sample):
        trade_sample = trade_sample.sort_values(["candidate_name", "entry_time"]).head(int(args.save_trade_sample))

    phases = phase_profile(signals, args)
    post_summary, post_yearly = post_path_summary(signals)

    # Hard audit: no future diagnostic may be a candidate feature.
    if feature_meta["feature"].astype(str).str.startswith("post_").any():
        raise AssertionError("future post_* field leaked into candidate feature metadata")
    if not (pd.to_datetime(signals["path_window_end"]) == pd.to_datetime(signals["event_time"])).all():
        raise AssertionError("path feature window extends beyond green signal time")

    _write_csv(coverage, out_dir / "01_trade_bar_field_coverage.csv")
    _write_csv(feature_meta, out_dir / "02_path_feature_dictionary.csv")
    _write_csv(path_features, out_dir / "03_orange_to_green_path_features.csv")
    _write_csv(signals, out_dir / "04_green_signals_with_causal_path_and_post_diagnostics.csv")
    _write_csv(contrast, out_dir / "05_path_winner_loser_contrast_train_holdout.csv")
    _write_csv(structural_contrast, out_dir / "06_path_structural_1_5R_contrast_train_holdout.csv")
    _write_csv(phases, out_dir / "07_normalized_phase_profile_winner_loser.csv")
    _write_csv(atomic, out_dir / "08_path_atomic_filters_train_holdout.csv")
    _write_csv(pairs, out_dir / "09_path_pairs_train_holdout.csv")
    _write_csv(scores, out_dir / "10_train_selected_path_score_train_holdout.csv")
    _write_csv(post_summary, out_dir / "11_post_green_path_class_summary.csv")
    _write_csv(post_yearly, out_dir / "12_post_green_path_class_yearly.csv")
    _write_csv(candidates, out_dir / "13_structural_filter_candidates.csv")
    _write_csv(structural, out_dir / "14_green_structural_stop_summary.csv")
    _write_csv(yearly, out_dir / "15_green_structural_stop_yearly.csv")
    _write_csv(trade_sample, out_dir / "16_green_structural_trades_sample.csv")
    _write_csv(causal_audit, out_dir / "18_causal_audit.csv")
    write_summary(out_dir, signals, contrast, structural_contrast, atomic, pairs, scores, structural, post_summary, score_features, args)

    meta = {
        "script": Path(__file__).name,
        "research_family": "liquidity/panic_selloff_rejection_recovery_long",
        "research_question": "Which causal orange-to-green trajectories improve later green-entry quality?",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "data_source": args.data_source,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "train_end_date": args.train_end_date,
        "bar_rows": int(len(bars)),
        "detector_feature_rows": int(len(detector_features)),
        "green_signal_count": int(len(signals)),
        "path_feature_count": int(len(feature_meta)),
        "selected_path_score_features": score_features,
        "structural_candidate_count": int(len(candidates)),
        "cost_convention": {
            "round_trip_fee": float(args.entry_fee_rate + args.exit_fee_rate),
            "round_trip_slippage": float(args.entry_slippage_pct + args.exit_slippage_pct),
            "cost_multipliers": _parse_list(args.cost_multipliers, cast=float, name="cost_multipliers"),
        },
        "causal_guards": [
            "all candidate path slices end exactly at green closed bar",
            "orange-to-low and low-to-green fields use only already printed bars",
            "post_* future path fields are labels/diagnostics only",
            "feature metadata is an explicit allow-list and contains no post_* field",
            "threshold boundaries and ranking use train only",
            "holdout never selects candidates",
            "entry executes next-bar open",
            "purple-low structural stop is already known at green time",
            "same-bar stop/target collision is stop-first",
        ],
        "params": vars(args),
    }
    _write_json(out_dir / "00_manifest.json", meta)
    finalize_research_report(out_dir, title="04 Panic Recovery Orange-to-Green Path Divergence", print_log=True)
    print(f"[done] reports -> {out_dir}", flush=True)
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    run_research(parse_args(argv))


if __name__ == "__main__":
    main()
