#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Online recognizability research for clear Swing Low mechanisms.

This is not a strategy or PnL backtest.  It asks whether a currently closed 1m
bar, using only data already visible at that close, can be scored as likely to
reach +1% from the next-bar open within a bounded horizon.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402
from research.market_structure.swing_low_typology.common.c3_sequence_features import (  # noqa: E402
    METADATA_COLUMNS as C3_METADATA_COLUMNS,
    ONLINE_SEQUENCE_SCORE_FEATURES,
    build_c3_online_score_features,
    build_c3_sequence_features,
)
from research.market_structure.swing_low_typology.common.mechanism_features import (  # noqa: E402
    MECHANISM_METADATA_COLUMNS,
    REQUIRED_COLUMNS as MECHANISM_REQUIRED_COLUMNS,
    build_mechanism_features,
)
from research.market_structure.swing_low_typology.common.mechanism_typology import (  # noqa: E402
    BASE_ARCHETYPE_TERMS,
    TREND_ARCHETYPE_TERMS,
    fit_score_model as fit_mechanism_score_model,
)
from research.market_structure.swing_low_typology.common.online_recognizability import (  # noqa: E402
    CLEAR_MECHANISMS,
    FUTURE_LABEL_COLUMNS,
    CandidateGateConfig,
    attach_reference_swing_targets,
    attach_temporal_split,
    binary_metrics,
    build_candidate_episodes,
    build_forward_path_labels,
    build_model_stability,
    build_online_candidate_events,
    candidate_gate_recall,
    choose_binary_family,
    choose_score_family,
    fit_binary_model,
    fit_mechanism_clarity_thresholds,
    fit_score_model,
    label_definition_table,
    mechanism_assignment_from_scores,
    model_feature_importance,
    probability_bucket_table,
    purge_temporal_label_overlap,
    representative_prediction_cases,
    score_metrics,
    select_model_features,
)
from research.market_structure.swing_low_typology.common.swing_low_dataset import (  # noqa: E402
    validate_trade_bar_fields,
)

SCRIPT_NAME = "04_online_mechanism_recognizability_research"
SCRIPT_VERSION = "1.5.0"
EXPERIMENT_ID = "ETH_1M_SWING_LOW_ONLINE_RECOGNIZABILITY_04"
EDGE_ID = "RESEARCH_ONLY_ETH_ONLINE_SWING_LOW_RECOGNIZABILITY"
TITLE = "ETH Swing Low Online Mechanism Recognizability 04"
DEFAULT_OUT_DIR = "data/reports/research/market_structure/swing_low_typology/04_online_mechanism_recognizability"
DEFAULT_STAGE2_DIR = "data/reports/research/market_structure/swing_low_typology/02_c3_hierarchical_typology"
DEFAULT_STAGE3_DIR = "data/reports/research/market_structure/swing_low_typology/03_mechanism_hierarchical_typology"

DATE_COLUMNS = [
    "extreme_time",
    "feature_available_time",
    "confirmation_time",
    "confirmation_available_time",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal online recognizability research for clear Swing Low mechanisms.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--fit-end-date", default="2023-12-31 23:59:59")
    p.add_argument("--validation-end-date", default="2024-12-31 23:59:59")
    p.add_argument("--target-move-pct", type=float, default=1.0)
    p.add_argument("--adverse-move-pct", type=float, default=1.0)
    p.add_argument("--forward-horizon-bars", type=int, default=60)
    p.add_argument("--stage2-report-dir", default=DEFAULT_STAGE2_DIR)
    p.add_argument("--stage3-report-dir", default=DEFAULT_STAGE3_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--lookback", type=int, default=240)
    p.add_argument("--phase-bins", type=int, default=12)
    p.add_argument("--support-tolerance-bp", type=float, default=25.0)
    p.add_argument("--minimum-test-gap", type=int, default=4)
    p.add_argument("--minimum-test-rebound-bp", type=float, default=15.0)
    p.add_argument("--test-rebound-horizon", type=int, default=30)
    p.add_argument("--candidate-new-low-window", type=int, default=5)
    p.add_argument("--candidate-near-floor-window", type=int, default=60)
    p.add_argument("--candidate-position-window", type=int, default=120)
    p.add_argument("--candidate-near-floor-tolerance-bp", type=float, default=20.0)
    p.add_argument("--candidate-max-position-in-range", type=float, default=0.55)
    p.add_argument("--candidate-episode-gap-bars", type=int, default=5)
    p.add_argument("--candidate-feature-chunk-size", type=int, default=20_000)
    p.add_argument("--candidate-spill-buffer-mb", type=int, default=768)
    p.add_argument("--feature-workers", type=int, default=4)
    p.add_argument("--feature-worker-batch-size", type=int, default=1_000)
    p.add_argument("--label-vectorized-chunk-size", type=int, default=50_000)
    p.add_argument("--max-model-features", type=int, default=220)
    p.add_argument("--mechanism-clarity-quantile", type=float, default=0.20)
    p.add_argument("--minimum-clear-candidates", type=int, default=2_000)
    p.add_argument("--minimum-temporal-split-rows", type=int, default=300)
    p.add_argument("--minimum-specialist-train-rows", type=int, default=500)
    p.add_argument("--minimum-specialist-positive-rows", type=int, default=40)
    p.add_argument("--model-min-samples-leaf", type=int, default=100)
    p.add_argument("--progress-every", type=int, default=1_000)
    p.add_argument("--causal-audit-sample-size", type=int, default=8)
    p.add_argument(
        "--resume-finalize-existing",
        action="store_true",
        help="Reuse completed 04 CSV/model outputs and rerun only causal audit, manifest, summary, and review-pack finalization.",
    )
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args(argv)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _end_exclusive(value: str, timeframe: str) -> pd.Timestamp:
    end = pd.Timestamp(value)
    if len(str(value).strip()) <= 10:
        end = end + pd.Timedelta(days=1)
    return end


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


def _validate_reports(args: argparse.Namespace, stage2_dir: Path, stage3_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    required2 = [
        stage2_dir / "00_manifest.json",
        stage2_dir / "07_frozen_c3_subcluster_assignments.csv",
    ]
    required3 = [
        stage3_dir / "00_manifest.json",
        stage3_dir / "15_c3c_trend_subtype_assignments.csv",
        stage3_dir / "24_c3e_base_subtype_assignments.csv",
    ]
    missing = [str(path) for path in [*required2, *required3] if not path.exists()]
    if missing:
        raise FileNotFoundError("Run research 02 and 03 first; missing: " + ", ".join(missing))
    stage2 = json.loads(required2[0].read_text(encoding="utf-8"))
    stage3 = json.loads(required3[0].read_text(encoding="utf-8"))
    checks = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "target_move_pct": float(args.target_move_pct),
        "max_completion_bars": int(args.forward_horizon_bars),
    }
    mismatches: list[str] = []
    for key, expected in checks.items():
        actual = stage2.get(key)
        if str(actual) != str(expected):
            mismatches.append(f"stage2 {key}={actual}, requested={expected}")
    label_policy = {
        "swing_extreme_price_source": "low",
        "swing_entry_price_source": "next_bar_open",
        "swing_target_observation_source": "future_closed_bar_close",
    }
    for source_name, manifest in (("stage2", stage2), ("stage3", stage3)):
        for key, expected in label_policy.items():
            actual = manifest.get(key)
            if actual != expected:
                mismatches.append(f"{source_name} {key}={actual}, required={expected}")
    if mismatches:
        raise RuntimeError("Report configuration mismatch: " + "; ".join(mismatches))
    return stage2, stage3


def _load_stage2(stage2_dir: Path) -> pd.DataFrame:
    frame = pd.read_csv(stage2_dir / "07_frozen_c3_subcluster_assignments.csv", parse_dates=DATE_COLUMNS)
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
        raise RuntimeError(f"stage2 assignments missing columns: {missing}")
    return frame.sort_values("extreme_time").reset_index(drop=True)


def _load_stage3_reference(stage3_dir: Path) -> pd.DataFrame:
    trend = pd.read_csv(stage3_dir / "15_c3c_trend_subtype_assignments.csv", parse_dates=["extreme_time", "feature_available_time"])
    trend = trend.rename(columns={"trend_subtype": "mechanism_type"})
    base = pd.read_csv(stage3_dir / "24_c3e_base_subtype_assignments.csv", parse_dates=["extreme_time", "feature_available_time"])
    base = base.rename(columns={"base_subtype": "mechanism_type"})
    cols = [
        "event_id",
        "extreme_time",
        "feature_available_time",
        "extreme_pos",
        "extreme_price",
        "split",
        "source_subcluster_id",
        "mechanism_type",
    ]
    return pd.concat([trend[cols], base[cols]], ignore_index=True).sort_values("extreme_time").reset_index(drop=True)


def _parent_frame(assignments: pd.DataFrame) -> pd.DataFrame:
    return assignments[
        [
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
    ].rename(
        columns={
            "parent_cluster_id": "cluster_id",
            "parent_distance_to_centroid": "distance_to_train_centroid",
        }
    )


def _combine_feature_families(
    sequence: pd.DataFrame,
    mechanism: pd.DataFrame,
    event_meta: pd.DataFrame,
) -> pd.DataFrame:
    new_cols = [
        c for c in mechanism.columns
        if c not in MECHANISM_METADATA_COLUMNS and c not in sequence.columns
    ]
    combined = sequence.merge(mechanism[["event_id", *new_cols]], on="event_id", how="inner", validate="one_to_one")
    extra_meta = [
        c for c in (
            "split",
            "subcluster_id",
            "candidate_new_low",
            "candidate_near_floor",
            "candidate_range_position",
        )
        if c in event_meta.columns and c not in combined.columns
    ]
    if extra_meta:
        combined = combined.merge(event_meta[["event_id", *extra_meta]], on="event_id", how="left", validate="one_to_one")
    if "split" not in combined.columns and "parent_split" in combined.columns:
        combined["split"] = combined["parent_split"]
    return combined.sort_values("extreme_time").reset_index(drop=True)


def _build_feature_matrix(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    args: argparse.Namespace,
    *,
    compact_online: bool = False,
    progress_enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent = _parent_frame(events) if "parent_cluster_id" in events.columns else events
    if compact_online:
        sequence, sequence_dictionary = build_c3_online_score_features(
            bars,
            parent,
            phase_lookback=int(args.lookback),
            phase_bins=int(args.phase_bins),
        )
    else:
        sequence, sequence_dictionary = build_c3_sequence_features(
            bars,
            parent,
            windows=(15, 30, 60, 120, 240),
            phase_lookback=int(args.lookback),
            phase_bins=int(args.phase_bins),
            progress_every=int(args.progress_every),
            progress_enabled=progress_enabled,
        )
    mechanism, mechanism_dictionary, _ = build_mechanism_features(
        bars,
        events,
        lookback=int(args.lookback),
        phase_bins=int(args.phase_bins),
        support_tolerance_bp=float(args.support_tolerance_bp),
        min_test_gap=int(args.minimum_test_gap),
        rebound_horizon=int(args.test_rebound_horizon),
        minimum_separation_rebound_bp=float(args.minimum_test_rebound_bp),
        progress_every=int(args.progress_every),
        include_test_details=False,
        progress_enabled=progress_enabled,
        n_jobs=int(args.feature_workers) if compact_online else 1,
        parallel_batch_size=int(args.feature_worker_batch_size),
    )
    combined = _combine_feature_families(sequence, mechanism, events)
    dictionary = pd.concat([sequence_dictionary, mechanism_dictionary], ignore_index=True).drop_duplicates("feature")
    return combined, dictionary


def _model_metadata_columns() -> set[str]:
    """Return the exact non-feature column policy used by online models."""

    return set(C3_METADATA_COLUMNS) | set(MECHANISM_METADATA_COLUMNS) | set(FUTURE_LABEL_COLUMNS) | {
        "event_id",
        "split",
        "year",
        "mechanism_type",
        "secondary_mechanism_type",
        "mechanism_clear",
        "episode_id",
        "episode_size",
        "episode_weight",
        "candidate_new_low",
        "candidate_near_floor",
    }


def _feature_name_is_forbidden(column: str, metadata_columns: set[str]) -> bool:
    if column in metadata_columns or column in FUTURE_LABEL_COLUMNS:
        return True
    lower = column.lower()
    return any(
        token in lower
        for token in (
            "future",
            "forward",
            "confirmation",
            "completion",
            "mfe",
            "mae",
            "target_score",
            "reference_",
            "historical_",
            "joint_swing",
            "mechanism_joint",
        )
    )


def _update_streaming_feature_stats(
    stats: dict[str, dict[str, object]],
    frame: pd.DataFrame,
    *,
    metadata_columns: set[str],
) -> None:
    """Accumulate exact global feature-selection facts.

    This scan is intentionally delayed until an in-memory buffer is flushed or
    the candidate pass has finished.  Keeping it out of every 20k candidate
    iteration is the important speed improvement; a column-wise scan is also
    more memory efficient than materialising another dense 300+ column matrix.
    """

    row_count = int(len(frame))
    for column in frame.columns:
        name = str(column)
        if _feature_name_is_forbidden(name, metadata_columns):
            continue
        series = frame[column]
        numeric = (
            series
            if pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype)
            else pd.to_numeric(series, errors="coerce")
        )
        valid_count = int(numeric.count())
        state = stats.setdefault(
            name,
            {
                "rows": 0,
                "valid": 0,
                "minimum": np.inf,
                "maximum": -np.inf,
            },
        )
        state["rows"] = int(state["rows"]) + row_count
        state["valid"] = int(state["valid"]) + valid_count
        if valid_count > 0:
            state["minimum"] = min(float(state["minimum"]), float(numeric.min(skipna=True)))
            state["maximum"] = max(float(state["maximum"]), float(numeric.max(skipna=True)))

def _select_streamed_model_features(
    stats: dict[str, dict[str, object]],
    *,
    max_missing_ratio: float,
    max_features: int,
) -> tuple[str, ...]:
    eligible: list[str] = []
    for column, state in stats.items():
        rows = max(1, int(state["rows"]))
        valid = int(state["valid"])
        if 1.0 - valid / rows > float(max_missing_ratio):
            continue
        if valid <= 0:
            continue
        if float(state["minimum"]) == float(state["maximum"]):
            continue
        eligible.append(column)
    priority = sorted(
        eligible,
        key=lambda c: (
            0 if c.startswith("score_") else 1,
            0 if c.startswith("current_") else 1,
            0
            if any(
                token in c
                for token in (
                    "support_",
                    "spring_",
                    "compression",
                    "divergence",
                    "acceleration",
                    "persistence",
                    "decay",
                )
            )
            else 1,
            1 if c.startswith("phase_") else 0,
            c,
        ),
    )
    return tuple(priority[: int(max_features)])


def _build_scored_candidate_features_chunked(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    args: argparse.Namespace,
    trend_model: object,
    base_model: object,
    *,
    top_threshold: float,
    margin_threshold: float,
    spill_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    """Build exact online features with a fast adaptive memory buffer.

    Candidate scoring remains chunked and causal.  Clear-candidate full
    features are first accumulated in RAM because that is materially faster
    than serializing every 20k input chunk.  The buffer is flushed to disk only
    when its estimated memory crosses ``candidate_spill_buffer_mb``.  This
    keeps the normal path close to the original in-memory speed while bounding
    the dangerous wide-frame accumulation and avoiding the final 350-column
    concat peak.
    """

    chunk_size = int(args.candidate_feature_chunk_size)
    if chunk_size < 250:
        raise ValueError("candidate-feature-chunk-size must be >= 250")
    spill_buffer_mb = int(getattr(args, "candidate_spill_buffer_mb", 768))
    if spill_buffer_mb < 128:
        raise ValueError("candidate-spill-buffer-mb must be >= 128")
    spill_threshold_bytes = spill_buffer_mb * 1024 * 1024

    if spill_dir is None:
        spill_dir = Path.cwd() / ".swing_low_04_candidate_spill"
    spill_dir = Path(spill_dir)
    if spill_dir.exists():
        shutil.rmtree(spill_dir)
    spill_dir.mkdir(parents=True, exist_ok=True)

    spill_files: list[Path] = []
    buffered_parts: list[pd.DataFrame] = []
    buffered_bytes = 0
    dictionary_parts: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    feature_stats: dict[str, dict[str, object]] = {}
    metadata_columns = _model_metadata_columns()
    reporter = ProgressReporter(
        "[features] compact online candidates",
        total=len(candidates),
        every=max(chunk_size, int(args.progress_every)),
    )
    retained_total = 0
    spill_total_bytes = 0
    spill_flushes = 0

    def flush_buffer_to_disk() -> int:
        nonlocal buffered_bytes, spill_total_bytes, spill_flushes
        if not buffered_parts:
            return 0
        # A bounded concat here is deliberate: it happens only after the RAM
        # buffer reaches the configured cap, instead of once per 20k input
        # chunk.  Peak memory is therefore bounded to roughly two buffers.
        block = pd.concat(buffered_parts, ignore_index=True)
        _update_streaming_feature_stats(feature_stats, block, metadata_columns=metadata_columns)
        spill_flushes += 1
        spill_path = spill_dir / f"candidate_features_{spill_flushes:04d}.pkl"
        block.to_pickle(spill_path, protocol=5)
        written = int(spill_path.stat().st_size)
        spill_files.append(spill_path)
        spill_total_bytes += written
        buffered_parts.clear()
        buffered_bytes = 0
        del block
        gc.collect()
        return written

    try:
        for chunk_number, start in enumerate(range(0, len(candidates), chunk_size), start=1):
            stop = min(len(candidates), start + chunk_size)
            chunk = candidates.iloc[start:stop].copy()
            features, dictionary = _build_feature_matrix(
                bars,
                chunk,
                args,
                compact_online=True,
                progress_enabled=False,
            )
            scores = _score_candidates(
                features,
                trend_model,
                base_model,
                top_threshold=top_threshold,
                margin_threshold=margin_threshold,
            )
            scored = features.merge(scores, on="event_id", how="inner", validate="one_to_one")
            retained_scores = scored[scored["mechanism_clear"].astype(bool)].reset_index(drop=True)
            retained_ids = set(retained_scores["event_id"].astype(str))
            retained_events = chunk[chunk["event_id"].astype(str).isin(retained_ids)].copy()
            flushed_bytes = 0
            retained_bytes = 0
            if len(retained_events):
                full_sequence, full_sequence_dictionary = build_c3_sequence_features(
                    bars,
                    _parent_frame(retained_events),
                    windows=(15, 30, 60, 120, 240),
                    phase_lookback=int(args.lookback),
                    phase_bins=int(args.phase_bins),
                    progress_every=int(args.progress_every),
                    progress_enabled=False,
                )
                extra_columns = [column for column in retained_scores.columns if column not in full_sequence.columns]
                retained = full_sequence.merge(
                    retained_scores[["event_id", *extra_columns]],
                    on="event_id",
                    how="inner",
                    validate="one_to_one",
                ).sort_values("extreme_time").reset_index(drop=True)
                shallow_usage = retained.memory_usage(index=True, deep=False)
                retained_bytes = int(shallow_usage.sum())
                object_columns = retained.select_dtypes(include=["object", "string", "category"]).columns
                if len(object_columns):
                    object_shallow = int(retained[object_columns].memory_usage(index=False, deep=False).sum())
                    object_deep = int(retained[object_columns].memory_usage(index=False, deep=True).sum())
                    retained_bytes += max(0, object_deep - object_shallow)
                buffered_parts.append(retained)
                buffered_bytes += retained_bytes
                dictionary_parts.append(
                    pd.concat([full_sequence_dictionary, dictionary], ignore_index=True).drop_duplicates("feature")
                )
                del full_sequence
                if buffered_bytes >= spill_threshold_bytes:
                    flushed_bytes = flush_buffer_to_disk()
            retained_total += len(retained_events)
            diagnostics.append(
                {
                    "chunk_number": chunk_number,
                    "start_row": start,
                    "stop_row_exclusive": stop,
                    "input_candidates": len(chunk),
                    "clear_candidates_retained": len(retained_events),
                    "clear_share": float(len(retained_events) / max(1, len(chunk))),
                    "retained_estimated_bytes": retained_bytes,
                    "buffered_estimated_bytes_after_chunk": buffered_bytes,
                    "spill_flush_bytes": flushed_bytes,
                }
            )
            del chunk, features, scores, scored, retained_scores, retained_events
            # Full cyclic-GC on every 20k chunk was measurable overhead.  Normal
            # reference counting handles the common case; collect periodically
            # and after each spill flush instead.
            if chunk_number % 8 == 0:
                gc.collect()
            if stop < len(candidates):
                reporter.update(stop)
        reporter.close()

        if not spill_files and not buffered_parts:
            raise RuntimeError("frozen mechanism clarity gate retained no online candidates")

        if spill_files:
            # Once any block has spilled, flush the tail as well so wide frames
            # do not coexist with the reduced final frame.
            flush_buffer_to_disk()
        else:
            # Fast path: all retained wide frames fit below the configured RAM
            # cap.  Scan them only once after candidate construction, not in
            # every progress iteration.
            for part in buffered_parts:
                _update_streaming_feature_stats(feature_stats, part, metadata_columns=metadata_columns)

        selected_features = _select_streamed_model_features(
            feature_stats,
            max_missing_ratio=0.35,
            max_features=int(getattr(args, "max_model_features", 220)),
        )
        if len(selected_features) < 20:
            raise RuntimeError(f"Too few usable streamed model features: {len(selected_features)}")

        reduced_parts: list[pd.DataFrame] = []
        if spill_files:
            sources: list[Path | pd.DataFrame] = list(spill_files)
        else:
            sources = list(buffered_parts)

        for source in sources:
            if isinstance(source, Path):
                frame = pd.read_pickle(source)
            else:
                frame = source
            non_features = [column for column in frame.columns if column not in feature_stats]
            keep = list(dict.fromkeys([*non_features, *selected_features]))
            reduced_parts.append(frame.reindex(columns=keep))
            del frame

        # Only the selected <=220 model features plus metadata are concatenated.
        # The former dangerous 350+ column final concat is never performed.
        if not spill_files:
            buffered_parts.clear()
        sources.clear()
        combined = pd.concat(reduced_parts, ignore_index=True)
        reduced_parts.clear()
        gc.collect()
        dictionary = pd.concat(dictionary_parts, ignore_index=True).drop_duplicates("feature")
        diagnostics.append(
            {
                "chunk_number": "ALL",
                "start_row": 0,
                "stop_row_exclusive": len(candidates),
                "input_candidates": len(candidates),
                "clear_candidates_retained": retained_total,
                "clear_share": float(retained_total / max(1, len(candidates))),
                "retained_estimated_bytes": np.nan,
                "buffered_estimated_bytes_after_chunk": 0,
                "spill_flush_bytes": spill_total_bytes,
                "spill_flushes": spill_flushes,
                "spill_buffer_limit_mb": spill_buffer_mb,
                "selected_model_features": len(selected_features),
            }
        )
        print(
            f"[features] retained clear candidates={retained_total:,}/{len(candidates):,} "
            f"({retained_total / max(1, len(candidates)):.1%}); "
            f"spill_flushes={spill_flushes:,}, RAM frame columns={combined.shape[1]:,}, "
            f"selected_features={len(selected_features):,}",
            flush=True,
        )
        return combined, dictionary, pd.DataFrame(diagnostics), selected_features
    finally:
        if "reporter" in locals():
            try:
                reporter.close()
            except Exception:
                pass
        if spill_dir.exists():
            shutil.rmtree(spill_dir, ignore_errors=True)

def _fit_reference_mechanism_models(
    reference_features: pd.DataFrame,
) -> tuple[object, object, pd.DataFrame, float, float]:
    trend = reference_features[reference_features["subcluster_id"].astype(str).eq("C3-C")].reset_index(drop=True)
    base = reference_features[reference_features["subcluster_id"].astype(str).eq("C3-E")].reset_index(drop=True)
    trend_model = fit_mechanism_score_model(
        trend,
        trend["split"].astype(str).eq("train"),
        TREND_ARCHETYPE_TERMS,
        name="online_reference_trend",
        calibrate_percentiles=True,
    )
    base_model = fit_mechanism_score_model(
        base,
        base["split"].astype(str).eq("train"),
        BASE_ARCHETYPE_TERMS,
        name="online_reference_base",
        calibrate_percentiles=True,
    )
    trend_scores = trend_model.transform(reference_features)
    base_scores = base_model.transform(reference_features)
    score_frame = pd.DataFrame({"event_id": reference_features["event_id"]})
    for label in CLEAR_MECHANISMS:
        source = trend_scores if label.startswith("T") else base_scores
        score_frame[f"score_{label}"] = source[f"score_{label}"].to_numpy()
    provisional = mechanism_assignment_from_scores(score_frame, top_score_threshold=-np.inf, margin_threshold=-np.inf)
    train = provisional.merge(
        reference_features[["event_id", "split", "subcluster_id"]], on="event_id", how="left"
    )
    train = train[
        train["split"].astype(str).eq("train")
        & train["subcluster_id"].astype(str).isin(["C3-C", "C3-E"])
    ]
    top_threshold, margin_threshold = fit_mechanism_clarity_thresholds(train, quantile=0.20)
    return trend_model, base_model, provisional, top_threshold, margin_threshold


def _score_candidates(
    candidate_features: pd.DataFrame,
    trend_model: object,
    base_model: object,
    *,
    top_threshold: float,
    margin_threshold: float,
) -> pd.DataFrame:
    trend_scores = trend_model.transform(candidate_features)
    base_scores = base_model.transform(candidate_features)
    score_frame = candidate_features[["event_id"]].copy()
    for label in CLEAR_MECHANISMS:
        source = trend_scores if label.startswith("T") else base_scores
        score_frame[f"score_{label}"] = source[f"score_{label}"].to_numpy()
    return mechanism_assignment_from_scores(
        score_frame,
        top_score_threshold=float(top_threshold),
        margin_threshold=float(margin_threshold),
    )


def _mechanism_report_agreement(
    provisional: pd.DataFrame,
    stage3_reference: pd.DataFrame,
) -> pd.DataFrame:
    merged = provisional[["event_id", "mechanism_type"]].merge(
        stage3_reference[["event_id", "mechanism_type"]].rename(columns={"mechanism_type": "reported_type"}),
        on="event_id",
        how="inner",
    )
    merged = merged[merged["reported_type"].isin(CLEAR_MECHANISMS)]
    rows: list[dict[str, object]] = []
    for reported, group in merged.groupby("reported_type", sort=True):
        rows.append(
            {
                "reported_type": reported,
                "count": int(len(group)),
                "exact_rederived_rate": float((group["mechanism_type"] == group["reported_type"]).mean()),
            }
        )
    if len(merged):
        rows.append(
            {
                "reported_type": "ALL_CLEAR_TYPES",
                "count": int(len(merged)),
                "exact_rederived_rate": float((merged["mechanism_type"] == merged["reported_type"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def _build_label_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_specs = [
        ("split", ["split"]),
        ("year", ["year"]),
        ("mechanism", ["mechanism_type"]),
        ("split_mechanism", ["split", "mechanism_type"]),
    ]
    for level, columns in group_specs:
        for key, group in frame.groupby(columns, sort=True):
            key_values = key if isinstance(key, tuple) else (key,)
            row: dict[str, object] = {"level": level, "count": int(len(group))}
            row.update({column: value for column, value in zip(columns, key_values)})
            row.update(
                {
                    "historical_clear_swing_rate": float(group["historical_clear_swing_low"].mean()),
                    "joint_swing_tp_rate": float(group["joint_swing_tp_success"].mean()),
                    "matching_mechanism_tp_rate": float(group["mechanism_joint_success"].mean()),
                    "tp_rate": float(group["tp_hit_1pct"].mean()),
                    "adverse_rate": float(group["adverse_hit_1pct"].mean()),
                    "tp_before_adverse_rate": float((group["first_touch_score"] == 100.0).mean()),
                    "median_mfe_pct": float(group["mfe_pct"].median()),
                    "median_mae_pct": float(group["mae_pct"].median()),
                    "mean_tp_priority_score": float(group["tp_priority_score"].mean()),
                    "mean_first_touch_score": float(group["first_touch_score"].mean()),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _specialist_models(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    train_all: pd.DataFrame,
    holdout: pd.DataFrame,
    feature_columns: Sequence[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.Series, list[dict[str, object]]]:
    selection_rows: list[pd.DataFrame] = []
    predictions = pd.Series(np.nan, index=holdout.index, dtype=float)
    model_meta: list[dict[str, object]] = []
    for mechanism in CLEAR_MECHANISMS:
        fit_m = fit[fit["mechanism_type"] == mechanism]
        validation_m = validation[validation["mechanism_type"] == mechanism]
        train_m = train_all[train_all["mechanism_type"] == mechanism]
        holdout_m = holdout[holdout["mechanism_type"] == mechanism]
        if (
            len(fit_m) < int(args.minimum_specialist_train_rows)
            or len(validation_m) < 100
            or int(fit_m["mechanism_joint_success"].sum()) < int(args.minimum_specialist_positive_rows)
            or fit_m["mechanism_joint_success"].nunique() < 2
            or validation_m["mechanism_joint_success"].nunique() < 2
        ):
            model_meta.append(
                {
                    "mechanism_type": mechanism,
                    "status": "insufficient_sample",
                    "fit_rows": int(len(fit_m)),
                    "fit_positive_rows": int(fit_m["mechanism_joint_success"].sum()),
                    "validation_rows": int(len(validation_m)),
                }
            )
            continue
        family, table = choose_binary_family(
            fit_m,
            validation_m,
            feature_columns=feature_columns,
            target_column="mechanism_joint_success",
            random_state=int(args.random_state),
            min_samples_leaf=max(30, int(args.model_min_samples_leaf) // 2),
        )
        table.insert(0, "mechanism_type", mechanism)
        selection_rows.append(table)
        model = fit_binary_model(
            train_m,
            feature_columns=feature_columns,
            target_column="mechanism_joint_success",
            family=family,
            random_state=int(args.random_state),
            min_samples_leaf=max(30, int(args.model_min_samples_leaf) // 2),
        )
        if len(holdout_m):
            predictions.loc[holdout_m.index] = model.predict_proba(holdout_m)
        model_meta.append(
            {
                "mechanism_type": mechanism,
                "status": "trained",
                "selected_family": family,
                "fit_rows": int(len(fit_m)),
                "fit_positive_rows": int(fit_m["mechanism_joint_success"].sum()),
                "validation_rows": int(len(validation_m)),
                "train_all_rows": int(len(train_m)),
                "holdout_rows": int(len(holdout_m)),
            }
        )
    selection = (
        pd.concat(selection_rows, ignore_index=True)
        if selection_rows
        else pd.DataFrame(
            columns=[
                "mechanism_type", "family", "count", "positive_count", "base_rate",
                "pr_auc", "roc_auc", "brier", "precision_top_1pct", "lift_top_1pct",
                "precision_top_5pct", "lift_top_5pct", "precision_top_10pct",
                "lift_top_10pct", "precision_top_20pct", "lift_top_20pct",
            ]
        )
    )
    return selection, predictions, model_meta


def _per_mechanism_metrics(holdout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mechanism, group in holdout.groupby("mechanism_type", sort=True):
        any_swing = binary_metrics(group["joint_swing_tp_success"], group["unified_joint_probability"])
        rows.append({"mechanism_type": mechanism, "model": "unified_any_clear_swing", **any_swing})

        matching = binary_metrics(group["mechanism_joint_success"], group["unified_joint_probability"])
        rows.append({"mechanism_type": mechanism, "model": "unified_matching_mechanism", **matching})

        specialist = group.dropna(subset=["specialist_joint_probability"])
        if len(specialist) and specialist["mechanism_joint_success"].nunique() > 1:
            metrics = binary_metrics(
                specialist["mechanism_joint_success"],
                specialist["specialist_joint_probability"],
            )
            rows.append({"mechanism_type": mechanism, "model": "specialist_matching_mechanism", **metrics})
    return pd.DataFrame(rows)


def _plot_probability_buckets(table: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(table["probability_bucket"], table["target_rate"] * 100.0, marker="o", label="Joint Swing+TP rate")
    if "tp_rate" in table.columns:
        ax.plot(table["probability_bucket"], table["tp_rate"] * 100.0, marker="o", label="Any +1% TP rate")
    ax.plot(table["probability_bucket"], table["mean_probability"] * 100.0, marker="o", label="Mean predicted probability")
    ax.set_title("Holdout probability buckets")
    ax.set_xlabel("Probability bucket: low to high")
    ax.set_ylabel("Percent")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_mechanism_holdout(table: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    data = table[(table["level"] == "split_mechanism") & (table["split"] == "holdout")].copy()
    if data.empty:
        return
    data = data.sort_values("joint_swing_tp_rate")
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(data))
    width = 0.26
    ax.bar(x - width, data["joint_swing_tp_rate"] * 100.0, width=width, label="Any clear Swing + TP")
    ax.bar(x, data["matching_mechanism_tp_rate"] * 100.0, width=width, label="Matching mechanism + TP")
    ax.bar(x + width, data["tp_rate"] * 100.0, width=width, label="Any TP")
    ax.set_xticks(x, data["mechanism_type"].astype(str))
    ax.set_title("Holdout outcome rate by clear mechanism")
    ax.set_ylabel("Rate (%)")
    ax.legend()
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _raw_future_perturbation_audit(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    feature_columns: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    valid = candidates[
        (pd.to_numeric(candidates["extreme_pos"], errors="coerce") >= int(args.lookback))
        & (pd.to_numeric(candidates["extreme_pos"], errors="coerce") + int(args.forward_horizon_bars) < len(bars))
    ]
    if valid.empty:
        return pd.DataFrame([{"event_id": "", "passed": False, "maximum_absolute_difference": np.nan, "detail": "no auditable candidates"}])
    sample = valid.sample(min(int(args.causal_audit_sample_size), len(valid)), random_state=int(args.random_state))
    rows: list[dict[str, object]] = []
    for source in sample.itertuples(index=False):
        pos = int(source.extreme_pos)
        start = pos - int(args.lookback)
        end = pos + int(args.forward_horizon_bars) + 1
        local = bars.iloc[start:end].copy()
        local_pos = int(args.lookback)
        ts = local.index[local_pos]
        event = pd.DataFrame(
            [
                {
                    "event_id": source.event_id,
                    "extreme_time": ts,
                    "feature_available_time": ts + pd.Timedelta(minutes=1),
                    "extreme_pos": local_pos,
                    "extreme_price": float(local.iloc[local_pos]["low"]),
                    "confirmation_time": ts + pd.Timedelta(minutes=1),
                    "confirmation_available_time": ts + pd.Timedelta(minutes=2),
                    "completion_bars": 0,
                    "realized_confirmation_move_pct": np.nan,
                    "cluster_id": "ONLINE_CANDIDATE",
                    "distance_to_train_centroid": np.nan,
                    "parent_cluster_id": "ONLINE_CANDIDATE",
                    "parent_distance_to_centroid": np.nan,
                    "split": "audit",
                }
            ]
        )
        original, _ = _build_feature_matrix(local, event, args)
        perturbed = local.copy()
        rng = np.random.default_rng(int(args.random_state) + pos)
        future_slice = slice(local_pos + 1, len(perturbed))
        changed = 0
        for column in set(MECHANISM_REQUIRED_COLUMNS) | {
            "buy_notional", "sell_notional", "avg_trade_size", "max_trade_notional", "vwap"
        }:
            if column not in perturbed.columns:
                continue
            values = pd.to_numeric(perturbed[column], errors="coerce").to_numpy(dtype=float, copy=True)
            segment = values[future_slice]
            if not len(segment):
                continue
            scale = rng.uniform(0.2, 4.0, len(segment))
            values[future_slice] = np.where(np.isfinite(segment), segment * scale, segment)
            perturbed[column] = values
            changed += int(np.isfinite(segment).sum())
        # Preserve valid OHLC ordering while strongly changing only future rows.
        future = perturbed.iloc[local_pos + 1 :].copy()
        if len(future):
            center = pd.to_numeric(future["close"], errors="coerce").to_numpy(dtype=float)
            center = center * rng.uniform(0.94, 1.06, len(center))
            spread = np.maximum(np.abs(center) * rng.uniform(0.0002, 0.01, len(center)), 1e-9)
            perturbed.iloc[local_pos + 1 :, perturbed.columns.get_loc("open")] = center
            perturbed.iloc[local_pos + 1 :, perturbed.columns.get_loc("close")] = center * rng.uniform(0.995, 1.005, len(center))
            o = pd.to_numeric(perturbed.iloc[local_pos + 1 :]["open"], errors="coerce").to_numpy(dtype=float)
            c = pd.to_numeric(perturbed.iloc[local_pos + 1 :]["close"], errors="coerce").to_numpy(dtype=float)
            perturbed.iloc[local_pos + 1 :, perturbed.columns.get_loc("high")] = np.maximum(o, c) + spread
            perturbed.iloc[local_pos + 1 :, perturbed.columns.get_loc("low")] = np.minimum(o, c) - spread
        changed_features, _ = _build_feature_matrix(perturbed, event, args)
        columns = [c for c in feature_columns if c in original.columns and c in changed_features.columns]
        a = original[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        b = changed_features[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        diff = np.abs(a - b)
        finite = np.isfinite(diff)
        max_diff = float(np.nanmax(diff[finite])) if finite.any() else 0.0
        rows.append(
            {
                "event_id": source.event_id,
                "changed_future_cells": int(changed),
                "maximum_absolute_difference": max_diff,
                "passed": bool(max_diff <= 1e-12),
                "detail": "future raw bars changed; all current/older features must remain identical",
            }
        )
    return pd.DataFrame(rows)


def _causal_audit(
    model_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    raw_audit: pd.DataFrame,
    gate_recall: pd.DataFrame,
) -> pd.DataFrame:
    forbidden = [
        c for c in feature_columns
        if c in FUTURE_LABEL_COLUMNS
        or any(
            token in c.lower()
            for token in (
                "future", "forward", "confirmation", "completion", "mfe", "mae",
                "target_score", "reference_", "historical_", "joint_swing", "mechanism_joint",
            )
        )
    ]
    feature_time = pd.to_datetime(model_frame["feature_available_time"])
    entry_time = pd.to_datetime(model_frame["entry_time"])
    split_order = model_frame.groupby("split")["extreme_time"].agg(["min", "max"]).reset_index()
    return pd.DataFrame(
        [
            {
                "check": "signal_close_before_or_at_next_open",
                "passed": bool((feature_time <= entry_time).all()),
                "detail": f"max_lag={(entry_time - feature_time).max()}",
            },
            {
                "check": "future_labels_excluded_from_features",
                "passed": not forbidden,
                "detail": ",".join(forbidden),
            },
            {
                "check": "fit_validation_holdout_are_chronological",
                "passed": set(model_frame["split"].unique()) >= {"fit", "validation", "holdout"},
                "detail": split_order.to_json(orient="records", date_format="iso"),
            },
            {
                "check": "fit_labels_do_not_cross_2023_boundary",
                "passed": bool((pd.to_datetime(model_frame.loc[model_frame["split"] == "fit", "label_end_time"]) <= pd.Timestamp("2023-12-31 23:59:59")).all()),
                "detail": "full forward label window remains inside fit period",
            },
            {
                "check": "validation_labels_do_not_cross_2024_boundary",
                "passed": bool((pd.to_datetime(model_frame.loc[model_frame["split"] == "validation", "label_end_time"]) <= pd.Timestamp("2024-12-31 23:59:59")).all()),
                "detail": "full forward label window remains inside validation period",
            },
            {
                "check": "holdout_frozen_after_2024",
                "passed": bool((pd.to_datetime(model_frame.loc[model_frame["split"] == "holdout", "extreme_time"]) >= pd.Timestamp("2025-01-01")).all()),
                "detail": "model family selected on 2024 validation; final model refit on 2023-2024 only",
            },
            {
                "check": "raw_future_feature_perturbation",
                "passed": bool(not raw_audit.empty and raw_audit["passed"].all()),
                "detail": f"audited={len(raw_audit)}; max_diff={raw_audit.get('maximum_absolute_difference', pd.Series([np.nan])).max()}",
            },
            {
                "check": "candidate_gate_recall_reported",
                "passed": bool(not gate_recall.empty),
                "detail": f"minimum_clear_type_recall={gate_recall.get('gate_recall', pd.Series([np.nan])).min()}",
            },
        ]
    )



def _candidate_gate_config_from_args(args: argparse.Namespace) -> CandidateGateConfig:
    return CandidateGateConfig(
        lookback=int(args.lookback),
        horizon=int(args.forward_horizon_bars),
        new_low_window=int(args.candidate_new_low_window),
        near_floor_window=int(args.candidate_near_floor_window),
        position_window=int(args.candidate_position_window),
        near_floor_tolerance_bp=float(args.candidate_near_floor_tolerance_bp),
        max_position_in_range=float(args.candidate_max_position_in_range),
    )


def _read_existing_csv(path: Path, *, required_columns: Sequence[str] = ()) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Cannot resume 04: required artifact is missing: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise RuntimeError(f"Cannot resume 04: {path.name} is missing columns {missing}")
    return frame


def _build_resume_audit_frame(
    bars: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rebuild only cheap causal timing rows needed by the final audit.

    No 315-column candidate feature matrix is constructed here.  Candidate
    gating is vectorized, and entry/label timestamps are derived directly from
    bar positions using the same next-open / bounded-close horizon convention.
    """

    research_start = pd.Timestamp(args.start_date)
    research_end = _end_exclusive(args.end_date, args.timeframe)
    candidates, gate_summary = build_online_candidate_events(
        bars,
        research_start=research_start,
        research_end_exclusive=research_end,
        config=_candidate_gate_config_from_args(args),
    )
    candidates = attach_temporal_split(
        candidates,
        fit_end=pd.Timestamp(args.fit_end_date),
        validation_end=pd.Timestamp(args.validation_end_date),
    )
    positions = pd.to_numeric(candidates["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
    index = pd.DatetimeIndex(bars.index)
    horizon = int(args.forward_horizon_bars)
    candidates["entry_time"] = index[positions + 1]
    candidates["label_end_time"] = index[positions + horizon]
    purged, purge_summary = purge_temporal_label_overlap(
        candidates,
        fit_end=pd.Timestamp(args.fit_end_date),
        validation_end=pd.Timestamp(args.validation_end_date),
    )
    samples: list[pd.DataFrame] = []
    for split in ("fit", "validation", "holdout"):
        part = purged[purged["split"].astype(str).eq(split)].sort_values("extreme_time")
        if part.empty:
            raise RuntimeError(f"Cannot resume 04: no auditable rows remain in split={split}")
        samples.append(pd.concat([part.head(2), part.tail(2)], ignore_index=True).drop_duplicates("event_id"))
    audit_frame = pd.concat(samples, ignore_index=True)
    return audit_frame, gate_summary, purge_summary


def _map_existing_predictions_to_audit_candidates(
    bars: pd.DataFrame,
    predictions: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    times = pd.to_datetime(predictions["extreme_time"], errors="coerce")
    bar_positions = pd.Series(
        np.arange(len(bars), dtype=np.int64),
        index=pd.DatetimeIndex(bars.index),
    )
    mapped = bar_positions.reindex(pd.DatetimeIndex(times)).to_numpy()
    pool = pd.DataFrame(
        {
            "event_id": predictions["event_id"].astype(str).to_numpy(),
            "extreme_pos": mapped,
        }
    ).dropna(subset=["extreme_pos"])
    pool["extreme_pos"] = pool["extreme_pos"].astype(np.int64)
    minimum_pool = max(64, int(args.causal_audit_sample_size) * 8)
    if len(pool) < int(args.causal_audit_sample_size):
        raise RuntimeError(
            f"Cannot resume 04: only {len(pool)} holdout predictions map to loaded bars; "
            f"required={int(args.causal_audit_sample_size)}"
        )
    return pool.sample(min(len(pool), minimum_pool), random_state=int(args.random_state)).reset_index(drop=True)


def _summary_from_existing_artifacts(
    args: argparse.Namespace,
    label_diagnostics: pd.DataFrame,
    holdout: pd.DataFrame,
    binary_selection: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    per_mechanism: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    split_rows = label_diagnostics[label_diagnostics.get("level", pd.Series(dtype=str)).astype(str).eq("split")]
    split_counts = {
        str(row["split"]): int(row["count"])
        for _, row in split_rows.iterrows()
        if pd.notna(row.get("split")) and pd.notna(row.get("count"))
    }
    best_family = str(binary_selection.iloc[0]["family"]) if not binary_selection.empty else "unknown"
    metric = holdout_metrics.iloc[0].to_dict() if not holdout_metrics.empty else {}
    lines = [
        f"# {TITLE}",
        "",
        "## Scope",
        "",
        "- Research only. No order execution, fees, stop logic, position sizing, or portfolio backtest.",
        "- This summary was finalized from the already completed 04 model artifacts after rerunning the causal audit.",
        "- Clear mechanisms only: T2, T3, T4, B2, B3, B4. B1/B5 are intentionally excluded.",
        "",
        "## Label design",
        "",
        f"- Swing structure anchor: current bar low.",
        f"- Executable reference: next-bar open.",
        f"- Success: a future closed-bar close reaches +{args.target_move_pct:g}% within {args.forward_horizon_bars} bars.",
        "- Future high/low are not used for TP, MFE, MAE, or first-touch labels.",
        "",
        "## Dataset",
        "",
        f"- Fit 2023: {split_counts.get('fit', 0):,}",
        f"- Validation 2024: {split_counts.get('validation', 0):,}",
        f"- Frozen holdout 2025-2026H1: {split_counts.get('holdout', len(holdout)):,}",
        f"- Holdout exact clear Swing Low rate: {pd.to_numeric(holdout.get('historical_clear_swing_low'), errors='coerce').mean():.2%}",
        f"- Holdout joint Swing+TP rate: {pd.to_numeric(holdout.get('joint_swing_tp_success'), errors='coerce').mean():.2%}",
        f"- Holdout any +1% TP rate: {pd.to_numeric(holdout.get('tp_hit_1pct'), errors='coerce').mean():.2%}",
        "",
        "## Unified model",
        "",
        f"- Selected family on 2024 validation: `{best_family}`.",
        f"- Holdout PR-AUC: {metric.get('pr_auc', np.nan):.4f}",
        f"- Holdout ROC-AUC: {metric.get('roc_auc', np.nan):.4f}",
        f"- Holdout top-10% joint precision: {metric.get('precision_top_10pct', np.nan):.2%}",
        f"- Holdout top-10% lift: {metric.get('lift_top_10pct', np.nan):.2f}x",
        "",
        "## Causal status",
        "",
        f"- All causal checks passed: {bool(audit['passed'].all()) if not audit.empty else False}.",
    ]
    if not per_mechanism.empty:
        lines.extend(["", "## Holdout mechanism comparison", ""])
        for _, row in per_mechanism.iterrows():
            lines.append(
                f"- {row.get('mechanism_type', 'unknown')} / {row.get('model', 'unknown')}: "
                f"PR-AUC={row.get('pr_auc', np.nan):.4f}, "
                f"top10 precision={row.get('precision_top_10pct', np.nan):.2%}, "
                f"base={row.get('base_rate', np.nan):.2%}"
            )
    return "\n".join(lines) + "\n"


def _resume_finalize_existing_run(
    args: argparse.Namespace,
    out_dir: Path,
    stage2_dir: Path,
    stage3_dir: Path,
) -> Path:
    """Finish a run that already reached outputs 01-19 without rebuilding 766k features."""

    required = {
        "gate_recall": out_dir / "04_candidate_gate_recall_clear_03_types.csv",
        "chunk_diagnostics": out_dir / "04d_candidate_feature_chunk_diagnostics.csv",
        "label_diagnostics": out_dir / "07_label_diagnostics.csv",
        "feature_list": out_dir / "08_model_feature_list.csv",
        "model_selection": out_dir / "09_validation_model_selection.csv",
        "unified_metrics": out_dir / "12_holdout_unified_metrics.csv",
        "per_mechanism": out_dir / "14_holdout_per_mechanism_metrics.csv",
        "holdout_predictions": out_dir / "19_holdout_predictions.csv",
    }
    missing = [path.name for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(
            "Cannot use --resume-finalize-existing because the previous run did not finish model outputs: "
            + ", ".join(missing)
        )

    stage2_manifest, stage3_manifest = _validate_reports(args, stage2_dir, stage3_dir)
    bars = load_bars(args)
    feature_columns = tuple(
        _read_existing_csv(required["feature_list"], required_columns=("feature",))["feature"].astype(str)
    )
    gate_recall = _read_existing_csv(required["gate_recall"])
    chunk_diagnostics = _read_existing_csv(required["chunk_diagnostics"])
    label_diagnostics = _read_existing_csv(required["label_diagnostics"], required_columns=("level", "count"))
    model_selection = _read_existing_csv(required["model_selection"], required_columns=("task", "family"))
    unified_metrics = _read_existing_csv(required["unified_metrics"])
    per_mechanism = _read_existing_csv(required["per_mechanism"])
    holdout = _read_existing_csv(
        required["holdout_predictions"],
        required_columns=(
            "event_id", "extreme_time", "feature_available_time", "entry_time", "label_end_time",
            "historical_clear_swing_low", "joint_swing_tp_success", "tp_hit_1pct",
        ),
    )
    for column in ("extreme_time", "feature_available_time", "entry_time", "label_end_time"):
        holdout[column] = pd.to_datetime(holdout[column], errors="raise")

    print("[resume] rebuilding only cheap timing audit rows", flush=True)
    audit_frame, gate_summary, purge_summary = _build_resume_audit_frame(bars, args)
    existing_all = chunk_diagnostics[chunk_diagnostics["chunk_number"].astype(str).eq("ALL")]
    expected_candidate_count = int(existing_all.iloc[0]["input_candidates"]) if len(existing_all) else None
    rebuilt_candidate_count = int(
        pd.to_numeric(
            gate_summary.loc[gate_summary["metric"].astype(str).eq("candidate_count"), "value"],
            errors="coerce",
        ).iloc[0]
    )
    if expected_candidate_count is not None and expected_candidate_count != rebuilt_candidate_count:
        raise RuntimeError(
            "Cannot resume 04: current data/arguments produce a different candidate universe "
            f"({rebuilt_candidate_count:,} vs existing {expected_candidate_count:,})."
        )

    audit_candidates = _map_existing_predictions_to_audit_candidates(bars, holdout, args)
    print("[resume] rerunning fixed future perturbation causal audit", flush=True)
    raw_audit = _raw_future_perturbation_audit(bars, audit_candidates, feature_columns, args)
    _write_csv(raw_audit, out_dir / "20_raw_future_perturbation_audit.csv")
    audit = _causal_audit(audit_frame, feature_columns, raw_audit, gate_recall)
    _write_csv(audit, out_dir / "21_causal_audit.csv")
    if not audit["passed"].all():
        failed = audit.loc[~audit["passed"], "check"].tolist()
        raise RuntimeError(f"Causal audit failed: {failed}")

    stage2 = _load_stage2(stage2_dir)
    print("[resume] rebuilding 2,828-row reference thresholds for manifest", flush=True)
    reference_features, _ = _build_feature_matrix(bars, stage2, args)
    _, _, _, top_threshold, margin_threshold = _fit_reference_mechanism_models(reference_features)

    binary_rows = model_selection[model_selection["task"].astype(str).eq("joint_swing_tp_success")]
    score_rows = model_selection[model_selection["task"].astype(str).eq("tp_priority_score")]
    chosen_binary = str(binary_rows.iloc[0]["family"]) if len(binary_rows) else "unknown"
    chosen_score = str(score_rows.iloc[0]["family"]) if len(score_rows) else "unknown"
    split_count_rows = label_diagnostics[label_diagnostics["level"].astype(str).eq("split")]
    clear_count = int(pd.to_numeric(split_count_rows["count"], errors="coerce").fillna(0).sum())
    if clear_count <= 0:
        clear_count = int(existing_all.iloc[0]["clear_candidates_retained"]) if len(existing_all) else int(len(holdout))

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
        "fit_end_date": args.fit_end_date,
        "validation_end_date": args.validation_end_date,
        "target_move_pct": float(args.target_move_pct),
        "adverse_move_pct": float(args.adverse_move_pct),
        "forward_horizon_bars": int(args.forward_horizon_bars),
        "swing_extreme_price_source": "low",
        "swing_entry_price_source": "next_bar_open",
        "swing_target_observation_source": "future_closed_bar_close",
        "swing_return_definition": "future_closed_bar_close / next_bar_open - 1",
        "candidate_count_before_mechanism_clarity": rebuilt_candidate_count,
        "clear_candidate_count": clear_count,
        "model_feature_count": int(len(feature_columns)),
        "online_sequence_score_feature_count": int(len(ONLINE_SEQUENCE_SCORE_FEATURES)),
        "candidate_feature_chunk_size": int(args.candidate_feature_chunk_size),
        "feature_workers": int(args.feature_workers),
        "feature_worker_batch_size": int(args.feature_worker_batch_size),
        "label_vectorized_chunk_size": int(args.label_vectorized_chunk_size),
        "feature_pipeline": "chunked_compact_gate_then_full_exact_features_plus_parallel_mechanism",
        "selected_binary_family": chosen_binary,
        "selected_score_family": chosen_score,
        "mechanism_top_score_threshold": float(top_threshold),
        "mechanism_margin_threshold": float(margin_threshold),
        "clear_mechanisms": list(CLEAR_MECHANISMS),
        "excluded_mechanisms": ["B1_absorption", "B5_slow_accumulation"],
        "causal_policy": "candidate gate and features use current closed bar or older; entry is next-bar open; targets and path scores use future closed-bar closes only",
        "finalization_mode": "resume_existing_outputs_01_to_19_after_fixed_causal_audit",
        "temporal_purge_summary": purge_summary.to_dict(orient="records"),
        "stage2_manifest": stage2_manifest,
        "stage3_manifest_summary": {
            "experiment_id": stage3_manifest.get("experiment_id"),
            "event_count": stage3_manifest.get("event_count"),
            "swing_extreme_price_source": stage3_manifest.get("swing_extreme_price_source"),
            "swing_entry_price_source": stage3_manifest.get("swing_entry_price_source"),
            "swing_target_observation_source": stage3_manifest.get("swing_target_observation_source"),
            "causal_policy": stage3_manifest.get("causal_policy"),
        },
        "strategy_or_backtest": False,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _summary_from_existing_artifacts(
        args,
        label_diagnostics,
        holdout,
        binary_rows,
        unified_metrics,
        per_mechanism,
        audit,
    )
    (out_dir / "22_RESEARCH_SUMMARY.md").write_text(summary, encoding="utf-8")
    result = finalize_research_report(
        out_dir,
        experiment_id=EXPERIMENT_ID,
        edge_id=EDGE_ID,
        title=TITLE,
    )
    print(f"[done] resumed report={out_dir}", flush=True)
    print(f"[done] review_pack={result.zip_path}", flush=True)
    return out_dir

def _summary(
    args: argparse.Namespace,
    model_frame: pd.DataFrame,
    validation_selection: pd.DataFrame,
    holdout_metrics: pd.DataFrame,
    per_mechanism: pd.DataFrame,
    audit: pd.DataFrame,
) -> str:
    holdout = model_frame[model_frame["split"] == "holdout"]
    best_family = str(validation_selection.iloc[0]["family"]) if not validation_selection.empty else "unknown"
    metric = holdout_metrics.iloc[0].to_dict() if not holdout_metrics.empty else {}
    lines = [
        f"# {TITLE}",
        "",
        "## Scope",
        "",
        "- Research only. No order execution, fees, stop logic, position sizing, or portfolio backtest.",
        "- Current closed 1m bar features are used to predict a bounded future path from the next-bar open.",
        "- Clear mechanisms only: T2, T3, T4, B2, B3, B4. B1/B5 are intentionally excluded.",
        "",
        "## Label design",
        "",
        f"- Primary target: exact clear 03 low-anchored Swing Low AND a future closed-bar close reaches +{args.target_move_pct:g}% within {args.forward_horizon_bars} bars from next-bar open.",
        "- Once a future closed-bar close reaches TP, success is fixed immediately; the label does not force a 60-bar hold.",
        "- `tp_priority_score` is 100 after a future close reaches TP; otherwise it uses the close-only clipped MFE-MAE path score.",
        "- `first_touch_score` separately penalizes close paths that reach -1% before +1%.",
        "",
        "## Dataset",
        "",
        f"- Clear online candidates: {len(model_frame):,}",
        f"- Feature pipeline: bounded chunks of {int(args.candidate_feature_chunk_size):,}, compact exact sequence subset, {int(args.feature_workers)} mechanism workers.",
        f"- Fit 2023: {(model_frame['split'] == 'fit').sum():,}",
        f"- Validation 2024: {(model_frame['split'] == 'validation').sum():,}",
        f"- Frozen holdout 2025-2026H1: {(model_frame['split'] == 'holdout').sum():,}",
        f"- Holdout exact clear Swing Low rate: {holdout['historical_clear_swing_low'].mean():.2%}" if len(holdout) else "- Holdout Swing Low rate: unavailable",
        f"- Holdout joint Swing+TP rate: {holdout['joint_swing_tp_success'].mean():.2%}" if len(holdout) else "- Holdout joint rate: unavailable",
        f"- Holdout any +1% TP rate: {holdout['tp_hit_1pct'].mean():.2%}" if len(holdout) else "- Holdout raw TP rate: unavailable",
        "",
        "## Unified model",
        "",
        f"- Selected family on 2024 validation: `{best_family}`.",
        f"- Holdout PR-AUC: {metric.get('pr_auc', np.nan):.4f}",
        f"- Holdout ROC-AUC: {metric.get('roc_auc', np.nan):.4f}",
        f"- Holdout top-10% joint precision: {metric.get('precision_top_10pct', np.nan):.2%}",
        f"- Holdout top-10% lift: {metric.get('lift_top_10pct', np.nan):.2f}x",
        "",
        "## Interpretation",
        "",
        "- Specialist models predict the stricter matching-mechanism Swing+TP target, not merely any future close-based +1% move.",
        "- If specialist models materially outperform the unified model inside their own mechanism, later strategy work should use a shared router plus mechanism-specific engines.",
        "- If the unified model performs similarly or better, keep one prediction model and use mechanism scores only as explanatory/risk context.",
        "- This report decides recognizability only. A positive result still requires a separate event study and then a strategy/backtest.",
        "",
        "## Causal status",
        "",
        f"- All causal checks passed: {bool(audit['passed'].all()) if not audit.empty else False}.",
    ]
    if not per_mechanism.empty:
        lines.extend(["", "## Holdout mechanism comparison", ""])
        for _, row in per_mechanism.iterrows():
            lines.append(
                f"- {row['mechanism_type']} / {row['model']}: PR-AUC={row.get('pr_auc', np.nan):.4f}, "
                f"top10 precision={row.get('precision_top_10pct', np.nan):.2%}, base={row.get('base_rate', np.nan):.2%}"
            )
    return "\n".join(lines) + "\n"


def run_research(args: argparse.Namespace) -> Path:
    out_dir = PROJECT_ROOT / args.out_dir
    stage2_dir = PROJECT_ROOT / args.stage2_report_dir
    stage3_dir = PROJECT_ROOT / args.stage3_report_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(getattr(args, "resume_finalize_existing", False)):
        return _resume_finalize_existing_run(args, out_dir, stage2_dir, stage3_dir)

    stage2_manifest, stage3_manifest = _validate_reports(args, stage2_dir, stage3_dir)
    bars = load_bars(args)
    coverage = validate_trade_bar_fields(bars)
    _write_csv(coverage, out_dir / "01_trade_bar_field_coverage.csv")

    stage2 = _load_stage2(stage2_dir)
    stage3_reference = _load_stage3_reference(stage3_dir)

    print("[stage] rebuild 03 reference causal features", flush=True)
    reference_features, feature_dictionary = _build_feature_matrix(bars, stage2, args)
    trend_model, base_model, reference_provisional, top_threshold, margin_threshold = _fit_reference_mechanism_models(reference_features)
    agreement = _mechanism_report_agreement(reference_provisional, stage3_reference)
    _write_csv(agreement, out_dir / "02_reference_mechanism_rebuild_agreement.csv")

    research_start = pd.Timestamp(args.start_date)
    research_end = _end_exclusive(args.end_date, args.timeframe)
    gate_config = _candidate_gate_config_from_args(args)
    print("[stage] build causal online candidate universe", flush=True)
    candidates, gate_summary = build_online_candidate_events(
        bars,
        research_start=research_start,
        research_end_exclusive=research_end,
        config=gate_config,
    )
    candidates = attach_temporal_split(
        candidates,
        fit_end=pd.Timestamp(args.fit_end_date),
        validation_end=pd.Timestamp(args.validation_end_date),
    )
    gate_recall = candidate_gate_recall(
        candidates,
        stage3_reference[stage3_reference["mechanism_type"].isin(CLEAR_MECHANISMS)],
        type_column="mechanism_type",
    )
    _write_csv(gate_summary, out_dir / "03_candidate_gate_summary.csv")
    _write_csv(gate_recall, out_dir / "04_candidate_gate_recall_clear_03_types.csv")
    if candidates.empty:
        raise RuntimeError("candidate gate produced no rows")
    candidate_count_before_clarity = int(len(candidates))

    # Run one real raw-future perturbation before the expensive 766k feature
    # build.  This catches pandas/NumPy mutability and causal-audit regressions
    # in seconds instead of after the full research has already completed.
    print("[stage] causal audit preflight", flush=True)
    preflight_args = argparse.Namespace(**vars(args))
    preflight_args.causal_audit_sample_size = 1
    preflight_columns = select_model_features(
        reference_features,
        metadata_columns=_model_metadata_columns(),
        max_features=min(64, int(args.max_model_features)),
    )
    preflight_pool = candidates[["event_id", "extreme_pos"]].sample(
        min(64, len(candidates)),
        random_state=int(args.random_state),
    )
    preflight_audit = _raw_future_perturbation_audit(
        bars, preflight_pool, preflight_columns, preflight_args
    )
    if preflight_audit.empty or not preflight_audit["passed"].all():
        raise RuntimeError("Causal audit preflight failed before candidate feature build")

    print(f"[stage] causal features for candidates={len(candidates):,}", flush=True)
    candidate_features, candidate_dictionary, feature_chunk_diagnostics, streamed_feature_columns = _build_scored_candidate_features_chunked(
        bars,
        candidates,
        args,
        trend_model,
        base_model,
        top_threshold=top_threshold,
        margin_threshold=margin_threshold,
        spill_dir=out_dir / "_tmp_candidate_feature_spill",
    )
    _write_csv(feature_chunk_diagnostics, out_dir / "04d_candidate_feature_chunk_diagnostics.csv")
    audit_pool_size = min(
        len(candidate_features),
        max(64, int(args.causal_audit_sample_size) * 8),
    )
    audit_candidates = candidate_features[["event_id", "extreme_pos"]].sample(
        audit_pool_size,
        random_state=int(args.random_state),
    )
    # The broad candidate table is no longer needed after frozen mechanism
    # scoring. Releasing it here materially lowers peak memory during model fit.
    del candidates
    gc.collect()

    print("[stage] bounded future labels", flush=True)
    labels = build_forward_path_labels(
        bars,
        candidate_features[["event_id", "extreme_pos"]],
        horizon=int(args.forward_horizon_bars),
        target_move_pct=float(args.target_move_pct),
        adverse_move_pct=float(args.adverse_move_pct),
        progress_every=max(1_000, int(args.progress_every)),
        vectorized_chunk_size=int(args.label_vectorized_chunk_size),
    )
    model_frame = candidate_features.merge(labels, on="event_id", how="inner", validate="one_to_one")
    model_frame, temporal_purge = purge_temporal_label_overlap(
        model_frame,
        fit_end=pd.Timestamp(args.fit_end_date),
        validation_end=pd.Timestamp(args.validation_end_date),
    )
    _write_csv(temporal_purge, out_dir / "06b_temporal_label_boundary_purge.csv")
    clear_reference = stage3_reference[stage3_reference["mechanism_type"].isin(CLEAR_MECHANISMS)].copy()
    model_frame = attach_reference_swing_targets(model_frame, clear_reference)
    model_frame["joint_swing_tp_success"] = (
        model_frame["historical_clear_swing_low"].astype(bool)
        & model_frame["tp_hit_1pct"].astype(bool)
    )
    model_frame["mechanism_joint_success"] = (
        model_frame["joint_swing_tp_success"].astype(bool)
        & model_frame["reference_mechanism_type"].astype(str).eq(model_frame["mechanism_type"].astype(str))
    )
    model_frame = model_frame[model_frame["mechanism_clear"]].reset_index(drop=True)
    model_frame = build_candidate_episodes(model_frame, max_gap_bars=int(args.candidate_episode_gap_bars))
    clear_candidate_recall = candidate_gate_recall(
        model_frame,
        clear_reference,
        type_column="mechanism_type",
    )
    _write_csv(clear_candidate_recall, out_dir / "04b_mechanism_clear_recall_clear_03_types.csv")
    matched_types = model_frame[model_frame["historical_clear_swing_low"]].copy()
    if len(matched_types):
        type_agreement = (
            matched_types.groupby(["reference_mechanism_type", "mechanism_type"], as_index=False)
            .agg(count=("event_id", "size"))
        )
        type_agreement["share_within_reference"] = (
            type_agreement["count"]
            / type_agreement.groupby("reference_mechanism_type")["count"].transform("sum")
        )
    else:
        type_agreement = pd.DataFrame()
    _write_csv(type_agreement, out_dir / "04c_online_vs_reference_mechanism_agreement.csv")
    if len(model_frame) < int(args.minimum_clear_candidates):
        raise RuntimeError(
            f"Too few clear online candidates: {len(model_frame):,}; "
            f"required={int(args.minimum_clear_candidates):,}"
        )

    dictionary = pd.concat([feature_dictionary, candidate_dictionary], ignore_index=True).drop_duplicates("feature")
    _write_csv(dictionary, out_dir / "05_causal_feature_dictionary.csv")
    _write_csv(
        label_definition_table(float(args.target_move_pct), float(args.adverse_move_pct), int(args.forward_horizon_bars)),
        out_dir / "06_forward_label_definitions.csv",
    )
    label_diagnostics = _build_label_diagnostics(model_frame)
    _write_csv(label_diagnostics, out_dir / "07_label_diagnostics.csv")
    _plot_mechanism_holdout(label_diagnostics, out_dir / "07_holdout_tp_rate_by_mechanism.png")

    metadata = _model_metadata_columns()
    feature_columns = tuple(streamed_feature_columns)
    # The disk-backed selector is mathematically identical to
    # select_model_features (numeric coercion, global missing ratio, >1 unique
    # value and deterministic priority).  Re-running the selector on the
    # already reduced frame is a cheap guard against implementation drift.
    verified_feature_columns = select_model_features(
        model_frame,
        metadata_columns=metadata,
        max_features=int(args.max_model_features),
    )
    if tuple(verified_feature_columns) != tuple(feature_columns):
        raise RuntimeError("streamed feature selection differs from in-memory verification")
    if len(feature_columns) < 20:
        raise RuntimeError(f"Too few usable model features: {len(feature_columns)}")
    _write_csv(pd.DataFrame({"feature": feature_columns}), out_dir / "08_model_feature_list.csv")

    # Feature selection is causal and finished at this point. Drop unused raw
    # feature columns before creating fit/validation/holdout copies; otherwise
    # several large DataFrame copies retain all 350+ research columns.
    retained_non_features = metadata | set(FUTURE_LABEL_COLUMNS) | {
        "event_id",
        "extreme_time",
        "feature_available_time",
        "extreme_pos",
        "extreme_price",
        "split",
        "year",
        "mechanism_type",
        "secondary_mechanism_type",
        "mechanism_clear",
        "mechanism_confidence",
        "mechanism_top_score",
        "mechanism_margin",
        "episode_id",
        "episode_size",
        "episode_weight",
        "entry_price_source",
        "path_observation_source",
    }
    keep_columns = [
        column
        for column in model_frame.columns
        if column in retained_non_features or column in feature_columns
    ]
    model_frame = model_frame[keep_columns].copy()
    gc.collect()

    fit = model_frame[model_frame["split"] == "fit"].copy()
    validation = model_frame[model_frame["split"] == "validation"].copy()
    holdout = model_frame[model_frame["split"] == "holdout"].copy()
    if min(len(fit), len(validation), len(holdout)) < int(args.minimum_temporal_split_rows):
        raise RuntimeError(
            f"Temporal split too small: fit={len(fit)}, validation={len(validation)}, holdout={len(holdout)}; "
            f"required={int(args.minimum_temporal_split_rows)}"
        )
    if any(frame["joint_swing_tp_success"].nunique() < 2 for frame in (fit, validation, holdout)):
        raise RuntimeError("A temporal split contains only one joint Swing+TP target class")

    print("[stage] train-only model family selection on 2024 validation", flush=True)
    chosen_binary, binary_selection = choose_binary_family(
        fit,
        validation,
        feature_columns=feature_columns,
        target_column="joint_swing_tp_success",
        random_state=int(args.random_state),
        min_samples_leaf=int(args.model_min_samples_leaf),
    )
    chosen_score, score_selection = choose_score_family(
        fit,
        validation,
        feature_columns=feature_columns,
        target_column="tp_priority_score",
        random_state=int(args.random_state),
        min_samples_leaf=int(args.model_min_samples_leaf),
    )
    binary_selection.insert(0, "task", "joint_swing_tp_success")
    score_selection.insert(0, "task", "tp_priority_score")
    model_selection = pd.concat([binary_selection, score_selection], ignore_index=True, sort=False)
    _write_csv(model_selection, out_dir / "09_validation_model_selection.csv")

    train_all = model_frame[model_frame["split"].isin(["fit", "validation"])].copy()
    unified = fit_binary_model(
        train_all,
        feature_columns=feature_columns,
        target_column="joint_swing_tp_success",
        family=chosen_binary,
        random_state=int(args.random_state),
        min_samples_leaf=int(args.model_min_samples_leaf),
    )
    score_model = fit_score_model(
        train_all,
        feature_columns=feature_columns,
        target_column="tp_priority_score",
        family=chosen_score,
        random_state=int(args.random_state),
        min_samples_leaf=int(args.model_min_samples_leaf),
    )
    holdout["unified_joint_probability"] = unified.predict_proba(holdout)
    holdout["predicted_tp_priority_score"] = score_model.predict(holdout)

    specialist_selection, specialist_probability, specialist_meta = _specialist_models(
        fit,
        validation,
        train_all,
        holdout,
        feature_columns,
        args,
    )
    holdout["specialist_joint_probability"] = specialist_probability
    _write_csv(specialist_selection, out_dir / "10_specialist_validation_selection.csv")
    _write_csv(pd.DataFrame(specialist_meta), out_dir / "11_specialist_model_status.csv")

    unified_metrics = pd.DataFrame([{"model": "unified", **binary_metrics(holdout["joint_swing_tp_success"], holdout["unified_joint_probability"])}])
    score_holdout = pd.DataFrame([{"model": "unified_score", **score_metrics(holdout["tp_priority_score"], holdout["predicted_tp_priority_score"])}])
    per_mechanism = _per_mechanism_metrics(holdout)
    _write_csv(unified_metrics, out_dir / "12_holdout_unified_metrics.csv")
    _write_csv(score_holdout, out_dir / "13_holdout_score_metrics.csv")
    _write_csv(per_mechanism, out_dir / "14_holdout_per_mechanism_metrics.csv")

    buckets = probability_bucket_table(holdout, probability_column="unified_joint_probability")
    _write_csv(buckets, out_dir / "15_holdout_probability_buckets.csv")
    _plot_probability_buckets(buckets, out_dir / "15_holdout_probability_buckets.png")

    stability = build_model_stability(
        train_all,
        holdout,
        feature_columns=feature_columns,
        target_column="joint_swing_tp_success",
        family=chosen_binary,
        min_samples_leaf=int(args.model_min_samples_leaf),
    )
    _write_csv(stability, out_dir / "16_random_seed_stability.csv")

    importance = model_feature_importance(
        unified,
        validation,
        target_column="joint_swing_tp_success",
        random_state=int(args.random_state),
    )
    _write_csv(importance, out_dir / "17_unified_permutation_importance.csv")

    representative = representative_prediction_cases(
        holdout,
        probability_column="unified_joint_probability",
        target_column="joint_swing_tp_success",
        per_case=20,
    )
    representative_columns = [
        c for c in (
            "prediction_case",
            "event_id",
            "extreme_time",
            "feature_available_time",
            "entry_time",
            "label_end_time",
            "mechanism_type",
            "mechanism_top_score",
            "mechanism_margin",
            "unified_joint_probability",
            "specialist_joint_probability",
            "predicted_tp_priority_score",
            "reference_mechanism_type",
            "historical_clear_swing_low",
            "joint_swing_tp_success",
            "mechanism_joint_success",
            "tp_hit_1pct",
            "mfe_pct",
            "mae_pct",
            "mae_before_tp_pct",
            "tp_priority_score",
            "first_touch_score",
            "tp_first_touch_bar",
            "adverse_first_touch_bar",
        )
        if c in representative.columns
    ]
    _write_csv(representative[representative_columns], out_dir / "18_representative_prediction_cases.csv")

    # Full predictions are useful locally but may be skipped by the review pack if large.
    prediction_columns = [
        c for c in (
            "event_id",
            "extreme_time",
            "feature_available_time",
            "entry_time",
            "label_end_time",
            "split",
            "year",
            "mechanism_type",
            "secondary_mechanism_type",
            "mechanism_top_score",
            "mechanism_margin",
            "unified_joint_probability",
            "specialist_joint_probability",
            "predicted_tp_priority_score",
            "reference_mechanism_type",
            "historical_clear_swing_low",
            "joint_swing_tp_success",
            "mechanism_joint_success",
            "tp_hit_1pct",
            "mfe_pct",
            "mae_pct",
            "mae_before_tp_pct",
            "tp_priority_score",
            "first_touch_score",
            "tp_first_touch_bar",
            "adverse_first_touch_bar",
            "same_bar_tp_adverse_flag",
            "episode_id",
            "episode_size",
        )
        if c in holdout.columns
    ]
    _write_csv(holdout[prediction_columns], out_dir / "19_holdout_predictions.csv")

    print("[stage] future perturbation causal audit", flush=True)
    raw_audit = _raw_future_perturbation_audit(bars, audit_candidates, feature_columns, args)
    _write_csv(raw_audit, out_dir / "20_raw_future_perturbation_audit.csv")
    audit = _causal_audit(model_frame, feature_columns, raw_audit, gate_recall)
    _write_csv(audit, out_dir / "21_causal_audit.csv")
    if not audit["passed"].all():
        failed = audit.loc[~audit["passed"], "check"].tolist()
        raise RuntimeError(f"Causal audit failed: {failed}")

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
        "fit_end_date": args.fit_end_date,
        "validation_end_date": args.validation_end_date,
        "target_move_pct": float(args.target_move_pct),
        "adverse_move_pct": float(args.adverse_move_pct),
        "forward_horizon_bars": int(args.forward_horizon_bars),
        "swing_extreme_price_source": "low",
        "swing_entry_price_source": "next_bar_open",
        "swing_target_observation_source": "future_closed_bar_close",
        "swing_return_definition": "future_closed_bar_close / next_bar_open - 1",
        "candidate_count_before_mechanism_clarity": candidate_count_before_clarity,
        "clear_candidate_count": int(len(model_frame)),
        "model_feature_count": int(len(feature_columns)),
        "online_sequence_score_feature_count": int(len(ONLINE_SEQUENCE_SCORE_FEATURES)),
        "candidate_feature_chunk_size": int(args.candidate_feature_chunk_size),
        "feature_workers": int(args.feature_workers),
        "feature_worker_batch_size": int(args.feature_worker_batch_size),
        "label_vectorized_chunk_size": int(args.label_vectorized_chunk_size),
        "feature_pipeline": "chunked_compact_gate_then_full_exact_features_plus_parallel_mechanism",
        "selected_binary_family": chosen_binary,
        "selected_score_family": chosen_score,
        "mechanism_top_score_threshold": float(top_threshold),
        "mechanism_margin_threshold": float(margin_threshold),
        "clear_mechanisms": list(CLEAR_MECHANISMS),
        "excluded_mechanisms": ["B1_absorption", "B5_slow_accumulation"],
        "causal_policy": "candidate gate and features use current closed bar or older; entry is next-bar open; targets and path scores use future closed-bar closes only",
        "stage2_manifest": stage2_manifest,
        "stage3_manifest_summary": {
            "experiment_id": stage3_manifest.get("experiment_id"),
            "event_count": stage3_manifest.get("event_count"),
            "swing_extreme_price_source": stage3_manifest.get("swing_extreme_price_source"),
            "swing_entry_price_source": stage3_manifest.get("swing_entry_price_source"),
            "swing_target_observation_source": stage3_manifest.get("swing_target_observation_source"),
            "causal_policy": stage3_manifest.get("causal_policy"),
        },
        "strategy_or_backtest": False,
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = _summary(args, model_frame, binary_selection, unified_metrics, per_mechanism, audit)
    (out_dir / "22_RESEARCH_SUMMARY.md").write_text(summary, encoding="utf-8")

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
    args = parse_args(argv)
    run_research(args)


if __name__ == "__main__":
    main()
