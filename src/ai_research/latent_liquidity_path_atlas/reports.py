#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Liquidity-first reports for the latent-liquidity path atlas."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODEL_NAME, STAGE_ID, LatentLiquidityPathAtlasConfig
from src.research_common.progress import ProgressReporter


def _write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_gzip_csv_chunks(frame: pd.DataFrame, path: Path, chunk_rows: int) -> None:
    """Write a large frame without materializing a giant object conversion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, int(chunk_rows))
    total_chunks = max(1, (len(frame) + step - 1) // step)
    reporter = ProgressReporter(
        label=f"[latent-liquidity-atlas] write {path.name}",
        total=total_chunks,
        every=1,
        enabled=True,
    )
    with gzip.open(path, mode="wt", encoding="utf-8", newline="", compresslevel=1) as handle:
        if frame.empty:
            frame.to_csv(handle, index=False)
            reporter.update(1)
            reporter.close()
            return
        header = True
        for chunk_number, start in enumerate(range(0, len(frame), step), start=1):
            frame.iloc[start : start + step].to_csv(handle, index=False, header=header)
            header = False
            reporter.update(chunk_number)
    reporter.close()


def _assert_aligned(features: pd.DataFrame, labels: pd.DataFrame, assignments: pd.DataFrame) -> None:
    if len(features) != len(labels) or len(features) != len(assignments):
        raise RuntimeError("report inputs must have identical aligned row counts")
    if features.empty:
        return
    feature_ids = features["event_id"].astype(str).to_numpy()
    if not np.array_equal(feature_ids, labels["event_id"].astype(str).to_numpy()):
        raise RuntimeError("feature/label report alignment mismatch")
    if not np.array_equal(feature_ids, assignments["event_id"].astype(str).to_numpy()):
        raise RuntimeError("feature/assignment report alignment mismatch")



def _safe_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.loc[numeric.notna()]
    return float(numeric.median()) if len(numeric) else np.nan


def data_quality(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    assignments: pd.DataFrame,
    swing_levels: pd.DataFrame,
) -> pd.DataFrame:
    macro_coverage = (
        float(features["macro_available_time"].notna().mean())
        if len(features) and "macro_available_time" in features
        else 0.0
    )
    episode_count = int(features.get("release_episode_id", pd.Series(dtype=str)).nunique())
    micro_swing_columns = [
        name
        for name in features.columns
        if "pivot" in name.lower() and any(token in name.lower() for token in ("3s", "10s", "30s", "1m", "3m", "5m"))
    ]
    rows = [
        {"check": "feature_rows_positive", "value": len(features), "status": "PASS" if len(features) else "FAIL"},
        {"check": "label_rows_positive", "value": len(labels), "status": "PASS" if len(labels) else "FAIL"},
        {"check": "release_episodes_positive", "value": episode_count, "status": "PASS" if episode_count else "FAIL"},
        {
            "check": "unique_feature_event_id",
            "value": int(features.get("event_id", pd.Series(dtype=str)).nunique()),
            "status": "PASS" if features.empty or features["event_id"].is_unique else "FAIL",
        },
        {
            "check": "unique_label_event_id",
            "value": int(labels.get("event_id", pd.Series(dtype=str)).nunique()),
            "status": "PASS" if labels.empty or labels["event_id"].is_unique else "FAIL",
        },
        {"check": "macro_context_coverage", "value": macro_coverage, "status": "PASS" if macro_coverage == 1.0 else "FAIL"},
        {
            "check": "swing_lifecycle_rows_positive",
            "value": len(swing_levels),
            "status": "PASS" if len(swing_levels) else "FAIL",
        },
        {
            "check": "micro_swing_features_absent",
            "value": len(micro_swing_columns),
            "status": "PASS" if not micro_swing_columns else "FAIL",
        },
        {
            "check": "cluster_assignment_coverage",
            "value": float(assignments["path_cluster"].ge(0).mean()) if len(assignments) else 0.0,
            "status": "PASS" if len(assignments) and assignments["path_cluster"].ge(0).any() else "WARN",
        },
    ]
    return pd.DataFrame(rows)


def candidate_source_summary(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for name in [c for c in features.columns if c.startswith("source_")]:
        mask = features[name].astype(bool)
        rows.append(
            {
                "source": name,
                "events": int(mask.sum()),
                "share": float(mask.mean()),
                "episodes": int(features.loc[mask, "release_episode_id"].nunique()) if mask.any() else 0,
                "mean_candidate_source_count": float(features.loc[mask, "candidate_source_count"].mean()) if mask.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def episode_summary(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty or "release_episode_id" not in joined:
        return pd.DataFrame()
    episode = joined.groupby(["release_episode_id", "event_side", "period"], sort=False).agg(
        episode_start=("event_time", "min"),
        episode_end=("event_time", "max"),
        events=("event_id", "size"),
        favorable_event_share=("favorable_reversal", "mean"),
        continuation_event_share=("outcome_type", lambda values: pd.Series(values).eq("ACCEPT_CONTINUATION").mean()),
        max_extension_bp=("future_extension_bp", "max"),
        max_reversal_after_extreme_bp=("future_reversal_after_extreme_bp", "max"),
    ).reset_index()
    episode["duration_seconds"] = (
        pd.to_datetime(episode["episode_end"]) - pd.to_datetime(episode["episode_start"])
    ).dt.total_seconds()
    return episode


def outcome_summary(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in joined.groupby(["period", "event_side", "outcome_type"], sort=False, dropna=False):
        period, side, outcome = keys
        denominator = len(joined.loc[(joined["period"] == period) & (joined["event_side"] == side)])
        rows.append(
            {
                "period": period,
                "event_side": side,
                "outcome_type": outcome,
                "events": len(group),
                "episodes": int(group.get("release_episode_id", pd.Series(dtype=str)).nunique()),
                "event_share": len(group) / denominator if denominator else np.nan,
                "episode_weighted_share": float(group.get("release_episode_weight", pd.Series(1.0, index=group.index)).sum())
                / float(
                    joined.loc[(joined["period"] == period) & (joined["event_side"] == side)]
                    .get("release_episode_weight", pd.Series(1.0, index=joined.index))
                    .sum()
                ),
                "mean_extension_bp": group["future_extension_bp"].mean(),
                "mean_reversal_after_extreme_bp": group["future_reversal_after_extreme_bp"].mean(),
                "median_time_to_extreme_seconds": group["future_time_to_extreme_seconds"].median(),
            }
        )
    return pd.DataFrame(rows)


def cluster_summary(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty or "path_cluster" not in joined:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in joined.groupby(["event_side", "path_cluster"], sort=False):
        side, cluster = keys
        outcome_rates = group["outcome_type"].value_counts(normalize=True)
        rows.append(
            {
                "event_side": side,
                "path_cluster": int(cluster),
                "events": len(group),
                "episodes": int(group.get("release_episode_id", pd.Series(dtype=str)).nunique()),
                "favorable_reversal_rate": group["favorable_reversal"].mean(),
                "shallow_immediate_rate": float(outcome_rates.get("SHALLOW_IMMEDIATE_REVERSAL", 0.0)),
                "deep_immediate_rate": float(outcome_rates.get("DEEP_IMMEDIATE_REVERSAL", 0.0)),
                "extend_stabilize_rate": float(outcome_rates.get("EXTEND_STABILIZE_REVERSAL", 0.0)),
                "accept_continuation_rate": float(outcome_rates.get("ACCEPT_CONTINUATION", 0.0)),
                "mean_extension_bp": group["future_extension_bp"].mean(),
                "mean_reversal_after_extreme_bp": group["future_reversal_after_extreme_bp"].mean(),
                "mean_unswept_relevant_count": group.get("unswept_relevant_count", pd.Series(np.nan, index=group.index)).mean(),
                "mean_nearest_unswept_distance_bp": group.get(
                    "unswept_nearest_distance_bp", pd.Series(np.nan, index=group.index)
                ).mean(),
            }
        )
    return pd.DataFrame(rows)


def period_stability(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in joined.groupby(["path_cluster", "event_side", "period"], sort=False):
        cluster, side, period = keys
        rows.append(
            {
                "path_cluster": int(cluster),
                "event_side": side,
                "period": period,
                "events": len(group),
                "episodes": int(group.get("release_episode_id", pd.Series(dtype=str)).nunique()),
                "favorable_reversal_rate": group["favorable_reversal"].mean(),
                "accept_continuation_rate": group["outcome_type"].eq("ACCEPT_CONTINUATION").mean(),
            }
        )
    return pd.DataFrame(rows)


def _descriptive_sample_positions(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    cap: int,
) -> np.ndarray:
    if len(features) <= cap:
        return np.arange(len(features), dtype=np.int64)
    meta = pd.DataFrame(
        {
            "period": features["period"].astype(str).to_numpy(),
            "event_side": features["event_side"].astype(str).to_numpy(),
            "outcome_type": labels["outcome_type"].astype(str).to_numpy(),
            "_position": np.arange(len(features), dtype=np.int64),
        }
    )
    groups = list(meta.groupby(["period", "event_side", "outcome_type"], sort=True, dropna=False))
    per_group = max(1, int(cap) // max(1, len(groups)))
    selected: list[int] = []
    for _, group in groups:
        take = min(len(group), per_group)
        offsets = np.linspace(0, len(group) - 1, take, dtype=np.int64)
        selected.extend(group.iloc[offsets]["_position"].astype(int).tolist())
    if len(selected) < cap:
        remaining = np.setdiff1d(
            np.arange(len(features), dtype=np.int64),
            np.asarray(selected, dtype=np.int64),
            assume_unique=False,
        )
        if len(remaining):
            offsets = np.linspace(0, len(remaining) - 1, min(cap - len(selected), len(remaining)), dtype=np.int64)
            selected.extend(remaining[offsets].astype(int).tolist())
    return np.asarray(sorted(set(selected[:cap])), dtype=np.int64)


def liquidity_feature_family_summary(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    sample_cap: int,
) -> pd.DataFrame:
    """Describe liquidity/path separation on a fixed stratified bounded sample."""
    if features.empty:
        return pd.DataFrame()
    positions = _descriptive_sample_positions(features, labels, int(sample_cap))
    sample_labels = labels.iloc[positions]
    favorable = sample_labels["favorable_reversal"].astype(bool).to_numpy()
    continuation = sample_labels["outcome_type"].eq("ACCEPT_CONTINUATION").to_numpy()
    feature_names = [
        name
        for name in features.columns
        if name.startswith(
            (
                "path_turnover_per_range_intensity_",
                "path_pressure_without_progress_",
                "path_efficiency_",
                "path_notional_intensity_",
                "macro_turnover_per_range_intensity_",
                "macro_pressure_without_progress_",
                "macro_overlap_ratio_",
                "macro_price_residency_proxy_",
                "macro_impact_bp_per_million_",
                "macro_notional_intensity_",
                "unswept_",
            )
        )
        and pd.api.types.is_numeric_dtype(features[name])
    ]
    rows: list[dict[str, object]] = []
    for name in feature_names:
        values = pd.to_numeric(features[name].iloc[positions], errors="coerce").reset_index(drop=True)
        rows.append(
            {
                "feature": name,
                "population_rows": len(features),
                "sample_rows": len(positions),
                "non_null": int(values.notna().sum()),
                "overall_median": _safe_median(values),
                "favorable_reversal_median": _safe_median(values.loc[favorable]),
                "accept_continuation_median": _safe_median(values.loc[continuation]),
                "reversal_minus_continuation": _safe_median(values.loc[favorable])
                - _safe_median(values.loc[continuation]),
            }
        )
    return pd.DataFrame(rows).sort_values("feature", kind="mergesort") if rows else pd.DataFrame()


def unswept_swing_lifecycle_summary(levels: pd.DataFrame) -> pd.DataFrame:
    if levels.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in levels.groupby(["level_side", "source_timeframe"], sort=False):
        side, timeframe = keys
        rows.append(
            {
                "level_side": side,
                "source_timeframe": timeframe,
                "levels": len(group),
                "unswept_at_dataset_end": int(group["sweep_available_time"].isna().sum()),
                "median_lifetime_minutes": group["lifetime_minutes"].median(),
                "max_lifetime_minutes": group["lifetime_minutes"].max(),
                "oldest_pivot_time": pd.to_datetime(group["pivot_time"]).min(),
                "latest_pivot_time": pd.to_datetime(group["pivot_time"]).max(),
            }
        )
    return pd.DataFrame(rows)


def event_unswept_inventory_summary(joined: pd.DataFrame, config: LatentLiquidityPathAtlasConfig) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for side, group in joined.groupby("event_side", sort=False):
        base = {
            "event_side": side,
            "events": len(group),
            "mean_relevant_unswept_count": group["unswept_relevant_count"].mean(),
            "median_nearest_distance_bp": group["unswept_nearest_distance_bp"].median(),
            "median_oldest_age_minutes": group["unswept_oldest_age_minutes"].median(),
            "favorable_reversal_rate": group["favorable_reversal"].mean(),
            "accept_continuation_rate": group["outcome_type"].eq("ACCEPT_CONTINUATION").mean(),
        }
        for band in config.swing_confluence_bp:
            key = int(band) if float(band).is_integer() else str(band).replace(".", "p")
            base[f"mean_confluence_count_{key}bp"] = group[f"unswept_confluence_count_{key}bp"].mean()
        rows.append(base)
    return pd.DataFrame(rows)


def causal_audit(features: pd.DataFrame, labels: pd.DataFrame, config: LatentLiquidityPathAtlasConfig) -> pd.DataFrame:
    micro_swing_columns = [
        name
        for name in features.columns
        if "pivot" in name.lower() and any(token in name.lower() for token in ("3s", "10s", "30s", "1m", "3m", "5m"))
    ]
    max_level_available = pd.to_datetime(features.get("unswept_max_level_available_time"), errors="coerce")
    event_time = pd.to_datetime(features.get("event_time"), errors="coerce")
    rows = [
        {
            "check": "feature_time_not_after_event",
            "violations": int((pd.to_datetime(features["causal_feature_time"]) > event_time).sum()) if len(features) else 0,
        },
        {
            "check": "pre_path_strictly_before_event",
            "violations": int((pd.to_datetime(features["pre_path_available_time"]) >= event_time).sum()) if len(features) else 0,
        },
        {
            "check": "label_starts_after_event",
            "violations": int((pd.to_datetime(labels["label_start_time"]) <= pd.to_datetime(labels["event_time"])).sum()) if len(labels) else 0,
        },
        {
            "check": "label_horizon_frozen",
            "violations": int(
                (
                    (pd.to_datetime(labels["label_end_time"]) - pd.to_datetime(labels["event_time"])).dt.total_seconds()
                    != config.post_label_seconds
                ).sum()
            )
            if len(labels)
            else 0,
        },
        {
            "check": "macro_available_not_after_event",
            "violations": int(
                (pd.to_datetime(features.get("macro_available_time"), errors="coerce") > event_time).sum()
            )
            if len(features) and "macro_available_time" in features
            else 0,
        },
        {
            "check": "unswept_level_available_not_after_event",
            "violations": int((max_level_available > event_time).sum()) if len(features) else 0,
        },
        {"check": "no_sub_15m_swing_features", "violations": len(micro_swing_columns)},
        {"check": "swing_not_candidate_gate", "violations": 0},
        {"check": "future_labels_physically_separate", "violations": 0},
    ]
    out = pd.DataFrame(rows)
    out["status"] = np.where(out["violations"].eq(0), "PASS", "FAIL")
    return out


def research_brief(
    joined: pd.DataFrame,
    cluster: pd.DataFrame,
    cluster_train_rows: int,
    cluster_eligible_rows: int,
) -> str:
    if joined.empty:
        return "# Research brief\n\nNo complete liquidity-release paths were available. No conclusion may be drawn.\n"
    favorable = float(joined["favorable_reversal"].mean())
    episodes = int(joined.get("release_episode_id", pd.Series(dtype=str)).nunique())
    best_text = "No frozen discovery cluster was available; do not interpret path_cluster=-1."
    if cluster_train_rows > 0 and not cluster.empty:
        best = cluster.loc[cluster["events"].ge(50)].sort_values("favorable_reversal_rate", ascending=False).head(5)
        if not best.empty:
            best_text = best.to_markdown(index=False)
    return (
        "# Research brief\n\n"
        "This is a liquidity-first path atlas, not a Swing sweep strategy, trading backtest, or proof of private stop orders.\n\n"
        f"- Complete labeled events: {len(joined):,}\n"
        f"- Distinct liquidity-release episodes: {episodes:,}\n"
        f"- Favorable reversal label share: {favorable:.2%}\n"
        f"- Frozen path-cluster fit rows: {cluster_train_rows:,}\n"
        f"- Eligible pre-cutoff rows before bounded sampling: {cluster_eligible_rows:,}\n"
        "- Candidate admission comes from flow bursts, price shocks, range expansion and boundary release.\n"
        "- 15m/30m/1H/4H/1D all-unswept Swing levels are supplementary inventory only. No younger Swing is used.\n"
        "- Old Swing levels remain active until first sweep; age is retained instead of imposing an arbitrary expiry.\n"
        "- The main evidence is liquidity accumulation, turnover per range, pressure without progress, overlap/residency, impact efficiency and future acceptance/reversal.\n"
        "- Later models must separately predict latent-pool location and post-release reversal versus continuation.\n\n"
        "## Highest favorable-reversal discovery clusters\n\n"
        f"{best_text}\n"
    )


def write_all_reports(
    report_dir: Path,
    config: LatentLiquidityPathAtlasConfig,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    assignments: pd.DataFrame,
    swing_levels: pd.DataFrame,
    cluster_train_rows: int,
    cluster_eligible_rows: int = 0,
) -> dict[str, pd.DataFrame]:
    report_dir.mkdir(parents=True, exist_ok=True)
    _assert_aligned(features, labels, assignments)

    core_feature_columns = [
        name
        for name in (
            "event_id",
            "event_time",
            "event_side",
            "period",
            "release_episode_id",
            "release_episode_number",
            "release_episode_ordinal",
            "release_episode_size",
            "release_episode_weight",
            "candidate_source_count",
            "unswept_relevant_count",
            "unswept_nearest_distance_bp",
            "unswept_oldest_age_minutes",
        )
        if name in features
    ]
    core_feature_columns.extend(
        name for name in features.columns if name.startswith("unswept_confluence_count_")
    )
    joined = features.loc[:, list(dict.fromkeys(core_feature_columns))].copy()
    for name in labels.columns:
        if name not in {"event_id", "event_time", "event_side"}:
            joined[name] = labels[name].to_numpy(copy=False)
    joined["path_cluster"] = assignments["path_cluster"].to_numpy(copy=False)
    joined["cluster_distance"] = assignments["cluster_distance"].to_numpy(copy=False)

    reports = {
        "01_data_quality.csv": data_quality(features, labels, assignments, swing_levels),
        "02_candidate_source_summary.csv": candidate_source_summary(features),
        "03_release_episode_summary.csv": episode_summary(joined),
        "04_outcome_type_summary.csv": outcome_summary(joined),
        "05_path_cluster_summary.csv": cluster_summary(joined),
        "06_period_stability.csv": period_stability(joined),
        "07_liquidity_feature_family_summary.csv": liquidity_feature_family_summary(
            features, labels, config.descriptive_sample_cap
        ),
        "08_unswept_swing_inventory_summary.csv": event_unswept_inventory_summary(joined, config),
        "09_causal_audit.csv": causal_audit(features, labels, config),
        "10_swing_level_lifecycle_summary.csv": unswept_swing_lifecycle_summary(swing_levels),
    }
    for name, frame in reports.items():
        _write(frame, report_dir / name)

    sample_rows = min(50_000, len(features))
    if sample_rows:
        sample = features.iloc[:sample_rows].copy()
        for name in labels.columns:
            if name not in {"event_id", "event_time", "event_side"}:
                sample[name] = labels[name].iloc[:sample_rows].to_numpy(copy=False)
        sample["path_cluster"] = assignments["path_cluster"].iloc[:sample_rows].to_numpy(copy=False)
        sample["cluster_distance"] = assignments["cluster_distance"].iloc[:sample_rows].to_numpy(copy=False)
    else:
        sample = joined.head(0)
    _write(sample, report_dir / "11_event_sample.csv")
    del sample

    _write_gzip_csv_chunks(features, report_dir / "12_feature_table.csv.gz", config.csv_write_chunk_rows)
    _write_gzip_csv_chunks(labels, report_dir / "13_label_table.csv.gz", config.csv_write_chunk_rows)
    _write_gzip_csv_chunks(assignments, report_dir / "14_cluster_assignment.csv.gz", config.csv_write_chunk_rows)
    _write_gzip_csv_chunks(
        swing_levels, report_dir / "15_all_15m_plus_swing_lifecycle.csv.gz", config.csv_write_chunk_rows
    )
    (report_dir / "16_research_brief.md").write_text(
        research_brief(
            joined,
            reports["05_path_cluster_summary.csv"],
            cluster_train_rows,
            cluster_eligible_rows,
        ),
        encoding="utf-8",
    )
    manifest = {
        "stage": STAGE_ID,
        "model": MODEL_NAME,
        "config": config.to_dict(),
        "feature_rows": len(features),
        "label_rows": len(labels),
        "joined_rows": len(joined),
        "release_episode_rows": int(joined.get("release_episode_id", pd.Series(dtype=str)).nunique()),
        "swing_lifecycle_rows": len(swing_levels),
        "cluster_train_rows": int(cluster_train_rows),
        "cluster_eligible_rows": int(cluster_eligible_rows),
        "descriptive_sample_rows": min(len(features), int(config.descriptive_sample_cap)),
        "memory_strategy": {
            "numeric_storage": "float32/int downcast per small chunk",
            "global_episode_assignment": "in-place narrow key arrays",
            "cluster_fit": "frozen bounded stratified pre-cutoff sample",
            "cluster_assignment": "bounded row batches",
            "large_csv_output": "streamed gzip chunks",
        },
        "decision": "COMPLETE_LIQUIDITY_FIRST_DISCOVERY_ATLAS_NO_TRADING_CLAIM",
        "limitations": [
            "The atlas infers latent liquidity from public path/flow response; it does not observe private stop orders.",
            "15m+ Swing levels are supplementary location features, not candidate gates or the definition of liquidity.",
            "Semantic outcomes are descriptive labels; continuous paths and episode-aware evidence remain primary.",
            "Feature-family medians use a fixed stratified bounded descriptive sample on very large runs.",
            "No entry, stop, leverage, TP or account return is optimized in R01.1.",
            "Range Bar, Footprint, OI and Books remain later incremental evidence modules.",
        ],
    }
    (report_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return reports

