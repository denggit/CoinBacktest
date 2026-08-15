#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train-only distance normalization and separate R02.3 ranking targets."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import ExcessLiquidityRankingConfig


@dataclass(frozen=True)
class DistanceNormalizer:
    table: pd.DataFrame


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame:
        return pd.Series(np.full(len(frame), float(default)), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def fit_distance_normalizer(frame: pd.DataFrame, config: ExcessLiquidityRankingConfig) -> DistanceNormalizer:
    """Freeze robust expected log-density by side x distance on TRAIN only."""
    w = config.primary_label_window_seconds
    density_col = f"ft_release_density_sum_{w}s"
    train = frame.loc[
        frame["period"].astype(str).eq(config.train_period)
        & frame["r02_3_source_eligible"].astype(bool)
    ].copy()
    if train.empty:
        raise RuntimeError("R02.3 has no TRAIN exact-first-touch rows for normalization")
    train["_log_density"] = np.log1p(_numeric(train, density_col).fillna(0.0).clip(lower=0.0))
    grouped = train.groupby(["zone_side", "zone_distance_bp"], sort=True)["_log_density"]
    table = grouped.agg(
        rows="size",
        expected_log_density="median",
        q25=lambda s: float(s.quantile(0.25)),
        q75=lambda s: float(s.quantile(0.75)),
        mean_log_density="mean",
    ).reset_index()
    side_fallback = train.groupby("zone_side", sort=True)["_log_density"].agg(
        fallback_expected="median",
        fallback_q25=lambda s: float(s.quantile(0.25)),
        fallback_q75=lambda s: float(s.quantile(0.75)),
    ).reset_index()
    table = table.merge(side_fallback, on="zone_side", how="left", validate="many_to_one")
    low_support = table["rows"].lt(config.normalizer_min_rows_per_distance)
    table.loc[low_support, "expected_log_density"] = table.loc[low_support, "fallback_expected"]
    table.loc[low_support, "q25"] = table.loc[low_support, "fallback_q25"]
    table.loc[low_support, "q75"] = table.loc[low_support, "fallback_q75"]
    table["robust_scale"] = (table["q75"] - table["q25"]).clip(lower=config.normalizer_iqr_floor)
    table["expected_density"] = np.expm1(table["expected_log_density"]).clip(lower=0.0)
    table["low_support_fallback"] = low_support.astype(bool)
    keep = [
        "zone_side", "zone_distance_bp", "rows", "expected_log_density", "expected_density",
        "robust_scale", "q25", "q75", "mean_log_density", "low_support_fallback",
    ]
    return DistanceNormalizer(table=table.loc[:, keep].copy())


def _add_group_relevance(
    out: pd.DataFrame,
    *,
    eligible: pd.Series,
    target: str,
    relevance_col: str,
    group_eligible_col: str,
    grades: int,
) -> None:
    out[relevance_col] = np.int16(-1)
    out[group_eligible_col] = False
    idx = out.index[eligible & pd.to_numeric(out[target], errors="coerce").notna()]
    if len(idx) == 0:
        return
    work = out.loc[idx, ["ranking_group", target]].copy()
    work[target] = pd.to_numeric(work[target], errors="coerce")
    stats = work.groupby("ranking_group", sort=False)[target].agg(["size", "min", "max"])
    good_groups = stats.index[(stats["size"] >= 2) & ((stats["max"] - stats["min"]) > 1e-12)]
    good = work["ranking_group"].isin(good_groups)
    if not good.any():
        return
    good_idx = work.index[good]
    pct = work.loc[good].groupby("ranking_group", sort=False)[target].rank(method="average", pct=True)
    bins = np.floor(np.clip(pct.to_numpy(dtype=float) - 1e-12, 0.0, 0.999999) * int(grades)).astype(np.int16)
    out.loc[good_idx, relevance_col] = bins
    out.loc[good_idx, group_eligible_col] = True


def attach_excess_and_reversal_targets(
    frame: pd.DataFrame,
    normalizer: DistanceNormalizer,
    config: ExcessLiquidityRankingConfig,
) -> pd.DataFrame:
    """Attach train-frozen excess-liquidity and separate reversal-quality labels."""
    w = config.primary_label_window_seconds
    out = frame.copy()
    out = out.merge(
        normalizer.table.loc[:, [
            "zone_side", "zone_distance_bp", "expected_log_density", "expected_density", "robust_scale",
        ]],
        on=["zone_side", "zone_distance_bp"],
        how="left",
        validate="many_to_one",
    )
    density = _numeric(out, f"ft_release_density_sum_{w}s").fillna(0.0).clip(lower=0.0)
    log_density = np.log1p(density)
    expected_log = _numeric(out, "expected_log_density", np.nan)
    scale = _numeric(out, "robust_scale", np.nan)
    out["excess_log_density"] = log_density - expected_log
    out["excess_liquidity_z"] = (log_density - expected_log) / scale
    out["density_vs_expected_ratio"] = (1.0 + density) / (1.0 + _numeric(out, "expected_density", 0.0).clip(lower=0.0))

    fav = _numeric(out, f"ft_favorable_density_sum_{w}s").fillna(0.0).clip(lower=0.0)
    cont = _numeric(out, f"ft_continuation_density_sum_{w}s").fillna(0.0).clip(lower=0.0)
    out["reversal_quality_target"] = np.log1p(fav) - np.log1p(cont)
    out["release_observed_180s"] = _numeric(out, f"ft_release_episode_count_{w}s").fillna(0.0).gt(0)
    out["favorable_observed_180s"] = _numeric(out, f"ft_favorable_episode_count_{w}s").fillna(0.0).gt(0)
    out["continuation_observed_180s"] = _numeric(out, f"ft_continuation_episode_count_{w}s").fillna(0.0).gt(0)
    out["sweep_depth_target_bp"] = _numeric(out, f"ft_sweep_depth_weighted_bp_{w}s", np.nan)
    out["reversal_room_target_bp"] = _numeric(out, f"ft_reversal_room_weighted_bp_{w}s", np.nan)
    out["ranking_group"] = out["decision_time"].astype(str) + "|" + out["zone_side"].astype(str)

    source_ok = out["r02_3_source_eligible"].astype(bool) & out["excess_liquidity_z"].notna()
    _add_group_relevance(
        out,
        eligible=source_ok,
        target="excess_liquidity_z",
        relevance_col="excess_relevance",
        group_eligible_col="excess_group_eligible",
        grades=config.rank_relevance_grades,
    )
    _add_group_relevance(
        out,
        eligible=source_ok & out["release_observed_180s"].astype(bool),
        target="reversal_quality_target",
        relevance_col="reversal_relevance",
        group_eligible_col="reversal_group_eligible",
        grades=config.rank_relevance_grades,
    )
    return out
