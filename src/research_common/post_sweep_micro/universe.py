#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build future-labelled R06 attempt cohorts without leaking labels into features."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_feed.binance_futures_metrics_loader import BinanceFuturesMetricsLoader
from src.research_common.post_sweep_oi import PostSweepOIConfig, causal_align_oi

from .config import PostSweepMicroConfig


FEATURE_COLUMNS: tuple[str, ...] = (
    "checkpoint_id", "zone_event_id", "event_kind", "period", "event_pos",
    "event_available_time", "checkpoint_pos", "checkpoint_time",
    "checkpoint_available_time", "elapsed_bars", "zone_floor_price",
    "zone_ceiling_price", "zone_center_price", "sweep_low", "checkpoint_open",
    "checkpoint_high", "checkpoint_low", "checkpoint_close",
    "running_low_since_sweep", "new_low_attempt_flag", "new_low_attempt_index",
    "bars_since_new_low_attempt", "new_low_extension_bp",
    "new_low_extension_to_pre_atr_240m", "attempt_delta_notional",
    "attempt_sell_notional", "attempt_extension_vs_previous",
    "close_vs_zone_floor_bp", "close_vs_running_low_bp",
    "running_low_vs_zone_floor_bp", "delta_ratio_1m", "sell_share_1m",
    "large_delta_ratio_1m", "price_change_1m_bp",
    "downside_bp_per_sell_million_1m",
    "downside_bp_per_abs_negative_delta_million_1m",
    "delta_ratio_5m", "sell_share_5m", "price_change_5m_bp",
)

LABEL_COLUMNS: tuple[str, ...] = (
    "checkpoint_id", "zone_event_id", "period", "elapsed_bars",
    "entry_reference_time", "entry_reference_price",
    "future_label_complete_15m", "future_mfe_15m", "future_mae_15m",
    "future_close_return_15m", "future_no_lower_low_15m",
    "future_label_complete_30m", "future_mfe_30m", "future_mae_30m",
    "future_close_return_30m", "future_no_lower_low_30m",
    "future_label_complete_60m", "future_mfe_60m", "future_mae_60m",
    "future_close_return_60m", "future_no_lower_low_60m",
    "future_reversal_dominant_60m", "future_continuation_dominant_60m",
    "future_label_complete_180m", "future_mfe_180m", "future_mae_180m",
    "future_close_return_180m", "future_no_lower_low_180m",
    "future_reversal_dominant_180m", "future_continuation_dominant_180m",
    "future_large_mfe_0p5_180m", "future_large_mfe_1_180m",
    "future_large_mfe_2_180m",
)

STATIC_COLUMNS: tuple[str, ...] = (
    "zone_event_id", "zone_member_count", "zone_timeframe_count",
    "zone_primary_timeframe", "zone_max_timeframe_min", "zone_has_1H",
    "zone_has_4H", "zone_has_1D", "zone_width_bp", "zone_age_median_minutes",
    "zone_age_max_minutes", "zone_fresh_member_share", "zone_all_members_fresh",
    "zone_prior_touch_median", "zone_prior_touch_max", "zone_confirmed_order_max",
    "zone_left_high_range_20_bp_max", "zone_confirmation_reaction_close_bp_max",
    "sweep_depth_below_floor_bp", "sweep_depth_to_pre_atr_240m",
    "pre_return_60m", "pre_down_efficiency_60m", "pre_atr_240m_bp",
    "current_delta_ratio",
)

OI_COLUMNS: tuple[str, ...] = (
    "checkpoint_id", "oi_context_present", "oi_metric_time", "oi_available_time",
    "oi_age_seconds", "oi_base", "oi_usd", "oi_base_change_5m",
    "oi_base_change_15m", "oi_base_change_30m", "oi_base_change_1h",
    "oi_base_change_4h", "oi_base_change_1d", "position_flow_state_5m",
    "delta_oi_state_5m", "down_oi_up_flag", "down_oi_down_flag",
    "negative_delta_oi_up_flag", "negative_delta_oi_down_flag",
    "taker_volume_imbalance", "top_trader_account_long_share",
    "top_trader_position_long_share", "global_account_long_share",
)


def _read_selected(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0).columns
    available = [name for name in columns if name in header]
    missing = sorted(set(columns) - set(available))
    if missing:
        raise RuntimeError(f"required columns missing from {path.name}: {missing}")
    return pd.read_csv(path, usecols=available, low_memory=False)


def load_r04_micro_source(report_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(report_dir)
    features = _read_selected(root / "13_checkpoint_feature_table.csv.gz", FEATURE_COLUMNS)
    labels = _read_selected(root / "14_checkpoint_label_table.csv.gz", LABEL_COLUMNS)
    static = _read_selected(root / "11_static_zone_event_features.csv", STATIC_COLUMNS)
    for frame in (features, labels):
        for name in ("event_available_time", "checkpoint_time", "checkpoint_available_time", "entry_reference_time"):
            if name in frame.columns:
                frame[name] = pd.to_datetime(frame[name], errors="coerce")
    features = features.dropna(subset=["checkpoint_id", "zone_event_id", "checkpoint_time", "checkpoint_available_time"]).copy()
    labels = labels.dropna(subset=["checkpoint_id", "zone_event_id"]).copy()
    if features["checkpoint_id"].duplicated().any() or labels["checkpoint_id"].duplicated().any():
        raise RuntimeError("R04 checkpoint_id must be unique")
    static = static.drop_duplicates("zone_event_id", keep="last").copy()
    return features, labels, static



def load_binance_oi_context(
    checkpoints: pd.DataFrame,
    *,
    symbol: str = "ETHUSDT",
    data_dir: str | Path | None = None,
    db_name: str = "binance_futures_metrics.db",
    publication_lag: str = "1min",
) -> pd.DataFrame:
    """Load and causally align OI directly from the indexed Binance store.

    This avoids scanning R05's 700k-row compressed checkpoint export merely to
    retrieve a sparse subset. The returned schema matches the OI context used
    by R05/R06.
    """

    if checkpoints.empty:
        return pd.DataFrame(columns=OI_COLUMNS)
    required = {"checkpoint_id", "checkpoint_available_time", "price_change_5m_bp", "delta_ratio_5m"}
    missing = sorted(required - set(checkpoints.columns))
    if missing:
        raise ValueError(f"OI checkpoints missing columns: {missing}")
    left = checkpoints.loc[:, sorted(required)].copy()
    left["checkpoint_available_time"] = pd.to_datetime(left["checkpoint_available_time"], errors="coerce")
    left = left.dropna(subset=["checkpoint_id", "checkpoint_available_time"])
    if left.empty:
        return pd.DataFrame(columns=OI_COLUMNS)

    cfg = PostSweepOIConfig(publication_lag=publication_lag).validate()
    loader = BinanceFuturesMetricsLoader(symbol=symbol, data_dir=data_dir, db_name=db_name)
    start = left["checkpoint_available_time"].min() - pd.Timedelta(cfg.alignment_tolerance)
    end = left["checkpoint_available_time"].max()
    metrics = loader.load_relative_features(
        start,
        end,
        windows=cfg.oi_windows,
        publication_lag=cfg.publication_lag,
        baseline_tolerance=cfg.baseline_tolerance,
        index_mode="available_time",
    )
    if metrics.empty:
        return pd.DataFrame(columns=OI_COLUMNS)
    aligned = causal_align_oi(left, metrics.reset_index(drop=True), cfg)
    available = [name for name in OI_COLUMNS if name in aligned.columns]
    return aligned.loc[:, available].drop_duplicates("checkpoint_id", keep="last").reset_index(drop=True)

def load_optional_r05_oi(
    report_dir: str | Path | None,
    checkpoint_ids: set[str] | None = None,
    *,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Load only requested R05 OI rows; the compressed table is streamed."""

    if report_dir is None:
        return pd.DataFrame(columns=OI_COLUMNS)
    path = Path(report_dir) / "15_checkpoint_oi_feature_table.csv.gz"
    if not path.exists():
        return pd.DataFrame(columns=OI_COLUMNS)
    header = pd.read_csv(path, nrows=0).columns
    available = [name for name in OI_COLUMNS if name in header]
    if "checkpoint_id" not in available:
        return pd.DataFrame(columns=OI_COLUMNS)
    wanted = {str(value) for value in checkpoint_ids} if checkpoint_ids is not None else None
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=available, low_memory=False, chunksize=max(10_000, int(chunksize))):
        if wanted is not None:
            chunk = chunk.loc[chunk["checkpoint_id"].astype(str).isin(wanted)]
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=available)
    frame = pd.concat(parts, ignore_index=True)
    for name in ("oi_metric_time", "oi_available_time"):
        if name in frame.columns:
            frame[name] = pd.to_datetime(frame[name], errors="coerce")
    return frame.drop_duplicates("checkpoint_id", keep="last")


def attach_optional_oi_context(features: pd.DataFrame, oi_features: pd.DataFrame) -> pd.DataFrame:
    if oi_features is None or oi_features.empty:
        return features.copy()
    return features.merge(oi_features, on="checkpoint_id", how="left", validate="one_to_one")

def build_attempt_universe(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    static: pd.DataFrame,
    config: PostSweepMicroConfig,
    *,
    oi_features: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return causal window features, physically separate labels and pair audit.

    Pair discovery is vectorized on the new-low-attempt table. Static Zone and
    OI columns are attached only after the selected cohort is known, avoiding a
    costly 700k-row wide merge.
    """

    cfg = config.validate()
    base = features.merge(
        labels,
        on=["checkpoint_id", "zone_event_id", "period", "elapsed_bars"],
        how="inner",
        validate="one_to_one",
    )
    base = base.sort_values(["zone_event_id", "elapsed_bars"], kind="mergesort").reset_index(drop=True)
    base["prior_running_low_before_attempt"] = base.groupby("zone_event_id", sort=False)["running_low_since_sweep"].shift(1)
    base["prior_checkpoint_available_time"] = base.groupby("zone_event_id", sort=False)["checkpoint_available_time"].shift(1)
    first = base["prior_running_low_before_attempt"].isna()
    base.loc[first, "prior_running_low_before_attempt"] = base.loc[first, "sweep_low"]

    oracle_mask = (
        base["future_no_lower_low_60m"].eq(True)
        & (pd.to_numeric(base["future_mfe_60m"], errors="coerce") >= cfg.oracle_min_mfe_60m)
        & base["future_reversal_dominant_60m"].eq(True)
    )
    oracle_ids = (
        base.loc[oracle_mask, ["checkpoint_id", "zone_event_id", "elapsed_bars"]]
        .sort_values(["zone_event_id", "elapsed_bars"], kind="mergesort")
        .drop_duplicates("zone_event_id", keep="first")
    )

    attempts = base.loc[base["new_low_attempt_flag"].eq(True)].copy()
    attempts = attempts.sort_values(["zone_event_id", "elapsed_bars"], kind="mergesort")
    attempts["prior_attempt_checkpoint_id"] = attempts.groupby("zone_event_id", sort=False)["checkpoint_id"].shift(1)
    oracle_attempts = attempts.merge(
        oracle_ids[["checkpoint_id"]], on="checkpoint_id", how="inner", validate="one_to_one"
    )
    oracle_attempts = oracle_attempts.dropna(subset=["prior_attempt_checkpoint_id"]).copy()
    prior_lookup = attempts.set_index("checkpoint_id", drop=False)
    prior_ids = oracle_attempts["prior_attempt_checkpoint_id"].astype(str).tolist()
    prior_attempts = prior_lookup.loc[prior_ids].reset_index(drop=True)
    if oracle_attempts.empty:
        raise RuntimeError("R06 could not build oracle/prior attempt pairs")

    pair_audit = pd.DataFrame(
        {
            "pair_id": oracle_attempts["zone_event_id"].astype(str).to_numpy(),
            "zone_event_id": oracle_attempts["zone_event_id"].to_numpy(),
            "period": oracle_attempts["period"].to_numpy(),
            "oracle_checkpoint_id": oracle_attempts["checkpoint_id"].to_numpy(),
            "prior_checkpoint_id": prior_attempts["checkpoint_id"].to_numpy(),
            "oracle_elapsed_bars": pd.to_numeric(oracle_attempts["elapsed_bars"]).astype(int).to_numpy(),
            "prior_elapsed_bars": pd.to_numeric(prior_attempts["elapsed_bars"]).astype(int).to_numpy(),
            "oracle_attempt_index": pd.to_numeric(oracle_attempts["new_low_attempt_index"]).astype(int).to_numpy(),
            "prior_attempt_index": pd.to_numeric(prior_attempts["new_low_attempt_index"]).astype(int).to_numpy(),
        }
    )
    pair_audit["attempt_gap_bars"] = pair_audit["oracle_elapsed_bars"] - pair_audit["prior_elapsed_bars"]

    oracle_attempts = oracle_attempts.copy()
    oracle_attempts["pair_id"] = oracle_attempts["zone_event_id"].astype(str)
    oracle_attempts["cohort"] = "ORACLE_TURN"
    oracle_attempts["selection_uses_future"] = True
    prior_attempts = prior_attempts.copy()
    prior_attempts["pair_id"] = prior_attempts["zone_event_id"].astype(str)
    prior_attempts["cohort"] = "PRIOR_FAILED_ATTEMPT"
    prior_attempts["selection_uses_future"] = True
    selected = pd.concat([oracle_attempts, prior_attempts], ignore_index=True, sort=False)

    used_checkpoints = set(selected["checkpoint_id"].astype(str))
    controls = attempts.loc[
        attempts["future_continuation_dominant_60m"].eq(True)
        & attempts["future_no_lower_low_60m"].eq(False)
        & (~attempts["checkpoint_id"].astype(str).isin(used_checkpoints))
    ].copy()
    n_control = int(round(len(oracle_attempts) * cfg.control_multiplier))
    if n_control > 0 and not controls.empty:
        controls["_stable_hash"] = pd.util.hash_pandas_object(
            controls[["checkpoint_id", "zone_event_id"]].astype(str), index=False
        ).to_numpy(dtype="uint64")
        period_count = max(1, controls["period"].nunique())
        per_period_target = max(1, int(np.ceil(n_control / period_count)))
        controls = (
            controls.sort_values(["period", "_stable_hash"], kind="mergesort")
            .groupby("period", sort=False, group_keys=False)
            .head(per_period_target)
            .head(n_control)
            .drop(columns=["_stable_hash"])
        )
        controls["pair_id"] = controls["zone_event_id"].astype(str) + "__CTRL"
        controls["cohort"] = "CONTINUATION_CONTROL"
        controls["selection_uses_future"] = True
        selected = pd.concat([selected, controls], ignore_index=True, sort=False)

    selected = selected.merge(static, on="zone_event_id", how="left", validate="many_to_one")
    if oi_features is not None and not oi_features.empty:
        selected = selected.merge(oi_features, on="checkpoint_id", how="left", validate="one_to_one")
    selected["window_id"] = selected["cohort"].astype(str) + "__" + selected["checkpoint_id"].astype(str)
    selected["start_time"] = selected["checkpoint_time"] - pd.to_timedelta(cfg.pre_window_seconds, unit="s")
    selected["end_time"] = selected["checkpoint_time"] + pd.to_timedelta(cfg.post_window_seconds, unit="s")

    future_cols = [name for name in selected.columns if name.startswith("future_")]
    label_cols = [
        "window_id", "checkpoint_id", "zone_event_id", "pair_id", "cohort", "period",
        "selection_uses_future", *future_cols,
    ]
    label_table = selected.loc[:, list(dict.fromkeys(label_cols))].copy()
    forbidden = set(future_cols) | {"selection_uses_future"}
    feature_table = selected.drop(columns=[name for name in forbidden if name in selected.columns]).copy()
    if any(name.startswith("future_") for name in feature_table.columns):
        raise RuntimeError("future label leaked into R06 attempt feature table")
    return feature_table.reset_index(drop=True), label_table.reset_index(drop=True), pair_audit.reset_index(drop=True)


__all__ = [
    "attach_optional_oi_context",
    "build_attempt_universe",
    "load_binance_oi_context",
    "load_optional_r05_oi",
    "load_r04_micro_source",
]
