#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal alignment and event-level transformations for R05."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import PostSweepOIConfig

EPS = 1e-12

R04_FEATURE_COLUMNS: tuple[str, ...] = (
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
    "running_low_vs_zone_floor_bp", "zone_floor_reclaimed",
    "zone_ceiling_reclaimed", "cum_delta_since_sweep",
    "cum_delta_ratio_since_sweep", "cvd_new_low_flag",
    "cvd_new_low_without_price_new_low", "negative_delta_without_price_new_low",
    "no_new_low_3bars", "no_new_low_5bars", "no_new_low_10bars",
    "micro_high_break_3bars", "micro_high_break_5bars", "micro_high_break_10bars",
    "delta_ratio_1m", "sell_share_1m", "large_delta_ratio_1m",
    "price_change_1m_bp", "downside_bp_per_sell_million_1m",
    "downside_bp_per_abs_negative_delta_million_1m", "delta_ratio_5m",
    "sell_share_5m", "large_delta_ratio_5m", "price_change_5m_bp",
    "downside_bp_per_sell_million_5m",
    "downside_bp_per_abs_negative_delta_million_5m", "delta_ratio_15m",
    "sell_share_15m", "price_change_15m_bp", "delta_ratio_30m",
    "sell_share_30m", "price_change_30m_bp",
)

R04_LABEL_COLUMNS: tuple[str, ...] = (
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
    "zone_timeframes", "zone_primary_timeframe", "zone_max_timeframe_min",
    "zone_has_1H", "zone_has_4H", "zone_has_1D", "zone_width_bp",
    "zone_age_median_minutes", "zone_age_max_minutes", "zone_fresh_member_share",
    "zone_all_members_fresh", "zone_prior_touch_median", "zone_prior_touch_max",
    "zone_confirmed_order_max", "zone_left_high_range_20_bp_max",
    "zone_confirmation_reaction_close_bp_max", "sweep_depth_below_floor_bp",
    "sweep_depth_to_pre_atr_240m", "pre_return_60m", "pre_down_efficiency_60m",
    "pre_atr_240m_bp", "current_delta_ratio",
)


def _read_selected(path: str | Path, columns: Iterable[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    wanted = set(columns)
    header = pd.read_csv(path, nrows=0).columns
    available = [name for name in header if name in wanted]
    missing = sorted(wanted - set(available))
    if missing:
        raise RuntimeError(f"required columns missing from {path.name}: {missing}")
    return pd.read_csv(path, usecols=available, low_memory=False)


def load_r04_tables(report_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only R04 columns required by R05 to keep memory bounded."""

    root = Path(report_dir)
    features = _read_selected(root / "13_checkpoint_feature_table.csv.gz", R04_FEATURE_COLUMNS)
    labels = _read_selected(root / "14_checkpoint_label_table.csv.gz", R04_LABEL_COLUMNS)
    static = _read_selected(root / "11_static_zone_event_features.csv", STATIC_COLUMNS)

    for frame in (features, labels):
        for name in ("event_available_time", "checkpoint_time", "checkpoint_available_time", "entry_reference_time"):
            if name in frame.columns:
                frame[name] = pd.to_datetime(frame[name], errors="coerce")
    features = features.dropna(subset=["checkpoint_id", "zone_event_id", "checkpoint_available_time"]).copy()
    labels = labels.dropna(subset=["checkpoint_id", "zone_event_id"]).copy()
    if features["checkpoint_id"].duplicated().any() or labels["checkpoint_id"].duplicated().any():
        raise RuntimeError("R04 checkpoint_id must be unique")
    static = static.drop_duplicates("zone_event_id", keep="last").copy()
    return features, labels, static


def causal_align_oi(
    checkpoints: pd.DataFrame,
    oi_features: pd.DataFrame,
    config: PostSweepOIConfig,
) -> pd.DataFrame:
    """Attach the latest published Binance metrics row to every checkpoint."""

    cfg = config.validate()
    left = checkpoints.copy()
    right = oi_features.copy()
    left["checkpoint_available_time"] = pd.to_datetime(left["checkpoint_available_time"], errors="coerce")
    if "available_time" not in right.columns:
        if isinstance(right.index, pd.DatetimeIndex) and right.index.name == "available_time":
            right["available_time"] = right.index
        else:
            raise RuntimeError("OI features require available_time")
    right["available_time"] = pd.to_datetime(right["available_time"], errors="coerce")
    right["timestamp"] = pd.to_datetime(right["timestamp"], errors="coerce")
    right = right.dropna(subset=["available_time", "timestamp"]).sort_values("available_time", kind="mergesort")
    right = right.drop_duplicates("available_time", keep="last")
    left = left.dropna(subset=["checkpoint_available_time"]).sort_values("checkpoint_available_time", kind="mergesort")

    keep = [
        "timestamp", "available_time", "sum_open_interest", "sum_open_interest_value",
        "taker_volume_imbalance", "top_trader_account_long_share",
        "top_trader_position_long_share", "global_account_long_share",
    ]
    for window in cfg.oi_windows:
        tag = _window_tag(window)
        keep.extend([
            f"oi_base_change_{tag}", f"oi_usd_change_{tag}",
            f"oi_baseline_age_seconds_{tag}",
        ])
    keep = [name for name in dict.fromkeys(keep) if name in right.columns]
    right = right.loc[:, keep].rename(columns={
        "timestamp": "oi_metric_time",
        "available_time": "oi_available_time",
        "sum_open_interest": "oi_base",
        "sum_open_interest_value": "oi_usd",
    })
    out = pd.merge_asof(
        left,
        right,
        left_on="checkpoint_available_time",
        right_on="oi_available_time",
        direction="backward",
        tolerance=pd.Timedelta(cfg.alignment_tolerance),
        allow_exact_matches=True,
    )
    out["oi_age_seconds"] = (
        out["checkpoint_available_time"] - pd.to_datetime(out["oi_available_time"], errors="coerce")
    ).dt.total_seconds()
    out["oi_context_present"] = out["oi_available_time"].notna()
    out["oi_causal_flag"] = out["oi_available_time"].notna() & (
        pd.to_datetime(out["oi_available_time"]) <= out["checkpoint_available_time"]
    )
    out["oi_source_exchange"] = "binance"
    out["oi_source_symbol"] = "ETHUSDT"
    return _add_current_states(out)


def _add_current_states(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    oi_change = pd.to_numeric(out.get("oi_base_change_5m"), errors="coerce")
    price_change = pd.to_numeric(out.get("price_change_5m_bp"), errors="coerce")
    delta = pd.to_numeric(out.get("delta_ratio_5m"), errors="coerce")

    price_sign = np.select([price_change < 0, price_change > 0], ["PRICE_DOWN", "PRICE_UP"], default="PRICE_FLAT")
    oi_sign = np.select([oi_change > 0, oi_change < 0], ["OI_UP", "OI_DOWN"], default="OI_FLAT")
    delta_sign = np.select([delta < 0, delta > 0], ["DELTA_NEG", "DELTA_POS"], default="DELTA_FLAT")
    valid = price_change.notna() & oi_change.notna()
    out["position_flow_state_5m"] = np.where(valid, np.char.add(np.char.add(price_sign, "__"), oi_sign), "MISSING")
    out["delta_oi_state_5m"] = np.where(
        delta.notna() & oi_change.notna(),
        np.char.add(np.char.add(delta_sign, "__"), oi_sign),
        "MISSING",
    )
    out["down_oi_up_flag"] = (price_change < 0) & (oi_change > 0)
    out["down_oi_down_flag"] = (price_change < 0) & (oi_change < 0)
    out["negative_delta_oi_up_flag"] = (delta < 0) & (oi_change > 0)
    out["negative_delta_oi_down_flag"] = (delta < 0) & (oi_change < 0)
    return out


def add_attempt_pair_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare each new-low attempt with the previous attempt in the same event."""

    out = frame.copy()
    out["prior_attempt_checkpoint_id"] = pd.NA
    attempt_mask = out["new_low_attempt_flag"].eq(True)
    attempts = out.loc[attempt_mask].sort_values(["zone_event_id", "new_low_attempt_index", "elapsed_bars"], kind="mergesort").copy()
    if attempts.empty:
        return out
    group = attempts.groupby("zone_event_id", sort=False)
    compare_cols = [
        "checkpoint_id", "oi_base", "oi_base_change_5m", "oi_base_change_15m",
        "downside_bp_per_sell_million_1m",
        "downside_bp_per_abs_negative_delta_million_1m",
        "delta_ratio_1m", "sell_share_1m", "close_vs_running_low_bp",
    ]
    for name in compare_cols:
        if name not in attempts.columns:
            continue
        prior = group[name].shift(1)
        target = "prior_attempt_checkpoint_id" if name == "checkpoint_id" else f"prior_attempt_{name}"
        attempts[target] = prior
    attempts["oi_base_change_since_prior_attempt"] = _safe_change(
        pd.to_numeric(attempts.get("oi_base"), errors="coerce"),
        pd.to_numeric(attempts.get("prior_attempt_oi_base"), errors="coerce"),
    )
    attempts["sell_impact_ratio_vs_prior_attempt"] = _safe_ratio(
        pd.to_numeric(attempts.get("downside_bp_per_sell_million_1m"), errors="coerce"),
        pd.to_numeric(attempts.get("prior_attempt_downside_bp_per_sell_million_1m"), errors="coerce"),
    )
    attempts["delta_impact_ratio_vs_prior_attempt"] = _safe_ratio(
        pd.to_numeric(attempts.get("downside_bp_per_abs_negative_delta_million_1m"), errors="coerce"),
        pd.to_numeric(attempts.get("prior_attempt_downside_bp_per_abs_negative_delta_million_1m"), errors="coerce"),
    )
    attempts["attempt_pair_complete"] = (
        attempts["sell_impact_ratio_vs_prior_attempt"].notna()
        & attempts["delta_impact_ratio_vs_prior_attempt"].notna()
        & attempts["oi_base_change_since_prior_attempt"].notna()
    )
    attempts["impact_weaker_vs_prior_attempt"] = (
        attempts["attempt_pair_complete"]
        & (attempts["sell_impact_ratio_vs_prior_attempt"] < 1.0)
        & (attempts["delta_impact_ratio_vs_prior_attempt"] < 1.0)
    )
    attempts["oi_rising_since_prior_attempt"] = (
        attempts["attempt_pair_complete"] & (attempts["oi_base_change_since_prior_attempt"] > 0)
    )
    attempts["oi_falling_since_prior_attempt"] = (
        attempts["attempt_pair_complete"] & (attempts["oi_base_change_since_prior_attempt"] < 0)
    )
    attempts["attempt_mechanism_state"] = np.select(
        [
            attempts["impact_weaker_vs_prior_attempt"] & attempts["oi_rising_since_prior_attempt"],
            attempts["impact_weaker_vs_prior_attempt"] & attempts["oi_falling_since_prior_attempt"],
            attempts["attempt_pair_complete"] & (~attempts["impact_weaker_vs_prior_attempt"]) & attempts["oi_rising_since_prior_attempt"],
            attempts["attempt_pair_complete"] & (~attempts["impact_weaker_vs_prior_attempt"]) & attempts["oi_falling_since_prior_attempt"],
        ],
        [
            "OI_UP_IMPACT_WEAKER", "OI_DOWN_IMPACT_WEAKER",
            "OI_UP_IMPACT_NOT_WEAKER", "OI_DOWN_IMPACT_NOT_WEAKER",
        ],
        default="NO_PRIOR_OR_MISSING",
    )
    updated = [
        name for name in attempts.columns
        if name.startswith("prior_attempt_")
        or name in {
            "oi_base_change_since_prior_attempt",
            "sell_impact_ratio_vs_prior_attempt",
            "delta_impact_ratio_vs_prior_attempt",
            "attempt_pair_complete",
            "impact_weaker_vs_prior_attempt",
            "oi_rising_since_prior_attempt",
            "oi_falling_since_prior_attempt",
            "attempt_mechanism_state",
        }
    ]
    patch = attempts.set_index("checkpoint_id")[updated]
    out = out.set_index("checkpoint_id")
    for name in updated:
        out.loc[patch.index, name] = patch[name]
    return out.reset_index()


def build_future_oi_labels(
    aligned: pd.DataFrame,
    oi_features: pd.DataFrame,
    config: PostSweepOIConfig,
) -> pd.DataFrame:
    """Build future-only OI path labels from publication-time observations."""

    cfg = config.validate()
    out = aligned[[
        "checkpoint_id", "zone_event_id", "period", "elapsed_bars",
        "checkpoint_available_time", "oi_available_time", "oi_base", "oi_usd",
    ]].copy()
    right = oi_features.copy()
    if "available_time" not in right.columns:
        right["available_time"] = right.index
    right["available_time"] = pd.to_datetime(right["available_time"], errors="coerce")
    right = right.dropna(subset=["available_time"]).sort_values("available_time", kind="mergesort")
    right = right.drop_duplicates("available_time", keep="last")
    times = right["available_time"].to_numpy(dtype="datetime64[ns]").astype("int64")
    base = pd.to_numeric(right["sum_open_interest"], errors="coerce").to_numpy(dtype="float64")
    usd = pd.to_numeric(right["sum_open_interest_value"], errors="coerce").to_numpy(dtype="float64")
    checkpoint_ns = pd.to_datetime(out["checkpoint_available_time"]).to_numpy(dtype="datetime64[ns]").astype("int64")
    tolerance_ns = int(pd.Timedelta(cfg.alignment_tolerance).value)
    current_base = pd.to_numeric(out["oi_base"], errors="coerce").to_numpy(dtype="float64")
    current_usd = pd.to_numeric(out["oi_usd"], errors="coerce").to_numpy(dtype="float64")

    for horizon in cfg.future_oi_horizons:
        target = checkpoint_ns + int(pd.Timedelta(minutes=int(horizon)).value)
        positions = np.searchsorted(times, target, side="right") - 1
        valid = positions >= 0
        safe = np.where(valid, positions, 0)
        age = target - times[safe]
        valid &= (age >= 0) & (age <= tolerance_ns)
        target_base = np.full(len(out), np.nan)
        target_usd = np.full(len(out), np.nan)
        target_time = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
        target_base[valid] = base[safe[valid]]
        target_usd[valid] = usd[safe[valid]]
        target_time[valid] = times[safe[valid]].astype("datetime64[ns]")
        out[f"future_oi_available_time_{horizon}m"] = pd.to_datetime(target_time)
        out[f"future_oi_label_complete_{horizon}m"] = valid & np.isfinite(current_base)
        out[f"future_oi_base_change_{horizon}m"] = _safe_change(target_base, current_base)
        out[f"future_oi_usd_change_{horizon}m"] = _safe_change(target_usd, current_usd)
    return out


def split_features_labels(
    aligned: pd.DataFrame,
    r04_labels: pd.DataFrame,
    future_oi_labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label = r04_labels.merge(future_oi_labels, on=["checkpoint_id", "zone_event_id", "period", "elapsed_bars"], how="left", validate="one_to_one")
    feature = aligned.copy()
    forbidden = [name for name in feature.columns if name.startswith("future_") or "oracle" in name.lower()]
    if forbidden:
        raise RuntimeError(f"future leakage in R05 feature table: {forbidden}")
    return feature, label


def first_per_event(frame: pd.DataFrame, *, sort_columns: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    cols = sort_columns or ["zone_event_id", "elapsed_bars"]
    return frame.sort_values(cols, kind="mergesort").groupby("zone_event_id", sort=False, as_index=False).first()


def oracle_turning_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Future-labelled earliest durable 60m reversal, for description only."""

    required = ["future_no_lower_low_60m", "future_mfe_60m", "future_reversal_dominant_60m"]
    if any(name not in frame.columns for name in required):
        return pd.DataFrame()
    mask = (
        frame["future_no_lower_low_60m"].eq(True)
        & (pd.to_numeric(frame["future_mfe_60m"], errors="coerce") >= 0.005)
        & frame["future_reversal_dominant_60m"].eq(True)
    )
    selected = frame.loc[mask].sort_values(["zone_event_id", "elapsed_bars"], kind="mergesort")
    if selected.empty:
        return selected
    out = selected.groupby("zone_event_id", sort=False, as_index=False).first()
    out["oracle_selection_uses_future"] = True
    return out


def pair_oracle_with_prior_attempt(frame: pd.DataFrame, oracle: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or oracle.empty:
        return pd.DataFrame()
    attempts = frame.loc[frame["new_low_attempt_flag"].eq(True)].copy()
    attempts = attempts.sort_values(["zone_event_id", "elapsed_bars"], kind="mergesort")
    rows: list[dict[str, object]] = []
    for row in oracle.itertuples(index=False):
        event_attempts = attempts.loc[
            (attempts["zone_event_id"] == row.zone_event_id)
            & (pd.to_numeric(attempts["elapsed_bars"]) < int(row.elapsed_bars))
        ]
        if event_attempts.empty:
            continue
        prior = event_attempts.iloc[-1]
        current = pd.Series(row._asdict())
        record: dict[str, object] = {
            "zone_event_id": row.zone_event_id,
            "period": row.period,
            "oracle_checkpoint_id": row.checkpoint_id,
            "prior_checkpoint_id": prior["checkpoint_id"],
            "oracle_elapsed_bars": row.elapsed_bars,
            "prior_elapsed_bars": prior["elapsed_bars"],
        }
        compare = [
            "oi_base_change_5m", "oi_base_change_15m", "oi_base_change_1h",
            "oi_base", "delta_ratio_1m", "sell_share_1m", "price_change_1m_bp",
            "downside_bp_per_sell_million_1m",
            "downside_bp_per_abs_negative_delta_million_1m",
            "close_vs_running_low_bp", "new_low_extension_bp",
            "future_oi_base_change_15m", "future_oi_base_change_30m",
            "future_oi_base_change_60m", "future_mfe_60m", "future_mae_60m",
        ]
        for name in compare:
            record[f"oracle_{name}"] = current.get(name, np.nan)
            record[f"prior_{name}"] = prior.get(name, np.nan)
        rows.append(record)
    return pd.DataFrame(rows)


def _window_tag(value: str) -> str:
    delta = pd.Timedelta(value)
    seconds = int(delta.total_seconds())
    if seconds % 86400 == 0:
        return f"{seconds // 86400}d"
    if seconds % 3600 == 0:
        return f"{seconds // 3600}h"
    return f"{seconds // 60}m"


def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=a.index, dtype="float64")
    valid = a.notna() & b.notna() & (np.abs(b) > EPS)
    out.loc[valid] = a.loc[valid] / b.loc[valid]
    return out


def _safe_change(current: pd.Series | np.ndarray, baseline: pd.Series | np.ndarray) -> pd.Series | np.ndarray:
    current_arr = np.asarray(current, dtype="float64")
    baseline_arr = np.asarray(baseline, dtype="float64")
    result = np.full(len(current_arr), np.nan, dtype="float64")
    valid = np.isfinite(current_arr) & np.isfinite(baseline_arr) & (np.abs(baseline_arr) > EPS)
    result[valid] = current_arr[valid] / baseline_arr[valid] - 1.0
    if isinstance(current, pd.Series):
        return pd.Series(result, index=current.index)
    return result
