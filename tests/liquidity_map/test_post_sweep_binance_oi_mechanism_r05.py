#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.post_sweep_oi import (
    PostSweepOIConfig,
    add_attempt_pair_features,
    build_future_oi_labels,
    causal_align_oi,
    causal_audit,
    split_features_labels,
)


def _checkpoints() -> pd.DataFrame:
    times = pd.date_range("2025-01-01 08:06:00", periods=12, freq="1min")
    return pd.DataFrame({
        "checkpoint_id": [f"E_C{i:03d}" for i in range(1, 13)],
        "zone_event_id": "E",
        "event_kind": "swing_zone_sweep",
        "period": "TEST",
        "event_pos": 1,
        "event_available_time": pd.Timestamp("2025-01-01 08:05:00"),
        "checkpoint_pos": np.arange(2, 14),
        "checkpoint_time": times - pd.Timedelta(minutes=1),
        "checkpoint_available_time": times,
        "elapsed_bars": np.arange(1, 13),
        "new_low_attempt_flag": True,
        "new_low_attempt_index": np.arange(1, 13),
        "price_change_5m_bp": -5.0,
        "delta_ratio_5m": -0.2,
        "downside_bp_per_sell_million_1m": np.linspace(2.0, 0.5, 12),
        "downside_bp_per_abs_negative_delta_million_1m": np.linspace(4.0, 1.0, 12),
        "delta_ratio_1m": -0.2,
        "sell_share_1m": 0.6,
        "close_vs_running_low_bp": 5.0,
        "new_low_extension_bp": 1.0,
    })


def _oi() -> pd.DataFrame:
    timestamps = pd.date_range("2025-01-01 07:45:00", periods=30, freq="5min")
    out = pd.DataFrame({
        "timestamp": timestamps,
        "available_time": timestamps + pd.Timedelta(minutes=1),
        "sum_open_interest": 1_000_000.0 + np.arange(30) * 1000.0,
        "sum_open_interest_value": 3_000_000_000.0 + np.arange(30) * 1_000_000.0,
        "taker_volume_imbalance": -0.1,
        "top_trader_account_long_share": 0.6,
        "top_trader_position_long_share": 0.65,
        "global_account_long_share": 0.55,
    })
    for tag, shift in (("5m", 1), ("15m", 3), ("30m", 6), ("1h", 12), ("4h", 48), ("1d", 288)):
        out[f"oi_base_change_{tag}"] = out["sum_open_interest"].pct_change(shift)
        out[f"oi_usd_change_{tag}"] = out["sum_open_interest_value"].pct_change(shift)
        out[f"oi_baseline_age_seconds_{tag}"] = pd.Timedelta(tag).total_seconds()
    return out


def _labels(checkpoints: pd.DataFrame) -> pd.DataFrame:
    out = checkpoints[["checkpoint_id", "zone_event_id", "period", "elapsed_bars"]].copy()
    out["future_mfe_60m"] = 0.01
    out["future_mae_60m"] = -0.002
    out["future_close_return_60m"] = 0.005
    out["future_no_lower_low_60m"] = True
    out["future_reversal_dominant_60m"] = True
    out["future_continuation_dominant_60m"] = False
    return out


def test_causal_alignment_uses_published_row_only() -> None:
    cfg = PostSweepOIConfig(sample_rows=100).validate()
    aligned = causal_align_oi(_checkpoints(), _oi(), cfg)
    assert aligned["oi_context_present"].all()
    assert (aligned["oi_available_time"] <= aligned["checkpoint_available_time"]).all()
    first = aligned.iloc[0]
    assert first["checkpoint_available_time"] == pd.Timestamp("2025-01-01 08:06:00")
    assert first["oi_available_time"] == pd.Timestamp("2025-01-01 08:06:00")


def test_alignment_does_not_cross_long_gap() -> None:
    cfg = PostSweepOIConfig(alignment_tolerance="10min", sample_rows=100).validate()
    oi = _oi().iloc[:2].copy()
    aligned = causal_align_oi(_checkpoints(), oi, cfg)
    assert not aligned["oi_context_present"].any()


def test_future_oi_labels_use_future_publication_axis() -> None:
    cfg = PostSweepOIConfig(future_oi_horizons=(15, 30), sample_rows=100).validate()
    aligned = causal_align_oi(_checkpoints(), _oi(), cfg)
    future = build_future_oi_labels(aligned, _oi(), cfg)
    assert future["future_oi_label_complete_15m"].all()
    assert (future["future_oi_available_time_15m"] <= aligned["checkpoint_available_time"] + pd.Timedelta(minutes=15)).all()
    assert (future["future_oi_base_change_15m"] > 0).all()


def test_attempt_pair_features_detect_rising_oi_and_weaker_impact() -> None:
    cfg = PostSweepOIConfig(sample_rows=100).validate()
    aligned = causal_align_oi(_checkpoints(), _oi(), cfg)
    paired = add_attempt_pair_features(aligned)
    candidates = paired.loc[
        paired["prior_attempt_checkpoint_id"].notna()
        & paired["impact_weaker_vs_prior_attempt"].eq(True)
        & paired["oi_rising_since_prior_attempt"].eq(True)
    ]
    assert not candidates.empty
    assert (candidates["attempt_mechanism_state"] == "OI_UP_IMPACT_WEAKER").all()


def test_feature_label_physical_separation() -> None:
    cfg = PostSweepOIConfig(future_oi_horizons=(15,), sample_rows=100).validate()
    checkpoints = _checkpoints()
    aligned = add_attempt_pair_features(causal_align_oi(checkpoints, _oi(), cfg))
    future = build_future_oi_labels(aligned, _oi(), cfg)
    features, labels = split_features_labels(aligned, _labels(checkpoints), future)
    assert not any(name.startswith("future_") for name in features.columns)
    assert "future_oi_base_change_15m" in labels.columns
    audit = causal_audit(features, labels)
    assert int(audit["violations"].sum()) == 0


def test_script_self_test_runs() -> None:
    path = Path("research/liquidity/05_post_sweep_binance_oi_mechanism_study.py")
    spec = importlib.util.spec_from_file_location("r05_script", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_self_test()
