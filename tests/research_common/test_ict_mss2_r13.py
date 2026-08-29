from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r13 import (
    R13Config,
    attach_reversal_quality_features,
    build_entry_candidate_outcomes,
    build_feature_bin_atlas,
    data_coverage_audit,
    prepare_reversal_comparison_universe,
    r13_causal_audit,
)


def _bars(n: int = 500, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min")
    x = np.full(n, 100.0)
    return pd.DataFrame({"open": x, "high": x + 0.1, "low": x - 0.1, "close": x, "volume": 1.0}, index=idx)


def _path(time: str, outcome: str = "direct_opposite_delivery", side: str = "SSL") -> dict[str, object]:
    t = pd.Timestamp(time)
    long = side == "SSL"
    return {
        "root_event_id": f"e_{t:%Y%m%d%H%M}_{side}", "root_sweep_time": t,
        "root_sweep_available_time": t + pd.Timedelta(minutes=1), "path_start_time": t + pd.Timedelta(minutes=1),
        "root_side": side, "path_outcome": outcome, "same_bar_two_sided_root_flag": 0,
        "opposite_1_available_flag": 1, "deeper_same_side_available_flag": 1,
        "next_open_price": 100.0, "next_open_time": t + pd.Timedelta(minutes=1),
        "opposite_1_touch_price": 105.0 if long else 95.0,
        "deeper_same_side_touch_price": 95.0 if long else 105.0,
        "opposite_1_touch_time": t + pd.Timedelta(minutes=120) if outcome == "direct_opposite_delivery" else pd.NaT,
        "deeper_same_side_touch_time": t + pd.Timedelta(minutes=120) if outcome != "direct_opposite_delivery" else pd.NaT,
        "path_horizon_minutes": 300,
        "root_zone_low": 99.0, "root_zone_high": 101.0,
        "root_sweep_depth_bps": 10.0, "root_rejection_wick_share": 0.5,
        "root_reversal_close_location": 0.5, "root_same_bar_full_reclaim_flag": 0,
        "root_oldest_age_days": 5.0, "pre_sweep_ret_15m": -0.01 if long else 0.01,
        "reclaim_available_time": t + pd.Timedelta(minutes=5), "reclaim_delay_min": 5.0,
    }


def test_direct_comparison_treats_cascade_as_failure_and_seals_holdout():
    rows = pd.DataFrame([
        _path("2024-01-01 01:00", "direct_opposite_delivery"),
        _path("2024-01-02 01:00", "cascade_then_opposite_delivery"),
        _path("2025-08-02 01:00", "direct_opposite_delivery"),
    ])
    q, seal = prepare_reversal_comparison_universe(rows)
    assert list(q["direct_reversal_label"]) == [1, 0]
    assert set(q["research_split"]) == {"discovery"}
    assert int(seal.iloc[0]["available_holdout_rows_in_r12"]) == 1
    assert int(seal.iloc[0]["included_in_r13_outputs"]) == 0


def test_embargo_rows_are_excluded_even_when_holdout_is_sealed():
    q, _ = prepare_reversal_comparison_universe(pd.DataFrame([
        _path("2025-06-30 01:00"), _path("2025-07-15 01:00"), _path("2025-08-15 01:00")
    ]))
    assert len(q) == 1
    assert q.iloc[0]["research_split"] == "validation"


def test_early_15m_features_do_not_change_from_later_bar_mutation():
    b = _bars()
    t = b.index[100]
    b.loc[t, "low"] = 98.5
    q, _ = prepare_reversal_comparison_universe(pd.DataFrame([_path(str(t))]))
    a = attach_reversal_quality_features(b, q)
    b2 = b.copy()
    b2.loc[b2.index[140], ["open", "high", "low", "close"]] = [100.0, 150.0, 50.0, 120.0]
    z = attach_reversal_quality_features(b2, q)
    cols = [c for c in a.columns if c.startswith("early_15m_")]
    pd.testing.assert_series_equal(a.loc[0, cols], z.loc[0, cols], check_names=False)


def test_mss_quality_becomes_available_after_break_bar_close():
    b = _bars()
    t = b.index[100]
    b.loc[t, "low"] = 98.5
    # post-sweep STH at 01:43, right confirmation 01:44, break 01:45.
    b.loc[b.index[102], ["open", "high", "low", "close"]] = [100.0, 101.0, 99.8, 100.6]
    b.loc[b.index[103], ["open", "high", "low", "close"]] = [100.6, 102.0, 100.4, 101.5]
    b.loc[b.index[104], ["open", "high", "low", "close"]] = [101.5, 101.7, 100.8, 101.0]
    b.loc[b.index[105], ["open", "high", "low", "close"]] = [101.0, 102.5, 100.9, 102.2]
    q, _ = prepare_reversal_comparison_universe(pd.DataFrame([_path(str(t))]))
    x = attach_reversal_quality_features(b, q)
    assert x.iloc[0]["mss_1m_available_time"] == b.index[105] + pd.Timedelta(minutes=1)
    assert float(x.iloc[0]["mss_1m_break_distance_atr"]) > 0


def test_market_entry_is_next_eligible_bar_and_same_bar_both_is_stop_first():
    b = _bars()
    t = b.index[100]
    b.loc[t, "low"] = 98.5
    # next eligible entry bar trades through both frozen target and stop.
    b.loc[b.index[101], ["open", "high", "low", "close"]] = [100.0, 106.0, 94.0, 100.0]
    q, _ = prepare_reversal_comparison_universe(pd.DataFrame([_path(str(t))]))
    features = attach_reversal_quality_features(b, q)
    entries = build_entry_candidate_outcomes(b, features)
    root = entries.loc[entries["entry_model"].eq("root_next_open")].iloc[0]
    assert root["entry_time"] == b.index[101]
    assert root["outcome"] == "sl_first"
    assert float(root["gross_r"]) == -1.0


def test_response_15m_market_enters_after_all_15_post_root_bars_close():
    b = _bars()
    t = b.index[100]
    b.loc[t, "low"] = 98.5
    q, _ = prepare_reversal_comparison_universe(pd.DataFrame([_path(str(t))]))
    features = attach_reversal_quality_features(b, q)
    entries = build_entry_candidate_outcomes(b, features)

    response = entries.loc[entries["entry_model"].eq("response_15m_market")].iloc[0]
    # Post-root bars 101..115 are the 15 observed bars.  Their final close is
    # available at 116, whose open is the first causal execution price.
    assert features.iloc[0]["early_15m_available_time"] == b.index[116]
    assert response["signal_available_time"] == b.index[116]
    assert response["entry_time"] == b.index[116]


def test_early_response_bin_economics_use_response_entry_not_root_entry():
    n = 40
    features = pd.DataFrame({
        "root_event_id": [f"e{i}" for i in range(n)],
        "root_side": "SSL",
        "research_split": "discovery",
        "early_15m_path_efficiency": np.linspace(-1.0, 1.0, n),
        "direct_reversal_label": np.tile([0, 1], n // 2),
        "root_structural_rr": 2.0,
    })
    root_entries = pd.DataFrame({
        "root_event_id": features["root_event_id"],
        "entry_model": "root_next_open",
        "entry_status": "filled",
        "outcome": "tp_first",
        "net_return_cost2x": 0.10,
    })
    response_entries = pd.DataFrame({
        "root_event_id": features["root_event_id"],
        "entry_model": "response_15m_market",
        "entry_status": "filled",
        "outcome": "sl_first",
        "net_return_cost2x": -0.02,
    })

    definitions, bins, monotonicity = build_feature_bin_atlas(
        features,
        entries=pd.concat([root_entries, response_entries], ignore_index=True),
        feature_names=["early_15m_path_efficiency"],
    )

    assert set(definitions["causal_entry_model"]) == {"response_15m_market"}
    assert set(bins["causal_entry_model"]) == {"response_15m_market"}
    assert (bins["mean_causal_net_return_cost2x"] < 0).all()
    assert (bins["causal_net_pf_cost2x"] == 0.0).all()
    assert set(monotonicity["causal_entry_model"]) == {"response_15m_market"}


def test_data_coverage_and_r13_audit_are_zero_for_valid_rows():
    b = _bars()
    t = b.index[100]
    b.loc[t, "low"] = 98.5
    b.loc[b.index[220], "high"] = 106.0
    q, _ = prepare_reversal_comparison_universe(pd.DataFrame([_path(str(t))]))
    features = attach_reversal_quality_features(b, q)
    entries = build_entry_candidate_outcomes(b, features)
    coverage = data_coverage_audit(b, requested_start=b.index.min(), requested_end=b.index.max())
    assert int(coverage.loc[coverage["check"].eq("requested_end_covered"), "value"].iloc[0]) == 1
    audit = r13_causal_audit(features, entries, holdout_start=R13Config().holdout_start)
    assert int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum()) == 0
