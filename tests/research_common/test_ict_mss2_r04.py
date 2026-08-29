from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r04 import (
    R04Config,
    build_multi_horizon_path_labels,
    build_rule_horizon_scoreboard,
    build_unique_opportunity_features,
    first_qualifying_opportunities,
    r04_causal_audit,
)


def _bars(highs, lows, closes=None, start="2023-01-01 00:00:00"):
    idx = pd.date_range(start, periods=len(highs), freq="1min")
    if closes is None:
        closes = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame({"open": closes, "high": highs, "low": lows, "close": closes, "volume": 1.0}, index=idx)


def _opportunity(entry_pos=0, entry=100.0, stop=99.0, target4h=102.0):
    return pd.DataFrame([
        {
            "trade_event_id": "R02_5M_TRADE_1",
            "episode_id": "EP1",
            "stage_id": "ST1",
            "signal_available_time": pd.Timestamp("2023-01-01 00:00:00") + pd.Timedelta(minutes=entry_pos),
            "entry_time": pd.Timestamp("2023-01-01 00:00:00") + pd.Timedelta(minutes=entry_pos),
            "entry_pos_1m": entry_pos,
            "entry_price": entry,
            "stop_price": stop,
            "target_htf240_price": target4h,
            "pool_n_bucket": "4+",
            "contains_4h_pool_flag": 1,
            "contains_lt_pool_flag": 1,
            "contains_it_plus_pool_flag": 1,
            "contains_multi_tf_pool_flag": 1,
            "contains_external50_pool_flag": 0,
            "contains_clean_pool_flag": 1,
            "contains_structural_key_pool_flag": 1,
            "ict_price_pools_cum": 4,
            "ict_structural_key_pools_cum": 1,
            "structural_risk_return": 0.01,
        }
    ])


def test_same_bar_target_and_stop_is_pessimistic_stop_first():
    # Entry=100.  Same first bar touches +0.5% and stop 99.0.
    bars = _bars([100.6] + [100.2] * 20, [98.8] + [99.9] * 20, [100.0] * 21)
    labels, audit = build_multi_horizon_path_labels(
        _opportunity(), bars, R04Config(path_horizons_minutes=(1, 5), post_4h_horizons_minutes=(5,), max_horizon_minutes=20)
    )
    assert labels.loc[0, "tp_0p5_before_stop_14d_flag"] == 0
    assert labels.loc[0, "tp_0p5_vs_stop_outcome_14d"] == "stop"
    assert audit["violations"].sum() == 0


def test_target_ladder_and_partial_fraction_are_measured_without_time_exit():
    highs = [100.1, 100.6, 101.1, 102.1, 103.2, 105.2] + [105.2] * 20
    lows = [99.8] * len(highs)
    bars = _bars(highs, lows, [100.0, 100.5, 101.0, 102.0, 103.0, 105.0] + [105.0] * 20)
    cfg = R04Config(path_horizons_minutes=(6, 12, 20), post_4h_horizons_minutes=(6, 12), max_horizon_minutes=20)
    labels, _ = build_multi_horizon_path_labels(_opportunity(target4h=102.0), bars, cfg)
    assert labels.loc[0, "tp_0p5_before_stop_14d_flag"] == 1
    assert labels.loc[0, "tp_1_before_stop_14d_flag"] == 1
    assert labels.loc[0, "tp_3_before_stop_14d_flag"] == 1
    assert labels.loc[0, "tp_5_before_stop_14d_flag"] == 1
    # (1% structural risk + 2x*0.11% cost) / (0.5% target + 1% risk)
    expected = (0.01 + 0.0022) / (0.005 + 0.01)
    assert np.isclose(labels.loc[0, "partial_fraction_at_0p5_to_cover_original_stop_cost2x"], expected)


def test_right_edge_horizon_is_censored_not_partial_label():
    bars = _bars([100.2] * 5, [99.8] * 5, [100.0] * 5)
    opp = _opportunity(entry_pos=3)
    labels, _ = build_multi_horizon_path_labels(
        opp, bars, R04Config(path_horizons_minutes=(1, 3), post_4h_horizons_minutes=(1,), max_horizon_minutes=3)
    )
    assert labels.loc[0, "label_complete_1m_flag"] == 1
    assert labels.loc[0, "label_complete_3m_flag"] == 0
    assert np.isnan(labels.loc[0, "mfe_3m"])
    assert np.isnan(labels.loc[0, "short_0p5_6h_flag"])


def test_post_4h_continuation_starts_next_bar():
    # 4H target=102 touched on bar 1. Bar 1 itself spikes to 110, but next bars
    # only reach 103; additional MFE must ignore the same target-touch bar.
    bars = _bars([100.1, 110.0, 103.0, 102.5, 102.3], [99.8, 101.5, 101.8, 101.9, 102.0], [100, 102, 102.5, 102.2, 102.1])
    labels, _ = build_multi_horizon_path_labels(
        _opportunity(target4h=102.0), bars, R04Config(path_horizons_minutes=(5,), post_4h_horizons_minutes=(2,), max_horizon_minutes=5)
    )
    assert labels.loc[0, "htf240_target_before_stop_flag"] == 1
    expected = 103.0 / 102.0 - 1.0
    assert np.isclose(labels.loc[0, "post4h_additional_mfe_2m"], expected)


def test_first_qualifying_is_one_stage_per_episode():
    f = pd.DataFrame([
        {"episode_id": "E1", "trade_event_id": "T1", "signal_available_time": "2023-01-01 00:05", "stage_id": "S1"},
        {"episode_id": "E1", "trade_event_id": "T2", "signal_available_time": "2023-01-01 00:10", "stage_id": "S2"},
        {"episode_id": "E2", "trade_event_id": "T3", "signal_available_time": "2023-01-01 00:07", "stage_id": "S3"},
    ])
    f["signal_available_time"] = pd.to_datetime(f["signal_available_time"])
    out = first_qualifying_opportunities(f, pd.Series([True, True, True]))
    assert set(out.trade_event_id) == {"T1", "T3"}


def test_unique_opportunity_collapses_cohort_labels_to_flags():
    h = pd.DataFrame([
        {"trade_event_id": "R02_5M_TRADE_1", "stage_id": "S1", "episode_id": "E1", "hierarchy_cohort": "first_any_pool", "trade_direction": 1, "execution_minutes": 5, "trigger_type": "episode_reclaim", "entry_time": "2023-01-01 00:05", "entry_price": 100, "stop_price": 99, "risk_bps": 100, "ict_price_pools_cum": 4, "ict_structural_key_pools_cum": 1, "ict_it_plus_pools_cum": 1, "ict_lt_pools_cum": 1, "ict_htf240_pools_cum": 1, "ict_multi_tf_pools_cum": 1, "ict_external50_pools_cum": 0, "ict_clean_pools_cum": 1, "ict_strongest_pool_rank_cum": 3, "target_htf240_net_return_cost2x": 0.123},
        {"trade_event_id": "R02_5M_TRADE_1", "stage_id": "S1", "episode_id": "E1", "hierarchy_cohort": "first_key_plus_ge4_total", "trade_direction": 1, "execution_minutes": 5, "trigger_type": "episode_reclaim", "entry_time": "2023-01-01 00:05", "entry_price": 100, "stop_price": 99, "risk_bps": 100, "ict_price_pools_cum": 4, "ict_structural_key_pools_cum": 1, "ict_it_plus_pools_cum": 1, "ict_lt_pools_cum": 1, "ict_htf240_pools_cum": 1, "ict_multi_tf_pools_cum": 1, "ict_external50_pools_cum": 0, "ict_clean_pools_cum": 1, "ict_strongest_pool_rank_cum": 3, "target_htf240_net_return_cost2x": 0.123},
    ])
    stages = pd.DataFrame([{"stage_id": "S1", "episode_stage_no": 4, "sweep_pos_1m": 4, "episode_start_pos_1m": 0, "episode_elapsed_minutes": 4}])
    f = pd.DataFrame([{"trade_event_id": "R02_5M_TRADE_1", "entry_pos_1m": 5, "entry_time": "2023-01-01 00:05", "entry_price": 100, "stop_price": 99, "signal_available_time": "2023-01-01 00:05", "episode_start_time_1m": "2023-01-01 00:00"}])
    l = pd.DataFrame([{"trade_event_id": "R02_5M_TRADE_1", "target_htf240_price": 102.0}])
    out = build_unique_opportunity_features(h, stages, f, l)
    assert len(out) == 1
    assert out.loc[0, "cohort_first_any_pool_flag"] == 1
    assert out.loc[0, "cohort_first_key_plus_ge4_total_flag"] == 1
    assert out.loc[0, "contains_4h_pool_flag"] == 1
    assert out.loc[0, "strongest_ict_class"] == "LT"
    assert "target_htf240_net_return_cost2x" not in out.columns


def test_r04_audit_rejects_future_label_columns_in_features():
    feat = _opportunity()
    feat["mfe_60m"] = 0.01
    lab = pd.DataFrame({"trade_event_id": feat.trade_event_id})
    audit = r04_causal_audit(feat, lab)
    row = audit.loc[audit.check.eq("future_label_columns_absent_from_features")].iloc[0]
    assert row.violations == 1
