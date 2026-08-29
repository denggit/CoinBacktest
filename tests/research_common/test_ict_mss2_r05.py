from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from src.research_common.ict_mss2.r04 import build_rule_horizon_scoreboard
from src.research_common.ict_mss2.r05 import (
    R05Config,
    attach_initial_structural_stops,
    build_execution_swing_hierarchy,
    build_initial_stop_target_atlas,
    build_quality_entry_universe,
    simulate_structural_trailing,
)


def _bars_1m(opens, highs, lows, closes=None, start="2023-01-01 00:00:00"):
    if closes is None:
        closes = opens
    idx = pd.date_range(start, periods=len(opens), freq="1min")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": 1.0}, index=idx)


def test_r05_config_forbids_1m_trailing():
    with pytest.raises(ValueError):
        R05Config(trail_minutes=(1, 5, 15)).validate()


def test_recursive_itl_is_only_known_after_right_st_confirmation():
    # Seven 2m bars with ST lows at 1,3,5 and the center low at 3 lower than
    # the two neighboring ST lows => ITL. Expand each 2m bar to two 1m bars.
    lows2 = [101, 99, 101, 98, 101, 99.5, 101]
    highs2 = [103, 102, 103, 102, 103, 102.5, 103]
    opens, highs, lows, closes = [], [], [], []
    for lo, hi in zip(lows2, highs2):
        for _ in range(2):
            opens.append((lo + hi) / 2)
            highs.append(hi)
            lows.append(lo)
            closes.append((lo + hi) / 2)
    bars = _bars_1m(opens, highs, lows, closes)
    h = build_execution_swing_hierarchy(bars, 2)
    itl = h.loc[h["pivot_side"].eq("low") & h["ict_it_available_time"].notna()]
    assert not itl.empty
    row = itl.sort_values("level_price").iloc[0]
    assert row["level_price"] == pytest.approx(98.0)
    assert pd.Timestamp(row["ict_it_available_time"]) > pd.Timestamp(row["pivot_time"])


def test_quality_entry_universe_uses_same_first_stage_across_1m_2m_5m():
    stages = pd.DataFrame([
        {"stage_id": "S1", "episode_id": "E1", "sweep_pos_1m": 10, "sweep_bar_time_1m": "2023-01-01 00:10", "ict_price_pools_cum": 3, "ict_htf240_pools_cum": 1, "ict_lt_pools_cum": 0, "ict_it_plus_pools_cum": 1, "ict_structural_key_pools_cum": 1},
        {"stage_id": "S2", "episode_id": "E1", "sweep_pos_1m": 12, "sweep_bar_time_1m": "2023-01-01 00:12", "ict_price_pools_cum": 4, "ict_htf240_pools_cum": 1, "ict_lt_pools_cum": 1, "ict_it_plus_pools_cum": 1, "ict_structural_key_pools_cum": 1},
    ])
    trades = []
    for stage in ("S1", "S2"):
        for tf in (1, 2, 5):
            trades.append({
                "trade_event_id": f"T_{stage}_{tf}", "stage_id": stage, "episode_id": "E1",
                "trade_direction": 1, "trigger_type": "episode_reclaim", "execution_minutes": tf,
                "entry_time": f"2023-01-01 00:{10+tf:02d}", "entry_price": 100.0, "stop_price": 98.0,
                "entry_pos_1m": 15, "signal_available_time": f"2023-01-01 00:{10+tf:02d}",
            })
    out = build_quality_entry_universe(pd.DataFrame(trades), stages)
    n3 = out.loc[out["quality_rule"].eq("n3_4h")]
    assert set(n3["stage_id"]) == {"S1"}
    assert set(n3["execution_minutes"]) == {1, 2, 5}
    n4 = out.loc[out["quality_rule"].eq("n4_4h")]
    assert set(n4["stage_id"]) == {"S2"}


def test_initial_structural_stops_are_entry_time_only():
    bars = _bars_1m(
        [100, 99, 98.5, 99.0, 99.5, 100.0, 100.5],
        [100.2, 99.3, 98.8, 99.4, 99.8, 100.3, 100.8],
        [99.8, 98.8, 98.0, 98.7, 99.2, 99.7, 100.1],
        [100, 99, 98.4, 99.2, 99.6, 100.1, 100.6],
    )
    opp = pd.DataFrame([{
        "quality_rule": "n3_4h", "episode_id": "E1", "stage_id": "S1", "trade_event_id": "T1",
        "execution_minutes": 2, "entry_pos_1m": 6, "entry_time": bars.index[6], "entry_price": 100.5,
        "stop_price": 97.9, "sweep_extreme_stage": 98.0, "sweep_pos_1m": 2,
        "signal_bar_time": bars.index[4], "signal_available_time": bars.index[6], "episode_start_time_1m": bars.index[1],
    }])
    out = attach_initial_structural_stops(opp, bars, hierarchy_by_tf={}, config=R05Config())
    assert out.loc[0, "stop_episode_extreme"] == pytest.approx(97.9)
    assert out.loc[0, "stop_qualifying_stage_extreme"] < 98.0
    # Reclaim-leg low only uses sweep_pos..entry-1, never future bar 6 low.
    assert out.loc[0, "stop_reclaim_leg_extreme"] == pytest.approx(98.0 * (1 - 0.0002))


def test_initial_stop_target_same_bar_is_stop_first_and_mae_is_conservative():
    bars = _bars_1m([100, 100, 100], [100.7, 100.2, 100.2], [99.4, 99.8, 99.8], [100, 100, 100])
    opp = pd.DataFrame([{
        "quality_rule": "n3_4h", "episode_id": "E1", "stage_id": "S1", "trade_event_id": "T1",
        "execution_minutes": 1, "entry_pos_1m": 0, "entry_time": bars.index[0], "entry_price": 100.0,
        "stop_episode_extreme": 99.5,
    }])
    outcomes, mae = build_initial_stop_target_atlas(opp, bars, config=R05Config(fixed_target_returns=(0.005,), max_horizon_minutes=1))
    assert len(outcomes) == 1
    assert outcomes.loc[0, "stop_first_flag"] == 1
    assert outcomes.loc[0, "target_first_flag"] == 0
    assert mae.empty


def test_structural_trailing_new_stop_is_not_retroactive_and_only_moves_up():
    bars = _bars_1m(
        [100, 101, 102, 103, 104, 104],
        [100.5, 101.5, 102.5, 103.5, 104.5, 104.2],
        [99.8, 100.8, 100.2, 102.0, 103.0, 102.9],
        [100.2, 101.2, 102.2, 103.2, 104.2, 103.5],
    )
    opp = pd.DataFrame([{
        "quality_rule": "n4_4h_or_lt", "episode_id": "E1", "trade_event_id": "T1",
        "execution_minutes": 2, "entry_pos_1m": 0, "entry_time": bars.index[0], "entry_price": 100.0,
        "stop_episode_extreme": 99.0,
    }])
    # First ITL anchor is confirmed/usable at bar 3 start. Bar 2 traded below
    # that future stop, but it must not retroactively stop the trade. Second
    # anchor is lower and must be ignored because stops never loosen.
    events = pd.DataFrame([
        {"trail_tf_min": 5, "event_type": "itl", "activation_time": bars.index[3], "anchor_time": bars.index[1], "anchor_price": 101.0},
        {"trail_tf_min": 5, "event_type": "itl", "activation_time": bars.index[4], "anchor_time": bars.index[2], "anchor_price": 100.5},
    ])
    out = simulate_structural_trailing(opp, bars, events, strategies=("itl_5m",), config=R05Config(max_horizon_minutes=5))
    assert len(out) == 1
    assert out.loc[0, "trail_updates"] == 1
    assert out.loc[0, "final_stop_price"] > 100.9
    # The trade cannot have exited on bar 2 before the anchor was available.
    assert out.loc[0, "exit_pos_1m"] != 2


def test_r04_scoreboard_does_not_emit_fragmentation_warning_on_wide_labels():
    features = pd.DataFrame([{
        "trade_event_id": "T1", "episode_id": "E1", "stage_id": "S1", "signal_available_time": pd.Timestamp("2023-01-01"),
        "entry_time": pd.Timestamp("2023-01-01"), "structural_risk_return": 0.01,
        "ict_price_pools_cum": 4, "ict_structural_key_pools_cum": 1,
        "contains_4h_pool_flag": 1, "contains_lt_pool_flag": 1, "contains_it_plus_pool_flag": 1,
    }])
    labels = pd.DataFrame({"trade_event_id": ["T1"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        for i in range(120):
            labels[f"dummy_{i}"] = i
        for c in ["short_0p5_6h_flag", "short_0p75_12h_flag", "medium_1p5_1d_flag", "medium_2p0_2d_flag", "swing_3p0_3d_flag", "major_5p0_7d_flag"]:
            labels[c] = 1
        for tok in ["0p5", "1", "2", "3", "5"]:
            labels[f"tp_{tok}_net_return_cost2x"] = 0.01
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_rule_horizon_scoreboard(features, labels, months=1.0)
    assert not any("highly fragmented" in str(w.message).lower() for w in caught)


def test_exclusive_opportunity_buckets_keep_long_tail_out_of_short_cohort():
    from src.research_common.ict_mss2.r05 import build_exclusive_opportunity_buckets, summarize_exclusive_opportunity_buckets

    bars = _bars_1m(
        [100, 100, 100, 100, 100, 100],
        [100.2, 100.6, 100.8, 101.2, 103.2, 105.3],
        [99.9, 99.8, 99.8, 99.7, 99.6, 99.5],
        [100, 100.4, 100.7, 101.0, 103.0, 105.0],
    )
    opps = pd.DataFrame([
        {
            "quality_rule": "short_case", "episode_id": "E1", "stage_id": "S1", "trade_event_id": "T1",
            "execution_minutes": 1, "entry_pos_1m": 0, "entry_time": bars.index[0], "entry_price": 100.0,
            "stop_episode_extreme": 99.0,
        },
        {
            "quality_rule": "major_case", "episode_id": "E2", "stage_id": "S2", "trade_event_id": "T2",
            "execution_minutes": 1, "entry_pos_1m": 0, "entry_time": bars.index[0], "entry_price": 100.0,
            "stop_episode_extreme": 99.0,
        },
    ])
    # Give the short case only the first three bars; keep the major case full.
    short_bars = bars.iloc[:3].copy()
    short = build_exclusive_opportunity_buckets(
        opps.iloc[[0]], short_bars, config=R05Config(max_horizon_minutes=2)
    )
    major = build_exclusive_opportunity_buckets(
        opps.iloc[[1]], bars, config=R05Config(max_horizon_minutes=5)
    )
    labels = pd.concat([short, major], ignore_index=True)
    assert labels.loc[labels["trade_event_id"].eq("T1"), "opportunity_bucket"].iloc[0] == "short_0p3_1p0"
    assert labels.loc[labels["trade_event_id"].eq("T2"), "opportunity_bucket"].iloc[0] == "major_ge_5p0"
    summary, _ = summarize_exclusive_opportunity_buckets(labels)
    short_summary = summary.loc[summary["opportunity_bucket"].eq("short_0p3_1p0")]
    major_summary = summary.loc[summary["opportunity_bucket"].eq("major_ge_5p0")]
    assert int(short_summary["opportunities"].sum()) == 1
    assert int(major_summary["opportunities"].sum()) == 1


def test_exclusive_bucket_same_bar_stop_does_not_credit_same_bar_high():
    from src.research_common.ict_mss2.r05 import build_exclusive_opportunity_buckets

    # Entry bar prints both +5% high and the thesis stop. Conservative semantics
    # must treat the stop as first and exclude that bar's high from attainable MFE.
    bars = _bars_1m([100, 100, 100], [105.5, 100.2, 100.1], [98.5, 99.8, 99.9], [100, 100, 100])
    opp = pd.DataFrame([{
        "quality_rule": "q", "episode_id": "E", "stage_id": "S", "trade_event_id": "T",
        "execution_minutes": 1, "entry_pos_1m": 0, "entry_time": bars.index[0], "entry_price": 100.0,
        "stop_episode_extreme": 99.0,
    }])
    out = build_exclusive_opportunity_buckets(opp, bars, config=R05Config(max_horizon_minutes=1))
    assert out.loc[0, "opportunity_bucket"] == "under_0p3"
    assert out.loc[0, "max_favorable_return_before_thesis_stop"] == pytest.approx(0.0)


def test_initial_stop_hierarchy_lookup_preserves_causal_it_to_lt_upgrade():
    from src.research_common.ict_mss2.r05 import _KnownLowLookup

    h = pd.DataFrame([
        {
            "pivot_side": "low", "pivot_time": pd.Timestamp("2023-01-01 00:10"), "level_price": 99.0,
            "ict_it_available_time": pd.Timestamp("2023-01-01 00:15"),
            "ict_lt_available_time": pd.Timestamp("2023-01-01 00:30"),
        },
        {
            "pivot_side": "low", "pivot_time": pd.Timestamp("2023-01-01 00:20"), "level_price": 100.0,
            "ict_it_available_time": pd.Timestamp("2023-01-01 00:40"),
            "ict_lt_available_time": pd.NaT,
        },
    ])
    lookup = _KnownLowLookup.from_hierarchy(h, ("IT", "LT"))
    # At 00:25 the later pivot is not yet confirmed, so the 00:10 ITL is used.
    r1 = lookup.latest(at_time=pd.Timestamp("2023-01-01 00:25"), min_pivot_time=pd.Timestamp("2023-01-01 00:00"))
    assert r1 is not None and r1[0] == pytest.approx(99.0) and r1[1] == "IT"
    # Once the same 00:10 pivot has causally upgraded to LT, LT wins for that pivot.
    r2 = lookup.latest(at_time=pd.Timestamp("2023-01-01 00:35"), min_pivot_time=pd.Timestamp("2023-01-01 00:00"))
    assert r2 is not None and r2[0] == pytest.approx(99.0) and r2[1] == "LT"
    # Once the newer 00:20 pivot is causally known, recency wins as in the old sort semantics.
    r3 = lookup.latest(at_time=pd.Timestamp("2023-01-01 00:45"), min_pivot_time=pd.Timestamp("2023-01-01 00:00"))
    assert r3 is not None and r3[0] == pytest.approx(100.0) and r3[1] == "IT"
    # Episode-local lower bound still excludes old structure.
    assert lookup.latest(at_time=pd.Timestamp("2023-01-01 00:35"), min_pivot_time=pd.Timestamp("2023-01-01 00:11")) is None
