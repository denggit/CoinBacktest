from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research_common.ict_mss2.r06 import (
    R06Config,
    attach_risk_sized_trade_returns,
    build_adaptive_base_universe,
    build_protected_structure_events,
    select_single_position_trades,
    simulate_adaptive_trade_paths,
)


def _bars(n=40, start="2023-01-01 00:00:00", price=100.0):
    idx = pd.date_range(start, periods=n, freq="1min")
    o = np.full(n, price, dtype=float)
    h = o + 0.2
    l = o - 0.2
    c = o.copy()
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1.0}, index=idx)


def test_r06_risk_tier_never_uses_future_n4_upgrade():
    f = pd.DataFrame([
        {
            "quality_rule": "n3_4h_or_lt", "episode_id": "E1", "stage_id": "S1", "trade_event_id": "T1",
            "execution_minutes": 2, "entry_time": "2023-01-01 00:10", "signal_available_time": "2023-01-01 00:10",
            "entry_price": 100.0, "stop_episode_extreme": 99.0, "entry_pos_1m": 10,
            "ict_price_pools_cum": 3, "ict_htf240_pools_cum": 1, "ict_lt_pools_cum": 0,
        },
        {
            "quality_rule": "n3_4h_or_lt", "episode_id": "E2", "stage_id": "S2", "trade_event_id": "T2",
            "execution_minutes": 2, "entry_time": "2023-01-01 01:10", "signal_available_time": "2023-01-01 01:10",
            "entry_price": 100.0, "stop_episode_extreme": 99.0, "entry_pos_1m": 70,
            "ict_price_pools_cum": 4, "ict_htf240_pools_cum": 1, "ict_lt_pools_cum": 1,
        },
    ])
    out = build_adaptive_base_universe(f)
    assert out.loc[out.trade_event_id.eq("T1"), "setup_tier"].iloc[0] == "B"
    assert out.loc[out.trade_event_id.eq("T2"), "setup_tier"].iloc[0] == "A_plus"


def test_protected_ltl_waits_for_later_higher_high_close():
    bars = _bars(35)
    # Build six 5m blocks with known closes/highs. The LTL is already known at
    # 00:15; frozen high through that point is 102. A later 5m close >102 occurs
    # only on the bar ending 00:25.
    for i in range(0, 35, 5):
        block = slice(i, min(i + 5, 35))
        bars.iloc[block, bars.columns.get_loc("open")] = 100.0
        bars.iloc[block, bars.columns.get_loc("close")] = 100.0
        bars.iloc[block, bars.columns.get_loc("high")] = 101.0
        bars.iloc[block, bars.columns.get_loc("low")] = 99.0
    bars.iloc[5:15, bars.columns.get_loc("high")] = 102.0
    bars.iloc[20:25, bars.columns.get_loc("close")] = 103.0
    bars.iloc[20:25, bars.columns.get_loc("high")] = 103.2
    events = pd.DataFrame([{
        "trail_tf_min": 5, "event_type": "ltl", "activation_time": pd.Timestamp("2023-01-01 00:15"),
        "anchor_time": pd.Timestamp("2023-01-01 00:05"), "anchor_price": 98.5,
        "bullish_fvg_flag": 0,
    }])
    out = build_protected_structure_events(bars, events)
    row = out.loc[out.event_type.eq("protected_ltl_5m_hh")].iloc[0]
    assert pd.Timestamp(row.promotion_time) > pd.Timestamp(row.candidate_activation_time)
    assert float(row.promotion_level) >= 102.0


def test_protected_addon_is_after_entry_and_not_averaging_down():
    bars = _bars(12)
    bars.loc[bars.index[1:9], "low"] = 100.2
    bars.loc[bars.index[1:9], "high"] = [100.5, 101, 102, 103, 104, 104.5, 105, 105.5]
    bars.loc[bars.index[1:9], "close"] = bars.loc[bars.index[1:9], "high"] - 0.1
    # Promotion bar opens above the promoted stop, so add-on is eligible.
    bars.loc[bars.index[6], "open"] = 104.0
    # Exit later at promoted stop.
    bars.loc[bars.index[10], "open"] = 103.0
    bars.loc[bars.index[10], "low"] = 102.0
    opp = pd.DataFrame([{
        "episode_id": "E", "trade_event_id": "T", "stage_id": "S", "execution_minutes": 2,
        "setup_tier": "B", "entry_time": bars.index[0], "entry_pos_1m": 0, "entry_price": 100.0,
        "stop_episode_extreme": 98.0,
    }])
    prot = pd.DataFrame([{
        "trail_tf_min": 5, "event_type": "protected_ltl_5m_hh", "promotion_time": bars.index[6],
        "candidate_activation_time": bars.index[4], "anchor_time": bars.index[2], "anchor_price": 102.5,
        "promotion_level": 103.0, "promotion_reason": "higher_high_close_after_anchor_known",
    }])
    old = pd.DataFrame(columns=["trail_tf_min", "event_type", "activation_time", "anchor_time", "anchor_price"])
    out = simulate_adaptive_trade_paths(
        opp, bars, old, prot, management_variants=("protected_ltl5",),
        config=R06Config(), show_progress=False,
    )
    assert len(out) == 1
    assert int(out.loc[0, "addon_pos_1m"]) == 6
    assert float(out.loc[0, "addon_price"]) >= 100.0
    assert pd.Timestamp(out.loc[0, "addon_time"]) > pd.Timestamp(out.loc[0, "entry_time"])


def test_risk_recycling_addon_stays_inside_setup_budget():
    path = pd.DataFrame([{
        "episode_id": "E", "trade_event_id": "T", "stage_id": "S", "execution_minutes": 2,
        "setup_tier": "B", "entry_time": pd.Timestamp("2023-01-01"), "entry_pos_1m": 0,
        "entry_price": 100.0, "initial_stop_price": 99.0, "initial_risk_return": 0.01,
        "management_variant": "protected_ltl5", "addon_variant": "risk_recycle_protected_ltl5",
        "trail_updates": 1, "protected_updates": 1, "first_promotion_minutes": 10,
        "major_state_reached_flag": 0, "major_state_minutes": np.nan, "last_trail_event_type": "protected_ltl_5m_hh",
        "final_stop_price": 102.0, "addon_pos_1m": 10, "addon_time": pd.Timestamp("2023-01-01 00:10"),
        "addon_price": 104.0, "addon_stop_price": 102.0, "addon_reason": "protected_5m_ltl_after_hh",
        "exit_pos_1m": 20, "exit_time": pd.Timestamp("2023-01-01 00:20"), "exit_price": 103.0,
        "holding_minutes": 20, "right_edge_open_flag": 0, "mfe_until_exit_or_data_end": 0.05,
        "reached_3pct_before_exit_flag": 1, "reached_5pct_before_exit_flag": 1, "reached_10pct_before_exit_flag": 0,
    }])
    cfg = R06Config(risk_schedules=(("only", 0.01, 0.01, 0.01),), cost_scales=(1.0,))
    out = attach_risk_sized_trade_returns(path, config=cfg)
    r = out.iloc[0]
    # Base stop has already moved above entry, so the add-on may use at most the
    # full 1% setup risk budget against its own common stop.
    addon_risk = float(r.addon_notional_multiple) * ((104.0 - 102.0) / 104.0)
    assert addon_risk <= 0.0100001
    assert float(r.total_notional_multiple) <= cfg.max_notional_multiple + 1e-12


def test_single_position_allocator_skips_overlapping_episode():
    df = pd.DataFrame([
        {"execution_minutes": 2, "management_variant": "m", "addon_variant": "none", "risk_schedule": "r", "cost_scale": 2.0,
         "entry_time": pd.Timestamp("2023-01-01 00:00"), "exit_time": pd.Timestamp("2023-01-01 02:00"), "episode_id": "E1"},
        {"execution_minutes": 2, "management_variant": "m", "addon_variant": "none", "risk_schedule": "r", "cost_scale": 2.0,
         "entry_time": pd.Timestamp("2023-01-01 01:00"), "exit_time": pd.Timestamp("2023-01-01 03:00"), "episode_id": "E2"},
        {"execution_minutes": 2, "management_variant": "m", "addon_variant": "none", "risk_schedule": "r", "cost_scale": 2.0,
         "entry_time": pd.Timestamp("2023-01-01 02:30"), "exit_time": pd.Timestamp("2023-01-01 04:00"), "episode_id": "E3"},
    ])
    ex, audit = select_single_position_trades(df)
    assert list(ex.episode_id) == ["E1", "E3"]
    assert int(audit.loc[0, "overlap_skipped"]) == 1


def test_r06_risk_schedule_never_exceeds_two_percent():
    R06Config().validate()
    with pytest.raises(ValueError):
        R06Config(risk_schedules=(("bad", 0.01, 0.02, 0.03),)).validate()


def test_major_upgrade_switches_from_5m_to_slower_15m_structure():
    bars = _bars(8)
    # Hit +3% after the first protected 5m anchor but before the later 15m anchor.
    bars.loc[bars.index[1], ["open", "high", "low", "close"]] = [100.8, 101.5, 100.5, 101.2]
    bars.loc[bars.index[2], ["open", "high", "low", "close"]] = [102.0, 103.5, 101.8, 103.2]
    bars.loc[bars.index[3], ["open", "high", "low", "close"]] = [103.2, 104.0, 102.8, 103.8]
    bars.loc[bars.index[4], ["open", "high", "low", "close"]] = [104.0, 104.5, 103.0, 104.1]
    bars.loc[bars.index[6], ["open", "high", "low", "close"]] = [102.5, 102.8, 101.5, 101.8]
    opp = pd.DataFrame([{
        "episode_id": "E", "trade_event_id": "T", "stage_id": "S", "execution_minutes": 2,
        "setup_tier": "B", "entry_time": bars.index[0], "entry_pos_1m": 0,
        "entry_price": 100.0, "stop_episode_extreme": 95.0,
    }])
    prot = pd.DataFrame([
        {"trail_tf_min": 5, "event_type": "protected_ltl_5m_hh", "promotion_time": bars.index[1],
         "candidate_activation_time": bars.index[0], "anchor_time": bars.index[0], "anchor_price": 99.0,
         "promotion_level": 100.5, "promotion_reason": "higher_high_close_after_anchor_known"},
        {"trail_tf_min": 15, "event_type": "protected_ltl_15m_hh", "promotion_time": bars.index[4],
         "candidate_activation_time": bars.index[2], "anchor_time": bars.index[1], "anchor_price": 102.0,
         "promotion_level": 103.0, "promotion_reason": "higher_high_close_after_anchor_known"},
    ])
    old = pd.DataFrame(columns=["trail_tf_min", "event_type", "activation_time", "anchor_time", "anchor_price"])
    out = simulate_adaptive_trade_paths(
        opp, bars, old, prot, management_variants=("protected_ltl5_then_ltl15_major",),
        config=R06Config(), show_progress=False,
    )
    assert int(out.loc[0, "major_state_reached_flag"]) == 1
    assert out.loc[0, "last_trail_event_type"] == "protected_ltl_15m_hh"
    assert float(out.loc[0, "final_stop_price"]) > 101.9
