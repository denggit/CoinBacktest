from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r10 import (
    R10Config,
    attach_risk_sizing,
    build_structural_mss_upgrade_map,
    build_unified_reclaim_base,
    r10_causal_audit,
    select_single_position,
    simulate_unified_lifecycles,
)


def _bars(n: int = 20, start: str = "2026-01-01") -> pd.DataFrame:
    idx=pd.date_range(start,periods=n,freq="1min")
    px=np.full(n,100.0)
    return pd.DataFrame({"open":px.copy(),"high":px.copy(),"low":px.copy(),"close":px.copy(),"volume":1.0},index=idx)


def _r09_rows() -> pd.DataFrame:
    return pd.DataFrame([
        {"trade_event_id":"r1","episode_id":"e1","liquidity_side":"SSL","trigger_type":"episode_reclaim","execution_minutes":2,"entry_kind":"market_next_open","entry_time":"2026-01-01 00:02","signal_available_time":"2026-01-01 00:01","sweep_bar_time_1m":"2026-01-01 00:00","entry_price":100.0,"stop_price":99.0,"entry_pos_1m":2,"context_tier":"A_1H_context","root_physical_level_count":1},
        {"trade_event_id":"r2","episode_id":"e1","liquidity_side":"SSL","trigger_type":"mss_structural_market","execution_minutes":2,"entry_kind":"market_next_open","entry_time":"2026-01-01 00:06","signal_available_time":"2026-01-01 00:05","sweep_bar_time_1m":"2026-01-01 00:00","entry_price":101.0,"stop_price":99.0,"entry_pos_1m":6,"context_tier":"A_1H_context"},
        {"trade_event_id":"r3","episode_id":"e2","liquidity_side":"BSL","trigger_type":"episode_reclaim","execution_minutes":2,"entry_kind":"market_next_open","entry_time":"2026-01-01 00:04","signal_available_time":"2026-01-01 00:03","entry_price":100.0,"stop_price":99.0,"entry_pos_1m":4,"context_tier":"A_1H_context"},
    ])


def test_unified_base_is_ssl_2m_reclaim_only():
    b=build_unified_reclaim_base(_r09_rows(),execution_minutes=2)
    assert list(b["episode_id"])==["e1"]
    assert float(b.iloc[0]["initial_risk_return"])==0.01


def test_structural_mss_is_later_state_upgrade_only():
    d=_r09_rows(); b=build_unified_reclaim_base(d,execution_minutes=2)
    u=build_structural_mss_upgrade_map(b,d,execution_minutes=2)
    assert pd.Timestamp(u.iloc[0]["mss_upgrade_time"])==pd.Timestamp("2026-01-01 00:05")


def test_partial_same_bar_stop_first_is_pessimistic():
    bars=_bars()
    # entry at pos2; same pos4 trades through both 2R target=102 and stop=99 => stop first.
    bars.iloc[4,bars.columns.get_loc("high")]=103.0
    bars.iloc[4,bars.columns.get_loc("low")]=98.0
    base=build_unified_reclaim_base(_r09_rows(),execution_minutes=2)
    trail=pd.DataFrame(columns=["event_type","trail_tf_min","activation_time","anchor_price"])
    up=build_structural_mss_upgrade_map(base,_r09_rows(),execution_minutes=2)
    p=simulate_unified_lifecycles(base,bars,trail,up,config=R10Config(lifecycle_variants=("base75_2r_runner25",)),show_progress=False)
    assert int(p.iloc[0]["stop_before_base_flag"])==1
    assert int(p.iloc[0]["base_target_hit_flag"])==0


def test_break_even_only_after_2r_next_bar():
    bars=_bars()
    # hit 2R on pos4 without touching stop, then next bar gaps/prints under entry.
    bars.iloc[4,bars.columns.get_loc("high")]=102.5
    bars.iloc[4,bars.columns.get_loc("low")]=100.0
    bars.iloc[5,bars.columns.get_loc("open")]=99.8
    bars.iloc[5,bars.columns.get_loc("low")]=99.7
    bars.iloc[5,bars.columns.get_loc("high")]=100.0
    bars.iloc[5,bars.columns.get_loc("close")]=99.9
    base=build_unified_reclaim_base(_r09_rows(),execution_minutes=2)
    trail=pd.DataFrame(columns=["event_type","trail_tf_min","activation_time","anchor_price"])
    up=build_structural_mss_upgrade_map(base,_r09_rows(),execution_minutes=2)
    p=simulate_unified_lifecycles(base,bars,trail,up,config=R10Config(lifecycle_variants=("base75_2r_runner25",)),show_progress=False)
    r=p.iloc[0]
    assert int(r["base_target_hit_flag"])==1
    assert pd.Timestamp(r["base_exit_time"])==bars.index[4]
    assert pd.Timestamp(r["exit_time"])==bars.index[5]


def test_risk_sizing_never_exceeds_budget_or_notional_cap():
    p=pd.DataFrame([{"episode_id":"e","trade_event_id":"t","context_tier":"A_1H_context","entry_time":"2026-01-01","entry_price":100.0,"initial_stop_price":99.0,"initial_risk_return":0.01,"lifecycle_variant":"base75_2r_runner25","base_fraction":0.75,"runner_fraction":0.25,"base_target_hit_flag":1,"mss_upgrade_flag":1,"major_upgrade_flag":0,"holding_minutes":60,"gross_return_unit_notional":0.02,"exit_time":"2026-01-01 01:00","exit_price":102.0}])
    s=attach_risk_sizing(p,config=R10Config(cost_scales=(2.0,),risk_schedules=(("x",0.003,0.001,0.0075,0.0075),)))
    assert float(s.iloc[0]["worst_case_initial_risk"]) <= float(s.iloc[0]["risk_budget_fraction"])+1e-12
    assert float(s.iloc[0]["notional_multiple"]) <= 3.0


def test_single_position_skips_overlap():
    s=pd.DataFrame([
        {"episode_id":"e1","entry_time":"2026-01-01 00:00","exit_time":"2026-01-01 02:00","lifecycle_variant":"x","risk_schedule":"r","cost_scale":2.0},
        {"episode_id":"e2","entry_time":"2026-01-01 01:00","exit_time":"2026-01-01 03:00","lifecycle_variant":"x","risk_schedule":"r","cost_scale":2.0},
        {"episode_id":"e3","entry_time":"2026-01-01 03:00","exit_time":"2026-01-01 04:00","lifecycle_variant":"x","risk_schedule":"r","cost_scale":2.0},
    ])
    e,a=select_single_position(s)
    assert list(e["episode_id"])==["e1","e3"]
    assert int(a.iloc[0]["overlap_skipped"])==1


def test_r10_causal_audit_zero_on_valid_rows():
    b=build_unified_reclaim_base(_r09_rows(),execution_minutes=2)
    paths=pd.DataFrame([{"entry_time":"2026-01-01 00:02","mss_upgrade_flag":1,"mss_upgrade_time":"2026-01-01 00:05","major_upgrade_flag":1,"major_upgrade_time":"2026-01-01 00:10"}])
    sized=pd.DataFrame([{"risk_budget_fraction":0.0075,"worst_case_initial_risk":0.0075,"notional_multiple":0.75}])
    a=r10_causal_audit(b,paths,sized)
    assert int(a["violations"].sum())==0
