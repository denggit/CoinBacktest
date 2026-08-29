import numpy as np
import pandas as pd

from src.research_common.ict.premarket_mss_fvg import NY_TZ
from src.research_common.ict.semantic_consolidation import (
    LiquidityConsumptionConfig,
    build_liquidity_consumption_query_index,
    query_liquidity_consumption_state,
    expand_target_state_variants,
    select_tiered_primary_narratives,
    add_market_next_open_choices,
)
from src.research_common.ict.structure_entry_semantics import expand_entry_target_variants


def _state_bars(day="2026-08-05"):
    idx = pd.date_range(f"{day} 08:20", f"{day} 10:10", freq="1min", tz=NY_TZ)
    base = np.full(len(idx), 105.0)
    df = pd.DataFrame({"open":base,"high":base+0.5,"low":base-0.5,"close":base,"volume":100.0},index=idx)
    # Shallow low raid: only 0.10 below, immediate close reclaim.
    ts=pd.Timestamp(f"{day} 09:00",tz=NY_TZ)
    df.loc[ts,["open","high","low","close"]]=[100.2,100.4,99.9,100.1]
    # Deep accepted high raid: multiple closes above 110 and ~3 points through.
    for hhmm,vals in {
        "09:30":(109.8,112.5,109.5,112.0),
        "09:31":(112.0,113.0,111.5,112.4),
        "09:32":(112.4,112.8,109.8,110.2),
    }.items():
        df.loc[pd.Timestamp(f"{day} {hhmm}",tz=NY_TZ),["open","high","low","close"]]=vals
    return df


def test_liquidity_state_distinguishes_shallow_equal_like_from_deep_consumed():
    bars=_state_bars()
    levels=pd.DataFrame([
        {"ny_date":"2026-08-05","liquidity_side":"low","level_price":100.0,"level_available_time":pd.Timestamp("2026-08-05 08:30",tz=NY_TZ)},
        {"ny_date":"2026-08-05","liquidity_side":"high","level_price":110.0,"level_available_time":pd.Timestamp("2026-08-05 08:30",tz=NY_TZ)},
    ])
    idx=build_liquidity_consumption_query_index(bars,levels,config=LiquidityConsumptionConfig())
    low=query_liquidity_consumption_state(idx,day_text="2026-08-05",side="low",price=100.0,query_time=pd.Timestamp("2026-08-05 09:10",tz=NY_TZ))
    high=query_liquidity_consumption_state(idx,day_text="2026-08-05",side="high",price=110.0,query_time=pd.Timestamp("2026-08-05 09:40",tz=NY_TZ))
    assert low["liquidity_state"] == "shallow_probe_equal_like"
    assert low["liquidity_max_penetration_abs"] < 0.2
    assert high["liquidity_state"] == "accepted_or_deep_consumed"
    assert high["liquidity_max_consecutive_outside_closes"] >= 2


def test_partial_target_is_kept_but_deep_consumed_target_is_only_diagnostic():
    base={
        "trade_side":"SHORT","entry_price":105.0,"stop_price":110.0,
        "target_price":100.0,"nearest_internal_target_price":102.0,
        "target_liquidity_state":"shallow_probe_equal_like",
    }
    out=expand_target_state_variants(pd.DataFrame([base]))
    assert "external_if_not_fully_consumed" in set(out.target_model_r13)
    deep={**base,"target_liquidity_state":"accepted_or_deep_consumed"}
    out2=expand_target_state_variants(pd.DataFrame([deep]))
    assert "external_if_not_fully_consumed" not in set(out2.target_model_r13)
    assert "external_any_state" in set(out2.target_model_r13)


def test_tiered_primary_keeps_micro_and_later_visible_structure_and_resets_after_deeper_terminal():
    t0=pd.Timestamp("2026-08-05 09:00",tz=NY_TZ)
    rows=[]
    for i,(pct,px,ver) in enumerate([(0.2,101.0,1),(0.65,102.0,1),(0.9,103.0,1),(0.7,104.0,2)]):
        rows.append({
            "event_id":"e","execution_tf":"1m","reference_model_r13":"outermost_barrier_newly_broken",
            "terminal_version":ver,"causal_visibility_percentile":pct,"break_available_time":t0+pd.Timedelta(minutes=i+1),
            "mss_reference_price":px,"trade_side":"LONG",
        })
    out=select_tiered_primary_narratives(pd.DataFrame(rows))
    assert len(out)==4
    assert set(out.structure_visibility_tier_r13)=={"micro_lt_p50","visible_p50_p80","strong_ge_p80"}


def test_r12_market_variant_now_carries_terminal_stop():
    breaks=pd.DataFrame([{
        "trade_side":"LONG","break_close_cross":True,"break_available_time":pd.Timestamp("2026-08-05 10:47",tz=NY_TZ),
        "target_price":140.0,"terminal_extreme_price":130.0,"mss_reference_price":134.0,
    }])
    out=expand_entry_target_variants(breaks,pd.DataFrame())
    m=out.loc[out.entry_order_type.eq("market_next_open")]
    assert len(m)==1
    assert float(m.iloc[0].stop_price)==130.0


def test_r13_market_choice_carries_terminal_stop_before_replay():
    narratives=pd.DataFrame([{
        "trade_side":"SHORT","break_close_cross":True,"break_available_time":pd.Timestamp("2026-08-05 10:01",tz=NY_TZ),
        "terminal_extreme_price":144.49,"event_id":"e","execution_tf":"2m","execution_tf_minutes":2,
        "ny_date":"2026-08-05","target_price":131.65,
    }])
    out=add_market_next_open_choices(narratives)
    assert len(out)==1
    assert float(out.iloc[0].stop_price)==144.49


def test_r13_compact_stage4_matches_wide_primary_and_fvg_selectors():
    from src.research_common.ict.structure_entry_semantics import (
        StructureSemanticConfig,
        build_visible_swing_catalog,
        build_structure_break_fvg_atlas,
        build_r13_primary_break_fvg_compact,
    )
    from src.research_common.ict.semantic_consolidation import (
        select_reference_narratives,
        select_tiered_primary_narratives,
        consolidate_fvg_entry_choices,
    )
    from tests.research.ict.test_soxl_ict_structure_semantics_r12 import _bars

    bars = _bars()
    day = "2026-08-05"
    sweeps = pd.DataFrame([{
        "ny_date": day, "event_id": "synthetic", "trade_side": "LONG",
        "sweep_time": pd.Timestamp(f"{day} 10:40", tz=NY_TZ),
        "sweep_bar_start": pd.Timestamp(f"{day} 10:39", tz=NY_TZ),
        "level_price": 131.48, "level_type": "intraday_15m_swing",
        "liquidity_family": "intraday_15m_swing", "target_price": 138.0,
        "setup_eligible_at_sweep": True,
    }])
    cfg = StructureSemanticConfig(execution_timeframes=(1,), structure_lookback_minutes=180)
    cat = build_visible_swing_catalog(bars, [pd.Timestamp(day).date()], config=cfg)

    wide_breaks, wide_fvgs = build_structure_break_fvg_atlas(bars, sweeps, cat, config=cfg)
    refs = select_reference_narratives(wide_breaks)
    expected_primary = select_tiered_primary_narratives(refs)
    expected_entries = consolidate_fvg_entry_choices(expected_primary, wide_fvgs)

    compact_primary, compact_fvgs, audit = build_r13_primary_break_fvg_compact(bars, sweeps, cat, config=cfg)
    compact_entries = consolidate_fvg_entry_choices(compact_primary, compact_fvgs)

    pcols = ["event_id", "execution_tf", "terminal_version", "structure_visibility_tier_r13",
             "break_available_time", "mss_reference_time", "mss_reference_price"]
    left = expected_primary[pcols].sort_values(pcols).reset_index(drop=True)
    right = compact_primary[pcols].sort_values(pcols).reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right, check_dtype=False)

    ecols = ["event_id", "execution_tf", "break_available_time", "mss_reference_time",
             "mss_reference_price", "entry_model_r13", "fvg_train_sequence", "fvg_near_edge_entry"]
    left_e = expected_entries[ecols].sort_values(ecols).reset_index(drop=True)
    right_e = compact_entries[ecols].sort_values(ecols).reset_index(drop=True)
    pd.testing.assert_frame_equal(left_e, right_e, check_dtype=False)
    assert int(audit["r12_wide_fvg_rows_equivalent"].sum()) >= len(compact_fvgs)
