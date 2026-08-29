from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r07 import (
    R07Config,
    build_fvg_lifecycle,
    build_liquidity_expansion_continuations,
    build_reversal_confirmation_atlas,
    r07_causal_audit,
)


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2023-01-01", periods=12, freq="1min")
    # First three bars create a bullish FVG on bar 2: low[2] > high[0].
    o = np.array([99.0, 99.4, 100.2, 100.6, 100.2, 100.8, 101.0, 101.3, 101.6, 101.8, 102.0, 102.2])
    h = np.array([99.5, 99.9, 100.8, 101.0, 100.7, 101.2, 101.4, 101.7, 102.0, 102.2, 102.4, 102.6])
    l = np.array([98.8, 99.2, 100.1, 100.0, 99.9, 100.6, 100.8, 101.1, 101.4, 101.6, 101.8, 102.0])
    c = np.array([99.3, 99.7, 100.6, 100.4, 100.5, 101.0, 101.2, 101.5, 101.8, 102.0, 102.2, 102.4])
    return pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":1.0}, index=idx)


def _hier_stage() -> pd.DataFrame:
    return pd.DataFrame([{
        "stage_id":"S1","episode_id":"E1","episode_stage_no":1,
        "sweep_pos_1m":1,"sweep_bar_time_1m":pd.Timestamp("2023-01-01 00:01"),
        "trade_direction":-1,"episode_start_pos_1m":0,"episode_start_time_1m":pd.Timestamp("2023-01-01"),
        "ict_price_pools_cum":3,"ict_st_only_pools_cum":1,"ict_it_plus_pools_cum":1,
        "ict_lt_pools_cum":1,"ict_htf240_pools_cum":1,"ict_multi_tf_pools_cum":1,
        "ict_external50_pools_cum":0,"ict_clean_pools_cum":1,"ict_structural_key_pools_cum":1,
        "ict_strongest_pool_rank_cum":3,"ict_max_pool_timeframes_cum":2,
    }])


def _r02_stage() -> pd.DataFrame:
    return pd.DataFrame([{
        "stage_id":"S1","sweep_available_time_1m":pd.Timestamp("2023-01-01 00:02"),
        "min_consumed_level_price_cum":99.8,"max_consumed_level_price_cum":100.0,
        "episode_extreme_so_far":100.2,"liquidity_side":"buy_side",
    }])


def _lifecycle() -> pd.DataFrame:
    return pd.DataFrame([
        {"level_id":"H1","pivot_side":"high","level_price":102.0,"source_timeframe_min":240,"active_pos_1m":0,"sweep_pos_1m":-1},
        {"level_id":"L1","pivot_side":"low","level_price":98.0,"source_timeframe_min":240,"active_pos_1m":0,"sweep_pos_1m":-1},
    ])


def test_r07_bsl_reversal_is_actual_confirmation_not_sweep_only():
    f = pd.DataFrame([{
        "trade_event_id":"T1","stage_id":"S1","episode_id":"E1","trade_direction":-1,
        "execution_minutes":2,"trigger_type":"episode_reclaim","entry_time":"2023-01-01 00:10",
    }])
    l = pd.DataFrame([{
        "trade_event_id":"T1","stage_id":"S1","episode_id":"E1",
        "target_htf240_net_return_cost2x":0.01,"target_htf240_holding_minutes":60,
    }])
    out = build_reversal_confirmation_atlas(f,l,_hier_stage())
    assert len(out) >= 1
    assert set(out["family"]) == {"bsl_reversal_short"}
    assert set(out["trigger_type"]) == {"episode_reclaim"}


def test_r07_fvg_lifecycle_created_only_after_fvg_bar_close():
    bars = _bars()
    fvg = build_fvg_lifecycle(bars, execution_minutes=(1,))
    bull = fvg.loc[fvg["direction"].eq(1)].iloc[0]
    assert pd.Timestamp(bull.created_time) == pd.Timestamp("2023-01-01 00:03")
    assert int(bull.active_pos_1m) == 3


def test_r07_continuation_uses_limit_after_close_through_and_fvg():
    bars = _bars()
    trades, outcomes = build_liquidity_expansion_continuations(
        bars, _hier_stage(), _r02_stage(), _lifecycle(),
        config=R07Config(execution_minutes=(1,), acceptance_bars=3, fvg_after_acceptance_bars=3, limit_wait_minutes=5),
        show_progress=False,
    )
    assert not trades.empty
    assert trades["entry_kind"].eq("fvg_limit").all()
    assert trades["trade_direction"].eq(1).all()
    assert (pd.to_datetime(trades["entry_time"]) >= pd.to_datetime(trades["signal_available_time"])).all()
    assert not outcomes.empty


def test_r07_continuation_never_shortens_to_market_chase():
    bars = _bars()
    trades, _ = build_liquidity_expansion_continuations(
        bars, _hier_stage(), _r02_stage(), _lifecycle(),
        config=R07Config(execution_minutes=(1,), limit_wait_minutes=5), show_progress=False,
    )
    assert not trades.empty
    assert set(trades["limit_variant"]).issubset({"proximal","ce"})
    assert "market" not in set(trades["entry_kind"])


def test_r07_causal_audit_zero_on_valid_limit_entries():
    cont = pd.DataFrame([{
        "signal_available_time":pd.Timestamp("2023-01-01 00:03"),"entry_time":pd.Timestamp("2023-01-01 00:04"),"entry_kind":"fvg_limit"
    }])
    corr = pd.DataFrame([{
        "signal_available_time":pd.Timestamp("2023-01-01 00:05"),"entry_time":pd.Timestamp("2023-01-01 00:06"),"entry_kind":"fvg_limit"
    }])
    fvg = pd.DataFrame([{"active_pos_1m":3,"full_rebalance_pos_1m":5}])
    audit = r07_causal_audit(cont,corr,fvg)
    assert int(audit["violations"].sum()) == 0


def test_r07_config_small_range_uses_limit_plus_taker_cost():
    cfg = R07Config().validate()
    assert abs(cfg.limit_market_roundtrip_cost - 0.0008) < 1e-12

def test_reversal_atlas_repairs_legacy_r02_ids_without_many_to_many_blowup():
    features = pd.DataFrame({
        "trade_event_id": ["R02_TRADE_000000001", "R02_TRADE_000000001"],
        "stage_id": ["S1", "S1"], "episode_id": ["E1", "E1"],
        "execution_minutes": [1, 2], "trigger_type": ["episode_reclaim", "episode_reclaim"],
        "trade_direction": [-1, -1], "entry_time": pd.to_datetime(["2025-01-01 00:01", "2025-01-01 00:02"]),
    })
    labels = pd.DataFrame({
        "trade_event_id": ["R02_TRADE_000000001", "R02_TRADE_000000001"],
        "stage_id": ["S1", "S1"], "episode_id": ["E1", "E1"],
        "target_htf240_net_return_cost2x": [0.01, 0.02],
    })
    stages = pd.DataFrame({
        "stage_id": ["S1"], "episode_id": ["E1"], "episode_stage_no": [1], "sweep_pos_1m": [10],
        "ict_price_pools_cum": [3], "ict_structural_key_pools_cum": [1], "ict_htf240_pools_cum": [1],
        "ict_lt_pools_cum": [0], "ict_it_plus_pools_cum": [1], "ict_multi_tf_pools_cum": [1],
        "ict_strongest_pool_rank_cum": [3],
    })
    out = build_reversal_confirmation_atlas(features, labels, stages)
    assert len(out.loc[out["quality_rule"].eq("n3_h4_or_lt")]) == 2
    assert out["trade_event_id"].nunique() == 2
    assert set(out["trade_event_id"]) == {"R02_1M_TRADE_000000001", "R02_2M_TRADE_000000001"}


def test_r07_corridor_is_limit_only_and_compares_local_fvg_stop():
    from src.research_common.ict_mss2.r07 import build_reversal_fvg_corridor_scalps
    bars = _bars()
    fvg = build_fvg_lifecycle(bars, execution_minutes=(1,))
    features = pd.DataFrame([{
        "trade_event_id":"T1","stage_id":"S1","episode_id":"E1","trade_direction":1,
        "execution_minutes":1,"trigger_type":"episode_reclaim",
        "signal_available_time":pd.Timestamp("2023-01-01 00:02"),"stop_price":98.5,
        "entry_time":pd.Timestamp("2023-01-01 00:02"),
    }])
    trades, _, _ = build_reversal_fvg_corridor_scalps(
        bars, features, _hier_stage().assign(trade_direction=1), fvg, _lifecycle(),
        config=R07Config(execution_minutes=(1,), corridor_limit_wait_minutes=5), show_progress=False,
    )
    assert not trades.empty
    assert trades["entry_kind"].eq("fvg_limit").all()
    assert {"reclaim_structural", "fvg_invalidation"}.issubset(set(trades["stop_variant"]))
    assert trades["entry_source"].eq("post_reclaim_first_fvg").all()


def test_r07_reversal_target_grid_does_not_only_test_4h_target():
    from src.research_common.ict_mss2.r07 import summarize_reversal_target_grid
    atlas = pd.DataFrame([{
        "family":"bsl_reversal_short","quality_rule":"n3_h4_or_lt","execution_minutes":2,
        "trigger_type":"episode_reclaim","entry_time":pd.Timestamp("2024-01-01"),
        "target_any_net_return_cost2x":0.002,"target_htf60_net_return_cost2x":-0.001,
        "target_htf240_net_return_cost2x":-0.003,"target_r1p0_net_return_cost2x":0.001,
    }])
    summary, yearly = summarize_reversal_target_grid(atlas)
    assert {"any", "htf60", "htf240", "r1p0"}.issubset(set(summary["target"]))
    assert set(yearly["year"]) == {2024}
