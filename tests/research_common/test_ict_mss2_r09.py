import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r09 import (
    R09Config,
    attach_r09_outcomes,
    build_immediate_entries,
    build_physical_liquidity_sweeps,
    build_root_sweep_episodes,
    r09_causal_audit,
)


def _bars(n=120, start='2026-01-01'):
    idx=pd.date_range(start,periods=n,freq='1min')
    px=np.full(n,100.0)
    return pd.DataFrame({'open':px,'high':px+0.5,'low':px-0.5,'close':px},index=idx)


def _liq_row(scope, swing_id, sweep, level=100.0, trend_tf=60, swing_tf=15, side='SSL', role='ITL', move=0.06, pivot='2025-12-01'):
    return dict(
        projection_scope=scope,swing_id=swing_id,liquidity_side=side,
        first_sweep_after_activation_time=pd.Timestamp(sweep),
        first_sweep_after_activation_available_time=pd.Timestamp(sweep)+pd.Timedelta(minutes=1),
        level_price=level,pivot_time=pd.Timestamp(pivot),liquidity_activation_time=pd.Timestamp('2025-12-20'),
        active_at_activation_flag=1,source_timeframe_min=trend_tf,source_timeframe=f'{trend_tf}m',
        swing_source_timeframe_min=swing_tf,swing_source_timeframe=f'{swing_tf}m',trend_move_pct=move,
        swing_is_lt=int(role in {'LTL','LTH'}),trend_leg_id=f'LEG_{scope}_{trend_tf}',
        scale_ge_05pct_flag=int(move>=0.05),scale_ge_07pct_flag=int(move>=0.07),
    )


def test_physical_swing_is_counted_once_across_contexts():
    n=pd.DataFrame([_liq_row('native','X','2026-01-01 00:10',trend_tf=60,swing_tf=60)])
    z=pd.DataFrame([_liq_row('nested_lower_tf','X','2026-01-01 00:10',trend_tf=240,swing_tf=60)])
    out=build_physical_liquidity_sweeps(n,z,research_start=pd.Timestamp('2026-01-01'),research_end=pd.Timestamp('2026-01-02'))
    assert len(out)==1
    r=out.iloc[0]
    assert r.native_flag==1 and r.nested_flag==1
    assert r.context_count==2
    assert r.max_trend_timeframe_min==240
    assert r.context_tier=='A+_4H_context'


def test_root_quality_does_not_use_future_cascade():
    n=pd.DataFrame([
        _liq_row('native','A','2026-01-01 00:10',trend_tf=15,swing_tf=15),
        _liq_row('native','B','2026-01-01 00:20',trend_tf=240,swing_tf=240),
        _liq_row('native','C','2026-01-01 00:40',trend_tf=60,swing_tf=60),
    ])
    p=build_physical_liquidity_sweeps(n,pd.DataFrame(),research_start=pd.Timestamp('2026-01-01'),research_end=pd.Timestamp('2026-01-02'))
    roots,_=build_root_sweep_episodes(p,_bars(),config=R09Config(episode_gap_minutes=15))
    assert len(roots)==2
    first=roots.iloc[0]
    # Future 4H sweep at 00:20 is in the same episode, but initial tier remains 15m C.
    assert first.context_tier=='C_15m_context'
    assert first.future_cascade_level_count_15m==2
    assert first.root_max_context_tf_min==15


def test_immediate_entry_waits_until_sweep_bar_is_complete():
    bars=_bars()
    # Sweep bar at 00:10 makes low 98; next bar opens 101.
    bars.loc[pd.Timestamp('2026-01-01 00:10'),['open','high','low','close']]=[100,101,98,99]
    bars.loc[pd.Timestamp('2026-01-01 00:11'),'open']=101
    n=pd.DataFrame([_liq_row('native','A','2026-01-01 00:10',level=99,trend_tf=60,swing_tf=60)])
    physical=build_physical_liquidity_sweeps(n,pd.DataFrame(),research_start=pd.Timestamp('2026-01-01'),research_end=pd.Timestamp('2026-01-02'))
    roots,_=build_root_sweep_episodes(physical,bars)
    trades=build_immediate_entries(bars,roots,config=R09Config(stop_buffer_bps=2.0))
    assert len(trades)==1
    r=trades.iloc[0]
    assert r.entry_time==pd.Timestamp('2026-01-01 00:11')
    assert r.entry_price==101
    assert abs(r.structural_extreme_pre_entry-98.0)<1e-12
    assert r.stop_price < 98.0


def test_same_bar_stop_beats_target_for_market_entry():
    bars=_bars(30)
    # Entry 100 with stop 99; same entry bar reaches both 98.5 and 103.
    bars.iloc[1]=[100,103,98.5,101]
    t=pd.DataFrame([dict(
        trade_event_id='T',episode_id='E',liquidity_side='SSL',context_tier='A_1H_context',trade_direction=1,
        trigger_type='sweep_immediate',execution_minutes=1,entry_kind='market_next_open',limit_variant='none',
        entry_pos_1m=1,entry_time=bars.index[1],entry_price=100.0,stop_price=99.0,
    )])
    out=attach_r09_outcomes(bars,t,config=R09Config(censor_minutes=20),show_progress=False)
    assert out.iloc[0].r2p0_outcome=='stop'
    assert out.iloc[0].r2p0_net_r_cost2x < -1.0


def test_causal_audit_accepts_valid_entries():
    roots=pd.DataFrame([dict(sweep_available_time=pd.Timestamp('2026-01-01 00:11'),sweep_bar_time_1m=pd.Timestamp('2026-01-01 00:10'))])
    trades=pd.DataFrame([dict(signal_available_time=pd.Timestamp('2026-01-01 00:11'),entry_time=pd.Timestamp('2026-01-01 00:11'),entry_kind='market_next_open')])
    audit=r09_causal_audit(roots,trades)
    assert int(audit.violations.sum())==0
