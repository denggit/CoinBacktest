import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r08 import (
    R08Config,
    build_classical_ict_hierarchy,
    build_completed_trend_legs,
    build_trend_qualified_liquidity,
    build_projection_impact_atlas,
    summarize_projection_impact,
    r08_causal_audit,
)


def _bars_from_15m(points):
    # Four 1m rows per synthetic point are not enough for 15m aggregation, so
    # construct 15 1m bars with the same OHLC around each point.
    rows=[]; idx=[]
    t0=pd.Timestamp('2026-01-01')
    for i,(o,h,l,c) in enumerate(points):
        base=t0+pd.Timedelta(minutes=15*i)
        for m in range(15):
            idx.append(base+pd.Timedelta(minutes=m))
            rows.append((o,h,l,c))
    return pd.DataFrame(rows,index=pd.DatetimeIndex(idx),columns=['open','high','low','close'])


def test_classical_recursive_hierarchy_is_causal():
    # High sequence at 15m bars creates ST highs 110, 120, 108 around which
    # 120 becomes an ITH after the right STH itself confirms.
    highs=[100,110,101,120,100,108,99,105,98]
    lows =[ 99,100, 95,100, 94, 99,93, 98,92]
    pts=[(100,h,l,100) for h,l in zip(highs,lows)]
    bars=_bars_from_15m(pts)
    h,_=build_classical_ict_hierarchy(bars,timeframe='15m',minutes=15)
    ith=h[(h.pivot_side=='high') & (h.is_it==1)]
    assert len(ith)>=1
    row=ith.loc[ith.level_price.idxmax()]
    assert row.it_available_time >= row.st_available_time


def _manual_hierarchy_bearish():
    # Explicit hierarchy table avoids coupling completed-leg tests to bar-pivot
    # construction.  LTH 120 -> lower ITH 112 -> lower ITH 108 -> LTL 90.
    t=pd.date_range('2026-01-01',periods=8,freq='15min')
    rows=[
      ('H0','high',t[0],120,1,1), ('L0','low', t[1],105,1,0),
      ('H1','high',t[2],112,1,0), ('L1','low', t[3],100,1,0),
      ('H2','high',t[4],108,1,0), ('L2','low', t[5], 95,1,0),
      ('H3','high',t[6],104,1,0), ('L3','low', t[7], 90,1,1),
    ]
    out=[]
    for sid,side,pt,px,it,lt in rows:
        out.append(dict(swing_id=sid,pivot_side=side,source_timeframe='15m',source_timeframe_min=15,
                        pivot_time=pt,level_price=px,is_it=it,is_lt=lt,
                        st_available_time=pt+pd.Timedelta(minutes=30),
                        it_available_time=pt+pd.Timedelta(minutes=45) if it else pd.NaT,
                        lt_available_time=pt+pd.Timedelta(minutes=60) if lt else pd.NaT))
    return pd.DataFrame(out)


def _htf_for_bos():
    idx=pd.date_range('2026-01-01',periods=20,freq='15min')
    close=np.linspace(110,92,len(idx))
    close[10:]=[94,96,99,103,105,109,110,111,112,113]
    df=pd.DataFrame({'open':close,'high':close+1,'low':close-1,'close':close},index=idx)
    df['bar_end_time']=df.index+pd.Timedelta(minutes=15)
    return df


def test_completed_bearish_leg_requires_monotonic_it_and_it_bos():
    h=_manual_hierarchy_bearish(); htf=_htf_for_bos()
    legs=build_completed_trend_legs(h,{15:htf},config=R08Config(structure_timeframes=(('15m',15),),trend_scales=(0.03,0.05,0.07),min_it_swings_per_side=2),progress=False)
    assert len(legs)==1
    r=legs.iloc[0]
    assert r.trend_direction=='bearish'
    assert r.directional_integrity_flag==1
    assert r.reversal_bos_confirmed_flag==1
    assert r.scale_ge_07pct_flag==1
    assert r.leg_available_time >= r.reversal_bos_available_time


def test_nonmonotonic_it_sequence_is_not_trend_qualified():
    h=_manual_hierarchy_bearish()
    h.loc[h.swing_id=='H2','level_price']=115.0  # breaks descending ITH sequence
    legs=build_completed_trend_legs(h,{15:_htf_for_bos()},config=R08Config(structure_timeframes=(('15m',15),)),progress=False)
    assert len(legs)==1
    assert legs.iloc[0].directional_integrity_flag==0
    assert legs.iloc[0].trend_qualified_ge3_flag==0


def test_key_liquidity_excludes_st_and_consumed_bos_reference():
    h=_manual_hierarchy_bearish(); htf=_htf_for_bos()
    cfg=R08Config(structure_timeframes=(('15m',15),),min_it_swings_per_side=2)
    legs=build_completed_trend_legs(h,{15:htf},config=cfg,progress=False)
    # Expand htf into true 1m bars for lifecycle checks.
    pts=[(float(r.open),float(r.high),float(r.low),float(r.close)) for _,r in htf.iterrows()]
    bars=_bars_from_15m(pts)
    liq=build_trend_qualified_liquidity(bars,h,legs,config=cfg,progress=False)
    assert set(liq.swing_role).issubset({'LTH','ITH','LTL','ITL'})
    assert set(liq.liquidity_side)=={'BSL'}
    # The last ITH is the BOS reference and is consumed by the confirmation;
    # at least one historical higher ITH/LTH should survive activation.
    assert (liq.consumed_before_activation_flag==1).any()
    assert (liq.active_at_activation_flag==1).any()
    audit=r08_causal_audit(h,legs,liq)
    assert int(audit.violations.sum())==0


def test_completed_bullish_leg_is_exact_mirror():
    t=pd.date_range('2026-02-01',periods=8,freq='15min')
    rows=[
      ('L0','low', t[0], 90,1,1), ('H0','high',t[1],100,1,0),
      ('L1','low', t[2], 94,1,0), ('H1','high',t[3],105,1,0),
      ('L2','low', t[4], 98,1,0), ('H2','high',t[5],110,1,0),
      ('L3','low', t[6],102,1,0), ('H3','high',t[7],120,1,1),
    ]
    out=[]
    for sid,side,pt,px,it,lt in rows:
        out.append(dict(swing_id=sid,pivot_side=side,source_timeframe='15m',source_timeframe_min=15,
                        pivot_time=pt,level_price=px,is_it=it,is_lt=lt,
                        st_available_time=pt+pd.Timedelta(minutes=30),
                        it_available_time=pt+pd.Timedelta(minutes=45) if it else pd.NaT,
                        lt_available_time=pt+pd.Timedelta(minutes=60) if lt else pd.NaT))
    h=pd.DataFrame(out)
    idx=pd.date_range('2026-02-01',periods=20,freq='15min')
    close=np.linspace(95,118,len(idx)); close[10:]=[116,112,108,104,100,96,94,92,91,90]
    htf=pd.DataFrame({'open':close,'high':close+1,'low':close-1,'close':close},index=idx)
    htf['bar_end_time']=htf.index+pd.Timedelta(minutes=15)
    legs=build_completed_trend_legs(h,{15:htf},config=R08Config(structure_timeframes=(('15m',15),)),progress=False)
    assert len(legs)==1
    r=legs.iloc[0]
    assert r.trend_direction=='bullish'
    assert r.directional_integrity_flag==1
    assert r.reversal_bos_structure=='ITL'
    assert r.reversal_bos_confirmed_flag==1


def test_r081_projection_scope_separates_native_nested_and_invalid():
    # One synthetic 30m bullish completed trend.  A 30m ITL is native, a 15m
    # ITL is nested lower-TF, and a 1H ITL must be rejected as a higher-TF
    # projection into the lower-timeframe trend.
    t0 = pd.Timestamp('2026-03-01 00:00')
    h = pd.DataFrame([
        dict(swing_id='N0', pivot_side='low', source_timeframe='30m', source_timeframe_min=30,
             pivot_time=t0, level_price=100.0, is_it=1, is_lt=1,
             st_available_time=t0+pd.Timedelta(minutes=60), it_available_time=t0+pd.Timedelta(minutes=90), lt_available_time=t0+pd.Timedelta(minutes=120)),
        dict(swing_id='N1', pivot_side='low', source_timeframe='30m', source_timeframe_min=30,
             pivot_time=t0+pd.Timedelta(hours=2), level_price=103.0, is_it=1, is_lt=0,
             st_available_time=t0+pd.Timedelta(hours=2, minutes=30), it_available_time=t0+pd.Timedelta(hours=3), lt_available_time=pd.NaT),
        dict(swing_id='L15', pivot_side='low', source_timeframe='15m', source_timeframe_min=15,
             pivot_time=t0+pd.Timedelta(hours=1), level_price=101.0, is_it=1, is_lt=0,
             st_available_time=t0+pd.Timedelta(hours=1, minutes=30), it_available_time=t0+pd.Timedelta(hours=2), lt_available_time=pd.NaT),
        dict(swing_id='H60', pivot_side='low', source_timeframe='1H', source_timeframe_min=60,
             pivot_time=t0+pd.Timedelta(hours=1, minutes=30), level_price=102.0, is_it=1, is_lt=0,
             st_available_time=t0+pd.Timedelta(hours=2, minutes=30), it_available_time=t0+pd.Timedelta(hours=3), lt_available_time=pd.NaT),
    ])
    legs = pd.DataFrame([dict(
        trend_leg_id='LEG30', source_timeframe='30m', source_timeframe_min=30,
        trend_direction='bullish', origin_swing_id='N0', origin_time=t0, terminal_time=t0+pd.Timedelta(hours=4),
        trend_move_pct=0.08, leg_available_time=t0+pd.Timedelta(hours=5), trend_qualified_ge3_flag=1,
    )])
    idx = pd.date_range(t0, periods=12*60, freq='1min')
    bars = pd.DataFrame({'open':110.0,'high':111.0,'low':109.0,'close':110.0}, index=idx)
    liq = build_trend_qualified_liquidity(bars,h,legs,config=R08Config(structure_timeframes=(('30m',30),)),progress=False)
    got = dict(zip(liq.swing_id, liq.projection_scope))
    assert got['N0'] == 'native'
    assert got['N1'] == 'native'
    assert got['L15'] == 'nested_lower_tf'
    assert got['H60'] == 'invalid_higher_tf_projection'
    assert liq.loc[liq.swing_id.eq('H60'),'canonical_key_liquidity_flag'].iloc[0] == 0


def test_r081_projection_impact_summary_reports_pf_expectancy():
    impact=pd.DataFrame({
        "projection_scope":["native"]*4, "liquidity_side":["SSL"]*4,
        "gross_return_60m":[0.01,0.006,-0.004,-0.003],
        "gross_return_180m":[0.02,0.01,-0.004,-0.003],
        "gross_return_360m":[0.02,0.01,-0.004,-0.003],
        "gross_return_720m":[0.02,0.01,-0.004,-0.003],
        "gross_return_1440m":[0.02,0.01,-0.004,-0.003],
    })
    out=summarize_projection_impact(impact,cost_multipliers=(2.0,),roundtrip_cost_1x=0.0011)
    r=out.loc[out.horizon_minutes.eq(60)].iloc[0]
    assert r.trades==4
    assert r.win_rate==0.5
    assert r.mean_net_return>0
    assert r.profit_factor>1
    assert r.expectancy_positive_flag==1


def test_r081_projection_impact_uses_next_bar_open_causally():
    idx=pd.date_range('2026-04-01',periods=200,freq='1min')
    px=np.linspace(100,110,len(idx))
    bars=pd.DataFrame({'open':px,'high':px+0.1,'low':px-0.1,'close':px},index=idx)
    liq=pd.DataFrame([dict(
        projection_scope='native',swing_id='X',trend_leg_id='L',source_timeframe='15m',swing_source_timeframe='15m',
        liquidity_side='SSL',swing_role='ITL',trend_move_pct=0.05,active_at_activation_flag=1,
        liquidity_activation_time=idx[10],first_sweep_after_activation_time=idx[20],
        first_sweep_after_activation_available_time=idx[21],
    )])
    out=build_projection_impact_atlas(bars,liq,research_start=idx[0],research_end=idx[-1],horizons_minutes=(60,))
    assert len(out)==1
    assert out.iloc[0].entry_time==idx[21]
    assert out.iloc[0].entry_price==bars.iloc[21].open
    assert out.iloc[0].gross_return_60m>0
