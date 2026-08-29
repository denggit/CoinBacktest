from __future__ import annotations
import numpy as np
import pandas as pd
from src.research_common.ict_mss2.r12 import R12Config,prepare_completed_trend_contexts,build_completed_trend_physical_liquidity,build_root_sweep_events,build_opposite_liquidity_paths,r12_causal_audit

def _bars(n=360):
    idx=pd.date_range("2026-01-01",periods=n,freq="1min"); x=np.full(n,100.0)
    return pd.DataFrame({"open":x,"high":x+.2,"low":x-.2,"close":x,"volume":1.0},index=idx)

def _ctx(sid,side,px,av,trend_tf=60,leg="L1"):
    av=pd.Timestamp(av)
    return {"trend_leg_id":leg,"source_timeframe":f"{trend_tf}m","source_timeframe_min":trend_tf,"trend_direction":"bullish" if side=="SSL" else "bearish","trend_move_pct":.06,"trend_origin_time":av-pd.Timedelta(hours=10),"trend_terminal_time":av-pd.Timedelta(hours=5),"trend_available_time":av,"liquidity_side":side,"swing_id":sid,"swing_source_timeframe":"15m","swing_source_timeframe_min":15,"projection_scope":"nested_lower_tf","canonical_key_liquidity_flag":0,"nested_lower_tf_flag":1,"invalid_projection_flag":0,"swing_is_lt":0,"swing_role":"ITL" if side=="SSL" else "ITH","pivot_time":av-pd.Timedelta(days=3),"level_price":px,"own_it_available_time":av-pd.Timedelta(days=2),"own_lt_available_time":pd.NaT,"liquidity_activation_time":av,"consumed_before_activation_flag":0,"active_at_activation_flag":1,"scale_ge_03pct_flag":1,"scale_ge_05pct_flag":1,"scale_ge_07pct_flag":0}

def _run(b,rows):
    c=prepare_completed_trend_contexts(pd.DataFrame(rows),pd.DataFrame()); p=build_completed_trend_physical_liquidity(b,c); cfg=R12Config(path_horizon_minutes=300,landmark_max_minutes=120); r=build_root_sweep_events(b,p,c,research_start=pd.Timestamp("2026-01-01"),research_end=pd.Timestamp("2026-01-01 05:59"),config=cfg); path=build_opposite_liquidity_paths(b,p,c,r,config=cfg,progress=False); return c,p,r,path

def test_dedup_and_no_future_context_promotion():
    b=_bars(); b.loc[pd.Timestamp("2026-01-01 01:00"),"low"]=94; b.loc[pd.Timestamp("2026-01-01 02:30"),"high"]=106
    c,p,r,path=_run(b,[_ctx("root","SSL",95,"2026-01-01 00:10",60,"early"),_ctx("root","SSL",95,"2026-01-01 02:00",240,"future"),_ctx("target","BSL",105,"2026-01-01 00:10"),_ctx("deep","SSL",90,"2026-01-01 00:10")])
    assert p.swing_id.nunique()==3
    rr=r.loc[r.root_swing_ids.str.contains("root")].iloc[0]
    assert rr.root_max_known_trend_tf_min==60 and rr.root_known_trend_leg_count==1
    assert path.loc[path.root_event_id.eq(rr.root_event_id)].iloc[0].path_outcome=="direct_opposite_delivery"

def test_direct_opposite_delivery():
    b=_bars(); b.loc[pd.Timestamp("2026-01-01 01:00"),"low"]=94; b.loc[pd.Timestamp("2026-01-01 02:00"),"high"]=106
    _,_,_,path=_run(b,[_ctx("root","SSL",95,"2026-01-01 00:10"),_ctx("deep","SSL",90,"2026-01-01 00:10"),_ctx("target","BSL",105,"2026-01-01 00:10")]); q=path.loc[path.root_swing_ids.str.contains("root")].iloc[0]
    assert q.path_outcome=="direct_opposite_delivery" and q.opposite_1_touch_time==pd.Timestamp("2026-01-01 02:00")

def test_same_side_failure():
    b=_bars(); b.loc[pd.Timestamp("2026-01-01 01:00"),"low"]=94; b.loc[pd.Timestamp("2026-01-01 01:30"),"low"]=89
    _,_,_,path=_run(b,[_ctx("root","SSL",95,"2026-01-01 00:10"),_ctx("deep","SSL",90,"2026-01-01 00:10"),_ctx("target","BSL",105,"2026-01-01 00:10")]); q=path.loc[path.root_swing_ids.str.contains("root")].iloc[0]
    assert q.path_outcome=="same_side_continuation_no_opposite_hit"

def test_cascade_then_opposite():
    b=_bars(); b.loc[pd.Timestamp("2026-01-01 01:00"),"low"]=94; b.loc[pd.Timestamp("2026-01-01 01:30"),"low"]=89; b.loc[pd.Timestamp("2026-01-01 03:00"),"high"]=106
    _,_,_,path=_run(b,[_ctx("root","SSL",95,"2026-01-01 00:10"),_ctx("deep","SSL",90,"2026-01-01 00:10"),_ctx("target","BSL",105,"2026-01-01 00:10")]); q=path.loc[path.root_swing_ids.str.contains("root")].iloc[0]
    assert q.path_outcome=="cascade_then_opposite_delivery" and q.deeper_same_side_touch_time<q.opposite_1_touch_time

def test_same_bar_two_sided_is_ambiguous():
    b=_bars(); t=pd.Timestamp("2026-01-01 01:00"); b.loc[t,"low"]=94; b.loc[t,"high"]=106
    _,_,r,path=_run(b,[_ctx("ssl","SSL",95,"2026-01-01 00:10"),_ctx("bsl","BSL",105,"2026-01-01 00:10"),_ctx("deep","SSL",90,"2026-01-01 00:10"),_ctx("up","BSL",110,"2026-01-01 00:10")]); q=r.loc[r.root_sweep_time.eq(t)]
    assert len(q)==2 and q.same_bar_two_sided_root_flag.eq(1).all(); assert path.loc[path.root_sweep_time.eq(t)].path_outcome.eq("same_bar_two_sided_root_ambiguous").all()

def test_causal_audit_zero():
    b=_bars(); b.loc[pd.Timestamp("2026-01-01 01:00"),"low"]=94; b.loc[pd.Timestamp("2026-01-01 02:00"),"high"]=106
    c,p,r,path=_run(b,[_ctx("root","SSL",95,"2026-01-01 00:10"),_ctx("target","BSL",105,"2026-01-01 00:10")]); a=r12_causal_audit(c,p,r,path)
    assert int(pd.to_numeric(a.violations,errors="coerce").fillna(0).sum())==0
