from __future__ import annotations
import pandas as pd
from src.research_common.post_sweep_staged_execution import (
    PostSweepStagedExecutionConfig, add_trigger_flags, earliest_trigger_rows,
    build_fill_table, scheme_specs, simulate_schemes,
)


def _path() -> pd.DataFrame:
    rows=[]
    for e in range(1,8):
        rows.append({
            "checkpoint_id":f"E_C{e}","zone_event_id":"E","period":"EARLY_2023_2024",
            "checkpoint_available_time":pd.Timestamp("2025-01-01")+pd.Timedelta(minutes=e),
            "entry_reference_time":pd.Timestamp("2025-01-01")+pd.Timedelta(minutes=e),
            "elapsed_bars":e,"checkpoint_high":100+e,"checkpoint_low":98+e/2,"checkpoint_close":99+e/2,
            "entry_reference_price":99+e/2,
            "no_new_low_3bars":e>=4,"no_new_low_5bars":e>=5,"no_new_low_10bars":e>=6,
            "micro_high_break_3bars":e>=4,"micro_high_break_5bars":e>=5,"micro_high_break_10bars":e>=6,
            "zone_floor_reclaimed":e>=6,"zone_ceiling_reclaimed":False,
            "future_mfe_15m":.01,"future_mae_15m":-.01,"future_mfe_60m":.02,"future_mae_60m":-.01,
            "future_mfe_180m":.03,"future_mae_180m":-.02,
        })
    return pd.DataFrame(rows)


def test_trigger_flags_are_causal_and_natural():
    flagged=add_trigger_flags(_path())
    t=earliest_trigger_rows(flagged,60)
    got=dict(zip(t["trigger_name"],t["elapsed_bars"]))
    assert got["INITIAL"]==1
    assert got["EARLY_REJECTION"]==4
    assert got["CONFIRMED_REJECTION"]==5
    assert got["STRONG_RECLAIM"]==6


def test_staged_fill_activates_next_bar():
    flagged=add_trigger_flags(_path())
    t=earliest_trigger_rows(flagged,60)
    fills=build_fill_table(t,scheme_specs())
    assert (fills["entry_activation_elapsed"]==fills["signal_elapsed"]+1).all()
    assert not (pd.to_datetime(fills["entry_time"])<pd.to_datetime(fills["signal_time"])).any()


def test_every_scheme_has_fixed_total_risk():
    for scheme in scheme_specs():
        assert abs(sum(s.weight for s in scheme.stages)-1.0)<1e-12


def test_replay_keeps_same_event_universe():
    cfg=PostSweepStagedExecutionConfig(horizons=(5,),max_deployment_minutes=60).validate()
    flagged=add_trigger_flags(_path())
    triggers=earliest_trigger_rows(flagged,60)
    fills=build_fill_table(triggers,scheme_specs())
    replay=simulate_schemes(flagged,fills,scheme_specs(),cfg,progress=False)
    assert replay["zone_event_id"].nunique()==1
    assert replay["scheme"].nunique()==len(scheme_specs())
    full=replay.loc[replay["scheme"]=="FULL_FIRST_CHECKPOINT"].iloc[0]
    assert full["final_deployed_weight"]==1.0
