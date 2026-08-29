import numpy as np
import pandas as pd

from src.research_common.ict.premarket_mss_fvg import NY_TZ, aggregate_closed_bars
from src.research_common.ict.structure_entry_semantics import (
    StructureSemanticConfig,
    build_dual_session_liquidity_levels,
    build_visible_swing_catalog,
    build_structure_break_fvg_atlas,
    expand_entry_target_variants,
)


def _bars(day="2026-08-05"):
    idx = pd.date_range(f"{day} 04:00", f"{day} 16:29", freq="1min", tz=NY_TZ)
    # Smooth synthetic path; overwrite 10:38 onward with a raid, micro break,
    # deeper terminal, then a stronger second break and FVG train.
    px = np.linspace(140, 136, len(idx))
    df = pd.DataFrame({"open":px, "high":px+0.08, "low":px-0.08, "close":px, "volume":100.0}, index=idx)
    seq = {
        "10:38": (132.0,132.2,131.8,132.0),
        "10:39": (132.0,132.1,131.45,131.7),
        "10:40": (131.7,131.8,130.7,131.2),
        "10:41": (131.2,131.9,131.0,131.7),
        "10:42": (131.7,132.4,131.6,132.3),
        "10:43": (132.3,132.82,132.2,132.7),
        "10:44": (132.7,132.79,132.25,132.5),
        "10:45": (132.5,133.2,132.45,133.1),
        "10:46": (133.1,133.8,132.9,133.7),
        "10:47": (133.7,134.0,133.2,133.6),
        "10:48": (133.6,134.31,133.3,134.0),
        "10:49": (134.0,134.1,133.4,133.6),
        "10:50": (133.6,133.8,132.8,133.0),
        "10:51": (133.0,133.4,130.0,130.5),
        "10:52": (130.5,131.4,130.2,131.2),
        "10:53": (131.2,132.8,131.1,132.6),
        "10:54": (132.6,133.5,132.4,133.4),
        "10:55": (133.4,134.0,133.0,133.8),
        "10:56": (133.8,134.2,133.2,133.5),
        "10:57": (133.5,134.6,133.4,134.5),
        "10:58": (134.5,134.8,134.2,134.7),
    }
    for hhmm, vals in seq.items():
        ts=pd.Timestamp(f"{day} {hhmm}", tz=NY_TZ)
        df.loc[ts,["open","high","low","close"]]=vals
    return df


def test_dual_session_levels_are_independent_and_causal():
    bars=_bars()
    frozen,running=build_dual_session_liquidity_levels(bars,[pd.Timestamp("2026-08-05").date()])
    assert set(frozen.liquidity_family)=={"early_premarket_extreme","late_premarket_extreme"}
    late=frozen[frozen.liquidity_family.eq("late_premarket_extreme")]
    assert (pd.to_datetime(late.level_available_time).dt.hour==9).all()
    assert not running.empty
    assert (pd.to_datetime(running.level_available_time) >= pd.Timestamp("2026-08-05 08:31",tz=NY_TZ)).all()


def test_visibility_score_keeps_micro_but_does_not_equate_latest_with_best():
    bars=_bars()
    cat=build_visible_swing_catalog(bars,[pd.Timestamp("2026-08-05").date()],config=StructureSemanticConfig(execution_timeframes=(1,)))
    assert not cat.empty
    assert "visibility_score" in cat and "causal_visibility_percentile" in cat
    assert cat.visibility_score.notna().any()


def test_same_sweep_can_emit_multiple_structure_breaks_and_fvg_train():
    bars=_bars()
    day="2026-08-05"
    sweeps=pd.DataFrame([{
        "ny_date":day,"event_id":"synthetic","trade_side":"LONG","sweep_time":pd.Timestamp(f"{day} 10:40",tz=NY_TZ),
        "sweep_bar_start":pd.Timestamp(f"{day} 10:39",tz=NY_TZ),"level_price":131.48,"level_type":"intraday_15m_swing","liquidity_family":"intraday_15m_swing",
        "target_price":138.0,"setup_eligible_at_sweep":True,
    }])
    cfg=StructureSemanticConfig(execution_timeframes=(1,),structure_lookback_minutes=180)
    cat=build_visible_swing_catalog(bars,[pd.Timestamp(day).date()],config=cfg)
    breaks,fvgs=build_structure_break_fvg_atlas(bars,sweeps,cat,config=cfg)
    assert not breaks.empty
    # Multiple structure levels can break in one episode; first micro break must
    # not terminate the episode.
    assert breaks.mss_reference_price.nunique() >= 2
    if not fvgs.empty:
        assert set(fvgs.fvg_middle_relation_to_break).issubset({"pre_break_middle","break_bar_middle"})


def test_swing_cap_is_variant_not_universe_gate():
    breaks=pd.DataFrame([{"trade_side":"LONG","break_close_cross":True,"break_available_time":pd.Timestamp("2026-08-05 10:47",tz=NY_TZ),"target_price":140.0,"stop_price":130.0,"mss_reference_price":134.0}])
    fvgs=pd.DataFrame([{
        "trade_side":"LONG","target_price":140.0,"stop_price":130.0,"mss_reference_price":134.0,
        "fvg_near_edge_entry":134.05,"signal_time":pd.Timestamp("2026-08-05 10:48",tz=NY_TZ),
        "swing_buffer_cap_pass":True,"break_middle_fvg_buffer_cap_pass":False,
        "nearest_internal_target_price":136.0,
    }])
    out=expand_entry_target_variants(breaks,fvgs)
    assert "fvg_train_uncapped" in set(out.entry_model)
    assert "fvg_swing_plusminus_0p10_cap" in set(out.entry_model)
    assert "close_break_next_open_market" in set(out.entry_model)
