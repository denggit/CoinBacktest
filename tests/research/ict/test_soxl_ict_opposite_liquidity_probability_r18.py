from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict.opposite_liquidity_probability import (
    DEFAULT_RANGE_MODEL,
    ENTRY_HYPOTHESIS_FEATURES,
    EVENT_HYPOTHESIS_FEATURES,
    ProbabilityStudyConfig,
    assert_no_forbidden_features,
    build_event_snapshot_dataset,
    fit_cumulative_hypothesis_models,
    paired_entry_scorecard,
)


def _paths_and_narratives(n: int = 90):
    days = pd.date_range("2023-07-03", periods=n, freq="B")
    paths=[]; narratives=[]
    for i,d in enumerate(days):
        ev=f"e{i}"; success=bool(i%3==0)
        base={
            "ny_date":str(d.date()),"range_model":DEFAULT_RANGE_MODEL,"path_event_id":ev,
            "upper_price":110.0,"lower_price":100.0,"upper_confirmation_time":d,
            "lower_confirmation_time":d,"upper_prominence_score":2.0,"lower_prominence_score":1.5,
            "range_width_abs":10.0,"first_raid_side":"low","first_raid_time":d+pd.Timedelta(hours=9),
            "source_level_price":100.0,"target_price":110.0,"traversal_complete":success,
            "first_raid_penetration_frac_range":0.02,"first_reclaim_time":d+pd.Timedelta(hours=9,minutes=3),
            "approach_efficiency":0.5,"approach_distance_contraction_ratio":0.8,
            "approach_recent_range_vs_prior":1.0,"approach_monotonic_swing_count":2,
            "approach_three_swing_contraction":False,
        }
        paths.append(base)
        # 1m occurs first.  The 2m confirmation is intentionally later so the
        # earlier snapshot cannot know it yet.
        for tf, minute in (("1m", 5), ("2m", 9)):
            narratives.append({
                **base,"event_id":ev,"execution_tf":tf,"causal_visibility_percentile":0.7,
                "break_available_time":d+pd.Timedelta(hours=9,minutes=minute),"terminal_version":2,
                "terminal_extreme_time":d+pd.Timedelta(hours=9,minutes=2),"terminal_extreme_price":99.5,
                "mss_reference_time":d+pd.Timedelta(hours=8,minutes=55),
                "mss_reference_available_time":d+pd.Timedelta(hours=8,minutes=56),
                "mss_reference_relation":"pre_sweep","two_sided_excursion_vs_prior_range":1.2+0.4*success,
                "local_prominence_vs_prior_range":1.1,"reference_is_latest_newly_broken":True,
                "reference_is_highest_visibility_newly_broken":True,
                "reference_is_outermost_barrier_newly_broken":success,"break_wick_cross":True,
                "break_close_cross":success,"terminal_to_break_minutes":3.0,
                "directional_bar_fraction":0.5+0.2*success,"path_net_distance_abs":2.0,
                "path_efficiency":0.4+0.3*success,"break_overshoot_abs":0.2+0.4*success,
                "narrative_attempt_sequence_r13":1,
            })
    return pd.DataFrame(paths), pd.DataFrame(narratives)


def test_r18_feature_maps_exclude_future_and_milestone_labels():
    assert_no_forbidden_features(EVENT_HYPOTHESIS_FEATURES)
    assert_no_forbidden_features(ENTRY_HYPOTHESIS_FEATURES)
    text="|".join(c for v in {**EVENT_HYPOTHESIS_FEATURES, **ENTRY_HYPOTHESIS_FEATURES}.values() for c in v)
    assert "milestone_25" not in text and "milestone_50" not in text and "milestone_75" not in text
    assert "traversal_complete" not in text and "mfe_r" not in text and "mae_r" not in text


def test_r18_cross_tf_confirmation_is_asof_not_future_backfilled():
    paths,narr=_paths_and_narratives(12)
    out=build_event_snapshot_dataset(paths,narr,config=ProbabilityStudyConfig())
    one=out.loc[out["execution_tf"].eq("1m")]
    two=out.loc[out["execution_tf"].eq("2m")]
    assert len(one)==12 and len(two)==12
    assert int(one["other_tf_visible_by_snapshot"].sum())==0
    # At the later 2m snapshot the earlier 1m structure is already known.
    assert bool(two["other_tf_visible_by_snapshot"].eq(1).all())


def test_r18_discovery_fit_evaluates_later_without_refit():
    paths,narr=_paths_and_narratives(650)
    out=build_event_snapshot_dataset(paths,narr)
    one=out.loc[out["execution_tf"].eq("1m")].copy()
    metrics,pred,coef=fit_cumulative_hypothesis_models(
        one,target="target_opposite_by_eod",groups=EVENT_HYPOTHESIS_FEATURES,
        group_order=list(EVENT_HYPOTHESIS_FEATURES),
    )
    assert not metrics.empty and not pred.empty and not coef.empty
    assert {"discovery_2023H2_2024","validation_2025"}.issubset(set(metrics["period"]))
    assert pred["predicted_probability"].between(0,1).all()


def test_r18_paired_entry_comparison_uses_same_physical_sweeps():
    q=pd.DataFrame({
        "path_event_id":["a","a","b","b","c"],
        "entry_archetype":["fvg","market","fvg","market","market"],
        "target_tp_before_terminal_sl":[1,0,0,1,1],
        "rr_to_100":[2.0,1.5,2.0,1.5,1.5],
    })
    score,pairs=paired_entry_scorecard(q)
    assert len(score)==2 and len(pairs)==1
    r=pairs.iloc[0]
    assert int(r["common_filled_events"])==2
    assert int(r["a_only_success"])+int(r["b_only_success"])==2
