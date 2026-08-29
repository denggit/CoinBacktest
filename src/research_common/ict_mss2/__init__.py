"""Causal ICT MSS2 research helpers."""

from .core import (
    MSS2Config,
    aggregate_bars,
    attach_execution_outcomes,
    attach_session_context,
    attach_sweep_baseline_outcomes,
    build_execution_pivots,
    build_first_sweep_lifecycle,
    build_liquidity_levels,
    build_mss_fvg_events,
    causal_audit,
    classify_liquidity,
    normalize_1m_bars,
    split_features_and_labels,
)
from .r02 import (
    R02Config,
    attach_stage_forward_paths,
    attach_structural_exit_outcomes,
    build_stack_execution_triggers,
    build_sweep_episodes,
    build_sweep_stages,
    r02_causal_audit,
    split_r02_features_and_labels,
)
from .r03 import (
    R03Config,
    attach_footprint_microstructure_features,
    attach_overlay_structural_outcomes,
    build_fvg_execution_overlay_attempts,
    build_hybrid_5050_outcomes,
    build_tradebar_microstructure_features,
    build_microstructure_checkpoint_union,
    microstructure_feature_join_audit,
    build_core_reclaim_execution_overlays,
    first_pool_threshold_crossing_trades,
    r03_causal_audit,
    r03_globalize_legacy_trade_ids,
)

from .r033 import (
    R033Config,
    attach_causal_ict_swing_hierarchy,
    attach_causal_pool_hierarchy_to_episode_stages,
    attach_cohorts_to_trades,
    build_hierarchy_stage_cohorts,
    build_displacement_payoff_atlas,
    grouped_metrics,
    hierarchy_causal_audit,
    mss_reference_causal_audit,
)

from .r04 import (
    R04Config,
    attach_tradebar_features as attach_r04_tradebar_features,
    build_4h_continuation_summary,
    build_multi_horizon_path_labels,
    build_partial_risk_coverage_summary,
    build_rule_horizon_scoreboard,
    build_tradebar_horizon_summary,
    build_transition_ladder,
    build_unique_opportunity_features,
    first_qualifying_opportunities,
    r04_causal_audit,
)


from .r05 import (
    R05Config,
    attach_initial_structural_stops,
    build_displacement_anchor_atlas,
    build_execution_swing_hierarchy,
    build_exclusive_opportunity_buckets,
    build_initial_stop_target_atlas,
    build_quality_entry_universe,
    build_trailing_events,
    r05_causal_audit,
    simulate_structural_trailing,
    summarize_initial_stop_atlas,
    summarize_exclusive_opportunity_buckets,
    summarize_initial_stop_by_bucket,
    summarize_mae_by_bucket,
    summarize_trailing_by_bucket,
    summarize_mae_before_target,
    summarize_trailing_results,
)

__all__ = [
    "MSS2Config",
    "R02Config",
    "aggregate_bars",
    "attach_execution_outcomes",
    "attach_session_context",
    "attach_stage_forward_paths",
    "attach_structural_exit_outcomes",
    "attach_sweep_baseline_outcomes",
    "build_execution_pivots",
    "build_first_sweep_lifecycle",
    "build_liquidity_levels",
    "build_mss_fvg_events",
    "build_stack_execution_triggers",
    "build_sweep_episodes",
    "build_sweep_stages",
    "causal_audit",
    "classify_liquidity",
    "normalize_1m_bars",
    "r02_causal_audit",
    "split_features_and_labels",
    "split_r02_features_and_labels",
    "R03Config",
    "attach_footprint_microstructure_features",
    "attach_overlay_structural_outcomes",
    "build_fvg_execution_overlay_attempts",
    "build_hybrid_5050_outcomes",
    "build_tradebar_microstructure_features",
    "build_microstructure_checkpoint_union",
    "microstructure_feature_join_audit",
    "build_core_reclaim_execution_overlays",
    "first_pool_threshold_crossing_trades",
    "r03_causal_audit",
    "r03_globalize_legacy_trade_ids",
    "R033Config",
    "attach_causal_ict_swing_hierarchy",
    "attach_causal_pool_hierarchy_to_episode_stages",
    "attach_cohorts_to_trades",
    "build_hierarchy_stage_cohorts",
    "build_displacement_payoff_atlas",
    "grouped_metrics",
    "hierarchy_causal_audit",
    "mss_reference_causal_audit",
    "R04Config",
    "attach_r04_tradebar_features",
    "build_4h_continuation_summary",
    "build_multi_horizon_path_labels",
    "build_partial_risk_coverage_summary",
    "build_rule_horizon_scoreboard",
    "build_tradebar_horizon_summary",
    "build_transition_ladder",
    "build_unique_opportunity_features",
    "first_qualifying_opportunities",
    "r04_causal_audit",
    "R05Config",
    "attach_initial_structural_stops",
    "build_displacement_anchor_atlas",
    "build_execution_swing_hierarchy",
    "build_exclusive_opportunity_buckets",
    "build_initial_stop_target_atlas",
    "build_quality_entry_universe",
    "build_trailing_events",
    "r05_causal_audit",
    "simulate_structural_trailing",
    "summarize_initial_stop_atlas",
    "summarize_exclusive_opportunity_buckets",
    "summarize_initial_stop_by_bucket",
    "summarize_mae_by_bucket",
    "summarize_trailing_by_bucket",
    "summarize_mae_before_target",
    "summarize_trailing_results",
    "R07Config",
    "build_reversal_confirmation_atlas",
    "summarize_reversal_atlas",
    "summarize_reversal_target_grid",
    "summarize_family_target_grid",
    "build_fvg_lifecycle",
    "build_liquidity_expansion_continuations",
    "build_reversal_fvg_corridor_scalps",
    "summarize_family_outcomes",
    "summarize_fvg_target_scalps",
    "build_family_complementarity",
    "r07_causal_audit",
]

# R06 adaptive lifecycle helpers are intentionally imported lazily by the R06
# research script; keep package boundary stable and avoid research-script imports.

from .r07 import (
    R07Config,
    build_reversal_confirmation_atlas,
    summarize_reversal_atlas,
    summarize_reversal_target_grid,
    summarize_family_target_grid,
    build_fvg_lifecycle,
    build_liquidity_expansion_continuations,
    build_reversal_fvg_corridor_scalps,
    summarize_family_outcomes,
    summarize_fvg_target_scalps,
    build_family_complementarity,
    r07_causal_audit,
)

from .r08 import (
    R08Config,
    build_bos_events,
    build_classical_ict_hierarchy,
    build_completed_trend_legs,
    build_multi_timeframe_hierarchy,
    build_trend_qualified_liquidity,
    build_projection_impact_atlas,
    r08_causal_audit,
    summarize_hierarchy,
    summarize_key_liquidity,
    summarize_projection_impact,
    summarize_trend_scales,
)

__all__.extend([
    "R08Config",
    "build_bos_events",
    "build_classical_ict_hierarchy",
    "build_completed_trend_legs",
    "build_multi_timeframe_hierarchy",
    "build_trend_qualified_liquidity",
    "build_projection_impact_atlas",
    "r08_causal_audit",
    "summarize_hierarchy",
    "summarize_key_liquidity",
    "summarize_projection_impact",
    "summarize_trend_scales",
])

# R09 ICT liquidity quality x execution atlas
from .r09 import (
    R09Config,
    build_physical_liquidity_sweeps,
    build_root_sweep_episodes,
    build_immediate_entries,
    build_reclaim_fvg_limit_entries,
    attach_r09_outcomes,
    summarize_execution_grid,
    summarize_quality_ladder,
    summarize_cascade_diagnostic,
    r09_causal_audit,
)

# R10 unified ICT liquidity engine
from .r10 import (
    R10Config,
    attach_risk_sizing,
    build_daily_partial_equity,
    build_structural_mss_upgrade_map,
    build_unified_reclaim_base,
    r10_causal_audit,
    select_single_position,
    simulate_unified_lifecycles,
    summarize_scenario,
    summarize_years,
)

# R11.1 continuous visible-liquidity path atlas (ETH 24/7)
from .r11 import (
    R11Config,
    build_visible_it_lt_liquidity,
    build_continuous_sweep_events,
    build_continuous_path_atlas,
    build_event_time_liquidity_snapshot,
    summarize_path_archetypes,
    summarize_first_sweep,
    r11_causal_audit,
)

__all__.extend([
    "R11Config",
    "build_visible_it_lt_liquidity",
    "build_continuous_sweep_events",
    "build_continuous_path_atlas",
    "build_event_time_liquidity_snapshot",
    "summarize_path_archetypes",
    "summarize_first_sweep",
    "r11_causal_audit",
])

# R12 completed-trend swing -> opposite-liquidity path atlas
from .r12 import (
    R12Config,
    prepare_completed_trend_contexts,
    build_completed_trend_physical_liquidity,
    build_root_sweep_events as build_r12_root_sweep_events,
    build_opposite_liquidity_paths,
    summarize_path_outcomes as summarize_r12_path_outcomes,
    summarize_root_taxonomy,
    summarize_success_failure_features,
    summarize_landmark_uplift,
    r12_causal_audit,
)

__all__.extend([
    "R12Config",
    "prepare_completed_trend_contexts",
    "build_completed_trend_physical_liquidity",
    "build_r12_root_sweep_events",
    "build_opposite_liquidity_paths",
    "summarize_r12_path_outcomes",
    "summarize_root_taxonomy",
    "summarize_success_failure_features",
    "summarize_landmark_uplift",
    "r12_causal_audit",
])

# R13 direct-reversal quality and causal entry discovery
from .r13 import (
    R13Config,
    attach_reversal_quality_features,
    build_entry_candidate_outcomes,
    build_feature_bin_atlas,
    data_coverage_audit,
    prepare_reversal_comparison_universe,
    r13_causal_audit,
    summarize_direct_failure_divergence,
    summarize_entry_models,
    summarize_entry_years,
)

__all__.extend([
    "R13Config",
    "attach_reversal_quality_features",
    "build_entry_candidate_outcomes",
    "build_feature_bin_atlas",
    "data_coverage_audit",
    "prepare_reversal_comparison_universe",
    "r13_causal_audit",
    "summarize_direct_failure_divergence",
    "summarize_entry_models",
    "summarize_entry_years",
])

# R14 completed-trend liquidity acceptance/continuation
from .r14 import (
    R14Config,
    attach_acceptance_features,
    build_continuation_entries,
    prepare_continuation_universe,
    r14_causal_audit,
    summarize_continuation_models,
    summarize_continuation_months,
    summarize_continuation_years,
)

__all__.extend([
    "R14Config",
    "attach_acceptance_features",
    "build_continuation_entries",
    "prepare_continuation_universe",
    "r14_causal_audit",
    "summarize_continuation_models",
    "summarize_continuation_months",
    "summarize_continuation_years",
])

# R15 SSL acceptance fixed-R first-passage diagnostic
from .r15 import (
    R15Config,
    build_fixed_r_first_passage,
    prepare_fixed_r_universe,
    r15_causal_audit,
    summarize_fixed_r,
    summarize_fixed_r_years,
)

__all__.extend([
    "R15Config",
    "build_fixed_r_first_passage",
    "prepare_fixed_r_universe",
    "r15_causal_audit",
    "summarize_fixed_r",
    "summarize_fixed_r_years",
])

# R16 SSL acceptance structural/behavioral stop atlas
from .r16 import (
    R16Config,
    build_stop_model_outcomes,
    prepare_stop_atlas_universe,
    r16_causal_audit,
    summarize_stop_models,
    summarize_stop_years,
)

__all__.extend([
    "R16Config",
    "build_stop_model_outcomes",
    "prepare_stop_atlas_universe",
    "r16_causal_audit",
    "summarize_stop_models",
    "summarize_stop_years",
])

# R17 aligned-trend pullback reclaim/re-acceleration path atlas
from .r17 import (
    R17Config,
    build_first_passage_paths,
    build_pullback_setup_atlas,
    build_structural_state,
    r17_causal_audit,
    summarize_path_models,
    summarize_path_years,
    summarize_setup_funnel,
)

__all__.extend([
    "R17Config",
    "build_first_passage_paths",
    "build_pullback_setup_atlas",
    "build_structural_state",
    "r17_causal_audit",
    "summarize_path_models",
    "summarize_path_years",
    "summarize_setup_funnel",
])

# R18 independent Binance-positioning unwind path atlas
from .r18 import (
    R18Config,
    build_positioning_unwind_events,
    build_positioning_unwind_paths,
    r18_causal_audit,
    r18_data_quality_audit,
    summarize_r18_funnel,
    summarize_r18_paths,
    summarize_r18_years,
)

__all__.extend([
    "R18Config",
    "build_positioning_unwind_events",
    "build_positioning_unwind_paths",
    "r18_causal_audit",
    "r18_data_quality_audit",
    "summarize_r18_funnel",
    "summarize_r18_paths",
    "summarize_r18_years",
])

# R19 positioning rebuild / continuation-resumption path atlas
from .r19 import (
    R19Config,
    build_positioning_rebuild_events,
    build_positioning_rebuild_paths,
    r19_causal_audit,
    summarize_r19_funnel,
    summarize_r19_paths,
    summarize_r19_years,
)

__all__.extend([
    "R19Config",
    "build_positioning_rebuild_events",
    "build_positioning_rebuild_paths",
    "r19_causal_audit",
    "summarize_r19_funnel",
    "summarize_r19_paths",
    "summarize_r19_years",
])

# R20 frozen LF V10B component visible-window falsification
from .r20 import (
    R20Config,
    build_r20_gate,
    prepare_r20_trades,
    r20_causal_audit,
    summarize_r20_components,
    summarize_r20_years,
)

__all__.extend([
    "R20Config",
    "build_r20_gate",
    "prepare_r20_trades",
    "r20_causal_audit",
    "summarize_r20_components",
    "summarize_r20_years",
])

# R21 canonical daily channel trend following
from .r21 import (
    R21Config,
    R21Model,
    build_daily_channel_features,
    r21_causal_audit,
    simulate_daily_channel,
    summarize_r21,
    summarize_r21_years,
)

__all__.extend([
    "R21Config",
    "R21Model",
    "build_daily_channel_features",
    "r21_causal_audit",
    "simulate_daily_channel",
    "summarize_r21",
    "summarize_r21_years",
])

# R22 BTC-led ETH catch-up first passage
from .r22 import (
    R22Config,
    build_r22_gate,
    build_catchup_events,
    build_cross_market_features,
    r22_causal_audit,
    simulate_catchup,
    summarize_r22,
    summarize_r22_years,
)

__all__.extend([
    "R22Config",
    "build_r22_gate",
    "build_catchup_events",
    "build_cross_market_features",
    "r22_causal_audit",
    "simulate_catchup",
    "summarize_r22",
    "summarize_r22_years",
])

# R23 frozen panic-wick structural Long falsification
from .r23 import (
    R23Config,
    build_panic_features,
    build_priority_union_events,
    build_r23_gate,
    r23_causal_audit,
    regularize_trade_bars,
    simulate_frozen_panic_long,
    summarize_r23,
    summarize_r23_years,
)

__all__.extend([
    "R23Config",
    "build_panic_features",
    "build_priority_union_events",
    "build_r23_gate",
    "r23_causal_audit",
    "regularize_trade_bars",
    "simulate_frozen_panic_long",
    "summarize_r23",
    "summarize_r23_years",
])

# R24 scheduled funding-window unwind
from .r24 import (
    R24Config,
    build_funding_window_events,
    build_r24_gate,
    r24_causal_audit,
    simulate_funding_unwind,
    summarize_r24,
    summarize_r24_years,
)

__all__.extend([
    "R24Config",
    "build_funding_window_events",
    "build_r24_gate",
    "r24_causal_audit",
    "simulate_funding_unwind",
    "summarize_r24",
    "summarize_r24_years",
])
