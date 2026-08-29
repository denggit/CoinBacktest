from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict.premarket_mss_fvg import make_synthetic_ict_day, replay_attempts, BASE_SCENARIO
from src.research_common.ict.premarket_mss_fvg_v2 import SweepEpisodeConfig, build_all_premarket_levels_v2, build_sweep_events_v2
from src.research_common.ict.premarket_mss_fvg_v4 import ICTDisplacementDiscoveryConfig, build_signal_attempts_v4
from src.research_common.ict.semantic_gap import (
    SemanticGapConfig,
    attach_causal_semantic_features,
    attach_outcome_path_labels,
    build_semantic_causal_audit,
    build_semantic_feature_atlas,
)


def _attempts_and_bars():
    bars = make_synthetic_ict_day()
    day = pd.Timestamp("2026-06-02").date()
    levels = build_all_premarket_levels_v2(
        bars, [day], pivot_left=2, pivot_right=2, episode_config=SweepEpisodeConfig()
    )
    sweeps = build_sweep_events_v2(bars, levels)
    attempts, _ = build_signal_attempts_v4(
        bars, sweeps, config=ICTDisplacementDiscoveryConfig(execution_timeframes=(1, 5))
    )
    return bars, attempts


def test_r07_semantic_features_do_not_filter_candidates() -> None:
    bars, attempts = _attempts_and_bars()
    assert not attempts.empty
    before = attempts["attempt_id"].tolist()
    enriched = attach_causal_semantic_features(attempts, bars)
    assert enriched["attempt_id"].tolist() == before
    assert len(enriched) == len(attempts)
    assert "semantic_structure_shape" in enriched.columns
    assert "semantic_reclaim_minutes_from_sweep" in enriched.columns


def test_r07_semantic_features_are_available_by_signal_and_no_outcome_leakage() -> None:
    bars, attempts = _attempts_and_bars()
    enriched = attach_causal_semantic_features(attempts, bars)
    audit = build_semantic_causal_audit(enriched)
    assert bool(audit["passed"].all())
    assert not any(c.startswith("outcome_") for c in enriched.columns)


def test_r07_outcome_labels_are_post_replay_only() -> None:
    bars, attempts = _attempts_and_bars()
    enriched = attach_causal_semantic_features(attempts, bars)
    replayed = replay_attempts(bars, enriched, scenario=BASE_SCENARIO, round_trip_cost=0.0011, risk_fraction=0.01, max_notional_multiple=2.0)
    labeled = attach_outcome_path_labels(replayed)
    assert any(c.startswith("outcome_") for c in labeled.columns)
    filled = labeled.loc[labeled["filled"].fillna(False).astype(bool)]
    if not filled.empty:
        expected = pd.to_numeric(filled["mfe_r"], errors="coerce") >= 1.0
        assert (filled["outcome_reached_1r"].to_numpy() == expected.to_numpy()).all()


def test_r07_frozen_feature_edges_use_discovery_only() -> None:
    # Build a controlled lifecycle where the 2025/2026 values are extreme. If
    # edges accidentally use the full sample they would move far above 5.
    rows = []
    for year, values in [(2024, np.linspace(1, 5, 50)), (2025, np.linspace(100, 200, 20)), (2026, np.linspace(300, 400, 20))]:
        for i, value in enumerate(values):
            rows.append({
                "ny_date": f"{year}-06-{(i % 20) + 1:02d}",
                "execution_tf": "1m",
                "liquidity_family": "premarket_extreme",
                "filled": True,
                "net_return": 0.001 if i % 2 == 0 else -0.001,
                "net_r": 0.5 if i % 2 == 0 else -0.5,
                "mfe_r": 1.2,
                "mae_r": -0.4,
                "outcome_reached_0_5r": True,
                "outcome_reached_1r": True,
                "outcome_reached_2r": False,
                "outcome_reached_3r": False,
                "outcome_target_hit": False,
                "outcome_stop_hit": False,
                "outcome_immediate_failure_15m": False,
                "outcome_favorable_then_failed_1r": False,
                "outcome_favorable_then_failed_2r": False,
                "semantic_terminal_extension_pct": float(value),
            })
    lifecycle = pd.DataFrame(rows)
    edges, perf = build_semantic_feature_atlas(
        lifecycle,
        config=SemanticGapConfig(min_discovery_samples=40, quantile_bins=5),
        features=("semantic_terminal_extension_pct",),
    )
    assert not edges.empty
    finite_edges = pd.to_numeric(edges.filter(like="edge_").stack(), errors="coerce").dropna()
    assert finite_edges.max() <= 5.0
    assert {"2025", "2026_late_holdout"}.issubset(set(perf["analysis_period"]))


def test_r07_v4_records_fvg_count_without_changing_signal_semantics() -> None:
    bars, attempts = _attempts_and_bars()
    assert not attempts.empty
    assert (pd.to_numeric(attempts["directional_fvg_count_to_signal"]) >= 1).all()
    assert (pd.to_numeric(attempts["selected_fvg_sequence_rank"]) >= 1).all()
