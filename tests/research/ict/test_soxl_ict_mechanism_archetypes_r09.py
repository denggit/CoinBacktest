from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict.equal_liquidity import (
    EqualLiquidityConfig,
    attach_equal_pool_context,
    build_equal_liquidity_pools,
    build_equal_pool_causal_audit,
)
from src.research_common.ict.liquidity_maturity import attach_liquidity_maturity_features
from src.research_common.ict.mechanism_archetypes import (
    MechanismArchetypeConfig,
    attach_mechanism_archetypes,
    build_mechanism_causal_audit,
    fit_mechanism_distribution_edges,
)
from src.research_common.ict.premarket_mss_fvg import make_synthetic_ict_day
from src.research_common.ict.premarket_mss_fvg_v2 import (
    SweepEpisodeConfig,
    build_all_premarket_levels_v2,
    build_sweep_events_v2,
)
from src.research_common.ict.premarket_mss_fvg_v4 import (
    ICTDisplacementDiscoveryConfig,
    build_signal_attempts_v4,
)
from src.research_common.ict.semantic_gap import attach_causal_semantic_features


def _base():
    bars = make_synthetic_ict_day()
    day = pd.Timestamp("2026-06-02").date()
    levels = build_all_premarket_levels_v2(
        bars, [day], pivot_left=2, pivot_right=2, episode_config=SweepEpisodeConfig()
    )
    levels = levels.copy()
    levels["liquidity_family"] = levels["level_type"].astype(str)
    return bars, day, levels


def test_r09_equal_pool_is_real_causal_swing_cluster_and_additive_family() -> None:
    bars, day, levels = _base()
    pools = build_equal_liquidity_pools(
        bars, [day], levels,
        config=EqualLiquidityConfig(source_timeframes=(1, 5, 15), pivot_left=1, pivot_right=1),
    )
    assert not pools.empty
    assert (pd.to_numeric(pools["eq_member_count"]) >= 2).all()
    cutoff = pd.Timestamp("2026-06-02 08:30", tz="America/New_York")
    assert (pd.to_datetime(pools["level_available_time"]) <= cutoff).all()
    assert set(pools["level_type"]).issubset({"equal_high_pool", "equal_low_pool"})

    combined = pd.concat([levels, pools], ignore_index=True, sort=False)
    sweeps = build_sweep_events_v2(bars, combined)
    assert "equal_liquidity_pool" in set(sweeps["liquidity_family"].astype(str))
    attempts, _ = build_signal_attempts_v4(
        bars, sweeps, config=ICTDisplacementDiscoveryConfig(execution_timeframes=(1,))
    )
    assert not attempts.empty
    assert "equal_liquidity_pool" in set(attempts["liquidity_family"].astype(str))


def test_r09_equal_context_and_mechanism_tags_do_not_filter_existing_attempts() -> None:
    bars, day, levels = _base()
    pools = build_equal_liquidity_pools(bars, [day], levels)
    sweeps = build_sweep_events_v2(bars, levels)
    attempts, _ = build_signal_attempts_v4(
        bars, sweeps, config=ICTDisplacementDiscoveryConfig(execution_timeframes=(1, 5))
    )
    attempts = attach_causal_semantic_features(attempts, bars)
    attempts = attach_liquidity_maturity_features(attempts, bars)
    ids = attempts["attempt_id"].astype(str).tolist()
    attempts = attach_equal_pool_context(attempts, pools)
    assert attempts["attempt_id"].astype(str).tolist() == ids

    # Synthetic single day has no discovery sample edges; the tagger must still
    # preserve every candidate and can use non-edge equal-pool context tags.
    tagged = attach_mechanism_archetypes(attempts, pd.DataFrame())
    assert tagged["attempt_id"].astype(str).tolist() == ids
    assert len(tagged) == len(attempts)
    assert not any(c.startswith("outcome_") for c in tagged.columns)
    assert bool(build_equal_pool_causal_audit(pools, tagged)["passed"].all())
    assert bool(build_mechanism_causal_audit(tagged)["passed"].all())


def test_r09_mechanism_edges_use_discovery_distribution_not_forward_or_pnl() -> None:
    rows = []
    for period, dates, vals in [
        ("disc", pd.date_range("2023-07-01", periods=60, freq="D"), np.linspace(1, 5, 60)),
        ("fwd", pd.date_range("2025-01-01", periods=20, freq="D"), np.linspace(100, 200, 20)),
        ("hold", pd.date_range("2026-01-01", periods=20, freq="D"), np.linspace(300, 400, 20)),
    ]:
        for i, (d, v) in enumerate(zip(dates, vals)):
            rows.append({
                "ny_date": str(d.date()),
                "execution_tf": "1m",
                "liquidity_family": "premarket_extreme",
                # PnL deliberately unrelated and extreme; fitter must ignore it.
                "net_return": 99.0 if period != "disc" else -99.0,
                "maturity_terminal_extension_bp": float(v),
            })
    df = pd.DataFrame(rows)
    edges = fit_mechanism_distribution_edges(
        df,
        config=MechanismArchetypeConfig(min_discovery_samples=40),
        features=("maturity_terminal_extension_bp",),
    )
    assert len(edges) == 1
    row = edges.iloc[0]
    assert float(row["q75"]) <= 5.0
