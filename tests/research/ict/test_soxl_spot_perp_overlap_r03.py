from __future__ import annotations

import pandas as pd

from src.research_common.ict.premarket_mss_fvg import ResearchConfig, make_synthetic_ict_day
from src.research_common.ict.premarket_mss_fvg_v2 import (
    build_all_premarket_levels_v2,
    build_signal_attempts_v2,
    build_sweep_events_v2,
)
from src.research_common.ict.spot_perp_overlap import (
    build_aligned_minute_paths,
    summarize_proxy_audit,
)


def _events(bars: pd.DataFrame):
    cfg = ResearchConfig()
    day = bars.index[0].date()
    levels = build_all_premarket_levels_v2(bars, [day], pivot_left=2, pivot_right=2)
    sweeps = build_sweep_events_v2(bars, levels)
    attempts = build_signal_attempts_v2(
        bars,
        sweeps,
        config=cfg,
        displacement_body_multipliers=[cfg.displacement_body_mult],
    )
    return sweeps, attempts


def test_constant_basis_proxy_passes_structural_overlap_gate() -> None:
    spot = make_synthetic_ict_day()
    perp = spot.copy()
    for col in ("open", "high", "low", "close"):
        perp[col] = pd.to_numeric(perp[col], errors="coerce") * 1.004
    aligned, daily = build_aligned_minute_paths(spot, perp)
    spot_sweeps, spot_attempts = _events(spot)
    perp_sweeps, perp_attempts = _events(perp)
    metrics, detail = summarize_proxy_audit(
        spot_ny=spot,
        perp_ny=perp,
        aligned=aligned,
        daily_paths=daily,
        spot_sweeps=spot_sweeps,
        perp_sweeps=perp_sweeps,
        spot_attempts=spot_attempts,
        perp_attempts=perp_attempts,
    )
    assert detail["verdict"] == "PASS"
    assert metrics["pass"].all()


def test_missing_structure_can_fail_even_when_price_rows_overlap() -> None:
    spot = make_synthetic_ict_day()
    perp = spot.copy()
    # Flatten post-08:30 path enough to remove the synthetic sweep/MSS while
    # leaving timestamps aligned. This verifies the audit is not just a row-count check.
    mask = (perp.index.hour > 8) | ((perp.index.hour == 8) & (perp.index.minute >= 30))
    anchor = float(perp.loc[~mask, "close"].iloc[-1])
    for col in ("open", "high", "low", "close"):
        perp.loc[mask, col] = anchor
    aligned, daily = build_aligned_minute_paths(spot, perp)
    spot_sweeps, spot_attempts = _events(spot)
    perp_sweeps, perp_attempts = _events(perp)
    metrics, detail = summarize_proxy_audit(
        spot_ny=spot,
        perp_ny=perp,
        aligned=aligned,
        daily_paths=daily,
        spot_sweeps=spot_sweeps,
        perp_sweeps=perp_sweeps,
        spot_attempts=spot_attempts,
        perp_attempts=perp_attempts,
    )
    assert detail["verdict"] == "FAIL"
    assert not metrics.loc[metrics["metric"].isin(["external_sweep_key_jaccard", "base_setup_key_jaccard"]), "pass"].all()


def test_stock_minute_densification_is_forward_only_and_same_day() -> None:
    from src.research_common.ict.spot_perp_overlap import densify_equity_minutes_causally

    idx = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-06-02 04:05", tz="America/New_York"),
            pd.Timestamp("2026-06-02 04:07", tz="America/New_York"),
            pd.Timestamp("2026-06-02 16:29", tz="America/New_York"),
        ]
    )
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 110.0],
            "high": [100.5, 101.5, 110.5],
            "low": [99.5, 100.5, 109.5],
            "close": [100.2, 101.2, 110.2],
            "volume": [10.0, 20.0, 30.0],
        },
        index=idx,
    )
    dense = densify_equity_minutes_causally(frame)
    assert pd.Timestamp("2026-06-02 04:04", tz="America/New_York") not in dense.index
    filled = dense.loc[pd.Timestamp("2026-06-02 04:06", tz="America/New_York")]
    assert bool(filled["is_synthetic_no_trade_bar"])
    assert float(filled["close"]) == 100.2
    assert float(filled["volume"]) == 0.0
