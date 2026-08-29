from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from src.research_common.ict.daily_liquidity_path import (
    DailyLiquidityPathConfig,
    build_daily_path_outcomes,
    build_daily_range_definitions,
    path_events_to_sweep_events,
)


def _load_r15_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "research" / "ict" / "soxl_premarket_mss_fvg" / "15_daily_liquidity_traversal_path_atlas.py"
    spec = importlib.util.spec_from_file_location("soxl_r15", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_daily_range_is_frozen_before_path_raid():
    mod = _load_r15_module()
    bars = mod._synthetic_day()
    day = pd.Timestamp("2026-08-05").date()
    ranges = build_daily_range_definitions(bars, [day], config=DailyLiquidityPathConfig())
    paths = build_daily_path_outcomes(bars, ranges, config=DailyLiquidityPathConfig())
    early = paths.loc[paths["range_model"].eq("early_extreme_0400_0830")].iloc[0]
    assert pd.Timestamp(early["range_available_time"]) == pd.Timestamp("2026-08-05 08:30", tz="America/New_York")
    assert pd.Timestamp(early["first_raid_time"]) >= pd.Timestamp(early["range_available_time"])


def test_first_raid_to_opposite_path_becomes_single_sweep_event():
    mod = _load_r15_module()
    bars = mod._synthetic_day()
    day = pd.Timestamp("2026-08-05").date()
    ranges = build_daily_range_definitions(bars, [day], config=DailyLiquidityPathConfig())
    paths = build_daily_path_outcomes(bars, ranges, config=DailyLiquidityPathConfig())
    early = paths.loc[paths["range_model"].eq("early_extreme_0400_0830")]
    assert len(early) == 1
    row = early.iloc[0]
    assert bool(row["traversal_complete"])
    assert row["path_archetype"] == "first_raid_to_opposite_boundary"
    sweeps = path_events_to_sweep_events(early)
    assert len(sweeps) == 1
    assert sweeps.iloc[0]["trade_side"] in {"LONG", "SHORT"}
    assert pd.Timestamp(sweeps.iloc[0]["sweep_time"]) == pd.Timestamp(row["first_raid_time"])


def test_r15_does_not_require_equal_target_or_dollar_cap():
    mod = _load_r15_module()
    args = mod.parse_args(["--self-test"])
    assert args.round_trip_cost == 0.0011
    source = Path(mod.__file__).read_text(encoding="utf-8")
    assert "target-state or Swing-$0.10 gating" in source
    assert "Swing +/- $0.10 is not a gate" in source


def test_success_failure_features_preserves_overlapping_path_metadata_names():
    mod = _load_r15_module()
    primary = pd.DataFrame([
        {
            "event_id": "PATH|2026-08-05|early_extreme_0400_0830|000001",
            "range_model": "early_extreme_0400_0830",
            "execution_tf": "1m",
            "traversal_complete": True,
            "terminal_to_break_minutes": 8.0,
            "path_efficiency": 0.75,
        }
    ])
    paths = pd.DataFrame([
        {
            "path_event_id": "PATH|2026-08-05|early_extreme_0400_0830|000001",
            "range_model": "early_extreme_0400_0830",
            "traversal_complete": True,
            "path_archetype": "first_raid_to_opposite_boundary",
            "range_width_abs": 10.0,
        }
    ])
    out = mod._success_failure_features(primary, paths)
    assert len(out) == 1
    assert out.iloc[0]["range_model"] == "early_extreme_0400_0830"
    assert out.iloc[0]["execution_tf"] == "1m"
    assert bool(out.iloc[0]["traversal_complete"])
    assert out.iloc[0]["n_events"] == 1
