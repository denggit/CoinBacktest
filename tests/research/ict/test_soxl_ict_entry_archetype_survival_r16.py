from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from src.research_common.ict.daily_liquidity_path import DailyLiquidityPathConfig, build_daily_path_outcomes, build_daily_range_definitions
from src.research_common.ict.entry_archetype_survival import (
    attach_approach_compression_features,
    attach_causal_entry_state,
    attach_path_metadata,
    build_reclaim_entry_candidates,
    replay_entry_survival,
    summarize_entry_archetypes,
)


def _load_r16_module():
    root = Path(__file__).resolve().parents[3]
    path = root / "research" / "ict" / "soxl_premarket_mss_fvg" / "16_entry_archetype_survival_atlas.py"
    spec = importlib.util.spec_from_file_location("soxl_r16", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _reclaim_fixture():
    mod = _load_r16_module()
    bars = mod._synthetic_day()
    day = pd.Timestamp("2026-08-05").date()
    ranges = build_daily_range_definitions(bars, [day], config=DailyLiquidityPathConfig())
    paths = build_daily_path_outcomes(bars, ranges, config=DailyLiquidityPathConfig())
    paths = attach_approach_compression_features(bars, paths)
    entries = build_reclaim_entry_candidates(bars, paths)
    entries = attach_path_metadata(entries, paths)
    entries = attach_causal_entry_state(bars, entries)
    return mod, bars, paths, entries


def test_reclaim_entry_uses_only_confirmed_reclaim_and_next_open():
    _, bars, _, entries = _reclaim_fixture()
    market = entries.loc[entries["entry_archetype"].eq("raid_reclaim_next_open_market")]
    assert not market.empty
    row = market.iloc[0]
    signal = pd.Timestamp(row["entry_available_time"])
    reclaim_start = pd.Timestamp(row["reclaim_bar_start"])
    assert signal == reclaim_start + pd.Timedelta(minutes=1)
    assert signal >= pd.Timestamp(row["first_raid_time"])
    assert float(row["stop_price"]) != float(row["entry_price"])


def test_same_path_has_at_most_one_candidate_per_entry_archetype():
    _, _, _, entries = _reclaim_fixture()
    counts = entries.groupby(["path_event_id", "entry_archetype"]).size()
    assert int(counts.max()) == 1


def test_survival_replay_exposes_immediate_stop_and_range_milestones():
    _, bars, _, entries = _reclaim_fixture()
    replay = replay_entry_survival(bars, entries)
    assert not replay.empty
    required = {
        "stop_within_1m", "stop_within_3m", "stop_within_5m", "stop_within_10m",
        "milestone_50_before_stop", "milestone_75_before_stop", "milestone_100_before_stop",
        "net_return_exit_50", "net_return_exit_75", "net_return_exit_100",
    }
    assert required.issubset(replay.columns)
    score = summarize_entry_archetypes(replay)
    assert "immediate_stop_5m_rate" in score.columns
    assert "profit_factor_exit_50" in score.columns


def test_r16_has_no_swing_dollar_cap_gate_and_keeps_limit_plus_market_entries():
    mod = _load_r16_module()
    src = Path(mod.__file__).read_text(encoding="utf-8")
    common = Path(mod.__file__).resolve().parents[3] / "src" / "research_common" / "ict" / "entry_archetype_survival.py"
    common_src = common.read_text(encoding="utf-8")
    assert "Swing +/- $0.10 is not used" in src
    assert "raid_reclaim_level_retest_limit" in common_src
    assert "close_break_next_open_market" in common_src
    assert "ob_fvg_overlap" in common_src


def test_attach_path_metadata_fills_row_level_nan_after_heterogeneous_concat():
    paths = pd.DataFrame([
        {
            "path_event_id": "PATH|2026-08-05|prominent_15m_pair_0830|000001",
            "ny_date": "2026-08-05",
            "range_model": "prominent_15m_pair_0830",
            "trade_side": "LONG",
            "source_level_price": 131.65,
            "target_price": 140.85,
            "lower_price": 131.65,
            "upper_price": 140.85,
            "range_width_abs": 9.20,
            "traversal_complete": True,
        },
        {
            "path_event_id": "PATH|2026-08-06|early_extreme_0400_0830|000002",
            "ny_date": "2026-08-06",
            "range_model": "early_extreme_0400_0830",
            "trade_side": "SHORT",
            "source_level_price": 150.0,
            "target_price": 140.0,
            "lower_price": 140.0,
            "upper_price": 150.0,
            "range_width_abs": 10.0,
            "traversal_complete": False,
        },
    ])
    # Mimic pd.concat across heterogeneous archetypes: one family carries
    # range_model, another has the union-schema column but NaN on its rows.
    entries = pd.DataFrame([
        {
            "path_event_id": "PATH|2026-08-05|prominent_15m_pair_0830|000001",
            "event_id": "PATH|2026-08-05|prominent_15m_pair_0830|000001",
            "entry_archetype": "mss_first_visible_break_fvg_near",
            "range_model": pd.NA,
            "range_width_abs": pd.NA,
            "target_price": pd.NA,
        },
        {
            "path_event_id": "PATH|2026-08-06|early_extreme_0400_0830|000002",
            "event_id": "PATH|2026-08-06|early_extreme_0400_0830|000002",
            "entry_archetype": "raid_reclaim_next_open_market",
            "range_model": "early_extreme_0400_0830",
            "range_width_abs": 10.0,
            "target_price": 140.0,
        },
    ])

    fixed = attach_path_metadata(entries, paths)
    first = fixed.iloc[0]
    assert first["range_model"] == "prominent_15m_pair_0830"
    assert float(first["range_width_abs"]) == 9.20
    assert float(first["target_price"]) == 140.85
    assert bool(first["traversal_complete"]) is True
    second = fixed.iloc[1]
    assert second["range_model"] == "early_extreme_0400_0830"
    assert float(second["range_width_abs"]) == 10.0


def test_attach_path_metadata_rejects_conflicting_existing_metadata():
    paths = pd.DataFrame([
        {"path_event_id": "P1", "range_model": "prominent_15m_pair_0830", "range_width_abs": 10.0}
    ])
    entries = pd.DataFrame([
        {"path_event_id": "P1", "event_id": "P1", "range_model": "early_extreme_0400_0830", "range_width_abs": 10.0}
    ])
    try:
        attach_path_metadata(entries, paths)
    except ValueError as exc:
        assert "range_model" in str(exc)
    else:
        raise AssertionError("expected conflicting path metadata to fail")
