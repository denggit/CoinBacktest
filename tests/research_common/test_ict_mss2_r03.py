from __future__ import annotations

import numpy as np
import pandas as pd

import src.research_common.ict_mss2.r03 as r03mod
from src.research_common.ict_mss2 import (
    R03Config,
    attach_overlay_structural_outcomes,
    build_fvg_execution_overlay_attempts,
    build_hybrid_5050_outcomes,
    build_tradebar_microstructure_features,
    first_pool_threshold_crossing_trades,
    r03_globalize_legacy_trade_ids,
)


def _bars(rows, start="2025-01-01 00:00"):
    idx = pd.date_range(start, periods=len(rows), freq="1min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def test_legacy_trade_ids_are_repaired_positionally_and_globally() -> None:
    features = pd.DataFrame(
        {
            "trade_event_id": ["R02_TRADE_000000001", "R02_TRADE_000000001"],
            "execution_minutes": [1, 5],
            "episode_id": ["E1", "E2"],
        }
    )
    labels = pd.DataFrame(
        {
            "trade_event_id": ["R02_TRADE_000000001", "R02_TRADE_000000001"],
            "target_htf240_outcome": ["target", "stop"],
        }
    )
    f, l = r03_globalize_legacy_trade_ids(features, labels)
    assert l is not None
    assert list(f["trade_event_id"]) == ["R02_1M_TRADE_000000001", "R02_5M_TRADE_000000001"]
    assert list(l["trade_event_id"]) == list(f["trade_event_id"])
    assert not f["trade_event_id"].duplicated().any()


def test_first_pool_threshold_trade_keeps_first_causal_crossing_per_episode_tf() -> None:
    frame = pd.DataFrame(
        [
            {"trade_event_id": "A", "episode_id": "E1", "trade_direction": 1, "execution_minutes": 5, "trigger_type": "episode_reclaim", "price_pools_10p0bp_cum": 3, "entry_pos_1m": 10},
            {"trade_event_id": "B", "episode_id": "E1", "trade_direction": 1, "execution_minutes": 5, "trigger_type": "episode_reclaim", "price_pools_10p0bp_cum": 4, "entry_pos_1m": 20},
            {"trade_event_id": "C", "episode_id": "E1", "trade_direction": 1, "execution_minutes": 5, "trigger_type": "episode_reclaim", "price_pools_10p0bp_cum": 5, "entry_pos_1m": 30},
            {"trade_event_id": "D", "episode_id": "E2", "trade_direction": -1, "execution_minutes": 5, "trigger_type": "episode_reclaim", "price_pools_10p0bp_cum": 5, "entry_pos_1m": 15},
        ]
    )
    out = first_pool_threshold_crossing_trades(frame, threshold=4, execution_minutes=(5,))
    assert list(out["trade_event_id"]) == ["B"]


def test_tradebar_features_exclude_left_labelled_bar_starting_at_decision(monkeypatch) -> None:
    rows = []
    for i in range(12):
        notional = 1_000.0
        rows.append(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "notional": notional,
                "buy_notional": 400.0,
                "sell_notional": 600.0,
                "delta_notional": -200.0,
                "trades_count": 10.0,
                "large_buy_notional": 0.0,
                "large_sell_notional": 0.0,
                "max_trade_notional": 100.0,
            }
        )
    idx = pd.date_range("2025-01-01 00:00", periods=len(rows), freq="1min")
    fake = pd.DataFrame(rows, index=idx)
    fake.loc[pd.Timestamp("2025-01-01 00:10"), "notional"] = 9_999_999.0  # unavailable at 00:10 decision

    class FakeLoader:
        def __init__(self, *args, **kwargs):
            pass

        def load_local_data(self, start_date=None, end_date=None):
            return fake.copy()

    monkeypatch.setattr(r03mod, "OKXTradeBarLoader", FakeLoader)
    checkpoints = pd.DataFrame(
        {
            "checkpoint_id": ["T1"],
            "decision_time": [pd.Timestamp("2025-01-01 00:10")],
            "episode_start_time": [pd.Timestamp("2025-01-01 00:05")],
        }
    )
    features, audit = build_tradebar_microstructure_features(checkpoints, config=R03Config(), show_progress=False)
    row = features.iloc[0]
    assert bool(row["tb_causal_valid"])
    assert row["tb_last_source_time"] == pd.Timestamp("2025-01-01 00:09")
    assert float(row["tb_episode_notional"]) == 5_000.0
    assert int(audit["causal_bad"].sum()) == 0


def test_fvg_market_limit_and_hybrid_share_signal_time_target() -> None:
    rows = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (98, 96, 94, 95),   # threshold sweep stage; high=96
        (95, 97, 95, 96),
        (97, 100, 97, 99),  # bullish FVG: low 97 > high[4] 96
        (99, 102, 98.5, 101),  # market entry at this open
        (101, 103, 97.0, 102), # proximal limit 97 fills
        (102, 108, 101, 107),
        (107, 111, 106, 110),  # 4H target 110 hits
        (110, 112, 109, 111),
    ]
    bars = _bars(rows)
    stages = pd.DataFrame(
        [
            {
                "stage_id": "S1",
                "episode_id": "E1",
                "trade_direction": 1,
                "sweep_pos_1m": 4,
                "sweep_bar_time_1m": bars.index[4],
                "episode_start_pos_1m": 4,
                "price_pools_10p0bp_cum": 4,
                "max_source_timeframe_min_cum": 240,
            }
        ]
    )
    lifecycle = pd.DataFrame(
        [
            {"pivot_side": "high", "active_pos_1m": 0, "level_price": 110.0, "source_timeframe_min": 240, "sweep_pos_1m": 10},
            {"pivot_side": "low", "active_pos_1m": 0, "level_price": 90.0, "source_timeframe_min": 240, "sweep_pos_1m": -1},
        ]
    )
    cfg = R03Config(fvg_signal_wait_minutes=20, fvg_limit_wait_minutes=20, execution_censor_minutes=20)
    attempts = build_fvg_execution_overlay_attempts(bars, stages, execution_minutes=1, config=cfg, show_progress=False)
    assert set(attempts["entry_kind"]) == {"market_next_open", "fvg_limit"}
    market = attempts.loc[attempts["entry_kind"].eq("market_next_open")].iloc[0]
    limit = attempts.loc[attempts["entry_kind"].eq("fvg_limit")].iloc[0]
    assert market["signal_available_time"] == pd.Timestamp("2025-01-01 00:07")
    assert market["entry_time"] == pd.Timestamp("2025-01-01 00:07")
    assert limit["entry_time"] == pd.Timestamp("2025-01-01 00:08")

    outcomes = attach_overlay_structural_outcomes(bars, lifecycle, attempts, config=cfg, show_progress=False)
    targets = outcomes["target_htf240_price"].dropna().unique()
    assert len(targets) == 1 and abs(float(targets[0]) - 110.0) < 1e-9
    assert set(outcomes.loc[outcomes["entry_fill_flag"].eq(1), "target_htf240_outcome"]) == {"target"}
    hybrid = build_hybrid_5050_outcomes(bars, outcomes, config=cfg)
    assert len(hybrid) == 1
    assert int(hybrid.iloc[0]["limit_filled_before_market_exit"]) == 1
    assert np.isfinite(float(hybrid.iloc[0]["hybrid_net_execution_cost2x"]))


def test_r03_position_alignment_audit_detects_shifted_bar_origin() -> None:
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "research" / "ict" / "mss2" / "03_liquidity_stack_orderflow_execution.py"
    spec = importlib.util.spec_from_file_location("ict_mss2_r03_script", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    bars = _bars([(100, 101, 99, 100)] * 8, start="2025-01-01 00:00")
    stages = pd.DataFrame(
        [{
            "sweep_pos_1m": 4,
            "sweep_bar_time_1m": bars.index[4],
            "episode_start_pos_1m": 2,
            "episode_start_time_1m": bars.index[2],
        }]
    )
    lifecycle = pd.DataFrame(
        [{
            "active_pos_1m": 1,
            "initial_available_time": bars.index[1],
            "sweep_pos_1m": 6,
            "sweep_bar_time_1m": bars.index[6],
        }]
    )
    ok = module._position_alignment_audit(bars, stages, lifecycle)
    assert int(ok["violations"].sum()) == 0

    shifted = bars.iloc[1:].copy()
    bad = module._position_alignment_audit(shifted, stages, lifecycle)
    assert int(bad["violations"].sum()) > 0


def test_r032_checkpoint_union_keeps_distinct_ge3_and_ge4_stage_rows() -> None:
    from src.research_common.ict_mss2 import build_microstructure_checkpoint_union, microstructure_feature_join_audit

    candidates = pd.DataFrame(
        [
            {"trade_event_id": "E1_GE3", "cohort": "expand_ge3", "signal_available_time": "2025-01-01 00:05", "episode_start_time_1m": "2025-01-01 00:00"},
            {"trade_event_id": "E1_GE4", "cohort": "core_ge4", "signal_available_time": "2025-01-01 00:10", "episode_start_time_1m": "2025-01-01 00:00"},
            {"trade_event_id": "E2_SAME", "cohort": "expand_ge3", "signal_available_time": "2025-01-01 01:05", "episode_start_time_1m": "2025-01-01 01:00"},
            {"trade_event_id": "E2_SAME", "cohort": "core_ge4", "signal_available_time": "2025-01-01 01:05", "episode_start_time_1m": "2025-01-01 01:00"},
        ]
    )
    checkpoints, _ = build_microstructure_checkpoint_union(candidates)
    assert set(checkpoints["checkpoint_id"]) == {"E1_GE3", "E1_GE4", "E2_SAME"}
    assert len(checkpoints) == 3

    features = pd.DataFrame({"checkpoint_id": ["E1_GE3", "E1_GE4", "E2_SAME"], "x": [1, 2, 3]})
    audit = microstructure_feature_join_audit(checkpoints, features, module="fake")
    assert bool(audit["passed"].astype(int).eq(1).all())

    missing = features.iloc[:2].copy()
    bad = microstructure_feature_join_audit(checkpoints, missing, module="fake")
    row = bad.loc[bad["check"].eq("missing_requested_checkpoint_ids")].iloc[0]
    assert int(row["value"]) == 1 and int(row["passed"]) == 0


def test_r032_core_execution_overlay_ties_baseline_and_uses_same_target_stop() -> None:
    from src.research_common.ict_mss2 import build_core_reclaim_execution_overlays

    # Base 5m reclaim enters at 00:05 open=100. A bullish 1m FVG is confirmed
    # at 00:07 (low[6] > high[4]) and becomes executable at 00:07.  Its proximal
    # 99 limit fills at 00:08.  The frozen R02 target 110 then hits.
    rows = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (98, 98, 94, 96),
        (100, 101, 99, 100),  # base reclaim market entry at 00:05 open 100
        (100, 102, 99, 101),  # bullish FVG vs bar4 high 98: low=99 > 98
        (101, 104, 100, 103), # FVG signal available at 00:07; market entry open 101
        (103, 104, 98.5, 102),# limit 99 fills
        (102, 108, 101, 107),
        (107, 111, 106, 110), # frozen target hits
        (110, 112, 109, 111),
    ]
    bars = _bars(rows)
    stop = 93.0
    target = 110.0
    base_entry = 100.0
    base_gross = target / base_entry - 1.0
    core = pd.DataFrame(
        [{
            "trade_event_id": "R02_5M_TRADE_1", "episode_id": "E1", "stage_id": "S1",
            "trade_direction": 1, "signal_available_time": bars.index[5],
            "entry_pos_1m": 5, "entry_time": bars.index[5], "entry_price": base_entry,
            "stop_price": stop, "target_htf240_price": target,
            "target_htf240_outcome": "target", "target_htf240_gross_return": base_gross,
            "year": 2025, "quarter": "2025Q1", "price_pools_10p0bp_cum": 4,
            "max_source_timeframe_min_cum": 240,
        }]
    )
    cfg = R03Config(fvg_execution_minutes=(1,), fvg_signal_wait_minutes=20, fvg_limit_wait_minutes=20, execution_censor_minutes=20)
    overlay, tie = build_core_reclaim_execution_overlays(bars, core, fvg_minutes=1, config=cfg, show_progress=False)
    assert len(tie) == 1
    assert int(tie.iloc[0]["outcome_match"]) == 1
    assert int(tie.iloc[0]["gross_match"]) == 1
    assert set(overlay["execution_variant"]) == {
        "reclaim_market", "post_reclaim_fvg_market", "post_reclaim_fvg_limit", "hybrid_reclaim_market_fvg_limit"
    }
    assert overlay["stop_price"].nunique() == 1 and float(overlay["stop_price"].iloc[0]) == stop
    assert overlay["target_htf240_price"].nunique() == 1 and float(overlay["target_htf240_price"].iloc[0]) == target
    limit = overlay.loc[overlay["execution_variant"].eq("post_reclaim_fvg_limit")].iloc[0]
    assert int(limit["entry_fill_flag"]) == 1
    assert abs(float(limit["entry_price"]) - 99.0) < 1e-9
    hybrid = overlay.loc[overlay["execution_variant"].eq("hybrid_reclaim_market_fvg_limit")].iloc[0]
    assert int(hybrid["limit_filled_flag"]) == 1


def test_r032_no_fvg_keeps_all_execution_opportunity_rows() -> None:
    from src.research_common.ict_mss2 import build_core_reclaim_execution_overlays

    bars = _bars([(100, 101, 99, 100)] * 12)
    core = pd.DataFrame(
        [{
            "trade_event_id": "R02_5M_TRADE_NOFVG", "episode_id": "E1", "stage_id": "S1",
            "trade_direction": 1, "signal_available_time": bars.index[2],
            "entry_pos_1m": 2, "entry_time": bars.index[2], "entry_price": 100.0,
            "stop_price": 90.0, "target_htf240_price": 110.0,
            "target_htf240_outcome": "censored", "target_htf240_gross_return": np.nan,
            "year": 2025, "quarter": "2025Q1", "price_pools_10p0bp_cum": 4,
            "max_source_timeframe_min_cum": 240,
        }]
    )
    cfg = R03Config(fvg_execution_minutes=(1,), fvg_signal_wait_minutes=5, fvg_limit_wait_minutes=5, execution_censor_minutes=8)
    overlay, tie = build_core_reclaim_execution_overlays(bars, core, fvg_minutes=1, config=cfg, show_progress=False)
    assert len(tie) == 1 and int(tie.iloc[0]["outcome_match"]) == 1
    assert set(overlay["execution_variant"]) == {
        "reclaim_market", "post_reclaim_fvg_market", "post_reclaim_fvg_limit", "hybrid_reclaim_market_fvg_limit"
    }
    assert len(overlay) == 4
    pure = overlay.loc[overlay["execution_variant"].isin(["post_reclaim_fvg_market", "post_reclaim_fvg_limit"])]
    assert pure["outcome"].eq("no_fvg_within_wait").all()
    assert pure["entry_fill_flag"].eq(0).all()
    hybrid = overlay.loc[overlay["execution_variant"].eq("hybrid_reclaim_market_fvg_limit")].iloc[0]
    assert int(hybrid["entry_fill_flag"]) == 1
    assert int(hybrid["limit_filled_flag"]) == 0
