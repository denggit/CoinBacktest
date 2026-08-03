from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.structured_pullback_entry import (
    StructuredPullbackConfig,
    attach_limit_fills,
    attach_trade_outcomes,
    build_pullback_candidate_universe,
    causal_audit,
)


def _bars(periods: int = 80) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="1min")
    close = np.full(periods, 106.0)
    open_ = np.r_[106.0, close[:-1]]
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.2,
            "low": np.minimum(open_, close) - 0.2,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )


def _family_rows(bars: pd.DataFrame, *, next_signal_pos: int = 50) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_family_id": "R10_P1_00000001",
                "level_id": 1,
                "family_id": "P1",
                "family_name": "P1",
                "source_timeframe": "15m",
                "source_timeframe_min": 15,
                "structure_available_time": bars.index[10],
                "next_same_timeframe_structure_available_time": bars.index[next_signal_pos],
                "period": "EARLY_2023_2024",
                "entry_limit_price": 100.0,
                "structural_target_h0_price": 110.0,
                "structural_anchor_price": 95.0,
                "stop_price": 94.95,
                "risk_distance_return": 0.0505,
                "h0_reward_return": 0.10,
                "h0_reward_risk_ratio": 0.10 / 0.0505,
                "family_geometry_valid": True,
                "anchor_rule": "PREVIOUS_SWING_LOW",
            }
        ]
    )


def _level_features(bars: pd.DataFrame) -> pd.DataFrame:
    flags = {
        "hyp_h1_first_higher_low_after_decline": True,
        "hyp_h2_bos_pullback_higher_low": True,
        "hyp_h3_layered_base_higher_low": True,
        "hyp_h4_strong_displacement_origin": True,
        "hyp_h5_base_breakout_pullback": True,
        "hyp_h6_multitimeframe_confluence": False,
        "hyp_h7_trend_continuation_higher_low": False,
        "hyp_h8_failed_breakdown_then_higher_low": True,
    }
    common = {
        "pivot_time": bars.index[5],
        "previous2_swing_low_price": 94.0,
        "predecessor_decline_atr": 3.0,
        "rebound_before_current_atr": 2.0,
        "higher_low_gap_atr": 1.0,
        "pullback_fraction_of_rebound": 0.5,
        "confirmation_reaction_high_bp": 100.0,
        "left_high_range_20_bp": 300.0,
        "prior_two_low_gap_atr": 0.1,
        "consecutive_higher_low_count": 1,
        "bos_before_current_low": True,
        "higher_high_before_current_low": True,
        "failed_breakdown_previous_low": True,
        "is_higher_low": True,
        **flags,
    }
    return pd.DataFrame(
        [
            {
                **common,
                "level_id": 1,
                "source_timeframe": "15m",
                "source_timeframe_min": 15,
                "level_price": 100.0,
                "structure_available_time": bars.index[10],
                "previous_swing_low_price": 95.0,
                "current_leg_high_price": 110.0,
            },
            {
                **{**common, "is_higher_low": False, **{name: False for name in flags}},
                "level_id": 2,
                "source_timeframe": "15m",
                "source_timeframe_min": 15,
                "level_price": 102.0,
                "structure_available_time": bars.index[50],
                "previous_swing_low_price": 100.0,
                "current_leg_high_price": 112.0,
            },
            {
                **{**common, "is_higher_low": False, **{name: False for name in flags}},
                "level_id": 3,
                "source_timeframe": "30m",
                "source_timeframe_min": 30,
                "level_price": 100.05,
                "structure_available_time": bars.index[3],
                "previous_swing_low_price": 94.0,
                "current_leg_high_price": 111.0,
            },
        ]
    )


def _lifecycle() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"level_id": 1, "level_price": 100.0, "source_timeframe": "15m", "active_pos": 10, "sweep_pos": 20},
            {"level_id": 2, "level_price": 102.0, "source_timeframe": "15m", "active_pos": 50, "sweep_pos": -1},
            {"level_id": 3, "level_price": 100.05, "source_timeframe": "30m", "active_pos": 3, "sweep_pos": -1},
        ]
    )


def test_family_anchors_and_formation_confluence_are_causal() -> None:
    bars = _bars()
    cfg = StructuredPullbackConfig(timeframes=(("15m", 15), ("30m", 30))).validate()
    unique, family = build_pullback_candidate_universe(
        _level_features(bars),
        _lifecycle(),
        bars,
        cfg,
        research_start=bars.index[0],
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
    )
    assert len(unique) == 1
    assert {"P1", "P2", "P3", "P4", "P5", "P6", "P8"}.issubset(set(family["family_id"]))
    p3 = family.loc[family["family_id"].eq("P3")].iloc[0]
    assert p3["structural_anchor_price"] == 94.0
    p1 = family.loc[family["family_id"].eq("P1")].iloc[0]
    assert p1["structural_anchor_price"] == 95.0
    assert p1["next_same_timeframe_structure_available_time"] == bars.index[50]


def test_limit_activates_after_confirmation_and_cancels_at_next_structure() -> None:
    bars = _bars()
    bars.loc[bars.index[20], "low"] = 99.5
    replay = attach_limit_fills(
        _family_rows(bars, next_signal_pos=50),
        bars,
        StructuredPullbackConfig(),
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
        show_progress=False,
    )
    row = replay.iloc[0]
    assert row["fill_status"] == "FILLED"
    assert int(row["fill_pos"]) == 20
    assert row["order_active_time"] == bars.index[10]

    late = _bars()
    late.loc[late.index[60], "low"] = 99.5
    cancelled = attach_limit_fills(
        _family_rows(late, next_signal_pos=50),
        late,
        StructuredPullbackConfig(),
        research_end_exclusive=late.index[-1] + pd.Timedelta(minutes=1),
        show_progress=False,
    )
    assert cancelled.iloc[0]["fill_status"] == "UNFILLED_NEXT_STRUCTURE_OR_END"


def test_same_bar_limit_and_stop_is_conservative_stop() -> None:
    bars = _bars()
    bars.loc[bars.index[20], ["open", "high", "low", "close"]] = [103.0, 111.0, 94.0, 105.0]
    cfg = StructuredPullbackConfig()
    replay = attach_limit_fills(
        _family_rows(bars),
        bars,
        cfg,
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
        show_progress=False,
    )
    replay = attach_trade_outcomes(
        replay,
        bars,
        cfg,
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
        show_progress=False,
    )
    assert replay.iloc[0]["h0_outcome"] == "SL"
    assert replay.iloc[0]["r1_outcome"] == "SL"


def test_intrabar_fill_cannot_claim_same_bar_high_target() -> None:
    bars = _bars()
    bars.loc[bars.index[20], ["open", "high", "low", "close"]] = [103.0, 111.0, 99.5, 100.5]
    bars.loc[bars.index[21], ["open", "high", "low", "close"]] = [100.5, 101.0, 100.0, 100.5]
    cfg = StructuredPullbackConfig(minimum_holding_minutes=2, maximum_holding_minutes=2)
    replay = attach_limit_fills(
        _family_rows(bars),
        bars,
        cfg,
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
        show_progress=False,
    )
    replay = attach_trade_outcomes(
        replay,
        bars,
        cfg,
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
        show_progress=False,
    )
    assert replay.iloc[0]["h0_outcome"] == "TIME"


def test_cost_is_converted_into_fixed_risk_r_units() -> None:
    bars = _bars()
    bars.loc[bars.index[20], ["open", "high", "low", "close"]] = [100.0, 111.0, 99.8, 110.0]
    cfg = StructuredPullbackConfig()
    replay = attach_limit_fills(
        _family_rows(bars),
        bars,
        cfg,
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
        show_progress=False,
    )
    replay = attach_trade_outcomes(
        replay,
        bars,
        cfg,
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
        show_progress=False,
    )
    row = replay.iloc[0]
    assert row["h0_outcome"] == "TP"
    assert row["h0_net_r_realistic"] < row["h0_gross_r"]
    assert row["h0_net_r_2x_cost"] < row["h0_net_r_realistic"]


def test_causal_audit_rejects_no_valid_r10_fixture() -> None:
    bars = _bars()
    cfg = StructuredPullbackConfig(timeframes=(("15m", 15), ("30m", 30))).validate()
    unique, family = build_pullback_candidate_universe(
        _level_features(bars),
        _lifecycle(),
        bars,
        cfg,
        research_start=bars.index[0],
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
    )
    bars.loc[bars.index[20], "low"] = 99.5
    replay = attach_limit_fills(
        family,
        bars,
        cfg,
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
        show_progress=False,
    )
    replay = attach_trade_outcomes(
        replay,
        bars,
        cfg,
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
        show_progress=False,
    )
    audit = causal_audit(unique, family, replay)
    assert audit["status"].eq("PASS").all()



def test_microsecond_datetime_storage_does_not_shift_activation_or_confluence() -> None:
    bars = _bars()
    bars.index = bars.index.as_unit("us")
    features = _level_features(bars)
    features["structure_available_time"] = features["structure_available_time"].astype("datetime64[us]")
    features["pivot_time"] = features["pivot_time"].astype("datetime64[us]")
    cfg = StructuredPullbackConfig(timeframes=(("15m", 15), ("30m", 30))).validate()
    unique, family = build_pullback_candidate_universe(
        features,
        _lifecycle(),
        bars,
        cfg,
        research_start=bars.index[0],
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
    )
    assert len(unique) == 1
    p1 = family.loc[family["family_id"].eq("P1")].head(1).copy()
    p1["structure_available_time"] = p1["structure_available_time"].astype("datetime64[us]")
    p1["next_same_timeframe_structure_available_time"] = p1[
        "next_same_timeframe_structure_available_time"
    ].astype("datetime64[us]")
    bars.loc[bars.index[20], "low"] = 99.5
    replay = attach_limit_fills(
        p1,
        bars,
        cfg,
        research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
        show_progress=False,
    )
    assert int(replay.iloc[0]["order_active_pos"]) == 10
    assert int(replay.iloc[0]["fill_pos"]) == 20


def test_trade_outcome_mask_is_writable_with_pandas_copy_on_write() -> None:
    bars = _bars()
    bars.loc[bars.index[20], ["open", "high", "low", "close"]] = [100.0, 111.0, 99.8, 110.0]
    cfg = StructuredPullbackConfig()
    with pd.option_context("mode.copy_on_write", True):
        replay = attach_limit_fills(
            _family_rows(bars),
            bars,
            cfg,
            research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
            show_progress=False,
        )
        replay = attach_trade_outcomes(
            replay,
            bars,
            cfg,
            research_end_exclusive=bars.index[-1] + pd.Timedelta(minutes=1),
            show_progress=False,
        )
    assert bool(replay.iloc[0]["valid_filled_trade"])
    assert replay.iloc[0]["h0_outcome"] == "TP"
