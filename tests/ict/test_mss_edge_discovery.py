from __future__ import annotations

import numpy as np
import pandas as pd

from research.ict.mss.common.evaluation import build_edge_gate, cost_stress_table
from research.ict.mss.common.execution import attach_limit_entry_and_outcomes
from research.ict.mss.common.models import MSSResearchSpec
from research.ict.mss.common.structure import (
    build_displacement_fvgs,
    build_htf_liquidity_levels,
    build_micro_structure_context,
    pair_sweeps_with_mss_fvgs,
)


def _bars(n: int, start: str = "2023-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=n, freq="1min")
    return pd.DataFrame(
        {
            "open": np.full(n, 100.0),
            "high": np.full(n, 101.0),
            "low": np.full(n, 99.0),
            "close": np.full(n, 100.0),
        },
        index=index,
    )


def test_htf_swing_is_not_available_until_right_confirmation_bar_closes() -> None:
    # Seven complete 15m candles.  Bin 2 (00:30) is a clear swing low.  With
    # order=1 its right neighbor is 00:45-00:59, so the level is first usable at
    # 01:00, never during the pivot/right-confirmation candle.
    bars = _bars(7 * 15)
    bin_lows = [98.0, 97.0, 94.0, 97.0, 98.0, 99.0, 100.0]
    for i, low in enumerate(bin_lows):
        sl = slice(i * 15, (i + 1) * 15)
        bars.iloc[sl, bars.columns.get_loc("low")] = low
        bars.iloc[sl, bars.columns.get_loc("open")] = low + 1.0
        bars.iloc[sl, bars.columns.get_loc("close")] = low + 1.0
        bars.iloc[sl, bars.columns.get_loc("high")] = low + 2.0

    levels = build_htf_liquidity_levels(
        bars,
        timeframes=(("15m", 15),),
        confirmation_orders=(1, 2),
    )
    target = levels.loc[
        (levels["pivot_kind"] == "low")
        & (pd.to_datetime(levels["pivot_time"]) == pd.Timestamp("2023-01-01 00:30:00"))
    ].iloc[0]
    assert pd.Timestamp(target["pivot_bar_end_time"]) == pd.Timestamp("2023-01-01 00:45:00")
    assert pd.Timestamp(target["order_1_available_time"]) == pd.Timestamp("2023-01-01 01:00:00")
    assert pd.Timestamp(target["initial_available_time"]) > pd.Timestamp(target["pivot_bar_end_time"])


def test_micro_swing_is_forward_filled_only_after_confirmation_closes() -> None:
    bars = _bars(30)
    bars.iloc[10, bars.columns.get_loc("high")] = 105.0
    ctx = build_micro_structure_context(bars, orders=(2,))[2]

    # order=2 pivot at pos10 needs bars 11 and 12 to close; it becomes usable at
    # bar-start pos13.
    assert ctx.last_high_pivot_pos[12] == -1
    assert ctx.last_high_pivot_pos[13] == 10
    assert np.isnan(ctx.last_high_level[12])
    assert ctx.last_high_level[13] == 105.0


def _bullish_mss_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    bars = _bars(60)
    bars.iloc[20, bars.columns.get_loc("high")] = 105.0

    # Sell-side liquidity sweep and the eventual structural stop extreme.
    bars.iloc[30, bars.columns.get_loc("low")] = 94.0
    bars.iloc[32, bars.columns.get_loc("low")] = 93.0

    # 34 / 35 / 36 create a bullish FVG.  Candle 35 is the displacement candle
    # and closes through the already-confirmed 105 micro swing high.
    bars.iloc[34] = [100.0, 101.0, 99.0, 100.0]
    bars.iloc[35] = [100.0, 106.5, 99.5, 106.0]
    bars.iloc[36] = [103.0, 107.0, 102.0, 105.0]

    # FVG near edge is 102.  It must NOT fill on completion candle 36.  Bar 37
    # stays above it and bar 38 provides the first valid retest.
    bars.iloc[37] = [104.0, 105.0, 103.0, 104.0]
    bars.iloc[38] = [103.0, 104.0, 101.5, 103.0]

    sweep = pd.DataFrame(
        [
            {
                "sweep_id": "SSL_30",
                "side": 1,
                "liquidity_side": "sell_side",
                "sweep_pos": 30,
                "sweep_bar_time": bars.index[30],
                "sweep_available_time": bars.index[30] + pd.Timedelta(minutes=1),
                "swept_level_count": 1,
                "swept_timeframe_count": 1,
                "max_timeframe_min": 60,
                "max_confirmed_order": 3,
                "swept_timeframes": "1H",
            }
        ]
    )
    return bars, sweep


def test_bullish_sweep_mss_fvg_uses_causal_structure_and_sweep_extreme_stop() -> None:
    bars, sweep = _bullish_mss_fixture()
    micro = build_micro_structure_context(bars, orders=(2,))
    fvgs = build_displacement_fvgs(bars, rolling_window=30)
    paired = pair_sweeps_with_mss_fvgs(
        bars,
        sweep,
        fvgs,
        micro,
        max_search_bars=30,
        show_progress=False,
    )
    pre = paired.loc[paired["structure_mode"] == "pre_sweep"].iloc[0]

    assert pre["side"] == 1
    assert int(pre["micro_structure_pivot_pos"]) == 20
    assert pre["micro_structure_level"] == 105.0
    assert int(pre["displacement_pos"]) == 35
    assert int(pre["fvg_completion_pos"]) == 36
    assert pre["fvg_near_price"] == 102.0
    assert pre["stop_extreme"] == 93.0
    assert int(pre["micro_structure_available_pos"]) <= int(pre["displacement_pos"])


def test_fvg_limit_order_cannot_fill_on_fvg_completion_candle() -> None:
    bars, sweep = _bullish_mss_fixture()
    paired = pair_sweeps_with_mss_fvgs(
        bars,
        sweep,
        build_displacement_fvgs(bars, rolling_window=30),
        build_micro_structure_context(bars, orders=(2,)),
        max_search_bars=30,
        show_progress=False,
    )
    pre = paired.loc[paired["structure_mode"] == "pre_sweep"].head(1)
    result = attach_limit_entry_and_outcomes(
        bars,
        pre,
        max_fill_wait_bars=10,
        outcome_horizon_bars=10,
        target_rs=(1.0,),
        show_progress=False,
    ).iloc[0]

    assert int(result["fvg_completion_pos"]) == 36
    assert int(result["order_active_pos"]) == 37
    assert int(result["first_fill_pos"]) == 38
    assert result["first_fill_time"] == bars.index[38]


def test_same_bar_target_and_stop_is_resolved_as_stop() -> None:
    bars = _bars(20)
    # Completion pos5 -> order active at pos6, fills 100 there.  At pos7 the
    # candle trades below 99 stop and above 101 target; bare OHLC cannot know
    # ordering, so the simulator must choose STOP.
    bars.iloc[6] = [100.2, 100.5, 99.5, 100.2]
    bars.iloc[7] = [100.0, 101.5, 98.5, 100.0]
    setup = pd.DataFrame(
        [{"side": 1, "fvg_completion_pos": 5, "fvg_near_price": 100.0, "stop_extreme": 99.0}]
    )
    result = attach_limit_entry_and_outcomes(
        bars,
        setup,
        max_fill_wait_bars=5,
        outcome_horizon_bars=5,
        target_rs=(1.0,),
        show_progress=False,
    ).iloc[0]
    assert int(result["first_fill_pos"]) == 6
    assert result["result_r1p0"] == "STOP"
    assert int(result["exit_pos_r1p0"]) == 7


def test_cost_stress_pre_holdout_scope_excludes_2026() -> None:
    frame = pd.DataFrame(
        {
            "first_fill_time": pd.to_datetime(["2025-06-01", "2026-02-01"]),
            "gross_r_r1p0": [0.5, 10.0],
            "risk_pct": [0.01, 0.01],
        }
    )
    stress = cost_stress_table({("X", 1.0): frame}, round_trip_cost_pct=0.0011, multipliers=(2.0,))
    pre = stress.loc[stress["scope"] == "PRE_HOLDOUT_2023_2025"].iloc[0]
    full = stress.loc[stress["scope"] == "FULL_2023_2026H1"].iloc[0]
    assert int(pre["trades"]) == 1
    assert int(full["trades"]) == 2


def test_edge_gate_does_not_use_overall_top_winners_to_freeze_candidate() -> None:
    overall = pd.DataFrame(
        [
            {
                "spec_id": "X",
                "target_r": 1.0,
                "trades": 200,
                "mean_net_r": 0.20,
                "profit_factor": 1.5,
                "positive_month_share": 0.7,
                "top10_removed_mean_net_r": 0.30,
                "top10_removed_profit_factor": 2.0,
                # Pre-holdout robustness deliberately fails.  A huge 2026
                # winner must not rescue candidate freezing.
                "pre_holdout_trades": 180,
                "pre_holdout_top10_removed_mean_net_r": -0.01,
                "pre_holdout_top10_removed_profit_factor": 0.95,
            }
        ]
    )
    periods = pd.DataFrame(
        [
            {"spec_id": "X", "target_r": 1.0, "period": "2023", "trades": 60, "mean_net_r": 0.1, "profit_factor": 1.2},
            {"spec_id": "X", "target_r": 1.0, "period": "2024", "trades": 60, "mean_net_r": 0.1, "profit_factor": 1.2},
            {"spec_id": "X", "target_r": 1.0, "period": "2025_VALIDATION", "trades": 60, "mean_net_r": 0.1, "profit_factor": 1.2},
            {"spec_id": "X", "target_r": 1.0, "period": "2026H1_SEALED", "trades": 20, "mean_net_r": 1.0, "profit_factor": 5.0},
        ]
    )
    stress = pd.DataFrame(
        [
            {
                "spec_id": "X",
                "target_r": 1.0,
                "scope": "PRE_HOLDOUT_2023_2025",
                "cost_multiplier": 2.0,
                "trades": 180,
                "mean_net_r": 0.05,
                "profit_factor": 1.2,
            }
        ]
    )
    gate = build_edge_gate(overall, periods, stress, [MSSResearchSpec("X", "test")]).iloc[0]
    assert not bool(gate["top10_winner_removal_pass"])
    assert not bool(gate["frozen_before_2026_holdout"])
    assert not bool(gate["edge_found"])


def test_fill_candle_favorable_extreme_cannot_be_used_as_target_or_mfe() -> None:
    bars = _bars(15)
    # Limit 100 first becomes active/fills at pos6.  Its high reaches the 1R
    # target, but that high may have occurred before the intrabar low touched
    # the limit.  No later candle reaches 101, so result must not be TARGET.
    bars.iloc[6] = [100.5, 102.0, 99.5, 100.2]
    for pos in range(7, 12):
        bars.iloc[pos] = [100.0, 100.8, 99.6, 100.0]
    setup = pd.DataFrame(
        [{"side": 1, "fvg_completion_pos": 5, "fvg_near_price": 100.0, "stop_extreme": 99.0}]
    )
    result = attach_limit_entry_and_outcomes(
        bars,
        setup,
        max_fill_wait_bars=3,
        outcome_horizon_bars=5,
        target_rs=(1.0,),
        show_progress=False,
    ).iloc[0]
    assert int(result["first_fill_pos"]) == 6
    assert result["result_r1p0"] == "TIMEOUT"
    assert result["mfe_r_horizon"] < 1.0


def test_bearish_mirror_sweep_mss_fvg_and_limit_entry() -> None:
    bars = _bars(60)
    bars.iloc[20, bars.columns.get_loc("low")] = 95.0
    bars.iloc[30, bars.columns.get_loc("high")] = 106.0
    bars.iloc[32, bars.columns.get_loc("high")] = 107.0
    bars.iloc[34] = [100.0, 101.0, 99.0, 100.0]
    bars.iloc[35] = [100.0, 100.5, 93.5, 94.0]
    bars.iloc[36] = [97.0, 98.0, 93.0, 95.0]
    bars.iloc[37] = [96.0, 97.0, 95.0, 96.0]
    bars.iloc[38] = [97.0, 98.5, 96.0, 97.0]
    sweep = pd.DataFrame(
        [
            {
                "sweep_id": "BSL_30",
                "side": -1,
                "liquidity_side": "buy_side",
                "sweep_pos": 30,
                "sweep_bar_time": bars.index[30],
                "sweep_available_time": bars.index[30] + pd.Timedelta(minutes=1),
                "swept_level_count": 1,
                "swept_timeframe_count": 1,
                "max_timeframe_min": 60,
                "max_confirmed_order": 3,
                "swept_timeframes": "1H",
            }
        ]
    )
    paired = pair_sweeps_with_mss_fvgs(
        bars,
        sweep,
        build_displacement_fvgs(bars, rolling_window=30),
        build_micro_structure_context(bars, orders=(2,)),
        max_search_bars=30,
        show_progress=False,
    )
    pre = paired.loc[paired["structure_mode"] == "pre_sweep"].head(1)
    assert not pre.empty
    row = pre.iloc[0]
    assert int(row["micro_structure_pivot_pos"]) == 20
    assert row["micro_structure_level"] == 95.0
    assert int(row["displacement_pos"]) == 35
    assert row["fvg_near_price"] == 98.0
    assert row["stop_extreme"] == 107.0

    result = attach_limit_entry_and_outcomes(
        bars,
        pre,
        max_fill_wait_bars=10,
        outcome_horizon_bars=10,
        target_rs=(1.0,),
        show_progress=False,
    ).iloc[0]
    assert int(result["order_active_pos"]) == 37
    assert int(result["first_fill_pos"]) == 38
