from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2 import (
    MSS2Config,
    R02Config,
    attach_structural_exit_outcomes,
    build_stack_execution_triggers,
    build_sweep_episodes,
    build_sweep_stages,
    r02_causal_audit,
    split_r02_features_and_labels,
)


def _bars(rows: list[tuple[float, float, float, float]], start: str = "2025-01-01 00:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(rows), freq="1min")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx, dtype=float)


def _level(
    level_id: int,
    price: float,
    *,
    direction: int,
    sweep_pos: int,
    tf_min: int = 15,
    order: int = 2,
    active_pos: int = 0,
) -> dict[str, object]:
    side = "low" if direction > 0 else "high"
    tf_name = "15m" if tf_min == 15 else "30m" if tf_min == 30 else "1H" if tf_min == 60 else "4H" if tf_min == 240 else "1D"
    sweep_time = pd.Timestamp("2025-01-01") + pd.Timedelta(minutes=sweep_pos) if sweep_pos >= 0 else pd.NaT
    return {
        "level_id": level_id,
        "pivot_side": side,
        "source_timeframe": tf_name,
        "source_timeframe_min": tf_min,
        "pivot_time": pd.Timestamp("2024-12-30"),
        "pivot_bar_end_time": pd.Timestamp("2024-12-30 00:15"),
        "level_price": price,
        "initial_available_time": pd.Timestamp("2024-12-30 00:30"),
        "order_1_available_time": pd.Timestamp("2024-12-30 00:30"),
        "order_2_available_time": pd.Timestamp("2024-12-30 00:45") if order >= 2 else pd.NaT,
        "order_3_available_time": pd.Timestamp("2024-12-30 01:00") if order >= 3 else pd.NaT,
        "order_5_available_time": pd.Timestamp("2024-12-30 01:30") if order >= 5 else pd.NaT,
        "liquidity_side": "sell_side" if direction > 0 else "buy_side",
        "trade_direction": direction,
        "active_pos_1m": active_pos,
        "sweep_pos_1m": sweep_pos,
        "sweep_bar_time_1m": sweep_time,
        "sweep_available_time_1m": sweep_time + pd.Timedelta(minutes=1) if sweep_pos >= 0 else pd.NaT,
        "age_minutes_since_pivot_at_sweep": 2_000.0 if sweep_pos >= 0 else np.nan,
        "confirmed_order_at_sweep": order if sweep_pos >= 0 else 0,
        "external_20_flag": 1 if tf_min >= 60 else 0,
        "external_50_flag": 1 if tf_min >= 240 else 0,
        "clean_sweep_no_prior_touch_flag": 1,
        "pretested_before_sweep_flag": 0,
        "old_remote_flag_24h": 1,
        "old_remote_flag_72h": 0,
    }


def test_sweep_stage_merges_level_events_and_counts_price_pools() -> None:
    bars = _bars([(100, 101, 99, 100)] * 10)
    bars.iloc[4, bars.columns.get_loc("low")] = 98.0
    lifecycle = pd.DataFrame(
        [
            _level(1, 100.00, direction=1, sweep_pos=4, tf_min=15),
            _level(2, 100.05, direction=1, sweep_pos=4, tf_min=60),  # 5bp from first
            _level(3, 99.00, direction=1, sweep_pos=4, tf_min=240),  # separate pool
        ]
    )
    stages = build_sweep_stages(bars, lifecycle, config=R02Config())
    assert len(stages) == 1
    row = stages.iloc[0]
    assert int(row["levels_consumed_stage"]) == 3
    assert int(row["distinct_timeframes_stage"]) == 3
    assert int(row["price_pools_10p0bp_stage"]) == 2
    assert int(row["htf_240m_plus_levels_stage"]) == 1


def test_episode_accumulation_is_past_only_and_requires_extension() -> None:
    bars = _bars([(100, 101, 99, 100)] * 30)
    bars.iloc[4, bars.columns.get_loc("low")] = 98.5
    bars.iloc[8, bars.columns.get_loc("low")] = 97.5
    bars.iloc[12, bars.columns.get_loc("low")] = 98.0  # non-extension -> new episode
    lifecycle = pd.DataFrame(
        [
            _level(1, 99.0, direction=1, sweep_pos=4),
            _level(2, 98.0, direction=1, sweep_pos=8, tf_min=60),
            _level(3, 98.5, direction=1, sweep_pos=12, tf_min=240),
        ]
    )
    stages = build_sweep_stages(bars, lifecycle, config=R02Config())
    episodes = build_sweep_episodes(stages, config=R02Config(episode_gap_minutes=15))
    assert episodes.iloc[0]["episode_id"] == episodes.iloc[1]["episode_id"]
    assert episodes.iloc[2]["episode_id"] != episodes.iloc[1]["episode_id"]
    assert int(episodes.iloc[1]["levels_consumed_cum"]) == 2
    assert int(episodes.iloc[0]["levels_consumed_cum"]) == 1
    assert r02_causal_audit(episodes, pd.DataFrame())["episode_start_after_stage"] == 0


def test_reclaim_can_confirm_on_sweep_bar_but_entry_is_after_bar_close() -> None:
    rows = [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 94, 96),  # sweeps 95 and closes back above 95
        (96, 98, 95, 97),
        (97, 99, 96, 98),
        (98, 100, 97, 99),
        (99, 101, 98, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
    ]
    bars = _bars(rows)
    lifecycle = pd.DataFrame([_level(1, 95.0, direction=1, sweep_pos=4)])
    stages = build_sweep_episodes(build_sweep_stages(bars, lifecycle), config=R02Config())
    trades = build_stack_execution_triggers(
        bars,
        stages,
        execution_minutes=1,
        base_config=MSS2Config(execution_confirmation_orders=(1, 2, 3)),
        config=R02Config(max_confirmation_minutes=5),
        reference_modes=("structural",),
        include_reclaims=True,
        include_mss_market=False,
        include_mss_fvg=False,
    )
    assert not trades.empty
    stage_reclaim = trades.loc[trades["trigger_type"].eq("stage_reclaim")].iloc[0]
    assert stage_reclaim["signal_available_time"] == pd.Timestamp("2025-01-01 00:05")
    assert stage_reclaim["entry_time"] == pd.Timestamp("2025-01-01 00:05")
    audit = r02_causal_audit(stages, trades)
    assert audit["entry_before_signal_available"] == 0
    assert audit["signal_before_sweep_exec_available"] == 0


def test_structural_exit_censors_instead_of_forcing_time_close() -> None:
    bars = _bars([(100, 101, 99, 100)] * 20)
    lifecycle = pd.DataFrame(
        [
            _level(1, 110.0, direction=-1, sweep_pos=-1, tf_min=60),
            _level(2, 90.0, direction=1, sweep_pos=-1, tf_min=60),
        ]
    )
    trade = pd.DataFrame(
        [
            {
                "trade_event_id": "T1",
                "stage_id": "S1",
                "episode_id": "E1",
                "trade_direction": 1,
                "execution_minutes": 1,
                "trigger_type": "stage_reclaim",
                "reference_mode": "none",
                "entry_kind": "market_next_open",
                "entry_pos_1m": 2,
                "entry_time": bars.index[2],
                "entry_price": 100.0,
                "stop_price": 90.0,
                "signal_available_time": bars.index[2],
                "sweep_exec_available_time": bars.index[1],
            }
        ]
    )
    out = attach_structural_exit_outcomes(
        bars,
        lifecycle,
        trade,
        config=R02Config(exit_censor_minutes=5, path_horizons_minutes=(5,)),
    )
    assert out.iloc[0]["target_any_outcome"] == "censored"
    assert pd.isna(out.iloc[0].get("target_any_gross_return", np.nan))
    assert np.isfinite(out.iloc[0]["mark_return_5m"])


def test_opposing_pool_target_requires_active_levels_and_can_be_multitf() -> None:
    rows = [(100, 101, 99, 100)] * 30
    # Later reach target zone around 105.
    rows[10] = (103, 106, 102, 105)
    bars = _bars(rows)
    lifecycle = pd.DataFrame(
        [
            _level(1, 105.00, direction=-1, sweep_pos=10, tf_min=15, active_pos=0),
            _level(2, 105.05, direction=-1, sweep_pos=10, tf_min=60, active_pos=0),
            _level(3, 110.00, direction=-1, sweep_pos=-1, tf_min=240, active_pos=0),
            _level(4, 95.00, direction=1, sweep_pos=-1, tf_min=60, active_pos=0),
        ]
    )
    trade = pd.DataFrame(
        [
            {
                "trade_event_id": "T1",
                "stage_id": "S1",
                "episode_id": "E1",
                "trade_direction": 1,
                "execution_minutes": 1,
                "trigger_type": "stage_reclaim",
                "reference_mode": "none",
                "entry_kind": "market_next_open",
                "entry_pos_1m": 2,
                "entry_time": bars.index[2],
                "entry_price": 100.0,
                "stop_price": 95.0,
                "signal_available_time": bars.index[2],
                "sweep_exec_available_time": bars.index[1],
            }
        ]
    )
    out = attach_structural_exit_outcomes(
        bars,
        lifecycle,
        trade,
        config=R02Config(exit_censor_minutes=20, path_horizons_minutes=(20,), target_pool_tolerance_bps=10.0),
    )
    row = out.iloc[0]
    assert abs(float(row["target_pool2_price"]) - 105.0) < 1e-9
    assert abs(float(row["target_pool2tf_price"]) - 105.0) < 1e-9
    assert int(row["target_pool2tf_pool_timeframes"]) >= 2
    assert row["target_pool2tf_outcome"] == "target"


def test_r02_feature_split_removes_all_forward_target_and_path_labels() -> None:
    frame = pd.DataFrame(
        [
            {
                "trade_event_id": "T1",
                "stage_id": "S1",
                "episode_id": "E1",
                "levels_consumed_cum": 4,
                "target_any_outcome": "target",
                "target_any_gross_return": 0.01,
                "mark_return_1440m": 0.02,
                "mfe_1440m": 0.03,
                "mae_1440m": 0.01,
            }
        ]
    )
    features, labels = split_r02_features_and_labels(frame)
    assert "levels_consumed_cum" in features.columns
    assert "target_any_outcome" not in features.columns
    assert "mark_return_1440m" not in features.columns
    assert "target_any_outcome" in labels.columns


def test_target_level_remains_active_at_start_of_its_future_sweep_bar() -> None:
    rows = [(100.0, 101.0, 99.0, 100.0)] * 12
    # Buy-side target at 105 is first swept during bar 5. At the *start* of bar
    # 5 this future sweep is unknown, so a long entered at that open must still
    # see 105 as active opposing liquidity.
    rows[5] = (100.0, 106.0, 99.5, 105.0)
    bars = _bars(rows)
    lifecycle = pd.DataFrame(
        [
            _level(1, 105.0, direction=-1, sweep_pos=5, tf_min=60, active_pos=0),
            _level(2, 95.0, direction=1, sweep_pos=-1, tf_min=60, active_pos=0),
        ]
    )
    trade = pd.DataFrame(
        [
            {
                "trade_event_id": "T1",
                "stage_id": "S1",
                "episode_id": "E1",
                "trade_direction": 1,
                "execution_minutes": 1,
                "trigger_type": "stage_reclaim",
                "reference_mode": "none",
                "entry_kind": "market_next_open",
                "entry_pos_1m": 5,
                "entry_time": bars.index[5],
                "entry_price": 100.0,
                "stop_price": 95.0,
                "signal_available_time": bars.index[5],
                "sweep_exec_available_time": bars.index[4],
            }
        ]
    )
    out = attach_structural_exit_outcomes(
        bars,
        lifecycle,
        trade,
        config=R02Config(exit_censor_minutes=5, path_horizons_minutes=(5,)),
    )
    row = out.iloc[0]
    assert float(row["target_any_price"]) == 105.0
    assert row["target_any_outcome"] == "target"


def test_trade_event_id_is_globally_unique_across_execution_timeframes() -> None:
    rows = []
    for i in range(120):
        close = 100.0 + 2.0 * np.sin(i / 3.0)
        open_ = 100.0 + 2.0 * np.sin((i - 1) / 3.0)
        rows.append((open_, max(open_, close) + 0.8, min(open_, close) - 0.8, close))
    rows[20] = (98.0, 99.0, 94.0, 96.0)
    bars = _bars(rows)
    lifecycle = pd.DataFrame([_level(1, 95.0, direction=1, sweep_pos=20)])
    stages = build_sweep_episodes(build_sweep_stages(bars, lifecycle), config=R02Config())
    frames = []
    for minutes in (1, 2, 5):
        part = build_stack_execution_triggers(
            bars,
            stages,
            execution_minutes=minutes,
            base_config=MSS2Config(execution_confirmation_orders=(1, 2, 3)),
            config=R02Config(max_confirmation_minutes=5),
            reference_modes=("structural",),
            include_reclaims=True,
            include_mss_market=False,
            include_mss_fvg=False,
        )
        if not part.empty:
            frames.append(part[["trade_event_id", "execution_minutes"]])
    merged = pd.concat(frames, ignore_index=True)
    assert not merged.empty
    assert not merged["trade_event_id"].duplicated().any()
    for row in merged.itertuples(index=False):
        assert str(row.trade_event_id).startswith(f"R02_{int(row.execution_minutes)}M_TRADE_")


def test_structural_stop_rejects_out_of_range_or_all_nan_slice() -> None:
    import src.research_common.ict_mss2.r02 as r02mod

    low = np.array([99.0, 98.0, 97.0], dtype=float)
    high = np.array([101.0, 102.0, 103.0], dtype=float)
    extreme, stop = r02mod._structural_stop_before_entry(
        low, high, direction=1, start_pos=10, end_pos=12, buffer_bps=2.0
    )
    assert np.isnan(extreme) and np.isnan(stop)

    extreme, stop = r02mod._structural_stop_before_entry(
        np.array([np.nan, np.nan]), np.array([np.nan, np.nan]),
        direction=-1, start_pos=0, end_pos=1, buffer_bps=2.0,
    )
    assert np.isnan(extreme) and np.isnan(stop)
