from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.research_common.post_sweep_process.config import PostSweepConfig
from src.research_common.post_sweep_process.process import (
    build_post_sweep_checkpoint_table,
    split_checkpoint_features_labels,
)
from src.research_common.post_sweep_process.reports import (
    causal_audit,
    oracle_turning_point_table,
)


def _bars(periods: int = 500) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="1min")
    close = np.full(periods, 100.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.1
    low = np.minimum(open_, close) - 0.1
    delta = np.zeros(periods)
    notional = np.full(periods, 1_000_000.0)
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "notional": notional,
            "buy_notional": (notional + delta) / 2.0,
            "sell_notional": (notional - delta) / 2.0,
            "delta_notional": delta,
            "trades_count": 100.0,
            "large_buy_notional": np.maximum(delta, 0.0),
            "large_sell_notional": np.maximum(-delta, 0.0),
            "large_delta_notional": delta,
        },
        index=index,
    )


def _event(bars: pd.DataFrame, pos: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_event_id": ["Z1"],
            "event_kind": ["swing_zone_sweep"],
            "event_pos": [pos],
            "event_bar_time": [bars.index[pos]],
            "event_available_time": [bars.index[pos] + pd.Timedelta(minutes=1)],
            "zone_floor_price": [99.5],
            "zone_ceiling_price": [100.0],
            "zone_center_price": [99.75],
            "sweep_low": [99.0],
            "pre_atr_240m_abs": [1.0],
        }
    )


def test_checkpoint_features_are_closed_bar_causal() -> None:
    bars = _bars()
    out = build_post_sweep_checkpoint_table(
        _event(bars), bars,
        PostSweepConfig(observation_horizon_bars=15, dense_checkpoint_bars=5, fixed_checkpoint_bars=(10, 15), future_horizons=(5, 15)),
        show_progress=False,
    )
    first = out.iloc[0]
    assert first["checkpoint_pos"] == 101
    assert first["checkpoint_available_time"] == bars.index[102]
    assert first["entry_reference_time"] == bars.index[102]


def test_new_low_attempts_are_retained_outside_fixed_schedule() -> None:
    bars = _bars()
    bars.iloc[107, bars.columns.get_loc("low")] = 98.0
    cfg = PostSweepConfig(observation_horizon_bars=15, dense_checkpoint_bars=3, fixed_checkpoint_bars=(10, 15), future_horizons=(5, 15))
    out = build_post_sweep_checkpoint_table(_event(bars), bars, cfg, show_progress=False)
    row = out.loc[out["checkpoint_pos"].eq(107)].iloc[0]
    assert bool(row["new_low_attempt_flag"])
    assert row["new_low_attempt_index"] == 1


def test_cvd_price_divergence_is_causal() -> None:
    bars = _bars()
    pos = 100
    bars.iloc[pos + 1 : pos + 4, bars.columns.get_loc("delta_notional")] = [-100_000, -100_000, -100_000]
    bars.iloc[pos + 1 : pos + 4, bars.columns.get_loc("buy_notional")] = [450_000, 450_000, 450_000]
    bars.iloc[pos + 1 : pos + 4, bars.columns.get_loc("sell_notional")] = [550_000, 550_000, 550_000]
    # Price never takes the original sweep low despite CVD making new lows.
    cfg = PostSweepConfig(observation_horizon_bars=10, dense_checkpoint_bars=5, fixed_checkpoint_bars=(10,), future_horizons=(5,))
    out = build_post_sweep_checkpoint_table(_event(bars, pos), bars, cfg, show_progress=False)
    assert out.loc[out["elapsed_bars"].eq(2), "cvd_new_low_without_price_new_low"].iloc[0]


def test_price_impact_efficiency_uses_sell_flow_not_only_bp() -> None:
    bars = _bars()
    pos = 100
    bars.iloc[pos + 1, bars.columns.get_loc("close")] = 99.9
    bars.iloc[pos + 1, bars.columns.get_loc("sell_notional")] = 2_000_000.0
    bars.iloc[pos + 1, bars.columns.get_loc("buy_notional")] = 0.0
    bars.iloc[pos + 1, bars.columns.get_loc("notional")] = 2_000_000.0
    bars.iloc[pos + 1, bars.columns.get_loc("delta_notional")] = -2_000_000.0
    cfg = PostSweepConfig(observation_horizon_bars=10, dense_checkpoint_bars=5, fixed_checkpoint_bars=(10,), future_horizons=(5,))
    out = build_post_sweep_checkpoint_table(_event(bars, pos), bars, cfg, show_progress=False).iloc[0]
    assert np.isfinite(out["downside_bp_per_sell_million_1m"])
    assert out["downside_bp_per_sell_million_1m"] > 0


def test_future_mfe_mae_begin_at_next_open() -> None:
    bars = _bars(200)
    pos = 50
    checkpoint_pos = pos + 1
    entry_pos = checkpoint_pos + 1
    bars.iloc[entry_pos, bars.columns.get_loc("open")] = 100.0
    bars.iloc[entry_pos : entry_pos + 5, bars.columns.get_loc("high")] = [101, 103, 102, 104, 101]
    bars.iloc[entry_pos : entry_pos + 5, bars.columns.get_loc("low")] = [99, 98, 99, 99.5, 100]
    bars.iloc[entry_pos + 4, bars.columns.get_loc("close")] = 102.0
    cfg = PostSweepConfig(observation_horizon_bars=10, dense_checkpoint_bars=2, fixed_checkpoint_bars=(10,), future_horizons=(5,))
    out = build_post_sweep_checkpoint_table(_event(bars, pos), bars, cfg, show_progress=False)
    first = out.loc[out["elapsed_bars"].eq(1)].iloc[0]
    assert abs(first["future_mfe_5m"] - 0.04) < 1e-12
    assert abs(first["future_mae_5m"] + 0.02) < 1e-12
    assert abs(first["future_close_return_5m"] - 0.02) < 1e-12


def test_features_labels_are_physically_separated() -> None:
    bars = _bars()
    cfg = PostSweepConfig(observation_horizon_bars=10, dense_checkpoint_bars=5, fixed_checkpoint_bars=(10,), future_horizons=(5,))
    checkpoints = build_post_sweep_checkpoint_table(_event(bars), bars, cfg, show_progress=False)
    features, labels = split_checkpoint_features_labels(checkpoints)
    assert not any(name.startswith("future_") for name in features.columns)
    assert "future_mfe_5m" in labels.columns
    audit = causal_audit(features, labels)
    assert int(audit["violations"].sum()) == 0


def test_oracle_turning_point_is_explicitly_future_labelled() -> None:
    bars = _bars(300)
    pos = 50
    # Make future path rise without lower lows after the first checkpoint.
    rising = np.linspace(100.0, 102.0, 80)
    bars.iloc[pos + 2 : pos + 82, bars.columns.get_loc("open")] = rising
    bars.iloc[pos + 2 : pos + 82, bars.columns.get_loc("high")] = rising + 0.5
    bars.iloc[pos + 2 : pos + 82, bars.columns.get_loc("low")] = rising - 0.1
    bars.iloc[pos + 2 : pos + 82, bars.columns.get_loc("close")] = rising
    cfg = PostSweepConfig(observation_horizon_bars=60, dense_checkpoint_bars=10, fixed_checkpoint_bars=(30, 60), future_horizons=(5, 15, 30, 60), reversal_mfe_return=0.005)
    checkpoints = build_post_sweep_checkpoint_table(_event(bars, pos), bars, cfg, show_progress=False)
    oracle = oracle_turning_point_table(checkpoints, cfg)
    assert len(oracle) == 1
    assert bool(oracle.iloc[0]["oracle_selection_uses_future"])


def test_microsecond_index_preserves_timing() -> None:
    bars = _bars()
    bars.index = pd.DatetimeIndex(bars.index.to_numpy(dtype="datetime64[us]"))
    cfg = PostSweepConfig(observation_horizon_bars=10, dense_checkpoint_bars=5, fixed_checkpoint_bars=(10,), future_horizons=(5,))
    out = build_post_sweep_checkpoint_table(_event(bars), bars, cfg, show_progress=False)
    assert out.iloc[0]["checkpoint_available_time"] == pd.Timestamp(bars.index[102])


def test_main_script_self_test_imports() -> None:
    script = Path(__file__).resolve().parents[2] / "research" / "liquidity" / "04_post_sweep_continuation_exhaustion_atlas.py"
    spec = importlib.util.spec_from_file_location("r04_script", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run_self_test()
