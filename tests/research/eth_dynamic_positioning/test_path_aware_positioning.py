from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.eth_dynamic_position_path import PathRule, build_path_table, replay_targets


def _decisions(n: int = 30) -> pd.DataFrame:
    t = pd.date_range("2023-01-01 00:00:00", periods=n, freq="4h")
    trend = np.full(n, 0.45)
    ext = np.full(n, 0.65)
    return pd.DataFrame({
        "timestamp": t - pd.Timedelta(hours=1),
        "available_time": t,
        "decision_close": True,
        "state_ready": True,
        "medium_trend": trend,
        "slow_trend": trend,
        "medium_extension": ext,
        "slow_extension": ext,
        "medium_location_multiplier": 1.0 - 0.25 * ext,
        "slow_location_multiplier": 1.0 - 0.25 * ext,
        "medium_desired_close": 0.30,
        "slow_desired_close": 0.30,
    })


def _equity(n_hours: int = 120) -> pd.DataFrame:
    t = pd.date_range("2023-01-01 00:00:00", periods=n_hours, freq="1h")
    return pd.DataFrame({
        "timestamp": t,
        "next_timestamp": t + pd.Timedelta(hours=1),
        "execution_decision": (np.arange(n_hours) % 4) == 0,
        "open": 100.0 * np.cumprod(np.full(n_hours, 1.0001)),
        "price_return": 0.0001,
        "funding_rate": 0.0,
        "equity": 1.0,
        "drawdown": 0.0,
        "gross_exposure": 0.0,
        "turnover": 0.0,
    })


def test_path_requires_past_persistence_not_just_current_strength() -> None:
    p = build_path_table(_decisions(), PathRule())
    assert not bool(p.loc[0, "mature_expansion"])
    assert bool(p["mature_expansion"].iloc[-1])


def test_future_decision_changes_do_not_change_current_path_state() -> None:
    d1 = _decisions()
    d2 = d1.copy()
    d2.loc[20:, ["medium_trend", "slow_trend", "medium_extension", "slow_extension"]] = -0.9
    p1 = build_path_table(d1, PathRule())
    p2 = build_path_table(d2, PathRule())
    cols = ["state", "state_age_hours", "aligned_extension_mean", "strong_share_72h", "mature_expansion"]
    pd.testing.assert_frame_equal(p1.loc[:19, cols], p2.loc[:19, cols])


def test_counterfactual_charges_gross_sleeve_turnover() -> None:
    p = build_path_table(_decisions(), PathRule())
    e = _equity()
    cfg = {
        "fee_rate_per_side": 0.00055,
        "slippage_rate_per_side": 0.00010,
        "no_trade_band": 0.20,
        "max_step_per_decision": 0.50,
        "sleeve_notional_cap": 1.0,
        "gross_notional_cap": 2.0,
        "net_notional_cap": 1.5,
    }
    out = replay_targets(p, e, cfg, target_suffix="base")
    assert float(out["trading_cost"].sum()) > 0
    assert (out["turnover"] >= 0).all()
