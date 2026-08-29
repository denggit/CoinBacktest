from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.research_common.ict.entry_expansion import (
    EntryExpansionConfig,
    build_intraday_15m_swing_catalog,
    build_intraday_15m_sweep_events,
    expand_entry_models,
    expand_intraday_target_models,
)
from src.research_common.ict.premarket_mss_fvg import NY_TZ


def _intraday_test_bars() -> pd.DataFrame:
    idx = pd.date_range("2026-06-02 04:00", "2026-06-02 16:29", freq="1min", tz=NY_TZ)
    out = pd.DataFrame(index=idx)
    out["open"] = 105.0
    out["high"] = 105.2
    out["low"] = 104.8
    out["close"] = 105.0
    out["volume"] = 1000.0
    # Premarket external range.
    out.loc[pd.Timestamp("2026-06-02 05:00", tz=NY_TZ), "high"] = 110.0
    out.loc[pd.Timestamp("2026-06-02 07:00", tz=NY_TZ), "low"] = 100.0

    def block(start: str, high: float, low: float, close: float) -> None:
        s = pd.Timestamp(start, tz=NY_TZ)
        e = s + pd.Timedelta(minutes=14)
        sl = out.loc[s:e]
        out.loc[s:e, "open"] = close
        out.loc[s:e, "close"] = close
        out.loc[s:e, "high"] = min(high, close + 0.2)
        out.loc[s:e, "low"] = max(low, close - 0.2)
        out.loc[s, "high"] = high
        out.loc[s, "low"] = low

    # Both premarket sides are consumed early.
    block("2026-06-02 08:30", 111.0, 104.0, 106.0)
    block("2026-06-02 08:45", 107.0, 99.0, 105.0)
    # Create a causal 15m pivot low at 09:15, confirmed by 09:45.
    block("2026-06-02 09:00", 107.0, 102.0, 105.0)
    block("2026-06-02 09:15", 106.0, 101.0, 104.0)
    block("2026-06-02 09:30", 108.0, 103.0, 106.0)
    # Create opposing pivot high at 09:45, confirmed by 10:15.
    block("2026-06-02 09:45", 109.0, 104.0, 107.0)
    block("2026-06-02 10:00", 108.0, 103.0, 105.0)
    # Sweep the 09:15 pivot low only after both pivots are causally known.
    block("2026-06-02 10:15", 107.0, 100.5, 104.0)
    block("2026-06-02 10:30", 108.0, 103.0, 106.0)
    return out


def _premarket_levels() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "ny_date": "2026-06-02", "liquidity_side": "high", "level_type": "premarket_extreme",
            "level_price": 110.0, "premarket_high": 110.0, "premarket_low": 100.0,
            "premarket_range": 10.0, "premarket_range_pct": 10.0 / 105.0,
            "premarket_close": 105.0, "premarket_median_15m_range": 1.0,
        },
        {
            "ny_date": "2026-06-02", "liquidity_side": "low", "level_type": "premarket_extreme",
            "level_price": 100.0, "premarket_high": 110.0, "premarket_low": 100.0,
            "premarket_range": 10.0, "premarket_range_pct": 10.0 / 105.0,
            "premarket_close": 105.0, "premarket_median_15m_range": 1.0,
        },
    ])


def test_intraday_15m_swing_is_causal_and_can_form_after_both_premarket_sides_consumed():
    bars = _intraday_test_bars()
    cfg = EntryExpansionConfig(intraday_pivot_left=1, intraday_pivot_right=1)
    catalog = build_intraday_15m_swing_catalog(bars, [date(2026, 6, 2)], _premarket_levels(), config=cfg)
    assert not catalog.empty
    target = catalog.loc[
        (catalog["liquidity_side"] == "low")
        & (pd.to_datetime(catalog["pivot_time"]) == pd.Timestamp("2026-06-02 09:15", tz=NY_TZ))
    ]
    assert len(target) == 1
    row = target.iloc[0]
    assert pd.Timestamp(row["level_available_time"]) == pd.Timestamp("2026-06-02 09:45", tz=NY_TZ)
    assert row["premarket_consumption_state_at_level_confirmation"] == "both_premarket_sides_consumed"


def test_intraday_sweep_builds_local_equilibrium_and_opposite_swing_targets():
    bars = _intraday_test_bars()
    cfg = EntryExpansionConfig(intraday_pivot_left=1, intraday_pivot_right=1)
    catalog = build_intraday_15m_swing_catalog(bars, [date(2026, 6, 2)], _premarket_levels(), config=cfg)
    sweeps = build_intraday_15m_sweep_events(bars, catalog, config=cfg)
    low_sweep = sweeps.loc[
        (sweeps["trade_side"] == "LONG")
        & np.isclose(pd.to_numeric(sweeps["level_price"]), 101.0)
    ]
    assert len(low_sweep) == 1
    row = low_sweep.iloc[0]
    assert row["premarket_consumption_state_at_sweep"] == "both_premarket_sides_consumed"
    assert np.isclose(float(row["local_opposite_15m_price"]), 109.0)
    assert np.isclose(float(row["local_equilibrium_50"]), 105.0)
    expanded = expand_intraday_target_models(low_sweep, config=cfg)
    assert set(expanded["target_model"]) == {"local_equilibrium_50", "local_opposite_15m_swing"}
    assert set(np.round(pd.to_numeric(expanded["target_price"]), 6)) == {105.0, 109.0}


def test_entry_expansion_preserves_fvg_and_adds_ce_and_order_block_proxies():
    idx = pd.date_range("2026-06-02 09:55", periods=20, freq="1min", tz=NY_TZ)
    bars = pd.DataFrame(index=idx)
    bars["open"] = 103.0; bars["high"] = 103.2; bars["low"] = 102.8; bars["close"] = 103.1; bars["volume"] = 1000.0
    # Latest opposite-close candle inside terminal->MSS leg.
    bars.loc[pd.Timestamp("2026-06-02 10:03", tz=NY_TZ), ["open", "high", "low", "close"]] = [103.0, 103.2, 102.4, 102.5]
    bars.loc[pd.Timestamp("2026-06-02 10:04", tz=NY_TZ), ["open", "high", "low", "close"]] = [102.6, 105.2, 102.5, 105.0]
    attempts = pd.DataFrame([{
        "attempt_id": "a", "ny_date": "2026-06-02", "execution_tf": "1m", "execution_tf_minutes": 1,
        "trade_side": "LONG", "episode_terminal_extreme_time": pd.Timestamp("2026-06-02 10:00", tz=NY_TZ),
        "mss_time": pd.Timestamp("2026-06-02 10:05", tz=NY_TZ), "signal_time": pd.Timestamp("2026-06-02 10:06", tz=NY_TZ),
        "signal_close": 105.0, "fvg_near_edge_entry": 103.0, "fvg_far_edge": 102.0,
        "stop_price": 100.0, "target_price": 110.0, "risk_abs": 3.0, "risk_pct": 3/103,
        "planned_reward_abs": 7.0, "planned_rr": 7/3,
    }])
    expanded, audit = expand_entry_models(attempts, bars, config=EntryExpansionConfig())
    assert set(expanded["entry_model"]) == {
        "fvg_near_edge", "fvg_ce_50", "order_block_open_proxy", "order_block_midpoint_proxy"
    }
    ce = expanded.loc[expanded["entry_model"] == "fvg_ce_50"].iloc[0]
    assert np.isclose(float(ce["entry_model_price"]), 102.5)
    ob = expanded.loc[expanded["entry_model"] == "order_block_open_proxy"].iloc[0]
    assert np.isclose(float(ob["entry_model_price"]), 103.0)
    assert audit["valid_entry_variant"].all()
