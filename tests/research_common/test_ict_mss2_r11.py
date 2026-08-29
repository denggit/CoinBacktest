from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r11 import (
    build_continuous_path_atlas,
    build_continuous_sweep_events,
    build_event_time_liquidity_snapshot,
    build_visible_it_lt_liquidity,
    r11_causal_audit,
)


def _bars(start="2026-01-01", n=24*60*3, base=100.0):
    idx = pd.date_range(start, periods=n, freq="1min")
    x = np.full(n, base, dtype=float)
    return pd.DataFrame({"open":x,"high":x+0.2,"low":x-0.2,"close":x,"volume":1.0}, index=idx)


def _level(swing_id, side, px, available, tf="15m", tf_min=15, is_lt=0, lt_available=pd.NaT):
    return {
        "swing_id": swing_id,
        "pivot_side": side,
        "source_timeframe": tf,
        "source_timeframe_min": tf_min,
        "pivot_time": pd.Timestamp(available) - pd.Timedelta(hours=1),
        "level_price": px,
        "is_it": 1,
        "is_lt": is_lt,
        "it_available_time": pd.Timestamp(available),
        "lt_available_time": lt_available,
    }


def test_broad_it_lt_universe_excludes_st_and_consumes_only_after_activation_bar_start():
    b = _bars(n=500)
    # A low on the 00:59 bar cannot consume a level that only becomes available at 01:00.
    b.loc[pd.Timestamp("2026-01-01 00:59"), "low"] = 94.0
    b.loc[pd.Timestamp("2026-01-01 02:00"), "low"] = 94.0
    h = pd.DataFrame([
        _level("it", "low", 95.0, "2026-01-01 01:00"),
        {"swing_id":"st","pivot_side":"high","source_timeframe":"15m","source_timeframe_min":15,
         "pivot_time":pd.Timestamp("2026-01-01 00:10"),"level_price":105.0,"is_it":0,"is_lt":0,
         "it_available_time":pd.NaT,"lt_available_time":pd.NaT},
    ])
    v = build_visible_it_lt_liquidity(b, h)
    assert list(v.swing_id) == ["it"]
    assert v.iloc[0].first_consumption_time == pd.Timestamp("2026-01-01 02:00")


def test_intraday_new_liquidity_enters_map_without_waiting_for_midnight():
    b = _bars(start="2026-01-01", n=24*60*2)
    b.loc[pd.Timestamp("2026-01-01 15:00"), "low"] = 94.0
    h = pd.DataFrame([_level("low", "low", 95.0, "2026-01-01 12:00")])
    v = build_visible_it_lt_liquidity(b, h)
    e = build_continuous_sweep_events(
        b, v, research_start=pd.Timestamp("2026-01-01"), research_end=pd.Timestamp("2026-01-02")
    )
    assert len(e) == 1
    assert e.iloc[0].sweep_time == pd.Timestamp("2026-01-01 15:00")


def test_continuous_path_can_cross_midnight_and_freezes_target_at_sweep_time():
    b = _bars(start="2026-01-01 20:00", n=12*60)
    b.loc[pd.Timestamp("2026-01-01 23:50"), "low"] = 94.0
    b.loc[pd.Timestamp("2026-01-02 00:20"), "high"] = 106.0
    h = pd.DataFrame([
        _level("l", "low", 95.0, "2026-01-01 21:00"),
        _level("h", "high", 105.0, "2026-01-01 21:00", tf="30m", tf_min=30),
    ])
    v = build_visible_it_lt_liquidity(b, h)
    e = build_continuous_sweep_events(
        b, v, research_start=pd.Timestamp("2026-01-01 20:00"), research_end=pd.Timestamp("2026-01-02 07:59")
    )
    p = build_continuous_path_atlas(b, v, e)
    root = p.loc[p.root_sweep_time.eq(pd.Timestamp("2026-01-01 23:50"))].iloc[0]
    assert root.root_sweep_side == "SSL"
    assert int(root.opposite_target_hit_1440m_flag) == 1
    assert root.opposite_target_hit_1440m_time == pd.Timestamp("2026-01-02 00:20")
    assert root.path_archetype == "ssl_to_frozen_bsl_within_24h"


def test_liquidity_confirmed_after_root_sweep_cannot_be_backfilled_as_target():
    b = _bars(start="2026-01-01", n=24*60)
    b.loc[pd.Timestamp("2026-01-01 10:00"), "low"] = 94.0
    b.loc[pd.Timestamp("2026-01-01 13:00"), "high"] = 111.0
    h = pd.DataFrame([
        _level("l", "low", 95.0, "2026-01-01 08:00"),
        # This BSL only becomes known AFTER the SSL root sweep.
        _level("late_h", "high", 110.0, "2026-01-01 12:00"),
    ])
    v = build_visible_it_lt_liquidity(b, h)
    e = build_continuous_sweep_events(
        b, v, research_start=pd.Timestamp("2026-01-01"), research_end=pd.Timestamp("2026-01-01 23:59")
    )
    p = build_continuous_path_atlas(b, v, e)
    root = p.loc[p.root_sweep_time.eq(pd.Timestamp("2026-01-01 10:00"))].iloc[0]
    assert pd.isna(root.get("opposite_target_region_id")) or root.get("opposite_target_region_id") is None


def test_same_minute_two_sided_is_ambiguous_not_directionally_relabelled():
    b = _bars(start="2026-01-01", n=24*60)
    t = pd.Timestamp("2026-01-01 10:00")
    b.loc[t, "low"] = 94.0
    b.loc[t, "high"] = 106.0
    h = pd.DataFrame([
        _level("l", "low", 95.0, "2026-01-01 08:00"),
        _level("h", "high", 105.0, "2026-01-01 08:00"),
    ])
    v = build_visible_it_lt_liquidity(b, h)
    e = build_continuous_sweep_events(
        b, v, research_start=pd.Timestamp("2026-01-01"), research_end=pd.Timestamp("2026-01-01 23:59")
    )
    p = build_continuous_path_atlas(b, v, e)
    row = p.loc[p.root_sweep_time.eq(t)].iloc[0]
    assert row.path_archetype == "same_bar_two_sided"
    assert row.root_sweep_side == "BOTH"


def test_manual_snapshot_uses_exact_root_time_active_map_not_day_open():
    b = _bars(start="2026-01-01", n=24*60)
    h = pd.DataFrame([
        _level("l", "low", 95.0, "2026-01-01 12:00"),
        _level("h", "high", 105.0, "2026-01-01 13:00"),
    ])
    v = build_visible_it_lt_liquidity(b, h)
    snap = build_event_time_liquidity_snapshot(b, v, [pd.Timestamp("2026-01-01 12:30")])
    assert "l" in "|".join(snap.swing_ids.astype(str))
    assert "h" not in "|".join(snap.swing_ids.astype(str))


def test_causal_audit_clean_on_continuous_path():
    b = _bars(start="2026-01-01", n=24*60)
    b.loc[pd.Timestamp("2026-01-01 10:00"), "low"] = 94.0
    b.loc[pd.Timestamp("2026-01-01 10:10"), "close"] = 96.0
    h = pd.DataFrame([_level("l", "low", 95.0, "2026-01-01 08:00")])
    v = build_visible_it_lt_liquidity(b, h)
    e = build_continuous_sweep_events(
        b, v, research_start=pd.Timestamp("2026-01-01"), research_end=pd.Timestamp("2026-01-01 23:59")
    )
    p = build_continuous_path_atlas(b, v, e)
    a = r11_causal_audit(v, e, p)
    assert int(a.violations.sum()) == 0
