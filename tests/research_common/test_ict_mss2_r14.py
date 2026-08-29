from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r14 import (
    R14Config,
    attach_acceptance_features,
    build_continuation_entries,
    prepare_continuation_universe,
    r14_causal_audit,
)


def _bars(n: int = 500, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1min")
    x = np.full(n, 100.0)
    return pd.DataFrame({"open": x, "high": x + 0.1, "low": x - 0.1, "close": x, "volume": 1.0}, index=idx)


def _path(time: str, side: str = "BSL") -> dict[str, object]:
    t = pd.Timestamp(time)
    long = side == "BSL"
    return {
        "root_event_id": f"e_{t:%Y%m%d%H%M}_{side}",
        "root_sweep_time": t,
        "root_sweep_available_time": t + pd.Timedelta(minutes=1),
        "path_start_time": t + pd.Timedelta(minutes=1),
        "root_side": side,
        "root_zone_low": 99.0,
        "root_zone_high": 101.0,
        "root_bar_close": 102.0 if long else 98.0,
        "same_bar_two_sided_root_flag": 0,
        "deeper_same_side_available_flag": 1,
        "deeper_same_side_touch_price": 105.0 if long else 95.0,
        "deeper_same_side_touch_time": t + pd.Timedelta(minutes=120),
        "same_side_first_flag": 1,
        "path_outcome": "same_side_continuation_no_opposite_hit",
        "path_horizon_minutes": 300,
        "next_open_time": t + pd.Timedelta(minutes=1),
        "next_open_price": 102.0 if long else 98.0,
    }


def test_continuation_universe_seals_holdout_and_requires_deeper_target():
    missing = _path("2024-01-02 01:00")
    missing["deeper_same_side_available_flag"] = 0
    rows = pd.DataFrame([
        _path("2024-01-01 01:00"), missing, _path("2025-08-02 01:00"),
    ])
    q, seal = prepare_continuation_universe(rows)
    assert len(q) == 1
    assert q.iloc[0]["research_split"] == "discovery"
    assert int(seal.iloc[0]["available_holdout_rows_in_r12"]) == 1
    assert int(seal.iloc[0]["included_in_r14_outputs"]) == 0


def test_root_acceptance_enters_next_bar_and_same_bar_both_is_stop_first():
    b = _bars()
    t = b.index[100]
    b.loc[t, ["open", "high", "low", "close"]] = [100.0, 102.5, 99.5, 102.0]
    b.loc[b.index[101], ["open", "high", "low", "close"]] = [102.0, 106.0, 98.0, 102.0]
    q, _ = prepare_continuation_universe(pd.DataFrame([_path(str(t))]))
    f = attach_acceptance_features(b, q)
    entries = build_continuation_entries(b, f)
    root = entries.loc[entries["entry_model"].eq("root_close_outside")].iloc[0]
    assert root["entry_time"] == b.index[101]
    assert root["outcome"] == "sl_first"
    assert float(root["gross_r"]) == -1.0


def test_five_minute_acceptance_uses_five_completed_bars_and_monotone_sensitivity():
    b = _bars()
    t = b.index[100]
    b.loc[t, ["open", "high", "low", "close"]] = [100.0, 102.5, 99.5, 102.0]
    for i in range(101, 106):
        b.loc[b.index[i], ["open", "high", "low", "close"]] = [102.0, 102.5, 101.5, 102.0]
    b.loc[b.index[103], "close"] = 100.5  # four of five closes remain outside
    b.loc[b.index[200], "high"] = 106.0
    q, _ = prepare_continuation_universe(pd.DataFrame([_path(str(t))]))
    f = attach_acceptance_features(b, q)
    assert f.iloc[0]["accept_5m_available_time"] == b.index[106]
    assert f.iloc[0]["accept_5m_p060_signal"] == 1
    assert f.iloc[0]["accept_5m_p080_signal"] == 1
    assert f.iloc[0]["accept_5m_p100_signal"] == 0
    entries = build_continuation_entries(b, f)
    p80 = entries.loc[entries["entry_model"].eq("accept_5m_p080")].iloc[0]
    assert p80["signal_available_time"] == b.index[106]
    assert p80["entry_time"] == b.index[106]


def test_prior_structural_reclaim_makes_later_acceptance_stale():
    b = _bars()
    t = b.index[100]
    b.loc[t, ["open", "high", "low", "close"]] = [100.0, 102.5, 99.5, 102.0]
    for i in range(101, 106):
        b.loc[b.index[i], ["open", "high", "low", "close"]] = [102.0, 102.5, 101.5, 102.0]
    b.loc[b.index[103], "low"] = 98.0
    q, _ = prepare_continuation_universe(pd.DataFrame([_path(str(t))]))
    f = attach_acceptance_features(b, q)
    entries = build_continuation_entries(b, f)
    p100 = entries.loc[entries["entry_model"].eq("accept_5m_p100")].iloc[0]
    assert p100["entry_status"] == "barrier_before_entry"
    assert p100["outcome"] == "stale"


def test_ssl_acceptance_uses_short_geometry_and_deeper_ssl_target():
    b = _bars()
    t = b.index[100]
    b.loc[t, ["open", "high", "low", "close"]] = [100.0, 100.5, 97.5, 98.0]
    b.loc[b.index[120], "low"] = 94.0
    q, _ = prepare_continuation_universe(pd.DataFrame([_path(str(t), side="SSL")]))
    f = attach_acceptance_features(b, q)
    entries = build_continuation_entries(b, f)
    root = entries.loc[entries["entry_model"].eq("root_close_outside")].iloc[0]
    assert root["outcome"] == "tp_first"
    assert float(root["target_price"]) == 95.0
    assert float(root["stop_price"]) > 101.0


def test_r14_causal_audit_zero_for_valid_rows():
    b = _bars()
    t = b.index[100]
    b.loc[t, ["open", "high", "low", "close"]] = [100.0, 102.5, 99.5, 102.0]
    b.loc[b.index[200], "high"] = 106.0
    q, _ = prepare_continuation_universe(pd.DataFrame([_path(str(t))]))
    f = attach_acceptance_features(b, q)
    entries = build_continuation_entries(b, f)
    audit = r14_causal_audit(f, entries, holdout_start=R14Config().holdout_start)
    assert int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum()) == 0
