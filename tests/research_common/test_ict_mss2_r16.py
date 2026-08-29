from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r16 import (
    build_stop_model_outcomes,
    prepare_stop_atlas_universe,
    r16_causal_audit,
)


def _bars(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    x = np.full(n, 100.0)
    return pd.DataFrame({"open": x, "high": x + 0.1, "low": x - 0.1, "close": x, "volume": 1.0}, index=idx)


def _inputs(b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries = pd.DataFrame([{
        "root_event_id": "e1", "root_sweep_time": b.index[10], "root_side": "SSL",
        "research_split": "discovery", "year": 2024, "entry_model": "root_close_outside",
        "entry_status": "filled", "signal_available_time": b.index[11], "entry_time": b.index[11],
        "entry_price": 100.0, "stop_price": 101.02,
    }])
    features = pd.DataFrame([{
        "root_event_id": "e1", "path_horizon_minutes": 80, "root_bar_high": 103.0,
        "root_zone_high": 101.0, "deeper_same_side_touch_price": 95.0,
    }])
    return entries, features


def test_stop_atlas_freezes_ssl_root_entry_and_three_models():
    b = _bars(); entries, features = _inputs(b)
    q = prepare_stop_atlas_universe(entries, features)
    b.loc[b.index[20], "low"] = 94.0
    paths = build_stop_model_outcomes(b, q)
    assert set(paths["stop_model"]) == {"region_edge_touch", "root_bar_extreme_touch", "close_reclaim_plus_extreme"}


def test_stop_atlas_does_not_suffix_fields_already_carried_by_entry_rows():
    b = _bars(); entries, features = _inputs(b)
    entries["root_zone_high"] = 101.0
    entries["deeper_same_side_touch_price"] = 95.0
    q = prepare_stop_atlas_universe(entries, features)
    assert "root_zone_high" in q
    assert "deeper_same_side_touch_price" in q
    assert not any(c.endswith("_x") or c.endswith("_y") for c in q.columns)


def test_touch_stop_is_pessimistic_when_target_and_stop_share_bar():
    b = _bars(); entries, features = _inputs(b)
    q = prepare_stop_atlas_universe(entries, features)
    b.loc[b.index[11], ["high", "low"]] = [104.0, 94.0]
    paths = build_stop_model_outcomes(b, q)
    touch = paths.loc[paths["stop_model"].eq("root_bar_extreme_touch")].iloc[0]
    assert touch["outcome"] == "sl_first"


def test_target_tied_with_reclaim_close_is_behavioral_failure_next_open():
    b = _bars(); entries, features = _inputs(b)
    q = prepare_stop_atlas_universe(entries, features)
    b.loc[b.index[20], ["high", "low", "close"]] = [102.0, 94.0, 102.0]
    b.loc[b.index[21], "open"] = 102.5
    paths = build_stop_model_outcomes(b, q)
    behavioral = paths.loc[paths["stop_model"].eq("close_reclaim_plus_extreme")].iloc[0]
    assert behavioral["outcome"] == "close_reclaim_exit"
    assert behavioral["exit_time"] == b.index[21]
    assert float(behavioral["exit_price"]) == 102.5


def test_target_on_earlier_bar_beats_later_reclaim():
    b = _bars(); entries, features = _inputs(b)
    q = prepare_stop_atlas_universe(entries, features)
    b.loc[b.index[20], "low"] = 94.0
    b.loc[b.index[21], "close"] = 102.0
    paths = build_stop_model_outcomes(b, q)
    behavioral = paths.loc[paths["stop_model"].eq("close_reclaim_plus_extreme")].iloc[0]
    assert behavioral["outcome"] == "tp_first"
    assert behavioral["exit_time"] == b.index[20]


def test_r16_causal_audit_zero_for_valid_paths():
    b = _bars(); entries, features = _inputs(b)
    q = prepare_stop_atlas_universe(entries, features)
    b.loc[b.index[20], "low"] = 94.0
    paths = build_stop_model_outcomes(b, q)
    audit = r16_causal_audit(paths, holdout_start=pd.Timestamp("2025-08-01"))
    assert int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum()) == 0
