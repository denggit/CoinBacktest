from __future__ import annotations

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.r15 import (
    build_fixed_r_first_passage,
    prepare_fixed_r_universe,
    r15_causal_audit,
)


def _bars(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    x = np.full(n, 100.0)
    return pd.DataFrame({"open": x, "high": x + 0.1, "low": x - 0.1, "close": x, "volume": 1.0}, index=idx)


def _inputs(b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    t = b.index[10]
    entry = b.index[11]
    entries = pd.DataFrame([{
        "root_event_id": "e1", "root_sweep_time": t, "root_side": "SSL",
        "research_split": "discovery", "year": 2024, "entry_model": "root_close_outside",
        "entry_status": "filled", "signal_available_time": entry, "entry_time": entry,
        "entry_price": 100.0, "stop_price": 102.0, "exit_time": b.index[50],
    }])
    features = pd.DataFrame([{"root_event_id": "e1", "path_horizon_minutes": 80}])
    return entries, features


def test_prepare_fixed_r_universe_freezes_only_ssl_root_acceptance():
    b = _bars()
    entries, features = _inputs(b)
    extra = entries.iloc[0].copy()
    extra["root_event_id"] = "e2"; extra["root_side"] = "BSL"
    entries = pd.concat([entries, extra.to_frame().T], ignore_index=True)
    features = pd.concat([features, pd.DataFrame([{"root_event_id": "e2", "path_horizon_minutes": 80}])], ignore_index=True)
    q = prepare_fixed_r_universe(entries, features)
    assert list(q["root_event_id"]) == ["e1"]


def test_fixed_r_targets_use_unchanged_stop_and_stop_first_same_bar():
    b = _bars()
    entries, features = _inputs(b)
    q = prepare_fixed_r_universe(entries, features)
    # Risk is 2 price units. On entry bar, both 1R target (98) and stop (102) trade.
    b.loc[b.index[11], ["open", "high", "low", "close"]] = [100.0, 103.0, 97.0, 100.0]
    paths = build_fixed_r_first_passage(b, q)
    one = paths.loc[paths["r_target"].eq(1.0)].iloc[0]
    assert float(one["target_price"]) == 98.0
    assert float(one["stop_price"]) == 102.0
    assert one["outcome"] == "sl_first"
    assert float(one["gross_r"]) == -1.0


def test_neighboring_r_targets_have_exact_short_geometry():
    b = _bars()
    entries, features = _inputs(b)
    q = prepare_fixed_r_universe(entries, features)
    b.loc[b.index[20], "low"] = 93.0
    paths = build_fixed_r_first_passage(b, q)
    assert dict(zip(paths["r_target"], paths["target_price"])) == {0.5: 99.0, 1.0: 98.0, 2.0: 96.0, 3.0: 94.0}
    assert set(paths["outcome"]) == {"tp_first"}


def test_r15_causal_audit_zero_for_valid_paths():
    b = _bars()
    entries, features = _inputs(b)
    q = prepare_fixed_r_universe(entries, features)
    b.loc[b.index[20], "low"] = 93.0
    paths = build_fixed_r_first_passage(b, q)
    audit = r15_causal_audit(paths, holdout_start=pd.Timestamp("2025-08-01"))
    assert int(pd.to_numeric(audit["violations"], errors="coerce").fillna(0).sum()) == 0
