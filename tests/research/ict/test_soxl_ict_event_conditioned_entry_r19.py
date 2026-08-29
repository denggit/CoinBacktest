from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "research" / "ict" / "soxl_premarket_mss_fvg" / "19_event_conditioned_entry_study.py"


def _load():
    spec = importlib.util.spec_from_file_location("soxl_r19", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_causal_probability_must_exist_before_order():
    m = _load()
    event = pd.DataFrame({
        "path_event_id": ["a"],
        "predicted_probability": [0.7],
        "event_probability_band": ["0.70-0.80"],
        "event_probability_available_time": ["2026-01-01T10:00:00Z"],
        "target_opposite_by_eod": [1],
    })
    life = pd.DataFrame({
        "path_event_id": ["a", "a"],
        "entry_available_time": ["2026-01-01T09:59:00Z", "2026-01-01T10:01:00Z"],
    })
    q = m._causal_join(life, event)
    assert q["event_probability_available_at_order"].tolist() == [False, True]


def test_profit_factor_and_payoff():
    m = _load()
    x = pd.Series([0.02, 0.01, -0.01, -0.01])
    assert abs(m._profit_factor(x) - 1.5) < 1e-12
    assert abs(m._rr_payoff(x) - 1.5) < 1e-12
