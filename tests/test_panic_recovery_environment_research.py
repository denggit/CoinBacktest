from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research/liquidity/panic_selloff_rejection_recovery_long/01_environment_and_cluster_scale_in_research.py"
spec = importlib.util.spec_from_file_location("panic_research_01", SCRIPT)
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)


def _multi_episode_sample() -> pd.DataFrame:
    n = 720
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    close = np.full(n, 100.0)
    volume = np.full(n, 100.0)
    for i in range(1, n):
        close[i] = close[i - 1] * (1.00003 if i % 2 else 0.99998)

    starts = [280, 410, 535]
    template = np.array(
        [99.90, 99.75, 99.58, 99.40, 99.20, 98.95, 98.72, 98.66, 98.70, 98.86, 99.05, 99.24, 99.42, 99.55, 99.70]
    )
    for base in starts:
        anchor = close[base - 1]
        scaled = template / 100.0 * anchor
        close[base : base + len(template)] = scaled
        for j in range(base + len(template), min(n, base + 75)):
            close[j] = close[j - 1] * 1.00025
        volume[base : base + 7] = [150, 170, 190, 220, 260, 320, 400]
        volume[base + 7 : base + 15] = [300, 240, 180, 160, 150, 140, 130, 120]

    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.0004
    low = np.minimum(open_, close) * 0.9996
    for base in starts:
        low[base + 6] = min(low[base + 6], close[base + 6] * 0.9975)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _args():
    return mod.parse_args(
        [
            "--start-date", "2024-01-01",
            "--end-date", "2024-01-02",
            "--train-end-date", "2024-01-01 08:00:00",
            "--horizons", "5,15,30",
            "--candidate-horizon", "15",
            "--min-filter-train", "1",
            "--min-filter-holdout", "1",
            "--top-atomic-for-pairs", "4",
            "--top-scale-filters", "1",
            "--cluster-gap-bars", "15",
            "--target-r-list", "1.0",
            "--cost-multipliers", "1.0",
            "--no-progress",
        ]
    )


def test_stage_and_environment_research_is_causal() -> None:
    bars = _multi_episode_sample()
    args = _args()
    horizons = (5, 15, 30)
    context = mod.build_context_features(bars)
    events, _ = mod.build_stage_events(bars, context, args, horizons)
    studied, audit = mod.attach_next_open_outcomes(events, bars, args, horizons)

    assert not studied.empty
    assert {"start", "exhaustion", "signal"}.issubset(set(studied["stage"]))
    assert audit["entry_is_after_event"].all()
    assert audit["entry_index_valid"].all()

    signals = studied[studied["stage"] == "signal"].copy().reset_index(drop=True)
    specs = mod.build_fixed_filters(signals)
    assert all("diagnostic" not in spec.name for spec in specs)
    signals = mod.add_filter_columns(signals, specs)
    atomic, pairs, candidates = mod.evaluate_environment_filters(signals, specs, args)
    assert not atomic.empty
    assert not candidates.empty
    assert candidates.iloc[0]["candidate_name"] == "ALL_GREEN"


def test_cluster_scale_in_is_capped_and_has_no_time_exit() -> None:
    bars = _multi_episode_sample()
    args = _args()
    context = mod.build_context_features(bars)
    events, _ = mod.build_stage_events(bars, context, args, (5, 15, 30))
    studied, _ = mod.attach_next_open_outcomes(events, bars, args, (5, 15, 30))
    signals = studied[studied["stage"] == "signal"].copy().reset_index(drop=True)
    specs = mod.build_fixed_filters(signals)
    signals = mod.add_filter_columns(signals, specs)
    candidates = pd.DataFrame(
        [{"candidate_name": "ALL_GREEN", "filter_expression": "ALL", "source": "baseline", "holdout_pass": True}]
    )

    # Keep unit test fast; production uses numba when installed.
    mod._simulate_cluster_fast = mod._simulate_cluster_python
    trades, summary, yearly = mod.simulate_cluster_variants(bars, signals, candidates, args)
    assert not trades.empty
    assert not summary.empty
    assert not yearly.empty
    assert float(trades["filled_weight"].max()) <= 1.0000001
    assert set(trades["exit_reason"]).issubset({"target", "structural_stop", "end_of_data"})
    assert "time_exit" not in set(trades["exit_reason"])
