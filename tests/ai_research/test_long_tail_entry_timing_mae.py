from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ai_research.long_tail_entry_timing_mae.analysis import policy_gate
from src.ai_research.long_tail_entry_timing_mae.config import EntryTimingConfig, EntryTimingPolicy
from src.ai_research.long_tail_entry_timing_mae.inputs import load_entry_timing_inputs
from src.ai_research.long_tail_entry_timing_mae.simulator import _entry_position_for_policy
from src.ai_research.long_tail_exit_audit.data import prepare_minute_path_frame


def _path() -> object:
    index = pd.date_range("2024-01-01", periods=240, freq="min")
    close = np.full(len(index), 100.0)
    frame = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1, "close": close}, index=index)
    return prepare_minute_path_frame(frame)


def _event() -> dict[str, object]:
    return {"event_id": "e1", "decision_time": pd.Timestamp("2024-01-01 01:00"), "score": 1.0}


def test_config_freezes_c2_and_bounded_wait() -> None:
    config = EntryTimingConfig()
    config.validate()
    assert config.hard_stop_distance == 0.02
    assert config.soft_failure_distance == 0.015
    with pytest.raises(ValueError):
        EntryTimingPolicy("bad", "score_rise", max_wait_minutes=90).validate()


def test_immediate_policy_uses_next_open() -> None:
    path = _path()
    policy = EntryTimingPolicy("E0_immediate_C2", "immediate", qualifying_candidate=False)
    position, meta = _entry_position_for_policy(_event(), pd.DataFrame(), path=path, policy=policy, delay_minutes=1)
    assert pd.Timestamp(path.index[position]) == pd.Timestamp("2024-01-01 01:01")
    assert meta["trigger_reason"] == "immediate_q70"


def test_score_rise_waits_for_higher_q70() -> None:
    path = _path()
    signals = pd.DataFrame([
        {"decision_time": pd.Timestamp("2024-01-01 01:15"), "score": 0.9},
        {"decision_time": pd.Timestamp("2024-01-01 01:30"), "score": 1.2},
    ])
    policy = EntryTimingPolicy("E1", "score_rise", max_wait_minutes=30)
    position, meta = _entry_position_for_policy(_event(), signals, path=path, policy=policy, delay_minutes=1)
    assert pd.Timestamp(path.index[position]) == pd.Timestamp("2024-01-01 01:31")
    assert meta["trigger_score_delta"] == pytest.approx(0.2)


def test_score_rise_no_chase_rejects_expensive_confirmation_and_falls_back() -> None:
    path = _path()
    # Inflate the confirmation open beyond the allowed ATR chase distance.
    path.open[91] = 102.0
    signals = pd.DataFrame([{"decision_time": pd.Timestamp("2024-01-01 01:30"), "score": 1.2}])
    policy = EntryTimingPolicy("E2", "score_rise_no_chase", max_wait_minutes=45, maximum_chase_atr=0.25)
    position, meta = _entry_position_for_policy(_event(), signals, path=path, policy=policy, delay_minutes=1)
    assert pd.Timestamp(path.index[position]) == pd.Timestamp("2024-01-01 01:46")
    assert meta["fallback_used"]


def test_pullback_reclaim_uses_completed_five_minute_close() -> None:
    path = _path()
    # Arm the pullback, then reclaim on the completed 01:05-01:09 five-minute bar.
    path.low[63] = 99.4
    path.close[69] = 100.0
    policy = EntryTimingPolicy("E3", "pullback_reclaim", max_wait_minutes=60)
    position, meta = _entry_position_for_policy(_event(), pd.DataFrame(), path=path, policy=policy, delay_minutes=1)
    assert pd.Timestamp(path.index[position]) == pd.Timestamp("2024-01-01 01:05")
    assert meta["trigger_reason"] == "pullback_reclaim_5m_close"


def test_policy_gate_requires_coverage_and_quality_uplift() -> None:
    config = EntryTimingConfig(policies=(
        EntryTimingPolicy("E0_immediate_C2", "immediate", qualifying_candidate=False),
        EntryTimingPolicy("candidate", "score_rise", max_wait_minutes=30),
    ))
    rows = []
    for fold in ("WF_2024", "WF_2025"):
        for delay in (1, 3, 5):
            for cost in (2.0, 3.0):
                rows.append({"fold_id": fold, "policy": "E0_immediate_C2", "delay_minutes": delay, "cost_multiplier": cost, "total_net_return": 1.0, "max_drawdown": -0.1, "coverage_ratio": 1.0, "win_rate": 0.45, "hard_stop_share": 0.15, "soft_failure_share": 0.10, "mean_mae_60m": -0.01, "top10_profit_share": 0.4, "total_return_without_top10": 0.1, "positive_quarters": 4})
                rows.append({"fold_id": fold, "policy": "candidate", "delay_minutes": delay, "cost_multiplier": cost, "total_net_return": 1.02, "max_drawdown": -0.09, "coverage_ratio": 0.95, "win_rate": 0.47, "hard_stop_share": 0.12, "soft_failure_share": 0.08, "mean_mae_60m": -0.008, "top10_profit_share": 0.41, "total_return_without_top10": 0.1, "positive_quarters": 4})
    gate = policy_gate(pd.DataFrame(rows), config)
    assert bool(gate.set_index("policy").loc["candidate", "pass_to_next_stage"])


def test_loader_requires_frozen_source_decisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    roots = {name: tmp_path / name for name in ("r8", "r12", "r13")}
    for root in roots.values(): root.mkdir()
    (roots["r8"] / "00_run_manifest.json").write_text(json.dumps({"stage": "R03.4.2.8A", "folds": []}), encoding="utf-8")
    (roots["r8"] / "99_decision.md").write_text("FAIL", encoding="utf-8")
    pd.DataFrame().to_csv(roots["r8"] / "04_frozen_baseline_summary.csv", index=False)
    pd.DataFrame().to_csv(roots["r8"] / "16_standalone_signal_outcomes.csv", index=False)
    (roots["r12"] / "00_run_manifest.json").write_text(json.dumps({"stage": "R03.4.2.12"}), encoding="utf-8")
    (roots["r12"] / "99_decision.md").write_text("FAIL", encoding="utf-8")
    pd.DataFrame().to_csv(roots["r12"] / "04_selected_p0_cycles.csv", index=False)
    (roots["r13"] / "00_run_manifest.json").write_text(json.dumps({"stage": "R03.4.2.13"}), encoding="utf-8")
    (roots["r13"] / "99_decision.md").write_text("FAIL", encoding="utf-8")
    for name in ("05_account_cycles.csv", "06_account_legs.csv", "08_policy_summary.csv", "10_causal_audit.csv", "12_failures.csv"):
        pd.DataFrame().to_csv(roots["r13"] / name, index=False)
    config = EntryTimingConfig()
    monkeypatch.setattr(type(config), "source_2_8a_path", property(lambda self: roots["r8"]))
    monkeypatch.setattr(type(config), "source_2_12_path", property(lambda self: roots["r12"]))
    monkeypatch.setattr(type(config), "source_2_13_path", property(lambda self: roots["r13"]))
    with pytest.raises(RuntimeError, match="passed C2"):
        load_entry_timing_inputs(config)
