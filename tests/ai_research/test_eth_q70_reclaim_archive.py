from __future__ import annotations

import json
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _model_dir() -> Path:
    return _root() / "research" / "eth_ai_trading" / "archived_models" / "eth_q70_reclaim_mf_long_v1"


def test_archive_required_files_exist() -> None:
    model_dir = _model_dir()
    required = {
        "README.md",
        "MODEL_CARD.md",
        "FROZEN_POLICY.json",
        "EMPIRICAL_RESULTS.md",
        "RESEARCH_TIMELINE.md",
        "FAILURE_AND_LESSONS.md",
        "REPRODUCTION_AND_REPORT_INDEX.md",
        "NEXT_MODEL_BOUNDARY.md",
        "LIFECYCLE_LOCK.json",
        "ARCHIVE_MANIFEST.json",
    }
    assert required.issubset({p.name for p in model_dir.iterdir() if p.is_file()})


def test_archive_is_zero_capital_and_not_live_approved() -> None:
    lifecycle = json.loads((_model_dir() / "LIFECYCLE_LOCK.json").read_text(encoding="utf-8"))
    assert lifecycle["binding_decision"] == "FAIL_2026_SEALED_HOLDOUT"
    assert lifecycle["final_diagnosis"] == "DIAGNOSIS_SCORE_DRIFT_DOMINANT"
    assert lifecycle["live_approved"] is False
    assert lifecycle["capital_allocation"] == 0.0
    assert lifecycle["retuning_on_2026_h1_or_july_allowed"] is False


def test_frozen_policy_preserves_c2_contract() -> None:
    policy = json.loads((_model_dir() / "FROZEN_POLICY.json").read_text(encoding="utf-8"))
    assert policy["entry"]["threshold_name"] == "q70"
    assert policy["risk"]["hard_stop_fraction"] == 0.02
    assert policy["risk"]["soft_failure_trigger_fraction"] == 0.015
    assert policy["exit"]["primary"] == "failed_reclaim deterministic non-time exit"
    assert policy["position_management"]["add_on"] is False
    assert policy["live_approved"] is False


def test_next_model_is_not_breakout_chasing() -> None:
    text = (_model_dir() / "NEXT_MODEL_BOUNDARY.md").read_text(encoding="utf-8")
    assert "not a breakout-chasing entry model" in text
    assert "pullback" in text.lower()
    assert "skip" in text.lower()
