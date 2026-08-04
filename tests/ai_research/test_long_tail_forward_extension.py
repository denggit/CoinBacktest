from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.ai_research.long_tail_forward_extension.config import (
    FOLD_ID,
    ForwardExtensionConfig,
)
from src.ai_research.long_tail_forward_extension.pipeline import (
    _build_gate,
    _causal_audit,
    _map_decision,
    _temporary_runtime_contract,
)
from src.ai_research.long_tail_forward_extension.reports import map_decision
from src.ai_research.long_tail_forward_extension.seal import ensure_pre_open_seal
from src.ai_research.long_tail_sealed_holdout import pipeline as sealed_pipeline


def test_forward_config_keeps_pre_2026_model_and_july_only():
    config = ForwardExtensionConfig()
    config.validate()
    assert str(config.fit_end) == "2025-09-30 06:00:00"
    assert config.calibration_end.startswith("2025-12-31")
    assert config.holdout_start.startswith("2026-07-01")
    assert config.holdout_end.startswith("2026-07-31")
    assert config.hard_stop_distance == 0.02
    assert config.soft_failure_distance == 0.015


def test_forward_decision_mapping_is_not_second_holdout_pass():
    assert map_decision("PASS_2026_SEALED_HOLDOUT") == "JULY_FORWARD_SUPPORTS_FROZEN_C2"
    assert _map_decision("FAIL_2026_SEALED_HOLDOUT") == "JULY_FORWARD_DOES_NOT_SUPPORT_FROZEN_C2"


def test_causal_audit_discloses_h1_failure_and_single_month_limit():
    audit = _causal_audit(ForwardExtensionConfig())
    assert audit["status"].eq("PASS").all()
    assert "h1_comparison_only" in set(audit["check"])
    assert "single_month_disclosure" in set(audit["check"])


def test_temporary_contract_restores_sealed_pipeline_globals():
    config = ForwardExtensionConfig()
    old_stage = sealed_pipeline.STAGE_ID
    old_base = sealed_pipeline.LONG_CONTEXT_BASE_CONFIG
    with _temporary_runtime_contract(config):
        assert sealed_pipeline.STAGE_ID == "R03.4.2.16.1"
        assert sealed_pipeline.LONG_CONTEXT_BASE_CONFIG.research_end == config.holdout_end
        assert "r03_4_2_16_1" in sealed_pipeline.LONG_CONTEXT_BASE_CONFIG.cache_dir
    assert sealed_pipeline.STAGE_ID == old_stage
    assert sealed_pipeline.LONG_CONTEXT_BASE_CONFIG == old_base


def test_changed_july_seal_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = ForwardExtensionConfig()
    report = tmp_path / "report"
    report.mkdir()
    monkeypatch.setattr(type(config), "report_path", property(lambda self: report))
    monkeypatch.setattr(
        "src.ai_research.long_tail_forward_extension.seal.build_seal_payload",
        lambda _: {"seal_sha256": "new"},
    )
    (report / "00_pre_open_seal.json").write_text(
        json.dumps({"seal_sha256": "old"}), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="post-open mutation"):
        ensure_pre_open_seal(config)


def test_forward_gate_requires_exact_h1_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = ForwardExtensionConfig()
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame([{"calibration_threshold": 0.123, "fit_rows": 100, "calibration_rows": 20}]).to_csv(source / "03_model_threshold_audit.csv", index=False)
    monkeypatch.setattr(type(config), "source_2_16_path", property(lambda self: source))
    summary = pd.DataFrame([
        {
            "delay_minutes": delay, "cost_multiplier": cost, "executed_cycles": 10,
            "total_net_return": 0.02, "profit_factor": 1.1, "max_drawdown": -0.03,
            "worst_cycle_loss_r": 1.1, "positive_months": 1, "positive_quarters": 0,
            "top10_profit_share": 0.9, "total_return_without_top10": -0.1,
            "censored_cycles": 0, "final_equity": 1.02,
        }
        for delay in (1, 3, 5) for cost in (2.0, 3.0)
    ])
    score = pd.DataFrame([{
        "feature_schema_matches_history": True, "feature_schema_hash": "x",
        "historical_feature_schema_hash": "x", "calibration_threshold": 0.123,
        "fit_rows": 100, "calibration_rows": 20,
    }])
    gate = _build_gate(summary, score, {"unchanged": True, "status": "PASS"}, config)
    exact = gate.loc[gate["check"].eq("frozen_threshold_matches_h1")].iloc[0]
    assert bool(exact["pass"])
