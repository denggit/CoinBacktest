#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent pre-open seal for the untouched July-2026 extension."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.ai_research.config import PROJECT_ROOT
from src.ai_research.long_tail_sealed_holdout.seal import canonical_hash, sha256_file

from .config import ForwardExtensionConfig, STAGE_ID

_FROZEN_CODE_PATHS = (
    "src/ai_research/long_state_calibration/modeling.py",
    "src/ai_research/long_tail_exit_audit/modeling.py",
    "src/ai_research/long_tail_exit_audit/simulator.py",
    "src/ai_research/long_tail_structural_exit/config.py",
    "src/ai_research/long_tail_structural_exit/structure.py",
    "src/ai_research/long_tail_structural_exit/simulator.py",
    "src/ai_research/long_tail_risk_migration/structure.py",
    "src/ai_research/long_tail_soft_failure_tail_compression/config.py",
    "src/ai_research/long_tail_soft_failure_tail_compression/simulator.py",
    "src/ai_research/long_tail_sealed_holdout/pipeline.py",
    "src/ai_research/long_tail_forward_extension/config.py",
    "src/ai_research/long_tail_forward_extension/pipeline.py",
    "src/ai_research/long_tail_forward_extension/reports.py",
    "src/ai_research/long_tail_forward_extension/seal.py",
    "research/eth_ai_trading/03_4_2_16_1_2026_july_forward_extension.py",
)

_SOURCE_2_15_FILES = (
    "00_run_manifest.json",
    "05_continuous_scenario_summary.csv",
    "12_final_gate.csv",
    "13_causal_audit.csv",
    "99_decision.md",
)

_SOURCE_2_16_FILES = (
    "00_pre_open_seal.json",
    "01_holdout_open_log.json",
    "03_model_threshold_audit.csv",
    "12_holdout_scenario_summary.csv",
    "15_extended_oos_summary.csv",
    "18_post_run_seal_check.json",
    "20_failures.csv",
    "99_decision.md",
)


def build_seal_payload(config: ForwardExtensionConfig) -> dict[str, Any]:
    config.validate()
    code_hashes: dict[str, str] = {}
    for relative in _FROZEN_CODE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        code_hashes[relative] = sha256_file(path)

    source_hashes: dict[str, str] = {}
    for name in _SOURCE_2_15_FILES:
        path = config.source_2_15_path / name
        if not path.exists():
            raise FileNotFoundError(path)
        source_hashes[f"R03.4.2.15/{name}"] = sha256_file(path)
    for name in _SOURCE_2_16_FILES:
        path = config.source_2_16_path / name
        if not path.exists():
            raise FileNotFoundError(path)
        source_hashes[f"R03.4.2.16/{name}"] = sha256_file(path)

    score_source = config.source_2_8a_path / "02_score_threshold_audit.csv"
    if not score_source.exists():
        raise FileNotFoundError(score_source)
    source_hashes["R03.4.2.8A/02_score_threshold_audit.csv"] = sha256_file(score_source)

    payload: dict[str, Any] = {
        "stage": STAGE_ID,
        "config": config.to_dict(),
        "frozen_code_sha256": code_hashes,
        "source_report_sha256": source_hashes,
        "forward_window": "2026-07-01 through 2026-07-31",
        "mutation_policy": "same seal may be reproduced; changed seal after July opening is forbidden",
    }
    payload["seal_sha256"] = canonical_hash(payload)
    return payload


def ensure_pre_open_seal(config: ForwardExtensionConfig) -> dict[str, Any]:
    root = config.report_path
    root.mkdir(parents=True, exist_ok=True)
    path = root / "00_pre_open_seal.json"
    current = build_seal_payload(config)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("seal_sha256") != current.get("seal_sha256"):
            raise RuntimeError(
                "July forward extension was previously opened under different code/config/source hashes; "
                "post-open mutation is forbidden"
            )
        return existing
    path.write_text(
        json.dumps(current, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return current


def verify_post_run_seal(
    config: ForwardExtensionConfig,
    pre_open: dict[str, Any],
) -> dict[str, Any]:
    current = build_seal_payload(config)
    passed = current.get("seal_sha256") == pre_open.get("seal_sha256")
    return {
        "status": "PASS" if passed else "FAIL",
        "pre_open_seal_sha256": pre_open.get("seal_sha256"),
        "post_run_seal_sha256": current.get("seal_sha256"),
        "unchanged": bool(passed),
    }
