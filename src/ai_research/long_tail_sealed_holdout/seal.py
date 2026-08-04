#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Pre-open immutable seal for the 2026 holdout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.ai_research.config import PROJECT_ROOT

from .config import SealedHoldoutConfig

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
    "src/ai_research/long_tail_sealed_holdout/config.py",
    "src/ai_research/long_tail_sealed_holdout/pipeline.py",
)

_SOURCE_REPORT_FILES = (
    "00_run_manifest.json",
    "05_continuous_scenario_summary.csv",
    "12_final_gate.csv",
    "13_causal_audit.csv",
    "99_decision.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_seal_payload(config: SealedHoldoutConfig) -> dict[str, Any]:
    config.validate()
    code_hashes: dict[str, str] = {}
    for relative in _FROZEN_CODE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.exists():
            raise FileNotFoundError(path)
        code_hashes[relative] = sha256_file(path)
    source_hashes: dict[str, str] = {}
    for name in _SOURCE_REPORT_FILES:
        path = config.source_2_15_path / name
        if not path.exists():
            raise FileNotFoundError(path)
        source_hashes[name] = sha256_file(path)
    score_source = config.source_2_8a_path / "02_score_threshold_audit.csv"
    if not score_source.exists():
        raise FileNotFoundError(score_source)
    source_hashes["R03.4.2.8A/02_score_threshold_audit.csv"] = sha256_file(score_source)
    payload = {
        "stage": "R03.4.2.16",
        "config": config.to_dict(),
        "frozen_code_sha256": code_hashes,
        "source_report_sha256": source_hashes,
        "holdout_mutation_policy": "same seal may be reproduced; changed seal after opening is forbidden",
    }
    payload["seal_sha256"] = canonical_hash(payload)
    return payload


def ensure_pre_open_seal(config: SealedHoldoutConfig) -> dict[str, Any]:
    """Create the seal, or require an exact match for a reproducibility rerun."""

    root = config.report_path
    root.mkdir(parents=True, exist_ok=True)
    path = root / "00_pre_open_seal.json"
    current = build_seal_payload(config)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("seal_sha256") != current.get("seal_sha256"):
            raise RuntimeError(
                "2026 holdout was previously sealed under different code/config/source hashes; "
                "post-open mutation is forbidden"
            )
        return existing
    path.write_text(json.dumps(current, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return current


def verify_post_run_seal(config: SealedHoldoutConfig, pre_open: dict[str, Any]) -> dict[str, Any]:
    current = build_seal_payload(config)
    passed = current.get("seal_sha256") == pre_open.get("seal_sha256")
    return {
        "status": "PASS" if passed else "FAIL",
        "pre_open_seal_sha256": pre_open.get("seal_sha256"),
        "post_run_seal_sha256": current.get("seal_sha256"),
        "unchanged": bool(passed),
    }
