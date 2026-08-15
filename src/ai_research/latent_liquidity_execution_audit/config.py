#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R01.2 stable-path explanation and execution audit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.ai_research.config import PROJECT_ROOT

MODEL_NAME = "ETH Latent Liquidity Pool Path Learning V1"
STAGE_ID = "R01.2"
STAGE_NAME = "Stable path explanation and executable-confirmation audit"


@dataclass(frozen=True)
class StablePathExecutionAuditConfig:
    symbol: str = "ETH-USDT-SWAP"
    source_report_dir: str = (
        "data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "01_1_liquidity_first_path_atlas"
    )
    report_dir: str = (
        "data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "01_2_stable_path_execution_audit"
    )
    cache_dir: str = (
        "data/cache/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "r01_2_stable_path_execution_audit"
    )
    source_feature_file: str = "12_feature_table.csv.gz"
    source_label_file: str = "13_label_table.csv.gz"
    source_assignment_file: str = "14_cluster_assignment.csv.gz"
    source_manifest_file: str = "00_manifest.json"
    source_causal_audit_file: str = "09_causal_audit.csv"
    csv_read_chunk_rows: int = 20_000
    # These clusters are frozen from the completed R01.1 review.  They are
    # diagnostic discovery labels, not live trading rules.
    target_clusters: tuple[int, ...] = (10, 4, 5, 8)
    target_cluster_roles: tuple[tuple[int, str], ...] = (
        (10, "CORE_REVERSAL_DISCOVERY"),
        (4, "STRONG_REVERSAL_DISCOVERY"),
        (5, "RARE_HIGH_CONVICTION_DISCOVERY"),
        (8, "CONTINUATION_CONTROL"),
    )
    profile_sample_per_stratum: int = 2_000
    replay_sample_per_stratum: int = 120
    random_state: int = 20260805
    bootstrap_repetitions: int = 1_000
    pre_replay_seconds: int = 300
    post_replay_seconds: int = 600
    # Match R01.1 second-bar normalization: short no-trade gaps are filled
    # causally, while longer gaps remain unsafe and are excluded.
    replay_max_fill_gap_seconds: int = 5
    max_confirmation_seconds: int = 300
    stabilization_seconds: int = 15
    reclaim_thresholds_bp: tuple[float, ...] = (5.0, 10.0, 15.0)
    second_push_rebound_bp: float = 5.0
    second_push_retest_tolerance_bp: float = 3.0
    second_push_new_extreme_tolerance_bp: float = 2.0
    structural_stop_buffer_bp: float = 3.0
    roundtrip_cost_bp: float = 11.0
    cost_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)
    entry_delay_seconds: tuple[int, ...] = (1, 3, 5)
    terminal_horizons_seconds: tuple[int, ...] = (60, 180, 300)
    minimum_period_episodes: int = 100
    # Source periods are frozen by R01.1.
    periods: tuple[str, ...] = (
        "TRAIN_2023_2024",
        "VALIDATION_2025Q1_Q3",
        "HOLDOUT_2025Q4_2026H1",
    )

    def validate(self) -> None:
        if not self.target_clusters:
            raise ValueError("target_clusters cannot be empty")
        if len(set(self.target_clusters)) != len(self.target_clusters):
            raise ValueError("target_clusters must be unique")
        role_ids = tuple(cluster for cluster, _ in self.target_cluster_roles)
        if set(role_ids) != set(self.target_clusters):
            raise ValueError("target_cluster_roles must cover target_clusters exactly")
        if self.csv_read_chunk_rows < 1_000:
            raise ValueError("csv_read_chunk_rows is too small")
        if self.profile_sample_per_stratum < 100:
            raise ValueError("profile sample is too small")
        if self.replay_sample_per_stratum < 20:
            raise ValueError("replay sample is too small")
        if self.bootstrap_repetitions < 200:
            raise ValueError("bootstrap_repetitions is too small")
        if self.pre_replay_seconds < 60 or self.post_replay_seconds < 300:
            raise ValueError("replay windows are too short")
        if self.replay_max_fill_gap_seconds < 1:
            raise ValueError("replay_max_fill_gap_seconds must be positive")
        if self.max_confirmation_seconds >= self.post_replay_seconds:
            raise ValueError("confirmation window must leave a post-entry path")
        if any(value <= 0 for value in self.reclaim_thresholds_bp):
            raise ValueError("reclaim thresholds must be positive")
        if tuple(sorted(set(self.reclaim_thresholds_bp))) != self.reclaim_thresholds_bp:
            raise ValueError("reclaim thresholds must be unique and increasing")
        if any(delay < 1 for delay in self.entry_delay_seconds):
            raise ValueError("entry must be next-second or later")
        if self.roundtrip_cost_bp <= 0:
            raise ValueError("roundtrip cost must be positive")
        if "eth_latent_liquidity_path_v1" not in self.report_dir:
            raise ValueError("report directory must stay inside the model namespace")

    @property
    def source_report_path(self) -> Path:
        self.validate()
        path = Path(self.source_report_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def report_path(self) -> Path:
        self.validate()
        path = Path(self.report_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def cache_path(self) -> Path:
        self.validate()
        path = Path(self.cache_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def role_for_cluster(self, cluster: int) -> str:
        return dict(self.target_cluster_roles)[int(cluster)]

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["target_clusters"] = list(self.target_clusters)
        payload["target_cluster_roles"] = [list(item) for item in self.target_cluster_roles]
        payload["reclaim_thresholds_bp"] = list(self.reclaim_thresholds_bp)
        payload["cost_multipliers"] = list(self.cost_multipliers)
        payload["entry_delay_seconds"] = list(self.entry_delay_seconds)
        payload["terminal_horizons_seconds"] = list(self.terminal_horizons_seconds)
        payload["periods"] = list(self.periods)
        return payload


DEFAULT_CONFIG = StablePathExecutionAuditConfig()
