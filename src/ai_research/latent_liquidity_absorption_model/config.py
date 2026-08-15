#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R01.3 absorption-completion and remaining-space learning."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.ai_research.config import PROJECT_ROOT

MODEL_NAME = "ETH Latent Liquidity Pool Path Learning V1"
STAGE_ID = "R01.3"
STAGE_NAME = "Absorption completion and remaining-space supervised audit"


@dataclass(frozen=True)
class AbsorptionModelConfig:
    symbol: str = "ETH-USDT-SWAP"
    source_report_dir: str = (
        "data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "01_1_liquidity_first_path_atlas"
    )
    report_dir: str = (
        "data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "01_3_absorption_remaining_space_model"
    )
    cache_dir: str = (
        "data/cache/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "r01_3_absorption_remaining_space_model"
    )
    source_feature_file: str = "12_feature_table.csv.gz"
    source_label_file: str = "13_label_table.csv.gz"
    source_assignment_file: str = "14_cluster_assignment.csv.gz"
    source_manifest_file: str = "00_manifest.json"
    source_causal_audit_file: str = "09_causal_audit.csv"
    csv_read_chunk_rows: int = 20_000
    target_clusters: tuple[int, ...] = (10, 4, 5, 8)
    target_cluster_roles: tuple[tuple[int, str], ...] = (
        (10, "CORE_REVERSAL_DISCOVERY"),
        (4, "STRONG_REVERSAL_DISCOVERY"),
        (5, "RARE_HIGH_CONVICTION_DISCOVERY"),
        (8, "CONTINUATION_CONTROL"),
    )
    # Deterministic sample from every cluster x side x period stratum.  This is
    # deliberately bounded so the 1-second replay remains practical.
    replay_sample_per_stratum: int = 400
    profile_sample_per_stratum: int = 500
    random_state: int = 20260806
    pre_replay_seconds: int = 300
    post_replay_seconds: int = 660
    replay_max_fill_gap_seconds: int = 5
    decision_offsets_seconds: tuple[int, ...] = (15, 30, 45, 60, 90, 120, 180, 240, 300)
    recent_windows_seconds: tuple[int, ...] = (5, 15, 30, 60)
    label_horizon_seconds: int = 300
    absorption_lookahead_seconds: int = 30
    absorption_extension_tolerance_bp: float = 3.0
    structural_stop_buffer_bp: float = 3.0
    roundtrip_cost_bp: float = 11.0
    minimum_net_room_bp: float = 15.0
    cost_multipliers: tuple[float, ...] = (1.0, 2.0, 3.0)
    entry_delay_seconds: tuple[int, ...] = (1, 3, 5)
    selection_quantile: float = 0.90
    model_n_estimators: int = 320
    model_learning_rate: float = 0.035
    model_num_leaves: int = 31
    model_min_child_samples: int = 60
    minimum_train_rows: int = 2_000
    minimum_class_rows: int = 200
    train_period: str = "TRAIN_2023_2024"
    calibration_period: str = "VALIDATION_2025Q1_Q3"
    holdout_period: str = "HOLDOUT_2025Q4_2026H1"
    periods: tuple[str, ...] = (
        "TRAIN_2023_2024",
        "VALIDATION_2025Q1_Q3",
        "HOLDOUT_2025Q4_2026H1",
    )

    def validate(self) -> None:
        if not self.target_clusters or len(set(self.target_clusters)) != len(self.target_clusters):
            raise ValueError("target_clusters must be non-empty and unique")
        if set(dict(self.target_cluster_roles)) != set(self.target_clusters):
            raise ValueError("target_cluster_roles must cover target_clusters")
        if self.replay_sample_per_stratum < 50:
            raise ValueError("replay_sample_per_stratum is too small")
        if self.pre_replay_seconds < 60 or self.post_replay_seconds < 600:
            raise ValueError("replay windows are too short")
        if tuple(sorted(set(self.decision_offsets_seconds))) != self.decision_offsets_seconds:
            raise ValueError("decision offsets must be unique and increasing")
        if max(self.decision_offsets_seconds) + max(self.entry_delay_seconds) + self.label_horizon_seconds > self.post_replay_seconds:
            raise ValueError("decision offset plus label horizon exceeds replay path")
        if self.absorption_lookahead_seconds < 5:
            raise ValueError("absorption lookahead is too short")
        if not 0.5 < self.selection_quantile < 1.0:
            raise ValueError("selection_quantile must be in (0.5, 1)")
        if self.roundtrip_cost_bp <= 0 or self.minimum_net_room_bp <= 0:
            raise ValueError("cost and net-room threshold must be positive")
        if any(delay < 1 for delay in self.entry_delay_seconds):
            raise ValueError("entry must be next-second or later")
        if self.train_period not in self.periods or self.calibration_period not in self.periods or self.holdout_period not in self.periods:
            raise ValueError("train/calibration/holdout period must be declared")
        if "eth_latent_liquidity_path_v1" not in self.report_dir:
            raise ValueError("report directory must remain inside model namespace")

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

    @property
    def gross_room_threshold_bp(self) -> float:
        return float(self.roundtrip_cost_bp + self.minimum_net_room_bp)

    def role_for_cluster(self, cluster: int) -> str:
        return dict(self.target_cluster_roles)[int(cluster)]

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        for key in (
            "target_clusters", "target_cluster_roles", "decision_offsets_seconds",
            "recent_windows_seconds", "cost_multipliers", "entry_delay_seconds", "periods",
        ):
            payload[key] = [list(item) if isinstance(item, tuple) else item for item in payload[key]]
        return payload


DEFAULT_CONFIG = AbsorptionModelConfig()
