#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for the latent-liquidity economic-ceiling audit."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.ai_research.config import PROJECT_ROOT

MODEL_NAME = "ETH Latent Liquidity Pool Path Learning V1"
STAGE_ID = "R02.4"
STAGE_NAME = "Economic ceiling audit"


@dataclass(frozen=True)
class EconomicCeilingConfig:
    symbol: str = "ETH-USDT-SWAP"
    source_report_dir: str = (
        "data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "01_1_liquidity_first_path_atlas"
    )
    report_dir: str = (
        "data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "02_4_economic_ceiling_audit"
    )
    source_feature_file: str = "12_feature_table.csv.gz"
    source_label_file: str = "13_label_table.csv.gz"
    source_assignment_file: str = "14_cluster_assignment.csv.gz"
    source_manifest_file: str = "00_manifest.json"
    source_causal_audit_file: str = "09_causal_audit.csv"
    csv_read_chunk_rows: int = 50_000
    horizons_seconds: tuple[int, ...] = (60, 180, 300, 600)
    primary_horizon_seconds: int = 300
    stop_buffer_bp: float = 3.0
    reward_risk_targets: tuple[float, ...] = (1.0, 1.5, 2.0)
    primary_reward_risk: float = 1.5
    # Project baseline is 11bp round trip. 6/8bp are optimistic passive-cost
    # diagnostics; 22/33bp are mandatory 2x/3x stress diagnostics.
    cost_scenarios_bp: tuple[float, ...] = (6.0, 8.0, 11.0, 22.0, 33.0)
    primary_cost_bp: float = 11.0
    stress_cost_bp: float = 22.0
    frozen_reversal_clusters: tuple[int, ...] = (10, 4, 5)
    continuation_control_cluster: int = 8
    periods: tuple[str, ...] = (
        "TRAIN_2023_2024",
        "VALIDATION_2025Q1_Q3",
        "HOLDOUT_2025Q4_2026H1",
    )
    validation_period: str = "VALIDATION_2025Q1_Q3"
    holdout_period: str = "HOLDOUT_2025Q4_2026H1"
    minimum_oracle_episodes_per_period: int = 100
    gate_min_base_mean_net_bp: float = 10.0
    gate_min_base_profit_factor: float = 1.50
    gate_min_stress_mean_net_bp: float = 0.0
    gate_min_stress_profit_factor: float = 1.00
    gate_min_top10_removed_mean_net_bp: float = 0.0
    gate_min_base_positive_mfe_rate: float = 0.65

    def validate(self) -> None:
        if self.primary_horizon_seconds not in self.horizons_seconds:
            raise ValueError("primary horizon must be frozen in horizons_seconds")
        if tuple(sorted(set(self.horizons_seconds))) != self.horizons_seconds:
            raise ValueError("horizons_seconds must be unique and increasing")
        if self.primary_reward_risk not in self.reward_risk_targets:
            raise ValueError("primary reward/risk must be in reward_risk_targets")
        if self.primary_cost_bp not in self.cost_scenarios_bp or self.stress_cost_bp not in self.cost_scenarios_bp:
            raise ValueError("primary/stress cost must be frozen in cost_scenarios_bp")
        if self.primary_cost_bp <= 0 or self.stress_cost_bp <= self.primary_cost_bp:
            raise ValueError("cost gates are invalid")
        if self.stop_buffer_bp <= 0:
            raise ValueError("stop buffer must be positive")
        if not self.frozen_reversal_clusters:
            raise ValueError("frozen reversal cluster set cannot be empty")
        if self.continuation_control_cluster in self.frozen_reversal_clusters:
            raise ValueError("continuation control cannot be a reversal cluster")
        if self.validation_period not in self.periods or self.holdout_period not in self.periods:
            raise ValueError("period gates are inconsistent")
        if self.csv_read_chunk_rows < 1_000:
            raise ValueError("csv chunk size is too small")
        if "eth_latent_liquidity_path_v1" not in self.report_dir:
            raise ValueError("report must remain inside the model namespace")

    @property
    def source_report_path(self) -> Path:
        self.validate()
        p = Path(self.source_report_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def report_path(self) -> Path:
        self.validate()
        p = Path(self.report_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        for key in ("horizons_seconds", "reward_risk_targets", "cost_scenarios_bp", "frozen_reversal_clusters", "periods"):
            payload[key] = list(payload[key])
        return payload


DEFAULT_CONFIG = EconomicCeilingConfig()
