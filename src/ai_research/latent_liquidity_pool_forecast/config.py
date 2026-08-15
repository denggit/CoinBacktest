#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R02 pre-event latent liquidity-pool forecasting."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.ai_research.config import PROJECT_ROOT

MODEL_NAME = "ETH Latent Liquidity Pool Path Learning V1"
STAGE_ID = "R02"
STAGE_NAME = "Pre-event latent liquidity-pool location and sweep-depth forecast"


@dataclass(frozen=True)
class LatentLiquidityPoolForecastConfig:
    symbol: str = "ETH-USDT-SWAP"
    source_report_dir: str = (
        "data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "01_1_liquidity_first_path_atlas"
    )
    report_dir: str = (
        "data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "02_latent_pool_location_depth_model"
    )
    cache_dir: str = (
        "data/cache/research/eth_ai_trading/eth_latent_liquidity_path_v1/"
        "r02_latent_pool_location_depth_model"
    )
    source_feature_file: str = "12_feature_table.csv.gz"
    source_label_file: str = "13_label_table.csv.gz"
    source_assignment_file: str = "14_cluster_assignment.csv.gz"
    source_manifest_file: str = "00_manifest.json"
    source_causal_audit_file: str = "09_causal_audit.csv"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2026-06-30 23:59:59"
    snapshot_interval_minutes: int = 15
    chunk_days: int = 14
    micro_context_seconds: int = 900
    micro_windows_seconds: tuple[int, ...] = (15, 60, 300, 900)
    macro_context_minutes: int = 10080
    macro_windows_minutes: tuple[int, ...] = (15, 60, 240, 1440, 4320, 10080)
    zone_offsets_bp: tuple[float, ...] = tuple(float(x) for x in range(10, 500, 20))
    zone_half_width_bp: float = 10.0
    touch_horizons_minutes: tuple[int, ...] = (60, 240, 720)
    primary_horizon_minutes: int = 720
    csv_read_chunk_rows: int = 50_000
    touched_control_keep_fraction: float = 0.25
    untouched_control_keep_fraction: float = 0.05
    full_lattice_audit_group_fraction: float = 0.05
    model_train_cap_rows: int = 700_000
    model_eval_cap_rows_per_period: int = 500_000
    random_state: int = 20260807
    model_n_estimators: int = 360
    model_learning_rate: float = 0.035
    model_num_leaves: int = 31
    model_min_child_samples: int = 100
    selection_quantile: float = 0.90
    train_period: str = "TRAIN_2023_2024"
    calibration_period: str = "VALIDATION_2025Q1_Q3"
    holdout_period: str = "HOLDOUT_2025Q4_2026H1"
    periods: tuple[str, ...] = (
        "TRAIN_2023_2024",
        "VALIDATION_2025Q1_Q3",
        "HOLDOUT_2025Q4_2026H1",
    )
    swing_band_bp: tuple[float, ...] = (5.0, 10.0, 25.0, 50.0)
    minimum_train_rows: int = 10_000
    minimum_class_rows: int = 500
    minimum_top_zone_releases: int = 50

    def validate(self) -> None:
        if self.snapshot_interval_minutes < 5:
            raise ValueError("snapshot cadence is too dense for R02")
        if tuple(sorted(set(self.micro_windows_seconds))) != self.micro_windows_seconds:
            raise ValueError("micro windows must be unique and increasing")
        if tuple(sorted(set(self.macro_windows_minutes))) != self.macro_windows_minutes:
            raise ValueError("macro windows must be unique and increasing")
        if max(self.micro_windows_seconds) > self.micro_context_seconds:
            raise ValueError("micro context must cover all micro windows")
        if max(self.macro_windows_minutes) > self.macro_context_minutes:
            raise ValueError("macro context must cover all macro windows")
        if tuple(sorted(set(self.zone_offsets_bp))) != self.zone_offsets_bp:
            raise ValueError("zone offsets must be unique and increasing")
        if min(self.zone_offsets_bp) <= 0 or max(self.zone_offsets_bp) < 300:
            raise ValueError("zone lattice must cover both near and far pools")
        if self.zone_half_width_bp <= 0:
            raise ValueError("zone width must be positive")
        if self.primary_horizon_minutes not in self.touch_horizons_minutes:
            raise ValueError("primary horizon must be among touch horizons")
        if tuple(sorted(set(self.touch_horizons_minutes))) != self.touch_horizons_minutes:
            raise ValueError("touch horizons must be unique and increasing")
        if not 0 < self.touched_control_keep_fraction <= 1:
            raise ValueError("touched control fraction must be in (0,1]")
        if not 0 < self.untouched_control_keep_fraction <= 1:
            raise ValueError("untouched control fraction must be in (0,1]")
        if not 0 < self.full_lattice_audit_group_fraction <= 1:
            raise ValueError("full-lattice audit group fraction must be in (0,1]")
        left = np.asarray(self.zone_offsets_bp, dtype=float) - float(self.zone_half_width_bp)
        right = np.asarray(self.zone_offsets_bp, dtype=float) + float(self.zone_half_width_bp)
        if left[0] > 0.0 or np.any(left[1:] > right[:-1] + 1e-9):
            raise ValueError("zone lattice must cover the full 0-to-max distance without gaps")
        if not 0.5 < self.selection_quantile < 1.0:
            raise ValueError("selection quantile must be in (0.5,1)")
        if self.train_period not in self.periods or self.holdout_period not in self.periods:
            raise ValueError("frozen periods are inconsistent")
        if self.minimum_top_zone_releases < 1:
            raise ValueError("minimum top-zone release count must be positive")
        if "eth_latent_liquidity_path_v1" not in self.report_dir:
            raise ValueError("R02 report must remain in model namespace")

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

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        for key in (
            "micro_windows_seconds", "macro_windows_minutes", "zone_offsets_bp",
            "touch_horizons_minutes", "periods", "swing_band_bp",
        ):
            payload[key] = list(payload[key])
        return payload


DEFAULT_CONFIG = LatentLiquidityPoolForecastConfig()
