#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for the liquidity-first latent-liquidity path atlas.

R01.1 keeps Swing information only as one supplementary structural family.  The
primary discovery space is broad liquidity accumulation/release: price path,
turnover, trade intensity, delta, impact efficiency, compression, residency and
acceptance.  No Swing is an event-admission requirement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.config import PROJECT_ROOT

STAGE_ID = "R01.1"
STAGE_NAME = "Liquidity-first release and reversal path atlas"
MODEL_NAME = "ETH Latent Liquidity Pool Path Learning V1"


@dataclass(frozen=True)
class LatentLiquidityPathAtlasConfig:
    symbol: str = "ETH-USDT-SWAP"
    warmup_start: str = "2022-01-01 00:00:00"
    research_start: str = "2023-01-01 00:00:00"
    research_end: str = "2026-06-30 23:59:59"
    micro_timeframe: str = "1s"
    chunk_days: int = 2
    pre_context_seconds: int = 3600
    post_label_seconds: int = 600
    debounce_seconds: int = 12
    release_episode_gap_seconds: int = 45
    baseline_seconds: int = 300
    path_windows_seconds: tuple[int, ...] = (5, 15, 30, 60, 180, 300, 900, 3600)
    macro_windows_minutes: tuple[int, ...] = (15, 60, 240, 1440, 4320, 10080)
    macro_context_minutes: int = 11520
    # Only actual 15m+ Swing/Pivot structures are admitted as supplementary
    # features.  Every confirmed level remains in the inventory until swept.
    swing_timeframes: tuple[tuple[str, int], ...] = (
        ("15m", 15),
        ("30m", 30),
        ("1H", 60),
        ("4H", 240),
        ("1D", 1440),
    )
    swing_confirmation_order: int = 1
    swing_sweep_epsilon_bp: float = 0.01
    swing_confluence_bp: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0)
    swing_cache_dir: str = "data/cache/research/eth_ai_trading/eth_latent_liquidity_path_v1"
    chunk_cache_dir: str = "data/cache/research/eth_ai_trading/eth_latent_liquidity_path_v1/r01_1_chunks"
    event_score_threshold: float = 3.0
    boundary_score_threshold: float = 1.5
    candidate_cap: int = 0
    cluster_count: int = 12
    cluster_train_end: str = "2024-12-31 23:59:59"
    minimum_cluster_rows: int = 1200
    cluster_train_sample_cap: int = 250_000
    cluster_assign_batch_rows: int = 50_000
    descriptive_sample_cap: int = 250_000
    csv_write_chunk_rows: int = 50_000
    random_state: int = 20260804
    max_fill_gap_seconds: int = 5
    shallow_extension_bp: float = 8.0
    immediate_reversal_bp: float = 20.0
    immediate_reversal_seconds: int = 30
    extended_min_extension_bp: float = 8.0
    extended_reversal_bp: float = 25.0
    extended_reversal_seconds: int = 300
    stabilization_seconds: int = 15
    continuation_extension_bp: float = 20.0
    continuation_acceptance_fraction: float = 2.0 / 3.0
    report_dir: str = "data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/01_1_liquidity_first_path_atlas"

    def validate(self) -> None:
        if self.micro_timeframe != "1s":
            raise ValueError("R01.1 freezes 1s Trade Bar as the primary micro path")
        if pd.Timestamp(self.warmup_start) >= pd.Timestamp(self.research_start):
            raise ValueError("warmup_start must precede research_start")
        if pd.Timestamp(self.research_start) >= pd.Timestamp(self.research_end):
            raise ValueError("research_start must precede research_end")
        if self.pre_context_seconds < max(self.path_windows_seconds):
            raise ValueError("pre_context_seconds must cover every path window")
        if self.post_label_seconds < self.extended_reversal_seconds:
            raise ValueError("post_label_seconds must cover the extended reversal label")
        if tuple(sorted(set(self.path_windows_seconds))) != self.path_windows_seconds:
            raise ValueError("path windows must be unique and increasing")
        if tuple(sorted(set(self.macro_windows_minutes))) != self.macro_windows_minutes:
            raise ValueError("macro windows must be unique and increasing")
        if self.macro_context_minutes < max(self.macro_windows_minutes):
            raise ValueError("macro context must cover every macro path window")
        if not self.swing_timeframes:
            raise ValueError("at least one 15m+ swing timeframe is required")
        labels = [str(name) for name, _ in self.swing_timeframes]
        minutes = [int(value) for _, value in self.swing_timeframes]
        if len(labels) != len(set(labels)) or len(minutes) != len(set(minutes)):
            raise ValueError("swing timeframes must be unique")
        if any(value < 15 for value in minutes):
            raise ValueError("second/minute micro Swings are forbidden; minimum Swing timeframe is 15m")
        if minutes != sorted(minutes):
            raise ValueError("swing timeframes must be increasing")
        if self.swing_confirmation_order < 1:
            raise ValueError("swing_confirmation_order must be positive")
        if self.swing_sweep_epsilon_bp < 0:
            raise ValueError("swing_sweep_epsilon_bp cannot be negative")
        if tuple(sorted(set(self.swing_confluence_bp))) != self.swing_confluence_bp:
            raise ValueError("swing confluence bands must be unique and increasing")
        if self.release_episode_gap_seconds < self.debounce_seconds:
            raise ValueError("release episode gap must not be shorter than debounce")
        if self.cluster_count < 6 or self.cluster_count > 24:
            raise ValueError("cluster_count must remain broad but interpretable")
        if self.minimum_cluster_rows < self.cluster_count * 20:
            raise ValueError("minimum_cluster_rows is too small")
        if self.cluster_train_sample_cap < self.minimum_cluster_rows:
            raise ValueError("cluster_train_sample_cap must cover minimum_cluster_rows")
        if self.cluster_assign_batch_rows < 100:
            raise ValueError("cluster_assign_batch_rows is too small")
        if self.descriptive_sample_cap < 100:
            raise ValueError("descriptive_sample_cap is too small")
        if self.csv_write_chunk_rows < 100:
            raise ValueError("csv_write_chunk_rows is too small")
        if self.event_score_threshold <= self.boundary_score_threshold:
            raise ValueError("event score threshold must exceed boundary-only threshold")
        if self.max_fill_gap_seconds < 1:
            raise ValueError("max_fill_gap_seconds must be positive")
        if "eth_latent_liquidity_path_v1" not in self.report_dir:
            raise ValueError("report path must be isolated to the new model")

    @property
    def report_path(self) -> Path:
        self.validate()
        return PROJECT_ROOT / self.report_dir

    @property
    def swing_cache_path(self) -> Path:
        self.validate()
        end = pd.Timestamp(self.research_end).strftime("%Y%m%d")
        symbol = self.symbol.replace("-", "_")
        return PROJECT_ROOT / self.swing_cache_dir / f"{symbol}_unswept_swing_lifecycle_to_{end}.csv.gz"

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        payload["path_windows_seconds"] = list(self.path_windows_seconds)
        payload["macro_windows_minutes"] = list(self.macro_windows_minutes)
        payload["swing_timeframes"] = [[name, minutes] for name, minutes in self.swing_timeframes]
        payload["swing_confluence_bp"] = list(self.swing_confluence_bp)
        return payload


DEFAULT_CONFIG = LatentLiquidityPathAtlasConfig()
