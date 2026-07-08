#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Configuration builders for the ETH LF V10B sleeve."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from src.edge_lib.lf_bear_short.config import PRESETS as BEAR_PRESETS
from src.edge_lib.lf_bear_short.config import BearConfig
from src.edge_lib.lf_bull_range_reclaim.config import PRESETS as BULL_PRESETS
from src.edge_lib.lf_bull_range_reclaim.config import BullRangeConfig
from src.edge_lib.lf_momentum_breakout.config import PRESETS as MOMENTUM_PRESETS
from src.edge_lib.lf_momentum_breakout.config import MomentumConfig
from src.portfolio_common.allocator import StrategyConfig

SLEEVE_ID = "ETH_SLEEVE_LF_V10B"

PRIORITY_MODES: dict[str, list[str]] = {
    "v8": ["MOMENTUM_V3", "BEAR_V3_ONLY", "BULL_RECLAIM_V2"],
    "reclaim_first": ["BULL_RECLAIM_V2", "MOMENTUM_V3", "BEAR_V3_ONLY"],
    "reclaim_bear_second": ["BULL_RECLAIM_V2", "BEAR_V3_ONLY", "MOMENTUM_V3"],
}


def priority_map(mode: str) -> dict[str, int]:
    order = PRIORITY_MODES[mode]
    return {engine: int((len(order) - i) * 50) for i, engine in enumerate(order)}


def build_lf_args(args: Any) -> SimpleNamespace:
    slippage = float(getattr(args, "slippage_pct", getattr(args, "slippage", 0.0002)))
    return SimpleNamespace(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        warmup_start_date=args.warmup_start_date,
        warmup_days=365,
        initial_capital=float(args.initial_capital),
        preset=getattr(args, "lf_preset", "turbo"),
        unit_risk_per_trade=None,
        max_total_notional_mult=None,
        max_units=None,
        min_risk_mult=0.35,
        max_risk_mult=None,
        fee_rate=float(getattr(args, "fee_rate", 0.00055)),
        slippage_pct=slippage,
        disable_short=False,
        bear_preset=getattr(args, "lf_bear_preset", "high"),
        bear_min_risk_mult=0.25,
        bear_standalone_risk_scale=1.0,
        bear_standalone_quality_scale=1.0,
        disable_bear_standalone=False,
        bull_preset=getattr(args, "lf_bull_preset", "high"),
        bull_min_risk_mult=0.25,
        bull_reclaim_risk_scale=1.0,
        bull_reclaim_quality_scale=1.0,
        bull_execution_mode="inherit",
        disable_bull_reclaim=False,
        priority_mode=getattr(args, "lf_priority_mode", "reclaim_first"),
        global_risk_scale=float(getattr(args, "lf_global_risk_scale", 1.30)),
        quality_mult_cap=2.20,
        micro_filter_mode=getattr(args, "lf_micro_filter_mode", "soft"),
        range_pct=0.002,
        price_step=1.0,
        range_data_dir=None,
        disable_footprint_context=False,
        micro_min_range_bars=5,
        micro_contra_imbalance=0.05,
        micro_aligned_imbalance=0.05,
        micro_bad_close_pos=0.35,
        micro_good_close_pos=0.65,
        micro_contra_risk_scale=0.50,
        micro_not_aligned_risk_scale=0.50,
        range_exit_mode="soft",
        range_exit_min_mfe_r=2.0,
        range_exit_giveback_frac=0.65,
        range_exit_min_hold_bars=2,
        range_exit_delay_bars=0,
        range_exit_contra_imbalance=0.05,
        range_exit_bad_close_pos=0.35,
        range_exit_require_reversal=True,
        disable_momentum_long_not_aligned_block=False,
        disable_momentum_short_fast_speed_block=False,
        rf_speed_rolling_window_bars=1080,
        rf_speed_min_periods=100,
        rf_speed_fast_quantile=0.75,
        out_dir=None,
    )


def make_momentum_config(args: Any) -> MomentumConfig:
    preset = MOMENTUM_PRESETS[str(args.preset)]
    return MomentumConfig(
        symbol=args.symbol,
        initial_capital=float(args.initial_capital),
        unit_risk_per_trade=float(args.unit_risk_per_trade if args.unit_risk_per_trade is not None else preset["unit_risk_per_trade"]),
        max_total_notional_mult=float(args.max_total_notional_mult if args.max_total_notional_mult is not None else preset["max_total_notional_mult"]),
        max_units=int(args.max_units if args.max_units is not None else preset["max_units"]),
        min_risk_mult=float(args.min_risk_mult),
        max_risk_mult=float(args.max_risk_mult if args.max_risk_mult is not None else preset["max_risk_mult"]),
        fee_rate=float(args.fee_rate),
        slippage_pct=float(args.slippage_pct),
        enable_short=not bool(args.disable_short),
    )


def make_bear_config(args: Any) -> BearConfig:
    preset = BEAR_PRESETS[str(args.bear_preset)]
    return BearConfig(
        symbol=args.symbol,
        initial_capital=float(args.initial_capital),
        unit_risk_per_trade=float(preset["unit_risk_per_trade"]),
        max_total_notional_mult=float(preset["max_total_notional_mult"]),
        max_units=int(preset["max_units"]),
        min_risk_mult=float(args.bear_min_risk_mult),
        max_risk_mult=float(preset["max_risk_mult"]),
        fee_rate=float(args.fee_rate),
        slippage_pct=float(args.slippage_pct),
        style=str(preset["style"]),
    )


def make_bull_config(args: Any) -> BullRangeConfig:
    preset = BULL_PRESETS[str(args.bull_preset)]
    return BullRangeConfig(
        symbol=args.symbol,
        initial_capital=float(args.initial_capital),
        unit_risk_per_trade=float(preset["unit_risk_per_trade"]),
        max_total_notional_mult=float(preset["max_total_notional_mult"]),
        max_units=int(preset["max_units"]),
        min_risk_mult=float(args.bull_min_risk_mult),
        max_risk_mult=float(preset["max_risk_mult"]),
        fee_rate=float(args.fee_rate),
        slippage_pct=float(args.slippage_pct),
    )


def make_exec_config(cfg: MomentumConfig) -> StrategyConfig:
    return StrategyConfig(
        symbol=cfg.symbol,
        initial_capital=cfg.initial_capital,
        unit_risk_per_trade=cfg.unit_risk_per_trade,
        max_total_notional_mult=cfg.max_total_notional_mult,
        max_units=cfg.max_units,
        min_risk_mult=cfg.min_risk_mult,
        max_risk_mult=cfg.max_risk_mult,
        fee_rate=cfg.fee_rate,
        slippage_pct=cfg.slippage_pct,
        enable_short=cfg.enable_short,
        initial_atr_mult=cfg.initial_atr_mult,
        trailing_atr_mult=cfg.trailing_atr_mult,
        add_every_r=cfg.add_every_r,
        max_hold_bars=cfg.max_hold_bars,
        cooldown_bars=cfg.cooldown_bars,
        breakeven_after_r=cfg.breakeven_after_r,
        breakeven_lock_r=cfg.breakeven_lock_r,
        lock_after_2r=cfg.lock_after_2r,
        lock_2r=cfg.lock_2r,
        lock_after_3r=cfg.lock_after_3r,
        lock_3r=cfg.lock_3r,
        no_progress_bars=10000,
    )


def bull_to_exec_config(cfg: BullRangeConfig) -> StrategyConfig:
    return StrategyConfig(
        symbol=cfg.symbol,
        initial_capital=cfg.initial_capital,
        unit_risk_per_trade=cfg.unit_risk_per_trade,
        max_total_notional_mult=cfg.max_total_notional_mult,
        max_units=cfg.max_units,
        min_risk_mult=cfg.min_risk_mult,
        max_risk_mult=cfg.max_risk_mult,
        fee_rate=cfg.fee_rate,
        slippage_pct=cfg.slippage_pct,
        enable_short=cfg.enable_short,
        initial_atr_mult=cfg.initial_atr_mult,
        trailing_atr_mult=cfg.trailing_atr_mult,
        add_every_r=cfg.add_every_r,
        max_hold_bars=cfg.max_hold_bars,
        cooldown_bars=cfg.cooldown_bars,
        breakeven_after_r=cfg.breakeven_after_r,
        breakeven_lock_r=cfg.breakeven_lock_r,
        lock_after_2r=cfg.lock_after_2r,
        lock_2r=cfg.lock_2r,
        lock_after_3r=cfg.lock_after_3r,
        lock_3r=cfg.lock_3r,
        no_progress_bars=cfg.no_progress_bars,
        no_progress_min_r=cfg.no_progress_min_r,
    )

