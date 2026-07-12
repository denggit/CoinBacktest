#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH LF Portfolio V9C Reclaim Priority Probe
===========================================

组合定位：
    在 V6 组合基础上新增 range_bar / footprint context 风险过滤层。
    LF 引擎仍然负责方向；range/footprint 只用于在 4H 信号确认时降低逆向订单流信号的 entry risk，
    默认不增加仓位，只尝试降低坏入场风险。

设计原则：
    - 不把 V4B/V8 fallback 硬塞进来；Bull V2 只补 V5 空档，不做叠加收益。
    - 同一时间只允许一个 active position。
    - 不对冲，不双开。
    - 当前 4H close 确认信号，下一根 4H open 执行。
    - 当前 bar close 更新的新 stop 下一根 bar 才生效。
    - 不用年份、月份、日期过滤。
    - range/footprint context 只使用当前 4H 信号 bar 内已经完成的 range bars；4H close 确认后才用于下一根 4H open 的入场风险。
    - 默认 micro-filter-mode=soft：没有得到 micro aligned 确认的 LF 信号会降低入场风险；不放大 aligned 信号。
    V9C 只测试引擎优先级变化，默认把 Bull Reclaim 放第一优先级；其余执行/微观过滤/风险放大保持 V8 思路。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backtest_common.data import load_ohlcv_data as load_data  # noqa: E402
from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage  # noqa: E402
from src.backtest_common.reporting import build_report_trades  # noqa: E402
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader  # noqa: E402
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

from backtest.lf.eth_1d_4h_momentum_breakout_v3_backtest import (  # noqa: E402
    PRESETS as MOMENTUM_PRESETS,
    MomentumConfig,
    build_features as build_momentum_features,
)
from backtest.lf.eth_1d_4h_bear_short_engine_v3_backtest import (  # noqa: E402
    PRESETS as BEAR_PRESETS,
    BearConfig,
    build_bear_features,
)

from backtest.lf.eth_1d_4h_bull_range_reclaim_v2_backtest import (  # noqa: E402
    PRESETS as BULL_PRESETS,
    BullRangeConfig,
    build_features as build_bull_features,
    to_exec_config as bull_to_exec_config,
)
from backtest.lf.eth_1d_4h_trend_rider_v8_position_lock_backtest import (  # noqa: E402
    StrategyConfig as ExecConfig,
    close_trade,
    protected_stop,
    summarize,
    unit_qty,
    weighted_avg_price,
)

STRATEGY_NAME = "eth_lf_portfolio_v9c_reclaim_priority"
REPORT_STRATEGY_NAME = "ETH_LF_Portfolio_V9C_ReclaimPriority"

PRIORITY_MODES: dict[str, list[str]] = {
    # V8/V7B baseline: Momentum first, Bear second, Bull Reclaim fills remaining gaps.
    "v8": ["MOMENTUM_V3", "BEAR_V3_ONLY", "BULL_RECLAIM_V2"],
    # Test requested by user: put Bull Reclaim first.
    "reclaim_first": ["BULL_RECLAIM_V2", "MOMENTUM_V3", "BEAR_V3_ONLY"],
    # Alternative sanity check: Bull Reclaim first, then Bear, then Momentum.
    "reclaim_bear_second": ["BULL_RECLAIM_V2", "BEAR_V3_ONLY", "MOMENTUM_V3"],
}


def priority_map(mode: str) -> dict[str, int]:
    order = PRIORITY_MODES[mode]
    return {engine: int((len(order) - i) * 50) for i, engine in enumerate(order)}


def make_momentum_config(args: argparse.Namespace) -> MomentumConfig:
    preset = MOMENTUM_PRESETS[args.preset]
    return MomentumConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        unit_risk_per_trade=float(args.unit_risk_per_trade if args.unit_risk_per_trade is not None else preset["unit_risk_per_trade"]),
        max_total_notional_mult=float(args.max_total_notional_mult if args.max_total_notional_mult is not None else preset["max_total_notional_mult"]),
        max_units=int(args.max_units if args.max_units is not None else preset["max_units"]),
        min_risk_mult=args.min_risk_mult,
        max_risk_mult=float(args.max_risk_mult if args.max_risk_mult is not None else preset["max_risk_mult"]),
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        enable_short=not args.disable_short,
    )


def make_exec_config(cfg: MomentumConfig) -> ExecConfig:
    return ExecConfig(
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


def make_bear_config(args: argparse.Namespace) -> BearConfig:
    preset = BEAR_PRESETS[args.bear_preset]
    return BearConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        unit_risk_per_trade=float(preset["unit_risk_per_trade"]),
        max_total_notional_mult=float(preset["max_total_notional_mult"]),
        max_units=int(preset["max_units"]),
        min_risk_mult=args.bear_min_risk_mult,
        max_risk_mult=float(preset["max_risk_mult"]),
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        style=str(preset["style"]),
    )


def make_bull_config(args: argparse.Namespace) -> BullRangeConfig:
    preset = BULL_PRESETS[args.bull_preset]
    return BullRangeConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        unit_risk_per_trade=float(preset["unit_risk_per_trade"]),
        max_total_notional_mult=float(preset["max_total_notional_mult"]),
        max_units=int(preset["max_units"]),
        min_risk_mult=args.bull_min_risk_mult,
        max_risk_mult=float(preset["max_risk_mult"]),
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
    )


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype("boolean").fillna(False).astype(bool)



def _ts_text(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class _FootprintMaxBucketStore:
    def __init__(self, symbol: str, range_pct: float, price_step: float, data_dir: str | None = None):
        self.loader = OKXRangeFootprintLoader(symbol=symbol, range_pct=range_pct, price_step=price_step, data_dir=data_dir)
        self.db_path = Path(self.loader.db_path)
        self.table_name = self.loader.table_name

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def load_max_bucket_features(self, start_date: str, end_date: str) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()
        sell_sql = f"""
            SELECT bar_id, price_bucket AS max_sell_bucket,
                   sell_notional AS max_bucket_sell_notional
            FROM (
                SELECT bar_id, price_bucket, sell_notional,
                       ROW_NUMBER() OVER (
                           PARTITION BY bar_id
                           ORDER BY sell_notional DESC, price_bucket ASC
                       ) AS rn
                FROM {self.table_name}
                WHERE end_ts >= ? AND end_ts <= ? AND sell_notional > 0
            ) ranked
            WHERE rn = 1
        """
        buy_sql = f"""
            SELECT bar_id, price_bucket AS max_buy_bucket,
                   buy_notional AS max_bucket_buy_notional
            FROM (
                SELECT bar_id, price_bucket, buy_notional,
                       ROW_NUMBER() OVER (
                           PARTITION BY bar_id
                           ORDER BY buy_notional DESC, price_bucket DESC
                       ) AS rn
                FROM {self.table_name}
                WHERE end_ts >= ? AND end_ts <= ? AND buy_notional > 0
            ) ranked
            WHERE rn = 1
        """
        params = (_ts_text(start_date), _ts_text(end_date))
        with self._connect() as conn:
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_bar_id ON {self.table_name}(bar_id)")
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{self.table_name}_end_ts ON {self.table_name}(end_ts)")
            sell = pd.read_sql_query(sell_sql, conn, params=params)
            buy = pd.read_sql_query(buy_sql, conn, params=params)
        return sell.merge(buy, on="bar_id", how="outer")


def _safe_sum_col(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def load_range_footprint_context(args: argparse.Namespace, start_date: str, end_date: str) -> pd.DataFrame:
    """Aggregate range-bar/orderflow context into 4H buckets.

    Bucket timing: range bars are grouped by end_ts.floor('4h'). For a 4H signal row,
    that bucket is fully known only at the next 4H open, exactly when V6 would enter.
    """
    if args.micro_filter_mode == "off":
        return pd.DataFrame()
    print(
        f"Loading range/footprint context: range_pct={args.range_pct} price_step={args.price_step} {start_date}->{end_date}",
        flush=True,
    )
    loader = OKXRangeBarLoader(symbol=args.symbol, range_pct=args.range_pct, data_dir=args.range_data_dir)
    rb = loader.load_local_data(start_date=start_date, end_date=end_date)
    if rb.empty:
        print("[WARN] No range bars found; micro context disabled.", flush=True)
        return pd.DataFrame()
    rb = rb.reset_index(drop=True).copy()
    rb["end_ts"] = pd.to_datetime(rb["end_ts"])
    rb = rb.sort_values(["end_ts", "bar_id"]).reset_index(drop=True)

    if not args.disable_footprint_context:
        try:
            fp = _FootprintMaxBucketStore(args.symbol, args.range_pct, args.price_step, args.range_data_dir).load_max_bucket_features(start_date, end_date)
            if not fp.empty:
                rb = rb.merge(fp, on="bar_id", how="left")
                print(f"Loaded footprint max-bucket context: {len(fp):,}", flush=True)
        except Exception as exc:
            print(f"[WARN] footprint context unavailable: {exc}", flush=True)

    for col in ["open", "high", "low", "close", "notional", "volume", "buy_notional", "sell_notional", "delta", "delta_notional", "taker_buy_ratio"]:
        if col in rb.columns:
            rb[col] = pd.to_numeric(rb[col], errors="coerce")
    rb["rf_bucket"] = rb["end_ts"].dt.floor("4h")
    rb["_buy_notional"] = _safe_sum_col(rb, ["buy_notional", "buy_volume"])
    rb["_sell_notional"] = _safe_sum_col(rb, ["sell_notional", "sell_volume"])
    if "delta_notional" in rb.columns:
        rb["_delta"] = pd.to_numeric(rb["delta_notional"], errors="coerce").fillna(0.0)
    else:
        rb["_delta"] = pd.to_numeric(rb.get("delta", 0.0), errors="coerce").fillna(0.0)
    rb["_notional"] = _safe_sum_col(rb, ["notional", "volume"])

    grouped_rows: list[dict[str, Any]] = []
    for bucket, g in rb.groupby("rf_bucket", sort=True):
        if g.empty:
            continue
        high = float(g["high"].max())
        low = float(g["low"].min())
        first_open = float(g["open"].iloc[0])
        last_close = float(g["close"].iloc[-1])
        span = max(high - low, 1e-12)
        buy_sum = float(g["_buy_notional"].sum())
        sell_sum = float(g["_sell_notional"].sum())
        delta_sum = float(g["_delta"].sum())
        notional_sum = float(g["_notional"].sum())
        denom = max(buy_sum + sell_sum, 1e-12)
        row = {
            "timestamp": bucket,
            "rf_bar_count": int(len(g)),
            "rf_first_open": first_open,
            "rf_last_close": last_close,
            "rf_high": high,
            "rf_low": low,
            "rf_micro_return_pct": (last_close - first_open) / first_open if first_open > 0 else 0.0,
            "rf_close_pos": (last_close - low) / span,
            "rf_buy_notional_sum": buy_sum,
            "rf_sell_notional_sum": sell_sum,
            "rf_delta_sum": delta_sum,
            "rf_notional_sum": notional_sum,
            "rf_imbalance": (buy_sum - sell_sum) / denom,
            "rf_taker_buy_ratio": buy_sum / denom,
        }
        if "max_bucket_sell_notional" in g.columns:
            row["rf_max_sell_bucket_share"] = float(pd.to_numeric(g["max_bucket_sell_notional"], errors="coerce").fillna(0.0).sum() / max(sell_sum, 1e-12))
        else:
            row["rf_max_sell_bucket_share"] = np.nan
        if "max_bucket_buy_notional" in g.columns:
            row["rf_max_buy_bucket_share"] = float(pd.to_numeric(g["max_bucket_buy_notional"], errors="coerce").fillna(0.0).sum() / max(buy_sum, 1e-12))
        else:
            row["rf_max_buy_bucket_share"] = np.nan
        grouped_rows.append(row)
    ctx = pd.DataFrame(grouped_rows)
    if ctx.empty:
        return ctx
    ctx = ctx.set_index("timestamp").sort_index()
    print(f"Range/footprint 4H context rows: {len(ctx):,} | {ctx.index[0]} -> {ctx.index[-1]}", flush=True)
    return ctx


def apply_micro_context_filter(features: pd.DataFrame, micro_ctx: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = features.copy()
    micro_cols = [
        "rf_bar_count", "rf_micro_return_pct", "rf_close_pos", "rf_delta_sum", "rf_imbalance", "rf_taker_buy_ratio",
        "rf_max_sell_bucket_share", "rf_max_buy_bucket_share",
    ]
    for col in micro_cols:
        out[col] = np.nan
    out["micro_context_available"] = False
    out["micro_aligned"] = False
    out["micro_contra"] = False
    out["micro_entry_risk_scale"] = 1.0
    out["micro_filter_action"] = "OFF" if args.micro_filter_mode == "off" else "NEUTRAL"
    if micro_ctx.empty or args.micro_filter_mode == "off":
        return out

    aligned = micro_ctx.reindex(out.index)
    for col in micro_cols:
        if col in aligned.columns:
            out[col] = aligned[col]
    out["micro_context_available"] = out["rf_bar_count"].fillna(0).astype(float) >= float(args.micro_min_range_bars)

    sig = out["signal"].fillna(0).astype(int)
    long_sig = sig == 1
    short_sig = sig == -1
    has_ctx = out["micro_context_available"]

    long_contra = long_sig & has_ctx & (out["rf_imbalance"] <= -abs(args.micro_contra_imbalance)) & (out["rf_close_pos"] <= args.micro_bad_close_pos)
    short_contra = short_sig & has_ctx & (out["rf_imbalance"] >= abs(args.micro_contra_imbalance)) & (out["rf_close_pos"] >= 1.0 - args.micro_bad_close_pos)
    long_aligned = long_sig & has_ctx & (out["rf_imbalance"] >= abs(args.micro_aligned_imbalance)) & (out["rf_close_pos"] >= args.micro_good_close_pos)
    short_aligned = short_sig & has_ctx & (out["rf_imbalance"] <= -abs(args.micro_aligned_imbalance)) & (out["rf_close_pos"] <= 1.0 - args.micro_good_close_pos)

    out.loc[long_contra | short_contra, "micro_contra"] = True
    out.loc[long_aligned | short_aligned, "micro_aligned"] = True

    signal_active = sig != 0
    not_aligned = signal_active & has_ctx & (~out["micro_aligned"].astype(bool))

    if args.micro_filter_mode == "strict":
        # V7B strict: only take LF signals confirmed by micro aligned context.
        blocked = not_aligned
        out.loc[blocked, "signal"] = 0
        out.loc[blocked, "long_signal"] = False
        out.loc[blocked, "short_signal"] = False
        out.loc[blocked, "micro_entry_risk_scale"] = 0.0
        out.loc[blocked, "micro_filter_action"] = "NOT_ALIGNED_BLOCKED"
    elif args.micro_filter_mode == "soft":
        # V7B default: aligned signals keep full risk; unaligned signals are still tradable but with lower entry risk.
        out.loc[not_aligned, "micro_entry_risk_scale"] = float(args.micro_not_aligned_risk_scale)
        out.loc[not_aligned, "micro_filter_action"] = "NOT_ALIGNED_RISK_REDUCED"
        contra = long_contra | short_contra
        out.loc[contra, "micro_entry_risk_scale"] = float(args.micro_contra_risk_scale)
        out.loc[contra, "micro_filter_action"] = "CONTRA_RISK_REDUCED"
    else:
        raise ValueError(f"Unsupported micro_filter_mode={args.micro_filter_mode}")

    print("Micro context counts:", {
        "available": int(out["micro_context_available"].sum()),
        "aligned_signal": int(out["micro_aligned"].sum()),
        "contra_signal": int(out["micro_contra"].sum()),
        "not_aligned_signal": int(not_aligned.sum()),
        "risk_reduced": int(out["micro_filter_action"].astype(str).str.contains("RISK_REDUCED", na=False).sum()),
        "blocked": int(out["micro_filter_action"].astype(str).str.contains("BLOCKED", na=False).sum()),
    }, flush=True)
    return out


def select_portfolio_signals(momentum: pd.DataFrame, bear: pd.DataFrame, bull: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Select one engine signal per 4H bar according to --priority-mode.

    This is the only intended change for V9C versus V8 logic: signal conflict routing.
    No lookahead is introduced; all three engine feature frames are already built from closed 4H bars,
    and execution remains next 4H open through the SAFE executor.
    """
    out = momentum.copy()
    bear = bear.reindex(out.index)
    bull = bull.reindex(out.index)

    out["momentum_signal"] = out["signal"].fillna(0).astype(int)
    out["bear_signal"] = bear["signal"].fillna(0).astype(int)
    out["bull_signal"] = bull["signal"].fillna(0).astype(int)
    out["momentum_long_exit_channel"] = _bool_col(momentum, "long_exit_channel")
    out["momentum_short_exit_channel"] = _bool_col(momentum, "short_exit_channel")
    out["bear_short_exit_channel"] = _bool_col(bear, "short_exit_channel")
    out["bull_long_exit_channel"] = _bool_col(bull, "long_exit_channel")

    out["selected_engine"] = "NONE"
    out["selected_priority"] = 0
    out["momentum_selected"] = False
    out["bear_only"] = False
    out["bull_reclaim"] = False

    mom_active = out["momentum_signal"] != 0
    bear_active = (out["bear_signal"] == -1) & (not args.disable_bear_standalone)
    bull_active = (out["bull_signal"] == 1) & (not args.disable_bull_reclaim)
    candidate_masks: dict[str, pd.Series] = {
        "MOMENTUM_V3": mom_active,
        "BEAR_V3_ONLY": bear_active,
        "BULL_RECLAIM_V2": bull_active,
    }
    candidate_count = sum(mask.astype(int) for mask in candidate_masks.values())
    out["portfolio_conflict"] = candidate_count > 1

    final_signal = pd.Series(0, index=out.index, dtype="int64")
    priorities = priority_map(args.priority_mode)

    for engine in PRIORITY_MODES[args.priority_mode]:
        mask = candidate_masks[engine] & (final_signal == 0)
        if not bool(mask.any()):
            continue
        out.loc[mask, "selected_engine"] = engine
        out.loc[mask, "selected_priority"] = priorities[engine]
        if engine == "MOMENTUM_V3":
            final_signal.loc[mask] = out.loc[mask, "momentum_signal"]
            out.loc[mask, "momentum_selected"] = True
            # Momentum risk_mult / quality_mult already comes from the momentum frame.
        elif engine == "BEAR_V3_ONLY":
            final_signal.loc[mask] = -1
            out.loc[mask, "bear_only"] = True
            out.loc[mask, "risk_mult"] = (
                bear.loc[mask, "risk_mult"].fillna(1.0) * args.bear_standalone_risk_scale
            ).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
            out.loc[mask, "quality_mult"] = (
                bear.loc[mask, "quality_mult"].fillna(1.0) * args.bear_standalone_quality_scale
            ).clip(0.20, args.quality_mult_cap)
        elif engine == "BULL_RECLAIM_V2":
            final_signal.loc[mask] = 1
            out.loc[mask, "bull_reclaim"] = True
            out.loc[mask, "risk_mult"] = (
                bull.loc[mask, "risk_mult"].fillna(1.0) * args.bull_reclaim_risk_scale
            ).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
            out.loc[mask, "quality_mult"] = (
                bull.loc[mask, "quality_mult"].fillna(1.0) * args.bull_reclaim_quality_scale
            ).clip(0.10, args.quality_mult_cap)

    # Final signal columns for the SAFE executor.
    out["signal"] = final_signal
    out["long_signal"] = out["signal"] == 1
    out["short_signal"] = out["signal"] == -1
    return out

def _entry_exit_channel(row: Any, entry_engine: str, side: int) -> bool:
    if entry_engine.startswith("BEAR") and side == -1:
        return bool(getattr(row, "bear_short_exit_channel", False))
    if entry_engine.startswith("BULL") and side == 1:
        return bool(getattr(row, "bull_long_exit_channel", False))
    return bool(getattr(row, "momentum_long_exit_channel" if side == 1 else "momentum_short_exit_channel", False))


def run_priority_backtest(
    df: pd.DataFrame,
    cfg: ExecConfig,
    engine_cfgs: dict[str, ExecConfig] | None = None,
    *,
    global_risk_scale: float = 1.0,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    capital = cfg.initial_capital
    peak = capital
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    in_pos = False
    side = 0
    entry_i = -1
    entry_time = None
    first_entry = 0.0
    avg_entry = 0.0
    initial_sl = 0.0
    stop_price = 0.0
    risk_per_coin = 0.0
    qty = 0.0
    total_entry_fee = 0.0
    units = 0
    max_fav = 0.0
    max_adv = 0.0
    entry_risk_mult = 1.0
    entry_engine = "NONE"
    pos_cfg = cfg
    engine_cfgs = engine_cfgs or {}
    last_exit_i = -10**9

    rows = list(df.itertuples())
    idx = df.index

    for i in range(len(rows) - 1):
        row = rows[i]
        ts = idx[i]
        if in_pos:
            high = float(row.high)
            low = float(row.low)
            close = float(row.close)
            atr_value = float(row.atr)
            hold_bars = i - entry_i
            active_stop = stop_price
            current_signal = int(getattr(row, "signal", 0))
            active_cfg = pos_cfg

            if side == 1:
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)
                touched_stop = low <= active_stop
                channel_exit = _entry_exit_channel(row, entry_engine, side)
                opposite = current_signal == -1
                next_stop = max(stop_price, close - active_cfg.trailing_atr_mult * atr_value)
                locked = protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, active_cfg)
                if locked is not None:
                    next_stop = max(next_stop, locked)
            else:
                max_fav = min(max_fav, low)
                max_adv = max(max_adv, high)
                touched_stop = high >= active_stop
                channel_exit = _entry_exit_channel(row, entry_engine, side)
                opposite = current_signal == 1
                next_stop = min(stop_price, close + active_cfg.trailing_atr_mult * atr_value)
                locked = protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, active_cfg)
                if locked is not None:
                    next_stop = min(next_stop, locked)

            exit_now = False
            reason = ""
            exit_price = 0.0
            exit_time = ts
            if touched_stop:
                exit_now = True
                exit_price = apply_exit_slippage(active_stop, side, active_cfg.slippage_pct)
                reason = "PROTECTED_TRAILING_STOP"
            elif channel_exit:
                exit_now = True
                exit_price = apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                exit_time = idx[i + 1]
                reason = "DONCHIAN_EXIT_NEXT_OPEN"
            elif opposite:
                exit_now = True
                exit_price = apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                exit_time = idx[i + 1]
                reason = "OPPOSITE_BREAKOUT_NEXT_OPEN"
            elif hold_bars >= active_cfg.max_hold_bars:
                exit_now = True
                exit_price = apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                exit_time = idx[i + 1]
                reason = "MAX_HOLD_EXIT_NEXT_OPEN"

            if exit_now:
                capital = close_trade(
                    trades=trades,
                    capital=capital,
                    side=side,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    first_entry=first_entry,
                    avg_entry=avg_entry,
                    exit_price=exit_price,
                    initial_sl=initial_sl,
                    stop_price=stop_price,
                    qty=qty,
                    units=units,
                    total_entry_fee=total_entry_fee,
                    fee_rate=active_cfg.fee_rate,
                    max_fav=max_fav,
                    max_adv=max_adv,
                    risk_per_coin=risk_per_coin,
                    holding_bars=hold_bars,
                    reason=reason,
                    risk_mult=entry_risk_mult,
                )
                peak = max(peak, capital)
                in_pos = False
                side = 0
                last_exit_i = i
            else:
                stop_price = next_stop

            if in_pos and units < active_cfg.max_units:
                next_unit_number = units + 1
                trigger_r = (next_unit_number - 1) * active_cfg.add_every_r
                add_triggered = high >= first_entry + trigger_r * risk_per_coin if side == 1 else low <= first_entry - trigger_r * risk_per_coin
                if add_triggered:
                    add_price = apply_entry_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                    add_stop_dist = max(active_cfg.initial_atr_mult * atr_value, risk_per_coin)
                    add_q = unit_qty(capital, add_price, add_stop_dist, qty, active_cfg, float(getattr(row, "risk_mult", entry_risk_mult)) * float(getattr(row, "quality_mult", 1.0)) * float(global_risk_scale))
                    if add_q > 0 and math.isfinite(add_q):
                        total_entry_fee += add_q * add_price * active_cfg.fee_rate
                        avg_entry = weighted_avg_price(avg_entry, qty, add_price, add_q)
                        qty += add_q
                        units += 1

        if not in_pos and i - last_exit_i >= cfg.cooldown_bars:
            signal = int(getattr(row, "signal", 0))
            if signal != 0:
                selected_engine = str(getattr(row, "selected_engine", "UNKNOWN"))
                entry_cfg = engine_cfgs.get(selected_engine, cfg)
                next_open = float(rows[i + 1].open)
                entry = apply_entry_slippage(next_open, signal, entry_cfg.slippage_pct)
                atr_value = float(row.atr)
                sl = entry - entry_cfg.initial_atr_mult * atr_value if signal == 1 else entry + entry_cfg.initial_atr_mult * atr_value
                stop_dist = abs(entry - sl)
                entry_risk_mult = (
                    float(getattr(row, "risk_mult", 1.0))
                    * float(getattr(row, "quality_mult", 1.0))
                    * float(getattr(row, "micro_entry_risk_scale", 1.0))
                    * float(global_risk_scale)
                )
                q = unit_qty(capital, entry, stop_dist, 0.0, entry_cfg, entry_risk_mult)
                if q > 0 and math.isfinite(q):
                    in_pos = True
                    side = signal
                    entry_i = i + 1
                    entry_time = idx[i + 1]
                    first_entry = entry
                    avg_entry = entry
                    initial_sl = sl
                    stop_price = sl
                    risk_per_coin = stop_dist
                    qty = q
                    total_entry_fee = qty * entry * entry_cfg.fee_rate
                    units = 1
                    max_fav = entry
                    max_adv = entry
                    entry_engine = selected_engine
                    pos_cfg = entry_cfg

        equity_rows.append({"time": ts, "capital": capital, "drawdown_pct": (peak - capital) / peak if peak > 0 else 0.0})

    if in_pos:
        ts = idx[-1]
        close = float(df.iloc[-1]["close"])
        exit_price = apply_exit_slippage(close, side, pos_cfg.slippage_pct)
        capital = close_trade(
            trades=trades,
            capital=capital,
            side=side,
            entry_time=entry_time,
            exit_time=ts,
            first_entry=first_entry,
            avg_entry=avg_entry,
            exit_price=exit_price,
            initial_sl=initial_sl,
            stop_price=stop_price,
            qty=qty,
            units=units,
            total_entry_fee=total_entry_fee,
            fee_rate=pos_cfg.fee_rate,
            max_fav=max_fav,
            max_adv=max_adv,
            risk_per_coin=risk_per_coin,
            holding_bars=len(df) - 1 - entry_i,
            reason="FORCE_CLOSE_END",
            risk_mult=entry_risk_mult,
        )
    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity


def attach_engine_to_trades(trades: list[dict[str, Any]], features: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for trade in trades:
        item = dict(trade)
        signal_time = pd.Timestamp(item["entry_time"]) - pd.Timedelta(hours=4)
        if signal_time in features.index:
            row = features.loc[signal_time]
            item["engine"] = str(row.get("selected_engine", "UNKNOWN"))
            item["engine_priority"] = int(row.get("selected_priority", 0))
            item["momentum_selected"] = bool(row.get("momentum_selected", False))
            item["bear_only"] = bool(row.get("bear_only", False))
            item["bull_reclaim"] = bool(row.get("bull_reclaim", False))
            item["micro_context_available"] = bool(row.get("micro_context_available", False))
            item["micro_aligned"] = bool(row.get("micro_aligned", False))
            item["micro_contra"] = bool(row.get("micro_contra", False))
            item["micro_entry_risk_scale"] = float(row.get("micro_entry_risk_scale", 1.0))
            item["micro_filter_action"] = str(row.get("micro_filter_action", "NA"))
            item["rf_imbalance"] = float(row.get("rf_imbalance", float("nan")))
            item["rf_close_pos"] = float(row.get("rf_close_pos", float("nan")))
        else:
            item["engine"] = "UNKNOWN"
            item["engine_priority"] = 0
            item["momentum_selected"] = False
            item["bear_only"] = False
            item["bull_reclaim"] = False
            item["micro_context_available"] = False
            item["micro_aligned"] = False
            item["micro_contra"] = False
            item["micro_entry_risk_scale"] = 1.0
            item["micro_filter_action"] = "UNKNOWN"
            item["rf_imbalance"] = float("nan")
            item["rf_close_pos"] = float("nan")
        out.append(item)
    return out


def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{STRATEGY_NAME}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{STRATEGY_NAME}_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx",
        "risk_mult", "quality_mult", "momentum_signal", "bear_signal", "bull_signal", "signal",
        "selected_engine", "selected_priority", "momentum_selected", "bear_only", "bull_reclaim",
        "long_signal", "short_signal",
        "micro_context_available", "micro_aligned", "micro_contra", "micro_entry_risk_scale", "micro_filter_action",
        "rf_bar_count", "rf_micro_return_pct", "rf_close_pos", "rf_delta_sum", "rf_imbalance", "rf_taker_buy_ratio",
        "rf_max_sell_bucket_share", "rf_max_buy_bucket_share",
        "momentum_long_exit_channel", "momentum_short_exit_channel", "bear_short_exit_channel", "bull_long_exit_channel",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_signal_audit.csv")
    with (out_dir / f"{STRATEGY_NAME}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 92)
    print("ETH LF Portfolio V9C Reclaim Priority Probe Backtest Summary")
    print("=" * 92)
    for k, v in summary.items():
        print(f"{k:>34}: {v}")
    print("-" * 92)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 92 + "\n")


def print_deep_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: ExecConfig, out_dir: Path) -> None:
    if not trades or features.empty:
        return
    total_days = max((features.index[-1] - features.index[0]).total_seconds() / 86400.0, 1e-9)
    print_full_report(
        trade_history=build_report_trades(trades),
        df=features,
        initial_capital=cfg.initial_capital,
        capital=float(pd.DataFrame(trades).iloc[-1]["capital"]),
        strategy_name=REPORT_STRATEGY_NAME,
        total_days=total_days,
        ai_enabled=False,
        symbol=cfg.symbol,
        report_dir=out_dir,
    )


def parse_args() -> argparse.Namespace:
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    p = argparse.ArgumentParser(description="ETH LF Portfolio V9C: V8 micro-confirm scaled + engine priority probe.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01", help="交易开始日期。warmup 数据只用于计算指标，不允许开仓。")
    p.add_argument("--end-date", default=today)
    p.add_argument("--warmup-start-date", default=None, help="指标预热数据开始日期。例如 2022-01-01；交易仍从 --start-date 开始。")
    p.add_argument("--warmup-days", type=int, default=365, help="如果未传 --warmup-start-date，默认向前多加载多少天用于指标预热。设为 0 可关闭。")
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--preset", choices=sorted(MOMENTUM_PRESETS), default="turbo")
    p.add_argument("--unit-risk-per-trade", type=float, default=None)
    p.add_argument("--max-total-notional-mult", type=float, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--min-risk-mult", type=float, default=0.35)
    p.add_argument("--max-risk-mult", type=float, default=None)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--disable-short", action="store_true")
    p.add_argument("--bear-preset", choices=sorted(BEAR_PRESETS), default="high")
    p.add_argument("--bear-min-risk-mult", type=float, default=0.25)
    p.add_argument("--bear-standalone-risk-scale", type=float, default=1.0)
    p.add_argument("--bear-standalone-quality-scale", type=float, default=1.0)
    p.add_argument("--disable-bear-standalone", action="store_true")
    p.add_argument("--bull-preset", choices=sorted(BULL_PRESETS), default="high")
    p.add_argument("--bull-min-risk-mult", type=float, default=0.25)
    p.add_argument("--bull-reclaim-risk-scale", type=float, default=1.0)
    p.add_argument("--bull-reclaim-quality-scale", type=float, default=1.0)
    p.add_argument("--bull-execution-mode", choices=["inherit", "own"], default="inherit", help="inherit=使用主引擎执行/保护参数，收益更强；own=使用 Bull V2 自己的短持仓保护参数，胜率更高。")
    p.add_argument("--disable-bull-reclaim", action="store_true")
    p.add_argument("--priority-mode", choices=sorted(PRIORITY_MODES), default="reclaim_first", help="Engine conflict routing. v8=Momentum>Bear>Bull; reclaim_first=Bull>Momentum>Bear; reclaim_bear_second=Bull>Bear>Momentum.")
    p.add_argument("--global-risk-scale", type=float, default=1.30, help="V8-style global risk multiplier applied to initial entries and add-ons. Default tests current champion scale 1.3.")
    p.add_argument("--quality-mult-cap", type=float, default=2.20)
    p.add_argument("--micro-filter-mode", choices=["off", "soft", "strict"], default="soft", help="off=V6 baseline; soft=reduce risk on contradicted range/footprint context; strict=block contradicted signals.")
    p.add_argument("--range-pct", type=float, default=0.002, help="Range bar size used for micro context.")
    p.add_argument("--price-step", type=float, default=1.0, help="Footprint price bucket step for optional max-bucket context.")
    p.add_argument("--range-data-dir", default=None)
    p.add_argument("--disable-footprint-context", action="store_true", help="Use range-bar aggregated orderflow only; skip footprint max-bucket features.")
    p.add_argument("--micro-min-range-bars", type=int, default=5)
    p.add_argument("--micro-contra-imbalance", type=float, default=0.05)
    p.add_argument("--micro-aligned-imbalance", type=float, default=0.05)
    p.add_argument("--micro-bad-close-pos", type=float, default=0.35)
    p.add_argument("--micro-good-close-pos", type=float, default=0.65)
    p.add_argument("--micro-contra-risk-scale", type=float, default=0.50, help="Soft mode risk scale for contradicted micro context.")
    p.add_argument("--micro-not-aligned-risk-scale", type=float, default=0.50, help="V7B soft mode risk scale for LF signals without micro aligned confirmation. Default only reduces risk, never boosts.")
    p.add_argument("--out-dir", default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    mom_cfg = make_momentum_config(args)
    bear_cfg = make_bear_config(args)
    bull_cfg = make_bull_config(args)
    exec_cfg = make_exec_config(mom_cfg)
    bull_exec_cfg = bull_to_exec_config(bull_cfg) if args.bull_execution_mode == "own" else exec_cfg
    out_dir = Path(args.out_dir) if args.out_dir else Path(PROJECT_ROOT) / "data/reports/lf" / STRATEGY_NAME / args.preset

    trade_start = pd.Timestamp(args.start_date)
    if args.warmup_start_date:
        load_start = pd.Timestamp(args.warmup_start_date)
    elif args.warmup_days and args.warmup_days > 0:
        load_start = trade_start - pd.Timedelta(days=int(args.warmup_days))
    else:
        load_start = trade_start
    load_start_str = load_start.strftime("%Y-%m-%d")

    print(f"Loading {args.symbol} 4H for warmup: {load_start_str} -> {args.end_date}; trade_start={args.start_date}")
    base = load_data(args.symbol, load_start_str, args.end_date, "4H")
    print(f"Loaded {len(base)} rows: {base.index[0]} -> {base.index[-1]}")
    momentum = build_momentum_features(base, mom_cfg)
    bear = build_bear_features(base, bear_cfg)
    bull = build_bull_features(base, bull_cfg)
    features = select_portfolio_signals(momentum, bear, bull, args)
    micro_ctx = load_range_footprint_context(args, load_start_str, args.end_date)
    features = apply_micro_context_filter(features, micro_ctx, args)
    # Critical: warmup rows are only used for indicators. Trading starts at --start-date.
    before_slice_rows = len(features)
    features = features.loc[trade_start: pd.Timestamp(args.end_date)].copy()
    print(f"Feature rows after warmup slice: {len(features)} / {before_slice_rows}; first tradeable bar={features.index[0] if not features.empty else 'NA'}")

    print("Signal counts:", {
        "momentum_long": int((features.momentum_signal == 1).sum()),
        "momentum_short": int((features.momentum_signal == -1).sum()),
        "bear_short": int((features.bear_signal == -1).sum()),
        "bull_long": int((features.bull_signal == 1).sum()),
        "portfolio_long": int((features.signal == 1).sum()),
        "portfolio_short": int((features.signal == -1).sum()),
        "bear_only": int(features.bear_only.sum()),
        "bull_reclaim": int(features.bull_reclaim.sum()),
        "portfolio_conflict": int(features.get("portfolio_conflict", pd.Series(False, index=features.index)).sum()),
        "priority_mode": args.priority_mode,
        "global_risk_scale": args.global_risk_scale,
        "micro_contra": int(features.get("micro_contra", pd.Series(False, index=features.index)).sum()),
        "micro_aligned": int(features.get("micro_aligned", pd.Series(False, index=features.index)).sum()),
    })

    trades, equity = run_priority_backtest(
        features,
        exec_cfg,
        engine_cfgs={"MOMENTUM_V3": exec_cfg, "BEAR_V3_ONLY": exec_cfg, "BULL_RECLAIM_V2": bull_exec_cfg},
        global_risk_scale=args.global_risk_scale,
    )
    trades = attach_engine_to_trades(trades, features)
    summary = summarize(trades, equity, exec_cfg.initial_capital)
    if trades:
        tdf = pd.DataFrame(trades)
        summary["engine_counts"] = tdf["engine"].value_counts().to_dict()
        summary["momentum_trade_count"] = int(tdf.get("momentum_selected", pd.Series(dtype=bool)).sum())
        summary["bear_only_trade_count"] = int(tdf.get("bear_only", pd.Series(dtype=bool)).sum())
        summary["bull_reclaim_trade_count"] = int(tdf.get("engine", pd.Series(dtype=str)).eq("BULL_RECLAIM_V2").sum())
    summary["preset"] = args.preset
    summary["bear_preset"] = args.bear_preset
    summary["bull_preset"] = args.bull_preset
    summary["bull_execution_mode"] = args.bull_execution_mode
    summary["priority_mode"] = args.priority_mode
    summary["priority_order"] = PRIORITY_MODES[args.priority_mode]
    summary["global_risk_scale"] = args.global_risk_scale
    summary["micro_filter_mode"] = args.micro_filter_mode
    summary["range_pct"] = args.range_pct
    summary["price_step"] = args.price_step
    summary["micro_contra_imbalance"] = args.micro_contra_imbalance
    summary["micro_contra_risk_scale"] = args.micro_contra_risk_scale
    summary["micro_not_aligned_risk_scale"] = args.micro_not_aligned_risk_scale
    summary["micro_contra_signal_count"] = int(features.get("micro_contra", pd.Series(False, index=features.index)).sum())
    summary["micro_aligned_signal_count"] = int(features.get("micro_aligned", pd.Series(False, index=features.index)).sum())
    summary["single_active_position"] = True
    summary["conflict_rule"] = f"Priority mode={args.priority_mode}; order={PRIORITY_MODES[args.priority_mode]}; range/footprint micro confirmation only risk-filters entry; no hedge; single active position."
    summary["fee_rate_per_side"] = args.fee_rate
    summary["warmup_start_date"] = load_start_str
    summary["trade_start_date"] = args.start_date
    summary["warmup_days"] = int(args.warmup_days or 0)

    write_outputs(trades, equity, features, summary, out_dir)
    print_summary(summary, out_dir)
    print_deep_report(trades, features, exec_cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
