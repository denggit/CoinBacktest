#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH LF Portfolio V10A Momentum Micro + Short Speed Filter
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
    V10 在 V9E 冻结基线之上新增 Momentum Long micro filter：仅当 MOMENTUM_V3 多头原始信号处于 NOT_ALIGNED_RISK_REDUCED 环境时，禁止该 Momentum 信号独立入场；不影响 Bull Reclaim、Bear V3、Momentum Short，不改出场逻辑、不改引擎优先级、不改仓位放大逻辑。
    V10A 在 V10 之上新增研究候选过滤：当 MOMENTUM_V3 空头原始信号发生在 past-only rolling range speed FAST_Q4 环境时，禁止该 Momentum Short 独立入场。该规则只使用已完成 4H 信号 bar 对应的 range-bar 数量，以及此前 4H buckets 的 rolling Q75 阈值，不使用未来分位。
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

STRATEGY_NAME = "eth_lf_portfolio_v10a_momentum_micro_short_speed_filter"
REPORT_STRATEGY_NAME = "ETH_LF_Portfolio_V10A_MomentumMicroShortSpeedFilter"

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


def build_momentum_long_not_aligned_block_mask(momentum: pd.DataFrame, micro_ctx: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    """Build the V10 Momentum Long block mask without lookahead.

    The mask uses only the completed range/footprint bucket aligned to the closed 4H
    signal bar. Execution is still next 4H open, so this does not use future data.

    Important:
        We intentionally do NOT include the research-only LOW_VOLUME/VOL_Q1 rule here.
        The robustness lab's VOL_Q1 was computed from full-sample signal-event quantiles,
        which is useful for diagnosis but not safe as a production backtest rule until
        it is converted to a rolling/past-only threshold and revalidated.
    """
    out = pd.Series(False, index=momentum.index, dtype=bool)
    if bool(getattr(args, "disable_momentum_long_not_aligned_block", False)):
        return out
    if micro_ctx.empty or getattr(args, "micro_filter_mode", "soft") == "off":
        return out

    aligned = micro_ctx.reindex(momentum.index)
    rf_bar_count = pd.to_numeric(aligned.get("rf_bar_count", pd.Series(np.nan, index=momentum.index)), errors="coerce")
    rf_imbalance = pd.to_numeric(aligned.get("rf_imbalance", pd.Series(np.nan, index=momentum.index)), errors="coerce")
    rf_close_pos = pd.to_numeric(aligned.get("rf_close_pos", pd.Series(np.nan, index=momentum.index)), errors="coerce")

    has_ctx = rf_bar_count.fillna(0.0).astype(float) >= float(args.micro_min_range_bars)
    sig = pd.to_numeric(momentum.get("signal", pd.Series(0, index=momentum.index)), errors="coerce").fillna(0).astype(int)
    long_sig = sig.eq(1)

    aligned_imb = abs(float(args.micro_aligned_imbalance))
    contra_imb = abs(float(args.micro_contra_imbalance))
    good_pos = float(args.micro_good_close_pos)
    bad_pos = float(args.micro_bad_close_pos)

    long_aligned = long_sig & has_ctx & (rf_imbalance >= aligned_imb) & (rf_close_pos >= good_pos)
    long_contra = long_sig & has_ctx & (rf_imbalance <= -contra_imb) & (rf_close_pos <= bad_pos)

    # Match the research lab's engine-specific micro action:
    # NOT_ALIGNED_RISK_REDUCED = signal has context but is neither aligned nor contra.
    return long_sig & has_ctx & (~long_aligned) & (~long_contra)


def apply_momentum_long_not_aligned_block(momentum: pd.DataFrame, micro_ctx: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """V10 entry filter: block only raw Momentum Long NOT_ALIGNED entries.

    This is engine-specific and direction-specific. It does not block:
        - BULL_RECLAIM_V2 long entries
        - BEAR_V3_ONLY short entries
        - MOMENTUM_V3 short entries
        - Momentum signals that are already aligned/neutral without completed micro context
    """
    out = momentum.copy()
    mask = build_momentum_long_not_aligned_block_mask(out, micro_ctx, args)
    out["momentum_long_not_aligned_blocked"] = False
    out["momentum_long_not_aligned_block_reason"] = "NONE"
    if not bool(mask.any()):
        print("V10 Momentum Long NOT_ALIGNED block count: 0", flush=True)
        return out

    for col in ["signal", "momentum_signal"]:
        if col in out.columns:
            out.loc[mask, col] = 0
    for col in ["long_signal", "short_signal"]:
        if col in out.columns:
            out.loc[mask, col] = False

    out.loc[mask, "momentum_long_not_aligned_blocked"] = True
    out.loc[mask, "momentum_long_not_aligned_block_reason"] = "MOMENTUM_LONG_NOT_ALIGNED_BLOCKED"
    print(f"V10 Momentum Long NOT_ALIGNED block count: {int(mask.sum())}", flush=True)
    return out




def build_momentum_short_fast_speed_block_mask(momentum: pd.DataFrame, micro_ctx: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    """Build the V10A Momentum Short FAST range-speed block mask without lookahead.

    Definition:
        FAST range speed means the completed 4H signal bucket's `rf_bar_count` is
        greater than or equal to the Q75 threshold computed only from earlier 4H
        range/footprint buckets:

            threshold[t] = rolling_quantile(rf_bar_count.shift(1), q75)

    This is engine-specific and direction-specific. It only blocks raw
    MOMENTUM_V3 short entries; it does not affect Bear V3, Bull Reclaim, or
    Momentum Long.
    """
    out = pd.Series(False, index=momentum.index, dtype=bool)
    if bool(getattr(args, "disable_momentum_short_fast_speed_block", False)):
        return out
    if micro_ctx.empty or getattr(args, "micro_filter_mode", "soft") == "off":
        return out
    if "rf_bar_count" not in micro_ctx.columns:
        return out

    ctx = micro_ctx.sort_index().copy()
    rf_count = pd.to_numeric(ctx["rf_bar_count"], errors="coerce")
    window = int(getattr(args, "rf_speed_rolling_window_bars", 1080) or 1080)
    min_periods = int(getattr(args, "rf_speed_min_periods", 100) or 100)
    q = float(getattr(args, "rf_speed_fast_quantile", 0.75) or 0.75)
    threshold = rf_count.shift(1).rolling(window, min_periods=min_periods).quantile(q)
    fast_speed = rf_count.ge(threshold)

    sig = pd.to_numeric(momentum.get("signal", pd.Series(0, index=momentum.index)), errors="coerce").fillna(0).astype(int)
    short_sig = sig.eq(-1)
    fast_at_signal = fast_speed.reindex(momentum.index).astype("boolean").fillna(False).astype(bool)
    return short_sig & fast_at_signal


def apply_momentum_short_fast_speed_block(momentum: pd.DataFrame, micro_ctx: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """V10A entry filter: block raw Momentum Short entries in FAST range-speed buckets."""
    out = momentum.copy()
    mask = build_momentum_short_fast_speed_block_mask(out, micro_ctx, args)
    out["momentum_short_fast_speed_blocked"] = False
    out["momentum_short_fast_speed_block_reason"] = "NONE"
    if not bool(mask.any()):
        print("V10A Momentum Short FAST speed block count: 0", flush=True)
        return out

    for col in ["signal", "momentum_signal"]:
        if col in out.columns:
            out.loc[mask, col] = 0
    for col in ["long_signal", "short_signal"]:
        if col in out.columns:
            out.loc[mask, col] = False

    out.loc[mask, "momentum_short_fast_speed_blocked"] = True
    out.loc[mask, "momentum_short_fast_speed_block_reason"] = "MOMENTUM_SHORT_FAST_SPEED_BLOCKED"
    print(f"V10A Momentum Short FAST speed block count: {int(mask.sum())}", flush=True)
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


def _float_attr(row: Any, name: str, default: float = float("nan")) -> float:
    value = getattr(row, name, default)
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _range_exit_signal(
    row: Any,
    *,
    side: int,
    avg_entry: float,
    risk_per_coin: float,
    max_fav: float,
    hold_bars: int,
    args: argparse.Namespace | None,
) -> tuple[bool, str, dict[str, float | bool | str]]:
    """Decide whether to exit next 4H open using completed-bar range/footprint context.

    No future-function rule:
        The row is the currently closed 4H bar. Its range/footprint context only contains
        range bars whose end_ts falls inside this completed 4H bucket. If triggered, the
        exit is executed at rows[i + 1].open, consistent with the existing SAFE executor.

    V9E scope:
        This overlay never creates entries, never flips side, never changes priority, and
        never moves the original stop. It only adds an optional next-open protective exit
        after a trade has already reached a configurable MFE threshold and then gives back
        a configurable fraction of peak R with hostile range/footprint context.
    """
    meta: dict[str, float | bool | str] = {
        "range_exit_triggered": False,
        "range_exit_peak_r": float("nan"),
        "range_exit_current_r": float("nan"),
        "range_exit_giveback_frac": float("nan"),
        "range_exit_reversal": False,
        "range_exit_reason": "",
    }
    if args is None or getattr(args, "range_exit_mode", "off") == "off":
        return False, "", meta
    if side not in (1, -1) or not math.isfinite(risk_per_coin) or risk_per_coin <= 0:
        return False, "", meta
    if hold_bars < int(getattr(args, "range_exit_min_hold_bars", 2)):
        return False, "", meta

    close = _float_attr(row, "close")
    if side == 1:
        peak_r = (float(max_fav) - float(avg_entry)) / risk_per_coin
        current_r = (close - float(avg_entry)) / risk_per_coin
    else:
        peak_r = (float(avg_entry) - float(max_fav)) / risk_per_coin
        current_r = (float(avg_entry) - close) / risk_per_coin
    if not (math.isfinite(peak_r) and math.isfinite(current_r)):
        return False, "", meta
    if peak_r < float(getattr(args, "range_exit_min_mfe_r", 2.0)):
        return False, "", meta

    giveback_frac = (peak_r - current_r) / max(abs(peak_r), 1e-12)
    meta["range_exit_peak_r"] = float(peak_r)
    meta["range_exit_current_r"] = float(current_r)
    meta["range_exit_giveback_frac"] = float(giveback_frac)
    if giveback_frac < float(getattr(args, "range_exit_giveback_frac", 0.65)):
        return False, "", meta

    has_ctx = bool(getattr(row, "micro_context_available", False))
    imbalance = _float_attr(row, "rf_imbalance")
    close_pos = _float_attr(row, "rf_close_pos")
    contra_imb = abs(float(getattr(args, "range_exit_contra_imbalance", 0.05)))
    bad_close_pos = float(getattr(args, "range_exit_bad_close_pos", 0.35))

    if side == 1:
        hostile_imb = math.isfinite(imbalance) and imbalance <= -contra_imb
        hostile_close = math.isfinite(close_pos) and close_pos <= bad_close_pos
    else:
        hostile_imb = math.isfinite(imbalance) and imbalance >= contra_imb
        hostile_close = math.isfinite(close_pos) and close_pos >= 1.0 - bad_close_pos
    reversal = bool(has_ctx and (hostile_imb or hostile_close))
    meta["range_exit_reversal"] = reversal

    if bool(getattr(args, "range_exit_require_reversal", True)) and not reversal:
        return False, "", meta

    delay_bars = int(getattr(args, "range_exit_delay_bars", 0) or 0)
    reason = "RANGE_EXIT_NEXT_OPEN" if delay_bars <= 0 else f"RANGE_EXIT_DELAY_{delay_bars}BAR_OPEN"
    meta["range_exit_triggered"] = True
    meta["range_exit_reason"] = reason
    meta["range_exit_delay_bars"] = float(delay_bars)
    return True, reason, meta


def run_priority_backtest(
    df: pd.DataFrame,
    cfg: ExecConfig,
    engine_cfgs: dict[str, ExecConfig] | None = None,
    *,
    global_risk_scale: float = 1.0,
    args: argparse.Namespace | None = None,
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
    pending_range_exit_i: int | None = None
    pending_range_exit_reason = ""
    pending_range_exit_meta: dict[str, Any] = {}

    rows = list(df.itertuples())
    idx = df.index

    for i in range(len(rows) - 1):
        row = rows[i]
        ts = idx[i]
        if in_pos:
            if pending_range_exit_i is not None and i >= pending_range_exit_i:
                active_cfg = pos_cfg
                hold_bars = i - entry_i
                exit_price = apply_exit_slippage(float(row.open), side, active_cfg.slippage_pct)
                exit_time = idx[i]
                reason = pending_range_exit_reason or "RANGE_EXIT_DELAYED_OPEN"
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
                if trades:
                    trades[-1].update(pending_range_exit_meta)
                    trades[-1]["range_exit_executed_after_delay"] = True
                peak = max(peak, capital)
                in_pos = False
                side = 0
                last_exit_i = i
                pending_range_exit_i = None
                pending_range_exit_reason = ""
                pending_range_exit_meta = {}
            else:
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

                range_exit_now, range_exit_reason, range_exit_meta = _range_exit_signal(
                    row,
                    side=side,
                    avg_entry=avg_entry,
                    risk_per_coin=risk_per_coin,
                    max_fav=max_fav,
                    hold_bars=hold_bars,
                    args=args,
                )

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
                elif range_exit_now:
                    delay_bars = int(getattr(args, "range_exit_delay_bars", 0) or 0)
                    if delay_bars <= 0:
                        exit_now = True
                        exit_price = apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                        exit_time = idx[i + 1]
                        reason = range_exit_reason
                    else:
                        pending_range_exit_i = min(i + 1 + delay_bars, max(i + 1, len(rows) - 2))
                        pending_range_exit_reason = range_exit_reason
                        pending_range_exit_meta = dict(range_exit_meta)
                        pending_range_exit_meta["range_exit_signal_time"] = str(ts)
                        pending_range_exit_meta["range_exit_scheduled_exit_time"] = str(idx[pending_range_exit_i])
                        pending_range_exit_meta["range_exit_delay_bars"] = float(delay_bars)
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
                    if trades and str(reason).startswith("RANGE_EXIT"):
                        trades[-1].update(range_exit_meta)
                    peak = max(peak, capital)
                    in_pos = False
                    side = 0
                    last_exit_i = i
                    pending_range_exit_i = None
                    pending_range_exit_reason = ""
                    pending_range_exit_meta = {}
                else:
                    stop_price = next_stop

                if in_pos and pending_range_exit_i is None and units < active_cfg.max_units:
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
                    pending_range_exit_i = None
                    pending_range_exit_reason = ""
                    pending_range_exit_meta = {}

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
        item.setdefault("range_exit_triggered", str(item.get("note", "")).startswith("RANGE_EXIT"))
        item.setdefault("range_exit_peak_r", float("nan"))
        item.setdefault("range_exit_current_r", float("nan"))
        item.setdefault("range_exit_giveback_frac", float("nan"))
        item.setdefault("range_exit_reversal", False)
        item.setdefault("range_exit_reason", str(item.get("note", "")) if str(item.get("note", "")).startswith("RANGE_EXIT") else "")
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
        "momentum_long_not_aligned_blocked", "momentum_long_not_aligned_block_reason",
        "momentum_short_fast_speed_blocked", "momentum_short_fast_speed_block_reason",
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
    print("ETH LF Portfolio V10A Momentum Micro + Short Speed Filter Backtest Summary")
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
    p = argparse.ArgumentParser(description="ETH LF Portfolio V9E: V9C frozen baseline + range/footprint exit overlay probe.")
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
    p.add_argument("--range-exit-mode", choices=["off", "soft"], default="soft", help="V9E overlay. off=V9C baseline; soft=exit next 4H open after MFE giveback plus hostile range/footprint context.")
    p.add_argument("--range-exit-min-mfe-r", type=float, default=2.0, help="Only consider range-exit overlay after the open trade has reached at least this peak MFE in R.")
    p.add_argument("--range-exit-giveback-frac", type=float, default=0.65, help="Exit only after giving back this fraction of peak R. Example: 0.65 means peak 4R can trigger at <=1.4R.")
    p.add_argument("--range-exit-min-hold-bars", type=int, default=2, help="Minimum completed 4H bars after entry before the overlay may exit.")
    p.add_argument("--range-exit-delay-bars", type=int, default=0, help="Stress test only. 0=exit next 4H open; 1=delay one additional 4H bar after the normal next-open exit; 2=delay two additional bars.")
    p.add_argument("--range-exit-contra-imbalance", type=float, default=0.05, help="Hostile range/footprint imbalance threshold for protective exit.")
    p.add_argument("--range-exit-bad-close-pos", type=float, default=0.35, help="Hostile close-position threshold inside the completed 4H range bucket.")
    p.add_argument("--range-exit-no-reversal-required", dest="range_exit_require_reversal", action="store_false", help="Allow MFE giveback exit without hostile range/footprint reversal context. Usually too aggressive; for stress testing only.")
    p.set_defaults(range_exit_require_reversal=True)
    p.add_argument("--disable-momentum-long-not-aligned-block", action="store_true", help="V10/V10A: disable the engine-specific block for raw MOMENTUM_V3 Long signals when completed micro context is NOT_ALIGNED_RISK_REDUCED.")
    p.add_argument("--disable-momentum-short-fast-speed-block", action="store_true", help="V10A only: disable the engine-specific block for raw MOMENTUM_V3 Short signals when completed range speed is FAST_Q4 by past-only rolling Q75.")
    p.add_argument("--rf-speed-rolling-window-bars", type=int, default=1080, help="V10A: rolling 4H bucket window for past-only range-speed Q75 threshold. Default 1080 ~= 180 days of 4H bars.")
    p.add_argument("--rf-speed-min-periods", type=int, default=100, help="V10A: minimum historical 4H buckets required before range-speed FAST_Q4 can be evaluated.")
    p.add_argument("--rf-speed-fast-quantile", type=float, default=0.75, help="V10A: past-only rolling quantile threshold for FAST range speed.")
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
    micro_ctx = load_range_footprint_context(args, load_start_str, args.end_date)
    momentum = apply_momentum_long_not_aligned_block(momentum, micro_ctx, args)
    momentum = apply_momentum_short_fast_speed_block(momentum, micro_ctx, args)
    features = select_portfolio_signals(momentum, bear, bull, args)
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
        "momentum_long_not_aligned_blocked": int(features.get("momentum_long_not_aligned_blocked", pd.Series(False, index=features.index)).sum()),
        "momentum_short_fast_speed_blocked": int(features.get("momentum_short_fast_speed_blocked", pd.Series(False, index=features.index)).sum()),
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
        args=args,
    )
    trades = attach_engine_to_trades(trades, features)
    summary = summarize(trades, equity, exec_cfg.initial_capital)
    if trades:
        tdf = pd.DataFrame(trades)
        summary["engine_counts"] = tdf["engine"].value_counts().to_dict()
        summary["momentum_trade_count"] = int(tdf.get("momentum_selected", pd.Series(dtype=bool)).sum())
        summary["bear_only_trade_count"] = int(tdf.get("bear_only", pd.Series(dtype=bool)).sum())
        summary["bull_reclaim_trade_count"] = int(tdf.get("engine", pd.Series(dtype=str)).eq("BULL_RECLAIM_V2").sum())
        note_col = tdf.get("note", pd.Series(dtype=str)).astype(str)
        summary["range_exit_trade_count"] = int(note_col.str.startswith("RANGE_EXIT").sum())
        if "range_exit_peak_r" in tdf.columns:
            summary["range_exit_avg_peak_r"] = float(pd.to_numeric(tdf.loc[note_col.str.startswith("RANGE_EXIT"), "range_exit_peak_r"], errors="coerce").mean())
        if "range_exit_giveback_frac" in tdf.columns:
            summary["range_exit_avg_giveback_frac"] = float(pd.to_numeric(tdf.loc[note_col.str.startswith("RANGE_EXIT"), "range_exit_giveback_frac"], errors="coerce").mean())
    else:
        summary["range_exit_trade_count"] = 0
    summary["preset"] = args.preset
    summary["bear_preset"] = args.bear_preset
    summary["bull_preset"] = args.bull_preset
    summary["bull_execution_mode"] = args.bull_execution_mode
    summary["priority_mode"] = args.priority_mode
    summary["priority_order"] = PRIORITY_MODES[args.priority_mode]
    summary["global_risk_scale"] = args.global_risk_scale
    summary["micro_filter_mode"] = args.micro_filter_mode
    summary["momentum_long_not_aligned_block_enabled"] = not bool(args.disable_momentum_long_not_aligned_block)
    summary["momentum_long_not_aligned_block_count"] = int(features.get("momentum_long_not_aligned_blocked", pd.Series(False, index=features.index)).sum())
    summary["momentum_short_fast_speed_block_enabled"] = not bool(args.disable_momentum_short_fast_speed_block)
    summary["momentum_short_fast_speed_block_count"] = int(features.get("momentum_short_fast_speed_blocked", pd.Series(False, index=features.index)).sum())
    summary["momentum_short_fast_speed_rolling_window_bars"] = int(args.rf_speed_rolling_window_bars)
    summary["momentum_short_fast_speed_min_periods"] = int(args.rf_speed_min_periods)
    summary["momentum_short_fast_speed_quantile"] = float(args.rf_speed_fast_quantile)
    summary["momentum_short_fast_speed_block_note"] = "V10A candidate: block raw MOMENTUM_V3 Short entries when completed 4H range-bar count is >= past-only rolling Q75 threshold. Robustness improved full/pre-2026/holdout/stress but 2024 year-reset was slightly below V10; treat as candidate until further validation."
    summary["momentum_long_low_volume_block_enabled"] = False
    summary["momentum_long_low_volume_block_note"] = "Not included in V10 because the research VOL_Q1 rule used full-sample signal-event quantiles; requires a no-lookahead rolling validation before production use."
    summary["range_pct"] = args.range_pct
    summary["price_step"] = args.price_step
    summary["micro_contra_imbalance"] = args.micro_contra_imbalance
    summary["micro_contra_risk_scale"] = args.micro_contra_risk_scale
    summary["micro_not_aligned_risk_scale"] = args.micro_not_aligned_risk_scale
    summary["range_exit_mode"] = args.range_exit_mode
    summary["range_exit_min_mfe_r"] = args.range_exit_min_mfe_r
    summary["range_exit_giveback_frac"] = args.range_exit_giveback_frac
    summary["range_exit_min_hold_bars"] = args.range_exit_min_hold_bars
    summary["range_exit_delay_bars"] = args.range_exit_delay_bars
    summary["range_exit_contra_imbalance"] = args.range_exit_contra_imbalance
    summary["range_exit_bad_close_pos"] = args.range_exit_bad_close_pos
    summary["range_exit_require_reversal"] = args.range_exit_require_reversal
    summary["micro_contra_signal_count"] = int(features.get("micro_contra", pd.Series(False, index=features.index)).sum())
    summary["micro_aligned_signal_count"] = int(features.get("micro_aligned", pd.Series(False, index=features.index)).sum())
    summary["single_active_position"] = True
    summary["conflict_rule"] = f"Priority mode={args.priority_mode}; order={PRIORITY_MODES[args.priority_mode]}; V9C entries unchanged; V9E range/footprint overlay may only add protective next-open exits; no hedge; single active position."
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
