#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LF V10B range/footprint micro context filters."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader


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


def load_range_footprint_context(args: Any, start_date: str, end_date: str) -> pd.DataFrame:
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
            fp = _FootprintMaxBucketStore(args.symbol, args.range_pct, args.price_step, args.range_data_dir).load_max_bucket_features(
                start_date,
                end_date,
            )
            if not fp.empty:
                rb = rb.merge(fp, on="bar_id", how="left")
                print(f"Loaded footprint max-bucket context: {len(fp):,}", flush=True)
        except Exception as exc:
            print(f"[WARN] footprint context unavailable: {exc}", flush=True)

    for col in [
        "open",
        "high",
        "low",
        "close",
        "notional",
        "volume",
        "buy_notional",
        "sell_notional",
        "delta",
        "delta_notional",
        "taker_buy_ratio",
    ]:
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
            row["rf_max_sell_bucket_share"] = float(
                pd.to_numeric(g["max_bucket_sell_notional"], errors="coerce").fillna(0.0).sum() / max(sell_sum, 1e-12)
            )
        else:
            row["rf_max_sell_bucket_share"] = np.nan
        if "max_bucket_buy_notional" in g.columns:
            row["rf_max_buy_bucket_share"] = float(
                pd.to_numeric(g["max_bucket_buy_notional"], errors="coerce").fillna(0.0).sum() / max(buy_sum, 1e-12)
            )
        else:
            row["rf_max_buy_bucket_share"] = np.nan
        grouped_rows.append(row)
    ctx = pd.DataFrame(grouped_rows)
    if ctx.empty:
        return ctx
    ctx = ctx.set_index("timestamp").sort_index()
    print(f"Range/footprint 4H context rows: {len(ctx):,} | {ctx.index[0]} -> {ctx.index[-1]}", flush=True)
    return ctx


def apply_micro_context_filter(features: pd.DataFrame, micro_ctx: pd.DataFrame, args: Any) -> pd.DataFrame:
    out = features.copy()
    micro_cols = [
        "rf_bar_count",
        "rf_micro_return_pct",
        "rf_close_pos",
        "rf_delta_sum",
        "rf_imbalance",
        "rf_taker_buy_ratio",
        "rf_max_sell_bucket_share",
        "rf_max_buy_bucket_share",
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
        blocked = not_aligned
        out.loc[blocked, "signal"] = 0
        out.loc[blocked, "long_signal"] = False
        out.loc[blocked, "short_signal"] = False
        out.loc[blocked, "micro_entry_risk_scale"] = 0.0
        out.loc[blocked, "micro_filter_action"] = "NOT_ALIGNED_BLOCKED"
    elif args.micro_filter_mode == "soft":
        out.loc[not_aligned, "micro_entry_risk_scale"] = float(args.micro_not_aligned_risk_scale)
        out.loc[not_aligned, "micro_filter_action"] = "NOT_ALIGNED_RISK_REDUCED"
        contra = long_contra | short_contra
        out.loc[contra, "micro_entry_risk_scale"] = float(args.micro_contra_risk_scale)
        out.loc[contra, "micro_filter_action"] = "CONTRA_RISK_REDUCED"
    else:
        raise ValueError(f"Unsupported micro_filter_mode={args.micro_filter_mode}")

    print(
        "Micro context counts:",
        {
            "available": int(out["micro_context_available"].sum()),
            "aligned_signal": int(out["micro_aligned"].sum()),
            "contra_signal": int(out["micro_contra"].sum()),
            "not_aligned_signal": int(not_aligned.sum()),
            "risk_reduced": int(out["micro_filter_action"].astype(str).str.contains("RISK_REDUCED", na=False).sum()),
            "blocked": int(out["micro_filter_action"].astype(str).str.contains("BLOCKED", na=False).sum()),
        },
        flush=True,
    )
    return out


def build_momentum_long_not_aligned_block_mask(momentum: pd.DataFrame, micro_ctx: pd.DataFrame, args: Any) -> pd.Series:
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
    long_aligned = long_sig & has_ctx & (rf_imbalance >= aligned_imb) & (rf_close_pos >= float(args.micro_good_close_pos))
    long_contra = long_sig & has_ctx & (rf_imbalance <= -contra_imb) & (rf_close_pos <= float(args.micro_bad_close_pos))
    return long_sig & has_ctx & (~long_aligned) & (~long_contra)


def apply_momentum_long_not_aligned_block(momentum: pd.DataFrame, micro_ctx: pd.DataFrame, args: Any) -> pd.DataFrame:
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


def build_momentum_short_fast_speed_block_mask(momentum: pd.DataFrame, micro_ctx: pd.DataFrame, args: Any) -> pd.Series:
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


def apply_momentum_short_fast_speed_block(momentum: pd.DataFrame, micro_ctx: pd.DataFrame, args: Any) -> pd.DataFrame:
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

