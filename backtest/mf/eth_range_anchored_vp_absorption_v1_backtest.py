#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Range Anchored Volume Profile Absorption Breakout V1.

Fabio-style prototype:
    downleg / lower-high context
    -> anchored volume profile over the prior up/down auction
    -> VAL/lower-POC absorption zone with repeated sell bubbles
    -> buy-stop breakout entry above absorption zone
    -> stop below lower POC, target at anchored VAH

This is intentionally a first-pass, long-only, anti-future implementation.
It uses prebuilt range bars and range-bar footprint data from:
    src.data_feed.okx_range_bar_loader.OKXRangeBarLoader
    src.data_feed.okx_range_footprint_loader.OKXRangeFootprintLoader
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage
from src.backtest_common.reporting import build_report_trades, summarize_r_trades
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader
from src.utils.report import print_full_report

STRATEGY_NAME = "ETH_Range_AnchoredVP_AbsorptionBreakout_V1"


@dataclass
class Config:
    symbol: str = "ETH-USDT-SWAP"
    start_date: str = "2023-01-01"
    end_date: str = "2026-06-15"
    warmup_start_date: str = "2022-01-01"
    initial_capital: float = 1000.0
    fee_rate: float = 0.00055
    slippage_pct: float = 0.00015
    range_pct: float = 0.0020
    price_step: float = 1.0
    data_dir: str | None = None

    # Position sizing is deliberately small for first prototype.
    unit_risk_per_trade: float = 0.0020
    max_notional_mult: float = 1.0

    # Swing / downleg context.
    pivot_lookback: int = 3
    min_downmove_pct: float = 0.008
    profile_padding_bars: int = 30
    max_profile_bars: int = 600
    max_active_profile_bars: int = 500
    max_pending_order_bars: int = 120
    value_area_pct: float = 0.70
    lower_poc_min_separation_pct: float = 0.0005
    lower_poc_zone_rel_vol: float = 0.35

    # Absorption / bubble rules.
    absorption_lookback_bars: int = 24
    min_absorption_bubbles: int = 2
    bubble_quantile_window: int = 200
    bubble_sell_quantile: float = 0.90
    bubble_large_sell_quantile: float = 0.85
    max_zone_width_pct: float = 0.0120
    val_tolerance_pct: float = 0.0025
    lower_poc_invalidate_buffer_pct: float = 0.0008

    # Order / exits.
    entry_buffer_pct: float = 0.0002
    stop_buffer_pct: float = 0.0003
    min_reward_risk: float = 1.25
    max_holding_bars: int = 180
    cooldown_bars: int = 30
    move_stop_to_zone_after_r: float = 0.75
    trail_retest_buffer_pct: float = 0.0003

    # Auditing.
    out_dir: Path = Path("data/reports/mf/eth_range_anchored_vp_absorption_v1")
    write_full_audit: bool = False


PRESETS = {
    "stable": {"unit_risk_per_trade": 0.0015, "max_notional_mult": 0.8},
    "high": {"unit_risk_per_trade": 0.0020, "max_notional_mult": 1.0},
    "turbo": {"unit_risk_per_trade": 0.0030, "max_notional_mult": 1.3},
}


def _normalize_loader_frame(df: pd.DataFrame, sort_cols: list[str]) -> pd.DataFrame:
    """Reset loader index so columns like end_ts are not both index and label.

    OKXRangeBarLoader returns end_ts as both index name and column. Pandas 2.x
    raises ValueError when sorting by such ambiguous labels.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    if out.index.name in set(sort_cols):
        out = out.reset_index(drop=True)
    else:
        # Drop any non-default index for consistent downstream iloc/columns behavior.
        out = out.reset_index(drop=True)
    return out.sort_values(sort_cols).copy()


def _to_timestamp(value: str) -> pd.Timestamp:
    return pd.Timestamp(value)


def load_range_data(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    bar_loader = OKXRangeBarLoader(symbol=cfg.symbol, range_pct=cfg.range_pct, data_dir=cfg.data_dir)
    fp_loader = OKXRangeFootprintLoader(
        symbol=cfg.symbol,
        range_pct=cfg.range_pct,
        price_step=cfg.price_step,
        data_dir=cfg.data_dir,
    )
    bars = bar_loader.load_local_data(start_date=cfg.warmup_start_date, end_date=cfg.end_date)
    fps = fp_loader.load_local_data(start_date=cfg.warmup_start_date, end_date=cfg.end_date)
    if bars.empty:
        raise RuntimeError("No range bar data loaded. Please prebuild range bars first.")
    if fps.empty:
        raise RuntimeError("No range footprint data loaded. Please prebuild range footprints first.")
    bars = _normalize_loader_frame(bars, ["end_ts", "bar_id"])
    fps = _normalize_loader_frame(fps, ["bar_id", "price_bucket"])
    # CVD inside loaded window for reproducibility.
    bars["cvd_volume"] = bars["delta_volume"].cumsum()
    bars["cvd_notional"] = bars["delta_notional"].cumsum()
    return bars, fps


def add_bar_features(bars: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = bars.copy().reset_index(drop=True)
    out["idx"] = np.arange(len(out))
    out["sell_bubble_threshold"] = out["sell_notional"].rolling(cfg.bubble_quantile_window, min_periods=50).quantile(cfg.bubble_sell_quantile).shift(1)
    out["large_sell_bubble_threshold"] = out["large_sell_notional"].rolling(cfg.bubble_quantile_window, min_periods=50).quantile(cfg.bubble_large_sell_quantile).shift(1)
    out["range_mid"] = (out["high"] + out["low"]) / 2.0
    out["body_high"] = out[["open", "close"]].max(axis=1)
    out["body_low"] = out[["open", "close"]].min(axis=1)
    out["bar_ret"] = out["close"].pct_change().fillna(0.0)
    return out


def _confirmed_pivots_at_i(high: np.ndarray, low: np.ndarray, i: int, k: int) -> tuple[int | None, int | None]:
    """Return pivot index confirmed at current index i, using only bars <= i.

    A pivot at c=i-k is confirmed only after k bars to its right have closed.
    This deliberately lags pivot recognition to avoid future leakage.
    """
    if i < 2 * k:
        return None, None
    c = i - k
    lo = i - 2 * k
    hi = i + 1
    win_h = high[lo:hi]
    win_l = low[lo:hi]
    ph = c if high[c] >= np.nanmax(win_h) else None
    pl = c if low[c] <= np.nanmin(win_l) else None
    return ph, pl


def _value_area_from_profile(prof: pd.DataFrame, value_area_pct: float) -> dict[str, float] | None:
    if prof.empty or prof["volume"].sum() <= 0:
        return None
    p = prof.sort_values("price_bucket").reset_index(drop=True)
    prices = p["price_bucket"].to_numpy(float)
    vols = p["volume"].to_numpy(float)
    poc_i = int(np.argmax(vols))
    total = float(vols.sum())
    target = total * value_area_pct
    lo = hi = poc_i
    cum = float(vols[poc_i])
    while cum < target and (lo > 0 or hi < len(vols) - 1):
        lv = vols[lo - 1] if lo > 0 else -1.0
        hv = vols[hi + 1] if hi < len(vols) - 1 else -1.0
        if hv >= lv:
            hi += 1
            cum += float(vols[hi])
        else:
            lo -= 1
            cum += float(vols[lo])
    return {
        "poc": float(prices[poc_i]),
        "vah": float(prices[hi]),
        "val": float(prices[lo]),
        "profile_low": float(prices[0]),
        "profile_high": float(prices[-1]),
        "profile_volume": total,
    }


def _find_lower_poc(prof: pd.DataFrame, val: float, cfg: Config) -> dict[str, float] | None:
    below = prof[prof["price_bucket"] < val * (1.0 - cfg.lower_poc_min_separation_pct)].copy()
    if below.empty or below["volume"].sum() <= 0:
        return None
    # Highest-volume node below VAL. This approximates the small lower POC/HVN in the screenshots.
    row = below.loc[below["volume"].idxmax()]
    lp = float(row["price_bucket"])
    lv = float(row["volume"])
    ordered = below.sort_values("price_bucket").reset_index(drop=True)
    prices = ordered["price_bucket"].to_numpy(float)
    vols = ordered["volume"].to_numpy(float)
    idx = int(np.argmin(np.abs(prices - lp)))
    lo = hi = idx
    threshold = lv * cfg.lower_poc_zone_rel_vol
    while lo > 0 and vols[lo - 1] >= threshold:
        lo -= 1
    while hi < len(vols) - 1 and vols[hi + 1] >= threshold:
        hi += 1
    return {
        "lower_poc": lp,
        "lower_poc_volume": lv,
        "lower_poc_zone_low": float(prices[lo]),
        "lower_poc_zone_high": float(prices[hi] + cfg.price_step),
    }


def compute_anchored_profile(
    fps: pd.DataFrame,
    start_bar_id: int,
    end_bar_id: int,
    cfg: Config,
) -> dict[str, float] | None:
    sub = fps[(fps["bar_id"] >= int(start_bar_id)) & (fps["bar_id"] <= int(end_bar_id))]
    if sub.empty:
        return None
    prof = (
        sub.groupby("price_bucket", as_index=False)
        .agg(
            volume=("volume", "sum"),
            buy_notional=("buy_notional", "sum"),
            sell_notional=("sell_notional", "sum"),
            delta_notional=("delta_notional", "sum"),
            large_sell_notional=("large_sell_notional", "sum"),
        )
        .sort_values("price_bucket")
    )
    va = _value_area_from_profile(prof, cfg.value_area_pct)
    if va is None:
        return None
    lp = _find_lower_poc(prof, va["val"], cfg)
    if lp is None:
        return None
    out = {**va, **lp}
    out["profile_rows"] = int(len(prof))
    return out


def generate_setups_and_signals(bars: pd.DataFrame, fps: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    b = add_bar_features(bars, cfg)
    n = len(b)
    high = b["high"].to_numpy(float)
    low = b["low"].to_numpy(float)
    close = b["close"].to_numpy(float)
    open_ = b["open"].to_numpy(float)
    bar_ids = b["bar_id"].to_numpy(np.int64)
    sell_notional = b["sell_notional"].to_numpy(float)
    large_sell_notional = b["large_sell_notional"].to_numpy(float)
    delta = b["delta_notional"].to_numpy(float)
    sell_q = b["sell_bubble_threshold"].to_numpy(float)
    large_sell_q = b["large_sell_bubble_threshold"].to_numpy(float)
    end_ts = pd.to_datetime(b["end_ts"]).to_numpy()

    # Signal/audit columns.
    sig = np.zeros(n, dtype=np.int8)
    entry_trigger = np.full(n, np.nan)
    initial_stop = np.full(n, np.nan)
    target_price = np.full(n, np.nan)
    profile_vah = np.full(n, np.nan)
    profile_val = np.full(n, np.nan)
    profile_poc = np.full(n, np.nan)
    lower_poc = np.full(n, np.nan)
    absorption_low = np.full(n, np.nan)
    absorption_high = np.full(n, np.nan)
    bubble_count_arr = np.zeros(n, dtype=int)
    setup_id_arr = np.full(n, -1, dtype=int)
    signal_reason = np.array([""] * n, dtype=object)

    events: list[dict[str, Any]] = []
    piv_highs: list[int] = []
    piv_lows: list[int] = []
    active: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    cooldown_until = -1
    setup_seq = 0
    profile_cache: dict[tuple[int, int], dict[str, float] | None] = {}

    def log_event(i: int, event: str, info: dict[str, Any]) -> None:
        row = {"event_time": pd.Timestamp(end_ts[i]), "bar_id": int(bar_ids[i]), "event": event}
        row.update(info)
        events.append(row)

    for i in range(n):
        ph, pl = _confirmed_pivots_at_i(high, low, i, cfg.pivot_lookback)
        if ph is not None:
            piv_highs.append(ph)
        if pl is not None:
            piv_lows.append(pl)

        # Pending buy-stop order, active only after setup qualification bar.
        if pending is not None and i > pending["active_from_i"]:
            if i - pending["active_from_i"] > cfg.max_pending_order_bars:
                log_event(i, "cancel_pending_timeout", pending)
                pending = None
                active = None
                cooldown_until = i + cfg.cooldown_bars
            elif low[i] < pending["lower_poc_zone_low"] * (1.0 - cfg.lower_poc_invalidate_buffer_pct):
                log_event(i, "cancel_pending_break_lower_poc", pending)
                pending = None
                active = None
                cooldown_until = i + cfg.cooldown_bars
            elif high[i] >= pending["entry_trigger"]:
                sig[i] = 1
                entry_trigger[i] = pending["entry_trigger"]
                initial_stop[i] = pending["initial_stop"]
                target_price[i] = pending["target_price"]
                profile_vah[i] = pending["vah"]
                profile_val[i] = pending["val"]
                profile_poc[i] = pending["poc"]
                lower_poc[i] = pending["lower_poc"]
                absorption_low[i] = pending["absorption_low"]
                absorption_high[i] = pending["absorption_high"]
                bubble_count_arr[i] = pending["bubble_count"]
                setup_id_arr[i] = pending["setup_id"]
                signal_reason[i] = "anchored_vp_absorption_zone_breakout"
                log_event(i, "entry_triggered", pending)
                pending = None
                active = None
                cooldown_until = i + cfg.cooldown_bars
                continue

        if i < cooldown_until:
            continue

        # Cancel active profile if it becomes stale or the lower POC invalidation level fails before entry.
        if active is not None:
            if i - active.get("created_i", i) > cfg.max_active_profile_bars:
                log_event(i, "cancel_active_timeout", active)
                active = None
                pending = None
                cooldown_until = i + cfg.cooldown_bars
                continue
            if low[i] < active["lower_poc_zone_low"] * (1.0 - cfg.lower_poc_invalidate_buffer_pct):
                log_event(i, "cancel_active_break_lower_poc", active)
                active = None
                pending = None
                cooldown_until = i + cfg.cooldown_bars
                continue

            # Track structure confirmation: after a lower high is confirmed, price breaks above it.
            if not active.get("structure_confirmed", False):
                # Find a confirmed pivot high after leg low and below leg high.
                lh_candidates = [x for x in piv_highs if active["leg_low_i"] < x < i and high[x] < active["leg_high_price"]]
                if lh_candidates:
                    last_lh = lh_candidates[-1]
                    active["lower_high_i"] = int(last_lh)
                    active["lower_high_price"] = float(high[last_lh])
                    if close[i] > high[last_lh]:
                        active["structure_confirmed"] = True
                        log_event(i, "lower_high_broken", active)

            # Absorption region: VAL area down to lower POC zone.
            in_region = (
                low[i] <= active["val"] * (1.0 + cfg.val_tolerance_pct)
                and low[i] >= active["lower_poc_zone_low"] * (1.0 - cfg.lower_poc_invalidate_buffer_pct)
                and close[i] >= active["lower_poc_zone_low"]
            )
            sell_bubble = (
                np.isfinite(sell_q[i])
                and sell_notional[i] >= sell_q[i]
                and delta[i] < 0
            ) or (
                np.isfinite(large_sell_q[i])
                and large_sell_notional[i] >= large_sell_q[i]
                and delta[i] < 0
            )
            if in_region and sell_bubble:
                active["bubble_count"] += 1
                active["last_bubble_i"] = i
                active["absorption_low"] = min(active.get("absorption_low", low[i]), float(low[i]))
                active["absorption_high"] = max(active.get("absorption_high", high[i]), float(max(high[i], close[i], open_[i])))
                active.setdefault("bubble_bar_ids", []).append(int(bar_ids[i]))
                log_event(i, "sell_bubble_absorbed", active)

            # Once enough sell bubbles cluster in a narrow zone and structure is confirmed, place buy stop.
            if active is not None and pending is None and active.get("bubble_count", 0) >= cfg.min_absorption_bubbles:
                zone_low = active["absorption_low"]
                zone_high = active["absorption_high"]
                if zone_low > 0 and (zone_high - zone_low) / zone_low <= cfg.max_zone_width_pct and active.get("structure_confirmed", False):
                    trigger = zone_high * (1.0 + cfg.entry_buffer_pct)
                    stop = active["lower_poc_zone_low"] * (1.0 - cfg.stop_buffer_pct)
                    target = active["vah"]
                    risk = trigger - stop
                    reward = target - trigger
                    rr = reward / risk if risk > 0 else float("nan")
                    if risk > 0 and reward > 0 and rr >= cfg.min_reward_risk:
                        setup_seq += 1
                        pending = {
                            **active,
                            "setup_id": setup_seq,
                            "active_from_i": i,
                            "entry_trigger": float(trigger),
                            "initial_stop": float(stop),
                            "target_price": float(target),
                            "reward_risk": float(rr),
                        }
                        log_event(i, "buy_stop_placed", pending)
                    else:
                        reject_info = {**active, "entry_trigger": float(trigger), "initial_stop": float(stop), "target_price": float(target), "reward_risk": float(rr) if risk > 0 else float("nan")}
                        log_event(i, "reject_order_low_rr", reject_info)
                        active = None
                        pending = None
                        cooldown_until = i + cfg.cooldown_bars
                elif zone_low > 0 and (zone_high - zone_low) / zone_low > cfg.max_zone_width_pct:
                    reject_info = {**active, "zone_width_pct": float((zone_high - zone_low) / zone_low)}
                    log_event(i, "reject_order_zone_too_wide", reject_info)
                    active = None
                    pending = None
                    cooldown_until = i + cfg.cooldown_bars
                continue

        # Create/refresh anchored profile after a confirmed downleg.
        if active is None and len(piv_highs) >= 1 and len(piv_lows) >= 1:
            # Use latest confirmed pivot low as downleg low, and nearest pivot high before it as leg high.
            leg_low_i = piv_lows[-1]
            highs_before_low = [x for x in piv_highs if x < leg_low_i]
            lows_before_high: list[int]
            if not highs_before_low:
                continue
            leg_high_i = highs_before_low[-1]
            if high[leg_high_i] <= 0 or (high[leg_high_i] - low[leg_low_i]) / high[leg_high_i] < cfg.min_downmove_pct:
                continue
            lows_before_high = [x for x in piv_lows if x < leg_high_i]
            profile_start_i = (lows_before_high[-1] if lows_before_high else max(0, leg_high_i - cfg.profile_padding_bars))
            profile_start_i = max(0, profile_start_i - cfg.profile_padding_bars)
            # Avoid giant profiles in V1; this also avoids inadvertently making a broad curve-fit profile.
            if leg_low_i - profile_start_i > cfg.max_profile_bars:
                profile_start_i = leg_low_i - cfg.max_profile_bars
            key = (int(bar_ids[profile_start_i]), int(bar_ids[leg_low_i]))
            if key not in profile_cache:
                profile_cache[key] = compute_anchored_profile(fps, key[0], key[1], cfg)
            prof = profile_cache[key]
            if prof is None:
                continue
            # Only accept profiles where VAL is above lower POC and VAH offers room.
            if not (prof["lower_poc_zone_low"] < prof["val"] < prof["vah"]):
                continue
            active = {
                "profile_start_i": int(profile_start_i),
                "profile_end_i": int(leg_low_i),
                "profile_start_bar_id": int(bar_ids[profile_start_i]),
                "profile_end_bar_id": int(bar_ids[leg_low_i]),
                "profile_start_time": pd.Timestamp(end_ts[profile_start_i]),
                "profile_end_time": pd.Timestamp(end_ts[leg_low_i]),
                "leg_high_i": int(leg_high_i),
                "leg_low_i": int(leg_low_i),
                "leg_high_price": float(high[leg_high_i]),
                "leg_low_price": float(low[leg_low_i]),
                "created_i": int(i),
                "bubble_count": 0,
                "structure_confirmed": False,
                "absorption_low": float("inf"),
                "absorption_high": float("-inf"),
                "bubble_bar_ids": [],
                **prof,
            }
            log_event(i, "anchored_profile_created", active)

    out = b.copy()
    out["signal"] = sig
    out["entry_trigger"] = entry_trigger
    out["initial_stop"] = initial_stop
    out["target_price"] = target_price
    out["profile_vah"] = profile_vah
    out["profile_val"] = profile_val
    out["profile_poc"] = profile_poc
    out["lower_poc"] = lower_poc
    out["absorption_low"] = absorption_low
    out["absorption_high"] = absorption_high
    out["bubble_count"] = bubble_count_arr
    out["setup_id"] = setup_id_arr
    out["signal_reason"] = signal_reason
    return out, pd.DataFrame(events)


def run_backtest(signals: pd.DataFrame, cfg: Config) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    start = _to_timestamp(cfg.start_date)
    end = _to_timestamp(cfg.end_date)
    d = signals[(signals["end_ts"] >= start) & (signals["start_ts"] <= end)].copy().reset_index(drop=True)
    n = len(d)
    if n == 0:
        raise RuntimeError("No signal bars inside requested backtest range")

    open_a = d["open"].to_numpy(float)
    high_a = d["high"].to_numpy(float)
    low_a = d["low"].to_numpy(float)
    close_a = d["close"].to_numpy(float)
    end_ts = pd.to_datetime(d["end_ts"]).to_numpy()
    sig = d["signal"].to_numpy(int)
    trigger_a = d["entry_trigger"].to_numpy(float)
    stop_a = d["initial_stop"].to_numpy(float)
    target_a = d["target_price"].to_numpy(float)
    poc_a = d["profile_poc"].to_numpy(float)
    abs_hi_a = d["absorption_high"].to_numpy(float)
    setup_id_a = d["setup_id"].to_numpy(int)

    capital = cfg.initial_capital
    peak = capital
    pos: dict[str, Any] | None = None
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for i in range(n):
        ts = pd.Timestamp(end_ts[i])
        mark_eq = capital if pos is None else capital + (close_a[i] - pos["entry"]) * pos["qty"]
        peak = max(peak, mark_eq)
        equity_rows.append({"timestamp": ts, "equity": mark_eq, "drawdown_pct": (peak - mark_eq) / peak if peak > 0 else 0.0})

        # Existing long position. Exit checks use stop from prior bars; stop updates happen after exit checks.
        if pos is not None:
            pos["mfe"] = max(pos["mfe"], high_a[i] - pos["entry"])
            pos["mae"] = max(pos["mae"], pos["entry"] - low_a[i])
            exit_price = None
            exit_reason = None
            if low_a[i] <= pos["stop"]:
                exit_price = pos["stop"]
                exit_reason = "stop"
            elif high_a[i] >= pos["target"]:
                exit_price = pos["target"]
                exit_reason = "target_vah"
            elif i - pos["entry_i"] >= cfg.max_holding_bars:
                exit_price = close_a[i]
                exit_reason = "time"

            if exit_price is not None:
                fill = apply_exit_slippage(float(exit_price), 1, cfg.slippage_pct)
                fee = pos["entry_fee"] + abs(pos["qty"] * fill) * cfg.fee_rate
                gross = (fill - pos["entry"]) * pos["qty"]
                pnl = gross - fee
                before = capital
                capital += pnl
                trades.append(
                    {
                        "entry_time": pos["entry_time"],
                        "exit_time": ts,
                        "type": "LONG",
                        "entry": pos["entry"],
                        "exit": fill,
                        "pnl": pnl,
                        "fee": fee,
                        "capital": capital,
                        "return_pct": pnl / max(before, 1e-12),
                        "mfe_r": pos["mfe"] / pos["risk_dist"] if pos["risk_dist"] > 0 else 0.0,
                        "mae_r": pos["mae"] / pos["risk_dist"] if pos["risk_dist"] > 0 else 0.0,
                        "holding_hours": (ts - pos["entry_time"]).total_seconds() / 3600,
                        "exit_reason": exit_reason,
                        "setup_id": pos["setup_id"],
                        "target_vah": pos["target"],
                        "profile_poc": pos["profile_poc"],
                        "initial_stop": pos["initial_stop"],
                    }
                )
                pos = None
                continue

            # Stop management: updates after bar close, effective from the next bar.
            r = pos["risk_dist"]
            if r > 0 and high_a[i] >= pos["entry"] + cfg.move_stop_to_zone_after_r * r:
                pos["stop"] = max(pos["stop"], pos["absorption_high"] * (1.0 - cfg.trail_retest_buffer_pct))
            if np.isfinite(pos["profile_poc"]) and close_a[i] > pos["profile_poc"] and low_a[i] <= pos["profile_poc"]:
                pos["stop"] = max(pos["stop"], pos["profile_poc"] * (1.0 - cfg.trail_retest_buffer_pct))

        # Entry from triggered stop-market signal. Exits start on following bars to avoid same-bar ambiguity.
        if pos is None and sig[i] == 1:
            trigger = trigger_a[i]
            stop = stop_a[i]
            target = target_a[i]
            if not (np.isfinite(trigger) and np.isfinite(stop) and np.isfinite(target)):
                continue
            raw_entry = max(open_a[i], trigger)
            entry = apply_entry_slippage(float(raw_entry), 1, cfg.slippage_pct)
            risk_dist = entry - stop
            reward_dist = target - entry
            if risk_dist <= 0 or reward_dist <= 0 or reward_dist / risk_dist < cfg.min_reward_risk:
                continue
            qty = capital * cfg.unit_risk_per_trade / risk_dist
            notional = abs(qty * entry)
            max_notional = capital * cfg.max_notional_mult
            if notional > max_notional:
                qty *= max_notional / notional
                notional = max_notional
            if qty <= 0 or notional <= 0:
                continue
            pos = {
                "entry": float(entry),
                "stop": float(stop),
                "initial_stop": float(stop),
                "target": float(target),
                "risk_dist": float(risk_dist),
                "qty": float(qty),
                "entry_fee": float(notional * cfg.fee_rate),
                "entry_time": ts,
                "entry_i": i,
                "mfe": 0.0,
                "mae": 0.0,
                "setup_id": int(setup_id_a[i]),
                "profile_poc": float(poc_a[i]) if np.isfinite(poc_a[i]) else np.nan,
                "absorption_high": float(abs_hi_a[i]) if np.isfinite(abs_hi_a[i]) else float(stop),
            }

    if pos is not None:
        ts = pd.Timestamp(end_ts[-1])
        fill = apply_exit_slippage(float(close_a[-1]), 1, cfg.slippage_pct)
        fee = pos["entry_fee"] + abs(pos["qty"] * fill) * cfg.fee_rate
        gross = (fill - pos["entry"]) * pos["qty"]
        pnl = gross - fee
        before = capital
        capital += pnl
        trades.append(
            {
                "entry_time": pos["entry_time"],
                "exit_time": ts,
                "type": "LONG",
                "entry": pos["entry"],
                "exit": fill,
                "pnl": pnl,
                "fee": fee,
                "capital": capital,
                "return_pct": pnl / max(before, 1e-12),
                "mfe_r": pos["mfe"] / pos["risk_dist"] if pos["risk_dist"] > 0 else 0.0,
                "mae_r": pos["mae"] / pos["risk_dist"] if pos["risk_dist"] > 0 else 0.0,
                "holding_hours": (ts - pos["entry_time"]).total_seconds() / 3600,
                "exit_reason": "final",
                "setup_id": pos["setup_id"],
                "target_vah": pos["target"],
                "profile_poc": pos["profile_poc"],
                "initial_stop": pos["initial_stop"],
            }
        )

    equity = pd.DataFrame(equity_rows).set_index("timestamp") if equity_rows else pd.DataFrame(columns=["equity", "drawdown_pct"])
    summary = summarize_r_trades(trades, equity, cfg.initial_capital)
    summary.update(
        {
            "signal_count": int((d["signal"] == 1).sum()),
            "range_pct": cfg.range_pct,
            "price_step": cfg.price_step,
            "fee_rate_per_side": cfg.fee_rate,
            "slippage_pct": cfg.slippage_pct,
            "unit_risk_per_trade": cfg.unit_risk_per_trade,
        }
    )
    return trades, equity, summary


def write_outputs(signals: pd.DataFrame, events: pd.DataFrame, trades: list[dict[str, Any]], equity: pd.DataFrame, summary: dict[str, Any], cfg: Config) -> None:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(cfg.out_dir / "eth_range_anchored_vp_absorption_v1_trades.csv", index=False)
    equity.to_csv(cfg.out_dir / "eth_range_anchored_vp_absorption_v1_equity.csv")
    events.to_csv(cfg.out_dir / "eth_range_anchored_vp_absorption_v1_setup_events.csv", index=False)
    sigs = signals[signals["signal"] == 1].copy()
    audit_cols = [
        "end_ts",
        "bar_id",
        "open",
        "high",
        "low",
        "close",
        "setup_id",
        "entry_trigger",
        "initial_stop",
        "target_price",
        "profile_vah",
        "profile_val",
        "profile_poc",
        "lower_poc",
        "absorption_low",
        "absorption_high",
        "bubble_count",
        "signal_reason",
    ]
    sigs[[c for c in audit_cols if c in sigs.columns]].to_csv(cfg.out_dir / "eth_range_anchored_vp_absorption_v1_signal_audit.csv", index=False)
    if cfg.write_full_audit:
        signals.to_csv(cfg.out_dir / "eth_range_anchored_vp_absorption_v1_full_audit.csv", index=False)
    with (cfg.out_dir / "eth_range_anchored_vp_absorption_v1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 100)
    print(f"{STRATEGY_NAME} Backtest Summary")
    print("=" * 100)
    for k, v in summary.items():
        print(f"{k:>32}: {v}")
    print("-" * 100)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 100 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=STRATEGY_NAME, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--preset", choices=sorted(PRESETS), default="high")
    p.add_argument("--range-pct", type=float, default=0.0020)
    p.add_argument("--price-step", type=float, default=1.0)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.00015)
    p.add_argument("--write-full-audit", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        warmup_start_date=args.warmup_start_date,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        range_pct=args.range_pct,
        price_step=args.price_step,
        data_dir=args.data_dir,
        write_full_audit=args.write_full_audit,
    )
    for k, v in PRESETS[args.preset].items():
        setattr(cfg, k, v)
    cfg.out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/reports/mf/eth_range_anchored_vp_absorption_v1/{args.preset}_r{int(cfg.range_pct*10000):04d}_step{cfg.price_step:g}")

    bars, fps = load_range_data(cfg)
    print(f"Loaded range bars: {len(bars):,} | {bars['end_ts'].min()} -> {bars['end_ts'].max()}")
    print(f"Loaded footprints: {len(fps):,} | {fps['end_ts'].min()} -> {fps['end_ts'].max()}")
    signals, events = generate_setups_and_signals(bars, fps, cfg)
    print(f"Signals generated: {int((signals['signal'] == 1).sum()):,}")
    trades, equity, summary = run_backtest(signals, cfg)
    write_outputs(signals, events, trades, equity, summary, cfg)
    print_summary(summary, cfg.out_dir)

    report_df = signals[(signals["end_ts"] >= pd.Timestamp(cfg.start_date)) & (signals["start_ts"] <= pd.Timestamp(cfg.end_date))].copy()
    report_df = report_df.set_index("end_ts", drop=False)
    final_capital = float(trades[-1]["capital"]) if trades else cfg.initial_capital
    if len(report_df) > 1:
        total_days = max((report_df.index[-1] - report_df.index[0]).total_seconds() / 86400.0, 1.0)
    else:
        total_days = 1.0
    print_full_report(
        build_report_trades(trades),
        report_df,
        cfg.initial_capital,
        final_capital,
        STRATEGY_NAME,
        total_days,
        False,
        symbol=cfg.symbol,
        report_dir=cfg.out_dir,
    )


if __name__ == "__main__":
    main()
