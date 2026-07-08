#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LF V10B meta-selector and sleeve runner."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.backtest_common.data import load_ohlcv_data
from src.edge_lib.lf_bear_short.features import build_bear_features
from src.edge_lib.lf_bull_range_reclaim.features import build_features as build_bull_features
from src.edge_lib.lf_momentum_breakout.features import build_features as build_momentum_features
from src.portfolio_common.allocator import LF_LEG, attach_lf_position_metrics
from src.sleeve_lib.lf_v10b.config import (
    PRIORITY_MODES,
    build_lf_args,
    bull_to_exec_config,
    make_bear_config,
    make_bull_config,
    make_exec_config,
    make_momentum_config,
    priority_map,
)
from src.sleeve_lib.lf_v10b.micro_filter import (
    apply_micro_context_filter,
    apply_momentum_long_not_aligned_block,
    apply_momentum_short_fast_speed_block,
    load_range_footprint_context,
)


def _bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype("boolean").fillna(False).astype(bool)


def select_portfolio_signals(momentum: pd.DataFrame, bear: pd.DataFrame, bull: pd.DataFrame, args: Any) -> pd.DataFrame:
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
    args: Any | None,
) -> tuple[bool, str, dict[str, float | bool | str]]:
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


def run_priority_backtest(*_args: Any, **_kwargs: Any) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    raise NotImplementedError("LF V10B Portfolio V1 uses the structural-stop executor.")


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
        item.setdefault(
            "range_exit_reason",
            str(item.get("note", "")) if str(item.get("note", "")).startswith("RANGE_EXIT") else "",
        )
        out.append(item)
    return out


def run_lf_v10b_leg(args: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from src.sleeve_lib.lf_v10b import structural_stop

    lf_args = build_lf_args(args)
    mom_cfg = make_momentum_config(lf_args)
    bear_cfg = make_bear_config(lf_args)
    bull_cfg = make_bull_config(lf_args)
    exec_cfg = make_exec_config(mom_cfg)
    bull_exec_cfg = bull_to_exec_config(bull_cfg) if lf_args.bull_execution_mode == "own" else exec_cfg

    trade_start = pd.Timestamp(args.start_date)
    load_start = pd.Timestamp(args.warmup_start_date or args.start_date)
    load_start_str = load_start.strftime("%Y-%m-%d")

    print(f"[lf] load 4H {args.symbol} {load_start_str}->{args.end_date}; trade_start={args.start_date}", flush=True)
    base = load_ohlcv_data(args.symbol, load_start_str, args.end_date, "4H")
    print(f"[lf] 4H rows={len(base):,}", flush=True)
    momentum = build_momentum_features(base, mom_cfg)
    bear = build_bear_features(base, bear_cfg)
    bull = build_bull_features(base, bull_cfg)
    micro_ctx = load_range_footprint_context(lf_args, load_start_str, args.end_date)
    momentum = apply_momentum_long_not_aligned_block(momentum, micro_ctx, lf_args)
    momentum = apply_momentum_short_fast_speed_block(momentum, micro_ctx, lf_args)
    features = select_portfolio_signals(momentum, bear, bull, lf_args)
    features = apply_micro_context_filter(features, micro_ctx, lf_args)

    before_slice_rows = len(features)
    features = structural_stop.add_structural_columns(
        features,
        lookback_bars=structural_stop.V10B_STRUCTURAL_STOP.lookback_bars,
    )
    features = features.loc[trade_start : pd.Timestamp(args.end_date)].copy()
    first_bar = features.index[0] if not features.empty else "NA"
    print(f"[lf] feature rows after warmup slice: {len(features):,}/{before_slice_rows:,}; first={first_bar}", flush=True)

    trades, equity = structural_stop.run_v10b_backtest(
        features,
        exec_cfg,
        engine_cfgs={"MOMENTUM_V3": exec_cfg, "BEAR_V3_ONLY": exec_cfg, "BULL_RECLAIM_V2": bull_exec_cfg},
        global_risk_scale=lf_args.global_risk_scale,
        args=lf_args,
    )
    trades = attach_engine_to_trades(trades, features)
    out = pd.DataFrame(trades)
    if out.empty:
        return pd.DataFrame(), equity, features
    out = out.copy()
    out["strategy_leg"] = LF_LEG
    out["variant_name"] = LF_LEG
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
    out["side"] = pd.to_numeric(out.get("side", np.nan), errors="coerce")
    out["return_on_sleeve"] = pd.to_numeric(out.get("return_pct", out.get("pnl", 0.0)), errors="coerce")
    if "return_pct" not in out.columns and {"pnl", "capital"}.issubset(out.columns):
        cap_after = pd.to_numeric(out["capital"], errors="coerce")
        pnl = pd.to_numeric(out["pnl"], errors="coerce")
        out["return_on_sleeve"] = pnl / (cap_after - pnl).replace(0, np.nan)
    out["exit_reason"] = out.get("note", "")
    out = attach_lf_position_metrics(out)
    return out.dropna(subset=["entry_time", "exit_time", "return_on_sleeve"]), equity, features

