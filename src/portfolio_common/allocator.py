#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Execution sizing helpers and Portfolio V1 allocation rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage

PORTFOLIO_ID = "ETH_PORTFOLIO_V1"
LF_SLEEVE_ID = "ETH_SLEEVE_LF_V10B"
MF_SLEEVE_ID = "ETH_SLEEVE_MF_LOW_SWEEP_V1"
LF_LEG = "LF_V10B"
MF_TIME48_LEG = "MF_LOW_SWEEP_TIME48"
DEFAULT_LEVERAGE = 15.0

EDGE_ID_BY_ENGINE = {
    "MOMENTUM_V3": "ETH_EDGE_LF_MOMENTUM_BREAKOUT_V3",
    "BEAR_V3_ONLY": "ETH_EDGE_LF_BEAR_SHORT_V3",
    "BULL_RECLAIM_V2": "ETH_EDGE_LF_BULL_RANGE_RECLAIM_V2",
}
MF_EDGE_ID = "ETH_EDGE_MF_LOW_SWEEP_A0_FOOTPRINT"


@dataclass(frozen=True)
class StrategyConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "4H"
    entry_lookback: int = 40
    exit_lookback: int = 36
    atr_period: int = 20
    adx_period: int = 14
    min_adx_long: float = 8.0
    min_adx_short: float = 18.0
    min_atr_pct: float = 0.0030
    max_atr_pct: float = 0.0800
    d1_ema_fast: int = 8
    d1_ema_slow: int = 30
    d1_slope_lookback: int = 10
    bull_slope_min: float = -0.0300
    short_slope_max: float = -0.0030
    initial_atr_mult: float = 2.5
    trailing_atr_mult: float = 4.5
    unit_risk_per_trade: float = 0.0060
    max_units: int = 3
    add_every_r: float = 1.0
    max_total_notional_mult: float = 5.0
    min_risk_mult: float = 0.35
    max_risk_mult: float = 1.50
    strong_adx: float = 20.0
    very_strong_adx: float = 30.0
    strong_d1_slope_abs: float = 0.015
    strong_price_distance_pct: float = 0.035
    high_atr_pct: float = 0.045
    weak_adx: float = 12.0
    weak_d1_slope_abs: float = 0.004
    breakeven_after_r: float = 1.0
    breakeven_lock_r: float = 0.10
    lock_after_2r: float = 1.7
    lock_2r: float = 0.70
    lock_after_3r: float = 2.8
    lock_3r: float = 1.50
    no_progress_bars: int = 10000
    no_progress_min_r: float = 0.0
    max_hold_bars: int = 360
    cooldown_bars: int = 8
    enable_short: bool = True
    initial_capital: float = 1000.0
    fee_rate: float = 0.0005
    slippage_pct: float = 0.0002


@dataclass(frozen=True)
class PortfolioScenario:
    scenario_name: str
    mf_variant_name: str
    mf_exposure: float
    conflict_mode: str
    lf_weight: float = 1.0
    guard_mode: str = "none"
    margin_cap: float | None = None
    notional_cap: float | None = None
    min_mf_exposure: float = 0.05


def parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if math.isfinite(value):
            out.append(value)
    return sorted(set(out))


def split_csv(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def safe_float(value: object, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def weighted_avg_price(old_price: float, old_qty: float, add_price: float, add_qty: float) -> float:
    total = old_qty + add_qty
    if total <= 0:
        return add_price
    return (old_price * old_qty + add_price * add_qty) / total


def unit_qty(
    capital: float,
    entry_price: float,
    stop_dist: float,
    current_qty: float,
    cfg: StrategyConfig,
    risk_mult: float = 1.0,
) -> float:
    if stop_dist <= 0:
        return 0.0
    effective_risk = cfg.unit_risk_per_trade * max(cfg.min_risk_mult, min(float(risk_mult), cfg.max_risk_mult))
    risk_qty = capital * effective_risk / stop_dist
    max_total_qty = (capital * cfg.max_total_notional_mult) / entry_price
    remaining_qty = max(0.0, max_total_qty - current_qty)
    return max(0.0, min(risk_qty, remaining_qty))


def protected_stop(
    first_entry: float,
    avg_entry: float,
    side: int,
    risk_per_coin: float,
    max_fav: float,
    cfg: StrategyConfig,
) -> float | None:
    if risk_per_coin <= 0:
        return None
    fav_r = (max_fav - first_entry) / risk_per_coin if side == 1 else (first_entry - max_fav) / risk_per_coin
    lock_r: float | None = None
    avg_lock_r: float | None = None
    if fav_r >= cfg.lock_after_3r:
        lock_r = cfg.lock_3r
        avg_lock_r = 0.50
    elif fav_r >= cfg.lock_after_2r:
        lock_r = cfg.lock_2r
        avg_lock_r = 0.00
    elif fav_r >= cfg.breakeven_after_r:
        lock_r = cfg.breakeven_lock_r
        avg_lock_r = None
    if lock_r is None:
        return None
    first_based = first_entry + side * lock_r * risk_per_coin
    if avg_lock_r is None:
        return first_based
    avg_based = avg_entry + side * avg_lock_r * risk_per_coin
    if side == 1:
        return max(first_based, avg_based)
    return min(first_based, avg_based)


def close_trade(
    *,
    trades: list[dict[str, Any]],
    capital: float,
    side: int,
    entry_time: Any,
    exit_time: Any,
    first_entry: float,
    avg_entry: float,
    exit_price: float,
    initial_sl: float,
    stop_price: float,
    qty: float,
    units: int,
    total_entry_fee: float,
    fee_rate: float,
    max_fav: float,
    max_adv: float,
    risk_per_coin: float,
    holding_bars: int,
    reason: str,
    risk_mult: float = 1.0,
) -> float:
    exit_fee = qty * exit_price * fee_rate
    if side == 1:
        pnl = (exit_price - avg_entry) * qty - total_entry_fee - exit_fee
        mfe_r = (max_fav - first_entry) / risk_per_coin
        mae_r = (first_entry - max_adv) / risk_per_coin
    else:
        pnl = (avg_entry - exit_price) * qty - total_entry_fee - exit_fee
        mfe_r = (first_entry - max_fav) / risk_per_coin
        mae_r = (max_adv - first_entry) / risk_per_coin
    cap_before = capital
    capital += pnl
    trades.append(
        {
            "entry_time": entry_time,
            "exit_time": exit_time,
            "type": "LONG" if side == 1 else "SHORT",
            "first_entry": first_entry,
            "avg_entry": avg_entry,
            "exit": exit_price,
            "initial_sl": initial_sl,
            "final_sl": stop_price,
            "qty": qty,
            "units": units,
            "pnl": pnl,
            "fee": total_entry_fee + exit_fee,
            "capital": capital,
            "return_pct": pnl / max(cap_before, 1e-12),
            "mfe_r": round(float(mfe_r), 4),
            "mae_r": round(float(mae_r), 4),
            "sl_pct": round(abs(first_entry - initial_sl) / first_entry * 100, 4),
            "holding_bars_4h": int(holding_bars),
            "holding_hours": int(holding_bars * 4),
            "risk_mult": round(float(risk_mult), 4),
            "note": reason,
        }
    )
    return capital


def summarize_exec_trades(trades: list[dict[str, Any]], equity: pd.DataFrame, initial_capital: float) -> dict[str, Any]:
    if not trades:
        return {"total_trades": 0, "final_capital": initial_capital, "total_return_pct": 0.0}
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["pnl"].sum()) if not losses.empty else 0.0
    final_capital = float(tdf.iloc[-1]["capital"])
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {
        "total_trades": int(len(tdf)),
        "long_trades": int((tdf["type"] == "LONG").sum()),
        "short_trades": int((tdf["type"] == "SHORT").sum()),
        "final_capital": round(final_capital, 4),
        "total_return_pct": round((final_capital / initial_capital - 1) * 100, 4),
        "win_rate": round(float((tdf["pnl"] > 0).mean() * 100), 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "profit_factor": round(pf, 4) if math.isfinite(pf) else "inf",
        "expectancy_pct": round(float(tdf["return_pct"].mean() * 100), 6),
        "max_drawdown_pct": round(float(equity["drawdown_pct"].max() * 100), 4) if not equity.empty else 0.0,
        "avg_mfe_r": round(float(tdf["mfe_r"].mean()), 4),
        "avg_mae_r": round(float(tdf["mae_r"].mean()), 4),
        "avg_units": round(float(tdf["units"].mean()), 4),
        "avg_risk_mult": round(float(tdf["risk_mult"].mean()), 4) if "risk_mult" in tdf.columns else 1.0,
        "avg_holding_hours": round(float(tdf["holding_hours"].mean()), 2),
        "total_fees": round(float(tdf["fee"].sum()), 4),
    }


def side_from_trade_row(row: pd.Series) -> int:
    side = pd.to_numeric(row.get("side", np.nan), errors="coerce")
    if pd.notna(side) and math.isfinite(float(side)) and float(side) != 0:
        return int(np.sign(float(side)))
    typ = str(row.get("type", "")).upper()
    if "LONG" in typ:
        return 1
    if "SHORT" in typ:
        return -1
    return 0


def active_lf_trade_at(ts: pd.Timestamp, lf_trades: pd.DataFrame) -> pd.Series | None:
    if lf_trades.empty:
        return None
    active = lf_trades.loc[(lf_trades["entry_time"] <= ts) & (lf_trades["exit_time"] > ts)]
    if active.empty:
        return None
    return active.sort_values("entry_time").iloc[-1]


def lf_active_side_at(ts: pd.Timestamp, lf_trades: pd.DataFrame) -> int:
    row = active_lf_trade_at(ts, lf_trades)
    if row is None:
        return 0
    return side_from_trade_row(row)


def attach_lf_position_metrics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    cap_after = pd.to_numeric(out.get("capital", np.nan), errors="coerce")
    pnl = pd.to_numeric(out.get("pnl", np.nan), errors="coerce")
    cap_before = (cap_after - pnl).replace(0, np.nan)
    qty = pd.to_numeric(out.get("qty", np.nan), errors="coerce").abs()
    entry = pd.to_numeric(out.get("avg_entry", out.get("first_entry", np.nan)), errors="coerce")
    first_entry = pd.to_numeric(out.get("first_entry", entry), errors="coerce")
    initial_sl = pd.to_numeric(out.get("initial_sl", np.nan), errors="coerce")
    notional = qty * entry
    out["capital_before_trade"] = cap_before
    out["position_notional"] = notional
    out["position_notional_mult"] = notional / cap_before
    out["initial_risk_amount"] = qty * (first_entry - initial_sl).abs()
    out["initial_risk_pct_equity"] = out["initial_risk_amount"] / cap_before
    out["margin_fraction_at_leverage15"] = out["position_notional_mult"] / DEFAULT_LEVERAGE
    return out


def attach_mf_position_metrics(trades: pd.DataFrame, assumed_exposure: float = 1.0) -> pd.DataFrame:
    if trades.empty:
        return trades
    out = trades.copy()
    out["position_notional_mult"] = float(assumed_exposure)
    out["initial_risk_pct_equity"] = np.nan
    out["margin_fraction_at_leverage15"] = float(assumed_exposure) / DEFAULT_LEVERAGE
    return out


def apply_conflict_filter(mf: pd.DataFrame, lf: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, int]:
    if mf.empty or mode == "independent":
        return mf.copy(), 0
    keep = []
    skipped = 0
    for _, row in mf.sort_values("entry_time").iterrows():
        side = lf_active_side_at(pd.Timestamp(row["entry_time"]), lf)
        if mode == "skip_when_lf_active" and side != 0:
            skipped += 1
            continue
        if mode == "skip_if_lf_short" and side < 0:
            skipped += 1
            continue
        keep.append(row)
    return pd.DataFrame(keep), int(skipped)


def parse_guard_mode(mode: str, args: Any) -> tuple[str, float | None, float | None]:
    raw = str(mode).strip().lower()
    if raw in {"", "none", "no_guard", "noguard"}:
        return "none", None, None
    margin_cap: float | None = None
    notional_cap: float | None = None
    for part in raw.split("_"):
        if part.startswith("margin"):
            suffix = part.replace("margin", "", 1)
            margin_cap = float(suffix) / 100.0 if suffix else float(args.guard_margin_cap)
        elif part.startswith("notional"):
            suffix = part.replace("notional", "", 1)
            notional_cap = float(suffix) if suffix else float(args.guard_notional_cap)
        elif part:
            raise ValueError(f"Unknown guard mode part: {part!r} in {mode!r}")
    normalized_parts: list[str] = []
    if margin_cap is not None:
        normalized_parts.append(f"margin{int(round(margin_cap * 100))}")
    if notional_cap is not None:
        normalized_parts.append(f"notional{notional_cap:g}")
    if not normalized_parts:
        return "none", None, None
    return "_".join(normalized_parts), margin_cap, notional_cap


def scenario_name(base: str, guard_mode: str) -> str:
    if guard_mode in {"", "none"}:
        return base
    return f"{base}_guard_{guard_mode.replace('.', 'p')}"


def build_scenarios(args: Any) -> list[PortfolioScenario]:
    scenarios: list[PortfolioScenario] = []
    guard_specs = [parse_guard_mode(mode, args) for mode in split_csv(args.guard_modes)]
    for exposure in parse_float_list(args.mf_exposures):
        for mode in split_csv(args.conflict_modes):
            if mode not in {"independent", "skip_when_lf_active", "skip_if_lf_short"}:
                raise ValueError(f"Unknown conflict mode: {mode}")
            for guard_mode, margin_cap, notional_cap in guard_specs:
                base = f"portfolio_v1_lf{int(round(args.lf_weight * 100))}_mf{int(round(exposure * 100))}_time48_{mode}"
                scenarios.append(
                    PortfolioScenario(
                        scenario_name=scenario_name(base, guard_mode),
                        mf_variant_name=MF_TIME48_LEG,
                        mf_exposure=float(exposure),
                        conflict_mode=mode,
                        lf_weight=float(args.lf_weight),
                        guard_mode=guard_mode,
                        margin_cap=margin_cap,
                        notional_cap=notional_cap,
                        min_mf_exposure=float(args.min_mf_exposure),
                    )
                )
    return scenarios


def lf_notional_at(ts: pd.Timestamp, lf: pd.DataFrame) -> float:
    active = active_lf_trade_at(ts, lf)
    if active is None:
        return 0.0
    return max(0.0, safe_float(active.get("position_notional_mult", 0.0), 0.0))


def apply_mf_sizing_guard(
    mf: pd.DataFrame,
    lf: pd.DataFrame,
    scenario: PortfolioScenario,
    *,
    leverage: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    stats: dict[str, object] = {
        "mf_guard_scaled_count": 0,
        "mf_guard_skipped_count": 0,
        "mf_exposure_actual_avg": float("nan"),
        "mf_exposure_actual_min": float("nan"),
        "mf_exposure_actual_max": float("nan"),
        "max_combined_notional_after_mf": float("nan"),
        "max_combined_margin_after_mf": float("nan"),
    }
    if mf.empty:
        return mf.copy(), stats
    target = float(scenario.mf_exposure)
    min_exp = max(0.0, float(scenario.min_mf_exposure))
    kept: list[pd.Series] = []
    actuals: list[float] = []
    combined_notional_values: list[float] = []
    combined_margin_values: list[float] = []
    for _, row in mf.sort_values("entry_time").iterrows():
        ts = pd.Timestamp(row["entry_time"])
        lf_notional = lf_notional_at(ts, lf)
        allowed = target
        if scenario.notional_cap is not None:
            allowed = min(allowed, max(0.0, float(scenario.notional_cap) - lf_notional))
        if scenario.margin_cap is not None:
            allowed = min(allowed, max(0.0, float(scenario.margin_cap) * float(leverage) - lf_notional))
        if allowed + 1e-12 < min_exp:
            stats["mf_guard_skipped_count"] = int(stats["mf_guard_skipped_count"]) + 1
            continue
        actual = max(0.0, min(target, allowed))
        if actual < target - 1e-12:
            stats["mf_guard_scaled_count"] = int(stats["mf_guard_scaled_count"]) + 1
        out = row.copy()
        out["guard_mode"] = scenario.guard_mode
        out["guard_target_mf_exposure"] = target
        out["guard_actual_mf_exposure"] = actual
        out["guard_scale"] = actual / target if target > 0 else 0.0
        out["guard_lf_notional_at_entry"] = lf_notional
        out["guard_combined_notional_after_mf"] = lf_notional + actual
        out["guard_combined_margin_after_mf"] = (lf_notional + actual) / float(leverage)
        kept.append(out)
        actuals.append(actual)
        combined_notional_values.append(lf_notional + actual)
        combined_margin_values.append((lf_notional + actual) / float(leverage))
    out_df = pd.DataFrame(kept)
    if actuals:
        stats["mf_exposure_actual_avg"] = float(np.mean(actuals))
        stats["mf_exposure_actual_min"] = float(np.min(actuals))
        stats["mf_exposure_actual_max"] = float(np.max(actuals))
        stats["max_combined_notional_after_mf"] = float(np.max(combined_notional_values))
        stats["max_combined_margin_after_mf"] = float(np.max(combined_margin_values))
    return out_df, stats


def profit_factor(returns: pd.Series) -> float:
    ret = pd.to_numeric(returns, errors="coerce").dropna()
    gains = float(ret[ret > 0].sum())
    losses = float(-ret[ret < 0].sum())
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def max_drawdown(capital: pd.Series) -> float:
    if capital.empty:
        return float("nan")
    peak = capital.cummax()
    dd = capital / peak - 1.0
    return float(dd.min())


def simulate_portfolio_scenario(
    lf: pd.DataFrame,
    mf_all: pd.DataFrame,
    scenario: PortfolioScenario,
    *,
    initial_capital: float,
    leverage: float = DEFAULT_LEVERAGE,
) -> tuple[pd.DataFrame, dict[str, object]]:
    lf_part = lf.copy()
    if "return_on_sleeve" not in lf_part.columns:
        lf_part["return_on_sleeve"] = pd.Series(dtype="float64")
    lf_part["scenario"] = scenario.scenario_name
    lf_part["portfolio_return"] = pd.to_numeric(lf_part["return_on_sleeve"], errors="coerce") * float(scenario.lf_weight)
    lf_part["portfolio_weight"] = float(scenario.lf_weight)
    lf_part["position_scope"] = "lf_v10b_sleeve"
    lf_part["close_scope"] = "lf_v10b_only"
    lf_part["scenario_notional_exposure"] = pd.to_numeric(lf_part.get("position_notional_mult", np.nan), errors="coerce") * float(scenario.lf_weight)

    if mf_all.empty or "variant_name" not in mf_all.columns:
        mf_raw = pd.DataFrame()
    else:
        mf_raw = mf_all.loc[mf_all["variant_name"].eq(scenario.mf_variant_name)].copy()
    mf_part, skipped = apply_conflict_filter(mf_raw, lf, scenario.conflict_mode)
    mf_part, guard_stats = apply_mf_sizing_guard(mf_part, lf, scenario, leverage=float(leverage))
    if not mf_part.empty:
        actual_exp = pd.to_numeric(mf_part.get("guard_actual_mf_exposure", scenario.mf_exposure), errors="coerce").fillna(float(scenario.mf_exposure))
        mf_part["scenario"] = scenario.scenario_name
        mf_part["portfolio_return"] = pd.to_numeric(mf_part["return_on_sleeve"], errors="coerce") * actual_exp
        mf_part["portfolio_weight"] = actual_exp
        mf_part["position_scope"] = "mf_low_sweep_sleeve"
        mf_part["close_scope"] = "mf_low_sweep_only"
        mf_part["scenario_notional_exposure"] = actual_exp

    combined = pd.concat([lf_part, mf_part], ignore_index=True, sort=False)
    if combined.empty:
        return combined, {
            "scenario": scenario.scenario_name,
            "total_trades": 0,
            "return_total": 0.0,
            "max_drawdown": 0.0,
            "mf_skipped_by_conflict": int(skipped),
            "guard_mode": scenario.guard_mode,
            "margin_cap": scenario.margin_cap,
            "notional_cap": scenario.notional_cap,
            **guard_stats,
        }
    combined["exit_time"] = pd.to_datetime(combined["exit_time"], errors="coerce")
    combined["entry_time"] = pd.to_datetime(combined["entry_time"], errors="coerce")
    combined = combined.dropna(subset=["entry_time", "exit_time", "portfolio_return"]).sort_values(["exit_time", "strategy_leg", "entry_time"]).reset_index(drop=True)

    capital = float(initial_capital)
    caps: list[float] = []
    pnls: list[float] = []
    for _, row in combined.iterrows():
        ret = float(row["portfolio_return"])
        pnl = capital * ret
        capital += pnl
        pnls.append(pnl)
        caps.append(capital)
    combined["portfolio_pnl"] = pnls
    combined["portfolio_capital"] = caps

    ret = pd.to_numeric(combined["portfolio_return"], errors="coerce")
    summary = {
        "scenario": scenario.scenario_name,
        "lf_weight": float(scenario.lf_weight),
        "mf_variant_name": scenario.mf_variant_name,
        "mf_exposure": float(scenario.mf_exposure),
        "mf_margin_fraction_at_15x": float(scenario.mf_exposure) / DEFAULT_LEVERAGE,
        "conflict_mode": scenario.conflict_mode,
        "guard_mode": scenario.guard_mode,
        "margin_cap": scenario.margin_cap,
        "notional_cap": scenario.notional_cap,
        "min_mf_exposure": float(scenario.min_mf_exposure),
        "total_trades": int(len(combined)),
        "lf_trades": int(combined["strategy_leg"].eq(LF_LEG).sum()),
        "mf_trades": int(combined["strategy_leg"].eq(MF_TIME48_LEG).sum()),
        "mf_skipped_by_conflict": int(skipped),
        **guard_stats,
        "return_total": float(capital / float(initial_capital) - 1.0),
        "final_capital": float(capital),
        "win_rate": float((ret > 0).mean()) if len(ret) else float("nan"),
        "profit_factor": profit_factor(ret),
        "max_drawdown": max_drawdown(pd.Series(caps, dtype=float)),
        "avg_trade_return": float(ret.mean()) if len(ret) else float("nan"),
        "median_trade_return": float(ret.median()) if len(ret) else float("nan"),
        "worst_trade_return": float(ret.min()) if len(ret) else float("nan"),
        "best_trade_return": float(ret.max()) if len(ret) else float("nan"),
        "first_entry_time": combined["entry_time"].min(),
        "last_exit_time": combined["exit_time"].max(),
    }
    return combined, summary


def build_equity_curve(combined: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if combined.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "drawdown"])
    df = combined.sort_values(["exit_time", "strategy_leg", "entry_time"]).copy()
    if "portfolio_capital" not in df.columns:
        capital = float(initial_capital)
        caps = []
        for ret in pd.to_numeric(df["portfolio_return"], errors="coerce").fillna(0.0):
            capital *= 1.0 + float(ret)
            caps.append(capital)
        df["portfolio_capital"] = caps
    equity = pd.to_numeric(df["portfolio_capital"], errors="coerce")
    peak = equity.cummax()
    out = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["exit_time"], errors="coerce"),
            "equity": equity,
            "drawdown": equity / peak - 1.0,
        }
    )
    return out.reset_index(drop=True)


def standardize_trades(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if combined.empty:
        return pd.DataFrame(
            columns=[
                "trade_id",
                "entry_time",
                "exit_time",
                "side",
                "entry_price",
                "exit_price",
                "qty",
                "pnl",
                "fee",
                "return",
                "edge_id",
                "sleeve_id",
                "exit_reason",
            ]
        )
    for i, (_, row) in enumerate(combined.sort_values(["exit_time", "strategy_leg", "entry_time"]).iterrows(), start=1):
        leg = str(row.get("strategy_leg", ""))
        engine = str(row.get("engine", ""))
        edge_id = MF_EDGE_ID if leg == MF_TIME48_LEG else EDGE_ID_BY_ENGINE.get(engine, "UNKNOWN")
        sleeve_id = MF_SLEEVE_ID if leg == MF_TIME48_LEG else LF_SLEEVE_ID
        side_i = 1 if leg == MF_TIME48_LEG else side_from_trade_row(row)
        rows.append(
            {
                "trade_id": f"T{i:06d}",
                "entry_time": pd.Timestamp(row["entry_time"]),
                "exit_time": pd.Timestamp(row["exit_time"]),
                "side": "LONG" if side_i >= 0 else "SHORT",
                "entry_price": safe_float(row.get("entry_price", row.get("avg_entry", row.get("first_entry", np.nan)))),
                "exit_price": safe_float(row.get("exit_price", row.get("exit", np.nan))),
                "qty": safe_float(row.get("qty", row.get("filled_weight", np.nan))),
                "pnl": safe_float(row.get("portfolio_pnl", row.get("pnl", np.nan))),
                "fee": safe_float(row.get("fee", row.get("total_entry_fee", 0.0)), 0.0),
                "return": safe_float(row.get("portfolio_return", row.get("return_on_sleeve", np.nan))),
                "edge_id": edge_id,
                "sleeve_id": sleeve_id,
                "exit_reason": str(row.get("exit_reason", row.get("note", ""))),
                "strategy_leg": leg,
                "engine": engine,
                "scenario": row.get("scenario", ""),
            }
        )
    return pd.DataFrame(rows)


def edge_attribution(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["edge_id", "trades", "pnl", "return", "win_rate", "profit_factor"])
    rows: list[dict[str, object]] = []
    for edge_id, grp in trades.groupby("edge_id", dropna=False):
        ret = pd.to_numeric(grp["return"], errors="coerce").dropna()
        pnl = pd.to_numeric(grp["pnl"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "edge_id": edge_id,
                "trades": int(len(grp)),
                "pnl": float(pnl.sum()),
                "return": float(np.prod(1.0 + ret.to_numpy(dtype=float)) - 1.0) if len(ret) else 0.0,
                "win_rate": float((ret > 0).mean()) if len(ret) else float("nan"),
                "profit_factor": profit_factor(ret),
            }
        )
    return pd.DataFrame(rows).sort_values("edge_id").reset_index(drop=True)


def daily_returns(equity: pd.DataFrame, initial_capital: float) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(columns=["date", "equity", "daily_return"])
    out = equity.copy()
    out["date"] = pd.to_datetime(out["timestamp"], errors="coerce").dt.date
    daily = out.groupby("date", dropna=False)["equity"].last().reset_index()
    daily["daily_return"] = pd.to_numeric(daily["equity"], errors="coerce").pct_change()
    if len(daily):
        first_ret = float(daily.loc[0, "equity"]) / float(initial_capital) - 1.0
        daily.loc[0, "daily_return"] = first_ret
    return daily


def stress_report(lf: pd.DataFrame, mf: pd.DataFrame, args: Any) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if mf.empty:
        return pd.DataFrame(rows)
    leverage = float(getattr(args, "leverage", DEFAULT_LEVERAGE) or DEFAULT_LEVERAGE)
    exposures = parse_float_list(args.mf_exposures)
    mfv = mf.loc[mf["variant_name"].eq(MF_TIME48_LEG)].copy()
    if mfv.empty:
        return pd.DataFrame(rows)
    active_records: list[dict[str, object]] = []
    for _, row in mfv.sort_values("entry_time").iterrows():
        ts = pd.Timestamp(row["entry_time"])
        active = active_lf_trade_at(ts, lf)
        lf_side = 0
        lf_notional = 0.0
        lf_engine = "NONE"
        if active is not None:
            lf_side = side_from_trade_row(active)
            lf_notional = safe_float(active.get("position_notional_mult", 0.0), 0.0)
            lf_engine = str(active.get("engine", "UNKNOWN"))
        active_records.append({"lf_side": lf_side, "lf_notional": lf_notional, "lf_engine": lf_engine})
    active_df = pd.DataFrame(active_records)
    lf_notional_series = pd.to_numeric(active_df.get("lf_notional", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    lf_active = lf_notional_series > 0
    for exp in exposures:
        total_notional = lf_notional_series + float(exp)
        total_margin = total_notional / leverage
        rows.append(
            {
                "mf_variant_name": MF_TIME48_LEG,
                "mf_exposure": float(exp),
                "mf_margin_fraction_at_leverage": float(exp) / leverage,
                "mf_trades": int(len(mfv)),
                "lf_active_at_mf_entry": int(lf_active.sum()),
                "lf_long_at_mf_entry": int((active_df.get("lf_side", pd.Series(dtype=int)) > 0).sum()),
                "lf_short_at_mf_entry": int((active_df.get("lf_side", pd.Series(dtype=int)) < 0).sum()),
                "lf_active_ratio": float(lf_active.mean()) if len(lf_active) else float("nan"),
                "lf_notional_at_mf_entry_max": float(lf_notional_series.max()) if len(lf_notional_series) else float("nan"),
                "combined_notional_at_mf_entry_max": float(total_notional.max()) if len(total_notional) else float("nan"),
                "combined_margin_fraction_max": float(total_margin.max()) if len(total_margin) else float("nan"),
                "combined_margin_over_80pct_count": int((total_margin > 0.80).sum()),
                "combined_margin_over_90pct_count": int((total_margin > 0.90).sum()),
                "combined_margin_over_100pct_count": int((total_margin > 1.00).sum()),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "DEFAULT_LEVERAGE",
    "EDGE_ID_BY_ENGINE",
    "LF_LEG",
    "LF_SLEEVE_ID",
    "MF_EDGE_ID",
    "MF_SLEEVE_ID",
    "MF_TIME48_LEG",
    "PORTFOLIO_ID",
    "PortfolioScenario",
    "StrategyConfig",
    "apply_entry_slippage",
    "apply_exit_slippage",
    "attach_lf_position_metrics",
    "attach_mf_position_metrics",
    "build_equity_curve",
    "build_scenarios",
    "daily_returns",
    "edge_attribution",
    "simulate_portfolio_scenario",
    "standardize_trades",
]
