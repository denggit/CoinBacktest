#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Controlled scale-in path probe for low-sweep panic reversal.

This lab is downstream of ``low_sweep_panic_reversal_strategy_probe``. It does
not discover new signals. It reuses the no-leakage A/B/C/ABC candidate families
and asks one narrow question:

    Given the observed large MAE and larger MFE, does a *finite, risk-capped*
    scale-in entry improve the trade path versus full next-open entry?

This is deliberately not a traditional martingale:
- max filled weight is capped at 1.0 by default;
- there is no unlimited doubling down;
- every scale-in rule is predefined before the path is evaluated;
- signals and filters remain based only on closed bars / historical rolling
  thresholds from the upstream no-leakage probe.

No direct SQLite/CSV/ZIP reading is performed here. Historical data access stays
inside ``src.data_feed.OKXTradeBarLoader`` via the upstream loader.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.focused_low_sweep_reversal_event_lab import (  # noqa: E402
    _parse_number_list,
    build_canonical_events,
    build_low_sweep_events,
)
from research.low_sweep_panic_reversal_strategy_probe import (  # noqa: E402
    add_filter_bins,
    attach_extra_features_to_events,
    build_enriched_features,
    build_fixed_candidate_masks,
    load_trade_bars,
)
from src.research_common.progress import ProgressReporter  # noqa: E402

SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}


@dataclass(frozen=True)
class ScaleInScheme:
    name: str
    description: str
    mode: str
    initial_weight: float
    levels: tuple[tuple[float, float], ...] = ()
    confirm_level: str | None = None
    confirm_weight: float = 0.0


@dataclass
class Leg:
    weight: float
    entry_price: float
    entry_bar_offset: int
    leg_type: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Controlled scale-in probe for ETH low-sweep panic reversal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", choices=sorted(SUPPORTED_TIMEFRAMES), default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default="data/reports/research/low_sweep_scale_in_path_probe")

    # Same event definition as the no-leakage low-sweep probe.
    p.add_argument("--pivot-left", type=int, default=6)
    p.add_argument("--pivot-right", type=int, default=3)
    p.add_argument("--min-swing-age", type=int, default=3)
    p.add_argument("--max-swing-ages", default="12,24,48")
    p.add_argument("--min-swing-prominence-pcts", default="0.0015,0.0030")
    p.add_argument("--spike-pcts", default="0.0060,0.0080,0.0100,0.0120")
    p.add_argument("--breakout-pcts", default="0.0000,0.0005")
    p.add_argument("--variants", default="fade_close_through")
    p.add_argument("--wick-min-frac", type=float, default=0.45)
    p.add_argument("--close-through-buffer-pct", type=float, default=0.0)

    # Upstream feature windows / no-leakage rolling thresholds.
    p.add_argument("--volume-window", type=int, default=120)
    p.add_argument("--atr-window", type=int, default=42)
    p.add_argument("--cvd-window", type=int, default=60)
    p.add_argument("--cvd-windows", default="5,15,30,60,120")
    p.add_argument("--volume-spike-threshold", type=float, default=1.50)
    p.add_argument("--delta-capitulation-quantile", type=float, default=0.20, help="Deprecated research-only setting; not used by causal filters.")
    p.add_argument("--buy-ratio-thresholds", default="0.60,0.65,0.70")
    p.add_argument("--buy-pressure-thresholds", default="0.60,0.65,0.70")
    p.add_argument("--delta-pressure-thresholds", default="0.00,0.10,0.20")
    p.add_argument("--cvd-pressure-thresholds", default="0.00,0.05,0.10")
    p.add_argument("--rolling-quantile-days", default="30,90")
    p.add_argument("--rolling-quantiles", default="0.75,0.80")

    # Entry/exit probe.
    p.add_argument("--exit-horizons", default="48,96")
    p.add_argument("--emergency-stop-pcts", default="0.0120,0.0150")
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--max-position-weight", type=float, default=1.0)
    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.00020)
    p.add_argument("--exit-slippage-pct", type=float, default=0.00020)
    p.add_argument("--cost-multipliers", default="1.0,1.5,2.0")
    p.add_argument("--min-count", type=int, default=80)
    p.add_argument("--progress-every", type=int, default=10000)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--save-trade-sample", type=int, default=20000)
    return p.parse_args(argv)


def _entry_cost(args: argparse.Namespace, mult: float = 1.0) -> float:
    return float(args.entry_fee_rate + args.entry_slippage_pct) * float(mult)


def _exit_cost(args: argparse.Namespace, mult: float = 1.0) -> float:
    return float(args.exit_fee_rate + args.exit_slippage_pct) * float(mult)


def _profit_factor_raw(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    gross_profit = float(vals[vals > 0].sum())
    gross_loss = float(-vals[vals <= 0].sum())
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return gross_profit / gross_loss


def _max_drawdown_simple(returns: pd.Series) -> float:
    vals = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + vals)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.nanmin(dd))


def get_scale_in_schemes() -> list[ScaleInScheme]:
    return [
        ScaleInScheme(
            name="full_entry",
            description="100% next-open entry; benchmark, no scale-in",
            mode="full",
            initial_weight=1.0,
        ),
        ScaleInScheme(
            name="half_entry_only",
            description="50% next-open entry only; no add, lower exposure benchmark",
            mode="full",
            initial_weight=0.5,
        ),
        ScaleInScheme(
            name="scale_50_50_dd04",
            description="50% initial, add 50% on 0.4% adverse move; max weight 100%",
            mode="adverse_limit",
            initial_weight=0.5,
            levels=((0.004, 0.5),),
        ),
        ScaleInScheme(
            name="scale_50_50_dd08",
            description="50% initial, add 50% on 0.8% adverse move; max weight 100%",
            mode="adverse_limit",
            initial_weight=0.5,
            levels=((0.008, 0.5),),
        ),
        ScaleInScheme(
            name="scale_50_25_25_dd04_dd08",
            description="50% initial, add 25% at -0.4% and 25% at -0.8%; max weight 100%",
            mode="adverse_limit",
            initial_weight=0.5,
            levels=((0.004, 0.25), (0.008, 0.25)),
        ),
        ScaleInScheme(
            name="confirm_add_reclaim_entry",
            description="50% initial, add 50% next open after a closed bar reclaims initial entry price",
            mode="confirm_reclaim",
            initial_weight=0.5,
            confirm_level="entry_price",
            confirm_weight=0.5,
        ),
        ScaleInScheme(
            name="confirm_add_reclaim_signal_close",
            description="50% initial, add 50% next open after a closed bar reclaims signal close",
            mode="confirm_reclaim",
            initial_weight=0.5,
            confirm_level="signal_close",
            confirm_weight=0.5,
        ),
    ]


def prepare_studied_events(bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    features = build_enriched_features(bars, args)
    raw = build_low_sweep_events(features, args)
    raw = attach_extra_features_to_events(raw, features)
    raw = add_filter_bins(raw)
    canonical = build_canonical_events(raw)
    if canonical.empty:
        return canonical
    canonical["signal_time"] = pd.to_datetime(canonical["signal_time"])
    start_ts = pd.Timestamp(args.start_date)
    end_ts = pd.Timestamp(args.end_date) + pd.Timedelta(days=1)
    studied = canonical[(canonical["signal_time"] >= start_ts) & (canonical["signal_time"] < end_ts)].copy()
    studied["year"] = studied["signal_time"].dt.year
    return studied.reset_index(drop=True)


def _candidate_events(events: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    masks = build_fixed_candidate_masks(events)
    meta = masks.get(candidate_name)
    if meta is None:
        return events.iloc[0:0].copy()
    mask = meta["mask"].fillna(False).astype(bool)
    return events.loc[mask].copy().reset_index(drop=True)


def _event_positions(bars: pd.DataFrame, events: pd.DataFrame, entry_delay_bars: int, horizon: int) -> tuple[pd.DataFrame, np.ndarray]:
    if events.empty:
        return events.copy(), np.asarray([], dtype=int)
    frame = bars.sort_index()
    event_times = pd.DatetimeIndex(pd.to_datetime(events["signal_time"]))
    signal_pos = frame.index.get_indexer(event_times)
    valid = (signal_pos >= 0) & ((signal_pos + int(entry_delay_bars)) < len(frame)) & ((signal_pos + int(horizon)) < len(frame))
    return events.loc[valid].copy().reset_index(drop=True), signal_pos[valid]


def _add_leg_if_possible(
    legs: list[Leg],
    *,
    weight: float,
    price: float,
    bar_offset: int,
    leg_type: str,
    max_position_weight: float,
) -> bool:
    current = sum(leg.weight for leg in legs)
    room = max(0.0, float(max_position_weight) - current)
    final_weight = min(float(weight), room)
    if final_weight <= 1e-12:
        return False
    legs.append(Leg(weight=final_weight, entry_price=float(price), entry_bar_offset=int(bar_offset), leg_type=str(leg_type)))
    return True


def simulate_one_trade(
    bars: pd.DataFrame,
    event: pd.Series,
    signal_pos: int,
    *,
    scheme: ScaleInScheme,
    horizon: int,
    stop_pct: float,
    args: argparse.Namespace,
    cost_mult: float = 1.0,
) -> dict[str, object]:
    opens = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    idx = bars.index

    entry_delay = int(args.entry_delay_bars)
    entry_pos = int(signal_pos) + entry_delay
    exit_pos = int(signal_pos) + int(horizon)
    if entry_pos >= len(bars) or exit_pos >= len(bars):
        return {"valid": False, "invalid_reason": "insufficient_future_bars"}

    entry_price = float(opens[entry_pos])
    signal_close = float(closes[int(signal_pos)])
    stop_price = entry_price * (1.0 - float(stop_pct))
    legs: list[Leg] = []
    _add_leg_if_possible(
        legs,
        weight=float(scheme.initial_weight),
        price=entry_price,
        bar_offset=0,
        leg_type="initial",
        max_position_weight=float(args.max_position_weight),
    )

    pending_levels = list(scheme.levels)
    confirm_added = False
    stop_hit = False
    exit_reason = f"time_exit_h{int(horizon)}"
    exit_price = float(closes[exit_pos])
    exit_bar_offset = int(horizon)

    # Mark-to-market path relative to max planned notional. This is approximate
    # but useful for comparing whether scale-in reduces path pain.
    mtm_close: list[float] = []
    mtm_low: list[float] = []
    mtm_high: list[float] = []

    for pos in range(entry_pos, exit_pos + 1):
        offset = pos - int(signal_pos)
        # Conservative ordering for long entries: if a bar breaches the emergency
        # stop, exit before filling any adverse add in the same bar.
        if lows[pos] <= stop_price:
            stop_hit = True
            exit_reason = "emergency_stop"
            exit_price = float(stop_price)
            exit_bar_offset = int(offset)
            break

        if scheme.mode == "adverse_limit" and pending_levels:
            remaining_levels: list[tuple[float, float]] = []
            for dd, weight in pending_levels:
                add_price = entry_price * (1.0 - float(dd))
                if lows[pos] <= add_price:
                    _add_leg_if_possible(
                        legs,
                        weight=float(weight),
                        price=float(add_price),
                        bar_offset=int(offset),
                        leg_type=f"adverse_dd{int(round(float(dd) * 10000)):04d}",
                        max_position_weight=float(args.max_position_weight),
                    )
                else:
                    remaining_levels.append((float(dd), float(weight)))
            pending_levels = remaining_levels
        elif scheme.mode == "confirm_reclaim" and not confirm_added and pos > entry_pos:
            threshold = entry_price if scheme.confirm_level == "entry_price" else signal_close
            # Closed-bar confirmation: add at next open, never on the same close.
            if closes[pos] >= threshold and (pos + 1) <= exit_pos:
                _add_leg_if_possible(
                    legs,
                    weight=float(scheme.confirm_weight),
                    price=float(opens[pos + 1]),
                    bar_offset=int(pos + 1 - signal_pos),
                    leg_type=f"confirm_{scheme.confirm_level}",
                    max_position_weight=float(args.max_position_weight),
                )
                confirm_added = True

        if legs:
            filled = sum(leg.weight for leg in legs)
            close_pnl = sum(leg.weight * (float(closes[pos]) / leg.entry_price - 1.0) for leg in legs)
            low_pnl = sum(leg.weight * (float(lows[pos]) / leg.entry_price - 1.0) for leg in legs)
            high_pnl = sum(leg.weight * (float(highs[pos]) / leg.entry_price - 1.0) for leg in legs)
            # Keep before-cost path; costs are handled in realized return.
            if filled > 0:
                mtm_close.append(close_pnl)
                mtm_low.append(low_pnl)
                mtm_high.append(high_pnl)

    filled_weight = float(sum(leg.weight for leg in legs))
    if filled_weight <= 0:
        return {"valid": False, "invalid_reason": "no_filled_weight"}

    gross_on_max = float(sum(leg.weight * (exit_price / leg.entry_price - 1.0) for leg in legs))
    total_entry_cost = float(sum(leg.weight for leg in legs) * _entry_cost(args, cost_mult))
    total_exit_cost = float(filled_weight * _exit_cost(args, cost_mult))
    net_on_max = gross_on_max - total_entry_cost - total_exit_cost
    net_per_filled = net_on_max / filled_weight if filled_weight > 0 else np.nan
    gross_per_filled = gross_on_max / filled_weight if filled_weight > 0 else np.nan

    avg_entry_price = float(sum(leg.weight * leg.entry_price for leg in legs) / filled_weight)
    mae_on_max = float(np.nanmin(mtm_low)) if mtm_low else np.nan
    mfe_on_max = float(np.nanmax(mtm_high)) if mtm_high else np.nan
    mae_per_filled = mae_on_max / filled_weight if filled_weight > 0 and np.isfinite(mae_on_max) else np.nan
    mfe_per_filled = mfe_on_max / filled_weight if filled_weight > 0 and np.isfinite(mfe_on_max) else np.nan

    return {
        "valid": True,
        "signal_time": event.get("signal_time"),
        "entry_time": idx[entry_pos],
        "exit_time": idx[min(int(signal_pos) + int(exit_bar_offset), len(idx) - 1)],
        "year": int(pd.Timestamp(event.get("signal_time")).year),
        "scheme_name": scheme.name,
        "scheme_description": scheme.description,
        "exit_horizon": int(horizon),
        "emergency_stop_pct": float(stop_pct),
        "cost_mult": float(cost_mult),
        "entry_delay_bars": int(entry_delay),
        "entry_price": entry_price,
        "avg_entry_price": avg_entry_price,
        "exit_price": float(exit_price),
        "stop_price": float(stop_price),
        "exit_reason": exit_reason,
        "stop_hit": bool(stop_hit),
        "bars_held": int(exit_bar_offset - entry_delay),
        "filled_weight": filled_weight,
        "max_position_weight": float(args.max_position_weight),
        "add_count": max(0, len(legs) - 1),
        "add_filled": bool(len(legs) > 1),
        "first_add_bar_offset": int(legs[1].entry_bar_offset) if len(legs) > 1 else np.nan,
        "leg_count": int(len(legs)),
        "leg_weights": "|".join(f"{leg.weight:.4f}" for leg in legs),
        "leg_entry_prices": "|".join(f"{leg.entry_price:.4f}" for leg in legs),
        "leg_types": "|".join(leg.leg_type for leg in legs),
        "gross_return_on_max": gross_on_max,
        "net_return_on_max": net_on_max,
        "gross_return_per_filled": gross_per_filled,
        "net_return_per_filled": net_per_filled,
        "mae_on_max": mae_on_max,
        "mfe_on_max": mfe_on_max,
        "mae_per_filled": mae_per_filled,
        "mfe_per_filled": mfe_per_filled,
    }


def simulate_scale_in_group(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    candidate_name: str,
    scheme: ScaleInScheme,
    horizon: int,
    stop_pct: float,
    args: argparse.Namespace,
    cost_mult: float = 1.0,
) -> pd.DataFrame:
    part = _candidate_events(events, candidate_name)
    part, signal_pos = _event_positions(bars, part, int(args.entry_delay_bars), int(horizon))
    rows: list[dict[str, object]] = []
    progress = ProgressReporter(
        label=f"[scale-in] {candidate_name} {scheme.name} h{horizon} stop{stop_pct:.4f}",
        total=len(part),
        every=max(1, int(args.progress_every)),
        enabled=not bool(args.no_progress) and len(part) >= int(args.progress_every),
    )
    for i, (_, event) in enumerate(part.iterrows(), start=1):
        rec = simulate_one_trade(
            bars,
            event,
            int(signal_pos[i - 1]),
            scheme=scheme,
            horizon=int(horizon),
            stop_pct=float(stop_pct),
            args=args,
            cost_mult=float(cost_mult),
        )
        if rec.get("valid"):
            rec["candidate_name"] = candidate_name
            rows.append(rec)
        progress.update(i)
    progress.close()
    return pd.DataFrame(rows)


def summarize_trade_returns(df: pd.DataFrame, *, group_cols: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = dict(zip(group_cols, keys))
        x = pd.to_numeric(grp["net_return_on_max"], errors="coerce").dropna()
        xf = pd.to_numeric(grp["net_return_per_filled"], errors="coerce").dropna()
        rec.update(
            {
                "trades": int(len(grp)),
                "mean_net_on_max": float(x.mean()) if not x.empty else np.nan,
                "median_net_on_max": float(x.median()) if not x.empty else np.nan,
                "win_rate_on_max": float((x > 0).mean()) if not x.empty else np.nan,
                "profit_factor_on_max": _profit_factor_raw(x),
                "max_dd_simple_on_max": _max_drawdown_simple(x),
                "top5_winner_share_on_max": _top_winner_share(x),
                "mean_net_per_filled": float(xf.mean()) if not xf.empty else np.nan,
                "median_net_per_filled": float(xf.median()) if not xf.empty else np.nan,
                "win_rate_per_filled": float((xf > 0).mean()) if not xf.empty else np.nan,
                "profit_factor_per_filled": _profit_factor_raw(xf),
                "avg_filled_weight": float(pd.to_numeric(grp["filled_weight"], errors="coerce").mean()),
                "add_fill_rate": float(pd.to_numeric(grp["add_filled"], errors="coerce").mean()),
                "avg_add_count": float(pd.to_numeric(grp["add_count"], errors="coerce").mean()),
                "stop_hit_rate": float(pd.to_numeric(grp["stop_hit"], errors="coerce").mean()),
                "mae_on_max_median": float(pd.to_numeric(grp["mae_on_max"], errors="coerce").median()),
                "mfe_on_max_median": float(pd.to_numeric(grp["mfe_on_max"], errors="coerce").median()),
                "mae_per_filled_median": float(pd.to_numeric(grp["mae_per_filled"], errors="coerce").median()),
                "mfe_per_filled_median": float(pd.to_numeric(grp["mfe_per_filled"], errors="coerce").median()),
            }
        )
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [c for c in ["candidate_name", "exit_horizon", "emergency_stop_pct", "mean_net_on_max"] if c in out.columns]
    asc = [True] * max(0, len(sort_cols) - 1) + [False] if sort_cols else True
    return out.sort_values(sort_cols, ascending=asc).reset_index(drop=True)


def _top_winner_share(x: pd.Series, top_n: int = 5) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    pos = vals[vals > 0].sort_values(ascending=False)
    if pos.empty:
        return float("nan")
    denom = float(pos.sum())
    if denom <= 0:
        return float("nan")
    return float(pos.head(int(top_n)).sum() / denom)


def compare_against_full(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    keys = ["candidate_name", "exit_horizon", "emergency_stop_pct", "cost_mult"]
    base = summary[summary["scheme_name"] == "full_entry"][keys + [
        "mean_net_on_max",
        "median_net_on_max",
        "win_rate_on_max",
        "profit_factor_on_max",
        "stop_hit_rate",
        "mae_on_max_median",
    ]].copy()
    base = base.rename(
        columns={
            "mean_net_on_max": "base_mean_net_on_max",
            "median_net_on_max": "base_median_net_on_max",
            "win_rate_on_max": "base_win_rate_on_max",
            "profit_factor_on_max": "base_profit_factor_on_max",
            "stop_hit_rate": "base_stop_hit_rate",
            "mae_on_max_median": "base_mae_on_max_median",
        }
    )
    out = summary.merge(base, on=keys, how="left")
    out["delta_mean_vs_full"] = out["mean_net_on_max"] - out["base_mean_net_on_max"]
    out["delta_median_vs_full"] = out["median_net_on_max"] - out["base_median_net_on_max"]
    out["delta_win_rate_vs_full"] = out["win_rate_on_max"] - out["base_win_rate_on_max"]
    out["delta_pf_vs_full"] = out["profit_factor_on_max"] - out["base_profit_factor_on_max"]
    out["delta_stop_hit_vs_full"] = out["stop_hit_rate"] - out["base_stop_hit_rate"]
    out["delta_mae_median_vs_full"] = out["mae_on_max_median"] - out["base_mae_on_max_median"]
    return out


def build_scale_in_edge_registry(compared: pd.DataFrame) -> pd.DataFrame:
    if compared.empty:
        return pd.DataFrame()
    rows = []
    for _, row in compared.iterrows():
        if str(row.get("scheme_name")) == "full_entry":
            continue
        trades = int(row.get("trades", 0) or 0)
        mean_net = float(row.get("mean_net_on_max", np.nan))
        median_net = float(row.get("median_net_on_max", np.nan))
        pf = float(row.get("profit_factor_on_max", np.nan))
        delta_mean = float(row.get("delta_mean_vs_full", np.nan))
        delta_median = float(row.get("delta_median_vs_full", np.nan))
        status = "research_only_not_better_than_full"
        if trades >= 80 and mean_net > 0 and median_net > 0 and pf >= 1.2 and delta_mean > 0 and delta_median >= 0:
            status = "scale_in_candidate_beats_full"
        elif trades >= 80 and mean_net > 0 and median_net > 0 and pf >= 1.2:
            status = "scale_in_candidate_positive_but_not_better_than_full"
        rows.append(
            {
                "edge_name": f"LOW_SWEEP_SCALE_IN::{row.get('candidate_name')}::{row.get('scheme_name')}::h{int(row.get('exit_horizon'))}::stop{float(row.get('emergency_stop_pct')):.4f}",
                "candidate_name": row.get("candidate_name"),
                "scheme_name": row.get("scheme_name"),
                "exit_horizon": int(row.get("exit_horizon")),
                "emergency_stop_pct": float(row.get("emergency_stop_pct")),
                "status": status,
                "trades": trades,
                "mean_net_on_max": mean_net,
                "median_net_on_max": median_net,
                "win_rate_on_max": float(row.get("win_rate_on_max", np.nan)),
                "profit_factor_on_max": pf,
                "delta_mean_vs_full": delta_mean,
                "delta_median_vs_full": delta_median,
                "avg_filled_weight": float(row.get("avg_filled_weight", np.nan)),
                "add_fill_rate": float(row.get("add_fill_rate", np.nan)),
                "stop_hit_rate": float(row.get("stop_hit_rate", np.nan)),
                "leakage_status": "no_full_sample_quantile_filters; scale-in decisions use post-entry path only",
                "next_step": "Only promote if yearly/delay/cost stress stays better than full_entry.",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["status", "delta_mean_vs_full", "mean_net_on_max"], ascending=[True, False, False]).reset_index(drop=True)


def run_scale_in_probe(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = load_trade_bars(args)
    studied = prepare_studied_events(bars, args)
    if studied.empty:
        raise RuntimeError("No canonical low-sweep events generated for scale-in probe")

    candidates = ["A_spike_close_large_share", "B_session_spike_atr", "C_session_extreme_spike", "ABC_union"]
    schemes = get_scale_in_schemes()
    horizons = _parse_number_list(args.exit_horizons, cast=int, name="exit_horizons")
    stops = _parse_number_list(args.emergency_stop_pcts, cast=float, name="emergency_stop_pcts")
    cost_multipliers = _parse_number_list(args.cost_multipliers, cast=float, name="cost_multipliers")

    # Full trade sample uses base cost only. Cost stress is summary-only so files
    # stay small enough for routine research runs.
    trade_parts: list[pd.DataFrame] = []
    total_jobs = len(candidates) * len(schemes) * len(horizons) * len(stops)
    job_progress = ProgressReporter(
        label="[scale-in] jobs",
        total=total_jobs,
        every=1,
        enabled=not bool(args.no_progress),
    )
    done = 0
    for candidate_name in candidates:
        for scheme in schemes:
            for horizon in horizons:
                for stop in stops:
                    part = simulate_scale_in_group(
                        bars,
                        studied,
                        candidate_name=candidate_name,
                        scheme=scheme,
                        horizon=int(horizon),
                        stop_pct=float(stop),
                        args=args,
                        cost_mult=1.0,
                    )
                    if not part.empty:
                        trade_parts.append(part)
                    done += 1
                    job_progress.update(done)
    job_progress.close()

    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    summary = summarize_trade_returns(
        trades,
        group_cols=["candidate_name", "scheme_name", "exit_horizon", "emergency_stop_pct", "cost_mult"],
    )
    compared = compare_against_full(summary)
    yearly = summarize_trade_returns(
        trades,
        group_cols=["candidate_name", "scheme_name", "exit_horizon", "emergency_stop_pct", "cost_mult", "year"],
    )

    # Cost stress for all schemes, summarized only.
    stress_parts: list[pd.DataFrame] = []
    for cost_mult in cost_multipliers:
        if float(cost_mult) == 1.0:
            continue
        for candidate_name in candidates:
            for scheme in schemes:
                for horizon in horizons:
                    for stop in stops:
                        part = simulate_scale_in_group(
                            bars,
                            studied,
                            candidate_name=candidate_name,
                            scheme=scheme,
                            horizon=int(horizon),
                            stop_pct=float(stop),
                            args=args,
                            cost_mult=float(cost_mult),
                        )
                        if not part.empty:
                            stress_parts.append(part)
    stress_trades = pd.concat(stress_parts, ignore_index=True) if stress_parts else pd.DataFrame()
    cost_stress = summarize_trade_returns(
        pd.concat([trades, stress_trades], ignore_index=True) if not trades.empty or not stress_trades.empty else pd.DataFrame(),
        group_cols=["candidate_name", "scheme_name", "exit_horizon", "emergency_stop_pct", "cost_mult"],
    )
    cost_stress_compared = compare_against_full(cost_stress)

    registry = build_scale_in_edge_registry(compared)

    # Keep trade file bounded; summaries are complete.
    if int(args.save_trade_sample) > 0 and len(trades) > int(args.save_trade_sample):
        trade_out = trades.sort_values(["candidate_name", "scheme_name", "signal_time"]).head(int(args.save_trade_sample))
    else:
        trade_out = trades

    write_csv(studied, out_dir / "01_studied_events.csv")
    write_csv(trade_out, out_dir / "02_scale_in_trades_sample.csv")
    write_csv(summary, out_dir / "03_scale_in_summary.csv")
    write_csv(compared, out_dir / "04_scale_in_vs_full.csv")
    write_csv(yearly, out_dir / "05_scale_in_yearly.csv")
    write_csv(cost_stress_compared, out_dir / "06_scale_in_cost_stress.csv")
    write_csv(registry, out_dir / "07_scale_in_edge_registry.csv")

    meta = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "studied_events": int(len(studied)),
        "trade_rows": int(len(trades)),
        "candidates": candidates,
        "schemes": [scheme.name for scheme in schemes],
        "exit_horizons": horizons,
        "emergency_stop_pcts": stops,
        "cost_multipliers": cost_multipliers,
        "max_position_weight": float(args.max_position_weight),
        "entry_delay_bars": int(args.entry_delay_bars),
        "leakage_guard": {
            "data_loader": "OKXTradeBarLoader via upstream load_trade_bars",
            "filters": "A/B/C from no-leakage probe; no full-sample qcut filters",
            "rolling_thresholds": "feature.shift(1).rolling(...).quantile(...) from upstream build_enriched_features",
            "scale_in": "post-entry path decisions only; max filled weight capped; no unlimited martingale",
        },
    }
    (out_dir / "08_lab_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] wrote scale-in probe reports to {out_dir}", flush=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_scale_in_probe(args)


if __name__ == "__main__":
    main()
