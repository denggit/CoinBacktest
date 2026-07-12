#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Research backtest probe for ETH low-sweep panic reversal.

This is the next step after the event-study/path/scale-in probes. It turns the
research edge candidates into full, sequential strategy variants so we can judge
whether the observed event edge survives realistic trade lifecycle choices:

- candidate family: A / B / C / ABC_union from the no-leakage probe;
- entry structure: full next-open entry or finite, capped scale-in;
- exit: fixed time exit at 48/96 bars;
- risk control: no stop, wide fixed emergency stop, ATR stop, structural stop;
- one-position-at-a-time conflict resolver: signals while a trade is open are
  skipped, not double-counted.

This script is research-only. It does not modify live strategies and it does not
read SQLite/CSV/ZIP files directly. Market data remains behind the existing
``OKXTradeBarLoader`` path exposed by the upstream low-sweep research helpers.

Leakage guards:
- signal candidates are built from closed bars only;
- rolling thresholds come from the upstream no-leakage implementation using
  ``feature.shift(1).rolling(...).quantile(...)``;
- entries happen at a future open after ``entry_delay_bars``;
- structural stops only use levels recorded on the signal bar;
- scale-in decisions are post-entry path decisions with max filled weight capped.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
from research.low_sweep_scale_in_path_probe import ScaleInScheme, get_scale_in_schemes  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402

SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}


@dataclass(frozen=True)
class StopSpec:
    name: str
    mode: str
    value: float | None = None
    description: str = ""


@dataclass(frozen=True)
class StrategyVariant:
    variant_name: str
    candidate_name: str
    scheme: ScaleInScheme
    exit_horizon: int
    stop_spec: StopSpec


@dataclass
class Leg:
    weight: float
    entry_price: float
    entry_pos: int
    leg_type: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full research backtest probe for ETH low-sweep panic reversal.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", choices=sorted(SUPPORTED_TIMEFRAMES), default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default="data/reports/research/low_sweep_panic_reversal_strategy_backtest_probe")

    # Same low-sweep event definition as the no-leakage probe.
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

    # Upstream no-leakage feature settings.
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

    # Strategy lifecycle.
    p.add_argument("--candidate-names", default="A_spike_close_large_share,ABC_union", help="Comma-separated candidates from build_fixed_candidate_masks.")
    p.add_argument("--entry-schemes", default="full_entry,scale_50_25_25_dd04_dd08", help="Comma-separated scheme names from get_scale_in_schemes.")
    p.add_argument("--exit-horizons", default="48,96")
    p.add_argument(
        "--stop-specs",
        default="no_stop,fixed_0200,fixed_0250,atr_6x,structural_0010",
        help="Comma-separated: no_stop, fixed_0200, atr_6x, structural_0010. Values use pct*10000 tags.",
    )
    p.add_argument("--entry-delay-bars", type=int, default=1)
    p.add_argument("--max-position-weight", type=float, default=1.0)
    p.add_argument("--conflict-policy", choices=["skip_while_in_position"], default="skip_while_in_position")
    p.add_argument("--min-structural-stop-pct", type=float, default=0.0020, help="If structural stop is too close or above entry, force at least this distance below entry.")
    p.add_argument("--min-atr-stop-pct", type=float, default=0.0080)
    p.add_argument("--max-atr-stop-pct", type=float, default=0.0300)

    # Cost convention: default 0.11% fee + 0.04% slippage round trip.
    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.00020)
    p.add_argument("--exit-slippage-pct", type=float, default=0.00020)
    p.add_argument("--cost-multipliers", default="1.0,1.5,2.0")
    p.add_argument("--delay-bars-list", default="1,2,3")
    p.add_argument("--skip-cost-stress", action="store_true")
    p.add_argument("--skip-delay-stress", action="store_true")
    p.add_argument("--fast", action="store_true", help="Run base variants only: skip cost and delay stress.")

    p.add_argument("--starting-equity", type=float, default=1.0)
    p.add_argument("--min-trades-for-edge", type=int, default=80)
    p.add_argument("--min-profit-factor-for-edge", type=float, default=1.20)
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--save-trade-sample", type=int, default=50000)
    return p.parse_args(argv)


def _entry_cost(args: argparse.Namespace, cost_mult: float = 1.0) -> float:
    return float(args.entry_fee_rate + args.entry_slippage_pct) * float(cost_mult)


def _exit_cost(args: argparse.Namespace, cost_mult: float = 1.0) -> float:
    return float(args.exit_fee_rate + args.exit_slippage_pct) * float(cost_mult)


def _split_csv_names(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _timeframe_to_minutes(timeframe: str) -> int:
    tf = str(timeframe).strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("H"):
        return int(tf[:-1]) * 60
    if tf.endswith("D"):
        return int(tf[:-1]) * 1440
    raise ValueError(f"Unsupported timeframe: {timeframe}")


def _profit_factor(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    gp = float(vals[vals > 0].sum())
    gl = float(-vals[vals < 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / gl


def _payoff_ratio(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    wins = vals[vals > 0]
    losses = vals[vals < 0]
    if wins.empty or losses.empty:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def _top_winner_share(x: pd.Series, top_n: int = 5) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    pos = vals[vals > 0].sort_values(ascending=False)
    if pos.empty:
        return float("nan")
    denom = float(pos.sum())
    if denom <= 0:
        return float("nan")
    return float(pos.head(int(top_n)).sum() / denom)


def _max_consecutive_losses(x: pd.Series) -> int:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    best = 0
    cur = 0
    for v in vals:
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _equity_and_dd(returns: pd.Series, starting_equity: float = 1.0) -> tuple[pd.Series, pd.Series]:
    vals = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = float(starting_equity) * (1.0 + vals).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return equity, dd


def parse_stop_spec(text: str) -> StopSpec:
    token = str(text).strip().lower()
    if token in {"", "none", "no", "no_stop", "nostop"}:
        return StopSpec(name="no_stop", mode="none", description="No intratrade stop; exit only by time horizon")
    m = re.fullmatch(r"fixed_(\d{4})", token)
    if m:
        pct = int(m.group(1)) / 10000.0
        return StopSpec(name=f"fixed_{m.group(1)}", mode="fixed_pct", value=float(pct), description=f"Fixed emergency stop {pct:.2%} below initial entry")
    m = re.fullmatch(r"atr_([0-9]+(?:\.[0-9]+)?)x", token)
    if m:
        mult = float(m.group(1))
        tag = (f"{mult:g}").replace(".", "p")
        return StopSpec(name=f"atr_{tag}x", mode="atr_mult", value=mult, description=f"ATR pct stop: atr_pct * {mult:g}, clipped by min/max")
    m = re.fullmatch(r"structural_(\d{4})", token)
    if m:
        buffer_pct = int(m.group(1)) / 10000.0
        return StopSpec(name=f"structural_{m.group(1)}", mode="structural", value=float(buffer_pct), description=f"Signal structural stop level minus {buffer_pct:.2%} buffer")
    raise ValueError(f"Unsupported stop spec: {text!r}")


def parse_stop_specs(raw: str) -> list[StopSpec]:
    return [parse_stop_spec(x) for x in _split_csv_names(raw)]


def prepare_studied_events(bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    print("[events] building enriched no-leakage low-sweep features", flush=True)
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
    studied = studied.sort_values("signal_time").reset_index(drop=True)
    print(f"[events] studied_events={len(studied):,}", flush=True)
    return studied


def select_candidate_events(events: pd.DataFrame, candidate_name: str) -> pd.DataFrame:
    masks = build_fixed_candidate_masks(events)
    meta = masks.get(str(candidate_name))
    if meta is None:
        known = ", ".join(sorted(masks.keys()))
        raise ValueError(f"Unknown candidate {candidate_name!r}. Known: {known}")
    mask = meta["mask"].fillna(False).astype(bool)
    return events.loc[mask].copy().sort_values("signal_time").reset_index(drop=True)


def _event_positions(bars: pd.DataFrame, events: pd.DataFrame, entry_delay_bars: int, max_horizon: int) -> tuple[pd.DataFrame, np.ndarray]:
    if events.empty:
        return events.copy(), np.asarray([], dtype=int)
    frame = bars.sort_index()
    event_times = pd.DatetimeIndex(pd.to_datetime(events["signal_time"]))
    signal_pos = frame.index.get_indexer(event_times)
    valid = (signal_pos >= 0) & ((signal_pos + int(entry_delay_bars)) < len(frame)) & ((signal_pos + int(max_horizon)) < len(frame))
    return events.loc[valid].copy().reset_index(drop=True), signal_pos[valid]


def _add_leg_if_possible(legs: list[Leg], *, weight: float, entry_price: float, entry_pos: int, leg_type: str, max_position_weight: float) -> bool:
    current = sum(float(leg.weight) for leg in legs)
    room = max(0.0, float(max_position_weight) - current)
    final_weight = min(float(weight), room)
    if final_weight <= 1e-12:
        return False
    legs.append(Leg(weight=final_weight, entry_price=float(entry_price), entry_pos=int(entry_pos), leg_type=str(leg_type)))
    return True


def _compute_stop_price(event: pd.Series, entry_price: float, stop_spec: StopSpec, args: argparse.Namespace) -> tuple[float | None, float | None, str]:
    if stop_spec.mode == "none":
        return None, None, "no_stop"
    if stop_spec.mode == "fixed_pct":
        pct = float(stop_spec.value or 0.0)
        return float(entry_price * (1.0 - pct)), pct, stop_spec.name
    if stop_spec.mode == "atr_mult":
        atr_pct = float(pd.to_numeric(pd.Series([event.get("atr_pct", np.nan)]), errors="coerce").iloc[0])
        if not math.isfinite(atr_pct) or atr_pct <= 0:
            atr_pct = float(args.min_atr_stop_pct)
        raw_pct = atr_pct * float(stop_spec.value or 1.0)
        pct = min(float(args.max_atr_stop_pct), max(float(args.min_atr_stop_pct), float(raw_pct)))
        return float(entry_price * (1.0 - pct)), pct, stop_spec.name
    if stop_spec.mode == "structural":
        raw_level = float(pd.to_numeric(pd.Series([event.get("structural_stop_level", np.nan)]), errors="coerce").iloc[0])
        buffer_pct = float(stop_spec.value or 0.0)
        min_pct = float(args.min_structural_stop_pct)
        fallback = float(entry_price * (1.0 - min_pct))
        if not math.isfinite(raw_level) or raw_level <= 0:
            return fallback, min_pct, f"{stop_spec.name}_fallback_min"
        stop_price = float(raw_level * (1.0 - buffer_pct))
        if stop_price >= entry_price:
            stop_price = fallback
        pct = max(0.0, float(1.0 - stop_price / entry_price))
        if pct < min_pct:
            stop_price = fallback
            pct = min_pct
        return float(stop_price), float(pct), stop_spec.name
    raise ValueError(f"Unsupported stop mode {stop_spec.mode!r}")


def simulate_trade_path(
    bars: pd.DataFrame,
    event: pd.Series,
    signal_pos: int,
    *,
    scheme: ScaleInScheme,
    exit_horizon: int,
    stop_spec: StopSpec,
    args: argparse.Namespace,
    cost_mult: float = 1.0,
    entry_delay_bars: int | None = None,
) -> dict[str, object]:
    opens = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    idx = bars.index

    delay = int(args.entry_delay_bars if entry_delay_bars is None else entry_delay_bars)
    entry_pos = int(signal_pos) + delay
    planned_exit_pos = int(signal_pos) + int(exit_horizon)
    if entry_pos >= len(bars) or planned_exit_pos >= len(bars) or planned_exit_pos <= entry_pos:
        return {"valid": False, "invalid_reason": "insufficient_future_bars"}

    initial_entry_price = float(opens[entry_pos])
    stop_price, stop_pct, stop_mode_used = _compute_stop_price(event, initial_entry_price, stop_spec, args)
    legs: list[Leg] = []
    _add_leg_if_possible(
        legs,
        weight=float(scheme.initial_weight),
        entry_price=initial_entry_price,
        entry_pos=entry_pos,
        leg_type="initial",
        max_position_weight=float(args.max_position_weight),
    )

    pending_levels = list(scheme.levels)
    stop_hit = False
    exit_reason = f"time_exit_h{int(exit_horizon)}"
    exit_price = float(closes[planned_exit_pos])
    exit_pos = int(planned_exit_pos)

    mtm_low: list[float] = []
    mtm_high: list[float] = []
    mtm_close: list[float] = []

    for pos in range(entry_pos, planned_exit_pos + 1):
        # Conservative for long: if the stop is touched, exit before filling an
        # adverse add on the same bar.
        if stop_price is not None and lows[pos] <= float(stop_price):
            stop_hit = True
            exit_reason = stop_mode_used
            exit_price = float(stop_price)
            exit_pos = int(pos)
            break

        if scheme.mode == "adverse_limit" and pending_levels:
            rest: list[tuple[float, float]] = []
            for dd, weight in pending_levels:
                add_price = initial_entry_price * (1.0 - float(dd))
                if lows[pos] <= add_price:
                    _add_leg_if_possible(
                        legs,
                        weight=float(weight),
                        entry_price=float(add_price),
                        entry_pos=int(pos),
                        leg_type=f"adverse_dd{int(round(float(dd) * 10000)):04d}",
                        max_position_weight=float(args.max_position_weight),
                    )
                else:
                    rest.append((float(dd), float(weight)))
            pending_levels = rest

        if legs:
            close_pnl = sum(leg.weight * (float(closes[pos]) / leg.entry_price - 1.0) for leg in legs)
            low_pnl = sum(leg.weight * (float(lows[pos]) / leg.entry_price - 1.0) for leg in legs)
            high_pnl = sum(leg.weight * (float(highs[pos]) / leg.entry_price - 1.0) for leg in legs)
            mtm_close.append(float(close_pnl))
            mtm_low.append(float(low_pnl))
            mtm_high.append(float(high_pnl))

    filled_weight = float(sum(leg.weight for leg in legs))
    if filled_weight <= 0:
        return {"valid": False, "invalid_reason": "no_filled_weight"}

    gross_on_equity = float(sum(leg.weight * (exit_price / leg.entry_price - 1.0) for leg in legs))
    entry_cost = float(sum(leg.weight for leg in legs) * _entry_cost(args, cost_mult))
    exit_cost = float(filled_weight * _exit_cost(args, cost_mult))
    net_on_equity = gross_on_equity - entry_cost - exit_cost
    avg_entry_price = float(sum(leg.weight * leg.entry_price for leg in legs) / filled_weight)
    mae_on_equity = float(np.nanmin(mtm_low)) if mtm_low else np.nan
    mfe_on_equity = float(np.nanmax(mtm_high)) if mtm_high else np.nan
    close_mtm_min = float(np.nanmin(mtm_close)) if mtm_close else np.nan

    return {
        "valid": True,
        "signal_time": event.get("signal_time"),
        "entry_time": idx[entry_pos],
        "exit_time": idx[exit_pos],
        "year": int(pd.Timestamp(event.get("signal_time")).year),
        "month": pd.Timestamp(idx[exit_pos]).strftime("%Y-%m"),
        "entry_pos": int(entry_pos),
        "exit_pos": int(exit_pos),
        "signal_pos": int(signal_pos),
        "entry_delay_bars": int(delay),
        "bars_held": int(exit_pos - entry_pos),
        "exit_horizon": int(exit_horizon),
        "candidate_name": None,
        "scheme_name": scheme.name,
        "scheme_description": scheme.description,
        "stop_name": stop_spec.name,
        "stop_mode": stop_spec.mode,
        "stop_pct": float(stop_pct) if stop_pct is not None else np.nan,
        "stop_price": float(stop_price) if stop_price is not None else np.nan,
        "stop_hit": bool(stop_hit),
        "exit_reason": exit_reason,
        "cost_mult": float(cost_mult),
        "initial_entry_price": initial_entry_price,
        "avg_entry_price": avg_entry_price,
        "exit_price": float(exit_price),
        "filled_weight": filled_weight,
        "add_count": max(0, len(legs) - 1),
        "add_filled": bool(len(legs) > 1),
        "leg_count": int(len(legs)),
        "leg_weights": "|".join(f"{leg.weight:.4f}" for leg in legs),
        "leg_entry_prices": "|".join(f"{leg.entry_price:.4f}" for leg in legs),
        "leg_types": "|".join(leg.leg_type for leg in legs),
        "gross_return_on_equity": gross_on_equity,
        "net_return_on_equity": net_on_equity,
        "net_return_per_filled": net_on_equity / filled_weight if filled_weight > 0 else np.nan,
        "mae_on_equity": mae_on_equity,
        "mfe_on_equity": mfe_on_equity,
        "mae_per_filled": mae_on_equity / filled_weight if filled_weight > 0 and math.isfinite(mae_on_equity) else np.nan,
        "mfe_per_filled": mfe_on_equity / filled_weight if filled_weight > 0 and math.isfinite(mfe_on_equity) else np.nan,
        "min_close_mtm_on_equity": close_mtm_min,
        "signal_close": event.get("close", np.nan),
        "signal_low": event.get("low", np.nan),
        "structural_stop_level": event.get("structural_stop_level", np.nan),
        "atr_pct": event.get("atr_pct", np.nan),
        "down_spike_pct": event.get("down_spike_pct", np.nan),
        "large_trade_share": event.get("large_trade_share", np.nan),
        "session_bucket": event.get("session_bucket", ""),
    }


def build_variants(args: argparse.Namespace) -> list[StrategyVariant]:
    scheme_map = {scheme.name: scheme for scheme in get_scale_in_schemes()}
    scheme_names = _split_csv_names(args.entry_schemes)
    missing = [name for name in scheme_names if name not in scheme_map]
    if missing:
        raise ValueError(f"Unknown entry scheme(s): {missing}. Known: {sorted(scheme_map)}")
    candidates = _split_csv_names(args.candidate_names)
    horizons = _parse_number_list(args.exit_horizons, cast=int, name="exit_horizons")
    stops = parse_stop_specs(args.stop_specs)
    variants: list[StrategyVariant] = []
    for candidate_name in candidates:
        for scheme_name in scheme_names:
            for horizon in horizons:
                for stop in stops:
                    scheme = scheme_map[scheme_name]
                    variant_name = f"{candidate_name}__{scheme.name}__h{int(horizon)}__{stop.name}"
                    variants.append(
                        StrategyVariant(
                            variant_name=variant_name,
                            candidate_name=candidate_name,
                            scheme=scheme,
                            exit_horizon=int(horizon),
                            stop_spec=stop,
                        )
                    )
    return variants


def simulate_variant(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    variant: StrategyVariant,
    args: argparse.Namespace,
    *,
    cost_mult: float = 1.0,
    entry_delay_bars: int | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    delay = int(args.entry_delay_bars if entry_delay_bars is None else entry_delay_bars)
    part = select_candidate_events(events, variant.candidate_name)
    part, signal_pos = _event_positions(bars, part, delay, int(variant.exit_horizon))
    rows: list[dict[str, object]] = []
    skipped_overlap = 0
    skipped_invalid = 0
    last_exit_pos = -1
    progress = ProgressReporter(
        label=f"[backtest] {variant.variant_name} cost{cost_mult:g} delay{delay}",
        total=len(part),
        every=max(1, int(args.progress_every)),
        enabled=not bool(args.no_progress) and len(part) >= int(args.progress_every),
    )
    for i, (_, event) in enumerate(part.iterrows(), start=1):
        sig_pos = int(signal_pos[i - 1])
        entry_pos = sig_pos + delay
        if entry_pos <= last_exit_pos:
            skipped_overlap += 1
            progress.update(i)
            continue
        rec = simulate_trade_path(
            bars,
            event,
            sig_pos,
            scheme=variant.scheme,
            exit_horizon=int(variant.exit_horizon),
            stop_spec=variant.stop_spec,
            args=args,
            cost_mult=float(cost_mult),
            entry_delay_bars=delay,
        )
        if not rec.get("valid"):
            skipped_invalid += 1
            progress.update(i)
            continue
        rec["variant_name"] = variant.variant_name
        rec["candidate_name"] = variant.candidate_name
        rows.append(rec)
        last_exit_pos = int(rec["exit_pos"])
        progress.update(i)
    progress.close()
    trades = pd.DataFrame(rows)
    if not trades.empty:
        trades = trades.sort_values(["entry_time", "exit_time"]).reset_index(drop=True)
    return trades, {
        "raw_candidate_signals": int(len(part)),
        "executed_trades": int(len(trades)),
        "skipped_overlap": int(skipped_overlap),
        "skipped_invalid": int(skipped_invalid),
    }


def summarize_trades(trades: pd.DataFrame, args: argparse.Namespace, *, extra: dict[str, int] | None = None) -> dict[str, object]:
    extra = extra or {}
    if trades.empty:
        rec: dict[str, object] = dict(extra)
        rec.update({"trades": 0})
        return rec
    x = pd.to_numeric(trades["net_return_on_equity"], errors="coerce").fillna(0.0)
    equity, dd = _equity_and_dd(x, float(args.starting_equity))
    trades = trades.copy()
    trades["equity_after_trade"] = equity.to_numpy(dtype=float)
    first_entry = pd.Timestamp(trades["entry_time"].iloc[0])
    last_exit = pd.Timestamp(trades["exit_time"].iloc[-1])
    days = max(1e-9, (last_exit - first_entry).total_seconds() / 86400.0)
    total_ret = float(equity.iloc[-1] / float(args.starting_equity) - 1.0)
    ann_ret = float((1.0 + total_ret) ** (365.0 / days) - 1.0) if total_ret > -1.0 else -1.0
    tf_minutes = _timeframe_to_minutes(str(args.timeframe))
    exposure_days = float(pd.to_numeric(trades["bars_held"], errors="coerce").sum() * tf_minutes / 1440.0)
    test_days = max(1e-9, (pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timestamp(args.start_date)).total_seconds() / 86400.0)
    wins = x[x > 0]
    losses = x[x < 0]
    rec = {
        "trades": int(len(trades)),
        "return_total": total_ret,
        "return_annualized": ann_ret,
        "mean_return": float(x.mean()),
        "median_return": float(x.median()),
        "win_rate": float((x > 0).mean()),
        "avg_win": float(wins.mean()) if not wins.empty else np.nan,
        "avg_loss": float(losses.mean()) if not losses.empty else np.nan,
        "payoff_ratio": _payoff_ratio(x),
        "profit_factor": _profit_factor(x),
        "max_drawdown": float(dd.min()),
        "max_consecutive_losses": _max_consecutive_losses(x),
        "top5_winner_share": _top_winner_share(x),
        "stop_hit_rate": float(pd.to_numeric(trades["stop_hit"], errors="coerce").mean()),
        "avg_bars_held": float(pd.to_numeric(trades["bars_held"], errors="coerce").mean()),
        "median_bars_held": float(pd.to_numeric(trades["bars_held"], errors="coerce").median()),
        "avg_filled_weight": float(pd.to_numeric(trades["filled_weight"], errors="coerce").mean()),
        "add_fill_rate": float(pd.to_numeric(trades["add_filled"], errors="coerce").mean()),
        "mae_median": float(pd.to_numeric(trades["mae_on_equity"], errors="coerce").median()),
        "mfe_median": float(pd.to_numeric(trades["mfe_on_equity"], errors="coerce").median()),
        "worst_trade": float(x.min()),
        "best_trade": float(x.max()),
        "exposure_days": exposure_days,
        "exposure_ratio": exposure_days / test_days,
        "first_entry_time": first_entry,
        "last_exit_time": last_exit,
    }
    rec.update(extra)
    if "raw_candidate_signals" in rec and rec.get("raw_candidate_signals"):
        rec["execution_rate"] = float(rec.get("executed_trades", len(trades))) / float(rec.get("raw_candidate_signals"))
    return rec


def build_equity_curve(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    parts = []
    for variant_name, grp in trades.groupby("variant_name", dropna=False):
        g = grp.sort_values(["exit_time", "entry_time"]).copy().reset_index(drop=True)
        x = pd.to_numeric(g["net_return_on_equity"], errors="coerce").fillna(0.0)
        equity, dd = _equity_and_dd(x, float(args.starting_equity))
        out = pd.DataFrame(
            {
                "variant_name": variant_name,
                "trade_no": np.arange(1, len(g) + 1),
                "exit_time": pd.to_datetime(g["exit_time"]),
                "trade_return": x,
                "equity": equity,
                "drawdown": dd,
            }
        )
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def summarize_by_period(trades: pd.DataFrame, args: argparse.Namespace, period_col: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in trades.groupby(["variant_name", period_col], dropna=False):
        variant_name, period = keys
        x = pd.to_numeric(grp.sort_values("exit_time")["net_return_on_equity"], errors="coerce").fillna(0.0)
        equity, dd = _equity_and_dd(x, 1.0)
        rows.append(
            {
                "variant_name": variant_name,
                period_col: period,
                "trades": int(len(grp)),
                "return_total": float(equity.iloc[-1] - 1.0) if not equity.empty else np.nan,
                "mean_return": float(x.mean()) if not x.empty else np.nan,
                "median_return": float(x.median()) if not x.empty else np.nan,
                "win_rate": float((x > 0).mean()) if not x.empty else np.nan,
                "profit_factor": _profit_factor(x),
                "max_drawdown": float(dd.min()) if not dd.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["variant_name", period_col]).reset_index(drop=True)


def run_variant_jobs(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    variants: list[StrategyVariant],
    args: argparse.Namespace,
    *,
    cost_mult: float = 1.0,
    entry_delay_bars: int | None = None,
    label: str = "[backtest] variants",
    keep_trades: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    trade_parts: list[pd.DataFrame] = []
    progress = ProgressReporter(label=label, total=len(variants), every=1, enabled=not bool(args.no_progress))
    for i, variant in enumerate(variants, start=1):
        print(f"[job] {label} {i}/{len(variants)} {variant.variant_name} cost={cost_mult:g} delay={entry_delay_bars if entry_delay_bars is not None else args.entry_delay_bars}", flush=True)
        trades, counters = simulate_variant(
            bars,
            events,
            variant,
            args,
            cost_mult=float(cost_mult),
            entry_delay_bars=entry_delay_bars,
        )
        if not trades.empty:
            trades["variant_name"] = variant.variant_name
            trades["candidate_name"] = variant.candidate_name
            trades["scheme_name"] = variant.scheme.name
            trades["exit_horizon"] = int(variant.exit_horizon)
            trades["stop_name"] = variant.stop_spec.name
            trades["cost_mult"] = float(cost_mult)
            trades["entry_delay_bars"] = int(args.entry_delay_bars if entry_delay_bars is None else entry_delay_bars)
            if keep_trades:
                trade_parts.append(trades)
        rec = summarize_trades(trades, args, extra=counters)
        rec.update(
            {
                "variant_name": variant.variant_name,
                "candidate_name": variant.candidate_name,
                "scheme_name": variant.scheme.name,
                "scheme_description": variant.scheme.description,
                "exit_horizon": int(variant.exit_horizon),
                "stop_name": variant.stop_spec.name,
                "stop_mode": variant.stop_spec.mode,
                "stop_description": variant.stop_spec.description,
                "cost_mult": float(cost_mult),
                "entry_delay_bars": int(args.entry_delay_bars if entry_delay_bars is None else entry_delay_bars),
            }
        )
        summary_rows.append(rec)
        progress.update(i)
    progress.close()
    trades_all = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["profit_factor", "return_total", "trades"], ascending=[False, False, False]).reset_index(drop=True)
    return trades_all, summary


def build_edge_registry(summary: pd.DataFrame, yearly: pd.DataFrame, monthly: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    year_stats = yearly.groupby("variant_name").agg(
        tested_years=("year", "count"),
        positive_years=("return_total", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        min_year_return=("return_total", "min"),
    ).reset_index() if not yearly.empty else pd.DataFrame(columns=["variant_name", "tested_years", "positive_years", "min_year_return"])
    month_stats = monthly.groupby("variant_name").agg(
        tested_months=("month", "count"),
        positive_months=("return_total", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
    ).reset_index() if not monthly.empty else pd.DataFrame(columns=["variant_name", "tested_months", "positive_months"])
    out = summary.merge(year_stats, on="variant_name", how="left").merge(month_stats, on="variant_name", how="left")
    rows = []
    for _, row in out.iterrows():
        trades = int(row.get("trades", 0) or 0)
        ret = float(row.get("return_total", np.nan))
        pf = float(row.get("profit_factor", np.nan))
        dd = float(row.get("max_drawdown", np.nan))
        median = float(row.get("median_return", np.nan))
        positive_years = int(row.get("positive_years", 0) or 0)
        tested_years = int(row.get("tested_years", 0) or 0)
        status = "research_only_not_promoted"
        if trades >= int(args.min_trades_for_edge) and ret > 0 and median > 0 and pf >= float(args.min_profit_factor_for_edge) and positive_years >= max(3, min(4, tested_years)):
            status = "strategy_backtest_edge_candidate"
        elif trades >= int(args.min_trades_for_edge) and ret > 0 and pf >= 1.05:
            status = "positive_but_needs_filter_or_exit_work"
        rows.append(
            {
                "edge_name": f"LOW_SWEEP_STRATEGY::{row.get('variant_name')}",
                "variant_name": row.get("variant_name"),
                "candidate_name": row.get("candidate_name"),
                "scheme_name": row.get("scheme_name"),
                "exit_horizon": int(row.get("exit_horizon", 0) or 0),
                "stop_name": row.get("stop_name"),
                "status": status,
                "trades": trades,
                "return_total": ret,
                "return_annualized": float(row.get("return_annualized", np.nan)),
                "profit_factor": pf,
                "win_rate": float(row.get("win_rate", np.nan)),
                "payoff_ratio": float(row.get("payoff_ratio", np.nan)),
                "max_drawdown": dd,
                "median_return": median,
                "stop_hit_rate": float(row.get("stop_hit_rate", np.nan)),
                "top5_winner_share": float(row.get("top5_winner_share", np.nan)),
                "positive_years": positive_years,
                "tested_years": tested_years,
                "positive_months": int(row.get("positive_months", 0) or 0),
                "tested_months": int(row.get("tested_months", 0) or 0),
                "leakage_status": "no_full_sample_quantile_filters; next-open entries; sequential one-position-at-a-time backtest",
                "next_step": "Promote only after cost/delay stress, parameter stability, and out-of-sample/live-sim validation.",
            }
        )
    registry = pd.DataFrame(rows)
    if registry.empty:
        return registry
    return registry.sort_values(["status", "return_total", "profit_factor"], ascending=[True, False, False]).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    print(f"[write] {path.name} rows={len(df):,}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_backtest_probe(args: argparse.Namespace) -> None:
    if bool(args.fast):
        args.skip_cost_stress = True
        args.skip_delay_stress = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] loading trade bars", flush=True)
    bars = load_trade_bars(args)
    events = prepare_studied_events(bars, args)
    if events.empty:
        raise RuntimeError("No studied low-sweep events generated")

    variants = build_variants(args)
    print(f"[setup] variants={len(variants):,} candidates={args.candidate_names} schemes={args.entry_schemes} stops={args.stop_specs}", flush=True)

    trades, summary = run_variant_jobs(bars, events, variants, args, cost_mult=1.0, label="[backtest] base variants", keep_trades=True)
    equity_curve = build_equity_curve(trades, args)
    yearly = summarize_by_period(trades, args, "year")
    monthly = summarize_by_period(trades, args, "month")

    cost_stress = pd.DataFrame()
    if not bool(args.skip_cost_stress):
        cost_multipliers = _parse_number_list(args.cost_multipliers, cast=float, name="cost_multipliers")
        stress_parts = [summary]
        stress_mults = [m for m in cost_multipliers if abs(float(m) - 1.0) > 1e-12]
        print(f"[stress] cost multipliers={stress_mults}", flush=True)
        progress = ProgressReporter(label="[stress] cost multipliers", total=len(stress_mults), every=1, enabled=not bool(args.no_progress))
        for i, mult in enumerate(stress_mults, start=1):
            _, stress_summary = run_variant_jobs(
                bars,
                events,
                variants,
                args,
                cost_mult=float(mult),
                label=f"[stress] cost {float(mult):g}x variants",
                keep_trades=False,
            )
            stress_parts.append(stress_summary)
            progress.update(i)
        progress.close()
        cost_stress = pd.concat(stress_parts, ignore_index=True) if stress_parts else pd.DataFrame()

    delay_stress = pd.DataFrame()
    if not bool(args.skip_delay_stress):
        delay_bars = _parse_number_list(args.delay_bars_list, cast=int, name="delay_bars_list")
        stress_parts = [summary]
        delay_values = [d for d in delay_bars if int(d) != int(args.entry_delay_bars)]
        print(f"[stress] delay bars={delay_values}", flush=True)
        progress = ProgressReporter(label="[stress] delay groups", total=len(delay_values), every=1, enabled=not bool(args.no_progress))
        for i, delay in enumerate(delay_values, start=1):
            _, delay_summary = run_variant_jobs(
                bars,
                events,
                variants,
                args,
                cost_mult=1.0,
                entry_delay_bars=int(delay),
                label=f"[stress] delay {int(delay)} variants",
                keep_trades=False,
            )
            stress_parts.append(delay_summary)
            progress.update(i)
        progress.close()
        delay_stress = pd.concat(stress_parts, ignore_index=True) if stress_parts else pd.DataFrame()

    registry = build_edge_registry(summary, yearly, monthly, args)

    trade_out = trades
    if int(args.save_trade_sample) > 0 and len(trade_out) > int(args.save_trade_sample):
        trade_out = trade_out.sort_values(["variant_name", "entry_time"]).head(int(args.save_trade_sample)).copy()

    # Signal flags file is useful for debugging candidate overlap without writing
    # huge full feature frames.
    signal_flags = events[[c for c in ["signal_time", "event_name", "close", "low", "structural_stop_level", "atr_pct", "down_spike_pct", "large_trade_share", "session_bucket"] if c in events.columns]].copy()
    masks = build_fixed_candidate_masks(events)
    for name in _split_csv_names(args.candidate_names):
        if name in masks:
            signal_flags[name] = masks[name]["mask"].fillna(False).astype(bool).to_numpy()

    write_csv(signal_flags, out_dir / "01_signals.csv")
    write_csv(trade_out, out_dir / "02_trades.csv")
    write_csv(summary, out_dir / "03_summary.csv")
    write_csv(yearly, out_dir / "04_yearly.csv")
    write_csv(monthly, out_dir / "05_monthly.csv")
    write_csv(equity_curve, out_dir / "06_equity_curve.csv")
    write_csv(cost_stress, out_dir / "07_stress_cost.csv")
    write_csv(delay_stress, out_dir / "08_stress_delay.csv")
    variant_compare = summary.sort_values(["status" if "status" in summary.columns else "profit_factor", "return_total"], ascending=[True, False]) if not summary.empty else summary
    write_csv(variant_compare, out_dir / "09_variant_compare.csv")
    audit_cols = [
        "variant_name", "signal_time", "entry_time", "exit_time", "exit_reason", "net_return_on_equity", "mae_on_equity", "mfe_on_equity",
        "bars_held", "filled_weight", "stop_name", "stop_hit", "add_count", "avg_entry_price", "exit_price", "signal_close", "signal_low", "structural_stop_level",
    ]
    write_csv(trade_out[[c for c in audit_cols if c in trade_out.columns]].copy(), out_dir / "10_trade_audit.csv")
    write_csv(registry, out_dir / "11_edge_registry_update.csv")

    meta = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "studied_events": int(len(events)),
        "variants": [v.variant_name for v in variants],
        "candidate_names": _split_csv_names(args.candidate_names),
        "entry_schemes": _split_csv_names(args.entry_schemes),
        "exit_horizons": _parse_number_list(args.exit_horizons, cast=int, name="exit_horizons"),
        "stop_specs": [s.name for s in parse_stop_specs(args.stop_specs)],
        "conflict_policy": args.conflict_policy,
        "entry_delay_bars": int(args.entry_delay_bars),
        "cost_multipliers": _parse_number_list(args.cost_multipliers, cast=float, name="cost_multipliers"),
        "delay_bars_list": _parse_number_list(args.delay_bars_list, cast=int, name="delay_bars_list"),
        "skip_cost_stress": bool(args.skip_cost_stress),
        "skip_delay_stress": bool(args.skip_delay_stress),
        "leakage_guard": {
            "data_loader": "OKXTradeBarLoader via upstream load_trade_bars",
            "signal_filters": "A/B/C/ABC from no-leakage probe; no full-sample qcut filters",
            "rolling_thresholds": "feature.shift(1).rolling(...).quantile(...) in upstream build_enriched_features",
            "entry_timing": "signal on closed bar; entry at future open after entry_delay_bars",
            "conflict_resolution": "one position at a time; skip overlapping signals",
            "scale_in": "post-entry only; capped max_position_weight; no unlimited martingale",
        },
    }
    print("[write] 12_lab_meta.json", flush=True)
    (out_dir / "12_lab_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[done] wrote strategy backtest probe reports to {out_dir}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_backtest_probe(args)


if __name__ == "__main__":
    main()
