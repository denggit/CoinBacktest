#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MF Low Sweep hard-stop + cooldown research pass 02.

Research-only. This script does not change the formal MF backtest, Portfolio V1,
edge library, or any live execution code.

Fixed entry anchor:
    ETH_EDGE_MF_LOW_SWEEP_A0_FOOTPRINT
    A0_fp_abs_delta_high + single_swing + next_open

Research goal:
    Keep the time48 low-sweep rebound thesis intact, but add a true emergency
    fuse: a wider hard stop and an optional 24h strategy cooldown after the fuse
    fires. Stop levels are not selected from liquidation distance. The script
    first audits the baseline trade-path MAE/MFE distribution and reports how
    each stop threshold would trade off winner stop-outs, loser cut-rate,
    extreme-path coverage, profit retention, and drawdown.

Causality convention:
- signal_time is a closed 1m trade bar;
- entry is next open plus optional delay stress;
- protective hard stops are post-entry intrabar risk controls;
- cooldown starts from the stop exit time and blocks new entries for N hours;
- entry risk gates use only closed signal-bar/context fields;
- no same-bar closed high/low/orderflow is used to enter at that same bar open.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.edge_lib.mf_low_sweep.config import EDGE_ID, VARIANT_NAME, build_mf_args  # noqa: E402
from src.edge_lib.mf_low_sweep.events import load_trade_bars  # noqa: E402
from src.edge_lib.mf_low_sweep.exits import (  # noqa: E402
    build_market_cache,
    equity_and_dd,
    payoff_ratio,
    profit_factor,
    top_winner_share,
)
from src.edge_lib.mf_low_sweep.features import prepare_events_and_context  # noqa: E402
from src.edge_lib.mf_low_sweep.signals import build_candidate_layer_masks, build_support_mask  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "02_low_sweep_hard_stop_cooldown_research"
SCRIPT_VERSION = "v1.1.0"
EXPERIMENT_ID = "ETH_MF_LOW_SWEEP_HARD_STOP_COOLDOWN_R02"
TITLE = "ETH MF Low Sweep Hard Stop Cooldown Research 02"
DEFAULT_OUT_DIR = "data/reports/research/mf_low_sweep/02_low_sweep_hard_stop_cooldown_research"
BASELINE_VARIANT = "baseline_time48_no_stop"


@dataclass(frozen=True)
class RiskVariant:
    variant_name: str
    family: str
    description: str
    stop_pct: float | None = None
    atr_mult: float | None = None
    early_bars: int | None = None
    early_mfe_cap: float | None = None
    early_close_floor: float | None = None
    early_buy_share_max: float | None = None
    early_delta_max: float | None = None
    giveback_trigger: float | None = None
    giveback_floor: float | None = None
    skip_atr_pct_ge: float | None = None
    skip_down_spike_pct_ge: float | None = None
    cooldown_hours_after_stop: float = 0.0
    entry_extra_delay: int = 0
    horizon: int = 48


@dataclass(frozen=True)
class FlowArrays:
    buy_notional: np.ndarray
    sell_notional: np.ndarray
    delta_notional: np.ndarray
    total_notional: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research MF low-sweep wide hard-stop plus stop-hit cooldown controls")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage", "--slippage-pct", dest="slippage", type=float, default=0.0)
    p.add_argument("--round-trip-cost-pct", type=float, default=0.0011, help="Default full entry+exit cost. Project default OKX round-trip cost is 0.11%%.")
    p.add_argument("--cost-mults", default="1.0,1.5,2.0,3.0")
    p.add_argument("--delay-bars", default="0,1,2,3")
    p.add_argument("--leverage", type=float, default=15.0)
    p.add_argument(
        "--liquidation-proxy-pct",
        type=float,
        default=0.0,
        help="Price move against long used for liquidation/tail proxy. 0 = auto 0.85/leverage.",
    )
    p.add_argument("--mf-exposure", type=float, default=1.5, help="Portfolio notional multiple used for account-level MAE diagnostics.")
    p.add_argument("--stop-extra-slippage-pct", type=float, default=0.00050, help="Extra conservative slippage when a stop is hit.")
    p.add_argument("--flow-window", type=int, default=5)
    p.add_argument("--focus-dates", default="2025-02-03")
    p.add_argument("--cooldown-hours", type=float, default=24.0, help="Default strategy cooldown hours after a hard stop fires.")
    p.add_argument(
        "--hard-stop-pcts",
        default="0.025,0.030,0.035,0.040,0.045,0.050,0.055,0.060,0.065",
        help="Actual hard-stop variants to simulate. Percent is price move against entry, e.g. 0.045 = 4.5%%.",
    )
    p.add_argument(
        "--threshold-grid-pcts",
        default="0.015,0.020,0.025,0.030,0.035,0.040,0.045,0.050,0.055,0.060,0.065,0.070,0.080",
        help="MAE/MFE diagnostic grid for stop threshold trade-off analysis. Does not by itself add variants.",
    )
    p.add_argument("--save-trades", type=int, default=200000)
    p.add_argument("--save-event-sample", type=int, default=10000)
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    args = p.parse_args(argv)
    args.slippage_pct = float(args.slippage)
    return args


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


def parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def split_csv(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def write_csv(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None or df.empty:
        pd.DataFrame().to_csv(path, index=False)
        print(f"[write] {label}: empty -> {path}", flush=True)
        return
    df.to_csv(path, index=False)
    print(f"[write] {label}: rows={len(df):,} cols={len(df.columns):,} -> {path}", flush=True)


def write_json(data: dict[str, Any], path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(f"[write] {label} -> {path}", flush=True)


def finite_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def col_array(frame: pd.DataFrame, name: str, fallback: float = np.nan) -> np.ndarray:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
    return np.full(len(frame), float(fallback), dtype=float)


def build_flow_arrays(bars: pd.DataFrame) -> FlowArrays:
    buy = col_array(bars, "buy_notional")
    sell = col_array(bars, "sell_notional")
    delta = col_array(bars, "delta_notional")
    if not np.isfinite(delta).any():
        delta = buy - sell
    total = buy + sell
    notional = col_array(bars, "notional")
    if np.isfinite(notional).any():
        total = np.where(np.isfinite(total) & (total > 0), total, notional)
    return FlowArrays(buy_notional=buy, sell_notional=sell, delta_notional=delta, total_notional=total)


def mf_args_from_cli(args: argparse.Namespace) -> SimpleNamespace:
    mf_args = build_mf_args(args)
    # The edge config intentionally freezes the formal entry/exit. For research,
    # preserve its data/context defaults but expose a few report/runtime flags.
    mf_args.no_progress = bool(args.no_progress)
    mf_args.progress_every = int(args.progress_every)
    return mf_args


def risk_variants(args: argparse.Namespace | None = None) -> list[RiskVariant]:
    cooldown = 24.0 if args is None else float(getattr(args, "cooldown_hours", 24.0))
    variants: list[RiskVariant] = [
        RiskVariant(
            variant_name=BASELINE_VARIANT,
            family="baseline",
            description="Current MF anchor: time48 close exit, no stop, no cooldown.",
        )
    ]

    # Wider hard stops are emergency fuses, not profit optimizers. They are
    # tested from MAE/MFE path trade-off grids, not chosen from liquidation
    # distance. The liquidation proxy remains only a tail/execution danger audit.
    wide_stops = parse_float_list(getattr(args, "hard_stop_pcts", "0.025,0.030,0.035,0.040,0.045,0.050,0.055,0.060,0.065")) if args is not None else [0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065]
    for stop in wide_stops:
        tag = int(round(stop * 10000))
        variants.append(
            RiskVariant(
                variant_name=f"hard_stop_{tag:04d}_time48",
                family="wide_hard_stop",
                description=f"Wide hard stop at -{stop:.2%}; no cooldown; time48 fallback.",
                stop_pct=stop,
            )
        )
        variants.append(
            RiskVariant(
                variant_name=f"hard_stop_{tag:04d}_cooldown{int(cooldown)}h_time48",
                family="wide_hard_stop_cooldown",
                description=f"Wide hard stop at -{stop:.2%}; if hit, block this MF strategy for {cooldown:g}h; time48 fallback.",
                stop_pct=stop,
                cooldown_hours_after_stop=cooldown,
            )
        )

    # Risk gates are filters, not exits. R01 suggested that skipping extreme
    # waterfall/ATR states may be more efficient than chopping every trade.
    gate_specs = [
        ("atr0080", 0.008, None),
        ("atr0100", 0.010, None),
        ("spike0700", None, 0.070),
        ("atr0080_or_spike0700", 0.008, 0.070),
        ("atr0100_or_spike0700", 0.010, 0.070),
    ]
    for gate_name, atr, spike in gate_specs:
        variants.append(
            RiskVariant(
                variant_name=f"gate_{gate_name}_time48",
                family="entry_risk_gate",
                description=f"Skip entry on gate={gate_name}; otherwise baseline time48.",
                skip_atr_pct_ge=atr,
                skip_down_spike_pct_ge=spike,
            )
        )

    # Main candidates: keep the gate as first line of defense, then use a wide
    # hard stop + cooldown only as the emergency fuse. Use the middle of the
    # user/grid-defined range, because the final choice should come from the
    # MAE/MFE threshold trade-off table.
    combo_stops = [s for s in wide_stops if 0.035 <= float(s) <= 0.055]
    if not combo_stops:
        combo_stops = wide_stops[:3]
    for stop in combo_stops:
        tag = int(round(stop * 10000))
        for gate_name, atr, spike in [
            ("atr0080", 0.008, None),
            ("atr0100", 0.010, None),
            ("atr0080_or_spike0700", 0.008, 0.070),
            ("atr0100_or_spike0700", 0.010, 0.070),
        ]:
            variants.append(
                RiskVariant(
                    variant_name=f"gate_{gate_name}_hard_stop_{tag:04d}_cooldown{int(cooldown)}h_time48",
                    family="gate_stop_cooldown_combo",
                    description=f"Risk gate={gate_name}; hard stop -{stop:.2%}; stop-hit cooldown {cooldown:g}h; time48 fallback.",
                    stop_pct=stop,
                    skip_atr_pct_ge=atr,
                    skip_down_spike_pct_ge=spike,
                    cooldown_hours_after_stop=cooldown,
                )
            )

    # Do not add liquidation-buffer stops here. Liquidation distance is only an
    # audit/tail-risk reference, not the source for hard-stop selection.
    return variants

def resolved_liquidation_proxy_pct(args: argparse.Namespace) -> float:
    if float(args.liquidation_proxy_pct) > 0:
        return float(args.liquidation_proxy_pct)
    lev = max(float(args.leverage), 1e-9)
    # Conservative proxy: maintenance margin/mark-price path is not modeled, so
    # use only 85% of the naive 1/leverage move.
    return 0.85 / lev


def resolved_stop_pct(variant: RiskVariant, event: pd.Series | dict[str, object], args: argparse.Namespace) -> float | None:
    liq = resolved_liquidation_proxy_pct(args)
    if variant.stop_pct is not None:
        if variant.stop_pct < 0:
            return abs(float(variant.stop_pct)) * liq
        return float(variant.stop_pct)
    if variant.atr_mult is not None:
        atr_pct = finite_float(event.get("atr_pct", np.nan))
        if math.isfinite(atr_pct) and atr_pct > 0:
            return max(0.0025, float(variant.atr_mult) * atr_pct)
    return None


def recent_flow(flow: FlowArrays, start_pos: int, pos: int, window: int) -> tuple[float, float]:
    left = max(int(start_pos), int(pos) - max(1, int(window)) + 1)
    right = int(pos) + 1
    buy = float(np.nansum(flow.buy_notional[left:right]))
    sell = float(np.nansum(flow.sell_notional[left:right]))
    total = float(np.nansum(flow.total_notional[left:right]))
    if not math.isfinite(total) or total <= 0:
        total = buy + sell
    buy_share = buy / total if total > 0 else np.nan
    delta_pressure = float(np.nansum(flow.delta_notional[left:right])) / total if total > 0 else np.nan
    return buy_share, delta_pressure


def entry_is_skipped(event: pd.Series | dict[str, object], variant: RiskVariant) -> tuple[bool, str]:
    reasons: list[str] = []
    if variant.skip_atr_pct_ge is not None:
        atr = finite_float(event.get("atr_pct", np.nan))
        if math.isfinite(atr) and atr >= float(variant.skip_atr_pct_ge):
            reasons.append(f"skip_atr_ge_{variant.skip_atr_pct_ge:.4f}")
    if variant.skip_down_spike_pct_ge is not None:
        spike = finite_float(event.get("down_spike_pct", np.nan))
        if math.isfinite(spike) and spike >= float(variant.skip_down_spike_pct_ge):
            reasons.append(f"skip_spike_ge_{variant.skip_down_spike_pct_ge:.4f}")
    return (bool(reasons), "+".join(reasons))


def timing_stats(mtm_low: Sequence[float], mtm_high: Sequence[float]) -> dict[str, object]:
    if not mtm_low or not mtm_high:
        return {
            "mae_time_bars": np.nan,
            "mfe_time_bars": np.nan,
            "first_positive_high_bars": np.nan,
            "mae_before_mfe_flag": np.nan,
        }
    low_arr = np.asarray(mtm_low, dtype=float)
    high_arr = np.asarray(mtm_high, dtype=float)
    mae_pos = int(np.nanargmin(low_arr)) if np.isfinite(low_arr).any() else -1
    mfe_pos = int(np.nanargmax(high_arr)) if np.isfinite(high_arr).any() else -1
    pos_high = np.where(high_arr > 0)[0]
    return {
        "mae_time_bars": mae_pos,
        "mfe_time_bars": mfe_pos,
        "first_positive_high_bars": int(pos_high[0]) if len(pos_high) else np.nan,
        "mae_before_mfe_flag": bool(mae_pos <= mfe_pos) if mae_pos >= 0 and mfe_pos >= 0 else np.nan,
    }


def effective_round_trip_cost_pct(args: argparse.Namespace) -> float:
    value = finite_float(getattr(args, "round_trip_cost_pct", np.nan), np.nan)
    if math.isfinite(value) and value >= 0:
        return float(value)
    return 2.0 * (float(args.fee_rate) + float(args.slippage_pct))


def entry_cost(args: argparse.Namespace, cost_mult: float) -> float:
    return 0.5 * effective_round_trip_cost_pct(args) * float(cost_mult)


def exit_cost(args: argparse.Namespace, cost_mult: float) -> float:
    return 0.5 * effective_round_trip_cost_pct(args) * float(cost_mult)


def simulate_trade(
    *,
    event: pd.Series,
    signal_pos: int,
    variant: RiskVariant,
    bars: pd.DataFrame,
    market: Any,
    flow: FlowArrays,
    args: argparse.Namespace,
    cost_mult: float,
) -> dict[str, object]:
    skipped, skip_reason = entry_is_skipped(event, variant)
    if skipped:
        return {"valid": False, "skipped_by_filter": True, "invalid_reason": skip_reason}

    opens = market.open
    highs = market.high
    lows = market.low
    closes = market.close
    idx = market.index
    n = len(idx)

    base_entry_delay = 1
    entry_pos = int(signal_pos) + base_entry_delay + int(variant.entry_extra_delay)
    planned_exit_pos = int(signal_pos) + int(variant.horizon)
    if entry_pos >= n or planned_exit_pos >= n or planned_exit_pos <= entry_pos:
        return {"valid": False, "invalid_reason": "insufficient_future_bars"}

    entry_price = float(opens[entry_pos])
    if not math.isfinite(entry_price) or entry_price <= 0:
        return {"valid": False, "invalid_reason": "bad_entry_price"}

    liq_pct = resolved_liquidation_proxy_pct(args)
    liq_price = entry_price * (1.0 - liq_pct)
    stop_pct = resolved_stop_pct(variant, event, args)
    stop_price = entry_price * (1.0 - float(stop_pct)) if stop_pct is not None and stop_pct > 0 else None
    stop_to_liq_ratio = float(stop_pct) / float(liq_pct) if stop_pct is not None and stop_pct > 0 and liq_pct > 0 else np.nan
    stop_near_liq_proxy_flag = bool(math.isfinite(stop_to_liq_ratio) and stop_to_liq_ratio >= 0.85)
    stop_beyond_liq_proxy_flag = bool(math.isfinite(stop_to_liq_ratio) and stop_to_liq_ratio >= 1.0)

    exit_pos = planned_exit_pos
    exit_price = float(closes[planned_exit_pos])
    exit_reason = f"time_exit_h{int(variant.horizon)}"
    stop_hit = False
    liquidation_proxy_hit = False
    high_water = entry_price
    mtm_low: list[float] = []
    mtm_high: list[float] = []
    early_trigger_buy_share = np.nan
    early_trigger_delta_pressure = np.nan

    for pos in range(entry_pos, planned_exit_pos + 1):
        low_ret = float(lows[pos]) / entry_price - 1.0
        high_ret = float(highs[pos]) / entry_price - 1.0
        mtm_low.append(float(low_ret))
        mtm_high.append(float(high_ret))
        high_water = max(high_water, float(highs[pos]))
        # Conservative long-path assumption for a long position: a resting stop
        # above the liquidation proxy is allowed to fire before the path reaches
        # the deeper liquidation proxy level. If there is no such stop, or the
        # stop is below the proxy, then the low crossing the proxy is counted.
        if stop_price is not None and float(lows[pos]) <= float(stop_price):
            stop_hit = True
            exit_pos = int(pos)
            exit_price = float(stop_price) * (1.0 - float(args.stop_extra_slippage_pct))
            if variant.atr_mult is not None:
                exit_reason = "atr_stop"
            elif variant.stop_pct is not None and variant.stop_pct < 0:
                exit_reason = "liquidation_buffer_stop"
            else:
                exit_reason = "hard_stop"
            break
        if float(lows[pos]) <= liq_price:
            liquidation_proxy_hit = True

        mfe_now = high_water / entry_price - 1.0
        close_ret = float(closes[pos]) / entry_price - 1.0
        buy_share, delta_pressure = recent_flow(flow, entry_pos, pos, int(args.flow_window))

        if variant.early_bars is not None:
            if pos >= entry_pos + int(variant.early_bars):
                weak_mfe = mfe_now < float(variant.early_mfe_cap if variant.early_mfe_cap is not None else 0.0)
                weak_close = close_ret <= float(variant.early_close_floor if variant.early_close_floor is not None else 0.0)
                weak_buy = (not math.isfinite(buy_share)) or buy_share <= float(variant.early_buy_share_max if variant.early_buy_share_max is not None else 1.0)
                weak_delta = (not math.isfinite(delta_pressure)) or delta_pressure <= float(variant.early_delta_max if variant.early_delta_max is not None else 1.0)
                if weak_mfe and weak_close and weak_buy and weak_delta:
                    next_pos = min(pos + 1, planned_exit_pos)
                    exit_pos = int(next_pos)
                    exit_price = float(opens[next_pos]) if next_pos < n else float(closes[pos])
                    exit_reason = f"early_fail_b{int(variant.early_bars)}_next_open"
                    early_trigger_buy_share = buy_share
                    early_trigger_delta_pressure = delta_pressure
                    break

        if variant.giveback_trigger is not None:
            armed = mfe_now >= float(variant.giveback_trigger)
            weak_close = close_ret <= float(variant.giveback_floor if variant.giveback_floor is not None else 0.0)
            weak_buy = (not math.isfinite(buy_share)) or buy_share <= float(variant.early_buy_share_max if variant.early_buy_share_max is not None else 1.0)
            weak_delta = (not math.isfinite(delta_pressure)) or delta_pressure <= float(variant.early_delta_max if variant.early_delta_max is not None else 1.0)
            if armed and weak_close and weak_buy and weak_delta:
                next_pos = min(pos + 1, planned_exit_pos)
                exit_pos = int(next_pos)
                exit_price = float(opens[next_pos]) if next_pos < n else float(closes[pos])
                exit_reason = "weak_bounce_giveback_next_open"
                early_trigger_buy_share = buy_share
                early_trigger_delta_pressure = delta_pressure
                break

    gross = float(exit_price) / entry_price - 1.0
    net = gross - entry_cost(args, cost_mult) - exit_cost(args, cost_mult)
    timing = timing_stats(mtm_low, mtm_high)
    mae = float(np.nanmin(mtm_low)) if mtm_low else np.nan
    mfe = float(np.nanmax(mtm_high)) if mtm_high else np.nan
    account_mae_proxy = mae * float(args.mf_exposure) if math.isfinite(mae) else np.nan

    return {
        "valid": True,
        "variant_name": variant.variant_name,
        "variant_family": variant.family,
        "variant_description": variant.description,
        "candidate_layer": "A0_fp_abs_delta_high",
        "support_mode": "single_swing",
        "entry_mode": "next_open" if int(variant.entry_extra_delay) == 0 else f"next_open_delay{int(variant.entry_extra_delay)}",
        "exit_mode": "time48_risk_control",
        "cost_mult": float(cost_mult),
        "signal_time": event.get("signal_time"),
        "entry_time": idx[int(entry_pos)],
        "exit_time": idx[int(exit_pos)],
        "signal_pos": int(signal_pos),
        "entry_pos": int(entry_pos),
        "exit_pos": int(exit_pos),
        "entry_delay_bars_actual": int(entry_pos - signal_pos),
        "bars_held": int(exit_pos - entry_pos),
        "entry_reason": "next_open" if int(variant.entry_extra_delay) == 0 else f"next_open_delay{int(variant.entry_extra_delay)}",
        "exit_reason": exit_reason,
        "stop_hit": bool(stop_hit),
        "liquidation_proxy_hit": bool(liquidation_proxy_hit),
        "liquidation_proxy_pct": float(liq_pct),
        "liquidation_proxy_price": float(liq_price),
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "stop_price": float(stop_price) if stop_price is not None else np.nan,
        "stop_pct": float(stop_pct) if stop_pct is not None else np.nan,
        "stop_to_liq_ratio": float(stop_to_liq_ratio) if math.isfinite(stop_to_liq_ratio) else np.nan,
        "stop_near_liq_proxy_flag": bool(stop_near_liq_proxy_flag),
        "stop_beyond_liq_proxy_flag": bool(stop_beyond_liq_proxy_flag),
        "cooldown_hours_after_stop": float(variant.cooldown_hours_after_stop),
        "gross_return_on_equity": float(gross),
        "net_return_on_equity": float(net),
        "mae_on_equity": mae,
        "mfe_on_equity": mfe,
        "account_mae_proxy": account_mae_proxy,
        "mae_time_bars": timing["mae_time_bars"],
        "mfe_time_bars": timing["mfe_time_bars"],
        "first_positive_high_bars": timing["first_positive_high_bars"],
        "mae_before_mfe_flag": timing["mae_before_mfe_flag"],
        "early_trigger_buy_share": early_trigger_buy_share,
        "early_trigger_delta_pressure": early_trigger_delta_pressure,
        "signal_close": finite_float(event.get("close", np.nan)),
        "signal_open": finite_float(event.get("open", np.nan)),
        "signal_low": finite_float(event.get("low", np.nan)),
        "signal_high": finite_float(event.get("high", np.nan)),
        "swing_level": finite_float(event.get("swing_level", np.nan)),
        "swing_age": finite_float(event.get("swing_age", np.nan)),
        "down_spike_pct": finite_float(event.get("down_spike_pct", np.nan)),
        "close_pos_in_bar": finite_float(event.get("close_pos_in_bar", np.nan)),
        "large_trade_share": finite_float(event.get("large_trade_share", np.nan)),
        "atr_pct": finite_float(event.get("atr_pct", np.nan)),
        "fp_max_bucket_abs_delta_pressure": finite_float(event.get("fp_max_bucket_abs_delta_pressure", np.nan)),
        "session_bucket": event.get("session_bucket", "NA"),
        "event_name": event.get("event_name", ""),
        "event_variant": event.get("variant", ""),
    }


def select_a0_events(events: pd.DataFrame, mf_args: Any) -> pd.DataFrame:
    masks = build_candidate_layer_masks(events, mf_args)
    layer = masks.get("A0_fp_abs_delta_high", pd.Series(False, index=events.index)).fillna(False).astype(bool)
    support = build_support_mask(events, "single_swing", mf_args).fillna(False).astype(bool)
    selected = events.loc[layer & support].copy().sort_values("signal_time").reset_index(drop=True)
    return selected


def build_signal_positions(bars: pd.DataFrame, events: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if events.empty:
        return events.copy(), np.asarray([], dtype=int)
    times = pd.DatetimeIndex(pd.to_datetime(events["signal_time"], errors="coerce"))
    signal_pos = bars.index.get_indexer(times)
    valid = signal_pos >= 0
    out = events.loc[valid].copy().reset_index(drop=True)
    return out, signal_pos[valid]


def simulate_variant_set(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    variants: Sequence[RiskVariant],
    args: argparse.Namespace,
    *,
    cost_mult: float = 1.0,
    label: str = "[simulate] variants",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty or not variants:
        return pd.DataFrame(), pd.DataFrame()
    print("[cache] building market arrays", flush=True)
    market = build_market_cache(bars, mf_args_from_cli(args))
    flow = build_flow_arrays(bars)
    ev, positions = build_signal_positions(bars, events)

    total = max(1, len(ev) * len(variants))
    progress = ProgressReporter(label=label, total=total, every=max(1, int(args.progress_every)), enabled=not bool(args.no_progress))
    done = 0
    trade_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []

    for variant in variants:
        rows: list[dict[str, object]] = []
        counters = {
            "candidate_events": int(len(ev)),
            "input_events": int(len(ev)),
            "valid_trades": 0,
            "skipped_overlap": 0,
            "skipped_invalid": 0,
            "skipped_by_filter": 0,
            "skipped_by_cooldown": 0,
            "cooldown_triggers": 0,
        }
        last_exit_pos = -1
        cooldown_until: pd.Timestamp | None = None
        for event, signal_pos in zip(ev.to_dict("records"), positions):
            signal_pos_i = int(signal_pos)
            signal_time = pd.to_datetime(event.get("signal_time"), errors="coerce")
            if cooldown_until is not None and pd.notna(signal_time) and signal_time < cooldown_until:
                counters["skipped_by_cooldown"] += 1
                done += 1
                progress.update(done)
                continue
            if signal_pos_i <= last_exit_pos:
                counters["skipped_overlap"] += 1
                done += 1
                progress.update(done)
                continue
            rec = simulate_trade(
                event=pd.Series(event),
                signal_pos=signal_pos_i,
                variant=variant,
                bars=bars,
                market=market,
                flow=flow,
                args=args,
                cost_mult=float(cost_mult),
            )
            if rec.get("valid"):
                rows.append(rec)
                counters["valid_trades"] += 1
                last_exit_pos = int(rec.get("exit_pos", signal_pos_i))
                if bool(rec.get("stop_hit", False)) and float(variant.cooldown_hours_after_stop) > 0:
                    exit_time = pd.to_datetime(rec.get("exit_time"), errors="coerce")
                    if pd.notna(exit_time):
                        cooldown_until = exit_time + pd.Timedelta(hours=float(variant.cooldown_hours_after_stop))
                        counters["cooldown_triggers"] += 1
            else:
                if rec.get("skipped_by_filter"):
                    counters["skipped_by_filter"] += 1
                else:
                    counters["skipped_invalid"] += 1
            done += 1
            progress.update(done)
        trades = pd.DataFrame(rows)
        if not trades.empty:
            trade_parts.append(trades)
        summary_rows.append(summarize_trades_for_variant(trades, variant, counters, args, cost_mult))
    progress.close()
    all_trades = pd.concat(trade_parts, ignore_index=True, sort=False) if trade_parts else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return all_trades, summary


def max_consecutive_losses(values: pd.Series) -> int:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    best = cur = 0
    for val in arr:
        if val < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def max_gap_days(times: pd.Series) -> float:
    t = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    if len(t) <= 1:
        return float("nan")
    diffs = t.diff().dropna().dt.total_seconds() / 86400.0
    return float(diffs.max()) if not diffs.empty else float("nan")


def summarize_trades_for_variant(
    trades: pd.DataFrame,
    variant: RiskVariant,
    counters: dict[str, int],
    args: argparse.Namespace,
    cost_mult: float,
) -> dict[str, object]:
    out: dict[str, object] = {
        "variant_name": variant.variant_name,
        "variant_family": variant.family,
        "description": variant.description,
        "cost_mult": float(cost_mult),
        "horizon": int(variant.horizon),
        "stop_pct_config": variant.stop_pct,
        "atr_mult": variant.atr_mult,
        "early_bars": variant.early_bars,
        "early_mfe_cap": variant.early_mfe_cap,
        "giveback_trigger": variant.giveback_trigger,
        "skip_atr_pct_ge": variant.skip_atr_pct_ge,
        "skip_down_spike_pct_ge": variant.skip_down_spike_pct_ge,
        "cooldown_hours_after_stop": float(variant.cooldown_hours_after_stop),
        **counters,
    }
    if trades.empty:
        out.update({"trades": 0})
        return out
    x = pd.to_numeric(trades["net_return_on_equity"], errors="coerce").fillna(0.0)
    equity, dd = equity_and_dd(x, 1.0)
    wins = x[x > 0]
    losses = x[x < 0]
    first = pd.to_datetime(trades["entry_time"], errors="coerce").min()
    last = pd.to_datetime(trades["exit_time"], errors="coerce").max()
    days = max((last - first).total_seconds() / 86400.0, 1e-9) if pd.notna(first) and pd.notna(last) else 1e-9
    total_ret = float(equity.iloc[-1] - 1.0)
    out.update(
        {
            "trades": int(len(trades)),
            "trades_per_month": float(len(trades) / max(days / 30.4375, 1e-9)),
            "max_days_without_trade": max_gap_days(trades["entry_time"]),
            "return_total": total_ret,
            "return_annualized": float((1.0 + total_ret) ** (365.0 / days) - 1.0) if total_ret > -1.0 else -1.0,
            "mean_return": float(x.mean()),
            "median_return": float(x.median()),
            "win_rate": float((x > 0).mean()),
            "avg_win": float(wins.mean()) if not wins.empty else np.nan,
            "avg_loss": float(losses.mean()) if not losses.empty else np.nan,
            "payoff_ratio": payoff_ratio(x),
            "profit_factor": profit_factor(x),
            "max_drawdown": float(dd.min()),
            "max_consecutive_losses": max_consecutive_losses(x),
            "top5_winner_share": top_winner_share(x),
            "worst_trade": float(x.min()),
            "best_trade": float(x.max()),
            "ret_p01": float(x.quantile(0.01)),
            "ret_p05": float(x.quantile(0.05)),
            "mae_mean": float(pd.to_numeric(trades["mae_on_equity"], errors="coerce").mean()),
            "mae_p01": float(pd.to_numeric(trades["mae_on_equity"], errors="coerce").quantile(0.01)),
            "mae_p05": float(pd.to_numeric(trades["mae_on_equity"], errors="coerce").quantile(0.05)),
            "mae_min": float(pd.to_numeric(trades["mae_on_equity"], errors="coerce").min()),
            "mfe_mean": float(pd.to_numeric(trades["mfe_on_equity"], errors="coerce").mean()),
            "avg_bars_held": float(pd.to_numeric(trades["bars_held"], errors="coerce").mean()),
            "stop_hit_rate": float(pd.to_numeric(trades["stop_hit"], errors="coerce").mean()),
            "liquidation_proxy_hit_rate": float(pd.to_numeric(trades["liquidation_proxy_hit"], errors="coerce").mean()),
            "account_mae_proxy_min": float(pd.to_numeric(trades["account_mae_proxy"], errors="coerce").min()),
            "resolved_stop_pct_median": float(pd.to_numeric(trades.get("stop_pct", np.nan), errors="coerce").median()),
            "stop_to_liq_ratio_median": float(pd.to_numeric(trades.get("stop_to_liq_ratio", np.nan), errors="coerce").median()),
            "stop_near_liq_proxy_rate": float(pd.to_numeric(trades.get("stop_near_liq_proxy_flag", False), errors="coerce").fillna(0).mean()),
            "stop_beyond_liq_proxy_rate": float(pd.to_numeric(trades.get("stop_beyond_liq_proxy_flag", False), errors="coerce").fillna(0).mean()),
        }
    )
    return out


def summarize_by_period(trades: pd.DataFrame, period: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    if period == "year":
        df["period"] = df["entry_time"].dt.year.astype("Int64").astype(str)
    elif period == "month":
        df["period"] = df["entry_time"].dt.to_period("M").astype(str)
    elif period == "day":
        df["period"] = df["entry_time"].dt.strftime("%Y-%m-%d")
    else:
        raise ValueError(f"Unsupported period={period}")
    rows: list[dict[str, object]] = []
    for (name, per), grp in df.groupby(["variant_name", "period"], dropna=False):
        x = pd.to_numeric(grp["net_return_on_equity"], errors="coerce").fillna(0.0)
        equity, dd = equity_and_dd(x, 1.0)
        rows.append(
            {
                period: per,
                "variant_name": name,
                "variant_family": grp["variant_family"].iloc[0] if "variant_family" in grp else "",
                "trades": int(len(grp)),
                "return_total": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
                "mean_return": float(x.mean()) if len(x) else np.nan,
                "median_return": float(x.median()) if len(x) else np.nan,
                "win_rate": float((x > 0).mean()) if len(x) else np.nan,
                "profit_factor": profit_factor(x),
                "max_drawdown": float(dd.min()) if len(dd) else np.nan,
                "worst_trade": float(x.min()) if len(x) else np.nan,
                "liquidation_proxy_hits": int(pd.to_numeric(grp.get("liquidation_proxy_hit", False), errors="coerce").fillna(0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["variant_name", period]).reset_index(drop=True)


def build_tail_risk(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for name, grp in trades.groupby("variant_name", dropna=False):
        x = pd.to_numeric(grp["net_return_on_equity"], errors="coerce").dropna()
        mae = pd.to_numeric(grp["mae_on_equity"], errors="coerce").dropna()
        worst = grp.sort_values("mae_on_equity", ascending=True).head(5)
        rows.append(
            {
                "variant_name": name,
                "variant_family": grp["variant_family"].iloc[0] if "variant_family" in grp else "",
                "trades": int(len(grp)),
                "ret_p01": float(x.quantile(0.01)) if len(x) else np.nan,
                "ret_p05": float(x.quantile(0.05)) if len(x) else np.nan,
                "mae_p01": float(mae.quantile(0.01)) if len(mae) else np.nan,
                "mae_p05": float(mae.quantile(0.05)) if len(mae) else np.nan,
                "mae_min": float(mae.min()) if len(mae) else np.nan,
                "liquidation_proxy_hits": int(pd.to_numeric(grp["liquidation_proxy_hit"], errors="coerce").fillna(0).sum()),
                "liquidation_proxy_hit_rate": float(pd.to_numeric(grp["liquidation_proxy_hit"], errors="coerce").fillna(0).mean()),
                "worst_path_entry_time": worst["entry_time"].iloc[0] if not worst.empty else pd.NaT,
                "worst_path_signal_time": worst["signal_time"].iloc[0] if not worst.empty else pd.NaT,
                "worst_path_mae": float(worst["mae_on_equity"].iloc[0]) if not worst.empty else np.nan,
                "worst_path_net": float(worst["net_return_on_equity"].iloc[0]) if not worst.empty else np.nan,
                "worst_5_path_mae_sum": float(pd.to_numeric(worst["mae_on_equity"], errors="coerce").sum()) if not worst.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["liquidation_proxy_hit_rate", "mae_min"], ascending=[False, True]).reset_index(drop=True)



def _safe_quantile(series: pd.Series, q: float) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return float("nan")
    return float(s.quantile(float(q)))


def _baseline_only(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "variant_name" not in trades.columns:
        return pd.DataFrame()
    base = trades.loc[trades["variant_name"].eq(BASELINE_VARIANT)].copy()
    if base.empty:
        return pd.DataFrame()
    base["entry_time"] = pd.to_datetime(base["entry_time"], errors="coerce")
    base["net_return_on_equity"] = pd.to_numeric(base["net_return_on_equity"], errors="coerce")
    base["mae_on_equity"] = pd.to_numeric(base["mae_on_equity"], errors="coerce")
    base["mfe_on_equity"] = pd.to_numeric(base["mfe_on_equity"], errors="coerce")
    base["mae_abs"] = (-base["mae_on_equity"]).clip(lower=0.0)
    base["mfe_pos"] = base["mfe_on_equity"].clip(lower=0.0)
    denom = base["mae_abs"].replace(0.0, np.nan)
    base["mfe_to_mae_abs_ratio"] = base["mfe_pos"] / denom
    return base


def build_mae_mfe_distribution(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Distribution table used to choose stops from observed path behavior.

    The point is to avoid selecting a hard stop from leverage/liquidation math.
    We split the current baseline trades into winners, losers, deep winners,
    and extreme paths, then show the MAE/MFE quantiles that matter for stop
    placement: a stop should not kill too many historical winners before they
    rebound, but it should cut a meaningful share of bad losers/extreme paths.
    """
    base = _baseline_only(trades)
    if base.empty:
        return pd.DataFrame()
    focus_dates = set(split_csv(getattr(args, "focus_dates", "")))
    entry_dates = base["entry_time"].dt.strftime("%Y-%m-%d")
    mae_abs_q95 = _safe_quantile(base["mae_abs"], 0.95)
    mae_abs_q90 = _safe_quantile(base["mae_abs"], 0.90)

    groups: list[tuple[str, pd.Series, str]] = [
        ("all_baseline", pd.Series(True, index=base.index), "All current time48/no-stop baseline trades."),
        ("baseline_winners", base["net_return_on_equity"] > 0, "Trades that ended positive under baseline time48."),
        ("baseline_losers", base["net_return_on_equity"] <= 0, "Trades that ended flat/negative under baseline time48."),
        (
            "winner_deep_mae_top10pct",
            (base["net_return_on_equity"] > 0) & (base["mae_abs"] >= mae_abs_q90),
            "Historical winners that first went deeply underwater; stops below this area may kill rebound winners.",
        ),
        (
            "extreme_path_top5pct_mae",
            base["mae_abs"] >= mae_abs_q95,
            "Worst 5% baseline paths by MAE; a useful emergency stop should cover most of these.",
        ),
        (
            "losers_with_positive_mfe",
            (base["net_return_on_equity"] <= 0) & (base["mfe_pos"] > 0),
            "Losers that had some favorable excursion before failing; useful for giveback/early-fail follow-up.",
        ),
        (
            "losers_no_positive_mfe",
            (base["net_return_on_equity"] <= 0) & (base["mfe_pos"] <= 0),
            "Losers that never traded favorably after entry; useful for fast failure diagnostics.",
        ),
    ]
    if focus_dates:
        groups.append(("focus_dates", entry_dates.isin(focus_dates), f"Focus dates: {','.join(sorted(focus_dates))}."))
    if "liquidation_proxy_hit" in base.columns:
        groups.append(("tail_proxy_hit", pd.to_numeric(base["liquidation_proxy_hit"], errors="coerce").fillna(0).astype(bool), "Trades whose path touched the tail-risk proxy; not a stop source."))

    rows: list[dict[str, object]] = []
    for group_name, mask, description in groups:
        grp = base.loc[mask.fillna(False)].copy()
        if grp.empty:
            rows.append({"group": group_name, "description": description, "trades": 0})
            continue
        x = grp["net_return_on_equity"]
        mae = grp["mae_on_equity"]
        mae_abs = grp["mae_abs"]
        mfe = grp["mfe_on_equity"]
        rows.append(
            {
                "group": group_name,
                "description": description,
                "trades": int(len(grp)),
                "share_of_baseline_trades": float(len(grp) / max(len(base), 1)),
                "net_sum": float(x.sum()),
                "net_mean": float(x.mean()),
                "win_rate": float((x > 0).mean()),
                "mae_min": float(mae.min()),
                "mae_p01": _safe_quantile(mae, 0.01),
                "mae_p05": _safe_quantile(mae, 0.05),
                "mae_p10": _safe_quantile(mae, 0.10),
                "mae_p25": _safe_quantile(mae, 0.25),
                "mae_p50": _safe_quantile(mae, 0.50),
                "mae_abs_p50": _safe_quantile(mae_abs, 0.50),
                "mae_abs_p75": _safe_quantile(mae_abs, 0.75),
                "mae_abs_p80": _safe_quantile(mae_abs, 0.80),
                "mae_abs_p85": _safe_quantile(mae_abs, 0.85),
                "mae_abs_p90": _safe_quantile(mae_abs, 0.90),
                "mae_abs_p95": _safe_quantile(mae_abs, 0.95),
                "mae_abs_p975": _safe_quantile(mae_abs, 0.975),
                "mae_abs_p99": _safe_quantile(mae_abs, 0.99),
                "mfe_p50": _safe_quantile(mfe, 0.50),
                "mfe_p75": _safe_quantile(mfe, 0.75),
                "mfe_p90": _safe_quantile(mfe, 0.90),
                "mfe_p95": _safe_quantile(mfe, 0.95),
                "mfe_p99": _safe_quantile(mfe, 0.99),
                "mfe_to_mae_abs_ratio_p50": _safe_quantile(grp["mfe_to_mae_abs_ratio"], 0.50),
                "mfe_to_mae_abs_ratio_p75": _safe_quantile(grp["mfe_to_mae_abs_ratio"], 0.75),
                "median_bars_to_mae": _safe_quantile(grp.get("mae_time_bars", pd.Series(dtype=float)), 0.50),
                "median_bars_to_mfe": _safe_quantile(grp.get("mfe_time_bars", pd.Series(dtype=float)), 0.50),
                "mae_before_mfe_rate": float(pd.to_numeric(grp.get("mae_before_mfe_flag", False), errors="coerce").fillna(0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_stop_threshold_tradeoff(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Evaluate candidate hard-stop thresholds against baseline MAE/MFE paths.

    This is an approximate threshold-selection diagnostic. It does not replace
    the real variant simulation because intrabar path ordering is not fully
    observable from 1m bars. It is still useful because it answers the key
    question: at a stop level X, how many historical winners would have been
    killed before rebounding, and how many losers/extreme paths would have been
    cut earlier?
    """
    base = _baseline_only(trades)
    if base.empty:
        return pd.DataFrame()
    grid = parse_float_list(getattr(args, "threshold_grid_pcts", ""))
    if not grid:
        grid = parse_float_list(getattr(args, "hard_stop_pcts", ""))
    if not grid:
        grid = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]

    net = base["net_return_on_equity"].fillna(0.0)
    mae_abs = base["mae_abs"].fillna(0.0)
    winners = net > 0
    losers = net <= 0
    tail_cutoff = _safe_quantile(mae_abs, 0.95)
    extreme = mae_abs >= tail_cutoff
    base_equity, base_dd = equity_and_dd(net, 1.0)
    base_total = float(base_equity.iloc[-1] - 1.0) if len(base_equity) else 0.0
    base_sum = float(net.sum())
    stop_extra = float(getattr(args, "stop_extra_slippage_pct", 0.0))
    cost = effective_round_trip_cost_pct(args)

    rows: list[dict[str, object]] = []
    for stop in grid:
        stop = float(stop)
        if stop <= 0:
            continue
        hit = mae_abs >= stop
        proxy_stopped_net = -stop - stop_extra - cost
        proxy_net = net.copy()
        proxy_net.loc[hit] = proxy_stopped_net
        proxy_equity, proxy_dd = equity_and_dd(proxy_net, 1.0)
        proxy_total = float(proxy_equity.iloc[-1] - 1.0) if len(proxy_equity) else 0.0
        winner_hit = hit & winners
        loser_hit = hit & losers
        extreme_hit = hit & extreme
        killed_winner_profit = float(net.loc[winner_hit].sum()) if winner_hit.any() else 0.0
        loser_delta = float((proxy_net.loc[loser_hit] - net.loc[loser_hit]).sum()) if loser_hit.any() else 0.0
        total_delta_sum = float(proxy_net.sum() - net.sum())
        false_true_ratio = float(winner_hit.sum() / max(int(loser_hit.sum()), 1))
        winner_stop_rate = float(winner_hit.sum() / max(int(winners.sum()), 1))
        loser_cut_rate = float(loser_hit.sum() / max(int(losers.sum()), 1))
        extreme_cover_rate = float(extreme_hit.sum() / max(int(extreme.sum()), 1))
        profit_retention = float(proxy_total / base_total) if abs(base_total) > 1e-12 else np.nan

        decision = "diagnostic"
        reason_parts: list[str] = []
        if winner_stop_rate <= 0.05 and extreme_cover_rate >= 0.70:
            decision = "candidate_conservative"
            reason_parts.append("kills <=5% baseline winners and covers >=70% extreme MAE paths")
        elif winner_stop_rate <= 0.10 and extreme_cover_rate >= 0.70 and profit_retention >= 0.70:
            decision = "candidate_balanced"
            reason_parts.append("kills <=10% winners, covers extreme paths, keeps >=70% compounded return proxy")
        elif winner_stop_rate > 0.15:
            decision = "too_tight"
            reason_parts.append("winner stop-out rate >15%")
        elif extreme_cover_rate < 0.50:
            decision = "too_loose_for_extremes"
            reason_parts.append("covers <50% of worst 5% MAE paths")
        if loser_cut_rate < 0.20:
            reason_parts.append("cuts <20% baseline losers")
        if total_delta_sum < 0:
            reason_parts.append("approx additive PnL lower than baseline")

        rows.append(
            {
                "stop_pct": stop,
                "stop_label": f"{stop:.2%}",
                "baseline_trades": int(len(base)),
                "stop_hit_count": int(hit.sum()),
                "stop_hit_rate": float(hit.mean()),
                "baseline_winner_count": int(winners.sum()),
                "baseline_loser_count": int(losers.sum()),
                "winner_stop_count": int(winner_hit.sum()),
                "winner_stop_rate": winner_stop_rate,
                "winner_survival_rate": float(1.0 - winner_stop_rate),
                "loser_cut_count": int(loser_hit.sum()),
                "loser_cut_rate": loser_cut_rate,
                "extreme_mae_cutoff": float(tail_cutoff),
                "extreme_path_count": int(extreme.sum()),
                "extreme_path_cover_count": int(extreme_hit.sum()),
                "extreme_path_cover_rate": extreme_cover_rate,
                "false_winner_to_loser_cut_ratio": false_true_ratio,
                "killed_winner_baseline_profit_sum": killed_winner_profit,
                "loser_cut_proxy_delta_sum": loser_delta,
                "total_proxy_delta_sum": total_delta_sum,
                "baseline_compounded_return": base_total,
                "proxy_compounded_return": proxy_total,
                "proxy_return_retention_vs_baseline": profit_retention,
                "baseline_additive_net_sum": base_sum,
                "proxy_additive_net_sum": float(proxy_net.sum()),
                "proxy_win_rate": float((proxy_net > 0).mean()),
                "proxy_profit_factor": profit_factor(proxy_net),
                "baseline_max_drawdown": float(base_dd.min()) if len(base_dd) else np.nan,
                "proxy_max_drawdown": float(proxy_dd.min()) if len(proxy_dd) else np.nan,
                "proxy_stopped_trade_net_assumption": proxy_stopped_net,
                "decision": decision,
                "reason": ";".join(reason_parts) if reason_parts else "threshold trade-off only; verify with full variant simulation",
            }
        )
    return pd.DataFrame(rows).sort_values(["decision", "winner_stop_rate", "extreme_path_cover_rate", "proxy_return_retention_vs_baseline"], ascending=[True, True, False, False]).reset_index(drop=True)


def build_extreme_trade_paths(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    base = _baseline_only(trades)
    if base.empty:
        return pd.DataFrame()
    focus_dates = set(split_csv(getattr(args, "focus_dates", "")))
    entry_dates = base["entry_time"].dt.strftime("%Y-%m-%d")
    worst_mae = base.sort_values("mae_on_equity", ascending=True).head(30).copy()
    worst_mae["extreme_reason"] = "worst_mae"
    worst_net = base.sort_values("net_return_on_equity", ascending=True).head(30).copy()
    worst_net["extreme_reason"] = "worst_net"
    parts = [worst_mae, worst_net]
    if focus_dates:
        focus = base.loc[entry_dates.isin(focus_dates)].copy()
        focus["extreme_reason"] = "focus_date"
        parts.append(focus)
    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out.drop_duplicates(["signal_time", "entry_time", "exit_time"], keep="first")
    cols = [
        "extreme_reason",
        "signal_time",
        "entry_time",
        "exit_time",
        "exit_reason",
        "bars_held",
        "entry_price",
        "exit_price",
        "net_return_on_equity",
        "mae_on_equity",
        "mfe_on_equity",
        "mae_abs",
        "mfe_to_mae_abs_ratio",
        "mae_time_bars",
        "mfe_time_bars",
        "first_positive_high_bars",
        "mae_before_mfe_flag",
        "liquidation_proxy_hit",
        "down_spike_pct",
        "atr_pct",
        "fp_max_bucket_abs_delta_pressure",
        "session_bucket",
    ]
    return out[[c for c in cols if c in out.columns]].sort_values(["extreme_reason", "mae_on_equity", "net_return_on_equity"]).reset_index(drop=True)

def build_focus_day_report(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    dates = set(split_csv(args.focus_dates))
    if not dates:
        return pd.DataFrame()
    out = trades.copy()
    out["entry_date"] = pd.to_datetime(out["entry_time"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.loc[out["entry_date"].isin(dates)].copy()
    cols = [
        "variant_name",
        "variant_family",
        "signal_time",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "exit_reason",
        "bars_held",
        "net_return_on_equity",
        "mae_on_equity",
        "mfe_on_equity",
        "account_mae_proxy",
        "stop_pct",
        "stop_to_liq_ratio",
        "stop_near_liq_proxy_flag",
        "stop_beyond_liq_proxy_flag",
        "cooldown_hours_after_stop",
        "stop_hit",
        "liquidation_proxy_hit",
        "down_spike_pct",
        "atr_pct",
        "fp_max_bucket_abs_delta_pressure",
        "session_bucket",
    ]
    return out[[c for c in cols if c in out.columns]].sort_values(["entry_time", "variant_name"]).reset_index(drop=True)


def build_causal_audit(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    cols = [
        "variant_name",
        "signal_time",
        "entry_time",
        "exit_time",
        "signal_pos",
        "entry_pos",
        "exit_pos",
        "entry_delay_bars_actual",
        "entry_reason",
        "exit_reason",
        "entry_price",
        "exit_price",
        "stop_hit",
        "stop_pct",
        "stop_to_liq_ratio",
    ]
    out = trades[[c for c in cols if c in trades.columns]].copy()
    out["signal_time"] = pd.to_datetime(out["signal_time"], errors="coerce")
    out["entry_time"] = pd.to_datetime(out["entry_time"], errors="coerce")
    out["exit_time"] = pd.to_datetime(out["exit_time"], errors="coerce")
    out["expected_entry_pos"] = pd.to_numeric(out["signal_pos"], errors="coerce") + pd.to_numeric(out["entry_delay_bars_actual"], errors="coerce")
    out["entry_not_expected_delay_flag"] = pd.to_numeric(out["entry_pos"], errors="coerce") != pd.to_numeric(out["expected_entry_pos"], errors="coerce")
    out["entry_before_signal_flag"] = out["entry_time"] <= out["signal_time"]
    out["exit_before_entry_flag"] = out["exit_time"] < out["entry_time"]
    out["same_bar_exit_flag"] = pd.to_numeric(out["exit_pos"], errors="coerce") == pd.to_numeric(out["entry_pos"], errors="coerce")
    out["same_bar_protective_stop_flag"] = out["same_bar_exit_flag"] & pd.to_numeric(out.get("stop_hit", False), errors="coerce").fillna(0).astype(bool)
    # Same-bar protective stop is conservative for a long: we assume the stop is
    # hit when low crosses it after the next-open entry. Mark it separately for
    # review instead of treating it as a lookahead entry bug.
    out["same_bar_path_uncertainty_flag"] = out["same_bar_exit_flag"] & ~out["same_bar_protective_stop_flag"]
    out["context_available_time_flag"] = False  # only 1m closed event + footprint end_ts asof <= signal_time in shared builder
    out["lookahead_flag"] = out[[
        "entry_not_expected_delay_flag",
        "entry_before_signal_flag",
        "exit_before_entry_flag",
        "context_available_time_flag",
    ]].any(axis=1)
    return out

def build_cost_stress(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    variants: Sequence[RiskVariant],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for mult in parse_float_list(args.cost_mults):
        _, summary = simulate_variant_set(bars, events, variants, args, cost_mult=mult, label=f"[stress] cost {mult:g}x")
        if not summary.empty:
            summary["stress_type"] = "cost"
            summary["stress_value"] = float(mult)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_delay_stress(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    base_variants: Sequence[RiskVariant],
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for delay in parse_int_list(args.delay_bars):
        variants = [
            RiskVariant(
                variant_name=f"{v.variant_name}_delay{delay}",
                family=v.family,
                description=v.description,
                stop_pct=v.stop_pct,
                atr_mult=v.atr_mult,
                early_bars=v.early_bars,
                early_mfe_cap=v.early_mfe_cap,
                early_close_floor=v.early_close_floor,
                early_buy_share_max=v.early_buy_share_max,
                early_delta_max=v.early_delta_max,
                giveback_trigger=v.giveback_trigger,
                giveback_floor=v.giveback_floor,
                skip_atr_pct_ge=v.skip_atr_pct_ge,
                skip_down_spike_pct_ge=v.skip_down_spike_pct_ge,
                cooldown_hours_after_stop=v.cooldown_hours_after_stop,
                entry_extra_delay=int(delay),
                horizon=v.horizon,
            )
            for v in base_variants
        ]
        _, summary = simulate_variant_set(bars, events, variants, args, cost_mult=1.0, label=f"[stress] delay {delay}bar")
        if not summary.empty:
            summary["stress_type"] = "delay"
            summary["stress_value"] = int(delay)
            rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_decision_tables(summary: pd.DataFrame, cost_stress: pd.DataFrame, delay_stress: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    base_rows = summary.loc[summary["variant_name"].eq(BASELINE_VARIANT)]
    baseline = base_rows.iloc[0] if not base_rows.empty else summary.iloc[0]
    base_trades = max(1.0, finite_float(baseline.get("trades", 0), 0.0))
    base_return = finite_float(baseline.get("return_total", np.nan))
    base_pf = finite_float(baseline.get("profit_factor", np.nan))
    base_mae_min = finite_float(baseline.get("mae_min", np.nan))
    base_liq_rate = finite_float(baseline.get("liquidation_proxy_hit_rate", 0.0), 0.0)

    cost2 = pd.DataFrame()
    if not cost_stress.empty:
        c = cost_stress.copy()
        c = c.loc[np.isclose(pd.to_numeric(c.get("stress_value", np.nan), errors="coerce"), 2.0)]
        cost2 = c[["variant_name", "return_total", "profit_factor", "mean_return"]].rename(
            columns={"return_total": "fee2_return_total", "profit_factor": "fee2_profit_factor", "mean_return": "fee2_mean_return"}
        )
    delay1 = pd.DataFrame()
    if not delay_stress.empty:
        d = delay_stress.copy()
        d = d.loc[pd.to_numeric(d.get("stress_value", np.nan), errors="coerce").eq(1)]
        # Remove the suffix added by delay stress so it joins to the original variant.
        d["variant_name"] = d["variant_name"].astype(str).str.replace(r"_delay1$", "", regex=True)
        delay1 = d[["variant_name", "return_total", "profit_factor", "mean_return"]].rename(
            columns={"return_total": "delay1_return_total", "profit_factor": "delay1_profit_factor", "mean_return": "delay1_mean_return"}
        )

    table = summary.copy()
    if not cost2.empty:
        table = table.merge(cost2, on="variant_name", how="left")
    if not delay1.empty:
        table = table.merge(delay1, on="variant_name", how="left")

    decisions: list[str] = []
    reasons: list[str] = []
    for _, row in table.iterrows():
        name = str(row.get("variant_name", ""))
        if name == BASELINE_VARIANT:
            decisions.append("baseline_reference")
            reasons.append("current time48/no-stop anchor")
            continue
        trades_ok = finite_float(row.get("trades", 0), 0.0) >= base_trades * 0.75
        return_ok = (not math.isfinite(base_return)) or finite_float(row.get("return_total", np.nan)) >= base_return * 0.70
        pf = finite_float(row.get("profit_factor", np.nan))
        pf_ok = math.isfinite(pf) and pf >= max(1.20, base_pf * 0.65 if math.isfinite(base_pf) else 1.20)
        liq_rate = finite_float(row.get("liquidation_proxy_hit_rate", 1.0))
        tail_better = finite_float(row.get("mae_min", np.nan)) > base_mae_min or liq_rate < base_liq_rate
        execution_danger = finite_float(row.get("stop_beyond_liq_proxy_rate", 0.0), 0.0) > 0.0
        near_liq_warning = finite_float(row.get("stop_near_liq_proxy_rate", 0.0), 0.0) > 0.0
        fee2_ok = True
        if "fee2_profit_factor" in row.index and math.isfinite(finite_float(row.get("fee2_profit_factor", np.nan))):
            fee2_ok = finite_float(row.get("fee2_profit_factor", np.nan)) >= 1.05
        delay_ok = True
        if "delay1_profit_factor" in row.index and math.isfinite(finite_float(row.get("delay1_profit_factor", np.nan))):
            delay_ok = finite_float(row.get("delay1_profit_factor", np.nan)) >= 1.00
        if execution_danger:
            decisions.append("diagnostic_execution_danger")
            reasons.append("configured stop is at/beyond conservative liquidation proxy; do not promote without execution model")
        elif trades_ok and return_ok and pf_ok and tail_better and fee2_ok and delay_ok:
            decisions.append("research_continue_candidate")
            suffix = "; near liquidation proxy, require extra execution review" if near_liq_warning else ""
            reasons.append("risk improved while preserving enough trades/return/PF; needs deeper portfolio overlay" + suffix)
        else:
            bad = []
            if not trades_ok:
                bad.append("too_many_trades_removed")
            if not return_ok:
                bad.append("return_loss_too_large")
            if not pf_ok:
                bad.append("pf_too_weak")
            if not tail_better:
                bad.append("tail_not_improved")
            if not fee2_ok:
                bad.append("fee2_weak")
            if not delay_ok:
                bad.append("delay1_weak")
            decisions.append("rejected_or_secondary_diagnostic")
            reasons.append(";".join(bad) if bad else "did_not_beat_baseline")
    table["decision"] = decisions
    table["reason"] = reasons

    shortlist = table.loc[table["decision"].isin(["baseline_reference", "research_continue_candidate"])].copy()
    rejected = table.loc[~table["decision"].isin(["baseline_reference", "research_continue_candidate"])].copy()
    shortlist = shortlist.sort_values(["decision", "liquidation_proxy_hit_rate", "return_total"], ascending=[True, True, False]).reset_index(drop=True)
    rejected = rejected.sort_values(["variant_family", "return_total"], ascending=[True, False]).reset_index(drop=True)
    return shortlist, rejected


def trim_trades_for_output(trades: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if trades.empty:
        return trades
    if int(max_rows) <= 0 or len(trades) <= int(max_rows):
        return trades
    primary = trades.loc[trades["variant_name"].eq(BASELINE_VARIANT)]
    risky = trades.loc[pd.to_numeric(trades.get("liquidation_proxy_hit", False), errors="coerce").fillna(0).astype(bool)]
    worst = trades.sort_values("mae_on_equity", ascending=True).head(max(1000, int(max_rows) // 5))
    sampled = trades.sample(n=max(0, int(max_rows) - len(primary) - len(risky) - len(worst)), random_state=7) if len(trades) > len(primary) + len(risky) + len(worst) else pd.DataFrame()
    out = pd.concat([primary, risky, worst, sampled], ignore_index=True, sort=False)
    return out.drop_duplicates(["variant_name", "signal_time", "entry_time", "exit_time"]).head(int(max_rows))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {SCRIPT_NAME} {SCRIPT_VERSION}", flush=True)
    print("[class] MF / Exit+Filter research only; no portfolio/live strategy changes", flush=True)
    print(f"[args] symbol={args.symbol} start={args.start_date} end={args.end_date} warmup={args.warmup_start_date}", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)
    print(f"[risk] leverage={args.leverage:g} liq_proxy_pct={resolved_liquidation_proxy_pct(args):.4%} mf_exposure={args.mf_exposure:g}", flush=True)

    mf_args = mf_args_from_cli(args)
    print(f"[load] trade bars {args.symbol} {args.timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    bars = load_trade_bars(mf_args)

    print("[events] build current MF low-sweep events + footprint context", flush=True)
    events_all = prepare_events_and_context(bars, mf_args)
    events = select_a0_events(events_all, mf_args)
    print(f"[events] all={len(events_all):,} selected_A0_single={len(events):,}", flush=True)

    variants = risk_variants(args)
    print(f"[simulate] risk variants={len(variants):,}", flush=True)
    trades, summary = simulate_variant_set(bars, events, variants, args, cost_mult=1.0, label="[simulate] risk variants")

    # Stress only the relevant compact subset: baseline, stop/cooldown variants,
    # risk gates, and gate+stop+cooldown combinations.
    stress_variants = [
        v for v in variants
        if v.family in {
            "baseline",
            "wide_hard_stop",
            "wide_hard_stop_cooldown",
            "entry_risk_gate",
            "gate_stop_cooldown_combo",
        }
    ]
    print(f"[stress] compact variants={len(stress_variants):,}", flush=True)
    cost_stress = build_cost_stress(bars, events, stress_variants, args)
    delay_stress = build_delay_stress(bars, events, stress_variants, args)

    yearly = summarize_by_period(trades, "year")
    monthly = summarize_by_period(trades, "month")
    daily = summarize_by_period(trades, "day")
    tail = build_tail_risk(trades)
    mae_mfe_distribution = build_mae_mfe_distribution(trades, args)
    stop_threshold_tradeoff = build_stop_threshold_tradeoff(trades, args)
    extreme_trade_paths = build_extreme_trade_paths(trades, args)
    focus = build_focus_day_report(trades, args)
    causal = build_causal_audit(trades)
    shortlist, rejected = build_decision_tables(summary, cost_stress, delay_stress)

    # Save a compact but useful event sample. Prioritize focus-date events plus
    # selected A0 candidates.
    event_sample = events.copy()
    if not event_sample.empty:
        event_sample["signal_date"] = pd.to_datetime(event_sample["signal_time"], errors="coerce").dt.strftime("%Y-%m-%d")
        focus_dates = set(split_csv(args.focus_dates))
        focus_events = event_sample.loc[event_sample["signal_date"].isin(focus_dates)]
        if len(event_sample) > int(args.save_event_sample):
            rest_n = max(0, int(args.save_event_sample) - len(focus_events))
            rest = event_sample.drop(focus_events.index, errors="ignore").head(rest_n)
            event_sample = pd.concat([focus_events, rest], ignore_index=True, sort=False)

    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "strategy_class": "MF / Exit / Filter research",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "baseline_edge_variant": VARIANT_NAME,
        "entry_anchor": "A0_fp_abs_delta_high + single_swing + next_open",
        "baseline_exit": "time48 + no_stop",
        "round_trip_cost_pct_default": effective_round_trip_cost_pct(args),
        "fee_rate_arg": float(args.fee_rate),
        "slippage_pct_arg": float(args.slippage_pct),
        "cost_multipliers": parse_float_list(args.cost_mults),
        "delay_bars_list": parse_int_list(args.delay_bars),
        "input_rows": int(len(bars)),
        "all_event_rows": int(len(events_all)),
        "selected_event_rows": int(len(events)),
        "variant_count": int(len(variants)),
        "hard_stop_pcts": parse_float_list(args.hard_stop_pcts),
        "threshold_grid_pcts": parse_float_list(args.threshold_grid_pcts),
        "stop_selection_policy": "MAE/MFE path distribution first; liquidation proxy is tail/execution audit only, not stop source",
        "liquidation_proxy_pct": resolved_liquidation_proxy_pct(args),
        "mf_exposure": float(args.mf_exposure),
        "cooldown_hours_default": float(args.cooldown_hours),
        "causal_policy": "closed 1m signal; next-open entry; post-entry closed-bar early decisions execute next open; no live execution",
        "focus_dates": split_csv(args.focus_dates),
    }

    write_json(manifest, out_dir / "00_manifest.json", "manifest")
    write_csv(summary, out_dir / "01_variant_summary.csv", "variant_summary")
    write_csv(yearly, out_dir / "02_yearly.csv", "yearly")
    write_csv(monthly, out_dir / "03_monthly.csv", "monthly")
    write_csv(tail, out_dir / "04_tail_risk.csv", "tail_risk")
    write_csv(focus, out_dir / "05_focus_day_2025_02_03.csv", "focus_day")
    write_csv(cost_stress, out_dir / "06_cost_stress.csv", "cost_stress")
    write_csv(delay_stress, out_dir / "07_delay_stress.csv", "delay_stress")
    write_csv(shortlist, out_dir / "08_candidate_shortlist.csv", "candidate_shortlist")
    write_csv(rejected, out_dir / "09_rejected_candidates.csv", "rejected_candidates")
    write_csv(causal, out_dir / "10_causal_audit.csv", "causal_audit")
    write_csv(event_sample, out_dir / "11_event_sample.csv", "event_sample")
    write_csv(trim_trades_for_output(trades, int(args.save_trades)), out_dir / "12_trade_sample.csv", "trade_sample")
    write_csv(daily, out_dir / "13_daily_breakdown.csv", "daily_breakdown")
    write_csv(mae_mfe_distribution, out_dir / "14_mae_mfe_distribution.csv", "mae_mfe_distribution")
    write_csv(stop_threshold_tradeoff, out_dir / "15_stop_threshold_tradeoff.csv", "stop_threshold_tradeoff")
    write_csv(extreme_trade_paths, out_dir / "16_extreme_trade_paths.csv", "extreme_trade_paths")

    if not bool(args.skip_review_pack):
        finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)

    print("[done] research report complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
