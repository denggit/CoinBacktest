#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Low Sweep V1 A0 + footprint formal MF backtest.

This is a formal backtest wrapper, not a live-trading strategy.  It freezes the
current best research candidate into a small, auditable MF backtest:

    A0_fp_abs_delta_high + single_swing + next_open + time48

The broad research script intentionally explores many combinations.  This file
keeps the backtest surface small and writes portfolio-ready validation reports:
trades, summary, yearly/monthly, cost stress, delay stress, path timing, tail
risk, and overlap diagnostics.

Causal notes:
- signal uses closed 1m trade bars only;
- confirmed swing lows are inherited from the no-leakage low-sweep pipeline;
- footprint context is attached only from closed footprint/range context whose
  available timestamp is <= signal_time;
- no live execution or order placement is performed here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.low_sweep_a_upgrade_research import (  # noqa: E402
    UpgradeVariant,
    _split_csv_names,
    attach_footprint_context,
    attach_micro_trade_context,
    attach_range_context,
    attach_support_zone_metrics,
    build_candidate_layer_masks,
    build_market_cache,
    build_support_mask,
    parse_args as _upgrade_parse_args,
    parse_stop_specs,
    prepare_studied_events,
    simulate_upgrade_trade,
    summarize_by_period,
    summarize_trades,
    write_csv,
)
from research.low_sweep_panic_reversal_strategy_backtest_probe import load_trade_bars  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

SCRIPT_NAME = "low_sweep_V1_a0_footprint_backtest"
REPORT_STRATEGY_NAME = "Low_Sweep_V1_A0_Footprint_MF"
PRIMARY_VARIANT_NAME = "V1_main_A0_fp_abs_delta_high_single_next_open_time48_no_stop"
DEFAULT_OUT_DIR = "data/reports/backtest/mf/low_sweep/V1_a0_footprint_backtest"


@dataclass(frozen=True)
class FormalVariantSpec:
    """A small named variant set for the formal report."""

    variant_name: str
    candidate_layer: str
    support_mode: str
    entry_mode: str
    exit_mode: str
    stop_name: str = "no_stop"
    role: str = "comparison"


def _parse_float_list(raw: str) -> list[float]:
    out: list[float] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        val = float(part)
        if math.isfinite(val):
            out.append(val)
    return sorted(set(out))


def _parse_int_list(raw: str) -> list[int]:
    out: list[int] = []
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(out))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse formal-backtest flags plus the existing low-sweep data args.

    Formal-only flags are parsed first and removed; everything else is delegated
    to the existing low-sweep parser so symbols, dates, fees, slippage, and data
    loader knobs stay compatible with the current project.
    """

    formal = argparse.ArgumentParser(add_help=False)
    formal.add_argument("--formal-cost-mults", default="1.0,1.5,2.0,3.0")
    formal.add_argument("--formal-delay-bars", default="0,1,2,3")
    formal.add_argument("--formal-save-trades", type=int, default=100000)
    formal.add_argument("--formal-save-events", type=int, default=5000)
    formal.add_argument("--initial-capital", type=float, default=1000.0, help="Initial capital used by the printed full report.")
    formal.add_argument("--skip-full-report", action="store_true", help="Do not call src.utils.report.print_full_report at the end.")
    formal.add_argument("--include-range-context", action="store_true", help="Attach range context in addition to footprint. Default off for speed because V1 only requires footprint.")
    formal.add_argument("--include-micro-context", action="store_true", help="Attach 5s/10s micro context for diagnostics. Default off until micro coverage is fixed.")
    known, rest = formal.parse_known_args(argv)

    context_sources = "trade_bar,footprint"
    if bool(known.include_range_context):
        context_sources = "trade_bar,range_bar,footprint"
    micro_timeframes = "5s,10s" if bool(known.include_micro_context) else ""

    defaults = [
        "--out-dir",
        DEFAULT_OUT_DIR,
        "--candidate-layers",
        "A0_fp_abs_delta_high,A0_spike_ge_0100,A1_current,A1_fp_abs_delta_high,A1_only_fp_abs_delta_high",
        "--support-modes",
        "single_swing,equal2_020",
        "--entry-modes",
        "next_open,next_open_delay1,next_open_delay2,next_open_delay3",
        "--exit-modes",
        "time24,time36,time48,target_signal_open_or_time48,swing_trail_after6_time72,swing_trail_after12_time72,swing_trail_after18_time96",
        "--upgrade-stop-specs",
        "no_stop,fixed_0250,atr_6x",
        "--context-sources",
        context_sources,
        "--micro-timeframes",
        micro_timeframes,
        "--micro-load-mode",
        "local",
        "--save-trades",
        str(int(known.formal_save_trades)),
        "--save-events",
        str(int(known.formal_save_events)),
    ]
    args = _upgrade_parse_args(defaults + list(rest))
    args.formal_cost_mults = ",".join(str(x) for x in _parse_float_list(known.formal_cost_mults))
    args.formal_delay_bars = ",".join(str(x) for x in _parse_int_list(known.formal_delay_bars))
    args.initial_capital = float(known.initial_capital)
    args.skip_full_report = bool(known.skip_full_report)
    args.include_range_context = bool(known.include_range_context)
    args.include_micro_context = bool(known.include_micro_context)
    return args


def _stop_spec_by_name(args: argparse.Namespace) -> dict[str, object]:
    return {s.name: s for s in parse_stop_specs(args.upgrade_stop_specs)}


def formal_variant_specs() -> list[FormalVariantSpec]:
    """Frozen V1 report set.

    The primary row is the MF candidate.  Other rows are included only to make
    trade-off review easier; they are not promoted to live trading.
    """

    return [
        FormalVariantSpec(
            variant_name=PRIMARY_VARIANT_NAME,
            candidate_layer="A0_fp_abs_delta_high",
            support_mode="single_swing",
            entry_mode="next_open",
            exit_mode="time48",
            stop_name="no_stop",
            role="primary_mf_candidate",
        ),
        FormalVariantSpec(
            variant_name="V1_main_A0_fp_abs_delta_high_single_next_open_time48_fixed_0250",
            candidate_layer="A0_fp_abs_delta_high",
            support_mode="single_swing",
            entry_mode="next_open",
            exit_mode="time48",
            stop_name="fixed_0250",
            role="primary_stop_probe",
        ),
        FormalVariantSpec(
            variant_name="V1_main_A0_fp_abs_delta_high_single_next_open_time48_atr_6x",
            candidate_layer="A0_fp_abs_delta_high",
            support_mode="single_swing",
            entry_mode="next_open",
            exit_mode="time48",
            stop_name="atr_6x",
            role="primary_stop_probe",
        ),
        FormalVariantSpec(
            variant_name="V1_quality_A0_fp_abs_delta_high_equal2_next_open_time48_no_stop",
            candidate_layer="A0_fp_abs_delta_high",
            support_mode="equal2_020",
            entry_mode="next_open",
            exit_mode="time48",
            stop_name="no_stop",
            role="low_dd_quality_comparison",
        ),
        FormalVariantSpec(
            variant_name="V1_benchmark_A0_no_footprint_single_next_open_time48_no_stop",
            candidate_layer="A0_spike_ge_0100",
            support_mode="single_swing",
            entry_mode="next_open",
            exit_mode="time48",
            stop_name="no_stop",
            role="no_footprint_benchmark",
        ),
        FormalVariantSpec(
            variant_name="V1_benchmark_A1_current_single_next_open_time48_no_stop",
            candidate_layer="A1_current",
            support_mode="single_swing",
            entry_mode="next_open",
            exit_mode="time48",
            stop_name="no_stop",
            role="old_A1_benchmark",
        ),
        FormalVariantSpec(
            variant_name="V1_incremental_A1_only_fp_abs_delta_high_single_swing_trail18_no_stop",
            candidate_layer="A1_only_fp_abs_delta_high",
            support_mode="single_swing",
            entry_mode="next_open",
            exit_mode="swing_trail_after18_time96",
            stop_name="no_stop",
            role="A1_non_A0_incremental_probe",
        ),
    ]


def _make_variant(spec: FormalVariantSpec, stops: dict[str, object]) -> UpgradeVariant:
    if spec.stop_name not in stops:
        raise ValueError(f"Missing stop spec {spec.stop_name!r}; available={sorted(stops)}")
    return UpgradeVariant(
        variant_name=spec.variant_name,
        candidate_layer=spec.candidate_layer,
        support_mode=spec.support_mode,
        entry_mode=spec.entry_mode,
        exit_mode=spec.exit_mode,
        stop_spec=stops[spec.stop_name],
    )


def prepare_events_and_context(bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    print("[events] building V1 candidate events", flush=True)
    events = prepare_studied_events(bars, args)
    events = attach_support_zone_metrics(events, bars, args)
    sources = set(_split_csv_names(getattr(args, "context_sources", "trade_bar,footprint")))
    if "range_bar" in sources:
        events = attach_range_context(events, args)
    if "footprint" in sources:
        events = attach_footprint_context(events, args)
    if _split_csv_names(getattr(args, "micro_timeframes", "")):
        events = attach_micro_trade_context(events, args)
    print(f"[events] ready rows={len(events):,} columns={len(events.columns):,}", flush=True)
    return events


def _attach_event_columns(trades: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "signal_time" not in trades.columns or "signal_time" not in events.columns:
        return trades
    keep_prefixes = ("fp_", "range_", "micro_")
    cols = [
        "signal_time",
        "event_name",
        "variant",
        "swing_level",
        "swing_age",
        "down_spike_pct",
        "close_pos_in_bar",
        "large_trade_share",
        "cluster_touch_count_020",
        "cluster_touch_count_030",
        "session_bucket",
    ]
    cols.extend(c for c in events.columns if c.startswith(keep_prefixes))
    cols = [c for c in dict.fromkeys(cols) if c in events.columns and (c not in trades.columns or c == "signal_time")]
    if len(cols) <= 1:
        return trades
    addon = events[cols].drop_duplicates("signal_time")
    return trades.merge(addon, on="signal_time", how="left", suffixes=("", "_event"))


def simulate_variant_set(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    variants: Sequence[UpgradeVariant],
    args: argparse.Namespace,
    *,
    cost_mult: float = 1.0,
    label: str = "[bt] variants",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a small fixed set without per-job log spam."""

    print("[cache] building reusable market arrays", flush=True)
    market = build_market_cache(bars, args)
    print("[cache] precomputing candidate/support masks", flush=True)
    layer_masks = {k: v.fillna(False).astype(bool) for k, v in build_candidate_layer_masks(events, args).items()}
    support_masks = {mode: build_support_mask(events, mode, args).fillna(False).astype(bool) for mode in sorted({v.support_mode for v in variants})}
    signal_pos_map = pd.Series(np.arange(len(bars), dtype=int), index=bars.index)

    total_events = 0
    prepared: list[tuple[UpgradeVariant, pd.DataFrame]] = []
    for variant in variants:
        layer_mask = layer_masks.get(variant.candidate_layer, pd.Series(False, index=events.index))
        support_mask = support_masks.get(variant.support_mode, pd.Series(False, index=events.index))
        part = events.loc[layer_mask & support_mask].copy().sort_values("signal_time")
        prepared.append((variant, part))
        total_events += int(len(part))

    progress = ProgressReporter(label=label, total=max(1, total_events), every=250, enabled=not bool(args.no_progress))
    done = 0
    trade_parts: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    roles = {v.variant_name: getattr(v, "formal_role", "") for v in variants}

    for variant, part in prepared:
        rows: list[dict[str, object]] = []
        counters = {
            "candidate_events": int(len(part)),
            "input_events": int(len(part)),
            "invalid_events": 0,
            "skipped_invalid": 0,
            "skipped_overlap": 0,
            "valid_trades": 0,
        }
        last_exit_pos = -1
        for _, event in part.iterrows():
            ts = pd.Timestamp(event.get("signal_time"))
            if ts not in signal_pos_map.index:
                counters["invalid_events"] += 1
                counters["skipped_invalid"] += 1
                done += 1
                progress.update(done)
                continue
            signal_pos = int(signal_pos_map.loc[ts])
            # Formal single-strategy backtest: only one active position at a time.
            # Research event tables may contain overlapping candidate events; those
            # are skipped until the current trade has exited.
            if signal_pos <= last_exit_pos:
                counters["skipped_overlap"] += 1
                done += 1
                progress.update(done)
                continue
            rec = simulate_upgrade_trade(bars, event, signal_pos, variant, args, cost_mult=float(cost_mult), market=market)
            if rec.get("valid"):
                rows.append(dict(rec))
                counters["valid_trades"] += 1
                last_exit_pos = int(rec.get("exit_pos", signal_pos))
            else:
                counters["invalid_events"] += 1
                counters["skipped_invalid"] += 1
            done += 1
            progress.update(done)
        trades = pd.DataFrame(rows)
        if not trades.empty:
            trades["formal_role"] = roles.get(variant.variant_name, "")
            trade_parts.append(trades)
        rec = summarize_trades(trades, args, counters)
        rec.update(
            {
                "variant_name": variant.variant_name,
                "candidate_layer": variant.candidate_layer,
                "support_mode": variant.support_mode,
                "entry_mode": variant.entry_mode,
                "exit_mode": variant.exit_mode,
                "stop_name": variant.stop_spec.name,
                "stop_mode": variant.stop_spec.mode,
                "cost_mult": float(cost_mult),
                "formal_role": roles.get(variant.variant_name, ""),
            }
        )
        summary_rows.append(rec)
    progress.close()

    all_trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame()
    all_trades = _attach_event_columns(all_trades, events)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["formal_role", "return_total", "profit_factor"], ascending=[True, False, False]).reset_index(drop=True)
    return all_trades, summary


def _with_roles(variants: list[UpgradeVariant], specs: Sequence[FormalVariantSpec]) -> list[UpgradeVariant]:
    roles = {s.variant_name: s.role for s in specs}
    out = []
    for v in variants:
        object.__setattr__(v, "formal_role", roles.get(v.variant_name, ""))  # frozen dataclass: annotate external metadata for this process only.
        out.append(v)
    return out


def build_cost_stress(bars: pd.DataFrame, events: pd.DataFrame, args: argparse.Namespace, stop_spec: object) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    print("[stress] cost multipliers", flush=True)
    for mult in _parse_float_list(args.formal_cost_mults):
        variant = UpgradeVariant(
            variant_name=f"V1_main_A0_fp_abs_delta_high_cost_{mult:g}x",
            candidate_layer="A0_fp_abs_delta_high",
            support_mode="single_swing",
            entry_mode="next_open",
            exit_mode="time48",
            stop_spec=stop_spec,
        )
        _, summary = simulate_variant_set(bars, events, [variant], args, cost_mult=float(mult), label=f"[stress] cost {mult:g}x")
        summary["stress_type"] = "cost"
        summary["stress_value"] = float(mult)
        rows.append(summary)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_delay_stress(bars: pd.DataFrame, events: pd.DataFrame, args: argparse.Namespace, stop_spec: object) -> pd.DataFrame:
    print("[stress] entry delay bars", flush=True)
    variants: list[UpgradeVariant] = []
    for delay in _parse_int_list(args.formal_delay_bars):
        entry = "next_open" if delay == 0 else f"next_open_delay{delay}"
        variants.append(
            UpgradeVariant(
                variant_name=f"V1_main_A0_fp_abs_delta_high_delay_{delay}bar",
                candidate_layer="A0_fp_abs_delta_high",
                support_mode="single_swing",
                entry_mode=entry,
                exit_mode="time48",
                stop_spec=stop_spec,
            )
        )
    _, summary = simulate_variant_set(bars, events, variants, args, cost_mult=1.0, label="[stress] delay")
    if not summary.empty:
        summary["stress_type"] = "delay"
        summary["stress_value"] = summary["entry_mode"].map(lambda x: 0 if x == "next_open" else int(str(x).replace("next_open_delay", "")))
    return summary


def build_path_timing(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "variant_name",
        "formal_role",
        "trades",
        "return_total",
        "profit_factor",
        "win_rate",
        "max_drawdown",
        "mae_median",
        "mae_p05",
        "mfe_median",
        "mae_time_median_bars",
        "mfe_time_median_bars",
        "first_positive_high_median_bars",
        "mae_before_mfe_rate",
        "avg_bars_held",
    ]
    return summary[[c for c in cols if c in summary.columns]].copy() if not summary.empty else pd.DataFrame()


def build_tail_risk(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for name, grp in trades.groupby("variant_name", dropna=False):
        x = pd.to_numeric(grp.get("net_return_on_equity"), errors="coerce").dropna()
        mae = pd.to_numeric(grp.get("mae_on_equity"), errors="coerce").dropna()
        mfe = pd.to_numeric(grp.get("mfe_on_equity"), errors="coerce").dropna()
        worst = grp.sort_values("net_return_on_equity", ascending=True).head(10).copy()
        rows.append(
            {
                "variant_name": name,
                "trades": int(len(grp)),
                "ret_p01": float(x.quantile(0.01)) if not x.empty else np.nan,
                "ret_p05": float(x.quantile(0.05)) if not x.empty else np.nan,
                "ret_median": float(x.median()) if not x.empty else np.nan,
                "ret_p95": float(x.quantile(0.95)) if not x.empty else np.nan,
                "mae_p01": float(mae.quantile(0.01)) if not mae.empty else np.nan,
                "mae_p05": float(mae.quantile(0.05)) if not mae.empty else np.nan,
                "mae_median": float(mae.median()) if not mae.empty else np.nan,
                "mfe_median": float(mfe.median()) if not mfe.empty else np.nan,
                "worst_trade_time": worst["entry_time"].iloc[0] if not worst.empty and "entry_time" in worst else pd.NaT,
                "worst_trade_return": float(worst["net_return_on_equity"].iloc[0]) if not worst.empty else np.nan,
                "worst_10_sum": float(pd.to_numeric(worst.get("net_return_on_equity"), errors="coerce").sum()) if not worst.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("worst_trade_return").reset_index(drop=True)


def build_overlap(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty or "signal_time" not in trades.columns:
        return pd.DataFrame()
    primary = PRIMARY_VARIANT_NAME
    primary_times = set(pd.to_datetime(trades.loc[trades["variant_name"].eq(primary), "signal_time"]).astype(str))
    rows: list[dict[str, object]] = []
    for name, grp in trades.groupby("variant_name", dropna=False):
        times = set(pd.to_datetime(grp["signal_time"]).astype(str))
        overlap = times & primary_times
        unique_mask = ~pd.to_datetime(grp["signal_time"]).astype(str).isin(primary_times)
        unique = grp.loc[unique_mask]
        rows.append(
            {
                "variant_name": name,
                "trades": int(len(grp)),
                "overlap_with_primary": int(len(overlap)),
                "overlap_ratio": float(len(overlap) / len(times)) if times else np.nan,
                "unique_trades": int(len(unique)),
                "unique_return_sum": float(pd.to_numeric(unique.get("net_return_on_equity"), errors="coerce").sum()) if not unique.empty else 0.0,
                "unique_win_rate": float((pd.to_numeric(unique.get("net_return_on_equity"), errors="coerce") > 0).mean()) if not unique.empty else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["overlap_ratio", "unique_trades"], ascending=[False, False]).reset_index(drop=True)


def build_print_report_trades(trades: pd.DataFrame, args: argparse.Namespace) -> tuple[list[dict[str, object]], float]:
    """Convert the primary variant to the shared print_full_report trade format.

    The low-sweep simulator reports net return on full equity. The shared
    report helper expects currency PnL and running capital, so we compound the
    primary variant sequentially from ``args.initial_capital``. This only affects
    the printed deep report; the CSV summaries are still produced independently.
    """

    initial_capital = float(getattr(args, "initial_capital", 1000.0))
    capital = initial_capital
    if trades.empty or "variant_name" not in trades.columns:
        return [], capital

    primary = trades.loc[trades["variant_name"].eq(PRIMARY_VARIANT_NAME)].copy()
    if primary.empty:
        return [], capital

    primary["entry_time"] = pd.to_datetime(primary["entry_time"], errors="coerce")
    primary["exit_time"] = pd.to_datetime(primary["exit_time"], errors="coerce")
    primary = primary.dropna(subset=["entry_time", "exit_time"]).sort_values("entry_time")

    fee_rate = float(getattr(args, "entry_fee_rate", 0.0) or 0.0) + float(getattr(args, "exit_fee_rate", 0.0) or 0.0)
    out: list[dict[str, object]] = []
    for _, row in primary.iterrows():
        cap_before = capital
        ret = float(pd.to_numeric(row.get("net_return_on_equity"), errors="coerce"))
        if not math.isfinite(ret):
            continue
        pnl = cap_before * ret
        capital = cap_before + pnl
        fee = cap_before * fee_rate * float(row.get("cost_mult", 1.0) or 1.0)
        out.append(
            {
                "entry_time": pd.Timestamp(row["entry_time"]),
                "exit_time": pd.Timestamp(row["exit_time"]),
                "type": "LONG",
                "entry": float(row.get("entry_price", np.nan)),
                "exit": float(row.get("exit_price", np.nan)),
                "pnl": float(pnl),
                "fee": float(fee),
                "capital": float(capital),
                "mfe_r": float(row.get("mfe_on_equity", np.nan)),
                "mae_r": float(row.get("mae_on_equity", np.nan)),
                "exit_reason": row.get("exit_reason", ""),
            }
        )
    return out, float(capital)


def print_primary_full_report(trades: pd.DataFrame, bars: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> None:
    """Print and save the shared deep report for the primary MF candidate."""

    if bool(getattr(args, "skip_full_report", False)):
        print("[report] skip print_full_report (--skip-full-report)", flush=True)
        return

    report_trades, final_capital = build_print_report_trades(trades, args)
    if not report_trades:
        print("[report] primary variant has no trades; skip print_full_report", flush=True)
        return

    report_bars = bars.copy()
    if isinstance(report_bars.index, pd.DatetimeIndex) and len(report_bars.index):
        start = pd.Timestamp(args.start_date)
        end = pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        report_bars = report_bars.loc[(report_bars.index >= start) & (report_bars.index <= end)]
    total_days = max((report_bars.index[-1] - report_bars.index[0]).total_seconds() / 86400.0, 1e-9) if len(report_bars.index) else 1e-9

    print("[report] print_full_report primary MF candidate", flush=True)
    print_full_report(
        trade_history=report_trades,
        df=report_bars,
        initial_capital=float(getattr(args, "initial_capital", 1000.0)),
        capital=final_capital,
        strategy_name=REPORT_STRATEGY_NAME,
        total_days=total_days,
        ai_enabled=False,
        symbol=args.symbol,
        report_dir=out_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {SCRIPT_NAME}", flush=True)
    print(f"[class] MF backtest only; no live execution", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)

    print(f"[load] trade bars {args.symbol} {args.timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    bars = load_trade_bars(args)
    events = prepare_events_and_context(bars, args)

    stops = _stop_spec_by_name(args)
    specs = formal_variant_specs()
    variants = _with_roles([_make_variant(spec, stops) for spec in specs], specs)

    print(f"[simulate] formal variants={len(variants):,}", flush=True)
    trades, summary = simulate_variant_set(bars, events, variants, args, cost_mult=1.0, label="[bt] formal variants")
    yearly = summarize_by_period(trades, "year")
    monthly = summarize_by_period(trades, "month")
    cost_stress = build_cost_stress(bars, events, args, stops["no_stop"])
    delay_stress = build_delay_stress(bars, events, args, stops["no_stop"])
    path_timing = build_path_timing(summary)
    tail_risk = build_tail_risk(trades)
    overlap = build_overlap(trades)

    event_cols = [
        "signal_time",
        "event_name",
        "variant",
        "swing_level",
        "swing_age",
        "down_spike_pct",
        "close_pos_in_bar",
        "large_trade_share",
        "cluster_touch_count_020",
        "cluster_touch_count_030",
        "session_bucket",
    ]
    event_cols.extend(c for c in events.columns if c.startswith("fp_"))
    events_sample = events[[c for c in dict.fromkeys(event_cols) if c in events.columns]].head(int(args.save_events)).copy()
    trades_out = trades.head(int(args.save_trades)).copy() if int(args.save_trades) > 0 else pd.DataFrame()

    write_csv(events_sample, out_dir / "01_events_sample.csv", "events_sample")
    write_csv(trades_out, out_dir / "02_trades.csv", "trades")
    write_csv(summary, out_dir / "03_summary.csv", "summary")
    write_csv(yearly, out_dir / "04_yearly.csv", "yearly")
    write_csv(monthly, out_dir / "05_monthly.csv", "monthly")
    write_csv(cost_stress, out_dir / "06_cost_stress.csv", "cost_stress")
    write_csv(delay_stress, out_dir / "07_delay_stress.csv", "delay_stress")
    write_csv(path_timing, out_dir / "08_path_timing.csv", "path_timing")
    write_csv(tail_risk, out_dir / "09_tail_risk.csv", "tail_risk")
    write_csv(overlap, out_dir / "10_overlap.csv", "overlap")

    meta = {
        "script": SCRIPT_NAME,
        "class": "MF",
        "live_trading": False,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "primary_variant": PRIMARY_VARIANT_NAME,
        "report_strategy_name": REPORT_STRATEGY_NAME,
        "initial_capital": float(getattr(args, "initial_capital", 1000.0)),
        "print_full_report": not bool(getattr(args, "skip_full_report", False)),
        "context_sources": _split_csv_names(getattr(args, "context_sources", "")),
        "micro_timeframes": _split_csv_names(getattr(args, "micro_timeframes", "")),
        "formal_cost_mults": _parse_float_list(args.formal_cost_mults),
        "formal_delay_bars": _parse_int_list(args.formal_delay_bars),
        "causal_guards": [
            "1m signal bar closes before next_open entry",
            "A0 spike and rolling large-share thresholds are inherited from the no-leakage event pipeline",
            "footprint context is attached only from closed context bars available no later than signal_time",
            "stress tests rerun the same candidate events without changing signal definitions",
            "this script is backtest-only and sends no orders",
            "formal variants enforce one active position at a time by skipping overlapping events",
        ],
        "outputs": [
            "01_events_sample.csv",
            "02_trades.csv",
            "03_summary.csv",
            "04_yearly.csv",
            "05_monthly.csv",
            "06_cost_stress.csv",
            "07_delay_stress.csv",
            "08_path_timing.csv",
            "09_tail_risk.csv",
            "10_overlap.csv",
            "Low_Sweep_V1_A0_Footprint_MF_<start>_<end>_False.txt",
            "Low_Sweep_V1_A0_Footprint_MF_<start>_<end>_False.csv",
        ],
    }
    print(f"[write] meta -> {out_dir / '11_meta.json'}", flush=True)
    (out_dir / "11_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print_primary_full_report(trades, bars, args, out_dir)
    print("[done] low_sweep V1 MF formal backtest complete", flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
