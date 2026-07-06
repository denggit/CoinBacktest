#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A-only formal research validation backtest for low-sweep panic reversal.

This script is the first A-specialized validation step after the broader
``low_sweep_panic_reversal_strategy_backtest_probe``.  It intentionally remains
under ``research/`` because the edge is not live-ready yet: tail risk, stop
selection, regime behavior, sizing, and portfolio interaction still need to be
validated before anything belongs in ``backtest/mf`` or AetherEdge.

Scope:
- candidate: A_spike_close_large_share only by default;
- entry: full entry plus one finite scale-in comparison by default;
- exit: time exits around the event-study optimum, 36/48/60 bars by default;
- risk control: no_stop upper bound, wider fixed stops, and ATR stops;
- execution: one position at a time; overlapping signals are skipped;
- reports: trades/equity/DD/monthly/weekly/tail/regime/sizing/stress outputs.

Leakage guards inherited from the upstream probes:
- A uses no full-sample qcut filters;
- rolling thresholds use shifted historical windows upstream;
- signals are generated on closed bars and entries use future opens;
- stops use levels known on the signal/entry path only;
- scale-in is capped and post-entry only; no unlimited martingale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.focused_low_sweep_reversal_event_lab import _parse_number_list  # noqa: E402
from research.low_sweep_panic_reversal_strategy_backtest_probe import (  # noqa: E402
    _profit_factor,
    _split_csv_names,
    build_edge_registry,
    build_equity_curve,
    build_variants,
    load_trade_bars,
    parse_args as _base_parse_args,
    parse_stop_specs,
    prepare_studied_events,
    run_variant_jobs,
    summarize_by_period,
    write_csv,
)
from src.research_common.progress import ProgressReporter  # noqa: E402

SCRIPT_NAME = "low_sweep_a_strategy_validation_backtest"
DEFAULT_OUT_DIR = "data/reports/research/low_sweep_a_strategy_validation_backtest_tradebar_1m"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse using the broader probe parser but override defaults for A validation.

    We prepend A-specific defaults and let user-supplied argv win by appending it
    after those defaults.  This keeps all upstream low-sweep/event/fee options
    available while making this script a focused A-only validation entry point.
    """

    defaults = [
        "--out-dir",
        DEFAULT_OUT_DIR,
        "--candidate-names",
        "A_spike_close_large_share",
        "--entry-schemes",
        "full_entry,scale_50_25_25_dd04_dd08",
        "--exit-horizons",
        "36,48,60",
        "--stop-specs",
        "no_stop,fixed_0200,fixed_0250,fixed_0300,atr_5x,atr_6x,atr_7x",
        "--delay-bars-list",
        "1,2,3",
        "--cost-multipliers",
        "1.0,1.5,2.0",
        "--save-trade-sample",
        "200000",
    ]
    return _base_parse_args(defaults + list(argv or []))


def _safe_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _equity_stats_from_returns(returns: pd.Series, starting_equity: float = 1.0) -> dict[str, float]:
    x = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if x.empty:
        return {
            "trades": 0,
            "return_total": np.nan,
            "profit_factor": np.nan,
            "win_rate": np.nan,
            "max_drawdown": np.nan,
            "worst_trade": np.nan,
            "best_trade": np.nan,
        }
    equity = float(starting_equity) * (1.0 + x).cumprod()
    dd = equity / equity.cummax() - 1.0
    return {
        "trades": int(len(x)),
        "return_total": float(equity.iloc[-1] / float(starting_equity) - 1.0),
        "profit_factor": _profit_factor(x),
        "win_rate": float((x > 0).mean()),
        "max_drawdown": float(dd.min()),
        "worst_trade": float(x.min()),
        "best_trade": float(x.max()),
    }


def build_weekly_summary(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    tmp = trades.copy()
    tmp["exit_time"] = pd.to_datetime(tmp["exit_time"])
    tmp["week"] = tmp["exit_time"].dt.to_period("W-SUN").astype(str)
    return summarize_by_period(tmp, args, "week")


def build_drawdown_curve(equity_curve: pd.DataFrame) -> pd.DataFrame:
    if equity_curve.empty:
        return pd.DataFrame()
    cols = ["variant_name", "trade_no", "exit_time", "equity", "drawdown"]
    return equity_curve[[c for c in cols if c in equity_curve.columns]].copy()


def build_trade_distribution(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    q_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    for variant_name, grp in trades.groupby("variant_name", dropna=False):
        ret = pd.to_numeric(grp["net_return_on_equity"], errors="coerce").dropna()
        mae = pd.to_numeric(grp.get("mae_on_equity", pd.Series(dtype=float)), errors="coerce").dropna()
        mfe = pd.to_numeric(grp.get("mfe_on_equity", pd.Series(dtype=float)), errors="coerce").dropna()
        row: dict[str, object] = {
            "variant_name": variant_name,
            "trades": int(len(ret)),
            "mean_return": float(ret.mean()) if not ret.empty else np.nan,
            "std_return": float(ret.std(ddof=0)) if len(ret) > 1 else np.nan,
            "skew_return": float(ret.skew()) if len(ret) > 2 else np.nan,
            "win_rate": float((ret > 0).mean()) if not ret.empty else np.nan,
            "profit_factor": _profit_factor(ret),
            "avg_mae": float(mae.mean()) if not mae.empty else np.nan,
            "median_mae": float(mae.median()) if not mae.empty else np.nan,
            "avg_mfe": float(mfe.mean()) if not mfe.empty else np.nan,
            "median_mfe": float(mfe.median()) if not mfe.empty else np.nan,
            "mfe_gt_abs_mae_rate": float((pd.to_numeric(grp.get("mfe_on_equity"), errors="coerce") > -pd.to_numeric(grp.get("mae_on_equity"), errors="coerce")).mean()) if {"mfe_on_equity", "mae_on_equity"}.issubset(grp.columns) else np.nan,
        }
        for q in q_levels:
            row[f"ret_q{int(q * 100):02d}"] = float(ret.quantile(q)) if not ret.empty else np.nan
            row[f"mae_q{int(q * 100):02d}"] = float(mae.quantile(q)) if not mae.empty else np.nan
            row[f"mfe_q{int(q * 100):02d}"] = float(mfe.quantile(q)) if not mfe.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["profit_factor", "mean_return"], ascending=[False, False]).reset_index(drop=True)


def build_tail_loss_audit(trades: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    keep_cols = [
        "variant_name",
        "entry_time",
        "exit_time",
        "exit_reason",
        "net_return_on_equity",
        "mae_on_equity",
        "mfe_on_equity",
        "bars_held",
        "stop_name",
        "stop_hit",
        "avg_entry_price",
        "exit_price",
        "signal_close",
        "signal_low",
        "atr_pct",
        "down_spike_pct",
        "large_trade_share",
        "session_bucket",
    ]
    rows = []
    for variant_name, grp in trades.groupby("variant_name", dropna=False):
        g = grp.copy()
        g["net_return_on_equity"] = pd.to_numeric(g["net_return_on_equity"], errors="coerce")
        worst = g.sort_values("net_return_on_equity", ascending=True).head(int(top_n)).copy()
        worst["tail_rank"] = np.arange(1, len(worst) + 1)
        rows.append(worst[[c for c in ["tail_rank"] + keep_cols if c in worst.columns]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_stop_reason_audit(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for keys, grp in trades.groupby(["variant_name", "stop_name", "exit_reason"], dropna=False):
        variant_name, stop_name, exit_reason = keys
        ret = pd.to_numeric(grp["net_return_on_equity"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "variant_name": variant_name,
                "stop_name": stop_name,
                "exit_reason": exit_reason,
                "trades": int(len(grp)),
                "return_sum_arithmetic": float(ret.sum()),
                "mean_return": float(ret.mean()) if not ret.empty else np.nan,
                "median_return": float(ret.median()) if not ret.empty else np.nan,
                "win_rate": float((ret > 0).mean()) if not ret.empty else np.nan,
                "profit_factor": _profit_factor(ret),
                "worst_trade": float(ret.min()) if not ret.empty else np.nan,
                "best_trade": float(ret.max()) if not ret.empty else np.nan,
                "avg_mae": float(pd.to_numeric(grp.get("mae_on_equity"), errors="coerce").mean()) if "mae_on_equity" in grp.columns else np.nan,
                "avg_mfe": float(pd.to_numeric(grp.get("mfe_on_equity"), errors="coerce").mean()) if "mfe_on_equity" in grp.columns else np.nan,
                "stop_hit_rate": float(pd.to_numeric(grp.get("stop_hit"), errors="coerce").mean()) if "stop_hit" in grp.columns else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["variant_name", "trades"], ascending=[True, False]).reset_index(drop=True)


def _build_bar_regimes(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    frame = bars.sort_index().copy()
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    if timeframe.endswith("m"):
        minutes = int(timeframe[:-1])
    elif timeframe.endswith("H"):
        minutes = int(timeframe[:-1]) * 60
    elif timeframe.endswith("D"):
        minutes = int(timeframe[:-1]) * 1440
    else:
        minutes = 1
    bars_4h = max(1, int(round(240 / minutes)))
    bars_1d = max(1, int(round(1440 / minutes)))
    bars_7d = max(1, int(round(10080 / minutes)))

    out = pd.DataFrame(index=frame.index)
    out["regime_close"] = close
    out["ema_4h"] = close.ewm(span=bars_4h, adjust=False, min_periods=max(3, bars_4h // 4)).mean()
    out["ema_1d"] = close.ewm(span=bars_1d, adjust=False, min_periods=max(3, bars_1d // 4)).mean()
    out["ret_24h"] = close / close.shift(bars_1d) - 1.0
    out["ret_7d"] = close / close.shift(bars_7d) - 1.0
    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    out["atr_pct_1h"] = tr.rolling(max(3, int(round(60 / minutes))), min_periods=3).mean() / close
    out["trend_4h"] = np.where(out["regime_close"] >= out["ema_4h"], "above_ema_4h", "below_ema_4h")
    out["trend_1d"] = np.where(out["regime_close"] >= out["ema_1d"], "above_ema_1d", "below_ema_1d")
    return out.reset_index(names="signal_time")


def _bucket_ret_24h(x: object) -> str:
    v = _safe_float(x)
    if not np.isfinite(v):
        return "NA"
    if v <= -0.03:
        return "ret24_le_-3pct"
    if v <= -0.01:
        return "ret24_-3_to_-1pct"
    if v < 0.01:
        return "ret24_-1_to_1pct"
    if v < 0.03:
        return "ret24_1_to_3pct"
    return "ret24_ge_3pct"


def _bucket_ret_7d(x: object) -> str:
    v = _safe_float(x)
    if not np.isfinite(v):
        return "NA"
    if v <= -0.10:
        return "ret7d_le_-10pct"
    if v <= -0.03:
        return "ret7d_-10_to_-3pct"
    if v < 0.03:
        return "ret7d_-3_to_3pct"
    if v < 0.10:
        return "ret7d_3_to_10pct"
    return "ret7d_ge_10pct"


def _bucket_atr_event(x: object) -> str:
    v = _safe_float(x)
    if not np.isfinite(v):
        return "NA"
    if v < 0.0015:
        return "event_atr_lt_015pct"
    if v < 0.0030:
        return "event_atr_015_030pct"
    if v < 0.0050:
        return "event_atr_030_050pct"
    return "event_atr_ge_050pct"


def attach_regimes(trades: pd.DataFrame, bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    regimes = _build_bar_regimes(bars, str(args.timeframe))
    left = trades.copy()
    left["signal_time"] = pd.to_datetime(left["signal_time"])
    regimes["signal_time"] = pd.to_datetime(regimes["signal_time"])
    out = pd.merge_asof(
        left.sort_values("signal_time"),
        regimes.sort_values("signal_time"),
        on="signal_time",
        direction="backward",
    )
    out["trend_4h_1d"] = out["trend_4h"].astype(str) + "__" + out["trend_1d"].astype(str)
    out["ret24_bucket"] = out["ret_24h"].map(_bucket_ret_24h)
    out["ret7d_bucket"] = out["ret_7d"].map(_bucket_ret_7d)
    out["event_atr_bucket"] = out["atr_pct"].map(_bucket_atr_event) if "atr_pct" in out.columns else "NA"
    out["session_bucket"] = out.get("session_bucket", "NA").fillna("NA").astype(str)
    return out.sort_values(["variant_name", "entry_time"]).reset_index(drop=True)


def build_regime_breakdown(trades_with_regime: pd.DataFrame, min_trades: int = 10) -> pd.DataFrame:
    if trades_with_regime.empty:
        return pd.DataFrame()
    regime_cols = ["session_bucket", "trend_4h", "trend_1d", "trend_4h_1d", "ret24_bucket", "ret7d_bucket", "event_atr_bucket"]
    rows: list[dict[str, object]] = []
    for regime_col in [c for c in regime_cols if c in trades_with_regime.columns]:
        for keys, grp in trades_with_regime.groupby(["variant_name", regime_col], dropna=False):
            variant_name, regime_value = keys
            ret = pd.to_numeric(grp["net_return_on_equity"], errors="coerce").fillna(0.0)
            if len(ret) < int(min_trades):
                continue
            stats = _equity_stats_from_returns(ret, 1.0)
            stats.update(
                {
                    "variant_name": variant_name,
                    "regime_name": regime_col,
                    "regime_value": regime_value,
                    "mean_return": float(ret.mean()) if not ret.empty else np.nan,
                    "median_return": float(ret.median()) if not ret.empty else np.nan,
                    "avg_mae": float(pd.to_numeric(grp.get("mae_on_equity"), errors="coerce").mean()) if "mae_on_equity" in grp.columns else np.nan,
                    "avg_mfe": float(pd.to_numeric(grp.get("mfe_on_equity"), errors="coerce").mean()) if "mfe_on_equity" in grp.columns else np.nan,
                }
            )
            rows.append(stats)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["variant_name", "regime_name", "return_total"], ascending=[True, True, False]).reset_index(drop=True)


def build_position_sizing_compare(trades: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    sizes = [0.25, 0.50, 0.75, 1.00, 1.25]
    rows: list[dict[str, object]] = []
    for variant_name, grp in trades.groupby("variant_name", dropna=False):
        g = grp.sort_values(["exit_time", "entry_time"]).copy()
        base = pd.to_numeric(g["net_return_on_equity"], errors="coerce").fillna(0.0)
        first_entry = pd.Timestamp(g["entry_time"].iloc[0])
        last_exit = pd.Timestamp(g["exit_time"].iloc[-1])
        days = max(1e-9, (last_exit - first_entry).total_seconds() / 86400.0)
        for size in sizes:
            scaled = base * float(size)
            stats = _equity_stats_from_returns(scaled, float(args.starting_equity))
            total_ret = float(stats["return_total"])
            ann_ret = float((1.0 + total_ret) ** (365.0 / days) - 1.0) if total_ret > -1.0 else -1.0
            rows.append(
                {
                    "variant_name": variant_name,
                    "position_size_mult": float(size),
                    "return_total": total_ret,
                    "return_annualized": ann_ret,
                    "profit_factor": stats["profit_factor"],
                    "win_rate": stats["win_rate"],
                    "max_drawdown": stats["max_drawdown"],
                    "worst_trade": stats["worst_trade"],
                    "best_trade": stats["best_trade"],
                    "notes": "Linear trade-return scaling for sizing sensitivity; not a liquidation or margin model.",
                }
            )
    return pd.DataFrame(rows).sort_values(["variant_name", "position_size_mult"]).reset_index(drop=True)


def build_variant_compare(summary: pd.DataFrame, yearly: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    out = summary.copy()
    if not yearly.empty:
        ys = yearly.groupby("variant_name").agg(
            tested_years=("year", "count"),
            positive_years=("return_total", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            worst_year=("return_total", "min"),
        ).reset_index()
        out = out.merge(ys, on="variant_name", how="left")
    if not monthly.empty:
        ms = monthly.groupby("variant_name").agg(
            tested_months=("month", "count"),
            positive_months=("return_total", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
            worst_month=("return_total", "min"),
        ).reset_index()
        out = out.merge(ms, on="variant_name", how="left")
    out["formal_priority_score"] = (
        pd.to_numeric(out.get("return_total"), errors="coerce").fillna(0.0) * 1.0
        + pd.to_numeric(out.get("profit_factor"), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) * 0.25
        + pd.to_numeric(out.get("win_rate"), errors="coerce").fillna(0.0) * 0.25
        + pd.to_numeric(out.get("max_drawdown"), errors="coerce").fillna(0.0) * 1.0
        - pd.to_numeric(out.get("top5_winner_share"), errors="coerce").fillna(1.0) * 0.10
    )
    return out.sort_values(["formal_priority_score", "return_total", "profit_factor"], ascending=[False, False, False]).reset_index(drop=True)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    print(f"[write] {path.name}", flush=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_a_validation_backtest(args: argparse.Namespace) -> None:
    if bool(args.fast):
        args.skip_cost_stress = True
        args.skip_delay_stress = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[scope] {SCRIPT_NAME} | research-only A validation, not live-ready", flush=True)
    print("[load] loading trade bars", flush=True)
    bars = load_trade_bars(args)

    events = prepare_studied_events(bars, args)
    if events.empty:
        raise RuntimeError("No studied low-sweep events generated")

    variants = build_variants(args)
    print(
        f"[setup] variants={len(variants):,} candidates={args.candidate_names} schemes={args.entry_schemes} horizons={args.exit_horizons} stops={args.stop_specs}",
        flush=True,
    )

    trades, summary = run_variant_jobs(
        bars,
        events,
        variants,
        args,
        cost_mult=1.0,
        label="[backtest] A base variants",
        keep_trades=True,
    )
    if trades.empty:
        raise RuntimeError("A validation backtest produced no trades")

    equity_curve = build_equity_curve(trades, args)
    yearly = summarize_by_period(trades, args, "year")
    monthly = summarize_by_period(trades, args, "month")
    weekly = build_weekly_summary(trades, args)
    drawdown_curve = build_drawdown_curve(equity_curve)
    trade_distribution = build_trade_distribution(trades)
    tail_loss_audit = build_tail_loss_audit(trades, top_n=25)
    stop_reason_audit = build_stop_reason_audit(trades)

    print("[regime] attaching causal trend/return regimes", flush=True)
    trades_with_regime = attach_regimes(trades, bars, args)
    regime_breakdown = build_regime_breakdown(trades_with_regime, min_trades=10)

    cost_stress = pd.DataFrame()
    if not bool(args.skip_cost_stress):
        cost_multipliers = _parse_number_list(args.cost_multipliers, cast=float, name="cost_multipliers")
        stress_parts = [summary]
        stress_mults = [m for m in cost_multipliers if abs(float(m) - 1.0) > 1e-12]
        print(f"[stress] cost multipliers={stress_mults}", flush=True)
        progress = ProgressReporter(label="[stress] A cost groups", total=len(stress_mults), every=1, enabled=not bool(args.no_progress))
        for i, mult in enumerate(stress_mults, start=1):
            _, stress_summary = run_variant_jobs(
                bars,
                events,
                variants,
                args,
                cost_mult=float(mult),
                label=f"[stress] A cost {float(mult):g}x variants",
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
        progress = ProgressReporter(label="[stress] A delay groups", total=len(delay_values), every=1, enabled=not bool(args.no_progress))
        for i, delay in enumerate(delay_values, start=1):
            _, delay_summary = run_variant_jobs(
                bars,
                events,
                variants,
                args,
                cost_mult=1.0,
                entry_delay_bars=int(delay),
                label=f"[stress] A delay {int(delay)} variants",
                keep_trades=False,
            )
            stress_parts.append(delay_summary)
            progress.update(i)
        progress.close()
        delay_stress = pd.concat(stress_parts, ignore_index=True) if stress_parts else pd.DataFrame()

    sizing_compare = build_position_sizing_compare(trades, args)
    variant_compare = build_variant_compare(summary, yearly, monthly)
    registry = build_edge_registry(summary, yearly, monthly, args)

    # Keep enriched trades for audit but avoid duplicating huge full feature frames.
    trade_out = trades_with_regime.copy()
    if int(args.save_trade_sample) > 0 and len(trade_out) > int(args.save_trade_sample):
        trade_out = trade_out.sort_values(["variant_name", "entry_time"]).head(int(args.save_trade_sample)).copy()

    signal_cols = [
        "signal_time",
        "event_name",
        "close",
        "low",
        "structural_stop_level",
        "atr_pct",
        "down_spike_pct",
        "large_trade_share",
        "session_bucket",
    ]
    signals = events[[c for c in signal_cols if c in events.columns]].copy()

    write_csv(signals, out_dir / "01_signals.csv")
    write_csv(trade_out, out_dir / "02_trades.csv")
    write_csv(summary, out_dir / "03_summary.csv")
    write_csv(yearly, out_dir / "04_yearly.csv")
    write_csv(monthly, out_dir / "05_monthly.csv")
    write_csv(weekly, out_dir / "06_weekly.csv")
    write_csv(equity_curve, out_dir / "07_equity_curve.csv")
    write_csv(drawdown_curve, out_dir / "08_drawdown_curve.csv")
    write_csv(trade_distribution, out_dir / "09_trade_distribution.csv")
    write_csv(tail_loss_audit, out_dir / "10_tail_loss_audit.csv")
    write_csv(stop_reason_audit, out_dir / "11_stop_reason_audit.csv")
    write_csv(regime_breakdown, out_dir / "12_regime_breakdown.csv")
    write_csv(cost_stress, out_dir / "13_cost_stress.csv")
    write_csv(delay_stress, out_dir / "14_delay_stress.csv")
    write_csv(sizing_compare, out_dir / "15_position_sizing_compare.csv")
    write_csv(variant_compare, out_dir / "16_variant_compare.csv")
    write_csv(registry, out_dir / "17_edge_registry_update.csv")

    meta = {
        "script": SCRIPT_NAME,
        "status": "research_validation_not_live_ready",
        "placement_decision": "kept under research/ until A passes tail-risk, regime, sizing, stress, and portfolio validation; promote to backtest/mf only after that.",
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
        "cost_multipliers": _parse_number_list(args.cost_multipliers, cast=float, name="cost_multipliers"),
        "delay_bars_list": _parse_number_list(args.delay_bars_list, cast=int, name="delay_bars_list"),
        "leakage_guard": {
            "data_loader": "OKXTradeBarLoader via upstream load_trade_bars",
            "signal_filters": "A_spike_close_large_share from no-leakage probe; no full-sample qcut filters",
            "rolling_thresholds": "upstream shifted rolling quantiles only",
            "entry_timing": "signal on closed bar; entry at future open after entry_delay_bars",
            "conflict_resolution": "one position at a time; overlapping signals skipped",
            "regime_breakdown": "uses signal-time completed-bar close/EMA/returns only; no future bars",
            "scale_in": "post-entry only; capped max_position_weight; no unlimited martingale",
        },
    }
    _write_json(out_dir / "18_lab_meta.json", meta)
    print(f"[done] wrote A validation backtest reports to {out_dir}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_a_validation_backtest(args)


if __name__ == "__main__":
    main()
