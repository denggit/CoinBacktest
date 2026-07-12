#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10B Structural Stop Neighbourhood Check
========================================

Runs the V10B all-engine swing structural stop family around the promoted
candidate without modifying official V10B defaults.

Default grid:
    lookback = 13, 21, 34
    buffer_atr = 0.0, 0.1, 0.25
    trigger_mfe_r = 0
    min_hold_bars = 0

This is a research-only robustness check. It imports the V10B executor and
passes custom StructuralStopConfig objects in-process, so the official V10B
backtest script can remain fixed at n=21/buffer=0.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.lf import eth_lf_portfolio_v10a_momentum_micro_short_speed_filter_backtest as v10a  # noqa: E402
from backtest.lf import eth_lf_portfolio_v10b_all_swing_structural_stop_backtest as v10b  # noqa: E402

OUT_NAME = "v10b_structural_neighbourhood"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V10B all-swing structural stop neighbourhood check.")
    p.add_argument("--out-dir", default=f"data/reports/research/{OUT_NAME}")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--initial-capital", default="1000")
    p.add_argument("--range-pct", default="0.002")
    p.add_argument("--price-step", default="1.0")
    p.add_argument("--lookbacks", default="13,21,34")
    p.add_argument("--buffers", default="0,0.1,0.25")
    p.add_argument("--write-trades", action="store_true")
    # Reuse the full V10A CLI by parsing known args after this parser is not viable,
    # so accept extra args and pass them into v10a defaults through a small shim.
    args, unknown = p.parse_known_args()
    args._unknown = unknown
    return args


def _grid_int(s: str) -> list[int]:
    return [int(float(x.strip())) for x in s.split(",") if x.strip()]


def _grid_float(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(PROJECT_ROOT) / p


def _summary_row(name: str, trades: list[dict[str, Any]], equity: pd.DataFrame, exec_cfg: Any) -> dict[str, Any]:
    summary = v10a.summarize(trades, equity, exec_cfg.initial_capital)
    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        notes = tdf.get("note", pd.Series(dtype=str)).astype(str)
        summary["structural_stop_exit_count"] = int(notes.eq("STRUCTURAL_STOP").sum())
        summary["protected_trailing_stop_exit_count"] = int(notes.eq("PROTECTED_TRAILING_STOP").sum())
        summary["structural_stop_total_updates"] = int(pd.to_numeric(tdf.get("structure_updates", 0), errors="coerce").fillna(0).sum())
        summary["structural_stop_trade_count"] = int(pd.to_numeric(tdf.get("structure_updates", 0), errors="coerce").fillna(0).gt(0).sum())
    else:
        summary["structural_stop_exit_count"] = 0
        summary["protected_trailing_stop_exit_count"] = 0
        summary["structural_stop_total_updates"] = 0
        summary["structural_stop_trade_count"] = 0
    return {"variant": name, **summary}


def main() -> int:
    args = parse_args()
    out_dir = _project_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build a normal V10A argparse namespace first, then override the common fields.
    old_argv = sys.argv[:]
    try:
        sys.argv = [sys.argv[0], *args._unknown]
        v10a_args = v10a.parse_args()
    finally:
        sys.argv = old_argv
    v10a_args.start_date = args.start_date
    v10a_args.end_date = args.end_date
    v10a_args.warmup_start_date = args.warmup_start_date
    v10a_args.initial_capital = float(args.initial_capital)
    v10a_args.range_pct = float(args.range_pct)
    v10a_args.price_step = float(args.price_step)

    mom_cfg = v10a.make_momentum_config(v10a_args)
    bear_cfg = v10a.make_bear_config(v10a_args)
    bull_cfg = v10a.make_bull_config(v10a_args)
    exec_cfg = v10a.make_exec_config(mom_cfg)
    bull_exec_cfg = v10a.bull_to_exec_config(bull_cfg) if v10a_args.bull_execution_mode == "own" else exec_cfg

    trade_start = pd.Timestamp(v10a_args.start_date)
    load_start = pd.Timestamp(v10a_args.warmup_start_date) if v10a_args.warmup_start_date else trade_start
    load_start_str = load_start.strftime("%Y-%m-%d")

    print(f"Loading {v10a_args.symbol} 4H for neighbourhood: {load_start_str} -> {v10a_args.end_date}")
    base = v10a.load_data(v10a_args.symbol, load_start_str, v10a_args.end_date, "4H")
    momentum = v10a.build_momentum_features(base, mom_cfg)
    bear = v10a.build_bear_features(base, bear_cfg)
    bull = v10a.build_bull_features(base, bull_cfg)
    micro_ctx = v10a.load_range_footprint_context(v10a_args, load_start_str, v10a_args.end_date)
    momentum = v10a.apply_momentum_long_not_aligned_block(momentum, micro_ctx, v10a_args)
    momentum = v10a.apply_momentum_short_fast_speed_block(momentum, micro_ctx, v10a_args)
    features_base = v10a.select_portfolio_signals(momentum, bear, bull, v10a_args)
    features_base = v10a.apply_micro_context_filter(features_base, micro_ctx, v10a_args)

    rows = []
    combined = []
    for lookback in _grid_int(args.lookbacks):
        features = v10b.add_structural_columns(features_base, lookback_bars=lookback)
        features = features.loc[trade_start: pd.Timestamp(v10a_args.end_date)].copy()
        for buffer_atr in _grid_float(args.buffers):
            cfg = v10b.StructuralStopConfig(
                enabled=True,
                lookback_bars=int(lookback),
                buffer_atr=float(buffer_atr),
                trigger_mfe_r=0.0,
                min_hold_bars=0,
                engine_scope="ALL",
                source="swing",
                tighten_only=True,
            )
            name = f"all_swing_n{lookback}_buf{str(buffer_atr).replace('.', 'p')}"
            print(f"Running {name}")
            trades, equity = v10b.run_v10b_backtest(
                features,
                exec_cfg,
                engine_cfgs={"MOMENTUM_V3": exec_cfg, "BEAR_V3_ONLY": exec_cfg, "BULL_RECLAIM_V2": bull_exec_cfg},
                global_risk_scale=v10a_args.global_risk_scale,
                args=v10a_args,
                structural_cfg=cfg,
            )
            trades = v10a.attach_engine_to_trades(trades, features)
            row = _summary_row(name, trades, equity, exec_cfg)
            row["lookback_bars"] = int(lookback)
            row["buffer_atr"] = float(buffer_atr)
            rows.append(row)
            if args.write_trades and trades:
                tdf = pd.DataFrame(trades)
                tdf["variant"] = name
                combined.append(tdf)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["total_return_pct", "max_drawdown_pct"], ascending=[False, True])
    df.to_csv(out_dir / "01_neighbourhood_summary.csv", index=False)
    if combined:
        pd.concat(combined, ignore_index=True, sort=False).to_csv(out_dir / "02_neighbourhood_trades.csv", index=False)
    meta = {
        "generated_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lookbacks": _grid_int(args.lookbacks),
        "buffers": _grid_float(args.buffers),
        "policy": "Research-only neighbourhood check; official V10B remains n=21/buffer=0 until promoted after review.",
        "args": vars(args),
    }
    with (out_dir / "99_research_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print("Neighbourhood outputs:", out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
