#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10B Win-Rate Full Research Suite
=================================

Research-only diagnostics for ETH LF Portfolio V10B.

Purpose:
  - Study V10B low win-rate from both signal layer and portfolio-backtest layer.
  - Produce evidence before any V10C/V11 promotion.
  - Never modify official backtest strategy files or AetherEdge live plugin.

Default mode reads existing V10B report CSV/JSON and writes diagnostics.
Optional --rebuild-features rebuilds V10B features in-process and can run
research-only counterfactual entry variants without writing a new strategy version.

No-lookahead policy:
  - Production/candidate variant rules in this file only use same closed-bar fields
    that V10B already writes to signal_audit.
  - Signal forward-return tables intentionally use future returns as labels for
    diagnostics only; they are explicitly marked as research labels, not trading rules.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.lf import eth_lf_portfolio_v10a_momentum_micro_short_speed_filter_backtest as v10a  # noqa: E402
from backtest.lf import eth_lf_portfolio_v10b_all_swing_structural_stop_backtest as v10b  # noqa: E402

V10B_NAME = "eth_lf_portfolio_v10b_all_swing_structural_stop"
DEFAULT_V10B_DIR = f"data/reports/lf/{V10B_NAME}/turbo"
OUT_NAME = "v10b_winrate_full_research"
ENGINE_COLUMNS = {
    "selected": ("signal", "selected_engine"),
    "momentum_raw": ("momentum_signal", "MOMENTUM_V3_RAW"),
    "bear_raw": ("bear_signal", "BEAR_V3_ONLY_RAW"),
    "bull_raw": ("bull_signal", "BULL_RECLAIM_V2_RAW"),
}
FORWARD_HORIZONS = (1, 2, 3, 6, 12, 18, 30)


@dataclass(frozen=True)
class GateConfig:
    min_win_rate_lift_pp: float = 2.0
    min_return_ratio: float = 0.80
    min_profit_factor_ratio: float = 0.95
    max_drawdown_ratio: float = 1.00
    min_trade_ratio: float = 0.70
    max_top3_return_share: float = 0.75


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research-only V10B win-rate diagnostics and candidate triage.")
    p.add_argument("--out-dir", default=f"data/reports/research/{OUT_NAME}")
    p.add_argument("--v10b-dir", default=DEFAULT_V10B_DIR, help="Existing V10B report directory.")
    p.add_argument("--rebuild-features", action="store_true", help="Rebuild V10B features and run in-process counterfactual variants.")
    p.add_argument("--write-variant-trades", action="store_true", help="When --rebuild-features is used, also write combined variant trades.")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--range-pct", type=float, default=0.002)
    p.add_argument("--price-step", type=float, default=1.0)
    p.add_argument("--min-win-rate-lift-pp", type=float, default=2.0)
    p.add_argument("--min-return-ratio", type=float, default=0.80)
    p.add_argument("--min-profit-factor-ratio", type=float, default=0.95)
    p.add_argument("--max-drawdown-ratio", type=float, default=1.00)
    p.add_argument("--min-trade-ratio", type=float, default=0.70)
    p.add_argument("--max-top3-return-share", type=float, default=0.75)
    args, unknown = p.parse_known_args()
    args._unknown = unknown
    return args


def project_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(PROJECT_ROOT) / p


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def find_file(report_dir: Path, suffix: str) -> Path:
    direct = report_dir / suffix
    if direct.exists():
        return direct
    matches = sorted(report_dir.glob(f"*{suffix}"))
    if matches:
        return matches[0]
    matches = sorted(report_dir.rglob(f"*{suffix}"))
    return matches[0] if matches else direct


def load_v10b_report(report_dir: Path) -> dict[str, Any]:
    summary = read_json(find_file(report_dir, f"{V10B_NAME}_summary.json"))
    trades = read_csv(find_file(report_dir, f"{V10B_NAME}_trades.csv"))
    equity = read_csv(find_file(report_dir, f"{V10B_NAME}_equity.csv"))
    audit = read_csv(find_file(report_dir, f"{V10B_NAME}_signal_audit.csv"))
    if not audit.empty and "timestamp" in audit.columns:
        audit["timestamp"] = pd.to_datetime(audit["timestamp"], errors="coerce")
        audit = audit.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if not trades.empty:
        for col in ["entry_time", "exit_time"]:
            if col in trades.columns:
                trades[col] = pd.to_datetime(trades[col], errors="coerce")
    return {"summary": summary, "trades": trades, "equity": equity, "audit": audit, "dir": report_dir}


def safe_num(s: Any, default: float = float("nan")) -> float:
    try:
        x = float(s)
    except Exception:
        return default
    return x if math.isfinite(x) else default


def num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def bool_series(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return df[col].astype("boolean").fillna(default).astype(bool)


def side_int(df: pd.DataFrame) -> pd.Series:
    if "type" in df.columns:
        return df["type"].astype(str).str.upper().map({"LONG": 1, "SHORT": -1}).fillna(0).astype(int)
    if "side" in df.columns:
        return pd.to_numeric(df["side"], errors="coerce").fillna(0).astype(int)
    return pd.Series(0, index=df.index, dtype="int64")


def summarize_trades(trades: pd.DataFrame, initial_capital: float = 1000.0) -> dict[str, Any]:
    if trades.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "total_return_pct": 0.0,
            "profit_factor": 0.0,
            "expectancy_pct": 0.0,
            "max_drawdown_pct": float("nan"),
        }
    ret = num_series(trades, "return_pct", 0.0)
    pnl = num_series(trades, "pnl", 0.0)
    wins = ret.gt(0)
    gp = float(pnl[pnl > 0].sum())
    gl = abs(float(pnl[pnl < 0].sum()))
    final_capital = safe_num(trades.iloc[-1].get("capital", initial_capital), initial_capital)
    return {
        "total_trades": int(len(trades)),
        "long_trades": int(side_int(trades).eq(1).sum()),
        "short_trades": int(side_int(trades).eq(-1).sum()),
        "win_rate": float(wins.mean() * 100.0),
        "total_return_pct": float((final_capital / max(initial_capital, 1e-12) - 1.0) * 100.0),
        "profit_factor": float(gp / gl) if gl > 0 else float("inf"),
        "expectancy_pct": float(ret.mean() * 100.0),
        "avg_mfe_r": float(num_series(trades, "mfe_r", float("nan")).mean()),
        "avg_mae_r": float(num_series(trades, "mae_r", float("nan")).mean()),
        "avg_holding_hours": float(num_series(trades, "holding_hours", float("nan")).mean()),
        "avg_units": float(num_series(trades, "units", float("nan")).mean()),
        "gross_profit": gp,
        "gross_loss": gl,
    }


def build_baseline_summary(summary: dict[str, Any], trades: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "total_trades", "long_trades", "short_trades", "total_return_pct", "win_rate",
        "profit_factor", "expectancy_pct", "max_drawdown_pct", "avg_mfe_r", "avg_mae_r",
        "avg_units", "avg_risk_mult", "avg_holding_hours", "mfe_ge_1r_ended_loss",
        "mfe_ge_2r_ended_loss", "total_fees", "range_exit_trade_count",
        "structural_stop_trade_count", "structural_stop_total_updates", "structural_stop_exit_count",
        "protected_trailing_stop_exit_count",
    ]
    rows = []
    for key in keys:
        rows.append({"metric": key, "value": summary.get(key, float("nan"))})
    if not trades.empty:
        ret = num_series(trades, "return_pct", 0.0)
        winners = trades.loc[ret.gt(0)].copy()
        winners["return_pct_num"] = ret[ret.gt(0)]
        total_win_ret = float(winners["return_pct_num"].sum())
        top3 = float(winners.sort_values("return_pct_num", ascending=False).head(3)["return_pct_num"].sum())
        rows.extend([
            {"metric": "top3_winner_return_share_of_winning_return", "value": top3 / total_win_ret if total_win_ret > 0 else float("nan")},
            {"metric": "mfe_ge_1r_ended_loss_from_trades", "value": int((num_series(trades, "mfe_r", 0.0).ge(1.0) & ret.le(0)).sum())},
            {"metric": "mfe_ge_2r_ended_loss_from_trades", "value": int((num_series(trades, "mfe_r", 0.0).ge(2.0) & ret.le(0)).sum())},
            {"metric": "add_on_trades_units_ge_2", "value": int(num_series(trades, "units", 0.0).ge(2).sum())},
            {"metric": "add_on_loser_units_ge_2", "value": int((num_series(trades, "units", 0.0).ge(2) & ret.le(0)).sum())},
        ])
    return pd.DataFrame(rows)


def group_trade_breakdown(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    t = trades.copy()
    t["side_int"] = side_int(t)
    t["return_pct_num"] = num_series(t, "return_pct", 0.0)
    t["pnl_num"] = num_series(t, "pnl", 0.0)
    t["mfe_r_num"] = num_series(t, "mfe_r", float("nan"))
    t["mae_r_num"] = num_series(t, "mae_r", float("nan"))
    t["units_num"] = num_series(t, "units", float("nan"))
    t["win"] = t["return_pct_num"].gt(0)
    t["mfe_ge_1r_loss"] = t["mfe_r_num"].ge(1.0) & t["return_pct_num"].le(0)
    t["mfe_ge_2r_loss"] = t["mfe_r_num"].ge(2.0) & t["return_pct_num"].le(0)
    t["exit_note"] = t.get("note", pd.Series("NA", index=t.index)).astype(str)
    t["micro_bucket"] = t.get("micro_filter_action", pd.Series("NA", index=t.index)).astype(str)
    t["unit_bucket"] = pd.cut(
        t["units_num"],
        bins=[-np.inf, 1, 2, 3, np.inf],
        labels=["1_unit", "2_units", "3_units", "4p_units"],
    ).astype(str)

    def agg(group_cols: list[str]) -> pd.DataFrame:
        grouped = []
        for key, g in t.groupby(group_cols, dropna=False):
            key_tuple = key if isinstance(key, tuple) else (key,)
            row = {col: val for col, val in zip(group_cols, key_tuple)}
            gp = float(g.loc[g["pnl_num"] > 0, "pnl_num"].sum())
            gl = abs(float(g.loc[g["pnl_num"] < 0, "pnl_num"].sum()))
            row.update({
                "trades": int(len(g)),
                "win_rate": float(g["win"].mean() * 100.0),
                "return_pct_sum": float(g["return_pct_num"].sum() * 100.0),
                "return_pct_avg": float(g["return_pct_num"].mean() * 100.0),
                "profit_factor_proxy": float(gp / gl) if gl > 0 else float("inf"),
                "avg_mfe_r": float(g["mfe_r_num"].mean()),
                "avg_mae_r": float(g["mae_r_num"].mean()),
                "mfe_ge_1r_loss_count": int(g["mfe_ge_1r_loss"].sum()),
                "mfe_ge_2r_loss_count": int(g["mfe_ge_2r_loss"].sum()),
                "avg_units": float(g["units_num"].mean()),
            })
            grouped.append(row)
        out = pd.DataFrame(grouped)
        if not out.empty:
            out = out.sort_values(["return_pct_sum", "trades"], ascending=[False, False])
        return out

    engine_side = agg(["engine", "side_int"])
    exit_micro_units = pd.concat(
        [
            agg(["exit_note"]).assign(breakdown="exit_note"),
            agg(["micro_bucket"]).assign(breakdown="micro_bucket"),
            agg(["unit_bucket"]).assign(breakdown="unit_bucket"),
            agg(["engine", "micro_bucket"]).assign(breakdown="engine_micro"),
        ],
        ignore_index=True,
        sort=False,
    )
    return engine_side, exit_micro_units


def yearly_monthly_breakdown(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty or "exit_time" not in trades.columns:
        return pd.DataFrame(), pd.DataFrame()
    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"], errors="coerce")
    t = t.dropna(subset=["exit_time"])
    t["return_pct_num"] = num_series(t, "return_pct", 0.0)
    t["win"] = t["return_pct_num"].gt(0)
    t["year"] = t["exit_time"].dt.year
    t["month"] = t["exit_time"].dt.to_period("M").astype(str)

    def agg(col: str) -> pd.DataFrame:
        rows = []
        for key, g in t.groupby(col):
            rows.append({
                col: key,
                "trades": int(len(g)),
                "win_rate": float(g["win"].mean() * 100.0),
                "return_pct_sum": float(g["return_pct_num"].sum() * 100.0),
                "avg_return_pct": float(g["return_pct_num"].mean() * 100.0),
                "best_trade_pct": float(g["return_pct_num"].max() * 100.0),
                "worst_trade_pct": float(g["return_pct_num"].min() * 100.0),
            })
        return pd.DataFrame(rows)

    return agg("year"), agg("month")


def _side_forward_return(audit: pd.DataFrame, signal_col: str, horizons: Iterable[int]) -> pd.DataFrame:
    rows = []
    close = num_series(audit, "close", float("nan"))
    signal = pd.to_numeric(audit.get(signal_col, pd.Series(0, index=audit.index)), errors="coerce").fillna(0).astype(int)
    for h in horizons:
        fwd = close.shift(-int(h)) / close - 1.0
        side_fwd = fwd * signal
        rows.append((h, side_fwd))
    return pd.DataFrame({f"fwd_{h}bar_side_return": s for h, s in rows}, index=audit.index)


def _quantile_bucket(s: pd.Series, name: str, q: int = 4) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    if valid.nunique() < 2:
        return pd.Series("NA", index=s.index, dtype="object")
    try:
        bucket = pd.qcut(s, q=q, labels=[f"{name}_Q{i}" for i in range(1, q + 1)], duplicates="drop")
    except ValueError:
        return pd.Series("NA", index=s.index, dtype="object")
    return bucket.astype("object").fillna("NA")


def signal_forward_edge(audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if audit.empty:
        return pd.DataFrame(), pd.DataFrame()
    out_rows = []
    bucket_rows = []
    base = audit.copy()
    close = num_series(base, "close", float("nan"))
    base["rf_bar_count_bucket"] = _quantile_bucket(base.get("rf_bar_count", pd.Series(np.nan, index=base.index)), "rf_count")
    base["atr_pct_bucket"] = _quantile_bucket(base.get("atr_pct", pd.Series(np.nan, index=base.index)), "atr_pct")
    base["risk_mult_bucket"] = _quantile_bucket(base.get("risk_mult", pd.Series(np.nan, index=base.index)), "risk")
    base["micro_bucket"] = base.get("micro_filter_action", pd.Series("NA", index=base.index)).astype(str)
    base["selected_engine_bucket"] = base.get("selected_engine", pd.Series("NA", index=base.index)).astype(str)

    for family, (sig_col, engine_label) in ENGINE_COLUMNS.items():
        if sig_col not in base.columns:
            continue
        signal = pd.to_numeric(base[sig_col], errors="coerce").fillna(0).astype(int)
        active = signal.ne(0)
        fwd_df = _side_forward_return(base, sig_col, FORWARD_HORIZONS)
        work = pd.concat([base, fwd_df], axis=1)
        work["signal_side"] = signal
        if family == "selected":
            work["engine_family"] = work["selected_engine_bucket"]
        else:
            work["engine_family"] = engine_label
        for h in FORWARD_HORIZONS:
            col = f"fwd_{h}bar_side_return"
            g = work.loc[active & close.gt(0) & work[col].notna()]
            if g.empty:
                continue
            out_rows.append({
                "signal_family": family,
                "engine_family": engine_label,
                "horizon_bars": int(h),
                "events": int(len(g)),
                "positive_rate": float(g[col].gt(0).mean() * 100.0),
                "avg_side_return_pct": float(g[col].mean() * 100.0),
                "median_side_return_pct": float(g[col].median() * 100.0),
                "p25_side_return_pct": float(g[col].quantile(0.25) * 100.0),
                "p75_side_return_pct": float(g[col].quantile(0.75) * 100.0),
            })
            for bucket_col in ["engine_family", "micro_bucket", "rf_bar_count_bucket", "atr_pct_bucket", "risk_mult_bucket"]:
                for bucket, bg in g.groupby(bucket_col, dropna=False):
                    if len(bg) < 5:
                        continue
                    bucket_rows.append({
                        "signal_family": family,
                        "bucket_type": bucket_col,
                        "bucket": str(bucket),
                        "horizon_bars": int(h),
                        "events": int(len(bg)),
                        "positive_rate": float(bg[col].gt(0).mean() * 100.0),
                        "avg_side_return_pct": float(bg[col].mean() * 100.0),
                        "median_side_return_pct": float(bg[col].median() * 100.0),
                    })

    # Blocked raw signal diagnostics: these are labels only, useful to verify whether existing blocks killed good signals.
    blocked_specs = [
        ("momentum_long_not_aligned_blocked", "momentum_signal", 1),
        ("momentum_short_fast_speed_blocked", "momentum_signal", -1),
    ]
    for flag_col, sig_col, forced_side in blocked_specs:
        if flag_col not in base.columns or sig_col not in base.columns:
            continue
        flag = bool_series(base, flag_col, False)
        if not flag.any():
            continue
        tmp = base.copy()
        tmp["_blocked_side"] = int(forced_side)
        for h in FORWARD_HORIZONS:
            col = f"blocked_fwd_{h}bar_side_return"
            tmp[col] = (close.shift(-int(h)) / close - 1.0) * forced_side
            g = tmp.loc[flag & tmp[col].notna()]
            if g.empty:
                continue
            out_rows.append({
                "signal_family": flag_col,
                "engine_family": "MOMENTUM_V3_BLOCKED_LABEL_ONLY",
                "horizon_bars": int(h),
                "events": int(len(g)),
                "positive_rate": float(g[col].gt(0).mean() * 100.0),
                "avg_side_return_pct": float(g[col].mean() * 100.0),
                "median_side_return_pct": float(g[col].median() * 100.0),
                "p25_side_return_pct": float(g[col].quantile(0.25) * 100.0),
                "p75_side_return_pct": float(g[col].quantile(0.75) * 100.0),
                "research_label_only": True,
            })

    signal_df = pd.DataFrame(out_rows)
    bucket_df = pd.DataFrame(bucket_rows)
    if not signal_df.empty:
        signal_df = signal_df.sort_values(["signal_family", "horizon_bars", "events"], ascending=[True, True, False])
    if not bucket_df.empty:
        bucket_df = bucket_df.sort_values(["signal_family", "horizon_bars", "avg_side_return_pct"], ascending=[True, True, False])
    return signal_df, bucket_df


def mfe_salvage_table(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["return_pct_num"] = num_series(t, "return_pct", 0.0)
    t["mfe_r_num"] = num_series(t, "mfe_r", float("nan"))
    t["mae_r_num"] = num_series(t, "mae_r", float("nan"))
    t["units_num"] = num_series(t, "units", 0.0)
    t["salvage_class"] = "OTHER"
    t.loc[t["return_pct_num"].le(0) & t["mfe_r_num"].ge(2.0), "salvage_class"] = "MFE_GE_2R_ENDED_LOSS"
    t.loc[t["return_pct_num"].le(0) & t["mfe_r_num"].ge(1.0) & t["mfe_r_num"].lt(2.0), "salvage_class"] = "MFE_1R_TO_2R_ENDED_LOSS"
    t.loc[t["return_pct_num"].le(0) & t["units_num"].ge(2), "salvage_class"] = t.loc[t["return_pct_num"].le(0) & t["units_num"].ge(2), "salvage_class"].astype(str) + "+ADDON_LOSS"
    cols = [
        "entry_time", "exit_time", "type", "engine", "first_entry", "avg_entry", "exit",
        "units", "return_pct", "pnl", "mfe_r", "mae_r", "holding_bars_4h", "note",
        "micro_filter_action", "micro_aligned", "micro_contra", "rf_imbalance", "rf_close_pos",
        "structure_updates", "active_stop_source_at_exit", "salvage_class",
    ]
    out = t.loc[t["salvage_class"].ne("OTHER"), [c for c in cols if c in t.columns]].copy()
    if not out.empty:
        out = out.sort_values(["mfe_r", "return_pct"], ascending=[False, True])
    return out


def _build_v10b_features(args: argparse.Namespace) -> tuple[pd.DataFrame, Any, Any, argparse.Namespace]:
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

    print(f"Loading V10B feature research data: {v10a_args.symbol} 4H {load_start_str}->{v10a_args.end_date}", flush=True)
    base = v10a.load_data(v10a_args.symbol, load_start_str, v10a_args.end_date, "4H")
    momentum = v10a.build_momentum_features(base, mom_cfg)
    bear = v10a.build_bear_features(base, bear_cfg)
    bull = v10a.build_bull_features(base, bull_cfg)
    micro_ctx = v10a.load_range_footprint_context(v10a_args, load_start_str, v10a_args.end_date)
    momentum = v10a.apply_momentum_long_not_aligned_block(momentum, micro_ctx, v10a_args)
    momentum = v10a.apply_momentum_short_fast_speed_block(momentum, micro_ctx, v10a_args)
    features = v10a.select_portfolio_signals(momentum, bear, bull, v10a_args)
    features = v10a.apply_micro_context_filter(features, micro_ctx, v10a_args)
    features = v10b.add_structural_columns(features, lookback_bars=v10b.V10B_STRUCTURAL_STOP.lookback_bars)
    features = features.loc[trade_start: pd.Timestamp(v10a_args.end_date)].copy()
    return features, exec_cfg, bull_exec_cfg, v10a_args


def _mask(features: pd.DataFrame, name: str) -> pd.Series:
    selected = features.get("selected_engine", pd.Series("NONE", index=features.index)).astype(str)
    action = features.get("micro_filter_action", pd.Series("NA", index=features.index)).astype(str)
    has_ctx = bool_series(features, "micro_context_available", False)
    aligned = bool_series(features, "micro_aligned", False)
    contra = bool_series(features, "micro_contra", False)
    signal = pd.to_numeric(features.get("signal", pd.Series(0, index=features.index)), errors="coerce").fillna(0).astype(int)
    if name == "bull_not_aligned":
        return selected.eq("BULL_RECLAIM_V2") & has_ctx & signal.eq(1) & action.eq("NOT_ALIGNED_RISK_REDUCED")
    if name == "bull_any_not_aligned":
        return selected.eq("BULL_RECLAIM_V2") & has_ctx & signal.eq(1) & (~aligned) & (~contra)
    if name == "bull_contra":
        return selected.eq("BULL_RECLAIM_V2") & has_ctx & signal.eq(1) & contra
    if name == "all_contra":
        return signal.ne(0) & has_ctx & contra
    if name == "all_not_aligned":
        return signal.ne(0) & has_ctx & (~aligned) & (~contra)
    if name == "low_quality_bottom_quartile":
        qm = pd.to_numeric(features.get("quality_mult", pd.Series(np.nan, index=features.index)), errors="coerce")
        active_q = qm[signal.ne(0)].dropna()
        if active_q.empty:
            return pd.Series(False, index=features.index)
        threshold = active_q.quantile(0.25)
        return signal.ne(0) & qm.le(threshold)
    return pd.Series(False, index=features.index)


def apply_variant(features: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = features.copy()
    meta: dict[str, Any] = {"variant": variant, "research_only": True}
    if variant == "baseline_v10b":
        meta.update({"changed_signals": 0, "rule": "No change; formal V10B baseline."})
        return out, meta

    if variant.startswith("block_"):
        mask_name = variant.removeprefix("block_")
        mask = _mask(out, mask_name)
        changed = int(mask.sum())
        out.loc[mask, "signal"] = 0
        for col in ["long_signal", "short_signal"]:
            if col in out.columns:
                out.loc[mask, col] = False
        if "selected_engine" in out.columns:
            out.loc[mask, "selected_engine"] = "RESEARCH_BLOCKED"
        meta.update({"changed_signals": changed, "rule": f"Set signal=0 where mask={mask_name}."})
        return out, meta

    scale_specs = {
        "scale_bull_not_aligned_0p25": ("bull_not_aligned", 0.25),
        "scale_bull_not_aligned_0p35": ("bull_not_aligned", 0.35),
        "scale_bull_not_aligned_0p50": ("bull_not_aligned", 0.50),
        "scale_all_not_aligned_0p35": ("all_not_aligned", 0.35),
        "scale_all_contra_0p25": ("all_contra", 0.25),
    }
    if variant in scale_specs:
        mask_name, scale = scale_specs[variant]
        mask = _mask(out, mask_name)
        changed = int(mask.sum())
        # Entry sizing multiplies risk_mult * quality_mult * micro_entry_risk_scale.
        # For research-only risk-down, change micro_entry_risk_scale on selected signal bars.
        if "micro_entry_risk_scale" not in out.columns:
            out["micro_entry_risk_scale"] = 1.0
        out.loc[mask, "micro_entry_risk_scale"] = pd.to_numeric(out.loc[mask, "micro_entry_risk_scale"], errors="coerce").fillna(1.0) * float(scale)
        out.loc[mask, "micro_filter_action"] = out.loc[mask, "micro_filter_action"].astype(str) + f"+RESEARCH_SCALE_{str(scale).replace('.', 'p')}"
        meta.update({"changed_signals": changed, "rule": f"Multiply micro_entry_risk_scale by {scale} where mask={mask_name}."})
        return out, meta

    raise ValueError(f"Unknown research variant: {variant}")


def run_variant_backtests(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features, exec_cfg, bull_exec_cfg, v10a_args = _build_v10b_features(args)
    variants = [
        "baseline_v10b",
        "scale_bull_not_aligned_0p25",
        "scale_bull_not_aligned_0p35",
        "scale_bull_not_aligned_0p50",
        "block_bull_not_aligned",
        "block_bull_any_not_aligned",
        "block_bull_contra",
        "scale_all_not_aligned_0p35",
        "scale_all_contra_0p25",
        "block_all_contra",
        "block_low_quality_bottom_quartile",
    ]
    rows = []
    combined_trades = []
    meta_rows = []
    for variant in variants:
        vf, meta = apply_variant(features, variant)
        print(f"Running research variant: {variant} | changed_signals={meta.get('changed_signals')}", flush=True)
        trades, equity = v10b.run_v10b_backtest(
            vf,
            exec_cfg,
            engine_cfgs={"MOMENTUM_V3": exec_cfg, "BEAR_V3_ONLY": exec_cfg, "BULL_RECLAIM_V2": bull_exec_cfg},
            global_risk_scale=v10a_args.global_risk_scale,
            args=v10a_args,
        )
        trades = v10a.attach_engine_to_trades(trades, vf)
        summary = v10a.summarize(trades, equity, exec_cfg.initial_capital)
        tdf = pd.DataFrame(trades)
        if not tdf.empty:
            notes = tdf.get("note", pd.Series(dtype=str)).astype(str)
            summary["structural_stop_exit_count"] = int(notes.eq("STRUCTURAL_STOP").sum())
            summary["range_exit_trade_count"] = int(notes.str.startswith("RANGE_EXIT").sum())
            summary["engine_counts"] = tdf.get("engine", pd.Series(dtype=str)).value_counts().to_dict()
            if args.write_variant_trades:
                tdf["variant"] = variant
                combined_trades.append(tdf)
        row = {"variant": variant, **summary, **meta}
        rows.append(row)
        meta_rows.append(meta)
    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["win_rate", "total_return_pct"], ascending=[False, False])
    trades_df = pd.concat(combined_trades, ignore_index=True, sort=False) if combined_trades else pd.DataFrame()
    meta_df = pd.DataFrame(meta_rows)
    return summary_df, trades_df, meta_df


def promotion_gate(variant_df: pd.DataFrame, baseline_summary: dict[str, Any], gate: GateConfig) -> pd.DataFrame:
    rows = []
    if variant_df.empty:
        rows.append({"variant": "NO_VARIANTS", "promote_to_backtest_version": False, "reason": "--rebuild-features not used or failed."})
        return pd.DataFrame(rows)
    base_row = variant_df.loc[variant_df["variant"].eq("baseline_v10b")]
    if base_row.empty:
        base_metrics = baseline_summary
    else:
        base_metrics = base_row.iloc[0].to_dict()

    b_win = safe_num(base_metrics.get("win_rate"), 0.0)
    b_return = safe_num(base_metrics.get("total_return_pct"), 0.0)
    b_pf = safe_num(base_metrics.get("profit_factor"), 0.0)
    b_dd = safe_num(base_metrics.get("max_drawdown_pct"), float("nan"))
    b_trades = safe_num(base_metrics.get("total_trades"), 0.0)
    for _, row in variant_df.iterrows():
        variant = str(row.get("variant"))
        if variant == "baseline_v10b":
            continue
        win = safe_num(row.get("win_rate"), 0.0)
        ret = safe_num(row.get("total_return_pct"), 0.0)
        pf = safe_num(row.get("profit_factor"), 0.0)
        dd = safe_num(row.get("max_drawdown_pct"), float("nan"))
        trades = safe_num(row.get("total_trades"), 0.0)
        checks = {
            "win_rate_lift_pass": (win - b_win) >= gate.min_win_rate_lift_pp,
            "return_ratio_pass": ret >= b_return * gate.min_return_ratio,
            "profit_factor_ratio_pass": pf >= b_pf * gate.min_profit_factor_ratio if math.isfinite(b_pf) else False,
            "drawdown_pass": dd <= b_dd * gate.max_drawdown_ratio if math.isfinite(b_dd) else False,
            "trade_count_pass": trades >= b_trades * gate.min_trade_ratio,
        }
        passed = bool(all(checks.values()))
        reason_parts = [k.replace("_pass", "") for k, v in checks.items() if not v]
        rows.append({
            "variant": variant,
            "promote_to_backtest_version": passed,
            "reason": "PASS_RESEARCH_GATE" if passed else "FAIL_" + ",".join(reason_parts),
            "win_rate": win,
            "baseline_win_rate": b_win,
            "win_rate_lift_pp": win - b_win,
            "total_return_pct": ret,
            "baseline_total_return_pct": b_return,
            "return_ratio": ret / b_return if abs(b_return) > 1e-12 else float("nan"),
            "profit_factor": pf,
            "baseline_profit_factor": b_pf,
            "profit_factor_ratio": pf / b_pf if abs(b_pf) > 1e-12 else float("nan"),
            "max_drawdown_pct": dd,
            "baseline_max_drawdown_pct": b_dd,
            "drawdown_ratio": dd / b_dd if math.isfinite(b_dd) and abs(b_dd) > 1e-12 else float("nan"),
            "total_trades": trades,
            "baseline_total_trades": b_trades,
            **checks,
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["promote_to_backtest_version", "win_rate_lift_pp", "return_ratio"], ascending=[False, False, False])
    return out


def write_markdown_brief(
    out_dir: Path,
    baseline: pd.DataFrame,
    engine_side: pd.DataFrame,
    exit_micro_units: pd.DataFrame,
    signal_edge: pd.DataFrame,
    bucket_edge: pd.DataFrame,
    salvage: pd.DataFrame,
    variants: pd.DataFrame,
    gate_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# V10B Win-Rate Full Research Brief")
    lines.append("")
    lines.append("## Policy")
    lines.append("- Research-only outputs. Do not promote any change to `backtest/lf` or AetherEdge unless the gate passes and a separate formal verification confirms it.")
    lines.append("- Signal forward returns use future labels for diagnosis only, not as production rules.")
    lines.append("- Candidate variants, when present, are in-process experiments and do not create a new strategy version.")
    lines.append("")
    lines.append("## Baseline highlights")
    if not baseline.empty:
        wanted = ["total_trades", "win_rate", "total_return_pct", "profit_factor", "max_drawdown_pct", "mfe_ge_1r_ended_loss_from_trades", "mfe_ge_2r_ended_loss_from_trades", "add_on_loser_units_ge_2"]
        b = baseline.set_index("metric")["value"].to_dict()
        for key in wanted:
            if key in b:
                lines.append(f"- {key}: {b[key]}")
    lines.append("")
    lines.append("## First reads")
    if not engine_side.empty:
        worst = engine_side.sort_values("return_pct_avg", ascending=True).head(5)
        lines.append("### Weakest engine/side buckets by average return")
        lines.append(worst.to_markdown(index=False))
        lines.append("")
    if not exit_micro_units.empty:
        loss_buckets = exit_micro_units.sort_values("mfe_ge_1r_loss_count", ascending=False).head(8)
        lines.append("### Buckets with most MFE>=1R then loss")
        lines.append(loss_buckets.to_markdown(index=False))
        lines.append("")
    if not signal_edge.empty:
        selected_6 = signal_edge[(signal_edge["signal_family"] == "selected") & (signal_edge["horizon_bars"] == 6)].copy()
        if not selected_6.empty:
            lines.append("### Selected signal 6-bar forward edge")
            lines.append(selected_6.to_markdown(index=False))
            lines.append("")
    if not bucket_edge.empty:
        poor = bucket_edge[(bucket_edge["signal_family"] == "selected")].sort_values("avg_side_return_pct", ascending=True).head(12)
        if not poor.empty:
            lines.append("### Weak selected-signal buckets by forward label")
            lines.append(poor.to_markdown(index=False))
            lines.append("")
    if not salvage.empty:
        lines.append("## MFE salvage candidates")
        lines.append(f"- Candidate trades: {len(salvage)}")
        lines.append("- These are the first place to study protective exits or add-on-aware breakeven. Do not use them to tune one-off exits.")
        lines.append(salvage.head(12).to_markdown(index=False))
        lines.append("")
    if not variants.empty:
        keep_cols = [c for c in ["variant", "total_trades", "win_rate", "total_return_pct", "profit_factor", "max_drawdown_pct", "changed_signals"] if c in variants.columns]
        lines.append("## Research-only counterfactual variants")
        lines.append(variants[keep_cols].head(20).to_markdown(index=False))
        lines.append("")
    if not gate_df.empty:
        lines.append("## Promotion gate")
        keep_cols = [c for c in ["variant", "promote_to_backtest_version", "reason", "win_rate_lift_pp", "return_ratio", "profit_factor_ratio", "drawdown_ratio", "total_trades"] if c in gate_df.columns]
        lines.append(gate_df[keep_cols].to_markdown(index=False))
        lines.append("")
    lines.append("## Next action")
    lines.append("- If no variant passes the gate, keep V10B unchanged and use the salvage table to design the next research-only executor experiment.")
    lines.append("- If a variant passes, rerun formal robustness: yearly reset, fee/slippage stress, delayed execution stress, top-winner dependency, and V10B vs candidate trade diff.")
    (out_dir / "10_research_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = project_path(args.out_dir)
    ensure_dir(out_dir)
    report = load_v10b_report(project_path(args.v10b_dir))
    summary = report["summary"]
    trades = report["trades"]
    audit = report["audit"]

    if not summary:
        raise FileNotFoundError(f"Missing V10B summary under {project_path(args.v10b_dir)}")
    if trades.empty:
        raise FileNotFoundError(f"Missing V10B trades under {project_path(args.v10b_dir)}")
    if audit.empty:
        raise FileNotFoundError(f"Missing V10B signal_audit under {project_path(args.v10b_dir)}")

    baseline_df = build_baseline_summary(summary, trades)
    engine_side_df, exit_micro_units_df = group_trade_breakdown(trades)
    yearly_df, monthly_df = yearly_monthly_breakdown(trades)
    signal_edge_df, bucket_edge_df = signal_forward_edge(audit)
    salvage_df = mfe_salvage_table(trades)

    variant_df = pd.DataFrame()
    variant_trades_df = pd.DataFrame()
    variant_meta_df = pd.DataFrame()
    if args.rebuild_features:
        variant_df, variant_trades_df, variant_meta_df = run_variant_backtests(args)

    gate_cfg = GateConfig(
        min_win_rate_lift_pp=float(args.min_win_rate_lift_pp),
        min_return_ratio=float(args.min_return_ratio),
        min_profit_factor_ratio=float(args.min_profit_factor_ratio),
        max_drawdown_ratio=float(args.max_drawdown_ratio),
        min_trade_ratio=float(args.min_trade_ratio),
        max_top3_return_share=float(args.max_top3_return_share),
    )
    gate_df = promotion_gate(variant_df, summary, gate_cfg)

    baseline_df.to_csv(out_dir / "01_baseline_summary.csv", index=False)
    engine_side_df.to_csv(out_dir / "02_trade_breakdown_engine_side.csv", index=False)
    exit_micro_units_df.to_csv(out_dir / "03_trade_breakdown_exit_micro_units.csv", index=False)
    yearly_df.to_csv(out_dir / "04a_yearly_trade_breakdown.csv", index=False)
    monthly_df.to_csv(out_dir / "04b_monthly_trade_breakdown.csv", index=False)
    signal_edge_df.to_csv(out_dir / "05_signal_forward_edge.csv", index=False)
    bucket_edge_df.to_csv(out_dir / "06_signal_bucket_edge.csv", index=False)
    salvage_df.to_csv(out_dir / "07_trade_mfe_salvage_candidates.csv", index=False)
    variant_df.to_csv(out_dir / "08_counterfactual_variant_summary.csv", index=False)
    if not variant_trades_df.empty:
        variant_trades_df.to_csv(out_dir / "08b_counterfactual_variant_trades.csv", index=False)
    variant_meta_df.to_csv(out_dir / "08c_counterfactual_variant_meta.csv", index=False)
    gate_df.to_csv(out_dir / "09_promotion_gate.csv", index=False)

    write_markdown_brief(
        out_dir,
        baseline_df,
        engine_side_df,
        exit_micro_units_df,
        signal_edge_df,
        bucket_edge_df,
        salvage_df,
        variant_df,
        gate_df,
    )

    meta = {
        "generated_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "research/v10b_winrate_full_research.py",
        "mode": "rebuild_features" if args.rebuild_features else "from_existing_reports",
        "v10b_dir": str(project_path(args.v10b_dir)),
        "out_dir": str(out_dir),
        "gate_config": gate_cfg.__dict__,
        "notes": [
            "Research-only. No official strategy/backtest version is modified.",
            "Signal forward-return tables use future labels for diagnosis only.",
            "Promotion requires separate formal robustness verification after this script.",
        ],
    }
    (out_dir / "99_research_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"V10B win-rate research outputs: {out_dir.resolve()}")
    if not gate_df.empty:
        pass_count = int(gate_df.get("promote_to_backtest_version", pd.Series(dtype=bool)).astype("boolean").fillna(False).sum())
        print(f"Promotion gate pass count: {pass_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
