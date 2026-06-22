#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LF strategy risk scaling lab.

This is a research tool, not a strategy. It reads an already-generated trade CSV
and simulates how the same realized trade-return sequence behaves under different
risk multipliers.

The goal is to answer: if entry/exit edge stays the same but risk per trade is
scaled up/down, what happens to CAGR, max drawdown, yearly stability and ruin risk?
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCRIPT_NAME = "LFRiskScalingLabV1"


@dataclass
class Config:
    trades_csv: Path
    out_dir: Path = Path("data/reports/research/lf_risk_scaling_lab_v1")
    initial_capital: float = 1000.0
    multipliers: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
    base_risk_per_trade: float = 0.002
    extra_cost_pct_per_trade: float = 0.0
    monte_carlo_runs: int = 1000
    monte_carlo_seed: int = 42
    ruin_drawdown_pct: float = 60.0
    block_size: int = 1


RETURN_CANDIDATES = [
    "return_pct",
    "net_return_pct",
    "pnl_pct",
    "pnl_return_pct",
    "account_return_pct",
    "trade_return_pct",
]
PNL_CANDIDATES = ["pnl", "net_pnl", "pnl_usdt", "profit", "net_profit", "realized_pnl"]
EQUITY_BEFORE_CANDIDATES = ["equity_before", "capital_before", "balance_before", "start_capital"]
R_CANDIDATES = ["r", "r_multiple", "net_r", "R", "trade_r"]
EXIT_TIME_CANDIDATES = ["exit_time", "exit_ts", "close_time", "closed_at", "exit_datetime"]
ENTRY_TIME_CANDIDATES = ["entry_time", "entry_ts", "open_time", "opened_at", "entry_datetime"]


def _find_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def _parse_multipliers(text: str | None) -> tuple[float, ...]:
    if not text:
        return Config(Path("dummy")).multipliers
    vals = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("--multipliers cannot be empty")
    return tuple(vals)


def _detect_time_col(df: pd.DataFrame) -> str | None:
    col = _find_col(df, EXIT_TIME_CANDIDATES) or _find_col(df, ENTRY_TIME_CANDIDATES)
    return col


def load_trade_returns(cfg: Config) -> pd.DataFrame:
    if not cfg.trades_csv.exists():
        raise FileNotFoundError(f"trades csv not found: {cfg.trades_csv}")
    raw = pd.read_csv(cfg.trades_csv)
    if raw.empty:
        raise RuntimeError("trades csv is empty")

    time_col = _detect_time_col(raw)
    if time_col is None:
        raise RuntimeError(
            "Cannot find a trade time column. Expected one of "
            f"{EXIT_TIME_CANDIDATES + ENTRY_TIME_CANDIDATES}."
        )
    raw["trade_time"] = pd.to_datetime(raw[time_col], errors="coerce")
    raw = raw[raw["trade_time"].notna()].copy()
    raw = raw.sort_values("trade_time").reset_index(drop=True)
    if raw.empty:
        raise RuntimeError("No rows have parseable trade_time")

    source = ""
    ret_col = _find_col(raw, RETURN_CANDIDATES)
    if ret_col is not None:
        base_return = pd.to_numeric(raw[ret_col], errors="coerce") / 100.0
        source = f"{ret_col}/100"
    else:
        pnl_col = _find_col(raw, PNL_CANDIDATES)
        eq_col = _find_col(raw, EQUITY_BEFORE_CANDIDATES)
        if pnl_col is not None and eq_col is not None:
            pnl = pd.to_numeric(raw[pnl_col], errors="coerce")
            eq = pd.to_numeric(raw[eq_col], errors="coerce").replace(0, np.nan)
            base_return = pnl / eq
            source = f"{pnl_col}/{eq_col}"
        else:
            r_col = _find_col(raw, R_CANDIDATES)
            if r_col is None:
                raise RuntimeError(
                    "Cannot infer account return. Provide a CSV containing one of: "
                    f"return columns={RETURN_CANDIDATES}, or pnl+equity_before, or R columns={R_CANDIDATES}."
                )
            base_return = pd.to_numeric(raw[r_col], errors="coerce") * float(cfg.base_risk_per_trade)
            source = f"{r_col}*base_risk_per_trade({cfg.base_risk_per_trade})"

    out = pd.DataFrame(
        {
            "trade_index": np.arange(len(raw), dtype=int),
            "trade_time": raw["trade_time"].to_numpy(),
            "base_return": base_return.to_numpy(dtype=float),
        }
    )
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["base_return"]).copy()
    if out.empty:
        raise RuntimeError("No valid base_return values")
    out["year"] = pd.to_datetime(out["trade_time"]).dt.year
    out["month"] = pd.to_datetime(out["trade_time"]).dt.to_period("M").astype(str)
    out["return_source"] = source
    return out


def _equity_curve(returns: np.ndarray, initial: float) -> np.ndarray:
    eq = np.empty(len(returns) + 1, dtype=float)
    eq[0] = float(initial)
    for i, r in enumerate(returns, start=1):
        if eq[i - 1] <= 0:
            eq[i] = 0.0
            continue
        eq[i] = eq[i - 1] * max(0.0, 1.0 + float(r))
    return eq


def _max_drawdown_pct(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return np.nan
    peak = np.maximum.accumulate(equity)
    dd = equity / np.where(peak == 0, np.nan, peak) - 1.0
    return float(np.nanmin(dd) * 100.0)


def _profit_factor(returns: np.ndarray) -> float:
    wins = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else np.nan
    return float(wins / losses)


def _cagr_pct(start_time: pd.Timestamp, end_time: pd.Timestamp, initial: float, final: float) -> float:
    days = max(1.0, (end_time - start_time).total_seconds() / 86400.0)
    years = days / 365.25
    if initial <= 0 or final <= 0 or years <= 0:
        return -100.0 if final <= 0 else np.nan
    return float(((final / initial) ** (1.0 / years) - 1.0) * 100.0)


def summarize_scaled(trades: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    base = trades["base_return"].to_numpy(dtype=float)
    times = pd.to_datetime(trades["trade_time"])
    start_time = times.iloc[0]
    end_time = times.iloc[-1]

    for m in cfg.multipliers:
        scaled = base * float(m) - float(cfg.extra_cost_pct_per_trade) / 100.0
        equity = _equity_curve(scaled, cfg.initial_capital)
        final = float(equity[-1])
        total_return_pct = (final / cfg.initial_capital - 1.0) * 100.0
        max_dd_pct = _max_drawdown_pct(equity)
        ruin = final <= 0 or max_dd_pct <= -abs(float(cfg.ruin_drawdown_pct))
        rows.append(
            {
                "multiplier": float(m),
                "trades": int(len(scaled)),
                "final_capital": final,
                "total_return_pct": total_return_pct,
                "cagr_pct": _cagr_pct(start_time, end_time, cfg.initial_capital, final),
                "max_drawdown_pct": max_dd_pct,
                "win_rate_pct": float((scaled > 0).mean() * 100.0),
                "avg_trade_return_pct": float(np.mean(scaled) * 100.0),
                "median_trade_return_pct": float(np.median(scaled) * 100.0),
                "profit_factor": _profit_factor(scaled),
                "worst_trade_pct": float(np.min(scaled) * 100.0),
                "best_trade_pct": float(np.max(scaled) * 100.0),
                "ruin_or_dd_breach": bool(ruin),
                "ruin_drawdown_threshold_pct": -abs(float(cfg.ruin_drawdown_pct)),
            }
        )
        tmp = trades.copy()
        tmp["scaled_return"] = scaled
        tmp["scaled_equity_before"] = equity[:-1]
        tmp["scaled_equity_after"] = equity[1:]
        for year, g in tmp.groupby("year"):
            yr_returns = g["scaled_return"].to_numpy(dtype=float)
            yr_eq = _equity_curve(yr_returns, float(g["scaled_equity_before"].iloc[0]))
            yearly_rows.append(
                {
                    "multiplier": float(m),
                    "year": int(year),
                    "trades": int(len(g)),
                    "start_capital": float(g["scaled_equity_before"].iloc[0]),
                    "end_capital": float(g["scaled_equity_after"].iloc[-1]),
                    "year_return_pct": float((g["scaled_equity_after"].iloc[-1] / g["scaled_equity_before"].iloc[0] - 1.0) * 100.0),
                    "year_max_drawdown_pct": _max_drawdown_pct(yr_eq),
                    "year_profit_factor": _profit_factor(yr_returns),
                    "year_win_rate_pct": float((yr_returns > 0).mean() * 100.0),
                }
            )
        for month, g in tmp.groupby("month"):
            mret = float((g["scaled_equity_after"].iloc[-1] / g["scaled_equity_before"].iloc[0] - 1.0) * 100.0)
            monthly_rows.append(
                {
                    "multiplier": float(m),
                    "month": month,
                    "trades": int(len(g)),
                    "month_return_pct": mret,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(yearly_rows), pd.DataFrame(monthly_rows)


def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if block_size <= 1:
        return rng.integers(0, n, size=n)
    starts = rng.integers(0, n, size=math.ceil(n / block_size))
    idxs: list[int] = []
    for st in starts:
        for k in range(block_size):
            idxs.append((int(st) + k) % n)
            if len(idxs) >= n:
                break
        if len(idxs) >= n:
            break
    return np.array(idxs, dtype=int)


def monte_carlo(trades: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if cfg.monte_carlo_runs <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(int(cfg.monte_carlo_seed))
    base = trades["base_return"].to_numpy(dtype=float)
    n = len(base)
    rows: list[dict[str, Any]] = []
    for m in cfg.multipliers:
        finals: list[float] = []
        dds: list[float] = []
        ruin_count = 0
        for _ in range(int(cfg.monte_carlo_runs)):
            idx = _block_bootstrap_indices(n, int(cfg.block_size), rng)
            sampled = base[idx] * float(m) - float(cfg.extra_cost_pct_per_trade) / 100.0
            eq = _equity_curve(sampled, cfg.initial_capital)
            final_return_pct = (float(eq[-1]) / cfg.initial_capital - 1.0) * 100.0
            dd = _max_drawdown_pct(eq)
            finals.append(final_return_pct)
            dds.append(dd)
            if eq[-1] <= 0 or dd <= -abs(float(cfg.ruin_drawdown_pct)):
                ruin_count += 1
        rows.append(
            {
                "multiplier": float(m),
                "runs": int(cfg.monte_carlo_runs),
                "final_return_p05": float(np.percentile(finals, 5)),
                "final_return_p25": float(np.percentile(finals, 25)),
                "final_return_p50": float(np.percentile(finals, 50)),
                "final_return_p75": float(np.percentile(finals, 75)),
                "final_return_p95": float(np.percentile(finals, 95)),
                "max_dd_p05": float(np.percentile(dds, 5)),
                "max_dd_p25": float(np.percentile(dds, 25)),
                "max_dd_p50": float(np.percentile(dds, 50)),
                "max_dd_p75": float(np.percentile(dds, 75)),
                "max_dd_p95": float(np.percentile(dds, 95)),
                "ruin_or_dd_breach_pct": float(ruin_count / int(cfg.monte_carlo_runs) * 100.0),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=SCRIPT_NAME, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--trades-csv", required=True, help="Path to a strategy trade CSV.")
    p.add_argument("--out-dir", default="data/reports/research/lf_risk_scaling_lab_v1")
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--multipliers", default="0.5,1,1.5,2,2.5,3,4,5")
    p.add_argument("--base-risk-per-trade", type=float, default=0.002, help="Used only when CSV has R multiple but no return_pct.")
    p.add_argument("--extra-cost-pct-per-trade", type=float, default=0.0, help="Extra penalty per trade in account percent, applied after scaling.")
    p.add_argument("--monte-carlo-runs", type=int, default=1000)
    p.add_argument("--monte-carlo-seed", type=int, default=42)
    p.add_argument("--ruin-drawdown-pct", type=float, default=60.0)
    p.add_argument("--block-size", type=int, default=1, help="1=random trade bootstrap; >1=block bootstrap preserving short trade clusters.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        trades_csv=Path(args.trades_csv),
        out_dir=Path(args.out_dir),
        initial_capital=float(args.initial_capital),
        multipliers=_parse_multipliers(args.multipliers),
        base_risk_per_trade=float(args.base_risk_per_trade),
        extra_cost_pct_per_trade=float(args.extra_cost_pct_per_trade),
        monte_carlo_runs=int(args.monte_carlo_runs),
        monte_carlo_seed=int(args.monte_carlo_seed),
        ruin_drawdown_pct=float(args.ruin_drawdown_pct),
        block_size=int(args.block_size),
    )
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{SCRIPT_NAME}] cfg={json.dumps({**asdict(cfg), 'trades_csv': str(cfg.trades_csv), 'out_dir': str(cfg.out_dir)}, ensure_ascii=False, default=str)}", flush=True)

    trades = load_trade_returns(cfg)
    summary, yearly, monthly = summarize_scaled(trades, cfg)
    mc = monte_carlo(trades, cfg)

    trades.to_csv(cfg.out_dir / "lf_risk_scaling_lab_v1_trade_returns.csv", index=False)
    summary.to_csv(cfg.out_dir / "lf_risk_scaling_lab_v1_summary.csv", index=False)
    yearly.to_csv(cfg.out_dir / "lf_risk_scaling_lab_v1_yearly.csv", index=False)
    monthly.to_csv(cfg.out_dir / "lf_risk_scaling_lab_v1_monthly.csv", index=False)
    mc.to_csv(cfg.out_dir / "lf_risk_scaling_lab_v1_monte_carlo.csv", index=False)
    (cfg.out_dir / "lf_risk_scaling_lab_v1_config.json").write_text(
        json.dumps({**asdict(cfg), "trades_csv": str(cfg.trades_csv), "out_dir": str(cfg.out_dir)}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n=== Risk Scaling Summary ===")
    print(summary.to_string(index=False))
    if not mc.empty:
        print("\n=== Monte Carlo Summary ===")
        print(mc.to_string(index=False))
    print(f"\nOutputs: {cfg.out_dir}")


if __name__ == "__main__":
    main()
