#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V8 pressure test tool for ETH LF Portfolio Micro Confirmation Scaled.

Purpose:
    Compare selected global-risk-scale candidates under stress tests:
    - base full sample
    - remove top 1 / top 3 winning trades (post-analysis)
    - no 2026 / 2023-2025 real rerun
    - fee 2x real rerun
    - slippage 2x real rerun
    - Monte Carlo bootstrap on realized account returns
    - yearly breakdown from equity curve

Notes:
    This is a research tool, not a strategy. It does not modify V6/V7B/V8.
    For real fee/slippage/no-2026 tests, it invokes the V8 backtest script with subprocess.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
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

SCRIPT_NAME = "V8PressureTest"
STRATEGY_SCRIPT = Path(PROJECT_ROOT) / "backtest" / "lf" / "eth_lf_portfolio_v8_micro_confirm_scaled_backtest.py"
DEFAULT_BASE_RESULTS_ROOT = Path(PROJECT_ROOT) / "data" / "reports" / "lf" / "eth_lf_portfolio_v8_micro_confirm_scaled"
DEFAULT_OUT_DIR = Path(PROJECT_ROOT) / "data" / "reports" / "research" / "v8_pressure_test"


@dataclass
class Config:
    scales: tuple[float, ...]
    start_date: str
    end_date: str
    warmup_start_date: str
    symbol: str = "ETH-USDT-SWAP"
    fee_rate: float = 0.00055
    slippage_pct: float = 0.0002
    initial_capital: float = 1000.0
    base_results_root: Path = DEFAULT_BASE_RESULTS_ROOT
    out_dir: Path = DEFAULT_OUT_DIR
    monte_carlo_runs: int = 2000
    monte_carlo_seed: int = 42
    mc_block_size: int = 1
    skip_reruns: bool = False
    force_rerun_base: bool = False


def _parse_scales(text: str) -> tuple[float, ...]:
    vals: list[float] = []
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    if not vals:
        raise ValueError("--scales cannot be empty")
    return tuple(vals)


def _tag_float(value: float) -> str:
    return f"{float(value):g}".replace(".", "p")


def _base_dir_for_scale(base_root: Path, scale: float) -> Path:
    return base_root / f"soft_05_gs_{_tag_float(scale)}"


def _summary_path(run_dir: Path) -> Path:
    return run_dir / "summary.json"


def _trades_path(run_dir: Path) -> Path:
    return run_dir / "trades.csv"


def _equity_path(run_dir: Path) -> Path:
    return run_dir / "equity.csv"


def _has_complete_run(run_dir: Path) -> bool:
    return _summary_path(run_dir).exists() and _trades_path(run_dir).exists()


def _run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[RUN] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd), text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with code={proc.returncode}: {' '.join(cmd)}")


def run_v8_backtest(cfg: Config, scale: float, out_dir: Path, *, end_date: str | None = None, fee_rate: float | None = None, slippage_pct: float | None = None, force: bool = False) -> Path:
    if _has_complete_run(out_dir) and not force:
        print(f"[SKIP] Existing run found: {out_dir}", flush=True)
        return out_dir
    if cfg.skip_reruns:
        raise FileNotFoundError(f"missing run and --skip-reruns is set: {out_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(STRATEGY_SCRIPT),
        "--symbol", cfg.symbol,
        "--start-date", cfg.start_date,
        "--end-date", end_date or cfg.end_date,
        "--warmup-start-date", cfg.warmup_start_date,
        "--global-risk-scale", f"{scale:g}",
        "--fee-rate", f"{fee_rate if fee_rate is not None else cfg.fee_rate:g}",
        "--slippage-pct", f"{slippage_pct if slippage_pct is not None else cfg.slippage_pct:g}",
        "--out-dir", str(out_dir),
    ]
    _run_cmd(cmd, Path(PROJECT_ROOT))
    if not _has_complete_run(out_dir):
        raise RuntimeError(f"V8 run completed but outputs are missing: {out_dir}")
    return out_dir


def read_summary(run_dir: Path, scale: float, test_name: str, source: str) -> dict[str, Any]:
    data = json.loads(_summary_path(run_dir).read_text(encoding="utf-8"))
    return {
        "scale": float(scale),
        "test": test_name,
        "source": source,
        "total_return_pct": float(data.get("total_return_pct", np.nan)),
        "max_drawdown_pct_abs": abs(float(data.get("max_drawdown_pct", np.nan))),
        "profit_factor": float(data.get("profit_factor", np.nan)),
        "win_rate_pct": float(data.get("win_rate", data.get("win_rate_pct", np.nan))),
        "total_trades": int(data.get("total_trades", 0)),
        "final_capital": float(data.get("final_capital", np.nan)),
        "path": str(run_dir),
    }


def load_trade_returns(trades_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(trades_csv)
    if df.empty:
        raise RuntimeError(f"trades csv is empty: {trades_csv}")
    if "exit_time" in df.columns:
        df["trade_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    elif "entry_time" in df.columns:
        df["trade_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    else:
        raise RuntimeError(f"cannot find entry_time/exit_time in {trades_csv}")
    df = df[df["trade_time"].notna()].sort_values("trade_time").reset_index(drop=True)

    if "pnl" in df.columns and "capital" in df.columns:
        pnl = pd.to_numeric(df["pnl"], errors="coerce")
        equity_after = pd.to_numeric(df["capital"], errors="coerce")
        equity_before = (equity_after - pnl).replace(0, np.nan)
        df["account_return"] = pnl / equity_before
    elif "return_pct" in df.columns:
        ret = pd.to_numeric(df["return_pct"], errors="coerce")
        # In this project, return_pct is already decimal account return in trades.csv.
        df["account_return"] = ret
    else:
        raise RuntimeError(f"cannot infer account return from {trades_csv}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["account_return"]).copy()
    if "pnl" not in df.columns:
        df["pnl"] = df["account_return"]
    df["year"] = df["trade_time"].dt.year
    return df


def equity_curve_from_returns(returns: np.ndarray, initial_capital: float) -> np.ndarray:
    eq = np.empty(len(returns) + 1, dtype=float)
    eq[0] = float(initial_capital)
    for i, ret in enumerate(returns, start=1):
        prev = eq[i - 1]
        if prev <= 0:
            eq[i] = 0.0
        else:
            eq[i] = prev * max(0.0, 1.0 + float(ret))
    return eq


def max_drawdown_pct_abs(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    dd = equity / np.where(peak == 0, np.nan, peak) - 1.0
    return abs(float(np.nanmin(dd) * 100.0))


def profit_factor_from_returns(returns: np.ndarray) -> float:
    wins = float(returns[returns > 0].sum())
    losses = float(-returns[returns < 0].sum())
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return wins / losses


def summarize_returns(returns: np.ndarray, scale: float, test_name: str, source: str, initial_capital: float, path: str = "") -> dict[str, Any]:
    eq = equity_curve_from_returns(returns, initial_capital)
    return {
        "scale": float(scale),
        "test": test_name,
        "source": source,
        "total_return_pct": float((eq[-1] / initial_capital - 1.0) * 100.0),
        "max_drawdown_pct_abs": max_drawdown_pct_abs(eq),
        "profit_factor": profit_factor_from_returns(returns),
        "win_rate_pct": float((returns > 0).mean() * 100.0) if len(returns) else float("nan"),
        "total_trades": int(len(returns)),
        "final_capital": float(eq[-1]),
        "path": path,
    }


def remove_top_winners_summary(trades: pd.DataFrame, scale: float, n: int, initial_capital: float) -> dict[str, Any]:
    if n <= 0:
        raise ValueError("n must be positive")
    df = trades.copy()
    top_idx = df.sort_values("pnl", ascending=False).head(n).index
    stressed = df.drop(index=top_idx).sort_values("trade_time")
    returns = stressed["account_return"].to_numpy(dtype=float)
    return summarize_returns(returns, scale, f"remove_top_{n}_winner", "post_analysis_from_base_trades", initial_capital)


def _block_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if block_size <= 1:
        return rng.integers(0, n, size=n)
    starts = rng.integers(0, n, size=math.ceil(n / block_size))
    out: list[int] = []
    for start in starts:
        for offset in range(block_size):
            out.append((int(start) + offset) % n)
            if len(out) >= n:
                break
        if len(out) >= n:
            break
    return np.array(out, dtype=int)


def monte_carlo(trades: pd.DataFrame, scale: float, cfg: Config) -> dict[str, Any]:
    returns = trades["account_return"].to_numpy(dtype=float)
    rng = np.random.default_rng(int(cfg.monte_carlo_seed))
    finals: list[float] = []
    dds: list[float] = []
    pfs: list[float] = []
    n = len(returns)
    for _ in range(int(cfg.monte_carlo_runs)):
        idx = _block_bootstrap_indices(n, int(cfg.mc_block_size), rng)
        sampled = returns[idx]
        eq = equity_curve_from_returns(sampled, cfg.initial_capital)
        finals.append((float(eq[-1]) / cfg.initial_capital - 1.0) * 100.0)
        dds.append(max_drawdown_pct_abs(eq))
        pfs.append(profit_factor_from_returns(sampled))
    return {
        "scale": float(scale),
        "runs": int(cfg.monte_carlo_runs),
        "block_size": int(cfg.mc_block_size),
        "final_return_p05": float(np.percentile(finals, 5)),
        "final_return_p25": float(np.percentile(finals, 25)),
        "final_return_p50": float(np.percentile(finals, 50)),
        "final_return_p75": float(np.percentile(finals, 75)),
        "final_return_p95": float(np.percentile(finals, 95)),
        "max_dd_p50": float(np.percentile(dds, 50)),
        "max_dd_p75": float(np.percentile(dds, 75)),
        "max_dd_p95": float(np.percentile(dds, 95)),
        "profit_factor_p50": float(np.percentile(pfs, 50)),
    }


def yearly_from_trades(trades: pd.DataFrame, scale: float, test_name: str, initial_capital: float) -> pd.DataFrame:
    """Build yearly breakdown from realized trade returns.

    This avoids a known issue where some backtest equity.csv files may not include
    the final force-close capital after the last trade.
    """
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    equity = float(initial_capital)
    for year, g in trades.sort_values("trade_time").groupby("year", sort=True):
        year_start = equity
        capitals = [equity]
        returns = g["account_return"].to_numpy(dtype=float)
        for ret in returns:
            equity = equity * max(0.0, 1.0 + float(ret))
            capitals.append(equity)
        cap_arr = np.array(capitals, dtype=float)
        rows.append({
            "scale": float(scale),
            "test": test_name,
            "year": int(year),
            "trades": int(len(g)),
            "start_capital": float(year_start),
            "end_capital": float(equity),
            "year_return_pct": float((equity / year_start - 1.0) * 100.0) if year_start > 0 else float("nan"),
            "year_max_drawdown_pct_abs": max_drawdown_pct_abs(cap_arr),
            "year_profit_factor": profit_factor_from_returns(returns),
            "year_win_rate_pct": float((returns > 0).mean() * 100.0) if len(returns) else float("nan"),
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=SCRIPT_NAME, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--scales", default="1.15,1.20,1.30", help="Comma-separated global-risk-scale candidates.")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--base-results-root", default=str(DEFAULT_BASE_RESULTS_ROOT), help="Existing V8 result root. Base runs are reused when available.")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    p.add_argument("--monte-carlo-runs", type=int, default=2000)
    p.add_argument("--monte-carlo-seed", type=int, default=42)
    p.add_argument("--mc-block-size", type=int, default=1)
    p.add_argument("--skip-reruns", action="store_true", help="Only analyze existing base runs; skip real rerun stress tests.")
    p.add_argument("--force-rerun-base", action="store_true", help="Rerun base scenarios even if existing outputs are found.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = Config(
        scales=_parse_scales(args.scales),
        start_date=args.start_date,
        end_date=args.end_date,
        warmup_start_date=args.warmup_start_date,
        symbol=args.symbol,
        fee_rate=float(args.fee_rate),
        slippage_pct=float(args.slippage_pct),
        initial_capital=float(args.initial_capital),
        base_results_root=Path(args.base_results_root),
        out_dir=Path(args.out_dir),
        monte_carlo_runs=int(args.monte_carlo_runs),
        monte_carlo_seed=int(args.monte_carlo_seed),
        mc_block_size=int(args.mc_block_size),
        skip_reruns=bool(args.skip_reruns),
        force_rerun_base=bool(args.force_rerun_base),
    )
    if not STRATEGY_SCRIPT.exists():
        raise FileNotFoundError(f"V8 strategy script not found: {STRATEGY_SCRIPT}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = cfg.out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    (cfg.out_dir / "v8_pressure_test_config.json").write_text(
        json.dumps({**asdict(cfg), "base_results_root": str(cfg.base_results_root), "out_dir": str(cfg.out_dir)}, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    summary_rows: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []
    mc_rows: list[dict[str, Any]] = []

    for scale in cfg.scales:
        tag = _tag_float(scale)
        base_dir = _base_dir_for_scale(cfg.base_results_root, scale)
        if not _has_complete_run(base_dir) or cfg.force_rerun_base:
            base_dir = run_v8_backtest(cfg, scale, base_dir, force=cfg.force_rerun_base)
        else:
            print(f"[BASE] Reusing existing base run: {base_dir}", flush=True)

        summary_rows.append(read_summary(base_dir, scale, "base", "real_backtest"))
        trades = load_trade_returns(_trades_path(base_dir))
        yearly_frames.append(yearly_from_trades(trades, scale, "base", cfg.initial_capital))
        summary_rows.append(remove_top_winners_summary(trades, scale, 1, cfg.initial_capital))
        summary_rows.append(remove_top_winners_summary(trades, scale, 3, cfg.initial_capital))
        mc_rows.append(monte_carlo(trades, scale, cfg))

        if not cfg.skip_reruns:
            fee2_dir = runs_dir / f"fee_2x_gs_{tag}"
            run_v8_backtest(cfg, scale, fee2_dir, fee_rate=cfg.fee_rate * 2.0)
            summary_rows.append(read_summary(fee2_dir, scale, "fee_2x", "real_backtest"))
            yearly_frames.append(yearly_from_trades(load_trade_returns(_trades_path(fee2_dir)), scale, "fee_2x", cfg.initial_capital))

            slip2_dir = runs_dir / f"slippage_2x_gs_{tag}"
            run_v8_backtest(cfg, scale, slip2_dir, slippage_pct=cfg.slippage_pct * 2.0)
            summary_rows.append(read_summary(slip2_dir, scale, "slippage_2x", "real_backtest"))
            yearly_frames.append(yearly_from_trades(load_trade_returns(_trades_path(slip2_dir)), scale, "slippage_2x", cfg.initial_capital))

            no2026_dir = runs_dir / f"no_2026_gs_{tag}"
            run_v8_backtest(cfg, scale, no2026_dir, end_date="2025-12-31")
            summary_rows.append(read_summary(no2026_dir, scale, "no_2026_2023_2025", "real_backtest"))
            yearly_frames.append(yearly_from_trades(load_trade_returns(_trades_path(no2026_dir)), scale, "no_2026_2023_2025", cfg.initial_capital))
        else:
            print(f"[SKIP] Real rerun stress tests skipped for scale={scale:g}", flush=True)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["test", "scale"]).reset_index(drop=True)
    summary["return_to_dd_ratio"] = summary["total_return_pct"] / summary["max_drawdown_pct_abs"].replace(0, np.nan)
    summary.to_csv(cfg.out_dir / "v8_pressure_test_summary.csv", index=False)

    yearly = pd.concat([x for x in yearly_frames if not x.empty], ignore_index=True) if yearly_frames else pd.DataFrame()
    yearly.to_csv(cfg.out_dir / "v8_pressure_test_yearly.csv", index=False)

    mc = pd.DataFrame(mc_rows).sort_values("scale").reset_index(drop=True)
    mc.to_csv(cfg.out_dir / "v8_pressure_test_monte_carlo.csv", index=False)

    print("\n=== V8 Pressure Test Summary ===")
    cols = ["scale", "test", "total_return_pct", "max_drawdown_pct_abs", "profit_factor", "total_trades", "return_to_dd_ratio"]
    print(summary[cols].to_string(index=False))
    print(f"\nOutputs: {cfg.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
