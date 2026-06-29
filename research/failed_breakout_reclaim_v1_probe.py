#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch probe for ETH Range Failed Breakdown / Failed Breakout Reclaim V1."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCRIPT = Path("backtest/mf/eth_range_failed_breakout_reclaim_v1_backtest.py")
FILE_STEM = "eth_range_failed_breakout_reclaim_v1"


def _py() -> str:
    return sys.executable or "python"


def _load_summary(out_dir: Path) -> dict[str, Any]:
    p = out_dir / f"{FILE_STEM}_summary.json"
    if not p.exists():
        return {"error": f"missing summary: {p}"}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_trades(out_dir: Path) -> pd.DataFrame:
    p = out_dir / f"{FILE_STEM}_trades.csv"
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(p)


def _simulate_remove_trades(trades: pd.DataFrame, initial_capital: float, remove_idx: set[int]) -> dict[str, Any]:
    if trades.empty:
        return {"total_trades": 0, "final_capital": initial_capital, "total_return_pct": 0.0}
    cap = float(initial_capital)
    kept = []
    for i, row in trades.reset_index(drop=True).iterrows():
        if i in remove_idx:
            continue
        # Replay using recorded return_pct to preserve compounding approximately.
        pnl = cap * float(row.get("return_pct", 0.0))
        item = row.to_dict()
        item["pnl_replayed"] = pnl
        item["capital_replayed"] = cap + pnl
        kept.append(item)
        cap += pnl
    if not kept:
        return {"total_trades": 0, "final_capital": round(cap, 4), "total_return_pct": round((cap / initial_capital - 1.0) * 100.0, 4)}
    kdf = pd.DataFrame(kept)
    wins = kdf[kdf["pnl_replayed"] > 0]
    losses = kdf[kdf["pnl_replayed"] <= 0]
    gp = float(wins["pnl_replayed"].sum()) if not wins.empty else 0.0
    gl = float(-losses["pnl_replayed"].sum()) if not losses.empty else 0.0
    pf = gp / gl if gl > 0 else float("inf")
    return {
        "total_trades": int(len(kdf)),
        "long_trades": int((kdf.get("type") == "LONG").sum()) if "type" in kdf else None,
        "short_trades": int((kdf.get("type") == "SHORT").sum()) if "type" in kdf else None,
        "final_capital": round(cap, 4),
        "total_return_pct": round((cap / initial_capital - 1.0) * 100.0, 4),
        "win_rate": round(float((kdf["pnl_replayed"] > 0).mean() * 100.0), 4),
        "profit_factor": round(pf, 4) if pd.notna(pf) and pf != float("inf") else "inf",
        "gross_profit_replayed": round(gp, 4),
        "gross_loss_replayed": round(gl, 4),
    }


def _postprocess(base_dir: Path, initial_capital: float) -> list[dict[str, Any]]:
    trades = _load_trades(base_dir)
    rows: list[dict[str, Any]] = []
    if trades.empty or "return_pct" not in trades.columns:
        return rows
    t = trades.copy().reset_index(drop=True)
    t["pnl_abs"] = pd.to_numeric(t.get("pnl", 0.0), errors="coerce").fillna(0.0)
    top_idx = list(t.sort_values("pnl_abs", ascending=False).index)
    scenarios = {
        "remove_top1_pnl": set(top_idx[:1]),
        "remove_top3_pnl": set(top_idx[:3]),
    }
    for name, idxs in scenarios.items():
        s = _simulate_remove_trades(t, initial_capital, idxs)
        s.update({"scenario": name, "kind": "post_remove", "removed_count": len(idxs)})
        rows.append(s)
    return rows


def _run_one(name: str, common: list[str], extra: list[str], out_root: Path, initial_capital: float) -> dict[str, Any]:
    out_dir = out_root / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [_py(), str(SCRIPT), *common, *extra, "--out-dir", str(out_dir)]
    print("\n" + "=" * 100)
    print(f"[probe] running {name}")
    print(" ".join(cmd))
    print("=" * 100, flush=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = "."
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)
    s = _load_summary(out_dir)
    s.update({"scenario": name, "kind": "backtest", "out_dir": str(out_dir)})
    return s


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Failed Breakout/Reclaim V1 scenario probe", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--range-pct", type=float, default=0.0020)
    p.add_argument("--price-step", type=float, default=1.0)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--out-dir", default="data/reports/research/failed_breakout_reclaim_v1_probe/default")
    p.add_argument("--data-dir", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    common = [
        "--start-date", args.start_date,
        "--end-date", args.end_date,
        "--warmup-start-date", args.warmup_start_date,
        "--range-pct", str(args.range_pct),
        "--price-step", str(args.price_step),
        "--initial-capital", str(args.initial_capital),
    ]
    if args.data_dir:
        common += ["--data-dir", args.data_dir]

    scenarios: list[tuple[str, list[str]]] = [
        ("base_both", ["--preset", "high", "--side-mode", "both"]),
        ("long_only", ["--preset", "high", "--side-mode", "long_only"]),
        ("short_only", ["--preset", "high", "--side-mode", "short_only"]),
        ("stable_both", ["--preset", "stable", "--side-mode", "both"]),
        ("turbo_both", ["--preset", "turbo", "--side-mode", "both"]),
        ("fee_2x", ["--preset", "high", "--side-mode", "both", "--fee-rate", "0.00110"]),
        ("slippage_2x", ["--preset", "high", "--side-mode", "both", "--slippage-pct", "0.00030"]),
        ("no_2026", ["--preset", "high", "--side-mode", "both", "--end-date", "2025-12-31"]),
        ("target_3r", ["--preset", "high", "--side-mode", "both", "--target-r", "3.0"]),
        ("stricter_effort", ["--preset", "high", "--side-mode", "both", "--max-effort-bucket-quantile", "0.94", "--bar-effort-quantile", "0.86"]),
        ("looser_signals", ["--preset", "high", "--side-mode", "both", "--max-effort-bucket-quantile", "0.86", "--bar-effort-quantile", "0.76"]),
    ]

    rows: list[dict[str, Any]] = []
    for name, extra in scenarios:
        rows.append(_run_one(name, common, extra, out_root, args.initial_capital))

    base_dir = out_root / "base_both"
    detail_dir = out_root / "details"
    detail_dir.mkdir(parents=True, exist_ok=True)
    trades = _load_trades(base_dir)
    if not trades.empty:
        trades.to_csv(detail_dir / "base_both_trades.csv", index=False)
    rows.extend(_postprocess(base_dir, args.initial_capital))

    summary = pd.DataFrame(rows)
    summary_path = out_root / "failed_breakout_reclaim_v1_probe_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("\n" + "=" * 100)
    print(f"Wrote summary: {summary_path.resolve()}")
    cols = [c for c in ["scenario", "kind", "total_return_pct", "profit_factor", "max_drawdown_pct", "total_trades", "long_trades", "short_trades", "signal_count"] if c in summary.columns]
    if cols:
        print(summary[cols].to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
