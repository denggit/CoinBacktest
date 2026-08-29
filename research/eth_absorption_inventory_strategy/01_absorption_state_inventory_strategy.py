#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Absorption / Market-Control Inventory Strategy R01.

Strategy thesis
---------------
Do not search for conventional entries/exits.  At each completed market state,
ask only whether current evidence says the market should gain long inventory,
gain short inventory, or do nothing.

Fresh 15m/1H/4H control events create one +1%/-1% margin vote.  5m is a causal
micro veto only.  The account is cross, netted, 10x by default.  There is no
TP/SL/time exit; opposite future evidence mechanically reduces/reverses inventory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.absorption_inventory_strategy import (
    AccountConfig,
    StrategyConfig,
    build_multiscale_votes,
    period_returns,
    simulate_cross_inventory,
)
from src.research_common.progress import ProgressReporter

DEFAULT_REPORT_DIR = ROOT / "data" / "reports" / "research" / "eth_absorption_inventory_strategy" / "01_absorption_state_inventory_strategy"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETH multi-scale absorption/control inventory strategy R01")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30 23:59:59")
    p.add_argument("--initial-equity", type=float, default=10_000.0)
    p.add_argument("--leverage", type=float, default=10.0)
    p.add_argument("--vote-margin-fraction", type=float, default=0.01)
    p.add_argument("--fee-rate-per-fill", type=float, default=0.00055)
    p.add_argument("--slippage-bps-per-fill", type=float, default=0.0)
    p.add_argument("--maintenance-margin-rate", type=float, default=0.005)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return p.parse_args()


def _year_chunks(start: pd.Timestamp, end: pd.Timestamp):
    for year in range(start.year, end.year + 1):
        left = max(start, pd.Timestamp(year=year, month=1, day=1))
        right = min(end, pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59))
        if left <= right:
            yield left, right


def _load_and_build_votes(loader: OKXTradeBarLoader, start: pd.Timestamp, end: pd.Timestamp, warmup_start: pd.Timestamp):
    parts: list[pd.DataFrame] = []
    audits: list[pd.DataFrame] = []
    chunks = list(_year_chunks(start, end))
    progress = ProgressReporter(label="[strategy] yearly multiscale control", total=len(chunks), every=1)
    # 4H baseline+defense needs ~3 weeks; 45d also protects year boundaries.
    warm = pd.Timedelta(days=45)
    for i, (left, right) in enumerate(chunks, 1):
        load_start = max(warmup_start, left - warm)
        raw = loader.fetch_data_by_date_range(load_start, right, cvd_mode="range", build_missing=False)
        if raw.empty:
            raise RuntimeError(f"no local 1m trade bars for {load_start} -> {right}")
        frame, audit = build_multiscale_votes(raw, config=StrategyConfig())
        keep = frame.loc[(frame.index >= left) & (frame.index <= right)].copy()
        parts.append(keep)
        if not audit.empty:
            audit = audit.copy()
            if "available_time" in audit.columns:
                audit = audit[(pd.to_datetime(audit["available_time"]) >= left) & (pd.to_datetime(audit["available_time"]) <= right)]
            audits.append(audit)
        progress.update(i)
    progress.close()
    full = pd.concat(parts).sort_index()
    full = full[~full.index.duplicated(keep="last")]
    audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    return full, audit


def _daily(path: pd.DataFrame) -> pd.DataFrame:
    out = path.resample("1D").agg(
        equity=("equity", "last"),
        min_equity=("equity", "min"),
        max_equity=("equity", "max"),
        mean_abs_exposure_x=("net_exposure_x", lambda s: float(np.nanmean(np.abs(s))) if len(s) else np.nan),
        max_abs_exposure_x=("net_exposure_x", lambda s: float(np.nanmax(np.abs(s))) if len(s) else np.nan),
        votes=("signal", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) != 0).sum())),
    ).dropna(subset=["equity"])
    out["return"] = out["equity"].pct_change().fillna(0.0)
    out["peak"] = out["equity"].cummax()
    out["drawdown"] = out["equity"] / out["peak"] - 1.0
    return out


def _write_report(report_dir: Path, args: argparse.Namespace, scenarios: pd.DataFrame, votes: pd.DataFrame, orders: pd.DataFrame) -> None:
    base = scenarios.loc[scenarios["cost_multiple"].eq(1.0)].iloc[0]
    family = orders.groupby(["family", "signal"], dropna=False).size().rename("orders").reset_index() if len(orders) else pd.DataFrame()
    lines = [
        "# ETH Absorption / Market-Control Inventory Strategy R01",
        "",
        "## Strategy",
        "",
        "This is a strategy backtest, not an event atlas. The signal engine never reads current position/PnL/entry price.",
        "",
        "- 15m / 1H / 4H fresh control events can create one inventory vote.",
        "- Failed aggressive pressure, repeated defense and spring/upthrust vote against the aggressor.",
        "- Efficient aggressive pressure that actually moves price votes with the aggressor.",
        "- 5m is confirmation/veto only; it never creates inventory by itself.",
        "- One vote = 1% of current equity as margin at 10x by default (~0.10x equity notional).",
        "- No conventional TP, SL or time exit. Opposite future evidence mechanically nets inventory.",
        "- Closed-bar signal; next 1m open execution. Higher-timeframe contexts are aligned by availability time.",
        "",
        "## Base result",
        "",
        f"- Total return: `{float(base['total_return']):.2%}`",
        f"- CAGR: `{float(base['cagr']):.2%}`",
        f"- Max drawdown: `{float(base['max_drawdown']):.2%}`",
        f"- Orders: `{int(base['orders']):,}`",
        f"- Mean / max absolute exposure: `{float(base['mean_abs_exposure_x']):.2f}x / {float(base['max_abs_exposure_x']):.2f}x`",
        f"- Liquidated: `{bool(base['liquidated'])}`",
        "",
        "## Acceptance rule",
        "",
        "Do not tune thresholds from this run. Reject R01 if base is economically poor, if 2x/3x costs destroy it, or if yearly performance is concentrated in one regime. Only then decide whether the mechanism deserves one structural revision.",
        "",
        "## Files",
        "",
        "- `01_scenario_summary.csv`: base + 2x/3x cost stress.",
        "- `02_yearly.csv`, `03_monthly.csv`: realized account returns.",
        "- `04_daily_base.csv`: equity/exposure path.",
        "- `05_votes.csv`: every strategy vote after multiscale confirmation.",
        "- `06_orders_base.csv`: every executed inventory change.",
        "- `07_raw_scale_events.csv`: diagnostic only; not a strategy result.",
        "- `run_config.json`: exact frozen mechanics.",
    ]
    (report_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not family.empty:
        family.to_csv(report_dir / "08_order_family_counts.csv", index=False)


def main() -> int:
    args = parse_args()
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    warmup = pd.Timestamp(args.warmup_start_date)
    if not warmup < start <= end:
        raise ValueError("require warmup_start < start <= end")

    print("[run] ETH Absorption / Market-Control Inventory Strategy R01")
    print(f"[window] warmup={warmup} research={start} -> {end}")
    print("[logic] 15m/1H/4H fresh control votes + 5m veto | no conventional TP/SL/exit")
    print(f"[account] cross={args.leverage:g}x vote_margin={args.vote_margin_fraction:.2%} fee/fill={args.fee_rate_per_fill:.5%}")

    loader = OKXTradeBarLoader(symbol=args.symbol, timeframe="1m")
    signal_frame, raw_events = _load_and_build_votes(loader, start, end, warmup)
    votes = signal_frame[signal_frame["signal"].ne(0)].copy()
    print(f"[signals] votes={len(votes):,} long={(votes['signal'] > 0).sum():,} short={(votes['signal'] < 0).sum():,}")

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    scenarios = []
    base_path = base_orders = None
    sim_progress = ProgressReporter(label="[backtest] cost stress", total=3, every=1)
    for j, multiple in enumerate((1.0, 2.0, 3.0), 1):
        account = AccountConfig(
            initial_equity=args.initial_equity,
            leverage=args.leverage,
            vote_margin_fraction=args.vote_margin_fraction,
            fee_rate_per_fill=args.fee_rate_per_fill * multiple,
            slippage_bps_per_fill=args.slippage_bps_per_fill * multiple,
            maintenance_margin_rate=args.maintenance_margin_rate,
        )
        path, orders, summary = simulate_cross_inventory(signal_frame, account=account)
        row = {"cost_multiple": multiple, **summary}
        scenarios.append(row)
        if multiple == 1.0:
            base_path, base_orders = path, orders
        sim_progress.update(j)
    sim_progress.close()

    scenario_df = pd.DataFrame(scenarios)
    scenario_df.to_csv(report_dir / "01_scenario_summary.csv", index=False)
    assert base_path is not None and base_orders is not None
    period_returns(base_path, "YS").to_csv(report_dir / "02_yearly.csv")
    period_returns(base_path, "MS").to_csv(report_dir / "03_monthly.csv")
    _daily(base_path).to_csv(report_dir / "04_daily_base.csv")
    votes.to_csv(report_dir / "05_votes.csv")
    base_orders.to_csv(report_dir / "06_orders_base.csv", index=False)
    raw_events.to_csv(report_dir / "07_raw_scale_events.csv", index=False)

    config = {
        "symbol": args.symbol,
        "warmup_start_date": str(warmup),
        "start_date": str(start),
        "end_date": str(end),
        "account": {
            "initial_equity": args.initial_equity,
            "leverage": args.leverage,
            "vote_margin_fraction": args.vote_margin_fraction,
            "fee_rate_per_fill": args.fee_rate_per_fill,
            "slippage_bps_per_fill": args.slippage_bps_per_fill,
            "maintenance_margin_rate": args.maintenance_margin_rate,
        },
        "strategy": StrategyConfig().__dict__,
        "note": "Thresholds are frozen semantic definitions. Do not optimize from R01 results.",
    }
    (report_dir / "run_config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    _write_report(report_dir, args, scenario_df, votes, base_orders)

    b = scenario_df.iloc[0]
    print(
        f"[result] return={float(b['total_return']):.2%} CAGR={float(b['cagr']):.2%} "
        f"MDD={float(b['max_drawdown']):.2%} orders={int(b['orders']):,} liquidated={bool(b['liquidated'])}"
    )
    print(f"[report] {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
