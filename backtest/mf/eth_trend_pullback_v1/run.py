#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run ETH Trend Pullback V1 on local CoinBacktest data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.mf.eth_trend_pullback_v1.engine import (  # noqa: E402
    ExecutionConfig,
    cost_stress_configs,
    run_backtest,
    summarize,
)
from backtest.mf.eth_trend_pullback_v1.strategy import (  # noqa: E402
    StrategyConfig,
    build_features,
    robustness_configs,
)
from src.data_feed.binance_funding_archive_loader import BinanceFundingArchiveLoader  # noqa: E402
from src.data_feed.okx_derivatives_loader import OKXDerivativesLoader  # noqa: E402
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.utils.report import print_full_report  # noqa: E402

STRATEGY_NAME = "ETH_TrendPullback_V1"
DEFAULT_WARMUP = "2022-01-01"
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2026-06-30"
DEFAULT_OUT = "data/reports/research/trend/eth_trend_pullback_v1"
DEFAULT_BINANCE_FUNDING = "research/eth_ict_price_action_portfolio/ict_pa_v3/inputs/binance_ethusdt_funding.csv"


def inclusive_end(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if len(str(value).strip()) <= 10:
        return ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    return ts


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="15m", choices=["15m"], help="V1 execution frame is frozen at 15m")
    p.add_argument("--warmup-start-date", default=DEFAULT_WARMUP)
    p.add_argument("--start-date", default=DEFAULT_START)
    p.add_argument("--end-date", default=DEFAULT_END)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--initial-capital", type=float, default=10_000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.01)
    p.add_argument("--max-notional-mult", type=float, default=3.0)
    p.add_argument("--fee-rate-per-side", type=float, default=0.00055)
    p.add_argument("--slippage-rate-per-side", type=float, default=0.00020)
    p.add_argument("--side", choices=["both", "long", "short"], default="both")
    p.add_argument("--funding-source", choices=["auto", "okx", "binance_proxy", "none"], default="auto")
    p.add_argument("--binance-funding-csv", default=DEFAULT_BINANCE_FUNDING)
    p.add_argument("--funding-timezone-offset-hours", type=float, default=8.0)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--skip-robustness", action="store_true")
    p.add_argument("--skip-cost-stress", action="store_true")
    p.add_argument("--write-full-audit", action="store_true")
    return p


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    kwargs: dict[str, Any] = {"symbol": args.symbol, "timeframe": args.timeframe, "db_name": args.db_name}
    if args.data_dir:
        kwargs["data_dir"] = Path(args.data_dir)
    loader = OKXTradeBarLoader(**kwargs)
    end_ts = inclusive_end(args.end_date)
    print(f"[load] {args.symbol} {args.timeframe} {args.warmup_start_date} -> {end_ts}", flush=True)
    bars = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        end_ts,
        chunksize=int(args.chunksize),
        cvd_mode="range",
        build_missing=not bool(args.no_build_missing),
    )
    if bars.empty:
        raise RuntimeError("OKXTradeBarLoader returned no rows")
    bars = bars.sort_index().copy()
    bars.index = pd.to_datetime(bars.index)
    for c in ("open", "high", "low", "close", "volume"):
        if c not in bars.columns:
            raise RuntimeError(f"trade bars missing required column: {c}")
        bars[c] = pd.to_numeric(bars[c], errors="coerce")
    bars = bars.dropna(subset=["open", "high", "low", "close", "volume"])
    print(f"[load] rows={len(bars):,} range={bars.index.min()} -> {bars.index.max()}", flush=True)
    return bars


def _funding_coverage_ratio(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    expected = max(1, int((end - start).total_seconds() // (8 * 3600)) + 1)
    return min(1.0, len(frame) / expected)


def _covers(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, tolerance: pd.Timedelta = pd.Timedelta(days=2)) -> bool:
    if frame.empty:
        return False
    span_ok = frame.index.min() <= start + tolerance and frame.index.max() >= end - tolerance
    return bool(span_ok and _funding_coverage_ratio(frame, start, end) >= 0.90)


def load_funding(args: argparse.Namespace, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    if args.funding_source == "none":
        return pd.DataFrame(), "NONE", {"note": "funding disabled by CLI"}

    if args.funding_source in {"auto", "okx"}:
        try:
            kwargs: dict[str, Any] = {"symbol": args.symbol}
            if args.data_dir:
                kwargs["data_dir"] = Path(args.data_dir)
            okx = OKXDerivativesLoader(**kwargs)
            frame = okx.load_funding_rates(start, end)
            if not frame.empty:
                frame = frame.copy()
                if "funding_rate" not in frame.columns:
                    raise RuntimeError("OKX funding table missing funding_rate")
                if "mark_price" not in frame.columns:
                    frame["mark_price"] = pd.NA
                frame["source"] = "OKX_LOCAL"
            if _covers(frame, start, end):
                meta = {"rows": len(frame), "start": frame.index.min(), "end": frame.index.max(), "proxy": False, "coverage_ratio": _funding_coverage_ratio(frame, start, end)}
                print(f"[funding] OKX local rows={len(frame):,} {frame.index.min()} -> {frame.index.max()}", flush=True)
                return frame, "OKX_LOCAL", meta
            if args.funding_source == "okx":
                meta = {"rows": len(frame), "start": frame.index.min() if not frame.empty else None, "end": frame.index.max() if not frame.empty else None, "proxy": False, "coverage_ratio": _funding_coverage_ratio(frame, start, end), "warning": "insufficient OKX funding coverage"}
                print("[funding] WARNING: OKX local funding coverage is insufficient; funding is not fabricated.", flush=True)
                return pd.DataFrame(), "OKX_INSUFFICIENT", meta
        except Exception as exc:
            if args.funding_source == "okx":
                raise
            print(f"[funding] OKX local unavailable: {type(exc).__name__}: {exc}", flush=True)

    if args.funding_source in {"auto", "binance_proxy"}:
        path = Path(args.binance_funding_csv)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        loader = BinanceFundingArchiveLoader(path, timezone_offset_hours=float(args.funding_timezone_offset_hours))
        frame = loader.load(start, end)
        if _covers(frame, start, end):
            meta = {"rows": len(frame), "start": frame.index.min(), "end": frame.index.max(), "proxy": True, "coverage_ratio": _funding_coverage_ratio(frame, start, end), "path": str(path)}
            print(f"[funding] Binance ETHUSDT proxy rows={len(frame):,} {frame.index.min()} -> {frame.index.max()}", flush=True)
            return frame, "BINANCE_ETHUSDT_PROXY", meta
        meta = {"rows": len(frame), "start": frame.index.min() if not frame.empty else None, "end": frame.index.max() if not frame.empty else None, "proxy": True, "coverage_ratio": _funding_coverage_ratio(frame, start, end), "path": str(path), "warning": "insufficient proxy coverage"}
        print("[funding] WARNING: no full-history funding source available; result will be labeled FUNDING_UNAVAILABLE.", flush=True)
        return pd.DataFrame(), "FUNDING_UNAVAILABLE", meta

    return pd.DataFrame(), "FUNDING_UNAVAILABLE", {"warning": "no funding source resolved"}


def _slice_backtest(features: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    # Warmup is never tradable.  Keeping the slice here rather than inside the
    # loader prevents the prior Turtle bug where 2022 warmup trades leaked into
    # reported performance.
    out = features.loc[(features.index >= start) & (features.index <= end)].copy()
    if out.empty:
        raise RuntimeError(f"no rows in official backtest window {start} -> {end}")
    return out


def write_markdown_report(summary: dict[str, Any], funding_meta: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# ETH Trend Pullback V1 — Summary",
        "",
        "## Status",
        "",
        "This is a frozen V1 baseline. Do not tune it from one losing period. Promotion requires causal audit, yearly stability, cost/funding stress, and parameter-neighborhood robustness.",
        "",
        "## Core metrics",
        "",
    ]
    for key in (
        "total_trades", "long_trades", "short_trades", "trades_per_year", "total_return_pct", "cagr_pct",
        "max_drawdown_pct", "calmar", "profit_factor", "win_rate_pct", "avg_holding_hours", "median_holding_hours",
        "p90_holding_hours", "max_no_entry_days", "max_consecutive_loss_trades", "max_consecutive_loss_days",
        "positive_month_ratio_pct", "trading_fees", "funding_pnl", "funding_source",
    ):
        lines.append(f"- **{key}**: {summary.get(key)}")
    lines += ["", "## Funding", "", f"```json\n{json.dumps(funding_meta, ensure_ascii=False, indent=2, default=str)}\n```", ""]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_once(features: pd.DataFrame, execution: ExecutionConfig, funding: pd.DataFrame, funding_source: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict[str, Any], list[dict[str, Any]], pd.DataFrame, list[dict[str, Any]]]:
    trades, equity, ledger = run_backtest(features, execution, funding=funding)
    summary = summarize(
        trades,
        equity,
        execution,
        window_start=start,
        window_end=end,
        funding_source=funding_source,
        funding_rows=len(funding),
    )
    return summary, trades, equity, ledger


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parser().parse_args(argv)
    start = pd.Timestamp(args.start_date)
    end = inclusive_end(args.end_date)
    warmup = pd.Timestamp(args.warmup_start_date)
    if warmup >= start:
        raise ValueError("warmup-start-date must be before start-date")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    bars = load_bars(args)
    strategy_cfg = StrategyConfig()
    print("[features] build 4H regime + 1H pullback + 15m re-acceleration with explicit available_time", flush=True)
    features_all = build_features(bars, strategy_cfg)
    features = _slice_backtest(features_all, start, end)
    causal_bad = features.loc[features["signal"].ne(0) & ~features["context_available_time_flag"]]
    if len(causal_bad):
        raise RuntimeError(f"causal context audit failed on {len(causal_bad)} signal rows")

    funding, funding_source, funding_meta = load_funding(args, start, end)
    base_exec = ExecutionConfig(
        initial_capital=float(args.initial_capital),
        risk_per_trade=float(args.risk_per_trade),
        max_notional_mult=float(args.max_notional_mult),
        fee_rate_per_side=float(args.fee_rate_per_side),
        slippage_rate_per_side=float(args.slippage_rate_per_side),
        side_mode=str(args.side),
    )

    print(f"[base] signals={int(features['signal'].ne(0).sum()):,} side={args.side}", flush=True)
    summary, trades, equity, ledger = run_once(features, base_exec, funding, funding_source, start, end)
    summary.update(
        {
            "strategy": STRATEGY_NAME,
            "symbol": args.symbol,
            "signal_timeframe": args.timeframe,
            "warmup_start_date": str(args.warmup_start_date),
            "backtest_start_date": str(args.start_date),
            "backtest_end_date": str(args.end_date),
            "signal_count": int(features["signal"].ne(0).sum()),
            "causal_context_failures": int(len(causal_bad)),
        }
    )

    pd.DataFrame(trades).to_csv(out_dir / "trades.csv", index=False, encoding="utf-8-sig")
    if not equity.empty:
        equity.to_csv(out_dir / "equity.csv", encoding="utf-8-sig")
    pd.DataFrame(ledger).to_csv(out_dir / "funding_ledger.csv", index=False, encoding="utf-8-sig")
    signal_audit_cols = [
        "open", "high", "low", "close", "signal", "stop", "signal_available_time",
        "used_h1_timestamp", "used_h1_available_time", "used_h4_timestamp", "used_h4_available_time",
        "context_available_time_flag", "h4_regime_long", "h4_regime_short", "h1_reclaim_long", "h1_reclaim_short",
        "h1_setup_long_active", "h1_setup_short_active", "trigger_high", "trigger_low", "recent_low", "recent_high",
    ]
    features.loc[features["signal"].ne(0), [c for c in signal_audit_cols if c in features.columns]].to_csv(
        out_dir / "signal_audit.csv", encoding="utf-8-sig"
    )
    if args.write_full_audit:
        features[[c for c in signal_audit_cols if c in features.columns]].to_csv(out_dir / "full_audit.csv", encoding="utf-8-sig")

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "strategy_config": asdict(strategy_cfg), "execution_config": asdict(base_exec), "funding": funding_meta}, f, ensure_ascii=False, indent=2, default=str)
    write_markdown_report(summary, funding_meta, out_dir)

    # Side diagnostics are not alternate strategies and are never selected as
    # the production baseline; they explain whether one side is carrying PnL.
    side_rows = []
    for side in ("long", "short"):
        s, _, _, _ = run_once(features, replace(base_exec, side_mode=side), funding, funding_source, start, end)
        side_rows.append(s)
    pd.DataFrame(side_rows).to_csv(out_dir / "side_breakdown.csv", index=False, encoding="utf-8-sig")

    if not args.skip_cost_stress:
        cost_rows = []
        for n, (name, ecfg) in enumerate(cost_stress_configs(base_exec), start=1):
            print(f"[cost {n}/4] {name}", flush=True)
            s, _, _, _ = run_once(features, ecfg, funding, funding_source, start, end)
            s["scenario"] = name
            cost_rows.append(s)
        pd.DataFrame(cost_rows).to_csv(out_dir / "cost_stress.csv", index=False, encoding="utf-8-sig")

    if not args.skip_robustness:
        robust_rows = []
        configs = robustness_configs(strategy_cfg)
        for n, (name, scfg) in enumerate(configs, start=1):
            print(f"[robustness {n}/{len(configs)}] {name}", flush=True)
            f = _slice_backtest(build_features(bars, scfg), start, end)
            s, _, _, _ = run_once(f, base_exec, funding, funding_source, start, end)
            s["scenario"] = name
            s["trigger_lookback_bars"] = scfg.trigger_lookback_bars
            s["setup_active_hours"] = scfg.setup_active_hours
            robust_rows.append(s)
        pd.DataFrame(robust_rows).to_csv(out_dir / "robustness.csv", index=False, encoding="utf-8-sig")

    # Preserve the project's familiar full report as a presentation sidecar.
    # Canonical MDD/Calmar remain summary.json because they use marked equity.
    if trades:
        report_trades = [dict(t) for t in trades]
        print_full_report(
            trade_history=report_trades,
            df=features,
            initial_capital=float(args.initial_capital),
            capital=float(summary["final_equity"]),
            strategy_name=STRATEGY_NAME,
            total_days=max((end - start).total_seconds() / 86400.0, 1.0),
            ai_enabled=False,
            symbol=args.symbol,
            report_dir=out_dir,
        )

    print("\n" + "=" * 100)
    print("ETH TREND PULLBACK V1")
    print("=" * 100)
    for k in (
        "total_trades", "long_trades", "short_trades", "total_return_pct", "cagr_pct", "max_drawdown_pct", "calmar",
        "profit_factor", "win_rate_pct", "avg_holding_hours", "max_no_entry_days", "max_consecutive_loss_days",
        "positive_month_ratio_pct", "trading_fees", "funding_pnl", "funding_source",
    ):
        print(f"{k:>30}: {summary.get(k)}")
    print(f"{'output':>30}: {out_dir.resolve()}")
    print("=" * 100)
    return summary


if __name__ == "__main__":
    main()
