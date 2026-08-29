#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Portfolio V2 - Trend Breakout V1 executable strategy backtest.

This is intentionally a *strategy* backtest, not an event study.  It produces
explicit entries, stops, exits, dynamic risk sizing, stress scenarios, period
breakdowns and a funnel audit.  The base event universe is kept broad; quality
features scale risk instead of repeatedly deleting events.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backtest_common.ohlcv_backtest import (  # noqa: E402
    SignalBacktestParams,
    emit_signal_report,
    run_signal_backtest,
    summarize_signal_backtest,
)
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.portfolio_common.strategy_catalog import build_core_strategy_catalog  # noqa: E402
from src.research_common.progress import progress_iter  # noqa: E402
from src.sleeve_lib.trend_breakout_v1 import TrendBreakoutConfig, build_features  # noqa: E402
from src.strategy_common import FunnelPolicy, FunnelStage, audit_funnel  # noqa: E402


STRATEGY_ID = "ETH_STRATEGY_TREND_BREAKOUT_V1"
DEFAULT_OUT_DIR = Path("data/reports/research/eth_portfolio_v2/trend_breakout_v1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    cfg = TrendBreakoutConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default=cfg.symbol)
    parser.add_argument("--warmup-start-date", default=cfg.warmup_start_date)
    parser.add_argument("--start-date", default=cfg.start_date)
    parser.add_argument("--end-date", default=cfg.end_date)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--build-missing", action="store_true", help="Allow the standard Loader to build genuinely missing 1m days.")
    parser.add_argument("--write-full-audit", action="store_true")
    parser.add_argument("--skip-full-report", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def _resample_trade_bars_15m(one_minute: pd.DataFrame) -> pd.DataFrame:
    if one_minute.empty:
        return one_minute.copy()
    agg: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    sum_candidates = (
        "volume",
        "trades_count",
        "buy_volume",
        "sell_volume",
        "notional",
        "buy_notional",
        "sell_notional",
        "buy_trades_count",
        "sell_trades_count",
        "delta_volume",
        "delta_notional",
        "large_buy_notional",
        "large_sell_notional",
        "large_buy_trades_count",
        "large_sell_trades_count",
        "large_delta_notional",
        "large_trades_count",
    )
    for column in sum_candidates:
        if column in one_minute.columns:
            agg[column] = "sum"
    for column in ("max_trade_notional", "max_trade_size"):
        if column in one_minute.columns:
            agg[column] = "max"

    ordered = one_minute.sort_index(kind="stable")
    bars = ordered.resample("15min", label="left", closed="left").agg(agg)
    bars["source_minutes"] = ordered["close"].resample("15min", label="left", closed="left").count()
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    return bars


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    print(f"[load] local OKX 1m trade bars {args.warmup_start_date} -> {args.end_date}", flush=True)
    loader = OKXTradeBarLoader(symbol=args.symbol, timeframe="1m")
    one = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
        build_missing=bool(args.build_missing),
        cvd_mode="range",
    )
    if one.empty:
        raise RuntimeError(
            "No local 1m trade bars were returned. Prebuild through src.data_feed or rerun with --build-missing; "
            "this backtest does not implement its own data interface."
        )
    print(f"[load] 1m rows={len(one):,}", flush=True)
    bars = _resample_trade_bars_15m(one)
    print(
        f"[resample] 15m rows={len(bars):,} source_minutes median={bars['source_minutes'].median():.1f} "
        f"min={bars['source_minutes'].min():.0f}",
        flush=True,
    )
    return bars


def _params(cfg: TrendBreakoutConfig, *, fee_mult: float = 1.0, slip_mult: float = 1.0) -> SignalBacktestParams:
    return SignalBacktestParams(
        initial_capital=cfg.initial_capital,
        risk_per_trade=cfg.base_risk_per_trade,
        max_notional_mult=cfg.max_notional_mult,
        fee_rate=cfg.fee_rate_per_side * fee_mult,
        slippage_pct=cfg.slippage_pct * slip_mult,
        risk_mult_col="risk_mult",
        min_risk_mult=cfg.min_risk_mult,
        max_risk_mult=cfg.max_risk_mult,
        signal_col="signal",
        stop_col="stop",
        target_col=None,
        target_r=cfg.target_r,
        min_stop_pct=cfg.min_stop_pct,
        max_stop_pct=cfg.max_stop_pct,
        cooldown_bars=cfg.cooldown_bars,
        # Disable close-known exits in V1. The generic helper executes those
        # paths at the same bar close, while a close-derived decision is only
        # safely executable from the next open. Protective stop / fixed-R
        # target remain pre-existing executable orders.
        max_hold_bars=10**9,
        no_progress_bars=0,
        exit_on_opposite_signal=False,
        # Generic trailing updates from a bar close and must not be applied to
        # that same bar's earlier high/low. Keep it disabled here.
        trailing_atr_col=None,
        trailing_atr_mult=0.0,
    )


def _delay_signal_frame(features: pd.DataFrame, delay_bars: int) -> pd.DataFrame:
    if delay_bars <= 0:
        return features
    out = features.copy()
    for column in ("signal", "stop", "risk_mult", "signal_reason"):
        if column in out.columns:
            out[column] = out[column].shift(delay_bars)
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce").fillna(0).astype("int8")
    out["risk_mult"] = pd.to_numeric(out["risk_mult"], errors="coerce").fillna(0.0)
    return out


def _execution_eligible_count(features: pd.DataFrame, cfg: TrendBreakoutConfig) -> int:
    if features.empty:
        return 0
    signal = pd.to_numeric(features["signal"], errors="coerce").fillna(0).astype(int)
    stop = pd.to_numeric(features["stop"], errors="coerce")
    next_open = pd.to_numeric(features["open"], errors="coerce").shift(-1)
    side = signal.astype(float)
    stop_pct = (next_open - stop).abs() / next_open
    direction_ok = ((side > 0) & (stop < next_open)) | ((side < 0) & (stop > next_open))
    eligible = (
        signal.ne(0)
        & next_open.notna()
        & stop.notna()
        & direction_ok
        & stop_pct.le(cfg.max_stop_pct)
    )
    # Stops narrower than min_stop_pct are widened by the execution engine and
    # remain eligible, matching _valid_stop().
    return int(eligible.sum())


def _period_metrics(trades: list[dict[str, Any]], freq: str) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["period", "trades", "return", "win_rate", "profit_factor"])
    frame = pd.DataFrame(trades).copy()
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], errors="coerce")
    frame = frame.dropna(subset=["exit_time"])
    if freq == "year":
        frame["period"] = frame["exit_time"].dt.year.astype(str)
    elif freq == "quarter":
        frame["period"] = frame["exit_time"].dt.to_period("Q").astype(str)
    elif freq == "month":
        frame["period"] = frame["exit_time"].dt.to_period("M").astype(str)
    else:
        raise ValueError(freq)

    rows: list[dict[str, object]] = []
    for period, group in frame.groupby("period", sort=True):
        returns = pd.to_numeric(group["return_pct"], errors="coerce").dropna()
        gains = float(returns[returns > 0].sum())
        losses = float(-returns[returns < 0].sum())
        pf = gains / losses if losses > 0 else (float("inf") if gains > 0 else float("nan"))
        rows.append(
            {
                "period": period,
                "trades": int(len(group)),
                "return": float(np.prod(1.0 + returns.to_numpy(dtype=float)) - 1.0),
                "win_rate": float((returns > 0).mean()) if len(returns) else float("nan"),
                "profit_factor": pf,
            }
        )
    return pd.DataFrame(rows)


def _max_flat_days(trades: list[dict[str, Any]], start: pd.Timestamp, end: pd.Timestamp) -> float:
    if not trades:
        return max(0.0, (end - start).total_seconds() / 86400.0)
    frame = pd.DataFrame(trades).sort_values("entry_time")
    entries = pd.to_datetime(frame["entry_time"], errors="coerce")
    exits = pd.to_datetime(frame["exit_time"], errors="coerce")
    gaps = [max(0.0, (entries.iloc[0] - start).total_seconds() / 86400.0)]
    for previous_exit, next_entry in zip(exits.iloc[:-1], entries.iloc[1:]):
        gaps.append(max(0.0, (next_entry - previous_exit).total_seconds() / 86400.0))
    gaps.append(max(0.0, (end - exits.iloc[-1]).total_seconds() / 86400.0))
    return float(max(gaps))


def _max_consecutive_losing_days(trades: list[dict[str, Any]]) -> int:
    if not trades:
        return 0
    frame = pd.DataFrame(trades)
    frame["date"] = pd.to_datetime(frame["exit_time"], errors="coerce").dt.normalize()
    daily = frame.groupby("date")["pnl"].sum().sort_index()
    if daily.empty:
        return 0
    full = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"), fill_value=0.0)
    best = current = 0
    for value in full.to_numpy(dtype=float):
        if value < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return int(best)


def _enrich_summary(
    summary: dict[str, Any],
    trades: list[dict[str, Any]],
    cfg: TrendBreakoutConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, Any]:
    out = dict(summary)
    years = max((end - start).total_seconds() / (365.25 * 86400.0), 1e-9)
    final_capital = float(out.get("final_capital", cfg.initial_capital))
    out["cagr"] = (final_capital / cfg.initial_capital) ** (1.0 / years) - 1.0 if final_capital > 0 else -1.0
    out["max_flat_days"] = _max_flat_days(trades, start, end)
    out["max_consecutive_losing_days"] = _max_consecutive_losing_days(trades)
    monthly = _period_metrics(trades, "month")
    out["positive_month_rate"] = float((monthly["return"] > 0).mean()) if not monthly.empty else 0.0
    out["months_with_trades"] = int(len(monthly))
    out["avg_trades_per_month"] = float(len(trades) / max(1, int((end.to_period("M") - start.to_period("M")).n + 1)))
    return out


def _top_trade_removal(trades: list[dict[str, Any]]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["removed_top_n", "remaining_trades", "compounded_return", "profit_factor"])
    frame = pd.DataFrame(trades)
    returns = pd.to_numeric(frame["return_pct"], errors="coerce").fillna(0.0)
    order = returns.sort_values(ascending=False).index
    rows: list[dict[str, object]] = []
    for top_n in (0, 5, 10):
        kept = returns.drop(index=order[:top_n]) if top_n else returns
        gains = float(kept[kept > 0].sum())
        losses = float(-kept[kept < 0].sum())
        rows.append(
            {
                "removed_top_n": top_n,
                "remaining_trades": int(len(kept)),
                "compounded_return": float(np.prod(1.0 + kept.to_numpy(dtype=float)) - 1.0),
                "profit_factor": gains / losses if losses > 0 else float("inf"),
            }
        )
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[write] {path}", flush=True)


def _scenario_definitions() -> tuple[tuple[str, float, float, int], ...]:
    return (
        ("base", 1.0, 1.0, 0),
        ("fee_1p5x", 1.5, 1.0, 0),
        ("fee_2x", 2.0, 1.0, 0),
        ("fee_3x", 3.0, 1.0, 0),
        ("slippage_2x", 1.0, 2.0, 0),
        ("delay_1bar", 1.0, 1.0, 1),
    )


def _run_one(
    features: pd.DataFrame,
    cfg: TrendBreakoutConfig,
    *,
    fee_mult: float,
    slip_mult: float,
    delay_bars: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    replay_frame = _delay_signal_frame(features, delay_bars)
    params = _params(cfg, fee_mult=fee_mult, slip_mult=slip_mult)
    trades, equity = run_signal_backtest(replay_frame, params)
    summary = summarize_signal_backtest(
        trades,
        equity,
        cfg.initial_capital,
        signal_count=int(pd.to_numeric(replay_frame["signal"], errors="coerce").fillna(0).ne(0).sum()),
    )
    start = pd.Timestamp(features.index.min())
    end = pd.Timestamp(features.index.max())
    return trades, equity, _enrich_summary(summary, trades, cfg, start, end)


def _neighbourhood_configs(cfg: TrendBreakoutConfig) -> tuple[tuple[str, TrendBreakoutConfig], ...]:
    # Predeclared neighborhood only.  Results are never used to select a new
    # parameter inside this run.
    return (
        ("shorter_36_12", replace(cfg, breakout_lookback=36, stop_lookback=12)),
        ("base_48_16", cfg),
        ("longer_60_20", replace(cfg, breakout_lookback=60, stop_lookback=20)),
    )


def _decision(
    funnel_pass: bool,
    base: dict[str, Any],
    stress: pd.DataFrame,
    yearly: pd.DataFrame,
) -> dict[str, object]:
    fee2 = stress.loc[stress["scenario"].eq("fee_2x")]
    fee2_return = float(fee2.iloc[0]["total_return_pct"]) if len(fee2) else float("nan")
    positive_years = int((pd.to_numeric(yearly.get("return", pd.Series(dtype=float)), errors="coerce") > 0).sum())
    checks = {
        "funnel_pass": bool(funnel_pass),
        "trades_ge_300": int(base.get("total_trades", 0)) >= 300,
        "base_profitable": float(base.get("total_return_pct", -math.inf)) > 0.0,
        "profit_factor_ge_1p15": _finite_or(base.get("profit_factor"), 0.0) >= 1.15,
        "max_drawdown_le_20pct": _finite_or(base.get("max_drawdown_pct"), math.inf) <= 20.0,
        "fee_2x_profitable": math.isfinite(fee2_return) and fee2_return > 0.0,
        "positive_years_ge_3": positive_years >= 3,
    }
    passed = all(checks.values())
    return {
        "strategy_id": STRATEGY_ID,
        "decision": "KEEP_FOR_PORTFOLIO_CANDIDATE" if passed else "REJECT_OR_REDESIGN",
        "passed": passed,
        "checks": checks,
        "note": "Fixed first-pass gate. Do not tune a failed row into passing; redesign the mechanism or move to the next sleeve.",
    }


def _finite_or(value: object, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrendBreakoutConfig(
        symbol=args.symbol,
        warmup_start_date=args.warmup_start_date,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    cfg.validate()
    contract = next(item for item in build_core_strategy_catalog() if item.strategy_id == STRATEGY_ID)

    print(f"[run] {STRATEGY_ID}", flush=True)
    print("[goal] executable ETH perpetual strategy; no spot/funding-arbitrage logic", flush=True)
    print("[timing] closed 15m structure signal -> next 15m open; no multi-timeframe context", flush=True)

    bars = load_bars(args)
    print("[features] causal Trend Breakout V1", flush=True)
    full_features = build_features(bars, cfg)
    start_ts = pd.Timestamp(args.start_date)
    end_ts = pd.Timestamp(args.end_date)
    features = full_features.loc[(full_features.index >= start_ts) & (full_features.index <= end_ts)].copy()
    if len(features) < 100:
        raise RuntimeError("research window is too small after warmup")

    base_trades: list[dict[str, Any]] = []
    base_equity = pd.DataFrame()
    stress_rows: list[dict[str, object]] = []
    for name, fee_mult, slip_mult, delay_bars in progress_iter(
        _scenario_definitions(),
        label="[stress]",
        total=len(_scenario_definitions()),
        every=1,
        enabled=not args.no_progress,
    ):
        trades, equity, summary = _run_one(
            features,
            cfg,
            fee_mult=fee_mult,
            slip_mult=slip_mult,
            delay_bars=delay_bars,
        )
        row = {"scenario": name, "fee_mult": fee_mult, "slip_mult": slip_mult, "delay_bars": delay_bars, **summary}
        stress_rows.append(row)
        if name == "base":
            base_trades, base_equity = trades, equity
    stress = pd.DataFrame(stress_rows)
    base_summary = dict(stress.loc[stress["scenario"].eq("base")].iloc[0].to_dict())

    source_events = int(features["structure_event"].sum())
    scored_events = int((features["structure_event"] & features["risk_mult"].notna()).sum())
    eligible_events = _execution_eligible_count(features, cfg)
    funnel = audit_funnel(
        [
            FunnelStage("structure_break_events", source_events, "source", "Broad first structure-break events."),
            FunnelStage("quality_scored_events", scored_events, "score", "Quality changes risk; it does not hard-delete events."),
            FunnelStage("next_open_execution_eligible", eligible_events, "execution", "Stop/execution geometry known at next open."),
            FunnelStage("executed_trades", len(base_trades), "execution", "Single-position occupancy/cooldown applied."),
        ],
        FunnelPolicy(strategy_class="core", min_executed_trades_core=contract.target_min_trades),
    )

    yearly = _period_metrics(base_trades, "year")
    quarterly = _period_metrics(base_trades, "quarter")
    monthly = _period_metrics(base_trades, "month")
    top_removal = _top_trade_removal(base_trades)

    neighbourhood_rows: list[dict[str, object]] = []
    for name, neighbour_cfg in progress_iter(
        _neighbourhood_configs(cfg),
        label="[neighbourhood]",
        total=len(_neighbourhood_configs(cfg)),
        every=1,
        enabled=not args.no_progress,
    ):
        nf = build_features(bars, neighbour_cfg)
        nf = nf.loc[(nf.index >= start_ts) & (nf.index <= end_ts)].copy()
        trades, _equity, summary = _run_one(nf, neighbour_cfg, fee_mult=1.0, slip_mult=1.0, delay_bars=0)
        neighbourhood_rows.append(
            {
                "variant": name,
                "breakout_lookback": neighbour_cfg.breakout_lookback,
                "stop_lookback": neighbour_cfg.stop_lookback,
                "signal_events": int(nf["structure_event"].sum()),
                **summary,
            }
        )
    neighbourhood = pd.DataFrame(neighbourhood_rows)

    decision = _decision(funnel.passed, base_summary, stress, yearly)
    manifest = {
        "strategy_id": STRATEGY_ID,
        "portfolio_id": "ETH_PORTFOLIO_V2",
        "contract": contract.to_dict(),
        "config": asdict(cfg),
        "data_source": "src.data_feed.OKXTradeBarLoader 1m local cache -> causal 15m aggregation",
        "backtest_window": {"warmup": args.warmup_start_date, "start": args.start_date, "end": args.end_date},
        "cost_model": {
            "fee_rate_per_side": cfg.fee_rate_per_side,
            "baseline_roundtrip_fee": cfg.fee_rate_per_side * 2.0,
            "slippage_per_side": cfg.slippage_pct,
        },
        "causal_guards": [
            "prior breakout levels use rolling(...).max/min().shift(1)",
            "signal is formed from a fully closed signal bar",
            "entry is the next bar open in the generic replay engine",
            "no higher-timeframe feature is forward-filled into the signal frame",
            "quality features size risk instead of filtering events",
            "same-bar stop/target conflict is resolved stop-first by the generic engine",
            "generic close-based trailing is disabled to avoid same-bar path leakage",
            "opposite-signal and max-hold close exits are disabled in V1 because close-derived decisions require next-open execution",
        ],
        "decision": decision,
    }

    pd.DataFrame(base_trades).to_csv(out_dir / "01_trades.csv", index=False)
    if not base_equity.empty:
        base_equity.to_csv(out_dir / "02_equity.csv")
    stress.to_csv(out_dir / "03_stress.csv", index=False)
    yearly.to_csv(out_dir / "04_yearly.csv", index=False)
    quarterly.to_csv(out_dir / "05_quarterly.csv", index=False)
    monthly.to_csv(out_dir / "06_monthly.csv", index=False)
    funnel.to_frame().to_csv(out_dir / "07_funnel.csv", index=False)
    funnel.issues_frame().to_csv(out_dir / "08_funnel_issues.csv", index=False)
    top_removal.to_csv(out_dir / "09_top_trade_removal.csv", index=False)
    neighbourhood.to_csv(out_dir / "10_parameter_neighbourhood.csv", index=False)
    _write_json(out_dir / "11_summary.json", base_summary)
    _write_json(out_dir / "12_funnel_summary.json", funnel.summary())
    _write_json(out_dir / "13_decision.json", decision)
    _write_json(out_dir / "00_manifest.json", manifest)

    audit_cols = [
        "open", "high", "low", "close", "prior_breakout_high", "prior_breakout_low",
        "atr", "ema_fast", "ema_slow", "signal", "signal_reason", "stop",
        "trend_component", "breakout_depth_atr", "body_component",
        "close_location_component", "risk_mult", "feature_complete", "source_minutes",
    ]
    signals = features.loc[features["signal"].ne(0), [c for c in audit_cols if c in features.columns]].copy()
    signals.insert(0, "signal_bar_start", signals.index)
    signals.insert(1, "signal_available_time", signals.index + pd.Timedelta(minutes=15))
    signals.insert(2, "expected_entry_time", signals.index + pd.Timedelta(minutes=15))
    signals.to_csv(out_dir / "14_signal_audit.csv", index=False)
    if args.write_full_audit:
        features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / "15_full_feature_audit.csv")

    if not args.skip_full_report:
        emit_signal_report(base_trades, features, cfg, out_dir, strategy_name="ETH Portfolio V2 Trend Breakout V1")

    print("\n" + "=" * 92)
    print(f"{STRATEGY_ID} | decision={decision['decision']}")
    print(f"source_events={source_events:,} executed={len(base_trades):,} funnel={funnel.verdict}")
    print(
        f"return={_finite_or(base_summary.get('total_return_pct'), 0.0):+.2f}% "
        f"PF={_finite_or(base_summary.get('profit_factor'), 0.0):.3f} "
        f"MDD={_finite_or(base_summary.get('max_drawdown_pct'), 0.0):.2f}% "
        f"max_flat={_finite_or(base_summary.get('max_flat_days'), 0.0):.2f}d"
    )
    print(f"reports -> {out_dir.resolve()}")
    print("=" * 92 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
