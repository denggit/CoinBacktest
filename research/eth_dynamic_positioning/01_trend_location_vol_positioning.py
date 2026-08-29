#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RDPOS-01 — ETH Trend + Location + Volatility Dynamic Positioning.

This is deliberately *not* an entry/exit strategy.  Every four hours the
system re-evaluates two independent position sleeves from completed hourly
prices, then only adjusts when the desired-vs-current gap is economically
meaningful.  The research question is narrow:

    Does current price location improve a continuous trend+vol position state
    enough to beat the repository's prior trend-only continuous baseline?

Official trading window is 2023-01-01 through 2026-06-30.  2022 is warmup only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.research_common.eth_dynamic_positioning import (  # noqa: E402
    DynamicPositionConfig,
    build_state_frame,
    extract_sleeve_episodes,
    live_candidate_verdict,
    period_summary,
    scenario_configs,
    simulate_dynamic_positioning,
    summarize_account,
    top_day_removal,
    validate_hourly_ohlcv,
)
from src.data_feed.okx_derivatives_loader import OKXDerivativesLoader  # noqa: E402
from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402

try:  # Added in recent CoinBacktest patches; keep optional for older clones.
    from src.data_feed.binance_funding_archive_loader import BinanceFundingArchiveLoader  # type: ignore # noqa: E402
except ImportError:  # pragma: no cover
    BinanceFundingArchiveLoader = None  # type: ignore[assignment]


DEFAULT_OUTPUT = Path("data/reports/research/eth_dynamic_positioning/01_trend_location_vol_positioning")
DEFAULT_BINANCE_FUNDING = Path(
    "research/eth_ict_price_action_portfolio/ict_pa_v3/inputs/binance_ethusdt_funding.csv"
)
PRIOR_BASELINE = Path(
    "research/eth_market_process_portfolio/portfolio/clean_causal_v1/results/summary.csv"
)


def _inclusive_end(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if len(str(value).strip()) <= 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--warmup-start", default="2022-01-01")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--allow-fetch", action="store_true", help="Allow OKXDataLoader to backfill missing 1H candles.")
    p.add_argument("--funding-source", choices=["auto", "okx", "binance_proxy", "none"], default="auto")
    p.add_argument("--binance-funding-csv", default=str(DEFAULT_BINANCE_FUNDING))
    p.add_argument("--funding-timezone-offset-hours", type=float, default=8.0)
    p.add_argument("--out-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--write-state-audit", action="store_true")
    return p.parse_args(argv)


def _load_hourly(args: argparse.Namespace, warmup: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Load local hourly candles without treating a short tail gap as fatal.

    The repository's research window may extend beyond the locally prebuilt 1H
    cache.  Warmup coverage is mandatory; trailing coverage is allowed to end
    early when ``--allow-fetch`` is not requested.  ``main`` then caps the
    official trade_end to the last complete local 1H bar and records that fact
    in the report.  This avoids silently going online and avoids pretending the
    requested end date was actually tested.
    """
    kwargs: dict[str, Any] = {"symbol": args.symbol, "timeframe": "1H"}
    if args.data_dir:
        kwargs["db_dir"] = args.data_dir
    loader = OKXDataLoader(**kwargs)
    local = loader.load_local_data()
    if not local.empty:
        local = local.sort_index(kind="stable")
        local = local[~local.index.duplicated(keep="last")]
        selected = local.loc[(local.index >= warmup) & (local.index <= end)].copy()
    else:
        selected = pd.DataFrame()

    warmup_ok = not selected.empty and selected.index.min() <= warmup + pd.Timedelta(hours=2)
    tail_ok = not selected.empty and selected.index.max() >= end.floor("h") - pd.Timedelta(hours=2)

    if not warmup_ok:
        if not args.allow_fetch:
            raise RuntimeError(
                "Local 1H warmup coverage is insufficient. Run the project's prebuild first, or explicitly pass "
                "--allow-fetch. "
                f"Observed={selected.index.min() if not selected.empty else None} -> "
                f"{selected.index.max() if not selected.empty else None}"
            )
        selected = loader.fetch_data_by_date_range(str(warmup), str(end))
    elif not tail_ok and args.allow_fetch:
        selected = loader.fetch_data_by_date_range(str(warmup), str(end))
    elif not tail_ok:
        print(
            "[coverage] requested 1H tail is not local; using the last available local bar instead of fetching: "
            f"requested_end={end} local_end={selected.index.max()}",
            flush=True,
        )

    if selected.empty:
        raise RuntimeError("No hourly ETH data loaded")
    for col in ("open", "high", "low", "close", "volume"):
        selected[col] = pd.to_numeric(selected[col], errors="coerce")
    selected = selected.dropna(subset=["open", "high", "low", "close", "volume"])
    selected = selected.sort_index(kind="stable")
    if selected.index.tz is not None:
        selected.index = selected.index.tz_convert(None)
    return selected.loc[(selected.index >= warmup) & (selected.index <= end)].copy()


def _funding_coverage_ratio(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if frame.empty:
        return 0.0
    expected = max(1, int((end - start).total_seconds() // (8 * 3600)) + 1)
    return min(1.0, len(frame) / expected)


def _covers_funding(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    if frame.empty:
        return False
    return bool(
        frame.index.min() <= start + pd.Timedelta(days=2)
        and frame.index.max() >= end - pd.Timedelta(days=2)
        and _funding_coverage_ratio(frame, start, end) >= 0.90
    )


def _normalise_funding(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["funding_rate", "source"])
    out = frame.copy()
    if out.index.tz is not None:
        out.index = out.index.tz_convert(None)
    if "funding_rate" not in out.columns:
        raise RuntimeError("funding source is missing funding_rate")
    out["funding_rate"] = pd.to_numeric(out["funding_rate"], errors="coerce")
    out = out.dropna(subset=["funding_rate"]).sort_index(kind="stable")
    out = out[~out.index.duplicated(keep="last")]
    out["source"] = source
    return out


def load_funding(
    args: argparse.Namespace, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, object]]:
    if args.funding_source == "none":
        return pd.DataFrame(), {"source": "NONE", "complete": False, "note": "funding disabled explicitly"}

    if args.funding_source in {"auto", "okx"}:
        try:
            kwargs: dict[str, Any] = {"symbol": args.symbol}
            if args.data_dir:
                kwargs["data_dir"] = Path(args.data_dir)
            frame = _normalise_funding(OKXDerivativesLoader(**kwargs).load_funding_rates(start, end), "OKX_LOCAL")
            if _covers_funding(frame, start, end):
                return frame, {
                    "source": "OKX_LOCAL",
                    "complete": True,
                    "proxy": False,
                    "rows": len(frame),
                    "start": str(frame.index.min()),
                    "end": str(frame.index.max()),
                    "coverage_ratio": _funding_coverage_ratio(frame, start, end),
                }
            if args.funding_source == "okx":
                return pd.DataFrame(), {
                    "source": "OKX_INSUFFICIENT",
                    "complete": False,
                    "rows": len(frame),
                    "coverage_ratio": _funding_coverage_ratio(frame, start, end),
                }
        except Exception as exc:
            if args.funding_source == "okx":
                raise
            print(f"[funding] OKX local unavailable: {type(exc).__name__}: {exc}", flush=True)

    if args.funding_source in {"auto", "binance_proxy"} and BinanceFundingArchiveLoader is not None:
        path = Path(args.binance_funding_csv)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        frame = _normalise_funding(
            BinanceFundingArchiveLoader(path, timezone_offset_hours=args.funding_timezone_offset_hours).load(start, end),
            "BINANCE_ETHUSDT_PROXY",
        )
        if _covers_funding(frame, start, end):
            return frame, {
                "source": "BINANCE_ETHUSDT_PROXY",
                "complete": True,
                "proxy": True,
                "rows": len(frame),
                "start": str(frame.index.min()),
                "end": str(frame.index.max()),
                "coverage_ratio": _funding_coverage_ratio(frame, start, end),
                "path": str(path),
            }
        return pd.DataFrame(), {
            "source": "FUNDING_UNAVAILABLE",
            "complete": False,
            "proxy": True,
            "rows": len(frame),
            "coverage_ratio": _funding_coverage_ratio(frame, start, end),
            "path": str(path),
        }

    return pd.DataFrame(), {"source": "FUNDING_UNAVAILABLE", "complete": False}


def _load_prior_baseline() -> dict[str, object] | None:
    path = PROJECT_ROOT / PRIOR_BASELINE
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
        if frame.empty:
            return None
        row = frame.loc[frame["scenario"].astype(str).eq("base")]
        if row.empty:
            row = frame.iloc[[0]]
        r = row.iloc[0].to_dict()
        return {
            "source": str(path.relative_to(PROJECT_ROOT)),
            "scenario": r.get("scenario"),
            "total_return": r.get("total_return"),
            "cagr": r.get("cagr"),
            "max_drawdown": r.get("max_drawdown"),
            "calmar": r.get("calmar"),
            "annual_turnover": r.get("annual_turnover"),
            "note": "Existing repository reference only; not used to fit RDPOS-01 parameters.",
        }
    except Exception as exc:  # pragma: no cover
        return {"source": str(path), "error": f"{type(exc).__name__}: {exc}"}


def _scenario_table(
    bars: pd.DataFrame,
    base_cfg: DynamicPositionConfig,
    funding: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    frames: dict[str, pd.DataFrame] = {}
    states: dict[str, pd.DataFrame] = {}
    scenarios = scenario_configs(base_cfg)
    for number, (name, cfg) in enumerate(scenarios, start=1):
        print(f"[scenario {number}/{len(scenarios)}] {name}", flush=True)
        state = build_state_frame(bars, cfg)
        replay = simulate_dynamic_positioning(state, cfg, funding=funding)
        summary = summarize_account(replay)
        rows.append({"scenario": name, **summary})
        frames[name] = replay
        states[name] = state
    return pd.DataFrame(rows), frames, states


def _write_report(
    out: Path,
    *,
    summary: dict[str, object],
    funding_meta: dict[str, object],
    verdict: dict[str, object],
    prior: dict[str, object] | None,
    top_days: dict[str, object],
    scenarios: pd.DataFrame,
) -> None:
    lines = [
        "# ETH Dynamic Positioning RDPOS-01",
        "",
        "## Research question",
        "",
        "Can **trend + current location + volatility** support an economically useful continuous ETH position state, "
        "without converting the problem back into entry/TP/SL trades?",
        "",
        "## Base result",
        "",
    ]
    for key in (
        "total_return", "cagr", "max_drawdown", "calmar", "profit_factor_hourly",
        "positive_month_rate", "max_consecutive_loss_days", "max_flat_days_below_0_1x",
        "mean_abs_net_exposure", "max_abs_net_exposure", "mean_gross_exposure", "max_gross_exposure",
        "annual_turnover", "position_adjustments", "adjustments_per_day",
        "total_trading_cost_return", "total_funding_return", "trading_cost_share_of_positive_gross",
    ):
        lines.append(f"- **{key}**: {summary.get(key)}")
    lines += [
        "",
        "## Funding coverage",
        "",
        "```json",
        json.dumps(funding_meta, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "If OKX funding coverage is incomplete (or only a Binance proxy is available), the base result is **not live-valid** even if PnL is positive. "
        "The 5% and 10% annual carry scenarios are stress diagnostics, not fabricated historical funding.",
        "",
        "## Promotion verdict",
        "",
        "```json",
        json.dumps(verdict, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Why this is different from the prior price-only continuous portfolio",
        "",
        "The repository already contains a 7/30/90-day trend + volatility continuous portfolio. RDPOS-01 does not "
        "pretend that idea is new. The incremental hypothesis is **location-aware sizing** plus an explicit no-trade "
        "band and independent medium/slow sleeves. `trend_only_no_location` is therefore the key ablation row.",
        "",
        "```json",
        json.dumps(prior, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Top-day dependence",
        "",
        "```json",
        json.dumps(top_days, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## Scenario snapshot",
        "",
        scenarios.to_markdown(index=False) if not scenarios.empty else "No scenario rows.",
        "",
        "## Frozen interpretation rule",
        "",
        "- Do **not** select the best scenario as a tuned strategy.",
        "- `base_location` must beat `trend_only_no_location` economically, not merely statistically, or the location hypothesis fails.",
        "- CAGR <= |MDD| fails the user's minimum capital-efficiency requirement.",
        "- 2x cost must remain positive before promotion.",
        "- Funding must be historically covered before any live claim.",
        "- 2022 is warmup only; no 2022 return is allowed in the official result.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    warmup = pd.Timestamp(args.warmup_start)
    start = pd.Timestamp(args.start_date)
    end = _inclusive_end(args.end_date)
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.symbol} 1H {warmup} -> {end}", flush=True)
    bars = _load_hourly(args, warmup, end)
    quality = validate_hourly_ohlcv(bars)
    pd.DataFrame([quality]).to_csv(out / "data_quality.csv", index=False, encoding="utf-8-sig")
    if not quality["ready"]:
        raise RuntimeError(f"OHLCV quality gate failed: {quality}")
    if float(quality["missing_timestamp_rate"]) > 0.01:
        raise RuntimeError(f"Hourly missing timestamp rate exceeds 1%: {quality['missing_timestamp_rate']:.4%}")
    print(f"[data] rows={len(bars):,} {bars.index.min()} -> {bars.index.max()}", flush=True)

    actual_end = min(end, pd.Timestamp(bars.index.max()))
    if actual_end <= start:
        raise RuntimeError(f"Local 1H data ends before the official trade window: start={start} actual_end={actual_end}")
    if actual_end < end:
        print(
            f"[coverage] official backtest end capped to local coverage: requested={end} actual={actual_end}",
            flush=True,
        )
    cfg = DynamicPositionConfig(
        symbol=args.symbol,
        warmup_start=str(warmup),
        trade_start=str(start),
        trade_end=str(actual_end),
    )
    cfg.validate()
    with (out / "requested_vs_actual_window.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "requested_warmup_start": str(warmup),
                "requested_trade_start": str(start),
                "requested_trade_end": str(end),
                "actual_trade_end": str(actual_end),
                "end_truncated_to_local_coverage": bool(actual_end < end),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    funding, funding_meta = load_funding(args, start, actual_end)
    print(
        f"[funding] source={funding_meta.get('source')} complete={funding_meta.get('complete')} rows={len(funding):,}",
        flush=True,
    )

    scenario_df, frames, states = _scenario_table(bars, cfg, funding)
    scenario_df.to_csv(out / "scenario_summary.csv", index=False, encoding="utf-8-sig")
    base = frames["base_location"]
    base_state = states["base_location"]
    summary = summarize_account(base)
    yearly = period_summary(base, "Y")
    monthly = period_summary(base, "M")
    episodes = extract_sleeve_episodes(base)
    top_days = top_day_removal(base, 10)
    yearly.to_csv(out / "yearly.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(out / "monthly.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(out / "sleeve_episodes.csv", index=False, encoding="utf-8-sig")
    base.to_csv(out / "equity_hourly.csv", encoding="utf-8-sig")

    audit_cols = [
        "open", "high", "low", "close", "available_time", "decision_close", "state_ready",
        "medium_trend", "slow_trend", "medium_anchor", "slow_anchor", "medium_extension", "slow_extension",
        "medium_range_location", "slow_range_location", "medium_location_multiplier", "slow_location_multiplier",
        "annual_vol", "risk_scalar", "medium_desired_close", "slow_desired_close",
    ]
    decision_audit = base_state.loc[base_state["decision_close"], [c for c in audit_cols if c in base_state.columns]].copy()
    decision_audit.to_csv(out / "decision_audit.csv", encoding="utf-8-sig")
    if args.write_state_audit:
        base_state[[c for c in audit_cols if c in base_state.columns]].to_csv(out / "full_state_audit.csv", encoding="utf-8-sig")

    prior = _load_prior_baseline()
    with (out / "funding_coverage.json").open("w", encoding="utf-8") as f:
        json.dump(funding_meta, f, ensure_ascii=False, indent=2, default=str)
    with (out / "prior_baseline_reference.json").open("w", encoding="utf-8") as f:
        json.dump(prior, f, ensure_ascii=False, indent=2, default=str)
    with (out / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, ensure_ascii=False, indent=2, default=str)

    verdict = live_candidate_verdict(
        summary,
        yearly,
        funding_complete=bool(funding_meta.get("complete", False) and not funding_meta.get("proxy", False)),
    )
    cost2 = scenario_df.loc[scenario_df["scenario"].eq("cost_2x")]
    trend_only = scenario_df.loc[scenario_df["scenario"].eq("trend_only_no_location")]
    verdict["checks"]["cost_2x_positive"] = bool(not cost2.empty and float(cost2.iloc[0]["total_return"]) > 0)
    verdict["checks"]["location_improves_calmar"] = bool(
        not trend_only.empty and float(summary["calmar"]) > float(trend_only.iloc[0]["calmar"])
    )
    verdict["pass"] = bool(all(verdict["checks"].values()))
    verdict["top_day_removal"] = top_days
    verdict["decision"] = "PROMOTE_TO_NEXT_STAGE" if verdict["pass"] else "STOP_OR_REVISE_HYPOTHESIS"
    with (out / "verdict.json").open("w", encoding="utf-8") as f:
        json.dump(verdict, f, ensure_ascii=False, indent=2, default=str)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "funding": funding_meta, "verdict": verdict}, f, ensure_ascii=False, indent=2, default=str)

    _write_report(
        out,
        summary=summary,
        funding_meta=funding_meta,
        verdict=verdict,
        prior=prior,
        top_days=top_days,
        scenarios=scenario_df,
    )

    print("=" * 100)
    print("ETH DYNAMIC POSITIONING RDPOS-01")
    print("=" * 100)
    for key in (
        "total_return", "cagr", "max_drawdown", "calmar", "positive_month_rate",
        "annual_turnover", "adjustments_per_day", "total_trading_cost_return", "total_funding_return",
    ):
        print(f"{key:>34}: {summary.get(key)}")
    print(f"{'funding_source':>34}: {funding_meta.get('source')}")
    print(f"{'funding_complete':>34}: {funding_meta.get('complete')}")
    print(f"{'decision':>34}: {verdict['decision']}")
    print(f"{'output':>34}: {out}")
    print("=" * 100)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
