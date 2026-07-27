#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R01: causal ETH order-flow process event study on tzplus8 1m trade bars.

This is a mechanism screen, not a finished strategy backtest. It evaluates four
pre-declared processes using closed 1m trade bars and next-bar-open execution:

- buy_continuation_long
- sell_continuation_short
- sell_absorption_long
- buy_absorption_short

Data access is exclusively through ``src.data_feed.OKXTradeBarLoader``. The
study loads one calendar year at a time with causal warmup/tail overlap, so the
multi-year trade-bar database is never loaded into memory all at once.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.trade_bar_orderflow import (  # noqa: E402
    build_trade_bar_orderflow_features,
    trade_bar_field_coverage,
    validate_trade_bar_orderflow,
)


@dataclass(frozen=True)
class StudyConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "1m"
    start: str = "2023-01-01"
    end: str = "2026-06-30 23:59:59"
    baseline_window: int = 240
    warmup_days: int = 3
    tail_bars: int = 61
    cooldown_bars: int = 30
    round_trip_cost: float = 0.0011
    horizons: tuple[int, ...] = (5, 15, 30, 60)


PROCESS_ORDER = (
    "buy_continuation_long",
    "sell_continuation_short",
    "sell_absorption_long",
    "buy_absorption_short",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--start-date", default="2023-01-01")
    parser.add_argument("--end-date", default="2026-06-30 23:59:59")
    parser.add_argument("--baseline-window", type=int, default=240)
    parser.add_argument("--cooldown-bars", type=int, default=30)
    parser.add_argument("--round-trip-cost", type=float, default=0.0011)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=_PROJECT_ROOT
        / "data"
        / "reports"
        / "research"
        / "eth_market_process_portfolio"
        / "order_flow"
        / "01_order_flow_process_event_study",
    )
    parser.add_argument("--no-event-export", action="store_true")
    return parser.parse_args(argv)


def _year_windows(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = pd.Timestamp(year=start.year, month=1, day=1)
    while cursor <= end:
        left = max(start, cursor)
        right = min(end, pd.Timestamp(year=cursor.year, month=12, day=31, hour=23, minute=59, second=59))
        if left <= right:
            yield left, right
        cursor = pd.Timestamp(year=cursor.year + 1, month=1, day=1)


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    return (pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )


def build_process_masks(features: pd.DataFrame) -> dict[str, pd.Series]:
    """Return pre-declared causal process masks.

    Thresholds are intentionally coarse and mechanism-led. R01 does not search a
    parameter grid. Later research may test a small neighbourhood only after a
    process shows cross-period value.
    """
    f = features
    buy_activity = _safe_divide(f["buy_notional"], f["buy_notional"].shift(1).rolling(240, min_periods=60).median())
    large_buy_share = _safe_divide(f["large_buy_notional"], f["buy_notional"])

    buy_continuation = (
        (buy_activity >= 2.0)
        & (f["delta_ratio_3"] >= 0.22)
        & (f["large_delta_ratio_3"] >= 0.10)
        & (f["price_return_3"] >= 0.0010)
        & (f["close_pos"] >= 0.68)
        & (f["notional_ratio_base"] >= 1.6)
    )
    sell_continuation = (
        (f["sell_notional_ratio_base"] >= 2.0)
        & (f["delta_ratio_3"] <= -0.22)
        & (f["large_delta_ratio_3"] <= -0.10)
        & (f["price_return_3"] <= -0.0010)
        & (f["close_pos"] <= 0.32)
        & (f["notional_ratio_base"] >= 1.6)
    )

    # Aggressive pressure is extreme but price damage is small and the close
    # reclaims the bar. These are absorption candidates, not generic wick bars.
    sell_absorption = (
        (f["sell_notional_ratio_base"] >= 2.4)
        & (f["delta_ratio_3"] <= -0.20)
        & (f["large_sell_share_of_sell"] >= 0.08)
        & (f["down_move_norm"] <= 1.15)
        & (f["close_pos"] >= 0.58)
        & (f["lower_wick_frac"] >= 0.18)
        & (f["absorption_score"] >= 3.0)
    )

    bar_range = (f["high"] - f["low"]).clip(lower=f["close"].abs() * 1e-9)
    upper_wick_frac = ((f["high"] - f[["open", "close"]].max(axis=1)).clip(lower=0.0) / bar_range).clip(0.0, 1.0)
    up_move_norm = pd.to_numeric(f["up_move_norm"], errors="coerce")
    buy_absorption_score = (
        buy_activity.fillna(1.0)
        + 0.40 * f["trades_ratio_base"].fillna(1.0)
        + 0.50 * large_buy_share.fillna(0.0)
        - up_move_norm.fillna(0.0)
        + 0.50 * (1.0 - f["close_pos"].fillna(0.5))
    )
    buy_absorption = (
        (buy_activity >= 2.4)
        & (f["delta_ratio_3"] >= 0.20)
        & (large_buy_share >= 0.08)
        & (up_move_norm <= 1.15)
        & (f["close_pos"] <= 0.42)
        & (upper_wick_frac >= 0.18)
        & (buy_absorption_score >= 3.0)
    )

    return {
        "buy_continuation_long": buy_continuation.fillna(False),
        "sell_continuation_short": sell_continuation.fillna(False),
        "sell_absorption_long": sell_absorption.fillna(False),
        "buy_absorption_short": buy_absorption.fillna(False),
    }


def _apply_cooldown(mask: pd.Series, cooldown_bars: int) -> pd.Series:
    arr = mask.to_numpy(dtype=bool)
    keep = np.zeros(len(arr), dtype=bool)
    next_allowed = 0
    for pos in np.flatnonzero(arr):
        if pos >= next_allowed:
            keep[pos] = True
            next_allowed = pos + max(1, int(cooldown_bars))
    return pd.Series(keep, index=mask.index)


def _events_from_masks(features: pd.DataFrame, masks: dict[str, pd.Series], cooldown_bars: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    feature_cols = [
        "open", "high", "low", "close", "notional_ratio_base", "buy_notional_ratio_base",
        "sell_notional_ratio_base", "trades_ratio_base", "delta_ratio", "delta_ratio_3",
        "large_delta_ratio_3", "price_return_3", "close_pos", "lower_wick_frac",
        "absorption_score", "large_trade_share", "large_sell_share_of_sell",
    ]
    for process in PROCESS_ORDER:
        selected = _apply_cooldown(masks[process], cooldown_bars)
        if not selected.any():
            continue
        part = features.loc[selected, [c for c in feature_cols if c in features.columns]].copy()
        part.insert(0, "signal_time", part.index)
        part.insert(1, "process", process)
        part.insert(2, "side", 1 if process.endswith("long") else -1)
        rows.append(part.reset_index(drop=True))
    if not rows:
        return pd.DataFrame(columns=["signal_time", "process", "side"])
    return pd.concat(rows, ignore_index=True).sort_values(["signal_time", "process"]).reset_index(drop=True)


def _attach_outcomes(events: pd.DataFrame, bars: pd.DataFrame, horizons: tuple[int, ...], cost: float) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    out = events.copy()
    idx = bars.index
    pos = idx.get_indexer(pd.DatetimeIndex(out["signal_time"]))
    out["signal_bar_pos"] = pos
    valid = pos >= 0
    opens = pd.to_numeric(bars["open"], errors="coerce").to_numpy(float)
    highs = pd.to_numeric(bars["high"], errors="coerce").to_numpy(float)
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(float)
    closes = pd.to_numeric(bars["close"], errors="coerce").to_numpy(float)
    sides = out["side"].to_numpy(int)

    out["entry_time"] = pd.NaT
    out["entry_price"] = np.nan
    entry_pos = pos + 1
    good_entry = valid & (entry_pos < len(bars))
    if good_entry.any():
        out.loc[good_entry, "entry_time"] = idx[entry_pos[good_entry]].to_numpy()
        out.loc[good_entry, "entry_price"] = opens[entry_pos[good_entry]]

    for h in horizons:
        gross = np.full(len(out), np.nan)
        future_pos = pos + int(h)
        good = good_entry & (future_pos < len(bars))
        gross[good] = (closes[future_pos[good]] / opens[entry_pos[good]] - 1.0) * sides[good]
        out[f"ret_h{h}_gross"] = gross
        out[f"ret_h{h}_net"] = gross - float(cost)

    h = max(horizons)
    mfe = np.full(len(out), np.nan)
    mae = np.full(len(out), np.nan)
    for i in np.flatnonzero(good_entry):
        p0 = entry_pos[i]
        p1 = min(len(bars), pos[i] + h + 1)
        if p1 <= p0:
            continue
        entry = opens[p0]
        if not np.isfinite(entry) or entry <= 0:
            continue
        if sides[i] == 1:
            mfe[i] = np.nanmax(highs[p0:p1]) / entry - 1.0
            mae[i] = np.nanmin(lows[p0:p1]) / entry - 1.0
        else:
            mfe[i] = entry / np.nanmin(lows[p0:p1]) - 1.0
            mae[i] = entry / np.nanmax(highs[p0:p1]) - 1.0
    out[f"mfe_h{h}"] = mfe
    out[f"mae_h{h}"] = mae
    out["year"] = pd.to_datetime(out["signal_time"]).dt.year
    out["month"] = pd.to_datetime(out["signal_time"]).dt.to_period("M").astype(str)
    out["causal_entry_flag"] = pd.to_datetime(out["entry_time"]) > pd.to_datetime(out["signal_time"])
    return out


def _profit_factor(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    gains = x[x > 0].sum()
    losses = -x[x < 0].sum()
    if losses <= 0:
        return math.inf if gains > 0 else math.nan
    return float(gains / losses)


def _summarize(events: pd.DataFrame, group_cols: list[str], horizons: tuple[int, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = [((), events)] if not group_cols else events.groupby(group_cols, dropna=False, sort=True)
    for keys, frame in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {name: value for name, value in zip(group_cols, keys)}
        months = max(1, frame["month"].nunique()) if not frame.empty else 1
        for h in horizons:
            ret = pd.to_numeric(frame[f"ret_h{h}_net"], errors="coerce").dropna()
            row = dict(base)
            row.update(
                {
                    "horizon": h,
                    "events": int(len(ret)),
                    "events_per_month": float(len(ret) / months),
                    "mean_net": float(ret.mean()) if len(ret) else np.nan,
                    "median_net": float(ret.median()) if len(ret) else np.nan,
                    "win_rate": float((ret > 0).mean()) if len(ret) else np.nan,
                    "profit_factor": _profit_factor(ret),
                    "t_stat": float(ret.mean() / (ret.std(ddof=1) / math.sqrt(len(ret)))) if len(ret) > 1 and ret.std(ddof=1) > 0 else np.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _write_markdown(
    path: Path,
    cfg: StudyConfig,
    overview: pd.DataFrame,
    yearly: pd.DataFrame,
    coverage: pd.DataFrame,
    total_events: int,
) -> None:
    lines = [
        "# R01 Order-Flow Process Event Study",
        "",
        "## Scope",
        "",
        f"- Symbol: `{cfg.symbol}`",
        f"- Research window: `{cfg.start}` to `{cfg.end}`",
        "- Source: `OKXTradeBarLoader`, timezone-aligned local `tzplus8` cache",
        "- Signal: closed 1m bar",
        "- Entry: next 1m bar open",
        f"- Round-trip cost deducted: `{cfg.round_trip_cost:.4%}`",
        f"- Cooldown: `{cfg.cooldown_bars}` bars per process",
        f"- Total events: `{total_events:,}`",
        "",
        "This is a mechanism screen, not a final TP/SL strategy backtest. No parameter grid was searched.",
        "",
        "## Process definitions",
        "",
        "- `buy_continuation_long`: intense aggressive buying, aligned large flow and effective upward price response.",
        "- `sell_continuation_short`: symmetric aggressive selling continuation.",
        "- `sell_absorption_long`: extreme aggressive selling with limited price damage and a reclaimed close.",
        "- `buy_absorption_short`: symmetric buy-pressure absorption candidate.",
        "",
        "## Overall results",
        "",
    ]
    lines.append(overview.to_markdown(index=False) if not overview.empty else "No events.")
    lines.extend(["", "## Yearly results", ""])
    lines.append(yearly.to_markdown(index=False) if not yearly.empty else "No yearly results.")
    lines.extend(["", "## Trade-bar field coverage", ""])
    lines.append(coverage.to_markdown(index=False) if not coverage.empty else "No coverage rows.")
    lines.extend(
        [
            "",
            "## Promotion rule",
            "",
            "A process is not promoted merely because one horizon is positive. It must clear cost across multiple years, have adequate frequency, stable neighbouring definitions, and later pass realistic TP/SL replay, delay stress and holdout checks.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    cfg = StudyConfig(
        symbol=args.symbol,
        start=args.start_date,
        end=args.end_date,
        baseline_window=int(args.baseline_window),
        cooldown_bars=int(args.cooldown_bars),
        round_trip_cost=float(args.round_trip_cost),
    )
    start = pd.Timestamp(cfg.start)
    end = pd.Timestamp(cfg.end)
    loader = OKXTradeBarLoader(
        symbol=cfg.symbol,
        timeframe=cfg.timeframe,
        align_with_okx_loader_timezone=True,
    )

    print("[run] ETH Market Process Portfolio - Order Flow R01")
    print(f"[window] {start} -> {end} timezone=tzplus8 source=OKXTradeBarLoader")
    print(f"[cost] round_trip={cfg.round_trip_cost:.4%} entry=next_open")

    all_events: list[pd.DataFrame] = []
    coverage: pd.DataFrame | None = None
    windows = list(_year_windows(start, end))
    for number, (left, right) in enumerate(windows, start=1):
        load_left = left - pd.Timedelta(days=cfg.warmup_days)
        load_right = right + pd.Timedelta(minutes=cfg.tail_bars)
        print(f"[chunk {number}/{len(windows)}] load {load_left} -> {load_right}")
        bars = loader.fetch_data_by_date_range(
            load_left,
            load_right,
            cvd_mode="range",
            build_missing=False,
        )
        if bars.empty:
            raise RuntimeError(f"no local trade bars for chunk {left} -> {right}")
        if coverage is None:
            coverage = validate_trade_bar_orderflow(bars)
        print(f"[chunk {number}/{len(windows)}] rows={len(bars):,} features")
        features = build_trade_bar_orderflow_features(bars, baseline_window=cfg.baseline_window)
        masks = build_process_masks(features)
        events = _events_from_masks(features, masks, cfg.cooldown_bars)
        events = events[(events["signal_time"] >= left) & (events["signal_time"] <= right)].copy()
        events = _attach_outcomes(events, features, cfg.horizons, cfg.round_trip_cost)
        all_events.append(events)
        counts = events.groupby("process").size().to_dict() if not events.empty else {}
        print(f"[chunk {number}/{len(windows)}] events={len(events):,} by_process={counts}")
        del bars, features, masks, events

    final_events = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
    if not final_events.empty:
        final_events = final_events.sort_values(["signal_time", "process"]).reset_index(drop=True)
    overview = _summarize(final_events, ["process"], cfg.horizons) if not final_events.empty else pd.DataFrame()
    yearly = _summarize(final_events, ["process", "year"], cfg.horizons) if not final_events.empty else pd.DataFrame()
    side = _summarize(final_events, ["side"], cfg.horizons) if not final_events.empty else pd.DataFrame()

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    overview.to_csv(report_dir / "overview.csv", index=False)
    yearly.to_csv(report_dir / "yearly.csv", index=False)
    side.to_csv(report_dir / "side.csv", index=False)
    (coverage if coverage is not None else pd.DataFrame()).to_csv(report_dir / "field_coverage.csv", index=False)
    if not args.no_event_export:
        final_events.to_csv(report_dir / "events.csv", index=False)
    (report_dir / "run_config.json").write_text(
        json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_markdown(
        report_dir / "report.md",
        cfg,
        overview,
        yearly,
        coverage if coverage is not None else pd.DataFrame(),
        len(final_events),
    )

    print(f"[done] events={len(final_events):,}")
    if not overview.empty:
        print(overview.to_string(index=False))
    print(f"[report] {report_dir / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
