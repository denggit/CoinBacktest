#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent raw-bar replay for R21 daily channel trades.

This validator deliberately does not import the R21 simulator or feature
builder.  It reconstructs daily channels and Wilder ATR from data loaded
through ``src.data_feed`` and reconciles every emitted trade path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

DEFAULT_REPORT_DIR = Path("data/reports/research/ict/mss2/r21_canonical_daily_channel_trend")
EXPERIMENT_ID = "ETH_ICT_MSS2_CANONICAL_DAILY_CHANNEL_TREND_R21"
EDGE_ID = "RESEARCH_ONLY_DAILY_DONCHIAN_LONG_SHORT"
TITLE = "ETH ICT MSS2 R21 Canonical Daily Channel Trend Following"
MODEL_WINDOWS = {"D20_X10": (20, 10), "D55_X20": (55, 20)}
SPLIT_ENDS = {"discovery": pd.Timestamp("2025-01-01"), "validation": pd.Timestamp("2025-07-01")}
STOP_ATR = 2.0
TOL = 1e-10


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--warmup-start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-06-30 23:59:59")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--skip-review-pack", action="store_true")
    return parser.parse_args(argv)


def _daily_from_raw(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    if not isinstance(frame.index, pd.DatetimeIndex):
        timestamp = "timestamp" if "timestamp" in frame.columns else "datetime"
        frame.index = pd.to_datetime(frame[timestamp])
    frame.index = pd.to_datetime(frame.index)
    frame = frame.sort_index()
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    daily = frame.resample("1D", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")
    ).dropna()
    previous_close = daily["close"].shift(1)
    true_range = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - previous_close).abs(),
            (daily["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    daily["atr20"] = true_range.ewm(alpha=1.0 / 20.0, adjust=False, min_periods=20).mean()
    for entry_window, exit_window in MODEL_WINDOWS.values():
        daily[f"entry_high_{entry_window}"] = daily["high"].rolling(entry_window).max().shift(1)
        daily[f"entry_low_{entry_window}"] = daily["low"].rolling(entry_window).min().shift(1)
        daily[f"exit_high_{exit_window}"] = daily["high"].rolling(exit_window).max().shift(1)
        daily[f"exit_low_{exit_window}"] = daily["low"].rolling(exit_window).min().shift(1)
    return daily


def _first_stop(
    bars: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    direction: int,
    stop: float,
) -> tuple[pd.Timestamp, float] | None:
    section = bars.loc[(bars.index >= start) & (bars.index < end)]
    hit = section["low"].le(stop) if direction > 0 else section["high"].ge(stop)
    if section.empty or not bool(hit.any()):
        return None
    timestamp = pd.Timestamp(hit.index[np.flatnonzero(hit.to_numpy(dtype=bool))[0]])
    raw_open = float(section.loc[timestamp, "open"])
    price = min(raw_open, stop) if direction > 0 else max(raw_open, stop)
    return timestamp, price


def _exit_signal(daily: pd.DataFrame, day: pd.Timestamp, direction: int, window: int) -> bool:
    row = daily.loc[day]
    level = float(row[f"exit_low_{window}"] if direction > 0 else row[f"exit_high_{window}"])
    return bool(float(row["close"]) < level) if direction > 0 else bool(float(row["close"]) > level)


def _prior_executable_exit(
    daily: pd.DataFrame,
    *,
    entry_time: pd.Timestamp,
    outcome_time: pd.Timestamp,
    direction: int,
    exit_window: int,
) -> pd.Timestamp | None:
    last_signal_day = outcome_time.normalize() - pd.Timedelta(days=1)
    signal_days = daily.loc[(daily.index >= entry_time.normalize()) & (daily.index <= last_signal_day)].index
    for day in signal_days:
        if _exit_signal(daily, pd.Timestamp(day), direction, exit_window):
            execution = pd.Timestamp(day) + pd.Timedelta(days=1)
            if execution <= outcome_time:
                return execution
    return None


def replay(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    raw = bars.copy()
    raw.index = pd.to_datetime(raw.index)
    raw = raw.sort_index()
    daily = _daily_from_raw(raw)
    rows: list[dict[str, object]] = []
    for trade in trades.itertuples(index=False):
        model = str(trade.model)
        entry_window, exit_window = MODEL_WINDOWS[model]
        direction = int(trade.trade_direction)
        signal_day = pd.Timestamp(trade.entry_signal_bar_time)
        entry_time = pd.Timestamp(trade.entry_time)
        exit_time = pd.Timestamp(trade.exit_time) if pd.notna(trade.exit_time) else pd.NaT
        stop = float(trade.initial_stop_price)
        split_end = SPLIT_ENDS[str(trade.research_split)]
        entry_row = daily.loc[signal_day]
        entry_level = float(
            entry_row[f"entry_high_{entry_window}"] if direction > 0 else entry_row[f"entry_low_{entry_window}"]
        )
        signal_ok = bool(float(entry_row["close"]) > entry_level) if direction > 0 else bool(float(entry_row["close"]) < entry_level)
        raw_entry_open = float(raw.loc[entry_time, "open"])
        atr = float(entry_row["atr20"])
        expected_stop = raw_entry_open - direction * STOP_ATR * atr
        checks: dict[str, bool] = {
            "entry_next_day": entry_time == signal_day + pd.Timedelta(days=1),
            "entry_signal_true": signal_ok,
            "entry_raw_open": abs(float(trade.entry_price) - raw_entry_open) <= TOL,
            "atr_reconstructed": abs(float(trade.atr20_at_signal) - atr) <= TOL,
            "stop_reconstructed": abs(stop - expected_stop) <= TOL,
            "outcome_replayed": False,
            "no_earlier_channel_exit": False,
            "split_boundary_exact": True,
        }
        expected_time: pd.Timestamp | pd.NaT = pd.NaT
        expected_price = np.nan

        if str(trade.exit_reason) == "INITIAL_ATR_STOP":
            touch = _first_stop(
                raw,
                start=entry_time,
                end=exit_time + pd.Timedelta(minutes=1),
                direction=direction,
                stop=stop,
            )
            if touch is not None:
                expected_time, expected_price = touch
                checks["outcome_replayed"] = expected_time == exit_time and abs(expected_price - float(trade.exit_price)) <= TOL
            checks["no_earlier_channel_exit"] = _prior_executable_exit(
                daily,
                entry_time=entry_time,
                outcome_time=exit_time,
                direction=direction,
                exit_window=exit_window,
            ) is None
        elif str(trade.exit_reason) == "CHANNEL_EXIT_NEXT_OPEN":
            signal_exit_day = exit_time - pd.Timedelta(days=1)
            expected_time = exit_time
            expected_price = float(raw.loc[exit_time, "open"])
            no_prior_stop = _first_stop(
                raw,
                start=entry_time,
                end=exit_time,
                direction=direction,
                stop=stop,
            ) is None
            checks["outcome_replayed"] = (
                _exit_signal(daily, signal_exit_day, direction, exit_window)
                and no_prior_stop
                and abs(expected_price - float(trade.exit_price)) <= TOL
            )
            first_exit = _prior_executable_exit(
                daily,
                entry_time=entry_time,
                outcome_time=exit_time,
                direction=direction,
                exit_window=exit_window,
            )
            checks["no_earlier_channel_exit"] = first_exit == exit_time
            checks["split_boundary_exact"] = exit_time < split_end
        elif str(trade.exit_reason) == "SPLIT_BOUNDARY_CENSORED":
            expected_time = split_end
            no_stop = _first_stop(raw, start=entry_time, end=split_end, direction=direction, stop=stop) is None
            first_exit = _prior_executable_exit(
                daily,
                entry_time=entry_time,
                outcome_time=split_end - pd.Timedelta(minutes=1),
                direction=direction,
                exit_window=exit_window,
            )
            checks["outcome_replayed"] = no_stop and first_exit is None and pd.isna(trade.exit_time)
            checks["no_earlier_channel_exit"] = first_exit is None
            checks["split_boundary_exact"] = str(trade.path_status) == "boundary_censored"

        failed = [name for name, passed in checks.items() if not passed]
        rows.append(
            {
                "trade_id": trade.trade_id,
                "model": model,
                "research_split": trade.research_split,
                "direction": trade.direction,
                "exit_reason": trade.exit_reason,
                **{name: int(passed) for name, passed in checks.items()},
                "expected_outcome_time": expected_time,
                "expected_outcome_price": expected_price,
                "violations": len(failed),
                "failed_checks": "PASS" if not failed else ",".join(failed),
            }
        )
    return pd.DataFrame(rows)


def summarize(replayed: pd.DataFrame) -> pd.DataFrame:
    checks = [
        "entry_next_day",
        "entry_signal_true",
        "entry_raw_open",
        "atr_reconstructed",
        "stop_reconstructed",
        "outcome_replayed",
        "no_earlier_channel_exit",
        "split_boundary_exact",
    ]
    return pd.DataFrame(
        [
            {
                "check": check,
                "rows_checked": len(replayed),
                "violations": int(replayed[check].eq(0).sum()),
                "status": "PASS" if bool(replayed[check].eq(1).all()) else "FAIL",
            }
            for check in checks
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = Path(args.report_dir)
    trades = pd.read_csv(report_dir / "03_trade_paths.csv.gz")
    for column in ("entry_signal_bar_time", "entry_time", "exit_time"):
        trades[column] = pd.to_datetime(trades[column])
    bars = OKXDataLoader(symbol=args.symbol, timeframe="1m").fetch_data_by_date_range(
        args.warmup_start_date, args.end_date
    )
    replayed = replay(trades, bars)
    audit = summarize(replayed)
    replayed.to_csv(report_dir / "08_independent_trade_replay.csv", index=False, float_format="%.17g")
    audit.to_csv(report_dir / "09_independent_replay_audit.csv", index=False)
    if not args.skip_review_pack:
        finalize_research_report(
            report_dir,
            experiment_id=EXPERIMENT_ID,
            edge_id=EDGE_ID,
            title=TITLE,
        )
    print(audit.to_string(index=False))
    if int(audit["violations"].sum()) != 0:
        print(replayed.loc[replayed["violations"].gt(0)].to_string(index=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
