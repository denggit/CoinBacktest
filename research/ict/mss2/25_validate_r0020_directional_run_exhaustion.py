#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent raw-source replay for R25."""
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
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader  # noqa: E402

DEFAULT_OUT_DIR = "data/reports/research/ict/mss2/r25_r0020_directional_run_exhaustion"
COST = 0.0011
EMBARGO = pd.Timestamp("2025-07-01")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--range-db-name", default="okx_range_bars.db")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    return p.parse_args(argv)


def _range_checks(raw: pd.DataFrame, events: pd.DataFrame) -> dict[str, int]:
    bars = raw.reset_index(drop=True).copy()
    bars["start_ts"] = pd.to_datetime(bars["start_ts"], errors="coerce")
    bars["end_ts"] = pd.to_datetime(bars["end_ts"], errors="coerce")
    bars = bars.loc[bars["end_ts"].lt(EMBARGO)].sort_values(["end_ts", "bar_id"], kind="mergesort").reset_index(drop=True)
    for name in ("open", "high", "low", "close", "direction"):
        bars[name] = pd.to_numeric(bars[name], errors="coerce")
    valid = (
        bars["bar_id"].notna() & ~bars["bar_id"].duplicated(keep=False)
        & bars["start_ts"].notna() & bars["end_ts"].notna()
        & bars["start_ts"].le(bars["end_ts"])
        & bars["direction"].isin([-1, 1])
        & np.isfinite(bars[["open", "high", "low", "close"]].to_numpy(dtype=float)).all(axis=1)
        & (bars[["open", "high", "low", "close"]].to_numpy(dtype=float) > 0).all(axis=1)
    )
    id_to_pos = pd.Series(np.arange(len(bars)), index=bars["bar_id"].astype("int64")).to_dict()
    violations = {
        "event_missing_raw_bar": 0,
        "event_noncontiguous_run": 0,
        "event_direction_sequence": 0,
        "event_not_maximal": 0,
        "event_signal_time": 0,
        "event_target_formula": 0,
        "event_stop_formula": 0,
        "event_eligibility_formula": 0,
    }
    for event in events.itertuples(index=False):
        ids = (int(event.run_start_bar_id), int(event.run_end_bar_id), int(event.confirmation_bar_id))
        if any(value not in id_to_pos for value in ids):
            violations["event_missing_raw_bar"] += 1
            continue
        start, end, confirm = (int(id_to_pos[value]) for value in ids)
        if end - start + 1 != int(event.run_bars) or confirm != end + 1 or int(event.run_bars) < 4:
            violations["event_noncontiguous_run"] += 1
            continue
        run = bars.iloc[start : end + 1]
        confirm_row = bars.iloc[confirm]
        run_dir = int(event.run_direction)
        if not bool(valid.iloc[start : confirm + 1].all()) or not run["direction"].eq(run_dir).all() or int(confirm_row["direction"]) != -run_dir:
            violations["event_direction_sequence"] += 1
        if start > 0 and bool(valid.iloc[start - 1]) and int(bars.iloc[start - 1]["direction"]) == run_dir:
            violations["event_not_maximal"] += 1
        if pd.Timestamp(event.signal_time) != pd.Timestamp(confirm_row["end_ts"]):
            violations["event_signal_time"] += 1
        target = float(run.iloc[0]["open"])
        stop = float(min(run["low"].min(), confirm_row["low"])) if -run_dir > 0 else float(max(run["high"].max(), confirm_row["high"]))
        if not np.isclose(float(event.target_price), target, atol=1e-10, rtol=0):
            violations["event_target_formula"] += 1
        if not np.isclose(float(event.stop_price), stop, atol=1e-10, rtol=0):
            violations["event_stop_formula"] += 1
        span = float((pd.Timestamp(run.iloc[-1]["end_ts"]) - pd.Timestamp(run.iloc[0]["start_ts"])) / pd.Timedelta(seconds=1))
        touched = bool(confirm_row["high"] >= target) if -run_dir > 0 else bool(confirm_row["low"] <= target)
        expected_eligible = span > 0 and not touched
        if bool(event.signal_eligible) != expected_eligible:
            violations["event_eligibility_formula"] += 1
    return violations


def _path_checks(bars_1m: pd.DataFrame, trades: pd.DataFrame) -> dict[str, int]:
    bars = bars_1m.copy()
    bars.index = pd.to_datetime(bars.index)
    bars = bars.sort_index(kind="mergesort").loc[~bars.index.duplicated(keep="last")]
    times = bars.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(float)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(float)
    included = trades.loc[trades["path_status"].eq("included")].copy()
    violations = {
        "entry_timestamp": 0,
        "entry_open": 0,
        "first_passage_reason": 0,
        "first_passage_time": 0,
        "exit_fill": 0,
        "gross_return": 0,
        "cost_formula": 0,
        "position_overlap": 0,
        "split_boundary": 0,
        "holdout_absent": int(trades["research_split"].isin(["embargo", "holdout"]).sum()),
    }
    for trade in included.itertuples(index=False):
        signal_ns = int(pd.Timestamp(trade.signal_time).value)
        primary = int(np.searchsorted(times, signal_ns, side="right"))
        expected_floor = int((pd.Timestamp(times[primary]) + pd.Timedelta(minutes=int(trade.entry_delay_minutes))).value)
        entry_pos = int(np.searchsorted(times, expected_floor, side="left"))
        entry_time = pd.Timestamp(times[entry_pos])
        if entry_time != pd.Timestamp(trade.entry_time):
            violations["entry_timestamp"] += 1
        if not np.isclose(float(trade.entry_price), open_[entry_pos], atol=1e-10, rtol=0):
            violations["entry_open"] += 1
        direction, stop, target = int(trade.trade_direction), float(trade.stop_price), float(trade.target_price)
        expected = None
        for pos in range(entry_pos, len(times)):
            timestamp = pd.Timestamp(times[pos])
            if trade.research_split == "discovery" and timestamp >= pd.Timestamp("2025-01-01"):
                break
            if trade.research_split == "validation" and timestamp >= EMBARGO:
                break
            stop_hit = low[pos] <= stop if direction > 0 else high[pos] >= stop
            target_hit = high[pos] >= target if direction > 0 else low[pos] <= target
            if stop_hit:
                fill = min(open_[pos], stop) if direction > 0 else max(open_[pos], stop)
                expected = ("STOP", timestamp, float(fill))
                break
            if target_hit:
                expected = ("TARGET", timestamp, target)
                break
        if expected is None:
            violations["first_passage_reason"] += 1
            continue
        reason, exit_time, exit_price = expected
        if reason != str(trade.exit_reason): violations["first_passage_reason"] += 1
        if exit_time != pd.Timestamp(trade.exit_time): violations["first_passage_time"] += 1
        if not np.isclose(exit_price, float(trade.exit_price), atol=1e-10, rtol=0): violations["exit_fill"] += 1
        gross = direction * (exit_price / float(trade.entry_price) - 1.0)
        if not np.isclose(gross, float(trade.gross_return), atol=1e-12, rtol=0): violations["gross_return"] += 1
        for scale in (1, 2, 3):
            if not np.isclose(gross - scale * COST, float(getattr(trade, f"net_return_cost{scale}x")), atol=1e-12, rtol=0):
                violations["cost_formula"] += 1
    for _, part in included.groupby(["entry_delay_minutes", "research_split", "direction"], sort=False):
        ordered = part.sort_values("entry_time")
        previous_exit = pd.to_datetime(ordered["exit_time"]).shift(1) + pd.Timedelta(minutes=1)
        violations["position_overlap"] += int((pd.to_datetime(ordered["entry_time"]) < previous_exit).sum())
    violations["split_boundary"] = int(
        (included["research_split"].eq("discovery") & pd.to_datetime(included["exit_time"]).ge(pd.Timestamp("2025-01-01"))).sum()
        + (included["research_split"].eq("validation") & pd.to_datetime(included["exit_time"]).ge(EMBARGO)).sum()
    )
    return violations


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out_dir)
    events = pd.read_csv(out / "03_range_run_events.csv.gz", parse_dates=["run_start_ts", "run_end_ts", "confirmation_start_ts", "confirmation_end_ts", "signal_time"])
    trades = pd.read_csv(out / "05_trade_paths.csv.gz", parse_dates=["signal_time", "primary_entry_time", "entry_time", "exit_time"])
    loader = OKXRangeBarLoader(symbol=args.symbol, range_pct=0.0020, data_dir=args.data_dir, db_name=args.range_db_name, initialize_db=False)
    raw = loader.load_local_data("2022-01-01", "2025-06-30 23:59:59", columns=("bar_id", "start_ts", "end_ts", "duration_seconds", "open", "high", "low", "close", "direction"))
    bars_1m = OKXDataLoader(args.symbol, "1m").fetch_data_by_date_range("2022-01-01", "2025-06-30 23:59:59")
    results = {**_range_checks(raw, events), **_path_checks(bars_1m, trades)}
    audit = pd.DataFrame([{"check": key, "violations": value, "status": "PASS" if value == 0 else "FAIL"} for key, value in results.items()])
    audit.to_csv(out / "11_independent_replay.csv", index=False)
    print(audit.to_string(index=False), flush=True)
    if int(audit["violations"].sum()) != 0:
        raise AssertionError("R25 independent replay failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
