#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent raw-1m outcome replay for R22."""
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

DEFAULT_REPORT_DIR = Path("data/reports/research/ict/mss2/r22_btc_led_eth_catchup")
EXPERIMENT_ID = "ETH_ICT_MSS2_BTC_LED_ETH_CATCHUP_R22"
EDGE_ID = "RESEARCH_ONLY_BTC_LED_ETH_CATCHUP"
TITLE = "ETH ICT MSS2 R22 BTC-Led ETH Catch-Up First Passage"
SPLIT_END = {"discovery": pd.Timestamp("2025-01-01"), "validation": pd.Timestamp("2025-07-01")}
TOL = 1e-10


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--end-date", default="2025-06-30 23:59:59")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--skip-review-pack", action="store_true")
    return parser.parse_args(argv)


def replay(trades: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    market = bars.sort_index()
    times = market.index.to_numpy(dtype="datetime64[ns]")
    open_ = market["open"].to_numpy(dtype=float)
    high = market["high"].to_numpy(dtype=float)
    low = market["low"].to_numpy(dtype=float)
    rows = []
    for trade in trades.itertuples(index=False):
        entry_time = pd.Timestamp(trade.entry_time)
        direction = int(trade.trade_direction)
        stop = float(trade.stop_price)
        target = float(trade.target_price)
        boundary = SPLIT_END[str(trade.research_split)]
        deadline = entry_time + pd.Timedelta(hours=24)
        path_end = min(deadline, boundary)
        start_pos = int(np.searchsorted(times, np.datetime64(entry_time), side="left"))
        end_pos = int(np.searchsorted(times, np.datetime64(path_end), side="left"))
        entry_open = float(open_[start_pos])
        stop_hit = low[start_pos:end_pos] <= stop if direction > 0 else high[start_pos:end_pos] >= stop
        target_hit = high[start_pos:end_pos] >= target if direction > 0 else low[start_pos:end_pos] <= target
        stop_positions = np.flatnonzero(stop_hit)
        target_positions = np.flatnonzero(target_hit)
        first_stop = int(stop_positions[0]) if len(stop_positions) else None
        first_target = int(target_positions[0]) if len(target_positions) else None
        if first_stop is not None and (first_target is None or first_stop <= first_target):
            pos = start_pos + first_stop
            expected_reason = "STOP"
            expected_time = pd.Timestamp(times[pos])
            expected_price = min(float(open_[pos]), stop) if direction > 0 else max(float(open_[pos]), stop)
        elif first_target is not None:
            pos = start_pos + first_target
            expected_reason = "TARGET"
            expected_time = pd.Timestamp(times[pos])
            expected_price = target
        elif deadline < boundary:
            pos = int(np.searchsorted(times, np.datetime64(deadline), side="left"))
            expected_reason = "TIME_EXIT"
            expected_time = deadline
            expected_price = float(open_[pos])
        else:
            expected_reason = "SPLIT_BOUNDARY_CENSORED"
            expected_time = pd.NaT
            expected_price = np.nan
        expected_gross = direction * (expected_price / entry_open - 1.0) if np.isfinite(expected_price) else np.nan
        time_equal = (pd.isna(trade.exit_time) and pd.isna(expected_time)) or pd.Timestamp(trade.exit_time) == expected_time
        price_equal = (pd.isna(trade.exit_price) and not np.isfinite(expected_price)) or abs(float(trade.exit_price) - expected_price) <= TOL
        gross_equal = (pd.isna(trade.gross_return) and not np.isfinite(expected_gross)) or abs(float(trade.gross_return) - expected_gross) <= TOL
        checks = {
            "entry_raw_open": abs(float(trade.entry_price) - entry_open) <= TOL,
            "reason_replayed": str(trade.exit_reason) == expected_reason,
            "time_replayed": bool(time_equal),
            "price_replayed": bool(price_equal),
            "gross_replayed": bool(gross_equal),
        }
        failed = [name for name, passed in checks.items() if not passed]
        rows.append({
            "trade_id": trade.trade_id,
            **{name: int(value) for name, value in checks.items()},
            "expected_reason": expected_reason,
            "expected_time": expected_time,
            "expected_price": expected_price,
            "violations": len(failed),
            "failed_checks": "PASS" if not failed else ",".join(failed),
        })
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report_dir = Path(args.report_dir)
    trades = pd.read_csv(report_dir / "04_trade_paths.csv.gz", parse_dates=["entry_time", "exit_time"])
    bars = OKXDataLoader(args.symbol, "1m").fetch_data_by_date_range(args.start_date, args.end_date)
    detail = replay(trades, bars)
    checks = ["entry_raw_open", "reason_replayed", "time_replayed", "price_replayed", "gross_replayed"]
    audit = pd.DataFrame([
        {"check": check, "rows_checked": len(detail), "violations": int(detail[check].eq(0).sum()), "status": "PASS" if bool(detail[check].eq(1).all()) else "FAIL"}
        for check in checks
    ])
    detail.to_csv(report_dir / "09_independent_trade_replay.csv", index=False, float_format="%.17g")
    audit.to_csv(report_dir / "10_independent_replay_audit.csv", index=False)
    if not args.skip_review_pack:
        finalize_research_report(report_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)
    print(audit.to_string(index=False))
    return 0 if int(audit["violations"].sum()) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

