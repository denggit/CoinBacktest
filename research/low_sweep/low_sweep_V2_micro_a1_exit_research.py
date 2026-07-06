#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Low Sweep V2: micro confirmation + A1 rescue + exit-comfort research.

V1 locked the clean MF core:
``A0 + footprint abs-delta-high + single_swing + next_open + time48``.

V2 keeps that core as the anchor and asks three focused follow-up questions:
1. Can 5s/10s micro context improve A0 quality/MAE without destroying sample size?
2. Can the A1-only 0.8%-1.0% spike slice become a useful supplemental engine
   when filtered by footprint + micro exhaustion/turn signals?
3. Can exit-comfort rules reduce MAE/worst-trade/holding discomfort while
   preserving most of the V1 edge?  This includes both simple 1m swing trailing
   and a delayed structure-confirmed version that only arms the previous swing-low
   stop after a later swing high has been broken.

This is research-only.  It does not change the formal backtest or live strategy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.low_sweep_a_upgrade_research import main as upgrade_main  # noqa: E402

SCRIPT_NAME = "low_sweep_V2_micro_a1_exit_research"
DEFAULT_OUT_DIR = "data/reports/research/low_sweep/V2_micro_a1_exit_research"


def main(argv: Sequence[str] | None = None) -> int:
    # Keep the grid deliberately focused.  V2 is not a broad parameter sweep; it
    # is a sibling-engine/comfort research pass around the V1 A0+footprint core.
    defaults = [
        "--out-dir",
        DEFAULT_OUT_DIR,
        "--candidate-layers",
        ",".join(
            [
                # Anchor / controls
                "A0_fp_abs_delta_high",
                "A1_only_fp_abs_delta_high",
                # A0 micro upgrades: try to improve MAE/quality of the existing core.
                "A0_fp_abs_delta_high_micro5_buy_pressure",
                "A0_fp_abs_delta_high_micro10_buy_pressure",
                "A0_fp_abs_delta_high_micro5_sell_exhaustion",
                "A0_fp_abs_delta_high_micro10_sell_exhaustion",
                "A0_fp_abs_delta_high_micro5_no_new_low",
                "A0_fp_abs_delta_high_micro10_no_new_low",
                "A0_fp_abs_delta_high_micro5_combo",
                "A0_fp_abs_delta_high_micro10_combo",
                "A0_fp_abs_delta_high_micro5_large_buy",
                "A0_fp_abs_delta_high_micro10_large_buy",
                # A1-only rescue: add trades only if the 0.8%-1.0% slice survives micro checks.
                "A1_only_fp_abs_delta_high_micro5_buy_pressure",
                "A1_only_fp_abs_delta_high_micro10_buy_pressure",
                "A1_only_fp_abs_delta_high_micro5_sell_exhaustion",
                "A1_only_fp_abs_delta_high_micro10_sell_exhaustion",
                "A1_only_fp_abs_delta_high_micro5_no_new_low",
                "A1_only_fp_abs_delta_high_micro10_no_new_low",
                "A1_only_fp_abs_delta_high_micro5_combo",
                "A1_only_fp_abs_delta_high_micro10_combo",
            ]
        ),
        "--support-modes",
        "single_swing,equal2_020",
        "--entry-modes",
        "next_open",
        "--exit-modes",
        ",".join(
            [
                "time24",
                "time36",
                "time48",
                "target_signal_open_or_time48",
                "mfe_lock_10_03_time48",
                "mfe_lock_15_05_time48",
                "mfe_lock_20_10_time48",
                "partial_pct10_50_time48",
                "partial_pct15_50_time48",
                "fail_no_mfe6_002_time48",
                "fail_no_mfe12_003_time48",
                "swing_trail_after6_time72",
                "swing_trail_after12_time72",
                "swing_struct_trail_after6_time72",
                "swing_struct_trail_after12_time72",
            ]
        ),
        "--upgrade-stop-specs",
        "no_stop",
        "--context-sources",
        "trade_bar,range_bar,footprint",
        "--micro-timeframes",
        "5s,10s",
        # Auto remains monthly sliding-window.  It tries local cache first and
        # only falls back to monthly fetch when a month is empty, avoiding the
        # previous 2022-2026 full-table load.
        "--micro-load-mode",
        "auto",
        "--micro-last-seconds",
        "20",
        "--micro-buy-sell-ratio-min",
        "1.20",
        "--micro-sell-exhaustion-delta-min",
        "-0.15",
        "--micro-no-new-low-buffer-pct",
        "0.0000",
        "--save-trades",
        "0",
        "--save-events",
        "4000",
        "--min-trades-for-upgrade-edge",
        "50",
        "--min-pf-for-upgrade-edge",
        "2.0",
    ]
    print(f"[run] {SCRIPT_NAME}", flush=True)
    return upgrade_main(defaults + list(argv or []))


if __name__ == "__main__":
    raise SystemExit(main())
