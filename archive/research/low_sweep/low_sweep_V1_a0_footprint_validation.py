#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Low Sweep V1: A0 + footprint MF candidate validation.

Naming convention for the low-sweep research line:
- V1 is the first focused validation pass after the broad A-upgrade lab.
- It lives under ``research/low_sweep/`` so future V2/V3 scripts stay grouped.

Main questions answered by this focused run:
1. Is ``A0_fp_abs_delta_high + next_open + time48`` still the clean MF mainline?
2. Does the 0.8%-1.0% A1-only slice add useful incremental trades after A0?
3. Does delaying entry by 1-2 bars improve the special MAE/MFE timing pattern?
4. Does equal2 confirmation work better as a post-entry add-on than as an initial filter?
5. Did 5s/10s micro context actually attach, and how much candidate coverage does it have?

This script is a thin, named wrapper around the broad upgrade lab.  It does not
move anything into live trading; outputs remain research-only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.low_sweep_a_upgrade_research import main as upgrade_main  # noqa: E402

SCRIPT_NAME = "low_sweep_V1_a0_footprint_validation"
DEFAULT_OUT_DIR = "data/reports/research/low_sweep/V1_a0_footprint_validation"


def main(argv: Sequence[str] | None = None) -> int:
    defaults = [
        "--out-dir",
        DEFAULT_OUT_DIR,
        "--candidate-layers",
        ",".join(
            [
                "A0_fp_abs_delta_high",
                "A0_fp_low_delta_vneg",
                "A0_spike_ge_0100",
                "A1_fp_abs_delta_high",
                "A1_only_080_100",
                "A1_only_fp_abs_delta_high",
                "A1_only_fp_low_delta_vneg",
                "A0_micro5_last20_buy_pressure",
                "A0_micro10_last20_buy_pressure",
            ]
        ),
        "--support-modes",
        "single_swing,equal2_020",
        "--entry-modes",
        "next_open,next_open_delay1,next_open_delay2,next_open_add_equal2_050",
        "--exit-modes",
        "time24,time36,time48,target_signal_open_or_time48,swing_trail_after6_time72,swing_trail_after12_time72,swing_trail_after18_time96",
        "--upgrade-stop-specs",
        "no_stop,fixed_0250,atr_6x",
        "--context-sources",
        "trade_bar,range_bar,footprint",
        "--micro-timeframes",
        "5s,10s",
        "--micro-load-mode",
        "local",
        "--save-trades",
        "0",
        "--save-events",
        "3000",
    ]
    print(f"[run] {SCRIPT_NAME}", flush=True)
    return upgrade_main(defaults + list(argv or []))


if __name__ == "__main__":
    raise SystemExit(main())
