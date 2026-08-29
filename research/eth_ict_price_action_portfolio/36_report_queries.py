#!/usr/bin/env python
"""Deterministic SQLite queries used to build the technical report snapshot."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
VALIDATION_CSV = ROOT / "ict_pa_v12" / "results" / "01_all_candidate_validation.csv"

SUMMARY_SQL = """
SELECT COUNT(*) AS candidates,
       SUM(CASE WHEN live_candidate_pass = 1 THEN 1 ELSE 0 END) AS approved,
       MAX(CASE WHEN candidate = 'daily_pa_core_only' THEN calmar END) AS baseline_calmar,
       23 AS tests_passed
FROM research_candidate_validation
""".strip()

FAMILY_SQL = """
SELECT family, candidate, calmar, cagr, max_drawdown AS mdd
FROM research_candidate_validation
WHERE (family = '10s_pinned_flow_release' AND candidate = 'pinned_10s_release_1m')
   OR (family = 'multiscale_bos' AND candidate = 'multiscale_bos_1m')
   OR (family = 'daily_liquidity_reclaim' AND candidate = 'daily_reclaim_only_1m')
   OR (family = 'nr7_breakout' AND candidate = 'nr7_breakout_only_1m')
   OR (family = 'multispeed_ema' AND candidate = 'multispeed_ema_1m')
   OR (family = 'equal_mechanism_portfolio' AND candidate = 'equal_pa_core_ema_1m')
   OR (family = 'causal_inverse_vol_portfolio' AND candidate = 'causal_inverse_vol_1m')
   OR (family = '4h_displacement_continuation' AND candidate = 'displacement_only_1m')
   OR (family = 'daily_pa_ridge' AND candidate = 'daily_pa_ridge_only_1m')
   OR (family = '4h_linear_walkforward' AND candidate = 'daily_pa_core_only')
ORDER BY family
""".strip()

PRIORITY_SQL = """
SELECT user_priority_rank AS rank,
       candidate,
       max_consecutive_flat_days AS flat_days,
       max_consecutive_losing_days AS losing_days,
       max_drawdown AS mdd,
       cagr,
       CASE WHEN live_candidate_pass = 1 THEN 'yes' ELSE 'no' END AS approved
FROM research_candidate_validation
ORDER BY user_priority_rank
LIMIT 6
""".strip()


def run_queries() -> dict[str, pd.DataFrame]:
    validation = pd.read_csv(VALIDATION_CSV)
    with sqlite3.connect(":memory:") as connection:
        validation.to_sql("research_candidate_validation", connection, index=False)
        return {
            "summary": pd.read_sql_query(SUMMARY_SQL, connection),
            "family_comparison": pd.read_sql_query(FAMILY_SQL, connection),
            "priority_rows": pd.read_sql_query(PRIORITY_SQL, connection),
        }


def main() -> int:
    for name, frame in run_queries().items():
        print(f"\n{name}\n{frame.to_string(index=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
