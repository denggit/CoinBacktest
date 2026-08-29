# SOXL ICT R20 Broad Position-Management Patch v2

## Fix
The original R20 incorrectly froze only the 1m close-confirmed MSS market subset (376 / 783 = 0.480/session) and aborted before management.

R20 v2 restores the broad causal entry universe:
- prominent 15m liquidity path;
- earliest first-visible structure break across 1m or 2m;
- visibility percentile >= 0.50 only defines a tradable swing;
- wick-only versus close-confirmed is retained as a feature, not a filter;
- entry is the next available 1m open after break_available_time;
- one entry per physical liquidity path;
- no probability/session/HTF/CVD/profitability hard filters;
- frequency failure is reported after diagnostics instead of aborting before the position-management backtest.

R18/R19 source diagnostics show 758 visible 2m MSS events across 783 valid sessions (~0.968/session), so the broad universe itself is not scarce. Final executable frequency is measured by R20 after causal entry materialization.

## Tests
- R20 unit/self-test: 7 passed.
- Full `tests/research/ict`: 84 passed.

## Run
`python research\ict\soxl_premarket_mss_fvg\20_broad_position_management_backtest.py --alpaca-symbol SOXL --alpaca-feed sip --alpaca-adjustment split --start-date 2023-07-01 --end-date 2026-08-14 --local-only --r16-cache-dir data\reports\research\ict\soxl\mss\r16_entry_archetype_survival_atlas_alpaca_2023_2026_08 --out-dir data\reports\research\ict\soxl\mss\r20_broad_position_management_backtest_alpaca_2023_2026_08`
