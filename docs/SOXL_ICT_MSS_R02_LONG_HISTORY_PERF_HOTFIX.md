# SOXL ICT MSS R02 long-history performance hotfix

## Problem
The long-history Alpaca run could appear stuck at `[research] build stages 2/5` because Stage 3 repeatedly rebuilt identical timeframe bars/pivots for every displacement threshold and repeatedly sliced the growing 1m sweep path for every candidate signal bar.

## Fix
- Aggregate each day/timeframe once and share it across all displacement thresholds.
- Advance the causal 1m sweep path monotonically and maintain cumulative terminal extreme / target-touch state instead of rebuilding DataFrame slices.
- Preserve first-occurrence terminal-extreme tie semantics used by pandas `idxmin` / `idxmax`.
- Preserve legacy attempt ordering and column layout.
- Add nested `[research-signals] causal sweep/MSS scan` progress output.

## Causality
No strategy rule, timing rule, MSS/FVG definition, TP/SL rule, or available-time rule was changed.

## Validation
- Synthetic old-vs-new output compared field-by-field: exact equivalent on the reference ICT fixture (6 attempts across 1.25/1.50/1.75 and 1m/2m paths).
- R02 self-test: PASS.
- Targeted pytest: 11 passed.
- 250 synthetic full trading days: Stage-3 signal build ~18.4 seconds in the validation container.
