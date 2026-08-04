# R03.4.2.17 Runtime Hotfix

## Trigger

The first full local run stopped with:

`MergeError: incompatible merge keys dtype('<M8[us]') and dtype('<M8[ns]')`

Preflight data coverage and all source/seal audits passed. No attribution output was produced, so the failed report must not be interpreted as a research conclusion.

## Fix

- Normalize every `merge_asof` time key to timezone-naive `datetime64[ns]`.
- Normalize the minute-bar index before 4H/1D causal resampling.
- Preserve completed-bar availability and backward-only causal alignment.
- Add regression tests for mixed microsecond/nanosecond inputs.

## Validation

`python -m pytest -q tests/ai_research/test_long_tail_state_gate_diagnostic.py`

Result: `11 passed`.

## Rerun

`python research\eth_ai_trading\03_4_2_17_state_gate_diagnostic.py`

The previous `FAIL_RUNTIME` report contains no valid state-attribution findings and should be replaced by the rerun output.


## Hotfix 2: named DatetimeIndex month summary

The second local rerun passed source/seal/data checks but stopped with `KeyError: month_end`.
The monthly attribution helper incorrectly assumed that `reset_index()` always creates a column named `index`.
OKX Trade Bar frames preserve a named time index, so the first reset column may instead be `timestamp` or another loader-specific name.
The implementation now deterministically renames the first reset column to `month_end` and includes a named-index regression test.
This is a reporting/runtime fix only; it does not alter the frozen model, q70 threshold, C2 trades, costs, stops, exits, or any 2026 result.
