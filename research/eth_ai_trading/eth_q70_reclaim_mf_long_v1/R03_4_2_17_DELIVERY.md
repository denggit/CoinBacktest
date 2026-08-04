# R03.4.2.17 Delivery

Status: `CODE_COMPLETE_PENDING_LOCAL_RUN`

Delivered:

- sealed source integrity validation;
- causal completed-bar 1D/4H market-state timeline;
- frozen C2 state attribution across 2024, 2025, 2026 H1 and July;
- fixed-6h entry-Edge versus C2 exit-overlay separation;
- exact frozen model, q70 threshold and feature-schema audit;
- state-conditional score drift;
- monthly ETH/C2/state comparison;
- predeclared counterfactual gate diagnostics with explicit future-validation requirement;
- cumulative handoff, decision and roadmap updates;
- dedicated tests and runbook.

Run:

```bat
python research\eth_ai_trading\03_4_2_17_state_gate_diagnostic.py
```

Validation:

- stage-specific: 9 passed;
- all `test_long_tail_*.py`: 151 passed;
- `tests/ai_research` + `tests/data_feed`: 240 passed;
- full repository collection: 558 tests discovered, blocked by 5 pre-existing missing liquidity/Analyze Tool modules;
- import-boundary audit: 155 pre-existing violations, R03.4.2.17 adds 0;
- current container execution: `BLOCKED_DATA` because local Trade Bar data is unavailable; no empirical state conclusion is claimed.


## Runtime hotfix

The first full local run exposed a Pandas `datetime64[us]` versus `datetime64[ns]` `merge_asof` incompatibility. The patch now normalizes all causal merge keys to `datetime64[ns]` and includes dedicated regression tests. See `research/eth_ai_trading/R03_4_2_17_RUNTIME_HOTFIX.md`.


### Second runtime correction

- Fixed `KeyError: month_end` for loader-named DatetimeIndex values in monthly attribution.
- Added a regression test covering a `timestamp`-named index.
- No strategy or model contract changed; rerun the same R03.4.2.17 command.
