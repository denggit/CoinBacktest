# Patch Manifest — ETH External Strategy Tournament V1

New reusable domain:
- `src/strategy_research/eth_tournament/`

New research entrypoint/docs:
- `research/eth_strategy_factory/`

New tests:
- `tests/strategy_research/eth_tournament/`

No changes are required to `src.data_feed`, legacy strategies, AetherEdge, or prior AI/RL research.

Engineering properties:
- public data loaders only
- 1m causal execution axis
- sparse NumPy replay for event strategies
- vectorized target-exposure replay for trend sleeves
- monthly footprint aggregation/release to prevent multi-year footprint materialization
- 5s quarter-hour data loaded month-by-month and immediately filtered to first 10 seconds
- hard 2026 seal
- stress battery and source metadata frozen before local ETH result review

Validation before delivery:
- `python -m pytest tests/strategy_research/eth_tournament tests/data_feed -q` -> 28 passed
- all new Python files -> `py_compile` passed
- tournament CLI `--help` -> passed
- full repository pytest remains blocked at collection by 5 pre-existing missing liquidity/panic research modules; no tournament module appears in those failures
- new tournament code has no `research -> research` / `backtest -> backtest` import and performs data access only through `src.data_feed`


## R02 continuous portfolio patch

Adds:
- `research/eth_strategy_factory/02_continuous_risk_managed_portfolio.py`
- `research/eth_strategy_factory/R02_CONTINUOUS_PORTFOLIO_SPEC.md`
- `src/strategy_research/eth_continuous_portfolio/`
- `tests/strategy_research/eth_continuous_portfolio/`

Updates cumulative V1/R02 research trail docs. Does not modify `src.data_feed`, legacy strategies, or AetherEdge.

## R03 source-locked replication patch

Adds:
- `research/eth_strategy_factory/03_source_locked_trend_replication.py`
- `research/eth_strategy_factory/R03_SOURCE_LOCKED_SPEC.md`
- `research/eth_strategy_factory/R03_WEB_SOURCE_NOTES.md`
- `src/strategy_research/eth_source_locked_portfolio/`
- `tests/strategy_research/eth_source_locked_portfolio/`

Updates cumulative V1/R02/R03 work-trail docs only. No changes to `src.data_feed`, AetherEdge, legacy strategies, or AI/RL research.

Validation:
- R03 dedicated: 12/12 passed
- V1/R02/data_feed regression: 39/39 passed
- all R03 Python files compile; CLI help passes
- R03 data access is only through `src.data_feed.OKXTradeBarLoader`; no direct DB/HTTP/exchange API and no research-script imports
- Full-repo pytest after R03 test-module name deconfliction returns the same **5 pre-existing liquidity/panic collection errors** as the prior baseline; R03 adds no collection failure.

## R04 Turtle path-atlas patch

Adds:
- `research/eth_strategy_factory/04_turtle_path_atlas.py`
- `research/eth_strategy_factory/R04_TURTLE_PATH_ATLAS_SPEC.md`
- `src/strategy_research/eth_turtle_path_atlas/`
- `tests/strategy_research/eth_turtle_path_atlas/`

R04 reuses the exact source-locked R03 Turtle engine and reconstructs completed Turtle episodes on the existing local 1m execution bars. It does not change Turtle entry/exit/pyramiding rules and does not touch `src.data_feed`.

Validation:
- R04 + R03 + R02 + V1 + data_feed regression: 57/57 passed
- all R04 Python files compile; CLI help passes
- 2026 hard seal test passes
- full-repo pytest remains blocked only by the same 5 pre-existing liquidity/panic collection errors; R04 adds no collection failure

## R04.1 hotfix — BacktestResult equity contract

Fixes a runner integration bug found by the user's first local R04 run: `BacktestResult` exposes the minute mark-to-market equity series as `minute_equity`, while the original R04 runner incorrectly referenced a non-existent `.equity` attribute.

Changes:
- `src/strategy_research/eth_turtle_path_atlas/runner.py`: pass `baseline.minute_equity` into the path atlas.
- `tests/strategy_research/eth_turtle_path_atlas/test_turtle_path_atlas.py`: add a runner-level regression test with a baseline object that deliberately has `minute_equity` and no `equity`, so this interface mismatch cannot silently recur.

No strategy, sizing, event, path, source-data, timing, or seal behavior changed.

Validation after hotfix:
- R04 dedicated: **7/7 passed**
- R03 + R02 + V1 + data_feed regression: **51/51 passed**
- combined targeted: **58/58 passed**
- `py_compile`: PASS
- full-repo pytest remains blocked only by the same 5 pre-existing liquidity/panic collection errors; R04.1 adds no failure.
