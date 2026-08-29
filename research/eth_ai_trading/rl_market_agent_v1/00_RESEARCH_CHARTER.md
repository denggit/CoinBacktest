# Research Charter — RL Market Agent V1

## Objective

Find an ETH-USDT-SWAP model that remains economically profitable after realistic costs while producing a tradeable, low-drawdown and low-idle-time equity curve. The programme is allowed to fail; it is not allowed to hide leakage or overfit.

## Frozen data/timing contract

- Warmup: 2022-01-01 onward.
- Research dataset: 2023-01-01 through 2026-08-15.
- 2026-01-01 onward: sealed final holdout for later model/policy research.
- Decision cadence R00: 5 minutes.
- State may use only observations whose real available time is <= decision time.
- Left-labeled fixed bars become observable only at bar start + full timeframe.
- The execution/reference price for outcome labels is the 1m trade-bar open at decision time.
- Future data is permitted only inside labels/outcomes, never state features.

## Data sources

All reads go through `src.data_feed`. The research layer must not write new exchange clients, query market SQLite tables directly, or silently rebuild missing range/footprint history.

Core: tick-derived 1m trade bar + 5m/15m/1H/4H/1D closed K-line context.
Optional enrichments: 5s trade bars; r0.15/r0.20/r0.25 range bars; r0.20 step-1 footprint.

## Economic contract for later trading stages

- Base full round-trip fee: 0.11%.
- Mandatory 2x and 3x cost stress.
- No same-bar optimistic execution assumptions.
- No tuning against the sealed holdout.

## Strategy-first rule

Research is not complete when an abstract edge, IC, AUC, feature importance or model score looks good. Every stage must move toward an executable ETH strategy with explicit entry, exit, sizing and risk logic, then survive a causal backtest with realistic costs. If a modelling direction cannot materially improve a tradable strategy, stop or redirect it rather than extending it for academic completeness.

Final strategy comparison uses profitability/risk feasibility gates first, then the frozen lexicographic priority: max flat days, max consecutive losing days, MDD, CAGR, total return.
