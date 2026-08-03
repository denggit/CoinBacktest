# R03.4.2.3 Patch Manifest

## Goal

Build a strictly causal multi-stage holding-decision study after the R03.4.2.2 sample-size failure.

## Key changes

- Generates holding-model training events from rolling OOF opening scores over the full pre-test history.
- Uses q50 only as the broad training pool; q70 and q90 remain separate OOS scopes.
- Extends each event path to 120 hours while keeping 2026 sealed.
- Models persistent failure, recoverable drawdown, post-6h continuation, and post-24h long-hold value.
- Early exit requires both high failure risk and low recovery probability.
- Tests full entry, half-size probe/add, and delayed T+180 confirmation.
- Enforces a single non-overlapping ETH long position.
- Keeps positive expectancy, 2x costs, PF, drawdown, quarter stability and Top-10 concentration as hard gates.
- Requires profit uplift in both OOS years before declaring either a multi-stage holding upgrade or a q70 opportunity expansion.
- Does not load the abandoned market-state model.
