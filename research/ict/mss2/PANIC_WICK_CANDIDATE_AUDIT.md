# Repository Candidate Audit — ETH Panic-Wick Structural Long

Date: 2026-08-17

## Saved claim

The historical note records roughly 332 trades, +54.65% return, PF 1.58,
-9.27% MDD, and four positive years for:

```text
priority_union
+ multi_sweep_higher_low_trail
+ entry_delay 2
```

## Selection provenance

The claim is not an untouched candidate:

- broad wick, range, volatility, trend, flow, prior-move, and session tables
  were inspected on 2023–2026;
- V1 backtested three entry policies × nine exits × three delays = 81 rows;
- V1.1 then compared seven exit upgrades across the same entry/delay family;
- the same full window supplied the reported annual stability.

No discovery/validation/holdout contract preceded that selection.

## R23 frozen replay

R23 copies only the frozen rule into reusable common code, corrects the sparse
trade-bar calendar without fabricating signals, resets discovery and validation,
and loads no July/holdout outcome.

- Discovery: 119 trades, 2× PF 1.67, +0.163% mean.
- Validation: 111 trades, 2× PF 0.96, -0.013% mean.
- Discovery top-ten removed: PF 0.84, -4.52% sum.
- Validation top-ten removed: PF 0.42, -18.61% sum.
- 2025H1: 2× sum -1.40%.
- Twelve causal checks and eight independent structural replay checks pass.

## Decision

The prior is rejected. It shows a gross/base-cost panic-rebound tendency, not a
stable 2×-cost sleeve. Its selection history, validation failure, and top-ten
dependence prohibit session/flow/wick/exit rescue. Capital allocation remains
zero and the MSS2 holdout remains sealed.

