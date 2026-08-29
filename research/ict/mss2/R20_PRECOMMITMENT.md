# R20 Precommitment — Frozen LF V10B Component Falsification

Date frozen: 2026-08-17, before the R20 visible-window component scorecard is generated.

## Status and limitation

R20 is not an independent discovery study. The repository's V10B rules were
historically developed on overlapping 2023–2026 data, and headline component
results have already been inspected. R20 is a bounded falsification: it asks
whether the current frozen components retain simple unlevered economics under
the MSS2 split and cost conventions. A survival can justify only genuinely new
forward incubation; it cannot retroactively create an untouched validation.

## Frozen source path

- Market: OKX `ETH-USDT-SWAP`.
- Raw market access: existing loaders behind `src.data_feed`.
- Frozen feature/selector path: `src/edge_lib/lf_*` plus
  `src/sleeve_lib/lf_v10b`.
- Base timeframe: 4H, warmup from 2022-01-01.
- Component cells are fixed as:
  - `BULL_RECLAIM_V2 / Long`;
  - `BEAR_V3_ONLY / Short`;
  - `MOMENTUM_V3 / Long`;
  - `MOMENTUM_V3 / Short`.
- Priority, micro filters, 21-bar structural stop, protected/trailing exits,
  add-on rules, and next-4H-open execution remain exactly as current source.
- No source threshold or rule may change after the R20 scorecard.

## Outcome-free boundary

- Discovery entries: 2023-01-01 through 2024-12-31, with exits before
  2025-01-01.
- Validation entries: 2025-01-01 through 2025-06-30, with exits before
  2025-07-01.
- The loader stops at 2025-06-30 19:59:59. The 20:00-labelled 4H bar closes at
  the July boundary and is therefore not admitted.
- Cross-boundary and forced-end trades are censored from economics.
- July 2025 is embargoed.
- Holdout begins 2025-08-01; no holdout market data, candidate, trade, return,
  or path outcome is loaded or saved by R20.

## Frozen economic unit

R20 strips leverage, capital compounding, and dynamic quantity from the primary
score. For each completed trade it uses the signed percentage move from the
zero-cost average entry to zero-cost exit:

```text
gross_return = direction × (exit / average_entry - 1)
```

The frozen conservative round-trip deductions are:

- 1×: 0.15% = two sides × (0.055% fee + 0.020% slippage);
- 2×: 0.30%;
- 3×: 0.45%.

This is deliberately simpler than the historical leveraged account curve.

## Required reporting

For every split/component/direction cell:

- trades and trades/month;
- gross and 1×/2×/3× PF and mean return;
- win rate;
- positive-month rate including zero-trade months;
- longest entry gap;
- top-five and top-ten removed 2× PF and return sum;
- yearly stability;
- causal/path checks and exact cost arithmetic;
- manual recent, best, and worst examples.

## Decision gate

A component is merely **forward-incubation eligible** only if:

- discovery and visible validation both have at least 12 trades;
- 2× PF is at least 1.4 in both splits;
- mean 2× return is positive in both splits;
- top-five-removed 2× PF remains above one in both splits;
- direction/component identity is fixed rather than selected from the best cell;
- all causal, boundary, uniqueness, and cost checks pass.

Top-ten removal, positive months, frequency, and gaps remain mandatory
diagnostics. No R20 result promotes capital, constructs a portfolio, changes
V10B, or opens the holdout.

## Explicit prohibitions

- no parameter grid or neighboring thresholds;
- no engine deletion based on R20 outcome to create a composite;
- no structural-stop, add-on, range-filter, or risk-multiplier rescue;
- no leverage or compounding used to make PF look stronger;
- no ML, ranking, date filter, regime filter, or top-winner-dependent rule;
- no July or holdout outcome access.
