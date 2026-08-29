# R16 — Acceptance Structural / Behavioral Stop Atlas

Date: 2026-08-16

## Question

R15 proved that the R14 region-edge stop was too tight relative to costs and entry-bar noise. R16 kept the frozen SSL root-close-outside short entry and deeper same-side completed-trend target, and changed only thesis invalidation:

1. `region_edge_touch`: R14 baseline.
2. `root_bar_extreme_touch`: root sweep-bar high plus 2bps hard stop.
3. `close_reclaim_plus_extreme`: first close above root region -> next-open exit, with the root-bar-extreme hard stop always active.

A target on the same OHLC bar as a touch stop or reclaim close is pessimistically a failure. No admission, target, cost, allocation or holdout rule changed.

## Result

| Stop | Discovery 2x PF | Validation 2x PF | Discovery 3x PF | Validation 3x PF | Discovery top-5-removed PF | Validation top-5-removed PF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| region edge | 1.12 | 0.57 | 0.93 | 0.46 | 0.12 | 0.04 |
| root bar extreme | 1.20 | 0.36 | 1.07 | 0.31 | 0.27 | 0.05 |
| close reclaim + extreme | 0.98 | 0.50 | 0.84 | 0.42 | 0.17 | 0.06 |

The root-bar extreme is the most defensible structural stop, but its result is regime-dependent: 2x PF is 0.57 in 2023, 1.56 in 2024, and 0.36 in 2025 validation. Positive-month rates are 41.7% discovery and 0% validation. Wider risk reduces the cost/risk mismatch but does not create stable expectancy.

Behavioral close reclaim is not an edge. It exits most positions within 2–3 minutes and is negative in 2023 and 2025. The strong 2024 subset cannot justify a rule.

The full report contains 142 frozen entries, 426 stop-model rows, six causal checks with zero violations, and six focused tests. The initial full run exposed and corrected a duplicate-column merge bug before any results were produced; a real-schema regression test now covers it.

## Frozen conclusions

1. Region-edge touch is too tight and cost dominated.
2. Root-bar extreme is structurally better but economically unstable.
3. Close-reclaim behavioral invalidation does not repair validation.
4. No stop model survives top-five removal.
5. The entire R14–R16 acceptance-continuation branch is archived.
6. No additional stop, target, FVG, order-flow or filter search is allowed on this branch.
7. Holdout remains sealed and no strategy is promoted.

## Next boundary

The repository’s separate directional-impulse continuation program has already run many rounds and found no cost-after continuation entry, so R17 must not repeat raw impulse/breakout chasing.

The next independent direction is a new long/short trend-pullback continuation branch:

```text
causal 1D/4H trend persistence and remaining runway
    -> orderly 1H/30m pullback/compression
    -> 15m/5m reclaim and re-acceleration
    -> next observable 1m execution
```

It may reuse public loaders, causal alignment and structure utilities, but must not inherit the archived q70 model, the R13 reversal filters, or R14 acceptance rules. Long and Short are evaluated separately. Stops anchor to local pullback structure plus a causal volatility buffer, with a predeclared maximum distance. Breakout is state evidence only, never the entry.

## Primary evidence

- `data/reports/research/ict/mss2/r16_acceptance_structural_stop_atlas/04_stop_model_outcomes.csv.gz`
- `05_stop_model_scorecard.csv`
- `06_stop_model_years.csv`
- `07_causal_audit.csv`
