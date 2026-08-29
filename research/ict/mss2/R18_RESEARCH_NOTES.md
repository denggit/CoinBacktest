# R18 — Independent Positioning-Unwind Path Atlas

Date: 2026-08-17

## Overall assessment: ready to share as a rejection

R18 tested a genuinely independent all-market mechanism rather than repairing a
completed-trend sweep or price-only continuation branch:

```text
completed 1h directional OKX price move + rising Binance base OI
→ first causal 5m Binance base-OI transition from nonnegative to negative
→ completed OKX 5m close reacquires the prior 5m extreme in reversal direction
→ first observable OKX 1m open
```

Binance USD-M `ETHUSDT` OI is explicitly a cross-exchange positioning proxy.
Execution and all outcome paths remain OKX `ETH-USDT-SWAP`. The exact event,
one-minute publication lag, gap rules, two-bar stabilization stop plus 0.25×
5m ATR, 1.50% maximum stop, 1h build-range structural target, 1R/2R/3R
diagnostics, 24-hour horizon, costs, and split boundary were frozen in
`R18_PRECOMMITMENT.md` before outcomes were calculated.

## Source and data quality

- Discovery: 2023-01-01 through 2024-12-31.
- Validation: 2025-01-01 through 2025-06-30.
- July 2025: embargoed.
- Holdout begins 2025-08-01 and remains sealed. There are 3,376 aggregate
  causal holdout candidates and zero holdout outcome rows.
- Pre-embargo Binance feature data contain 262,341 rows, no timestamp
  duplicates, and no base-OI nulls.
- There are 416 non-exact-five-minute intervals, including one 10.5-hour gap,
  and 188 partial archive days. R18 requires adjacent 4–6 minute observations
  and gap-safe 5m/1h baselines; it never interpolates.
- Eighty-one rows have nonpositive base OI and are excluded. Ninety-one rows
  have nonpositive OI USD; that field is recorded as bad source data but is not
  an admission gate because OI USD was frozen as unused.
- The OKX execution series through validation has 1,838,880 consecutive 1m
  rows with no gap, duplicate, null OHLCV, or OHLC consistency error.
- The local Binance cache ends at project time 2026-07-01 07:55, so it cannot
  support a claim of current live-data readiness through 2026-08-15.

## Sample and result

The visible mechanism produced 9,350 transitions and 8,604 executable setups.
Twelve late-June validation setups were correctly censored because a full
24-hour path would enter the July embargo, leaving 8,592 independently replayed
setups and 34,368 setup-target paths.

| Direction / target | Discovery 2× PF | Validation 2× PF | Discovery top-5 removed | Validation top-5 removed |
| --- | ---: | ---: | ---: | ---: |
| Long / 1h build range | 0.20 | 0.37 | 0.18 | 0.32 |
| Long / 1R | 0.19 | 0.38 | 0.18 | 0.35 |
| Long / 2R | 0.36 | 0.55 | 0.35 | 0.50 |
| Long / 3R | 0.43 | 0.56 | 0.42 | 0.51 |
| Short / 1h build range | 0.24 | 0.28 | 0.23 | 0.25 |
| Short / 1R | 0.20 | 0.29 | 0.20 | 0.28 |
| Short / 2R | 0.37 | 0.46 | 0.36 | 0.43 |
| Short / 3R | 0.45 | 0.53 | 0.44 | 0.50 |

Gross PF is only 0.96–1.07 across every direction, split, and target. There is
no meaningful raw directional edge for costs to preserve. Every 2×-cost monthly
sum is negative, giving a 0% positive-month rate in all primary cells. Every
2023, 2024, and 2025 direction/target cell is negative at 2× cost, with PF from
0.11 to 0.56.

The mechanism also fires far too often: roughly 116–171 executable paths per
month per direction. Median stop distance is only about 0.24% in discovery and
0.38% in validation, while the required 2× round trip is 0.22%. This does not
mean the stop should be widened or events filtered after the fact. It means a
near-random high-frequency sign transition has no economic room after costs.

## Validation and calculation spot-checks

- Twenty causal checks pass with zero violations, including exact publication
  lag, prior-build availability, next-eligible-minute entry, direction signs,
  stop ceiling, target availability, and sealed split boundaries.
- An independent raw-array loop replayed all 34,368 paths without using the
  production segment-tree ordering routine. It found zero outcome, exit-time,
  or exit-price differences.
- Gross-return reconciliation is within `5.1e-16`, 2×-cost arithmetic within
  `1.1e-16`, and grouped PF within `3.2e-15`; all counts match exactly.
- Every included setup has exactly four target paths and no duplicate
  setup-target row. No embargo or holdout path exists.
- Six focused R18 tests cover causal admission, future-price mutation,
  forbidden future fields, nonpositive/gap exclusion, stop-first ambiguity,
  and cost/audit formulas.

## Frozen conclusions

1. The exact 1h price/base-OI build → first 5m OI release → opposite 5m price
   reacquisition transition has no economic edge for Long or Short.
2. Do not tune OI magnitude, price magnitude, publication lag, ATR buffer, stop
   ceiling, target, or horizon to rescue R18.
3. Do not add taker, top-trader, global-account, funding, or regime filters to
   this failed reversal event.
4. OI USD and ratio fields remain excluded; future OI and oracle turning points
   remain physically outside causal admission.
5. No strategy is promoted and portfolio construction remains premature.
6. Holdout remains sealed, and eventual live approval still requires genuinely
   new forward data beyond history already inspected elsewhere in the repo.

## Next independent hypothesis

The repository has not tested the opposite positioning path already separated
in the R18 prebuild audit: directional price/base-OI build, a temporary OI
release that does **not** reverse price, and then causal OI rebuild plus a break
of the release range in the original direction. That is a continuation-resumption
mechanism, not a threshold filter on the failed R18 reversal event.

R19 may test that one sequence with a small state machine, a one-hour maximum
release window, Long/Short separation, the same gap and publication controls,
and predeclared volatility/structural barriers. It must stop immediately if
unfiltered discovery and validation do not both survive 2× costs. It may not use
ratio or magnitude filters as rescue.

## Primary evidence

- `data/reports/research/ict/mss2/r18_positioning_unwind_path_atlas/05_setup_funnel.csv`
- `08_direction_target_scorecard.csv`
- `09_direction_target_years.csv`
- `10_causal_audit.csv`
- `12_independent_reconciliation.csv`
- `manual_review/`
