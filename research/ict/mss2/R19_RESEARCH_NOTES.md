# R19 — Positioning Rebuild Continuation-Resumption Atlas

Date: 2026-08-17

## Overall assessment: ready to share as a rejection

R19 tested the economically opposite positioning path separated before R18:

```text
completed 1h directional OKX price move + rising Binance base OI
→ first causal 5m Binance base-OI release transition
→ uninterrupted negative-OI episode, maximum 60 minutes
→ first nonnegative base-OI rebuild observation
→ completed OKX 5m close breaks the frozen release bar in the original direction
→ first observable OKX 1m open
```

This is a continuation-resumption mechanism, not a filter on the rejected R18
reversal setup. Binance USD-M `ETHUSDT` OI remains a cross-exchange positioning
proxy; price, entry, stop, target, and outcome replay are all OKX
`ETH-USDT-SWAP`. The sequence, one-minute publication lag, gap handling,
60-minute window, episode-extreme stop plus 0.25× causal ATR, 1.50% stop ceiling,
1h volatility-range target, 1R/2R/3R diagnostics, 24-hour horizon, costs, and
splits were frozen in `R19_PRECOMMITMENT.md` before outcomes were inspected.

## Source and data quality

- Discovery: 2023-01-01 through 2024-12-31.
- Validation: 2025-01-01 through 2025-06-30.
- July 2025 is embargoed.
- Holdout begins 2025-08-01 and remains sealed. There are 2,661 aggregate
  causal holdout candidates and zero holdout outcome rows.
- Pre-embargo Binance data contain 262,341 rows, no duplicates, and no base-OI
  nulls. Eighty-one nonpositive base-OI rows are excluded.
- There are 416 irregular intervals, one 10.5-hour gap, and 188 partial days.
  No gap is interpolated or bridged.
- Ninety-one nonpositive OI-USD rows are recorded but irrelevant because OI USD
  is not an admission feature.
- OKX validation execution data contain 1,838,880 continuous valid 1m rows.
- The Binance cache ends at project time 2026-07-01 07:55, so neither R18 nor
  R19 establishes current live-data readiness through 2026-08-15.

## Funnel and result

The state machine finds 61,779 release episodes. Its terminal accounting is
mutually exclusive: 188 gap/invalid expiries, 83 genuine 60-minute expiries,
one right-edge censor, 46,561 first rebuilds without the required price break,
and 14,946 successful rebuild breaks. There are 8,799 visible candidates and
8,591 executable setups. Seven late-June paths are censored before the July
embargo, leaving 8,584 fully observed setups and 34,336 setup-target paths.

| Direction / target | Discovery 2× PF | Validation 2× PF | Discovery top-5 removed | Validation top-5 removed |
| --- | ---: | ---: | ---: | ---: |
| Long / 1h volatility range | 0.44 | 0.60 | 0.43 | 0.56 |
| Long / 1R | 0.26 | 0.43 | 0.26 | 0.41 |
| Long / 2R | 0.41 | 0.61 | 0.40 | 0.58 |
| Long / 3R | 0.46 | 0.64 | 0.45 | 0.60 |
| Short / 1h volatility range | 0.46 | 0.56 | 0.45 | 0.52 |
| Short / 1R | 0.28 | 0.41 | 0.28 | 0.39 |
| Short / 2R | 0.46 | 0.59 | 0.45 | 0.55 |
| Short / 3R | 0.50 | 0.71 | 0.48 | 0.65 |

Discovery gross PF ranges only 0.95–1.02. Validation gross PF is 0.97–1.15,
but the small apparent Short/3R gross surplus cannot pay even base costs and is
not present in discovery. Every primary 2× monthly sum is negative, so all
positive-month rates are zero. Every 2023, 2024, and 2025 direction/target cell
loses after 2× cost.

The mechanism is also far too frequent: roughly 114–155 executable setups per
month per direction. Median risk is about 0.28–0.30% in discovery and
0.41–0.46% in validation, leaving insufficient economic room for the required
0.22% 2× round trip.

## Validation and engineering corrections

- Nineteen causal checks pass with zero violations.
- An independent raw-array loop replays all 34,336 paths with zero outcome,
  exit-time, exit-price, return, cost, count, PF, duplicate, or split mismatch.
- Event and path floats are saved with `%.17g`, and reconciliation reads with
  pandas `float_precision="round_trip"`. This prevents decimal parser rounding
  from turning `1894.1100000000001` into an apparent replay discrepancy.
- The final engineering correction replaces an unsafe post-hoc decrement of
  `expired_after_60m` with an explicit `right_edge_censored` counter. A
  regression test proves a right-edge episode cannot erase a prior episode's
  genuine time expiry.
- Release admission now requires a finite, nonzero 1h price return; `NaN != 0`
  can no longer admit a warmup row.
- Thirteen combined R18/R19 focused tests pass.

Neither correction changes an admitted visible event, an outcome, or any
economic conclusion.

## Frozen conclusions

1. The exact price/OI build → OI release → first OI rebuild plus
   original-direction release-bar break has no cost-surviving edge for Long or
   Short.
2. R18 and R19 jointly close the simple Binance base-OI transition branch.
3. Do not tune release magnitude, rebuild magnitude, publication lag, window,
   ATR buffer, stop ceiling, target, horizon, funding, ratios, or volatility
   filters to rescue it.
4. No strategy is promoted, and the MSS2 holdout remains sealed.
5. Positioning-transition research should be archived unless a genuinely new
   causal data source or market mechanism is proposed rather than another
   threshold on the same sign sequence.

## Next decision boundary

R17–R19 are the three studies after `STRATEGIC_RESET_R16.md`. The mandatory
`STRATEGIC_RESET_R19.md` must precede R20. A repository candidate audit also
found the historical LF V10B composite, whose large headline return requires a
separate provenance, split, cost, and winner-dependence review before MSS2 may
reuse any of its mechanisms.

