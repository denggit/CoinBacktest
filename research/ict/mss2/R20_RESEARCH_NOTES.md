# R20 — Frozen LF V10B Component Falsification

Date: 2026-08-17

## Overall assessment: rejected; archive historical V10B reuse

R20 did not search for a new rule. It froze the current repository LF V10B
feature, selector, micro-filter, add-on, and exit path, ran it at zero execution
cost only through visible 2025H1, and converted every completed trade into a
simple unlevered signed move from average entry to exit. It then deducted the
precommitted 0.15%/0.30%/0.45% round trip for 1×/2×/3× cost.

This was deliberately a falsification of a contaminated historical prior. The
V10B rules were developed on overlapping 2023–2026 data, so the R20 validation
cell is visible and cannot be called independent. July and holdout market data
are absent from the R20 feature/trade/economic artifacts.

## Causal and boundary design

- Existing 4H signals execute at the next 4H open.
- Higher-timeframe regimes and Donchian channels retain their existing shifted
  completed-bar construction.
- The 21-bar structural stop update remains future-effective only; stop touch is
  evaluated first on the active stop.
- The loader ends at 2025-06-30 19:59:59. The 20:00-labelled 4H bar would close
  at the July boundary and is excluded.
- Discovery requires both entry and exit before 2025-01-01.
- Validation requires both entry and exit before 2025-07-01.
- Boundary and forced-end paths would be censored, although the final run has no
  such trade among its 82 rows.
- Quantity, leverage, dynamic account risk, and compounding are outside the
  primary return unit. Average entry still reflects the frozen add-on path.

Fourteen causal/arithmetic checks pass: unique IDs, direction, entry/exit
ordering, exact next-4H entry, signal and selected-engine parity, finite positive
prices, split boundaries, holdout absence, signed return, and all three cost
formulas.

## Result

| Component | Discovery trades | Discovery 2× PF | Validation trades | Validation 2× PF | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Bear V3 Short | 15 | 0.72 | 4 | 0.72 | fail both economics and sample |
| Bull Reclaim V2 Long | 44 | 0.65 | 3 | 0.22 | fail both economics and sample |
| Momentum V3 Long | 5 | 0.00 | 3 | 2.69 | sparse, no discovery edge |
| Momentum V3 Short | 6 | 1.37 | 2 | 16.67 | sparse, discovery below gate |

The two engines responsible for most historical V10B trades have negative
gross expectancy even before costs in both the aggregate discovery and visible
validation cells. Bear Short discovery gross PF is 0.93 and Bull Reclaim Long
is 0.78. Their 2× positive-month rates are only 4.2%/12.5% in discovery and
16.7%/0% in validation, including zero-trade months.

Momentum Short has positive discovery mean return, but only six discovery and
two validation trades. Removing the top five discovery trades leaves only a
loss, and the large validation PF comes from two trades. Momentum Long has five
discovery trades with no winner. No component passes the frozen forward-
incubation gate.

Year decomposition confirms instability. Bear Short and Bull Reclaim lose
heavily in 2023, show a small positive 2024 cell, then lose in 2025H1. The
historical V10B account curve therefore depends on dynamic sizing, add-ons,
compounding, and a small number of convex winners rather than a stable
unlevered component-level trade edge.

## Frozen conclusions

1. Do not transplant any V10B component into MSS2 as a proved sleeve.
2. Do not rescue the raw component result with quality/risk multipliers,
   component deletion, structural-stop variants, or another micro filter.
3. The full-window V10B headline remains an optimization-contaminated sizing
   and winner-concentration result, not evidence that its typical setup has
   positive expectancy.
4. No component is eligible even for forward incubation under the precommitted
   R20 gate.
5. July 2025 and the MSS2 holdout remain outcome-free in R20.
6. Portfolio construction remains premature; R20 supplies no independent edge.

## Next boundary

R20 closes the V10B reuse shortcut. The next research must be a genuinely new
mechanism with simple raw edge before sizing—for example a precommitted
volatility-expansion or longer-horizon trend state that is not the existing
V10B signal stack. It must not inherit V10B quality multipliers, add-ons, or the
optimized 21-bar stop.

