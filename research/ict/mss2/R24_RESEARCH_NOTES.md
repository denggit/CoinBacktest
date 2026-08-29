# R24 — Scheduled Funding-Window Unwind

Date: 2026-08-17

## Overall assessment: rejected

R24 tested whether unusually large completed 1h moves into the canonical OKX
00:00/08:00/16:00 funding clock reverse after scheduled settlement. It used no
funding-rate value or sign because pre-2026 local history is unavailable.

The signal reverses a pre-settlement move of at least 1.5 prior 720h sigmas at
the exact scheduled open. R1 primary and R2 sensitivity use a fixed 1.5×
completed hourly ATR stop, exact stop-first 1m passage, and an eight-hour
next-schedule timeout.

## Data and causality

- Bare ETH data contain all 1,838,880 requested 1m rows with no gap.
- 376 visible scheduled events produce 550 closed R1/R2 paths and no boundary
  censor. July and holdout rows are absent.
- Fourteen schedule, closed-hour, reversal-direction, threshold, barrier,
  split, and cost checks pass.
- Two focused regressions cover same-minute stop-first and exact next-schedule
  timeout execution.

## Result

| Primary R1 | Discovery trades | Discovery gross PF | Discovery 2× PF | Validation trades | Validation gross PF | Validation 2× PF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Long | 119 | 1.02 | 0.70 | 23 | 0.92 | 0.72 |
| Short | 102 | 0.98 | 0.70 | 31 | 0.49 | 0.37 |

The mechanism has no gross directional edge. Every year/direction primary cell
loses at 2× costs; PF ranges from 0.37 to 0.78. R2 diagnostics also fail in
every visible cell. Discovery top-ten removal is deeply negative.

R1 timeout rates are 28.6%/19.6% in Long/Short discovery and 30.4%/32.3% in
validation. The scheduled clock neither improves first-passage direction nor
resolves the path consistently before the next settlement.

## Frozen conclusions

1. Reject pre-settlement move reversal without actual funding-state data.
2. Do not search schedule subsets, z thresholds, directions, ATR multipliers,
   targets, or hold periods.
3. A known settlement clock alone is not sufficient positioning information.
4. Funding, mark, liquidation, books, and spot/perpetual basis remain unavailable
   for a valid pre-holdout discovery/validation study.
5. No sleeve is promoted and the holdout remains sealed.

