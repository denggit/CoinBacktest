# R22 — BTC-Led ETH Catch-Up First Passage

Date: 2026-08-17

## Overall assessment: rejected; cross-market lag does not clear costs

R22 tested a new cross-market mechanism after funding/basis was rejected for
insufficient local history. A completed 1h BTC impulse identifies direction;
ETH must already move the same way but lag its prior-only 720h beta expectation
by at least 0.75 prior residual sigma. The BTC impulse threshold is two prior
168h sigmas. Entry is the next-hour ETH 1m open.

The primary R1 and diagnostic R2 paths use a fixed 1.5× completed ETH hourly
ATR(20) stop, exact 1m stop-first passage, and a 24h boundary-open safety exit.
Long/Short and discovery/validation are independent one-position sleeves.

## Data and validation

- ETH and BTC each contain exactly 1,838,880 requested 1m rows from 2022-01-01
  through 2025-06-30 23:59, with 100% coverage and no internal gap.
- All 1,838,880 timestamps match; there are no ETH-only or BTC-only minutes.
- 30,648 complete aligned 1h bars produce 576 pre-embargo signal events.
- The simulator emits 780 closed target paths and one boundary censor.
- Fifteen prior-feature, signal, execution, split, stop/target, and cost checks
  pass with zero violations.
- A separate raw-array replay checks all 781 paths without the R22 simulator.
  Entry open, exit reason, exact time, exact price, and gross return all match.

## Result

| Target / direction | Discovery trades | Discovery 2× PF | Validation trades | Validation 2× PF | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| R1 Long | 207 | 0.85 | 36 | 1.13 | fail discovery and validation PF |
| R1 Short | 138 | 0.72 | 25 | 0.92 | fail both splits |
| R2 Long | 186 | 0.92 | 34 | 1.14 | diagnostic also fails |
| R2 Short | 132 | 0.66 | 22 | 1.08 | diagnostic also fails |

R1 Long is the strongest visible path, but its discovery gross PF is only 1.19
and its mean gross return is 0.115%; the 0.22% 2× round trip changes expectancy
to -0.105%. R1 Short is approximately flat before costs. Removing the top ten
discovery winners leaves R1 Long/Short 2× sums of -46.96%/-53.31%.

The apparent 2025H1 Long improvement is not a valid rescue. R1 Long 2× PF is
0.75 in 2023, 0.93 in 2024, and 1.13 in 2025H1. R2 Long follows the same pattern
at 0.85/0.97/1.14. No visible year prior to validation has positive net
economics.

Frequency is useful diagnostically: R1 produces 8.63 Long and 5.75 Short trades
per discovery month, with only 2.4%/5.1% timeouts. This shows that BTC impulses
often create prompt ETH paths, but the direction is not sufficiently asymmetric
after costs. The failure is edge strength, not event scarcity or path timeout.

## Frozen conclusions

1. Reject the exact BTC impulse → same-direction ETH catch-up rule.
2. Do not rescue it with beta/sigma windows, impulse/lag thresholds, sessions,
   target/stop variants, trend filters, or 2025 selection.
3. BTC lead/lag contains a small gross tendency for Long catch-up but no stable
   stressed-cost edge in 2023–2024.
4. R22 contributes no sleeve to portfolio construction; July and holdout remain
   untouched.

## Next boundary

The mandatory R20–R22 strategic reset follows. An older ETH panic-wick branch
has a reported PF near 1.9 and 2025 PF near 1.47, but it was developed on a full
2023–2026 window using many feature exclusions, entry policies, exit modes, and
delays. It may be audited as a contaminated prior, not accepted as evidence.

