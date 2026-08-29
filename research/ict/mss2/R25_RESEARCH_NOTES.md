# R25 — r0020 Directional-Run Exhaustion Reversal

Date: 2026-08-17

## Overall assessment: rejected

R25 tested one fixed event that was distinct from the previously rejected
Range-Bar activity-continuation rule. A maximal run of at least four completed
r0020 bars in one direction had to end with the first opposite completed bar.
Entry was the first strictly later source-observed 1-minute open, the run origin
was the target, and the run-plus-confirmation extreme was the stop.

The event is frequent but has no gross directional edge. Costs turn a nearly
fair first-passage process into a large, stable loss in every split, direction,
year, quarter, and delay cell. No sleeve is promoted.

## Prior-work and precommitment

The repository audit found that earlier work already used Range-Bar activity,
single-bar direction, rolling direction balance, speed, duration, duration
change, flow, footprint, and nonlinear models. R25 therefore did not test those
features again. It froze only the previously untested event sequence:

```text
four-plus same-direction r0020 bars
    -> first opposite completed r0020 bar
    -> next-observed 1m entry
    -> run origin before sequence extreme
```

There is no scale family, run-length sensitivity, activity bucket, flow or
duration filter, time exit, trailing stop, target ladder, or holdout selection.
One additional minute is execution stress only.

## Data quality and causality

- The local visible r0020 read contains 474,704 rows through the loader's
  overlap query. One row ends after the visible cutoff and is explicitly
  removed because R25 requires `end_ts < 2025-07-01`.
- Bar IDs are unique; required columns have no nulls. Two rows have
  `start_ts > end_ts`; they are source-invalid and reset the sequence.
- 37,114 rows have zero duration and 47,691 rows share an `end_ts` with another
  row. These are retained under the frozen rule because ordered raw trades can
  share a millisecond. Their `(end_ts, bar_id)` order is deterministic and all
  constituent bars are known before the strictly later 1-minute entry.
- Zero-duration share is temporally heterogeneous (roughly 10% median by month
  in 2023, 5% in 2024, and 1% in 2025H1). This is a medium-confidence source-
  shape caveat for duration-based research. Duration is diagnostic only in R25,
  and the economic rejection is present in every visible year and quarter.
- Bare OKX execution data contain all 1,838,880 requested 1-minute rows with no
  internal gap through 2025H1.
- Six focused regressions pass: deterministic equal-time ordering, invalid-row
  reset, strict post-signal entry, next-observed entry across a gap, stop-first
  same-minute ambiguity, and split-boundary censoring.
- Sixteen internal causal/cost checks pass. A separate validator that does not
  call the R25 event builder or simulator passes 18 raw-source checks across all
  events and paths.
- July and holdout rows are absent.

## Event and path funnel

The primary zero-delay visible simulation sees 15,705 qualifying run-end
events. After pre-entry staleness, invalid entry geometry, and open-position
overlap, 13,684 paths close; one validation Short path is boundary-censored.
The one-minute-delay stress closes 12,756 paths.

The event is far too common for the portfolio target—roughly 190–196 closed
trades/month by direction in discovery and 364–371/month in validation. This is
not useful coverage because the marginal trade is negative after costs.

## Primary result

| Direction | Discovery trades | Gross PF | 2x PF | Validation trades | Gross PF | 2x PF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Long | 4,567 | 1.05 | 0.41 | 2,228 | 1.08 | 0.42 |
| Short | 4,707 | 1.01 | 0.39 | 2,182 | 0.98 | 0.38 |

The median 2x net loss per trade is not a right-tail accident. Mean 2x return is
about -0.205% to -0.224% in the four primary cells, positive-month rate is 0%,
and every quarter loses. Removing the top five or top ten winners makes the
already negative sums worse.

Median risk distance is about 0.28–0.30%, while the 2x round trip is 0.22%.
Gross win rate is only 28–31%. Although the median initial target/stop ratio is
roughly 2.46–2.64, median realized R is -1: the first opposite r0020 bar does
not mark a sufficiently reliable terminal exhaustion point.

The additional one-minute delay remains negative in all cells: 2x PF is
0.38–0.40 and mean 2x return is about -0.208% to -0.222%. Execution delay
therefore cannot rescue the mechanism.

## Why it failed

1. A four-bar directional run is not terminal-state information. It describes
   displacement that can keep extending or alternate noisily.
2. One opposite r0020 bar is too weak as reversal confirmation. Gross PF near
   one shows almost fair path ordering before costs, not exploitable asymmetry.
3. The structural target is attractive only conditionally. The low target-first
   rate overwhelms the favorable nominal reward/risk.
4. The signal frequency is evidence of generic two-sided path churn, consistent
   with Momentum R11's volatility-expansion conclusion.
5. With median risk near the stressed round-trip cost, a nearly fair event
   cannot survive executable fees and slippage.

## Frozen conclusions

1. Reject four-plus r0020 run followed by first opposite-bar reversal.
2. Do not search r0015/r0025, run lengths, duration acceleration, flow, delta,
   footprint, session, volatility, target, stop, or confirmation variants.
3. Do not combine R25 with the rejected Range-Bar activity result or use an ML
   filter to rescue a gross PF near one.
4. Range-Bar duration features retain a temporal source-shape caveat and should
   not become a new standalone branch without a separate provenance audit.
5. No sleeve is promoted; July and the holdout remain sealed.

