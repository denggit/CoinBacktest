# R15 — Acceptance Fixed-R First Passage

Date: 2026-08-16

## Question

R14 SSL root-close acceptance preserved 4.29 discovery and 6.50 validation entries/month but rarely reached distant deeper completed-trend liquidity. R15 tested whether the same frozen entry and stop instead had a robust, exact 0.5R/1R/2R/3R first-passage edge.

R15 changed no admission rule. It retained the R14 next-open short entry and full-region-reclaim-plus-2bps stop. It used no time exit, runner, partial allocation, risk schedule or holdout data. Same-bar target/stop was stop-first.

## Result

The fixed-R hypothesis fails every economic gate.

| Target | Discovery gross PF | Discovery 2x PF | Validation gross PF | Validation 2x PF |
| --- | ---: | ---: | ---: | ---: |
| 0.5R | 0.37 | 0.10 | 0.40 | 0.12 |
| 1R | 0.45 | 0.16 | 0.79 | 0.35 |
| 2R | 0.72 | 0.36 | 0.95 | 0.53 |
| 3R | 0.84 | 0.45 | 1.24 | 0.75 |

All 2023/2024/2025 2x PF values remain below one. Top-five and top-ten removal make the result worse.

The median R14 stop distance is only 0.169% in discovery and 0.181% in validation. A 2x round trip costs 0.22%, so cost exceeds median planned risk. Median first-passage holding time is zero minutes: most rows resolve on the entry bar.

This explains why the preceding MFE/risk screen looked more promising. MFE can include a favorable intrabar excursion on a bar that also reaches the stop. With unknown OHLC ordering, exact stop-first first passage correctly counts that bar as a loss. MFE alone was not actionable evidence.

## Frozen conclusions

1. Fixed 0.5R–3R targets do not rescue SSL root acceptance.
2. The root-region touch stop is economically too tight relative to ETH 1m noise and costs.
3. Do not optimize an intermediate R target or build Base+Runner from this ladder.
4. Same-bar MFE must never substitute for first-passage ordering.
5. Holdout remains sealed; no strategy is promoted.

## Next hypothesis

R16 will audit whether R14 encoded thesis invalidation incorrectly. It will keep the frozen SSL root-close entry and deeper same-side structural target, and compare only:

- region-edge touch stop (R14 baseline);
- root sweep-bar extreme plus 2bps hard stop;
- causal close reclaim of the region, executed next open, protected by the same root-bar-extreme disaster stop.

This is a stop-semantics atlas, not an admission or target search. If wider/behavioral invalidation does not produce stable 2x economics and top-winner resilience, the whole acceptance-continuation branch stops at the following strategic reset.

## Primary evidence

- `data/reports/research/ict/mss2/r15_acceptance_fixed_r_first_passage/04_fixed_r_first_passage_rows.csv.gz`
- `05_fixed_r_scorecard.csv`
- `06_fixed_r_years.csv`
- `07_causal_audit.csv`
