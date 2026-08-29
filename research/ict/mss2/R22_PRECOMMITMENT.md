# R22 Precommitment — BTC-Led ETH Catch-Up First Passage

Date frozen: 2026-08-17, before R22 outcomes are calculated.

## Independent economic hypothesis

ETH and BTC share a strong market component. A completed BTC impulse that ETH
has begun to follow but materially under-reacted to may leave a short-lived,
causally observable catch-up path in ETH. This is cross-market information, not
an ETH breakout, sweep, pullback, OI transition, V10B, or daily-channel filter.

Repository search found no prior ETH/BTC lead-lag study. Bare ETH and BTC OKX
1m caches both contain the exact 1,838,880 aligned minutes from 2022-01-01
through 2025-06-30 23:59, with no timestamp mismatch.

## Frozen signal

All calculations use complete left-labelled 1h bars. A bar labelled `t` is
available only at `t + 1h`; entry is the ETH 1m open at that time.

- Estimate ETH beta to BTC from the prior 720 complete hourly returns. The
  signal hour itself is excluded from beta estimation.
- Estimate BTC hourly volatility from the prior 168 complete returns.
- Estimate beta-residual volatility from the prior 720 valid residuals.
- BTC impulse: absolute signal-hour BTC return is at least 2.0 prior BTC sigmas.
- Direction is the BTC return sign.
- ETH must already have a nonnegative return in that direction; opposite-sign
  divergence is excluded as possible ETH-specific information.
- Signed lag `(beta * BTC return - ETH return)` must be at least 0.75 prior
  residual sigmas in the BTC direction.

No threshold neighbor, beta window, sigma window, session, trend, volume,
funding, OI, order-flow, or volatility-regime variant may replace a failure.

## Frozen execution and exits

Long and Short are independent one-position sleeves. Overlapping signals in an
already-open same-direction sleeve are ignored.

- Initial risk: 1.5× completed signal-hour ETH Wilder ATR(20).
- Primary target `R1`: 1.0R.
- Diagnostic sensitivity `R2`: 2.0R; it cannot rescue a failed R1 primary.
- Exact 1m first passage from entry, including the entry minute.
- Same-minute stop/target ambiguity is stop-first.
- A gap through the stop exits at the worse 1m open; target fills at the frozen
  target price.
- If neither barrier is touched in 24 hours, exit at the 24h boundary 1m open.
- A path reaching a research split boundary first is censored, not force-closed.

Gross return is the unlevered signed price return. Round-trip 1×/2×/3× costs are
0.11%/0.22%/0.33%. There is no leverage, compounding, dynamic risk, add-on,
trailing stop, or portfolio netting.

## Time contract

- Warmup: 2022-01-01 onward.
- Discovery: reset at 2023-01-01 and end before 2025-01-01.
- Validation: reset at 2025-01-01 and end before 2025-07-01.
- July 2025 is embargoed.
- Holdout begins 2025-08-01 and no holdout outcome is loaded.

## Decision rule

An R1 direction is only a research candidate if:

- discovery and validation 2× PF are each at least 1.4 with positive mean net
  return;
- discovery has at least 50 completed trades and validation at least 10;
- every visible year has positive 2× sum;
- discovery top-ten-winner removal remains positive;
- 24h timeout share is at most 20% in both splits; and
- the same direction's R2 discovery mean 2× return is positive.

These are minimum research gates, not live approval. No holdout inspection or
post-result rule change may rescue a failure.

