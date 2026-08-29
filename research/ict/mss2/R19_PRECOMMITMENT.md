# R19 Precommitment — Positioning Rebuild / Continuation-Resumption Atlas

Date frozen: 2026-08-17, before any R19 outcome calculation.

## Independent mechanism

R19 tests the economically opposite state separated before R18. It is not a
filter on the failed R18 reversal event:

```text
completed 1h directional price move + rising Binance base OI
→ first causal 5m base-OI transition from nonnegative to negative
→ temporary negative-OI release episode, at most 60 minutes
→ first causal 5m base-OI transition back to nonnegative
   whose completed OKX 5m close breaks the release bar in the original direction
→ first observable OKX 1m open
```

Long requires an upward price/base-OI build and a later close above the frozen
release-bar high. Short is the exact downward mirror. Binance USD-M `ETHUSDT`
base OI remains a cross-exchange positioning proxy; price and execution remain
OKX `ETH-USDT-SWAP`.

## Frozen causal state machine

- Binance rows come only from
  `src.data_feed.binance_futures_metrics_loader`; OKX bars come only through
  `src.data_feed`.
- Metrics availability is `source timestamp + 1 minute`. Completed OKX 5m bars
  are aligned backward by their explicit end time.
- A release may arm only when the immediately prior valid observation has
  rising 1h base OI and directional 1h OKX price: both positive for Long, price
  negative/OI positive for Short.
- Release is the first 5m base-OI sign transition from nonnegative to negative.
  No price response on the release row is used as an admission filter.
- The release row freezes its completed OKX 5m high/low and the one-hour build
  range already visible at the prior observation.
- The release episode may contain only subsequent negative 5m base-OI changes.
  The first subsequent nonnegative 5m base-OI observation is the only rebuild
  observation considered. If it arrives more than 60 minutes after release,
  crosses a metric gap outside 4–6 minutes, or lacks a gap-safe baseline, the
  setup expires.
- At that first rebuild observation, Long requires the latest completed OKX 5m
  close strictly above the frozen release-bar high. Short requires a strict
  close below the release-bar low. If the first rebuild does not break, the
  setup fails; later positive-OI bars cannot rescue it.
- Current and baseline base OI must be strictly positive. OI USD, taker and
  account ratios, funding, magnitude thresholds, future OI, and oracle fields
  are excluded from admission.
- Signal time is the later of rebuild publication and price-bar completion.
  Entry is the first OKX 1m open at or after signal time. Seconds-after-boundary
  publication waits for the following minute.
- One event is allowed per release episode. Long and Short stay separate.

## Frozen stop, targets, and ordering

- Causal volatility is the 12-bar simple mean true range on completed OKX 5m
  bars, with all 12 observations required.
- Long stop is the lowest completed OKX 5m low from release through rebuild,
  minus `0.25 ×` rebuild-time 5m ATR. Short uses the highest high plus the same
  buffer.
- Maximum stop distance is `1.50%` of entry; invalid or wider setups are skipped.
- Primary volatility target is entry plus/minus one full causal 1h high-low
  range measured from the 12 completed 5m bars available at rebuild. This
  target is frozen before entry and is labelled `H0_1H_VOLATILITY_RANGE`.
- Also report 1R, 2R, and 3R diagnostic barriers. No target is selected from
  outcomes.
- Diagnostic horizon is 24 hours. Full validation paths that would enter the
  July embargo are censored without outcome calculation.
- Same-bar target and stop is stop-first.
- Round-trip market cost is 0.11%; report 1×, 2×, and 3× cost.

## Data quality and splits

R19 inherits the pre-outcome R18 source audit rather than reopening data-quality
choices after economics: base-OI invalid rows are excluded, partial/irregular
intervals are never interpolated, and every transition edge must be 4–6 minutes.

- Warmup: 2022 onward.
- Discovery: 2023-01-01 through 2024-12-31.
- Validation: 2025-01-01 through 2025-06-30.
- July 2025: embargoed.
- Holdout begins 2025-08-01 and remains sealed. Only aggregate candidate counts
  may be emitted; no holdout path or economics may be calculated.
- Causal feature artifacts physically reject columns beginning `future_` or
  containing `oracle`.

## Decision boundary

R19 is a path atlas, not a strategy or target search. A direction can justify a
later strategy version only if one predeclared path has positive 2×-cost evidence
in both discovery and validation, credible sample size, yearly breadth, and
top-five resilience. Otherwise reject that direction without tuning the 60m
window, sign definitions, ATR buffer, stop ceiling, target, ratios, magnitude,
or regime. If both directions fail, the positioning-transition branch is closed.
