# R26 Precommitment — Relative Positioning Leadership Repricing

Date frozen: 2026-08-17, before any R26 price-path outcome calculation.

## Independent mechanism

R26 tests whether large-position leadership moves before broad-account
positioning during a directional repricing:

```text
Binance top-trader position long share crosses global-account long share
→ the relative lead keeps its new sign
→ first same-direction completed OKX 5m price confirmation within one hour
→ next observable OKX 1m open
```

A cross from nonpositive to positive relative spread arms Long. A cross from
nonnegative to negative arms Short. This is not an R18/R19 rescue: R26 does not
condition on base-OI level/change, an OI release/rebuild, liquidity sweeps,
taker flow, funding, session, volatility regime, or any prior failed setup.

## Source gate frozen before outcomes

The proposed OKX spot/perpetual study was rejected because local reads through
`src.data_feed` returned zero spot rows. OKX funding, mark, liquidation, and
books also lack usable visible history. No download or proxy substitution is
allowed.

The official Binance USD-M `ETHUSDT` 5-minute metrics cache contains 262,341
visible rows from 2023-01-01 through 2025-06-30, with zero duplicate timestamps.
Top-trader position share has 35 null rows and global-account share has 27;
262,303 rows have both. There are 416 non-five-minute intervals, including one
10.5-hour gap. Exact gap-safe raw crosses number 516 Long and 515 Short before
price confirmation, with both directions represented in 2023, 2024, and
2025H1. Invalid shares and gap edges are excluded without interpolation.

R05 aligned these ratios as descriptive post-sweep context but did not test a
standalone relative-spread cross. R18/R19 explicitly excluded all ratio fields.
A repository search found no equivalent standalone strategy.

## Frozen causal event contract

- Binance metrics are loaded only through
  `src.data_feed.binance_futures_metrics_loader`.
- OKX `ETH-USDT-SWAP` price and execution bars are loaded only through
  `src.data_feed.okx_loader`.
- `metric_available_time = source timestamp + 1 minute`.
- Both current and prior top-trader/global shares must be finite and within
  `[0, 1]`. Their source timestamps must be 4–6 minutes apart.
- `relative_spread = top_trader_position_long_share - global_account_long_share`.
- Long arms only on `prior_spread <= 0` and `current_spread > 0`; Short is the
  exact `prior_spread >= 0` and `current_spread < 0` mirror.
- The cross freezes the completed OKX one-hour high/low range already available
  at the cross.
- For at most 60 minutes after the cross, each subsequent metric observation
  must remain gap-safe and keep the new spread sign. A recross, invalid row, or
  gap cancels the episode.
- Long confirms at the first completed OKX 5m close strictly above the
  immediately prior completed 5m high. Short confirms below the prior low.
- If no confirmation occurs within 60 minutes, the episode expires. A later
  confirmation cannot rescue it.
- Signal time is the later of the confirming metric availability and completed
  price-bar availability. Entry is the first OKX 1m open at or after signal.
- Long and Short are evaluated separately. Within each split, direction, and
  target model, a new position is ignored until the prior position has exited.

## Frozen stop, targets, and ordering

- Causal volatility is the simple mean true range of the latest 12 completed
  OKX 5m bars.
- Long stop is the lower low of the confirmation bar and its immediately prior
  5m bar minus `0.25 × ATR(12)`. Short is the mirror.
- Setups with invalid geometry or stop distance above 1.50% are skipped.
- Primary structural target is the direction-side extreme of the one-hour range
  frozen at the leadership cross: cross-time high for Long, low for Short. It
  must still lie beyond entry.
- 1R, 2R, and 3R are diagnostic target paths only and cannot replace the
  structural target after results are known.
- The diagnostic path horizon is 24 hours. Unresolved paths use the last
  visible close and are labelled `horizon_exit`; this is not a proposed live
  time stop.
- Discovery paths must exit before 2025-01-01. Validation paths must exit before
  2025-07-01. Boundary-crossing paths are censored without outcome calculation.
- Same-minute target/stop ambiguity is stop-first.
- Round-trip market cost is 0.11%; report 1×, 2×, and 3× costs.

## Physical split and holdout seal

- Warmup: 2022-01-01 onward.
- Discovery: 2023-01-01 through 2024-12-31.
- Validation: 2025-01-01 through 2025-06-30.
- The R26 process physically loads nothing at or after 2025-07-01.
- July 2025 remains embargoed.
- Holdout beginning 2025-08-01 remains sealed; no candidate, price path,
  economics, or ranking is computed there.
- Causal event artifacts reject columns beginning `future_` or containing
  `oracle`.

## Decision boundary

The primary decision surface is the unfiltered structural-target path, Long and
Short separately. A direction may advance only if:

- gross and 2×-cost expectancy are positive in both discovery and validation;
- 1× PF is at least 1.40 and 2× PF exceeds 1.00 in both splits;
- discovery has at least 100 non-overlapping trades and validation at least 30;
- each visible year is positive at 2× cost;
- discovery and validation remain positive after removing the top five winners;
- no causal or independent replay check fails.

Otherwise reject the direction immediately. Do not tune the sign threshold,
cross definition, 60-minute confirmation window, price confirmation, ratio
magnitude, stop, target, session, OI, taker flow, or add a model/filter rescue.
If both directions fail, archive the relative-positioning ratio branch.
