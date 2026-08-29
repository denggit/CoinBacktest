# R03 Deep Research Notes

## Why this version exists

R02 established that a small but persistent long-side candidate appears after rapid consumption of several independent sell-side liquidity pools, especially when the stack reaches higher-timeframe liquidity. The user also noted that many profitable ETH trades require long holding periods and that index-style NY-open assumptions should not be carried into a 24/7 crypto market.

R03 therefore stays crypto-native: liquidity-stack structure first, microstructure second, execution third.

## Working mechanism

The proposed mechanism is not “FVG causes reversal”. It is:

```text
multiple old / multi-TF sell-side liquidity pools accumulate
-> a fast liquidation impulse consumes several independent pools
-> aggressive selling increases but price makes less incremental downside
-> Delta / close-location starts improving
-> exhaustion / absorption becomes visible
-> causal reclaim / FVG execution enters
-> structural stop remains beyond the liquidation extreme
-> opposing active 4H liquidity is the primary draw
```

Trade-bar and footprint data are useful only if they improve this mechanism in a forward-stable way. They should not become a large feature soup.

## Frequency question

The R02 core >=4-pool 5m candidate has only 269 historical trades. R03 does not solve this by weakening every structural rule. The only expansion is >=3 pools, which has much higher raw frequency but is negative without additional information. The main frequency question is therefore:

> Can order-flow / footprint evidence identify a materially larger subset of >=3-pool episodes that retains the >=4-pool economics?

If not, the correct conclusion is that this is a lower-frequency sleeve, not that the thresholds should keep being relaxed.

## Entry research

R03 deliberately distinguishes:

- episode reclaim: earliest structure-native confirmation from R02;
- first post-stack FVG market: do not wait for a retracement;
- first post-stack FVG proximal limit: better price if filled;
- 50/50 hybrid: partial participation in runaway reversals while preserving some price improvement on retraces.

The FVG overlay uses the same frozen 4H target at signal time. This avoids giving the limit version a different future liquidity book simply because it filled later.

## Exit research

R03 keeps the R02 target hierarchy report, with 4H+ as the primary target. No short fixed time exit is reintroduced. The output continues to show holding duration and >1d participation because the right tail is part of the observed ETH edge candidate.

## Stop rules for this branch

Stop or de-prioritize the microstructure upgrade if:

- trade-bar mechanism flags do not improve 2025-2026 relative to their own broad >=3 matched baseline;
- footprint uplift appears only because coverage starts in a favorable regime;
- execution improvement exists only in one timeframe/year;
- market/limit differences disappear under realistic execution-cost stress;
- sample-size recovery requires stacking several post-hoc filters.
