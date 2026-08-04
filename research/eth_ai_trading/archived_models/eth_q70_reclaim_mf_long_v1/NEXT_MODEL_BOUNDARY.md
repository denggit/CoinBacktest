# Boundary for the Next Independent Model

The next model must not be presented as a repair or continuation of V1. It may reuse public data loaders, causal alignment utilities, progress/report infrastructure and sealed-validation discipline, but it must not inherit the q70 threshold, V1 score, labels, entry rule or historical approval.

## Proposed next direction

Working name: **ETH Trend Pullback Continuation Long/Short V1**.

This is **not a breakout-chasing entry model**. The intended architecture is:

```text
1D/4H: estimate trend persistence, direction and remaining runway
1H/30m: identify an orderly pullback, compression or absorption phase
15m/5m: confirm support/resistance reclaim and re-acceleration
1m: execute at the next observable price
```

A breakout or volatility expansion may be used as evidence that a trend exists, but the strategy must not automatically buy the breakout high or short the breakdown low.

## Risk boundary

- Start with one entry and no add-ons.
- Anchor the stop to the local pullback structure plus a causal volatility buffer.
- Predeclare a maximum acceptable stop distance; skip trades whose structure is too far away.
- Size notional from actual hard-stop distance and account risk.
- Exchange leverage is only a margin-efficiency setting. It must not be used to justify a wider account-risk tail.
- If a multi-day trend requires a stop too wide to support the desired notional under the risk cap, the correct action is to reduce notional or skip the trade—not to pretend high leverage fixes it.

## Research objective

Capture a meaningful portion of multi-day 3%–15% ETH moves with low-risk pullback entries, separately for Long and Short, while avoiding late-stage breakout chasing.
