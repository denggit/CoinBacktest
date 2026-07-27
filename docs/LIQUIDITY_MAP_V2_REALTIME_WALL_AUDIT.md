# Liquidity Map V2.0 Realtime Snapshot Wall Audit

## Purpose

V2.0 replaces the experimental V1 persistent-wall detector. It is only for
human audit of wall detection. It is not a trading signal and must not be used
for backtest/live order placement yet.

## Detector definition

For every causal order-book snapshot:

1. Split bid and ask sides.
2. Restrict levels to the configured distance from current mid price.
3. Compute current same-side non-zero median depth.
4. Mark a thick point when its depth is at least `snapshot_depth_multiplier`
   times the current median.
5. Mark a wall core when the thick point is a local peak and is sufficiently
   larger than the local neighbourhood, or when it exceeds the isolated-core
   multiplier.
6. Join nearby thick points only inside the same snapshot. Small thin gaps are
   allowed, but cluster width is capped.
7. Track the resulting point wall / wall zone across short time intervals using
   side, price overlap and centre movement.
8. Confirm only after short persistence and minimum appearance ratio.

There is no 24-hour historical wall threshold and no two-hour confirmation.

## Causality

Every displayed yellow slice uses the price range visible in that exact
snapshot. Later wall expansion or movement cannot rewrite an earlier slice.
The plugin does not draw one final hindsight rectangle across the whole wall
lifecycle.

## Current limitations

V2.0 detects wall candidates only. It does not yet classify:

- trade attack;
- actual consumption;
- cancellation / pre-touch withdrawal;
- replenishment;
- absorption;
- break and reclaim.

Those require Raw Trades and causal depth-flow attribution in a separate V2.1
research layer.

## Recommended first audit settings

- Chart timeframe: 1m or 15m (display only)
- Wall resolution: 15s first; switch to 5s for close inspection
- Thick point / median: 3x
- Isolated core: 5x
- Local contrast: 3x
- Minimum confirmation: 20s
- Presence ratio: 70%
- Missing snapshots: 2
- Price gaps inside zone: 2 bins

The `墙` overlay remains disabled by default.
