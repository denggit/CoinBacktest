# R17 Precommitment — Trend Pullback Re-acceleration Path Atlas

Date frozen: 2026-08-16, before any R17 outcome calculation.

## Independent mechanism

R17 does not repair the completed-trend sweep branch, q70 reclaim, directional impulse breakout, or the earlier broad Higher-Low limit-entry study. It tests one new sequence:

```text
causal aligned 1D + 4H structural trend
→ local 30m counter-trend pivot/pullback
→ 15m close reclaims the 30m pivot-bar range
→ later 5m close breaks the reclaim bar in trend direction
→ enter at the next observable 1m open
```

A breakout is evidence that the higher-timeframe trend exists. It is never the entry event.

## Structural definitions

- A timeframe pivot is a strict order-1 three-bar pivot.
- The pivot becomes available only when the full right-hand bar has closed.
- A bullish timeframe state requires the two latest causally confirmed pivot highs and pivot lows both to be higher. A bearish state is the exact lower-high/lower-low mirror.
- Long setups require bullish 1D and bullish 4H state both when the 30m pullback becomes available and when the final 5m signal closes. Short setups require the bearish mirror.
- Long pullbacks use a confirmed 30m pivot low; shorts use a confirmed 30m pivot high.
- The 15m reclaim is the first closed 15m bar at or after pullback availability whose close is beyond the 30m pivot-bar high for Long or below its low for Short.
- Re-acceleration is the first later closed 5m bar whose close breaks the reclaim-bar high for Long or low for Short.
- A setup expires 12 hours after the 30m pivot becomes available or when a newer same-side 30m pivot becomes available, whichever occurs first.
- If several pullbacks produce the same direction and final signal timestamp, keep the most recently available pullback.

## Entry, target, stop, and ordering

- The closed 5m signal executes at the 1m open timestamped exactly at the 5m close boundary.
- The structural runway target is the latest causally confirmed 4H pivot high for Long or pivot low for Short, frozen at signal time. It must still lie beyond entry.
- The hard stop is the 30m pullback extreme plus a `0.25 ×` causal 30m ATR buffer.
- Maximum allowed stop distance is `1.50%` of entry. Wider setups are skipped, not resized into a different thesis.
- R17 labels the structural target plus fixed 1R, 2R, and 3R first-passage paths.
- Path horizon is 72 hours. A path unresolved at 72 hours is marked `horizon_exit`; this is a diagnostic mark, not a proposed final strategy time stop.
- If target and stop are both touched in one 1m OHLC bar, the stop wins.
- Long and Short are never pooled for the primary evidence tables.

## Costs and splits

- Round-trip market cost: 0.11%.
- Report 1x, 2x, and 3x cost.
- Discovery: 2023-01-01 through 2024-12-31.
- Validation: 2025-01-01 through 2025-06-30.
- July 2025 is embargoed.
- Existing MSS2 holdout begins 2025-08-01 and remains sealed. R17 may report only aggregate sealed candidate counts, never holdout path or economic results.

## Decision boundary

R17 is a mechanism/path atlas, not a portfolio or live strategy. No feature bin, filter, target, or stop may be selected from holdout. R17 can justify a later execution/backtest version only if the same simple event has positive 2x-cost evidence in both discovery and validation, is not dependent on one year or the top five winners, and has enough independent events to be credible. Otherwise the exact branch is frozen or rejected without rescue tuning.
