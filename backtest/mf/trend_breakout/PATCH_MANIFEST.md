# ETH Portfolio V2 / Trend Breakout V1 patch manifest

## What changed

- Added an executable strategy contract layer so an event study cannot be
  promoted as a strategy without entry/stop/exit/sizing rules.
- Added a generic funnel audit that distinguishes hard-filter collapse from
  execution/occupancy loss.
- Added the five frozen ETH Portfolio V2 core sleeve destinations.
- Added continuous risk-multiplier support to the generic one-position OHLCV
  replay engine (default behavior remains 1.0 and is backward compatible).
- Added Trend Breakout V1 as the first actual Portfolio V2 strategy.
- Added fee/slippage/delay stress, yearly/quarterly/monthly reports, top-trade
  deletion, fixed parameter-neighbourhood replay and a deterministic decision
  gate.

## Important design choice

Trend Breakout V1 does **not** hard-filter on every quality feature.  Base
structure breaks remain signals; trend alignment, breakout depth, body quality
and close location scale risk.  This directly addresses the historical
"thousands of events -> dozens of trades" failure mode.

## Causal safeguards

- Prior structure breakout levels are shifted by one completed bar.
- Signals are based on fully closed 15m bars.
- Entries occur on the next 15m open.
- Stops use only signal-bar-or-earlier structure.
- No higher-timeframe context is joined in V1.
- Same-bar stop/target ambiguity remains conservative stop-first.
- Existing generic close-based ATR trailing is disabled for this strategy to
  avoid applying a close-known trail to an earlier same-bar low/high.
- Opposite-signal and max-hold close exits are also disabled in V1 because a
  close-derived decision cannot honestly receive that same close as fill.

## Tests added

- funnel collapse detection
- strategy contract completeness
- future mutation causality
- quality-score event retention
- next-open entry
- dynamic risk multiplier sizing
