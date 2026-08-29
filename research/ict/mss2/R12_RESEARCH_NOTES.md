# R12 Research Notes — Completed-Trend Swing Sweep -> Opposite Liquidity

## Why R12

Recent work became too admission-driven. R12 intentionally stops optimizing a strategy and returns to a path-first question: after a meaningful historical Swing from a completed ICT trend is swept, when does price actually deliver to the other side and when does it continue through deeper same-side structure?

## Universe

- R08.1 completed-trend native and nested-lower-TF ITH/ITL/LTH/LTL only.
- ST construction swings excluded.
- Invalid higher-TF -> lower-trend projection excluded.
- Only levels active when a completed-trend context became causally available are eligible.
- One physical `swing_id` is counted once even if several completed trends later reference it.
- At a root sweep, only trend contexts already active by that root may contribute features.

## Path semantics

The sweep is known at root-bar close. The first-passage race begins on the next 1m bar, never inside the root bar. At root close freeze:

1. nearest opposite completed-trend liquidity region (plus second/third for target ladder),
2. nearest deeper same-side completed-trend liquidity region.

Primary classification is the order in which those frozen barriers are reached. Thirty days is censoring only; it is not a time exit.

## Features for later discovery

Pre-sweep: Swing timeframe / IT-vs-LT role / completed-trend context / trend size / age / prior 5m/15m/60m attack.

Sweep: levels/regions consumed, sweep depth, bar range, rejection wick, close location, same-bar reclaim.

Early post-sweep: reclaim, 1m/2m/5m post-sweep ST-MSS, first directional FVG. These are future path landmarks and cannot be used to alter root-event features.

## Explicit non-goals

R12 does not choose an entry, structural stop, TP, risk tier, session, or portfolio. It does not use UTC+8 midnight as a market boundary. The next strategy version is allowed only after successful and failed liquidity-to-liquidity paths show a robust causal separation.
