# R10 Unified ICT Liquidity Trading Engine

## Objective
Stop expanding independent ICT signal atlases and consolidate the corrected R08/R09 liquidity foundation into one coherent trade lifecycle whose success criterion is the capital curve rather than a single filtered PF.

## Frozen from R09 before R10 results
- First deployable engine remains SSL -> Long only. BSL/Short stays research-only because no >=30-trade candidate had stable positive R-PF in every year.
- One base signal per independent root episode.
- Unified base entry = 2m episode reclaim, next-open market execution.
- Structural MSS is no longer an independent trade; it is a later causal state upgrade.
- FVG remains useful execution/structure information but R10 v1 does not create a second FVG trade after the position already exists.
- Initial stop = causal sweep/reclaim structural extreme plus 2bps execution buffer. Do not tighten it simply to improve R:R.
- No add-on in R10 v1; R06 add-on V1 was rejected.

## Lifecycle variants (small frozen set, not a grid)
1. `full_5m_ltl`: comparator; full position uses causal 5m LTL trailing.
2. `base75_2r_runner25`: no early trailing; 75% Base realizes 2R, 25% Runner moves to BE from the next 1m bar, then follows 5m LTL; after structural MSS + 3R it slows to newly-confirmed 15m LTL.
3. `base50_2r_runner50`: same state machine with a larger runner, used only to measure smoothness vs right-tail tradeoff.

## Risk schedules (frozen before R10 results)
- `equal_low`: 0.50% per setup across C/B/A/A+ control.
- `quality_scaled`: C=0.35%, B=0.10%, A=0.75%, A+=0.75%.
- `quality_scaled_no_B`: C=0.35%, B=0, A=0.75%, A+=0.75%.

A+ is intentionally not given more risk than A because R09 showed 4H context was sparse and less year-stable than 1H context. These are research schedules, not final live risk settings.

## Portfolio semantics
- Single ETH net position. New independent episodes are skipped while a position is active.
- No time stop. A right-edge open position remains open/MTM rather than being force-closed at an arbitrary horizon.
- Costs: project default market round-trip 0.11%, stressed at 1x/2x/3x.
- Notional is risk-sized from actual structural stop distance and capped at 3x equity.

## Main acceptance metrics
Do not promote based on PF alone. Review 2x first: trades/month, total return, daily MTM MDD, longest underwater duration, positive-month and positive-quarter rate, rolling-90d positive rate, equity trend R2, yearly results, and equity after zeroing top-5/top-10 winners.
