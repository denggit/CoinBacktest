# R09 — ICT Liquidity Quality × Execution Atlas

## Decision question
Can the corrected R08.1 full-trend liquidity foundation preserve a broad opportunity stream while using structural context as a **risk-quality ladder** rather than another hard filter, and which causal execution method best converts the underlying sweep drift into a tradable stop-based distribution?

## Frozen research semantics
- Source liquidity: R08.1 canonical native IT/LT + separately-labelled nested lower-TF IT/LT only. Rejected higher-TF -> lower-trend projections never enter R09.
- Independent opportunity: first same-side physical sweep after a 15-minute inactivity gap.
- Root quality is frozen at the first sweep minute. Later 15-minute cascade fields are outcome/path diagnostics only.
- Structural context tiers are not fit to returns: C=15m, B=30m, A=1H, A+=4H completed-trend context.
- No ST-only swing is restored as trade liquidity.
- No NY/London session admission gate.

## Execution families
1. `sweep_immediate`: next 1m open after the sweep bar is fully known.
2. `episode_reclaim`: next open after execution-TF close reclaims the swept root price set.
3. `mss_structural_market`: causal pre-sweep structural MSS.
4. `mss_post_sweep_st_market`: causal post-sweep ST forms, confirms, then a later close breaks it.
5. `reclaim_then_fvg_limit`: after reclaim, wait for a directional FVG and rest proximal/CE limit.
6. R02 MSS+FVG limit variants: MSS confirms first; limit rests at the causal FVG, no market chase.

## Outcome contract
- Structural stop = causal sweep/confirmation extreme + 2bps execution buffer.
- Stop-first pessimism on same-bar stop/target collisions.
- FVG-limit target cannot be credited on the fill bar; stop may still trigger on that bar.
- Fixed R = 0.5/1/2/3/5R and fixed percentage opportunities = 0.5/1/2/3/5% are diagnostics, not final TP rules.
- 7-day horizon is censoring only.
- Report both net-percent PF/expectancy and **risk-normalized R PF/expectancy**.

## Required interpretation
Do not choose the maximum-PF row.  Review in this order:
1. broad event count / month and SSL vs BSL split;
2. fill/coverage by execution method;
3. 2x cost expectancy and PF in both percent and R units;
4. 2023/2024/2025/2026 stability;
5. whether higher context tiers deserve higher risk without deleting lower tiers;
6. whether future cascade diagnostics identify persistent breakdown versus precision sweep;
7. MAE/MFE and right-tail retention.

## Manual chart review
Use `manual_review/01_recent_20_root_sweep_events.csv` first, then the recent 10 execution files. `structural_extreme_pre_entry` is the actual chart extreme; `stop_price` includes the execution buffer.
