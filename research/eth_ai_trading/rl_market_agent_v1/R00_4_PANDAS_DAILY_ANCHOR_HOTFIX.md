# R00.4 Pandas daily-anchor compatibility hotfix

## Trigger

A real local run under the user's current Pandas emitted:

`RuntimeWarning: The 'offset' keyword does not take effect when resampling with a 'freq' that is not Tick-like`

This affected the active official-1m -> 1D resample path. Continuing R00.3 could therefore anchor daily bars at 00:00 instead of the intended local +08:00 boundary.

## Fix

- Do not pass `offset=` to daily `DataFrame.resample`.
- For 1D only, shift the 1m index by `-daily_offset`, resample at midnight, then shift the aggregated bar-start index back by `+daily_offset`.
- Keep the same complete-bucket gate (1440 source 1m rows required).
- Keep later `available_time` alignment unchanged, so an 08:00 daily bar is not visible until the following 08:00.
- No data loader, labels, trading logic, reward, or model code changed.

## Cache safety

Dataset revision is bumped from `R00.3` to `R00.4` and the cache root moves from `r00_3` to `r00_4`. R00.3 shards are intentionally not reused because daily feature timestamps may be wrong under the user's Pandas version.
