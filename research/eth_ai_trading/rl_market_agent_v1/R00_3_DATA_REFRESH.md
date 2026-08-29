# R00.3 Data Refresh / 2026-08-15

## Why this revision exists

Local source data was prebuilt through 2026-08-15. R00.3 updates the clean-sheet dataset boundary and removes the short-lived assumption that independent higher-timeframe K-line tables must all have identical refresh dates.

## Frozen changes

1. `research_end` is now `2026-08-15 23:59:59`.
2. `2026-01-01` remains the sealed holdout boundary. The new July/August data extends the sealed holdout; it does **not** become training/tuning data.
3. Official K-line context now uses the locally prebuilt official **1m K-line** as its single base and causally resamples it to `5m/15m/1H/4H/1D`.
4. Tick-derived 1m/5s trade bars remain independent microstructure sources and are no longer used to patch K-line context.
5. The last `max(label_horizon)` tail of raw data is reserved only for forward labels. No decision row is persisted unless its full largest-horizon label can be observed without reading after `research_end`.
6. R00.3 uses a new cache root `.../r00_3` because the K-line source semantics changed. R00/R00.2 shards are intentionally not reused; all R00.3 shards are internally consistent and then resumable within this revision.

## Causality

All resampled K-lines remain left-labeled and are aligned by `bar_available_time = bar_start + timeframe`. The 8-hour daily offset follows the project timezone convention.
