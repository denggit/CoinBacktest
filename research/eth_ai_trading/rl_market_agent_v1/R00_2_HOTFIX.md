# R00.2 K-line coverage hotfix

Observed production-local failure:

- 2026-06 reached only ~49%-63% aligned coverage for official 5m/15m/1H/4H/1D K-line contexts.
- The official K-line caches ended around 2026-06-15 to 2026-06-17.
- R00 had already completed 41/42 shards, so the duplicate K-line cache was the blocker rather than the shard/cache mechanism.

Fix:

1. Keep `trade_1m` mandatory.
2. Load enough 1m left context for the longest fixed-bar rolling feature (60 days for the 48-bar 1D state).
3. Prefer official K-lines for each timestamp.
4. Fill only missing fixed-bar timestamps from local 1m trade bars using complete, left-labeled OHLCV buckets.
5. Preserve `available_time = bar_start + timeframe` before decision alignment.
6. Record fallback row counts in source coverage notes.
7. Do not silently mark missing K-lines optional and do not download/rebuild local history.

Existing successful shards remain reusable. Because the failed 2026-06 shard was never persisted, rerunning the normal command skips the prior 41 shards and rebuilds only the failed month.
