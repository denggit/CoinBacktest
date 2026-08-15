# R01.1 Full-History Memory and Resume Hotfix

## Failure

The full 2023-01-01 through 2026-06-30 run completed all 639 two-day chunks in about three hours and produced about 2.43 million candidate events. It then failed before clustering/reporting because Pandas attempted to consolidate and deep-copy 384 float64 columns, requiring an additional contiguous allocation of approximately 6.96 GiB.

This was an execution-engine failure, not a negative research result.

## Frozen research behavior

The following remain unchanged:

- broad liquidity-release candidate union;
- pre-event 1-second and completed 1-minute path features;
- 15m/30m/1H/4H/1D all-unswept Swing inventory as supplementary context only;
- DOWN/UP independent release Episodes and 45-second Episode gap;
- 600-second future-path labels;
- 12 discovery clusters and 2024-12-31 training cutoff;
- no entry, stop, TP, leverage, fee or account-return optimization.

## New bounded-memory execution

1. Each small chunk is consolidated and safely downcast before being retained.
2. Complete feature/label matching happens inside the small chunk.
3. The full frame is never sorted and deep-copied for global Episode assignment.
4. Cluster fitting uses at most 250,000 deterministically stratified eligible training rows.
5. Cluster assignment runs in 50,000-row batches.
6. Summary reports join only required columns.
7. Descriptive medians use a disclosed fixed stratified sample capped at 250,000 rows.
8. Full tables are written to gzip in 50,000-row chunks with progress output.
9. Every completed two-day chunk is cached, allowing future runs to resume after finalization failures.

## Rerun

```bat
python research\eth_ai_trading\eth_latent_liquidity_path_v1\01_release_reversal_path_atlas.py --start-date 2023-01-01 --end-date "2026-06-30 23:59:59"
```

The old failed process did not write chunk checkpoints, so that specific three-hour in-memory result cannot be recovered. The first run after this hotfix must recompute the chunks. Every completed chunk from the new run will be reusable afterward.

Do not use `--no-chunk-cache` for the full-history run. The cache may consume several gigabytes; it can be deleted after the final report is verified.
