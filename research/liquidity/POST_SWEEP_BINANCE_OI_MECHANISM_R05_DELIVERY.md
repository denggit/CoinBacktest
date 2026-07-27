# R05 Post-Sweep Binance OI Mechanism Study

## Goal

Causally enrich the R04 post-Sweep process atlas with Binance ETHUSDT USD-M 5-minute OI and positioning metrics. The study tests position-building, position-release, selling-impact failure, and short-covering-compatible future paths without fitting an entry rule.

## Run

```bat
python research\liquidity\05_post_sweep_binance_oi_mechanism_study.py --symbol ETHUSDT --start-date 2023-01-01 --end-date "2026-06-30 23:59:59"
```

R04 must already have produced its full checkpoint feature and label tables. Binance metrics must already exist in `data\binance_futures_metrics.db`.

## Causal semantics

- Binance metrics timestamps are 5-minute interval-end observations.
- `available_time = timestamp + publication_lag`, default 1 minute.
- Relative-change baselines require a near-exact 5-minute grid match (1-minute tolerance); missing official rows become NaN rather than being mislabeled as a shorter-window change.
- Every checkpoint uses the latest row with `oi_available_time <= checkpoint_available_time`.
- Maximum accepted staleness is 10 minutes, allowing one missing official 5-minute row without silently bridging long gaps.
- Future OI changes are labels and are physically separated from causal features.

## Interpretation boundary

Binance OI is an external ETH perpetual positioning proxy. It must never be called OKX OI. Price down + OI up is compatible with new position building but cannot identify direction on its own; it becomes chase-short-compatible only when combined with aggressive sell flow. Price up + future OI down is compatible with short covering, not proof of a specific participant.
