# Liquidity Primitives V1 Runbook

## Purpose

This layer accelerates liquidity-wall research without deciding what a wall is.
It converts the existing canonical 5-second liquidity-map day artifacts into
sorted NumPy arrays with snapshot offsets and causal relative-depth summaries.

It does **not**:

- download Books;
- rebuild the offline liquidity map;
- label walls;
- use future returns;
- fix candidate widths, persistence or wall thresholds.

## Cached information

Per price cell:

- price index and side;
- completed bucket-end depth;
- added, removed, executed, cancelled, consumed and replenished amounts;
- flow-valid flag.

Per snapshot:

- best bid, best ask and midpoint;
- bid/ask Q25, Q50 and Q75 depth baselines;
- bid/ask total and maximum depth;
- snapshot Q95/Q99 depth;
- causal rolling 24-hour Q95/Q99 reference;
- reference warm-up fraction;
- row offsets into the flat cell arrays.

Absolute depth remains present. Relative depth should be the primary cross-period
measure, while absolute depth/notional remains an auxiliary feature.

## Storage

```text
data/okx/derived/liquidity_primitives/
  ETH-USDT-SWAP/
    books_5000/
      v1/
        YYYY/MM/YYYY-MM-DD.primitives.npz
        YYYY/MM/YYYY-MM-DD.metadata.json
```

Metadata is published last and acts as the day checkpoint. Interrupted days are
rebuilt; completed days are skipped unless `--force` is passed.

## Run

```text
python tools\prebuild_okx_liquidity_primitives.py --symbol ETH-USDT-SWAP --start-date 2026-01-01 --end-date 2026-06-30 --books-depth 5000 --cache-version v1
```

The command reads existing liquidity-map NPZ files only. The preceding UTC day
is used as reference warm-up when available but is not written unless it belongs
to the requested range.

## Wall discovery from cache

```text
python research\liquidity\liquidity_wall_discovery_v2\01_liquidity_wall_discovery_from_primitives.py --symbol ETH-USDT-SWAP --start-date 2026-01-01 --end-date "2026-06-30 23:59:59" --books-depth 5000 --primitive-cache-version v1 --touch-timeframe 5s --price-step 1 --out-dir data\reports\research\liquidity\liquidity_wall_discovery_v2
```

Wall widths, depth multiples, occupancy, contrast, drift and persistence remain
command-line research parameters. Changing them does not require rebuilding the
primitive cache.

## Causal limits

- Snapshot features are available at `bucket_end_ms` only.
- Rolling references contain current/past completed snapshots only.
- The cache contains no touch outcome or future price label.
- Current schema supports causal rolling reference quantiles Q95 and Q99.
- Touch outcome research still requires closed data and next-bar execution where
  a strategy is eventually tested.
