# ETH Multi-scale Absorption

Run the full R01 atlas from the repository root:

```bash
python research/eth_absorption_multiscale/01_multiscale_absorption_floor_atlas.py
```

Default research window: 2023-01-01 through 2026-06-30, with causal warmup no earlier than 2022-01-01.

The script is cache-only. It reads existing `data/okx_trade_bars.db` through `src.data_feed.OKXTradeBarLoader` and never downloads missing data. If the local 5s table is missing, 5s is skipped by default while 1m+ scales still run.

Useful targeted runs:

```bash
python research/eth_absorption_multiscale/01_multiscale_absorption_floor_atlas.py --scales 1m 5m 15m 1H 4H
```

```bash
python research/eth_absorption_multiscale/01_multiscale_absorption_floor_atlas.py --scales 5s
```
