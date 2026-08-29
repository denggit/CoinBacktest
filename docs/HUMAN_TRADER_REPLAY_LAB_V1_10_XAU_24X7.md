# Human Trader Replay Lab V1.10 — XAU 24/7

## Goal

Add OKX gold perpetual `XAU-USDT-SWAP` to the Human Trader Replay Lab without inheriting SOXL equity-session restrictions.

## Profile

- `SOXL-USDT-SWAP`: weekday Episode, 07:30-16:00 America/New_York.
- `ETH-USDT-SWAP`: 24/7, Beijing random/specific start, TP/SL auto-finalize.
- `XAU-USDT-SWAP`: 24/7, Beijing random/specific start, TP/SL auto-finalize.

Chart context remains all locally available OKX 1m bars. XAU is selectable only after `data/crypto_history.db` contains `XAU_USDT_SWAP_1m`.

## Local XAU OHLCV

```bash
python tools/prebuild_okx_ohlcv.py --symbol XAU-USDT-SWAP --timeframe 1m --start-date 2026-01-15 --end-date 2026-08-24
```

The tool delegates all exchange access to `src.data_feed.OKXDataLoader`.
