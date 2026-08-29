# Human Trader Replay Lab V1.11 — From-Date Sequential Replay

## Goal

Add a chronological blind-replay workflow without removing random or one-off specific starts.

## UI

Start modes:

- `从日期顺序 Replay` (default)
- `随机`
- `指定单次起点`

For ETH/XAU the sequential start is entered in Beijing time. Example: `2026-01-01 00:00`. If that exact minute is absent locally, the first available local 1m row after it is used.

After a sequential Episode closes, the same button becomes `继续下一 Episode`. For 24/7 symbols the next Episode begins at the first locally available 1m row strictly after the previous final cursor. This keeps replay chronological and avoids jumping to random dates.

SOXL keeps its session semantics: sequential start picks the first available weekday on/after the chosen date at 07:30 ET; subsequent Episodes advance to the next available weekday at 07:30 ET.

## Causality

Finding the next available timestamp is not a signal-generation operation. Only a timestamp is selected; future OHLC values are not returned to the UI. Once an Episode begins, the existing causal replay rules remain unchanged.

## ETH backfill command

```bash
python tools/prebuild_okx_ohlcv.py --symbol ETH-USDT-SWAP --timeframe 1m --start-date "2026-07-01 00:00:00" --end-date "2026-08-24 23:59:59"
```
