# Human Trader Replay Lab V1.4 — Full Context + Limit Orders

## Frozen workflow

- SOXL uses local OKX `SOXL-USDT-SWAP` 1m data.
- A new Episode can only choose a weekday and always starts at 07:30 New York time.
- **Weekday eligibility is not a chart filter.** Historical context displays every local OKX bar available before the decision cursor, including weekday off-hours and weekend bars if present in the local table.
- Default panes remain 30m / 15m for setup and 2m / 1m for execution.
- All panes share annotations; source timeframe and anchor candle are retained.
- Magnet remains enabled by default and snaps to O/H/L/C.

## Entry workflow

LIMIT is now the default order type. The user can wait for an FVG/reference candle to close, click its Low/High, and place a LONG/SHORT limit at that selected price. Orders stay pending until a later causally-known 1m bar touches the limit. MARKET is still optional. Pending orders can be cancelled.

## Causality

- Higher-timeframe bars are returned only when `bar_start + timeframe <= cursor`.
- A resting limit is checked only after replay advances; the newly closed 1m interval is then eligible for a fill decision.
- No later OHLC is used at order placement time.

## Performance

The local OKX 1m table is loaded once into process memory. Replay-day windows and resampled frames are cached; normal playback sends incremental bars rather than rebuilding full chart history every minute.
