# Unconsumed Swing Low Liquidity Atlas R02

This research builds a broad causal pool of historical Swing Low liquidity:

- 15m, 30m, 1H, 4H and 1D;
- every order-1 causal pivot is admitted;
- order 2/3/5 confirmations are later attributes, not admission filters;
- every level remains active until its first true downward sweep or dataset end;
- touch does not consume the level;
- first sweep consumes stop liquidity;
- reclaim/acceptance below is tracked separately as support outcome;
- no arbitrary maximum age;
- no OBI, footprint, wall, volume, trend, prominence or confluence entry filter.

Default Windows command:

```bat
python research\liquidity\02_unconsumed_swing_liquidity_atlas.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --swing-timeframes 15m,30m,1H,4H,1D --confirmation-orders 1,2,3,5 --no-build-missing
```

The output is an event atlas, not a strategy backtest.  The next research pass
must test attributes one at a time before any final signal is assembled.
