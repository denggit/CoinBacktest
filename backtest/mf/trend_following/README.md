# ETH Trend Following Baseline Suite

Six independent 15m ETH-USDT-SWAP trend-following baselines using the existing `src.data_feed.OKXTradeBarLoader`.

1. `donchian_breakout.py` — 24h Donchian breakout.
2. `ema_momentum.py` — EMA50/EMA200 + 12h time-series momentum.
3. `trend_pullback.py` — established EMA trend + EMA20 pullback/reclaim.
4. `market_structure.py` — causal confirmed swings + HH/HL or LH/LL structure breakout.
5. `volatility_expansion.py` — ATR compression -> range/volume expansion breakout.
6. `orderflow_trend.py` — price trend + multi-bar OKX aggressive order-flow confirmation.

## Frozen comparison assumptions

- warmup: 2022-01-01
- backtest: 2023-01-01 -> 2026-06-30
- execution: closed 15m bar signal -> next 15m open
- one position at a time
- risk/trade: 1% equity
- notional cap: 3x equity
- fee: 0.055% per side (0.11% round trip)
- slippage: 0.02% per side
- no nearby fixed TP; trend is allowed to run
- initial stop: ATR or causal structure depending on strategy
- trailing exit: ATR after the trade reaches positive R
- maximum stop distance: 3%
- maximum holding: 7 days

These are deliberately baseline values, not an optimized grid.

## Windows one-line commands

Run all six with one shared data load:

```text
python backtest\mf\trend_following\run_all.py --no-build-missing
```

Run individually:

```text
python backtest\mf\trend_following\donchian_breakout.py --no-build-missing
python backtest\mf\trend_following\ema_momentum.py --no-build-missing
python backtest\mf\trend_following\trend_pullback.py --no-build-missing
python backtest\mf\trend_following\market_structure.py --no-build-missing
python backtest\mf\trend_following\volatility_expansion.py --no-build-missing
python backtest\mf\trend_following\orderflow_trend.py --no-build-missing
```

If local trade-bar cache is missing coverage, omit `--no-build-missing` and the existing loader may build missing days from its configured source.

Outputs are written under `data/reports/backtest/mf/trend_following/` including trades, equity, signal audit, full project report, and `trend_following_comparison.csv` for the suite.
