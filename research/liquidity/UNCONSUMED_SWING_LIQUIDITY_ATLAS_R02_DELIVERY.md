# R02 Delivery Notes — Unconsumed Swing Low Liquidity Atlas

## Scope

This delivery creates a broad causal structural atlas, not a final strategy backtest.

It includes:

- 15m, 30m, 1H, 4H and 1D Swing Lows;
- every causal order-1 pivot as the initial universe;
- later order-2/3/5 confirmations as time-aware attributes;
- an active unconsumed-level pool with no arbitrary age expiry;
- approach, touch, first sweep and reclaim events;
- separate stop-liquidity consumption and support reclaim/acceptance states;
- cross-timeframe active-level confluence counts;
- next-open forward close paths at 5/15/30/60/180 minutes;
- fixed-period summaries and causal audits;
- GPT review-pack generation.

No OBI, footprint, Books, trend, volume, prominence, age or confluence condition is used to admit a Swing Low.

## Default Windows command

```bat
python research\liquidity\02_unconsumed_swing_liquidity_atlas.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --swing-timeframes 15m,30m,1H,4H,1D --confirmation-orders 1,2,3,5 --no-build-missing
```

## Output directory

```text
data\reports\research\liquidity\unconsumed_swing_liquidity_atlas_r02
```

## Validation performed

- New atlas tests: 11 passed.
- Related causal alignment, Liquidity Map loader/builder, prebuild speed and review-pack tests: 31 passed total.
- Built-in self-test passed.
- `compileall` passed.
- Import-boundary scan found no violations caused by the new files.
- Real-project OHLCV smoke run over 2026-03-01 to 2026-06-30 produced:
  - 5,332 causal Swing Low levels;
  - 20,609 approach/touch/sweep/reclaim events;
  - 5,308 first sweeps;
  - 24 levels still unconsumed at dataset end;
  - zero causal-audit violations.

The smoke run validates code paths and event scale only. It is not a profitability claim and does not replace the full local trade-bar run.

## Full-suite limitation in the uploaded project archive

The complete `pytest` collection is blocked by pre-existing missing research directories in the uploaded archive, including:

- `research/liquidity/panic_selloff_rejection_recovery_long`
- `research/liquidity/liquidity_touch_rebound_v1`

Those collection errors are unrelated to this R02 delivery.
