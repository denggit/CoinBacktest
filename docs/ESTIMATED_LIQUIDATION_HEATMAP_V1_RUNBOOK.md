# Estimated Liquidation Heatmap V1

## Positioning

This is a transparent **estimated liquidation heatmap**, not CoinGlass data and
not direct access to exchange account positions.  It combines public OKX open
interest, funding, mark price, observed liquidation events and local Trade Bar
order flow to maintain probabilistic position cohorts across leverage buckets.

It must never be described as exact liquidation prices or guaranteed price
magnets.

## Data preparation

Run from the repository root:

```bash
python tools\prebuild_okx_liquidation_inputs.py --symbol ETH-USDT-SWAP --start-date 2026-06-01 --end-date "2026-06-30 23:59:59" --oi-period 5m --mark-timeframe 1m
```

All external data interaction lives in:

```text
src/data_feed/okx_derivatives_loader.py
```

Local cache:

```text
data/okx_derivatives.db
```

OKX public endpoints have different history limits.  A zero-row dataset means
that the requested period was not returned; the loader does not synthesize it.
External archived CSV can be imported through `OKXDerivativesLoader.import_csv`.

## Analyze Tool

```bash
python analyze_tool/server.py --host 127.0.0.1 --port 8765
```

Recommended:

```text
Data type: Trade Bar
Timeframe: 1m
Plugin: 推定清算热力图 V1
```

Display:

- cyan: estimated short-liquidation potential above price
- red: estimated long-liquidation potential below price
- brighter: higher model density
- concise card: current distribution, nearest upper and lower zones, confidence

No indicator lines are shown by default.

## Model outline

1. Positive OI changes create paired probabilistic long/short cohorts.
2. Trade imbalance, recent price motion and funding tilt relative crowding.
3. Cohorts are allocated to transparent leverage buckets: 5x/10x/20x/50x.
4. Approximate liquidation prices include configurable maintenance margin and
   fee buffer.
5. OI decreases, time decay, crossed levels and observed liquidations reduce
   cohort weight.
6. Every source is aligned causally; mark candles become available only after
   the bar closes.

## Limitations

- Account margin mode, collateral, exact leverage and entry prices are unknown.
- OI measures aggregate contracts; directional allocation is probabilistic.
- Public liquidation history may be sparse or limited.
- Heat zones are context, not entry signals.
