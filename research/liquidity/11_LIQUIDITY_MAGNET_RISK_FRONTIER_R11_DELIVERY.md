# 11 Liquidity Magnet and Risk Frontier R11

## Goal

R11 tests whether a causally active lower Swing-Low liquidity pool acts as a
tradable price magnet before it is swept.

This is not a post-sweep reversal study. It asks:

1. When price first approaches an active lower pool from 150/100/50/25bp away,
   does the lower target get hit before a causal risk barrier above entry?
2. Which stop style is least harmful?
3. Does a directional magnet effect survive realistic fees, slippage, and
   three-period stability checks?

## Frozen execution

- Signal: first closed 1m bar that enters a predeclared distance band while the
  pool is still active and unswept.
- Entry: next 1m open.
- Target: 5bp before the upper edge of the active lower pool.
- Stops:
  - equal-distance stop;
  - completed prior 15m high + 5bp;
  - completed prior 60m high + 5bp.
- Horizon: 180 minutes.
- Same bar target + stop: conservative stop.
- Costs:
  - fee 0.055% per side;
  - slippage 0.01% per side;
  - 2x cost stress.

No distance grid, stop optimization, or family-combination mining is allowed.

## Commands

Smoke:

```bat
python research\liquidity\11_liquidity_magnet_risk_frontier_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --max-candidates 1000 --skip-review-pack
```

Full:

```bat
python research\liquidity\11_liquidity_magnet_risk_frontier_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59"
```

If cache is missing:

```bat
python research\liquidity\11_liquidity_magnet_risk_frontier_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --rebuild-r02-if-missing --rebuild-r09-if-missing
```

## Output

```text
data/reports/research/liquidity/11_liquidity_magnet_risk_frontier_r11
```

Important files:

- `04_risk_frontier_summary.csv`
- `05_equal_distance_directional_magnet.csv`
- `06_timeframe_confluence_summary.csv`
- `08_period_stability.csv`
- `09_candidate_scorecard.csv`
- `10_causal_audit.csv`
- `15_research_brief.md`
- `gpt_review_pack.zip`

## Interpretation

The equal-distance stop is the cleanest magnet test. A target-before-stop rate
above 50% supports a directional tendency, but it is not enough for a strategy.
The route must remain positive after costs and across all three periods.

If no frozen specification survives, do not add more distance bands or tune
stop buffers. The liquidity atlas may still be useful for target mapping, but
the pre-sweep short route is not independently tradable.


## Fix1 (v1.0.1)

- Discards candidates when the strict next 1m bar is missing instead of entering on a later observed bar.
- Keeps the strict next-open causal gate; it is not relaxed.
- Adds explicit event-position and available-time audit checks plus a data-gap regression test.
