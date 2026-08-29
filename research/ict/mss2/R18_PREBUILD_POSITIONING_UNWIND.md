# R18 Prebuild — Independent Positioning-Unwind Path Atlas

Date: 2026-08-16

This is a source and non-duplication boundary, not yet the frozen R18 event contract.

## Why positioning is the next layer

R13–R17 found no stable price-only entry across completed-trend reversal, acceptance continuation, fixed-R/stop rescue, or aligned trend-pullback re-acceleration. Adding another price threshold to those exact branches would be filter rescue. The master charter permits OI only after simple price paths stop separating outcomes; that condition is now met.

## Repository non-duplication audit

- Do not run failed-auction range re-entry. `eth_market_process_portfolio/integration/R02` already tested it: base PF 0.34, 2×-fee PF 0.12, every year negative, and loose/strict neighbors also failed.
- Do not rerun post-sweep OI enrichment as a new strategy. `research/liquidity/R05` already causally aligned Binance 5m metrics to 709,731 sweep checkpoints with 99.95% coverage and stored oracle turning points as future-only labels.
- Reuse public loader, relative-change, availability, and alignment utilities from `src.data_feed.binance_futures_metrics*` and `src.research_common.post_sweep_oi`; never import the R05 research script.

## Available source

- Execution market remains OKX `ETH-USDT-SWAP` 1m K through `src.data_feed`.
- Positioning proxy is Binance USD-M `ETHUSDT` official 5m metrics through `src.data_feed.binance_futures_metrics_loader`.
- Metrics expose OI base/USD, taker imbalance, and long/short ratios with explicit `available_time` and a pre-existing one-minute publication-lag convention.
- Existing R05 evidence shows near-complete 2023–2026 alignment. Binance OI must always be labeled a cross-exchange proxy, never OKX-local OI.
- Local OKX derivatives coverage is not sufficient for primary R18 research: OI is only daily (912 rows from 2024), funding has only June 2026, and liquidation history is absent.

## Proposed independent question

Across all market time—not conditioned on a liquidity sweep—does a causal price/OI state transition create a stable asymmetric path?

```text
position build: price and OI expand together over a completed window
→ position release: OI contracts while price either continues or stops responding
→ first causal price stabilization / reacquisition state
→ next-observable OKX 1m path
```

Long and Short must remain separate. R18 should first compare path classes and first passage, not optimize an entry. The economically distinct states are:

1. same-direction OI build continuing (new positions support the move);
2. price continuation with OI release (liquidation/covering-compatible move);
3. OI release with price stabilization (potential exhaustion/reversal);
4. OI rebuild after stabilization (potential continuation resumption).

## Required precommitment before outcomes

- Choose one small set of completed 5m/1h sign-transition definitions; no quantile or magnitude grid.
- Make publication availability explicit and execute closed signals at the next eligible 1m bar.
- Use broad discovery 2023–2024 and validation 2025H1; keep the existing 2025-08-01 holdout sealed.
- Define structural or volatility-based path barriers and maximum risk before economics.
- Keep future OI and oracle turning-point fields physically outside the causal feature table.
- Stop R18 immediately if the unfiltered transition lacks same-sign discovery/validation path economics; do not rescue it with taker ratio, top-trader ratio, funding, or ML.
