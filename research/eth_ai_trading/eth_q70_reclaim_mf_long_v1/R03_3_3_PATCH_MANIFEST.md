# R03.3.3 Patch Manifest

## Purpose

Add a causal multi-timescale market-state continuity and transition research stage.

The stage is auxiliary context only. It does not open, close or size positions.

## Data routing

- 2020-01-01 through 2021-12-31: `src.data_feed.okx_loader.OKXDataLoader`
- 2022-01-01 through 2025-12-31: `src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader`
- Universal branch: common OHLCV-derived features only
- Trade-enhanced branch: real Trade Bar features only where available
- 2026H1 remains sealed

## New command

```bat
python research\eth_ai_trading\03_3_3_market_state_continuity.py
```

## New cache

```text
data/cache/eth_ai_trading/r03_3_3_universal_state
```

## New report

```text
data/reports/research/eth_ai_trading/03_3_3_market_state_continuity
```

## Main outputs

- state duration and flip atlas
- persistence/transition target distribution
- state combination to future 6h opportunity linkage
- 2024/2025 OOS continuity model metrics
- raw-vs-hysteretic stability comparison
- training-year attribution matrix
- Trade-feature increment comparison
- feature importance and prediction samples

## Validation

- Python compilation
- all AI Research tests
- Data Feed regression tests
- command entrypoint
- synthetic 421-day multi-timeframe state build
