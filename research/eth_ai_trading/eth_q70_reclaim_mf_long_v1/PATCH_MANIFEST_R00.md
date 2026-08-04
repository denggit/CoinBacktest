# Patch Manifest — ETH AI Trading R00

## Scope

Framework and governance only. No raw data build, feature implementation, label generation, model training, strategy backtest, or AetherEdge modification is included.

## Added

- `src/ai_research/`: frozen research configuration, typed stage contracts, canonical stage DAG, validation, and artifact writers.
- `research/eth_ai_trading/`: charter, README, and framework initializer.
- `docs/ETH_AI_TRADING_RESEARCH_PLAN.md`: authoritative staged research-to-live plan.
- `tests/ai_research/`: plan validation, documentation parity, status preservation, and initializer artifact tests.

## Run

```bash
python research/eth_ai_trading/00_initialize_framework.py
```

## Next gate

R01 — Raw trades and 1s trade-bar audit.
