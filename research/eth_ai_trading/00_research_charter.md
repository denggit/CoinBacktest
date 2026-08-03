# ETH AI Trading Research Charter

## 1. Final objective

Build one ETH-USDT-SWAP AI trading system that can survive realistic market-order costs, latency, walk-forward validation, sealed holdout testing, and later AetherEdge shadow/live execution.

The target is not a collection of disconnected model demos. Every retained result must eventually feed one unified ETH target-position decision and risk contract.

## 2. Project boundaries

- **CoinBacktest** owns data audit, dataset construction, labels, feature research, model training, backtesting, stress testing, portfolio research, model packaging, and golden replay vectors.
- **AetherEdge** owns real-time market data, incremental online features, model loading/inference, deterministic risk controls, order management, state recovery, and live execution.
- CoinBacktest must not contain real account or order execution logic.
- AetherEdge must not contain model training, experiment mining, or research backtests.
- AetherEdge must never import CoinBacktest research scripts directly.

## 3. Frozen initial assumptions

- Symbol: `ETH-USDT-SWAP`
- Raw source: OKX historical trades
- First input representation: causal `1s` trade bars
- First decision cadence: every `5s`
- Initial execution: market orders
- Default full round-trip fee: `0.11%`, before extra slippage
- Warmup: `2022-01-01`
- Research: `2023-01-01` to `2026-06-30`
- Sealed holdout starts: `2026-01-01`
- Latency stress: `200ms / 500ms / 1s / 2s`
- Cost stress: `1x / 2x / 3x`

These assumptions can only change through an explicit plan version and must not be changed after observing a weak result merely to rescue it.

## 4. Research rules

1. One gated stage at a time.
2. Every feature must have an explicit availability time.
3. No random train/test split for time-series model selection.
4. The sealed holdout cannot be used to tune features, labels, thresholds, or hyperparameters.
5. Prediction metrics are secondary; complete net trading results are primary.
6. A later complex model cannot be used to hide failure of a simpler prerequisite.
7. A failed direction must be stopped, redesigned from a mechanism hypothesis, or rejected explicitly.
8. Long-running work must be chunked, memory bounded, resumable, and visibly progress-reported.
9. Existing Trade/OI/Books/Range/Footprint work is treated as candidate incremental evidence, not automatically retained edge.
10. Models never override hard position, loss, exposure, or kill-switch limits.

## 5. AI methods inside one system

The eight AI uses are integrated rather than built as isolated bots:

- AI-assisted research: all stages.
- Supervised prediction: R02-R08.
- Market-state recognition: R05 and later.
- Signal scoring/fusion: R04, R06, R08-R09.
- Deep sequence learning: R07 challenger only.
- Reinforcement learning: R11 constrained exit/holding overlay only.
- Execution optimisation: R10 and AetherEdge deployment.
- Risk/portfolio management: R06, R09, R12-R14.

## 6. Authoritative plan

The detailed stage plan is `docs/ETH_AI_TRADING_RESEARCH_PLAN.md`. The stage IDs in that document are tested against `src.ai_research.plan.DEFAULT_RESEARCH_PLAN`; code and documentation are not allowed to drift silently.
