# CoinBacktest Edge Library Runbook

CoinBacktest is now split into two kinds of code:

1. Reusable logic under `src/`.
2. Entry scripts under `research/`, `backtest/`, and `tools/`.

For single-market ETH perpetual research, the project should manage **edges**
instead of a heavy cross-sectional alpha library. An edge is a verified market
phenomenon, such as low-sweep reversal, order-flow absorption, momentum retest,
funding squeeze, or volatility expansion.

## Lifecycle

```text
Idea -> Research -> Edge -> Backtest Candidate -> Portfolio -> AetherEdge
```

Each stage can stop:

- No research edge: mark `rejected`.
- Edge exists but cannot survive costs: mark `backtest_failed`.
- Strategy has edge but does not improve portfolio: mark `portfolio_rejected`.
- Portfolio improves and parity is possible: mark `promoted`.

## Registry Files

```text
edge_library/registry.json
experiments/registry.json
```

Use the registries as metadata, not execution code. They prevent repeated work
and make old failures searchable.

## Commands

List edges:

```bash
python tools/list_edges.py
```

Register an edge:

```bash
python tools/register_edge.py --id ETH_EDGE_EXAMPLE --name "Example Edge" --family example --status researching --data-required 1m_trade_bar range_footprint
```

List experiments:

```bash
python tools/list_experiments.py
```

Register an experiment:

```bash
python tools/register_experiment.py --id ETH_EXAMPLE_RESEARCH --title "Example Research" --stage research --status researching --family example --hypothesis "Example hypothesis"
```

Write a standard decision artifact:

```bash
python tools/write_experiment_decision.py --experiment-id ETH_EXAMPLE_RESEARCH --stage research --status rejected --reason "No stable edge after costs" --next-action do_not_continue --out-dir data/reports/experiments/ETH_EXAMPLE_RESEARCH/research
```

Check import boundaries:

```bash
python tools/check_import_boundaries.py
```

## Import Rules

New code must follow these rules:

- `src/` may import only stable library code, never `research/`, `backtest/`, or `tools/`.
- `research/` scripts may import `src/`, but not other `research/` or `backtest/` scripts.
- `backtest/` scripts may import `src/`, but not `research/` scripts.
- `portfolio` should consume standard artifacts and manifests, not child backtest internals.

Known legacy coupling is frozen in:

```text
config/import_boundary_legacy_allowlist.json
```

The allowlist should shrink over time. Do not add new entries unless you are
explicitly documenting a temporary migration exception.
