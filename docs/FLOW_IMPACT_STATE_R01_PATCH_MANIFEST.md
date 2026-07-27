# Flow–Impact State Round 01 Patch Manifest

## Scope

This patch adds the first long-history research round for the OKX aggressive-flow / price-response strategy family.

It deliberately does **not** implement a tradable strategy, TP/SL grid, Liquidity filter, 4H hard gate, machine-learning selector, AetherEdge plugin, or live execution path.

## Added files

```text
src/research_common/flow_impact.py
research/mhf/flow_impact_state/__init__.py
research/mhf/flow_impact_state/00_research_log.md
research/mhf/flow_impact_state/01_pressure_event_atlas.py
docs/FLOW_IMPACT_STATE_R01_RUNBOOK.md
docs/FLOW_IMPACT_STATE_R01_PATCH_MANIFEST.md
tests/research_common/test_flow_impact.py
```

## Design

- Uses `OKXTradeBarLoader` in cache-only mode; no direct SQLite access and no automatic download/build.
- Requires rich OKX trade-bar fields and refuses to infer aggressor direction from OHLCV.
- Defines events from abnormal rolling signed notional, not from a price impulse or a single candlestick.
- Builds 1/3/5-bar causal pressure windows and a historical baseline ending before the full current pressure window.
- Detects threshold onset or pressure-direction flips, then clusters overlapping 1/3/5-bar detections into one underlying pressure process.
- Uses the immediate next bar open as the executable path origin.
- Excludes any event whose feature window, entry row, or complete forward path touches a synthetic gap row.
- Studies continuation and reversal symmetrically across 30s/2m/5m/15m/30m when the selected base timeframe can represent them exactly.
- Reports MFE, MAE, conservative first-touch labels, post-event flow, pressure-state duration, yearly/monthly stability, causal audit, and deterministic samples.
- Applies 0.11% fee-only round-trip cost and a default 0.15% conservative normal execution cost.
- Calibrates the broad pressure threshold by event frequency only. Forward returns are forbidden for selecting the Round 01 event threshold.

## Default command

```bat
python research\mhf\flow_impact_state\01_pressure_event_atlas.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30 --min-pressure-z 1.5
```

## Validation completed

```text
python -m py_compile src/research_common/flow_impact.py research/mhf/flow_impact_state/01_pressure_event_atlas.py tests/research_common/test_flow_impact.py
PYTHONPATH=. pytest tests/research_common -q
PYTHONPATH=. python research/mhf/flow_impact_state/01_pressure_event_atlas.py --self-test
```

Results:

```text
6 passed
self-test PASS
synthetic valid events: 265
synthetic excluded events: 0
```

## Baseline limitations observed in the supplied archive

The supplied repository archive already contains unrelated modified/deleted files and does not collect the complete test suite because several pre-existing research modules referenced by tests are absent. The patch does not modify or repair those unrelated paths.

The archive also does not contain the user's local `data/okx_trade_bars.db`, so the full 2023-01-01 through 2026-06-30 production atlas could not be executed in this environment. The deterministic self-test and focused research-common tests were executed instead.

## Git

No commit was created.
