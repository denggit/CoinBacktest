# Patch Manifest — ETH Absorption / Market-Control Inventory Strategy R01

## Added

- `src/research_common/multiscale_absorption.py`
  - causal multi-scale pressure/defense/spring primitives carried forward from the prior atlas patch;
  - no future bars in feature construction.
- `src/research_common/absorption_inventory_strategy.py`
  - market-control state classification;
  - causal 5m/15m/1H/4H alignment;
  - fresh-event vote engine;
  - 5m reversal veto;
  - safe cross-net inventory simulator;
  - long-vote/short-vote sign invariants;
  - intrabar conservative liquidation check;
  - fee/slippage accounting.
- `research/eth_absorption_inventory_strategy/01_absorption_state_inventory_strategy.py`
  - executable strategy backtest;
  - default 2023-01-01 -> 2026-06-30 with 2022 warmup;
  - 1x/2x/3x execution-cost stress;
  - yearly/monthly/daily reports and exact votes/orders.
- `research/eth_absorption_inventory_strategy/README.md`
- `research/eth_absorption_inventory_strategy/00_research_log.md`
- `tests/research_common/test_absorption_inventory_strategy.py`
- `tests/research_common/test_multiscale_absorption.py`

## Frozen strategy semantics

- 15m / 1H / 4H can create inventory votes.
- 5m only confirms/vetoes reversal evidence.
- failed pressure / impact decay -> vote opposite aggressor.
- qualified repeated defense + spring/upthrust -> defending-side vote.
- efficient pressure -> vote with aggressor.
- one vote = 1% current-equity margin at 10x by default.
- no conventional TP/SL/time exit.
- neutral market = no inventory change.
- opposite future evidence mechanically reduces/reverses net inventory.
- a LONG vote can never execute negative notional; a SHORT vote can never execute positive notional.
- same-side votes above leverage cap are blocked/clipped, never converted into forced opposite trades.

## Causality

- HTF left-labelled bar becomes visible only at `bar_start + timeframe`.
- only persistent state may be carried forward.
- fresh event votes exist only at the exact causal `available_time`; they are never forward-filled.
- signal executes at the next 1m open.

## Tests run

- `python -m pytest tests/research_common/test_absorption_inventory_strategy.py tests/research_common/test_multiscale_absorption.py -q`
- `python -m pytest tests/research_common -q`
- `python -m compileall -q ...`

No git commit was executed.

## Full-suite note

`python -m pytest -q` was also attempted. Collection stopped on 5 pre-existing missing-module/file errors in the provided repository snapshot (`liquidity_touch_rebound_v1`, `panic_selloff_rejection_recovery_long`, `panic_low_excursion_rejection`). These files are not modified or supplied by this patch. The strategy/research-common test group itself is green.
