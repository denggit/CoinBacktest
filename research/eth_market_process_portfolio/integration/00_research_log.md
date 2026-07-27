# Integration Research Log

## R02 — Environment-conditioned strategy lab

Purpose: move from standalone order-flow event averages to complete, causally
tradable market processes.

Frozen strategy families:

1. `compression_breakout`: a completed compression environment followed by a
   directional breakout with effective aligned order flow;
2. `expansion_exhaustion`: a completed directional expansion followed by a
   sweep, absorption and reversal trigger;
3. `balance_failed_auction`: a balanced auction followed by a failed edge
   excursion and order-flow reversal.

R02 is a strategy replay rather than a fixed-horizon event study. Each family
has a structural stop, a mechanism target, causal trailing rules and a maximum
holding period used only as a fail-safe. Stop is assumed first when stop and
target are both touched in one 1m bar.

The primary definition is `base`. `loose` and `strict` are frozen coherent
neighbourhood checks; they are not selected by best result. Stress scenarios are
base, 2x fees, 1m delay, 3m delay and 2 bps adverse slippage per side.

Run:

```bash
python research/eth_market_process_portfolio/integration/01_environment_conditioned_strategy_lab.py
```

Output:

```text
data/reports/research/eth_market_process_portfolio/integration/01_environment_conditioned_strategy_lab/
```
