# Order Flow Research Log

## R01 — Order-flow process event study

Purpose: screen four pre-declared causal market processes on complete tzplus8
ETH 1m trade bars before designing a trading strategy.

Processes:

1. aggressive-buy continuation;
2. aggressive-sell continuation;
3. sell-pressure absorption and long reversal;
4. buy-pressure absorption and short reversal.

R01 is deliberately not a broad parameter factory. It uses closed bars, enters
at the next bar open, deducts 0.11% round-trip cost, reports 5/15/30/60 minute
paths, and evaluates each calendar year separately.

Run:

```bash
python research/eth_market_process_portfolio/order_flow/01_order_flow_process_event_study.py
```

Output:

```text
data/reports/research/eth_market_process_portfolio/order_flow/01_order_flow_process_event_study/
```

### R01 decision

The completed run produced 61,527 events. After deducting 0.11% round-trip
cost, all four standalone order-flow processes had negative mean net return and
profit factor below 1 at every 5/15/30/60 minute horizon. The failure was also
present in each calendar-year split. R01 is therefore not promoted as a
standalone strategy and is retained only as evidence that order flow needs an
explicit market environment and complete trade lifecycle.

## R03 — Broad multi-window order-flow + single-PA path atlas

R02 demonstrated that building a finished strategy by AND-ing many environment,
structure and order-flow requirements at one timestamp can destroy the candidate
universe before the mechanism is understood. R03 therefore returns to the
widest practical path study.

Frozen method:

1. use every valid non-neutral pressure observation;
2. compare 1/3/5/10/15/30/60 minute pressure windows;
3. study level, strengthening, weakening and direction-reversal paths;
4. add only one PA context at a time: prior-trend aligned, prior-trend opposed,
   sweep/reclaim, or breakout acceptance;
5. report exact retention, return increment and calendar-year consistency;
6. do not add Range Bar, Books, volatility, large-trade or positioning filters.

Run:

```bash
python research/eth_market_process_portfolio/order_flow/02_broad_order_flow_pa_path_atlas.py
```

Output:

```text
data/reports/research/eth_market_process_portfolio/order_flow/02_broad_order_flow_pa_path_atlas/
```

R03 is still a hypothesis/path screen. A row is promoted to a real strategy
only after it retains adequate frequency, clears 0.11% cost with sufficient
margin across years, and passes de-overlapped next-open TP/SL replay.

## R04 — Sell-pressure shock: spike/reclaim versus breakdown continuation

R03 confirmed that aggressive-flow direction alone is insufficient and that
isolated acceptance/reclaim rows require a dedicated causal study. R04 tests one
specific mechanism without assuming the answer in advance:

> A sudden aggressive-sell shock can either continue lower or reverse. The
> observable price response—downside impulse, lower wick, prior-low sweep,
> reclaim, or acceptance below—may separate the two paths.

Frozen method:

1. study broad 1/3/5/10/15/30 minute sell-pressure shocks;
2. define shocks as sell-band entry, sell strengthening, or buy-to-sell reversal;
3. report both immediate short-follow and long-fade outcomes;
4. add one condition at a time: activity, downside impulse, aggregated lower
   wick, prior-low sweep, same-window reclaim, delayed reclaim, or multi-bar
   breakdown acceptance;
5. use a single-pass causal state machine for post-shock confirmation;
6. do not add trend, Range Bar, footprint, Books, OI or liquidation filters;
7. do not search TP/SL parameters in this stage.

Run:

```bash
python research/eth_market_process_portfolio/order_flow/03_sell_pressure_shock_path_study.py
```

Output:

```text
data/reports/research/eth_market_process_portfolio/order_flow/03_sell_pressure_shock_path_study/
```
