# Open Items and Roadmap

1. Run R01 locally against the frozen R00.4 cache and inspect the generated strategy/OOS report.
2. If `PASS_R01_STRATEGY_CANDIDATE`: harden the selected strategy with delay/slippage, top-trade removal, subperiod stability and structural/non-time exit challengers before any sealed holdout is opened.
3. If `PROMISING_BUT_NOT_R01_PASS`: diagnose the concrete failed strategy dimension (frequency, losing-day streak, MDD, cost robustness, or year stability) and change the strategy architecture rather than micro-tuning one loss.
4. If `NO_TRADABLE_STRATEGY_R01`: stop this tabular baseline and move to a sequence/state challenger only if it has a clear path to a different executable strategy.
5. 2026 remains sealed until the strategy family and all tuning decisions are frozen.
6. Only after a durable strategy exists should R02 offline-RL optimize dynamic sizing/hold/reduce/exit.
7. Final surviving policy is migrated to AetherEdge as a pluggable live strategy; CoinBacktest remains research/backtest only.
