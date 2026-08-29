# Decision Log

1. Do not continue abstract edge/ML prediction work unless it directly supports an already-defined tradable strategy.
2. First validate publicly specified strategies on ETH under one common protocol.
3. Freeze all V1 rules before inspecting ETH tournament outcomes.
4. Do not tune failed strategies against 2023-2025 losses. A failed family is archived and replaced with a new externally specified strategy batch.
5. Keep 2026 sealed until the survivor set and portfolio construction are frozen.
6. Portfolio V1 uses equal sleeves; no optimizer is allowed to fit weights to the backtest.

7. V1 result changes the active research object: do not spend the next stage looking for better isolated entries. Test continuous position management using the simple trend families that actually survived.
8. R02 exchange execution is single net ETH exposure. Internal sleeves may be long and short simultaneously, but the portfolio aggregator nets them before execution.
9. R02 risk parameters and signal lookbacks are frozen before the user's full-data run. Do not tune them after seeing R02 losses.
10. 2026 stays sealed until one continuous portfolio specification is frozen from 2023-2025 evidence.
11. R02 is rejected as a live portfolio. Do not tune its DD governor, deadband, family weights, or volatility target against the observed losses.
12. Before any new portfolio construction, replicate complete externally specified trend systems. Every rule must be tagged as source core, required ETH adaptation, or our experiment.
13. R03 contains no portfolio optimizer and no invented drawdown overlay. 2026 stays sealed.
14. Turtle V1 was not a fully source-faithful sizing replication; R03 corrects the core Unit definition to 1N ≈ 1% equity. Historical Turtle discretionary annual notional-account resets cannot be mechanically reproduced and are explicitly excluded/adapted rather than guessed.
15. If R03 produces fewer than two robust source survivors, do not force a portfolio and do not tune the failed baselines. Search for additional fully specified public systems.

## R04 decision

Do not tune Turtle entry/exit parameters. Study the path of the existing profitable R03 Turtle first. Any future position overlay must be a simple causal rule motivated by a path pattern that appears in both 2023-2024 discovery and 2025 validation. Do not tune the overlay on 2025 and do not open 2026.

16. R04.1 fixes only the `BacktestResult.minute_equity` integration contract. Do not interpret the first local crash as a strategy/path result, and do not change Turtle logic in response to this engineering bug.
