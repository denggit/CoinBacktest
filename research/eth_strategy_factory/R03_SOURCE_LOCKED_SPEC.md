# R03 — Source-Locked ETH Trend Replication

## Objective

Stop inventing portfolio heuristics before the public baselines are replicated. R03 tests only strategies whose core rules can be traced to a specific original paper/rule set, then makes only the adaptations required to run them on a single ETH-USDT-SWAP perpetual with CoinBacktest's causal execution contract.

This stage does **not** optimize portfolio weights, add drawdown governors, tune lookbacks after seeing ETH results, or use 2026.

## Replications

### SL01 — Zarattini/Pagani/Barbon long-only crypto Donchian

Source core:
- close-channel lookbacks: 5/10/20/30/60/90/150/250/360 days
- long on upper Donchian close-channel breakout
- monotone channel-midpoint trailing exit
- 90-day annualized realized volatility
- 25% volatility target, 2x cap
- equal-weight nine submodels
- 20% threshold only for volatility-driven rebalancing; signal changes execute immediately

Required adaptation:
- one ETH-USDT-SWAP rather than BTC / a rotating crypto universe
- +8 daily boundary
- CoinBacktest 0.11% round-trip cost
- causal execution at the first 1m open strictly after the source signal becomes observable
- ambiguous wording "difference ... exceeds 20%" is frozen as 0.20 absolute portfolio-weight points and disclosed, not tuned

### SL02 — Zarattini/Pagani/Barbon long-short appendix

Same core implementation, with symmetric short lower-channel entry and direction-aware monotone midpoint trailing stop. ETH perpetual shorting is directly tradable; financing/funding is not introduced into this baseline because the project baseline cost contract is the frozen transaction-cost model.

### SL03 — Moskowitz/Ooi/Pedersen 12-month TSMOM

Source core:
- sign of past 12-month return determines long/short
- one-month holding/rebalance
- position weight = 40% / ex-ante annualized volatility

Required adaptation:
- single ETH instead of the paper's diversified 58-market portfolio
- past 12-month raw ETH-perpetual return is the tradable proxy for the paper's contract excess return
- 12 calendar months mapped to 365 crypto daily bars
- 60-day EWMA volatility is an explicitly disclosed standard replication convention, not claimed to be the paper's only possible estimator

### SL04 — Original Turtle System 2 core

Source core:
- intraday break of preceding 55-day high/low
- every System-2 breakout taken
- N = 20-day EMA of True Range, initialized by 20-day average
- 1N price move corresponds to 1% account-equity Unit
- 2N hard stop
- add one Unit every 0.5N from actual previous fill
- max 4 Units in one market
- System-2 exit = opposite 20-day breakout

Required adaptation:
- ETH perpetual has no contract rollover or historical futures contract point-value conversion
- one ETH has $1 PnL per $1 price move
- +8 daily context and 1m intraday execution
- current live equity is used for Unit sizing. The historical Turtle notional account was reset annually using Dennis's discretionary assessment, so an exact mechanical historical capital-accounting replication is impossible from the published rules. R03 therefore locks the mechanical entry/exit/N/Unit/pyramiding core and discloses the capital-accounting adaptation.

## Deliberately omitted

MA trend is not included in R03. Public crypto MA research often reports horizons selected by rolling/walk-forward performance; treating one ex-post winning pair as an immutable original rule would blur replication and parameter selection.

## Validation protocol

- warmup: 2022-01-01
- evaluation: 2023-01-01 through 2025-12-31
- 2026-01-01 onward: hard sealed
- costs: base 0.11% round trip, 2x and 3x stress
- execution delay: base causal execution plus +1m / +2m stress
- minute mark-to-market drawdown
- top 1/5/10 positive-day dependency
- no failed source baseline is rescued with parameter changes

If at least two source baselines survive, R04 may combine the **unchanged** survivors as independent sleeves. If fewer survive, add new complete public strategies instead of tuning these.
