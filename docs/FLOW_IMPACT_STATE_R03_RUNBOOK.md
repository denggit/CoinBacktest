# Flow-Impact R03 Runbook

## Purpose

Test whether accumulated OKX taker pressure becomes tradable only after causal Price Action confirms continuation or exhaustion.

## Run

```bat
python research\mhf\flow_impact_state\03_accumulated_pressure_pa.py --symbol ETH-USDT-SWAP --timeframe 1m --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date 2026-06-30
```

The study is cache-only and reads `data/okx_trade_bars.db` through `OKXTradeBarLoader`.

## Processes

### Exhaustion reversal

```text
5/10/20m accumulated taker pressure
-> sweeps a causally known swing level
-> marginal impact decays
-> later closed bar reclaims the level and resumes oppositely
-> next open entry
```

Stop: attack extreme plus buffer.
Target: nearest causally known opposite structure.

### Continuation

```text
5/10/20m accumulated taker pressure
-> breaks a causally known swing level
-> retest holds outside the old range
-> later closed bar resumes with pressure
-> next open entry
```

Stop: retest invalidation plus buffer.
Target: nearest causally known structure, otherwise a measured move from the broken range.

## Hard gates

- Conflict-resolved trades >=1,000.
- Discovery >=500, validation >=200, holdout >=200.
- Net expectancy positive in all three splits.
- Full net PF >=1.20.
- Positive-month ratio >=65%.
- At least three positive years.
- Safety-timeout share <=10%.
- Top five winners <=20% of gross winning return.

## Important interpretation

The 240-bar timeout is not a strategy take-profit or tuned exit. It is an operational censor/fallback and is separately reported. A candidate with material timeout reliance fails.
