# R24 Precommitment — Scheduled Funding-Window Unwind

Date frozen: 2026-08-17, before R24 outcomes are calculated.

## Economic hypothesis

OKX perpetual funding is normally exchanged on the fixed eight-hour clock.
Positions established into a scheduled settlement can be reduced after the
transfer, creating a short-lived reversal after unusually large pre-settlement
price moves. Historical funding-rate values are unavailable before June 2026,
so R24 tests only the known canonical settlement clock and price path; it does
not infer or fabricate funding sign.

This is a scheduled positioning-flow hypothesis, distinct from generic impulse
continuation/exhaustion, BTC lead/lag, OI transitions, sweeps, or wick filters.

## Frozen signal and execution

- Use complete left-labelled ETH 1h bars from bare OKX 1m data.
- A bar `[t-1h, t)` becomes available at scheduled local/project time `t`.
- Scheduled times are 00:00, 08:00, and 16:00. These are invariant under the
  project's fixed +8 conversion of the canonical UTC 00/08/16 clock.
- Standardize the completed pre-settlement 1h return by the prior 720 complete
  hourly-return standard deviation, excluding the signal hour.
- Event threshold: absolute z-score >=1.5.
- Trade direction is opposite the pre-settlement return.
- Entry is the ETH 1m open exactly at scheduled time `t`.
- Initial risk is 1.5× the completed signal-hour Wilder ATR(20).
- Primary target is 1R; 2R is one diagnostic sensitivity and cannot rescue R1.
- Exact 1m first passage is stop-first on same-minute ambiguity. Stop gaps use
  the worse open; target fills at its frozen price.
- If neither barrier is touched, exit at the next scheduled eight-hour boundary
  open. A research split boundary censors first.

Long and Short are separate one-position sleeves. There are no session subsets,
funding sign/rate filters, trend/volatility filters, alternate z thresholds,
targets, stops, delays, or hold periods.

## Time, cost, and gate

- Warmup: 2022 onward.
- Discovery: 2023–2024 reset simulation.
- Validation: 2025H1 reset simulation.
- July embargoed; holdout begins 2025-08-01 and remains unloaded.
- Round-trip costs: 0.11%/0.22%/0.33% at 1×/2×/3×.

An R1 direction is only a research candidate if discovery/validation 2× PF are
both >=1.4 with positive expectancy, discovery/validation have at least 50/10
closed trades, every visible year has positive 2× sum, discovery top-ten
removal stays positive, timeout share is <=20% in both splits, and R2 discovery
expectancy is positive. Passing is not live approval.

