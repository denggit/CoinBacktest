# R21 — Canonical Daily Channel Trend Following

Date: 2026-08-17

## Overall assessment: rejected; no daily-channel sleeve

R21 tested a genuinely different horizon after the R20 V10B closeout. The
primary model was a canonical daily 20-day breakout with a 10-day channel exit;
the single sensitivity was 55/20. Both used Wilder ATR(20), a fixed 2×ATR
initial stop, next-calendar-day 00:00 market execution, and no leverage,
compounding, add-on, filter, target, or time-profit exit.

The study loaded bare OKX `ETH-USDT-SWAP` 1m bars only through
`src.data_feed`. Discovery (2023–2024) and validation (2025H1) were reset and
simulated separately for Long and Short. July 2025 is embargoed and the
2025-08-01 holdout remains sealed.

## Data quality and causal design

- Requested and observed coverage are both 1,838,880 consecutive 1m rows from
  2022-01-01 through 2025-06-30 23:59, with no internal gap.
- Daily bars are left-labelled UTC/project-time calendar days. A completed
  daily signal executes only at the next day's first 1m open.
- Entry and exit channels use rolling highs/lows shifted by one complete day.
- The fixed initial stop is computed from the signal day's completed ATR(20).
- Stop exits use the first exact 1m touch; a later-day gap through the stop uses
  the worse 1m open. A pre-existing channel exit executes at the next open
  before that day's stop path.
- A position still open at a split boundary is censored. The final run has 41
  closed paths and one boundary-censored path.
- Eleven simulator causal/cost checks pass with zero violations.
- An independent validator reconstructed daily channels and ATR directly from
  raw 1m bars without importing the R21 simulator. Across all 42 emitted paths,
  all eight entry, price, stop, earliest-exit, outcome, and boundary checks pass.

## Result

| Model / direction | Discovery trades | Discovery 2× PF | Validation trades | Validation 2× PF | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| D20_X10 Long | 13 | 1.76 | 1 | inf | fail validation sample and top-five resilience |
| D20_X10 Short | 10 | 0.34 | 4 | 0.64 | fail economics |
| D55_X20 Long | 8 | 2.83 | 1 | 0.00 | sensitivity fails validation; discovery concentrated |
| D55_X20 Short | 3 | 0.16 | 1 | inf | sparse and no discovery edge |

The primary Long headline is not stable evidence. Its 13 discovery trades
average only 0.54 per month, and removing the top five winners changes the 2×
sum from positive to -54.96%. The yearly split is also adverse: D20 Long loses
18.85% in 2023 at 2× costs and makes +60.73% in 2024. Its only validation
trade wins, which is insufficient for the precommitted two-trade minimum and
cannot establish forward stability.

The canonical 55/20 Long sensitivity exhibits the same winner-concentrated
shape: five 2023 trades lose -13.73%, three 2024 trades make +71.72%, and the
single 2025H1 trade loses -5.73%. Both Short variants lack discovery edge.

Coverage is far below the portfolio objective. D20 discovery Long has a
125-day longest entry gap and 107-day longest flat interval; D55 discovery Long
has a 198-day entry gap and 149-day longest flat interval. Positive-month rates
are 20.8% and 12.5%, respectively, including zero-trade months.

## Frozen conclusions

1. Reject the exact D20/X10 and D55/X20 daily-channel sleeves.
2. Do not rescue them with nearby channel grids, trend/volatility filters,
   trailing stops, pyramiding, leverage, sizing, or holdout inspection.
3. The visible Long profit is a small-number convex-winner pattern concentrated
   in 2024, not a stable repeatable trade edge.
4. Daily trend following cannot satisfy the master frequency or flat-duration
   requirements as a standalone sleeve even if its sparse headline survived.
5. No R21 result is eligible for portfolio construction or forward incubation.
6. July and holdout outcomes remain untouched.

## Next boundary

R22 must change mechanism rather than tune the daily channel. The next audit
should prioritize a higher-frequency, economically grounded state not already
closed by intraday breakout, completed-trend acceptance/reversal, OI transition,
V10B, or daily trend research. Candidate selection must be based on repository
coverage and causal source availability before outcomes are run.
