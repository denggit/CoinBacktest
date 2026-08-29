# R23 — Frozen Panic-Wick Structural Long Falsification

Date: 2026-08-17

## Overall assessment: rejected; archive shadow-wick shortcut

R23 audited the strongest executable prior found in the historical shadow-wick
branch: the exact `priority_union` Long entry, `multi_sweep_deeper_higher_low_trail`
exit, and historical entry delay two. It did not test the branch's attractive
Asia session, flow bucket, wick threshold, fixed target, or other exits.

This is a contaminated-prior falsification. The source branch selected on the
full 2023–2026 window after at least three entry policies, nine V1 exit modes,
three delays (81 visible combinations), many earlier feature exclusions, and a
later seven-mode exit ladder. Its reported PF 1.58 was never an untouched split.

## Data quality and causal design

- `src.data_feed.OKXTradeBarLoader` supplies 1,837,343 of 1,838,880 requested
  minutes through 2025H1.
- There are 1,537 absent minutes in 72 gap runs; the largest gap is 10h12m.
- The calendar is regularized with flat, zero-volume placeholders only to
  preserve time. Events require 240 consecutive source-observed minutes;
  synthetic bars cannot signal, enter, or resolve a trade.
- The frozen rule produces 713 eligible visible events and 230 non-overlapping
  closed trades. No emitted trade crosses a data gap and no boundary censor is
  present.
- Twelve signal, delay, next-open, split, uniqueness, and cost checks pass.
- An independent state-machine replay checks all 230 trades against source
  bars. Entry observation/open, first exit reason, decision time, next-open
  time/price, path-gap absence, and gross return all match exactly.

## Result

| Split | Trades | Trades/month | 1× PF | 2× PF | Mean 2× | Top-10-removed 2× sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Discovery 2023–2024 | 119 | 4.96 | 2.45 | 1.67 | +0.163% | -4.52% |
| Validation 2025H1 | 111 | 18.50 | 1.42 | 0.96 | -0.013% | -18.61% |

The legacy rule contains a real base-cost tendency, but it does not meet the
master's stressed-cost requirement. At 2× cost, 2023 PF is 1.24 on only 22
trades, 2024 PF is 1.76 on 97 trades, and 2025H1 PF is 0.96 on 111 trades.
Validation compounded return is -1.76% with -11.83% MDD before any risk sizing.

Discovery winner concentration is also binding. Removing five winners leaves
PF 1.15 and +4.37%, but removing ten changes PF to 0.84 and sum to -4.52%.
Validation top-five and top-ten removal are both strongly negative.

The validation frequency surge from 4.96 to 18.50 trades/month is not a benefit:
the same rule fires much more often while expectancy disappears. That is
consistent with state/calibration drift and the historical branch's own weak
2025 fixed-horizon statistics.

## Frozen conclusions

1. Reject the frozen panic-wick structural Long as an MSS2 sleeve.
2. Do not rescue it with Asia session, prior-move, delta/taker, volatility,
   wick size, fixed TP, exit ladder, delay, or threshold selection.
3. The base-cost effect is too thin for 2× costs and is not top-ten resilient.
4. The historical full-window headline materially overstates forward evidence.
5. No portfolio construction or holdout opening is justified.

## Next boundary

The simple-price, trade-flow, daily trend, BTC lead/lag, OI-transition, V10B,
and panic-wick shortcuts are now exhausted. New research should only proceed
from a genuinely different data-generating mechanism with full pre-holdout
coverage. Spot/perpetual basis is unavailable locally; books, funding, mark,
and liquidation histories begin inside already-sealed or late windows and
cannot support R24 discovery/validation.

