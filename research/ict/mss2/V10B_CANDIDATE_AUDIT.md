# Repository Candidate Audit — LF V10B

Date: 2026-08-17

## Assessment: needs revision; not an independently proved sleeve

The saved artifact at
`data/reports/lf/eth_lf_portfolio_v10b_all_swing_structural_stop/turbo`
contains a real and unusually large historical result: 105 trades, saved dollar
P&L PF 8.29, 17,231.75% compounded return, and 21.13% realized-capital MDD from
2023-01-01 through 2026-06-15. The current source lives in
`src/sleeve_lib/lf_v10b` and combines three 4H engines, add-ons, dynamic risk,
microstructure filters, and a 21-bar structural stop.

That headline is not evidence of an untouched strategy edge.

## Provenance and selection integrity

Git history identifies the original official runner as
`backtest/lf/eth_lf_portfolio_v10b_all_swing_structural_stop_backtest.py` and
the selection work as `research/v10a_structural_stop_grid_research.py`,
`research/v10a_structural_stop_anti_overfit_verification.py`,
`research/v10b_final_verification.py`, and
`research/v10b_structural_neighbourhood.py`. Those files were later removed in
repository cleanup; the reusable implementation remains.

The original grid used the full 2023-01-01 through 2026-06-15 window and tested
multiple:

- stop sources and engine scopes;
- lookbacks 3/5/8/13/21;
- buffers 0/0.10/0.25/0.50 ATR;
- MFE triggers 0/0.5/1.0/1.5R;
- minimum holds and initial-stop variants;
- follow-up shortlist and neighborhood comparisons.

The same window then supplied the published yearly, winner-dependence, and
verification results. There is no discovery/validation split and no untouched
holdout. The existing report therefore reflects full-window model selection,
even though its stop timing is causally implemented.

## Causality and execution review

- 4H signals execute at the next 4H open.
- Daily and weekly regime features are shifted to completed higher-timeframe
  bars.
- Donchian entry/exit channels use shifted rolling windows.
- The active stop is snapshotted before current-bar touch checks; a structure
  level calculated from the completed current bar affects only later bars.
- Range/footprint context is loaded through `src.data_feed` loaders, and 4H
  OHLCV reaches `OKXDataLoader` through `src.backtest_common.data`.
- Base execution charges 0.055% fee per side plus 0.020% slippage per side.

No direct future-price leak was found in the inspected current path. This does
not cure full-window selection bias.

## Independent reproduction and cost stress

`19_audit_v10b_candidate.py` rebuilds the current features once and reruns the
exact executor at 1×/2×/3× fee and slippage. The path remains 105 trades. The
saved and current artifacts do not reproduce exactly: the saved return is
17,231.75% and PF 8.29, while the current rerun gives 19,152.08% and PF 9.05.
The current range data block 45 Momentum Long and 60 Momentum Short signals,
versus 44 and 55 in the saved summary. MDD still matches to rounding. A frozen
candidate needs a source/data hash and exact parity baseline before further use.

| KPI | Saved 1× | Current 1× | Current 2× | Current 3× |
| --- | ---: | ---: | ---: | ---: |
| Trades/month | 2.53 | 2.53 | 2.53 | 2.53 |
| Dollar P&L PF | 8.29 | 9.05 | 8.42 | 7.87 |
| Trade-return PF | 7.36 | 7.39 | 6.69 | 6.31 |
| Compounded return | 17,231.75% | 19,152.08% | 15,327.31% | 13,650.61% |
| Realized-capital MDD | 21.13% | 21.13% | 25.20% | 29.07% |
| Positive months | 45.24% | 45.24% | 42.86% | 40.48% |
| Longest flat | 62.0d | 62.0d | 62.0d | 62.0d |
| Top-5 removed return PF | 2.16 | 2.15 | 1.90 | 1.67 |
| Top-10 removed return PF | 0.78 | 0.80 | 0.69 | 0.59 |

The top-five test survives, but top-ten removal makes compounded return
negative (saved -29.66%; current -28.99%). This is material winner dependence,
not a cosmetic caveat. The 2×/3× MDD also exceeds the master's 20% hard ceiling.

## Metric-definition reconciliation

- The saved summary's PF 8.29 is dollar-P&L PF.
- The human-readable report's PF 7.36 is trade-return PF. Both are internally
  meaningful, but they are different units and must not be compared as one KPI.
- Engine-table `return_pct_sum` is the additive sum of per-trade account returns,
  not the 17,231.75% compounded account return.
- The equity file records realized capital only. Its MDD, rolling-90d rate, and
  227.7-day longest underwater period are not daily mark-to-market measures.
- 2026 is a partial year, and all annual results are in-sample relative to the
  historical model-selection process.

## Decision

V10B is a promising contaminated prior, not a promoted MSS2 sleeve. It fails
frequency, longest-flat, positive-month, MDD, top-ten resilience, exact parity,
and independent-holdout requirements. Its mechanisms may justify one frozen
visible-window falsification or future forward incubation, but its historical
headline must not be used to select R20 parameters or to open the MSS2 holdout.

