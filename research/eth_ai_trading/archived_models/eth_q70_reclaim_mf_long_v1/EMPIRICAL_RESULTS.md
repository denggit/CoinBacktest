# Empirical Results

All figures below retain their original metric scope. Fixed-six-hour signal diagnostics, single-position path diagnostics and risk-sized account results are not interchangeable.

## Development account — C2 equal-one-R

Anchor scenario: 1-minute delay, 2x cost.

| Period | Trades | Return | MDD | PF | Win rate | Positive months |
|---|---:|---:|---:|---:|---:|---:|
| 2024 | 236 | +85.1% | -9.4% | 1.71 | 43.6% | 10/12 |
| 2025 | 244 | +100.7% | -8.4% | 1.74 | 49.6% | 8/12 |
| Continuous 2024–2025 | 480 | +271.4% | -9.4% | 1.73 | 46.7% | 18/24 |

Additional continuous-account facts:

- CAGR: 92.8%.
- Positive quarters: 8/8.
- Return after removing ten largest winners: +86.1%.
- Longest losing streak: 9 trades.
- Longest drawdown period: approximately 74 days.
- Median holding: approximately 14.2 hours.

These results justified opening the untouched holdout; they did not constitute live approval by themselves.

## Binding sealed holdout — January through June 2026

Decision: `FAIL_2026_SEALED_HOLDOUT`.

| Metric | Result |
|---|---:|
| Trades | 134 |
| Trades/month | 22.5 |
| Return | +4.8% |
| MDD | -15.9% |
| PF | 1.09 |
| Win rate | 36.6% |
| Positive months | 2/6 |
| Return without ten largest winners | -30.7% |

Three-times-cost scenarios were negative, so the frozen V1 failed the robustness gate. The holdout failure is permanent for this version.

## July 2026 unchanged forward extension

Decision: `JULY_FORWARD_SUPPORTS_FROZEN_C2`, diagnosis only.

| Metric | Result |
|---|---:|
| Trades | 17 |
| Return | +8.9% |
| MDD | -4.3% |
| PF | 2.74 |
| Win rate | 58.8% |

Important limitations:

- ETH rose about 19% during the month.
- The model operated near 0.5x research notional, so +8.9% was close to passive 0.5x Long exposure, not evidence of exceptional broad opening selection.
- Fixed-six-hour opening expectancy remained slightly negative after 2x costs.
- Removing the largest winners made July negative.
- q70 exceedance rose to 70.36%, so score calibration did not recover.

## Final attribution

R03.4.2.17 decision: `DIAGNOSIS_SCORE_DRIFT_DOMINANT`.

- No predeclared simple 1D/4H Long gate consistently improved the original C2 across 2024, 2025, 2026 H1 and July.
- Bear-aligned periods were not consistently bad; historically they contributed substantial profit.
- Score exceedance drift was broad across market states.
- July profitability was mainly an exit-overlay/concentrated-winner phenomenon rather than broad recovery of the opening pool.
