# ETH AI Trading — Completed Work

Updated through: **R03.4.2.15 empirical completion**

Read `RESEARCH_HANDOFF.md` first for the complete stage table.

## Durable positive results

- R03.3.2: future six-hour opportunity intensity is stably rankable.
- R03.4.1 / R03.4.2.4: the frozen long opening model contains real Edge; q70 is the main opening pool.
- R03.4.2.7: `failed_reclaim` is the best current deterministic non-time exit component.
- R03.4.2.12: C2 passed — real 2% hard stop plus 1.5% completed-close soft failure while retaining `failed_reclaim`.
- R03.4.2.13: score-tier risk maps failed cross-year monotonicity; equal one-R is frozen.
- R03.4.2.14: delayed entry, score-wait fallback and pullback/reclaim entry did not improve C2; immediate next-open remains frozen.
- R03.4.2.15: final continuous account and live-readiness gates passed.

## Frozen account evidence

At 1-minute delay / 2x cost:

| Scope | Trades | Return | MDD | PF | Positive months |
|---|---:|---:|---:|---:|---:|
| WF_2024 | 236 | +85.1% | -9.4% | 1.71 | 10/12 |
| WF_2025 | 244 | +100.7% | -8.4% | 1.74 | 8/12 |
| Continuous 2024-2025 | 480 | +271.4% | -9.4% | 1.73 | 18/24 |

Continuous account details:

- CAGR: 92.8%.
- Positive quarters: 8/8.
- Median holding: 14.2 hours; P90: 35.0 hours.
- Frequency: 20 trades/month.
- Longest losing streak: 9 trades.
- Longest daily drawdown period: 74 days.
- Longest gap between new entries: 12.1 days.
- Return after removing the ten largest winners: +86.1%.
- Historical worst net cycle: -1.129R at the nominal 1% price-risk budget.

All frozen 1/3/5-minute delay and 2x/3x cost cells remained profitable. The weakest cell returned +127.3% with approximately -12.6% MDD.

## Durable negative results

Do not repeat:

- market-state direction/filter/sizing layers;
- holding ML and probability-threshold failure exits;
- local Pivot hard stops on the complete base;
- static dual risk slots;
- partial reduction or old-to-new risk migration;
- split entry, Turtle add and classic pyramid;
- real 1.5% hard stop for the complete base;
- F1 as a one-R policy, because its headline result contained a hidden two-R tail;
- score-tier sizing or score-rise adding;
- fixed waiting, no-chase waiting and pullback/reclaim as replacement entry rules;
- fixed-time final exit;
- parameter grids after frozen results.

## Deployment conclusions

- This is a medium-frequency swing-long sleeve, not MHF/HF.
- The strategy has no fixed take profit; profits exit through `failed_reclaim`.
- Initial live price-risk budget should be around 0.83%-0.85%, not a raw 1.00%, to reserve fees/slippage/jump risk.
- Under the conservative live budget, 1,000U still has about 7% untradable historical entries and materially underuses risk; about 3,000U is fully tradable and reaches roughly 88% sizing efficiency.
- Model monitoring may be monthly; automatic monthly replacement is forbidden.

## R03.4.2.16 sealed-validation implementation

Completed: immutable pre-open seal, pre-2026 final fit/Q4-2025 calibration, exact frozen C2 stress grid, hard/quality gates, post-open mutation rejection and partial-year disclosure. No empirical 2026 claim exists in the patch environment because its local Trade Bar table is empty.

## R03.4.2.16 — completed sealed failure

Decision: `FAIL_2026_SEALED_HOLDOUT`.

- Seal unchanged and feature schema matched.
- 1m/2x: 134 trades, +4.8%, MDD -15.9%, PF 1.09, win rate 36.6%.
- Only 2/6 positive months; without top ten winners -30.7%.
- Multiple 3x-cost/delay cells were negative.
- q70 exceedance drifted to 58.14%.
- Frozen C2 is not approved for live deployment.

## R03.4.2.16.1 — implementation complete

Decision status: `CODE_COMPLETE_PENDING_LOCAL_JULY_RUN`.

- New July-only forward window.
- Same pre-2026 fit and Q4-2025 threshold.
- Independent immutable seal.
- H1-versus-July score/account comparison.
- No post-H1 repair or post-July tuning.


## R03.4.2.16 and R03.4.2.16.1 empirical update

- H1 2026 seal: `FAIL_2026_SEALED_HOLDOUT`; 134 trades, +4.8%, -15.9% MDD, PF 1.09, 2/6 positive months at 1m/2x.
- H1 fixed-6h opening expectancy fell to +0.081% at 2x and negative at 3x.
- q70 exceedance drifted to 58.14%.
- July forward extension: `JULY_FORWARD_SUPPORTS_FROZEN_C2`; 17 trades, +8.9%, -4.3% MDD, PF 2.74 at 1m/2x.
- July fixed-6h opening expectancy remained -0.030% at 2x and q70 exceedance rose to 70.36%.
- July profit depended on a small number of `failed_reclaim` winners; it cannot reverse the H1 failure or approve live trading.

## R03.4.2.17 implementation

A post-seal diagnostic is ready. It does not modify C2. It causally aligns completed 1D/4H contexts, attributes C2/fixed-6h results by state, verifies the exact frozen model recipe, measures conditional score drift, and labels all counterfactual gates as unvalidated development evidence.


## R03.4.2.17 hotfix3 reporting audit

- Final diagnostic: `DIAGNOSIS_SCORE_DRIFT_DOMINANT`.
- No simple causal 1D/4H Long gate explains or repairs the 2026 seal failure.
- Broad state-conditional q70 score drift is the dominant finding.
- July profit depends on the non-time exit overlay and concentrated winners.
- V1 remains not live-approved; no gate from opened 2026 data is validation.
- Hotfix3 corrects attribution wording, exact calendar monthly return sourcing and 2026 MAE fallback.


## R03.4.2.18 — archived ETH Q70 Reclaim MF Long V1

- Assigned stable name and version identity.
- Created immutable model archive and zero-capital lifecycle lock.
- Preserved 2024–2025 development success, 2026 sealed failure, July diagnostic recovery and score-drift attribution in one authoritative directory.
- Closed the V1 branch without deleting reproducible scripts or pretending the failure was repaired.
- Defined the next independent model boundary as trend pullback continuation Long/Short, not breakout chasing.
