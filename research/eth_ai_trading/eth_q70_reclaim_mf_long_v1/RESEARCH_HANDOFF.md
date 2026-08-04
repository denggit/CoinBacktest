# ETH AI Trading Research Handoff

Updated through: **R03.4.2.18 archive closeout**

## 1. Current exact position

```text
Last completed stage: R03.4.2.18
Model: ETH Q70 Reclaim MF Long V1
Binding decision: FAIL_2026_SEALED_HOLDOUT
Final diagnosis: DIAGNOSIS_SCORE_DRIFT_DOMINANT
Lifecycle: ARCHIVED_AFTER_SEALED_HOLDOUT_FAILURE
Live approved: NO
Capital allocation: 0
```

The V1 research branch is closed. Historical reproduction is allowed; retuning against opened 2026 H1/July is forbidden.

## 2. Archived frozen V1 policy

```text
q70 ML long opening pool
+
next observable 1m open immediately
+
equal risk for every q70 score tier
+
real 2% exchange-side hard stop
+
1.5% adverse excursion followed by completed 15m-close soft-failure confirmation
+
deterministic failed_reclaim non-time exit
```

Frozen facts:

- No fixed take profit and no fixed-time final exit.
- Fixed six hours remains an independent opening-signal diagnostic only.
- `failed_reclaim` is deterministic market-structure logic, not ML.
- Later score changes cannot renew, add, average down, reduce or exit a live position.
- No split entry, Turtle add, pyramid or occupied-q70 add-on.
- The model uses multi-timeframe causal features; 15m is the primary decision cadence and 1m is the execution/hard-stop path.
- 2023 is training/development history, not independent OOS account performance.
- January–June 2026 has been opened and consumed as a failed sealed holdout; V1 is permanently not live-approved.

## 3. Historical metric contract

Never describe these as one strategy suddenly changing win rate:

| Scope | Meaning |
|---|---|
| fixed 6h all signals | every q70 signal independently exits after six hours; full-notional diagnostic |
| P0 failed_reclaim single position | one active position with structural exit; full-notional path diagnostic |
| C2 equal-one-R account | risk-sized account simulation with 2% hard stop and 1.5% soft failure |

Fixed-6h win rate was 62.4% / 66.1%; C2 account win rate is 43.6% / 49.6%. The difference mainly comes from the non-time structural exit, not from R03.4.2.13 or R03.4.2.14 changing the model.

## 4. R03.4.2.12 — real tail compression

Decision:

```text
PASS_REAL_1R_TAIL_COMPRESSION_CANDIDATE
```

Primary annual account results at 1-minute delay / 2x cost:

| Policy | WF_2024 return / MDD | WF_2025 return / MDD | Initial notional | Conclusion |
|---|---|---|---:|---|
| P0 real 3% hard stop | +56.2% / -8.6% | +68.6% / -7.4% | ~0.33x | comfortable baseline |
| F1 1.5%-sized / 3% real tail | +134.4% / -12.5% | +193.6% / -12.9% | ~0.67x | two-R attribution only |
| **C2 real 2% + soft1.5%** | **+85.1% / -9.4%** | **+100.7% / -8.4%** | **~0.50x** | passed candidate |
| C15 real 1.5% | ~+123% / ~-14% | ~+139% / ~-15% | ~0.67x | too many normal ETH sweeps |

C2 improves capital efficiency by making the executable hard-stop distance equal the sizing distance. It removes more losers than winners, but does sacrifice a small number of recoverable trades.

## 5. R03.4.2.13 — score risk sizing

Decision:

```text
PASS_EQUAL_RISK_RETAINED
```

Score-tier expectancy changed order across years:

```text
2024: q70-q80 strongest
2025: q90+ strongest; q80-q90 negative
```

Therefore q70 remains a valid opening threshold, but q70/q80/q90 does not receive different risk. All C2 entries use equal risk.

## 6. R03.4.2.14 — entry timing and MAE

Decision:

```text
PASS_C2_FROZEN_NO_ENTRY_UPLIFT
```

Reliable conclusions:

- Immediate next-open C2 remained best.
- Fixed 30-60 minute waiting, no-chase waiting and pullback/reclaim entry all reduced return, win rate and MDD quality.
- q70 signals had positive short-horizon drift; waiting generally paid a worse price.
- Deep-MAE winners were rare, so poor initial entry was not the dominant loss source.
- The score-rise E1/E2 implementations fell back 100% of the time in this historical report, so the stage proves bounded fixed waiting is bad; it does not prove that a future true continuous score stream can never contain information. This limitation does not justify reopening C2 now.

## 7. R03.4.2.15 — continuous account and live readiness

Decision:

```text
PASS_FINAL_ACCOUNT_LIVE_READINESS
```

### Main continuous OOS account

WF_2024 and WF_2025 are compounded continuously without resetting equity:

| Metric | Result |
|---|---:|
| Trades | 480 |
| Trades/month | 20.0 |
| Total return | +271.4% |
| CAGR | 92.8% |
| MDD | -9.4% |
| PF | 1.73 |
| Win rate | 46.7% |
| Positive months | 18/24 |
| Positive quarters | 8/8 |
| Return without ten largest winners | +86.1% |
| Longest losing streak | 9 trades |
| Longest daily drawdown period | 74 days |
| Longest gap between new entries | 12.1 days |
| Median / P90 holding | 14.2h / 35.0h |
| Worst historical net cycle | -1.129R |

### Cost and delay robustness

All frozen scenarios remained profitable:

| Scenario | Return | MDD | PF | Without top 10 |
|---|---:|---:|---:|---:|
| 1m / 2x cost | +271.4% | -9.4% | 1.73 | +86.1% |
| 1m / 3x cost | +171.7% | -12.2% | 1.51 | +37.1% |
| 3m / 2x cost | +211.1% | -10.5% | 1.62 | +65.0% |
| 3m / 3x cost | +127.3% | -12.6% | 1.41 | +16.4% |
| 5m / 2x cost | +255.5% | -9.4% | 1.73 | +90.6% |
| 5m / 3x cost | +160.8% | -12.2% | 1.50 | +31.5% |

### Live price-risk reserve

At a raw 1% price-risk budget, the historical worst net loss is about 1.129R under the anchor and about 1.193R under 3x cost. Therefore deploy initially with:

```text
price risk: 0.83%-0.85% equity
fee/slippage/jump reserve: 0.15%-0.17%
net tail target: near 1% equity
```

### OKX whole-contract sizing

With ETH-USDT-SWAP contract value 0.1 ETH in the current project configuration:

- 500U is frequently unable to place one contract at the target risk.
- Under the conservative ~0.84% live budget, 1,000U has about 7.3% historically untradable entries and ~67% sizing efficiency.
- 3,000U is fully tradable in the historical sample and reaches ~88% sizing efficiency.
- 10,000U+ closely tracks the conservative target risk.

## 8. Model retraining and release governance

Do not automatically replace the model every month.

```text
continuous: serve one immutable champion artifact
daily: data/feature/schema health
monthly: score distribution, q70 frequency, calibration, MAE, PF and cost audit
monthly optional: train a shadow candidate with causal cutoff and embargo
quarterly or event-driven: explicit promotion gate
emergency: rollback to last known-good artifact
```

A candidate may be retrained monthly, but the live champion remains unchanged unless the candidate passes frozen OOS, stress, shadow, state-recovery and rollback gates. Every deployed artifact must carry model version, feature-schema hash, training cutoff and calibration threshold.

## 9. Cumulative stage results

| Stage | Decision / status | Durable result |
|---|---|---|
| R00 | framework delivered | causal/reporting layout |
| R01 | FAIL_NO_VALIDATION_EDGE | trades-only baseline insufficient |
| R02 | framework delivered | separate short/medium/long sleeves |
| R03 | FAIL_VALIDATION | first swing target unstable |
| R03.1 | FAIL_VALIDATION | target-before-risk labels failed |
| R03.2 | FAIL_VALIDATION | long context did not create robust entry Edge |
| R03.3 | FAIL_NO_STABLE_PROCESS_FORECAST | discrete process labels unstable |
| R03.3.1 | FAIL_ACTIONABLE_PROCESS_ALERT | alerts too late/thin |
| R03.3.2 | PASS_STABLE_INTENSITY_RANKING | opportunity thickness rankable |
| R03.3.3 | PASS_STATE_CONTINUITY_SIGNAL | state persistence descriptive |
| R03.3.3.1 | PASS_STATE_CONTINUITY_INCREMENT | continuity exceeds mechanical persistence only |
| R03.4 | FAIL_NO_STABLE_STATE_CONTEXT_UPLIFT | no stable trading uplift |
| R03.4.1 | FAIL_NO_STABLE_LONG_STATE_UPLIFT | state meta failed; base long tail revealed Edge |
| R03.4.2 | FAIL_NO_ROBUST_PATH_EXIT | simple stop/target/trailing damaged winners |
| R03.4.2.1 | PASS_PATH_ATLAS_READY_FOR_CAUSAL_EXIT_RESEARCH | path atlas completed |
| R03.4.2.2 | FAIL_NO_STABLE_PATH_RECOGNITION | early path classifier unstable |
| R03.4.2.3 | RESEARCH_CONTINUE_NO_ROBUST_POLICY | holding policy not robust |
| R03.4.2.4 | PASS_Q70_CROSS_YEAR_EXPANSION | q70 main opening pool |
| R03.4.2.5 | FAIL_NO_ROBUST_FAILURE_OVERLAY | executable ML failure threshold rejected |
| R03.4.2.6 | RESEARCH_CONTINUE_RANKING_ONLY | stop holding ML |
| R03.4.2.7 | FAIL_NO_ROBUST_NON_TIME_STRUCTURAL_EXIT | failed_reclaim retained as best component |
| R03.4.2.8A | FAIL_NO_CROSS_YEAR_OCCUPIED_SIGNAL_ELIGIBILITY | strict add subset too small/concentrated |
| R03.4.2.8B | FAIL_NO_ROBUST_DUAL_SLOT_ACCOUNT_POLICY | static slots dilute primary |
| R03.4.2.9 | FAIL_NO_ROBUST_STRUCTURE_PROTECTION | Pivot hard stops swept |
| R03.4.2.10 | FAIL_NO_ROBUST_PARTIAL_OR_MIGRATION | partial/migration underperform |
| R03.4.2.11 | FAIL_NO_ROBUST_STAGED_EXECUTION | split/Turtle/pyramid rejected |
| R03.4.2.12 | **PASS_REAL_1R_TAIL_COMPRESSION_CANDIDATE** | C2 real 2% + soft1.5% passed |
| R03.4.2.13 | **PASS_EQUAL_RISK_RETAINED** | all q70 equal risk |
| R03.4.2.14 | **PASS_C2_FROZEN_NO_ENTRY_UPLIFT** | immediate entry remains best |
| R03.4.2.15 | **PASS_FINAL_ACCOUNT_LIVE_READINESS** | continuous account and deployment gates passed |
| R03.4.2.16 | **FAIL_2026_SEALED_HOLDOUT** | untouched H1 edge thinned; V1 not live-approved |
| R03.4.2.16.1 | **JULY_FORWARD_SUPPORTS_FROZEN_C2** | July C2 recovered but fixed-6h and calibration remained weak/drifted |
| R03.4.2.17 | **DIAGNOSIS_SCORE_DRIFT_DOMINANT** | no simple Long-state separation; broad conditional score drift |

## 10. Explicitly abandoned directions

- previously tested complex/learned market-state trading layers as direct live policies; simple completed-bar states may now be used only for post-seal diagnosis and separately versioned V2 research;
- holding renewal by score;
- score-rise add, averaging down or pyramiding;
- static second-slot reservation;
- Pivot hard stop on the complete base;
- partial reduction and risk migration;
- split entry, Turtle and classic pyramid;
- 1.5% complete-position hard stop;
- fixed-time final exit;
- score-tier risk maps;
- fixed delayed entry and pullback/reclaim replacement entries;
- year-specific live rules;
- deleting trades to improve PF;
- automatic monthly champion replacement.

## 11. Current next branch

1. Keep `ETH Q70 Reclaim MF Long V1` archived with zero capital.
2. Do not repair q70, stop, exit or state gates on opened 2026 H1/July.
3. Start a separately versioned `ETH Trend Pullback Continuation Long/Short V1`.
4. Use 1D/4H for trend persistence and remaining-runway estimation.
5. Seek entry through 1H/30m pullback plus 15m/5m reclaim/re-acceleration; do not chase the breakout.
6. Predeclare a maximum hard-stop distance and skip structurally wide entries.
7. Validate Long and Short independently, then wait for a new untouched future holdout.


## 10. R03.4.2.16 — one-time 2026 sealed validation implementation history

Historical implementation status before local opening: `CODE_COMPLETE_PENDING_LOCAL_2026_DATA`. The seal was later opened and failed as recorded below. Before any 2026 loader access, the stage seals frozen code, configuration, R03.4.2.15 source reports and historical feature-schema evidence with SHA-256. Fit ends 2025-09-30 06:00, q70 calibrates only on Q4 2025, and January-June 2026 is used only for final inference/account scoring. Changed code/config/source after opening is rejected.

## 12. R03.4.2.16 sealed result — opened and failed

The immutable January-June 2026 seal completed with `FAIL_2026_SEALED_HOLDOUT`.

Anchor 1m delay / 2x cost:

| Metric | 2026 H1 |
|---|---:|
| Executed cycles | 134 |
| Return | +4.8% |
| MDD | -15.9% |
| PF | 1.09 |
| Win rate | 36.6% |
| Positive months | 2/6 |
| Return without top 10 | -30.7% |

The fixed-6h opening diagnostic also weakened to mean +0.081%, win rate 53.2% and PF 1.16 at 2x cost; at 3x cost it became negative. Q4-2025 q70 threshold exceedance rose to 58.14% in 2026, indicating material score/calibration drift. The H1 result is permanently archived and may not be repaired by parameter changes on the same period.

## 13. R03.4.2.16.1 — July forward extension

Status: `JULY_FORWARD_SUPPORTS_FROZEN_C2`.

July completed with 17 anchor trades, +8.9%, -4.3% MDD and PF 2.74 at 1m/2x. All cost/delay cells were profitable, but fixed-6h opening expectancy remained -0.030% at 2x, q70 exceedance rose to 70.36%, and return without the largest winners was negative. This supports regime dependence and `failed_reclaim` concentration, not general model recovery.

Interpretation discipline:

- good July: supports regime dependence, but cannot reverse H1 failure or authorize live deployment;
- mixed July: retain uncertainty and proceed to drift/regime attribution;
- weak July: further supports model decay, score drift or missing Long-regime gating;
- no threshold, entry, stop, exit or sizing change is permitted after July is opened.


## 14. R03.4.2.17 — sealed-failure attribution and Long-state diagnostic

Status: `DIAGNOSIS_SCORE_DRIFT_DOMINANT`.

Validated diagnostic facts:

- all 631 frozen C2 cycles received causal completed-bar 1D/4H state context;
- 2026 H1 `BEAR_ALIGNED` remained slightly positive (mean +0.105%), while weighted Bull contexts were slightly negative (mean -0.057%); therefore a simple Long trend gate does not explain the seal failure;
- 2024 and 2025 also earned substantial return in `BEAR_ALIGNED`, so excluding bearish states destroys historical Edge;
- state-conditional q70 exceedance shifted broadly, with median H1 minus Q4-2025 calibration drift of +29.6 percentage points;
- July C2 was positive while fixed-6h expectancy remained non-positive, confirming dependence on the `failed_reclaim` exit overlay and a few large winners;
- no predeclared gate beat frozen `G0_ALL` in all four opened periods. `G1_EXCLUDE_BEAR_ALIGNED` stayed nominally positive but materially reduced 2024, 2025 and H1 return, so it is not an uplift.

Reporting corrections in hotfix3:

- dynamic attribution text now reflects actual state means;
- calendar C2 monthly returns come from the frozen source account reports rather than entry-month trade cohorts;
- trade-path MAE remains unavailable for 2026 source cycles and is kept distinct from account drawdown; `cycle_max_drawdown` is reported separately.

No R03.4.2.17 gate is validated. V1 remains `NOT LIVE APPROVED`. A separately versioned V2 may study score recalibration/drift detection, but must use future untouched data for qualification.


## R03.4.2.18 — formal model archive

Archived as `ETH Q70 Reclaim MF Long V1` under `research/eth_ai_trading/archived_models/eth_q70_reclaim_mf_long_v1`. The archive contains the frozen policy, model card, full timeline, empirical closeout, failure lessons and zero-capital lifecycle lock.

Next independent model boundary: trend persistence plus pullback/reclaim/re-acceleration execution; no breakout chasing, no inherited q70 score, and no claim that high exchange leverage makes a wide stop safe.
