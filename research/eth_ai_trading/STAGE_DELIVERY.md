# ETH AI Trading — Current Stage Delivery

Updated through: **R03.4.2.9 code delivery; R03.4.2.8B empirical result complete**

This file is cumulative. Every later patch must update it rather than creating an isolated progress note.

## Current position

- Last completed empirical stage: `R03.4.2.8B`.
- R03.4.2.8B decision: `FAIL_NO_ROBUST_DUAL_SLOT_ACCOUNT_POLICY`.
- Current active code stage: `R03.4.2.9`, structural protection and dynamic risk-release audit.
- Frozen working chain: q70 ML entry + deterministic `failed_reclaim` + 3% disaster protection.
- Fixed six hours is diagnostic only.
- Every new primary remains 1R; static 0.5/0.5 and 0.65/0.35 reservations are abandoned.
- 2026 remains sealed.

## Full stage result index

| Stage | Decision / status | Retained conclusion |
|---|---|---|
| R00 | Framework complete | AI research structure, causal and reporting contracts established. |
| R01 | `FAIL_NO_VALIDATION_EDGE` | Trades-only short-horizon baseline was insufficient. |
| R02 | Framework complete | Short/medium/long Sleeve separation retained. |
| R03 | `FAIL_VALIDATION` | Original multi-frame 3%–5% Swing model did not validate. |
| R03.1 | `FAIL_VALIDATION` | Target-before-risk causal labels still did not validate. |
| R03.2 | `FAIL_VALIDATION` | Longer context did not create a stable Swing entry Edge. |
| R03.3 | `FAIL_NO_STABLE_PROCESS_FORECAST` | Sparse future-process classes were not stably predictable. |
| R03.3.1 | `FAIL_ACTIONABLE_PROCESS_ALERT` | Alerts were too late or lacked remaining tradable space. |
| R03.3.2 | `PASS_STABLE_INTENSITY_RANKING` | Future opportunity intensity is context, not a direct order trigger. |
| R03.3.3 | `PASS_STATE_CONTINUITY_SIGNAL` | Market-state continuity has predictive value. |
| R03.3.3.1 | `PASS_STATE_CONTINUITY_INCREMENT` | Continuity beat mechanical baselines but remains context only. |
| R03.4 | `FAIL_NO_STABLE_STATE_CONTEXT_UPLIFT` | Market states did not improve entry trading results cross-year. |
| R03.4.1 | `FAIL_NO_STABLE_LONG_STATE_UPLIFT` | State meta-layer failed; frozen base-model high-score tail exposed entry Edge. |
| R03.4.2 | `FAIL_NO_ROBUST_PATH_EXIT` | Tight stops, small targets and early trailing damaged delayed winners. |
| R03.4.2.1 | `PASS_PATH_ATLAS_READY_FOR_CAUSAL_EXIT_RESEARCH` | Full long-tail path atlas completed. |
| R03.4.2.2 | `FAIL_NO_STABLE_PATH_RECOGNITION` | Early path recognition was not stable across years. |
| R03.4.2.3 | `RESEARCH_CONTINUE_NO_ROBUST_POLICY` | Multi-stage holding policies did not robustly improve total profit. |
| R03.4.2.4 | `PASS_Q70_CROSS_YEAR_EXPANSION` | q70 is the main opening pool; q70/q80/q90 tiers retained. |
| R03.4.2.5 | `FAIL_NO_ROBUST_FAILURE_OVERLAY` | Failure ranking existed; executable ML exit Overlay failed. |
| R03.4.2.6 | `RESEARCH_CONTINUE_RANKING_ONLY` | Incremental-hold ML had local ranking only; holding-ML line stopped. |
| R03.4.2.7 | `FAIL_NO_ROBUST_NON_TIME_STRUCTURAL_EXIT` | No final exit passed; `failed_reclaim` retained as best deterministic working baseline. |
| R03.4.2.8A | `FAIL_NO_CROSS_YEAR_OCCUPIED_SIGNAL_ELIGIBILITY` | Strict healthy/recovered add subset was too small and concentrated. |
| R03.4.2.8B | `FAIL_NO_ROBUST_DUAL_SLOT_ACCOUNT_POLICY` | Static dual slots restored frequency and lowered MDD but diluted primary returns too much. |
| R03.4.2.9 | `FAIL_NO_ROBUST_STRUCTURE_PROTECTION` | Direct Pivot hard stops exited too often; S2 was cross-year unstable. |
| R03.4.2.10 | Code delivered; run pending | Test real partial closes and one-R-conserving q70 risk migration. |

## Frozen empirical facts

### q70 fixed-6h diagnostic, 1-minute delay, 2x cost

- 2024: 431 signals, mean +0.281%, PF 1.64, win 62.4%, diagnostic MDD -18.7%.
- 2025: 419 signals, mean +0.517%, PF 2.22, win 66.1%, diagnostic MDD -17.5%.
- Original pool frequency is roughly 35 signals/month.

### Single-position `failed_reclaim`, 1-minute delay, 2x cost

- 2024: 236 trades; account return 56.2%; account MDD -8.6%.
- 2025: 244 trades; account return 68.6%; account MDD -7.4%.
- Occupancy skipped 45.2% of q70 signals in 2024 and 41.8% in 2025.
- Benefit: thicker winners and no scheduled exit.
- Cost: roughly 20 executed trades/month and slower opportunity turnover.

### R03.4.2.8A occupied-signal atlas

- Strict healthy/recovered candidates: 47 in 2024 and 28 in 2025.
- Dangerous-average-down classifications: 61.5% and 70.3% of occupied signals.
- The strict subset failed count, concentration and delay requirements.
- This did not invalidate the full q70 pool or justify reducing strategy frequency.

### R03.4.2.8B static dual slots

At 1-minute delay and 2x cost:

- P1/P2 coverage: 81.9% in 2024 and 87.1% in 2025; roughly 29–30 Tranches/month.
- P2 return: 43.1% / 54.3%, versus P0 56.2% / 68.6%.
- P2 MDD: -6.3% / -5.4%, versus P0 -8.6% / -7.4%.
- P3 removed dangerous adds but coverage fell to 66.6% / 69.0% and returns remained below P0.
- The second Tranche itself was profitable. Permanent primary dilution caused the main opportunity cost.

## Active R03.4.2.9 contract

Protection policies:

1. S0 disaster-only 3% hard protection.
2. S1 latest causally confirmed structural floor.
3. S2 one confirmed floor behind the latest structure.

Dynamic policies:

1. D0 one full-R primary only.
2. D1 released-risk secondary capped at 0.35R.
3. D2 released-risk secondary capped at 0.50R.
4. D3 0.50R cap with a non-losing active-position requirement.

Hard boundaries:

- every standalone primary starts at 1R;
- no static reservation for a future signal;
- protection updates use completed 15m bars and activate at the next 1m open;
- stops only rise and gap fills use the open;
- second risk cannot exceed enforceable released risk;
- maximum two simultaneous virtual Tranches;
- live stop-defined remaining loss cannot exceed 1R;
- no new signal resets or widens the old Tranche;
- 2x/3x cost and 1/3/5-minute delay;
- 2026 sealed.

## Planned order after R03.4.2.9

1. Run R03.4.2.9 and inspect `gpt_review_pack.zip`.
2. If a dynamic policy passes, retain it as account-capacity candidate.
3. If only protection passes, retain the stop and stop the current add rule.
4. Optimize entry timing and MAE without deleting most q70 signals.
5. Freeze q70/q80/q90 risk only after the executable stop is known.
6. Re-audit the final non-time exit and complete account backtest.
7. Open 2026 only after all rules are frozen.
8. Design AetherEdge plugin and shadow-live validation.

## Delivery validation

- New R03.4.2.9专项 tests: 9 passed.
- Related R03.4.2.7/8A/8B/9 tests: 32 passed.
- `tests/ai_research` plus `tests/data_feed`: 171 passed.
- Source-report smoke reached `BLOCKED_DATA` cleanly because the mounted Sources database contains no Trade Bar rows; no empirical R03.4.2.9 result is claimed.
- Full repository collection remains blocked by five pre-existing missing liquidity/analyze-tool modules.
- Import-boundary test remains red from 155 pre-existing cross-research imports outside this stage; this patch adds zero new violations.
- This patch does not modify q70, `failed_reclaim` or shared Loader behavior.


### R03.4.2.9 Pivot hard-protection result

At 1-minute delay and 2x cost:

- S1 latest-confirmed: hard-stop share about 99.8%; 2024 return 7.3%, 2025 return 45.9%.
- S2 one-level-lagged: hard-stop share about 86%; 2024 return 28.2%, 2025 return 88.7%.
- P0 remained 56.2% / 68.6% with MDD -8.6% / -7.4%.
- No structure stop passed the prerequisite, so dynamic stop-funded Tranche policies were not run.

## Active R03.4.2.10 contract

Policies:

1. P0 single full-R baseline.
2. R1/R2 real 25%/50% partial close on first proven non-losing soft break.
3. M1/M2 0.35R/0.50R risk migration to a later q70 signal.
4. H1 25% partial de-risking followed by up to 0.35R migration.

Hard boundaries:

- 3% is the only hard safety floor; no Pivot hard stop.
- Failed-Reclaim remains each tranche's independent non-time exit.
- every flat-to-position cycle starts with a full 1R primary;
- cycle risk budget is fixed in dollars at primary entry;
- migration uses real free capacity or same-open old-unit reduction;
- maximum two simultaneous Tranches;
- no migration from losing, BROKEN or pending-Failed-Reclaim roots;
- fixed 6h diagnostic only; 2026 sealed.

## Delivery validation for R03.4.2.10

- New R03.4.2.10专项 tests: 9 passed.
- Related R03.4.2.7/8A/8B/9/10 tests: 41 passed.
- `tests/ai_research` plus `tests/data_feed`: 180 passed.
- P0 event sequence, total return and MDD match the frozen R03.4.2.8B simulator on synthetic parity tests.
- Source-report smoke reaches `BLOCKED_DATA` cleanly when the mounted environment lacks Trade Bar rows; no empirical result is claimed by code delivery alone.
- Full repository collection remains blocked by five pre-existing missing liquidity/analyze-tool modules.
- Import-boundary audit remains red from pre-existing cross-research imports outside this stage; the new entrypoint adds no `research -> research` dependency.
