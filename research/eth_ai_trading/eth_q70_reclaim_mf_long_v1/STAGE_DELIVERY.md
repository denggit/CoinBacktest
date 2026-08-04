# ETH AI Trading — Cumulative Stage Delivery

Updated through: **R03.4.2.18 archive closeout**

## Current authoritative status

```text
R03.4.2.12: PASS_REAL_1R_TAIL_COMPRESSION_CANDIDATE
R03.4.2.13: PASS_EQUAL_RISK_RETAINED
R03.4.2.14: PASS_C2_FROZEN_NO_ENTRY_UPLIFT
R03.4.2.15: PASS_FINAL_ACCOUNT_LIVE_READINESS
R03.4.2.16: FAIL_2026_SEALED_HOLDOUT
R03.4.2.16.1: JULY_FORWARD_SUPPORTS_FROZEN_C2
R03.4.2.17: DIAGNOSIS_SCORE_DRIFT_DOMINANT
```

## Latest empirical account

| Scenario | Trades | Return | MDD | PF | Positive months | Without top 10 |
|---|---:|---:|---:|---:|---:|---:|
| 1m delay / 2x cost | 480 | +271.4% | -9.4% | 1.73 | 18/24 | +86.1% |
| 1m delay / 3x cost | 480 | +171.7% | -12.2% | 1.51 | 16/24 | +37.1% |
| 3m delay / 3x cost | 482 | +127.3% | -12.6% | 1.41 | 15/24 | +16.4% |
| 5m delay / 3x cost | 476 | +160.8% | -12.2% | 1.50 | 18/24 | +31.5% |

## R03.4.2.15 delivered files

```text
src/ai_research/long_tail_final_account_audit/
research/eth_ai_trading/03_4_2_15_final_account_live_readiness.py
tests/ai_research/test_long_tail_final_account_audit.py
docs/ETH_AI_TRADING_R03_4_2_15_FINAL_ACCOUNT_LIVE_READINESS_RUNBOOK.md
research/eth_ai_trading/R03_4_2_15_PATCH_MANIFEST.md
research/eth_ai_trading/R03_4_2_15_DELIVERY.md
```

## Engineering contract

- Consume only a passed `R03.4.2.14 = PASS_C2_FROZEN_NO_ENTRY_UPLIFT` report.
- Do not rebuild or modify the model, features, q70 threshold, entry, stop or exit.
- Compound WF_2024 and WF_2025 continuously without an annual reset.
- Keep 2023 as development history and 2026 sealed.
- Audit all frozen cost/delay cells, top-10 removal, lot rounding and net-risk reserve.
- Separate monthly audit/shadow retraining from manual quarterly/event-driven promotion.
- Emit restart-safe AetherEdge state fields.

## Latest validation

- R03.4.2.15专项: 7 passed.
- R03.4.2.7 through R03.4.2.15 related regression: 79 passed.
- `tests/ai_research` plus `tests/data_feed`: 218 passed.
- Actual uploaded R03.4.2.14 report consumed successfully.
- Decision: `PASS_FINAL_ACCOUNT_LIVE_READINESS`.
- Full-repository collection remains blocked by five pre-existing missing liquidity/Analyze Tool modules.
- Import-boundary audit remains at 155 pre-existing violations; R03.4.2.15 adds zero.
- No git commit was executed.

## R03.4.2.16 validation

- Stage-specific: 7 passed.
- R03.4.2.7 through R03.4.2.16 related regression: 86 passed.
- AI Research + Data Feed: 225 passed.
- Current container result: `BLOCKED_DATA`; no holdout-open log was created and no empirical 2026 result is claimed.

## R03.4.2.16 sealed result

Completed locally with `FAIL_2026_SEALED_HOLDOUT`; archived as a binding failure. Anchor: 134 trades, +4.8%, -15.9% MDD, PF 1.09 under 1m/2x.

## R03.4.2.16.1 delivery

Status: `JULY_FORWARD_SUPPORTS_FROZEN_C2`.

Includes immutable July-only forward inference, threshold/fit equality checks against R03.4.2.16, H1-versus-July comparison, isolated caches, reports and tests.


## R03.4.2.17 delivery

Status: `DIAGNOSIS_SCORE_DRIFT_DOMINANT`.

Delivered:

- source-report and seal integrity checks;
- completed-bar causal 1D/4H state timeline;
- C2 attribution across 2024, 2025, 2026 H1 and July;
- fixed-6h entry-Edge attribution for H1 and July;
- exact frozen model/threshold/schema recheck and conditional score drift;
- predeclared counterfactual gate tables with explicit non-validation disclosure;
- monthly ETH-vs-C2/state report;
- tests, runbook and cumulative documentation.

## R03.4.2.17 validation

- Stage-specific: 9 passed.
- All `test_long_tail_*.py`: 151 passed.
- AI Research + Data Feed: 240 passed.
- Current container result: `BLOCKED_DATA`; no regime or gate result is claimed.
- Full-repository collection discovered 558 tests and remains blocked by five pre-existing missing liquidity/Analyze Tool modules.
- Import-boundary audit remains at 155 pre-existing violations; R03.4.2.17 adds zero.
- No git commit was executed.


## R03.4.2.17 hotfix3 reporting audit

- Final diagnostic: `DIAGNOSIS_SCORE_DRIFT_DOMINANT`.
- No simple causal 1D/4H Long gate explains or repairs the 2026 seal failure.
- Broad state-conditional q70 score drift is the dominant finding.
- July profit depends on the non-time exit overlay and concentrated winners.
- V1 remains not live-approved; no gate from opened 2026 data is validation.
- Hotfix3 corrects attribution wording, exact calendar monthly return sourcing and 2026 MAE fallback.


## R03.4.2.18 delivery

Status: `ARCHIVE_MODEL_CLOSE_RESEARCH_BRANCH`.

Delivered:

- stable model name `ETH Q70 Reclaim MF Long V1`;
- authoritative archive under `research/eth_ai_trading/archived_models`;
- frozen policy and model card;
- cumulative empirical results and full research timeline;
- failure/lesson record and reproduction index;
- zero-capital lifecycle lock;
- next-model boundary explicitly prohibiting breakout chasing and leverage-based risk distortion.

No model behavior or historical result was changed.
