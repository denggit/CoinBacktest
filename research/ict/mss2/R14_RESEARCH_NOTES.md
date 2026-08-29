# R14 — Liquidity Acceptance / Continuation

Date: 2026-08-16

## Hypothesis

R12/R13 showed that deeper same-side completed-trend liquidity is reached before opposite delivery much more often than direct reversal. R14 tested whether outside acceptance after the root sweep provides a causal continuation entry in the sweep direction.

- BSL sweep -> long continuation.
- SSL sweep -> short continuation.
- Target: root-time frozen deeper same-side completed-trend liquidity touch.
- Stop: full reclaim beyond the far edge of the swept region plus 2bps.
- Entry family: root close outside; 5m and 15m final close outside with 60/80/100% outside-close persistence.
- Closed-bar signals enter next 1m open; prior target/stop makes delayed signals stale; same-bar TP/SL is stop-first.

No FVG, order flow, OI, books, ML, filters, risk tiers or portfolio sizing were added.

## Data and causality

R14 uses 753 pre-holdout events with a valid deeper-same-side target. The 2025-08-01 holdout contains 322 eligible R12 events and remains sealed. Bare 1m coverage is complete through 2026-08-15 23:59. The report contains 5,271 model/event rows and 13 causal checks with zero violations. Six focused R14 tests pass.

## Result

BSL continuation fails consistently:

- root-close outside 2x PF: 0.45 discovery / 0.49 validation;
- 5m acceptance PF: about 0.64 / 0.51;
- 15m acceptance PF: about 0.61–0.63 / 0.65;
- all 2023/2024/2025 BSL variants are below PF 1.

SSL root-close acceptance also fails as a distant-target strategy:

- 103 discovery trades, 4.29/month, PF 1.11, 29.2% positive months;
- 39 validation trades, 6.50/month, PF 0.56, 16.7% positive months;
- 3x discovery PF already falls below one.

SSL persistent acceptance has positive headline PF but does not pass robustness:

- 5m variants: 28 discovery trades at PF 2.91 and 8 validation trades at PF 1.82;
- 15m p60: 22 discovery trades at PF 4.40 and 7 validation trades at PF 2.16;
- persistence thresholds often produce identical filled rows because weaker signals are invalidated before delayed entry;
- top-five removal reduces discovery PF to about 0.17–0.22 and validation PF to zero or undefined because nearly the entire tiny winner set is removed;
- longest entry gaps are roughly 123–148 days in discovery.

Therefore the positive SSL cells are sparse right-tail observations, not a live sleeve.

## Frozen conclusions

1. BSL acceptance continuation is stopped.
2. Root-close SSL continuation to distant completed-trend liquidity is stopped as a strategy.
3. Waiting 5/15 minutes for outside persistence improves selection but destroys sample size and does not solve top-winner dependence.
4. Do not optimize the 60/80/100% threshold; the surviving filled rows are nearly identical.
5. Do not add FVG or order-flow execution to rescue R14.
6. Holdout remains sealed and no capital strategy is promoted.

## Next question

The higher-frequency SSL root-close entry often travels materially in the continuation direction before reclaim even though the distant completed-trend target is rarely reached. MFE/risk is descriptive and cannot establish same-bar first passage. R15 will therefore run an exact, stop-first 0.5R/1R/2R/3R first-passage ladder on the frozen SSL root-close entry.

This is an exit/path diagnostic, not a claim that fixed R is the final target. It tests whether the R14 failure is specifically distant-target/right-tail dependence. No time exit, admission filter, allocation rule, runner, or holdout access is allowed in R15.

## Primary evidence

- `data/reports/research/ict/mss2/r14_liquidity_acceptance_continuation/00_manifest.json`
- `02_holdout_seal.csv`
- `04_acceptance_feature_rows.csv.gz`
- `05_continuation_entry_rows.csv.gz`
- `06_entry_model_scorecard.csv`
- `07_entry_model_years.csv`
- `09_causal_audit.csv`
