# R03.4.2.8B Delivery

## Stage status

- Previous empirical stage: `R03.4.2.8A`.
- Previous decision: `FAIL_NO_CROSS_YEAR_OCCUPIED_SIGNAL_ELIGIBILITY`.
- Current delivery: R03.4.2.8B code, tests, runbook and cumulative handoff updates.
- Empirical R03.4.2.8B result: pending local 2024–2025 Trade Bar run.
- 2026 remains sealed.

## Why 2.8B continues after the strict 2.8A gate failed

R03.4.2.8A proved that restricting second entries to only the strict healthy/recovered subset leaves 47 events in 2024 and 28 in 2025. That is too narrow for the project objective and its top-10 removal gate was disproportionately severe for the small 2025 subset.

R03.4.2.8B does not loosen into unlimited averaging down. It tests a different hypothesis: preserve broader q70 opportunity coverage through two pre-allocated account-risk slots, with no third tranche and no risk budget above one full R.

## Frozen inputs

- R03.4.1 q70 opening model and calibration.
- Standalone deterministic `failed_reclaim` outcome for every q70 event.
- 3% disaster protection.
- Same policies in WF_2024 and WF_2025.
- 1/3/5-minute delay and 2x/3x cost stress.
- Fixed 6h as diagnostic only.

## Policies

- P0: one 1.0R slot.
- P1: 0.5R + 0.5R.
- P2: 0.65R + 0.35R.
- P3: 0.65R + 0.35R, blocking a second tranche during dangerous average-down or failed-reclaim confirmation.

Every tranche has independent entry, virtual ledger, cost, PnL and `failed_reclaim` exit. Exchange execution remains one ETH net position later in AetherEdge.

## Main acceptance questions

1. Is q70 coverage restored above 70%, with at least 300 tranches/year and roughly 25+/month?
2. Does one policy beat P0 total account return in both 2024 and 2025?
3. Is minute-marked MDD no worse than 20%?
4. Is 3x-cost return positive in both years and under 3/5-minute delay?
5. Does return remain positive after removing the top ten realized winners?
6. Does the selected policy keep dangerous and losing-position second entries low?
7. Does the maximum planned slot budget remain at or below 1R?

## Run

```bat
python research\eth_ai_trading\03_4_2_8b_dual_risk_slot_account.py
```

## Output

```text
data\reports\research\eth_ai_trading\03_4_2_8b_dual_risk_slot_account
```

Review `99_decision.md`, `05_account_policy_summary.csv`, `06_account_trades.csv`, `07_daily_equity.csv`, and `09_policy_gate.csv`.

## Next stage

- Pass: entry timing, executable structural stop and MAE optimization.
- Fail: do not add a third tranche or select policies by year; move to entry/stop refinement and separate complementary Sleeves.

## Validation completed

- New tests: `6 passed`.
- Related R03.4.2.7/8A/8B tests: `23 passed`.
- `tests/ai_research tests/data_feed`: `162 passed`.
- P0 event parity with the actual uploaded R03.4.2.8A report: exact match for both folds and all 1/3/5-minute delays.
- CLI import/help smoke: passed.
- Source-report pipeline smoke: completed with `BLOCKED_DATA` because this container does not contain the local Trade Bar dataset.
- Full `pytest -q`: collection blocked by five unrelated missing liquidity/analyze-tool modules in the supplied Sources archive.
- `tests/test_import_boundaries.py`: one existing failure from cross-research imports outside this stage.

No git commit was executed.
