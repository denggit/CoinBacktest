# R03.4.2.8A Delivery — Occupied Signal Atlas and Tranche Eligibility Gate

## Purpose

Determine whether q70 signals skipped by an existing `failed_reclaim` position contain one causal, cross-year healthy/recovered subset worth taking into a separate account-risk Tranche simulation.

## Run

```text
python research\eth_ai_trading\03_4_2_8a_occupied_signal_atlas.py
```

Optional outcome-cache rebuild:

```text
python research\eth_ai_trading\03_4_2_8a_occupied_signal_atlas.py --force-rebuild-outcomes
```

## Report directory

```text
data\reports\research\eth_ai_trading\03_4_2_8a_occupied_signal_atlas
```

Key outputs:

- `04_frozen_baseline_summary.csv`
- `05_occupancy_summary.csv`
- `06_occupied_signal_atlas.csv`
- `07_signal_class_summary.csv`
- `08_eligible_quarter_summary.csv`
- `10_score_price_diagnostic.csv`
- `11_candidate_risk_release_distribution.csv`
- `12_tranche_eligibility_gate.csv`
- `15_p0_failed_reclaim_trades.csv`
- `16_standalone_signal_outcomes.csv`
- `99_decision.md`
- `gpt_review_pack.zip`

## Hard boundaries

- q70 model unchanged.
- `failed_reclaim` parameters unchanged.
- 3% disaster floor unchanged.
- No P2/P3 position is opened.
- No fixed time exit is introduced as a final rule.
- Score increase alone never qualifies a signal.
- BROKEN / failed-reclaim-in-progress signals are rejected.
- Candidate protection is diagnostic and not treated as executed risk release.
- Same rule runs in 2024 and 2025.
- 2026 remains sealed.

## Decision outputs

- `PASS_TO_R03_4_2_8B_TRANCHE_SIMULATION`: only authorizes building real P2/P3 account-risk simulation.
- `FAIL_NO_CROSS_YEAR_OCCUPIED_SIGNAL_ELIGIBILITY`: stop Tranche research and move to entry-MAE refinement or an independent long-trend Sleeve.
- `BLOCKED_DATA`: public Loader/data prerequisites failed.

## Validation in this delivery

- New tests: 8 passed.
- New plus frozen R03.4.2.7 tests: 17 passed.
- All `tests/ai_research` plus `tests/data_feed`: 156 passed.
- Full repository collection remains blocked by five pre-existing missing liquidity/analyze-tool modules in the supplied Sources copy; none is imported or modified by this patch.
- The standalone import-boundary test also remains red because of pre-existing cross-research imports outside `eth_ai_trading`; this patch adds no `research -> research` import.
- Full-data empirical run: not possible in the supplied Sources copy because local Trade Bar/cache data are not included.
