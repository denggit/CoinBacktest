# R03.4.2.17 Report Logic Hotfix3

The third local run completed successfully and produced valid raw state tables. A post-run audit found three reporting-layer issues that did not alter frozen trades, scores, states or the final `DIAGNOSIS_SCORE_DRIFT_DOMINANT` classification:

1. the `h1_regime_separation` detail string was static and could contradict a `False` result;
2. the monthly table labeled entry-month cycle cohorts as calendar account returns;
3. 2026 source cycles lacked `full_mae`; account-level `cycle_max_drawdown` must not be mislabeled as trade-path MAE, so the two metrics are now reported separately.

Hotfix3 corrects all three and adds a separate distinction between a gate that merely stays positive and a gate that beats frozen `G0_ALL` in every opened period. No trade rule, model, q70 threshold, state definition or opened-period result is changed.
