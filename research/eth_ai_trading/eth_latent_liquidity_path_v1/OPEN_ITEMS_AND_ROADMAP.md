# Open Items and Roadmap

## Immediate — R02.4 Economic Ceiling Audit

1. Reuse completed R01.1 future path labels; do not rebuild historical 1s data.
2. Stream source tables in chunks and retain one first representative per release episode.
3. Train no model and add no Range/Footprint/OI/Books.
4. Treat all future-informed fields as explicit oracle-only diagnostics.
5. Measure 60/180/300/600-second favorable excursion, adverse excursion and reward/risk.
6. Measure perfect-exit net-MFE ceiling under 6/8/11/22/33bp costs.
7. Measure fixed 1R/1.5R/2R realizations using future-MAE + 3bp oracle stop and horizon close when target is not reached.
8. Report ALL_RELEASE, FAVORABLE_REVERSAL_ORACLE, frozen R01 reversal clusters 10/4/5 and continuation control 8 separately.
9. Require Validation and Holdout to pass the frozen 11bp + 22bp favorable-oracle economic ceiling gate before any identification research continues.
10. Append the full real R02.4 result to `CUMULATIVE_STAGE_RESULTS.md` before opening another stage.

## If R02.4 economic ceiling fails

Stop the latent-liquidity reversal branch.

- no nuisance-regime rescue;
- no PATH model rescue;
- no Range/Footprint/OI/Books rescue;
- no execution parameter search.

Archive the retained descriptive findings and move research budget to a different economic mechanism.

## If R02.4 economic ceiling passes

The problem is confirmed to be **identification**, not lack of payoff room.

Only then decide which missing information is most likely to identify the profitable oracle subset. Prefer independent information increments over more transformations of the same price/trade target:

1. test whether existing causal 1s/Range path can identify the oracle-rich episodes;
2. Footprint as the next microstructure increment;
3. OI/Funding if coverage and timing are reliable;
4. Books last because coverage is shorter and data volume is much larger.

Each new data family must beat the same frozen baseline and may not redefine the economic target after seeing results.

## Closed / prohibited

- no R01.3 confirmation/stop/threshold rescue;
- no R02.1 absolute cumulative-strength rescue;
- no R02.2 raw-density rescue;
- no R02.3 median/IQR rescue;
- no R02.3.1 / R02.3.1b target-loss tuning loop;
- no Swing-centered rescue;
- no new data family before the economic ceiling is known;
- no oracle field may appear in a causal predictor or live strategy;
- no live deployment from development holdout evidence.
