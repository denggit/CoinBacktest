# Post-Sweep Continuation, Exhaustion and Reversal Atlas R04

## Purpose

R04 does **not** enter immediately after the first Swing Liquidity Zone sweep.
It follows the post-sweep process and studies when downside continuation remains
effective, when negative Delta/CVD stop producing equivalent price progress,
and which causal confirmation state preserves future MFE while reducing future
MAE.

Large future-MFE events are extracted for feature discovery, but all future
outcomes remain labels and never become admission filters in this version.

## Data

Required primary data:

- `OKXTradeBarLoader`
- ETH-USDT-SWAP 1m trade bars
- `buy_notional`, `sell_notional`, `delta_notional`, `notional`

Optional large-trade columns are used when present.

R04 reuses the R03 causal Zone event table by default:

```text
data/reports/research/liquidity/swing_liquidity_zone_sweep_mechanism_r03/04_online_first_zone_feature_table.csv
```

If it is absent, R04 rebuilds the causal Swing Low universe and Zone events from
shared `src.research_common` modules. It never imports another research script.

## Default Windows command

```bat
python research\liquidity\04_post_sweep_continuation_exhaustion_atlas.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --no-build-missing
```

Force Zone reconstruction instead of reusing R03:

```bat
python research\liquidity\04_post_sweep_continuation_exhaustion_atlas.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --no-build-missing --rebuild-zone-events
```

## Causal sequence

```text
closed Swing Zone sweep bar
-> closed post-sweep 1m checkpoint
-> causal flow / price-response features
-> next 1m open begins the future label path
```

Dense checkpoints are kept for the first 30 minutes, then at fixed later times.
Every additional new-low attempt is retained even outside the fixed schedule.

## Main features

- Buy/sell notional and Delta over 1/3/5/15/30 minutes.
- Cumulative Delta from the sweep.
- CVD new low without a simultaneous price new low.
- Price movement per USD 1m sell notional.
- Price movement per USD 1m negative Delta.
- Repeated new-low attempt number and extension shrinkage.
- Bars since the latest new-low attempt.
- Micro-high breaks and Zone floor/ceiling reclaim.
- Large-trade Delta when present.

Absolute bp movement is retained as one description, but it is not the only
mechanism variable.

## Future-only labels

- MFE, MAE and close return over 5/15/30/60/180 minutes.
- Whether a lower low occurs after a checkpoint.
- Reversal-dominant versus continuation-dominant path labels.
- Fixed large-MFE labels: 0.5%, 1% and 2%.
- Future-labelled oracle turning point for descriptive feature extraction.

The oracle table is explicitly non-tradable because its row selection uses
future outcomes.

## Outputs

```text
01_data_quality.csv
02_new_low_attempt_summary.csv
03_checkpoint_path_summary.csv
04_confirmation_state_summary.csv
05_orderflow_fixed_bin_summary.csv
06_confirmation_period_stability.csv
07_large_mfe_summary.csv
08_large_mfe_feature_profile.csv
09_oracle_turning_point_sample.csv
10_large_reversal_opportunity_sample.csv
11_static_zone_event_features.csv
12_causal_audit.csv
13_checkpoint_feature_table.csv.gz
14_checkpoint_label_table.csv.gz
15_checkpoint_sample.csv
16_research_brief.md
gpt_review_pack.zip
```

The full feature and label tables are gzip-compressed and intentionally skipped
from the small GPT review pack. Summaries and samples are included.

## Overfitting controls

- No model fitting.
- No threshold chosen by maximum return.
- No irregular quantile edge promoted into a strategy rule.
- Fixed time periods: 2023-2024, 2025Q1-Q3, 2025Q4-2026H1.
- Fixed, rounded large-MFE thresholds.
- Feature and label tables physically separated.
- The holdout period cannot modify event admission.

## Validation performed

- R04 unit tests.
- R02/R03/R04 causal regression tests.
- Review-pack and speed-path regression tests.
- Synthetic database end-to-end main-script smoke test.
- R03 event-table reuse path smoke test.
- Microsecond timestamp test.
- FutureWarning-as-error self-test and smoke test.
