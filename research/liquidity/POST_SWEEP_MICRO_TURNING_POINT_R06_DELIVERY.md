# R06 Post-Sweep Micro Turning-Point Study

## Goal

Compare, within the same Swing Liquidity Zone event:

1. the final durable turning new-low attempt;
2. the immediately preceding failed new-low attempt;
3. continuation controls that keep falling.

The goal is to find a causal entry trigger that waits for selling impact to fail, lowers post-entry MAE, retains enough MFE after costs, and avoids blindly buying an ongoing crash.

R06 is mechanism research, not a final strategy backtest.

## Run

```bat
python research\liquidity\06_post_sweep_micro_turning_point_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59"
```

Requirements:

- the full R04 report is present at `data\reports\research\liquidity\post_sweep_continuation_exhaustion_r04`;
- Binance metrics have already been downloaded to `data\binance_futures_metrics.db`;
- local OKX raw trade ZIPs and Range-Bar cache are available.

R06 reads Binance OI directly from the indexed SQLite database. It does not scan the 700k-row compressed R05 checkpoint export unless `--oi-source r05` is explicitly selected.

To run without OI deliberately:

```bat
python research\liquidity\06_post_sweep_micro_turning_point_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --oi-source none
```

Missing raw trade days are not downloaded automatically. To allow official OKX raw-trade downloads:

```bat
python research\liquidity\06_post_sweep_micro_turning_point_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --allow-download-missing-raw
```

## Sparse 1-second design

R06 does not materialize a complete multi-year 1-second database. It:

- identifies only the selected attempt windows;
- reads each required raw UTC day once;
- retains only event-local windows;
- aggregates those slices to 1-second Trade Bars;
- releases each daily raw batch before moving to the next day.

This keeps memory and disk use bounded while preserving the original trade sequence.

## Causal timing

- A 1-second bar is available only at `timestamp + 1 second`.
- Every candidate signal uses a completed 1-second bar.
- Entry is the next 1-second bar open.
- `ORACLE_LOW_PLUS_1S` is explicitly future-labelled and excluded from the deployable candidate scorecard.
- Binance 5-minute metrics use a 1-minute publication lag and backward causal as-of alignment.
- Future MFE, MAE, close returns and first-passage results are stored as labels, not features.

## Predeclared candidate triggers

- `FIRST_NEW_LOW`: causal early baseline;
- `IMPACT_COLLAPSE_67`;
- `IMPACT_COLLAPSE_50`;
- `IMPACT_COLLAPSE_50_HIGH_BREAK`;
- `MICRO_RECLAIM_5S`;
- `MINUTE_CLOSE`: later causal baseline;
- `ORACLE_LOW_PLUS_1S`: non-deployable upper bound.

The 0.50/0.67 variants are a small natural neighbourhood, not a fitted parameter grid.

## Range-Bar interpretation boundary

R06 reports whether the final and previous attempts reference the same completed Range bar. This matters because attempts are often only one or two minutes apart. A shorter `first_up_delay_seconds` is not independent evidence when both attempts point to the same later Range bar.

The full Range-only audit found substantial paired overlap, increasing with Range size. Therefore Range Bars are treated as process context; the exact entry decision must come primarily from 1-second Trade Bars and later Footprint/Books confirmation.

## Main output

```text
data\reports\research\liquidity\post_sweep_micro_turning_point_r06
```

Important files:

```text
04_micro_oracle_vs_prior_paired_profile.csv
06_trigger_path_summary.csv
07_trigger_relative_to_baselines.csv
08_range_pair_overlap_summary.csv
09_range_oracle_vs_prior_paired_profile.csv
10_candidate_scorecard.csv
11_causal_audit.csv
12_oracle_prior_pair_audit.csv
13_raw_window_coverage.csv
14_micro_window_audit.csv
19_attempt_feature_table.csv.gz
20_attempt_label_table.csv.gz
21_micro_window_feature_table.csv.gz
22_trigger_path_table.csv.gz
23_range_feature_table.csv.gz
24_research_brief.md
gpt_review_pack.zip
```

## Promotion boundary

A candidate is not promoted merely because it separates future-labelled oracle turns from prior failed attempts. It must also:

- separate oracle, prior-failed and continuation-control cohorts in every period;
- materially reduce MAE relative to `FIRST_NEW_LOW` and `MINUTE_CLOSE`;
- preserve enough MFE after the default 0.11% round-trip cost;
- remain stable across the predeclared threshold neighbourhood;
- pass a separate walk-forward strategy backtest before any risk amplification.
