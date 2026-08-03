# ETH AI Trading R03.4.2.4 Runbook

## Purpose

Confirm whether the frozen q70 opening pool robustly improves total cost-adjusted profit over q90 in both WF_2024 and WF_2025. The six-hour close is retained only as the original opening-edge benchmark; it is not the final live exit.

## Run

```bat
python research\eth_ai_trading\03_4_2_4_q70_cross_year_audit.py
```

The script reuses the existing R03.2 feature caches and state-context outcome caches. It loads one-minute Trade Bars through `src.data_feed` in bounded monthly chunks.

## Outputs

```text
data\reports\research\eth_ai_trading\03_4_2_4_q70_cross_year_audit
```

Review these first:

```text
04_policy_summary.csv
05_period_summary.csv
06_q70_score_band_summary.csv
08_q70_vs_q90_comparison.csv
10_stable_candidate.csv
99_decision.md
gpt_review_pack.zip
```

## Interpretation

- `broad_q70` is the full q70 event pool.
- `primary_q90` is the independently constructed q90 event pool.
- `q70_to_q90` isolates q70 events whose calibrated score percentile is below q90; this is the true incremental opportunity band.
- A stable q70 expansion requires positive 2x-cost expectancy and PF in both years, a positive incremental band, positive 3x-cost and 5-minute-delay stress, acceptable MDD/concentration, and higher total compounded return than q90 in both years.

## 2024 repair

R03.4.2.3 skipped the entire WF_2024 policy because the recoverable-drawdown classifier had too few eligible training rows. R03.4.2.4 removes that dependency completely. Any remaining missing WF_2024 result is therefore a genuine data or execution blocker and is written to `12_failures.csv` rather than silently omitted.

## Next step

After q70 passes, build a conservative persistent-failure overlay that can be evaluated without a recoverable-drawdown classifier. Selective long holding should then predict incremental value of continuing from the current checkpoint, not an abstract long-hold class and not an opening score threshold.
