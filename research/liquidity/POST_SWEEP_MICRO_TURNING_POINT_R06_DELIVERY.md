# R06 Fix2 — UTC+8 Raw Archive Partition Repair

## Root cause

CoinBacktest strategy timestamps use naive UTC+8 project time. The OKX raw ZIP
filename date also represents the UTC+8 local calendar day, while each trade row
stores Unix UTC milliseconds.

The previous R06 converted the event to UTC first and then used that UTC date to
select the ZIP. Therefore local 00:00–07:59 events selected the previous local-day
ZIP. The market data was continuous; only the file-partition lookup was wrong.

## Fix2 behavior

1. Select the ZIP using `start_time.date()` in project time.
2. Convert the event boundaries to UTC milliseconds only for filtering rows.
3. If a window crosses project-local midnight, load and join both adjacent ZIPs.
4. Produce `01b_raw_hourly_coverage.csv`.
5. Fail before Range/report interpretation if a fixed local hour has systematic
   no-trade windows or if archive-day mapping disagrees with `start_time.date()`.

## Validation run

First run the full micro phase without Range Bars:

```bat
python research\liquidity\06_post_sweep_micro_turning_point_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --skip-range --skip-review-pack
```

Required output:

```text
[micro-gate] PASS
hourly_failed_hours=[]
archive_partition_mismatches=0
legacy_cross_utc_day_windows=0
```

Then run the full study:

```bat
python research\liquidity\06_post_sweep_micro_turning_point_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59"
```

## Research stop rule

The existing partial-day R06 suggests that the simple one-shot micro triggers
improve 180-second MAE by only roughly 0–2 bp while losing a similar amount of
MFE, and trigger at least as often during continuation crashes. Fix2 is the final
all-day verification. If the complete sample retains that result, stop the
simple micro-entry branch rather than stacking filters and shrinking event count.
