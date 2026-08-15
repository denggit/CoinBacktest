# ETH 1m Multi-timeframe Liquidity Sweep, MSS and FVG Limit Study

## Technical summary

**Result: NO ROBUST EDGE CONFIRMED BY THE PREDECLARED GATE.** The 2023 discovery winner is **V4_strict**.  The validation gate is **not passed** after 15bp round-trip cost and conservative same-bar handling.

This is a descriptive backtest of a fully mechanical interpretation of the requested ICT sequence. It does not establish that discretionary ICT labels or live execution will produce the same result.

## Key findings with visual evidence

The primary specification holds a 60-minute FVG limit order, targets 1.5R, exits remaining positions after six hours, and deducts 15bp round-trip cost. Variants were frozen before outcomes were inspected; the winner is selected only from 2023, then checked in 2024-2026.

| year | n_placed | n_filled | fill_rate | win_rate_15bp | avg_net_r_15bp | profit_factor_15bp | t_stat_15bp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2023 | 983 | 902 | 0.918 | 0.369 | -0.783 | 0.273 | -11.351 |
| 2024 | 693 | 628 | 0.906 | 0.396 | -0.446 | 0.470 | -9.053 |
| 2025 | 816 | 758 | 0.929 | 0.414 | -0.343 | 0.555 | -7.844 |
| 2026 | 552 | 503 | 0.911 | 0.376 | -0.496 | 0.427 | -8.983 |

## Scope, data, and metric definitions

Only `open`, `high`, `low`, and `close` from `src.data_feed.OKXTradeBarLoader` were used. The query requested 2022-01-01 through 2026-12-31; cached coverage ended at **2026-07-31 23:59**. 2022 is warm-up only; orders activated from 2023 onward are measured.

| year | bars | first_bar | last_bar | gaps_gt_1m | largest_gap_minutes |
| --- | --- | --- | --- | --- | --- |
| 2022 | 524127 | 2022-01-01 | 2022-12-31 23:59:00 | 59 | 612.000 |
| 2023 | 525536 | 2023-01-01 | 2023-12-31 23:59:00 | 13 | 35.000 |
| 2024 | 527040 | 2024-01-01 | 2024-12-31 23:59:00 | 0 | 1.000 |
| 2025 | 525600 | 2025-01-01 | 2025-12-31 23:59:00 | 0 | 1.000 |
| 2026 | 305280 | 2026-01-01 | 2026-07-31 23:59:00 | 0 | 1.000 |

A macro liquidity level is a 15m, 30m, 1h, or 4h swing extreme, resampled strictly from the same 1m series. The strategy only sees it after the right-side swing window completes. A low/high is consumed on its first strict cross; simultaneous high-and-low sweeps are excluded as path-ambiguous.

## Methodology

1. Scan the causal first sweep of active macro swing levels.
2. Within 60 minutes, require a countertrend 1m structure break of a previously confirmed micro swing plus a displacement candle (body/previous 14-bar ATR and body/range filters).
3. On the next candle, require a three-candle FVG. Enter at that candle's low for a long or high for a short, with eligibility delayed one more 1m bar.
4. Stop at the running sweep extreme through FVG confirmation. Model first touch of the resting limit; for a candle touching both stop and target, record the stop first.
5. Report 30/60/180-minute order-life and 1/1.5/2R target sensitivity. Cost columns use 11bp, 15bp, and 20bp round-trip deductions.

Candidate count across the four frozen structural variants: **18,841**. The full event audit is `02_candidate_events.csv.gz`; individual replay results are `04_trade_replay.csv.gz`.

## Limitations, uncertainty, and robustness checks

| selected_variant | gate | passed | detail |
| --- | --- | --- | --- |
| V4_strict | 2023 discovery | False | positive average net R after 15bp costs; >=25 fills |
| V4_strict | OOS sample | True | 2024-2026 filled trades=1889 |
| V4_strict | OOS pooled expectancy | False | weighted net R=-0.418 |
| V4_strict | Full-year stability | False | positive among 2024 and 2025=0/2 |
| V4_strict | 2026 YTD guardrail | False | not worse than -0.10R after 15bp costs |

The data are 1m OHLC, so intra-candle order is not observable; same-bar ambiguity is conservatively adverse. The model does not include spread, queue priority, adverse selection, partial fills, funding, liquidation, or exchange outages. Minute gaps remain visible in the coverage table and no higher-timeframe bar spanning a gap is used as a swing pivot. Parameter sensitivity is descriptive, not a license to select an in-sample optimum.

## Recommended next steps

If the gate passed, forward-test the exact frozen selected variant with order-book/queue-aware fills and a strictly out-of-sample paper-trading period. If it did not pass, do not promote the unfiltered MSS/FVG pattern to a strategy; any follow-up should add one economically motivated filter at a time and keep a final untouched holdout.

## Further questions

Does an executable FVG fill differ materially from the OHLC first-touch assumption? Do macro swing quality, session, or volatility regime improve the result out of sample rather than merely re-ranking in-sample variants? These questions require a new frozen experiment rather than changing this study's gate.
