# Cumulative Stage Results

## R00 — Causal state dataset

Status: **PASS_R00 on full local data; dataset layer frozen for R01.**

Purpose: establish one common causal data contract for all later supervised and offline-RL experiments. It intentionally trains no policy and makes no profitability claim.

Design decisions:

- monthly resumable shards;
- mmap-compatible NumPy cache rather than a mandatory Parquet dependency;
- explicit source-availability flags and coverage audit;
- hard failure on future-visibility violations or missing required core coverage;
- 2026 sealed-holdout flag embedded in each shard;
- forward labels include entry open, final return, long/short MFE, long/short MAE and path width at 15/30/60/180/360 minutes.

Full local R00 review passed: 44 shards, 380,942 decision rows, 347 features, 31 labels, zero causal-audit failures and zero required-source coverage failures. 2026 remains sealed.


## R00.1 — Range loader schema compatibility hotfix

Status: **FIXED, awaiting resumed local full-data run**.

Observed on the first real local R00 run: `OKXRangeBarLoader.load_local_data()` retains `end_ts` both as a column and as the index name. Pandas therefore rejected `sort_values(["end_ts", "bar_id"])` as ambiguous. This is a valid loader schema, not a data error.

Fix: the reusable feature-preparation layer now detaches only colliding incoming index labels before column-oriented normalization, preserving the explicit `end_ts` column and all causal availability semantics. Regression tests mirror the real loader schema.

Validation: RL tests 11/11 passed; data-feed tests 13/13 passed. No model, trading rule, label, time alignment, or source loader was changed. Existing R00 monthly shards, if any were completed before the crash, remain schema-compatible and can be resumed without `--overwrite`.

## R00.2 - partial official K-line coverage fallback

- Trigger: real local run reached 41/42 shards, then 2026-06 official K-line tables stopped around 2026-06-15~17 while mandatory tick-derived 1m trade bars continued.
- Root cause: R00 incorrectly treated every independent official K-line cache as a hard-required source with no fallback, so a stale duplicate cache blocked the entire sealed-holdout month.
- Fix: official K-lines remain preferred. Missing fixed-bar timestamps are filled only from the mandatory local 1m tick-derived trade-bar cache via causal OHLCV resampling. No network download, DB rebuild, strategy logic, label logic, or execution logic was added.
- Causality: fallback bars remain left-labeled and are aligned by `bar_start + timeframe`; incomplete resample buckets are discarded. Daily fallback uses the project's UTC+8 local K-line anchor (08:00).
- Auditability: per-shard K-line coverage notes report `official_kline_plus_trade1m_fallback_rows=<N>` when fallback was used.
- Tests: RL market-agent 14/14 passed; `tests/data_feed` 13/13 passed; py_compile passed.


## R00.3 - data refresh through 2026-08-15

- Local data boundary extended to 2026-08-15 23:59:59.
- Sealed holdout remains 2026-01-01 onward; July/August are holdout extension, not training data.
- Active K-line path changed from independent HTF caches / R00.2 fallback to official 1m K-line -> causal 5m/15m/1H/4H/1D resampling.
- Tick-derived 1m/5s trade bars remain separate microstructure sources.
- Final max-horizon (360m) raw-data tail is label-only context, preventing incomplete forward labels at dataset end.
- A new `r00_3` cache root prevents mixing old R00/R00.2 shards with the new K-line semantics.
- Targeted validation: RL + data_feed 28/28 passed on the full CoinBacktest baseline.

## R00.4 - Pandas daily-anchor compatibility hotfix

- Real Windows/Pandas run exposed that `resample(..., freq='1D', offset='8h')` is not portable: the user's Pandas ignores the offset and warns.
- Active 1D construction now uses explicit shift-resample-unshift, preserving the +08:00 daily start across Pandas versions.
- Dataset revision/cache root bumped to `R00.4` / `r00_4`; R00.3 shards must not be reused.
- No labels, market-state definitions other than the corrected daily timestamp anchor, source loaders, or trading logic were changed.


## R01 — Opportunity model -> executable strategy walk-forward

Status: **IMPLEMENTED; awaiting local full-data run.**

Purpose: stop treating predictive edge as the deliverable. R01 trains a fixed LightGBM opportunity baseline on concrete trade-template net returns and immediately converts scores into one non-overlapping ETH strategy sleeve with real 1m TP/SL replay, position sizing and cost stress.

Leakage hardening:

- whole-shard `iter_training_shards()` is blocked because month boundaries are not forward-label-safe;
- all train/calibration/OOS windows use horizon-aware purge (`label_end < right_boundary`);
- this closes the 2025-12 -> 2026 label leak as well as train->calibration and prior-year->next-year fold edges;
- 2026 shards are never opened in R01.

Frozen first-pass strategy templates:

- H60: TP 0.60%, SL 0.40%;
- H180: TP 1.00%, SL 0.60%;
- H360: TP 1.50%, SL 0.80%.

Training targets are conservative template returns after the 0.11% base round-trip cost. If MFE and MAE imply both TP and SL were touched within the R00 horizon but order is unknown, the training target assumes the stop. Calibration/OOS strategy replay then resolves first-hit order on the actual local 1m trade-bar path; same-minute TP+SL remains adverse-first.

Walk-forward:

- WF_2024: train 2023-01..Sep, calibrate 2023-Q4, OOS 2024;
- WF_2025: train 2023-01..2024-Sep, calibrate 2024-Q4, OOS 2025;
- templates, feature family and score threshold are selected only on the pre-OOS calibration window;
- 2026 remains unopened.

Champion selection is applied only after basic profitability/risk feasibility gates, then uses the user-frozen lexicographic priority: max flat days -> max consecutive losing days -> MDD -> CAGR -> total return. Base/2x/3x cost results are mandatory.
