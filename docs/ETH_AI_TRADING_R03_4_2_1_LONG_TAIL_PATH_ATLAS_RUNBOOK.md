# ETH AI Trading R03.4.2.1 — q90 Long Event Complete Path Atlas

## Goal

Understand how the frozen q90 long edge actually develops before testing another exit mechanism.

This stage does **not** optimize a stop, take-profit, trailing rule, renewal rule, or holding period. It creates a one-by-one path atlas for every complete q90 event and asks:

- Does the event rise immediately or first suffer a deep drawdown?
- How long does it remain underwater before recovery?
- Does it spike early and then give back profit?
- Does it grind upward slowly?
- Does the main move occur after the original six-hour horizon?
- Do failed events show causal base-score decay before price failure?
- Are the same path families present in both 2024 and 2025?

All winners and losers are included. Looking only at successful trades would create survivor bias.

## Frozen entry model

- Target: `long_utility_h6 = long_mfe_h6 - 1.25 * long_mae_h6`
- LightGBM objective: `regression_l1`
- `n_estimators=420`
- `learning_rate=0.035`
- `num_leaves=31`
- `min_child_samples=300`
- `train_sample_cap=400000`
- `random_state=20260801`
- Entry signal: prior-quarter calibrated q90
- Entry price: next one-minute open
- Dense signals: same episode merging and six-hour independent-event cooldown as R03.4.2

The market-state model remains formally abandoned for trading and is not loaded.

## Research split

### WF_2024

- Fit base model: embargoed 2023 fit period
- Discovery paths: 2023Q4 calibration events
- OOS path audit: 2024

### WF_2025

- Fit base model: embargoed 2023 through 2024Q3 fit period
- Discovery path pool: all calibration paths available through 2024Q4
- OOS path audit: 2025

The path types use future prices as historical labels. They are not live features. OOS paths are not used to redesign the labels or cluster structure.

## Complete path extraction

Each accepted event requires exactly 2,880 uninterrupted one-minute bars after entry, covering 48 hours. Events with missing minutes or insufficient year-end path are excluded.

The large per-minute files contain:

- one-minute OHLC returns from entry
- running MFE and MAE
- drawdown from the running peak
- latest base-model score available at or before each minute
- score percentile relative to the prior calibration quarter

Files are stored as gzip CSV shards under:

`data/reports/research/eth_ai_trading/03_4_2_1_long_tail_path_atlas/event_paths`

They are intentionally excluded from the GPT review pack because of their size.

## Per-event path features

At 5, 15, 30, 60, 120, 180, 360, 720, 1,440, and 2,880 minutes:

- close return
- MFE and MAE
- time to MFE and MAE
- peak giveback
- underwater share and longest underwater run
- directional path efficiency

Additional fields include:

- time to first +0.5%, +1%, +1.5%, +2%, and +3%
- time to first -0.5%, -1%, -1.5%, -2%, and -3%
- MAE before first +1%
- six-hour fixed diagnostic net return
- post-six-hour MFE and close-return increments
- best historical fixed close among 1h, 3h, 6h, 12h, 24h, and 48h
- score percentile decay, q90 reconfirmations, and first q70/q50 loss

The oracle fields are historical diagnostics only and cannot be used as strategy results.

## Path types

### Fixed semantic labels

The preregistered labels include:

- `immediate_clean_winner`
- `early_spike_giveback_winner`
- `delayed_recovery_winner`
- `slow_grind_winner`
- `other_6h_winner`
- `late_rescue_after_6h`
- `persistent_failure`
- `volatile_giveback_failure`
- `other_6h_failure`

Independent flags preserve overlapping behavior such as deep MAE, post-six-hour continuation, score reconfirmation, and score decay.

These labels describe completed paths; they are not entry or exit rules.

### Discovery-only clusters

A six-cluster KMeans atlas is fit on robust-scaled path summaries from prior calibration periods only. The cluster model is then frozen and applied to the next OOS year.

The fixed cluster feature set includes early and late returns, MFE/MAE, time to peak, giveback, underwater share, post-six-hour continuation, and causal score evolution.

## Run

```text
python research\eth_ai_trading\03_4_2_1_long_tail_path_atlas.py
```

Only rebuild the reusable R03.4 outcome cache when necessary:

```text
python research\eth_ai_trading\03_4_2_1_long_tail_path_atlas.py --force-rebuild-outcomes
```

## Reports

`data\reports\research\eth_ai_trading\03_4_2_1_long_tail_path_atlas`

Important files:

- `03_event_extraction_audit.csv`
- `04_discovery_path_features.csv`
- `05_oos_path_features.csv`
- `06_path_type_assignments.csv`
- `07_path_type_summary.csv`
- `08_path_type_by_quarter.csv`
- `09_target_hit_timing.csv`
- `10_winner_loser_path_contrast.csv`
- `11_representative_events.csv`
- `12_discovery_cluster_centroids.csv`
- `13_oos_cluster_summary.csv`
- `14_oracle_exit_envelope.csv`
- `15_score_path_bins.csv`
- `16_event_path_files.json`
- `99_decision.md`
- `gpt_review_pack.zip`

## Next-stage rule

R03.4.2.2 may begin only after the atlas identifies path families that:

- exist in both 2024 and 2025
- have sufficient samples
- show materially different path and exit behavior
- can plausibly be distinguished using information available early in the trade

The next stage must predict the developing path type from causal early features. It must not use the completed path label directly in backtests.
