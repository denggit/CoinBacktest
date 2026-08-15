# R01.1 Patch Manifest — Liquidity-first Path Atlas

## Stage

```text
ETH Latent Liquidity Pool Path Learning V1
R01.1 Liquidity-first release and reversal path atlas
```

R01.1 supersedes the Swing-centered interpretation of R01. It does not define a liquidity pool from a Swing and does not use a Swing as a candidate gate.

## Core implementation

```text
src/ai_research/latent_liquidity_path_atlas/config.py
src/ai_research/latent_liquidity_path_atlas/candidates.py
src/ai_research/latent_liquidity_path_atlas/features.py
src/ai_research/latent_liquidity_path_atlas/macro.py
src/ai_research/latent_liquidity_path_atlas/outcomes.py
src/ai_research/latent_liquidity_path_atlas/clustering.py
src/ai_research/latent_liquidity_path_atlas/reports.py
src/ai_research/latent_liquidity_path_atlas/pipeline.py
src/ai_research/latent_liquidity_path_atlas/time_axis.py
src/ai_research/latent_liquidity_path_atlas/unswept_swings.py
```

## Entrypoints

Primary R01.1 entry:

```text
research/eth_ai_trading/eth_latent_liquidity_path_v1/01_1_liquidity_first_path_atlas.py
```

R01 compatibility entry retained:

```text
research/eth_ai_trading/eth_latent_liquidity_path_v1/01_release_reversal_path_atlas.py
```

## Liquidity-first discovery space

Candidate admission is the broad union of:

- trade-notional / trade-count / directional-flow release;
- 1s/5s/15s price shock;
- range expansion;
- release through rolling price boundaries.

Pre-event model features prioritize:

- turnover per traveled/ranged price;
- directional pressure without price progress;
- impact efficiency and travel efficiency;
- price overlap/residency and compression/expansion;
- multi-window price, flow, Delta, trade-count and large-trade paths;
- release Episode structure and episode-aware sample weight.

No Swing field is used to admit a candidate.

## Swing inventory boundary

Only causally confirmed structures at these timeframes are allowed:

```text
15m / 30m / 1H / 4H / 1D
```

For every timeframe and side, R01.1 retains every confirmed level until the first true sweep:

- no nearest-only reduction;
- no arbitrary maximum age;
- old levels remain available to the model;
- consumed levels disappear only after their sweep is causally observable;
- future/unconfirmed levels are invisible;
- counts, distance distribution, age distribution and multi-timeframe confluence are features only.

All sub-15m Swing/Pivot features are prohibited by configuration and causal audit.

## Episode correction

Continuous same-side release impulses are grouped into one Liquidity Release Episode. DOWN and UP maintain independent active Episode state, so interleaved opposite-side events cannot be merged into one Episode. Episode size and inverse-size weight prevent tens of overlapping 1s triggers from being treated as independent evidence.

## Reports

```text
00_manifest.json
01_data_quality.csv
02_candidate_source_summary.csv
03_release_episode_summary.csv
04_outcome_type_summary.csv
05_path_cluster_summary.csv
06_period_stability.csv
07_liquidity_feature_family_summary.csv
08_unswept_swing_inventory_summary.csv
09_causal_audit.csv
10_swing_level_lifecycle_summary.csv
11_event_sample.csv
12_feature_table.csv.gz
13_label_table.csv.gz
14_cluster_assignment.csv.gz
15_all_15m_plus_swing_lifecycle.csv.gz
16_research_brief.md
```

## Tests

```text
tests/ai_research/test_latent_liquidity_path_atlas.py
```

Coverage includes:

- no sub-15m Swing configuration;
- broad candidate admission without Swing;
- absence of micro Swing features;
- all unswept 15m+ levels, including very old levels;
- consumed/future levels hidden;
- liquidity-first feature contract;
- release second excluded from latent-pool features;
- DOWN/UP Episode isolation;
- causal completed-1m macro alignment;
- mixed `datetime64[us]` / `datetime64[ns]` safety;
- clean failure for empty or macro-incomplete research windows.

## Frozen limits

R01.1 is a discovery atlas only. It does not claim to observe private stop orders and does not optimize entry, stop, leverage, TP, account return or trading costs. Range Bar, Footprint, OI/Funding, liquidation and Books are later incremental evidence layers.

## Full-history memory / resume hotfix — 2026-08-05

Observed failure after all 639 chunks completed:

```text
rows=2,431,174
wide float columns=384
failed allocation=6.96 GiB
location=_assign_global_release_episodes -> sort_values(...).reset_index(...).copy()
```

The hotfix changes execution only:

- compact every small chunk before retention (`float64 -> float32`, safe integer downcast, block consolidation);
- filter incomplete future-label rows inside each chunk, avoiding a later full-width intersection copy;
- preserve chronological chunk order and reject unexpected cross-chunk duplicates;
- rebuild cross-chunk DOWN/UP Episodes from two narrow arrays and attach metadata without a deep full-frame copy;
- bound frozen discovery-cluster fitting to a deterministic stratified sample of at most 250,000 eligible pre-2025 rows;
- assign clusters to the complete dataset in 50,000-row batches;
- use a narrow joined frame for outcome/Episode/cluster summaries;
- use a fixed stratified sample for descriptive feature-family medians and disclose population/sample counts;
- stream large gzip CSV outputs in 50,000-row chunks with progress;
- cache every completed two-day feature/label chunk under a configuration signature and reuse it on rerun;
- expose `--no-chunk-cache` only for users who explicitly do not want resumability.

No research semantics changed: candidate union, path windows, causal timing, outcome labels, 15m+ all-unswept Swing inventory, Episode gap, cluster count, and train cutoff remain frozen.
