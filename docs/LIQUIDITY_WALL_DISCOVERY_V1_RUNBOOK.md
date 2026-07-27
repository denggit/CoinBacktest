# Liquidity Wall Discovery V1 Runbook

## Purpose

This research corrects the previous assumption that the existing wall detector
or V1 liquidity bands are already correct. It runs before strategy design.

The research asks two linked questions:

1. Which causal order-book structures behave like real liquidity walls?
2. Under which already-visible market environments does a touch bounce, versus
   break with volume and continue through the wall?

It does not optimize entry, stop loss, take profit, leverage, or fees.

## Causal sequence

```text
completed 5s Books snapshot
-> broad multi-scale candidate extraction
-> cross-snapshot lifecycle tracking
-> state available at bucket_end
-> next 5s execution bar may touch the state
-> future path labels bounce/break outcomes
```

A Books state produced after a touch bar starts cannot explain that touch.
Future outcomes never change candidate boundaries, morphology, or lifecycle.

## Candidate vocabulary

The extractor keeps continuous features instead of immediately claiming a
binary wall label:

- point concentration;
- wide band mass;
- composite bands with small internal holes;
- local median multiple;
- causal 24h scale ratio;
- spatial contrast;
- support/thick/non-zero occupancy;
- persistence and current retention;
- center drift and ghost score;
- recent fade slope;
- cancellation, consumption, and replenishment flows.

The causal 24h reference is only one scale feature. It is not used as the sole
wall definition or as visual ground truth.

## Symmetric outcomes

For each touch, the same frozen wall state is used to study:

```text
BOUNCE
BREAK
AMBIGUOUS (both thresholds in one 5s bar)
NEITHER
```

A break is separately marked `volume_confirmed_break` when the first break bar
has abnormal notional and directional imbalance. Continuation beyond the wall
is then measured.

## Environment groups

The report compares Train and Holdout using Train-defined quantile edges for:

- wall width, mass, occupancy, contrast and morphology;
- age, coverage, retention, fade and drift;
- first touch versus repeated touch;
- distance from market;
- 30s/2m/5m/15m approach return;
- pre-touch realized volatility;
- pre-touch sell imbalance and large-sell share;
- touch penetration, reclaim position, notional and imbalance;
- cancellation, consumption and replenishment.

Holdout does not define feature bins or select candidate boundaries.

## Visual acceptance

After a bounded research run, choose the Analyze Tool plugin:

```text
流动性墙发现 V1（研究覆盖层）
```

It reads:

```text
data/reports/research/liquidity/liquidity_wall_discovery_v1/13_wall_overlay_segments.csv
```

The overlay is separate from the existing order-book heatmap plugin and is
explicitly marked research-only. Compare it with the raw heatmap, especially:

- 2026-06-01 21:45, Bid 1800-1806;
- 2026-01-26 04:00;
- isolated deep-price lines such as 1850/1877 cases.

Acceptance requires:

- wide persistent areas are not fragmented into many rectangles;
- shallow areas do not become walls by themselves;
- isolated point lines remain distinct from main bands;
- drifting quote-following structures are marked ghost;
- removed/consumed areas disappear;
- replenished areas regain retention;
- known visual cases are represented plausibly.

## Output files

```text
00_manifest.json
01_candidate_audit_sample.csv
02_wall_track_summaries.csv
03_touch_events.csv
04_daily_discovery_counts.csv
05_environment_feature_uplift.csv
06_wall_shape_outcomes.csv
07_monthly_outcomes.csv
08_causal_replay_audit.csv
09_known_case_candidate_rows.csv
10_known_case_state_rows.csv
11_known_case_audit.csv
12_decision.md
13_wall_overlay_segments.csv
gpt_review_pack.zip
```

## Windows command

```text
python research\liquidity\liquidity_wall_discovery_v1\01_liquidity_wall_discovery_research.py --symbol ETH-USDT-SWAP --start-date 2026-01-01 --end-date "2026-06-30 23:59:59" --books-depth 5000 --touch-timeframe 5s --price-step 1 --out-dir data\reports\research\liquidity\liquidity_wall_discovery_v1
```

The command reads the existing 5s trade-bar cache. It does not silently build
missing 5s bars. Add `--build-missing-trade-bars` only when explicitly desired.

## Decision boundary

This stage cannot promote a trading strategy. Its best possible decision is:

```text
WALL_DISCOVERY_RESEARCH_CONTINUE
```

That requires causal audit pass, known-case visual coverage, and multiple
Train/Holdout-stable environment separators. Otherwise the wall vocabulary
remains unaccepted and strategy research must not start.
