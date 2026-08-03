# R12 Post-Sweep Rejection vs Acceptance

## Purpose

R12 performs the missing accurate test of the original liquidity-sweep idea:
a real Swing-Low stop pool is swept and abnormal sell flow is released; does the
market reject lower prices or accept the breakdown?

It does not predict the exact low. It observes only closed 1m bars at 1, 3, 5,
and 10 minutes after the sweep, then enters at the strict next 1m open.

## States

- `PRESSURE_TEST_REJECT`: floor reclaim occurs, later visible bars still contain net aggressive selling, that second pressure does not create a new low, and the checkpoint remains above the floor.
- `STRONG_REJECT`: checkpoint close is back above the zone ceiling.
- `REJECT`: checkpoint close is back above the zone floor.
- `RECLAIM_FAILED`: price reclaimed the floor during the visible window but is
  back below it at the checkpoint.
- `PERSISTENT_ACCEPT`: no floor reclaim and at least two thirds of visible closes
  remain below the floor.
- `MIXED_BELOW`: below the floor without meeting the persistent definition.

Rejection states are tested long. Acceptance states are tested short. Both
opposite directions remain in the full outcome table for diagnostics.

## Risk

- Long stop: lowest price visible by the checkpoint minus 5bp.
- Short stop: highest price visible by the checkpoint plus 5bp.
- Targets: 1R, 2R, and 3R.
- Horizon: 180 minutes.
- Same-bar target and stop: stop wins conservatively.
- 1x costs: 0.11% fees plus 2bp round-trip slippage.
- 2x stress doubles all costs.

## Run

Smoke gate:

```bat
python research\liquidity\12_post_sweep_rejection_acceptance_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59" --max-events 1000 --skip-review-pack
```

Full:

```bat
python research\liquidity\12_post_sweep_rejection_acceptance_study.py --symbol ETH-USDT-SWAP --start-date 2023-01-01 --end-date "2026-06-30 23:59:59"
```

R09 full report cache must exist. The loader accepts the historical directory
`structured_swing_stop_pool_hypotheses_r09` and the preferred numeric-prefix
alternative `09_structured_swing_stop_pool_hypotheses_r09`.

## Stop rule

If no rejection-long or acceptance-short state remains positive after 1x and 2x
costs across at least two periods, do not add further threshold combinations.
Retire the liquidity-sweep trading branch and preserve the atlas only as market
context.

## Main outputs

```text
00_manifest.json
01_data_quality.csv
02_frozen_design.csv
03_state_distribution.csv
04_state_feature_profile.csv
05_rejection_long_summary.csv
06_acceptance_short_summary.csv
07_period_stability.csv
08_state_transition_matrix.csv
09_release_interaction.csv
10_family_timeframe_summary.csv
11_candidate_scorecard.csv
12_causal_audit.csv
13_event_sample.csv
14_checkpoint_feature_table.csv.gz
15_outcome_label_table.csv.gz
16_research_brief.md
gpt_review_pack.zip
```

The state/profile reports also measure the price paid for confirmation:
`long_entry_delay_bp`, `short_entry_delay_bp`, and MFE already visible before
entry. This prevents a late, high-win-rate reclaim from being mistaken for a
profitable strategy after most of the move has already happened.
