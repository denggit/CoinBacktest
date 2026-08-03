# ETH AI Trading R03.4.2 — Frozen q90 Long-Tail Exit Audit

## Goal

Turn the frozen R03.4.1 `base_identity` q90 long-opportunity signal into a real path-based strategy candidate without sacrificing positive expectancy.

Positive expectancy is the primary gate. Win rate, trade count, payoff thickness, and drawdown are secondary diagnostics. A candidate is rejected when any cosmetic improvement makes either 2024 or 2025 net expectancy non-positive.

## Frozen opening model

- Target: `long_utility_h6 = long_mfe_h6 - 1.25 * long_mae_h6`
- LightGBM objective: `regression_l1`
- `n_estimators=420`
- `learning_rate=0.035`
- `num_leaves=31`
- `min_child_samples=300`
- `train_sample_cap=400000`
- `random_state=20260801`
- Primary entry threshold: prior-quarter q90 of the frozen score
- q95: quality-control sleeve only

The market-state model is formally abandoned for trading. R03.4.2 does not load state caches and does not use strategic, tactical, entry, or activity state fields.

## Walk-forward

### WF_2024

- Fit: 2023-01-01 through the embargoed end before 2023Q4
- Calibration: 2023Q4
- Test: 2024

### WF_2025

- Fit: 2023-01-01 through the embargoed end before 2024Q4
- Calibration: 2024Q4
- Test: 2025

2026 remains sealed. Events that cannot complete their required path before 2026 are excluded.

## Entry and event construction

- Decision interval: 15 minutes
- Entry: open at decision + 1 minute
- Delay audit: decision + 3 and +5 minute opens
- Dense q90/q95 alerts are merged within 30 minutes
- Independent event peaks use a six-hour pairwise cooldown
- Only one long can be open; later events are skipped while the position remains open
- No scale-in

## Structural stops

### S60

- Prior 60 completed one-minute lows
- Buffer: max of 4 bps and 0.5 times prior 60-minute ATR
- Minimum stop distance: 0.35%
- Maximum accepted stop distance: 1.80%

### S180

- Prior 180 completed one-minute lows
- Same buffer
- Minimum stop distance: 0.45%
- Maximum accepted stop distance: 2.20%

The rolling low and ATR are shifted by one minute. The entry minute and future bars cannot affect the stop. Trades with an unavailable or excessively distant structural stop are rejected rather than forced into a fixed stop.

## Preregistered exits

1. `fixed_6h_diagnostic` — original reference only
2. `s60_tp_1p5r`
3. `s60_tp_2p0r`
4. `s180_tp_2p0r`
5. `s60_trail_a1p0_g0p5`
6. `s60_trail_a1p5_g0p75`
7. `s60_renew_q70_trail`
8. `s60_renew_q60_trail`
9. `s180_renew_q70_trail`
10. `s60_renew_q70_invalidate_q50_trail`

Trailing stops activated by the current minute high can only execute from the next minute. If stop and target are both touched in one minute, the stop is assumed first.

## Rolling renewal

- Re-evaluate the same frozen base score every six hours
- Renewal threshold comes from the prior calibration quarter
- q70 and q60 are tested as preregistered maintenance thresholds
- Failed renewal exits at the next minute open
- Optional early invalidation requires two consecutive q50 failures after at least one hour
- 48 hours is a safety cap, not a target exit

A candidate cannot pass if more than 20% of trades end through the safety cap.

## Cost and risk audit

- Base round trip: 0.13%
- Stress: 1x, 2x, and 3x
- Risk-sized equity: 1% equity risk per trade, capped at 1.5x notional
- Monthly and quarterly summaries
- 1/3/5 minute entry delay
- Top-ten profit concentration and results after removing the top ten winners
- Exit reason and holding-duration distributions

## Positive-expectancy pass gate

For q90 and one-minute entry delay:

- 2024 and 2025 positive after 1x and 2x costs
- 2x PF at least 1.20 in both years
- At least 80 trades in each year
- At least six of eight positive quarters
- Positive after removing each year's top ten winners
- Positive after a three-minute entry delay in both years
- Risk-sized MDD no worse than 20%
- Safety-cap share no more than 20%
- Top-ten profit share no more than 60%
- At least 60% of the fixed-six-hour 2x expectancy retained in both years

## Run

```text
python research\eth_ai_trading\03_4_2_long_tail_exit_audit.py
```

Rebuild only the reusable R03.4 future-outcome cache when required:

```text
python research\eth_ai_trading\03_4_2_long_tail_exit_audit.py --force-rebuild-outcomes
```

## Reports

`data/reports/research/eth_ai_trading/03_4_2_long_tail_exit_audit`

Key files:

- `03_signal_event_audit.csv`
- `04_score_threshold_audit.csv`
- `05_trade_summary.csv`
- `06_period_summary.csv`
- `07_exit_reason_summary.csv`
- `08_duration_summary.csv`
- `09_cost_delay_stress.csv`
- `10_top10_concentration.csv`
- `11_stable_candidates.csv`
- `12_trade_details.csv`
- `99_decision.md`
- `gpt_review_pack.zip`
