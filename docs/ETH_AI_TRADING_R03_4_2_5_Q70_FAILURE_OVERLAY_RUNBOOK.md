# ETH AI Trading R03.4.2.5 Runbook

## Objective

Audit a highly selective early-loss overlay on the frozen q70 long opening pool. The overlay must improve or protect positive expectancy without deleting the lower score tiers or converting the strategy into a tight-stop system.

## Command

```bat
python research\eth_ai_trading\03_4_2_5_q70_failure_overlay.py
```

Optional cache rebuild:

```bat
python research\eth_ai_trading\03_4_2_5_q70_failure_overlay.py --force-rebuild-outcomes
```

## Frozen contract

- Opening model: R03.4.1 six-hour long utility LightGBM.
- OOS opening pool: causal prior-quarter q70 percentile.
- Score tiers: q70-q80, q80-q90, q90+.
- Holding-risk train pool: rolling OOF q50 events only.
- T+60: warning only.
- T+180: early exit requires prior warning, extreme model probability and structural deterioration.
- Higher opening scores require stricter exit evidence.
- 2026 remains sealed.
- Market-state outputs remain abandoned and are not loaded.

## Policies

- `fixed_6h`: unchanged opening-edge benchmark.
- `fixed_6h_disaster_stop`: benchmark plus a wide 3% safety floor.
- `global_failure_overlay`: global OOF warning and confirmation thresholds.
- `tiered_failure_overlay`: progressively stricter confirmation from q70-q80 to q90+.
- `ultra_failure_overlay`: extreme probability plus at least four structural failures.

## Structural confirmation

The T+180 decision counts independently observable failures such as:

- price remains below entry;
- the last 60 minutes remain negative;
- the pre-entry 60-minute low was broken or not reclaimed;
- the 15-minute path keeps printing lower lows;
- recovery from the path trough is weak;
- most closes have remained underwater.

Probability alone never exits a trade.

## Important reports

- `05_model_metrics.csv`
- `06_probability_thresholds.csv`
- `07_feature_importance.csv`
- `09_policy_summary.csv`
- `10_quarter_summary.csv`
- `11_score_tier_policy_summary.csv`
- `12_exit_reason_summary.csv`
- `14_score_upgrade_diagnostics.csv`
- `15_stable_candidates.csv`
- `18_trade_details.csv`
- `99_decision.md`
- `gpt_review_pack.zip`

## Interpretation

A pass requires an overlay to remain robust in both 2024 and 2025 and to raise total 2x-cost profit versus the identical q70 fixed-six-hour benchmark. A risk-control-only result may be retained as a safety candidate, but it is not the final exit. The next research stage will independently model incremental value beyond six hours so profitable positions may remain open for days without a mechanical time limit.
