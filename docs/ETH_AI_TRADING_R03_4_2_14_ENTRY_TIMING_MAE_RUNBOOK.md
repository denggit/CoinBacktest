# ETH AI Trading R03.4.2.14 Runbook

## Windows command

```text
python research\eth_ai_trading\03_4_2_14_entry_timing_mae.py
```

Optional data directory:

```text
python research\eth_ai_trading\03_4_2_14_entry_timing_mae.py --data-dir data\okx_trade_bars
```

## Required source reports

```text
data\reports\research\eth_ai_trading\03_4_2_8a_occupied_signal_atlas
data\reports\research\eth_ai_trading\03_4_2_12_soft_failure_tail_compression
data\reports\research\eth_ai_trading\03_4_2_13_score_risk_sizing
```

## Frozen strategy

```text
q70 opening model
+ equal one-R
+ real 2% hard stop
+ 1.5% completed-close soft failure
+ failed_reclaim non-time profit exit
```

Only first-entry timing changes.

## Policies

- E0: immediate next-open C2 anchor.
- E1: wait up to 30 minutes for a higher q70 score; otherwise bounded fallback.
- E2: wait up to 45 minutes for a higher q70 score without paying more than 0.25 prior 1m ATR above the immediate entry; otherwise fallback.
- E3: wait up to 60 minutes for a 0.5% pullback and completed 5m reclaim; otherwise fallback.

Formal candidates must retain at least 90% of frozen C2 trades. No fixed-time exit is introduced.

## Output

```text
data\reports\research\eth_ai_trading\03_4_2_14_entry_timing_mae
```

Read first:

```text
99_decision.md
02_historical_metric_contract.csv
05_mae_attribution_summary.csv
06_entry_decisions.csv
10_policy_summary.csv
11_policy_gate.csv
gpt_review_pack.zip
```
