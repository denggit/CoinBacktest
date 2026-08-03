# ETH AI Trading R03.4.2.3 Multi-stage Holding Runbook

Run from the CoinBacktest project root:

```text
python research\eth_ai_trading\03_4_2_3_long_tail_multistage_decision.py
```

The study reuses the existing R03.2 base feature cache and R03.4 outcome cache. It reads one-minute Trade Bars through `src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader` in bounded monthly chunks.

Output directory:

```text
data\reports\research\eth_ai_trading\03_4_2_3_long_tail_multistage_decision
```

Priority files:

```text
03_entry_oof_audit.csv
06_model_selection_audit.csv
07_model_metrics.csv
08_probability_thresholds.csv
11_policy_summary.csv
12_quarter_summary.csv
14_overlap_and_skip_audit.csv
15_stable_candidates.csv
18_trade_details.csv
99_decision.md
gpt_review_pack.zip
```

Interpretation rules:

- q50 is training-only and is never promoted as a trading threshold.
- q70 and q90 are evaluated separately.
- T+60 is observation-only.
- A T+180 or T+360 early exit requires high persistent-failure risk and low recovery probability.
- T+6h and T+24h are re-evaluation points, not mandatory time exits.
- Five days is a research safety cap for complete labels, not a proposed live time stop.
- No policy is accepted if it improves trade count or win rate while losing positive expectancy.
- A multi-stage policy is promoted only if it also increases two-times-cost total compounded profit in both 2024 and 2025 versus the same signal pool fixed-6h baseline.
- q70 expansion is promoted only if it increases two-times-cost total compounded profit in both years versus the q90 fixed-6h baseline.
