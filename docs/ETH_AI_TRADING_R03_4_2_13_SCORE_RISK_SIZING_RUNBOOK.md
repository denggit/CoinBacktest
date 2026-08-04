# ETH AI Trading R03.4.2.13 Runbook

## Windows command

```text
python research\eth_ai_trading\03_4_2_13_score_risk_sizing.py
```

Optional data directory:

```text
python research\eth_ai_trading\03_4_2_13_score_risk_sizing.py --data-dir data\okx_trade_bars
```

## Required source

A completed passed report at:

```text
data\reports\research\eth_ai_trading\03_4_2_12_soft_failure_tail_compression
```

## Output

```text
data\reports\research\eth_ai_trading\03_4_2_13_score_risk_sizing
```

Read:

```text
99_decision.md
03_score_tier_attribution.csv
04_cross_year_score_order.csv
08_policy_summary.csv
09_policy_gate.csv
gpt_review_pack.zip
```

## Interpretation

- `PASS_SCORE_TIER_RISK_POLICY`: one fixed tier map passed both years and all stress cells.
- `PASS_EQUAL_RISK_RETAINED`: score ordering was unstable or tiering did not justify its return cost; keep all q70 at equal one-R.
- `FAIL_RUNTIME`: do not interpret results.
