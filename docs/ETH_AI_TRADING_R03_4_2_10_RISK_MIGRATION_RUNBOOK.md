# ETH AI Trading R03.4.2.10 Runbook

Run on Windows from the repository root:

```bat
python research\eth_ai_trading\03_4_2_10_risk_migration.py
```

Optional explicit data directory:

```bat
python research\eth_ai_trading\03_4_2_10_risk_migration.py --data-dir D:\your\data\root
```

The script reuses validated R03.4.2.8A, R03.4.2.8B and R03.4.2.9 reports plus the public one-minute Trade Bar Loader. It does not import another research script.

Expected report directory:

```text
data\reports\research\eth_ai_trading\03_4_2_10_risk_migration
```

Review in this order:

1. `01_preflight.json`
2. `10_account_policy_summary.csv`
3. `11_policy_gate.csv`
4. `06_account_actions.csv`
5. `05_event_decisions.csv`
6. `99_decision.md`
7. `gpt_review_pack.zip`

Do not interpret results when `14_failures.csv` is non-empty or the decision is `BLOCKED_DATA`, `BLOCKED_SOURCE_REPORT` or `FAIL_RUNTIME`.
