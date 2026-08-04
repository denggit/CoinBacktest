# ETH AI Trading R03.4.2.15 Runbook

## Windows command

```text
python research\eth_ai_trading\03_4_2_15_final_account_live_readiness.py
```

Optional explicit source report:

```text
python research\eth_ai_trading\03_4_2_15_final_account_live_readiness.py --source-report-dir data\reports\research\eth_ai_trading\03_4_2_14_entry_timing_mae
```

## Required source decision

```text
R03.4.2.14 = PASS_C2_FROZEN_NO_ENTRY_UPLIFT
```

## Frozen strategy

```text
q70 immediate next 1m open
+ equal one-R
+ real 2% exchange-side hard stop
+ 1.5% completed 15m-close soft failure
+ failed_reclaim non-time structural exit
```

No model, threshold, entry, stop or exit parameter is changed.

## Audit scope

- Continuously compound WF_2024 and WF_2025 without resetting equity at the year boundary.
- Keep 2023 as training/development history; do not report it as independent OOS account return.
- Audit 1/3/5-minute delay and 2x/3x cost cells.
- Report monthly/quarterly returns, losing streaks, holding duration, inactivity, drawdown duration and top-10 concentration.
- Convert the observed worst net loss into a live fee/slippage risk reserve.
- Audit OKX whole-contract sizing at multiple initial equity levels.
- Emit an immutable model-release governance contract and restart-safe live state fields.
- Keep 2026 sealed.

## Model training governance

A calendar month is an audit cadence, not an automatic release cadence.

```text
Daily: data and feature health
Monthly: drift/performance/calibration audit
Monthly optional: shadow candidate retraining
Quarterly or event-driven: explicit release gate
Never: automatic monthly champion replacement
```

## Output

```text
data\reports\research\eth_ai_trading\03_4_2_15_final_account_live_readiness
```

Read first:

```text
99_decision.md
05_continuous_scenario_summary.csv
06_monthly_returns.csv
08_okx_lot_size_audit.csv
09_net_risk_reserve.csv
10_model_governance.csv
11_live_state_contract.csv
12_final_gate.csv
gpt_review_pack.zip
```
