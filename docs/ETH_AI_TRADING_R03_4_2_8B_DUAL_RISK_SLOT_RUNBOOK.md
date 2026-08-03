# R03.4.2.8B Dual Risk-Slot Account Audit

## Purpose

Test whether maximum-two-tranche account execution can keep the deterministic `failed_reclaim` exit while recovering q70 opportunities skipped by the single-position baseline.

This stage does not retrain q70, tune `failed_reclaim`, open 2026, or allow unlimited averaging down.

## Required upstream artifact

Run R03.4.2.8A first. The following directory must exist:

```text
data\reports\research\eth_ai_trading\03_4_2_8a_occupied_signal_atlas
```

R03.4.2.8B validates the source manifest, causal audit, failures file and standalone outcomes before simulation.

## Run

```bat
python research\eth_ai_trading\03_4_2_8b_dual_risk_slot_account.py
```

Optional local data root:

```bat
python research\eth_ai_trading\03_4_2_8b_dual_risk_slot_account.py --data-dir D:\your\trade_bar_root
```

## Frozen policies

- `P0_single_1R`: one 1.0R slot; all overlapping signals skipped.
- `P1_equal_05_05`: two equal 0.5R slots.
- `P2_primary_065_secondary_035`: 0.65R primary plus 0.35R secondary.
- `P3_protected_065_035`: same slot budget, but a second tranche is blocked when the active root is in a dangerous average-down or failed-reclaim process.

One full R is 1% of marked account equity at entry. Notional is derived from `risk budget / 3% disaster distance`.

## Main outputs

```text
data\reports\research\eth_ai_trading\03_4_2_8b_dual_risk_slot_account\99_decision.md
data\reports\research\eth_ai_trading\03_4_2_8b_dual_risk_slot_account\05_account_policy_summary.csv
data\reports\research\eth_ai_trading\03_4_2_8b_dual_risk_slot_account\06_account_trades.csv
data\reports\research\eth_ai_trading\03_4_2_8b_dual_risk_slot_account\07_daily_equity.csv
data\reports\research\eth_ai_trading\03_4_2_8b_dual_risk_slot_account\09_policy_gate.csv
```

## Interpretation

A pass only proves that a unified two-slot account policy restores opportunity coverage and improves the single-position account result without breaking the pre-registered risk constraints. It does not freeze final entry timing, structural stop distance, score-tier sizing or the final live exit chain.
