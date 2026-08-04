# ETH AI Trading R03.4.2.11 Runbook

## Purpose

Test whether q70 P0 can carry more useful nominal exposure through staged execution and asymmetric add-ons, rather than by tightening the complete base stop or increasing unconstrained leverage.

## Command

```text
python research\eth_ai_trading\03_4_2_11_staged_entry_pyramiding.py
```

## Dynamic sizing

```text
units = allowed account-risk dollars / stop distance in price
```

The base and each add-on have separate risk budgets and stop distances.

## Policies

- P0: 1R base, 3% disaster, failed reclaim.
- F1: size from 1.5% operating risk; completed-close failure exit; 3% / 2R tail.
- S1: 0.60R base plus 0.40R staged completion.
- T1: 1R base plus one 0.35R causal-N add.
- P1: 1R base plus two 0.35R causal-N adds.

## Important safeguards

- Add-ons do not sell or modify the base.
- Add-ons have independent hard stops.
- No add in BROKEN or pending failed-reclaim state.
- Visible unrealized profit must cover add risk for Turtle/pyramid policies.
- Maximum hard tail is 2R.
- Maximum gross nominal exposure is 1.5x equity.
- No fixed-time exit.
- 2026 remains sealed.

## Outputs

Read `99_decision.md`, then inspect:

- `08_policy_summary.csv`
- `09_policy_gate.csv`
- `04_account_cycles.csv`
- `05_account_legs.csv`
- `06_account_actions.csv`
- `07_daily_equity.csv`

## Interpretation

A larger nominal position is useful only if account return improves after real fees, stops, drawdown and tail-risk accounting. A policy that merely uses more leverage but underperforms P0 fails.
