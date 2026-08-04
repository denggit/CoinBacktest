# ETH AI Trading R03.4.2.16.1 — July 2026 Forward Extension

## Purpose

Run the unchanged frozen C2 MF Long sleeve on the newly available 2026-07-01 through 2026-07-31 window.

This is **not** a repair of the failed R03.4.2.16 holdout and is **not** a second sealed holdout pass. It is a one-month forward extension used to test whether the poor January-June 2026 result was strongly regime-dependent.

## Frozen model and execution

- Fit: 2023-01-01 through 2025-09-30 06:00.
- q70 calibration: 2025 Q4 only.
- Entry: immediate next 1m open after q70.
- Risk: equal q70 risk; no add-on.
- Protection: real 2% hard stop and 1.5% completed-15m-close soft failure.
- Exit: `failed_reclaim`; no fixed take profit.
- January-June 2026: comparison only, never fit or calibration.
- July 2026: inference and diagnostic scoring only.

## Required data

Only the public 1m OKX Trade Bar cache is required. Higher timeframes are derived causally from 1m.

Expected July coverage:

```text
2026-07-01 through 2026-07-31
31 UTC days
1,440 rows per day
44,640 rows total
```

## Command

```bat
python research\eth_ai_trading\03_4_2_16_1_2026_july_forward_extension.py
```

Optional clean isolated-cache rebuild:

```bat
python research\eth_ai_trading\03_4_2_16_1_2026_july_forward_extension.py --force-rebuild-base --force-rebuild-outcomes
```

## Report directory

```text
data\reports\research\eth_ai_trading\03_4_2_16_1_2026_july_forward_extension
```

Review first:

- `00_pre_open_seal.json`
- `03_model_threshold_audit.csv`
- `05_fixed_6h_summary.csv`
- `12_july_scenario_summary.csv`
- `16_forward_diagnostic_gate.csv`
- `21_h1_vs_july_comparison.csv`
- `99_decision.md`
- `gpt_review_pack.zip`

## Interpretation

Possible decisions:

- `JULY_FORWARD_SUPPORTS_FROZEN_C2`
- `JULY_FORWARD_MIXED_SUPPORT`
- `JULY_FORWARD_DOES_NOT_SUPPORT_FROZEN_C2`

A supportive July result means only that regime dependence becomes more plausible. It does not reverse the failed January-June seal or authorize live deployment. A weak July result further supports score drift, model decay or missing Long-regime gating.
