# R02.4 Delivery — Economic Ceiling Audit

## Goal

Stop asking whether another model can improve a metric until the underlying latent-liquidity reversal mechanism proves that it contains enough money to trade.

## What is intentionally oracle-only

- true release event/reference price;
- favorable-reversal label;
- future MFE/MAE;
- stop distance = future MAE + 3bp.

These fields exist only to create an upper bound. They are prohibited from future causal/live logic.

## Primary economics

- horizons: 60 / 180 / 300 / 600 seconds;
- primary horizon: 300 seconds;
- costs: 6 / 8 / 11 / 22 / 33bp;
- primary/stress: 11 / 22bp;
- fixed-R diagnostics: 1R / 1.5R / 2R;
- one first row per release episode, avoiding repeated rows within the same release episode.

## Frozen decision gate

Validation and Holdout favorable-reversal oracle episodes must each have >=100 rows.

At 11bp:

- mean perfect-exit net-MFE >=10bp;
- perfect-exit net-MFE PF >=1.50;
- positive net-MFE rate >=65%;
- top-10-removed mean net-MFE >0.

At 22bp:

- mean perfect-exit net-MFE >0;
- PF >1.00.

Failure stops the branch. Passing only proves an economic ceiling exists and allows a later identification study.

## Run

`python research\eth_ai_trading\eth_latent_liquidity_path_v1\02_4_economic_ceiling_audit.py`

Send back `data\reports\research\eth_ai_trading\eth_latent_liquidity_path_v1\02_4_economic_ceiling_audit\gpt_review_pack.zip`.
