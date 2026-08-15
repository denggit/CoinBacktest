# Current State

## Active research family

`ETH Latent Liquidity Pool Path Learning V1`

## Closed execution branch

`R01.3 post-release confirmation execution`

Decision: `STOP_LATENT_LIQUIDITY_PATH_V1_EXECUTION_NOT_VIABLE`.

Prediction existed, but confirmation came too late and reward/risk failed after realistic costs. No threshold/confirmation/stop rescue is allowed.

## Spatial/decontamination branch status

R02 -> R02.3.1b progressively removed exposure, distance and zero-inflation artifacts rather than tuning for prettier metrics.

Latest real result:

`BLOCKED_R02_3_1B_TARGET_CONSISTENCY_STILL_DISTANCE_CONTAMINATED`.

Key facts:

- target-scale correction was real but did not remove the core contamination;
- DOWN corrected residual distance correlation = **0.090 Validation / 0.140 Holdout**;
- UP = **0.149 / 0.203**;
- Huber -> L2 objective change was tiny (~0.005-0.009 mean absolute gap);
- release-hurdle AUC fell from **~0.69 TRAIN** to **~0.50 future periods**;
- actual release rate rose from roughly **7-10% in 2023** and **28-29% in 2024** to **~43-45% in 2025-2026**.

Conclusion: do not open another nuisance-target rescue stage merely to keep this branch alive.

## Retained evidence

- Sweep Depth / Reversal Room remain partially predictable and frozen as separate geometry tasks.
- R01 release-path structure remains useful descriptive/label infrastructure.
- Swing remains 15m+ supplemental context only; never a pool definition or candidate gate.

## Active stage

`R02.4 — Latent Liquidity Economic Ceiling Audit`

Primary question:

> Before building another predictor, does the true liquidity-release / favorable-reversal mechanism contain enough economic room to matter at all?

R02.4 trains **no model** and adds **no new data family**. It reuses one representative per R01.1 release episode and intentionally uses future-informed oracle geometry to calculate an upper bound.

Primary frozen horizon: **300 seconds**.

Primary cost: **11bp** round trip; mandatory stress: **22bp**; 33bp is diagnostic.

The hard stop/go gate uses perfect-exit net-MFE on `FAVORABLE_REVERSAL_ORACLE` in Validation and Holdout. Fixed 1R/1.5R/2R outcomes are secondary realizability diagnostics.

## Decision meaning

If R02.4 returns `STOP_LATENT_LIQUIDITY_REVERSAL_ECONOMIC_CEILING_TOO_THIN`, stop this latent-liquidity reversal branch rather than adding Range/Footprint/OI/Books.

If it returns `CONTINUE_LATENT_LIQUIDITY_IDENTIFICATION_ECONOMIC_CEILING_EXISTS`, it means only that enough money exists in the ideal mechanism to justify further causal identification work. It is not live approval.

## Validation status

No tradable strategy and no live approval exists. 2025Q4-2026H1 remains development holdout, not newly sealed evidence.
