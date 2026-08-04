# Failure Analysis and Durable Lessons

## Why V1 was closed

The failure was not a single bad stop, a missing Bull filter or a lack of leverage. The binding issue was that the frozen opening score did not retain stable calibration and ranking quality in 2026. A threshold intended to select roughly the top 30% of opportunities exceeded 58% in 2026 H1 and about 70% in Q2/July. The strategy became insufficiently selective, and high score tiers were not reliably better.

Simple market-state filters did not solve the problem. Bear-aligned trades were profitable in several historical periods, while some Bull-aligned samples lost money. Retrofitting a Bull-only or Bear-exclusion gate after opening 2026 would be both empirically weak and methodologically invalid.

## What remains valuable

- Multi-timeframe causal feature and label infrastructure.
- Immutable model/feature-schema/version seals.
- The distinction between opening Edge and exit-overlay Edge.
- Real hard-stop sizing rather than sizing from a softer operational threshold.
- 2% hard-stop plus 1.5% completed-close soft-failure mechanics as a reusable hypothesis.
- Deterministic `failed_reclaim` logic for protecting occasional long winners.
- Continuous-account, cost, delay, top-winner and lot-size audit tooling.
- A proven discipline for consuming and preserving a sealed holdout failure.

## What must not be repeated

- Do not retune q70, the score model, stop, soft failure or exit on 2026 H1/July and then call the result validated.
- Do not use score increases to add, average down or enlarge risk.
- Do not force simple Bull/Bear gates because they sound intuitive.
- Do not optimize a large parameter grid around the opened period.
- Do not confuse leverage setting with account risk. Higher exchange leverage only reduces posted margin at a fixed notional; it does not make a wide stop safe.

## Conditions for a future V2

A future V2 must have a separate version identity and may use 2026 only as development data. It must explicitly address calibration drift and ranking stability, then wait for new untouched forward data. It cannot inherit live approval from the historical V1 development curve.
