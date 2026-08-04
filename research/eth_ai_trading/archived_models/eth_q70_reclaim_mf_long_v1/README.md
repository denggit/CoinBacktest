# ETH Q70 Reclaim MF Long V1

## Lifecycle

```text
Model name: ETH Q70 Reclaim MF Long V1
Model slug: eth_q70_reclaim_mf_long_v1
Research family: medium-frequency / medium-horizon Long
Lifecycle: ARCHIVED_AFTER_SEALED_HOLDOUT_FAILURE
Live approved: NO
Capital allocation: 0
Final research stage: R03.4.2.17
Archive closeout stage: R03.4.2.18
```

This model is archived, not deleted. Historical code remains in its original paths so prior reports and causal audits remain reproducible. This directory is the authoritative model card, frozen rule set, empirical closeout and lifecycle lock.

## Why this name

- **Q70**: the opening pool used a frozen 70th-percentile score threshold.
- **Reclaim**: profitable positions were held by deterministic `failed_reclaim` structure logic rather than a fixed take profit.
- **MF Long**: the sleeve was a medium-horizon ETH perpetual Long model, normally holding around fourteen to eighteen hours.
- **V1**: any future repair must receive a new version identity and new untouched validation.

## Final status

The model passed 2024–2025 development and account audits but failed the untouched January–June 2026 sealed holdout. July recovered under the same frozen artifact, yet opening expectancy remained weak, score exceedance drift worsened, and returns were concentrated in a few long-held winners. R03.4.2.17 found no simple causal 1D/4H gate that repaired the failure; broad score drift was the dominant diagnosis.

**Do not deploy, retune on the opened 2026 periods, or describe this model as a validated live strategy.**

## Archive contents

- `MODEL_CARD.md` — model purpose, inputs, training and execution contract.
- `FROZEN_POLICY.json` — machine-readable frozen trading rules.
- `EMPIRICAL_RESULTS.md` — development, sealed holdout and July results.
- `RESEARCH_TIMELINE.md` — cumulative stage history and failed upgrade paths.
- `FAILURE_AND_LESSONS.md` — why V1 was closed and what remains reusable.
- `REPRODUCTION_AND_REPORT_INDEX.md` — authoritative scripts and report locations.
- `NEXT_MODEL_BOUNDARY.md` — constraints preventing contamination of the next independent model.
- `LIFECYCLE_LOCK.json` — explicit zero-capital and no-retuning lock.
- `ARCHIVE_MANIFEST.json` — archive metadata.
