# R13 — Reversal Quality & Causal Entry Discovery

Date: 2026-08-16

## Hypothesis

Completed-trend liquidity sweeps should not be traded as automatic reversals. The research question was whether liquidity quality, sweep morphology, expected response, reclaim quality, MSS quality, displacement/FVG quality and structural RR identify a causal point where direct opposite delivery becomes economically distinguishable from a deeper-same-side-first failure.

## Data and split

- Market: OKX `ETH-USDT-SWAP`.
- Bare 1m source: `src.data_feed.OKXDataLoader` only.
- Exact 1m coverage: 2022-01-01 00:00 through 2026-08-15 23:59, 2,430,720 expected/observed rows, zero internal gaps.
- R12 source: completed-trend ITH/ITL/LTH/LTL first-passage paths.
- Discovery: through 2024-12-31.
- Validation: 2025-01-01 through 2025-06-30.
- Embargo: 2025-07-01 through 2025-07-31 so 30-day labels cannot cross into holdout.
- Sealed R13 holdout: from 2025-08-01; 305 R12 rows available and zero included in R13 outputs.
- Holdout qualification: not pristine relative to the earlier R12 aggregate atlas, but untouched for R13 feature/rule selection.

The comparison contains 719 events:

- discovery BSL: 306, direct reversal 48 (15.69%);
- discovery SSL: 207, direct reversal 56 (27.05%);
- validation BSL: 101, direct reversal 26 (25.74%);
- validation SSL: 105, direct reversal 19 (18.10%).

`cascade_then_opposite_delivery` is a failure for the direct-reversal thesis because deeper same-side liquidity was reached first.

## Causality design

- Every feature has an information-availability time.
- Closed-bar signals execute at the next eligible 1m open.
- Same-bar TP/SL is stop-first.
- An FVG limit fill cannot receive target credit on its fill bar.
- TP and SL are frozen from root-time completed-trend liquidity.
- Feature-bin thresholds are discovery quartiles applied unchanged to validation.
- Root/pre-sweep bins use `root_next_open` economics.
- 15-minute expected-response bins use `response_15m_market` economics.
- Reclaim, MSS and FVG bins use their respective causal confirmation entries.

`12_causal_audit.csv` contains 17 checks and zero violations. Eight focused R13 tests cover holdout sealing, embargo, response-window immutability, MSS availability, next-open execution, pessimistic barrier ordering, response-entry timing, and causal feature-bin attribution.

## Critical reporting correction

The first R13 report incorrectly credited post-root 15-minute response bins with root-next-open PnL. Although the feature values themselves used only closed bars, that economic attribution entered before the feature existed and was an oracle report.

The report was invalidated and regenerated as script version 13.0.1. It now creates `early_15m_available_time`, the `response_15m_market` and `fvg_market` entries, and maps every feature family to its own causal entry. Any pre-correction PF quoted for early-response bins is withdrawn.

## Corrected economic result

No unfiltered entry is promotable at 2x costs.

SSL examples:

| Entry | Discovery PF | Validation PF | Discovery mean net | Validation mean net |
| --- | ---: | ---: | ---: | ---: |
| root next open | 1.10 | 0.80 | +0.232% | -0.561% |
| response 15m market | 1.12 | 0.77 | +0.306% | -0.820% |
| reclaim market | 1.32 | 0.63 | +0.666% | -1.255% |
| reclaim FVG proximal limit | 1.45 | 0.65 | +0.925% | -1.222% |
| 2m MSS FVG proximal limit | 1.22 | 0.78 | +0.562% | -0.796% |

BSL is negative or approximately flat across the entry family. Validation PF ranges from about 0.61 to 1.02 and mean net return is non-positive except for economically negligible near-zero cases.

Yearly SSL behavior confirms regime dependence: most entry families are profitable in 2023, flat/negative in 2024, and clearly negative in 2025 validation.

## Feature observations

Stable descriptive separation exists, especially for SSL: younger liquidity, stronger/moderate early response, higher MFE/MAE, higher path efficiency, and FVG timing all contain information about path class. Descriptive classification information is not automatically a tradeable edge.

Corrected causal economics reject the earlier apparent top-quartile 15-minute path-efficiency edge: Q4 PF is 1.88 in discovery but 0.83 in validation. Q4 15-minute MFE/MAE falls to PF 1.42 / 1.05. Q3 early target progress is 1.99 / 1.31, but the neighboring Q4 validation bin is only 0.74, so it is not a stable threshold family.

Several post-hoc broad bins remain hypothesis-generating only:

- youngest half of liquidity age: 2x PF 1.24 discovery / 1.41 validation;
- middle half of structural RR: 1.52 / 1.41;
- middle half of 15-minute maximum body/ATR: 1.87 / 1.20;
- top pre-sweep-return quartile: 1.63 / 3.00, but the neighboring Q3 bin fails.

These do not qualify for promotion. Removing the top five winners drops validation PF to approximately 0.57, 0.36, 0.46 and 0.32 respectively. Only the discovery half of the moderate-body diagnostic remains above one after top-five removal (about 1.27). The apparent edges are too dependent on a small right tail.

## Frozen conclusions and stopped directions

1. Universal completed-trend sweep reversal is stopped.
2. Boolean reclaim/MSS/FVG confirmation is insufficient.
3. No R13 entry model or one-dimensional feature bin is a strategy.
4. Do not combine the attractive post-hoc bins into a multi-filter rescue rule.
5. Do not open the R13 holdout for these candidates.
6. Long/short symmetry is rejected; BSL reversal is particularly weak.
7. Feature/path separation may still be useful as state information for another mechanism.

Research correction (2026-08-17): these conclusions reject sweep-immediate,
Boolean landmark, and one-dimensional-bin entries. They do **not** reject an
ordered, quality-aware sequence in which post-sweep structure is established
before a meaningful MSS, displacement, executable FVG retracement, and protected
swing. That remaining hypothesis is preregistered in `R27_PRECOMMITMENT.md` and
must be completed before the completed-trend reversal branch is permanently
archived.

## Next hypothesis

The dominant path is not direct reversal: deeper same-side liquidity is reached first in roughly 73–84% of discovery events and 74–82% of validation events. R14 therefore tested a separate liquidity-acceptance/continuation sleeve, while the later R27 correction reopens only the previously untested ordered sequential-reversal hypothesis:

`completed-trend liquidity sweep -> causal acceptance outside the swept region -> continuation toward frozen deeper same-side liquidity`

R14 must use a small predeclared acceptance family, a reclaim-based structural invalidation, realistic 1x/2x/3x costs, and the same sealed holdout. FVG execution is deferred until a simple market-entry continuation edge exists.

## Primary evidence

- `data/reports/research/ict/mss2/r13_reversal_quality_entry_discovery/00_manifest.json`
- `02_holdout_seal.csv`
- `04_reversal_quality_feature_rows.csv.gz`
- `07_feature_bin_atlas.csv`
- `08_feature_monotonicity.csv`
- `09_entry_candidate_outcome_rows.csv.gz`
- `10_entry_model_summary_cost2x.csv`
- `11_entry_model_year_summary_cost2x.csv`
- `12_causal_audit.csv`
