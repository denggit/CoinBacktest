# ETH Latent Liquidity Pool Path Learning V1

Research family for learning where latent stop/liquidation/fragile-position liquidity is likely to accumulate and what happens when price reaches it.

## Core discipline

- liquidity first, not Swing first;
- Swing only 15m+ and only as supplemental ablation;
- all high-timeframe context must be causally available;
- future outcomes may be labels only, never real-time features;
- large-data stages must use vectorized/indexed/chunked algorithms and resumable caches rather than Pandas row loops;
- mechanical nuisance (distance/activity) must be separated from the candidate spatial edge before adding richer data;
- no live approval from development evidence.

## Stage status

- R01 / R01.1: broad release atlas and liquidity-first path discovery — completed.
- R01.2: stable-path explanation / fixed confirmation audit — completed; fixed confirmations not profitable.
- R01.3: absorption completion / remaining-space post-release execution — **stopped**; prediction existed but post-confirmation reward/risk failed.
- R02: pre-event spatial location baseline — completed; Touch strong, old geometric pool score retired.
- R02.1: conditional absolute pool strength — completed; target retired due first-touch/exposure and threshold drift.
- R02.2: equal-window first-touch raw-density ranking — completed; raw density remained mechanically near-distance biased.
- R02.3: median/IQR distance-normalized Excess Liquidity — completed; **normalizer failed under zero inflation** although causal gates passed.
- R02.3.1: hurdle nuisance residualization + reversal residual ranking — **completed / blocked**; Excess remained distance contaminated although causal gates passed.
- R02.3.1b: hurdle target consistency + residual distance audit — **completed / blocked**; target correction did not remove the core distance/nonstationarity problem.
- **R02.4: latent-liquidity economic ceiling audit — active.**

R02.3.1b confirmed that the target-scale mismatch was real but not the root problem. Release background drift dominates and the corrected Excess target remains blocked.

R02.4 therefore asks a more commercial stop/go question before any new model is allowed: if the true favorable release were known by an oracle, is there enough MFE/reward-risk after 11bp and 22bp costs to justify further identification research? Oracle fields are future-only and may never become strategy logic. Sweep Depth / Reversal Room remain separate retained geometry tasks.

Read `CUMULATIVE_STAGE_RESULTS.md` before changing any stage or interpreting a report.
