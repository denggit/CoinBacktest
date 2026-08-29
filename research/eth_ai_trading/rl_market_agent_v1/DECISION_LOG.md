# Decision Log

- Clean-sheet reset: do not inherit old `eth_ai_trading` model assumptions or strategy gates.
- RL is a decision-policy layer, not the first alpha-discovery layer.
- Supervised opportunity prediction is a mandatory gate before offline RL.
- Maximum flat days is the first final-selection metric, but no reward may force meaningless trades merely to improve it.
- 2026 is sealed for later final validation.
- R00 uses public `src.data_feed` loaders only; no data API duplication.

## R00.2
Do not weaken K-line coverage to optional when the official cache is stale, because a calendar-shaped NaN block would contaminate sealed holdout evaluation. Prefer official K-lines, but causally fill missing fixed-bar timestamps from the already-mandatory local 1m tick-derived trade bars and disclose fallback counts in coverage metadata.


## R00.3 - 2026-08-15 local data refresh

R00.2's trade-bar fallback remains documented as history but is superseded for the active dataset path. After local prebuild, the active K-line contract is now: one official 1m K-line source -> causal resample to 5m/15m/1H/4H/1D. Tick-derived trade bars stay independent. The sealed holdout still starts at 2026-01-01. The final 360-minute tail is reserved for forward labels and is not emitted as decision rows.


## R01 strategy-first rule

The programme deliverable is a profitable, robust, live-migratable ETH strategy—not a paper, IC score, feature-importance story, or abstract edge. R01 therefore predicts the net return of frozen executable trade templates and immediately runs a full strategy replay. Model-quality statistics are secondary diagnostics only.

The R00 storage-level `iter_training_shards()` API is intentionally blocked for model work. Forward labels must be accessed through horizon-aware purged windows so no label can cross a fold boundary or the 2026 seal.
