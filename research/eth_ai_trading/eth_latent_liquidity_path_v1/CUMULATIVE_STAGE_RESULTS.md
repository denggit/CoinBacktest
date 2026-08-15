# ETH Latent Liquidity Pool Path Learning V1 — Cumulative Stage Results

> This file is the cumulative handoff for every patch. New patches must carry it forward and update it. The research target is latent stop/liquidation liquidity, not Swing sweeping. Swing is only one supplementary 15m+ structural family.

## Predecessor model closeout

### ETH Q70 Reclaim MF Long V1

- Multi-timeframe supervised MF Long entry model with q70 immediate entry, 2% hard stop, 1.5% soft-failure review and `failed_reclaim` exit.
- 2024–2025 development looked strong, but the untouched 2026 H1 holdout failed: +4.8%, MDD -15.9%, PF 1.09, only 2/6 positive months; 3x cost turned negative.
- The q70 exceed rate drifted from the calibrated 30% to roughly 70%, and simple Bull/Bear gates did not repair the model.
- Final lifecycle: archived, zero capital, not live approved. Reusable pieces are the causal feature infrastructure, structural exit research, drift audit and sealed-validation discipline.

## New independent model

### Model name

`ETH Latent Liquidity Pool Path Learning V1`

### Frozen objective

Learn from broad multi-scale paths:

1. where latent stop/liquidation/fragile-position liquidity may accumulate;
2. what type of liquidity release occurs when price reaches it;
3. whether the release is absorbed and reversed or accepted and continued;
4. whether a causal entry can retain enough thickness after real costs.

This model does **not** equate liquidity with Swing. It does not claim direct observation of private stops.

---

## R01 — Broad release/reversal path atlas

### Purpose

Build a high-recall 1-second liquidity-release universe from a union of:

- turnover/trade-count/large-trade bursts;
- directional flow release;
- price shock;
- range expansion;
- rolling-boundary release.

Pre-event features exclude the release second. Future 600-second paths are labels only.

### Initial smoke-test findings

A seven-day command capped at 5,000 events actually covered roughly 40 hours because the old cap truncated earliest events. It produced 4,910 events but no cluster model because the discovery cutoff was 2024 and only 2026 was run.

Descriptive findings:

- `EXTEND_STABILIZE_REVERSAL` was much more common than immediate V-shaped reversal.
- DOWN releases showed stronger favorable reversal incidence than UP in that tiny sample.
- Pre-release multi-scale path features separated reversal/continuation more strongly than release-second burst size.

### R01 engineering fixes

- normalized pandas microsecond/nanosecond time axes;
- added causal macro-context coverage gate;
- prevented empty-cluster/report `KeyError` paths;
- clarified that a smoke test without discovery rows has no cluster conclusion.

---

## R01.1 — Liquidity-first full-history path atlas

### Structural correction

- removed all sub-15m Swing/Pivot features;
- retained all causally confirmed, not-yet-swept 15m/30m/1H/4H/1D levels until first sweep;
- Swing inventory became supplementary only and never an admission gate;
- added liquidity-first families: turnover per range, pressure without progress, overlap/residency, impact efficiency, compression, path efficiency and flow intensity;
- merged overlapping 1-second triggers into side-specific Liquidity Release Episodes.

### Full-history data

- complete labeled events: **2,373,610**;
- distinct Liquidity Release Episodes: **1,024,890**;
- favorable reversal label share: **12.02%**;
- frozen cluster fit: 250,000 sampled pre-2025 rows from 691,486 eligible rows;
- 12 discovery clusters;
- training period: 2023–2024;
- validation: 2025 Q1–Q3;
- holdout label: 2025 Q4–2026 H1.

### Stable reversal discovery paths

#### Cluster 10 — broad core reversal discovery

DOWN:

- 253,582 events / 118,893 Episodes;
- overall favorable reversal 29.04%, continuation 15.46%;
- training 30.00% vs 15.65%;
- validation 29.34% vs 15.63%;
- holdout 27.61% vs 15.06%.

UP:

- 205,688 events / 100,514 Episodes;
- overall favorable reversal 30.16%, continuation 16.03%;
- training 28.40% vs 16.96%;
- validation 30.47% vs 16.64%;
- holdout 31.75% vs 14.36%.

Interpretation: large-sample, bidirectional, cross-period path information exists.

#### Cluster 4 — stronger but narrower reversal discovery

DOWN:

- 16,768 events / 7,376 Episodes;
- overall reversal 37.82%, continuation 12.60%;
- holdout reversal 34.94%.

UP:

- 18,909 events / 8,611 Episodes;
- overall reversal 27.12%, continuation 14.12%;
- holdout reversal 26.32%.

#### Cluster 5 — rare high-conviction discovery

DOWN:

- 942 events / 541 Episodes;
- overall reversal 47.35%, continuation 7.32%;
- training/validation/holdout reversal 49.70% / 46.58% / 45.72%.

UP:

- 2,553 events / 1,361 Episodes;
- overall reversal 26.99%, continuation 11.16%;
- validation was weaker than training/holdout and requires caution.

#### Cluster 8 — continuation control

DOWN:

- 665,207 events / 302,578 Episodes;
- reversal 6.12%, continuation 11.05%.

UP:

- 409,902 events / 214,115 Episodes;
- reversal 6.18%, continuation 11.72%.

Interpretation: the path representation also identifies broad release types that should not be blindly faded.

### Outcome-shape finding

Across strong reversal clusters, the dominant class is not a one-tick V-shaped rebound. Most favorable labels are `EXTEND_STABILIZE_REVERSAL`: price continues in the release direction, stabilizes/loses impact efficiency, then reverses.

### Swing finding

- Simple presence of unswept Swing inventory did not create a useful standalone edge.
- Cluster 10 was often far from the nearest unswept Swing, proving this is not a Swing-sweep-only model.
- Cluster 5 was much nearer to an unswept Swing and may represent one special structural subtype.
- Several aggregate Swing features became constant or time proxies; they must not dominate future models.

### R01.1 engineering fixes

- global Episodes are assigned without copying the 384-column full frame;
- float data compressed to float32 where safe;
- cluster fitting capped with deterministic stratified sampling;
- cluster assignment and gzip report output are batched;
- two-day chunk checkpoint cache added so a crash does not repeat the full 3-hour path build;
- opposite release directions maintain independent Episode lifecycles.

### R01.1 decision

`research_continue`

The research proved predictive path information, not tradable net expectancy.

---

## R01.2 — Stable path explanation and executable-confirmation audit

### Purpose

Explain Cluster 10/4/5 against the continuation control Cluster 8, then test whether fixed causal confirmation rules preserve enough post-entry movement after costs.

### Fixed diagnostic targets

- Cluster 10: core reversal discovery;
- Cluster 4: strong reversal discovery;
- Cluster 5: rare high-conviction discovery;
- Cluster 8: continuation control.

These IDs were selected after R01.1 review. R01.2 is therefore post-hoc development evidence, not a new sealed validation.

### Fixed confirmation rules

- 15 seconds without a new release-direction extreme;
- 5bp / 10bp / 15bp reclaim from the currently known extreme;
- second-push failure.

All confirmations use completed 1-second bars. Entry is the next second open or later. Structural stop uses the extreme known at confirmation plus a fixed 3bp buffer. Same-second stop/target ambiguity is conservative stop-first.

### Fixed execution stress

- round-trip cost: 11bp, 22bp, 33bp;
- entry delay: 1s, 3s, 5s;
- fixed terminal horizons: 60s, 180s, 300s;
- Episode-level day-block bootstrap;
- deterministic replay sampling across cluster, side and frozen period.

### Required output

R01.2 must determine:

1. which non-Swing liquidity/path feature families define the stable clusters;
2. whether reversal-minus-continuation remains positive under day-block uncertainty;
3. how far price usually extends before confirmation;
4. how much structural stop distance confirmation requires;
5. whether any fixed confirmation remains positive in every source period after realistic costs;
6. whether the project should proceed to R02 supervised latent-pool/location and release-outcome models.

### Status

Implementation complete; local full-table and 1-second replay must be run. No R01.2 result is claimed inside the patch.

---

## Frozen abandoned directions

- do not treat every Swing as a stop pool;
- do not use second/minute micro-Swings;
- do not fade every turnover burst;
- do not assume every boundary break is a sweep reversal;
- do not optimize hanging order price, stop or leverage before causal confirmation thickness is proven;
- do not use R01.1 holdout labels as a new sealed proof after selecting Cluster IDs from them.

## Next decision boundary

- If stable path gaps survive but confirmation returns are thin: improve outcome/absorption modeling, not execution parameter grids.
- If at least one fixed rule is positive across periods and directions after 11bp cost: proceed to R02 development, then require a genuinely unseen future holdout.
- If fixed causal confirmation destroys the entire signal: stop before building a trading strategy.


## R01.2 full-run result — 2026-08-05

### Reliable path-level findings

- Primary decision: `RESEARCH_CONTINUE_PATH_SIGNAL_EXECUTION_THIN`.
- Cluster 10 retained a positive reversal-minus-continuation day-block bootstrap gap in both directions and all three frozen source periods.
- Cluster 4 also retained a positive gap in both directions and all periods.
- Cluster 5 DOWN remained positive across periods; Cluster 5 UP validation uncertainty crossed zero and is not uniformly stable.
- Cluster 8 remained a robust continuation/no-fade control with negative gap in both directions and every period.
- Cluster 10 is predominantly a non-Swing high-activity path: high turnover, path travel, realized volatility and range across seconds to 15 minutes.
- Cluster 4 resembles a mature directional release with heavy 15m/60m flow and large prior directional excursion.
- Cluster 5 includes a rare 15m+ unswept-level-confluence subtype, but this remains only one liquidity mechanism and not the project admission rule.

### Initial execution audit result — not final because replay coverage was biased

- Requested replay Episodes: 2,842.
- Complete exact-second replays: 978 (34.41%).
- Every fixed confirmation rule was negative when aggregated at default 11bp round-trip cost; no rule was positive in all three source periods.
- The best broad default combination was still approximately -4.5bp mean net per trade.
- However, missing replay coverage was highly non-uniform: the low-activity continuation Cluster 8 had only about 3% completion, while other clusters had materially higher completion.
- Root cause: R01.2 required every second to be physically present, whereas R01.1 causally interprets short absent seconds as no-trade bars and rejects only longer unsafe gaps. Therefore the first execution table is selection-biased and must be rerun after normalization parity.

### R01.2 replay-quality hotfix

- Replay now reuses the R01.1 short-gap semantics: price is causally carried forward, flow fields are zero for short no-trade seconds, and gaps longer than five seconds remain unsafe and are rejected.
- Per-cluster/side/period completion rates are written into `12_replay_quality.csv`.
- `np.nanquantile` is called only on columns containing finite observations; legitimate all-NaN offsets remain NaN without `RuntimeWarning: All-NaN slice encountered`.
- The warning observed in the first run came from `impact_bp_per_million` at offset -300, where the first-difference metric is intentionally undefined.

### Decision after the first R01.2 run

- Keep the path-learning direction.
- Do not promote any fixed confirmation or trading rule from the first replay.
- Rerun R01.2 after the replay-quality hotfix; only then decide whether the execution signal is truly thin or whether the earlier negative result was materially distorted by missing low-activity seconds.

---

## R01.3 — Absorption completion and remaining-space supervised audit

### Why this stage exists

R01.1 proved that broad liquidity-release paths contain stable reversal-versus-continuation information. R01.2 then showed that five simple fixed confirmations did not translate that information into robust execution: the initial 11bp-cost results were negative, and the first replay also exposed a non-uniform missing-second bias that required normalization parity with R01.1.

R01.3 is the declared final commercial gate for this path family. It does not add more Swing hypotheses or continue a confirmation-parameter grid. It asks whether a supervised causal model can identify the moment when absorption is genuinely complete and enough executable reversal space remains.

### Frozen source and chronology

- upstream discovery universe: R01.1 Cluster 10 / 4 / 5 and continuation-control Cluster 8;
- train: `TRAIN_2023_2024` only;
- calibration and score-threshold freeze: `VALIDATION_2025Q1_Q3` only;
- evaluation-only holdout inside R01.3: `HOLDOUT_2025Q4_2026H1`;
- cluster selection is still post-R01.1 review, so R01.3 is development evidence and not a new sealed proof;
- entry is always the next completed 1-second bar open or later;
- 11bp / 22bp / 33bp round-trip costs and 1s / 3s / 5s delays are reported.

### Causal decision snapshots

For every sampled Liquidity Release Episode, R01.3 creates snapshots at fixed offsets:

`15 / 30 / 45 / 60 / 90 / 120 / 180 / 240 / 300 seconds`.

Each snapshot sees only information available through that second, including:

- current release-direction extension and reclaim from the known extreme;
- seconds since the latest known extreme and number of extreme updates;
- 5s/15s/30s/60s path travel, range and price efficiency;
- turnover/trade-count intensity relative to the pre-event baseline;
- release-aligned Delta share;
- impact per million notional and impact/efficiency decay;
- burst counts, pressure without progress, maximum reclaim and reclaim giveback.

No direct Swing or unswept-Swing feature enters R01.3. Cluster is only the upstream liquidity-path context. This stage therefore cannot collapse back into a Swing-sweep strategy.

### Fixed multi-task targets

R01.3 trains four fixed LightGBM tasks:

1. `absorption_complete_target`: no more than 3bp additional release-direction extension over the next 30 seconds;
2. `tradeable_before_stop_target`: the price reaches 11bp cost plus 15bp net room before the decision-time known-extreme structural stop;
3. regression of remaining additional extension;
4. regression of remaining favorable MFE.

The final causal trade score is the geometric mean of the tradeability probability and absorption-completion probability. The q90 score threshold is fixed on the calibration period. Each Episode may select only its first qualifying snapshot.

### Commercial gate

Promotion requires at least one direction to satisfy all frozen conditions:

- holdout tradeability AUC at least 0.58;
- at least 0.02 holdout AUC uplift over a cluster/side/checkpoint mechanical baseline;
- at least 100 trades in both calibration and holdout at 1s delay / 1x cost;
- mean net at least +3bp and PF at least 1.20 in both calibration and holdout;
- positive top-10-removed mean;
- holdout remains non-negative at 2x cost.

Passing only promotes the direction to a formal R02 strategy backtest and a later genuinely unseen future validation. It is not live approval.

If no side passes, the frozen decision is:

`STOP_LATENT_LIQUIDITY_PATH_V1_EXECUTION_NOT_VIABLE`

This prevents another indefinite sequence of confirmation, stop and threshold patches.

### Implementation status

- implementation complete;
- deterministic source and per-day 1-second snapshot caches included;
- warning-safe numerical paths retained; all-NaN slices are not globally suppressed;
- local full R01.1 tables and 1-second Trade Bar data are required to produce the result;
- no R01.3 performance result is claimed inside the patch.

---

## R01.3 full-run result — 2026-08-07

### Primary decision

`STOP_LATENT_LIQUIDITY_PATH_V1_EXECUTION_NOT_VIABLE`

This decision closes the **post-release confirmation / market-entry branch**.  It does not erase the R01.1 path signal and it does not claim that pre-event liquidity-pool location is unforecastable.

### Prediction retained real information

Holdout `HOLDOUT_2025Q4_2026H1`:

- DOWN tradeability full-model AUC: **0.7161** vs mechanical baseline **0.6869** (+0.0292);
- UP tradeability full-model AUC: **0.7198** vs mechanical baseline **0.6896** (+0.0302);
- DOWN absorption-completion AUC: **0.8337**;
- UP absorption-completion AUC: **0.8300**;
- remaining-extension Spearman: roughly **0.43** in both directions;
- remaining-MFE Spearman: roughly **0.43 DOWN / 0.40 UP**.

Therefore the release/absorption process is not random.  Ranking information survives outside the train period.

### Commercial execution failed decisively

At the frozen q90 calibration threshold, next-second entry and default 11bp round-trip cost:

- validation DOWN: 421 trades, mean **-10.76bp**, win 51.1%, PF **0.41**;
- validation UP: 518 trades, mean **-11.07bp**, win 45.2%, PF **0.37**;
- holdout DOWN: 448 trades, mean **-11.65bp**, win 49.8%, PF **0.37**;
- holdout UP: 449 trades, mean **-11.13bp**, win 43.7%, PF **0.35**.

The selected-entry structural stop remained too far from the late confirmation price (roughly mid-30s to mid-40s bp on average), while remaining MFE was only around 30–37bp.  Waiting for confirmation improved certainty but destroyed the reward/risk geometry.

### R01.3 interpretation

Do **not** rescue this branch by tuning q90, fixed reclaim distance, stop buffer or wait time on the same history.

The useful information is instead carried forward as evidence for a distinct question:

> Can the market path before release forecast **where** latent stop/liquidation/fragile-position liquidity sits and how deep a future sweep is likely to extend, so execution can occur near the pool rather than after the rebound is already underway?

That question is R02 and is a new pre-event spatial branch, not an optimization of the failed R01.3 execution rule.

---

## R02 — Pre-event latent liquidity-pool location and sweep-depth forecast

### Research question

At a causal decision time, before the release occurs, evaluate a continuous price-zone lattice above and below ETH and predict:

1. whether the zone will be touched within 1h / 4h / 12h;
2. conditional on touch, whether a broad R01.1 liquidity-release Episode will occur there;
3. conditional on release, whether it becomes a favorable reversal or continuation;
4. how far release is likely to continue beyond the pool (`sweep_depth_bp`);
5. how much reversal room remains after the extreme.

### Candidate-price universe

R02 does not use a list of Swing hypotheses as its search universe.  It uses a fixed contiguous spatial lattice from roughly current price to +/-500bp.  The lattice is only a numerical discretization of price space, not a stop-location hypothesis.

### Liquidity/path feature families

The pre-event model uses:

- 1-second normalized path/flow context over 15s / 60s / 300s / 900s;
- completed 1-minute macro context over 15m / 1h / 4h / 1d / 3d / 7d;
- turnover intensity, Delta share, trade intensity, price efficiency, path travel, realized volatility, overlap/residency, pressure-without-progress and impact efficiency;
- candidate-zone relationship to nested multi-horizon highs/lows, including whether a zone has remained outside/totally untouched by multiple completed path windows;
- causal turnover accumulated while a zone remains untouched as a fragile-position/position-buildup proxy;
- **supplemental only**: every active 15m / 30m / 1H / 4H / 1D Swing level that has not yet been swept, including very old levels.  No recent-only truncation is allowed.

### Explicit Swing ablation

R02 fits and reports three release-location models:

1. distance/boundary mechanical baseline;
2. full liquidity/path model **without Swing**;
3. full liquidity/path model **with all active 15m+ unswept Swing features**.

The report therefore measures the incremental value of Swing rather than allowing Swing to define liquidity by assumption.

### Frozen chronology

- train: 2023–2024;
- calibration: 2025 Q1–Q3;
- diagnostic holdout: 2025 Q4–2026 H1.

Because R02 was designed after review of earlier holdout evidence, this is development chronology only; it is not a newly sealed validation.

### R02 promotion boundary

Promotion is only to an independent **R02.1 limit-placement study**, never directly to live trading.  The initial gate asks for materially better holdout release-location ranking than a distance-only baseline, favorable-release ranking, sweep-depth rank information, and a useful top-zone subset.

If the long-history price/Trade-Bar baseline retains moderate information but misses the gate, the next allowed action is an independent Range Footprint / OI / Books increment.  Do not patch the result with Swing-only filters or threshold grids.

### R02 implementation freeze — 2026-08-07

- Implementation is complete; no R02 empirical result is claimed inside the patch because the container does not contain the user's full local R01.1 wide tables and 1s Trade Bar history.
- The spatial lattice is a contiguous numerical cover from 0 to 500bp on both sides in 20bp-wide cells (centers 10, 30, ..., 490bp). It is not a Swing or stop-location hypothesis.
- Every 15-minute decision is evaluated only if its complete 12-hour primary future-label window exists. The last 12 hours of the requested research window are label support and are never silently treated as untouched/no-release negatives.
- The full model is explicitly compared against (a) a distance/boundary baseline and (b) a liquidity/path model with every Swing feature removed. Swing therefore has to prove incremental information and cannot define the candidate universe.
- Swing supplement requires the R01.1 lifecycle cache and includes every still-active 15m/30m/1H/4H/1D historical level, including very old levels, until first causal sweep. Missing lifecycle data is a hard setup error rather than a silent no-Swing fallback.
- `--no-cache` now bypasses both final-dataset and per-spatial-chunk caches. Normal runs checkpoint each 14-day spatial chunk for recovery.
- R02 candidate labels are quality-audited so every release-mapped zone must also be touched within the primary horizon, and every retained row must have a complete primary future window.
- Model chronology remains train 2023–2024, calibration 2025 Q1–Q3, diagnostic holdout 2025 Q4–2026 H1. Because R02 was designed after review of earlier holdout evidence, this remains development evidence and is not a new sealed validation.
- Promotion can only reach `R02.1 limit-placement study`; no R02 result can directly authorize live capital.

### R02 full-lattice audit hardening

- Model fitting uses deterministic inverse-probability control sampling to keep the multi-million-row spatial dataset bounded.
- A separate deterministic 5% sample of complete `decision_time x side` lattices retains **all 25 candidate cells** on that side. This audit sample is not used as extra model weight.
- Calibration q90 thresholds, pool-score deciles, price-zone label summaries and Top-zone selection are computed from these complete lattices, so the reported best location is never chosen from a partially sampled set of prices.
- Binary AUC/AP metrics on the model-control sample now use inverse-probability sample weights.
- Promotion additionally requires at least 50 realized release labels in the holdout Top-zone subset, preventing a tiny location subset from passing the gate by chance.
- A price cell is considered touched when the future path enters its near boundary, not only when price reaches the numerical center. The cell center remains a spatial coordinate; near/far boundaries define the actual region.

---

## R02 full-run result — pre-event location baseline

### Quality-gate result

Primary report decision: `BLOCKED_R02_QUALITY_OR_CAUSAL_FAILURE`.

The block came from only **5** `release_implies_primary_touch` inconsistencies among **1,662,429** spatial rows.  Root cause was an exact 12-hour right-edge mismatch: Touch used the completed future-minute horizon while release mapping included Episodes exactly at `t + 720m`.  R02.1 changes the release horizon to the strict interval `(t, t + 720m)` and adds regression tests; no rows are silently ignored.

### Useful pre-event information survived outside train

Holdout `HOLDOUT_2025Q4_2026H1`:

- Touch/arrival AUC: **0.8255 DOWN / 0.8327 UP**.
- Release-on-touch AUC full/path-no-Swing/distance-baseline:
  - DOWN **0.5495 / 0.5465 / 0.4930**;
  - UP **0.5325 / 0.5421 / 0.4536**.
- Favorable-release AUC full/path-no-Swing:
  - DOWN **0.6711 / 0.6692**;
  - UP **0.6688 / 0.6715**.
- Sweep-depth Spearman: **0.2852 DOWN / 0.2570 UP**.
- Reversal-room Spearman: **0.1924 DOWN / 0.2130 UP**.

### R02 interpretation

1. **Arrival/location is highly predictable**, but that is not the same as latent stop/liquidity strength.
2. The broad R01.1 binary `release_within_horizon` label is too permissive once a zone is touched: release prevalence was about **88% DOWN / 87% UP** in holdout.  Binary release therefore does not cleanly represent "there is a large stop/liquidation pool here".
3. Swing has almost no independent holdout value:
   - DOWN release AUC Swing uplift only about **+0.003**;
   - UP full-with-Swing was about **-0.010** worse than no-Swing;
   - favorable-release Swing uplift was similarly near zero / negative.
   Swing remains supplemental only and must not define the pool universe.
4. The original geometric `pool_score = touch x release x favorable` was dominated by arrival probability and selected very near zones (holdout mean roughly **25bp DOWN / 22bp UP**).  That score is therefore retired as a latent-pool-strength definition.
5. High-score favorable releases still showed useful geometry: realized sweep depth roughly ~30bp with post-extreme reversal room roughly ~50–60bp in favorable subsets, supporting a later pre-positioned limit-placement question if true pool strength can be forecast independently of arrival.

### R02 branch decision

Do not proceed directly to hanging-order optimization from the old R02 pool score.

Proceed to **R02.1 conditional pool-strength / release-density deconfounding**:

- Touch/arrival probability is a separate model/output only.
- Pool strength is learned only on zones that were actually touched, so "not reached" is not mislabeled as "no liquidity".
- Multiple realized release Episodes inside a zone are aggregated into continuous strength labels rather than reduced to the first binary release.
- Primary pool-strength score excludes both Touch probability and Swing.
- All 15m+ unswept Swing levels remain only as an explicit incremental ablation.

---

## R02.1 — Conditional pool strength / release density deconfounding

### Research question

Before any future release occurs, can the pre-event liquidity/path state rank **how much liquidity will actually be released if price reaches a candidate zone**, independently of the probability that price reaches that zone?

This is the closest observable historical proxy available for latent stop/liquidation/fragile-position concentration without pretending that actual hidden stop orders are directly visible.

### Multi-label strength definition — no single hand-picked hypothesis

For every touched time-price zone, R02.1 aggregates **all** R01.1 Liquidity Release Episodes inside the strict future horizon `(decision_time, decision_time + 12h)` and retains multiple targets:

- distinct release Episode count;
- cumulative release-density proxy;
- maximum single-Episode release density;
- cumulative Episode size;
- cumulative release score;
- favorable-reversal Episode count/density;
- continuation Episode count/density;
- density-weighted sweep depth;
- density-weighted reversal room.

The primary continuous target is `log1p(cumulative release density)`.  A side-specific high-strength label is frozen from the **TRAIN 2023–2024 touched-zone** distribution only (q80).  Holdout does not redefine the label threshold.

### Deconfounding rules

- **Primary score:** `pool_strength_score = p_high_strength_path_no_swing`.
- Touch probability is not multiplied into or fed into the primary pool-strength score.
- Swing is excluded from the primary model.
- A full model with all active 15m+ unswept Swing features is fitted only for ablation.
- A strict distance+side baseline is fitted independently.
- Untouched zones are reported for arrival diagnostics but are not treated as zero-strength supervision.

### Frozen chronology

- train: 2023–2024;
- calibration: 2025 Q1–Q3;
- development holdout: 2025 Q4–2026 H1.

This is still post-review development evidence, not new sealed validation.

### Promotion boundary

At least one direction must show, in holdout:

- high-strength path/no-Swing AUC >= 0.60;
- >= 0.03 AUC uplift over strict distance+side baseline;
- release-density Spearman >= 0.20 and >= 0.05 uplift over distance baseline;
- favorable-if-release AUC >= 0.62;
- sweep-depth Spearman >= 0.20;
- full-lattice Top-1 selected zone has >=50 touched observations, >=1.5x high-strength lift and >=1.35x realized-density lift versus all touched audit zones.

Passing only promotes to **R02.2 causal limit-placement / depth study**.  It is not live approval.

If moderate conditional strength remains but misses the gate, the next allowed increment is Range Footprint / OI / Funding (and shorter-history Books as an independent increment).  Swing-only rescue is prohibited.

## R02.1 performance hotfix — full-history strength aggregation

### Trigger
A full R02.1 run reached:

```text
[source] R02 spatial rows=2,049,259 Episodes=1,024,890
[stage] aggregate all future release Episodes into touched-zone strength labels
```

and then remained CPU/RAM-bound without progress output.

### Root cause
The original R02.1 labeler was semantically correct but computationally inappropriate for full history. For each decision timestamp and each side it repeatedly:

1. sliced the next 12h of Episodes with pandas;
2. deep-copied that future slice;
3. rebuilt an Episode x all-zone distance matrix;
4. copied per-zone candidate frames again;
5. wrote results row-by-row with ``DataFrame.at``.

With ~2.05M spatial rows and ~1.02M Episodes, this creates tens of millions of decision/Episode comparisons plus hundreds of thousands of Python/Pandas object operations. CPU and memory activity therefore looked like a hang even though the process was still computing.

### Hotfix
The exact R02.1 causal/statistical semantics are frozen. Only the aggregation implementation changes:

- bounded decision-time chunks (default 1,024 decisions);
- NumPy ``searchsorted`` for exclusive future windows ``(t, t+12h)``;
- vectorized nearest-zone mapping with exact midpoint tie behavior preserved;
- ``bincount`` / ``maximum.at`` / ``minimum.at`` reductions for Episode count, density, size, score, favorable/continuation density, weighted sweep depth, weighted reversal room and first-release time;
- no per-decision pandas ``copy`` and no Episode x 25 matrix;
- one stable row grouping pass for sampled/full-lattice spatial rows;
- progress bar for the entire strength-label aggregation;
- shallow frame copies where R02.1 only appends target columns.

No candidate, Touch, Release, Strength, Swing, period, model or gate definition changed.

### Equivalence / performance evidence
- Added a randomized multi-decision, dual-side, partially sampled-lattice reference test comparing the optimized labeler against the original pandas semantics field-by-field.
- Exclusive 12h right-edge behavior remains covered.
- A synthetic realistic-density benchmark covering ~52 days / 40,096 Episodes / 87,413 retained spatial rows completed the optimized strength aggregation in ~0.28s in the validation container. This is a relative engineering benchmark only; real Windows runtime depends on hardware and source-cache layout.

### Tests after hotfix
- R01 -> R02.1 focused regression: 63 passed.
- All AI Research: 305 passed.
- Data Feed + Research Common: 23 passed.
- R02.1 strict RuntimeWarning/FutureWarning: 8 passed.
- compileall: PASS.
- Full repository collection: 631 tests discovered but still blocked by the same five pre-existing Liquidity / Analyze Tool missing modules.
- Import-boundary test still fails on pre-existing legacy research-import violations; this hotfix adds no new research-import boundary coupling.

---

## R02.1 full-run result — conditional strength label exposed distance/exposure bias

### Formal report decision

`CONTINUE_R02_1_WITH_RANGE_FOOTPRINT_OI_INCREMENT`

The report itself passed all causal/source gates after the exclusive 12h right-edge fix.  However, review of the realized holdout statistics showed that the **absolute pool-strength target is not a clean latent-liquidity label** and should not be rescued with more features before fixing the modeling question.

### Holdout model results (`HOLDOUT_2025Q4_2026H1`)

**DOWN**

- High-strength AUC path-no-Swing / full-with-Swing / distance baseline: **0.5624 / 0.5581 / 0.6715**.
- Continuous density Spearman path-no-Swing / full-with-Swing / distance baseline: **-0.0130 / -0.0106 / -0.2847**.
- Favorable-if-release AUC path-no-Swing / full-with-Swing: **0.6519 / 0.6502**.
- Continuation AUC path-no-Swing / full-with-Swing: **0.5768 / 0.5776**.
- Sweep-depth Spearman: **0.3645**.
- Full-lattice Top-1 selected zone: **967** touched observations; high-strength lift **1.442x**; realized release-density lift **1.606x**; mean selected distance about **79bp**.

**UP**

- High-strength AUC path-no-Swing / full-with-Swing / distance baseline: **0.5177 / 0.5183 / 0.5876**.
- Continuous density Spearman path-no-Swing / full-with-Swing / distance baseline: **-0.0529 / -0.0528 / -0.3356**.
- Favorable-if-release AUC path-no-Swing / full-with-Swing: **0.6639 / 0.6632**.
- Continuation AUC path-no-Swing / full-with-Swing: **0.5560 / 0.5558**.
- Sweep-depth Spearman: **0.3032**.
- Full-lattice Top-1 selected zone: **928** touched observations; high-strength lift **1.298x**; realized release-density lift **1.398x**; mean selected distance about **90bp**.

### Swing conclusion — downgraded to archive/supplement only

Swing did not add material holdout information:

- DOWN strength AUC Swing uplift: about **-0.0043**.
- UP strength AUC Swing uplift: about **+0.0006**.
- Favorable-release AUC was also slightly worse with Swing on both sides.

Tree feature-importance shares for Swing are not treated as proof of value because the direct holdout ablation is the relevant test.  All causally confirmed 15m+ unswept Swing levels may remain in storage for future ablation, including very old levels, but **Swing is not the pool definition, candidate universe, gate, or primary model family**.

### Why the R02.1 absolute-strength target is retired

Three structural problems were identified after the full run:

1. **Exposure-time / distance contamination.**  Strength was accumulated from decision time through the whole 12h horizon.  A 20bp zone touched after 30 minutes could accumulate releases for ~11.5 further hours, while a 300bp zone first touched after 11 hours had only ~1 hour remaining.  Near zones therefore received more opportunity to accumulate density even when the underlying first-touch pool was not stronger.
2. **Repeated-visit mixing.**  A zone could contain a favorable release on one visit and a continuation release on another later visit.  This is why Top-1 holdout zones could show both favorable-any and continuation-any rates around 70%+.  That is useful as an activity label but not as the label for the first sweep that a passive order would face.
3. **Absolute-threshold drift.**  The TRAIN q80 high-strength label had ~20% positives by definition, but Validation/Holdout positive prevalence rose to roughly **41–43%**, echoing the earlier Q70 failure mode.  A fixed absolute activity threshold is therefore not a stable definition of spatial pool quality across changing market-activity regimes.

### What is retained from R02.1

The stage is not discarded.  Three capabilities remain valuable:

- **Cross-sectional location signal:** despite the bad absolute target, the model-selected Top-1 zone had realized density lift **1.61x DOWN / 1.40x UP** versus all touched audit zones.
- **Favorable-vs-continuation information:** holdout favorable AUC **0.652 DOWN / 0.664 UP**.
- **Sweep-depth information:** holdout Spearman **0.364 DOWN / 0.303 UP**.

These results motivate changing the modeling problem before adding Range/Footprint/OI.

---

## R02.2 — First-touch relative liquidity ranking

### Why R02.2 exists

R02.2 replaces the contaminated question:

> "Is this zone absolutely high strength over the next 12 hours?"

with the trading-relevant question:

> **"At this decision time and on this side of price, which candidate zone will release the most liquidity when price first touches it?"**

No absolute q80/q90 pool-strength threshold is used in the primary ranking problem.

### Fixed first-touch label design

For each complete 25-zone lattice on each side:

1. locate the first future 1m bar whose path enters the zone;
2. refine that minute to the **exact 1-second first-touch bar** using local 1s Trade Bars;
3. anchor every label on that exact first-touch second;
4. evaluate equal post-touch windows of **30 / 60 / 180 / 300 seconds** for every near or far zone.

Primary ranking target:

- cumulative R01.1 release-density proxy in the **first 180 seconds after exact first touch**.

Secondary diagnostics:

- release Episode count / max density / Episode size / release score;
- favorable and continuation density;
- density-weighted sweep depth and reversal room;
- post-touch 1s notional / trades / absolute-Delta ratios versus the preceding 60-second local baseline.

A 20bp zone and a 300bp zone therefore receive exactly the same post-touch observation duration.  The old exposure-time advantage disappears.

### Relative ranking, not absolute classification

R02.2 trains side-specific LightGBM LambdaRank models on **within-snapshot touched-zone relative relevance**:

- group = `decision_time x zone_side`;
- primary model = full liquidity/path features **without Swing**;
- full-with-Swing model = supplemental ablation only;
- mechanical baseline = nearest-zone / distance ranking;
- no Touch probability is multiplied into the rank score.

Training chronology remains:

- TRAIN: 2023–2024;
- Validation: 2025 Q1–Q3;
- development Holdout: 2025 Q4–2026 H1.

### Why complete lattices are mandatory

R02.2 uses only the deterministic R02 complete-lattice audit groups.  Every retained decision-time/side group contains all 25 price cells.  It does **not** train/evaluate the spatial ranker on a partially sampled lattice, so "Top-1" always means best among the actual full candidate surface.

### Planned diagnostics / promotion boundary

R02.2 reports:

- mean/median within-group Spearman;
- NDCG@1 / NDCG@3;
- pairwise ordering accuracy;
- Top-1 exact-touch rate;
- Top-1 first-touch density lift versus all touched zones;
- within-touched Top-1 density lift;
- probability that the realized strongest touched zone appears in predicted Top-3;
- Swing ablation;
- fixed-window 30/60/180/300-second release/flow profiles;
- distance profile after first-touch deconfounding.

Research-only promotion to `R02.3 limit placement / sweep-depth study` requires at least one side to show in holdout:

- mean within-group Spearman >= **0.15** and better than distance baseline;
- Top-1 first-touch density lift >= **1.25x** with >=50 touched Top-1 observations;
- realized strongest touched zone appears in predicted Top-3 >= **30%**.

Failure does not permit threshold grids or Swing rescue.  If modest relative ranking remains, the only allowed next increment is independent Range / Footprint / OI evidence on the same ranking target.

### Engineering / performance rules

- first touch is located with 1m cumulative extrema and refined only inside the touched minute with 1s bars; R02.2 does **not** scan every second of every 12h window for every zone;
- 1s flow labels use prefix sums for O(1) post-touch window aggregation;
- replay is chunk-cached and resumable;
- the 1s normalizer materializes only the six columns needed for first-touch labeling rather than the full R01 feature surface;
- fixed-window Episode labels use time-index search, not DataFrame row-by-row scanning of the full history.

This stage does not place orders, choose stop distances, or claim a live strategy.  It changes the **label and ranking problem first**, exactly because the R02.1 absolute label was structurally contaminated.

### R02.2 implementation validation and speed audit

Implementation was finalized on 2026-08-07 with the following verification:

- R01 through R02.2 focused regression: **71 passed**.
- All `tests/ai_research`: **313 passed**.
- AI Research + Data Feed + Research Common under strict `RuntimeWarning` / `FutureWarning`: **336 passed**.
- R02.2 dedicated tests: **8 passed** under full `-W error`.
- CLI `--help`: PASS.
- `compileall` for the new package / CLI: PASS.
- Import-boundary scan still reports **155 repository-history violations**; **R02.2 adds 0**.
- Full repository collection finds **639 tests** and is still blocked by the same **5 pre-existing Liquidity / Analyze Tool missing-module collection errors**; R02.2 adds none.

Algorithm-speed audit:

- first-touch discovery does not scan each zone over a 12-hour 1-second path; it uses 1m cumulative extrema to find the first touched minute and refines only that minute at 1s resolution;
- post-touch 30/60/180/300-second flow labels use NumPy prefix sums;
- Episode aggregation uses indexed time search / vectorized spatial mapping rather than row-wise Pandas loops;
- a synthetic engineering benchmark with a dense ~650k-second path processed 25,000 exact-touch lookups in sub-second order in the development environment, and a 100k-Episode / 25k-zone aggregation completed in low-single-digit seconds. These are implementation benchmarks only, not promises for a user's machine.

No empirical R02.2 market result is recorded yet. The next work-log update must record the actual report result before changing the target or adding Range / Footprint / OI.


---

## R02.2 full-run result — raw first-touch density ranking rejected; target corrected before adding new data

### Formal report status

The generated R02.2 report returned:

`BLOCKED_R02_2_QUALITY_OR_CAUSAL_FAILURE`

This block did **not** mean that 8,282 rows contained true future leakage. Post-run code/report review separated two different issues:

1. **1-second bar timestamp semantics were audited incorrectly.** `first_touch_time` is the start timestamp of a completed 1s bar. A bar stamped exactly at `decision_time` covers `[decision_time, decision_time+1s)` and is only knowable at `decision_time+1s`. The old audit compared `first_touch_time <= decision_time`, falsely flagging 8,282 rows. R02.2/R02.3 now audit `first_touch_available_time = first_touch_time + 1s` instead.
2. **22 exact-touch rows disagreed with the older R02 1m `touch_720m` cache.** This is only about ~0.022% of the ~102k exact first-touch rows, but it is not ignored. R02.3 quarantines every such row from training/evaluation and reports the count explicitly. No silent relabeling is allowed.

### Actual R02.2 dataset

- complete-lattice rows: **310,250**;
- decision-time x side groups: **12,410**;
- exact 1s first-touch rows: **102,230**;
- complete fixed-window first-touch labels: **98,246**;
- rankable rows: **80,173**;
- rankable groups: **7,467**.

### Raw first-touch release-density ranking — FAIL

Primary target was cumulative R01.1 release density during the first fixed 180 seconds after exact first touch.

Holdout mean within-group Spearman:

**DOWN**
- PATH_NO_SWING: **0.0413**
- FULL_WITH_SWING: **0.0410**
- nearest-distance baseline: **0.1739**

**UP**
- PATH_NO_SWING: **0.0745**
- FULL_WITH_SWING: **0.1025**
- nearest-distance baseline: **0.1560**

Training Spearman was much higher (~0.55 DOWN / ~0.53 UP) but collapsed in Validation/Holdout. Raw density ranking is therefore not a stable latent stop-pool objective and must not be rescued with LightGBM tuning or Swing filters.

### Why raw density remained the wrong target

Even after equalizing the post-touch observation window, the realized 180s density profile remained mechanically distance-dependent. In Holdout, the nearest 10bp zone had the highest raw release density, while farther zones increasingly showed higher favorable-reversal rates. A nearest-distance baseline therefore wins the raw-density ranking task for a trivial reason: very near price regions naturally contain more ordinary microstructure activity.

The research objective is **not** "which zone has the most ordinary post-touch trading activity?" It is:

> which zone releases *abnormally large liquidity relative to what is normal for that distance*, and does that release have reversal quality rather than acceptance/continuation quality?

### Important retained R02.2 signal

Although raw density ranking failed, the No-Swing path model selected materially farther zones with better reversal tendency than the nearest-distance baseline.

Holdout Top-1 diagnostics:

**DOWN PATH_NO_SWING**
- selected mean distance: ~**131.6bp**;
- top1 touched: **788**;
- raw density lift vs all touched zones: **1.38x**;
- favorable-release rate: **12.31%**;
- continuation rate: **6.60%**.

**DOWN nearest-distance baseline**
- mean distance: **10bp**;
- favorable: **7.39%**;
- continuation: **7.23%**.

**UP PATH_NO_SWING**
- selected mean distance: ~**182.7bp**;
- top1 touched: **643**;
- raw density lift: **1.38x**;
- favorable: **11.66%**;
- continuation: **8.55%**.

**UP nearest-distance baseline**
- mean distance: **10bp**;
- favorable: **6.74%**;
- continuation: **7.68%**.

This suggests the path model may contain information about *liquidity-release quality* even though raw activity density itself is the wrong ranking target.

### Swing conclusion after R02.2

Swing remains non-primary.

- DOWN Holdout raw-density Spearman: No-Swing **0.0413**, With-Swing **0.0410** — no uplift.
- UP Holdout: No-Swing **0.0745**, With-Swing **0.1025**, but Validation moved in the opposite direction (No-Swing **0.0674**, With-Swing **0.0627**), so the apparent Holdout uplift is not cross-period stable.

Therefore all causally confirmed 15m+ unswept Swing levels remain stored only for ablation. Swing is not the pool definition, admission gate, ranking target, or primary model family.

### Frozen R02.2 decision after review

- raw first-touch release-density ranking: **RETIRED AS PRIMARY TARGET**;
- R02.2 post-run confirmation does **not** authorize Range/Footprint/OI feature additions on the old raw-density target;
- first fix the target with distance normalization and separate reversal quality;
- R01.3 post-confirmation market-entry branch remains stopped.

---

## R02.3 — Distance-normalized Excess Liquidity + Reversal Quality Ranking

### Why R02.3 exists

R02.3 changes the modeling question before adding any new data family.

The primary question is now:

> **At this decision time, which price zone would release unusually large liquidity *relative to what is normal for that distance* when first touched?**

A second independent question is:

> **Conditional on a release occurring, which zone is more reversal-dominant than continuation-dominant?**

Sweep depth / reversal room remain separate geometry tasks.

### Source reuse and performance design

R02.3 does **not** replay all 1m/1s data again. It reuses the completed R02.2 exact-first-touch cache.

The only new full-history transformation is vectorized:

- TRAIN-only side x distance robust statistics;
- merge expected density / robust scale onto rows;
- vectorized excess / reversal target construction;
- LightGBM ranking/regression.

There is no row-by-row 12h scan, no wide Episode x Zone matrix, and no repeated pandas slicing loop in R02.3.

### Exact first-touch quality handling

R02.3 interprets 1s time causally:

`first_touch_available_time = first_touch_bar_start + 1 second`

Rows with an old R02 `touch_720m` mismatch are **quarantined**, not silently corrected. They cannot enter R02.3 train/evaluation. The report records the mismatch count and separately proves that zero mismatched rows were used.

### Primary Excess Liquidity target

For each side x distance, using only `TRAIN_2023_2024` exact-touch rows:

1. transform realized fixed-180s release density with `log1p`;
2. freeze TRAIN median expected log density;
3. freeze TRAIN IQR as a robust scale (with a small fixed floor);
4. define:

`excess_liquidity_z = (log1p(actual_density) - train_distance_median) / train_distance_IQR`

No Validation/Holdout statistics are used to define this target.

The primary ranker also excludes the raw `zone_distance_bp` feature itself. Distance remains only as explicit mechanical near/far baselines. This prevents the model from simply relearning the nuisance distance curve that the target normalization removed.

### Separate Reversal Quality target

Conditional on at least one fixed-window release Episode:

`reversal_quality_target = log1p(favorable_density) - log1p(continuation_density)`

This avoids mixing "how much unusual liquidity is here?" with "what happens after it releases?".

### Separate sweep geometry

R02.3 keeps two independent No-Swing regressions:

- density-weighted 180s sweep depth;
- density-weighted 180s reversal room.

These are not multiplied into Excess Liquidity and are not allowed to define a favorable pool retrospectively.

### Swing boundary

Primary Excess and Reversal models exclude Swing. A full-with-Swing model is trained only as an ablation on the identical targets.

Swing may only earn renewed attention if it creates stable Validation **and** Holdout uplift. One-period uplift is not enough.

### Evaluation

R02.3 reports, by period and side:

- within-group Spearman / NDCG@1 / NDCG@3 / pairwise ordering for Excess Liquidity;
- the same ranking metrics for Reversal Quality;
- nearest-distance and farthest-distance mechanical baselines;
- Top-1 realized Excess Z and `(1+density)/(1+expected_density)`;
- favorable / continuation rates of selected zones;
- realized strongest Excess zone appearing in predicted Top-3;
- sweep-depth / reversal-room Spearman and MAE;
- distance-vs-raw-density Spearman versus distance-vs-Excess-Z Spearman, explicitly checking whether normalization actually removed the distance bias;
- Swing ablation for both ranking tasks.

### Research-only promotion boundary

At least one side must clear all major Holdout gates before any passive-order study:

- Excess PATH_NO_SWING mean group Spearman >= **0.10** and better than both near/far distance baselines;
- selected Top-1 mean density-vs-expected ratio >= **1.20** with >=50 touched Top-1 rows;
- realized strongest Excess zone appears in predicted Top-3 >= **30%**;
- Reversal Quality PATH_NO_SWING mean group Spearman >= **0.08**;
- Sweep Depth Spearman >= **0.20**.

If cleared, promotion is only to an R02.4 causal passive-limit / sweep-geometry study. It is **not** live approval.

If the corrected target has modest but real signal, only then may Range / Footprint / OI be added as independent feature-family increments. No Swing rescue, q-grid, distance-grid or threshold hunting is allowed.

### R02.3 implementation validation (before real full-history run)

- R02.3 dedicated synthetic/causal tests: **6 passed** under `-W error`.
- R02.2 + R02.3 focused tests: **15 passed** under `-W error`.
- R01 through R02.3 focused regression: **78 passed** under `-W error`.
- synthetic end-to-end report generation: PASS.
- New model feature audit caught and removed `robust_scale` from the feature set; normalizer outputs are labels/report metadata only.
- Primary model feature audit also removes raw `zone_distance_bp`; distance is benchmark-only.

No empirical R02.3 market result is recorded yet. The next cumulative update **must record the actual R02.3 report result before changing the target, adding Range/Footprint/OI, or proceeding to limit placement**.

### R02.3 final engineering validation before delivery

- R02.3 dedicated causal/synthetic tests: **6 passed** under full `-W error`.
- R02.2 + R02.3 focused tests: **15 passed** under full `-W error`.
- R01 through R02.3 focused regression: **78 passed** under full `-W error`.
- all `tests/ai_research`: **320 passed** under full `-W error`.
- `tests/data_feed tests/research_common`: **23 passed** normally and **23 passed** with `RuntimeWarning`/`FutureWarning` promoted to errors.
- Running those unrelated Data Feed/Common tests with *every* warning promoted to an error exposes two pre-existing SQLite `ResourceWarning` / unclosed-connection warnings; R02.3 does not modify those modules and does not suppress those warnings.
- R02.3 CLI `--help`: PASS.
- R02.3 package/CLI compile check: PASS.
- Import-boundary audit: repository has **155** pre-existing unexpected research/backtest couplings; **R02.3 adds 0**.
- Full repository collection: **646 tests collected**, then collection remains blocked by the same **5** pre-existing missing Liquidity / Analyze Tool modules.
- Performance principle retained: target generation is vectorized and reuses the R02.2 exact-first-touch cache; no historical 1m->1s replay is repeated.

No empirical R02.3 market result exists at delivery time. The real R02.3 `gpt_review_pack.zip` must be reviewed and appended here before any R02.4, Range/Footprint/OI feature expansion, or target change.

---

## R02.3 full-run result — median/IQR distance normalization failed under zero inflation

### Formal quality status

R02.3 completed with **all causal audits passing**, but the intended `Distance-normalized Excess Liquidity` label did not actually remove distance contamination.

This is a statistical target-construction failure, not a future-function or source-quality failure.

### Exact normalizer failure

The TRAIN-only `side x zone_distance_bp` table has 50 buckets (25 distances x 2 sides):

- `expected_density == 0` in **50 / 50** buckets;
- `IQR == 0` in **34 / 50** buckets;
- those zero-IQR buckets fell back to the fixed `0.1` scale floor.

Because fixed-180s first-touch Release Density is heavily zero-inflated, the TRAIN median of `log1p(density)` is almost always exactly zero. Consequently many R02.3 `excess_liquidity_z` values are effectively just a scaled transform of raw density, not a true distance-conditioned abnormal-liquidity residual.

### Out-of-sample distance contamination returned

Distance-vs-Excess-Z Spearman:

- Validation DOWN: **-0.1709**;
- Holdout DOWN: **-0.1843**;
- Validation UP: **-0.1547**;
- Holdout UP: **-0.1586**.

TRAIN appeared near neutral only because TRAIN itself defined the median/IQR normalizer.

Mean Excess Z drift also remained large:

- DOWN: TRAIN **2.64** -> Validation **5.44** -> Holdout **5.94**;
- UP: TRAIN **2.00** -> Validation **4.06** -> Holdout **4.20**.

Frozen conclusion: **R02.3 did not prove that distance bias was removed. Do not add Range/Footprint/OI to this failed target.**

### R02.3 No-Swing Excess ranking

PATH_NO_SWING mean within-group Spearman:

**DOWN**
- Validation: **0.0968**;
- Holdout: **0.0685**.

**UP**
- Validation: **0.0504**;
- Holdout: **0.0293**.

The signal is weak and cannot be interpreted as true abnormal-pool ranking because the target itself remained distance-contaminated.

### R02.3 Reversal Quality retained

PATH_NO_SWING Reversal Quality ranking was materially more stable:

**DOWN**
- Validation: **0.1740**;
- Holdout: **0.1697**.

**UP**
- Validation: **0.1478**;
- Holdout: **0.1884**.

However, far-distance mechanical baselines remained stronger in Holdout. Therefore the retained question is not merely "can the model predict reversal?" but:

> can liquidity/path structure predict **reversal quality beyond what is mechanically normal for that distance and current market activity?**

### R02.3 Sweep Geometry retained

Holdout:

- DOWN Sweep Depth Spearman: **0.2652**;
- UP Sweep Depth: **0.1662**;
- DOWN Reversal Room: **0.2594**;
- UP Reversal Room: **0.2570**.

These remain useful independent geometry tasks if a real spatial pool-location residual signal survives later stages.

### R02.3 Swing conclusion

Swing remains non-primary and provides no stable cross-period increment.

- DOWN Holdout Excess: No-Swing **0.0685**, With-Swing **0.0472**;
- UP Holdout Excess: No-Swing **0.0293**, With-Swing **0.0440**, but Validation moved in the opposite direction (**0.0504 vs 0.0428**);
- DOWN Holdout Reversal: No-Swing **0.1697**, With-Swing **0.1650**;
- UP Holdout Reversal: No-Swing **0.1884**, With-Swing **0.1790**.

All 15m+ unswept Swing inventory may remain stored for identical-target ablation only. No Swing gate, target, candidate universe or rescue is allowed.

### R02.3 causal status

All causal checks passed.

- 1s first-touch timestamps are audited by `available_time = bar_start + 1s`;
- old R02 exact-touch disagreements remain quarantined and **0** quarantined rows entered R02.3 train/evaluation;
- future labels, normalizer outputs, raw distance and Swing were excluded from the primary ranker.

### Frozen post-run decision

- retire R02.3 median/IQR distance normalization;
- do not add new data families yet;
- proceed to **R02.3.1 Zero-Inflated Hurdle Nuisance Residualization**;
- residualize both Pool Strength and Reversal Quality against mechanical distance/activity nuisance;
- retain No-Swing as primary and keep geometry separate.

---

## R02.3.1 — Zero-inflated Hurdle Nuisance Residualization + Reversal Residual Ranking

### Why this stage exists

R02.3.1 fixes one specific failure before opening a new research direction:

> first-touch Release Density has a large point mass at zero, so median/IQR normalization cannot estimate a useful expected level.

The target is now generated by a two-part **hurdle nuisance model** rather than a zero-median lookup table.

### Frozen nuisance family

The nuisance model is deliberately mechanical and narrow. It may use only:

- raw `zone_distance_bp`;
- calendar/session derived causally from `decision_time`;
- broad notional/trade-count/realized-vol/range activity features that are constant across all zones in the same `decision_time x side` group.

It may not use:

- Swing;
- zone-specific boundary/path structure;
- future labels;
- first-touch outcomes as features.

Every activity nuisance feature is explicitly audited to be group-level constant.

### Hurdle expectation

Using the fixed 180-second first-touch label window:

1. binary nuisance head estimates `P(release > 0 | distance + broad activity)`;
2. conditional magnitude nuisance head estimates `E[log1p(density) | release > 0, distance + broad activity]`;
3. nuisance expected density is `P(release) * smearing-adjusted E[density | release]`;
4. primary pool target becomes:

`Excess Liquidity Residual = log1p(actual density) - log1p(nuisance expected density)`.

R02.3.1 also reports release-probability surprise and positive-density magnitude residual separately so a future review can see whether any result is coming from release occurrence or release size.

### Reversal residualization

Raw reversal quality remains:

`log1p(favorable density) - log1p(continuation density)`.

A separate nuisance regressor estimates the normal reversal quality for the same distance/activity state. The primary reversal target becomes:

`Reversal Quality Residual = raw reversal quality - nuisance expected reversal quality`.

This is necessary because R02.3 showed that far distance by itself can predict reversal tendency. A path model is only interesting if it adds information **beyond** that mechanical effect.

### Strong TRAIN anti-overfit rule

TRAIN nuisance predictions are **not** fitted in-sample.

Default chronology:

- first six TRAIN months are nuisance warm-up and cannot train/evaluate residual rankers;
- subsequent TRAIN months are predicted in six-month forward blocks;
- every block is predicted by nuisance models trained only on strictly earlier TRAIN rows with a 13-hour purge covering the 12h touch horizon plus label tail;
- Validation/Holdout use one nuisance family frozen on all 2023-2024 TRAIN.

Therefore a TRAIN row may enter residual-ranker learning only if its nuisance expectation is `TRAIN_EXPANDING_OOS`.

### Primary residual rankers

Primary PATH_NO_SWING rankers exclude:

- raw distance;
- all nuisance activity features;
- all nuisance predictions and residual-target metadata;
- Swing;
- all future labels.

They retain zone-specific causal liquidity/path structure such as historical-boundary relationships and position-buildup proxies, because that is the candidate spatial edge being tested *after* mechanical nuisance removal.

A FULL_WITH_SWING model is trained only as an identical-target ablation.

### Hard residualization quality gate

Before any ranking result is interpreted, Validation and Holdout must prove that nuisance removal actually worked.

The report checks:

- distance vs raw density;
- distance vs Excess Residual;
- distance vs raw reversal quality;
- distance vs Reversal Residual;
- actual vs nuisance-expected release rate / density;
- non-degenerate positive nuisance expected density.

If future residual-distance correlations exceed the frozen tolerances, decision is:

`BLOCKED_R02_3_1_RESIDUALIZATION_NOT_REMOVED`

and Range/Footprint/OI additions remain prohibited.

### Research-only promotion boundary

Only after residualization quality passes may a side promote to R02.4. It must show, in both Validation and Holdout:

- PATH_NO_SWING Excess Residual Spearman >= **0.08**;
- PATH_NO_SWING Reversal Residual Spearman >= **0.08**;
- Holdout Excess Residual better than nuisance/near/far mechanical baselines;
- Holdout Top-1 actual/nuisance-expected density ratio >= **1.20** with >=50 touched selections;
- realized strongest residual zone in predicted Top-3 >= **30%**;
- Holdout Sweep Depth Spearman >= **0.20**.

Promotion is only to `R02.4 causal passive-limit / sweep-geometry research`, never to live trading.

If residualized No-Swing signal is weaker but stable across Validation/Holdout (>=0.04), only then may Range/Footprint/OI be tested as independent feature-family increments on the frozen residual targets.

### Performance design

R02.3.1 reuses the completed R02.2 exact-first-touch cache and does not repeat the historical 1m->1s replay.

- target construction is vectorized;
- nuisance fits use a small fixed number of six-month expanding blocks;
- LightGBM uses all CPU cores;
- long nuisance fitting has a project-standard progress reporter;
- no row-wise historical 12h scan, no repeated wide DataFrame copy loop, no Episode x Zone full matrix.
- quality-control metadata (`r02_touch_consistent`, prior-stage eligibility flags, split-purge flags) are explicitly excluded from both residual and geometry feature schemas and locked by regression tests.

### Pre-run engineering validation

- R02.3.1 dedicated zero-inflation / causal / residual tests: **8 passed** under full `-W error`;
- R01 through R02.3.1 focused regression: **86 passed** under full `-W error`;
- all `tests/ai_research`: **328 passed** under full `-W error`;
- `tests/data_feed tests/research_common`: **23 passed** normally and **23 passed** with RuntimeWarning/FutureWarning strict;
- CLI and compile checks: PASS.
- import-boundary audit: repository historical unexpected violations **155**, R02.3.1 additions **0**;
- full-repo collection: **654 tests collected**, still blocked only by the same 5 pre-existing Liquidity / Analyze Tool missing modules.
- cumulative-patch clean-baseline replay: fresh `CoinBacktest(8)` + **only** the R02.3.1 cumulative patch reproduced **86 focused / 328 AI Research / 23 Data Feed+Common**, CLI+compile PASS, Import Boundary new violations **0**, and the same 5 historical collection blockers.

No empirical R02.3.1 market result exists at delivery time. The next cumulative update **must append the real R02.3.1 report before any R02.4, feature-family expansion, target change or stop decision**.

### R02.3.1 real-run result — 2026-08-08

The first full R02.3.1 market run completed and the formal decision was:

`BLOCKED_R02_3_1_RESIDUALIZATION_NOT_REMOVED`

All **17/17 causal checks passed**. The block is statistical, not a lookahead/timestamp failure.

#### Excess Liquidity residualization failed the hard distance gate

Validation/Holdout absolute distance correlations (raw density -> Excess Residual):

- DOWN Validation: **0.1482 -> 0.0886** — improved and below the 0.12 limit;
- DOWN Holdout: **0.1611 -> 0.1362** — improved but still above the limit;
- UP Validation: **0.1354 -> 0.1519** — worsened;
- UP Holdout: **0.1729 -> 0.2074** — materially worsened.

Therefore the current Excess Residual cannot be interpreted as distance-clean abnormal latent liquidity.

#### Reversal nuisance removal itself worked

Distance correlation (raw reversal -> Reversal Residual):

- DOWN Validation: **0.2706 -> 0.0619**;
- DOWN Holdout: **0.2587 -> 0.0967**;
- UP Validation: **0.2387 -> 0.0666**;
- UP Holdout: **0.2448 -> 0.0830**.

This confirms that nuisance residualization as a general idea is viable. However, after removing the mechanical distance component, most of the previous No-Swing reversal ranking edge disappeared:

- Holdout DOWN Reversal Residual PATH_NO_SWING Spearman: **0.0024**;
- Holdout UP: **0.0696**.

The older ~0.15-0.19 raw Reversal Quality ranking therefore contained substantial mechanical distance information.

#### Frozen nuisance model drift is severe

Release-hurdle ROC AUC:

- TRAIN DOWN/UP: **0.6922 / 0.6874**;
- Validation DOWN/UP: **0.5117 / 0.5110**;
- Holdout DOWN/UP: **0.5124 / 0.4964**.

Actual release rate also shifted from roughly **23-24% in TRAIN** to **42-45% in Validation/Holdout**, while frozen predicted means remained roughly **25-29%** in future periods. Raw expected-density calibration ratios were **1.45-1.89x actual/predicted** out of sample.

This is evidence that the normal release background learned from 2023-2024 is not stable enough by itself for 2025-2026.

#### PATH ranking after current residualization is weak

Holdout PATH_NO_SWING mean within-group Spearman:

- Excess Residual DOWN: **0.0279**;
- Excess Residual UP: **0.0050**;
- Reversal Residual DOWN: **0.0024**;
- Reversal Residual UP: **0.0696**.

These values must not be interpreted as a clean spatial-edge verdict because the Excess target failed its residualization gate.

#### Sweep geometry remains retained

Holdout geometry remains the strongest stable evidence:

- DOWN Sweep Depth: **0.2456**;
- UP Sweep Depth: **0.1704**;
- DOWN Reversal Room: **0.2591**;
- UP Reversal Room: **0.2643**.

Validation is similar: Sweep Depth **0.2478 / 0.1847**, Reversal Room **0.2740 / 0.2408** for DOWN/UP.

These tasks remain frozen and independent. They do not rescue the failed Excess target.

#### Swing remains supplemental only

Holdout Swing uplift remained small:

- Excess DOWN **+0.0071**;
- Excess UP **+0.0178**;
- Reversal DOWN **+0.0018**;
- Reversal UP **+0.0027**.

Validation signs were not consistently positive. No Swing-centered rescue is allowed.

#### New code-review finding after the real run

R02.3.1 estimated a positive-release model on `log1p(density)`, but the primary Excess target was constructed as:

`log1p(actual density) - log1p(P(release) * E[density | release])`.

For zero-inflated Y this is not the same target scale as:

`E[log1p(Y) | X] = P(Y>0|X) * E[log1p(Y) | Y>0,X]`.

In addition, the positive-log nuisance head used Huber loss, which estimates a robust conditional location rather than the conditional mean required by the expectation identity.

Therefore R02.3.1's failed Excess residualization does **not** yet prove the latent-liquidity spatial idea is dead. It first requires a bounded target-consistency audit.

### R02.3.1b implementation freeze — Hurdle Target Consistency + Residual Distance Audit

R02.3.1b is intentionally an **audit-only** stage. It does not train another PATH model and does not add Range Bar, Footprint, OI or Books.

It decomposes the old target problem into three directly comparable residuals using the same causal nuisance feature family and chronology:

1. **Legacy proxy**: `log1p(P(release) * smearing-adjusted positive raw density)`;
2. **Formula-only correction**: `P(release) * Huber[log1p(density) | release]`;
3. **Primary mean-aligned correction**: `P(release) * L2-mean[log1p(density) | release]`.

The primary corrected residual is:

`log1p(actual density) - P(release) * E_L2[log1p(density) | release, X]`.

This separates:

- the nonlinear `log(E[Y])` versus `E[log(Y)]` mismatch;
- the Huber robust-location versus conditional-mean mismatch.

The stage reports exact distance-cell residual means, Validation/Holdout distance correlations, yearly stability, transform/objective gaps, release-hurdle drift and the full causal audit.

Hard rule: if corrected Validation/Holdout residual-distance correlation still exceeds the frozen **0.12** tolerance or fails to improve over raw distance dependence, remain blocked. Do not add new data families and do not train a new PATH ranker.

If target consistency passes but frozen nuisance calibration/discrimination still drifts, the next stage is nuisance-regime conditioning only. PATH retesting waits until that nuisance background is credible.

### R02.3.1b pre-run engineering validation

- dedicated R02.3.1b tests: **9 passed**;
- primary expectation identity is locked to `P(release) * E_L2[log1p(density)|release]`;
- TRAIN nuisance predictions remain expanding past-only OOS with 13h purge;
- Validation/Holdout remain full-2023-2024-TRAIN frozen;
- nuisance features remain distance + group-level calendar/activity only;
- Swing and zone-specific path features are excluded from nuisance estimation;
- no PATH ranker and no new data family are introduced in this stage.

No empirical R02.3.1b market result exists at delivery time. The next cumulative update must append the real R02.3.1b report before opening a nuisance-regime stage or re-testing PATH.

## 2026-08-08 — R02.3.1b real run: target correction did not solve the core problem

Decision: `BLOCKED_R02_3_1B_TARGET_CONSISTENCY_STILL_DISTANCE_CONTAMINATED`.

### What was corrected

R02.3.1b separated two possible implementation/statistical problems in the old Excess target:

1. nonlinear scale mismatch between `log1p(E[Y])` and `E[log1p(Y)]`;
2. Huber robust-location objective versus an L2 conditional mean on positive `log1p(density)`.

The primary corrected expectation was exactly:

`P(release > 0 | X) * E_L2[log1p(density) | release > 0, X]`.

All 16 causal/oracle-free target-consistency checks passed. No PATH ranker, Swing rescue or new data family was used.

### Distance contamination after the correction

Absolute distance correlations (raw / legacy residual / formula-only residual / L2 mean-aligned residual):

- DOWN Validation: **0.148 / 0.084 / 0.102 / 0.090**;
- DOWN Holdout: **0.161 / 0.134 / 0.149 / 0.140**;
- UP Validation: **0.135 / 0.150 / 0.160 / 0.149**;
- UP Holdout: **0.173 / 0.208 / 0.212 / 0.203**.

The frozen 0.12 gate therefore still fails. The formula correction is real, but it is not the root cause of the failed Excess target.

### Huber versus L2 was not the main issue

Mean absolute Huber-to-L2 positive-log objective gaps were only about **0.005-0.009** across periods/sides. Do not spend another stage tuning loss functions or LightGBM parameters.

### The dominant problem is nonstationary release background

Release-hurdle ROC AUC:

- TRAIN DOWN/UP: **0.691 / 0.689**;
- Validation DOWN/UP: **0.513 / 0.511**;
- Holdout DOWN/UP: **0.513 / 0.496**.

Actual release rates changed materially by year:

- 2023: **7.1% DOWN / 9.5% UP**;
- 2024: **27.9% / 29.1%**;
- 2025: **43.4% / 43.5%**;
- 2026: **44.9% / 42.5%**.

Full-TRAIN-frozen future nuisance models therefore ceased to discriminate release and materially underpredicted the later release background. This is a structural/nonstationarity problem, not a small target-transform bug.

### Research-management conclusion

Do **not** open R02.3.1c/1d to keep rescuing this nuisance target. The sequence R02.1 -> R02.2 -> R02.3 -> R02.3.1 -> R02.3.1b has already answered the decontamination question sufficiently.

Before spending more time on nuisance-regime conditioning, Range, Footprint, OI, Books or another ranker, first answer a more basic commercial question:

> If the true liquidity-release location and favorable reversal were known by an oracle, is the realized price path economically thick enough after 11bp and 22bp costs to justify any further identification research?

This opens R02.4 Economic Ceiling Audit.

## R02.4 implementation freeze — Latent Liquidity Economic Ceiling Audit

R02.4 is a **stop/go upper-bound audit**, not a strategy and not a predictive model.

### Why this stage exists

The project has accumulated many predictive studies that later failed once execution and costs were applied. R02.4 reverses that sequence: prove that money exists in the mechanism before allowing another model to try to identify it.

### Source universe

R02.4 reuses the completed R01.1 1-second future path labels and keeps exactly one first representative per `release_episode_id`. It does not rebuild raw trades, train a model or add a feature family.

It reports four frozen universes:

1. `ALL_RELEASE_EPISODES`;
2. `FAVORABLE_REVERSAL_ORACLE` — explicitly future-informed, upper bound only;
3. `FROZEN_R01_REVERSAL_CLUSTERS` — frozen clusters 10/4/5;
4. `CONTINUATION_CONTROL_CLUSTER_8`.

### Oracle economics

For 60/180/300/600-second horizons:

- entry level is the future-known true release reference price;
- adverse excursion is the realized same-direction extension;
- favorable excursion is the realized opposite excursion;
- oracle risk is future MAE + fixed 3bp buffer;
- perfect-exit net-MFE ceiling is evaluated at 6/8/11/22/33bp round-trip cost;
- separate fixed 1R/1.5R/2R realizations use the oracle stop but fixed targets, with horizon close when the target is not reached.

The MAE-based stop and favorable-reversal universe are intentionally non-causal. They may never become strategy logic.

### Frozen commercial stop/go gate

The hard gate uses the **perfect-exit net-MFE ceiling** on favorable-reversal oracle episodes at the frozen 300-second horizon. Fixed-R results are diagnostics, not the hard ceiling gate.

Validation and Holdout must each have at least 100 oracle favorable episodes and must pass:

- 11bp: mean net-MFE >= 10bp, PF >= 1.50, positive net-MFE rate >= 65%, top-10-removed mean > 0;
- 22bp: mean net-MFE > 0 and PF > 1.00.

If any required Validation/Holdout gate fails, the decision is:

`STOP_LATENT_LIQUIDITY_REVERSAL_ECONOMIC_CEILING_TOO_THIN`.

If all pass, the decision is only:

`CONTINUE_LATENT_LIQUIDITY_IDENTIFICATION_ECONOMIC_CEILING_EXISTS`.

Passing does not approve any current model, strategy or live trading. It only proves that further identification research has a sufficiently thick economic target.
