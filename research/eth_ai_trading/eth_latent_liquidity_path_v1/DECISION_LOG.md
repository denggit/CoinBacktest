# Decision Log

## D01 — 不做突破追价模型

突破或波动扩张只能作为路径证据。最终入场必须靠近局部真实失效结构，结构过远则减仓或跳过。

## D02 — Swing保留，但不作为唯一候选入口

Swing High/Low、多周期Pivot、等高低和波段嵌套全部进入特征。候选事件同时允许由成交释放、价格冲击、边界越界和区间扩张产生。

## D03 — 位置模型与释放模型物理拆分

路径聚类严格截止到事件前1秒。扫损当秒的爆量、Delta、价格冲击和越界分数只用于释放阶段描述，不能进入潜在池位置聚类。

## D04 — 先学习连续路径，再看语义类别

R01输出完整的扫穿深度、到极值时间、反转幅度、接受比例和多时间点路径。语义分类只是报告层，不得成为唯一标签。

## D05 — 第一版固定结果类型

- `SHALLOW_IMMEDIATE_REVERSAL`；
- `DEEP_IMMEDIATE_REVERSAL`；
- `EXTEND_STABILIZE_REVERSAL`；
- `ACCEPT_CONTINUATION`；
- `MIXED_OR_UNRESOLVED`。

前三类只代表值得后续研究的反转路径，不代表已经可交易。

## D06 — R01不用Range/Footprint硬筛

R01先用历史最完整的1秒与1分钟Trade Bar建立基础图谱。Range、Footprint、OI与Books在后续模块做固定顺序增量，避免短覆盖数据决定全历史样本。

## 2026-08-04 — R01.1 liquidity-first correction

Decision: supersede the Swing-centered R01 interpretation. Remove all sub-15m Swing features. Keep every causally confirmed 15m/30m/1H/4H/1D Swing High/Low active until first true sweep, with no arbitrary age expiry and no nearest-only reduction. Treat this inventory only as supplementary structural context. Make liquidity accumulation/release paths, episodes, turnover per range, pressure without progress, overlap/residency and impact efficiency the primary discovery space.

## 2026-08-05 — R01.1 full-history memory and resumability hotfix

Decision: keep every research definition frozen and replace only the execution path that failed after 2,431,174 full-history events. The old finalization sorted and deep-copied a 384-column frame, requesting a 6.96 GiB contiguous float64 allocation; later clustering and report joins would have repeated similar peaks. R01.1 now compacts each small chunk to float32/downcast integers, assigns global release Episodes from narrow key arrays without consolidating the wide frame, fits frozen clusters on a deterministic stratified pre-cutoff sample capped at 250,000 rows, assigns all rows in 50,000-row batches, constructs narrow report joins, and streams large gzip tables. Two-day chunk results are cached with a configuration signature so a later finalization failure can resume without recomputing the 1-second path atlas.

## R01.2 decision — 2026-08-05

Proceed with a stable-path explanation and causal execution-thickness audit.

Frozen target discovery clusters are 10, 4, 5 and continuation-control 8. This selection is explicitly post-R01.1 and cannot serve as sealed validation. Fixed confirmation rules are 15-second stabilization, 5/10/15bp reclaim and second-push failure. Entry is next-second-or-later open, structural stop uses only the extreme known at confirmation, and cost stress is 11/22/33bp with 1/3/5-second delay.

Swing remains supplementary 15m+ inventory only. The research center is liquidity accumulation, release, absorption and price acceptance.

## R01.3 decision — 2026-08-06

Proceed with one final, bounded supervised stage: absorption completion and remaining executable space.

R01.3 does not search more reclaim/stop parameters. It creates fixed causal snapshots at 15–300 seconds, trains only on 2023–2024, freezes a q90 score threshold on 2025 Q1–Q3 and evaluates 2025 Q4–2026 H1 without threshold changes. The model uses post-release flow, impact efficiency, pressure-without-progress, extreme updates and reclaim/giveback; it uses no direct Swing feature or Swing gate.

Commercial promotion requires holdout predictive uplift over a mechanical baseline plus positive, non-concentrated execution under 11bp cost and survival at 22bp cost. Failure ends this executable model family rather than starting another parameter-patch sequence.

## 2026-08-07 — Close R01.3 execution branch; open R02 spatial branch

- R01.3 full run: post-release supervised confirmation retained ranking information but failed the frozen commercial gate with PF well below 1 in validation and holdout.
- Decision: stop the post-release market-entry branch and prohibit threshold/confirmation/stop tuning on the same evidence.
- New independent research question: forecast latent liquidity **before** release in price space, including touch probability, release-on-touch, reversal-vs-continuation and sweep depth.
- R02 is not a continuation of the failed entry rule.  It reuses only the validated path/label infrastructure and moves execution research earlier, closer to the latent pool.
- Swing is limited to 15m+ all-unswept inventory and is explicitly treated as an incremental supplemental family, never the candidate universe.

## 2026-08-07 — R02 implementation freeze

- R02 searches continuous time-price space before release; it does not enumerate a small set of stop-pool hypotheses.
- Primary model family is liquidity/path. 15m+ all-unswept Swing inventory is retained only for an explicit incremental ablation against the no-Swing model.
- The last 12 hours of the requested sample cannot become negative examples because their full future label is unavailable; they are excluded from decision rows.
- Missing R01.1 Swing lifecycle is an explicit setup failure rather than a silent downgrade.
- No grid search over distance, threshold, Swing age, stop or execution parameters is allowed in R02. Any promotion only opens a separate R02.1 passive-limit execution study.

## 2026-08-07 — R02.1 result: retire absolute cumulative strength target

- R02.1 full run passed causal gates but the absolute 12h cumulative strength label was exposed as structurally contaminated by first-touch timing and repeated visits.
- TRAIN high-strength prevalence was ~20% by frozen q80 definition, but Validation/Holdout rose to ~41–43%, repeating the fixed-threshold drift pattern seen in the archived Q70 model.
- Distance-only baseline beat the path model on holdout absolute high-strength classification.
- Nevertheless, Top-1 path-selected zones retained ~1.61x DOWN / ~1.40x UP realized-density lift, favorable AUC remained ~0.65–0.66, and sweep-depth rank information remained ~0.30–0.36.
- Swing added no material holdout value and remains supplemental only.
- Decision: **do not add more features to the contaminated absolute target. Change the modeling problem first.**

## 2026-08-07 — R02.2 first-touch relative ranking freeze

- Replace absolute strength classification with within-snapshot price-zone ranking.
- Use only complete 25-zone lattices.
- Resolve each zone's exact first touch using completed 1m path plus exact 1s crossing.
- Give near and far zones identical fixed post-touch windows: 30/60/180/300 seconds.
- Primary target is first-touch release-density sum over 180 seconds; other windows and 1s flow ratios are diagnostics.
- Primary ranker excludes Swing and Touch probability. All active 15m+ unswept Swing remains only as a full-model ablation.
- Distance/nearest-zone ordering is a mechanical baseline.
- No absolute q80/q90 pool threshold is allowed in the primary ranking question.
- Passing can only promote to R02.3 causal limit-placement/sweep-depth research, never live capital.
- Every future cumulative patch must preserve actual stage results, failed branches, engineering incidents and next-step rationale in `CUMULATIVE_STAGE_RESULTS.md`.

## 2026-08-07 — R02.2 post-run correction: raw density is not latent pool strength

- R02.2 Holdout raw-density ranking failed: PATH_NO_SWING mean within-group Spearman was ~0.041 DOWN / ~0.074 UP, below nearest-distance baselines ~0.174 / ~0.156.
- The fixed first-touch window removed R02.1's exposure-duration bias but not the mechanical fact that near-price zones naturally contain more ordinary microstructure activity.
- Path-selected farther zones nevertheless had better favorable-vs-continuation behavior than the nearest 10bp baseline, so path information is retained while the raw-density target is retired.
- The 8,282 same-timestamp causal flags were an audit semantics error: 1s Trade Bar timestamps are bar starts; availability is one second later. The audit is corrected to `first_touch_available_time = first_touch_time + 1s`.
- 22 exact-touch / old-R02-touch disagreements are not ignored. They are quarantined from R02.3 train/evaluation and reported explicitly.
- Swing remains ablation-only; no stable cross-period uplift exists.

## 2026-08-07 — R02.3 modeling freeze

- Primary target becomes TRAIN-only distance-normalized **Excess Liquidity**, using robust median/IQR of `log1p(first-touch 180s release density)` per side x distance.
- Validation/Holdout statistics are forbidden from target normalization.
- Raw `zone_distance_bp` is removed from the primary model so the model cannot trivially relearn the nuisance distance curve. Near/far distance remain explicit baselines only.
- Reversal Quality is trained separately as `log1p(favorable density) - log1p(continuation density)` conditional on a release.
- Sweep Depth and Reversal Room remain separate No-Swing regressions.
- No composite execution score, limit-order backtest, Range/Footprint/OI increment or threshold grid is allowed before the corrected target is evaluated.
- Every later cumulative patch must append the actual R02.3 market result before opening another stage.

## 2026-08-07 — R02.3 post-run: robust median/IQR excess normalizer retired

- All R02.3 causal gates passed, but the zero-inflated 180s Release Density target made all 50 TRAIN side x distance medians equal zero and 34/50 IQRs equal zero.
- Validation/Holdout distance-vs-Excess correlations returned to about 0.15–0.18 in absolute value, proving the target was not truly distance-normalized.
- Do not add Range/Footprint/OI to this failed target.
- No-Swing remains primary; Swing again showed no stable cross-period increment.
- Reversal Quality and Sweep Geometry remain worth preserving, but Reversal must also be residualized because far distance alone predicts reversal tendency.

## 2026-08-07 — R02.3.1 hurdle nuisance residualization freeze

- Replace zero-median lookup normalization with a two-part hurdle expectation: release probability plus positive-release magnitude.
- Nuisance features are restricted to raw distance, calendar/session and broad group-level activity/volatility that are identical across zones within a decision-time/side group.
- TRAIN nuisance predictions must be expanding past-only OOS with a purge; full TRAIN models may score only Validation/Holdout.
- Primary residual rankers exclude raw distance, nuisance activity and Swing. Zone-specific liquidity-path structure remains the tested edge family.
- Reversal Quality is residualized against the same nuisance family.
- Residualization quality is a hard gate before any ranking result can be promoted or any new data family can be added.

## 2026-08-08 — R02.3.1 real run blocked; open R02.3.1b target-consistency audit

- R02.3.1 completed with `BLOCKED_R02_3_1_RESIDUALIZATION_NOT_REMOVED` despite all 17 causal checks passing.
- Excess distance removal improved DOWN Validation but remained above tolerance in DOWN Holdout and worsened on UP Validation/Holdout.
- Reversal residual distance removal worked, but most previously observed raw Reversal Quality ranking disappeared after decontamination; treat the old reversal edge as substantially mechanical.
- Sweep Depth / Reversal Room remain retained independent geometry tasks. Swing remains supplemental only.
- Frozen 2023-2024 release nuisance showed major 2025-2026 drift (future AUC near 0.50 and release-rate underprediction).
- Code review identified a target-scale mismatch: old Excess used `log1p(E[Y])` while its positive nuisance head modeled `log1p(Y)`. The positive-log Huber head also did not estimate the conditional mean required by `E[log1p(Y)|X]`.
- Open one bounded audit stage, R02.3.1b. Compare legacy, formula-only and L2-mean-aligned hurdle expectations with identical causal source/features/folds.
- R02.3.1b trains no PATH ranker and adds no Range/Footprint/OI/Books. If target consistency still fails the frozen 0.12 distance gate, remain blocked. If it passes but nuisance drift remains, diagnose nuisance regime before PATH retest.

## 2026-08-08 — R02.3.1b blocked; stop target-rescue loop and open economic ceiling

- R02.3.1b returned `BLOCKED_R02_3_1B_TARGET_CONSISTENCY_STILL_DISTANCE_CONTAMINATED`.
- Correcting `log1p(E[Y])` versus `E[log1p(Y)]` did not solve the core problem; DOWN Holdout and both UP future periods still failed the 0.12 distance gate.
- Huber versus L2 was immaterial relative to the observed drift, so do not spend another stage tuning nuisance loss/objective.
- The dominant evidence is nonstationarity: release-hurdle future AUC is near 0.50 and actual release prevalence roughly doubled versus the 2023-2024 training background.
- Research-management correction: before any nuisance-regime or richer-data work, prove that the favorable latent-liquidity reversal mechanism has enough post-cost economic room in an oracle upper bound.
- Open R02.4. Passing only authorizes further identification research; failing kills the reversal branch.
