# Open Items and Roadmap

1. Run V1 locally on the full prebuilt ETH datasets.
2. Review `gpt_review_pack.zip` and reject/retain strategies based on after-cost results.
3. If robust survivors exist, freeze exact specs and portfolio sleeves before opening 2026.
4. Run 2026-01-01 through 2026-08-15 once as final sealed validation.
5. Only strategies surviving that final gate are converted into AetherEdge strategy plugins.
6. If V1 has no robust survivors, do not parameter-mine it. Build V2 from a new batch of externally published complete strategies (e.g. VWAP/ORB/session/volatility families) and repeat the same protocol.


## Active: R02 Continuous Risk-Managed Portfolio

1. Run the frozen R02 continuous portfolio locally on 2023-2025.
2. Review exposure continuity, drawdown, turnover, costs, yearly/monthly stability and top-day dependence.
3. If a spec survives, freeze it and only then open 2026 once for final sealed validation.
4. If none survive, change portfolio/risk construction or add a new externally specified simple family; do not parameter-mine the existing signals.

## Active: R03 Source-Locked Trend Replication

1. Run all four frozen source baselines on local 2023-2025 ETH data.
2. Review base, 2x/3x costs, +1m/+2m delay, yearly/monthly stability, inactivity, MDD, and top-day concentration.
3. Do not modify any R03 parameter after seeing the result.
4. If at least two source baselines survive, freeze them unchanged and build R04 as an independent-sleeve portfolio construction test.
5. If fewer than two survive, source a new batch of complete public systems; do not rescue R03 through parameter mining.
6. Open 2026 only after the final portfolio construction is frozen.

## R04 next gate

Run `04_turtle_path_atlas.py`. Review whether winner/loser path separation repeats in 2025, especially: speed to Unit 2/3/4, loss concentration by maximum Unit reached, MFE giveback before 20D exit, and early MAE. Only then define R05 position-management overlays; no entry-rule changes.

R04.1: rerun the same `04_turtle_path_atlas.py` command after applying the cumulative hotfix; the prior run produced no valid atlas output because it stopped before path reconstruction.
