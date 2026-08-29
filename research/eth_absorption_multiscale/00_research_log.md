# ETH Multi-scale Absorption Research Log

## Goal

Find whether ETH has a real, causal and economically usable process behind the discretionary observation:

- heavy selling but price repeatedly cannot progress lower;
- heavy buying but price repeatedly cannot progress higher;
- repeated tests of the same defended zone;
- price may briefly break the defended zone, then reclaim quickly (spring / failed breakdown; symmetric upthrust for shorts);
- the defended zone may persist from seconds/minutes to hours/days.

The end goal is a deployable ETH perpetual strategy. This research stage does **not** optimize entries/exits or portfolio sizing.

## Why this line exists

The prior continuous-inventory R01 reduced absorption too aggressively to a few rolling 1m values and then accumulated those votes into persistent inventory. That experiment mixed two questions:

1. was absorption recognized correctly?
2. was persistent inventory a good account mechanism?

This line isolates question (1) first.

## R01 — Multi-scale absorption / floor-defense / spring atlas

Status: code ready for local full-data run.

### What R01 studies

Six causal scales:

- 5s: 30s / 1m / 3m pressure process, 30m defended zone.
- 1m: 3m / 5m / 15m pressure process, 6h defended zone.
- 5m: 15m / 30m / 1h pressure process, 12h defended zone.
- 15m: 45m / 1.5h / 4h pressure process, 1d defended zone.
- 1H: 3h / 6h / 12h pressure process, 3d defended zone.
- 4H: 12h / 1d / 2d pressure process, 14d defended zone.

### Frozen morphology definitions

R01 does not search thresholds from outcomes.

- `strong_pressure_control`: abnormal taker pressure with directional persistence.
- `pressure_stall`: strong pressure but weak normalized price progress.
- `pressure_rejection`: strong pressure but close moves against the pressure direction.
- `impact_decay`: similar/stronger same-side pressure across adjacent windows, but materially less price response.
- `floor_retest` / `ceiling_retest`: a distinct revisit to a prior-only rolling extreme zone.
- `spring_same_bar` / `upthrust_same_bar`: break of a previously-known extreme and reclaim before the bar closes.
- `spring_reclaim` / `upthrust_reclaim`: break occurs during a short closed-bar sequence and the extreme is reclaimed before the signal is emitted.

### “多次跌不下去” representation

R01 deliberately separates three concepts instead of counting every bar near the low as another test:

- `prior_defense_count`: number of **distinct** enter-zone episodes before the current event.
- `hold_ratio`: how often price stayed on the defended side of the zone.
- `zone_stability_bucket`: whether the rolling floor/ceiling itself stayed in roughly the same ATR-normalized area or drifted.

This is intended to distinguish:

- repeated attacks on one level;
- simply sitting near a low for a long time;
- a moving/down-trending floor that should not be treated as stable support.

### Timing / causality

- All features use only closed bars.
- Higher scales are aggregated from local 1m trade bars and become available only after the aggregate bar closes.
- Signal time is `bar_start + timeframe`.
- Outcome entry is the next bar open.
- Forward returns/MFE/MAE never participate in event construction.

### Economic diagnostic

Fixed-horizon results also deduct the project default full round-trip cost of 0.11%. This is not a final execution model; it is used to reject effects too thin to trade.

### R01 outputs

- pattern × scale × horizon summary;
- yearly stability;
- strong-pressure response-state comparison;
- spring/retest performance by prior defense count;
- performance by zone stability;
- performance by hold-above / hold-below ratio;
- event examples;
- causal timing audit;
- GPT review pack.

## Frozen next-step rule

Do **not** build another continuous-inventory backtest from R01 automatically.

Only continue if at least one morphology shows:

1. a meaningful difference versus its broad pressure/control population;
2. consistent sign across multiple adjacent scales/horizons;
3. non-trivial sample count across years;
4. enough gross thickness to survive realistic cost/latency;
5. sensible monotonicity, e.g. 3+ prior defenses or stable-zone spring is stronger than 0-defense spring rather than one arbitrary parameter point winning.

If those conditions fail, reject the morphology instead of parameter-tuning it.
