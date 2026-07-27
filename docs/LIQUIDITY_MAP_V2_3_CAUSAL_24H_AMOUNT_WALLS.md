# Liquidity Map V2.3 — Causal 24h Amount Scale + Wall Audit

## Goal

Make the heatmap colour, wall detector and future live strategy read the same
causal quantity:

```text
depth_ratio(t, price) = current $1-bin depth / maximum $1-bin depth seen in the completed past 24h
```

Bid and ask share one scale. This avoids making a uniformly shallow side look
100% dark merely because it is the deepest level on that side.

## Colour semantics

Default front-end range:

```text
hide below: 0%
saturate at: 50%
```

Therefore a raw `depth_ratio >= 0.50` remains the darkest colour. Moving the
saturation slider upward does not uniformly fade every cell; it changes the
amount ratio required to reach the darkest colour.

The rolling reference includes the current completed snapshot and never uses a
later snapshot. Extending a query into the future cannot recolour earlier rows.
The first 24h of a dataset is a causal bootstrap because older source data may
not be available.

## Wall semantics

### POINT wall

A narrow one/two-bin level with at least one `50%+` amount cell. It is a small
support/resistance level, not automatically a main entry wall.

### MAIN wall

At the same completed snapshot:

- body bins: `30%+`;
- connecting/support bins: `15%+`;
- default minimum body bins: 3;
- default minimum total bins: 4;
- a main wall may have no `50%+` core only when at least four body bins exist;
- small price gaps are allowed, but different-time price-bin unions are never
  used to manufacture a wide wall.

Every timeline slice carries the price bounds that were known at that exact
snapshot. Later expansion cannot rewrite earlier bounds.

## Lifecycle

```text
FORMING -> STABLE -> PERSISTENT -> MAJOR
```

Defaults:

- STABLE after 30 seconds and 70% observation coverage;
- PERSISTENT after 30 minutes;
- MAJOR after 120 minutes.

A confirmed wall that disappears before touch after approaching within the
configured distance is tagged as `SUSPECTED_PRE_TOUCH_WITHDRAWAL`. This is only
a preliminary ghost-wall flag; Trade-based cancellation/consumption and moving
wall detection are not implemented in this patch.

## Safety boundary

This remains an Analyze Tool audit version. It must not be used for strategy
backtest or live orders until manual review confirms that POINT and MAIN boxes
match the visible amount distribution across a representative set of dates.

No liquidity-map NPZ rebuild is required.
