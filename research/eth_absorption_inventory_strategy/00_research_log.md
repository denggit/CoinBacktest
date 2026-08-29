# Research log

## R01 — 2026-08-17

### Why this revision exists

The prior absorption atlas overproduced small-scale events and was not a strategy. The prior continuous-inventory R01 also made two mistakes: it treated absorption mostly as a short-horizon reversal signal, and its leverage-cap implementation could turn a LONG vote into a SELL or a SHORT vote into a BUY after equity deterioration.

### Frozen strategy change

R01 here implements current market-control voting:

1. failed aggressive pressure / impact decay -> opposite vote;
2. repeated defense + causal spring/upthrust -> defending-side vote;
3. effective aggressive pressure that truly moves price -> same-side continuation vote;
4. only 15m / 1H / 4H can change inventory;
5. 5m can veto a reversal when clearly effective opposite pressure is present;
6. each fresh event creates one 1%-margin vote; no conventional exit model;
7. higher-timeframe bars are available only after close and execute next 1m open;
8. leverage cap may block/clip a same-side vote but can never reverse the vote direction.

### What is deliberately not done

- no parameter optimization against R01 outcomes;
- no selecting one lucky year;
- no TP/SL tuning;
- no event-atlas expansion;
- no 5s independent trading signals;
- no funding-rate modeling yet (must be added before any live-ready claim).

### Acceptance gate

The mechanism deserves R02 only if the actual strategy equity curve is economically credible, not liquidated, survives 2x/3x execution costs, and is not dependent on one isolated year.
