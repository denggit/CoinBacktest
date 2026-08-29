# ETH Absorption / Market-Control Inventory Strategy

This research line tests a position-blind inventory strategy rather than a conventional trade-entry/exit system.

R01 mechanics:

- signal axis: closed 1m trade bars;
- higher contexts: 5m / 15m / 1H / 4H with causal available-time alignment;
- inventory votes: fresh 15m / 1H / 4H control events only;
- 5m: confirmation/veto only;
- failed pressure / repeated defense / spring -> reversal vote;
- effective pressure -> continuation vote;
- one vote = 1% equity margin at 10x by default;
- cross net inventory;
- no conventional TP/SL/time exit;
- opposite future votes mechanically reduce/reverse inventory;
- default fees: 0.055% per executed fill; 1x/2x/3x stress.

Run (Windows or Unix, from repo root):

`python research/eth_absorption_inventory_strategy/01_absorption_state_inventory_strategy.py`
