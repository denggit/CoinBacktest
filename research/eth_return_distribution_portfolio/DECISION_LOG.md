# Decision Log

## D01 — No event gate
任何 RDP forecast 都不得通过 q70/q90 threshold 变成“只有达到阈值才有一笔交易”。

## D02 — Continuous position is the end product
研究最终输出 target exposure time series；交易只是 target exposure 变化后的执行结果。

## D03 — Multi-horizon sleeves remain separate
30m/2h/6h/24h/72h 的预测先分开评价、分开记账，Portfolio 才负责组合。禁止过早平均成一个总分掩盖某个 horizon 失效。

## D04 — 2026 is not a new sealed holdout
历史项目已读取 2026 数据，因此 RDP 的 WF_2026 只能叫 chronological OOS，不能重新包装成 untouched holdout。
