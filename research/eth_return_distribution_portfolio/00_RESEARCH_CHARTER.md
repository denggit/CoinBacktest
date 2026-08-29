# ETH Return Distribution Portfolio V1 — Research Charter

## 唯一目标

构建一个长期可迭代的 ETH 连续仓位系统：**持续预测多个未来周期的收益分布和路径风险，并直接决定 target exposure**，而不是先定义 Setup/Entry/TP/SL，再把市场压缩成少量交易事件。

## 冻结原则

1. 决策轴：每 5 分钟一个 observation；每个 observation 都保留，不用事件筛选删掉大部分样本。
2. 第一批 horizon：30m / 2h / 6h / 24h / 72h。
3. 每个 horizon 至少预测：return q10/q25/q50/q75/q90；后续加入 MFE/MAE/volatility 的条件分布。
4. 连续预测不得被切成 q70/q90 才“允许交易”。旧 Q70 分支已经证明阈值化以后会发生 calibration / score drift。
5. Portfolio 的研究对象是 **target-position time series + marked account equity**，不是 trade count。
6. 未来允许不同 horizon sleeve 同时多空；虚拟 sleeve 分开记账，执行层再处理净敞口、总敞口、冲突和风险。
7. 2022 只 warmup；正式研究从 2023-01-01 开始。
8. 手续费完整开平默认 0.11%；连续调仓时只对实际 `delta_notional` 收交易成本，禁止每 5 分钟虚构一次完整 round trip。
9. Funding 在进入账户回测前必须逐结算点计入；不能把长期仓位的 funding 当 0。
10. 禁止为了某一年、某个 horizon 或某笔亏损临时加过滤器和参数网格。

## 与旧 ETH AI Trading 的关系

旧 R03.3.2 已经证明“未来 6h opportunity intensity 可稳定排序”，这是可复用的正证据；但后续研究重新收缩到 q70 事件开仓池，2026 sealed holdout 因 score drift 失败。

本主线继承：
- continuous forecast；
- multi-horizon causal features；
- rolling / walk-forward OOS；
- live artifact versioning。

本主线明确放弃：
- q70/q90 event gate；
- 先定义一笔交易再预测它成功与否；
- 为了提高 PF 不断减少样本；
- “模型分数达到阈值 = 开一笔仓，退出后才重新决策”的状态机。

## 分阶段路线

### RDP-01 Price + Trade Flow Distribution Baseline
只使用全历史覆盖最完整的 1m rich Trade Bar，验证有方向的未来收益分布是否存在稳定 OOS 信息。

### RDP-02 Information Source Increment
逐模块加入 Funding / OI / Basis / Liquidation / Books / Range Footprint / Cross-exchange。每类信息必须报告增量，不因“特征更多”默认保留。

### RDP-03 Distribution Calibration
校准 q10~q90、direction probability、MFE/MAE / tail risk；建立 drift diagnostics。

### RDP-04 Continuous Target Exposure
每个 horizon 独立生成 signed exposure；再用固定的风险预算组合。只对 target position 的变化产生手续费和滑点。

### RDP-05 Portfolio Account
虚拟 short/mid/long sleeve 可同时持有相反方向；账户层统一处理 gross/net exposure、funding、margin、MDD、drawdown governor。

### RDP-06 Shadow / Forward Validation
历史 walk-forward 通过后进入长期 shadow；必须有新的未来数据才能升级到 live candidate。
