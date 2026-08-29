# ETH Trend Pullback V1 — Work Log

## 2026-08-17 / V1

### 背景

- Turtle baseline 已证明简单趋势策略在 ETH 上可以得到正收益，但正式 2023+ 的 CAGR 预计低于 MDD，且平均持仓过长，funding/资本效率不理想。
- 项目原有 `backtest/mf/trend_following/trend_pullback.py` 是单一 15m EMA20 reclaim + EMA50/200 baseline，历史已经表现较差，因此本版本明确不重复该逻辑。

### 本阶段设计

- 4H 只做方向；
- 1H 必须先出现真实回调并 reclaim；
- 15m 只在 reclaim 后等待重新加速；
- next-open execution；
- initial structure stop + later 1H structure trail；
- 12h no-progress 和 72h max-hold 控制资本占用；
- funding 纳入交易净收益；
- 2022 只 warmup，正式交易从 2023-01-01 开始；
- 参数邻域预先固定，只用于 robustness，不允许挑冠军后重命名 BASE。

### 工程修复/防错

- 新增 local-only Binance funding archive loader 到 `src/data_feed`，避免 backtest 直接读取研究 CSV。
- 高周期 context 使用 explicit `available_time`，不按 bar start 直接 ffill。
- 标记权益逐 15m 更新，MDD 不只看已平仓 capital。
- funding 边界按保守原则处理：同一时刻的 adverse funding 可以计入，favorable funding 不在模糊边界白拿。
- 交易执行保持 closed-bar -> next-open。

### 当前状态

- 代码完成；
- synthetic causality / funding / next-open / warmup tests 已通过；
- 由于交付 ZIP 不包含用户本地多年 trade-bar SQLite，当前环境无法生成真实 2023–2026 回测结论；
- 下一步是用户本地运行后，根据 `summary.json + cost_stress.csv + robustness.csv + trades.csv` 决定 PASS/REJECT。
