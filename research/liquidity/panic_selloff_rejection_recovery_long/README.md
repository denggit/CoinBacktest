# Panic Selloff -> Rejection -> Recovery Long

该研究属于 `research/liquidity`，因为核心机制是流动性冲击、持续卖压、卖压衰竭与价格恢复，不是单根 K 线形态。

## 共享检测器

- 研究专属实现：`research/liquidity/panic_selloff_rejection_recovery_long/common/panic_episode.py`
- 兼容旧研究路径：`research/liquidity/panic_selloff_rejection_recovery_long/panic_episode.py`
- 通用订单流模块：`src/research_common/trade_bar_orderflow.py`
- analyze_tool 插件：`analyze_tool/plugins/panic_selloff_recovery.py`

## 01 基础研究

文件：

`01_environment_and_cluster_scale_in_research.py`

主要研究：

1. 橙色开始观察、黄色卖压衰减、绿色恢复确认，分别在下一根 open 入场后的表现。
2. 预定义、因果的趋势/位置/波动/严重度/时序/连续信号环境过滤。
3. 2023-2024 训练筛选，2025-2026H1 留出验证。
4. 近距离连续绿灯的有限加仓：最多 2/3 次，总仓位不超过 100%，统一结构止损与统一目标退出。
5. 不使用普通 time exit；仍持仓到数据结束的交易单独标记为 `end_of_data`。

运行：

```bash
python research/liquidity/panic_selloff_rejection_recovery_long/01_environment_and_cluster_scale_in_research.py --symbol ETH-USDT-SWAP --timeframe 1m --data-source trade_bar --start-date 2023-01-01 --end-date 2026-06-30 --warmup-start-date 2022-01-01
```

默认报告目录：

`data/reports/research/liquidity/panic_selloff_rejection_recovery_long/01_environment_and_cluster_scale_in`

完成后会自动生成 `gpt_review_pack.zip`。

## 02 Trade-Bar Order-Flow / Absorption

文件：

`02_trade_bar_orderflow_absorption_research.py`

本轮不允许退化成普通 OHLCV。脚本会检查并使用：

- `buy_notional / sell_notional / delta_notional`
- `buy_trades_count / sell_trades_count / trades_count`
- `taker_buy_ratio`
- `large_buy_notional / large_sell_notional / large_delta_notional`
- `large_trades_count / max_trade_notional`
- normalized CVD、订单流反转、成交活跃度、价格冲击与吸收代理

研究内容：

1. 绿色恢复信号的恐慌流、真实低点吸收、普通/大单流恢复过滤。
2. 橙色开始观察节点是否能仅凭当时可见订单流识别“已经接近底部”的样本。
3. 近距离重复绿灯只有在普通 delta 或大单 delta 继续改善时才作为加仓候选。
4. 固定规则与训练段四分位边界分别在 2025-2026H1 留出段验证。
5. 仍保持下一根 open 入场、结构止损、总仓位不超过 100%、无普通 time exit。

运行：

```bash
python research/liquidity/panic_selloff_rejection_recovery_long/02_trade_bar_orderflow_absorption_research.py --symbol ETH-USDT-SWAP --timeframe 1m --data-source trade_bar --start-date 2023-01-01 --end-date 2026-06-30 --warmup-start-date 2022-01-01
```

默认报告目录：

`data/reports/research/liquidity/panic_selloff_rejection_recovery_long/02_trade_bar_orderflow_absorption`

如果 trade bar 的核心订单流字段缺失、未回填或全部为常数，脚本会直接报错，不会悄悄改用 OHLCV。

