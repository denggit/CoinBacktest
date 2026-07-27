# Liquidity Hunt Momentum R01

## 目标

验证两条互相独立的微观结构事件链，而不是直接把原始策略描述当成已成立的 edge：

1. **M1 流动性猎杀反转**：扫前高/前低 → 主动成交放量 → 下一根 Range Bar 缩量收回 → OBI 翻转 → 同侧流动性重建。
2. **M2 流动性空洞动量**：连续两根 Range Bar 主动进攻 → OBI 持续同向 → 前方 25bps 深度相对自身历史和另一侧都显著稀薄。

R01 是事件研究和固定规则策略探针，不是参数工厂，也不能仅凭一次运行就声明可实盘。

## 数据接口

只通过 CoinBacktest 公共接口读取：

- `src.data_feed.okx_range_bar_loader.OKXRangeBarLoader`
- `src.data_feed.okx_range_footprint_loader.OKXRangeFootprintLoader`
- `src.data_feed.okx_liquidity_map_loader.OKXLiquidityMapLoader`

研究脚本不直接读取 SQLite 表，也不重新扫描原始 Books/Trades。

## OBI 定义说明

当前离线 liquidity feature schema 提供的是 5bps / 25bps 深度，而不是逐档保存的“精确前 5 档”。因此 R01 使用：

```text
OBI_5bps = (bid_depth_5bps - ask_depth_5bps) / (bid_depth_5bps + ask_depth_5bps)
```

并做 5 秒因果滚动平滑。它是原策略“前 5 档 OBI”的可部署代理，不应在报告中伪装成完全相同的指标。未来若公共 data-feed 增加 exact top-N depth，再做 R02 对照研究。

## 因果时序

- Range Bar 完成后才生成信号。
- Books 仅使用 `available_time <= signal_time` 的最新一行。
- 只允许在第一根 `start_ts > signal_time` 的 Range Bar `open` 入场；如果下一根 Range Bar 与信号共享同一毫秒时间戳，必须跳过。延迟压力从这个首个严格可用 open 再向后延迟。
- 动量衰减和时间退出只在完整 Range Bar 结束后决定，并在下一根 Range Bar 开盘执行。
- 同一 Range Bar 同时触及止损和止盈时，按止损处理。
- 10 分钟 Books 基线先 `shift(1)`，不包含当前秒。

## 防过拟合

- 阈值在运行前固定，写入 `15_predeclared_thresholds.csv`。
- 不做大参数网格；只比较 r0015 / r0020 / r0025 三个相邻 Range 尺寸。
- 每条逻辑逐层输出：基础流量事件、加 OBI、再加重建/空洞。
- 按时间顺序输出 60% train、20% validation、20% holdout，不随机打乱。
- 严格阶段必须同时通过 1.5x / 2x 手续费和 1 / 2 / 3 Range-Bar 延迟压力。
- Books 可用窗口较短时，不允许根据 holdout 结果反复修改阈值。

## 性能设计

- Range Bar 每个尺寸只加载一次。
- Footprint 按月读取并立即聚合成每个 `bar_id` 一行，避免把全部价格桶长期留在内存。
- Books 按天流式读取；同一天只构造一次 5 秒滚动特征，然后同时对齐所有 Range 尺寸。
- 每日 Range-Bar 切片使用 `numpy.searchsorted`，不做“每天 × 每个尺寸”的全表布尔扫描。
- Books 对齐使用有序时间戳二分查找，不做笛卡尔 join。

## 输出

默认目录：

```text
data/reports/research/liquidity/liquidity_hunt_momentum_r01
```

重点文件：

- `01_data_quality.csv`：Books/Footprint 覆盖率与因果违规数。
- `02_event_stage_summary.csv`：各事件层样本数和频率。
- `03_forward_path_summary.csv`：5/15/30/60 分钟固定前向路径。
- `04_split_summary.csv`：train/validation/holdout。
- `05_strategy_summary.csv`：固定出场探针。
- `06_cost_stress.csv`：1x/1.5x/2x 成本。
- `07_delay_stress.csv`：1/2/3 Range-Bar 延迟。
- `08_range_neighborhood.csv`：相邻 Range 尺寸。
- `09_yearly.csv`、`10_monthly.csv`：时间稳定性。
- `11_fixed_feature_uplift.csv`：预先声明的粗分箱诊断，不用于自动选参。
- `12_causal_audit.csv`：逐事件/逐交易因果审计。
- `13_event_sample.csv`、`14_trade_sample.csv`：复核样本。
- `16_research_brief.md`：自动摘要。

## 运行

Windows：

```text
python research\liquidity\01_liquidity_hunt_momentum_event_study.py --symbol ETH-USDT-SWAP --start-date 2025-10-01 --end-date "2026-06-30 23:59:59" --range-pcts 0.0015,0.002,0.0025 --books-depth 5000
```

Unix：

```text
python research/liquidity/01_liquidity_hunt_momentum_event_study.py --symbol ETH-USDT-SWAP --start-date 2025-10-01 --end-date "2026-06-30 23:59:59" --range-pcts 0.0015,0.002,0.0025 --books-depth 5000
```

快速自检：

```text
python research/liquidity/01_liquidity_hunt_momentum_event_study.py --self-test --no-progress
```

## 晋级门槛

R01 只负责判断机制是否存在。进入下一轮前至少要求：

- 严格阶段在 validation 和 holdout 都为正，而不是只有 train 为正。
- 2x 手续费仍为正，3 Range-Bar 延迟不发生结构性崩溃。
- r0015 / r0020 / r0025 至少两个相邻尺寸方向一致。
- 多空、月份和市场阶段不依赖单一小区间。
- 样本量满足对应频率层级；低样本高 PF 不作为 edge。
- `12_causal_audit.csv` 无未来 Books、同 bar 入场等失败。
